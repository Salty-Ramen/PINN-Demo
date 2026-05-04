#=-------------------------------------------------------------------------------
Training.jl

Orchestrates the two-stage PINN training pipeline:

  Stage 1 — Supervised: fit the state network to training data
  Stage 2 — ODE-regularised: add physics residual loss, train
            g-network and recover ODE parameters

This file handles parameter assembly, context construction,
optimizer wiring, and the between-stages frozen-scale computation.
Loss math lives in Losses.jl; predictor construction in Predictors.jl.
-------------------------------------------------------------------------------=#

#=-------------------------------------------------------------------------------

Utilities

-------------------------------------------------------------------------------=#

using Optimization, OptimizationOptimisers, OptimizationOptimJL
import Optim


"""
A hyperparam container struct containing ... hyperparams.
"""
struct HyperParams
    ϵ_ic        :: Float32
    ϵ_ode       :: Float32
    ϵ_Data      :: Float32
    ϵ_L1_state  :: Float32
    ϵ_L1_g      :: Float32
end

"""
A results container struct containing ... results.
"""
struct TrainResult{PS,CTX,M1,M2}
    hyper    :: HyperParams
    params   :: PS
    ctx2     :: CTX
    state_mlp:: M1
    g_mlp    :: M2
    metrics  :: NamedTuple
end

"""
Stage 1 context: supervised data fitting.

The predictor closures satisfy:
  - `predict_state(ps, t_grid)` → (n_states × B) in transformed coords
  - `predict_deriv(ps, t_grid)` → (n_states × B) time derivatives
"""
struct PINNCtxStage1{PS,PD,T,Y,YS,V,TR}
    predict_state  :: PS
    predict_deriv  :: PD
    t_train        :: T
    Y_train        :: Y
    Y_train_std    :: YS
    y0_obs         :: V
    transform      :: TR
    skip_ic_loss   :: Bool
    ϵ_ic           :: Float32
    ϵ_Data         :: Float32
    ϵ_L1_state     :: Float32
    t_span         :: Vector{Float32}
end

"""
Stage 2 context: ODE-regularised training.

Adds the g-predictor, dense collocation grid, ODE-specific epsilon, and
parameter values and bounds on top of Stage 1 fields. The ODE parameters
contained in this struct are fixed, i.e., they won't be optimized.
"""
struct PINNCtxStage2{PS,PD,PG,T,Y,YS,GS,V,TR,TD}
    predict_state  :: PS
    predict_deriv  :: PD
    predict_g      :: PG
    t_train        :: T
    Y_train        :: Y
    Y_train_std    :: YS
    G_stage1_std   :: GS
    y0_obs         :: V
    transform      :: TR
    skip_ic_loss   :: Bool
    ϵ_ic           :: Float32
    ϵ_Data         :: Float32
    ϵ_ode          :: Float32
    ϵ_L1_state     :: Float32
    ϵ_L1_g         :: Float32
    t_dense        :: TD
    t_span         :: Vector{Float32}
    ODE_par_bounds :: NamedTuple
    ODE_par_fixed  :: NamedTuple
end

"""
   compute_frozen_ode_scale(ps, ctx; min_scale=1f-6) -> Matrix{Float32}

Since it was a bit awkward to decide on how the g NNs would be normalized, I
think one way is by using the std of the derivative of post stage-1 state NN,
since that would approximately be the dimensions where g would be.
"""
function compute_frozen_ode_scale(ps, ctx; min_scale::Float32 = 1f-06)
    dYdt = ctx.predict_deriv(ps, ctx.t_dense)
    scale = std(dYdt; dims = 2)
    scale = max.(scale, min_scale)
    return Float32.(Array(scale))
end

"""
The default callback that just prints an iter count.
"""
function callback_function_default(state, l)
    if state.iter % 2500 == 0
        println("iteration = ", state.iter)
    end
    return false
end


using Zygote
# using Enzyme

"""

A wrapper/builder that makes an Optimization.jl style loss function
"""
function make_optfun(loss)
    Optimization.OptimizationFunction(
        (θ, p) -> loss(θ, p),
        Optimization.AutoZygote()
    )
end

"""
    extract_learned_hyper(ps) -> HyperParams

Recover the final ε values from the trained parameter vector.
For adaptive mode, exponentiates the log-space values.
For fixed mode, returns NaN placeholders.
"""
function extract_learned_hyper(ps)
    if hasproperty(ps, :hyper)
        h = ps.hyper
        return HyperParams(
            exp(h.log_ϵ_ic),
            exp(h.log_ϵ_ode),
            exp(h.log_ϵ_Data),
            exp(h.log_ϵ_L1_state),
            exp(h.log_ϵ_L1_g),
        )
    else
        nan = NaN32
        return HyperParams(nan, nan, nan, nan, nan)
    end
end


function Normalize_Data_for_ODE_dispatch(data,
                                         ::Val{:AllFixedODEParams})

    return haskey(data, :ODE_par_init) ? data : merge(data, (ODE_par_init = NamedTuple()))
end

function Normalize_Data_for_ODE_dispatch(data,
                                         ::Val{:AllFreeODEParams})

    return haskey(data, :ODE_par_fixed) ? data : merge(data, (ODE_par_fixed = NamedTuple()))
end

Normalize_Data_for_ODE_dispatch(data, ::Val{:SomeFixedODEParams}) = datac



#=-------------------------------------------------------------------------------

Parameter assembly

Packs MLP weights, ODE parameters, and (for adaptive mode)
log-space hyperparameters into a single ComponentArray that
the optimizer updates.

-------------------------------------------------------------------------------=#

using ComponentArrays
include("Losses.jl")

"""
    build_trainables(mode, ps_state, ps_g, ode_par_init, hp)

Assemble the flat trainable parameter vector. Fixed mode
omits ε entries.
"""
function build_trainables(::FixedHyper, ps_state, ps_g, ode_par_init, hp::HyperParams)
    ComponentArray(
        StateMLP = ps_state,
        gMLP     = ps_g,
        ODE_par  = ode_par_init,
    )
end

"""
    build_trainables(mode, ps_state, ps_g, ode_par_init, hp)

Assemble the flat trainable parameter vector.  Adaptive mode
adds log-space ε entries.
"""
function build_trainables(::AdaptiveHyper, ps_state, ps_g, ode_par_init, hp::HyperParams)
    ComponentArray(
        StateMLP = ps_state,
        gMLP     = ps_g,
        ODE_par  = ode_par_init,
        hyper    = (
            log_ϵ_ic       = log(hp.ϵ_ic),
            log_ϵ_ode      = log(hp.ϵ_ode),
            log_ϵ_Data     = log(hp.ϵ_Data),
            log_ϵ_L1_state = log(hp.ϵ_L1_state),
            log_ϵ_L1_g     = log(hp.ϵ_L1_g),
        ),
    )
end

"""
    build_box_bounds(ps, ode_bounds) -> (lb, ub)

Construct lower/upper bound ComponentArrays matching the structure of `ps`.
NN weights and hyper params are unconstrained (±Inf); ODE parameters get
finite bounds from `ode_bounds`.  Keys missing from `ode_bounds` default
to (0, +Inf).
"""
function build_box_bounds(ps::ComponentArray, ode_bounds::NamedTuple)
    lb = copy(ps); lb .= -Inf32
    ub = copy(ps); ub .= +Inf32

    hasproperty(ps, :ODE_par) || return lb, ub

    ci         = getaxes(ps)[1][:ODE_par]
    ode_offset = first(ci.idx) - 1
    ode_ax     = ci.ax
    lb_data    = ComponentArrays.getdata(lb)
    ub_data    = ComponentArrays.getdata(ub)

    for k in keys(ode_ax)
        lo, hi = if k in propertynames(ode_bounds)
            let b = getproperty(ode_bounds, k); (b[1], b[2]) end
        else
            (0f0, Inf32)
        end
        flat_idx = ode_offset + ode_ax[k].idx
        lb_data[flat_idx] = lo
        ub_data[flat_idx] = hi
    end

    return lb, ub
end

#=-------------------------------------------------------------------------------

Context construction

Transforms raw data, builds predictor closures via
`build_predictors`, and packs everything into the
Stage 1 / Stage 2 context structs.

-------------------------------------------------------------------------------=#
include("Predictors.jl")
include("Transforms.jl")

# Checks the data tuple for raw data/IC and returns accordingly
raw_training_targets(data) = hasproperty(data, :Y_train_raw) ? data.Y_train_raw : data.Y_train
raw_initial_state(data) = hasproperty(data, :y0_init_raw) ? data.y0_init_raw : data.y0_init

target_scale(Y_train) = max.(std(Y_train; dims = 2), 1f-6)

"""
    build_contexts(State_MLP, st_StateMLP, g_MLP, st_gMLP,
                   data, hp; transform, arch_mode)

Build both training contexts.  The architecture mode controls
which predictor closures are created and whether IC loss is
skipped.
"""
function build_contexts(
    State_MLP, st_StateMLP, g_MLP, st_gMLP,
    data, hp::HyperParams;
    transform::AbstractStateTransform = IdentityTransform(),
    arch_mode::AbstractArchitectureMode = LuxMLP(),
    param_train_mode::Symbol = :AllFreeODEParams,
)
    # Transform raw training data into network coordinates
    Y_train_raw = raw_training_targets(data)
    y0_raw      = raw_initial_state(data)

    Y_train     = forward_state(transform, Y_train_raw)
    y0_obs      = forward_state(transform, y0_raw)
    Y_train_std = target_scale(Y_train)

    # Build predictor closures — the arch_mode dispatch selects
    # soft vs hardwired IC, FD vs analytic derivatives, etc.
    predictors = build_predictors(
        arch_mode,
        State_MLP, st_StateMLP,
        g_MLP, st_gMLP,
        y0_obs,
    )

    ctx_stage1 = PINNCtxStage1(
        predictors.predict_state,
        predictors.predict_deriv,
        data.t_train,
        Y_train,
        Y_train_std,
        y0_obs,
        transform,
        predictors.skip_ic_loss,
        hp.ϵ_ic,
        hp.ϵ_Data,
        hp.ϵ_L1_state,
        data.t_span,
    )

    ctx_stage2 = PINNCtxStage2(
        predictors.predict_state,
        predictors.predict_deriv,
        predictors.predict_g,
        data.t_train,
        Y_train,
        Y_train_std,
        Y_train_std,      # placeholder for G_stage1_std; replaced before Stage 2
        y0_obs,
        transform,
        predictors.skip_ic_loss,
        hp.ϵ_ic,
        hp.ϵ_Data,
        hp.ϵ_ode,
        hp.ϵ_L1_state,
        hp.ϵ_L1_g,
        data.t_dense,
        data.t_span,
        data.ODE_par_bounds,
        data.ODE_par_fixed
    )

    return ctx_stage1, ctx_stage2
end

"""
    rebuild_ctx_stage2(ctx, frozen_scale)

Return a new Stage 2 context identical to `ctx` except that
`G_stage1_std` is replaced with `frozen_scale`.
"""
function rebuild_ctx_stage2(ctx::PINNCtxStage2, frozen_scale::AbstractMatrix)
    PINNCtxStage2(
        ctx.predict_state,
        ctx.predict_deriv,
        ctx.predict_g,
        ctx.t_train,
        ctx.Y_train,
        ctx.Y_train_std,
        frozen_scale,
        ctx.y0_obs,
        ctx.transform,
        ctx.skip_ic_loss,
        ctx.ϵ_ic,
        ctx.ϵ_Data,
        ctx.ϵ_ode,
        ctx.ϵ_L1_state,
        ctx.ϵ_L1_g,
        ctx.t_dense,
        ctx.t_span,
        ctx.ODE_par_bounds,
        ctx.ODE_par_fixed
    )
end

#=-------------------------------------------------------------------------------

Stage Runners

-------------------------------------------------------------------------------=#

"""
    run_stage1(initial_params, ctx, opt, maxiters,
               supervised_loss_fn, callback, mode)

Run Stage 1 (supervised) optimisation and return the trained
parameter vector.
"""
function run_stage1(
    initial_params,
    ctx_stage1,
    Opt_alg_stage1,
    maxiters_stage1,
    default_supervised_loss,
    callback_function,
    mode::AbstractHyperMode,
)
    println("########## Starting Stage 1: Supervised Training ##########")

    loss_closure(ps, ctx) = default_supervised_loss(ps, ctx, mode)
    optfun = make_optfun(loss_closure)
    prob   = Optimization.OptimizationProblem(optfun, initial_params, ctx_stage1)

    res = Optimization.solve(
        prob, Opt_alg_stage1;
        maxiters = maxiters_stage1,
        callback = callback_function,
    )

    return res.u
end

_maybe_wrap_fminbox(opt, ::Val{false}) = opt
_maybe_wrap_fminbox(opt, ::Val{true})                          = opt
_maybe_wrap_fminbox(opt::Optim.AbstractOptimizer, ::Val{true}) = Optim.Fminbox(opt)

"""
    run_stage2(stage1_params, ctx, opt, maxiters, architecture,
               ode_param_constructor, user_losses, callback,
               unsupervised_loss_fn, mode; lb, ub)

Run Stage 2 (ODE-regularised) optimisation and return the
trained parameter vector.  When `lb`/`ub` are provided the
optimizer enforces box constraints natively.
"""
function run_stage2(
    stage1_params,
    ctx_stage2,
    Opt_alg_stage2,
    maxiters_stage2,
    architecture::Function,
    ode_param_constructor,
    user_loss_functions::AbstractVector{<:Function},
    callback_function::Function,
    default_unsupervised_loss,
    mode::AbstractHyperMode,
    param_train_mode::Symbol;
    lb = nothing,
    ub = nothing,
)
    println("########## Starting Stage 2: ODE-Regularized Training ##########")

    # Core physics-informed loss
    loss_closure(ps, ctx) = default_unsupervised_loss(
        ps, ctx, mode, architecture, ode_param_constructor, param_train_mode
    )

    # Combine with any user-supplied auxiliary losses
    loss_combined(ps, ctx) = begin
        l1 = loss_closure(ps, ctx)
        l2 = isempty(user_loss_functions) ? zero(l1) :
            sum(f(ps, ctx) for f in user_loss_functions)
        l1 + l2
    end

    optfun = make_optfun(loss_combined)
    prob   = Optimization.OptimizationProblem(optfun, stage1_params, ctx_stage2;
                                              lb = lb, ub = ub)

    has_bounds = lb !== nothing && ub !== nothing
    opt_effective = _maybe_wrap_fminbox(Opt_alg_stage2, Val(has_bounds))

    res = Optimization.solve(
        prob, opt_effective;
        maxiters = maxiters_stage2,
        callback = callback_function,
    )

    return res.u
end

#=-------------------------------------------------------------------------------

Top-level entry points

-------------------------------------------------------------------------------=#

using Random
using Base.Threads

"""
    train(mode, data, architecture, ode_param_constructor, hp; kwargs...)

Full two-stage training with explicit hyper mode (Fixed or Adaptive).
"""
function train(
    mode::AbstractHyperMode,
    data,
    architecture::Function,
    ode_param_constructor,
    hp::HyperParams = HyperParams(1f0, 1f0, 1f0, 1f0, 1f0);
    transform::AbstractStateTransform = IdentityTransform(),
    arch_mode::AbstractArchitectureMode = LuxMLP(),
    param_train_mode::Symbol = :AllFreeODEParams,
    user_loss_functions::AbstractVector{<:Function} = Function[],
    seed::Integer = 1,
    maxiters_stage1::Integer = 10_000,
    maxiters_stage2::Integer = 1_000,
    build_state_mlp::Function = build_state_mlp,
    build_g_mlp::Function     = build_g_mlp,
    callback_function::Function = callback_function_default,
    Opt_alg_stage1 = OptimizationOptimisers.Adam(1f-3),
    Opt_alg_stage2 = OptimizationOptimJL.LBFGS(m = 25),
    default_supervised_loss   = supervised_loss,
    default_unsupervised_loss = unsupervised_loss,
    )
    rng = MersenneTwister(seed)

    # 0. Make sure the Data tuple has everything needed
    data = Normalize_Data_for_ODE_dispatch(data, Val(param_train_mode))
    @assert all(haskey(data, k) for k in (:ODE_par_init, :ODE_par_fixed)) "data is missing one or more required fields: $([k for k in (:ODE_par_init, :ODE_par_fixed) if !haskey(data, k)])"

    # 1. Build networks and raw parameters (arch_mode dispatches in Predictors.jl)
    State_MLP, st_StateMLP, g_MLP, st_gMLP, raw_ps =
        initialize_components(arch_mode, rng, data, build_state_mlp, build_g_mlp)

    # 2. Append hyper entries if adaptive, then wrap in ComponentArray
    initial_params = build_trainables(mode, raw_ps.StateMLP, raw_ps.gMLP, raw_ps.ODE_par, hp)

    # 3. Build training contexts (applies transform, creates predictor closures)
    ctx_stage1, ctx_stage2 = build_contexts(
        State_MLP, st_StateMLP, g_MLP, st_gMLP, data, hp;
        transform = transform, arch_mode = arch_mode,
    )

    # 3. Stage 1: supervised data fitting
    ps_post_stage1 = run_stage1(
        initial_params, ctx_stage1, Opt_alg_stage1, maxiters_stage1,
        default_supervised_loss, callback_function, mode,
    )

    # 4. Freeze derivative scale from Stage 1 network
    frozen_scale = compute_frozen_ode_scale(ps_post_stage1, ctx_stage2)
    ctx_stage2 = rebuild_ctx_stage2(ctx_stage2, frozen_scale)

    # 5. Build box bounds and run Stage 2: ODE-regularised training
    lb, ub = build_box_bounds(ps_post_stage1, data.ODE_par_bounds)
    ps_trained = run_stage2(
        ps_post_stage1, ctx_stage2, Opt_alg_stage2, maxiters_stage2,
        architecture, ode_param_constructor,
        user_loss_functions, callback_function,
        default_unsupervised_loss, mode,
        param_train_mode;
        lb = lb, ub = ub,
    )

    # 6. Final metrics
    println("########## Training Complete. Calculating Final Metrics ##########")
    d_mse, o_mse, g_s = metrics_stage2(
        ps_trained, ctx_stage2, architecture, ode_param_constructor, param_train_mode
    )
    metrics = (data_mse = d_mse, ode_mse = o_mse, g_std = g_s)

    hyper_final = extract_learned_hyper(ps_trained)

    return TrainResult{typeof(ps_trained), typeof(ctx_stage2),
                       typeof(State_MLP), typeof(g_MLP)}(
        hyper_final, ps_trained, ctx_stage2, State_MLP, g_MLP, metrics
    )
end
