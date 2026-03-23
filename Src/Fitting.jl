using ComponentArrays
using Lux, NNlib
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Zygote
# using Enzyme
using ForwardDiff
using Statistics
using Random
using Base.Threads


#--------Context Structs--------
struct HyperParams
    ϵ_ic        :: Float32
    ϵ_ode       :: Float32
    ϵ_Data      :: Float32
    ϵ_L1_state  :: Float32
    ϵ_L1_g      :: Float32
end

struct PINNCtxStage1{M,ST,T,Y,YS,V,TR}
    State_MLP   :: M
    st_StateMLP :: ST
    t_train     :: T
    Y_train     :: Y
    Y_train_std :: YS
    y0_obs      :: V
    transform   :: TR
    ϵ_ic        :: Float32
    ϵ_Data      :: Float32
    ϵ_L1_state  :: Float32
    t_span      :: Vector{Float32}
end

struct PINNCtxStage2{M,ST,GM,STG,T,Y,YS,V,TD,TR}
    State_MLP      :: M
    st_StateMLP    :: ST
    g_MLP          :: GM
    st_gMLP        :: STG
    t_train        :: T
    Y_train        :: Y
    Y_train_std    :: YS
    y0_obs         :: V
    transform      :: TR
    ϵ_ic           :: Float32
    ϵ_Data         :: Float32
    ϵ_ode          :: Float32
    ϵ_L1_state     :: Float32
    ϵ_L1_g         :: Float32
    t_dense        :: TD
    t_span         :: Vector{Float32}
    ODE_par_bounds :: NamedTuple
end

include("Losses.jl")
include("Transforms.jl")
 
struct TrainResult{PS,CTX,M1,M2}
    hyper    :: HyperParams
    params   :: PS
    ctx2     :: CTX
    state_mlp:: M1
    g_mlp    :: M2
    metrics  :: NamedTuple
end

#--------Helpers--------


state_model(ps, ctx) =
    Lux.StatefulLuxLayer(ctx.State_MLP, ps.StateMLP, ctx.st_StateMLP)

g_model(ps, ctx::PINNCtxStage2) =
    Lux.StatefulLuxLayer(ctx.g_MLP, ps.gMLP, ctx.st_gMLP)


function metrics_stage2(ps, ctx::PINNCtxStage2, architecture::Function, ODE_params)
    smodel  = state_model(ps, ctx)
    gmodel  = g_model(ps, ctx)
    data_mse = MSE(smodel(ctx.t_train), ctx.Y_train, ctx.Y_train_std)

    ODE_par = ODE_params(ps.ODE_par)
    Tdense  = ctx.t_dense
    dNNdt   = dNNdt_fd(smodel, vec(Tdense))
    ẑ       = smodel(Tdense) 
    f_ŷ     = transformed_rhs(ctx.transform, ẑ, gmodel(Tdense), ODE_par, architecture)
    ode_mse = MSE(dNNdt, f_ŷ, std(dNNdt; dims = 2))

    g_out   = gmodel(Tdense)
    g_std   = std(vec(g_out))

    return data_mse, ode_mse, g_std
end

# callback_function_default = (state, l) -> false 

function callback_function_default(state, l)
    N = 25
    if state.iter % (N*100) == 0
        println("iteration =", state.iter)
    end
    return false
end


function make_optfun(loss)
    Optimization.OptimizationFunction(
        (θ, p) -> loss(θ, p),
        Optimization.AutoZygote()
    )
end

function initialize_components(rng, data, hp::HyperParams, build_state_mlp::Function, build_g_mlp::Function)
    # 2a. Build networks
    State_MLP = build_state_mlp()
    ps_StateMLP, st_StateMLP = Lux.setup(rng, State_MLP)

    g_MLP = build_g_mlp()
    ps_gMLP, st_gMLP = Lux.setup(rng, g_MLP)

    # 2b. Initial ODE parameters (NamedTuple)
    trainable_params = ComponentArrays.ComponentArray(
        StateMLP = ps_StateMLP,
        gMLP     = ps_gMLP,
        ODE_par  = data.ODE_par_init,
        # log space search might be better conditioned
        hyper = (
            log_ϵ_ic       = log(hp.ϵ_ic),
            log_ϵ_ode      = log(hp.ϵ_ode),
            log_ϵ_Data     = log(hp.ϵ_Data),
            log_ϵ_L1_state = log(hp.ϵ_L1_state),
            log_ϵ_L1_g     = log(hp.ϵ_L1_g)
        )
    )

    return State_MLP, st_StateMLP, g_MLP, st_gMLP, trainable_params
end

# Checks the data tuple for raw data/IC and returns accordingly
raw_training_targets(data) = hasproperty(data, :Y_train_raw) ? data.Y_train_raw : data.Y_train
raw_initial_state(data) = hasproperty(data, :y0_init_raw) ? data.y0_init_raw : data.y0_init

target_scale(Y_train) = max.(std(Y_train; dims = 2), 1f-6)

function build_contexts(State_MLP, st_StateMLP, g_MLP, st_gMLP, data, hp::HyperParams;
                        transform::AbstractStateTransform = IdentityTransform())

    # Transform the training targets (raw data)
    Y_train_raw = raw_training_targets(data)
    y0_raw = raw_initial_state(data)

    Y_train = forward_state(transform, Y_train_raw)
    y0_obs = forward_state(transform, y0_raw)
    Y_train_std = target_scale(Y_train)

    
    
    # Context for Stage 1 (Supervised/Data-fitting)
    ctx_stage1 = PINNCtxStage1(
        State_MLP,
        st_StateMLP,
        data.t_train,
        Y_train,
        Y_train_std,
        y0_obs,
        transform,
        hp.ϵ_ic,
        hp.ϵ_Data,
        hp.ϵ_L1_state,
        data.t_span,
    )
    
    # Context for Stage 2 (ODE-regularized/Unsupervised)
    ctx_stage2 = PINNCtxStage2(
        State_MLP,
        st_StateMLP,
        g_MLP,
        st_gMLP,
        data.t_train,
        Y_train,
        Y_train_std,
        y0_obs,
        transform,
        hp.ϵ_ic,
        hp.ϵ_Data,
        hp.ϵ_ode,
        hp.ϵ_L1_state,
        hp.ϵ_L1_g,
        data.t_dense,
        data.t_span,
        data.ODE_par_bounds,
    )
    
    return ctx_stage1, ctx_stage2
end

function run_stage1(initial_params,
                    ctx_stage1,
                    Opt_alg_stage1,
                    maxiters_stage1,
                    default_supervised_loss,
                    callback_function)
    println("########## Starting Stage 1: Supervised Training ##########")
    optfun1 = make_optfun(default_supervised_loss)
    prob1   = Optimization.OptimizationProblem(optfun1, initial_params, ctx_stage1)

    res1    = Optimization.solve(
        prob1,
        Opt_alg_stage1;
        maxiters = maxiters_stage1,
        callback = callback_function
    )

    return res1.u
end

function run_stage2(
    stage1_params, 
    ctx_stage2, 
    Opt_alg_stage2, 
    maxiters_stage2, 
    architecture::Function, 
    ode_param_constructor, 
    user_loss_functions::AbstractVector{<:Function}, 
    callback_function::Function, 
    default_unsupervised_loss
)
    println("########## Starting Stage 2: ODE-Regularized Training ##########")

    # Define the core unsupervised loss closure (used for ODE and L1 regularization)
    loss_closure(ps, ctx) = default_unsupervised_loss(ps, ctx, architecture, ode_param_constructor)

    # Define the combined loss, including user-defined losses
    loss_combined(ps, ctx) = begin
        l1 = loss_closure(ps, ctx)
        l2 = isempty(user_loss_functions) ? zero(l1) :
            sum(f(ps, ctx) for f in user_loss_functions)
        l1 + l2
    end

    optfun2 = make_optfun(loss_combined)
    prob2   = Optimization.OptimizationProblem(optfun2, stage1_params, ctx_stage2)

    res2    = Optimization.solve(
        prob2,
        Opt_alg_stage2;
        maxiters = maxiters_stage2,
        callback = callback_function
    )
    
    return res2.u
end

function train_once(
    data,
    architecture::Function,
    ode_param_constructor,
    hp::HyperParams= HyperParams(1f0, 1f0, 1f0, 1f0, 1f0);
    transform::AbstractStateTransform = IdentityTransform(),
    user_loss_functions::AbstractVector{<:Function} = Function[], 
    seed::Integer = 1,
    maxiters_stage1::Integer = 10_000,
    maxiters_stage2::Integer = 1_000,
    build_state_mlp::Function = build_state_mlp,
    build_g_mlp::Function     = build_g_mlp,
    callback_function::Function= callback_function_default,
    Opt_alg_stage1 =  OptimizationOptimisers.Adam(1f-3),
    Opt_alg_stage2 = OptimizationOptimJL.LBFGS(m = 25),
    default_supervised_loss = self_adaptive_supervised_loss,
    default_unsupervised_loss = self_adaptive_unsupervised_loss
    )

    rng = MersenneTwister(seed)
    
    # 1. Initialization and Parameter Assembly
    State_MLP, st_StateMLP, g_MLP, st_gMLP, initial_params = initialize_components(
        rng, data, hp, build_state_mlp, build_g_mlp
    )

    # 2. Context Building
    ctx_stage1, ctx_stage2 = build_contexts(
        State_MLP, st_StateMLP, g_MLP, st_gMLP, data, hp; transform = transform
    )
    
    # 3. Stage 1: Supervised Training
    trainable_params_stage1 = run_stage1(
        initial_params, 
        ctx_stage1, 
        Opt_alg_stage1, 
        maxiters_stage1, 
        default_supervised_loss,
        callback_function
    )

    # 4. Stage 2: ODE-regularized Training
    ps_trained = run_stage2(
        trainable_params_stage1, 
        ctx_stage2, 
        Opt_alg_stage2, 
        maxiters_stage2, 
        architecture, 
        ode_param_constructor, 
        user_loss_functions, 
        callback_function, 
        default_unsupervised_loss
    )

    # 5. Metrics and Return
    println("########## Training Complete. Calculating Final Metrics ##########")
    d_mse, o_mse, g_s = metrics_stage2(ps_trained, ctx_stage2, architecture, ode_param_constructor)
    metrics = (data_mse = d_mse, ode_mse = o_mse, g_std = g_s)

    return TrainResult{typeof(ps_trained), typeof(ctx_stage2), typeof(State_MLP), typeof(g_MLP)}(
        hp, ps_trained, ctx_stage2, State_MLP, g_MLP, metrics
    )
end

########## FIXED HYPERPARAM TRAINING ##########
    
function initialize_components_fixed(rng, data, hp::HyperParams,
                                     build_state_mlp::Function,
                                     build_g_mlp::Function)
    # 1. Build networks
    State_MLP = build_state_mlp()
    ps_StateMLP, st_StateMLP = Lux.setup(rng, State_MLP)

    g_MLP = build_g_mlp()
    ps_gMLP, st_gMLP = Lux.setup(rng, g_MLP)

    # 2. Initial ODE parameters (NamedTuple)
    # Note: `hyper` is EXCLUDED here compared to the standard train_once
    trainable_params = ComponentArrays.ComponentArray(
        StateMLP = ps_StateMLP,
        gMLP     = ps_gMLP,
        ODE_par  = data.ODE_par_init
    )

    return State_MLP, st_StateMLP, g_MLP, st_gMLP, trainable_params
end

function train_fixed_hyper(
    data,
    architecture::Function,
    ode_param_constructor,
    hp::HyperParams= HyperParams(1f0, 1f0, 1f0, 1);
    transform::AbstractStateTransform = IdentityTransform(),
    user_loss_functions::AbstractVector{<:Function} = Function[], 
    seed::Integer = 1,
    maxiters_stage1::Integer = 10_000,
    maxiters_stage2::Integer = 1_000,
    build_state_mlp::Function = build_state_mlp,
    build_g_mlp::Function     = build_g_mlp,
    callback_function::Function= callback_function_default,
    Opt_alg_stage1 =  OptimizationOptimisers.Adam(1f-3),
    Opt_alg_stage2 = OptimizationOptimJL.LBFGS(m = 25),
    # Defaults changed to the fixed versions
    default_supervised_loss = fixed_supervised_loss,
    default_unsupervised_loss = fixed_unsupervised_loss
    )

    rng = MersenneTwister(seed)
    
    # 1. Initialization (Fixed version)
    State_MLP, st_StateMLP, g_MLP, st_gMLP, initial_params = initialize_components_fixed(
        rng, data, hp, build_state_mlp, build_g_mlp
    )

    # 2. Context Building (Standard, as it populates ctx with hp values)
    ctx_stage1, ctx_stage2 = build_contexts(
        State_MLP, st_StateMLP, g_MLP, st_gMLP, data, hp; transform = transform
    )
    
    # 3. Stage 1: Supervised Training
    trainable_params_stage1 = run_stage1(
        initial_params, 
        ctx_stage1, 
        Opt_alg_stage1, 
        maxiters_stage1, 
        default_supervised_loss,
        callback_function
    )

    # 4. Stage 2: ODE-regularized Training
    ps_trained = run_stage2(
        trainable_params_stage1, 
        ctx_stage2, 
        Opt_alg_stage2, 
        maxiters_stage2, 
        architecture, 
        ode_param_constructor, 
        user_loss_functions, 
        callback_function, 
        default_unsupervised_loss
    )

    # 5. Metrics and Return
    println("########## Training Complete (Fixed HyperParams). Calculating Final Metrics ##########")
    d_mse = loss_Data(ps_trained, ctx_stage2)
    o_mse = loss_ODE(ps_trained, ctx_stage2, architecture, ode_param_constructor)
    metrics = (data_mse = d_mse, ode_mse = o_mse)

    return TrainResult{typeof(ps_trained), typeof(ctx_stage2), typeof(State_MLP), typeof(g_MLP)}(
        hp, ps_trained, ctx_stage2, State_MLP, g_MLP, metrics
    )
end








#--------Training Function--------
"""
Run the two-stage training for given `hp::HyperParams` and `data` NamedTuple.

`data` must contain:
    t_train       :: AbstractMatrix (1×N or similar)
    Y_train       :: AbstractMatrix (states×N)
    Y_train_std   :: same shape as Y_train (for normalization)
    t_dense       :: dense time grid (1×N₁)
    t_span        :: Vector{Float32} (e.g. [0f0, t_max])
    y0_init       :: initial condition vector
    ODE_par_init  :: NamedTuple of initial ODE parameters
    ODE_par_bounds:: NamedTuple of (lo,hi) containers for each param
"""
function train_once_legacy(
    data,
    architecture::Function,
    ode_param_constructor,
    hp::HyperParams= HyperParams(1f0, 1f0, 1f0, 1);
    transform = IdentityTransform(),
    user_loss_functions::AbstractVector{<:Function} = Function[], 
    seed::Integer = 1,
    maxiters_stage1::Integer  = 10_000,
    maxiters_stage2::Integer  = 1_000,
    build_state_mlp::Function = build_state_mlp,
    build_g_mlp::Function     = build_g_mlp,
    callback_function::Function= callback_function_default,
    Opt_alg_stage1 =  OptimizationOptimisers.Adam(1f-4),
    Opt_alg_stage2 = OptimizationOptimJL.LBFGS(m = 25),
    default_supervised_loss = loss_supervised,
    default_unsupervised_loss = loss_unsupervised
    )
    rng = MersenneTwister(seed)

    # Build networks
    State_MLP = build_state_mlp()
    ps_StateMLP, st_StateMLP = Lux.setup(rng, State_MLP)

    g_MLP    = build_g_mlp()
    ps_gMLP, st_gMLP = Lux.setup(rng, g_MLP)

    # Initial ODE parameters (NamedTuple)
    trainable_params = ComponentArrays.ComponentArray(
        StateMLP = ps_StateMLP,
        gMLP     = ps_gMLP,
        ODE_par  = data.ODE_par_init,
        hyper    = (
            ϵ_ic  = hp.ϵ_ic,
            ϵ_ode = hp.ϵ_ode,
            ϵ_Data = hp.ϵ_Data,
            ϵ_L1  = hp.ϵ_L1
        )
    )

    ctx_stage1, ctx_stage2 = build_contexts(
        State_MLP, st_StateMLP, g_MLP, st_gMLP, data, hp; transform = transform
    )


    # Small helper for OptimizationFunction
    make_optfun(loss) = Optimization.OptimizationFunction(
        (θ, p) -> loss(θ, p),
        Optimization.AutoZygote()
    )

    ########## Stage 1: supervised ##########
    optfun1 = make_optfun(default_supervised_loss)
    prob1   = Optimization.OptimizationProblem(optfun1, trainable_params, ctx_stage1)

    res1    = Optimization.solve(
        prob1,
        Opt_alg_stage1;
        maxiters = maxiters_stage1,
    )

    trainable_params_stage1 = res1.u

    ########## Stage 2: ODE-regularized ##########
    loss_closure(ps, ctx) = default_unsupervised_loss(ps, ctx, architecture, ode_param_constructor)

    loss_combined(ps, ctx) = begin
	l1 = loss_closure(ps, ctx)
        l2 = isempty(user_loss_functions) ? zero(l1) :
            sum(f(ps, ctx) for f in user_loss_functions)
        l1 + l2
    end

    optfun2 = make_optfun(loss_combined)
    prob2   = Optimization.OptimizationProblem(optfun2, trainable_params_stage1, ctx_stage2)

    res2    = Optimization.solve(
        prob2,
        Opt_alg_stage2;
        maxiters = maxiters_stage2,
        callback = callback_function
    )
    # trainable_params_stage2 = res2.u
    
    
    # prob3   = Optimization.OptimizationProblem(optfun2, trainable_params_stage2, ctx_stage2)    

    # res3    = Optimization.solve(
    #     prob3,
    #     Opt_alg_stage2;
    #     maxiters = maxiters_stage2,
    #     callback = callback_function
    # )

    # ps_trained = res3.u
    ps_trained = res2.u
    
    d_mse, o_mse, g_s = metrics_stage2(ps_trained, ctx_stage2, architecture, ode_param_constructor)
    metrics = (data_mse = d_mse, ode_mse = o_mse, g_std = g_s)

    return TrainResult{typeof(ps_trained), typeof(ctx_stage2), typeof(State_MLP), typeof(g_MLP)}(
        hp, ps_trained, ctx_stage2, State_MLP, g_MLP, metrics
    )
end
