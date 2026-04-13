#=-------------------------------------------------------------------------------
Losses.jl

Loss computations for the PINN training pipeline.

Every loss function accesses model predictions exclusively through the
three predictor closures stored on the context struct:

    ctx.predict_state(ps, t_grid)  →  (n_states × B)
    ctx.predict_deriv(ps, t_grid)  →  (n_states × B)
    ctx.predict_g(ps, t_grid)      →  (n_g × B)

The file is organised top-down:
  1. Scalar utilities (MSE, distance penalties)
  2. Individual loss terms
  3. Hyperparameter resolution (fixed vs adaptive)
  4. Composite loss assembly (supervised / unsupervised)
-------------------------------------------------------------------------------=#

#=-------------------------------------------------------------------------------

Scalar Utilities

-------------------------------------------------------------------------------=#

using Statistics

"""
    MSE(ŷ, y, denom)

Normalised mean squared error.  `denom` is a per-state scale factor
(typically std of training targets) that prevents states with large
magnitudes from dominating the loss.
"""
function MSE(ŷ, y, denom)
    t = vec((ŷ - y) ./ denom)
    Statistics.mean(abs2, t)
end

"""
    dist_outside(x, lower, upper)

Sum of constraint violations: how far `x` lies outside `[lower, upper]`.
Returns zero when all elements are within bounds. This is used as a means to
calculate how far ODE params in the known physics are violating their
corresponding bounds.
"""
dist_outside(x, lower, upper) = sum(@. max(0f0, lower - x) + max(0f0, x - upper))


"""
    l1_mean(x)

Mean absolute value — an L1 regularization term that encourages sparsity.
"""
l1_mean(x) = sum(abs, x) / length(x)

"""
    metrics_stage2(ps, ctx, architecture, ode_param_constructor)
        -> (data_mse, ode_mse, g_std)

Compute final scalar diagnostics using the trained parameters.
"""
function metrics_stage2(ps, ctx, architecture::Function, ode_param_constructor)
    data_mse = loss_Data(ps, ctx)
    ode_mse  = loss_ODE(ps, ctx, architecture, ode_param_constructor)

    # Standard deviation of g-network output as a health check:
    # near-zero std suggests the g-network collapsed to a constant
    g_out = ctx.predict_g(ps, ctx.t_dense)
    g_std = std(vec(g_out))

    return (data_mse = data_mse, ode_mse = ode_mse, g_std = g_std)
end

#=-------------------------------------------------------------------------------

Individual Loss Functions

Each function takes (ps, ctx) and returns a scalar. The predictors are expected
to be in ctx

-------------------------------------------------------------------------------=#

"""
    loss_Data(ps, ctx) -> Float32

MSE between state predictions at training time points and the
(transformed) training targets.
"""
function loss_Data(ps, ctx)
    ŷ = ctx.predict_state(ps, ctx.t_train)
    return MSE(ŷ, ctx.Y_train, ctx.Y_train_std)
end


"""
    loss_IC(ps, ctx) -> Float32

MSE between the predicted initial state and the (transformed)
observed IC.  Should be skipped when the IC is structurally
enforced (hard-wired into the state predictor).
"""
function loss_IC(ps, ctx)
    ŷ =  ctx.predict_state(ps, ctx.t_train)
    denom = max.(abs.(ctx.y0_obs), 1f-06)
    return MSE(ŷ, ctx.y0_obs, denom)
end


"""
   loss_ODE(ps, ctx, architecture, ode_param_constructor)

Physics-residual loss: MSE between the state network's time derivative
(from `predict_deriv`) and the architecture RHS evaluated with
the current state and g predictions.

`architecture(y_raw, g, ode_params)` is the user's grey-box RHS in raw
coordinates; `transformed_rhs` handles the chain-rule bridge to the
network's coordinate space
"""
function loss_ODE(ps, ctx, architecture::Function, ode_param_constructor)
    t_dense = ctx.t_dense
    ode_par = ode_param_constructor(ps.ODE_par)

    # All three predictions via contract closures
    ẑ     = ctx.predict_state(ps, t_dense)
    dẑdt  = ctx.predict_deriv(ps, t_dense)
    g_arr = ctx.predict_g(ps, t_dense)

    # RHS in the transformed coordinate space
    f_ẑ = transformed_rhs(ctx.transform, ẑ, g_arr, ode_par, architecture)

    return MSE(dẑdt, f_ẑ, ctx.G_stage1_std)
end


#-----L1 Regularization----------------------------------------------------------

"""
    l1_penalty_state(ps) -> Float32

Mean absolute weight magnitude of the state network.
"""
function l1_penalty_state(ps)
    ws = ComponentArrays.getdata(ps.StateMLP)
    return l1_mean(ws)
end


"""
    l1_penalty_g(ps) -> Float32

Mean absolute weight magnitude of the g-network.
"""
function l1_penalty_g(ps)
    wg = ComponentArrays.getdata(ps.gMLP)
    return l1_mean(wg)
end

#-----Parameter Bound Penalty----------------------------------------------------

"""
    param_bound_penalty(ps, ctx) -> Float32

Soft constraint that penalises ODE parameters straying outside
their declared bounds.  Uses a one-sided quadratic barrier.
"""
function param_bound_penalty(ps, ctx)
    p = ps.ODE_par
    b = ctx.ODE_par_bounds
    keys = propertynames(b)

    mapreduce(+, keys) do k
        x      = getproperty(p, k)
        lo, hi = getproperty(b, k)
        dist_outside(x, lo, hi)
    end
end

"""
    bound_loss(ps, ctx; scale=1000f0) -> Float32

Scaled wrapper around `param_bound_penalty` for direct use in
composite losses.
"""
bound_loss(ps, ctx; scale = 1000f0) = scale * param_bound_penalty(ps, ctx)


#=-------------------------------------------------------------------------------

Hyperparameter Resolution

Maps (ps, ctx, mode) → a NamedTuple of ε values.
Fixed mode reads ε from the context; adaptive mode reads
log(ε) from the trainable parameters and exponentiates.

-------------------------------------------------------------------------------=#

#-----Dispatch Tags--------------------------------------------------------------

abstract type AbstractHyperMode end
struct FixedHyper <: AbstractHyperMode end
struct AdaptiveHyper <: AbstractHyperMode end

#-----Fixed Hyperparameter ϵ resolver--------------------------------------------

function resolve_epsilons(ps, ctx::PINNCtxStage1, ::FixedHyper)
    (ϵ_ic       = ctx.ϵ_ic,
     ϵ_Data     = ctx.ϵ_Data,
     ϵ_L1_state = ctx.ϵ_L1_state)
end

function resolve_epsilons(ps, ctx::PINNCtxStage2, ::FixedHyper)
    (ϵ_ic       = ctx.ϵ_ic,
     ϵ_Data     = ctx.ϵ_Data,
     ϵ_ode      = ctx.ϵ_ode,
     ϵ_L1_state = ctx.ϵ_L1_state,
     ϵ_L1_g     = ctx.ϵ_L1_g)
end

#-----Adaptive Hyperparameter ϵ resolver-----------------------------------------

function resolve_epsilons(ps, ctx::PINNCtxStage1, ::AdaptiveHyper)
    h = ps.hyper
    (ϵ_ic       = exp(h.log_ϵ_ic),
     ϵ_Data     = exp(h.log_ϵ_Data),
     ϵ_L1_state = exp(h.log_ϵ_L1_state))
end

function resolve_epsilons(ps, ctx::PINNCtxStage2, ::AdaptiveHyper)
    h = ps.hyper
    (ϵ_ic       = exp(h.log_ϵ_ic),
     ϵ_Data     = exp(h.log_ϵ_Data),
     ϵ_ode      = exp(h.log_ϵ_ode),
     ϵ_L1_state = exp(h.log_ϵ_L1_state),
     ϵ_L1_g     = exp(h.log_ϵ_L1_g))
end

#-----Hyper Penalty--------------------------------------------------------------

hyper_penalty(ps, ::FixedHyper) = 0f0

function hyper_penalty(ps, ::AdaptiveHyper)
    h = ps.hyper
    2f0 * (h.log_ϵ_ic + h.log_ϵ_ode + h.log_ϵ_Data +
           h.log_ϵ_L1_state + h.log_ϵ_L1_g)
end


#=-------------------------------------------------------------------------------

Composite loss assembly

These combine the individual terms with ε weighting.
The `skip_ic_loss` flag on the context controls whether
the IC term is included (soft IC) or dropped (hardwired IC).

-------------------------------------------------------------------------------=#

"""
    supervised_loss(ps, ctx, mode) -> Float32

Stage 1 composite: data fit + IC penalty + L1 state regularisation.
IC term is omitted when `ctx.skip_ic_loss == true`.
"""
function supervised_loss(ps, ctx, mode::AbstractHyperMode)
    hp = resolve_epsilons(ps, ctx, mode)
    penalty = hyper_penalty(ps, mode)

    L = loss_Data(ps, ctx) / (hp.ϵ_Data^2 + 1f-6) +
        l1_penalty_state(ps) / (hp.ϵ_L1_state^2 + 1f-6) +
        penalty

    # IC term: skip when the architecture structurally enforces it
    if !ctx.skip_ic_loss
        L += loss_IC(ps, ctx) / (hp.ϵ_ic^2 + 1f-6)
    end

    return 0.5f0 * L
end


"""
    unsupervised_loss(ps, ctx, mode, architecture, ode_param_constructor) -> Float32

Stage 2 composite: data fit + IC penalty + ODE residual + L1 on both
networks + parameter bound penalty.
IC term is omitted when `ctx.skip_ic_loss == true`.
"""
function unsupervised_loss(ps, ctx, mode::AbstractHyperMode,
                           architecture, ode_param_constructor)
    hp = resolve_epsilons(ps, ctx, mode)
    penalty = hyper_penalty(ps, mode)

    L = loss_Data(ps, ctx)    / (hp.ϵ_Data^2 + 1f-6) +
        l1_penalty_state(ps)  / (hp.ϵ_L1_state^2 + 1f-6) +
        loss_ODE(ps, ctx,
                 architecture,
                 ode_param_constructor) / (hp.ϵ_ode^2 + 1f-6) +
        l1_penalty_g(ps)      / (hp.ϵ_L1_g^2 + 1f-6) +
        penalty

    if !ctx.skip_ic_loss
        L += loss_IC(ps, ctx) / (hp.ϵ_ic^2 + 1f-6)
    end

    return 0.5f0 * L
end
