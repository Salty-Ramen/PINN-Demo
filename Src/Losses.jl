using ComponentArrays
using Lux, NNlib
using Optimization, OptimizationOptimisers
#using Zygote
using Enzyme
using ForwardDiff
using Statistics
using Random

#================================================================================

Since we essentially have the same training procedure but for different flags,
we can use julia's multiple dispatch to make the code lean and simpler

================================================================================#


abstract type AbstractHyperMode end
struct FixedHyper <: AbstractHyperMode end
struct AdaptiveHyper <: AbstractHyperMode end



dist_outside(x, lower, upper) = sum(@. max(0f0, lower - x) + max(0f0, x - upper))

# MSE(ŷ, y) = Statistics.mean(abs2, vec(ŷ .- y))

function MSE(ŷ, y, denom)  # normalized MSE
    t = vec((ŷ .- y) ./ denom)
    Statistics.mean(abs2, t)
end

function dNNdt_fd(smodel, tvec::AbstractVector; h::Real = 1f-3)
    # Build two time batches: t + h and t - h, each as 1×B
    Tplus  = reshape(tvec .+ h, 1, :)
    Tminus = reshape(tvec .- h, 1, :)

    Yplus  = smodel(Tplus)    # n_states×B
    Yminus = smodel(Tminus)   # n_states×B

    dYdt = (Yplus .- Yminus) ./ (2h)  # n_states×B
    return dYdt
end

"""
    compute_frozen_ode_scale(ps_stage1, ctx_stage1_or_2; min_scale=1f-6)

Differentiate the Stage 1 state network on the dense collocation grid and
return a frozen (n_states × 1) normalization scale.

Call this *once* between Stage 1 and Stage 2. The returned matrix is detached
from the computation graph — it's a plain array constant.

Arguments
- `ps_stage1`:  parameter vector at the end of Stage 1
- `ctx_stage1_or_2`: any context that has `.State_MLP`, `.st_StateMLP`,
                      and `.t_dense` (Stage 2 context works fine)
- `min_scale`:  floor to prevent degenerate normalization on flat states
"""
function compute_frozen_ode_scale(
    ps_stage1,
    ctx;
    min_scale::Float32 = 1f-6,
)
    smodel = state_model(ps_stage1, ctx)
    t_dense_vec = vec(ctx.t_dense)

    # Reuse the existing FD helper — the Stage 1 network is smooth so
    # the O(h²) error from central differences is negligible here
    dYdt = dNNdt_fd(smodel, t_dense_vec)

    scale = std(dYdt; dims = 2)
    scale = max.(scale, min_scale)

    # Detach from any AD tape so this is treated as a constant in Stage 2
    return Float32.(Array(scale))
end


# L1 penalty is now network specific to have finer control of the optimization
function l1_mean(x)
    return sum(abs, x) / length(x)
end

function l1_penalty_state(ps)
    ws = ComponentArrays.getdata(ps.StateMLP)
    return l1_mean(ws)
end

function l1_penalty_g(ps)
    wg = ComponentArrays.getdata(ps.gMLP)
    return l1_mean(wg)
end

function param_bound_penalty(ps, ctx::PINNCtxStage2)
    p = ps.ODE_par
    b = ctx.ODE_par_bounds
    keys = propertynames(b)

    mapreduce(+, keys) do k
        x      = getproperty(p, k)
        lo, hi = getproperty(b, k)
        dist_outside(x, lo, hi)
    end
end

function bound_loss(ps, ctx::PINNCtxStage2; scale=1000f0)
    scale * param_bound_penalty(ps, ctx)
end


function loss_Data(ps, ctx)
    smodel = state_model(ps, ctx)
    Ttrain = ctx.t_train
    Ytrain = ctx.Y_train
    YtrainSTD = ctx.Y_train_std
    Data_MSE = MSE(smodel(Ttrain), Ytrain, YtrainSTD)
    return Data_MSE
end

function loss_IC(ps, ctx)
    smodel = state_model(ps, ctx)
    # tspan = ctx.t_span
    init_obs = ctx.y0_obs
    denom = max.(abs.(init_obs), 1f-6)
    # This needs to change. Currently is an arbitrary value
    IC_MSE = MSE(smodel([0f0]'), init_obs, denom)
    return IC_MSE
end

function loss_ODE(
    ps,
    ctx::PINNCtxStage2,
    architecture::Function,
    ODE_param_constructor,
    )
    smodel   = state_model(ps, ctx)
    gmodel  = g_model(ps, ctx)
    ODE_par = ODE_param_constructor(ps.ODE_par)
    Tdense  = ctx.t_dense
    ẑ = smodel(Tdense)
    g_arr = gmodel(Tdense)
    dNNdt   = dNNdt_fd(smodel, vec(Tdense))         # 3×B
    f_ŷ     = transformed_rhs(ctx.transform, ẑ, g_arr, ODE_par, architecture)

    return  MSE(dNNdt, f_ŷ, ctx.G_stage1_std)
end


# function self_adaptive_supervised_loss(ps, ctx::PINNCtxStage1)

#     hyperparams = ps.hyper
   
#     ϵ_ic_sq        = exp(2f0 * hyperparams.log_ϵ_ic)       + 1f-6
#     ϵ_Data_sq      = exp(2f0 * hyperparams.log_ϵ_Data)     + 1f-6
#     ϵ_L1_state_sq  = exp(2f0 * hyperparams.log_ϵ_L1_state) + 1f-6

#     hyper_loss = 2f0 * hyperparams.log_ϵ_ic +
#         2f0 * hyperparams.log_ϵ_Data +
#         2f0 * hyperparams.log_ϵ_L1_state

#     return 0.5 * (
#         loss_IC(ps, ctx) / ϵ_ic_sq +
#         loss_Data(ps, ctx) / ϵ_Data_sq +
#         l1_penalty_state(ps) / ϵ_L1_state_sq + 
#         hyper_loss
#     )
# end

# function self_adaptive_unsupervised_loss(ps, ctx::PINNCtxStage2,
#                                          architecture::Function,
#                                          ODE_param_constructor
#                                          )

#     hyperparams = ps.hyper
    
#     ϵ_ic_sq       = exp(2f0 * hyperparams.log_ϵ_ic)       + 1f-6
#     ϵ_Data_sq     = exp(2f0 * hyperparams.log_ϵ_Data)     + 1f-6
#     ϵ_ode_sq      = exp(2f0 * hyperparams.log_ϵ_ode)      + 1f-6
#     ϵ_L1_state_sq = exp(2f0 * hyperparams.log_ϵ_L1_state) + 1f-6
#     ϵ_L1_g_sq     = exp(2f0 * hyperparams.log_ϵ_L1_g)     + 1f-6

#     hyper_loss = 2f0 * hyperparams.log_ϵ_ic +
#         2f0 * hyperparams.log_ϵ_Data +
#         2f0 * hyperparams.log_ϵ_ode +
#         2f0 * hyperparams.log_ϵ_L1_state +
#         2f0 * hyperparams.log_ϵ_L1_g
    
#     return 0.5 * (
#         loss_IC(ps, ctx) / ϵ_ic_sq +
#         loss_Data(ps, ctx) / ϵ_Data_sq +
#         loss_ODE(ps, ctx, architecture, ODE_param_constructor) / ϵ_ode_sq +
#         l1_penalty_state(ps) / ϵ_L1_state_sq +
#         l1_penalty_g(ps) / ϵ_L1_g_sq +
#         hyper_loss +
#         param_bound_penalty(ps, ctx)
#     )
# end

# function fixed_supervised_loss(ps, ctx::PINNCtxStage1)
#     ϵ_ic_sq       = ctx.ϵ_ic^2 + 1f-6
#     ϵ_Data_sq     = ctx.ϵ_Data^2 + 1f-6
#     ϵ_L1_state_sq = ctx.ϵ_L1_state^2 + 1f-6

#     return 0.5f0 * (
#         loss_IC(ps, ctx) / ϵ_ic_sq +
#         loss_Data(ps, ctx) / ϵ_Data_sq +
#         l1_penalty_state(ps) / ϵ_L1_state_sq
#     )
# end

# function fixed_unsupervised_loss(ps, ctx::PINNCtxStage2,
#                                  architecture::Function,
#                                  ODE_param_constructor)

#     ϵ_ic_sq       = ctx.ϵ_ic^2 + 1f-6
#     ϵ_Data_sq     = ctx.ϵ_Data^2 + 1f-6
#     ϵ_ode_sq      = ctx.ϵ_ode^2 + 1f-6
#     ϵ_L1_state_sq = ctx.ϵ_L1_state^2 + 1f-6
#     ϵ_L1_g_sq     = ctx.ϵ_L1_g^2 + 1f-6

#     L_ic      = loss_IC(ps, ctx)
#     L_data    = loss_Data(ps, ctx)
#     L_ode     = loss_ODE(ps, ctx, architecture, ODE_param_constructor)
#     L_L1s     = l1_penalty_state(ps)
#     L_L1g     = l1_penalty_g(ps)
#     L_bnd     = param_bound_penalty(ps, ctx)

#     return 0.5f0 * (
#         L_ic   / ϵ_ic_sq +
#         L_data / ϵ_Data_sq +
#         L_ode  / ϵ_ode_sq +
#         L_L1s  / ϵ_L1_state_sq +
#         L_L1g  / ϵ_L1_g_sq +
#         L_bnd
#     )
# end

function resolve_epsilons(ps, ctx, ::FixedHyper)
    (ϵ_ic = ctx.ϵ_ic,
     ϵ_Data = ctx.ϵ_Data,
     # ϵ_ode = ctx.ϵ_ode,
     ϵ_L1_state = ctx.ϵ_L1_state,
     # ϵ_L1_g = ctx.ϵ_L1_g
     ) 
end

function resolve_epsilons(ps, ctx, ::AdaptiveHyper)
    hyperparams = ps.hyper 
  
    (ϵ_ic       = exp(hyperparams.log_ϵ_ic),
     ϵ_Data     = exp(hyperparams.log_ϵ_Data),
     ϵ_ode      = exp(hyperparams.log_ϵ_ode),
     ϵ_L1_state = exp(hyperparams.log_ϵ_L1_state),
     ϵ_L1_g     = exp(hyperparams.log_ϵ_L1_g)) 
end

# Adds a penalty punishing exteme ϵ values for adaptive loss. 0 for fixed.
hyper_penalty(ps, ::FixedHyper) = 0f0

function hyper_penalty(ps, ::AdaptiveHyper)
    hp = ps.hyper
    2f0 * (h.log_ϵ_ic + h.log_ϵ_ode + h.log_ϵ_Data + h.log_ϵ_L1_state + h.log_ϵ_L1_g)
end 

function supervised_loss(
    ps,
    ctx,
    mode::AbstractHyperMode
    )

    hp = resolve_epsilons(ps, ctx, mode)

    penalty = hyper_penalty(ps, mode)

    return 0.5f0 * (
        loss_IC(ps, ctx)        / (hp.ϵ_ic^2 + 1f-6) +
        loss_Data(ps, ctx)      / (hp.ϵ_Data^2 + 1f-6) +
        l1_penalty_state(ps)    / (hp.ϵ_L1_state^2 + 1f-6) +
        penalty
    )
end

function unsupervised_loss(ps,
                           ctx,
                           mode::AbstractHyperMode,
                           architecture,
                           ode_param_constructor)
    hp = resolve_epsilons(ps, ctx, mode)

    penalty = hyper_penalty(ps, mode)

    return 0.5f0 * (
        loss_IC(ps, ctx)                / (hp.ϵ_ic^2 + 1f-6) +
        loss_Data(ps, ctx)              / (hp.ϵ_Data^2 + 1f-6) +
        l1_penalty_state(ps)            / (hp.ϵ_L1_state^2 + 1f-6) +
        loss_ODE(ps,
                 ctx,
                 architecture,
                 ODE_param_constructor) / (hp.ϵ_ode^2 + 1f-6) +
        l1_penalty_g(ps)                / (hp.ϵ_L1_g^2 + 1f-6) +
        penalty
    )    
end



    

    
    

    

    
    
    
    
