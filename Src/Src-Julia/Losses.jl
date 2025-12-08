using ComponentArrays
using Lux, NNlib
using Optimization, OptimizationOptimisers
#using Zygote
using Enzyme
using ForwardDiff
using Statistics
using Random

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

# Simple L1 penalty on network parameters
function l1_penalty(ps)
    ws = ComponentArrays.getdata(ps.StateMLP)
    wg = ComponentArrays.getdata(ps.gMLP)
    ls = length(ComponentArrays.getdata(ps.StateMLP))
    lg = length(ComponentArrays.getdata(ps.gMLP))
    return sum(abs, ws)/ls + sum(abs, wg)/lg
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
    tspan = ctx.t_span
    init_obs = ctx.y0_obs
    IC_MSE = MSE(smodel([tspan[1]]'), init_obs, std(init_obs, dims = 2))
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
    g_arr = gmodel(Tdense)
    dNNdt   = dNNdt_fd(smodel, vec(Tdense))         # 3×B
    f_ŷ     = architecture(smodel(Tdense), g_arr, ODE_par)
    ODE_MSE  = MSE(dNNdt, f_ŷ, std(dNNdt; dims =2))

    return ODE_MSE
end


function self_adaptive_supervised_loss(ps, ctx::PINNCtxStage1)
    hyperparams = ps.hyper
    
    hyper_loss = log(hyperparams.ϵ_ic^2) +
        log(hyperparams.ϵ_Data^2) +
        log(hyperparams.ϵ_ode^2) +
        log(hyperparams.ϵ_L1^2)

   
    ϵ_ic_sq = hyperparams.ϵ_ic^2 + 1f-6
    ϵ_Data_sq = hyperparams.ϵ_Data^2 + 1f-6
    # ϵ_L1_sq = hyperparams.ϵ_L1^2 + 1f-6 

    return 0.5 * (
        loss_IC(ps, ctx) / ϵ_ic_sq +
        loss_Data(ps, ctx) / ϵ_Data_sq +
      #  l1_penalty(ps) / ϵ_L1_sq + 
        hyper_loss
    )
end

function self_adaptive_unsupervised_loss(ps, ctx::PINNCtxStage2,
                                         architecture::Function,
                                         ODE_param_constructor
                                         )

    hyperparams = ps.hyper
    hyper_loss = log(hyperparams.ϵ_ic^2) +
        log(hyperparams.ϵ_Data^2) +
        log(hyperparams.ϵ_ode^2) +
        log(hyperparams.ϵ_L1^2)


    ϵ_ic_sq = hyperparams.ϵ_ic^2 + 1f-6
    ϵ_Data_sq = hyperparams.ϵ_Data^2 + 1f-6
    ϵ_ode_sq = hyperparams.ϵ_ode^2 + 1f-6
    ϵ_L1_sq = hyperparams.ϵ_L1^2 + 1f-6


    return 0.5 * (
        loss_IC(ps, ctx) / ϵ_ic_sq +
        loss_Data(ps, ctx) / ϵ_Data_sq +
        loss_ODE(ps, ctx, architecture, ODE_param_constructor) / ϵ_ode_sq +
        l1_penalty(ps) / ϵ_L1_sq +
        hyper_loss +
        param_bound_penalty(ps, ctx)
    )
end


function fixed_supervised_loss(ps, ctx::PINNCtxStage1)
    # Extract fixed tolerances from context
    ϵ_ic_sq   = ctx.ϵ_ic^2 + 1f-6
    ϵ_Data_sq = ctx.ϵ_Data^2 + 1f-6
    ϵ_L1_sq   = ctx.ϵ_L1^2 + 1f-6 

    return 0.5 * (
        loss_IC(ps, ctx) / ϵ_ic_sq +
        loss_Data(ps, ctx) / ϵ_Data_sq 
        #+ l1_penalty(ps) / ϵ_L1_sq  # Uncomment if L1 is desired in Stage 1
    )
end

function fixed_unsupervised_loss(ps, ctx::PINNCtxStage2,
                                 architecture::Function,
                                 ODE_param_constructor
                                 )

    # Extract fixed tolerances from context
    ϵ_ic_sq   = ctx.ϵ_ic^2 + 1f-6
    ϵ_Data_sq = ctx.ϵ_Data^2 + 1f-6
    ϵ_ode_sq  = ctx.ϵ_ode^2 + 1f-6
    ϵ_L1_sq   = ctx.ϵ_L1^2 + 1f-6

    # Calculate individual loss components
    L_ic   = loss_IC(ps, ctx)
    L_data = loss_Data(ps, ctx)
    L_ode  = loss_ODE(ps, ctx, architecture, ODE_param_constructor)
    L_L1   = l1_penalty(ps)
    L_bnd  = param_bound_penalty(ps, ctx)

    # Weighted Sum
    return 0.5 * (
        L_ic / ϵ_ic_sq +
        L_data / ϵ_Data_sq +
        L_ode / ϵ_ode_sq +
        L_L1 / ϵ_L1_sq +
        L_bnd # Bound penalty is usually hard-constrained or scaled internally
    )
end
