#=-------------------------------------------------------------------------------
Predictors.jl

Builder functions that produce closures satisfying the three predictor contracts:

  predict_state(ps, t_grid) → (n_states × B)   transformed coordinates
  predict_deriv(ps, t_grid) → (n_states × B)   time derivatives of state
  predict_g(ps, t_grid)     → (n_g × B)        learned unknown physics terms

Each builder captures architecture-specific internals (MLP structs, static
states, transforms, initial conditions) at construction time.  The returned
closures are opaque to the rest of the pipeline — loss functions and training
loops call them without knowing which architecture produced them.

Dispatch on `AbstractArchitectureMode` subtypes selects the right combination
of builders via `build_predictors`.
-------------------------------------------------------------------------------=#


#=-------------------------------------------------------------------------------

Architecture Mode Tags

These are lightweight structs used purely for dispatch purposes. Configuration
details specific to the modes are packed into these structs as fields.

-------------------------------------------------------------------------------=#


using Lux, NNlib
using ForwardDiff

abstract type AbstractArchitectureMode end

"""
LuxMLP mode is the current default mode for default AI-Aristotle like
behaviour. The state network and the g network are both Neural Networks.

Fields
- 'fd_step': step size for central difference derivative estimation
"""
Base.@kwdef struct LuxMLP <: AbstractArchitectureMode
    # I don't particularly like using finite diff but Automatic Differentiation
    # in Julia has been finicky
    fd_step::Float32 = 1f-03
end

"""
Lux MLP with hard-wired IC: `ŷ(t) = y₀ + t · MLP(t, θ)`. This aims to become the
default method if it tests well.

The IC is satisfied structurally for any parameter values, so
`loss_IC` is skipped during training.

Fields
- `fd_step`: step size for central-difference derivative estimation
"""
Base.@kwdef struct LuxMLPHardwiredIC <: AbstractArchitectureMode
    fd_step::Float32 = 1f-03
end


#=-------------------------------------------------------------------------------

Individual builder functions

Each returns a closure based on the contract signature outlined above. These
should be accessed by build_predictors only.

-------------------------------------------------------------------------------=#

#----State Predictors------------------------------------------------------------

"""
    make_state_predictor_loose_ic(mlp, st) -> (ps, t_grid) -> states

Soft-IC constrained. Lux MLPs assume that the st (state) never updates.
"""
function make_state_predictor_loose_ic(mlp, st)
    function predict(ps, t_grid::AbstractMatrix) #why abstract matrix tho
        return first(mlp(t_grid, ps.StateMLP, st))
    end
    return predict
end

"""
   make_state_predictor_hardwired_ic(mlp, st, y0_transformed) -> (ps, t_grid) -> states

Hard-wired IC variant.  Returns `y₀ + t · MLP(t, θ)` so that
`predict(ps, [0]') == y₀` identically for any `ps`.

`y0_transformed` is a column vector in the network's coordinate space
(i.e., already passed through `forward_state(transform, y0_raw)`).
The MLP learns the departure from the IC, scaled by time.
"""
function make_state_predictor_hardwired_ic(mlp, st, y0_transformed::AbstractVector)
    # y0 has is made to be an (n, ) vector that natively broadcasts over the
    # (n,B) matrix from an NN
    function predict(ps, t_grid::AbstractMatrix)
        raw_nn = first(mlp(t_grid, ps.StateMLP, st))
        return y0_transformed .+ t_grid .* raw_nn
    end
    return predict
end


#----Derivative Predictors-------------------------------------------------------

# The goal of a prepackaged deriv predictor is to provide the LHS in the ODE
# Loss equation, which is comparing the f(y, p) + g(t, θ). 

"""
   make_deriv_predictor_fd(state_predictor; h=1f-3) -> (ps, t_grid) -> dstate_dt

Central-difference approximation of the state predictor's time derivative.

Works for soft and hard-constrained architectures since it only calls the
predictor at perturbed time points.  Assumes pointwise independence — each
column of `t_grid` is an independent query.
"""
function make_deriv_predictor_fd(state_predictor; h::Float32 = 1f-3)
    function predict_deriv(ps,t_grid::AbstractMatrix)
        t_plus = t_grid .+ h
        t_minus = t_grid .- h
        return (state_predictor(ps, t_plus) .- state_predictor(ps, t_minus)) ./ (2f0 * h)
    end
    return predict_deriv
end

#-----Physics Residual (g) Predictors--------------------------------------------

"""
   make_g_predictor(g_mlp, st_g) -> (ps, t_grid) -> g_values

Standard `t → g` mapping.  The g-network is independent of
the state predictor and only receives time as input.
"""
function make_g_predictor(g_mlp, st_g)
    function predict(ps, t_grid::AbstractMatrix)
        return first(g_mlp(t_grid, ps.gMLP, st_g))
    end
    return predict
end


#=-------------------------------------------------------------------------------

Top Level Dispatchers

Dispatches on the architecture mode tag to assemble the correct combination of
builders. Returns a NamedTuple of three closures plus a flag (skip_IC loss?)

-------------------------------------------------------------------------------=#

"""
   build_predictors(mode, mlp, st, g_mlp, st_g, y0_transformed)
        -> (predict_state, predict_deriv, predict_g, skip_ic_loss)

Assemble predictor closures for the given architecture mode.

`skip_ic_loss` is a Bool that tells the loss assembly whether the
IC is structurally enforced (true) or needs a soft penalty (false).

"""
function build_predictors(ArchMode::LuxMLP,
                          mlp, st,
                          g_mlp, st_g,
                          y0_transformed)
    pred_state = make_state_predictor_loose_ic(mlp, st)
    pred_deriv = make_deriv_predictor_fd(pred_state; h = ArchMode.fd_step)
    pred_g = make_g_predictor(g_mlp, st_g)
    return (predict_state = pred_state,
            predict_deriv = pred_deriv,
            predict_g     = pred_g,
            skip_ic_loss  = false)
end

function build_predictors(ArchMode::LuxMLPHardwiredIC,
                          mlp, st,
                          g_mlp, st_g,
                          y0_transformed) 
    pred_state = make_state_predictor_hardwired_ic(mlp, st, y0_transformed)
    pred_deriv = make_deriv_predictor_fd(pred_state; h = ArchMode.fd_step)
    pred_g = make_g_predictor(g_mlp, st_g)
    return (predict_state = pred_state,
            predict_deriv = pred_deriv,
            predict_g     = pred_g,
            skip_ic_loss  = true)
end


#=-------------------------------------------------------------------------------

Component initialisation

Dispatches on AbstractArchitectureMode to build networks and pack raw
parameters.  The architecture mode controls what gets built (which networks,
what layout), therefore the initialization has to be architecture specific.

-------------------------------------------------------------------------------=#

"""
    initialize_components(mode, rng, data, build_state_mlp, build_g_mlp)
        -> (State_MLP, st_StateMLP, g_MLP, st_gMLP, raw_ps)

Create networks, run `Lux.setup`, and return the raw parameter
NamedTuple.

The returned `raw_ps` is a NamedTuple with fields `StateMLP`,
`gMLP`, and `ODE_par`.  Future architecture modes may return
a different layout.
"""
function initialize_components(
    ::Union{LuxMLP, LuxMLPHardwiredIC},
    rng,
    data,
    build_state_mlp_fn::Function,
    build_g_mlp_fn::Function,
)
    State_MLP = build_state_mlp_fn()
    ps_StateMLP, st_StateMLP = Lux.setup(rng, State_MLP)

    g_MLP = build_g_mlp_fn()
    ps_gMLP, st_gMLP = Lux.setup(rng, g_MLP)

    # Raw parameter NamedTuple — no hyper entries yet
    raw_ps = (
        StateMLP = ps_StateMLP,
        gMLP     = ps_gMLP,
        ODE_par  = data.ODE_par_init,
    )

    return State_MLP, st_StateMLP, g_MLP, st_gMLP, raw_ps
end



















