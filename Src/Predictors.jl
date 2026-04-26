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

Architecture is parameterised on two orthogonal axes:

  IC axis  — how y(0) = y₀ is enforced (SoftIC vs HardwiredIC).
  Deriv axis — how dŷ/dt is computed (HandFD, FiniteDiffJL, ForwardDiffAD).

Both axes are independent. `LuxMLP{IC, D}` is the combined tag that
`build_predictors` dispatches on.  IC dispatches govern `make_state_predictor`;
Deriv dispatches govern `make_deriv_predictor`; the two never see each other.
-------------------------------------------------------------------------------=#

using Lux, NNlib
using ForwardDiff
using FiniteDiff


#=-------------------------------------------------------------------------------

Architecture mode axes

Two orthogonal abstract type hierarchies for the IC and derivative concerns.
A concrete `LuxMLP{IC, D}` pairs one of each.

-------------------------------------------------------------------------------=#

abstract type AbstractArchitectureMode end

# ── IC axis ───────────────────────────────────────────────────────────────

abstract type AbstractICMode end

"""
Soft IC: the network learns the IC via the `loss_IC` penalty term.
"""
struct SoftIC <: AbstractICMode end

"""
Hard-wired IC: `ŷ(t) = y₀ + t · MLP(t, θ)`.  The IC is satisfied
structurally for any parameter values, so `loss_IC` is skipped.
"""
struct HardwiredIC <: AbstractICMode end


# ── Derivative axis ───────────────────────────────────────────────────────

abstract type AbstractDerivMode end

"""
Hand-rolled central-difference derivative with fixed step size.
Cheapest of the three; least accurate.

Field
- `h`: finite-difference step size
"""
Base.@kwdef struct HandFD <: AbstractDerivMode
    h::Float32 = 1f-3
end

"""
FiniteDiff.jl central-difference derivative.  Uses the package's
default step-size selection (Richardson-style heuristic), giving
better accuracy than `HandFD` at similar cost.
"""
struct FiniteDiffJL <: AbstractDerivMode end

"""
ForwardDiff.jl exact derivative via dual numbers.

Used inside an outer Zygote pass (the Optimization.jl gradient),
this composition is the "Path 1" recommended in Lux's nested-AD
manual: forward-mode for the inner derivative, reverse-mode for
the outer ∂L/∂θ.

The state predictor closure must internally route its MLP call
through a `StatefulLuxLayer` so that Lux's nested-AD switching
applies.  See `make_state_predictor` for that wiring.
"""
struct ForwardDiffAD <: AbstractDerivMode end


# ── Combined architecture tag ─────────────────────────────────────────────

"""
LuxMLP architecture, parameterised on IC and derivative modes.

Examples:
- `LuxMLP()` — soft IC, hand FD (matches the previous default)
- `LuxMLP(SoftIC(), ForwardDiffAD())` — soft IC, exact AD derivative
- `LuxMLP(HardwiredIC(), FiniteDiffJL())` — hardwired IC, FiniteDiff.jl

Fields are independent; `build_predictors` composes them.
"""
struct LuxMLP{IC<:AbstractICMode, D<:AbstractDerivMode} <: AbstractArchitectureMode
    ic::IC
    deriv::D
end

# Default constructor preserves previous behaviour: SoftIC + HandFD
LuxMLP() = LuxMLP(SoftIC(), HandFD())


#=-------------------------------------------------------------------------------

State predictors (IC-axis dispatch)

Each method takes the MLP, its stateless state, the transformed IC vector,
and returns a closure of contract `(ps, t_grid) → (n_states × B)`.

For the AD derivative path to work without dropping gradients, the closure
must call the MLP through a `StatefulLuxLayer` wrapper.  We do that
unconditionally — there's no cost when AD isn't being used, and it makes
the predictor identical regardless of which derivative method consumes it.

Why StatefulLuxLayer:
  When ForwardDiff.derivative is invoked inside a Zygote.gradient call
  (which is what happens when loss_ODE differentiates the predictor and
  the outer optimiser then differentiates the loss), Lux's nested-AD
  switching needs a hook to detect the inner pass and substitute its
  custom rule.  StatefulLuxLayer provides that hook by wrapping the
  forward call in a function the rule can recognise.  Without it,
  ForwardDiff inside Zygote silently drops the inner gradient.
  See: https://lux.csail.mit.edu/stable/manual/nested_autodiff
-------------------------------------------------------------------------------=#

"""
    make_state_predictor(ic, mlp, st, y0_transformed) -> (ps, t_grid) -> states

Soft IC: returns `MLP(t, θ)` directly.  The IC is enforced via the
loss term, not structurally.
"""
function make_state_predictor(::SoftIC, mlp, st, y0_transformed::AbstractVector)
    smlp = Lux.StatefulLuxLayer{true}(mlp, nothing, st)
    function predict(ps, t_grid::AbstractMatrix)
        return smlp(t_grid, ps.StateMLP)
    end
    return predict
end

"""
    make_state_predictor(ic, mlp, st, y0_transformed) -> (ps, t_grid) -> states

Hard-wired IC: returns `y₀ + t · MLP(t, θ)`, which equals `y₀` exactly
at `t = 0` for any parameters.  The MLP learns the departure from the
IC, scaled by time.
"""
function make_state_predictor(::HardwiredIC, mlp, st, y0_transformed::AbstractVector)
    smlp = Lux.StatefulLuxLayer{true}(mlp, nothing, st)
    # y0 is an (n,) vector; broadcasts over the (n, B) MLP output
    function predict(ps, t_grid::AbstractMatrix)
        raw_nn = smlp(t_grid, ps.StateMLP)
        return y0_transformed .+ t_grid .* raw_nn
    end
    return predict
end


#=-------------------------------------------------------------------------------

Derivative predictors (Deriv-axis dispatch)

Each method takes a state predictor closure and returns a closure of
contract `(ps, t_grid) → (n_states × B)` giving dŷ/dt.

All three methods see *only* the predictor closure, never the underlying
MLP.  This decoupling means that adding a new IC mode requires no changes
to the derivative code, and vice versa.
-------------------------------------------------------------------------------=#

"""
    make_deriv_predictor(d, predict_state) -> (ps, t_grid) -> dstate_dt

Hand-rolled central difference: `(ŷ(t+h) - ŷ(t-h)) / (2h)`.
Two forward passes per derivative evaluation; truncation error O(h²).
"""
function make_deriv_predictor(d::HandFD, predict_state)
    h = d.h
    function predict_deriv(ps, t_grid::AbstractMatrix)
        return (predict_state(ps, t_grid .+ h) .- predict_state(ps, t_grid .- h)) ./ (2f0 * h)
    end
    return predict_deriv
end

"""
    make_deriv_predictor(d, predict_state) -> (ps, t_grid) -> dstate_dt

FiniteDiff.jl central-difference derivative with adaptive step size.
Per-column evaluation: each time point gets its own scalar derivative.

Uses `reduce(hcat, cols)` rather than mutation so Zygote can trace it.
"""
function make_deriv_predictor(::FiniteDiffJL, predict_state)
    function predict_deriv(ps, t_grid::AbstractMatrix)
        B = size(t_grid, 2)
        cols = map(1:B) do j
            tj = t_grid[1, j]
            FiniteDiff.finite_difference_derivative(
                t -> vec(predict_state(ps, reshape([t], 1, 1))),
                tj,
            )
        end
        return reduce(hcat, cols)
    end
    return predict_deriv
end

"""
    make_deriv_predictor(d, predict_state) -> (ps, t_grid) -> dstate_dt

ForwardDiff.jl exact derivative.  Per-column evaluation via
`ForwardDiff.derivative` over the scalar time input.

This composes correctly inside Zygote because the state predictor was
built with a `StatefulLuxLayer`, which triggers Lux's nested-AD rule
(see comments in `make_state_predictor`).  Without that wrapper, the
inner gradient would be silently dropped.
"""
function make_deriv_predictor(::ForwardDiffAD, predict_state)
    function predict_deriv(ps, t_grid::AbstractMatrix)
        B = size(t_grid, 2)
        cols = map(1:B) do j
            tj = t_grid[1, j]
            ForwardDiff.derivative(
                t -> vec(predict_state(ps, reshape([t], 1, 1))),
                tj,
            )
        end
        return reduce(hcat, cols)
    end
    return predict_deriv
end


#=-------------------------------------------------------------------------------

g predictor (no axis — single implementation)

The g-network is independent of the state predictor and only receives time
as input.  No IC or derivative dispatch needed.
-------------------------------------------------------------------------------=#

"""
    make_g_predictor(g_mlp, st_g) -> (ps, t_grid) -> g_values

Standard `t → g` mapping.  Wrapped in `StatefulLuxLayer` for symmetry
with the state predictor (cheap; no functional consequence).
"""
function make_g_predictor(g_mlp, st_g)
    sg = Lux.StatefulLuxLayer{true}(g_mlp, nothing, st_g)
    function predict(ps, t_grid::AbstractMatrix)
        return sg(t_grid, ps.gMLP)
    end
    return predict
end


#=-------------------------------------------------------------------------------

Top-level dispatcher

Single method dispatching on the combined `LuxMLP{IC, D}` tag.
Delegates to the IC and derivative dispatchers and assembles the
predictor bundle.

-------------------------------------------------------------------------------=#

"""
    build_predictors(mode, mlp, st, g_mlp, st_g, y0_transformed)
        -> (predict_state, predict_deriv, predict_g, skip_ic_loss)

Compose IC-specific state predictor with derivative-specific derivative
predictor.  `skip_ic_loss` is set to `true` for `HardwiredIC` so the
loss assembly drops the IC term.
"""
function build_predictors(mode::LuxMLP,
                          mlp, st,
                          g_mlp, st_g,
                          y0_transformed)
    pred_state = make_state_predictor(mode.ic, mlp, st, y0_transformed)
    pred_deriv = make_deriv_predictor(mode.deriv, pred_state)
    pred_g     = make_g_predictor(g_mlp, st_g)
    return (predict_state = pred_state,
            predict_deriv = pred_deriv,
            predict_g     = pred_g,
            skip_ic_loss  = mode.ic isa HardwiredIC)
end


#=-------------------------------------------------------------------------------

Component initialisation

Builds networks and packs raw parameters.  Initialisation depends only on
the MLP structure, not on the IC or derivative modes, so a single method
covers every `LuxMLP{IC, D}`.
-------------------------------------------------------------------------------=#

"""
    initialize_components(mode, rng, data, build_state_mlp, build_g_mlp)
        -> (State_MLP, st_StateMLP, g_MLP, st_gMLP, raw_ps)

Create networks, run `Lux.setup`, and return the raw parameter NamedTuple.

The returned `raw_ps` is a NamedTuple with fields `StateMLP`, `gMLP`,
and `ODE_par`.
"""
function initialize_components(
    ::LuxMLP,
    rng,
    data,
    build_state_mlp_fn::Function,
    build_g_mlp_fn::Function,
)
    State_MLP = build_state_mlp_fn()
    ps_StateMLP, st_StateMLP = Lux.setup(rng, State_MLP)

    g_MLP = build_g_mlp_fn()
    ps_gMLP, st_gMLP = Lux.setup(rng, g_MLP)

    raw_ps = (
        StateMLP = ps_StateMLP,
        gMLP     = ps_gMLP,
        ODE_par  = data.ODE_par_init,
    )

    return State_MLP, st_StateMLP, g_MLP, st_gMLP, raw_ps
end


#=-------------------------------------------------------------------------------

Backwards-compatibility alias

Old code used `LuxMLPHardwiredIC()` as a standalone tag.  Provide an
alias so existing scripts keep working without modification.
The alias is a function rather than `const` so it can take a derivative
mode kwarg if callers want it.
-------------------------------------------------------------------------------=#

"""
    LuxMLPHardwiredIC(; deriv = HandFD())

Backwards-compatibility shim: equivalent to `LuxMLP(HardwiredIC(), deriv)`.
"""
LuxMLPHardwiredIC(; deriv::AbstractDerivMode = HandFD()) = LuxMLP(HardwiredIC(), deriv)
