#
# sweep_config.jl
#
# Single source of truth for the hyperparameter grid and shared constants
# used by both the sweep driver (run_sweep.sh) and each worker
# (run_single_baccam.jl).  Changing the grid here automatically propagates
# to every worker — no risk of index mismatches.
#

using LinearAlgebra
BLAS.set_num_threads(1)

include(joinpath(pwd(), "Src", "Fitting.jl"))

# ── Hyperparameter grid ──────────────────────────────────
# Each ϵ is an inverse-weight: larger ⟹ weaker penalty.
const HP_GRID = [
    HyperParams(ic, ode, dat, l1s, l1g)
    for ic  in Float32[0.1, 0.5, 1.0],
        ode in Float32[0.01, 0.1, 0.5],
        dat in Float32[0.01, 0.04, 0.1],
        l1s in Float32[0.02, 0.5, 2.0],
        l1g in Float32[0.02, 0.5, 2.0]
] |> vec

const VALID_CONFIGS = ["A_5x24", "B_10x12", "C_20x6"]

# ── Training constants ────────────────────────────────────
const MAXITERS_S1 = 50_000
const MAXITERS_S2 = 120_000
const OPT_S1 = OptimizationOptimisers.Adam(1f-4)
const OPT_S2 = OptimizationOptimisers.Adam(1f-4)

# ── Initial ODE parameter guesses and bounds ──────────────
const ODE_PAR_INIT   = (β = 1f-4, δ = 1f0, c = 1f0)
const ODE_PAR_BOUNDS = (
    β = Float32[0, 1f-2],
    δ = Float32[0, 20],
    c = Float32[0, 20],
)

# ── State transform (log₁₀, floor = 1e-6) ────────────────
const STATE_TRANSFORM = LogTransform(
    10f0,
    Float32[1f-6, 1f-6, 1f-6],
    Bool[true, true, true],
)

# ── Initial conditions (from Baccam et al. Table 2) ──────
const U0 = Float32[4.0f8, 0.0f0, 9.3f-2]
