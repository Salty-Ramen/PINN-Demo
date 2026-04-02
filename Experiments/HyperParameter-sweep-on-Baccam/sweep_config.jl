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

using HDF5

include(joinpath(pwd(), "Src", "Fitting.jl"))

# ── Hyperparameter sampling ───────────────────────────────
# Each ϵ is an inverse-weight: larger ⟹ weaker penalty.
#
# We sample in log-space so that the optimizer explores
# orders of magnitude uniformly (0.01 → 1.0 gets as many
# samples as 0.1 → 10.0).
#
# Set USE_LHS = true  for Latin Hypercube (better coverage per sample)
#     USE_LHS = false for Cartesian grid  (regular structure, heatmaps)

const USE_LHS    = true
const N_LHS      = 900      # samples per config when using LHS; 2 is good for testing
const LHS_SEED   = 123      # reproducible sampling

# Per-axis bounds in log₁₀ space
const HP_LOG_BOUNDS = (
    ϵ_ic       = (-1.5f0, 0.5f0),   # 10^-1.5 ≈ 0.03  to  10^0.5 ≈ 3.2
    ϵ_ode      = (-2.0f0, 0.0f0),   # 0.01  to  1.0
    ϵ_Data     = (-2.0f0, -0.5f0),  # 0.01  to  0.32
    ϵ_L1_state = (-2.0f0, 0.5f0),   # 0.01  to  3.2
    ϵ_L1_g     = (-2.0f0, 0.5f0),   # 0.01  to  3.2
)

"""
    latin_hypercube(rng, n, d) -> Matrix{Float32}

Generate an n × d Latin Hypercube sample in [0, 1]^d.
Each column (dimension) is divided into n equal strata with
exactly one sample per stratum, then columns are independently
shuffled.
"""
function latin_hypercube(rng, n::Int, d::Int)
    lhs = zeros(Float32, n, d)
    for j in 1:d
        perm = randperm(rng, n)
        for i in 1:n
            lhs[perm[i], j] = (i - 1 + rand(rng, Float32)) / n
        end
    end
    return lhs
end

"""
    build_hp_grid_lhs(rng, n, bounds) -> Vector{HyperParams}

Map n LHS samples from [0,1]^5 into physical ϵ values via
log-space interpolation within the given bounds.
"""
function build_hp_grid_lhs(rng, n, bounds)
    raw = latin_hypercube(rng, n, 5)

    to_phys(u, lo, hi) = Float32(10^(lo + u * (hi - lo)))

    return [
        HyperParams(
            to_phys(raw[i, 1], bounds.ϵ_ic...),
            to_phys(raw[i, 2], bounds.ϵ_ode...),
            to_phys(raw[i, 3], bounds.ϵ_Data...),
            to_phys(raw[i, 4], bounds.ϵ_L1_state...),
            to_phys(raw[i, 5], bounds.ϵ_L1_g...),
        )
        for i in 1:n
    ]
end

"""
    build_hp_grid_cartesian() -> Vector{HyperParams}

Full Cartesian product of fixed per-axis values.
"""
function build_hp_grid_cartesian()
    [
        HyperParams(ic, ode, dat, l1s, l1g)
        for ic  in Float32[0.1, 0.5, 1.0],
            ode in Float32[0.01, 0.1, 0.5],
            dat in Float32[0.01, 0.04, 0.1],
            l1s in Float32[0.02, 0.5, 2.0],
            l1g in Float32[0.02, 0.5, 2.0]
    ] |> vec
end

const HP_GRID = if USE_LHS
    build_hp_grid_lhs(MersenneTwister(LHS_SEED), N_LHS, HP_LOG_BOUNDS)
else
    build_hp_grid_cartesian()
end

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

# ── Trajectory evaluation grid (shared across workers) ────
# Workers evaluate the trained model on this grid and store
# the predictions in their per-run HDF5 file.
const N_TRAJ_EVAL = 300
