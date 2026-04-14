#
# sweep_config.jl — Hardwired IC architecture sweep
#
# Single source of truth for the hyperparameter grid and shared
# constants.  Imported by every worker and by the metadata writer.
#
# The hardwired IC architecture (LuxMLPHardwiredIC) enforces y(0) = y₀
# structurally, so ϵ_ic has no effect on training.  The LHS samples
# only the 4 active dimensions: ϵ_ode, ϵ_Data, ϵ_L1_state, ϵ_L1_g.
#

using LinearAlgebra
BLAS.set_num_threads(1)

using HDF5

include(joinpath(pwd(), "Src", "Fitting.jl"))

# ── Hyperparameter sampling ───────────────────────────────

const USE_LHS  = true
const N_LHS    = 1
const LHS_SEED = 456       # different seed from the soft-IC sweep

# Per-axis bounds in log₁₀ space
const HP_LOG_BOUNDS = (
    ϵ_ode      = (-2.0f0, 0.0f0),   # 0.01  →  1.0
    ϵ_Data     = (-2.0f0, -0.5f0),  # 0.01  →  0.32
    ϵ_L1_state = (-2.0f0, 0.5f0),   # 0.01  →  3.2
    ϵ_L1_g     = (-2.0f0, 0.5f0),   # 0.01  →  3.2
)

"""
    latin_hypercube(rng, n, d) -> Matrix{Float32}

n × d Latin Hypercube sample in [0,1]^d.
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

Map n LHS samples from [0,1]^4 into physical ϵ values.
ϵ_ic is fixed at 1.0 (structurally unused by LuxMLPHardwiredIC).
"""
function build_hp_grid_lhs(rng, n, bounds)
    raw = latin_hypercube(rng, n, 4)
    to_phys(u, lo, hi) = Float32(10^(lo + u * (hi - lo)))

    return [
        HyperParams(
            1.0f0,                                      # ϵ_ic — unused
            to_phys(raw[i, 1], bounds.ϵ_ode...),
            to_phys(raw[i, 2], bounds.ϵ_Data...),
            to_phys(raw[i, 3], bounds.ϵ_L1_state...),
            to_phys(raw[i, 4], bounds.ϵ_L1_g...),
        )
        for i in 1:n
    ]
end

const HP_GRID = build_hp_grid_lhs(MersenneTwister(LHS_SEED), N_LHS, HP_LOG_BOUNDS)

const VALID_CONFIGS = ["B_10x12"]

# ── Training constants ────────────────────────────────────

const MAXITERS_S1 = 50_000
const MAXITERS_S2 = 120_000
const OPT_S1 = OptimizationOptimisers.Adam(1f-4)
const OPT_S2 = OptimizationOptimisers.Adam(1f-4)

# ── ODE parameter guesses and bounds ─────────────────────

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

# ── Initial conditions (Baccam et al. Table 2) ───────────

const U0 = Float32[4.0f8, 0.0f0, 9.3f-2]

# ── Trajectory evaluation grid ────────────────────────────

const N_TRAJ_EVAL = 300
