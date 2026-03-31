#!/usr/bin/env julia
#
# run_single_baccam.jl
#
# Trains one (config, hyperparameter) combination and writes the result
# as a single-row CSV.  Called by GNU Parallel:
#
#   julia --project=. run_single_baccam.jl <config_key> <hp_index>
#
# Prerequisites:
#   1. Run generate_sweep_data.jl once to create the training CSVs.
#   2. sweep_config.jl must be in the project root (defines HP grid
#      and shared constants so every worker uses the same grid).

using Pkg; Pkg.activate(".")

using ComponentArrays
using Lux, NNlib
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ForwardDiff
using Statistics, Random
using CSV, DataFrames
using Printf

# ── Shared config (HP grid, transforms, training constants) ──
include(joinpath(pwd(),
                 "Experiments", "HyperParameter-sweep-on-Baccam", "sweep_config.jl"))
# ══════════════════════════════════════════════════════════
# 1. Parse CLI arguments
# ══════════════════════════════════════════════════════════

if length(ARGS) < 2
    error("Usage: julia run_single_baccam.jl <config_key> <hp_index>\n" *
          "  config_key : A_5x24 | B_10x12 | C_20x6\n" *
          "  hp_index   : 1..$(length(HP_GRID))")
end

const CFG_KEY = ARGS[1]
const HP_IDX  = parse(Int, ARGS[2])

CFG_KEY in VALID_CONFIGS || error("Unknown config '$CFG_KEY'. Use one of $VALID_CONFIGS.")
HP_IDX in eachindex(HP_GRID) || error("hp_index $HP_IDX out of range (grid has $(length(HP_GRID)) entries).")

hp = HP_GRID[HP_IDX]

# ── Output path with idempotency guard ────────────────────
const OUTDIR  = joinpath(pwd(), "Results", "sweep_rows")
mkpath(OUTDIR)
const OUTFILE = joinpath(OUTDIR, "row_$(CFG_KEY)_hp$(HP_IDX).csv")

if isfile(OUTFILE) && filesize(OUTFILE) > 0
    println("  [SKIP] $OUTFILE already exists — exiting.")
    exit(0)
end

# ══════════════════════════════════════════════════════════
# 2. Load pre-generated training data from disk
#    (created by generate_sweep_data.jl — never regenerated)
#
#    The CSV has one row per (timepoint, mouse) pair.  We unpack
#    every individual mouse observation so the optimizer sees the
#    full replicate scatter — this is what makes Config A (few
#    timepoints, dense replicates) genuinely different from
#    Config C (many timepoints, sparse replicates).
# ══════════════════════════════════════════════════════════

csv_path = joinpath(pwd(), "Results", "Baccam_$(CFG_KEY).csv")
isfile(csv_path) || error("Training data not found at $csv_path. " *
                          "Run generate_sweep_data.jl first.")

df = CSV.read(csv_path, DataFrame)

# Each row is one mouse at one timepoint.  The training matrix has
# one column per observation: for Config A that's 5×24 = 120 columns,
# for Config C it's 20×6 = 120 columns — same budget, different structure.
t_all = Float32.(df.t)
Y_all = permutedims(Float32.(Matrix(df[:, [:T_obs, :I_obs, :V_obs]])))  # 3 × N_obs

t_train    = permutedims(t_all)                           # 1 × N_obs
t_dense    = permutedims(Float32.(range(
    minimum(t_all), maximum(t_all); length = 1000)))      # 1 × 1000
t_span_vec = Float32[minimum(t_all), maximum(t_all)]

data = (
    t_train        = t_train,
    Y_train        = Y_all,
    Y_train_std    = ones(Float32, 3, 1),
    t_dense        = t_dense,
    t_span         = t_span_vec,
    y0_init        = U0,
    ODE_par_init   = ODE_PAR_INIT,
    ODE_par_bounds = ODE_PAR_BOUNDS,
)

# ══════════════════════════════════════════════════════════
# 3. Architecture and MLP builders
# ══════════════════════════════════════════════════════════

struct BaccamParams{T<:AbstractFloat}
    β::T; δ::T; c::T
end
BaccamParams(nt::ComponentVector) = BaccamParams(nt.β, nt.δ, nt.c)

function BaccamArchitectureRaw(y::AbstractMatrix{<:Real},
                               g::AbstractMatrix{<:Real},
                               p::BaccamParams)
    T, I, V = y[1,:], y[2,:], y[3,:]
    g_T, g_V = g[1,:], g[2,:]
    dT = g_T
    dI = p.β .* T .* V .- p.δ .* I
    dV = g_V .- p.c .* V
    return permutedims(hcat(dT, dI, dV))
end

build_state_mlp() = Lux.Chain(
    Lux.Dense(1, 8, tanh_fast),
    Lux.Dense(8, 6, tanh_fast),
    Lux.Dense(6, 3),
)

build_g_mlp() = Lux.Chain(
    Lux.Dense(1, 6, tanh_fast),
    Lux.Dense(6, 4, tanh_fast),
    Lux.Dense(4, 2),
)

# ══════════════════════════════════════════════════════════
# 4. Callback — early stopping, minimal stdout
# ══════════════════════════════════════════════════════════

function make_sweep_callback(label; patience=600, min_delta=1f-7)
    best      = Ref(Inf32)
    stall_cnt = Ref(0)

    function cb(state, l)
        if state.iter % 50_000 == 0
            @printf("  [%s] iter=%d\n", label, state.iter)
        end

        state.iter % 50 != 0 && return false
        ctx = state.p
        ctx isa PINNCtxStage2 || return false

        ps    = state.u
        d_mse = loss_Data(ps, ctx)
        o_mse = loss_ODE(ps, ctx, BaccamArchitectureRaw, BaccamParams)
        combined = d_mse + o_mse

        if combined < best[] - min_delta
            best[]      = combined
            stall_cnt[] = 0
        else
            stall_cnt[] += 1
        end

        if stall_cnt[] >= patience
            @printf("  [%s] Early stop at iter %d\n", label, state.iter)
            return true
        end
        return false
    end
    return cb
end

# ══════════════════════════════════════════════════════════
# 5. Train
# ══════════════════════════════════════════════════════════

run_label = "$(CFG_KEY)_hp$(HP_IDX)"
println("Starting $run_label  (ϵ_ic=$(hp.ϵ_ic), ϵ_ode=$(hp.ϵ_ode), " *
        "ϵ_Data=$(hp.ϵ_Data), ϵ_L1s=$(hp.ϵ_L1_state), ϵ_L1g=$(hp.ϵ_L1_g))")

cb = make_sweep_callback(run_label; patience = 600, min_delta = 1f-7)

# Time the training, then handle success/failure outside @elapsed
# so there are no scoping issues with the result variable.
t_start = time()

res = try
    train_fixed_hyper(
        data, BaccamArchitectureRaw, BaccamParams, hp;
        transform            = STATE_TRANSFORM,
        user_loss_functions   = Function[],
        maxiters_stage1       = MAXITERS_S1,
        maxiters_stage2       = MAXITERS_S2,
        Opt_alg_stage1        = OPT_S1,
        Opt_alg_stage2        = OPT_S2,
        callback_function     = cb,
    )
catch e
    @warn "Run $run_label failed" exception=(e, catch_backtrace())
    nothing
end

wall_t = time() - t_start

# ══════════════════════════════════════════════════════════
# 6. Write result row
# ══════════════════════════════════════════════════════════

if isnothing(res)
    CSV.write(OUTFILE, DataFrame(
        config = CFG_KEY, hp_idx = HP_IDX,
        eps_ic = hp.ϵ_ic, eps_ode = hp.ϵ_ode, eps_Data = hp.ϵ_Data,
        eps_L1_state = hp.ϵ_L1_state, eps_L1_g = hp.ϵ_L1_g,
        data_mse = NaN, ode_mse = NaN,
        beta_recovered = NaN, delta_recovered = NaN, c_recovered = NaN,
        wall_s = Float32(wall_t), converged = false,
    ))
    exit(1)
end

CSV.write(OUTFILE, DataFrame(
    config          = CFG_KEY,
    hp_idx          = HP_IDX,
    eps_ic          = hp.ϵ_ic,
    eps_ode         = hp.ϵ_ode,
    eps_Data        = hp.ϵ_Data,
    eps_L1_state    = hp.ϵ_L1_state,
    eps_L1_g        = hp.ϵ_L1_g,
    data_mse        = res.metrics.data_mse,
    ode_mse         = res.metrics.ode_mse,
    beta_recovered  = res.params.ODE_par.β,
    delta_recovered = res.params.ODE_par.δ,
    c_recovered     = res.params.ODE_par.c,
    wall_s          = Float32(wall_t),
    converged       = true,
))

@printf("Done %s — data=%.3e  ode=%.3e  β=%.2e  δ=%.3f  c=%.3f  (%.0fs)\n",
        run_label, res.metrics.data_mse, res.metrics.ode_mse,
        res.params.ODE_par.β, res.params.ODE_par.δ, res.params.ODE_par.c, wall_t)
