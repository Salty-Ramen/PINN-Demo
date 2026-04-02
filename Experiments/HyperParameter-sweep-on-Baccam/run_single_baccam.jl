#!/usr/bin/env julia
#
# run_single_baccam.jl
#
# Trains one (config, hyperparameter) combination and writes the result
# as a per-worker HDF5 file containing both scalar metrics and the
# trained model's trajectory predictions.
#
# Called by GNU Parallel:
#   julia --project=. run_single_baccam.jl <config_key> <hp_index> <script_dir>
#
# Prerequisites:
#   1. Run generate_sweep_data.jl once to create the training CSVs.
#   2. sweep_config.jl must be alongside this script (defines HP grid
#      and shared constants so every worker uses the same grid).

using Pkg; Pkg.activate(".")

using ComponentArrays
using Lux, NNlib
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ForwardDiff
using Statistics, Random
using CSV, DataFrames
using HDF5
using Printf

# ── Shared config (HP grid, transforms, training constants) ──
# SCRIPT_DIR is passed as CLI arg so workers resolve paths relative
# to the experiment directory, not the project root.
const SCRIPT_DIR = length(ARGS) >= 3 ? ARGS[3] :
    joinpath(pwd(), "Experiments", "HyperParameter-sweep-on-Baccam")

include(joinpath(SCRIPT_DIR, "sweep_config.jl"))

# ══════════════════════════════════════════════════════════
# 1. Parse CLI arguments
# ══════════════════════════════════════════════════════════

if length(ARGS) < 2
    error("Usage: julia run_single_baccam.jl <config_key> <hp_index> [script_dir]\n" *
          "  config_key : A_5x24 | B_10x12 | C_20x6\n" *
          "  hp_index   : 1..$(length(HP_GRID))")
end

const CFG_KEY = ARGS[1]
const HP_IDX  = parse(Int, ARGS[2])

CFG_KEY in VALID_CONFIGS || error("Unknown config '$CFG_KEY'. Use one of $VALID_CONFIGS.")
HP_IDX in eachindex(HP_GRID) || error("hp_index $HP_IDX out of range (grid has $(length(HP_GRID)) entries).")

hp = HP_GRID[HP_IDX]

# ── All output lives under SCRIPT_DIR/Results ─────────────
const RESULTS_DIR = joinpath(SCRIPT_DIR, "Results")
const ROW_DIR     = joinpath(RESULTS_DIR, "sweep_rows")
mkpath(ROW_DIR)
const OUTFILE = joinpath(ROW_DIR, "row_$(CFG_KEY)_hp$(HP_IDX).h5")

# Idempotency: skip if this run already produced output
if isfile(OUTFILE) && filesize(OUTFILE) > 0
    println("  [SKIP] $OUTFILE already exists — exiting.")
    exit(0)
end

# ══════════════════════════════════════════════════════════
# 2. Load pre-generated training data
#    CSVs now live under SCRIPT_DIR/Results alongside sweep output
# ══════════════════════════════════════════════════════════

csv_path = joinpath(RESULTS_DIR, "Baccam_$(CFG_KEY).csv")
isfile(csv_path) || error("Training data not found at $csv_path. " *
                          "Run generate_sweep_data.jl first.")

df = CSV.read(csv_path, DataFrame)

t_all = Float32.(df.t)
Y_all = permutedims(Float32.(Matrix(df[:, [:T_obs, :I_obs, :V_obs]])))  # 3 × N_obs

t_train    = permutedims(t_all)
t_dense    = permutedims(Float32.(range(
    minimum(t_all), maximum(t_all); length = 1000)))
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

"""
    make_sweep_callback(label; ...) -> (callback, history)

Returns a callback closure AND a mutable history NamedTuple.
The history accumulates loss diagnostics at `log_every` intervals
and is written to HDF5 after training completes.
"""
function make_sweep_callback(label; patience=600, min_delta=1f-7, log_every=25)
    best      = Ref(Inf32)
    stall_cnt = Ref(0)

    # Mutable history vectors — returned to caller for HDF5 serialization
    hist = (
        iter     = Int32[],
        data_mse = Float32[],
        ode_mse  = Float32[],
    )

    function cb(state, l)
        if state.iter % 50_000 == 0
            @printf("  [%s] iter=%d\n", label, state.iter)
        end

        state.iter % log_every != 0 && return false

        ps  = state.u
        ctx = state.p

        d_mse = loss_Data(ps, ctx)

        # ODE loss is only meaningful in Stage 2
        o_mse = if ctx isa PINNCtxStage2
            loss_ODE(ps, ctx, BaccamArchitectureRaw, BaccamParams)
        else
            NaN32
        end

        push!(hist.iter,     Int32(state.iter))
        push!(hist.data_mse, d_mse)
        push!(hist.ode_mse,  o_mse)

        # Early stopping logic (Stage 2 only)
        if ctx isa PINNCtxStage2
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
        end

        return false
    end

    return cb, hist
end

# ══════════════════════════════════════════════════════════
# 5. Train
# ══════════════════════════════════════════════════════════

run_label = "$(CFG_KEY)_hp$(HP_IDX)"
println("Starting $run_label  (ϵ_ic=$(hp.ϵ_ic), ϵ_ode=$(hp.ϵ_ode), " *
        "ϵ_Data=$(hp.ϵ_Data), ϵ_L1s=$(hp.ϵ_L1_state), ϵ_L1g=$(hp.ϵ_L1_g))")

cb, loss_hist = make_sweep_callback(run_label; patience = 600, min_delta = 1f-7, log_every = 25)

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
# 6. Write result to per-worker HDF5
# ══════════════════════════════════════════════════════════

"""
    write_result_h5(path, cfg_key, hp_idx, hp, res, wall_t, transform, loss_hist)

Write scalar metrics, trajectory predictions, and the training loss
history to a single HDF5 file. Failed runs still get a file (with NaN
metrics, no trajectories, and whatever loss history was recorded before
failure) so the merge step can distinguish failure from "not yet started".
"""
function write_result_h5(path, cfg_key, hp_idx, hp, res, wall_t, transform, loss_hist)
    h5open(path, "w") do fid

        # ── /scalars: one dataset per metric field ────────
        g_sc = create_group(fid, "scalars")
        g_sc["config"]          = cfg_key
        g_sc["hp_idx"]          = Int32(hp_idx)
        g_sc["eps_ic"]          = hp.ϵ_ic
        g_sc["eps_ode"]         = hp.ϵ_ode
        g_sc["eps_Data"]        = hp.ϵ_Data
        g_sc["eps_L1_state"]    = hp.ϵ_L1_state
        g_sc["eps_L1_g"]        = hp.ϵ_L1_g
        g_sc["wall_s"]          = Float32(wall_t)

        if isnothing(res)
            g_sc["data_mse"]        = NaN32
            g_sc["ode_mse"]         = NaN32
            g_sc["beta_recovered"]  = NaN32
            g_sc["delta_recovered"] = NaN32
            g_sc["c_recovered"]     = NaN32
            g_sc["converged"]       = false
        else
            g_sc["data_mse"]        = res.metrics.data_mse
            g_sc["ode_mse"]         = res.metrics.ode_mse
            g_sc["beta_recovered"]  = res.params.ODE_par.β
            g_sc["delta_recovered"] = res.params.ODE_par.δ
            g_sc["c_recovered"]     = res.params.ODE_par.c
            g_sc["converged"]       = true
        end

        # ── /diagnostics: training loss trajectory ────────
        # Written for both successful and failed runs — partial
        # curves from failed runs are useful for diagnosing divergence.
        if !isempty(loss_hist.iter)
            g_diag = create_group(fid, "diagnostics")
            g_diag["iter"]     = loss_hist.iter
            g_diag["data_mse"] = loss_hist.data_mse
            g_diag["ode_mse"]  = loss_hist.ode_mse
        end

        # ── /trajectories: model predictions (successful runs only) ──
        isnothing(res) && return

        t_eval = permutedims(Float32.(range(
            minimum(data.t_train), maximum(data.t_train);
            length = N_TRAJ_EVAL)))

        smodel = state_model(res.params, res.ctx2)
        gmodel = g_model(res.params, res.ctx2)

        y_hat_transformed = smodel(t_eval)
        y_hat_raw = inverse_state(transform, y_hat_transformed)
        g_hat = gmodel(t_eval)

        g_tr = create_group(fid, "trajectories")
        g_tr["t_eval"]     = vec(t_eval)
        g_tr["state_pred"] = Array(y_hat_raw)
        g_tr["g_pred"]     = Array(g_hat)
    end
end

write_result_h5(OUTFILE, CFG_KEY, HP_IDX, hp, res, wall_t, STATE_TRANSFORM, loss_hist)

if isnothing(res)
    @printf("FAILED %s — wrote %s (%.0fs)\n", run_label, OUTFILE, wall_t)
    exit(1)
end

@printf("Done %s — data=%.3e  ode=%.3e  β=%.2e  δ=%.3f  c=%.3f  (%.0fs)\n",
        run_label, res.metrics.data_mse, res.metrics.ode_mse,
        res.params.ODE_par.β, res.params.ODE_par.δ, res.params.ODE_par.c, wall_t)
