#!/usr/bin/env julia
#
# generate_sweep_data.jl
#
# Run ONCE before the sweep.  Produces per-config CSVs where each row
# is a single mouse observation (not a per-timepoint mean), preserving
# the replicate structure that distinguishes the three sampling designs.
#
# Output format per CSV:
#   t, T_obs, I_obs, V_obs, mouse_id, T_true, I_true, V_true
#
# The true (noiseless) values are included for downstream plotting but
# are NOT used during training.
#
# Usage:
#   julia --project=. generate_sweep_data.jl

using Pkg; Pkg.activate(".")

# using DifferentialEquations
using OrdinaryDiffEq
using Statistics, Random
using CSV, DataFrames

# ── ODE definition ────────────────────────────────────────

function Baccam_model!(du, u, p, t)
    T, I, V = u
    du[1] = -p.β * T * V
    du[2] =  p.β * T * V - p.δ * I
    du[3] =  p.p * I     - p.c * V
end

true_params = (β = 2.7f-5, δ = 4.0f0, p = 1.2f-2, c = 3.0f0)
u0     = Float32[4.0f8, 0.0f0, 9.3f-2]
t_span = (0.0f0, 8.0f0)
t_fine = collect(range(t_span[1], t_span[2], length = 2001))

sol = solve(
    ODEProblem(Baccam_model!, u0, t_span, true_params),
    Tsit5(); saveat = t_fine
)
sol_array = Array(sol)

# ── Noisy sampling ────────────────────────────────────────

function generate_config(rng, sol_array, t_fine, n_tp, n_mice, noise_σ)
    idx    = round.(Int, range(1, length(t_fine), length = n_tp))
    t_obs  = Float32.(t_fine[idx])
    Y_true = Float32.(sol_array[:, idx])                     # 3 × n_tp
    ε      = randn(rng, Float32, 3, n_tp, n_mice)
    Y_noisy = clamp.(
        Y_true .* exp.(noise_σ .* ε .- 0.5f0 * noise_σ^2),
        0f0, Inf32
    )                                                         # 3 × n_tp × n_mice
    return (t_obs = t_obs, Y_true = Y_true, Y_noisy = Y_noisy,
            n_tp = n_tp, n_mice = n_mice)
end

rng = MersenneTwister(42)
configs = Dict(
    "A_5x24"  => generate_config(rng, sol_array, t_fine, 5,  24, 0.5f0),
    "B_10x12" => generate_config(rng, sol_array, t_fine, 10, 12, 0.5f0),
    "C_20x6"  => generate_config(rng, sol_array, t_fine, 20,  6, 0.5f0),
)

# ── Write to disk ─────────────────────────────────────────
# One row per (timepoint, mouse) pair — this is the rawest usable
# form.  The worker script reshapes it into the 3 × N matrix that
# train_fixed_hyper expects.

outdir = joinpath(pwd(), "Results")
mkpath(outdir)

for (label, cfg) in configs
    rows = NamedTuple[]
    for m in 1:cfg.n_mice
        for tp in 1:cfg.n_tp
            push!(rows, (
                t      = cfg.t_obs[tp],
                T_obs  = cfg.Y_noisy[1, tp, m],
                I_obs  = cfg.Y_noisy[2, tp, m],
                V_obs  = cfg.Y_noisy[3, tp, m],
                mouse  = m,
                T_true = cfg.Y_true[1, tp],
                I_true = cfg.Y_true[2, tp],
                V_true = cfg.Y_true[3, tp],
            ))
        end
    end
    df = DataFrame(rows)
    path = joinpath(outdir, "Baccam_$(label).csv")
    CSV.write(path, df)
    n_obs = cfg.n_tp * cfg.n_mice
    println("Wrote $path  ($(cfg.n_tp) tp × $(cfg.n_mice) mice = $n_obs observations)")
end

println("\nData generation complete.")
