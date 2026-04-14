#!/usr/bin/env julia
#
# generate_sweep_data.jl — Hardwired IC sweep
#
# Produces the B_10x12 training CSV (10 timepoints × 12 mice).
# Each row is a single mouse observation with log-normal noise.
#
# Usage:
#   julia --project=. generate_sweep_data.jl [output_dir]

using Pkg; Pkg.activate(".")

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
    Y_true = Float32.(sol_array[:, idx])

    # Log-normal multiplicative noise, mean-unbiased via -0.5σ² shift
    ε = randn(rng, Float32, 3, n_tp, n_mice)
    Y_noisy = clamp.(
        Y_true .* exp.(noise_σ .* ε .- 0.5f0 * noise_σ^2),
        0f0, Inf32
    )
    return (t_obs = t_obs, Y_true = Y_true, Y_noisy = Y_noisy,
            n_tp = n_tp, n_mice = n_mice)
end

rng = MersenneTwister(42)
cfg = generate_config(rng, sol_array, t_fine, 10, 12, 0.5f0)

# ── Write to disk ─────────────────────────────────────────

outdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(pwd(), "Results")
mkpath(outdir)

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
path = joinpath(outdir, "Baccam_B_10x12.csv")
CSV.write(path, df)
n_obs = cfg.n_tp * cfg.n_mice
println("Wrote $path  ($(cfg.n_tp) tp × $(cfg.n_mice) mice = $n_obs observations)")
println("\nData generation complete.")
