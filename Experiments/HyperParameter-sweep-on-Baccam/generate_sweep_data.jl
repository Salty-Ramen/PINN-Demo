#!/usr/bin/env julia
#
# generate_sweep_data.jl
#
# Run ONCE before the sweep to produce the training CSVs that every
# worker reads.  This is the single source of truth for the synthetic
# data — workers never regenerate it themselves.
#
# Usage:
#   julia --project=. generate_sweep_data.jl

using Pkg; Pkg.activate(".")

using DifferentialEquations
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
    ε      = randn(rng, Float32, 3, n_tp, n_mice)
    Y_noisy = clamp.(Y_true .* exp.(noise_σ .* ε .- 0.5f0 * noise_σ^2), 0f0, Inf32)
    Y_mean  = dropdims(mean(Y_noisy; dims = 3); dims = 3)
    return (t_obs = t_obs, Y_true = Y_true, Y_mean = Y_mean,
            n_tp = n_tp, n_mice = n_mice)
end

rng = MersenneTwister(42)
configs = Dict(
    "A_5x24"  => generate_config(rng, sol_array, t_fine, 5,  24, 0.5f0),
    "B_10x12" => generate_config(rng, sol_array, t_fine, 10, 12, 0.5f0),
    "C_20x6"  => generate_config(rng, sol_array, t_fine, 20,  6, 0.5f0),
)

# ── Write to disk ─────────────────────────────────────────

outdir = joinpath(pwd(), "Results")
mkpath(outdir)

for (label, cfg) in configs
    df = DataFrame(
        t      = cfg.t_obs,
        T_true = cfg.Y_true[1, :],
        I_true = cfg.Y_true[2, :],
        V_true = cfg.Y_true[3, :],
        T_mean = cfg.Y_mean[1, :],
        I_mean = cfg.Y_mean[2, :],
        V_mean = cfg.Y_mean[3, :],
    )
    path = joinpath(outdir, "Baccam_$(label).csv")
    CSV.write(path, df)
    println("Wrote $path  ($(cfg.n_tp) timepoints × $(cfg.n_mice) mice)")
end

println("\nData generation complete. Run the sweep next.")
