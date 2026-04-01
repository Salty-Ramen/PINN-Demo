#!/usr/bin/env julia
#
# Baccam_sweep_analysis.jl
#
# Reads the CSV produced by Baccam_hyperparam_sweep.jl and generates
# diagnostic plots to identify the best hyperparameter configurations.
#
# Usage:
#   julia Baccam_sweep_analysis.jl [path/to/Baccam_hp_sweep.csv]

using Pkg; Pkg.activate(".")

using CSV, DataFrames
using CairoMakie, Makie
using Printf, Statistics

# ══════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════

csv_path = length(ARGS) >= 1 ? ARGS[1] :
           joinpath(pwd(), "Results", "Baccam_hp_sweep.csv")

df_raw = CSV.read(csv_path, DataFrame)

# Drop failed runs and work with finite losses only
df = filter(r -> r.converged && isfinite(r.data_mse) && isfinite(r.ode_mse), df_raw)
println("Loaded $(nrow(df_raw)) rows, $(nrow(df)) converged with finite losses.\n")

# Normalise column names across both sweep scripts (the GNU Parallel version
# uses ascii names like eps_ode; the threaded version uses ϵ_ode)
function normalise_columns!(df)
    renames = Dict(
        "eps_ic" => "ϵ_ic", "eps_ode" => "ϵ_ode", "eps_Data" => "ϵ_Data",
        "eps_L1_state" => "ϵ_L1_state", "eps_L1_g" => "ϵ_L1_g",
        "beta_recovered" => "β_recovered", "delta_recovered" => "δ_recovered",
    )
    for (old, new) in renames
        if old in names(df) && !(new in names(df))
            rename!(df, old => new)
        end
    end
end
normalise_columns!(df)

# True parameter values for reference lines
TRUE_β = 2.7f-5
TRUE_δ = 4.0f0
TRUE_c = 3.0f0

config_keys  = sort(unique(df.config))
config_colors = Dict(
    "A_5x24"  => :royalblue,
    "B_10x12" => :darkorange,
    "C_20x6"  => :forestgreen,
)

# ══════════════════════════════════════════════════════════
# 2. Pareto front: Data MSE vs ODE MSE per config
#    This is the key plot — points in the lower-left corner
#    are the sweet spot (low data loss AND low ODE loss).
# ══════════════════════════════════════════════════════════

fig1 = Figure(size = (1000, 400), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)
    ax = Makie.Axis(fig1[1, col];
        title  = cfg,
        xlabel = "Data MSE",
        ylabel = col == 1 ? "ODE MSE" : "",
        xscale = log10, yscale = log10,
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
    )

    scatter!(ax, sub.data_mse, sub.ode_mse;
        color = config_colors[cfg], markersize = 6, alpha = 0.6)

    # Highlight the Pareto-optimal front (non-dominated points)
    pareto_mask = pareto_front_mask(sub.data_mse, sub.ode_mse)
    pareto_sub = sub[pareto_mask, :]
    scatter!(ax, pareto_sub.data_mse, pareto_sub.ode_mse;
        color = :red, markersize = 10, marker = :star5,
        label = "Pareto front")

    col == 1 && axislegend(ax; position = :rt, labelsize = 9)
end

Label(fig1[0, 1:length(config_keys)],
    "Data MSE vs ODE MSE — each dot is one HP combination";
    fontsize = 14, tellwidth = false)

save(joinpath(pwd(), "Results", "sweep_pareto.pdf"), fig1; px_per_unit = 600/72)
display(fig1)

# ══════════════════════════════════════════════════════════
# 3. Parameter recovery accuracy for Pareto-best runs
# ══════════════════════════════════════════════════════════

fig2 = Figure(size = (1000, 350), fontsize = 12)

param_info = [
    ("β", :β_recovered, TRUE_β),
    ("δ", :δ_recovered, TRUE_δ),
    ("c", :c_recovered, TRUE_c),
]

for (col, (name, sym, truth)) in enumerate(param_info)
    ax = Makie.Axis(fig2[1, col];
        title  = "Recovered $name (true = $truth)",
        xlabel = "Config",
        ylabel = col == 1 ? "Recovered value" : "",
        xgridvisible = false, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
    )

    # For each config, take the top-5 runs by ODE loss and show their
    # recovered parameter as a jittered strip + median marker
    for (i, cfg) in enumerate(config_keys)
        sub = sort(filter(r -> r.config == cfg, df), :ode_mse)
        top = first(sub, min(10, nrow(sub)))
        vals = top[!, sym]

        # Jittered x positions for visibility
        xs = fill(Float64(i), length(vals)) .+ 0.1 .* randn(length(vals))
        scatter!(ax, xs, vals;
            color = (config_colors[cfg], 0.6), markersize = 7)

        # Median of top runs
        med = median(vals)
        hlines!(ax, [med]; color = config_colors[cfg],
            linewidth = 2, linestyle = :solid)
    end

    # Ground truth as dashed line
    hlines!(ax, [truth]; color = :black, linewidth = 1.5, linestyle = :dash)

    ax.xticks = (1:length(config_keys), config_keys)
end

Label(fig2[0, 1:3],
    "Parameter recovery — top 10 runs by ODE loss per config";
    fontsize = 14, tellwidth = false)

save(joinpath(pwd(), "Results", "sweep_param_recovery.pdf"), fig2; px_per_unit = 600/72)
display(fig2)

# ══════════════════════════════════════════════════════════
# 4. Sensitivity heatmaps: which ϵ matter most?
#    For each config, pivot on pairs of hyperparameters and
#    show median ODE loss.
# ══════════════════════════════════════════════════════════

fig3 = Figure(size = (1200, 800), fontsize = 11)

# We focus on the two most impactful axes: ϵ_ode vs ϵ_Data
for (row, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)

    # ── Panel (row, 1): ϵ_ode vs ϵ_Data, colour = median ODE MSE ──
    ax1 = Makie.Axis(fig3[row, 1];
        title  = "$cfg — ODE loss by (ϵ_ode, ϵ_Data)",
        xlabel = "ϵ_ode", ylabel = "ϵ_Data",
        xscale = log10, yscale = log10,
    )
    sc1 = scatter!(ax1, sub.ϵ_ode, sub.ϵ_Data;
        color = log10.(sub.ode_mse),
        colormap = :viridis, markersize = 8)
    Colorbar(fig3[row, 2], sc1; label = "log₁₀(ODE MSE)")

    # ── Panel (row, 3): ϵ_L1_state vs ϵ_L1_g, colour = median ODE MSE ──
    ax2 = Makie.Axis(fig3[row, 3];
        title  = "$cfg — ODE loss by (ϵ_L1_state, ϵ_L1_g)",
        xlabel = "ϵ_L1_state", ylabel = "ϵ_L1_g",
        xscale = log10, yscale = log10,
    )
    sc2 = scatter!(ax2, sub.ϵ_L1_state, sub.ϵ_L1_g;
        color = log10.(sub.ode_mse),
        colormap = :viridis, markersize = 8)
    Colorbar(fig3[row, 4], sc2; label = "log₁₀(ODE MSE)")
end

save(joinpath(pwd(), "Results", "sweep_sensitivity.pdf"), fig3; px_per_unit = 600/72)
display(fig3)

# ══════════════════════════════════════════════════════════
# 5. Ranked table: best runs per config
# ══════════════════════════════════════════════════════════

println("="^90)
println("  Top 5 runs per configuration (sorted by ODE MSE)")
println("="^90)

for cfg in config_keys
    sub = sort(filter(r -> r.config == cfg, df), :ode_mse)
    top = first(sub, min(5, nrow(sub)))

    println("\n── $cfg ──")
    @printf("  %-6s  %-10s  %-10s  %-8s  %-8s  %-8s  %-10s  %-10s  %-10s\n",
            "hp_idx", "ode_mse", "data_mse", "ϵ_ode", "ϵ_Data", "ϵ_ic",
            "β_rec", "δ_rec", "c_rec")
    println("  ", "-"^84)
    for r in eachrow(top)
        @printf("  %-6d  %.3e  %.3e  %-8.3f  %-8.3f  %-8.3f  %.2e  %8.3f  %8.3f\n",
                r.hp_idx, r.ode_mse, r.data_mse, r.ϵ_ode, r.ϵ_Data, r.ϵ_ic,
                r.β_recovered, r.δ_recovered, r.c_recovered)
    end
end

# ══════════════════════════════════════════════════════════
# Utility: compute Pareto front mask (minimize both objectives)
# ══════════════════════════════════════════════════════════

"""
    pareto_front_mask(x, y) -> BitVector

Returns a mask where `true` indicates a non-dominated point
(no other point has both smaller x AND smaller y).
"""
function pareto_front_mask(x, y)
    n = length(x)
    dominated = falses(n)
    for i in 1:n
        for j in 1:n
            i == j && continue
            # j dominates i if j is ≤ on both objectives and < on at least one
            if x[j] <= x[i] && y[j] <= y[i] && (x[j] < x[i] || y[j] < y[i])
                dominated[i] = true
                break
            end
        end
    end
    return .!dominated
end
