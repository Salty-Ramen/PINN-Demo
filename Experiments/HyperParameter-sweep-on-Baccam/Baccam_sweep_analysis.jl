#!/usr/bin/env julia
#
# Baccam_sweep_analysis.jl
#
# Merges per-worker HDF5 files from the sweep into a single master HDF5,
# then generates diagnostic plots. Replaces the old CSV-based workflow.
#
# Usage:
#   julia --project=. Baccam_sweep_analysis.jl [path/to/experiment/Results]

using Pkg; Pkg.activate(".")

using HDF5
using CSV, DataFrames
using CairoMakie, Makie
using Printf, Statistics
using OrdinaryDiffEq

# ══════════════════════════════════════════════════════════
# 0. Resolve experiment results directory
# ══════════════════════════════════════════════════════════

const RESULTS_DIR = length(ARGS) >= 1 ? ARGS[1] :
    joinpath(pwd(), "Experiments", "HyperParameter-sweep-on-Baccam", "Results")

const ROW_DIR    = joinpath(RESULTS_DIR, "sweep_rows")
const MASTER_H5  = joinpath(RESULTS_DIR, "Baccam_hp_sweep.h5")

# ══════════════════════════════════════════════════════════
# 1. Merge per-worker HDF5 → master HDF5
#
# Each worker file has /scalars (flat metrics) and optionally
# /trajectories (model predictions). We aggregate scalars into
# a columnar layout and copy trajectories under a per-run group.
# ══════════════════════════════════════════════════════════

"""
    merge_sweep_rows(row_dir, master_path, results_dir) -> DataFrame

Read every `row_*.h5` in `row_dir`, merge scalars into a master HDF5
at `master_path`, and return the combined DataFrame for plotting.
Trajectory and diagnostics groups are copied verbatim under per-run keys.
Sweep metadata is copied from `results_dir/sweep_metadata.h5` if present.
"""
function merge_sweep_rows(row_dir, master_path, results_dir)
    h5_files = filter(f -> endswith(f, ".h5"), readdir(row_dir; join=true))
    isempty(h5_files) && error("No .h5 files found in $row_dir")

    println("Found $(length(h5_files)) worker files to merge.")

    scalar_keys = [
        "config", "hp_idx",
        "eps_ic", "eps_ode", "eps_Data", "eps_L1_state", "eps_L1_g",
        "data_mse", "ode_mse",
        "beta_recovered", "delta_recovered", "c_recovered",
        "wall_s", "converged",
    ]

    rows = Dict{String,Any}[]

    h5open(master_path, "w") do master
        g_traj = create_group(master, "trajectories")
        g_diag = create_group(master, "diagnostics")

        for fpath in h5_files
            row = Dict{String,Any}()
            h5open(fpath, "r") do fid
                # ── Scalars ───────────────────────────────
                g_sc = fid["scalars"]
                for k in scalar_keys
                    row[k] = read(g_sc, k)
                end

                run_label = "$(row["config"])_hp$(row["hp_idx"])"

                # ── Trajectories (successful runs only) ───
                if haskey(fid, "trajectories")
                    g_run = create_group(g_traj, run_label)
                    g_src = fid["trajectories"]
                    for dset_name in keys(g_src)
                        g_run[dset_name] = read(g_src, dset_name)
                    end
                end

                # ── Diagnostics (loss curves, all runs) ───
                if haskey(fid, "diagnostics")
                    g_drun = create_group(g_diag, run_label)
                    g_dsrc = fid["diagnostics"]
                    for dset_name in keys(g_dsrc)
                        g_drun[dset_name] = read(g_dsrc, dset_name)
                    end
                end
            end
            push!(rows, row)
        end

        # ── Merged scalars as columnar datasets ───────────
        g_sc_merged = create_group(master, "scalars")
        for k in scalar_keys
            col = [r[k] for r in rows]
            g_sc_merged[k] = col
        end

        # ── Copy sweep metadata if it exists ──────────────
        meta_path = joinpath(results_dir, "sweep_metadata.h5")
        if isfile(meta_path)
            h5open(meta_path, "r") do mfid
                g_meta = create_group(master, "metadata")
                # Recursively copy all datasets from the metadata file
                function copy_h5_group(src, dst)
                    for k in keys(src)
                        obj = src[k]
                        if obj isa HDF5.Group
                            sub = create_group(dst, k)
                            copy_h5_group(obj, sub)
                        else
                            dst[k] = read(obj)
                        end
                    end
                end
                copy_h5_group(mfid["metadata"], g_meta)
            end
            println("Copied sweep metadata into master HDF5.")
        else
            @warn "No sweep_metadata.h5 found — master file will lack metadata."
        end
    end

    df = DataFrame(rows)
    rename_map = Dict(
        "eps_ic" => "ϵ_ic", "eps_ode" => "ϵ_ode", "eps_Data" => "ϵ_Data",
        "eps_L1_state" => "ϵ_L1_state", "eps_L1_g" => "ϵ_L1_g",
        "beta_recovered" => "β_recovered",
        "delta_recovered" => "δ_recovered",
    )
    for (old, new) in rename_map
        old in names(df) && rename!(df, old => new)
    end

    println("Merged $(length(rows)) runs → $master_path")
    return df
end

df_all = merge_sweep_rows(ROW_DIR, MASTER_H5, RESULTS_DIR)

# Keep only converged runs with finite losses for plotting
df = filter(r -> r.converged && isfinite(r.data_mse) && isfinite(r.ode_mse), df_all)
println("$(nrow(df)) converged runs with finite losses (of $(nrow(df_all)) total).\n")

# ══════════════════════════════════════════════════════════
# Utility: Pareto front mask (minimize both objectives)
# ══════════════════════════════════════════════════════════

function pareto_front_mask(x, y)
    n = length(x)
    dominated = falses(n)
    for i in 1:n
        for j in 1:n
            i == j && continue
            if x[j] <= x[i] && y[j] <= y[i] && (x[j] < x[i] || y[j] < y[i])
                dominated[i] = true
                break
            end
        end
    end
    return .!dominated
end

# ══════════════════════════════════════════════════════════
# Reference values
# ══════════════════════════════════════════════════════════

const TRUE_β = 2.7f-5
const TRUE_δ = 4.0f0
const TRUE_c = 3.0f0

const config_keys  = sort(unique(df.config))
const config_colors = Dict(
    "A_5x24"  => :royalblue,
    "B_10x12" => :darkorange,
    "C_20x6"  => :forestgreen,
)

# ══════════════════════════════════════════════════════════
# 2. Pareto front: Data MSE vs ODE MSE
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

save(joinpath(RESULTS_DIR, "sweep_pareto.pdf"), fig1; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 3. Parameter recovery: top runs per config
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

    for (i, cfg) in enumerate(config_keys)
        sub = sort(filter(r -> r.config == cfg, df), :ode_mse)
        top = first(sub, min(10, nrow(sub)))
        vals = top[!, sym]

        xs = fill(Float64(i), length(vals)) .+ 0.1 .* randn(length(vals))
        scatter!(ax, xs, vals;
            color = (config_colors[cfg], 0.6), markersize = 7)

        med = median(vals)
        hlines!(ax, [med]; color = config_colors[cfg],
            linewidth = 2, linestyle = :solid)
    end

    hlines!(ax, [truth]; color = :black, linewidth = 1.5, linestyle = :dash)
    ax.xticks = (1:length(config_keys), config_keys)
end

Label(fig2[0, 1:3],
    "Parameter recovery — top 10 runs by ODE loss per config";
    fontsize = 14, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_param_recovery.pdf"), fig2; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 4. Sensitivity heatmaps
# ══════════════════════════════════════════════════════════

fig3 = Figure(size = (1200, 800), fontsize = 11)

for (row, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)

    ax1 = Makie.Axis(fig3[row, 1];
        title  = "$cfg — ODE loss by (ϵ_ode, ϵ_Data)",
        xlabel = "ϵ_ode", ylabel = "ϵ_Data",
        xscale = log10, yscale = log10,
    )
    sc1 = scatter!(ax1, sub.ϵ_ode, sub.ϵ_Data;
        color = log10.(sub.ode_mse),
        colormap = :viridis, markersize = 8)
    Colorbar(fig3[row, 2], sc1; label = "log₁₀(ODE MSE)")

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

save(joinpath(RESULTS_DIR, "sweep_sensitivity.pdf"), fig3; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 5. Best trajectory per config — loaded from HDF5
#
# For each config, find the run with the lowest ODE MSE and
# plot its stored state and g predictions against the true ODE.
# No model re-evaluation needed.
# ══════════════════════════════════════════════════════════

# Solve the ground-truth ODE once for the reference curves
function Baccam_model!(du, u, p, t)
    T, I, V = u
    du[1] = -p.β * T * V
    du[2] =  p.β * T * V - p.δ * I
    du[3] =  p.p * I     - p.c * V
end

true_params = (β = 2.7f-5, δ = 4.0f0, p = 1.2f-2, c = 3.0f0)
u0_true     = Float32[4.0f8, 0.0f0, 9.3f-2]
t_fine      = collect(range(0f0, 8f0, length = 2001))
sol_true    = solve(
    ODEProblem(Baccam_model!, u0_true, (0f0, 8f0), true_params),
    Tsit5(); saveat = t_fine
)
sol_array = Array(sol_true)

state_labels = ["Target T(t)", "Infected I(t)", "Virus V(t)"]

fig4 = Figure(size = (1200, 700), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    # Identify the best run (lowest ODE MSE) for this config
    sub = sort(filter(r -> r.config == cfg, df), :ode_mse)
    best = first(sub)
    run_label = "$(best.config)_hp$(best.hp_idx)"

    # Read the stored trajectory from the master HDF5
    t_eval, y_hat, g_hat = h5open(MASTER_H5, "r") do fid
        g_run = fid["trajectories/$run_label"]
        (
            read(g_run, "t_eval"),
            read(g_run, "state_pred"),
            read(g_run, "g_pred"),
        )
    end

    # Load training data for scatter overlay
    csv_path = joinpath(RESULTS_DIR, "Baccam_$(cfg).csv")
    df_data = CSV.read(csv_path, DataFrame)
    # Compute per-timepoint mean across mice (for the scatter)
    df_mean = combine(
        groupby(df_data, :t),
        :T_obs => mean => :T_mean,
        :I_obs => mean => :I_mean,
        :V_obs => mean => :V_mean,
    )

    for row in 1:3
        ax = Makie.Axis(fig4[row, col];
            title  = col == 1 ? state_labels[row] : "",
            xlabel = row == 3 ? "Days post-infection" : "",
            ylabel = col == 1 ? "Concentration" : "",
            yscale = Makie.pseudolog10,
            xgridvisible = true, ygridvisible = true,
            topspinevisible = false, rightspinevisible = false,
            xticklabelsize = 9, yticklabelsize = 9,
            titlesize = 11, titlealign = :left,
        )
        if row == 1
            label_text = @sprintf("%s  (hp%d, ODE=%.1e)", cfg, best.hp_idx, best.ode_mse)
            Label(fig4[0, col], label_text; fontsize = 12, tellwidth = false)
        end

        # True ODE reference curve
        lines!(ax, t_fine, sol_array[row, :];
            color = :grey60, linewidth = 1.5, linestyle = :dash)

        # Best-run NN prediction (loaded from HDF5, no refitting)
        lines!(ax, t_eval, y_hat[row, :];
            color = :black, linewidth = 2)

        # Training data (mouse-averaged)
        obs_col = [:T_mean, :I_mean, :V_mean][row]
        scatter!(ax, df_mean.t, df_mean[!, obs_col];
            markersize = 7, color = :white,
            strokecolor = :black, strokewidth = 1.5)
    end
end

Legend(fig4[4, 1:3],
    [LineElement(color = :grey60, linewidth = 1.5, linestyle = :dash),
     LineElement(color = :black, linewidth = 2),
     MarkerElement(marker = :circle, markersize = 7, color = :white,
                   strokecolor = :black, strokewidth = 1.5)],
    ["True ODE", "Best NN(t, θ)", "Training data (mouse mean)"],
    orientation = :horizontal, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_best_trajectories.pdf"), fig4; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 6. Best g-functions per config (from HDF5)
# ══════════════════════════════════════════════════════════

fig5 = Figure(size = (1200, 400), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = sort(filter(r -> r.config == cfg, df), :ode_mse)
    best = first(sub)
    run_label = "$(best.config)_hp$(best.hp_idx)"

    t_eval, g_hat = h5open(MASTER_H5, "r") do fid
        g_run = fid["trajectories/$run_label"]
        (read(g_run, "t_eval"), read(g_run, "g_pred"))
    end

    ax1 = Makie.Axis(fig5[1, col];
        title = "g_T  ($cfg, hp$(best.hp_idx))",
        xlabel = "Days p.i.",
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9,
        titlesize = 11, titlealign = :left)
    lines!(ax1, t_eval, vec(g_hat[1, :]); color = :black, linewidth = 2)

    ax2 = Makie.Axis(fig5[2, col];
        title = "g_V  ($cfg, hp$(best.hp_idx))",
        xlabel = "Days p.i.",
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9,
        titlesize = 11, titlealign = :left)
    lines!(ax2, t_eval, vec(g_hat[2, :]); color = :black, linewidth = 2)
end

Label(fig5[0, 1:3], "Learned unknown physics g(t, θ_g) — best run per config";
    fontsize = 14, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_best_g_functions.pdf"), fig5; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 7. Loss curves for the best run per config
#
# Loaded from /diagnostics in the master HDF5. Shows both
# stages of training with a vertical line at the stage boundary.
# ══════════════════════════════════════════════════════════

fig6 = Figure(size = (1200, 400), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = sort(filter(r -> r.config == cfg, df), :ode_mse)
    best = first(sub)
    run_label = "$(best.config)_hp$(best.hp_idx)"

    # Read loss trajectory from the master HDF5
    has_diag, iters, d_mse_hist, o_mse_hist = h5open(MASTER_H5, "r") do fid
        diag_path = "diagnostics/$run_label"
        if !haskey(fid, diag_path)
            return false, Int32[], Float32[], Float32[]
        end
        g_d = fid[diag_path]
        (
            true,
            read(g_d, "iter"),
            read(g_d, "data_mse"),
            read(g_d, "ode_mse"),
        )
    end

    ax = Makie.Axis(fig6[1, col];
        title  = "$cfg — best run (hp$(best.hp_idx))",
        xlabel = "Iteration",
        ylabel = col == 1 ? "Loss" : "",
        yscale = Makie.log10,
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9,
        titlesize = 11, titlealign = :left,
    )

    if !has_diag
        # No diagnostics recorded — show a placeholder note
        text!(ax, 0.5, 0.5; text = "No diagnostics", align = (:center, :center),
              space = :relative, fontsize = 10)
        continue
    end

    # Stage 1 → Stage 2 boundary (vertical marker)
    # Stage 1 iters are identifiable as those where ode_mse is NaN
    stage2_start = let idx = findfirst(!isnan, o_mse_hist)
        isnothing(idx) ? nothing : iters[idx]
    end
    if !isnothing(stage2_start)
        vlines!(ax, [stage2_start]; color = :grey50, linestyle = :dash, linewidth = 1)
    end

    # Data MSE (present across both stages)
    lines!(ax, iters, d_mse_hist;
        color = :royalblue, linewidth = 1.5, label = "Data MSE")

    # ODE MSE (Stage 2 only — filter NaN from Stage 1)
    ode_mask = .!isnan.(o_mse_hist)
    if any(ode_mask)
        lines!(ax, iters[ode_mask], o_mse_hist[ode_mask];
            color = :firebrick, linewidth = 1.5, label = "ODE MSE")
    end

    col == 1 && axislegend(ax; position = :rt, labelsize = 9)
end

Label(fig6[0, 1:length(config_keys)],
    "Training loss curves — best run per config";
    fontsize = 14, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_best_loss_curves.pdf"), fig6; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 8. Ranked table (stdout)
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

println("\nAll plots saved to $RESULTS_DIR")
println("  sweep_pareto.pdf            — Data vs ODE loss Pareto fronts")
println("  sweep_param_recovery.pdf    — Recovered β, δ, c for top runs")
println("  sweep_sensitivity.pdf       — HP sensitivity heatmaps")
println("  sweep_best_trajectories.pdf — State fits from best run per config")
println("  sweep_best_g_functions.pdf  — Learned g_T, g_V from best runs")
println("  sweep_best_loss_curves.pdf  — Training loss curves from best runs")
