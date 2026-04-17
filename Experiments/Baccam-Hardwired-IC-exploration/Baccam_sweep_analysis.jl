#!/usr/bin/env julia
#
# Baccam_sweep_analysis.jl — Composite transform sweep
#
# Merges per-worker HDF5 files into a master HDF5 and generates
# diagnostic plots.  Handles all three sampling configs
# (A_5x24, B_10x12, C_20x6).
#
# Usage:
#   julia --project=. Baccam_sweep_analysis.jl [results_dir]
#   julia --project=. Baccam_sweep_analysis.jl [results_dir] --master path/to/master.h5

using Pkg; Pkg.activate(".")

using HDF5
using CSV, DataFrames
using CairoMakie, Makie
using Printf, Statistics
using OrdinaryDiffEq

# ══════════════════════════════════════════════════════════
# 0. CLI
# ══════════════════════════════════════════════════════════

function parse_cli(args)
    results_dir = joinpath(pwd(), "Experiments", "CompositeTransform-sweep", "Results")
    master_override = nothing
    i = 1
    while i <= length(args)
        if args[i] == "--master"
            i + 1 <= length(args) || error("--master requires a path")
            master_override = args[i + 1]
            i += 2
        else
            results_dir = args[i]
            i += 1
        end
    end
    return results_dir, master_override
end

const RESULTS_DIR, MASTER_OVERRIDE = parse_cli(ARGS)
const ROW_DIR   = joinpath(RESULTS_DIR, "sweep_rows")
const MASTER_H5 = isnothing(MASTER_OVERRIDE) ?
    joinpath(RESULTS_DIR, "Baccam_composite_hp_sweep.h5") : MASTER_OVERRIDE

# ══════════════════════════════════════════════════════════
# 1. Merge / load
# ══════════════════════════════════════════════════════════

const SCALAR_KEYS = [
    "config", "hp_idx",
    "eps_ic", "eps_ode", "eps_Data", "eps_L1_state", "eps_L1_g",
    "data_mse", "ode_mse",
    "beta_recovered", "delta_recovered", "c_recovered",
    "wall_s", "converged",
]

function merge_sweep_rows(row_dir, master_path, results_dir)
    h5_files = filter(f -> endswith(f, ".h5"), readdir(row_dir; join=true))
    isempty(h5_files) && error("No .h5 files found in $row_dir")
    println("Found $(length(h5_files)) worker files to merge.")

    rows = Dict{String,Any}[]

    h5open(master_path, "w") do master
        g_traj = create_group(master, "trajectories")
        g_diag = create_group(master, "diagnostics")

        for fpath in h5_files
            row = Dict{String,Any}()
            h5open(fpath, "r") do fid
                g_sc = fid["scalars"]
                for k in SCALAR_KEYS
                    haskey(g_sc, k) && (row[k] = read(g_sc, k))
                end

                run_label = "$(row["config"])_hp$(row["hp_idx"])"

                for (group_name, dst_group) in [("trajectories", g_traj),
                                                 ("diagnostics", g_diag)]
                    haskey(fid, group_name) || continue
                    g_run = create_group(dst_group, run_label)
                    g_src = fid[group_name]
                    for dset in keys(g_src)
                        g_run[dset] = read(g_src, dset)
                    end
                end
            end
            push!(rows, row)
        end

        g_sc_merged = create_group(master, "scalars")
        for k in SCALAR_KEYS
            vals = [get(r, k, missing) for r in rows]
            any(ismissing, vals) && continue
            g_sc_merged[k] = collect(vals)
        end

        meta_path = joinpath(results_dir, "sweep_metadata.h5")
        if isfile(meta_path)
            h5open(meta_path, "r") do mfid
                g_meta = create_group(master, "metadata")
                function copy_group(src, dst)
                    for k in keys(src)
                        obj = src[k]
                        if obj isa HDF5.Group
                            copy_group(obj, create_group(dst, k))
                        else
                            dst[k] = read(obj)
                        end
                    end
                end
                copy_group(mfid["metadata"], g_meta)
            end
            println("Copied sweep metadata into master.")
        end
    end

    return DataFrame(rows)
end

function load_master_h5(path)
    isfile(path) || error("Master HDF5 not found: $path")
    cols = h5open(path, "r") do fid
        g = fid["scalars"]
        Dict(k => read(g, k) for k in SCALAR_KEYS if haskey(g, k))
    end
    println("Loaded $(length(cols["config"])) runs from $path")
    return DataFrame(cols)
end

df_all = isnothing(MASTER_OVERRIDE) ?
    merge_sweep_rows(ROW_DIR, MASTER_H5, RESULTS_DIR) :
    load_master_h5(MASTER_H5)

for (old, new) in ["eps_ic"=>"ϵ_ic", "eps_ode"=>"ϵ_ode", "eps_Data"=>"ϵ_Data",
                    "eps_L1_state"=>"ϵ_L1_state", "eps_L1_g"=>"ϵ_L1_g",
                    "beta_recovered"=>"β_recovered", "delta_recovered"=>"δ_recovered"]
    old in names(df_all) && rename!(df_all, old => new)
end

df = filter(r -> r.converged && isfinite(r.data_mse) && isfinite(r.ode_mse), df_all)
println("$(nrow(df)) converged / finite of $(nrow(df_all)) total.\n")

# ══════════════════════════════════════════════════════════
# Pareto utilities
# ══════════════════════════════════════════════════════════

function pareto_front_mask(x, y)
    n = length(x)
    dominated = falses(n)
    for i in 1:n, j in 1:n
        i == j && continue
        if x[j] <= x[i] && y[j] <= y[i] && (x[j] < x[i] || y[j] < y[i])
            dominated[i] = true
            break
        end
    end
    return .!dominated
end

function _nadir_scores(data_mse, ode_mse)
    d, o = Float64.(data_mse), Float64.(ode_mse)
    d_n = (d .- minimum(d)) ./ max(maximum(d) - minimum(d), 1e-12)
    o_n = (o .- minimum(o)) ./ max(maximum(o) - minimum(o), 1e-12)
    return sqrt.((1.0 .- d_n).^2 .+ (1.0 .- o_n).^2)
end

function select_pareto_best(sub)
    mask = pareto_front_mask(sub.data_mse, sub.ode_mse)
    ps = sub[mask, :]
    nrow(ps) <= 1 && return first(eachrow(ps))
    return ps[argmax(_nadir_scores(ps.data_mse, ps.ode_mse)), :]
end

function select_pareto_top(sub, n)
    mask = pareto_front_mask(sub.data_mse, sub.ode_mse)
    ps = copy(sub[mask, :])
    nrow(ps) <= 1 && (ps.nadir_dist = [0.0]; return ps)
    ps.nadir_dist = _nadir_scores(ps.data_mse, ps.ode_mse)
    return first(sort(ps, :nadir_dist; rev=true), min(n, nrow(ps)))
end

function nadir_scores_all(sub)
    return _nadir_scores(sub.data_mse, sub.ode_mse)
end

# ══════════════════════════════════════════════════════════
# Reference ODE
# ══════════════════════════════════════════════════════════

const TRUE_β = 2.7f-5
const TRUE_δ = 4.0f0
const TRUE_c = 3.0f0

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
    Tsit5(); saveat = t_fine)
sol_array = Array(sol_true)

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
        topspinevisible = false, rightspinevisible = false)

    scatter!(ax, sub.data_mse, sub.ode_mse;
        color = config_colors[cfg], markersize = 6, alpha = 0.6)

    pmask = pareto_front_mask(sub.data_mse, sub.ode_mse)
    scatter!(ax, sub.data_mse[pmask], sub.ode_mse[pmask];
        color = :red, markersize = 10, marker = :star5, label = "Pareto front")

    best = select_pareto_best(sub)
    scatter!(ax, [best.data_mse], [best.ode_mse];
        color = :gold, markersize = 14, marker = :diamond,
        strokecolor = :black, strokewidth = 1.5, label = "Pareto best")

    col == 1 && axislegend(ax; position = :rt, labelsize = 9)
end

Label(fig1[0, 1:length(config_keys)],
    "Data MSE vs ODE MSE — ComposedTransform (Log₁₀ + ZScore)";
    fontsize = 14, tellwidth = false)
save(joinpath(RESULTS_DIR, "sweep_pareto.pdf"), fig1; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 3. Parameter recovery: Pareto-optimal runs per config
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
        topspinevisible = false, rightspinevisible = false)

    for (i, cfg) in enumerate(config_keys)
        sub = filter(r -> r.config == cfg, df)
        pmask = pareto_front_mask(sub.data_mse, sub.ode_mse)
        vals = sub[pmask, sym]

        xs = fill(Float64(i), length(vals)) .+ 0.1 .* randn(length(vals))
        scatter!(ax, xs, vals;
            color = (config_colors[cfg], 0.6), markersize = 7)
        hlines!(ax, [median(vals)]; color = config_colors[cfg],
            linewidth = 2, linestyle = :solid)
    end

    hlines!(ax, [truth]; color = :black, linewidth = 1.5, linestyle = :dash)
    ax.xticks = (1:length(config_keys), config_keys)
end

Label(fig2[0, 1:3],
    "Parameter recovery — Pareto-optimal runs per config";
    fontsize = 14, tellwidth = false)
save(joinpath(RESULTS_DIR, "sweep_param_recovery.pdf"), fig2; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 4. Sensitivity heatmaps — pairwise triangle per config
# ══════════════════════════════════════════════════════════

const hp_names = ["ϵ_ode", "ϵ_Data", "ϵ_L1_state", "ϵ_L1_g"]
const hp_syms  = Symbol.(hp_names)
const N_HP     = length(hp_names)

for cfg in config_keys
    sub = filter(r -> r.config == cfg, df)
    scores = nadir_scores_all(sub)

    fig_tri = Figure(size = (1000, 1000), fontsize = 10)
    Label(fig_tri[0, 1:N_HP], "$cfg — pairwise ϵ sensitivity (colour = nadir distance)";
        fontsize = 14, tellwidth = false)

    for row in 1:N_HP, col in 1:N_HP
        col > row && continue

        if col == row
            ax = Makie.Axis(fig_tri[row, col];
                xlabel = row == N_HP ? hp_names[col] : "", xscale = log10,
                topspinevisible = false, rightspinevisible = false,
                xticklabelsize = 8, yticklabelsize = 8)
            hist!(ax, sub[!, hp_syms[col]]; bins = 25, color = (config_colors[cfg], 0.6))
            hideydecorations!(ax; grid = false)
        else
            ax = Makie.Axis(fig_tri[row, col];
                xlabel = row == N_HP ? hp_names[col] : "",
                ylabel = col == 1    ? hp_names[row] : "",
                xscale = log10, yscale = log10,
                topspinevisible = false, rightspinevisible = false,
                xticklabelsize = 8, yticklabelsize = 8)

            sc = scatter!(ax, sub[!, hp_syms[col]], sub[!, hp_syms[row]];
                color = scores, colormap = Reverse(:viridis),
                markersize = 5, alpha = 0.7)

            if row == 2 && col == 1
                Colorbar(fig_tri[1, N_HP], sc;
                    label = "Nadir dist (higher = better)", flipaxis = true)
            end

            row != N_HP && hidexdecorations!(ax; grid = false)
            col != 1    && hideydecorations!(ax; grid = false)
        end
    end

    save(joinpath(RESULTS_DIR, "sweep_sensitivity_triangle_$(cfg).pdf"),
         fig_tri; px_per_unit = 600/72)
    println("Saved triangle heatmap for $cfg")
end

# ══════════════════════════════════════════════════════════
# 5. Top-N Pareto trajectories per config
# ══════════════════════════════════════════════════════════

const N_TOP = 5
const PLOT_FLOOR = 1f-2
const state_labels = ["Target T(t)", "Infected I(t)", "Virus V(t)"]
const obs_syms     = [:T_obs, :I_obs, :V_obs]

fig4 = Figure(size = (1200, 800), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)
    top_runs = select_pareto_top(sub, N_TOP)

    csv_path = joinpath(RESULTS_DIR, "Baccam_$(cfg).csv")
    df_data  = CSV.read(csv_path, DataFrame)

    unique_t  = sort(unique(df_data.t))
    box_width = length(unique_t) > 1 ? 0.6f0 * minimum(diff(unique_t)) : 0.3f0

    for row in 1:3
        ax = Makie.Axis(fig4[row, col];
            title  = col == 1 ? state_labels[row] : "",
            xlabel = row == 3 ? "Days post-infection" : "",
            ylabel = col == 1 ? "Concentration" : "",
            yscale = Makie.pseudolog10,
            xgridvisible = true, ygridvisible = true,
            topspinevisible = false, rightspinevisible = false,
            xticklabelsize = 9, yticklabelsize = 9,
            titlesize = 11, titlealign = :left)

        row == 1 && Label(fig4[0, col], cfg; fontsize = 13, tellwidth = false)

        lines!(ax, t_fine, sol_array[row, :];
            color = :grey60, linewidth = 1.5, linestyle = :dash)

        nd_min, nd_max = extrema(top_runs.nadir_dist)
        nd_range = max(nd_max - nd_min, 1e-12)

        for r in eachrow(top_runs)
            run_label = "$(r.config)_hp$(r.hp_idx)"
            t_eval, y_hat = h5open(MASTER_H5, "r") do fid
                g = fid["trajectories/$run_label"]
                (read(g, "t_eval"), read(g, "state_pred"))
            end
            lines!(ax, t_eval, y_hat[row, :] .+ PLOT_FLOOR;
                color = (r.nadir_dist - nd_min) / nd_range,
                colorrange = (0.0, 1.0), colormap = Reverse(:viridis), linewidth = 2)
        end

        boxplot!(ax, df_data.t, df_data[!, obs_syms[row]];
            width = box_width, color = (:grey80, 0.5),
            whiskerwidth = 0.4, strokewidth = 0.8,
            mediancolor = :black, medianlinewidth = 1.5,
            outliercolor = (:black, 0.3), markersize = 3)
    end
end

Colorbar(fig4[4, 3]; colormap = Reverse(:viridis), limits = (0.0, 1.0),
    label = "Nadir distance (normalised, blue = best)",
    vertical = false, flipaxis = false, tellwidth = false)
Legend(fig4[4, 1:2],
    [LineElement(color = :grey60, linewidth = 1.5, linestyle = :dash),
     MarkerElement(marker = :rect, markersize = 12, color = (:grey80, 0.5),
                   strokecolor = :black, strokewidth = 0.8)],
    ["True ODE", "Mouse data (boxplot)"],
    orientation = :horizontal, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_best_trajectories.pdf"), fig4; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 6. Top-N g-functions per config
# ══════════════════════════════════════════════════════════

fig5 = Figure(size = (1200, 400), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)
    top_runs = select_pareto_top(sub, N_TOP)

    ax1 = Makie.Axis(fig5[1, col];
        title = "g_T  ($cfg)", xlabel = "Days p.i.",
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9, titlesize = 11, titlealign = :left)
    ax2 = Makie.Axis(fig5[2, col];
        title = "g_V  ($cfg)", xlabel = "Days p.i.",
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9, titlesize = 11, titlealign = :left)

    nd_min, nd_max = extrema(top_runs.nadir_dist)
    nd_range = max(nd_max - nd_min, 1e-12)

    for r in eachrow(top_runs)
        run_label = "$(r.config)_hp$(r.hp_idx)"
        t_eval, g_hat = h5open(MASTER_H5, "r") do fid
            g = fid["trajectories/$run_label"]
            (read(g, "t_eval"), read(g, "g_pred"))
        end
        nd_frac = (r.nadir_dist - nd_min) / nd_range
        lines!(ax1, t_eval, vec(g_hat[1, :]);
            color = nd_frac, colorrange = (0.0, 1.0),
            colormap = Reverse(:viridis), linewidth = 2)
        lines!(ax2, t_eval, vec(g_hat[2, :]);
            color = nd_frac, colorrange = (0.0, 1.0),
            colormap = Reverse(:viridis), linewidth = 2)
    end
end

Label(fig5[0, 1:3],
    "Learned unknown physics g(t, θ_g) — top $N_TOP Pareto runs per config";
    fontsize = 14, tellwidth = false)
Colorbar(fig5[3, 3]; colormap = Reverse(:viridis), limits = (0.0, 1.0),
    label = "Nadir distance (normalised, blue = best)",
    vertical = false, flipaxis = false, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_best_g_functions.pdf"), fig5; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 7. Loss curves for Pareto-best run per config
# ══════════════════════════════════════════════════════════

fig6 = Figure(size = (1200, 400), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)
    best = select_pareto_best(sub)
    run_label = "$(best.config)_hp$(best.hp_idx)"

    has_diag, iters, d_hist, o_hist = h5open(MASTER_H5, "r") do fid
        p = "diagnostics/$run_label"
        haskey(fid, p) || return (false, Int32[], Float32[], Float32[])
        g = fid[p]
        (true, read(g, "iter"), read(g, "data_mse"), read(g, "ode_mse"))
    end

    ax = Makie.Axis(fig6[1, col];
        title  = "$cfg — Pareto-best (hp$(best.hp_idx))",
        xlabel = "Iteration",
        ylabel = col == 1 ? "Loss" : "",
        yscale = Makie.log10,
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9, titlesize = 11, titlealign = :left)

    if !has_diag
        text!(ax, 0.5, 0.5; text = "No diagnostics",
              align = (:center, :center), space = :relative)
        continue
    end

    s2_start = let idx = findfirst(!isnan, o_hist)
        isnothing(idx) ? nothing : iters[idx] end
    !isnothing(s2_start) && vlines!(ax, [s2_start]; color = :grey50, linestyle = :dash)

    lines!(ax, iters, d_hist; color = :royalblue, linewidth = 1.5, label = "Data MSE")
    omask = .!isnan.(o_hist)
    any(omask) && lines!(ax, iters[omask], o_hist[omask];
        color = :firebrick, linewidth = 1.5, label = "ODE MSE")

    col == 1 && axislegend(ax; position = :rt, labelsize = 9)
end

Label(fig6[0, 1:length(config_keys)],
    "Training loss curves — Pareto-best run per config";
    fontsize = 14, tellwidth = false)
save(joinpath(RESULTS_DIR, "sweep_best_loss_curves.pdf"), fig6; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 8. Ranked table
# ══════════════════════════════════════════════════════════

println("="^100)
println("  Top 5 Pareto-optimal runs per configuration (sorted by nadir distance, descending)")
println("="^100)

for cfg in config_keys
    sub = filter(r -> r.config == cfg, df)
    top = select_pareto_top(sub, 5)

    n_pareto = sum(pareto_front_mask(sub.data_mse, sub.ode_mse))
    println("\n── $cfg  ($n_pareto Pareto-optimal of $(nrow(sub))) ──")
    @printf("  %-6s  %-10s  %-10s  %-10s  %-8s  %-8s  %-10s  %-10s  %-10s\n",
            "hp_idx", "nadir_d", "ode_mse", "data_mse", "ϵ_ode", "ϵ_Data",
            "β_rec", "δ_rec", "c_rec")
    println("  ", "-"^96)
    for r in eachrow(top)
        @printf("  %-6d  %.3e  %.3e  %.3e  %-8.3f  %-8.3f  %.2e  %8.3f  %8.3f\n",
                r.hp_idx, r.nadir_dist, r.ode_mse, r.data_mse, r.ϵ_ode, r.ϵ_Data,
                r.β_recovered, r.δ_recovered, r.c_recovered)
    end
end

println("\nAll plots saved to $RESULTS_DIR")
println("  sweep_pareto.pdf                        — Data vs ODE loss Pareto fronts")
println("  sweep_param_recovery.pdf                — Recovered β, δ, c for Pareto-optimal runs")
println("  sweep_sensitivity_triangle_<cfg>.pdf    — Full pairwise ϵ triangle heatmaps")
println("  sweep_best_trajectories.pdf             — State fits from Pareto-best run per config")
println("  sweep_best_g_functions.pdf              — Learned g_T, g_V from Pareto-best runs")
println("  sweep_best_loss_curves.pdf              — Training loss curves from Pareto-best runs")
