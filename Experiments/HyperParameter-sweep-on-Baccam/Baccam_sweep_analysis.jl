#!/usr/bin/env julia
#
# Baccam_sweep_analysis.jl
#
# Merges per-worker HDF5 files from the sweep into a single master HDF5,
# then generates diagnostic plots. Can skip the merge step by passing
# a pre-existing master HDF5 as a second CLI argument.
#
# Usage:
#   # Full merge + plot (default):
#   julia --project=. Baccam_sweep_analysis.jl [results_dir]
#
#   # Skip merge, use existing master HDF5:
#   julia --project=. Baccam_sweep_analysis.jl [results_dir] --master path/to/master.h5

using Pkg; Pkg.activate(".")

using HDF5
using CSV, DataFrames
using CairoMakie, Makie
using Printf, Statistics
using OrdinaryDiffEq

# ══════════════════════════════════════════════════════════
# 0. Parse CLI arguments
#
# Positional:
#   ARGS[1]  — results directory (default: project-relative)
#
# Optional flag:
#   --master <path>  — skip merge, load this HDF5 directly
# ══════════════════════════════════════════════════════════

function parse_cli(args)
    results_dir = joinpath(pwd(), "Experiments", "HyperParameter-sweep-on-Baccam", "Results")
    master_override = nothing

    i = 1
    while i <= length(args)
        if args[i] == "--master"
            i + 1 <= length(args) || error("--master requires a path argument")
            master_override = args[i + 1]
            i += 2
        else
            # First positional arg is results_dir
            results_dir = args[i]
            i += 1
        end
    end

    return results_dir, master_override
end

const RESULTS_DIR, MASTER_OVERRIDE = parse_cli(ARGS)

const ROW_DIR   = joinpath(RESULTS_DIR, "sweep_rows")
const MASTER_H5 = isnothing(MASTER_OVERRIDE) ?
    joinpath(RESULTS_DIR, "Baccam_hp_sweep.h5") : MASTER_OVERRIDE

# ══════════════════════════════════════════════════════════
# 1. Merge per-worker HDF5 → master HDF5
#    (skipped entirely when --master is provided)
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
                g_sc = fid["scalars"]
                for k in scalar_keys
                    row[k] = read(g_sc, k)
                end

                run_label = "$(row["config"])_hp$(row["hp_idx"])"

                if haskey(fid, "trajectories")
                    g_run = create_group(g_traj, run_label)
                    g_src = fid["trajectories"]
                    for dset_name in keys(g_src)
                        g_run[dset_name] = read(g_src, dset_name)
                    end
                end

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

        g_sc_merged = create_group(master, "scalars")
        for k in scalar_keys
            col = [r[k] for r in rows]
            g_sc_merged[k] = col
        end

        meta_path = joinpath(results_dir, "sweep_metadata.h5")
        if isfile(meta_path)
            h5open(meta_path, "r") do mfid
                g_meta = create_group(master, "metadata")
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
    return df
end

"""
    load_master_h5(master_path) -> DataFrame

Read the columnar scalars from an already-merged master HDF5
and return them as a DataFrame. No worker files are touched.
"""
function load_master_h5(master_path)
    isfile(master_path) || error("Master HDF5 not found: $master_path")

    scalar_keys = [
        "config", "hp_idx",
        "eps_ic", "eps_ode", "eps_Data", "eps_L1_state", "eps_L1_g",
        "data_mse", "ode_mse",
        "beta_recovered", "delta_recovered", "c_recovered",
        "wall_s", "converged",
    ]

    cols = h5open(master_path, "r") do fid
        g_sc = fid["scalars"]
        Dict(k => read(g_sc, k) for k in scalar_keys)
    end

    println("Loaded $(length(cols["config"])) runs from $master_path (merge skipped).")
    return DataFrame(cols)
end

# ── Decide whether to merge or load ──────────────────────
df_all = if !isnothing(MASTER_OVERRIDE)
    load_master_h5(MASTER_H5)
else
    merge_sweep_rows(ROW_DIR, MASTER_H5, RESULTS_DIR)
end

# Standardise column names regardless of source
rename_map = Dict(
    "eps_ic" => "ϵ_ic", "eps_ode" => "ϵ_ode", "eps_Data" => "ϵ_Data",
    "eps_L1_state" => "ϵ_L1_state", "eps_L1_g" => "ϵ_L1_g",
    "beta_recovered" => "β_recovered",
    "delta_recovered" => "δ_recovered",
)
for (old, new) in rename_map
    old in names(df_all) && rename!(df_all, old => new)
end

# Keep only converged runs with finite losses for plotting
df = filter(r -> r.converged && isfinite(r.data_mse) && isfinite(r.ode_mse), df_all)
println("$(nrow(df)) converged runs with finite losses (of $(nrow(df_all)) total).\n")

# ══════════════════════════════════════════════════════════
# Utility: Pareto front mask (minimize both objectives)
# ══════════════════════════════════════════════════════════

"""
    pareto_front_mask(x, y) -> BitVector

Return a boolean mask where `true` marks non-dominated points.
A point (xᵢ, yᵢ) is dominated if some other point is ≤ on both
objectives and strictly < on at least one.
"""
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
# Utility: Pareto-based "best" run selection
#
# Normalisation is done in log₁₀ space. Both Data MSE and
# ODE MSE span orders of magnitude, so raw min-max would
# compress the meaningful variation at the low end. Working
# in log₁₀ ensures that a 10× improvement contributes
# equally regardless of absolute scale.
# ══════════════════════════════════════════════════════════

"""
    _nadir_scores(data_mse, ode_mse) -> Vector{Float64}

Score each run by its distance from the nadir (worst) point in
min-max normalised objective space. Higher score = better compromise.

Each axis is min-max scaled to [0, 1] in raw (linear) space.
The nadir is (1, 1). The returned score is the Euclidean distance
from (1, 1), so larger = better.
"""
function _nadir_scores(data_mse, ode_mse)
    d = Float64.(data_mse)
    o = Float64.(ode_mse)

    d_range = max(maximum(d) - minimum(d), 1e-12)
    o_range = max(maximum(o) - minimum(o), 1e-12)

    # Normalise to [0, 1] where 0 = best, 1 = worst on each axis
    d_norm = (d .- minimum(d)) ./ d_range
    o_norm = (o .- minimum(o)) ./ o_range

    # Distance from nadir (1, 1) — larger means farther from worst
    return sqrt.((1.0 .- d_norm) .^ 2 .+ (1.0 .- o_norm) .^ 2)
end

"""
    select_pareto_best(sub::DataFrame) -> DataFrameRow

From a per-config subset `sub`, identify the Pareto front in
(data_mse, ode_mse) space, then return the point farthest from
the nadir (1, 1) in log₁₀-normalised coordinates. This selects
the "elbow" of convex fronts where neither objective is sacrificed.

Falls back to the single Pareto point if the front is degenerate.
"""
function select_pareto_best(sub::DataFrame)
    mask = pareto_front_mask(sub.data_mse, sub.ode_mse)
    pareto_sub = sub[mask, :]

    nrow(pareto_sub) <= 1 && return first(eachrow(pareto_sub))

    scores = _nadir_scores(pareto_sub.data_mse, pareto_sub.ode_mse)
    # Maximize distance from nadir → best compromise
    return pareto_sub[argmax(scores), :]
end

"""
    select_pareto_top(sub::DataFrame, n::Int) -> DataFrame

Return up to `n` Pareto-optimal runs from `sub`, sorted by
nadir distance (descending — best compromise first).

The `nadir_dist` column is added so downstream code can use it
for colouring and ranking.
"""
function select_pareto_top(sub::DataFrame, n::Int)
    mask = pareto_front_mask(sub.data_mse, sub.ode_mse)
    pareto_sub = copy(sub[mask, :])

    nrow(pareto_sub) <= 1 && begin
        pareto_sub.nadir_dist = [0.0]
        return pareto_sub
    end

    pareto_sub.nadir_dist = _nadir_scores(pareto_sub.data_mse, pareto_sub.ode_mse)
    # Sort descending: highest nadir distance = best elbow compromise
    sorted = sort(pareto_sub, :nadir_dist; rev = true)
    return first(sorted, min(n, nrow(sorted)))
end

"""
    nadir_scores_all(sub::DataFrame) -> Vector{Float64}

Compute log₁₀-normalised nadir distance for ALL runs in `sub`
(not just Pareto-optimal ones). Used for colouring heatmaps —
higher score means better balanced performance on both objectives.
"""
function nadir_scores_all(sub::DataFrame)
    return _nadir_scores(sub.data_mse, sub.ode_mse)
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

    best = select_pareto_best(sub)
    scatter!(ax, [best.data_mse], [best.ode_mse];
        color = :gold, markersize = 14, marker = :diamond,
        strokecolor = :black, strokewidth = 1.5,
        label = "Pareto best")

    col == 1 && axislegend(ax; position = :rt, labelsize = 9)
end

Label(fig1[0, 1:length(config_keys)],
    "Data MSE vs ODE MSE — each dot is one HP combination";
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
        topspinevisible = false, rightspinevisible = false,
    )

    for (i, cfg) in enumerate(config_keys)
        sub = filter(r -> r.config == cfg, df)
        pareto_mask = pareto_front_mask(sub.data_mse, sub.ode_mse)
        top = sub[pareto_mask, :]
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
    "Parameter recovery — Pareto-optimal runs per config";
    fontsize = 14, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_param_recovery.pdf"), fig2; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 4. Sensitivity heatmaps — full pairwise triangle matrix
#
# All C(5,2) = 10 ϵ pairs are shown in a lower-triangle
# layout. Each panel is a scatter of the two ϵ values
# coloured by log₁₀-normalised utopia distance (lower =
# better on both objectives simultaneously).
#
# Diagonal panels show marginal histograms of each ϵ,
# weighted by inverse utopia distance so that good runs
# contribute more visual mass.
# ══════════════════════════════════════════════════════════

const hp_names  = ["ϵ_ic", "ϵ_ode", "ϵ_Data", "ϵ_L1_state", "ϵ_L1_g"]
const hp_syms   = [Symbol(n) for n in hp_names]
const N_HP      = length(hp_names)

for cfg in config_keys
    sub = filter(r -> r.config == cfg, df)
    scores = nadir_scores_all(sub)

    # High nadir distance = good compromise; use directly for colouring
    color_vals = scores

    fig_tri = Figure(size = (1200, 1200), fontsize = 10)
    Label(fig_tri[0, 1:N_HP], "$cfg — pairwise ϵ sensitivity (colour = utopia distance)";
        fontsize = 14, tellwidth = false)

    for row in 1:N_HP, col in 1:N_HP
        if col > row
            # Upper triangle: leave empty
            continue
        elseif col == row
            # Diagonal: marginal histogram of this ϵ
            ax = Makie.Axis(fig_tri[row, col];
                xlabel = row == N_HP ? hp_names[col] : "",
                ylabel = "",
                xscale = log10,
                topspinevisible = false, rightspinevisible = false,
                xticklabelsize = 8, yticklabelsize = 8,
            )
            # Pass raw values — the axis xscale = log10 handles the transform
            hist!(ax, sub[!, hp_syms[col]];
                bins = 25, color = (config_colors[cfg], 0.6))

            # Hide y-axis ticks on diagonal — counts aren't meaningful
            hideydecorations!(ax; grid = false)
        else
            # Lower triangle: pairwise scatter
            ax = Makie.Axis(fig_tri[row, col];
                xlabel = row == N_HP ? hp_names[col] : "",
                ylabel = col == 1    ? hp_names[row] : "",
                xscale = log10, yscale = log10,
                topspinevisible = false, rightspinevisible = false,
                xticklabelsize = 8, yticklabelsize = 8,
            )

            sc = scatter!(ax,
                sub[!, hp_syms[col]],
                sub[!, hp_syms[row]];
                color = color_vals,
                colormap = Reverse(:viridis),
                markersize = 5,
                alpha = 0.7,
            )

            # Shared colorbar: place once in the upper-right corner
            if row == 2 && col == 1
                Colorbar(fig_tri[1, N_HP], sc;
                    label = "Nadir dist (higher = better)",
                    flipaxis = true,
                )
            end

            # Suppress axis labels on interior panels to reduce clutter
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
#
# Shows the top N_TOP Pareto-optimal runs coloured by their
# utopia distance (darker = closer to ideal). Training data
# is shown as boxplots at each timepoint to visualise the
# per-mouse noise spread, replacing the old scatter of means.
# ══════════════════════════════════════════════════════════

const N_TOP = 15   # number of Pareto-best trajectories to overlay

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
obs_syms     = [:T_obs, :I_obs, :V_obs]

fig4 = Figure(size = (1200, 800), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)
    top_runs = select_pareto_top(sub, N_TOP)

    # Load per-mouse training data for boxplots
    csv_path = joinpath(RESULTS_DIR, "Baccam_$(cfg).csv")
    df_data = CSV.read(csv_path, DataFrame)

    # Determine boxplot width from the minimum gap between timepoints
    unique_t = sort(unique(df_data.t))
    box_width = length(unique_t) > 1 ?
        0.6f0 * minimum(diff(unique_t)) : 0.3f0

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
            Label(fig4[0, col], cfg; fontsize = 13, tellwidth = false)
        end

        # True ODE reference curve
        lines!(ax, t_fine, sol_array[row, :];
            color = :grey60, linewidth = 1.5, linestyle = :dash)

        # Plot trajectories first so boxplots render on top
        nd_min = minimum(top_runs.nadir_dist)
        nd_max = maximum(top_runs.nadir_dist)
        nd_range = max(nd_max - nd_min, 1e-12)

        for r in eachrow(top_runs)
            run_label = "$(r.config)_hp$(r.hp_idx)"

            t_eval, y_hat = h5open(MASTER_H5, "r") do fid
                g_run = fid["trajectories/$run_label"]
                (read(g_run, "t_eval"), read(g_run, "state_pred"))
            end

            # Normalised: 1 = best (highest nadir dist), 0 = worst among top-N
            nd_frac = (r.nadir_dist - nd_min) / nd_range

            lines!(ax, t_eval, y_hat[row, :];
                color = nd_frac,
                colorrange = (0.0, 1.0),
                colormap = Reverse(:viridis),
                linewidth = 2)
        end

        # Boxplots on top for readability
        boxplot!(ax,
            df_data.t,
            df_data[!, obs_syms[row]];
            width = box_width,
            color = (:grey80, 0.5),
            whiskerwidth = 0.4,
            strokewidth = 0.8,
            mediancolor = :black,
            medianlinewidth = 1.5,
            outliercolor = (:black, 0.3),
            markersize = 3,
        )
    end
end

# Colorbar for the utopia distance colour ramp
Colorbar(fig4[4, 3];
    colormap = Reverse(:viridis),
    limits = (0.0, 1.0),
    label = "Nadir distance (normalised, blue = best)",
    vertical = false,
    flipaxis = false,
    tellwidth = false,
)

Legend(fig4[4, 1:2],
    [LineElement(color = :grey60, linewidth = 1.5, linestyle = :dash),
     MarkerElement(marker = :rect, markersize = 12, color = (:grey80, 0.5),
                   strokecolor = :black, strokewidth = 0.8)],
    ["True ODE", "Mouse data (boxplot)"],
    orientation = :horizontal, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_best_trajectories.pdf"), fig4; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 6. Top-N g-functions per config (Pareto-selected, from HDF5)
#
# Same top-N overlay approach as the state trajectories,
# coloured by utopia distance.
# ══════════════════════════════════════════════════════════

fig5 = Figure(size = (1200, 400), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)
    top_runs = select_pareto_top(sub, N_TOP)

    ax1 = Makie.Axis(fig5[1, col];
        title = "g_T  ($cfg)",
        xlabel = "Days p.i.",
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9,
        titlesize = 11, titlealign = :left)

    ax2 = Makie.Axis(fig5[2, col];
        title = "g_V  ($cfg)",
        xlabel = "Days p.i.",
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9,
        titlesize = 11, titlealign = :left)

    nd_min = minimum(top_runs.nadir_dist)
    nd_max = maximum(top_runs.nadir_dist)
    nd_range = max(nd_max - nd_min, 1e-12)

    for r in eachrow(top_runs)
        run_label = "$(r.config)_hp$(r.hp_idx)"

        t_eval, g_hat = h5open(MASTER_H5, "r") do fid
            g_run = fid["trajectories/$run_label"]
            (read(g_run, "t_eval"), read(g_run, "g_pred"))
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

Legend(fig5[3, 1:2],
    [LineElement(color = :grey60, linewidth = 1.5, linestyle = :dash)],
    ["True ODE (if overlaid)"],
    orientation = :horizontal, tellwidth = false)

Colorbar(fig5[3, 3];
    colormap = Reverse(:viridis),
    limits = (0.0, 1.0),
    label = "Nadir distance (normalised, blue = best)",
    vertical = false,
    flipaxis = false,
    tellwidth = false,
)

save(joinpath(RESULTS_DIR, "sweep_best_g_functions.pdf"), fig5; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 7. Loss curves for the Pareto-best run per config
# ══════════════════════════════════════════════════════════

fig6 = Figure(size = (1200, 400), fontsize = 12)

for (col, cfg) in enumerate(config_keys)
    sub = filter(r -> r.config == cfg, df)
    best = select_pareto_best(sub)
    run_label = "$(best.config)_hp$(best.hp_idx)"

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
        title  = "$cfg — Pareto-best (hp$(best.hp_idx))",
        xlabel = "Iteration",
        ylabel = col == 1 ? "Loss" : "",
        yscale = Makie.log10,
        xgridvisible = true, ygridvisible = true,
        topspinevisible = false, rightspinevisible = false,
        xticklabelsize = 9, yticklabelsize = 9,
        titlesize = 11, titlealign = :left,
    )

    if !has_diag
        text!(ax, 0.5, 0.5; text = "No diagnostics", align = (:center, :center),
              space = :relative, fontsize = 10)
        continue
    end

    stage2_start = let idx = findfirst(!isnan, o_mse_hist)
        isnothing(idx) ? nothing : iters[idx]
    end
    if !isnothing(stage2_start)
        vlines!(ax, [stage2_start]; color = :grey50, linestyle = :dash, linewidth = 1)
    end

    lines!(ax, iters, d_mse_hist;
        color = :royalblue, linewidth = 1.5, label = "Data MSE")

    ode_mask = .!isnan.(o_mse_hist)
    if any(ode_mask)
        lines!(ax, iters[ode_mask], o_mse_hist[ode_mask];
            color = :firebrick, linewidth = 1.5, label = "ODE MSE")
    end

    col == 1 && axislegend(ax; position = :rt, labelsize = 9)
end

Label(fig6[0, 1:length(config_keys)],
    "Training loss curves — Pareto-best run per config";
    fontsize = 14, tellwidth = false)

save(joinpath(RESULTS_DIR, "sweep_best_loss_curves.pdf"), fig6; px_per_unit = 600/72)

# ══════════════════════════════════════════════════════════
# 8. Ranked table — Pareto-optimal runs, sorted by
#    utopia distance in log₁₀ space
# ══════════════════════════════════════════════════════════

println("="^100)
println("  Top 5 Pareto-optimal runs per configuration (sorted by nadir distance, descending)")
println("="^100)

for cfg in config_keys
    sub = filter(r -> r.config == cfg, df)
    top = select_pareto_top(sub, 5)

    println("\n── $cfg  ($(sum(pareto_front_mask(sub.data_mse, sub.ode_mse))) Pareto-optimal of $(nrow(sub))) ──")
    @printf("  %-6s  %-10s  %-10s  %-10s  %-8s  %-8s  %-8s  %-10s  %-10s  %-10s\n",
            "hp_idx", "nadir_d", "ode_mse", "data_mse", "ϵ_ode", "ϵ_Data", "ϵ_ic",
            "β_rec", "δ_rec", "c_rec")
    println("  ", "-"^96)
    for r in eachrow(top)
        @printf("  %-6d  %.3e  %.3e  %.3e  %-8.3f  %-8.3f  %-8.3f  %.2e  %8.3f  %8.3f\n",
                r.hp_idx, r.nadir_dist, r.ode_mse, r.data_mse, r.ϵ_ode, r.ϵ_Data, r.ϵ_ic,
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
