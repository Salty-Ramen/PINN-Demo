#!/usr/bin/env julia
#
# write_sweep_metadata.jl — Hardwired IC sweep
#
# Captures the sweep configuration and environment into HDF5
# for reproducibility. Called once by run_sweep.sh before workers launch.
#
# Usage:
#   julia --project=. write_sweep_metadata.jl <results_dir> <project_dir> <max_jobs>

using Pkg; Pkg.activate(".")

using HDF5
using Dates

if length(ARGS) < 3
    error("Usage: julia write_sweep_metadata.jl <results_dir> <project_dir> <max_jobs>")
end

const RESULTS_DIR = ARGS[1]
const PROJECT_DIR = ARGS[2]
const MAX_JOBS    = parse(Int, ARGS[3])

include(joinpath(dirname(@__FILE__), "sweep_config.jl"))

mkpath(RESULTS_DIR)
const OUTFILE = joinpath(RESULTS_DIR, "sweep_metadata.h5")

function git_sha(project_dir)
    try strip(read(`git -C $project_dir rev-parse HEAD`, String))
    catch; "unknown" end
end

function git_is_dirty(project_dir)
    try !isempty(strip(read(`git -C $project_dir status --porcelain`, String)))
    catch; false end
end

h5open(OUTFILE, "w") do fid
    g = create_group(fid, "metadata")

    g["N_LHS"]     = N_LHS
    g["LHS_SEED"]  = LHS_SEED
    g["USE_LHS"]   = USE_LHS
    g["n_hp"]      = length(HP_GRID)
    g["arch_mode"] = "LuxMLPHardwiredIC"

    gb = create_group(g, "HP_LOG_BOUNDS")
    for (k, v) in pairs(HP_LOG_BOUNDS)
        gb[string(k)] = Float32[v[1], v[2]]
    end

    g["MAXITERS_S1"] = MAXITERS_S1
    g["MAXITERS_S2"] = MAXITERS_S2

    g["julia_version"] = string(VERSION)
    g["timestamp"]     = Dates.format(now(), "yyyy-mm-dd HH:MM:SS")
    g["max_jobs"]      = MAX_JOBS

    g["git_sha"]   = git_sha(PROJECT_DIR)
    g["git_dirty"] = git_is_dirty(PROJECT_DIR)
end

println("Wrote sweep metadata → $OUTFILE")
