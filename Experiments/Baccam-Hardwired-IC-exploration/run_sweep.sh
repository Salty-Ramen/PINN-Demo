#!/usr/bin/env bash
#
# run_sweep.sh — Composite transform (Log₁₀ + ZScore) hyperparameter sweep
#
# Orchestrates the sweep via GNU Parallel.  Each (config, HP) pair
# runs as an independent Julia process writing a per-worker HDF5.
#
# Usage:
#   chmod +x run_sweep.sh
#   ./run_sweep.sh              # all available cores
#   ./run_sweep.sh 16           # cap at 16 concurrent jobs
#   ./run_sweep.sh 16 resume    # resume after interruption

set -euo pipefail

# ── Resolve paths ─────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_DIR="${SCRIPT_DIR}/Results"
ROW_DIR="${RESULTS_DIR}/sweep_rows"
JOBLIST="${RESULTS_DIR}/joblist.txt"
JOBLOG="${RESULTS_DIR}/sweep_joblog.txt"
SWEEP_CFG="${SCRIPT_DIR}/sweep_config.jl"

MAX_JOBS="${1:-$(nproc)}"
RESUME_FLAG="${2:-}"

mkdir -p "${ROW_DIR}"

# ── Dependency checks ─────────────────────────────────────
command -v parallel >/dev/null 2>&1 || {
    echo "ERROR: GNU Parallel not found. Install with:"
    echo "  Ubuntu/Debian : sudo apt install parallel"
    echo "  macOS         : brew install parallel"
    echo "  Conda         : conda install -c conda-forge parallel"
    exit 1
}

command -v julia >/dev/null 2>&1 || {
    echo "ERROR: Julia not found in PATH."
    exit 1
}

# ── Precompile ────────────────────────────────────────────
echo "Precompiling project dependencies..."
julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# ── Generate training data ────────────────────────────────
if [[ ! -f "${RESULTS_DIR}/Baccam_A_5x24.csv" ]] || \
   [[ ! -f "${RESULTS_DIR}/Baccam_B_10x12.csv" ]] || \
   [[ ! -f "${RESULTS_DIR}/Baccam_C_20x6.csv" ]]; then
    echo "Generating training data into ${RESULTS_DIR}..."
    (cd "${PROJECT_DIR}" && julia --project="${PROJECT_DIR}" \
        "${SCRIPT_DIR}/generate_sweep_data.jl" "${RESULTS_DIR}")
else
    echo "Training data CSVs already exist — skipping generation."
fi

# ── Build job list ────────────────────────────────────────
HP_COUNT=$(cd "${PROJECT_DIR}" && julia --project="${PROJECT_DIR}" -e "
    include(\"${SWEEP_CFG}\")
    print(length(HP_GRID))
")

echo "Generating job list (${HP_COUNT} HP combinations × 3 configs)..."
> "${JOBLIST}"
for cfg in A_5x24 B_10x12 C_20x6; do
    for hp in $(seq 1 "${HP_COUNT}"); do
        echo "${cfg} ${hp}"
    done
done >> "${JOBLIST}"

TOTAL=$(wc -l < "${JOBLIST}")
echo "Total jobs: ${TOTAL}   Concurrent workers: ${MAX_JOBS}"

# ── GNU Parallel options ──────────────────────────────────
PARALLEL_OPTS=(
    --jobs "${MAX_JOBS}"
    --colsep ' '
    --joblog "${JOBLOG}"
    --progress
    --bar
    --timeout 7200
    --retries 1
)

[[ "${RESUME_FLAG}" == "resume" ]] && {
    echo "Resuming from previous joblog..."
    PARALLEL_OPTS+=(--resume-failed)
}

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Launching CompositeTransform sweep — $(date)"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Metadata ──────────────────────────────────────────────
(cd "${PROJECT_DIR}" && julia --project="${PROJECT_DIR}" \
    "${SCRIPT_DIR}/write_sweep_metadata.jl" \
    "${RESULTS_DIR}" "${PROJECT_DIR}" "${MAX_JOBS}")

# ── Parallel sweep ────────────────────────────────────────
parallel "${PARALLEL_OPTS[@]}" \
    "cd ${PROJECT_DIR} && julia --project=${PROJECT_DIR} \
          ${SCRIPT_DIR}/run_single_baccam.jl {1} {2} ${SCRIPT_DIR}" \
    :::: "${JOBLIST}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  CompositeTransform sweep complete — $(date)"
echo ""
echo "  Merge & analyse with:"
echo "    cd ${PROJECT_DIR}"
echo "    julia --project=. ${SCRIPT_DIR}/Baccam_sweep_analysis.jl"
echo "═══════════════════════════════════════════════════"
