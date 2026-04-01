#!/usr/bin/env bash
#
# run_sweep.sh
#
# Orchestrates the Baccam hyperparameter sweep using GNU Parallel.
# Each (config, hp_index) pair runs as an independent Julia process.
#
# Usage:
#   chmod +x run_sweep.sh
#   ./run_sweep.sh              # uses all available cores
#   ./run_sweep.sh 16           # cap at 16 concurrent jobs
#   ./run_sweep.sh 16 resume    # resume a previously interrupted sweep

set -euo pipefail

# ── Resolve project root from this script's location ──────
# The script lives at <project>/Experiments/HyperParameter-sweep-on-Baccam/,
# so we walk two levels up to reach the actual project root.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_DIR="${PROJECT_DIR}/Results"
ROW_DIR="${RESULTS_DIR}/sweep_rows"
JOBLIST="${RESULTS_DIR}/joblist.txt"
JOBLOG="${RESULTS_DIR}/sweep_joblog.txt"
MERGED_CSV="${RESULTS_DIR}/Baccam_hp_sweep.csv"

# Sweep config path (used in multiple places below)
SWEEP_CFG="${SCRIPT_DIR}/sweep_config.jl"

MAX_JOBS="${1:-$(nproc)}"        # default: all cores on this machine
RESUME_FLAG="${2:-}"             # pass "resume" as 2nd arg to skip completed jobs

mkdir -p "${ROW_DIR}"

# ── Check dependencies ────────────────────────────────────
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

# ── Precompile once to avoid redundant JIT across workers ─
echo "Precompiling project dependencies (one-time cost)..."
julia --project="${PROJECT_DIR}" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# ── Generate training data if it doesn't already exist ────
if [[ ! -f "${RESULTS_DIR}/Baccam_A_5x24.csv" ]] || \
   [[ ! -f "${RESULTS_DIR}/Baccam_B_10x12.csv" ]] || \
   [[ ! -f "${RESULTS_DIR}/Baccam_C_20x6.csv" ]]; then
    echo "Generating training data..."
    # cd into project root so pwd()-based includes resolve correctly
    (cd "${PROJECT_DIR}" && julia --project="${PROJECT_DIR}" \
        "${SCRIPT_DIR}/generate_sweep_data.jl")
else
    echo "Training data CSVs already exist — skipping generation."
fi

# ── Query HP grid size from sweep_config.jl ───────────────
# cd into project root because sweep_config.jl includes Fitting.jl via pwd()
HP_COUNT=$(cd "${PROJECT_DIR}" && julia --project="${PROJECT_DIR}" -e "
    include(\"${SWEEP_CFG}\")
    print(length(HP_GRID))
")

echo "Generating job list (${HP_COUNT} HP combinations per config)..."
> "${JOBLIST}"   # truncate
for cfg in A_5x24 B_10x12 C_20x6; do
    for hp in $(seq 1 "${HP_COUNT}"); do
        echo "${cfg} ${hp}"
    done
done >> "${JOBLIST}"

TOTAL=$(wc -l < "${JOBLIST}")
echo "Total jobs: ${TOTAL}"
echo "Concurrent workers: ${MAX_JOBS}"

# ── Build the GNU Parallel command ────────────────────────
PARALLEL_OPTS=(
    --jobs "${MAX_JOBS}"
    --colsep ' '
    --joblog "${JOBLOG}"
    --progress
    --bar
    --timeout 7200           # kill any single run that exceeds 2 hours
    --retries 1              # retry once on failure (covers transient OOM etc.)
)

if [[ "${RESUME_FLAG}" == "resume" ]]; then
    echo "Resuming from previous joblog..."
    PARALLEL_OPTS+=(--resume-failed)
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Launching sweep — $(date)"
echo "═══════════════════════════════════════════════════"
echo ""

# Each worker cd's into the project root before launching Julia,
# ensuring that pwd()-based path resolution in the Julia code
# (e.g. joinpath(pwd(), "Src", "Fitting.jl")) finds the right files.
parallel "${PARALLEL_OPTS[@]}" \
    "cd ${PROJECT_DIR} && julia --project=${PROJECT_DIR} \
          ${SCRIPT_DIR}/run_single_baccam.jl {1} {2}" \
    :::: "${JOBLIST}"

# ── Merge per-row CSVs into a single file ─────────────────
echo ""
echo "Merging results..."

FIRST_ROW=$(find "${ROW_DIR}" -name 'row_*.csv' | head -1)
if [[ -z "${FIRST_ROW}" ]]; then
    echo "WARNING: No result files found in ${ROW_DIR}."
    exit 1
fi

head -1 "${FIRST_ROW}" > "${MERGED_CSV}"
tail -n +2 -q "${ROW_DIR}"/row_*.csv >> "${MERGED_CSV}"

ROW_COUNT=$(tail -n +2 "${MERGED_CSV}" | wc -l)
echo "Merged ${ROW_COUNT} result rows → ${MERGED_CSV}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Sweep complete — $(date)"
echo "  Analyse with:  julia --project=. ${SCRIPT_DIR}/Baccam_sweep_analysis.jl"
echo "═══════════════════════════════════════════════════"
