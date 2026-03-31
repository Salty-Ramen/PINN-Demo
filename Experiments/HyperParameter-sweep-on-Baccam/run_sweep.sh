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

# ── Configuration ─────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${PROJECT_DIR}/Results"
ROW_DIR="${RESULTS_DIR}/sweep_rows"
JOBLIST="${RESULTS_DIR}/joblist.txt"
JOBLOG="${RESULTS_DIR}/sweep_joblog.txt"
MERGED_CSV="${RESULTS_DIR}/Baccam_hp_sweep.csv"

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
# This runs once and produces the CSVs that every worker reads.
# Workers never regenerate data themselves.
if [[ ! -f "${RESULTS_DIR}/Baccam_A_5x24.csv" ]] || \
   [[ ! -f "${RESULTS_DIR}/Baccam_B_10x12.csv" ]] || \
   [[ ! -f "${RESULTS_DIR}/Baccam_C_20x6.csv" ]]; then
    echo "Generating training data..."
    julia --project="${PROJECT_DIR}" "${PROJECT_DIR}/generate_sweep_data.jl"
else
    echo "Training data CSVs already exist — skipping generation."
fi

# ── Generate the job list ─────────────────────────────────
# 3 configs × 243 HP combos = 729 jobs
# Each line: <config_key> <hp_index>
HP_COUNT=243   # 3^5 grid — must match the grid in run_single_baccam.jl

echo "Generating job list..."
> "${JOBLIST}"   # truncate
for cfg in A_5x24 B_10x12 C_20x6; do
    for hp in $(seq 1 ${HP_COUNT}); do
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

# --resume-failed re-runs jobs that previously exited non-zero;
# --resume skips jobs that appear in the joblog at all
if [[ "${RESUME_FLAG}" == "resume" ]]; then
    echo "Resuming from previous joblog..."
    PARALLEL_OPTS+=(--resume-failed)
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Launching sweep — $(date)"
echo "═══════════════════════════════════════════════════"
echo ""

parallel "${PARALLEL_OPTS[@]}" \
    julia --project="${PROJECT_DIR}" \
          "${PROJECT_DIR}/Run-Single-Baccam.jl" {1} {2} \
    :::: "${JOBLIST}"

# ── Merge per-row CSVs into a single file ─────────────────
echo ""
echo "Merging results..."

# Grab the header from the first CSV, then append data rows from all
FIRST_ROW=$(find "${ROW_DIR}" -name 'row_*.csv' | head -1)
if [[ -z "${FIRST_ROW}" ]]; then
    echo "WARNING: No result files found in ${ROW_DIR}."
    exit 1
fi

head -1 "${FIRST_ROW}" > "${MERGED_CSV}"
# tail -n +2 skips the header from each file; -q suppresses filename printing
tail -n +2 -q "${ROW_DIR}"/row_*.csv >> "${MERGED_CSV}"

ROW_COUNT=$(tail -n +2 "${MERGED_CSV}" | wc -l)
echo "Merged ${ROW_COUNT} result rows → ${MERGED_CSV}"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Sweep complete — $(date)"
echo "  Analyse with:  julia --project=. Baccam_sweep_analysis.jl"
echo "═══════════════════════════════════════════════════"
