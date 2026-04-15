#!/usr/bin/env bash
# =============================================================================
# run_full_sweep.sh  —  NF4-naive baseline sweep (no comparison tiers)
#
# Runs ONLY the pure NF4 / bitsandbytes baseline (no DartQuant rotations).
# The comparison group (whip / swd_unif / swd_gauss + rotations) and the
# FP16 ceiling are NOT re-run here — they've already been measured.
#
# Default tier: W4A4KV4 only. The W4A16KV16 tier is already complete for
# the cached models, so by default we only run the remaining W4A4KV4 cell.
#
# Override the tier selection:
#   NF4_TIERS="W4A16KV16 W4A4KV4"  bash scripts/run_full_sweep.sh   # both
#   NF4_TIERS="W4A16KV16"           bash scripts/run_full_sweep.sh   # only wt-only
#
# Usage:
#   bash scripts/run_full_sweep.sh                       # all cached models, W4A4KV4
#   bash scripts/run_full_sweep.sh --models meta-llama/Llama-3.2-1B
#   bash scripts/run_full_sweep.sh --dry-run
#   bash scripts/run_full_sweep.sh --resume
#
# Extra flags forwarded to run_all_experiments.py — safe extras:
#   --models, --resume, --dry-run, --lm_eval, --nsamples, --seqlen
# Avoid: --group, --nf4-tiers, --output_root (already set here).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_all_experiments.py"

# ---------------------------------------------------------------------------
# Output directory (override by exporting OUTPUT_ROOT)
# ---------------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiment_results_nf4_naive}"

# ---------------------------------------------------------------------------
# Tier selection (override by exporting NF4_TIERS)
# Default: W4A4KV4 only (W4A16KV16 is already complete)
# ---------------------------------------------------------------------------
NF4_TIERS="${NF4_TIERS:-W4A4KV4}"
read -ra NF4_TIERS_ARR <<< "${NF4_TIERS}"

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------
if ! python3 -c "import bitsandbytes" >/dev/null 2>&1; then
    echo "[ERROR] bitsandbytes is not installed."
    echo "        Install it with: pip install 'bitsandbytes>=0.41.0'"
    exit 1
fi

echo "============================================================"
echo "  DartQuant v2 — NF4-naive sweep (no comparison)"
echo "============================================================"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Runner       : ${RUNNER}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  NF4 tiers    : ${NF4_TIERS_ARR[*]}"
echo "  Comparison   : SKIPPED by default (already measured)"
echo "============================================================"

cd "${PROJECT_ROOT}"

python3 "${RUNNER}" \
    --group       nf4_naive \
    --nf4-tiers   "${NF4_TIERS_ARR[@]}" \
    --output_root "${OUTPUT_ROOT}" \
    "$@"

echo ""
echo "============================================================"
echo "  Sweep complete. Results under: ${OUTPUT_ROOT}"
echo "============================================================"
