#!/usr/bin/env bash
# =============================================================================
# run_nf4_naive.sh  —  Pure NF4 (bitsandbytes) baseline, NO DartQuant rotations
#
# Runs the `nf4_naive` experiment group only:
#   * weight-only NF4 quantization via bitsandbytes (W4A16KV16)
#   * R1, R2, R3, R4 all DISABLED — exactly the standard QLoRA-style setup
#
# Purpose: isolates the contribution of DartQuant's orthogonal rotations.
# Compared against `swd_gauss_nf4` (NF4 + full rotations), this answers:
#   "Does DartQuant's rotation training actually help NF4, or is NF4 alone
#    already good enough?"
#
# Usage:
#   bash scripts/run_nf4_naive.sh                          # auto-detect models
#   bash scripts/run_nf4_naive.sh --models meta-llama/Llama-3.2-1B
#   bash scripts/run_nf4_naive.sh --dry-run
#
# Any extra arguments are forwarded verbatim to run_all_experiments.py.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve paths from this script's location so the script works regardless
# of the caller's CWD.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${SCRIPT_DIR}/run_all_experiments.py"

# ---------------------------------------------------------------------------
# Output directory (override by exporting OUTPUT_ROOT before invoking)
# ---------------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiment_results_nf4_naive}"

# ---------------------------------------------------------------------------
# Dependency check — bitsandbytes is required for NF4
# ---------------------------------------------------------------------------
if ! python3 -c "import bitsandbytes" >/dev/null 2>&1; then
    echo "[ERROR] bitsandbytes is not installed."
    echo "        Install it with: pip install 'bitsandbytes>=0.41.0'"
    exit 1
fi

echo "============================================================"
echo "  DartQuant v2 — NF4-naive baseline (no rotations)"
echo "============================================================"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Runner       : ${RUNNER}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  Group        : nf4_naive"
echo "  Quantizer    : NF4 (bitsandbytes), weight-only W4A16KV16"
echo "  Rotations    : DISABLED (no R1/R2/R3/R4)"
echo "============================================================"

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Run only the nf4_naive group. Forward any extra CLI flags
# (e.g. --models, --lm_eval, --dry-run, --resume).
# ---------------------------------------------------------------------------
python3 "${RUNNER}" \
    --group       nf4_naive \
    --output_root "${OUTPUT_ROOT}" \
    "$@"
