#!/usr/bin/env bash
# =============================================================================
# run_full_sweep.sh  —  Complete experimental matrix for the report
#
# Runs every tier × method that the paper reports, in order:
#
#   1. FP16 baseline             (no quantization, ceiling PPL)
#   2. NF4-naive  @ W4A16KV16    (bitsandbytes only — no DartQuant rotations)
#   3. Comparison @ W4A16KV16    (whip / swd_unif / swd_gauss + full rotations)
#   4. Comparison @ W4A4KV4      (whip / swd_unif / swd_gauss + full rotations)
#
# Note: bitsandbytes NF4 is weight-only by design (QLoRA convention), so the
# NF4-naive baseline only exists at W4A16KV16. There is no W4A4KV4 NF4-naive.
#
# Each tier writes to its own subdirectory under OUTPUT_ROOT, so a failure
# in one tier does not invalidate the others, and --resume works per tier.
#
# Usage:
#   bash scripts/run_full_sweep.sh                          # auto-detect models
#   bash scripts/run_full_sweep.sh --models meta-llama/Llama-3.2-1B
#   bash scripts/run_full_sweep.sh --dry-run
#
# Extra flags are forwarded to every run_all_experiments.py invocation.
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
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiment_results_full_sweep}"

# ---------------------------------------------------------------------------
# Dependency check — bitsandbytes is required for NF4
# ---------------------------------------------------------------------------
if ! python3 -c "import bitsandbytes" >/dev/null 2>&1; then
    echo "[ERROR] bitsandbytes is not installed."
    echo "        Install it with: pip install 'bitsandbytes>=0.41.0'"
    exit 1
fi

echo "============================================================"
echo "  DartQuant v2 — FULL experimental sweep"
echo "============================================================"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Runner       : ${RUNNER}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  Tiers        : 1) FP16  2) NF4-naive(W4A16KV16)"
echo "                 3) Comparison@W4A16KV16  4) Comparison@W4A4KV4"
echo "============================================================"

cd "${PROJECT_ROOT}"

# ---------------------------------------------------------------------------
# Tier 1: FP16 baseline (unquantized ceiling)
# ---------------------------------------------------------------------------
echo ""
echo "[1/4] FP16 baseline (no quantization) ..."
python3 "${RUNNER}" \
    --group       baseline \
    --output_root "${OUTPUT_ROOT}/01_fp16_baseline" \
    "$@"

# ---------------------------------------------------------------------------
# Tier 2: NF4-naive baseline (bitsandbytes only, no DartQuant rotations)
# ---------------------------------------------------------------------------
echo ""
echo "[2/4] NF4-naive baseline @ W4A16KV16 (no rotations) ..."
python3 "${RUNNER}" \
    --group       nf4_naive \
    --output_root "${OUTPUT_ROOT}/02_nf4_naive_W4A16KV16" \
    "$@"

# ---------------------------------------------------------------------------
# Tier 3: Weight-only quantization comparison (W4A16KV16)
# ---------------------------------------------------------------------------
echo ""
echo "[3/4] Comparison @ W4A16KV16 (whip / swd_unif / swd_gauss + rotations) ..."
python3 "${RUNNER}" \
    --group       comparison \
    --w4-only \
    --output_root "${OUTPUT_ROOT}/03_comparison_W4A16KV16" \
    "$@"

# ---------------------------------------------------------------------------
# Tier 4: Full quantization comparison (W4A4KV4)
# ---------------------------------------------------------------------------
echo ""
echo "[4/4] Comparison @ W4A4KV4 (whip / swd_unif / swd_gauss + rotations) ..."
python3 "${RUNNER}" \
    --group       comparison \
    --output_root "${OUTPUT_ROOT}/04_comparison_W4A4KV4" \
    "$@"

echo ""
echo "============================================================"
echo "  Full sweep complete. Results under:"
echo "    ${OUTPUT_ROOT}/01_fp16_baseline/"
echo "    ${OUTPUT_ROOT}/02_nf4_naive_W4A16KV16/"
echo "    ${OUTPUT_ROOT}/03_comparison_W4A16KV16/"
echo "    ${OUTPUT_ROOT}/04_comparison_W4A4KV4/"
echo "============================================================"
