#!/usr/bin/env bash
# =============================================================================
# run_sweep2.sh  —  Sweep half #2: large Llama-3.1 family (8B)
#
# Runs the NF4-naive baseline @ W4A4KV4 only (by default) on the 8B models.
# Comparison and FP16 tiers are skipped — those are already done.
# Override NF4_TIERS to run additional tiers if ever needed.
#
# Default model set (override with MODELS_SWEEP2 env var):
#   meta-llama/Llama-3.1-8B
#   meta-llama/Llama-3.1-8B-Instruct
#
# Usage:
#   bash scripts/run_sweep2.sh                       # default models
#   bash scripts/run_sweep2.sh --dry-run             # preview commands
#   bash scripts/run_sweep2.sh --resume              # resume per-tier
#   MODELS_SWEEP2="meta-llama/Llama-3.1-8B" \
#       bash scripts/run_sweep2.sh                   # only the base 8B
#   OUTPUT_ROOT=/data/sweep2 bash scripts/run_sweep2.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Default model set for sweep #2: large Llama-3.1 family
# Override by exporting MODELS_SWEEP2="space separated model ids"
# ---------------------------------------------------------------------------
DEFAULT_MODELS=(
    meta-llama/Llama-3.1-8B
    meta-llama/Llama-3.1-8B-Instruct
)
if [[ -n "${MODELS_SWEEP2:-}" ]]; then
    read -ra DEFAULT_MODELS <<< "${MODELS_SWEEP2}"
fi

# Distinct output root so sweep1 and sweep2 outputs never collide,
# even if both servers share a network filesystem.
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiment_results_sweep2}"

echo "============================================================"
echo "  DartQuant v2 — SWEEP #2 (large models)"
echo "============================================================"
echo "  Models       :"
for m in "${DEFAULT_MODELS[@]}"; do
    echo "    - ${m}"
done
echo "  Output root  : ${OUTPUT_ROOT}"
echo "============================================================"

# Delegate to the full-sweep driver with the pinned model list.
# Extra CLI args ("$@") are forwarded — safe extras: --dry-run, --resume,
# --lm_eval, --nsamples, --seqlen. Avoid --models / --group / --w4-only /
# --output_root (already set by the wrapper).
bash "${SCRIPT_DIR}/run_full_sweep.sh" --models "${DEFAULT_MODELS[@]}" "$@"
