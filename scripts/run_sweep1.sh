#!/usr/bin/env bash
# =============================================================================
# run_sweep1.sh  —  Sweep half #1: small Llama-3.2 family (1B + 3B)
#
# Runs the full 4-tier experimental matrix (FP16 baseline + NF4-naive +
# comparison @ W4A16KV16 + comparison @ W4A4KV4) on the smaller models so
# this server finishes ~2× faster and frees up while sweep2 (8B family)
# continues on the second server.
#
# Default model set (override with MODELS_SWEEP1 env var):
#   meta-llama/Llama-3.2-1B
#   meta-llama/Llama-3.2-1B-Instruct
#   meta-llama/Llama-3.2-3B
#   meta-llama/Llama-3.2-3B-Instruct
#
# Usage:
#   bash scripts/run_sweep1.sh                       # default models
#   bash scripts/run_sweep1.sh --dry-run             # preview commands
#   bash scripts/run_sweep1.sh --resume              # resume per-tier
#   MODELS_SWEEP1="meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B" \
#       bash scripts/run_sweep1.sh                   # custom model list
#   OUTPUT_ROOT=/data/sweep1 bash scripts/run_sweep1.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------------------------------------------------------------------------
# Default model set for sweep #1: small Llama-3.2 family
# Override by exporting MODELS_SWEEP1="space separated model ids"
# ---------------------------------------------------------------------------
DEFAULT_MODELS=(
    meta-llama/Llama-3.2-1B
    meta-llama/Llama-3.2-1B-Instruct
    meta-llama/Llama-3.2-3B
    meta-llama/Llama-3.2-3B-Instruct
)
if [[ -n "${MODELS_SWEEP1:-}" ]]; then
    read -ra DEFAULT_MODELS <<< "${MODELS_SWEEP1}"
fi

# Distinct output root so sweep1 and sweep2 outputs never collide,
# even if both servers share a network filesystem.
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/experiment_results_sweep1}"

echo "============================================================"
echo "  DartQuant v2 — SWEEP #1 (small models)"
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
