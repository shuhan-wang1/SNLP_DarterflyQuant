#!/usr/bin/env bash
# Sweep one PDF per layer comparing raw vs. R1 trained with
# {whip, swd_unif, swd_gauss}.
#
# One Python run does everything:
#   - load model once
#   - hook every selected layer in ONE forward pass
#   - train one global R1 per loss on pooled activations (matches the
#     DartQuant pipeline; pass --per_layer_r1 to train per-layer instead)
#   - write one <out_dir>/layer_<idx>.pdf per layer
#
# Usage:
#   scripts/run_compare_activations.sh [MODEL] [OUT_DIR] [LAYERS]
#
# Defaults (autodl-friendly):
#   MODEL   = meta-llama/Llama-3.2-1B
#   OUT_DIR = artifacts/activation_comparison/<model-basename>
#   LAYERS  = all
#
# Examples:
#   scripts/run_compare_activations.sh                                  # all layers, 1B model
#   scripts/run_compare_activations.sh meta-llama/Llama-3.1-8B
#   scripts/run_compare_activations.sh meta-llama/Llama-3.2-1B artifacts/foo 0-15:2
#
# Env overrides (propagated to the Python script):
#   HF_HOME              default /root/autodl-tmp/huggingface
#   HF_DATASETS_CACHE    default /root/autodl-tmp/datasets
#   TRANSFORMERS_OFFLINE default 1 (set to 0 on a login node to allow downloads)
#   PYTHON               interpreter to use (default: python)
#   EXTRA_ARGS           appended verbatim to the python call (e.g. --per_layer_r1)
#
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.2-1B}"
_MODEL_TAG="$(basename "${MODEL}")"
OUT_DIR="${2:-artifacts/activation_comparison/${_MODEL_TAG}}"
LAYERS="${3:-all}"

PYTHON="${PYTHON:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-tmp/datasets}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "  model      : ${MODEL}"
echo "  layers     : ${LAYERS}"
echo "  out_dir    : ${OUT_DIR}"
echo "  HF_HOME    : ${HF_HOME}"
echo "  offline    : TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
echo "  python     : ${PYTHON}"
echo "============================================================"

"${PYTHON}" scripts/compare_activations.py \
    --model     "${MODEL}" \
    --layers    "${LAYERS}" \
    --out_dir   "${OUT_DIR}" \
    ${EXTRA_ARGS:-}

echo ""
echo "done → ${OUT_DIR}"
ls -1 "${OUT_DIR}" | sed 's/^/  /'
