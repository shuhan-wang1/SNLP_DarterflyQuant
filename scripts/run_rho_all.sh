#!/usr/bin/env bash
# =============================================================================
# run_rho_all.sh  -  Batch runner for measure_rho_rope.py across the model set
#
# Measures pre-RoPE intra-band correlation rho_k and variance heterogeneity
# sigma_1^2/sigma_2^2 on each model in turn, so the butterfly-motivation
# appendix can be backed by real data across the full Llama-3 family.
#
# Default models (override with MODELS env var):
#   meta-llama/Llama-3.2-1B
#   meta-llama/Llama-3.2-1B-Instruct
#   meta-llama/Llama-3.2-3B
#   meta-llama/Llama-3.2-3B-Instruct
#   meta-llama/Llama-3.1-8B
#   meta-llama/Llama-3.1-8B-Instruct
#
# Usage:
#   bash scripts/run_rho_all.sh                         # all default models
#   bash scripts/run_rho_all.sh --dry-run               # preview commands
#   bash scripts/run_rho_all.sh --force                 # re-run even if done
#   MODELS="meta-llama/Llama-3.2-1B meta-llama/Llama-3.1-8B" \
#       bash scripts/run_rho_all.sh                     # custom list
#   OUTPUT_ROOT=/data/rho bash scripts/run_rho_all.sh   # custom root
#   NSAMPLES=64 SEQLEN=1024 bash scripts/run_rho_all.sh # lighter budget
#
# Any extra args are forwarded verbatim to measure_rho_rope.py (e.g.
# --cache_dir, --dtype, --max_layers, --hf_token).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MEASURE="${SCRIPT_DIR}/measure_rho_rope.py"

if [[ ! -f "${MEASURE}" ]]; then
    echo "[ERROR] measure_rho_rope.py not found at ${MEASURE}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Defaults (override via env vars)
# ---------------------------------------------------------------------------
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/rho}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"

DEFAULT_MODELS=(
    "meta-llama/Llama-3.2-1B"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B-Instruct"
    "meta-llama/Llama-3.1-8B"
    "meta-llama/Llama-3.1-8B-Instruct"
)
if [[ -n "${MODELS:-}" ]]; then
    # shellcheck disable=SC2206
    MODEL_LIST=(${MODELS})
else
    MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

# ---------------------------------------------------------------------------
# Parse our own flags; the rest is forwarded to measure_rho_rope.py
# ---------------------------------------------------------------------------
DRY_RUN=0
FORCE=0
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --force)   FORCE=1;   shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  rho / variance sweep across models"
echo "============================================================"
echo "  Project root : ${PROJECT_ROOT}"
echo "  Runner       : ${MEASURE}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  nsamples     : ${NSAMPLES}"
echo "  seqlen       : ${SEQLEN}"
echo "  seed         : ${SEED}"
echo "  models       : ${#MODEL_LIST[@]}"
for m in "${MODEL_LIST[@]}"; do echo "                 - ${m}"; done
if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    echo "  fwd args     : ${FORWARD_ARGS[*]}"
fi
echo "  dry-run      : ${DRY_RUN}"
echo "  force re-run : ${FORCE}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Helper: derive a filesystem-safe short name for a model id
# ---------------------------------------------------------------------------
short_name() {
    # "meta-llama/Llama-3.2-1B-Instruct" -> "llama-3.2-1b-instruct"
    local m="$1"
    m="${m##*/}"
    # lowercase
    echo "${m,,}"
}

# ---------------------------------------------------------------------------
# Run loop with pass/fail tracking
# ---------------------------------------------------------------------------
mkdir -p "${OUTPUT_ROOT}"
PASSED=()
SKIPPED=()
FAILED=()

for MODEL in "${MODEL_LIST[@]}"; do
    OUT_DIR="${OUTPUT_ROOT}/$(short_name "${MODEL}")"
    DONE_MARK="${OUT_DIR}/summary.txt"

    echo
    echo "------------------------------------------------------------"
    echo "  ${MODEL}"
    echo "  -> ${OUT_DIR}"
    echo "------------------------------------------------------------"

    if [[ ${FORCE} -eq 0 && -f "${DONE_MARK}" ]]; then
        echo "  [SKIP] summary.txt already exists; use --force to redo"
        SKIPPED+=("${MODEL}")
        continue
    fi

    CMD=(python3 "${MEASURE}"
        --model     "${MODEL}"
        --nsamples  "${NSAMPLES}"
        --seqlen    "${SEQLEN}"
        --seed      "${SEED}"
        --out_dir   "${OUT_DIR}")
    if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
        CMD+=("${FORWARD_ARGS[@]}")
    fi

    echo "  \$ ${CMD[*]}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        SKIPPED+=("${MODEL} (dry-run)")
        continue
    fi

    if "${CMD[@]}"; then
        PASSED+=("${MODEL}")
    else
        echo "  [FAIL] ${MODEL} exited non-zero"
        FAILED+=("${MODEL}")
        # continue with the next model instead of aborting the whole sweep
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "  Sweep complete"
echo "============================================================"
echo "  passed  (${#PASSED[@]}):"
for m in "${PASSED[@]}";  do echo "    OK    ${m}"; done
echo "  skipped (${#SKIPPED[@]}):"
for m in "${SKIPPED[@]}"; do echo "    SKIP  ${m}"; done
echo "  failed  (${#FAILED[@]}):"
for m in "${FAILED[@]}";  do echo "    FAIL  ${m}"; done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi
