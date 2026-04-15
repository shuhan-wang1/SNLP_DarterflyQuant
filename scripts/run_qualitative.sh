#!/usr/bin/env bash
# =============================================================================
# run_qualitative.sh  -  Capture down_proj-input activations under the four
# qualitative conditions (raw / whip / swd_unif / swd_gauss) for a single
# model, then render the 4 publication-quality figures.
#
# Steps:
#   1. For each config in {raw, whip, swd_unif, swd_gauss}, invoke
#      scripts/capture_qualitative_activations.py, writing to
#        ${OUTPUT_ROOT}/${short_model}/${config}/down_proj_inputs.npz
#   2. Invoke scripts/plot_activation_qualitative.py on the four captures,
#      writing activation_{main,histogram,absmax_curve,variance_heatmap}.pdf
#      into ${FIG_DIR} (default: report_writing/figures).
#
# Defaults (override via env vars):
#   MODEL        meta-llama/Llama-3.2-1B        (smallest = fastest R1 training)
#   OUTPUT_ROOT  artifacts/qualitative
#   FIG_DIR      report_writing/figures
#   NSAMPLES     32                             (R1 training + hook forward)
#   SEQLEN       2048
#   SEED         0
#   CONFIGS      "raw whip swd_unif swd_gauss"
#
# Any extra CLI args after the recognised flags are forwarded verbatim to
# capture_qualitative_activations.py (e.g. --cache_dir, --hf_token, --dtype).
#
# Usage:
#   bash scripts/run_qualitative.sh
#   bash scripts/run_qualitative.sh --dry-run
#   bash scripts/run_qualitative.sh --force         # re-run even if done
#   MODEL=meta-llama/Llama-3.2-3B bash scripts/run_qualitative.sh
#   CONFIGS="whip swd_unif" bash scripts/run_qualitative.sh   # partial sweep
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CAPTURE="${SCRIPT_DIR}/capture_qualitative_activations.py"
PLOT="${SCRIPT_DIR}/plot_activation_qualitative.py"

for f in "${CAPTURE}" "${PLOT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] required script not found: ${f}" >&2
        exit 1
    fi
done

MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/artifacts/qualitative}"
FIG_DIR="${FIG_DIR:-${PROJECT_ROOT}/report_writing/figures}"
NSAMPLES="${NSAMPLES:-32}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"
CONFIGS="${CONFIGS:-raw whip swd_unif swd_gauss}"

DRY_RUN=0
FORCE=0
SKIP_PLOT=0
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN=1;   shift ;;
        --force)    FORCE=1;     shift ;;
        --no-plot)  SKIP_PLOT=1; shift ;;
        -h|--help)
            sed -n '2,35p' "$0"; exit 0 ;;
        *) FORWARD_ARGS+=("$1"); shift ;;
    esac
done

short_name() {
    local m="$1"
    m="${m##*/}"
    echo "${m,,}"
}

SHORT="$(short_name "${MODEL}")"
MODEL_OUT="${OUTPUT_ROOT}/${SHORT}"

echo "============================================================"
echo "  Qualitative activation sweep"
echo "============================================================"
echo "  model        : ${MODEL}"
echo "  output root  : ${MODEL_OUT}"
echo "  fig dir      : ${FIG_DIR}"
echo "  configs      : ${CONFIGS}"
echo "  nsamples     : ${NSAMPLES}"
echo "  seqlen       : ${SEQLEN}"
echo "  seed         : ${SEED}"
if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
    echo "  fwd args     : ${FORWARD_ARGS[*]}"
fi
echo "  dry-run      : ${DRY_RUN}"
echo "  force re-run : ${FORCE}"
echo "  skip plot    : ${SKIP_PLOT}"
echo "============================================================"

mkdir -p "${MODEL_OUT}"
PASSED=()
SKIPPED=()
FAILED=()

# shellcheck disable=SC2206
CONFIG_LIST=(${CONFIGS})
for CFG in "${CONFIG_LIST[@]}"; do
    case "${CFG}" in
        raw|whip|swd_unif|swd_gauss) ;;
        *) echo "[ERROR] unknown config: ${CFG}" >&2; exit 1 ;;
    esac

    OUT_DIR="${MODEL_OUT}/${CFG}"
    DONE_MARK="${OUT_DIR}/down_proj_inputs.npz"
    mkdir -p "${OUT_DIR}"

    echo
    echo "------------------------------------------------------------"
    echo "  ${MODEL}  /  ${CFG}"
    echo "  -> ${OUT_DIR}"
    echo "------------------------------------------------------------"

    if [[ ${FORCE} -eq 0 && -f "${DONE_MARK}" ]]; then
        echo "  [SKIP] down_proj_inputs.npz already exists; --force to redo"
        SKIPPED+=("${CFG}")
        continue
    fi

    CMD=(python3 "${CAPTURE}"
        --model    "${MODEL}"
        --config   "${CFG}"
        --out_dir  "${OUT_DIR}"
        --nsamples "${NSAMPLES}"
        --seqlen   "${SEQLEN}"
        --seed     "${SEED}")
    if [[ ${#FORWARD_ARGS[@]} -gt 0 ]]; then
        CMD+=("${FORWARD_ARGS[@]}")
    fi

    echo "  \$ ${CMD[*]}"
    if [[ ${DRY_RUN} -eq 1 ]]; then
        SKIPPED+=("${CFG} (dry-run)")
        continue
    fi

    if "${CMD[@]}"; then
        PASSED+=("${CFG}")
    else
        echo "  [FAIL] ${CFG} exited non-zero"
        FAILED+=("${CFG}")
    fi
done

echo
echo "============================================================"
echo "  Capture phase complete"
echo "============================================================"
echo "  passed  (${#PASSED[@]}):"
for c in "${PASSED[@]}";  do echo "    OK    ${c}"; done
echo "  skipped (${#SKIPPED[@]}):"
for c in "${SKIPPED[@]}"; do echo "    SKIP  ${c}"; done
echo "  failed  (${#FAILED[@]}):"
for c in "${FAILED[@]}";  do echo "    FAIL  ${c}"; done

if [[ ${#FAILED[@]} -gt 0 ]]; then
    exit 1
fi

if [[ ${SKIP_PLOT} -eq 1 || ${DRY_RUN} -eq 1 ]]; then
    exit 0
fi

# ---------------------------------------------------------------------------
# Verify every config has a capture file before plotting
# ---------------------------------------------------------------------------
for CFG in raw whip swd_unif swd_gauss; do
    if [[ ! -f "${MODEL_OUT}/${CFG}/down_proj_inputs.npz" ]]; then
        echo "  [WARN] missing capture for ${CFG}; skipping plot step" >&2
        exit 0
    fi
done

echo
echo "------------------------------------------------------------"
echo "  Rendering figures"
echo "  in : ${MODEL_OUT}"
echo "  out: ${FIG_DIR}"
echo "------------------------------------------------------------"

mkdir -p "${FIG_DIR}"
python3 "${PLOT}" --in_dir "${MODEL_OUT}" --out_dir "${FIG_DIR}"
echo "Done."
