#!/usr/bin/env bash
# =============================================================================
# run_qualitative_reuse_r1.sh
#
# End-to-end qualitative capture + plotting that REUSES pre-trained R1
# matrices from previous run_quantize.py experiments (no retraining).
#
# Defaults target meta-llama/Llama-3.2-1B with saved rotations under
#   experiment_results/llama3.2_1B_quant_a4w4kv4/comparison__Llama-3.2-1B__*/rotations.pt
#
# Override targets via env vars:
#   MODEL      meta-llama/Llama-3.2-1B
#   WHIP_PT    absolute/relative path to whip rotations.pt
#   SWDU_PT    absolute/relative path to swd_unif rotations.pt
#   SWDG_PT    absolute/relative path to swd_gauss rotations.pt
#   HOOK       up_proj   (hidden-dim hook where R1 is visible)
#   OUT_ROOT   artifacts/qualitative
#   FIG_DIR    report_writing/figures
#   NSAMPLES   32
#   SEQLEN     2048
#   SEED       0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
BASE_DEFAULT="experiment_results/llama3.2_1B_quant_a4w4kv4"
WHIP_PT="${WHIP_PT:-${BASE_DEFAULT}/comparison__Llama-3.2-1B__whip_int4/rotations.pt}"
SWDU_PT="${SWDU_PT:-${BASE_DEFAULT}/comparison__Llama-3.2-1B__swd_unif_int4/rotations.pt}"
SWDG_PT="${SWDG_PT:-${BASE_DEFAULT}/comparison__Llama-3.2-1B__swd_gauss_nf4/rotations.pt}"

HOOK="${HOOK:-up_proj}"
NSAMPLES="${NSAMPLES:-32}"
SEQLEN="${SEQLEN:-2048}"
SEED="${SEED:-0}"

# Pin the interpreter so nohup children don't inherit a broken PATH.
# Override with e.g. PYTHON=/opt/conda/envs/foo/bin/python ...
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "/root/miniconda3/bin/python" ]]; then
        PYTHON="/root/miniconda3/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON="$(command -v python)"
    else
        echo "[ERROR] no python interpreter found; set PYTHON=..." >&2
        exit 1
    fi
fi

SHORT="$(basename "${MODEL}")"
SHORT_LC="${SHORT,,}"
OUT_ROOT="${OUT_ROOT:-artifacts/qualitative}"
OUT="${OUT_ROOT}/${SHORT_LC}"
FIG_DIR="${FIG_DIR:-report_writing/figures}"

mkdir -p "${OUT}/raw" "${OUT}/whip" "${OUT}/swd_unif" "${OUT}/swd_gauss" \
         "${FIG_DIR}"

echo "============================================================"
echo "  qualitative reuse-R1 pipeline"
echo "============================================================"
echo "  model      : ${MODEL}"
echo "  hook       : ${HOOK}"
echo "  nsamples   : ${NSAMPLES}   seqlen: ${SEQLEN}   seed: ${SEED}"
echo "  whip R1    : ${WHIP_PT}"
echo "  swd_unif R1: ${SWDU_PT}"
echo "  swd_gauss  : ${SWDG_PT}"
echo "  captures   : ${OUT}"
echo "  figures    : ${FIG_DIR}"
echo "  python     : ${PYTHON}"
echo "============================================================"

for f in "${WHIP_PT}" "${SWDU_PT}" "${SWDG_PT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "[ERROR] missing rotations.pt: ${f}" >&2
        exit 1
    fi
done

EXPECTED="${HOOK}_inputs.npz"
if [[ "${HOOK}" == "down_proj" ]]; then
    EXPECTED="down_proj_inputs.npz"
fi

run_capture() {
    local cfg="$1"
    local out_dir="${OUT}/${cfg}"
    local done_mark="${out_dir}/${EXPECTED}"
    shift
    if [[ -f "${done_mark}" ]]; then
        echo "[skip] ${cfg}: ${done_mark} already exists"
        return 0
    fi
    echo "---- capture: ${cfg} ----"
    "${PYTHON}" scripts/capture_qualitative_activations.py \
        --model    "${MODEL}" \
        --config   "${cfg}" \
        --hook     "${HOOK}" \
        --out_dir  "${out_dir}" \
        --nsamples "${NSAMPLES}" \
        --seqlen   "${SEQLEN}" \
        --seed     "${SEED}" \
        "$@"
}

run_capture raw
run_capture whip      --r1_path "${WHIP_PT}"
run_capture swd_unif  --r1_path "${SWDU_PT}"
run_capture swd_gauss --r1_path "${SWDG_PT}"

echo
echo "---- plotting ----"
"${PYTHON}" scripts/plot_rotation_before_after.py \
    --in_dir "${OUT}" \
    --out    "${FIG_DIR}/rotation_before_after.pdf"

"${PYTHON}" scripts/plot_activation_qualitative.py \
    --in_dir  "${OUT}" \
    --out_dir "${FIG_DIR}"

echo
echo "Figures written to: ${FIG_DIR}"
ls -la "${FIG_DIR}"
