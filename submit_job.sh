#!/usr/bin/env bash
# =============================================================================
# submit_job.sh  —  UCL Myriad SGE job submission script for DartQuant v2
#
# Submit with:
#   qsub submit_job.sh
#
# Override experiment parameters without editing this file:
#   qsub -v MODEL=meta-llama/Llama-3.2-1B,LOSS=kl_unif submit_job.sh
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SGE directives
# ─────────────────────────────────────────────────────────────────────────────

# Job name (appears in qstat output)
#$ -N dartquant_v2

# Run in the current shell environment (inherits $PATH, modules, etc.)
#$ -V

# Merge stdout and stderr into one file (easier to read logs)
#$ -j y

# Redirect combined output to the project's log directory.
# $JOB_ID is substituted by SGE at runtime.
#$ -o /home/ucab327/Scratch/projects/int4_quantization_darkquant/logs/job_$JOB_ID.log

# Set the working directory to the project root.
# All relative paths in the Python scripts resolve from here.
#$ -wd /home/ucab327/Scratch/projects/int4_quantization_darkquant

# Wall-clock time limit: 4 hours (hh:mm:ss)
#$ -l h_rt=4:00:00

# Memory per slot (RAM allocated to the job)
#$ -l mem=16G

# Request 1 GPU
#$ -l gpu=1

# Number of CPU slots (1 is usually enough for single-GPU work)
#$ -pe smp 1

# ─────────────────────────────────────────────────────────────────────────────
# Experiment parameters
# (override at submit time with -v VAR=value)
# ─────────────────────────────────────────────────────────────────────────────

MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"
LOSS="${LOSS:-swd_unif}"
QUANTIZER_TYPE="${QUANTIZER_TYPE:-int4}"
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
CAL_DATASET="${CAL_DATASET:-wikitext2}"
NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-2048}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/ucab327/Scratch/projects/int4_quantization_darkquant/outputs}"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Print job metadata (useful for debugging logs)
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================================"
echo "  DartQuant v2 — Myriad SGE Job"
echo "============================================================"
echo "  Job ID       : $JOB_ID"
echo "  Host         : $(hostname)"
echo "  Start time   : $(date)"
echo "  Model        : $MODEL"
echo "  Loss         : $LOSS"
echo "  Quantizer    : $QUANTIZER_TYPE"
echo "  W/A bits     : W${W_BITS}A${A_BITS}"
echo "  Cal dataset  : $CAL_DATASET  (n=$NSAMPLES, seqlen=$SEQLEN)"
echo "  Output dir   : $OUTPUT_DIR"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load required environment modules
# ─────────────────────────────────────────────────────────────────────────────

# Load Python 3.11 (adjust version to match what's available on Myriad:
#   module avail python  →  lists all installed versions)
module load python3/3.11

# Load CUDA toolkit (match the version your PyTorch was compiled against;
#   check with: python -c "import torch; print(torch.version.cuda)")
module load cuda/12.1.0

# Print loaded modules for the log
echo ""
echo "[INFO] Loaded modules:"
module list 2>&1

# ─────────────────────────────────────────────────────────────────────────────
# 3. Activate the project's Python virtual environment
# ─────────────────────────────────────────────────────────────────────────────

VENV_PATH="/home/ucab327/Scratch/projects/int4_quantization_darkquant/venv"

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
    echo "[ERROR] Virtual environment not found at ${VENV_PATH}"
    echo "        Create it on the login node first:"
    echo "          python3 -m venv ${VENV_PATH}"
    echo "          source ${VENV_PATH}/bin/activate"
    echo "          pip install -r requirements.txt"
    exit 1
fi

source "${VENV_PATH}/bin/activate"
echo ""
echo "[INFO] Python : $(which python3)"
echo "[INFO] Pip    : $(pip --version)"

# ─────────────────────────────────────────────────────────────────────────────
# 4. HuggingFace offline environment variables
#
#    HF_HOME          — root cache directory (models in $HF_HOME/hub/,
#                       datasets in $HF_HOME/datasets/)
#    TRANSFORMERS_OFFLINE=1  — prevents transformers from touching the network
#    HF_DATASETS_OFFLINE=1   — prevents datasets from touching the network
#
#    These must be exported BEFORE any Python import so that the HF libraries
#    pick them up during module initialisation.  run_quantize.py also sets
#    them as a safety net, but exporting here ensures child processes inherit
#    the correct values.
# ─────────────────────────────────────────────────────────────────────────────

export HF_HOME="/home/ucab327/Scratch/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo ""
echo "[INFO] HF_HOME            : ${HF_HOME}"
echo "[INFO] TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"
echo "[INFO] HF_DATASETS_OFFLINE : ${HF_DATASETS_OFFLINE}"

# ─────────────────────────────────────────────────────────────────────────────
# 5. GPU diagnostics
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "[INFO] GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"

# ─────────────────────────────────────────────────────────────────────────────
# 6. Create output directory
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"
mkdir -p "/home/ucab327/Scratch/projects/int4_quantization_darkquant/logs"

# ─────────────────────────────────────────────────────────────────────────────
# 7. Run DartQuant v2
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "[INFO] Launching run_quantize.py ..."
echo ""

python3 dartquant_v2/run_quantize.py \
    --model          "${MODEL}"          \
    --loss           "${LOSS}"           \
    --quantizer_type "${QUANTIZER_TYPE}" \
    --w_bits         "${W_BITS}"         \
    --a_bits         "${A_BITS}"         \
    --cal_dataset    "${CAL_DATASET}"    \
    --nsamples       "${NSAMPLES}"       \
    --seqlen         "${SEQLEN}"         \
    --output_dir     "${OUTPUT_DIR}"     \
    --save_rotations

EXIT_CODE=$?

# ─────────────────────────────────────────────────────────────────────────────
# 8. Job summary
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Job finished"
echo "  End time  : $(date)"
echo "  Exit code : ${EXIT_CODE}"
echo "  Output    : ${OUTPUT_DIR}"
echo "============================================================"

exit ${EXIT_CODE}
