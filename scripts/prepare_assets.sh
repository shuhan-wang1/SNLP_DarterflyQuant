#!/usr/bin/env bash
# =============================================================================
# scripts/prepare_assets.sh
#
# Purpose : Pre-download all required models and datasets onto the Myriad
#           Scratch filesystem BEFORE submitting compute jobs.
#           Myriad compute nodes have NO internet access, so everything must
#           be cached here first.
#
# Usage   : Run this script on the Myriad LOGIN node (NOT in a job):
#               bash scripts/prepare_assets.sh
#
#           Optional overrides (export before calling the script):
#               HF_TOKEN=hf_...     HuggingFace access token (required for
#                                   gated models such as Meta-Llama).
#               HF_HOME=/custom/path  Override the cache root (default below).
#               MODELS="model1 model2"  Space-separated list of HF model IDs.
#               SKIP_DATASETS=1     Skip dataset downloads.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------

# Root of the HuggingFace cache on Myriad Scratch.
# All models go to $HF_HOME/hub/  and datasets to $HF_HOME/datasets/
export HF_HOME="${HF_HOME:-/home/ucab327/Scratch/huggingface}"

# HuggingFace access token — required for gated models (Llama, etc.).
# Set HF_TOKEN in your environment or ~/.bashrc:
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
HF_TOKEN="${HF_TOKEN:-}"

# Models to download (space-separated HF model IDs).
# Add or remove entries as needed for your experiments.
MODELS="${MODELS:-meta-llama/Llama-3.2-1B}"

# Datasets to download (must match keys in data_utils._DATASET_CONFIGS).
DATASETS_WIKITEXT="wikitext"           # HF dataset ID
DATASETS_WIKITEXT_CONFIG="wikitext-2-raw-v1"
DATASETS_PTB="ptb_text_only"           # HF dataset ID (no extra config)

# ---------------------------------------------------------------------------
# 2. Environment setup
# ---------------------------------------------------------------------------

echo "============================================================"
echo "  DartQuant — Myriad asset preparation"
echo "============================================================"
echo "  HF_HOME          : $HF_HOME"
echo "  Models           : $MODELS"
echo ""

# Create the cache directory if it does not already exist.
mkdir -p "${HF_HOME}/hub"
mkdir -p "${HF_HOME}/datasets"

# Make sure the Python environment has the required tools.
# If you use a virtual environment, activate it first:
#   source /home/ucab327/Scratch/projects/int4_quantization_darkquant/venv/bin/activate
#
# Verify that huggingface_hub and datasets are importable.
python3 -c "import huggingface_hub; import datasets; import transformers" \
    || { echo "[ERROR] Required Python packages not found."; \
         echo "        Activate your virtualenv and install:"; \
         echo "          pip install huggingface_hub datasets transformers"; \
         exit 1; }

# ---------------------------------------------------------------------------
# 3. Authenticate with HuggingFace (for gated models)
# ---------------------------------------------------------------------------

if [[ -n "$HF_TOKEN" ]]; then
    echo "[INFO] Logging in to HuggingFace Hub..."
    # Store the token so both huggingface-cli and Python API can use it.
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true
else
    echo "[WARN] HF_TOKEN is not set."
    echo "       Gated models (e.g. Meta-Llama) will FAIL without a valid token."
    echo "       Export HF_TOKEN=hf_xxxx before running this script."
fi

# ---------------------------------------------------------------------------
# 4. Download models
# ---------------------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "  Downloading models"
echo "------------------------------------------------------------"

for MODEL_ID in $MODELS; do
    echo ""
    echo "[INFO] Downloading model: $MODEL_ID"

    # huggingface-cli download caches files under $HF_HOME/hub/ automatically.
    # --local-dir-use-symlinks=False avoids symlink issues on some filesystems.
    DOWNLOAD_ARGS=(
        "$MODEL_ID"
        "--cache-dir" "${HF_HOME}/hub"
        "--local-dir-use-symlinks" "False"
    )
    if [[ -n "$HF_TOKEN" ]]; then
        DOWNLOAD_ARGS+=("--token" "$HF_TOKEN")
    fi

    if huggingface-cli download "${DOWNLOAD_ARGS[@]}"; then
        echo "[OK]  $MODEL_ID downloaded successfully."
    else
        echo "[ERROR] Failed to download $MODEL_ID."
        echo "        Check your HF_TOKEN and that you have accepted the model licence at:"
        echo "        https://huggingface.co/$MODEL_ID"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# 5. Download datasets
# ---------------------------------------------------------------------------

if [[ "${SKIP_DATASETS:-0}" == "1" ]]; then
    echo ""
    echo "[INFO] SKIP_DATASETS=1 — skipping dataset downloads."
else
    echo ""
    echo "------------------------------------------------------------"
    echo "  Downloading datasets"
    echo "------------------------------------------------------------"

    # We use a small Python snippet so the datasets library handles the
    # caching layout itself — this guarantees that load_dataset() in the
    # training pipeline finds the files in the expected location.
    python3 - <<PYEOF
import os, sys
os.environ['HF_HOME'] = '${HF_HOME}'
# Allow downloads for this preparation script only.
os.environ['HF_DATASETS_OFFLINE'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

from datasets import load_dataset

cache_dir = os.path.join('${HF_HOME}', 'datasets')
print(f'  Dataset cache dir: {cache_dir}')

# ── wikitext-2-raw-v1 ────────────────────────────────────────────────
print()
print('[INFO] Downloading wikitext / wikitext-2-raw-v1 ...')
for split in ('train', 'validation', 'test'):
    ds = load_dataset(
        '${DATASETS_WIKITEXT}',
        '${DATASETS_WIKITEXT_CONFIG}',
        split=split,
        cache_dir=cache_dir,
    )
    print(f'  [OK] wikitext2 {split}: {len(ds)} examples')

# ── ptb_text_only ────────────────────────────────────────────────────
print()
print('[INFO] Downloading ${DATASETS_PTB} ...')
for split in ('train', 'validation', 'test'):
    ds = load_dataset(
        '${DATASETS_PTB}',
        split=split,
        cache_dir=cache_dir,
    )
    print(f'  [OK] ptb {split}: {len(ds)} examples')

print()
print('[OK] All datasets downloaded.')
PYEOF

fi  # SKIP_DATASETS

# ---------------------------------------------------------------------------
# 6. Verification
# ---------------------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "  Verification"
echo "------------------------------------------------------------"

python3 - <<PYEOF
import os, sys
os.environ['HF_HOME'] = '${HF_HOME}'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

errors = []

# --- Check models ---
from transformers import AutoTokenizer
for model_id in '${MODELS}'.split():
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir='${HF_HOME}',
            local_files_only=True,
        )
        print(f'[OK]  Tokenizer loaded: {model_id}')
    except Exception as e:
        errors.append(f'[FAIL] Tokenizer not found for {model_id}: {e}')

# --- Check datasets ---
if '${SKIP_DATASETS:-0}' != '1':
    from datasets import load_dataset
    cache_dir = os.path.join('${HF_HOME}', 'datasets')
    for name, hf_id, hf_cfg in [
        ('wikitext2', '${DATASETS_WIKITEXT}', '${DATASETS_WIKITEXT_CONFIG}'),
        ('ptb',       '${DATASETS_PTB}',      None),
    ]:
        try:
            if hf_cfg:
                ds = load_dataset(hf_id, hf_cfg, split='train', cache_dir=cache_dir)
            else:
                ds = load_dataset(hf_id, split='train', cache_dir=cache_dir)
            print(f'[OK]  Dataset loaded: {name} ({len(ds)} train examples)')
        except Exception as e:
            errors.append(f'[FAIL] Dataset not cached for {name}: {e}')

if errors:
    print()
    print('The following assets are MISSING:')
    for err in errors:
        print(' ', err)
    sys.exit(1)
else:
    print()
    print('All assets verified — ready to submit jobs!')
PYEOF

echo ""
echo "============================================================"
echo "  Preparation complete."
echo "  You can now submit jobs with:"
echo "    qsub submit_job.sh"
echo "============================================================"
