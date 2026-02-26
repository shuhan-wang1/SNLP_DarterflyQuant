#!/usr/bin/env python3
"""
Activation Distribution Analysis for LLMs (DartQuant Methodology)
Analyzes activation value distributions across different layers.

─────────────────────────────────────────────────────────────────
UCL Myriad usage
─────────────────────────────────────────────────────────────────
Run this script on the LOGIN NODE (not inside a job) to pre-download
all models and datasets before submitting compute jobs.

  # (Optional) set your HuggingFace token for gated models
  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

  # (Optional) override the cache root (default: Scratch below)
  export HF_HOME=/home/ucab327/Scratch/huggingface

  python scripts/stat_and_download.py

The script automatically sets offline flags for the training run
(TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE) — see run_quantize.py.
─────────────────────────────────────────────────────────────────
"""

import os
import sys
import gc
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# USER CONFIGURATION
# ============================================================================

# HuggingFace access token — required for gated models such as Meta-Llama.
# Set via environment variable (preferred) or replace None with your token.
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# ── Cache directories ────────────────────────────────────────────────────────
# Priority:
#   1. $HF_HOME already exported in the shell  (e.g. by the user or a wrapper)
#   2. autodl default cache directory
#
# The same variable is read by dartquant_v2/run_quantize.py so models and
# datasets downloaded here are found automatically by the training pipeline.
_DEFAULT_HF_HOME = "/root/autodl-tmp/huggingface"
HF_HOME = os.environ.get("HF_HOME", _DEFAULT_HF_HOME)

# Expose to sub-processes and the HuggingFace libraries imported below.
os.environ["HF_HOME"]           = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")

# NOTE: We do NOT set HF_DATASETS_OFFLINE or TRANSFORMERS_OFFLINE here.
#       This script is meant to run on the login node where internet IS
#       available.  The training pipeline (run_quantize.py) sets those
#       flags at job time.

# CACHE_DIR is kept as an alias for legacy code paths in this file.
CACHE_DIR = HF_HOME

# Output directory for activation plots produced by analyze_model()
PLOT_DIR = os.path.join(HF_HOME, "activation_plots")

# Create directories if they don't already exist.
Path(HF_HOME).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

print(f"✓ HF_HOME / model cache : {HF_HOME}")
print(f"✓ Dataset cache         : {os.environ['HF_DATASETS_CACHE']}")
print(f"✓ Plot directory        : {PLOT_DIR}")
print(f"✓ HF token              : {'set ✓' if HF_TOKEN else 'not set (gated models will fail)'}")

# Import libraries
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    print("✓ Libraries imported successfully")
except ImportError as e:
    print(f"✗ Error importing libraries: {e}")
    print("Please install: pip install transformers accelerate datasets matplotlib seaborn")
    sys.exit(1)

# Target models to download and analyse
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-8B-Instruct",
    "meta-llama/Llama-3.2-8B",
]

# Configuration
SEQ_LENGTH = 2048
NUM_SAMPLES = 1000  # Sample 1000 activation values per layer
DTYPE = torch.float16


class ActivationCollector:
    """Collects activation values from specific layers using forward hooks."""

    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.activations: Dict[str, List[float]] = {}
        self.hooks = []

    def get_hook(self, name: str):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # Get input activations (they're usually tuples)
            if isinstance(input, tuple):
                tensor = input[0]
            else:
                tensor = input

            if isinstance(tensor, torch.Tensor):
                # Flatten and sample
                flat = tensor.detach().cpu().float().flatten().numpy()
                if len(flat) > 0:
                    # Randomly sample up to num_samples
                    sample_size = min(self.num_samples, len(flat))
                    sampled = np.random.choice(flat, size=sample_size, replace=False)

                    if name not in self.activations:
                        self.activations[name] = []
                    self.activations[name].extend(sampled.tolist())

        return hook

    def register_hooks(self, model, layer_indices: List[int]):
        """Register hooks on specific layers."""
        # Get all layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'layers'):
            layers = model.layers
        else:
            raise ValueError("Cannot find model layers")

        print(f"  Total layers in model: {len(layers)}")

        for idx in layer_indices:
            if idx >= len(layers):
                print(f"  Warning: Layer {idx} out of range, skipping")
                continue

            layer = layers[idx]

            # Hook on MLP down_proj (common in LLMs)
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
                hook = layer.mlp.down_proj.register_forward_hook(
                    self.get_hook(f"layer_{idx}_mlp_down")
                )
                self.hooks.append(hook)
                print(f"  ✓ Registered hook on Layer {idx} MLP down_proj")

            # Hook on attention output projection
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                hook = layer.self_attn.o_proj.register_forward_hook(
                    self.get_hook(f"layer_{idx}_attn_out")
                )
                self.hooks.append(hook)
                print(f"  ✓ Registered hook on Layer {idx} Attention o_proj")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self) -> Dict[str, np.ndarray]:
        """Get collected activations."""
        return {k: np.array(v[:self.num_samples]) for k, v in self.activations.items()}


def load_calibration_data(tokenizer, seq_length: int = 2048) -> torch.Tensor:
    """Load calibration data from wikitext dataset."""
    try:
        print("  Loading wikitext-2-raw-v1 dataset...")
        dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1",
            split="train",
            cache_dir=os.environ["HF_DATASETS_CACHE"],
            token=HF_TOKEN,
        )

        # Concatenate texts
        texts = []
        for item in dataset:
            if item['text'].strip():
                texts.append(item['text'])

        full_text = " ".join(texts[:100])  # Use first 100 non-empty entries
        print(f"  ✓ Loaded {len(full_text)} characters from wikitext")

        # Tokenize
        tokens = tokenizer(full_text, return_tensors="pt", truncation=False)
        input_ids = tokens['input_ids']

        if input_ids.shape[1] < seq_length:
            print(f"  Warning: Text too short ({input_ids.shape[1]} tokens), padding...")
            # Repeat text if needed
            repeats = (seq_length // input_ids.shape[1]) + 1
            input_ids = input_ids.repeat(1, repeats)[:, :seq_length]
        else:
            # Random crop
            start_idx = random.randint(0, input_ids.shape[1] - seq_length)
            input_ids = input_ids[:, start_idx:start_idx + seq_length]

        print(f"  ✓ Generated input sequence of length {input_ids.shape[1]}")
        return input_ids

    except Exception as e:
        print(f"  Warning: Failed to load wikitext ({e}), using random fallback")
        # Fallback: random token IDs
        vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 32000
        return torch.randint(0, vocab_size, (1, seq_length))


def plot_activation_distribution(activations: np.ndarray, title: str, save_path: str):
    """Plot activation distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics
    mean_val = np.mean(activations)
    std_val = np.std(activations)
    min_val = np.min(activations)
    max_val = np.max(activations)

    # Determine if log scale is needed (large range or outliers)
    value_range = max_val - min_val
    use_log = value_range > 100 or (np.abs(max_val) > 10 * np.abs(mean_val))

    # Plot histogram
    n_bins = 50
    counts, bins, patches = ax.hist(activations, bins=n_bins, alpha=0.7,
                                     color='steelblue', edgecolor='black')

    if use_log and np.any(counts > 0):
        ax.set_yscale('log')

    # Add statistics text
    stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

    # Add vertical lines for mean
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')

    # Labels
    ax.set_xlabel('Activation Value', fontsize=12)
    ax.set_ylabel('Count' + (' (log scale)' if use_log else ''), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    ✓ Saved plot: {save_path}")


# ============================================================================
# DATASET DEFINITIONS
# ============================================================================
# Each entry: (description, hf_repo, config_name_or_None, splits_to_download)
STANDARD_DATASETS = [
    ("WikiText-2",   "wikitext",          "wikitext-2-raw-v1", ["train", "validation", "test"]),
    ("MMLU",         "cais/mmlu",          "all",               ["test", "validation"]),
    ("GSM8K",        "openai/gsm8k",       "main",              ["train", "test"]),
    ("ARC-Challenge","allenai/ai2_arc",    "ARC-Challenge",     ["train", "validation", "test"]),
    ("HellaSwag",    "Rowan/hellaswag",    None,                ["train", "validation"]),
    ("WinoGrande",   "allenai/winogrande", "winogrande_xl",     ["train", "validation"]),
    ("PTB",          "ptb-text-only",      "penn_treebank",     ["train", "validation", "test"]),
]

# C4: only download 1 validation shard for evaluation (train not needed; wikitext used for calibration)
C4_DATA_FILES = {
    "validation": ["en/c4-validation.00000-of-00008.json.gz"],
}


def download_datasets():
    """Download all benchmark / calibration datasets to HF_DATASETS_CACHE."""
    print("\n" + "="*80)
    print("Downloading Datasets")
    print("="*80)

    cache = os.environ["HF_DATASETS_CACHE"]

    # ── Standard datasets ────────────────────────────────────────────────────
    for desc, repo, cfg, splits in STANDARD_DATASETS:
        print(f"\n  [{desc}] {repo}" + (f" / {cfg}" if cfg else ""))
        for split in splits:
            try:
                if cfg:
                    ds = load_dataset(repo, cfg, split=split,
                                     cache_dir=cache, token=HF_TOKEN)
                else:
                    ds = load_dataset(repo, split=split,
                                     cache_dir=cache, token=HF_TOKEN)
                print(f"    ✓ {split}: {len(ds):,} samples")
                del ds
            except Exception as e:
                print(f"    ✗ {split}: {e}")

    # ── C4 (validation only) ─────────────────────────────────────────────────
    print(f"\n  [C4] allenai/c4  (validation only: {len(C4_DATA_FILES['validation'])} shard)")
    for split, files in C4_DATA_FILES.items():
        try:
            ds = load_dataset(
                "allenai/c4",
                data_files={split: files},
                split=split,
                cache_dir=cache,
                token=HF_TOKEN,
            )
            print(f"    ✓ {split}: {len(ds):,} samples")
            del ds
        except Exception as e:
            print(f"    ✗ {split}: {e}")

    print("\n  ✓ Dataset download complete")
    print(f"  ✓ Saved to: {cache}")


def analyze_model(model_name: str):
    """Analyze activation distributions for a single model."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}")

    try:
        # Load tokenizer
        print("  Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_HOME,
            trust_remote_code=True,
            token=HF_TOKEN
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        print("  Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            device_map="auto",
            cache_dir=HF_HOME,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=HF_TOKEN
        )
        model.eval()

        # Get number of layers
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            num_layers = len(model.model.layers)
        elif hasattr(model, 'layers'):
            num_layers = len(model.layers)
        else:
            raise ValueError("Cannot determine number of layers")

        print(f"  Model has {num_layers} layers")

        # Select layers: first, middle, last
        layer_indices = [0, num_layers // 2, num_layers - 1]
        print(f"  Analyzing layers: {layer_indices}")

        # Load calibration data
        input_ids = load_calibration_data(tokenizer, SEQ_LENGTH)

        # Setup activation collector
        collector = ActivationCollector(num_samples=NUM_SAMPLES)
        collector.register_hooks(model, layer_indices)

        # Forward pass to collect activations
        print("  Running forward pass to collect activations...")
        with torch.no_grad():
            input_ids = input_ids.to(model.device)
            _ = model(input_ids)

        # Get activations
        activations_dict = collector.get_activations()
        print(f"  ✓ Collected activations from {len(activations_dict)} hooks")

        # Plot for each layer/hook
        model_name_clean = model_name.replace("/", "_")
        for hook_name, activations in activations_dict.items():
            if len(activations) > 0:
                print(f"    Processing {hook_name}: {len(activations)} samples")

                title = f"{model_name} - {hook_name.replace('_', ' ').title()}"
                save_path = os.path.join(PLOT_DIR, f"{model_name_clean}_{hook_name}.png")

                plot_activation_distribution(activations, title, save_path)

        # Cleanup
        collector.remove_hooks()
        del model
        del tokenizer
        del input_ids
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  ✓ Completed analysis for {model_name}")
        print(f"  ✓ GPU memory cleared")

    except Exception as e:
        print(f"  ✗ Error analyzing {model_name}: {e}")
        import traceback
        traceback.print_exc()

        # Cleanup on error
        gc.collect()
        torch.cuda.empty_cache()


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("LLM Activation Distribution Analysis (DartQuant Methodology)")
    print("="*80)
    print(f"HF token        : {'Set ✓' if HF_TOKEN else 'Not set (may fail for gated models)'}")
    print(f"Target models   : {len(MODELS)}")
    print(f"Sequence length : {SEQ_LENGTH}")
    print(f"Samples / layer : {NUM_SAMPLES}")
    print(f"Data type       : {DTYPE}")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available, will use CPU (slow)")

    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Download datasets first
    download_datasets()

    # Process each model sequentially
    for model_name in MODELS:
        analyze_model(model_name)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All plots saved to: {PLOT_DIR}")
    print("\nSummary:")
    plot_files = list(Path(PLOT_DIR).glob("*.png"))
    print(f"  Generated {len(plot_files)} plots")
    for pf in sorted(plot_files):
        print(f"    - {pf.name}")


if __name__ == "__main__":
    main()
