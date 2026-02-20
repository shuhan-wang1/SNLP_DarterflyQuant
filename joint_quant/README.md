# JointQuant: INT4 Quantization with Joint R1 Rotation Optimization

This implementation extends DartQuant with a key innovation: **Joint R1 Rotation Optimization**.

## Key Differences from DartQuant

| Aspect | DartQuant (Original) | JointQuant (Ours) |
|--------|---------------------|-------------------|
| R1 Training | Layer-by-layer (greedy) | Global joint optimization |
| Optimization | Each layer optimized independently | Single R1 matrix for all layers |
| Theoretical | Locally optimal per layer | Globally optimal across model |

## Installation

```bash
pip install torch transformers datasets tqdm
```

## Quick Start

```bash
# Basic usage with Llama-3.2-1B
python main.py --model meta-llama/Llama-3.2-1B

# With custom settings
python main.py \
    --model meta-llama/Llama-3.2-1B \
    --w_bits 4 \
    --a_bits 4 \
    --use_smooth \
    --smooth_alpha 0.5 \
    --r1_epochs 10 \
    --r2_epochs 5 \
    --output_dir ./my_output
```

## Module Structure

```
joint_quant/
├── __init__.py          # Package exports
├── config.py            # Configuration presets
├── utils.py             # General utilities
├── model_utils.py       # Model loading and manipulation
├── hadamard_utils.py    # Hadamard matrix generation
├── quant_utils.py       # Quantization utilities
├── rotation_utils.py    # Rotation and LayerNorm fusion
├── smooth_utils.py      # SmoothQuant scaling
├── data_utils.py        # Data loading
├── eval_utils.py        # Perplexity evaluation
└── joint_training.py    # Joint R1 and R2 training (key innovation)
```

## Key Components

### 1. Joint R1 Training (`joint_training.py`)

The core innovation: instead of training R1 layer-by-layer, we optimize a **single global R1 matrix** that minimizes Whip loss across ALL layers simultaneously.

```python
from joint_quant import train_r1_joint

R1 = train_r1_joint(
    activations,           # Dict of {layer_name: activation_tensor}
    hidden_size=2048,
    num_layers=16,
    device=torch.device('cuda'),
    lr=1e-3,               # Learning rate (matches DartQuant default)
    epochs=10,
    init_mode='hadamard',  # Hadamard init recommended
)
```

### 2. SmoothQuant Integration (`smooth_utils.py`)

Computes smooth scaling factors to migrate quantization difficulty from activations to weights:

```python
from joint_quant import compute_simple_smooth_scale

smooth_scale = compute_simple_smooth_scale(
    model,
    calibration_data,
    device,
    alpha=0.5,  # 0.5 = balanced migration
)
```

### 3. Full Pipeline

```python
from joint_quant import (
    get_model, get_tokenizer, get_calibration_data,
    fuse_layer_norms, rotate_model,
    train_r1_joint, train_r2_independent, collect_activations,
    compute_simple_smooth_scale,
    evaluate_perplexity
)

# Load model
model = get_model('meta-llama/Llama-3.2-1B')
tokenizer = get_tokenizer('meta-llama/Llama-3.2-1B')

# Get calibration data
calib_data = get_calibration_data(tokenizer, nsamples=128, seqlen=2048)

# Collect activations
r1_targets = [f"model.layers.{i}.mlp.up_proj" for i in range(num_layers)]
r1_acts = collect_activations(model, calib_data, r1_targets, device)

# Train R1 (Joint - our innovation!)
R1 = train_r1_joint(r1_acts, hidden_size, num_layers, device)

# Train R2 (same as DartQuant)
R2_matrices = train_r2_independent(r2_acts, hidden_size, num_heads, num_kv_heads, device)

# Compute smooth scales
smooth_scale = compute_simple_smooth_scale(model, calib_data, device, alpha=0.5)

# Apply transformations
fuse_layer_norms(model)  # MUST be before rotation!
rotate_model(model, R1, R2_matrices, smooth_scale)

# Evaluate
ppl = evaluate_perplexity(model, tokenizer, device)
```

## Configuration

See `joint_quant/config.py` for configuration presets:

```python
from joint_quant.config import get_llama_1b_config, get_llama_8b_config

config = get_llama_1b_config(
    use_smooth=True,
    smooth_alpha=0.5,
    output_dir='./my_output'
)
```

## Alignment with Official DartQuant

This implementation aligns with official DartQuant in:

- ✅ **Hadamard initialization** for R1/R2 (not random QR)
- ✅ **Whip loss** without outlier clamping (critical!)
- ✅ **LayerNorm fusion** before rotation
- ✅ **SmoothQuant** integration
- ✅ **Full INT4 range** [-8, 7] for weights
- ✅ **Group-wise weight quantization** (default 128)
- ✅ **Per-token asymmetric activation quantization**

Key innovation:
- 🆕 **Joint R1 optimization** instead of greedy layer-by-layer

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `meta-llama/Llama-3.2-1B` | Model name/path |
| `--nsamples` | 128 | Calibration samples |
| `--seqlen` | 2048 | Sequence length |
| `--r1_lr` | 1e-3 | R1 learning rate |
| `--r1_epochs` | 10 | R1 training epochs |
| `--r2_lr` | 1e-3 | R2 learning rate |
| `--r2_epochs` | 5 | R2 training epochs |
| `--use_smooth` | True | Enable SmoothQuant |
| `--smooth_alpha` | 0.5 | Smooth migration strength |
| `--w_bits` | 4 | Weight bits |
| `--a_bits` | 4 | Activation bits |
| `--w_groupsize` | 128 | Weight group size |
| `--output_dir` | `./jointquant_output` | Output directory |

## Expected Results

With proper settings, expected perplexity on WikiText2:

| Model | FP16 Baseline | JointQuant W4A4 |
|-------|---------------|-----------------|
| Llama-3.2-1B | ~8.5 | ~10-12 |
| Llama-3-8B | ~6.5 | ~7.5-9 |

## Troubleshooting

### High Perplexity (>100)

1. **Check outlier handling**: Ensure activations are NOT clamped during collection
2. **Check LayerNorm fusion**: Must be called BEFORE rotation
3. **Check weight tying**: lm_head and embed_tokens may share weights

### NaN Loss

1. Reduce learning rate (try 5e-4 or 1e-4)
2. Ensure Hadamard initialization (not random)
3. Check for Inf values in activations

## Citation

Based on DartQuant (arXiv:2511.04063v1, NeurIPS 2025).
