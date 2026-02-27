#!/usr/bin/env python3
"""
DartQuant: Independent (Greedy Layer-by-Layer) R1/R2 Experiment.

Pipeline:
  1. Load model & tokenizer
  2. Fuse LayerNorms
  3. Collect calibration data
  4. Collect activations → train R1 (greedy, layer-by-layer)
  5. Apply R1, train R2 (per-head)
  6. Compute smooth scales
  7. Apply rotations + quantize (W4A4)
  8. Evaluate perplexity
"""

import os
import sys
import copy
import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# IMPORTS
# ============================================================================
from joint_quant import (
    JointQuantConfig,
    DEV,
    set_seed,
    cleanup_memory,
    get_model,
    get_tokenizer,
    fuse_layer_norms,
    rotate_model,
    QuantizedLinear,
    get_calibration_data,
    collect_activations,
    train_r2_independent,
    compute_simple_smooth_scale,
    evaluate_perplexity,
)
from joint_quant.joint_training import R1Module, whip_loss

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)


# ============================================================================
# INDEPENDENT R1 TRAINING (Greedy Layer-by-Layer)
# ============================================================================

def train_r1_independent(
    activations: Dict[str, torch.Tensor],
    hidden_size: int,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 10,
    momentum: float = 0.9,
    init_mode: str = 'hadamard',
) -> torch.Tensor:
    """
    Train R1 using the Independent (Greedy Layer-by-Layer) strategy.
    Iterates through layers sequentially, accumulating gradients per layer.
    """
    logging.info("Training R1 (Independent/Greedy Strategy)...")

    r1_module = R1Module(hidden_size, device, init_mode).to(device)
    optimizer = torch.optim.SGD(r1_module.parameters(), lr=lr, momentum=momentum)

    layer_names = sorted(activations.keys())
    r1_module.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for name in tqdm(layer_names, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            acts = activations[name].to(device).float()
            acts = torch.nan_to_num(acts, nan=0.0)

            if acts.shape[0] > 4096:
                indices = torch.randperm(acts.shape[0])[:4096]
                acts = acts[indices]

            optimizer.zero_grad()
            rotated = r1_module(acts)
            loss = whip_loss(rotated)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(r1_module.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

    R1 = r1_module.get_rotation().detach().cpu()
    logging.info("R1 Independent training complete.")
    return R1


# ============================================================================
# QUANTIZATION
# ============================================================================

def replace_linear_with_quantized(model: nn.Module, config: JointQuantConfig) -> int:
    """Recursively replace nn.Linear with QuantizedLinear (W4A4)."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if 'lm_head' in name:
                continue

            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent = model.get_submodule(parts[0])
                child_name = parts[1]
            else:
                parent = model
                child_name = name

            q_module = QuantizedLinear(
                weight=module.weight.data.clone(),
                bias=module.bias.data.clone() if module.bias is not None else None,
                w_bits=config.w_bits,
                a_bits=config.a_bits,
                group_size=config.group_size,
            )
            setattr(parent, child_name, q_module)
            count += 1
    return count


def quantize_pipeline(
    model: nn.Module,
    R1: torch.Tensor,
    R2_matrices: Dict[str, torch.Tensor],
    smooth_scale: Optional[Dict],
    config: JointQuantConfig,
) -> nn.Module:
    """Apply R1, R2, Smoothing, then replace layers with QuantizedLinear."""
    rotate_model(model, R1, R2_matrices, smooth_scale)

    logging.info(f"Quantizing linear layers to W{config.w_bits}A{config.a_bits}...")
    replace_linear_with_quantized(model, config)

    logging.info("Moving quantized model to GPU...")
    return model.to(DEV)


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(config: JointQuantConfig):
    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"DartQuant Independent Experiment: {config.model_name}")
    print("=" * 60)

    # 1. Load Model
    logging.info("Loading model and tokenizer...")
    tokenizer = get_tokenizer(config.model_name, config.hf_token, config.cache_dir)
    model = get_model(config.model_name, config.hf_token, config.cache_dir, config.dtype)
    model = model.to(DEV)

    # 2. Fuse LayerNorms
    fuse_layer_norms(model)

    # 3. Get Calibration Data
    calib_data = get_calibration_data(
        tokenizer, config.nsamples, config.seqlen, config.seed, config.cal_dataset
    )

    # 4. Collect Activations for R1
    logging.info("Collecting activations for R1 training...")
    num_layers = len(model.model.layers)
    r1_targets = []
    for i in range(num_layers):
        r1_targets.append(f"model.layers.{i}.self_attn.q_proj")
        r1_targets.append(f"model.layers.{i}.mlp.up_proj")

    r1_activations = collect_activations(model, calib_data, r1_targets, DEV)

    # 5. Train R1 (Greedy Layer-by-Layer)
    R1 = train_r1_independent(
        activations=r1_activations,
        hidden_size=model.config.hidden_size,
        device=DEV,
        lr=config.r1_lr,
        epochs=config.r1_epochs,
        momentum=config.r1_momentum,
    )
    del r1_activations
    cleanup_memory()

    # 6. Apply R1 temporarily → train R2
    model_temp = copy.deepcopy(model)
    rotate_model(model_temp, R1, R2_matrices=None, smooth_scale=None)
    model_temp = model_temp.to(DEV)

    logging.info("Collecting activations for R2 training...")
    r2_targets = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]
    r2_acts = collect_activations(model_temp, calib_data, r2_targets, DEV)

    R2 = train_r2_independent(
        activations=r2_acts,
        hidden_size=model.config.hidden_size,
        num_heads=model.config.num_attention_heads,
        num_kv_heads=model.config.num_key_value_heads,
        device=DEV,
        lr=config.r2_lr,
        epochs=config.r2_epochs,
    )
    del r2_acts
    cleanup_memory()

    # 7. Compute Smooth Scales
    smooth_scale = compute_simple_smooth_scale(model_temp, calib_data, DEV)
    del model_temp
    cleanup_memory()

    # 8. Apply Rotations + Quantize
    model = quantize_pipeline(model, R1, R2, smooth_scale, config)

    # 9. Evaluate
    ppl = evaluate_perplexity(model, tokenizer, DEV, config.eval_dataset)

    print("\n" + "=" * 60)
    print(f"RESULT: Independent (Greedy) PPL = {ppl:.4f}")
    print("=" * 60)

    results_path = Path(config.output_dir) / "results.txt"
    with open(results_path, "w") as f:
        f.write(f"model: {config.model_name}\n")
        f.write(f"method: independent_greedy\n")
        f.write(f"ppl: {ppl:.4f}\n")
        f.write(f"w_bits: {config.w_bits}\n")
        f.write(f"a_bits: {config.a_bits}\n")
    logging.info(f"Results saved to {results_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DartQuant Independent Experiment")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/huggingface")

    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)

    parser.add_argument("--r1_lr", type=float, default=1e-3)
    parser.add_argument("--r1_epochs", type=int, default=10)
    parser.add_argument("--r1_momentum", type=float, default=0.9)
    parser.add_argument("--r2_lr", type=float, default=1e-3)
    parser.add_argument("--r2_epochs", type=int, default=5)

    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--a_bits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./dartquant_output")

    return parser.parse_args()


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET_TRANSFERS", "1")

    args = parse_args()

    config = JointQuantConfig(
        model_name=args.model,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        r1_lr=args.r1_lr,
        r1_epochs=args.r1_epochs,
        r2_lr=args.r2_lr,
        r2_epochs=args.r2_epochs,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        group_size=args.group_size,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    config.r1_momentum = args.r1_momentum

    run_experiment(config)


if __name__ == "__main__":
    main()
