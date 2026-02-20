#!/usr/bin/env python3
"""
DartQuant: Complete Experiment Script (Fixed Device Mismatch)
- Independent Mode (DartQuant): Greedy layer-by-layer optimization of global R1.
- Joint Mode (JointQuant): Joint optimization of global R1 across all layers.

Refactored to explicitly handle device placement (CPU <-> GPU) to prevent RuntimeErrors.
"""

import os
import sys
import copy
import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# IMPORT FROM JOINT_QUANT
# ============================================================================
from joint_quant import (
    # Config
    JointQuantConfig,
    
    # Utils
    DEV,
    set_seed,
    cleanup_memory,
    
    # Model Setup
    get_model,
    get_tokenizer,
    
    # Rotation & Quantization
    fuse_layer_norms,
    rotate_model,
    QuantizedLinear,
    
    # Training
    get_calibration_data,
    collect_activations,
    train_r1_joint,       # For Joint Mode
    train_r2_independent, # For both modes (R2 is always per-head)
    compute_simple_smooth_scale,
    
    # Evaluation
    evaluate_perplexity
)

# Import internal modules for Independent Loop implementation
from joint_quant.joint_training import R1Module, whip_loss

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)


# ============================================================================
# INDEPENDENT R1 TRAINING (DartQuant Baseline)
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
    Train R1 using the 'Independent' (Greedy Layer-by-Layer) strategy.
    """
    logging.info("Training R1 (Independent/Greedy Strategy)...")
    
    # Initialize global R1 module
    r1_module = R1Module(hidden_size, device, init_mode).to(device)
    optimizer = torch.optim.SGD(r1_module.parameters(), lr=lr, momentum=momentum)
    
    layer_names = sorted(activations.keys())
    r1_module.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        # Iterate strictly layer-by-layer (Greedy)
        for name in tqdm(layer_names, desc=f"Epoch {epoch+1}/{epochs} (Layers)", leave=False):
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
# QUANTIZATION HELPER
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
    config: JointQuantConfig
) -> nn.Module:
    """Apply R1, R2, Smoothing, then replace layers with QuantizedLinear."""
    # 1. Apply Rotations
    # NOTE: rotate_model moves weights to CPU to save memory!
    rotate_model(model, R1, R2_matrices, smooth_scale)
    
    # 2. Quantize Linear Layers
    logging.info(f"Quantizing linear layers to W{config.w_bits}A{config.a_bits}...")
    replace_linear_with_quantized(model, config)
    
    # 3. FIX: Move quantized model back to GPU for evaluation
    logging.info("Moving quantized model to GPU...")
    return model.to(DEV)


# ============================================================================
# MAIN EXPERIMENT LOGIC
# ============================================================================

def run_experiment(config: JointQuantConfig):
    set_seed(config.seed)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"DartQuant Experiment: {config.model_name}")
    print("="*60)
    
    # 1. Load Model
    logging.info("Loading model and tokenizer...")
    tokenizer = get_tokenizer(config.model_name, config.hf_token, config.cache_dir)
    original_model = get_model(config.model_name, config.hf_token, config.cache_dir, config.dtype)
    original_model = original_model.to(DEV)
    
    # 2. Fuse LayerNorms
    fuse_layer_norms(original_model)
    
    # 3. Get Calibration Data
    calib_data = get_calibration_data(
        tokenizer, config.nsamples, config.seqlen, config.seed, config.cal_dataset
    )
    
    # 4. Collect Activations for R1
    logging.info("Collecting activations for R1 training...")
    r1_targets = []
    num_layers = len(original_model.model.layers)
    for i in range(num_layers):
        r1_targets.append(f"model.layers.{i}.self_attn.q_proj")
        r1_targets.append(f"model.layers.{i}.mlp.up_proj")
    
    r1_activations = collect_activations(original_model, calib_data, r1_targets, DEV)
    
    # Helper to train R2
    def train_r2_on_rotated_model(model_rotated):
        """Train R2 (Per-Head) on a model that already has R1 applied."""
        logging.info("Collecting activations for R2 training...")
        r2_targets = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]
        
        # FIX: Ensure model is on the correct device for inference
        # collect_activations expects model to handle GPU inputs if DEV is GPU
        if next(model_rotated.parameters()).device.type == 'cpu' and DEV.type == 'cuda':
             logging.info("  Moving temp model to GPU for activation collection...")
             model_rotated = model_rotated.to(DEV)
        
        r2_acts = collect_activations(model_rotated, calib_data, r2_targets, DEV)
        
        return train_r2_independent(
            activations=r2_acts,
            hidden_size=model_rotated.config.hidden_size,
            num_heads=model_rotated.config.num_attention_heads,
            num_kv_heads=model_rotated.config.num_key_value_heads,
            device=DEV,
            lr=config.r2_lr,
            epochs=config.r2_epochs
        )

    # ---------------------------------------------------------
    # MODE A: INDEPENDENT (Original DartQuant - Greedy)
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("Running Mode: INDEPENDENT (DartQuant)")
    print("-"*40)
    
    # A1. Train R1
    R1_indep = train_r1_independent(
        activations=r1_activations,
        hidden_size=original_model.config.hidden_size,
        device=DEV,
        lr=config.r1_lr,
        epochs=config.r1_epochs,
        momentum=config.r1_momentum
    )
    
    # A2. Apply R1 temporarily to train R2
    model_indep_temp = copy.deepcopy(original_model)
    rotate_model(model_indep_temp, R1_indep, R2_matrices=None, smooth_scale=None)
    
    # FIX: rotate_model moves weights to CPU. We MUST move back to GPU for the forward pass in train_r2.
    logging.info("Moving independent temp model to GPU...")
    model_indep_temp = model_indep_temp.to(DEV)
    
    # A3. Train R2
    R2_indep = train_r2_on_rotated_model(model_indep_temp)
    
    # A4. Compute Smooth Scales (requires GPU model)
    smooth_scale_indep = compute_simple_smooth_scale(model_indep_temp, calib_data, DEV)
    del model_indep_temp 
    cleanup_memory()
    
    # A5. Apply Everything & Quantize
    model_indep = copy.deepcopy(original_model)
    # quantize_pipeline now handles the .to(DEV) return
    model_indep = quantize_pipeline(model_indep, R1_indep, R2_indep, smooth_scale_indep, config)
    
    # A6. Evaluate
    ppl_indep = evaluate_perplexity(model_indep, tokenizer, DEV, config.eval_dataset)
    print(f"Independent PPL: {ppl_indep:.4f}")
    
    del model_indep, R1_indep, R2_indep, smooth_scale_indep
    cleanup_memory()

    # ---------------------------------------------------------
    # MODE B: JOINT (JointQuant - Ours)
    # ---------------------------------------------------------
    print("\n" + "-"*40)
    print("Running Mode: JOINT (JointQuant)")
    print("-"*40)
    
    # B1. Train R1
    R1_joint = train_r1_joint(
        activations=r1_activations,
        hidden_size=original_model.config.hidden_size,
        num_layers=num_layers,
        device=DEV,
        lr=config.joint_lr,
        epochs=config.joint_epochs,
        momentum=config.r1_momentum,
        use_cosine_lr=True 
    )
    
    # B2. Apply R1 temporarily to train R2
    model_joint_temp = copy.deepcopy(original_model)
    rotate_model(model_joint_temp, R1_joint, R2_matrices=None, smooth_scale=None)
    
    # FIX: Move to GPU
    logging.info("Moving joint temp model to GPU...")
    model_joint_temp = model_joint_temp.to(DEV)
    
    # B3. Train R2
    R2_joint = train_r2_on_rotated_model(model_joint_temp)
    
    # B4. Compute Smooth Scales
    smooth_scale_joint = compute_simple_smooth_scale(model_joint_temp, calib_data, DEV)
    del model_joint_temp
    cleanup_memory()
    
    # B5. Apply Everything & Quantize
    model_joint = copy.deepcopy(original_model)
    model_joint = quantize_pipeline(model_joint, R1_joint, R2_joint, smooth_scale_joint, config)
    
    # B6. Evaluate
    ppl_joint = evaluate_perplexity(model_joint, tokenizer, DEV, config.eval_dataset)
    print(f"Joint PPL:       {ppl_joint:.4f}")

    # Results
    print("\n" + "="*60)
    print(f"FINAL RESULTS")
    print("="*60)
    print(f"{'Method':<20} | {'PPL':<10}")
    print("-" * 35)
    print(f"{'Independent':<20} | {ppl_indep:.4f}")
    print(f"{'Joint':<20} | {ppl_joint:.4f}")
    print("="*60)
    
    results_path = Path(config.output_dir) / "results.txt"
    with open(results_path, "w") as f:
        f.write(f"Independent: {ppl_indep}\n")
        f.write(f"Joint: {ppl_joint}\n")
    print(f"Results saved to {results_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DartQuant Experiment")
    
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="/root/autodl-tmp/huggingface")
    
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    
    # Training Params
    parser.add_argument("--r1_lr", type=float, default=1e-3)
    parser.add_argument("--r1_epochs", type=int, default=10)
    parser.add_argument("--r1_momentum", type=float, default=0.9)
    parser.add_argument("--r2_lr", type=float, default=1e-3)
    parser.add_argument("--r2_epochs", type=int, default=5)
    parser.add_argument("--joint_lr", type=float, default=5e-4)
    parser.add_argument("--joint_epochs", type=int, default=100)
    
    # Quant Params
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
        joint_lr=args.joint_lr,
        joint_epochs=args.joint_epochs,
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