#!/usr/bin/env python3
"""
JointQuant Main Entry Point

Usage:
    python main.py --model meta-llama/Llama-3.2-1B --w_bits 4 --a_bits 4

This script implements:
1. Joint R1 rotation optimization (our innovation)
2. R2 per-head rotation (same as DartQuant)
3. SmoothQuant integration
4. W4A4 quantization with group-wise weights
5. WikiText2 perplexity evaluation
"""

import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joint_quant import (
    # Utils
    DEV, set_seed, cleanup_memory, config_logging,
    # Model
    get_model, get_tokenizer, get_model_type,
    # Rotation
    fuse_layer_norms, rotate_model,
    # Smooth
    compute_simple_smooth_scale,
    # Data
    get_calibration_data,
    # Eval
    evaluate_perplexity,
    # Joint Training
    train_r1_joint, train_r2_independent, collect_activations,
    # Quantization
    QuantizedLinear,
)


@dataclass
class Config:
    """Configuration for JointQuant"""
    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"
    hf_token: Optional[str] = None
    cache_dir: Optional[str] = None
    
    # Calibration
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 0
    cal_dataset: str = "wikitext2"
    
    # R1 Training (Joint - our innovation)
    r1_lr: float = 1e-3
    r1_epochs: int = 10
    r1_momentum: float = 0.9
    r1_init_mode: str = "hadamard"
    
    # R2 Training
    r2_lr: float = 1e-3
    r2_epochs: int = 5
    r2_momentum: float = 0.9
    
    # Smoothing
    use_smooth: bool = True
    smooth_alpha: float = 0.5
    
    # Quantization
    w_bits: int = 4
    a_bits: int = 4
    w_groupsize: int = 128
    
    # Evaluation
    eval_dataset: str = "wikitext2"
    
    # Output
    output_dir: str = "./jointquant_output"
    save_rotations: bool = True
    
    # Device
    device: str = "cuda"


def apply_quantization(model: nn.Module, config: Config) -> nn.Module:
    """Apply INT4 quantization to all linear layers"""
    logging.info(f"Quantizing to W{config.w_bits}A{config.a_bits}...")
    
    quantized_count = 0
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Skip lm_head
            if 'lm_head' in name:
                logging.info(f"  Skipping {name}")
                continue
            
            # Get parent module
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name
            
            # Create quantized version
            q_module = QuantizedLinear(
                module.weight.data.float(),
                module.bias.data.float() if module.bias is not None else None,
                w_bits=config.w_bits,
                a_bits=config.a_bits,
                group_size=config.w_groupsize
            )
            
            setattr(parent, child_name, q_module)
            quantized_count += 1
    
    logging.info(f"  Quantized {quantized_count} linear layers")
    return model


def main():
    parser = argparse.ArgumentParser(description='JointQuant: INT4 Quantization with Joint R1 Optimization')
    
    # Model
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                        help='Model name or path')
    parser.add_argument('--hf_token', type=str, default=None,
                        help='HuggingFace token')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory for models/datasets')
    
    # Calibration
    parser.add_argument('--nsamples', type=int, default=128,
                        help='Number of calibration samples')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--cal_dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4', 'ptb'],
                        help='Calibration dataset')
    
    # R1 Training
    parser.add_argument('--r1_lr', type=float, default=1e-3,
                        help='R1 learning rate')
    parser.add_argument('--r1_epochs', type=int, default=10,
                        help='R1 training epochs')
    parser.add_argument('--r1_init', type=str, default='hadamard',
                        choices=['hadamard', 'random'],
                        help='R1 initialization mode')
    
    # R2 Training
    parser.add_argument('--r2_lr', type=float, default=1e-3,
                        help='R2 learning rate')
    parser.add_argument('--r2_epochs', type=int, default=5,
                        help='R2 training epochs')
    
    # Smoothing
    parser.add_argument('--use_smooth', action='store_true', default=True,
                        help='Use SmoothQuant scaling')
    parser.add_argument('--no_smooth', action='store_false', dest='use_smooth',
                        help='Disable SmoothQuant')
    parser.add_argument('--smooth_alpha', type=float, default=0.5,
                        help='SmoothQuant migration strength')
    
    # Quantization
    parser.add_argument('--w_bits', type=int, default=4,
                        help='Weight bits')
    parser.add_argument('--a_bits', type=int, default=4,
                        help='Activation bits')
    parser.add_argument('--w_groupsize', type=int, default=128,
                        help='Weight quantization group size')
    
    # Evaluation
    parser.add_argument('--eval_dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'c4', 'ptb'],
                        help='Evaluation dataset')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./jointquant_output',
                        help='Output directory')
    parser.add_argument('--save_rotations', action='store_true', default=True,
                        help='Save trained rotation matrices')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        model_name=args.model,
        hf_token=args.hf_token,
        cache_dir=args.cache_dir,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        cal_dataset=args.cal_dataset,
        r1_lr=args.r1_lr,
        r1_epochs=args.r1_epochs,
        r1_init_mode=args.r1_init,
        r2_lr=args.r2_lr,
        r2_epochs=args.r2_epochs,
        use_smooth=args.use_smooth,
        smooth_alpha=args.smooth_alpha,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        w_groupsize=args.w_groupsize,
        eval_dataset=args.eval_dataset,
        output_dir=args.output_dir,
        save_rotations=args.save_rotations,
    )
    
    # Setup
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config_logging(os.path.join(config.output_dir, 'jointquant.log'))
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    logging.info("="*70)
    logging.info("JointQuant: INT4 Quantization with Joint R1 Optimization")
    logging.info("="*70)
    logging.info(f"Model: {config.model_name}")
    logging.info(f"Quantization: W{config.w_bits}A{config.a_bits}")
    logging.info(f"Samples: {config.nsamples}, SeqLen: {config.seqlen}")
    logging.info(f"Smoothing: {config.use_smooth} (alpha={config.smooth_alpha})")
    
    # ========== 1. Load Model and Tokenizer ==========
    logging.info("\n[1/7] Loading Model and Tokenizer...")
    tokenizer = get_tokenizer(config.model_name, config.hf_token, config.cache_dir)
    model = get_model(config.model_name, config.hf_token, config.cache_dir)
    
    # Resize embeddings if needed
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        logging.info(f"  Resizing embeddings to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
    
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    
    logging.info(f"  Hidden: {hidden_size}, Layers: {num_layers}, Heads: {num_heads}/{num_kv_heads}")
    
    # ========== 2. Load Calibration Data ==========
    logging.info("\n[2/7] Loading Calibration Data...")
    calib_data = get_calibration_data(
        tokenizer, config.nsamples, config.seqlen, 
        config.seed, config.cal_dataset
    )
    logging.info(f"  Calibration data: {calib_data.shape}")
    
    # ========== 3. Evaluate Baseline ==========
    logging.info("\n[3/7] Evaluating Baseline Perplexity...")
    model = model.to(device)
    baseline_ppl = evaluate_perplexity(model, tokenizer, device, config.eval_dataset)
    model = model.cpu()
    cleanup_memory()
    
    # ========== 4. Collect Activations ==========
    logging.info("\n[4/7] Collecting Activations...")
    model = model.to(device)
    
    # R1 targets: MLP up_proj inputs
    r1_targets = [f"model.layers.{i}.mlp.up_proj" for i in range(num_layers)]
    r1_activations = collect_activations(model, calib_data, r1_targets, device)
    
    # R2 targets: Attention o_proj inputs  
    r2_targets = [f"model.layers.{i}.self_attn.o_proj" for i in range(num_layers)]
    r2_activations = collect_activations(model, calib_data, r2_targets, device)
    
    model = model.cpu()
    cleanup_memory()
    
    # ========== 5. Train Rotations ==========
    logging.info("\n[5/7] Training Rotation Matrices...")
    
    # Train R1 jointly (our innovation!)
    R1 = train_r1_joint(
        r1_activations, hidden_size, num_layers, device,
        lr=config.r1_lr, epochs=config.r1_epochs,
        momentum=config.r1_momentum, init_mode=config.r1_init_mode
    )
    
    # Train R2 independently (same as DartQuant)
    R2_matrices = train_r2_independent(
        r2_activations, hidden_size, num_heads, num_kv_heads, device,
        lr=config.r2_lr, epochs=config.r2_epochs,
        momentum=config.r2_momentum
    )
    
    cleanup_memory()
    
    # ========== 6. Compute Smooth Scales (if enabled) ==========
    smooth_scale = None
    if config.use_smooth:
        logging.info("\n[6/7] Computing SmoothQuant Scales...")
        model = model.to(device)
        smooth_scale = compute_simple_smooth_scale(
            model, calib_data, device, 
            alpha=config.smooth_alpha
        )
        model = model.cpu()
        cleanup_memory()
    else:
        logging.info("\n[6/7] Skipping SmoothQuant (disabled)")
    
    # ========== 7. Apply Rotations and Quantize ==========
    logging.info("\n[7/7] Applying Rotations and Quantization...")
    
    # Reload fresh model
    model = get_model(config.model_name, config.hf_token, config.cache_dir)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    
    # Fuse LayerNorms (MUST be before rotation!)
    fuse_layer_norms(model)
    
    # Apply rotations
    rotate_model(model, R1, R2_matrices, smooth_scale)
    
    # Quantize
    model = apply_quantization(model, config)
    model = model.to(device)
    
    # ========== Evaluate ==========
    logging.info("\nEvaluating Quantized Model...")
    quant_ppl = evaluate_perplexity(model, tokenizer, device, config.eval_dataset)
    
    # ========== Results ==========
    logging.info("\n" + "="*70)
    logging.info("RESULTS")
    logging.info("="*70)
    logging.info(f"{'Method':<35} {'Perplexity':>15}")
    logging.info("-"*50)
    logging.info(f"{'FP16 Baseline':<35} {baseline_ppl:>15.2f}")
    logging.info(f"{'JointQuant W4A4':<35} {quant_ppl:>15.2f}")
    logging.info("-"*50)
    logging.info(f"{'Degradation':<35} {quant_ppl - baseline_ppl:>+15.2f}")
    
    # Save results
    results = {
        'model': config.model_name,
        'quantization': f'W{config.w_bits}A{config.a_bits}',
        'use_smooth': config.use_smooth,
        'smooth_alpha': config.smooth_alpha,
        'baseline_ppl': baseline_ppl,
        'quant_ppl': quant_ppl,
        'r1_lr': config.r1_lr,
        'r1_epochs': config.r1_epochs,
        'r2_lr': config.r2_lr,
        'r2_epochs': config.r2_epochs,
    }
    
    results_path = Path(config.output_dir) / "results.txt"
    with open(results_path, 'w') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    logging.info(f"\nResults saved to {results_path}")
    
    # Save rotations
    if config.save_rotations:
        rotations_path = Path(config.output_dir) / "rotations.pt"
        torch.save({
            'R1': R1,
            'R2': R2_matrices,
            'smooth_scale': smooth_scale,
        }, rotations_path)
        logging.info(f"Rotations saved to {rotations_path}")
    
    logging.info("\nDone!")
    return results


if __name__ == "__main__":
    main()
