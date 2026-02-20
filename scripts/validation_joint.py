#!/usr/bin/env python3
"""
DartQuant with Joint Rotation Optimization - Fixed Visualization
"""

import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
HF_TOKEN = None 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
CACHE_DIR = "/root/autodl-tmp"
HF_HOME = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_HOME"] = HF_HOME
OUTPUT_DIR = os.path.join(CACHE_DIR, "dartquant_joint")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PAIRS = [("meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.2-1B-Instruct")]

SEQ_LENGTH = 2048
NUM_SAMPLES = 128  # Fixed as requested
DTYPE_MODEL = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
DTYPE_OPT = torch.float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters - reduced LR for stability
LR = 5e-5
MOMENTUM = 0.9
EPOCHS = 500
BATCH_SIZE = 64

# ============================================================================
# JOINT ROTATION TRAINER
# ============================================================================

class JointRotationTrainer(nn.Module):
    def __init__(self, hidden_size, num_layers, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.Z_matrices = nn.ParameterList([
            nn.Parameter(self._init_hadamard(hidden_size, device))
            for _ in range(num_layers)
        ])
        
        self.layer_weights = nn.Parameter(torch.ones(num_layers), requires_grad=False)
    
    def _init_hadamard(self, size, device):
        random_matrix = torch.randn(size, size, dtype=torch.float64, device=device)
        q, r = torch.linalg.qr(random_matrix)
        q *= torch.sign(torch.diag(r)).unsqueeze(0)
        return q.float()
    
    def get_rotation(self, layer_idx):
        Q, _ = torch.linalg.qr(self.Z_matrices[layer_idx], mode='complete')
        return Q
    
    def compute_whip_loss(self, x):
        return torch.sum(torch.exp(-x.abs()), dim=-1).mean()
    
    def forward(self, activations_dict):
        total_loss = 0
        per_layer_losses = {}
        
        for layer_idx in sorted(activations_dict.keys()):
            acts = activations_dict[layer_idx].float().reshape(-1, self.hidden_size)
            Q = self.get_rotation(layer_idx)
            rotated = torch.matmul(acts, Q)
            layer_loss = self.compute_whip_loss(rotated)
            
            weight = self.layer_weights[layer_idx]
            total_loss = total_loss + weight * layer_loss
            per_layer_losses[layer_idx] = layer_loss.item()
        
        return total_loss, per_layer_losses


# ============================================================================
# ACTIVATION COLLECTION
# ============================================================================

def collect_all_layer_activations(model, input_ids, target_layers, hook_target='up_proj'):
    activations = {}
    hooks = []
    
    for idx in target_layers:
        if hook_target == 'up_proj':
            layer = model.model.layers[idx].mlp.up_proj
        else:
            layer = model.model.layers[idx].self_attn.q_proj
        
        def make_hook(layer_idx):
            def hook(module, inp, out):
                tensor = inp[0] if isinstance(inp, tuple) else inp
                flat = tensor.detach().float().cpu().reshape(-1, tensor.shape[-1])
                if flat.shape[0] > 200:
                    indices = torch.randperm(flat.shape[0])[:200]
                    flat = flat[indices]
                if layer_idx not in activations:
                    activations[layer_idx] = []
                activations[layer_idx].append(flat)
            return hook
        
        hooks.append(layer.register_forward_hook(make_hook(idx)))
    
    with torch.no_grad():
        model(input_ids)
    
    for h in hooks:
        h.remove()
    
    for idx in activations:
        activations[idx] = torch.cat(activations[idx], dim=0)
    
    return activations


# ============================================================================
# JOINT TRAINING
# ============================================================================

def train_joint_rotations(all_activations, hidden_size, num_total_layers, target_layers, config):
    device = config['device']
    
    # Move to device, subsample
    for idx in all_activations:
        acts = all_activations[idx]
        if acts.shape[0] > config['num_samples']:
            indices = torch.randperm(acts.shape[0])[:config['num_samples']]
            acts = acts[indices]
        all_activations[idx] = acts.to(device)
    
    trainer = JointRotationTrainer(hidden_size, num_total_layers, device).to(device)
    
    optimizer = torch.optim.SGD(trainer.parameters(), lr=config['lr'], momentum=config['momentum'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=0)
    
    loss_history = {idx: [] for idx in target_layers}
    
    for epoch in range(config['epochs']):
        optimizer.zero_grad()
        total_loss, per_layer = trainer(all_activations)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        for idx in target_layers:
            if idx in per_layer:
                loss_history[idx].append(per_layer[idx])
        
        if (epoch + 1) % 20 == 0:
            layer_str = " | ".join([f"L{idx}:{per_layer.get(idx, 0):.1f}" for idx in target_layers])
            print(f"    Epoch {epoch+1}/{config['epochs']} | {layer_str}")
    
    Q_matrices = {}
    with torch.no_grad():
        for idx in target_layers:
            Q_matrices[idx] = trainer.get_rotation(idx).detach().cpu()
    
    return Q_matrices, loss_history


# ============================================================================
# VISUALIZATION - 4 PANELS PER LAYER
# ============================================================================

def plot_all_results(base_acts, instruct_acts, rotated_base, rotated_instruct, 
                     layer_indices, model_name, loss_history):
    """
    4-panel plot per layer:
    1. Base vs Instruct (RLHF Impact)
    2. Base vs Rotated Base
    3. Instruct vs Rotated Instruct
    4. Rotated Base vs Rotated Instruct
    """
    n_layers = len(layer_indices)
    fig, axes = plt.subplots(n_layers, 4, figsize=(24, 5*n_layers))
    
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    def clean(data):
        return data[np.abs(data) < 50]
    
    for i, idx in enumerate(layer_indices):
        b = clean(base_acts[idx].flatten())
        inst = clean(instruct_acts[idx].flatten())
        rb = clean(rotated_base[idx].flatten())
        ri = clean(rotated_instruct[idx].flatten())
        
        # Panel 1: Base vs Instruct
        axes[i, 0].hist(b, bins=100, alpha=0.5, color='blue', label='Base', density=True, log=True)
        axes[i, 0].hist(inst, bins=100, alpha=0.5, color='orange', label='Instruct', density=True, log=True)
        axes[i, 0].set_title(f"Layer {idx}: RLHF Impact (Base vs Instruct)")
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xlabel("Activation Value")
        axes[i, 0].set_ylabel("Density (log)")
        
        # Panel 2: Base vs Rotated Base
        axes[i, 1].hist(b, bins=100, alpha=0.5, color='blue', label='Base (Original)', density=True, log=True)
        axes[i, 1].hist(rb, bins=100, alpha=0.5, color='cyan', label='Base (Rotated)', density=True, log=True)
        axes[i, 1].set_title(f"Layer {idx}: Rotation Effect on Base")
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlabel("Activation Value")
        axes[i, 1].set_ylabel("Density (log)")
        
        # Panel 3: Instruct vs Rotated Instruct
        axes[i, 2].hist(inst, bins=100, alpha=0.5, color='red', label='Instruct (Original)', density=True, log=True)
        axes[i, 2].hist(ri, bins=100, alpha=0.5, color='green', label='Instruct (Rotated)', density=True, log=True)
        axes[i, 2].set_title(f"Layer {idx}: DartQuant on Instruct")
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_xlabel("Activation Value")
        axes[i, 2].set_ylabel("Density (log)")
        
        # Panel 4: Rotated Base vs Rotated Instruct
        axes[i, 3].hist(rb, bins=100, alpha=0.5, color='cyan', label='Base (Rotated)', density=True, log=True)
        axes[i, 3].hist(ri, bins=100, alpha=0.5, color='green', label='Instruct (Rotated)', density=True, log=True)
        axes[i, 3].set_title(f"Layer {idx}: Rotated Base vs Rotated Instruct")
        axes[i, 3].legend()
        axes[i, 3].grid(True, alpha=0.3)
        axes[i, 3].set_xlabel("Activation Value")
        axes[i, 3].set_ylabel("Density (log)")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"{model_name}_joint_4panel.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ 4-panel plot saved: {save_path}")
    
    # Separate loss curve plot
    fig2, axes2 = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    if n_layers == 1:
        axes2 = [axes2]
    
    for i, idx in enumerate(layer_indices):
        if idx in loss_history and len(loss_history[idx]) > 0:
            axes2[i].plot(loss_history[idx], linewidth=1)
            axes2[i].set_title(f"Layer {idx}: Whip Loss")
            axes2[i].set_xlabel("Epoch")
            axes2[i].set_ylabel("Loss")
            axes2[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path2 = os.path.join(OUTPUT_DIR, f"{model_name}_loss_curves.png")
    plt.savefig(save_path2, dpi=150)
    plt.close()
    print(f"  ✓ Loss curves saved: {save_path2}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Starting DartQuant with Joint Rotation Optimization...")
    
    config = {
        'device': DEVICE,
        'seq_len': SEQ_LENGTH,
        'num_samples': NUM_SAMPLES,
        'lr': LR,
        'momentum': MOMENTUM,
        'epochs': EPOCHS,
    }
    
    for base_name, instruct_name in MODEL_PAIRS:
        short_name = instruct_name.split("/")[-1]
        print(f"\nProcessing: {short_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(base_name, token=HF_TOKEN, cache_dir=HF_HOME)
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, SEQ_LENGTH)).to(DEVICE)
        
        # Load base model
        print("  Loading Base Model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=DTYPE_MODEL, device_map="auto", token=HF_TOKEN, cache_dir=HF_HOME
        )
        
        num_layers = base_model.config.num_hidden_layers
        hidden_size = base_model.config.hidden_size
        
        # 5 layers: first, last, and 3 evenly spaced middle
        target_layers = [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]
        print(f"  Target layers: {target_layers}")
        
        base_acts = collect_all_layer_activations(base_model, input_ids, target_layers)
        
        del base_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Load instruct model
        print("  Loading Instruct Model...")
        instruct_model = AutoModelForCausalLM.from_pretrained(
            instruct_name, torch_dtype=DTYPE_MODEL, device_map="auto", token=HF_TOKEN, cache_dir=HF_HOME
        )
        
        instruct_acts = collect_all_layer_activations(instruct_model, input_ids, target_layers)
        
        # Joint training on INSTRUCT activations
        print("\n  === Joint Rotation Training ===")
        instruct_acts_copy = {k: v.clone() for k, v in instruct_acts.items()}
        Q_matrices, loss_history = train_joint_rotations(
            instruct_acts_copy, hidden_size, num_layers, target_layers, config
        )
        
        del instruct_model
        gc.collect()
        torch.cuda.empty_cache()
        
        # Apply rotations to both base and instruct
        rotated_base = {}
        rotated_instruct = {}
        for idx in target_layers:
            Q = Q_matrices[idx].numpy()
            rotated_base[idx] = np.matmul(base_acts[idx].numpy(), Q)
            rotated_instruct[idx] = np.matmul(instruct_acts[idx].numpy(), Q)
        
        # Convert to numpy
        base_acts_np = {idx: base_acts[idx].numpy() for idx in target_layers}
        instruct_acts_np = {idx: instruct_acts[idx].numpy() for idx in target_layers}
        
        # Plot
        plot_all_results(base_acts_np, instruct_acts_np, rotated_base, rotated_instruct,
                        target_layers, short_name, loss_history)
    
    print("\n✓ Complete.")


if __name__ == "__main__":
    main()