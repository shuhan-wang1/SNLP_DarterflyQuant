#!/usr/bin/env python3
"""
DartQuant Comparison Experiment: Whip Loss vs. Wasserstein (SWD) Loss
"""

import os
import sys
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
import math
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
# 设置 Modelscope 缓存目录环境变量（必须在 import modelscope 之前设置）
CACHE_DIR = "/root/autodl-tmp"
os.environ['MODELSCOPE_CACHE'] = CACHE_DIR

# 本地已下载的模型路径
LOCAL_MODEL_PATH = os.path.join(CACHE_DIR, "hub", "shakechen", "Llama-2-7b-hf")
# 如果本地路径不存在，使用 Modelscope 模型 ID
MODEL_NAME = "shakechen/Llama-2-7b-hf"

OUTPUT_DIR = os.path.join(CACHE_DIR, "dartquant_swd_comparison")
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

try:
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    print("✓ Modelscope libraries imported successfully")
except ImportError:
    print("✗ Modelscope library missing. Install with: pip install modelscope")
    sys.exit(1)

# Hyperparameters
SEQ_LENGTH = 2048
NUM_SAMPLES = 128
DTYPE_MODEL = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
DTYPE_OPT = torch.float32  # Optimization MUST be float32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1.5e-3  
MOMENTUM = 0.9
EPOCHS = 20 
BATCH_SIZE = 64
OPTIMIZER_NAME = 'sgd'

# ============================================================================
# LOSS FUNCTIONS (The Core Methodology)
# ============================================================================

def calc_whip_loss(outputs):
    """
    Original DartQuant Whip Loss:
    L = sum( exp(-|x|) )
    Promotes pushing values away from zero (repulsion).
    """
    # outputs shape: (Batch, Hidden_Dim)
    return torch.sum(torch.exp((-outputs.abs())), dim=-1, keepdim=True).mean()

def calc_swd_loss(outputs):
    """
    Proposed Wasserstein Distance (W2^2) Loss:
    Matches the distribution to a perfect Uniform[-b, b] while preserving energy.
    """
    # 1. Flatten all activations in the batch (Treat as 1D distribution)
    # outputs shape: (Batch, Hidden_Dim) -> (N,)
    x_flat = outputs.view(-1)
    N = x_flat.numel()
    
    # 2. Sort current activations (Quantile Function of Input)
    x_sorted, _ = torch.sort(x_flat)
    
    # 3. Dynamic Target Generation (Constraint: Energy Conservation)
    # The rotation R cannot change the L2 norm (Energy).
    # A Uniform[-b, b] distribution has expected energy E[x^2] = b^2 / 3.
    # We set b = sqrt(3) * RMS(x) to match the energy of the input.
    with torch.no_grad():
        rms = torch.sqrt(torch.mean(x_flat ** 2))
        b = math.sqrt(3) * rms
        
        # Generate ideal Uniform Quantiles
        # linspace creates a perfectly flat distribution
        target = torch.linspace(-b, b, steps=N, device=DEVICE)
    
    # 4. Calculate MSE between Sorted Input and Target
    loss = F.mse_loss(x_sorted, target)
    
    return loss

# ============================================================================
# UTILS
# ============================================================================

def random_orthogonal_matrix(size, device):
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q

class R1_QR(nn.Module):
    def __init__(self, hidden_size: int):
        super(R1_QR, self).__init__()
        self.hidden_size = hidden_size
        self.matrix = nn.Parameter(torch.eye(hidden_size, dtype=DTYPE_OPT))
        self.rotate = None

    def forward(self, x):
        self.rotate, _ = torch.linalg.qr(self.matrix, mode='complete')
        o_x = torch.matmul(x, self.rotate)
        return o_x

def train_rotation(activations, hidden_size, loss_type="whip", label=""):
    """
    Train rotation matrix using specified loss function.
    loss_type: 'whip' or 'swd'
    """
    # 1. Prepare Data
    activations = np.nan_to_num(activations, nan=0.0, posinf=65504, neginf=-65504)
    
    dataset = TensorDataset(torch.tensor(activations, dtype=DTYPE_OPT, device=DEVICE))
    sampler = RandomSampler(dataset) 
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE)

    # 2. Initialize Model
    model = R1_QR(hidden_size=hidden_size).to(DEVICE)
    print(f"    [{label}] Init Orthogonal Matrix...")
    init_matrix = random_orthogonal_matrix(hidden_size, DEVICE).float()
    model.matrix.data = init_matrix

    # 3. Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)

    # 4. Training Loop
    model.train()
    
    print(f"    [{label}] Starting training with {loss_type.upper()} Loss...")
    
    for epoch in range(EPOCHS):
        loss_log = []
        for batch_samples in dataloader:
            x = batch_samples[0].float()
            x = x.reshape(-1, hidden_size)
            
            outputs = model(x)
            
            # --- SWITCH LOSS HERE ---
            if loss_type == "whip":
                loss = calc_whip_loss(outputs)
            elif loss_type == "swd":
                loss = calc_swd_loss(outputs)
            else:
                raise ValueError("Unknown loss type")
            # ------------------------
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_log.append(loss.detach())

        scheduler.step()
        
        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            mean_loss = torch.stack(loss_log).mean()
            print(f"      Epoch {epoch+1:02d}/{EPOCHS} | Loss: {mean_loss.item():.6f}")

    with torch.no_grad():
        Q, _ = torch.linalg.qr(model.matrix, mode='complete')
    
    return Q.detach().cpu()

# ============================================================================
# ANALYSIS TOOLS
# ============================================================================

class ActivationCollector:
    def __init__(self, num_samples: int):
        self.num_samples = num_samples
        self.activations = []
        self.hook_handle = None

    def get_hook(self):
        def hook(module, input, output):
            tensor = input[0] if isinstance(input, tuple) else input
            if isinstance(tensor, torch.Tensor):
                flat = tensor.detach().float().cpu().reshape(-1, tensor.shape[-1])
                # Randomly sample to save memory if batch is huge
                if flat.shape[0] > 100:
                    indices = torch.randperm(flat.shape[0])[:100]
                    flat = flat[indices]
                self.activations.append(flat)
        return hook

    def register_hook(self, layer):
        self.activations = []
        self.hook_handle = layer.register_forward_hook(self.get_hook())

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()
    
    def get_results(self) -> np.ndarray:
        if not self.activations: return np.array([])
        acts = torch.cat(self.activations, dim=0)
        if acts.shape[0] > self.num_samples:
            indices = torch.randperm(acts.shape[0])[:self.num_samples]
            acts = acts[indices]
        return acts.numpy()

def get_calibration_data(tokenizer, seq_len=2048):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, seq_len))
    return input_ids

def plot_comparison(original, rotated_whip, rotated_swd, layer_idx, model_name):
    """
    3-Panel Comparison: Original vs Whip vs SWD
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    
    def clean(data):
        return data[np.abs(data) < 50] # Zoom in to relevant range

    d_orig = clean(original).flatten()
    d_whip = clean(rotated_whip).flatten()
    d_swd = clean(rotated_swd).flatten()

    # Common settings
    bins = 150
    alpha = 0.6
    
    # 1. Original
    axes[0].hist(d_orig, bins=bins, color='gray', density=True, log=True)
    axes[0].set_title(f"Layer {layer_idx}: Original (Laplacian-like)")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Log Density")
    axes[0].grid(True, alpha=0.3)

    # 2. Whip Loss Result
    axes[1].hist(d_whip, bins=bins, color='orange', density=True, log=True)
    axes[1].set_title(f"Layer {layer_idx}: Whip Loss Result\n(Note: Potential Twin Peaks)")
    axes[1].set_xlabel("Value")
    axes[1].grid(True, alpha=0.3)
    # Overlay original faintly for context
    axes[1].hist(d_orig, bins=bins, color='gray', density=True, log=True, alpha=0.1)

    # 3. SWD Loss Result
    axes[2].hist(d_swd, bins=bins, color='green', density=True, log=True)
    axes[2].set_title(f"Layer {layer_idx}: SWD (W2^2) Result\n(Target: Flattened Uniform)")
    axes[2].set_xlabel("Value")
    axes[2].grid(True, alpha=0.3)
    # Overlay original faintly
    axes[2].hist(d_orig, bins=bins, color='gray', density=True, log=True, alpha=0.1)
    
    # Calculate Kurtosis for metric comparison
    k_orig = kurtosis(d_orig)
    k_whip = kurtosis(d_whip)
    k_swd = kurtosis(d_swd)
    
    plt.suptitle(f"Methodology Comparison - Layer {layer_idx}\nKurtosis: Orig={k_orig:.2f} | Whip={k_whip:.2f} | SWD={k_swd:.2f} (Target ~1.8)", fontsize=16)
    
    save_path = os.path.join(OUTPUT_DIR, f"Compare_Whip_vs_SWD_L{layer_idx}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✓ Plot saved: {save_path}")

def kurtosis(x):
    return np.mean((x - np.mean(x))**4) / (np.std(x)**4)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Starting DartQuant Methodology Comparison (Whip vs SWD)...")
    print(f"Model: {MODEL_NAME}")
    
    # 确定加载路径：优先使用本地路径，否则使用 Modelscope ID
    if os.path.exists(LOCAL_MODEL_PATH):
        load_path = LOCAL_MODEL_PATH
        print(f"Loading from local path: {load_path}")
    else:
        load_path = MODEL_NAME
        print(f"Loading from Modelscope: {load_path}")
    
    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        load_path, torch_dtype=DTYPE_MODEL, device_map="auto", trust_remote_code=True, cache_dir=CACHE_DIR
    )
    input_ids = get_calibration_data(tokenizer, SEQ_LENGTH).to(DEVICE)
    
    # Select Layers to Analyze
    num_layers = model.config.num_hidden_layers
    layers_to_analyze = [0, num_layers // 2, num_layers - 1] # First, Middle, Last
    
    for idx in layers_to_analyze:
        print(f"\n================ Analyzing Layer {idx} ================")
        
        # 1. Collect Activations
        # Assuming typical Llama/Qwen structure. Adjust 'mlp.up_proj' if architecture differs.
        try:
            layer = model.model.layers[idx].mlp.up_proj 
        except:
            print(f"Skipping Layer {idx}: Architecture mismatch (cannot find mlp.up_proj)")
            continue

        c = ActivationCollector(NUM_SAMPLES)
        c.register_hook(layer)
        
        with torch.no_grad():
            model(input_ids)
        
        original_acts = c.get_results()
        c.remove_hook()
        
        if len(original_acts) == 0:
            print("No activations collected.")
            continue
            
        dim = original_acts.shape[-1]
        print(f"    Vector Dim: {dim}")
        
        # 2. Train Rotation using WHIP Loss
        print("    --> Phase 1: Training with Whip Loss...")
        Q_whip = train_rotation(original_acts, dim, loss_type="whip", label="Whip")
        
        # 3. Train Rotation using SWD Loss
        print("    --> Phase 2: Training with SWD Loss...")
        Q_swd = train_rotation(original_acts, dim, loss_type="swd", label="SWD")
        
        # 4. Apply Rotations
        rot_whip_acts = np.matmul(original_acts, Q_whip.numpy())
        rot_swd_acts = np.matmul(original_acts, Q_swd.numpy())
        
        # 5. Visualize Comparison
        plot_comparison(original_acts, rot_whip_acts, rot_swd_acts, idx, MODEL_NAME.split("/")[-1])
        
    print("\n✓ Experiment Complete. Check output folder.")

if __name__ == "__main__":
    main()