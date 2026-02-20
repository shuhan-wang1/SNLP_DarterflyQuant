"""
Smoothing utilities for JointQuant
Computes smooth scaling factors (SmoothQuant) to reduce quantization error.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import tqdm
import logging

from . import utils


def compute_smooth_scale(
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    Compute SmoothQuant scaling factors for the model.
    
    SmoothQuant migrates quantization difficulty from activations to weights:
    Y = (X * diag(s)^-1) @ (diag(s) @ W) = X_smooth @ W_smooth
    
    The smooth scale s is computed as:
    s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
    
    Where alpha controls the migration strength:
    - alpha = 0: All difficulty on weights
    - alpha = 1: All difficulty on activations (no smoothing)
    - alpha = 0.5: Balanced (recommended for W4A4)
    
    Args:
        model: The model to compute scales for
        activations: Dictionary of activation tensors {layer_name: activations}
        alpha: Migration strength (0.5 recommended for W4A4)
    
    Returns:
        Dictionary of smooth scales {layer_name: scale_tensor}
    """
    logging.info(f"Computing smooth scales with alpha={alpha}...")
    
    smooth_scales = {}
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = hidden_size // num_heads
    
    for layer_idx in tqdm(range(num_layers), desc="Computing smooth scales"):
        layer = model.model.layers[layer_idx]
        
        # =====================================================================
        # MLP Smoothing (down_proj input)
        # =====================================================================
        # The activation entering down_proj has shape [batch, seq, intermediate_size]
        # We compute per-channel max for smoothing
        
        down_proj_key = f"model.layers.{layer_idx}.mlp.down_proj"
        if down_proj_key in activations:
            # Get activation statistics
            acts = activations[down_proj_key]
            # Per-channel max absolute value
            act_max = acts.abs().max(dim=0)[0]  # [intermediate_size]
            
            # Get weight statistics
            weight = layer.mlp.down_proj.weight.data.float()  # [hidden_size, intermediate_size]
            weight_max = weight.abs().max(dim=0)[0]  # [intermediate_size]
            
            # Compute smooth scale
            # s = act_max^alpha / weight_max^(1-alpha)
            # Clamp to avoid division by zero
            act_max = torch.clamp(act_max, min=1e-5)
            weight_max = torch.clamp(weight_max, min=1e-5)
            
            smooth = torch.pow(act_max, alpha) / torch.pow(weight_max, 1 - alpha)
            smooth = torch.clamp(smooth, min=1e-5)
            
            smooth_scales[f'model.layers.{layer_idx}.mlp.down_smooth'] = smooth.cpu()
        
        # =====================================================================
        # Attention Smoothing (o_proj input)
        # =====================================================================
        # The activation entering o_proj has shape [batch, seq, hidden_size]
        # For per-head smoothing, we reshape to [num_heads, head_dim]
        
        o_proj_key = f"model.layers.{layer_idx}.self_attn.o_proj"
        if o_proj_key in activations:
            acts = activations[o_proj_key]
            
            # Reshape to per-head
            acts_reshaped = acts.reshape(-1, num_heads, head_dim)
            
            # Per-head, per-dim max
            # Average over heads to get [num_kv_heads, head_dim] for GQA
            n_rep = num_heads // num_kv_heads
            acts_kv = acts_reshaped.reshape(-1, num_kv_heads, n_rep, head_dim)
            act_max = acts_kv.abs().amax(dim=(0, 2))  # [num_kv_heads, head_dim]
            
            # Get weight statistics
            weight = layer.self_attn.o_proj.weight.data.float()  # [hidden_size, hidden_size]
            weight_reshaped = weight.reshape(hidden_size, num_heads, head_dim)
            weight_kv = weight_reshaped.reshape(hidden_size, num_kv_heads, n_rep, head_dim)
            weight_max = weight_kv.abs().amax(dim=(0, 2))  # [num_kv_heads, head_dim]
            
            # Compute smooth scale
            act_max = torch.clamp(act_max, min=1e-5)
            weight_max = torch.clamp(weight_max, min=1e-5)
            
            smooth = torch.pow(act_max, alpha) / torch.pow(weight_max, 1 - alpha)
            smooth = torch.clamp(smooth, min=1e-5)
            
            smooth_scales[f'model.layers.{layer_idx}.self_attn.o_smooth'] = smooth.cpu()
    
    logging.info(f"Computed smooth scales for {len(smooth_scales)} modules")
    return smooth_scales


def compute_simple_smooth_scale(
    model: nn.Module,
    calibration_data: torch.Tensor,
    device: torch.device,
    alpha: float = 0.5,
    batch_size: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Compute smooth scales by running calibration data through the model.
    This is a simplified version that collects activations on-the-fly.
    
    Args:
        model: The model to compute scales for
        calibration_data: Input token IDs [nsamples, seqlen]
        device: Device to run on
        alpha: Migration strength
        batch_size: Batch size for forward passes
    
    Returns:
        Dictionary of smooth scales
    """
    logging.info("Computing smooth scales from calibration data...")
    
    model.eval()
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    
    # Initialize accumulators for max values
    down_act_max = {i: None for i in range(num_layers)}
    o_act_max = {i: None for i in range(num_layers)}
    
    # Hooks to capture activations
    handles = []
    
    def make_down_hook(layer_idx):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.detach().float().reshape(-1, x.shape[-1])
            x_max = x.abs().max(dim=0)[0]
            if down_act_max[layer_idx] is None:
                down_act_max[layer_idx] = x_max
            else:
                down_act_max[layer_idx] = torch.maximum(down_act_max[layer_idx], x_max)
        return hook
    
    def make_o_hook(layer_idx):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            x = x.detach().float().reshape(-1, x.shape[-1])
            x_max = x.abs().max(dim=0)[0]
            if o_act_max[layer_idx] is None:
                o_act_max[layer_idx] = x_max
            else:
                o_act_max[layer_idx] = torch.maximum(o_act_max[layer_idx], x_max)
        return hook
    
    # Register hooks
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        handles.append(layer.mlp.down_proj.register_forward_hook(make_down_hook(layer_idx)))
        handles.append(layer.self_attn.o_proj.register_forward_hook(make_o_hook(layer_idx)))
    
    # Run forward passes
    with torch.no_grad():
        for i in tqdm(range(0, len(calibration_data), batch_size), desc="Collecting smooth stats"):
            batch = calibration_data[i:i+batch_size].to(device)
            model(batch)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Compute smooth scales
    smooth_scales = {}
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        # MLP smooth
        if down_act_max[layer_idx] is not None:
            act_max = torch.clamp(down_act_max[layer_idx], min=1e-5)
            weight = layer.mlp.down_proj.weight.data.float()
            weight_max = torch.clamp(weight.abs().max(dim=0)[0], min=1e-5)
            smooth = torch.pow(act_max, alpha) / torch.pow(weight_max, 1 - alpha)
            smooth_scales[f'model.layers.{layer_idx}.mlp.down_smooth'] = torch.clamp(smooth, min=1e-5).cpu()
        
        # O_proj smooth (simplified - per-channel, not per-head)
        if o_act_max[layer_idx] is not None:
            act_max = torch.clamp(o_act_max[layer_idx], min=1e-5)
            weight = layer.self_attn.o_proj.weight.data.float()
            weight_max = torch.clamp(weight.abs().max(dim=0)[0], min=1e-5)
            smooth = torch.pow(act_max, alpha) / torch.pow(weight_max, 1 - alpha)
            
            # Reshape to per-head for consistency with rotation
            num_heads = model.config.num_attention_heads
            num_kv_heads = model.config.num_key_value_heads
            head_dim = hidden_size // num_heads
            n_rep = num_heads // num_kv_heads
            
            smooth_reshaped = smooth.reshape(num_heads, head_dim)
            smooth_kv = smooth_reshaped.reshape(num_kv_heads, n_rep, head_dim).mean(dim=1)
            smooth_scales[f'model.layers.{layer_idx}.self_attn.o_smooth'] = smooth_kv.cpu()
    
    logging.info(f"Computed {len(smooth_scales)} smooth scales")
    return smooth_scales
