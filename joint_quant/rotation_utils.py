"""
Rotation utilities for JointQuant
Includes R1 (global), R2 (per-head) rotation and Smoothing
Adapted from DartQuant/fake_quant/rotation_utils.py
"""

import torch
import torch.nn as nn
import transformers
import logging
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from . import utils
from . import model_utils
from . import hadamard_utils


# ============================================================================
# LayerNorm Fusion
# ============================================================================

def fuse_ln_linear(layernorm: nn.Module, linear_layers: List[nn.Linear]) -> None:
    """
    Fuse the linear operations in Layernorm into the adjacent linear blocks.
    After fusion, the layernorm becomes a simple RMS normalization.
    
    New_weight = Old_weight * gamma
    New_bias = Old_bias + Old_weight @ beta (if beta exists)
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        
        # Fuse weight: W_new = W_old * gamma
        W_ = linear.weight.data.double()
        if hasattr(layernorm, 'weight') and layernorm.weight is not None:
            linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        
        # Fuse bias: b_new = b_old + W @ beta (if beta exists)
        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model: nn.Module) -> None:
    """
    Fuse LlamaRMSNorm into adjacent Linear layers.
    This makes the Norm layer rotation-invariant (identity scaling).
    
    CRITICAL: This must be called BEFORE applying rotations!
    """
    logging.info("Fusing LayerNorms into adjacent linear layers...")
    
    model_type = model_utils.get_model_type(model)
    
    # 1. Center Embedding weights (subtract mean)
    for W in model_utils.get_embeddings(model, model_type):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)
    logging.info("  - Centered embedding weights")
    
    # 2. Fuse layer norms in each transformer layer
    layers = model_utils.get_transformer_layers(model, model_type)
    for layer_idx, layer in enumerate(layers):
        # Fuse input_layernorm into q, k, v projections
        fuse_ln_linear(layer.input_layernorm, [
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj
        ])
        
        # Fuse post_attention_layernorm into up, gate projections
        fuse_ln_linear(layer.post_attention_layernorm, [
            layer.mlp.up_proj,
            layer.mlp.gate_proj
        ])
        
        # Replace original norms with simple RMSN
        hidden_size = model.config.hidden_size
        input_ln_eps = getattr(layer.input_layernorm, 'variance_epsilon',
                               getattr(layer.input_layernorm, 'eps', 1e-5))
        post_ln_eps = getattr(layer.post_attention_layernorm, 'variance_epsilon',
                              getattr(layer.post_attention_layernorm, 'eps', 1e-5))
        
        layer.input_layernorm = model_utils.RMSN(hidden_size, input_ln_eps)
        layer.post_attention_layernorm = model_utils.RMSN(hidden_size, post_ln_eps)
    
    logging.info(f"  - Fused LayerNorms in {len(layers)} layers")
    
    # 3. Fuse Final Norm into LM Head
    # CRITICAL: Handle weight tying!
    pre_head_ln = model_utils.get_pre_head_layernorm(model, model_type)
    lm_head = model_utils.get_lm_head(model, model_type)
    
    # Check if weights are tied
    if hasattr(model.model, 'embed_tokens'):
        weights_are_tied = (lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr())
        
        if weights_are_tied:
            logging.info("  - Detected tied weights, untying before fusion...")
            # Create independent copy for lm_head
            lm_head_weight_copy = lm_head.weight.data.clone()
            old_lm_head = lm_head
            model.lm_head = nn.Linear(
                old_lm_head.in_features,
                old_lm_head.out_features,
                bias=old_lm_head.bias is not None,
                dtype=old_lm_head.weight.dtype,
                device=old_lm_head.weight.device
            )
            model.lm_head.weight.data = lm_head_weight_copy
            if old_lm_head.bias is not None:
                model.lm_head.bias.data = old_lm_head.bias.data.clone()
            lm_head = model.lm_head
    
    # Now safe to fuse
    fuse_ln_linear(pre_head_ln, [lm_head])
    final_norm_eps = getattr(pre_head_ln, 'variance_epsilon',
                             getattr(pre_head_ln, 'eps', 1e-5))
    model.model.norm = model_utils.RMSN(model.config.hidden_size, final_norm_eps)
    logging.info("  - Fused final norm into lm_head")
    
    # 4. Replace all remaining RMSNorm with RMSN
    model_utils.replace_modules(
        model,
        transformers.models.llama.modeling_llama.LlamaRMSNorm,
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )
    
    logging.info("LayerNorm fusion complete!")


# ============================================================================
# Rotation Application
# ============================================================================

def rotate_embeddings(model: nn.Module, Q: torch.Tensor) -> None:
    """Rotate the embedding layer: W_new = W @ Q"""
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_head(model: nn.Module, Q: torch.Tensor) -> None:
    """Rotate the lm_head: W_new = W @ Q"""
    model_type = model_utils.model_type_extractor(model)
    lm_head = model_utils.get_lm_head(model, model_type)
    
    # Check if tied to embeddings
    if hasattr(model.model, 'embed_tokens'):
        if lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr():
            logging.info("  Skipping lm_head rotation (tied to embedding)")
            return
    
    dtype = lm_head.weight.data.dtype
    W_ = lm_head.weight.data.to(device=utils.DEV, dtype=torch.float64)
    lm_head.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer: nn.Module, Q: torch.Tensor) -> None:
    """Rotate Q, K, V projections: W_new = W @ Q"""
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer: nn.Module, Q: torch.Tensor) -> None:
    """Rotate O projection: W_new = Q^T @ W"""
    W = layer.self_attn.o_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer: nn.Module, Q: torch.Tensor, layer_idx: int,
                     smooth_scale: Optional[Dict] = None) -> None:
    """
    Rotate MLP input (up_proj, gate_proj): W_new = W @ Q
    Also apply smoothing if provided.
    """
    mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
    
    for W in mlp_inputs:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
    
    # Apply smoothing to up_proj if provided
    if smooth_scale is not None:
        smooth_key = f'model.layers.{layer_idx}.mlp.down_smooth'
        if smooth_key in smooth_scale:
            up_proj = layer.mlp.up_proj
            dtype = up_proj.weight.dtype
            down_smooth = smooth_scale[smooth_key].to(device=utils.DEV, dtype=torch.float64)
            up_proj.weight.data = torch.div(
                up_proj.weight.data.to(device=utils.DEV, dtype=torch.float64).t(),
                down_smooth
            ).t().to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer: nn.Module, Q: torch.Tensor, layer_idx: int,
                      smooth_scale: Optional[Dict] = None) -> None:
    """
    Rotate MLP output (down_proj): W_new = Q^T @ W
    Also apply smoothing if provided.
    """
    W = layer.mlp.down_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_)
    
    # Apply smoothing if provided
    if smooth_scale is not None:
        smooth_key = f'model.layers.{layer_idx}.mlp.down_smooth'
        if smooth_key in smooth_scale:
            down_smooth = smooth_scale[smooth_key].to(device=utils.DEV, dtype=torch.float64)
            W.weight.data = torch.mul(W.weight.data, down_smooth)
    
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)
    
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer: nn.Module, head_dim: int, num_heads: int, num_kv_heads: int,
                   layer_idx: int, R2: torch.Tensor,
                   smooth_scale: Optional[Dict] = None) -> None:
    """
    Apply R2 per-head rotation to V and O projections.
    
    V_proj: R2 on right (output side)
    O_proj: R2^T on left (input side)
    
    Args:
        layer: Transformer layer
        head_dim: Dimension per head
        num_heads: Total number of attention heads
        num_kv_heads: Number of key-value heads (for GQA)
        layer_idx: Layer index
        R2: Rotation matrix [num_kv_heads, head_dim, head_dim]
        smooth_scale: Optional smoothing scales
    """
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj
    
    hidden_size = num_heads * head_dim
    n_rep = num_heads // num_kv_heads
    
    # Rotate V projection (R2 on output)
    dtype = v_proj.weight.dtype
    W_v = v_proj.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W_v = W_v.t().reshape(-1, num_kv_heads, head_dim).transpose(0, 1)
    
    # Apply smoothing if provided
    if smooth_scale is not None:
        smooth_key = f'model.layers.{layer_idx}.self_attn.o_smooth'
        if smooth_key in smooth_scale:
            o_smooth = smooth_scale[smooth_key].to(device=utils.DEV, dtype=torch.float64)
            W_v = W_v / o_smooth.view(num_kv_heads, 1, head_dim)
    
    W_v = torch.matmul(W_v, R2)
    W_v = W_v.transpose(0, 1).reshape(-1, num_kv_heads * head_dim).t()
    v_proj.weight.data = W_v.to(device="cpu", dtype=dtype)
    
    # Rotate O projection (R2^T on input)
    W_o = o_proj.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W_o = W_o.reshape(-1, num_heads, head_dim).transpose(0, 1)
    
    # Expand R2 for GQA
    if len(R2.shape) == 3:
        R2_exp = R2.unsqueeze(1).expand(num_kv_heads, n_rep, head_dim, head_dim)
        R2_exp = R2_exp.reshape(num_heads, head_dim, head_dim)
    else:
        R2_exp = R2
    
    # Apply smoothing if provided
    if smooth_scale is not None:
        smooth_key = f'model.layers.{layer_idx}.self_attn.o_smooth'
        if smooth_key in smooth_scale:
            o_smooth = smooth_scale[smooth_key].to(device=utils.DEV, dtype=torch.float64)
            o_smooth_exp = o_smooth.unsqueeze(1).expand(num_kv_heads, n_rep, head_dim)
            o_smooth_exp = o_smooth_exp.reshape(num_heads, head_dim)
            W_o = W_o * o_smooth_exp.view(num_heads, 1, head_dim)
    
    W_o = torch.matmul(W_o, R2_exp)
    W_o = W_o.transpose(0, 1).reshape(-1, num_heads * head_dim)
    o_proj.weight.data = W_o.to(device="cpu", dtype=dtype)


# ============================================================================
# Complete Model Rotation
# ============================================================================

@torch.inference_mode()
def rotate_model(model: nn.Module, R1: torch.Tensor,
                 R2_matrices: Optional[Dict[str, torch.Tensor]] = None,
                 smooth_scale: Optional[Dict] = None) -> None:
    """
    Apply rotation matrices to the entire model.
    
    Args:
        model: The model to rotate
        R1: Global rotation matrix [hidden_size, hidden_size]
        R2_matrices: Per-layer R2 matrices {layer_name: [num_kv_heads, head_dim, head_dim]}
        smooth_scale: Smoothing factors from calibration
    """
    logging.info("Applying rotations to model...")
    
    Q = R1.to(device=utils.DEV, dtype=torch.float64)
    
    config = model.config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    
    model_type = model_utils.model_type_extractor(model)
    
    # Rotate embeddings and head
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    utils.cleanup_memory(verbose=False)
    
    # Rotate each layer
    layers = model_utils.get_transformer_layers(model, model_type)
    for idx, layer in enumerate(tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layer, Q)
        rotate_attention_output(layer, Q)
        rotate_mlp_input(layer, Q, idx, smooth_scale)
        rotate_mlp_output(layer, Q, idx, smooth_scale)
        
        # Apply R2 if provided
        if R2_matrices is not None:
            r2_key = f"model.layers.{idx}.self_attn.o_proj"
            if r2_key in R2_matrices:
                R2 = R2_matrices[r2_key].to(device=utils.DEV, dtype=torch.float64)
                rotate_ov_proj(layer, head_dim, num_heads, num_kv_heads, idx, R2, smooth_scale)
    
    logging.info("Rotation complete!")
