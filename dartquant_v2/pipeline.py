"""
Full DartQuant v2 quantization pipeline.

Pipeline steps:
  1.  Load model via UnifiedQuantModel
  2.  Fuse LayerNorms
  3.  Collect calibration data
  4.  Collect & train R1 activations
  5.  Apply R1 to model
  6.  Collect & train R2 activations
  7.  Apply R2 to model
  8.  Handle R3/R4 (Hadamard or Butterfly)
  9.  Add activation quantization wrappers
  10. Weight quantization (INT4 GPTQ/RTN or NF4)
  11. Configure per-layer activation quantization
  12. Evaluate PPL
"""

import os
import sys
import gc
import math
import time
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import transformers
from tqdm import tqdm

# Add DartQuant paths for importing existing modules
_DARTQUANT_PATH = os.path.join(os.path.dirname(__file__), '..', 'DartQuant', 'fake_quant')
_CALIBRATER_PATH = os.path.join(os.path.dirname(__file__), '..', 'DartQuant', 'calibrater')
if _DARTQUANT_PATH not in sys.path:
    sys.path.insert(0, _DARTQUANT_PATH)
if _CALIBRATER_PATH not in sys.path:
    sys.path.insert(0, _CALIBRATER_PATH)

from .unified_model import UnifiedQuantModel, _deep_getattr, _deep_setattr
from .trainers import train_butterfly
from .nf4_quantizer import apply_nf4_to_model

DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Disable TF32 for numerical precision
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


# ---------------------------------------------------------------------------
# Hadamard CUDA kernel compatibility check & monkey-patch
# ---------------------------------------------------------------------------
def _check_and_patch_hadamard():
    """Test the fast_hadamard_transform CUDA kernel; monkey-patch if it fails.

    The `fast_hadamard_transform` package ships pre-compiled CUDA kernels that
    may not match the GPU's compute capability.  If the kernel is unusable we
    replace `hadamard_utils.matmul_hadU_cuda` (used in both `apply_exact_had_to_linear`
    and the runtime `ActQuantWrapper.forward`) with a pure-PyTorch fallback
    so the entire pipeline can still run — just a bit slower.
    """
    if not torch.cuda.is_available():
        return

    try:
        import fast_hadamard_transform  # noqa: F401
        # Quick smoke test on the current GPU
        _test = torch.randn(1, 64, device='cuda', dtype=torch.float32)
        fast_hadamard_transform.hadamard_transform(_test, 1.0 / 8.0)
        logging.info("fast_hadamard_transform CUDA kernel OK.")
        return  # kernel works fine
    except ImportError:
        logging.info("fast_hadamard_transform not installed – will use PyTorch fallback.")
    except RuntimeError as e:
        logging.warning(
            f"fast_hadamard_transform CUDA kernel unusable ({e}). "
            "Monkey-patching with pure-PyTorch fallback."
        )

    # ---- Build the fallback & patch ----
    try:
        import hadamard_utils as _had_mod
    except ImportError:
        return  # DartQuant path not available, nothing to patch

    _original_matmul_hadU = _had_mod.matmul_hadU  # pure-PyTorch version

    def _matmul_hadU_cuda_fallback(X, hadK, K):
        """Drop-in replacement for matmul_hadU_cuda using PyTorch ops."""
        n = X.shape[-1]
        inp = X.clone().view(-1, n, 1)
        out = inp.clone()
        while inp.shape[1] > K:
            inp = inp.view(inp.shape[0], inp.shape[1] // 2, 2, inp.shape[2])
            out = out.view(inp.shape)
            out[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
            out[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
            out = out.view(inp.shape[0], inp.shape[1], -1)
            inp, out = out, inp
        del out
        if K > 1:
            inp = hadK.view(1, K, K).to(inp) @ inp
        return inp.view(X.shape) / torch.tensor(n).sqrt()

    # Monkey-patch the module-level CUDA function so all callers
    # (apply_exact_had_to_linear, ActQuantWrapper.forward, etc.) use the fallback.
    _had_mod.matmul_hadU_cuda = _matmul_hadU_cuda_fallback
    logging.info("Patched hadamard_utils.matmul_hadU_cuda → PyTorch fallback.")

    # Also patch fast_hadamard_transform.hadamard_transform for direct callers
    # (e.g. quant_utils.py line 270 calls it directly)
    try:
        import fast_hadamard_transform as _fht_mod

        def _hadamard_transform_fallback(x, scale=1.0):
            """Pure-PyTorch replacement for fast_hadamard_transform.hadamard_transform."""
            shape = x.shape
            n = shape[-1]
            x = x.contiguous().view(-1, n)
            inp = x.unsqueeze(-1)
            out = inp.clone()
            k = 1
            while k < n:
                inp = inp.view(inp.shape[0], n // (2 * k), 2, k)
                out = out.view(inp.shape)
                out[:, :, 0, :] = inp[:, :, 0, :] + inp[:, :, 1, :]
                out[:, :, 1, :] = inp[:, :, 0, :] - inp[:, :, 1, :]
                out = out.view(inp.shape[0], -1, k)
                inp, out = out, inp
                k *= 2
            del out
            return (inp.view(shape) * scale)

        _fht_mod.hadamard_transform = _hadamard_transform_fallback
        logging.info("Patched fast_hadamard_transform.hadamard_transform → PyTorch fallback.")
    except ImportError:
        pass


# Run the check at import time so the patch is in place before any pipeline code
_check_and_patch_hadamard()


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================
# LayerNorm Fusion (adapted from rotation_utils.py for UnifiedQuantModel)
# ============================================================================

def _fuse_ln_linear(layernorm, linear_layers):
    """Fuse LayerNorm parameters into adjacent linear layers."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)
        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64)
                )
            linear.bias.data = (
                linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            ).to(linear_dtype)


def _bake_mean_into_linear(linear):
    """Subtract mean from weights/bias (OPT-style)."""
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = (W_ - W_.mean(dim=-2, keepdim=True)).to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = (b_ - b_.mean()).to(linear_dtype)


class _RMSN(nn.Module):
    """Replacement RMS Norm (identity weights, used after fusion)."""
    def __init__(self, mean_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def fuse_layer_norms(umodel: UnifiedQuantModel):
    """Fuse LayerNorm weights into adjacent linear layers.

    Adapted from DartQuant/fake_quant/rotation_utils.py:fuse_layer_norms
    """
    model = umodel.model

    # Fuse embeddings (subtract mean)
    for W in umodel.get_embeddings():
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = umodel.get_layers()

    for layer in layers:
        # Fuse input layernorm into attention Q/K/V
        input_ln = umodel.get_input_ln(layer)
        q, k, v, o = umodel.get_attn_projs(layer)
        _fuse_ln_linear(input_ln, [q, k, v])

        # Fuse post-attention layernorm into MLP inputs
        post_attn_ln = umodel.get_post_attn_ln(layer)
        mlp_inputs = umodel.get_mlp_input_projs(layer)
        _fuse_ln_linear(post_attn_ln, mlp_inputs)

        # OPT-style mean baking
        if umodel.needs_mean_baking:
            _bake_mean_into_linear(o)
            mlp_out = umodel.get_mlp_output_proj(layer)
            _bake_mean_into_linear(mlp_out)

    # Fuse pre-head norm into lm_head
    _fuse_ln_linear(umodel.get_pre_head_norm(), [umodel.get_lm_head()])

    # Replace all norm layers with identity RMSN
    _replace_norms(model, umodel)
    logging.info("LayerNorm fusion complete.")


def _replace_norms(model, umodel: UnifiedQuantModel):
    """Replace all norm layers with RMSN after fusion."""
    norm_class_name = umodel.arch.norm_class_name
    hidden_size = umodel.hidden_size

    if norm_class_name == "LlamaRMSNorm":
        target_class = transformers.models.llama.modeling_llama.LlamaRMSNorm
    elif norm_class_name == "LayerNorm":
        target_class = nn.LayerNorm
    else:
        logging.warning(f"Unknown norm class: {norm_class_name}, skipping replacement")
        return

    def _replace_recursive(module):
        for name, child in module.named_children():
            if isinstance(child, target_class):
                setattr(module, name, _RMSN(hidden_size))
            else:
                _replace_recursive(child)

    _replace_recursive(model)


# ============================================================================
# Activation Collection
# ============================================================================

def collect_activations(model, calibration_data, target_names, device):
    """Collect input activations at specified module paths.

    Args:
        model: The model
        calibration_data: Input ids tensor (nsamples, seqlen)
        target_names: List of module paths like "model.layers.0.mlp.up_proj"
        device: Device for computation

    Returns:
        Dict mapping target_name to activation tensor
    """
    activations = {name: [] for name in target_names}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            tensor = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(tensor, torch.Tensor):
                flat = tensor.detach().float().cpu().reshape(-1, tensor.shape[-1])
                if flat.shape[0] > 512:
                    indices = torch.randperm(flat.shape[0])[:512]
                    flat = flat[indices]
                activations[name].append(flat)
        return hook

    for name in target_names:
        try:
            module = model.get_submodule(name)
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
        except AttributeError:
            logging.warning(f"Module {name} not found, skipping")

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(calibration_data.shape[0]),
                      desc="  Calibration forward passes", leave=False):
            inp = calibration_data[i:i+1].to(device)
            try:
                model(inp)
            except Exception:
                pass  # Some models may error on certain inputs

    for h in hooks:
        h.remove()

    result = {}
    for name in target_names:
        if activations[name]:
            result[name] = torch.cat(activations[name], dim=0)
        else:
            logging.warning(f"No activations collected for {name}")

    return result


def collect_r3_activations(model, calibration_data, umodel, device):
    """Collect Q/K activations after RoPE for butterfly R3 training.

    Hooks into the self-attention module to capture post-RoPE Q/K.
    """
    activations = []
    hooks = []

    # We need to hook the rope output
    # In Llama, apply_rotary_pos_emb is called inside the attention forward
    # We hook into the attention module and capture the Q/K after projection
    # (pre-RoPE, but that's what R3 acts on in the QK space)
    layers = umodel.get_layers()
    for layer in layers:
        q_proj, k_proj, _, _ = umodel.get_attn_projs(layer)
        def make_hook(name):
            def hook(module, inp, out):
                flat = out.detach().float().cpu().reshape(-1, out.shape[-1])
                if flat.shape[0] > 256:
                    indices = torch.randperm(flat.shape[0])[:256]
                    flat = flat[indices]
                activations.append(flat)
            return hook
        # Hook Q projection output (R3 acts on head_dim chunks)
        hooks.append(q_proj.register_forward_hook(make_hook('q')))

    model.eval()
    with torch.no_grad():
        for i in range(min(calibration_data.shape[0], 16)):
            inp = calibration_data[i:i+1].to(device)
            try:
                model(inp)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    if activations:
        all_acts = torch.cat(activations, dim=0)
        # Reshape to head_dim chunks
        head_dim = umodel.head_dim
        all_acts = all_acts.reshape(-1, head_dim)
        return all_acts
    return None


def _r4_hook_path(umodel: UnifiedQuantModel, layer_idx: int) -> str:
    """Return the module path to hook for R4 activation collection.

    Dense: layer.mlp.down_proj
    MoE:   layer.block_sparse_moe.experts.0.w2  (representative expert 0)
    A single hook on experts[0] suffices because all experts share the same R4.
    """
    base = f"{umodel.arch.layers_path}.{layer_idx}"
    if umodel.arch.is_moe:
        return f"{base}.{umodel.arch.experts_attr}.0.{umodel.arch.expert_down_proj_attr}"
    return f"{base}.{umodel.arch.mlp_down_proj_attr}"


def collect_r4_activations(model, calibration_data, umodel, device):
    """Collect down_proj input activations for butterfly R4 training."""
    target_names = []
    for i in range(umodel.num_layers):
        target_names.append(_r4_hook_path(umodel, i))

    acts = collect_activations(model, calibration_data, target_names, device)
    if acts:
        all_acts = torch.cat(list(acts.values()), dim=0)
        return all_acts
    return None


def collect_r4_weight_bank(umodel: 'UnifiedQuantModel',
                           max_layers: int = 8) -> torch.Tensor:
    """Collect a bank of down_proj weight matrices for R4 weight-aware recon.

    Samples up to ``max_layers`` down_proj weight matrices (evenly spaced
    across layers) and returns them as a tensor.  Used for the paper's Eq 17
    joint weight+activation reconstruction loss.

    For MoE models, expert 0 is used as the representative since all experts
    share the same R4 rotation.

    Args:
        umodel: UnifiedQuantModel with accessible layers
        max_layers: Maximum number of layers to sample

    Returns:
        Tensor of shape (num_samples, hidden_size, intermediate_size),
        float32, on CPU.
    """
    layers = umodel.get_layers()
    num_layers = len(layers)

    # Sample evenly spaced layer indices
    if num_layers <= max_layers:
        layer_indices = list(range(num_layers))
    else:
        step = num_layers / max_layers
        layer_indices = [int(i * step) for i in range(max_layers)]

    weights = []
    for idx in layer_indices:
        layer = layers[idx]
        down_proj = umodel.get_mlp_output_proj(layer)
        W = down_proj.weight.data.float().cpu()  # (hidden_size, intermediate_size)
        weights.append(W)

    weight_bank = torch.stack(weights, dim=0)
    logging.info(
        f"  R4 weight bank: {weight_bank.shape[0]} layers sampled, "
        f"shape per weight = {tuple(weight_bank.shape[1:])}"
    )
    return weight_bank


# ============================================================================
# Rotation Application
# ============================================================================

def apply_r1_rotation(model, R1, umodel: UnifiedQuantModel, smooth_scale=None):
    """Apply R1 rotation to model weights (offline).

    R1 is fused into: embeddings, lm_head, and per-layer
    attention inputs (Q,K,V), attention output (O), MLP inputs, MLP output.

    Reference: DartQuant/fake_quant/rotation_utils.py:rotate_model
    """
    Q = R1.to(device=DEV, dtype=torch.float64)

    # Rotate embeddings: W_emb = W_emb @ R1
    for W in umodel.get_embeddings():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    # Rotate lm_head: W_head = W_head @ R1
    W = umodel.get_lm_head()
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

    cleanup_memory()

    layers = umodel.get_layers()
    for idx, layer in enumerate(tqdm(layers, desc="Applying R1")):
        q_proj, k_proj, v_proj, o_proj = umodel.get_attn_projs(layer)

        # Rotate attention inputs (Q, K, V): W = W @ R1
        for W in [q_proj, k_proj, v_proj]:
            dtype = W.weight.dtype
            W_ = W.weight.to(device=DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

        # Rotate attention output: W_o = R1^T @ W_o
        dtype = o_proj.weight.data.dtype
        W_ = o_proj.weight.data.to(device=DEV, dtype=torch.float64)
        o_proj.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if o_proj.bias is not None:
            b = o_proj.bias.data.to(device=DEV, dtype=torch.float64)
            o_proj.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

        # Rotate MLP inputs: W = W @ R1
        for W in umodel.get_mlp_input_projs(layer):
            dtype = W.weight.dtype
            W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

        # Rotate MLP output: W_down = R1^T @ W_down
        # For MoE, the same R1^T is applied to every expert's down_proj.
        for W in umodel.get_all_mlp_output_projs(layer):
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=DEV, dtype=torch.float64)
                W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

    cleanup_memory()
    logging.info("R1 rotation applied to model.")


def apply_r1_rotation_per_layer(model, R1_dict: dict,
                                 umodel: UnifiedQuantModel, smooth_scale=None):
    """Apply per-layer R1 rotations to model weights (offline).

    Unlike apply_r1_rotation (which uses a single global R1), each transformer
    layer l uses its own R1_l from R1_dict[l].

    Within each layer the rotation is applied symmetrically:
      Input projections  (Q, K, V, up_proj, gate_proj): W = W @ R1_l
      Output projections (o_proj, down_proj):            W = R1_l^T @ W

    This ensures the layer's internal computation is in the rotated space while
    the residual bypass (x_l = x + layer(x)) remains in the original space —
    the R1_l from the output projection exactly cancels the R1_l baked into the
    input projections for the bypass path.

    Embeddings and lm_head are NOT rotated (no single global rotation exists).

    Args:
        model:    HuggingFace model.
        R1_dict:  dict {layer_idx (int) → Tensor (hidden_size, hidden_size)}.
        umodel:   UnifiedQuantModel for arch-aware accessor methods.
        smooth_scale: Unused; kept for API parity with apply_r1_rotation.
    """
    layers = umodel.get_layers()
    for idx, layer in enumerate(tqdm(layers, desc="Applying per-layer R1")):
        if idx not in R1_dict:
            continue
        Q = R1_dict[idx].to(device=DEV, dtype=torch.float64)

        q_proj, k_proj, v_proj, o_proj = umodel.get_attn_projs(layer)

        # Input sides: W = W @ R1_l
        for W in [q_proj, k_proj, v_proj]:
            dtype = W.weight.dtype
            W_    = W.weight.to(device=DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

        # Output side: W_o = R1_l^T @ W_o
        dtype = o_proj.weight.data.dtype
        W_    = o_proj.weight.data.to(device=DEV, dtype=torch.float64)
        o_proj.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
        if o_proj.bias is not None:
            b = o_proj.bias.data.to(device=DEV, dtype=torch.float64)
            o_proj.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

        # MLP inputs: W = W @ R1_l
        for W in umodel.get_mlp_input_projs(layer):
            dtype = W.weight.dtype
            W_    = W.weight.data.to(device=DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)

        # MLP output: W_down = R1_l^T @ W_down
        for W in umodel.get_all_mlp_output_projs(layer):
            dtype = W.weight.data.dtype
            W_    = W.weight.data.to(device=DEV, dtype=torch.float64)
            W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=DEV, dtype=torch.float64)
                W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

        cleanup_memory()

    logging.info(f"Per-layer R1 rotations applied to model "
                 f"({len(R1_dict)} layers).")


def apply_r2_rotation(model, R2_dict, umodel: UnifiedQuantModel, smooth_scale=None):
    """Apply R2 per-head rotation to V-proj and O-proj.

    Reference: DartQuant/fake_quant/rotation_utils.py:rotate_ov_proj
    """
    head_dim = umodel.head_dim
    kv_heads = umodel.kv_heads
    num_heads = umodel.num_heads

    layers = umodel.get_layers()
    for idx, layer in enumerate(tqdm(layers, desc="Applying R2")):
        key = f"model.layers.{idx}.self_attn.R2"
        if key not in R2_dict:
            continue

        Q_r2 = R2_dict[key].to(device=DEV, dtype=torch.float64)
        if len(Q_r2.shape) == 2:
            Q_r2 = Q_r2.unsqueeze(0).repeat(kv_heads, 1, 1)

        _, _, v_proj, o_proj = umodel.get_attn_projs(layer)

        # V-proj output side: W_v = R2^T @ W_v (per head)
        _apply_multi_head_rotate(v_proj, Q_r2, head_dim, kv_heads, num_heads,
                                 output=True, smooth=None)

        # O-proj input side: W_o = W_o @ R2 (per head)
        _apply_multi_head_rotate(o_proj, Q_r2, head_dim, kv_heads, num_heads,
                                 output=False, smooth=None)

    cleanup_memory()
    logging.info("R2 rotation applied to model.")


def _apply_multi_head_rotate(module, Q, head_dim, kv_head, num_head,
                              output=False, smooth=None):
    """Apply per-head rotation to a linear module.

    Reference: DartQuant/fake_quant/rotation_utils.py:apply_multi_head_rotate
    """
    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    n_rep = num_head // kv_head
    W_ = W_.to(device=DEV, dtype=torch.float64)

    if output:
        W_ = W_.t()
        transposed_shape = W_.shape
        W_ = W_.reshape(-1, kv_head, head_dim).transpose(0, 1)
        W_ = torch.matmul(W_, Q)
        W_ = W_.transpose(0, 1).reshape(transposed_shape).t()
    else:
        W_ = W_.reshape(-1, init_shape[1] // head_dim, head_dim).transpose(0, 1)
        if len(Q.shape) == 3:
            Q_exp = Q[:, None, :, :].expand(kv_head, n_rep, head_dim, head_dim)
            Q_exp = Q_exp.reshape(num_head, head_dim, head_dim)
        else:
            Q_exp = Q
        W_ = torch.matmul(W_, Q_exp)
        W_ = W_.transpose(0, 1).reshape(init_shape)

    module.weight.data = W_.to(device=dev, dtype=dtype)


def apply_r4_hadamard(model, umodel: UnifiedQuantModel):
    """Apply R4 Hadamard to down-proj weights (offline baking).

    Reference: DartQuant/fake_quant/rotation_utils.py:rotate_mlp_output (line 193)

    The CUDA kernel compatibility is handled globally by _check_and_patch_hadamard()
    which monkey-patches the CUDA functions at import time if needed.
    """
    try:
        from hadamard_utils import apply_exact_had_to_linear
    except ImportError:
        logging.warning("hadamard_utils not available, skipping R4 offline baking")
        return

    layers = umodel.get_layers()
    for layer in tqdm(layers, desc="Applying R4 Hadamard"):
        # For MoE: all experts share the same R4; bake into every expert's down_proj.
        for W in umodel.get_all_mlp_output_projs(layer):
            apply_exact_had_to_linear(W, had_dim=-1, output=False)

    logging.info("R4 Hadamard applied to down_proj weights.")


def apply_r4_butterfly(model, butterfly_r4, umodel: UnifiedQuantModel):
    """Apply trained Butterfly R4 to down-proj weights (offline baking)."""
    R4_matrix = butterfly_r4.get_matrix().to(device=DEV, dtype=torch.float64)

    layers = umodel.get_layers()
    for layer in tqdm(layers, desc="Applying R4 Butterfly"):
        # For MoE: all experts share the same R4; bake into every expert's down_proj.
        for W in umodel.get_all_mlp_output_projs(layer):
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=DEV, dtype=torch.float64)
            # W_down_new = W_down @ R4^T (absorb R4 into down_proj input)
            W.weight.data = torch.matmul(W_, R4_matrix.T).to(device="cpu", dtype=dtype)

    logging.info("R4 Butterfly applied to down_proj weights.")


# ============================================================================
# Quantization Wrappers Setup (from existing DartQuant)
# ============================================================================

def setup_quantization_wrappers(model, args, umodel: UnifiedQuantModel):
    """Add activation quantization wrappers and configure per-layer settings.

    Adapted from DartQuant/fake_quant/main_for_test.py lines 69-211
    """
    try:
        import quant_utils
        import hadamard_utils
    except ImportError:
        logging.error("Cannot import quant_utils/hadamard_utils from DartQuant")
        raise

    # Add ActQuantWrapper to all linear layers
    quant_utils.add_actquant(model)
    qlayers = quant_utils.find_qlayers(model)

    # Configure online Hadamard for R4 (down_proj)
    if args.use_r4 and not args.butterfly:
        for name in qlayers:
            if 'down_proj' in name or 'fc2' in name:
                had_K, K = hadamard_utils.get_hadK(umodel.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had

    # Configure online Hadamard for R2 (o_proj) if online mode
    if args.use_r2 == 'online':
        for name in qlayers:
            if 'o_proj' in name or 'out_proj' in name:
                had_K, K = hadamard_utils.get_hadK(umodel.num_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = umodel.head_dim
                qlayers[name].fp32_had = args.fp32_had

    # Configure per-layer activation quantization bits
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(
            model, layers=[quant_utils.ActQuantWrapper]
        )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not args.a_asym
            layer_a_clip = args.a_clip_ratio
            residual = args.a_residual

            if 'v_proj' in name and args.v_bits < 16:
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=args.v_groupsize,
                    sym=not args.v_asym,
                    clip_ratio=args.v_clip_ratio
                )

            if 'lm_head' in name:
                layer_input_bits = 16

            if 'down_proj' in name or 'fc2' in name:
                pass  # Use default groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
                residual=residual
            )

    # Configure K-cache quantization with R3
    if args.k_bits < 16 and umodel.has_rope:
        try:
            import rotation_utils as rot_utils
            import model_utils
        except ImportError:
            logging.warning("Cannot set up K-cache quantization (missing imports)")
            return

        k_quant_config = {
            'k_bits': args.k_bits,
            'k_groupsize': args.k_groupsize,
            'k_sym': not args.k_asym,
            'k_clip_ratio': args.k_clip_ratio,
            'use_r3': args.use_r3 and not args.butterfly,
        }

        layers = umodel.get_layers()
        for layer in layers:
            self_attn = umodel.get_self_attn(layer)
            rot_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                self_attn,
                umodel.rope_function_name,
                config=model.config,
                **k_quant_config
            )


def setup_butterfly_r3_online(model, butterfly_r3, umodel: UnifiedQuantModel):
    """Register butterfly R3 as online rotation for Q/K.

    When butterfly is used, we replace the Hadamard transform in
    QKRotationWrapper with the learned butterfly.
    """
    R3_matrix = butterfly_r3.get_matrix().float()

    # Register as a pre-forward hook on attention modules
    layers = umodel.get_layers()
    for layer in layers:
        self_attn = umodel.get_self_attn(layer)
        # Store the R3 matrix as a buffer
        if not hasattr(self_attn, '_butterfly_r3'):
            self_attn.register_buffer(
                '_butterfly_r3', R3_matrix.to(DEV)
            )

    logging.info("Butterfly R3 registered for online Q/K rotation.")


def setup_butterfly_r4_online(model, butterfly_r4, umodel: UnifiedQuantModel):
    """Register butterfly R4 for online down_proj input rotation.

    After butterfly R4 is baked into weights (offline), we still need
    an online component for the activation transformation.
    """
    try:
        import quant_utils
    except ImportError:
        return

    qlayers = quant_utils.find_qlayers(model)
    R4_module = butterfly_r4.to(DEV)
    R4_module.eval()

    for name in qlayers:
        if 'down_proj' in name or 'fc2' in name:
            # Store butterfly module reference
            qlayers[name].register_buffer(
                '_butterfly_r4_matrix',
                butterfly_r4.get_matrix().float().to(DEV)
            )
            qlayers[name].online_full_had = True  # Will be handled by wrapper
            qlayers[name].K = 1  # Signal butterfly mode

    logging.info("Butterfly R4 registered for online activation rotation.")


# ============================================================================
# Weight Quantization
# ============================================================================

def run_weight_quantization(model, args, umodel: UnifiedQuantModel, tokenizer):
    """Run weight quantization (GPTQ, RTN, or NF4).

    For INT4: uses existing DartQuant GPTQ/RTN pipeline.
    For NF4: uses bitsandbytes Linear4bit.
    """
    if args.quantizer_type == 'nf4':
        logging.info("Applying NF4 weight quantization via bitsandbytes...")
        apply_nf4_to_model(model, skip_lm_head=True)
        return

    # INT4 quantization
    if args.w_bits >= 16:
        logging.info("No weight quantization needed (w_bits >= 16)")
        return

    try:
        import gptq_utils
        import data_utils
    except ImportError:
        logging.error("Cannot import gptq_utils/data_utils for INT4 quantization")
        raise

    if args.w_rtn:
        logging.info(f"Applying RTN weight quantization (W{args.w_bits})...")
        gptq_utils.rtn_fwrd(model, DEV, args)
    else:
        logging.info(f"Applying GPTQ weight quantization (W{args.w_bits})...")
        trainloader = data_utils.get_loaders(
            args.cal_dataset, nsamples=args.nsamples,
            seed=args.seed, model=args.model,
            seqlen=model.seqlen, eval_mode=False
        )
        gptq_utils.gptq_fwrd(model, trainloader, DEV, args)


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, args, umodel: UnifiedQuantModel):
    """Evaluate model perplexity on specified datasets."""
    try:
        import eval_utils
        import data_utils
    except ImportError:
        logging.error("Cannot import eval_utils/data_utils for evaluation")
        return {}

    results = {}
    for dataset in args.ppl_eval_dataset:
        testenc = data_utils.get_loaders(
            dataset, seed=args.seed, model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )
        ppl = eval_utils.ppl_evaluator(model, testenc, DEV, args)
        results[dataset] = ppl
        logging.info(f"  {dataset.upper()} PPL: {ppl:.2f}")

    return results


def evaluate_model_lm_eval(model, args, umodel: UnifiedQuantModel):
    """Run zero-shot benchmarks via lm-evaluation-harness."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logging.error("lm_eval not installed. pip install lm-eval==0.4.3")
        return {}

    tokenizer = umodel.get_tokenizer()
    hflm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.lm_eval_batch_size,
    )

    task_names = args.lm_eval_tasks
    logging.info(f"  lm_eval tasks: {task_names}")

    raw_results = lm_eval.simple_evaluate(hflm, tasks=task_names)['results']

    # Extract accuracy (prefer acc_norm over acc)
    metrics = {}
    for task, result in raw_results.items():
        acc = result.get('acc_norm,none', result.get('acc,none', None))
        if acc is not None:
            metrics[task] = round(acc * 100, 2)
    if metrics:
        metrics['acc_avg'] = round(
            sum(metrics.values()) / len(metrics), 2
        )

    for task, acc in metrics.items():
        logging.info(f"  {task}: {acc:.2f}%")

    return metrics


# ============================================================================
# Main Pipeline
# ============================================================================

def run_full_pipeline(args):
    """Execute the complete DartQuant v2 quantization pipeline.

    Args:
        args: Parsed arguments from args.py:create_parser()
    """
    transformers.set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.info("=" * 70)
    logging.info("DartQuant v2: Unified Quantization Pipeline")
    logging.info("=" * 70)
    logging.info(f"  Model:          {args.model}")
    logging.info(f"  Loss:           {args.loss}")
    logging.info(f"  Quantizer:      {args.quantizer_type}")
    logging.info(f"  Butterfly:      {args.butterfly}")
    logging.info(f"  W{args.w_bits}A{args.a_bits}KV{args.k_bits}")
    logging.info("=" * 70)

    _pipeline_start = time.time()

    # ======== Step 1: Load Model ========
    logging.info("[1/12] Loading model...")
    _t = time.time()
    umodel = UnifiedQuantModel(
        args.model, args.hf_token, args.cache_dir
    )
    model = umodel.model
    tokenizer = umodel.get_tokenizer()
    model.model_name = args.model.split('/')[-1]

    logging.info(f"  Architecture: {model.config.__class__.__name__}")
    logging.info(f"  Hidden: {umodel.hidden_size}, Layers: {umodel.num_layers}, "
                 f"Heads: {umodel.num_heads}/{umodel.kv_heads}")
    logging.info(f"  Step 1 done in {time.time() - _t:.1f}s")

    # ======== Step 2: Fuse LayerNorms ========
    logging.info("[2/12] Fusing LayerNorms...")
    _t = time.time()
    fuse_layer_norms(umodel)
    logging.info(f"  Step 2 done in {time.time() - _t:.1f}s")

    # ======== Step 3: Load Calibration Data ========
    logging.info("[3/12] Loading calibration data...")
    _t = time.time()
    try:
        import data_utils
        trainloader = data_utils.get_loaders(
            args.cal_dataset, nsamples=args.nsamples,
            seed=args.seed, model=args.model,
            seqlen=model.seqlen, eval_mode=False
        )
        # Convert to tensor for activation collection
        calib_data = torch.cat(
            [batch[0] for batch in trainloader], dim=0
        )
    except ImportError:
        # Fallback: generate random calibration data
        logging.warning("data_utils not available, using random calibration data")
        calib_data = torch.randint(
            0, model.config.vocab_size, (args.nsamples, args.seqlen)
        )
    logging.info(f"  Calibration data: {calib_data.shape}")
    logging.info(f"  Step 3 done in {time.time() - _t:.1f}s")

    # ======== Step 4: Train R1 (per-layer) ========
    R1_dict = {}
    if args.use_r1:
        if args.r1_path:
            logging.info(f"[4/12] Loading R1 from {args.r1_path}...")
            r1_data = torch.load(args.r1_path, map_location='cpu')
            # Support both new per-layer dict {int: Tensor} and legacy single Tensor
            if isinstance(r1_data, dict) and all(isinstance(k, int) for k in r1_data):
                R1_dict = r1_data
            elif isinstance(r1_data, dict) and 'R1' in r1_data:
                # Legacy format: broadcast single matrix to all layers
                R1_single = r1_data['R1']
                R1_dict = {i: R1_single for i in range(umodel.num_layers)}
            else:
                # Legacy format: plain tensor — broadcast to all layers
                R1_dict = {i: r1_data for i in range(umodel.num_layers)}
        else:
            logging.info("[4/12] Collecting activations and training R1 (per-layer)...")
            _t4 = time.time()
            model.to(DEV)

            from .trainers import train_r1_single_layer
            layers_prefix = umodel.arch.layers_path

            # Build ALL target names across all layers so we can collect
            # every layer's activations in a SINGLE forward pass (N_samples
            # passes instead of N_layers × N_samples — ~22× faster).
            all_layer_targets = {}   # layer_idx -> [target_name, ...]
            all_target_names = []
            for layer_idx in range(umodel.num_layers):
                targets = []
                if umodel.arch.mlp_up_proj_attr:
                    targets.append(
                        f"{layers_prefix}.{layer_idx}.{umodel.arch.mlp_up_proj_attr}"
                    )
                targets.append(
                    f"{layers_prefix}.{layer_idx}.{umodel.arch.q_proj_attr}"
                )
                all_layer_targets[layer_idx] = targets
                all_target_names.extend(targets)

            logging.info(f"  Collecting R1 activations for all {umodel.num_layers} "
                         f"layers in one pass ({calib_data.shape[0]} samples)...")
            _t_collect = time.time()
            all_acts = collect_activations(
                model, calib_data, all_target_names, DEV
            )
            logging.info(f"  R1 activation collection done in "
                         f"{time.time() - _t_collect:.1f}s")

            # Train R1 per-layer (greedy), freeing activations after each
            for layer_idx in range(umodel.num_layers):
                layer_act_tensors = [
                    all_acts[t] for t in all_layer_targets[layer_idx]
                    if t in all_acts
                ]
                if not layer_act_tensors:
                    logging.warning(
                        f"  No activations for layer {layer_idx}, skipping R1"
                    )
                    continue

                combined = torch.cat(
                    [a.reshape(-1, umodel.hidden_size)
                     for a in layer_act_tensors],
                    dim=0
                )

                R1_dict[layer_idx] = train_r1_single_layer(
                    acts=combined,
                    hidden_size=umodel.hidden_size,
                    loss_fn_name=args.loss,
                    lr=args.lr,
                    momentum=args.momentum,
                    epochs=args.r1_epochs,
                    batch_size=args.batch_size,
                    cos_lr=args.cos_lr,
                    optim=args.optim,
                    init_mode=args.rotate_mode,
                    accumulation_steps=args.accumulation_steps,
                    train_subset_size=args.train_subset_size,
                    device=DEV,
                    layer_idx=layer_idx,
                )

                # Free this layer's activations to limit peak RAM
                for t in all_layer_targets[layer_idx]:
                    if t in all_acts:
                        del all_acts[t]
                del combined, layer_act_tensors
                cleanup_memory()

            del all_acts
            logging.info(f"  R1 training complete for {len(R1_dict)} layers "
                         f"in {time.time() - _t4:.1f}s")
            model.cpu()
            cleanup_memory()
    else:
        logging.info("[4/12] R1 disabled, skipping...")

    # ======== Step 5: Apply R1 (per-layer) ========
    if R1_dict:
        logging.info("[5/12] Applying per-layer R1 rotations to model...")
        _t = time.time()
        smooth_scale = None
        if args.smooth:
            smooth_scale = torch.load(args.smooth)
        apply_r1_rotation_per_layer(model, R1_dict, umodel, smooth_scale)
        logging.info(f"  Step 5 done in {time.time() - _t:.1f}s")
    else:
        logging.info("[5/12] No R1, skipping rotation...")

    # ======== Step 6: Train R2 ========
    R2_dict = {}
    if args.use_r2 != 'none':
        if args.r2_path:
            logging.info(f"[6/12] Loading R2 from {args.r2_path}...")
            R2_dict = torch.load(args.r2_path, map_location='cpu')
        else:
            logging.info("[6/12] Collecting activations and training R2...")
            _t6 = time.time()
            model.to(DEV)

            from .trainers import train_r2_single_layer
            layers_prefix = umodel.arch.layers_path

            # Build ALL R2 targets (o_proj for each layer) and collect
            # in a single forward pass — same optimisation as R1.
            all_r2_targets = {}
            all_r2_target_names = []
            for layer_idx in range(umodel.num_layers):
                target = (f"{layers_prefix}.{layer_idx}."
                          f"{umodel.arch.o_proj_attr}")
                all_r2_targets[layer_idx] = target
                all_r2_target_names.append(target)

            logging.info(f"  Collecting R2 activations for all "
                         f"{umodel.num_layers} layers in one pass...")
            _t_collect = time.time()
            all_acts = collect_activations(
                model, calib_data, all_r2_target_names, DEV
            )
            logging.info(f"  R2 activation collection done in "
                         f"{time.time() - _t_collect:.1f}s")

            # Train R2 per-layer (greedy), freeing activations after each
            for layer_idx in range(umodel.num_layers):
                target = all_r2_targets[layer_idx]
                acts = all_acts.get(target)
                if acts is None:
                    logging.warning(
                        f"  No R2 activations for layer {layer_idx}"
                    )
                    continue

                key, rotation = train_r2_single_layer(
                    acts=acts,
                    hidden_size=umodel.hidden_size,
                    num_heads=umodel.num_heads,
                    kv_heads=umodel.kv_heads,
                    loss_fn_name=args.loss,
                    lr=args.lr,
                    momentum=args.momentum,
                    epochs=args.r2_epochs,
                    batch_size=args.batch_size * 2,
                    cos_lr=args.cos_lr,
                    optim=args.optim,
                    accumulation_steps=max(args.accumulation_steps, 2),
                    device=DEV,
                    layer_idx=layer_idx,
                    layers_path=layers_prefix,
                )
                R2_dict[key] = rotation

                # Free this layer's activations
                if target in all_acts:
                    del all_acts[target]
                del acts
                cleanup_memory()

            del all_acts
            logging.info(f"  R2 training complete for {len(R2_dict)} layers "
                         f"in {time.time() - _t6:.1f}s")
            model.cpu()
            cleanup_memory()
    else:
        logging.info("[6/12] R2 disabled, skipping...")

    # ======== Step 7: Apply R2 ========
    if R2_dict:
        logging.info("[7/12] Applying R2 to model...")
        _t = time.time()
        apply_r2_rotation(model, R2_dict, umodel)
        logging.info(f"  Step 7 done in {time.time() - _t:.1f}s")
    else:
        logging.info("[7/12] No R2, skipping...")

    # ======== Step 8: Handle R3/R4 ========
    butterfly_r3 = None
    butterfly_r4 = None

    if args.butterfly:
        logging.info("[8/12] Training Butterfly R3/R4...")
        _t8 = time.time()
        model.to(DEV)

        # Auto-select KL divergence loss for R3/R4 based on quantizer type.
        # INT4 → kl_unif (maximise entropy → uniform coverage of quantiser bins)
        # NF4  → kl_gauss (minimise skewness² + kurtosis² → Gaussian shape)
        _bf_loss = 'kl_unif' if args.quantizer_type == 'int4' else 'kl_gauss'
        logging.info(f"  Butterfly R3/R4 distribution loss: {_bf_loss} "
                     f"(auto-selected for {args.quantizer_type})")

        # Train Butterfly R3 (for Q/K after RoPE)
        # R3 is applied online (not baked into weights), so we use
        # activation-only reconstruction loss (weight_matrices=None).
        if args.use_r3:
            logging.info("  Training Butterfly R3...")
            r3_acts = collect_r3_activations(model, calib_data, umodel, DEV)
            if r3_acts is not None:
                butterfly_r3 = train_butterfly(
                    activations=r3_acts,
                    dim=umodel.head_dim,
                    loss_fn_name=_bf_loss,
                    label="R3",
                    lr=args.lr,
                    momentum=args.momentum,
                    epochs=args.butterfly_epochs,
                    batch_size=args.batch_size,
                    cos_lr=True,
                    optim=args.optim,
                    quantizer_type=args.quantizer_type,
                    lambda_recon=args.lambda_recon,
                    quant_block_size=args.quant_block_size,
                    k_factor_mode=args.k_factor_mode,
                )
                del r3_acts

        # Train Butterfly R4 (for down_proj input)
        # R4 is baked into down_proj weights offline, so we use the
        # paper's Eq 17 weight-aware reconstruction loss.
        if args.use_r4:
            logging.info("  Training Butterfly R4...")
            r4_acts = collect_r4_activations(model, calib_data, umodel, DEV)
            if r4_acts is not None:
                # Collect weight bank for joint weight+activation recon loss
                r4_weight_bank = collect_r4_weight_bank(umodel, max_layers=8)

                butterfly_r4 = train_butterfly(
                    activations=r4_acts,
                    dim=umodel.intermediate_size,
                    loss_fn_name=_bf_loss,
                    label="R4",
                    lr=args.lr,
                    momentum=args.momentum,
                    epochs=args.butterfly_epochs,
                    batch_size=args.batch_size,
                    cos_lr=True,
                    optim=args.optim,
                    quantizer_type=args.quantizer_type,
                    lambda_recon=args.lambda_recon,
                    quant_block_size=args.quant_block_size,
                    weight_matrices=r4_weight_bank,
                    weight_quantizer_type=args.quantizer_type,
                    k_factor_mode=args.k_factor_mode,
                )
                del r4_acts, r4_weight_bank

        model.cpu()
        cleanup_memory()

        # Apply Butterfly R4 offline (bake into weights)
        if butterfly_r4 is not None:
            apply_r4_butterfly(model, butterfly_r4, umodel)
        logging.info(f"  Step 8 done in {time.time() - _t8:.1f}s")
    else:
        logging.info("[8/12] Using Hadamard for R3/R4...")
        _t8 = time.time()
        # Apply R4 Hadamard offline (bake into down_proj weights)
        if args.use_r4:
            apply_r4_hadamard(model, umodel)
        logging.info(f"  Step 8 done in {time.time() - _t8:.1f}s")

    # ======== Step 9: Setup Quantization Wrappers ========
    logging.info("[9/12] Setting up quantization wrappers...")
    _t = time.time()
    if args.quantizer_type == 'int4':
        setup_quantization_wrappers(model, args, umodel)

        # Setup butterfly online components
        if args.butterfly:
            if butterfly_r3 is not None:
                setup_butterfly_r3_online(model, butterfly_r3, umodel)
            if butterfly_r4 is not None:
                setup_butterfly_r4_online(model, butterfly_r4, umodel)
    else:
        logging.info("  NF4 mode: skipping activation quantization wrappers")
    logging.info(f"  Step 9 done in {time.time() - _t:.1f}s")

    # ======== Step 10: Weight Quantization ========
    logging.info(f"[10/12] Running {args.quantizer_type.upper()} weight quantization...")
    _t = time.time()
    if args.quantizer_type == 'int4':
        model.to(DEV)
    run_weight_quantization(model, args, umodel, tokenizer)
    logging.info(f"  Step 10 done in {time.time() - _t:.1f}s")

    # ======== Step 11: Move to device ========
    logging.info("[11/12] Moving model to device...")
    _t = time.time()
    if args.distribute:
        try:
            from utils import distribute_model
            distribute_model(model)
        except ImportError:
            model.to(DEV)
    else:
        model.to(DEV)
    logging.info(f"  Step 11 done in {time.time() - _t:.1f}s")

    # ======== Step 12: Evaluate ========
    ppl_results = {}
    lm_eval_results = {}
    _t = time.time()

    if args.ppl_eval:
        logging.info("[12/12] Evaluating perplexity...")
        ppl_results = evaluate_model(model, args, umodel)

    if args.lm_eval:
        logging.info("[12/12] Running lm_eval zero-shot benchmarks...")
        lm_eval_results = evaluate_model_lm_eval(model, args, umodel)

    if not args.ppl_eval and not args.lm_eval:
        logging.info("[12/12] Evaluation disabled.")

    logging.info(f"  Step 12 done in {time.time() - _t:.1f}s")
    logging.info(f"  Total pipeline time: {time.time() - _pipeline_start:.1f}s")

    results = ppl_results  # backward compat for return value

    # ======== Save Results ========
    logging.info("\n" + "=" * 70)
    logging.info("RESULTS")
    logging.info("=" * 70)
    logging.info(f"  Model:     {args.model}")
    logging.info(f"  Loss:      {args.loss}")
    logging.info(f"  Quantizer: {args.quantizer_type}")
    logging.info(f"  Butterfly: {args.butterfly}")
    logging.info(f"  Config:    W{args.w_bits}A{args.a_bits}KV{args.k_bits}")
    for dataset, ppl in ppl_results.items():
        logging.info(f"  {dataset.upper()} PPL: {ppl:.2f}")
    for task, acc in lm_eval_results.items():
        logging.info(f"  {task}: {acc:.2f}%")
    logging.info("=" * 70)

    # Save rotations
    if args.save_rotations:
        save_data = {
            'R1': R1_dict,
            'R2': R2_dict,
            'config': {
                'model': args.model,
                'loss': args.loss,
                'quantizer_type': args.quantizer_type,
                'butterfly': args.butterfly,
            },
        }
        if butterfly_r3 is not None:
            save_data['butterfly_r3_state'] = butterfly_r3.state_dict()
        if butterfly_r4 is not None:
            save_data['butterfly_r4_state'] = butterfly_r4.state_dict()

        save_path = os.path.join(args.output_dir, 'rotations.pt')
        torch.save(save_data, save_path)
        logging.info(f"Rotations saved to {save_path}")

    # Save results
    results_path = os.path.join(args.output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write(f"model: {args.model}\n")
        f.write(f"loss: {args.loss}\n")
        f.write(f"quantizer_type: {args.quantizer_type}\n")
        f.write(f"butterfly: {args.butterfly}\n")
        f.write(f"w_bits: {args.w_bits}\n")
        f.write(f"a_bits: {args.a_bits}\n")
        f.write(f"k_bits: {args.k_bits}\n")
        f.write(f"v_bits: {args.v_bits}\n")
        for dataset, ppl in ppl_results.items():
            f.write(f"{dataset}_ppl: {ppl:.4f}\n")
        for task, acc in lm_eval_results.items():
            f.write(f"lm_eval_{task}: {acc:.4f}\n")
    logging.info(f"Results saved to {results_path}")

    logging.info("\nDone!")
    return results
