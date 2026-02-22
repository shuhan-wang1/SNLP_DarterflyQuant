"""
NF4 (Normal Float 4-bit) Quantizer using bitsandbytes.

NF4 is a non-uniform quantization scheme that assumes Gaussian-distributed inputs.
It computes optimal 4-bit quantization levels based on the normal distribution,
making it the natural counterpart to the swd_gauss loss function.

Requires: pip install bitsandbytes>=0.41.0
"""

import logging

import torch
import torch.nn as nn


def apply_nf4_to_model(model: nn.Module, skip_lm_head: bool = True):
    """Replace nn.Linear layers with bitsandbytes NF4-quantized Linear4bit layers.

    NF4 is weight-only quantization (following QLoRA convention).
    Activations are NOT quantized when using NF4.

    Args:
        model: The model to quantize (should already have rotations applied)
        skip_lm_head: Whether to skip quantizing the lm_head layer
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for NF4 quantization. "
            "Install it with: pip install bitsandbytes>=0.41.0"
        )

    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if skip_lm_head and 'lm_head' in name:
            logging.info(f"  NF4: Skipping {name}")
            continue

        # Get parent module
        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent = model.get_submodule(parts[0])
            child_name = parts[1]
        else:
            parent = model
            child_name = name

        in_features = module.in_features
        out_features = module.out_features
        has_bias = module.bias is not None

        # Create NF4 Linear layer
        nf4_layer = bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=has_bias,
            compute_dtype=torch.float16,
            compress_statistics=True,
            quant_type='nf4',
        )

        # Copy weights - bitsandbytes handles quantization on .cuda()
        nf4_layer.weight = bnb.nn.Params4bit(
            module.weight.data,
            requires_grad=False,
            quant_type='nf4',
        )
        if has_bias:
            nf4_layer.bias = nn.Parameter(module.bias.data.clone())

        setattr(parent, child_name, nf4_layer)
        count += 1

    logging.info(f"  NF4: Quantized {count} linear layers")
    return model


class NF4FakeQuantizer(nn.Module):
    """Fake NF4 quantizer for simulation / analysis purposes.

    Implements the NF4 quantization codebook from QLoRA:
    16 quantization levels optimized for N(0, 1) distributed inputs.

    This can be used for fake quantization (quantize-dequantize round trip)
    without requiring bitsandbytes.
    """

    # NF4 quantization levels (from QLoRA paper, optimal for N(0,1))
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ])

    def __init__(self, block_size: int = 64):
        super().__init__()
        self.block_size = block_size
        self.register_buffer('levels', self.NF4_LEVELS.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake quantize: quantize then dequantize."""
        orig_shape = x.shape
        orig_dtype = x.dtype
        x = x.float()

        # Reshape into blocks
        if x.numel() % self.block_size != 0:
            # Pad to block_size multiple
            pad_size = self.block_size - (x.numel() % self.block_size)
            x_flat = torch.cat([x.reshape(-1), torch.zeros(pad_size, device=x.device)])
        else:
            x_flat = x.reshape(-1)
            pad_size = 0

        x_blocks = x_flat.reshape(-1, self.block_size)

        # Per-block absmax normalization
        absmax = x_blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        x_norm = x_blocks / absmax  # Normalize to [-1, 1]

        # Quantize: find nearest NF4 level
        levels = self.levels.to(x.device)
        # (blocks, block_size, 1) - (1, 1, 16)
        distances = (x_norm.unsqueeze(-1) - levels.reshape(1, 1, -1)).abs()
        indices = distances.argmin(dim=-1)  # (blocks, block_size)

        # Dequantize
        x_deq = levels[indices] * absmax  # Scale back

        x_deq = x_deq.reshape(-1)
        if pad_size > 0:
            x_deq = x_deq[:x.numel() - pad_size] if pad_size > 0 else x_deq

        return x_deq.reshape(orig_shape).to(orig_dtype)
