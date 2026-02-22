"""
INT4 symmetric uniform quantizer (fake-quantization for training).

INT4 assumes inputs are approximately uniformly distributed over [-absmax, +absmax].
It is the natural counterpart to the swd_unif loss function, which aligns
activations toward a Uniform[-b, b] target distribution before quantization.

This module is intentionally separate from nf4_quantizer.py because the two
quantization schemes rest on fundamentally different distribution assumptions:
  - INT4  → uniform distribution  (pairs with swd_unif)
  - NF4   → Gaussian distribution (pairs with swd_gauss)
"""

import torch
import torch.nn as nn


class INT4FakeQuantizer(nn.Module):
    """Fake INT4 symmetric uniform quantizer for simulation purposes.

    Applies per-block absmax symmetric INT4 quantization:
      scale = absmax / q_max          (q_max = 7 for 4-bit)
      q     = clamp(round(x / scale), -q_max, q_max)
      x_hat = q * scale

    Pairs naturally with the swd_unif loss function.
    """

    def __init__(self, block_size: int = 64, bits: int = 4):
        super().__init__()
        self.block_size = block_size
        self.q_max = 2 ** (bits - 1) - 1  # 7 for INT4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake quantize: quantize then dequantize."""
        orig_shape = x.shape
        orig_dtype = x.dtype
        flat = x.float().reshape(-1)

        # Pad to a multiple of block_size
        pad_size = (-flat.numel()) % self.block_size
        if pad_size:
            flat = torch.cat([flat, torch.zeros(pad_size, device=x.device)])

        blocks = flat.reshape(-1, self.block_size)

        # Per-block absmax symmetric scaling
        absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale = absmax / self.q_max

        # Quantize and dequantize
        q = torch.clamp(torch.round(blocks / scale), -self.q_max, self.q_max)
        x_deq = q * scale

        x_deq = x_deq.reshape(-1)
        if pad_size:
            x_deq = x_deq[: flat.numel() - pad_size]

        return x_deq.reshape(orig_shape).to(orig_dtype)
