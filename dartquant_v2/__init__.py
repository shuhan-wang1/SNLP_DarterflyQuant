"""
DartQuant v2: Unified Quantization Pipeline

Provides a one-click quantization pipeline with pluggable loss functions,
quantizer types (INT4/NF4), and optional Butterfly Givens rotations for R3/R4.

Importing this package automatically registers all built-in model architectures
(Llama, OPT, Mixtral, Qwen2-MoE, ...) via the arch/ sub-package.
"""

# Register all built-in architectures (dense + MoE) on package import.
from dartquant_v2 import arch as _arch  # noqa: F401
