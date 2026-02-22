"""
DartQuant v2 Architecture Registry
====================================
Importing this package registers all built-in model architectures.

Structure
---------
arch/
  dense/     -- Dense (non-MoE) models
    llama.py     Meta Llama family (all versions share LlamaConfig)
    opt.py       Meta OPT family
  moe/       -- Mixture-of-Experts models
    mixtral.py   Mistral AI Mixtral 8x7B / 8x22B
    qwen_moe.py  Alibaba Qwen2-MoE

Adding a new architecture
--------------------------
1. Create a new file in dense/ or moe/ (whichever applies).
2. Call register_arch("YourModelConfig", ModelArchConfig(...)) at module level.
3. Import the new file in the appropriate __init__.py (dense/ or moe/).
That's all — no changes to any core pipeline code are needed.
"""

# Trigger registration of all built-in architectures by importing sub-packages.
from dartquant_v2.arch import dense as _dense  # noqa: F401
from dartquant_v2.arch import moe as _moe      # noqa: F401

__all__ = []  # This module exists only for side-effects (registration).
