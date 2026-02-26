"""Dense (non-MoE) model architecture registrations."""

from dartquant_v2.arch.dense import llama as _llama  # noqa: F401
from dartquant_v2.arch.dense import opt as _opt      # noqa: F401
from dartquant_v2.arch.dense import qwen3 as _qwen3   # noqa: F401

__all__ = []
