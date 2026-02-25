"""MoE (Mixture-of-Experts) model architecture registrations."""

from dartquant_v2.arch.moe import mixtral as _mixtral    # noqa: F401
from dartquant_v2.arch.moe import qwen_moe as _qwen_moe  # noqa: F401
from dartquant_v2.arch.moe import deepseek_moe as _deepseek_moe  # noqa: F401

__all__ = []
