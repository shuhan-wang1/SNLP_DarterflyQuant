"""
Model utilities for JointQuant
Adapted from DartQuant/fake_quant/model_utils.py
"""

import torch
import torch.nn as nn
import transformers
import logging
import os
from typing import List, Optional, Tuple

# Model type constants
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer


def model_type_extractor(model):
    """Extract model type from model instance"""
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    else:
        raise ValueError(f'Unknown model type {type(model)}. Only LLaMA models are supported.')


def get_model_type(model):
    """Alias for model_type_extractor"""
    return model_type_extractor(model)


def skip(*args, **kwargs):
    """Helper function to skip initialization for faster loading"""
    pass


def get_model(model_name: str, hf_token: Optional[str] = None, 
              cache_dir: Optional[str] = None, dtype=None):
    """Load model with optimizations for faster loading"""
    # Skip random initialization for faster loading
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    if dtype is None:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = transformers.LlamaForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=dtype,
        token=hf_token,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.seqlen = 2048
    logging.info(f'---> Loaded {model_name} with seq_len: {model.seqlen}')
    return model


def get_tokenizer(model_name: str, hf_token: Optional[str] = None, 
                  cache_dir: Optional[str] = None):
    """Load tokenizer"""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        cache_dir=cache_dir,
        use_fast=False,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_embeddings(model, model_type) -> List[nn.Module]:
    """Get embedding layers"""
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type) -> List[nn.Module]:
    """Get transformer layers"""
    if model_type == LLAMA_MODEL:
        return list(model.model.layers)
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_lm_head(model, model_type) -> nn.Module:
    """Get language model head"""
    if model_type == LLAMA_MODEL:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_pre_head_layernorm(model, model_type) -> nn.Module:
    """Get the layernorm before lm_head"""
    if model_type == LLAMA_MODEL:
        return model.model.norm
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_mlp_bottleneck_size(model) -> int:
    """Get MLP intermediate size"""
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    else:
        raise ValueError(f'Unknown model type {model_type}')


class RMSN(nn.Module):
    """
    Root Mean Square Normalization (RMSN) layer with fixed weight=1.
    Used to replace LlamaRMSNorm after fusion - effectively identity scaling.
    Matches official DartQuant implementation.
    """
    def __init__(self, mean_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.variance_epsilon = eps  # For compatibility
        # Dummy weight parameter (not used but needed for compatibility)
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


def replace_modules(
    root: nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool = False,
) -> None:
    """
    Replace modules of given type using the supplied module factory.
    Perform a depth-first search of a module hierarchy.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:
                new_module = new_module_factory(module, int(name))
            else:
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)
