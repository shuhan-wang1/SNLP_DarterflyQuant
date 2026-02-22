"""
UnifiedQuantModel: Architecture-agnostic model wrapper for DartQuant quantization.

Provides uniform accessors for model components (embeddings, layers, projections, etc.)
regardless of the underlying architecture (Llama, OPT, etc.).

New model architectures can be registered via register_arch().
"""

import operator
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import transformers


@dataclass
class ModelArchConfig:
    """Architecture-specific configuration for accessing model components.

    Each field stores the attribute path relative to the model or layer object.
    Developers adding new model architectures should fill in these paths.
    """
    # Embedding paths (relative to model)
    embed_tokens_path: str
    embed_positions_path: Optional[str] = None  # OPT has position embeddings

    # Transformer layers path (relative to model)
    layers_path: str = "model.layers"

    # Pre-head norm and lm_head (relative to model)
    pre_head_norm_path: str = "model.norm"
    lm_head_path: str = "lm_head"

    # Attention projection attributes (relative to a single layer)
    q_proj_attr: str = "self_attn.q_proj"
    k_proj_attr: str = "self_attn.k_proj"
    v_proj_attr: str = "self_attn.v_proj"
    o_proj_attr: str = "self_attn.o_proj"

    # MLP attributes (relative to a single layer)
    mlp_up_proj_attr: Optional[str] = "mlp.up_proj"
    mlp_gate_proj_attr: Optional[str] = "mlp.gate_proj"
    mlp_down_proj_attr: str = "mlp.down_proj"

    # LayerNorm attributes (relative to a single layer)
    input_ln_attr: str = "input_layernorm"
    post_attn_ln_attr: str = "post_attention_layernorm"

    # Norm class used in the model (for replacement after fusion)
    norm_class_name: str = "LlamaRMSNorm"

    # Whether the model uses RoPE
    has_rope: bool = True
    rope_function_name: str = "apply_rotary_pos_emb"

    # Whether OPT-style mean baking is needed
    needs_mean_baking: bool = False

    # Self-attention module attribute (relative to layer)
    self_attn_attr: str = "self_attn"

    # No-split module class name (for multi-GPU distribution)
    no_split_module_class: str = "LlamaDecoderLayer"


# ============================================================================
# Architecture Registry
# ============================================================================

_ARCH_REGISTRY: dict[str, ModelArchConfig] = {}


def register_arch(config_class_name: str, arch_config: ModelArchConfig):
    """Register a new model architecture.

    Args:
        config_class_name: The __class__.__name__ of the HuggingFace model config
                           (e.g., "LlamaConfig", "OPTConfig")
        arch_config: ModelArchConfig with attribute paths for this architecture
    """
    _ARCH_REGISTRY[config_class_name] = arch_config


def get_arch_config(config_class_name: str) -> ModelArchConfig:
    """Look up the architecture config for a given HF config class name."""
    if config_class_name not in _ARCH_REGISTRY:
        raise ValueError(
            f"Unsupported model architecture: '{config_class_name}'. "
            f"Registered architectures: {list(_ARCH_REGISTRY.keys())}. "
            f"Use register_arch() to add support for new architectures."
        )
    return _ARCH_REGISTRY[config_class_name]


# ============================================================================
# Pre-registered architectures
# ============================================================================

# Llama family (Llama-2, Llama-3, Llama-3.1, Llama-3.2 all share LlamaConfig)
register_arch("LlamaConfig", ModelArchConfig(
    embed_tokens_path="model.embed_tokens",
    embed_positions_path=None,
    layers_path="model.layers",
    pre_head_norm_path="model.norm",
    lm_head_path="lm_head",
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.o_proj",
    mlp_up_proj_attr="mlp.up_proj",
    mlp_gate_proj_attr="mlp.gate_proj",
    mlp_down_proj_attr="mlp.down_proj",
    input_ln_attr="input_layernorm",
    post_attn_ln_attr="post_attention_layernorm",
    norm_class_name="LlamaRMSNorm",
    has_rope=True,
    rope_function_name="apply_rotary_pos_emb",
    needs_mean_baking=False,
    self_attn_attr="self_attn",
    no_split_module_class="LlamaDecoderLayer",
))

# OPT family
register_arch("OPTConfig", ModelArchConfig(
    embed_tokens_path="model.decoder.embed_tokens",
    embed_positions_path="model.decoder.embed_positions",
    layers_path="model.decoder.layers",
    pre_head_norm_path="model.decoder.final_layer_norm",
    lm_head_path="lm_head",
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.out_proj",
    mlp_up_proj_attr=None,  # OPT uses fc1
    mlp_gate_proj_attr=None,  # OPT has no gate
    mlp_down_proj_attr="fc2",
    input_ln_attr="self_attn_layer_norm",
    post_attn_ln_attr="final_layer_norm",
    norm_class_name="LayerNorm",
    has_rope=False,
    rope_function_name="",
    needs_mean_baking=True,
    self_attn_attr="self_attn",
    no_split_module_class="OPTDecoderLayer",
))


def _deep_getattr(obj, attr_path: str):
    """Get a nested attribute using dot-separated path."""
    return operator.attrgetter(attr_path)(obj)


def _deep_setattr(obj, attr_path: str, value):
    """Set a nested attribute using dot-separated path."""
    parts = attr_path.rsplit(".", 1)
    if len(parts) == 2:
        parent = _deep_getattr(obj, parts[0])
        setattr(parent, parts[1], value)
    else:
        setattr(obj, parts[0], value)


# ============================================================================
# UnifiedQuantModel
# ============================================================================

class UnifiedQuantModel:
    """Architecture-agnostic model wrapper for DartQuant quantization.

    Auto-detects the model architecture from HuggingFace config and provides
    uniform accessors for all model components needed by the quantization pipeline.

    Usage:
        umodel = UnifiedQuantModel("meta-llama/Llama-3.2-1B")
        layers = umodel.get_layers()
        for layer in layers:
            q, k, v, o = umodel.get_attn_projs(layer)
            ...
    """

    def __init__(self, model_name: str, hf_token: str = None,
                 cache_dir: str = None, dtype: str = 'auto'):
        self.model_name = model_name
        self.hf_token = hf_token

        # Load model
        self.model = self._load_model(model_name, hf_token, cache_dir, dtype)
        self.model.eval()

        # Detect architecture
        config_class_name = self.model.config.__class__.__name__
        self.arch = get_arch_config(config_class_name)
        logging.info(f"Detected architecture: {config_class_name}")

        # Set seqlen if not already set
        if not hasattr(self.model, 'seqlen'):
            if hasattr(self.model.config, 'max_position_embeddings'):
                self.model.seqlen = min(self.model.config.max_position_embeddings, 2048)
            else:
                self.model.seqlen = 2048

    def _load_model(self, model_name, hf_token, cache_dir, dtype):
        """Load model, trying transformers first then falling back to modelscope."""
        # Skip init for faster loading
        orig_kaiming = torch.nn.init.kaiming_uniform_
        orig_uniform = torch.nn.init.uniform_
        orig_normal = torch.nn.init.normal_
        torch.nn.init.kaiming_uniform_ = lambda *a, **k: None
        torch.nn.init.uniform_ = lambda *a, **k: None
        torch.nn.init.normal_ = lambda *a, **k: None

        try:
            kwargs = {
                'torch_dtype': dtype,
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }
            if hf_token:
                kwargs['token'] = hf_token
            if cache_dir:
                kwargs['cache_dir'] = cache_dir

            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_name, **kwargs
            )
            logging.info(f"Loaded model via transformers: {model_name}")
        except Exception as e:
            logging.warning(f"transformers load failed ({e}), trying modelscope...")
            try:
                from modelscope import AutoModelForCausalLM as MSAutoModel
                ms_kwargs = {
                    'torch_dtype': dtype,
                    'trust_remote_code': True,
                    'low_cpu_mem_usage': True,
                }
                if cache_dir:
                    ms_kwargs['cache_dir'] = cache_dir
                model = MSAutoModel.from_pretrained(model_name, **ms_kwargs)
                logging.info(f"Loaded model via modelscope: {model_name}")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load model '{model_name}' via both "
                    f"transformers ({e}) and modelscope ({e2})"
                )
        finally:
            torch.nn.init.kaiming_uniform_ = orig_kaiming
            torch.nn.init.uniform_ = orig_uniform
            torch.nn.init.normal_ = orig_normal

        return model

    # ------------------------------------------------------------------
    # Model-level accessors
    # ------------------------------------------------------------------

    def get_embeddings(self) -> list[nn.Module]:
        """Get all embedding modules."""
        embeds = [_deep_getattr(self.model, self.arch.embed_tokens_path)]
        if self.arch.embed_positions_path:
            embeds.append(_deep_getattr(self.model, self.arch.embed_positions_path))
        return embeds

    def get_layers(self) -> list[nn.Module]:
        """Get all transformer decoder layers."""
        layers_container = _deep_getattr(self.model, self.arch.layers_path)
        return list(layers_container)

    def get_lm_head(self) -> nn.Linear:
        """Get the language model head."""
        return _deep_getattr(self.model, self.arch.lm_head_path)

    def get_pre_head_norm(self) -> nn.Module:
        """Get the normalization layer before lm_head."""
        return _deep_getattr(self.model, self.arch.pre_head_norm_path)

    def get_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """Load the tokenizer for this model."""
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                use_fast=False,
                trust_remote_code=True,
            )
        except Exception:
            from modelscope import AutoTokenizer as MSAutoTokenizer
            tokenizer = MSAutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=False,
                trust_remote_code=True,
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    # ------------------------------------------------------------------
    # Layer-level accessors
    # ------------------------------------------------------------------

    def get_attn_projs(self, layer) -> tuple:
        """Get (q_proj, k_proj, v_proj, o_proj) from a layer."""
        q = _deep_getattr(layer, self.arch.q_proj_attr)
        k = _deep_getattr(layer, self.arch.k_proj_attr)
        v = _deep_getattr(layer, self.arch.v_proj_attr)
        o = _deep_getattr(layer, self.arch.o_proj_attr)
        return q, k, v, o

    def get_mlp_input_projs(self, layer) -> list[nn.Linear]:
        """Get MLP input projection layers.

        Returns [up_proj, gate_proj] for Llama or [fc1] for OPT.
        """
        projs = []
        if self.arch.mlp_up_proj_attr:
            projs.append(_deep_getattr(layer, self.arch.mlp_up_proj_attr))
        if self.arch.mlp_gate_proj_attr:
            projs.append(_deep_getattr(layer, self.arch.mlp_gate_proj_attr))
        # OPT-style: if no up/gate, look for fc1 directly on the layer
        if not projs:
            projs.append(layer.fc1)
        return projs

    def get_mlp_output_proj(self, layer) -> nn.Linear:
        """Get the MLP output projection (down_proj or fc2)."""
        return _deep_getattr(layer, self.arch.mlp_down_proj_attr)

    def get_input_ln(self, layer) -> nn.Module:
        """Get input layer norm (before attention)."""
        return _deep_getattr(layer, self.arch.input_ln_attr)

    def get_post_attn_ln(self, layer) -> nn.Module:
        """Get post-attention layer norm (before MLP)."""
        return _deep_getattr(layer, self.arch.post_attn_ln_attr)

    def get_self_attn(self, layer) -> nn.Module:
        """Get the self-attention module from a layer."""
        return _deep_getattr(layer, self.arch.self_attn_attr)

    # ------------------------------------------------------------------
    # Model config properties
    # ------------------------------------------------------------------

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def num_heads(self) -> int:
        return self.model.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_heads(self) -> int:
        if hasattr(self.model.config, 'num_key_value_heads'):
            return self.model.config.num_key_value_heads
        return self.num_heads  # MHA fallback

    @property
    def intermediate_size(self) -> int:
        if hasattr(self.model.config, 'intermediate_size'):
            return self.model.config.intermediate_size
        if hasattr(self.model.config, 'ffn_dim'):
            return self.model.config.ffn_dim
        raise AttributeError("Cannot determine intermediate_size from model config")

    @property
    def num_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def has_rope(self) -> bool:
        return self.arch.has_rope

    @property
    def rope_function_name(self) -> str:
        return self.arch.rope_function_name

    @property
    def needs_mean_baking(self) -> bool:
        return self.arch.needs_mean_baking
