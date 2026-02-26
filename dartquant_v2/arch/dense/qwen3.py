"""
DartQuant architecture registration for Qwen3 models.
"""

from dartquant_v2.unified_model import ModelArchConfig, register_arch

# Define the architecture mapping for Qwen3
qwen3_arch_config = ModelArchConfig(
    # Embedding paths (relative to model)
    embed_tokens_path="model.embed_tokens",
    embed_positions_path=None,  # Qwen3 uses RoPE, not learned position embeddings

    # Transformer layers path (relative to model)
    layers_path="model.layers",

    # Pre-head norm and lm_head (relative to model)
    pre_head_norm_path="model.norm",
    lm_head_path="lm_head",

    # Attention projection attributes (relative to a single layer)
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.o_proj",

    # MLP attributes (relative to a single layer)
    mlp_up_proj_attr="mlp.up_proj",
    mlp_gate_proj_attr="mlp.gate_proj",
    mlp_down_proj_attr="mlp.down_proj",

    # LayerNorm attributes (relative to a single layer)
    input_ln_attr="input_layernorm",
    post_attn_ln_attr="post_attention_layernorm",

    # Norm class used in the model (used for replacement after fusion)
    norm_class_name="Qwen3RMSNorm", 

    # RoPE settings
    has_rope=True,
    rope_function_name="apply_rotary_pos_emb",

    # Mean baking is specific to OPT and older architectures
    needs_mean_baking=False,

    # Self-attention module attribute (relative to layer)
    self_attn_attr="self_attn",

    # No-split module class name (for multi-GPU FSDP/DeepSpeed distribution)
    no_split_module_class="Qwen3DecoderLayer",

    # MoE settings (Qwen3 4B is a dense model)
    is_moe=False,
    experts_attr=None,
    expert_up_proj_attr=None,
    expert_gate_proj_attr=None,
    expert_down_proj_attr=None,
    shared_expert_attr=None,
    shared_expert_up_attr=None,
    shared_expert_gate_attr=None,
    shared_expert_down_attr=None,
    moe_intermediate_size_attr=None
)

# The config class name defined in HuggingFace for Qwen3
# e.g., AutoConfig.from_pretrained("Qwen/Qwen3-4B").__class__.__name__
register_arch("Qwen3Config", qwen3_arch_config)