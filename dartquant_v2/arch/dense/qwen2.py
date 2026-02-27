"""
Qwen2.5 Dense family — architecture registration.

Covers: Qwen2.5-0.5B through Qwen2.5-72B (Base and Instruct).
All share the same HuggingFace config class name: Qwen2Config.

Architecture is nearly identical to Llama (RoPE + GQA + SwiGLU),
with the main difference being the norm class name (Qwen2RMSNorm).

Layer structure (per decoder layer):
  self_attn.{q,k,v}_proj   -- attention projections (bias=True)
  self_attn.o_proj          -- output projection
  mlp.up_proj               -- FFN input (SiLU gate)
  mlp.gate_proj             -- FFN gate
  mlp.down_proj             -- FFN output
  input_layernorm           -- pre-attention RMSNorm
  post_attention_layernorm  -- pre-MLP RMSNorm
"""

from dartquant_v2.unified_model import ModelArchConfig, register_arch

register_arch("Qwen2Config", ModelArchConfig(
    # Model-level paths
    embed_tokens_path="model.embed_tokens",
    embed_positions_path=None,
    layers_path="model.layers",
    pre_head_norm_path="model.norm",
    lm_head_path="lm_head",

    # Attention projections (relative to layer)
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.o_proj",
    self_attn_attr="self_attn",

    # MLP projections (relative to layer)
    mlp_up_proj_attr="mlp.up_proj",
    mlp_gate_proj_attr="mlp.gate_proj",
    mlp_down_proj_attr="mlp.down_proj",

    # Layer norms
    input_ln_attr="input_layernorm",
    post_attn_ln_attr="post_attention_layernorm",
    norm_class_name="Qwen2RMSNorm",

    # RoPE
    has_rope=True,
    rope_function_name="apply_rotary_pos_emb",

    # Other
    needs_mean_baking=False,
    no_split_module_class="Qwen2DecoderLayer",

    # MoE: not applicable
    is_moe=False,
))
