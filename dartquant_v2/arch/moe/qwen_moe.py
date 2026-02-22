"""
Alibaba Qwen2-MoE family — MoE architecture registration.

Covers: Qwen1.5-MoE-A2.7B, Qwen2-57B-A14B-Instruct, and similar variants
Config class name: Qwen2MoeConfig

MoE structure (per decoder layer):
  self_attn.{q,k,v,o}_proj    -- standard attention projections
  mlp.gate                     -- router (Linear, d_model -> num_experts)
  mlp.experts                  -- ModuleList of Qwen2MoeMLP
    experts[i].up_proj
    experts[i].gate_proj
    experts[i].down_proj
  mlp.shared_expert            -- a single always-active Qwen2MoeMLP
    shared_expert.up_proj
    shared_expert.gate_proj
    shared_expert.down_proj
  mlp.shared_expert_gate       -- gate scalar for shared expert
  input_layernorm              -- pre-attention RMSNorm
  post_attention_layernorm     -- pre-MoE RMSNorm

Key differences from Mixtral:
  - Expert attribute names match Llama convention (up/gate/down_proj)
  - Has an always-active shared expert alongside the routed experts
  - intermediate size for routed experts: config.moe_intermediate_size
    (shared expert may use config.shared_expert_intermediate_size — same value typically)

R1 / R4 rotation notes (from SNLP report section 2.4.4):
  R1: applied to all routed experts' up/gate_proj and down_proj (R1^T),
      plus the shared expert's up/gate_proj and down_proj
  R4: single shared R4 baked into every routed expert's down_proj
      and the shared expert's down_proj
"""

from dartquant_v2.unified_model import ModelArchConfig, register_arch

register_arch("Qwen2MoeConfig", ModelArchConfig(
    # Model-level paths
    embed_tokens_path="model.embed_tokens",
    embed_positions_path=None,
    layers_path="model.layers",
    pre_head_norm_path="model.norm",
    lm_head_path="lm_head",

    # Attention projections
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.o_proj",
    self_attn_attr="self_attn",

    # Dense MLP paths: None (all FFN is inside MoE block)
    mlp_up_proj_attr=None,
    mlp_gate_proj_attr=None,
    mlp_down_proj_attr=None,

    # Layer norms
    input_ln_attr="input_layernorm",
    post_attn_ln_attr="post_attention_layernorm",
    norm_class_name="Qwen2RMSNorm",

    # RoPE
    has_rope=True,
    rope_function_name="apply_rotary_pos_emb",

    # Other
    needs_mean_baking=False,
    no_split_module_class="Qwen2MoeDecoderLayer",

    # MoE configuration
    is_moe=True,
    experts_attr="mlp.experts",
    expert_up_proj_attr="up_proj",
    expert_gate_proj_attr="gate_proj",
    expert_down_proj_attr="down_proj",
    # Shared expert (always active, separate from routed experts)
    shared_expert_attr="mlp.shared_expert",
    shared_expert_up_attr=None,    # reuses expert_up_proj_attr ("up_proj")
    shared_expert_gate_attr=None,  # reuses expert_gate_proj_attr ("gate_proj")
    shared_expert_down_attr=None,  # reuses expert_down_proj_attr ("down_proj")
    moe_intermediate_size_attr="moe_intermediate_size",
))
