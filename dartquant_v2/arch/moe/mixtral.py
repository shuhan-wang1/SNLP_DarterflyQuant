"""
Mistral AI Mixtral family — MoE architecture registration.

Covers: Mixtral-8x7B-v0.1, Mixtral-8x22B-v0.1
Config class name: MixtralConfig

MoE structure (per decoder layer):
  self_attn.{q,k,v,o}_proj     -- standard attention projections
  block_sparse_moe.gate         -- router (Linear, d_model -> num_experts)
  block_sparse_moe.experts      -- ModuleList of MixtralBlockSparseTop2MLP
    experts[i].w1               -- up/gate projection (SiLU input)
    experts[i].w3               -- gate projection
    experts[i].w2               -- down projection
  input_layernorm               -- pre-attention RMSNorm
  post_attention_layernorm      -- pre-MoE RMSNorm

Key differences from dense Llama:
  - experts use w1/w2/w3 naming (not up/gate/down_proj)
  - No shared expert (all experts are routed, top-2 active)
  - intermediate size is stored as config.ffn_dim (not intermediate_size)

R1 / R4 rotation notes (from SNLP report section 2.4.4):
  R1: applied to all experts' w1, w3 (input side) and w2 (output side, R1^T)
  R4: all experts share a single R4 matrix to reduce overhead;
      the same R4 is baked into every expert's w2
"""

from dartquant_v2.unified_model import ModelArchConfig, register_arch

register_arch("MixtralConfig", ModelArchConfig(
    # Model-level paths
    embed_tokens_path="model.embed_tokens",
    embed_positions_path=None,
    layers_path="model.layers",
    pre_head_norm_path="model.norm",
    lm_head_path="lm_head",

    # Attention projections (same layout as Llama)
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.o_proj",
    self_attn_attr="self_attn",

    # Dense MLP paths: None (Mixtral has no dense FFN, only MoE experts)
    mlp_up_proj_attr=None,
    mlp_gate_proj_attr=None,
    mlp_down_proj_attr=None,

    # Layer norms
    input_ln_attr="input_layernorm",
    post_attn_ln_attr="post_attention_layernorm",
    norm_class_name="MistralRMSNorm",

    # RoPE
    has_rope=True,
    rope_function_name="apply_rotary_pos_emb",

    # Other
    needs_mean_baking=False,
    no_split_module_class="MixtralDecoderLayer",

    # MoE configuration
    is_moe=True,
    experts_attr="block_sparse_moe.experts",
    expert_up_proj_attr="w1",       # SiLU activation input
    expert_gate_proj_attr="w3",     # gate
    expert_down_proj_attr="w2",     # output projection
    shared_expert_attr=None,        # Mixtral has no shared expert
    moe_intermediate_size_attr="ffn_dim",
))
