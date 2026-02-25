"""
DeepSeek-MoE 16B family — MoE architecture registration.

Covers: deepseek-moe-16b-base, deepseek-moe-16b-chat/instruct
Config class name: DeepseekConfig

MoE structure (per decoder layer):
  self_attn.{q,k,v,o}_proj     -- standard attention projections (No MLA)
  mlp.gate                     -- router (Linear, d_model -> 64 experts)
  mlp.experts                  -- ModuleList of 64 routed experts
    experts[i].up_proj
    experts[i].gate_proj
    experts[i].down_proj
  mlp.shared_experts           -- always-active shared expert module (2 experts bundled)
    shared_experts.up_proj
    shared_experts.gate_proj
    shared_experts.down_proj
  input_layernorm              -- pre-attention RMSNorm
  post_attention_layernorm     -- pre-MoE RMSNorm

Key differences from Mixtral / Qwen2-MoE:
  - Uses highly fine-grained routing: 64 routed experts (activates 6 per token).
  - Features an always-active shared expert (pluralized attribute `shared_experts`).
  - Standard Multi-Head Attention is used, so global R3 rotation on Q/K heads works natively.

R1 / R4 rotation notes (from SNLP report section 2.4.4):
  R1: Absorbed into all 64 routed experts' up/gate_proj and down_proj (R1^T),
      plus the shared_experts' up/gate_proj and down_proj. Also absorbed by standard Q/K/V.
  R4: Single shared R4 Hadamard/Butterfly rotation baked into every routed expert's 
      down_proj and the shared_experts' down_proj after the SwiGLU activation.
"""

from dartquant_v2.unified_model import ModelArchConfig, register_arch

register_arch("DeepseekConfig", ModelArchConfig(
    embed_tokens_path="model.embed_tokens",
    layers_path="model.layers",
    pre_head_norm_path="model.norm",
    lm_head_path="lm_head",
    
    # Standard Attention Projections 
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.o_proj",
    self_attn_attr="self_attn",
    
    # Dense MLP path set to None
    mlp_up_proj_attr=None,
    mlp_gate_proj_attr=None,
    mlp_down_proj_attr=None,
    
    # Layer Norms
    input_ln_attr="input_layernorm",
    post_attn_ln_attr="post_attention_layernorm",
    norm_class_name="DeepseekRMSNorm",
    
    # RoPE Setup
    has_rope=True,
    rope_function_name="apply_rotary_pos_emb", 
    no_split_module_class="DeepseekDecoderLayer",

    # --- MoE Specific Fields ---
    is_moe=True,
    
    # Path to the 64 routed experts
    experts_attr="mlp.experts",
    expert_up_proj_attr="up_proj",
    expert_gate_proj_attr="gate_proj",
    expert_down_proj_attr="down_proj",
    
    shared_expert_attr="mlp.shared_experts", 
    shared_expert_up_attr=None,
    shared_expert_gate_attr=None,
    shared_expert_down_attr=None,
    
    moe_intermediate_size_attr="moe_intermediate_size",
))