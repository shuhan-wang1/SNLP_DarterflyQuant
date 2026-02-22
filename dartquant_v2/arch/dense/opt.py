"""
Meta OPT family — Dense architecture registration.

Covers: OPT-125M, OPT-1.3B, OPT-6.7B, OPT-13B, OPT-30B, OPT-66B
All share the config class name: OPTConfig.

Layer structure (per decoder layer):
  self_attn.{q,k,v}_proj  -- attention projections
  self_attn.out_proj       -- output projection  (note: not o_proj)
  fc1                      -- FFN input (no gate, no up/gate split)
  fc2                      -- FFN output (analogous to down_proj)
  self_attn_layer_norm     -- pre-attention LayerNorm
  final_layer_norm         -- pre-FFN LayerNorm

OPT differences from Llama:
  - No RoPE (absolute position embeddings)
  - No gate projection (fc1 only, no up/gate SiLU)
  - Requires mean-baking (OPT adds a mean shift via LayerNorm)
  - Decoder is nested: model.decoder.layers (not model.layers)
"""

from dartquant_v2.unified_model import ModelArchConfig, register_arch

register_arch("OPTConfig", ModelArchConfig(
    # Model-level paths (nested under model.decoder)
    embed_tokens_path="model.decoder.embed_tokens",
    embed_positions_path="model.decoder.embed_positions",
    layers_path="model.decoder.layers",
    pre_head_norm_path="model.decoder.final_layer_norm",
    lm_head_path="lm_head",

    # Attention projections (relative to layer)
    q_proj_attr="self_attn.q_proj",
    k_proj_attr="self_attn.k_proj",
    v_proj_attr="self_attn.v_proj",
    o_proj_attr="self_attn.out_proj",   # OPT uses out_proj, not o_proj
    self_attn_attr="self_attn",

    # MLP projections: OPT has no gate, uses fc1/fc2
    mlp_up_proj_attr=None,
    mlp_gate_proj_attr=None,
    mlp_down_proj_attr="fc2",

    # Layer norms
    input_ln_attr="self_attn_layer_norm",
    post_attn_ln_attr="final_layer_norm",
    norm_class_name="LayerNorm",

    # No RoPE
    has_rope=False,
    rope_function_name="",

    # OPT requires mean subtraction before rotation
    needs_mean_baking=True,
    no_split_module_class="OPTDecoderLayer",

    # MoE: not applicable
    is_moe=False,
))
