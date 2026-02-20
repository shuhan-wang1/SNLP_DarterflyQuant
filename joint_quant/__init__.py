"""
JointQuant: INT4 Quantization with Joint R1 Rotation Optimization

This package implements:
- Joint R1 rotation training (our innovation)
- R2 per-head rotation (same as DartQuant)
- SmoothQuant integration
- W4A4 quantization

Key difference from DartQuant:
- DartQuant trains R1 layer-by-layer (greedy)
- JointQuant trains a single global R1 for all layers (joint optimization)
"""

from .config import JointQuantConfig, get_llama_1b_config, get_llama_3b_config, get_llama_8b_config, get_llama_70b_config
from .utils import DEV, set_seed, cleanup_memory, config_logging
from .model_utils import (
    get_model, get_tokenizer, get_model_type, 
    get_embeddings, get_transformer_layers, get_lm_head,
    RMSN
)
from .hadamard_utils import (
    get_hadamard_matrix, random_hadamard_matrix, 
    random_orthogonal_matrix, get_orthogonal_matrix
)
from .quant_utils import (
    WeightQuantizer, ActQuantizer, QuantizedLinear,
    ActQuantWrapper, add_actquant, find_qlayers
)
from .rotation_utils import (
    fuse_layer_norms, rotate_model,
    rotate_embeddings, rotate_head,
    rotate_attention_inputs, rotate_attention_output,
    rotate_mlp_input, rotate_mlp_output, rotate_ov_proj
)
from .smooth_utils import compute_smooth_scale, compute_simple_smooth_scale
from .data_utils import get_loaders, get_calibration_data, get_wikitext2, get_c4, get_ptb
from .eval_utils import evaluate_perplexity, evaluate_perplexity_simple
from .joint_training import (
    train_r1_joint, train_r2_independent,
    train_r1_joint_from_model, train_r2_from_model,
    collect_activations, collect_activations_from_model,
    whip_loss,
    R1Module, R2Module, JointR1Module
)

__version__ = "0.1.0"
__all__ = [
    # Config
    'JointQuantConfig', 'get_llama_1b_config', 'get_llama_3b_config', 'get_llama_8b_config', 'get_llama_70b_config',
    # Utils
    'DEV', 'set_seed', 'cleanup_memory', 'config_logging',
    # Model
    'get_model', 'get_tokenizer', 'get_model_type',
    'get_embeddings', 'get_transformer_layers', 'get_lm_head', 'RMSN',
    # Hadamard
    'get_hadamard_matrix', 'random_hadamard_matrix',
    'random_orthogonal_matrix', 'get_orthogonal_matrix',
    # Quantization
    'WeightQuantizer', 'ActQuantizer', 'QuantizedLinear',
    'ActQuantWrapper', 'add_actquant', 'find_qlayers',
    # Rotation
    'fuse_layer_norms', 'rotate_model',
    'rotate_embeddings', 'rotate_head',
    'rotate_attention_inputs', 'rotate_attention_output',
    'rotate_mlp_input', 'rotate_mlp_output', 'rotate_ov_proj',
    # Smooth
    'compute_smooth_scale', 'compute_simple_smooth_scale',
    # Data
    'get_loaders', 'get_calibration_data', 'get_wikitext2', 'get_c4', 'get_ptb',
    # Eval
    'evaluate_perplexity', 'evaluate_perplexity_simple',
    # Joint Training
    'train_r1_joint', 'train_r2_independent',
    'train_r1_joint_from_model', 'train_r2_from_model',
    'collect_activations', 'collect_activations_from_model',
    'whip_loss',
    'R1Module', 'R2Module', 'JointR1Module',
]
