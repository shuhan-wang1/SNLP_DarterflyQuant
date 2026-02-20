"""
Configuration presets for JointQuant
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class JointQuantConfig:
    """Configuration for JointQuant pipeline"""
    
    # Model
    model_name: str = "meta-llama/Llama-3.2-1B"
    hf_token: Optional[str] = None
    cache_dir: Optional[str] = None
    dtype: str = "auto"  # 'auto', 'float16', 'bfloat16', 'float32'
    
    # Calibration
    nsamples: int = 128
    seqlen: int = 2048
    seed: int = 0
    cal_dataset: str = "wikitext2"
    
    # R1 Training (Joint Optimization)
    # These match official DartQuant defaults
    r1_lr: float = 1e-3
    r1_epochs: int = 10
    r1_batch_size: int = 64
    r1_momentum: float = 0.9
    r1_init_mode: str = "hadamard"  # 'hadamard' recommended for stability
    r1_use_cosine_lr: bool = True
    
    # R2 Training (Per-Head, Per-Layer)
    r2_lr: float = 1e-3
    r2_epochs: int = 5
    r2_batch_size: int = 128
    r2_momentum: float = 0.9
    r2_accumulation_steps: int = 2
    r2_use_cosine_lr: bool = True
    
    # Joint Training (for comparison experiments)
    joint_lr: float = 5e-4
    joint_epochs: int = 100
    
    # Smoothing (SmoothQuant)
    use_smooth: bool = True
    smooth_alpha: float = 0.5  # 0.5 is balanced, good for W4A4
    
    # Quantization
    w_bits: int = 4
    a_bits: int = 4
    kv_bits: int = 16  # KV cache bits (16 = no quantization)
    w_groupsize: int = 128  # Group-wise weight quantization
    group_size: int = 128  # Alias for w_groupsize (for compatibility)
    w_sym: bool = True  # Symmetric weight quantization
    a_sym: bool = False  # Asymmetric activation quantization (safer)
    a_groupsize: int = -1  # -1 = per-token, >0 = group-wise
    
    # Evaluation
    eval_dataset: str = "wikitext2"
    eval_batch_size: int = 1
    
    # Output
    output_dir: str = "./jointquant_output"
    save_rotations: bool = True
    save_model: bool = False
    
    # Device
    device: str = "cuda"


# Preset configurations

def get_llama_1b_config(**kwargs) -> JointQuantConfig:
    """Preset for Llama-3.2-1B"""
    config = JointQuantConfig(
        model_name="meta-llama/Llama-3.2-1B",
        nsamples=128,
        seqlen=2048,
        r1_epochs=10,
        r2_epochs=5,
        joint_epochs=100,
        **kwargs
    )
    return config


def get_llama_3b_config(**kwargs) -> JointQuantConfig:
    """Preset for Llama-3.2-3B"""
    config = JointQuantConfig(
        model_name="meta-llama/Llama-3.2-3B",
        nsamples=128,
        seqlen=2048,
        r1_epochs=10,
        r2_epochs=5,
        joint_epochs=100,
        **kwargs
    )
    return config


def get_llama_8b_config(**kwargs) -> JointQuantConfig:
    """Preset for Llama-3-8B"""
    config = JointQuantConfig(
        model_name="meta-llama/Meta-Llama-3-8B",
        nsamples=128,
        seqlen=2048,
        r1_epochs=10,
        r2_epochs=5,
        joint_epochs=100,
        # Lower batch size for larger models
        r1_batch_size=32,
        r2_batch_size=64,
        **kwargs
    )
    return config


def get_llama_70b_config(**kwargs) -> JointQuantConfig:
    """Preset for Llama-3-70B"""
    config = JointQuantConfig(
        model_name="meta-llama/Meta-Llama-3-70B",
        nsamples=128,
        seqlen=2048,
        r1_epochs=10,
        r2_epochs=5,
        joint_epochs=100,
        # Much lower batch size for 70B
        r1_batch_size=8,
        r2_batch_size=16,
        **kwargs
    )
    return config
