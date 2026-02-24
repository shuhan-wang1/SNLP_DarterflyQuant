#!/usr/bin/env python3
"""
DartQuant v2 - Loss Function & Butterfly Rotation Visualization

Generates empirical experiment plots by fully mirroring the dartquant_v2
pipeline's rotation matrix computation (R1, R2, R3, R4).  Weight quantization
and PPL evaluation are intentionally omitted — only rotation matrices are
computed and visualized.

Pipeline mirroring:
  R1: collect from ALL layers' mlp.up_proj + self_attn.q_proj inputs
      → train via QR-Orth parameterization (R1_QR from dartquant_v2.trainers)
  R2: collect from ALL layers' self_attn.o_proj inputs
      → per-head rotation (R2_Per_Head from dartquant_v2.trainers)
  R3: hook q_proj outputs, reshape to (N*num_heads, head_dim)
      → Butterfly Givens rotation (mirrors pipeline.collect_r3_activations)
  R4: fixed Random Hadamard transform (no butterfly training)
      → Hadamard applied at pipeline runtime; no comparison experiment.

Experiments:
  Experiment 1: R1 loss function comparison (Whip / SWD_Unif / SWD_Gauss)
    - All-layer pipeline-style activation collection
    - Distribution histograms + training loss curves
  Experiment 2: R2 per-head rotation (report final per-layer loss)
  Experiment 3: Butterfly R3 vs Hadamard (R4 skipped — uses fixed Hadamard)
    - R3 distributions + variance uniformity + training curves

Configuration mirrors scripts/stat_and_download.py exactly.
Output directory: {CACHE_DIR}/dartquant_v2_plots/
"""

import os
import sys
import gc
import math
import random
import csv
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# USER CONFIGURATION  (mirrors stat_and_download.py)
# ============================================================================

HF_TOKEN = None   # Set to "hf_xxx" if needed for gated models

# Use HuggingFace mirror (faster in China / AutoDL)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Cache directories (same as stat_and_download.py)
CACHE_DIR = "/root/autodl-tmp"
HF_HOME   = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_HOME"]            = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = HF_HOME
os.environ["HF_DATASETS_CACHE"]  = os.path.join(CACHE_DIR, "datasets")

# Output directory for plots
PLOT_DIR = os.path.join(CACHE_DIR, "dartquant_v2_plots")

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(HF_HOME).mkdir(parents=True, exist_ok=True)
Path(os.environ["HF_DATASETS_CACHE"]).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

print(f"Cache dir : {CACHE_DIR}")
print(f"HF Home   : {HF_HOME}")
print(f"Plot dir  : {PLOT_DIR}")

# Target models (same as user setup)
MODELS = [
    "meta-llama/Llama-3.2-1B",
    #"meta-llama/Llama-3.2-1B-Instruct",
]

# ── Data / collection settings ──────────────────────────────────────────────
SEQ_LENGTH  = 2048    # token sequence length
NUM_SAMPLES = 64      # number of calibration sequences
BATCH_FWD   = 4       # sequences per forward pass during collection
MAX_ACTS    = 512     # max activation rows kept per hook call (speed vs quality)
DTYPE       = torch.float16

# ── Training hyperparameters (match dartquant_v2.pipeline defaults) ──────────
LR_LOSS     = 1e-3    # learning rate for R1 / R2
LR_BF       = 1e-3    # learning rate for Butterfly R3/R4
MOMENTUM    = 0.9
EPOCHS_LOSS = 10      # R1/R2 epochs (matches pipeline --r1_epochs 10)
EPOCHS_BF   = 250      # Butterfly epochs (matches pipeline --butterfly_epochs)
BATCH_SIZE  = 64

# ============================================================================
# sys.path setup: make dartquant_v2 and DartQuant importable
# ============================================================================

_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "DartQuant", "fake_quant"))
sys.path.insert(0, os.path.join(_project_root, "DartQuant", "calibrater"))

# ── Import dartquant_v2 modules (full pipeline imports) ──────────────────────
_DQ_IMPORTS_OK = False
try:
    import dartquant_v2  # triggers arch/ registration
    from dartquant_v2.unified_model import get_arch_config
    from dartquant_v2.loss_functions import (
        calc_whip_loss, calc_swd_unif_loss, calc_swd_gauss_loss,
        calc_kl_unif_loss, calc_kl_gauss_loss,
        get_loss_fn,
    )
    from dartquant_v2.butterfly import ButterflyRotation, ButterflyFactored
    from dartquant_v2.trainers import (
        R1_QR,
        R2_Per_Head,
        _get_init_matrix  as _dq_get_init_matrix,
        _get_multi_head_init as _dq_get_multi_head_init,
    )
    from dartquant_v2.int4_quantizer import INT4FakeQuantizer
    from dartquant_v2.nf4_quantizer  import NF4FakeQuantizer
    _DQ_IMPORTS_OK = True
    print("dartquant_v2 imported successfully (full pipeline mode)")

except ImportError as e:
    print(f"Warning: dartquant_v2 import failed ({e}). Using inline fallbacks.")

    # ── Inline fallback loss functions ──────────────────────────────────────
    def calc_whip_loss(x):
        return torch.sum(torch.exp(-x.abs()), dim=-1, keepdim=True).mean()

    def calc_swd_unif_loss(x):
        import torch.nn.functional as F
        xf = x.reshape(-1)
        xs, _ = torch.sort(xf)
        with torch.no_grad():
            rms = torch.sqrt(torch.mean(xf ** 2))
            b   = math.sqrt(3) * rms
            t   = torch.linspace(-b.item(), b.item(), len(xf), device=x.device)
        return F.mse_loss(xs, t)

    def calc_swd_gauss_loss(x):
        import torch.nn.functional as F
        xf = x.reshape(-1)
        xs, _ = torch.sort(xf)
        N = len(xf)
        with torch.no_grad():
            sigma = torch.sqrt(torch.mean(xf ** 2))
            p = (torch.arange(1, N + 1, device=x.device) - 0.5) / N
            p = p.clamp(1e-6, 1 - 1e-6)
            t = math.sqrt(2) * torch.erfinv(2 * p - 1) * sigma
        return F.mse_loss(xs, t)

    # ── Inline fallback KL divergence losses (mirrors dartquant_v2) ────────

    def calc_kl_unif_loss(x):
        """KL divergence to Uniform via differential-entropy maximisation.

        KL(P || Unif[-b,b]) = log(2b) - H(P).  Since b = sqrt(3)*RMS is
        rotation-invariant, minimising KL reduces to maximising H(P),
        estimated by the Vasicek (1976) spacing estimator:
            H(P) ≈ (1/n) Σ_i log(n · Δx_i)
        The gradient equalises neighbouring spacings → uniform coverage.
        Pairs with INT4 uniform quantiser.
        """
        x_flat = x.reshape(-1).float()
        n = x_flat.numel()
        x_sorted, _ = torch.sort(x_flat)
        spacings = torch.empty(n, device=x_flat.device, dtype=x_flat.dtype)
        spacings[1:-1] = (x_sorted[2:] - x_sorted[:-2]) / 2.0
        spacings[0]    = x_sorted[1] - x_sorted[0]
        spacings[-1]   = x_sorted[-1] - x_sorted[-2]
        spacings = spacings.clamp(min=1e-8)
        neg_entropy = -(torch.log(n * spacings)).mean()
        return neg_entropy

    def calc_kl_gauss_loss(x):
        """KL divergence to Gaussian via Gram-Charlier moment matching.

        KL(P || N(0,σ²)) ≈ γ₁²/12 + γ₂²/96.  Minimising the surrogate
        L = γ₁² + γ₂² (skewness² + excess_kurtosis²) drives the
        distribution toward Gaussian shape (γ₁=0, γ₂=0).
        Pairs with NF4 (Normal Float 4-bit) quantiser.
        """
        x_flat = x.reshape(-1).float()
        mu = x_flat.mean()
        centered = x_flat - mu
        sigma2 = (centered ** 2).mean()
        sigma  = torch.sqrt(sigma2 + 1e-8)
        skewness        = (centered ** 3).mean() / (sigma ** 3)
        excess_kurtosis = (centered ** 4).mean() / (sigma ** 4) - 3.0
        return skewness ** 2 + excess_kurtosis ** 2

    def get_loss_fn(name):
        return _LOSS_FNS_FALLBACK[name]

    def get_arch_config(config_class_name):
        return None

    # ── Inline fallback R1_QR ───────────────────────────────────────────────
    class R1_QR(nn.Module):
        """QR-Orth parameterized rotation (fallback, mirrors dartquant_v2)."""
        def __init__(self, h):
            super().__init__()
            self.hidden_size = h
            self.matrix = nn.Parameter(torch.eye(h))
            self.rotate = None
        def forward(self, x):
            self.rotate, _ = torch.linalg.qr(self.matrix, mode='complete')
            return torch.matmul(x, self.rotate)

    # ── Inline fallback R2_Per_Head ─────────────────────────────────────────
    class R2_Per_Head(nn.Module):
        """Per-head QR-Orth rotation (fallback, mirrors dartquant_v2)."""
        def __init__(self, hidden_size, head_num, kv_head):
            super().__init__()
            assert hidden_size % head_num == 0
            self.hidden_size = hidden_size
            self.head_num = head_num
            self.head_dim = hidden_size // head_num
            self.kv_head  = kv_head
            self.matrix = nn.Parameter(
                torch.eye(self.head_dim).repeat(kv_head, 1, 1)
            )
            self.rotate = None
        def forward(self, x):
            x_shape = x.shape
            x = x.reshape(-1, x_shape[-1])
            x = x.reshape(-1, self.head_num, self.head_dim)
            x = x.transpose(0, 1)
            self.rotate, _ = torch.linalg.qr(self.matrix)
            rotate_exp = self.rotate[:, None, :, :].expand(
                self.kv_head, self.head_num // self.kv_head,
                self.head_dim, self.head_dim
            )
            rotate_exp = rotate_exp.reshape(self.head_num, self.head_dim, self.head_dim)
            r_x = torch.matmul(x, rotate_exp)
            r_x = r_x.transpose(0, 1).reshape(x_shape)
            return r_x

    # ── Inline fallback Butterfly modules ───────────────────────────────────
    class ButterflyRotation(nn.Module):
        def __init__(self, dim):
            super().__init__()
            if dim <= 0 or (dim & (dim - 1)) != 0:
                raise ValueError(f"ButterflyRotation requires power-of-2 dim, got {dim}")
            self.dim = dim
            self.num_layers = int(math.log2(dim))
            # Initialize to 0 (identity); Hadamard warm-start done by
            # _init_butterfly_from_hadamard() before the main training loop.
            self.angles = nn.Parameter(torch.zeros(self.num_layers, dim // 2))
            for l in range(self.num_layers):
                stride = 2 ** l
                bs = 2 ** (l + 1)
                i_idx = [i for i in range(dim) if i % bs < stride]
                j_idx = [i + stride for i in i_idx]
                self.register_buffer(f'_i_idx_{l}', torch.tensor(i_idx, dtype=torch.long))
                self.register_buffer(f'_j_idx_{l}', torch.tensor(j_idx, dtype=torch.long))
        def forward(self, x):
            orig_shape = x.shape
            x = x.reshape(-1, self.dim)
            for l in range(self.num_layers):
                thetas = self.angles[l]
                cos_t, sin_t = torch.cos(thetas), torch.sin(thetas)
                i_idx = getattr(self, f'_i_idx_{l}')
                j_idx = getattr(self, f'_j_idx_{l}')
                xi, xj = x[:, i_idx], x[:, j_idx]
                result = x.clone()
                result[:, i_idx] = cos_t * xi + sin_t * xj
                result[:, j_idx] = -sin_t * xi + cos_t * xj
                x = result
            return x.reshape(orig_shape)
        def inverse_forward(self, x):
            orig_shape = x.shape
            x = x.reshape(-1, self.dim)
            for l in reversed(range(self.num_layers)):
                thetas = self.angles[l]
                cos_t, sin_t = torch.cos(thetas), torch.sin(thetas)
                i_idx = getattr(self, f'_i_idx_{l}')
                j_idx = getattr(self, f'_j_idx_{l}')
                xi, xj = x[:, i_idx], x[:, j_idx]
                result = x.clone()
                result[:, i_idx] = cos_t * xi - sin_t * xj
                result[:, j_idx] = sin_t * xi + cos_t * xj
                x = result
            return x.reshape(orig_shape)
        def get_matrix(self):
            I = torch.eye(self.dim, device=self.angles.device, dtype=self.angles.dtype)
            return self.forward(I)

    class ButterflyFactored(nn.Module):
        """Factored butterfly for non-power-of-2 dims (fallback)."""
        def __init__(self, total_dim, k_factor_mode='latent'):
            super().__init__()
            self.total_dim = total_dim
            self.k_factor_mode = k_factor_mode
            if total_dim > 0 and (total_dim & (total_dim - 1)) == 0:
                self.K = 1; self.m = total_dim
                self.butterfly = ButterflyRotation(total_dim)
            else:
                self.K, self.m = self._factor(total_dim)
                self.butterfly = ButterflyRotation(self.m)
                if k_factor_mode == 'latent':
                    self.latent_matrix = nn.Parameter(torch.eye(self.K))
                elif k_factor_mode == 'cayley':
                    self.cayley_A = nn.Parameter(torch.zeros(self.K, self.K))
                else:
                    raise ValueError(f"Unknown k_factor_mode: {k_factor_mode}")
        @staticmethod
        def _factor(n):
            for K in [172,156,140,108,60,52,36,28,40,20,12]:
                if n % K == 0:
                    m = n // K
                    if m > 0 and (m & (m-1)) == 0:
                        return K, m
            for K in range(2, n+1):
                if n % K == 0:
                    m = n // K
                    if m > 0 and (m & (m-1)) == 0:
                        return K, m
            raise ValueError(f"Cannot factorize {n}")
        def _get_Q(self):
            if self.k_factor_mode == 'cayley':
                A = (self.cayley_A - self.cayley_A.T) / 2
                I = torch.eye(self.K, device=A.device, dtype=A.dtype)
                return torch.linalg.solve(I + A, I - A)
            else:
                Q, _ = torch.linalg.qr(self.latent_matrix)
                return Q
        def forward(self, x):
            if self.K == 1:
                return self.butterfly(x)
            orig = x.shape
            x = x.reshape(-1, self.K, self.m)
            b = x.shape[0]
            x = self.butterfly(x.reshape(-1, self.m)).reshape(b, self.K, self.m)
            Q_K = self._get_Q().to(x.dtype)
            x = torch.einsum('ij,bjk->bik', Q_K, x)
            return (x / math.sqrt(self.total_dim)).reshape(orig)
        def inverse_forward(self, x):
            if self.K == 1:
                return self.butterfly.inverse_forward(x)
            orig = x.shape
            x = x.reshape(-1, self.K, self.m) * math.sqrt(self.total_dim)
            Q_K = self._get_Q().to(x.dtype)
            x = torch.einsum('ji,bjk->bik', Q_K, x)
            b = x.shape[0]
            x = self.butterfly.inverse_forward(x.reshape(-1, self.m)).reshape(b, self.K, self.m)
            return x.reshape(orig)
        def get_matrix(self):
            I = torch.eye(self.total_dim, device=self.butterfly.angles.device,
                          dtype=self.butterfly.angles.dtype)
            return self.forward(I)

    def _dq_get_init_matrix(size, mode, device):
        Q, R = torch.linalg.qr(torch.randn(size, size, dtype=torch.float64).to(device))
        return Q * torch.sign(torch.diag(R)).unsqueeze(0)

    def _dq_get_multi_head_init(hidden_size, head_num, kv_head, mode, device):
        head_dim = hidden_size // head_num
        org = _dq_get_init_matrix(head_dim, mode, device)
        return org.unsqueeze(0).repeat(kv_head, 1, 1)

    # ── Inline fallback fake quantizers (mirrors dartquant_v2 quantizers) ───
    class INT4FakeQuantizer(nn.Module):
        """INT4 symmetric fake quantizer (fallback, mirrors dartquant_v2.int4_quantizer)."""
        def __init__(self, block_size: int = 64, bits: int = 4):
            super().__init__()
            self.block_size = block_size
            self.q_max = 2 ** (bits - 1) - 1  # 7 for INT4
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_shape, orig_dtype = x.shape, x.dtype
            flat = x.float().reshape(-1)
            pad_size = (-flat.numel()) % self.block_size
            if pad_size:
                flat = torch.cat([flat, torch.zeros(pad_size, device=x.device)])
            blocks = flat.reshape(-1, self.block_size)
            absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
            scale  = absmax / self.q_max
            q      = torch.clamp(torch.round(blocks / scale), -self.q_max, self.q_max)
            x_deq  = (q * scale).reshape(-1)
            if pad_size:
                x_deq = x_deq[:flat.numel() - pad_size]
            return x_deq.reshape(orig_shape).to(orig_dtype)

    class NF4FakeQuantizer(nn.Module):
        """NF4 fake quantizer (fallback, mirrors dartquant_v2.nf4_quantizer)."""
        _LEVELS = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                   -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                   0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                   0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                   0.7229568362236023, 1.0]
        def __init__(self, block_size: int = 64):
            super().__init__()
            self.block_size = block_size
            self.register_buffer('levels', torch.tensor(self._LEVELS))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_shape, orig_dtype = x.shape, x.dtype
            x = x.float()
            flat = x.reshape(-1)
            pad_size = (-flat.numel()) % self.block_size
            if pad_size:
                flat = torch.cat([flat, torch.zeros(pad_size, device=x.device)])
            blocks = flat.reshape(-1, self.block_size)
            absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
            x_norm = blocks / absmax
            levels = self.levels.to(x.device)
            idx    = (x_norm.unsqueeze(-1) - levels.reshape(1, 1, -1)).abs().argmin(dim=-1)
            x_deq  = (levels[idx] * absmax).reshape(-1)
            if pad_size:
                x_deq = x_deq[:flat.numel() - pad_size]
            return x_deq.reshape(orig_shape).to(orig_dtype)

# ── Lazy fallback registry (used when _DQ_IMPORTS_OK is False) ──────────────
_LOSS_FNS_FALLBACK = {
    'whip':      calc_whip_loss,
    'swd_unif':  calc_swd_unif_loss,
    'swd_gauss': calc_swd_gauss_loss,
    'kl_unif':   calc_kl_unif_loss,
    'kl_gauss':  calc_kl_gauss_loss,
}

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    print("transformers + datasets imported successfully")
except ImportError as e:
    print(f"Error: required library missing: {e}")
    print("Install: pip install transformers accelerate datasets")
    sys.exit(1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


def _get_loss_fn(name: str):
    """Resolve loss function; uses dartquant_v2.get_loss_fn when available."""
    if _DQ_IMPORTS_OK:
        return get_loss_fn(name)
    return _LOSS_FNS_FALLBACK[name]


# ============================================================================
# SECTION 1 – ARCHITECTURE UTILITIES & PIPELINE-STYLE ACTIVATION COLLECTION
# ============================================================================

class _ArchWrapper:
    """
    Thin wrapper providing architecture-aware accessors for an already-loaded
    model, without double-loading from disk.

    Mirrors the interface of dartquant_v2.UnifiedQuantModel that the pipeline
    uses to build activation collection targets and model-dimension properties.
    """

    def __init__(self, model):
        self.model = model
        cfg = model.config

        # ── Model dimensions ────────────────────────────────────────────────
        self.hidden_size       = cfg.hidden_size
        self.num_heads         = cfg.num_attention_heads
        self.head_dim          = cfg.hidden_size // cfg.num_attention_heads
        self.kv_heads          = getattr(cfg, 'num_key_value_heads', self.num_heads)
        self.num_layers        = cfg.num_hidden_layers
        self.intermediate_size = (
            getattr(cfg, 'intermediate_size', None) or
            getattr(cfg, 'ffn_dim', None)
        )

        # ── Architecture config (from dartquant_v2 registry) ────────────────
        self._arch = None
        if _DQ_IMPORTS_OK and get_arch_config is not None:
            try:
                self._arch = get_arch_config(cfg.__class__.__name__)
                print(f"  Arch config: {cfg.__class__.__name__} (dartquant_v2 registry)")
            except Exception as exc:
                print(f"  Warning: arch config lookup failed ({exc}); using defaults")

    # ── Attribute path helpers (Llama-style defaults if no arch config) ──────

    def _layers_path(self):
        return self._arch.layers_path if self._arch else "model.layers"

    def _up_proj_attr(self):
        if self._arch and self._arch.mlp_up_proj_attr:
            return self._arch.mlp_up_proj_attr
        return "mlp.up_proj"

    def _q_proj_attr(self):
        return self._arch.q_proj_attr if self._arch else "self_attn.q_proj"

    def _o_proj_attr(self):
        return self._arch.o_proj_attr if self._arch else "self_attn.o_proj"

    def _down_proj_attr(self):
        return self._arch.mlp_down_proj_attr if self._arch else "mlp.down_proj"

    # ── Pipeline activation target lists ────────────────────────────────────
    # These exactly mirror the target construction in pipeline.py Step 4 / Step 6.

    def r1_collection_targets(self):
        """
        Module paths for R1 activation collection.
        Mirrors pipeline.py lines 806–813:
          for each layer: up_proj input + q_proj input
        """
        lp = self._layers_path()
        up = self._up_proj_attr()
        q  = self._q_proj_attr()
        targets = []
        for i in range(self.num_layers):
            if up:
                targets.append(f"{lp}.{i}.{up}")
            targets.append(f"{lp}.{i}.{q}")
        return targets

    def r2_collection_targets(self):
        """
        Module paths for R2 activation collection.
        Mirrors pipeline.py lines 862–865:
          for each layer: o_proj input
        """
        lp = self._layers_path()
        o  = self._o_proj_attr()
        return [f"{lp}.{i}.{o}" for i in range(self.num_layers)]

    def r4_collection_targets(self):
        """
        Module paths for R4 activation collection.
        Mirrors pipeline.collect_r4_activations: down_proj input, all layers.
        """
        lp   = self._layers_path()
        down = self._down_proj_attr()
        return [f"{lp}.{i}.{down}" for i in range(self.num_layers)]

    def _k_proj_attr(self):
        return self._arch.k_proj_attr if self._arch else "self_attn.k_proj"

    def _down_proj_weight_for_layer(self, layer_idx: int):
        """
        Return the down_proj weight matrix (cpu, float32) for a given layer.
        Used to build the R4 weight bank for joint weight+activation recon loss.
        """
        import operator
        lp   = self._layers_path()
        down = self._down_proj_attr()
        layer = list(operator.attrgetter(lp)(self.model))[layer_idx]
        proj  = operator.attrgetter(down)(layer)
        return proj.weight.data.float().cpu()   # (hidden_size, intermediate_size)

    def get_q_proj_modules(self):
        """
        Return all q_proj nn.Module objects (for output-side R3 hooks).
        Mirrors pipeline.collect_r3_activations hook registration.
        """
        import operator
        modules = []
        lp = self._layers_path()
        q  = self._q_proj_attr()
        for layer in operator.attrgetter(lp)(self.model):
            modules.append(operator.attrgetter(q)(layer))
        return modules

    def get_qk_proj_modules_for_layer(self, layer_idx: int):
        """
        Return (q_proj, k_proj) modules for a specific layer.
        Flaw 3 fix: R3 acts jointly on Q and K after RoPE, so both projections
        must be included in the activation distribution being optimized.
        """
        import operator
        lp    = self._layers_path()
        q_att = self._q_proj_attr()
        k_att = self._k_proj_attr()
        layer = list(operator.attrgetter(lp)(self.model))[layer_idx]
        return (operator.attrgetter(q_att)(layer),
                operator.attrgetter(k_att)(layer))

    def layer_index_from_name(self, name: str) -> int:
        """Parse integer layer index from a dotted module path."""
        for part in name.split('.'):
            if part.isdigit():
                return int(part)
        return -1


def collect_pipeline_activations(model, calib_data, target_names, device,
                                  max_per_hook: int = MAX_ACTS):
    """
    Collect *input* activations at the specified module paths.

    Exactly mirrors dartquant_v2.pipeline.collect_activations:
      - forward hooks on input[0]
      - subsample to max_per_hook rows per call
      - concatenate across all forward passes

    Args:
        model        : HuggingFace model
        calib_data   : (nsamples, seqlen) int64 tensor
        target_names : list of module path strings (e.g. "model.layers.0.mlp.up_proj")
        device       : torch.device for forward pass
        max_per_hook : max rows kept per hook invocation

    Returns:
        dict[str, Tensor]  –  name → float32 cpu tensor of shape (N, dim)
    """
    activations = {name: [] for name in target_names}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            tensor = inp[0] if isinstance(inp, tuple) else inp
            if not isinstance(tensor, torch.Tensor):
                return
            flat = tensor.detach().float().cpu().reshape(-1, tensor.shape[-1])
            if flat.shape[0] > max_per_hook:
                idx  = torch.randperm(flat.shape[0])[:max_per_hook]
                flat = flat[idx]
            activations[name].append(flat)
        return hook

    for name in target_names:
        try:
            module = model.get_submodule(name)
            hooks.append(module.register_forward_hook(make_hook(name)))
        except AttributeError:
            print(f"    Warning: module '{name}' not found, skipping")

    model.eval()
    with torch.no_grad():
        for i in range(0, calib_data.shape[0], BATCH_FWD):
            batch = calib_data[i:i + BATCH_FWD].to(device)
            try:
                model(batch)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    result = {}
    for name in target_names:
        if activations[name]:
            result[name] = torch.cat(activations[name], dim=0)
    return result


def collect_r3_activations_pipeline(model, calib_data, arch_wrapper: _ArchWrapper,
                                     device, max_per_layer: int = 256):
    """
    Collect Q *and* K proj output activations for R3 training.

    Flaw 3 fix: R3 is an online operator applied to *both* Q and K vectors
    after RoPE:  Attention(Q·R3^T, R3·K^T, V).  Training on Q alone misses
    K's marginal distribution, which causes R3 to produce outliers in K and
    can lead to exponential overflow in attention scores.

    This function hooks both q_proj and k_proj outputs for every layer,
    concatenates Q‖K per token, and reshapes to (N_total, head_dim) for
    the butterfly optimizer — so the learned R3 simultaneously optimizes
    the joint QK distribution.

    Collection uses only the first 16 calibration samples (same as pipeline).

    Returns:
        np.ndarray of shape (N_total_qk * num_heads, head_dim), float32
        or None if no activations were collected.
    """
    activations = []
    hooks = []

    def make_hook():
        def hook(module, inp, out):
            flat = out.detach().float().cpu().reshape(-1, out.shape[-1])
            if flat.shape[0] > max_per_layer:
                idx  = torch.randperm(flat.shape[0])[:max_per_layer]
                flat = flat[idx]
            activations.append(flat)
        return hook

    # Hook BOTH q_proj and k_proj outputs (Flaw 3 fix)
    import operator
    lp    = arch_wrapper._layers_path()
    q_att = arch_wrapper._q_proj_attr()
    k_att = arch_wrapper._k_proj_attr()
    for layer in operator.attrgetter(lp)(model):
        q_mod = operator.attrgetter(q_att)(layer)
        k_mod = operator.attrgetter(k_att)(layer)
        hooks.append(q_mod.register_forward_hook(make_hook()))
        hooks.append(k_mod.register_forward_hook(make_hook()))

    model.eval()
    with torch.no_grad():
        # Pipeline uses only 16 samples for R3 (collect_r3_activations line 255)
        for i in range(min(calib_data.shape[0], 16)):
            inp = calib_data[i:i + 1].to(device)
            try:
                model(inp)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    if not activations:
        return None

    all_acts = torch.cat(activations, dim=0)  # (N_total, hidden_size)
    head_dim  = arch_wrapper.head_dim
    return all_acts.reshape(-1, head_dim).numpy().astype(np.float32)


def collect_r3_activations_per_layer(model, calib_data, arch_wrapper: _ArchWrapper,
                                      device, max_per_hook: int = 256):
    """
    Collect Q *and* K proj output activations **per layer** for per-layer R3 training.

    Flaw 2 fix variant of collect_r3_activations_pipeline: instead of returning
    one concatenated array across all layers, returns a dict keyed by layer index
    so that each layer's butterfly can be trained independently on *that* layer's
    own covariance structure — preventing gradient cancellation that would arise
    from mixing activations with different covariances Σ_l.

    Flaw 3 fix is preserved: both q_proj and k_proj outputs from the same layer are
    concatenated before being added to that layer's activation pool, so the optimizer
    sees the joint Q+K distribution for each individual layer.

    A single model forward pass collects all layers simultaneously (no per-layer
    re-runs needed), keeping total compute at O(num_layers) × (16 samples).

    Args:
        model        : HuggingFace model
        calib_data   : (nsamples, seqlen) int64 tensor
        arch_wrapper : _ArchWrapper instance
        device       : torch.device for forward pass
        max_per_hook : max head-vectors kept per hook invocation per layer

    Returns:
        dict {int layer_idx → np.ndarray of shape (N, head_dim), float32}
        Empty dict if no activations were collected.
    """
    import operator

    head_dim   = arch_wrapper.head_dim
    lp         = arch_wrapper._layers_path()
    q_att      = arch_wrapper._q_proj_attr()
    k_att      = arch_wrapper._k_proj_attr()
    layers     = list(operator.attrgetter(lp)(model))
    num_layers = len(layers)

    layer_acts: dict = {i: [] for i in range(num_layers)}
    hooks: list = []

    for layer_idx, layer in enumerate(layers):
        # Closure captures lidx by value via default argument trick
        def make_hook(lidx):
            def hook(module, inp, out):
                flat = out.detach().float().cpu().reshape(-1, head_dim)
                if flat.shape[0] > max_per_hook:
                    idx  = torch.randperm(flat.shape[0])[:max_per_hook]
                    flat = flat[idx]
                layer_acts[lidx].append(flat)
            return hook

        q_mod = operator.attrgetter(q_att)(layer)
        k_mod = operator.attrgetter(k_att)(layer)
        hooks.append(q_mod.register_forward_hook(make_hook(layer_idx)))
        hooks.append(k_mod.register_forward_hook(make_hook(layer_idx)))

    model.eval()
    with torch.no_grad():
        # Pipeline uses only 16 samples for R3 (collect_r3_activations line 255)
        for i in range(min(calib_data.shape[0], 16)):
            inp = calib_data[i:i + 1].to(device)
            try:
                model(inp)
            except Exception:
                pass

    for h in hooks:
        h.remove()

    result: dict = {}
    for lidx, acts_list in layer_acts.items():
        if acts_list:
            result[lidx] = torch.cat(acts_list, dim=0).numpy().astype(np.float32)
    return result


# ── Helpers shared by multiple functions ─────────────────────────────────────

def load_model_and_tokenizer(model_name: str):
    """Load model + tokenizer, mirrors stat_and_download.py pattern."""
    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=HF_HOME, trust_remote_code=True, token=HF_TOKEN
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
        device_map="auto",
        cache_dir=HF_HOME,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    )
    model.eval()
    return model, tokenizer


def load_calibration_data(tokenizer, seq_length: int = SEQ_LENGTH,
                          nsamples: int = NUM_SAMPLES) -> torch.Tensor:
    """Load WikiText-2 calibration data. Falls back to random tokens."""
    try:
        print("  Loading WikiText-2 calibration data ...")
        dataset = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train",
            cache_dir=os.environ["HF_DATASETS_CACHE"], token=HF_TOKEN
        )
        texts = [x['text'] for x in dataset if x['text'].strip()]
        full_text = " ".join(texts[:500])
        toks = tokenizer(full_text, return_tensors="pt", truncation=False)
        ids  = toks['input_ids']
        chunks = []
        for i in range(0, ids.shape[1] - seq_length, seq_length):
            chunks.append(ids[:, i:i + seq_length])
            if len(chunks) >= nsamples:
                break
        if not chunks:
            raise ValueError("text too short for even one chunk")
        data = torch.cat(chunks[:nsamples], dim=0)
        print(f"  Calibration data shape: {data.shape}")
        return data
    except Exception as e:
        print(f"  WikiText-2 unavailable ({e}), using random fallback")
        vocab_size = getattr(tokenizer, 'vocab_size', 32000)
        return torch.randint(0, vocab_size, (nsamples, seq_length))


def get_layer_indices(num_layers: int):
    """First / middle / last layers – mirrors stat_and_download.py."""
    return [0, num_layers // 2, num_layers - 1]


def _model_device(model) -> torch.device:
    """Return device of first model parameter (handles device_map='auto')."""
    return next(model.parameters()).device


# ============================================================================
# SECTION 2 – TRAINING HELPERS  (dartquant_v2.trainers logic + history)
# ============================================================================
#
# Each function is a faithful mirror of the corresponding dartquant_v2.trainers
# function (same modules, same initialization, same optimizer & scheduler),
# extended with per-epoch loss history tracking for visualization.
#
# Initialization:
#   _make_random_orthogonal → mirrors trainers._random_orthogonal_matrix
#   _make_init_matrix       → mirrors trainers._get_init_matrix
#   (tries hadamard_utils, falls back to random)
# ============================================================================

def _make_random_orthogonal(size: int, device) -> torch.Tensor:
    """Random orthogonal matrix via QR (float64). Mirrors trainers._random_orthogonal_matrix."""
    M   = torch.randn(size, size, dtype=torch.float64).to(device)
    Q, R = torch.linalg.qr(M)
    Q   *= torch.sign(torch.diag(R)).unsqueeze(0)
    return Q


def _make_init_matrix(size: int, device, mode: str = 'random') -> torch.Tensor:
    """
    Get initialization matrix for R1.
    Mirrors dartquant_v2.trainers._get_init_matrix:
      'hadamard' → tries hadamard_utils.random_hadamard_matrix, else random
      'random'   → _random_orthogonal_matrix
    """
    if mode == 'hadamard':
        try:
            from hadamard_utils import random_hadamard_matrix
            return random_hadamard_matrix(size, device)
        except ImportError:
            pass  # fall through to random
    return _make_random_orthogonal(size, device)


# ── R1 ───────────────────────────────────────────────────────────────────────

def train_r1_with_history(
    acts_single_layer,        # np.ndarray or Tensor (N, hidden_size) for ONE layer
    hidden_size: int,
    loss_fn_name: str,
    lr: float             = LR_LOSS,
    momentum: float       = MOMENTUM,
    epochs: int           = EPOCHS_LOSS,
    batch_size: int       = BATCH_SIZE,
    cos_lr: bool          = False,
    optim_name: str       = 'sgd',
    init_mode: str        = 'random',     # 'hadamard' or 'random'
    accumulation_steps: int = 1,
    train_subset_size: float = 1.0,
):
    """
    Train R1 rotation matrix for a SINGLE LAYER's activations.

    Mirrors dartquant_v2.trainers.train_r1 for one layer:
      - Accepts a single tensor/array of shape (N, hidden_size)
      - Hadamard or random initialization via _make_init_matrix
      - SGD (default) or Adam optimizer
      - Optional CosineAnnealingLR scheduler
      - Gradient accumulation support
      - QR extraction at end: R1 = QR(R1.matrix)[0]

    Called per-layer by run_experiment1_r1_comparison so that each layer's
    rotation is optimised independently on that layer's own activation
    distribution (same approach as R2 / train_r2_with_history).

    Returns:
        Q       : np.ndarray (hidden_size, hidden_size), float32
        history : list[float]  per-epoch mean loss
    """
    loss_fn = _get_loss_fn(loss_fn_name)

    # ── Prepare data (single layer) ──────────────────────────────────────────
    if isinstance(acts_single_layer, torch.Tensor):
        all_acts = acts_single_layer.reshape(-1, hidden_size).float().cpu()
    else:
        arr      = np.nan_to_num(np.asarray(acts_single_layer), nan=0.0,
                                  posinf=65504, neginf=-65504)
        all_acts = torch.tensor(arr, dtype=torch.float32).reshape(-1, hidden_size)

    dataset = TensorDataset(all_acts)

    # ── Initialize R1  (mirrors train_r1 lines 210–212) ─────────────────────
    R1 = R1_QR(hidden_size).to(DEVICE)
    try:
        R1.matrix.data = _make_init_matrix(hidden_size, DEVICE, init_mode).float()
    except Exception:
        pass  # keep eye(hidden_size) from R1_QR.__init__

    # ── Optimizer  (mirrors _create_optimizer) ───────────────────────────────
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(R1.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(R1.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

    R1.train()
    history = []

    for epoch in range(epochs):
        # Subset sampling  (mirrors train_r1 lines 223–226)
        num_samples = max(1, int(len(dataset) * train_subset_size))
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        sub = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(sub, sampler=RandomSampler(sub), batch_size=batch_size)

        epoch_losses = []
        for batch_idx, (batch,) in enumerate(loader):
            batch   = batch.to(DEVICE).float().reshape(-1, hidden_size)
            outputs = R1(batch)
            loss    = loss_fn(outputs) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_losses.append(loss.item() * accumulation_steps)

        if scheduler:
            scheduler.step()
        history.append(float(np.mean(epoch_losses)))

    # ── Extract final rotation  (mirrors train_r1 line 252) ──────────────────
    R1.eval()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(R1.matrix, mode='complete')
    return Q.detach().cpu().float().numpy(), history


# ── R2 ───────────────────────────────────────────────────────────────────────

def train_r2_with_history(
    acts_dict,                # dict {name: Tensor} from o_proj inputs, all layers
    arch_wrapper: _ArchWrapper,
    loss_fn_name: str,
    lr: float             = LR_LOSS,
    momentum: float       = MOMENTUM,
    epochs: int           = EPOCHS_LOSS,
    batch_size: int       = BATCH_SIZE * 2,   # pipeline default: 128
    cos_lr: bool          = False,
    optim_name: str       = 'sgd',
    accumulation_steps: int = 2,              # pipeline default: 2
):
    """
    Train per-head R2 rotation for each layer.

    Exactly mirrors dartquant_v2.trainers.train_r2_all_layers:
      - Processes each layer's activations independently
      - R2_Per_Head module with head_dim × head_dim rotation per KV-head group
      - Same optimizer / scheduler defaults
      - R2 = QR(R2.matrix)[0] per layer

    Extra: returns per-layer per-epoch history for plotting.

    Returns:
        R2_dict  : dict {"model.layers.{i}.self_attn.R2": Tensor(kv_heads, head_dim, head_dim)}
        histories: dict {layer_id: list[float]}
    """
    loss_fn  = _get_loss_fn(loss_fn_name)
    hidden   = arch_wrapper.hidden_size
    n_heads  = arch_wrapper.num_heads
    kv_heads = arch_wrapper.kv_heads
    head_dim = arch_wrapper.head_dim

    # ── Organise activations by layer index (mirrors train_r2_all_layers lines 292–299)
    acts_per_layer: dict = {}
    for name, acts in acts_dict.items():
        lid = arch_wrapper.layer_index_from_name(name)
        if lid >= 0:
            acts_per_layer[lid] = acts.float()

    R2_dict  = {}
    histories = {}

    for layer_id in sorted(acts_per_layer.keys()):
        acts   = acts_per_layer[layer_id]
        loader = DataLoader(TensorDataset(acts), batch_size=batch_size, shuffle=True)

        # ── Initialize R2  (mirrors train_r2_all_layers lines 305–308)
        R2 = R2_Per_Head(hidden, n_heads, kv_heads).to(DEVICE)
        try:
            init_data = _make_random_orthogonal(head_dim, DEVICE).float()
            R2.matrix.data = init_data.unsqueeze(0).repeat(kv_heads, 1, 1)
        except Exception:
            pass

        # ── Optimizer
        if optim_name == 'sgd':
            optimizer = torch.optim.SGD(R2.parameters(), lr=lr, momentum=momentum)
        else:
            optimizer = torch.optim.Adam(R2.parameters(), lr=lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

        R2.train()
        layer_hist = []

        for epoch in range(epochs):
            epoch_losses = []
            for batch_idx, (batch,) in enumerate(loader):
                batch   = batch.to(DEVICE).float()
                outputs = R2(batch)
                loss    = loss_fn(outputs) / accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_losses.append(loss.item() * accumulation_steps)

            if scheduler:
                scheduler.step()
            layer_hist.append(float(np.mean(epoch_losses)))

        # ── Extract rotation (mirrors train_r2_all_layers line 340)
        R2.eval()
        R2_dict[f"model.layers.{layer_id}.self_attn.R2"] = R2.rotate.data.detach().cpu()
        histories[layer_id] = layer_hist

    return R2_dict, histories


# ── Butterfly (R3 / R4) ──────────────────────────────────────────────────────

def _init_butterfly_from_hadamard(butterfly: nn.Module, dim: int,
                                   n_steps: int = 300, lr: float = 0.05):
    """Warm-start butterfly angles by fitting to a random Hadamard matrix.

    Mirrors dartquant_v2.trainers._init_butterfly_from_hadamard:
      H = WHT_normalized * diag(D),  D = random ±1 diagonal
      Minimise ||B(θ) - H||_F with Adam for n_steps steps.

    Only applies to power-of-2 dims (ButterflyRotation with .angles attribute).
    """
    if not (dim > 0 and (dim & (dim - 1)) == 0):
        return  # ButterflyFactored: skip
    if not hasattr(butterfly, 'angles'):
        return

    # ── Build random Hadamard target ──────────────────────────────────────────
    try:
        from hadamard_utils import random_hadamard_matrix
        H = random_hadamard_matrix(dim, DEVICE).float()
    except (ImportError, Exception):
        # Pure PyTorch fallback: WHT_normalized * random column signs
        D = (torch.randint(0, 2, (dim,)) * 2 - 1).float().to(DEVICE)
        x = torch.eye(dim, device=DEVICE)
        h = 1
        while h < dim:
            x = x.reshape(dim, dim // (2 * h), 2, h)
            a = x[:, :, 0, :].contiguous()
            b = x[:, :, 1, :].contiguous()
            x = torch.stack([a + b, a - b], dim=2).reshape(dim, dim)
            h *= 2
        H = (x / math.sqrt(dim)) * D.unsqueeze(0)

    H = H.detach()
    I_mat = torch.eye(dim, device=DEVICE, dtype=torch.float32)
    init_opt = torch.optim.Adam(butterfly.parameters(), lr=lr)
    butterfly.train()
    for _ in range(n_steps):
        init_opt.zero_grad()
        loss = ((butterfly.forward(I_mat) - H) ** 2).sum()
        loss.backward()
        init_opt.step()

    with torch.no_grad():
        fit_loss = ((butterfly.forward(I_mat) - H) ** 2).sum().item()
    print(f"    Butterfly Hadamard init (dim={dim}): fit loss = {fit_loss:.4f}")


def train_butterfly_with_history(
    acts_np: np.ndarray,
    dim: int,
    loss_fn_name: str,
    lr: float             = LR_BF,
    momentum: float       = MOMENTUM,
    epochs: int           = EPOCHS_BF,
    batch_size: int       = BATCH_SIZE,
    cos_lr: bool          = True,          # pipeline default: True for butterfly
    optim_name: str       = 'sgd',
    k_factor_mode: str    = 'latent',
    # ── Flaw 1 fix: quantization reconstruction loss (mirrors trainers.train_butterfly) ──
    quantizer_type: str   = 'none',        # 'int4', 'nf4', or 'none'
    lambda_recon: float   = 0.1,           # weight for L_recon (pipeline default)
    quant_block_size: int = 64,            # per-block quantizer block size
    weight_matrices       = None,          # Tensor (K, out_dim, dim) for R4 Eq 17 loss
    weight_quantizer_type: str = 'none',   # quantizer for weight path in Eq 17
):
    """
    Train Butterfly Givens rotation for R3 or R4.

    Exactly mirrors dartquant_v2.trainers.train_butterfly, including the
    optional quantization reconstruction loss (Flaw 1 fix):

    **Weight-aware mode** (weight_matrices provided → R4):
        L_recon = ||W·x − Q(W·B^T) · Q(B·x)||²_F   (paper Eq 17)
    This jointly minimises activation + weight quantization error, preventing
    the optimizer from driving down distribution loss at the expense of
    catastrophic condition-number blow-up in W·B^T.

    **Activation-only mode** (weight_matrices=None → R3):
        L_recon = ||X_in − B^T · Q(B·X_in)||²
    Straight-through estimator (STE) ensures gradients flow through Q(·).

    Total loss:  L = L_dist + lambda_recon * L_recon

    Returns:
        butterfly : trained ButterflyRotation or ButterflyFactored (eval mode)
        history   : list[float] per-epoch mean total loss
    """
    loss_fn = _get_loss_fn(loss_fn_name)
    acts_np = np.nan_to_num(acts_np, nan=0.0, posinf=65504, neginf=-65504)
    data    = torch.tensor(acts_np, dtype=torch.float32).reshape(-1, dim)

    loader = DataLoader(TensorDataset(data.to(DEVICE)),
                        batch_size=batch_size, shuffle=True)

    # ── Choose butterfly type  (mirrors train_butterfly lines 471–474) ───────
    is_pow2   = dim > 0 and (dim & (dim - 1)) == 0
    butterfly = (ButterflyRotation(dim) if is_pow2
                 else ButterflyFactored(dim, k_factor_mode=k_factor_mode)).to(DEVICE)

    # ── Warm-start from random Hadamard (mirrors DartQuant R3/R4 baseline) ───
    _init_butterfly_from_hadamard(butterfly, dim)

    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(butterfly.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(butterfly.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

    # ── Build fake quantizers for reconstruction loss (Flaw 1 fix) ───────────
    def _build_fq(qt):
        if qt == 'int4':
            return INT4FakeQuantizer(block_size=quant_block_size).to(DEVICE)
        elif qt == 'nf4':
            return NF4FakeQuantizer(block_size=quant_block_size).to(DEVICE)
        return None

    act_fq    = _build_fq(quantizer_type)
    weight_fq = _build_fq(weight_quantizer_type) if weight_matrices is not None else None

    use_weight_recon = (weight_matrices is not None and
                        act_fq is not None and weight_fq is not None and
                        lambda_recon > 0.0)
    use_act_recon    = (not use_weight_recon and act_fq is not None and
                        lambda_recon > 0.0)

    if use_weight_recon:
        W_bank = weight_matrices.float().to(DEVICE)   # (K, out_dim, dim)
        n_w    = W_bank.shape[0]

    import torch.nn.functional as _F

    butterfly.train()
    history = []

    for epoch in range(epochs):
        epoch_losses = []
        for (batch,) in loader:
            batch   = batch.to(DEVICE)
            outputs = butterfly(batch)   # B·x

            # ── Distribution loss (always present) ──────────────────────────
            dist_loss = loss_fn(outputs)

            # ── Reconstruction loss (Flaw 1 fix) ────────────────────────────
            if use_weight_recon:
                # Paper Eq 17: joint weight + activation loss
                # L_recon = ||W·x − Q(W·B^T)·Q(B·x)||²
                w_idx = torch.randint(0, n_w, (1,)).item()
                W     = W_bank[w_idx]          # (out_dim, dim)
                # STE for activation path
                with torch.no_grad():
                    Bx_q = act_fq(outputs)
                Bx_ste = outputs + (Bx_q - outputs).detach()
                # Rotate weight rows by B^T and STE-quantise
                W_rot = butterfly.inverse_forward(W)   # (out_dim, dim)
                with torch.no_grad():
                    WBt_q = weight_fq(W_rot)
                WBt_ste = W_rot + (WBt_q - W_rot).detach()
                y_true  = batch @ W.T             # (B, out_dim)
                y_hat   = Bx_ste @ WBt_ste.T      # (B, out_dim)
                recon   = _F.mse_loss(y_true, y_hat)
                loss    = dist_loss + lambda_recon * recon
            elif use_act_recon:
                # Activation-only reconstruction: L_recon = ||X - B^T·Q(B·x)||²
                with torch.no_grad():
                    x_q = act_fq(outputs)
                x_ste   = outputs + (x_q - outputs).detach()  # STE
                x_recon = butterfly.inverse_forward(x_ste)
                recon   = _F.mse_loss(batch, x_recon)
                loss    = dist_loss + lambda_recon * recon
            else:
                loss = dist_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_losses.append(loss.item())

        if scheduler:
            scheduler.step()
        history.append(float(np.mean(epoch_losses)))

    butterfly.eval()
    return butterfly, history


# ── Hadamard (Fast Walsh-Hadamard Transform) ─────────────────────────────────

def apply_hadamard_fwht(acts_np: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Apply normalized random Hadamard transform (FWHT × random diagonal signs).
    Pure numpy – no external dependency.  Requires dim to be a power of 2.
    """
    N, D = acts_np.shape
    assert D > 0 and (D & (D - 1)) == 0, f"dim must be power of 2, got {D}"
    rng   = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=D)
    x     = acts_np.astype(np.float64) * signs[np.newaxis, :]
    h = 1
    while h < D:
        nb  = D // (2 * h)
        xv  = x.reshape(N, nb, 2, h)
        a   = xv[:, :, 0, :].copy()
        b   = xv[:, :, 1, :].copy()
        xv[:, :, 0, :] = a + b
        xv[:, :, 1, :] = a - b
        x   = xv.reshape(N, D)
        h  *= 2
    return (x / math.sqrt(D)).astype(np.float32)


def compute_hadamard_loss(acts_np: np.ndarray, loss_fn_name: str) -> float:
    """Evaluate loss after Hadamard transform (static baseline)."""
    had = apply_hadamard_fwht(acts_np)
    with torch.no_grad():
        t = torch.tensor(had, dtype=torch.float32, device=DEVICE)
        return _get_loss_fn(loss_fn_name)(t).item()


def apply_butterfly_to_acts(butterfly: nn.Module,
                             acts_np: np.ndarray) -> np.ndarray:
    """Apply trained butterfly to numpy activations; returns numpy."""
    butterfly.eval()
    with torch.no_grad():
        t = torch.tensor(acts_np, dtype=torch.float32, device=DEVICE)
        return butterfly(t).cpu().numpy().astype(np.float32)


# ============================================================================
# SECTION 3 – METRIC UTILITIES
# ============================================================================

def compute_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis: Normal=0, Uniform≈-1.2, Laplacian=3."""
    xf = x.flatten()
    if len(xf) < 4:
        return float('nan')
    mu, sigma = np.mean(xf), np.std(xf)
    if sigma < 1e-9:
        return float('nan')
    return float(np.mean(((xf - mu) / sigma) ** 4) - 3.0)


def compute_per_dim_variance(acts: np.ndarray) -> np.ndarray:
    """Per-dimension variance. acts: (N, D) → (D,)."""
    return np.var(acts, axis=0, dtype=np.float64).astype(np.float32)


def _clip_range(x: np.ndarray, pct: float = 99.5):
    lo, hi = np.percentile(x.flatten(), [100 - pct, pct])
    return float(lo), float(hi)


def compute_uniform_pdf(x_range: np.ndarray, ref_acts: np.ndarray) -> np.ndarray:
    """Ideal Uniform[-b, b] PDF; b = sqrt(3) * RMS(ref_acts)."""
    rms = math.sqrt(max(float(np.mean(ref_acts.astype(np.float64) ** 2)), 1e-12))
    b   = math.sqrt(3) * rms
    return np.where(np.abs(x_range) <= b, 0.5 / b, 0.0).astype(np.float32)


def compute_gaussian_pdf(x_range: np.ndarray, ref_acts: np.ndarray) -> np.ndarray:
    """Ideal Gaussian N(0, sigma^2) PDF; sigma = sqrt(mean(x^2))."""
    sigma = math.sqrt(max(float(np.mean(ref_acts.astype(np.float64) ** 2)), 1e-12))
    return (1.0 / (sigma * math.sqrt(2 * math.pi)) *
            np.exp(-0.5 * (x_range / sigma) ** 2)).astype(np.float32)


# ============================================================================
# SECTION 4 – PLOTTING FUNCTIONS
# ============================================================================

_COLORS = {
    'original':  'slategray',
    'whip':      'darkorange',
    'swd_unif':  'steelblue',
    'swd_gauss': 'mediumorchid',
    'kl_unif':   'steelblue',
    'kl_gauss':  'mediumorchid',
    'hadamard':  'tomato',
    'butterfly': 'steelblue',
}

_LABELS = {
    'whip':      'After Whip',
    'swd_unif':  'After SWD-Unif',
    'swd_gauss': 'After SWD-Gauss',
}


def plot_loss_distribution(original_acts: np.ndarray,
                            rotated: dict,   # {loss_name: 1D-flat np.ndarray}
                            layer_idx: int,
                            model_name: str,
                            save_path: str):
    """
    4-panel histogram: Original | Whip | SWD_Unif (+ Uniform target) | SWD_Gauss (+ Gaussian target)
    """
    orig_flat = original_acts.flatten()
    lo, hi    = _clip_range(orig_flat)
    x_range   = np.linspace(lo, hi, 600)
    bins      = 100

    panels = [
        ('Original',       orig_flat,                           'slategray', None,                  None),
        ('After Whip',     rotated.get('whip', orig_flat),      'darkorange', None,                 None),
        ('After SWD-Unif', rotated.get('swd_unif', orig_flat),  'steelblue',
         compute_uniform_pdf(x_range, original_acts), 'Target Uniform'),
        ('After SWD-Gauss', rotated.get('swd_gauss', orig_flat), 'mediumorchid',
         compute_gaussian_pdf(x_range, original_acts), 'Target Gaussian'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    csv_rows  = []
    for ax, (title, data, color, pdf, pdf_lbl) in zip(axes, panels):
        d = np.clip(data.flatten(), lo, hi)
        ax.hist(d, bins=bins, density=True, color=color,
                alpha=0.72, log=True, edgecolor='none', label=title)

        if title != 'Original':
            ax.hist(np.clip(orig_flat, lo, hi), bins=bins, density=True,
                    color='slategray', alpha=0.18, log=True, edgecolor='none')

        if pdf is not None:
            ax.plot(x_range, np.maximum(pdf, 1e-6),
                    color='black', lw=2, ls='--', label=pdf_lbl)

        k   = compute_kurtosis(data)
        rms = math.sqrt(max(float(np.mean(data.astype(np.float64) ** 2)), 1e-12))
        std = float(data.std())
        ax.text(0.97, 0.97, f'Kurt={k:.2f}\nRMS={rms:.3f}\nStd={std:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        csv_rows.append([title, layer_idx, k, rms, std])

        ax.set_title(f'{title}\n(Layer {layer_idx})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Activation Value', fontsize=9)
        ax.set_ylabel('Log Density', fontsize=9)
        ax.set_xlim(lo, hi)
        ax.grid(True, alpha=0.25)
        if pdf is not None:
            ax.legend(fontsize=8)

    model_short = model_name.split('/')[-1]
    fig.suptitle(f'{model_short} – R1 Loss Function Comparison (all-layer collection, Layer {layer_idx} shown)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {os.path.basename(save_path)}')

    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['panel', 'layer_idx', 'kurtosis', 'rms', 'std'])
        writer.writerows(csv_rows)
    print(f'    Saved: {os.path.basename(csv_path)}')


def plot_loss_training_curves(histories: dict,   # {loss_name: {layer_idx: [float]}}
                               layer_indices: list,
                               model_name: str,
                               save_path: str):
    """One subplot per layer; 3 lines: Whip / SWD-Unif / SWD-Gauss."""
    n = len(layer_indices)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]

    styles = {
        'whip':      ('darkorange',   '-',  'Whip'),
        'swd_unif':  ('steelblue',    '--', 'SWD-Unif'),
        'swd_gauss': ('mediumorchid', ':',  'SWD-Gauss'),
    }

    for ax, layer_idx in zip(axes, layer_indices):
        for loss_name, (color, ls, lbl) in styles.items():
            h = histories.get(loss_name, {}).get(layer_idx)
            if h:
                ax.plot(range(1, len(h) + 1), h, color=color, ls=ls, lw=2, label=lbl)
        ax.set_title(f'Layer {layer_idx} (all-layer R1)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    model_short = model_name.split('/')[-1]
    fig.suptitle(f'{model_short} – R1 Training Loss Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {os.path.basename(save_path)}')

    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_idx', 'loss_name', 'epoch', 'loss_value'])
        for loss_name in styles:
            for layer_idx in layer_indices:
                h = histories.get(loss_name, {}).get(layer_idx)
                if h:
                    for epoch, val in enumerate(h, 1):
                        writer.writerow([layer_idx, loss_name, epoch, val])
    print(f'    Saved: {os.path.basename(csv_path)}')


def plot_r2_training_curves(r2_histories: dict,   # {layer_id: list[float]}
                              loss_fn_name: str,
                              model_name: str,
                              save_path: str):
    """R2 per-layer training loss curves."""
    layer_ids = sorted(r2_histories.keys())
    n = len(layer_ids)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, lid in zip(axes, layer_ids):
        h = r2_histories[lid]
        ax.plot(range(1, len(h) + 1), h, color='steelblue', lw=2)
        ax.set_title(f'Layer {lid} R2', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.grid(True, alpha=0.3)
        if h:
            ax.text(0.97, 0.97, f'Final={h[-1]:.4f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    model_short = model_name.split('/')[-1]
    fig.suptitle(f'{model_short} – R2 Per-Head Training Curves [{loss_fn_name}]',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {os.path.basename(save_path)}')

    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_id', 'loss_fn', 'epoch', 'loss_value'])
        for lid in layer_ids:
            for epoch, val in enumerate(r2_histories[lid], 1):
                writer.writerow([lid, loss_fn_name, epoch, val])
    print(f'    Saved: {os.path.basename(csv_path)}')


def plot_butterfly_distribution(original_flat: np.ndarray,
                                 had_flat: np.ndarray,
                                 bf_flat: np.ndarray,
                                 had_loss: float,
                                 bf_loss: float,
                                 rotation_label: str,
                                 layer_idx: int,
                                 loss_fn_name: str,
                                 model_name: str,
                                 save_path: str):
    """3-panel: Original | Hadamard baseline | Butterfly (trained)."""
    lo, hi = _clip_range(original_flat)
    bins   = 100

    panels = [
        ('Original',                     original_flat, 'slategray', None),
        (f'Hadamard\n(loss={had_loss:.4f})', had_flat, 'tomato',    had_loss),
        (f'Butterfly ({loss_fn_name})\n(loss={bf_loss:.4f})', bf_flat, 'steelblue', bf_loss),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    for ax, (title, data, color, _) in zip(axes, panels):
        d = np.clip(data.flatten(), lo, hi)
        ax.hist(d, bins=bins, density=True, color=color, alpha=0.72,
                log=True, edgecolor='none')
        if title != 'Original':
            ax.hist(np.clip(original_flat, lo, hi), bins=bins, density=True,
                    color='slategray', alpha=0.18, log=True, edgecolor='none')
        k = compute_kurtosis(data)
        ax.text(0.97, 0.97, f'Kurt={k:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Activation Value', fontsize=9)
        ax.set_ylabel('Log Density', fontsize=9)
        ax.set_xlim(lo, hi)
        ax.grid(True, alpha=0.25)

    model_short = model_name.split('/')[-1]
    suffix = f"(all-layer collection, {rotation_label})"
    fig.suptitle(f'{model_short} – Butterfly vs Hadamard {suffix}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {os.path.basename(save_path)}')

    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['panel', 'rotation_label', 'loss_fn_name', 'loss_value', 'kurtosis'])
        for pname, pdata, ploss in zip(
                ['original', 'hadamard', 'butterfly'],
                [original_flat, had_flat, bf_flat],
                [None, had_loss, bf_loss]):
            k = compute_kurtosis(pdata)
            writer.writerow([pname, rotation_label, loss_fn_name,
                             '' if ploss is None else ploss, k])
    print(f'    Saved: {os.path.basename(csv_path)}')


def plot_variance_uniformity(var_orig: np.ndarray,
                              var_had: np.ndarray,
                              var_bf: np.ndarray,
                              layer_idx: int,
                              model_name: str,
                              save_path: str):
    """R3: Per-dimension variance bar chart."""
    D     = len(var_orig)
    x     = np.arange(D)
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(12, D // 2), 5))
    ax.bar(x - width, var_orig, width, label='Original', color='slategray', alpha=0.8)
    ax.bar(x,         var_had,  width, label='Hadamard', color='tomato',    alpha=0.8)
    ax.bar(x + width, var_bf,   width, label='Butterfly', color='steelblue', alpha=0.8)

    ideal = float(np.mean(var_orig))
    ax.axhline(ideal, color='crimson', ls='--', lw=1.5,
               label=f'Ideal (mean var={ideal:.4f})')

    ax.set_xlabel('Head Dimension Index (0 = highest RoPE frequency)', fontsize=11)
    ax.set_ylabel('Per-Dimension Variance', fontsize=11)
    model_short = model_name.split('/')[-1]
    ax.set_title(
        f'{model_short} – R3 Per-Dimension Variance (all-layer collection)\n'
        f'Lower spread = more uniform = better for quantisation',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    step = max(1, D // 16)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(x[::step])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {os.path.basename(save_path)}')

    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dim_index', 'var_original', 'var_hadamard', 'var_butterfly'])
        for i, (vo, vh, vb) in enumerate(zip(var_orig, var_had, var_bf)):
            writer.writerow([i, float(vo), float(vh), float(vb)])
    print(f'    Saved: {os.path.basename(csv_path)}')


def plot_butterfly_training_curves(bf_histories: dict,   # {loss_name: {layer_idx: [float]}}
                                    had_losses:   dict,   # {loss_name: {layer_idx: float}}
                                    rotation_label: str,
                                    layer_indices: list,
                                    model_name: str,
                                    save_path: str):
    """
    One subplot per representative label.
    Butterfly SWD-Unif (solid) and SWD-Gauss (dashed) vs Hadamard baselines.
    """
    n = len(layer_indices)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]

    bf_styles = {
        'kl_unif':  ('steelblue',    '-',  'Butterfly KL-Unif'),
        'kl_gauss': ('mediumorchid', '--', 'Butterfly KL-Gauss'),
        # Legacy SWD keys (kept for backward compatibility if called with old dicts)
        'swd_unif':  ('steelblue',    '-',  'Butterfly SWD-Unif'),
        'swd_gauss': ('mediumorchid', '--', 'Butterfly SWD-Gauss'),
    }
    had_colors = {
        'kl_unif':   'deepskyblue',
        'kl_gauss':  'violet',
        'swd_unif':  'deepskyblue',
        'swd_gauss': 'violet',
    }

    for ax, layer_idx in zip(axes, layer_indices):
        for loss_name, (color, ls, lbl) in bf_styles.items():
            h = bf_histories.get(loss_name, {}).get(layer_idx)
            if h:
                ax.plot(range(1, len(h) + 1), h, color=color, ls=ls, lw=2, label=lbl)
            hv = had_losses.get(loss_name, {}).get(layer_idx)
            if hv is not None and not math.isinf(hv):
                suffix = loss_name.split('_')[-1].capitalize()
                ax.axhline(hv, color=had_colors[loss_name], ls=':', lw=2,
                           label=f'Hadamard {suffix} ({hv:.4f})')

        ax.set_title(f'{rotation_label} Layer {layer_idx} (KL divergence)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    model_short = model_name.split('/')[-1]
    fig.suptitle(
        f'{model_short} – Butterfly vs Hadamard Curves ({rotation_label})',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {os.path.basename(save_path)}')

    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer_idx', 'loss_name', 'type', 'epoch', 'loss_value'])
        for loss_name in bf_styles:
            for layer_idx in layer_indices:
                h = bf_histories.get(loss_name, {}).get(layer_idx)
                if h:
                    for epoch, val in enumerate(h, 1):
                        writer.writerow([layer_idx, loss_name, 'butterfly', epoch, val])
                hv = had_losses.get(loss_name, {}).get(layer_idx)
                if hv is not None and not math.isinf(hv):
                    writer.writerow([layer_idx, loss_name, 'hadamard_baseline', '', hv])
    print(f'    Saved: {os.path.basename(csv_path)}')


# ============================================================================
# SECTION 5 – EXPERIMENT RUNNERS
# ============================================================================

def run_experiment1_r1_comparison(model, arch_wrapper: _ArchWrapper,
                                   calib_data, layer_indices,
                                   model_name, model_clean):
    """
    Experiment 1: R1 loss function comparison — per-layer independent training.

    Collection mirrors pipeline.py Step 4:
      targets = [up_proj, q_proj] for EVERY layer (single forward pass)

    Training (per-layer, like R2):
      activations are grouped by layer index; up_proj + q_proj inputs for the
      same layer are concatenated, then each layer trains its own R1_QR
      independently.  No cross-layer gradient mixing.

    Three loss functions are compared (Whip / SWD-Unif / SWD-Gauss), each
    producing a separate per-layer R1 for visualization.

    Plots:
      {model_clean}_loss_dist_R1.png      – 4-panel histogram (representative layer)
      {model_clean}_loss_curves_R1.png    – per-layer training convergence
    """
    print("\n  [Exp 1] R1 Loss Function Comparison — per-layer independent training")
    dev = _model_device(model)

    # ── Collect (mirrors pipeline Step 4 — single forward pass) ──────────────
    targets = arch_wrapper.r1_collection_targets()
    print(f"  Collecting R1 activations: {len(targets)} targets "
          f"(all {arch_wrapper.num_layers} layers × up_proj+q_proj) ...")
    acts_dict = collect_pipeline_activations(model, calib_data, targets, dev)
    if not acts_dict:
        print("  Skipping Exp 1: no activations collected")
        return
    total_rows = sum(v.shape[0] for v in acts_dict.values())
    print(f"  Collected {len(acts_dict)} tensors, total rows: {total_rows}")

    hidden_size = arch_wrapper.hidden_size

    # ── Group activations by layer (up_proj + q_proj combined per layer) ─────
    acts_per_layer: dict = {}
    for name, acts in acts_dict.items():
        lid = arch_wrapper.layer_index_from_name(name)
        if lid < 0:
            continue
        if lid not in acts_per_layer:
            acts_per_layer[lid] = []
        acts_per_layer[lid].append(acts.float().reshape(-1, hidden_size))
    for lid in acts_per_layer:
        acts_per_layer[lid] = torch.cat(acts_per_layer[lid], dim=0)

    available_layers = sorted(acts_per_layer.keys())
    print(f"  Grouped into {len(available_layers)} layers "
          f"({sum(v.shape[0] for v in acts_per_layer.values())} total rows)")

    # Representative layer for distribution plot (first in layer_indices)
    rep = layer_indices[0]
    if rep not in acts_per_layer:
        rep = available_layers[0]
    rep_acts_2d = acts_per_layer[rep].numpy().astype(np.float32)

    all_histories = {k: {} for k in ('whip', 'swd_unif', 'swd_gauss')}
    rotated_flat  = {}

    # ── Train R1 per layer for each loss function ─────────────────────────────
    for loss_name in ('whip', 'swd_unif', 'swd_gauss'):
        print(f"\n  Training R1 [{loss_name}] — per-layer independent ...")
        for layer_idx in layer_indices:
            if layer_idx not in acts_per_layer:
                print(f"    Layer {layer_idx}: no activations, skipping")
                continue
            acts_l = acts_per_layer[layer_idx].numpy().astype(np.float32)
            print(f"    Layer {layer_idx}: {acts_l.shape[0]} rows ...")
            Q_l, history = train_r1_with_history(
                acts_l, hidden_size, loss_name,
                init_mode='random', accumulation_steps=1,
            )
            all_histories[loss_name][layer_idx] = history
            # Store representative layer's rotated acts for distribution plot
            if layer_idx == rep:
                rotated_flat[loss_name] = (rep_acts_2d @ Q_l).flatten()

        # Fallback: if rep layer skipped, store zeros placeholder
        if loss_name not in rotated_flat:
            rotated_flat[loss_name] = rep_acts_2d.flatten()

        final_losses = {lid: all_histories[loss_name][lid][-1]
                        for lid in layer_indices if lid in all_histories[loss_name]}
        print(f"    [{loss_name}] final losses: "
              + ", ".join(f"L{lid}={v:.4f}" for lid, v in final_losses.items()))

    # ── Distribution plot (representative layer) ──────────────────────────────
    save_path = os.path.join(PLOT_DIR, f"{model_clean}_loss_dist_R1.png")
    plot_loss_distribution(rep_acts_2d, rotated_flat, rep, model_name, save_path)

    # ── Training curve plot (one subplot per selected layer) ──────────────────
    if any(all_histories[k] for k in all_histories):
        save_path = os.path.join(PLOT_DIR, f"{model_clean}_loss_curves_R1.png")
        plot_loss_training_curves(all_histories, layer_indices, model_name, save_path)


def run_experiment2_r2(model, arch_wrapper: _ArchWrapper,
                        calib_data, layer_indices,
                        model_name, model_clean,
                        loss_fn_name: str = 'whip'):
    """
    Experiment 2: R2 per-head rotation — pipeline-style collection.

    Collection mirrors pipeline.py Step 6:
      targets = [o_proj] for EVERY layer
    Training mirrors dartquant_v2.trainers.train_r2_all_layers via
    train_r2_with_history.

    Plots:
      {model_clean}_r2_curves_{loss_fn_name}.png  – per-layer loss curves
    """
    print(f"\n  [Exp 2] R2 Per-Head Rotation [{loss_fn_name}] (all-layer pipeline collection)")
    dev = _model_device(model)

    # ── Collect (mirrors pipeline Step 6) ────────────────────────────────────
    targets = arch_wrapper.r2_collection_targets()
    print(f"  Collecting R2 activations: {len(targets)} targets "
          f"(all {arch_wrapper.num_layers} layers × o_proj) ...")
    acts_dict = collect_pipeline_activations(model, calib_data, targets, dev)
    if not acts_dict:
        print("  Skipping Exp 2: no activations collected")
        return

    total_rows = sum(v.shape[0] for v in acts_dict.values())
    print(f"  Collected {len(acts_dict)} R2 tensors, total rows: {total_rows}")
    print(f"  Training R2 [{loss_fn_name}] for {arch_wrapper.num_layers} layers ...")

    # ── Train R2 (mirrors train_r2_all_layers) ────────────────────────────────
    R2_dict, r2_histories = train_r2_with_history(
        acts_dict, arch_wrapper, loss_fn_name,
        accumulation_steps=2,   # pipeline default
    )

    print(f"  R2 trained for {len(R2_dict)} layers:")
    for layer_id in sorted(r2_histories.keys()):
        h = r2_histories[layer_id]
        if h:
            print(f"    Layer {layer_id}: init_loss={h[0]:.4f}, final_loss={h[-1]:.4f}")

    # ── Training curve plot (selected layers) ─────────────────────────────────
    sel_histories = {lid: r2_histories[lid]
                     for lid in layer_indices if lid in r2_histories}
    if sel_histories:
        save_path = os.path.join(PLOT_DIR, f"{model_clean}_r2_curves_{loss_fn_name}.png")
        plot_r2_training_curves(sel_histories, loss_fn_name, model_name, save_path)

    return R2_dict, r2_histories


def run_experiment3_butterfly(model, arch_wrapper: _ArchWrapper,
                               calib_data, layer_indices,
                               model_name, model_clean):
    """
    Experiment 3: Butterfly (R3 only) — per-layer independent training (Flaw 2 fix).

    R4 uses a fixed Random Hadamard transform (no butterfly training or comparison).

    Flaw 2 fix: Each transformer layer has a distinct activation covariance Σ_l.
    Training a single global butterfly on all-layer concatenated activations causes
    gradient cancellation (∇_θ ≈ Σ_l ∇L(B; Σ_l) → 0 when Σ_l differ in orientation),
    yielding a rotation matrix that is suboptimal for every individual layer.

    Fix: One forward pass collects all layers simultaneously; then each layer in
    layer_indices gets its own independent butterfly optimized on that layer only.

    Distribution loss uses KL divergence (not SWD):
      kl_unif  — Vasicek entropy maximisation → Uniform target (INT4)
      kl_gauss — Gram-Charlier moment matching → Gaussian target (NF4)

    R3 (Q+K proj outputs, dim=head_dim):
      Activation-only reconstruction loss (Flaw 1 fix), joint Q+K (Flaw 3 fix):
        L_total = L_KL(B·x) + λ · ||x − B^T · Q(B·x)||²

    Plots per selected layer:
      {model_clean}_butterfly_R3_layer{l}.png
      {model_clean}_butterfly_R3_curves_perlayer.png
      {model_clean}_R3_variance_layer{l}.png
    """
    print("\n  [Exp 3] Butterfly R3 — per-layer training (Flaw 2 fix) | R4 = Random Hadamard (fixed)")
    dev = _model_device(model)

    # ═══════════════════════════════════════════════════════════════════════
    # R3: Collect ALL layers (Q+K) in a single forward pass, then train per-layer.
    # Joint Q+K distribution preserved (Flaw 3 fix); each layer gets its own
    # butterfly trained only on that layer's head-vector distribution.
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  --- R3 (Q+K outputs, per-layer butterfly + act recon, dim=head_dim) ---")
    head_dim = arch_wrapper.head_dim
    print(f"  head_dim={head_dim}, num_heads={arch_wrapper.num_heads}")
    print("  Collecting R3 per-layer activations (Q+K outputs, 16 samples) ...")
    r3_per_layer = collect_r3_activations_per_layer(model, calib_data, arch_wrapper, dev)

    if not r3_per_layer:
        print("  Skipping R3: no activations collected")
        return

    bf_hist_r3  = {'kl_unif': {}, 'kl_gauss': {}}
    had_loss_r3 = {'kl_unif': {}, 'kl_gauss': {}}
    is_pow2_r3  = head_dim > 0 and (head_dim & (head_dim - 1)) == 0

    for layer_idx in layer_indices:
        if layer_idx not in r3_per_layer:
            print(f"  Layer {layer_idx}: R3 activations not found, skipping")
            continue

        acts_l = r3_per_layer[layer_idx]   # (N, head_dim)
        print(f"  Layer {layer_idx}: R3 acts={acts_l.shape}")

        # ── Hadamard baseline ───────────────────────────────────────────────
        if not is_pow2_r3:
            had_acts_l = acts_l.copy()
            hv_u3 = hv_g3 = float('inf')
            print(f"  Layer {layer_idx}: head_dim={head_dim} not pow-2, skip Hadamard")
        else:
            had_acts_l = apply_hadamard_fwht(acts_l)
            hv_u3      = compute_hadamard_loss(acts_l, 'kl_unif')
            hv_g3      = compute_hadamard_loss(acts_l, 'kl_gauss')
            print(f"  Layer {layer_idx}: Hadamard R3 kl_unif={hv_u3:.4f},"
                  f" kl_gauss={hv_g3:.4f}")
        had_loss_r3['kl_unif'][layer_idx]  = hv_u3
        had_loss_r3['kl_gauss'][layer_idx] = hv_g3

        # ── Per-layer butterfly [kl_unif] + activation-only recon (INT4) ───
        print(f"  Layer {layer_idx}: Training R3 butterfly [kl_unif + act recon INT4] ...")
        bf_u3, hist_u3 = train_butterfly_with_history(
            acts_l, head_dim, 'kl_unif',
            quantizer_type='int4', lambda_recon=0.1,
        )
        bf_acts_u3 = apply_butterfly_to_acts(bf_u3, acts_l)
        with torch.no_grad():
            bf_loss_u3 = _get_loss_fn('kl_unif')(
                torch.tensor(bf_acts_u3, device=DEVICE)).item()
        bf_hist_r3['kl_unif'][layer_idx] = hist_u3

        # ── Per-layer butterfly [kl_gauss] + activation-only recon (NF4) ───
        print(f"  Layer {layer_idx}: Training R3 butterfly [kl_gauss + act recon NF4] ...")
        bf_g3, hist_g3 = train_butterfly_with_history(
            acts_l, head_dim, 'kl_gauss',
            quantizer_type='nf4', lambda_recon=0.1,
        )
        bf_acts_g3 = apply_butterfly_to_acts(bf_g3, acts_l)
        with torch.no_grad():
            bf_loss_g3 = _get_loss_fn('kl_gauss')(
                torch.tensor(bf_acts_g3, device=DEVICE)).item()
        bf_hist_r3['kl_gauss'][layer_idx] = hist_g3

        print(f"  Layer {layer_idx}: R3 bf_kl_unif={bf_loss_u3:.4f} (had={hv_u3:.4f}), "
              f"bf_kl_gauss={bf_loss_g3:.4f} (had={hv_g3:.4f})")

        # ── Per-layer distribution plot ─────────────────────────────────────
        save_path = os.path.join(PLOT_DIR,
                                 f"{model_clean}_butterfly_R3_layer{layer_idx}.png")
        plot_butterfly_distribution(
            acts_l.flatten(), had_acts_l.flatten(), bf_acts_u3.flatten(),
            hv_u3, bf_loss_u3, 'R3', layer_idx, 'kl_unif', model_name, save_path
        )

        # ── Per-layer variance uniformity plot ─────────────────────────────
        var_orig = compute_per_dim_variance(acts_l)
        var_had  = compute_per_dim_variance(had_acts_l)
        var_bf   = compute_per_dim_variance(bf_acts_u3)
        save_path = os.path.join(PLOT_DIR,
                                 f"{model_clean}_R3_variance_layer{layer_idx}.png")
        plot_variance_uniformity(var_orig, var_had, var_bf, layer_idx, model_name, save_path)

    # ── Summary: per-layer training curves ─────────────────────────────────
    if any(bf_hist_r3[k] for k in bf_hist_r3):
        save_path = os.path.join(PLOT_DIR,
                                 f"{model_clean}_butterfly_R3_curves_perlayer.png")
        plot_butterfly_training_curves(
            bf_hist_r3, had_loss_r3, 'R3', layer_indices, model_name, save_path
        )


# ============================================================================
# SECTION 6 – PER-MODEL ORCHESTRATOR
# ============================================================================

def analyze_model(model_name: str):
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}")

    model = tokenizer = calib_data = None
    try:
        model, tokenizer = load_model_and_tokenizer(model_name)

        # Build architecture wrapper (uses dartquant_v2 arch registry)
        arch_wrapper = _ArchWrapper(model)
        print(f"  Architecture: {model.config.__class__.__name__}")
        print(f"  hidden={arch_wrapper.hidden_size}, "
              f"layers={arch_wrapper.num_layers}, "
              f"heads={arch_wrapper.num_heads}/{arch_wrapper.kv_heads}, "
              f"head_dim={arch_wrapper.head_dim}, "
              f"intermediate={arch_wrapper.intermediate_size}")

        layer_indices = get_layer_indices(arch_wrapper.num_layers)
        print(f"  Representative layers for plots: {layer_indices}")

        calib_data  = load_calibration_data(tokenizer)
        model_clean = model_name.replace('/', '_')

        # ── Experiment 1: R1 loss comparison (all-layer collection) ─────────
        run_experiment1_r1_comparison(
            model, arch_wrapper, calib_data, layer_indices, model_name, model_clean
        )

        # ── Experiment 2: R2 per-head rotation (all-layer collection) ───────
        run_experiment2_r2(
            model, arch_wrapper, calib_data, layer_indices, model_name, model_clean,
            loss_fn_name='whip'    # default pipeline loss for R2
        )

        # ── Experiment 3: Butterfly R3/R4 (all-layer collection) ────────────
        run_experiment3_butterfly(
            model, arch_wrapper, calib_data, layer_indices, model_name, model_clean
        )

        print(f"\n  Finished: {model_name}")

    except Exception as exc:
        import traceback
        print(f"  ERROR analyzing {model_name}: {exc}")
        traceback.print_exc()

    finally:
        for obj in (model, tokenizer, calib_data):
            try:
                del obj
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  GPU memory cleared")


# ============================================================================
# SECTION 7 – MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("DartQuant v2 – R1/R2/R3/R4 Rotation Matrix Visualization")
    print("Pipeline mode: all-layer collection, dartquant_v2.trainers training")
    print("=" * 80)
    print(f"HF Endpoint : {os.environ.get('HF_ENDPOINT', 'default')}")
    print(f"HF Token    : {'Set' if HF_TOKEN else 'Not set'}")
    print(f"Cache dir   : {CACHE_DIR}")
    print(f"Plot dir    : {PLOT_DIR}")
    print(f"Models      : {MODELS}")
    print(f"Epochs R1/R2: {EPOCHS_LOSS}  |  Epochs Butterfly: {EPOCHS_BF}")
    print(f"Max acts/hook: {MAX_ACTS}    |  Batch size: {BATCH_SIZE}")
    print(f"dartquant_v2 full imports: {_DQ_IMPORTS_OK}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    else:
        print("CUDA not available, using CPU (will be slow)")

    # Reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    for model_name in MODELS:
        analyze_model(model_name)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)
    print(f"Plots saved to: {PLOT_DIR}")
    plots = sorted(Path(PLOT_DIR).glob("*.png"))
    print(f"  {len(plots)} PNG files generated:")
    for p in plots:
        print(f"    {p.name}")


if __name__ == "__main__":
    main()
