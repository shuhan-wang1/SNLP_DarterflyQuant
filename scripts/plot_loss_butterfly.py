#!/usr/bin/env python3
"""
DartQuant v2 - Loss Function & Butterfly Rotation Visualization

Generates two sets of empirical experiment plots:

  Experiment 1: Loss function comparison (Whip vs SWD_Unif vs SWD_Gauss)
    - Activation distribution histograms before/after each rotation
    - Overlaid target distributions (Uniform / Gaussian)
    - Training loss convergence curves

  Experiment 2: Butterfly vs Hadamard for R3 (head_dim) and R4 (intermediate_size)
    - Activation distribution 3-panel comparison
    - Per-dimension variance uniformity bar chart (R3)
    - Training loss curves: Butterfly line vs Hadamard dashed baseline

Configuration mirrors scripts/stat_and_download.py exactly:
  same CACHE_DIR, HF_HOME, HF_ENDPOINT, model loading pattern, file naming.

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
    "meta-llama/Llama-3.2-1B-Instruct",
]

# ── Data / collection settings ──────────────────────────────────────────────
SEQ_LENGTH  = 2048    # token sequence length
NUM_SAMPLES = 64      # number of calibration sequences
BATCH_FWD   = 4       # sequences per forward pass during collection
MAX_ACTS    = 512     # max activation rows kept per layer (speed vs quality)
DTYPE       = torch.float16

# ── Training hyperparameters (lightweight for visualization) ─────────────────
LR_LOSS     = 1e-3    # learning rate for R1 loss comparison experiment
LR_BF       = 1e-3    # learning rate for Butterfly experiment
MOMENTUM    = 0.9
EPOCHS_LOSS = 10      # R1 epochs (matches pipeline default --r1_epochs 10)
EPOCHS_BF   = 20      # Butterfly epochs
BATCH_SIZE  = 64

# ============================================================================
# sys.path setup: make dartquant_v2 and DartQuant importable
# ============================================================================

_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "DartQuant", "fake_quant"))
sys.path.insert(0, os.path.join(_project_root, "DartQuant", "calibrater"))

# ── Import dartquant_v2 modules ──────────────────────────────────────────────
try:
    import dartquant_v2  # noqa: F401 – triggers arch/ registration
    from dartquant_v2.loss_functions import (
        calc_whip_loss, calc_swd_unif_loss, calc_swd_gauss_loss
    )
    from dartquant_v2.butterfly import ButterflyRotation, ButterflyFactored
    from dartquant_v2.trainers import R1_QR
    print("dartquant_v2 imported successfully")
except ImportError as e:
    print(f"Warning: dartquant_v2 import failed ({e}). Using inline fallbacks.")

    # ── Inline fallback implementations ────────────────────────────────────
    def calc_whip_loss(x):
        return torch.sum(torch.exp(-x.abs()), dim=-1, keepdim=True).mean()

    def calc_swd_unif_loss(x):
        xf = x.reshape(-1)
        xs, _ = torch.sort(xf)
        with torch.no_grad():
            rms = torch.sqrt(torch.mean(xf ** 2))
            b   = math.sqrt(3) * rms
            t   = torch.linspace(-b.item(), b.item(), len(xf), device=x.device)
        import torch.nn.functional as F
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

    class R1_QR(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.hidden_size = h
            self.matrix = nn.Parameter(torch.eye(h))
            self.rotate = None
        def forward(self, x):
            self.rotate, _ = torch.linalg.qr(self.matrix, mode='complete')
            return torch.matmul(x, self.rotate)

    class ButterflyRotation(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim
            self.num_layers = int(math.log2(dim))
            self.angles = nn.Parameter(torch.zeros(self.num_layers, dim // 2))
            self._pairs = []
            for l in range(self.num_layers):
                stride = 2 ** (l + 1)
                half   = 2 ** l
                i_idx  = [i for i in range(dim) if i % stride < half]
                j_idx  = [i + half for i in i_idx]
                self._pairs.append((
                    torch.tensor(i_idx), torch.tensor(j_idx)
                ))
        def forward(self, x):
            out = x.clone()
            for l, (i_idx, j_idx) in enumerate(self._pairs):
                i_idx = i_idx.to(x.device)
                j_idx = j_idx.to(x.device)
                th = self.angles[l]
                cos_t, sin_t = torch.cos(th), torch.sin(th)
                xi, xj = out[:, i_idx], out[:, j_idx]
                out[:, i_idx] = cos_t * xi + sin_t * xj
                out[:, j_idx] = -sin_t * xi + cos_t * xj
            return out
        def get_matrix(self):
            I = torch.eye(self.dim, dtype=torch.float64, device=self.angles.device)
            with torch.no_grad():
                return self.forward(I.unsqueeze(0)).squeeze(0).T.float()

    class ButterflyFactored(ButterflyRotation):
        pass  # simplified fallback

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

# ── Loss function lookup ─────────────────────────────────────────────────────
_LOSS_FNS = {
    'whip':      calc_whip_loss,
    'swd_unif':  calc_swd_unif_loss,
    'swd_gauss': calc_swd_gauss_loss,
}

# ============================================================================
# SECTION 1 – UTILITY FUNCTIONS
# ============================================================================

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
        data = torch.cat(chunks[:nsamples], dim=0)   # (nsamples, seq_length)
        print(f"  Calibration data shape: {data.shape}")
        return data
    except Exception as e:
        print(f"  WikiText-2 unavailable ({e}), using random fallback")
        vocab_size = getattr(tokenizer, 'vocab_size', 32000)
        return torch.randint(0, vocab_size, (nsamples, seq_length))


def get_layers(model):
    """Return the list of transformer decoder layers."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return list(model.model.layers)
    if hasattr(model, 'layers'):
        return list(model.layers)
    raise ValueError("Cannot locate decoder layers in model")


def get_layer_indices(num_layers: int):
    """First / middle / last layers – mirrors stat_and_download.py."""
    return [0, num_layers // 2, num_layers - 1]


def _model_device(model) -> torch.device:
    """Return device of first model parameter (handles device_map='auto')."""
    return next(model.parameters()).device


# ── Activation hook collector ─────────────────────────────────────────────────

class HookCollector:
    """Collect input or output activations via a single forward hook."""

    def __init__(self, capture: str = 'input', max_rows: int = MAX_ACTS):
        assert capture in ('input', 'output')
        self.capture  = capture
        self.max_rows = max_rows
        self._data    = []
        self._handle  = None

    def _hook_fn(self, module, inp, out):
        tensor = (inp[0] if isinstance(inp, tuple) else inp) \
                  if self.capture == 'input' else out
        if not isinstance(tensor, torch.Tensor):
            return
        flat = tensor.detach().float().cpu().reshape(-1, tensor.shape[-1])
        # Subsample per-call to limit memory usage
        if flat.shape[0] > 128:
            idx  = torch.randperm(flat.shape[0])[:128]
            flat = flat[idx]
        self._data.append(flat)

    def register(self, module: nn.Module):
        self._data   = []
        self._handle = module.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get(self) -> np.ndarray:
        if not self._data:
            return np.zeros((0, 1), dtype=np.float32)
        acts = torch.cat(self._data, dim=0)
        if acts.shape[0] > self.max_rows:
            idx  = torch.randperm(acts.shape[0])[:self.max_rows]
            acts = acts[idx]
        return acts.numpy().astype(np.float32)


def collect_activations(model, calib_data: torch.Tensor,
                        module: nn.Module, capture: str = 'input') -> np.ndarray:
    """
    Run forward passes to collect activations from a module.

    Returns:
        np.ndarray of shape (N, dim), dtype float32
    """
    collector = HookCollector(capture=capture, max_rows=MAX_ACTS)
    collector.register(module)

    dev = _model_device(model)
    with torch.no_grad():
        for i in range(0, calib_data.shape[0], BATCH_FWD):
            batch = calib_data[i:i + BATCH_FWD].to(dev)
            try:
                model(batch)
            except Exception:
                pass

    collector.remove()
    acts = collector.get()
    return acts


# ============================================================================
# SECTION 2 – TRAINING HELPERS (return loss history for plotting)
# ============================================================================

def train_r1_with_history(acts_np: np.ndarray, hidden_size: int,
                           loss_fn_name: str,
                           lr: float = LR_LOSS, momentum: float = MOMENTUM,
                           epochs: int = EPOCHS_LOSS,
                           batch_size: int = BATCH_SIZE):
    """
    Train R1 rotation (QR-Orth) with the given loss function.

    Returns:
        Q          : np.ndarray (hidden_size, hidden_size) – orthogonal rotation
        history    : list[float] – mean loss per epoch
    """
    acts_np = np.nan_to_num(acts_np, nan=0.0, posinf=65504, neginf=-65504)
    data    = torch.tensor(acts_np, dtype=torch.float32).reshape(-1, hidden_size)

    dataset = TensorDataset(data.to(DEVICE))
    loader  = DataLoader(dataset, sampler=RandomSampler(dataset),
                         batch_size=batch_size, drop_last=False)

    rot = R1_QR(hidden_size).to(DEVICE).float()
    opt = torch.optim.SGD(rot.parameters(), lr=lr, momentum=momentum)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

    loss_fn = _LOSS_FNS[loss_fn_name]
    rot.train()
    history = []

    for epoch in range(epochs):
        epoch_losses = []
        for (x,) in loader:
            out  = rot(x.float())
            loss = loss_fn(out)
            loss.backward()
            opt.step()
            opt.zero_grad()
            epoch_losses.append(loss.item())
        sch.step()
        history.append(float(np.mean(epoch_losses)))

    rot.eval()
    with torch.no_grad():
        Q, _ = torch.linalg.qr(rot.matrix, mode='complete')
    return Q.detach().cpu().numpy().astype(np.float32), history


def train_butterfly_with_history(acts_np: np.ndarray, dim: int,
                                  loss_fn_name: str,
                                  lr: float = LR_BF, momentum: float = MOMENTUM,
                                  epochs: int = EPOCHS_BF,
                                  batch_size: int = BATCH_SIZE):
    """
    Train Butterfly rotation for the given dimension and loss function.

    Returns:
        butterfly : trained ButterflyRotation or ButterflyFactored module
        history   : list[float] – mean loss per epoch
    """
    acts_np = np.nan_to_num(acts_np, nan=0.0, posinf=65504, neginf=-65504)
    data    = torch.tensor(acts_np, dtype=torch.float32).reshape(-1, dim)

    dataset = TensorDataset(data.to(DEVICE))
    loader  = DataLoader(dataset, sampler=RandomSampler(dataset),
                         batch_size=batch_size, drop_last=False)

    is_pow2  = dim > 0 and (dim & (dim - 1)) == 0
    butterfly = (ButterflyRotation(dim) if is_pow2
                 else ButterflyFactored(dim)).to(DEVICE)

    opt = torch.optim.SGD(butterfly.parameters(), lr=lr, momentum=momentum)
    sch = CosineAnnealingLR(opt, T_max=epochs, eta_min=0)

    loss_fn = _LOSS_FNS[loss_fn_name]
    butterfly.train()
    history = []

    for epoch in range(epochs):
        epoch_losses = []
        for (x,) in loader:
            out  = butterfly(x.float())
            loss = loss_fn(out)
            loss.backward()
            opt.step()
            opt.zero_grad()
            epoch_losses.append(loss.item())
        sch.step()
        history.append(float(np.mean(epoch_losses)))

    butterfly.eval()
    return butterfly, history


# ── Hadamard (Fast Walsh-Hadamard Transform) ──────────────────────────────────

def apply_hadamard_fwht(acts_np: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Apply normalized random Hadamard transform (FWHT × random diagonal signs).
    Pure numpy/python – no external dependency.
    Requires dim to be a power of 2.
    """
    N, D = acts_np.shape
    assert D > 0 and (D & (D - 1)) == 0, f"dim must be power of 2, got {D}"

    rng   = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=D)

    x = acts_np.astype(np.float64) * signs[np.newaxis, :]

    # Vectorised butterfly:  x.reshape(N, num_blocks, 2, h) → butterfly last two dims
    h = 1
    while h < D:
        nb  = D // (2 * h)
        xv  = x.reshape(N, nb, 2, h)
        a   = xv[:, :, 0, :].copy()
        b   = xv[:, :, 1, :].copy()
        xv[:, :, 0, :] = a + b
        xv[:, :, 1, :] = a - b
        x = xv.reshape(N, D)
        h *= 2

    return (x / math.sqrt(D)).astype(np.float32)


def compute_hadamard_loss(acts_np: np.ndarray, loss_fn_name: str) -> float:
    """Evaluate loss after Hadamard transform (static baseline)."""
    had = apply_hadamard_fwht(acts_np)
    with torch.no_grad():
        t = torch.tensor(had, dtype=torch.float32, device=DEVICE)
        return _LOSS_FNS[loss_fn_name](t).item()


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

    Args:
        original_acts : (N, D) raw activations (used for RMS-based target scaling)
        rotated       : dict with keys 'whip', 'swd_unif', 'swd_gauss'
                        each value is a 1-D flat np.ndarray of rotated values
        layer_idx     : layer index
        model_name    : full model name (used in title)
        save_path     : output PNG path
    """
    orig_flat = original_acts.flatten()
    lo, hi    = _clip_range(orig_flat)
    x_range   = np.linspace(lo, hi, 600)
    bins      = 100

    panels = [
        ('Original',     orig_flat,               'slategray', None,                  None),
        ('After Whip',   rotated.get('whip', orig_flat), 'darkorange', None,          None),
        ('After SWD-Unif', rotated.get('swd_unif', orig_flat), 'steelblue',
         compute_uniform_pdf(x_range, original_acts), 'Target Uniform'),
        ('After SWD-Gauss', rotated.get('swd_gauss', orig_flat), 'mediumorchid',
         compute_gaussian_pdf(x_range, original_acts), 'Target Gaussian'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    csv_rows = []
    for ax, (title, data, color, pdf, pdf_lbl) in zip(axes, panels):
        d = np.clip(data.flatten(), lo, hi)
        counts, _, _ = ax.hist(d, bins=bins, density=True, color=color,
                               alpha=0.72, log=True, edgecolor='none',
                               label=title)

        # Faint original overlay on rotated panels
        if title != 'Original':
            ax.hist(np.clip(orig_flat, lo, hi), bins=bins, density=True,
                    color='slategray', alpha=0.18, log=True, edgecolor='none')

        # Overlay target distribution
        if pdf is not None:
            # Scale to log-density; filter near-zero
            ax.plot(x_range, np.maximum(pdf, 1e-6),
                    color='black', lw=2, ls='--', label=pdf_lbl)

        # Statistics annotation
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
    fig.suptitle(f'{model_short} – Loss Function Comparison (Layer {layer_idx})',
                 fontsize=13, fontweight='bold')
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
    """
    One subplot per layer; 3 lines: Whip / SWD-Unif / SWD-Gauss.
    """
    n = len(layer_indices)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]

    styles = {
        'whip':      ('darkorange',    '-',   'Whip'),
        'swd_unif':  ('steelblue',     '--',  'SWD-Unif'),
        'swd_gauss': ('mediumorchid',  ':',   'SWD-Gauss'),
    }

    for ax, layer_idx in zip(axes, layer_indices):
        for loss_name, (color, ls, lbl) in styles.items():
            h = histories.get(loss_name, {}).get(layer_idx)
            if h:
                ax.plot(range(1, len(h) + 1), h, color=color, ls=ls, lw=2, label=lbl)
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
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


def plot_butterfly_distribution(original_flat: np.ndarray,
                                 had_flat: np.ndarray,
                                 bf_flat: np.ndarray,
                                 had_loss: float,
                                 bf_loss: float,
                                 rotation_label: str,   # 'R3' or 'R4'
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
    fig.suptitle(
        f'{model_short} – Butterfly vs Hadamard ({rotation_label}), Layer {layer_idx}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'    Saved: {os.path.basename(save_path)}')

    csv_path = save_path.replace('.png', '.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['panel', 'layer_idx', 'rotation_label', 'loss_fn_name',
                         'loss_value', 'kurtosis'])
        panel_names  = ['original', 'hadamard', 'butterfly']
        panel_arrays = [original_flat, had_flat, bf_flat]
        panel_losses = [None, had_loss, bf_loss]
        for pname, pdata, ploss in zip(panel_names, panel_arrays, panel_losses):
            k = compute_kurtosis(pdata)
            writer.writerow([pname, layer_idx, rotation_label, loss_fn_name,
                             '' if ploss is None else ploss, k])
    print(f'    Saved: {os.path.basename(csv_path)}')


def plot_variance_uniformity(var_orig: np.ndarray,
                              var_had: np.ndarray,
                              var_bf: np.ndarray,
                              layer_idx: int,
                              model_name: str,
                              save_path: str):
    """
    R3: Per-dimension variance bar chart.
    Shows how well each method equalises variance across head_dim dimensions
    (motivation: RoPE creates unequal variance across frequency bands).
    """
    D     = len(var_orig)
    x     = np.arange(D)
    width = 0.28

    fig, ax = plt.subplots(figsize=(max(12, D // 2), 5))
    ax.bar(x - width, var_orig, width, label='Original', color='slategray', alpha=0.8)
    ax.bar(x,         var_had,  width, label='Hadamard', color='tomato',    alpha=0.8)
    ax.bar(x + width, var_bf,   width, label='Butterfly',color='steelblue', alpha=0.8)

    ideal = float(np.mean(var_orig))
    ax.axhline(ideal, color='crimson', ls='--', lw=1.5,
               label=f'Ideal (mean var={ideal:.4f})')

    ax.set_xlabel('Head Dimension Index (0 = highest RoPE frequency)', fontsize=11)
    ax.set_ylabel('Per-Dimension Variance', fontsize=11)
    model_short = model_name.split('/')[-1]
    ax.set_title(
        f'{model_short} – R3 Per-Dimension Variance Uniformity (Layer {layer_idx})\n'
        f'Lower variance spread = more uniform = better for quantisation',
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
    One subplot per layer.
    Lines : Butterfly with SWD-Unif (solid blue) and SWD-Gauss (dashed purple).
    Horizontal dashed lines: Hadamard static baselines.
    """
    n = len(layer_indices)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]

    bf_styles = {
        'swd_unif':  ('steelblue',    '-',  'Butterfly SWD-Unif'),
        'swd_gauss': ('mediumorchid', '--', 'Butterfly SWD-Gauss'),
    }
    had_colors = {
        'swd_unif':  'deepskyblue',
        'swd_gauss': 'violet',
    }

    for ax, layer_idx in zip(axes, layer_indices):
        for loss_name, (color, ls, lbl) in bf_styles.items():
            h = bf_histories.get(loss_name, {}).get(layer_idx)
            if h:
                ax.plot(range(1, len(h) + 1), h, color=color, ls=ls, lw=2, label=lbl)

            # Hadamard baseline (horizontal dashed line)
            hv = had_losses.get(loss_name, {}).get(layer_idx)
            if hv is not None and not math.isinf(hv):
                suffix = loss_name.split('_')[-1].capitalize()
                ax.axhline(hv, color=had_colors[loss_name], ls=':', lw=2,
                           label=f'Hadamard {suffix} ({hv:.4f})')

        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Loss', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    model_short = model_name.split('/')[-1]
    fig.suptitle(
        f'{model_short} – Butterfly vs Hadamard Training Curves ({rotation_label})',
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

def run_experiment1_loss_comparison(model, layers, calib_data,
                                     layer_indices, model_name, model_clean):
    """
    Experiment 1: Compare activation distributions after Whip / SWD-Unif / SWD-Gauss rotations.

    Plots generated:
      {model_clean}_loss_dist_L{layer}.png   – 4-panel histogram per layer
      {model_clean}_loss_curves.png          – training convergence curves
    """
    print("\n  [Exp 1] Loss Function Distribution Comparison")

    # {loss_name: {layer_idx: [float]}}
    all_histories = {k: {} for k in ('whip', 'swd_unif', 'swd_gauss')}

    for layer_idx in layer_indices:
        print(f"  -- Layer {layer_idx} --")

        # Choose target module: mlp.up_proj input (shape: batch×seq×hidden_size)
        layer = layers[layer_idx]
        if not (hasattr(layer, 'mlp') and hasattr(layer.mlp, 'up_proj')):
            print(f"    Skipping: no mlp.up_proj at layer {layer_idx}")
            continue

        acts = collect_activations(model, calib_data, layer.mlp.up_proj, capture='input')
        if acts.shape[0] < 16:
            print(f"    Skipping: too few activations ({acts.shape[0]})")
            continue
        print(f"    Collected activations: {acts.shape}")

        hidden_size  = acts.shape[-1]
        rotated_flat = {}

        for loss_name in ('whip', 'swd_unif', 'swd_gauss'):
            print(f"    Training R1 [{loss_name}] ...")
            Q, history = train_r1_with_history(acts, hidden_size, loss_name)
            # Apply rotation: (N, D) @ (D, D) → (N, D) → flatten
            rot_acts = acts.reshape(-1, hidden_size) @ Q
            rotated_flat[loss_name]              = rot_acts.flatten()
            all_histories[loss_name][layer_idx]  = history

        # Distribution comparison plot
        save_path = os.path.join(PLOT_DIR, f"{model_clean}_loss_dist_L{layer_idx}.png")
        plot_loss_distribution(acts, rotated_flat, layer_idx, model_name, save_path)

    # Training curve plot (all layers in one figure)
    if any(all_histories[k] for k in all_histories):
        save_path = os.path.join(PLOT_DIR, f"{model_clean}_loss_curves.png")
        plot_loss_training_curves(all_histories, layer_indices, model_name, save_path)


def run_experiment2_butterfly(model, layers, calib_data,
                               layer_indices, model_name, model_clean):
    """
    Experiment 2: Butterfly Givens vs Hadamard for R3 (head_dim) and R4 (intermediate_size).

    Plots generated per rotation type:
      {model_clean}_butterfly_R4_L{layer}.png   – 3-panel distribution
      {model_clean}_butterfly_R4_curves.png      – training curves
      {model_clean}_butterfly_R3_L{layer}.png
      {model_clean}_butterfly_R3_curves.png
      {model_clean}_R3_variance_L{layer}.png     – per-dim variance bar chart
    """
    print("\n  [Exp 2] Butterfly vs Hadamard (R3 and R4)")

    # ── R4: MLP down_proj inputs (dim = intermediate_size) ──────────────────
    print("\n  --- R4 (MLP down_proj input, dim=intermediate_size) ---")
    bf_hist_r4  = {'swd_unif': {}, 'swd_gauss': {}}
    had_loss_r4 = {'swd_unif': {}, 'swd_gauss': {}}

    for layer_idx in layer_indices:
        print(f"  Layer {layer_idx} R4:")
        layer = layers[layer_idx]
        if not (hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj')):
            print(f"    Skipping: no mlp.down_proj at layer {layer_idx}")
            continue

        acts = collect_activations(model, calib_data, layer.mlp.down_proj, capture='input')
        if acts.shape[0] < 8:
            print(f"    Skipping: too few activations ({acts.shape[0]})")
            continue
        print(f"    Collected activations: {acts.shape}")

        dim     = acts.shape[-1]
        acts_2d = acts.reshape(-1, dim)

        # Verify power-of-2 for FWHT (intermediate_size=8192 for Llama-3.2-1B is 2^13)
        is_pow2 = dim > 0 and (dim & (dim - 1)) == 0
        if not is_pow2:
            print(f"    dim={dim} is not power-of-2; skipping Hadamard for R4")
            had_acts_r4 = acts_2d.copy()
            hv_unif = hv_gauss = float('inf')
        else:
            print(f"    Applying Hadamard (FWHT, dim={dim}) ...")
            had_acts_r4 = apply_hadamard_fwht(acts_2d)
            hv_unif     = compute_hadamard_loss(acts_2d, 'swd_unif')
            hv_gauss    = compute_hadamard_loss(acts_2d, 'swd_gauss')
            print(f"    Hadamard losses: swd_unif={hv_unif:.4f}, swd_gauss={hv_gauss:.4f}")

        had_loss_r4['swd_unif'][layer_idx]  = hv_unif
        had_loss_r4['swd_gauss'][layer_idx] = hv_gauss

        # Train Butterfly with SWD-Unif
        print(f"    Training Butterfly R4 [swd_unif] ...")
        bf_u, hist_u = train_butterfly_with_history(acts_2d, dim, 'swd_unif')
        bf_acts_u    = apply_butterfly_to_acts(bf_u, acts_2d)
        with torch.no_grad():
            bf_loss_u = _LOSS_FNS['swd_unif'](
                torch.tensor(bf_acts_u, device=DEVICE)).item()
        bf_hist_r4['swd_unif'][layer_idx] = hist_u

        # Train Butterfly with SWD-Gauss
        print(f"    Training Butterfly R4 [swd_gauss] ...")
        bf_g, hist_g = train_butterfly_with_history(acts_2d, dim, 'swd_gauss')
        bf_acts_g    = apply_butterfly_to_acts(bf_g, acts_2d)
        with torch.no_grad():
            bf_loss_g = _LOSS_FNS['swd_gauss'](
                torch.tensor(bf_acts_g, device=DEVICE)).item()
        bf_hist_r4['swd_gauss'][layer_idx] = hist_g

        print(f"    Butterfly losses: swd_unif={bf_loss_u:.4f}, swd_gauss={bf_loss_g:.4f}")

        # Distribution plot (use swd_unif as primary comparison)
        save_path = os.path.join(PLOT_DIR, f"{model_clean}_butterfly_R4_L{layer_idx}.png")
        plot_butterfly_distribution(
            acts_2d.flatten(), had_acts_r4.flatten(), bf_acts_u.flatten(),
            hv_unif, bf_loss_u, 'R4', layer_idx, 'swd_unif', model_name, save_path
        )

    # R4 training curves
    if any(bf_hist_r4[k] for k in bf_hist_r4):
        save_path = os.path.join(PLOT_DIR, f"{model_clean}_butterfly_R4_curves.png")
        plot_butterfly_training_curves(
            bf_hist_r4, had_loss_r4, 'R4', layer_indices, model_name, save_path
        )

    # ── R3: Attention Q-proj outputs (dim = head_dim) ───────────────────────
    print("\n  --- R3 (Q-proj output, dim=head_dim) ---")
    bf_hist_r3  = {'swd_unif': {}, 'swd_gauss': {}}
    had_loss_r3 = {'swd_unif': {}, 'swd_gauss': {}}

    try:
        num_heads = model.config.num_attention_heads
        head_dim  = model.config.hidden_size // num_heads
    except Exception:
        head_dim  = 64   # Llama-3.2-1B default
        num_heads = 32
    print(f"  head_dim={head_dim}, num_heads={num_heads}")

    for layer_idx in layer_indices:
        print(f"  Layer {layer_idx} R3 (head_dim={head_dim}):")
        layer = layers[layer_idx]
        if not (hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'q_proj')):
            print(f"    Skipping: no self_attn.q_proj at layer {layer_idx}")
            continue

        acts = collect_activations(model, calib_data, layer.self_attn.q_proj, capture='output')
        if acts.shape[0] < 8:
            print(f"    Skipping: too few activations ({acts.shape[0]})")
            continue
        print(f"    Collected activations: {acts.shape}")

        # Reshape from (N_tok, hidden_size) → (N_tok * num_heads, head_dim)
        N_tok, full_dim = acts.shape
        if full_dim % head_dim != 0:
            print(f"    Skipping: dim {full_dim} not divisible by head_dim {head_dim}")
            continue
        acts_hd = acts.reshape(-1, head_dim)   # (N_tok * num_heads, head_dim)
        print(f"    Reshaped to head vectors: {acts_hd.shape}")

        is_pow2 = head_dim > 0 and (head_dim & (head_dim - 1)) == 0
        if not is_pow2:
            print(f"    head_dim={head_dim} not power-of-2; skipping Hadamard for R3")
            had_acts_hd = acts_hd.copy()
            hv_unif3 = hv_gauss3 = float('inf')
        else:
            print(f"    Applying Hadamard (FWHT, dim={head_dim}) ...")
            had_acts_hd = apply_hadamard_fwht(acts_hd)
            hv_unif3    = compute_hadamard_loss(acts_hd, 'swd_unif')
            hv_gauss3   = compute_hadamard_loss(acts_hd, 'swd_gauss')
            print(f"    Hadamard losses: swd_unif={hv_unif3:.4f}, swd_gauss={hv_gauss3:.4f}")

        had_loss_r3['swd_unif'][layer_idx]  = hv_unif3
        had_loss_r3['swd_gauss'][layer_idx] = hv_gauss3

        # Train Butterfly R3 with SWD-Unif
        print(f"    Training Butterfly R3 [swd_unif] ...")
        bf_u3, hist_u3 = train_butterfly_with_history(acts_hd, head_dim, 'swd_unif')
        bf_acts_u3     = apply_butterfly_to_acts(bf_u3, acts_hd)
        with torch.no_grad():
            bf_loss_u3 = _LOSS_FNS['swd_unif'](
                torch.tensor(bf_acts_u3, device=DEVICE)).item()
        bf_hist_r3['swd_unif'][layer_idx] = hist_u3

        # Train Butterfly R3 with SWD-Gauss
        print(f"    Training Butterfly R3 [swd_gauss] ...")
        bf_g3, hist_g3 = train_butterfly_with_history(acts_hd, head_dim, 'swd_gauss')
        bf_acts_g3     = apply_butterfly_to_acts(bf_g3, acts_hd)
        with torch.no_grad():
            bf_loss_g3 = _LOSS_FNS['swd_gauss'](
                torch.tensor(bf_acts_g3, device=DEVICE)).item()
        bf_hist_r3['swd_gauss'][layer_idx] = hist_g3

        print(f"    Butterfly losses: swd_unif={bf_loss_u3:.4f}, swd_gauss={bf_loss_g3:.4f}")

        # Distribution plot (use swd_unif as primary)
        save_path = os.path.join(PLOT_DIR, f"{model_clean}_butterfly_R3_L{layer_idx}.png")
        plot_butterfly_distribution(
            acts_hd.flatten(), had_acts_hd.flatten(), bf_acts_u3.flatten(),
            hv_unif3, bf_loss_u3, 'R3', layer_idx, 'swd_unif', model_name, save_path
        )

        # Per-dimension variance uniformity plot (R3 specific)
        var_orig = compute_per_dim_variance(acts_hd)
        var_had  = compute_per_dim_variance(had_acts_hd)
        var_bf   = compute_per_dim_variance(bf_acts_u3)

        save_path = os.path.join(PLOT_DIR, f"{model_clean}_R3_variance_L{layer_idx}.png")
        plot_variance_uniformity(var_orig, var_had, var_bf, layer_idx, model_name, save_path)

    # R3 training curves
    if any(bf_hist_r3[k] for k in bf_hist_r3):
        save_path = os.path.join(PLOT_DIR, f"{model_clean}_butterfly_R3_curves.png")
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

    model = tokenizer = calib_data = layers = None
    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        layers       = get_layers(model)
        num_layers   = len(layers)
        layer_indices = get_layer_indices(num_layers)
        print(f"  num_layers={num_layers}, analyzing layers={layer_indices}")

        calib_data = load_calibration_data(tokenizer)

        model_clean = model_name.replace('/', '_')

        # ── Experiment 1 ────────────────────────────────────────────────────
        run_experiment1_loss_comparison(
            model, layers, calib_data, layer_indices, model_name, model_clean
        )

        # ── Experiment 2 ────────────────────────────────────────────────────
        run_experiment2_butterfly(
            model, layers, calib_data, layer_indices, model_name, model_clean
        )

        print(f"\n  Finished: {model_name}")

    except Exception as exc:
        import traceback
        print(f"  ERROR analyzing {model_name}: {exc}")
        traceback.print_exc()

    finally:
        # Release GPU memory
        for obj in (model, tokenizer, calib_data, layers):
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
    print("DartQuant v2 – Loss Function & Butterfly Rotation Visualization")
    print("=" * 80)
    print(f"HF Endpoint : {os.environ.get('HF_ENDPOINT', 'default')}")
    print(f"HF Token    : {'Set' if HF_TOKEN else 'Not set (may fail for gated models)'}")
    print(f"Cache dir   : {CACHE_DIR}")
    print(f"Plot dir    : {PLOT_DIR}")
    print(f"Models      : {MODELS}")
    print(f"Epochs R1   : {EPOCHS_LOSS}   |  Epochs Butterfly: {EPOCHS_BF}")
    print(f"Max acts    : {MAX_ACTS}       |  Batch size: {BATCH_SIZE}")

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
