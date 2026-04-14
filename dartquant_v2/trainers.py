"""
Trainers for rotation matrices with pluggable loss functions.

All training uses independent (greedy) per-layer optimization:
  - R1: Per-layer residual rotation (QR-Orth parameterization)
  - R2: Per-layer per-head value rotation (QR-Orth parameterization)
  - R3/R4 Butterfly: Learnable butterfly Givens rotations

All training hyperparameters match the original DartQuant Whip loss pipeline
defaults (SGD, lr=1e-3, momentum=0.9, CosineAnnealingLR).
"""

import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from .loss_functions import get_loss_fn
from .butterfly import ButterflyRotation, ButterflyFactored
from .nf4_quantizer import NF4FakeQuantizer
from .int4_quantizer import INT4FakeQuantizer


# ============================================================================
# QR-Orth parameterization modules (from DartQuant calibrater)
# ============================================================================

class R1_QR(nn.Module):
    """QR-Orth parameterized rotation for R1.

    A latent unconstrained matrix is optimized, and the orthogonal rotation
    is extracted via QR decomposition at each forward pass.

    Reference: DartQuant/calibrater/r1_base_qr.py:32-41
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.matrix = nn.Parameter(torch.eye(hidden_size))
        self.rotate = None

    def forward(self, x):
        self.rotate, _ = torch.linalg.qr(self.matrix, mode='complete')
        return torch.matmul(x, self.rotate)


class R2_Per_Head(nn.Module):
    """QR-Orth parameterized per-head rotation for R2.

    Each KV-head group gets an independent rotation matrix in the
    head_dim subspace.

    Reference: DartQuant/calibrater/r2_base_qr.py:36-66
    """

    def __init__(self, hidden_size: int, head_num: int, kv_head: int):
        super().__init__()
        assert hidden_size % head_num == 0
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_dim = hidden_size // head_num
        self.kv_head = kv_head
        self.matrix = nn.Parameter(
            torch.eye(self.head_dim).repeat(self.kv_head, 1, 1)
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
        r_x = r_x.transpose(0, 1)
        r_x = r_x.reshape(x_shape)
        return r_x


# ============================================================================
# Initialization helpers
# ============================================================================

def _random_orthogonal_matrix(size, device):
    """Generate random orthogonal matrix via QR decomposition."""
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def _get_init_matrix(size, mode, device):
    """Get initialization matrix for R1.

    Args:
        size: Matrix dimension
        mode: 'hadamard' or 'random'
        device: torch device
    """
    if mode == 'hadamard':
        try:
            import sys, os
            sys.path.insert(0, os.path.join(
                os.path.dirname(__file__), '..', 'DartQuant', 'fake_quant'))
            from hadamard_utils import random_hadamard_matrix
            return random_hadamard_matrix(size, device)
        except ImportError:
            logging.warning("hadamard_utils not available, using random init")
            return _random_orthogonal_matrix(size, device)
    elif mode == 'random':
        return _random_orthogonal_matrix(size, device)
    else:
        raise ValueError(f"Unknown init mode: {mode}")


def _get_multi_head_init(hidden_size, head_num, kv_head, mode, device):
    """Get initialization for R2 (per KV-head)."""
    head_dim = hidden_size // head_num
    org = _get_init_matrix(head_dim, mode, device)
    return org.unsqueeze(0).repeat(kv_head, 1, 1)


def _create_optimizer(params, optim_name, lr, momentum):
    """Create optimizer matching original DartQuant defaults."""
    if optim_name == 'sgd':
        return torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif optim_name == 'adam':
        return torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


# ============================================================================
# R1 Training (Independent / Greedy per-layer)
# ============================================================================

def _parse_layer_idx(name: str) -> int:
    """Parse integer layer index from a dotted module path.

    E.g. "model.layers.3.mlp.up_proj" → 3
    Returns -1 if no digit part is found.
    """
    for part in name.split('.'):
        if part.isdigit():
            return int(part)
    return -1


# ============================================================================
# Single-layer R1 Training (greedy/independent: called per-layer from pipeline)
# ============================================================================

def train_r1_single_layer(
    acts: torch.Tensor,
    hidden_size: int,
    loss_fn_name: str,
    lr: float = 1e-3,
    momentum: float = 0.9,
    epochs: int = 10,
    batch_size: int = 64,
    cos_lr: bool = False,
    optim: str = 'sgd',
    init_mode: str = 'hadamard',
    accumulation_steps: int = 1,
    train_subset_size: float = 1.0,
    device='cuda',
    layer_idx: int = 0,
) -> torch.Tensor:
    """Train R1 rotation for a single layer.

    Memory-efficient: called from the pipeline per-layer instead of
    collecting all layers' activations at once.

    Args:
        acts: Activation tensor (N, hidden_size) — already concatenated
              from all targets for this layer.
        layer_idx: Layer index (for logging only).

    Returns:
        Trained rotation matrix tensor (hidden_size, hidden_size), float64.
    """
    loss_fn = get_loss_fn(loss_fn_name)
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Keep data on CPU — stream batches to GPU via DataLoader.
    # This allows handling datasets much larger than GPU memory
    # (matching the official disk-based DataLoader approach).
    acts = acts.float()  # CPU
    dataset = TensorDataset(acts)

    R1 = R1_QR(hidden_size).to(device)
    R1.matrix.data = _get_init_matrix(hidden_size, init_mode, device).float()

    optimizer = _create_optimizer(R1.parameters(), optim, lr, momentum)
    scheduler = (CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
                 if cos_lr else None)

    R1.train()

    # Subsample once before training (matching official DartQuant r1_base_qr.py).
    # The SAME random subset is used for all epochs; RandomSampler shuffles
    # the order within this subset each epoch.
    train_data = dataset
    if train_subset_size < 1.0:
        num_samples = max(1, int(len(dataset) * train_subset_size))
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        train_data = torch.utils.data.Subset(dataset, indices.tolist())

    logging.info(f"  Training R1 layer {layer_idx} ({loss_fn_name}, "
                 f"{len(train_data)} samples, batch_size={batch_size})")

    sampler = RandomSampler(train_data)
    dataloader = DataLoader(train_data, sampler=sampler, batch_size=batch_size,
                            pin_memory=True, num_workers=0)

    for epoch in range(epochs):
        loss_log = []

        for batch_idx, (batch_samples,) in enumerate(dataloader):
            batch_samples = batch_samples.to(device).reshape(-1, hidden_size)
            outputs = R1(batch_samples)
            loss = loss_fn(outputs) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_log.append(loss.detach())

        if scheduler:
            scheduler.step()

        mean_loss = torch.stack(loss_log).mean()
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            lr_str = (f", LR: {scheduler.get_last_lr()[0]:.4e}"
                      if scheduler else "")
            logging.info(
                f"    R1 L{layer_idx} Epoch [{epoch+1}/{epochs}], "
                f"Loss: {mean_loss.item():.6f}{lr_str}"
            )

    return R1.rotate.data.detach()


# ============================================================================
# Single-layer R2 Training (memory-efficient)
# ============================================================================

def train_r2_single_layer(
    acts: torch.Tensor,
    hidden_size: int,
    num_heads: int,
    kv_heads: int,
    loss_fn_name: str,
    lr: float = 1e-3,
    momentum: float = 0.9,
    epochs: int = 5,
    batch_size: int = 128,
    cos_lr: bool = False,
    optim: str = 'sgd',
    accumulation_steps: int = 2,
    device='cuda',
    layer_idx: int = 0,
    layers_path: str = 'model.layers',
) -> tuple:
    """Train R2 rotation for a single layer.

    Returns:
        (key, rotation_tensor) where key = "model.layers.{i}.self_attn.R2"
    """
    loss_fn = get_loss_fn(loss_fn_name)
    if isinstance(device, str):
        device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if isinstance(acts, np.ndarray):
        acts = torch.tensor(
            np.nan_to_num(acts, nan=0.0, posinf=65504, neginf=-65504),
            dtype=torch.float32
        )
    # Keep data on CPU — stream batches to GPU via DataLoader.
    acts = acts.float()  # CPU
    dataset = TensorDataset(acts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=True, num_workers=0)

    R2 = R2_Per_Head(hidden_size, num_heads, kv_heads).to(device)
    R2.matrix.data = _get_multi_head_init(
        hidden_size, num_heads, kv_heads, 'hadamard', device
    ).float()

    optimizer = _create_optimizer(R2.parameters(), optim, lr, momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

    R2.train()
    n_batches = (len(dataset) + batch_size - 1) // batch_size
    logging.info(f"  Training R2 layer {layer_idx} "
                 f"({len(acts)} rows, batch_size={batch_size}, "
                 f"{n_batches} batches/epoch, accum={accumulation_steps})...")

    for epoch in range(epochs):
        loss_log = []
        for batch_idx, (batch_samples,) in enumerate(dataloader):
            batch_samples = batch_samples.to(device)
            outputs = R2(batch_samples)
            loss = loss_fn(outputs) / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_log.append(loss.detach())

        if scheduler:
            scheduler.step()

        if (epoch + 1) == epochs:
            mean_loss = torch.stack(loss_log).mean()
            logging.info(
                f"    R2 L{layer_idx} final Loss: {mean_loss.item():.6f}"
            )

    key = f"{layers_path}.{layer_idx}.self_attn.R2"
    return key, R2.rotate.data.detach()


# ============================================================================
# Butterfly Training (for R3 and R4)
# ============================================================================

def _init_butterfly_from_hadamard(
    butterfly: 'ButterflyRotation',
    dim: int,
    device,
    n_steps: int = 300,
    lr: float = 0.05,
):
    """Warm-start ButterflyRotation angles by fitting to a random Hadamard matrix.

    Generates H = WHT_normalized * diag(D), where D is a random ±1 diagonal,
    then minimises ||B(θ) - H||_F with Adam for n_steps steps.
    This starts the main butterfly training from the Hadamard point rather than
    the identity (θ = 0), matching DartQuant's original Random Hadamard baseline.

    Skipped silently for ButterflyFactored (non-power-of-2 dims).
    """
    if not isinstance(butterfly, ButterflyRotation):
        return  # ButterflyFactored: leave at identity

    # ── Build random Hadamard target ─────────────────────────────────────────
    try:
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(
            _os.path.dirname(__file__), '..', 'DartQuant', 'fake_quant'))
        from hadamard_utils import random_hadamard_matrix
        H = random_hadamard_matrix(dim, device).float()
    except (ImportError, Exception):
        # Pure PyTorch fallback: WHT_normalized * random column signs
        D = (torch.randint(0, 2, (dim,)) * 2 - 1).float().to(device)
        x = torch.eye(dim, device=device)
        h = 1
        while h < dim:
            x = x.reshape(dim, dim // (2 * h), 2, h)
            a = x[:, :, 0, :].contiguous()
            b = x[:, :, 1, :].contiguous()
            x = torch.stack([a + b, a - b], dim=2).reshape(dim, dim)
            h *= 2
        H = (x / math.sqrt(dim)) * D.unsqueeze(0)

    H = H.detach()
    I_mat = torch.eye(dim, device=device, dtype=torch.float32)
    init_opt = torch.optim.Adam(butterfly.parameters(), lr=lr)
    butterfly.train()
    for _ in range(n_steps):
        init_opt.zero_grad()
        loss = ((butterfly.forward(I_mat) - H) ** 2).sum()
        loss.backward()
        init_opt.step()

    with torch.no_grad():
        fit_loss = ((butterfly.forward(I_mat) - H) ** 2).sum().item()
    logging.info(f"  Butterfly Hadamard init (dim={dim}): fit loss = {fit_loss:.4f}")


def _butterfly_train_step(
    butterfly: nn.Module,
    batch: torch.Tensor,
    loss_fn,
    fake_quant,
    weight_fake_quant,
    use_recon: bool,
    use_weight_recon: bool,
    weight_matrices,
    num_weight_samples: int,
    lambda_recon: float,
) -> torch.Tensor:
    """Single training step for butterfly rotation (extracted for clarity).

    Returns the total loss tensor (un-backpropagated).
    """
    # Forward: X_out = B @ X_in  (butterfly acts on row vectors)
    outputs = butterfly(batch)

    # Distribution loss: L_dist(X_out)
    dist_loss = loss_fn(outputs)

    if use_weight_recon:
        # ── Paper Eq 17: joint weight + activation reconstruction ──
        # L_recon = ||Wx - Q(W@B^T) · Q(B@x)||^2
        w_idx = torch.randint(0, num_weight_samples, (1,)).item()
        W = weight_matrices[w_idx]

        with torch.no_grad():
            Bx_q = fake_quant(outputs)
        Bx_ste = outputs + (Bx_q - outputs).detach()

        W_rotated = butterfly.inverse_forward(W)
        with torch.no_grad():
            WBt_q = weight_fake_quant(W_rotated)
        WBt_ste = W_rotated + (WBt_q - W_rotated).detach()

        y_true = batch @ W.T
        y_hat = Bx_ste @ WBt_ste.T
        recon_loss = F.mse_loss(y_true, y_hat)
        return dist_loss + lambda_recon * recon_loss

    elif use_recon:
        # ── Activation-only reconstruction (for R3 online rotation) ──
        with torch.no_grad():
            x_hat_q = fake_quant(outputs)
        x_hat = outputs + (x_hat_q - outputs).detach()
        x_recon = butterfly.inverse_forward(x_hat)
        recon_loss = F.mse_loss(batch, x_recon)
        return dist_loss + lambda_recon * recon_loss

    return dist_loss


def train_butterfly(
    activations: torch.Tensor,
    dim: int,
    loss_fn_name: str,
    label: str = "butterfly",
    lr: float = 1e-3,
    momentum: float = 0.9,
    epochs: int = 10,
    batch_size: int = 64,
    cos_lr: bool = True,
    optim: str = 'sgd',
    device: str = 'cuda',
    quantizer_type: str = 'none',
    lambda_recon: float = 0.1,
    quant_block_size: int = 64,
    weight_matrices: torch.Tensor = None,
    weight_quantizer_type: str = 'none',
    k_factor_mode: str = 'latent',
    butterfly_init: str = 'identity',
) -> nn.Module:
    """Train a Butterfly Givens rotation.

    Used for R3 (head_dim, typically power of 2) and R4 (intermediate_size,
    may not be power of 2).

    The total loss combines the distribution-shaping loss with an optional
    quantization reconstruction loss (paper Eq 17).

    **Weight-aware mode** (when weight_matrices is provided, for R4):

        L_recon = ||W @ x - Q(W @ B^T) @ Q(B @ x)||^2

    where Q denotes Dequant(Quant(·)), B is the butterfly matrix, and W is
    the down_proj weight.  This jointly accounts for both weight and
    activation quantization error, matching the paper's formulation.

    **Activation-only mode** (when weight_matrices is None, for R3):

        L_recon = ||X_in - B^T @ Q(B @ X_in)||^2

    Both modes use a straight-through estimator (STE) so that gradients
    flow back through the rotation parameters despite the non-differentiable
    quantizer.

    Args:
        activations: Tensor of shape (N, dim)
        dim: Rotation dimension
        loss_fn_name: Loss function name
        label: Label for logging (e.g., "R3" or "R4")
        lr: Learning rate
        momentum: SGD momentum
        epochs: Training epochs
        batch_size: Mini-batch size
        cos_lr: Use cosine annealing LR scheduler
        optim: Optimizer ('sgd' or 'adam')
        device: Device string
        quantizer_type: Fake quantizer for activation L_recon. One of:
            'nf4'  - NF4 non-uniform quantizer (pairs with swd_gauss)
            'int4' - INT4 symmetric uniform quantizer (pairs with swd_unif)
            'none' - Disable L_recon (original behaviour, default)
        lambda_recon: Weight for L_recon relative to L_dist (default 0.1)
        quant_block_size: Per-block size used by the fake quantizer (default 64)
        weight_matrices: Optional tensor of shape (num_weights, out_dim, dim)
            containing down_proj weight matrices for weight-aware reconstruction
            loss (paper Eq 17). Only used for R4 where butterfly is baked into
            weights. When None, falls back to activation-only reconstruction.
        weight_quantizer_type: Fake quantizer for weight path. Same choices as
            quantizer_type. Only used when weight_matrices is not None.
        k_factor_mode: Parameterization for the K-dimensional factor in
            ButterflyFactored (non-power-of-2 dims only).
            'latent' (default) - unconstrained matrix + QR decomposition
                (OR-Orth, same as R1/R2).
            'cayley' - Cayley transform with skew-symmetric param.
        butterfly_init: Initialization strategy for butterfly angles.
            'identity' (default, recommended by ButterflyQuant paper) -
                all angles=0, enabling gradual rotation learning.
            'hadamard' - warm-start by fitting to a random Hadamard matrix.

    Returns:
        Trained ButterflyRotation or ButterflyFactored module
    """
    loss_fn = get_loss_fn(loss_fn_name)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Build optional fake quantizer for reconstruction loss
    fake_quant: nn.Module | None = None
    if quantizer_type == 'nf4':
        fake_quant = NF4FakeQuantizer(block_size=quant_block_size).to(device)
    elif quantizer_type == 'int4':
        fake_quant = INT4FakeQuantizer(block_size=quant_block_size).to(device)
    elif quantizer_type != 'none':
        raise ValueError(
            f"Unknown quantizer_type '{quantizer_type}'. "
            "Choose from 'nf4', 'int4', or 'none'."
        )

    use_recon = fake_quant is not None and lambda_recon > 0.0

    # Build optional weight fake quantizer for joint reconstruction loss (Eq 17)
    weight_fake_quant: nn.Module | None = None
    if weight_matrices is not None and weight_quantizer_type != 'none':
        if weight_quantizer_type == 'nf4':
            weight_fake_quant = NF4FakeQuantizer(block_size=quant_block_size).to(device)
        elif weight_quantizer_type == 'int4':
            weight_fake_quant = INT4FakeQuantizer(block_size=quant_block_size).to(device)
        else:
            raise ValueError(
                f"Unknown weight_quantizer_type '{weight_quantizer_type}'. "
                "Choose from 'nf4', 'int4', or 'none'."
            )

    use_weight_recon = (weight_matrices is not None and
                        fake_quant is not None and
                        weight_fake_quant is not None and
                        lambda_recon > 0.0)

    if use_weight_recon:
        weight_matrices = weight_matrices.float().to(device)
        num_weight_samples = weight_matrices.shape[0]
        logging.info(
            f"  Weight-aware recon (Eq 17) enabled: "
            f"{num_weight_samples} weight matrices, "
            f"shape {tuple(weight_matrices.shape)}"
        )

    # Choose butterfly type based on dimension
    if dim > 0 and (dim & (dim - 1)) == 0:
        butterfly = ButterflyRotation(dim).to(device)
    else:
        butterfly = ButterflyFactored(dim, k_factor_mode=k_factor_mode).to(device)

    # ButterflyQuant paper (Section 4.4): identity init (θ=0) outperforms
    # Hadamard init by 6.3% and random init by 13.4%.  Identity enables
    # "gradual rotation learning through small incremental adjustments."
    if butterfly_init == 'hadamard':
        _init_butterfly_from_hadamard(butterfly, dim, device)
    else:
        logging.info(f"  Butterfly {label} using identity initialization (θ=0)")

    optimizer = _create_optimizer(butterfly.parameters(), optim, lr, momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

    # ── Prepare data ────────────────────────────────────────────────────────
    if isinstance(activations, np.ndarray):
        activations = torch.tensor(activations, dtype=torch.float32)
    activations = activations.float().reshape(-1, dim)
    n_samples = activations.shape[0]
    data_mb = n_samples * dim * 4 / (1 << 20)  # float32 size in MB

    # Move data to GPU when it fits (< 2 GB).  This eliminates the
    # CPU → GPU transfer bottleneck that dominates training time for the
    # small tensors typical of butterfly training (e.g. 131 K × 128 ≈ 64 MB).
    _on_gpu = False
    if data_mb < 2048:
        try:
            activations = activations.to(device)
            _on_gpu = True
        except RuntimeError:
            pass  # OOM — keep data on CPU
    if _on_gpu:
        logging.info(f"  Data on GPU ({data_mb:.0f} MB, {n_samples} samples)")
    else:
        logging.info(f"  Data on CPU ({data_mb:.0f} MB, {n_samples} samples)")

    # Auto-scale batch_size for GPU efficiency.  Butterfly operates on
    # small dimensions (head_dim ≈ 64-128), so each kernel launch does
    # very little work.  Larger batches amortize kernel-launch overhead
    # and fill the GPU's compute units.  We scale up to min(n_samples, 4096)
    # while keeping ≥ 50 steps/epoch (or using all data if < 50 batches).
    min_steps_per_epoch = 50
    auto_batch = min(n_samples, max(batch_size, n_samples // min_steps_per_epoch))
    auto_batch = min(auto_batch, 4096)  # cap at 4096 to avoid excessive memory
    if auto_batch != batch_size:
        logging.info(
            f"  Auto-scaled batch_size {batch_size} → {auto_batch} "
            f"for GPU efficiency (dim={dim}, n={n_samples})"
        )
        batch_size = auto_batch

    steps_per_epoch = max(1, (n_samples + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs

    # Try torch.compile for kernel fusion (PyTorch ≥ 2.0).
    # The butterfly forward pass consists of log₂(d) layers of tiny
    # element-wise ops; torch.compile fuses them into fewer GPU kernels,
    # dramatically reducing kernel-launch overhead.
    _compiled = False
    try:
        butterfly = torch.compile(butterfly)
        _compiled = True
        logging.info("  torch.compile enabled")
    except Exception:
        pass

    # CPU-data fallback: build DataLoader with pin_memory
    _dataloader = None
    if not _on_gpu:
        dataset = TensorDataset(activations)
        _dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            pin_memory=True, num_workers=2,
        )

    butterfly.train()
    if use_weight_recon:
        recon_tag = f" + {lambda_recon}*L_recon_Eq17[act={quantizer_type},w={weight_quantizer_type}]"
    elif use_recon:
        recon_tag = f" + {lambda_recon}*L_recon[{quantizer_type}]"
    else:
        recon_tag = ""
    logging.info(
        f"Training Butterfly {label} (dim={dim}), {epochs} epochs × "
        f"{steps_per_epoch} steps = {total_steps} total, "
        f"batch_size={batch_size}, loss={loss_fn_name}{recon_tag}"
    )

    _t_train_start = time.time()

    for epoch in range(epochs):
        loss_log = []

        if _on_gpu:
            # ── GPU-native random batching (no DataLoader overhead) ──
            perm = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, batch_size):
                batch = activations[perm[start:start + batch_size]]
                loss = _butterfly_train_step(
                    butterfly, batch, loss_fn, fake_quant, weight_fake_quant,
                    use_recon, use_weight_recon, weight_matrices,
                    num_weight_samples if use_weight_recon else 0,
                    lambda_recon,
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_log.append(loss.detach())
        else:
            # ── CPU-data fallback with DataLoader ──
            for (batch_cpu,) in _dataloader:
                batch = batch_cpu.to(device)
                loss = _butterfly_train_step(
                    butterfly, batch, loss_fn, fake_quant, weight_fake_quant,
                    use_recon, use_weight_recon, weight_matrices,
                    num_weight_samples if use_weight_recon else 0,
                    lambda_recon,
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_log.append(loss.detach())

        if scheduler:
            scheduler.step()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            mean_loss = torch.stack(loss_log).mean()
            elapsed = time.time() - _t_train_start
            steps_done = (epoch + 1) * steps_per_epoch
            logging.info(
                f"  {label} Epoch [{epoch+1}/{epochs}], "
                f"Loss: {mean_loss.item():.6f}, "
                f"{steps_done}/{total_steps} steps, "
                f"{elapsed:.1f}s elapsed"
            )

    elapsed = time.time() - _t_train_start
    logging.info(
        f"Butterfly {label} training complete in {elapsed:.1f}s "
        f"({total_steps / elapsed:.0f} steps/s)"
    )
    return butterfly
