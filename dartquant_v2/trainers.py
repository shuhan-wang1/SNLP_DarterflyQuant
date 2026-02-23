"""
Trainers for rotation matrices with pluggable loss functions.

Provides training functions for:
  - R1: Global residual rotation (QR-Orth parameterization)
  - R2: Per-head value rotation (QR-Orth parameterization)
  - R3/R4 Butterfly: Learnable butterfly Givens rotations

All training hyperparameters match the original DartQuant Whip loss pipeline
defaults (SGD, lr=1e-3, momentum=0.9, CosineAnnealingLR).
"""

import logging
import math

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
# R1 Training
# ============================================================================

def train_r1(
    activations: dict,
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
    device: str = 'cuda',
) -> torch.Tensor:
    """Train R1 global rotation matrix using specified loss function.

    All optimizer hyperparameters are identical to the original DartQuant
    Whip loss pipeline (r1_base_qr.py).

    Args:
        activations: Dict mapping layer names to activation tensors,
                     or a single stacked tensor of shape (N, hidden_size)
        hidden_size: Model hidden dimension
        loss_fn_name: 'whip', 'swd_unif', or 'swd_gauss'
        lr: Learning rate (default: 1e-3)
        momentum: SGD momentum (default: 0.9)
        epochs: Number of training epochs (default: 10)
        batch_size: Training batch size (default: 64)
        cos_lr: Use cosine annealing LR scheduler
        optim: Optimizer name ('sgd' or 'adam')
        init_mode: Initialization ('hadamard' or 'random')
        accumulation_steps: Gradient accumulation steps
        train_subset_size: Fraction of data to use per epoch
        device: Device string

    Returns:
        R1 rotation matrix of shape (hidden_size, hidden_size), float64
    """
    loss_fn = get_loss_fn(loss_fn_name)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Prepare data
    if isinstance(activations, dict):
        all_acts = []
        for name in sorted(activations.keys()):
            act = activations[name]
            if isinstance(act, torch.Tensor):
                all_acts.append(act.reshape(-1, hidden_size).float().cpu())
            else:
                all_acts.append(torch.tensor(
                    np.nan_to_num(act, nan=0.0, posinf=65504, neginf=-65504),
                    dtype=torch.float32
                ).reshape(-1, hidden_size))
        all_acts = torch.cat(all_acts, dim=0)
    else:
        all_acts = activations.reshape(-1, hidden_size).float().cpu()

    dataset = TensorDataset(all_acts)

    # Initialize R1 module
    R1 = R1_QR(hidden_size).to(device)
    R1.matrix.data = _get_init_matrix(hidden_size, init_mode, device).float()

    optimizer = _create_optimizer(R1.parameters(), optim, lr, momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

    # Training loop (matches r1_base_qr.py structure)
    R1.train()
    logging.info(f"Training R1 with {loss_fn_name} loss, {epochs} epochs, lr={lr}")

    for epoch in range(epochs):
        loss_log = []
        num_samples = max(1, int(len(dataset) * train_subset_size))
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)
        sampler = RandomSampler(torch.utils.data.Subset(dataset, indices))
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

        for batch_idx, (batch_samples,) in enumerate(dataloader):
            batch_samples = batch_samples.to(device).float().reshape(-1, hidden_size)
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
            lr_str = f", LR: {scheduler.get_last_lr()[0]:.4e}" if scheduler else ""
            logging.info(
                f"  R1 Epoch [{epoch+1}/{epochs}], "
                f"{loss_fn_name} Loss: {mean_loss.item():.6f}{lr_str}"
            )

    logging.info("R1 training complete.")
    return R1.rotate.data.detach()


def _parse_layer_idx(name: str) -> int:
    """Parse integer layer index from a dotted module path.

    E.g. "model.layers.3.mlp.up_proj" → 3
    Returns -1 if no digit part is found.
    """
    for part in name.split('.'):
        if part.isdigit():
            return int(part)
    return -1


def train_r1_all_layers(
    activations: dict,
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
    device: str = 'cuda',
) -> dict:
    """Train per-layer R1 rotation matrices independently (like train_r2_all_layers).

    Instead of pooling all layers' activations into a single gradient descent
    (as train_r1 does), this function trains a separate R1_QR for each layer
    using only that layer's own activations.  This preserves each layer's
    individual activation distribution rather than averaging over all layers,
    and prevents gradient cancellation from mixing layers with different
    covariance structures.

    Per-layer entries from the same layer (e.g. both up_proj and q_proj inputs
    of layer l) are concatenated before training — giving the optimizer a
    richer sample of that layer's residual-stream distribution.

    Args:
        activations:  Dict mapping module paths to activation tensors.
                      Multiple entries per layer are concatenated automatically.
                      Example keys: "model.layers.0.mlp.up_proj",
                                    "model.layers.0.self_attn.q_proj"
        hidden_size:  Model hidden dimension.
        loss_fn_name: Loss function name (same choices as train_r1).
        lr, momentum, epochs, batch_size, cos_lr, optim,
        init_mode, accumulation_steps, train_subset_size, device:
                      Same semantics as train_r1.

    Returns:
        dict {layer_idx (int) → R1 matrix Tensor (hidden_size, hidden_size),
              dtype float64}  — one entry per layer found in activations.
    """
    loss_fn = get_loss_fn(loss_fn_name)
    device  = torch.device(device if torch.cuda.is_available() else 'cpu')

    # ── Group activations by layer index ─────────────────────────────────────
    acts_per_layer: dict = {}
    for name, acts in activations.items():
        lid = _parse_layer_idx(name)
        if lid < 0:
            continue
        if isinstance(acts, torch.Tensor):
            t = acts.reshape(-1, hidden_size).float().cpu()
        else:
            t = torch.tensor(
                np.nan_to_num(acts, nan=0.0, posinf=65504, neginf=-65504),
                dtype=torch.float32,
            ).reshape(-1, hidden_size)
        acts_per_layer.setdefault(lid, []).append(t)

    for lid in acts_per_layer:
        acts_per_layer[lid] = torch.cat(acts_per_layer[lid], dim=0)

    # ── Train one R1_QR per layer ─────────────────────────────────────────────
    R1_dict: dict = {}
    for layer_idx in sorted(acts_per_layer.keys()):
        all_acts = acts_per_layer[layer_idx]
        dataset  = TensorDataset(all_acts)

        R1 = R1_QR(hidden_size).to(device)
        R1.matrix.data = _get_init_matrix(hidden_size, init_mode, device).float()

        optimizer = _create_optimizer(R1.parameters(), optim, lr, momentum)
        scheduler = (CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
                     if cos_lr else None)

        R1.train()
        logging.info(f"Training R1 layer {layer_idx} with {loss_fn_name} loss, "
                     f"{len(all_acts)} samples")

        for epoch in range(epochs):
            loss_log = []
            num_samples = max(1, int(len(dataset) * train_subset_size))
            indices  = np.random.choice(len(dataset), size=num_samples, replace=False)
            sampler  = RandomSampler(torch.utils.data.Subset(dataset, indices))
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

            for batch_idx, (batch_samples,) in enumerate(dataloader):
                batch_samples = batch_samples.to(device).float().reshape(-1, hidden_size)
                outputs = R1(batch_samples)
                loss    = loss_fn(outputs) / accumulation_steps
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
                    f"  R1 Layer {layer_idx} Epoch [{epoch+1}/{epochs}], "
                    f"{loss_fn_name} Loss: {mean_loss.item():.6f}{lr_str}"
                )

        R1_dict[layer_idx] = R1.rotate.data.detach()

    logging.info(f"Per-layer R1 training complete. "
                 f"Trained {len(R1_dict)} layers.")
    return R1_dict


# ============================================================================
# R2 Training
# ============================================================================

def train_r2_all_layers(
    activations_per_layer: dict,
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
    device: str = 'cuda',
) -> dict:
    """Train R2 per-head rotation for all layers.

    Args:
        activations_per_layer: Dict mapping layer_id (int) to activation tensor
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        kv_heads: Number of KV heads
        loss_fn_name: Loss function name
        ... (same training hyperparams as R1)

    Returns:
        Dict mapping "model.layers.{i}.self_attn.R2" to rotation tensor
    """
    loss_fn = get_loss_fn(loss_fn_name)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    save_dict = {}

    for layer_id in sorted(activations_per_layer.keys()):
        acts = activations_per_layer[layer_id]
        if isinstance(acts, np.ndarray):
            acts = torch.tensor(
                np.nan_to_num(acts, nan=0.0, posinf=65504, neginf=-65504),
                dtype=torch.float32
            )
        acts = acts.float()

        dataset = TensorDataset(acts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Initialize R2
        R2 = R2_Per_Head(hidden_size, num_heads, kv_heads).to(device)
        R2.matrix.data = _get_multi_head_init(
            hidden_size, num_heads, kv_heads, 'hadamard', device
        ).float()

        optimizer = _create_optimizer(R2.parameters(), optim, lr, momentum)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

        R2.train()
        logging.info(f"  Training R2 for layer {layer_id}...")

        for epoch in range(epochs):
            loss_log = []
            for batch_idx, (batch_samples,) in enumerate(dataloader):
                batch_samples = batch_samples.to(device).float()
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
                    f"    Layer {layer_id} final {loss_fn_name} Loss: "
                    f"{mean_loss.item():.6f}"
                )

        save_dict[f"model.layers.{layer_id}.self_attn.R2"] = R2.rotate.data.detach()

    logging.info("R2 training complete for all layers.")
    return save_dict


# ============================================================================
# Butterfly Training (for R3 and R4)
# ============================================================================

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

    optimizer = _create_optimizer(butterfly.parameters(), optim, lr, momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

    # Prepare data
    if isinstance(activations, np.ndarray):
        activations = torch.tensor(activations, dtype=torch.float32)
    activations = activations.float().reshape(-1, dim)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    butterfly.train()
    if use_weight_recon:
        recon_tag = f" + {lambda_recon}*L_recon_Eq17[act={quantizer_type},w={weight_quantizer_type}]"
    elif use_recon:
        recon_tag = f" + {lambda_recon}*L_recon[{quantizer_type}]"
    else:
        recon_tag = ""
    logging.info(
        f"Training Butterfly {label} (dim={dim}), {epochs} epochs, "
        f"loss={loss_fn_name}{recon_tag}"
    )

    for epoch in range(epochs):
        loss_log = []
        for (batch,) in dataloader:
            batch = batch.to(device)

            # Forward: X_out = R3 @ X_in  (butterfly acts on row vectors)
            outputs = butterfly(batch)

            # Distribution loss: L_dist(X_out)
            dist_loss = loss_fn(outputs)

            if use_weight_recon:
                # ── Paper Eq 17: joint weight + activation reconstruction ──
                # L_recon = ||Wx - Q(W@B^T) · Q(B@x)||^2
                # where Q = Dequant∘Quant (fake quantize-dequantize)

                # Sample a random layer's weight from the bank
                w_idx = torch.randint(0, num_weight_samples, (1,)).item()
                W = weight_matrices[w_idx]  # (out_dim, dim)

                # STE for activation quantisation: Bx
                with torch.no_grad():
                    Bx_q = fake_quant(outputs)
                Bx_ste = outputs + (Bx_q - outputs).detach()

                # Rotate weight rows by B^T and STE-quantise: WB^T
                W_rotated = butterfly.inverse_forward(W)  # (out_dim, dim)
                with torch.no_grad():
                    WBt_q = weight_fake_quant(W_rotated)
                WBt_ste = W_rotated + (WBt_q - W_rotated).detach()

                # Ground-truth vs quantised output
                y_true = batch @ W.T            # (batch_sz, out_dim)
                y_hat = Bx_ste @ WBt_ste.T      # (batch_sz, out_dim)
                recon_loss = F.mse_loss(y_true, y_hat)

                loss = dist_loss + lambda_recon * recon_loss

            elif use_recon:
                # ── Activation-only reconstruction (for R3 online rotation) ──
                # L_recon = ||X_in - B^T @ FakeQuant(B @ X_in)||^2
                with torch.no_grad():
                    x_hat_q = fake_quant(outputs)
                x_hat = outputs + (x_hat_q - outputs).detach()  # STE

                x_recon = butterfly.inverse_forward(x_hat)
                recon_loss = F.mse_loss(batch, x_recon)

                loss = dist_loss + lambda_recon * recon_loss
            else:
                loss = dist_loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_log.append(loss.detach())

        if scheduler:
            scheduler.step()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            mean_loss = torch.stack(loss_log).mean()
            logging.info(
                f"  {label} Epoch [{epoch+1}/{epochs}], "
                f"total Loss: {mean_loss.item():.6f}"
            )

    logging.info(f"Butterfly {label} training complete.")
    return butterfly
