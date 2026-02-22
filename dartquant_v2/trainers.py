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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from .loss_functions import get_loss_fn
from .butterfly import ButterflyRotation, ButterflyFactored


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
) -> nn.Module:
    """Train a Butterfly Givens rotation.

    Used for R3 (head_dim, typically power of 2) and R4 (intermediate_size,
    may not be power of 2).

    Args:
        activations: Tensor of shape (N, dim)
        dim: Rotation dimension
        loss_fn_name: Loss function name
        label: Label for logging (e.g., "R3" or "R4")
        ... (same training hyperparams)

    Returns:
        Trained ButterflyRotation or ButterflyFactored module
    """
    loss_fn = get_loss_fn(loss_fn_name)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Choose butterfly type based on dimension
    if dim > 0 and (dim & (dim - 1)) == 0:
        butterfly = ButterflyRotation(dim).to(device)
    else:
        butterfly = ButterflyFactored(dim).to(device)

    optimizer = _create_optimizer(butterfly.parameters(), optim, lr, momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

    # Prepare data
    if isinstance(activations, np.ndarray):
        activations = torch.tensor(activations, dtype=torch.float32)
    activations = activations.float().reshape(-1, dim)

    dataset = TensorDataset(activations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    butterfly.train()
    logging.info(f"Training Butterfly {label} (dim={dim}), {epochs} epochs")

    for epoch in range(epochs):
        loss_log = []
        for (batch,) in dataloader:
            batch = batch.to(device)
            outputs = butterfly(batch)
            loss = loss_fn(outputs)
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
                f"{loss_fn_name} Loss: {mean_loss.item():.6f}"
            )

    logging.info(f"Butterfly {label} training complete.")
    return butterfly
