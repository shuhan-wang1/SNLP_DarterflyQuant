"""
Joint R1 training module for JointQuant
This is the key innovation: training a single global R1 rotation matrix
that minimizes Whip loss across ALL layers jointly.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import logging

from . import hadamard_utils


class ActivationDataset(Dataset):
    """Dataset for activation tensors"""
    def __init__(self, activations: torch.Tensor):
        self.activations = activations
    
    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx]


def whip_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Whip loss: sum(exp(-|x|))
    Pushes small values away from zero -> uniform distribution
    
    CRITICAL: For large outliers (e.g., |x| = 1000), exp(-1000) ≈ 0 which is perfect.
    We must NOT clamp the input values - the loss naturally handles large values.
    """
    x = x.float()
    
    # Handle NaN only
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    
    # Get absolute values - DO NOT clamp!
    x_abs = x.abs()
    
    # Only clamp the exponent to prevent exp() underflow
    # exp(-88) ≈ 1e-38 which is smallest positive float32
    x_abs_clamped = torch.clamp(x_abs, max=88.0)
    
    loss = torch.sum(torch.exp(-x_abs_clamped), dim=-1).mean()
    
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(1e6, device=x.device, dtype=x.dtype, requires_grad=True)
    
    return loss


class R1Module(nn.Module):
    """R1 rotation matrix with QR-Orth optimization (single layer)"""
    
    def __init__(self, hidden_size: int, device: torch.device, mode: str = 'hadamard'):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Initialize with Hadamard (recommended) or random
        init_matrix = hadamard_utils.get_orthogonal_matrix(hidden_size, mode, device)
        self.Z = nn.Parameter(init_matrix)
    
    def get_rotation(self) -> torch.Tensor:
        """Get orthogonal rotation matrix via QR decomposition"""
        Q, R = torch.linalg.qr(self.Z, mode='complete')
        diag_sign = torch.sign(torch.diag(R))
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        Q = Q * diag_sign.unsqueeze(0)
        return Q
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation to activations"""
        R = self.get_rotation()
        return torch.matmul(x, R)


class JointR1Module(nn.Module):
    """
    Joint R1 training: Learns ONE global rotation for ALL layers.
    This is the key innovation of our approach.
    
    Unlike greedy layer-by-layer training (original DartQuant),
    we optimize a single rotation matrix that minimizes Whip loss
    across all layers simultaneously.
    """
    
    def __init__(self, hidden_size: int, num_layers: int, device: torch.device,
                 mode: str = 'hadamard'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # SINGLE global matrix for the whole model
        self.Z = nn.Parameter(hadamard_utils.get_orthogonal_matrix(hidden_size, mode, device))
        
        # Optional layer weights (can be used to down-weight problematic layers)
        self.layer_weights = nn.Parameter(torch.ones(num_layers), requires_grad=False)
    
    def get_rotation(self) -> torch.Tensor:
        """Get the single global rotation matrix"""
        Q, R = torch.linalg.qr(self.Z, mode='complete')
        diag_sign = torch.sign(torch.diag(R))
        diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
        Q = Q * diag_sign.unsqueeze(0)
        return Q
    
    def forward(self, activations_dict: Dict[int, torch.Tensor]) -> Tuple[torch.Tensor, Dict[int, float]]:
        """
        Compute joint loss across all layers using a SINGLE rotation.
        
        Args:
            activations_dict: {layer_idx: activations_tensor}
        
        Returns:
            total_loss: Weighted sum of per-layer losses
            per_layer_losses: Dictionary of per-layer loss values
        """
        total_loss = 0
        per_layer_losses = {}
        
        R = self.get_rotation()
        
        for layer_idx, acts in activations_dict.items():
            rotated = torch.matmul(acts, R)
            layer_loss = whip_loss(rotated)
            
            weight = self.layer_weights[layer_idx] if layer_idx < len(self.layer_weights) else 1.0
            total_loss = total_loss + weight * layer_loss
            per_layer_losses[layer_idx] = layer_loss.item()
        
        return total_loss, per_layer_losses


class R2Module(nn.Module):
    """R2 per-head rotation matrix with QR-Orth optimization"""
    
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 device: torch.device, mode: str = 'hadamard'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        # Initialize Z matrices for each KV head
        init_matrix = hadamard_utils.get_orthogonal_matrix(self.head_dim, mode, device)
        self.Z = nn.Parameter(init_matrix.unsqueeze(0).repeat(num_kv_heads, 1, 1))
    
    def get_rotation(self) -> torch.Tensor:
        """Get orthogonal rotation matrices via QR decomposition"""
        rotations = []
        for i in range(self.num_kv_heads):
            Q, R = torch.linalg.qr(self.Z[i], mode='complete')
            diag_sign = torch.sign(torch.diag(R))
            diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
            Q = Q * diag_sign.unsqueeze(0)
            rotations.append(Q)
        return torch.stack(rotations)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-head rotation to activations"""
        original_shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        x = x.reshape(-1, self.num_heads, self.head_dim)
        x = x.transpose(0, 1)  # [num_heads, tokens, head_dim]
        
        R = self.get_rotation()  # [num_kv_heads, head_dim, head_dim]
        
        # Expand rotation for GQA
        heads_per_kv = self.num_heads // self.num_kv_heads
        R_expanded = R.unsqueeze(1).expand(-1, heads_per_kv, -1, -1)
        R_expanded = R_expanded.reshape(self.num_heads, self.head_dim, self.head_dim)
        
        x_rotated = torch.matmul(x, R_expanded)
        x_rotated = x_rotated.transpose(0, 1)
        x_rotated = x_rotated.reshape(original_shape)
        
        return x_rotated


def train_r1_joint(
    activations: Dict[str, torch.Tensor],
    hidden_size: int,
    num_layers: int,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
    momentum: float = 0.9,
    init_mode: str = 'hadamard',
    use_cosine_lr: bool = True
) -> torch.Tensor:
    """
    Train R1 rotation matrix jointly across all layers.
    
    This is the key innovation: instead of training layer-by-layer (greedy),
    we optimize a single global rotation that works well for ALL layers.
    
    Args:
        activations: Dict of {layer_name: activation_tensor}
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        device: Training device
        lr: Learning rate (default 1e-3, matching official DartQuant)
        epochs: Number of training epochs
        batch_size: Batch size
        momentum: SGD momentum
        init_mode: 'hadamard' (recommended) or 'random'
        use_cosine_lr: Whether to use cosine annealing LR
    
    Returns:
        Trained R1 rotation matrix [hidden_size, hidden_size]
    """
    logging.info("Training R1 (Joint Optimization)...")
    logging.info(f"  LR: {lr}, Epochs: {epochs}, Init: {init_mode}")
    
    # Convert activations to indexed format and subsample
    layer_names = sorted(activations.keys())
    acts_dict = {}
    max_samples = 128 * 256  # Match official nsamples * tokens_per_sample
    
    for idx, name in enumerate(layer_names):
        acts = activations[name]
        if acts.shape[0] > max_samples:
            indices = torch.randperm(acts.shape[0])[:max_samples]
            acts = acts[indices]
        acts = acts.to(device).float()
        # Only handle NaN, preserve outliers!
        acts = torch.nan_to_num(acts, nan=0.0)
        acts_dict[idx] = acts
    
    # Create joint module
    joint_module = JointR1Module(hidden_size, len(layer_names), device, init_mode).to(device)
    
    # Optimizer (SGD with momentum, matching official DartQuant)
    optimizer = torch.optim.SGD(joint_module.parameters(), lr=lr, momentum=momentum)
    
    if use_cosine_lr:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    # Training loop
    joint_module.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        total_loss, per_layer = joint_module(acts_dict)
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.warning(f"Epoch {epoch}: NaN/Inf loss, skipping...")
            continue
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(joint_module.parameters(), 1.0)
        optimizer.step()
        
        if use_cosine_lr:
            scheduler.step()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            layer_str = " | ".join([f"L{idx}:{per_layer[idx]:.1f}" for idx in sorted(per_layer.keys())[:5]])
            lr_now = scheduler.get_last_lr()[0] if use_cosine_lr else lr
            logging.info(f"  Epoch {epoch+1}/{epochs} | LR: {lr_now:.2e} | {layer_str}...")
    
    # Extract trained rotation
    R1 = joint_module.get_rotation().detach().cpu()
    
    logging.info("R1 training complete!")
    return R1


# ============================================================================
# WRAPPER FUNCTIONS FOR EASY CALLING WITH CONFIG
# ============================================================================

def train_r1_joint_from_model(
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    config
) -> Dict[str, torch.Tensor]:
    """
    Wrapper to train R1 jointly using model and config.
    
    Args:
        model: The model (used to extract hidden_size, num_layers)
        activations: Dict of {layer_name: activation_tensor}
        config: JointQuantConfig object
    
    Returns:
        Dict with 'global' R1 rotation and per-layer entries
    """
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    num_layers = len(model.model.layers)
    
    R1 = train_r1_joint(
        activations=activations,
        hidden_size=hidden_size,
        num_layers=num_layers,
        device=device,
        lr=config.joint_lr if hasattr(config, 'joint_lr') else config.r1_lr,
        epochs=config.joint_epochs if hasattr(config, 'joint_epochs') else 100,
        batch_size=config.r1_batch_size,
        momentum=config.r1_momentum,
        init_mode=config.r1_init_mode,
        use_cosine_lr=config.r1_use_cosine_lr,
    )
    
    # Create dict with global R1 and per-layer entries
    layer_names = sorted(activations.keys())
    R1_dict = {name: R1 for name in layer_names}
    R1_dict['global'] = R1
    
    return R1_dict


def train_r2_from_model(
    model: nn.Module,
    activations: Dict[str, torch.Tensor],
    config
) -> Dict[str, torch.Tensor]:
    """
    Wrapper to train R2 using model and config.
    
    Args:
        model: The model (used to extract head info)
        activations: Dict of {layer_name: activation_tensor}
        config: JointQuantConfig object
    
    Returns:
        Dict of {layer_name: R2_tensor}
    """
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    
    return train_r2_independent(
        activations=activations,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        device=device,
        lr=config.r2_lr,
        epochs=config.r2_epochs,
        batch_size=config.r2_batch_size,
        momentum=config.r2_momentum,
        init_mode=config.r1_init_mode,
        use_cosine_lr=config.r2_use_cosine_lr,
    )


def train_r2_independent(
    activations: Dict[str, torch.Tensor],
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    device: torch.device,
    lr: float = 1e-3,
    epochs: int = 5,
    batch_size: int = 64,
    momentum: float = 0.9,
    init_mode: str = 'hadamard',
    use_cosine_lr: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Train R2 rotation matrices independently for each layer.
    (Same as original DartQuant - per-head, per-layer)
    
    Args:
        activations: Dict of {layer_name: activation_tensor}
        hidden_size: Model hidden size
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        device: Training device
        lr: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        momentum: SGD momentum
        init_mode: 'hadamard' or 'random'
        use_cosine_lr: Whether to use cosine annealing LR
    
    Returns:
        Dict of {layer_name: R2_tensor [num_kv_heads, head_dim, head_dim]}
    """
    logging.info("Training R2 (Per-Head, Per-Layer)...")
    
    R2_matrices = {}
    
    for layer_name, acts in tqdm(activations.items(), desc="Training R2"):
        # Create module
        r2_module = R2Module(hidden_size, num_heads, num_kv_heads, device, init_mode).to(device)
        
        # Subsample activations
        max_samples = 128 * 256
        if acts.shape[0] > max_samples:
            indices = torch.randperm(acts.shape[0])[:max_samples]
            acts = acts[indices]
        acts = acts.to(device).float()
        acts = torch.nan_to_num(acts, nan=0.0)
        
        # Create dataloader
        dataset = ActivationDataset(acts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.SGD(r2_module.parameters(), lr=lr, momentum=momentum)
        if use_cosine_lr:
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        
        # Training
        r2_module.train()
        for epoch in range(epochs):
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                rotated = r2_module(batch)
                loss = whip_loss(rotated)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(r2_module.parameters(), 1.0)
                optimizer.step()
            
            if use_cosine_lr:
                scheduler.step()
        
        # Save rotation
        R2_matrices[layer_name] = r2_module.get_rotation().detach().cpu()
        
        del r2_module, optimizer
        if use_cosine_lr:
            del scheduler
        torch.cuda.empty_cache()
    
    logging.info(f"R2 training complete! Trained {len(R2_matrices)} layers.")
    return R2_matrices


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_modules: List[str],
    device: torch.device,
    batch_size: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Collect activations from specified modules.
    
    CRITICAL: Do NOT clamp or truncate outliers!
    Llama models have activation outliers reaching 1000+.
    R1 rotation's purpose is to "spread out" these outliers.
    """
    activations = {name: [] for name in target_modules}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            tensor = inp[0] if isinstance(inp, tuple) else inp
            flat = tensor.detach().float().cpu().reshape(-1, tensor.shape[-1])
            # CRITICAL: Only handle NaN, preserve outliers!
            flat = torch.nan_to_num(flat, nan=0.0)
            # Subsample tokens
            if flat.shape[0] > 256:
                indices = torch.randperm(flat.shape[0])[:256]
                flat = flat[indices]
            activations[name].append(flat)
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if name in target_modules:
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run forward passes
    model.eval()
    for i in tqdm(range(0, len(input_ids), batch_size), desc="Collecting activations"):
        batch = input_ids[i:i+batch_size].to(device)
        with torch.no_grad():
            model(batch)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Concatenate
    for name in activations:
        if activations[name]:
            activations[name] = torch.cat(activations[name], dim=0)
    
    return activations


@torch.no_grad()
def collect_activations_from_model(
    model: nn.Module,
    calibration_data: torch.Tensor,
    collect_mlp: bool = True,
    batch_size: int = 4
) -> Dict[str, torch.Tensor]:
    """
    Simpler interface to collect activations from model.
    
    Automatically determines target modules based on model architecture.
    
    Args:
        model: The model to collect from
        calibration_data: Input token IDs [nsamples, seqlen]
        collect_mlp: Whether to also collect MLP activations (for SmoothQuant)
        batch_size: Batch size for forward passes
    
    Returns:
        Dict of {module_name: activation_tensor}
    """
    device = next(model.parameters()).device
    num_layers = len(model.model.layers)
    
    # Build list of target modules
    target_modules = []
    for i in range(num_layers):
        # Always collect attention input (for R1/R2 training)
        target_modules.append(f"model.layers.{i}.self_attn.q_proj")
        
        if collect_mlp:
            # Also collect MLP input (for SmoothQuant)
            target_modules.append(f"model.layers.{i}.mlp.up_proj")
    
    return collect_activations(
        model=model,
        input_ids=calibration_data,
        target_modules=target_modules,
        device=device,
        batch_size=batch_size,
    )
