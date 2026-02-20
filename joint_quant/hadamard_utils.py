"""
Hadamard matrix utilities for JointQuant
Adapted from DartQuant/fake_quant/hadamard_utils.py
"""

import torch
import math
from typing import Tuple, Optional


def is_pow2(n: int) -> bool:
    """Check if n is a power of 2"""
    return (n & (n - 1) == 0) and (n > 0)


def get_hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    """
    Generate normalized Hadamard matrix of given size (must be power of 2).
    Uses recursive construction: H_2n = [[H_n, H_n], [H_n, -H_n]]
    """
    if size == 1:
        return torch.ones(1, 1, device=device, dtype=torch.float32)
    
    assert is_pow2(size), f"Size must be power of 2, got {size}"
    
    # Recursive construction
    n = 1
    H = torch.ones(1, 1, device=device, dtype=torch.float32)
    while n < size:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1)
        ], dim=0)
        n *= 2
    
    # Normalize to make it orthogonal
    return H / math.sqrt(size)


def random_hadamard_matrix(size: int, device: torch.device) -> torch.Tensor:
    """
    Generate a random Hadamard-based orthogonal matrix.
    H_random = H * D where D is a random diagonal sign matrix.
    
    This is the "flattest" rotation matrix, ideal as optimization starting point.
    See: https://cornell-relaxml.github.io/quip-sharp/
    """
    # Get base Hadamard matrix (already normalized)
    H = get_hadamard_matrix(size, device)
    
    # Multiply by random sign diagonal for randomization
    # This preserves the "flatness" property while adding randomness
    random_signs = torch.randint(0, 2, (size,), device=device, dtype=torch.float32) * 2 - 1
    H_random = H * random_signs.unsqueeze(0)  # Broadcast multiply columns
    
    return H_random


def random_orthogonal_matrix(size: int, device: torch.device) -> torch.Tensor:
    """
    Generate a random orthogonal matrix using QR decomposition.
    Uses float64 for numerical stability during generation.
    """
    # Generate on CPU with float64 for numerical stability
    random_matrix = torch.randn(size, size, dtype=torch.float64, device='cpu')
    q, r = torch.linalg.qr(random_matrix, mode='complete')
    
    # Ensure proper rotation (determinant +1)
    diag_sign = torch.sign(torch.diag(r))
    diag_sign = torch.where(diag_sign == 0, torch.ones_like(diag_sign), diag_sign)
    q *= diag_sign.unsqueeze(0)
    
    return q.float().to(device)


def get_orthogonal_matrix(size: int, mode: str, device: torch.device) -> torch.Tensor:
    """
    Get orthogonal matrix initialization.
    
    Args:
        size: Matrix dimension (must be power of 2 for hadamard mode)
        mode: 'hadamard' (recommended) or 'random'
        device: Target device
    
    Returns:
        Orthogonal matrix of shape (size, size)
    """
    if mode == 'hadamard':
        if is_pow2(size):
            return random_hadamard_matrix(size, device)
        else:
            print(f"  [Warning] Size {size} is not power of 2, falling back to random init")
            return random_orthogonal_matrix(size, device)
    elif mode == 'random':
        return random_orthogonal_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'hadamard' or 'random'.")


# ============================================================================
# Extended Hadamard matrices for non-power-of-2 dimensions
# These are used for specific model architectures
# ============================================================================

def get_had12() -> torch.Tensor:
    """12x12 Hadamard matrix"""
    return torch.tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1],
        [1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1],
        [1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
    ], dtype=torch.float32) / math.sqrt(12)


def get_hadK(n: int, transpose: bool = False) -> Tuple[Optional[torch.Tensor], int]:
    """
    Get Hadamard matrix and K factor for dimension n.
    Used for non-power-of-2 dimensions.
    
    Returns:
        (hadK, K) where hadK is the Hadamard matrix and K is the factor
    """
    hadK, K = None, None
    
    # Check various factors
    if n % 28 == 0 and is_pow2(n // 28):  # llama-3
        K = 28
        # For now, use identity - full implementation would need had28
        hadK = None
    elif n % 12 == 0 and is_pow2(n // 12):
        K = 12
        hadK = get_had12().T if transpose else get_had12()
    elif is_pow2(n):
        K = 1
        hadK = None
    else:
        # Default: use power of 2 closest factor
        K = 1
        hadK = None
    
    return hadK, K
