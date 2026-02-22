"""
Butterfly Givens Rotation for learnable R3 and R4.

Replaces fixed random Hadamard matrices with learnable structured rotations
that maintain O(d log d) complexity while adapting to the data distribution.

Key properties:
  - K = log2(d) layers, each with d/2 independent Givens rotations
  - Total learnable parameters: (d/2) * log2(d)
  - Always orthogonal by construction (product of Givens rotations)
  - Identity initialization (all angles = 0) for stable training start

Reference: docs/report_2_en.md Section 2.1
"""

import math
import torch
import torch.nn as nn


class ButterflyRotation(nn.Module):
    """Learnable Butterfly Givens rotation matrix.

    Structure: K = log2(d) layers of butterfly Givens rotations.
    Each layer l pairs dimensions (i, i + 2^l) and applies independent
    Givens rotations G(theta) = [[cos(theta), sin(theta)],
                                  [-sin(theta), cos(theta)]].

    Args:
        dim: Dimension of the rotation matrix. Must be a power of 2.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0 or (dim & (dim - 1)) != 0:
            raise ValueError(
                f"ButterflyRotation requires power-of-2 dimension, got {dim}"
            )

        self.dim = dim
        self.num_layers = int(math.log2(dim))

        # (d/2) angles per butterfly layer, K = log2(d) layers
        # Total: (d/2) * log2(d) learnable parameters
        # Initialize to 0 (identity rotation)
        self.angles = nn.Parameter(torch.zeros(self.num_layers, dim // 2))

        # Pre-compute pairing indices for each layer
        self._register_pair_indices()

    def _register_pair_indices(self):
        """Pre-compute and register the butterfly pairing indices."""
        for l in range(self.num_layers):
            stride = 2 ** l
            block_size = 2 ** (l + 1)
            i_indices = []
            j_indices = []
            for i in range(self.dim):
                if i % block_size < stride:
                    i_indices.append(i)
                    j_indices.append(i + stride)
            self.register_buffer(
                f'_i_idx_{l}',
                torch.tensor(i_indices, dtype=torch.long)
            )
            self.register_buffer(
                f'_j_idx_{l}',
                torch.tensor(j_indices, dtype=torch.long)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply butterfly rotation to input tensor.

        Args:
            x: Tensor of shape (..., dim)

        Returns:
            Rotated tensor of same shape, complexity O(d log d)
        """
        orig_shape = x.shape
        x = x.reshape(-1, self.dim)  # (batch, dim)

        for l in range(self.num_layers):
            x = self._apply_layer(x, l)

        return x.reshape(orig_shape)

    def _apply_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply one butterfly layer of parallel Givens rotations.

        Args:
            x: (batch, dim) tensor
            layer_idx: which butterfly layer (0 to K-1)
        """
        thetas = self.angles[layer_idx]  # (dim//2,)
        cos_t = torch.cos(thetas)  # (dim//2,)
        sin_t = torch.sin(thetas)  # (dim//2,)

        i_idx = getattr(self, f'_i_idx_{layer_idx}')  # (dim//2,)
        j_idx = getattr(self, f'_j_idx_{layer_idx}')  # (dim//2,)

        xi = x[:, i_idx]  # (batch, dim//2)
        xj = x[:, j_idx]  # (batch, dim//2)

        # Givens rotation: [cos, sin; -sin, cos] @ [xi; xj]
        new_xi = cos_t * xi + sin_t * xj
        new_xj = -sin_t * xi + cos_t * xj

        result = x.clone()
        result[:, i_idx] = new_xi
        result[:, j_idx] = new_xj

        return result

    def inverse_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply R^T (inverse rotation) to input tensor.

        Since all Givens rotations are orthogonal, R^T = R^{-1}.
        The inverse is computed by applying layers in reverse order
        and transposing each Givens rotation (negate sin, keep cos).

        Args:
            x: Tensor of shape (..., dim)

        Returns:
            Inversely rotated tensor of same shape, complexity O(d log d)
        """
        orig_shape = x.shape
        x = x.reshape(-1, self.dim)

        for l in reversed(range(self.num_layers)):
            x = self._apply_layer_inverse(x, l)

        return x.reshape(orig_shape)

    def _apply_layer_inverse(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Apply the transpose of one butterfly layer.

        G(theta)^T = [[cos, -sin], [sin, cos]] (negate the off-diagonal sin terms).
        """
        thetas = self.angles[layer_idx]  # (dim//2,)
        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)

        i_idx = getattr(self, f'_i_idx_{layer_idx}')
        j_idx = getattr(self, f'_j_idx_{layer_idx}')

        xi = x[:, i_idx]
        xj = x[:, j_idx]

        # Transposed Givens: [cos, -sin; sin, cos] @ [xi; xj]
        new_xi = cos_t * xi - sin_t * xj
        new_xj = sin_t * xi + cos_t * xj

        result = x.clone()
        result[:, i_idx] = new_xi
        result[:, j_idx] = new_xj

        return result

    def get_matrix(self) -> torch.Tensor:
        """Compute the full d x d rotation matrix.

        This is used for offline baking into model weights (e.g., R4 into
        down_proj.weight). Complexity O(d^2) - only call during rotation
        application, not during inference.

        Returns:
            Orthogonal matrix of shape (dim, dim)
        """
        # Apply butterfly to identity matrix columns
        I = torch.eye(self.dim, device=self.angles.device, dtype=self.angles.dtype)
        return self.forward(I)


class ButterflyFactored(nn.Module):
    """Factored Butterfly rotation for non-power-of-2 dimensions.

    Handles dimensions like 11008 (Llama-2-7B intermediate_size) by
    decomposing as n = K * m where m is power of 2, and applying
    butterfly rotation to the m-dimensional sub-blocks while using a
    fixed factor matrix for the K-dimensional part.

    For R4 where intermediate_size may not be power of 2.

    Args:
        total_dim: The full dimension (may not be power of 2)
    """

    def __init__(self, total_dim: int):
        super().__init__()
        self.total_dim = total_dim

        if total_dim > 0 and (total_dim & (total_dim - 1)) == 0:
            # Pure power of 2 - use standard butterfly
            self.K = 1
            self.m = total_dim
            self.butterfly = ButterflyRotation(total_dim)
            self.register_buffer('hadK', torch.ones(1, 1))
        else:
            # Factor as K * m where m = total_dim / K is power of 2
            self.K, self.m = self._find_factorization(total_dim)
            self.butterfly = ButterflyRotation(self.m)
            # Use a fixed Hadamard-like matrix for the K-dimensional part
            self.register_buffer('hadK', self._get_hadamard_factor(self.K))

    @staticmethod
    def _find_factorization(n: int) -> tuple:
        """Find K, m such that n = K * m and m is power of 2.

        Tries common factors used in Hadamard decomposition.
        """
        # Common factors from hadamard_utils.py get_hadK()
        for K in [172, 156, 140, 108, 60, 52, 36, 28, 40, 20, 12]:
            if n % K == 0:
                m = n // K
                if m > 0 and (m & (m - 1)) == 0:
                    return K, m
        # Fallback: find smallest K such that n/K is power of 2
        for K in range(2, n + 1):
            if n % K == 0:
                m = n // K
                if m > 0 and (m & (m - 1)) == 0:
                    return K, m
        raise ValueError(f"Cannot factorize {n} as K * (power of 2)")

    @staticmethod
    def _get_hadamard_factor(K: int) -> torch.Tensor:
        """Get a K x K orthogonal matrix for the factor dimension."""
        # Generate via QR decomposition of random matrix
        torch.manual_seed(42)  # Deterministic for reproducibility
        M = torch.randn(K, K, dtype=torch.float64)
        Q, R = torch.linalg.qr(M)
        Q *= torch.sign(torch.diag(R)).unsqueeze(0)
        return Q.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply factored butterfly rotation.

        Args:
            x: Tensor of shape (..., total_dim)
        """
        if self.K == 1:
            return self.butterfly(x)

        orig_shape = x.shape
        x = x.reshape(-1, self.K, self.m)  # (batch, K, m)

        # Apply butterfly to the m dimension
        batch = x.shape[0]
        x = x.reshape(-1, self.m)  # (batch*K, m)
        x = self.butterfly(x)
        x = x.reshape(batch, self.K, self.m)

        # Apply hadK factor to the K dimension
        # x: (batch, K, m) -> hadK @ x over K dim
        hadK = self.hadK.to(x.device).to(x.dtype)
        x = torch.einsum('ij,bjk->bik', hadK, x)

        # Normalize
        x = x / math.sqrt(self.total_dim)

        return x.reshape(orig_shape)

    def inverse_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply R^T (inverse rotation) to input tensor.

        Reverses the three steps of forward():
          1. Undo normalization (multiply by sqrt(total_dim))
          2. Apply hadK^T to the K dimension
          3. Apply butterfly.inverse_forward to the m dimension
        """
        if self.K == 1:
            return self.butterfly.inverse_forward(x)

        orig_shape = x.shape
        x = x.reshape(-1, self.K, self.m)

        # Step 1: undo normalization
        x = x * math.sqrt(self.total_dim)

        # Step 2: apply hadK^T over the K dimension
        hadK = self.hadK.to(x.device).to(x.dtype)
        x = torch.einsum('ji,bjk->bik', hadK, x)  # hadK.T

        # Step 3: apply inverse butterfly to each m-dim sub-block
        batch = x.shape[0]
        x = x.reshape(-1, self.m)
        x = self.butterfly.inverse_forward(x)
        x = x.reshape(batch, self.K, self.m)

        return x.reshape(orig_shape)

    def get_matrix(self) -> torch.Tensor:
        """Compute full rotation matrix for offline baking."""
        I = torch.eye(self.total_dim, device=self.butterfly.angles.device,
                       dtype=self.butterfly.angles.dtype)
        return self.forward(I)
