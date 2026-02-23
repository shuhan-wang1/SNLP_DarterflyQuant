"""
Butterfly Givens Rotation for learnable R3 and R4.

Replaces fixed random Hadamard matrices with learnable structured rotations
that maintain O(d log d) complexity while adapting to the data distribution.

Key properties:
  - K = log2(d) layers, each with d/2 independent Givens rotations
  - Total learnable parameters: (d/2) * log2(d)
  - Always orthogonal by construction (product of Givens rotations)
  - Identity initialization (angles = 0); Hadamard warm-start done at
    training time via _init_butterfly_from_hadamard() in trainers.py

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
        # Initialize to 0 (identity); Hadamard warm-start is applied by
        # _init_butterfly_from_hadamard() in trainers.py at training time.
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
    butterfly rotation to the m-dimensional sub-blocks.  For the
    K-dimensional cross-block mixing factor, two parameterizations are
    available:

    **Latent QR-Orth** (default, ``k_factor_mode='latent'``):
        An unconstrained latent matrix ``Z ∈ R^{K×K}`` is optimized,
        and the orthogonal factor is extracted via QR decomposition
        ``Q, _ = torch.linalg.qr(Z)`` at each forward pass.  This is
        the same OR-Orth approach used by R1/R2 (Algorithm 1 in
        Section 2.3 of the report).  It is numerically robust and
        computationally cheap for the small K values encountered in
        practice (e.g. K = 40 for dim = 5120).

    **Cayley** (``k_factor_mode='cayley'``):
        ``Q = (I − A)(I + A)^{-1}`` where ``A`` is skew-symmetric.
        Guarantees strict orthogonality but requires an O(K³) matrix
        solve per forward pass.  May suffer from gradient instability
        when eigenvalues of ``(I + A)`` approach zero.

    For R4 where intermediate_size may not be power of 2.

    Args:
        total_dim: The full dimension (may not be power of 2)
        k_factor_mode: Parameterization for the K-dimensional factor.
            ``'latent'`` (default) — unconstrained matrix + QR (OR-Orth).
            ``'cayley'`` — Cayley transform with skew-symmetric param.
    """

    def __init__(self, total_dim: int, k_factor_mode: str = 'latent'):
        super().__init__()
        self.total_dim = total_dim
        self.k_factor_mode = k_factor_mode

        if total_dim > 0 and (total_dim & (total_dim - 1)) == 0:
            # Pure power of 2 - use standard butterfly directly
            self.K = 1
            self.m = total_dim
            self.butterfly = ButterflyRotation(total_dim)
        else:
            # Factor as K * m where m = total_dim / K is power of 2
            self.K, self.m = self._find_factorization(total_dim)
            self.butterfly = ButterflyRotation(self.m)

            if k_factor_mode == 'latent':
                # OR-Orth: unconstrained latent matrix, orthogonal Q
                # extracted via QR at each forward pass.
                # Identity initialisation => training starts with K-dim
                # unmixed, matching the Cayley A=0 => Q=I behaviour.
                self.latent_matrix = nn.Parameter(
                    torch.eye(self.K, self.K))
            elif k_factor_mode == 'cayley':
                # Cayley: skew-symmetric param, A=0 => Q=I.
                self.cayley_A = nn.Parameter(
                    torch.zeros(self.K, self.K))
            else:
                raise ValueError(
                    f"Unknown k_factor_mode '{k_factor_mode}'. "
                    "Choose 'latent' or 'cayley'.")

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

    def _get_cayley_Q(self) -> torch.Tensor:
        """Compute orthogonal K x K matrix via Cayley parameterization.

        Q = (I + A_skew)^{-1} (I - A_skew)

        where A_skew = (A - A^T) / 2 is the skew-symmetric part of
        self.cayley_A.  This guarantees Q is orthogonal for any real A.
        torch.linalg.solve is used for numerical stability and is
        auto-differentiable, so gradients flow back to cayley_A.
        """
        A_skew = (self.cayley_A - self.cayley_A.T) / 2.0
        I = torch.eye(self.K, device=self.cayley_A.device,
                       dtype=self.cayley_A.dtype)
        # solve(I + A, I - A) = (I + A)^{-1} (I - A)
        Q = torch.linalg.solve(I + A_skew, I - A_skew)
        return Q

    def _get_latent_Q(self) -> torch.Tensor:
        """Compute orthogonal K x K matrix via QR decomposition (OR-Orth).

        An unconstrained latent matrix Z is maintained as nn.Parameter.
        At each forward pass, ``Q, _ = torch.linalg.qr(Z)`` extracts the
        orthogonal factor.  torch.linalg.qr is auto-differentiable, so
        gradients flow back through Z.

        This is the same approach used by R1 (R1_QR) and R2 (R2_Per_Head)
        in trainers.py, extended here to the K-dimensional cross-block
        factor of the factored butterfly.
        """
        Q, _ = torch.linalg.qr(self.latent_matrix)
        return Q

    def _get_Q(self) -> torch.Tensor:
        """Dispatch to the appropriate K-factor parameterization."""
        if self.k_factor_mode == 'cayley':
            return self._get_cayley_Q()
        else:
            return self._get_latent_Q()

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

        # Apply learnable K-factor to the K dimension
        # x: (batch, K, m) -> Q_K @ x over K dim
        Q_K = self._get_Q().to(x.dtype)
        x = torch.einsum('ij,bjk->bik', Q_K, x)

        # Normalize
        x = x / math.sqrt(self.total_dim)

        return x.reshape(orig_shape)

    def inverse_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply R^T (inverse rotation) to input tensor.

        Reverses the three steps of forward():
          1. Undo normalization (multiply by sqrt(total_dim))
          2. Apply Q_K^T to the K dimension (orthogonal)
          3. Apply butterfly.inverse_forward to the m dimension
        """
        if self.K == 1:
            return self.butterfly.inverse_forward(x)

        orig_shape = x.shape
        x = x.reshape(-1, self.K, self.m)

        # Step 1: undo normalization
        x = x * math.sqrt(self.total_dim)

        # Step 2: apply Q_K^T over the K dimension (Q is orthogonal)
        Q_K = self._get_Q().to(x.dtype)
        x = torch.einsum('ji,bjk->bik', Q_K, x)  # Q_K.T

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
