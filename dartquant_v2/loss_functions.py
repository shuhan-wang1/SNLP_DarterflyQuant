"""
Loss functions for rotation matrix training.

Seven loss functions:
  - whip:        Original DartQuant Whip loss (exp-based repulsion from zero)
  - swd_unif:    Sliced Wasserstein Distance to Uniform distribution
  - swd_gauss:   Sliced Wasserstein Distance to Gaussian distribution
  - kl_unif:     KL divergence to Uniform via differential-entropy maximization
  - kl_gauss:    KL divergence to Gaussian via moment matching (skewness + kurtosis)
  - bin_kl_unif: Discrete-bin KL to Uniform over INT4 quantizer levels (paper Eq 19)
  - bin_kl_nf4:  Discrete-bin KL to Uniform over NF4 quantizer levels (paper Eq 19)

Recommended pairings:
  INT4 quantizer → whip / swd_unif / kl_unif / bin_kl_unif
  NF4  quantizer → swd_gauss / kl_gauss / bin_kl_nf4
"""

import math
import torch


# ============================================================================
# Original losses
# ============================================================================

def calc_whip_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Original DartQuant Whip Loss.

    L = mean( sum_i exp(-|x_i|) )

    Pushes activation values away from zero, transforming Laplacian-like
    distributions toward more uniform shapes suitable for INT4 quantization.

    Reference: DartQuant/calibrater/r1_base_qr.py line 99
    """
    return torch.sum(torch.exp(-outputs.abs()), dim=-1, keepdim=True).mean()


def calc_swd_unif_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Sliced Wasserstein Distance to Uniform[-b, b] with GLOBAL scale.

    Each feature dimension is sorted independently, then compared against
    Uniform[-b, b] quantile targets where b = sqrt(3) * RMS_global.

    CRITICAL: b is computed from the GLOBAL RMS (all B×D values pooled),
    NOT per-dimension RMS.  A single global b ensures every dimension
    shares the same target range, preventing the rotation from
    concentrating variance into a few dimensions — which would create
    devastating outliers for INT4 quantization.

    (Per-dimension b_j allows dimensions to have wildly different scales;
    diagnostic experiments showed this causes absmax to jump from ~5.6
    (Hadamard init) to ~32.5, and cross-dim RMS σ from 0.11 to 0.44.)

    Note: the reference document (Section 3.1) actually describes global
    flattening of all B×D values into one vector with one sort, but
    per-dimension sort with global b achieves the same effect (all dims
    constrained to the same scale) while being more memory-efficient.

    Reduction: sum over D (feature dims), mean over B (batch).  This
    matches the Whip loss convention and keeps per-element gradients at
    O(1/B), independent of D.

    Pairs naturally with INT4 uniform quantizer.

    Reference: docs/SNLP_report_1_v1_en.md Section 3.1
    """
    D = outputs.shape[-1]
    x = outputs.reshape(-1, D).float()   # (B, D)
    B = x.shape[0]

    x_sorted, _ = torch.sort(x, dim=0)  # (B, D) — each column sorted independently

    with torch.no_grad():
        rms = torch.sqrt((x ** 2).mean())                # scalar — GLOBAL RMS
        b = math.sqrt(3) * rms                            # scalar — same for ALL dims
        t = torch.linspace(0, 1, steps=B, device=outputs.device)   # (B,)
        target = b * (2 * t.unsqueeze(1) - 1)             # (B, D) — uniform scale

    # sum over D (features), mean over B (batch) — matches Whip convention
    loss = (x_sorted - target).pow(2).sum(dim=-1).mean()
    return loss


def calc_swd_gauss_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Gaussian SWD Loss — per-dim normalisation + N(0,1) shape matching.

    Each feature dimension is normalised to unit variance (no_grad), then
    sorted and compared against standard N(0,1) quantiles.

    WHY NOT global sigma (like swd_unif uses global b)?
    Gaussian quantiles have heavy tails (±4.5σ).  If any dimension's RMS
    differs from the global RMS, the tail residuals explode, causing
    loss ~ 10⁶.  Uniform targets are bounded (±b), so global b works for
    swd_unif but NOT here.

    WHY NOT per-dim sigma (the original code)?
    Per-dim sigma_j lets the rotation concentrate variance into a few
    dimensions — each dim matches N(0, sigma_j²) regardless of sigma_j's
    magnitude, so the loss is blind to cross-dim variance uniformity.
    This is the same "cheating" pathology that broke swd_unif.

    THE FIX: divide each dimension by its own RMS (no_grad) to get unit-
    variance, then match against N(0,1).  Gradients still flow through x
    (only the normalisation constant is detached).  Because the scale is
    factored out, the rotation CANNOT reduce loss by concentrating
    variance — it can only reduce loss by improving per-dim Gaussianity,
    which is exactly what NF4 quantization needs.

    (This mirrors calc_bin_kl_nf4_loss, which already normalises per-dim
    by absmax before matching against the fixed NF4 codebook.)

    Reduction: sum over D (feature dims), mean over B (batch).

    Pairs naturally with NF4 (Normal Float 4-bit) quantizer.

    Reference: docs/report_2_en.md Definition 17
    """
    D = outputs.shape[-1]
    x = outputs.reshape(-1, D).float()   # (B, D)
    B = x.shape[0]

    # Per-dim normalisation to unit variance (no_grad on the scale factor).
    # Gradients still flow through x: ∂loss/∂x[i,j] = (1/rms_j) · ∂loss/∂z[i,j]
    with torch.no_grad():
        dim_rms = torch.sqrt((x ** 2).mean(dim=0)).clamp(min=1e-8)  # (D,)
    x_norm = x / dim_rms.unsqueeze(0)     # (B, D) — unit variance per dim

    x_sorted, _ = torch.sort(x_norm, dim=0)  # (B, D) — sort normalised values

    with torch.no_grad():
        probs = (torch.arange(1, B + 1, device=outputs.device, dtype=torch.float32) - 0.5) / B
        probs = probs.clamp(1e-6, 1 - 1e-6)             # (B,)
        std_quantiles = math.sqrt(2) * torch.erfinv(2 * probs - 1)  # (B,) — N(0,1) quantiles
        target = std_quantiles.unsqueeze(1)               # (B, 1) broadcast to (B, D)

    # sum over D (features), mean over B (batch) — matches Whip convention
    loss = (x_sorted - target).pow(2).sum(dim=-1).mean()
    return loss


# ============================================================================
# KL divergence losses (for butterfly training)
# ============================================================================

def calc_kl_unif_loss(outputs: torch.Tensor) -> torch.Tensor:
    """KL divergence to Uniform distribution via differential-entropy maximization,
    applied independently per feature dimension.

    For each dimension j, the Vasicek (1976) spacing estimator is applied to
    the B samples x[:,j], maximising the per-dimension differential entropy:

        H_j(P) ≈ (1/B) Σ_i log(B · Δx_i^j)

    where Δx_i^j are symmetric spacings of the sorted order statistics of x[:,j].

    The gradient equalises neighbouring spacings in each dimension independently,
    spreading each dimension's samples to achieve uniform coverage — exactly
    what INT4 uniform quantization needs.  Averaging over D dimensions enforces
    variance uniformity as a consequence.

    Pairs naturally with INT4 uniform quantizer.
    """
    D = outputs.shape[-1]
    x = outputs.reshape(-1, D).float()   # (B, D)
    B = x.shape[0]

    x_sorted, _ = torch.sort(x, dim=0)  # (B, D) — each column sorted independently

    # Symmetric spacings along the batch (sample) axis for all dims at once
    spacings = torch.empty_like(x_sorted)              # (B, D)
    spacings[1:-1] = (x_sorted[2:] - x_sorted[:-2]) / 2.0
    spacings[0]    = x_sorted[1] - x_sorted[0]
    spacings[-1]   = x_sorted[-1] - x_sorted[-2]

    spacings = spacings.clamp(min=1e-8)

    # Negative Vasicek entropy estimator averaged over B and D
    neg_entropy = -(torch.log(B * spacings)).mean()
    return neg_entropy


def calc_kl_gauss_loss(outputs: torch.Tensor) -> torch.Tensor:
    """KL divergence to Gaussian distribution via moment matching, per dimension.

    For each feature dimension j independently, the Gram-Charlier surrogate is:

        L_j = γ₁_j²  +  γ₂_j²

    where γ₁_j and γ₂_j are the skewness and excess kurtosis of x[:,j].
    The total loss is the mean across all D dimensions.

    Applying per-dimension moment matching ensures each dimension independently
    converges to a Gaussian shape, which is the prerequisite for NF4 optimality.

    Pairs naturally with NF4 (Normal Float 4-bit) quantizer.
    """
    D = outputs.shape[-1]
    x = outputs.reshape(-1, D).float()   # (B, D)

    mu = x.mean(dim=0)                   # (D,)
    centered = x - mu.unsqueeze(0)       # (B, D)

    sigma2 = (centered ** 2).mean(dim=0)              # (D,)
    sigma  = torch.sqrt(sigma2 + 1e-8)                # (D,)

    skewness        = (centered ** 3).mean(dim=0) / (sigma ** 3)        # (D,)
    excess_kurtosis = (centered ** 4).mean(dim=0) / (sigma ** 4) - 3.0  # (D,)

    return (skewness ** 2 + excess_kurtosis ** 2).mean()   # mean over D


# ============================================================================
# Discrete-bin KL divergence losses (paper Eq 19)
# ============================================================================

def calc_bin_kl_unif_loss(outputs: torch.Tensor,
                          num_levels: int = 15,
                          temperature: float = 50.0) -> torch.Tensor:
    """Discrete-bin KL divergence to Uniform over INT4 quantizer levels.

    Computes KL(P_bins || Uniform) where bins are INT4 levels scaled by
    b = sqrt(3) * RMS_global.

    CRITICAL: b is computed from the GLOBAL RMS (all B×D values pooled),
    NOT per-dimension RMS.  A single global b ensures every dimension
    shares the same quantiser grid, preventing variance concentration
    into a few dimensions (same fix as swd_unif — see that docstring).

    Uses a differentiable soft-histogram via sigmoid.

    Args:
        outputs:     Rotated activations, shape (*, D)
        num_levels:  Number of quantiser levels (default 15 for INT4 symmetric)
        temperature: Sigmoid temperature for soft binning (higher = sharper)

    Pairs naturally with INT4 uniform quantiser.
    """
    D = outputs.shape[-1]
    x = outputs.reshape(-1, D).float()   # (B, D)

    with torch.no_grad():
        rms = torch.sqrt((x ** 2).mean()).clamp(min=1e-8)         # scalar — GLOBAL RMS
        b = math.sqrt(3) * rms                                     # scalar
        # Vectorised edge construction — same grid for all dims
        base_levels = torch.linspace(-1.0, 1.0, steps=num_levels,
                                     device=x.device)             # (num_levels,)
        levels = base_levels * b                                   # (num_levels,) — scalar b
        midpoints = (levels[:-1] + levels[1:]) / 2.0              # (num_levels-1,)
        step = levels[1] - levels[0]                               # scalar
        neg_sentinel = (levels[0] - step).unsqueeze(0)            # (1,)
        pos_sentinel = (levels[-1] + step).unsqueeze(0)           # (1,)
        edges = torch.cat([neg_sentinel, midpoints, pos_sentinel], dim=0)  # (num_levels+1,)

    # Soft histogram: x is (B, D), edges is (num_levels+1,)
    # sigs[k, i, j] = sigmoid(T * (x[i,j] - edges[k]))
    sigs = torch.sigmoid(
        temperature * (x.unsqueeze(0) - edges.view(-1, 1, 1))    # (num_levels+1, B, D)
    )
    bin_probs = (sigs[:-1] - sigs[1:]).mean(dim=1)                # (num_levels, D)

    bin_probs = bin_probs.clamp(min=1e-8)
    bin_probs = bin_probs / bin_probs.sum(dim=0, keepdim=True)    # normalise per dim

    # KL(P_j || Uniform) for each dim j, then mean over D
    kl_per_dim = (bin_probs * torch.log(bin_probs * num_levels)).sum(dim=0)  # (D,)
    return kl_per_dim.mean()


def calc_bin_kl_nf4_loss(outputs: torch.Tensor,
                          temperature: float = 50.0) -> torch.Tensor:
    """Discrete-bin KL divergence to Uniform over NF4 quantizer levels, per dimension.

    Same approach as calc_bin_kl_unif_loss but uses the non-uniform NF4
    codebook levels (16 levels from the QLoRA paper, optimal for N(0,1)).
    Each dimension is independently normalised by its own absmax to match
    NF4's per-block [-1, 1] normalisation.

    When activations are Gaussian, the NF4 levels produce equal-probability
    bins, so KL(P_bins_j || Uniform) = 0.  Per-dimension normalisation ensures
    each column is independently driven toward Gaussian shape.

    Args:
        outputs:     Rotated activations, shape (*, D)
        temperature: Sigmoid temperature for soft binning (higher = sharper)

    Pairs naturally with NF4 (Normal Float 4-bit) quantiser.
    """
    D = outputs.shape[-1]
    x = outputs.reshape(-1, D).float()   # (B, D)

    # Per-dimension absmax normalisation (matching NF4 per-block convention)
    with torch.no_grad():
        absmax = x.abs().max(dim=0).values.clamp(min=1e-8)  # (D,)
    x_norm = x / absmax.unsqueeze(0)                         # (B, D) in [-1, 1]

    # NF4 quantisation levels (from nf4_quantizer.py / QLoRA paper)
    nf4_levels = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0,
    ], device=x.device, dtype=x.dtype)
    num_levels = nf4_levels.shape[0]   # 16

    # Bin edges: shared across dims (NF4 grid is fixed in [-1, 1])
    midpoints = (nf4_levels[:-1] + nf4_levels[1:]) / 2.0
    neg_sentinel = torch.tensor([-1.5], device=x.device, dtype=x.dtype)
    pos_sentinel = torch.tensor([1.5],  device=x.device, dtype=x.dtype)
    edges = torch.cat([neg_sentinel, midpoints, pos_sentinel])  # (17,)

    # Soft histogram: x_norm (B, D), edges (17,)
    # sigs[k, i, j] = sigmoid(T * (x_norm[i,j] - edges[k]))
    sigs = torch.sigmoid(
        temperature * (x_norm.unsqueeze(0) - edges.unsqueeze(1).unsqueeze(2))  # (17, B, D)
    )
    bin_probs = (sigs[:-1] - sigs[1:]).mean(dim=1)   # (16, D) — mean over B

    bin_probs = bin_probs.clamp(min=1e-8)
    bin_probs = bin_probs / bin_probs.sum(dim=0, keepdim=True)  # normalise per dim

    # KL(P_j || Uniform) for each dim j, then mean over D
    kl_per_dim = (bin_probs * torch.log(bin_probs * num_levels)).sum(dim=0)  # (D,)
    return kl_per_dim.mean()


# ============================================================================
# Registry
# ============================================================================

_LOSS_REGISTRY = {
    'whip':        calc_whip_loss,
    'swd_unif':    calc_swd_unif_loss,
    'swd_gauss':   calc_swd_gauss_loss,
    'kl_unif':     calc_kl_unif_loss,
    'kl_gauss':    calc_kl_gauss_loss,
    'bin_kl_unif': calc_bin_kl_unif_loss,
    'bin_kl_nf4':  calc_bin_kl_nf4_loss,
}


def get_loss_fn(loss_name: str):
    """Get loss function by name.

    Args:
        loss_name: One of 'whip', 'swd_unif', 'swd_gauss', 'kl_unif',
                   'kl_gauss', 'bin_kl_unif', 'bin_kl_nf4'

    Returns:
        Callable loss function that takes (outputs: Tensor) -> Tensor
    """
    if loss_name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. Choose from {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[loss_name]
