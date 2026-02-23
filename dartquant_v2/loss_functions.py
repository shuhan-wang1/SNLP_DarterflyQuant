"""
Loss functions for rotation matrix training.

Five loss functions:
  - whip:     Original DartQuant Whip loss (exp-based repulsion from zero)
  - swd_unif: Sliced Wasserstein Distance to Uniform distribution
  - swd_gauss: Sliced Wasserstein Distance to Gaussian distribution
  - kl_unif:  KL divergence to Uniform via differential-entropy maximization
  - kl_gauss: KL divergence to Gaussian via moment matching (skewness + kurtosis)

Recommended pairings:
  INT4 quantizer → whip / swd_unif / kl_unif
  NF4  quantizer → swd_gauss / kl_gauss
"""

import math
import torch
import torch.nn.functional as F


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
    """Sliced Wasserstein Distance to Uniform[-b, b] distribution.

    L = (1/N) * sum_i (y_sorted[i] - t[i])^2
    where t[i] = linspace(-b, b, N), b = sqrt(3) * RMS(x)

    The boundary b is chosen to match the second moment of the input
    (energy conservation under orthogonal rotation).

    Pairs naturally with INT4 uniform quantizer.

    Reference: docs/SNLP_report_1_v1_en.md Section 3.1
    """
    x_flat = outputs.reshape(-1)
    N = x_flat.numel()

    x_sorted, _ = torch.sort(x_flat)

    with torch.no_grad():
        rms = torch.sqrt(torch.mean(x_flat ** 2))
        b = math.sqrt(3) * rms
        target = torch.linspace(-b.item(), b.item(), steps=N, device=outputs.device)

    loss = F.mse_loss(x_sorted, target)
    return loss


def calc_swd_gauss_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Gaussian SWD Loss - matches distribution to N(0, sigma_hat^2).

    L = (1/n) * sum_i (x_sorted[i] - q[i])^2
    where q[i] = Phi^{-1}((i - 0.5)/n) * sigma_hat
          sigma_hat = sqrt(mean(x^2))
          Phi^{-1} is the inverse CDF of the standard normal

    Uses scale-invariant sigma_hat so only the distribution shape is matched,
    not the absolute scale.

    Pairs naturally with NF4 (Normal Float 4-bit) quantizer which assumes
    Gaussian-distributed inputs.

    Reference: docs/report_2_en.md Definition 17
    """
    x_flat = outputs.reshape(-1)
    n = x_flat.numel()

    x_sorted, _ = torch.sort(x_flat)

    with torch.no_grad():
        sigma_hat = torch.sqrt(torch.mean(x_flat ** 2))
        # Compute standard normal quantiles: Phi^{-1}((i - 0.5) / n)
        # Using: Phi^{-1}(p) = sqrt(2) * erfinv(2p - 1)
        probs = (torch.arange(1, n + 1, device=outputs.device, dtype=torch.float32) - 0.5) / n
        # Clamp to avoid inf at boundaries
        probs = probs.clamp(1e-6, 1 - 1e-6)
        quantiles = math.sqrt(2) * torch.erfinv(2 * probs - 1) * sigma_hat

    loss = F.mse_loss(x_sorted, quantiles)
    return loss


# ============================================================================
# KL divergence losses (for butterfly training)
# ============================================================================

def calc_kl_unif_loss(outputs: torch.Tensor) -> torch.Tensor:
    """KL divergence to Uniform distribution via differential-entropy maximization.

    The KL divergence from P to Uniform[-b, b] decomposes as:

        KL(P || Unif[-b,b]) = log(2b) - H(P)

    Since the target entropy log(2b) is constant under orthogonal rotation
    (b = sqrt(3)*RMS(x) is rotation-invariant), minimizing KL reduces to
    maximizing the differential entropy H(P).

    H(P) is estimated using the Vasicek (1976) spacing estimator, which is
    differentiable through torch.sort:

        H(P) ≈ (1/n) Σ_i log(n · Δx_i)

    where Δx_i are symmetric spacings between consecutive order statistics:
        Δx_i = (x_(i+1) - x_(i-1)) / 2   for interior points
        Δx_0 = x_(1) - x_(0)              (left boundary)
        Δx_{n-1} = x_(n-1) - x_(n-2)     (right boundary)

    The gradient w.r.t. x_(i) equalises neighbouring spacings, which
    iteratively spreads samples to achieve uniform coverage — exactly
    what INT4 uniform quantization needs.

    Pairs naturally with INT4 uniform quantizer.
    """
    x_flat = outputs.reshape(-1).float()
    n = x_flat.numel()

    x_sorted, _ = torch.sort(x_flat)

    # Symmetric spacings
    spacings = torch.empty(n, device=x_flat.device, dtype=x_flat.dtype)
    spacings[1:-1] = (x_sorted[2:] - x_sorted[:-2]) / 2.0
    spacings[0]    = x_sorted[1] - x_sorted[0]
    spacings[-1]   = x_sorted[-1] - x_sorted[-2]

    # Guard against numerically identical samples
    spacings = spacings.clamp(min=1e-8)

    # Negative Vasicek entropy estimator (minimise to maximise H)
    neg_entropy = -(torch.log(n * spacings)).mean()
    return neg_entropy


def calc_kl_gauss_loss(outputs: torch.Tensor) -> torch.Tensor:
    """KL divergence to Gaussian distribution via Gaussian moment matching.

    By the Gram-Charlier A expansion a distribution P can be written as a
    perturbation of the Gaussian φ, and to leading order:

        KL(P || N(0, σ²)) ≈ γ₁² / 12  +  γ₂² / 96  + O(γ³)

    where γ₁ is the skewness and γ₂ is the excess kurtosis.  Minimising
    the surrogate loss

        L = γ₁²  +  γ₂²

    is therefore equivalent to minimising the KL divergence from Gaussian
    to leading order.  A Gaussian has γ₁ = 0 and γ₂ = 0, so L = 0 iff
    the distribution is Gaussian.

    Starting from the heavy-tailed Laplacian-like activations typical of
    LLMs (γ₁ ≈ 0, γ₂ ≈ 3), the gradient reduces kurtosis toward 0 by
    spreading the peak and lightening the tails — transforming the
    distribution toward the Gaussian shape that NF4 is optimised for.

    Pairs naturally with NF4 (Normal Float 4-bit) quantizer.
    """
    x = outputs.reshape(-1).float()
    mu = x.mean()
    centered = x - mu

    sigma2 = (centered ** 2).mean()
    sigma  = torch.sqrt(sigma2 + 1e-8)  # avoid division by zero

    skewness       = (centered ** 3).mean() / (sigma ** 3)
    excess_kurtosis = (centered ** 4).mean() / (sigma ** 4) - 3.0

    return skewness ** 2 + excess_kurtosis ** 2


# ============================================================================
# Registry
# ============================================================================

_LOSS_REGISTRY = {
    'whip':      calc_whip_loss,
    'swd_unif':  calc_swd_unif_loss,
    'swd_gauss': calc_swd_gauss_loss,
    'kl_unif':   calc_kl_unif_loss,
    'kl_gauss':  calc_kl_gauss_loss,
}


def get_loss_fn(loss_name: str):
    """Get loss function by name.

    Args:
        loss_name: One of 'whip', 'swd_unif', 'swd_gauss', 'kl_unif', 'kl_gauss'

    Returns:
        Callable loss function that takes (outputs: Tensor) -> Tensor
    """
    if loss_name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. Choose from {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[loss_name]
