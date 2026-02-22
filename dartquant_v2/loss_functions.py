"""
Loss functions for rotation matrix training.

Three loss functions:
  - whip: Original DartQuant Whip loss (exp-based repulsion from zero)
  - swd_unif: Sliced Wasserstein Distance to Uniform distribution
  - swd_gauss: Sliced Wasserstein Distance to Gaussian distribution
"""

import math
import torch
import torch.nn.functional as F


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


# Registry for loss function selection
_LOSS_REGISTRY = {
    'whip': calc_whip_loss,
    'swd_unif': calc_swd_unif_loss,
    'swd_gauss': calc_swd_gauss_loss,
}


def get_loss_fn(loss_name: str):
    """Get loss function by name.

    Args:
        loss_name: One of 'whip', 'swd_unif', 'swd_gauss'

    Returns:
        Callable loss function that takes (outputs: Tensor) -> Tensor
    """
    if loss_name not in _LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. Choose from {list(_LOSS_REGISTRY.keys())}"
        )
    return _LOSS_REGISTRY[loss_name]
