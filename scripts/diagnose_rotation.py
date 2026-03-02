#!/usr/bin/env python3
"""
Diagnostic script: compare how Whip vs SWD losses train the R1 rotation.

Uses the EXACT same optimizer/scheduler/batch-size/epoch auto-adjustments
as the real pipeline (pipeline.py lines 1254-1304), so the numbers you see
here are what the real run_all_experiments.py produces.

Outputs:
  - Per-epoch loss curves for each loss function
  - Per-epoch gradient norm of the latent rotation matrix
  - Frobenius distance between trained rotation and Hadamard init
  - Per-dimension activation statistics (RMS, skewness, kurtosis) after rotation
  - Histograms of rotated activations saved as PNG

Usage:
  python scripts/diagnose_rotation.py --model meta-llama/Llama-3.2-1B
  python scripts/diagnose_rotation.py --model meta-llama/Llama-3.2-1B --layers 0 1 2
"""

import os
import sys
import math
import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn

# ── Project path setup ──────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_DARTQUANT_FQ = os.path.join(_PROJECT_ROOT, "DartQuant", "fake_quant")
if _DARTQUANT_FQ not in sys.path:
    sys.path.insert(0, _DARTQUANT_FQ)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Re-usable QR-Orth R1 module (same as trainers.py)
# ═══════════════════════════════════════════════════════════════════════════

class R1_QR(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.matrix = nn.Parameter(torch.eye(dim))
        self.rotate = None

    def forward(self, x):
        self.rotate, _ = torch.linalg.qr(self.matrix, mode="complete")
        return torch.matmul(x, self.rotate)


def _get_hadamard(size, device):
    try:
        from hadamard_utils import random_hadamard_matrix
        return random_hadamard_matrix(size, device)
    except ImportError:
        # Pure-PyTorch Walsh-Hadamard fallback
        D = (torch.randint(0, 2, (size,)) * 2 - 1).float().to(device)
        x = torch.eye(size, device=device)
        h = 1
        while h < size:
            x = x.reshape(size, size // (2 * h), 2, h)
            a, b = x[:, :, 0, :].contiguous(), x[:, :, 1, :].contiguous()
            x = torch.stack([a + b, a - b], dim=2).reshape(size, size)
            h *= 2
        return (x / math.sqrt(size)) * D.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop with full diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def train_r1_diagnostic(
    acts,              # (N, D) float32
    hidden_size,
    loss_fn_name,
    *,
    # --- hyperparams exactly matching run_all_experiments.py defaults ---
    lr=1e-3,
    momentum=0.9,
    epochs=None,       # auto-set below
    batch_size=None,   # auto-set below
    cos_lr=None,       # auto-set below
    optim=None,        # auto-set below
    train_subset_size=0.1,
    device="cuda",
):
    """Train R1, returning (rotation, diagnostics_dict)."""
    from dartquant_v2.loss_functions import get_loss_fn

    loss_fn = get_loss_fn(loss_fn_name)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # ── Pipeline auto-adjustments (mirrors pipeline.py:1254-1304) ────────
    _DIST_LOSSES = ("swd_unif", "swd_gauss", "kl_unif", "kl_gauss",
                    "bin_kl_unif", "bin_kl_nf4")
    is_dist = loss_fn_name in _DIST_LOSSES

    if epochs is None:
        epochs = 30 if is_dist else 10
    if cos_lr is None:
        cos_lr = True if is_dist else False
    if optim is None:
        optim = "adam" if is_dist else "sgd"
    if batch_size is None:
        # Same auto-adjustment as pipeline.py:1428-1439
        _default_bs = 131072
        _subset_rows = max(1, int(acts.shape[0] * train_subset_size))
        if _subset_rows < _default_bs * 6:
            batch_size = max(64, _subset_rows // 6)
        else:
            batch_size = _default_bs

    log.info(f"  [{loss_fn_name}] epochs={epochs}, optim={optim}, lr={lr}, "
             f"cos_lr={cos_lr}, batch_size={batch_size}, "
             f"subset={train_subset_size:.0%}")

    # ── Data ─────────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader, TensorDataset, RandomSampler
    acts_f = acts.float()
    dataset = TensorDataset(acts_f)

    if train_subset_size < 1.0:
        num = max(1, int(len(dataset) * train_subset_size))
        idx = np.random.choice(len(dataset), size=num, replace=False)
        dataset = torch.utils.data.Subset(dataset, idx.tolist())

    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size,
                        pin_memory=True, num_workers=0)

    # ── Model ────────────────────────────────────────────────────────────
    R1 = R1_QR(hidden_size).to(device)
    H_init = _get_hadamard(hidden_size, device).float()
    R1.matrix.data = H_init.clone()
    R_init = H_init.clone()  # store for delta comparison

    from torch.optim.lr_scheduler import CosineAnnealingLR
    if optim == "adam":
        optimizer = torch.optim.Adam(R1.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(R1.parameters(), lr=lr, momentum=momentum)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0) if cos_lr else None

    # ── Training with diagnostics ────────────────────────────────────────
    diag = {
        "loss_curve": [],
        "grad_norm": [],
        "rotation_delta": [],
        "lr_curve": [],
    }

    R1.train()
    for epoch in range(epochs):
        losses = []
        grads = []
        for (batch,) in loader:
            batch = batch.to(device).reshape(-1, hidden_size)
            out = R1(batch)
            loss = loss_fn(out)
            loss.backward()

            # Capture gradient norm BEFORE optimizer step
            g = R1.matrix.grad
            if g is not None:
                grads.append(g.norm().item())

            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        if scheduler:
            scheduler.step()

        epoch_loss = np.mean(losses)
        epoch_grad = np.mean(grads) if grads else 0.0
        cur_lr = scheduler.get_last_lr()[0] if scheduler else lr

        # Rotation distance from Hadamard init
        with torch.no_grad():
            R_cur, _ = torch.linalg.qr(R1.matrix.data, mode="complete")
            delta = (R_cur - R_init).norm().item()

        diag["loss_curve"].append(epoch_loss)
        diag["grad_norm"].append(epoch_grad)
        diag["rotation_delta"].append(delta)
        diag["lr_curve"].append(cur_lr)

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            log.info(f"    Epoch [{epoch+1}/{epochs}]  loss={epoch_loss:.6f}  "
                     f"grad_norm={epoch_grad:.6e}  R_delta={delta:.4f}  "
                     f"lr={cur_lr:.4e}")

    diag["final_rotation"] = R1.rotate.data.detach().cpu()
    return R1.rotate.data.detach(), diag


# ═══════════════════════════════════════════════════════════════════════════
# Activation statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(x):
    """Compute per-dimension statistics. x: (N, D)."""
    x = x.float()
    mu = x.mean(dim=0)
    var = x.var(dim=0)
    rms = torch.sqrt((x ** 2).mean(dim=0))
    std = torch.sqrt(var + 1e-8)
    centered = x - mu
    skew = (centered ** 3).mean(dim=0) / (std ** 3 + 1e-8)
    kurt = (centered ** 4).mean(dim=0) / (std ** 4 + 1e-8) - 3.0
    absmax = x.abs().max(dim=0).values
    return {
        "rms_mean": rms.mean().item(),
        "rms_std": rms.std().item(),
        "skew_mean": skew.mean().item(),
        "skew_std": skew.std().item(),
        "skew_abs_mean": skew.abs().mean().item(),
        "kurt_mean": kurt.mean().item(),
        "kurt_std": kurt.std().item(),
        "kurt_abs_mean": kurt.abs().mean().item(),
        "absmax_mean": absmax.mean().item(),
        "absmax_max": absmax.max().item(),
        "flat_min": x.min().item(),
        "flat_max": x.max().item(),
        "flat_std": x.std().item(),
        # For histograms
        "_rms": rms.cpu().numpy(),
        "_skew": skew.cpu().numpy(),
        "_kurt": kurt.cpu().numpy(),
        "_flat": x.cpu().reshape(-1).numpy(),
    }


def print_stats_table(label, stats):
    print(f"\n  [{label}]")
    print(f"    RMS   mean={stats['rms_mean']:.4f}  std={stats['rms_std']:.4f}")
    print(f"    Skew  mean={stats['skew_mean']:.4f}  |mean|={stats['skew_abs_mean']:.4f}  "
          f"std={stats['skew_std']:.4f}")
    print(f"    Kurt  mean={stats['kurt_mean']:.4f}  |mean|={stats['kurt_abs_mean']:.4f}  "
          f"std={stats['kurt_std']:.4f}")
    print(f"    Abs-max  mean={stats['absmax_mean']:.4f}  max={stats['absmax_max']:.4f}")
    print(f"    Range    [{stats['flat_min']:.4f}, {stats['flat_max']:.4f}]  "
          f"std={stats['flat_std']:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def save_diagnostic_plots(all_diags, all_stats, output_dir):
    """Generate and save diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    loss_names = list(all_diags.keys())
    colors = {"whip": "tab:red", "swd_unif": "tab:blue", "swd_gauss": "tab:purple"}

    # ── 1. Training curves (2×2) ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("R1 Training Diagnostics: Whip vs SWD", fontsize=14)

    for name in loss_names:
        d = all_diags[name]
        c = colors.get(name, "tab:gray")
        axes[0, 0].plot(d["loss_curve"], label=name, color=c)
        axes[0, 1].plot(d["grad_norm"], label=name, color=c)
        axes[1, 0].plot(d["rotation_delta"], label=name, color=c)
        axes[1, 1].plot(d["lr_curve"], label=name, color=c)

    axes[0, 0].set_title("Loss per Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].set_yscale("log")

    axes[0, 1].set_title("Gradient Norm (latent matrix)")
    axes[0, 1].set_ylabel("||∇W||")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].set_yscale("log")

    axes[1, 0].set_title("||R_trained − R_init||_F")
    axes[1, 0].set_ylabel("Frobenius distance")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()

    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_ylabel("LR")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {path}")

    # ── 2. Activation histograms ─────────────────────────────────────────
    fig, axes = plt.subplots(1, len(all_stats), figsize=(6 * len(all_stats), 5))
    if len(all_stats) == 1:
        axes = [axes]
    fig.suptitle("Rotated Activation Histograms (all dims flattened)", fontsize=13)

    for ax, (label, st) in zip(axes, all_stats.items()):
        flat = st["_flat"]
        clip = np.percentile(np.abs(flat), 99.5)
        ax.hist(flat, bins=200, range=(-clip, clip), density=True, alpha=0.7,
                color=colors.get(label, "tab:gray"), label=label)
        ax.set_title(label)
        ax.set_xlabel("Activation value")
        ax.set_ylabel("Density")
        ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "activation_histograms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {path}")

    # ── 3. Per-dimension RMS / Skewness / Kurtosis distributions ─────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Per-Dimension Distribution Statistics After Rotation", fontsize=13)

    for label, st in all_stats.items():
        c = colors.get(label, "tab:gray")
        axes[0].hist(st["_rms"], bins=100, alpha=0.5, label=label, color=c, density=True)
        axes[1].hist(st["_skew"], bins=100, alpha=0.5, label=label, color=c, density=True)
        axes[2].hist(st["_kurt"], bins=100, alpha=0.5, label=label, color=c, density=True)

    axes[0].set_title("Per-dim RMS")
    axes[0].legend()
    axes[1].set_title("Per-dim Skewness")
    axes[1].legend()
    axes[2].set_title("Per-dim Excess Kurtosis")
    axes[2].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "per_dim_stats.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnose R1 rotation training: Whip vs SWD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    p.add_argument("--cache_dir", type=str,
                   default=os.environ.get("HF_HOME", "/root/autodl-tmp/huggingface"))
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--layers", type=int, nargs="+", default=[0],
                   help="Which layers to collect activations from (for speed)")
    p.add_argument("--losses", type=str, nargs="+",
                   default=["whip", "swd_unif"],
                   help="Loss functions to compare")
    p.add_argument("--output_dir", type=str, default="./diagnose_output",
                   help="Directory for diagnostic outputs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  R1 Rotation Diagnostic: Whip vs SWD")
    print("=" * 70)
    print(f"  Model:   {args.model}")
    print(f"  Losses:  {args.losses}")
    print(f"  Layers:  {args.layers}")
    print(f"  Device:  {device}")
    print()

    # ── Step 1: Load model ───────────────────────────────────────────────
    log.info("Loading model...")
    from dartquant_v2.unified_model import UnifiedQuantModel
    umodel = UnifiedQuantModel(args.model, args.hf_token, args.cache_dir)
    model = umodel.model
    hidden_size = umodel.hidden_size
    log.info(f"  hidden_size={hidden_size}, layers={umodel.num_layers}")

    # ── Step 2: Load calibration data ────────────────────────────────────
    log.info("Loading calibration data...")
    try:
        import data_utils
        trainloader = data_utils.get_loaders(
            "wikitext2", nsamples=args.nsamples, seed=args.seed,
            model=args.model, seqlen=model.seqlen, eval_mode=False)
        calib_data = torch.cat([b[0] for b in trainloader], dim=0)
    except ImportError:
        log.warning("data_utils not found, using random tokens")
        calib_data = torch.randint(0, model.config.vocab_size,
                                   (args.nsamples, args.seqlen))

    log.info(f"  Calibration: {calib_data.shape}")

    # ── Step 3: Fuse LayerNorms (same as pipeline step 2) ────────────────
    log.info("Fusing LayerNorms...")
    from dartquant_v2.pipeline import fuse_layer_norms, _untie_word_embeddings
    _untie_word_embeddings(umodel)
    fuse_layer_norms(umodel)

    # ── Step 4: Collect activations ──────────────────────────────────────
    log.info("Collecting activations...")
    model.to(device)

    layers_prefix = umodel.arch.layers_path
    target_names = []
    for li in args.layers:
        if umodel.arch.mlp_up_proj_attr:
            target_names.append(f"{layers_prefix}.{li}.{umodel.arch.mlp_up_proj_attr}")
        target_names.append(f"{layers_prefix}.{li}.{umodel.arch.q_proj_attr}")

    from dartquant_v2.pipeline import collect_activations
    all_acts = collect_activations(model, calib_data, target_names, device)

    all_tensors = [a.reshape(-1, hidden_size) for a in all_acts.values()]
    acts = torch.cat(all_tensors, dim=0)
    log.info(f"  Collected {acts.shape[0]} activation rows, dim={acts.shape[1]}")

    model.cpu()
    torch.cuda.empty_cache()

    # ── Step 5: Compute baseline stats (before rotation) ─────────────────
    log.info("Computing baseline (no rotation) statistics...")
    baseline_stats = compute_stats(acts)
    print_stats_table("Original (no rotation)", baseline_stats)

    # ── Step 6: Compute Hadamard-only stats ──────────────────────────────
    log.info("Computing Hadamard-only rotation statistics...")
    H = _get_hadamard(hidden_size, device).float()
    acts_gpu = acts.to(device).float()
    had_rotated = torch.matmul(acts_gpu, H).cpu()
    had_stats = compute_stats(had_rotated)
    print_stats_table("Hadamard only (no training)", had_stats)
    del had_rotated

    # ── Step 7: Train R1 with each loss and diagnose ─────────────────────
    all_diags = {}
    all_stats = {"original": baseline_stats, "hadamard": had_stats}

    for loss_name in args.losses:
        print()
        log.info(f"{'='*60}")
        log.info(f"Training R1 with loss={loss_name}")
        log.info(f"{'='*60}")

        # Use the SAME seed for fair comparison (same Hadamard init, same subset)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        rotation, diag = train_r1_diagnostic(
            acts, hidden_size, loss_name, device=str(device))

        all_diags[loss_name] = diag

        # Apply rotation to activations
        rotated = torch.matmul(acts_gpu, rotation.to(device)).cpu()
        st = compute_stats(rotated)
        all_stats[loss_name] = st
        print_stats_table(f"After {loss_name} rotation", st)

        # Final diagnostics
        print(f"\n  [{loss_name}] Final Training Summary:")
        print(f"    Loss:       {diag['loss_curve'][0]:.6f} → {diag['loss_curve'][-1]:.6f}")
        print(f"    Grad norm:  {diag['grad_norm'][0]:.6e} → {diag['grad_norm'][-1]:.6e}")
        print(f"    R_delta:    {diag['rotation_delta'][-1]:.4f}")

        del rotated

    del acts_gpu

    # ── Step 8: Summary comparison ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    header = f"  {'Config':<25} {'RMS μ':>8} {'RMS σ':>8} {'|Skew|':>8} {'|Kurt|':>8} {'AbsMax':>8} {'R_Δ':>8}"
    print(header)
    print("  " + "-" * 75)
    for label, st in all_stats.items():
        r_delta = "-"
        if label in all_diags:
            r_delta = f"{all_diags[label]['rotation_delta'][-1]:.4f}"
        print(f"  {label:<25} {st['rms_mean']:>8.4f} {st['rms_std']:>8.4f} "
              f"{st['skew_abs_mean']:>8.4f} {st['kurt_abs_mean']:>8.4f} "
              f"{st['absmax_max']:>8.2f} {r_delta:>8}")

    # ── Step 9: Save plots ───────────────────────────────────────────────
    log.info("\nSaving diagnostic plots...")
    save_diagnostic_plots(all_diags, all_stats, args.output_dir)

    # ── Step 10: Save raw diagnostics as .pt ─────────────────────────────
    diag_path = os.path.join(args.output_dir, "diagnostics.pt")
    torch.save({
        "diags": {k: {kk: vv for kk, vv in v.items() if kk != "final_rotation"}
                  for k, v in all_diags.items()},
        "stats": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                  for k, v in all_stats.items()},
    }, diag_path)
    log.info(f"  Saved: {diag_path}")

    print(f"\n  All outputs saved to: {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
