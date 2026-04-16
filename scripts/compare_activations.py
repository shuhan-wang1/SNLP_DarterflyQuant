#!/usr/bin/env python3
"""Compare the hidden-state activation distribution:
raw pretrained Llama vs. R1 trained with {whip, swd_unif, swd_gauss}.

Produces one PDF with four columns — raw / whip / swd_unif / swd_gauss —
each showing a histogram (top) and per-channel absmax bars (bottom) of the
input to a chosen layer's mlp.up_proj (hidden dim, where R1 lives).

Pipeline:
  1. Load model + WikiText-2 calibration (stock HF transformers + datasets).
  2. Hook one layer's mlp.up_proj, accumulate inputs in a single forward pass.
  3. For each loss, train a QR-parameterized R1 on the collected activations.
  4. Apply R1 by matrix multiplication: X @ R1  (no weight modification — this
     is the same distribution the weight-baked rotation would produce at the
     same hook point, since R1 is fused into the preceding projections).
  5. Plot 2x4 PDF.

Example:
    python scripts/compare_activations.py \
        --model meta-llama/Llama-3.2-1B --layer 12 \
        --out artifacts/activation_comparison.pdf
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Local-only HF cache (autodl / Myriad compute nodes have no internet) ────
# Matches dartquant_v2/run_quantize.py exactly.  Models live directly under
# HF_HOME (not HF_HOME/hub), so HF_HUB_CACHE = HF_HOME.  To allow downloads,
# export TRANSFORMERS_OFFLINE=0 / HF_DATASETS_OFFLINE=0 before running.
_DEFAULT_HF_HOME = "/root/autodl-tmp/huggingface"
_HF_HOME = os.environ.get("HF_HOME", _DEFAULT_HF_HOME)
os.environ.setdefault("HF_HOME",           _HF_HOME)
os.environ.setdefault("HF_HUB_CACHE",      _HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_HOME)
os.environ.setdefault("HF_DATASETS_CACHE",
                       os.environ.get("HF_DATASETS_CACHE",
                                      "/root/autodl-tmp/datasets"))
if os.environ.get("TRANSFORMERS_OFFLINE") != "0":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
if os.environ.get("HF_DATASETS_OFFLINE") != "0":
    os.environ["HF_DATASETS_OFFLINE"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dartquant_v2.loss_functions import get_loss_fn


# ---------------------------------------------------------------------------
# 1) WikiText-2 calibration
# ---------------------------------------------------------------------------
def load_wikitext_calib(tokenizer, n_samples: int = 16, seqlen: int = 2048,
                         seed: int = 0) -> torch.Tensor:
    """Return a (n_samples, seqlen) input_ids tensor sampled from WikiText-2 train."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.shape[0] < seqlen + 1:
        raise RuntimeError(f"Corpus too small: {ids.shape[0]} < {seqlen + 1}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, ids.shape[0] - seqlen - 1, size=n_samples)
    return torch.stack([ids[s:s + seqlen] for s in starts])


# ---------------------------------------------------------------------------
# 2) Activation capture (multi-layer: one forward pass → dict[layer -> acts])
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_activations_multi(model, calib: torch.Tensor,
                                target_modules: dict[int, nn.Module],
                                device: torch.device) -> dict[int, torch.Tensor]:
    """Hook every target module and return {layer_idx: (N_tokens, d)}."""
    bufs: dict[int, list[torch.Tensor]] = {i: [] for i in target_modules}
    handles: list = []

    def make_hook(layer_idx: int):
        def hook(_m, inp, _out):
            t = inp[0] if isinstance(inp, tuple) else inp
            bufs[layer_idx].append(
                t.detach().reshape(-1, t.shape[-1]).float().cpu()
            )
        return hook

    for layer_idx, mod in target_modules.items():
        handles.append(mod.register_forward_hook(make_hook(layer_idx)))

    model.eval()
    try:
        for i in range(calib.shape[0]):
            model(calib[i:i + 1].to(device))
    finally:
        for h in handles:
            h.remove()

    return {i: torch.cat(bufs[i], dim=0) for i in target_modules}


# ---------------------------------------------------------------------------
# 3) R1 training (QR-parameterized, trained ONLY against collected activations)
# ---------------------------------------------------------------------------
class R1_QR(nn.Module):
    """R1 = Q from QR(matrix). Identity init; optimiser updates matrix, QR projects."""

    def __init__(self, d: int):
        super().__init__()
        self.matrix = nn.Parameter(torch.eye(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q, _ = torch.linalg.qr(self.matrix, mode="complete")
        return x @ Q

    @torch.no_grad()
    def rotation(self) -> torch.Tensor:
        Q, _ = torch.linalg.qr(self.matrix, mode="complete")
        return Q.detach()


_LOSS_PRESETS = {
    # (optimizer, lr, cosine LR, epochs)
    "whip":      ("sgd",  1e-3, False, 10),
    "swd_unif":  ("adam", 1e-3, True,  30),
    "swd_gauss": ("adam", 1e-2, True,  100),
}


def train_r1(acts: torch.Tensor, loss_name: str, device: torch.device,
              batch_size: int = 2048,
              epochs_override: int | None = None) -> torch.Tensor:
    """Train an orthogonal R1 on already-collected activations."""
    optim_name, lr, cos_lr, epochs = _LOSS_PRESETS[loss_name]
    if epochs_override:
        epochs = epochs_override

    d = acts.shape[-1]
    loss_fn = get_loss_fn(loss_name)
    R1 = R1_QR(d).to(device)

    if optim_name == "sgd":
        opt = torch.optim.SGD(R1.parameters(), lr=lr, momentum=0.9)
    else:
        opt = torch.optim.Adam(R1.parameters(), lr=lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if cos_lr else None)

    N = acts.shape[0]
    print(f"  [{loss_name}] optim={optim_name} lr={lr} epochs={epochs} "
          f"N={N} d={d}")
    for ep in range(epochs):
        perm = torch.randperm(N)
        total, n_batches = 0.0, 0
        for s in range(0, N, batch_size):
            x = acts[perm[s:s + batch_size]].to(device)
            loss = loss_fn(R1(x))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n_batches += 1
        if sched:
            sched.step()
        if ep == 0 or (ep + 1) % max(1, epochs // 5) == 0 or ep == epochs - 1:
            print(f"    epoch {ep + 1:3d}/{epochs}  loss={total / n_batches:.4f}")

    return R1.rotation().cpu()


# ---------------------------------------------------------------------------
# 4) Plot (2 rows x 4 cols: histogram / per-channel absmax)
# ---------------------------------------------------------------------------
_ORDER = ("raw", "whip", "swd_unif", "swd_gauss")
_LABEL = {
    "raw":       "Raw (no rotation)",
    "whip":      r"R1 — Whip",
    "swd_unif":  r"R1 — SWD-Uniform",
    "swd_gauss": r"R1 — SWD-Gaussian",
}
# Muted, publication-grade palette (seaborn "deep" family).  Gray baseline for
# "raw" keeps visual attention on the three learned rotations; the three method
# colours are distinct in both hue and luminance so they reproduce cleanly in
# grayscale print as well.
_COLOR = {
    "raw":       "#545454",   # charcoal — baseline
    "whip":      "#C44E52",   # muted brick red
    "swd_unif":  "#4C72B0",   # steel blue
    "swd_gauss": "#55A868",   # sage green
}

# rcParams for NeurIPS / ICLR / ICML-ready figures.  Key points:
#   - serif body text + Computer Modern math (matches LaTeX paper body)
#   - Type-42 fonts in PDF/PS (required for submission: searchable + embeddable)
#   - no top/right spines, thin axes, subtle grid — clean modern look
#   - tight default savefig at 300 dpi for any rasterised elements
_PUB_RC = {
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Computer Modern Roman",
                           "Times New Roman", "serif"],
    "mathtext.fontset":   "cm",
    "axes.titlesize":     10.5,
    "axes.labelsize":     9.5,
    "xtick.labelsize":    8.5,
    "ytick.labelsize":    8.5,
    "legend.fontsize":    9.0,
    "axes.linewidth":     0.8,
    "axes.edgecolor":     "#444444",
    "axes.labelcolor":    "#222222",
    "text.color":         "#222222",
    "xtick.color":        "#444444",
    "ytick.color":        "#444444",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "grid.color":         "#B8B8B8",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.35,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "savefig.dpi":        300,
}


def plot_comparison(bundles: dict[str, torch.Tensor], out_path: Path,
                     layer_idx: int, model_name: str) -> None:
    plt.rcParams.update(_PUB_RC)

    # Shared x-range from raw tails (avoid clipping outliers in the raw panel)
    raw_v = bundles["raw"].numpy().ravel()
    span = float(np.percentile(np.abs(raw_v), 99.95)) * 1.05
    bins = np.linspace(-span, span, 121)
    ymax_absmax = max(float(t.abs().max()) for t in bundles.values()) * 1.05

    fig, axes = plt.subplots(
        2, 4,
        figsize=(13, 5.8),
        gridspec_kw={"hspace": 0.46, "wspace": 0.26,
                     "left": 0.065, "right": 0.985,
                     "bottom": 0.10, "top": 0.88},
    )

    annot_bbox = dict(boxstyle="round,pad=0.28", facecolor="white",
                      alpha=0.88, edgecolor="#CCCCCC", linewidth=0.5)

    for col, name in enumerate(_ORDER):
        x = bundles[name].numpy()
        v = x.ravel()
        absmax = np.abs(x).max(axis=0)
        order = np.argsort(-absmax)
        color = _COLOR[name]

        # --- Top: value histogram (log y) ---
        ax = axes[0, col]
        ax.hist(v, bins=bins, color=color, edgecolor="none", alpha=0.9)
        ax.set_yscale("log")
        ax.set_xlim(-span, span)
        ax.set_title(_LABEL[name], pad=6)
        ax.grid(True, which="major", axis="y",
                linestyle="-", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        kurt = float(((v - v.mean()) ** 4).mean() / (v.std() ** 4 + 1e-12) - 3.0)
        ax.text(0.04, 0.94, rf"$\kappa_4 = {kurt:.1f}$",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=annot_bbox)
        if col == 0:
            ax.set_ylabel("Count (log)")
        ax.set_xlabel("Activation value")

        # --- Bottom: per-channel absmax, sorted desc ---
        ax = axes[1, col]
        ax.bar(np.arange(len(absmax)), absmax[order],
               width=1.0, color=color, linewidth=0, alpha=0.95)
        ax.set_ylim(0, ymax_absmax)
        ax.set_xlim(0, len(absmax))
        ax.grid(True, which="major", axis="y",
                linestyle="-", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(0.96, 0.94, f"peak = {absmax.max():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=annot_bbox)
        if col == 0:
            ax.set_ylabel(r"per-channel $\max_t |x|$")
        ax.set_xlabel("Channel (sorted desc.)")

    fig.suptitle(
        f"mlp.up_proj input distribution — layer {layer_idx} — {Path(model_name).name}",
        fontsize=11.5, y=0.965,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# 5) Layer-list parsing
# ---------------------------------------------------------------------------
def parse_layers(spec: str, num_layers: int) -> list[int]:
    """Parse a layer spec: 'all' | '0,3,7' | '0-15' | '0-15:2' (step).

    Returns a sorted list of unique layer indices in range [0, num_layers).
    """
    spec = spec.strip()
    if spec.lower() == "all":
        return list(range(num_layers))

    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, step_s = part.split(":", 1)
            step = int(step_s)
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1, step))
        else:
            out.add(int(part))

    bad = [i for i in out if i < 0 or i >= num_layers]
    if bad:
        raise ValueError(f"layer indices out of range [0,{num_layers}): {bad}")
    return sorted(out)


# ---------------------------------------------------------------------------
# 6) Metrics (printed to stdout + saved as CSV for analysis)
# ---------------------------------------------------------------------------
_METRIC_COLS = (
    "kurt",        # excess kurtosis of flat values (heavy tail if >> 0)
    "skew",        # skewness (sanity: ~0 for a good rotation)
    "rms",         # global RMS of x
    "absmax",      # max|x| — the outlier story
    "p99_9",       # 99.9th percentile of |x| — sub-max tail
    "chan_ratio",  # max(channel_absmax) / mean(channel_absmax) — outlier concentration
    "amax_rms",    # absmax / rms — clipping headroom (lower = friendlier to INT4)
    "chan_cv",     # std(channel_rms) / mean(channel_rms) — channel balance (lower = more uniform)
)


def compute_metrics(x: np.ndarray) -> dict[str, float]:
    """Scalar metrics summarising the distribution shape of a (N, d) tensor."""
    v = x.ravel()
    mu = float(v.mean())
    s = float(v.std()) + 1e-12
    rms = float(np.sqrt((v ** 2).mean()))

    channel_absmax = np.abs(x).max(axis=0)
    channel_rms = np.sqrt((x ** 2).mean(axis=0))

    return {
        "kurt":       float(((v - mu) ** 4).mean() / (s ** 4) - 3.0),
        "skew":       float(((v - mu) ** 3).mean() / (s ** 3)),
        "rms":        rms,
        "absmax":     float(np.abs(v).max()),
        "p99_9":      float(np.percentile(np.abs(v), 99.9)),
        "chan_ratio": float(channel_absmax.max() / (channel_absmax.mean() + 1e-12)),
        "amax_rms":   float(np.abs(v).max() / (rms + 1e-12)),
        "chan_cv":    float(channel_rms.std() / (channel_rms.mean() + 1e-12)),
    }


def _fmt(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:8.1f}"
    if abs(v) >= 10:
        return f"{v:8.2f}"
    return f"{v:8.3f}"


def print_layer_table(layer: int, per_config: dict[str, dict[str, float]]) -> None:
    header = f"  {'config':<10}" + "".join(f" {c:>8}" for c in _METRIC_COLS)
    print(f"\n--- layer {layer:03d} ---")
    print(header)
    for cfg in _ORDER:
        row = per_config[cfg]
        line = f"  {cfg:<10}" + "".join(_fmt(row[c]) for c in _METRIC_COLS)
        print(line)


def print_summary_table(rows: list[dict]) -> None:
    """rows: list of {layer, config, ...metric floats...}."""
    print("\n=== mean +/- std across layers ===")
    header = f"  {'config':<10}" + "".join(f" {c:>8}" for c in _METRIC_COLS)
    print(header)
    for cfg in _ORDER:
        by_cfg = [r for r in rows if r["config"] == cfg]
        means: list[str] = []
        for c in _METRIC_COLS:
            vals = np.array([r[c] for r in by_cfg], dtype=float)
            means.append(f"{vals.mean():8.3f}")
        print(f"  {cfg:<10}" + "".join(means))
    print()


def save_metrics_csv(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "config", *_METRIC_COLS])
        for r in rows:
            w.writerow([r["layer"], r["config"], *(f"{r[c]:.6g}" for c in _METRIC_COLS)])
    print(f"wrote metrics CSV: {path}")


# ---------------------------------------------------------------------------
# 7) Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--layers", default="all",
                   help="Layer selector: 'all' | '0,3,7' | '0-15' | '0-15:2'")
    p.add_argument("--n_samples", type=int, default=16,
                   help="Number of calibration sequences from WikiText-2")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=2048,
                   help="Rows per R1 gradient step")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override per-loss epoch preset")
    p.add_argument("--per_layer_r1", action="store_true",
                   help="Train a separate R1 per layer instead of one global "
                        "R1 on pooled activations (default: global, matches "
                        "the DartQuant pipeline).")
    p.add_argument("--max_rows_per_layer", type=int, default=4096,
                   help="Subsample each layer's captured activations to this "
                        "many rows before R1 training (keeps memory bounded "
                        "when sweeping many layers).")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--out_dir", type=Path,
                   default=Path("artifacts/activation_comparison"),
                   help="Directory to write one PDF per layer into "
                        "(layer_<idx>.pdf).")
    args = p.parse_args()

    if args.cache_dir:
        os.environ.setdefault("HF_HOME", args.cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", args.cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, cache_dir=args.cache_dir,
    ).to(device).eval()

    num_layers = len(model.model.layers)
    layers = parse_layers(args.layers, num_layers)
    print(f"model has {num_layers} layers; sweeping {len(layers)}: {layers}")

    print("loading WikiText-2 calibration")
    calib = load_wikitext_calib(tokenizer, args.n_samples, args.seqlen)

    # Resolve hook targets for every selected layer
    targets: dict[int, nn.Module] = {}
    for L in layers:
        try:
            targets[L] = model.model.layers[L].mlp.up_proj
        except (AttributeError, IndexError) as e:
            raise SystemExit(f"cannot resolve layer {L}.mlp.up_proj: {e}")

    print(f"capturing mlp.up_proj inputs for {len(layers)} layers (one pass) ...")
    acts_by_layer = capture_activations_multi(model, calib, targets, device)
    for L in list(acts_by_layer):
        t = acts_by_layer[L]
        if t.shape[0] > args.max_rows_per_layer:
            idx = torch.randperm(t.shape[0])[:args.max_rows_per_layer]
            acts_by_layer[L] = t[idx].contiguous()
    a0 = next(iter(acts_by_layer.values()))
    print(f"  per-layer: {a0.shape[0]} rows x {a0.shape[1]} dim")

    # Free the model — R1 training only needs the captured tensors
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train R1 per loss: one global R1 (default) or one per layer (opt-in).
    # Global R1 on pooled activations matches the actual DartQuant pipeline,
    # where a single R1 is baked into every layer's weights.
    rotations: dict[str, dict[int, torch.Tensor]] = {}
    for loss_name in ("whip", "swd_unif", "swd_gauss"):
        rotations[loss_name] = {}
        if args.per_layer_r1:
            for L in layers:
                print(f"training R1 [{loss_name}] layer {L}")
                R1 = train_r1(acts_by_layer[L], loss_name, device,
                              batch_size=args.batch_size,
                              epochs_override=args.epochs)
                rotations[loss_name][L] = R1
        else:
            pool = torch.cat([acts_by_layer[L] for L in layers], dim=0)
            print(f"training global R1 [{loss_name}] on pooled "
                  f"{pool.shape[0]} rows")
            R1 = train_r1(pool, loss_name, device,
                          batch_size=args.batch_size,
                          epochs_override=args.epochs)
            for L in layers:
                rotations[loss_name][L] = R1
            del pool

    # Plot per layer + accumulate metrics
    args.out_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(max(layers))))
    metric_rows: list[dict] = []
    for L in layers:
        acts = acts_by_layer[L]
        bundles: dict[str, torch.Tensor] = {"raw": acts}
        for loss_name in ("whip", "swd_unif", "swd_gauss"):
            bundles[loss_name] = (acts @ rotations[loss_name][L]).contiguous()
        out_path = args.out_dir / f"layer_{L:0{width}d}.pdf"
        plot_comparison(bundles, out_path, L, args.model)

        per_config = {cfg: compute_metrics(bundles[cfg].numpy()) for cfg in _ORDER}
        print_layer_table(L, per_config)
        for cfg, m in per_config.items():
            metric_rows.append({"layer": L, "config": cfg, **m})

    print_summary_table(metric_rows)
    save_metrics_csv(metric_rows, args.out_dir / "metrics.csv")
    print(f"done — wrote {len(layers)} PDFs to {args.out_dir}")


if __name__ == "__main__":
    main()
