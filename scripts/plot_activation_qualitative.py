#!/usr/bin/env python3
"""Publication-quality qualitative figures for the loss comparison.

Reads the four .npz files produced by capture_qualitative_activations.py and
renders four PDFs that visualise how Whip, SWD-Uniform and SWD-Gaussian
reshape the ``down_proj`` input distribution relative to the raw FP16 model.

Figures
-------
  activation_main.pdf        (main body)  Per-channel absolute-max bars at a
                             representative mid-network layer and at the last
                             layer, side-by-side across all four conditions.
                             This is the QuaRot-style motivating picture.

  activation_histogram.pdf   (appendix)   Log-scale histogram of all captured
                             activations at one fixed layer, four-panel.
                             Shape differences between uniform-targeted and
                             Gaussian-targeted losses are directly visible.

  activation_absmax_curve.pdf (appendix)  Per-layer absmax curve (p99.9 and
                             max) across every layer, 4-config overlay.
                             Shows whether outlier suppression is local to
                             one layer or global.

  activation_variance_heatmap.pdf (appendix)  Heatmap of sorted per-channel
                             variance across layers (rows) × rank-ordered
                             channels (columns), one panel per config. A
                             flatter column profile means a more uniform
                             variance distribution, which is the prerequisite
                             for low absmax and therefore effective 4-bit
                             quantisation.

Styling mirrors scripts/plot_rho.py (TACL serif, colorblind-safe Okabe-Ito
palette, thin axes, pdf.fonttype=42 so TACL's PDF inspector can parse the
embedded fonts).

Usage
-----
    python scripts/plot_activation_qualitative.py \
        --in_dir artifacts/qualitative/llama-3.2-1b \
        --out_dir report_writing/figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


# ---------------------------------------------------------------------------
# Publication style (copied from scripts/plot_rho.py for consistency)
# ---------------------------------------------------------------------------
def apply_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Okabe-Ito + a grey for the raw baseline; deliberately places the three
# trained conditions in warm colours so the baseline stays visually passive.
CONFIG_ORDER = ["raw", "whip", "swd_unif", "swd_gauss"]
CONFIG_LABEL = {
    "raw":       r"FP16 (no rotation)",
    "whip":      r"Whip + $R_1$",
    "swd_unif":  r"SWD-Uniform + $R_1$",
    "swd_gauss": r"SWD-Gaussian + $R_1$",
}
CONFIG_COLOR = {
    "raw":       "#8c8c8c",   # neutral grey
    "whip":      "#D55E00",   # vermillion
    "swd_unif":  "#0072B2",   # blue
    "swd_gauss": "#009E73",   # bluish green
}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _load_one(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    layer_idxs = data["_layer_idxs"].astype(int).tolist()
    acts = {i: data[f"layer_{i:03d}"] for i in layer_idxs}
    return {
        "config": str(data["_config"]),
        "model": str(data["_model"]),
        "hidden_size": int(data["_hidden_size"]),
        "intermediate_size": int(data["_intermediate_size"]),
        "num_layers": int(data["_num_layers"]),
        "layers": layer_idxs,
        "acts": acts,
    }


def load_all(in_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cfg in CONFIG_ORDER:
        p = in_dir / cfg / "down_proj_inputs.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing capture file: {p}")
        out[cfg] = _load_one(p)
    return out


def _pick_layer(num_layers: int, fraction: float) -> int:
    """Pick a representative layer by depth fraction (0 = first, 1 = last)."""
    idx = int(round(fraction * (num_layers - 1)))
    return max(0, min(num_layers - 1, idx))


# ---------------------------------------------------------------------------
# Figure 1 (main body): per-channel absmax bars at 2 representative layers
# ---------------------------------------------------------------------------
def plot_absmax_bars(bundles: dict[str, dict], out_path: Path) -> None:
    num_layers = bundles["raw"]["num_layers"]
    layer_mid = _pick_layer(num_layers, 0.5)
    layer_late = _pick_layer(num_layers, 0.95)
    layers_to_plot = [("Mid-network layer "
                       + str(layer_mid), layer_mid),
                      ("Late layer " + str(layer_late), layer_late)]

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.3), sharex="col",
                             sharey="row",
                             gridspec_kw={"wspace": 0.08, "hspace": 0.35})

    for row, (row_title, li) in enumerate(layers_to_plot):
        row_axes = axes[row]
        # Global y-limit (shared across the 4 panels in this row) driven by the
        # raw baseline, so the three rotations are directly comparable to it.
        raw_absmax = np.max(np.abs(bundles["raw"]["acts"][li]), axis=0)
        y_cap = float(np.percentile(raw_absmax, 99.5)) * 1.05
        y_cap = max(y_cap, 1e-3)

        for col, cfg in enumerate(CONFIG_ORDER):
            ax = row_axes[col]
            x = bundles[cfg]["acts"][li]
            absmax = np.max(np.abs(x), axis=0)
            order = np.argsort(-absmax)  # descending
            ax.bar(
                np.arange(len(absmax)),
                absmax[order],
                width=1.0,
                color=CONFIG_COLOR[cfg],
                linewidth=0,
                alpha=0.95,
            )
            ax.set_ylim(0, y_cap)
            ax.set_xlim(0, len(absmax))
            ax.set_xticks([])
            ax.grid(axis="y", linestyle=":", alpha=0.35)
            if row == 0:
                ax.set_title(CONFIG_LABEL[cfg], pad=4)
            if col == 0:
                ax.set_ylabel(row_title + "\n$\\max_t |x_{t,c}|$", fontsize=8)
            # Numeric callout: maximum absmax for that condition, to make the
            # "outlier smashing" quantitative even without reading the body.
            peak = absmax.max()
            ax.text(
                0.97, 0.92,
                f"peak={peak:.1f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color="#404040",
            )

    fig.text(0.5, -0.02, "Channel (sorted by $|x|$, descending)",
             ha="center", va="top", fontsize=9)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 (appendix): stacked histograms at one fixed layer
# ---------------------------------------------------------------------------
def plot_histograms(bundles: dict[str, dict], out_path: Path) -> None:
    num_layers = bundles["raw"]["num_layers"]
    layer = _pick_layer(num_layers, 0.75)

    all_vals = np.concatenate(
        [bundles[c]["acts"][layer].ravel() for c in CONFIG_ORDER]
    )
    # Symmetric range from the raw baseline so rotated conditions are
    # directly comparable.
    raw_vals = bundles["raw"]["acts"][layer].ravel()
    span = float(np.percentile(np.abs(raw_vals), 99.9))
    x_edges = np.linspace(-span * 1.05, span * 1.05, 161)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharex=True,
                             sharey=True,
                             gridspec_kw={"wspace": 0.10})
    for ax, cfg in zip(axes, CONFIG_ORDER):
        v = bundles[cfg]["acts"][layer].ravel()
        ax.hist(
            v, bins=x_edges, color=CONFIG_COLOR[cfg], alpha=0.85,
            edgecolor="none",
        )
        ax.set_yscale("log")
        ax.set_xlim(x_edges[0], x_edges[-1])
        ax.set_title(CONFIG_LABEL[cfg], pad=4)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        # Annotate kurtosis — a single scalar that separates heavy-tailed from
        # Gaussian-like distributions and therefore distinguishes the three
        # losses directly.
        m = v.mean()
        s = v.std() + 1e-12
        kurt = float(((v - m) ** 4).mean() / (s ** 4) - 3.0)
        ax.text(
            0.03, 0.92,
            f"$\\kappa={kurt:.1f}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, color="#404040",
        )
    axes[0].set_ylabel("Count (log)", fontsize=8)
    for ax in axes:
        ax.set_xlabel("Activation value at layer " + str(layer))
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 (appendix): absmax / p99.9 curves across all layers
# ---------------------------------------------------------------------------
def plot_absmax_curves(bundles: dict[str, dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4),
                             gridspec_kw={"wspace": 0.22})
    titles = [r"Per-layer peak: $\max_{t,c} |x|$",
              r"Per-layer tail: $P_{99.9}(|x|)$"]
    agg_fns = [
        lambda x: float(np.max(np.abs(x))),
        lambda x: float(np.percentile(np.abs(x).ravel(), 99.9)),
    ]

    for ax, title, fn in zip(axes, titles, agg_fns):
        for cfg in CONFIG_ORDER:
            b = bundles[cfg]
            ys = [fn(b["acts"][i]) for i in b["layers"]]
            ax.plot(
                b["layers"], ys,
                color=CONFIG_COLOR[cfg],
                label=CONFIG_LABEL[cfg],
                marker="o", markersize=2.5,
                linewidth=1.1,
                alpha=0.95,
            )
        ax.set_title(title, pad=4)
        ax.set_xlabel("Transformer layer index")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_yscale("log")

    axes[0].set_ylabel(r"Magnitude")
    axes[0].legend(
        loc="upper left", bbox_to_anchor=(0.0, 1.02),
        ncol=2, columnspacing=1.0, handlelength=1.6, borderaxespad=0.0,
        fontsize=7,
    )
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 (appendix): per-channel variance heatmap
# ---------------------------------------------------------------------------
def plot_variance_heatmap(bundles: dict[str, dict], out_path: Path) -> None:
    # Build (num_layers, num_channels) matrix of per-channel variances with
    # channels sorted in descending order PER LAYER so each row is a rank
    # profile. Uniform rows → flat profile; peaked rows → outlier channel.
    # Shared vmin/vmax from the raw baseline to keep colour scales comparable.
    num_layers = bundles["raw"]["num_layers"]
    ffn = bundles["raw"]["intermediate_size"]

    mats = {}
    for cfg in CONFIG_ORDER:
        b = bundles[cfg]
        m = np.zeros((num_layers, ffn), dtype=np.float32)
        for li in b["layers"]:
            x = b["acts"][li]
            v = x.var(axis=0)
            v_sorted = np.sort(v)[::-1]
            if v_sorted.size < ffn:
                pad = np.zeros(ffn - v_sorted.size, dtype=v_sorted.dtype)
                v_sorted = np.concatenate([v_sorted, pad])
            m[li] = v_sorted[:ffn]
        mats[cfg] = m

    raw_mat = mats["raw"]
    vmax = float(np.percentile(raw_mat, 99.9))
    vmin = max(float(np.percentile(raw_mat[raw_mat > 0], 0.5))
               if np.any(raw_mat > 0) else 1e-6, 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True,
                             gridspec_kw={"wspace": 0.08,
                                          "width_ratios": [1, 1, 1, 1.12]})
    im = None
    for ax, cfg in zip(axes, CONFIG_ORDER):
        m = mats[cfg]
        im = ax.imshow(
            m,
            aspect="auto",
            origin="lower",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_title(CONFIG_LABEL[cfg], pad=4)
        ax.set_xlabel("Channel rank (by variance)")
        ax.set_xticks([0, ffn // 2, ffn - 1])
    axes[0].set_ylabel("Transformer layer")

    # Single right-hand colorbar shared by all panels
    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.05, pad=0.02)
    cbar.set_label(r"Per-channel variance $\mathrm{Var}(x_{\cdot, c})$",
                   fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in_dir", required=True, type=Path,
                   help="Directory containing one subdir per config "
                        "(raw, whip, swd_unif, swd_gauss) with "
                        "down_proj_inputs.npz inside.")
    p.add_argument("--out_dir", required=True, type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    bundles = load_all(args.in_dir)

    plot_absmax_bars(
        bundles, args.out_dir / "activation_main.pdf"
    )
    plot_histograms(
        bundles, args.out_dir / "activation_histogram.pdf"
    )
    plot_absmax_curves(
        bundles, args.out_dir / "activation_absmax_curve.pdf"
    )
    plot_variance_heatmap(
        bundles, args.out_dir / "activation_variance_heatmap.pdf"
    )

    print(f"Wrote 4 figures into {args.out_dir}")


if __name__ == "__main__":
    main()
