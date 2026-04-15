#!/usr/bin/env python3
"""Single-figure "before vs. after rotation" comparison.

Reads the four .npz files produced by capture_qualitative_activations.py
(raw / whip / swd_unif / swd_gauss) and renders ONE publication-quality PDF
that directly contrasts

    BEFORE:  pristine FP16 Llama (no R1)
    AFTER:   LN-fused + R1 trained with {whip, swd_unif, swd_gauss}

so the effectiveness of each loss at flattening activation outliers is
visible at a glance.

Figure layout (4 rows x 3 cols)
-------------------------------
              col 0: histogram      col 1: absmax bars     col 2: sample heatmap
    row 0: raw (BEFORE)
    row 1: whip         (AFTER)
    row 2: swd_unif     (AFTER)
    row 3: swd_gauss    (AFTER)

    * col 0 — activation value histogram (log y).  Kurtosis annotated.
    * col 1 — per-channel absmax bars sorted descending.  Peak value annotated.
    * col 2 — (N rows x C channels) heatmap (channels sorted by absmax desc,
              rows subsampled), symmetric colour scale shared per row so
              outlier streaks stand out as bright columns.

A bottom-of-figure caption shows the layer index.  Column-wise axes are
SHARED so rotated conditions are directly comparable to the raw baseline.

Usage
-----
    python scripts/plot_rotation_before_after.py \
        --in_dir  artifacts/qualitative/llama-3.2-1b \
        --out     report_writing/figures/rotation_before_after.pdf \
        [--layer_frac 0.75]  [--rows_heatmap 400]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# ---------------------------------------------------------------------------
# Style (matches plot_activation_qualitative.py / plot_rho.py)
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


CONFIG_ORDER = ["raw", "whip", "swd_unif", "swd_gauss"]
CONFIG_LABEL = {
    "raw":       r"BEFORE: FP16 (no $R_1$)",
    "whip":      r"AFTER: Whip + $R_1$",
    "swd_unif":  r"AFTER: SWD-Uniform + $R_1$",
    "swd_gauss": r"AFTER: SWD-Gaussian + $R_1$",
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
    idx = int(round(fraction * (num_layers - 1)))
    return max(0, min(num_layers - 1, idx))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot_before_after(bundles: dict[str, dict], out_path: Path,
                       layer_frac: float = 0.75,
                       rows_heatmap: int = 400) -> None:
    num_layers = bundles["raw"]["num_layers"]
    layer = _pick_layer(num_layers, layer_frac)

    # --- shared column 0 (histogram) bin edges ---
    # Span computed from the raw config so tails are never clipped.
    span = max(
        float(np.percentile(np.abs(bundles[c]["acts"][layer].ravel()), 99.95))
        for c in CONFIG_ORDER
    )
    span = max(span, 1e-3) * 1.05
    hist_edges = np.linspace(-span, span, 161)

    # --- shared column 1 (absmax bars) y-cap from max across all configs ---
    absmax_cap = max(
        float(np.max(np.abs(bundles[c]["acts"][layer]))) for c in CONFIG_ORDER
    ) * 1.05

    # --- shared column 2 (heatmap) colour range from raw ---
    raw_absmax = float(np.max(np.abs(bundles["raw"]["acts"][layer])))
    heat_lim = raw_absmax  # symmetric ±

    n_rows, n_cols = 4, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.2, 8.2),
        gridspec_kw={
            "width_ratios": [1.0, 1.0, 1.25],
            "hspace": 0.28, "wspace": 0.28,
            "left": 0.09, "right": 0.97,
            "bottom": 0.07, "top": 0.92,
        },
    )

    # column titles
    axes[0, 0].set_title(r"Activation histogram (log $y$)", pad=6)
    axes[0, 1].set_title(r"Per-channel $\max_t |x|$ (sorted)", pad=6)
    axes[0, 2].set_title(r"Token $\times$ channel heatmap", pad=6)

    for row, cfg in enumerate(CONFIG_ORDER):
        x = bundles[cfg]["acts"][layer]       # (N, C)
        colour = CONFIG_COLOR[cfg]

        # ---- row label on the far left ----
        ax_hist = axes[row, 0]
        ax_bars = axes[row, 1]
        ax_heat = axes[row, 2]

        # ---- col 0: histogram ----
        v = x.ravel()
        ax_hist.hist(v, bins=hist_edges, color=colour, alpha=0.9,
                     edgecolor="none")
        ax_hist.set_yscale("log")
        ax_hist.set_xlim(-span, span)
        ax_hist.grid(axis="y", linestyle=":", alpha=0.35)

        m = float(v.mean())
        s = float(v.std()) + 1e-12
        kurt = float(((v - m) ** 4).mean() / (s ** 4) - 3.0)
        ax_hist.text(
            0.03, 0.93, f"$\\kappa_4={kurt:.1f}$",
            transform=ax_hist.transAxes, ha="left", va="top",
            fontsize=7.5, color="#222222",
        )
        ax_hist.set_ylabel(
            CONFIG_LABEL[cfg] + "\n\nCount (log)",
            fontsize=8.5, labelpad=4,
        )

        # ---- col 1: sorted per-channel absmax bars ----
        absmax = np.max(np.abs(x), axis=0)
        order = np.argsort(-absmax)
        ax_bars.bar(
            np.arange(len(absmax)), absmax[order],
            width=1.0, color=colour, linewidth=0, alpha=0.95,
        )
        ax_bars.set_ylim(0, absmax_cap)
        ax_bars.set_xlim(0, len(absmax))
        ax_bars.set_xticks([])
        ax_bars.grid(axis="y", linestyle=":", alpha=0.35)

        peak = float(absmax.max())
        ax_bars.text(
            0.97, 0.92, f"peak={peak:.1f}",
            transform=ax_bars.transAxes, ha="right", va="top",
            fontsize=7.5, color="#222222",
        )

        # ---- col 2: heatmap (rows sub-sampled, cols sorted by absmax) ----
        # Sort channels by absmax so outlier columns sit on the left.
        x_sorted_cols = x[:, order]
        if x_sorted_cols.shape[0] > rows_heatmap:
            rng = np.random.default_rng(0)
            ridx = rng.choice(x_sorted_cols.shape[0], size=rows_heatmap,
                              replace=False)
            ridx.sort()
            x_heat = x_sorted_cols[ridx]
        else:
            x_heat = x_sorted_cols

        im = ax_heat.imshow(
            x_heat, aspect="auto", origin="lower",
            cmap="RdBu_r", vmin=-heat_lim, vmax=heat_lim,
            interpolation="nearest",
        )
        ax_heat.set_yticks([])
        ax_heat.set_xticks([0, x_heat.shape[1] // 2, x_heat.shape[1] - 1])
        ax_heat.set_xlabel("Channel rank", fontsize=8.5, labelpad=2)

        # Colorbar on the RIGHT of every heatmap (each row uses the
        # same ±raw_absmax scale, so the same tick marks work everywhere —
        # but placing one per row keeps the association obvious).
        cax = ax_heat.inset_axes([1.02, 0.0, 0.035, 1.0])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=7)
        if row == 0:
            cbar.set_label("activation", fontsize=7.5)

    # column-0 shared xlabel
    axes[-1, 0].set_xlabel("Activation value", fontsize=8.5)
    # column-1 shared xlabel
    axes[-1, 1].set_xlabel("Channel rank (sorted by absmax)",
                            fontsize=8.5)

    fig.suptitle(
        f"Down-proj input: before vs. after rotation   "
        f"(layer {layer} of {num_layers}, model "
        f"{Path(bundles['raw']['model']).name})",
        fontsize=10, y=0.975,
    )

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
    p.add_argument("--out", required=True, type=Path,
                   help="Output PDF path")
    p.add_argument("--layer_frac", type=float, default=0.75,
                   help="Layer to visualise, as depth fraction in [0,1] "
                        "(default 0.75 — late layer, where outliers "
                        "are worst in raw Llama).")
    p.add_argument("--rows_heatmap", type=int, default=400,
                   help="Max token rows drawn in the heatmap column "
                        "(default 400). Sub-sampled uniformly at random.")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    apply_paper_style()
    bundles = load_all(args.in_dir)

    plot_before_after(
        bundles, args.out,
        layer_frac=args.layer_frac,
        rows_heatmap=args.rows_heatmap,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
