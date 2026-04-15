#!/usr/bin/env python3
"""Publication-quality figures for the rho / variance-heterogeneity sweep.

Reads artifacts/rho/per_band_profile.csv (produced by aggregate_rho.py)
and renders two PDF figures ready for inclusion in the TACL draft:

  rho_profile.pdf           Main figure: mean |rho| and mean |log sigma^2
                            ratio| vs RoPE frequency theta_k, for Q and K
                            streams across the Llama-3 family. This is the
                            empirical validation of Theorem thm:rope_var's
                            ideal assumptions and Corollary hadamard_rope's
                            preconditions.
  rho_distribution.pdf      Supplement: distribution of |rho| across all
                            (layer, head, band) triples per model,
                            visualised as violin plots. Makes the point
                            that the non-zero mean is not an artefact of a
                            few outliers.

Styling matches the TACL Times-Roman body text: 8-9 pt serif, STIX math,
thin axes, no top/right spines, colorblind-safe palette.

Usage:
  python scripts/plot_rho.py
  python scripts/plot_rho.py --in_dir artifacts/rho \
      --out_dir report_writing/figures
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Publication style
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
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.5,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Colorblind-safe palette (Okabe-Ito)
PALETTE = {
    "1B": "#0072B2",   # blue
    "3B": "#D55E00",   # vermillion
    "8B": "#009E73",   # green
}
STREAM_STYLE = {
    "Q": dict(linestyle="-",  lw=1.4),
    "K": dict(linestyle="--", lw=1.4),
}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
def _scale_of(model: str) -> str:
    if "1b" in model: return "1B"
    if "3b" in model: return "3B"
    if "8b" in model: return "8B"
    raise ValueError(model)


def _tuned_of(model: str) -> str:
    return "Instruct" if "instruct" in model else "Base"


def aggregate_by_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Average Base and Instruct into one curve per (scale, stream, band).

    Base and Instruct are empirically indistinguishable (max absolute
    delta in mean |rho| < 0.005 across all models), so collapsing them
    gives a cleaner plot and the averaging does not hide structure.
    """
    df = df.copy()
    df["scale"] = df["model"].map(_scale_of)
    grouped = (
        df.groupby(["scale", "stream", "band"], as_index=False)
          .agg(theta_k=("theta_k", "first"),
               C_k=("C_k", "first"),
               rho_mean=("rho_mean", "mean"),
               rho_p95=("rho_p95", "mean"),
               rho_max=("rho_max", "mean"),
               varhet_mean=("varhet_mean", "mean"),
               varhet_p95=("varhet_p95", "mean"))
    )
    return grouped.sort_values(["scale", "stream", "theta_k"])


# ---------------------------------------------------------------------------
# Main figure: rho and varhet vs theta_k
# ---------------------------------------------------------------------------
def plot_rho_profile(df_agg: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(6.5, 2.55), gridspec_kw=dict(wspace=0.28)
    )
    ax_rho, ax_var = axes

    for scale in ("1B", "3B", "8B"):
        for stream in ("Q", "K"):
            sub = df_agg[(df_agg["scale"] == scale) & (df_agg["stream"] == stream)]
            if sub.empty:
                continue
            color = PALETTE[scale]
            style = STREAM_STYLE[stream]
            x = sub["theta_k"].to_numpy()
            # rho: line with very faint shaded mean->p95 band
            ax_rho.plot(x, sub["rho_mean"], color=color,
                         label=f"{scale} {stream}", **style)
            ax_rho.fill_between(
                x, sub["rho_mean"], sub["rho_p95"],
                color=color, alpha=0.05, linewidth=0,
            )
            ax_var.plot(x, sub["varhet_mean"], color=color,
                         label=f"{scale} {stream}", **style)

    # --- rho panel cosmetics --------------------------------------------
    ax_rho.set_xscale("log")
    ax_rho.invert_xaxis()  # band index k=0 (high-freq) on the LEFT
    ax_rho.axhline(0.05, color="0.5", lw=0.7, ls=":", zorder=0)
    ax_rho.set_xlabel(r"RoPE frequency $\theta_k$")
    ax_rho.set_ylabel(r"mean $|\rho_k|$  (pre-RoPE pair)")
    ax_rho.set_ylim(0.0, None)
    ax_rho.set_title("(a) Intra-band correlation", pad=4)
    ax_rho.grid(True, which="major", axis="y")

    # --- var-het panel cosmetics ----------------------------------------
    ax_var.set_xscale("log")
    ax_var.invert_xaxis()
    ax_var.set_xlabel(r"RoPE frequency $\theta_k$")
    ax_var.set_ylabel(r"mean $|\log(\sigma_1^2 / \sigma_2^2)|$")
    ax_var.set_ylim(0.0, None)
    ax_var.set_title("(b) Within-pair variance heterogeneity", pad=4)
    ax_var.grid(True, which="major", axis="y")

    # --- frequency-direction annotations (consistent with inverted axis)
    for ax in (ax_rho, ax_var):
        ax.text(0.02, -0.31, "high freq", transform=ax.transAxes,
                fontsize=7.5, color="0.35", ha="left", va="top")
        ax.text(0.98, -0.31, "low freq", transform=ax.transAxes,
                fontsize=7.5, color="0.35", ha="right", va="top")
        ax.annotate(
            "", xy=(0.96, -0.30), xytext=(0.04, -0.30),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->,head_length=0.3,head_width=0.2",
                             color="0.6", lw=0.5),
        )

    # Threshold annotation in rho panel
    ax_rho.text(ax_rho.get_xlim()[0] * 0.85, 0.052,
                r"$|\rho|=0.05$ threshold",
                color="0.35", fontsize=7, va="bottom", ha="left")

    # --- shared legend below the panels ---------------------------------
    scale_handles = [
        Line2D([], [], color=PALETTE[s], lw=1.4, label=f"Llama-3.x-{s}")
        for s in ("1B", "3B", "8B")
    ]
    stream_handles = [
        Line2D([], [], color="black", lw=1.2,
                linestyle=STREAM_STYLE[s]["linestyle"], label=f"{s} stream")
        for s in ("Q", "K")
    ]
    fig.legend(
        handles=scale_handles + stream_handles,
        loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02),
        columnspacing=1.2, handlelength=2.2,
    )
    fig.subplots_adjust(bottom=0.30)

    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {out_path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Supplementary figure: distribution of |rho|
# ---------------------------------------------------------------------------
def plot_rho_distribution(summary_csvs: dict[str, pd.DataFrame],
                          out_path: Path) -> None:
    """Violin plot of |rho| distribution per (model, stream)."""
    fig, ax = plt.subplots(figsize=(6.5, 2.7))

    labels = []
    data = []
    colors = []
    positions = []
    pos = 0
    x_ticks = []
    x_tick_labels = []
    for model in ("llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b"):
        scale = _scale_of(model)
        x_ticks.append(pos + 0.5)
        x_tick_labels.append(f"Llama-3.x-{scale}")
        for stream in ("Q", "K"):
            df_full = summary_csvs[model]
            r = np.abs(df_full[df_full["stream"] == stream]["rho"].to_numpy())
            # Average base+instruct to match the main figure
            df_inst = summary_csvs.get(f"{model}-instruct")
            if df_inst is not None:
                r_inst = np.abs(df_inst[df_inst["stream"] == stream]["rho"].to_numpy())
                r = np.concatenate([r, r_inst])
            labels.append(f"{scale}\n{stream}")
            data.append(r)
            colors.append(PALETTE[scale])
            positions.append(pos)
            pos += 1
        pos += 0.6  # gap between scales

    parts = ax.violinplot(
        data, positions=positions, widths=0.85, showmeans=False,
        showmedians=True, showextrema=False,
    )
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.30)
        body.set_edgecolor(c)
        body.set_linewidth(0.6)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(0.8)

    # Mean markers
    for x, r, c in zip(positions, data, colors):
        ax.scatter(x, r.mean(), marker="o", s=9, color=c,
                     edgecolor="black", linewidth=0.4, zorder=3)

    ax.axhline(0.05, color="0.5", lw=0.7, ls=":", zorder=0)
    ax.text(ax.get_xlim()[1] - 0.1, 0.052,
            r"$\rho\!\approx\!0$ threshold",
            color="0.35", fontsize=7, va="bottom", ha="right")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_xticks(positions, minor=True)
    ax.set_xticklabels([l.split("\n")[1] for l in labels], minor=True)
    ax.tick_params(axis="x", which="minor", pad=12, length=0, labelsize=7.5)
    ax.set_ylabel(r"$|\rho_k|$  over all (layer, head, band)")
    ax.set_title("Distribution of intra-band correlation across the Llama-3 family",
                 pad=4)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y")

    # Mini legend for mean marker + median line
    legend_handles = [
        Line2D([], [], marker="o", color="black", markerfacecolor="white",
                 markersize=4, lw=0, label="mean"),
        Line2D([], [], color="black", lw=0.9, label="median"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
                frameon=False, fontsize=7.5)

    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".png"))
    plt.close(fig)
    print(f"Wrote {out_path.with_suffix('.pdf')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="artifacts/rho")
    ap.add_argument("--out_dir", default="report_writing/figures")
    args = ap.parse_args()

    apply_paper_style()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- main figure -----------------------------------------------------
    per_band = pd.read_csv(in_dir / "per_band_profile.csv")
    agg = aggregate_by_scale(per_band)
    plot_rho_profile(agg, out_dir / "rho_profile")

    # -- supplementary violin --------------------------------------------
    summaries = {}
    for d in sorted(in_dir.iterdir()):
        if d.is_dir():
            p = d / "summary.csv"
            if p.is_file():
                summaries[d.name] = pd.read_csv(p)
    if summaries:
        plot_rho_distribution(summaries, out_dir / "rho_distribution")


if __name__ == "__main__":
    main()
