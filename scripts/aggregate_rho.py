#!/usr/bin/env python3
"""Aggregate measure_rho_rope.py outputs across the model sweep.

Reads every artifacts/rho/<model>/summary.csv under --in_dir and produces:

  cross_model_summary.csv   one row per (model, stream, band_class) with
                            percentile stats of |rho| and variance
                            heterogeneity aggregated over layer x head x
                            band, ready for the paper's appendix table.
  per_band_profile.csv      one row per (model, stream, band) with
                            mean/median/p95/max of |rho| and variance
                            ratio, aggregated over layer x head. Enables
                            "|rho| vs band index k" plots.
  per_layer_profile.csv     one row per (model, stream, layer) with the
                            same stats aggregated over head x band.
                            Shows depth dependence.

Usage:
  python scripts/aggregate_rho.py
  python scripts/aggregate_rho.py --in_dir artifacts/rho --out_dir artifacts/rho
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "llama-3.2-1b",
    "llama-3.2-1b-instruct",
    "llama-3.2-3b",
    "llama-3.2-3b-instruct",
    "llama-3.1-8b",
    "llama-3.1-8b-instruct",
]

PERCENTILES = [50, 90, 95, 99]


def _pct(arr: np.ndarray, p: float) -> float:
    return float(np.nanpercentile(arr, p)) if arr.size else float("nan")


def _summary_row(sub: pd.DataFrame, model: str, stream: str,
                 band_class: str) -> dict:
    r = np.abs(sub["rho"].to_numpy())
    s1 = sub["sigma1_sq"].to_numpy()
    s2 = sub["sigma2_sq"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.abs(np.log(np.where(s2 > 0, s1 / np.clip(s2, 1e-30, None), np.nan)))
    return {
        "model": model,
        "stream": stream,
        "band_class": band_class,
        "n": int(r.size),
        "rho_mean": float(r.mean()) if r.size else float("nan"),
        "rho_median": _pct(r, 50),
        "rho_p90": _pct(r, 90),
        "rho_p95": _pct(r, 95),
        "rho_p99": _pct(r, 99),
        "rho_max": float(r.max()) if r.size else float("nan"),
        "rho_frac_gt_05": float((r > 0.05).mean()) if r.size else float("nan"),
        "rho_frac_gt_10": float((r > 0.10).mean()) if r.size else float("nan"),
        "rho_frac_gt_20": float((r > 0.20).mean()) if r.size else float("nan"),
        "varhet_mean": float(np.nanmean(log_ratio)),
        "varhet_p90": _pct(log_ratio, 90),
        "varhet_p99": _pct(log_ratio, 99),
        "varhet_max": float(np.nanmax(log_ratio)) if log_ratio.size else float("nan"),
    }


def _per_band_rows(sub_model_stream: pd.DataFrame, model: str,
                   stream: str) -> list[dict]:
    rows = []
    grouped = sub_model_stream.groupby("band", sort=True)
    for band, g in grouped:
        r = np.abs(g["rho"].to_numpy())
        s1 = g["sigma1_sq"].to_numpy()
        s2 = g["sigma2_sq"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.abs(np.log(np.where(s2 > 0, s1 / np.clip(s2, 1e-30, None),
                                                np.nan)))
        rows.append({
            "model": model,
            "stream": stream,
            "band": int(band),
            "theta_k": float(g["theta_k"].iloc[0]),
            "C_k": float(g["C_k"].iloc[0]),
            "S_k": float(g["S_k"].iloc[0]),
            "band_class": "low_freq" if float(g["C_k"].iloc[0]) > 0.5 else "high_freq",
            "n": int(r.size),
            "rho_mean": float(r.mean()),
            "rho_median": _pct(r, 50),
            "rho_p95": _pct(r, 95),
            "rho_max": float(r.max()),
            "varhet_mean": float(np.nanmean(log_ratio)),
            "varhet_p95": _pct(log_ratio, 95),
            "varhet_max": float(np.nanmax(log_ratio)),
        })
    return rows


def _per_layer_rows(sub_model_stream: pd.DataFrame, model: str,
                    stream: str) -> list[dict]:
    rows = []
    for layer, g in sub_model_stream.groupby("layer", sort=True):
        r = np.abs(g["rho"].to_numpy())
        s1 = g["sigma1_sq"].to_numpy()
        s2 = g["sigma2_sq"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.abs(np.log(np.where(s2 > 0, s1 / np.clip(s2, 1e-30, None),
                                                np.nan)))
        rows.append({
            "model": model,
            "stream": stream,
            "layer": int(layer),
            "n": int(r.size),
            "rho_mean": float(r.mean()),
            "rho_p95": _pct(r, 95),
            "rho_max": float(r.max()),
            "varhet_mean": float(np.nanmean(log_ratio)),
            "varhet_p95": _pct(log_ratio, 95),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="artifacts/rho")
    p.add_argument("--out_dir", default="artifacts/rho")
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_per_band = []
    all_per_layer = []

    # Discover models; keep known ordering at front, then anything extra.
    found = sorted(d.name for d in in_dir.iterdir() if d.is_dir())
    ordered = [m for m in MODEL_ORDER if m in found] + \
              [m for m in found if m not in MODEL_ORDER]
    print(f"Aggregating {len(ordered)} model(s) from {in_dir}:")
    for m in ordered:
        print(f"  - {m}")

    for model in ordered:
        csv_path = in_dir / model / "summary.csv"
        if not csv_path.is_file():
            print(f"  [WARN] missing {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)

        for stream in ("Q", "K"):
            ss = df[df["stream"] == stream]
            if ss.empty:
                continue
            # Band-class buckets by C_k threshold.
            low = ss[ss["C_k"] > 0.5]
            high = ss[ss["C_k"] <= 0.5]
            all_summary.append(_summary_row(ss, model, stream, "all"))
            all_summary.append(_summary_row(low, model, stream, "low_freq"))
            all_summary.append(_summary_row(high, model, stream, "high_freq"))
            all_per_band.extend(_per_band_rows(ss, model, stream))
            all_per_layer.extend(_per_layer_rows(ss, model, stream))

    # Write outputs.
    summary_df = pd.DataFrame(all_summary)
    per_band_df = pd.DataFrame(all_per_band)
    per_layer_df = pd.DataFrame(all_per_layer)

    summary_path = out_dir / "cross_model_summary.csv"
    per_band_path = out_dir / "per_band_profile.csv"
    per_layer_path = out_dir / "per_layer_profile.csv"

    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    per_band_df.to_csv(per_band_path, index=False, float_format="%.6f")
    per_layer_df.to_csv(per_layer_path, index=False, float_format="%.6f")
    print(f"\nWrote {summary_path}  ({len(summary_df)} rows)")
    print(f"Wrote {per_band_path}  ({len(per_band_df)} rows)")
    print(f"Wrote {per_layer_path} ({len(per_layer_df)} rows)")

    # Console table: paper-ready summary on "all bands" rows.
    all_rows = summary_df[summary_df["band_class"] == "all"].copy()
    display = all_rows[[
        "model", "stream", "n",
        "rho_mean", "rho_median", "rho_p95", "rho_p99", "rho_max",
        "rho_frac_gt_05", "rho_frac_gt_10", "rho_frac_gt_20",
        "varhet_mean", "varhet_p90", "varhet_max",
    ]]
    print("\n" + "=" * 92)
    print("Cross-model summary (band_class=all)")
    print("=" * 92)
    with pd.option_context("display.float_format", "{:.4f}".format,
                            "display.max_columns", None,
                            "display.width", 200):
        print(display.to_string(index=False))

    # Cross-model means by stream (for the one-sentence paper claim).
    print("\nCross-family aggregates (mean across 6 models, band_class=all):")
    for stream in ("Q", "K"):
        sub = all_rows[all_rows["stream"] == stream]
        if sub.empty:
            continue
        print(f"  Stream {stream}:")
        print(f"    mean |rho|            = {sub['rho_mean'].mean():.4f} "
              f"+/- {sub['rho_mean'].std(ddof=0):.4f}")
        print(f"    mean |log sig2 ratio| = {sub['varhet_mean'].mean():.4f} "
              f"+/- {sub['varhet_mean'].std(ddof=0):.4f}")
        print(f"    frac |rho| > 0.05     = {sub['rho_frac_gt_05'].mean():.4f}")


if __name__ == "__main__":
    main()
