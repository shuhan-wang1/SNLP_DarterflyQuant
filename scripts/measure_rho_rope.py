#!/usr/bin/env python3
"""Measure pre-RoPE intra-band correlation rho_k for Q/K activations of Llama.

Validates the rho ~= 0 assumption of Theorem (Post-RoPE variance) in the paper's
Appendix C (tex:641-681), and collects the co-factors the proofs depend on:
variance heterogeneity sigma_1^2/sigma_2^2 and the RoPE oscillation average
C_k = (1/L) sum_{m=1..L} cos(2 m theta_k). These three quantities jointly
determine whether fixed Hadamard R3/R4 is suboptimal in practice
(Proposition hadamard_subopt, Corollary hadamard_rope).

Band pairing matches HuggingFace's `rotate_half` convention used by Llama-3.x:
for each head, channel k is paired with channel k + head_dim/2 and rotated at
frequency theta_k = rope_theta^(-2k/head_dim) for k in 0..head_dim/2-1.

Measurement is one-pass streaming so GPU memory stays O(num_heads * head_dim)
per layer regardless of the number of calibration samples.

Outputs into --out_dir:
  raw_stats.npz   arrays rho/sigma1_sq/sigma2_sq/n indexed [layer, head, band],
                  separate entries for the Q and K streams, plus per-band
                  theta_k / C_k / S_k and model metadata.
  summary.csv     long-form flat table ready for plotting scripts.
  summary.txt     human-readable distribution summary (mean |rho|, percentiles,
                  stratified by high-frequency vs low-frequency bands).

Usage:
  python scripts/measure_rho_rope.py --model meta-llama/Llama-3.2-1B
  python scripts/measure_rho_rope.py --model meta-llama/Llama-3.1-8B \
      --nsamples 128 --seqlen 2048 --out_dir artifacts/rho/llama-3.1-8b
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import transformers

# Project path setup (matches diagnose_rotation.py)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_DQ_FQ = os.path.join(_PROJECT_ROOT, "DartQuant", "fake_quant")
if _DQ_FQ not in sys.path:
    sys.path.insert(0, _DQ_FQ)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("rho_rope")


# ---------------------------------------------------------------------------
# Calibration-data loaders
# ---------------------------------------------------------------------------
def _load_calib_via_dartquant(model_name: str, nsamples: int, seqlen: int,
                              seed: int) -> torch.Tensor | None:
    """Prefer DartQuant's data_utils so rho numbers match the paper's R1 run."""
    try:
        import data_utils
    except Exception as e:
        log.info("data_utils unavailable (%s); falling back to HF datasets", e)
        return None
    try:
        loader = data_utils.get_loaders(
            "wikitext2", nsamples=nsamples, seed=seed,
            model=model_name, seqlen=seqlen, eval_mode=False,
        )
        calib = torch.cat([b[0] for b in loader], dim=0)
        log.info("Loaded calibration via DartQuant data_utils: %s", tuple(calib.shape))
        return calib
    except Exception as e:
        log.warning("DartQuant data_utils failed (%s); falling back to HF datasets", e)
        return None


def _load_calib_via_hf_datasets(tokenizer, nsamples: int, seqlen: int,
                                seed: int) -> torch.Tensor:
    """Fallback: sample 128 random windows of seqlen 2048 from WikiText-2 train."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n = ids.shape[0]
    if n < seqlen + 1:
        raise RuntimeError(f"Corpus too small: {n} tokens < seqlen+1 = {seqlen+1}")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - seqlen - 1, size=nsamples)
    calib = torch.stack([ids[s:s + seqlen] for s in starts])
    log.info("Loaded calibration via HF datasets: %s", tuple(calib.shape))
    return calib


def load_calibration(model_name: str, tokenizer, nsamples: int,
                     seqlen: int, seed: int) -> torch.Tensor:
    calib = _load_calib_via_dartquant(model_name, nsamples, seqlen, seed)
    if calib is not None:
        return calib
    return _load_calib_via_hf_datasets(tokenizer, nsamples, seqlen, seed)


# ---------------------------------------------------------------------------
# RoPE frequencies and oscillation averages
# ---------------------------------------------------------------------------
def compute_rope_quantities(head_dim: int, rope_theta: float,
                            seqlen: int) -> dict[str, np.ndarray]:
    """theta_k, C_k, S_k for k = 0..head_dim/2-1 and sequence length L."""
    half = head_dim // 2
    k = np.arange(half, dtype=np.float64)
    theta_k = rope_theta ** (-2.0 * k / head_dim)  # (half,)
    m = np.arange(1, seqlen + 1, dtype=np.float64)  # (L,)
    # 2 m theta_k: (L, half)
    arg = 2.0 * np.outer(m, theta_k)
    C_k = np.cos(arg).mean(axis=0)  # (half,)
    S_k = np.sin(arg).mean(axis=0)  # (half,)
    return {"theta_k": theta_k, "C_k": C_k, "S_k": S_k}


# ---------------------------------------------------------------------------
# Streaming per-(head, band) accumulator
# ---------------------------------------------------------------------------
class BandStats:
    """Streaming sums-of-products for per-(head, band) Pearson correlation.

    For each head h and band k, we track scalar accumulators (as fp64 tensors
    on CPU to avoid precision loss) for channels a = x[..., k] and
    b = x[..., k + head_dim/2]:
        n, sum_a, sum_b, sum_aa, sum_bb, sum_ab
    After ingest, rho = (sum_ab/n - mean_a*mean_b) / sqrt(var_a * var_b).
    """

    def __init__(self, num_heads: int, head_dim: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.half = head_dim // 2
        shape = (num_heads, self.half)
        self.n = 0
        self.sum_a = torch.zeros(shape, dtype=torch.float64)
        self.sum_b = torch.zeros(shape, dtype=torch.float64)
        self.sum_aa = torch.zeros(shape, dtype=torch.float64)
        self.sum_bb = torch.zeros(shape, dtype=torch.float64)
        self.sum_ab = torch.zeros(shape, dtype=torch.float64)

    def update(self, x_fp32: torch.Tensor) -> None:
        """x_fp32: (N, num_heads, head_dim) on any device."""
        assert x_fp32.ndim == 3 and x_fp32.shape[1:] == (self.num_heads, self.head_dim), \
            f"BandStats expected (N, {self.num_heads}, {self.head_dim}), got {tuple(x_fp32.shape)}"
        a = x_fp32[..., :self.half]             # (N, H, half)
        b = x_fp32[..., self.half:]             # (N, H, half)
        N = a.shape[0]
        self.n += N
        # Reductions then move to CPU fp64 for stable accumulation.
        self.sum_a += a.sum(dim=0).double().cpu()
        self.sum_b += b.sum(dim=0).double().cpu()
        self.sum_aa += (a * a).sum(dim=0).double().cpu()
        self.sum_bb += (b * b).sum(dim=0).double().cpu()
        self.sum_ab += (a * b).sum(dim=0).double().cpu()

    def finalize(self) -> dict[str, np.ndarray]:
        """Returns rho, sigma1_sq, sigma2_sq, mean_a, mean_b as (H, half) arrays."""
        if self.n == 0:
            raise RuntimeError("BandStats.finalize called with n=0")
        n = float(self.n)
        mean_a = self.sum_a / n
        mean_b = self.sum_b / n
        var_a = self.sum_aa / n - mean_a * mean_a
        var_b = self.sum_bb / n - mean_b * mean_b
        cov = self.sum_ab / n - mean_a * mean_b
        # Numerical floor to avoid div-by-zero on dead channels.
        denom = torch.sqrt(var_a.clamp(min=0.0) * var_b.clamp(min=0.0))
        rho = torch.where(denom > 0, cov / denom.clamp(min=1e-30),
                          torch.zeros_like(cov))
        rho = rho.clamp(-1.0, 1.0)
        return {
            "rho": rho.numpy(),
            "sigma1_sq": var_a.numpy(),
            "sigma2_sq": var_b.numpy(),
            "mean_a": mean_a.numpy(),
            "mean_b": mean_b.numpy(),
            "n": int(self.n),
        }


# ---------------------------------------------------------------------------
# Main measurement
# ---------------------------------------------------------------------------
def measure(model, tokenizer, calib: torch.Tensor, device: torch.device,
            max_layers: int | None = None) -> dict:
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_q_heads = cfg.num_attention_heads
    num_k_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = getattr(cfg, "head_dim",
                       cfg.hidden_size // cfg.num_attention_heads)
    rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
    seqlen = calib.shape[1]
    log.info(
        "Model: layers=%d q_heads=%d kv_heads=%d head_dim=%d rope_theta=%g seqlen=%d",
        num_layers, num_q_heads, num_k_heads, head_dim, rope_theta, seqlen,
    )

    if max_layers is not None and max_layers < num_layers:
        log.info("Restricting to first %d layers (of %d)", max_layers, num_layers)
        layers_to_hook = list(range(max_layers))
    else:
        layers_to_hook = list(range(num_layers))

    # Find the transformer block list in a robust way
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        raise RuntimeError(
            "Could not find model.model.layers -- is this a Llama-family model?"
        )

    q_stats = {i: BandStats(num_q_heads, head_dim) for i in layers_to_hook}
    k_stats = {i: BandStats(num_k_heads, head_dim) for i in layers_to_hook}

    def _make_hook(stats: BandStats, num_heads: int):
        def _hook(_module, _inputs, output):
            # q_proj/k_proj output: (B, S, num_heads * head_dim)
            x = output
            if isinstance(x, tuple):
                x = x[0]
            x = x.detach().to(torch.float32)
            B, S, D = x.shape
            assert D == num_heads * head_dim, \
                f"Unexpected proj output dim {D} (expected {num_heads}*{head_dim})"
            x = x.reshape(B * S, num_heads, head_dim)
            stats.update(x)
        return _hook

    handles = []
    for i in layers_to_hook:
        attn = layers[i].self_attn
        handles.append(attn.q_proj.register_forward_hook(
            _make_hook(q_stats[i], num_q_heads)))
        handles.append(attn.k_proj.register_forward_hook(
            _make_hook(k_stats[i], num_k_heads)))

    model.eval()
    try:
        with torch.no_grad():
            for s in range(calib.shape[0]):
                ids = calib[s:s + 1].to(device)
                model(ids, use_cache=False)
                if (s + 1) % 16 == 0 or s == calib.shape[0] - 1:
                    log.info("  processed %d / %d samples", s + 1, calib.shape[0])
    finally:
        for h in handles:
            h.remove()

    q_out = {i: q_stats[i].finalize() for i in layers_to_hook}
    k_out = {i: k_stats[i].finalize() for i in layers_to_hook}

    rope = compute_rope_quantities(head_dim, rope_theta, seqlen)
    return {
        "q_stats": q_out,
        "k_stats": k_out,
        "rope": rope,
        "meta": {
            "num_layers": num_layers,
            "layers_hooked": layers_to_hook,
            "num_q_heads": num_q_heads,
            "num_k_heads": num_k_heads,
            "head_dim": head_dim,
            "rope_theta": rope_theta,
            "seqlen": seqlen,
            "convention": "hf_rotate_half",
        },
    }


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def write_outputs(result: dict, out_dir: Path, model_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = result["meta"]
    rope = result["rope"]
    theta_k = rope["theta_k"]
    C_k = rope["C_k"]
    S_k = rope["S_k"]

    # --- raw_stats.npz ----------------------------------------------------
    def _stack(stream_dict: dict) -> dict[str, np.ndarray]:
        idxs = sorted(stream_dict.keys())
        rho = np.stack([stream_dict[i]["rho"] for i in idxs])
        s1 = np.stack([stream_dict[i]["sigma1_sq"] for i in idxs])
        s2 = np.stack([stream_dict[i]["sigma2_sq"] for i in idxs])
        ma = np.stack([stream_dict[i]["mean_a"] for i in idxs])
        mb = np.stack([stream_dict[i]["mean_b"] for i in idxs])
        return {"layer_idx": np.array(idxs, dtype=np.int32),
                "rho": rho, "sigma1_sq": s1, "sigma2_sq": s2,
                "mean_a": ma, "mean_b": mb}

    q_arr = _stack(result["q_stats"])
    k_arr = _stack(result["k_stats"])
    np.savez(
        out_dir / "raw_stats.npz",
        model_name=np.array(model_name),
        q_layer_idx=q_arr["layer_idx"], q_rho=q_arr["rho"],
        q_sigma1_sq=q_arr["sigma1_sq"], q_sigma2_sq=q_arr["sigma2_sq"],
        q_mean_a=q_arr["mean_a"], q_mean_b=q_arr["mean_b"],
        k_layer_idx=k_arr["layer_idx"], k_rho=k_arr["rho"],
        k_sigma1_sq=k_arr["sigma1_sq"], k_sigma2_sq=k_arr["sigma2_sq"],
        k_mean_a=k_arr["mean_a"], k_mean_b=k_arr["mean_b"],
        theta_k=theta_k, C_k=C_k, S_k=S_k,
        head_dim=np.int32(meta["head_dim"]),
        rope_theta=np.float64(meta["rope_theta"]),
        seqlen=np.int32(meta["seqlen"]),
        num_q_heads=np.int32(meta["num_q_heads"]),
        num_k_heads=np.int32(meta["num_k_heads"]),
    )
    log.info("Wrote %s", out_dir / "raw_stats.npz")

    # --- summary.csv ------------------------------------------------------
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "layer", "stream", "head", "band",
            "theta_k", "C_k", "S_k",
            "rho", "sigma1_sq", "sigma2_sq",
            "var_ratio",
            "rope_var_rho0", "rope_var_full",
        ])
        for stream_name, arr in (("Q", q_arr), ("K", k_arr)):
            rho = arr["rho"]         # (Llayers, H, half)
            s1 = arr["sigma1_sq"]
            s2 = arr["sigma2_sq"]
            for li, layer in enumerate(arr["layer_idx"]):
                for h in range(rho.shape[1]):
                    for b in range(rho.shape[2]):
                        sigma1_sq = float(s1[li, h, b])
                        sigma2_sq = float(s2[li, h, b])
                        r = float(rho[li, h, b])
                        ratio = sigma1_sq / sigma2_sq if sigma2_sq > 0 else float("nan")
                        # Theorem prediction under rho=0 and full-rho:
                        v_rho0 = 0.5 * (sigma1_sq + sigma2_sq) \
                            + 0.5 * (sigma1_sq - sigma2_sq) * float(C_k[b])
                        v_full = v_rho0 - r * math.sqrt(max(sigma1_sq, 0.0) *
                                                       max(sigma2_sq, 0.0)) * float(S_k[b])
                        w.writerow([
                            int(layer), stream_name, h, b,
                            f"{float(theta_k[b]):.9e}",
                            f"{float(C_k[b]):.6f}",
                            f"{float(S_k[b]):.6f}",
                            f"{r:.6f}",
                            f"{sigma1_sq:.6e}",
                            f"{sigma2_sq:.6e}",
                            f"{ratio:.6f}",
                            f"{v_rho0:.6e}",
                            f"{v_full:.6e}",
                        ])
    log.info("Wrote %s", csv_path)

    # --- summary.txt (human-readable) -------------------------------------
    def _dist_summary(rho: np.ndarray, tag: str) -> str:
        a = np.abs(rho).ravel()
        pcts = np.percentile(a, [50, 90, 95, 99])
        lines = [
            f"[{tag}]  n={a.size}",
            f"  mean |rho|     = {a.mean():.4f}",
            f"  median |rho|   = {pcts[0]:.4f}",
            f"  p90  |rho|     = {pcts[1]:.4f}",
            f"  p95  |rho|     = {pcts[2]:.4f}",
            f"  p99  |rho|     = {pcts[3]:.4f}",
            f"  max  |rho|     = {a.max():.4f}",
            f"  frac |rho|>.05 = {(a > 0.05).mean():.4f}",
            f"  frac |rho|>.10 = {(a > 0.10).mean():.4f}",
            f"  frac |rho|>.20 = {(a > 0.20).mean():.4f}",
        ]
        return "\n".join(lines)

    # Low-frequency bands (C_k > 0.5) are where the Hadamard failure mode
    # bites hardest per Corollary hadamard_rope.
    low_freq = C_k > 0.5           # (half,)
    high_freq = ~low_freq
    lines = [f"# rho / variance summary for {model_name}", ""]
    lines.append(f"meta: layers_hooked={len(q_arr['layer_idx'])}, "
                 f"q_heads={meta['num_q_heads']}, kv_heads={meta['num_k_heads']}, "
                 f"head_dim={meta['head_dim']}, rope_theta={meta['rope_theta']:g}, "
                 f"seqlen={meta['seqlen']}")
    lines.append(f"RoPE bands: {int(low_freq.sum())} low-freq (C_k>0.5), "
                 f"{int(high_freq.sum())} high-freq")
    lines.append("")
    for stream_name, arr in (("Q", q_arr), ("K", k_arr)):
        rho = arr["rho"]          # (L, H, half)
        lines.append(f"## Stream {stream_name}")
        lines.append(_dist_summary(rho, "all bands"))
        if low_freq.any():
            lines.append(_dist_summary(rho[..., low_freq], "low-freq bands (C_k>0.5)"))
        if high_freq.any():
            lines.append(_dist_summary(rho[..., high_freq], "high-freq bands (C_k<=0.5)"))
        # Variance heterogeneity: fraction of pairs with sigma1^2 / sigma2^2
        # outside [0.8, 1.25] (i.e. >25% variance asymmetry).
        s1 = arr["sigma1_sq"]; s2 = arr["sigma2_sq"]
        ratio = np.where(s2 > 0, s1 / np.clip(s2, 1e-30, None), np.nan)
        het = np.abs(np.log(np.clip(ratio, 1e-30, None)))   # |log ratio|
        lines.append(
            f"  variance heterogeneity |log(sigma1^2/sigma2^2)|: "
            f"mean={np.nanmean(het):.3f}, p90={np.nanpercentile(het, 90):.3f}, "
            f"max={np.nanmax(het):.3f}"
        )
        lines.append("")
    lines.append("# Conclusions:")
    lines.append("# - If mean |rho| << 0.05 across all bands, Theorem's rho=0 "
                 "assumption is well-supported and the post-RoPE variance formula "
                 "is predictive.")
    lines.append("# - The Hadamard failure mode (Corollary hadamard_rope) needs "
                 "EITHER non-zero rho OR sigma_1^2 != sigma_2^2 within a band. "
                 "Check the variance-heterogeneity line: if |log(sigma1^2/sigma2^2)| "
                 "is materially > 0, the Hadamard-suboptimality argument holds "
                 "independent of rho.")
    txt = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(txt)
    log.info("Wrote %s", out_dir / "summary.txt")
    print("\n" + txt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id, e.g. meta-llama/Llama-3.2-1B")
    p.add_argument("--nsamples", type=int, default=128,
                   help="Calibration samples (matches DartQuant default)")
    p.add_argument("--seqlen", type=int, default=2048,
                   help="Calibration sequence length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", type=str, default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_layers", type=int, default=None,
                   help="If set, hook only the first N layers (smoke test)")
    p.add_argument("--hf_token", type=str, default=None,
                   help="HF token; falls back to HF_TOKEN env var")
    args = p.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    token = args.hf_token or os.environ.get("HF_TOKEN")
    tok_kwargs = {"token": token} if token else {}

    log.info("Loading tokenizer %s", args.model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    log.info("Loading model %s in %s", args.model, args.dtype)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, **tok_kwargs,
    )
    model.to(device)
    model.eval()

    calib = load_calibration(args.model, tokenizer, args.nsamples, args.seqlen, args.seed)
    result = measure(model, tokenizer, calib, device, max_layers=args.max_layers)
    write_outputs(result, Path(args.out_dir), args.model)


if __name__ == "__main__":
    main()
