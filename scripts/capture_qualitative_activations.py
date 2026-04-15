#!/usr/bin/env python3
"""Capture residual-stream input activations under four rotation conditions.

Produces one .npz per (model, config) whose contents are the raw per-layer
activation tensors needed to draw the qualitative figures comparing

    raw        : pristine FP16 Llama (no LN fusion, no rotation)
    whip       : LN fused + R1 trained with Whip loss + R1 applied
    swd_unif   : LN fused + R1 trained with SWD-Uniform loss + R1 applied
    swd_gauss  : LN fused + R1 trained with SWD-Gaussian loss + R1 applied

Everything downstream of R1 (R2, R3/R4, weight quant, activation quant) is
INTENTIONALLY skipped: the qualitative claim the paper makes is about how the
loss reshapes the activation distribution, which is already fully visible at
the output of R1. Skipping the rest keeps the script fast, deterministic, and
trivially comparable across the three losses.

Hook point choice (--hook): defaults to the INPUT of every layer's ``up_proj``
(MLP input). This is in the HIDDEN dimension (d_model) and so lives in the
space R1 rotates — making R1's flattening effect visible.

IMPORTANT: do NOT hook ``down_proj`` if you want to see R1's effect. The input
to ``down_proj`` is in the FFN/intermediate dimension, where R1 has cancelled
out mathematically (W_up <- W_up @ R1 on input, W_down <- R1^T @ W_down on
output), so raw and rotated configs give numerically identical tensors at
that point. Use ``--hook up_proj`` (default) / ``gate_proj`` / ``q_proj`` /
``k_proj`` / ``v_proj`` / ``o_proj`` to see a rotated hidden-dim tensor.

Data / model loading mirror dartquant_v2/unified_model.py and
scripts/measure_rho_rope.py exactly: local-only HF cache via HF_HUB_CACHE /
HF_HOME, and DartQuant's ``data_utils.get_loaders`` for the WikiText-2
calibration split when available.

Usage
-----
    python scripts/capture_qualitative_activations.py \
        --model meta-llama/Llama-3.2-1B --config raw \
        --out_dir artifacts/qualitative/llama-3.2-1b/raw

    python scripts/capture_qualitative_activations.py \
        --model meta-llama/Llama-3.2-1B --config whip \
        --out_dir artifacts/qualitative/llama-3.2-1b/whip

    python scripts/capture_qualitative_activations.py \
        --model meta-llama/Llama-3.2-1B --config swd_unif \
        --out_dir artifacts/qualitative/llama-3.2-1b/swd_unif

    python scripts/capture_qualitative_activations.py \
        --model meta-llama/Llama-3.2-1B --config swd_gauss \
        --out_dir artifacts/qualitative/llama-3.2-1b/swd_gauss
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup (mirrors run_quantize.py / measure_rho_rope.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
for _p in (
    os.path.join(_PROJECT_ROOT, "DartQuant", "fake_quant"),
    os.path.join(_PROJECT_ROOT, "DartQuant", "calibrater"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Local-only HF cache (matches dartquant_v2/run_quantize.py)
_DEFAULT_HF_HOME = "/root/autodl-tmp/huggingface"
_HF_HOME = os.environ.get("HF_HOME", _DEFAULT_HF_HOME)
os.environ.setdefault("HF_HOME", _HF_HOME)
os.environ.setdefault("HF_HUB_CACHE", _HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_HOME)
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    os.environ.get("HF_DATASETS_CACHE", "/root/autodl-tmp/datasets"),
)
if os.environ.get("TRANSFORMERS_OFFLINE") != "0":
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
if os.environ.get("HF_DATASETS_OFFLINE") != "0":
    os.environ["HF_DATASETS_OFFLINE"] = "1"

from dartquant_v2.unified_model import UnifiedQuantModel
from dartquant_v2.pipeline import (
    _untie_word_embeddings,
    fuse_layer_norms,
    apply_r1_rotation,
    collect_activations,
    cleanup_memory,
)
from dartquant_v2.trainers import train_r1_single_layer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
log = logging.getLogger("capture_qual")

DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Order the four configs deliberately so plotting can consume them left-to-right
CONFIG_CHOICES = ("raw", "whip", "swd_unif", "swd_gauss")


# ---------------------------------------------------------------------------
# Calibration loader (same two-stage fallback as measure_rho_rope.py)
# ---------------------------------------------------------------------------
def _load_calib_dartquant(model_name: str, nsamples: int, seqlen: int,
                          seed: int) -> torch.Tensor | None:
    try:
        import data_utils
    except Exception as e:
        log.info("DartQuant data_utils unavailable (%s); HF fallback", e)
        return None
    try:
        loader = data_utils.get_loaders(
            "wikitext2", nsamples=nsamples, seed=seed,
            model=model_name, seqlen=seqlen, eval_mode=False,
        )
        calib = torch.cat([b[0] for b in loader], dim=0)
        log.info("Loaded calibration via DartQuant data_utils: %s",
                 tuple(calib.shape))
        return calib
    except Exception as e:
        log.warning("DartQuant data_utils failed (%s); HF fallback", e)
        return None


def _load_calib_hf(tokenizer, nsamples: int, seqlen: int,
                   seed: int) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.shape[0] < seqlen + 1:
        raise RuntimeError(
            f"Corpus too small: {ids.shape[0]} tokens < seqlen+1 = {seqlen+1}"
        )
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, ids.shape[0] - seqlen - 1, size=nsamples)
    calib = torch.stack([ids[s:s + seqlen] for s in starts])
    log.info("Loaded calibration via HF datasets: %s", tuple(calib.shape))
    return calib


def load_calibration(model_name: str, tokenizer, nsamples: int,
                     seqlen: int, seed: int) -> torch.Tensor:
    c = _load_calib_dartquant(model_name, nsamples, seqlen, seed)
    if c is not None:
        return c
    return _load_calib_hf(tokenizer, nsamples, seqlen, seed)


# ---------------------------------------------------------------------------
# R1 training wrapper (mirrors pipeline.run_full_pipeline steps 4-5, stripped)
# ---------------------------------------------------------------------------
def _collect_r1_activations(model, umodel: UnifiedQuantModel,
                            calib: torch.Tensor, max_rows_per_hook: int
                            ) -> torch.Tensor:
    """Collect the R1-training activation pool: q_proj and up_proj inputs,
    concatenated across all layers — matches pipeline.run_full_pipeline.
    """
    layers_prefix = umodel.arch.layers_path
    target_names = []
    for layer_idx in range(umodel.num_layers):
        if umodel.arch.mlp_up_proj_attr:
            target_names.append(
                f"{layers_prefix}.{layer_idx}.{umodel.arch.mlp_up_proj_attr}"
            )
        target_names.append(
            f"{layers_prefix}.{layer_idx}.{umodel.arch.q_proj_attr}"
        )
    acts = collect_activations(
        model, calib, target_names, DEV,
        max_rows_per_hook=max_rows_per_hook,
    )
    rows = []
    for name in list(acts.keys()):
        t = acts.pop(name)
        rows.append(t.reshape(-1, umodel.hidden_size))
    combined = torch.cat(rows, dim=0)
    del rows
    gc.collect()
    return combined


def train_and_apply_r1(model, umodel: UnifiedQuantModel, calib: torch.Tensor,
                       loss_name: str, r1_epochs: int, r1_batch_size: int,
                       max_rows_per_hook: int, seed: int) -> None:
    """Collect R1 activations → train global R1 with the given loss → bake in.

    Hyperparameters follow pipeline.run_full_pipeline's auto-tuning logic
    for SWD/Gauss losses so the visualisation reflects the same training
    regime the paper reports.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    _DIST = ("swd_unif", "swd_gauss", "kl_unif", "kl_gauss",
             "bin_kl_unif", "bin_kl_nf4")
    if loss_name in _DIST:
        r1_epochs = max(r1_epochs, 30)
        r1_batch_size = max(r1_batch_size, 256)
        optim = "adam"
        cos_lr = True
        lr = 1e-3
    else:
        optim = "sgd"
        cos_lr = False
        lr = 1e-3
    if loss_name == "swd_gauss":
        lr = 1e-2
        r1_epochs = max(r1_epochs, 100)

    log.info("R1 training config: loss=%s epochs=%d bs=%d optim=%s lr=%g cos=%s",
             loss_name, r1_epochs, r1_batch_size, optim, lr, cos_lr)

    acts = _collect_r1_activations(model, umodel, calib, max_rows_per_hook)
    log.info("R1 activation pool: %s", tuple(acts.shape))

    R1 = train_r1_single_layer(
        acts=acts,
        hidden_size=umodel.hidden_size,
        loss_fn_name=loss_name,
        lr=lr,
        momentum=0.9,
        epochs=r1_epochs,
        batch_size=r1_batch_size,
        cos_lr=cos_lr,
        optim=optim,
        init_mode="hadamard",
        accumulation_steps=1,
        train_subset_size=0.1,
        device=DEV,
        layer_idx=0,
    )
    del acts
    cleanup_memory()

    log.info("Applying R1 to model weights (%d x %d)", R1.shape[0], R1.shape[1])
    apply_r1_rotation(model, R1, umodel, smooth_scale=None)
    cleanup_memory()


# ---------------------------------------------------------------------------
# Down-proj input capture
# ---------------------------------------------------------------------------
_HOOK_ATTR_MAP = {
    "up_proj":   "mlp_up_proj_attr",
    "gate_proj": "mlp_gate_proj_attr",
    "down_proj": "mlp_down_proj_attr",
    "q_proj":    "q_proj_attr",
    "k_proj":    "k_proj_attr",
    "v_proj":    "v_proj_attr",
    "o_proj":    "o_proj_attr",
}


def _resolve_hook_attr(umodel: UnifiedQuantModel, hook: str) -> str:
    field = _HOOK_ATTR_MAP[hook]
    attr = getattr(umodel.arch, field, None)
    if attr is None:
        raise ValueError(
            f"Architecture {type(umodel.arch).__name__} has no '{field}' "
            f"(hook='{hook}'). Pick a different --hook."
        )
    return attr


def capture_submodule_inputs(model, umodel: UnifiedQuantModel,
                             calib: torch.Tensor,
                             rows_per_layer: int,
                             hook: str) -> dict[int, torch.Tensor]:
    """Hook every layer's `hook` INPUT, return {layer_idx: (rows, dim)}."""
    layers_prefix = umodel.arch.layers_path
    hook_attr = _resolve_hook_attr(umodel, hook)
    target_names = [
        f"{layers_prefix}.{i}.{hook_attr}"
        for i in range(umodel.num_layers)
    ]
    acts = collect_activations(
        model, calib, target_names, DEV,
        max_rows_per_hook=rows_per_layer,
    )
    out: dict[int, torch.Tensor] = {}
    for i, name in enumerate(target_names):
        if name not in acts:
            continue
        t = acts[name]
        out[i] = t.reshape(-1, t.shape[-1]).float().contiguous()
    return out


def compress_per_layer(acts: dict[int, torch.Tensor], max_rows: int,
                       ) -> dict[int, torch.Tensor]:
    """Deterministic row subsample so the saved .npz stays small."""
    rng = np.random.default_rng(0)
    out = {}
    for i, t in acts.items():
        if t.shape[0] > max_rows:
            idx = rng.choice(t.shape[0], size=max_rows, replace=False)
            out[i] = t[idx].clone()
        else:
            out[i] = t.clone()
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True,
                   help="HF model id, e.g. meta-llama/Llama-3.2-1B")
    p.add_argument("--config", required=True, choices=CONFIG_CHOICES,
                   help="Which rotation condition to capture")
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--hook", default="up_proj",
                   choices=["up_proj", "gate_proj", "q_proj", "k_proj",
                            "v_proj", "o_proj", "down_proj"],
                   help="Which sub-module's INPUT to capture (default: up_proj). "
                        "up_proj / gate_proj / q_proj / k_proj / v_proj are in "
                        "the hidden dim and expose R1's rotation. "
                        "down_proj is in the FFN dim where R1 cancels — only "
                        "useful for showing R4's effect, not R1.")

    p.add_argument("--nsamples", type=int, default=32,
                   help="Calibration samples for R1 training and capture")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--r1_epochs", type=int, default=10,
                   help="R1 training epochs (raised automatically for SWD "
                        "losses to match pipeline.run_full_pipeline)")
    p.add_argument("--r1_batch_size", type=int, default=4096,
                   help="Rows per R1 gradient step")
    p.add_argument("--rows_per_layer_r1", type=int, default=256,
                   help="Rows subsampled per hook during R1 activation "
                        "collection (bounds CPU RAM)")
    p.add_argument("--rows_per_layer_capture", type=int, default=2048,
                   help="Rows per layer captured for visualisation")
    p.add_argument("--rows_per_layer_save", type=int, default=4096,
                   help="Maximum rows written to the output .npz per layer")

    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--hf_token", default=None)
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HF_HUB_CACHE"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir

    log.info("=" * 70)
    log.info("Capture qualitative activations")
    log.info("  model   = %s", args.model)
    log.info("  config  = %s", args.config)
    log.info("  out_dir = %s", args.out_dir)
    log.info("  device  = %s", DEV)
    log.info("=" * 70)

    t0 = time.time()

    # ---- 1. Load model (same path as pipeline.run_full_pipeline step 1) ----
    umodel = UnifiedQuantModel(
        args.model, args.hf_token, args.cache_dir, dtype=dtype,
    )
    model = umodel.model
    tokenizer = umodel.get_tokenizer()
    log.info("  arch=%s hidden=%d layers=%d ffn=%d",
             model.config.__class__.__name__,
             umodel.hidden_size, umodel.num_layers, umodel.intermediate_size)

    # ---- 2. Calibration data ----
    calib = load_calibration(args.model, tokenizer, args.nsamples,
                             args.seqlen, args.seed)
    model.to(DEV)
    model.eval()

    # ---- 3. Depending on config, optionally fuse LN + train/apply R1 ----
    if args.config == "raw":
        log.info("[raw] skipping LN fusion and R1 training — capturing "
                 "pristine pretrained activations")
    else:
        log.info("[%s] fusing layer norms", args.config)
        _untie_word_embeddings(umodel)
        fuse_layer_norms(umodel)
        model.to(DEV)  # fuse_layer_norms may move tensors around
        cleanup_memory()

        train_and_apply_r1(
            model=model, umodel=umodel, calib=calib,
            loss_name=args.config,
            r1_epochs=args.r1_epochs,
            r1_batch_size=args.r1_batch_size,
            max_rows_per_hook=args.rows_per_layer_r1,
            seed=args.seed,
        )
        model.to(DEV)
        cleanup_memory()

    # ---- 4. Capture sub-module inputs on every layer ----
    log.info("Capturing %s inputs (layers=%d, rows/layer<=%d)...",
             args.hook, umodel.num_layers, args.rows_per_layer_capture)
    capture_calib = calib[:min(args.nsamples, 16)]     # 16 sequences is enough
    acts = capture_submodule_inputs(model, umodel, capture_calib,
                                     args.rows_per_layer_capture,
                                     hook=args.hook)
    acts = compress_per_layer(acts, args.rows_per_layer_save)

    # ---- 5. Write out ----
    # Output filename reflects the hook so multiple hook points can coexist
    # in the same artifact directory without overwriting each other.
    # Legacy alias "down_proj_inputs.npz" preserved when --hook=down_proj so
    # the existing plot scripts keep working without a rename.
    if args.hook == "down_proj":
        out_npz = args.out_dir / "down_proj_inputs.npz"
    else:
        out_npz = args.out_dir / f"{args.hook}_inputs.npz"
    np_arrays = {}
    layer_idxs = sorted(acts.keys())
    for i in layer_idxs:
        np_arrays[f"layer_{i:03d}"] = acts[i].numpy().astype(np.float32)
    np_arrays["_layer_idxs"] = np.array(layer_idxs, dtype=np.int32)
    np_arrays["_model"] = np.array(args.model)
    np_arrays["_config"] = np.array(args.config)
    np_arrays["_hook"] = np.array(args.hook)
    np_arrays["_hidden_size"] = np.int32(umodel.hidden_size)
    np_arrays["_intermediate_size"] = np.int32(umodel.intermediate_size)
    np_arrays["_num_layers"] = np.int32(umodel.num_layers)
    np.savez_compressed(out_npz, **np_arrays)
    log.info("Wrote %s (%d layers, %.1f MB)",
             out_npz, len(layer_idxs), out_npz.stat().st_size / 1e6)
    log.info("Total wall time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
