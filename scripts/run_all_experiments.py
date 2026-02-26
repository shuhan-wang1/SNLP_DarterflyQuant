#!/usr/bin/env python3
"""
Unified One-Click Experiment Runner for DartQuant v2.

Automatically discovers locally cached models and datasets, then runs all
comparison and ablation experiments.

Experiment groups:
  1. Loss Function Comparison:  SWD-Gaussian vs SWD-Uniform vs Whip
  2. Butterfly Ablation:        Butterfly R3/R4 vs Hadamard R3/R4

Usage:
  # Run everything (auto-detect models & datasets)
  python scripts/run_all_experiments.py

  # Filter to specific experiment group
  python scripts/run_all_experiments.py --group comparison
  python scripts/run_all_experiments.py --group ablation

  # Specify models explicitly (skip auto-detection)
  python scripts/run_all_experiments.py --models meta-llama/Llama-3.2-1B

  # Dry run (print commands without executing)
  python scripts/run_all_experiments.py --dry-run
"""

import os
import sys
import json
import argparse
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Cache directory resolution (same logic as run_quantize.py)
# ---------------------------------------------------------------------------
_DEFAULT_HF_HOME = "/root/autodl-tmp/huggingface"
_HF_HOME = os.environ.get("HF_HOME", _DEFAULT_HF_HOME)
_DATASETS_CACHE = os.environ.get(
    "HF_DATASETS_CACHE", os.path.join(_HF_HOME, "datasets")
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known models & datasets
# ---------------------------------------------------------------------------
KNOWN_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-8B",
    "meta-llama/Llama-3.2-8B-Instruct",
    "facebook/opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-6.7b",
]

# Evaluation datasets supported by the pipeline (perplexity)
EVAL_DATASETS = ["wikitext2", "ptb", "c4"]

# Default lm_eval zero-shot tasks
LM_EVAL_TASKS = ["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "mmlu"]

# Calibration datasets
CAL_DATASETS = ["wikitext2", "ptb", "c4"]


# ============================================================================
# Auto-detection helpers
# ============================================================================

def detect_cached_models(cache_dir: str) -> list[str]:
    """Scan HuggingFace hub cache for locally downloaded models.

    HF stores models under:
      <cache_dir>/hub/models--<org>--<name>/snapshots/<hash>/
    or directly under:
      <cache_dir>/models--<org>--<name>/snapshots/<hash>/
    """
    found = []

    # Search patterns: <cache_dir>/hub/models--*, <cache_dir>/models--*
    search_roots = [
        Path(cache_dir) / "hub",
        Path(cache_dir),
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if not name.startswith("models--"):
                continue
            # models--meta-llama--Llama-3.2-1B  → meta-llama/Llama-3.2-1B
            parts = name[len("models--"):].split("--", 1)
            if len(parts) == 2:
                model_id = f"{parts[0]}/{parts[1]}"
            else:
                model_id = parts[0]

            # Verify there's at least one snapshot
            snapshots = entry / "snapshots"
            if snapshots.is_dir() and any(snapshots.iterdir()):
                if model_id not in found:
                    found.append(model_id)

    return sorted(found)


def detect_cached_datasets(datasets_cache: str) -> list[str]:
    """Check which evaluation datasets are available locally.

    For each dataset name (wikitext2, ptb, c4), we try a fast import check
    via the datasets library fingerprint structure.  Falls back to searching
    for known directory patterns.
    """
    available = []

    # Quick heuristic: look for common directory patterns
    ds_dir = Path(datasets_cache)
    if not ds_dir.exists():
        # Also check the parent hub directory
        ds_dir = Path(_HF_HOME)

    dataset_markers = {
        "wikitext2": [
            "wikitext",
            "wikitext-2-raw-v1",
            "wikitext___wikitext-2-raw-v1",
        ],
        "ptb": [
            "ptb-text-only",
            "ptb_text_only",
            "xeon09112___ptb_text_only",
        ],
        "c4": [
            "allenai___c4",
            "c4",
        ],
    }

    for ds_name, markers in dataset_markers.items():
        for root in [Path(datasets_cache), Path(_HF_HOME), Path(_HF_HOME) / "hub"]:
            if not root.exists():
                continue
            for marker in markers:
                # Check direct child directories
                if (root / marker).is_dir():
                    available.append(ds_name)
                    break
                # Check nested patterns (datasets/wikitext/...)
                for child in root.iterdir():
                    if child.is_dir() and marker in child.name:
                        available.append(ds_name)
                        break
                if ds_name in available:
                    break
            if ds_name in available:
                break

    return sorted(set(available))


# ============================================================================
# Experiment definitions
# ============================================================================

def build_comparison_experiments(
    models: list[str],
    eval_datasets: list[str],
) -> list[dict]:
    """Loss function comparison: SWD-Gaussian vs SWD-Uniform vs Whip.

    - whip + int4:      Original DartQuant exponential repulsion
    - swd_unif + int4:  Sliced Wasserstein Distance to Uniform
    - swd_gauss + nf4:  Sliced Wasserstein Distance to Gaussian
    """
    experiments = []
    configs = [
        {"loss": "whip",      "quantizer_type": "int4", "tag": "whip_int4"},
        {"loss": "swd_unif",  "quantizer_type": "int4", "tag": "swd_unif_int4"},
        {"loss": "swd_gauss", "quantizer_type": "nf4",  "tag": "swd_gauss_nf4"},
    ]

    for model in models:
        for cfg in configs:
            exp = {
                "name": f"comparison__{_model_short(model)}__{cfg['tag']}",
                "group": "comparison",
                "model": model,
                "loss": cfg["loss"],
                "quantizer_type": cfg["quantizer_type"],
                "butterfly": False,
                "eval_datasets": eval_datasets,
            }
            experiments.append(exp)

    return experiments


def build_ablation_experiments(
    models: list[str],
    eval_datasets: list[str],
) -> list[dict]:
    """Butterfly ablation: with Butterfly vs without (Hadamard).

    Uses swd_unif + int4 as the base configuration (strongest INT4 combo).
    """
    experiments = []

    for model in models:
        for butterfly in [False, True]:
            tag = "butterfly" if butterfly else "hadamard"
            exp = {
                "name": f"ablation__{_model_short(model)}__{tag}",
                "group": "ablation",
                "model": model,
                "loss": "swd_unif",
                "quantizer_type": "int4",
                "butterfly": butterfly,
                "eval_datasets": eval_datasets,
            }
            experiments.append(exp)

    return experiments


def _model_short(model_name: str) -> str:
    """meta-llama/Llama-3.2-1B → Llama-3.2-1B"""
    return model_name.split("/")[-1]


# ============================================================================
# Runner
# ============================================================================

def build_command(exp: dict, output_root: str, extra_args: list[str],
                  lm_eval: bool = False) -> list[str]:
    """Build the CLI command for a single experiment."""
    output_dir = os.path.join(output_root, exp["name"])
    cmd = [
        sys.executable,
        "dartquant_v2/run_quantize.py",
        "--model", exp["model"],
        "--loss", exp["loss"],
        "--quantizer_type", exp["quantizer_type"],
        "--output_dir", output_dir,
        "--ppl_eval_dataset", *exp["eval_datasets"],
    ]

    if exp["butterfly"]:
        cmd.append("--butterfly")

    # w4a4kv4 defaults
    cmd.extend(["--w_bits", "4", "--a_bits", "4", "--k_bits", "4", "--v_bits", "4"])

    if lm_eval:
        cmd.append("--lm_eval")

    cmd.extend(extra_args)

    return cmd


def run_experiment(cmd: list[str], exp: dict, dry_run: bool = False) -> dict:
    """Execute a single experiment and return the result summary."""
    log.info("-" * 70)
    log.info(f"Experiment:  {exp['name']}")
    log.info(f"  Model:     {exp['model']}")
    log.info(f"  Loss:      {exp['loss']}")
    log.info(f"  Quantizer: {exp['quantizer_type']}")
    log.info(f"  Butterfly: {exp['butterfly']}")
    log.info(f"  Command:   {' '.join(cmd)}")
    log.info("-" * 70)

    if dry_run:
        log.info("  [DRY RUN] Skipping execution.")
        return {"name": exp["name"], "status": "dry_run", "ppl": {}, "lm_eval": {}}

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per experiment
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            log.error(f"  FAILED (exit code {result.returncode})")
            log.error(f"  stderr: {result.stderr[-500:]}")
            return {
                "name": exp["name"],
                "status": "failed",
                "returncode": result.returncode,
                "stderr": result.stderr[-1000:],
                "elapsed_s": elapsed,
                "ppl": {},
                "lm_eval": {},
            }

        # Parse PPL results from the output directory
        output_dir = None
        for i, arg in enumerate(cmd):
            if arg == "--output_dir" and i + 1 < len(cmd):
                output_dir = cmd[i + 1]
                break

        ppl_results = {}
        lm_eval_results = {}
        if output_dir:
            results_file = os.path.join(output_dir, "results.txt")
            if os.path.isfile(results_file):
                ppl_results, lm_eval_results = _parse_results_file(results_file)

        log.info(f"  SUCCESS ({elapsed:.0f}s)")
        for ds, ppl in ppl_results.items():
            log.info(f"    {ds} PPL: {ppl:.2f}")
        for task, acc in lm_eval_results.items():
            log.info(f"    {task}: {acc:.2f}%")

        return {
            "name": exp["name"],
            "status": "success",
            "elapsed_s": elapsed,
            "ppl": ppl_results,
            "lm_eval": lm_eval_results,
        }

    except subprocess.TimeoutExpired:
        log.error("  TIMEOUT (>2h)")
        return {"name": exp["name"], "status": "timeout", "ppl": {}, "lm_eval": {}}
    except Exception as e:
        log.error(f"  ERROR: {e}")
        return {"name": exp["name"], "status": "error", "error": str(e), "ppl": {}, "lm_eval": {}}


def _parse_results_file(path: str) -> tuple[dict, dict]:
    """Parse the results.txt file written by pipeline.py.

    Returns (ppl_dict, lm_eval_dict).
    """
    ppl = {}
    lm_eval = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("lm_eval_"):
                parts = line.split(":")
                task = parts[0].replace("lm_eval_", "").strip()
                try:
                    lm_eval[task] = float(parts[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "_ppl:" in line:
                parts = line.split(":")
                ds_name = parts[0].replace("_ppl", "").strip()
                try:
                    ppl[ds_name] = float(parts[1].strip())
                except (ValueError, IndexError):
                    pass
    return ppl, lm_eval


# ============================================================================
# Summary report
# ============================================================================

def print_summary(all_results: list[dict], experiments: list[dict]):
    """Print a final summary table."""
    print("\n")
    print("=" * 100)
    print("  EXPERIMENT SUMMARY")
    print("=" * 100)

    # Build lookup from name to experiment config
    exp_map = {e["name"]: e for e in experiments}

    # Check if any results have lm_eval data
    has_lm_eval = any(res.get("lm_eval") for res in all_results)

    # Group by experiment group
    groups = {}
    for res in all_results:
        exp = exp_map.get(res["name"], {})
        grp = exp.get("group", "unknown")
        groups.setdefault(grp, []).append((exp, res))

    for group_name, items in groups.items():
        print(f"\n  [{group_name.upper()}]")
        header = f"  {'Experiment':<45} {'Status':<10} {'wikitext2':>10} {'ptb':>10} {'c4':>10}"
        if has_lm_eval:
            header += f" {'acc_avg':>10}"
        header += f" {'Time':>8}"
        print(header)
        sep = f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10}"
        if has_lm_eval:
            sep += f" {'-'*10}"
        sep += f" {'-'*8}"
        print(sep)

        for exp, res in items:
            status = res.get("status", "?")
            ppl = res.get("ppl", {})
            lm = res.get("lm_eval", {})
            w2 = f"{ppl['wikitext2']:.2f}" if "wikitext2" in ppl else "-"
            pt = f"{ppl['ptb']:.2f}" if "ptb" in ppl else "-"
            c4 = f"{ppl['c4']:.2f}" if "c4" in ppl else "-"
            elapsed = res.get("elapsed_s", 0)
            time_str = f"{elapsed:.0f}s" if elapsed > 0 else "-"

            label = (
                f"{_model_short(exp.get('model', '?'))}"
                f" | {exp.get('loss', '?')}"
                f" | {exp.get('quantizer_type', '?')}"
            )
            if exp.get("butterfly"):
                label += " | BF"
            else:
                label += " | Had"

            row = f"  {label:<45} {status:<10} {w2:>10} {pt:>10} {c4:>10}"
            if has_lm_eval:
                avg = f"{lm['acc_avg']:.2f}" if "acc_avg" in lm else "-"
                row += f" {avg:>10}"
            row += f" {time_str:>8}"
            print(row)

    print("\n" + "=" * 100)


# ============================================================================
# Entry point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all DartQuant v2 comparison and ablation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--group", type=str, default="all",
        choices=["all", "comparison", "ablation"],
        help="Which experiment group to run.",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Override model list (skip auto-detection). "
             "E.g.: --models meta-llama/Llama-3.2-1B",
    )
    parser.add_argument(
        "--output_root", type=str, default="./experiment_results",
        help="Root directory for all experiment outputs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128,
        help="Number of calibration samples.",
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048,
        help="Sequence length.",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="HuggingFace cache directory (auto-detected if not set).",
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace access token.",
    )
    parser.add_argument(
        "--lm_eval", action="store_true", default=False,
        help="Also run lm_eval zero-shot benchmarks (MMLU, ARC, etc.).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir or _HF_HOME

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(args.output_root, timestamp)

    print("=" * 70)
    print("DartQuant v2: Unified Experiment Runner")
    print("=" * 70)
    print(f"  Cache dir:    {cache_dir}")
    print(f"  Output root:  {output_root}")
    print(f"  Group:        {args.group}")
    print(f"  Dry run:      {args.dry_run}")
    print()

    # ---- Auto-detect models ----
    if args.models:
        models = args.models
        log.info(f"Using specified models: {models}")
    else:
        log.info(f"Scanning for cached models in {cache_dir} ...")
        models = detect_cached_models(cache_dir)
        if not models:
            log.warning(
                "No cached models found! Please download models first via "
                "scripts/stat_and_download.py, or specify --models explicitly."
            )
            sys.exit(1)
        log.info(f"Found {len(models)} cached model(s):")
        for m in models:
            log.info(f"  - {m}")

    # ---- Auto-detect datasets ----
    log.info(f"\nScanning for cached datasets in {_DATASETS_CACHE} ...")
    eval_datasets = detect_cached_datasets(_DATASETS_CACHE)
    if not eval_datasets:
        log.warning("No cached evaluation datasets found, defaulting to wikitext2.")
        eval_datasets = ["wikitext2"]
    else:
        log.info(f"Found cached datasets: {eval_datasets}")

    # ---- Build experiment list ----
    experiments = []
    if args.group in ("all", "comparison"):
        experiments.extend(build_comparison_experiments(models, eval_datasets))
    if args.group in ("all", "ablation"):
        experiments.extend(build_ablation_experiments(models, eval_datasets))

    log.info(f"\nTotal experiments to run: {len(experiments)}")
    for i, exp in enumerate(experiments, 1):
        log.info(f"  [{i:2d}] {exp['name']}")

    # ---- Extra args passed to run_quantize.py ----
    extra_args = [
        "--nsamples", str(args.nsamples),
        "--seqlen", str(args.seqlen),
    ]
    if args.cache_dir:
        extra_args.extend(["--cache_dir", args.cache_dir])
    if args.hf_token:
        extra_args.extend(["--hf_token", args.hf_token])

    # ---- Run experiments sequentially ----
    Path(output_root).mkdir(parents=True, exist_ok=True)
    all_results = []

    for i, exp in enumerate(experiments, 1):
        log.info(f"\n[{i}/{len(experiments)}] Starting: {exp['name']}")
        cmd = build_command(exp, output_root, extra_args, lm_eval=args.lm_eval)
        result = run_experiment(cmd, exp, dry_run=args.dry_run)
        all_results.append(result)

    # ---- Summary ----
    print_summary(all_results, experiments)

    # Save full results JSON
    results_json_path = os.path.join(output_root, "all_results.json")
    Path(output_root).mkdir(parents=True, exist_ok=True)
    with open(results_json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "models": models,
                "eval_datasets": eval_datasets,
                "experiments": [
                    {**exp_map, **res}
                    for exp_map, res in zip(experiments, all_results)
                ],
            },
            f,
            indent=2,
            default=str,
        )
    log.info(f"\nFull results saved to: {results_json_path}")


if __name__ == "__main__":
    main()
