#!/usr/bin/env python3
"""
Run all instruct-model experiments.

Experiments:
  Condition 1 — W4A4KV4:    whip, swd_unif, swd_gauss  (× each instruct model)
  Condition 2 — W4A16KV16:  whip, swd_unif, swd_gauss  (× each instruct model)

Also runs FP16 baselines for each instruct model (once, shared across conditions).

Usage:
  python scripts/run_instruct_experiments.py
  python scripts/run_instruct_experiments.py --dry-run
  python scripts/run_instruct_experiments.py --models meta-llama/Llama-3.2-1B-Instruct
  python scripts/run_instruct_experiments.py --resume
  python scripts/run_instruct_experiments.py --no-baseline
"""

import os
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)

# Re-use run_all_experiments as a library
sys.path.insert(0, _SCRIPTS_DIR)
from run_all_experiments import (
    build_comparison_experiments,
    build_baseline_experiments,
    build_command,
    run_experiment,
    print_summary,
    detect_cached_models,
    detect_cached_datasets,
    detect_cached_benchmarks,
    find_latest_run_dir,
    detect_completed_experiments,
    _parse_results_file,
    _model_short,
    _HF_HOME,
    _DATASETS_CACHE,
    log,
)

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def _is_instruct_model(name: str) -> bool:
    lower = name.lower()
    return 'instruct' in lower or 'chat' in lower


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all instruct-model experiments (W4A4KV4 + W4A16KV16).",
    )
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Override instruct model list.")
    parser.add_argument("--output_root", type=str,
                        default="./experiment_results_instruct",
                        help="Root directory for outputs.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--lm_eval", action="store_true", default=False)
    parser.add_argument("--no-baseline", dest="no_baseline",
                        action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    cache_dir = args.cache_dir or _HF_HOME

    # ---- Resolve models (auto-detect all instruct models from cache) ----
    if args.models:
        models = args.models
    else:
        cached = detect_cached_models(cache_dir)
        log.info("All cached models: %s", cached)
        models = [m for m in cached if _is_instruct_model(m)]
        if not models:
            log.warning(
                "No instruct models found in cache. "
                "Available: %s", cached,
            )
            sys.exit(1)
    log.info("Instruct models (%d): %s", len(models), models)

    # ---- Resolve datasets ----
    eval_datasets = detect_cached_datasets(_DATASETS_CACHE) or ["wikitext2"]
    log.info("Eval datasets: %s", eval_datasets)

    # ---- Auto-detect lm_eval benchmarks ----
    if not args.lm_eval:
        cached_benchmarks = detect_cached_benchmarks(_DATASETS_CACHE)
        if cached_benchmarks:
            log.info("Found lm_eval benchmarks: %s → auto-enabling", cached_benchmarks)
            args.lm_eval = True

    # ---- Resolve output dir ----
    if args.resume:
        latest = find_latest_run_dir(args.output_root)
        if latest:
            output_root = latest
            resumed = True
        else:
            log.info("No previous run found — starting fresh.")
            output_root = os.path.join(
                args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
            resumed = False
    else:
        output_root = os.path.join(
            args.output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
        resumed = False

    # ---- Build experiment list ----
    # Condition 1: W4A4KV4
    w4a4kv4 = build_comparison_experiments(
        models, eval_datasets, w4_only=False)
    # Condition 2: W4A16KV16
    w4a16kv16 = build_comparison_experiments(
        models, eval_datasets, w4_only=True)
    # Baselines
    baselines = [] if args.no_baseline else build_baseline_experiments(
        models, eval_datasets)

    experiments = baselines + w4a4kv4 + w4a16kv16

    # ---- Skip completed when resuming ----
    if resumed:
        completed = detect_completed_experiments(output_root, experiments)
        if completed:
            log.info("Skipping %d completed experiment(s):", len(completed))
            for n in sorted(completed):
                log.info("  [DONE] %s", n)
            experiments = [e for e in experiments if e["name"] not in completed]

    # ---- Summary ----
    print("=" * 70)
    print("Instruct-Model Experiment Runner")
    print("=" * 70)
    print(f"  Models:      {[_model_short(m) for m in models]}")
    print(f"  Conditions:  W4A4KV4 ({len(w4a4kv4)} exp) + "
          f"W4A16KV16 ({len(w4a16kv16)} exp)")
    print(f"  Baselines:   {len(baselines)}")
    print(f"  Total:       {len(experiments)} experiment(s) to run")
    print(f"  Output:      {output_root}")
    print(f"  lm_eval:     {args.lm_eval}")
    print("=" * 70)

    for i, exp in enumerate(experiments, 1):
        if exp.get("baseline"):
            print(f"  [{i:2d}] {_model_short(exp['model'])} | FP16 baseline")
        else:
            a = exp.get('a_bits', 4)
            tag = f"W{exp.get('w_bits', 4)}A{a}KV{exp.get('k_bits', 4)}"
            print(f"  [{i:2d}] {_model_short(exp['model'])} | "
                  f"{exp['loss']} | {exp['quantizer_type']} | {tag}")
    print()

    # ---- Extra CLI args ----
    extra_args = ["--nsamples", str(args.nsamples), "--seqlen", str(args.seqlen)]
    if args.cache_dir:
        extra_args.extend(["--cache_dir", args.cache_dir])
    if args.hf_token:
        extra_args.extend(["--hf_token", args.hf_token])

    # ---- Run ----
    Path(output_root).mkdir(parents=True, exist_ok=True)
    all_results = []
    total = len(experiments)

    exp_bar = tqdm(
        experiments, desc="Overall progress", unit="exp",
        bar_format=("\n{l_bar}{bar}| {n_fmt}/{total_fmt} experiments "
                    "[{elapsed}<{remaining}] {postfix}"),
        position=0, leave=True, ncols=90,
    )

    for i, exp in enumerate(exp_bar, 1):
        exp_bar.set_postfix_str(
            f"{_model_short(exp['model'])} | {exp['loss']}")
        cmd = build_command(exp, output_root, extra_args,
                            lm_eval=args.lm_eval)
        result = run_experiment(cmd, exp, dry_run=args.dry_run,
                                exp_idx=i, total_exps=total)
        all_results.append(result)

        ok = sum(1 for r in all_results if r["status"] == "success")
        fail = sum(1 for r in all_results if r["status"] == "failed")
        exp_bar.set_postfix_str(f"done={ok} fail={fail}")

    exp_bar.close()

    # ---- Merge previous results when resuming ----
    if resumed:
        full_experiments = (
            ([] if args.no_baseline
             else build_baseline_experiments(models, eval_datasets))
            + build_comparison_experiments(models, eval_datasets, w4_only=False)
            + build_comparison_experiments(models, eval_datasets, w4_only=True)
        )
        done_names = detect_completed_experiments(output_root, full_experiments)
        prev = []
        for exp in full_experiments:
            if (exp["name"] in done_names
                    and exp["name"] not in {r["name"] for r in all_results}):
                rf = os.path.join(output_root, exp["name"], "results.txt")
                ppl, lm = (_parse_results_file(rf)
                           if os.path.isfile(rf) else ({}, {}))
                prev.append({"name": exp["name"], "status": "success (prev)",
                             "ppl": ppl, "lm_eval": lm})
        all_results = prev + all_results
        experiments = full_experiments

    # ---- Summary ----
    print_summary(all_results, experiments)

    # ---- Save JSON ----
    results_json = os.path.join(output_root, "all_results.json")
    exp_map = {e["name"]: e for e in experiments}
    with open(results_json, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "eval_datasets": eval_datasets,
            "experiments": [
                {**exp_map.get(r["name"], {}), **r} for r in all_results
            ],
        }, f, indent=2, default=str)
    log.info("Results saved to %s", results_json)


if __name__ == "__main__":
    main()
