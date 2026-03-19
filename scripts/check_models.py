#!/usr/bin/env python3
"""Check which models and datasets are cached locally."""

import os
import sys
from pathlib import Path

_DEFAULT_HF_HOME = "/root/autodl-tmp/huggingface"
HF_HOME = os.environ.get("HF_HOME", _DEFAULT_HF_HOME)
DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", "/root/autodl-tmp/datasets")

EXPECTED_MODELS = [
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def scan_model_dirs(cache_dir):
    """List all models--* directories and what they map to."""
    results = []
    search_roots = [Path(cache_dir) / "hub", Path(cache_dir)]

    for root in search_roots:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            raw = entry.name[len("models--"):]
            parts = raw.split("--", 1)
            model_id = f"{parts[0]}/{parts[1]}" if len(parts) == 2 else parts[0]

            # Check snapshots
            snapshots_dir = entry / "snapshots"
            has_snapshots = (snapshots_dir.is_dir()
                            and any(snapshots_dir.iterdir()))

            # Check size
            total_bytes = 0
            n_files = 0
            for f in entry.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
                    n_files += 1

            results.append({
                "dir_name": entry.name,
                "root": str(root),
                "model_id": model_id,
                "has_snapshots": has_snapshots,
                "n_files": n_files,
                "size_gb": total_bytes / (1024**3),
            })
    return results


def main():
    print("=" * 70)
    print("Model Cache Diagnostic")
    print("=" * 70)
    print(f"  HF_HOME:        {HF_HOME}")
    print(f"  DATASETS_CACHE: {DATASETS_CACHE}")
    print()

    # 1) Raw directory listing
    print("--- Raw models--* directories found ---")
    models_found = scan_model_dirs(HF_HOME)
    if not models_found:
        print("  (none)")
    for m in models_found:
        status = "OK" if m["has_snapshots"] else "NO SNAPSHOTS"
        print(f"  [{status}] {m['dir_name']}")
        print(f"         → model_id: {m['model_id']}")
        print(f"         → root:     {m['root']}")
        print(f"         → files:    {m['n_files']}, size: {m['size_gb']:.2f} GB")

    # 2) Check expected models
    detected_ids = {m["model_id"] for m in models_found if m["has_snapshots"]}
    print()
    print("--- Expected models status ---")
    for expected in EXPECTED_MODELS:
        if expected in detected_ids:
            info = next(m for m in models_found
                        if m["model_id"] == expected and m["has_snapshots"])
            print(f"  [OK]      {expected}  ({info['size_gb']:.2f} GB)")
        else:
            # Check if directory exists but no snapshots
            partial = [m for m in models_found if m["model_id"] == expected]
            if partial:
                print(f"  [PARTIAL] {expected}  (dir exists but no snapshots)")
            else:
                print(f"  [MISSING] {expected}")

    # 3) Extra models (not in expected list)
    extra = detected_ids - set(EXPECTED_MODELS)
    if extra:
        print()
        print("--- Extra models (not in expected list) ---")
        for e in sorted(extra):
            info = next(m for m in models_found
                        if m["model_id"] == e and m["has_snapshots"])
            print(f"  {e}  ({info['size_gb']:.2f} GB)")

    # 4) Dataset check
    print()
    print("--- Datasets cache ---")
    ds_root = Path(DATASETS_CACHE)
    if ds_root.exists():
        for entry in sorted(ds_root.iterdir()):
            if entry.is_dir():
                print(f"  {entry.name}")
    else:
        print(f"  Directory not found: {DATASETS_CACHE}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
