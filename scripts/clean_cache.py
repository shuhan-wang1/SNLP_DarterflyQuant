#!/usr/bin/env python3
"""
Clean Python cache files to ensure fresh imports.
Run this if you encounter ImportError issues.
"""

import os
import shutil
from pathlib import Path


def clean_pycache(root_dir="."):
    """Remove all __pycache__ directories and .pyc files"""
    root_path = Path(root_dir)
    removed_count = 0
    
    # Remove __pycache__ directories
    for pycache_dir in root_path.rglob("__pycache__"):
        if pycache_dir.is_dir():
            print(f"Removing: {pycache_dir}")
            shutil.rmtree(pycache_dir)
            removed_count += 1
    
    # Remove .pyc files
    for pyc_file in root_path.rglob("*.pyc"):
        if pyc_file.is_file():
            print(f"Removing: {pyc_file}")
            pyc_file.unlink()
            removed_count += 1
    
    # Remove .pyo files (optimized bytecode)
    for pyo_file in root_path.rglob("*.pyo"):
        if pyo_file.is_file():
            print(f"Removing: {pyo_file}")
            pyo_file.unlink()
            removed_count += 1
    
    print(f"\nCleaned {removed_count} cache files/directories")
    print("You can now run your script again.")


if __name__ == "__main__":
    print("Cleaning Python cache files...")
    clean_pycache()
