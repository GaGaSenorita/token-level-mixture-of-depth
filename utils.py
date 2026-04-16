"""
Utility helpers for random seeds, device selection, logging, and more.
"""
import random
import numpy as np
import torch
import json
import os
import logging
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # PyTorch RNG on CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # PyTorch RNG on a single GPU
        torch.cuda.manual_seed_all(seed) # PyTorch RNG across multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str | Path) -> Path:
    """Ensure the directory exists, creating it if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_results(results: dict, output_dir: str | Path, filename: str = "results.json"):
    out_dir = ensure_dir(output_dir)
    out_file = out_dir / filename

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[OK] Results saved to {out_file}")

def setup_logger(log_dir: str | Path, name="run", level=logging.INFO): # Save logs to log_dir/name.log and record INFO or higher
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Write logs to both a file and stdout using the same format
    fh = logging.FileHandler(log_dir / f"{name}.log")
    sh = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
