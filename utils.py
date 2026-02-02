"""
工具函数: 随机种子、设备获取、日志等
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
    torch.manual_seed(seed) # Pytorch在CPU上的随机性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # Pytorch在单个GPU上的随机性
        torch.cuda.manual_seed_all(seed) # Pytorch在多个GPU上的随机性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_dir(path: str | Path) -> Path:
    '''确保目录存在，如果不存在则创建'''
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_results(results: dict, output_dir: str | Path, filename: str = "results.json"):
    out_dir = ensure_dir(output_dir)
    out_file = out_dir / filename

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"[OK] Results saved to {out_file}")

def setup_logger(log_dir: str | Path, name="run", level=logging.INFO): # 日志存到log_dir/name.log, 记录INFO及以上级别
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # 日志同时“写进文件”和“输出到控制台”，而且格式相同
    fh = logging.FileHandler(log_dir / f"{name}.log")
    sh = logging.StreamHandler()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger