# pyre-ignore-all-errors
"""
utils.py — Logging & Config Utilities
======================================
Shared helpers used across all pipeline modules.
"""

import logging
import os
import yaml
from pathlib import Path

def get_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers: return logger
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

def load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        fallback = Path(__file__).resolve().parent.parent / "config.yaml"
        if fallback.exists():
            p = fallback
        else:
            raise FileNotFoundError(f"Config file not found. Looked at {p} and {fallback}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
