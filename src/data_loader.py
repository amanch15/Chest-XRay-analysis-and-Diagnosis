# pyre-ignore-all-errors
"""
data_loader.py — NIH ChestX-ray Image Preprocessing Pipeline
=============================================================
Week 1 Core Module

Steps:
  1. Scan data/raw/ recursively for all .png / .jpg images
  2. Apply CLAHE contrast enhancement (L-channel in LAB space)
  3. Resize to 224x224
  4. Save to data/processed/
"""

import os
import cv2
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config

logger = get_logger(__name__, log_file="logs/data_loader.log")


# ─── CLAHE Enhancement
def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 2.0,
                grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE on the L-channel of LAB color space."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l_channel)
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def preprocess_image(img_path: str, target_size: int = 224,
                     clip_limit: float = 2.0,
                     grid_size: tuple = (8, 8)) -> np.ndarray:
    """Load → CLAHE → Resize. Returns None on failure."""
    img = cv2.imread(img_path)
    if img is None:
        logger.warning(f"Could not load: {img_path}")
        return None
    enhanced = apply_clahe(img, clip_limit, grid_size)
    return cv2.resize(enhanced, (target_size, target_size),
                      interpolation=cv2.INTER_AREA)


# ─── Main Pipeline 

def run_pipeline(cfg: dict):
    raw_dir       = Path(cfg["paths"]["raw_data"])
    processed_dir = Path(cfg["paths"]["processed_data"])
    img_size      = cfg["preprocessing"]["image_size"]
    clip_limit    = cfg["preprocessing"]["clahe_clip"]
    grid_size     = tuple(cfg["preprocessing"]["clahe_grid"])
    max_samples: int = int(cfg["preprocessing"]["max_samples"])

    processed_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images recursively
    all_images = sorted(
        list(raw_dir.rglob("*.png")) +
        list(raw_dir.rglob("*.jpg"))
    )

    if not all_images:
        logger.error(f"No images found in {raw_dir}. Check your data/raw/ folder.")
        return

    logger.info(f"Found {len(all_images)} images in {raw_dir}")

    if max_samples != -1:
        all_images = all_images[0:max_samples]
        logger.info(f"Limiting to {max_samples} samples as per config")

    processed = 0
    skipped = 0

    for img_path in tqdm(all_images, desc="Applying CLAHE + Resize"):
        out_path = processed_dir / img_path.name
        if out_path.exists():         
            processed = processed + 1
            continue

        result = preprocess_image(str(img_path), img_size, clip_limit, grid_size)
        if result is None:
            skipped = skipped + 1
            continue

        cv2.imwrite(str(out_path), result)
        processed = processed + 1

    logger.info(f"Done! Processed: {processed} | Skipped: {skipped}")
    logger.info(f"Output: {processed_dir}")


if __name__ == "__main__":
    config_path = str(Path(__file__).resolve().parent.parent / "config.yaml")
    cfg = load_config(config_path)
    run_pipeline(cfg)
