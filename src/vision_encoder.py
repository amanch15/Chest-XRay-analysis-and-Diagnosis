# pyre-ignore-all-errors
"""
vision_encoder.py — CLIP Vision Feature Extractor
===================================================
Week 2 Core Module

Steps:
  1. Load pre-trained CLIP ViT-B/32 from HuggingFace
  2. Encode each processed image into a 512-dim L2-normalized vector
  3. Save embeddings.npy + image_paths.txt for FAISS (Week 3)
"""

import os 
import numpy as np
import torch
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config

logger = get_logger(__name__, log_file="logs/vision_encoder.log")


# ─── Device Selection 

def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(preference)
    logger.info(f"Using device: {device}")
    return device


# ─── Model Loader ──────────────────────────────────────────────────────────

def load_clip_model(model_name: str = "openai/clip-vit-base-patch32",
                    device: torch.device = None):
    if device is None:
        device = get_device()
    logger.info(f"Loading CLIP model: {model_name}")
    processor = CLIPProcessor.from_pretrained(model_name)
    model     = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    logger.info("CLIP model loaded ✅")
    return model, processor


# ─── Single Image Embedding ────────────────────────────────────────────────

def encode_image(image_path: str, model, processor,
                 device: torch.device) -> np.ndarray:
    """Returns a 512-dim L2-normalized CLIP embedding for one image."""
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # Handle variations in HuggingFace transformer versions
        if hasattr(features, "image_embeds"):
            features = features.image_embeds
        elif not isinstance(features, torch.Tensor) and hasattr(features, "pooler_output"):
            features = features.pooler_output
            
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy().astype(np.float32)


# ─── Batch Embedding Pipeline ──────────────────────────────────────────────

def build_embedding_index(cfg: dict):
    """
    Encode all images in data/processed/ and save:
      - models/embeddings.npy     shape (N, 512)
      - models/image_paths.txt    one filename per line
    """
    processed_dir = Path(cfg["paths"]["processed_data"])
    models_dir    = Path(cfg["paths"]["checkpoints"]).parent   # models/
    model_name    = cfg["encoder"]["model_name"]
    device_pref   = cfg["encoder"]["device"]

    models_dir.mkdir(parents=True, exist_ok=True)

    device           = get_device(device_pref)
    model, processor = load_clip_model(model_name, device)

    image_files = sorted(
        list(processed_dir.glob("*.png")) +
        list(processed_dir.glob("*.jpg"))
    )
    logger.info(f"Found {len(image_files)} processed images to encode")

    if not image_files:
        logger.error(f"No images in {processed_dir}. Run data_loader.py first!")
        return

    embeddings  = []
    valid_paths = []
    skipped     = 0

    for img_path in tqdm(image_files, desc="Encoding with CLIP"):
        try:
            emb = encode_image(str(img_path), model, processor, device)
            embeddings.append(emb)
            valid_paths.append(img_path.name)
        except Exception as e:
            logger.warning(f"Skipping {img_path.name}: {e}")
            skipped += 1

    emb_array = np.vstack(embeddings).astype(np.float32)
    np.save(str(models_dir / "embeddings.npy"), emb_array)

    with open(str(models_dir / "image_paths.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(valid_paths))

    logger.info(f"Done! Saved {len(embeddings)} embeddings | Shape: {emb_array.shape}")
    logger.info(f"   embeddings.npy  -> {models_dir / 'embeddings.npy'}")
    logger.info(f"   image_paths.txt -> {models_dir / 'image_paths.txt'}")
    logger.info(f"   Skipped: {skipped}")


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    build_embedding_index(cfg)
