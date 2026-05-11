# pyre-ignore-all-errors
"""
vision_encoder.py — Dual Encoder: DenseNet121 (CheXNet-style) + BiomedCLIP
===========================================================================
Phase 2 Upgrade: Combines two complementary encoders into a 1536-dim joint embedding.

  [0:1024]    → DenseNet121 (ImageNet pre-trained CNN — local pathology features)
  [1024:1536] → BiomedCLIP  (Medical ViT — global semantic context)

The combined, L2-normalized vector is stored in FAISS and used by the cross-encoder reranker.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import open_clip
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import torchvision.models as tv_models
import torchvision.transforms as tv_transforms

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config

logger = get_logger(__name__, log_file="logs/vision_encoder.log")


# ─── BiomedCLIP ──────────────────────────────────────────────────────────────

def load_biomed_clip(model_name: str, device: torch.device):
    """Loads the BiomedCLIP Vision-Language model (ViT backbone)."""
    logger.info(f"Loading BiomedCLIP: {model_name}")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    model = model.to(device).eval()
    logger.info("BiomedCLIP loaded!")
    return model, preprocess


# ─── DenseNet121 (CheXNet-style backbone) ────────────────────────────────────

def load_densenet121(device: torch.device):
    """
    Loads ImageNet pre-trained DenseNet121 with the classifier head removed.
    Returns a 1024-dim CNN feature extractor (CheXNet-style backbone).
    """
    logger.info("Loading DenseNet121 (CheXNet-style backbone)...")
    model = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Identity()   # Strip head → raw 1024-dim pool output
    model = model.to(device).eval()
    logger.info("DenseNet121 loaded!")
    return model


def get_densenet_transform():
    """Standard ImageNet normalization pipeline for DenseNet121."""
    return tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
        tv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])


# ─── Encoding ────────────────────────────────────────────────────────────────

def encode_combined_image(
    image_path: str,
    biomed_model, biomed_preprocess,
    densenet_model, densenet_transform,
    device: torch.device
) -> np.ndarray:
    """
    Dual-Encoder: runs image through both BiomedCLIP and DenseNet121.
    Returns a 1536-dim L2-normalized vector: [DenseNet(1024) | BiomedCLIP(512)].
    """
    try:
        image = Image.open(image_path).convert("RGB")

        # BiomedCLIP path — 512-dim
        biomed_input = biomed_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            bf = biomed_model.encode_image(biomed_input)
        bf = bf / bf.norm(p=2, dim=-1, keepdim=True)
        biomed_vec = bf.squeeze().cpu().numpy().astype(np.float32)   # (512,)

        # DenseNet121 path — 1024-dim
        dn_input = densenet_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            df = densenet_model(dn_input)
        df = df / df.norm(p=2, dim=-1, keepdim=True)
        densenet_vec = df.squeeze().cpu().numpy().astype(np.float32)  # (1024,)

        # Concatenate → final L2 normalization
        combined = np.concatenate([densenet_vec, biomed_vec])         # (1536,)
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined

    except Exception as e:
        logger.error(f"Failed to encode {image_path}: {str(e)}")
        return None


def encode_medical_image(image_path: str, model, preprocess, device: torch.device) -> np.ndarray:
    """Legacy BiomedCLIP-only encoder (512-dim). Kept for backward compatibility."""
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(input_tensor)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {str(e)}")
        return None


# ─── Master Pipeline ──────────────────────────────────────────────────────────

def build_biomed_embeddings(cfg: dict):
    """Dual-Encoder pipeline: processes the full dataset → saves 1536-dim embeddings."""
    processed_dir = Path(cfg["paths"]["processed_data"])
    embeddings_out = cfg["paths"]["embeddings"]
    paths_out      = cfg["paths"]["image_paths"]

    model_name = cfg["encoder"]["model_name"]
    device = torch.device(
        cfg["encoder"]["device"] if cfg["encoder"]["device"] != "auto"
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger.info(f"Dual Encoder firing on device: {device}")

    biomed_model, biomed_preprocess = load_biomed_clip(model_name, device)
    densenet_model                  = load_densenet121(device)
    densenet_transform              = get_densenet_transform()

    image_paths = sorted(
        list(processed_dir.rglob("*.png")) + list(processed_dir.rglob("*.jpg"))
    )
    if not image_paths:
        logger.error(f"No processed images found in {processed_dir}!")
        return

    logger.info(f"Found {len(image_paths)} X-Rays to encode.")

    import pandas as pd, json

    csv_path = Path("data/raw/images/Data_Entry_2017.csv/Data_Entry_2017.csv")
    if not csv_path.exists():
        logger.error(f"FATAL: Ground Truth CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    disease_map = dict(zip(df["Image Index"], df["Finding Labels"]))

    embeddings, metadata = [], []

    for img_path in tqdm(image_paths, desc="Dual Encoder (DenseNet121 + BiomedCLIP)"):
        vector = encode_combined_image(
            str(img_path),
            biomed_model, biomed_preprocess,
            densenet_model, densenet_transform,
            device
        )
        if vector is not None:
            actual_diagnosis = disease_map.get(img_path.name, "No Finding")
            embeddings.append(vector)
            metadata.append({
                "file_path": str(img_path.relative_to(processed_dir.parent.parent)),
                "actual_diagnosis": actual_diagnosis
            })

    embeddings_np = np.vstack(embeddings)
    os.makedirs(os.path.dirname(embeddings_out), exist_ok=True)
    np.save(embeddings_out, embeddings_np)

    metadata_out = paths_out.replace(".txt", ".json")
    with open(metadata_out, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Done! {len(embeddings_np)} vectors ({embeddings_np.shape[1]}-dim) saved.")


if __name__ == "__main__":
    config_path = str(Path(__file__).resolve().parent.parent / "config.yaml")
    config = load_config(config_path)
    build_biomed_embeddings(config)
