# pyre-ignore-all-errors
"""
vision_encoder.py — BiomedCLIP Feature Extractor
=================================================
Phase 1 Upgrade: Swapped generic OpenAI CLIP for Microsoft's BiomedCLIP!
This processes images via `open_clip` (because it's natively supported there)
and pumps out highly accurate medical vectors.
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


load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config

logger = get_logger(__name__, log_file="logs/vision_encoder.log")

def load_biomed_clip(model_name: str, device: torch.device):
    """Loads the specialized BiomedCLIP model built specifically for medical research."""
    logger.info(f"Loading Medical Foundation Model: {model_name}")
    # BiomedCLIP 
    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    model.eval()
    logger.info("BiomedCLIP successfully locked and loaded into memory!")
    return model, preprocess

def encode_medical_image(image_path: str, model, preprocess, device: torch.device) -> np.ndarray:
    """Returns a highly nuanced L2-normalized geometric vector from the medical image."""
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Extract the raw visual features
            features = model.encode_image(input_tensor)
            
        # L2-Normalization (Forces vector length to exactly 1 so Inner Product = Cosine Similarity)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze().cpu().numpy().astype(np.float32)
        
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {str(e)}")
        return None

def build_biomed_embeddings(cfg: dict):
    """The master pipeline to digest the dataset."""
    processed_dir = Path(cfg["paths"]["processed_data"])
    embeddings_out = cfg["paths"]["embeddings"]
    paths_out      = cfg["paths"]["image_paths"]
    
    model_name = cfg["encoder"]["model_name"]
    device = torch.device(cfg["encoder"]["device"] if cfg["encoder"]["device"] != "auto" 
                          else "cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Firing up calculations on device: {device}")
    
    # Load Model
    model, preprocess = load_biomed_clip(model_name, device)
    
    # Collect Processed Images
    image_paths = sorted(list(processed_dir.rglob("*.png")) + list(processed_dir.rglob("*.jpg")))
    if not image_paths:
        logger.error(f"No processed images found in {processed_dir}!")
        return

    logger.info(f"Found {len(image_paths)} heavily processed X-Rays to analyze.")
    
    import pandas as pd
    import json
    
    csv_path = Path("data/raw/images/Data_Entry_2017.csv/Data_Entry_2017.csv")
    if not csv_path.exists():
        logger.error(f"FATAL: Ground Truth CSV not found at {csv_path}")
        return
        
    logger.info("Parsing Official NIH Databank Labels...")
    df = pd.read_csv(csv_path)
    # Create a giant, ultra-fast dictionary of filename -> Disease Label (e.g., '00006585_007.png' -> 'Atelectasis|Effusion')
    disease_map = dict(zip(df["Image Index"], df["Finding Labels"]))
    
    embeddings = []
    metadata = []
    
    # Process sequentially with progress bar
    for img_path in tqdm(image_paths, desc="🧠 BiomedCLIP + Ground Truth Analysis"):
        vector = encode_medical_image(str(img_path), model, preprocess, device)
        if vector is not None:
            filename = img_path.name
            
            # Lookup the actual patient's clinical diagnosis!
            # If for some reason the map doesn't have it, default to "No Finding"
            actual_diagnosis = disease_map.get(filename, "No Finding")
            
            embeddings.append(vector)
            
            # Phase 3: We now save RICH METADATA instead of just file paths
            # This is what gets sent to the LLM to cure hallucination!
            patient_record = {
                "file_path": str(img_path.relative_to(processed_dir.parent.parent)),
                "actual_diagnosis": actual_diagnosis
            }
            metadata.append(patient_record)

    # Save to Disk
    embeddings_np = np.vstack(embeddings)
    os.makedirs(os.path.dirname(embeddings_out), exist_ok=True)
    np.save(embeddings_out, embeddings_np)
    
    # Save the structured Ground Truth JSON 
    metadata_out = paths_out.replace(".txt", ".json")
    with open(metadata_out, "w") as f:
        json.dump(metadata, f, indent=4)
            
    logger.info(f"Success! {len(embeddings_np)} vectors and medical labels saved to {embeddings_out}")

if __name__ == "__main__":
    config_path = str(Path(__file__).resolve().parent.parent / "config.yaml")
    config = load_config(config_path)
    build_biomed_embeddings(config)
