# pyre-ignore-all-errors
"""
explainability.py — XAI: GradCAM + Saliency Maps
==================================================
Implements two complementary Explainable AI techniques on the Dual Encoder:

  1. DenseNet121 Feature Map Visualization (CNN — local spatial features)
     Hooks the last DenseBlock's output, averages channels → 7×7 heatmap → 224×224

  2. BiomedCLIP Input Gradient Saliency (ViT — global attention features)
     Backpropagates through the output norm → gradient magnitude at each pixel → 224×224

  3. Combined Heatmap: weighted average of both → most reliable activation signal

  4. Anatomical Region Detection: divides heatmap into 6 lung zones,
     returns the name of the highest-activated region (e.g., "Right Lower Lobe (RLL)")
     → this text is injected into the LLM prompt for grounded report generation.

  5. Overlay Generation: blends heatmap with original X-ray using JET colormap
     → display-ready PIL images for Streamlit.
"""

import sys
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger

logger = get_logger(__name__, log_file="logs/explainability.log")


# ─── 1. DenseNet121 Feature Map Heatmap ──────────────────────────────────────

def gradcam_densenet(
    image_path: str,
    densenet_model,
    densenet_transform,
    device: torch.device
) -> np.ndarray:
    """
    Hooks the last DenseBlock of DenseNet121 to capture spatial feature maps.
    Averages them across channels to produce a 7x7 heatmap, upsampled to 224x224.
    No gradients required — uses raw feature activations (CAM-lite approach).
    """
    activation_store = {}

    def hook_fn(module, input, output):
        activation_store["features"] = output.detach()

    # Register hook on the last DenseBlock (deepest CNN features)
    hook = densenet_model.features.denseblock4.register_forward_hook(hook_fn)

    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = densenet_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            densenet_model(input_tensor)

        # feature maps shape: [1, 1024, 7, 7]
        feature_maps = activation_store["features"].squeeze(0)   # [1024, 7, 7]
        heatmap = feature_maps.mean(dim=0).cpu().numpy()          # [7, 7]

        # Normalize to [0, 1]
        heatmap = np.maximum(heatmap, 0)
        hmin, hmax = heatmap.min(), heatmap.max()
        heatmap = (heatmap - hmin) / (hmax - hmin + 1e-8)

        # Upsample to 224x224
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)
        return heatmap

    except Exception as e:
        logger.error(f"DenseNet GradCAM failed: {e}")
        return np.zeros((224, 224), dtype=np.float32)
    finally:
        hook.remove()


# ─── 2. BiomedCLIP Input Gradient Saliency Map ───────────────────────────────

def saliency_biomedclip(
    image_path: str,
    biomed_model,
    biomed_preprocess,
    device: torch.device
) -> np.ndarray:
    """
    Computes an input gradient saliency map through BiomedCLIP's ViT.
    Backpropagates the norm of the output embedding to find which pixels
    influenced the feature vector most — a well-established XAI technique.
    """
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = biomed_preprocess(image).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        with torch.set_grad_enabled(True):
            features = biomed_model.encode_image(input_tensor)
            loss = features.norm()
            loss.backward()

        # Gradient magnitude: max across RGB channels → [224, 224]
        saliency = input_tensor.grad.data.abs().squeeze(0)  # [3, 224, 224]
        saliency = saliency.max(dim=0)[0].cpu().numpy()      # [224, 224]

        smin, smax = saliency.min(), saliency.max()
        saliency = (saliency - smin) / (smax - smin + 1e-8)
        return saliency

    except Exception as e:
        logger.error(f"BiomedCLIP saliency failed: {e}")
        return np.zeros((224, 224), dtype=np.float32)


# ─── 3. Overlay Generator ────────────────────────────────────────────────────

def create_heatmap_overlay(
    image_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> Image.Image:
    """
    Blends a [0,1] float heatmap with the original image using JET colormap.
    Returns a display-ready PIL image for Streamlit.
    """
    try:
        orig = np.array(Image.open(image_path).convert("RGB").resize((224, 224)))

        heatmap_uint8   = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        overlay = (alpha * heatmap_colored + (1 - alpha) * orig).astype(np.uint8)
        return Image.fromarray(overlay)

    except Exception as e:
        logger.error(f"Overlay creation failed: {e}")
        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


# ─── 4. Anatomical Region Detector ───────────────────────────────────────────

def detect_activated_region(heatmap: np.ndarray) -> str:
    """
    Divides the 224x224 heatmap into 6 anatomical lung zones and returns
    the name of the zone with the highest mean activation.
    This text is injected into the LLM prompt for grounded report generation.

    Zone Layout (PA/AP view, patient perspective):
        ┌──────────────────┬──────────────────┐
        │  Left Upper (LUL)│ Right Upper (RUL) │  ← top 1/3
        ├──────────────────┼──────────────────┤
        │  Left Lingula    │ Right Middle (RML)│  ← middle 1/3
        ├──────────────────┼──────────────────┤
        │  Left Lower (LLL)│ Right Lower (RLL) │  ← bottom 1/3
        └──────────────────┴──────────────────┘
    Note: image left = patient right (standard radiological convention)
    """
    h, w = heatmap.shape
    h3, w2 = h // 3, w // 2

    zones = {
        "Right Upper Lobe (RUL)":   heatmap[:h3,    w2:],
        "Left Upper Lobe (LUL)":    heatmap[:h3,    :w2],
        "Right Middle Lobe (RML)":  heatmap[h3:2*h3, w2:],
        "Left Lingula":             heatmap[h3:2*h3, :w2],
        "Right Lower Lobe (RLL)":   heatmap[2*h3:,   w2:],
        "Left Lower Lobe (LLL)":    heatmap[2*h3:,   :w2],
    }

    activated = max(zones.items(), key=lambda x: float(x[1].mean()))
    logger.info(f"Highest activation detected in: {activated[0]}")
    return activated[0]


# ─── 5. Master XAI Pipeline ──────────────────────────────────────────────────

def generate_xai_heatmaps(
    image_path: str,
    biomed_model,
    biomed_preprocess,
    densenet_model,
    densenet_transform,
    device: torch.device
) -> dict:
    """
    Runs both XAI methods and returns:
      - densenet_heatmap   : raw 224x224 numpy heatmap from DenseNet121
      - biomed_heatmap     : raw 224x224 numpy saliency from BiomedCLIP
      - combined_heatmap   : weighted average (60% DenseNet, 40% BiomedCLIP)
      - densenet_overlay   : PIL image — DenseNet heatmap blended on X-ray
      - biomed_overlay     : PIL image — BiomedCLIP saliency blended on X-ray
      - combined_overlay   : PIL image — combined heatmap blended on X-ray
      - activated_region   : string name of highest-activation anatomical zone
    """
    logger.info("Running XAI pipeline (GradCAM + Saliency)...")

    dn_heatmap  = gradcam_densenet(image_path, densenet_model, densenet_transform, device)
    bio_heatmap = saliency_biomedclip(image_path, biomed_model, biomed_preprocess, device)

    # Weighted combination: DenseNet (local CNN) + BiomedCLIP (global ViT)
    combined = 0.6 * dn_heatmap + 0.4 * bio_heatmap
    norm = combined.max()
    if norm > 0:
        combined = combined / norm

    activated_region = detect_activated_region(combined)

    return {
        "densenet_heatmap":  dn_heatmap,
        "biomed_heatmap":    bio_heatmap,
        "combined_heatmap":  combined,
        "densenet_overlay":  create_heatmap_overlay(image_path, dn_heatmap),
        "biomed_overlay":    create_heatmap_overlay(image_path, bio_heatmap),
        "combined_overlay":  create_heatmap_overlay(image_path, combined),
        "activated_region":  activated_region,
    }
