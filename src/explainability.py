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

        # Use only the TOP 20% most-activated channels (not all 1024 averaged equally)
        # This prevents low-activation channels from diluting the pathological signal
        channel_importance = feature_maps.mean(dim=[-2, -1])      # [1024] mean per channel
        top_k_channels     = int(0.2 * feature_maps.shape[0])     # Top 20% = 205 channels
        top_indices        = torch.topk(channel_importance, top_k_channels).indices
        heatmap = feature_maps[top_indices].mean(dim=0).cpu().numpy()  # [7, 7]

        # Normalize to [0, 1]
        heatmap = np.maximum(heatmap, 0)
        hmin, hmax = heatmap.min(), heatmap.max()
        heatmap = (heatmap - hmin) / (hmax - hmin + 1e-8)

        # Upsample to 224x224 with smooth cubic interpolation
        heatmap = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Smooth out blocky CNN artifacts with moderate Gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), sigmaX=5)

        # Normalize the raw smooth heatmap (thresholding happens later after ViT masking)
        hmin, hmax = heatmap.min(), heatmap.max()
        heatmap = (heatmap - hmin) / (hmax - hmin + 1e-8)
        return heatmap

    except Exception as e:
        logger.error(f"DenseNet GradCAM failed: {e}")
        return np.zeros((224, 224), dtype=np.float32)
    finally:
        hook.remove()


# ─── 2. BiomedCLIP SmoothGrad Saliency Map ───────────────────────────────────

def saliency_biomedclip(
    image_path: str,
    biomed_model,
    biomed_preprocess,
    device: torch.device,
    n_samples: int = 20,
    noise_level: float = 0.10
) -> np.ndarray:
    """
    SmoothGrad saliency for BiomedCLIP ViT.
    Runs n_samples noisy forward passes and averages the gradients,
    cancelling noise and converging on the spatially important region.
    Reference: Smilkov et al., SmoothGrad 2017.
    """
    try:
        image       = Image.open(image_path).convert("RGB")
        base_tensor = biomed_preprocess(image).unsqueeze(0).to(device)

        accumulated = torch.zeros_like(base_tensor)
        sigma       = noise_level * (base_tensor.max() - base_tensor.min()).item()

        for _ in range(n_samples):
            noisy = (base_tensor + torch.randn_like(base_tensor) * sigma).detach().requires_grad_(True)
            with torch.set_grad_enabled(True):
                loss = biomed_model.encode_image(noisy).norm()
                loss.backward()
            accumulated += noisy.grad.data.abs()

        avg_grad = (accumulated / n_samples).squeeze(0)       # [3, H, W]
        saliency = avg_grad.max(dim=0)[0].cpu().numpy()       # [H, W]

        # Heavy Gaussian blur to eliminate the ViT 16x16 patch grid checkerboard
        # The patch grid is a ViT artifact — strong blur merges patches into regions
        saliency = cv2.GaussianBlur(saliency, (31, 31), sigmaX=12)

        # Percentile thresholding: keep only top 10% activations
        # Forces the ViT map to show a specific region, not the whole lung
        threshold = np.percentile(saliency, 90)
        saliency  = np.where(saliency >= threshold, saliency, 0.0)

        # Final gentle smoothing to clean edges
        saliency = cv2.GaussianBlur(saliency, (11, 11), sigmaX=4)

        smin, smax = saliency.min(), saliency.max()
        saliency   = (saliency - smin) / (smax - smin + 1e-8)
        return saliency

    except Exception as e:
        logger.error(f"BiomedCLIP SmoothGrad saliency failed: {e}")
        return np.zeros((224, 224), dtype=np.float32)


# ─── 3. Overlay Generator ────────────────────────────────────────────────────

def create_heatmap_overlay(
    image_path: str,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET   # Overridable per model
) -> Image.Image:
    """
    Blends a [0,1] float heatmap with the original image.
    - DenseNet GradCAM    → COLORMAP_INFERNO (black→red→yellow) — warm/fire tones
    - BiomedCLIP Saliency → COLORMAP_VIRIDIS (purple→blue→green) — cool tones
    - Combined            → COLORMAP_JET     (blue→green→red)    — standard medical
    Returns a display-ready PIL image for Streamlit.
    """
    try:
        orig = np.array(Image.open(image_path).convert("RGB").resize((224, 224)))

        heatmap_uint8   = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
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
    logger.info("Running XAI pipeline (SmoothGrad + GradCAM)...")

    bio_heatmap = saliency_biomedclip(image_path, biomed_model, biomed_preprocess, device)
    dn_heatmap  = gradcam_densenet(image_path, densenet_model, densenet_transform, device)

    # Adaptive blending based on spatial correlation
    flat_dn  = dn_heatmap.flatten()
    flat_bio = bio_heatmap.flatten()
    correlation = float(np.corrcoef(flat_dn, flat_bio)[0, 1])
    correlation = max(0.0, correlation)
    cnn_weight = 0.7 - 0.2 * correlation
    vit_weight = 1.0 - cnn_weight
    logger.info(f"XAI spatial correlation: {correlation:.2f} → CNN:{cnn_weight:.2f} / ViT:{vit_weight:.2f}")

    combined = cnn_weight * dn_heatmap + vit_weight * bio_heatmap
    norm = combined.max()
    if norm > 0:
        combined = combined / norm

    activated_region = detect_activated_region(combined)

    return {
        "densenet_heatmap":  dn_heatmap,
        "biomed_heatmap":    bio_heatmap,
        "combined_heatmap":  combined,
        "densenet_overlay":  create_heatmap_overlay(image_path, dn_heatmap, colormap=cv2.COLORMAP_INFERNO),
        "biomed_overlay":    create_heatmap_overlay(image_path, bio_heatmap, colormap=cv2.COLORMAP_VIRIDIS),
        "combined_overlay":  create_heatmap_overlay(image_path, combined,    colormap=cv2.COLORMAP_JET),
        "activated_region":  activated_region,
    }
