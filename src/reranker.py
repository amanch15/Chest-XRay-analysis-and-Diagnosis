# pyre-ignore-all-errors
"""
reranker.py — BiomedCLIP Cross-Encoder Reranker
================================================
After FAISS retrieves the Top-50 nearest neighbours by vector distance,
this module reranks them using BiomedCLIP's text encoder (PubMedBERT).

For each candidate, it encodes the diagnosis label as clinical text and
computes image-to-text cosine similarity. This gives a combined score:

    final_score = alpha * faiss_similarity + (1 - alpha) * text_similarity

where the BiomedCLIP image-text alignment acts as a genuine cross-modal
second-stage reranker — boosting matches where both visual similarity
AND the clinical text match the query image.
"""

import sys
import numpy as np
import torch
import open_clip
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger

logger = get_logger(__name__, log_file="logs/reranker.log")

# ─── Clinical text prompts for each NIH diagnosis label ──────────────────────
# Rich prompts give BiomedCLIP's PubMedBERT encoder better embeddings than raw labels.

DIAGNOSIS_PROMPTS = {
    "No Finding":          "A normal chest X-ray with no significant pathological findings",
    "Atelectasis":         "Chest X-ray showing atelectasis with partial lung collapse",
    "Consolidation":       "Chest X-ray showing pulmonary consolidation with airspace opacity",
    "Infiltration":        "Chest X-ray showing diffuse lung infiltration with patchy opacity",
    "Pneumothorax":        "Chest X-ray showing pneumothorax with visible pleural line and air collection",
    "Effusion":            "Chest X-ray showing pleural effusion with blunted costophrenic angle",
    "Pleural_Thickening":  "Chest X-ray showing pleural thickening along the chest wall",
    "Mass":                "Chest X-ray showing a large pulmonary mass or tumor",
    "Nodule":              "Chest X-ray showing a small pulmonary nodule",
    "Emphysema":           "Chest X-ray showing pulmonary emphysema with hyperinflation",
    "Fibrosis":            "Chest X-ray showing pulmonary fibrosis with reticular markings",
    "Cardiomegaly":        "Chest X-ray showing cardiomegaly with an enlarged cardiac silhouette",
    "Hernia":              "Chest X-ray showing diaphragmatic hernia with bowel in the chest",
    "Pneumonia":           "Chest X-ray showing pneumonia with lobar or patchy consolidation",
}


def load_text_tokenizer(model_name: str):
    """Returns BiomedCLIP's PubMedBERT tokenizer for encoding diagnosis text."""
    return open_clip.get_tokenizer(model_name)


def _encode_label(label: str, biomed_model, tokenizer, device: torch.device) -> np.ndarray:
    """Encodes a single diagnosis label string → 512-dim L2-normalized text vector."""
    prompt = DIAGNOSIS_PROMPTS.get(label.strip(), f"Chest X-ray showing {label.strip().lower()}")
    try:
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            text_features = biomed_model.encode_text(tokens)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:
        logger.warning(f"Text encoding failed for '{label}': {e}")
        return None


def cross_encoder_rerank(
    query_combined_vector: np.ndarray,
    candidates: list,
    biomed_model,
    tokenizer,
    device: torch.device,
    alpha: float = 0.6,
    top_k: int = 3
) -> list:
    """
    Cross-encoder reranker using BiomedCLIP vision-language alignment.

    Args:
        query_combined_vector : 1536-dim vector [DenseNet(1024) | BiomedCLIP(512)]
        candidates            : list of dicts from FAISS with 'diagnosis' and 'similarity'
        biomed_model          : BiomedCLIP model (has encode_text)
        tokenizer             : BiomedCLIP tokenizer
        device                : torch device
        alpha                 : weight for FAISS score (1-alpha = weight for text sim)
        top_k                 : number of final results to return

    Returns:
        top_k reranked candidates sorted by combined_score descending.
    """
    # Extract the BiomedCLIP slice from the 1536-dim query vector
    query_biomed_vec = query_combined_vector[1024:]   # shape (512,)

    # Pre-compute text vectors for all unique diagnosis labels
    unique_labels = set()
    for c in candidates:
        for part in c["diagnosis"].split("|"):
            unique_labels.add(part.strip())

    logger.info(f"Encoding {len(unique_labels)} unique diagnosis labels via text encoder...")
    text_cache = {}
    for label in unique_labels:
        vec = _encode_label(label, biomed_model, tokenizer, device)
        if vec is not None:
            text_cache[label] = vec

    # Score each candidate
    scored = []
    for cand in candidates:
        faiss_sim = float(cand["similarity"])   # already converted to cosine [0,1]

        # For multi-label diagnoses (e.g. "Atelectasis|Effusion"), take max text sim
        sub_labels  = [l.strip() for l in cand["diagnosis"].split("|")]
        text_sims   = [
            float(np.dot(query_biomed_vec, text_cache[l]))
            for l in sub_labels if l in text_cache
        ]
        text_score = max(text_sims) if text_sims else 0.0

        # Normalise text score from [-1,1] → [0,1]
        text_score_norm = (text_score + 1.0) / 2.0

        combined_score = alpha * faiss_sim + (1.0 - alpha) * text_score_norm

        scored.append({
            **cand,
            "text_similarity":  round(text_score, 4),
            "combined_score":   round(combined_score, 4),
        })

    scored.sort(key=lambda x: x["combined_score"], reverse=True)
    logger.info(f"Reranked {len(candidates)} candidates → top {top_k} selected.")
    return scored[:top_k]
