# pyre-ignore-all-errors
"""
vector_db.py — The Vector Database
===================================
Upgraded: search_for_similar_images now supports a configurable initial_pull
parameter to return more candidates for the cross-encoder reranker.
"""

import os
import sys
import faiss
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import get_logger, load_config

logger = get_logger(__name__, log_file="logs/vector_db.log")


def build_and_save_database(embeddings_file: str, save_location: str, cfg: dict):
    """
    Loads embeddings.npy and builds an HNSW FAISS index.
    The embedding dimension is read automatically from the npy file,
    so it works for both 512-dim (old) and 1536-dim (new dual encoder) vectors.
    """
    if not os.path.exists(embeddings_file):
        logger.error("Could not find the embeddings! Did you run vision_encoder.py?")
        return

    logger.info("1. Loading embeddings from disk...")
    image_vectors = np.load(embeddings_file).astype(np.float32)

    total_images = len(image_vectors)
    vector_size  = image_vectors.shape[1]
    logger.info(f"   -> {total_images} images, {vector_size}-dim vectors.")

    logger.info("2. Building HNSW FAISS index...")
    hnsw_m = cfg["database"].get("hnsw_m", 32)
    search_database = faiss.IndexHNSWFlat(vector_size, hnsw_m)
    search_database.add(image_vectors)

    logger.info("3. Saving index to disk...")
    os.makedirs(os.path.dirname(save_location), exist_ok=True)
    faiss.write_index(search_database, save_location)

    logger.info(f"✅ FAISS index saved to: {save_location}")


def search_for_similar_images(
    query_vector: np.ndarray,
    database,
    image_paths: list,
    top_k: int = 50
) -> list:
    """
    Searches the FAISS index and returns the top_k most similar candidates.
    Set top_k=50 to feed into the cross-encoder reranker; top_k=3 for direct use.

    Returns list of dicts with:
        image_path  : relative path to the matched image
        diagnosis   : ground-truth NIH diagnosis label
        similarity  : cosine similarity score in [0, 1]
    """
    query_vector = np.array(query_vector).astype(np.float32)
    if query_vector.ndim == 1:
        query_vector = np.expand_dims(query_vector, axis=0)

    distances, indices = database.search(query_vector, top_k)

    results = []
    for distance, row_number in zip(distances[0], indices[0]):
        if row_number < 0:              # FAISS returns -1 for invalid entries
            continue
        meta = image_paths[row_number]
        # Convert L2 distance → cosine similarity for L2-normalized vectors
        # d = ||a-b||² = 2 - 2·cos(a,b)  →  cos = 1 - d/2
        faiss_similarity = float(1.0 - distance / 2.0)
        results.append({
            "image_path": meta["file_path"],
            "diagnosis":  meta["actual_diagnosis"],
            "similarity": faiss_similarity,
        })

    return results


# ─── This part only runs if you run the file directly ───────────────────────
if __name__ == "__main__":
    config_path = str(Path(__file__).resolve().parent.parent / "config.yaml")
    settings    = load_config(config_path)

    emb_file  = settings["paths"]["embeddings"]
    save_file = settings["paths"].get("faiss_index", "models/faiss_index.bin")

    logger.info("🚀 Starting FAISS Database Builder...")
    build_and_save_database(emb_file, save_file, settings)
