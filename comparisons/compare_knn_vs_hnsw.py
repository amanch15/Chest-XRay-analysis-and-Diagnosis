# pyre-ignore-all-errors
"""
comparisons/compare_knn_vs_hnsw.py
====================================
Benchmarks Flat KNN (brute-force) vs HNSW (graph-based) retrieval
on the actual project embeddings and metadata.

Measures for each method:
  - Index build time
  - Per-query search time (averaged over N queries)
  - Diagnosis consistency (do top-3 results agree on a label?)
  - Recall overlap (do both methods return the same top-K results?)

Run:
    python comparisons/compare_knn_vs_hnsw.py
"""

import sys
import time
import json
import numpy as np
import faiss
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, get_logger

logger = get_logger(__name__, log_file="logs/knn_vs_hnsw.log")

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH    = str(Path(__file__).resolve().parent.parent / "config.yaml")
cfg            = load_config(CONFIG_PATH)

EMBEDDINGS_FILE = cfg["paths"]["embeddings"]
METADATA_FILE   = cfg["paths"]["image_paths"].replace(".txt", ".json")
HNSW_M          = cfg["database"].get("hnsw_m", 32)
TOP_K           = 3
NUM_QUERIES     = 50   # Number of random vectors to use as test queries


# ─── Load Data ───────────────────────────────────────────────────────────────
def load_data():
    print("\nLoading embeddings and metadata...")
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    print(f"  Loaded {len(embeddings)} vectors | dim = {embeddings.shape[1]}")
    return embeddings, metadata


# ─── Build Indexes ────────────────────────────────────────────────────────────
def build_flat_knn(embeddings):
    """Builds a Flat (brute-force KNN) FAISS index — exact search, O(N)."""
    dim = embeddings.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    build_time = time.perf_counter() - t0
    return index, build_time


def build_hnsw(embeddings, M=32):
    """Builds an HNSW graph FAISS index — approximate search, O(log N)."""
    dim = embeddings.shape[1]
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(dim, M)
    index.add(embeddings)
    build_time = time.perf_counter() - t0
    return index, build_time


# ─── Query Benchmarks ─────────────────────────────────────────────────────────
def benchmark_search(index, query_vectors, top_k):
    """Runs all queries and returns total time + all result indices."""
    all_indices = []
    t0 = time.perf_counter()
    for qv in query_vectors:
        qv_2d = np.expand_dims(qv, axis=0)
        _, idx = index.search(qv_2d, top_k)
        all_indices.append(idx[0].tolist())
    total_time = time.perf_counter() - t0
    return total_time, all_indices


# ─── Diagnosis Consistency ────────────────────────────────────────────────────
def diagnosis_consistency(all_indices, metadata):
    """
    For each query, checks if the top-3 retrieved cases agree on a diagnosis.
    Returns the % of queries where at least 2/3 results share the same label.
    """
    consistent = 0
    for indices in all_indices:
        labels = []
        for idx in indices:
            if 0 <= idx < len(metadata):
                # Handle multi-label by taking the first label
                label = metadata[idx]["actual_diagnosis"].split("|")[0].strip()
                labels.append(label)
        if labels:
            most_common_count = Counter(labels).most_common(1)[0][1]
            if most_common_count >= 2:   # At least 2 of 3 agree
                consistent += 1
    return (consistent / len(all_indices)) * 100


# ─── Recall Overlap ───────────────────────────────────────────────────────────
def recall_overlap(knn_indices, hnsw_indices):
    """
    Measures how often HNSW returns the same results as exact KNN.
    This is the "recall@K" of HNSW relative to ground-truth KNN.
    """
    overlaps = []
    for knn_idx, hnsw_idx in zip(knn_indices, hnsw_indices):
        knn_set  = set(knn_idx)
        hnsw_set = set(hnsw_idx)
        overlap  = len(knn_set & hnsw_set) / len(knn_set) if knn_set else 0
        overlaps.append(overlap)
    return np.mean(overlaps) * 100


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    embeddings, metadata = load_data()

    # Pick random query vectors from the dataset itself
    rng          = np.random.default_rng(seed=42)
    query_idx    = rng.choice(len(embeddings), size=NUM_QUERIES, replace=False)
    query_vectors = embeddings[query_idx]

    # ── Build both indexes ──
    print(f"\nBuilding indexes on {len(embeddings)} vectors (dim={embeddings.shape[1]})...")
    knn_index,  knn_build  = build_flat_knn(embeddings)
    hnsw_index, hnsw_build = build_hnsw(embeddings, M=HNSW_M)

    # ── Run searches ──
    print(f"\nRunning {NUM_QUERIES} queries (top-{TOP_K}) on both indexes...")
    knn_time,  knn_results  = benchmark_search(knn_index,  query_vectors, TOP_K)
    hnsw_time, hnsw_results = benchmark_search(hnsw_index, query_vectors, TOP_K)

    # ── Metrics ──
    knn_consistency  = diagnosis_consistency(knn_results,  metadata)
    hnsw_consistency = diagnosis_consistency(hnsw_results, metadata)
    recall           = recall_overlap(knn_results, hnsw_results)

    knn_avg_ms  = (knn_time  / NUM_QUERIES) * 1000
    hnsw_avg_ms = (hnsw_time / NUM_QUERIES) * 1000

    # ── Print Results ──
    print("\n" + "=" * 65)
    print("  KNN (Brute-Force Flat)  vs  HNSW (Graph-Based)  —  Results")
    print("=" * 65)
    print(f"  {'Metric':<35} {'KNN':>10} {'HNSW':>10}")
    print("-" * 65)
    print(f"  {'Index Build Time (seconds)':<35} {knn_build:>9.3f}s {hnsw_build:>9.3f}s")
    print(f"  {'Avg Query Time (ms per query)':<35} {knn_avg_ms:>9.3f}  {hnsw_avg_ms:>9.3f}")
    print(f"  {'Total Search Time (seconds)':<35} {knn_time:>9.4f}s {hnsw_time:>9.4f}s")
    print(f"  {'Diagnosis Consistency (%)':<35} {knn_consistency:>9.1f}% {hnsw_consistency:>9.1f}%")
    print(f"  {'HNSW Recall vs KNN (%)':<35} {'(ground truth)':>10} {recall:>9.1f}%")
    print(f"  {'Index Type':<35} {'Exact':>10} {'Approx':>10}")
    print(f"  {'Complexity':<35} {'O(N)':>10} {'O(log N)':>10}")
    print(f"  {'Database Size':<35} {len(embeddings):>10} {len(embeddings):>10}")
    print("=" * 65)

    # Speed improvement
    if hnsw_avg_ms > 0:
        speedup = knn_avg_ms / hnsw_avg_ms
        print(f"\n  HNSW is {speedup:.1f}x faster per query than brute-force KNN")
    print(f"  HNSW Recall@{TOP_K}: {recall:.1f}% "
          f"({'Excellent' if recall >= 90 else 'Good' if recall >= 75 else 'Fair'})")
    print()


if __name__ == "__main__":
    main()
