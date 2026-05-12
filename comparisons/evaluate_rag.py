import os
import sys
import numpy as np
import faiss
import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__, log_file="logs/evaluate_rag.log")

def evaluate_rag():
    print("="*60)
    print("RUNNING ACADEMIC RAG EVALUATION (BiomedCLIP + FAISS)")
    print("="*60)

    # 1. Load the knowledge base
    embeddings_path = "models/embeddings.npy"
    metadata_path = "models/image_paths.json"

    if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
        logger.error("Vectors or Metadata not found! Please run vision_encoder.py first.")
        return

    logger.info("Loading 512-D Vectors and Ground Truth JSON...")
    embeddings = np.load(embeddings_path).astype(np.float32)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Make sure we have the exact same number of images and labels!
    assert len(embeddings) == len(metadata), "Mismatch between vector count and label count!"

    # Evaluate on ALL classes present in the database — no cherry-picking
    def extract_core_disease(label):
        if "No Finding" in label: return "Normal"
        return label.split("|")[0].strip()

    for i, m in enumerate(metadata):
        m["actual_diagnosis"] = extract_core_disease(m["actual_diagnosis"])

    logger.info(f"Total images in database: {len(metadata)} across all NIH classes.")

    # 2. Strict 70/15/15 Train/Val/Test Split
    logger.info("Splitting dataset: 70% Train | 15% Validation | 15% Test")
    
    indices = np.arange(len(embeddings))
    train_temp, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    train_idx, val_idx = train_test_split(train_temp, test_size=0.1764, random_state=42)

    db_embeddings = embeddings[train_idx]
    db_metadata = [metadata[i] for i in train_idx]

    query_embeddings = embeddings[test_idx]
    query_metadata = [metadata[i] for i in test_idx]

    logger.info(f"Database (Train) Size: {len(db_embeddings)} X-Rays")
    logger.info(f"Validation Set Size: {len(val_idx)} X-Rays (Ignored in zero-shot RAG)")
    logger.info(f"Test Set Size: {len(query_embeddings)} X-Rays")

    logger.info("Building FAISS Test Database...")
    vector_dimension = db_embeddings.shape[1]   # Auto-detect: don't hardcode 512
    logger.info(f"Detected embedding dimension: {vector_dimension}")
    index = faiss.IndexHNSWFlat(vector_dimension, 32)
    index.add(db_embeddings)

    # 4. Run the Evaluation
    logger.info("Searching Database for Test Patients...")
    top_k = 15                  # Retrieve more candidates for better voting
    MIN_SIMILARITY = 0.0        # Threshold: ignore matches below this cosine sim
    distances, closest_indices = index.search(query_embeddings, top_k)

    y_true      = []
    y_pred      = []   # Weighted vote prediction
    y_pred_top1 = []   # Precision@1: is the single closest match correct?

    for i in range(len(query_embeddings)):
        true_label = query_metadata[i]["actual_diagnosis"]
        y_true.append(true_label)

        # ── Similarity-Weighted Voting ───────────────────────────────────────
        # FAISS HNSW returns L2 distances. Convert to cosine-like similarity:
        # similarity = 1 / (1 + distance)  → closer = higher weight
        vote_weights = {}  # {label: total_weight}
        top1_label   = None
        top1_sim     = -1

        for rank, (row_number, dist) in enumerate(zip(closest_indices[i], distances[i])):
            if row_number < 0 or row_number >= len(db_metadata):
                continue
            similarity  = 1.0 / (1.0 + float(dist))   # Convert distance → similarity
            if similarity < MIN_SIMILARITY:
                continue
            match_label = db_metadata[row_number]["actual_diagnosis"]

            # Accumulate weighted votes
            vote_weights[match_label] = vote_weights.get(match_label, 0) + similarity

            # Track top-1 match (closest)
            if similarity > top1_sim:
                top1_sim   = similarity
                top1_label = match_label

        # Weighted prediction: pick the label with the highest total weight
        if vote_weights:
            weighted_pred = max(vote_weights, key=vote_weights.get)
        else:
            weighted_pred = "Normal"   # Fallback

        y_pred.append(weighted_pred)
        y_pred_top1.append(top1_label if top1_label else "Normal")

    # 5. Output Academic Metrics
    print("\n" + "="*60)
    print("FINAL RAG EVALUATION METRICS")
    print("="*60)

    weighted_acc = accuracy_score(y_true, y_pred)
    top1_acc     = accuracy_score(y_true, y_pred_top1)

    print(f"Weighted Vote Accuracy  (Top-{top_k}): {weighted_acc * 100:.2f}%")
    print(f"Precision@1             (Closest Match): {top1_acc * 100:.2f}%")
    print(f"Retrieval Database Size: {len(db_embeddings):,} X-Rays")
    print(f"Test Set Size          : {len(query_embeddings):,} X-Rays")
    print(f"Embedding Dimension    : {vector_dimension}-D\n")

    top_labels = [k for k, v in Counter(y_true).most_common(6)]
    print("--- Weighted Vote Classification Report ---")
    print(classification_report(y_true, y_pred, labels=top_labels, zero_division=0))
    print("--- Precision@1 Classification Report ---")
    print(classification_report(y_true, y_pred_top1, labels=top_labels, zero_division=0))

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # Filter data to only show the Top 6 most common diseases so the graph looks clean and academic
    filter_indices = [i for i, label in enumerate(y_true) if label in top_labels]
    y_true_filtered = [y_true[i] for i in filter_indices]
    y_pred_filtered = [y_pred[i] for i in filter_indices]
    
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=top_labels, yticklabels=top_labels)
    plt.title('Medical RAG AI - Disease Confusion Matrix')
    plt.ylabel('Actual Diagnosis (Ground Truth)')
    plt.xlabel('AI Predicted Diagnosis (FAISS Majority Vote)')
    
    os.makedirs('logs/graphs', exist_ok=True)
    graph_path = 'logs/graphs/rag_confusion_matrix.png'
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    
    print(f"\n📈 Saved beautiful Confusion Matrix strictly to '{graph_path}' for your presentation!")

if __name__ == "__main__":
    evaluate_rag()
