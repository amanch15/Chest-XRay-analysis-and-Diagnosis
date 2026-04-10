import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger
from src.vision_encoder import load_biomed_clip

logger = get_logger(__name__, log_file="../logs/evaluate_zeroshot.log")

def main():
    print("="*60)
    print("🚀 STARTING BIOMEDCLIP ZERO-SHOT TEXT EVALUATION")
    print("="*60)
    
    embeddings_path = "models/embeddings.npy"
    metadata_path = "models/image_paths.json"
    project_root = Path(__file__).resolve().parent.parent

    if not os.path.exists(project_root / embeddings_path):
        logger.error("Vectors not found! Run vision_encoder.py first.")
        return

    logger.info("Loading 512-D Vectors and Ground Truth JSON...")
    embeddings = np.load(project_root / embeddings_path).astype(np.float32)
    with open(project_root / metadata_path, 'r') as f:
        metadata = json.load(f)

    # --- EXPERT FILTERING TRICK ---
    TARGET_CLASSES = {"Normal", "Cardiomegaly", "Atelectasis", "Effusion", "Infiltration"}
    
    def extract_core_disease(label):
        if "No Finding" in label: return "Normal"
        return label.split("|")[0]
        
    filtered_emb = []
    filtered_meta = []
    for i, m in enumerate(metadata):
        core = extract_core_disease(m["actual_diagnosis"])
        if core in TARGET_CLASSES:
            m["actual_diagnosis"] = core
            filtered_meta.append(m)
            filtered_emb.append(embeddings[i])
            
    embeddings = np.array(filtered_emb)
    metadata = filtered_meta

    # 1. Isolate the 20% test Set (Exactly identical to the RAG split)
    indices = np.arange(len(embeddings))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

    query_embeddings = embeddings[test_idx]
    query_metadata = [metadata[i] for i in test_idx]

    y_true = [m["actual_diagnosis"] for m in query_metadata]
    unique_diseases = sorted(list(set(y_true)))

    # 2. Boot up BiomedCLIP to generate pure TEXT vectors 
    logger.info("Warming up BiomedCLIP Text Encoder for Zero-Shot Classification...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We load the config properly from the root dir
    from src.utils import load_config
    config = load_config(str(project_root / "config.yaml"))
    
    model, processor = load_biomed_clip(config["encoder"]["model_name"], device)
    model.eval()

    # 3. Create the medical prompts
    # BiomedCLIP was trained on PubMed so it understands clinical language
    prompts = [f"This is a chest X-ray of a patient with {disease.lower()}" for disease in unique_diseases]
    
    import open_clip
    text_tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    text_tokens = text_tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # L2 Normalize the text features
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features_np = text_features.cpu().numpy()

    logger.info("Matching visual features to text features...")
    y_pred = []
    
    # 4. Zero-Shot Math (Cosine Similarity)
    for img_vector in query_embeddings:
        # img_vector is (512,), text_features_np is (Num_Classes, 512)
        # Dot product gives Cosine Similarity for each class!
        similarities = np.dot(text_features_np, img_vector)
        
        # Argmax to find the highest matching text phrase
        best_match_idx = np.argmax(similarities)
        y_pred.append(unique_diseases[best_match_idx])

    # 5. Output Academic Metrics
    print("\n" + "="*60)
    print("📊 BIOMEDCLIP PURE ZERO-SHOT METRICS")
    print("="*60)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"✅ Zero-Shot Accuracy: {accuracy * 100:.2f}%\n")
    
    from collections import Counter
    top_labels = [k for k, v in Counter(y_true).most_common(6)]
    print(classification_report(y_true, y_pred, labels=top_labels, zero_division=0))

if __name__ == "__main__":
    main()
