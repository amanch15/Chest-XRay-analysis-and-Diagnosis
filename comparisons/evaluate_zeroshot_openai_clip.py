import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import open_clip

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__, log_file="../logs/evaluate_openai.log")

def main():
    print("="*60)
    print("🚀 STARTING OPENAI CLIP (GENERAL) ZERO-SHOT ABLATION STUDY")
    print("="*60)
    
    metadata_path = "models/image_paths.json"
    project_root = Path(__file__).resolve().parent.parent

    if not os.path.exists(project_root / metadata_path):
        logger.error("Metadata not found! Run vision_encoder.py first.")
        return

    logger.info("Loading Ground Truth JSON...")
    with open(project_root / metadata_path, 'r') as f:
        metadata = json.load(f)

   
    TARGET_CLASSES = {"Normal", "Cardiomegaly", "Atelectasis", "Effusion", "Infiltration"}
    
    def extract_core_disease(label):
        if "No Finding" in label: return "Normal"
        return label.split("|")[0]
        
    filtered_meta = []
    for m in metadata:
        core = extract_core_disease(m["actual_diagnosis"])
        if core in TARGET_CLASSES:
            m["actual_diagnosis"] = core
            filtered_meta.append(m)
            
    metadata = filtered_meta

   
    indices = np.arange(len(metadata))
    _, test_idx = train_test_split(indices, test_size=0.15, random_state=42)

    query_metadata = [metadata[i] for i in test_idx]
    y_true = [m["actual_diagnosis"] for m in query_metadata]
    unique_diseases = sorted(list(set(y_true)))

    
    logger.info("Warming up standard OpenAI CLIP Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='openai', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32-quickgelu')
    model.eval()

    
    prompts = [f"This is a chest X-ray of a patient with {disease.lower()}" for disease in unique_diseases]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features_np = text_features.cpu().numpy()

    logger.info(f"Extracting visual features for {len(query_metadata)} test images using standard CLIP...")
    y_pred = []
    
   
    for idx, m in enumerate(tqdm(query_metadata, desc="Analyzing test images")):
        img_path = project_root / m["file_path"]
        try:
            image = Image.open(img_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                img_vector = model.encode_image(image_input)
                img_vector = img_vector / img_vector.norm(p=2, dim=-1, keepdim=True)
                img_vector = img_vector.squeeze().cpu().numpy()
                
           
            similarities = np.dot(text_features_np, img_vector)
            best_match_idx = np.argmax(similarities)
            y_pred.append(unique_diseases[best_match_idx])
            
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
            y_pred.append("Normal") 

    
    print("\n" + "="*60)
    print(" OPENAI CLIP (GENERAL) ZERO-SHOT METRICS")
    print("="*60)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"OpenAI CLIP Accuracy: {accuracy * 100:.2f}%\n")
    
    from collections import Counter
    top_labels = [k for k, v in Counter(y_true).most_common(6)]
    print(classification_report(y_true, y_pred, labels=top_labels, zero_division=0))

    #  Confusion Matrix
    filter_indices = [i for i, label in enumerate(y_true) if label in top_labels]
    y_true_filtered = [y_true[i] for i in filter_indices]
    y_pred_filtered = [y_pred[i] for i in filter_indices]
    
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=top_labels, yticklabels=top_labels)
    plt.title('Standard OpenAI CLIP - Disease Confusion Matrix')
    plt.ylabel('Actual Diagnosis')
    plt.xlabel('OpenAI Predicted Diagnosis')
    
    os.makedirs('../logs/graphs', exist_ok=True)
    graph_path = '../logs/graphs/openai_confusion_matrix.png'
    plt.savefig(graph_path, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved OpenAI Confusion Matrix to '{graph_path}'")

if __name__ == "__main__":
    main()
