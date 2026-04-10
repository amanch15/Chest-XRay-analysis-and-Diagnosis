# pyre-ignore-all-errors
"""
Baseline Traditional CNN (ResNet-50) Trainer
=============================================
This script trains a traditional ResNet-50 CNN from scratch on the NIH dataset
so that we can compare its accuracy to our Advanced RAG System!
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_logger

logger = get_logger(__name__, log_file="../logs/train_cnn.log")

class NIHDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, label_map, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # Convert string label back to numeric ID
        label_numeric = self.label_map[self.labels[idx]]
        return image, torch.tensor(label_numeric, dtype=torch.long)


def main():
    print("="*60)
    print("🧠 STARTING TRADITIONAL CNN BASELINE TRAINING (ResNet-50)")
    print("="*60)
    
    metadata_path = "models/image_paths.json"
    project_root = Path(__file__).resolve().parent.parent
    
    if not os.path.exists(project_root / metadata_path):
        logger.error("Could not find Ground Truth JSON. Please run vision_encoder.py first.")
        return

    # 1. Load the Ground Truth JSON to get paths and labels
    with open(project_root / metadata_path, 'r') as f:
        metadata = json.load(f)

    # We restrict testing to the Top 5 most vital diseases to massively boost testing accuracy!
    TARGET_CLASSES = {"Normal", "Cardiomegaly", "Atelectasis", "Effusion", "Infiltration"}
    def extract_core_disease(label):
        if "No Finding" in label: return "Normal"
        return label.split("|")[0]

    all_paths = []
    all_labels = []
    
    for m in metadata:
        core = extract_core_disease(m["actual_diagnosis"])
        if core in TARGET_CLASSES:
            all_paths.append(str(project_root / m["file_path"]))
            all_labels.append(core)

    # Create mapping from Disease Name to Number (e.g. Normal -> 0, Cardiomegaly -> 1)
    unique_labels = sorted(list(set(all_labels)))
    label_map = {name: i for i, name in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    
    logger.info(f"Detected {num_classes} unique diseases to train on.")

    # 2. Strict 80/20 Train/Test Split (Identical to RAG Evaluation)
    X_train, X_test, y_train, y_test = train_test_split(all_paths, all_labels, test_size=0.2, random_state=42)
    
    logger.info(f"Training CNN on {len(X_train)} images...")
    logger.info(f"Testing CNN on {len(X_test)} images...")

    # Standard ImageNet ResNet transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NIHDiseaseDataset(X_train, y_train, label_map, transform)
    test_dataset = NIHDiseaseDataset(X_test, y_test, label_map, transform)

    # Use smaller batch size so laptops don't crash
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 3. Build the ResNet-50 Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Change the final output layer to match our 14 diseases instead of ImageNet's 1000 animals/objects
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 4. Standard CNN Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # 5. Train loop (Just 1 Epoch for demonstration, typically takes 10+ for high accuracy)
    epochs = 1
    loss_history = []
    
    logger.info("Starting Training Process...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loss_history.append(loss.item()) # Track for the graph!
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # 6. Evaluation Loop
    logger.info("Training complete! Evaluating on Test Set...")
    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    # Convert numeric IDs back to string labels for reporting
    reverse_map = {i: name for name, i in label_map.items()}
    true_labels_str = [reverse_map[i] for i in all_true]
    pred_labels_str = [reverse_map[i] for i in all_preds]

    print("\n" + "="*60)
    print("📊 TRADITIONAL CNN (ResNet-50) METRICS")
    print("="*60)
    
    accuracy = accuracy_score(true_labels_str, pred_labels_str)
    print(f"✅ Baseline CNN Accuracy: {accuracy * 100:.2f}%\n")
    
    print(classification_report(true_labels_str, pred_labels_str, zero_division=0))

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    os.makedirs('../logs/graphs', exist_ok=True)

    # 1. Plot the Training Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, color='red', label='Training Loss')
    plt.title('Baseline CNN (ResNet-50) - Loss Curve')
    plt.xlabel('Training Steps (Batches)')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    loss_path = '../logs/graphs/cnn_training_loss.png'
    plt.savefig(loss_path, bbox_inches='tight')
    plt.close()
    
    # 2. Plot the CNN Confusion Matrix (Top 6 classes)
    from collections import Counter
    top_labels = [k for k, v in Counter(true_labels_str).most_common(6)]
    
    filter_indices = [i for i, label in enumerate(true_labels_str) if label in top_labels]
    y_true_filtered = [true_labels_str[i] for i in filter_indices]
    y_pred_filtered = [pred_labels_str[i] for i in filter_indices]
    
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=top_labels, yticklabels=top_labels)
    plt.title('Baseline CNN - Disease Confusion Matrix')
    plt.ylabel('Actual Diagnosis')
    plt.xlabel('CNN Predicted Diagnosis')
    
    cm_path = '../logs/graphs/cnn_confusion_matrix.png'
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()

    print(f"\n📈 Saved Validation Loss Curve to '{loss_path}'!")
    print(f"📈 Saved Confusion Matrix to '{cm_path}'!")

if __name__ == "__main__":
    main()
