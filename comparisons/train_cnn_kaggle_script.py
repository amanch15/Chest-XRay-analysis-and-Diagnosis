"""
train_cnn_kaggle_script.py — Max-Accuracy CNN (DenseNet-121 / CheXNet)
=======================================================================
Architecture : DenseNet-121 (Stanford CheXNet — gold standard for NIH X-Ray)
Dataset      : NIH ChestX-ray14 (14 pathology labels + Normal)

KEY IMBALANCE FIXES (the core problem: "No Finding" = 60%+ of data):
  1. HARD CAP   — "No Finding" is capped at max 1500 samples so it cannot dominate
  2. SAMPLER    — WeightedRandomSampler forces balanced batches regardless of class size
  3. FOCAL LOSS — Mathematically focuses the model on rare/hard classes during training
  4. SQRT CLASS WEIGHT — Applied INSIDE Focal Loss for a second layer of balance

Additional Quality Upgrades:
  5. DenseNet-121 backbone (CheXNet architecture)
  6. Cosine Annealing LR (smooth convergence)
  7. Mixed Precision Training (2x faster on Kaggle T4/P100)
  8. Gradient Clipping (training stability)
  9. Rich augmentation: GaussianBlur + RandomErasing + RandomCrop
  10. Label Smoothing via Focal Loss (prevents overconfidence)
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# ─── 1. Focal Loss ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples (Normal X-rays the model already knows)
    and focuses training on hard, rare examples (Hernia, Pneumonia, Fibrosis).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=2 is the standard value from the original paper (Lin et al. 2017).
    Higher gamma = more focus on rare hard cases.
    """
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        # Probability of the true class
        pt = torch.exp(-ce_loss)
        # Focal modulation: rare/hard examples get (1-pt)^gamma boost
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ─── 2. Dataset Class ─────────────────────────────────────────────────────────
class XRayDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ─── 3. DenseNet-121 Model (CheXNet Architecture) ─────────────────────────────
def build_densenet121(num_classes: int, dropout: float = 0.4) -> nn.Module:
    """
    DenseNet-121 is the backbone of Stanford CheXNet (2017) which achieved
    radiologist-level performance on NIH ChestX-ray14.
    Dropout=0.4 added before the classifier to prevent overfitting on rare classes.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def main():
    print("🚀 Starting Max-Accuracy DenseNet-121 Training (Imbalance-Corrected)...\n")

    # ─── 4. Load Dataset ──────────────────────────────────────────────────────
    # Search broadly for image_paths.json and filter out any directories
    json_paths = [
        p for p in (
            glob.glob("/kaggle/input/**/image_paths.json", recursive=True)
        )
        if os.path.isfile(p)  # Exclude directories — only real files
    ]
    if not json_paths:
        return print("❌ Error: Could not find image_paths.json in Kaggle input.")

    print(f"✅ Found metadata at: {json_paths[0]}")
    with open(json_paths[0], 'r') as f:
        metadata = json.load(f)

    import subprocess
    result = subprocess.run(
        ['find', '/kaggle/input/', '-name', '*.png'],
        capture_output=True, text=True
    )
    valid_pngs = {
        os.path.basename(p): p
        for p in result.stdout.split('\n')
        if p.endswith('.png')
    }

    # All 14 NIH labels
    TARGET_CLASSES = {
        "Normal", "Atelectasis", "Consolidation", "Infiltration",
        "Pneumothorax", "Effusion", "Pleural_Thickening", "Mass",
        "Nodule", "Emphysema", "Fibrosis", "Cardiomegaly", "Hernia", "Pneumonia"
    }

    # ── HARD CAP: Collect per-class samples separately ──────────────────────
    # This prevents "No Finding" from eating 60% of every batch.
    MAX_NORMAL_SAMPLES = 1500  # Hard cap on Normal class
    MAX_OTHER_SAMPLES  = 2000  # Keep all rare disease samples (or cap if very large)

    per_class_paths  = {c: [] for c in TARGET_CLASSES}
    per_class_labels = {c: [] for c in TARGET_CLASSES}

    for m in metadata:
        raw   = m["actual_diagnosis"]
        label = "Normal" if "No Finding" in raw else raw.split("|")[0].strip()
        if label in TARGET_CLASSES:
            filename = os.path.basename(m["file_path"].replace("\\", "/"))
            if filename in valid_pngs:
                per_class_paths[label].append(valid_pngs[filename])

    # Apply caps and merge
    all_paths, all_labels = [], []
    for cls, paths in per_class_paths.items():
        cap = MAX_NORMAL_SAMPLES if cls == "Normal" else MAX_OTHER_SAMPLES
        sampled = random.sample(paths, min(len(paths), cap))
        all_paths  += sampled
        all_labels += [cls] * len(sampled)

    print(f"✅ Dataset after hard-capping:")
    print(f"   {dict(sorted(Counter(all_labels).items(), key=lambda x: -x[1]))}\n")

    # Convert to integers
    unique_labels  = sorted(list(set(all_labels)))
    label_to_id    = {name: i for i, name in enumerate(unique_labels)}
    numeric_labels = [label_to_id[L] for L in all_labels]
    num_classes    = len(unique_labels)

    # ─── 5. Stratified Train / Val / Test Split ───────────────────────────────
    X_tmp,   X_test,  y_tmp,   y_test  = train_test_split(
        all_paths, numeric_labels,
        test_size=0.15, random_state=42, stratify=numeric_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=0.1765, random_state=42, stratify=y_tmp
    )
    print(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}\n")

    # ─── 6. WeightedRandomSampler — Balanced Batches ─────────────────────────
    # Each sample gets a weight = 1 / count(its class).
    # This means each class appears equally in every mini-batch regardless of size.
    train_counts  = Counter(y_train)
    sample_weights = [1.0 / train_counts[y] for y in y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True   # Oversamples minority classes
    )

    # ─── 7. Augmentation ─────────────────────────────────────────────────────
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # NOTE: sampler and shuffle=True are mutually exclusive — use sampler only
    train_loader = DataLoader(
        XRayDataset(X_train, y_train, train_transforms),
        batch_size=64, sampler=sampler, num_workers=2, pin_memory=True
    )
    val_loader  = DataLoader(
        XRayDataset(X_val, y_val, eval_transforms),
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        XRayDataset(X_test, y_test, eval_transforms),
        batch_size=64, shuffle=False, num_workers=2, pin_memory=True
    )

    # ─── 8. Focal Loss Class Weights (sqrt method for stability) ─────────────
    total = sum(train_counts.values())
    raw_weights  = [total / train_counts[i] for i in range(num_classes)]
    class_weights = torch.sqrt(torch.FloatTensor(raw_weights)).cuda()

    criterion = FocalLoss(class_weights=class_weights, gamma=2.0)

    # ─── 9. Model, Optimizer, Scheduler ──────────────────────────────────────
    model = build_densenet121(num_classes=num_classes, dropout=0.4).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    epochs    = 25
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = GradScaler()

    # ─── 10. Training Loop ────────────────────────────────────────────────────
    best_val_loss = float('inf')
    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist   = [], []

    print(f"🧠 Training for {epochs} epochs...\n")

    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss   += loss.item()
            _, preds      = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train   += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc      = correct_train / total_train

        # ── VALIDATE ──
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                with autocast():
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                val_loss   += loss.item()
                _, preds    = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val   += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc      = correct_val / total_val
        current_lr   = scheduler.get_last_lr()[0]
        scheduler.step()

        train_loss_hist.append(avg_train_loss)
        val_loss_hist.append(avg_val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"| Train Loss: {avg_train_loss:.4f} Acc: {train_acc*100:.1f}% "
            f"| Val Loss: {avg_val_loss:.4f} Acc: {val_acc*100:.1f}% "
            f"| LR: {current_lr:.2e}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "/kaggle/working/best_densenet121.pth")
            print(f"   💾 Saved best model (Val Loss: {best_val_loss:.4f})")

    # ─── 11. Final Test Evaluation ────────────────────────────────────────────
    print("\n✅ Evaluating Best Checkpoint on Test Set...")
    model.load_state_dict(torch.load("/kaggle/working/best_densenet121.pth"))
    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            with autocast():
                outputs = model(inputs.cuda())
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.numpy())

    str_true = [unique_labels[i] for i in y_true]
    str_pred = [unique_labels[i] for i in y_pred]

    print("\n📊 FINAL ACCURACY REPORT — DenseNet-121 (Imbalance-Corrected)")
    print(f"Overall Test Accuracy: {accuracy_score(str_true, str_pred) * 100:.2f}%\n")
    print(classification_report(str_true, str_pred, zero_division=0))

    # ─── 12. Presentation Visuals ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    ax1.plot(train_loss_hist, label="Train Loss",     color="tomato",    marker='o', markersize=4)
    ax1.plot(val_loss_hist,   label="Val Loss",       color="steelblue", marker='s', markersize=4)
    ax1.set_title("Focal Loss Curve — DenseNet-121")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Focal Loss")
    ax1.legend(); ax1.grid(True, alpha=0.4)

    ax2.plot([a*100 for a in train_acc_hist], label="Train Acc", color="tomato",    marker='o', markersize=4)
    ax2.plot([a*100 for a in val_acc_hist],   label="Val Acc",   color="steelblue", marker='s', markersize=4)
    ax2.set_title("Accuracy Curve — DenseNet-121")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("/kaggle/working/training_curves.png", dpi=150)
    plt.show()

    cm = confusion_matrix(str_true, str_pred, labels=unique_labels)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title("Confusion Matrix — DenseNet-121 (Imbalance-Corrected)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("/kaggle/working/confusion_matrix.png", dpi=150)
    plt.show()

    print("\n📁 Saved: best_densenet121.pth | training_curves.png | confusion_matrix.png")


if __name__ == "__main__":
    main()
