import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm
import argparse
import json

print("\n" + "="*70)
print("AI vs REAL IMAGE DETECTION - Testing Suite")
print("FastCNN Model (2-layer, 128x128)")
print("="*70 + "\n")

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

# MODEL ARCHITECTURE
class FastCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(FastCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(32, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=dropout_rate)
        
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# TRANSFORMS

def get_direct_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_cifake_style_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# DATASETS
class CIFAKEDataset(Dataset):
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform if transform else get_direct_transform()
        self.images = []
        self.labels = []
        
        class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        print(f"Loading CIFAKE test data from: {root_dir}")
        print(f"Classes found: {class_names}")
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = self.root_dir / class_name
            count = 0
            for img_path in class_dir.rglob('*'):
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
                    count += 1
            print(f"  {class_name} (label={class_idx}): {count} images")
        
        print(f"Total: {len(self.images)} images\n")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (128, 128), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class CrossDataset(Dataset):
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}
    
    def __init__(self, root_dir, use_cifake_style=True, limit_per_class=None):
        self.root_dir = Path(root_dir)
        self.use_cifake_style = use_cifake_style
        self.images = []
        self.labels = []
        
        self.cifake_transform = get_cifake_style_transform()
        self.direct_transform = get_direct_transform()
        
        fake_keywords = ['fake', 'ai', 'generated', 'synthetic', 'midjourney', 
                        'dalle', 'stable', 'diffusion', 'gan']
        real_keywords = ['real', 'genuine', 'authentic', 'original', 'natural']
        
        print(f"Loading cross-dataset from: {root_dir}")
        
        for subdir in sorted(self.root_dir.iterdir()):
            if not subdir.is_dir():
                continue
            
            name_lower = subdir.name.lower()
            is_fake = any(kw in name_lower for kw in fake_keywords)
            is_real = any(kw in name_lower for kw in real_keywords)
            
            if is_fake:
                label = 0
                print(f"  {subdir.name} -> AI/FAKE (label=0)")
                self._load_from_dir(subdir, label)
            elif is_real:
                label = 1
                print(f"  {subdir.name} -> REAL (label=1)")
                self._load_from_dir(subdir, label)
        
        if limit_per_class:
            self._limit_samples(limit_per_class)
        
        ai_count = sum(1 for l in self.labels if l == 0)
        real_count = sum(1 for l in self.labels if l == 1)
        print(f"Total: {len(self.images)} images (AI: {ai_count}, Real: {real_count})\n")
    
    def _load_from_dir(self, directory, label):
        for img_path in directory.rglob('*'):
            if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                self.images.append(str(img_path))
                self.labels.append(label)
    
    def _limit_samples(self, limit_per_class):
        fake_idx = [i for i, l in enumerate(self.labels) if l == 0]
        real_idx = [i for i, l in enumerate(self.labels) if l == 1]
        
        np.random.seed(42)
        np.random.shuffle(fake_idx)
        np.random.shuffle(real_idx)
        
        selected = fake_idx[:limit_per_class] + real_idx[:limit_per_class]
        self.images = [self.images[i] for i in selected]
        self.labels = [self.labels[i] for i in selected]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.use_cifake_style and (image.size[0] > 256 or image.size[1] > 256):
                image = self.cifake_transform(image)
            else:
                image = self.direct_transform(image)
        except:
            image = torch.zeros(3, 128, 128)
        
        return image, torch.tensor(label, dtype=torch.long)

# TESTING FUNCTIONS

def test_single_image(model, image_path, device):
    model.eval()
    
    print("="*70)
    print("SINGLE IMAGE TEST")
    print("="*70)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    try:
        image = Image.open(image_path).convert('RGB')
        orig_size = image.size
        print(f"Image: {image_path}")
        print(f"Original size: {orig_size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    if orig_size[0] > 256 or orig_size[1] > 256:
        transform = get_cifake_style_transform()
        print(f"Preprocessing: CIFAKE-style (32x32 -> 128x128)")
    else:
        transform = get_direct_transform()
        print(f"Preprocessing: Direct resize (128x128)")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(1).item()
    
    prob_ai = probs[0][0].item() * 100
    prob_real = probs[0][1].item() * 100
    
    print("-"*70)
    print("RESULTS")
    print("-"*70)
    
    if predicted_class == 0:
        print(f"Prediction: AI-GENERATED")
        print(f"Confidence: {prob_ai:.2f}%")
    else:
        print(f"Prediction: REAL")
        print(f"Confidence: {prob_real:.2f}%")
    
    print(f"\nClass Probabilities:")
    print(f"  AI-Generated (class 0): {prob_ai:.2f}%")
    print(f"  Real         (class 1): {prob_real:.2f}%")
    print("="*70)
    
    return {
        'image': image_path,
        'prediction': 'AI-GENERATED' if predicted_class == 0 else 'REAL',
        'confidence': prob_ai if predicted_class == 0 else prob_real,
        'prob_ai': prob_ai,
        'prob_real': prob_real
    }


def test_cifake(model, test_dir, device, batch_size=32):
    print("="*70)
    print("CIFAKE TEST SET EVALUATION")
    print("="*70 + "\n")
    
    dataset = CIFAKEDataset(test_dir, transform=get_direct_transform())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*70)
    print("CIFAKE TEST RESULTS")
    print("="*70)
    print(f"Total Samples: {len(all_labels)}")
    print(f"Correct: {sum(1 for p, l in zip(all_preds, all_labels) if p == l)}")
    print("-"*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print("-"*70)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 AI    REAL")
    print(f"  Actual AI     [{cm[0][0]:5d}  {cm[0][1]:5d}]")
    print(f"  Actual REAL   [{cm[1][0]:5d}  {cm[1][1]:5d}]")
    print("="*70)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels)
    }


def test_cross_dataset(model, cross_data_dir, device, batch_size=32, limit_samples=None, cifake_accuracy=96.18):
    print("="*70)
    print("CROSS-DATASET EVALUATION")
    print("="*70 + "\n")
    
    limit_per_class = limit_samples // 2 if limit_samples else None
    dataset = CrossDataset(cross_data_dir, use_cifake_style=True, limit_per_class=limit_per_class)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    cross_acc_pct = accuracy * 100
    drop = cifake_accuracy - cross_acc_pct
    
    print("\n" + "="*70)
    print("CROSS-DATASET RESULTS")
    print("="*70)
    print(f"Test Dataset: {Path(cross_data_dir).name}")
    print(f"Preprocessing: CIFAKE-style (32x32 -> 128x128)")
    print(f"Total Samples: {len(all_labels)}")
    print("-"*70)
    print(f"Accuracy:  {accuracy:.4f} ({cross_acc_pct:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print("-"*70)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 AI    REAL")
    print(f"  Actual AI     [{cm[0][0]:5d}  {cm[0][1]:5d}]")
    print(f"  Actual REAL   [{cm[1][0]:5d}  {cm[1][1]:5d}]")
    print("-"*70)
    print("\nGENERALIZATION ANALYSIS")
    print(f"CIFAKE Accuracy:        {cifake_accuracy:.2f}%")
    print(f"Cross-Dataset Accuracy: {cross_acc_pct:.2f}%")
    print(f"Performance Drop:       {drop:.2f} percentage points")
    print("="*70)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc_roc),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels),
        'cifake_baseline': cifake_accuracy,
        'performance_drop': float(drop)
    }


def find_optimal_threshold(model, cross_data_dir, device, limit_samples=None):
    """
    Find optimal decision threshold with comprehensive metrics:
    - PR-AUC (Precision-Recall Area Under Curve)
    - Recall at fixed precision (80%)
    - Calibration error (Expected Calibration Error)
    - Per-threshold accuracy breakdown
    """
    
    print("="*70)
    print("THRESHOLD OPTIMIZATION WITH COMPREHENSIVE METRICS")
    print("="*70 + "\n")
    
    limit_per_class = limit_samples // 2 if limit_samples else None
    dataset = CrossDataset(cross_data_dir, use_cifake_style=True, limit_per_class=limit_per_class)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model.eval()
    all_probs_ai = []
    all_probs_real = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Getting predictions"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs_ai.extend(probs[:, 0].cpu().numpy())
            all_probs_real.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_probs_ai = np.array(all_probs_ai)
    all_probs_real = np.array(all_probs_real)
    all_labels = np.array(all_labels)
    
    #COMPREHENSIVE METRICS
    
    print("="*70)
    print("COMPREHENSIVE METRICS")
    print("="*70)
    
    # 1. PR-AUC (Precision-Recall AUC) for AI class detection
    pr_auc_ai = average_precision_score(1 - all_labels, all_probs_ai)  
    pr_auc_real = average_precision_score(all_labels, all_probs_real) 
    
    print(f"\n1. PR-AUC (Precision-Recall Area Under Curve)")
    print(f"   PR-AUC (AI detection):   {pr_auc_ai:.4f}")
    print(f"   PR-AUC (Real detection): {pr_auc_real:.4f}")
    print(f"   Mean PR-AUC:             {(pr_auc_ai + pr_auc_real)/2:.4f}")
    
    # 2. Recall at Fixed Precision
    precision_curve, recall_curve, thresholds_pr = precision_recall_curve(1 - all_labels, all_probs_ai)
    
    recall_at_80_precision = 0.0
    for p, r in zip(precision_curve, recall_curve):
        if p >= 0.80:
            recall_at_80_precision = max(recall_at_80_precision, r)
    
    recall_at_90_precision = 0.0
    for p, r in zip(precision_curve, recall_curve):
        if p >= 0.90:
            recall_at_90_precision = max(recall_at_90_precision, r)
    
    print(f"\n2. Recall at Fixed Precision")
    print(f"   Recall at 80% Precision: {recall_at_80_precision:.4f} ({recall_at_80_precision*100:.2f}%)")
    print(f"   Recall at 90% Precision: {recall_at_90_precision:.4f} ({recall_at_90_precision*100:.2f}%)")
    
    # 3. Calibration Error (Expected Calibration Error - ECE)
    n_bins = 10
    try:
        prob_true, prob_pred = calibration_curve(all_labels, all_probs_real, n_bins=n_bins, strategy='uniform')
        ece = np.mean(np.abs(prob_true - prob_pred))
    except:
        ece = 0.0
    
    # Brier Score (lower is better)
    brier = brier_score_loss(all_labels, all_probs_real)
    
    print(f"\n3. Calibration Metrics")
    print(f"   Expected Calibration Error (ECE): {ece:.4f}")
    print(f"   Brier Score:                      {brier:.4f}")
    
    # 4. ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs_real)
    except:
        roc_auc = 0.0
    
    print(f"\n4. ROC-AUC: {roc_auc:.4f}")
    
    #THRESHOLD OPTIMIZATION
    
    print("\n" + "="*70)
    print("THRESHOLD SEARCH")
    print("="*70)
    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*70)
    
    best_acc = 0
    best_threshold = 0.5
    best_metrics = {}
    results = []
    
    for threshold in np.arange(0.10, 0.91, 0.05):
        preds = np.where(all_probs_ai > threshold, 0, 1)
        
        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, average='weighted', zero_division=0)
        rec = recall_score(all_labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)
        
        marker = " <- BEST" if acc > best_acc else ""
        print(f"{threshold:<12.2f} {acc*100:<12.2f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}{marker}")
        
        results.append({
            'threshold': float(threshold),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        })
        
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            best_metrics = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1)
            }
    
    #FINAL RESULTS AT OPTIMAL THRESHOLD
    
    preds = np.where(all_probs_ai > best_threshold, 0, 1)
    cm = confusion_matrix(all_labels, preds)
    
    # Per-class metrics at optimal threshold
    ai_mask = all_labels == 0
    real_mask = all_labels == 1
    ai_acc = accuracy_score(all_labels[ai_mask], preds[ai_mask]) if ai_mask.sum() > 0 else 0
    real_acc = accuracy_score(all_labels[real_mask], preds[real_mask]) if real_mask.sum() > 0 else 0
    
    print("\n" + "="*70)
    print("FINAL RESULTS AT OPTIMAL THRESHOLD")
    print("="*70)
    print(f"\nOptimal Threshold: {best_threshold:.2f}")
    print(f"\nPerformance Metrics:")
    print(f"   Accuracy:  {best_metrics['accuracy']*100:.2f}%")
    print(f"   Precision: {best_metrics['precision']:.4f}")
    print(f"   Recall:    {best_metrics['recall']:.4f}")
    print(f"   F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"\nPer-Class Accuracy:")
    print(f"   AI Detection Accuracy:   {ai_acc*100:.2f}%")
    print(f"   Real Detection Accuracy: {real_acc*100:.2f}%")
    print(f"\nAdvanced Metrics:")
    print(f"   PR-AUC:                  {(pr_auc_ai + pr_auc_real)/2:.4f}")
    print(f"   ROC-AUC:                 {roc_auc:.4f}")
    print(f"   Recall at 80% Precision: {recall_at_80_precision*100:.2f}%")
    print(f"   Calibration Error (ECE): {ece:.4f}")
    print(f"   Brier Score:             {brier:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 AI    REAL")
    print(f"  Actual AI     [{cm[0][0]:5d}  {cm[0][1]:5d}]")
    print(f"  Actual REAL   [{cm[1][0]:5d}  {cm[1][1]:5d}]")
    
    # Impact statement
    total_samples = len(all_labels)
    correct_at_default = sum(1 for p, l in zip((all_probs_ai > 0.5).astype(int), 1-all_labels) if p == l)
    correct_at_optimal = sum(1 for p, l in zip(preds, all_labels) if p == l)
    improvement = correct_at_optimal - correct_at_default
    
    print(f"\nIMPACT ANALYSIS")
    print("-"*70)
    print(f"   Samples correctly classified: {correct_at_optimal}/{total_samples}")
    print(f"   Improvement over default (0.5): +{improvement} samples")
    print(f"   At 80% precision, recall: {recall_at_80_precision*100:.2f}%")
    
    print("="*70)
    
    return {
        'optimal_threshold': float(best_threshold),
        'best_accuracy': float(best_acc),
        'best_metrics': best_metrics,
        'pr_auc_ai': float(pr_auc_ai),
        'pr_auc_real': float(pr_auc_real),
        'pr_auc_mean': float((pr_auc_ai + pr_auc_real)/2),
        'roc_auc': float(roc_auc),
        'recall_at_80_precision': float(recall_at_80_precision),
        'recall_at_90_precision': float(recall_at_90_precision),
        'expected_calibration_error': float(ece),
        'brier_score': float(brier),
        'per_class_accuracy': {
            'ai': float(ai_acc),
            'real': float(real_acc)
        },
        'confusion_matrix': cm.tolist(),
        'all_threshold_results': results
    }

# MAIN

def main(args):
    print("Loading model...")
    model = FastCNN(num_classes=2, dropout_rate=0.3).to(DEVICE)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    print(f"Model loaded from {args.model_path}\n")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if args.mode == 'single':
        if not args.image:
            print("Error: Please provide --image path for single mode")
            return
        
        result = test_single_image(model, args.image, DEVICE)
        if result:
            with open(output_dir / "single_image_result.json", 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\nResult saved to {output_dir}/single_image_result.json")
    
    elif args.mode == 'cifake':
        if not args.test_dir:
            print("Error: Please provide --test_dir for CIFAKE test mode")
            return
        
        result = test_cifake(model, args.test_dir, DEVICE, args.batch_size)
        with open(output_dir / "cifake_test_results.json", 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {output_dir}/cifake_test_results.json")
    
    elif args.mode == 'cross':
        if not args.cross_data_dir:
            print("Error: Please provide --cross_data_dir for cross-dataset mode")
            return
        
        result = test_cross_dataset(
            model, args.cross_data_dir, DEVICE, 
            args.batch_size, args.limit_samples, args.cifake_accuracy
        )
        dataset_name = Path(args.cross_data_dir).name
        with open(output_dir / f"cross_test_{dataset_name}.json", 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {output_dir}/cross_test_{dataset_name}.json")
    
    elif args.mode == 'threshold':
        if not args.cross_data_dir:
            print("Error: Please provide --cross_data_dir for threshold mode")
            return
        
        result = find_optimal_threshold(
            model, args.cross_data_dir, DEVICE, args.limit_samples
        )
        with open(output_dir / "threshold_optimization.json", 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {output_dir}/threshold_optimization.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AI vs Real Image Detection - Testing Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test single image:
    python test.py --mode single --image ./photo.jpg
  
  Test on CIFAKE test set (15%):
    python test.py --mode cifake --test_dir ./data_splits/test
  
  Test on cross-dataset (Midjourney):
    python test.py --mode cross --cross_data_dir ./cross_test_data
  
  Threshold optimization with comprehensive metrics:
    python test.py --mode threshold --cross_data_dir ./cross_test_data
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'cifake', 'cross', 'threshold'],
                        help='Testing mode')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--cross_data_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, 
                        default='./models_phase2/best_custom_cnn_fast.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--limit_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--cifake_accuracy', type=float, default=96.18)
    
    args = parser.parse_args()
    main(args)