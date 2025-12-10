import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, specificity_score, roc_curve, auc
)
from tqdm import tqdm
from pathlib import Path
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

print(f"\n{'='*70}")
print("PHASE 1: CLASSICAL ML BASELINE - LOGISTIC REGRESSION")
print("AI vs Real Image Detection")
print(f"{'='*70}\n")

#CONFIGURATION 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()

#FEATURE EXTRACTION METHODS 

class FeatureExtractor:
    """Extract features from images for classical ML"""
    
    def __init__(self, method='mean_color'):
        self.method = method
    
    def extract_mean_color(self, image):
        """Extract mean RGB values"""
        return image.view(3, -1).mean(dim=1).cpu().numpy()
    
    def extract_color_histogram(self, image):
        """Extract color histogram (16 bins per channel)"""
        image_np = image.cpu().numpy()
        hist = []
        for c in range(3):
            hist.extend(np.histogram(image_np[c], bins=16)[0])
        return np.array(hist) / hist.sum() if hist.sum() > 0 else hist
    
    def extract_edges_histogram(self, image):
        """Extract edge-based features"""
        image_np = image.cpu().numpy()
        # Simple Sobel-like edge detection
        edges = np.zeros(3)
        for c in range(3):
            ch = image_np[c]
            edge = np.abs(np.diff(ch, axis=0)).mean() + np.abs(np.diff(ch, axis=1)).mean()
            edges[c] = edge
        return edges
    
    def extract_combined(self, image):
        """Combine multiple feature types"""
        features = []
        features.extend(self.extract_mean_color(image))  # 3 features
        features.extend(self.extract_color_histogram(image))  # 48 features
        features.extend(self.extract_edges_histogram(image))  # 3 features
        return np.array(features)
    
    def extract(self, image):
        """Main extraction method"""
        if self.method == 'mean_color':
            return self.extract_mean_color(image)
        elif self.method == 'histogram':
            return self.extract_color_histogram(image)
        elif self.method == 'edges':
            return self.extract_edges_histogram(image)
        elif self.method == 'combined':
            return self.extract_combined(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

#DATA LOADING

def get_transforms():
    """Simple transforms for classical ML"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])
    return transform

def load_data(data_dir, batch_size=32):
    """Load and split dataset"""
    
    transform = get_transforms()
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    class_names = dataset.classes
    
    print(f"\nDATASET STATISTICS:")
    print(f"   Total images: {len(dataset)}")
    print(f"   Classes: {class_names}")
    
    # Manual split to maintain consistency
    from torch.utils.data import random_split
    
    total = len(dataset)
    test_size = int(total * 0.15)
    val_size = int(total * 0.15)
    train_size = total - test_size - val_size
    
    print(f"   Train: {train_size} ({train_size/total*100:.1f}%)")
    print(f"   Val: {val_size} ({val_size/total*100:.1f}%)")
    print(f"   Test: {test_size} ({test_size/total*100:.1f}%)")
    
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, class_names

#FEATURE EXTRACTION

def extract_features_from_loader(data_loader, feature_extractor, max_samples=None):
    """Extract features from dataset"""
    
    features_list = []
    labels_list = []
    
    count = 0
    pbar = tqdm(data_loader, desc='Extracting features')
    
    for images, labels in pbar:
        images = images.to(DEVICE)
        
        for i, image in enumerate(images):
            if max_samples and count >= max_samples:
                break
            
            feature = feature_extractor.extract(image)
            features_list.append(feature)
            labels_list.append(labels[i].item())
            count += 1
        
        if max_samples and count >= max_samples:
            break
    
    return np.array(features_list), np.array(labels_list)

# ============ TRAINING & EVALUATION ============

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train logistic regression with validation"""
    
    print(f"\n{'='*70}")
    print("TRAINING LOGISTIC REGRESSION")
    print(f"{'='*70}\n")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    print(f"Feature scaling:")
    print(f"   Training features: {X_train_scaled.shape}")
    print(f"   Validation features: {X_val_scaled.shape}\n")
    
    # Train logistic regression
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, verbose=0, n_jobs=-1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Validation metrics
    val_pred = model.predict(X_val_scaled)
    val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    val_accuracy = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    print(f"✓ Training complete")
    print(f"   Validation Accuracy: {val_accuracy:.4f}")
    print(f"   Validation ROC-AUC: {val_auc:.4f}")
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test, class_names):
    """Comprehensive evaluation on test set"""
    
    print(f"\n{'='*70}")
    print("TEST SET EVALUATION - LOGISTIC REGRESSION BASELINE")
    print(f"{'='*70}\n")
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1], zero_division=0)
    specificity = specificity_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    roc_auc_curve = auc(fpr, tpr)
    
    # Print results
    print(f"{'='*70}")
    print("OVERALL METRICS:")
    print(f"{'='*70}")
    print(f"   Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision:    {precision:.4f}")
    print(f"   Recall:       {recall:.4f}")
    print(f"   Specificity:  {specificity:.4f}")
    print(f"   F1-Score:     {f1:.4f}")
    print(f"   ROC-AUC:      {roc_auc:.4f}")
    
    print(f"\n{'='*70}")
    print("PER-CLASS METRICS:")
    print(f"{'='*70}")
    for i, class_name in enumerate(class_names):
        print(f"\n   {class_name}:")
        print(f"      Precision:   {precision_per_class[i]:.4f}")
        print(f"      Recall:      {recall_per_class[i]:.4f}")
        print(f"      F1-Score:    {f1_per_class[i]:.4f}")
    
    print(f"\n{'='*70}")
    print("CONFUSION MATRIX:")
    print(f"{'='*70}")
    print(f"\n   {'':15} {class_names[0]:15} {class_names[1]:15}")
    for i, actual_class in enumerate(class_names):
        print(f"   {actual_class:15} {cm[i,0]:15} {cm[i,1]:15}")
    
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    print(f"\n   True Negatives:  {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   True Positives:  {tp}")
    
    print(f"\n{'='*70}\n")
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'specificity': float(specificity),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'precision_per_class': {class_names[i]: float(p) for i, p in enumerate(precision_per_class)},
        'recall_per_class': {class_names[i]: float(r) for i, r in enumerate(recall_per_class)},
        'f1_per_class': {class_names[i]: float(f) for i, f in enumerate(f1_per_class)},
        'confusion_matrix': cm.tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_auc_curve': float(roc_auc_curve)
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Phase 1: Logistic Regression Baseline')
    parser.add_argument('--data_dir', default='./data', help='Data directory')
    parser.add_argument('--feature_method', default='combined', 
                       choices=['mean_color', 'histogram', 'edges', 'combined'],
                       help='Feature extraction method')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--output_dir', default='./results_phase1', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    Path('./models_phase1').mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("PHASE 1: CLASSICAL ML BASELINE - LOGISTIC REGRESSION")
    print("="*70)
    print(f"Data Dir: {args.data_dir}")
    print(f"Feature Method: {args.feature_method}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Device: {DEVICE}")
    print("="*70)
    
    # Load data
    train_loader, val_loader, test_loader, class_names = load_data(
        args.data_dir,
        args.batch_size
    )
    
    # Feature extractor
    feature_extractor = FeatureExtractor(method=args.feature_method)
    
    # Extract features
    print(f"\n{'='*70}")
    print("FEATURE EXTRACTION")
    print(f"{'='*70}")
    
    print("\nExtracting training features...")
    X_train, y_train = extract_features_from_loader(train_loader, feature_extractor)
    print(f"✓ Training: {X_train.shape}")
    
    print("\nExtracting validation features...")
    X_val, y_val = extract_features_from_loader(val_loader, feature_extractor)
    print(f"✓ Validation: {X_val.shape}")
    
    print("\nExtracting test features...")
    X_test, y_test = extract_features_from_loader(test_loader, feature_extractor)
    print(f"✓ Test: {X_test.shape}")
    
    # Train model
    model, scaler = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, scaler, X_test, y_test, class_names)
    
    # Save model
    model_path = './models_phase1/logistic_regression.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_method': args.feature_method}, f)
    print(f"✓ Model saved to {model_path}")
    
    # Save metrics
    metrics_path = f'{args.output_dir}/test_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"✓ Metrics saved to {metrics_path}")
    
    # Summary
    print(f"\n{'='*70}")
    print("✓ PHASE 1 COMPLETE - LOGISTIC REGRESSION BASELINE")
    print(f"{'='*70}")
    print(f"\nSUMMARY:")
    print(f"   Model: Logistic Regression")
    print(f"   Features: {X_test.shape[1]} ({args.feature_method})")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"\n   Model saved: {model_path}")
    print(f"   Metrics saved: {metrics_path}")
    print(f"\n{'='*70}\n")
    
    print("NEXT PHASE: Phase 1B - Simple CNN (3-4 layers)")
    print("   Expected improvement: 75-85% accuracy")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()