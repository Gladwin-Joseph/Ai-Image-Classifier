import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, precision_recall_curve,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("PHASE 1: LOGISTIC REGRESSION - Testing Suite")
print("Features + Classical ML")
print("="*70 + "\n")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}\n")



# FEATURE EXTRACTOR 

class FeatureExtractor:
    """Extract features from images - MUST BE SAME AS TRAINING"""
    
    def __init__(self, method='combined'):
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
        return np.array(hist) / np.sum(hist) if np.sum(hist) > 0 else np.array(hist)
    
    def extract_edges_histogram(self, image):
        """Extract edge-based features"""
        image_np = image.cpu().numpy()
        edges = np.zeros(3)
        for c in range(3):
            ch = image_np[c]
            edge = np.abs(np.diff(ch, axis=0)).mean() + np.abs(np.diff(ch, axis=1)).mean()
            edges[c] = edge
        return edges
    
    def extract_combined(self, image):
        """Combine multiple feature types"""
        features = []
        features.extend(self.extract_mean_color(image))
        features.extend(self.extract_color_histogram(image))
        features.extend(self.extract_edges_histogram(image))
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

# TRANSFORMS

def get_transform():
    """Standard transform for images"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_cifake_style_transform():
    """Transform for high-res images (downscale to match CIFAKE characteristics)"""
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# CROSS-DATASET LOADER

class CrossDatasetLoader:
    """Load cross-dataset images with proper preprocessing"""
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.gif'}
    
    def __init__(self, root_dir, use_cifake_style=True, limit_per_class=None):
        self.root_dir = Path(root_dir)
        self.use_cifake_style = use_cifake_style
        self.images = []
        self.labels = []
        
        self.cifake_transform = get_cifake_style_transform()
        self.direct_transform = get_transform()
        
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
    
    def get_image_tensor(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.use_cifake_style and (image.size[0] > 256 or image.size[1] > 256):
                return self.cifake_transform(image)
            else:
                return self.direct_transform(image)
        except:
            return torch.zeros(3, 224, 224)
    
    def __len__(self):
        return len(self.images)

# TESTING FUNCTIONS

def test_single_image(image_path, model, scaler, feature_extractor, class_names):
    """Test on a single image"""
    
    print("="*70)
    print("SINGLE IMAGE TEST")
    print("="*70)
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None
    
    try:
        pil_image = Image.open(image_path).convert('RGB')
        orig_size = pil_image.size
        print(f"Image: {image_path}")
        print(f"Original size: {orig_size}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Choose transform based on size
    if orig_size[0] > 256 or orig_size[1] > 256:
        transform = get_cifake_style_transform()
        print("Preprocessing: CIFAKE-style (32x32 -> 224x224)")
    else:
        transform = get_transform()
        print("Preprocessing: Direct resize (224x224)")
    
    image_tensor = transform(pil_image).to(DEVICE)
    
    # Extract features
    feature = feature_extractor.extract(image_tensor)
    feature_scaled = scaler.transform(feature.reshape(1, -1))
    
    # Predict
    y_pred = model.predict(feature_scaled)[0]
    y_pred_proba = model.predict_proba(feature_scaled)[0]
    
    prob_ai = y_pred_proba[0] * 100
    prob_real = y_pred_proba[1] * 100
    
    print("-"*70)
    print("RESULTS")
    print("-"*70)
    
    if y_pred == 0:
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
        'prediction': 'AI-GENERATED' if y_pred == 0 else 'REAL',
        'confidence': prob_ai if y_pred == 0 else prob_real,
        'prob_ai': prob_ai,
        'prob_real': prob_real
    }


def test_cifake(test_dir, model, scaler, feature_extractor, batch_size=32):
    """Test on CIFAKE test set (15% held out during training)"""
    
    print("="*70)
    print("CIFAKE TEST SET EVALUATION (Phase 1)")
    print("="*70 + "\n")
    
    transform = get_transform()
    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    class_names = dataset.classes
    
    print(f"Loading from: {test_dir}")
    print(f"Classes: {class_names}")
    print(f"Total images: {len(dataset)}\n")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for images, labels in tqdm(loader, desc="Testing"):
        images = images.to(DEVICE)
        
        for i, image in enumerate(images):
            feature = feature_extractor.extract(image)
            feature_scaled = scaler.transform(feature.reshape(1, -1))
            
            y_pred = model.predict(feature_scaled)[0]
            y_pred_proba = model.predict_proba(feature_scaled)[0]
            
            all_preds.append(y_pred)
            all_labels.append(labels[i].item())
            all_probs.append(y_pred_proba)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        roc_auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*70)
    print("CIFAKE TEST RESULTS (Phase 1)")
    print("="*70)
    print(f"Total Samples: {len(all_labels)}")
    print(f"Correct: {sum(1 for p, l in zip(all_preds, all_labels) if p == l)}")
    print("-"*70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {roc_auc:.4f}")
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
        'auc_roc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels)
    }


def test_cross_dataset(cross_data_dir, model, scaler, feature_extractor, 
                       limit_samples=None, cifake_accuracy=68.50):
    """Test on cross-dataset (Midjourney, DALL-E, etc.)"""
    
    print("="*70)
    print("CROSS-DATASET EVALUATION (Phase 1)")
    print("="*70 + "\n")
    
    limit_per_class = limit_samples // 2 if limit_samples else None
    dataset = CrossDatasetLoader(cross_data_dir, use_cifake_style=True, limit_per_class=limit_per_class)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for idx in tqdm(range(len(dataset)), desc="Testing"):
        image_tensor = dataset.get_image_tensor(idx).to(DEVICE)
        label = dataset.labels[idx]
        
        feature = feature_extractor.extract(image_tensor)
        feature_scaled = scaler.transform(feature.reshape(1, -1))
        
        y_pred = model.predict(feature_scaled)[0]
        y_pred_proba = model.predict_proba(feature_scaled)[0]
        
        all_preds.append(y_pred)
        all_labels.append(label)
        all_probs.append(y_pred_proba)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        roc_auc = 0.0
    
    cm = confusion_matrix(all_labels, all_preds)
    cross_acc_pct = accuracy * 100
    drop = cifake_accuracy - cross_acc_pct
    
    print("\n" + "="*70)
    print("CROSS-DATASET RESULTS (Phase 1)")
    print("="*70)
    print(f"Test Dataset: {Path(cross_data_dir).name}")
    print(f"Preprocessing: CIFAKE-style (32x32 -> 224x224)")
    print(f"Total Samples: {len(all_labels)}")
    print("-"*70)
    print(f"Accuracy:  {accuracy:.4f} ({cross_acc_pct:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {roc_auc:.4f}")
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
        'auc_roc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_labels),
        'cifake_baseline': cifake_accuracy,
        'performance_drop': float(drop)
    }


def find_optimal_threshold(cross_data_dir, model, scaler, feature_extractor, limit_samples=None):
    """Find optimal decision threshold with comprehensive metrics"""
    
    print("="*70)
    print("THRESHOLD OPTIMIZATION (Phase 1)")
    print("="*70 + "\n")
    
    limit_per_class = limit_samples // 2 if limit_samples else None
    dataset = CrossDatasetLoader(cross_data_dir, use_cifake_style=True, limit_per_class=limit_per_class)
    
    all_probs_ai = []
    all_probs_real = []
    all_labels = []
    
    for idx in tqdm(range(len(dataset)), desc="Getting predictions"):
        image_tensor = dataset.get_image_tensor(idx).to(DEVICE)
        label = dataset.labels[idx]
        
        feature = feature_extractor.extract(image_tensor)
        feature_scaled = scaler.transform(feature.reshape(1, -1))
        
        y_pred_proba = model.predict_proba(feature_scaled)[0]
        
        all_probs_ai.append(y_pred_proba[0])
        all_probs_real.append(y_pred_proba[1])
        all_labels.append(label)
    
    all_probs_ai = np.array(all_probs_ai)
    all_probs_real = np.array(all_probs_real)
    all_labels = np.array(all_labels)
    
    # Comprehensive Metrics
    print("="*70)
    print("COMPREHENSIVE METRICS")
    print("="*70)
    
    # 1. PR-AUC
    pr_auc_ai = average_precision_score(1 - all_labels, all_probs_ai)
    pr_auc_real = average_precision_score(all_labels, all_probs_real)
    
    print(f"\n1. PR-AUC (Precision-Recall Area Under Curve)")
    print(f"   PR-AUC (AI detection):   {pr_auc_ai:.4f}")
    print(f"   PR-AUC (Real detection): {pr_auc_real:.4f}")
    print(f"   Mean PR-AUC:             {(pr_auc_ai + pr_auc_real)/2:.4f}")
    
    # 2. Recall at Fixed Precision
    precision_curve, recall_curve, _ = precision_recall_curve(1 - all_labels, all_probs_ai)
    
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
    
    # 3. Calibration Error
    n_bins = 10
    try:
        prob_true, prob_pred = calibration_curve(all_labels, all_probs_real, n_bins=n_bins, strategy='uniform')
        ece = np.mean(np.abs(prob_true - prob_pred))
    except:
        ece = 0.0
    
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
    
    # Threshold Search
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
            best_metrics = {'accuracy': float(acc), 'precision': float(prec), 
                          'recall': float(rec), 'f1_score': float(f1)}
    
    # Final Results
    preds = np.where(all_probs_ai > best_threshold, 0, 1)
    cm = confusion_matrix(all_labels, preds)
    
    ai_mask = all_labels == 0
    real_mask = all_labels == 1
    ai_acc = accuracy_score(all_labels[ai_mask], preds[ai_mask]) if ai_mask.sum() > 0 else 0
    real_acc = accuracy_score(all_labels[real_mask], preds[real_mask]) if real_mask.sum() > 0 else 0
    
    print("\n" + "="*70)
    print("FINAL RESULTS AT OPTIMAL THRESHOLD (Phase 1)")
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
        'per_class_accuracy': {'ai': float(ai_acc), 'real': float(real_acc)},
        'confusion_matrix': cm.tolist(),
        'all_threshold_results': results
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Logistic Regression Testing Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Test single image:
    python test_phase1.py --mode single --image ./photo.jpg --test_dir ./data_splits/test
  
  Test on CIFAKE test set (15%):
    python test_phase1.py --mode cifake --test_dir ./data_splits/test
  
  Test on cross-dataset (Midjourney):
    python test_phase1.py --mode cross --test_dir ./data_splits/test --cross_data_dir ./cross_test_data
  
  Threshold optimization:
    python test_phase1.py --mode threshold --test_dir ./data_splits/test --cross_data_dir ./cross_test_data
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'cifake', 'cross', 'threshold'],
                        help='Testing mode')
    parser.add_argument('--image', type=str, default=None,
                        help='Image path (for single mode)')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='CIFAKE test directory (needed for class names)')
    parser.add_argument('--cross_data_dir', type=str, default=None,
                        help='Cross-dataset directory')
    parser.add_argument('--model_path', type=str, 
                        default='../models_phase1/logistic_regression.pkl',
                        help='Path to saved model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--limit_samples', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--cifake_accuracy', type=float, default=68.50,
                        help='CIFAKE accuracy for comparison')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
    
    with open(args.model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        scaler = data['scaler']
        feature_method = data.get('feature_method', 'combined')
    
    print(f"Model loaded from {args.model_path}")
    print(f"Feature method: {feature_method}\n")
    
    feature_extractor = FeatureExtractor(method=feature_method)
    
    # Get class names
    dataset = datasets.ImageFolder(root=args.test_dir)
    class_names = dataset.classes
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Execute based on mode
    if args.mode == 'single':
        if not args.image:
            print("Error: Please provide --image for single mode")
            return
        
        result = test_single_image(args.image, model, scaler, feature_extractor, class_names)
        if result:
            with open(output_dir / "phase1_single_result.json", 'w') as f:
                json.dump(result, f, indent=4)
            print(f"\nResult saved to {output_dir}/phase1_single_result.json")
    
    elif args.mode == 'cifake':
        result = test_cifake(args.test_dir, model, scaler, feature_extractor, args.batch_size)
        with open(output_dir / "phase1_cifake_results.json", 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {output_dir}/phase1_cifake_results.json")
    
    elif args.mode == 'cross':
        if not args.cross_data_dir:
            print("Error: Please provide --cross_data_dir for cross mode")
            return
        
        result = test_cross_dataset(args.cross_data_dir, model, scaler, feature_extractor,
                                   args.limit_samples, args.cifake_accuracy)
        dataset_name = Path(args.cross_data_dir).name
        with open(output_dir / f"phase1_cross_{dataset_name}.json", 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {output_dir}/phase1_cross_{dataset_name}.json")
    
    elif args.mode == 'threshold':
        if not args.cross_data_dir:
            print("Error: Please provide --cross_data_dir for threshold mode")
            return
        
        result = find_optimal_threshold(args.cross_data_dir, model, scaler, 
                                       feature_extractor, args.limit_samples)
        with open(output_dir / "phase1_threshold_optimization.json", 'w') as f:
            json.dump(result, f, indent=4)
        print(f"\nResults saved to {output_dir}/phase1_threshold_optimization.json")


if __name__ == '__main__':
    main()