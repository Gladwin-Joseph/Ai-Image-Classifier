import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from tqdm import tqdm
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

print(f"\n{'='*70}")
print("PHASE 2: CUSTOM CNN (FAST VERSION - 30 MINS)")
print("AI vs Real Image Detection")
print(f"{'='*70}\n")

#DEVICE SETUP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.empty_cache()

#FAST CNN ARCHITECTURE

class FastCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super(FastCNN, self).__init__()
        
        # LAYER 1: 16 filters (reduced from 32)
        # Input: 128×128×3 → Output: 64×64×16
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # LAYER 2: 32 filters (reduced from 64)
        # Input: 64×64×16 → Output: 32×32×32
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers (smaller)
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
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


#DATASET

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, limit_samples=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        print(f"\n📁 Found {len(self.class_names)} classes: {self.class_names}")
        
        # Load image paths
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = self.root_dir / class_name
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.png')) + \
                         list(class_dir.glob('*.jpeg'))
            
            # SPEED OPTIMIZATION: Limit samples per class
            if limit_samples:
                image_files = image_files[:limit_samples]
            
            print(f"  Class {class_idx} ({class_name}): {len(image_files)} images")
            
            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(class_idx)
        
        print(f"\n✓ Total images loaded: {len(self.images)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        from PIL import Image
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (128, 128), color='white')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


#TRANSFORMS 

def get_transforms():
    """Fast transforms without augmentation"""
    
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform


#TRAINING

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc="Training", disable=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        batch_acc = (predicted == labels).sum().item() / labels.size(0)
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{batch_acc:.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, f1


#VALIDATION

def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    try:
        auc_roc = roc_auc_score(all_labels, all_probs)
    except:
        auc_roc = 0.0
    
    return avg_loss, accuracy, f1, auc_roc, all_labels, all_preds, all_probs


#MAIN

def main(args):
    """Main training function"""
    
    Path("./models_phase2").mkdir(exist_ok=True)
    Path("./logs_phase2").mkdir(exist_ok=True)
    
    #DATA LOADING
    print("\n" + "="*70)
    print("LOADING DATA (FAST MODE)")
    print("="*70)
    
    train_transform, val_transform = get_transforms()
    
    # Load dataset with optional limit
    limit = args.limit_samples if args.limit_samples else None
    full_dataset = ImageDataset("./data", transform=train_transform, limit_samples=limit)
    
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    print(f"\n✓ Train: {train_size} | Val: {val_size} | Test: {test_size}")
    
    # Data loaders with HIGH batch size for speed
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ Batch size: {args.batch_size} (LARGE for speed)")
    
    #MODEL SETUP
    print("\n" + "="*70)
    print("INITIALIZING MODEL (FAST VERSION)")
    print("="*70)
    
    model = FastCNN(num_classes=2, dropout_rate=0.3).to(DEVICE)
    
    print(f"\nArchitecture (OPTIMIZED):")
    print(f"  Layer 1: 16 filters (was 32)")
    print(f"  Layer 2: 32 filters (was 64)")
    print(f"  Image size: 128×128 (was 224×224)")
    print(f"  Dense: 128 → 64 → 2")
    print(f"\nParameters: {model.count_parameters():,} (was 623K)")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"✓ Optimizer: Adam (lr={args.lr})")
    print(f"✓ Scheduler: StepLR")
    
    #TRAINING
    print("\n" + "="*70)
    print(f"STARTING TRAINING ({args.epochs} epochs)")
    print("="*70 + "\n")
    
    best_val_loss = float('inf')
    best_model_path = "./models_phase2/best_custom_cnn_fast.pth"
    no_improve_count = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        val_loss, val_acc, val_f1, val_auc, _, _, _ = validate(
            model, val_loader, criterion, DEVICE
        )
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train: Loss={train_loss:.6f}, Acc={train_acc:.6f}, F1={train_f1:.6f}")
        print(f"  Val:   Loss={val_loss:.6f}, Acc={val_acc:.6f}, F1={val_f1:.6f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved")
        else:
            no_improve_count += 1
        
        scheduler.step()
        
        # Early stopping
        if no_improve_count >= 10:
            print(f"\n⚠ Early stopping (no improvement for 10 epochs)")
            break
    
    #TESTING
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, average='weighted')
    test_rec = recall_score(all_labels, all_preds, average='weighted')
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n" + "="*70)
    print("FINAL TEST RESULTS - PHASE 2 (FAST)")
    print("="*70)
    print(f"Accuracy:  {test_acc:.6f}")
    print(f"Precision: {test_prec:.6f}")
    print(f"Recall:    {test_rec:.6f}")
    print(f"F1-Score:  {test_f1:.6f}")
    print(f"AUC-ROC:   {test_auc:.6f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print("="*70)
    
    results = {
        'phase': 'Phase 2: Custom CNN (FAST)',
        'model_type': 'FastCNN (2 layers)',
        'image_size': '128×128',
        'accuracy': float(test_acc),
        'precision': float(test_prec),
        'recall': float(test_rec),
        'f1_score': float(test_f1),
        'auc_roc': float(test_auc),
        'confusion_matrix': cm.tolist(),
        'model_params': model.count_parameters(),
        'epochs_trained': epoch + 1
    }
    
    with open('./logs_phase2/test_results_fast.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✓ Results saved to logs_phase2/test_results_fast.json")
    print(f"✓ Model saved to {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 2: Fast CNN Training')
    parser.add_argument('--epochs', type=int, default=40, help='Epochs (default: 40)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--limit_samples', type=int, default=None, 
                       help='Limit samples per class (default: use all)')
    
    args = parser.parse_args()
    
    print(f"\nPhase 2 (FAST) Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Limit samples: {args.limit_samples if args.limit_samples else 'None (use all)'}")
    print(f"  Expected time: ~20-30 minutes")
    
    main(args)