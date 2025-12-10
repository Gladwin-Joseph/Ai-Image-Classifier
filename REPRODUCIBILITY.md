# Reproducibility Notes

This document provides detailed instructions for reproducing all experiments and results reported in this project.

---

## 1. Environment Specification

### Hardware Used

- GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
- CPU: Intel Core i5/i7 or equivalent
- RAM: 16GB recommended
- Storage: 10GB free space for dataset and models

### Software Versions

```
Python: 3.8.10 or higher
PyTorch: 1.9.0 or higher
torchvision: 0.10.0 or higher
numpy: 1.21.0 or higher
scikit-learn: 0.24.0 or higher
Pillow: 8.0.0 or higher
tqdm: 4.62.0 or higher
```

### Exact Package Installation

```bash
pip install torch==2.0.1 torchvision==0.15.2
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install Pillow==10.0.0
pip install tqdm==4.66.1
```

---

## 2. Random Seeds

All experiments use fixed random seeds for reproducibility:

```python
# In training scripts
import torch
import numpy as np
import random

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The data split uses `torch.Generator().manual_seed(42)` for consistent train/val/test partitions.

---

## 3. Dataset

### CIFAKE Dataset

- Source: Kaggle CIFAKE dataset
- URL: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- Total images: 120,000 (60,000 real, 60,000 AI-generated)
- Original resolution: 32x32 pixels
- Format: JPEG/PNG

### Expected Directory Structure

```
data/
    ai_generated/
        0.jpg
        1.jpg
        ...
        59999.jpg
    real/
        0.jpg
        1.jpg
        ...
        59999.jpg
```

### Data Integrity Check

```bash
# Verify image counts
find ./data/ai_generated -type f -name "*.jpg" | wc -l  # Should be ~60000
find ./data/real -type f -name "*.jpg" | wc -l          # Should be ~60000
```

---

## 4. Phase 1: Logistic Regression

### Training Command

```bash
python phase_1/train.py --feature_method combined
```

### Expected Output

```
Accuracy:  0.6850 (68.50%)
Precision: 0.6892
Recall:    0.6850
F1-Score:  0.6841
AUC-ROC:   0.7234
```

### Model Artifacts

- Location: `./models_phase1/logistic_regression.pkl`
- Contents: Trained model, scaler, feature method name
- File size: ~50KB

### Verification

```bash
python phase_1/test.py --mode cifake --test_dir ./data_splits/test
```

---

## 5. Phase 2: Custom CNN

### Training Command

```bash
python train.py --epochs 40 --batch_size 64 --lr 0.001
```

### Training Hyperparameters

| Parameter       | Value                       |
| --------------- | --------------------------- |
| Epochs          | 40 (with early stopping)    |
| Batch Size      | 64                          |
| Learning Rate   | 0.001                       |
| Weight Decay    | 1e-4                        |
| Dropout Rate    | 0.3                         |
| Label Smoothing | 0.1                         |
| Image Size      | 128x128                     |
| Optimizer       | Adam                        |
| Scheduler       | StepLR (step=10, gamma=0.5) |
| Early Stopping  | 10 epochs patience          |

### Expected Output

```
Accuracy:  0.9618 (96.18%)
Precision: 0.9618
Recall:    0.9618
F1-Score:  0.9618
AUC-ROC:   0.9938
```

### Model Artifacts

- Location: `./models_phase2/best_custom_cnn_fast.pth`
- Architecture: FastCNN (2-layer CNN)
- Parameters: ~150,000
- File size: ~0.6MB

### Verification

```bash
python test.py --mode cifake --test_dir ./data_splits/test
```

---

## 6. Cross-Dataset Evaluation

Dataset Used

Name: MidJourney CIFAKE-Inspired Dataset

Source: Kaggle

URL: https://www.kaggle.com/datasets/mariammarioma/midjourney-cifake-inspired

Description:

A CIFAKE-inspired dataset generated using Midjourney v5/v6

Includes high-quality AI-generated images and real photographs

Designed to mimic the CIFAKE structure but with more complex image semantics

### Dataset Preparation

Organize Midjourney dataset images as:

```
cross_test_data/
    FAKE/       # AI-generated images
    REAL/       # Real photographs
```

### Testing Command

```bash
python test.py --mode threshold --cross_data_dir ./cross_test_data
```

### Expected Results (with threshold optimization)

```
Optimal Threshold: 0.35
Accuracy:  71.90%
Precision: 0.7234
Recall:    0.7190
F1-Score:  0.7156
PR-AUC:    0.7845
ROC-AUC:   0.7623
```

### Preprocessing Details

High-resolution images undergo CIFAKE-style preprocessing:

1. Resize to 32x32 (match CIFAKE original resolution)
2. Resize to 128x128 (match model input size)
3. Normalize with ImageNet mean/std

---

## 7. Complete Reproduction Steps

### Step-by-Step Instructions

```bash
# 1. Clone/setup project
mkdir ai-image-detection
cd ai-image-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install torch torchvision numpy scikit-learn pillow tqdm

# 4. Download CIFAKE dataset from Kaggle
# Extract to ./data folder

# 5. Split data
python split_data.py --input ./data --output ./data_splits

# 6. Train Phase 1
python phase_1/train.py --feature_method combined

# 7. Test Phase 1 on CIFAKE
python phase_1/test.py --mode cifake --test_dir ./data_splits/test

# 8. Train Phase 2
python train.py --epochs 40 --batch_size 64 --lr 0.001

# 9. Test Phase 2 on CIFAKE
python test.py --mode cifake --test_dir ./data_splits/test

# 10. Cross-dataset evaluation (if you have Midjourney data)
python test.py --mode threshold --cross_data_dir ./cross_test_data
```

---

## 8. Result Verification

### Expected Result Files

After running all experiments, you should have:

```
results/
    phase1_cifake_results.json
    phase1_threshold_optimization.json
    cifake_test_results.json
    threshold_optimization.json
```

### Checksum Verification (optional)

```bash
md5sum ./models_phase2/best_custom_cnn_fast.pth
# Your checksum will vary based on exact training run
```

---

## 9. Known Variations

Results may vary slightly due to:

1. GPU architecture differences (cuDNN algorithms)
2. PyTorch version differences
3. Floating-point precision on different hardware

Expected variation: +/- 0.5% accuracy

To minimize variation:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## 10. Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 32
```

### Slow Training

```bash
# Use fewer samples for quick testing
python train.py --limit_samples 10000 --epochs 10
```

### Missing Model File

```
Error: Model not found at ./models_phase2/best_custom_cnn_fast.pth
```

Solution: Run training first with `python train.py`

### Class Mismatch

Ensure folder names are exactly:

- `ai_generated` (not `AI_Generated` or `fake`)
- `real` (not `Real` or `genuine`)

Classes are loaded alphabetically, so `ai_generated` becomes class 0 and `real` becomes class 1.

---
