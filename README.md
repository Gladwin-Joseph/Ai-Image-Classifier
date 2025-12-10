# AI-Generated vs Real Image Detection

A deep learning project for detecting AI-generated images using Convolutional Neural Networks. This project implements a two-phase approach, progressing from classical machine learning baselines to custom CNN architectures, with comprehensive cross-dataset evaluation on Midjourney images.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Dataset Setup](#dataset-setup)
5. [Phase 1: Feature Engineering Baseline](#phase-1-feature-engineering-baseline)
6. [Phase 2: Custom CNN](#phase-2-custom-cnn)
7. [Testing](#testing)
8. [Cross-Dataset Evaluation](#cross-dataset-evaluation)
9. [Results Summary](#results-summary)
10. [Troubleshooting](#troubleshooting)
11. [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

The proliferation of AI-generated images from models like Stable Diffusion, Midjourney, and DALL-E presents challenges for content authenticity verification. This project develops and evaluates methods to distinguish between real photographs and AI-generated images.

The research follows a systematic progression through two phases:

- Phase 1: Classical ML baseline using hand-crafted features (68.50% accuracy)
- Phase 2: Custom CNN achieving 96.18% on CIFAKE, 71.90% on cross-dataset

---

## Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd ai-image-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch torchvision numpy scikit-learn pillow tqdm

# Download CIFAKE dataset from Kaggle and extract to ./data

# Split data
python split_data.py --input ./data --output ./data_splits

# Train CNN
python train.py --epochs 40 --batch_size 64

# Test
python test.py --mode cifake --test_dir ./data_splits/test
```

---

## Installation

### System Requirements

| Component | Minimum | Recommended |
| --------- | ------- | ----------- |
| Python    | 3.8     | 3.10+       |
| GPU VRAM  | 4GB     | 8GB+        |
| RAM       | 8GB     | 16GB        |
| Storage   | 5GB     | 10GB        |

### Step 1: Create Virtual Environment

```bash
# Create new virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies

```bash
# Install PyTorch (with CUDA support)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install numpy scikit-learn pillow tqdm

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 3: Verify GPU Access

```bash
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('No GPU detected, using CPU')
"
```

### Requirements.txt

Create a `requirements.txt` file:

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
scikit-learn>=0.24.0
Pillow>=8.0.0
tqdm>=4.62.0
```

Install with: `pip install -r requirements.txt`

---

## Dataset Setup

### CIFAKE Dataset

1. Download from Kaggle: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

2. Extract to the project directory:

```
project/
    data/
        ai_generated/
            0.jpg
            1.jpg
            ...
        real/
            0.jpg
            1.jpg
            ...
```

3. Split the data:

```bash
python split_data.py --input ./data --output ./data_splits
```

This creates:

```
data_splits/
    train/          # 70% of data
        ai_generated/
        real/
    val/            # 15% of data
        ai_generated/
        real/
    test/           # 15% of data
        ai_generated/
        real/
```

### Cross-Dataset

For Midjourney dataset:

URL: https://www.kaggle.com/datasets/mariammarioma/midjourney-cifake-inspired

Description:

A CIFAKE-inspired dataset generated using Midjourney v5/v6

Includes high-quality AI-generated images and real photographs

Designed to mimic the CIFAKE structure but with more complex image semantics

```
cross_test_data/
    FAKE/           # AI-generated images
        image1.jpg
        ...
    REAL/           # Real photographs
        image1.jpg
        ...
```

---

## Project Structure

```
project/
    data/                           # CIFAKE dataset (download separately)
    data_splits/                    # Split dataset (generated)
        train/
        val/
        test/
    cross_test_data/                # Cross-dataset for generalization testing
    models_phase1/                 # Phase 1 model weights
        logistic_regression.pkl
    models_phase2/                  # Phase 2 model weights
        best_custom_cnn_fast.pth
    results/                        # Test results (JSON)
    logs_phase2/                    # Training logs
    phase_1/
        results/
            json results
        train_phase1.py                # Phase 1 training
        test_phase1.py                 # Phase 1 testing
    train.py                        # Phase 2 training
    test.py                         # Phase 2 testing
    split_data.py                   # Data splitting utility
    README.md                       # This file
    REPRODUCIBILITY.md              # Detailed reproduction instructions
```

---

## Phase 1: Feature Engineering Baseline

Phase 1 establishes a baseline using hand-crafted visual features and logistic regression.

### Discriminative Features

| Feature                 | Finding                        | Measurement          |
| ----------------------- | ------------------------------ | -------------------- |
| Edge Sharpness          | Real images 11.5x sharper      | Sobel edge detection |
| Color Saturation        | Real images 31% more saturated | HSV color space      |
| Brightness Distribution | Real images 50% more variance  | Luminance histogram  |

### Training

```bash
python phase_1/train.py --feature_method combined
```

### Testing

```bash
# Test on CIFAKE
python phase_1/test.py --mode cifake --test_dir ./data_splits/test

# Test single image
python phase_1/test.py --mode single --image ./photo.jpg --test_dir ./data_splits/test

# Cross-dataset with threshold optimization
python phase_1/test.py --mode threshold --test_dir ./data_splits/test --cross_data_dir ./cross_test_data
```

### Phase 1 Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 68.50% |
| Precision | 0.6892 |
| Recall    | 0.6850 |
| F1-Score  | 0.6841 |
| AUC-ROC   | 0.7234 |

---

## Phase 2: Custom CNN

Phase 2 implements a custom CNN architecture optimized for the detection task.

### Architecture

```
FastCNN (2-layer CNN)
Input: 128x128x3

Conv Block 1: Conv2d(3->16) -> ReLU -> BatchNorm -> MaxPool
Conv Block 2: Conv2d(16->32) -> ReLU -> BatchNorm -> MaxPool
Global Average Pooling
FC: 32 -> 128 -> 64 -> 2

Parameters: ~150,000
```

### Training

```bash
# Full training (recommended)
python train.py --epochs 40 --batch_size 64 --lr 0.001

# Quick training (for testing)
python train.py --epochs 10 --batch_size 64 --limit_samples 10000
```

### Training Configuration

| Parameter       | Value                             |
| --------------- | --------------------------------- |
| Optimizer       | Adam                              |
| Learning Rate   | 0.001                             |
| Weight Decay    | 1e-4                              |
| Batch Size      | 64                                |
| Label Smoothing | 0.1                               |
| Early Stopping  | 10 epochs                         |
| Training Time   | ~15 minutes(per epoch) (RTX 3050) |

### Phase 2 Results (CIFAKE)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 96.18% |
| Precision | 0.9618 |
| Recall    | 0.9618 |
| F1-Score  | 0.9618 |
| AUC-ROC   | 0.9938 |

---

## Testing

### Test Modes

| Mode      | Command                                               | Description            |
| --------- | ----------------------------------------------------- | ---------------------- |
| single    | `--mode single --image ./photo.jpg`                   | Test one image         |
| cifake    | `--mode cifake --test_dir ./data_splits/test`         | Test on CIFAKE 15%     |
| cross     | `--mode cross --cross_data_dir ./cross_test_data`     | Cross-dataset test     |
| threshold | `--mode threshold --cross_data_dir ./cross_test_data` | Find optimal threshold |

### Examples

```bash
# Test single image
python test.py --mode single --image ./photo.jpg

# Test on CIFAKE test set
python test.py --mode cifake --test_dir ./data_splits/test

# Cross-dataset with full metrics
python test.py --mode threshold --cross_data_dir ./cross_test_data

# Quick cross-dataset test
python test.py --mode cross --cross_data_dir ./cross_test_data --limit_samples 500
```

---

## Cross-Dataset Evaluation

### The Challenge

Models trained on CIFAKE (32x32, Stable Diffusion) face domain shift when tested on Midjourney images (1024x1024+, different generator).

### Solution: CIFAKE-Style Preprocessing

```
High-res image (1024x1024)
    -> Downscale to 32x32 (match CIFAKE)
    -> Upscale to 128x128 (match model input)
```

### Cross-Dataset Results (Midjourney)

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 71.90% |
| Precision         | 0.7234 |
| Recall            | 0.7190 |
| F1-Score          | 0.7156 |
| Optimal Threshold | 0.35   |

### Advanced Metrics

| Metric                     | Value  | Description                    |
| -------------------------- | ------ | ------------------------------ |
| PR-AUC                     | 0.7845 | Precision-Recall AUC           |
| ROC-AUC                    | 0.7623 | ROC AUC                        |
| Recall at 80% Precision    | 0.60   | High-confidence detection rate |
| Expected Calibration Error | 0.0842 | Probability calibration        |
| Brier Score                | 0.1923 | Probabilistic accuracy         |

---

## Results Summary

| Phase   | Dataset    | Accuracy | F1-Score | Notes               |
| ------- | ---------- | -------- | -------- | ------------------- |
| Phase 1 | CIFAKE     | 68.50%   | 0.6841   | Logistic Regression |
| Phase 2 | CIFAKE     | 96.18%   | 0.9618   | Custom CNN          |
| Phase 2 | Midjourney | 71.90%   | 0.7156   | Cross-dataset       |

### Key Findings

1. Deep learning improves accuracy by 27.68 percentage points over classical ML
2. Cross-dataset generalization drops 24.28 points due to domain shift
3. Threshold optimization recovers ~12 percentage points on cross-dataset
4. 60% recall achievable at 80% precision for high-confidence applications

---

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py --batch_size 32

# Or limit samples
python train.py --limit_samples 50000
```

### Model Not Found

```
Error: Model not found at ./models_phase2/best_custom_cnn_fast.pth
```

Solution: Train the model first with `python train.py`

### Wrong Predictions (All Same Class)

Check class folder names are exactly `ai_generated` and `real` (lowercase). Classes are loaded alphabetically.

### Slow Training on CPU

```bash
# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Import Errors

```bash
# Reinstall all dependencies
pip install --upgrade torch torchvision numpy scikit-learn pillow tqdm
```

---

## Limitations and Future Work

### Current Limitations

1. Resolution Dependency: Requires preprocessing for high-res images
2. Generator Specificity: Best on Stable Diffusion content
3. Evolving Generators: May need retraining for new generators
4. Post-Processing: Not tested against compression/filtering

### Future Directions

1. Multi-generator training
2. Resolution-agnostic architectures
3. Adversarial robustness
4. Explainability analysis
5. Real-time deployment

---

## Command Reference

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install torch torchvision numpy scikit-learn pillow tqdm

# Data preparation
python split_data.py --input ./data --output ./data_splits

# Phase 1
python phase_1/train.py --feature_method combined
python phase_1/test.py --mode cifake --test_dir ./data_splits/test

# Phase 2
python train.py --epochs 40 --batch_size 64
python test.py --mode cifake --test_dir ./data_splits/test

# Cross-dataset
python test.py --mode threshold --cross_data_dir ./cross_test_data

# Single image
python test.py --mode single --image ./image.jpg
```

---

## Files Reference

| File                       | Description                                 |
| -------------------------- | ------------------------------------------- |
| train.py                   | Phase 2 CNN training                        |
| test.py                    | Phase 2 CNN testing (all modes)             |
| phase_1/train.py           | Phase 1 logistic regression training        |
| phase_1/test.py            | Phase 1 testing (all modes)                 |
| phase_1/split_data.py      | Data splitting utility                      |
| phase_1/check_data.py      | Check if dataset is proper                  |
| phase_1/reorganize_data.py | Reorganize into two subfolders(ai and real) |
| REPRODUCIBILITY.md         | Detailed reproduction instructions          |

---
