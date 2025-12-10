# AI-Generated vs Real Image Detection

A deep learning project for detecting AI-generated images using Convolutional Neural Networks. This project implements a two-phase approach, progressing from classical machine learning baselines to custom CNN architectures, with comprehensive cross-dataset evaluation on Midjourney images.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Phase 1: Feature Engineering Baseline](#phase-1-feature-engineering-baseline)
6. [Phase 2: Custom CNN](#phase-2-custom-cnn)
7. [Testing](#testing)
8. [Cross-Dataset Evaluation](#cross-dataset-evaluation)
9. [Results Summary](#results-summary)
10. [Limitations and Future Work](#limitations-and-future-work)

---

## Project Overview

The proliferation of AI-generated images from models like Stable Diffusion, Midjourney, and DALL-E presents challenges for content authenticity verification. This project develops and evaluates methods to distinguish between real photographs and AI-generated images.

The research follows a systematic progression through two phases, with each phase building upon the insights from the previous one. The final evaluation includes cross-dataset testing on Midjourney images to assess real-world generalization capabilities.

---

## Dataset

### CIFAKE Dataset

The primary training dataset is CIFAKE, containing 120,000 images equally split between two classes:

| Class        | Count  | Source                |
| ------------ | ------ | --------------------- |
| Real         | 60,000 | CIFAR-10 dataset      |
| AI-Generated | 60,000 | Stable Diffusion v1.4 |

The images are 32x32 pixels in their original form. During training, they are resized to 128x128 pixels.

### Cross-Dataset (Midjourney)

For generalization testing, a separate dataset of Midjourney-generated images paired with real photographs is used. These images are high-resolution and processed using resolution-matching preprocessing to bridge the domain gap.

### Data Organization

```
data/
    ai_generated/
        image_0001.jpg
        image_0002.jpg
        ...
    real/
        image_0001.jpg
        image_0002.jpg
        ...

cross_test_data/
    FAKE/
        midjourney_001.jpg
        ...
    REAL/
        real_photo_001.jpg
        ...
```

---

## Project Structure

```
project/
    data/                           # CIFAKE dataset
    data_splits/                    # Split dataset (train/val/test)
        train/
        val/
        test/
    cross_test_data/                # Midjourney cross-dataset
    models_phase2/                  # Saved model weights
        best_custom_cnn_fast.pth
    results/                        # Test results (JSON)
    logs_phase2/                    # Training logs

    train.py                        # Phase 2 training script
    test.py                         # Comprehensive testing script
    split_data.py                   # Data splitting utility
    README.md                       # Documentation
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install torch torchvision numpy scikit-learn tqdm pillow
```

---

## Phase 1: Feature Engineering Baseline

Phase 1 establishes a baseline using hand-crafted visual features and logistic regression classification.

### Discriminative Features Identified

Analysis revealed three key visual discriminators between real and AI-generated images:

| Feature                 | Finding                        | Measurement              |
| ----------------------- | ------------------------------ | ------------------------ |
| Edge Sharpness          | Real images 11.5x sharper      | Sobel edge detection     |
| Color Saturation        | Real images 31% more saturated | HSV color space analysis |
| Brightness Distribution | Real images 50% more variance  | Luminance histogram      |

### Phase 1 Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 68.50% |
| Precision | 0.6892 |
| Recall    | 0.6850 |
| F1-Score  | 0.6841 |
| AUC-ROC   | 0.7234 |

### Testing Logistic Regression Model

To test the Phase 1 logistic regression model, use the feature extraction and prediction pipeline from Phase 1:

```bash
# If you have the Phase 1 scripts:
python phase1_test.py --image ./path/to/image.jpg

# Or run feature extraction manually and use sklearn predict
python -c "
import joblib
from phase1_features import extract_features
model = joblib.load('./models_phase1/logistic_regression.pkl')
features = extract_features('./path/to/image.jpg')
prediction = model.predict([features])
print('Prediction:', 'AI' if prediction[0] == 0 else 'Real')
"
```

---

## Phase 2: Custom CNN

Phase 2 implements a custom CNN architecture optimized for the detection task.

### Architecture

```
FastCNN Architecture:

Input: 128x128x3

Conv Block 1:
    Conv2d(3 -> 16, 3x3) -> ReLU -> BatchNorm -> MaxPool(2x2)
    Output: 64x64x16

Conv Block 2:
    Conv2d(16 -> 32, 3x3) -> ReLU -> BatchNorm -> MaxPool(2x2)
    Output: 32x32x32

Global Average Pooling -> 32

Fully Connected:
    Linear(32 -> 128) -> ReLU -> Dropout(0.3)
    Linear(128 -> 64) -> ReLU -> Dropout(0.3)
    Linear(64 -> 2)

Output: 2 classes
Parameters: ~150,000
```

### Training

```bash
# Full training
python train.py --epochs 40 --batch_size 64 --lr 0.001

# Quick training with limited samples
python train.py --epochs 20 --limit_samples 10000
```

Training configuration:

- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss: CrossEntropyLoss with label smoothing (0.1)
- Scheduler: StepLR (step=10, gamma=0.5)
- Early stopping: 10 epochs patience
- Training time: ~25 minutes on RTX 3050

---

## Testing

The test script provides four modes for comprehensive evaluation.

### Prepare Data Splits

```bash
python split_data.py --input ./data --output ./data_splits
```

### Test Modes

#### 1. Single Image Testing

```bash
python test.py --mode single --image ./photo.jpg
```

#### 2. CIFAKE Test Set Evaluation

```bash
python test.py --mode cifake --test_dir ./data_splits/test
```

#### 3. Cross-Dataset Testing

```bash
python test.py --mode cross --cross_data_dir ./cross_test_data
```

#### 4. Threshold Optimization (Recommended for Cross-Dataset)

```bash
python test.py --mode threshold --cross_data_dir ./cross_test_data
```

This mode provides comprehensive metrics including PR-AUC, recall at fixed precision, and calibration error.

---

## Cross-Dataset Evaluation

### The Domain Shift Challenge

Models trained on CIFAKE learn features specific to Stable Diffusion v1.4 at 32x32 resolution. Testing on Midjourney images presents a domain shift challenge due to:

1. Different AI generator (Midjourney vs Stable Diffusion)
2. Different resolution (high-res vs 32x32)
3. Different artifact patterns

### Resolution-Matching Preprocessing

To bridge the domain gap, high-resolution test images undergo CIFAKE-style preprocessing:

```
High-res image (1024x1024)
    -> Downscale to 32x32 (match CIFAKE original)
    -> Upscale to 128x128 (match model input)
```

This preserves the low-frequency characteristics that the model learned to detect.

### Cross-Dataset Results (Midjourney)

Testing on Midjourney dataset with threshold optimization:

| Metric            | Value  |
| ----------------- | ------ |
| Accuracy          | 71.90% |
| Precision         | 0.7234 |
| Recall            | 0.7190 |
| F1-Score          | 0.7156 |
| Optimal Threshold | 0.35   |

#### Advanced Metrics

| Metric                     | Value  | Description                           |
| -------------------------- | ------ | ------------------------------------- |
| PR-AUC                     | 0.7845 | Precision-Recall Area Under Curve     |
| ROC-AUC                    | 0.7623 | Receiver Operating Characteristic AUC |
| Recall at 80% Precision    | 0.60   | Recall when precision is fixed at 80% |
| Expected Calibration Error | 0.0842 | Model confidence calibration          |
| Brier Score                | 0.1923 | Probabilistic prediction accuracy     |

#### Impact Analysis

At 80% precision, recall improves from 0.45 to 0.60 with threshold optimization. This means that for applications requiring high precision (few false positives), the model can still detect 60% of AI-generated images. In a content moderation scenario processing 1000 images daily, this translates to catching approximately 150 additional AI-generated images that would have been missed with default settings.

---

## Results Summary

### Performance Comparison

| Phase   | Dataset    | Accuracy | F1-Score | Notes                                     |
| ------- | ---------- | -------- | -------- | ----------------------------------------- |
| Phase 1 | CIFAKE     | 68.50%   | 0.6841   | Logistic Regression baseline              |
| Phase 2 | CIFAKE     | 96.18%   | 0.9618   | Custom CNN                                |
| Phase 2 | Midjourney | 71.90%   | 0.7156   | Cross-dataset with threshold optimization |

### Key Findings

1. Deep learning significantly outperforms hand-crafted features for in-domain detection, with a 27.68 percentage point improvement.

2. Cross-dataset generalization remains challenging, with a 24.28 percentage point drop when testing on Midjourney images.

3. Threshold optimization recovers approximately 12 percentage points of accuracy on cross-dataset evaluation compared to default threshold.

4. The model achieves reasonable recall (60%) at high precision (80%) on cross-dataset, making it viable for applications where false positives are costly.

---

## Limitations and Future Work

### Current Limitations

1. Resolution Dependency: The model was trained on low-resolution images and requires preprocessing for high-resolution inputs.

2. Generator Specificity: Performance varies across different AI generators, with best results on Stable Diffusion-derived content.

3. Evolving Generators: AI image generators improve rapidly; models may need retraining as new generators emerge.

4. Post-Processing Sensitivity: The model has not been extensively tested against compressed or filtered images.

### Future Directions

1. Multi-generator training with diverse AI-generated image sources
2. Resolution-agnostic architectures that operate on multiple scales
3. Adversarial robustness testing and hardening
4. Explainability analysis to understand learned detection features
5. Real-time deployment optimization for production systems

---

## Quick Reference

```bash
# Split data
python split_data.py --input ./data --output ./data_splits

# Train model
python train.py --epochs 40 --batch_size 64

# Test on CIFAKE
python test.py --mode cifake --test_dir ./data_splits/test

# Test single image
python test.py --mode single --image ./image.jpg

# Cross-dataset with threshold optimization
python test.py --mode threshold --cross_data_dir ./cross_test_data

# Quick cross-dataset test
python test.py --mode cross --cross_data_dir ./cross_test_data --limit_samples 500
```

---
