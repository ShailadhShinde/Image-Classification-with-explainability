# Image Classification with Explainability

A deep learning project that classifies images of cats, dogs, and horses using transfer learning and provides visual explanations of model predictions through Grad-CAM.

## 📋 Overview

This project demonstrates image classification with explainability using:
- **Transfer Learning** with pre-trained ResNet18
- **Grad-CAM** visualization for model interpretability
- **CIFAR-10 Dataset** (subset of 3 animal classes)

The model achieves **83% test accuracy** and provides visual explanations of what features the network focuses on when making predictions.

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.7+
PyTorch
torchvision
numpy
matplotlib
scikit-learn
```
## 🚀 Installation & Running

### Google Colab (Recommended - Free GPU)

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click `File` → `Open notebook` → `GitHub` tab
3. Enter repository: `https://github.com/ShailadhShinde/Image-Classification-with-explainability`
4. Select `image-classification.ipynb`
5. Enable GPU: `Runtime` → `Change runtime type` → Select `GPU`
6. Run all cells

### Kaggle Notebooks (Free GPU)

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click `New Notebook` → `File` → `Import Notebook`
3. Enter repository: `https://github.com/ShailadhShinde/Image-Classification-with-explainability`
4. Select `image-classification.ipynb`
5. Enable GPU: `Settings` → `Accelerator` → Select `GPU T4 x2`
6. Run all cells
---

## 📊 Dataset

- **Source:** CIFAR-10
- **Classes:** Cat, Dog, Horse (3 classes only)
- **Total Images:** ~5,000 (balanced across classes)
- **Split:** Train (70%), Validation (15%), Test (15%)
- **Preprocessing:** 
  - Resized to 224×224
  - Random flips and rotations
  - Normalized using ImageNet statistics

The dataset uses stratified sampling to maintain equal class distribution across all splits.

---

## 🏗️ Model Architecture

### ResNet18 (Transfer Learning)

- **Base Model:** Pre-trained ResNet18 (ImageNet weights)
- **Training Strategy:** Froze all layers except final classifier
- **Output Layer:** Modified to 3 classes
- **Optimizer:** Adam
- **Loss Function:** Cross-Entropy Loss
- **Epochs:** 15

### Why ResNet18?

ResNet18 was chosen for several key reasons:

- **Strong pre-trained features:** ImageNet weights provide robust feature extraction for animal classification
- **Residual connections:** Skip connections help with training stability and gradient flow
- **Computational efficiency:** Lightweight enough for quick experimentation on limited hardware
- **Proven performance:** Well-documented success on small datasets with transfer learning

**Note on model selection:** Given the small dataset size (~5,000 images), other architectures like MobileNetV2, EfficientNet-B0, or VGG16 would likely produce similar results with minimal metric differences. The choice of ResNet18 was primarily for simplicity and reliability. Future experimentation with different architectures is recommended once the dataset is expanded, as performance differences may become more pronounced with additional training data.

---

## 📈 Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **83%** |

### Per-Class Metrics

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Cat    | 0.84      | 0.72   | 0.78     | 334     |
| Dog    | 0.78      | 0.83   | 0.81     | 334     |
| Horse  | 0.87      | 0.93   | 0.90     | 334     |
| **Macro Avg** | **0.83** | **0.83** | **0.83** | **1002** |

### Key Observations

- **Horses** are classified best (93% recall) due to distinct body shapes and features
- **Cats** have lower recall (72%), often confused with dogs due to similar fur textures and poses
- **Dogs** show balanced performance, with occasional confusion with cats
- The model struggles more with cat-dog similarities than with horses

---

## 🔍 Explainability: Grad-CAM

The project uses **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize which regions of an image the model focuses on when making predictions.

### Findings

- **Correct Predictions:** Model focuses strongly on key animal shapes and features (activation intensities: 0.23-0.46)
- **Incorrect Predictions:** Focus shifts to background or irrelevant regions, causing misclassifications (activation intensities: 0.22-0.30)

Grad-CAM heatmaps are generated for both correct and incorrect predictions to understand model behavior.

#### Sample Predictions
*Examples of correctly classified images*

![Sample Correct Predictions](https://github.com/ShailadhShinde/Image-Classification-with-explainability/blob/main/Correct%20Predictions%20few%20examples.JPG)

*Examples of misclassified images*

![Sample Incorrect Predictions](https://github.com/ShailadhShinde/Image-Classification-with-explainability/blob/main/INCORRECT%20predictions%20few%20examples.JPG)

---

## 🔧 Limitations & Future Improvements

### Current Limitations
Only 5000 images in the dataset, limited augmentation, all layers are froze;hence no fine tuning

### Potential Improvements

- [ ] Increase dataset size 
- [ ] Advanced augmentations  inorder to work with 
- [ ] Test other architectures (ResNet50, EfficientNet, MobileNetV2)
- [ ] Unfreeze more layers for fine-tuning
- [ ] Implement learning rate scheduling
- [ ] Add early stopping mechanism
- [ ] Ensemble multiple pre-trained models
- [ ] Explore additional explainability methods (LIME, SHAP)

---
