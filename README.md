# Brain Tumor Classification with ResNet50
Deep Learning-based Brain Tumor Classification using ResNet50 and Otsu Thresholding


## 🧠 Overview

This project implements a **deep learning-based brain tumor classification system** using **ResNet50** architecture with transfer learning. The model classifies brain MRI images into 4 categories:

- 🔴 **Glioma**
- 🟠 **Meningioma**
- 🟢 **No Tumor**
- 🔵 **Pituitary**

The system uses **Otsu's thresholding** for automatic image segmentation and preprocessing, followed by ResNet50-based classification with data augmentation.

## ✨ Features

- **Transfer Learning with ResNet50 (ImageNet weights)**
- **Otsu Edge Detection for automatic tumor region extraction**
- **Data Augmentation (rotation, zoom, flip, shift)**
- **Multi-Class ROC Curve with micro/macro averaging**
- **Confusion Matrix (raw counts + normalized)**
- **Classification Report with precision, recall, F1-score**
- **Training History Visualization (loss & accuracy curves)**
- **Prediction Samples with confidence scores**
- **Model Checkpointing & Learning Rate Reduction**


## 📊 Dataset

The dataset should be organized as follows:
```kotlin
data/
├── training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── testing/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── cropped-otsu/
    ├── training/
    └── testing/
```

**Recommended Datasets**:
- [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Brain Tumor Classification (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

## 📁 Project Structure

``` bash
brain-tumor-resnet50/
├── data/
│   ├── training/               # Original training images
│   ├── testing/                # Original testing images
│   └── cropped-otsu/           # Preprocessed images
├── models/
│   ├── model_res50.h5.keras    # Final trained model
│   └── .mdl_wts.hdf5.keras     # Best model checkpoint
├── outputs/
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve_multiclass.png
│   └── prediction_samples.png
├── otsu-edge-detection.ipynb   # Otsu edge detection preprocessing
├── brain-tumor.ipynb           # Main training script
└── README.md                   # This file

```


## 🏗️ Model Architecture
```scss
Input (200×200×3)
    ↓
ResNet50 (ImageNet weights, trainable)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.4)
    ↓
Dense (4, softmax)
    ↓
Output (4 classes)
```
Key Components:

- **Base Model**: ResNet50 (pre-trained on ImageNet)
- **Pooling**: Global Average Pooling
- **Regularization**: Dropout (40%)
- **Activation**: Softmax (multi-class)
- **Total Parameters**: ~23.6M

## 📈 Results

<div align="center">

### ***Performance Metrics***


Class | Precision | Recall | F1-Score
--- | --- | --- | ---
Glioma | 1.00 | 0.99 |	0.99
Meningioma | 0.98 |	0.99 | 0.98
No Tumor | 1.00 | 1.00 | 1.00
Pituitary | 1.00 | 1.00 | 1.00
Accuracy | - | - | 0.99

## ***Plots***



Training History
 --- 
![](https://github.com/MohammadHMazarei/brain-tumor-resnet50/blob/main/outputs/training_history.png))


Confusion Matrix 
 --- 
![](https://github.com/MohammadHMazarei/brain-tumor-resnet50/blob/main/outputs/confusion_matrix.png)


ROC Curve
 --- 
![](https://github.com/MohammadHMazarei/brain-tumor-resnet50/blob/main/outputs/roc_curve_multiclass.png)


Prediction Samples
 --- 
![](https://github.com/MohammadHMazarei/brain-tumor-resnet50/blob/main/outputs/prediction_samples.png)



## 🔬 Preprocessing Pipeline

### Otsu Edge Detection Steps
1. **Grayscale Conversion** - Convert RGB to grayscale
2. **Gaussian Blur** - Reduce noise (kernel=5×5)
3. **Otsu Thresholding** - Automatic optimal threshold
4. **Morphological Operations** - Erosion + Dilation
5. **Contour Detection** - Find largest contour (tumor region)
6. **Extreme Points** - Extract bounding box coordinates
7. **Cropping** - Extract tumor region
8. **Resizing** - Normalize to 256×256

### Data Augmentation

- **Rotation**: ±10°
- **Width/Height Shift**: 5%
- **Zoom**: 20%
- **Shear**: 5%
- **Horizontal/Vertical Flip**: Enabled


## 🙏 Acknowledgments

- **ResNet50** - [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Otsu's Method** - [A Threshold Selection Method from Gray-Level Histograms](https://ieeexplore.ieee.org/document/4310076)
- **Dataset Contributors** - Kaggle community







