# Monkey Species Image Classification using CNNs and Transfer Learning

This project performs **multi-class image classification** on a Monkey Species dataset using:

1. **Custom Convolutional Neural Networks (CNNs)**  
2. **Transfer Learning with VGG16 (pretrained on ImageNet)**  
3. **Qualitative error analysis** on misclassified images  

The project compares model performance using **test accuracy** and **confusion matrices**, and analyzes why certain images are difficult to classify.

---

## Project Highlights
- Designed and trained **two custom CNN architectures**
- Selected the **best-performing CNN** based on test accuracy
- Implemented **transfer learning** using a pretrained **VGG16** model
- Generated **confusion matrices** for model comparison
- Performed **error analysis** on challenging test images
- Organized as a **reproducible, GitHub-ready ML project**

---

## Assignment Tasks Overview

### Task 1: Custom CNN Architectures
- Built two CNN models with different architectural choices
- Trained both models until performance stabilized
- Evaluated both on a held-out test set
- Selected the better-performing CNN
- Generated a confusion matrix for the best CNN
- Saved the trained model locally

### Task 2: Transfer Learning (VGG16)
- Used **VGG16** pretrained on ImageNet as a feature extractor
- Added custom classification layers on top
- Trained the new classifier head
- Evaluated on the same test dataset
- Generated a confusion matrix for comparison with the custom CNN

### Task 3: Error Analysis
- Selected **10 test images** misclassified by the best CNN
- Analyzed why these images are difficult (pose, lighting, background, similarity)
- Compared predictions of the fine-tuned VGG16 model on the same images

---

## Dataset (Not Included)

⚠️ **The dataset is NOT included in this repository due to its large size.**

You must download the **Monkey Species Image Dataset** separately and place it locally.

### Expected Local Folder Structure

data/
├── Training Data/
│ ├── Class_1/
│ ├── Class_2/
│ └── ...
├── Prediction Data/
│ ├── Class_1/
│ ├── Class_2/
│ └── ...
└── test_error_images/
└── (10 selected test images for error analysis)


- Each class folder must contain image files (`.jpg`, `.png`, `.jpeg`)
- Folder names must be **consistent across training and prediction data**
- The `test_error_images` folder is used only for Task 3

---

## Trained Models (Not Included)

⚠️ **Trained model files are NOT included** in this repository because of size constraints.

When run locally, the script will generate:

models/
├── best_model.keras
└── tuned.keras


These files are intentionally excluded from GitHub.

---

## How to Run (Local Only)

> This project requires **Python 3.11**.  
> TensorFlow does **not** currently support Python 3.14 on Windows.

### 1. Create and activate virtual environment
```bash
py -3.11 -m venv venv
venv\Scripts\activate
