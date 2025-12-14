# Deep Learning for Medical Image Segmentation and Traffic Sign Detection

This repository contains the implementation of two deep learning projects developed as part of the **Neural Networks and Deep Learning** course. The project focuses on applying convolutional neural networks (CNNs) and transfer learning techniques to solve real-world computer vision problems in medical imaging and autonomous driving.

---

## Project Overview

The project consists of two main parts:

1. **Brain Tumor Segmentation from MRI Images**
2. **Traffic Sign Detection using Object Detection Models**

Each part explores different deep learning paradigms, evaluation metrics, and model architectures.

---

## Part 1: Brain Tumor Segmentation

### Task Description
The goal of this task is to perform pixel-wise segmentation of brain tumors from MRI images. Accurate segmentation is critical for medical diagnosis and treatment planning.

### Methodology
- Implemented a **VGG16-UNet** architecture inspired by the original U-Net model.
- Leveraged **transfer learning** by using a pre-trained VGG16 encoder to improve feature extraction.
- Applied data augmentation techniques such as rotation, scaling, and intensity variation.
- Used **Dice Loss** as the primary loss function.
- Evaluated the model using **Dice Coefficient**, **IoU (Intersection over Union)**, and **Accuracy**.

### Key Features
- Encoder-decoder segmentation architecture
- Transfer learning for improved convergence and performance
- Visualization of predicted masks alongside ground truth masks
- Training and validation metric analysis

---

## Part 2: Traffic Sign Detection

### Task Description
This task focuses on detecting and classifying traffic signs, a key component of Advanced Driver-Assistance Systems (ADAS) and autonomous driving.

### Dataset
- **GTSDB (German Traffic Sign Detection Benchmark)**
- Traffic signs categorized into *prohibitory*, *danger*, *mandatory*, and *other*
- Objects analyzed across different scales (small, medium, large)

### Models Implemented
- **Faster R-CNN with ResNet50-FPN** (two-stage detector)
- **SSD300 with VGG16 backbone** (one-stage detector)
