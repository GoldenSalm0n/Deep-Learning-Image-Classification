# Deep Learning Research: Image Classification & Generative Models

![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active_Development-blue)

> **Note:** This project was developed as a university coursework assignment to demonstrate proficiency in **Deep Learning architectures, Computer Vision pipelines, and Generative AI concepts**.

## Project Overview
This repository contains a comprehensive study of Deep Learning techniques applied to Computer Vision tasks. The project evolves from training a **custom CNN** from scratch to implementing advanced **Transfer Learning** strategies and exploring **Generative AI** concepts using Autoencoders.

The goal was to analyze how different architectures, loss functions (MSE vs L1), and regularization techniques impact model performance and image reconstruction quality.

---

## Research Objectives
The project was designed to fulfill the following technical requirements:
1.  **Baseline Modeling:** Develop and train a custom Convolutional Neural Network (CNN) from scratch.
2.  **Transfer Learning Analysis:** Compare **Feature Extraction** vs. **Fine-Tuning** using state-of-the-art architectures (**ResNet18**, **DenseNet121**).
3.  **Data Optimization:** Implement advanced Data Augmentation (Rotation, Flipping, Color Jitter) to prevent overfitting.
4.  **Generative Tasks:**
    * Build a **Denoising Autoencoder** to restore clean images from noisy inputs.
    * Develop a **Variational Autoencoder (VAE)** to generate new synthetic image samples.

---

## Key Results & Visualizations

### 1. Denoising Autoencoder
*Restoring clean images from inputs with added Gaussian noise.*
<p align="center">
  <img src="results/dae_test.png" width="800" alt="Denoising Result">
</p>

### 2. Variational Autoencoder (VAE) Generation
*Generating new flower samples from the latent space using L1 Loss for sharper details.*
<p align="center">
  <img src="results/vae_new_flowers_l1.png" width="800" alt="VAE Generation">
</p>

---

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/GoldenSalm0n/Deep-Learning-Image-Classification.git](https://github.com/GoldenSalm0n/Deep-Learning-Image-Classification.git)
   cd Deep-Learning-Image-Classification


## Project Roadmap()
This project was developed as an individual coursework assignment with the goal of mastering Deep Learning pipelines from scratch to advanced generative models.

Below is the implementation status of the project requirements:

### ✅ Stage 1: Dataset Preparation
- [x] Select a dataset with 3-20 classes and >1000 images.
- [x] **Chosen Dataset:** [Назва твого датасету, напр. Flowers Recognition]
- [x] Preprocessing: Resizing to 64x64/128x128, Normalization.

### ✅ Stage 2: Baseline CNN Model
- [x] Design a custom CNN architecture from scratch (`models/custom_cnn.py`).
- [x] Implement training loop with **Early Stopping** (patience=3-5).
- [x] Visualize Learning Curves (Loss/Accuracy).
- [x] **Target:** Achieve >60% Validation Accuracy (Achieved: **XX%**).

### ✅ Stage 3: Transfer Learning
- [x] Compare pretrained models: **ResNet18** vs **DenseNet121**.
- [x] Experiment: **Feature Extraction** vs **Fine-Tuning**.
- [x] Implement Differential Learning Rates (different LR for backbone and head).
- [x] **Outcome:** ResNet18 showed the best performance/time ratio.

### [Prossesing...] Stage 4: Optimization & Regularization
- [x] Compare Baseline Augmentation vs Advanced Augmentation (Rotation, ColorJitter).
- [x] Analyze Regularization impact: Dropout, L2, Label Smoothing.
- [ ] *Ensembling.*

### ✅ Stage 5: Autoencoders & Generative AI
- [x] **Architecture:** Built Convolutional Autoencoder (`models/autoencoder.py`).
- [x] **Task A (Denoising):** Restoring clean images from noisy inputs.
- [x] **Task B (Generation - VAE):** Implementing Variational Autoencoder for generating new samples from latent space.

### [Prossesing...] Stage 6: Beyond the Curriculum (Ongoing Research)
> **Status:** *Roadmap is currently being updated. These objectives represent personal initiative and are not part of the university coursework.*

- [ ] **Model Deployment:** Planning to wrap the model into a simple web application for demo usage.
- [ ] **Real-world Testing:** Testing the model on custom photos taken from smartphone cameras to check robustness.
- [ ] **Code Refactoring:** Optimizing the project structure and adding detailed docstrings.
[In prossesing...]





