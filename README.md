# 🍃 Plant_Pathology_2020

# Plant Pathology Classification

This project implements a deep learning solution for plant pathology classification using PyTorch. The system is designed to classify plant diseases from images with high accuracy.

## 🔑 Key Components

### Model Architecture
- EfficientNet-B0 as the base model
- Custom head with linear layers for classification
- Pre-trained weights from ImageNet for better feature extraction

### Training Pipeline
- Custom `Trainer` class that handles:
  - Training and validation loops
  - Learning rate scheduling
  - Progress tracking with tqdm
  - Metric computation
  - CutMix augmentation

### Data Processing
- Custom `PlantPathologyDataset` class for:
  - Image loading and preprocessing
  - Label handling
  - Transform application
- DataLoader configuration:
  - Training: batch_size=16, shuffle=True, num_workers=4
  - Validation: batch_size=64, shuffle=False, num_workers=4

### Optimization
- Optimizer: Adam
- Learning Rate Scheduler: ReduceLROnPlateau
- Loss Function: CrossEntropyLoss

### Augmentation
- CutMix augmentation with configurable probability
- Random bbox generation for mixing images
- Beta distribution for mixing ratio

### Metrics
- Custom metric implementation for tracking model performance
- Support for both training and validation metrics

## Features
- Flexible training pipeline that can run with or without validation data
- Progress tracking with detailed metrics
- Learning rate scheduling based on validation loss
- History tracking for:
  - Training loss
  - Training metric
  - Validation loss (when validation data is available)
  - Validation metric (when validation data is available)
  - Learning rate

## Usage
The codebase is designed to be modular and easy to use:
1. Prepare your data in the required format
2. Configure your model and training parameters
3. Initialize the Trainer with your components
4. Run the training process

## Requirements
- PyTorch
- torchvision
- numpy
- opencv-python
- tqdm

## Result 
<img width="1178" alt="Screenshot 2025-05-04 at 11 51 52 pm" src="https://github.com/user-attachments/assets/07afa9f1-4ff8-4d48-a3c6-0f2103854902" />

### First Submission ( with val_data ) : 0.76 private score
<img width="1064" alt="first" src="https://github.com/user-attachments/assets/93f61c91-e609-420e-8be2-ba0f66626933" />

- Overfitting has occured
- To solve this problem, execute training without val_data

### Second Submission : 0.92 private score
<img width="900" alt="second" src="https://github.com/user-attachments/assets/5d63ef62-f62e-4978-8e88-bffe40f6fead" />

- With Early Stopping or Save Model would be able to get better score



