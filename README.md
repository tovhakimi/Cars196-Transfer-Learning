# Transfer Learning Model - README

## Overview  
This repository implements a **Transfer Learning approach** for **image classification** using the Stanford Cars Dataset. The goal is to leverage pre-trained models and fine-tune them to accurately classify car images into 196 categories.

## Models Used  
- **SqueezeNet (Baseline Model):** A lightweight architecture used as a starting point for transfer learning. Its smaller size allows for faster training and evaluation.
- **MobileNetV2 (Enhanced Model):** A more advanced architecture with depthwise separable convolutions and residual connections, improving accuracy and reducing computation.
- **EfficientNet-B0 (Final Model):** A state-of-the-art model utilizing compound scaling to balance depth, width, and resolution, resulting in higher accuracy with minimal computational cost.

## Techniques Applied  
- **Transfer Learning:** Using pre-trained weights from ImageNet and fine-tuning on the Stanford Cars Dataset to enhance feature extraction.  
- **Data Augmentation:** Cropping, resizing, color jittering, random horizontal flipping, and normalization to improve generalization and robustness.  
- **Optimization Techniques:** Utilizing Adam optimizer, learning rate scheduling, L2 regularization, and dropout layers to enhance training efficiency and performance.  
- **Evaluation:** Accuracy, loss, confusion matrix, and F1-score are used to evaluate model performance.

## Installation and Usage  
1. Clone the repository.   
2. Upload the Stanford Cars Dataset to your working directory.  
3. Run the provided Jupyter notebooks for training and evaluation.

## Results  
The best model achieved improved classification performance compared to the baseline through various enhancements in architecture and training techniques.

## References  
- Stanford Cars Dataset: [Stanford AI Lab](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)  
- PyTorch Documentation: [PyTorch](https://pytorch.org/docs/stable/index.html)  
- EfficientNet Paper: [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
