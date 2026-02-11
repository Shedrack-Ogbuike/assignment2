# Training Summary — ENEL 645 Assignment 2

This document summarizes the GPU-based training and evaluation of the multimodal
garbage classification model executed on the University of Calgary TALC cluster
using Slurm.

---

## Compute Environment
- Platform: TALC (University of Calgary HPC)
- Scheduler: Slurm
- Device: NVIDIA GPU (CUDA)
- Framework: PyTorch
- Execution mode: Batch job (`sbatch`)

---

## Model Description
- Image branch: ResNet50 (ImageNet pretrained)
- Text branch: Bag-of-Words features extracted from image filenames
- Fusion strategy: Concatenation of image and text features
- Classifier: Fully connected multilayer perceptron
- Number of classes: 4 (Black, Blue, Green, TTR)

---

## Training Progress

The model was trained for **8 epochs** with validation-based checkpointing.
The best model was saved based on validation accuracy.

### Epoch-wise Results

Epoch 1/8  
train: loss = 0.2935, acc = 0.8987  
val:   loss = 0.3988, acc = 0.8728  

Epoch 2/8  
train: loss = 0.2336, acc = 0.9181  
val:   loss = 0.3906, acc = 0.8661  

Epoch 3/8  
train: loss = 0.2436, acc = 0.9132  
val:   loss = 0.3791, acc = 0.8744  

Epoch 4/8  
train: loss = 0.2051, acc = 0.9291  
val:   loss = 0.3898, acc = 0.8778  

Epoch 5/8  
train: loss = 0.1717, acc = 0.9405  
val:   loss = 0.4286, acc = 0.8678  

Epoch 6/8  
train: loss = 0.1398, acc = 0.9515  
val:   loss = 0.4519, acc = 0.8711  

Epoch 7/8  
train: loss = 0.1219, acc = 0.9579  
val:   loss = 0.4823, acc = 0.8711  

Epoch 8/8  
train: loss = 0.0979, acc = 0.9665  
val:   loss = 0.6013, acc = 0.8444  

**Best validation accuracy achieved: 87.78%**

---

## Test Set Evaluation

The best validation checkpoint was evaluated on the held-out test set.

- Test accuracy: **83.77%**

### Classification Report (Test Set)

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| Black | 0.78 | 0.68 | 0.73 | 695 |
| Blue  | 0.79 | 0.90 | 0.84 | 1086 |
| Green | 0.92 | 0.91 | 0.91 | 799 |
| TTR   | 0.87 | 0.82 | 0.84 | 852 |

Overall accuracy: **0.84**

---

## Per-Class Accuracy

- Black: 68.35%
- Blue: 89.96%
- Green: 90.86%
- TTR: 81.81%

---

## Misclassification Summary

Number of misclassified samples per class:
- Black: 220
- Blue: 109
- Green: 73
- TTR: 155

---

## Generated Outputs

- Trained model checkpoint:  
  `garbage_data/outputs/best_model.pth`

- Training curves:  
  `results/loss_curve.png`  
  `results/acc_curve.png`

- Confusion matrix:  
  `results/confusion_matrix.png`

- Misclassified example visualization:  
  `results/misclassified_examples.png`

- Test predictions CSV:  
  `results/test_predictions.csv`

---

## Reproducibility

- Training script: `src/train_multimodal.py`
- Slurm job file: `run_a2.slurm`
- Full training log: `logs/slurm_599762.out`

This summary was generated directly from the TALC Slurm output to provide
transparent and reproducible evidence of model training and performance.
# Training Summary — ENEL 645 Assignment 2

This document summarizes the GPU-based training and evaluation of the multimodal
garbage classification model executed on the University of Calgary TALC cluster
using Slurm.

---

## Compute Environment
- Platform: TALC (University of Calgary HPC)
- Scheduler: Slurm
- Device: NVIDIA GPU (CUDA)
- Framework: PyTorch
- Execution mode: Batch job (`sbatch`)

---

## Model Description
- Image branch: ResNet50 (ImageNet pretrained)
- Text branch: Bag-of-Words features extracted from image filenames
- Fusion strategy: Concatenation of image and text features
- Classifier: Fully connected multilayer perceptron
- Number of classes: 4 (Black, Blue, Green, TTR)

---

## Training Progress

The model was trained for **8 epochs** with validation-based checkpointing.
The best model was saved based on validation accuracy.

### Epoch-wise Results

Epoch 1/8  
train: loss = 0.2935, acc = 0.8987  
val:   loss = 0.3988, acc = 0.8728  

Epoch 2/8  
train: loss = 0.2336, acc = 0.9181  
val:   loss = 0.3906, acc = 0.8661  

Epoch 3/8  
train: loss = 0.2436, acc = 0.9132  
val:   loss = 0.3791, acc = 0.8744  

Epoch 4/8  
train: loss = 0.2051, acc = 0.9291  
val:   loss = 0.3898, acc = 0.8778  

Epoch 5/8  
train: loss = 0.1717, acc = 0.9405  
val:   loss = 0.4286, acc = 0.8678  

Epoch 6/8  
train: loss = 0.1398, acc = 0.9515  
val:   loss = 0.4519, acc = 0.8711  

Epoch 7/8  
train: loss = 0.1219, acc = 0.9579  
val:   loss = 0.4823, acc = 0.8711  

Epoch 8/8  
train: loss = 0.0979, acc = 0.9665  
val:   loss = 0.6013, acc = 0.8444  

**Best validation accuracy achieved: 87.78%**

---

## Test Set Evaluation

The best validation checkpoint was evaluated on the held-out test set.

- Test accuracy: **83.77%**

### Classification Report (Test Set)

| Class | Precision | Recall | F1-score | Support |
|------|-----------|--------|----------|---------|
| Black | 0.78 | 0.68 | 0.73 | 695 |
| Blue  | 0.79 | 0.90 | 0.84 | 1086 |
| Green | 0.92 | 0.91 | 0.91 | 799 |
| TTR   | 0.87 | 0.82 | 0.84 | 852 |

Overall accuracy: **0.84**

---

## Per-Class Accuracy

- Black: 68.35%
- Blue: 89.96%
- Green: 90.86%
- TTR: 81.81%

---

## Misclassification Summary

Number of misclassified samples per class:
- Black: 220
- Blue: 109
- Green: 73
- TTR: 155

---

## Generated Outputs

- Trained model checkpoint:  
  `garbage_data/outputs/best_model.pth`

- Training curves:  
  `results/loss_curve.png`  
  `results/acc_curve.png`

- Confusion matrix:  
  `results/confusion_matrix.png`

- Misclassified example visualization:  
  `results/misclassified_examples.png`

- Test predictions CSV:  
  `results/test_predictions.csv`

---

## Reproducibility

- Training script: `src/train_multimodal.py`
- Slurm job file: `run_a2.slurm`
- Full training log: `logs/slurm_599762.out`

This summary was generated directly from the TALC Slurm output to provide
transparent and reproducible evidence of model training and performance.

