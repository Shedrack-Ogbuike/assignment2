# **ENEN 645 – Assignment 2: Multimodal Garbage Classification**

## **Overview**

This assignment investigates multimodal learning for garbage classification by combining visual information from images with textual information extracted from image filenames. A ResNet-50 convolutional neural network is used for image feature extraction, while a bag-of-words representation encodes textual features. The fused features are used to classify images into four garbage categories: Black, Blue, Green, and TTR.

---

## **Dataset**

The CVPR 2024 Garbage Classification dataset is used, consisting of RGB images organized into four class folders:

* **Black**
* **Blue**
* **Green**
* **TTR**

The dataset is **not included** in this repository due to size constraints. Training, validation, and test splits were used as provided.

---

## **Model Architecture**

* **Image Encoder:** ResNet-50 pretrained on ImageNet (final classification layer removed)
* **Text Encoder:** Bag-of-Words vector built from cleaned filename tokens
* **Fusion:** Concatenation of image and text feature embeddings
* **Classifier:** Fully connected layers with ReLU and dropout

This architecture allows the model to leverage complementary visual and semantic cues.

---

## **Training Details**

* **Loss Function:** Cross-Entropy Loss
* **Optimizer:** AdamW
* **Batch Size:** 32
* **Epochs:** 8
* **Hardware:** NVIDIA Tesla T4 GPU via **TALC (Slurm)**
* **Checkpointing:** Best model saved based on validation accuracy

Training was resumed automatically if a saved checkpoint was detected.

---

## **Results**

* **Final Test Accuracy:** **83.77%**
* **Per-Class Accuracy:**

  * Black: 68.35%
  * Blue: 89.96%
  * Green: 90.86%
  * TTR: 81.81%

Green and Blue bins achieved the highest accuracy, while Black bins were more challenging due to visual similarity and less distinctive filename text. Confusion matrices, learning curves, and misclassified examples are included in the `results/` folder.

---

## **How to Run**

### **On TALC (GPU)**

```bash
sbatch run_a2.slurm
```

The Slurm script requests a GPU node and executes the training script located in `src/train_multimodal.py`.

### **Notebook**

A Jupyter notebook (`notebooks/train_multimodal.ipynb`) is provided for result inspection, visualization, and rubric verification. The notebook loads saved outputs rather than retraining the model.

---

## **Repository Structure**

```
assignment2/
├── notebooks/          # Jupyter notebook (results + evaluation)
├── src/                # Training script (TALC)
├── results/            # Plots, confusion matrix, predictions CSV
├── run_a2.slurm        # Slurm submission script
├── README.md
└── TRAINING_SUMMARY.md
```




