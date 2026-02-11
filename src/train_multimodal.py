import os
import re
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# TALC paths
BASE_DIR = "/work/TALC/ensf617_2026w/garbage_data"

TRAIN_DIR = os.path.join(BASE_DIR, "CVPR_2024_dataset_Train")
VAL_DIR   = os.path.join(BASE_DIR, "CVPR_2024_dataset_Val")
TEST_DIR  = os.path.join(BASE_DIR, "CVPR_2024_dataset_Test")

# Always align class order to folder names
CLASS_NAMES = sorted([d for d in os.listdir(TRAIN_DIR)
                      if os.path.isdir(os.path.join(TRAIN_DIR, d))])
print("CLASS_NAMES:", CLASS_NAMES)

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    print(d, "exists:", os.path.exists(d))


# Quick count
def count_images(root_dir):
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.exists(cls_dir):
            continue
        total += len([f for f in os.listdir(cls_dir)
                      if f.lower().endswith((".jpg",".jpeg",".png"))])
    return total

print("TRAIN:", count_images(TRAIN_DIR))
print("VAL  :", count_images(VAL_DIR))
print("TEST :", count_images(TEST_DIR))


# Text processing + vocab
def filename_to_text(fname: str) -> str:
    base = os.path.splitext(fname)[0]
    base = re.sub(r"_\d+$", "", base)
    return base.replace("_", " ").strip()

def tokenize(text: str):
    return re.findall(r"[a-zA-Z]+", text.lower())

def build_vocab_from_dirs(dirs, class_names, max_vocab=5000, min_freq=2):
    counter = Counter()
    for root in dirs:
        for cls in class_names:
            cls_dir = os.path.join(root, cls)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    counter.update(tokenize(filename_to_text(f)))

    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq >= min_freq and len(vocab) < max_vocab:
            vocab[word] = len(vocab)
    return vocab

VOCAB = build_vocab_from_dirs([TRAIN_DIR, VAL_DIR], CLASS_NAMES)
VOCAB_SIZE = len(VOCAB)
print("Vocab size:", VOCAB_SIZE)

# Transforms
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ]),
}


# Dataset (image + text_vec)
class ImageTextGarbageDataset(Dataset):
    def __init__(self, root_dir, transform=None, vocab=None, class_names=None):
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = vocab
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        self.samples = []
        for cls in class_names:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".jpg",".jpeg",".png")):
                    self.samples.append((
                        os.path.join(cls_dir, f),
                        filename_to_text(f),
                        self.class_to_idx[cls]
                    ))

    def __len__(self):
        return len(self.samples)

    def encode_text_bow(self, text):
        vec = torch.zeros(len(self.vocab), dtype=torch.float32)
        for w in tokenize(text):
            vec[self.vocab.get(w, self.vocab["<unk>"])] += 1.0
        if vec.sum() > 0:
            vec /= vec.sum()
        return vec

    def __getitem__(self, idx):
        path, text, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return {
            "image": img,
            "text_vec": self.encode_text_bow(text),
            "label": torch.tensor(label, dtype=torch.long),
            "path": path,
            "text": text
        }

# Datasets + Dataloaders
datasets = {
    "train": ImageTextGarbageDataset(TRAIN_DIR, transform["train"], VOCAB, CLASS_NAMES),
    "val":   ImageTextGarbageDataset(VAL_DIR,   transform["val"],   VOCAB, CLASS_NAMES),
    "test":  ImageTextGarbageDataset(TEST_DIR,  transform["test"],  VOCAB, CLASS_NAMES),
}
print("Dataset sizes:", {k: len(v) for k, v in datasets.items()})

pin = device.type == "cuda"
dataloaders = {
    "train": DataLoader(datasets["train"], batch_size=32, shuffle=True,  num_workers=0, pin_memory=pin),
    "val":   DataLoader(datasets["val"],   batch_size=32, shuffle=False, num_workers=0, pin_memory=pin),
    "test":  DataLoader(datasets["test"],  batch_size=32, shuffle=False, num_workers=0, pin_memory=pin),
}


# Model (ResNet50 + text MLP)
class ResNetMultimodalClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        try:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        except AttributeError:
            resnet = models.resnet50(pretrained=True)

        self.image_features = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Linear(2048, 512)

        self.text_fc = nn.Sequential(
            nn.Linear(vocab_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_vec):
        img = self.image_features(images).squeeze(-1).squeeze(-1)
        img = self.image_fc(img)
        txt = self.text_fc(text_vec)
        fused = torch.cat((img, txt), dim=1)
        return self.classifier(fused)


# Training function
def train_model(model, loaders, criterion, optimizer, epochs, device, save_path):
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for ep in range(epochs):
        print(f"\nEpoch {ep+1}/{epochs}")
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            loss_sum = 0.0
            correct = 0

            for batch in tqdm(loaders[phase], leave=False):
                imgs = batch["image"].to(device)
                txts = batch["text_vec"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(imgs, txts)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                loss_sum += loss.item() * imgs.size(0)
                correct += (preds == labels).sum().item()

            epoch_loss = loss_sum / len(loaders[phase].dataset)
            epoch_acc = correct / len(loaders[phase].dataset)

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"{phase}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"✅ Saved best model: {save_path} (val acc={best_acc:.4f})")

    return history



# Train model
model = ResNetMultimodalClassifier(VOCAB_SIZE, len(CLASS_NAMES)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUT_DIR, "best_model.pth")
# Resume if checkpoint exists
if os.path.exists(MODEL_PATH):
    print("🔁 Resuming from:", MODEL_PATH)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
history = train_model(model, dataloaders, criterion, optimizer, epochs=8,
                      device=device, save_path=MODEL_PATH)


# Save curves
plt.figure()
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss vs Epoch")
plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history["train_acc"], label="train_acc")
plt.plot(history["val_acc"], label="val_acc")
plt.legend()
plt.title("Accuracy vs Epoch")
plt.savefig(os.path.join(OUT_DIR, "acc_curve.png"), dpi=200, bbox_inches="tight")
plt.close()


# Test evaluation
print("\nLoading best model from:", MODEL_PATH)
print("Exists:", os.path.exists(MODEL_PATH))

model = ResNetMultimodalClassifier(vocab_size=VOCAB_SIZE, num_classes=len(CLASS_NAMES)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

test_loader = dataloaders["test"]

all_preds, all_labels, all_paths, all_texts = [], [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader):
        images = batch["image"].to(device)
        text_vec = batch["text_vec"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images, text_vec)
        predicted = outputs.argmax(dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_paths.extend(batch["path"])
        all_texts.extend(batch["text"])

accuracy = 100 * (np.array(all_preds) == np.array(all_labels)).mean()
print(f"\nAccuracy on test set: {accuracy:.2f}%\n")

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title("Confusion Matrix (Test)")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
plt.close()

per_class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
for name, acc in zip(CLASS_NAMES, per_class_accuracy):
    print(f"Accuracy for {name}: {acc:.2f}%")


# Misclassified examples
misclassified = {name: [] for name in CLASS_NAMES}

mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])

for i, (y, p) in enumerate(zip(all_labels, all_preds)):
    if y != p:
        true_name = CLASS_NAMES[y]
        pred_name = CLASS_NAMES[p]

        img = Image.open(all_paths[i]).convert("RGB")
        img = transform["test"](img).cpu().numpy().transpose(1,2,0)
        img = (img * std) + mean
        img = np.clip(img, 0, 1)

        misclassified[true_name].append({
            "image": img,
            "true": true_name,
            "pred": pred_name,
            "text": all_texts[i]
        })

plt.figure(figsize=(15, 12))
rows = len(CLASS_NAMES)
for row, cname in enumerate(CLASS_NAMES):
    examples = misclassified[cname]
    if len(examples) == 0:
        continue
    selected = random.sample(examples, min(3, len(examples)))
    for col, ex in enumerate(selected):
        plt.subplot(rows, 3, row*3 + col + 1)
        plt.imshow(ex["image"])
        plt.title(f"True: {ex['true']}\nPred: {ex['pred']}\n{ex['text'][:20]}")
        plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "misclassified_examples.png"), dpi=200, bbox_inches="tight")
plt.close()

for cname in CLASS_NAMES:
    print(f"{cname}: {len(misclassified[cname])} misclassified examples")


# Save predictions
import pandas as pd

df = pd.DataFrame({
    "path": all_paths,
    "text": all_texts,
    "true": [CLASS_NAMES[i] for i in all_labels],
    "pred": [CLASS_NAMES[i] for i in all_preds],
})
csv_path = os.path.join(OUT_DIR, "test_predictions.csv")
df.to_csv(csv_path, index=False)
print("Saved predictions CSV:", csv_path)

print("\n✅ DONE. Outputs saved to:", OUT_DIR)

