import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# ==============================================================================
# CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
DATASET_PATH = "dataset/IRDID/images"
MODEL_PATH = "model/flooded_road_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==============================================================================
# DATASET DISCOVERY
# ==============================================================================
classes = sorted([
    d for d in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith(".")
])
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

image_paths = []
image_labels = []

for cls_name in classes:
    cls_folder = os.path.join(DATASET_PATH, cls_name)
    files = [
        f for f in os.listdir(cls_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
    ]
    for f in files:
        image_paths.append(os.path.join(cls_folder, f))
        image_labels.append(class_to_idx[cls_name])

print("==================================================")
print("       FLOODED ROAD AI - MODEL TRAINING          ")
print("   JIGNASA 2026: Track B (CNN & Computer Vision)  ")
print("==================================================")
print(f"Total dataset images found: {len(image_paths)}")
print("Classes and distribution:")
for cls_name in classes:
    count = image_labels.count(class_to_idx[cls_name])
    print(f"  [{class_to_idx[cls_name]}] {cls_name:20s}: {count:4d} images")

# Stratified split: ensures equal class proportions in train & val
train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    image_labels,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=image_labels
)

print(f"\nSplit: {len(train_paths)} training images, {len(val_paths)} validation images (80/20 stratified)")

# ==============================================================================
# PYTORCH DATASET WITH INDEPENDENT TRANSFORMS
# ==============================================================================
train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class RoadHazardDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = RoadHazardDataset(train_paths, train_labels, train_transforms)
val_dataset = RoadHazardDataset(val_paths, val_labels, val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==============================================================================
# HARDWARE ACCELERATION SETUP (MPS / CPU)
# ==============================================================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\n🚀 Hardware Acceleration: Apple Silicon MPS (Metal Performance Shaders) active.")
else:
    device = torch.device("cpu")
    print(f"\n⚙️  Hardware Acceleration: CPU mode.")

# ==============================================================================
# MODEL ARCHITECTURE (RESNET18 TRANSFER LEARNING)
# ==============================================================================
print("Loading pretrained ResNet18 weights...")
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace final classification head with 6 target hazard classes
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_features, len(classes))
)
model = model.to(device)

# Optional balanced class weighting for loss
class_counts = [train_labels.count(i) for i in range(len(classes))]
weights_per_class = [1.0 / c for c in class_counts]
weights_tensor = torch.tensor(weights_per_class, dtype=torch.float32).to(device)
weights_tensor = weights_tensor / weights_tensor.sum() * len(classes)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ==============================================================================
# TRAINING & EVALUATION LOOP
# ==============================================================================
print(f"\nStarting training for {EPOCHS} epochs...\n")
best_macro_f1 = 0.0
best_accuracy = 0.0
best_state = None
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    scheduler.step()
    train_loss = running_loss / total
    train_acc = (correct / total) * 100.0

    # Validation evaluation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss = val_loss / val_total
    val_acc = (val_correct / val_total) * 100.0

    prec, rec, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    print(
        f"Epoch [{epoch:2d}/{EPOCHS:2d}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | "
        f"Val Macro-F1: {f1 * 100:5.2f}%"
    )

    if f1 > best_macro_f1:
        best_macro_f1 = f1
        best_accuracy = val_acc
        best_state = {
            "model_state_dict": model.state_dict(),
            "classes": classes,
            "class_to_idx": class_to_idx,
            "image_size": IMAGE_SIZE,
            "val_accuracy": val_acc,
            "val_macro_f1": f1 * 100.0,
            "architecture": "resnet18"
        }

total_duration = time.time() - start_time
print(f"\nTraining completed in {total_duration:.1f} seconds.")
print(f"Best Validation Accuracy: {best_accuracy:.2f}% | Best Macro-F1: {best_macro_f1 * 100:.2f}%")

# ==============================================================================
# FINAL DETAILED METRICS EVALUATION
# ==============================================================================
# Load best weights for reporting
model.load_state_dict(best_state["model_state_dict"])
model.eval()

final_preds = []
final_targets = []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        final_preds.extend(preds.cpu().numpy())
        final_targets.extend(labels.numpy())

# Map clean display names for reporting
display_class_names = [
    cls_name.replace("_", " ").title() for cls_name in classes
]

print("\n" + "=" * 65)
print("           DETAILED EVALUATION REPORT (VALIDATION SET)")
print("=" * 65)

print("\n--- PER-CLASS CLASSIFICATION REPORT ---")
print(classification_report(
    final_targets,
    final_preds,
    target_names=display_class_names,
    digits=4,
    zero_division=0
))

macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    final_targets, final_preds, average="macro", zero_division=0
)
weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
    final_targets, final_preds, average="weighted", zero_division=0
)

print("--- OVERALL METRICS ---")
print(f"Accuracy:           {best_accuracy:6.2f}%")
print(f"Macro Precision:    {macro_p * 100:6.2f}%")
print(f"Macro Recall:       {macro_r * 100:6.2f}%")
print(f"Macro F1-Score:     {macro_f1 * 100:6.2f}%")
print(f"Weighted F1-Score:  {weighted_f1 * 100:6.2f}%")

# Highlight safety-critical recall
waterlog_idx = class_to_idx.get("waterlogged_roads")
manhole_idx = class_to_idx.get("open_manholes")
pothole_idx = class_to_idx.get("potholes")

per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
    final_targets, final_preds, average=None, zero_division=0
)

print("\n--- SAFETY-CRITICAL RECALL SUMMARY ---")
if waterlog_idx is not None:
    print(f"  Waterlogged Roads Recall: {per_class_r[waterlog_idx] * 100:6.2f}%")
if manhole_idx is not None:
    print(f"  Open Manholes Recall:     {per_class_r[manhole_idx] * 100:6.2f}%")
if pothole_idx is not None:
    print(f"  Potholes Recall:          {per_class_r[pothole_idx] * 100:6.2f}%")

print("\n--- CONFUSION MATRIX ---")
cm = confusion_matrix(final_targets, final_preds)
print("Pred ->  " + "  ".join([f"{c[:6]:>6}" for c in classes]))
for idx, row in enumerate(cm):
    print(f"{classes[idx][:7]:<8}: " + "  ".join([f"{val:6d}" for val in row]))

# ==============================================================================
# SAVE BEST MODEL
# ==============================================================================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(best_state, MODEL_PATH)

print("\n" + "=" * 65)
print(f"✅ MODEL SUCCESSFULLY TRAINED AND SAVED TO: {MODEL_PATH}")
print("=" * 65)