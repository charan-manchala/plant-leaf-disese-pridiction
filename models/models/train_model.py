"""
train_model.py — Train CNN on a SELECTED set of plant types.

Usage — select specific plants (recommended for low-spec systems):
    python models/train_model.py \
        --dataset_path C:/path/to/plantvillage \
        --plants Tomato Potato Corn Apple Grape Pepper \
        --samples_per_class 20 \
        --epochs 10

Usage — use ALL plants and ALL images:
    python models/train_model.py \
        --dataset_path C:/path/to/plantvillage \
        --epochs 25
"""

import os
import json
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, models
from collections import defaultdict
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 8    # safe for low RAM
NUM_WORKERS  = 0    # 0 = no multiprocessing (required on Windows)
LR           = 1e-4
WEIGHT_DECAY = 1e-4

# ── Default 6 plants to use ───────────────────────────────────────────────────
DEFAULT_PLANTS = [
    "Tomato",
    "Potato",
    "Corn_(maize)",
    "Apple",
    "Grape",
    "Pepper,_bell",
]

# ── All available plants in PlantVillage ─────────────────────────────────────
ALL_PLANTS = [
    "Tomato", "Potato", "Corn_(maize)", "Apple", "Grape",
    "Pepper,_bell", "Strawberry", "Peach", "Cherry_(including_sour)",
    "Orange", "Squash", "Soybean", "Raspberry", "Blueberry",
]

# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ── Filter dataset by selected plants ────────────────────────────────────────
def filter_by_plants(dataset, selected_plants: list):
    """
    From a full ImageFolder dataset, keep only classes
    that belong to selected plant types.

    Example: selected_plants = ['Tomato', 'Potato']
    Keeps: Tomato___Early_blight, Tomato___healthy,
           Potato___Late_blight, etc.
    """
    selected_plants_lower = [p.lower() for p in selected_plants]

    # Find matching class indices
    kept_classes   = []
    kept_class_idx = []
    for cls_name, cls_idx in dataset.class_to_idx.items():
        plant_name = cls_name.split("___")[0].lower()
        if any(plant_name == sp for sp in selected_plants_lower):
            kept_classes.append(cls_name)
            kept_class_idx.append(cls_idx)

    if not kept_classes:
        raise ValueError(
            f"No classes matched plants: {selected_plants}\n"
            f"Available classes sample: {list(dataset.class_to_idx.keys())[:5]}"
        )

    # Filter image list
    kept_idx_set = set(kept_class_idx)
    filtered_imgs = [(path, label)
                     for path, label in dataset.imgs
                     if label in kept_idx_set]

    # Remap labels to 0..N-1
    old_to_new = {old: new for new, old in enumerate(sorted(kept_class_idx))}
    remapped_imgs = [(path, old_to_new[label])
                     for path, label in filtered_imgs]

    new_class_to_idx = {
        cls: old_to_new[idx]
        for cls, idx in dataset.class_to_idx.items()
        if idx in kept_idx_set
    }

    return remapped_imgs, new_class_to_idx


# ── Subset sampler (limit images per class) ───────────────────────────────────
def sample_per_class(imgs: list, samples_per_class: int, seed: int = 42):
    """
    Randomly pick up to `samples_per_class` images per class.
    imgs: list of (path, label) tuples
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for path, label in imgs:
        by_class[label].append((path, label))

    selected = []
    for label, items in by_class.items():
        chosen = rng.sample(items, min(samples_per_class, len(items)))
        selected.extend(chosen)

    rng.shuffle(selected)
    return selected


# ── Custom Dataset from (path, label) list ────────────────────────────────────
class LeafDataset(Dataset):
    def __init__(self, imgs: list, transform=None):
        self.imgs      = imgs       # list of (path, label)
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path, label = self.imgs[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(num_classes: int, model_name: str):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    return model


# ── Train one epoch ───────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += inputs.size(0)
        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}", end="\r")
    return total_loss / total, correct / total


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs    = model(inputs)
        loss       = criterion(outputs, labels)
        total_loss += loss.item() * inputs.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += inputs.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Which plants to use
    selected_plants = args.plants if args.plants else DEFAULT_PLANTS

    print(f"\n{'='*55}")
    print(f"  Plant Disease Trainer")
    print(f"{'='*55}")
    print(f"  Device           : {device}")
    print(f"  Model            : {args.model_name}")
    print(f"  Dataset path     : {args.dataset_path}")
    print(f"  Selected plants  : {selected_plants}")
    print(f"  Samples/class    : {args.samples_per_class or 'ALL'}")
    print(f"  Epochs           : {args.epochs}")
    print(f"{'='*55}\n")

    # ── Step 1: Load full dataset ─────────────────────────────────────────────
    print("Step 1: Scanning dataset folder...")
    full_dataset = datasets.ImageFolder(
        root=args.dataset_path,
        transform=train_transforms
    )
    print(f"  Total classes found : {len(full_dataset.classes)}")
    print(f"  Total images found  : {len(full_dataset)}")

    # ── Step 2: Filter to selected plants ────────────────────────────────────
    print(f"\nStep 2: Filtering for selected plants...")
    filtered_imgs, new_class_to_idx = filter_by_plants(
        full_dataset, selected_plants)

    num_classes  = len(new_class_to_idx)
    print(f"  Matched classes     : {num_classes}")
    print(f"  Matched images      : {len(filtered_imgs)}")
    print(f"\n  Classes kept:")
    for cls in sorted(new_class_to_idx.keys()):
        print(f"    [{new_class_to_idx[cls]}] {cls}")

    # ── Step 3: Sample per class if requested ────────────────────────────────
    if args.samples_per_class:
        print(f"\nStep 3: Sampling {args.samples_per_class} images/class...")
        filtered_imgs = sample_per_class(
            filtered_imgs, args.samples_per_class)
        print(f"  Images after sampling : {len(filtered_imgs)}")
    else:
        print(f"\nStep 3: Using all {len(filtered_imgs)} images")

    # ── Step 4: Train / Val split ─────────────────────────────────────────────
    random.Random(42).shuffle(filtered_imgs)
    n_val   = max(1, int(0.2 * len(filtered_imgs)))
    n_train = len(filtered_imgs) - n_val
    train_imgs = filtered_imgs[:n_train]
    val_imgs   = filtered_imgs[n_train:]
    print(f"\nStep 4: Split → Train: {n_train}  Val: {n_val}")

    train_dataset = LeafDataset(train_imgs, transform=train_transforms)
    val_dataset   = LeafDataset(val_imgs,   transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    # ── Step 5: Save class indices ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    class_idx_path = "models/class_indices.json"
    with open(class_idx_path, "w") as f:
        json.dump(new_class_to_idx, f, indent=2)
    print(f"\nStep 5: Class indices saved → {class_idx_path}")

    # ── Step 6: Build model ───────────────────────────────────────────────────
    print(f"\nStep 6: Loading {args.model_name} (pretrained ImageNet)...")
    model     = build_model(num_classes, args.model_name).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    print(f"  Model ready. Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    # ── Step 7: Training loop ─────────────────────────────────────────────────
    print(f"\nStep 7: Training for {args.epochs} epochs...\n")
    best_val_acc    = 0.0
    best_model_path = f"models/{args.model_name}_best.pth"

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch:02d}/{args.epochs}  ", end="")
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            marker = "  ✓ saved"

        print(f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc*100:.1f}%{marker}")

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training Complete!")
    print(f"  Plants trained on  : {selected_plants}")
    print(f"  Classes            : {num_classes}")
    print(f"  Best Val Accuracy  : {best_val_acc*100:.1f}%")
    print(f"  Model saved        : {best_model_path}")
    print(f"  Class indices      : {class_idx_path}")
    print(f"{'='*55}")
    print("\n  Next step → run:  streamlit run app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train plant disease classifier on selected plants"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to PlantVillage dataset root folder"
    )
    parser.add_argument(
        "--plants", type=str, nargs="+", default=None,
        help=(
            "Plant types to include. Default: Tomato Potato Corn_(maize) "
            "Apple Grape Pepper,_bell\n"
            "Example: --plants Tomato Potato Corn_(maize)"
        )
    )
    parser.add_argument(
        "--samples_per_class", type=int, default=None,
        help="Max images per disease class (e.g. 50). Omit to use all."
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--model_name", type=str, default="efficientnet_b0",
        choices=["efficientnet_b0", "resnet50"]
    )
    args = parser.parse_args()
    main(args)