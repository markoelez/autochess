#!/usr/bin/env python3
"""
Fine-tuned chess-piece classifier based on a pretrained ResNet-18 backbone.
Now robust to *tiny* datasets (even one image per class) by falling back to a
non-stratified split when stratification is impossible.
"""

import os
from typing import Dict, List, Tuple
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ───────────────────────────── DATASET ──────────────────────────────


class ChessPieceDataset(Dataset):
  """Dataset that performs *on-the-fly* augmentation with torchvision transforms."""

  def __init__(
    self,
    image_paths: List[str],
    labels: List[str],
    transform: transforms.Compose,
    label_encoder: LabelEncoder,
  ):
    self.image_paths = image_paths
    self.encoded_labels = label_encoder.transform(labels)
    self.transform = transform

  def __len__(self) -> int:
    return len(self.image_paths)

  def __getitem__(self, idx):
    img = Image.open(self.image_paths[idx]).convert("RGB")
    img = self.transform(img)
    label = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
    return img, label


def load_chess_pieces_data(
  pieces_dir: str = "dat/labeled_dataset",
) -> Tuple[List[str], List[str], LabelEncoder]:
  """Load <file, label> pairs; labels are drawn from directory names."""
  image_paths, labels = [], []

  # Check if the pieces_dir has subdirectories (new labeled_dataset structure)
  subdirs = [d for d in os.listdir(pieces_dir) if os.path.isdir(os.path.join(pieces_dir, d))]

  if subdirs:
    # New structure: pieces_dir contains subdirectories for each piece type
    for piece_type in subdirs:
      piece_dir = os.path.join(pieces_dir, piece_type)
      for f in os.listdir(piece_dir):
        fp = os.path.join(piece_dir, f)
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
          labels.append(piece_type)
          image_paths.append(fp)
  else:
    # Old structure: flat directory with piece names in filenames
    for f in os.listdir(pieces_dir):
      fp = os.path.join(pieces_dir, f)
      if f.lower().endswith((".png", ".jpg", ".jpeg")):
        name = f.rsplit(".", 1)[0]
        # Extract piece type from augmented filename (e.g., "black_bishop_aug_0" -> "black_bishop")
        if "_aug_" in name:
          piece_name = name.rsplit("_aug_", 1)[0]
        else:
          piece_name = name
        labels.append("empty" if piece_name == "blank" else piece_name)
        image_paths.append(fp)

  if not image_paths:
    raise RuntimeError(f"No images found in {pieces_dir}")

  le = LabelEncoder().fit(labels)
  return image_paths, labels, le


# ───────────────────────────── MODEL ────────────────────────────────


class ChessPieceNet(nn.Module):
  """ResNet-18 backbone; final layer replaced for K-class output."""

  def __init__(self, num_classes: int):
    super().__init__()
    self.backbone: models.ResNet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final FC
    in_feats = self.backbone.fc.in_features
    self.backbone.fc = nn.Sequential(
      nn.Dropout(p=0.4),
      nn.Linear(in_feats, num_classes),
    )

  def forward(self, x):
    return self.backbone(x)


# ───────────────────────────── TRAINING ─────────────────────────────


def build_loaders(
  paths: List[str],
  labels: List[str],
  le: LabelEncoder,
  img_size: int = 224,
  batch: int = 32,
  val_frac: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
  """Create train / val loaders with robust splitting and weighted sampling."""
  normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
  )

  train_tf = transforms.Compose(
    [
      transforms.RandomResizedCrop(img_size, scale=(0.7, 1.1)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(25),
      transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
      transforms.ToTensor(),
      normalize,
    ]
  )
  val_tf = transforms.Compose(
    [
      transforms.Resize(int(img_size * 1.15)),
      transforms.CenterCrop(img_size),
      transforms.ToTensor(),
      normalize,
    ]
  )

  # ── robust split ──────────────────────────
  label_counts = Counter(labels)
  min_count = min(label_counts.values())
  stratify = labels if min_count >= 2 else None
  if stratify is None:
    print("⚠️  Some classes have <2 images – falling back to random split without stratification.")

  p_train, p_val, l_train, l_val = train_test_split(
    paths,
    labels,
    test_size=val_frac,
    random_state=42,
    stratify=stratify,
  )

  train_ds = ChessPieceDataset(p_train, l_train, train_tf, le)
  val_ds = ChessPieceDataset(p_val, l_val, val_tf, le)

  # ── weighted sampler to handle imbalance ──
  class_counts = np.bincount(train_ds.encoded_labels, minlength=len(le.classes_))
  class_weights = np.where(class_counts > 0, 1.0 / class_counts, 1.0)
  sample_weights = class_weights[train_ds.encoded_labels]
  sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

  train_loader = DataLoader(train_ds, batch_size=batch, sampler=sampler, num_workers=2, pin_memory=True)
  val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True)

  return train_loader, val_loader


def train(
  model: nn.Module,
  train_loader: DataLoader,
  val_loader: DataLoader,
  device: torch.device,
  epochs: int = 20,
  patience: int = 6,
):
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)

  scaler = torch.amp.GradScaler(device="cuda")
  best_acc, wait = 0.0, 0

  for epoch in range(1, epochs + 1):
    # ── training ───────────────────────────
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
      x, y = x.to(device), y.to(device)
      optimizer.zero_grad(set_to_none=True)
      with torch.amp.autocast(device_type="cuda"):
        logits = model(x)
        loss = criterion(logits, y)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()

      running_loss += loss.item() * y.size(0)
      pred = logits.argmax(1)
      correct += (pred == y).sum().item()
      total += y.size(0)

    train_acc = 100 * correct / total
    train_loss = running_loss / total

    # ── validation ─────────────────────────
    model.eval()
    correct, total = 0, 0
    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
      for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    val_acc = 100 * correct / total if total else 0.0
    scheduler.step(val_acc)

    print(
      f"Epoch {epoch:02d}/{epochs}  "
      f"Train-loss: {train_loss:.4f}  Train-acc: {train_acc:.1f}%  "
      f"Val-acc: {val_acc:.1f}%  LR: {optimizer.param_groups[0]['lr']:.1e}"
    )

    # early-stopping
    if val_acc > best_acc:
      best_acc, wait = val_acc, 0
      torch.save(model.state_dict(), "models/best_resnet18.pth")
    else:
      wait += 1
      if wait >= patience:
        print(f"Early stopping — best val acc {best_acc:.2f}%")
        break

  # load best if checkpoint exists
  checkpoint_path = "models/best_resnet18.pth"
  if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
  return best_acc


# ───────────────────────────── MAIN ────────────────────────────────


def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using {device}")

  paths, labels, le = load_chess_pieces_data()
  print(f"{len(paths)} images  |  classes: {list(le.classes_)}")

  train_loader, val_loader = build_loaders(paths, labels, le)

  net = ChessPieceNet(num_classes=len(le.classes_)).to(device)
  best = train(net, train_loader, val_loader, device)

  model_data = {
    "model_state_dict": net.state_dict(),
    "label_mapping": {i: c for i, c in enumerate(le.classes_)},
    "best_val_acc": best,
  }

  torch.save(model_data, "models/model_latest.pth")
  print(f"✅ finished — best val accuracy {best:.2f}%")


if __name__ == "__main__":
  main()
