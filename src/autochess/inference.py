#!/usr/bin/env python3
"""
Chess piece inference script - loads a trained model and classifies chess piece images.
"""

import os
import argparse
from typing import Dict, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


class ChessPieceNet(nn.Module):
  """Fine-tuned ResNet-18 for chess piece classification."""

  def __init__(self, num_classes: int):
    super().__init__()
    from torchvision import models

    self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Replace final FC with Sequential (matching classifier.py)
    in_feats = self.backbone.fc.in_features
    self.backbone.fc = nn.Sequential(
      nn.Dropout(p=0.4),
      nn.Linear(in_feats, num_classes),
    )

  def forward(self, x):
    return self.backbone(x)


def load_model(model_path: str = "models/model_latest.pth") -> Tuple[ChessPieceNet, Dict[int, str]]:
  """Load trained model and label mapping."""
  if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

  checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
  label_mapping = checkpoint["label_mapping"]

  model = ChessPieceNet(num_classes=len(label_mapping))
  model.load_state_dict(checkpoint["model_state_dict"])
  model.eval()

  return model, label_mapping


def preprocess_image(image_path: str) -> torch.Tensor:
  """Preprocess image for inference - matching validation transform from training."""
  img_size = 224
  transform = transforms.Compose(
    [
      transforms.Resize(int(img_size * 1.15)),  # Resize to ~256
      transforms.CenterCrop(img_size),  # Center crop to 224
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
  )

  image = Image.open(image_path).convert("RGB")
  return transform(image).unsqueeze(0)  # Add batch dimension


def predict(model: ChessPieceNet, image_tensor: torch.Tensor, label_mapping: Dict[int, str]) -> Tuple[str, float]:
  """Make prediction on preprocessed image."""
  with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

  predicted_label = label_mapping[predicted_class]
  return predicted_label, confidence


def main():
  parser = argparse.ArgumentParser(description="Chess piece inference")
  parser.add_argument("image_path", help="Path to chess piece image")
  parser.add_argument("--model", default="models/model_latest.pth", help="Path to model file")
  args = parser.parse_args()

  if not os.path.exists(args.image_path):
    print(f"Error: Image file not found: {args.image_path}")
    return

  try:
    # Load model
    print(f"Loading model from {args.model}...")
    model, label_mapping = load_model(args.model)

    # Preprocess image
    print(f"Processing image: {args.image_path}")
    image_tensor = preprocess_image(args.image_path)

    # Make prediction
    predicted_label, confidence = predict(model, image_tensor, label_mapping)

    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.2%}")

  except Exception as e:
    print(f"Error during inference: {e}")


if __name__ == "__main__":
  main()
