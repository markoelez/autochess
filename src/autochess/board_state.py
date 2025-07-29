#!/usr/bin/env python3
"""
Chess board state analyzer - segments a chess board image and classifies each square.
"""

import os
import argparse
import tempfile
from typing import List, Tuple
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .segment import ChessBoardSplitter
from .inference import ChessPieceNet, load_model


def preprocess_cell_for_classifier(cell_img: np.ndarray) -> torch.Tensor:
  """Preprocess a cell image for the chess piece classifier."""
  # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
  cell_rgb = cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)
  pil_image = Image.fromarray(cell_rgb)

  # Same preprocessing as training
  transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
  )

  return transform(pil_image).unsqueeze(0)  # Add batch dimension


def classify_cell(model: ChessPieceNet, cell_img: np.ndarray, label_mapping: dict) -> Tuple[str, float]:
  """Classify a single chess board cell."""
  tensor = preprocess_cell_for_classifier(cell_img)

  with torch.no_grad():
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()

  predicted_label = label_mapping[predicted_class]
  return predicted_label, confidence


def board_to_string(board_matrix: List[List[str]], notation: str = "algebraic") -> str:
  """Convert board matrix to string representation."""
  if notation == "algebraic":
    # Standard chess notation with files a-h and ranks 1-8
    result = []
    for rank in range(8):
      rank_str = f"{8 - rank}: "
      for file in range(8):
        piece = board_matrix[rank][file]
        if piece == "empty":
          rank_str += ".. "
        else:
          # Convert to short notation
          piece_map = {
            "white_pawn": "wP",
            "white_rook": "wR",
            "white_knight": "wN",
            "white_bishop": "wB",
            "white_queen": "wQ",
            "white_king": "wK",
            "black_pawn": "bP",
            "black_rook": "bR",
            "black_knight": "bN",
            "black_bishop": "bB",
            "black_queen": "bQ",
            "black_king": "bK",
          }
          rank_str += f"{piece_map.get(piece, '??')} "
      result.append(rank_str)

    # Add file labels
    result.append("   a  b  c  d  e  f  g  h")
    return "\n".join(result)

  elif notation == "fen":
    # FEN (Forsyth-Edwards Notation) style
    fen_map = {
      "white_pawn": "P",
      "white_rook": "R",
      "white_knight": "N",
      "white_bishop": "B",
      "white_queen": "Q",
      "white_king": "K",
      "black_pawn": "p",
      "black_rook": "r",
      "black_knight": "n",
      "black_bishop": "b",
      "black_queen": "q",
      "black_king": "k",
    }

    fen_ranks = []
    for rank in range(8):
      fen_rank = ""
      empty_count = 0
      for file in range(8):
        piece = board_matrix[rank][file]
        if piece == "empty":
          empty_count += 1
        else:
          if empty_count > 0:
            fen_rank += str(empty_count)
            empty_count = 0
          fen_rank += fen_map.get(piece, "?")

      if empty_count > 0:
        fen_rank += str(empty_count)
      fen_ranks.append(fen_rank)

    return "/".join(fen_ranks)

  else:
    raise ValueError(f"Unknown notation: {notation}")


def analyze_board_state(image_path: str, model_path: str, confidence_threshold: float = 0.5, use_preprocessing: bool = True) -> Tuple[List[List[str]], List[List[float]]]:
  """Analyze chess board image and return piece positions with confidence scores."""

  # Load the classifier model
  model, label_mapping = load_model(model_path)

  # Initialize board segmenter
  splitter = ChessBoardSplitter()

  # Load and segment the board image
  img = splitter.load_image(image_path)

  # Create temporary directory for cell extraction
  with tempfile.TemporaryDirectory() as temp_dir:
    cells = splitter.split_board_into_cells(img, output_dir=temp_dir, preprocess=use_preprocessing)

    # Initialize result matrices
    board_matrix = [["empty" for _ in range(8)] for _ in range(8)]
    confidence_matrix = [[0.0 for _ in range(8)] for _ in range(8)]

    # Classify each cell
    for cell_data in cells:
      row, col = cell_data["row"], cell_data["col"]
      
      # Use preprocessed image if available, otherwise use original
      if use_preprocessing and cell_data["preprocessed_path"]:
        # Load preprocessed image
        cell_img = cv2.imread(cell_data["preprocessed_path"])
      else:
        cell_img = cell_data["image"]

      predicted_piece, confidence = classify_cell(model, cell_img, label_mapping)

      # Only assign piece if confidence is above threshold
      if confidence >= confidence_threshold:
        board_matrix[row][col] = predicted_piece
      else:
        board_matrix[row][col] = "empty"

      confidence_matrix[row][col] = confidence

  return board_matrix, confidence_matrix


def main():
  parser = argparse.ArgumentParser(description="Analyze chess board state from image")
  parser.add_argument("image_path", help="Path to chess board image")
  parser.add_argument("--model", default="models/model_latest.pth", help="Path to model file")
  parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for piece detection")
  parser.add_argument("--notation", choices=["algebraic", "fen"], default="algebraic", help="Output notation format")
  parser.add_argument("--show-confidence", action="store_true", help="Show confidence scores")
  parser.add_argument("--no-preprocess", action="store_true", help="Disable cell preprocessing")

  args = parser.parse_args()

  if not os.path.exists(args.image_path):
    print(f"Error: Image file not found: {args.image_path}")
    return

  if not os.path.exists(args.model):
    print(f"Error: Model file not found: {args.model}")
    return

  try:
    print(f"Analyzing chess board: {args.image_path}")
    print(f"Using model: {args.model}")
    print(f"Confidence threshold: {args.confidence}")
    print()

    # Analyze the board
    board_matrix, confidence_matrix = analyze_board_state(
        args.image_path, args.model, args.confidence, use_preprocessing=not args.no_preprocess
    )

    # Output the board state
    board_string = board_to_string(board_matrix, args.notation)
    print("Board State:")
    print("=" * 40)
    print(board_string)

    if args.show_confidence:
      print("\nConfidence Scores:")
      print("=" * 40)
      for rank in range(8):
        rank_str = f"{8 - rank}: "
        for file in range(8):
          conf = confidence_matrix[rank][file]
          rank_str += f"{conf:.2f} "
        print(rank_str)
      print("   a    b    c    d    e    f    g    h")

    if args.notation == "fen":
      print(f"\nFEN: {board_string}")

  except Exception as e:
    print(f"Error during analysis: {e}")


if __name__ == "__main__":
  main()
