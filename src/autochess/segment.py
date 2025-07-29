from typing import Tuple
from pathlib import Path

import cv2
import numpy as np


class ChessBoardSplitter:
  def __init__(self, board_size: int = 8):
    self.board_size = board_size
    self.cells = []
    self.piece_map = {
      "empty": "",
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

  def load_image(self, image_path: str) -> np.ndarray:
    """Load chess board image"""
    img = cv2.imread(image_path)
    if img is None:
      raise ValueError(f"Could not load image from {image_path}")
    return img

  def preprocess_piece(self, cell_img: np.ndarray) -> np.ndarray:
    """Extract piece from background and convert to pure black/white"""
    # Convert to grayscale for processing
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to separate piece from background
    # This works well for chess pieces on alternating colored squares
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours to identify the piece
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the piece
    mask = np.zeros_like(gray)

    # Filter contours to find significant ones (pieces)
    significant_contours = []
    if contours:
      for contour in contours:
        area = cv2.contourArea(contour)
        # Increase minimum area threshold and add shape checks
        if area > 500:  # Increased threshold to filter out noise
          # Check if contour is reasonably centered (not edge artifacts)
          x, y, w, h = cv2.boundingRect(contour)
          cell_h, cell_w = gray.shape
          # Ensure contour is not touching edges too much
          if x > 5 and y > 5 and (x + w) < (cell_w - 5) and (y + h) < (cell_h - 5):
            significant_contours.append(contour)

    # If no significant contours found, return blank white image
    if not significant_contours:
      result = np.ones_like(cell_img) * 255
      return result

    # Find the largest significant contour
    largest_contour = max(significant_contours, key=cv2.contourArea)
    cv2.drawContours(mask, [largest_contour], -1, (255,), -1)

    # Determine if piece is black or white based on average intensity
    piece_pixels = gray[mask > 0]
    if len(piece_pixels) > 0:
      avg_intensity = np.mean(piece_pixels)

      # Create output image
      result = np.ones_like(cell_img) * 255  # White background

      # If dark piece (black), make it pure black
      if avg_intensity < 127:
        result[mask > 0] = [0, 0, 0]  # Pure black
      else:
        # If light piece (white), make it more visible with a stronger outline
        # Add thicker, black outline for better visibility
        contour_mask = np.zeros_like(gray)
        cv2.drawContours(contour_mask, [largest_contour], -1, (255,), 4)  # Thicker outline
        result[contour_mask > 0] = [0, 0, 0]  # Black outline
    else:
      # No piece detected, return white image
      result = np.ones_like(cell_img) * 255

    return result

  def detect_board_corners(self, img: np.ndarray) -> Tuple[int, int, int, int]:
    """Detect the chess board boundaries"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
      largest_contour = max(contours, key=cv2.contourArea)
      x, y, w, h = cv2.boundingRect(largest_contour)
      return x, y, w, h
    else:
      h, w = img.shape[:2]
      return 0, 0, w, h

  def split_board_into_cells(self, img: np.ndarray, output_dir: str = "chess_cells", preprocess: bool = True):
    """Split the chess board into 64 individual cells"""
    h, w = img.shape[:2]

    cell_height = h // self.board_size
    cell_width = w // self.board_size

    Path(output_dir).mkdir(exist_ok=True)

    # Create subdirectories for original and preprocessed images
    original_dir = Path(output_dir) / "original"
    preprocessed_dir = Path(output_dir) / "preprocessed"
    original_dir.mkdir(exist_ok=True)
    preprocessed_dir.mkdir(exist_ok=True)

    cells_data = []

    for row in range(self.board_size):
      for col in range(self.board_size):
        y1 = row * cell_height
        y2 = (row + 1) * cell_height
        x1 = col * cell_width
        x2 = (col + 1) * cell_width

        cell = img[y1:y2, x1:x2]

        chess_notation = f"{chr(ord('a') + col)}{8 - row}"

        # Save original cell
        filename = f"cell_{chess_notation}_row{row}_col{col}.png"
        original_filepath = original_dir / filename
        cv2.imwrite(str(original_filepath), cell)

        # Save preprocessed cell if enabled
        if preprocess:
          preprocessed_cell = self.preprocess_piece(cell)
          preprocessed_filepath = preprocessed_dir / filename
          cv2.imwrite(str(preprocessed_filepath), preprocessed_cell)

        cells_data.append(
          {
            "file": filename,
            "position": chess_notation,
            "row": row,
            "col": col,
            "image": cell,
            "original_path": str(original_filepath),
            "preprocessed_path": str(preprocessed_filepath) if preprocess else None,
          }
        )

    self.cells = cells_data
    return cells_data

  def visualize_grid(self, img: np.ndarray, save_path: str = "grid_overlay.png"):
    """Draw grid overlay on the original image for verification"""
    img_copy = img.copy()
    h, w = img_copy.shape[:2]

    cell_height = h // self.board_size
    cell_width = w // self.board_size

    for i in range(1, self.board_size):
      cv2.line(img_copy, (i * cell_width, 0), (i * cell_width, h), (0, 255, 0), 2)
      cv2.line(img_copy, (0, i * cell_height), (w, i * cell_height), (0, 255, 0), 2)

    for row in range(self.board_size):
      for col in range(self.board_size):
        chess_notation = f"{chr(ord('a') + col)}{8 - row}"
        x = col * cell_width + 5
        y = row * cell_height + 20
        cv2.putText(img_copy, chess_notation, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imwrite(save_path, img_copy)
    return img_copy

  def get_piece_label(self, row: int, col: int) -> str:
    """Get piece label assuming standard starting position with black on top."""
    if row == 0:
      if col in [0, 7]:
        return "black_rook"
      elif col in [1, 6]:
        return "black_knight"
      elif col in [2, 5]:
        return "black_bishop"
      elif col == 3:
        return "black_queen"
      elif col == 4:
        return "black_king"
    elif row == 1:
      return "black_pawn"
    elif row == 6:
      return "white_pawn"
    elif row == 7:
      if col in [0, 7]:
        return "white_rook"
      elif col in [1, 6]:
        return "white_knight"
      elif col in [2, 5]:
        return "white_bishop"
      elif col == 3:
        return "white_queen"
      elif col == 4:
        return "white_king"
    else:
      return "empty"

  def augment_cell(self, cell: np.ndarray) -> np.ndarray:
    """Apply random augmentations to the cell image."""
    # Random brightness
    factor = np.random.uniform(0.8, 1.2)
    augmented = np.clip(cell * factor, 0, 255).astype(np.uint8)

    # Random contrast
    contrast = np.random.uniform(0.8, 1.2)
    mean = augmented.mean()
    augmented = np.clip((augmented - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # Slight rotation
    angle = np.random.uniform(-5, 5)
    h, w = augmented.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    augmented = cv2.warpAffine(augmented, M, (w, h), borderValue=(255, 255, 255))

    # Add Gaussian noise
    sigma = np.random.uniform(0, 10)
    noise = np.random.normal(0, sigma, augmented.shape).astype(np.float32)
    augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)

    # Random horizontal flip
    if np.random.random() < 0.5:
      augmented = cv2.flip(augmented, 1)

    return augmented

  def create_labeled_dataset(self, cells: list, output_dir: str = "labeled_dataset", num_augmentations: int = 50):
    """Create labeled dataset with augmentations."""
    base_path = Path(output_dir)
    base_path.mkdir(exist_ok=True)

    for piece_type in self.piece_map.keys():
      piece_dir = base_path / piece_type
      piece_dir.mkdir(exist_ok=True)

    for cell_info in cells:
      row = cell_info["row"]
      col = cell_info["col"]
      label = self.get_piece_label(row, col)
      label_dir = base_path / label
      position = cell_info["position"]

      # Save original preprocessed
      original_cell = cell_info["image"]
      preprocessed = self.preprocess_piece(original_cell)
      cv2.imwrite(str(label_dir / f"{position}_original.png"), preprocessed)

      # Generate and save augmented versions
      for i in range(num_augmentations):
        augmented_cell = self.augment_cell(original_cell)
        aug_preprocessed = self.preprocess_piece(augmented_cell)
        cv2.imwrite(str(label_dir / f"{position}_aug_{i}.png"), aug_preprocessed)

    return str(base_path)


def main():
  splitter = ChessBoardSplitter()

  img_path = "dat/alt.png"
  img = splitter.load_image(img_path)

  print(f"Image loaded: {img.shape}")

  cells = splitter.split_board_into_cells(img, "dat/chess_cells")
  print(f"Split board into {len(cells)} cells")

  splitter.visualize_grid(img, "dat/chess_board_grid.png")
  print("Created grid overlay visualization")

  dataset_path = splitter.create_labeled_dataset(cells, "dat/labeled_dataset")
  print(f"Created labeled dataset structure at: {dataset_path}")

  print("\nNext steps:")
  print("1. Review the cells in 'dat/chess_cells' directory")
  print("2. Use the labeled dataset in 'dat/labeled_dataset' to train your chess piece classifier")


if __name__ == "__main__":
  main()
