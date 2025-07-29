#!/usr/bin/env python3
import os
import sys
import argparse
import tempfile
from typing import Optional

import cv2
import numpy as np
import torch

from autochess.segment import ChessBoardSplitter
from autochess.inference import load_model
from autochess.stockfish import SimpleStockfish
from autochess.board_state import board_to_string
from autochess.screen_capture import ChessBoardDetector
from autochess.overlay_selector import OverlaySelector


def preprocess_piece(cell_img: np.ndarray) -> np.ndarray:
  """Extract piece from background: black outline for white pieces, filled black for black pieces."""
  gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (5, 5), 0)
  thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
  contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  result = np.ones_like(cell_img) * 255  # White background

  significant_contours = []
  if contours:
    for contour in contours:
      area = cv2.contourArea(contour)
      if area > 300:
        x, y, w, h = cv2.boundingRect(contour)
        cell_h, cell_w = gray.shape
        if x > 2 and y > 2 and (x + w) < (cell_w - 2) and (y + h) < (cell_h - 2):
          significant_contours.append(contour)

  if not significant_contours:
    return result

  largest_contour = max(significant_contours, key=cv2.contourArea)

  mask = np.zeros_like(gray)
  cv2.drawContours(mask, [largest_contour], -1, 255, -1)

  piece_pixels = gray[mask > 0]
  if len(piece_pixels) == 0:
    return result

  avg_intensity = np.mean(piece_pixels)

  if avg_intensity < 127:
    cv2.drawContours(result, [largest_contour], -1, (0, 0, 0), -1)  # Filled black
  else:
    cv2.drawContours(result, [largest_contour], -1, (0, 0, 0), 4)  # Black outline, thickness 4

  return result


def analyze_board_correct(image_path: str, model_path: str, save_debug: bool = False):
  """Analyze board with correct preprocessing."""
  debug_dir = "/tmp/chess_debug"
  if save_debug:
    os.makedirs(debug_dir, exist_ok=True)

  model, label_mapping = load_model(model_path)

  splitter = ChessBoardSplitter()
  img = splitter.load_image(image_path)

  with tempfile.TemporaryDirectory() as temp_dir:
    cells = splitter.split_board_into_cells(img, output_dir=temp_dir, preprocess=False)

    board_matrix = [["empty" for _ in range(8)] for _ in range(8)]
    confidence_matrix = [[0.0 for _ in range(8)] for _ in range(8)]

    low_confidence_cells = []

    for idx, cell_info in enumerate(cells):
      row = cell_info["row"]
      col = cell_info["col"]

      chess_notation = f"{chr(ord('a') + col)}{8 - row}"

      from PIL import Image

      cell_image = cell_info["image"]
      preprocessed_cell = preprocess_piece(cell_image)

      preprocessed_pil = Image.fromarray(cv2.cvtColor(preprocessed_cell, cv2.COLOR_BGR2RGB))

      if save_debug:
        preprocessed_path = f"{debug_dir}/{chess_notation}_1_preprocessed.png"
        cv2.imwrite(preprocessed_path, preprocessed_cell)

        import torchvision
        from torchvision import transforms

        preprocessed_pil.save(f"{debug_dir}/{chess_notation}_2_pil_ready.png")

        img_size = 224

        resize_step = transforms.Resize(int(img_size * 1.15))
        resized_img = resize_step(preprocessed_pil)
        resized_img.save(f"{debug_dir}/{chess_notation}_3_resized.png")

        crop_step = transforms.CenterCrop(img_size)
        cropped_img = crop_step(resized_img)
        cropped_img.save(f"{debug_dir}/{chess_notation}_4_cropped.png")

        to_tensor_step = transforms.ToTensor()
        tensor_unnorm = to_tensor_step(cropped_img)

        torchvision.utils.save_image(tensor_unnorm, f"{debug_dir}/{chess_notation}_5_tensor_unnorm.png")

        normalize_step = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor_norm = normalize_step(tensor_unnorm)

        denorm = tensor_norm.clone()
        for t, m, s in zip(denorm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
          t.mul_(s).add_(m)
        denorm = torch.clamp(denorm, 0, 1)
        torchvision.utils.save_image(denorm, f"{debug_dir}/{chess_notation}_6_tensor_norm.png")

        tensor = tensor_norm.unsqueeze(0)

        final_denorm = tensor.squeeze(0).clone()
        for t, m, s in zip(final_denorm, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
          t.mul_(s).add_(m)
        final_denorm = torch.clamp(final_denorm, 0, 1)
        torchvision.utils.save_image(final_denorm, f"{debug_dir}/{chess_notation}_7_final_model_input.png")

      else:
        img_size = 224
        from torchvision import transforms

        transform = transforms.Compose(
          [
            transforms.Resize(int(img_size * 1.15)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ]
        )

        tensor = transform(preprocessed_pil).unsqueeze(0)

      with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

      predicted_label = label_mapping[predicted_class]
      board_matrix[row][col] = predicted_label.replace("blank", "empty")
      confidence_matrix[row][col] = confidence

      if confidence < 0.8:
        low_confidence_cells.append((chess_notation, predicted_label, confidence))

    fen = board_to_string(board_matrix, notation="fen")

    return fen, board_matrix, confidence_matrix, low_confidence_cells


def print_board(board_matrix):
  symbols = {
    "empty": ".",
    "white_pawn": "P",
    "white_knight": "N",
    "white_bishop": "B",
    "white_rook": "R",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_knight": "n",
    "black_bishop": "b",
    "black_rook": "r",
    "black_queen": "q",
    "black_king": "k",
  }

  print("\nBoard Position:")
  print("+---+---+---+---+---+---+---+---+")
  for row_idx, row in enumerate(board_matrix):
    print("|", end="")
    for piece in row:
      symbol = symbols.get(piece, "?")
      print(f" {symbol} |", end="")
    print(f" {8 - row_idx}")
    print("+---+---+---+---+---+---+---+---+")
  print("  a   b   c   d   e   f   g   h ")


def prompt_castling_rights() -> str:
  """
  Ask the user for castling rights and return a validated string.

  Valid inputs are any combination of the letters K Q k q in *canonical*
  order (KQkq) or “-” when neither side can castle.
  """
  rights_list = ["-"]
  order = "KQkq"
  for mask in range(1, 16):  # 1 … 15
    rights = "".join(ch for i, ch in enumerate(order) if mask & (1 << i))
    rights_list.append(rights)

  print("\nValid castling-rights strings:")
  print(", ".join(rights_list))
  valid_set = set(rights_list)
  while True:
    answer = input("\nCastling rights: ").strip() or "-"
    if answer in valid_set:
      return answer
    print("  ✘  Invalid input. Please choose one from the list above.")


def prompt_turn() -> str:
  while True:
    ans = input("\nWhose turn is it? (w/b): ").strip().lower()
    if ans in ["w", "b"]:
      return ans
    print("  ✘  Invalid input. Please enter 'w' or 'b'.")


def display_moves(fen: str, castling: str, stockfish_path: str, depth: int = 10, turn: str = "w") -> Optional[str]:
  """Calculate and display best moves for both colours, with isolated engines to handle failures."""
  print("\nBEST MOVES:")
  print("-" * 10)

  white_best = None
  black_best = None

  for color, t in [("White", "w"), ("Black", "b")]:
    try:
      engine = SimpleStockfish(stockfish_path, depth)
      engine.new_game()  # clear state

      full_fen = f"{fen} {t} {castling} - 0 1"

      engine.set_position(full_fen)
      top_moves = engine.get_top_moves(5)
      if top_moves:
        print(f"\n{color} top 5 moves:")
        print("-" * 3)
        for i, info in enumerate(top_moves, 1):
          move = info["move"]
          if info["type"] == "mate":
            mate = info["mate"]
            score_str = f"Mate in {mate}" if mate > 0 else f"Gets mated in {-mate}"
          else:
            score_str = f"{info['score']:+.2f}"
          print(f"{i}. {move} (eval: {score_str})")
        best = top_moves[0]["move"]
        if color == "White":
          white_best = best
        else:
          black_best = best
      else:
        print(f"\n{color}: No moves found")
    except Exception as e:
      print(f"\n{color}: Error - {e}")
    finally:
      if "engine" in locals():
        engine.close()

  print("-" * 30)

  if turn == "w":
    return white_best
  else:
    return black_best


def main():
  parser = argparse.ArgumentParser(description="Chess analyzer with correct preprocessing")
  parser.add_argument("--model", default="models/model_latest.pth", help="Path to classifier model")
  parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish", help="Path to Stockfish")
  parser.add_argument("--depth", type=int, default=10, help="Search depth")
  parser.add_argument("--save-debug", action="store_true", help="Save debug images to /tmp/chess_debug")

  args = parser.parse_args()

  print("AutoChess Analyzer")
  print("==================\n")

  if not os.path.exists(args.model):
    print(f"Error: Model not found: {args.model}")
    return 1

  if not os.path.exists(args.stockfish):
    print(f"Error: Stockfish not found: {args.stockfish}")
    return 1

  try:
    print("Board Selection")
    print("---------------")
    selector = OverlaySelector()
    detector = ChessBoardDetector()

    print("Click and drag to select the chess board area...")
    board_rect = selector.select_area()

    if not board_rect:
      print("Selection cancelled")
      return 1

    if len(board_rect) == 5:
      x, y, w, h, monitor_idx = board_rect
      board_rect = (x, y, w, h)
    else:
      x, y, w, h = board_rect
      monitor_idx = 1

    print(f"Selected region: {w}x{h} at position ({x}, {y})")

    print("\nPress ENTER to analyze the current position, or Ctrl+C to exit.")

    while True:
      input()

      print("\nAnalyzing current position...")
      screen = detector.capture_screen(monitor_idx)
      board_image = detector.extract_board_image(screen, board_rect)

      temp_path = "/tmp/chess_current.png"
      cv2.imwrite(temp_path, board_image)

      fen, board_matrix, confidence_matrix, low_confidence_cells = analyze_board_correct(temp_path, args.model, save_debug=args.save_debug)

      print_board(board_matrix)
      print(f"\nFEN: {fen}")

      castling = prompt_castling_rights()

      turn = prompt_turn()

      best_move_uci = display_moves(fen, castling, args.stockfish, args.depth, turn)

      if best_move_uci:
        from_square = best_move_uci[:2]
        to_square = best_move_uci[2:4]
        from autochess.move_overlay import ShowMoveOverlay

        shower = ShowMoveOverlay()
        shower.show_on_monitor(monitor_idx, board_rect, from_square, to_square)

      # Removed duplicate: display_moves(fen, castling, args.stockfish, args.depth)

      print("\nPress ENTER to analyze again, or Ctrl+C to exit.")

  except KeyboardInterrupt:
    print("\n\nExiting...")
  except Exception as e:
    print(f"\nError: {e}")
    import traceback

    traceback.print_exc()
    return 1

  return 0


if __name__ == "__main__":
  sys.exit(main())
