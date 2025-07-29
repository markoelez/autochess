#!/usr/bin/env python3
"""
Analyze chess board image and predict best move.
Combines board_state.py and predict.py functionality.
"""

import os
import sys
import argparse

from .predict import StockfishEngine, complete_fen
from .board_state import board_to_string, analyze_board_state


def main():
  parser = argparse.ArgumentParser(description="Analyze chess board and predict best move")
  parser.add_argument("image_path", help="Path to chess board image")
  parser.add_argument("turn", choices=["w", "b"], help="Whose turn (w=white, b=black)")
  parser.add_argument("--model", default="models/model_latest.pth", help="Path to classifier model")
  parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish", help="Path to Stockfish")
  parser.add_argument("--depth", type=int, default=15, help="Search depth")
  parser.add_argument("--confidence", type=float, default=0.5, help="Piece detection confidence threshold")
  parser.add_argument("--show-board", action="store_true", help="Show detected board state")
  parser.add_argument("--evaluate", action="store_true", help="Show position evaluation")

  args = parser.parse_args()

  if not os.path.exists(args.image_path):
    print(f"Error: Image file not found: {args.image_path}")
    return 1

  if not os.path.exists(args.model):
    print(f"Error: Model file not found: {args.model}")
    return 1

  try:
    # Analyze board
    print(f"Analyzing chess board from image: {args.image_path}")
    board_matrix, confidence_matrix = analyze_board_state(args.image_path, args.model, args.confidence)

    # Convert to FEN
    fen_position = board_to_string(board_matrix, notation="fen")

    # Complete FEN with turn and default values
    full_fen = f"{fen_position} {args.turn}"
    full_fen = complete_fen(full_fen)

    if args.show_board:
      print("\nDetected board state:")
      print("=" * 40)
      print(board_to_string(board_matrix, notation="algebraic"))
      print(f"\nFEN: {fen_position}")

    # Initialize Stockfish
    print(f"\nInitializing Stockfish engine...")
    engine = StockfishEngine(args.stockfish, args.depth)

    # Set position and get best move
    engine.set_position(full_fen)
    turn = "White" if args.turn == "w" else "Black"
    print(f"Turn: {turn}")

    print(f"\nCalculating best move (depth {args.depth})...")
    best_move, ponder = engine.get_best_move()

    if best_move:
      print(f"Best move: {best_move}")
      if ponder:
        print(f"Expected response: {ponder}")
    else:
      print("No legal moves available")

    # Show evaluation if requested
    if args.evaluate:
      print("\nPosition evaluation:")
      eval_info = engine.get_evaluation()

      if eval_info["mate"] is not None:
        print(f"Mate in {abs(eval_info['mate'])} moves")
      else:
        score = eval_info["score"]
        # Adjust score perspective for black
        if args.turn == "b":
          score = -score
        print(f"Score: {score:+.2f} pawns")

      print(f"Depth: {eval_info['depth']}")
      if eval_info["pv"]:
        print(f"Principal variation: {' '.join(eval_info['pv'][:5])}")

    engine.close()

  except Exception as e:
    print(f"Error: {e}")
    return 1

  return 0


if __name__ == "__main__":
  sys.exit(main())
