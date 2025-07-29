#!/usr/bin/env python3
"""
Chess move predictor using Stockfish engine.
Takes a board state (FEN notation) and returns the best move.
"""

import os
import sys
import argparse
import subprocess
from typing import Tuple, Optional


class StockfishEngine:
  """Wrapper for Stockfish chess engine."""

  def __init__(self, stockfish_path: str = "/opt/homebrew/bin/stockfish", depth: int = 15):
    """Initialize Stockfish engine.

    Args:
        stockfish_path: Path to Stockfish executable
        depth: Search depth for engine analysis
    """
    self.stockfish_path = stockfish_path
    self.depth = depth
    self.process = None

    # Check if Stockfish exists
    if not os.path.exists(stockfish_path):
      raise FileNotFoundError(f"Stockfish not found at {stockfish_path}")

    self._start_engine()

  def _start_engine(self):
    """Start Stockfish process."""
    try:
      self.process = subprocess.Popen(
        [self.stockfish_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1
      )

      # Initialize UCI protocol
      self._send_command("uci")
      output = self._read_output("uciok")
      if "uciok" not in output:
        raise RuntimeError("Failed to initialize UCI protocol")

    except Exception as e:
      raise RuntimeError(f"Failed to start Stockfish: {e}")

  def _send_command(self, command: str):
    """Send command to Stockfish."""
    if self.process and self.process.stdin:
      self.process.stdin.write(f"{command}\n")
      self.process.stdin.flush()

  def _read_output(self, until: str = None) -> str:
    """Read output from Stockfish."""
    output_lines = []
    if self.process and self.process.stdout:
      while True:
        line = self.process.stdout.readline().strip()
        if line:
          output_lines.append(line)
          if until and line == until:
            break
          elif not until and (line == "uciok" or line == "readyok" or line.startswith("bestmove")):
            break
    return "\n".join(output_lines)

  def _wait_for_ready(self):
    """Wait for engine to be ready."""
    self._send_command("isready")
    output = self._read_output("readyok")
    if "readyok" not in output:
      raise RuntimeError("Stockfish engine not ready")

  def set_position(self, fen: str):
    """Set board position using FEN notation."""
    self._send_command(f"position fen {fen}")

  def set_multipv(self, value: int):
    """Set MultiPV option."""
    self._send_command(f"setoption name MultiPV value {value}")

  def get_top_moves(self, num: int = 5, time_limit: Optional[int] = None) -> list[dict]:
    """Get top N moves with scores.

    Returns:
        List of dicts with 'move', 'type' ('cp' or 'mate'), 'score' or 'mate'
    """
    self.set_multipv(num)

    if time_limit:
      self._send_command(f"go movetime {time_limit}")
    else:
      self._send_command(f"go depth {self.depth}")

    output = self._read_output()

    pv_dict = {}

    for line in output.splitlines():
      if not line.startswith("info "):
        continue

      parts = line.split()
      multipv = 1
      depth = 0
      score = None
      mate = None
      pv = []

      i = 1  # after 'info'
      while i < len(parts):
        word = parts[i]
        if word == "depth":
          depth = int(parts[i + 1])
          i += 2
        elif word == "multipv":
          multipv = int(parts[i + 1])
          i += 2
        elif word == "score":
          i += 1
          if i < len(parts):
            if parts[i] == "cp":
              if i + 1 < len(parts):
                score = int(parts[i + 1]) / 100.0
                mate = None
                i += 2
            elif parts[i] == "mate":
              if i + 1 < len(parts):
                mate = int(parts[i + 1])
                score = None
                i += 2
            else:
              i += 1
        elif word == "pv":
          i += 1
          pv = parts[i:]
          i = len(parts)
        else:
          i += 1

      if pv:
        pv_dict[multipv] = {"depth": depth, "score": score, "mate": mate, "pv": pv}

    top = []
    for k in range(1, num + 1):
      if k in pv_dict and pv_dict[k]["pv"]:
        info = pv_dict[k]
        move = info["pv"][0]
        if info["mate"] is not None:
          top.append({"move": move, "type": "mate", "mate": info["mate"]})
        elif info["score"] is not None:
          top.append({"move": move, "type": "cp", "score": info["score"]})

    return top

  def get_best_move(self, time_limit: Optional[int] = None) -> Tuple[str, Optional[str]]:
    """Get best move for current position.

    Args:
        time_limit: Time limit in milliseconds (optional)

    Returns:
        Tuple of (best_move, ponder_move)
    """
    if time_limit:
      self._send_command(f"go movetime {time_limit}")
    else:
      self._send_command(f"go depth {self.depth}")

    output = self._read_output()

    # Parse best move from output
    for line in output.split("\n"):
      if line.startswith("bestmove"):
        parts = line.split()
        best_move = parts[1] if len(parts) > 1 else None
        ponder_move = parts[3] if len(parts) > 3 and parts[2] == "ponder" else None
        return best_move, ponder_move

    return None, None

  def get_evaluation(self) -> dict:
    """Get position evaluation."""
    self._send_command(f"go depth {self.depth}")
    output = self._read_output()

    evaluation = {"depth": 0, "score": 0, "mate": None, "pv": []}

    # Parse evaluation from output
    for line in output.split("\n"):
      if line.startswith("info depth"):
        parts = line.split()
        if "depth" in parts:
          idx = parts.index("depth")
          evaluation["depth"] = int(parts[idx + 1])

        if "score" in parts:
          idx = parts.index("score")
          if idx + 2 < len(parts):
            if parts[idx + 1] == "cp":
              evaluation["score"] = int(parts[idx + 2]) / 100  # Centipawns to pawns
            elif parts[idx + 1] == "mate":
              evaluation["mate"] = int(parts[idx + 2])

        if "pv" in parts:
          idx = parts.index("pv")
          evaluation["pv"] = parts[idx + 1 :]

    return evaluation

  def close(self):
    """Close Stockfish process."""
    if self.process:
      self._send_command("quit")
      self.process.terminate()
      self.process.wait()
      self.process = None

  def __del__(self):
    """Cleanup on deletion."""
    self.close()


def validate_fen(fen: str) -> bool:
  """Basic FEN validation."""
  parts = fen.split()

  # FEN should have 6 parts (position, turn, castling, en passant, halfmove, fullmove)
  # But we'll accept partial FEN with at least position and turn
  if len(parts) < 2:
    return False

  # Check board position (8 ranks separated by /)
  ranks = parts[0].split("/")
  if len(ranks) != 8:
    return False

  # Check turn indicator
  if parts[1] not in ["w", "b"]:
    return False

  return True


def complete_fen(fen: str) -> str:
  """Complete partial FEN notation with default values."""
  parts = fen.split()

  # Ensure we have all 6 parts of FEN
  while len(parts) < 6:
    if len(parts) == 2:  # Add castling rights
      parts.append("KQkq")
    elif len(parts) == 3:  # Add en passant
      parts.append("-")
    elif len(parts) == 4:  # Add halfmove clock
      parts.append("0")
    elif len(parts) == 5:  # Add fullmove number
      parts.append("1")

  return " ".join(parts)


def get_fen_for_player(original_fen: str, turn: str) -> str:
  """Get FEN for a specific player to move, resetting en passant."""
  parts = original_fen.split()
  parts[1] = turn
  parts[3] = "-"
  return " ".join(parts)


def main():
  parser = argparse.ArgumentParser(description="Predict best chess move using Stockfish")
  parser.add_argument("fen", help="Board position in FEN notation")
  parser.add_argument("--depth", type=int, default=15, help="Search depth (default: 15)")
  parser.add_argument("--time", type=int, help="Time limit in milliseconds")
  parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish", help="Path to Stockfish executable")
  parser.add_argument("--evaluate", action="store_true", help="Show position evaluation")

  args = parser.parse_args()

  # Validate and complete FEN
  if not validate_fen(args.fen):
    print("Error: Invalid FEN notation")
    print("FEN format: <position> <turn> [castling] [en passant] [halfmove] [fullmove]")
    print("Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    return 1

  fen = complete_fen(args.fen)

  try:
    # Initialize Stockfish
    print(f"Initializing Stockfish engine...")
    engine = StockfishEngine(args.stockfish, args.depth)

    # Set position
    print(f"Position: {args.fen}")
    print()

    print(f"Calculating top 5 moves for each player (depth {args.depth})...")
    for player, turn_char in [("White", "w"), ("Black", "b")]:
      player_fen = get_fen_for_player(fen, turn_char)
      engine.set_position(player_fen)
      top_moves = engine.get_top_moves(5, args.time)
      print(f"\n{player} top 5 moves:")
      for i, info in enumerate(top_moves, 1):
        move = info["move"]
        if info["type"] == "cp":
          score_str = f"{info['score']:+.2f}"
        else:
          mate = info["mate"]
          if mate > 0:
            score_str = f"Mate in {mate}"
          else:
            score_str = f"Gets mated in {-mate}"
        print(f"{i}. {move} (score: {score_str})")

    # Show evaluation if requested
    if args.evaluate:
      engine.set_multipv(1)
      engine.set_position(fen)
      print("\nPosition evaluation (for original turn):")
      eval_info = engine.get_evaluation()

      if eval_info["mate"] is not None:
        mate = eval_info["mate"]
        if mate > 0:
          print(f"Mate in {mate} moves")
        else:
          print(f"Gets mated in {-mate} moves")
      else:
        score = eval_info["score"]
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
