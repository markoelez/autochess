#!/usr/bin/env python3
"""
Simple Stockfish wrapper with robust UCI-hand-shaking and timeout handling.
"""

import time
import subprocess
from typing import Dict, List, Optional

ReadyTimeout = 5  # seconds to wait for “readyok”
SearchTimeout = 60  # fail-safe in case the engine hangs


class SimpleStockfish:
  """Robust Stockfish wrapper (UCI)."""

  def __init__(self, path: str = "/opt/homebrew/bin/stockfish", depth: int = 10):
    self.path = path
    self.depth = depth
    self.process: Optional[subprocess.Popen] = None
    self._start()

  def _send(self, cmd: str) -> None:
    """Write a line to the engine and flush immediately."""
    assert self.process and self.process.stdin
    self.process.stdin.write(cmd + "\n")
    self.process.stdin.flush()

  def _wait_for(self, token: str, timeout: float) -> None:
    """Read engine output until a line *starts with* <token>."""
    assert self.process and self.process.stdout
    deadline = time.time() + timeout
    while True:
      if time.time() > deadline:
        raise TimeoutError(f"Engine did not respond with “{token}” in time")
      line = self.process.stdout.readline()
      if not line:  # engine died
        raise RuntimeError("Stockfish process terminated unexpectedly")
      if line.startswith(token):
        break

  def _ready(self) -> None:
    """Block until the engine reports it is ready."""
    self._send("isready")
    self._wait_for("readyok", ReadyTimeout)

  def _start(self) -> None:
    """Launch the engine and finish UCI initialisation."""
    # in _start(...)
    self.process = subprocess.Popen(
      [self.path],
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.DEVNULL,  # avoid filling a pipe we never read
      text=True,
      bufsize=1,
    )
    self._send("uci")
    self._wait_for("uciok", ReadyTimeout)
    self._ready()

  def new_game(self) -> None:
    self._send("ucinewgame")
    self._ready()

  def close(self) -> None:
    if self.process:
      try:
        if self.process.poll() is None:
          self._send("quit")
          self.process.communicate(timeout=1)
      except Exception:
        if self.process and self.process.poll() is None:
          self.process.kill()
      finally:
        self.process = None

  def __del__(self):
    self.close()

  def set_position(self, fen: str) -> None:
    self._send(f"position fen {fen}")
    self._ready()

  def get_top_moves(self, num: int = 5) -> List[Dict]:
    # Ensure MultiPV is set correctly for this request
    self._send(f"setoption name MultiPV value {num}")
    self._ready()

    # Start the search
    self._send(f"go depth {self.depth}")

    lines: List[str] = []
    assert self.process and self.process.stdout
    deadline = time.time() + SearchTimeout

    while True:
      if time.time() > deadline:
        raise TimeoutError("Engine search exceeded timeout")

      line = self.process.stdout.readline()
      if not line:
        raise RuntimeError("Engine stopped unexpectedly during search")

      line = line.strip()
      lines.append(line)
      if line.startswith("bestmove"):
        break

    # ------------------------------------------------------------------
    # Parse the MultiPV info lines we just collected
    # ------------------------------------------------------------------
    moves: Dict[int, Dict] = {}
    for line in lines:
      if not (line.startswith("info") and "multipv" in line):
        continue

      parts = line.split()
      mpv, score_cp, mate, pv_move = None, None, None, None
      it = iter(range(len(parts)))
      for i in it:
        token = parts[i]
        if token == "multipv" and i + 1 < len(parts):
          mpv = int(parts[i + 1])
          next(it, None)
        elif token == "score" and i + 2 < len(parts):
          t, val = parts[i + 1], parts[i + 2]
          if t == "cp":
            score_cp = int(val) / 100.0
          elif t == "mate":
            mate = int(val)
          next(it, None)
        elif token == "pv" and i + 1 < len(parts):
          pv_move = parts[i + 1]
          break  # rest of the line is the PV

      if mpv and pv_move:
        moves[mpv] = {"move": pv_move, "score": score_cp, "mate": mate}

    # Convert to a list, multi-PV order 1…num
    result: List[Dict] = []
    for i in range(1, num + 1):
      if i not in moves:
        continue
      m = moves[i]
      if m["mate"] is not None:
        result.append({"move": m["move"], "type": "mate", "mate": m["mate"]})
      else:
        result.append({"move": m["move"], "type": "cp", "score": m["score"]})

    return result
