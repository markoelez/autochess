#!/usr/bin/env python3
"""
Simple Stockfish wrapper with better timeout handling.
"""

import subprocess
import time
from typing import List, Dict, Optional, Tuple


class SimpleStockfish:
    """Simple Stockfish wrapper."""
    
    def __init__(self, path: str = "/opt/homebrew/bin/stockfish", depth: int = 10):
        self.path = path
        self.depth = depth
        self.process = None
        self._start()
    
    def _start(self):
        """Start Stockfish process."""
        self.process = subprocess.Popen(
            [self.path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        
        # Send UCI and read response
        self.process.stdin.write("uci\n")
        self.process.stdin.flush()
        
        # Read until uciok
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Stockfish process died")
            if line.strip() == "uciok":
                break
    
    def set_position(self, fen: str):
        """Set position from FEN."""
        self.process.stdin.write(f"position fen {fen}\n")
        self.process.stdin.flush()
    
    def get_top_moves(self, num: int = 5) -> List[Dict]:
        """Get top moves."""
        # Set MultiPV
        self.process.stdin.write(f"setoption name MultiPV value {num}\n")
        self.process.stdin.flush()
        
        # Start analysis
        self.process.stdin.write(f"go depth {self.depth}\n")
        self.process.stdin.flush()
        
        # Collect output
        lines = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            line = line.strip()
            lines.append(line)
            if line.startswith("bestmove"):
                break
        
        # Parse moves
        moves = {}
        for line in lines:
            if line.startswith("info") and "multipv" in line:
                parts = line.split()
                multipv = None
                score = None
                mate = None
                pv = []
                
                i = 0
                while i < len(parts):
                    if parts[i] == "multipv" and i + 1 < len(parts):
                        multipv = int(parts[i + 1])
                    elif parts[i] == "score" and i + 2 < len(parts):
                        if parts[i + 1] == "cp":
                            score = int(parts[i + 2]) / 100.0
                        elif parts[i + 1] == "mate":
                            mate = int(parts[i + 2])
                    elif parts[i] == "pv":
                        pv = parts[i + 1:]
                        break
                    i += 1
                
                if multipv and pv:
                    moves[multipv] = {
                        "move": pv[0],
                        "score": score,
                        "mate": mate
                    }
        
        # Convert to list
        result = []
        for i in range(1, num + 1):
            if i in moves:
                m = moves[i]
                if m["mate"] is not None:
                    result.append({
                        "move": m["move"],
                        "type": "mate",
                        "mate": m["mate"]
                    })
                elif m["score"] is not None:
                    result.append({
                        "move": m["move"],
                        "type": "cp",
                        "score": m["score"]
                    })
        
        return result
    
    def close(self):
        """Close engine."""
        if self.process:
            self.process.stdin.write("quit\n")
            self.process.stdin.flush()
            time.sleep(0.1)
            self.process.terminate()
            self.process = None
    
    def __del__(self):
        self.close()