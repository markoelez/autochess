#!/usr/bin/env python3
"""
Screen capture and board detection module for chess overlay.
"""

from typing import Dict, Tuple, Optional

import cv2
import mss
import numpy as np


class ChessBoardDetector:
  """Detects chess board on screen and extracts board region."""

  def __init__(self):
    self.sct = mss.mss()
    self.board_rect = None
    self.last_board_image = None

  def capture_screen(self, monitor_idx: int = 1) -> np.ndarray:
    """Capture screenshot of specified monitor."""
    # Use the exact monitor index provided
    if monitor_idx <= 0 or monitor_idx >= len(self.sct.monitors):
      # Fallback to primary monitor if invalid index
      monitor_idx = 1 if len(self.sct.monitors) > 1 else 0

    monitor = self.sct.monitors[monitor_idx]

    screenshot = self.sct.grab(monitor)
    img = np.array(screenshot)
    # Convert BGRA to BGR
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

  def detect_chessboard(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect chess board in image using edge detection and contour finding.
    Returns (x, y, width, height) of board region or None.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest square-like contour
    best_rect = None
    max_area = 0

    for contour in contours:
      # Approximate contour to polygon
      peri = cv2.arcLength(contour, True)
      approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

      # Check if it's roughly square (4 corners)
      if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect_ratio = w / h

        # Chess board should be roughly square (aspect ratio close to 1)
        # Also ensure minimum size (at least 200x200 pixels)
        if 0.8 < aspect_ratio < 1.2 and area > max_area and area > 40000 and w > 200 and h > 200:
          max_area = area
          best_rect = (x, y, w, h)

    # Alternative detection using Hough lines for grid pattern
    if best_rect is None:
      best_rect = self._detect_using_grid_lines(edges)

    # If still no detection, try finding large square regions
    if best_rect is None:
      best_rect = self._detect_using_color_regions(image)

    return best_rect

  def _detect_using_grid_lines(self, edges: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect chess board using Hough line detection to find grid pattern.
    """
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
      return None

    # Separate horizontal and vertical lines
    h_lines = []
    v_lines = []

    for line in lines:
      x1, y1, x2, y2 = line[0]
      angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

      if angle < 10 or angle > 170:  # Horizontal
        h_lines.append((y1 + y2) // 2)
      elif 80 < angle < 100:  # Vertical
        v_lines.append((x1 + x2) // 2)

    # Find clusters of parallel lines (8x8 grid)
    if len(h_lines) >= 8 and len(v_lines) >= 8:
      h_lines = sorted(h_lines)
      v_lines = sorted(v_lines)

      # Find the most regular grid pattern
      h_clusters = self._find_line_clusters(h_lines, 8)
      v_clusters = self._find_line_clusters(v_lines, 8)

      if h_clusters and v_clusters:
        x = min(v_clusters)
        y = min(h_clusters)
        w = max(v_clusters) - x
        h = max(h_clusters) - y

        # Verify it's roughly square and has minimum size
        if 0.8 < w / h < 1.2 and w > 200 and h > 200:
          return (x, y, w, h)

    return None

  def _find_line_clusters(self, lines: list, target_count: int) -> Optional[list]:
    """Find clusters of equally spaced lines."""
    if len(lines) < target_count:
      return None

    # Try different starting points
    best_cluster = None
    min_variance = float("inf")

    for start_idx in range(len(lines) - target_count + 1):
      cluster = lines[start_idx : start_idx + target_count]

      # Calculate spacing variance
      spacings = [cluster[i + 1] - cluster[i] for i in range(len(cluster) - 1)]
      variance = np.var(spacings)

      if variance < min_variance:
        min_variance = variance
        best_cluster = cluster

    # Check if spacing is regular enough
    if min_variance < 100:  # Threshold for regular spacing
      return best_cluster

    return None

  def _detect_using_color_regions(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect chess board by looking for large square regions with consistent colors.
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Look for regions with low saturation (chess boards are usually not very colorful)
    _, saturation, _ = cv2.split(hsv)

    # Threshold to find low saturation areas
    _, thresh = cv2.threshold(saturation, 50, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest square-like contour
    best_rect = None
    max_area = 0

    for contour in contours:
      x, y, w, h = cv2.boundingRect(contour)
      area = w * h
      aspect_ratio = w / h if h > 0 else 0

      # Chess board should be roughly square and large
      if 0.8 < aspect_ratio < 1.2 and area > max_area and w > 200 and h > 200:
        max_area = area
        best_rect = (x, y, w, h)

    return best_rect

  def extract_board_image(self, full_image: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract and resize board region from full image."""
    # Handle both old format (x,y,w,h) and new format (x,y,w,h,monitor_idx)
    if len(rect) == 5:
      x, y, w, h, _ = rect  # Ignore monitor index
    else:
      x, y, w, h = rect
    board_img = full_image[y : y + h, x : x + w]

    # Resize to standard size for processing (512x512)
    board_img = cv2.resize(board_img, (512, 512))

    return board_img

  def get_board_coordinates(self) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Get screen coordinates for each square on the chess board.
    Returns dict mapping square names (e.g., 'a1', 'e4') to (x, y) coordinates.
    """
    if self.board_rect is None:
      return None

    x, y, w, h = self.board_rect
    square_w = w // 8
    square_h = h // 8

    coordinates = {}

    for row in range(8):
      for col in range(8):
        # Chess board coordinates (a1 is bottom-left for white)
        file = chr(ord("a") + col)
        rank = str(8 - row)
        square = f"{file}{rank}"

        # Center of square
        cx = x + col * square_w + square_w // 2
        cy = y + row * square_h + square_h // 2

        coordinates[square] = (cx, cy)

    return coordinates
