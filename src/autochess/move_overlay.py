#!/usr/bin/env python3
"""
Move overlay for highlighting chess moves using PyQt6.
"""

import sys
from typing import Tuple

import mss
from PyQt6.QtGui import QPen, QFont, QColor, QPainter
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtWidgets import QWidget, QApplication


class MoveOverlay(QWidget):
  """Fullscreen overlay for highlighting move squares."""

  def __init__(self, qt_screen, board_rect: Tuple[int, int, int, int], from_square: str, to_square: str):
    super().__init__()
    self.qt_screen = qt_screen
    self.board_rect = board_rect
    self.from_square = from_square
    self.to_square = to_square
    self.from_rect = None
    self.to_rect = None
    self.dpr = self.qt_screen.devicePixelRatio()
    self.setup_overlay()
    self.compute_rects()

  def setup_overlay(self):
    self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
    self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    geometry = self.qt_screen.geometry()
    self.setGeometry(geometry)
    self.setCursor(Qt.CursorShape.ArrowCursor)
    self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
    self.show()
    self.raise_()
    self.activateWindow()
    self.setFocus()

  def compute_rects(self):
    x_p, y_p, w_p, h_p = self.board_rect
    x_l = x_p / self.dpr
    y_l = y_p / self.dpr
    w_l = w_p / self.dpr
    h_l = h_p / self.dpr
    sq_w = w_l / 8
    sq_h = h_l / 8

    def get_rect(square: str) -> QRect:
      file, rank_str = square[0], square[1]
      col = ord(file) - ord("a")
      rank = int(rank_str)
      row = 8 - rank
      tl_x = x_l + col * sq_w
      tl_y = y_l + row * sq_h
      return QRect(int(tl_x), int(tl_y), int(sq_w), int(sq_h))

    self.from_rect = get_rect(self.from_square)
    self.to_rect = get_rect(self.to_square)

  def paintEvent(self, event):
    painter = QPainter(self)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor("green"), 3)
    painter.setPen(pen)
    painter.drawRect(self.from_rect)
    pen = QPen(QColor("red"), 3)
    painter.setPen(pen)
    painter.drawRect(self.to_rect)
    painter.setPen(QPen(QColor(255, 255, 255), 2))
    painter.setFont(QFont("Arial", 12))
    painter.drawText(20, 30, f"Best move: {self.from_square} to {self.to_square}")
    painter.drawText(20, 50, "Press ENTER or click to dismiss")

  def keyPressEvent(self, event):
    if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
      self.close()
    else:
      super().keyPressEvent(event)

  def mousePressEvent(self, event):
    self.close()


class ShowMoveOverlay:
  """Main move overlay display class."""

  def __init__(self):
    self.app = QApplication.instance() or QApplication(sys.argv)
    self.overlay = None  # Added for reference

  def show_on_monitor(self, monitor_idx: int, board_rect: Tuple[int, int, int, int], from_square: str, to_square: str) -> bool:
    with mss.mss() as sct:
      monitor = sct.monitors[monitor_idx]
    for qt_screen in self.app.screens():
      geom = qt_screen.geometry()
      if (
        geom.x() == monitor["left"]
        and geom.y() == monitor["top"]
        and geom.width() == monitor["width"]
        and geom.height() == monitor["height"]
      ):
        self.overlay = MoveOverlay(qt_screen, board_rect, from_square, to_square)  # Keep reference
        self.app.exec()
        self.overlay = None  # Clean up
        return True
    print("Warning: No matching screen found for overlay")
    return False
