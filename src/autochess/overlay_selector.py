#!/usr/bin/env python3
"""
Screen overlay for selecting chess board area using PyQt6.
"""

import sys
from typing import Tuple, Optional

import mss
import numpy as np
from PyQt6.QtGui import QPen, QFont, QColor, QImage, QPixmap, QPainter
from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtWidgets import QWidget, QApplication


class SelectionOverlay(QWidget):
  """Fullscreen overlay for area selection."""

  # Signal emitted when selection is complete
  selection_complete = pyqtSignal(tuple)

  def __init__(self, qt_screen):
    super().__init__()
    self.qt_screen = qt_screen
    self.selection_start = None
    self.selection_end = None
    self.selection_rect = None
    self.is_selecting = False
    self.screenshot = None
    self.monitor = None
    self.monitor_index = None
    self.mss_width = 0
    self.mss_height = 0
    self.qt_width = 0
    self.qt_height = 0
    self.setup_overlay()
    self.take_screenshot()

  def setup_overlay(self):
    """Setup the fullscreen overlay."""
    # Make window fullscreen and always on top
    self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)

    # Make window accept keyboard input
    self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # Get screen geometry and make fullscreen
    geometry = self.qt_screen.geometry()
    self.setGeometry(geometry)

    # Set cursor
    self.setCursor(Qt.CursorShape.CrossCursor)

    # Set background to transparent
    self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

    # Show window
    self.show()
    self.raise_()  # Bring to front
    self.activateWindow()  # Make it active

  def take_screenshot(self):
    """Take a screenshot for the background."""
    try:
      with mss.mss() as sct:
        qt_geom = self.qt_screen.geometry()
        qt_x = qt_geom.x()
        qt_y = qt_geom.y()
        qt_w = qt_geom.width()
        qt_h = qt_geom.height()

        self.monitor = None
        for i, m in enumerate(sct.monitors):
          if i == 0:
            continue
          if m["left"] == qt_x and m["top"] == qt_y and m["width"] == qt_w and m["height"] == qt_h:
            self.monitor = m
            self.monitor_index = i
            break

        if self.monitor is None:
          print("Warning: No matching monitor found, using primary")
          self.monitor_index = 1 if len(sct.monitors) > 1 else 0
          self.monitor = sct.monitors[self.monitor_index]

        screenshot = sct.grab(self.monitor)

        img = np.array(screenshot)
        img_rgb = img[:, :, [2, 1, 0]]  # BGRA -> RGB
        img_rgb = np.ascontiguousarray(img_rgb)

        self.mss_height, self.mss_width = img_rgb.shape[:2]
        self.qt_width = qt_w
        self.qt_height = qt_h

        bytes_per_line = 3 * self.mss_width
        img_bytes = img_rgb.tobytes()
        q_image = QImage(img_bytes, self.mss_width, self.mss_height, bytes_per_line, QImage.Format.Format_RGB888)

        self.screenshot = QPixmap.fromImage(q_image).scaled(
          self.qt_width, self.qt_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

    except Exception as e:
      print(f"Error taking screenshot: {e}")
      self.screenshot = None

  def paintEvent(self, event):
    """Paint the overlay."""
    painter = QPainter(self)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    dim_color = QColor(0, 0, 0, 100)

    if self.selection_rect:
      sel = self.selection_rect
      # Top dim
      painter.fillRect(0, 0, self.width(), sel.top(), dim_color)
      # Bottom dim
      painter.fillRect(0, sel.bottom() + 1, self.width(), self.height() - sel.bottom() - 1, dim_color)
      # Left dim
      painter.fillRect(0, sel.top(), sel.left(), sel.height(), dim_color)
      # Right dim
      painter.fillRect(sel.right() + 1, sel.top(), self.width() - sel.right() - 1, sel.height(), dim_color)
    else:
      # Dim entire screen if no selection
      painter.fillRect(self.rect(), dim_color)

    # Draw selection rectangle
    if self.selection_rect:
      # Draw border
      pen = QPen(QColor(0, 255, 0), 3)
      painter.setPen(pen)
      painter.drawRect(self.selection_rect)

      # Draw coordinates
      x, y, w, h = self.selection_rect.x(), self.selection_rect.y(), self.selection_rect.width(), self.selection_rect.height()
      text = f"{x},{y},{w},{h}"
      painter.setPen(QPen(QColor(255, 255, 255), 2))
      painter.drawText(x + 5, y - 5, text)

    # Draw instructions
    painter.setPen(QPen(QColor(255, 255, 0), 2))
    painter.drawText(20, 30, "Click and drag to select area")
    painter.drawText(20, 50, "Press ENTER to confirm, ESC to cancel")

  def mousePressEvent(self, event):
    """Handle mouse press."""
    if event.button() == Qt.MouseButton.LeftButton:
      self.selection_start = event.position().toPoint()
      self.is_selecting = True

  def mouseMoveEvent(self, event):
    """Handle mouse move."""
    if self.is_selecting and self.selection_start:
      self.selection_end = event.position().toPoint()

      # Update selection rectangle
      x1, y1 = self.selection_start.x(), self.selection_start.y()
      x2, y2 = self.selection_end.x(), self.selection_end.y()

      # Ensure positive width/height
      x = min(x1, x2)
      y = min(y1, y2)
      w = abs(x2 - x1)
      h = abs(y2 - y1)

      self.selection_rect = QRect(x, y, w, h)
      self.update()  # Trigger repaint

  def mouseReleaseEvent(self, event):
    """Handle mouse release."""
    if event.button() == Qt.MouseButton.LeftButton:
      self.is_selecting = False

  def keyPressEvent(self, event):
    """Handle key press."""
    if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
      if self.selection_rect:
        coords = (self.selection_rect.x(), self.selection_rect.y(), self.selection_rect.width(), self.selection_rect.height())
        self.selection_complete.emit(coords)
      else:
        self.selection_complete.emit(())
      self.close()
    elif event.key() == Qt.Key.Key_Escape:
      self.selection_complete.emit(())
      self.close()
    else:
      super().keyPressEvent(event)


class OverlaySelector:
  """Main selector class."""

  def __init__(self):
    self.app = None
    self.overlays = []
    self.result = None

  def select_area(self) -> Optional[Tuple[int, int, int, int, int]]:
    """
    Show overlay and allow user to select an area.

    Returns:
        (x, y, width, height, monitor_idx) of selected area or None if cancelled
    """
    # Create QApplication if it doesn't exist
    if not QApplication.instance():
      self.app = QApplication(sys.argv)
    else:
      self.app = QApplication.instance()

    # Reset result
    self.result = None

    # Create overlays for each screen
    self.overlays = []
    for qt_screen in self.app.screens():
      overlay = SelectionOverlay(qt_screen)
      overlay.selection_complete.connect(lambda coords, o=overlay: self._on_selection_complete(coords, o))
      self.overlays.append(overlay)

    # Run event loop
    self.app.exec()

    return self.result

  def _on_selection_complete(self, local_coords, sender):
    """Handle selection completion."""
    if local_coords:
      x, y, w, h = local_coords

      # Compute absolute logical coordinates
      abs_x = sender.x() + x
      abs_y = sender.y() + y

      # Compute relative to monitor in logical
      monitor = sender.monitor
      if monitor:
        rel_x_l = abs_x - monitor["left"]
        rel_y_l = abs_y - monitor["top"]
      else:
        rel_x_l = abs_x
        rel_y_l = abs_y

      # Compute DPR
      dpr = sender.mss_width / sender.qt_width if sender.qt_width else 1.0

      # Convert to physical
      rel_x_p = int(rel_x_l * dpr)
      rel_y_p = int(rel_y_l * dpr)
      w_p = int(w * dpr)
      h_p = int(h * dpr)

      # Include monitor index
      monitor_idx = sender.monitor_index or 1
      self.result = (rel_x_p, rel_y_p, w_p, h_p, monitor_idx)
    else:
      self.result = None

    # Close all overlays
    for o in self.overlays:
      o.close()

    if self.app:
      self.app.quit()


def main():
  """Test the overlay selector."""
  selector = OverlaySelector()
  result = selector.select_area()

  if result:
    x, y, w, h, mon = result
    print(f"Selected area: x={x}, y={y}, width={w}, height={h} on monitor {mon}")
    print(f"Command line argument: --rect {x},{y},{w},{h}")
  else:
    print("Selection cancelled")


if __name__ == "__main__":
  main()
