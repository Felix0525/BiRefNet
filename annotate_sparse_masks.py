"""Local pixel editor for paired sparse foreground masks.

The tool is intentionally independent from BiRefNet training dependencies.  It
loads same-stem images/masks, keeps edits binary, and writes lossless PNG files
to a separate output directory by default.
"""

import argparse
import json
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import QPointF, QProcess, QRectF, QSettings, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPainter, QPen
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QShortcut,
    QSlider,
    QSpinBox,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OVERLAY_COLOR = (96, 230, 135)


def discover_pairs(image_dir, mask_dir):
    """Return sorted (stem, image_path, mask_path) tuples."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError("图片目录不存在: {}".format(image_dir))
    if not mask_dir.is_dir():
        raise FileNotFoundError("mask 目录不存在: {}".format(mask_dir))

    masks = {
        path.stem.lower(): path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }
    pairs = []
    missing = []
    seen = set()
    for path in sorted(image_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = path.stem.lower()
        if key in seen:
            raise ValueError("图片 basename 重复: {}".format(path.stem))
        seen.add(key)
        mask_path = masks.get(key)
        if mask_path is None:
            missing.append(path.name)
        else:
            pairs.append((path.stem, path, mask_path))
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError("有 {} 张图片缺少同名 PNG mask: {}".format(len(missing), preview))
    if not pairs:
        raise ValueError("没有找到可编辑的图片/mask 配对")
    return pairs


def load_rgb(path):
    with Image.open(str(path)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def load_binary_mask(path, expected_shape):
    with Image.open(str(path)) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            "尺寸不一致: mask={}，图片={}".format(mask.shape, expected_shape)
        )
    return np.where(mask > 128, 255, 0).astype(np.uint8)


def save_mask_atomic(mask, path):
    """Write a verified binary PNG, then atomically replace the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    temp_path = path.with_name("{}.tmp.png".format(path.stem))
    Image.fromarray(binary, mode="L").save(str(temp_path), format="PNG", compress_level=3)
    with Image.open(str(temp_path)) as check:
        checked = np.asarray(check.convert("L"), dtype=np.uint8)
    if checked.shape != binary.shape or not np.array_equal(checked, binary):
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise IOError("保存后校验失败: {}".format(path))
    os.replace(str(temp_path), str(path))


def np_rgb_to_qimage(array):
    array = np.ascontiguousarray(array)
    height, width = array.shape[:2]
    return QImage(
        array.data, width, height, array.strides[0], QImage.Format_RGB888
    ).copy()


def np_gray_to_qimage(array):
    array = np.ascontiguousarray(array)
    height, width = array.shape
    return QImage(
        array.data, width, height, array.strides[0], QImage.Format_Grayscale8
    ).copy()


class MaskCanvas(QWidget):
    dirty_changed = pyqtSignal(bool)
    cursor_info = pyqtSignal(str)
    tool_changed = pyqtSignal(str)
    selection_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.CrossCursor)
        self.image = None
        self.mask = None
        self.image_qimage = None
        self.overlay_qimage = None
        self.boundary_qimage = None
        self.mask_qimage = None
        self.scale = 1.0
        self.offset = QPointF(0.0, 0.0)
        self.fit_mode = True
        self.tool = "brush"
        self.display_mode = "fill"
        self.opacity = 105
        self.brush_size = 12
        self.dirty = False
        self.drawing = False
        self.panning = False
        self.selecting = False
        self.selection_start = None
        self.selection_current = None
        self.selection_region = None
        self.last_image_point = None
        self.last_mouse_pos = None
        self.stroke_before = None
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 60
        self.setMinimumSize(640, 480)

    def has_data(self):
        return self.image is not None and self.mask is not None

    def set_data(self, image, mask):
        self.image = image
        self.mask = mask.copy()
        self.image_qimage = np_rgb_to_qimage(image)
        self.undo_stack = []
        self.redo_stack = []
        self.clear_selection()
        self._set_dirty(False)
        self._refresh_mask_views()
        self.fit_to_window()

    def replace_mask(self, mask):
        """Replace the displayed mask while preserving zoom and a selected ROI."""
        if self.image is None or mask.shape != self.image.shape[:2]:
            raise ValueError("替换 mask 的尺寸与当前图片不一致")
        self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        self.undo_stack = []
        self.redo_stack = []
        self._set_dirty(False)
        self._refresh_mask_views()
        self.update()

    def current_foreground_pixels(self):
        return int(np.count_nonzero(self.mask)) if self.mask is not None else 0

    def _set_dirty(self, value):
        value = bool(value)
        if self.dirty != value:
            self.dirty = value
            self.dirty_changed.emit(value)

    def mark_saved(self):
        self._set_dirty(False)

    def set_tool(self, tool):
        if tool not in {"brush", "eraser", "pan", "region"}:
            return
        self.tool = tool
        self.setCursor(Qt.OpenHandCursor if tool == "pan" else Qt.CrossCursor)
        self.tool_changed.emit(tool)
        self.update()

    def clear_selection(self, emit_signal=True):
        self.selecting = False
        self.selection_start = None
        self.selection_current = None
        if self.selection_region is not None:
            self.selection_region = None
            if emit_signal:
                self.selection_changed.emit(None)
        self.update()

    def current_selection(self):
        return self.selection_region

    def _normalize_selection(self, start, end):
        height, width = self.mask.shape
        left = max(0, min(width - 1, int(np.floor(min(start[0], end[0])))))
        top = max(0, min(height - 1, int(np.floor(min(start[1], end[1])))))
        right = max(left + 1, min(width, int(np.ceil(max(start[0], end[0])))))
        bottom = max(top + 1, min(height, int(np.ceil(max(start[1], end[1])))))
        return left, top, right, bottom

    def set_display_mode(self, mode):
        if mode in {"fill", "boundary", "mask"}:
            self.display_mode = mode
            self._refresh_mask_views()
            self.update()

    def set_opacity(self, opacity):
        self.opacity = max(0, min(255, int(opacity)))
        self._refresh_mask_views()
        self.update()

    def set_brush_size(self, size):
        self.brush_size = max(1, min(200, int(size)))
        self.update()

    def _refresh_mask_views(self):
        if self.mask is None:
            return
        present = self.mask > 0
        rgba = np.zeros((self.mask.shape[0], self.mask.shape[1], 4), dtype=np.uint8)
        rgba[present, 0] = OVERLAY_COLOR[0]
        rgba[present, 1] = OVERLAY_COLOR[1]
        rgba[present, 2] = OVERLAY_COLOR[2]
        rgba[present, 3] = self.opacity
        rgba = np.ascontiguousarray(rgba)
        height, width = self.mask.shape
        self.overlay_qimage = QImage(
            rgba.data, width, height, rgba.strides[0], QImage.Format_RGBA8888
        ).copy()

        interior = np.zeros_like(present)
        if present.shape[0] > 2 and present.shape[1] > 2:
            interior[1:-1, 1:-1] = (
                present[1:-1, 1:-1]
                & present[:-2, 1:-1]
                & present[2:, 1:-1]
                & present[1:-1, :-2]
                & present[1:-1, 2:]
            )
        boundary = present & ~interior
        boundary_rgba = np.zeros_like(rgba)
        boundary_rgba[boundary, 0] = 255
        boundary_rgba[boundary, 1] = 70
        boundary_rgba[boundary, 2] = 45
        boundary_rgba[boundary, 3] = max(self.opacity, 180)
        boundary_rgba = np.ascontiguousarray(boundary_rgba)
        self.boundary_qimage = QImage(
            boundary_rgba.data,
            width,
            height,
            boundary_rgba.strides[0],
            QImage.Format_RGBA8888,
        ).copy()
        self.mask_qimage = np_gray_to_qimage(self.mask)

    def fit_to_window(self):
        if not self.has_data() or self.width() <= 0 or self.height() <= 0:
            return
        height, width = self.mask.shape
        self.scale = min(self.width() / float(width), self.height() / float(height)) * 0.96
        self.scale = max(0.02, self.scale)
        self.offset = QPointF(
            (self.width() - width * self.scale) / 2.0,
            (self.height() - height * self.scale) / 2.0,
        )
        self.fit_mode = True
        self.update()

    def actual_pixels(self):
        if not self.has_data():
            return
        height, width = self.mask.shape
        self.scale = 1.0
        self.offset = QPointF(
            (self.width() - width) / 2.0, (self.height() - height) / 2.0
        )
        self.fit_mode = False
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.fit_mode:
            self.fit_to_window()

    def image_point(self, widget_point):
        if not self.has_data() or self.scale <= 0:
            return None
        x = (widget_point.x() - self.offset.x()) / self.scale
        y = (widget_point.y() - self.offset.y()) / self.scale
        height, width = self.mask.shape
        if 0 <= x < width and 0 <= y < height:
            return float(x), float(y)
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(35, 38, 42))
        if not self.has_data():
            painter.setPen(QColor(210, 210, 210))
            painter.drawText(self.rect(), Qt.AlignCenter, "没有加载图片")
            return

        height, width = self.mask.shape
        target = QRectF(
            self.offset.x(), self.offset.y(), width * self.scale, height * self.scale
        )
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        if self.display_mode == "mask":
            painter.drawImage(target, self.mask_qimage)
        else:
            painter.drawImage(target, self.image_qimage)
            overlay = self.overlay_qimage if self.display_mode == "fill" else self.boundary_qimage
            painter.drawImage(target, overlay)

        selection = self.selection_region
        if self.selecting and self.selection_start is not None and self.selection_current is not None:
            selection = self._normalize_selection(
                self.selection_start, self.selection_current
            )
        if selection is not None:
            left, top, right, bottom = selection
            selection_rect = QRectF(
                self.offset.x() + left * self.scale,
                self.offset.y() + top * self.scale,
                (right - left) * self.scale,
                (bottom - top) * self.scale,
            )
            painter.setPen(QPen(QColor(255, 180, 0), 2.0, Qt.DashLine))
            painter.setBrush(QColor(255, 180, 0, 35))
            painter.drawRect(selection_rect)

        if self.last_mouse_pos is not None and self.tool in {"brush", "eraser"}:
            point = self.image_point(self.last_mouse_pos)
            if point is not None:
                radius = self.brush_size * self.scale / 2.0
                center = QPointF(
                    self.offset.x() + point[0] * self.scale,
                    self.offset.y() + point[1] * self.scale,
                )
                color = QColor(80, 255, 130) if self.tool == "brush" else QColor(255, 80, 80)
                painter.setPen(QPen(color, 1.2))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(center, radius, radius)

    def wheelEvent(self, event):
        if not self.has_data():
            return
        cursor = event.pos()
        before = self.image_point(cursor)
        if before is None:
            before = (
                (cursor.x() - self.offset.x()) / self.scale,
                (cursor.y() - self.offset.y()) / self.scale,
            )
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        new_scale = max(0.03, min(20.0, self.scale * factor))
        self.scale = new_scale
        self.offset = QPointF(
            cursor.x() - before[0] * new_scale,
            cursor.y() - before[1] * new_scale,
        )
        self.fit_mode = False
        self.update()

    def mousePressEvent(self, event):
        self.setFocus()
        self.last_mouse_pos = event.pos()
        if not self.has_data():
            return
        if event.button() == Qt.MiddleButton or self.tool == "pan":
            self.panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        point = self.image_point(event.pos())
        if point is None:
            return
        if self.tool == "region":
            if event.button() == Qt.LeftButton:
                self.selecting = True
                self.selection_start = point
                self.selection_current = point
                self.update()
            return
        if event.modifiers() & Qt.AltModifier and event.button() == Qt.LeftButton:
            self._delete_component(int(point[0]), int(point[1]))
            return
        if event.button() not in {Qt.LeftButton, Qt.RightButton}:
            return

        self.drawing = True
        self.stroke_before = self.mask.copy()
        self.last_image_point = point
        erase = event.button() == Qt.RightButton or self.tool == "eraser"
        self._draw_segment(point, point, erase)

    def mouseMoveEvent(self, event):
        current_widget_pos = event.pos()
        if self.panning and self.last_mouse_pos is not None:
            delta = current_widget_pos - self.last_mouse_pos
            self.offset += QPointF(delta.x(), delta.y())
            self.fit_mode = False
            self.last_mouse_pos = current_widget_pos
            self.update()
            return

        self.last_mouse_pos = current_widget_pos
        point = self.image_point(current_widget_pos)
        if point is not None:
            x, y = int(point[0]), int(point[1])
            value = int(self.mask[y, x]) if self.mask is not None else 0
            self.cursor_info.emit("x={}, y={}, mask={}".format(x, y, value))
        if self.selecting and point is not None:
            self.selection_current = point
            left, top, right, bottom = self._normalize_selection(
                self.selection_start, self.selection_current
            )
            self.cursor_info.emit(
                "框选区域：x=[{}, {})，y=[{}, {})".format(left, right, top, bottom)
            )
            self.update()
            return
        if self.drawing and point is not None and self.last_image_point is not None:
            erase = bool(event.buttons() & Qt.RightButton) or self.tool == "eraser"
            self._draw_segment(self.last_image_point, point, erase)
            self.last_image_point = point
        self.update()

    def mouseReleaseEvent(self, event):
        if self.panning:
            self.panning = False
            self.setCursor(Qt.OpenHandCursor if self.tool == "pan" else Qt.CrossCursor)
            return
        if self.selecting:
            self.selecting = False
            point = self.image_point(event.pos())
            if point is not None:
                self.selection_current = point
            if self.selection_start is not None and self.selection_current is not None:
                self.selection_region = self._normalize_selection(
                    self.selection_start, self.selection_current
                )
                self.selection_changed.emit(self.selection_region)
            self.selection_start = None
            self.selection_current = None
            self.update()
            return
        if self.drawing:
            self.drawing = False
            self.last_image_point = None
            self._finish_history(self.stroke_before)
            self.stroke_before = None
            self._refresh_mask_views()
            self.update()

    def leaveEvent(self, event):
        self.last_mouse_pos = None
        self.cursor_info.emit("")
        self.update()

    def _draw_segment(self, start, end, erase):
        x0, y0 = start
        x1, y1 = end
        distance = max(abs(x1 - x0), abs(y1 - y0))
        steps = max(1, int(distance / max(1.0, self.brush_size / 4.0)) + 1)
        for index in range(steps + 1):
            ratio = index / float(steps)
            x = x0 + (x1 - x0) * ratio
            y = y0 + (y1 - y0) * ratio
            self._paint_circle(x, y, 0 if erase else 255)
        self._update_overlay_segment(start, end, erase)
        self._set_dirty(True)
        if self.display_mode != "fill":
            self._refresh_mask_views()

    def _paint_circle(self, x, y, value):
        radius = self.brush_size / 2.0
        height, width = self.mask.shape
        left = max(0, int(np.floor(x - radius)))
        right = min(width, int(np.ceil(x + radius)) + 1)
        top = max(0, int(np.floor(y - radius)))
        bottom = min(height, int(np.ceil(y + radius)) + 1)
        if left >= right or top >= bottom:
            return
        yy, xx = np.ogrid[top:bottom, left:right]
        area = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        patch = self.mask[top:bottom, left:right]
        patch[area] = value

    def _update_overlay_segment(self, start, end, erase):
        if self.overlay_qimage is None:
            return
        painter = QPainter(self.overlay_qimage)
        if erase:
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            color = QColor(0, 0, 0, 0)
        else:
            painter.setCompositionMode(QPainter.CompositionMode_Source)
            color = QColor(OVERLAY_COLOR[0], OVERLAY_COLOR[1], OVERLAY_COLOR[2], self.opacity)
        pen = QPen(color, float(self.brush_size), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(QPointF(*start), QPointF(*end))
        painter.end()

    def _finish_history(self, before):
        if before is None or np.array_equal(before, self.mask):
            return
        changed = np.nonzero(before != self.mask)
        top, bottom = int(changed[0].min()), int(changed[0].max()) + 1
        left, right = int(changed[1].min()), int(changed[1].max()) + 1
        entry = (
            (top, bottom, left, right),
            before[top:bottom, left:right].copy(),
            self.mask[top:bottom, left:right].copy(),
        )
        self.undo_stack.append(entry)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack = []
        self._set_dirty(True)

    def undo(self):
        if not self.undo_stack or self.mask is None:
            return
        entry = self.undo_stack.pop()
        top, bottom, left, right = entry[0]
        self.mask[top:bottom, left:right] = entry[1]
        self.redo_stack.append(entry)
        self._set_dirty(True)
        self._refresh_mask_views()
        self.update()

    def redo(self):
        if not self.redo_stack or self.mask is None:
            return
        entry = self.redo_stack.pop()
        top, bottom, left, right = entry[0]
        self.mask[top:bottom, left:right] = entry[2]
        self.undo_stack.append(entry)
        self._set_dirty(True)
        self._refresh_mask_views()
        self.update()

    def _delete_component(self, x, y):
        if self.mask[y, x] == 0:
            self.cursor_info.emit("Alt+左键位置不是前景")
            return
        before = self.mask.copy()
        height, width = self.mask.shape
        queue = deque([(x, y)])
        self.mask[y, x] = 0
        count = 0
        while queue:
            px, py = queue.popleft()
            count += 1
            for ny in range(max(0, py - 1), min(height, py + 2)):
                for nx in range(max(0, px - 1), min(width, px + 2)):
                    if self.mask[ny, nx] != 0:
                        self.mask[ny, nx] = 0
                        queue.append((nx, ny))
        self._finish_history(before)
        self._refresh_mask_views()
        self.cursor_info.emit("已删除连通域：{} 像素".format(count))
        self.update()


class SparseMaskEditor(QMainWindow):
    def __init__(
        self,
        image_dir,
        mask_dir,
        output_dir,
        start_name=None,
        inference_python=r"D:\new_conda_envs\edu-infer\python.exe",
    ):
        super().__init__()
        self.image_dir = Path(image_dir).resolve()
        self.mask_dir = Path(mask_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.pairs = discover_pairs(self.image_dir, self.mask_dir)
        self.index = 0
        self.original_foreground = 0
        self.inference_python = Path(inference_python).resolve()
        self.inference_process = None
        self.inference_key = None
        self.inference_ready = False
        self.inference_stdout_buffer = ""
        self.inference_stderr = ""
        self.pending_inference = None
        self.settings = QSettings("BiRefNet", "SparseMaskEditor")
        self.setWindowTitle("Sparse Mask 标注工具")
        self.resize(1380, 900)
        self.setStyleSheet(
            "QMainWindow, QWidget { color: #202124; }"
            "QToolBar, QStatusBar { background: #f3f4f6; color: #202124; }"
            "QToolButton, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {"
            " color: #202124; background: #ffffff; border: 1px solid #b8bdc5;"
            " border-radius: 3px; padding: 4px 7px; }"
            "QToolButton:hover, QPushButton:hover { background: #e8f3ff; }"
            "QToolButton:pressed, QPushButton:pressed { background: #cfe7ff; }"
            "QToolBar QLabel, QStatusBar QLabel { color: #202124; }"
        )

        self.canvas = MaskCanvas()
        self.canvas.dirty_changed.connect(self._update_title)
        self.canvas.cursor_info.connect(self._show_cursor_info)
        self.canvas.tool_changed.connect(self._sync_tool_combo)
        self.canvas.selection_changed.connect(self._selection_changed)
        self._build_ui()
        self._install_shortcuts()

        if start_name:
            lowered = Path(start_name).stem.lower()
            for i, item in enumerate(self.pairs):
                if item[0].lower() == lowered:
                    self.index = i
                    break
        self.load_current()

    def _build_ui(self):
        toolbar = QToolBar("编辑工具", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        previous_action = QAction("◀ 上一张", self)
        previous_action.triggered.connect(lambda: self.navigate(-1))
        toolbar.addAction(previous_action)
        next_action = QAction("下一张 ▶", self)
        next_action.triggered.connect(lambda: self.navigate(1))
        toolbar.addAction(next_action)
        toolbar.addSeparator()

        save_action = QAction("保存", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_current)
        toolbar.addAction(save_action)
        save_next_action = QAction("保存并下一张", self)
        save_next_action.triggered.connect(self.save_and_next)
        toolbar.addAction(save_next_action)
        toolbar.addSeparator()

        toolbar.addWidget(QLabel(" 工具 "))
        self.tool_combo = QComboBox()
        self.tool_combo.addItem("增加前景 (B)", "brush")
        self.tool_combo.addItem("擦除前景 (E/右键)", "eraser")
        self.tool_combo.addItem("平移 (V)", "pan")
        self.tool_combo.addItem("框选推理区域 (R)", "region")
        self.tool_combo.currentIndexChanged.connect(
            lambda _: self.canvas.set_tool(self.tool_combo.currentData())
        )
        toolbar.addWidget(self.tool_combo)

        toolbar.addWidget(QLabel("  笔刷 "))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 200)
        self.brush_spin.setValue(12)
        self.brush_spin.setSuffix(" px")
        self.brush_spin.valueChanged.connect(self.canvas.set_brush_size)
        toolbar.addWidget(self.brush_spin)

        toolbar.addWidget(QLabel("  显示 "))
        self.display_combo = QComboBox()
        self.display_combo.addItem("绿色填充 (1)", "fill")
        self.display_combo.addItem("红色边界 (2)", "boundary")
        self.display_combo.addItem("纯 mask (3)", "mask")
        self.display_combo.currentIndexChanged.connect(
            lambda _: self.canvas.set_display_mode(self.display_combo.currentData())
        )
        toolbar.addWidget(self.display_combo)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(5)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型路径："))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("选择 BiRefNet .pth checkpoint")
        self.model_path_edit.setText(self.settings.value("model_path", ""))
        model_layout.addWidget(self.model_path_edit, 1)
        browse_model_button = QPushButton("浏览…")
        browse_model_button.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_model_button)
        model_layout.addWidget(QLabel("模型类型："))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItem("swin-large", "swin-large")
        self.model_type_combo.addItem("swin-base", "swin-base")
        self.model_type_combo.addItem("swin-tiny", "swin-tiny")
        saved_model_type = self.settings.value("model_type", "swin-base")
        saved_model_index = self.model_type_combo.findData(saved_model_type)
        self.model_type_combo.setCurrentIndex(max(0, saved_model_index))
        model_layout.addWidget(self.model_type_combo)
        model_layout.addWidget(QLabel("推理高："))
        self.inference_height_spin = QSpinBox()
        self.inference_height_spin.setRange(64, 8192)
        self.inference_height_spin.setSingleStep(32)
        self.inference_height_spin.setValue(
            int(self.settings.value("inference_height", 1536))
        )
        model_layout.addWidget(self.inference_height_spin)
        model_layout.addWidget(QLabel("宽："))
        self.inference_width_spin = QSpinBox()
        self.inference_width_spin.setRange(64, 8192)
        self.inference_width_spin.setSingleStep(32)
        self.inference_width_spin.setValue(
            int(self.settings.value("inference_width", 1088))
        )
        model_layout.addWidget(self.inference_width_spin)
        model_layout.addWidget(QLabel("阈值："))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(float(self.settings.value("threshold", 0.5)))
        model_layout.addWidget(self.threshold_spin)
        self.generate_mask_button = QPushButton("全图生成并替换")
        self.generate_mask_button.setToolTip(
            "按宽度无重叠切块，末块右侧补白；按当前阈值二值化，不做后处理"
        )
        self.generate_mask_button.clicked.connect(self.generate_and_replace_mask)
        model_layout.addWidget(self.generate_mask_button)
        self.region_mask_button = QPushButton("框选区域推理")
        self.region_mask_button.setEnabled(False)
        self.region_mask_button.setToolTip("先选择“框选推理区域 (R)”，再在图片上拖出矩形")
        self.region_mask_button.clicked.connect(self.generate_selected_region)
        model_layout.addWidget(self.region_mask_button)
        self.clear_region_button = QPushButton("清除框选")
        self.clear_region_button.setEnabled(False)
        self.clear_region_button.clicked.connect(self.canvas.clear_selection)
        model_layout.addWidget(self.clear_region_button)
        layout.addLayout(model_layout)

        info_layout = QHBoxLayout()
        self.file_label = QLabel()
        self.file_label.setStyleSheet("font-weight: 600; font-size: 14px;")
        info_layout.addWidget(self.file_label, 1)
        info_layout.addWidget(QLabel("跳转："))
        self.jump_edit = QLineEdit()
        self.jump_edit.setPlaceholderText("输入文件名或序号")
        self.jump_edit.setMaximumWidth(220)
        self.jump_edit.returnPressed.connect(self.jump_to_text)
        info_layout.addWidget(self.jump_edit)
        jump_button = QPushButton("跳转")
        jump_button.clicked.connect(self.jump_to_text)
        info_layout.addWidget(jump_button)
        fit_button = QPushButton("适应窗口 (F)")
        fit_button.clicked.connect(self.canvas.fit_to_window)
        info_layout.addWidget(fit_button)
        actual_button = QPushButton("1:1 (0)")
        actual_button.clicked.connect(self.canvas.actual_pixels)
        info_layout.addWidget(actual_button)
        layout.addLayout(info_layout)
        layout.addWidget(self.canvas, 1)

        bottom = QHBoxLayout()
        self.progress_label = QLabel()
        bottom.addWidget(self.progress_label, 1)
        bottom.addWidget(QLabel("Mask 透明度"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(105)
        self.opacity_slider.setMaximumWidth(180)
        self.opacity_slider.valueChanged.connect(self.canvas.set_opacity)
        bottom.addWidget(self.opacity_slider)
        layout.addLayout(bottom)

        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "左键增加，右键擦除，Alt+左键删除整块连通域，滚轮缩放，中键拖动画布"
        )

    def _install_shortcuts(self):
        bindings = [
            ("B", lambda: self.canvas.set_tool("brush")),
            ("E", lambda: self.canvas.set_tool("eraser")),
            ("V", lambda: self.canvas.set_tool("pan")),
            ("R", lambda: self.canvas.set_tool("region")),
            ("Escape", self.canvas.clear_selection),
            ("1", lambda: self._set_display_index(0)),
            ("2", lambda: self._set_display_index(1)),
            ("3", lambda: self._set_display_index(2)),
            ("F", self.canvas.fit_to_window),
            ("0", self.canvas.actual_pixels),
            ("[", lambda: self.brush_spin.setValue(self.brush_spin.value() - 1)),
            ("]", lambda: self.brush_spin.setValue(self.brush_spin.value() + 1)),
            ("Left", lambda: self.navigate(-1)),
            ("Right", lambda: self.navigate(1)),
            ("Ctrl+Z", self.canvas.undo),
            ("Ctrl+Y", self.canvas.redo),
            ("Ctrl+Shift+Z", self.canvas.redo),
            ("Return", self.save_and_next),
        ]
        self.shortcuts = []
        for sequence, callback in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(callback)
            self.shortcuts.append(shortcut)

    def _set_display_index(self, index):
        self.display_combo.setCurrentIndex(index)

    def _sync_tool_combo(self, tool):
        index = self.tool_combo.findData(tool)
        if index >= 0 and self.tool_combo.currentIndex() != index:
            self.tool_combo.blockSignals(True)
            self.tool_combo.setCurrentIndex(index)
            self.tool_combo.blockSignals(False)

    def _selection_changed(self, region):
        has_region = region is not None
        self.region_mask_button.setEnabled(has_region and self.pending_inference is None)
        self.clear_region_button.setEnabled(has_region and self.pending_inference is None)
        if region is None:
            self.region_mask_button.setText("框选区域推理")
            return
        left, top, right, bottom = region
        self.region_mask_button.setText(
            "框选推理 {}×{}".format(right - left, bottom - top)
        )

    def browse_model(self):
        current = self.model_path_edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.cwd())
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 BiRefNet 模型",
            start_dir,
            "PyTorch checkpoint (*.pth *.pt);;所有文件 (*)",
        )
        if path:
            self.model_path_edit.setText(path)

    def _current_inference_key(self):
        return (
            str(Path(self.model_path_edit.text().strip()).resolve()),
            self.model_type_combo.currentData(),
            int(self.inference_height_spin.value()),
            int(self.inference_width_spin.value()),
        )

    def _set_inference_busy(self, busy, text=None):
        self.generate_mask_button.setEnabled(not busy)
        self.region_mask_button.setEnabled(
            not busy and self.canvas.current_selection() is not None
        )
        self.clear_region_button.setEnabled(
            not busy and self.canvas.current_selection() is not None
        )
        self.model_path_edit.setEnabled(not busy)
        self.model_type_combo.setEnabled(not busy)
        self.inference_height_spin.setEnabled(not busy)
        self.inference_width_spin.setEnabled(not busy)
        self.threshold_spin.setEnabled(not busy)
        self.canvas.setEnabled(not busy)
        self.generate_mask_button.setText(text if busy and text else "全图生成并替换")

    def generate_and_replace_mask(self):
        self._request_generation(region=None)

    def generate_selected_region(self):
        region = self.canvas.current_selection()
        if region is None:
            QMessageBox.information(self, "未框选区域", "请先选择“框选推理区域 (R)”并拖出矩形。")
            return
        self._request_generation(region=region)

    def _request_generation(self, region):
        model_text = self.model_path_edit.text().strip()
        if not model_text:
            QMessageBox.information(self, "缺少模型", "请先选择模型路径。")
            return
        model_path = Path(model_text).resolve()
        if not model_path.is_file():
            QMessageBox.warning(self, "模型不存在", str(model_path))
            return
        if not self.inference_python.is_file():
            QMessageBox.warning(
                self,
                "推理环境不存在",
                "找不到推理 Python：{}".format(self.inference_python),
            )
            return
        worker_path = Path(__file__).with_name("sparse_mask_inference_worker.py")
        if not worker_path.is_file():
            QMessageBox.warning(self, "推理脚本不存在", str(worker_path))
            return

        stem, image_path, _ = self.pairs[self.index]
        target = self.output_path(stem)
        if self.canvas.dirty or target.exists():
            if region is None:
                replacement_text = "模型结果将替换整张 mask：\n{}".format(target)
            else:
                left, top, right, bottom = region
                replacement_text = (
                    "模型结果只替换框选区域 x=[{}, {})，y=[{}, {})：\n{}".format(
                        left, right, top, bottom, target
                    )
                )
            answer = QMessageBox.question(
                self,
                "确认替换当前 mask",
                "{}\n\n原始 gt 不会修改。是否继续？".format(replacement_text),
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        self.settings.setValue("model_path", str(model_path))
        self.settings.setValue("model_type", self.model_type_combo.currentData())
        self.settings.setValue("inference_height", self.inference_height_spin.value())
        self.settings.setValue("inference_width", self.inference_width_spin.value())
        self.settings.setValue("threshold", self.threshold_spin.value())

        self.output_dir.mkdir(parents=True, exist_ok=True)
        preview_kind = "region" if region is not None else "full"
        temporary = self.output_dir / ".{}.{}.model-preview.png".format(
            stem, preview_kind
        )
        self.pending_inference = {
            "stem": stem,
            "index": self.index,
            "image": str(image_path),
            "temporary": temporary,
            "target": target,
            "region": region,
            "threshold": float(self.threshold_spin.value()),
        }
        key = self._current_inference_key()
        self._set_inference_busy(True, "准备模型…")
        if (
            self.inference_process is not None
            and self.inference_process.state() == QProcess.Running
            and self.inference_key == key
            and self.inference_ready
        ):
            self._send_inference_request()
            return
        self._stop_inference_worker()
        self._start_inference_worker(key, worker_path)

    def _start_inference_worker(self, key, worker_path):
        process = QProcess(self)
        self.inference_process = process
        self.inference_key = key
        self.inference_ready = False
        self.inference_stdout_buffer = ""
        self.inference_stderr = ""
        process.setWorkingDirectory(str(Path(__file__).resolve().parent))
        process.readyReadStandardOutput.connect(
            lambda process=process: self._read_inference_stdout(process)
        )
        process.readyReadStandardError.connect(
            lambda process=process: self._read_inference_stderr(process)
        )
        process.finished.connect(
            lambda code, status, process=process: self._inference_process_finished(
                process, code, status
            )
        )
        checkpoint, backbone, height, width = key
        arguments = [
            "-u",
            str(worker_path),
            "--checkpoint",
            checkpoint,
            "--backbone",
            backbone,
            "--height",
            str(height),
            "--width",
            str(width),
            "--threshold",
            "0.5",
        ]
        process.start(str(self.inference_python), arguments)
        if not process.waitForStarted(5000):
            self._fail_pending_inference(
                "无法启动推理进程：{}".format(process.errorString())
            )
            self.inference_process = None
            return
        self.statusBar().showMessage("正在加载模型：{}".format(checkpoint))

    def _send_inference_request(self):
        if self.inference_process is None or self.pending_inference is None:
            return
        request = {
            "command": "infer",
            "image": self.pending_inference["image"],
            "output": str(self.pending_inference["temporary"]),
            "threshold": self.pending_inference["threshold"],
            "region": self.pending_inference["region"],
        }
        self.inference_process.write(
            (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")
        )
        self._set_inference_busy(True, "推理中…")
        self.statusBar().showMessage(
            "正在生成 {} 的{}mask，阈值 {:.4f}…".format(
                self.pending_inference["stem"],
                "框选区域 " if self.pending_inference["region"] is not None else "全图 ",
                self.pending_inference["threshold"],
            )
        )

    def _read_inference_stdout(self, process):
        if process is not self.inference_process:
            return
        chunk = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        self.inference_stdout_buffer += chunk
        while "\n" in self.inference_stdout_buffer:
            line, self.inference_stdout_buffer = self.inference_stdout_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.inference_stderr += line + "\n"
                continue
            self._handle_inference_event(event)

    def _read_inference_stderr(self, process):
        if process is self.inference_process:
            self.inference_stderr += bytes(process.readAllStandardError()).decode(
                "utf-8", "replace"
            )

    def _handle_inference_event(self, event):
        event_name = event.get("event")
        if event_name == "loading":
            self._set_inference_busy(True, "加载模型…")
            self.statusBar().showMessage(
                "正在加载模型，设备：{}".format(event.get("device", ""))
            )
        elif event_name == "ready":
            self.inference_ready = True
            self.statusBar().showMessage(
                "模型已加载：{}，{}×{}，设备 {}".format(
                    event.get("backbone"),
                    event.get("height"),
                    event.get("width"),
                    event.get("device"),
                )
            )
            self._send_inference_request()
        elif event_name == "progress":
            self.statusBar().showMessage(
                "推理分块 {}/{}，原图 x=[{}, {})".format(
                    event.get("tile"),
                    event.get("tiles"),
                    event.get("left"),
                    event.get("right"),
                )
            )
        elif event_name == "done":
            self._complete_pending_inference(event)
        elif event_name in {"error", "fatal"}:
            self._fail_pending_inference(event.get("message", "未知推理错误"))

    def _complete_pending_inference(self, event):
        pending = self.pending_inference
        if pending is None:
            return
        try:
            with Image.open(str(pending["image"])) as opened:
                image_shape = (opened.height, opened.width)
            region = pending["region"]
            if region is None:
                mask = load_binary_mask(pending["temporary"], image_shape)
            else:
                left, top, right, bottom = region
                region_mask = load_binary_mask(
                    pending["temporary"], (bottom - top, right - left)
                )
                mask = self.canvas.mask.copy()
                mask[top:bottom, left:right] = region_mask
            save_mask_atomic(mask, pending["target"])
        except Exception as error:
            self._fail_pending_inference("结果校验或保存失败：{}".format(error))
            return
        finally:
            try:
                pending["temporary"].unlink()
            except OSError:
                pass

        stem = pending["stem"]
        self.pending_inference = None
        self._set_inference_busy(False)
        if self.pairs[self.index][0].lower() == stem.lower():
            self.canvas.replace_mask(mask)
            self.original_foreground = int(np.count_nonzero(mask))
            self.file_label.setText("{}  ·  模型生成输出".format(stem))
            self._update_progress()
            self._update_title(False)
        self.statusBar().showMessage(
            "模型 mask 已生成并替换：{}（前景 {:,} px）".format(
                pending["target"], int(event.get("foreground", np.count_nonzero(mask)))
            ),
            6000,
        )

    def _fail_pending_inference(self, message):
        pending = self.pending_inference
        if pending is not None:
            try:
                pending["temporary"].unlink()
            except OSError:
                pass
        self.pending_inference = None
        self._set_inference_busy(False)
        details = message
        if self.inference_stderr.strip():
            details += "\n\n" + self.inference_stderr.strip()[-3000:]
        QMessageBox.critical(self, "模型推理失败", details)

    def _inference_process_finished(self, process, code, status):
        if process is not self.inference_process:
            return
        self._read_inference_stderr(process)
        was_pending = self.pending_inference is not None
        self.inference_process = None
        self.inference_ready = False
        self.inference_key = None
        if was_pending:
            self._fail_pending_inference(
                "推理进程提前退出，退出码 {}".format(code)
            )

    def _stop_inference_worker(self):
        process = self.inference_process
        if process is None:
            return
        # Detach first so the old process's finished signal cannot clear a new
        # request while model settings are being changed.
        self.inference_process = None
        self.inference_ready = False
        self.inference_key = None
        if process.state() == QProcess.Running:
            process.write(b'{"command":"quit"}\n')
            process.waitForFinished(1500)
        if process.state() != QProcess.NotRunning:
            process.kill()
            process.waitForFinished(1500)

    def output_path(self, stem=None):
        if stem is None:
            stem = self.pairs[self.index][0]
        return self.output_dir / "{}.png".format(stem)

    def load_current(self):
        stem, image_path, source_mask_path = self.pairs[self.index]
        edited_path = self.output_path(stem)
        mask_path = edited_path if edited_path.exists() else source_mask_path
        try:
            image = load_rgb(image_path)
            mask = load_binary_mask(mask_path, image.shape[:2])
        except Exception as error:
            QMessageBox.critical(self, "加载失败", str(error))
            return
        self.original_foreground = int(np.count_nonzero(mask))
        self.canvas.set_data(image, mask)
        source_text = "已编辑输出" if edited_path.exists() else "原始 GT"
        self.file_label.setText("{}  ·  {}".format(stem, source_text))
        self._update_progress()
        self._update_title(False)

    def _update_progress(self):
        edited = sum(1 for item in self.pairs if self.output_path(item[0]).exists())
        foreground = self.canvas.current_foreground_pixels()
        total_pixels = self.canvas.mask.size if self.canvas.mask is not None else 0
        ratio = foreground / float(total_pixels) * 100.0 if total_pixels else 0.0
        self.progress_label.setText(
            "第 {}/{} 张  |  已输出 {} 张  |  前景 {:,} px ({:.3f}%)  |  输出：{}".format(
                self.index + 1,
                len(self.pairs),
                edited,
                foreground,
                ratio,
                self.output_dir,
            )
        )

    def _update_title(self, dirty=None):
        if dirty is None:
            dirty = self.canvas.dirty
        stem = self.pairs[self.index][0] if self.pairs else ""
        mark = " *未保存" if dirty else ""
        self.setWindowTitle("Sparse Mask 标注工具 - {}{}".format(stem, mark))
        self._update_progress()

    def _show_cursor_info(self, text):
        if text:
            self.statusBar().showMessage(text)
        else:
            self.statusBar().showMessage(
                "左键增加，右键擦除，Alt+左键删除整块连通域，滚轮缩放，中键拖动画布"
            )

    def maybe_leave_current(self):
        if not self.canvas.dirty:
            return True
        answer = QMessageBox.question(
            self,
            "当前修改未保存",
            "当前 mask 已修改，是否保存后继续？",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save_current()
        return True

    def navigate(self, delta):
        if self.pending_inference is not None:
            self.statusBar().showMessage("模型推理中，暂不能切换图片", 2500)
            return
        if not self.maybe_leave_current():
            return
        new_index = self.index + delta
        if new_index < 0 or new_index >= len(self.pairs):
            self.statusBar().showMessage("已经到达数据集边界", 2500)
            return
        self.index = new_index
        self.load_current()

    def save_current(self):
        if self.pending_inference is not None:
            self.statusBar().showMessage("模型推理中，暂不能保存", 2500)
            return False
        stem = self.pairs[self.index][0]
        target = self.output_path(stem)
        try:
            save_mask_atomic(self.canvas.mask, target)
        except Exception as error:
            QMessageBox.critical(self, "保存失败", str(error))
            return False
        self.canvas.mark_saved()
        self.file_label.setText("{}  ·  已编辑输出".format(stem))
        self._update_progress()
        self.statusBar().showMessage("已保存并校验：{}".format(target), 4000)
        return True

    def save_and_next(self):
        if not self.save_current():
            return
        if self.index + 1 < len(self.pairs):
            self.index += 1
            self.load_current()

    def jump_to_text(self):
        if self.pending_inference is not None:
            self.statusBar().showMessage("模型推理中，暂不能切换图片", 2500)
            return
        text = self.jump_edit.text().strip()
        if not text:
            return
        target = None
        if text.isdigit():
            number = int(text)
            if 1 <= number <= len(self.pairs):
                target = number - 1
        if target is None:
            lowered = Path(text).stem.lower()
            for index, item in enumerate(self.pairs):
                if item[0].lower() == lowered:
                    target = index
                    break
        if target is None:
            QMessageBox.information(self, "未找到", "找不到序号或文件名：{}".format(text))
            return
        if target == self.index:
            return
        if not self.maybe_leave_current():
            return
        self.index = target
        self.load_current()

    def closeEvent(self, event):
        if self.maybe_leave_current():
            self.pending_inference = None
            self._stop_inference_worker()
            event.accept()
        else:
            event.ignore()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Edit paired sparse binary masks")
    parser.add_argument("--images", required=True, help="source image directory")
    parser.add_argument("--masks", required=True, help="source PNG mask directory")
    parser.add_argument("--output", required=True, help="edited PNG output directory")
    parser.add_argument("--start", help="optional first filename/stem")
    parser.add_argument(
        "--inference-python",
        default=os.environ.get(
            "BIREFNET_INFERENCE_PYTHON",
            r"D:\new_conda_envs\edu-infer\python.exe",
        ),
        help="Python executable containing PyTorch/CUDA for model inference",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0]] + list(argv))
    app.setStyle("Fusion")
    try:
        window = SparseMaskEditor(
            args.images,
            args.masks,
            args.output,
            args.start,
            args.inference_python,
        )
    except Exception as error:
        QMessageBox.critical(None, "启动失败", str(error))
        return 1
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
