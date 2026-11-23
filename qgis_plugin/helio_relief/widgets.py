"""
Custom Qt widgets for Helio Relief: the sun-position
dial, the live preview canvas, collapsible sections and linked slider/spin
controls.
"""

import math
import time

import numpy as np
from qgis.PyQt.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from qgis.PyQt.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QPainter, QPalette, QPen, QPixmap,
                             QRadialGradient)
from qgis.PyQt.QtWidgets import (QDoubleSpinBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QScrollArea,
                                 QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget)


class SliderSpin(QWidget):
    """Slider and spin box kept in sync; optional logarithmic slider travel."""

    valueChanged = pyqtSignal(float)
    _STEPS = 1000

    def __init__(self, minimum, maximum, value, decimals=2, step=0.05, log=False, suffix="", parent=None):
        super().__init__(parent)
        self._min, self._max, self._log = float(minimum), float(maximum), bool(log and minimum > 0)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._STEPS)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setSuffix(suffix)
        self.spin.setKeyboardTracking(False)
        self.spin.setMinimumWidth(84)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self._guard = False
        self._default = float(value)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.slider.installEventFilter(self)
        self.setValue(value)

    def eventFilter(self, obj, event):
        if obj is self.slider and event.type() == QEvent.Type.MouseButtonDblClick:
            self.setValue(self._default)          # double-click a slider to restore its default
            return True
        return super().eventFilter(obj, event)

    def defaultValue(self):
        return self._default

    def _to_pos(self, v):
        if self._log:
            t = math.log(v / self._min) / math.log(self._max / self._min)
        else:
            t = (v - self._min) / (self._max - self._min)
        return int(round(min(max(t, 0.0), 1.0) * self._STEPS))

    def _from_pos(self, pos):
        t = pos / float(self._STEPS)
        if self._log:
            return self._min * (self._max / self._min) ** t
        return self._min + t * (self._max - self._min)

    def _from_slider(self, pos):
        if self._guard:
            return
        self._guard = True
        self.spin.setValue(self._from_pos(pos))
        self._guard = False
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v):
        if not self._guard:
            self._guard = True
            self.slider.setValue(self._to_pos(v))
            self._guard = False
        self.valueChanged.emit(v)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        self.spin.setValue(float(v))
        self._guard = True
        self.slider.setValue(self._to_pos(self.spin.value()))
        self._guard = False

    def setToolTip(self, text):
        text = (text + "\n" if text else "") + "Double-click the slider to reset."
        super().setToolTip(text)
        self.slider.setToolTip(text)
        self.spin.setToolTip(text)


class Section(QWidget):
    """Collapsible titled block with a form layout body and a reset button."""

    resetRequested = pyqtSignal()

    def __init__(self, title, expanded=True, parent=None):
        super().__init__(parent)
        self.toggle = QToolButton()
        self.toggle.setText(title.replace("&", "&&"))   # '&' would otherwise become a mnemonic
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setAutoRaise(True)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        f = self.toggle.font()
        f.setBold(True)
        self.toggle.setFont(f)
        self.body = QWidget()
        self.form = QFormLayout(self.body)
        self.form.setContentsMargins(18, 2, 4, 8)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.reset_btn = QToolButton()
        self.reset_btn.setText("\u21ba")
        self.reset_btn.setAutoRaise(True)
        self.reset_btn.setToolTip("Reset this section to its defaults")
        self.reset_btn.clicked.connect(self.resetRequested.emit)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(0)
        header.addWidget(self.toggle, 1)
        header.addWidget(self.reset_btn)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(header)
        lay.addWidget(self.body)
        lay.addWidget(line)
        self.toggle.toggled.connect(self.setExpanded)
        self.setExpanded(expanded)

    def setExpanded(self, on):
        self.toggle.setChecked(on)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.body.setVisible(on)

    def addRow(self, label, widget=None):
        if widget is None:
            self.form.addRow(label)
        else:
            self.form.addRow(label, widget)
        return widget if widget is not None else label

    def setRowVisible(self, widget, visible):
        """Show/hide a row (label and field) by its field widget."""
        label = self.form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)
        widget.setVisible(visible)


class SunDial(QWidget):
    """Top-down view of the lighting hemisphere. Centre = overhead (90 deg),
    rim = horizon (0 deg), north (azimuth 0) at the top. The disc is shaded
    like a sphere lit by the current key light, so the dial previews the
    lighting at a glance. Drag the sun to move it; scroll to nudge altitude."""

    lightChanged = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._az, self._alt = 315.0, 35.0
        self._markers = []
        self._fill = None
        self.setMinimumSize(150, 150)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setToolTip("Drag the sun to set the key light.\nCentre = overhead (90°), rim = horizon (0°).\n"
                        "Scroll to fine-tune altitude.")

    def sizeHint(self):
        return QSize(180, 180)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return w

    def light(self):
        return self._az, self._alt

    def setLight(self, azimuth, altitude, emit=False):
        self._az, self._alt = azimuth % 360.0, min(max(altitude, 0.0), 90.0)
        self.update()
        if emit:
            self.lightChanged.emit(self._az, self._alt)

    def setMarkers(self, markers):
        self._markers = list(markers or [])
        self.update()

    def setFill(self, fill):
        self._fill = fill
        self.update()

    def _geom(self):
        side = min(self.width(), self.height()) - 26
        return QPointF(self.width() / 2.0, self.height() / 2.0), max(side / 2.0, 10.0)

    def _point(self, az, alt):
        c, r = self._geom()
        rr = (90.0 - alt) / 90.0 * r
        a = math.radians(az)
        return QPointF(c.x() + rr * math.sin(a), c.y() - rr * math.cos(a))

    def _from_pos(self, pos):
        c, r = self._geom()
        dx, dy = pos.x() - c.x(), c.y() - pos.y()
        alt = 90.0 * (1.0 - min(math.hypot(dx, dy) / r, 1.0))
        az = math.degrees(math.atan2(dx, dy)) % 360.0
        return az, max(alt, 1.0)

    def mousePressEvent(self, e):
        self.setLight(*self._from_pos(e.position()), emit=True)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self.setLight(*self._from_pos(e.position()), emit=True)

    def wheelEvent(self, e):
        step = 1.0 if e.angleDelta().y() > 0 else -1.0
        self.setLight(self._az, min(max(self._alt + step, 1.0), 90.0), emit=True)

    def paintEvent(self, _e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c, r = self._geom()
        text = self.palette().color(QPalette.ColorRole.WindowText)

        # Sphere lit from the key light: highlight displaced toward the light.
        a, alt = math.radians(self._az), math.radians(self._alt)
        lx, ly = math.sin(a) * math.cos(alt), math.cos(a) * math.cos(alt)
        focal = QPointF(c.x() + lx * r * 0.8, c.y() - ly * r * 0.8)
        g = QRadialGradient(c, r * 1.05, focal)
        g.setColorAt(0.0, QColor(246, 240, 226))
        g.setColorAt(0.45, QColor(168, 162, 150))
        g.setColorAt(1.0, QColor(34, 40, 56))
        p.setBrush(QBrush(g))
        p.setPen(QPen(QColor(70, 82, 104), 1.4))
        p.drawEllipse(c, r, r)

        ring = QPen(QColor(255, 255, 255, 70), 1.0, Qt.PenStyle.DashLine)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(ring)
        for ring_alt in (30, 60):
            rr = (90 - ring_alt) / 90.0 * r
            p.drawEllipse(c, rr, rr)
        p.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
        p.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))

        f = QFont(self.font())
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1.5))
        f.setBold(True)
        p.setFont(f)
        p.setPen(text)
        for label, az in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
            pt = self._point(az, -11.0)
            p.drawText(QRectF(pt.x() - 8, pt.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, label)

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 206, 110, 150))
        for m_az, m_alt in self._markers:
            p.drawEllipse(self._point(m_az, m_alt), 3.0, 3.0)

        if self._fill:
            p.setPen(QPen(QColor(150, 190, 255), 1.6))
            p.setBrush(QColor(150, 190, 255, 60))
            p.drawEllipse(self._point(*self._fill), 5.0, 5.0)

        sun = self._point(self._az, self._alt)
        p.setPen(QPen(QColor(255, 190, 60, 170), 1.5))
        p.drawLine(c, sun)
        glow = QRadialGradient(sun, 18)
        glow.setColorAt(0.0, QColor(255, 220, 120, 230))
        glow.setColorAt(1.0, QColor(255, 180, 40, 0))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(glow))
        p.drawEllipse(sun, 18, 18)
        p.setPen(QPen(QColor(255, 196, 70), 1.6))
        for i in range(8):
            t = i * math.pi / 4
            p.drawLine(QPointF(sun.x() + 8 * math.cos(t), sun.y() + 8 * math.sin(t)),
                       QPointF(sun.x() + 12 * math.cos(t), sun.y() + 12 * math.sin(t)))
        p.setBrush(QColor(255, 214, 90))
        p.setPen(QPen(QColor(150, 90, 10), 1.0))
        p.drawEllipse(sun, 5.5, 5.5)
        p.end()


def _qimage(img):
    """float [0, 1] (H, W) or (H, W, 3) -> (QImage, backing uint8 buffer)."""
    a = np.ascontiguousarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))
    h, w = a.shape[:2]
    if a.ndim == 2:
        return QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8), a
    return QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888), a


class PreviewCanvas(QWidget):
    """Zoomable, pannable preview of a full-resolution raster.

    The view is (cx, cy) in full-resolution pixels at the widget centre and a
    scale in screen pixels per full-resolution pixel (1.0 = 1:1, 2.0 = 2:1).
    Rendered images arrive as layers with their own full-resolution rectangle:
    a whole-surface 'overview' for context and a sharp 'detail' of the view,
    so pan and zoom respond instantly while sharper renders are on their way.
    With compare on, a second image shows left of a draggable divider."""

    lightDragged = pyqtSignal(float, float)
    viewChanged = pyqtSignal()
    measured = pyqtSignal(str, float, float, float, float)     # 'line' / 'box', full-resolution x0, y0, x1, y1
    MAX_SCALE = 8.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = None
        self._scale, self._cx, self._cy, self._fitted = 1.0, 0.0, 0.0, True
        self._layers = {}
        self._compare, self._cmp_label, self._split = False, "", 0.5
        self._tool, self._drag = "pan", None
        self._measure = None                 # ('line' | 'box', x0, y0, x1, y1) in full-resolution pixels
        self._lines = []                     # traced polylines [(kind, [(x, y), ...])] in full-resolution pixels
        self._busy, self._info, self._readout = False, "", ""
        self._message = "Choose a height map to start the live preview."
        self._unit_px, self._unit = None, ""
        self._overlay = []
        self._hint_until = 0.0
        self._toast = ""
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(lambda: self.flash(""))
        self._previous = {}                    # layers as they were before the last burst of changes
        self._cmp_source = "cmp"
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_cursor()

    # ---- content
    def setFullSize(self, width, height):
        self._full = (int(width), int(height))
        self._layers.clear()
        self._message = ""
        self._hint_until = time.monotonic() + 10.0      # the mouse hint fades after a few seconds
        QTimer.singleShot(10200, self.update)
        self.fit()

    def clear(self):
        self._full = None
        self._layers.clear()
        self._readout = self._info = ""
        self.update()

    def setLayer(self, kind, img=None, rect=None, cmp=None, probe=None, px_scale=1.0):
        if img is None:
            if self._layers.pop(kind, None) is not None:
                self.update()
            return
        q, buf = _qimage(img)
        layer = dict(img=img, qimg=q, buf=buf, rect=QRectF(*rect), probe=probe, px=float(px_scale),
                     cmp_q=None, cmp_buf=None)
        if cmp is not None:
            layer["cmp_q"], layer["cmp_buf"] = _qimage(cmp)
        self._layers[kind] = layer
        self._message = ""
        self.update()

    def hasLayer(self, kind):
        return kind in self._layers

    def currentImage(self):
        layer = self._layers.get("detail") or self._layers.get("overview")
        return layer["qimg"].copy() if layer else None

    def setMessage(self, text):
        self._message = text
        self.update()

    def setBusy(self, busy):
        self._busy = busy
        self.update()

    def setInfo(self, text):
        self._info = text
        self.update()

    def setCompare(self, on, label="", source="cmp"):
        """source 'cmp' = comparison image rendered with each layer,
        'previous' = the render before the last change."""
        self._compare, self._cmp_label, self._cmp_source = bool(on), label, source
        self.update()

    def snapshotPrevious(self):
        self._previous = {k: dict(qimg=v["qimg"], rect=QRectF(v["rect"]), px=v["px"])
                          for k, v in self._layers.items() if k != "draft"}

    def flash(self, text, ms=1400):
        """Short-lived message in the middle of the top edge (what just changed)."""
        self._toast = text
        if text:
            self._toast_timer.start(ms)
        self.update()

    def setTool(self, tool):
        self._tool = tool
        self._update_cursor()
        self.update()

    def setPixelSize(self, size, unit):
        self._unit_px = size if size and size > 0 else None
        self._unit = unit
        self.update()

    # ---- view
    def scale(self):
        return self._scale

    def fitScale(self):
        if not self._full:
            return 1.0
        return min(self.width() / float(self._full[0]), self.height() / float(self._full[1]))

    def _min_scale(self):
        return min(self.fitScale(), 1.0)

    def fit(self):
        if not self._full:
            return
        self._scale = min(self.fitScale(), self.MAX_SCALE)
        self._cx, self._cy = self._full[0] / 2.0, self._full[1] / 2.0
        self._fitted = True
        self._changed()

    def zoomTo(self, scale, anchor=None):
        if not self._full:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        ix, iy = self._to_image(anchor)
        self._scale = min(max(scale, self._min_scale()), self.MAX_SCALE)
        self._cx = ix - (anchor.x() - self.width() / 2.0) / self._scale
        self._cy = iy - (anchor.y() - self.height() / 2.0) / self._scale
        self._fitted = abs(self._scale - self.fitScale()) < 1e-6
        self._changed()

    def zoomBy(self, factor, anchor=None):
        self.zoomTo(self._scale * factor, anchor)

    def showWindow(self, window):
        """Zoom so the full-resolution window (x, y, w, h) fills the view."""
        x, y, w, h = window
        if not self._full or w <= 0 or h <= 0:
            return
        self._scale = min(max(min(self.width() / float(w), self.height() / float(h)), self._min_scale()),
                          self.MAX_SCALE)
        self._cx, self._cy = x + w / 2.0, y + h / 2.0
        self._fitted = False
        self._changed()

    def viewWindow(self):
        """Visible part of the raster as an integer full-resolution window."""
        if not self._full:
            return None
        w, h = self._full
        hw, hh = self.width() / (2.0 * self._scale), self.height() / (2.0 * self._scale)
        x0, y0 = max(0, int(math.floor(self._cx - hw))), max(0, int(math.floor(self._cy - hh)))
        x1, y1 = min(w, int(math.ceil(self._cx + hw))), min(h, int(math.ceil(self._cy + hh)))
        return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    def _changed(self):
        self._clamp()
        self.update()
        self.viewChanged.emit()

    def _clamp(self):
        w, h = self._full
        hw, hh = self.width() / (2.0 * self._scale), self.height() / (2.0 * self._scale)
        self._cx = w / 2.0 if 2 * hw >= w else min(max(self._cx, hw), w - hw)
        self._cy = h / 2.0 if 2 * hh >= h else min(max(self._cy, hh), h - hh)

    def _to_screen(self, x, y):
        return QPointF(self.width() / 2.0 + (x - self._cx) * self._scale,
                       self.height() / 2.0 + (y - self._cy) * self._scale)

    def _to_image(self, pt):
        return ((pt.x() - self.width() / 2.0) / self._scale + self._cx,
                (pt.y() - self.height() / 2.0) / self._scale + self._cy)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._full:
            if self._fitted:
                self.fit()
            else:
                self._changed()

    # ---- interaction
    def _update_cursor(self):
        self.setCursor(Qt.CursorShape.OpenHandCursor if self._tool == "pan" else Qt.CursorShape.CrossCursor)

    def _near_divider(self, pos):
        return self._compare and abs(pos.x() - self._split * self.width()) < 10

    def mousePressEvent(self, e):
        if not self._full:
            return
        pos, btn = e.position(), e.button()
        if btn == Qt.MouseButton.LeftButton and self._near_divider(pos):
            self._drag = ("split",)
            return
        if btn == Qt.MouseButton.LeftButton:
            action = self._tool
        elif btn == Qt.MouseButton.RightButton:
            action = "light" if self._tool == "pan" else "pan"
        elif btn == Qt.MouseButton.MiddleButton:
            action = "pan"
        else:
            return
        if action == "pan":
            self._drag = ("pan", pos, self._cx, self._cy)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif action == "measure":
            x, y = self._to_image(pos)
            kind = "box" if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else "line"
            self._measure = (kind, x, y, x, y)
            self._drag = ("measure",)
            self.update()
        else:
            self._drag = ("light",)
            self._emit_light(pos)

    def mouseMoveEvent(self, e):
        pos = e.position()
        if self._drag:
            kind = self._drag[0]
            if kind == "split":
                self._split = min(max(pos.x() / max(self.width(), 1), 0.02), 0.98)
                self.update()
            elif kind == "pan":
                _, start, cx, cy = self._drag
                self._cx = cx - (pos.x() - start.x()) / self._scale
                self._cy = cy - (pos.y() - start.y()) / self._scale
                self._fitted = abs(self._scale - self.fitScale()) < 1e-6
                self._changed()
            elif kind == "measure" and self._measure:
                x, y = self._to_image(pos)
                k, x0, y0, _x, _y = self._measure
                self._measure = (k, x0, y0, x, y)
                self.update()
            else:
                self._emit_light(pos)
        elif self._near_divider(pos):
            self.setCursor(Qt.CursorShape.SplitHCursor)
        else:
            self._update_cursor()
        self._update_readout(pos)

    def mouseReleaseEvent(self, _e):
        if self._drag and self._drag[0] == "measure" and self._measure:
            k, x0, y0, x1, y1 = self._measure
            if math.hypot(x1 - x0, y1 - y0) * self._scale >= 3:
                self.measured.emit(k, x0, y0, x1, y1)
            else:
                self._measure = None
                self.update()
        self._drag = None
        self._update_cursor()

    def setLines(self, lines):
        """Traced lines drawn over the preview: [(kind 'incised' | 'raised', [(x, y) full-resolution px])]."""
        self._lines = list(lines or [])
        self.update()

    def _draw_lines(self, p):
        from qgis.PyQt.QtGui import QPolygonF
        colours = {"incised": QColor(230, 40, 40, 230), "raised": QColor(40, 110, 240, 230)}
        for kind, pts in self._lines:
            p.setPen(QPen(colours.get(kind, QColor(230, 40, 40, 230)), 1.6))
            p.drawPolyline(QPolygonF([self._to_screen(x, y) for x, y in pts]))

    def setMeasure(self, m):
        """Show a measurement - ('line' | 'box', x0, y0, x1, y1) in full-resolution pixels - or None."""
        self._measure = tuple(m) if m else None
        self.update()

    def measurement(self):
        return self._measure

    def _draw_measure(self, p):
        k, x0, y0, x1, y1 = self._measure
        a, b = self._to_screen(x0, y0), self._to_screen(x1, y1)
        rect = QRectF(a, b).normalized()
        for pen in (QPen(QColor(0, 0, 0, 170), 4), QPen(QColor(255, 214, 90), 2)):
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if k == "line":
                p.drawLine(a, b)
                p.drawEllipse(a, 3, 3)
                p.drawEllipse(b, 3, 3)
            else:
                p.drawRect(rect)

        def size(v):
            return f"{v * self._unit_px:.4g} {self._unit}".strip() if self._unit_px else f"{v:.0f} px"

        if k == "line":
            text = size(math.hypot(x1 - x0, y1 - y0))
            at = QPointF((a.x() + b.x()) / 2.0 + 10, (a.y() + b.y()) / 2.0 + 6)
        else:
            text = f"{size(abs(x1 - x0))} × {size(abs(y1 - y0))}"
            at = QPointF(rect.right() + 8, rect.top())
        self._pill(p, text, at, Qt.AlignmentFlag.AlignLeft, QColor(255, 214, 90))

    def contextMenuEvent(self, e):
        e.accept()            # right-drag moves the light / pans

    def mouseDoubleClickEvent(self, e):
        if not self._full:
            return
        if self._scale < 0.999:
            self.zoomTo(1.0, e.position())
        else:
            self.fit()

    def wheelEvent(self, e):
        if not self._full:
            return
        d = e.angleDelta().y() or e.pixelDelta().y()
        if d:
            self.zoomBy(1.0015 ** d, e.position())

    def keyPressEvent(self, e):
        k, K = e.key(), Qt.Key
        step = 0.15 * min(self.width(), self.height()) / max(self._scale, 1e-9)
        if k in (K.Key_Plus, K.Key_Equal):
            self.zoomBy(1.25)
        elif k == K.Key_Minus:
            self.zoomBy(0.8)
        elif k in (K.Key_0, K.Key_F):
            self.fit()
        elif k in (K.Key_1, K.Key_2, K.Key_4):
            self.zoomTo({K.Key_1: 1.0, K.Key_2: 2.0, K.Key_4: 4.0}[k])
        elif k in (K.Key_Left, K.Key_Right, K.Key_Up, K.Key_Down) and self._full:
            self._cx += {K.Key_Left: -step, K.Key_Right: step}.get(k, 0.0)
            self._cy += {K.Key_Up: -step, K.Key_Down: step}.get(k, 0.0)
            self._fitted = False
            self._changed()
        else:
            super().keyPressEvent(e)

    def leaveEvent(self, _e):
        if self._readout:
            self._readout = ""
            self.update()

    def _emit_light(self, pos):
        dx, dy = pos.x() - self.width() / 2.0, self.height() / 2.0 - pos.y()
        radius = min(self.width(), self.height()) / 2.0
        alt = 90.0 * (1.0 - min(math.hypot(dx, dy) / radius, 1.0))
        self.lightDragged.emit(math.degrees(math.atan2(dx, dy)) % 360.0, max(alt, 2.0))

    def _update_readout(self, pos):
        text = ""
        if self._full and self._layers:
            x, y = self._to_image(pos)
            if 0 <= x < self._full[0] and 0 <= y < self._full[1]:
                for kind in ("detail", "overview"):
                    layer = self._layers.get(kind)
                    if layer is None or not layer["rect"].contains(QPointF(x, y)):
                        continue
                    img = layer["img"]
                    i = min(int((y - layer["rect"].y()) / layer["px"]), img.shape[0] - 1)
                    j = min(int((x - layer["rect"].x()) / layer["px"]), img.shape[1] - 1)
                    v = img[i, j]
                    tone = f"{float(v) * 255:.0f}" if np.ndim(v) == 0 else "/".join(f"{c * 255:.0f}" for c in v)
                    probe, zt = layer["probe"], ""
                    if probe is not None and probe.shape[:2] == img.shape[:2] and np.isfinite(probe[i, j]):
                        zt = f"   z {float(probe[i, j]):.5g}"
                    text = f"x {x:,.0f}  y {y:,.0f}{zt}   tone {tone}"
                    break
        if text != self._readout:
            self._readout = text
            self.update()

    # ---- painting
    def _draw(self, p, key, layers=None):
        layers = self._layers if layers is None else layers
        for kind in ("overview", "detail", "draft"):
            layer = layers.get(kind)
            if layer is None or layer.get(key) is None:
                continue
            r = layer["rect"]
            tl = self._to_screen(r.x(), r.y())
            target = QRectF(tl.x(), tl.y(), r.width() * self._scale, r.height() * self._scale)
            # Smooth when shrinking, crisp pixels once magnified 2x or more.
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self._scale * layer["px"] < 2.0)
            p.drawImage(target, layer[key])

    def paintEvent(self, _e):
        p = QPainter(self)
        bg = QColor(30, 32, 38)
        p.fillRect(self.rect(), bg)
        has = bool(self._full and self._layers)
        if has:
            self._draw(p, "qimg")
            prev = self._cmp_source == "previous"
            if self._compare and ((prev and self._previous) or
                                  (not prev and any(l.get("cmp_q") is not None for l in self._layers.values()))):
                x = self._split * self.width()
                p.save()
                p.setClipRect(QRectF(0, 0, x, self.height()))
                p.fillRect(self.rect(), bg)
                if prev:
                    self._draw(p, "qimg", self._previous)
                else:
                    self._draw(p, "cmp_q")
                p.restore()
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                p.setPen(QPen(QColor(255, 255, 255, 230), 2))
                p.drawLine(QPointF(x, 0), QPointF(x, self.height()))
                c = QPointF(x, self.height() / 2.0)
                p.setBrush(QColor(255, 255, 255, 235))
                p.setPen(QPen(QColor(40, 44, 56), 1))
                p.drawEllipse(c, 11, 11)
                p.drawText(QRectF(c.x() - 11, c.y() - 11, 22, 22), Qt.AlignmentFlag.AlignCenter, "↔")
                self._pill(p, self._cmp_label, QPointF(x - 8, 44), Qt.AlignmentFlag.AlignRight)
                self._pill(p, "Render", QPointF(x + 8, 44), Qt.AlignmentFlag.AlignLeft)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._message:
            p.setPen(QColor(200, 204, 214))
            p.drawText(self.rect().adjusted(24, 24, -24, -24),
                       Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self._message)
        if has:
            hint = {"pan": "Drag: pan · Right-drag: light", "light": "Drag: light · Right-drag: pan",
                    "measure": "Drag: profile · Shift-drag: area · Right-drag: pan"}.get(self._tool, "") + \
                " · Scroll: zoom · Double-click: 1:1 / fit"
            if time.monotonic() < self._hint_until:
                self._corner(p, hint, "bl")
            self._scalebar(p)
            if self._readout:
                self._corner(p, self._readout, "br")
            if self._info:
                self._corner(p, self._info, "tr")
        if self._busy:
            self._corner(p, "Rendering…", "tl", QColor(255, 196, 70))
        if has and self._lines:
            self._draw_lines(p)
        if has and self._measure:
            self._draw_measure(p)
        if has and self._overlay:
            self._draw_overlay(p)
        if self._toast:
            f = QFont(self.font())
            f.setPointSizeF(f.pointSizeF() + 3)
            f.setBold(True)
            p.setFont(f)
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(self._toast) + 28
            r = QRectF((self.width() - w) / 2.0, self.height() * 0.42, w, fm.height() + 14)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 175))
            p.drawRoundedRect(r, 10, 10)
            p.setPen(QColor(255, 214, 120))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, self._toast)
        p.end()

    def setOverlay(self, lines):
        """Settings summary drawn top-left over the preview (first line highlighted)."""
        self._overlay = list(lines or [])
        self.update()

    def _draw_overlay(self, p):
        fm = p.fontMetrics()
        lh = fm.height()
        limit = int(self.width() * 0.62)
        lines = [fm.elidedText(t, Qt.TextElideMode.ElideRight, limit) for t in self._overlay]
        w = max(fm.horizontalAdvance(t) for t in lines) + 20
        top = 8 + (lh + 16 if self._busy else 0)
        r = QRectF(8, top, w, lh * len(lines) + 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 165))
        p.drawRoundedRect(r, 8, 8)
        for i, t in enumerate(lines):
            p.setPen(QColor(255, 206, 110) if i == 0 else QColor(228, 230, 236))
            p.drawText(QRectF(r.x() + 10, r.y() + 6 + i * lh, w - 20, lh),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, t)

    def _pill(self, p, text, pt, align, colour=None):
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 16, fm.height() + 8
        x = pt.x() - w if align & Qt.AlignmentFlag.AlignRight else pt.x()
        r = QRectF(x, pt.y(), w, h)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 155))
        p.drawRoundedRect(r, h / 2.0, h / 2.0)
        p.setPen(colour or QColor(225, 228, 235))
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)
        return r

    def _corner(self, p, text, corner, colour=None):
        h = p.fontMetrics().height() + 8
        left, top = corner in ("tl", "bl"), corner in ("tl", "tr")
        pt = QPointF(8 if left else self.width() - 8, 8 if top else self.height() - h - 8)
        return self._pill(p, text, pt, Qt.AlignmentFlag.AlignLeft if left else Qt.AlignmentFlag.AlignRight, colour)

    def _scalebar(self, p):
        if not self._unit_px:
            return
        units = 110.0 / self._scale * self._unit_px
        if units <= 0:
            return
        mag = 10.0 ** math.floor(math.log10(units))
        nice = max(m * mag for m in (1, 2, 5) if m * mag <= units)
        length = nice / self._unit_px * self._scale
        fm = p.fontMetrics()
        h = fm.height()
        label = f"{nice:g} {self._unit}".strip()
        width = max(length, fm.horizontalAdvance(label))
        x0, y0 = 16.0, self.height() - (h + 8) - 8 - 22
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 155))
        p.drawRoundedRect(QRectF(x0 - 8, y0 - h - 6, width + 16, h + 16), 6, 6)
        p.setBrush(QColor(240, 240, 244))
        p.drawRect(QRectF(x0, y0, length, 3))
        p.drawRect(QRectF(x0, y0 - 4, 1.5, 7))
        p.drawRect(QRectF(x0 + length - 1.5, y0 - 4, 1.5, 7))
        p.setPen(QColor(235, 236, 240))
        p.drawText(QRectF(x0, y0 - h - 3, width, h), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   label)


def _nice_ticks(lo, hi, n=5):
    """Round tick values covering lo..hi."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(n, 1)
    mag = 10.0 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    first = math.ceil(lo / step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 1)]


class ProfileView(QWidget):
    """A height profile: the heights along a measured line, their straight trend (dashed - the surface the marks
    are cut into) and the deepest point below it, with axes in the height map's units. The pointer reads out
    distance and height."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._d = self._h = self._trend = self._mark = None
        self._ux, self._uz, self._hover = "px", "", None
        self.setMinimumHeight(180)
        self.setMouseTracking(True)

    def setProfile(self, dist, heights, trend=None, mark=None, unit_xy="px", unit_z=""):
        """dist / heights arrays; trend = (h at start, h at end); mark = (distance, height, label)."""
        self._d, self._h = np.asarray(dist, float), np.asarray(heights, float)
        self._trend, self._mark, self._ux, self._uz = trend, mark, unit_xy, unit_z
        self.update()

    def clear(self):
        self._d = self._h = None
        self.update()

    def data(self):
        return None if self._d is None else (self._d, self._h)

    def _frame(self):
        return QRectF(58, 10, max(10.0, self.width() - 70.0), max(10.0, self.height() - 40.0))

    def mouseMoveEvent(self, e):
        self._hover = e.position().x() if self._d is not None else None
        self.update()

    def leaveEvent(self, _e):
        self._hover = None
        self.update()

    def paintEvent(self, _e):
        from qgis.PyQt.QtGui import QPolygonF
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(250, 250, 252))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._d is None or len(self._d) < 2:
            p.setPen(QColor(120, 122, 132))
            p.drawText(self.rect().adjusted(12, 12, -12, -12), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                       "Choose the Measure tool and drag a line across the preview - its height profile appears here.")
            p.end()
            return
        r, d, h = self._frame(), self._d, self._h
        lo, hi = float(np.nanmin(h)), float(np.nanmax(h))
        pad = (hi - lo) * 0.08 or max(abs(hi), 1.0) * 1e-3
        lo, hi = lo - pad, hi + pad
        d0, d1 = float(d[0]), float(d[-1]) if d[-1] > d[0] else float(d[0]) + 1.0

        def pt(x, y):
            return QPointF(r.left() + (x - d0) / (d1 - d0) * r.width(), r.bottom() - (y - lo) / (hi - lo) * r.height())

        fm = p.fontMetrics()
        p.setPen(QColor(222, 224, 230))
        for v in _nice_ticks(lo, hi):
            y = pt(d0, v).y()
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
        p.setPen(QColor(90, 94, 106))
        for v in _nice_ticks(lo, hi):
            y = pt(d0, v).y()
            p.drawText(QRectF(0, y - fm.height() / 2, r.left() - 6, fm.height()),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{v:.4g}")
        for v in _nice_ticks(d0, d1):
            x = pt(v, lo).x()
            p.drawText(QRectF(x - 40, r.bottom() + 3, 80, fm.height()), Qt.AlignmentFlag.AlignHCenter, f"{v:.4g}")
        p.drawText(QRectF(r.left(), r.bottom() + 3 + fm.height(), r.width(), fm.height()),
                   Qt.AlignmentFlag.AlignRight, f"distance ({self._ux})   height ({self._uz or 'units'})")
        p.setPen(QColor(170, 172, 182))
        p.drawRect(r)
        if self._trend is not None:
            pen = QPen(QColor(150, 150, 160), 1)
            pen.setStyle(Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawLine(pt(d0, self._trend[0]), pt(d1, self._trend[1]))
        p.setPen(QPen(QColor(30, 70, 140), 1.6))
        p.drawPolyline(QPolygonF([pt(float(x), float(y)) for x, y in zip(d, h)]))
        if self._mark is not None:
            md, mh, label = self._mark
            c = pt(md, mh)
            p.setPen(QPen(QColor(200, 60, 40), 2))
            p.drawEllipse(c, 4, 4)
            p.drawText(QRectF(c.x() + 6, c.y() + 2, 220, fm.height()), Qt.AlignmentFlag.AlignLeft, label)
        if self._hover is not None and r.left() <= self._hover <= r.right():
            x = d0 + (self._hover - r.left()) / r.width() * (d1 - d0)
            i = int(np.clip(np.searchsorted(d, x), 0, len(d) - 1))
            c = pt(float(d[i]), float(h[i]))
            p.setPen(QPen(QColor(0, 0, 0, 90), 1))
            p.drawLine(QPointF(c.x(), r.top()), QPointF(c.x(), r.bottom()))
            p.setPen(QColor(20, 20, 30))
            p.drawText(QRectF(r.left() + 6, r.top() + 4, r.width() - 12, fm.height()), Qt.AlignmentFlag.AlignLeft,
                       f"{d[i]:.4g} {self._ux}  ·  {h[i]:.5g} {self._uz}")
        p.end()


class ShadeStrip(QScrollArea):
    """Vertical strip of clickable thumbnails - shade types, frequency bands or
    lights - rendered from exactly what the preview shows."""

    clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        self._lay = QVBoxLayout(inner)
        self._lay.setContentsMargins(2, 2, 2, 2)
        self._lay.setSpacing(2)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._lay.addWidget(self._status)
        self._lay.addStretch(1)
        self.setWidget(inner)
        self._buttons, self._bufs, self._current = {}, [], None

    def setItems(self, items, current=None):
        """items: [(key, label, float image), ...]"""
        for b in self._buttons.values():
            b.setParent(None)
            b.deleteLater()
        self._buttons, self._bufs = {}, []
        for key, label, img in items:
            q, buf = _qimage(img)
            self._bufs.append(buf)
            pix = QPixmap.fromImage(q.copy())
            b = QToolButton()
            b.setCheckable(True)
            b.setAutoRaise(True)
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            b.setIcon(QIcon(pix))
            b.setIconSize(pix.size())
            # long names are elided to the thumbnail width so the strip never grows wider than the panel
            short = b.fontMetrics().elidedText(label, Qt.TextElideMode.ElideRight, pix.width())
            b.setText(short.replace("&", "&&"))
            b.setToolTip(label)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setMinimumWidth(pix.width() + 8)
            b.setFixedHeight(b.sizeHint().height())      # keep image + name; the strip scrolls instead of squashing
            b.clicked.connect(lambda _c=False, k=key: self._click(k))
            self._lay.insertWidget(self._lay.count() - 1, b)
            self._buttons[key] = b
        self.setCurrent(current)

    def _click(self, key):
        self.setCurrent(key)
        self.clicked.emit(key)

    def setCurrent(self, key):
        self._current = key
        for k, b in self._buttons.items():
            b.setChecked(k == key)

    def setStatus(self, text):
        self._status.setText(text)
        self._status.setVisible(bool(text))

    def keys(self):
        return list(self._buttons)


class SpectrumView(QWidget):
    """How much relief the surface holds at each feature size (radially averaged power, log-log), the fitted noise
    (when there is any to resolve), the bands kept - the band being edited between two handles you drag, the others
    shaded - and what the separation does at each size (the orange 'kept' curve, right-hand scale). A handle
    dragged off either end means 'no limit'."""

    bandChanged = pyqtSignal(float, float)          # finest, broadest size of the edited band (full-res px; 0 = none)
    BAND_COLOURS = ((255, 196, 60), (120, 200, 120), (170, 140, 230))

    def __init__(self, parent=None):
        super().__init__(parent)
        self._s = self._p = self._model = None
        self._bands, self._active = [(2.0, 64.0, True)], 0
        self._resp = None
        self._drag = None
        self.setMinimumHeight(190)
        self.setMouseTracking(True)
        self.setToolTip("Drag the two handles to choose the feature sizes kept. Blue: how much relief there is at "
                        "each size in the middle of the view. Orange: how much of it is kept (right-hand scale).")

    def setSpectrum(self, sizes, power, model=None):
        self._s, self._p, self._model = np.asarray(sizes, float), np.asarray(power, float), model
        self.update()

    def setBands(self, bands, active=0):
        """bands: [(finest, broadest, on)] full-resolution px (0 = no limit); active: the one the handles edit."""
        self._bands = [(float(a), float(b), bool(on)) for a, b, on in bands] or [(0.0, 0.0, True)]
        self._active = min(max(int(active), 0), len(self._bands) - 1)
        self.update()

    def setResponse(self, sizes, gains):
        self._resp = (np.asarray(sizes, float), np.asarray(gains, float)) if sizes is not None else None
        self.update()

    def band(self):
        return self._bands[self._active][:2]

    def _frame(self):
        return QRectF(50, 10, max(10.0, self.width() - 104.0), max(10.0, self.height() - 46.0))

    def _range(self):
        if self._s is None or not len(self._s):
            return 1.5, 1024.0
        return max(1.2, float(self._s.min()) * 0.8), float(self._s.max())

    def _x(self, size):
        r, (lo, hi) = self._frame(), self._range()
        v = min(max(size, lo), hi)
        return r.left() + (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * r.width()

    def _size(self, x):
        r, (lo, hi) = self._frame(), self._range()
        t = min(max((x - r.left()) / max(r.width(), 1.0), 0.0), 1.0)
        return math.exp(math.log(lo) + t * (math.log(hi) - math.log(lo)))

    def _span(self, i):
        r = self._frame()
        fine, broad, _on = self._bands[i]
        return (self._x(fine) if fine > 0 else r.left()), (self._x(broad) if broad > 0 else r.right())

    def mousePressEvent(self, e):
        x = e.position().x()
        fx, bx = self._span(self._active)
        self._drag = "fine" if abs(x - fx) <= abs(x - bx) else "broad"
        self._move(x)

    def mouseMoveEvent(self, e):
        if self._drag:
            self._move(e.position().x())
        else:
            fx, bx = self._span(self._active)
            near = min(abs(e.position().x() - fx), abs(e.position().x() - bx)) < 8
            self.setCursor(Qt.CursorShape.SizeHorCursor if near else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def _move(self, x):
        r, (lo, hi) = self._frame(), self._range()
        fine, broad, on = self._bands[self._active]
        if self._drag == "fine":
            fine = 0.0 if x < r.left() + 3 else round(min(self._size(x), broad / 1.25 if broad > 0 else hi), 2)
        else:
            broad = 0.0 if x > r.right() - 3 else round(max(self._size(x), fine * 1.25 if fine > 0 else lo), 1)
        self._bands[self._active] = (fine, broad, on)
        self.update()
        self.bandChanged.emit(fine, broad)

    def paintEvent(self, _e):
        from qgis.PyQt.QtGui import QPolygonF
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(250, 250, 252))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self._frame()
        fm = p.fontMetrics()
        lo, hi = self._range()
        if self._s is not None and len(self._s) >= 3:
            pos = self._p[self._p > 0]
            ylo, yhi = math.log(float(pos.min())), math.log(float(pos.max()))
            ylo -= 0.05 * (yhi - ylo)
        else:
            ylo, yhi = 0.0, 1.0

        def y(v):
            return r.bottom() - (math.log(max(v, 1e-300)) - ylo) / max(yhi - ylo, 1e-9) * r.height()

        m = self._model
        if m and m.get("detected") and m["crossover"] > lo:
            xc = self._x(m["crossover"])
            p.fillRect(QRectF(r.left(), r.top(), xc - r.left(), r.height()), QColor(220, 60, 60, 28))
        for i, (fine, broad, on) in enumerate(self._bands):
            if on or i == self._active:
                fx, bx = self._span(i)
                c = self.BAND_COLOURS[i % len(self.BAND_COLOURS)]
                p.fillRect(QRectF(fx, r.top(), bx - fx, r.height()), QColor(*c, 70 if on else 25))
        p.setPen(QColor(170, 172, 182))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(r)
        for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
            if lo <= v <= hi:
                x = self._x(v)
                p.setPen(QColor(226, 228, 234))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.setPen(QColor(90, 94, 106))
                p.drawText(QRectF(x - 40, r.bottom() + 2, 80, fm.height()), Qt.AlignmentFlag.AlignHCenter, f"{v}")
        p.setPen(QColor(90, 94, 106))
        p.drawText(QRectF(r.left(), r.bottom() + 2 + fm.height(), r.width(), fm.height()),
                   Qt.AlignmentFlag.AlignHCenter, "feature size, full-resolution px  (fine ← → broad)")
        p.save()
        p.translate(14, r.center().y())
        p.rotate(-90)
        p.drawText(QRectF(-70, -8, 140, 16), Qt.AlignmentFlag.AlignCenter, "relief (log)")
        p.restore()
        if self._s is None or len(self._s) < 3:
            p.setPen(QColor(120, 122, 132))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "The spectrum of the view appears here.")
        else:
            if m:
                sizes = np.geomspace(lo, hi, 48)
                pen = QPen(QColor(150, 152, 162), 1)
                pen.setStyle(Qt.PenStyle.DashLine)
                p.setPen(pen)
                p.drawPolyline(QPolygonF([QPointF(self._x(v), y(m["amp"] * v ** m["slope"])) for v in sizes]))
                if m.get("detected"):
                    pen.setColor(QColor(200, 70, 60))
                    p.setPen(pen)
                    p.drawPolyline(QPolygonF([QPointF(self._x(v), y(m["noise"] * v ** m["noise_slope"]))
                                              for v in sizes]))
            p.setPen(QPen(QColor(30, 70, 140), 1.8))
            p.drawPolyline(QPolygonF([QPointF(self._x(s), y(v)) for s, v in zip(self._s, self._p) if v > 0]))
            note = (f"noise outweighs relief below ≈ {m['crossover']:.1f} px" if m and m.get("detected")
                    else "no noise resolvable here - the finest detail is already smooth" if m else "")
            p.setPen(QColor(190, 60, 60) if m and m.get("detected") else QColor(110, 112, 122))
            p.drawText(QRectF(r.left() + 5, r.top() + 3, r.width() - 10, fm.height()), Qt.AlignmentFlag.AlignLeft,
                       note)
        if self._resp is not None:
            rs, rg = self._resp
            top = max(1.0, float(np.max(rg)) if len(rg) else 1.0)

            def yg(g):
                return r.bottom() - g / top * r.height() * 0.92

            p.setPen(QPen(QColor(225, 120, 20), 2.0))
            p.drawPolyline(QPolygonF([QPointF(self._x(s), yg(g)) for s, g in zip(rs, rg)]))
            p.setPen(QColor(190, 100, 20))
            for g in (0.0, 1.0, top) if top > 1.05 else (0.0, 0.5, 1.0):
                p.drawText(QRectF(r.right() + 4, yg(g) - fm.height() / 2, 34, fm.height()),
                           Qt.AlignmentFlag.AlignLeft, f"×{g:.2g}")
            p.save()
            p.translate(self.width() - 5, r.center().y())
            p.rotate(90)
            p.drawText(QRectF(-50, -8, 100, 16), Qt.AlignmentFlag.AlignCenter, "kept")
            p.restore()
        fine, broad, _on = self._bands[self._active]
        c = self.BAND_COLOURS[self._active % len(self.BAND_COLOURS)]
        fx, bx = self._span(self._active)
        for x, label, align in ((fx, f"{fine:.3g} px" if fine > 0 else "all", Qt.AlignmentFlag.AlignRight),
                                (bx, f"{broad:.3g} px" if broad > 0 else "all", Qt.AlignmentFlag.AlignLeft)):
            p.setPen(QPen(QColor(max(c[0] - 60, 0), max(c[1] - 60, 0), max(c[2] - 60, 0)), 2.2))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setBrush(QColor(*c))
            p.drawEllipse(QPointF(x, r.center().y()), 6, 6)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QColor(80, 60, 20))
            box = (QRectF(x - 76, r.bottom() - fm.height() - 4, 72, fm.height()) if align == Qt.AlignmentFlag.AlignRight
                   else QRectF(x + 4, r.bottom() - fm.height() - 4, 72, fm.height()))
            p.drawText(box, align, label)
        p.end()


from qgis.PyQt.QtWidgets import QCheckBox, QDialog, QPushButton  # noqa: E402  (the guide dialog)


class StartHere(QDialog):
    """The usual path through the plugin in numbered steps, each with buttons that do it."""

    def __init__(self, parent, steps, show_at_start=True, on_toggle=None):
        super().__init__(parent)
        self.setWindowTitle("Start here")
        v = QVBoxLayout(self)
        head = QLabel("<b>From a scan to a publishable relief</b><br>Each step's buttons do it for you; every control "
                      "has a tooltip that says more.")
        head.setWordWrap(True)
        v.addWidget(head)
        for i, (title, text, buttons) in enumerate(steps, start=1):
            box = QFrame()
            box.setFrameShape(QFrame.Shape.StyledPanel)
            h = QHBoxLayout(box)
            num = QLabel(f"<span style='font-size:20pt; color:#3a6fb0'><b>{i}</b></span>")
            num.setFixedWidth(30)
            h.addWidget(num)
            col = QVBoxLayout()
            col.addWidget(QLabel(f"<b>{title}</b>"))
            desc = QLabel(text)
            desc.setWordWrap(True)
            col.addWidget(desc)
            h.addLayout(col, 1)
            for label, callback in buttons:
                b = QPushButton(label)
                b.clicked.connect(lambda _c=False, cb=callback: cb())
                h.addWidget(b)
            v.addWidget(box)
        row = QHBoxLayout()
        self.check = QCheckBox("Show this guide when the plugin opens")
        self.check.setChecked(bool(show_at_start))
        if on_toggle is not None:
            self.check.toggled.connect(on_toggle)
        row.addWidget(self.check, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        v.addLayout(row)
        self.resize(680, 620)
