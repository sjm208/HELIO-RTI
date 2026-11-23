"""Inspector building blocks: folding sections, parameter rows, a segmented
switch, a histogram and the loupe (VIEW-002).

Every row shows its number and unit next to its slider, so a setting can be
read off and written down - the thing a measurement needs and a bare slider
does not give.  Double-click a slider (or its number) to put it back to its
default, as in the classic panel.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QAbstractSlider, QAbstractSpinBox, QButtonGroup, QComboBox,
                               QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QSlider, QToolButton, QVBoxLayout, QWidget)

from helio_qt.widgets import label


class VScroll(QWidget):
    """A panel that scrolls down, never sideways: its content is held to the
    panel's width, so a long row wraps or shrinks instead of being cut off."""

    def __init__(self, body: QWidget):
        from PySide6.QtWidgets import QScrollArea
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.area.setWidget(body)
        self.body = body
        body.setMinimumWidth(0)
        lay.addWidget(self.area)
        self.area.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        from PySide6.QtCore import QEvent
        if ev.type() == QEvent.Type.Resize:
            self.body.setFixedWidth(self.area.viewport().width())
        return False

    def ensureWidgetVisible(self, w):
        self.area.ensureWidgetVisible(w)


# What each folding section holds, for its header's tooltip (by its key).
SECTION_TIPS = {
    "cap_ball": "Find the chrome ball in the frames: its highlights give each light's direction.",
    "cap_lights": "Where each light was, found or loaded. Drag one on the dome to correct it.",
    "cap_frames": "The photographs of this capture. Untick one to leave it out of the solve.",
    "cap_corr": "Corrections made to the frames before the solve: brightness and alignment.",
    "hand_sel": "The light picked on the dome: its bearing and its height.",
    "hand_arrange": "Lay the lights out from a template: a ring, a grid, a known dome.",
    "hand_sym": "Mirror or turn every light at once.",
    "hand_ref": "A photograph of the real dome under the lights, to place them against.",
    "hand_check": "These lights against the lights the solve itself implies.",
    "hand_more": "Lights from elsewhere: EXIF, a video, another file; or export these.",
    "m_scale": "Millimetres per pixel from a known length, and a height scale from a known "
               "height.",
    "light": "Where the light is: drag on the dome, or set its bearing and height.",
    "settings": "This view's own settings.",
    "tone": "Brightness, contrast, gamma and fine detail of the picture on screen.",
    "adjust": "Whites, blacks, midtones, clarity, texture and haze, as in a photo editor.",
    "layer": "What the view is drawn on: grey shading, the colour relit, or roughness; and "
             "the confidence highlight.",
    "display": "Turn or mirror the picture: for a printing plate, or an object photographed the "
               "other way up.",
    "compare": "A second view beside this one, with a slider between them.",
    "states": "Views with their settings, saved to come back to.",
    "quick_export": "One click: a web viewer, QGIS, or this view as a file. Every other export, USDZ for AR and VR among them, is in Publish.",
    "measure_summary": "What has been measured: the scale, a height scale, the colour check.",
    "relief": "The 3D relief beside or instead of the image: surface, exaggeration, mesh and "
              "the section profile.",
    "loupe": "The pixels under the pointer, magnified, at full resolution.",
    "histogram": "How the tones of the view on screen are spread, with clipping marked.",
    "model3d": "The separate 3D window: open it, set its look, record a turntable.",
    "m3d_clay": "The clay finish of the separate 3D model.",
    "m3d_sharp": "Sharpen the fine detail on the 3D model's surface.",
    "p_solve": "How the photographs become normals, how normals become height, how the "
               "photographs record light, and the uneven-light correction for flat objects.",
    "p_prep": "Corrections to the photographs before the solve: brightness, alignment, "
              "contrast, grazing lights.",
    "p_light": "How the lights are modelled: far away, or near the object (a small dome).",
    "p_refine": "Extra passes that refine the normals after the solve.",
    "p_depth": "How height is built from the normals, smoothed and corrected.",
    "p_flat": "Take out the rig's own uneven light using photographs of a blank, even target.",
    "p_region": "Solve only part of the image, at full resolution.",
    "p_qc": "Checks on the solve: residuals, bias, a reference map, uncertainty.",
    "annotate": "Draw arrows, text, shapes and measurements on the view, in layers.",
    "batch_folders": "The capture folders to process, one after another.",
    "batch_cal": "Where each folder's lights come from.",
    "batch_exports": "What is written for every folder.",
    "batch_out": "Where the results go.",
    "meta_poi": "Points of interest on the 3D model, for viewers that show them.",
    "sketchfab_slots": "Which map goes to which of Sketchfab's material slots.",
}


class Section(QWidget):
    """A titled group that folds away.  Long explanations live in tooltips;
    the open/closed state is remembered by *key* (``store`` is a QSettings)."""
    toggled = Signal(bool)

    def __init__(self, title: str, key: str = "", store=None, open_: bool = True, badge: str = "",
                 tip: str = ""):
        super().__init__()
        self.key, self.store = key, store
        if store is not None and key:
            v = store.value(f"sections/{key}")
            if v is not None:
                open_ = str(v).lower() in ("true", "1")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        head = QHBoxLayout()
        head.setSpacing(6)
        self.button = QToolButton()
        self.button.setProperty("section", "true")
        self.button.setText(title.upper())
        self.button.setCheckable(True)
        self.button.setChecked(open_)
        self.button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.button.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setAccessibleName(title)
        about = tip or SECTION_TIPS.get(key, "") or (f"How {title.lower()} works"
                                                     if title == "How it works" else "")
        self.button.setToolTip((about + "\n" if about else "") + "Click to show or hide.")
        self.button.toggled.connect(self._toggled)
        head.addWidget(self.button)
        head.addStretch(1)
        self.badge = label(badge, "caption")
        head.addWidget(self.badge)
        lay.addLayout(head)
        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 0, 0, 2)
        self.body_lay.setSpacing(8)
        lay.addWidget(self.body)
        self.body.setVisible(open_)

    def add(self, w):
        self.body_lay.addWidget(w)
        return w

    def add_layout(self, lay):
        self.body_lay.addLayout(lay)
        return lay

    def is_open(self) -> bool:
        return self.button.isChecked()

    def set_open(self, on: bool):
        self.button.setChecked(on)

    def _toggled(self, on: bool):
        self.button.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)
        self.body.setVisible(on)
        if self.store is not None and self.key:
            self.store.setValue(f"sections/{self.key}", on)
        self.toggled.emit(on)


class _Slider(QSlider):
    reset = Signal()

    def mouseDoubleClickEvent(self, e):
        self.reset.emit()


class _Spin(QDoubleSpinBox):
    reset = Signal()

    def mouseDoubleClickEvent(self, e):
        self.reset.emit()


class ValueBar(QAbstractSlider):
    """A setting as one bar: its name on the left, its value on the right,
    and a fill showing where the value sits in its range (from the middle,
    when the range runs either side of zero).

    Drag sideways to change it: the bar's width is the whole range, and
    holding ⇧ makes it ten times finer.  A click without dragging opens the
    number for typing; a double-click puts the default back; the arrow keys
    step it (⇧ finer, ⌥ coarser), Home and End go to the ends, Backspace
    resets.  The scroll wheel is left to the panel, so scrolling past a bar
    never changes it.
    """
    reset = Signal()
    edit = Signal()

    def __init__(self, text: str, fmt, default_pos: int, bipolar_pos: int | None = None):
        super().__init__()
        from PySide6.QtCore import QTimer
        self.setOrientation(Qt.Orientation.Horizontal)
        self.text, self.fmt = text, fmt
        self.default_pos, self.zero_pos = default_pos, bipolar_pos
        self.setFixedHeight(26)
        self.setMinimumWidth(120)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._press = None                    # (x, value) where a drag started
        self._dragging = False
        self._fine = False
        self._click = QTimer(self)
        self._click.setSingleShot(True)
        self._click.timeout.connect(self.edit.emit)

    # ── mouse ────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self._press = (e.position().x(), self.value())
        self._fine = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self._dragging = False
        e.accept()

    def mouseMoveEvent(self, e):
        if self._press is None:
            return
        x0, v0 = self._press
        x = e.position().x()
        fine = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if fine != self._fine:                # rebase, so changing gear never jumps
            self._press = (x, self.value())
            x0, v0 = self._press
            self._fine = fine
        if not self._dragging and abs(x - x0) < 3:
            return
        if not self._dragging:
            self._dragging = True
            self._click.stop()
            self.setSliderDown(True)
        span = (self.maximum() - self.minimum()) * (0.1 if fine else 1.0)
        self.setValue(int(round(v0 + (x - x0) / max(1.0, float(self.width())) * span)))
        e.accept()

    def mouseReleaseEvent(self, e):
        if self._press is None:
            return
        if self._dragging:
            self.setSliderDown(False)         # sliderReleased: the value is final
        else:
            from PySide6.QtWidgets import QApplication
            self._click.start(QApplication.styleHints().mouseDoubleClickInterval())
        self._press, self._dragging = None, False
        e.accept()

    def mouseDoubleClickEvent(self, e):
        self._click.stop()
        self._press, self._dragging = None, False
        self.reset.emit()
        e.accept()

    def wheelEvent(self, e):
        e.ignore()

    # ── keys ─────────────────────────────────────────────────────────
    def keyPressEvent(self, e):
        k, mods = e.key(), e.modifiers()
        step = max(1, self.singleStep())
        if mods & Qt.KeyboardModifier.ShiftModifier:
            step = max(1, step // 10)
        elif mods & Qt.KeyboardModifier.AltModifier:
            step = step * 10
        if k in (Qt.Key.Key_Left, Qt.Key.Key_Down):
            self.setValue(self.value() - step)
        elif k in (Qt.Key.Key_Right, Qt.Key.Key_Up):
            self.setValue(self.value() + step)
        elif k == Qt.Key.Key_Home:
            self.setValue(self.minimum())
        elif k == Qt.Key.Key_End:
            self.setValue(self.maximum())
        elif k in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.reset.emit()
            return
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.edit.emit()
            return
        else:
            return super().keyPressEvent(e)
        e.accept()                            # valueChanged, not dragging: a finished change

    # ── drawing ──────────────────────────────────────────────────────
    def paintEvent(self, _e):
        from helio_qt import theme
        t = theme.CURRENT
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        lo, hi = self.minimum(), self.maximum()
        span = max(1, hi - lo)
        frac = (self.value() - lo) / span
        hot = self.hasFocus() or self.underMouse() or self.isSliderDown()
        path = QPainterPath()
        path.addRoundedRect(r, 5, 5)
        p.fillPath(path, QColor(t.well))
        p.save()
        p.setClipPath(path)
        fill = QColor(t.raised if not self.isSliderDown() else t.accent)
        if self.isSliderDown():
            fill.setAlpha(70)
        if self.zero_pos is not None:
            z = (self.zero_pos - lo) / span
            a, b = sorted((z, frac))
            p.fillRect(QRectF(r.left() + a * r.width(), r.top(), (b - a) * r.width(), r.height()), fill)
            p.setPen(QPen(QColor(t.text3), 1))
            zx = r.left() + z * r.width()
            p.drawLine(QPointF(zx, r.top() + 4), QPointF(zx, r.bottom() - 4))
        else:
            p.fillRect(QRectF(r.left(), r.top(), frac * r.width(), r.height()), fill)
        p.restore()
        p.setPen(QPen(QColor(t.accent if self.hasFocus() else (t.text3 if hot else t.line)), 1))
        p.drawPath(path)
        f = p.font()
        p.setPen(QColor(t.text2))
        p.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                   self.text)
        mono = p.font()
        mono.setFamilies(["SF Mono", "Menlo", "Monaco"])
        p.setFont(mono)
        p.setPen(QColor(t.text))
        vtext = self.fmt()
        p.drawText(r.adjusted(8, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                   vtext)
        if self.value() != self.default_pos:
            w = p.fontMetrics().horizontalAdvance(vtext)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(t.accent))
            p.drawEllipse(QPointF(r.right() - 8 - w - 7, r.center().y()), 2.5, 2.5)
        p.setFont(f)
        p.end()


class ParamRow(QWidget):
    """One setting, as a ValueBar: its name, its value with unit, the fill.

    ``changed(key, value, final)``: *final* is False while the bar is being
    dragged, so a slow view can wait for the release.  ``slider`` is the bar
    and ``spin`` the number, which opens over the bar for typing."""
    changed = Signal(str, float, bool)

    STEPS = 1000

    def __init__(self, control):
        super().__init__()
        self.c = control
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.name = label(control.label, "")          # kept for callers; the bar draws the name
        self.name.setVisible(False)
        self.spin = _Spin(self)
        self.spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spin.setDecimals(control.decimals)
        self.spin.setRange(control.lo, control.hi)
        self.spin.setSingleStep(control.step)
        self.spin.setSuffix(f" {control.unit}" if control.unit and control.unit != "°" else control.unit)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.spin.setKeyboardTracking(False)
        self.spin.setAccessibleName(control.label)
        self.spin.setToolTip(control.tip)
        self.spin.hide()
        bipolar = self._to_pos(0.0) if control.lo < 0 < control.hi else None
        self.slider = ValueBar(control.label, lambda: self.spin.textFromValue(self.spin.value())
                               + self.spin.suffix(), self._to_pos(control.default), bipolar)
        self.slider.setRange(0, self.STEPS)
        if control.step:
            self.slider.setSingleStep(max(1, int(round(control.step / max(control.hi - control.lo,
                                                                          1e-12) * self.STEPS))))
        self.slider.setPageStep(self.STEPS // 10)
        self.slider.setAccessibleName(control.label)
        self.slider.setAccessibleDescription(control.tip)
        self.slider.setToolTip((control.tip + "\n\n" if control.tip else "")
                               + "Drag to change (⇧ finer) · click to type · double-click resets")
        lay.addWidget(self.slider)
        self._quiet = False
        self.slider.valueChanged.connect(self._from_slider)
        self.slider.sliderReleased.connect(lambda: self.changed.emit(self.c.key, self.value(), True))
        self.slider.reset.connect(self.reset)
        self.slider.edit.connect(self._open_editor)
        self.spin.valueChanged.connect(self._from_spin)
        self.spin.reset.connect(self.reset)
        self.spin.editingFinished.connect(self._close_editor)
        self.set_value(control.default)

    def _open_editor(self):
        self.spin.setGeometry(self.slider.geometry())
        self.spin.show()
        self.spin.raise_()
        self.spin.setFocus(Qt.FocusReason.OtherFocusReason)
        self.spin.selectAll()

    def _close_editor(self):
        if self.spin.isVisible():
            self.spin.hide()
            self.slider.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.spin.isVisible():
            self.spin.setGeometry(self.slider.geometry())

    def _to_pos(self, v: float) -> int:
        c = self.c
        return int(round((float(v) - c.lo) / max(c.hi - c.lo, 1e-12) * self.STEPS))

    def _from_pos(self, p: int) -> float:
        c = self.c
        v = c.lo + (c.hi - c.lo) * p / self.STEPS
        if c.step:
            v = round(v / c.step) * c.step
        return int(round(v)) if c.integer else round(v, 6)

    def value(self) -> float:
        return self.spin.value()

    def set_value(self, v):
        self._quiet = True
        try:
            self.spin.setValue(float(v))
            self.slider.setValue(self._to_pos(v))
        finally:
            self._quiet = False
        self.slider.update()

    def reset(self):
        self.set_value(self.c.default)
        self.changed.emit(self.c.key, float(self.c.default), True)

    def _from_slider(self, p: int):
        if self._quiet:
            return
        v = self._from_pos(p)
        self._quiet = True
        self.spin.setValue(v)
        self._quiet = False
        self.changed.emit(self.c.key, float(v), not self.slider.isSliderDown())

    def _from_spin(self, v: float):
        if self._quiet:
            return
        self._quiet = True
        self.slider.setValue(self._to_pos(v))
        self._quiet = False
        self.slider.update()
        self.changed.emit(self.c.key, float(v), True)


class ViewPicker(QWidget):
    """The view, as a dropdown that searches.

    Shows the current view and its family; a click (or Space) opens the
    families of views with the search field on top, typing narrows them,
    and choosing one closes it.  The search field and tree are the
    workspace's own (``InspectWorkspace.search`` / ``.tree``)."""

    def __init__(self, search, tree):
        super().__init__()
        from PySide6.QtWidgets import QFrame as _F
        self.search, self.tree = search, tree
        self.setToolTip("Choose a view: type to find one.  E next · ⇧E previous")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton("")
        self.button.setObjectName("ViewPicker")
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.setAccessibleName("View")
        self.button.setToolTip("Choose a view: type to find one.  E next · ⇧E previous")
        self.button.setMinimumHeight(32)
        self.button.clicked.connect(self.open)
        bl = QHBoxLayout(self.button)
        bl.setContentsMargins(10, 0, 8, 0)
        bl.setSpacing(8)
        self.name_lbl = label("Choose a view", "")
        self.fam_lbl = label("", "caption")
        self.chev = label("▾", "caption")
        for lb in (self.name_lbl, self.fam_lbl, self.chev):
            lb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.name_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        bl.addWidget(self.name_lbl, 1)
        bl.addWidget(self.fam_lbl)
        bl.addWidget(self.chev)
        lay.addWidget(self.button)
        self.popup = _F(self, Qt.WindowType.Popup)
        self.popup.setObjectName("Palette")
        pl = QVBoxLayout(self.popup)
        pl.setContentsMargins(8, 8, 8, 8)
        pl.setSpacing(6)
        pl.addWidget(search)
        pl.addWidget(tree, 1)
        self.popup.resize(320, 460)
        tree.itemClicked.connect(self._chosen)
        tree.itemActivated.connect(self._chosen)
        search.returnPressed.connect(self._first)
        self.family = ""

    def set_current(self, name: str, family: str = ""):
        self.family = family
        self.name_lbl.setText(name or "Choose a view")
        self.name_lbl.setToolTip(name)
        self.fam_lbl.setText(family)
        self.button.setAccessibleName(f"View: {name}" if name else "View")

    def open(self):
        g = self.button.mapToGlobal(self.button.rect().bottomLeft())
        self.popup.resize(max(self.button.width(), 320), 460)
        self.popup.move(g.x(), g.y() + 4)
        self.popup.show()
        self.search.setFocus(Qt.FocusReason.PopupFocusReason)
        self.search.selectAll()

    def _chosen(self, item, _col=0):
        if item is not None and item.childCount() == 0:
            self.popup.hide()

    def _first(self):
        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            if top.isHidden():
                continue
            for j in range(top.childCount()):
                it = top.child(j)
                if not it.isHidden():
                    self.tree.setCurrentItem(it)
                    self.tree.itemActivated.emit(it, 0)
                    return


class ChoiceRow(QWidget):
    """A setting with named values (Normal Residual's output mode)."""
    changed = Signal(str, float, bool)

    def __init__(self, control):
        super().__init__()
        self.c = control
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(label(control.label, ""))
        self.combo = QComboBox()
        self.combo.setAccessibleName(control.label)
        self.combo.setToolTip(control.tip)
        for v, text in control.choices:
            self.combo.addItem(text, v)
        lay.addWidget(self.combo)
        self.combo.currentIndexChanged.connect(
            lambda i: self.changed.emit(self.c.key, float(self.combo.itemData(i)), True))

    def value(self) -> float:
        return float(self.combo.currentData())

    def set_value(self, v):
        self.combo.blockSignals(True)
        for i in range(self.combo.count()):
            if float(self.combo.itemData(i)) == float(v):
                self.combo.setCurrentIndex(i)
        self.combo.blockSignals(False)

    def reset(self):
        self.set_value(self.c.default)
        self.changed.emit(self.c.key, float(self.c.default), True)


def row_for(control):
    return ChoiceRow(control) if control.choices else ParamRow(control)


class Segmented(QFrame):
    """A row of mutually exclusive buttons: one choice, all visible."""
    chosen = Signal(str)

    def __init__(self, options: list[tuple[str, str]], tips: dict | None = None, columns: int = 0):
        super().__init__()
        self.setProperty("segmented", "true")
        from PySide6.QtWidgets import QGridLayout
        lay = QGridLayout(self) if columns else QHBoxLayout(self)
        lay.setContentsMargins(3, 3, 3, 3)
        lay.setSpacing(2)
        self._columns = columns
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for key, text in options:
            b = QPushButton(text)
            b.setCheckable(True)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            if tips and key in tips:
                b.setToolTip(tips[key])
            b.clicked.connect(lambda _c=False, k=key: self.chosen.emit(k))
            self.group.addButton(b)
            if columns:
                n = len(self.buttons)
                lay.addWidget(b, n // columns, n % columns)
            else:
                lay.addWidget(b)
            self.buttons[key] = b

    def set_current(self, key: str):
        """Light *key*'s button; a key not offered (or "") lights none: an
        exclusive group cannot be unchecked, so it lets go for a moment."""
        none = key not in self.buttons
        if none:
            self.group.setExclusive(False)
        for k, b in self.buttons.items():
            b.blockSignals(True)
            b.setChecked(k == key)
            b.blockSignals(False)
        if none:
            self.group.setExclusive(True)

    def current(self) -> str:
        for k, b in self.buttons.items():
            if b.isChecked():
                return k
        return ""


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering lo..hi, about *n* of them (1, 2, 2.5, 5 × 10^k)."""
    import math
    span = hi - lo
    if not (span > 0):
        return [lo]
    raw = span / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    out, v = [], first
    while v <= hi + step * 1e-6:
        out.append(round(v, 10))
        v += step
    return out


class ProfileChart(QWidget):
    """The section A–B through the relief, drawn as a section drawing reads:
    the cut filled below the surface line, real axes in mm (or px), the
    vertical exaggeration stated, the grooves and ridges it crosses marked
    and numbered.  The pointer reads height and distance and moves a marker
    along the line in 3D (``hovered``); two clicks measure between points.

    ``set_profile`` takes helio_qt.relief3d.ReliefView.section_profile's dict."""
    letters = ("A", "B")                       # what the ends are called (the section card names them)

    hovered = Signal(float)                # 0..1 from A to B, or -1 off the chart

    BLUE = QColor("#6FB7FF")
    GOLD = QColor("#E6A93E")
    L, R, T, B = 66, 22, 26, 30            # margins: the axes' labels live in them

    def __init__(self, height: int = 190):
        super().__init__()
        self.setMinimumHeight(height)
        self.setMouseTracking(True)
        self.setAccessibleName("Section profile")
        self.prof = None
        self.cursor_t = None
        self.picks: list[float] = []        # up to two points along it, 0..1
        self.lines: list[float] = []        # chain lines (or laid, ruling) found along it, 0..1

    # ── data ─────────────────────────────────────────────────────────
    def set_profile(self, prof):
        self.prof = prof or None
        self.picks = []
        if self.prof:
            self.setToolTip("Move along the section to read it; click two points to measure "
                            "between them.")
        self.update()

    def set_cursor(self, t):
        t = None if t is None or t < 0 else float(t)
        if t != self.cursor_t:
            self.cursor_t = t
            self.update()

    def set_lines(self, places):
        """Chain lines (or laid lines, ruling) to mark across the chart, 0..1 along the section."""
        self.lines = [float(v) for v in places or ()]
        self.update()

    def plot_rect(self) -> QRectF:
        return QRectF(self.L, self.T, max(10, self.width() - self.L - self.R),
                      max(10, self.height() - self.T - self.B))

    def _scales(self):
        import numpy as np
        pr = self.prof
        d, z = np.asarray(pr["dist"], float), np.asarray(pr["z"], float)
        r = self.plot_rect()
        lo, hi = float(z.min()), float(z.max())
        pad = (hi - lo) * 0.12 or 1.0
        lo, hi = lo - pad, hi + pad * 0.6
        xmax = float(d[-1]) or 1.0

        def X(v):
            return r.left() + r.width() * v / xmax

        def Y(v):
            return r.bottom() - r.height() * (v - lo) / (hi - lo)
        return d, z, lo, hi, xmax, X, Y, r

    def exaggeration(self):
        """How much taller than true the chart draws the height (same units only)."""
        pr = self.prof
        if not pr or pr["unit"].split()[0] != pr["length_unit"]:
            return None
        d, z, lo, hi, xmax, X, Y, r = self._scales()
        return (r.height() / (hi - lo)) / (r.width() / xmax)

    def value_at(self, t: float):
        import numpy as np
        d, z = np.asarray(self.prof["dist"], float), np.asarray(self.prof["z"], float)
        x = t * float(d[-1])
        return x, float(np.interp(x, d, z))

    # ── pointer ──────────────────────────────────────────────────────
    def _t_at(self, x: float):
        r = self.plot_rect()
        if not self.prof or not (r.left() - 2 <= x <= r.right() + 2):
            return None
        return max(0.0, min(1.0, (x - r.left()) / r.width()))

    def mouseMoveEvent(self, e):
        t = self._t_at(e.position().x())
        self.set_cursor(t)
        self.hovered.emit(-1.0 if t is None else t)

    def leaveEvent(self, e):
        self.set_cursor(None)
        self.hovered.emit(-1.0)
        super().leaveEvent(e)

    def mousePressEvent(self, e):
        t = self._t_at(e.position().x())
        if t is None:
            return
        self.picks = [t] if len(self.picks) >= 2 else self.picks + [t]
        self.update()

    def measurement(self):
        """Between the two clicked points: (distance, height difference) or None."""
        if not self.prof or len(self.picks) != 2:
            return None
        (x1, z1), (x2, z2) = (self.value_at(t) for t in self.picks)
        return abs(x2 - x1), z2 - z1

    # ── drawing ──────────────────────────────────────────────────────
    def paintEvent(self, _e):
        from PySide6.QtGui import QLinearGradient
        from helio_qt import theme
        t = theme.CURRENT
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        box = QPainterPath()
        box.addRoundedRect(QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0), 6, 6)
        p.fillPath(box, QColor(t.well))
        p.setPen(QPen(QColor(t.line), 1))
        p.drawPath(box)
        if not self.prof:
            p.setPen(QColor(t.text3))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Draw a section: press S in the 3D view, then drag across the relief")
            p.end()
            return
        pr = self.prof
        d, z, lo, hi, xmax, X, Y, r = self._scales()
        small = QFont(p.font())
        small.setPointSizeF(max(8.0, small.pointSizeF() - 1.5))
        mono = QFont(small)
        mono.setFamilies(["SF Mono", "Menlo", "Monaco"])
        # grid and axes
        p.setFont(mono)
        grid = QColor(t.line)
        grid.setAlpha(120)
        zu = pr["unit"].split()[0]
        for v in nice_ticks(lo, hi, 5):
            y = Y(v)
            p.setPen(QPen(grid, 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            p.setPen(QColor(t.text3))
            p.drawText(QRectF(0, y - 8, self.L - 16, 16),
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{v:.3g}")
        for v in nice_ticks(0.0, xmax, 6):
            x = X(v)
            p.setPen(QPen(grid, 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.setPen(QColor(t.text3))
            p.drawText(QRectF(x - 30, r.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, f"{v:.3g}")
        p.setFont(small)
        p.drawText(QRectF(r.right() - 140, r.bottom() + 15, 140, 14),
                   Qt.AlignmentFlag.AlignRight, f"distance ({pr['length_unit']})")
        p.drawText(QRectF(4, 4, 200, 16), Qt.AlignmentFlag.AlignLeft, f"height ({zu})")
        ve = self.exaggeration()
        if ve:
            p.drawText(QRectF(r.left(), 4, r.width(), 16), Qt.AlignmentFlag.AlignHCenter,
                       f"height drawn ×{ve:.0f}" if ve >= 2 else "height to scale")
        # the cut: filled below the surface line
        line = QPainterPath()
        line.moveTo(X(d[0]), Y(z[0]))
        for dv, zv in zip(d[1:], z[1:]):
            line.lineTo(X(dv), Y(zv))
        fill = QPainterPath(line)
        fill.lineTo(X(d[-1]), r.bottom())
        fill.lineTo(X(d[0]), r.bottom())
        fill.closeSubpath()
        g = QLinearGradient(0, r.top(), 0, r.bottom())
        top, bottom = QColor(self.BLUE), QColor(self.BLUE)
        top.setAlpha(110)
        bottom.setAlpha(18)
        g.setColorAt(0.0, top)
        g.setColorAt(1.0, bottom)
        p.fillPath(fill, g)
        p.setPen(QPen(self.BLUE, 2.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(line)
        # A and B, as on the relief
        bold = QFont(small)
        bold.setWeight(QFont.Weight.Bold)
        p.setFont(bold)
        for xv, zv, name in ((d[0], z[0], self.letters[0]), (d[-1], z[-1], self.letters[1])):
            c = QPointF(X(xv), Y(zv))
            p.setPen(QPen(QColor("#0E1B2C"), 1))
            p.setBrush(self.BLUE)
            p.drawEllipse(c, 8, 8)
            p.setPen(QColor("#0E1B2C"))
            p.drawText(QRectF(c.x() - 8, c.y() - 8, 16, 16), Qt.AlignmentFlag.AlignCenter, name)
        # grooves (▼) and ridges (▲), numbered as in the list
        for f in pr.get("features") or []:
            i = min(len(z) - 1, max(0, int(f["index"])))
            c = QPointF(X(d[i]), Y(z[i]))
            up = f["kind"] == "ridge"
            tri = QPainterPath()
            s = 5.0
            yy = c.y() - 9 if up else c.y() + 9
            if up:
                tri.moveTo(c.x(), yy - s); tri.lineTo(c.x() - s, yy + s * 0.6); tri.lineTo(c.x() + s, yy + s * 0.6)
            else:
                tri.moveTo(c.x(), yy + s); tri.lineTo(c.x() - s, yy - s * 0.6); tri.lineTo(c.x() + s, yy - s * 0.6)
            tri.closeSubpath()
            p.fillPath(tri, self.GOLD if not up else QColor("#46C79A"))
            p.setPen(QColor(t.text))
            p.drawText(QRectF(c.x() - 12, (yy + 7) if not up else (yy - 21), 24, 14),
                       Qt.AlignmentFlag.AlignCenter, str(f["n"]))
        # chain lines: a line across the chart at each, numbered along the top
        if self.lines:
            violet = QColor("#B48CF0")
            for k, tl in enumerate(self.lines, 1):
                x = X(tl * xmax)
                p.setPen(QPen(violet, 1.4, Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x, r.top() + 12), QPointF(x, r.bottom()))
                p.setPen(violet)
                p.drawText(QRectF(x - 12, r.top() - 2, 24, 14), Qt.AlignmentFlag.AlignCenter, str(k))
        # two clicked points: the distance and height between them
        p.setFont(mono)
        for tp in self.picks:
            x, zv = self.value_at(tp)
            p.setPen(QPen(self.GOLD, 1.2, Qt.PenStyle.DashLine))
            p.drawLine(QPointF(X(x), r.top()), QPointF(X(x), r.bottom()))
            p.setBrush(self.GOLD)
            p.drawEllipse(QPointF(X(x), Y(zv)), 3.5, 3.5)
        m = self.measurement()
        if m:
            self._badge(p, QPointF(r.left() + 16, r.top() + 2),
                        f"between the points: {m[0]:.3g} {pr['length_unit']} · Δh {m[1]:+.3g} {zu}")
        # the pointer, with its reading
        if self.cursor_t is not None:
            x, zv = self.value_at(self.cursor_t)
            p.setPen(QPen(QColor(t.text2), 1))
            p.drawLine(QPointF(X(x), r.top()), QPointF(X(x), r.bottom()))
            p.setPen(QPen(QColor("#FFFFFF"), 1.5))
            p.setBrush(self.GOLD)
            p.drawEllipse(QPointF(X(x), Y(zv)), 5, 5)
            self._badge(p, QPointF(min(X(x) + 8, r.right() - 150), r.top() + (22 if m else 2)),
                        f"{x:.3g} {pr['length_unit']} · h {zv:.3g} {zu}")
        p.end()

    def _badge(self, p, at: QPointF, text: str, right: bool = False):
        from helio_qt import theme
        t = theme.CURRENT
        fm = p.fontMetrics()
        w, h = fm.horizontalAdvance(text) + 12, fm.height() + 4
        x = at.x() - w if right else at.x()
        rr = QRectF(x, at.y(), w, h)
        path = QPainterPath()
        path.addRoundedRect(rr, 4, 4)
        bg = QColor(t.panel)
        bg.setAlpha(230)
        p.fillPath(path, bg)
        p.setPen(QColor(t.text))
        p.drawText(rr, Qt.AlignmentFlag.AlignCenter, text)

    # ── for saving ───────────────────────────────────────────────────
    def save_pictures(self, base: str, w: int = 1400, h: int = 520) -> list[str]:
        """The chart as *base*.svg (vector, for a figure) and *base*.png, drawn
        afresh at *w* × *h* off screen: the one on screen is left alone."""
        from PySide6.QtCore import QSize as _QSize
        from PySide6.QtGui import QImage
        twin = ProfileChart()
        twin.prof, twin.picks, twin.lines = self.prof, list(self.picks), list(self.lines)
        twin.resize(w, h)
        out = []
        img = QImage(w * 2, h * 2, QImage.Format.Format_ARGB32)
        img.setDevicePixelRatio(2.0)
        img.fill(0)
        twin.render(img)
        if img.save(base + ".png"):
            out.append(base + ".png")
        try:
            from PySide6.QtSvg import QSvgGenerator
            gen = QSvgGenerator()
            gen.setFileName(base + ".svg")
            gen.setSize(_QSize(w, h))
            gen.setViewBox(QRectF(0, 0, w, h))
            gen.setTitle(f"Section {self.letters[0]}-{self.letters[1]}")
            twin.render(gen)
            out.append(base + ".svg")
        except ImportError:
            pass
        return out


class Histogram(QWidget):
    """Luminance histogram of the view, with its numbers underneath.

    Mean, standard deviation, and the share of pixels clipped at black and
    at white: the figures that say whether a view's settings are throwing
    information away."""

    def __init__(self, height: int = 96):
        super().__init__()
        self.setMinimumHeight(height + 4)
        self.setAccessibleName("Histogram")
        self.hist = None          # dict channel -> counts (256)
        self.stats = ""
        self.colours = {}

    def recolour(self, t):
        self.colours = dict(bg=t.well, line=t.line, text=t.text3, lum=t.text2, accent=t.accent)
        self.update()

    @staticmethod
    def measure(rgb):
        """(counts per channel, the stats line) for a uint8 image.

        GUI-free, so the render thread works it out with the picture.  OpenCV
        counts (calcHist) and weighs the luminance (Rec. 709, as before):
        about 4 ms at 2500 x 1875, where NumPy's float sums and bincounts
        took most of a frame's time when a drag ended."""
        import cv2
        import numpy as np
        a = np.ascontiguousarray(rgb)
        grey = a.ndim == 2
        if not grey and a.shape[2] >= 3:
            r, g, b = cv2.split(a[..., :3])
            grey = cv2.norm(r, g, cv2.NORM_INF) == 0 and cv2.norm(g, b, cv2.NORM_INF) == 0
        if a.ndim == 2:
            lum = a
        elif grey:
            lum = np.ascontiguousarray(a[..., 0])
        else:
            lum = cv2.transform(a[..., :3], np.array([[0.2126, 0.7152, 0.0722]], np.float32))

        def count(ch):
            return cv2.calcHist([np.ascontiguousarray(ch)], [0], None, [256], [0, 256]).ravel() \
                .astype(np.int64)
        counts = count(lum)
        hs = {"L": counts}
        if not grey:
            for name, ch in zip("RGB", (r, g, b)):
                hs[name] = count(ch)
        n = max(1, int(lum.size))
        lo = 100.0 * float(counts[:2].sum()) / n
        hi = 100.0 * float(counts[254:].sum()) / n
        v = np.arange(256, dtype=np.float64)
        mean = float((counts * v).sum() / n)
        sd = float(np.sqrt(max(0.0, (counts * (v - mean) ** 2).sum() / n)))
        return hs, f"mean {mean:.1f} · sd {sd:.1f} · clipped {lo:.1f}% / {hi:.1f}%"

    def set_data(self, hist, stats: str):
        self.hist, self.stats = hist, stats
        self.setToolTip(f"Luminance, 0-255\n{self.stats}\n(clipped: black / white)")
        self.update()

    def set_image(self, rgb):
        self.set_data(*self.measure(rgb))

    def paintEvent(self, e):
        import numpy as np
        c = self.colours or dict(bg="#0D1224", line="#2C3756", text="#959DB6", lum="#AEB5C8",
                                 accent="#E9B44C")
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(QColor(c["line"]), 1))
        p.setBrush(QColor(c["bg"]))
        p.drawRoundedRect(r, 8, 8)
        f = p.font()
        f.setPointSize(12)
        p.setFont(f)
        plot = QRectF(r.left() + 8, r.top() + 8, r.width() - 16, r.height() - 30)
        if not self.hist:
            p.setPen(QColor(c["text"]))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "No view yet")
            p.end()
            return
        top = max(float(np.max(v[1:-1])) for v in self.hist.values()) or 1.0
        cols = {"R": QColor("#E5534B"), "G": QColor("#3CCB7F"), "B": QColor("#539BF5"),
                "L": QColor(c["lum"])}
        for ch in ("R", "G", "B", "L"):
            if ch not in self.hist:
                continue
            v = np.minimum(self.hist[ch].astype(np.float64) / top, 1.0)
            path = QPainterPath(QPointF(plot.left(), plot.bottom()))
            for i in range(256):
                path.lineTo(plot.left() + i / 255.0 * plot.width(), plot.bottom() - v[i] * plot.height())
            path.lineTo(plot.right(), plot.bottom())
            col = QColor(cols[ch])
            col.setAlpha(150 if ch == "L" else 90)
            p.setPen(QPen(col, 1.2))
            fill = QColor(col)
            fill.setAlpha(40 if ch == "L" else 22)
            p.setBrush(fill)
            p.drawPath(path)
        p.setPen(QColor(c["text"]))
        p.drawText(QRectF(r.left() + 8, r.bottom() - 20, r.width() - 16, 16),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.stats)
        p.end()


class Loupe(QWidget):
    """The full-resolution view under the pointer (the classic magnifier).

    Click the image to hold it in place; double-click to let it follow again."""
    zoom_changed = Signal(float)

    LEVELS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0)

    def __init__(self, size: int = 220):
        super().__init__()
        self.level = 0.5                 # the classic default (mag_level 0.5)
        self.held = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        self.picture = QLabel()
        self.picture.setFixedSize(QSize(size, size))
        self.picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.picture.setProperty("loupe", "true")
        self.picture.setAccessibleName("Loupe")
        self.picture.setText("Move over the image")
        lay.addWidget(self.picture, 0, Qt.AlignmentFlag.AlignHCenter)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.btn_out = QPushButton("−")
        self.btn_out.setProperty("kind", "icon")
        self.btn_out.setAccessibleName("Loupe zoom out")
        self.btn_out.setToolTip("Magnify less")
        self.pct = label("50%", "secondary")
        self.pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_in = QPushButton("+")
        self.btn_in.setProperty("kind", "icon")
        self.btn_in.setAccessibleName("Loupe zoom in")
        self.btn_in.setToolTip("Magnify more")
        self.state = label("Following the pointer", "caption")
        row.addWidget(self.btn_out)
        row.addWidget(self.pct)
        row.addWidget(self.btn_in)
        row.addStretch(1)
        row.addWidget(self.state)
        lay.addLayout(row)
        self.where = label("", "caption")
        lay.addWidget(self.where)
        self.btn_out.clicked.connect(lambda: self.step(-1))
        self.btn_in.clicked.connect(lambda: self.step(1))

    def side(self) -> int:
        return self.picture.width()

    def crop_px(self) -> int:
        """Full-resolution pixels the loupe spans (classic: mag_size / mag_level)."""
        return max(4, int(350 / self.level))

    def step(self, d: int):
        levels = self.LEVELS
        i = min(range(len(levels)), key=lambda k: abs(levels[k] - self.level))
        self.level = levels[max(0, min(len(levels) - 1, i + d))]
        self.pct.setText(f"{self.level * 100:.0f}%")
        self.zoom_changed.emit(self.level)

    def set_held(self, on: bool):
        self.held = on
        self.state.setText("Held · double-click to free" if on else "Following the pointer")

    def show_pixmap(self, px, where: str = ""):
        s = self.side()
        self.picture.setPixmap(px.scaled(s, s, Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.FastTransformation))
        self.where.setText(where)
