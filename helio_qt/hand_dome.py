"""Place the lights by hand: the Virtual Dome Calibrator (CAL-005…CAL-029).

For captures without a chrome sphere.  Each frame's light is a numbered dot
on a dome seen from above, beside the photograph it lit: drag the dot, or
draw an arrow along a shadow in the photograph and the dot turns to face it.
Templates, rings, symmetry lock, mirror, rotation, radial scaling, a
reference photograph under the dome, EXIF, a video walkthrough, merging
another light file and checking the lights against a solve are the classic
calibrator's, from the same arithmetic (helio.application.dome_tools).

Every change goes straight to the project's lights (source "dome"), so the
conditioning and coverage below the dome follow the dots as they move, and
Undo steps back through the changes.  The lights are +Y up throughout; the
dome flips y once to draw, so a dot at the top is a light from the north
(KD-027).
"""
from __future__ import annotations

import math
import os

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QMenu, QMessageBox, QSizePolicy, QSlider,
                               QToolTip, QVBoxLayout, QWidget)

from helio.application import dome_tools as dt
from helio_qt.controls import ParamRow, Section, Segmented
from helio_qt.widgets import FrameList, Viewport, button, label, read_image

# The classic dot colours.
DOT_COLOURS = ["#f9e547", "#f97316", "#22d3ee", "#a78bfa", "#4ade80", "#fb7185",
               "#38bdf8", "#fbbf24", "#e879f9", "#86efac", "#67e8f9", "#fca5a5",
               "#c4b5fd", "#6ee7b7", "#93c5fd", "#fde68a", "#f0abfc", "#a5f3fc",
               "#bbf7d0", "#fef08a"]


def score_colour(s: float) -> QColor:
    """The classic confidence colour: red (poor) to green (good)."""
    return QColor(int((1 - s) * 220), int(s * 200), 0x40)


class _HoverCard(QFrame):
    """A frame's thumbnail and its light, beside the pointer, while it rests on a dot."""

    def __init__(self):
        super().__init__(None, Qt.WindowType.ToolTip)
        self.setProperty("hud", "true")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        self.pic = QLabel()
        self.pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text = QLabel()
        lay.addWidget(self.pic)
        lay.addWidget(self.text)

    def show_at(self, pos: QPoint, pixmap, text: str):
        if pixmap is not None and not pixmap.isNull():
            self.pic.setPixmap(pixmap.scaled(QSize(168, 168), Qt.AspectRatioMode.KeepAspectRatio,
                                             Qt.TransformationMode.SmoothTransformation))
            self.pic.show()
        else:
            self.pic.hide()
        self.text.setText(text)
        self.adjustSize()
        self.move(pos + QPoint(18, 18))
        self.show()


class DomeEditor(QWidget):
    """The dome seen from above, north at the top, the rim the horizon.

    Drag a dot to move its light; with ``sym_lock`` every light keeps the
    dragged one's elevation.  Arrow keys nudge the selected dot 4 px, Tab
    selects the next, Escape none.  With ``place_on_click`` a click anywhere
    moves the selected dot there (the video walkthrough's small dome)."""
    edited = Signal(bool)                 # the lights changed; final (released)
    picked = Signal(int)                  # a dot was selected
    menuAsked = Signal(int, object)       # right-click: dot index (-1 for none), global pos

    DOT = 14

    def __init__(self, minimum: int = 320, place_on_click: bool = False):
        super().__init__()
        self.setMinimumSize(minimum, minimum)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName("Light dome: drag the numbered dots")
        self.vecs: list[tuple[float, float, float]] = []
        self.sel: int | None = None
        self.scores: list[float] | None = None
        self.names: list[str] = []
        self.thumb_for = None             # i -> QPixmap of that light's frame, when known
        self._card = None
        self.sym_lock = False
        self.place_on_click = place_on_click
        self.only_selected = False
        self.ref: QImage | None = None
        self.ref_opacity = 0.35
        self.colours = dict(well="#0E0F12", line="#2C3756", text="#ECE7DA", accent="#E9B44C")
        self._drag = False

    def recolour(self, t):
        self.colours = dict(well=t.image_well, line=t.line, text=t.text, accent=t.accent)
        self.update()

    # geometry
    KEY_H = 22                            # the strip along the bottom for the rings' key

    def _geo(self):
        w, h = self.width(), self.height() - self.KEY_H
        return w / 2.0, h / 2.0, max(20.0, min(w, h) / 2.0 - 30.0)

    def pos_of(self, v) -> QPointF:
        cx, cy, R = self._geo()
        dx, dy = dt.to_dome_point(v)
        return QPointF(cx + dx * R, cy + dy * R)

    def vec_at(self, x: float, y: float):
        cx, cy, R = self._geo()
        return dt.from_dome_point((x - cx) / R, (y - cy) / R)

    def hit(self, x: float, y: float) -> int | None:
        best, bd = None, self.DOT + 8.0
        for i, v in enumerate(self.vecs):
            if self.only_selected and i != self.sel:
                continue
            p = self.pos_of(v)
            d = math.hypot(p.x() - x, p.y() - y)
            if d < bd:
                best, bd = i, d
        return best

    def set_lights(self, vecs, sel=None):
        self.vecs = [tuple(float(c) for c in v) for v in vecs]
        if sel is not None:
            self.sel = sel if 0 <= sel < len(self.vecs) else None
        self.update()

    def select(self, i):
        self.sel = i if (i is not None and 0 <= i < len(self.vecs)) else None
        self.update()

    def _move(self, x, y):
        t = self.vec_at(x, y)
        if self.sym_lock:
            self.vecs = dt.symmetric_move(self.vecs, self.sel, t)
        else:
            self.vecs[self.sel] = t
        self.update()
        self.edited.emit(False)

    # mouse and keys
    def mousePressEvent(self, e):
        self._hide_card()
        if e.button() != Qt.MouseButton.LeftButton:
            return
        x, y = e.position().x(), e.position().y()
        i = self.hit(x, y)
        if i is None and self.place_on_click and self.sel is not None:
            self._drag = True
            self._move(x, y)
            return
        self.sel = i
        self._drag = i is not None
        self.update()
        if i is not None:
            self.picked.emit(i)

    def mouseMoveEvent(self, e):
        x, y = e.position().x(), e.position().y()
        if self._drag and self.sel is not None and e.buttons() & Qt.MouseButton.LeftButton:
            self._move(x, y)
            return
        i = self.hit(x, y)
        if i is None:
            QToolTip.hideText()
            self._hide_card()
            return
        az, el = dt.light_angles(self.vecs[i])
        name = self.names[i] if i < len(self.names) else ""
        conf = (f"  ·  fit {self.scores[i]:.2f}" if self.scores and i < len(self.scores) else "")
        text = f"#{i + 1}  {name}\n{az:.1f}° · {el:.1f}° up{conf}"
        thumb = self.thumb_for(i) if self.thumb_for is not None else None
        if thumb is not None and not thumb.isNull():
            if self._card is None:
                self._card = _HoverCard()
            self._card.show_at(e.globalPosition().toPoint(), thumb, text)
        else:
            QToolTip.showText(e.globalPosition().toPoint(), text, self)

    def _hide_card(self):
        if self._card is not None:
            self._card.hide()

    def leaveEvent(self, e):
        self._hide_card()
        super().leaveEvent(e)

    def hideEvent(self, e):
        self._hide_card()
        super().hideEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag and e.button() == Qt.MouseButton.LeftButton:
            self._drag = False
            self.edited.emit(True)

    def contextMenuEvent(self, e):
        i = self.hit(e.pos().x(), e.pos().y())
        if i is not None:
            self.sel = i
            self.update()
            self.picked.emit(i)
        self.menuAsked.emit(-1 if i is None else i, e.globalPos())

    def focusNextPrevChild(self, nxt):           # Tab selects the next dot instead
        return False

    def keyPressEvent(self, e):
        k = e.key()
        steps = {Qt.Key.Key_Left: (-1, 0), Qt.Key.Key_Right: (1, 0),
                 Qt.Key.Key_Up: (0, -1), Qt.Key.Key_Down: (0, 1)}
        if k in steps and self.sel is not None:
            _cx, _cy, R = self._geo()
            self.vecs[self.sel] = dt.nudge(self.vecs[self.sel], *steps[k], R)
            self.update()
            self.edited.emit(True)
            return
        if k == Qt.Key.Key_Escape:
            self.sel = None
            self.update()
            return
        if k in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab) and self.vecs:
            d = -1 if k == Qt.Key.Key_Backtab else 1
            self.sel = 0 if self.sel is None else (self.sel + d) % len(self.vecs)
            self.update()
            self.picked.emit(self.sel)
            return
        super().keyPressEvent(e)

    # drawing
    def paintEvent(self, _e):
        c = self.colours
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.fillRect(self.rect(), QColor(c["well"]))
        cx, cy, R = self._geo()
        circle = QPainterPath()
        circle.addEllipse(QPointF(cx, cy), R, R)
        if self.ref is not None and not self.ref.isNull():
            p.save()
            p.setClipPath(circle)
            p.setOpacity(max(0.05, min(0.9, self.ref_opacity)))
            box = QRectF(cx - R, cy - R, 2 * R, 2 * R)
            img = self.ref.size()
            k = max(box.width() / max(img.width(), 1), box.height() / max(img.height(), 1))
            tw, th = img.width() * k, img.height() * k
            p.drawImage(QRectF(cx - tw / 2, cy - th / 2, tw, th), self.ref)
            p.restore()
        faint = QColor("#FFFFFF")
        faint.setAlpha(38)
        rim = QColor("#FFFFFF")
        rim.setAlpha(110)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(rim, 1.5))
        p.drawEllipse(QPointF(cx, cy), R, R)
        lab = QColor("#F4F1EA")
        f = QFont(p.font())
        f.setPointSize(11)
        p.setFont(f)
        p.setPen(QPen(faint, 1, Qt.PenStyle.DashLine))
        for el in (30, 60):
            rr = R * math.cos(math.radians(el))
            p.drawEllipse(QPointF(cx, cy), rr, rr)
        # The rings' key in a corner, where no dot can sit on it.
        p.setPen(lab)
        p.drawText(QRectF(10, self.height() - self.KEY_H, self.width() - 20, self.KEY_H - 4),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   "Rings 30° and 60° up · rim the horizon" if self.width() >= 300 else
                   "Rings 30°, 60° up")
        for az in range(0, 360, 45):
            a = math.radians(az)
            card = az % 90 == 0
            t = 10 if card else 5
            p.setPen(QPen(rim if card else faint, 2 if card else 1))
            p.drawLine(QPointF(cx + (R - t) * math.sin(a), cy - (R - t) * math.cos(a)),
                       QPointF(cx + R * math.sin(a), cy - R * math.cos(a)))
        fb = QFont(f)
        fb.setBold(True)
        p.setFont(fb)
        p.setPen(lab)
        for az, txt in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            a = math.radians(az)
            x, y = cx + (R + 15) * math.sin(a), cy - (R + 15) * math.cos(a)
            p.drawText(QRectF(x - 10, y - 9, 20, 18), Qt.AlignmentFlag.AlignCenter, txt)
        p.setPen(QPen(QColor("#FFFFFF"), 1))
        p.setBrush(QColor(c["accent"]))
        p.drawEllipse(QPointF(cx, cy), 3.5, 3.5)
        # spokes, then the dots, the selected one last
        spoke = QColor("#FFFFFF")
        spoke.setAlpha(45)
        p.setPen(QPen(spoke, 1, Qt.PenStyle.DotLine))
        show = [i for i in range(len(self.vecs)) if not self.only_selected or i == self.sel]
        for i in show:
            p.drawLine(QPointF(cx, cy), self.pos_of(self.vecs[i]))
        order = [i for i in show if i != self.sel] + ([self.sel] if self.sel in show else [])
        fn = QFont(f)
        fn.setPointSize(10 if len(self.vecs) < 100 else 9)
        fn.setBold(True)
        p.setFont(fn)
        for i in order:
            pt = self.pos_of(self.vecs[i])
            r = self.DOT * 0.62 + (3 if i == self.sel else 0)
            col = (score_colour(self.scores[i]) if self.scores and i < len(self.scores)
                   else QColor(DOT_COLOURS[i % len(DOT_COLOURS)]))
            if i == self.sel:
                glow = QColor(c["accent"])
                glow.setAlpha(90)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(glow)
                p.drawEllipse(pt, r + 6, r + 6)
            p.setPen(QPen(QColor("#FFFFFF") if i == self.sel else QColor("#10131C"),
                          2 if i == self.sel else 1))
            p.setBrush(col)
            p.drawEllipse(pt, r, r)
            p.setPen(QColor("#10131C") if col.lightnessF() > 0.45 else QColor("#FFFFFF"))
            p.drawText(QRectF(pt.x() - 16, pt.y() - 9, 32, 18), Qt.AlignmentFlag.AlignCenter,
                       str(i + 1))
        if self.hasFocus():
            p.setPen(QPen(QColor(c["accent"]), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(1, 1, self.width() - 2, self.height() - 2))
        p.end()


class VideoWalkthrough(QDialog):
    """Step through a video of a moving light; for each light, find the
    frame where it is lit and click the small dome where the light was."""

    def __init__(self, hand: "HandMode", path: str):
        import cv2
        super().__init__(hand.win)
        self.setWindowTitle(f"Video walkthrough — {os.path.basename(path)}")
        self.hand = hand
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open the video {os.path.basename(path)}.")
        self.total = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        lay = QHBoxLayout(self)
        left = QVBoxLayout()
        self.frame = QLabel()
        self.frame.setMinimumSize(560, 380)
        self.frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame.setObjectName("ImageWell")
        left.addWidget(self.frame, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.total - 1)
        self.slider.valueChanged.connect(self.show_frame)
        left.addWidget(self.slider)
        row = QHBoxLayout()
        for txt, d in (("◀", -1), ("▶", 1), ("▶▶", 10)):
            b = button(txt, "outline", "Step frames: ← →, Shift for 10")
            b.clicked.connect(lambda _c=False, d=d: self.step(d))
            row.addWidget(b)
        self.frame_lbl = label("", "secondary")
        row.addWidget(self.frame_lbl, 1)
        left.addLayout(row)
        lay.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(label("Light to set", "section"))
        self.which = QComboBox()
        n = len(hand.vecs)
        self.which.addItems([f"Light {i + 1}  ·  {hand.names[i] if i < len(hand.names) else ''}"
                             for i in range(n)])
        self.which.currentIndexChanged.connect(self._pick)
        right.addWidget(self.which)
        self.dome = DomeEditor(minimum=240, place_on_click=True)
        self.dome.only_selected = True
        self.dome.colours = hand.editor.colours
        self.dome.set_lights(hand.vecs, 0)
        self.dome.edited.connect(self._edited)
        right.addWidget(self.dome, 1)
        right.addWidget(label("Click the dome where this frame's light was.", "caption", wrap=True))
        nxt = button("Set and go to the next light", "")
        nxt.clicked.connect(self.set_and_next)
        right.addWidget(nxt)
        done = button("Done", "outline")
        done.clicked.connect(self.accept)
        right.addWidget(done)
        lay.addLayout(right)
        self.resize(980, 600)
        self.show_frame(0)

    def _pick(self, i):
        self.dome.set_lights(self.hand.vecs, i)

    def _edited(self, final):
        if final:
            i = self.dome.sel
            vecs = list(self.hand.vecs)
            vecs[i] = self.dome.vecs[i]
            self.hand.commit(vecs, select=i)

    def set_and_next(self):
        self.which.setCurrentIndex((self.which.currentIndex() + 1) % max(1, self.which.count()))

    def step(self, d):
        self.slider.setValue(max(0, min(self.total - 1, self.slider.value() + d)))

    def show_frame(self, idx):
        import cv2
        from helio_qt.widgets import pixmap_from_array
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, fr = self.cap.read()
        if not ok:
            return
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        px = pixmap_from_array(rgb)
        self.frame.setPixmap(px.scaled(self.frame.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation))
        self.frame_lbl.setText(f"Frame {idx} of {self.total}")

    def keyPressEvent(self, e):
        big = 10 if e.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
        if e.key() == Qt.Key.Key_Left:
            self.step(-big)
        elif e.key() == Qt.Key.Key_Right:
            self.step(big)
        else:
            super().keyPressEvent(e)

    def done(self, r):
        try:
            self.cap.release()
        except Exception:
            pass
        super().done(r)


class HandMode:
    """The Capture workspace's "By hand" page and its inspector sections."""

    def __init__(self, ws):
        self.ws, self.win, self.project = ws, ws.win, ws.project
        self.vecs: list[tuple[float, float, float]] = []
        self.names: list[str] = []
        self.undo: list = []
        self.redo: list = []
        self._committed = None          # the project's lights array we last wrote
        self._radius_from = None        # positions the radial slider scales from
        self._rot_last = 0.0
        self.last_mode = "ring60"
        self.basis = None               # the last solve: normals, albedo, crop, tone
        self._arrow = None

    # ── the page ──────────────────────────────────────────────────────
    def build_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        tools = QHBoxLayout()
        tools.setSpacing(8)
        self.btn_prev = button("◀", "icon", "Previous frame")
        self.btn_next = button("▶", "icon", "Next frame")
        self.frame_lbl = label("", "secondary")
        self.btn_prev.clicked.connect(lambda: self.ws._step_frame(-1))
        self.btn_next.clicked.connect(lambda: self.ws._step_frame(1))
        self.tool = Segmented([("move", "Move lights"), ("shadow", "Draw along a shadow")],
                              {"move": "Drag the numbered dots on the dome.",
                               "shadow": "Drag on the photograph from an object to the tip of "
                                         "its shadow: the light is opposite. Set the elevation "
                                         "in the inspector."})
        self.tool.set_current("move")
        self.tool.chosen.connect(self._tool_chosen)
        self.btn_undo = button("Undo", "outline", "Undo the last change to the lights")
        self.btn_redo = button("Redo", "outline", "Put back the change just undone")
        self.btn_undo.clicked.connect(self.do_undo)
        self.btn_redo.clicked.connect(self.do_redo)
        for w in (self.btn_prev, self.frame_lbl, self.btn_next):
            tools.addWidget(w)
        tools.addSpacing(8)
        tools.addWidget(self.tool)
        tools.addStretch(1)
        tools.addWidget(self.btn_undo)
        tools.addWidget(self.btn_redo)
        v.addLayout(tools)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.editor = DomeEditor()
        self.editor.edited.connect(self._editor_edited)
        self.editor.picked.connect(lambda i: self.project.select(i))
        self.editor.menuAsked.connect(self._menu)
        row.addWidget(self.editor, 1)
        self.photo = Viewport("No frame")
        self.photo.lineDrawn.connect(self._shadow_drawn)
        row.addWidget(self.photo, 1)
        v.addLayout(row, 1)
        self.hint = label("Drag a dot to where its light was; arrow keys nudge, Tab picks the "
                          "next. Right-click a dot for more.", "caption", wrap=True)
        v.addWidget(self.hint)
        self.strip = FrameList(self.win.thumbs, QSize(56, 72))
        self.strip.itemClicked.connect(lambda _it: self.project.select(self.strip.currentRow()))
        v.addWidget(self.strip)
        return page

    # ── the inspector ─────────────────────────────────────────────────
    def build_sections(self, store) -> list[Section]:
        from helio.domain.view_controls import Control
        secs = []
        s = Section("Selected light", "hand_sel", store)
        self.sel_name = label("", "secondary", wrap=True)
        s.add(self.sel_name)
        self.az = ParamRow(Control("az", "Azimuth from north", 0.0, 360.0, 0.0, 0.5,
                                   "Clockwise from the top of the photograph.", unit="°"))
        self.el = ParamRow(Control("el", "Elevation", 1.0, 90.0, 35.0, 0.5,
                                   "Above the object's plane: 90° is straight overhead.", unit="°"))
        for r in (self.az, self.el):
            r.changed.connect(self._angles_changed)
            s.add(r)
        g = QGridLayout()
        g.setSpacing(6)
        b1 = button("Copy elevation to all", "outline", "Every light at this light's elevation, "
                    "each keeping its azimuth.")
        b1.clicked.connect(lambda: self.sel() is not None and
                           self.commit(dt.copy_elevation(self.vecs, self.sel())))
        b2 = button("Mirror ↔", "outline", "Swap this light east–west.")
        b2.clicked.connect(lambda: self._mirror("h", self.sel()))
        b3 = button("Mirror ↕", "outline", "Swap this light north–south.")
        b3.clicked.connect(lambda: self._mirror("v", self.sel()))
        g.addWidget(b1, 0, 0, 1, 2)
        g.addWidget(b2, 1, 0)
        g.addWidget(b3, 1, 1)
        s.add_layout(g)
        self.btn_apply_next = button("Next frame →", "", "Go to the next frame (the shadow "
                                     "estimator's Confirm & Advance: changes are already applied).")
        self.btn_apply_next.clicked.connect(lambda: self.ws._step_frame(1))
        s.add(self.btn_apply_next)
        secs.append(s)

        s = Section("Arrange", "hand_arrange", store)
        self.tpl = QComboBox()
        self.tpl.addItem("Templates library…")
        self.tpl.addItems(list(dt.TEMPLATES))
        self.tpl.setAccessibleName("Templates library")
        self.tpl.setToolTip("Start from a known layout: rings, grids and published domes.")
        self.tpl.activated.connect(self._template)
        s.add(self.tpl)
        g = QGridLayout()
        g.setSpacing(6)
        for k, (mode, txt) in enumerate((("ring60", "Ring"), ("ring35", "Ring 35°"),
                                         ("two_rings", "Two rings"), ("hemisphere", "Hemisphere"))):
            b = button(txt, "outline", {"ring60": "Every light in one ring at the elevation below.",
                                        "ring35": "Every light in one ring at 35°.",
                                        "two_rings": "Half at 55°, half at 30°.",
                                        "hemisphere": "8 at 60°, 16 at 38°, 32 at 22°, the rest "
                                                      "overhead."}[mode])
            b.clicked.connect(lambda _c=False, m=mode: self.arrange(m))
            g.addWidget(b, k // 2, k % 2)
        s.add_layout(g)
        self.elev = ParamRow(Control("elev", "Ring elevation", 5.0, 85.0, 35.0, 1.0,
                                     "Re-arranges the ring live as it moves.", unit="°"))
        self.elev.changed.connect(lambda _k, _v, final: self.arrange(self.last_mode, final))
        s.add(self.elev)
        self.radial = ParamRow(Control("radial", "Radial distance", 10.0, 100.0, 100.0, 1.0,
                                       "Pull every light toward the zenith, keeping its azimuth "
                                       "(relative to where they were when you started).",
                                       unit="%"))
        self.radial.changed.connect(self._radial)
        s.add(self.radial)
        secs.append(s)

        s = Section("Symmetry and rotation", "hand_sym", store)
        self.chk_sym = QCheckBox("Symmetry lock")
        self.chk_sym.setToolTip("Dragging a light turns it only; every light shares its elevation.")
        self.chk_sym.toggled.connect(lambda on: setattr(self.editor, "sym_lock", on))
        s.add(self.chk_sym)
        r = QHBoxLayout()
        for axis, txt in (("h", "Mirror all ↔"), ("v", "Mirror all ↕")):
            b = button(txt, "outline", "East–west" if axis == "h" else "North–south")
            b.clicked.connect(lambda _c=False, a=axis: self._mirror(a, None))
            r.addWidget(b)
        s.add_layout(r)
        self.rot = ParamRow(Control("rot", "Rotate all", -180.0, 180.0, 0.0, 1.0,
                                    "Turn the whole dome: for a sequence that starts away from "
                                    "north.", unit="°"))
        self.rot.changed.connect(self._rotate_to)
        s.add(self.rot)
        r = QHBoxLayout()
        for d in (-90, -45, 45, 90):
            b = button(f"{d:+d}°", "outline", f"Turn every light {abs(d)}° "
                       + ("clockwise" if d > 0 else "anticlockwise") + ", seen from above")
            b.clicked.connect(lambda _c=False, d=d: self._rotate_by(d))
            r.addWidget(b)
        s.add_layout(r)
        b = button("Reset the rotation slider", "outline", "Sets the slider back to 0° without "
                   "turning the lights back.")
        b.clicked.connect(self._reset_rotation)
        s.add(b)
        secs.append(s)

        s = Section("Reference under the dome", "hand_ref", store, open_=False)
        b = button("Load a reference photograph…", "outline", "For example the dome rig seen "
                   "from above, to line the dots up with its lamps.")
        b.clicked.connect(self.load_reference)
        s.add(b)
        self.ref_op = ParamRow(Control("op", "Opacity", 0.05, 0.9, 0.35, 0.05, "How strongly the reference photograph shows under "
                                          "the dome: lower to see the dots, higher to line them up with the lamps."))
        self.ref_op.changed.connect(lambda _k, v, _f: (setattr(self.editor, "ref_opacity", v),
                                                        self.editor.update()))
        s.add(self.ref_op)
        b = button("Clear the reference", "outline", "Take the reference photograph away")
        b.clicked.connect(lambda: (setattr(self.editor, "ref", None), self.editor.update()))
        s.add(b)
        secs.append(s)

        s = Section("Check against the solve", "hand_check", store)
        self.btn_orient = button("Check which way round", "outline", "From the photographs "
                                 "alone, no solve needed: are the lights the photographer's "
                                 "mirror image, east and west (or north and south) swapped? "
                                 "Offers to mirror them when the photographs clearly say so.")
        self.btn_orient.clicked.connect(self.check_orientation)
        s.add(self.btn_orient)
        self.check_note = label("", "caption", wrap=True)
        s.add(self.check_note)
        r = QHBoxLayout()
        self.btn_score = button("Score the lights", "outline", "How well each light explains its "
                                "photograph, from the last solve: dots turn green (good) to "
                                "red (poor).")
        self.btn_score.clicked.connect(self.score)
        self.btn_clear_scores = button("Clear", "outline", "Clear the scores: the dots go back to "
                                       "their usual colour")
        self.btn_clear_scores.clicked.connect(self.clear_scores)
        r.addWidget(self.btn_score, 1)
        r.addWidget(self.btn_clear_scores)
        s.add_layout(r)
        self.iters = ParamRow(Control("iters", "Refinement rounds", 5, 100, 20, 1,
                                      "Each round tries small moves of every light and keeps "
                                      "the ones that fit its photograph better."))
        s.add(self.iters)
        self.btn_refine = button("Refine the lights", "", "Nudge each light to fit its "
                                 "photograph's shading under the last solve's surface. Then "
                                 "process again.")
        self.btn_refine.clicked.connect(self.refine)
        s.add(self.btn_refine)
        secs.append(s)

        s = Section("From elsewhere", "hand_more", store, open_=False)
        for txt, tip, fn in (
                ("Estimate from EXIF", "A GPS bearing in the photographs gives each azimuth; a "
                 "fired flash spaces them round the dome. Elevation 35°.", self.exif),
                ("Video walkthrough…", "For a light moved during a video: step through it and "
                 "click where each light was.", self.video),
                ("Import and merge a light file…", "Another .lp or a Helio session: average it "
                 "with these lights, or replace them.", self.import_merge),
                ("Export these lights as .lp…", "The lights as a light-position file, for "
                 "RTIBuilder, Relight and other programs.", self.export_lp),
                ("Import an RTI project folder…", "An RTIBuilder or Relight folder: its .lp and "
                 "photographs, matched by name.", lambda: self.win.registry.trigger("file.import_rti"))):
            b = button(txt, "outline", tip)
            b.clicked.connect(fn)
            s.add(b)
        secs.append(s)
        self.sections = secs
        return secs

    def recolour(self, t):
        self.editor.recolour(t)
        self.photo.recolour(t)
        self.strip.well = t.image_well

    # ── state ─────────────────────────────────────────────────────────
    def sel(self) -> int | None:
        return self.editor.sel

    def start(self):
        """Place by hand: from the current lights where there are any, else
        the classic default hemisphere."""
        p = self.project
        n = len(p.images)
        if p.light_source == "dome" and p.lights is not None and len(p.lights) == n:
            return
        base = dt.hemisphere(n)
        if p.lights is not None and len(p.lights) == n:
            base = [tuple(float(c) for c in v) if np.isfinite(v).all() else base[i]
                    for i, v in enumerate(p.lights)]
        self.undo.clear()
        self.redo.clear()
        self.editor.scores = None
        self.commit(base, push=False)

    def commit(self, vecs, push: bool = True, select=None, keep_radius: bool = False):
        p = self.project
        if push:
            self.undo.append(list(self.vecs))
            del self.undo[:-100]
            self.redo.clear()
        if not keep_radius:
            self._radius_from = None
            self.radial.set_value(100.0)
        self.vecs = [tuple(float(c) for c in v) for v in vecs]
        L = dt.as_lights(self.vecs)
        self._committed = L
        # Opening Place by hand on lights already set: they become the dome's
        # without being re-set, so the solve made with them stays (re-setting
        # them threw it away though nothing had moved).
        if (not push and p.lights is not None and np.shape(p.lights) == np.shape(L)
                and np.allclose(np.nan_to_num(p.lights), np.nan_to_num(L), atol=1e-6)):
            p.light_source = "dome"
            p.changed.emit()
        else:
            p.set_lights(L, "dome", sphere=p.sphere, calibration=p.calibration)
        if select is not None:
            p.select(select)

    def do_undo(self):
        if self.undo:
            self.redo.append(list(self.vecs))
            prev = self.undo.pop()
            self.commit(prev, push=False)

    def do_redo(self):
        if self.redo:
            self.undo.append(list(self.vecs))
            self.commit(self.redo.pop(), push=False)

    def refresh(self):
        p = self.project
        if p.lights is not None and p.lights is not self._committed:
            self.vecs = [tuple(float(c) for c in v) for v in p.lights]     # a session's lights
            self._committed = p.lights
        if p.result is not None:
            r = p.result
            prox = r.proxy or {}
            nrm = prox.get("normals")
            if nrm is not None:
                self.basis = dict(normals=nrm, albedo=prox.get("albedo"), crop=r.crop,
                                  tone=r.settings.photo_tone, images=list(p.images))
        if self.basis is not None and self.basis["images"] != p.images:
            self.basis = None
        self.names = [os.path.basename(x) for x in p.images]
        self.editor.names = self.names
        imgs = list(p.images)
        self.editor.thumb_for = (lambda i, imgs=imgs: self.win.thumbs.cache.get(imgs[i])
                                 if 0 <= i < len(imgs) else None)
        self.editor.set_lights(self.vecs, p.current)
        if self.editor.scores is not None and len(self.editor.scores) != len(self.vecs):
            self.editor.scores = None
        if self.strip.paths != p.images:
            self.strip.set_frames(p.images)
        self.strip.set_captions([str(i + 1) for i in range(len(p.images))])
        if self.strip.currentRow() != p.current:
            self.strip.blockSignals(True)
            self.strip.setCurrentRow(p.current)
            self.strip.blockSignals(False)
        if p.images and self.photo.path != p.images[p.current]:
            self.photo.set_image(p.images[p.current])
            self.photo.clear_line()
            self._arrow = None
        self.frame_lbl.setText(f"Frame {p.current + 1} of {len(p.images)}" if p.images else "")
        i = self.sel()
        if i is not None and i < len(self.vecs):
            az, el = dt.light_angles(self.vecs[i])
            self.az.set_value(round(az, 1))
            self.el.set_value(round(el, 1))
            sc = self.editor.scores
            self.sel_name.setText(f"Light {i + 1} of {len(self.vecs)} · {self.names[i]}"
                                  + (f"\nFit {sc[i]:.2f}" if sc else ""))
        for w in (self.az, self.el):
            w.setEnabled(i is not None)
        self.btn_undo.setEnabled(bool(self.undo))
        self.btn_redo.setEnabled(bool(self.redo))
        self.btn_orient.setEnabled(len(p.images) >= 4 and len(self.vecs) == len(p.images))
        ok = self.basis is not None
        self.btn_score.setEnabled(ok)
        self.btn_refine.setEnabled(ok)
        self.check_note.setText(
            "Uses the last solve's surface. After refining, process again." if ok else
            "Process once with these lights; then score them, or refine them against the "
            "surface it gives.")

    # ── editing ───────────────────────────────────────────────────────
    def _editor_edited(self, final):
        if final:
            self.commit(self.editor.vecs, select=self.editor.sel)
        else:
            i = self.sel()
            if i is not None:
                az, el = dt.light_angles(self.editor.vecs[i])
                self.az.set_value(round(az, 1))
                self.el.set_value(round(el, 1))

    def _angles_changed(self, _k, _v, final):
        i = self.sel()
        if i is None:
            return
        vecs = list(self.vecs)
        vecs[i] = dt.light_from_angles(self.az.value(), self.el.value())
        if final:
            self.commit(vecs)
        else:
            self.editor.set_lights(vecs)

    def _mirror(self, axis, index):
        if index is None and axis and self.vecs:
            self.commit(dt.mirror(self.vecs, axis))
        elif index is not None:
            self.commit(dt.mirror(self.vecs, axis, index))

    def arrange(self, mode, final=True):
        self.last_mode = mode
        vecs = dt.arrange(mode, len(self.project.images), float(self.elev.value()))
        if final:
            self.commit(vecs)
        else:
            self.editor.set_lights(vecs)

    def _template(self, idx):
        if idx <= 0:
            return
        name = self.tpl.itemText(idx)
        self.tpl.setCurrentIndex(0)
        self.commit(dt.template(name, len(self.project.images)))
        self.win.flash(f"“{name}” laid out on {len(self.vecs)} lights: drag to fine-tune.")

    def _radial(self, _k, v, final):
        if self._radius_from is None:
            self._radius_from = list(self.vecs)
        vecs = dt.radius_scale(self._radius_from, float(v) / 100.0)
        if final:
            self.commit(vecs, keep_radius=True)
        else:
            self.editor.set_lights(vecs)

    def _rotate_to(self, _k, v, final):
        delta = float(v) - self._rot_last
        vecs = dt.rotate(self.vecs, delta)
        if final:
            self._rot_last = float(v)
            self.commit(vecs)
            self.rot.set_value(self._rot_last)
        else:
            self.editor.set_lights(vecs)

    def _rotate_by(self, d):
        self.commit(dt.rotate(self.vecs, d))
        self._rot_last = max(-180.0, min(180.0, self._rot_last + d))
        self.rot.set_value(self._rot_last)

    def _reset_rotation(self):
        self._rot_last = 0.0
        self.rot.set_value(0.0)

    def _menu(self, i, gpos):
        m = QMenu(self.editor)
        if i >= 0:
            a = m.addAction(f"Light {i + 1}: {self.names[i] if i < len(self.names) else ''}")
            a.setEnabled(False)
            m.addSeparator()
            m.addAction("Draw along a shadow for this light",
                        lambda: (self.tool.set_current("shadow"), self._tool_chosen("shadow")))
            m.addAction("Copy this elevation to all",
                        lambda: self.commit(dt.copy_elevation(self.vecs, i)))
            m.addAction("Mirror this light ↔", lambda: self._mirror("h", i))
            m.addAction("Mirror this light ↕", lambda: self._mirror("v", i))
            m.addSeparator()
        m.addAction("Mirror all ↔", lambda: self._mirror("h", None))
        m.addAction("Mirror all ↕", lambda: self._mirror("v", None))
        m.addAction("Reset to the hemisphere", lambda: self.arrange("hemisphere"))
        m.exec(gpos)

    # ── the shadow estimator ──────────────────────────────────────────
    def _tool_chosen(self, key):
        self.photo.set_tool("line" if key == "shadow" else "")
        self.hint.setText(
            "Drag on the photograph from the base of an object to the tip of its shadow. The "
            "light is the opposite way; its elevation is the one in the inspector."
            if key == "shadow" else
            "Drag a dot to where its light was; arrow keys nudge, Tab picks the next. "
            "Right-click a dot for more.")

    def _shadow_drawn(self, x0, y0, x1, y1):
        if math.hypot(x1 - x0, y1 - y0) < 3:
            return
        i = self.project.current
        if not 0 <= i < len(self.vecs):
            return
        az = dt.shadow_azimuth(x0, y0, x1, y1)
        el = dt.light_angles(self.vecs[i])[1]
        vecs = list(self.vecs)
        vecs[i] = dt.light_from_angles(az, el)
        self._arrow = (x0, y0, x1, y1)
        self.commit(vecs)
        self.win.flash(f"Light {i + 1}: from {az:.0f}° (opposite the shadow), {el:.0f}° up.")

    # ── the reference, EXIF, video, files ─────────────────────────────
    def load_reference(self):
        path, _ = QFileDialog.getOpenFileName(self.win, "A reference photograph",
                                              str(self.win.prefs.value("last_folder", "")),
                                              "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if path:
            img = read_image(path)
            if img.isNull():
                QMessageBox.warning(self.win, "Reference", "That image could not be read.")
                return
            self.editor.ref = img
            self.editor.update()

    def exif(self):
        new, found = dt.exif_estimate(self.project.images, self.vecs)
        if found:
            self.commit(new)
            self.win.flash(f"Set {found} of {len(new)} lights from the photographs' EXIF.")
        else:
            QMessageBox.information(self.win, "EXIF", "No usable GPS bearing or flash direction "
                                    "in these photographs' EXIF.")

    def video(self):
        path, _ = QFileDialog.getOpenFileName(self.win, "A video of the moving light",
                                              str(self.win.prefs.value("last_folder", "")),
                                              "Video (*.mp4 *.mov *.avi *.mkv *.m4v);;All files (*)")
        if not path:
            return
        try:
            VideoWalkthrough(self, path).exec()
        except ValueError as e:
            QMessageBox.warning(self.win, "Video walkthrough", str(e))

    def import_merge(self):
        path, _ = QFileDialog.getOpenFileName(
            self.win, "Light file or session to merge", str(self.win.prefs.value("last_folder", "")),
            "Light files and sessions (*.lp *.dome *.txt *.helio);;All files (*)")
        if not path:
            return
        try:
            imp = dt.read_vectors(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self.win, "Could not read the file", str(e))
            return
        box = QMessageBox(self.win)
        box.setWindowTitle("Merge the lights")
        box.setText(f"{len(imp)} lights in {os.path.basename(path)}; {len(self.vecs)} here.")
        box.setInformativeText("Average each with the light here, or replace them?")
        avg = box.addButton("Average", QMessageBox.ButtonRole.AcceptRole)
        rep = box.addButton("Replace", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        how = "average" if box.clickedButton() is avg else (
            "replace" if box.clickedButton() is rep else "")
        if how:
            self.commit(dt.merge(self.vecs, imp, how))
            self.win.flash(f"{'Averaged' if how == 'average' else 'Replaced'} "
                           f"{min(len(imp), len(self.vecs))} lights.")

    def export_lp(self):
        from helio.io.sidecars import write_lp
        start = os.path.join(str(self.win.prefs.value("last_folder", "")), "dome_lights.lp")
        path, _ = QFileDialog.getSaveFileName(self.win, "Export the lights", start,
                                              "RTIBuilder light file (*.lp)")
        if not path:
            return
        if not path.lower().endswith(".lp"):
            path += ".lp"
        write_lp(path, dt.as_lights(self.vecs), self.project.images)
        self.win.flash(f"Saved {len(self.vecs)} lights to {os.path.basename(path)}.")

    # ── score and refine ──────────────────────────────────────────────
    def _plates(self, b):
        """The frames at the solve's preview size (cropped as it was)."""
        h, w = b["normals"].shape[:2]
        if not b["crop"]:
            return dt.load_plates(b["images"], (h, w), b["tone"])
        import cv2
        x0, y0, x1, y1 = (int(c) for c in b["crop"])
        out = []
        from helio.core.frame_read import read_frame
        from helio.io.raw_loader import is_raw
        for p in b["images"]:
            g = read_frame(p, cv2.IMREAD_GRAYSCALE, linear=True, sharp=True)
            if g is None:
                out.append(None)
                continue
            g = cv2.resize(g[y0:y1, x0:x1], (w, h)).astype(np.float32)
            if b["tone"] == "srgb" and not is_raw(p):
                from helio.core.radiometry import srgb_to_linear
                g = srgb_to_linear(g, 255.0).astype(np.float32)
            else:
                g /= 255.0
            out.append(g)
        return out

    def score(self):
        b, vecs = self.basis, list(self.vecs)
        if b is None:
            return

        def done(sc, err):
            if err or sc is None:
                self.win.flash(f"Could not score the lights: {err}")
                return
            self.editor.scores = sc
            best = int(np.argmax(sc))
            worst = int(np.argmin(sc))
            self.win.flash(f"Scored {len(sc)} lights: best #{best + 1} ({sc[best]:.2f}), worst "
                           f"#{worst + 1} ({sc[worst]:.2f}). Green fits well, red poorly.")
            self.refresh()
        self.win.run_background(lambda: dt.score(vecs, b["normals"], self._plates(b),
                                                 b["albedo"]), done)

    def check_orientation(self):
        imgs, vecs = list(self.project.images), list(self.vecs)
        if len(imgs) < 4 or len(vecs) != len(imgs):
            return

        def work():
            return dt.orientation_check(vecs, dt.orientation_plates(imgs))

        def done(res, err):
            if err or res is None:
                self.win.flash(f"Could not check the lights: {err}")
                return
            k = res["suggest"]
            if k is None:
                self.win.flash("The photographs agree with the lights as they are: neither "
                               "mirror fits them better.")
                return
            way = "east and west" if k == "h" else "north and south"
            box = QMessageBox(self.win)
            box.setWindowTitle("Lights mirrored?")
            box.setText(f"The photographs fit these lights better with {way} swapped.")
            box.setInformativeText(
                "Each photograph is brighter on the side of its lamp, and the surface solved "
                "from them holds together, only with the lights mirrored. A dome placed as "
                "seen from in front of the object, instead of from the camera, comes out "
                "this way. Mirror them now?")
            yes = box.addButton("Mirror all " + ("↔" if k == "h" else "↕"),
                                QMessageBox.ButtonRole.AcceptRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is yes:
                self.commit(dt.mirror(self.vecs, k))
                self.win.flash("Lights mirrored. Process again to see the surface with them.")
        self.win.flash("Checking the lights against the photographs…", 30)
        self.win.run_background(work, done)

    def clear_scores(self):
        self.editor.scores = None
        self.refresh()

    def refine(self):
        b, vecs, n = self.basis, list(self.vecs), int(self.iters.value())
        if b is None:
            return

        def done(res, err):
            if err or res is None:
                self.win.flash(f"Could not refine the lights: {err}")
                return
            new, moves, before, after = res
            self.commit(new)
            self.win.flash(f"Refined in {n} rounds: {moves} moves; residual "
                           f"{np.nanmean(before):.4f} → {np.nanmean(after):.4f}. Process again "
                           "to see the surface with them.")
        self.win.flash("Refining the lights…", 30)
        self.win.run_background(lambda: dt.refine(vecs, b["normals"], self._plates(b), n,
                                                  b["albedo"]), done)
