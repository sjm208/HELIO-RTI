"""Capture: the photographs and where their lights were (CAL-001…CAL-004, CAL-030…CAL-032).

The chrome sphere is shown, not hidden.  Its outline sits on the
photograph; every frame's highlight is marked on the ball; the lights they
give are plotted on a dome beside the numbers that say whether the set is
good for photometric stereo (coverage, conditioning, elevation range).  The
outline can be dragged, resized, nudged a quarter-pixel at a time, refitted
to the ball's rim, or drawn afresh, as in the classic Sphere Calibrator, and
every change re-reads the highlights at once from frames held in memory.

A frame whose highlight cannot be found is left out of the solve, with the
reason, instead of taking its neighbour's light (KD-020).  The calibration
itself is helio.core.sphere, shared with the classic app.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGraphicsEllipseItem,
                               QGraphicsPixmapItem, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QScrollArea, QStackedWidget, QVBoxLayout, QWidget)

from helio_qt.controls import ParamRow, Section, Segmented
from helio_qt.widgets import (Card, FrameList, LightDome, Viewport, button, fit_list, label,
                              panel, pixmap_from_array, rule)


def _frames_worded(numbers: list[int]) -> str:
    """"frame 7", "frames 3 and 4", "frames 3, 4 and 5"."""
    n = [str(i) for i in numbers]
    if len(n) == 1:
        return f"frame {n[0]}"
    return f"frames {', '.join(n[:-1])} and {n[-1]}"


# ── the ball on the photograph ───────────────────────────────────────────

class SphereView(Viewport):
    """A photograph with the sphere's outline and every frame's highlight.

    Drag the middle of the circle to move it, its ring to resize it; Shift-
    drag anywhere draws a new circle.  Arrow keys nudge, + and - resize, by
    the step set in the inspector.  ``circleChanged(which, cx, cy, r, final)``."""
    circleChanged = Signal(int, float, float, float, bool)
    markCancelled = Signal()               # Esc while marking a ball by hand

    def __init__(self):
        super().__init__("Import images to see them here")
        self.marking = False               # marking a ball by hand: a plain drag draws it
        self.circles: list[tuple[float, float, float] | None] = [None, None]
        self.editing = 0
        self.step = 1.0
        self._drag = None
        self.overlay_item = QGraphicsPixmapItem()
        self.overlay_item.setZValue(2)
        self.scene().addItem(self.overlay_item)
        self.rings, self.dots = [], []
        for i in range(2):
            ring = QGraphicsEllipseItem()
            ring.setZValue(8)
            self.scene().addItem(ring)
            dot = QGraphicsEllipseItem()
            dot.setZValue(9)
            self.scene().addItem(dot)
            self.rings.append(ring)
            self.dots.append(dot)
        self.markers: list[QGraphicsEllipseItem] = []
        self.colours = dict(accent=QColor("#E9B44C"), second=QColor("#F0A35E"),
                            ok=QColor("#6CC4F0"), bad=QColor("#FF8A65"), cur=QColor("#FFFFFF"))
        self._draw_circles()

    def recolour(self, t):
        super().recolour(t)
        self.colours.update(accent=QColor(t.accent), ok=QColor(t.info), bad=QColor(t.problem),
                            cur=QColor(t.text))
        self._draw_circles()

    def set_circle(self, which: int, c):
        self.circles[which] = None if c is None else tuple(float(v) for v in c)
        self._draw_circles()

    def _draw_circles(self):
        for i, (ring, dot) in enumerate(zip(self.rings, self.dots)):
            c = self.circles[i]
            if c is None:
                ring.hide()
                dot.hide()
                continue
            cx, cy, r = c
            col = self.colours["accent"] if i == 0 else self.colours["second"]
            pen = QPen(col, 2.5 if i == self.editing else 1.5)
            pen.setCosmetic(True)
            if i != self.editing:
                pen.setStyle(Qt.PenStyle.DashLine)
            ring.setPen(pen)
            ring.setBrush(Qt.BrushStyle.NoBrush)
            ring.setRect(QRectF(cx - r, cy - r, 2 * r, 2 * r))
            ring.show()
            k = max(r * 0.06, 0.8)
            dot.setPen(QPen(Qt.PenStyle.NoPen))
            dot.setBrush(QBrush(QColor("#FF5A4E")))
            dot.setRect(QRectF(cx - k, cy - k, 2 * k, 2 * k))
            dot.show()

    def set_markers(self, highlights, current: int | None, r: float):
        """One small disc per frame's highlight: blue found, orange not usable,
        white (larger) the frame on show."""
        for m in self.markers:
            self.scene().removeItem(m)
        self.markers = []
        if not highlights:
            return
        k = max(r * 0.045, 0.7)
        for i, h in enumerate(highlights):
            if h is None or not np.isfinite(h.x):
                continue
            m = QGraphicsEllipseItem()
            big = (i == current)
            kk = k * (1.8 if big else 1.0)
            m.setRect(QRectF(h.x - kk, h.y - kk, 2 * kk, 2 * kk))
            col = self.colours["cur"] if big else (self.colours["ok"] if h.ok else self.colours["bad"])
            pen = QPen(QColor("#000000"), 1)
            pen.setCosmetic(True)
            m.setPen(pen)
            m.setBrush(QBrush(col))
            m.setZValue(10)
            m.setToolTip(f"Frame {i + 1}" + ("" if h.ok else f": {h.reason}"))
            self.scene().addItem(m)
            self.markers.append(m)

    def set_ball_overlay(self, rgb, box):
        """A brightened (or max-projected) picture of the ball over the photograph."""
        if rgb is None:
            self.overlay_item.hide()
            return
        x1, y1, _x2, _y2 = box
        self.overlay_item.setPixmap(pixmap_from_array(rgb))
        self.overlay_item.setPos(x1, y1)
        self.overlay_item.show()

    def zoom_to(self, cx, cy, r, margin=1.9):
        rect = QRectF(cx - r * margin, cy - r * margin, 2 * r * margin, 2 * r * margin)
        self.resetTransform()
        vp = self.viewport().rect()
        k = min(vp.width() / max(rect.width(), 1), vp.height() / max(rect.height(), 1))
        self.scale(k, k)
        self.centerOn(rect.center())
        self._fitted = False
        self._zoom_changed()

    # ── editing the circle ─────────────────────────────────────────────
    def _hit(self, pos):
        c = self.circles[self.editing]
        if c is None:
            return None
        cx, cy, r = c
        pt = self.mapToScene(pos.toPoint())
        d = math.hypot(pt.x() - cx, pt.y() - cy)
        scale = self.zoom() or 1.0
        if abs(d - r) * scale < 9:
            return "resize"
        if d < r * 0.7:
            return "move"
        return None

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and not self.item.pixmap().isNull():
            pt = self.mapToScene(e.position().toPoint())
            if e.modifiers() & Qt.KeyboardModifier.ShiftModifier or (
                    self.marking and self._hit(e.position()) is None):
                self._drag = ("draw", pt.x(), pt.y())
                self.circles[self.editing] = (pt.x(), pt.y(), 1.0)
                self._draw_circles()
                return
            hit = self._hit(e.position())
            if hit == "move":
                cx, cy, r = self.circles[self.editing]
                self._drag = ("move", cx - pt.x(), cy - pt.y())
                return
            if hit == "resize":
                self._drag = ("resize", 0, 0)
                return
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            pt = self.mapToScene(e.position().toPoint())
            kind, a, b = self._drag
            cx, cy, r = self.circles[self.editing]
            if kind == "move":
                cx, cy = pt.x() + a, pt.y() + b
            elif kind == "resize":
                r = max(2.0, math.hypot(pt.x() - cx, pt.y() - cy))
            else:
                cx, cy = a, b
                r = max(2.0, math.hypot(pt.x() - a, pt.y() - b))
            self.circles[self.editing] = (cx, cy, r)
            self._draw_circles()
            self.circleChanged.emit(self.editing, cx, cy, r, False)
            return
        if not self.item.pixmap().isNull():
            hit = self._hit(e.position())
            self.viewport().setCursor(Qt.CursorShape.SizeAllCursor if hit == "move" else
                                      Qt.CursorShape.SizeFDiagCursor if hit == "resize" else
                                      Qt.CursorShape.OpenHandCursor)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self._drag is not None:
            self._drag = None
            c = self.circles[self.editing]
            if c is not None:
                self.circleChanged.emit(self.editing, *c, True)
            return
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if self.marking and e.key() == Qt.Key.Key_Escape:
            self.markCancelled.emit()
            return
        c = self.circles[self.editing]
        if c is None:
            return super().keyPressEvent(e)
        cx, cy, r = c
        s = self.step
        k = e.key()
        moves = {Qt.Key.Key_Left: (-s, 0, 0), Qt.Key.Key_Right: (s, 0, 0), Qt.Key.Key_Up: (0, -s, 0),
                 Qt.Key.Key_Down: (0, s, 0), Qt.Key.Key_Plus: (0, 0, s), Qt.Key.Key_Equal: (0, 0, s),
                 Qt.Key.Key_Minus: (0, 0, -s), Qt.Key.Key_Underscore: (0, 0, -s)}
        if k not in moves:
            return super().keyPressEvent(e)
        dx, dy, dr = moves[k]
        self.circles[self.editing] = (cx + dx, cy + dy, max(1.0, r + dr))
        self._draw_circles()
        self.circleChanged.emit(self.editing, *self.circles[self.editing], True)


# ── the calibration off the main thread ─────────────────────────────────

class _CalSignals(QObject):
    done = Signal(int, object, str)


class _CalJob(QRunnable):
    def __init__(self, gen, fn, sig):
        super().__init__()
        self.gen, self.fn, self.sig = gen, fn, sig

    def run(self):
        try:
            self.sig.done.emit(self.gen, self.fn(), "")
        except Exception as e:                       # reported, never swallowed
            import logging
            logging.getLogger(__name__).exception("sphere calibration failed")
            self.sig.done.emit(self.gen, None, f"{type(e).__name__}: {e}")


# ── the workspace ────────────────────────────────────────────────────────

class CaptureWorkspace:
    name = "Capture"
    STEPS = (("Images", "images"), ("Lights", "lights"), ("Check frames", "drift"),
             ("Flat reference", "flat"), ("Backlit image", "backlit"))

    def __init__(self, win):
        self.win, self.project, self.reg = win, win.project, win.registry
        self.sidebar, self.center, self.inspector = self.build()

    def build(self):
        from helio_qt import icons  # noqa: F401  (recolour uses it)
        from helio_qt.state import CLASSIC
        from helio_qt.workspaces import INSPECTOR_W, SIDE_W, _list, _page
        from helio.domain.view_controls import Control
        store = self.win.prefs
        side, sl = panel("Sidebar", SIDE_W)
        sl.addWidget(label("STEPS", "heading"))
        self.steps = _list([], selectable=False)
        sl.addWidget(self.steps)
        sl.addStretch(1)

        self.stack = QStackedWidget()
        # Empty: one clear way in.
        empty, el = _page((48, 48, 48, 48), 14)
        el.addStretch(1)
        self.empty_icon = QLabel()
        self.empty_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        el.addWidget(self.empty_icon)
        t = label("Import a capture", "title")
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        el.addWidget(t)
        sub = label("Choose the folder of photographs, or drop it anywhere in this window.",
                    "secondary", wrap=True)
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        el.addWidget(sub)
        row = QHBoxLayout()
        row.addStretch(1)
        self.btn_import = button("Choose a folder…", "", "A folder of photographs, each lit from "
                                 "one direction (⌘⇧I). Or drop the folder on the window.")
        self.btn_import.clicked.connect(lambda: self.reg.trigger("file.import"))
        row.addWidget(self.btn_import)
        row.addStretch(1)
        el.addLayout(row)
        el.addStretch(2)
        self.stack.addWidget(empty)

        # With images: the light source, then the frames or the ball.
        page = QWidget()
        page.setObjectName("Page")
        page.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        pl = QVBoxLayout(page)
        pl.setContentsMargins(24, 18, 24, 12)
        pl.setSpacing(12)
        head = QHBoxLayout()
        head.addWidget(label("Where were the lights?", "title"))
        info = button("i", "info", "Helio needs the direction of the light in each photograph. "
                                   "A chrome sphere in the photos gives it automatically; a "
                                   "light-position file gives it from a dome controller.")
        info.setAccessibleName("How Helio uses light positions")
        head.addWidget(info)
        head.addStretch(1)
        pl.addLayout(head)
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.card_sphere = Card("Chrome sphere in the photos",
                                "Helio reads each light from the highlight on the sphere.",
                                "Sphere", "sphere")
        self.card_file = Card("Light-position file",
                              "From a dome controller, RTIBuilder or Relight.", ".lp", "file")
        self.card_hand = Card("Place lights by hand",
                              "Drag each light onto a virtual dome; estimate from shadows.",
                              "No sphere", "dome")
        self.card_sphere.clicked.connect(lambda: self.reg.trigger("capture.sphere"))
        self.card_file.clicked.connect(lambda: self.reg.trigger("file.import_lp"))
        self.card_hand.clicked.connect(lambda: self.reg.trigger("capture.dome"))
        self.cards_box = QWidget()
        self.cards_box.setLayout(cards)
        cards.setContentsMargins(0, 0, 0, 0)
        for c in (self.card_sphere, self.card_file, self.card_hand):
            cards.addWidget(c, 1)
        pl.addWidget(self.cards_box)
        # Once the lights have a source, the cards fold into one line.
        self.method_row = QWidget()
        mr = QHBoxLayout(self.method_row)
        mr.setContentsMargins(0, 0, 0, 0)
        mr.setSpacing(8)
        self.method = Segmented([("sphere", "Chrome sphere"), ("file", "Light file"),
                                 ("hand", "By hand")],
                                {"sphere": "A chrome ball photographed with the object: its "
                                           "highlight in each frame gives that light.",
                                 "file": "A light-position file (.lp) from a calibrated dome or "
                                         "another program.",
                                 "hand": "Place the lights yourself on a dome, from a template, "
                                         "a reference photograph or the shadows."})
        self.method.chosen.connect(self._method_chosen)
        mr.addWidget(self.method)
        self.method_status = label("", "secondary")
        mr.addWidget(self.method_status, 1)
        pl.addWidget(self.method_row)
        # Rig profiles: a fixed rig's lights, ball, near-field and scale, kept.
        self.rig_row = QWidget()
        rr = QHBoxLayout(self.rig_row)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.setSpacing(8)
        rr.addWidget(label("Rig profile", "secondary"))
        self.rig_combo = QComboBox()
        self.rig_combo.setAccessibleName("Rig profile")
        self.rig_combo.setToolTip("A rig's lights, where its ball was, near-field and the "
                                  "scale, saved from an earlier capture of the same number of frames.")
        rr.addWidget(self.rig_combo, 1)
        self.btn_rig_use = button("Use", "quiet", "Set this capture's lights, near-field and scale "
                                  "from the rig profile.")
        self.btn_rig_use.clicked.connect(lambda: self.win.apply_rig_profile(
            self.rig_combo.currentData() or ""))
        rr.addWidget(self.btn_rig_use)
        self.btn_rig_save = button("Save as rig…", "quiet", "Keep this capture's lights, ball, "
                                   "near-field and scale for the next capture on this rig.")
        self.btn_rig_save.clicked.connect(lambda: self.win.save_rig_profile())
        rr.addWidget(self.btn_rig_save)
        self.rig_note = label("", "caption", wrap=True)
        pl.addWidget(self.rig_row)
        pl.addWidget(self.rig_note)

        self.work = QStackedWidget()
        # work 0: the frames
        grid_page = QWidget()
        gl = QVBoxLayout(grid_page)
        gl.setContentsMargins(0, 0, 0, 0)
        fh = QHBoxLayout()
        fh.addWidget(label("Frames and the light each one found", "section"))
        fh.addStretch(1)
        self.frames_note = label("", "caption")
        fh.addWidget(self.frames_note)
        gl.addLayout(fh)
        self.grid = FrameList(self.win.thumbs, QSize(96, 128), grid=True, name="FrameGrid")
        self.grid.itemSelectionChanged.connect(lambda: self.project.select(self.grid.currentRow()))
        gl.addWidget(self.grid, 1)
        self.work.addWidget(grid_page)
        # work 1: the ball
        ball = QWidget()
        bl = QVBoxLayout(ball)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        tools = QHBoxLayout()
        tools.setSpacing(8)
        self.btn_prev = button("◀", "icon", "Previous frame")
        self.btn_next = button("▶", "icon", "Next frame")
        self.frame_lbl = label("", "secondary")
        self.btn_prev.clicked.connect(lambda: self._step_frame(-1))
        self.btn_next.clicked.connect(lambda: self._step_frame(1))
        self.show_seg = Segmented([("frame", "Photo"), ("all", "Highlights")],
                                  {"frame": "The photograph, with the ball brightened.",
                                   "all": "The brightest of every frame over the ball: the "
                                          "highlights' ring shows the outline fits."})
        self.show_seg.set_current("frame")
        self.show_seg.chosen.connect(lambda _k: self._draw_ball())
        self.btn_zoom_ball = button("Zoom to ball", "outline",
                                    "Fill the view with the chrome ball, to check the outline "
                                    "fitted to it.")
        self.btn_zoom_ball.clicked.connect(self._zoom_ball)
        self.btn_whole = button("Whole frame", "outline", "Show the whole photograph again.")
        self.btn_whole.clicked.connect(lambda: self.view.fit())
        for w in (self.btn_prev, self.frame_lbl, self.btn_next):
            tools.addWidget(w)
        tools.addSpacing(8)
        tools.addWidget(self.show_seg)
        tools.addStretch(1)
        tools.addWidget(self.btn_zoom_ball)
        tools.addWidget(self.btn_whole)
        bl.addLayout(tools)
        self.view = SphereView()
        self.view.circleChanged.connect(self._circle_moved)
        self.view.markCancelled.connect(lambda: (self.cancel_marking(),
                                                 self.win.flash("Marking cancelled.")))
        bl.addWidget(self.view, 1)
        self.view_hint = label("Drag the ring to resize, the middle to move; Shift-drag to draw a "
                               "new outline. Arrow keys nudge, + and − resize.", "caption", wrap=True)
        bl.addWidget(self.view_hint)
        self.strip = FrameList(self.win.thumbs, QSize(56, 72))
        self.strip.itemClicked.connect(lambda _it: self.project.select(self.strip.currentRow()))
        bl.addWidget(self.strip)
        self.work.addWidget(ball)
        # work 2: the lights placed by hand (the Virtual Dome Calibrator)
        from helio_qt.hand_dome import HandMode
        self.hand = HandMode(self)
        self.work.addWidget(self.hand.build_page())
        pl.addWidget(self.work, 1)
        self.stack.addWidget(page)

        # Inspector: the ball's figures, the lights, the corrections
        insp, il = panel("Inspector", INSPECTOR_W)
        from helio_qt.controls import VScroll
        body = QWidget()
        vl = QVBoxLayout(body)
        vl.setContentsMargins(0, 0, 4, 0)
        vl.setSpacing(12)
        il.addWidget(VScroll(body), 1)

        self.sec_ball = Section("Chrome sphere", "cap_ball", store)
        self.sphere_lines = label("Not searched yet", "secondary", wrap=True)
        self.sec_ball.add(self.sphere_lines)
        self._find_tip = ("Look for the chrome ball in the frames and read each light from its "
                          "highlight (⌘⌥F).")
        self.btn_find = button("Find the sphere", "", self._find_tip)
        self.btn_find.clicked.connect(lambda: self.reg.trigger("capture.sphere"))
        self.sec_ball.add(self.btn_find)
        self.btn_mark = button("Mark the ball by hand…", "quiet",
                               "When the search misses the ball (a very dark one, say): on a "
                               "brightened picture of every frame, drag from its centre to its edge.")
        self.btn_mark.clicked.connect(lambda: self.reg.trigger("capture.mark_ball"))
        self.sec_ball.add(self.btn_mark)
        self.ball_seg = Segmented([("0", "Ball 1"), ("1", "Ball 2")],
                                  {"0": "The first sphere: its outline and highlights.",
                                   "1": "A second sphere: each light becomes the mean of both."})
        self.ball_seg.set_current("0")
        self.ball_seg.chosen.connect(self._choose_ball)
        self.chk_dual = QCheckBox("A second sphere is in the photos")
        self.chk_dual.setToolTip("Two balls (for example at two corners): each frame's light is the "
                                 "mean of both, which averages out small errors.")
        self.chk_dual.toggled.connect(self._dual_toggled)
        self.sec_ball.add(self.chk_dual)
        self.sec_ball.add(self.ball_seg)
        self.fields = {}
        self.fields_box = QWidget()          # hidden until there is a ball to adjust
        grid = QHBoxLayout(self.fields_box)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(6)
        for key, text in (("cx", "x"), ("cy", "y"), ("r", "r")):
            col = QVBoxLayout()
            col.setSpacing(1)
            col.addWidget(label(text, "caption"))
            sp = QDoubleSpinBox()
            sp.setDecimals(2)
            sp.setRange(0.0, 100000.0)
            sp.setSingleStep(0.25)
            sp.setKeyboardTracking(False)
            sp.setAccessibleName({"cx": "Centre x", "cy": "Centre y", "r": "Radius"}[key])
            sp.setToolTip("Full-resolution pixels. Quarter-pixel steps are real: the centre and "
                          "radius stay fractional all the way to the light directions.")
            sp.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            sp.setMinimumWidth(40)
            from PySide6.QtWidgets import QSizePolicy
            sp.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            sp.valueChanged.connect(self._fields_changed)
            col.addWidget(sp)
            grid.addLayout(col)
            self.fields[key] = sp
        self.sec_ball.add(self.fields_box)
        self.step_row = ParamRow(Control("step", "Nudge step", 0.25, 5.0, 1.0, 0.25,
                                         "How far an arrow key moves the outline.", unit="px"))
        self.step_row.changed.connect(lambda _k, v, _f: setattr(self.view, "step", float(v)))
        self.sec_ball.add(self.step_row)
        row2 = QHBoxLayout()
        self.btn_refit = button("Fit to the rim", "", "Snap the outline to the ball's edge, "
                                "measured along 180 rays.")
        self.btn_refit.clicked.connect(self.refit)
        row2.addWidget(self.btn_refit)
        self.sec_ball.add_layout(row2)
        self.exposure = ParamRow(Control("exposure", "Exposure", 0.1, 3.0, 1.0, 0.05,
                                         "Brighten or darken the ball on screen to see the "
                                         "highlight clearly. Display only."))
        self.exposure.changed.connect(lambda _k, _v, _f: self._draw_ball())
        self.sec_ball.add(self.exposure)
        self.chk_robust = QCheckBox("Robust mode")
        self.chk_robust.setToolTip("For stray reflections on the ball: a highlight far from where "
                                   "the others agree is read again more strictly. Slower.")
        self.chk_robust.toggled.connect(lambda _v: self.recalibrate())
        self.sec_ball.add(self.chk_robust)
        self.chk_persp = QCheckBox("Correct for perspective")
        self.chk_persp.setToolTip("A ball near the edge of the frame is seen at an angle. With the "
                                  "lens's focal length the angle is known and taken out. Leave off "
                                  "for results identical to Helio Classic.")
        self.chk_persp.toggled.connect(self._persp_toggled)
        self.sec_ball.add(self.chk_persp)
        self.focal = ParamRow(Control("focal35", "Focal length (35 mm eq.)", 10, 800, 50, 1,
                                      "The lens's focal length as on a full-frame camera.", unit="mm"))
        self.focal.changed.connect(lambda _k, _v, final: final and self.recalibrate())
        self.sec_ball.add(self.focal)
        self.focal_note = label("", "caption", wrap=True)
        self.sec_ball.add(self.focal_note)
        vl.addWidget(self.sec_ball)
        self.hand_secs = self.hand.build_sections(store)
        for s_ in self.hand_secs:
            vl.addWidget(s_)

        self.sec_lights = Section("Lights", "cap_lights", store)
        self.dome = LightDome(size=220)
        self.dome.setEnabled(False)
        self.dome.setToolTip("Each frame's light, seen from above: north at the top, the rim the "
                             "horizon, the inner rings 30° and 60° up. The frame on show is the "
                             "gold dot.")
        dl = QHBoxLayout()
        dl.addStretch(1)
        dl.addWidget(self.dome)
        dl.addStretch(1)
        self.sec_lights.add_layout(dl)
        self.light_stats = label("", "secondary", wrap=True)
        self.light_stats.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.sec_lights.add(self.light_stats)
        self.coverage = label("", "caption", wrap=True)
        self.sec_lights.add(self.coverage)
        # North at the top? (capture_tools.light_frame_check): only says anything when not.
        self.frame_warn = QWidget()
        fw = QVBoxLayout(self.frame_warn)
        fw.setContentsMargins(0, 0, 0, 0)
        fw.setSpacing(6)
        self.frame_warn_text = label("", "reason", wrap=True)
        self.frame_warn_text.setProperty("status", "warn")
        fw.addWidget(self.frame_warn_text)
        b = button("Swap north and south", "outline",
                   "Exchange north and south in every light (their y), then process again.")
        b.clicked.connect(self.win.swap_lights_north_south)
        fw.addWidget(b)
        self.frame_warn.setVisible(False)
        self.sec_lights.add(self.frame_warn)
        vl.addWidget(self.sec_lights)

        self.sec_frames = Section("Frames", "cap_frames", store)
        # Which frames are not in the solve, and whether that was Helio's doing or the
        # user's. A frame ticked off in silence is a light thrown away in silence.
        self.left_out = label("", "reason", wrap=True)
        self.left_out.setProperty("status", "warn")
        self.left_out.setVisible(False)
        self.sec_frames.add(self.left_out)
        self.frame_list = QListWidget()
        self.frame_list.setAccessibleName("Frames and their lights")
        self.frame_list.setToolTip("Untick a frame to leave it out of the solve.")
        self.frame_list.itemChanged.connect(self._frame_ticked)
        self.frame_list.currentRowChanged.connect(lambda i: i >= 0 and self.project.select(i))
        self.frame_list.setMinimumHeight(160)
        self.sec_frames.add(self.frame_list)
        vl.addWidget(self.sec_frames)

        self.sec_corr = Section("Before solving", "cap_corr", store)
        self.chk_align = QCheckBox("Line up shifted frames")
        self.chk_align.setToolTip("Register frames that moved slightly between shots, before the "
                                  "solve: for a camera that was nudged.")
        self.chk_align.toggled.connect(lambda v: self.project.update_settings(auto_align=v))
        self.sec_corr.add(self.chk_align)
        vl.addWidget(self.sec_corr)
        vl.addStretch(1)

        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.gen = 0
        self.sig = _CalSignals()
        self.sig.done.connect(self._calibrated)
        self._shown = None
        self._spheres_from = None
        return side, self.stack, insp

    # ── colour ────────────────────────────────────────────────────────
    def recolour(self, t):
        from helio_qt import icons
        self.empty_icon.setPixmap(icons.icon("images", t.text3, 56).pixmap(56, 56))
        for c in (self.card_sphere, self.card_file, self.card_hand):
            c.recolour(t)
        self.grid.well = t.image_well
        self.strip.well = t.image_well
        self.view.recolour(t)
        self.hand.recolour(t)
        self.dome.recolour(t)
        self.ok_colour, self.accent = t.ok, t.accent

    def _method_chosen(self, key):
        aid = {"sphere": "capture.sphere", "file": "file.import_lp", "hand": "capture.dome"}[key]
        if key == "sphere" and self.project.calibration is not None:
            self.project.set_lights(self.project.calibration.lights, "sphere",
                                    sphere=self.project.calibration.sphere.as_tuple(),
                                    calibration=self.project.calibration)
            return
        why = self.reg.reason(aid)
        if why:
            self.win.flash(why)
            self.refresh()
            return
        self.reg.trigger(aid)

    # ── the ball ──────────────────────────────────────────────────────
    def spheres(self):
        cal = self.project.calibration
        if cal is None:
            return None, None
        return cal.sphere, cal.sphere2

    def _choose_ball(self, key):
        self.view.editing = int(key)
        self.view._draw_circles()
        self._fill_fields()

    def _dual_toggled(self, on):
        from helio.core.sphere import Sphere
        cal = self.project.calibration
        self.ball_seg.setVisible(on)
        if cal is None:
            return
        if on and cal.sphere2 is None:
            # The finder's second candidate, else a circle beside the first.
            c2 = getattr(self.project, "sphere_candidates", [])[1:2]
            s1 = cal.sphere
            s2 = Sphere(*c2[0]) if c2 else Sphere(s1.cx + 4 * s1.r, s1.cy, s1.r)
            self.recalibrate(sphere2=s2)
            self.ball_seg.set_current("1")
            self._choose_ball("1")
        elif not on and cal.sphere2 is not None:
            self.ball_seg.set_current("0")
            self._choose_ball("0")
            self.recalibrate(sphere2=None, drop_second=True)

    def _fill_fields(self):
        s1, s2 = self.spheres()
        s = s2 if (self.view.editing == 1 and s2 is not None) else s1
        for k, sp in self.fields.items():
            sp.blockSignals(True)
            sp.setValue(0.0 if s is None else float(getattr(s, k)))
            sp.blockSignals(False)
            sp.setEnabled(s is not None)

    def _fields_changed(self, *_):
        from helio.core.sphere import Sphere
        s = Sphere(self.fields["cx"].value(), self.fields["cy"].value(), self.fields["r"].value())
        self.view.set_circle(self.view.editing, s.as_tuple())
        self._apply_circle(self.view.editing, s)

    def _circle_moved(self, which, cx, cy, r, final):
        from helio.core.sphere import Sphere
        if getattr(self, "marking", False):
            if final:
                self._marked(cx, cy, r)
            return
        if which == self.view.editing:
            for k, v in (("cx", cx), ("cy", cy), ("r", r)):
                self.fields[k].blockSignals(True)
                self.fields[k].setValue(v)
                self.fields[k].blockSignals(False)
        if final:
            self._apply_circle(which, Sphere(cx, cy, r))

    def _apply_circle(self, which, s):
        if which == 0:
            self.recalibrate(sphere=s)
        else:
            self.recalibrate(sphere2=s)

    # ── marking the ball by hand (when the search misses it) ─────────
    def mark_by_hand(self):
        """Show every frame's brightest, stretched, and let a drag from the
        ball's centre to its edge outline it; then fit the rim if it is
        there to fit, and read each frame's light."""
        from helio.core.sphere import brightest_view
        p = self.project
        if not p.images:
            return
        self.marking = True
        self.view.marking = True
        self.view.editing = 0
        self.view.set_circle(0, None)
        self.view.set_circle(1, None)
        self.view.set_markers(None, None, 1.0)
        self.view.set_ball_overlay(None, None)
        self.win.go_workspace("Capture")
        self.refresh()
        self.win.flash("Brightening every frame to show the ball…", 10)
        paths = list(p.images)

        def done(res, err):
            if not self.marking or paths != list(self.project.images):
                return
            img, size = res if res else (None, None)
            if err or img is None:
                self.cancel_marking()
                self.win.flash("The photographs could not be read to mark the ball.")
                return
            self.view.show_array(img, full_size=size)
            self.view.fit()
            self.view.setFocus()
            self.frame_lbl.setText("Every frame's brightest, stretched to show the ball")
            self.win.flash("Drag from the ball's centre out to its edge. Zoom in for a closer "
                           "outline; Esc to cancel.", 30)
        self.win.run_background(lambda: brightest_view(paths), done)

    def cancel_marking(self):
        if not getattr(self, "marking", False):
            return
        self.marking = False
        self.view.marking = False
        self.view.path = ""                       # the photograph comes back on refresh
        cal = self.project.calibration
        self.view.set_circle(0, cal.sphere.as_tuple() if cal is not None else None)
        self._shown = None
        self.refresh()

    def _marked(self, cx, cy, r):
        """The circle drawn: fitted to the rim if it is close, then the lights."""
        from helio.core.sphere import hand_sphere
        self.marking = False
        self.view.marking = False
        if r < 4:
            self.win.flash("That circle is too small: drag from the centre to the ball's edge.")
            self.mark_by_hand()
            return
        paths = list(self.project.images)
        self.win.flash("Fitting the outline to the ball's rim…", 10)

        def done(s, err):
            if paths != list(self.project.images):
                return
            from helio.core.sphere import Sphere
            s = s if (s is not None and not err) else Sphere(cx, cy, r)
            fitted = math.isfinite(s.rms)
            self.view.path = ""
            self._shown = None
            self.recalibrate(sphere=s)
            self.win.flash(("Outline fitted to the rim" if fitted else
                            "Outline kept as drawn (the rim was not clear enough to fit)")
                           + ": reading each frame's light.", 8)
        self.win.run_background(lambda: hand_sphere(paths, cx, cy, r), done)

    def refit(self):
        """Fit to the rim (helio.core.sphere.refine) from the outline as it stands."""
        from helio.core.sphere import refine
        s1, s2 = self.spheres()
        s = s2 if (self.view.editing == 1 and s2 is not None) else s1
        if s is None:
            return
        fitted = refine(self.project.images, s.cx, s.cy, s.r)
        if not math.isfinite(fitted.rms):
            self.win.flash("The ball's rim could not be seen clearly: adjust the outline by hand.")
            return
        self.view.set_circle(self.view.editing, fitted.as_tuple())
        self._apply_circle(self.view.editing, fitted)
        self.win.flash(f"Fitted to the rim: radius {fitted.r:.2f} px, {fitted.inliers} of "
                       f"{fitted.rays} rim points, {fitted.rms:.2f} px RMS.")

    def _persp_toggled(self, on):
        if on and self.project.exif_focal35:
            self.focal.set_value(round(self.project.exif_focal35))
        self.focal.setVisible(on)
        self.focal_note.setVisible(on)
        self.recalibrate()

    def focal_px(self) -> float | None:
        if not self.chk_persp.isChecked() or not self.project.images:
            return None
        from helio.core.sphere import image_size
        sz = image_size(self.project.images[0]) or (0, 0)
        return float(self.focal.value()) * math.hypot(*sz) / 43.267

    def recalibrate(self, sphere=None, sphere2=None, drop_second=False):
        """Re-read every highlight for the outline(s) as they now are."""
        from helio.core.sphere import calibrate
        cal = self.project.calibration
        if cal is None and sphere is None:
            return
        s1 = sphere or cal.sphere
        s2 = None if drop_second else (sphere2 or (cal.sphere2 if cal else None))
        crops = getattr(cal, "crops", None) if cal is not None else None
        paths = list(self.project.images)
        robust, focal = self.chk_robust.isChecked(), self.focal_px()
        self.gen += 1
        g = self.gen
        self.pool.start(_CalJob(g, lambda: calibrate(paths, s1, robust=robust, sphere2=s2,
                                                     focal_px=focal, crops=crops), self.sig))

    def wait(self):
        from PySide6.QtWidgets import QApplication
        self.pool.waitForDone(120000)
        QApplication.processEvents()

    def _calibrated(self, gen, cal, err):
        if gen != self.gen:
            return
        if err or cal is None:
            self.win.flash(f"Could not read the highlights: {err}")
            return
        self.win.apply_calibration(cal)

    # ── frames ────────────────────────────────────────────────────────
    def _step_frame(self, d):
        p = self.project
        if p.images:
            p.select((p.current + d) % len(p.images))

    def _frame_ticked(self, item):
        i = self.frame_list.row(item)
        on = item.checkState() == Qt.CheckState.Checked
        self.project.set_excluded(i, not on)

    def _zoom_ball(self):
        s1, _s2 = self.spheres()
        if s1 is not None:
            self.view.zoom_to(s1.cx, s1.cy, s1.r)

    def _draw_ball(self):
        """The ball brightened by the exposure, or every frame's brightest over it."""
        cal = self.project.calibration
        crops = getattr(cal, "crops", None) if cal is not None else None
        if crops is None or not crops.boxes:
            self.view.set_ball_overlay(None, None)
            return
        k = 0
        if self.show_seg.current() == "all":
            stack = [c[k] for c in crops.crops if c[k] is not None]
            g = np.maximum.reduce(stack) if stack else None
        else:
            i = min(self.project.current, len(crops.crops) - 1)
            g = crops.crops[i][k]
        if g is None:
            self.view.set_ball_overlay(None, None)
            return
        ex = float(self.exposure.value())
        if ex != 1.0:
            g = np.clip(g.astype(np.float32) * ex, 0, 255).astype(np.uint8)
        self.view.set_ball_overlay(np.repeat(g[..., None], 3, axis=2), crops.boxes[k])

    # ── state → screen ───────────────────────────────────────────────
    def _refresh_rigs(self):
        from helio.application import rigs
        p = self.project
        allp = self.win.rig_profiles()
        fit = rigs.fitting(allp, len(p.images)) if p.images else []
        cur = self.rig_combo.currentData()
        self.rig_combo.blockSignals(True)
        self.rig_combo.clear()
        for name in fit:
            self.rig_combo.addItem(f"{name}  ·  {rigs.describe(allp[name])}", name)
        if cur in fit:
            self.rig_combo.setCurrentIndex(fit.index(cur))
        self.rig_combo.blockSignals(False)
        self.rig_row.setVisible(bool(p.images))
        self.rig_combo.setEnabled(bool(fit))
        self.btn_rig_use.setEnabled(bool(fit))
        self.btn_rig_save.setEnabled(p.lights_ready())
        using = p.light_source.startswith("rig:")
        self.rig_note.setVisible(bool(p.images) and (bool(fit) or bool(allp)) and not using)
        self.rig_note.setText(
            (f"A saved rig fits these {len(p.images)} frames: Use sets the lights, near-field "
             "and scale in one go." if fit else
             f"No saved rig has {len(p.images)} lights.") if p.images else "")

    def refresh(self):
        from helio_qt.workspaces import _disable, light_angles
        p = self.project
        self._refresh_rigs()
        self.stack.setCurrentIndex(1 if p.images else 0)
        self.steps.clear()
        lights_state = ("done" if p.lights_ready() else "now") if p.images else "todo"
        for i, (name, key) in enumerate(self.STEPS):
            if key == "images":
                state, note = ("done", f"{len(p.images)} frames") if p.images else ("now", "None yet")
            elif key == "lights":
                state = lights_state
                src = p.light_source.replace('file:', '')
                note = ((f"From {src}" if src not in ("measured", "dome", "session")
                         else "Placed by hand" if src == "dome" else "From the session")
                        if p.lights_ready() else "Not set")
            else:
                state, note = self._optional_step(key)
            it = QListWidgetItem(f"{'✓' if state == 'done' else i + 1}   {name}\n      {note}")
            if state == "todo" and key in ("images", "lights"):
                _disable(it, "")
            if state == "now":
                it.setSelected(True)
            self.steps.addItem(it)
        fit_list(self.steps)
        if not p.images:
            for w in (self.chk_dual, self.fields_box, self.btn_refit, self.step_row, self.exposure,
                      self.chk_robust, self.chk_persp, self.ball_seg, self.focal, self.focal_note,
                      *self.hand_secs):
                w.setVisible(False)
            self._lights_panel()            # no lights: the last capture's dome and frames go too
            return
        cal = p.calibration
        sphere_mode = p.light_source == "sphere" and cal is not None
        hand_mode = p.light_source == "dome"
        marking = getattr(self, "marking", False)
        self.work.setCurrentIndex(2 if hand_mode else 1 if (sphere_mode or marking) else 0)
        self.sec_ball.setVisible(not hand_mode)
        for s_ in self.hand_secs:
            s_.setVisible(hand_mode)
        if hand_mode:
            self.hand.refresh()
        # cards, or the one-line switch once a source is chosen
        chosen = p.lights is not None or p.busy == "sphere"
        self.cards_box.setVisible(not chosen)
        self.method_row.setVisible(chosen)
        self.method.set_current("sphere" if p.light_source == "sphere" or p.busy == "sphere" else
                                "file" if p.light_source.startswith("file:") else
                                "hand" if hand_mode else "")
        self.card_sphere.set_selected(sphere_mode)
        self.card_file.set_selected(p.light_source.startswith("file:"))
        self.card_hand.set_selected(hand_mode)
        self.card_hand.set_status(f"{len(p.lights)} lights placed" if hand_mode and p.lights is not None
                                  else "Drag dots, or draw along shadows", "ok" if hand_mode else "")
        n_ok = int(np.isfinite(p.lights).all(axis=1).sum()) if p.lights is not None else 0
        if p.busy == "sphere":
            self.card_sphere.set_status("Looking for the sphere…", "info")
        elif sphere_mode:
            bad = len(p.images) - n_ok
            self.card_sphere.set_status(f"{n_ok} of {len(p.images)} lights found"
                                        + (f" · {bad} left out" if bad else ""),
                                        "ok" if not bad else "warn")
        elif p.sphere is False:
            self.card_sphere.set_status("Not found: mark it by hand (Inspector)", "problem")
        else:
            self.card_sphere.set_status("Click to find it", "")
        if p.light_source.startswith("file:"):
            n = len(p.lights)
            ok = n == len(p.images)
            self.card_file.set_status(
                f"{p.light_source[5:]}: {n} lights" + ("" if ok else f" for {len(p.images)} frames"),
                "ok" if ok else "problem")
        else:
            self.card_file.set_status("Choose a file…", "")
        self.method_status.setText(
            self.card_sphere.status.text() if sphere_mode or p.busy == "sphere" else
            self.card_file.status.text() if p.light_source.startswith("file:") else
            f"Placed by hand · {len(p.lights)} lights" if hand_mode and p.lights is not None else
            "From the session" if p.lights is not None else "")
        # frames
        caps = []
        for i in range(len(p.images)):
            if p.lights is not None and i < len(p.lights) and np.isfinite(p.lights[i]).all():
                az, el = light_angles(p.lights[i])
                caps.append(f"{i + 1}{'  ✕' if i in p.excluded else ''}\n{az:.0f}° · {el:.0f}°")
            else:
                caps.append(f"{i + 1}\nno light" if p.lights is not None else str(i + 1))
        short = [f"{i + 1}{' ✕' if (i in p.excluded) else ''}" for i in range(len(p.images))]
        for fl, cc in ((self.grid, caps), (self.strip, short)):
            if fl.paths != p.images:
                fl.set_frames(p.images)
            fl.set_captions(cc)
        if self.strip.currentRow() != p.current:
            self.strip.blockSignals(True)
            self.strip.setCurrentRow(p.current)
            self.strip.blockSignals(False)
        self.frames_note.setText("Azimuth from north, clockwise · elevation"
                                 if p.lights is not None else "Lights not set yet")
        # the ball view
        if sphere_mode and not marking:
            if self.view.path != p.images[p.current]:
                self.view.set_image(p.images[p.current])
                if self._shown is None:
                    self._zoom_ball()
            if self._shown is not cal:
                self._shown = cal
                self.view.set_circle(0, cal.sphere.as_tuple())
                self.view.set_circle(1, cal.sphere2.as_tuple() if cal.sphere2 else None)
            self.view.set_markers(cal.highlights, p.current, cal.sphere.r)
            self._draw_ball()
            self.frame_lbl.setText(f"Frame {p.current + 1} of {len(p.images)}")
        # inspector
        self._fill_fields()
        self.chk_dual.blockSignals(True)
        self.chk_dual.setChecked(bool(cal is not None and cal.sphere2 is not None))
        self.chk_dual.blockSignals(False)
        # The outline's controls only when there is an outline: before a ball
        # is found they showed x, y and r at 0.00 and a second ball's switch.
        for w in (self.chk_dual, self.fields_box, self.btn_refit, self.step_row, self.exposure,
                  self.chk_robust, self.chk_persp):
            w.setVisible(sphere_mode)
        self.ball_seg.setVisible(sphere_mode and self.chk_dual.isChecked())
        self.focal.setVisible(sphere_mode and self.chk_persp.isChecked())
        self.focal_note.setVisible(sphere_mode and self.chk_persp.isChecked())
        if cal is not None:
            s = cal.sphere
            fit = (f"rim fit {s.rms:.2f} px RMS ({s.inliers} of {s.rays} points)"
                   if math.isfinite(s.rms) else "rim not measured" if s.note
                   else "outline set by hand")
            note = (f"\n{s.note[0].upper()}{s.note[1:]}. Check the circle sits on the ball's "
                    "edge: drag its ring if not." if s.note else "")
            self.sphere_lines.setText(f"Centre {s.cx:.1f}, {s.cy:.1f}\nradius {s.r:.2f} px · {fit}"
                                      f"{note}")
        elif marking:
            self.sphere_lines.setText("Marking the ball: drag from its centre out to its edge on "
                                      "the brightened picture. Esc to cancel.")
        elif p.sphere is False:
            self.sphere_lines.setText("No sphere found. Mark the ball by hand, use a light-position "
                                      "file, or place the lights by hand.")
        else:
            self.sphere_lines.setText("Not searched yet")
        why = self.reg.reason("capture.sphere")
        self.btn_find.setEnabled(not why)
        self.btn_find.setToolTip(why or self._find_tip)
        self.btn_find.setText("Find it again" if cal is not None else "Find the sphere")
        exif_f = getattr(p, "exif_focal35", None)
        self.focal_note.setText(
            (f"From the photos: {exif_f:.0f} mm (35 mm equivalent)." if exif_f else
             "The photos do not say the focal length reliably: enter it to use this.")
            if sphere_mode else "")
        self._lights_panel()
        for chk, val in ((self.chk_align, p.settings.auto_align),):
            chk.blockSignals(True)
            chk.setChecked(val)
            chk.blockSignals(False)

    def _lights_panel(self):
        from helio.core.sphere import condition_number, dome_coverage
        from helio_qt.workspaces import light_angles
        p = self.project
        L = p.lights
        self.sec_lights.setVisible(L is not None)
        self.sec_frames.setVisible(L is not None)
        if L is None:
            return
        use = [i for i in range(len(L)) if np.isfinite(L[i]).all() and i not in p.excluded]
        Lu = L[use] if use else np.zeros((0, 3))
        self.dome.set_captured(Lu)
        cur = L[p.current] if p.current < len(L) and np.isfinite(L[p.current]).all() else None
        if cur is not None:
            self.dome.set_light(float(cur[0]), float(-cur[1]), float(cur[2]))
        cond = condition_number(Lu)
        cov = dome_coverage(Lu)
        els = [light_angles(v)[1] for v in Lu] or [0.0]
        verdict = ("well spread" if cond < 3 else "usable" if cond < 6 else
                   "poorly spread: normals will be noisy")
        self.light_stats.setText(
            f"{len(use)} of {len(L)} frames used · elevations {min(els):.0f}°–{max(els):.0f}°\n"
            f"Conditioning {cond:.2f} ({verdict})\n"
            f"Dome coverage {cov['total']}/100 ({cov['grade']}) · "
            f"{cov.get('sectors_filled', 0)} of 12 compass sectors")
        self.light_stats.setToolTip(
            "Conditioning is the light matrix's condition number: how much image noise is "
            "amplified into the normals. Under 3 is good. Dome coverage is the classic score "
            "(azimuth spread, elevation balance, near-zenith lights, count).")
        gaps = cov.get("gaps", [])
        self.coverage.setText(("No light from " + ", ".join(f"{a}–{b}°" for a, b in gaps[:6])
                               + (" …" if len(gaps) > 6 else "")) if gaps else "")
        lf = p.light_frame or {}
        upside = lf.get("verdict") == "upside down"
        self.frame_warn.setVisible(upside)
        if upside:
            self.frame_warn_text.setText(
                f"These lights read upside down: the surface they give is "
                f"{1 / max(lf['ratio'], 1e-9):.1f}× more consistent with north and south "
                f"swapped. Light files from Helio's dome calibrator before October 2026 were "
                f"saved that way.")
        # the list
        cal = p.calibration
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for i in range(len(L)):
            if np.isfinite(L[i]).all():
                az, el = light_angles(L[i])
                text = f"{i + 1:>3}   {az:5.1f}°  {el:4.1f}° up"
                if cal is not None and p.light_source == "sphere" and i < len(cal.highlights):
                    text += f"   q {cal.highlights[i].quality:.0f}"
            else:
                why = (cal.highlights[i].reason if cal is not None and p.light_source == "sphere"
                       and i < len(cal.highlights) else "no light")
                text = f"{i + 1:>3}   — {why}"
            it = QListWidgetItem(text)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            usable = bool(np.isfinite(L[i]).all())
            if not usable:
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if (usable and i not in p.excluded)
                             else Qt.CheckState.Unchecked)
            self.frame_list.addItem(it)
        self.frame_list.setCurrentRow(p.current)
        self.frame_list.blockSignals(False)
        no_light = [i + 1 for i in range(len(L)) if not np.isfinite(L[i]).all()]
        spare = [i + 1 for i in range(len(L)) if np.isfinite(L[i]).all() and i in p.excluded]
        says = []
        if no_light:
            says.append(f"No light on {_frames_worded(no_light)} — each row says why.")
        if spare:
            says.append(f"{_frames_worded(spare).capitalize()} left out though the light was "
                        f"found. Tick to use {'them' if len(spare) > 1 else 'it'}.")
        self.left_out.setText("  ".join(says))
        self.left_out.setVisible(bool(says))

    def _optional_step(self, key):
        """(state, note) for the optional steps: the drift check, the flat
        reference and the backlit frame."""
        w, p = self.win, self.project
        if key == "drift":
            d = p.drift
            if not d:
                return "todo", "Optional · Capture › Check Frames for Drift"
            return "done", ("No movement" if d["verdict"] == "clean" else
                            f"Largest shift {d['peak_px']:.1f} px")
        if key == "flat":
            info = getattr(w, "flat_ref_info", {}) or {}
            if self.project.settings.light_field:      # the white board: the exact correction
                return "done", "White-board light field"
            return (("done", f"{info.get('n', 0)} flat frames (older correction)")
                    if getattr(w, "flat_ref_normals", None) is not None
                    else ("todo", "Optional · Process › Precision › Calibrate Lights on a White Board"))
        if key == "backlit":
            b = getattr(w, "backlit", {}) or {}
            return (("done", os.path.basename(b.get("path", ""))) if b.get("path")
                    else ("todo", "Optional · Capture › Add Backlit Image"))
        return "todo", "Optional"

    def next_step(self):
        p = self.project
        if not p.images:
            return "Import images…", "file.import"
        if not p.lights_ready():
            return "Find the chrome sphere", "capture.sphere"
        return "Continue to Process →", "view.ws:Process"

    def status(self):
        from helio_qt.workspaces import light_angles
        p = self.project
        if p.notice and not p.busy:
            return "info", "Session", p.notice
        if not p.images:
            return "idle", "No images yet", "Drop a folder of captures anywhere in this window"
        if p.busy == "sphere":
            return "info", "Finding the sphere…", "Reading the highlight in each frame"
        if p.lights_ready():
            Lu = p.solve_lights()
            els = [light_angles(v)[1] for v in Lu]
            left = len(p.images) - len(Lu)
            return ("ok", "Lights ready",
                    f"{len(Lu)} directions · elevations {min(els):.0f}°–{max(els):.0f}°"
                    + (f" · {left} frame{'s' if left > 1 else ''} left out" if left else ""))
        if p.lights is not None:
            return ("problem", "Lights do not match the frames",
                    f"{len(p.lights)} lights for {len(p.images)} frames")
        return "warn", "Lights not set", f"{len(p.images)} frames · {self.win.frame_size_text()}"
