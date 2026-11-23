"""The Research menu: the classic Research, Metrology and AI Tools windows.

Each tool is a small window: settings on the left, a preview on the right,
Run, then Save once there is something to save.  The computing is
helio.application.research and helio.application.ai, the same calls the
classic app makes; the method notes fold away under "How it works".
"""
from __future__ import annotations

import math
import os

import cv2
import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPlainTextEdit, QSizePolicy, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

from helio.domain.view_controls import Control
from helio_qt.controls import ParamRow, Section, Segmented
from helio_qt.widgets import button, label, pixmap_from_array


def current_view(win):
    """The view on screen at full resolution, as Current View exports it."""
    from helio.application.viewing import screen_image
    ins = win.workspaces["Inspect"]
    if win.render_data is None:
        raise RuntimeError("Process the images first.")
    d = win.display
    if win.view == ins.DEPTH_MAP:
        import copy
        d = copy.deepcopy(d)
        d.layer = "Depth Map (Greyscale)"
    return screen_image(win.engine, win.render_data, dict(win.engine_settings.get(win.engine, {})), d,
                        use_proxy=False)


def view_picture(win, engine: str, layer: str | None = None):
    """*engine* at full resolution with its own settings, on *layer* (a
    Display layer: "Combined Colour", "Shading" ...) or the one on screen."""
    import copy
    from helio.application.viewing import screen_image
    if win.render_data is None:
        raise RuntimeError("Process the images first.")
    d = copy.deepcopy(win.display)
    if layer:
        d.layer = layer
    return screen_image(engine, win.render_data, dict(win.engine_settings.get(engine, {})), d,
                        use_proxy=False)


def view_signature(win) -> tuple:
    """What the view on screen is drawn from: changes when the view, its
    layer or its settings do."""
    d = win.display
    return (win.engine, win.view, d.layer, d.colour_opacity, d.colour_gain,
            repr(sorted(win.engine_settings.get(win.engine, {}).items())))


def object_name(p) -> str:
    return (dict(p.capture_meta).get("title") or p.name or "object").strip()


def _start(win, name: str) -> str:
    from helio_qt.publish import file_stem
    return os.path.join(str(win.prefs.value("export_folder", "")) or win.default_export_dir(),
                        f"{file_stem(win.project.name)}{name}")



def _true_heights(D):
    from helio.application.heights import height_px
    return height_px(D)

class Preview(QLabel):
    """An image scaled to fit, its proportions kept."""

    def __init__(self, empty: str = "Run the tool to see the result here."):
        super().__init__(empty)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(QSize(420, 380))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setProperty("role", "secondary")
        self.img = None

    def set_image(self, rgb):
        self.img = None if rgb is None else np.ascontiguousarray(rgb)
        self._fit()

    def _fit(self):
        if self.img is None:
            return
        pm = pixmap_from_array(self.img)
        self.setPixmap(pm.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def image_pos(self, pos):
        """(x, y, scale) of widget point *pos* on the image shown, or None outside it."""
        if self.img is None:
            return None
        ih, iw = self.img.shape[:2]
        sc = min(self.width() / iw, self.height() / ih)
        ox, oy = (self.width() - iw * sc) / 2, (self.height() - ih * sc) / 2
        x, y = (pos.x() - ox) / sc, (pos.y() - oy) / sc
        return (x, y, sc) if 0 <= x < iw and 0 <= y < ih else None


class DragPreview(Preview):
    """A Preview the outline on it can be dragged on: it reports where on the
    image the mouse goes down, moves and comes up."""
    pressed = Signal(float, float, float, bool)       # x, y, scale, Option / Alt held
    moved = Signal(float, float)
    released = Signal()
    double = Signal(float, float, float)

    def __init__(self, empty: str = "Run the tool to see the result here."):
        super().__init__(empty)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._down = False

    def mousePressEvent(self, e):
        p = self.image_pos(e.position())
        if p is None or e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        self._down = True
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.pressed.emit(p[0], p[1], p[2], bool(e.modifiers() & Qt.KeyboardModifier.AltModifier))

    def mouseMoveEvent(self, e):
        if not self._down:
            return super().mouseMoveEvent(e)
        p = self.image_pos(e.position())
        if p is not None:
            self.moved.emit(p[0], p[1])

    def mouseReleaseEvent(self, e):
        if self._down:
            self._down = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            self.released.emit()
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        p = self.image_pos(e.position())
        if p is not None:
            self.double.emit(p[0], p[1], p[2])


class ToolDialog(QDialog):
    """Settings, Run, a preview, then Save."""
    PREVIEW = Preview
    TITLE = ""
    INTRO = ""
    METHOD = ""
    SAVE = "Save…"

    def __init__(self, win):
        super().__init__(win)
        self.win, self.project = win, win.project
        self.result = None
        self.setWindowTitle(self.TITLE)
        self.setMinimumSize(900, 600)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(16)
        left = QWidget()
        left.setFixedWidth(310)
        self.form = QVBoxLayout(left)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(8)
        self.form.addWidget(label(self.TITLE, "title"))
        if self.INTRO:
            self.form.addWidget(label(self.INTRO, "secondary", wrap=True))
        self.build()
        if self.METHOD:
            sec = Section("How it works", f"research_{type(self).__name__}", win.prefs, open_=False)
            sec.add(label(self.METHOD, "caption", wrap=True))
            self.form.addWidget(sec)
        self.form.addStretch(1)
        # every tool's result can go on the publication sheet, the same way
        self.form.addWidget(label("PUBLICATION SHEET", "heading"))
        srow = QHBoxLayout()
        self.sheet_caption = QLineEdit()
        self.sheet_caption.setPlaceholderText(self.TITLE)
        self.sheet_caption.setAccessibleName("Caption on the sheet")
        self.sheet_caption.setToolTip("The caption under it on the sheet; blank uses the tool's name.")
        self.btn_sheet = button("Add to sheet", "quiet", "Put what this tool shows on the publication "
                                                         "sheet, under From the tools. Publish lists "
                                                         "everything on the sheet.")
        self.btn_sheet.clicked.connect(self.add_to_sheet)
        srow.addWidget(self.sheet_caption, 1)
        srow.addWidget(self.btn_sheet)
        self.form.addLayout(srow)
        self.status = label("", "caption", wrap=True)
        self.form.addWidget(self.status)
        row = QHBoxLayout()
        close = button("Close", "quiet")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        row.addStretch(1)
        self.btn_run = button(self.RUN if hasattr(self, "RUN") else "Run", "primary")
        self.btn_run.clicked.connect(self.run)
        self.btn_save = button(self.SAVE, "outline")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_save)
        self.form.addLayout(row)
        outer.addWidget(left)
        self.preview = self.PREVIEW()
        outer.addWidget(self.preview, 1)

    # what each tool fills in
    def build(self):
        pass

    def args(self) -> dict:
        return {}

    def work(self, **kw):
        raise NotImplementedError

    def show_result(self, res):
        pass

    def save(self):
        pass

    def sheet_picture(self):
        """What goes on the sheet: the result as the preview shows it, or None."""
        img = getattr(self.preview, "img", None)
        return None if self.result is None or img is None else img

    def sheet_caption_text(self) -> str:
        return self.sheet_caption.text().strip() or self.TITLE

    def add_to_sheet(self):
        pic = self.sheet_picture()
        if pic is None:
            self.status.setText(f"{self.RUN if hasattr(self, 'RUN') else 'Run'} first, then add the "
                                "result to the sheet.")
            return
        if self.win.add_sheet_tool(self.sheet_caption_text(), np.ascontiguousarray(pic)):
            self.status.setText("On the publication sheet, under From the tools.")

    # shared plumbing
    def param(self, key, text, lo, hi, default, step, tip="", unit="") -> ParamRow:
        r = ParamRow(Control(key, text, lo, hi, default, step, tip, (), unit))
        self.form.addWidget(r)
        return r

    def combo(self, text: str, items, current: str = "", tip: str = "") -> QComboBox:
        self.form.addWidget(label(text, ""))
        c = QComboBox()
        c.addItems(list(items))
        c.setAccessibleName(text)
        if tip:
            c.setToolTip(tip)
        if current:
            c.setCurrentText(current)
        self.form.addWidget(c)
        return c

    def run(self):
        try:
            kw = self.args()
        except (RuntimeError, ValueError) as e:
            self.status.setText(str(e))
            return
        if kw is None:
            return
        self.btn_run.setEnabled(False)
        self.status.setText("Working…")
        self.win.run_background(lambda: self.work(**kw), self._done, tool=True)

    def _done(self, res, err):
        self.btn_run.setEnabled(True)
        if err:
            self.status.setText(err.split(": ", 1)[-1])
            return
        self.result = res
        self.btn_save.setEnabled(res is not None)
        self.show_result(res)

    def data(self):
        """The solve's render data: the selection's alone when Inspect has one."""
        if self.win.render_data is None:
            raise RuntimeError("Process the images first.")
        return self.win.selected_result()[1]


# ── reflectance unmixing ─────────────────────────────────────────────────

class UnmixDialog(ToolDialog):
    TITLE = "Reflectance unmixing"
    INTRO = "Splits how each pixel answers the lights into separate channels."
    METHOD = ("PCA (or ICA) on the light stack: one row per pixel, one column per photograph, "
              "decoded to linear light as the solve decodes it. Materials, layers and coatings "
              "that answer the lights differently separate into different components. Helio Classic "
              "called this HSH unmixing; no HSH or PTM basis is involved. The maps use "
              "perceptually uniform colours. Save them as PNG or JPEG maps, with 16-bit greyscale "
              "TIFFs for an image editor or 32-bit value TIFFs for analysis; with three or more "
              "components a colour composite (C1-C3 as red, green, blue) is saved too.")
    SAVE = "Save components…"

    def build(self):
        self.form.addWidget(label("Components", ""))
        self.n = Segmented([(str(k), str(k)) for k in (2, 3, 4, 5)])
        self.n.set_current("3")
        self.form.addWidget(self.n)
        self.form.addWidget(label("Method", ""))
        self.method = Segmented([("PCA", "PCA"), ("ICA", "ICA")],
                                {"PCA": "Uncorrelated components, by share of variance.",
                                 "ICA": "Statistically independent components (FastICA)."})
        self.method.set_current("PCA")
        self.form.addWidget(self.method)
        from helio.application.research import UNMIX_SAVE
        self.save_as = self.combo("Save as", [s[0] for s in UNMIX_SAVE],
                                  tip="The colour maps are for looking at; the TIFFs keep each "
                                      "component's values.")

    def args(self):
        p = self.project
        r = self.win.selected_result()[0]
        if r is None:
            raise RuntimeError("Process the images first.")
        ph, pw = r.proxy["normals"].shape[:2]
        return dict(paths=list(p.images), w=pw, h=ph, crop=r.crop, base=(r.base_w, r.base_h),
                    tone=p.settings.resolved_photo_tone(),
                    n=int(self.n.current() or 3), method=self.method.current() or "PCA")

    def work(self, paths, w, h, crop, base, tone, n, method):
        from helio.application.export_images import load_stack
        from helio.application.research import unmix
        stack = load_stack(paths, w, h, crop=crop, base=base, tone=tone)
        if stack is None:
            raise ValueError("A photograph could not be read.")
        return unmix(stack, n, method)

    def show_result(self, res):
        from helio.application.research import UNMIX_MAPS
        comps, var = res
        tiles = []
        for i, ch in enumerate(comps):
            u8 = cv2.normalize(ch, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            t = cv2.cvtColor(cv2.applyColorMap(u8, UNMIX_MAPS[i % len(UNMIX_MAPS)]), cv2.COLOR_BGR2RGB)
            tiles.append(cv2.copyMakeBorder(t, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(20, 20, 20)))
        cols = 2 if len(tiles) > 2 else len(tiles)
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 20))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        self.preview.set_image(np.vstack(rows))
        if self.method.current() == "ICA":
            # ICA scales every component to the same strength: no shares to give
            self.status.setText(f"{len(comps)} independent components (ICA gives no share of "
                                f"variance; switch to PCA for that).")
        else:
            self.status.setText("Variance: " + "  ".join(f"C{i + 1} {v * 100:.1f}%"
                                                          for i, v in enumerate(var)))

    def save(self):
        from helio.application.research import UNMIX_SAVE, save_components
        _label, ext, values = UNMIX_SAVE[self.save_as.currentIndex()]
        path, _ = QFileDialog.getSaveFileName(self, "Save the components",
                                              _start(self.win, "_unmix" + ext),
                                              "Base name (the component number is added)")
        if not path:
            return
        base = os.path.splitext(path)[0]
        files = save_components(self.result[0], base, ext, values)
        self.status.setText(f"Saved {len(files)} files in {os.path.basename(os.path.dirname(base))}.")
        self.win.flash(f"Unmixing saved: {os.path.basename(files[0])} …")


# ── micro-features ───────────────────────────────────────────────────────

class MicroDialog(ToolDialog):
    TITLE = "Micro-features"
    INTRO = "Finds pinholes, stylus marks and small pits in the surface."
    METHOD = ("Laplacian-of-Gaussian blobs in the height map. Each feature's depth is how far its "
              "centre lies below the ring around it (from r to 2r); Helio Classic reported the surface's "
              "height there, so a pit on a high part of the object counted as shallow. Positions "
              "are in mm once the scale is set, depths in µm once the height scale is set.")
    SAVE = "Export…"

    def build(self):
        # 2-12 px suits a capture of a few megapixels; on 100 MP the same
        # pits are about three times wider, so the defaults follow the size.
        from helio.application.research import groove_smoothing
        D = self.win.render_data
        k = groove_smoothing(D.w, D.h) if D is not None else 1.0
        self.min_r = self.param("min_r", "Smallest radius", 1, 80, round(2 * k), 1,
                                "Small pinholes.", "px")
        self.max_r = self.param("max_r", "Largest radius", 2, 160, round(12 * k), 1,
                                "Larger stylus marks.", "px")
        self.thr = self.param("thr", "Detection threshold", 0.01, 0.3, 0.02, 0.01,
                              "Lower finds more, including noise.")
        self.min_d = self.param("min_d", "Least depth", 0.0, 0.5, 0.01, 0.01,
                                "Below the surrounding surface, as a share of the depth range.")
        self.pits = QCheckBox("Depressions only (ignore raised bumps)")
        self.pits.setChecked(True)
        self.form.addWidget(self.pits)
        self.src = self.combo("Height map", ("Display depth (levelled)", "Integrated depth"))

    def args(self):
        D = self.data()
        if self.min_r.value() >= self.max_r.value():
            raise ValueError("The smallest radius must be below the largest.")
        if self.src.currentIndex() == 0 or getattr(D, "depth_poisson", None) is None:
            depth = np.asarray(D.depth_vis_p, np.float32)
        else:
            depth = np.asarray(D.depth_poisson, np.float32)
            rng = float(np.ptp(depth))
            depth = (depth - depth.min()) / rng if rng > 1e-9 else depth
        p = self.project
        return dict(depth=depth, min_r=self.min_r.value(), max_r=self.max_r.value(),
                    threshold=self.thr.value(), min_depth=self.min_d.value(),
                    depressions_only=self.pits.isChecked(), mm_per_px=p.scale_mm_per_px,
                    z_mm_per_unit=p.z_mm_per_unit, heights=_true_heights(D))

    def work(self, depth, **kw):
        from helio.application.research import micro_features
        return depth, micro_features(depth, **kw)

    def show_result(self, res):
        depth, rows = res
        img = cv2.cvtColor((np.clip(depth, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for r in rows:
            cv2.circle(img, (r["x_px"], r["y_px"]), max(2, int(round(r["radius_px"]))), (255, 68, 102), 1,
                       cv2.LINE_AA)
        self.preview.set_image(img)
        if not rows:
            self.status.setText("None found. Try a lower threshold or least depth.")
            self.btn_save.setEnabled(False)
            return
        unit = "µm" if rows[0]["depth_um"] is not None else "of the depth range"
        deepest = max(rows, key=lambda r: r["depth_norm"])
        dv = deepest["depth_um"] if deepest["depth_um"] is not None else deepest["depth_norm"]
        self.status.setText(f"Found {len(rows)}. Deepest {dv:g} {unit}.")

    def save(self):
        from helio.application.research import export_micro_features
        from helio_qt.publish import file_stem
        folder = QFileDialog.getExistingDirectory(self, "Folder for the micro-features",
                                                  os.path.dirname(_start(self.win, "")))
        if not folder:
            return
        depth, rows = self.result
        name = object_name(self.project)
        files = export_micro_features(
            rows, depth, folder, file_stem(name),
            f"{name}  —  {len(rows)} micro-features  (r={self.min_r.value():.0f}–"
            f"{self.max_r.value():.0f}px, thresh={self.thr.value():.3f})")
        self.status.setText("Saved " + ", ".join(os.path.basename(f) for f in files) + ".")
        self.win.flash(f"Micro-features saved: {len(rows)} in {os.path.basename(folder)}.")


# ── toolpaths ────────────────────────────────────────────────────────────

class ToolpathDialog(ToolDialog):
    TITLE = "Toolpaths"
    INTRO = "Traces scratches, incisions and grooves as strokes."
    METHOD = ("Where the surface curves up into a groove (positive Laplacian of the height, or "
              "the divergence of the slopes given by the normals) more than the threshold × the "
              "strongest groove, thinned to one pixel. Each stroke's length is measured along that "
              "line, its groove depth below the surface around it, its direction from the main "
              "axis of its points. Helio Classic kept the groove shoulders and the flat background, "
              "and counted lengths about three times too long.")
    SAVE = "Export…"

    def build(self):
        self.min_len = self.param("min_len", "Shortest stroke", 5, 200, 20, 1, "", "px")
        self.thr = self.param("thr", "Groove threshold", 0.02, 0.5, 0.15, 0.01,
                              "Share of the strongest groove curvature.")
        self.dil = self.param("dil", "Join gaps up to", 0, 3, 1, 1, "Bridges breaks in a stroke.", "px")
        from helio.application.research import groove_smoothing
        D = self.win.render_data
        auto = groove_smoothing(D.w, D.h) if D is not None else 1.0
        self.smooth = self.param("smooth", "Smoothing", 1, 12, auto, 0.5,
                                 "About a third of a groove's width. Higher: cleaner, single "
                                 "lines on wide grooves; lower: fine scratches.", "px")
        self.src = self.combo("Curvature from", ("Normals (no integration)", "Depth map"))
        # How the strokes are coloured: redrawn at once, the strokes kept.
        self.chk_colour = QCheckBox("Colour by direction and depth")
        self.chk_colour.setToolTip("The colour shows which way each line runs (fillets, rolls "
                                   "and the sides of a border part by colour); deeper grooves "
                                   "are brighter, ranked within this result.")
        self.chk_colour.toggled.connect(self._recolour)
        self.form.addWidget(self.chk_colour)
        self.key = QLabel()
        self.key.setAccessibleName("Colour key")
        self.key.setVisible(False)
        self.form.addWidget(self.key)
        self.key_text = label("Direction: red ─  green ╲  cyan │  violet ╱. Brighter is deeper.",
                              "caption", wrap=True)
        self.key_text.setVisible(False)
        self.form.addWidget(self.key_text)

    def colours(self, strokes) -> dict:
        from helio.application.research import stroke_colours
        on = self.chk_colour.isChecked()
        return stroke_colours(strokes, by_direction=on, by_depth=on)

    def _recolour(self):
        from helio.application.research import colour_key
        on = self.chk_colour.isChecked()
        self.key.setVisible(on)
        self.key_text.setVisible(on)
        if on:
            self.key.setPixmap(pixmap_from_array(colour_key(True, True)))
        if self.result is not None:
            self.show_result(self.result)

    def args(self):
        D = self.data()
        p = self.project
        hts = _true_heights(D)                         # groove depths in px of height, µm with the scale
        return dict(depth=np.asarray(D.depth_vis_p if hts is None else hts, np.float32), normals=D.normals_full,
                    use_normals=self.src.currentIndex() == 0, min_len=int(self.min_len.value()),
                    smooth=float(self.smooth.value()),
                    threshold=self.thr.value(), dilation=int(self.dil.value()),
                    mm_per_px=p.scale_mm_per_px, z_mm_per_unit=p.z_mm_per_unit,
                    albedo=D.albedo_rgb_full)

    def work(self, albedo, **kw):
        from helio.application.research import toolpaths
        strokes, labels, skel = toolpaths(**kw)
        return strokes, labels, skel, albedo

    def show_result(self, res):
        strokes, labels, skel, albedo = res
        img = (np.asarray(albedo).astype(np.uint8).copy() if albedo is not None
               else np.zeros((*labels.shape, 3), np.uint8))
        img = (img * 0.55).astype(np.uint8)
        # One pass: each kept stroke's label looks up its colour (a pass per
        # stroke took minutes on a large capture).
        lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
        keep = np.zeros(len(lut), bool)
        for sid, rgb in self.colours(strokes).items():
            lut[sid] = rgb
            keep[sid] = True
        on = (cv2.dilate(skel, np.ones((2, 2), np.uint8)) > 0) & keep[labels]
        img[on] = lut[labels[on]]
        self.preview.set_image(img)
        if not strokes:
            self.status.setText("No strokes. Try a lower threshold or shorter strokes.")
            self.btn_save.setEnabled(False)
            return
        mm = strokes[0]["length_mm"] is not None
        total = sum(s["length_mm"] if mm else s["length_px"] for s in strokes)
        self.status.setText(f"{len(strokes)} strokes, {total:.1f} {'mm' if mm else 'px'} in all.")

    def save(self):
        from helio.application.research import export_toolpaths
        from helio_qt.publish import file_stem
        folder = QFileDialog.getExistingDirectory(self, "Folder for the toolpaths",
                                                  os.path.dirname(_start(self.win, "")))
        if not folder:
            return
        strokes, labels, skel, albedo = self.result
        files = export_toolpaths(strokes, labels, skel, albedo, folder, file_stem(object_name(self.project)),
                                 colours=self.colours(strokes))
        self.status.setText("Saved " + ", ".join(os.path.basename(f) for f in files) + ".")
        self.win.flash(f"Toolpaths saved: {len(strokes)} strokes.")


# ── inscription zones ────────────────────────────────────────────────────

class ZonesDialog(ToolDialog):
    TITLE = "Inscription zones"
    INTRO = "Marks the areas most likely to hold writing or worked decoration."
    METHOD = ("Edges and curvature in the depth and normals, specular contrast and dark-on-light "
              "colour, weighted for the kind of object, merged into letters, lines and blocks and "
              "ranked by density. Image processing, not reading: it finds where to look.")
    SAVE = "Save…"

    def build(self):
        from helio.application.research import INSCRIPTION_MODES
        self.mode = self.combo("Surface type", INSCRIPTION_MODES)
        self.sens = self.param("sens", "Sensitivity", 0.1, 1.0, 0.7, 0.05, "Higher finds fainter zones.")
        self.zones = self.combo("Most zones", ("8", "12", "16", "24", "32"), "16")
        self.form.addWidget(label("Show", ""))
        self.show_what = Segmented([("zones", "Zones"), ("signal", "Signal map")])
        self.show_what.set_current("zones")
        self.show_what.chosen.connect(lambda _k: self.show_result(self.result) if self.result else None)
        self.form.addWidget(self.show_what)

    def args(self):
        D = self.data()
        return dict(depth=D.depth_vis_p, normals=D.normals_full, spec=getattr(D, "spec_var", None),
                    albedo=D.albedo_rgb_full, mode=self.mode.currentText(), sens=self.sens.value(),
                    max_zones=int(self.zones.currentText()), render=current_view(self.win))

    def work(self, render, **kw):
        from helio.application.research import draw_zones, inscription_zones
        boxes, accum = inscription_zones(kw.pop("depth"), kw.pop("normals"), kw.pop("spec"),
                                         kw.pop("albedo"), **kw)
        return boxes, accum, draw_zones(render, boxes)

    def show_result(self, res):
        boxes, accum, drawn = res
        if self.show_what.current() == "signal":
            self.preview.set_image(cv2.cvtColor(cv2.applyColorMap(accum, cv2.COLORMAP_INFERNO),
                                                cv2.COLOR_BGR2RGB))
        else:
            self.preview.set_image(drawn)
        self.status.setText(f"{len(boxes)} zones." if boxes else
                            "No zones at this sensitivity. Raise it, or try another surface type.")
        self.btn_save.setEnabled(bool(boxes))

    def save(self):
        import csv
        path, _ = QFileDialog.getSaveFileName(self, "Save the zones", _start(self.win, "_inscription_zones.png"),
                                              "PNG (*.png)")
        if not path:
            return
        boxes, accum, drawn = self.result
        cv2.imwrite(path, cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
        base = os.path.splitext(path)[0]
        cv2.imwrite(base + "_signal.png", cv2.applyColorMap(accum, cv2.COLORMAP_INFERNO))
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["zone", "x_px", "y_px", "w_px", "h_px"])
            for i, b in enumerate(boxes, 1):
                w.writerow([f"Z{i}", *b])
        self.status.setText(f"Saved {os.path.basename(path)}, its signal map and the zone list.")
        self.win.flash(f"Inscription zones saved: {os.path.basename(path)}.")


# ── cross-object registration ────────────────────────────────────────────

class ClickImage(Preview):
    """An image that takes numbered control points."""
    clicked = Signal(float, float)

    def __init__(self, empty: str, colour: str):
        super().__init__(empty)
        self.setMinimumSize(QSize(260, 260))
        self.colour = QColor(colour)
        self.points: list[tuple[float, float]] = []
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _geom(self):
        if self.img is None:
            return None
        h, w = self.img.shape[:2]
        s = min(self.width() / w, self.height() / h)
        return s, (self.width() - w * s) / 2, (self.height() - h * s) / 2

    def mousePressEvent(self, e):
        g = self._geom()
        if g is None:
            return
        s, ox, oy = g
        x, y = (e.position().x() - ox) / s, (e.position().y() - oy) / s
        h, w = self.img.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.clicked.emit(x, y)

    def paintEvent(self, e):
        super().paintEvent(e)
        g = self._geom()
        if g is None or not self.points:
            return
        s, ox, oy = g
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setPen(QPen(self.colour, 2))
        for i, (x, y) in enumerate(self.points, 1):
            cx, cy = ox + x * s, oy + y * s
            qp.drawEllipse(int(cx - 5), int(cy - 5), 10, 10)
            qp.drawText(int(cx + 8), int(cy + 4), str(i))
        qp.end()


class RegisterDialog(QDialog):
    """Two images, matching points on each, one warped onto the other."""
    METHOD = ("Click the same features on both images, in the same order (at least 4). Similarity "
              "allows turning, uniform scaling and shifting; Affine adds shear and unequal "
              "scaling; Projective a full perspective change (fitted with RANSAC). The fit's RMS "
              "is how far, in pixels of B, the points miss after warping. Helio Classic's \"Affine\" was "
              "the similarity.")

    def __init__(self, win):
        from helio.application.research import MODELS
        super().__init__(win)
        self.win = win
        self.img_a = self.img_b = self.warped = self.composite = None
        self.M = self.rms = None
        self.setWindowTitle("Register two images")
        self.setMinimumSize(1100, 640)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.addWidget(label("Register two images", "title"))
        lay.addWidget(label("Overlay a pouncing pattern, a tracing or another object on this one.",
                            "secondary", wrap=True))
        bar = QHBoxLayout()
        b = button("Load image A…", "outline")
        b.clicked.connect(self.load_a)
        bar.addWidget(b)
        b = button("B: the current view", "outline")
        b.clicked.connect(self.use_view)
        bar.addWidget(b)
        self.model = QComboBox()
        self.model.addItems(list(MODELS))
        self.model.setAccessibleName("Transform")
        bar.addWidget(self.model)
        self.mirror = QCheckBox("Mirror A")
        bar.addWidget(self.mirror)
        bar.addWidget(label("Blend", ""))
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(0, 100)
        self.alpha.setValue(50)
        self.alpha.setFixedWidth(120)
        self.alpha.setAccessibleName("Blend")
        self.alpha.valueChanged.connect(self.reblend)
        bar.addWidget(self.alpha)
        bar.addStretch(1)
        lay.addLayout(bar)
        imgs = QHBoxLayout()
        self.view_a = ClickImage("Load image A, then click points on it.", "#ff4466")
        self.view_b = ClickImage("The current view.", "#44ff88")
        self.view_r = Preview("The overlay appears here.")
        self.view_r.setMinimumSize(QSize(260, 260))
        self.view_a.clicked.connect(lambda x, y: self.add_point("A", x, y))
        self.view_b.clicked.connect(lambda x, y: self.add_point("B", x, y))
        for v in (self.view_a, self.view_b, self.view_r):
            imgs.addWidget(v, 1)
        lay.addLayout(imgs, 1)
        sec = Section("How it works", "research_register", win.prefs, open_=False)
        sec.add(label(self.METHOD, "caption", wrap=True))
        lay.addWidget(sec)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        close = button("Close", "quiet")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        for text, fn in (("Clear A", lambda: self.clear("A")), ("Clear B", lambda: self.clear("B"))):
            b = button(text, "quiet")
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        self.btn_run = button("Register", "primary")
        self.btn_run.clicked.connect(self.register)
        self.btn_save = button("Export…", "outline")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_save)
        lay.addLayout(row)
        try:
            self.use_view()
        except RuntimeError:
            pass
        self.counts()

    def counts(self):
        a, b = len(self.view_a.points), len(self.view_b.points)
        self.status.setText(f"Points: A {a}, B {b}. Need at least 4 on each, in the same order.")

    def load_a(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Image A", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        if not path:
            return
        raw = cv2.imread(path)
        if raw is None:
            self.status.setText("That image could not be read.")
            return
        self.set_a(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))

    def set_a(self, rgb):
        self.img_a = np.ascontiguousarray(rgb)
        self.view_a.points = []
        self.view_a.set_image(self.img_a)
        self.counts()

    def use_view(self):
        self.img_b = np.ascontiguousarray(current_view(self.win))
        self.view_b.points = []
        self.view_b.set_image(self.img_b)
        self.counts()

    def add_point(self, which: str, x: float, y: float):
        v = self.view_a if which == "A" else self.view_b
        v.points.append((float(x), float(y)))
        v.update()
        self.counts()

    def clear(self, which: str):
        v = self.view_a if which == "A" else self.view_b
        v.points = []
        v.update()
        self.counts()

    def register(self):
        from helio.application.research import register_points, warp_to
        if self.img_a is None or self.img_b is None:
            self.status.setText("Load both images first.")
            return
        try:
            self.M, self.rms = register_points(self.view_a.points, self.view_b.points,
                                               self.model.currentText(), mirror=self.mirror.isChecked(),
                                               width_a=self.img_a.shape[1])
        except ValueError as e:
            self.status.setText(str(e))
            return
        self.warped = warp_to(self.img_a, self.M, self.img_b.shape, mirror=self.mirror.isChecked())
        self.reblend()
        self.btn_save.setEnabled(True)
        from helio.application.heritage_tools import match_words, relief_match
        try:
            self.match = relief_match(self.warped, self.img_b)
            match = f" Relief match {self.match:.2f} (1 = identical): {match_words(self.match)}."
        except ValueError:
            self.match, match = None, ""
        self.status.setText(f"{self.model.currentText()}: the points fit to {self.rms:.2f} px (RMS)."
                            + match)

    def reblend(self, *_):
        from helio.application.research import blend
        if self.warped is None:
            return
        self.composite = blend(self.warped, self.img_b, self.alpha.value() / 100.0)
        self.view_r.set_image(self.composite)

    def save(self):
        from helio.application.research import export_registration
        path, _ = QFileDialog.getSaveFileName(self, "Export the overlay", _start(self.win, "_registered.png"),
                                              "PNG (*.png);;TIFF (*.tif)")
        if not path:
            return
        files = export_registration(path, self.composite, self.warped, self.img_b, self.M, self.rms,
                                    self.model.currentText())
        self.status.setText("Saved " + ", ".join(os.path.basename(f) for f in files) + ".")
        self.win.flash(f"Registration saved: {os.path.basename(path)}.")


# ── focus stacking ───────────────────────────────────────────────────────

class FocusDialog(ToolDialog):
    TITLE = "Focus stacking"
    INTRO = ("For coins, seals and gems at high magnification: one sharp photograph per light "
             "from its focus bracket.")
    METHOD = ("Choose all the photographs, named so they sort light by light (light 1's frames, "
              "then light 2's …). Each bracket is aligned (ECC, translation) and fused. "
              "Laplacian-pyramid fusion (Burt & Kolczynski 1993) keeps the strongest detail at "
              "each scale and blends the rest, so there are no seams where the sharpest frame "
              "changes; Helio Classic took each pixel from one frame. 16-bit photographs stay 16-bit.")
    RUN = "Choose and stack…"
    SAVE = "Save merged…"

    def build(self):
        row = QHBoxLayout()
        row.addWidget(label("Lights", ""))
        self.lights = QSpinBox()
        self.lights.setRange(3, 500)
        self.lights.setValue(4)
        self.lights.setAccessibleName("Number of light positions")
        row.addWidget(self.lights)
        row.addWidget(label("Frames per light", ""))
        self.frames = QSpinBox()
        self.frames.setRange(2, 32)
        self.frames.setValue(8)
        self.frames.setAccessibleName("Focus frames per light")
        row.addWidget(self.frames)
        self.form.addLayout(row)
        self.align = QCheckBox("Align the frames first")
        self.align.setChecked(True)
        self.form.addWidget(self.align)
        self.form.addWidget(label("Fusion", ""))
        self.method = Segmented([("pyramid", "Pyramid"), ("select", "Sharpest pixel")],
                                {"pyramid": "Seamless (recommended).", "select": "Helio Classic's method."})
        self.method.set_current("pyramid")
        self.form.addWidget(self.method)
        self.btn_open = None
        self.paths = None

    def args(self):
        n = self.lights.value() * self.frames.value()
        paths = self.paths
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, f"Choose {n} photographs", "",
                                                    "Images (*.png *.jpg *.jpeg *.tif *.tiff)")
        self.paths = None
        if not paths:
            return None
        if len(paths) != n:
            raise ValueError(f"{self.lights.value()} lights × {self.frames.value()} frames is {n} "
                             f"photographs; {len(paths)} were chosen.")
        return dict(paths=list(paths), n_lights=self.lights.value(), n_frames=self.frames.value(),
                    align=self.align.isChecked(), method=self.method.current() or "pyramid")

    def work(self, paths, n_lights, n_frames, align, method):
        from helio.application.research import focus_stack
        return focus_stack(paths, n_lights, n_frames, align=align, method=method)

    def show_result(self, merged):
        first = merged[0]
        if first.dtype == np.uint16:
            first = (first / 257).astype(np.uint8)
        self.preview.set_image(first)
        self.status.setText(f"{len(merged)} merged photographs ready (light 1 shown).")

    def save(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder for the merged photographs",
                                                  os.path.dirname(_start(self.win, "")))
        if not folder:
            return
        for i, img in enumerate(self.result):
            cv2.imwrite(os.path.join(folder, f"light_{i + 1:02d}_stacked.tif"),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.saved_to = folder
        self.status.setText(f"Saved {len(self.result)} photographs. Open them to solve.")
        if self.btn_open is None:
            self.btn_open = button("Open them", "primary")
            self.btn_open.clicked.connect(lambda: (self.win.open_paths([self.saved_to]), self.accept()))
            self.form.itemAt(self.form.count() - 1).layout().addWidget(self.btn_open)


# ── recto / verso ────────────────────────────────────────────────────────

FACE_KINDS = (("pages", "Recto / verso"), ("coin", "Obverse / reverse"), ("seal", "Matrix / impression"))
FACES = {"pages": (("Recto", "Recto (front)"), ("Verso", "Verso (back)")),
         "coin": (("Obverse", "Obverse"), ("Reverse", "Reverse")),
         "seal": (("Matrix", "Matrix"), ("Impression", "Impression"))}


class UprightDial(QWidget):
    """A face's picture with a dial: turn until the design stands upright."""
    turned = Signal(float)

    def __init__(self, empty: str):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = Preview(empty)
        self.view.setMinimumSize(QSize(220, 220))
        lay.addWidget(self.view, 1)
        self.dial = QSlider(Qt.Orientation.Horizontal)
        self.dial.setRange(-180, 180)
        self.dial.setAccessibleName("Turn")
        self.dial.setToolTip("Turn until the design stands upright. The angle is how far the "
                             "design's top is turned clockwise in the photograph.")
        self.dial.valueChanged.connect(self._turn)
        lay.addWidget(self.dial)
        self.note = label("Turned 0°", "caption")
        lay.addWidget(self.note)
        self.img = None

    def set_image(self, rgb):
        h, w = rgb.shape[:2]
        k = 600.0 / max(h, w)
        self.img = cv2.resize(np.ascontiguousarray(rgb[..., :3]), (max(1, int(w * k)), max(1, int(h * k))),
                              interpolation=cv2.INTER_AREA) if k < 1 else np.ascontiguousarray(rgb[..., :3])
        self._turn(self.dial.value())

    def angle(self) -> float:
        return float(self.dial.value())

    def _turn(self, v):
        self.note.setText(f"Turned {v:+d}°")
        if self.img is not None:
            h, w = self.img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), float(v), 1.0)
            self.view.set_image(cv2.warpAffine(self.img, M, (w, h), borderValue=(40, 40, 40)))
        self.turned.emit(float(v))


class RectoVersoDialog(QDialog):
    """Link this face to the scan of the other face of the same object: a leaf's recto and verso,
    a coin's obverse and reverse (with its die axis), a seal's matrix and impression."""

    def __init__(self, win):
        from helio.domain.object_profiles import of
        super().__init__(win)
        self.win, self.project = win, win.project
        self.partner = ""
        self.link = None
        self.setWindowTitle("Two faces")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(8)
        lay.addWidget(label("Two faces", "title"))
        lay.addWidget(label(f"Object: {object_name(self.project)}", "secondary"))
        self.kind = Segmented(list(FACE_KINDS))
        prof = of(self.project.capture_meta).key
        self.kind.set_current("coin" if prof == "coin" else "seal" if prof in ("seal", "gem")
                              else "pages")
        self.kind.chosen.connect(self._kind)
        lay.addWidget(self.kind)
        lay.addWidget(label("This side is the", ""))
        self.side_box = QVBoxLayout()
        lay.addLayout(self.side_box)
        self.side = None
        row = QHBoxLayout()
        b = button("Choose the other side's file…", "outline")
        b.clicked.connect(self.choose_partner)
        row.addWidget(b)
        self.partner_lbl = label("No partner yet.", "caption")
        row.addWidget(self.partner_lbl, 1)
        lay.addLayout(row)
        self.flip = QCheckBox("Flip the partner horizontally (usual for plates and leaves)")
        self.flip.setChecked(True)
        lay.addWidget(self.flip)
        # die axis: each face turned upright by eye, the difference in clock hours
        self.axis_box = QWidget()
        al = QVBoxLayout(self.axis_box)
        al.setContentsMargins(0, 6, 0, 0)
        al.addWidget(label("DIE AXIS", "heading"))
        al.addWidget(label("Turn each face until its design stands upright. Photograph the "
                           "reverse after turning the coin over left to right.", "caption", wrap=True))
        faces = QHBoxLayout()
        self.dial_obv = UprightDial("This face: the current view.")
        self.dial_rev = UprightDial("The other face: choose an image of it.")
        faces.addWidget(self.dial_obv, 1)
        faces.addWidget(self.dial_rev, 1)
        al.addLayout(faces)
        arow = QHBoxLayout()
        self.axis_lbl = label("", "")
        arow.addWidget(self.axis_lbl, 1)
        self.btn_axis = button("Record in Capture details", "outline")
        self.btn_axis.clicked.connect(self.record_axis)
        arow.addWidget(self.btn_axis)
        al.addLayout(arow)
        self.dial_obv.turned.connect(self._axis)
        self.dial_rev.turned.connect(self._axis)
        lay.addWidget(self.axis_box)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        close = button("Close", "quiet")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        row.addStretch(1)
        self.btn_link = button("Save the link…", "primary")
        self.btn_link.clicked.connect(self.save_link)
        self.btn_export = button("Export this side…", "outline")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export)
        row.addWidget(self.btn_link)
        row.addWidget(self.btn_export)
        lay.addLayout(row)
        self._kind(self.kind.current())
        try:
            self.dial_obv.set_image(current_view(win))
        except RuntimeError:
            pass

    def _kind(self, kind):
        if self.side is not None:
            self.side_box.removeWidget(self.side)
            self.side.deleteLater()
        faces = FACES[kind]
        self.side = Segmented(list(faces))
        self.side.set_current(faces[0][0])
        self.side_box.addWidget(self.side)
        self.flip.setChecked(kind == "pages")
        self.axis_box.setVisible(kind == "coin")
        self.adjustSize()
        self._axis()

    def die_axis(self) -> int | None:
        """Clock hours, once the other face's picture is in; None before."""
        from helio.domain.object_profiles import die_axis_hours
        if self.kind.current() != "coin" or self.dial_rev.img is None:
            return None
        obv, rev = self.dial_obv.angle(), self.dial_rev.angle()
        if self.side.current() == "Reverse":
            obv, rev = rev, obv
        return die_axis_hours(obv, rev)

    def _axis(self, *_):
        h = self.die_axis()
        self.axis_lbl.setText(f"Die axis: {h} h  ({h * 30 % 360}°)" if h else
                              "Die axis: choose an image of the other face.")
        self.btn_axis.setEnabled(h is not None)

    def record_axis(self):
        h = self.die_axis()
        if h is None:
            return
        meta = dict(self.project.capture_meta)
        meta["die_axis"] = str(h)
        meta["face"] = self.side.current()
        self.project.capture_meta = meta
        self.project.changed.emit()
        self.status.setText(f"Die axis {h} h recorded in Capture details.")

    def choose_partner(self, path: str | None = None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "The other side", "",
                                                  "Sessions and images (*.helio *.tif *.tiff *.png *.jpg "
                                                  "*.jpeg *.json);;All (*)")
        if path:
            self.partner = path
            self.partner_lbl.setText(os.path.basename(path))
            if os.path.splitext(path)[1].lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                raw = cv2.imread(path)
                if raw is not None:
                    self.dial_rev.set_image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            self._axis()

    def save_link(self):
        from helio.application.research import recto_verso_link
        from helio_qt.publish import file_stem
        if not self.partner:
            self.status.setText("Choose the other side's file first.")
            return
        name = object_name(self.project)
        pair = "recto_verso" if self.kind.current() == "pages" else "two_faces"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the link", os.path.join(os.path.dirname(_start(self.win, "")),
                                                f"{file_stem(name)}_{pair}_link.json"), "JSON (*.json)")
        if not path:
            return
        s = self.project.settings
        self.link = recto_verso_link(path, object_id=name, this_surface=self.side.current(),
                                     partner_file=self.partner, auto_flip_partner=self.flip.isChecked(),
                                     solver=s.depth_engine, normal_engine=s.resolved_normal_engine(),
                                     die_axis=self.die_axis())
        self.btn_export.setEnabled(self.win.render_data is not None)
        self.status.setText(f"Linked: this is the {self.link['this_surface'].lower()}. "
                            "Export this side next, then do the same in the other side's session.")

    def export(self):
        from helio.application.research import recto_verso_export
        D = self.win.render_data
        if D is None or self.link is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for this side", os.path.dirname(_start(self.win, "")))
        if not folder:
            return
        depth = getattr(D, "depth_poisson", None)
        files = recto_verso_export(folder, self.link, D.albedo_rgb_full,
                                   depth if depth is not None else D.depth_vis_p)
        self.status.setText("Saved " + ", ".join(os.path.basename(f) for f in files) + ".")
        self.win.flash(f"{self.link['this_surface']} exported.")


# ── legend unwrap ────────────────────────────────────────────────────────

UNWRAP_SOURCES = (("relief", "Relief"), ("raking", "Even raking light"), ("colour", "Colour"),
                  ("view", "View on screen"), ("other", "Another view…"))
UNWRAP_LAYERS = (("Combined Colour", "Colour"), ("Shading", "Clay"), ("Roughness Fusion", "Roughness"))


class UnwrapDialog(ToolDialog):
    TITLE = "Legend unwrap"
    INTRO = ("Unrolls the lettering round a seal, coin, stamp, binding centrepiece or volvelle into a "
             "strip that reads left to right. On a page with several, Find objects numbers them: "
             "click the one to read.")
    METHOD = ("Find fits the edge: a circle by the Hough transform; an oval or pointed oval (vesica) "
              "through the strongest edge along 180 rays, in the relief when the colour hardly shows "
              "it. Straighten moves the outline until the borders run flat along the strip. The band "
              "is resampled at a constant depth inside the outline, so letters at the ends of an oval "
              "are as tall as at its sides. Relief is how steeply the surface tilts; the even raking "
              "light turns with the legend so every letter is lit from the same side of itself. Whole "
              "face lays every ring of a volvelle, astrolabe or rota out as rows.")
    RUN = "Unwrap"
    SAVE = "Save…"
    PREVIEW = DragPreview

    def __init__(self, win):
        super().__init__(win)
        self.preview.pressed.connect(self._press)          # the preview is made after build()
        self.preview.moved.connect(self._drag)
        self.preview.released.connect(self._release)
        self.preview.double.connect(self._double)

    def build(self):
        from helio.application.heritage_tools import SHAPES
        self.shape = Segmented(list(SHAPES.items()), columns=2, tips={
            "circle": "Coins, most seals, round library and ownership stamps, volvelles.",
            "oval": "Oval seals, stamps and binding centrepieces.",
            "vesica": "Pointed ovals: the usual shape of ecclesiastics' and women's seals.",
            "free": "Any outline, through points you drag: lozenges, shields, cartouches, a "
                    "worn or broken edge. Double-click the edge to add a point; Option-click "
                    "a point to remove it."})
        self.shape.set_current("circle")
        self.shape.chosen.connect(lambda _k: self._shape_changed())
        self.form.addWidget(self.shape)
        row = QHBoxLayout()
        b = button("Find the edge", "outline", "Fit the outline to the object's edge (in the "
                                               "selection, when Inspect has one).")
        b.clicked.connect(self.find)
        st = button("Straighten", "outline", "Move the outline until the borders run flat.")
        st.clicked.connect(self.straighten)
        row.addWidget(b)
        row.addWidget(st)
        self.form.addLayout(row)
        orow = QHBoxLayout()
        fo = button("Find objects", "", "Rings every seal, coin or stamp on the page "
                                                    "and numbers them: click one to unwrap it.")
        fo.clicked.connect(self.find_objects)
        self.which = QComboBox()
        self.which.setAccessibleName("Object")
        self.which.setToolTip("Which of the objects found to unwrap.")
        self.which.activated.connect(lambda i: self.choose_object(i))
        self.which.setVisible(False)
        self.btn_page = button("Page", "", "Show the whole page with the objects found, to pick another.")
        self.btn_page.clicked.connect(self.show_page)
        self.btn_page.setVisible(False)
        orow.addWidget(fo, 1)
        orow.addWidget(self.which)
        orow.addWidget(self.btn_page)
        self.form.addLayout(orow)
        self.objects, self._page = [], False
        self.size = self.param("size", "Edge", 5, 300, 97, 0.5, "Where the band's outer edge is, as a "
                               "share of the outline found.", "%")
        self.width_ = self.param("width", "Band depth", 1, 95, 20, 0.5, "How deep the band is, as a share "
                                 "of the outline's half-width.", "%")
        self.start = self.param("start", "Start at", 0, 359, 0, 1,
                                "Where the strip begins, clockwise from 12 o'clock: most legends begin "
                                "with a cross at the top.", "°")
        self.direction = self.combo("Legend runs", ("Clockwise", "Anticlockwise"),
                                    tip="Clockwise: letters' tops to the edge, the usual. "
                                        "Anticlockwise: their feet to the edge.")
        self.source = self.combo("Read from", [t for _k, t in UNWRAP_SOURCES],
                                 tip="Relief reads worn and paper-covered impressions; the even "
                                     "raking light lights every letter alike.")
        self.source.currentTextChanged.connect(lambda _t: self._source_changed())
        from helio.domain.registry import RENDER_ENGINES
        self.other = QWidget()
        ol = QHBoxLayout(self.other)
        ol.setContentsMargins(0, 0, 0, 0)
        self.other_view = QComboBox()
        self.other_view.addItems(list(RENDER_ENGINES))
        self.other_view.setAccessibleName("View")
        self.other_view.setToolTip("Any view, drawn with the settings it has in Inspect.")
        self.other_layer = QComboBox()
        self.other_layer.addItems([t for _k, t in UNWRAP_LAYERS])
        self.other_layer.setCurrentText("Clay")
        self.other_layer.setAccessibleName("Layer")
        self.other_layer.setToolTip("Colour: lit over the photograph's colour. Clay: the shape alone.")
        ol.addWidget(self.other_view, 1)
        ol.addWidget(self.other_layer)
        self.other.setVisible(False)
        self.form.addWidget(self.other)
        self.other_view.activated.connect(lambda _i: self._draw_other())
        self.other_layer.activated.connect(lambda _i: self._draw_other())
        self.light = self.param("light", "Light from", 0, 359, 90, 5, "On the strip, clockwise from its "
                                "top: 90 lights each letter from its right.", "°")
        self.height_ = self.param("elev", "Light height", 5, 70, 25, 1, "Lower rakes harder.", "°")
        more = Section("More", "research_unwrap_more", self.win.prefs, open_=False)
        self._more = {}
        for key, text, lo, hi, dflt, step, tip, unit in (
                ("span", "Span", 10, 360, 360, 5, "Less than all the way round, for a fragment or "
                 "part of a rim.", "°"),
                ("dx", "Centre across", -100, 100, 0, 0.25, "Nudge the outline.", "%"),
                ("dy", "Centre down", -100, 100, 0, 0.25, "Nudge the outline.", "%"),
                ("stretch", "Proportions", -200, 200, 0, 0.5, "Taller (+) or wider (-).", "%"),
                ("turn", "Turn", -90, 90, 0, 0.5, "Turn the outline.", "°")):
            r = ParamRow(Control(key, text, lo, hi, dflt, step, tip, (), unit))
            more.add(r)
            self._more[key] = r
        self.mirror = QCheckBox("Mirror: a matrix or die")
        self.mirror.setToolTip("Reads a seal matrix or coin die, whose legend is cut in reverse.")
        self.whole = QCheckBox("Whole face: every ring as a row")
        self.whole.setToolTip("For volvelles, astrolabe plates and circular diagrams: from the edge to "
                              "the centre, the outer ring at its true size.")
        self.scale = QCheckBox("Scale beneath: degrees, and mm once set")
        self.scale.setChecked(True)
        for c in (self.mirror, self.whole, self.scale):
            more.add(c)
        self.form.addWidget(more)
        self.form.addWidget(label("Legend reads", ""))
        self.reads = QLineEdit()
        self.reads.setPlaceholderText("e.g. + SIGILLVM ... (saved with the strip)")
        self.reads.setText(str(self.project.capture_meta.get("legend", "")))
        self.form.addWidget(self.reads)
        self.with_pic = QCheckBox("Save the object above the strip")
        self.with_pic.setToolTip("One picture: the object, cropped round its outline, over its strip.")
        self.form.addWidget(self.with_pic)
        self.form.addWidget(label("Drag on the picture: inside to move the outline, its edge to "
                                  "resize it (an oval's side or end alone; Option-drag for both), "
                                  "the inner line to set the band's depth, a free shape's points "
                                  "to reshape it.", "caption", wrap=True))
        self._map = self._box = None
        self._frozen = None
        self._grab = None
        for r in (self.size, self.width_, self.start, *self._more.values()):
            r.changed.connect(lambda *_a: self._ring())
        self.whole.toggled.connect(lambda _on: self._ring())
        self.direction.currentTextChanged.connect(lambda _t: self._ring())
        self.img = self.pics = None
        self.outline = None
        self._shape_changed()

    # what the strip is read from, all at the view's size
    def _pictures(self):
        from helio.application.heritage_tools import _as_rgb8, relief_picture
        h, w = self.img.shape[:2]
        D = self.win.render_data
        pics = {"view": self.img}
        nrm = getattr(D, "normals_full", None)
        if nrm is not None:
            nrm = np.asarray(nrm, np.float32)
            if nrm.shape[:2] != (h, w):
                nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_LINEAR)
            pics["normals"] = nrm
            pics["relief"] = relief_picture(nrm)
        alb = getattr(D, "albedo_rgb_full", None)
        if alb is not None:
            alb = _as_rgb8(alb)
            pics["colour"] = alb if alb.shape[:2] == (h, w) else cv2.resize(alb, (w, h))
        return pics

    def showEvent(self, e):
        super().showEvent(e)
        if self.img is not None:
            self._refresh_view()                 # the view may have changed since
            return
        try:
            self.img = current_view(self.win)
        except RuntimeError as err:
            self.status.setText(str(err))
            return
        self._view_sig = view_signature(self.win)
        self.pics = self._pictures()
        if "relief" not in self.pics:
            self.source.setCurrentText("Colour" if "colour" in self.pics else "View on screen")
        self.other_view.setCurrentText(self.win.engine)
        self._source_changed()
        sel = self.win.selection
        if not (sel and sel[2] > 16 and sel[3] > 16) and self.find_objects(quiet=True) > 1:
            return                               # several on the page: the person picks one
        if self.outline is None:
            self.find()

    def _refresh_view(self) -> bool:
        """Draw the view on screen again if it changed (another view, Colour
        or Clay, a slider moved) since it was last read."""
        if self.win.render_data is None:
            return False
        sig = view_signature(self.win)
        if sig == getattr(self, "_view_sig", None):
            return False
        try:
            pic = current_view(self.win)
        except RuntimeError:
            return False
        self._view_sig = sig
        if pic.shape[:2] == self.img.shape[:2]:
            self.img = pic
            if self.pics is not None:
                self.pics["view"] = pic
        if self._key(self.source, UNWRAP_SOURCES) == "view":
            self._ring()
        return True

    def _draw_other(self):
        """Another view, drawn in the background (a full-size picture of some
        views takes a few seconds)."""
        if self.img is None:
            return
        eng = self.other_view.currentText()
        layer = {t: k for k, t in UNWRAP_LAYERS}[self.other_layer.currentText()]
        key = (eng, layer, repr(sorted(self.win.engine_settings.get(eng, {}).items())))
        if self.pics.get("other_key") == key:
            self._ring()
            return
        self.status.setText(f"Drawing {eng}…")
        h, w = self.img.shape[:2]

        def done(pic, err):
            if err:
                self.status.setText(err.split(": ", 1)[-1])
                return
            if pic.shape[:2] != (h, w):
                pic = cv2.resize(pic, (w, h), interpolation=cv2.INTER_AREA)
            self.pics["other"], self.pics["other_key"] = pic, key
            self.status.setText(f"Reading from {eng}, {self.other_layer.currentText()}.")
            if self.result is not None:
                self.result = None
                self.btn_save.setEnabled(False)
            self._ring()
        self.win.run_background(lambda: view_picture(self.win, eng, layer), done, tool=True)

    # several objects on one page
    def find_objects(self, quiet: bool = False) -> int:
        """Ring and number every seal or coin on the page; the most likely
        one is chosen.  Returns how many were found."""
        from helio.application.heritage_tools import find_objects
        if self.img is None:
            return 0
        pics = [p for p in (self.pics.get("colour"), self.pics.get("relief")) if p is not None] \
            or [self.img]
        self.objects = find_objects(*pics)
        n = len(self.objects)
        self.which.clear()
        self.which.addItems([f"Object {i + 1}" for i in range(n)])
        self.which.setVisible(n > 1)
        self.btn_page.setVisible(n > 1)
        if n == 0:
            if not quiet:
                self.status.setText("No seal or coin stood out. Select one in Inspect, or drag the "
                                    "outline onto it.")
            return 0
        best = max(range(n), key=lambda i: self.objects[i]["score"])
        if n == 1:
            self.choose_object(best)
            return 1
        self.which.setCurrentIndex(best)
        self.show_page()
        self.status.setText(f"{n} objects found. Click one on the picture, or choose it from the "
                            "list.")
        return n

    def show_page(self):
        if not self.objects:
            return
        self._page = True
        self._ring()

    def choose_object(self, i: int):
        """Unwrap object *i*: its edge found again in the part round it, in
        the shape chosen; if that strays, the ring found on the page."""
        from helio.application.heritage_tools import find_outline, free_from, object_crop
        if not 0 <= i < len(self.objects):
            return
        ob = self.objects[i]
        self.which.setCurrentIndex(i)
        h, w = self.img.shape[:2]
        x, y, cw, ch = object_crop(ob, w, h)
        pic = self.pics.get("relief", self.img) if self.pics else self.img
        shape = self.shape.current()
        o = find_outline(pic[y:y + ch, x:x + cw], shape,
                         near=(ob["cx"] - x, ob["cy"] - y, math.sqrt(ob["a"] * ob["b"])))
        o = dict(o, cx=o["cx"] + x, cy=o["cy"] + y)
        size, size0 = math.sqrt(o["a"] * o["b"]), math.sqrt(ob["a"] * ob["b"])
        if math.hypot(o["cx"] - ob["cx"], o["cy"] - ob["cy"]) > 0.3 * size0 or not 0.75 < size / size0 < 1.25:
            o = dict(ob, shape=shape)
            if shape == "circle":
                o.update(a=size0, b=size0, turn=0.0)
            elif shape == "free":
                o = free_from(dict(ob, shape="oval"))
        o.pop("score", None)
        self.outline = o
        for r in self._more.values():
            r.set_value(r.c.default)
        self._page = False
        if self.result is not None:
            self.result = None
            self.btn_save.setEnabled(False)
        self._ring()
        self.status.setText(f"Object {i + 1}. " + (self._size_words() or "Straighten it, then Unwrap."))

    def _key(self, combo, pairs):
        return {t: k for k, t in pairs}[combo.currentText()]

    def _shape_changed(self):
        from helio.application.heritage_tools import free_from
        shape = self.shape.current()
        for k in ("stretch", "turn"):
            self._more[k].setVisible(shape != "circle")
        if self.outline is not None:
            was = self.outline.get("shape")
            if shape == "free" and was != "free":
                # the outline on screen becomes points to drag, nothing moved
                self._bake()
                k = self.size.value() / 100.0
                eff = dict(self.outline, a=self.outline["a"] * k, b=self.outline["b"] * k)
                fr = free_from(eff)
                self.outline = dict(fr, a=fr["a"] / k, b=fr["b"] / k)
            elif was == "free" and shape != "free":
                self.find()
                return
            else:
                self.outline = dict(self.outline, shape=shape)
                if shape == "circle":           # round: an oval's two radii made one
                    r = math.sqrt(self.outline["a"] * self.outline["b"])
                    self.outline.update(a=r, b=r, turn=0.0)
                if shape in ("oval", "vesica") and abs(self.outline["a"] - self.outline["b"]) < 1:
                    self.find()
                    return
        self._ring()

    def _bake(self):
        """Fold the nudges (centre, proportions, turn) into the outline and set them back to 0,
        so a drag starts from the outline as drawn."""
        if self.outline is None:
            return
        o, _w = self.ring()
        k = self.size.value() / 100.0
        self.outline = dict(o, a=o["a"] / k, b=o["b"] / k)
        for key in ("dx", "dy", "stretch", "turn"):
            self._more[key].set_value(0)

    # dragging on the preview
    def _to_picture(self, x, y):
        m = self._map
        f = m["k"] * m["r"]
        return m["x0"] + (x - m["ox"]) / f, m["y0"] + (y - m["oy"]) / f, f

    def _press(self, x, y, sc, alt):
        from helio.application.heritage_tools import free_anchors, outline_points
        if self._map is None or (self.outline is None and not self._page):
            return
        px, py, f = self._to_picture(x, y)
        if self._page:                         # the page: a click picks the object under it
            inside = [i for i, ob in enumerate(self.objects)
                      if math.hypot(px - ob["cx"], py - ob["cy"]) < 1.15 * max(ob["a"], ob["b"])]
            if inside:
                self.choose_object(min(inside, key=lambda i: math.hypot(
                    px - self.objects[i]["cx"], py - self.objects[i]["cy"])))
            return
        if self.result is not None:            # the strip no longer matches once the outline moves
            self.result = None
            self.btn_save.setEnabled(False)
        self._bake()
        o, _w = self.ring()
        # 12 screen pixels, in the picture's, but never so wide that a small ring is all edge
        tol = min(12.0 / max(sc * f, 1e-6), 0.2 * min(o["a"], o["b"]))
        self._frozen = self._box
        grab = dict(o=o, outline=dict(self.outline), p=(px, py))
        if o.get("shape") == "free":
            A = free_anchors(o)
            d = np.hypot(A[:, 0] - px, A[:, 1] - py)
            i = int(np.argmin(d))
            if d[i] < tol:
                if alt:
                    if len(A) > 3:
                        self._set_anchors(o, np.delete(A, i, axis=0))
                        self.status.setText(f"Point removed: {len(A) - 1} left.")
                    self._frozen = None
                    self._ring()
                    return
                self._grab = dict(grab, mode="anchor", i=i, anchors=A)
                self._ring()
                return
        P = outline_points(o, 720).astype(np.float32)
        inside = cv2.pointPolygonTest(P.reshape(-1, 1, 2), (float(px), float(py)), True)
        to_edge, to_inner = abs(inside), abs(inside - _w)
        if to_edge < tol and to_edge <= to_inner:
            th = math.radians(float(o.get("turn", 0.0)))
            du, dv = px - o["cx"], py - o["cy"]
            lu, lv = du * math.cos(th) + dv * math.sin(th), -du * math.sin(th) + dv * math.cos(th)
            # an oval's side or end moves alone; a circle, a free shape or Option: both, in proportion
            axis = "" if alt or o.get("shape") not in ("oval", "vesica") \
                else "a" if abs(lu) > 2 * abs(lv) else "b" if abs(lv) > 2 * abs(lu) else ""
            self._grab = dict(grab, mode="edge", axis=axis, r0=max(math.hypot(du, dv), 1.0),
                              span0=max(abs(lu if axis == "a" else lv), 1.0), th=th)
        elif not self.whole.isChecked() and inside > 0 and to_inner < tol:
            self._grab = dict(grab, mode="inner", side=min(o["a"], o["b"]), P=P.reshape(-1, 1, 2))
        else:
            self._grab = dict(grab, mode="move")
        self._ring()

    def _set_anchors(self, o_eff, anchors):
        from helio.application.heritage_tools import with_anchors
        k = self.size.value() / 100.0
        q = with_anchors(o_eff, anchors)
        self.outline = dict(q, a=q["a"] / k, b=q["b"] / k)

    def _drag(self, x, y):
        g = self._grab
        if g is None:
            return
        px, py, _f = self._to_picture(x, y)
        dx, dy = px - g["p"][0], py - g["p"][1]
        if g["mode"] == "move":
            self.outline = dict(g["outline"], cx=g["outline"]["cx"] + dx, cy=g["outline"]["cy"] + dy)
        elif g["mode"] == "edge":
            du, dv = px - g["o"]["cx"], py - g["o"]["cy"]
            q = dict(g["outline"])
            if g["axis"]:
                lu, lv = (du * math.cos(g["th"]) + dv * math.sin(g["th"]),
                          -du * math.sin(g["th"]) + dv * math.cos(g["th"]))
                f = abs(lu if g["axis"] == "a" else lv) / g["span0"]
                q[g["axis"]] = max(1.0, q[g["axis"]] * f)
            else:
                f = math.hypot(du, dv) / g["r0"]
                q.update(a=max(1.0, q["a"] * f), b=max(1.0, q["b"] * f))
            self.outline = q
        elif g["mode"] == "inner":
            depth = cv2.pointPolygonTest(g["P"], (float(px), float(py)), True)
            self.width_.set_value(100.0 * max(depth, 0.0) / g["side"])
        else:
            A = g["anchors"].copy()
            A[g["i"]] = (px, py)
            self._set_anchors(g["o"], A)
        self._ring()

    def _release(self):
        if self._grab is not None:
            self.status.setText((self._size_words() + " " if self._size_words() else "")
                                + "Unwrap to see the strip.")
        self._grab = None
        self._frozen = None
        self._ring()

    def _double(self, x, y, sc):
        from helio.application.heritage_tools import free_anchors
        if self.outline is None or self._map is None or self.outline.get("shape") != "free" or self._page:
            return
        px, py, f = self._to_picture(x, y)
        self._bake()
        o, _w = self.ring()
        A = free_anchors(o)
        n = len(A)
        # after the anchor whose segment passes nearest the click
        best, at = None, 0
        for i in range(n):
            a, b = A[i], A[(i + 1) % n]
            ab = b - a
            t = float(np.clip(np.dot((px, py) - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
            d = float(np.hypot(*((a + t * ab) - (px, py))))
            if best is None or d < best:
                best, at = d, i + 1
        if best > 24.0 / max(sc * f, 1e-6):
            return
        self._set_anchors(o, np.insert(A, at, (px, py), axis=0))
        self.status.setText(f"Point added: {n + 1} points.")
        self._ring()

    def _source_changed(self):
        k = self._key(self.source, UNWRAP_SOURCES)
        self.light.setVisible(k == "raking")
        self.height_.setVisible(k == "raking")
        self.other.setVisible(k == "other")
        if k == "view" and self.img is not None:
            self._refresh_view()
        if k == "other":
            self._draw_other()
            return
        self._ring()

    def _find_in(self):
        """The picture Find looks in: the relief when there is one, within the selection."""
        pic = self.pics.get("relief", self.img) if self.pics else self.img
        sel = self.win.selection
        h, w = self.img.shape[:2]
        D = self.win.render_data
        if sel and D is not None and getattr(D, "w", 0):
            k = w / float(D.w)
            x, y, sw, sh = (int(round(v * k)) for v in sel)
            if sw > 16 and sh > 16:
                return pic[y:y + sh, x:x + sw], x, y
        return pic, 0, 0

    def find(self):
        from helio.application.heritage_tools import find_outline
        if self.img is None:
            return
        self._page = False
        pic, x, y = self._find_in()
        o = find_outline(pic, self.shape.current())
        self.outline = dict(o, cx=o["cx"] + x, cy=o["cy"] + y)
        for r in self._more.values():
            r.set_value(r.c.default)
        self._ring()
        self.status.setText(self._size_words() or "Edge found. Straighten it, then Unwrap.")

    def straighten(self):
        from helio.application.heritage_tools import straighten
        if self.img is None or self.outline is None or self._page:
            return
        o, width = self.ring()
        pic = self.pics.get("relief", self.img)
        self.status.setText("Straightening…")

        def done(res, err):
            if err:
                self.status.setText(err.split(": ", 1)[-1])
                return
            # back from the band's edge to the outline the sliders adjust
            k = self.size.value() / 100.0
            self.outline = dict(res, a=res["a"] / k, b=res["b"] / k)
            for key in ("dx", "dy", "stretch", "turn"):
                self._more[key].set_value(0)
            self._ring()
            self.status.setText("Straightened. " + self._size_words())
        self.win.run_background(lambda: straighten(pic, o, width), done, tool=True)

    def ring(self):
        """(outline of the band's outer edge, band depth in pixels)."""
        o = dict(self.outline)
        g = {k: r.value() for k, r in self._more.items()}
        half = math.sqrt(o["a"] * o["b"])
        o["cx"] += g["dx"] / 100 * half
        o["cy"] += g["dy"] / 100 * half
        f = math.exp(g["stretch"] / 100.0)
        k = self.size.value() / 100.0
        o["a"], o["b"] = o["a"] * k / math.sqrt(f), o["b"] * k * math.sqrt(f)
        o["turn"] = float(o.get("turn", 0.0)) + g["turn"]
        o["shape"] = self.shape.current()
        return o, max(2.0, min(o["a"], o["b"]) * self.width_.value() / 100.0)

    def _size_words(self) -> str:
        mm = self.project.scale_mm_per_px
        if not mm or self.outline is None:
            return ""
        o = self.outline
        if self.shape.current() == "circle":
            return f"{(o['a'] + o['b']) * mm:.1f} mm across."
        return f"{2 * o['b'] * mm:.1f} × {2 * o['a'] * mm:.1f} mm (height × width)."

    def _shown(self):
        k = self._key(self.source, UNWRAP_SOURCES)
        if k == "view":
            return self.img
        pic = self.pics.get(k if k != "raking" else "relief") if self.pics else None
        return pic if pic is not None else self.img

    def _drawn(self, max_side=1600):
        """The picture round the object (a seal is a small part of a charter),
        with the band and the strip's start drawn on it."""
        from helio.application.heritage_tools import draw_outline, outline_points
        o, width = self.ring()
        pic = self._shown()
        h, w = pic.shape[:2]
        if self._frozen is not None:            # held still while dragging, so the point stays under the mouse
            x0, y0, x1, y1 = self._frozen
        else:
            P = outline_points(o, 360)
            (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
            m = 0.3 * max(x1 - x0, y1 - y0)
            x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
            x1, y1 = int(min(w, x1 + m)), int(min(h, y1 + m))
            if x1 - x0 < 16 or y1 - y0 < 16:
                x0, y0, x1, y1 = 0, 0, w, h
        self._box = (x0, y0, x1, y1)
        self._map = dict(x0=x0, y0=y0, k=min(1.0, max_side / float(max(x1 - x0, y1 - y0))),
                         ox=0.0, oy=0.0, r=1.0)
        return draw_outline(pic[y0:y1, x0:x1], dict(o, cx=o["cx"] - x0, cy=o["cy"] - y0), width,
                            start_deg=self.start.value(), span_deg=self._more["span"].value(),
                            clockwise=self.direction.currentText() == "Clockwise",
                            whole_face=self.whole.isChecked(), max_side=max_side)

    def _ring(self):
        if self.img is None:
            return
        if self._page and self.objects:
            from helio.application.heritage_tools import draw_objects
            pic = self._shown()
            h, w = pic.shape[:2]
            self._map = dict(x0=0, y0=0, k=min(1.0, 1600 / float(max(h, w))), ox=0.0, oy=0.0, r=1.0)
            self.preview.set_image(draw_objects(pic, self.objects, self.which.currentIndex()))
            return
        if self.outline is None:
            return
        self.preview.set_image(self._drawn())

    def args(self):
        if self.img is None or self.outline is None:
            raise RuntimeError("Process the images first.")
        if self._page:
            raise RuntimeError("Choose an object first: click one on the page.")
        src = self._key(self.source, UNWRAP_SOURCES)
        if src == "view":
            self._refresh_view()
        if src == "other" and "other" not in self.pics:
            raise RuntimeError("The other view is still being drawn.")
        o, width = self.ring()
        if src in ("relief", "raking") and "normals" not in self.pics:
            raise RuntimeError("This solve has no normals: read from the colour or the view.")
        return dict(pics=self.pics, o=o, width=width, src=src, start=self.start.value(),
                    span=self._more["span"].value(), clockwise=self.direction.currentText() == "Clockwise",
                    whole=self.whole.isChecked(), mirror=self.mirror.isChecked(),
                    light=self.light.value(), elev=self.height_.value(), scale=self.scale.isChecked(),
                    mm=self.project.scale_mm_per_px or 0.0)

    def work(self, pics, o, width, src, start, span, clockwise, whole, mirror, light, elev, scale, mm):
        from helio.application import heritage_tools as T
        pic = pics["normals"] if src == "raking" else pics.get(src, pics["view"])
        if mirror:
            w = pics["view"].shape[1]
            pic = pic[:, ::-1].copy()
            if src == "raking":
                pic[..., 0] *= -1
            o = T.mirrored(o, w)
        maps = T.legend_maps(o, width, start_deg=(-start) % 360 if mirror else start, span_deg=span,
                             clockwise=clockwise, whole_face=whole)
        if src == "raking":
            strip = T.even_raking(pic, maps, light_deg=light, elevation=elev)
        else:
            strip = T.unwrap(pic, maps)
        strip = T._as_rgb8(strip)
        mid_mm = mm * maps["length"] / max(maps["map_x"].shape[1], 1) if mm else 0.0
        band = T.legend_scale(maps, mm_per_px=mid_mm) if scale else None
        rec = T.outline_record(o, width, mm_per_px=mm, source=pics.get("other_key", ("",))[0] or "Another view"
                               if src == "other" else dict(UNWRAP_SOURCES)[src],
                               start_deg=start, span_deg=span,
                               direction="clockwise" if clockwise else "anticlockwise",
                               mirrored=mirror, whole_face=whole,
                               strip_px=[int(strip.shape[1]), int(strip.shape[0])])
        if src == "raking":
            rec.update(light_deg=light, light_height_deg=elev)
        return dict(strip=strip, band=band, record=rec)

    def sheet_picture(self):
        """The strip and its scale, as Save writes it (the object above it when ticked)."""
        from helio.application.heritage_tools import legend_figure
        if self.result is None:
            return None
        out = self._composed(self.result)
        if self.with_pic.isChecked():
            o, _w = self.ring()
            out = legend_figure(self._shown(), o, out)
        return out

    def sheet_caption_text(self) -> str:
        if self.sheet_caption.text().strip():
            return self.sheet_caption.text().strip()
        reads = self.reads.text().strip()
        return "Legend unrolled" + (f": {reads}" if reads else "")

    def _composed(self, res):
        strip = res["strip"]
        return strip if res["band"] is None else np.vstack([strip, res["band"]])

    def show_result(self, res):
        ringed = self._drawn(max_side=900)
        strip = self._composed(res)
        # a whole legend is some thirty times longer than it is tall: shown
        # in two to four rows so its letters are large enough to read (the
        # file saved is one strip)
        rows = int(np.clip(round(math.sqrt(strip.shape[1] / max(strip.shape[0], 1) / 6)), 1, 4))
        if rows > 1:
            step = -(-strip.shape[1] // rows)
            pad = np.full((strip.shape[0], step * rows - strip.shape[1], 3), 40, np.uint8)
            full = np.hstack([strip, pad])
            sep = np.full((max(4, strip.shape[0] // 8), step, 3), 40, np.uint8)
            parts = [full[:, i * step:(i + 1) * step] for i in range(rows)]
            strip = np.vstack([x for part in parts for x in (part, sep)][:-1])
        W = max(strip.shape[1], 1200)
        if strip.shape[1] < W:
            strip = cv2.resize(strip, (W, int(round(strip.shape[0] * W / strip.shape[1]))),
                               interpolation=cv2.INTER_CUBIC)
        th = int(W / 3 * ringed.shape[0] / ringed.shape[1])
        top = cv2.resize(ringed, (W // 3, th), interpolation=cv2.INTER_AREA)
        ox = (W - W // 3) // 2
        self._map = dict(self._map, ox=float(ox), oy=0.0, r=(W // 3) / ringed.shape[1])
        top = np.hstack([np.full((th, ox, 3), 40, np.uint8), top,
                         np.full((th, W - W // 3 - ox, 3), 40, np.uint8)])
        gap = np.full((max(8, strip.shape[0] // 6), W, 3), 40, np.uint8)
        self.preview.set_image(np.vstack([top, gap, strip]))
        self.status.setText(f"Strip {res['strip'].shape[1]} × {res['strip'].shape[0]} px. "
                            + self._size_words())

    def save(self):
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Save the strip", _start(self.win, "_legend.png"),
                                              "PNG (*.png);;TIFF (*.tif)")
        if not path:
            return
        from helio.application.heritage_tools import legend_figure
        res = self.result
        out = self._composed(res)
        if self.with_pic.isChecked():
            o, _w = self.ring()
            out = legend_figure(self._shown(), o, out)
        cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        rec = dict(res["record"], object=object_name(self.project), image=os.path.basename(path))
        reads = self.reads.text().strip()
        if reads:
            rec["legend"] = reads
        with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
        meta, note = dict(self.project.capture_meta), []
        if reads and meta.get("legend") != reads:
            meta["legend"] = reads
            note.append("the legend")
        if not meta.get("diameter_mm"):
            if "diameter_mm" in rec:
                meta["diameter_mm"] = f"{rec['diameter_mm']:.2f}"
            elif "height_mm" in rec:
                meta["diameter_mm"] = f"{rec['height_mm']:.2f} × {rec['width_mm']:.2f}"
            if meta.get("diameter_mm"):
                note.append("the size")
        if note:
            self.project.capture_meta = meta
            self.project.changed.emit()
        self.status.setText(f"Saved {os.path.basename(path)} and its record (.json)."
                            + (f" {' and '.join(note).capitalize()} recorded in Capture details."
                               if note else ""))
        self.win.flash(f"Legend saved: {os.path.basename(path)}.")


# ── virtual impression ───────────────────────────────────────────────────

IMPRESSION_KINDS = (("stamp", "Impression"), ("raised", "Raised letters"), ("as_is", "As is"))


class ImpressionDialog(ToolDialog):
    TITLE = "Virtual impression"
    INTRO = ("What a seal matrix or an intaglio would stamp, or an inscription's cut letters "
             "standing up as on a squeeze, under one raking light.")
    METHOD = ("Impression: the surface mirrored left to right and turned inside out (bumps and "
              "hollows swapped), as a matrix prints into wax. Raised letters: turned inside out only, "
              "so a cut inscription reads the right way round in relief. Lit from the normals by one "
              "light, in the colour of the material. Nothing is changed in the solve.")
    RUN = "Make"

    def build(self):
        from helio.application.heritage_tools import IMPRESSION_COLOURS
        from helio.domain.object_profiles import of
        self.form.addWidget(label("Show", ""))
        self.kind = Segmented(list(IMPRESSION_KINDS))
        self.kind.set_current("raised" if of(self.project.capture_meta).key == "inscription" else "stamp")
        self.form.addWidget(self.kind)
        self.colour = self.combo("Material", list(IMPRESSION_COLOURS),
                                 "Paper squeeze" if self.kind.current() == "raised" else "Plaster")
        self.azi = self.param("azi", "Light from", 0, 359, 315, 5,
                              "Clockwise from the top of the picture: 315 is the upper left.", "°")
        self.alt = self.param("alt", "Light height", 5, 85, 30, 1, "Lower rakes harder.", "°")

    def args(self):
        from helio.application.heritage_tools import IMPRESSION_COLOURS
        D = self.data()
        k = self.kind.current()
        return dict(normals=D.normals_full, mirror=k == "stamp", invert=k in ("stamp", "raised"),
                    azimuth=self.azi.value(), elevation=self.alt.value(),
                    colour=IMPRESSION_COLOURS[self.colour.currentText()])

    def work(self, **kw):
        from helio.application.heritage_tools import virtual_impression
        return virtual_impression(**kw)

    def show_result(self, res):
        self.preview.set_image(res)
        self.status.setText(dict(IMPRESSION_KINDS)[self.kind.current()] + ".")

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save the impression",
                                              _start(self.win, f"_{self.kind.current()}.png"),
                                              "PNG (*.png);;TIFF (*.tif)")
        if path:
            cv2.imwrite(path, cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR))
            self.status.setText(f"Saved {os.path.basename(path)}.")
            self.win.flash(f"Impression saved: {os.path.basename(path)}.")


# ── line drawing ─────────────────────────────────────────────────────────

class LineDrawingDialog(ToolDialog):
    TITLE = "Line drawing"
    INTRO = ("Cut lines and raised edges traced as vector lines, for an archaeological or "
             "epigraphic drawing. Saved as SVG in millimetres once a scale is set.")
    METHOD = ("Grooves are where the height curves up into a cut (a positive Laplacian of the "
              "smoothed height, as Toolpaths finds them); crests where it curves over an edge. Each "
              "is thinned to one pixel and traced into polylines, each layer its own group for "
              "Inkscape or Illustrator. A starting point for a drawing, not a finished one: "
              "expect to delete and add lines.")
    RUN = "Trace"
    SAVE = "Save SVG…"

    def build(self):
        self.detail = self.param("detail", "Detail", 0.03, 0.6, 0.15, 0.01,
                                 "Lower traces fainter lines; higher only the strongest.")
        self.min_len = self.param("min_len", "Shortest line", 3, 200, 12, 1,
                                  "Lines shorter than this are left out.", "px")
        self.crests = QCheckBox("Trace raised edges too")
        self.crests.setChecked(True)
        self.form.addWidget(self.crests)
        self.under = QCheckBox("Show the object under the lines")
        self.under.setChecked(True)
        self.under.toggled.connect(lambda _on: self.show_result(self.result) if self.result else None)
        self.form.addWidget(self.under)

    def args(self):
        D = self.data()
        return dict(depth=D.depth_vis_p, detail=self.detail.value(), min_len=int(self.min_len.value()),
                    crests=self.crests.isChecked())

    def work(self, **kw):
        from helio.application.heritage_tools import line_drawing
        return line_drawing(**kw)

    def show_result(self, res):
        from helio.application.heritage_tools import draw_lines
        bg = None
        if self.under.isChecked():
            try:
                bg = self.data().albedo_rgb_full
            except RuntimeError:
                bg = None
        if bg is not None and bg.shape[:2] != (res["h"], res["w"]):
            bg = None
        self.preview.set_image(draw_lines(res, bg))
        self.status.setText(f"{len(res['grooves'])} groove lines, {len(res['crests'])} crest lines.")

    def save(self):
        from helio.application.heritage_tools import line_drawing_svg
        path, _ = QFileDialog.getSaveFileName(self, "Save the drawing", _start(self.win, "_drawing.svg"),
                                              "SVG (*.svg)")
        if not path:
            return
        bg = ""
        try:
            alb = self.data().albedo_rgb_full
            if alb is not None:
                bg = os.path.splitext(path)[0] + "_photo.jpg"
                cv2.imwrite(bg, cv2.cvtColor(alb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        except RuntimeError:
            bg = ""
        line_drawing_svg(path, self.result, mm_per_px=self.project.scale_mm_per_px or 0.0,
                         title=object_name(self.project), background_png=os.path.basename(bg) if bg else "")
        self.status.setText(f"Saved {os.path.basename(path)}" + (" with the photograph as a hidden layer."
                                                                 if bg else "."))
        self.win.flash(f"Line drawing saved: {os.path.basename(path)}.")


# ── AI ───────────────────────────────────────────────────────────────────

AI_TASKS = (("identify", "Identify"), ("report", "Condition report"), ("caption", "Figure caption"))


class AIDialog(QDialog):
    """Ask a language model about the current view: identification,
    condition report or figure caption (helio.application.ai)."""

    def __init__(self, win, task: str = "identify"):
        from helio.application import ai
        super().__init__(win)
        self.win, self.project = win, win.project
        self.opener = None                       # tests replace the network
        self.data = None
        self.image = None
        if not hasattr(win, "ai_keys"):
            win.ai_keys = {}                     # this session only: never written to disk
        self.setWindowTitle("Ask AI")
        self.setMinimumSize(980, 640)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 16)
        outer.setSpacing(16)
        left = QWidget()
        left.setFixedWidth(330)
        f = QVBoxLayout(left)
        f.setContentsMargins(0, 0, 0, 0)
        f.setSpacing(8)
        f.addWidget(label("Ask AI", "title"))
        self.task = Segmented(list(AI_TASKS), columns=0)
        self.task.set_current(task)
        self.task.chosen.connect(self._task_changed)
        f.addWidget(self.task)
        f.addWidget(label("Provider", ""))
        self.vendor = QComboBox()
        self.vendor.addItems(list(ai.VENDORS))
        self.vendor.setAccessibleName("Provider")
        self.vendor.setCurrentText(str(win.prefs.value("ai_vendor", ai.DEFAULT_VENDOR)))
        f.addWidget(self.vendor)
        self.hint = label("", "caption", wrap=True)
        f.addWidget(self.hint)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.key.setPlaceholderText("API key (kept for this session only)")
        self.key.setAccessibleName("API key")
        f.addWidget(self.key)
        f.addWidget(label("Model", ""))
        mrow = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setAccessibleName("Model")
        mrow.addWidget(self.model, 1)
        self.btn_find = button("Find", "quiet", "List the models loaded on this computer.")
        self.btn_find.clicked.connect(self.find_models)
        mrow.addWidget(self.btn_find)
        f.addLayout(mrow)
        self.send_img = QCheckBox("Send the image")
        self.send_img.setChecked(True)
        f.addWidget(self.send_img)
        self.detail_lbl = label("Detail", "")
        f.addWidget(self.detail_lbl)
        self.detail = Segmented([(d, d) for d in ai.DEPTHS])
        self.detail.set_current("Standard")
        f.addWidget(self.detail)
        sec = Section("How it works", "research_ai", win.prefs, open_=False)
        sec.add(label("The current view (at most 1024 px) and the surface's ISO 25178 statistics, "
                      "levelled, go to the provider with Helio's prompt. Comprehensive "
                      "identification asks for located findings, drawn as numbered boxes. Ollama "
                      "and LM Studio run on this computer: nothing leaves it.", "caption", wrap=True))
        f.addWidget(sec)
        f.addStretch(1)
        self.caveat = label(ai.CAVEAT, "caption", wrap=True)
        f.addWidget(self.caveat)
        self.status = label("", "caption", wrap=True)
        f.addWidget(self.status)
        row = QHBoxLayout()
        close = button("Close", "quiet")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        row.addStretch(1)
        self.btn_ask = button("Ask", "primary")
        self.btn_ask.clicked.connect(self.ask)
        self.btn_copy = button("Copy", "outline")
        self.btn_copy.setEnabled(False)
        self.btn_copy.clicked.connect(self.copy)
        self.btn_save = button("Save…", "outline")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.save)
        for b in (self.btn_ask, self.btn_copy, self.btn_save):
            row.addWidget(b)
        f.addLayout(row)
        outer.addWidget(left)
        right = QVBoxLayout()
        self.preview = Preview("The view sent appears here.")
        self.preview.setMinimumSize(QSize(420, 260))
        right.addWidget(self.preview, 3)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setAccessibleName("The answer")
        self.text.setPlaceholderText("The answer appears here.")
        right.addWidget(self.text, 2)
        outer.addLayout(right, 1)
        self.vendor.currentTextChanged.connect(self._vendor_changed)
        self._vendor_changed(self.vendor.currentText())
        self._task_changed(task)

    def _vendor_changed(self, v: str):
        from helio.application import ai
        spec = ai.VENDORS.get(v, {})
        self.win.prefs.setValue("ai_vendor", v)
        self.hint.setText(spec.get("hint", ""))
        self.key.setVisible(not spec.get("local", False))
        self.key.setText(self.win.ai_keys.get(v) or ai.env_key(v))
        self.btn_find.setVisible(bool(spec.get("local")))
        self.model.clear()
        if spec.get("model"):
            self.model.addItem(spec["model"])
        self.send_img.setEnabled(bool(spec.get("vision")))
        self.send_img.setChecked(bool(spec.get("vision")))

    def _task_changed(self, t: str):
        on = t != "caption"
        self.detail_lbl.setVisible(on)
        self.detail.setVisible(on)

    def find_models(self):
        from helio.application import ai
        v = self.vendor.currentText()
        self.status.setText("Looking…")

        def done(models, err):
            self.model.clear()
            self.model.addItems(models or [])
            self.status.setText(f"{len(models)} models loaded." if models else
                                f"{v.split(' (')[0]} is not answering, or has no model loaded.")
        self.win.run_background(lambda: ai.list_local_models(v, opener=self.opener), done)

    def ask(self):
        from helio.application import ai
        p = self.project
        D = self.win.render_data
        if D is None:
            self.status.setText("Process the images first.")
            return
        v = self.vendor.currentText()
        key = self.key.text().strip()
        if key:
            self.win.ai_keys[v] = key
        try:
            img = current_view(self.win)
        except RuntimeError as e:
            self.status.setText(str(e))
            return
        self.image, self.data = img, None
        self.preview.set_image(img)
        task, depth = self.task.current(), self.detail.current()
        from helio.application.heights import height_px
        stats = ai.surface_stats(height_px(D), p.z_mm_per_unit)
        name = object_name(p)
        s = p.settings
        if task == "caption":
            prompt, tok = ai.caption_prompt(obj_id=name, engine=self.win.view or self.win.engine)
        elif task == "report":
            spec = getattr(D, "spec_var", None)
            prompt, tok = ai.report_prompt(depth, obj_id=name, engine=self.win.view or self.win.engine,
                                           solver=s.resolved_normal_engine(), stats=stats,
                                           spec_mean=float(np.mean(spec)) if spec is not None else 0.0,
                                           n_images=len(p.images))
        else:
            prompt, tok = ai.recognition_prompt(depth, obj_id=name, engine=self.win.view or self.win.engine,
                                                stats=stats, n_images=len(p.images),
                                                solver=s.resolved_normal_engine())
        b64 = ai.image_b64(img) if self.send_img.isChecked() else None
        model = self.model.currentText().strip() or None
        self.btn_ask.setEnabled(False)
        self.status.setText(f"Asking {v.split(' (')[0]}…")
        self.last = dict(task=task, depth=depth, vendor=v, model=model or ai.VENDORS[v]["model"])
        opener = self.opener
        self.win.run_background(lambda: ai.call(v, key, prompt, b64, tok, model=model, opener=opener),
                                self._answered)

    def _answered(self, text, err):
        from helio.application import ai
        self.btn_ask.setEnabled(True)
        if err:
            self.status.setText(err.split(": ", 1)[-1])
            return
        shown = text
        if self.last["task"] == "identify" and self.last["depth"] == "Comprehensive":
            data = ai.parse_json(text)
            if isinstance(data, dict):
                self.data = data
                shown = ai.format_report(data)
                self.preview.set_image(ai.draw_detections(self.image, data))
        self.answer = shown
        self.text.setPlainText(shown)
        self.status.setText(f"Answered by {self.last['vendor']} ({self.last['model']}).")
        self.btn_copy.setEnabled(True)
        self.btn_save.setEnabled(True)

    def full_text(self) -> str:
        from helio.application import ai
        return f"{self.answer}\n\n— {ai.CAVEAT}\n  ({self.last['vendor']}, {self.last['model']})\n"

    def copy(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.full_text())
        self.win.flash("Answer copied.")

    def save(self, path: str | None = None):
        import json
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save the answer",
                                                  _start(self.win, f"_ai_{self.last['task']}.txt"),
                                                  "Text (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.full_text())
        files = [path]
        if self.data is not None:
            base = os.path.splitext(path)[0]
            with open(base + ".json", "w", encoding="utf-8") as fh:
                json.dump(dict(self.data, _caveat=self.full_text().split("— ", 1)[1].strip()), fh, indent=2)
            from helio.application.ai import draw_detections
            cv2.imwrite(base + "_annotated.png", cv2.cvtColor(draw_detections(self.image, self.data),
                                                               cv2.COLOR_RGB2BGR))
            files += [base + ".json", base + "_annotated.png"]
        self.status.setText("Saved " + ", ".join(os.path.basename(f) for f in files) + ".")


RESEARCH = {"unmix": UnmixDialog, "micro": MicroDialog, "toolpaths": ToolpathDialog,
            "zones": ZonesDialog, "register": RegisterDialog, "focus": FocusDialog,
            "recto": RectoVersoDialog, "unwrap": UnwrapDialog, "impression": ImpressionDialog,
            "linedraw": LineDrawingDialog}
