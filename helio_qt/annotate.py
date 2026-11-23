"""Annotate the render: the classic Annotation Layer and Scholar Annotations, together.

Arrows, text, ellipses, rectangles, lines, freehand strokes, polygons,
measurements, dots and numbered points, drawn straight onto Inspect's view
in full-image pixels, each on a named layer (the scholar presets, or your
own).  Undo, delete, save and reload the set as JSON; export the view with
the annotations burned in plus an SVG overlay, or the scholar three-panel
publication set.  The model and the writers are helio.application.annotations.

A measurement is in millimetres once Measure has a scale, else in pixels.
"""
from __future__ import annotations

import math
import os

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QFileDialog, QGraphicsItem,
                               QGraphicsItemGroup, QGraphicsPathItem, QGraphicsSimpleTextItem,
                               QHBoxLayout, QInputDialog, QLineEdit, QMessageBox, QPlainTextEdit,
                               QVBoxLayout, QWidget)

from helio.application import annotations as A
from helio.domain.view_controls import Control
from helio_qt.controls import ParamRow, Section, Segmented
from helio_qt.widgets import button, label

TOOL_NAMES = (("arrow", "Arrow"), ("text", "Text"), ("ellipse", "Ellipse"), ("rect", "Rect"),
              ("line", "Line"), ("freehand", "Freehand"), ("polygon", "Polygon"),
              ("measure", "Measure"), ("dot", "Dot"), ("number", "Number"), ("delete", "Delete"))
TWO_POINT = ("arrow", "ellipse", "rect", "line", "measure")


class Annotator:
    def __init__(self, win, viewport):
        self.win, self.view = win, viewport
        viewport.annotator = self
        self.anns: list[dict] = []
        self.items: list = []
        self.layers = [{"name": A.DEFAULT_LAYER, "color": "#ff0000", "type": "arrow",
                        "visible": True}]
        self.layer = A.DEFAULT_LAYER
        self.tool = "arrow"
        self.colour = "#ff0000"
        self._start = None
        self._temp = None
        self._poly: list = []
        self._free: list = []

    # ── settings ──────────────────────────────────────────────────────
    def mm_per_px(self):
        return self.win.project.scale_mm_per_px or None

    def style(self) -> dict:
        return dict(color=self.colour, text=self.text.text() if hasattr(self, "text") else "Label",
                    fsize=int(self.fsize.value()) if hasattr(self, "fsize") else 18,
                    fill=self.chk_fill.isChecked() if hasattr(self, "chk_fill") else False,
                    opacity=float(self.opacity.value()) / 100.0 if hasattr(self, "opacity") else 0.6,
                    width=int(self.width.value()) if hasattr(self, "width") else 3,
                    layer=self.layer)

    # ── drawing ───────────────────────────────────────────────────────
    def _pen(self, a, dashed=False):
        pen = QPen(QColor(a["color"]), max(1, int(a.get("width", 3))))
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _label(self, text, x, y, colour, size=12):
        t = QGraphicsSimpleTextItem(text)
        f = t.font()
        f.setPointSize(int(size))
        f.setBold(True)
        t.setFont(f)
        t.setBrush(QBrush(QColor(colour)))
        t.setPos(x, y)
        t.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        return t

    def _item(self, a, n: int = 0):
        """A scene item for annotation *a* (image pixels = scene units)."""
        t, pts = a["type"], a["pts"]
        group = QGraphicsItemGroup()
        fill = QColor(a["color"])
        fill.setAlphaF(a.get("opacity", 0.6))
        brush = QBrush(fill) if a.get("fill") else QBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        if t in ("freehand", "polygon") and len(pts) > 1:
            path.addPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
            if t == "polygon":
                path.closeSubpath()
            it = QGraphicsPathItem(path)
            it.setPen(self._pen(a))
            it.setBrush(brush if t == "polygon" else QBrush(Qt.BrushStyle.NoBrush))
            group.addToGroup(it)
        elif t in ("dot", "number"):
            x, y = pts[0]
            r = max(3.0, a.get("width", 3) * 2.0) / max(self.view.zoom(), 1e-6)
            path.addEllipse(QPointF(x, y), r, r)
            it = QGraphicsPathItem(path)
            it.setPen(QPen(Qt.PenStyle.NoPen))
            it.setBrush(QBrush(QColor(a["color"])))
            group.addToGroup(it)
            if t == "number":
                group.addToGroup(self._label(str(a.get("text") or n), x + r, y - r, a["color"],
                                             a.get("fsize", 18) * 0.6))
        elif len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            if t == "text":
                group.addToGroup(self._label(a.get("text", ""), x2, y2, a["color"],
                                             a.get("fsize", 18) * 0.6))
            else:
                if t == "ellipse":
                    path.addEllipse(QRectF(QPointF(min(x1, x2), min(y1, y2)),
                                           QPointF(max(x1, x2), max(y1, y2))))
                elif t == "rect":
                    path.addRect(QRectF(QPointF(min(x1, x2), min(y1, y2)),
                                        QPointF(max(x1, x2), max(y1, y2))))
                else:
                    path.moveTo(x1, y1)
                    path.lineTo(x2, y2)
                    if t == "arrow":
                        ang = math.atan2(y2 - y1, x2 - x1)
                        hl = 16.0 / max(self.view.zoom(), 1e-6)
                        for s in (+0.45, -0.45):
                            path.moveTo(x2, y2)
                            path.lineTo(x2 - hl * math.cos(ang + s), y2 - hl * math.sin(ang + s))
                it = QGraphicsPathItem(path)
                it.setPen(self._pen(a, dashed=t == "measure"))
                it.setBrush(brush if t in ("ellipse", "rect") else QBrush(Qt.BrushStyle.NoBrush))
                group.addToGroup(it)
                if t == "measure":
                    group.addToGroup(self._label(A.length_label(pts[0], pts[-1], self.mm_per_px()),
                                                 (x1 + x2) / 2, (y1 + y2) / 2, a["color"]))
        group.setZValue(20)
        return group

    def redraw(self):
        sc = self.view.scene()
        for it in self.items:
            sc.removeItem(it)
        self.items = []
        visible = {ly["name"] for ly in self.layers if ly.get("visible", True)}
        for n, a in enumerate(self.anns, 1):
            it = self._item(a, n)
            it.setVisible(a.get("layer") in visible)
            sc.addItem(it)
            self.items.append(it)
        self.refresh()

    def _temp_show(self, a):
        sc = self.view.scene()
        if self._temp is not None:
            sc.removeItem(self._temp)
        self._temp = self._item(a)
        sc.addItem(self._temp)

    def _temp_clear(self):
        if self._temp is not None:
            self.view.scene().removeItem(self._temp)
            self._temp = None

    def add(self, a):
        self.anns.append(a)
        it = self._item(a, len(self.anns))
        self.view.scene().addItem(it)
        self.items.append(it)
        self.refresh()

    # ── the mouse ─────────────────────────────────────────────────────
    def press(self, x, y, _e=None):
        t = self.tool
        if t == "delete":
            self.delete_at(x, y)
        elif t in TWO_POINT or t == "text":
            self._start = (x, y)
        elif t == "freehand":
            self._free = [(x, y)]
        elif t == "polygon":
            self._poly.append((x, y))
            if len(self._poly) > 1:
                self._temp_show(A.make("freehand", self._poly, **self.style()))
        elif t in ("dot", "number"):
            self.add(A.make(t, [(x, y)], **dict(self.style(), text="" if t == "number" else "")))

    def move(self, x, y, held):
        if not held:
            return
        st = self.style()
        if self._start is not None and self.tool in TWO_POINT:
            self._temp_show(A.make(self.tool, [self._start, (x, y)], **st))
        elif self.tool == "freehand" and self._free:
            self._free.append((x, y))
            self._temp_show(A.make("freehand", self._free, **st))

    def release(self, x, y):
        st = self.style()
        if self._start is not None and self.tool in TWO_POINT:
            p0 = self._start
            self._start = None
            self._temp_clear()
            if math.hypot(x - p0[0], y - p0[1]) * max(self.view.zoom(), 1e-6) >= 3:
                self.add(A.make(self.tool, [p0, (x, y)], **st))
        elif self._start is not None and self.tool == "text":
            p0 = self._start
            self._start = None
            if st["text"].strip():
                self.add(A.make("text", [p0, (x, y)], **st))
        elif self.tool == "freehand" and len(self._free) > 1:
            pts, self._free = self._free, []
            self._temp_clear()
            self.add(A.make("freehand", pts, **st))

    def double_click(self):
        if self.tool == "polygon" and len(self._poly) >= 3:
            pts, self._poly = self._poly, []
            self._temp_clear()
            self.add(A.make("polygon", pts, **self.style()))

    def cancel(self):
        self._start, self._poly, self._free = None, [], []
        self._temp_clear()

    def delete_at(self, x, y):
        tol = 8.0 / max(self.view.zoom(), 1e-6)
        best, bd = None, tol
        for i, a in enumerate(self.anns):
            for px, py in a["pts"]:
                d = math.hypot(px - x, py - y)
                if d < bd:
                    best, bd = i, d
        if best is not None:
            self.anns.pop(best)
            self.redraw()

    def undo(self):
        if self.anns:
            self.anns.pop()
            self.redraw()

    def clear(self):
        self.anns = []
        self.cancel()
        self.redraw()

    # ── layers ────────────────────────────────────────────────────────
    def add_layer(self, name, colour, kind="arrow"):
        if any(ly["name"] == name for ly in self.layers):
            name = f"{name} {len(self.layers) + 1}"
        self.layers.append({"name": name, "color": colour, "type": kind, "visible": True})
        self.layer = name
        self.colour = colour
        if kind in dict(TOOL_NAMES):
            self.tool = kind
        self.refresh()

    def set_layer_visible(self, name, on):
        for ly in self.layers:
            if ly["name"] == name:
                ly["visible"] = bool(on)
        self.redraw()

    # ── files ─────────────────────────────────────────────────────────
    def save(self, path: str) -> str:
        p = self.win.project
        return A.save_json(path, self.anns, layers=self.layers,
                           notes=self.notes.toPlainText().split("\n") if hasattr(self, "notes") else [],
                           meta={"object": p.name, "scale_mm_per_px": p.scale_mm_per_px})

    def load(self, path: str):
        d = A.load_json(path)
        self.anns = d["annotations"]
        if d["layers"]:
            self.layers = [dict(ly, visible=ly.get("visible", True)) for ly in d["layers"]]
            self.layer = self.layers[0]["name"]
        if hasattr(self, "notes") and d["notes"]:
            self.notes.setPlainText("\n".join(d["notes"]))
        self.redraw()

    def current_image(self, use_proxy: bool = False):
        """The view on screen at full resolution (or at the on-screen render's
        size), as Current View exports it."""
        from helio.application.viewing import screen_image
        w = self.win
        ins = w.workspaces["Inspect"]
        if w.render_data is None:
            raise RuntimeError("Process the images first.")
        d = w.display
        if w.view == ins.DEPTH_MAP:
            import copy
            d = copy.deepcopy(d)
            d.layer = "Depth Map (Greyscale)"
        return screen_image(w.engine, w.render_data, dict(w.engine_settings.get(w.engine, {})), d,
                            use_proxy=use_proxy)

    def export_side(self) -> int:
        return int(A.EXPORT_SIDES.get(self.size_combo.currentText(), 4000)) \
            if hasattr(self, "size_combo") else 4000

    def screen_long(self) -> float:
        """The picture's long side on screen, fitted: what the annotations'
        widths and text sizes are seen against."""
        D = self.win.render_data
        vp = self.view.viewport()
        if D is None or not D.w or not D.h:
            return A.SCREEN_FIT
        k = min(vp.width() / float(D.w), vp.height() / float(D.h))
        return max(400.0, k * max(D.w, D.h))

    def export_render(self, path: str, long_side: int | None = None) -> list[str]:
        """The annotated view at *long_side* px (the panel's size), JPEG by
        default, with its SVG beside it."""
        D = self.win.render_data
        side = self.export_side() if long_side is None else long_side
        target = side or max(D.w, D.h)
        pn = getattr(D, "normals_proxy", None)
        proxy_long = max(pn.shape[:2]) if pn is not None else 0
        img = self.current_image(use_proxy=bool(proxy_long and proxy_long >= target))
        return A.annotated_render(img, self.anns, path, self.mm_per_px(), title=self.win.project.name,
                                  long_side=side, screen_long=self.screen_long(), layers=self.layers,
                                  image_size=(D.w, D.h))

    def export_scholar(self, folder: str) -> list[str]:
        from helio_qt.publish import file_stem
        p = self.win.project
        r = p.result
        return A.scholar_set(folder, file_stem(p.name), r.depth_poisson, r.albedo_rgb_full, self.anns,
                             self.layers, notes=self.notes.toPlainText().strip(),
                             include_albedo=self.chk_albedo.isChecked(), mm_per_px=self.mm_per_px(),
                             obj_id=p.name, solver=r.settings.depth_engine,
                             long_side=self.export_side(), screen_long=self.screen_long())

    # ── the inspector section ─────────────────────────────────────────
    def section(self, store) -> Section:
        sec = Section("Annotate", "annotate", store, open_=False)
        self.btn_on = button("Annotate the view", "", "Draw on the view. Annotations are kept in "
                             "image pixels and exported with the view or as the scholar set.")
        self.btn_on.setCheckable(True)
        self.btn_on.setProperty("tool", "true")
        self.btn_on.toggled.connect(self.set_active)
        sec.add(self.btn_on)
        self.tools = Segmented(list(TOOL_NAMES), {
            "arrow": "Drag from the tail to the head.",
            "text": "Click where the text goes; the words come from the Text box below.",
            "ellipse": "Drag across the ellipse's box.",
            "rect": "Drag from corner to corner.",
            "line": "Drag from one end to the other.",
            "freehand": "Draw by dragging.",
            "dot": "Click to place a dot.",
            "polygon": "Click each corner; double-click to close.",
            "measure": "A length: in mm once Measure has a scale, else in pixels.",
            "delete": "Click an annotation to remove it.",
            "number": "A numbered point; numbers follow the order they were placed."}, columns=3)
        self.tools.set_current(self.tool)
        self.tools.chosen.connect(self._tool_chosen)
        sec.add(self.tools)
        row = QHBoxLayout()
        self.swatches = []
        for hexc, name in A.COLOURS:
            b = button("", "icon", name)
            b.setFixedSize(26, 22)
            b.setAccessibleName(f"{name} colour")
            b.clicked.connect(lambda _c=False, h=hexc: self._colour(h))
            row.addWidget(b)
            self.swatches.append((hexc, b))
        custom = button("Other…", "quiet", "Choose any colour")
        custom.clicked.connect(self._custom_colour)
        row.addWidget(custom)
        row.addStretch(1)
        sec.add_layout(row)
        self.width = ParamRow(Control("w", "Line width", 1, 10, 3, 1, "How thick new arrows, lines and outlines are drawn, in screen pixels."))
        sec.add(self.width)
        # Only what the tool in hand uses: fill for shapes, words for text.
        self.fill_box = QWidget()
        fb = QVBoxLayout(self.fill_box)
        fb.setContentsMargins(0, 0, 0, 0)
        fb.setSpacing(6)
        self.chk_fill = QCheckBox("Fill shapes")
        self.chk_fill.setToolTip("Fill ellipses, rectangles and polygons, at the fill opacity "
                                 "below.")
        fb.addWidget(self.chk_fill)
        self.opacity = ParamRow(Control("op", "Fill opacity", 10, 100, 60, 5, "How solid filled shapes are: lower lets the surface show through.", unit="%"))
        fb.addWidget(self.opacity)
        sec.add(self.fill_box)
        self.text_box = QWidget()
        tb = QVBoxLayout(self.text_box)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.setSpacing(6)
        tb.addWidget(label("Text", "caption"))
        self.text = QLineEdit("Label")
        self.text.setAccessibleName("Text for the Text tool")
        self.text.setToolTip("What the Text tool writes.")
        tb.addWidget(self.text)
        self.fsize = ParamRow(Control("fs", "Font size", 10, 60, 18, 1, "The size of new text and numbers."))
        tb.addWidget(self.fsize)
        sec.add(self.text_box)
        g = QHBoxLayout()
        for txt, fn, tip in (("Undo", self.undo, "Take away the last annotation  ⌘Z"),
                             ("Clear", self._clear_ask, "Remove every annotation, after "
                              "asking")):
            b = button(txt, "outline", tip)
            b.clicked.connect(fn)
            g.addWidget(b)
        g.addStretch(1)
        sec.add_layout(g)

        layers = Section("Layers", "annotate_layers", store, open_=False,
                         tip="Annotations go on named layers (damage, inscription…) that can be "
                             "shown, hidden and exported apart.")
        lrow = QHBoxLayout()
        self.layer_combo = QComboBox()
        self.layer_combo.setAccessibleName("Layer to draw on")
        self.layer_combo.setToolTip("The layer new annotations go on.")
        self.layer_combo.currentTextChanged.connect(self._layer_chosen)
        lrow.addWidget(self.layer_combo, 1)
        self.chk_layer = QCheckBox("Shown")
        self.chk_layer.setToolTip("Show or hide this layer's annotations.")
        self.chk_layer.setChecked(True)
        self.chk_layer.toggled.connect(lambda on: self.set_layer_visible(self.layer, on))
        lrow.addWidget(self.chk_layer)
        layers.add_layout(lrow)
        self.preset = QComboBox()
        self.preset.addItem("Add a layer…")
        self.preset.addItems([n for n, _c, _t in A.LAYER_PRESETS])
        self.preset.setAccessibleName("Add a layer")
        self.preset.setToolTip("A new layer, named for what it marks.")
        self.preset.activated.connect(self._preset)
        layers.add(self.preset)
        sec.add(layers)

        exp = Section("Export", "annotate_export", store, open_=True,
                      tip="The annotated picture (JPEG, PNG or TIFF, with an SVG beside it), or "
                          "the scholar set of panels.")
        srow = QHBoxLayout()
        srow.addWidget(label("Size", "secondary"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(A.EXPORT_SIDES))
        self.size_combo.setCurrentText("4,000 px")
        self.size_combo.setAccessibleName("Export size")
        self.size_combo.setToolTip("The picture's long side. 4,000 px suits print and slides; "
                                   "Full size can be hundreds of megabytes as PNG or TIFF.")
        srow.addWidget(self.size_combo, 1)
        exp.add_layout(srow)
        self.btn_export = button("Export picture…", "", "The view with the annotations drawn in, "
                                 "as they look on screen: JPEG (or PNG or TIFF), and an SVG "
                                 "beside it with the view and each layer.")
        self.btn_export.clicked.connect(self._export)
        exp.add(self.btn_export)
        self.btn_scholar = button("Scholar set…", "outline", "Clean relief, annotated relief, "
                                  "albedo overlay, JSON and SVG, by layer.")
        self.btn_scholar.clicked.connect(self._scholar)
        exp.add(self.btn_scholar)
        self.chk_albedo = QCheckBox("With the colour photograph")
        self.chk_albedo.setToolTip("Add the colour photograph beside the view in the scholar "
                                   "set's figure.")
        self.chk_albedo.setChecked(True)
        exp.add(self.chk_albedo)
        exp.add(label("Notes for the scholar set", "caption"))
        self.notes = QPlainTextEdit()
        self.notes.setFixedHeight(60)
        self.notes.setAccessibleName("Notes for the scholar set")
        exp.add(self.notes)
        g = QHBoxLayout()
        g.addWidget(label("Annotations file", "caption"))
        g.addStretch(1)
        for txt, fn, tip in (("Save…", self._save, "The annotations as a file, to carry on "
                              "later (JSON)."),
                             ("Open…", self._load, "Annotations saved earlier.")):
            b = button(txt, "quiet", tip)
            b.clicked.connect(fn)
            g.addWidget(b)
        exp.add_layout(g)
        sec.add(exp)
        self.status = label("", "caption", wrap=True)
        sec.add(self.status)
        self.sec = sec
        self.refresh()
        return sec

    def refresh(self):
        if not hasattr(self, "status"):
            return
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        self.layer_combo.addItems([ly["name"] for ly in self.layers])
        self.layer_combo.setCurrentText(self.layer)
        self.layer_combo.blockSignals(False)
        ly = next((x for x in self.layers if x["name"] == self.layer), None)
        self.chk_layer.blockSignals(True)
        self.chk_layer.setChecked(bool(ly and ly.get("visible", True)))
        self.chk_layer.blockSignals(False)
        self.tools.set_current(self.tool)
        self._mark_colour()
        self.fill_box.setVisible(self.tool in ("ellipse", "rect", "polygon"))
        self.text_box.setVisible(self.tool in ("text", "number"))
        solved = self.win.project.result is not None
        for b in (self.btn_on, self.btn_export, self.btn_scholar):
            b.setEnabled(solved)
        n = len(self.anns)
        self.status.setText(f"{n} annotation{'s' if n != 1 else ''} on {len(self.layers)} "
                            f"layer{'s' if len(self.layers) != 1 else ''}."
                            + ("" if self.mm_per_px() else " Lengths in pixels until a scale is set."))

    def set_active(self, on: bool):
        self.view.set_tool("annotate" if on else "")
        if not on:
            self.cancel()
        self.btn_on.setText("Stop annotating" if on else "Annotate the view")

    def _tool_chosen(self, key):
        self.cancel()
        self.tool = key
        self.refresh()
        if not self.btn_on.isChecked():
            self.btn_on.setChecked(True)

    def _colour(self, hexc):
        self.colour = hexc
        self._mark_colour()

    def _mark_colour(self):
        """The chosen swatch drawn with a thick ring."""
        for hexc, b in getattr(self, "swatches", []):
            on = hexc.lower() == self.colour.lower()
            b.setStyleSheet(f"background:{hexc}; border:{3 if on else 1}px solid "
                            f"{'#E9B44C' if on else '#555'}; border-radius:4px;")

    def _custom_colour(self):
        c = QColorDialog.getColor(QColor(self.colour), self.win, "Annotation colour")
        if c.isValid():
            self._colour(c.name())

    def _layer_chosen(self, name):
        if name:
            self.layer = name
            ly = next((x for x in self.layers if x["name"] == name), None)
            if ly:
                self.colour = ly["color"]
            self.refresh()

    def _preset(self, idx):
        if idx <= 0:
            return
        name, colour, kind = A.LAYER_PRESETS[idx - 1]
        self.preset.setCurrentIndex(0)
        if name == "Custom":
            text, ok = QInputDialog.getText(self.win, "New layer", "Layer name", text="Layer")
            if not ok or not text.strip():
                return
            name = text.strip()
        self.add_layer(name, colour, kind)

    def _clear_ask(self):
        if self.anns and QMessageBox.question(
                self.win, "Clear annotations", f"Remove all {len(self.anns)} annotations?") != \
                QMessageBox.StandardButton.Yes:
            return
        self.clear()

    def _start_dir(self):
        return str(self.win.prefs.value("export_folder", "")) or self.win.default_export_dir()

    def _save(self):
        from helio_qt.publish import file_stem
        path, _ = QFileDialog.getSaveFileName(
            self.win, "Save the annotations",
            os.path.join(self._start_dir(), f"{file_stem(self.win.project.name)}_annotations.json"),
            "Annotations (*.json)")
        if path:
            self.save(path)
            self.win.flash(f"Saved {len(self.anns)} annotations.")

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(self.win, "Load annotations", self._start_dir(),
                                              "Annotations (*.json)")
        if path:
            try:
                self.load(path)
            except (OSError, ValueError, KeyError) as e:
                QMessageBox.warning(self.win, "Annotations", f"Could not read them: {e}")

    def _export(self):
        from helio_qt.publish import file_stem
        path, flt = QFileDialog.getSaveFileName(
            self.win, "Export the annotated picture",
            os.path.join(self._start_dir(), f"{file_stem(self.win.project.name)}_annotated.jpg"),
            "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tif)")
        if path:
            if not os.path.splitext(path)[1]:
                path += ".png" if "PNG" in flt else ".tif" if "TIFF" in flt else ".jpg"
            files = self.export_render(path)
            mb = os.path.getsize(files[0]) / 1e6
            self.win.flash(f"Saved {os.path.basename(files[0])} ({mb:.1f} MB) and "
                           f"{os.path.basename(files[1])}.")

    def _scholar(self):
        folder = QFileDialog.getExistingDirectory(self.win, "Folder for the scholar set",
                                                  self._start_dir())
        if folder:
            files = self.export_scholar(folder)
            self.win.flash(f"Scholar set: {len(files)} files written.")

