"""The five workspaces of the approved design (UX-002).

Each one shows only its own few controls: a short list on the left, the
work in the middle, settings on the right.  Each names its next step for the
bottom bar, and says why when that step cannot run yet.  What this preview
cannot do yet is shown, disabled, with the reason beside it, so the layout
is the real one and nothing pretends to work.
"""
from __future__ import annotations

import math
import os

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QLabel, QPushButton,
                               QListWidget, QListWidgetItem, QProgressBar, QScrollArea,
                               QStackedWidget, QVBoxLayout, QWidget)

from helio_qt import icons
from helio_qt.publish_catalog import DESTINATIONS, PRESETS

# Ticked when Publish first opens: nothing; you tick what you need, or pick a preset.
DEFAULT_FORMATS: tuple[str, ...] = ()
from helio_qt.state import CLASSIC, NEED_LIGHTS, NEED_SOLVE
from helio_qt.widgets import (Card, FrameList, LightDome, Viewport, button, fit_list, label,
                              panel, rule)

SIDE_W, INSPECTOR_W = 300, 320


def _page(margins=(32, 28, 32, 28), spacing=18) -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    w.setObjectName("Page")
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(*margins)
    lay.setSpacing(spacing)
    return w, lay


def _scroll(w: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(w)
    sa.setFrameShape(QScrollArea.Shape.NoFrame)
    return sa


def _list(items, selectable=True) -> QListWidget:
    lw = QListWidget()
    lw.setFocusPolicy(Qt.FocusPolicy.StrongFocus if selectable else Qt.FocusPolicy.NoFocus)
    for it in items:
        lw.addItem(it if isinstance(it, QListWidgetItem) else QListWidgetItem(it))
    fit_list(lw)
    return lw


def _disable(item: QListWidgetItem, reason: str):
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
    item.setToolTip(reason)


def light_angles(v) -> tuple[float, float]:
    """Azimuth (0 = north, clockwise) and elevation in degrees of a +Y-up light."""
    x, y, z = (float(c) for c in v)
    return (math.degrees(math.atan2(x, y)) % 360.0,
            math.degrees(math.asin(max(-1.0, min(1.0, z)))))


def _analyses_text(keys) -> str:
    """"tool marks and line drawing", for the sheet's summary."""
    from helio.application.sheet_tools import ANALYSES
    names = [ANALYSES[k][0].lower() for k in (keys or []) if k in ANALYSES]
    if not names:
        return ""
    return " and ".join([", ".join(names[:-1]), names[-1]] if len(names) > 1 else names)


class Workspace:
    name = ""

    def __init__(self, win):
        self.win, self.project, self.reg = win, win.project, win.registry
        self.sidebar, self.center, self.inspector = self.build()

    def build(self):
        raise NotImplementedError

    def refresh(self):
        pass

    def recolour(self, t):
        pass

    def next_step(self) -> tuple[str, str]:
        """(button words, action id)."""
        return "", ""

    def status(self) -> tuple[str, str, str]:
        """(status kind, short words, detail)."""
        p = self.project
        if not p.images:
            return "idle", "No images yet", "Drop a folder of captures anywhere in this window"
        return "idle", f"{len(p.images)} frames", self.win.frame_size_text()


# ── Capture ─────────────────────────────────────────────────────────────

# ── Process ─────────────────────────────────────────────────────────────

# ── Inspect ─────────────────────────────────────────────────────────────

# ── Measure ─────────────────────────────────────────────────────────────

# ── Publish ─────────────────────────────────────────────────────────────

class PublishWorkspace(Workspace):
    name = "Publish"
    ROW_MARK = {"waiting": "Waiting", "writing": "Writing…", "done": "Done", "problem": "Problem",
                "stopped": "Not written"}

    def build(self):
        from helio_qt.publish_catalog import all_formats, available
        self.formats = {f.key: f for f in all_formats()}
        self.available = available
        self.selected: list[str] = list(DEFAULT_FORMATS)
        side, side_outer = panel("Sidebar", SIDE_W)
        # It scrolls: with the preset buttons, the two lists ran past a
        # 760 px window and were drawn over each other.
        from helio_qt.controls import VScroll
        side_body = QWidget()
        sl = QVBoxLayout(side_body)
        sl.setContentsMargins(0, 0, 4, 0)
        sl.setSpacing(8)
        side_outer.addWidget(VScroll(side_body), 1)
        sl.addWidget(label("WHAT FOR", "heading"))
        self.dests = _list([f"{d}   {len(fs)}" for d, fs in DESTINATIONS.items()])
        self.dests.setCurrentRow(1)
        self.dests.currentRowChanged.connect(lambda _: self._show_formats())
        sl.addWidget(self.dests)
        sl.addWidget(label("PRESETS", "heading"))
        self.presets = _list(list(self.all_presets()))
        self.presets.itemClicked.connect(self._preset)
        self.presets.currentRowChanged.connect(lambda _r: self._preset_buttons())
        sl.addWidget(self.presets)
        self.btn_save_preset = button("Save as preset…", "quiet",
                                      "Keep the formats ticked now under a name of your own.")
        self.btn_save_preset.clicked.connect(self.save_preset)
        sl.addWidget(self.btn_save_preset)
        self.btn_del_preset = button("Delete preset", "quiet", "Remove the preset of your own "
                                                               "chosen above (the built-in ones stay).")
        self.btn_del_preset.clicked.connect(self.delete_preset)
        sl.addWidget(self.btn_del_preset)
        sl.addStretch(1)

        page, self.pl = _page()
        head = QHBoxLayout()
        self.dest_title = label("", "title")
        head.addWidget(self.dest_title)
        head.addStretch(1)
        self.btn_clear = button("Clear", "quiet", "Untick every format.")
        self.btn_clear.clicked.connect(self.clear_selection)
        head.addWidget(self.btn_clear)
        self.pl.addLayout(head)
        # under the title, where it can wrap: beside it, it pushed Clear off
        self.pl.addWidget(label("Tick one or more · each says what you get", "caption", wrap=True))
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(12)
        self.pl.addWidget(self.grid_host)
        self.pl.addStretch(1)
        qh = QHBoxLayout()
        qh.addWidget(label("Queue", "section"))
        qh.addStretch(1)
        self.pl.addLayout(qh)
        self.queue = QListWidget()
        self.queue.setAccessibleName("Export queue")
        self.queue.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.queue.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pl.addWidget(self.queue)
        self.queue_note = label("Nothing exported yet.", "caption", wrap=True)
        self.pl.addWidget(self.queue_note)
        # Finished: the way on is the next object (SES-005).
        self.btn_next = QPushButton("Start the next capture…")
        self.btn_next.setToolTip("Save this one if it needs it, then choose the next folder of "
                                 "photographs. The solver, views and 3D settings stay as they are.")
        self.btn_next.clicked.connect(lambda: self.reg.trigger("file.next"))
        self.btn_next.setVisible(False)
        nh = QHBoxLayout()
        nh.addWidget(self.btn_next)
        nh.addStretch(1)
        self.pl.addLayout(nh)

        insp, outer = panel("Inspector", INSPECTOR_W)
        # It scrolls, as Capture's does: squeezed into the window's height,
        # its density buttons and sliders were cut in half.
        from helio_qt.controls import VScroll
        body = QWidget()
        il = QVBoxLayout(body)
        il.setContentsMargins(0, 0, 4, 0)
        il.setSpacing(8)
        outer.addWidget(VScroll(body), 1)
        il.addWidget(label("SELECTED", "heading"))
        self.sel_text = label("Nothing selected", "secondary", wrap=True)
        il.addWidget(self.sel_text)
        il.addWidget(rule())
        self._build_settings(il)
        il.addWidget(label("WHERE", "heading"))
        self.where = label("", "caption", wrap=True)
        il.addWidget(self.where)
        il.addWidget(label("You choose the folder when you export. Each format writes its own "
                           "file or folder there.", "caption", wrap=True))
        il.addStretch(1)
        self.cards: list[Card] = []
        self._show_formats()
        self._preset_buttons()
        return side, _scroll(page), insp

    # ── what the exports are made with, here where they are made ──────
    def _build_settings(self, il):
        """The settings that change what is written, beside the formats: the
        mesh (with its size), the scale and the credit.  Each also lives where
        it began (3D settings, Measure, the web package); they are one value."""
        from helio.domain.view_controls import Control
        from helio_qt.controls import ParamRow, Segmented
        self.mesh_box = QWidget()
        mb = QVBoxLayout(self.mesh_box)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.setSpacing(6)
        mb.addWidget(label("3D MODEL", "heading"))
        mb.addWidget(label("Mesh density", "secondary"))
        self.mesh_density = Segmented([("1", "1 : 1"), ("2", "1 : 2"), ("4", "1 : 4"), ("8", "1 : 8")],
                                      {"1": "Every pixel a point: full detail, largest files",
                                       "2": "Every second pixel: a quarter of the size",
                                       "4": "A light mesh", "8": "A sketch"})
        self.mesh_density.chosen.connect(lambda k: self._set_shared("export_step", int(k)))
        mb.addWidget(self.mesh_density)
        self.mesh_rows = {}
        for key, text, lo, hi, step, tip, unit in (
                ("height", "Height exaggeration", 10, 5000, 10,
                 "Scales the relief in the 3D exports. Match the real depth for physical "
                 "accuracy; raise it to see shallow marks.", ""),
                ("edge_trim", "Edge trim", 0.0, 15.0, 0.5,
                 "Percent cut from every edge of the mesh, where the solve is least sure.", "%")):
            r = ParamRow(Control(key, text, lo, hi, float(self.win.shared.get(key, lo)), step, tip,
                                 unit=unit))
            r.changed.connect(lambda k, v, final: final and self._set_shared(k, v))
            mb.addWidget(r)
            self.mesh_rows[key] = r
        self._build_model_options(mb)
        self.mesh_estimate = label("", "caption", wrap=True)
        self.mesh_estimate.setTextFormat(Qt.TextFormat.RichText)
        from PySide6.QtWidgets import QSizePolicy
        self.mesh_estimate.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.mesh_estimate.setMinimumWidth(0)
        mb.addWidget(self.mesh_estimate)
        b = button("More 3D shape settings…", "quiet", "Smoothing, flattening, detail, flips, "
                                                        "the clay colour and the 3D lighting.")
        b.clicked.connect(lambda: self.reg.trigger("view3d.settings"))
        mb.addWidget(b)
        mb.addWidget(rule())
        il.addWidget(self.mesh_box)
        il.addWidget(label("COLOUR IN THE EXPORTS", "heading"))
        self.export_colour = Segmented(
            [("photo", "As photographed"), ("matte", "As Matte Relight")],
            {"photo": "The colour as solved from the photographs, for every export",
             "matte": "The colour as the Matte Relight view draws it, at its light and the Colour "
                      "brightness, in every export's place for the colour: viewers, 3D models, "
                      "IIIF, the sheet and the plate"})
        self.export_colour.chosen.connect(lambda k: self._set_export("export_colour", k))
        il.addWidget(self.export_colour)
        il.addWidget(label("As Matte Relight carries the light in the colour: a viewer that "
                           "relights lights it again, so keep that light high (overhead) for "
                           "those.", "caption", wrap=True))
        il.addWidget(rule())
        self._build_sheet_box(il)
        il.addWidget(label("SCALE", "heading"))
        self.scale_text = label("", "caption", wrap=True)
        il.addWidget(self.scale_text)
        b = button("Set Scale…", "quiet", "Millimetres per pixel: true-size STL, USDZ, GIS and "
                                          "the plate's scale bar.")
        b.clicked.connect(lambda: self.reg.trigger("measure.scale"))
        il.addWidget(b)
        il.addWidget(rule())
        il.addWidget(label("CREDIT", "heading"))
        self.credit_text = label("", "caption", wrap=True)
        il.addWidget(self.credit_text)
        b = button("Edit credit and licence…", "quiet", "Who made it, where, and its licence: on "
                                                       "the publication plate and the web package.")
        b.clicked.connect(self.edit_credit)
        il.addWidget(b)
        il.addWidget(rule())

    def _build_model_options(self, mb):
        """What the model files are made for: textures, solid and colour."""
        from PySide6.QtWidgets import QCheckBox
        from helio.application.exports import MESH_TARGETS
        from helio_qt.controls import Segmented
        mb.addWidget(label("Made for", "secondary"))
        self.mesh_target = QComboBox()
        self.mesh_target.setAccessibleName("What the 3D files are made for")
        self.mesh_target.setToolTip("Sets the density, texture size and file options together "
                                    "for where the model is going. Change any of them and it "
                                    "reads Custom.")
        self.mesh_target.addItem("Custom", "")
        for key, (name, _v) in MESH_TARGETS.items():
            self.mesh_target.addItem(name, key)
        self.mesh_target.activated.connect(self._apply_target)
        mb.addWidget(self.mesh_target)
        mb.addWidget(label("Texture size", "secondary"))
        self.tex_size = Segmented(
            [("mesh", "Mesh"), ("2048", "2K"), ("4096", "4K"), ("8192", "8K"), ("full", "Full")],
            {"mesh": "One texel a point: the texture is as coarse as the mesh",
             "2048": "2048 px on the longer side: phones, quick sharing",
             "4096": "4096 px: the web and Sketchfab", "8192": "8192 px: close study on a desktop",
             "full": "Every pixel of the photographs: largest files"})
        self.tex_size.setToolTip("The colour and normal pictures wrapped on the model (GLB, OBJ, "
                                 "Sketchfab), taken from the full photographs whatever the mesh "
                                 "density, so a light mesh can still look sharp.")
        self.tex_size.chosen.connect(lambda k: self._set_export("tex_size", k))
        mb.addWidget(self.tex_size)
        mb.addWidget(label("Texture files", "secondary"))
        self.tex_format = Segmented([("0", "PNG"), ("1", "JPEG")],
                                    {"0": "Lossless: archives and further work",
                                     "1": "About a quarter of the size: the web. The normal map "
                                          "stays PNG, which JPEG would spoil"})
        self.tex_format.chosen.connect(lambda k: self._set_export("tex_jpeg", int(k)))
        mb.addWidget(self.tex_format)
        self.chk_watertight = QCheckBox("Closed solid with a base")
        self.chk_watertight.setToolTip("Walls and a flat base under the relief: what printing and "
                                       "volume measurement need. Off: the surface alone, lighter "
                                       "and simpler to view on the web.")
        self.chk_watertight.toggled.connect(lambda on: self._set_export("mesh_watertight", int(on)))
        mb.addWidget(self.chk_watertight)
        self.chk_vcol = QCheckBox("Colour on the model")
        self.chk_vcol.setToolTip("The photograph's colour on every point (PLY) and as the OBJ's "
                                 "texture. Off for 3D printing and plain shape.")
        self.chk_vcol.toggled.connect(lambda on: self._set_export("mesh_vertex_colours", int(on)))
        mb.addWidget(self.chk_vcol)

    def _set_export(self, key, value):
        """A model choice Publish keeps for itself (never sent to the 3D viewer)."""
        self.win.shared[key] = value
        self.refresh()

    def _apply_target(self, i):
        from helio.application.exports import MESH_TARGETS
        key = self.mesh_target.itemData(i)
        if not key:
            return
        for k, v in MESH_TARGETS[key][1].items():
            if k == "export_step":
                self.win.model3d.set(k, int(v))
            else:
                self.win.shared[k] = v
        self.refresh()

    def _refresh_model_options(self):
        from helio.application.exports import EXPORT_DEFAULTS, mesh_target
        sh = self.win.shared
        for k, v in EXPORT_DEFAULTS.items():
            sh.setdefault(k, v)
        self.export_colour.set_current(str(sh.get("export_colour") or "photo"))
        self.tex_size.set_current(str(sh["tex_size"]))
        self.tex_format.set_current(str(int(sh["tex_jpeg"])))
        for box, key in ((self.chk_watertight, "mesh_watertight"), (self.chk_vcol, "mesh_vertex_colours")):
            box.blockSignals(True)
            box.setChecked(bool(sh[key]))
            box.blockSignals(False)
        i = self.mesh_target.findData(mesh_target(sh))
        self.mesh_target.setCurrentIndex(max(0, i))

    # ── the publication sheet's choices ────────────────────────────────
    def _build_sheet_box(self, il):
        from PySide6.QtWidgets import QCheckBox
        self.sheet_box = QWidget()
        sb = QVBoxLayout(self.sheet_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(6)
        sb.addWidget(label("PUBLICATION SHEET", "heading"))
        self.sheet_text = label("", "caption", wrap=True)
        sb.addWidget(self.sheet_text)
        b = button("Choose the views…", "quiet", "The large view and the views beside it.")
        b.clicked.connect(self.choose_sheet_views)
        sb.addWidget(b)
        b = button("Choose the detail views…", "quiet", "The views each boxed detail is shown "
                                                         "in, side by side.")
        b.clicked.connect(lambda: self.choose_sheet_views(detail=True))
        sb.addWidget(b)
        row = QHBoxLayout()
        b = button("Add the 3D relief", "quiet", "The 3D relief as it is now, as a picture on "
                                                 "the sheet. Set the view first.")
        b.clicked.connect(lambda: self.win.add_sheet_shot("relief"))
        row.addWidget(b)
        b = button("Add high-detail 3D", "quiet", "The high-detail 3D window as it is now.")
        b.clicked.connect(lambda: self.win.add_sheet_shot("hd"))
        row.addWidget(b)
        sb.addLayout(row)
        row = QHBoxLayout()
        b = button("Add pictures…", "quiet", "Photographs or screenshots from files, shown with "
                                             "the 3D pictures.")
        b.clicked.connect(lambda: self.win.add_sheet_pictures())
        row.addWidget(b)
        b = button("Add this section", "quiet", "Keep the section line drawn on the 3D relief "
                                                "(or the high-detail window, when it is open), "
                                                "then draw another. Up to six.")
        b.clicked.connect(lambda: self.win.add_sheet_section(
            "hd" if getattr(self.win, "hd3d", None) is not None and self.win.hd3d.isVisible()
            else "relief"))
        row.addWidget(b)
        sb.addLayout(row)
        sb.addWidget(label("On the sheet", "secondary"))
        from PySide6.QtWidgets import QListWidget
        self.sheet_list = QListWidget()
        self.sheet_list.setAccessibleName("On the sheet")
        self.sheet_list.setToolTip("Everything added to the sheet, in its order. Select one to show "
                                   "it; double-click a picture to change its caption.")
        self.sheet_list.setMinimumHeight(150)
        self.sheet_list.setIconSize(QSize(48, 32))
        self.sheet_list.setWordWrap(True)                  # the ⚠ notes are at the end of a row
        self.sheet_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sheet_list.currentRowChanged.connect(lambda _r: self._sheet_row_chosen())
        self.sheet_list.itemDoubleClicked.connect(lambda _it: self._rename_sheet_row())
        sb.addWidget(self.sheet_list)
        row = QHBoxLayout()
        self.btn_sheet_show = button("Show", "quiet", "Show the section on the picture in Inspect.")
        self.btn_sheet_show.clicked.connect(self._show_sheet_row)
        self.btn_sheet_up = button("Up", "quiet", "Earlier on the sheet (a section takes the letters "
                                                  "of its new place).")
        self.btn_sheet_up.clicked.connect(lambda: self._move_sheet_row(-1))
        self.btn_sheet_down = button("Down", "quiet", "Later on the sheet.")
        self.btn_sheet_down.clicked.connect(lambda: self._move_sheet_row(1))
        self.btn_sheet_remove = button("Remove", "quiet", "Take it off the sheet.")
        self.btn_sheet_remove.clicked.connect(self._remove_sheet_row)
        self.btn_sheet_keep = button("Keep", "quiet", "Add the line drawn now to the sheet.")
        self.btn_sheet_keep.clicked.connect(self._keep_live_section)
        for b_ in (self.btn_sheet_show, self.btn_sheet_up, self.btn_sheet_down, self.btn_sheet_remove,
                   self.btn_sheet_keep):
            row.addWidget(b_)
        sb.addLayout(row)
        self.btn_sheet_clear = button("Clear what was added", "quiet",
                                      "Take the added pictures and sections off the sheet.")
        self.btn_sheet_clear.clicked.connect(self._clear_sheet_shots)
        sb.addWidget(self.btn_sheet_clear)
        self.chk_notes = QCheckBox("Your annotations on the large view")
        self.chk_notes.setToolTip("Everything drawn with Annotate (arrows, circles, lines, numbered points, "
                                  "text) on the layers that are showing. Each one with a caption gets a tag "
                                  "n1, n2 … and is listed under the picture. Boxes stay the boxed details.")
        self.chk_notes.toggled.connect(lambda on: self.win.set_sheet_prefs(notes_on=bool(on)))
        sb.addWidget(self.chk_notes)
        self.chk_compass = QCheckBox("Second page: lit from eight directions")
        self.chk_compass.setToolTip("The surface relit from the eight compass points, round its "
                                    "normal map.")
        self.chk_compass.toggled.connect(lambda on: self.win.set_sheet_prefs(compass=bool(on)))
        sb.addWidget(self.chk_compass)
        self.chk_clay = QCheckBox("Clay 3D at an angle")
        self.chk_clay.setToolTip("The 3D relief in plain clay, tilted, lit low from the side. "
                                 "Needs the 3D relief opened once in Inspect.")
        self.chk_clay.toggled.connect(lambda on: self.win.set_sheet_prefs(clay=bool(on)))
        sb.addWidget(self.chk_clay)
        self.chk_score = QCheckBox("The overall score, top right")
        self.chk_score.setToolTip("The capture's score out of 100 beside the title.")
        self.chk_score.toggled.connect(lambda on: self.win.set_sheet_prefs(hide_score=not on))
        sb.addWidget(self.chk_score)
        self.chk_capture_box = QCheckBox("The capture box (camera, lights, scale)")
        self.chk_capture_box.setToolTip("The box of capture details beside the object's.")
        self.chk_capture_box.toggled.connect(
            lambda on: self.win.set_sheet_prefs(hide_capture=not on))
        sb.addWidget(self.chk_capture_box)
        self.chk_png = QCheckBox("Also save PNG (lossless)")
        self.chk_png.setToolTip("Each page as a PNG too, beside the JPEG and the PDF.")
        self.chk_png.toggled.connect(lambda on: self.win.set_sheet_prefs(png=bool(on)))
        sb.addWidget(self.chk_png)
        from helio_qt.controls import Segmented
        row = QHBoxLayout()
        row.addWidget(label("Paper", "secondary"))
        self.sheet_paper = Segmented([("A3", "A3"), ("A4", "A4"), ("Letter", "Letter")],
                                     {"A3": "297 × 420 mm: 3508 × 4961 px at 300 dpi.",
                                      "A4": "210 × 297 mm: 2480 × 3508 px at 300 dpi.",
                                      "Letter": "US Letter, 8.5 × 11 in: the page fitted and "
                                                "centred, 2550 × 3300 px at 300 dpi."})
        self.sheet_paper.chosen.connect(lambda k: self.win.set_sheet_prefs(paper=k))
        row.addWidget(self.sheet_paper)
        row.addStretch(1)
        sb.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(label("For", "secondary"))
        self.sheet_quality = Segmented([("print", "Print"), ("screen", "Screen and email")],
                                       {"print": "300 dpi: for printing and publication.",
                                        "screen": "150 dpi: a quarter of the pixels, a much "
                                                  "smaller PDF to email or show on screen."})
        self.sheet_quality.chosen.connect(lambda k: self.win.set_sheet_prefs(quality=k))
        row.addWidget(self.sheet_quality)
        row.addStretch(1)
        sb.addLayout(row)
        row = QHBoxLayout()
        b = button("Title and subtitle…", "quiet", "The sheet's own title and line under it; "
                                                   "blank uses the capture details.")
        b.clicked.connect(self.edit_sheet_title)
        row.addWidget(b)
        b = button("Capture details…", "quiet", "The object's title, institution, operator, "
                                                "camera and notes: the sheet's boxes.")
        b.clicked.connect(lambda: self.reg.trigger("capture.details"))
        row.addWidget(b)
        sb.addLayout(row)
        sb.addWidget(label("Notes: what you draw with Annotate goes on the large view, and its caption is "
                           "listed under it. Details: boxes drawn with Annotate › Rect, shown close up. "
                           "Measured lines come from Measure › Distance, sections from the 3D relief "
                           "(Add to sheet, with its copies).", "caption", wrap=True))
        sb.addWidget(rule())
        il.addWidget(self.sheet_box)

    # the contents list: what is on the sheet, like ImageJ's ROI Manager
    def _sheet_rows(self) -> list[tuple[str, int]]:
        """What each row of the list is: ('section', i), ('picture', i) or ('live', -1)."""
        return list(getattr(self, "_sheet_row_keys", []))

    def _fill_sheet_list(self):
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QListWidgetItem
        from helio.application.sheet_contents import duplicates
        import numpy as np
        from helio_qt.widgets import pixmap_from_array
        w = self.win
        here = w.selected_result()[1] if w.project.result is not None else None
        secs = list(getattr(w, "sheet_sections", []))
        shots = list(getattr(w, "sheet_shots", []))
        cur = self.sheet_list.currentRow()
        self.sheet_list.blockSignals(True)
        self.sheet_list.clear()
        keys = []
        twins = duplicates([p_ for p_, _o in secs])
        for i, (p_, of) in enumerate(secs):
            a, b = w.section_letters(p_)
            n = len(p_.get("series") or [])
            length = p_.get("length")
            words = [f"{a}–{b}  section"]
            if length:
                words.append(f"{float(length):.1f} {p_.get('length_unit', 'mm')}")
            if n > 1:
                words.append(f"{n - 1} parallel copies")
            if not (of is here or (of is None and here is w.render_data)):
                words.append("on another part: not on this sheet")
            if i in twins:
                ta, tb = w.section_letters(secs[twins[i]][0])
                words.append(f"⚠ close to {ta}–{tb}")
            it = QListWidgetItem(", ".join(words))
            it.setToolTip(it.text())
            self.sheet_list.addItem(it)
            keys.append(("section", i))
        live = getattr(w.workspaces["Inspect"], "section_prof", None)
        if live and not any((tuple(p_["a_px"]), tuple(p_["b_px"])) == (tuple(live["a_px"]), tuple(live["b_px"]))
                            for p_, _o in secs):
            a, b = w.section_letters(live)
            txt = (f"{a}–{b}  drawn now: on the sheet only while none is kept; Keep it to be sure"
                   if not secs else f"{a}–{b}  drawn now, not on the sheet: Keep to add it")
            it = QListWidgetItem(txt)
            it.setForeground(self.sheet_list.palette().placeholderText())
            self.sheet_list.addItem(it)
            keys.append(("live", -1))
        kinds = {"3d": "3D", "file": "picture", "tool": "from a tool"}
        for i, (cap, im, kind) in enumerate(shots):
            it = QListWidgetItem(f"{cap}  ({kinds.get(kind, kind)})")
            try:
                it.setIcon(QIcon(pixmap_from_array(np.ascontiguousarray(im)).scaled(
                    96, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)))
            except Exception:
                pass
            self.sheet_list.addItem(it)
            keys.append(("picture", i))
        self._sheet_row_keys = keys
        if not keys:
            it = QListWidgetItem("Nothing added yet: sections from the 3D relief, 3D pictures, "
                                 "pictures from files and the tools' results go here.")
            it.setFlags(Qt.ItemFlag.NoItemFlags)
            self.sheet_list.addItem(it)
        self.sheet_list.setCurrentRow(min(cur, len(keys) - 1) if keys else -1)
        self.sheet_list.blockSignals(False)
        self._sheet_row_chosen()

    def _sheet_key(self):
        r = self.sheet_list.currentRow()
        keys = self._sheet_rows()
        return keys[r] if 0 <= r < len(keys) else (None, -1)

    def _sheet_row_chosen(self):
        kind, i = self._sheet_key()
        n_kind = len(getattr(self.win, "sheet_sections" if kind == "section" else "sheet_shots", []))
        self.btn_sheet_show.setEnabled(kind in ("section", "live"))
        self.btn_sheet_up.setEnabled(kind in ("section", "picture") and i > 0)
        self.btn_sheet_down.setEnabled(kind in ("section", "picture") and i < n_kind - 1)
        self.btn_sheet_remove.setEnabled(kind in ("section", "picture"))
        self.btn_sheet_keep.setEnabled(kind == "live")
        ins = self.win.workspaces["Inspect"]
        if hasattr(ins, "highlight_kept_section"):
            ins.highlight_kept_section(i if kind == "section" else None)

    def _show_sheet_row(self):
        kind, i = self._sheet_key()
        ins = self.win.workspaces["Inspect"]
        if kind == "section" and hasattr(ins, "highlight_kept_section"):
            ins.highlight_kept_section(i)
            ins.btn_section.setChecked(True)           # the kept lines show while it is on
        self.win.go_workspace("Inspect")

    def _move_sheet_row(self, by: int):
        kind, i = self._sheet_key()
        if kind not in ("section", "picture"):
            return
        j = self.win.move_sheet_item(kind, i, by)
        self.refresh()
        self.sheet_list.setCurrentRow(self._sheet_rows().index((kind, j)))

    def _remove_sheet_row(self):
        kind, i = self._sheet_key()
        if kind in ("section", "picture"):
            self.win.remove_sheet_item(kind, i)
            self.refresh()

    def _rename_sheet_row(self):
        from PySide6.QtWidgets import QInputDialog
        kind, i = self._sheet_key()
        if kind != "picture":
            return
        cap = self.win.sheet_shots[i][0]
        text, ok = QInputDialog.getText(self.win, "Caption", "The caption under it on the sheet:", text=cap)
        if ok:
            self.win.rename_sheet_picture(i, text)
            self.refresh()

    def _keep_live_section(self):
        ins = self.win.workspaces["Inspect"]
        if self.win.add_sheet_section("relief"):
            self.refresh()
        elif getattr(ins, "section_prof", None) is None:
            self.win.flash("Draw a section line on the 3D relief first.")

    def _clear_sheet_shots(self):
        self.win.sheet_shots = []
        self.win.sheet_sections = []
        self.win.name_sections()
        self.refresh()

    def choose_sheet_views(self, detail: bool = False):
        """Tick the views the sheet shows (or shows each detail in); the large view too."""
        from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QListWidget,
                                       QListWidgetItem)
        from helio.application import pub_sheet as PS
        from helio.domain.registry import RENDER_ENGINES
        prefs = self.win.sheet_prefs()
        key, limit = ("detail_views", 5) if detail else ("views", 20)
        current = list(prefs.get(key) or (PS.DEFAULT_DETAIL_VIEWS if detail else PS.DEFAULT_VIEWS))
        d = QDialog(self.win)
        d.setWindowTitle("Detail views" if detail else "Views on the sheet")
        lay = QVBoxLayout(d)
        lay.addWidget(label(f"Tick up to {limit}; they are shown in this order."
                            if not detail else f"Tick up to {limit}: each detail is shown in "
                            "each of these, side by side.", "secondary", wrap=True))
        lst = QListWidget()
        names = [e for e in RENDER_ENGINES]
        for e in current + [e for e in names if e not in current]:
            it = QListWidgetItem(e)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if e in current else Qt.CheckState.Unchecked)
            lst.addItem(it)
        lst.setMinimumHeight(300 if not detail else 420)
        lay.addWidget(lst)
        hero = None
        found = None
        if not detail:
            from helio.application.sheet_tools import ANALYSES
            lay.addWidget(label("From the analyses", "secondary"))
            lay.addWidget(label("Worked out when the sheet is made, at their usual settings, and "
                                "shown under From the tools. A result made in its Research tool "
                                "with other settings can be added there instead.", "caption", wrap=True))
            found = QListWidget()
            found.setAccessibleName("Analyses on the sheet")
            on = set(prefs.get("analyses") or [])
            for k, (t, blurb) in ANALYSES.items():
                it = QListWidgetItem(t)
                it.setData(Qt.ItemDataRole.UserRole, k)
                it.setToolTip(blurb)
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked if k in on else Qt.CheckState.Unchecked)
                found.addItem(it)
            found.setFixedHeight(found.sizeHintForRow(0) * len(ANALYSES) + 8)
            lay.addWidget(found)
            lay.addWidget(label("The large view", "secondary"))
            hero = QComboBox()
            hero.addItems(names)
            hero.setCurrentText(prefs.get("hero") or PS.HERO_VIEW)
            lay.addWidget(hero)
        note = label("", "caption")
        lay.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        lay.addWidget(bb)

        def ticked():
            return [lst.item(i).text() for i in range(lst.count())
                    if lst.item(i).checkState() == Qt.CheckState.Checked]

        def check(_it=None):
            n = len(ticked())
            note.setText(f"{n} ticked" + (f": {n - limit} too many" if n > limit else ""))
            bb.button(QDialogButtonBox.StandardButton.Save).setEnabled(0 < n <= limit)
        lst.itemChanged.connect(check)
        check()
        self._sheet_dialog = d
        if d.exec():
            kw = {key: ticked()}
            if hero is not None:
                kw["hero"] = hero.currentText()
            if found is not None:
                kw["analyses"] = [found.item(i).data(Qt.ItemDataRole.UserRole)
                                  for i in range(found.count())
                                  if found.item(i).checkState() == Qt.CheckState.Checked]
            self.win.set_sheet_prefs(**kw)
            self.refresh()

    def _refresh_sheet(self):
        on = "pub_sheet" in self.chosen()
        self.sheet_box.setVisible(on)
        if not on:
            return
        from helio.application import pub_sheet as PS
        w = self.win
        prefs = w.sheet_prefs()
        opt = w._sheet_options() if w.project.result is not None else {}
        n_views = len(prefs.get("views") or PS.DEFAULT_VIEWS)
        n_dv = len(prefs.get("detail_views") or PS.DEFAULT_DETAIL_VIEWS)
        shots = len(getattr(w, "sheet_shots", []))
        ins = w.workspaces["Inspect"]
        kept = len(opt.get("sections", []))
        live = getattr(ins, "section_prof", None)
        n_sec = kept + (1 if live and not any(
            (tuple(p["a_px"]), tuple(p["b_px"])) == (tuple(live["a_px"]), tuple(live["b_px"]))
            for p in opt.get("sections", [])) else 0)
        bits = [f"{n_views} views beside {prefs.get('hero') or PS.HERO_VIEW}",
                f"{len(opt.get('details', []))} boxed details × {n_dv} views",
                f"{len(opt.get('measures', []))} measured lines",
                (f"{len(opt.get('notes', []))} notes" if prefs.get("notes_on", True) and opt.get("notes") else ""),
                f"{n_sec} section{'s' if n_sec != 1 else ''}" if n_sec else "no section",
                "the scale" if w.project.scale_mm_per_px else "no scale yet",
                f"{shots} added picture{'s' if shots != 1 else ''}",
                _analyses_text(prefs.get("analyses"))]
        self.sheet_text.setText(" · ".join(b_ for b_ in bits if b_))
        self.btn_sheet_clear.setVisible(shots > 0 or kept > 0)
        self._fill_sheet_list()
        self.chk_notes.blockSignals(True)
        self.chk_notes.setChecked(bool(prefs.get("notes_on", True)))
        self.chk_notes.blockSignals(False)
        self.chk_compass.blockSignals(True)
        self.chk_compass.setChecked(bool(prefs.get("compass")))
        self.chk_compass.blockSignals(False)
        self.chk_clay.blockSignals(True)
        self.chk_clay.setChecked(bool(prefs.get("clay", True)))
        self.chk_clay.blockSignals(False)
        for chk, on in ((self.chk_score, not prefs.get("hide_score")),
                        (self.chk_capture_box, not prefs.get("hide_capture")),
                        (self.chk_png, bool(prefs.get("png")))):
            chk.blockSignals(True)
            chk.setChecked(on)
            chk.blockSignals(False)
        self.sheet_paper.set_current(prefs.get("paper") if prefs.get("paper") in ("A4", "Letter") else "A3")
        self.sheet_quality.set_current("screen" if prefs.get("quality") == "screen" else "print")

    def edit_sheet_title(self):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit
        prefs = self.win.sheet_prefs()
        meta = dict(self.win.project.capture_meta)
        d = QDialog(self.win)
        d.setWindowTitle("The sheet's title")
        f = QFormLayout(d)
        title = QLineEdit(str(prefs.get("title") or ""))
        title.setPlaceholderText(meta.get("title") or self.win.project.name)
        sub = QLineEdit(str(prefs.get("subtitle") or ""))
        sub.setPlaceholderText(" · ".join(x for x in (meta.get("collection", ""),
                                                       meta.get("institution", "")) if x)
                               or "A line under the title")
        f.addRow("Title", title)
        f.addRow("Subtitle", sub)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        self._title_dialog = d
        if d.exec():
            self.win.set_sheet_prefs(title=title.text().strip(), subtitle=sub.text().strip())
            self.refresh()

    def _set_shared(self, key, value):
        self.win.model3d.set(key, value)          # the one value, as the 3D settings set it
        self.refresh()

    def edit_credit(self):
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLineEdit
        pr = self.win.prefs
        d = QDialog(self.win)
        d.setWindowTitle("Credit and licence")
        f = QFormLayout(d)
        fields = {}
        for key, text, default in (("rti_web/creator", "Created by", ""),
                                   ("rti_web/institution", "Institution", ""),
                                   ("publish/licence", "Licence", "")):
            e = QLineEdit(str(pr.value(key, default)))
            f.addRow(text, e)
            fields[key] = e
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        f.addRow(bb)
        if d.exec():
            for key, e in fields.items():
                pr.setValue(key, e.text().strip())
            self.refresh()

    def _refresh_settings(self):
        from helio.application.exports import MESH_BYTES, mesh_estimate
        p, sh = self.project, self.win.shared
        mesh_keys = [k for k in self.chosen() if k in MESH_BYTES]
        self.mesh_box.setVisible(bool(mesh_keys))
        self._refresh_sheet()
        step = int(sh.get("export_step", 1) or 1)
        self.mesh_density.set_current(str(step) if step in (1, 2, 4, 8) else "")
        for k, r in self.mesh_rows.items():
            r.set_value(float(sh.get(k, r.c.default)))
        self._refresh_model_options()
        r, _D = self.win.selected_result()
        if mesh_keys and r is not None:
            from helio.application.exports import texture_bytes
            e = mesh_estimate(r.w, r.h, step, float(sh.get("edge_trim", 0.0)), keys=mesh_keys)
            tex = texture_bytes(r.w, r.h, float(sh.get("edge_trim", 0.0)), str(sh.get("tex_size")),
                                bool(sh.get("tex_jpeg")))
            for k in ("glb_mesh", "obj", "sketchfab"):           # the formats that carry the textures
                if k in e["bytes"]:
                    e["bytes"][k] += tex
            names = {k: self.formats[k].name for k in mesh_keys}

            def size(b):
                return f"{b / 1e9:.1f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"
            rows = "".join(f"<br>{names[k]}: about {size(b)}" for k, b in e["bytes"].items())
            big = max(e["bytes"].values(), default=0)
            warn = (f"<br><span style='color:#E0675B'>Large: {size(big)}. A density of 1 : 2 "
                    f"makes it about a quarter.</span>" if big > 2e9 and step == 1 else "")
            cw, ch = e["size"]
            self.mesh_estimate.setText(
                f"{cw:,} × {ch:,} px → {e['points'] / 1e6:.1f} million points, "
                f"{e['triangles'] / 1e6:.1f} million triangles{rows}{warn}")
        else:
            self.mesh_estimate.setText("")
        mm = p.scale_mm_per_px
        self.scale_text.setText(
            (f"{mm * 1000:.2f} µm per pixel ({25.4 / mm:.0f} PPI)" if mm else "Not set: models "
             "and GIS layers come out in pixels, not millimetres")
            + ("; heights in mm" if p.z_mm_per_unit else "; heights relative" if mm else ""))
        pr = self.win.prefs
        who = ", ".join(v for v in (str(pr.value("rti_web/creator", "")),
                                    str(pr.value("rti_web/institution", ""))) if v)
        lic = str(pr.value("publish/licence", "")) or "no licence stated"
        self.credit_text.setText(f"{who or 'No creator set'} · {lic}")

    def chosen(self) -> list[str]:
        """The ticked formats this app can write, in catalogue order."""
        return [k for k in self.formats if k in self.selected and self.available(self.formats[k])]

    def _show_formats(self):
        for c in self.cards:
            c.setParent(None)
            c.deleteLater()
        self.cards = []
        dest = list(DESTINATIONS)[max(0, self.dests.currentRow())]
        self.dest_title.setText(dest)
        for i, f in enumerate(DESTINATIONS[dest]):
            ok = self.available(f)
            c = Card(f.name, f.what, "" if ok else "classic")
            c.key = f.key
            if not ok:
                c.setCursor(Qt.CursorShape.ArrowCursor)
                c.setToolTip(CLASSIC)
                c.title.setProperty("role", "secondary")
            c.set_selected(f.key in self.selected)
            c.clicked.connect(lambda k=f.key: self._toggle(k))
            self.grid.addWidget(c, i // 3, i % 3)
            self.grid.setColumnStretch(i % 3, 1)
            self.cards.append(c)
        if hasattr(self, "t"):
            self.recolour(self.t)
        self.refresh()

    def _toggle(self, key):
        if not self.available(self.formats[key]) or self.project.busy:
            return
        if key in self.selected:
            self.selected.remove(key)
        else:
            self.selected.append(key)
        self._after_choice()

    # ── presets: the four built in, and your own ──────────────────────
    def own_presets(self) -> dict:
        import json
        try:
            d = json.loads(str(self.win.prefs.value("publish/presets", "{}")) or "{}")
        except ValueError:
            d = {}
        return {str(k): [str(x) for x in v] for k, v in d.items() if isinstance(v, list)}

    def all_presets(self) -> dict:
        """The object profile's preset first (helio.domain.object_profiles), then Helio's and yours."""
        from helio.domain.object_profiles import of, preset_name
        p = of(self.win.project.capture_meta)
        own = {preset_name(p): list(p.exports)} if p.exports else {}
        return {**own, **PRESETS, **self.own_presets()}

    def _reload_presets(self, current: str = ""):
        self.presets.blockSignals(True)
        self.presets.clear()
        self.presets.addItems(list(self.all_presets()))
        self.presets.blockSignals(False)
        from helio_qt.widgets import fit_list
        fit_list(self.presets)
        if current:
            items = self.presets.findItems(current, Qt.MatchFlag.MatchExactly)
            if items:
                self.presets.setCurrentItem(items[0])
        self._preset_buttons()

    def _preset_buttons(self):
        it = self.presets.currentItem()
        self.btn_del_preset.setEnabled(it is not None and it.text() in self.own_presets())
        self.btn_save_preset.setEnabled(bool(self.chosen()))
        self.btn_clear.setEnabled(bool(self.selected))

    def save_preset(self):
        import json
        from PySide6.QtWidgets import QInputDialog
        keys = self.chosen()
        if not keys:
            return
        name, ok = QInputDialog.getText(self.win, "Save as preset",
                                        f"A name for these {len(keys)} formats:")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in PRESETS:
            self.win.flash(f"“{name}” is a built-in preset: choose another name.")
            return
        own = self.own_presets()
        own[name] = keys
        self.win.prefs.setValue("publish/presets", json.dumps(own))
        self._reload_presets(name)
        self.win.flash(f"Preset “{name}” saved: {len(keys)} formats.")

    def delete_preset(self):
        import json
        it = self.presets.currentItem()
        own = self.own_presets()
        if it is None or it.text() not in own:
            return
        name = it.text()                     # the item goes when the list is rebuilt
        del own[name]
        self.win.prefs.setValue("publish/presets", json.dumps(own))
        self._reload_presets()
        self.win.flash(f"Preset “{name}” deleted.")

    def clear_selection(self):
        if self.project.busy:
            return
        self.selected = []
        self.presets.clearSelection()
        self._after_choice()

    def _preset(self, item):
        if self.project.busy:
            return
        keys = self.all_presets().get(item.text(), ())
        self.selected = [k for k in keys if k in self.formats and self.available(self.formats[k])]
        self._show_formats()
        self._after_choice()

    def _after_choice(self):
        for c in self.cards:
            c.set_selected(c.key in self.selected)
        self._preset_buttons()
        if self.project.export_rows and not self.project.busy:
            self.project.export_rows = {}           # a new choice: the last queue is history
        self.refresh()
        self.win.update_bottom()

    def recolour(self, t):
        self.t = t
        for c in self.cards:
            c.recolour(t)

    def refresh(self):
        p = self.project
        names = [self.formats[k].name for k in self.chosen()]
        self.sel_text.setText("\n".join(names) if names else "Nothing selected")
        self.where.setText(p.last_export_dir or self.win.default_export_dir())
        self._refresh_settings()
        self.queue.clear()
        for k, (state, why, files) in p.export_rows.items():
            name = self.formats[k].name if k in self.formats else k
            mark = {"done": "✓", "problem": "×", "writing": "›"}.get(state, " ")
            extra = (f"  ·  {len(files)} file{'s' if len(files) != 1 else ''}" if files else "")
            it = QListWidgetItem(f"{mark}  {name}   —   {self.ROW_MARK.get(state, state)}{extra}")
            if why:
                it.setToolTip(why)
                it.setText(it.text() + f"\n      {why}")
            self.queue.addItem(it)
        fit_list(self.queue)
        self.queue.setVisible(bool(p.export_rows))
        if p.busy == "export":
            self.queue_note.setText(f"{p.stage}…  {p.fraction * 100:.0f}%")
        elif p.export_rows:
            done = sum(1 for v in p.export_rows.values() if v[0] == "done")
            self.queue_note.setText(f"{done} of {len(p.export_rows)} written to "
                                    f"{p.last_export_dir}" + (f". {p.message}" if p.message else ""))
        else:
            self.queue_note.setText("Nothing exported yet.")
        self.btn_next.setVisible(bool(p.export_rows) and p.busy != "export")

    def next_step(self):
        p = self.project
        if p.busy == "export":
            return "Cancel export", "process.cancel"
        if p.export_rows and not self.reg.reason("file.reveal"):
            return "Show in Finder", "file.reveal"
        n = len(self.chosen())
        return (f"Export {n} format{'s' if n != 1 else ''}…" if n else "Export…"), "file.export_now"

    def status(self):
        p = self.project
        if p.busy == "export":
            return "info", p.stage or "Exporting…", f"{p.fraction * 100:.0f}%"
        if p.export_rows:
            bad = [k for k, v in p.export_rows.items() if v[0] != "done"]
            if bad:
                return "warn", "Exported with problems", p.message or f"{len(bad)} not written"
            return "ok", "Exported", f"{len(p.export_rows)} formats · {p.last_export_dir}"
        n = len(self.chosen())
        return "idle", f"{n} format{'s' if n != 1 else ''} selected", "Pick what it is for, then tick formats"


from helio_qt.capture_ws import CaptureWorkspace  # noqa: E402  (it uses this module's helpers)
from helio_qt.process_ws import ProcessWorkspace  # noqa: E402
from helio_qt.inspect_ws import InspectWorkspace  # noqa: E402
from helio_qt.measure_ws import MeasureWorkspace  # noqa: E402

WORKSPACE_CLASSES = (CaptureWorkspace, ProcessWorkspace, InspectWorkspace, MeasureWorkspace,
                     PublishWorkspace)
