"""Process: how the normals and the depth are solved, every classic setting
included (REC-001…REC-021, QC-001…008, ENG-018…025, IMP-004).

The settings are grouped as the solve uses them - the photographs, how the
frames are prepared, the light model, refinement, the depth - and folded so
the screen shows the few that matter until more are wanted.  After a solve
the centre shows the normal map, the depth and the normals' measured
uncertainty, with the figures that say how far the result can be trusted.
Recalculate Depth, Iterative Refine and the quality checks act on the solve
in hand; the computations are helio.application.process_tools, shared with
the classic app.
"""
from __future__ import annotations

import os
import time

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
                               QVBoxLayout, QWidget)

from helio_qt.controls import ParamRow, Section, Segmented, VScroll
from helio_qt.widgets import button, label, panel, pixmap_from_array, rule



# The working sizes offered (ProcessingSettings.work_scale): full size, a fraction, or automatic (0).
WORK_SIZES = ((1.0, "Full size"), (0.75, "75%"), (0.5, "50%"), (0.35, "35%"), (0.25, "25%"),
              (0.0, "Automatic: fit this machine's memory"))


def strength_key(s) -> str:
    """Which Lamp strengths choice *s* amounts to, in the order the solve
    takes them (pipeline): flat areas, the grey patch, the photographs."""
    if s.even_flat and (s.flat_areas or not (s.even_flash and s.flash_ref)):
        return "flat"
    if s.even_flash:
        return "grey"
    if s.semi_calibrated:
        return "photos"
    return "off"


def _found(p):
    r = getattr(p, "result", None) if getattr(p, "solved", False) else None
    return getattr(r, "lamp_distance", None) if r is not None else None


def lamp_distance_text(p, s, positions) -> str:
    """What the near-field correction uses for the lamps' distance, in words."""
    if positions is not None:
        return "Each lamp where the two balls place it."
    rec = _found(p)
    if s.near_field_distance == "radius":
        return ("Radius from the measured scale." if p.scale_mm_per_px else
                "No scale yet: the lamps are assumed 1.5 image-widths away. Set the scale in "
                "Measure to use the radius.")
    if rec is None:
        return "The lamps' distance is found from the photographs when you process."
    if rec.get("ratio") and rec.get("sure"):
        return (f"The photographs put the lamps about {rec['ratio']:.1f} times the picture's "
                "longer side away.")
    if rec.get("far_away"):
        return ("The photographs show the lamps far enough away not to matter: the correction "
                "changes nothing.")
    return ("The photographs do not show the lamps' distance clearly: "
            + ("the dome radius with the measured scale is used." if p.scale_mm_per_px else
               "they are assumed 1.5 image-widths away."))


def near_field_hint(p, s) -> str:
    """A line for the Solve section when the lamps were close and nothing corrects it."""
    rec = _found(p)
    if s.near_field or not rec or not rec.get("sure") or not rec.get("ratio") or rec["ratio"] > 4.0:
        return ""
    return (f"The lamps look close: about {rec['ratio']:.1f} times the picture's longer side away. "
            "On a flat object that bends the result into a dish. Near-field correction "
            "takes it out; check it once on a flat card shot the same way.")

class _Sig(QObject):
    done = Signal(str, object, str)


class _Task(QRunnable):
    def __init__(self, kind, fn, sig):
        super().__init__()
        self.kind, self.fn, self.sig = kind, fn, sig

    def run(self):
        try:
            self.sig.done.emit(self.kind, self.fn(), "")
        except Exception as e:                       # reported, never swallowed
            import logging
            logging.getLogger(__name__).exception("process tool %s failed", self.kind)
            self.sig.done.emit(self.kind, None, f"{type(e).__name__}: {e}")


def _ctl(key, text, lo, hi, default, step, tip, unit=""):
    from helio.domain.view_controls import Control
    return Control(key, text, lo, hi, default, step, tip, unit=unit)


class ProcessWorkspace:
    name = "Process"
    STAGES = (("Prepare frames", 0.0), ("Solve normals", 0.35), ("Integrate depth", 0.70),
              ("Build views", 0.95))

    def __init__(self, win):
        self.win, self.project, self.reg = win, win.project, win.registry
        self.sidebar, self.center, self.inspector = self.build()

    # ── layout ────────────────────────────────────────────────────────
    def build(self):
        from helio.application.process_tools import PRESETS, QUICK_PROFILES
        from helio.core.radiometry import PHOTO_TONES
        from helio.domain.registry import DEPTH_BLURBS, SOLVER_BLURBS
        from helio_qt.workspaces import INSPECTOR_W, SIDE_W, _list, _page, _scroll
        self.solver_blurbs, self.depth_blurbs = SOLVER_BLURBS, DEPTH_BLURBS
        store = self.win.prefs

        side, sl = panel("Sidebar", SIDE_W)
        sl.addWidget(label("PIPELINE", "heading"))
        self.pipeline = _list([n for n, _ in self.STAGES], selectable=False)
        sl.addWidget(self.pipeline)
        sl.addWidget(rule())
        sl.addWidget(label("QUICK PROFILES", "heading"))
        self.quick = QListWidget()
        self.quick.setAccessibleName("Quick profiles")
        for name, (_n, _d, _s, _a, tip) in QUICK_PROFILES.items():
            it = QListWidgetItem(name)
            it.setToolTip(tip)
            self.quick.addItem(it)
        self.quick.itemClicked.connect(lambda it: self.apply_quick(it.text()))
        self.quick.setMaximumHeight(140)
        sl.addWidget(self.quick)
        sl.addWidget(label("MATERIALS", "heading"))
        self.materials = QListWidget()
        self.materials.setAccessibleName("Material presets")
        for key, (text, _p) in PRESETS.items():
            it = QListWidgetItem(text)
            it.setData(Qt.ItemDataRole.UserRole, key)
            it.setToolTip("Sets the solver, depth method and the 3D settings for this material.")
            self.materials.addItem(it)
        self.materials.itemClicked.connect(lambda it: self.apply_material(
            it.data(Qt.ItemDataRole.UserRole)))
        sl.addWidget(self.materials, 1)

        page, pl = _page()
        self.title = label("Ready to solve", "title")
        pl.addWidget(self.title)
        self.summary = label("", "secondary", wrap=True)
        pl.addWidget(self.summary)
        # Said here as well as in Capture, because this is where the solve is started
        # and a frame left out is a light the solve will not have.
        self.left_out = label("", "reason", wrap=True)
        self.left_out.setProperty("status", "warn")
        self.left_out.setVisible(False)
        pl.addWidget(self.left_out)
        # A quick look: the solve small, in seconds, before the full one.
        qrow = QHBoxLayout()
        self.btn_quick = button("Quick look", "outline",
                                "The solve at about 1,000 pixels across, in seconds: to check "
                                "the settings before the full solve. Not kept.")
        self.btn_quick.clicked.connect(self.quick_look)
        qrow.addWidget(self.btn_quick)
        self.btn_try = button("Try a few", "outline",
                              "Several quick looks in a row, each changing one thing from your "
                              "settings (a correction, near-field, a shine- or shadow-resistant "
                              "solver), scored and listed below; the highest score is shown. The score is "
                              "consistency, not accuracy: compare their pictures.")
        self.btn_try.clicked.connect(self.try_a_few)
        qrow.addWidget(self.btn_try)
        self.quick_note = label("", "caption", wrap=True)
        qrow.addWidget(self.quick_note, 1)
        pl.addLayout(qrow)
        self._quick = None
        self._quick_busy = False
        self.run_box = QWidget()
        rl = QVBoxLayout(self.run_box)
        rl.setContentsMargins(0, 6, 0, 6)
        rl.setSpacing(8)
        self.stage = label("", "section")
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setAccessibleName("Processing progress")
        row = QHBoxLayout()
        self.detail = label("", "caption")
        row.addWidget(self.detail, 1)
        self.btn_cancel = button("Cancel", "", "Stop at the next step  ⌘.")
        self.btn_cancel.clicked.connect(lambda: self.reg.trigger("process.cancel"))
        row.addWidget(self.btn_cancel)
        rl.addWidget(self.stage)
        rl.addWidget(self.bar)
        rl.addLayout(row)
        pl.addWidget(self.run_box)
        self.message = label("", "reason", wrap=True)
        pl.addWidget(self.message)
        # When solved: what came out, and how sure it is.
        self.quick_head = label("", "section", wrap=True)
        self.quick_head.setVisible(False)
        pl.addWidget(self.quick_head)
        self.previews = QWidget()
        pv = QHBoxLayout(self.previews)
        pv.setContentsMargins(0, 8, 0, 0)
        pv.setSpacing(16)
        self.prev = {}
        self.prev_caps = {}
        for key, cap in (("normals", "Surface directions (normal map)"),
                         ("depth", "Height (depth map)"),
                         ("sigma", "How sure each direction is: dark sure, yellow doubtful")):
            col = QVBoxLayout()
            img = QLabel()
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img.setAccessibleName(cap)
            col.addWidget(img)
            self.prev_caps[key] = label(cap, "caption", wrap=True)
            col.addWidget(self.prev_caps[key])
            pv.addLayout(col)
            self.prev[key] = img
        pv.addStretch(1)
        pl.addWidget(self.previews)
        # One figure to choose by, and what it is made of.
        srow = QHBoxLayout()
        self.score_big = label("", "title")
        self.score_big.setAccessibleName("Ranking score")
        self.score_big.setToolTip("Out of 100: how sure the surface directions are, how well the "
                                  "lights were spread, how flat a marked grey patch or flat area came "
                                  "out, and whether the depth fits the directions. It measures how "
                                  "consistent a solve is, not how accurate: a low score warns that "
                                  "something is wrong, but between two sound solves a flatter, "
                                  "shallower one can score higher (DiLiGenT-Pi). Compare their "
                                  "pictures and the 3D relief; Measure › Accuracy Check on a "
                                  "reference measures accuracy.")
        srow.addWidget(self.score_big)
        self.score_parts = label("", "caption", wrap=True)
        srow.addWidget(self.score_parts, 1)
        pl.addLayout(srow)
        self.precision = label("", "secondary", wrap=True)
        self.precision.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pl.addWidget(self.precision)
        # The figures in plain words, folded away until asked for.
        self.btn_explain = button("What do these numbers mean?", "quiet",
                                  "A short explanation of the uncertainty, the light "
                                  "conditioning and the rating.")
        self.btn_explain.setCheckable(True)
        self.btn_explain.toggled.connect(lambda on: self.explain.setVisible(on))
        pl.addWidget(self.btn_explain, 0, Qt.AlignmentFlag.AlignLeft)
        self.explain = label(self.EXPLAIN, "secondary", wrap=True)
        self.explain.setTextFormat(Qt.TextFormat.RichText)
        self.explain.setMaximumWidth(720)
        self.explain.setVisible(False)
        pl.addWidget(self.explain)
        # Every quick look kept with its score: choose the best, put its settings back.
        self.looks_head = label("QUICK LOOKS", "heading")
        self.looks_head.setVisible(False)
        pl.addWidget(self.looks_head)
        self.looks_box = QVBoxLayout()
        self.looks_box.setSpacing(4)
        pl.addLayout(self.looks_box)
        self.looks = []                     # [{n, score, word, summary, settings}]
        self.reason = label("", "reason", wrap=True)
        pl.addWidget(self.reason)
        pl.addStretch(1)
        self._preview_of = None

        # Inspector
        insp, il = panel("Inspector", INSPECTOR_W)
        body = QWidget()
        vl = QVBoxLayout(body)
        vl.setContentsMargins(0, 0, 4, 0)
        vl.setSpacing(12)
        il.addWidget(VScroll(body), 1)
        self.rows = {}
        self.chk = {}

        sec = Section("Solve", "p_solve", store)
        sec.add(label("Normal solver", "secondary"))
        self.solver = QComboBox()
        from helio.domain.registry import MAIN_SOLVERS, MORE_SOLVERS
        self.solver.addItems(list(MAIN_SOLVERS))
        self.solver.insertSeparator(self.solver.count())
        self.solver.addItem("More solvers")             # a heading: it cannot be chosen
        self.solver.model().item(self.solver.count() - 1).setEnabled(False)
        self.solver.addItems(list(MORE_SOLVERS))
        self.solver.setAccessibleName("Normal solver")
        self.solver.setToolTip("How each pixel's normal is found from its brightness under each "
                               "light. The note below says what each method is for.")
        sec.add(self.solver)
        self.solver_note = label("", "caption", wrap=True)
        sec.add(self.solver_note)
        # Lamps close to a flat object bend it into a dish: said when the photographs show it.
        self.nf_hint = label("", "caption", wrap=True)
        sec.add(self.nf_hint)
        self.btn_nf_on = button("Turn on near-field correction", "quiet",
                                "Light model › Near-field correction, with the lamp distance "
                                "found from the photographs. Process again to see it.")
        self.btn_nf_on.clicked.connect(lambda: self.project.update_settings(near_field=True))
        sec.add(self.btn_nf_on)
        # No ball: what the solve found, and the one choice it cannot always make.
        self.no_ball_note = label("", "caption", wrap=True)
        sec.add(self.no_ball_note)
        c = QCheckBox("Turn the relief inside out")
        c.setToolTip("No ball: raised and sunken are told apart from the cast shadows, or by "
                     "taking the object to stand up from its surround. If the relief came out "
                     "inside out (the tooling sunk where it is raised), tick this and process "
                     "again.")
        c.toggled.connect(lambda v: self.project.update_settings(no_ball_inside_out=bool(v)))
        sec.add(c)
        self.chk["no_ball_inside_out"] = c
        self.rows["shadow_thresh"] = sec.add(ParamRow(_ctl(
            "shadow_thresh", "Shadow threshold", 5, 40, 12, 1,
            "A frame much darker at a pixel than that pixel's own bright frames is treated as shadow "
            "there: 12 is a tenth of them, at any exposure. Shadow-Rejection LS only. Higher rejects "
            "more.")))
        self.rows["rpca_lambda"] = sec.add(ParamRow(_ctl(
            "rpca_lambda", "Outlier threshold", 0.1, 1.0, 0.3, 0.05,
            "How far a reading may stray from what a matte surface could give before it is set "
            "aside, as a share of the pixel's mean brightness. Lower sets aside more (fainter "
            "sheen too); higher only the brightest highlights. Sparse-Outlier LS only.")))
        sec.add(label("Depth method", "secondary"))
        self.depth = QComboBox()
        self.depth.addItems(list(DEPTH_BLURBS))
        self.depth.setAccessibleName("Depth method")
        self.depth.setToolTip("How height is built from the normals. The note below says what "
                              "each method is for.")
        sec.add(self.depth)
        self.depth_note = label("", "caption", wrap=True)
        sec.add(self.depth_note)
        sec.add(label("The photographs", "secondary"))
        self.tone = QComboBox()
        for k, text in PHOTO_TONES.items():
            self.tone.addItem(text, k)
        self.tone.setAccessibleName("How the photographs record light")
        self.tone.setToolTip("How the photographs' values relate to light. Automatic decodes camera "
                             "files to linear light by their own curve: as stored, fine relief comes "
                             "out at about 60% of its depth (DiLiGenT-Pi). Uneven lighting can then "
                             "show as a broad tilt: Flattened in the 3D relief, or a white board, "
                             "takes it out. As stored keeps the old, shallower result. RAW files are "
                             "always read as linear.")
        sec.add(self.tone)
        sec.add(label("Working size", "secondary"))
        self.work = QComboBox()
        for value, text in WORK_SIZES:
            self.work.addItem(text, value)
        self.work.setAccessibleName("Working size")
        self.work.setToolTip(
            "Full size solves from every pixel of the photographs. A smaller working size reduces "
            "each photograph as it is read, so a capture too big for this machine's memory can be "
            "solved, in a fraction of the memory and the time: 50% needs about a quarter. The "
            "result keeps the detail a screen or a print shows, but its pixels are bigger, so "
            "everything measured in it is in those pixels (the scale is converted for you). "
            "Automatic picks full size when the capture fits in free memory, otherwise the "
            "largest size that does.")
        sec.add(self.work)
        self.work_note = label("", "caption", wrap=True)
        sec.add(self.work_note)
        self._size_cache, self._note_cache = {}, None
        vl.addWidget(sec)

        # Light and flash: uneven light is the largest cause of false tilt on
        # flat work, so its corrections sit together, open, in the order to try.
        sec = Section("Light and flash", "p_lightflash", store,
                      tip="Uneven light makes flat things look tilted. Work down the list; "
                          "Compare shows what each correction does.")
        self.light_sec = sec
        sec.add(label("Uneven light makes flat things look tilted. Work down the list.",
                      "caption", wrap=True))
        # Ambient: the light every frame shares (daylight, room light), from a lights-off frame.
        sec.add(label("Lights off · daylight or room light", "secondary", wrap=True))
        self.ambient_note = label("", "caption", wrap=True)
        sec.add(self.ambient_note)
        row = QHBoxLayout()
        self.btn_ambient = button("Choose lights-off frame…", "quiet",
                                  "A photograph taken with the lamps off and everything else as "
                                  "for the lit frames. Light that reaches every frame alike tells "
                                  "the solve nothing about direction and makes the relief come out "
                                  "shallower; it is subtracted from every frame first.")
        self.btn_ambient.clicked.connect(lambda: self.win.load_ambient())
        row.addWidget(self.btn_ambient)
        self.btn_ambient_clear = button("Remove", "quiet", "Solve without the lights-off frame.")
        self.btn_ambient_clear.clicked.connect(lambda: self.win.clear_ambient())
        row.addWidget(self.btn_ambient_clear)
        row.addStretch(1)
        sec.add_layout(row)
        # Lamp strengths: one choice (it was three checkboxes, each "not needed
        # when" another was on).  The solve takes them in this order anyway
        # (pipeline: flat areas, else the grey patch, else semi-calibrated).
        sec.add(label("Lamp strengths · when the lamps or flashes are not equal", "secondary",
                      wrap=True))
        self.strength = Segmented(
            [("off", "Off"), ("flat", "Flat areas"), ("grey", "Grey patch"),
             ("photos", "From the photographs")],
            {"off": "The lamps are taken as equally strong.",
             "flat": "Best for a hand-held flash. A flash held to one side lights that side "
                     "more, and the solve reads it as the surface tilting. Mark areas you know "
                     "are flat (the board round a book, the grey card, a margin), on both sides "
                     "of the object if you can: each frame's strength and fall-off are measured "
                     "across them and evened out.",
             "grey": "Each flash's strength, from a flat matte grey card in the shot. A flash "
                     "carried round by hand is stronger in some frames than others (on six "
                     "bindings, up to 2.6 times). Found automatically when you open the "
                     "photographs, or marked by hand.",
             "photos": "Nothing to mark (semi-calibrated). Every lamp's strength is found from "
                       "the photographs themselves, with the light directions held: LEDs on a "
                       "dome differ by 10-30%. A marked reference measures it more surely."},
            columns=2)
        self.strength.chosen.connect(self._choose_strength)
        sec.add(self.strength)
        # Flat areas (CAL-004): marked boxes, shown when Flat areas is chosen.
        self.flat_area_note = label("", "caption", wrap=True)
        sec.add(self.flat_area_note)
        row = QHBoxLayout()
        self.btn_flat_area = button("Mark a flat area…", "quiet",
                                    "Drag a box over an area you know is flat and plain, lit by "
                                    "every flash. Mark two or more, either side of the object, "
                                    "for the best.")
        self.btn_flat_area.clicked.connect(lambda: self.win.mark_flat_area())
        row.addWidget(self.btn_flat_area)
        self.btn_flat_clear = button("Clear", "quiet", "Forget the marked flat areas.")
        self.btn_flat_clear.clicked.connect(lambda: self.win.clear_flat_areas())
        row.addWidget(self.btn_flat_clear)
        row.addStretch(1)
        self.flat_row = QWidget()
        self.flat_row.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        sec.add(self.flat_row)
        # Grey patch (CAL-003): shown when Grey patch is chosen.
        self.flash_note = label("", "caption", wrap=True)
        sec.add(self.flash_note)
        self.btn_grey = button("Mark the grey patch…", "quiet",
                               "Drag a box over a flat, matte grey patch on a photograph: a "
                               "grey card lying beside the object, lit by every flash.")
        self.btn_grey.clicked.connect(lambda: self.win.mark_grey_reference())
        sec.add(self.btn_grey)
        self.semi_note = label("", "caption", wrap=True)
        sec.add(self.semi_note)
        sec.add(label("Flat objects only · it flattens real shape too", "secondary", wrap=True))
        c = QCheckBox("Flatten broad light gradients")
        c.setToolTip("For flat or nearly flat objects (documents, coins, printing plates): "
                     "removes the broad brightness gradient that lamps at different distances "
                     "leave (multi-scale Retinex): a flat copper plate went from 11° of false "
                     "tilt to 2.4°. The colour picture is left as photographed. Leave it off "
                     "for curved or deep objects: it flattens their real shape as well.")
        c.toggled.connect(lambda v: self.project.update_settings(flatten_illumination=bool(v)))
        sec.add(c)
        self.chk["flatten_illumination"] = c
        self.btn_compare = button("Compare corrections…", "",
                                  "Solves small with each correction on its own, in seconds, "
                                  "and shows how flat the grey patch and flat areas come out.")
        self.btn_compare.clicked.connect(self.compare_corrections)
        sec.add(self.btn_compare)
        self.compare_note = label("", "caption", wrap=True)
        self.compare_note.setVisible(False)
        sec.add(self.compare_note)
        vl.addWidget(sec)

        # Precision measurement: a fixed rig calibrated to measure with (CAL-005..007). All optional.
        sec = Section("Precision measurement", "p_precision", store, open_=False,
                      tip="For a fixed rig shot to measure: each light calibrated on a white board, "
                          "the lens corrected, and the result checked on a reference.")
        self.precision_sec = sec
        sec.add(label("Optional. For a fixed rig shot to measure as precisely as it can: calibrate "
                      "once per rig set-up, save a rig profile, and every capture on it uses the "
                      "calibration.", "caption", wrap=True))
        b = button("Set up for measurement", "",
                   "The solve's choices for measuring, at once: the solver that measured best on the "
                   "DiLiGenT benchmark, free-boundary depth, the tone decoded by the file's curve, "
                   "and every look-only adjustment off.")
        b.clicked.connect(lambda: self.win.registry.trigger("process.measurement"))
        sec.add(b)
        sec.add(label("White-board light field", "secondary", wrap=True))
        self.lf_note = label("", "caption", wrap=True)
        sec.add(self.lf_note)
        row = QHBoxLayout()
        b = button("Calibrate on a white board…", "quiet",
                   "A matte white board filling the frame, photographed once under each light from "
                   "the rig's fixed positions, same camera, exposure and flash power as the capture. "
                   "Measures every light's strength and fall-off; every frame is evened by it.")
        b.clicked.connect(lambda: self.win.registry.trigger("process.light_field"))
        row.addWidget(b)
        self.btn_lf_off = button("Off", "quiet", "Solve without the white-board light field.")
        self.btn_lf_off.clicked.connect(lambda: self.win.registry.trigger("process.light_field_clear"))
        row.addWidget(self.btn_lf_off)
        row.addStretch(1)
        sec.add_layout(row)
        sec.add(label("Older: a flat reference", "secondary", wrap=True))
        sec.add(label("The same board photographs, used the older way: their tilt is subtracted "
                      "after the solve. The light field above evens the frames before it, which "
                      "is more exact; use one, not both.", "caption", wrap=True))
        self.btn_flat = button("Load flat reference images…", "quiet", "Photographs of a blank, even "
                               "card under the same lights, one per light, in the same order.")
        self.btn_flat.clicked.connect(lambda: self.reg.trigger("process.flat"))
        sec.add(self.btn_flat)
        self.flat_note = label("No flat reference loaded", "caption", wrap=True)
        sec.add(self.flat_note)
        self.chk["flat_correction"] = QCheckBox("Apply the correction")
        self.chk["flat_correction"].setToolTip("Subtract the flat reference's tilt from the solved "
                                               "directions (after the solve).")
        self.chk["flat_correction"].toggled.connect(
            lambda v: self.project.update_settings(flat_correction=bool(v)))
        sec.add(self.chk["flat_correction"])
        self.rows["flat_corr_strength"] = sec.add(ParamRow(_ctl(
            "flat_corr_strength", "Correction strength", 0.0, 1.0, 1.0, 0.01,
            "1 removes all of the measured distortion.")))
        sec.add(label("Lens", "secondary", wrap=True))
        self.lens_note = label("", "caption", wrap=True)
        sec.add(self.lens_note)
        row = QHBoxLayout()
        b = button("Calibrate from a chessboard…", "quiet",
                   "10-20 photographs of a flat printed chessboard at different places and tilts in "
                   "the frame, same lens, focal length and focus as the capture. Takes the lens's "
                   "distortion out of every frame before the solve.")
        b.clicked.connect(lambda: self.win.registry.trigger("process.lens"))
        row.addWidget(b)
        self.btn_lens_off = button("Off", "quiet", "Solve without the lens correction.")
        self.btn_lens_off.clicked.connect(lambda: self.win.registry.trigger("process.lens_clear"))
        row.addWidget(self.btn_lens_off)
        row.addStretch(1)
        sec.add_layout(row)
        row = QHBoxLayout()                     # its own row: three buttons do not fit the panel
        b = button("Print a chessboard…", "quiet",
                   "A PDF of the calibration chessboard at A4 to A1, Letter or Tabloid. Any laser print "
                   "will do, mounted flat.")
        b.clicked.connect(lambda: self.win.registry.trigger("process.print_chessboard"))
        row.addWidget(b)
        row.addStretch(1)
        sec.add_layout(row)
        sec.add(label("Ready to measure?", "secondary", wrap=True))
        self.ready_note = label("", "caption", wrap=True)
        self.ready_note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sec.add(self.ready_note)
        vl.addWidget(sec)

        sec = Section("Preparing the frames", "p_prep", store, open_=False)
        for key, text, tip in (
                ("auto_align", "Line up shifted frames",
                 "Phase correlation against the median frame: undoes small camera movement."),
                ("pre_clahe", "Per-image local contrast (CLAHE)",
                 "Raking enhancement before the solve. Not photometrically exact: for difficult "
                 "low-contrast surfaces.")):
            c = QCheckBox(text)
            c.setToolTip(tip)
            c.toggled.connect(lambda v, k=key: self.project.update_settings(**{k: bool(v)}))
            sec.add(c)
            self.chk[key] = c
        self.rows["pre_clahe_clip"] = sec.add(ParamRow(_ctl(
            "pre_clahe_clip", "CLAHE clip limit", 1.0, 8.0, 2.0, 0.1, "Contrast limit per tile.")))
        self.rows["pre_clahe_tile"] = sec.add(ParamRow(_ctl(
            "pre_clahe_tile", "CLAHE tile size", 4, 64, 16, 1, "Tile edge in pixels.", "px")))
        self.rows["adaptive_light"] = sec.add(ParamRow(_ctl(
            "adaptive_light", "Adaptive light weight", 0.0, 1.0, 0.0, 0.01,
            "Down-weights near-grazing lights before the solve. 0 = all equal; 0.5 halves lights "
            "below about 25°. For rough stone and uneven parchment.")))
        vl.addWidget(sec)

        sec = Section("Light model", "p_light", store, open_=False)
        self.chk["near_field"] = QCheckBox("Near-field correction (small dome)")
        self.chk["near_field"].setToolTip(
            "Lamps close to the object light each point from a slightly different direction, "
            "and the near side more brightly. On a flat object that bends the result into a "
            "dish. This takes it out, given how far the lamps were: found from the "
            "photographs, or the dome radius with a measured scale.")
        self.chk["near_field"].toggled.connect(
            lambda v: self.project.update_settings(near_field=bool(v)))
        sec.add(self.chk["near_field"])
        self.nf_dist_label = sec.add(label("Lamp distance", "secondary"))
        self.nf_dist = Segmented([("photos", "From the photographs"), ("radius", "Dome radius")], {
            "photos": "Found from the photographs: the distance at which every pixel's "
                      "brightnesses best fit a matte surface. With two balls, their measured "
                      "positions are used instead.",
            "radius": "The dome radius below, with the scale measured in Measure."})
        self.nf_dist.chosen.connect(lambda k: self.project.update_settings(near_field_distance=k))
        sec.add(self.nf_dist)
        self.rows["dome_radius_mm"] = sec.add(ParamRow(_ctl(
            "dome_radius_mm", "Dome radius", 50, 2000, 200, 5,
            "Distance from the lights to the object.", "mm")))
        # Where the dome is centred: the lights were measured on the ball, so
        # a ball at the side of the frame measured them from there.
        self.nf_from_label = sec.add(label("Lamps measured from", "secondary"))
        self.nf_from = Segmented([("ball", "The ball"), ("centre", "The centre")], {
            "ball": "The dome is centred where the chrome ball is: right for a ball at the side "
                    "of the frame, and the same as the centre for a ball in the middle.",
            "centre": "The dome is centred on the picture's middle, as before. For lights from "
                      "a file measured somewhere else."})
        self.nf_from.chosen.connect(lambda k: self.project.update_settings(near_field_from=k))
        sec.add(self.nf_from)
        self.nf_note = label("", "caption", wrap=True)
        sec.add(self.nf_note)
        vl.addWidget(sec)

        # Gold and metal: a robust solver only where the surface shines.
        sec = Section("Gold and metal", "p_metal", store, open_=False,
                      tip="Gilding, clasps and metal point throw ordinary least squares off; "
                          "solve them with a robust solver and the rest as chosen above.")
        c = QCheckBox("Metal-aware solve")
        c.setToolTip("Finds where the surface shines (gold tooling, gilt edges, embroidered gold, "
                     "clasps, bosses, metal point): places whose readings a matte surface could not "
                     "give, or that flare far above the matte about them under one lamp and are not "
                     "dark. Those are solved again with the metal solver below and blended in at "
                     "their edges; paper, leather and parchment keep the chosen solver. With only "
                     "four lamps every reading is needed, so the re-solve changes only the "
                     "glints. Needs calibrated lights.")
        c.toggled.connect(lambda v: self.project.update_settings(metal_aware=bool(v)))
        sec.add(c)
        self.chk["metal_aware"] = c
        self.metal_label = sec.add(label("Solver where it shines", "secondary"))
        from helio.domain.registry import METAL_SOLVERS
        self.metal_solver = QComboBox()
        self.metal_solver.addItems(list(METAL_SOLVERS))
        self.metal_solver.setAccessibleName("Solver where the surface shines")
        self.metal_solver.setToolTip("Sparse-Outlier LS sets the highlights aside entirely; IRLS "
                                     "and Robust LS turn them down.")
        self.metal_solver.currentTextChanged.connect(
            lambda v: self.project.update_settings(metal_solver=v))
        sec.add(self.metal_solver)
        self.rows["metal_sensitivity"] = sec.add(ParamRow(_ctl(
            "metal_sensitivity", "Sensitivity", 0.0, 1.0, 0.3, 0.05,
            "Higher marks fainter sheen as metal too (worn gilding, varnish); lower only bright "
            "gold and metal. 0.3 suits most bindings; from 0.5 matte areas close to a lamp "
            "and the grey card start to be marked.")))
        self.metal_note = label("", "caption", wrap=True)
        sec.add(self.metal_note)
        vl.addWidget(sec)

        sec = Section("Refinement", "p_refine", store, open_=False)
        self.chk["sfs_refine"] = QCheckBox("Shape-from-shading refinement")
        self.chk["sfs_refine"].setToolTip(
            "After the solve, tightens the normals against the first frame's shading: less "
            "noise on smooth areas, edges kept. Needs calibrated lights.")
        self.chk["sfs_refine"].toggled.connect(
            lambda v: self.project.update_settings(sfs_refine=bool(v)))
        sec.add(self.chk["sfs_refine"])
        self.rows["sfs_iters"] = sec.add(ParamRow(_ctl(
            "sfs_iters", "Iterations", 1, 10, 5, 1, "3-5 is a good balance; 8-10 for the most.")))
        self.rows["sfs_prior"] = sec.add(ParamRow(_ctl(
            "sfs_prior", "Hold to the solve", 0.3, 0.95, 0.70, 0.01,
            "How strongly the refined normals stay with the solved ones. 0.65-0.75 for flat "
            "inscriptions, 0.8+ for 3D objects.")))
        vl.addWidget(sec)

        sec = Section("Depth", "p_depth", store, open_=False)
        self.chk["parabolic_corr"] = QCheckBox("Parabolic correction")
        self.chk["parabolic_corr"].setToolTip(
            "Removes the best-fit bowl or saddle (a quadratic surface) from the depth: the "
            "warp integration leaves on convex objects. Fine relief is kept.")
        self.chk["parabolic_corr"].toggled.connect(
            lambda v: self.project.update_settings(parabolic_corr=bool(v)))
        sec.add(self.chk["parabolic_corr"])
        self.chk["mask_surround"] = QCheckBox("Depth on the object alone")
        self.chk["mask_surround"].setToolTip(
            "Off by default. For a small object on a dark surround (a punch, a coin on velvet): "
            "the depth is worked out on the object only, so the surround's noise stays out of it, "
            "and the surround is flat. The object is found from its brightness, so dark parts of "
            "it (deep letters, shadowed edges) can be taken for surround and change the relief "
            "and terrain views there: compare with it off. Does nothing when the object fills "
            "the frame.")
        self.chk["mask_surround"].setChecked(False)
        self.chk["mask_surround"].toggled.connect(
            lambda v: self.project.update_settings(mask_surround=bool(v)))
        sec.add(self.chk["mask_surround"])
        self.rows["pre_smooth"] = sec.add(ParamRow(_ctl(
            "pre_smooth", "Smoothing before integration", 0, 10, 0, 1,
            "Edge-keeping smoothing of the normals before the depth is integrated (0 = off). "
            "The normals themselves stay as solved.")))
        self.clip_row = sec.add(ParamRow(_ctl(
            "depth_clip_pct", "Depth outlier clip", 0.01, 4.9, 0.5, 0.01,
            "Percent of the deepest and highest pixels clipped before the depth is scaled for "
            "viewing and export.", "%")))
        self.clip_row.changed.connect(lambda _k, v, _f: self.win.shared.__setitem__(
            "depth_clip_pct", float(v)))
        self.btn_recalc = button("Recalculate depth", "", "Integrate the solved normals again "
                                 "with the depth method and smoothing above. Seconds, not minutes.")
        self.btn_recalc.clicked.connect(lambda: self.reg.trigger("process.recalc"))
        sec.add(self.btn_recalc)
        self.rows["refine_iters"] = sec.add(ParamRow(_ctl(
            "refine_iters", "Refine iterations", 2, 8, 4, 1,
            "Iterative normal refinement passes.")))
        self.rows["refine_weight"] = sec.add(ParamRow(_ctl(
            "refine_weight", "Keep the solved detail", 0.5, 0.97, 0.85, 0.01,
            "0.85 keeps 85% of the solved normals and blends 15% from the integrated depth.")))
        self.btn_refine = button("Iterative normal refine", "",
                                 "Makes the normals integrable (alternately integrating and "
                                 "differentiating), then re-integrates the depth. The views keep "
                                 "the solved normals.")
        self.btn_refine.clicked.connect(lambda: self.reg.trigger("process.refine"))
        sec.add(self.btn_refine)
        vl.addWidget(sec)

        sec = Section("Region", "p_region", store, open_=False)
        sec.add(label("Solve only part of the image, at full resolution: faster, and the "
                      "depth is integrated over that area alone.", "caption", wrap=True))
        self.btn_region = button("Draw a region and process it…", "", "Drag a rectangle on the "
                                 "image; only that part is solved, at full resolution.")
        self.btn_region.clicked.connect(lambda: self.reg.trigger("process.reprocess"))
        sec.add(self.btn_region)
        self.btn_whole = button("Process the whole image again", "quiet",
                                "Leave the region and solve every pixel again.")
        self.btn_whole.clicked.connect(self.whole_image)
        sec.add(self.btn_whole)
        self.region_note = label("", "caption", wrap=True)
        sec.add(self.region_note)
        vl.addWidget(sec)

        sec = Section("Quality checks", "p_qc", store, open_=False)
        self.rows["qc_sigma"] = sec.add(ParamRow(_ctl(
            "qc_sigma", "Broad / fine split", 5, 50, 10, 1,
            "Scale separating broad shape (where calibration error lives) from fine detail.",
            "px")))
        for aid, text, tip in (
                ("qc.report", "Normal QC report", "Residuals, coverage and the normals' "
                 "statistics, as a page you can save."),
                ("qc.bias", "Low-frequency bias", "The broad tilt left in the normals: the mark "
                 "of light-position error."),
                ("qc.compare", "Compare with a reference normal map…", "Angle by angle against "
                 "a normal map you trust, with the differences mapped."),
                ("qc.disagreement", "Integrator disagreement", "Where two ways of building "
                 "height from the normals disagree: those heights are least sure."),
                ("qc.uncertainty", "Save uncertainty maps…", "Each pixel's normal uncertainty "
                 "and residual, as images."),
                ("qc.dome3d", "3D light dome", "The lights in 3D round the object, to check "
                 "their spread.")):
            b = button(text, "quiet", tip)
            b.clicked.connect(lambda _c=False, a=aid: self.reg.trigger(a))
            sec.add(b)
        vl.addWidget(sec)
        vl.addStretch(1)

        for key, r in self.rows.items():
            if key in ("refine_iters", "refine_weight", "qc_sigma"):
                continue
            if key == "dome_radius_mm":
                r.changed.connect(lambda _k, v, _f: self.project.update_settings(
                    dome_radius_mm=float(v)))
                continue
            r.changed.connect(lambda k, v, final: final and self.project.update_settings(
                **{k: (int(round(v)) if k in ("shadow_thresh", "pre_clahe_tile", "sfs_iters")
                       else float(v))}))
        self.solver.currentTextChanged.connect(lambda v: self.win.set_solver(v))
        self.depth.currentTextChanged.connect(lambda v: self.win.set_depth(v))
        self.tone.currentIndexChanged.connect(
            lambda i: self.project.update_settings(photo_tone=self.tone.itemData(i)))
        self.work.currentIndexChanged.connect(
            lambda i: self.project.update_settings(work_scale=float(self.work.itemData(i))))

        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.sig = _Sig()
        self.sig.done.connect(self._task_done)
        sa = _scroll(page)                     # the quick looks list grows: the page scrolls
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return side, sa, insp

    def recolour(self, t):
        pass

    # ── profiles ──────────────────────────────────────────────────────
    def apply_quick(self, name):
        from helio.application.process_tools import quick_profile
        self.project.update_settings(**quick_profile(name))
        self.win.flash(f"Quick profile “{name}”: solver, depth method, smoothing and light "
                       "weighting set. Process to apply.")

    def apply_material(self, key):
        from helio.application.process_tools import PRESETS, preset
        settings, shared, view = preset(key)
        self.project.update_settings(**settings)
        self.win.shared.update(shared)
        if view:
            self.win.engine = view
            self.win.view = view
        self.win.flash(f"Material “{PRESETS[key][0]}”: solver, depth method and 3D settings set. "
                       "Process to apply.")

    # ── tools on the solve in hand ────────────────────────────────────
    def run_task(self, kind, fn, busy_text):
        self.project.set_busy("tool")
        self.project.set_progress(busy_text, 0.5, "")
        self.pool.start(_Task(kind, fn, self.sig))

    def wait(self):
        from PySide6.QtWidgets import QApplication
        self.pool.waitForDone(600000)
        QApplication.processEvents()

    def recalc(self):
        from helio.application.process_tools import with_new_depth
        r, s = self.project.result, self.project.settings
        clip = float(self.win.shared.get("depth_clip_pct", 0.5))
        self.run_task("recalc", lambda: with_new_depth(
            r, s.depth_engine, smooth=int(round(s.pre_smooth)), clip_pct=clip,
            parabolic=s.parabolic_corr), f"Integrating with {s.depth_engine}…")

    def refine(self):
        from helio.application.process_tools import refine_normals, with_new_depth
        r, s = self.project.result, self.project.settings
        it = int(self.rows["refine_iters"].value())
        wt = float(self.rows["refine_weight"].value())
        clip = float(self.win.shared.get("depth_clip_pct", 0.5))
        self.run_task("refine", lambda: with_new_depth(
            r, s.depth_engine, smooth=int(round(s.pre_smooth)), clip_pct=clip,
            parabolic=s.parabolic_corr, normals=refine_normals(r.normals_full, it, wt)),
            f"Refining the normals ({it} passes)…")

    def _task_done(self, kind, out, err):
        p = self.project
        p.busy = ""
        if err or out is None:
            p.finish(f"{'Recalculate depth' if kind == 'recalc' else 'Refinement'} stopped: {err}")
            return
        p.set_result(out, p.solve_seconds)
        self.win.flash("Depth recalculated with " + out.settings.depth_engine
                       if kind == "recalc" else
                       "Normals refined; the depth is integrated from them. The views keep the "
                       "solved normals.")

    def load_flat(self):
        from helio.application.process_tools import flat_reference_normals
        files, _ = QFileDialog.getOpenFileNames(
            self.win, "Flat reference images (a blank surface, the same lights)",
            str(self.win.prefs.value("last_folder", "")),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*)")
        if not files:
            return
        try:
            flat, info = flat_reference_normals(files, self.project.solve_lights(),
                                                self.project.settings.resolved_photo_tone())
        except ValueError as e:
            QMessageBox.warning(self.win, "Flat reference", str(e))
            return
        self.win.flat_ref_normals, self.win.flat_ref_info = flat, info
        self.project.update_settings(flat_correction=True)
        self.win.flash(f"Flat reference: {info['n']} images. Mean tilt X {info['mean_abs_x']:.4f}, "
                       f"Y {info['mean_abs_y']:.4f}. Applied on the next Process.")

    EXPLAIN = (
        "<p><b>The score</b>, out of 100, is for ranking solves of this capture: higher is "
        "better. Run a few quick looks with different settings, then Process with the best one "
        "(Use these puts its settings back). Over 80 very good, 65 good, 50 fair.</p>"
        "<p><b>It is not an accuracy.</b> It is worked out from the photographs themselves, "
        "which cannot see a light direction measured wrong, uneven light or a camera's tone "
        "curve: a capture can score well and still be out. Only Measure › Accuracy Check, on "
        "a gauge block or a flat plate shot like the object, measures the error. Its result "
        "is shown beside the score once you have run it.</p>"
        "<p>It adds up four things:</p>"
        "<p><b>Sureness</b> (60): how well the photographs agree about which way each point "
        "faces. The figures below it, the <i>normal uncertainty</i>, say how far a direction "
        "could be off: half the object is surer than the <i>median</i>, nearly all of it "
        "within <i>95% under</i>. Smaller is better; 1° is excellent, 8° poor. Parts too dark "
        "to read (a black surround) are left out.</p>"
        "<p><b>Lights</b> (20). Light conditioning says how well the lights were spread "
        "round the object: 2 or under is good; over 5 the lights were bunched or all at one "
        "height.</p>"
        "<p><b>Flatness</b> (20): how far a grey patch or flat area you marked leans: 0° is "
        "flat. Only when one is marked.</p>"
        "<p><b>Depth</b> (20): whether the depth map is the surface the directions describe: "
        "the slopes of the depth against the solved directions, over the object. 4° or under "
        "full marks, 20° none. It is the part that tells solvers apart: a solver that fits "
        "noise in the dark makes a blotchy depth and loses it here, though the other parts "
        "are the same.</p>"
        "<p><b>The word under the figures:</b> <i>Metrological</i>, the photographs agree to "
        "1°: precise enough to measure with, once the accuracy check confirms it; "
        "<i>Research</i>, to 3°, fit for comparing and publishing; "
        "<i>Exploratory</i>, good for seeing the surface, not for exact figures.</p>"
        "<p><b>What lowers it:</b> shine on gilding or varnish, shadows, a flash that moved "
        "or changed strength, the object moving. The yellow map shows where.</p>")

    TIP_GREY = ("For hand-held flash, place a flat, matte grey card beside the object "
                "(a QPcard or a ColorChecker grey patch), lying flat and lit by every flash: "
                "Helio evens out the flash with it.")

    def _frame_size(self):
        """(w, h) of the photographs, read once per capture."""
        imgs = self.project.images
        if not imgs:
            return None
        if getattr(self, "_size_of", (None, None))[0] != imgs[0]:
            from helio.core.sphere import image_size
            self._size_of = (imgs[0], image_size(imgs[0]))
        return self._size_of[1]

    def _flat_status(self, s) -> tuple[str, str]:
        """(one short line, status colour) for the flat areas, with a warning
        when they cannot speak for the whole object (all on one side; read
        differently from one another, which means the light falls off)."""
        from helio.core.flash import DISAGREE_LIMIT, coverage, parse_boxes
        boxes = parse_boxes(s.flat_areas)
        n = len(boxes)
        r = self.project.result
        fl = getattr(r, "flat_light", None) if r is not None else None
        size = self._frame_size()
        cov = coverage(boxes, size) if (boxes and size) else {"side": ""}
        other = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        if fl and s.even_flat and not self.project.settings_changed_since_solve():
            dis, side = fl.get("disagree"), (fl.get("coverage") or {}).get("side", "")
            if dis and dis > DISAGREE_LIMIT and side:
                return (f"Evened, but the areas disagree by ×{dis:.1f} (the flash falls off across "
                        f"the frame) and all lie on the {side}: the {other[side]} may come out "
                        f"tilted. Mark one on the {other[side]}, or compare with this off", "warn")
            return (f"Evened from {len(fl['boxes'])} flat area{'s' if len(fl['boxes']) != 1 else ''}: "
                    f"the flash fell off up to {fl['falloff'] * 100:.0f}% across the frame", "ok")
        if not n:
            return "No flat areas marked", "idle"
        where = f"{n} flat area{'s' if n != 1 else ''} marked"
        if cov["side"]:
            return (where + f", all on the {cov['side']}: the light's change towards the "
                    f"{other[cov['side']]} is not measured. Mark one on the {other[cov['side']]}",
                    "warn")
        if n == 1:
            where += " (one more, across the object, measures the fall-off better)"
        return (where + (": used when you process" if s.even_flat else "; off"),
                "ok" if s.even_flat else "idle")

    def _flash_status(self, s) -> tuple[str, str, str]:
        """(one short line, its explanation for the tooltip, status colour):
        the panel is narrow, so the line stays short and the how-to waits
        under the pointer."""
        text = self._flash_text(s)
        if text.startswith("No grey card"):
            return "No grey card in these photos (hover for how)", self.TIP_GREY, "warn"
        status = ("info" if "compare with this off" in text else
                  "ok" if text.startswith(("Evened", "Grey reference")) else "info")
        return text, text, status

    def _choose_strength(self, key: str):
        """Lamp strengths: exactly one of the three, or none."""
        self.project.update_settings(even_flat=key == "flat", even_flash=key == "grey",
                                     semi_calibrated=key == "photos")

    def _semi_text(self, s) -> str:
        """What the semi-calibrated correction found, or why it is not used."""
        if not s.semi_calibrated:
            return ""
        if (s.even_flat and s.flat_areas) or (s.even_flash and s.flash_ref):
            return "The flat areas or the grey patch measure the lamps; they are used instead."
        r = self.project.result
        ls = getattr(r, "light_strength", None) if r is not None else None
        if ls and not self.project.settings_changed_since_solve():
            return (f"Found: the lamps varied ×{ls['spread']:.2f} over {len(ls['strength'])} "
                    f"frames, evened")
        return "Each lamp's strength is found when you process."

    def _metal_text(self, s) -> str:
        """What the metal-aware solve re-solved, or why it is not needed."""
        if s.resolved_normal_engine() == "Sparse-Outlier LS (RPCA)":
            return ("Not needed: the whole surface is solved with Sparse-Outlier LS (Solve, "
                    "above). Choose another solver there to use this on the shiny parts only.")
        r = self.project.result
        m = getattr(r, "metal", None) if r is not None else None
        if not s.metal_aware or not m or self.project.settings_changed_since_solve():
            return ""
        if not m["pixels"]:
            return "Nothing shone enough to re-solve: raise the sensitivity if there is gilding."
        return f"Re-solved {m['share'] * 100:.1f}% of the surface with {m['solver']}."

    def _ambient_text(self, s) -> str:
        """What the lights-off frame will do, did, or would."""
        if not s.ambient_frame:
            return ("None. Outdoors or in a lit room, take one photograph with the lamps off and "
                    "choose it here.")
        name = os.path.basename(s.ambient_frame)
        r = self.project.result
        amb = getattr(r, "ambient", None) if r is not None else None
        if amb and not self.project.settings_changed_since_solve():
            lvl = amb["level"] * 100
            more = (" That is a lot: the relief would have come out noticeably shallower."
                    if lvl >= 10 else "")
            return f"{name}: subtracted; the ambient light was {lvl:.0f}% of the lit frames'.{more}"
        if not os.path.exists(s.ambient_frame):
            return f"{name}: not found. Choose it again."
        return f"{name}: subtracted from every frame when you process."

    def _flash_text(self, s) -> str:
        """What the grey-card correction will do, did, or needs."""
        r = self.project.result
        fl = getattr(r, "flash", None) if r is not None else None
        ref = getattr(self.win, "grey_ref", {}) or {}
        if fl and s.even_flash and not self.project.settings_changed_since_solve():
            off = (fl.get("coverage") or {}).get("offcentre", 0.0)
            if off > 0.6:
                return (f"Evened: the flash varied ×{fl['spread']:.2f}, read on one card at the "
                        f"edge. If the flash was close, the far side can come out worse: compare "
                        f"with this off, or mark flat areas on both sides")
            return f"Evened: the flash varied ×{fl['spread']:.2f} ({fl['used']} frames)"
        if ref.get("state") == "looking":
            return "Looking for a grey card in the photographs…"
        if ref.get("state") == "checking":
            return "Checking the marked patch in every frame…"
        if s.flash_ref:
            where = "marked by hand" if ref.get("state") == "marked" else "found on a grey card"
            seen = (f", flash ×{ref['spread']:.1f}"
                    if ref.get("state") == "marked" and ref.get("spread") else "")
            return (f"Grey reference {where}{seen}: used when you process"
                    if s.even_flash else f"Grey reference {where}; off")
        return "No grey card found. " + self.TIP_GREY

    def region(self):
        self.win.pick_rect(self._region_drawn, "Drag a rectangle over the area to process.")

    def _region_drawn(self, x1, y1, x2, y2):
        D = self.win.render_data
        if D is None:
            return
        # The view is drawn at full size (scene units are image pixels).
        from helio.application.viewing import image_point
        ax, ay = image_point(x1, y1, D.w, D.h, self.win.display)
        bx, by = image_point(x2, y2, D.w, D.h, self.win.display)
        r = self.project.result
        ox, oy = (r.crop[0], r.crop[1]) if r is not None and r.crop else (0, 0)
        box = (int(min(ax, bx)) + ox, int(min(ay, by)) + oy, int(max(ax, bx)) + ox,
               int(max(ay, by)) + oy)
        if box[2] - box[0] < 32 or box[3] - box[1] < 32:
            self.win.flash("That region is too small: draw at least 32 × 32 pixels.")
            return
        self.win.solve_crop = box
        # The box is in this result's own base pixels: reduced ones when it was solved at a
        # smaller working size.  The solve reads the region from the photographs at full size.
        self.win.solve_crop_ref = (r.base_w, r.base_h) if r is not None else None
        if r is not None and float(getattr(r, "work_scale", 1.0) or 1.0) < 1.0:
            self.win.flash("Solving the region at full size, from the photographs' own pixels.", 6)
        self.reg.trigger("process.run")

    def whole_image(self):
        self.win.solve_crop = None
        self.reg.trigger("process.run")

    # ── state → screen ───────────────────────────────────────────────
    def _stage_index(self) -> int:
        p = self.project
        if p.busy != "solve":
            return len(self.STAGES) if p.solved else -1
        return max(i for i, (_, f) in enumerate(self.STAGES) if p.fraction >= f - 1e-9)

    def _work_note(self, s) -> str:
        """What this capture needs at the chosen working size, on this machine."""
        import os
        import time
        env = os.environ.get("HELIO_WORK_SCALE", "").strip()
        p = self.project
        paths = p.solve_images() if p.images else []
        if not paths:
            return ("Reduces the photographs as they are read, for a machine short of memory. "
                    "Pixels, scales and measurements are then in the reduced pixels.")
        key = (tuple(paths), float(s.work_scale), s.flatten_illumination, s.pre_clahe, s.auto_align,
               s.multi_norm, float(s.adaptive_light), env)
        c = self._note_cache
        if c and c[0] == key and time.monotonic() - c[1] < 5.0:      # the memory reading is not free
            return c[2]
        first = paths[0]
        if first not in self._size_cache:
            from helio.application.pipeline import _photo_size
            self._size_cache[first] = _photo_size([first])
        size = self._size_cache[first]
        if size is None:
            text = ""
        else:
            from helio.utils.memory_plan import describe_choice
            text = describe_choice(size[1], size[0], len(paths), s, float(s.work_scale))
            if env:
                text += f" HELIO_WORK_SCALE={env} overrides this."
        self._note_cache = (key, time.monotonic(), text)
        return text

    _cal_cache: dict = {}

    def _cal_text(self, path: str, kind: str) -> str:
        """A calibration file in a few words (cached by path and modification time)."""
        if not path:
            return ""
        try:
            key = (path, os.path.getmtime(path))
        except OSError:
            return f"{os.path.basename(path)}: not found — calibrate again or turn it off."
        if key not in self._cal_cache:
            try:
                if kind == "lf":
                    from helio.core import light_field as LF
                    self._cal_cache[key] = LF.describe(LF.load(path))
                else:
                    from helio.core import lens
                    self._cal_cache[key] = lens.describe(lens.load(path))
            except Exception as e:                       # noqa: BLE001
                self._cal_cache[key] = f"unreadable ({e})"
        return f"{os.path.basename(path)}: {self._cal_cache[key]}"

    def _refresh_precision(self, p, s):
        from helio.application.precision import readiness, readiness_text
        lf = self._cal_text(s.light_field, "lf")
        self.lf_note.setText(lf or "Off. Every light taken as equally strong everywhere in the frame.")
        self.btn_lf_off.setVisible(bool(s.light_field))
        ln = self._cal_text(s.lens_calibration, "lens")
        self.lens_note.setText(ln or "Off. The photographs used as the lens drew them.")
        self.btn_lens_off.setVisible(bool(s.lens_calibration))
        L = p.solve_lights() if p.lights is not None else None
        cond = float("inf")
        if L is not None and len(L) >= 3:
            from helio.core.sphere import condition_number
            cond = condition_number(L)
        from helio.io.raw_loader import is_raw
        imgs = p.solve_images() if p.images else []
        items = readiness(n_lights=0 if L is None else len(L), conditioning=cond,
                          raw=bool(imgs) and all(is_raw(x) for x in imgs), photo_tone=s.photo_tone,
                          light_field=s.light_field, lens_calibration=s.lens_calibration,
                          near_field=bool(s.near_field), scale=bool(p.scale_mm_per_px),
                          z_from_scale=bool(p.z_from_scale), accuracy=p.accuracy)
        self.ready_note.setText(readiness_text(items))
        self.ready_note.setToolTip(readiness_text(items, details=True))

    def refresh(self):
        p, s = self.project, self.project.settings
        running = p.busy in ("solve", "tool")
        for w, v in ((self.solver, s.normal_engine), (self.depth, s.depth_engine)):
            w.blockSignals(True)
            w.setCurrentText(v)
            w.blockSignals(False)
        self.tone.blockSignals(True)
        self.tone.setCurrentIndex(max(0, self.tone.findData(s.photo_tone)))
        self.tone.blockSignals(False)
        self.work.blockSignals(True)
        i = self.work.findData(float(s.work_scale))
        if i < 0:                                   # a size from a session that is not in the list
            self.work.addItem(f"{float(s.work_scale) * 100:.0f}%", float(s.work_scale))
            i = self.work.count() - 1
        self.work.setCurrentIndex(i)
        self.work.blockSignals(False)
        self.work_note.setText(self._work_note(s))
        for key, c in self.chk.items():
            c.blockSignals(True)
            c.setChecked(bool(getattr(s, key)))
            c.blockSignals(False)
        txt, tip, st = self._flash_status(s)
        self.flash_note.setText(txt)
        self.flash_note.setToolTip(tip)
        if self.flash_note.property("status") != st:
            from helio_qt.widgets import restyle
            self.flash_note.setProperty("status", st)
            restyle(self.flash_note)
        self.ambient_note.setText(self._ambient_text(s))
        self._refresh_precision(p, s)
        self.btn_ambient_clear.setVisible(bool(s.ambient_frame))
        key = strength_key(s)
        self.strength.set_current(key)
        for w_ in (self.flat_area_note, self.flat_row):
            w_.setVisible(key == "flat")
        for w_ in (self.flash_note, self.btn_grey):
            w_.setVisible(key == "grey")
        self.btn_flat_clear.setVisible(bool(s.flat_areas))
        ft, fst = self._flat_status(s)
        self.flat_area_note.setText(ft)
        if self.flat_area_note.property("status") != fst:
            from helio_qt.widgets import restyle
            self.flat_area_note.setProperty("status", fst)
            restyle(self.flat_area_note)
        for key, r in self.rows.items():
            if hasattr(s, key):
                r.set_value(getattr(s, key))
        self.clip_row.set_value(float(self.win.shared.get("depth_clip_pct", 0.5)))
        self.rows["shadow_thresh"].setVisible(s.resolved_normal_engine() == "Shadow-Rejection LS")
        rpca = "Sparse-Outlier LS (RPCA)"
        self.rows["rpca_lambda"].setVisible(s.resolved_normal_engine() == rpca
                                            or (s.metal_aware and s.metal_solver == rpca))
        for w_ in (self.metal_label, self.metal_solver, self.rows["metal_sensitivity"]):
            w_.setVisible(s.metal_aware)
        # Sparse-Outlier LS for the whole surface already does what the metal-aware
        # solve would do in places: say so rather than offer it twice.
        whole = s.resolved_normal_engine() == rpca
        self.chk["metal_aware"].setEnabled(not whole)
        for w_ in (self.metal_label, self.metal_solver, self.rows["metal_sensitivity"]):
            w_.setEnabled(not whole)
        self.metal_solver.blockSignals(True)
        self.metal_solver.setCurrentText(s.metal_solver)
        self.metal_solver.blockSignals(False)
        self.semi_note.setText(self._semi_text(s))
        self.semi_note.setVisible(key == "photos" and bool(self.semi_note.text()))
        self.metal_note.setText(self._metal_text(s))
        self.metal_note.setVisible(bool(self.metal_note.text()))
        for k in ("pre_clahe_clip", "pre_clahe_tile"):
            self.rows[k].setVisible(s.pre_clahe)
        for k in ("sfs_iters", "sfs_prior"):
            self.rows[k].setVisible(s.sfs_refine)
        near = s.near_field or s.normal_engine == "Near-Field Attenuation"    # the solver needs them too
        self.nf_dist.set_current(s.near_field_distance)
        for w_ in (self.nf_dist, self.nf_dist_label):
            w_.setVisible(near and self.win.light_positions() is None)
        self.rows["dome_radius_mm"].setVisible(near and s.near_field_distance == "radius")
        self.nf_note.setVisible(near)
        self.nf_from.setVisible(near)
        self.nf_from_label.setVisible(near)
        self.nf_from.set_current(s.near_field_from)
        origin = self.win.light_origin()
        if s.near_field_from == "centre":
            where = "Centred on the picture's middle."
        elif origin is not None:
            where = f"Centred on the ball, at {origin[0]:,.0f}, {origin[1]:,.0f} px."
        else:
            where = ("No ball position (lights from a file or by hand): centred on the "
                     "picture's middle.")
        self.nf_note.setText(where + " " + lamp_distance_text(p, s, self.win.light_positions()))
        tip = near_field_hint(p, s)
        self.nf_hint.setText(tip)
        for w_ in (self.nf_hint, self.btn_nf_on):
            w_.setVisible(bool(tip))
        self.solver_note.setText(self.solver_blurbs.get(s.normal_engine, ""))
        no_ball = s.resolved_normal_engine() == "Uncalibrated PCA"
        self.chk["no_ball_inside_out"].setVisible(no_ball)
        if no_ball:
            from helio_qt.capture_ws import no_ball_status
            line, tip = no_ball_status(p)
            from helio_qt.capture_ws import _short
            self.no_ball_note.setText(_short(line))
            self.no_ball_note.setToolTip(tip)
        self.no_ball_note.setVisible(no_ball)
        self.depth_note.setText(self.depth_blurbs.get(s.depth_engine, ""))
        info = self.win.flat_ref_info
        self.flat_note.setText(f"{info['n']} flat images, {info['w']} × {info['h']} px"
                               if info else "No flat reference loaded")
        self.chk["flat_correction"].setEnabled(self.win.flat_ref_normals is not None)
        self.rows["flat_corr_strength"].setVisible(self.win.flat_ref_normals is not None)
        crop = self.win.solve_crop
        self.region_note.setText(f"Last region: {crop[2] - crop[0]} × {crop[3] - crop[1]} px at "
                                 f"{crop[0]}, {crop[1]}" if crop else "")
        self.btn_whole.setVisible(crop is not None)
        for b in (self.btn_recalc, self.btn_refine):
            b.setEnabled(p.solved and not running)
        for w in (self.solver, self.depth, self.tone):
            w.setEnabled(not running)

        at = self._stage_index()
        for i, (name, _) in enumerate(self.STAGES):
            it = self.pipeline.item(i)
            mark = "✓" if i < at else ("›" if i == at and p.busy == "solve" else " ")
            it.setText(f"{mark}  {name}")

        if p.busy == "solve":
            self.title.setText("Processing…")
        elif p.busy == "tool":
            self.title.setText(p.stage or "Working…")
        elif p.settings_changed_since_solve():
            self.title.setText("Settings changed since this solve")
        elif p.solved:
            self.title.setText("Solved")
        else:
            self.title.setText("Ready to solve")
        used = len(p.solve_images()) if p.images else 0
        lines = [f"{used} of {len(p.images)} frames" if p.excluded else
                 (f"{len(p.images)} frames" if p.images else "No images yet"),
                 (f"Lights from {p.light_source.replace('file:', '')}" if p.lights_ready()
                  else "Lights not set"),
                 f"Solver: {s.normal_engine}", f"Depth: {s.depth_engine}",
                 f"Photographs: {self.tone.currentText()}"]
        if self.win.solve_crop:
            lines.append("Region only")
        if p.solved and not running:
            r = p.result
            lines.append(f"Last solve: {r.w} × {r.h} px from {r.n_images} frames, "
                         f"{p.solve_seconds:.1f} s")
        self.summary.setText("\n".join(lines))
        out = sorted(i + 1 for i in p.excluded) if p.images else []
        if out and not running:
            where = ", ".join(str(i) for i in out)
            self.left_out.setText(f"Solving without {len(out)} of {len(p.images)} frames "
                                  f"({where}). Capture › Frames puts them back.")
            self.left_out.setVisible(True)
        else:
            self.left_out.setVisible(False)

        self.run_box.setVisible(running)
        if running:
            self.stage.setText(p.stage or "Starting…")
            self.bar.setValue(int(p.fraction * 1000))
            self.bar.setRange(0, 1000 if p.busy == "solve" else 0)
            self.detail.setText(p.detail or (f"{p.fraction * 100:.0f}%" if p.busy == "solve" else ""))
            self.btn_cancel.setVisible(p.busy == "solve")
            self.btn_cancel.setEnabled(not self.reg.reason("process.cancel"))
        self.message.setText(p.message)
        self.message.setVisible(bool(p.message))

        # A quick look stands until a full solve, or another capture, replaces it.
        q = self._quick
        if q is not None and (q["of"] is not p.result or q["paths"] != p.solve_images()):
            q = self._quick = None
        shown = q["result"] if q else (p.result if p.solved else None)
        vis = shown is not None and not running
        self.previews.setVisible(vis)
        self.precision.setVisible(vis)
        self.score_big.setVisible(vis and getattr(self, "_score_ok", False))
        self.score_parts.setVisible(vis and getattr(self, "_score_ok", False))
        if self.looks and self.looks[-1]["paths"] != p.solve_images():
            self.looks = []                         # another capture: its own quick looks
            self._draw_looks()
        self.btn_explain.setVisible(vis)
        self.explain.setVisible(vis and self.btn_explain.isChecked())
        self.quick_head.setVisible(vis and q is not None)
        if q is not None:
            r = q["result"]
            later = "  ·  settings changed since" if q["settings"] != s else ""
            self.quick_head.setText(f"Quick look · {r.w:,} × {r.h:,} px · {q['seconds']:.1f} s"
                                    f"{later}\nA preview, not kept: Process for the full solve.")
        if vis and self._preview_of is not shown:
            self._preview_of = shown
            self._make_previews(shown)
        ready = not running and not self._quick_busy and not self.reg.reason("process.run")
        self.btn_quick.setEnabled(ready)
        self.btn_try.setEnabled(ready)
        self.btn_compare.setEnabled(ready and bool(s.flash_ref or s.flat_areas))
        self.btn_compare.setToolTip(
            "Solves small with each correction on its own, in seconds, and shows how flat the "
            "grey patch and flat areas come out." if (s.flash_ref or s.flat_areas) else
            "Mark a flat area or the grey patch first: Compare measures how flat they come out.")
        on = [n for k, n in (("even_flat", "flat areas"), ("even_flash", "grey patch"),
                             ("flatten_illumination", "flat object")) if getattr(s, k)]
        self.light_sec.badge.setText(", ".join(on) + " on" if on else "")
        label_, aid = self.next_step()
        why = self.reg.reason(aid) if aid else ""
        self.reason.setText(why)
        self.reason.setVisible(bool(why))

    # ── quick look and compare (seconds, small) ──────────────────────
    def _small_job(self, fn, note, done):
        from helio_qt.widgets import restyle
        why = self.reg.reason("process.run")
        if why:
            self.win.flash(why)
            return
        req = self.win.solve_request()
        self._quick_busy = True
        note.setText("Working… a few seconds" + (" (reading the photographs small first)"
                                                 if not self._small_ready(req) else ""))
        note.setVisible(True)
        note.setProperty("status", "")
        restyle(note)
        self.refresh()
        t = time.time()
        of, paths = self.project.result, list(req.paths)

        def back(out, err):
            self._quick_busy = False
            if paths != self.project.solve_images():
                note.setText("")
                self.refresh()
                return
            if err or out is None:
                note.setText(f"Could not solve small: {err}")
                note.setProperty("status", "warn")
                restyle(note)
            else:
                done(out, req, of, paths, time.time() - t)
            self.refresh()
        self.win.run_background(lambda: fn(req), back)

    @staticmethod
    def _small_ready(req) -> bool:
        from helio.application import quick_look
        return quick_look._cache.get("key", (None,))[0] == tuple(req.paths)

    def quick_look(self):
        from helio.application import quick_look

        def done(out, req, of, paths, secs):
            r, _scale = out
            self._quick = dict(result=r, of=of, paths=paths, settings=req.settings, seconds=secs)
            self._preview_of = None
            self.quick_note.setText("")
            self._add_look(r, req.settings, secs)
        self._small_job(quick_look.quick_solve, self.quick_note, done)

    def try_a_few(self):
        """Quick looks of each change worth trying, scored and listed; the best shown."""
        import dataclasses
        from helio.application import quick_look
        from helio.application.process_tools import try_variants
        variants = try_variants(self.project.settings, self.win.light_origin() is not None)

        def run(req):
            out = []
            for name, kw in variants:
                st = dataclasses.replace(req.settings, **kw)
                r, _s = quick_look.quick_solve(req, settings=st)
                out.append((name, st, r))
            return out

        def done(out, req, of, paths, secs):
            scored = []
            for _name, st, r in out:
                self._add_look(r, st, secs / max(len(out), 1))
                scored.append((self.looks[-1]["score"] or -1, r, st, self.looks[-1]["n"]))
            best = max(scored, key=lambda t: t[0])
            self.shown_look = best[3]
            self._draw_looks()
            self._quick = dict(result=best[1], of=of, paths=paths, settings=best[2],
                               seconds=secs / max(len(out), 1))
            self._preview_of = None
            same = len({sc_ for sc_, *_ in scored}) == 1
            self.quick_note.setText(
                f"Tried {len(out)} in {secs:.0f} s: " +
                ("they all score the same; compare their pictures with Show below."
                 if same else "the highest score is shown, and every one is in the list below. "
                 "The score is consistency, not accuracy: compare the pictures."))
        self._small_job(run, self.quick_note, done)
        if self._quick_busy:
            self.quick_note.setText(f"Trying {len(variants)} quick looks, one after another…")

    def compare_corrections(self):
        from helio.application import quick_look
        from helio_qt.compare_dialog import CompareDialog, summary

        def done(out, req, of, paths, secs):
            self.last_compare = out
            self.compare_note.setText(summary(out))
            self.compare_dialog = CompareDialog(self.win, out, self.use_correction)
            self.compare_dialog.show()
        self._small_job(quick_look.compare, self.compare_note, done)

    def use_correction(self, name: str):
        """One light correction on alone (or none), as Compare offers."""
        from helio.application.quick_look import VARIANTS
        kw = dict(VARIANTS)[name]
        self.project.update_settings(**kw)

    def _make_previews(self, r):
        n = r.proxy["normals"]
        nrgb = np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        d = np.clip(r.proxy["depth_p"] * 255.0, 0, 255).astype(np.uint8)
        imgs = {"normals": nrgb, "depth": d}
        sd = r.proxy.get("quality_sigma_deg")
        full = (r.quality or {}).get("sigma_deg")
        from helio.application.process_tools import precision_rating, readable
        mask = readable(r.albedo_grey_full, full) if full is not None else None
        if sd is not None:
            import matplotlib.cm as cm
            # 0-10°, as Inspect's view of it; further when most of the object
            # is beyond that, so the map never turns solid yellow.
            top = 10.0
            if full is not None:
                p95 = precision_rating(full, mask)[2]
                if p95 > top:
                    top = float(min(45.0, np.ceil(p95 / 5.0) * 5.0))
            imgs["sigma"] = (cm.cividis(np.clip(sd / top, 0, 1))[..., :3] * 255).astype(np.uint8)
            self.prev_caps["sigma"].setText(
                f"How sure each direction is: dark sure, yellow doubtful (0–{top:.0f}°)")
        for key, lbl in self.prev.items():
            arr = imgs.get(key)
            lbl.setVisible(arr is not None)
            if arr is not None:
                lbl.setPixmap(pixmap_from_array(arr).scaled(
                    230, 230, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
        sc = self._score_of(r)
        self._score_ok = sc is not None
        self.score_big.setVisible(sc is not None)
        self.score_parts.setVisible(sc is not None)
        if sc is not None:
            self.score_big.setText(f"{sc['score']}  {sc['word']}")
            bits = [f"{n} {p}/{o}" for n, p, o in sc["parts"]]
            if not any(n == "Flatness" for n, _p, _o in sc["parts"]):
                bits.append("flatness: mark a grey patch or flat area to include it")
            from helio.application.process_tools import accuracy_note
            bits.append(accuracy_note(getattr(self.project, "accuracy", None)))
            bits.append("consistency, not accuracy: a flatter solve can score higher")
            self.score_parts.setText("  ·  ".join(bits))
        if full is not None:
            rating, med, p95 = precision_rating(full, mask)
            from helio.core.sphere import condition_number
            cond = condition_number(r.light_matrix) if r.light_matrix is not None else float("nan")
            left = (f"  ·  {100 - np.mean(mask) * 100:.0f}% too dark to judge, left out"
                    if mask is not None and np.mean(mask) < 0.995 else "")
            self.precision.setText(
                f"Normal uncertainty (1σ): median {med:.2f}°, 95% under {p95:.2f}°  ·  "
                f"light conditioning {cond:.2f}{left}\n{rating}")
        else:
            self.precision.setText("No precision estimate: it needs calibrated lights and at "
                                   "least four frames.")

    def _score_of(self, r):
        """The overall score of solve *r* (process_tools.overall_score), or None
        without an uncertainty map (uncalibrated lights, too few frames)."""
        from helio.application.process_tools import (depth_agreement_of, overall_score,
                                                     precision_rating, readable, reference_leans)
        full = (r.quality or {}).get("sigma_deg")
        if full is None:
            return None
        _rating, med, p95 = precision_rating(full, readable(r.albedo_grey_full, full))
        from helio.core.sphere import condition_number
        cond = condition_number(r.light_matrix) if r.light_matrix is not None else None
        return overall_score(med, p95, cond, reference_leans(r), depth_agreement_of(r))

    def _add_look(self, r, settings, seconds: float = 0.0):
        """A quick look into the list, with its score, what was on, a small
        normal map, and the solve itself so it can be shown again."""
        from helio.application.process_tools import settings_summary
        sc = self._score_of(r)
        n = (self.looks[-1]["n"] + 1) if self.looks else 1
        thumb = None
        nm = (getattr(r, "proxy", None) or {}).get("normals")
        if nm is not None:
            import cv2
            rgb = np.clip((np.asarray(nm) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            k = 44.0 / max(1, rgb.shape[0])
            thumb = cv2.resize(rgb, (max(1, int(rgb.shape[1] * k)), 44), interpolation=cv2.INTER_AREA)
        self.looks.append(dict(n=n, paths=list(self.project.solve_images()),
                               score=None if sc is None else sc["score"],
                               word="" if sc is None else sc["word"],
                               summary=settings_summary(settings), settings=settings,
                               result=r, seconds=seconds, thumb=thumb))
        self.looks = self.looks[-12:]
        self.shown_look = n
        self._draw_looks()

    def show_look(self, n: int):
        """Quick look *n*'s pictures and score at the top again."""
        lk = next((x for x in self.looks if x["n"] == n), None)
        if lk is None:
            return
        self._quick = dict(result=lk["result"], of=self.project.result, paths=lk["paths"],
                           settings=lk["settings"], seconds=lk["seconds"])
        self._preview_of = None
        self.shown_look = n
        self._draw_looks()
        self.refresh()

    def _draw_looks(self):
        def drop(w_):                     # gone at once, not drawn over until Qt deletes it
            w_.hide()
            w_.setParent(None)
            w_.deleteLater()
        while self.looks_box.count():
            it = self.looks_box.takeAt(0)
            lay = it.layout()
            if lay is not None:
                while lay.count():
                    w_ = lay.takeAt(0).widget()
                    if w_ is not None:
                        drop(w_)
            elif it.widget() is not None:
                drop(it.widget())
        self.looks_head.setVisible(bool(self.looks))
        scores = [lk["score"] for lk in self.looks if lk["score"] is not None]
        best = max(scores, default=None)
        tied = scores.count(best) if best is not None else 0
        all_same = len(self.looks) > 1 and len(set(scores)) == 1 and len(scores) == len(self.looks)
        if all_same:
            note = label("These all score the same. The score tells corrections apart by how flat "
                         "a marked grey patch or flat area comes out, and nothing is marked here: "
                         "compare their pictures with Show, or mark one in Light and flash.",
                         "caption", wrap=True)
            note.setProperty("status", "warn")
            self.looks_box.addWidget(note)
        shown = getattr(self, "shown_look", None)
        for lk in reversed(self.looks):
            row = QHBoxLayout()
            row.setSpacing(10)
            if lk.get("thumb") is not None:
                pic = QLabel()
                pic.setPixmap(pixmap_from_array(lk["thumb"]))
                pic.setToolTip("Its normal map")
                row.addWidget(pic)
            top = (not all_same and best is not None and lk["score"] == best and len(self.looks) > 1)
            mark = ("  ✓ joint highest score" if tied > 1 else "  ✓ highest score") if top else ""
            here = "  ◀ shown" if lk["n"] == shown else ""
            sc = "–" if lk["score"] is None else f"{lk['score']} {lk['word']}"
            t = label(f"Quick look {lk['n']}  ·  {sc}  ·  {lk['summary']}{mark}{here}",
                      "section" if (top or here) else "secondary")
            row.addWidget(t, 1)
            b = button("Show", "quiet", "This quick look's pictures and score at the top.")
            b.clicked.connect(lambda _c=False, n_=lk["n"]: self.show_look(n_))
            b.setEnabled(lk["n"] != shown)
            row.addWidget(b)
            b = button("Use these", "quiet", "Put this quick look's settings back, to look again "
                       "or Process with them.")
            b.clicked.connect(lambda _c=False, st=lk["settings"]: self.use_look(st))
            row.addWidget(b)
            self.looks_box.addLayout(row)

    def use_look(self, settings):
        """A quick look's settings back (the marked areas as they are now)."""
        cur = self.project.settings
        self.project.update_settings(**{k: getattr(settings, k) for k in settings.__dataclass_fields__
                                        if k not in ("flash_ref", "flat_areas")
                                        and getattr(settings, k) != getattr(cur, k)})
        self.win.flash("Those settings are back: Quick look to see it again, or Process.")

    def next_step(self):
        p = self.project
        if not p.images:
            return "Import images…", "file.import"
        if p.busy == "solve":
            return "Cancel processing", "process.cancel"
        if p.solved and not p.settings_changed_since_solve():
            return "Inspect the result →", "view.ws:Inspect"
        from helio_qt.state import NEED_LIGHTS
        if self.reg.reason("process.run") == NEED_LIGHTS:
            return "Set the lights in Capture →", "view.ws:Capture"
        n = len(p.solve_images())
        return (f"Process {n} images again" if p.solved else f"Process {n} images"), "process.run"

    def status(self):
        p = self.project
        if not p.images:
            return "idle", "No images yet", "Drop a folder of captures anywhere in this window"
        if p.busy == "solve":
            return "info", p.stage or "Processing…", f"{p.fraction * 100:.0f}%"
        if p.busy == "tool":
            return "info", p.stage or "Working…", ""
        if p.message:
            return "warn", "Not processed", p.message
        if p.solved:
            r = p.result
            return ("ok", "Solved", f"{r.settings.normal_engine} · {r.settings.depth_engine} · "
                                    f"{p.solve_seconds:.1f} s")
        why = self.reg.reason("process.run")
        if why:
            return "warn", "Not ready", why
        s = p.settings
        return "info", "Ready to solve", f"{s.normal_engine} · {s.depth_engine}"
