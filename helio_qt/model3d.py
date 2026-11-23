"""The 3D model: the classic OpenGL viewer, its settings and its recordings (GEO-001…014, EXM-004…008).

The viewer is helio.render.viewer.run_opengl_viewer in its own process, the
same window the classic app opens; helio.application.viewer3d builds what
it is given and drives it.  Its settings live in ``win.shared`` (what the
exports read) and are copied into the viewer's shared values as they change,
so the open window follows every slider, and the Inspect light, live.

Inspect shows a short "3D model" section; the full set of the classic 3D
tab's controls opens in a window of its own, folded into groups.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QMessageBox,
                               QVBoxLayout, QWidget)

from helio.application import viewer3d as V3
from helio.domain.view_controls import Control
from helio_qt.controls import ParamRow, Section, VScroll
from helio_qt.widgets import button, label

KIND_NAMES = (("standard", "As captured"), ("flattened", "Auto-flattened"),
              ("inverted", "Inverted (peaks become valleys)"), ("heatmap", "Elevation heatmap"))

# (key, label, lo, hi, step, tip): the classic 3D tab's sliders, its words shortened.
GEOMETRY = (
    ("step", "Mesh density (step)", 1, 10, 1, "Vertex spacing in the viewer: 1 is every pixel, "
     "2–4 a good balance, 5–10 a fast preview."),
    ("height", "Height exaggeration", 10, 5000, 10, "Scales the relief. Match the real depth "
     "for physical accuracy; raise it to see shallow marks."),
    ("shape_blend", "Detail vs shape", 0.0, 1.0, 0.01, "0 keeps the whole shape (buckle, warp); "
     "1 keeps only fine detail on a flat base. 0.3 suits most objects."),
    ("hp_sigma", "Warp flatten (taco fix)", 0.0, 1.0, 0.01, "High-pass on the depth before meshing: "
     "removes the bowl that integration can leave."),
    ("warp_rem", "Broad warp remove", 0.0, 1.0, 0.01, "Removes broad curvature, bow or tilt."),
    ("micro_relief", "Fine detail boost", 0.0, 3.0, 0.05, "Boosts fine surface texture. 1–2 for "
     "shallow coins and engravings."),
    ("mesh_smooth", "Mesh smoothing", 0.0, 5.0, 0.1, "Gaussian smoothing of the depth grid; "
     "0.5–1.5 is usually enough."),
    ("hf_normals_cutoff", "HF + normals detail cutoff", 0.05, 1.0, 0.01, "For the HF + normals "
     "viewer: lower keeps more shape, higher isolates detail (and noise)."),
    ("depth_clip_pct", "Depth outlier clip", 0.01, 4.9, 0.01, "Clips the top and bottom percent "
     "of heights: stops tears and spikes squashing fine relief."),
    ("x_tilt", "X tilt correction", -15.0, 15.0, 0.1, "Removes a left–right slope (copy-stand tilt)."),
    ("y_tilt", "Y tilt correction", -15.0, 15.0, 0.1, "Removes a front–back slope."),
    ("edge_trim", "Edge trim", 0.0, 15.0, 0.5, "Percent cut from every edge of the mesh."),
)
UNITS = {"x_tilt": "°", "y_tilt": "°", "depth_clip_pct": "%", "edge_trim": "%"}
MATERIAL = (
    ("ambient_str", "Ambient", 0.0, 1.0, 0.01, "Light on every surface regardless of angle."),
    ("diffuse_str", "Diffuse (matte)", 0.0, 2.0, 0.01, "How strongly the surface answers the light."),
    ("specular_str", "Specular (shine)", 0.0, 2.0, 0.01, "Strength of the glossy highlight."),
    ("shininess", "Shininess", 1.0, 128.0, 1.0, "Size of the highlight: 1 broad, 128 a tight spot."),
)
TONE3D = (
    ("viewer_gamma", "Display gamma", 0.3, 3.0, 0.01, "Display only."),
    ("viewer_saturation", "Colour saturation", 0.0, 2.0, 0.01, "0 greyscale, 1 natural."),
    ("shadow_lift", "Shadow lift", 0.0, 0.4, 0.01, "Keeps the darkest areas from going black."),
    ("highlight_clip", "Highlight clip", 0.5, 1.0, 0.01, "Rolls off bright highlights."),
)
EXPORT = (
    ("export_step", "Export mesh density (step)", 1, 10, 1, "Vertex spacing for OBJ, PLY, GLB, "
     "E57: 1 full resolution, 2 a good balance."),
)
SHARPEN = (
    ("strength", "Normal strength", 0.1, 3.0, 0.1, "Detail above the smoothing scale: 1 original, "
     "2 doubled, 0.5 halved. Needs Normal smooth above 0."),
    ("smooth", "Normal smooth", 0, 8, 1, "The scale that separates detail from shape."),
    ("usm_amt", "Unsharp amount", 0.0, 3.0, 0.1, "Edge contrast added to the depth."),
    ("usm_rad", "Unsharp radius", 0.3, 6.0, 0.1, "Small for fine lines, large for broad ridges."),
    ("hf_boost", "Fine detail boost", 0.0, 4.0, 0.1, "Lifts shallow micro-texture in the depth."),
    ("hf_rad", "Fine detail radius", 0.5, 8.0, 0.1, "The scale of the finest detail to lift."),
    ("clahe", "Local contrast (CLAHE)", 0.0, 8.0, 0.1, "0 off; 2–4 subtle."),
)


def _row(key, text, lo, hi, step, tip, value):
    r = ParamRow(Control(key, text, lo, hi, value, step, tip, unit=UNITS.get(key, "")))
    r.set_value(value)
    return r


class Model3D(QObject):
    """The viewer process, its shared values, and the controls for both."""

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.proc = None
        self.kind = ""
        self.state = None
        self.sharpen = dict(V3.SHARPEN_DEFAULTS)
        self.clay = "Grey"
        self.dialog = None
        self.timer = QTimer(self)
        self.timer.setInterval(150)
        self.timer.timeout.connect(self._tick)

    # ── state ─────────────────────────────────────────────────────────
    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def ensure_state(self):
        if self.state is None:
            self.state = V3.make_state(self.win.shared)
        return self.state

    def set(self, key, value):
        """A 3D setting: kept for the exports and sent to an open viewer."""
        cur = self.win.shared.get(key)
        self.win.shared[key] = int(value) if isinstance(cur, int) and not isinstance(cur, bool) \
            else float(value)
        if self.state is not None:
            V3.write(self.state, key, value)

    def push_light(self):
        """The Inspect light, as the classic dome sends it (screen frame)."""
        if self.state is None:
            return
        st = self.win.engine_settings.get(self.win.engine, {})
        for k, d in (("lx", 0.0), ("ly", 0.0), ("lz", 1.0)):
            V3.write(self.state, k, float(st.get(k, d)))

    def _tick(self):
        if self.alive():
            self.push_light()
        elif self.proc is not None:
            self.proc = None
            self.kind = ""
            self.timer.stop()
            self.refresh()

    # ── the viewer ────────────────────────────────────────────────────
    def open(self, kind: str = "standard"):
        from helio.application.render import composite, render
        p, D = self.win.project, self.win.render_data
        r = p.result
        if r is None or D is None:
            self.win.flash("Process the images first.")
            return
        if self.alive() and kind == self.kind == "standard":
            V3.write(self.state, "raise_3d", 1)            # the open window comes forward
            return
        self.close()
        d = self.win.display
        layer = d.layer if d.layer in ("Shading", "Combined Colour", "Roughness Fusion") \
            else "Combined Colour"

        def colour(eng, h, w, cc):
            st = dict(self.win.engine_settings.get(eng, {}))
            return composite(eng, render(eng, D, st, h, w, False, cc), D, st, h, w, False, cc,
                             view_mode=layer)
        ins = self.win.workspaces.get("Inspect")
        if ins is not None and hasattr(ins, "sync_to_window"):
            ins.sync_to_window()                 # it opens as the 3D panel is set
        try:
            inp = V3.prepare(kind, D, engine=self.win.engine, colour=colour,
                             depth_normalised=r.depth_normalised_full,
                             edge_trim_pct=float(self.win.shared.get("edge_trim", 2.0)),
                             pretrimmed=r.crop is not None)
            st = self.ensure_state()
            V3.sync(st, self.win.shared)
            V3.set_clay(st, self.clay, bool(self.win.shared.get("use_custom_clay", 0)))
            self.push_light()
            V3.calibrate(st, D.w, inp, float(p.scale_mm_per_px or 0.0),
                         float(p.z_mm_per_unit or 0.0))
            if kind == "heatmap":
                V3.write(st, "clay_blend", 0.0)
            self.proc = V3.launch(inp, st)
        except Exception as e:
            QMessageBox.warning(self.win, "3D viewer", f"Could not open the 3D viewer: {e}")
            return
        self.kind = kind
        self.timer.start()
        self.win.flash("The 3D viewer opens in its own window. It follows the light and the 3D "
                       "settings as you change them.")
        self.refresh()

    def close(self):
        if self.proc is not None:
            try:
                if self.proc.is_alive():
                    self.proc.terminate()
                    self.proc.join(timeout=2)
            except Exception:
                pass
        self.proc, self.kind = None, ""
        self.timer.stop()

    def record(self, mode: int):
        """A turntable (1 GIF, 2 MP4) or flyover (3 GIF, 4 MP4) of the open viewer."""
        if not self.alive():
            self.win.flash("Open the 3D viewer first.")
            return
        frames = int(self.frames.currentText()) if hasattr(self, "frames") else 72
        total = V3.start_spin(self.state, mode, frames)
        lbl, ext = V3.SPIN_MODES[mode]
        from helio_qt.publish import file_stem
        start = os.path.join(str(self.win.prefs.value("export_folder", "")) or
                             self.win.default_export_dir(),
                             f"{file_stem(self.win.project.name)}_{lbl.lower()}{ext}")
        path, _ = QFileDialog.getSaveFileName(self.win, f"Save the {lbl.lower()}", start,
                                              "GIF (*.gif)" if ext == ".gif" else "MP4 (*.mp4)")
        if not path:
            V3.write(self.state, "spin_capture", 0)
            return
        state, alive = self.state, self.alive

        def work():
            if not V3.wait_spin(state, total, alive=alive):
                raise TimeoutError("The viewer did not finish recording.")
            return V3.assemble_spin(mode, path)

        def done(res, err):
            V3.write(state, "auto_spin", 0)
            if hasattr(self, "chk_spin"):
                self.chk_spin.setChecked(False)
            if err or res is None:
                QMessageBox.warning(self.win, "Recording", str(err))
                return
            self.win.flash(f"{lbl} saved: {os.path.basename(res[0])} ({res[1]} frames).")
        self.win.flash(f"Recording a {lbl.lower()} of {total} frames…", 20)
        self.win.run_background(work, done)

    def clean_frame(self):
        if not self.alive():
            self.win.flash("Open the 3D viewer first, and set the model as you want it.")
            return
        from helio_qt.publish import file_stem
        start = os.path.join(str(self.win.prefs.value("export_folder", "")) or
                             self.win.default_export_dir(),
                             f"HELIO_3D_{file_stem(self.win.project.name)}.png")
        path, _ = QFileDialog.getSaveFileName(self.win, "Save a clean 3D frame", start,
                                              "PNG (*.png);;JPEG (*.jpg)")
        if not path:
            return
        try:
            V3.clean_render(self.state, path)
        except (TimeoutError, FileNotFoundError) as e:
            QMessageBox.warning(self.win, "Clean frame", str(e))
            return
        self.win.flash(f"Saved {os.path.basename(path)}.")

    # ── sharpening ────────────────────────────────────────────────────
    def preview_sharpen(self):
        """The sharpened normals on screen (Inspect), the solve unchanged."""
        import numpy as np
        r = self.win.project.result
        if r is None:
            return
        n, _d = V3.sharpen(r.proxy["normals"], None, self.sharpen)
        ins = self.win.workspaces["Inspect"]
        self.win.go_workspace("Inspect")
        rgb = np.clip((n * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        ins.viewport.show_array(rgb, (r.w, r.h))
        self.win.flash("Preview: the sharpened normals. Apply to keep them; any change of view "
                       "returns to the surface as it is.")

    def apply_sharpen(self):
        from helio.application.process_tools import with_sharpened
        p = self.win.project
        r = p.result
        if r is None:
            return
        params = dict(self.sharpen)
        reopen = self.kind if self.alive() else ""

        def done(out, err):
            p.busy = ""
            if err or out is None:
                p.finish(f"Sharpening stopped: {err}")
                return
            p.set_result(out, p.solve_seconds)
            self.win.flash("Sharpening applied to the normals and depth.")
            if reopen:
                self.close()
                self.open(reopen)
        p.set_busy("solve")
        self.win.run_background(lambda: with_sharpened(r, params), done)

    # ── the Inspect section ───────────────────────────────────────────
    def section(self, store) -> Section:
        """Inspect's short "3D model" section: the settings that are not on
        the 3D panel (detail sharpening, the export mesh, flips).  The 3D
        views themselves are the 3D relief and the high-detail window."""
        sec = Section("3D model", "model3d", store, open_=False)
        self.btn_settings = button("3D settings…", "outline", "Detail sharpening, the export "
                                   "mesh density, flips, clay and display tone.")
        self.btn_settings.clicked.connect(self.show_settings)
        sec.add(self.btn_settings)
        sec.add(label("Turntables and flyovers: the 3D relief's buttons, or the high-detail "
                      "3D window.", "caption", wrap=True))
        self.sec = sec
        return sec

    def refresh(self):
        ins = getattr(self.win, "workspaces", {}).get("Inspect")
        if ins is not None and hasattr(ins, "update_relief_panel"):
            ins.update_relief_panel()
        if hasattr(self, "btn_settings"):
            self.btn_settings.setEnabled(self.win.project.result is not None)

    # ── the settings window ───────────────────────────────────────────
    def show_settings(self):
        if self.dialog is None:
            self.dialog = self._build_dialog()
        self._sync_dialog()
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()

    def _from_dialog(self, key, value):
        """A change in this window: kept, sent to the viewer, and shown on
        Inspect's 3D panel and its relief, which edit the same values."""
        self.set(key, value)
        ins = self.win.workspaces.get("Inspect")
        if ins is not None and hasattr(ins, "shared_3d_changed"):
            ins.shared_3d_changed(key)

    def _sync_dialog(self):
        """Every control here as the shared values stand: Inspect's 3D panel
        may have changed them since this window was last open."""
        sh = self.win.shared
        for k, r in self.rows3d.items():
            if k in sh:
                r.set_value(sh[k])
        for k, (c, on_value) in self.check_boxes.items():
            c.blockSignals(True)
            c.setChecked(int(sh.get(k, 0)) == on_value)
            c.blockSignals(False)
        for w in (self.clay_combo, self.chk_clay):
            w.blockSignals(True)
        self.clay_combo.setCurrentText(self.clay)
        self.chk_clay.setChecked(bool(sh.get("use_custom_clay", 0)))
        for w in (self.clay_combo, self.chk_clay):
            w.blockSignals(False)

    def _build_dialog(self) -> QDialog:
        sh = self.win.shared
        dlg = QDialog(self.win)
        dlg.setWindowTitle("3D settings")
        dlg.setModal(False)
        dlg.resize(420, 760)
        outer = QVBoxLayout(dlg)
        body = QWidget()
        vl = QVBoxLayout(body)
        vl.setSpacing(12)
        outer.addWidget(VScroll(body), 1)
        self.rows3d = {}
        self.check_boxes = {}                     # key: (box, the shared value when ticked)
        store = self.win.prefs

        def group(title, key, specs, open_=True):
            s = Section(title, key, store, open_=open_)
            for k, t, lo, hi, step, tip in specs:
                r = _row(k, t, lo, hi, step, tip, sh.get(k, lo))
                r.changed.connect(lambda kk, v, _f: self._from_dialog(kk, v))
                s.add(r)
                self.rows3d[k] = r
            vl.addWidget(s)
            return s
        g = group("Mesh", "m3d_geo", GEOMETRY)
        flips = QHBoxLayout()
        flips.addWidget(label("Flip", "secondary"))
        self.flip_boxes = {}
        for axis, key, default, tip in (("X", "fx", 1, "Mirror left–right."),
                                        ("Y", "fy", -1, "Mirror top–bottom: if the model is upside down."),
                                        ("Z", "fz", 1, "Invert the relief.")):
            c = QCheckBox(axis)
            c.setToolTip(tip)
            c.setChecked(int(sh.get(key, default)) != default)
            c.toggled.connect(lambda on, k=key, d=default: self._from_dialog(k, -d if on else d))
            flips.addWidget(c)
            self.flip_boxes[key] = c
            self.check_boxes[key] = (c, -default)
        flips.addStretch(1)
        g.add_layout(flips)
        for key, text, tip in (("center_origin", "Centre the origin", "Put the mesh's centre at "
                                "(0, 0, 0): what most 3D software expects."),
                               ("bake_3d", "Bake the 2D view's shading onto the mesh", "The "
                                "current view becomes the vertex colour."),
                               ("hq_normals", "High-quality normals", "Shade from the full-"
                                "resolution normals: sharper, slower."),
                               ("hybrid_enabled", "Hybrid depth blend", "A shape layer plus a "
                                "detail layer, mixed below: for warped vellum and bindings.")):
            c = QCheckBox(text)
            c.setToolTip(tip)
            c.setChecked(bool(sh.get(key, 0)))
            c.toggled.connect(lambda on, k=key: self._from_dialog(k, 1 if on else 0))
            g.add(c)
            self.check_boxes[key] = (c, 1)
        r = _row("hybrid_mix", "Hybrid detail mix", 0.0, 1.0, 0.01,
                 "Share of the detail layer: 10–15 % is the archival range.", sh.get("hybrid_mix", 0.12))
        r.changed.connect(lambda kk, v, _f: self._from_dialog(kk, v))
        g.add(r)
        self.rows3d["hybrid_mix"] = r

        s = Section("Clay", "m3d_clay", store, open_=False)
        row = QHBoxLayout()
        self.chk_clay = QCheckBox("Custom clay colour")
        self.chk_clay.setChecked(bool(sh.get("use_custom_clay", 0)))
        self.clay_combo = QComboBox()
        self.clay_combo.addItems(list(V3.CLAY))
        self.clay_combo.setCurrentText(self.clay)

        def clay_changed(*_a):
            self.clay = self.clay_combo.currentText()
            self.win.shared["use_custom_clay"] = 1 if self.chk_clay.isChecked() else 0
            if self.state is not None:
                V3.set_clay(self.state, self.clay, self.chk_clay.isChecked())
        self.chk_clay.toggled.connect(clay_changed)
        self.clay_combo.currentTextChanged.connect(clay_changed)
        row.addWidget(self.chk_clay, 1)
        row.addWidget(self.clay_combo)
        s.add_layout(row)
        r = _row("clay_blend", "Colour → clay", 0.0, 1.0, 0.01, "0 the photograph's colour, 1 clay.",
                 sh.get("clay_blend", 0.0))
        r.changed.connect(lambda kk, v, _f: self.set(kk, v))
        s.add(r)
        self.rows3d["clay_blend"] = r
        vl.addWidget(s)
        group("Material", "m3d_mat", MATERIAL, open_=False)
        group("Display tone", "m3d_tone", TONE3D, open_=False)
        group("Mesh exports", "m3d_exp", EXPORT, open_=False)

        s = Section("Detail sharpening", "m3d_sharp", store, open_=False)
        self.sharp_rows = {}
        for k, t, lo, hi, step, tip in SHARPEN:
            r = _row(k, t, lo, hi, step, tip, self.sharpen[k])
            r.changed.connect(lambda kk, v, _f: self.sharpen.__setitem__(kk, v))
            s.add(r)
            self.sharp_rows[k] = r
        row = QHBoxLayout()
        b1 = button("Preview", "outline", "Show the sharpened normals in Inspect.")
        b1.clicked.connect(self.preview_sharpen)
        b2 = button("Apply to the surface", "", "Sharpen the normals and depth; the 3D viewer "
                    "reopens with them.")
        b2.clicked.connect(self.apply_sharpen)
        b3 = button("Reset", "quiet")

        def reset():
            self.sharpen = dict(V3.SHARPEN_DEFAULTS)
            for kk, rr in self.sharp_rows.items():
                rr.set_value(self.sharpen[kk])
        b3.clicked.connect(reset)
        for b in (b1, b2, b3):
            row.addWidget(b)
        s.add_layout(row)
        vl.addWidget(s)
        vl.addStretch(1)
        close = button("Done", "")
        close.clicked.connect(dlg.hide)
        outer.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        return dlg
