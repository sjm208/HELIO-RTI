"""The quick exports (EXP-010…012): the three sheets the Export panel opens.

RTI web package, Send to QGIS, USDZ for AR.  Each is filled in from the
project, says what it will write and where, and finishes with one button.
The work runs off the window's thread (MainWindow.run_background); the
sheet stays open with the outcome and a way to see the files.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout,
                               QLineEdit, QVBoxLayout, QWidget)

from helio_qt.controls import ParamRow, Segmented
from helio_qt.widgets import button, label

LICENCES = ("CC BY 4.0", "CC BY-SA 4.0", "CC BY-NC 4.0", "CC0 1.0 (public domain)",
            "All rights reserved")


def _folder_row(win, start: str, title: str):
    """A path with a Change… button: (widget, line edit)."""
    row = QWidget()
    lay = QHBoxLayout(row)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(6)
    edit = QLineEdit(start)
    edit.setAccessibleName(title)
    lay.addWidget(edit, 1)
    b = button("Change…", "", f"Choose where to write {title.lower()}")

    def pick():
        d = QFileDialog.getExistingDirectory(win, title, os.path.dirname(edit.text()) or start)
        if d:
            edit.setText(os.path.join(d, os.path.basename(edit.text().rstrip(os.sep))))
    b.clicked.connect(pick)
    lay.addWidget(b)
    return row, edit


def _sheet(win, title: str, lead: str) -> tuple[QDialog, QVBoxLayout]:
    d = QDialog(win)
    d.setWindowTitle(title)
    d.setMinimumWidth(540)
    lay = QVBoxLayout(d)
    lay.setContentsMargins(24, 20, 24, 16)
    lay.setSpacing(8)
    lay.addWidget(label(title, "title"))
    lay.addWidget(label(lead, "secondary", wrap=True))
    return d, lay


def _stem(name: str) -> str:
    from helio_qt.publish import file_stem
    return file_stem(name)


class RtiWebDialog:
    """RTI web package: the deep-zoom relighting viewer, a one-file quick view,
    the Relight / OpenLIME data and everything to put it online."""

    def __init__(self, win):
        self.win, self.p = win, win.project
        r = self.p.result
        self.d, lay = _sheet(win, "RTI web package",
                             "A folder that opens in any browser, offline: relight with the mouse, "
                             "zoom to single pixels, share as a zip or put it on a website.")
        self.title = QLineEdit(self.p.name)
        self.title.setAccessibleName("Title")
        lay.addWidget(label("Title", ""))
        lay.addWidget(self.title)
        grid = QHBoxLayout()
        self.obj = QLineEdit(self.p.capture_meta.get("object_id", "") if self.p.capture_meta else "")
        self.obj.setPlaceholderText("Object ID (optional)")
        self.obj.setAccessibleName("Object ID")
        self.licence = QComboBox()
        self.licence.addItems(LICENCES)
        self.licence.setAccessibleName("Licence")
        grid.addWidget(self.obj, 1)
        grid.addWidget(self.licence, 1)
        lay.addLayout(grid)
        who = QHBoxLayout()
        self.creator = QLineEdit(str(win.prefs.value("rti_web/creator", "")))
        self.creator.setPlaceholderText("Creator (optional)")
        self.creator.setAccessibleName("Creator")
        self.inst = QLineEdit(str(win.prefs.value("rti_web/institution", "")))
        self.inst.setPlaceholderText("Institution (optional)")
        self.inst.setAccessibleName("Institution")
        who.addWidget(self.creator, 1)
        who.addWidget(self.inst, 1)
        lay.addLayout(who)
        self.url = QLineEdit()
        self.url.setPlaceholderText("Web address it will live at (optional: for the QR code and embed code)")
        self.url.setAccessibleName("Web address")
        lay.addWidget(self.url)
        lay.addWidget(label("Size", ""))
        self.size = Segmented([("0", "Full size"), ("4096", "4096 px"), ("2048", "2048 px")],
                              {"0": "Every pixel the solve has", "4096": "Long side 4096 px",
                               "2048": "Long side 2048 px: small and quick to load"})
        self.size.set_current("0")
        lay.addWidget(self.size)
        self.chk_quick = QCheckBox("A one-file quick view for e-mail (under 25 MB)")
        self.chk_quick.setChecked(True)
        self.chk_relight = QCheckBox("Relight / OpenLIME data, with the frames to fit again")
        self.chk_relight.setChecked(r is not None and (getattr(r, "ptm_coeffs", None) is not None
                                                        or getattr(r, "ptm_deferred", False)))
        self.chk_zip = QCheckBox("Zip the folder as well")
        for c in (self.chk_quick, self.chk_relight, self.chk_zip):
            lay.addWidget(c)
        base = str(win.prefs.value("export_folder", "")) or win.default_export_dir()
        row, self.out = _folder_row(win, os.path.join(base, f"{_stem(self.p.name)} — web"),
                                    "The package folder")
        lay.addWidget(row)
        self.est = label("", "caption", wrap=True)
        lay.addWidget(self.est)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.b_cancel = button("Close", "")
        self.b_cancel.clicked.connect(self.d.reject)
        self.b_show = button("Show in Finder", "")
        self.b_show.hide()
        self.b_go = button("Export package", "primary")
        self.b_go.clicked.connect(self.run)
        for b in (self.b_cancel, self.b_show, self.b_go):
            btns.addWidget(b)
        lay.addLayout(btns)
        self.size.chosen.connect(lambda _k: self._estimate())
        self.chk_quick.toggled.connect(lambda _o: self._estimate())
        self.chk_relight.toggled.connect(lambda _o: self._estimate())
        self._estimate()
        self.result = None

    def _options(self):
        from helio.application.rti_web import Options
        return Options(title=self.title.text().strip() or self.p.name, obj_id=self.obj.text().strip(),
                       licence=self.licence.currentText(), creator=self.creator.text().strip(),
                       institution=self.inst.text().strip(), url=self.url.text().strip(),
                       quick_view=self.chk_quick.isChecked(), relight_data=self.chk_relight.isChecked(),
                       zip=self.chk_zip.isChecked(), max_side=int(self.size.current() or 0))

    def _estimate(self):
        from helio.application.rti_web import estimate
        r = self.p.result
        e = estimate(r.w, r.h, self._options())
        self.est.setText(f"About {e['mb']:.0f} MB in {e['files']} files · {e['width']} × {e['height']} px"
                         + ("" if self.p.scale_mm_per_px else
                            " · no scale set, so the viewer has no scale bar (Measure › Set Scale)"))

    def run(self):
        from helio.application import rti_web
        from helio.application.export_web import app_like
        win, p, r = self.win, self.p, self.p.result
        opts = self._options()
        out = self.out.text().strip()
        if not out:
            return
        win.prefs.setValue("rti_web/creator", opts.creator)
        win.prefs.setValue("rti_web/institution", opts.institution)
        self.b_go.setEnabled(False)
        self.status.setText("Writing the package…")
        src = win.publish_source()
        prov = app_like(name=p.name, images=src.images, lights=src.lights,
                        light_kind=src.light_source, h=r.h, w=r.w)
        solve = dict(normals=r.settings.resolved_normal_engine(),
                     height=r.settings.resolved_depth_engine(), photos=r.settings.photo_tone)

        want_relight = self.chk_relight.isChecked()

        def work():
            from helio.application.pipeline import ensure_ptm_hsh, region_fraction
            if want_relight:
                ensure_ptm_hsh(r)
            return rti_web.package(out, normals=r.normals_full, albedo=r.albedo_rgb_full,
                                   lights=src.lights, ptm_coeffs=getattr(r, "ptm_coeffs", None),
                                   roughness=getattr(r, "roughness_full", None),
                                   spec_var=getattr(r, "spec_var", None), images=src.images,
                                   mm_per_px=p.scale_mm_per_px, z_mm_per_unit=p.z_mm_per_unit,
                                   solve=solve, opts=opts, provenance_app=prov,
                                   region=region_fraction(r.crop, r.base_w, r.base_h))

        def done(res, err):
            self.b_go.setEnabled(True)
            if err:
                self.status.setText(f"Not finished: {err}")
                return
            self.result = res
            p.last_export_dir = res["dir"]
            extra = " · " + " ".join(res["notes"]) if res["notes"] else ""
            self.status.setText(f"Written: {res['files']} files, {res['mb']:.1f} MB, in {res['seconds']:.1f} s"
                                f"{extra}.  To look at it now, open quick-view.html, or double-click "
                                f"1_Mac.command in the folder.")
            self.b_show.show()
            self.b_show.clicked.connect(lambda: win.reveal(res["dir"]))
            win.flash(f"RTI web package written: {os.path.basename(res['dir'])}.")
        win.run_background(work, done)

    def exec(self):
        return self.d.exec()


class QgisDialog:
    """Send to QGIS: the GIS package, then QGIS with Helio Relief."""

    def __init__(self, win, profiles_path: str | None = None):
        from helio.application import export_qgis as Q
        self.win, self.p, self.Q = win, win.project, Q
        self.profiles_path = profiles_path
        self.d, lay = _sheet(win, "Send to QGIS",
                             "Opens this surface in QGIS at the object's true scale, with your "
                             "Helio Relief plugin when it is installed.")
        self.qgis_all = Q.find_qgis()
        self.qgis = Q.choose(self.qgis_all, profiles_path)
        sp = Q.plugin_path(self.qgis["major"], profiles_path) if self.qgis else None
        self.plugin = Q.plugin_state(self.qgis["major"], profiles_path) if self.qgis else {}
        rows = []
        if self.qgis:
            rows.append(("ok", f"QGIS {self.qgis['version']}", os.path.basename(self.qgis["app"])))
        else:
            rows.append(("warn", "QGIS not found", "in Applications. The package is still written."))
        if sp:
            old = self.plugin.get("install")
            rows.append(("warn" if old else "ok", f"Helio Relief {Q.plugin_version(sp)}",
                         (f"installed, but this Helio ships {self.plugin['bundled']}" if old
                          else "installed: it will import the package")))
        else:
            rows.append(("info", "Helio Relief not found",
                         "the layers will be loaded plainly until it is installed"))
        for kind, a, b in rows:
            r = label(f"{'✓' if kind == 'ok' else '•'}  {a} — {b}", "secondary", wrap=True)
            r.setProperty("status", "ok" if kind == "ok" else ("warn" if kind == "warn" else "info"))
            lay.addWidget(r)
        mm = self.p.scale_mm_per_px
        zmm = self.p.z_mm_per_unit
        lay.addWidget(label(
            f"Pixel size: {mm:.4g} mm, from the ruler" if mm else
            "Pixel size: not measured, so the package says 1 mm per pixel (Measure › Set Scale)",
            "caption", wrap=True))
        z = label("Heights: " + (f"measured, {zmm * 1000:.4g} µm per pixel of height" if zmm else
                                 "relative: no height scale was measured. The README says so, "
                                 "so nobody reads them as millimetres."), "caption", wrap=True)
        if not zmm:
            z.setProperty("status", "warn")
        lay.addWidget(z)
        lay.addWidget(label("Layers: the height from two integrations (Poisson and "
                            "Frankot-Chellappa) as 32-bit GeoTIFFs with world files, the colour, "
                            "the normals, the uncertainty, any metal mask, and the README the "
                            "plugin reads.", "caption", wrap=True))
        base = str(win.prefs.value("export_folder", "")) or win.default_export_dir()
        row, self.out = _folder_row(win, os.path.join(base, f"{_stem(self.p.name)} — QGIS"),
                                    "The package folder")
        lay.addWidget(row)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        btns = QHBoxLayout()
        btns.addStretch(1)
        b_close = button("Close", "")
        b_close.clicked.connect(self.d.reject)
        self.b_install = button(f"Install Helio Relief {self.plugin.get('bundled', '')}", "")
        self.b_install.setToolTip("Puts the plugin that came with Helio into QGIS's plugins folder. "
                                  "Quit QGIS first if it is open.")
        self.b_install.clicked.connect(self.install_plugin)
        self.b_install.setVisible(bool(self.plugin.get("install")))
        self.b_save = button("Save the package only", "")
        self.b_save.clicked.connect(lambda: self.run(open_qgis=False))
        self.b_go = button("Open in QGIS", "primary")
        self.b_go.setEnabled(self.qgis is not None)
        self.b_go.clicked.connect(lambda: self.run(open_qgis=True))
        for b in (b_close, self.b_install, self.b_save, self.b_go):
            btns.addWidget(b)
        lay.addLayout(btns)
        self.result = None
        self.launched = None

    def install_plugin(self):
        try:
            self.Q.install_plugin(self.qgis["major"], self.profiles_path)
        except OSError as e:
            self.status.setText(f"Could not install it: {e}")
            return
        self.plugin = self.Q.plugin_state(self.qgis["major"], self.profiles_path)
        self.b_install.setVisible(bool(self.plugin.get("install")))
        self.status.setText(f"Helio Relief {self.plugin['version']} is installed. If QGIS is open, "
                            "quit it first so it loads this copy.")

    def run(self, open_qgis: bool = True, popen=None):
        from helio.application.export_web import app_like
        win, p, r = self.win, self.p, self.p.result
        out = self.out.text().strip()
        if not out:
            return
        for b in (self.b_go, self.b_save):
            b.setEnabled(False)
        self.status.setText("Writing the package…")
        src = win.publish_source()
        D = win.render_data
        prov = app_like(name=p.name, images=src.images, lights=src.lights,
                        light_kind=src.light_source, h=r.h, w=r.w)

        def work():
            return self.Q.package(D, out, p.name, mm_per_px=p.scale_mm_per_px,
                                  z_mm_per_unit=p.z_mm_per_unit,
                                  normal_engine=r.settings.resolved_normal_engine(),
                                  lights=src.lights, provenance_app=prov)

        def done(res, err):
            self.b_save.setEnabled(True)
            self.b_go.setEnabled(self.qgis is not None)
            if err:
                self.status.setText(f"Not finished: {err}")
                return
            self.result = res
            p.last_export_dir = res["folder"]
            msg = f"Written to {os.path.basename(res['folder'])}."
            if open_qgis and self.qgis is not None:
                try:
                    self.launched = self.Q.launch(self.qgis, res["script"], self.profiles_path, popen)
                    msg += f" Opening QGIS {self.qgis['version']}…"
                except OSError as e:
                    msg += f" QGIS did not start: {e}."
            self.status.setText(msg)
            win.flash(msg)
        win.run_background(work, done)

    def exec(self):
        return self.d.exec()


class UsdzDialog:
    """USDZ for AR: one OpenUSD file at the object's true size."""

    def __init__(self, win):
        from helio.domain.view_controls import Control
        self.win, self.p = win, win.project
        self.d, lay = _sheet(win, "USDZ for AR",
                             "One OpenUSD file. On an iPhone or iPad it opens in Quick Look and sits "
                             "on the table, or hangs on the wall, at its real size.")
        mm = self.p.scale_mm_per_px
        r = self.p.result
        if mm:
            lay.addWidget(label(f"Real size: {r.w * mm:.1f} × {r.h * mm:.1f} mm, from the ruler",
                                "secondary"))
        else:
            n = label("No scale is set, so it will be 200 mm across. Measure › Set Scale makes it "
                      "its true size.", "secondary", wrap=True)
            n.setProperty("status", "warn")
            lay.addWidget(n)
        self.exag = ParamRow(Control("exag", "Height exaggeration", 1.0, 20.0, 1.0, 0.5,
                                     "1 is the true shape (the height map scaled to agree with the "
                                     "normals). Raise it to see shallow marks in AR."))
        lay.addWidget(self.exag)
        lay.addWidget(label("Place it on", ""))
        self.place = Segmented([("table", "A table"), ("wall", "A wall")],
                               {"table": "Flat objects: it lies on the floor or a table",
                                "wall": "Paintings, panels: it hangs on a wall"})
        self.place.set_current("table")
        lay.addWidget(self.place)
        self.chk_usda = QCheckBox("Also save a .usda scene with its lights (Blender, Houdini, Omniverse)")
        lay.addWidget(self.chk_usda)
        lay.addWidget(label("The colour photograph is its colour and the photometric normals its "
                            "normal map, so the detail stays sharp on a light mesh. The capture's "
                            "lights and how it was made travel inside the file.", "caption", wrap=True))
        base = str(win.prefs.value("export_folder", "")) or win.default_export_dir()
        self.path = os.path.join(base, f"{_stem(self.p.name)}.usdz")
        row, self.out = _folder_row(win, self.path, "The USDZ file")
        lay.addWidget(row)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        btns = QHBoxLayout()
        btns.addStretch(1)
        b_close = button("Close", "")
        b_close.clicked.connect(self.d.reject)
        self.b_go = button("Export USDZ", "primary")
        self.b_go.clicked.connect(self.run)
        btns.addWidget(b_close)
        btns.addWidget(self.b_go)
        lay.addLayout(btns)
        self.result = None

    def run(self):
        from helio.application import export_usd as U
        win, p, r = self.win, self.p, self.p.result
        path = self.out.text().strip()
        if not path:
            return
        if not path.lower().endswith(".usdz"):
            path += ".usdz"
        self.b_go.setEnabled(False)
        self.status.setText("Writing the USDZ…")
        src = win.publish_source()
        from helio.domain.view_controls import light_of
        lx, ly, lz = light_of(win.engine_settings.get(win.engine, {}), win.engine)
        try:
            from helio_qt import __version__ as v
        except Exception:
            v = ""
        meta = dict(object=p.name, helio=f"Helio {v}".strip(),
                    normals=r.settings.resolved_normal_engine(),
                    height=r.settings.resolved_depth_engine())
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        def work():
            out = U.export(path, depth=r.depth_vis_p, normals=r.normals_full, albedo=r.albedo_rgb_full,
                           mm_per_px=p.scale_mm_per_px, exaggeration=self.exag.value(),
                           wall=self.place.current() == "wall", lights=src.lights,
                           view_light=(lx, -ly, lz), meta=meta, also_usda=self.chk_usda.isChecked())
            ok, problems = U.check(path)
            out["check"] = (ok, problems)
            return out

        def done(res, err):
            self.b_go.setEnabled(True)
            if err:
                self.status.setText(f"Not finished: {err}")
                return
            self.result = res
            ok, problems = res["check"]
            verdict = ("Checked against Apple's AR rules: passes." if ok and not problems else
                       "Checked against Apple's AR rules: passes (" + "; ".join(problems) + ")."
                       if ok else "Apple's AR rules found problems: " + "; ".join(problems[:3]))
            self.status.setText(f"Written {os.path.basename(path)}: {res['vertices']:,} points, "
                                f"{res['mb']:.1f} MB, {res['width_mm']:.0f} mm across. {verdict} "
                                + " ".join(res["notes"]))
            win.flash(f"USDZ written: {os.path.basename(path)}.")
        win.run_background(work, done)

    def exec(self):
        return self.d.exec()


class ChartGrid(QWidget):
    """The 24 patches: each the photograph's colour above the chart's, with ΔE₀₀."""

    def __init__(self, result: dict):
        super().__init__()
        self.res = result
        self.setMinimumSize(480, 300)
        self.setAccessibleName("ColorChecker patches, measured above reference")

    def paintEvent(self, _e):
        import numpy as np
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QPainter, QPen
        from helio.application.colour_check import lab_d50_to_srgb
        from helio_qt import theme
        t = theme.CURRENT
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        cols, rows, gap = 6, 4, 6
        cw = (self.width() - gap * (cols + 1)) / cols
        ch = (self.height() - gap * (rows + 1)) / rows
        for row in self.res["rows"]:
            i = row["n"] - 1
            r, c = divmod(i, cols)
            x, y = gap + c * (cw + gap), gap + r * (ch + gap)
            meas = lab_d50_to_srgb(np.array(row["measured"]))
            ref = lab_d50_to_srgb(np.array(row["ref"]))
            p.fillRect(QRectF(x, y, cw, ch * 0.5), QColor.fromRgbF(*[float(v) for v in meas]))
            p.fillRect(QRectF(x, y + ch * 0.5, cw, ch * 0.5), QColor.fromRgbF(*[float(v) for v in ref]))
            de = row["de"]
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(QRectF(x + 4, y + ch - 22, 52, 18), 4, 4)
            p.setPen(QColor(t.warn if de > 3 else "#FFFFFF"))
            p.drawText(QRectF(x + 4, y + ch - 22, 52, 18), Qt.AlignmentFlag.AlignCenter, f"{de:.1f}")
            p.setPen(QPen(QColor(t.line), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x, y, cw, ch))
        p.end()


class ColourCheckDialog:
    """The colour accuracy of a photograph, against a ColorChecker (MET-010)."""

    def __init__(self, win, result: dict):
        self.win, self.res = win, result
        name = os.path.basename(result.get("path", ""))
        self.d, lay = _sheet(win, "Colour accuracy",
                             f"{name}: the ColorChecker found and read, each patch against the "
                             f"chart's published colour (CIEDE2000, D50).")
        head = label(f"Average ΔE₀₀ {result['mean']:.2f}  ·  median {result['median']:.2f}  ·  "
                     f"90% under {result['p90']:.2f}  ·  worst {result['max']:.2f} ({result['worst']})",
                     "section", wrap=True)
        lay.addWidget(head)
        v = label(result["verdict"] + ".", "secondary", wrap=True)
        v.setProperty("status", "ok" if result["mean"] <= 2 else "warn")
        lay.addWidget(v)
        lay.addWidget(label(f"Grey steps: lightness off by {result['grey_mean_dL']:.2f} L* on average "
                            f"({', '.join(f'{x:+.1f}' for x in result['grey_dL'])}, white to black).",
                            "caption", wrap=True))
        lay.addWidget(label("Each square: the photograph above, the chart below. Read as "
                            f"{result['encoding']}. The chart's cyan lies outside sRGB, so an sRGB "
                            "photograph cannot reach it exactly.", "caption", wrap=True))
        lay.addWidget(ChartGrid(result), 1)
        btns = QHBoxLayout()
        btns.addStretch(1)
        b_csv = button("Save as CSV…", "", "Every patch: reference and measured L*a*b*, and ΔE₀₀")
        b_csv.clicked.connect(self.save_csv)
        b_close = button("Close", "primary")
        b_close.clicked.connect(self.d.accept)
        btns.addWidget(b_csv)
        btns.addWidget(b_close)
        lay.addLayout(btns)
        self.d.resize(640, 640)

    def save_csv(self, path: str = ""):
        from helio.application.colour_check import write_csv
        if not path:
            base = str(self.win.prefs.value("export_folder", "")) or self.win.default_export_dir()
            stem = os.path.splitext(os.path.basename(self.res.get("path", "chart")))[0]
            path, _ = QFileDialog.getSaveFileName(self.d, "Save the colour check",
                                                  os.path.join(base, f"{stem}_colour_check.csv"),
                                                  "CSV (*.csv)")
            if not path:
                return ""
        write_csv(self.res, path)
        self.win.flash(f"Saved {os.path.basename(path)}.")
        return path

    def exec(self):
        return self.d.exec()
