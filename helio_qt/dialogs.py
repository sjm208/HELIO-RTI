"""Dialogs that edit project details (IMP-003, MET-012)."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLineEdit,
                               QMessageBox, QVBoxLayout, QWidget)

from helio_qt.controls import Section, VScroll
from helio_qt.widgets import button, label


class CaptureDetailsDialog(QDialog):
    """Who captured what, with which camera and lights; and the measured
    positions of the lights (points of interest).  Saved with the session
    and written into the provenance record and the reports."""

    def __init__(self, win):
        from helio.domain.capture_meta import FIELDS
        super().__init__(win)
        self.win, self.project = win, win.project
        self.setWindowTitle("Capture details")
        self.setMinimumSize(560, 640)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(8)
        lay.addWidget(label("Capture details", "title"))
        lay.addWidget(label("Saved with the session, and written into the processing record and "
                            "every report.", "secondary", wrap=True))
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 6, 0)
        bl.setSpacing(12)
        self.edits: dict[str, QLineEdit] = {}
        meta = dict(self.project.capture_meta)
        if not meta.get("img_count"):
            meta["img_count"] = str(len(self.project.images))
        if self.project.scale_mm_per_px and not meta.get("scale_mmpx"):
            meta["scale_mmpx"] = f"{self.project.photo_scale():.6f}"      # the photographs' pixels
        self.meta = meta
        # The object profile first: it decides the discipline fields below it.
        from helio.domain.object_profiles import PROFILES, of
        sec = Section("Object type", "meta_object_type", win.prefs)
        self.kind = QComboBox()
        self.kind.setAccessibleName("Object type")
        for p in PROFILES.values():
            self.kind.addItem(p.name, p.key)
        self.kind.setCurrentIndex(max(0, self.kind.findData(of(meta).key)))
        self.kind.setToolTip("What kind of object this is. Helio sets a solver for it, puts its "
                             "views first in Inspect, offers its exports in Publish and adds its "
                             "fields here. Nothing else is hidden or changed.")
        sec.add(self.kind)
        self.kind_blurb = label("", "caption", wrap=True)
        sec.add(self.kind_blurb)
        bl.addWidget(sec)
        self.disc_sec = Section("Object", "meta_discipline", win.prefs)
        self.disc_form = QWidget()
        self.disc_sec.add(self.disc_form)
        bl.addWidget(self.disc_sec)
        self.disc_edits: dict[str, QLineEdit] = {}
        self.kind.currentIndexChanged.connect(self._kind_changed)
        for group, fields in FIELDS:
            sec = Section(group, f"meta_{group}", win.prefs)
            form = QFormLayout()
            form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
            for text, key, hint, tip in fields:
                e = QLineEdit(str(meta.get(key, "")))
                e.setPlaceholderText(hint)
                e.setAccessibleName(text)
                if tip:
                    e.setToolTip(tip)
                form.addRow(text, e)
                self.edits[key] = e
            w = QWidget()
            w.setLayout(form)
            sec.add(w)
            bl.addWidget(sec)
        sec = Section("Points of interest (3D positions)", "meta_poi", win.prefs, open_=False)
        sec.add(label("Measured positions of the lights, camera and origin, in millimetres, for "
                      "metrological traceability. A CSV of name, X, Y, Z.", "caption", wrap=True))
        self.poi_note = label("", "secondary", wrap=True)
        sec.add(self.poi_note)
        row = QHBoxLayout()
        b1 = button("Load CSV…", "")
        b1.clicked.connect(self.load_poi)
        b2 = button("Export…", "")
        b2.clicked.connect(self.export_poi)
        self.btn_poi_export = b2
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch(1)
        sec.add_layout(row)
        bl.addWidget(sec)
        bl.addStretch(1)
        lay.addWidget(VScroll(body), 1)
        btns = QHBoxLayout()
        for text, fn in (("Defaults", self.reset), ("Load JSON…", self.load_json),
                         ("Save JSON…", self.save_json), ("Export CSV…", self.export_csv)):
            b = button(text, "quiet")
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        cancel = button("Cancel", "")
        cancel.clicked.connect(self.reject)
        ok = button("Save", "primary")
        ok.clicked.connect(self.save)
        btns.addWidget(cancel)
        btns.addWidget(ok)
        lay.addLayout(btns)
        self._kind_changed()
        self._refresh_poi()

    def _kind_changed(self, *_):
        """Show the chosen profile's own fields, keeping what was typed in any field."""
        from helio.domain.object_profiles import get
        for k, e in self.disc_edits.items():
            self.meta[k] = e.text().strip()
        p = get(self.kind.currentData())
        self.kind_blurb.setText(p.blurb)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.disc_edits = {}
        for text, key, hint, tip in p.fields:
            e = QLineEdit(str(self.meta.get(key, "")))
            e.setPlaceholderText(hint)
            e.setAccessibleName(text)
            if tip:
                e.setToolTip(tip)
            form.addRow(text, e)
            self.disc_edits[key] = e
        old = self.disc_form
        self.disc_form = QWidget()
        self.disc_form.setLayout(form)
        self.disc_sec.body_lay.replaceWidget(old, self.disc_form)
        old.deleteLater()
        self.disc_sec.setVisible(bool(p.fields))
        self.disc_sec.button.setText(p.name.upper())
        self.disc_sec.button.setAccessibleName(p.name)

    def values(self) -> dict:
        """The details as typed.  Discipline fields of another profile, typed earlier, are kept:
        changing the object type back and forth loses nothing."""
        from helio.domain.object_profiles import all_field_keys
        out = {k: v for k, v in self.meta.items() if k in all_field_keys() and str(v).strip()}
        for d in (self.edits, self.disc_edits):
            for k, e in d.items():
                if e.text().strip():
                    out[k] = e.text().strip()
                else:
                    out.pop(k, None)
        kind = str(self.kind.currentData() or "general")
        if kind != "general":
            out["object_type"] = kind
        return out

    def save(self):
        before = str(dict(self.project.capture_meta).get("object_type") or "general")
        self.project.capture_meta = self.values()
        after = str(self.project.capture_meta.get("object_type") or "general")
        if after != before and hasattr(self.win, "apply_object_profile"):
            self.win.apply_object_profile(after)
        self.project.changed.emit()
        self.accept()

    def reset(self):
        from helio.domain.capture_meta import defaults
        d = defaults(len(self.project.images))
        for k, e in self.edits.items():
            e.setText(d.get(k, ""))

    def load_json(self):
        from helio.domain.capture_meta import read_json
        path, _ = QFileDialog.getOpenFileName(self, "Load capture details", "", "JSON (*.json)")
        if not path:
            return
        try:
            d = read_json(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Could not load", str(e))
            return
        for k, e in self.edits.items():
            if k in d:
                e.setText(d[k])

    def _stem(self):
        from helio_qt.publish import file_stem
        return file_stem(self.project.name)

    def save_json(self):
        from helio.domain.capture_meta import write_json
        path, _ = QFileDialog.getSaveFileName(self, "Save capture details",
                                              f"{self._stem()}_metadata.json", "JSON (*.json)")
        if path:
            write_json(path, self.values())

    def export_csv(self):
        from helio.domain.capture_meta import write_csv
        path, _ = QFileDialog.getSaveFileName(self, "Export capture details",
                                              f"{self._stem()}_metadata.csv", "CSV (*.csv)")
        if path:
            write_csv(path, self.values())

    def _refresh_poi(self):
        poi = self.project.poi
        self.poi_note.setText(f"{len(poi)} points loaded" + (f" ({self.project.poi_source})"
                                                             if self.project.poi_source else "")
                              if poi else "No points loaded.")
        self.btn_poi_export.setEnabled(bool(poi))

    def load_poi(self):
        from helio.domain.capture_meta import read_poi
        path, _ = QFileDialog.getOpenFileName(self, "Points of interest (name, X, Y, Z in mm)", "",
                                              "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            self.project.poi = read_poi(path)
            self.project.poi_source = os.path.basename(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Could not load the points", str(e))
        self._refresh_poi()

    def export_poi(self):
        from helio.domain.capture_meta import write_poi
        path, _ = QFileDialog.getSaveFileName(self, "Export points of interest",
                                              f"{self._stem()}_poi_positions.csv", "CSV (*.csv)")
        if path:
            write_poi(path, self.project.poi, self.project.poi_source)


class QrDialog(QDialog):
    """A QR code for a viewer a phone can open: on screen to scan, and saved
    on a card naming the object, its institution and scale
    (helio.application.export_web.qr_png).  The address is tidied first
    (helio.application.phone_share.tidy_url), and a web export can be shared
    from this Mac to phones on the same Wi-Fi."""

    def __init__(self, win):
        from PySide6.QtWidgets import QCheckBox
        super().__init__(win)
        self.win = win
        self.setWindowTitle("QR code")
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.addWidget(label("QR code", "title"))
        lay.addWidget(label("Scan it with a phone, or save it for a label or a catalogue.",
                            "secondary", wrap=True))
        self.url = QLineEdit()
        self.url.setPlaceholderText("https://your-site.org/viewer/")
        self.url.setAccessibleName("Address to encode")
        self.url.textChanged.connect(self._preview)
        lay.addWidget(self.url)
        self.code = label("", "caption")
        self.code.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.code.setMinimumHeight(240)
        lay.addWidget(self.code)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        # From this Mac: a web export served to the phones on the same Wi-Fi.
        lay.addWidget(label("On a phone, from this Mac", "section"))
        lay.addWidget(label("For an RTI web package or web viewer you exported. The phone "
                            "must be on the same Wi-Fi; allow incoming connections if macOS "
                            "asks.", "caption", wrap=True))
        srow = QHBoxLayout()
        self.btn_share = button("Share a web export…", "", "Choose the exported folder (the "
                                "one with index.html); its address goes in the code.")
        self.btn_share.clicked.connect(lambda: self.share())
        self.btn_stop = button("Stop sharing", "quiet", "Phones can no longer open it.")
        self.btn_stop.clicked.connect(self.stop_share)
        srow.addWidget(self.btn_share)
        srow.addWidget(self.btn_stop)
        srow.addStretch(1)
        lay.addLayout(srow)
        self.chk = QCheckBox("Printable label (object, institution, scale, Helio build)")
        self.chk.setChecked(True)
        self.chk.setToolTip("A bare code pinned beside an object tells whoever finds it nothing "
                            "about what it points at. Blank capture details are left out.")
        lay.addWidget(self.chk)
        row = QHBoxLayout()
        row.addStretch(1)
        close = button("Close", "outline")
        close.clicked.connect(self.reject)
        save = button("Save QR code…", "")
        save.clicked.connect(lambda: self.save())
        row.addWidget(close)
        row.addWidget(save)
        lay.addLayout(row)
        sh = getattr(win, "phone_share", None)
        if sh is not None and sh.server is not None:
            self.url.setText(sh.url)
        self._preview()

    def address(self) -> tuple[str, str]:
        from helio.application.phone_share import tidy_url
        return tidy_url(self.url.text())

    def _preview(self, *_):
        """The code for the address as typed, tidied, on screen to scan."""
        from PySide6.QtGui import QImage, QPixmap
        url, problem = self.address()
        sh = getattr(self.win, "phone_share", None)
        sharing = sh is not None and sh.server is not None
        self.btn_stop.setVisible(sharing)
        if not self.url.text().strip():
            self.code.setPixmap(QPixmap())
            self.code.setText("Enter an address, or share a web export from this Mac.")
            self.status.setText("")
            return
        if problem and not problem.startswith("localhost"):
            self.code.setPixmap(QPixmap())
            self.code.setText("")
            self.status.setText(problem)
            return
        try:
            import numpy as np
            import qrcode
            qr = qrcode.QRCode(version=1, box_size=6, border=4)
            qr.add_data(url)
            qr.make(fit=True)
            a = np.ascontiguousarray(np.asarray(qr.make_image(fill_color="black",
                                                              back_color="white").convert("L")))
        except Exception as e:
            self.status.setText(f"No QR code: {e}")
            return
        img = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format.Format_Grayscale8)
        pm = QPixmap.fromImage(img.copy())
        side = min(240, max(pm.width(), 120))
        self.code.setPixmap(pm.scaled(side, side, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.FastTransformation))
        note = problem if problem else f"Opens {url}"
        if sharing and url == sh.url:
            note = f"Shared from this Mac: {url}. Scan it with the phone's camera."
        self.status.setText(note)

    def share(self, folder: str = "") -> str:
        """Serve *folder* (asked for, starting at the last export) to the
        network and put its address in the code."""
        from helio.application.phone_share import Share, web_folder
        p = self.win.project
        if not folder:
            start = web_folder(p.last_export_dir) or p.last_export_dir \
                or str(self.win.prefs.value("export_folder", "")) or self.win.default_export_dir()
            folder = QFileDialog.getExistingDirectory(self, "The web export to share", start)
            if not folder:
                return ""
        folder = web_folder(folder) or folder
        self.stop_share()
        try:
            self.win.phone_share = Share(folder)
        except (OSError, RuntimeError) as e:
            self.status.setText(str(e))
            return ""
        if not os.path.exists(os.path.join(folder, "index.html")):
            self.status.setText("That folder has no index.html: the phone will see a list of "
                                "its files.")
        self.url.setText(self.win.phone_share.url)
        self.win.flash(f"Sharing {os.path.basename(folder)} on the network.")
        return self.win.phone_share.url

    def stop_share(self):
        sh = getattr(self.win, "phone_share", None)
        if sh is not None:
            sh.stop()
            self.win.phone_share = None
        self._preview()

    def save(self, path: str = "") -> str:
        from helio.application.export_web import qr_png
        from helio_qt.publish import file_stem
        url, problem = self.address()
        if problem and not problem.startswith("localhost"):
            self.status.setText(problem)
            return ""
        p = self.win.project
        if not path:
            start = os.path.join(str(self.win.prefs.value("export_folder", "")) or
                                 self.win.default_export_dir(), f"{file_stem(p.name)}_qr.png")
            path, _ = QFileDialog.getSaveFileName(self, "Save the QR code", start, "PNG (*.png)")
            if not path:
                return ""
        meta = dict(p.capture_meta)
        scale = meta.get("scale_mmpx") or (f"{p.photo_scale():.6f}" if p.scale_mm_per_px else "")
        try:
            qr_png(url, path, label=self.chk.isChecked(),
                   obj=(meta.get("title") or p.name if p.images else "").strip(),
                   institution=str(meta.get("institution", "") or "").strip(), scale=str(scale))
        except (RuntimeError, ValueError, OSError) as e:
            self.status.setText(str(e))
            return ""
        self.status.setText(f"Saved {os.path.basename(path)}: it opens {url}.")
        self.win.flash(f"QR code saved: {os.path.basename(path)}.")
        return path


class EmbedDialog(QDialog):
    """The HTML to put a hosted viewer in a web page (export_web.embed_snippet)."""

    def __init__(self, win):
        from PySide6.QtWidgets import QPlainTextEdit
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Embed snippet")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.addWidget(label("Embed snippet", "title"))
        form = QFormLayout()
        self.url = QLineEdit("https://your-site.com/viewer/index.html")
        self.w = QLineEdit("100%")
        self.h = QLineEdit("600px")
        form.addRow("Viewer address", self.url)
        form.addRow("Width", self.w)
        form.addRow("Height", self.h)
        lay.addLayout(form)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setAccessibleName("The HTML to paste")
        self.text.setFixedHeight(96)
        lay.addWidget(self.text)
        for e, nm in ((self.url, "Viewer address"), (self.w, "Width"), (self.h, "Height")):
            e.setAccessibleName(nm)
            e.textChanged.connect(self.update_snippet)
        row = QHBoxLayout()
        row.addStretch(1)
        close = button("Close", "outline")
        close.clicked.connect(self.reject)
        copy = button("Copy", "")
        copy.clicked.connect(self.copy)
        row.addWidget(close)
        row.addWidget(copy)
        lay.addLayout(row)
        self.update_snippet()

    def update_snippet(self):
        from helio.application.export_web import embed_snippet
        self.text.setPlainText(embed_snippet(self.url.text().strip(), self.w.text().strip(),
                                             self.h.text().strip(), self.win.project.name
                                             if self.win.project.images else ""))

    def copy(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.text.toPlainText())
        self.win.flash("Embed snippet copied.")


class FacsimileDialog(QDialog):
    """Physical reproduction: 3D printing, elevated printing, CNC, plates
    and seals (helio.application.export_3d.facsimile_package)."""

    def __init__(self, win):
        from PySide6.QtWidgets import QCheckBox, QComboBox, QSpinBox
        from helio.application.export_3d import (FACSIMILE_FORMATS, FACSIMILE_RES,
                                                  FACSIMILE_TYPES)
        from helio.domain.view_controls import Control
        from helio_qt.controls import ParamRow
        super().__init__(win)
        self.win, self.project = win, win.project
        self.result = None
        self.setWindowTitle("Facsimile")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(8)
        lay.addWidget(label("Facsimile", "title"))
        lay.addWidget(label("A relief for 3D printing, elevated printing or CNC.", "secondary", wrap=True))
        self.otype = QComboBox()
        for k, t in FACSIMILE_TYPES:
            self.otype.addItem(t, k)
        self.otype.setAccessibleName("Object type")
        lay.addWidget(label("Object type", ""))
        lay.addWidget(self.otype)
        self.invert = QCheckBox("Invert the relief (embossed ↔ debossed)")
        lay.addWidget(self.invert)

        def row(key, text, lo, hi, default, step, tip="", unit=""):
            r = ParamRow(Control(key, text, lo, hi, default, step, tip, (), unit))
            lay.addWidget(r)
            return r
        self.mag = row("mag", "Magnification", 0.1, 10, 1.0, 0.1, "× actual size", "×")
        self.zmul = row("zmul", "Height exaggeration", 0.1, 20, 1.0, 0.1,
                        "× the relief, after the bowl is flattened", "×")
        self.warp = row("warp", "Flatten the bowl", 0, 100, 80, 1,
                        "Removes the integrator's broad bowl or saddle before exaggerating, so "
                        "only the object's relief is amplified. 80 % recommended.", "%")
        self.taper = row("taper", "Edge taper", 0, 20, 4, 0.5,
                         "A cosine ramp to zero over this share of each border, for a flat print bed.", "%")
        lay.addWidget(label("Mesh resolution", ""))
        rr = QHBoxLayout()
        self.res = QComboBox()
        for k, _t, text in FACSIMILE_RES:
            self.res.addItem(text, k)
        self.res.addItem("Custom grid", "custom")
        self.res.setCurrentIndex(1)
        self.res.setAccessibleName("Mesh resolution")
        rr.addWidget(self.res, 1)
        self.custom = QSpinBox()
        self.custom.setRange(64, 20000)
        self.custom.setValue(768)
        self.custom.setSuffix(" px")
        self.custom.setAccessibleName("Custom grid, longest edge")
        rr.addWidget(self.custom)
        lay.addLayout(rr)
        self.fmt = QComboBox()
        for k, t in FACSIMILE_FORMATS:
            self.fmt.addItem(t, k)
        self.fmt.setCurrentIndex(1)
        self.fmt.setAccessibleName("Format")
        lay.addWidget(label("Format", ""))
        lay.addWidget(self.fmt)
        self.scale_note = label("", "caption", wrap=True)
        lay.addWidget(self.scale_note)
        self.grid_note = label("", "caption", wrap=True)
        lay.addWidget(self.grid_note)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        btns = QHBoxLayout()
        close = button("Close", "quiet")
        close.clicked.connect(self.reject)
        btns.addWidget(close)
        btns.addStretch(1)
        self.btn_preview = button("Quick Look draft", "outline",
                                  "A fast 256 px mesh with these settings, opened in Quick Look.")
        self.btn_preview.clicked.connect(self.preview)
        self.btn_export = button("Export…", "primary")
        self.btn_export.clicked.connect(self.export)
        btns.addWidget(self.btn_preview)
        btns.addWidget(self.btn_export)
        lay.addLayout(btns)
        for w in (self.res, self.fmt):
            w.currentIndexChanged.connect(self.update_notes)
        self.custom.valueChanged.connect(self.update_notes)
        self.update_notes()

    def options(self) -> dict:
        return dict(otype=self.otype.currentData(), invert=self.invert.isChecked(),
                    mag=self.mag.value(), zmul=self.zmul.value(), warp=self.warp.value() / 100.0,
                    taper=self.taper.value() / 100.0, res=self.res.currentData(),
                    custom=self.custom.value(), fmt=self.fmt.currentData())

    def update_notes(self, *_):
        from helio.application.export_3d import ASSUMED_MM_PER_PX, facsimile_grid
        p, D = self.project, self.win.render_data
        self.custom.setVisible(self.res.currentData() == "custom")
        mm = p.scale_mm_per_px
        z = p.z_mm_per_unit
        self.scale_note.setText(
            (f"Scale {mm:.4f} mm/px." if mm else
             f"No scale set: {ASSUMED_MM_PER_PX} mm/px assumed. Set it for a true-size copy.")
            + (f" Height scale {z * 1000:.1f} µm per pixel of height: heights to scale." if z else
               " No height scale: heights in relative units."))
        if D is None:
            return
        H, W = D.depth_vis_p.shape
        step, rH, rW = facsimile_grid((H, W), self.res.currentData(), self.custom.value())
        px = mm or ASSUMED_MM_PER_PX
        self.grid_note.setText(f"Grid {rW} × {rH}: {rW * rH:,} vertices, {(rW - 1) * (rH - 1) * 2:,} faces; "
                               f"{W * px * self.mag.value():.0f} × {H * px * self.mag.value():.0f} mm.")

    def preview(self):
        import subprocess
        import sys
        from helio.application.export_3d import facsimile_preview
        p, D = self.project, self.win.render_data
        if D is None:
            return
        from helio.application.heights import height_px
        src = height_px(D) if p.z_mm_per_unit else D.depth_vis_p      # true heights to be to scale
        path = facsimile_preview(src, self.options(), mm_per_px=p.scale_mm_per_px,
                                 z_mm_per_unit=p.z_mm_per_unit)
        self.preview_path = path
        if sys.platform == "darwin" and not os.environ.get("HELIO_NO_QUICKLOOK"):
            subprocess.Popen(["qlmanage", "-p", path], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        self.status.setText("Draft opened in Quick Look." if sys.platform == "darwin"
                            else f"Draft written: {path}")

    def export(self, folder: str | None = None):
        from helio.application.export_3d import facsimile_package
        from helio_qt.publish import file_stem
        p, D = self.project, self.win.render_data
        if D is None:
            return
        if not folder:
            folder = QFileDialog.getExistingDirectory(
                self, "Folder for the facsimile",
                str(self.win.prefs.value("export_folder", "")) or self.win.default_export_dir())
        if not folder:
            return
        name = (dict(p.capture_meta).get("title") or p.name or "helio").strip()
        o, s = self.options(), p.settings
        from helio.application.heights import height_px
        depth = height_px(D) if p.z_mm_per_unit else D.depth_vis_p    # true heights to be to scale
        albedo = D.albedo_rgb_full
        self.btn_export.setEnabled(False)
        self.status.setText("Building the facsimile…")

        def work():
            return facsimile_package(depth, folder, file_stem(name), o, mm_per_px=p.scale_mm_per_px,
                                     z_mm_per_unit=p.z_mm_per_unit, albedo=albedo, obj_id=name,
                                     depth_engine=s.depth_engine,
                                     normal_engine=s.resolved_normal_engine())

        def done(res, err):
            self.btn_export.setEnabled(True)
            if err:
                self.status.setText(err.split(": ", 1)[-1])
                return
            self.result = res
            sx, sy, sz = res["size_mm"]
            self.status.setText(f"Saved {', '.join(os.path.basename(f) for f in res['files'])}: "
                                f"{sx:.1f} × {sy:.1f} × {sz:.2f} mm"
                                + (", mirrored as a tool." if res["mirrored"] else "."))
            self.win.flash(f"Facsimile saved: {os.path.basename(res['files'][0])}.")
        self.win.run_background(work, done)


class SketchfabDialog(QDialog):
    """Upload the textured model to the person's Sketchfab account
    (helio.application.export_web.sketchfab_upload)."""

    def __init__(self, win):
        from helio.application.export_web import SKETCHFAB_SLOTS
        super().__init__(win)
        self.win, self.project = win, win.project
        self.opener = None                     # tests replace the network
        self.url = ""
        self.setWindowTitle("Upload to Sketchfab")
        self.setMinimumWidth(500)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(8)
        lay.addWidget(label("Upload to Sketchfab", "title"))
        lay.addWidget(label("The textured model goes to sketchfab.com, under your account.",
                            "secondary", wrap=True))
        form = QFormLayout()
        self.token = QLineEdit(getattr(win, "sketchfab_token", ""))
        self.token.setEchoMode(QLineEdit.EchoMode.Password)
        self.token.setPlaceholderText("Kept for this session only")
        self.token.setAccessibleName("Sketchfab API token")
        self.name = QLineEdit((dict(self.project.capture_meta).get("title") or self.project.name
                               or "Helio Export").strip())
        self.name.setAccessibleName("Model name")
        form.addRow("API token", self.token)
        form.addRow("Model name", self.name)
        lay.addLayout(form)
        sec = Section("Material slots on Sketchfab", "sketchfab_slots", win.prefs, open_=False)
        sec.add(label("\n".join(f"{s}: {f}" + (f"  ({n})" if n else "") for s, f, n in SKETCHFAB_SLOTS)
                      + "\n\nIn Sketchfab's 3D Settings › Materials, assign each file to its slot.",
                      "caption", wrap=True))
        lay.addWidget(sec)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        close = button("Close", "quiet")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        row.addStretch(1)
        self.btn_up = button("Upload", "primary")
        self.btn_up.clicked.connect(self.upload)
        self.btn_open = button("Open on Sketchfab", "outline")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self.open_page)
        row.addWidget(self.btn_up)
        row.addWidget(self.btn_open)
        lay.addLayout(row)

    def upload(self):
        import shutil
        import tempfile
        from helio.application.export_web import sketchfab_upload
        from helio_qt import publish
        token = self.token.text().strip()
        if not token:
            self.status.setText("Enter your Sketchfab API token (sketchfab.com › Settings › Password & API).")
            return
        self.win.sketchfab_token = token
        src = self.win.publish_source()
        name, opener = self.name.text().strip(), self.opener
        self.btn_up.setEnabled(False)
        self.status.setText("Building the model, then uploading…")

        def work():
            tmp = tempfile.mkdtemp(prefix="helio_sketchfab_")
            try:
                files = publish.MESH_EXPORTS["glb_mesh"](src, tmp)
                glb = next(f for f in files if f.lower().endswith(".glb"))
                return sketchfab_upload(glb, token, name, opener=opener)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

        def done(url, err):
            self.btn_up.setEnabled(True)
            if err:
                self.status.setText("Upload failed: " + err.split(": ", 1)[-1])
                return
            self.url = url
            self.btn_open.setEnabled(True)
            self.status.setText(f"Uploaded: {url}\nSketchfab processes the model for a minute or two.")
        self.win.run_background(work, done)

    def open_page(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        if self.url:
            QDesktopServices.openUrl(QUrl(self.url))
