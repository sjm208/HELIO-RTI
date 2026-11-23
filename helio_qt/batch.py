"""File › Batch Process Folders (the classic Overnight Batch).

Choose a parent folder whose subfolders are objects, how each is to be
calibrated and what to export; the run solves and exports each in turn in
the background, with a log.  helio.application.batch holds the scan, the
calibration chain and the loop; the exports are Publish's own.
"""
from __future__ import annotations

import dataclasses
import os

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QListWidget,
                               QListWidgetItem, QProgressBar, QVBoxLayout, QWidget)

from helio_qt.controls import Section, VScroll
from helio_qt.widgets import button, label


class _Sig(QObject):
    progress = Signal(int, int, str)


def source_for(result, images, name, *, shared=None, mm_per_px=None, estimated=False):
    """A Publish source for a solve made outside the project."""
    from helio.application import exports as X
    from helio.application.pipeline import store
    from helio.data_manager import DataManager
    from helio.domain.registry import default_engine_settings
    from helio_qt import publish
    D = DataManager()
    store(result, D)
    D.plate_paths = list(images)
    es = default_engine_settings()
    D.engine_settings = es
    return publish.Source(result=result, lights=result.light_matrix,
                          light_source="estimated" if estimated else "measured",
                          images=list(images), name=name, mm_per_px=float(mm_per_px or 0.0),
                          shared=dict(shared or X.shared_defaults()), data=D, view="Standard PBR",
                          view_settings=dict(es.get("Standard PBR", {})), engine_settings=es,
                          tone=result.settings.photo_tone)


def export_for(key: str, src, out_dir: str, name: str) -> list[str]:
    """One batch export: a Publish format, or the two batch-only sets."""
    from helio.application import batch as B
    from helio_qt import publish
    pk = B.batch_keys().get(key, key)
    if pk == "hillshade":
        return B.multi_hillshade(src.result.depth_vis_p, out_dir, publish.file_stem(name))
    if pk == "raking":
        from helio.application.render import composite, render
        D = src.data
        base = dict(src.engine_settings.get("Standard PBR", {}))

        def draw(lx, ly, lz):
            st = dict(base, lx=lx, ly=ly, lz=lz)
            img = render("Standard PBR", D, st, D.h, D.w, False)
            return composite("Standard PBR", img, D, st, D.h, D.w, False, view_mode="Combined Colour")
        return B.raking_set(draw, out_dir, publish.file_stem(name))
    return publish.export_one(pk, src, out_dir)


class BatchDialog(QDialog):
    def __init__(self, win):
        from helio.application.batch import BATCH_EXPORTS
        super().__init__(win)
        self.win, self.project = win, win.project
        self.parent_dir = ""
        self.shared_lp = None
        self.out_root = None
        self.result = None
        self.stop = False
        self.setWindowTitle("Batch process folders")
        self.setMinimumSize(640, 720)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(8)
        lay.addWidget(label("Batch process folders", "title"))
        lay.addWidget(label("One subfolder per object. Each is calibrated, solved and exported in turn.",
                            "secondary", wrap=True))
        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 6, 0)
        bl.setSpacing(12)

        sec = Section("Folders", "batch_folders", win.prefs)
        row = QHBoxLayout()
        b = button("Choose the parent folder…", "outline")
        b.clicked.connect(self.choose_parent)
        row.addWidget(b)
        row.addStretch(1)
        sec.add_layout(row)
        self.folders = QListWidget()
        self.folders.setAccessibleName("Object folders")
        self.folders.setMinimumHeight(120)
        sec.add(self.folders)
        self.folders_note = label("No folder chosen yet.", "caption", wrap=True)
        sec.add(self.folders_note)
        bl.addWidget(sec)

        sec = Section("Calibration", "batch_cal", win.prefs)
        sec.add(label("A light file in an object's own folder is always used first.", "caption", wrap=True))
        self.chk_auto = QCheckBox("Find the chrome ball in each folder")
        self.chk_auto.setChecked(True)
        sec.add(self.chk_auto)
        cal = getattr(self.project, "calibration", None)
        self.chk_roi = QCheckBox("Look for the ball where it sits in this project")
        self.chk_roi.setEnabled(cal is not None and getattr(cal, "sphere", None) is not None)
        self.chk_roi.setToolTip("For captures made on the same stand: the ball is read in the same "
                                "place in every folder (RTIBuilder's way).")
        sec.add(self.chk_roi)
        self.chk_proj = QCheckBox("Otherwise use this project's light positions")
        self.chk_proj.setEnabled(self.project.lights is not None)
        sec.add(self.chk_proj)
        row = QHBoxLayout()
        b = button("Shared light file…", "quiet")
        b.clicked.connect(self.choose_lp)
        row.addWidget(b)
        self.lp_note = label("None: uncalibrated as a last resort.", "caption")
        row.addWidget(self.lp_note, 1)
        sec.add_layout(row)
        self.chk_scale = QCheckBox("Apply this project's scale (same camera set-up)")
        self.chk_scale.setEnabled(bool(self.project.scale_mm_per_px))
        sec.add(self.chk_scale)
        bl.addWidget(sec)

        sec = Section("Exports", "batch_exports", win.prefs)
        self.export_chk: dict[str, QCheckBox] = {}
        for group, items in BATCH_EXPORTS:
            sec.add(label(group, "section"))
            for key, text, default, what, _pk in items:
                c = QCheckBox(text)
                c.setChecked(default)
                c.setToolTip(what)
                self.export_chk[key] = c
                sec.add(c)
        bl.addWidget(sec)

        sec = Section("Output", "batch_out", win.prefs)
        row = QHBoxLayout()
        b = button("Choose an output folder…", "quiet")
        b.clicked.connect(self.choose_out)
        row.addWidget(b)
        self.out_note = label("Each object's own folder (name_HELIO_exports).", "caption")
        row.addWidget(self.out_note, 1)
        sec.add_layout(row)
        bl.addWidget(sec)
        bl.addStretch(1)
        lay.addWidget(VScroll(body), 1)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1)
        self.bar.setValue(0)
        self.bar.setVisible(False)
        lay.addWidget(self.bar)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        close = button("Close", "quiet")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        row.addStretch(1)
        self.btn_stop = button("Stop after this object", "outline")
        self.btn_stop.setVisible(False)
        self.btn_stop.clicked.connect(self.request_stop)
        self.btn_run = button("Start", "primary")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.start)
        self.btn_log = button("Show the log", "outline")
        self.btn_log.setVisible(False)
        self.btn_log.clicked.connect(self.show_log)
        for w in (self.btn_stop, self.btn_run, self.btn_log):
            row.addWidget(w)
        lay.addLayout(row)
        self.sig = _Sig()
        self.sig.progress.connect(self._progress)

    # ── choosing ──────────────────────────────────────────────────────
    def choose_parent(self, path: str | None = None):
        from helio.application.batch import scan
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Parent folder: one subfolder per object",
                                                    str(self.win.prefs.value("last_folder", "")))
        if not path:
            return
        self.parent_dir = path
        found, lp = scan(path)
        self.folders.clear()
        for folder, n in found:
            it = QListWidgetItem(f"{os.path.basename(folder)}   ·   {n} images")
            it.setData(Qt.ItemDataRole.UserRole, folder)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            self.folders.addItem(it)
        if lp and not self.shared_lp:
            self.set_lp(lp)
        total = sum(n for _f, n in found)
        self.folders_note.setText(f"{len(found)} folders, {total} images." if found else
                                  "No subfolders with 3 or more images here. Each object's "
                                  "images go in their own subfolder.")
        self.btn_run.setEnabled(bool(found))

    def set_lp(self, path):
        self.shared_lp = path
        self.lp_note.setText(os.path.basename(path) if path else "None: uncalibrated as a last resort.")

    def choose_lp(self):
        p, _ = QFileDialog.getOpenFileName(self, "Shared light-position file", self.parent_dir,
                                           "Light files (*.lp *.dome *.txt);;All files (*)")
        if p:
            self.set_lp(p)

    def choose_out(self, path: str | None = None):
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Output folder (one subfolder per object)")
        if path:
            self.out_root = path
            self.out_note.setText(path)

    def chosen_folders(self) -> list[str]:
        return [self.folders.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.folders.count())
                if self.folders.item(i).checkState() == Qt.CheckState.Checked]

    def chosen_exports(self) -> list[str]:
        return [k for k, c in self.export_chk.items() if c.isChecked()]

    # ── running ───────────────────────────────────────────────────────
    def start(self):
        from helio.application import batch as B
        from helio.application import pipeline
        from helio.application.jobs import Cancelled
        folders, keys = self.chosen_folders(), self.chosen_exports()
        if not folders:
            self.status.setText("Tick at least one folder.")
            return
        if not keys:
            self.status.setText("Tick at least one export.")
            return
        p = self.project
        cal = getattr(p, "calibration", None)
        roi = None
        if self.chk_roi.isChecked() and cal is not None and getattr(cal, "sphere", None) is not None:
            s = cal.sphere
            roi = (int(round(s.cx)), int(round(s.cy)), int(round(s.r)))
        proj_L = p.solve_lights() if (self.chk_proj.isChecked() and p.lights is not None) else None
        auto, shared_lp = self.chk_auto.isChecked(), self.shared_lp
        settings = p.settings
        shared = dict(self.win.shared)
        mm = p.photo_scale() if self.chk_scale.isChecked() else None
        out_root = self.out_root
        self.stop = False

        def calibrate(folder, imgs, log):
            return B.calibration_for(folder, imgs, roi=roi, auto_sphere=auto, sphere_L=proj_L,
                                     shared_lp=shared_lp, log=log)

        def solve(folder, imgs, L, kind):
            eng = settings.resolved_normal_engine()
            if kind == "pca":
                eng = "Uncalibrated PCA"
            elif kind in ("sphere", "vdc") or eng == "Uncalibrated PCA":
                eng = "Linear Least Squares"
            st = dataclasses.replace(settings, normal_engine=eng)
            req = pipeline.SolveRequest(paths=imgs, lights=L, settings=st, mm_per_px=mm,
                                        depth_clip_pct=float(shared.get("depth_clip_pct", 0.5)))
            return (pipeline.solve(req, check=check), imgs, L is None)

        def export(key, solved, out_dir, name):
            r, imgs, est = solved
            src = source_for(r, imgs, name, shared=shared, mm_per_px=mm, estimated=est)
            return export_for(key, src, out_dir, name)

        def check():
            if self.stop:
                raise Cancelled()

        def work():
            return B.run(folders, keys, out_root, solve=solve, export=export, calibrate=calibrate,
                         progress=lambda i, n, name: self.sig.progress.emit(i, n, name), check=check)

        self.bar.setRange(0, len(folders))
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.btn_run.setEnabled(False)
        self.btn_stop.setVisible(True)
        self.btn_log.setVisible(False)
        self.status.setText(f"Starting: {len(folders)} objects, {len(keys)} exports each.")
        self.win.run_background(work, self._done)

    def request_stop(self):
        self.stop = True
        self.status.setText("Stopping after the current step…")

    def _progress(self, i, n, name):
        self.bar.setValue(i)
        if name:
            self.status.setText(f"Object {i + 1} of {n}: {name}")

    def _done(self, res, err):
        self.btn_run.setEnabled(True)
        self.btn_stop.setVisible(False)
        if err:
            self.status.setText(err.split(": ", 1)[-1])
            return
        self.result = res
        self.bar.setValue(res["total"])
        self.btn_log.setVisible(bool(res["log"]))
        bad = len(res["errors"]) + sum(1 for o in res["objects"].values() if o["failed"])
        self.status.setText(f"Done: {res['ok']} of {res['total']} objects processed"
                            + (f", {bad} with problems (see the log)." if bad else "."))
        self.win.flash(f"Batch finished: {res['ok']}/{res['total']}.")

    def show_log(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        if self.result and self.result["log"]:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.result["log"]))
