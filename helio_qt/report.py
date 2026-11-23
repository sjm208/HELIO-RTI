"""A results window for the quality checks: a verdict, the figures, the
pictures, and Save (QC-001…008).

Every check reports the same way, so the numbers are always in the same
place: a one-line verdict with its status colour, a table of figures with
their units, the images side by side with captions, and buttons to save
the images and a text record of the figures.
"""
from __future__ import annotations

import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                               QMessageBox, QVBoxLayout, QWidget)

from helio_qt.widgets import button, label, pixmap_from_array


class ReportDialog(QDialog):
    def __init__(self, parent, title: str, verdict: str = "", status: str = "",
                 figures: list[tuple[str, str]] | None = None,
                 images: list[tuple[object, str]] | None = None, note: str = "",
                 save_name: str = "report", raw: dict | None = None):
        """*figures*: (name, value with unit).  *images*: (uint8 RGB array, caption).
        *raw*: {file name: array} written as 32-bit TIFF alongside the PNGs."""
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(720)
        self.title, self.verdict, self.figures = title, verdict, figures or []
        self.images, self.raw, self.save_name = images or [], raw or {}, save_name
        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        lay.setSpacing(12)
        lay.addWidget(label(title, "title"))
        if verdict:
            v = label(verdict, "section", wrap=True, status=status)
            lay.addWidget(v)
        if self.figures:
            grid = QGridLayout()
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(4)
            for i, (name, value) in enumerate(self.figures):
                grid.addWidget(label(name, "secondary"), i, 0)
                v = label(value, "")
                v.setProperty("role", "mono")
                v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                grid.addWidget(v, i, 1)
            grid.setColumnStretch(2, 1)
            lay.addLayout(grid)
        if self.images:
            row = QHBoxLayout()
            row.setSpacing(12)
            for arr, cap in self.images:
                col = QVBoxLayout()
                pic = QLabel()
                pic.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pic.setPixmap(pixmap_from_array(arr).scaled(
                    260, 220, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
                pic.setAccessibleName(cap)
                col.addWidget(pic)
                col.addWidget(label(cap, "caption", wrap=True))
                w = QWidget()
                w.setLayout(col)
                row.addWidget(w)
            row.addStretch(1)
            lay.addLayout(row)
        if note:
            lay.addWidget(label(note, "caption", wrap=True))
        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_save = button("Save pictures and figures…", "")
        self.btn_save.clicked.connect(self.save)
        close = button("Close", "primary")
        close.clicked.connect(self.accept)
        btns.addWidget(self.btn_save)
        btns.addWidget(close)
        lay.addLayout(btns)

    def text(self) -> str:
        lines = [f"Helio — {self.title}", "=" * 48, time.strftime("%Y-%m-%d %H:%M"), ""]
        if self.verdict:
            lines += [self.verdict, ""]
        lines += [f"{n:<40} {v}" for n, v in self.figures]
        return "\n".join(lines) + "\n"

    def save_to(self, folder: str) -> list[str]:
        import cv2
        import numpy as np
        os.makedirs(folder, exist_ok=True)
        out = []
        for i, (arr, cap) in enumerate(self.images):
            stem = "".join(c if c.isalnum() else "_" for c in cap.lower())[:40].strip("_") or f"image_{i}"
            p = os.path.join(folder, f"{self.save_name}_{stem}.png")
            cv2.imwrite(p, cv2.cvtColor(np.asarray(arr), cv2.COLOR_RGB2BGR))
            out.append(p)
        for name, arr in self.raw.items():
            p = os.path.join(folder, f"{self.save_name}_{name}.tif")
            cv2.imwrite(p, np.asarray(arr, np.float32))
            out.append(p)
        p = os.path.join(folder, f"{self.save_name}.txt")
        with open(p, "w") as f:
            f.write(self.text())
        out.append(p)
        return out

    def save(self):
        folder = QFileDialog.getExistingDirectory(self, "Save the pictures and figures to")
        if not folder:
            return
        files = self.save_to(folder)
        QMessageBox.information(self, "Saved", f"{len(files)} files saved to\n{folder}")
