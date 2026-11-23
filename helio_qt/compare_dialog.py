"""Compare corrections: each light correction on its own, solved small,
and how flat the things marked flat come out (helio.application.quick_look).

One row per correction: its normal map, the lean of the grey patch and of
every flat area, and of the picture as a whole, in degrees.  Flat is 0°.
The flattest row is marked; "Use this" turns that correction on alone.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout

from helio_qt.widgets import button, label, pixmap_from_array

HOW = ("<p>Each row is the solve with one correction on its own, made small so it takes "
       "seconds. The numbers say how far each marked area leans, in degrees: "
       "<b>flat is 0°</b>, and under 1° is good.</p>"
       "<p>An area a correction is measured on comes out flat by design (Flat areas evens "
       "out the very areas you marked). The other areas, and the whole picture, say more: "
       "if they get worse, the correction is hurting.</p>"
       "<p>The whole picture includes the object's real shape, so it is only a guide; but a "
       "correction that tilts it more than 5° beyond no correction is never marked flattest.</p>")


def _deg(v) -> str:
    return "–" if v is None else f"{v:.1f}°"


WHOLE_WORSE_LIMIT = 5.0     # degrees: a correction that tilts the whole picture more is never "flattest"


def best_row(rows) -> dict | None:
    """The row whose marked areas lean least (the worst area counts most),
    among those that do not tilt the whole picture more than
    WHOLE_WORSE_LIMIT beyond no correction: a correction can flatten a
    card beside the ball and tilt everything else (near-field with a
    guessed distance, on a binding with the card and ball at one edge:
    the card 12 to 8 degrees, the whole picture 10 to 29)."""
    none = next((r for r in rows if r["name"] == "None"), None)
    base = none["whole"] if none is not None and none["whole"] is not None else None
    scored = [r for r in rows if r["worst"] is not None and r["applied"]
              and (base is None or r["whole"] is None or r["whole"] <= base + WHOLE_WORSE_LIMIT)]
    return min(scored, key=lambda r: (r["worst"] + r["mean"]) / 2.0) if scored else None


def summary(out) -> str:
    rows = out["rows"]
    b = best_row(rows)
    none = next((r for r in rows if r["name"] == "None"), None)
    if b is None:
        return "Nothing to measure: mark a flat area or the grey patch first."
    if none is not None and b is none:
        return f"Flattest without a correction (worst area {_deg(none['worst'])})."
    base = f", none {_deg(none['worst'])}" if none is not None else ""
    return f"Flattest: {b['name'].lower()} (worst area {_deg(b['worst'])}{base})."


class CompareDialog(QDialog):
    def __init__(self, win, out, use):
        super().__init__(win)
        self.win, self.out, self.use = win, out, use
        self.setWindowTitle("Compare corrections")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(10)
        lay.addWidget(label("Compare corrections", "title"))
        w, h = out["rows"][0]["normals"].shape[1], out["rows"][0]["normals"].shape[0]
        lay.addWidget(label(f"How far each marked area leans. Flat is 0°. "
                            f"Solved small, {w:,} × {h:,} px.", "secondary", wrap=True))
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        heads = ["Correction", ""] + [n for n, _b in out["refs"]] + ["Whole picture", ""]
        for c, t in enumerate(heads):
            grid.addWidget(label(t.upper(), "heading"), 0, c)
        best = best_row(out["rows"])
        none_whole = next((r["whole"] for r in out["rows"] if r["name"] == "None"), None)
        self.use_buttons = {}
        for i, r in enumerate(out["rows"], start=1):
            worse = (r is not best and none_whole is not None and r["whole"] is not None
                     and r["whole"] > none_whole + WHOLE_WORSE_LIMIT)
            name = r["name"] + ("  ✓ flattest" if r is best else
                                "  ✗ tilts the whole picture" if worse else "")
            nm = label(name, "section" if r is best else "secondary")
            if not r["applied"]:
                nm.setText(r["name"] + "  (could not be read)")
            grid.addWidget(nm, i, 0)
            thumb = QLabel()
            n = r["normals"]
            rgb = np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            thumb.setPixmap(pixmap_from_array(rgb).scaled(
                140, 140, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            thumb.setAccessibleName(f"{r['name']}: normal map")
            grid.addWidget(thumb, i, 1)
            for c, v in enumerate(r["tilts"], start=2):
                grid.addWidget(label(_deg(v), "secondary"), i, c)
            grid.addWidget(label(_deg(r["whole"]), "secondary"), i, 2 + len(r["tilts"]))
            b = button("Use this", "quiet", f"Turn on {r['name'].lower()} alone." if
                       r["name"] != "None" else "Turn every light correction off.")
            b.clicked.connect(lambda _=False, nm_=r["name"]: self._use(nm_))
            self.use_buttons[r["name"]] = b
            grid.addWidget(b, i, 3 + len(r["tilts"]))
        lay.addLayout(grid)
        self.btn_how = button("How to read this", "quiet")
        self.btn_how.setCheckable(True)
        lay.addWidget(self.btn_how, 0, Qt.AlignmentFlag.AlignLeft)
        how = label(HOW, "secondary", wrap=True)
        how.setTextFormat(Qt.TextFormat.RichText)
        how.setMaximumWidth(640)
        how.setVisible(False)
        self.btn_how.toggled.connect(how.setVisible)
        lay.addWidget(how)
        row = QHBoxLayout()
        row.addStretch(1)
        close = button("Close", "")
        close.clicked.connect(self.accept)
        row.addWidget(close)
        lay.addLayout(row)

    def _use(self, name: str):
        self.use(name)
        self.win.flash(f"{name}: on alone. Process to solve full size." if name != "None"
                       else "Every light correction off.")
        self.accept()
