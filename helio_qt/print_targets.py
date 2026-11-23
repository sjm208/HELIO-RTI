"""Calibration targets to print: exact vector PDFs at any common paper size.

* Chessboard: the lens calibration's board (Process › Precision measurement
  › Calibrate from a chessboard), 9 x 6 inner corners, the squares as large as
  the paper allows.  The size of the squares does not matter to the lens
  calibration; only that they are equal and the board flat.
* Scale bars: cut-out scales for the shot, from a fine 20 mm scale in half
  millimetres (coins, seals) to the longest the paper takes, an L-shaped
  corner scale and an inch rule.  Measure's "Scale from a ruler" reads their
  ticks.
* Square-to-camera grid: crosses every 10 mm.  Photographed on the copy
  stand, its rows run parallel to the frame and its spacing is the same in
  every corner when the camera is square to the baseboard.

Every sheet carries a 100 mm check bar: a printer set to "fit to page" shrinks
the page by a few percent, and a ruler on the bar shows it at once.  A laser
printer is usually true to within half a percent; for publication, check a
printed scale against a steel rule (or set the scale from the measured
length).

Written with Qt's own PDF writer, in millimetres, at 1200 dpi.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QMarginsF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPageLayout, QPageSize, QPainter, QPdfWriter, QPen

DPI = 1200
PAPERS = {"A4": QPageSize.PageSizeId.A4, "A3": QPageSize.PageSizeId.A3,
          "A2": QPageSize.PageSizeId.A2, "A1": QPageSize.PageSizeId.A1,
          "Letter": QPageSize.PageSizeId.Letter, "Tabloid (11 × 17 in)": QPageSize.PageSizeId.Tabloid}
KINDS = {"chessboard": "Chessboard (lens calibration)",
         "scales": "Scale bars (detailed, to cut out)",
         "grid": "Square-to-camera grid (10 mm)"}
PATTERN = (9, 6)                   # inner corners: 10 x 7 squares, never the same turned 180°
PRINT_NOTE = "Print at 100% / Actual size, not “Fit to page”. Matte paper; mount flat."


@dataclass
class _Page:
    painter: QPainter
    w_mm: float
    h_mm: float

    def px(self, mm: float) -> float:
        return mm * DPI / 25.4

    def rect(self, x, y, w, h, fill=True):
        r = QRectF(self.px(x), self.px(y), self.px(w), self.px(h))
        if fill:
            self.painter.fillRect(r, QColor("black"))
        else:
            self.painter.drawRect(r)

    def line(self, x1, y1, x2, y2, width_mm=0.15, dash=False):
        pen = QPen(QColor("black"))
        pen.setWidthF(self.px(width_mm))
        if dash:
            pen.setStyle(Qt.PenStyle.DashLine)
        pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.painter.setPen(pen)
        self.painter.drawLine(QPointF(self.px(x1), self.px(y1)), QPointF(self.px(x2), self.px(y2)))

    def text(self, x, y, s, pt=9, bold=False, align="left"):
        f = QFont("Helvetica")
        f.setPointSizeF(pt)
        f.setBold(bold)
        self.painter.setFont(f)
        self.painter.setPen(QColor("black"))
        if align == "left":
            self.painter.drawText(QPointF(self.px(x), self.px(y)), s)
        else:
            fm = self.painter.fontMetrics()
            w = fm.horizontalAdvance(s)
            dx = w / 2 if align == "centre" else w
            self.painter.drawText(QPointF(self.px(x) - dx, self.px(y)), s)


def paper_mm(paper: str, landscape: bool) -> tuple[float, float]:
    s = QPageSize(PAPERS[paper]).size(QPageSize.Unit.Millimeter)
    w, h = s.width(), s.height()
    return (max(w, h), min(w, h)) if landscape else (min(w, h), max(w, h))


def _open(path: str, paper: str, landscape: bool, title: str):
    wr = QPdfWriter(path)
    wr.setResolution(DPI)
    wr.setTitle(title)
    wr.setCreator("Helio")
    wr.setPageLayout(QPageLayout(QPageSize(PAPERS[paper]),
                                 QPageLayout.Orientation.Landscape if landscape
                                 else QPageLayout.Orientation.Portrait, QMarginsF(0, 0, 0, 0)))
    p = QPainter(wr)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    w, h = paper_mm(paper, landscape)
    return wr, _Page(p, w, h)


def _check_bar(pg: _Page, x: float, y: float):
    """A 100 mm bar with centimetre ticks: measure it to see the printer kept the scale."""
    pg.line(x, y, x + 100, y, 0.3)
    for k in range(11):
        t = 3.0 if k in (0, 5, 10) else 1.5
        pg.line(x + 10 * k, y - t, x + 10 * k, y, 0.2)
    pg.text(x + 103, y + 1, "← this bar is 100 mm: measure it to check the print", 7)


def chessboard(path: str, paper: str = "A4", pattern=PATTERN) -> dict:
    """The lens-calibration chessboard: squares in whole millimetres, as large as fit."""
    cols, rows = pattern[0] + 1, pattern[1] + 1
    wr, pg = _open(path, paper, True, "Helio lens calibration chessboard")
    margin, band = 15.0, 22.0
    sq = float(int(min((pg.w_mm - 2 * margin) / cols, (pg.h_mm - 2 * margin - band) / rows)))
    bw, bh = cols * sq, rows * sq
    x0, y0 = (pg.w_mm - bw) / 2, margin
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                pg.rect(x0 + c * sq, y0 + r * sq, sq, sq)
    ty = y0 + bh + 7
    pg.text(x0, ty, f"Helio lens calibration · {pattern[0]} × {pattern[1]} inner corners "
                    f"({cols} × {rows} squares) · squares {sq:.0f} mm", 9, True)
    pg.text(x0, ty + 4.5, PRINT_NOTE + " Photograph it 10-20 times across the whole frame, "
                                       "some tilted up to 30°.", 7)
    _check_bar(pg, x0, ty + 11)
    pg.painter.end()
    return dict(path=path, paper=paper, pattern=tuple(pattern), square_mm=sq,
                board_mm=(bw, bh), origin_mm=(x0, y0), page_mm=(pg.w_mm, pg.h_mm))


def _ruler(pg: _Page, x: float, y: float, length: float, *, step: float = 1.0, label_every: float = 10.0,
           blocks: bool = True, unit: str = "mm", unit_mm: float = 1.0):
    """A ruler starting at (x, y): ticks downward from the top edge, numbers, and (blocks)
    alternating black and white bands a label step long, readable in a small photograph.
    *step* and *label_every* in *unit*s; *unit_mm* the unit's length in mm."""
    L = length * unit_mm
    pg.line(x, y, x + L, y, 0.2)
    n = int(round(length / step))
    for k in range(n + 1):
        v = k * step
        xx = x + v * unit_mm
        major = abs(v / label_every - round(v / label_every)) < 1e-9
        half = abs((2 * v / label_every) - round(2 * v / label_every)) < 1e-9
        t = 3.2 if major else (2.2 if half else 1.3)
        pg.line(xx, y, xx, y + t, 0.12 if not major else 0.18)
        if major:
            lab = f"{v:g}"
            pg.text(xx, y + 6.3, lab, 6, align="centre")
    if blocks:
        bh = 2.5
        k = 0
        while k * label_every < length - 1e-9:
            if k % 2 == 0:
                w = min(label_every, length - k * label_every) * unit_mm
                pg.rect(x + k * label_every * unit_mm, y + 7.5, w, bh)
            k += 1
        pg.line(x, y + 7.5, x + L, y + 7.5, 0.12)
        pg.line(x, y + 10.0, x + L, y + 10.0, 0.12)
        pg.line(x, y + 7.5, x, y + 10.0, 0.12)
        pg.line(x + L, y + 7.5, x + L, y + 10.0, 0.12)
    pg.text(x + L + 2, y + 3.5, unit, 7)
    return 11.0 if blocks else 8.0                     # height used, mm


def scales(path: str, paper: str = "A4") -> dict:
    """Detailed scale bars to cut out: a fine 20 mm, then 50 mm, 100 mm and longer as the
    paper allows, an L-shaped corner scale and an inch rule."""
    wr, pg = _open(path, paper, False, "Helio scale bars")
    m = 14.0
    usable = pg.w_mm - 2 * m - 12
    pg.text(m, m, "Helio scale bars · " + paper, 11, True)
    pg.text(m, m + 5, PRINT_NOTE + " Cut along the dashed lines; lay a scale beside the object, "
                                   "in the same plane.", 7)
    y = m + 14
    strips = []

    def strip(height, draw, label):
        nonlocal y
        top = y
        pg.text(m + 2, top + 4, label, 7, True)
        h = draw(m + 4, top + 7)
        box = (m, top, pg.w_mm - m, top + 7 + h + 2)
        pg.line(box[0], box[1], box[2], box[1], 0.1, dash=True)
        pg.line(box[0], box[3], box[2], box[3], 0.1, dash=True)
        strips.append(dict(label=label, box_mm=box, ruler_x=m + 4, ruler_y=top + 7))
        y = box[3] + 3

    strip(10, lambda x, yy: _ruler(pg, x, yy, 20, step=0.5, label_every=5, blocks=True),
          "Fine scale · 20 mm in 0.5 mm (coins, seals, small detail)")
    for L in (50, 100, 150, 200, 250, 300, 400, 500):
        if L <= usable and (L <= 100 or L >= usable * 0.55):
            strip(10, lambda x, yy, L=L: _ruler(pg, x, yy, L, step=1, label_every=10),
                  f"{L} mm scale · 1 mm ticks, 10 mm blocks")
    inches = int(min(12, usable // 25.4))
    strip(10, lambda x, yy: _ruler(pg, x, yy, inches, step=1 / 16, label_every=1, unit="in", unit_mm=25.4),
          f"{inches} inch rule · 1/16 in ticks")
    # an L-shaped corner scale: 50 mm each way
    top = y
    pg.text(m + 2, top + 4, "L scale · 50 mm each way (lay it at a corner of the object)", 7, True)
    cx, cy = m + 8, top + 62
    pg.line(cx, cy, cx + 50, cy, 0.25)
    pg.line(cx, cy, cx, cy - 50, 0.25)
    for k in range(51):
        t = 3.2 if k % 10 == 0 else (2.2 if k % 5 == 0 else 1.3)
        pg.line(cx + k, cy, cx + k, cy - t, 0.12)
        pg.line(cx, cy - k, cx + t, cy - k, 0.12)
        if k % 10 == 0 and k:
            pg.text(cx + k, cy + 4, str(k), 6, align="centre")
            pg.text(cx - 2, cy - k + 1, str(k), 6, align="right")
    lbox = (m, top, pg.w_mm - m, cy + 8)
    pg.line(lbox[0], lbox[3], lbox[2], lbox[3], 0.1, dash=True)
    strips.append(dict(label="L scale", box_mm=lbox, ruler_x=cx, ruler_y=cy))
    y = lbox[3] + 6
    _check_bar(pg, m, y + 4)
    pg.text(m, y + 12, "A printed scale is as true as the printer (a laser printer is usually within "
                       "0.5%). For publication, check it against a steel rule; if it differs, set the "
                       "scale in Helio from the measured length.", 6.5)
    pg.painter.end()
    return dict(path=path, paper=paper, strips=strips, page_mm=(pg.w_mm, pg.h_mm))


def grid(path: str, paper: str = "A4", spacing: float = 10.0) -> dict:
    """Crosses every *spacing* mm, a heavier one every 50 mm and at the centre, with a frame."""
    wr, pg = _open(path, paper, True, "Helio square-to-camera grid")
    m, band = 15.0, 18.0
    nx = int((pg.w_mm - 2 * m) // spacing)
    ny = int((pg.h_mm - 2 * m - band) // spacing)
    gw, gh = nx * spacing, ny * spacing
    x0, y0 = (pg.w_mm - gw) / 2, m
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = x0 + i * spacing, y0 + j * spacing
            heavy = (i * spacing) % 50 == 0 and (j * spacing) % 50 == 0
            a = 2.5 if heavy else 1.5
            wdt = 0.25 if heavy else 0.15
            pg.line(x - a, y, x + a, y, wdt)
            pg.line(x, y - a, x, y + a, wdt)
    pg.line(x0, y0, x0 + gw, y0, 0.2)
    pg.line(x0, y0 + gh, x0 + gw, y0 + gh, 0.2)
    pg.line(x0, y0, x0, y0 + gh, 0.2)
    pg.line(x0 + gw, y0, x0 + gw, y0 + gh, 0.2)
    cx, cy = x0 + (nx // 2) * spacing, y0 + (ny // 2) * spacing
    pg.painter.setBrush(QColor("black"))
    pg.painter.setPen(Qt.PenStyle.NoPen)
    pg.painter.drawEllipse(QPointF(pg.px(cx), pg.px(cy)), pg.px(1.2), pg.px(1.2))
    ty = y0 + gh + 7
    pg.text(x0, ty, f"Helio square-to-camera grid · crosses every {spacing:g} mm · "
                    f"{gw:g} × {gh:g} mm", 9, True)
    pg.text(x0, ty + 4.5, PRINT_NOTE + " On the copy stand: rows parallel to the frame and the "
                                       "same spacing in every corner mean the camera is square.", 7)
    _check_bar(pg, x0, ty + 10)
    pg.painter.end()
    return dict(path=path, paper=paper, spacing_mm=spacing, grid_mm=(gw, gh), origin_mm=(x0, y0),
                n=(nx + 1, ny + 1), page_mm=(pg.w_mm, pg.h_mm))


MAKERS = {"chessboard": chessboard, "scales": scales, "grid": grid}


def make(kind: str, path: str, paper: str) -> dict:
    if kind not in MAKERS:
        raise ValueError(f"unknown target {kind!r}")
    if paper not in PAPERS:
        raise ValueError(f"unknown paper {paper!r}")
    return MAKERS[kind](path, paper)


# ── the dialog ────────────────────────────────────────────────────────

from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QVBoxLayout  # noqa: E402

ABOUT = {
    "chessboard": "For Process › Precision measurement › Calibrate from a chessboard. Any laser print "
                  "will do: what matters is that it is flat (mount it on foam board or glass) and "
                  "matte. Choose a paper so the board fills a third to half of the frame; then "
                  "photograph it 10-20 times across the whole frame, some tilted up to 30°. "
                  "The same board, flat on the baseboard, checks the camera is square "
                  "(Capture › Check Camera Alignment).",
    "scales": "Scale bars to cut out and lay beside the object, in the same plane: a fine 20 mm "
              "scale in half millimetres for coins and seals, longer ones as the paper allows, an "
              "L-shaped corner scale and an inch rule. Measure › Scale from a ruler reads their "
              "ticks. Check a printed scale against a steel rule before publishing measurements.",
    "grid": "Crosses every 10 mm. Lay it on the baseboard and photograph it: the rows should run "
            "parallel to the frame and the spacing be the same in every corner. If not, the "
            "camera is not square to the stand (or the lens distorts: calibrate it).",
}


class PrintTargetsDialog(QDialog):
    """Print a calibration target: which target, which paper, and a PDF to print at 100%."""

    def __init__(self, win, kind: str = "chessboard"):
        from helio_qt.widgets import button, label
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Print a calibration target")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(10)
        lay.addWidget(label("Print a calibration target", "title"))
        self.kind = QComboBox()
        for k, t in KINDS.items():
            self.kind.addItem(t, k)
        self.kind.setAccessibleName("Target")
        self.kind.setToolTip("Which target to print.")
        self.kind.setCurrentIndex(max(0, self.kind.findData(kind)))
        lay.addWidget(self.kind)
        self.paper = QComboBox()
        self.paper.addItems(list(PAPERS))
        self.paper.setAccessibleName("Paper")
        self.paper.setToolTip("The paper you will print on. Larger paper makes larger squares and "
                              "longer scales: the chessboard should fill a third to half of the frame.")
        lay.addWidget(self.paper)
        self.about = label("", "caption", wrap=True)
        lay.addWidget(self.about)
        self.note = label(PRINT_NOTE + " Each sheet has a 100 mm check bar: measure it after printing.",
                          "secondary", wrap=True)
        lay.addWidget(self.note)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        close = button("Close", "quiet", "Close this window.")
        close.clicked.connect(self.reject)
        row.addWidget(close)
        row.addStretch(1)
        self.btn = button("Save PDF…", "primary", "Write the target as a PDF to print at 100%, and open it.")
        self.btn.clicked.connect(lambda: self.save())
        row.addWidget(self.btn)
        lay.addLayout(row)
        self.kind.currentIndexChanged.connect(self._sync)
        self._sync()

    def _sync(self):
        self.about.setText(ABOUT[self.kind.currentData()])

    def save(self, path: str = "", open_after: bool = True) -> dict | None:
        import os
        from PySide6.QtWidgets import QFileDialog
        kind, paper = self.kind.currentData(), self.paper.currentText()
        if not path:
            start = os.path.join(str(self.win.prefs.value("last_folder", "")) or os.path.expanduser("~"),
                                 f"Helio {kind} {paper.split(' ')[0]}.pdf")
            path, _ = QFileDialog.getSaveFileName(self, "Save the target", start, "PDF (*.pdf)")
            if not path:
                return None
        res = make(kind, path, paper)
        extra = (f" Squares {res['square_mm']:.0f} mm; Helio needs only the corner count, "
                 f"{res['pattern'][0]} x {res['pattern'][1]}." if kind == "chessboard" else "")
        self.status.setText(f"Saved {os.path.basename(path)}.{extra} Print it at 100%.")
        if open_after:
            from PySide6.QtCore import QUrl
            from PySide6.QtGui import QDesktopServices
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        return res
