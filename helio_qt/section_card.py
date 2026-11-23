"""The section card: a section A–B's profile, its length and roughness, the
grooves and ridges along it, and Save (profile CSV, features CSV, the chart
as SVG and PNG).  Under the 3D relief in Inspect, and in the high-detail 3D
window, each for its own view's line.

Copies: the same section repeated at an even spacing to both sides (a series), drawn on the relief and
as stacked or overlaid profiles; Drawing… writes a scaled section drawing (SVG), a CAD file (DXF) and,
for a series, a spreadsheet (CSV)."""
from __future__ import annotations

import csv
import os

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QCheckBox, QDoubleSpinBox, QFrame, QHBoxLayout, QSpinBox, QVBoxLayout,
                               QWidget)

from helio_qt.controls import ProfileChart, nice_ticks
from helio_qt.widgets import button, label

SERIES_MAX = 25


class SeriesChart(QWidget):
    """A series of sections drawn together: stacked one above another (a ridgeline plot, each at its own
    baseline, offset written at the right) or overlaid on shared axes."""
    LINES = ["#6FB7FF", "#E6A93E", "#46C79A", "#E86B8B", "#B48CF0", "#7FD1E8", "#D8C45A", "#F29D6B"]

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(140)
        self.series, self.stacked = None, True
        self.setAccessibleName("Parallel sections")

    def set_series(self, series, stacked: bool = True):
        self.series, self.stacked = series, bool(stacked)
        self.update()

    def paintEvent(self, _e):
        from helio_qt import theme
        import numpy as np
        t = theme.CURRENT
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        box = QPainterPath()
        box.addRoundedRect(QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0), 6, 6)
        p.fillPath(box, QColor(t.well))
        p.setPen(QPen(QColor(t.line), 1))
        p.drawPath(box)
        profs = (self.series or {}).get("profiles") or []
        if len(profs) < 2:
            p.end()
            return
        from helio.core.sections import stack_gap
        lu, zu = self.series["length_unit"], self.series["unit"].split()[0]
        n = len(profs)
        gap = stack_gap(profs) if self.stacked else 0.0
        zmin = min(float(np.min(q["z"])) for q in profs)
        zmax = max(float(np.max(q["z"])) for q in profs) + gap * (n - 1)
        pad = (zmax - zmin) * 0.08 or 1.0
        lo, hi = zmin - pad, zmax + pad
        xmax = float(profs[0]["dist"][-1]) or 1.0
        L, R, T, B = 44, 52, 12, 28
        r = QRectF(L, T, max(10, self.width() - L - R), max(10, self.height() - T - B))

        def X(v):
            return r.left() + r.width() * v / xmax

        def Y(v, k):
            return r.bottom() - r.height() * ((v - lo) + k * gap) / (hi - lo)
        small = QFont(p.font())
        small.setPointSizeF(max(8.0, small.pointSizeF() - 1.5))
        p.setFont(small)
        grid = QColor(t.line)
        grid.setAlpha(120)
        for v in nice_ticks(0.0, xmax, 6):
            p.setPen(QPen(grid, 1, Qt.PenStyle.DotLine))
            p.drawLine(QPointF(X(v), r.top()), QPointF(X(v), r.bottom()))
            p.setPen(QColor(t.text3))
            p.drawText(QRectF(X(v) - 30, r.bottom() + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, f"{v:g}")
        p.setPen(QColor(t.text3))
        p.drawText(QRectF(r.right() - 150, r.bottom() + 14, 150, 14), Qt.AlignmentFlag.AlignRight, f"distance ({lu})")
        if not self.stacked:
            for v in nice_ticks(lo, hi, 5):
                p.setPen(QPen(grid, 1, Qt.PenStyle.DotLine))
                p.drawLine(QPointF(r.left(), Y(v, 0)), QPointF(r.right(), Y(v, 0)))
                p.setPen(QColor(t.text3))
                p.drawText(QRectF(0, Y(v, 0) - 8, L - 6, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{v:.3g}")
        else:                                   # one height scale bar for all of them
            bar = next((b for b in nice_ticks(0.0, gap * 1.2 or 1.0, 3) if b > 0), gap or 1.0)
            x = r.left() - 8
            p.setPen(QPen(QColor(t.text2), 1.5))
            p.drawLine(QPointF(x, Y(zmin, 0)), QPointF(x, Y(zmin + bar, 0)))
            p.setPen(QColor(t.text3))
            p.drawText(QRectF(0, Y(zmin + bar / 2, 0) - 8, x - 4, 16), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{bar:g} {zu}")
        for i, q in enumerate(profs):
            d, z = np.asarray(q["dist"], float), np.asarray(q["z"], float)
            k = i if self.stacked else 0
            col = QColor(self.LINES[i % len(self.LINES)] if not self.stacked else self.LINES[0])
            path = QPainterPath()
            path.moveTo(X(d[0]), Y(z[0], k))
            for dv, zv in zip(d[1:], z[1:]):
                path.lineTo(X(dv), Y(zv, k))
            if self.stacked:
                base = QColor(t.line)
                base.setAlpha(150)
                p.setPen(QPen(base, 1, Qt.PenStyle.DotLine))
                p.drawLine(QPointF(r.left(), Y(zmin, k)), QPointF(r.right(), Y(zmin, k)))
            p.setPen(QPen(col, 1.6))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)
            p.setPen(col if not self.stacked else QColor(t.text2))
            p.drawText(QRectF(r.right() + 4, Y(float(z[-1]), k) - 8, R - 6, 16),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{q['offset']:+.3g}")
        p.setPen(QColor(t.text3))
        p.drawText(QRectF(r.right() + 4, 0, R, T + 4), Qt.AlignmentFlag.AlignLeft, f"{lu} off")
        p.end()


class SectionCard(QFrame):
    hovered = Signal(float)              # 0..1 along A-B under the pointer, or -1
    clear_requested = Signal()
    save_requested = Signal()
    share_toggled = Signal(bool)
    series_requested = Signal(int, float)    # copies (1 = the section alone), spacing in the section's length unit
    draw_requested = Signal()
    sheet_requested = Signal()           # keep this section (and its copies) for the publication sheet
    lines_requested = Signal()           # count the chain lines along it again (the options or the line changed)
    lines_changed = Signal(list)         # the lines' places 0..1 along A-B, to mark on the views

    def __init__(self, share: str = ""):
        super().__init__()
        self.setObjectName("SectionCard")
        self.prof = None
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        head = QHBoxLayout()
        self.letters = ("A", "B")
        self.title = label("Section A–B", "section")
        head.addWidget(self.title)
        self.summary = label("", "secondary")
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        head.addWidget(self.summary, 1)
        self.btn_big = button("Larger", "quiet", "A taller chart.")
        self.btn_big.setCheckable(True)
        self.btn_big.toggled.connect(lambda on: self.chart.setMinimumHeight(360 if on else 140))
        head.addWidget(self.btn_big)
        self.btn_fold = button("Fold", "quiet", "Fold the card to this one line, so the picture and the 3D "
                                                "relief have the room; Unfold brings the chart back.")
        self.btn_fold.setCheckable(True)
        self.btn_fold.toggled.connect(self.set_folded)
        head.addWidget(self.btn_fold)
        b = button("Save…", "quiet", "The profile as CSV, its grooves and ridges as CSV, and "
                                     "the chart as SVG and PNG.")
        b.clicked.connect(self.save_requested.emit)
        head.addWidget(b)
        self.btn_sheet = button("Add to sheet", "quiet", "Keep this section, and its parallel copies if they are on, "
                                                         "for the publication sheet. Then draw another: the sheet "
                                                         "shows up to six.")
        self.btn_sheet.clicked.connect(self.sheet_requested.emit)
        head.addWidget(self.btn_sheet)
        b = button("Clear", "quiet", "Remove the section line.")
        b.clicked.connect(self.clear_requested.emit)
        head.addWidget(b)
        v.addLayout(head)
        self.body = QWidget()                # all below the head: folded away on request
        bv = QVBoxLayout(self.body)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(6)
        v.addWidget(self.body)
        self.chart = ProfileChart(140)
        self.chart.hovered.connect(self.hovered.emit)
        # the copies' chart sits to the right of the section's own while copies are on, so the card
        # keeps its height; alone, the section has the whole width
        charts = QHBoxLayout()
        charts.setSpacing(8)
        charts.addWidget(self.chart, 3)
        self.series_chart = SeriesChart()
        self.series_chart.setVisible(False)
        charts.addWidget(self.series_chart, 2)
        bv.addLayout(charts)
        self._build_lines(bv)                # chain lines straight under the chart they mark
        self.features = label("", "caption", wrap=True)
        self.features.setTextFormat(Qt.TextFormat.RichText)
        self.features.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bv.addWidget(self.features)
        # copies of the section across the surface, and the drawing
        row = QHBoxLayout()
        row.addWidget(label("Copies", "secondary"))
        self.count = QSpinBox()
        self.count.setRange(1, SERIES_MAX)
        self.count.setValue(1)
        self.count.setToolTip("1 is the section alone. More draws that many parallel sections, evenly spaced to both "
                              "sides of it (odd numbers keep yours in the middle).")
        row.addWidget(self.count)
        row.addWidget(label("every", "secondary"))
        self.spacing = QDoubleSpinBox()
        self.spacing.setDecimals(3)
        self.spacing.setRange(0.001, 100000.0)
        self.spacing.setValue(1.0)
        self.spacing.setToolTip("The distance between neighbouring sections, in the section's own unit (mm once a scale is set).")
        row.addWidget(self.spacing)
        self.chk_stack = QCheckBox("Stack")
        self.chk_stack.setChecked(True)
        self.chk_stack.setToolTip("Each section above the last at its own baseline (a ridgeline plot); off overlays them on one set of axes.")
        row.addWidget(self.chk_stack)
        row.addStretch(1)
        row.addWidget(label("Heights ×", "secondary"))
        self.vexag = QDoubleSpinBox()
        self.vexag.setRange(1.0, 100.0)
        self.vexag.setValue(1.0)
        self.vexag.setDecimals(1)
        self.vexag.setToolTip("For the drawing: 1 draws height and distance to the same scale; raise it to show shallow relief.")
        row.addWidget(self.vexag)
        self.btn_draw = button("Drawing…", "quiet", "A scaled section drawing (SVG), a CAD file (DXF), and for copies a spreadsheet (CSV).")
        self.btn_draw.clicked.connect(self.draw_requested.emit)
        row.addWidget(self.btn_draw)
        bv.addLayout(row)
        self.series_msg = label("", "caption", wrap=True)
        bv.addWidget(self.series_msg)
        self.series = None
        self._spacing_set = False
        self.count.valueChanged.connect(lambda _v: self._ask_series())
        self.spacing.valueChanged.connect(lambda _v: self._spacing_edited())
        self.chk_stack.toggled.connect(lambda _on: self.series_chart.set_series(self.series, self.chk_stack.isChecked()))
        # A card that can send its line elsewhere (the high-detail window's, to
        # Inspect's 2D picture) says so with a tick box.
        self.chk_share = QCheckBox(share) if share else None
        if self.chk_share is not None:
            self.chk_share.toggled.connect(self.share_toggled.emit)
            bv.addWidget(self.chk_share)
        self.setVisible(False)
        self.btn_fold.setChecked(self._stored_fold())

    # ── chain lines along the section (helio.application.chain_lines) ──
    def _build_lines(self, bv):
        """Count chain lines, laid lines or ruling along the section: off until ticked, and only on a card
        whose view offers it (enable_lines)."""
        from PySide6.QtWidgets import QComboBox, QSizePolicy
        from helio.application.chain_lines import LINE_KINDS, SOURCES
        from helio.core.paper_lines import suggest_swath_mm
        self.lines_box = QWidget()
        lv = QVBoxLayout(self.lines_box)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)
        row = QHBoxLayout()
        self.chk_lines = QCheckBox("Count")
        self.chk_lines.setToolTip("Count the paper's chain lines (or laid lines, or a parchment's ruling) that the "
                                  "section crosses, and measure their spacing. Draw the section across the lines.")
        # The style's indicator is wider than the size hint allows for: room for it and the word, never "Coun".
        self.chk_lines.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.chk_lines.setMinimumWidth(self.chk_lines.fontMetrics().horizontalAdvance("Count") + 44)
        row.addWidget(self.chk_lines)
        self.lines_kind = QComboBox()
        for k, text in LINE_KINDS.items():
            self.lines_kind.addItem(text, k)
        self.lines_kind.setToolTip("Chain lines: 12–45 mm apart. Laid lines: 0.6–1.7 mm apart, counted along a "
                                   "chain line. Ruling: a parchment's ruled lines, 3–20 mm apart.")
        row.addWidget(self.lines_kind)
        row.addWidget(label("on", "secondary"))
        self.lines_source = QComboBox()
        for k, text in SOURCES.items():
            self.lines_source.addItem(text, k)
        self.lines_source.setToolTip("The picture the lines are looked for in. A backlit (transmitted-light) frame "
                                     "shows them best; the albedo then masks the writing. This view: the picture "
                                     "Inspect is showing, as shown (a low raking relight, a high-pass view).")
        row.addWidget(self.lines_source)
        row.addWidget(label("strip", "secondary"))
        self.lines_swath = QDoubleSpinBox()
        self.lines_swath.setRange(1.0, 200.0)
        self.lines_swath.setDecimals(1)
        self.lines_swath.setValue(suggest_swath_mm("chain"))
        self.lines_swath.setSuffix(" mm")
        self.lines_swath.setToolTip("How wide a strip along the section is read: wider finds fainter lines, as long "
                                    "as they run straight across it. Without a scale, 60 px.")
        row.addWidget(self.lines_swath)
        row.addStretch(1)
        lv.addLayout(row)
        self.lines_msg = label("", "caption", wrap=True)
        self.lines_msg.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lv.addWidget(self.lines_msg)
        bv.addWidget(self.lines_box)
        self.lines_box.setVisible(False)
        self.line_count = None
        self._swath_set = False
        for w in (self.lines_kind, self.lines_source, self.lines_swath):
            w.setEnabled(False)

        def kind_changed(_i):
            if not self._swath_set:
                self.lines_swath.blockSignals(True)
                self.lines_swath.setValue(suggest_swath_mm(self.lines_kind.currentData()))
                self.lines_swath.blockSignals(False)
            self._ask_lines()

        def swath_changed(_v):
            self._swath_set = True
            self._ask_lines()
        self.chk_lines.toggled.connect(self._lines_toggled)
        self.lines_kind.currentIndexChanged.connect(kind_changed)
        self.lines_source.currentIndexChanged.connect(lambda _i: self._ask_lines())
        self.lines_swath.valueChanged.connect(swath_changed)

    def enable_lines(self, has_backlit: bool = False):
        """Offer the chain-line counter on this card; *has_backlit*: a lined-up backlit frame is there."""
        self.lines_box.setVisible(True)
        self.set_backlit(has_backlit)

    def set_backlit(self, has: bool):
        """The backlit frame there or not: chosen when it comes, unavailable while it is not."""
        from PySide6.QtGui import QStandardItemModel
        i = self.lines_source.findData("backlit")
        model = self.lines_source.model()
        if isinstance(model, QStandardItemModel):
            model.item(i).setEnabled(bool(has))
        was = getattr(self, "_had_backlit", None)
        self._had_backlit = bool(has)
        self.lines_source.blockSignals(True)
        if has and was is not True:
            self.lines_source.setCurrentIndex(i)
        elif not has and self.lines_source.currentData() == "backlit":
            self.lines_source.setCurrentIndex(self.lines_source.findData("albedo"))
        self.lines_source.blockSignals(False)

    def set_view_name(self, name: str):
        """Name the view Inspect is showing on the "This view" choice."""
        i = self.lines_source.findData("view")
        if i >= 0:
            self.lines_source.setItemText(i, f"This view: {name}" if name else "This view (as shown)")

    def lines_options(self) -> dict | None:
        """What the counter is asked for, or None while it is off."""
        if not self.chk_lines.isChecked():
            return None
        return dict(kind=self.lines_kind.currentData(), source=self.lines_source.currentData(),
                    swath_mm=float(self.lines_swath.value()))

    def _lines_toggled(self, on: bool):
        for w in (self.lines_kind, self.lines_source, self.lines_swath):
            w.setEnabled(bool(on))
        if not on:
            self.set_line_count(None)
        self._ask_lines()

    def _ask_lines(self):
        if self.prof and self.chk_lines.isChecked():
            self.lines_requested.emit()

    def set_line_count(self, found, places=(), note: str = ""):
        """The lines found (helio.core.paper_lines.LineCount) and their places 0..1 along the section, marked
        on the chart; or None to clear them. *note*: said under the count (the picture used, its limits)."""
        self.line_count = found
        self.chart.set_lines(list(places) if found is not None else [])
        if found is None:
            self.lines_msg.setText(note)
            self.lines_msg.setToolTip("")
            self.lines_changed.emit([])
            return
        text = found.summary()
        irregular = found.irregular()
        if irregular:
            text += f" · {len(irregular)} spacing{'s' if len(irregular) != 1 else ''} out of rhythm"
        if found.count >= 2 and found.mm_per_px:
            text += "  ·  spacings " + ", ".join(f"{s * found.mm_per_px:.1f}" for s in found.spacings[:14]) + " mm"
            if len(found.spacings) > 14:
                text += " …"
        # Two lines at most, so the relief above keeps its room; every note is in the tooltip.
        self.lines_msg.setText(text + (f"\n{note}" if note else ""))
        self.lines_msg.setToolTip("\n".join([text] + [n for n in found.notes if n] + ([note] if note else [])))
        self.lines_changed.emit(list(places))

    FOLD_KEY = "section_card/folded"

    @classmethod
    def _stored_fold(cls) -> bool:
        try:
            from helio_qt.window import settings
            return str(settings().value(cls.FOLD_KEY, False)).lower() in ("true", "1")
        except Exception:
            return False

    def set_folded(self, on: bool):
        """Only the head line (title, summary, buttons), or the whole card."""
        on = bool(on)
        if self.btn_fold.isChecked() != on:
            self.btn_fold.setChecked(on)         # comes back here through toggled
            return
        self.body.setVisible(not on)
        self.btn_big.setVisible(not on)
        self.btn_fold.setText("Unfold" if on else "Fold")
        try:
            from helio_qt.window import settings
            settings().setValue(self.FOLD_KEY, on)
        except Exception:
            pass

    def set_letters(self, a: str, b: str):
        """Name the section's ends (A–B, then C–D, E–F … for the next ones kept for the sheet)."""
        self.letters = (a, b)
        self.title.setText(f"Section {a}–{b}")
        self.chart.letters = (a, b)
        self.chart.update()

    def set_cursor(self, t):
        self.chart.set_cursor(t)

    def _spacing_edited(self):
        self._spacing_set = True
        self._ask_series()

    def _ask_series(self):
        if self.prof:
            self.series_requested.emit(int(self.count.value()), float(self.spacing.value()))

    def set_series(self, series: dict | None):
        """The copies computed for the section, or None: shown stacked or overlaid under the chart."""
        self.series = series if series and len(series.get("profiles") or []) > 1 else None
        self.series_chart.set_series(self.series, self.chk_stack.isChecked())
        self.series_chart.setVisible(self.series is not None)
        if self.series is None:
            self.series_msg.setText("")
            return
        lu = self.series["length_unit"]
        drop = self.series["dropped"]
        self.series_msg.setText(
            f"{len(self.series['profiles'])} parallel sections, {self.series['spacing']:g} {lu} apart"
            + (f"; {drop} would run off the surface and are left out" if drop else "")
            + ". Offsets: positive is to the right looking from A to B.")

    def set_profile(self, prof: dict | None):
        """A section drawn, moved or cleared: the card shown with it, or hidden."""
        self.chart.set_profile(prof or None)
        self.setVisible(bool(prof))
        self.prof = prof or None
        if not prof:
            self.set_series(None)
            if self.line_count is not None:
                self.set_line_count(None)
            return
        if self.chk_lines.isChecked():
            if prof.get("features") is None:           # an end being dragged: the old lines no longer lie there
                if self.line_count is not None:
                    self.set_line_count(None, note="Counting when the line is let go…")
            else:
                self._ask_lines()
        lu = prof["length_unit"]
        self.spacing.setSuffix(f" {lu}")
        if not self._spacing_set:                      # a sensible step until the person chooses one
            step = float(f"{prof['length'] / 10.0:.2g}") or 1.0
            self.spacing.blockSignals(True)
            self.spacing.setValue(step)
            self.spacing.blockSignals(False)
        if prof.get("features") is not None and self.count.value() > 1:
            self._ask_series()                         # the line was let go: its copies follow
        zu = prof["unit"].split()[0]
        self.summary.setText(
            f"{prof['length']:.3g} {prof['length_unit']} long  ·  Ra {prof['ra']:.3g}  ·  "
            f"Rq {prof['rq']:.3g}  ·  Rz {prof['rz']:.3g} {zu}"
            + ("" if prof["unit"] != "px" else "  (set a scale for mm)"))
        if prof.get("features") is None:
            return                        # an end being dragged: the list follows on release
        fs = prof["features"]
        if not fs:
            self.features.setText("No grooves or ridges stand out from the surface's "
                                  "grain along this line.")
            return
        rows = "".join(
            f"<tr><td>{f['n']}</td><td>{'▼ groove' if f['kind'] == 'groove' else '▲ ridge'}</td>"
            f"<td align='right'>{f['at']:.3g}</td><td align='right'>{f['depth']:.3g}</td>"
            f"<td align='right'>{f['width']:.3g}</td></tr>" for f in fs[:12])
        more = f"<br>…and {len(fs) - 12} more in the saved list." if len(fs) > 12 else ""
        lu = prof["length_unit"]
        self.features.setText(
            f"<table cellspacing='0' cellpadding='2'><tr><th align='left'>#</th>"
            f"<th align='left'>feature</th><th>at ({lu})</th><th>depth ({zu})</th>"
            f"<th>width ({lu})</th></tr>{rows}</table>{more}")

    def save(self, win, name: str):
        """The section: profile CSV, features CSV, chart SVG and PNG, asked where."""
        from PySide6.QtWidgets import QFileDialog
        from helio_qt.publish import file_stem
        prof = self.prof
        if not prof:
            return []
        start = os.path.join(str(win.prefs.value("last_folder", os.path.expanduser("~"))),
                             f"{file_stem(name)}_section.csv")
        path, _ = QFileDialog.getSaveFileName(win, "Save the section", start, "CSV (*.csv)")
        if not path:
            return []
        base = os.path.splitext(path)[0]
        zu = prof["unit"].split()[0]
        with open(base + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            la, lb = self.letters
            w.writerow([f"# Section {la}-{lb}, {prof['length']:.6g} {prof['length_unit']}; "
                        f"{la} at {prof['a_px'][0]:.1f}, {prof['a_px'][1]:.1f} px; "
                        f"{lb} at {prof['b_px'][0]:.1f}, {prof['b_px'][1]:.1f} px; height in {prof['unit']}"])
            w.writerow([f"distance_{prof['length_unit']}", f"height_{zu}"])
            for dv, zv in zip(prof["dist"], prof["z"]):
                w.writerow([f"{dv:.6g}", f"{zv:.6g}"])
        files = [base + ".csv"]
        if prof.get("features"):
            with open(base + "_features.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["n", "kind", f"at_{prof['length_unit']}", f"depth_{zu}",
                            f"width_{prof['length_unit']}"])
                for ft in prof["features"]:
                    w.writerow([ft["n"], ft["kind"], f"{ft['at']:.6g}", f"{ft['depth']:.6g}",
                                f"{ft['width']:.6g}"])
            files.append(base + "_features.csv")
        if self.series:
            from helio.application.section_export import series_csv
            series_csv(base + "_series.csv", self.series["profiles"])
            files.append(base + "_series.csv")
        files += self.chart.save_pictures(base + "_chart")
        win.flash(f"Section saved: {', '.join(os.path.basename(f) for f in files)}.", 10)
        return files

    def save_drawing(self, win, name: str, extra: dict | None = None):
        """The section drawing: SVG (to a stated scale, with a title block), DXF, and a series' CSV, asked where."""
        from PySide6.QtWidgets import QFileDialog
        from helio.application.section_export import section_dxf, section_svg, series_csv
        from helio_qt.publish import file_stem
        prof = self.prof
        if not prof:
            return []
        start = os.path.join(str(win.prefs.value("last_folder", os.path.expanduser("~"))),
                             f"{file_stem(name)}_section_drawing.svg")
        path, _ = QFileDialog.getSaveFileName(win, "Save the section drawing", start, "SVG drawing (*.svg)")
        if not path:
            return []
        return self.write_drawing(os.path.splitext(path)[0], name, win.project, extra)

    def write_drawing(self, base: str, name: str, project=None, extra: dict | None = None) -> list[str]:
        """Write base.svg, base.dxf (and base_series.csv for copies). Returns the files."""
        from helio.application.section_export import section_dxf, section_svg, series_csv
        profs = self.series["profiles"] if self.series else [self.prof]
        stacked = self.chk_stack.isChecked()
        meta = dict(extra or {})
        meta.setdefault("Object", name)
        r = getattr(project, "result", None)
        if r is not None:
            meta.setdefault("Solved with", f"{r.settings.resolved_normal_engine()} · {r.settings.resolved_depth_engine()}")
            if getattr(r, "work_scale", 1.0) < 1.0:
                meta.setdefault("Working size", f"{r.work_scale * 100:.0f}% of the photographs")
        ve = float(self.vexag.value())
        info = section_svg(base + ".svg", profs, title=f"{name}: section {self.letters[0]}–{self.letters[1]}" + (f" and {len(profs) - 1} parallel" if len(profs) > 1 else ""),
                           meta=meta, v_exag=ve, stacked=stacked)
        section_dxf(base + ".dxf", profs, v_exag=ve, stacked=stacked, title=name)
        files = [base + ".svg", base + ".dxf"]
        if self.series:
            series_csv(base + "_series.csv", profs)
            files.append(base + "_series.csv")
        self.last_drawing = info
        return files

    def attach(self, get_view, win, name=lambda: ""):
        """Connect this card to a 3D view (a callable, since the view may be made later): copies are
        sampled on it, drawn on it, and cleared with the section; Drawing… writes the files."""
        def ask(count, spacing):
            view = get_view()
            if view is None:
                return
            if count <= 1:
                view.clear_series()
                self.set_series(None)
                return
            self.set_series(view.section_series(count, spacing))
        self.series_requested.connect(ask)

        def clear():
            view = get_view()
            if view is not None:
                view.clear_series()
            self.set_series(None)
        self.clear_requested.connect(clear)

        def draw():
            files = self.save_drawing(win, name(), None)
            if files:
                win.flash(f"Section drawing saved: {', '.join(os.path.basename(f) for f in files)}.", 10)
        self.draw_requested.connect(draw)
