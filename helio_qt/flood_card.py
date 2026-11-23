"""The flood plane's card: a level on a slider, and what lies below it.

Under the 3D relief (Inspect's, and the high-detail window's) while the Flood tool is on: the plane's height above
the surface's lowest point (slider, number, or "set from the surface": click a hollow's rim in 3D), the part of the
surface to measure (all of it, or inside the Area outline), and the figures: the area below the plane, what it
would hold, the deepest and mean depth, and each separate patch with whether its rim holds it (closed) or it runs
off the edge (open).  Save writes them as CSV.
"""
from __future__ import annotations

import csv
import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QSlider, QVBoxLayout)

from helio_qt.widgets import button, label

STEPS = 1000


def _vol_text(v: float, unit: str) -> str:
    if unit == "mm":
        ml = v / 1000.0
        return f"{v:,.3g} mm³ ({ml:.3g} ml)" if v >= 1.0 else f"{v:.3g} mm³"
    return f"{v:,.3g} px³"


def _area_text(a: float, unit: str) -> str:
    return f"{a:,.4g} {unit}²"


class FloodCard(QFrame):
    level_changed = Signal(float, bool)       # height above the lowest point, and whether a drag is still going
    region_changed = Signal(str)              # "all" or "area"
    pick_requested = Signal()
    close_requested = Signal()
    save_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("FloodCard")
        self.info = None
        self._range, self._unit = 1.0, "mm"
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        head = QHBoxLayout()
        head.addWidget(label("Flood plane", "section"))
        self.summary = label("", "secondary")
        head.addWidget(self.summary, 1)
        self.btn_pick = button("Set from the surface", "quiet",
                               "Then click the relief: the plane goes to the height of that point. Click a hollow's rim to fill it to the rim.")
        self.btn_pick.clicked.connect(self.pick_requested.emit)
        head.addWidget(self.btn_pick)
        b = button("Save…", "quiet", "The figures and each patch as a CSV.")
        b.clicked.connect(self.save_requested.emit)
        head.addWidget(b)
        b = button("Close", "quiet", "Take the plane away.")
        b.clicked.connect(self.close_requested.emit)
        head.addWidget(b)
        v.addLayout(head)
        row = QHBoxLayout()
        row.addWidget(label("Level", "secondary"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.setToolTip("The plane's height above the surface's lowest point.")
        row.addWidget(self.slider, 1)
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(3)
        self.spin.setRange(0.0, 1e9)
        self.spin.setToolTip("Height above the lowest point.")
        row.addWidget(self.spin)
        row.addWidget(label("measure", "secondary"))
        self.region = QComboBox()
        self.region.addItem("the whole surface", "all")
        self.region.addItem("inside the Area outline", "area")
        self.region.setToolTip("Draw and close an outline with the Area tool first, then choose it here to measure just that part.")
        row.addWidget(self.region)
        v.addLayout(row)
        self.figures = label("", "caption", wrap=True)
        self.figures.setTextFormat(Qt.TextFormat.RichText)
        self.figures.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.figures)
        self.table = label("", "caption", wrap=True)
        self.table.setTextFormat(Qt.TextFormat.RichText)
        self.table.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.table)
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(160)
        self._settle.timeout.connect(lambda: self.level_changed.emit(self.spin.value(), False))
        self.slider.valueChanged.connect(self._slid)
        self.slider.sliderReleased.connect(lambda: self.level_changed.emit(self.spin.value(), False))
        self.spin.valueChanged.connect(self._spun)
        self.region.currentIndexChanged.connect(lambda _i: self.region_changed.emit(self.region.currentData()))
        self._quiet = False
        self.setVisible(False)

    # slider and number follow each other without echoing
    def _slid(self, i):
        if self._quiet:
            return
        self._quiet = True
        self.spin.setValue(self._range * i / STEPS)
        self._quiet = False
        self.level_changed.emit(self.spin.value(), self.slider.isSliderDown())
        if not self.slider.isSliderDown():
            self._settle.start()

    def _spun(self, val):
        if self._quiet:
            return
        self._quiet = True
        self.slider.setValue(int(round(STEPS * val / self._range)) if self._range else 0)
        self._quiet = False
        self.level_changed.emit(float(val), True)
        self._settle.start()

    def set_info(self, info: dict | None):
        """The plane's figures (``ReliefView.flood_info``): empty or None when the plane is off."""
        if not info:
            self.info = None
            self.setVisible(False)
            return
        self.setVisible(True)
        prev = self.info
        self.info = info
        unit = info["unit"]
        self._range, self._unit = max(float(info["range"]), 1e-12), unit
        self._quiet = True
        self.spin.setRange(0.0, self._range)
        self.spin.setSuffix(f" {unit}")
        self.spin.setSingleStep(self._range / 100.0)
        if not self.spin.hasFocus():
            self.spin.setValue(info["level"])
        if not self.slider.isSliderDown():
            self.slider.setValue(int(round(STEPS * info["level"] / self._range)))
        self._quiet = False
        self.summary.setText(f"{info['level']:.3g} {unit} above the lowest point (of {self._range:.3g} {unit} in all)")
        pct = 100.0 * info["fraction"]
        if not info["cells"]:
            self.figures.setText(f"Nothing in {info['region']} is below the plane.")
        else:
            self.figures.setText(
                f"<b>Below the plane:</b> {_area_text(info['area'], unit)} ({pct:.3g}% of {info['region']}); "
                f"it would hold <b>{_vol_text(info['volume'], unit)}</b>; deepest {info['deepest']:.3g} {unit}, "
                f"mean depth {info['mean_depth']:.3g} {unit}.")
        bs = info.get("basins")
        if bs is None:
            if prev is None or not prev.get("basins"):
                self.table.setText("")
            return                                    # a drag: the patches follow when it is let go
        if not bs:
            self.table.setText("")
            return
        rows = "".join(
            f"<tr><td>{i + 1}</td><td align='right'>{b['area']:.3g}</td><td align='right'>{b['volume']:.3g}</td>"
            f"<td align='right'>{b['deepest']:.3g}</td><td>{'closed: its rim holds it' if b['closed'] else 'open: runs off the edge'}</td></tr>"
            for i, b in enumerate(bs[:8]))
        self.table.setText(
            f"<table cellspacing='0' cellpadding='2'><tr><th align='left'>patch</th><th>area ({unit}²)</th>"
            f"<th>holds ({unit}³)</th><th>deepest ({unit})</th><th align='left'>&nbsp;</th></tr>{rows}</table>"
            + (f"…and {len(bs) - 8} more in the saved list." if len(bs) > 8 else ""))

    def save(self, win, name: str) -> list[str]:
        from PySide6.QtWidgets import QFileDialog
        from helio_qt.publish import file_stem
        if not self.info:
            return []
        start = os.path.join(str(win.prefs.value("last_folder", os.path.expanduser("~"))), f"{file_stem(name)}_flood.csv")
        path, _ = QFileDialog.getSaveFileName(win, "Save the flood figures", start, "CSV (*.csv)")
        return [self.write_csv(path)] if path else []

    def write_csv(self, path: str) -> str:
        i, u = self.info, self.info["unit"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([f"# flood plane at {i['level']:.6g} {u} above the lowest point; {i['region']}; heights in {u}"])
            w.writerow(["quantity", "value", "unit"])
            for k, val, un in (("level_above_lowest", i["level"], u), ("surface_height_range", i["range"], u),
                               ("region_area", i["region_area"], f"{u}2"), ("area_below", i["area"], f"{u}2"),
                               ("fraction_below", i["fraction"], ""), ("volume_below", i["volume"], f"{u}3"),
                               ("deepest", i["deepest"], u), ("mean_depth", i["mean_depth"], u), ("cell_size", i["cell"], u)):
                w.writerow([k, f"{val:.8g}", un])
            w.writerow([])
            w.writerow(["patch", f"area_{u}2", f"volume_{u}3", f"deepest_{u}", "closed", "centre_u", "centre_v"])
            for n, b in enumerate(i.get("basins") or [], 1):
                w.writerow([n, f"{b['area']:.8g}", f"{b['volume']:.8g}", f"{b['deepest']:.8g}",
                            "yes" if b["closed"] else "no", f"{b['uv'][0]:.5f}", f"{b['uv'][1]:.5f}"])
        return path

    def attach(self, get_view, win=None, name=lambda: ""):
        """Connect to a 3D view (a callable, the view may be made later) and back."""
        def view():
            return get_view()
        self.level_changed.connect(lambda lv, live: view() and view().set_flood_level(lv, live))
        self.region_changed.connect(lambda r: view() and view().set_flood_region(r))
        self.pick_requested.connect(lambda: view() and view().begin_flood_pick())
        self.close_requested.connect(lambda: view() and view().set_flood(False))
        if win is not None:
            def save():
                files = self.save(win, name())
                if files:
                    win.flash(f"Flood figures saved: {os.path.basename(files[0])}.", 8)
            self.save_requested.connect(save)
