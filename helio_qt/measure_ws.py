"""Measure: scale, distances, heights, profiles, areas, volumes and surface
texture (MET-001…MET-015).

Every figure carries its unit, and says when it cannot have one: lengths
are in pixels until a scale is set; heights are true heights from the
normals (helio.application.heights), in pixels of height until a height
scale (mm per pixel of height) is set.  Region tools (area and volume, surface statistics) work on a rectangle
dragged over the image.  The arithmetic is helio.core.metrology, shared
with the classic app (with the classic volume, area and Str faults fixed:
KD-023, KD-024, KD-025).
"""
from __future__ import annotations

from helio.core.frame_read import read_frame

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QHBoxLayout, QListWidgetItem,
                               QVBoxLayout, QWidget)

from helio_qt.controls import Section, Segmented, VScroll
from helio_qt.widgets import Viewport, button, fit_list, label, panel, rule


class MeasureWorkspace:
    name = "Measure"
    TOOLS = (("Set scale", "measure.scale"), ("Scale from a ruler", "measure.ruler"),
             ("Measure distance", "measure.distance"), ("Height scale", "measure.zscale"),
             ("Height profile", "measure.profile"), ("Area and volume", "measure.volume"),
             ("Surface statistics", "measure.stats"), ("Compare two regions", "measure.compare"),
             ("Accuracy check", "measure.accuracy"))
    TOOL_OF = {"measure.scale": "scale", "measure.ruler": "ruler", "measure.distance": "distance",
               "measure.zscale": "zscale", "measure.profile": "profile",
               "measure.volume": "volume", "measure.stats": "stats", "measure.compare": "compare",
               "measure.accuracy": "accuracy"}

    def __init__(self, win):
        self.win, self.project, self.reg = win, win.project, win.registry
        self.sidebar, self.center, self.inspector = self.build()

    def build(self):
        from helio_qt.workspaces import INSPECTOR_W, SIDE_W, _list
        from helio_qt.widgets import ProfileChart, ScaleBar
        store = self.win.prefs
        side, sl = panel("Sidebar", SIDE_W)
        sl.addWidget(label("TOOLS", "heading"))
        self.tools = _list([])
        self.tools.itemClicked.connect(self._tool_picked)
        sl.addWidget(self.tools)
        self.tool_note = label("", "caption", wrap=True)
        sl.addWidget(self.tool_note)
        sl.addStretch(1)

        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        self.viewport = Viewport("Import images to measure them")
        self.viewport.lineDrawn.connect(self._line_drawn)
        self.viewport.rectDrawn.connect(self._rect_drawn)
        self.scalebar = ScaleBar(self.viewport, lambda: self.project.photo_scale())   # this tab's pixels are the photograph's
        self.viewport.add_overlay(self.scalebar, "bc")
        cl.addWidget(self.viewport)
        self.show_seg = Segmented([("photo", "Photograph"), ("relief", "Relief"),
                                   ("depth", "Height")],
                                  {"photo": "The frame as photographed.",
                                   "relief": "The solved surface, raking-lit, to see what you measure.",
                                   "depth": "The height map."})
        self.show_seg.set_current("photo")
        self.show_seg.chosen.connect(lambda k: (self.show_seg.set_current(k), self._show_image()))
        self.viewport.add_overlay(self.show_seg, "tl")

        insp, il = panel("Inspector", INSPECTOR_W)
        body = QWidget()
        vl = QVBoxLayout(body)
        vl.setContentsMargins(0, 0, 4, 0)
        vl.setSpacing(12)
        il.addWidget(VScroll(body), 1)

        sec = Section("Scale", "m_scale", store)
        self.scale = label("", "secondary", wrap=True)
        self.scale.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        sec.add(self.scale)
        self.btn_clear_z = button("Clear the height scale", "quiet",
                                  "Heights go back to relative units.")
        self.btn_clear_z.clicked.connect(self.clear_zscale)
        sec.add(self.btn_clear_z)
        vl.addWidget(sec)

        # Set scale (line) and Scale from a ruler (ticks)
        self.scale_box = QWidget()
        sb = QVBoxLayout(self.scale_box)
        sb.setContentsMargins(0, 0, 0, 0)
        sb.setSpacing(6)
        sb.addWidget(label("Set the scale", "section"))
        self.line_len = label("", "caption", wrap=True)
        sb.addWidget(self.line_len)
        row = QHBoxLayout()
        self.length = QDoubleSpinBox()
        self.length.setRange(0.0, 100000.0)
        self.length.setDecimals(3)
        self.length.setValue(10.0)
        self.length.setAccessibleName("Real length")
        self.length.setToolTip("The real length of the line drawn on the image.")
        self.unit = QComboBox()
        self.unit.addItems(["mm", "cm", "in"])
        self.unit.setAccessibleName("Unit")
        self.unit.setToolTip("The unit of that length.")
        row.addWidget(self.length, 1)
        row.addWidget(self.unit)
        sb.addLayout(row)
        self.btn_set = button("Set scale", "", "Millimetres per pixel from the line and its "
                              "real length.")
        self.btn_set.clicked.connect(self._set_scale)
        sb.addWidget(self.btn_set)
        self.scale_err = label("", "reason", wrap=True)
        sb.addWidget(self.scale_err)
        b = button("Print scale bars…", "quiet", "Detailed scale bars to print, cut out and lay beside "
                   "the object: 20 mm in half millimetres up to the longest the paper takes.")
        b.clicked.connect(lambda: self.reg.trigger("measure.print_scale"))
        sb.addWidget(b)
        vl.addWidget(self.scale_box)

        # Height scale
        self.z_box = QWidget()
        zb = QVBoxLayout(self.z_box)
        zb.setContentsMargins(0, 0, 0, 0)
        zb.setSpacing(6)
        zb.addWidget(label("Set the height scale", "section"))
        self.z_line = label("", "caption", wrap=True)
        zb.addWidget(self.z_line)
        self.z_known = QDoubleSpinBox()
        self.z_known.setRange(0.0, 1000.0)
        self.z_known.setDecimals(3)
        self.z_known.setValue(1.0)
        self.z_known.setSuffix(" mm")
        self.z_known.setAccessibleName("Known height of the feature")
        self.z_known.setToolTip("The real height of the feature profiled, measured another way.")
        zb.addWidget(self.z_known)
        self.btn_z = button("Set height scale", "", "Heights in millimetres from the profile and "
                            "the feature's known height.")
        self.btn_z.clicked.connect(self._set_zscale)
        zb.addWidget(self.btn_z)
        self.z_err = label("", "reason", wrap=True)
        zb.addWidget(self.z_err)
        self.btn_z_lateral = button("Use the lateral scale (calibrated capture)", "quiet",
                                    "For a capture shot to measure (RAW, the white-board light field, "
                                    "lens corrected): the normals' slopes are true, so a pixel of "
                                    "height is a pixel of width. Check it with the accuracy check.")
        self.btn_z_lateral.clicked.connect(self.z_from_lateral)
        zb.addWidget(self.btn_z_lateral)
        vl.addWidget(self.z_box)

        # Distances
        self.dist_box = QWidget()
        db = QVBoxLayout(self.dist_box)
        db.setContentsMargins(0, 0, 0, 0)
        db.addWidget(label("Distances", "section"))
        self.results = label("Distances you measure appear here.", "caption", wrap=True)
        self.results.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        db.addWidget(self.results)
        vl.addWidget(self.dist_box)

        # Height profile
        self.profile_box = QWidget()
        pb = QVBoxLayout(self.profile_box)
        pb.setContentsMargins(0, 0, 0, 0)
        pb.setSpacing(6)
        pb.addWidget(label("Height profile", "section"))
        self.chart = ProfileChart()
        pb.addWidget(self.chart)
        self.rough = label("", "secondary", wrap=True)
        self.rough.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pb.addWidget(self.rough)
        self.btn_csv = button("Save profile as CSV…", "", "The profile's distances and heights, "
                              "for a spreadsheet.")
        self.btn_csv.clicked.connect(self._save_profile)
        pb.addWidget(self.btn_csv)
        vl.addWidget(self.profile_box)

        # Region results: area and volume, or surface statistics
        self.region_box = QWidget()
        rb = QVBoxLayout(self.region_box)
        rb.setContentsMargins(0, 0, 0, 0)
        rb.setSpacing(6)
        self.region_title = label("", "section")
        rb.addWidget(self.region_title)
        self.region_text = label("", "secondary", wrap=True)
        self.region_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.region_text.setProperty("role", "mono")
        rb.addWidget(self.region_text)
        self.bearing = ProfileChart()
        self.bearing.setAccessibleName("Bearing area curve")
        rb.addWidget(self.bearing)
        self.btn_whole = button("Use the whole image", "quiet", "Area, volume and roughness of "
                                "the whole image, not a drawn region.")
        self.btn_whole.clicked.connect(self._whole_region)
        rb.addWidget(self.btn_whole)
        self.btn_region_csv = button("Save these figures as CSV…", "quiet", "The roughness "
                                     "figures, for a spreadsheet.")
        self.btn_region_csv.clicked.connect(self._save_region_csv)
        rb.addWidget(self.btn_region_csv)
        vl.addWidget(self.region_box)

        # Accuracy check on a reference (CAL-007): a gauge block's step, a flat plate
        self.acc_box = QWidget()
        ab = QVBoxLayout(self.acc_box)
        ab.setContentsMargins(0, 0, 0, 0)
        ab.setSpacing(6)
        ab.addWidget(label("Accuracy check on a reference", "section"))
        ab.addWidget(label("Box 1 on the reference surface (the base, or a flat plate); box 2 on the "
                           "top of the step. Box 1 alone checks flatness.", "caption", wrap=True))
        self.acc_known = QDoubleSpinBox()
        self.acc_known.setRange(0.0, 100.0)
        self.acc_known.setDecimals(4)
        self.acc_known.setValue(1.0)
        self.acc_known.setSuffix(" mm")
        self.acc_known.setAccessibleName("Known step height")
        self.acc_known.setToolTip("The step's certified height: a gauge block's size, a shim's "
                                  "measured thickness.")
        self.acc_known.valueChanged.connect(lambda _v: self.accuracy_check())
        ab.addWidget(self.acc_known)
        self.acc_text = label("", "secondary", wrap=True)
        self.acc_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.acc_text.setProperty("role", "mono")
        ab.addWidget(self.acc_text)
        self.btn_acc_again = button("Start again", "quiet", "Draw both boxes again.")
        self.btn_acc_again.clicked.connect(lambda: self.use("accuracy"))
        ab.addWidget(self.btn_acc_again)
        vl.addWidget(self.acc_box)
        vl.addStretch(1)

        self.z_diff = 0.0
        self.profile = None
        self.tool = ""
        self.drawn_px = 0.0
        self.measured: list[tuple[float, float | None]] = []
        # where each measured line was, in the solve's pixels, for the publication sheet
        self.lines: list[tuple[float, float, float, float, str]] = []
        self.region_rows: list[tuple[str, str]] = []
        self.compare_boxes: list = []
        self.compare_stats = None
        self.acc_boxes: list = []
        self.acc_rows: list[tuple[str, str]] = []
        self._shown_key = None
        return side, center, insp

    def recolour(self, t):
        from PySide6.QtGui import QColor
        self.viewport.recolour(t)
        self.viewport.line_colour = QColor(t.accent)
        self.chart.recolour(t)
        self.bearing.recolour(t)
        self.scalebar.recolour(t)

    # ── coordinates ───────────────────────────────────────────────────
    # Everything on this tab is drawn and measured in the PHOTOGRAPH's pixels, whichever picture is shown
    # (the photograph, or the solved relief laid over it at the size of the photograph).  A solve at a
    # reduced working size has fewer pixels (ws of them per photograph pixel, and a region solve starts at
    # crop, in the result's pixels), so the solved data is read through these two conversions, and
    # lengths use photo_scale(), the size of a photograph's pixel.
    def _ws(self) -> float:
        return self.project.result_work_scale()

    def _crop0(self):
        r = self.project.result
        return (r.crop[0], r.crop[1]) if r is not None and r.crop else (0, 0)

    def _to_result(self, x, y):
        """A point in the photograph's pixels, in the solve's pixels."""
        ws, (ox, oy) = self._ws(), self._crop0()
        return x * ws - ox, y * ws - oy

    def result_box(self, x1, y1, x2, y2):
        """A rectangle on the photograph (its pixels) in the solve's pixels."""
        r = self.project.result
        ax, ay = self._to_result(min(x1, x2), min(y1, y2))
        bx, by = self._to_result(max(x1, x2), max(y1, y2))
        return max(0, int(round(ax))), max(0, int(round(ay))), min(r.w, int(round(bx))), min(r.h, int(round(by)))

    def _result_line(self, x1, y1, x2, y2):
        ax, ay = self._to_result(x1, y1)
        bx, by = self._to_result(x2, y2)
        return ax, ay, bx, by

    # ── tools ─────────────────────────────────────────────────────────
    def use(self, tool: str):
        self.tool = tool
        kind = {"scale": "line", "distance": "line", "zscale": "line", "profile": "line",
                "ruler": "rect", "volume": "rect", "stats": "rect",
                "compare": "rect", "accuracy": "rect"}.get(tool, "")
        self.viewport.set_tool(kind)
        self.viewport.clear_line()
        self.viewport.clear_rect()
        if tool in ("volume", "stats", "compare"):
            self.region_rows = []
        self.compare_boxes = []
        if tool == "accuracy":
            self.acc_boxes, self.acc_rows = [], []
        self.viewport.setFocus()
        self.refresh()
        self.win.update_bottom()

    def _tool_picked(self, item):
        name = item.text().replace("›", "").strip()
        for text, aid in self.TOOLS:
            if text == name:
                self.reg.trigger(aid)

    def _line_drawn(self, x1, y1, x2, y2):
        from helio.core.metrology import distance, line_length_px
        if self.tool == "scale":
            self.drawn_px = line_length_px(x1, y1, x2, y2)
            self.scale_err.setText("")
        elif self.tool == "distance":
            px, mm = distance(x1, y1, x2, y2, self.project.photo_scale())      # the photograph's pixels
            self.measured.append((px, mm))
            self.lines.append((*self._result_line(x1, y1, x2, y2),
                               f"{mm:.2f} mm" if mm else f"{px:.0f} px"))
        elif self.tool in ("zscale", "profile") and self.project.result is not None:
            from helio.core.metrology import height_reference, profile_line, profile_stats
            from helio.application.heights import height_px
            z, _ends, num = profile_line(height_px(self.project.result),
                                         *self._result_line(x1, y1, x2, y2))
            if z is None:
                self.z_err.setText("Line too short: draw a longer line across the surface.")
            elif self.tool == "zscale":
                self.z_diff = height_reference(z)
                self.z_err.setText("")
            else:
                import math
                self.profile = profile_stats(z, num, self.project.scale_mm_per_px,
                                             self.project.z_mm_per_unit,
                                             length_px=math.hypot(_ends[2] - _ends[0], _ends[3] - _ends[1]))
                from helio.core.metrology import path_height_sigma
                ends = self._result_line(x1, y1, x2, y2)
                self.profile["sigma_px"] = path_height_sigma(
                    getattr(self.win.render_data, "qc_sigma_full", None), *ends)
        self.refresh()
        self.win.update_bottom()

    def _rect_drawn(self, x1, y1, x2, y2):
        if self.tool == "ruler":
            return self._ruler(x1, y1, x2, y2)
        if self.tool in ("volume", "stats") and self.project.result is not None:
            self.measure_region(self.result_box(x1, y1, x2, y2))
        elif self.tool == "accuracy" and self.project.result is not None:
            box = self.result_box(x1, y1, x2, y2)
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                self.win.flash("Draw a larger box.")
                return
            self.acc_boxes = [box] if len(self.acc_boxes) >= 2 else self.acc_boxes + [box]
            self.accuracy_check()
        elif self.tool == "compare" and self.project.result is not None:
            box = self.result_box(x1, y1, x2, y2)
            if box[2] - box[0] < 5 or box[3] - box[1] < 5:
                self.win.flash("Draw a larger rectangle.")
                return
            self.compare_boxes = (self.compare_boxes + [box])[-2:] if len(self.compare_boxes) < 2 \
                else [box]
            self.compare_regions()

    def _whole_region(self):
        r = self.project.result
        if r is not None:
            self.viewport.clear_rect()
            self.measure_region((0, 0, r.w, r.h))

    def measure_region(self, box):
        """Area and volume, or ISO 25178 statistics, of a box in the solve's pixels."""
        from helio.core.metrology import areal, true_surface_area, volume
        r, p = self.project.result, self.project
        x1, y1, x2, y2 = box
        if x2 - x1 < 8 or y2 - y1 < 8:
            self.region_rows = [("Region", "too small: drag at least 8 × 8 pixels")]
            self.refresh()
            return
        from helio.application.heights import height_px
        d = height_px(r)[y1:y2, x1:x2]                     # pixels of true height, not the display depth
        n = r.normals_full[y1:y2, x1:x2]
        mm, zmm = p.scale_mm_per_px, p.z_mm_per_unit
        rows = [("Region", f"{x2 - x1} × {y2 - y1} px" +
                 (f"  ({(x2 - x1) * mm:.2f} × {(y2 - y1) * mm:.2f} mm)" if mm else ""))]
        self.bearing.setVisible(False)
        if self.tool == "volume":
            a = true_surface_area(n, mm)
            v = volume(d, mm, zmm, base="plane")
            if mm:
                rows += [("True surface area", f"{a['true_mm2']:,.2f} mm²"),
                         ("Projected area", f"{a['projected_mm2']:,.2f} mm²")]
            else:
                rows += [("True surface area", f"{a['true_px2']:,.0f} px²  (no scale)")]
            rows.append(("Surface excess", f"+{a['excess_pct']:.2f}%"))
            if "above_mm3" in v:
                rows += [("Material above the base plane", f"{v['above_mm3']:.4f} mm³"),
                         ("Space below it", f"{v['below_mm3']:.4f} mm³"),
                         ("Highest / deepest", f"{v['max_height_mm']:.4f} / {v['max_depth_mm']:.4f} mm")]
            else:
                rows += [("Volume", "in mm³ with the scale and the height scale"),
                         ("Above / below the base (px³)",
                          f"{v['above_units']:.4g} / {v['below_units']:.4g}"),
                         ("Highest / deepest (px of height)",
                          f"{v['max_height_units']:.3g} / {v['max_depth_units']:.3g}")]
        else:
            A = areal(d, n, zmm)
            u = " mm" if A["unit"] == "mm" else " px"
            rows += [("Sa  arithmetic mean height", f"{A['Sa']:.5g}{u}"),
                     ("Sq  root mean square height", f"{A['Sq']:.5g}{u}"),
                     ("Ssk skewness", f"{A['Ssk']:.4f}"), ("Sku kurtosis", f"{A['Sku']:.4f}"),
                     ("Sp  maximum peak height", f"{A['Sp']:.5g}{u}"),
                     ("Sv  maximum pit depth", f"{A['Sv']:.5g}{u}"),
                     ("Sz  maximum height", f"{A['Sz']:.5g}{u}"),
                     ("Str texture aspect ratio", f"{A['Str']:.4f}"),
                     ("Std texture direction", f"{A['Std']:.1f}°"),
                     ("Sdq root mean square gradient", f"{A['Sdq']:.4f}"),
                     ("Sdr developed area ratio", f"{A['Sdr']:.3f}%")]
            bx, by = A["bearing"]
            self.bearing.set_profile(bx, by, "Material ratio (%)",
                                     "mm" if A["unit"] == "mm" else "px")
            self.bearing.setVisible(True)
        self.region_rows = rows
        self.refresh()
        self.win.update_bottom()

    def compare_regions(self):
        """Region A (a feature) against region B (the background): the classic
        Surface Comparison's figures (helio.application.export_reports)."""
        from helio.application.export_reports import region_stats
        r, p = self.project.result, self.project
        if len(self.compare_boxes) < 2:
            self.region_rows = [("Region A", "{} × {} px".format(
                self.compare_boxes[0][2] - self.compare_boxes[0][0],
                self.compare_boxes[0][3] - self.compare_boxes[0][1])),
                ("Next", "drag region B: the background or reference")]
            self.compare_stats = None
            self.refresh()
            return
        from helio.application.heights import height_px
        H = height_px(r)
        rois = [H[y1:y2, x1:x2] for x1, y1, x2, y2 in self.compare_boxes]
        sa = region_stats(rois[0], "ROI A", z_mm_per_unit=p.z_mm_per_unit, mm_per_px=p.scale_mm_per_px)
        sb = region_stats(rois[1], "ROI B", z_mm_per_unit=p.z_mm_per_unit, mm_per_px=p.scale_mm_per_px)
        self.compare_stats = (sa, sb, rois)
        rows = [("", f"{'A':>12} {'B':>12} {'A − B':>12}")]
        for k in ("Sa", "Sq", "Ssk", "Sku", "Sp", "Sv", "Sz"):
            rows.append((k, f"{sa[k]:>12.5g} {sb[k]:>12.5g} {sa[k] - sb[k]:>12.5g}"))
        rows.append(("Units", sa["Unit"]))
        self.region_rows = rows
        self.refresh()
        self.win.update_bottom()

    def save_comparison(self, path: str) -> list[str]:
        """The comparison CSV, and its chart as a PNG beside it."""
        import time
        from helio.application.export_reports import (surface_comparison_csv,
                                                      surface_comparison_figure)
        sa, sb, rois = self.compare_stats
        p, r = self.project, self.project.result
        surface_comparison_csv(path, sa, sb, [
            ("Object ID", p.name), ("Date", time.strftime("%Y-%m-%d")),
            ("Normal solver", r.settings.normal_engine), ("Depth engine", r.settings.depth_engine),
            ("ROI A coords", str(tuple(self.compare_boxes[0]))),
            ("ROI B coords", str(tuple(self.compare_boxes[1]))),
            ("mm/px", p.scale_mm_per_px or "not calibrated"),
            ("µm per pixel of height", (p.z_mm_per_unit * 1000) if p.z_mm_per_unit else "not calibrated"),
            ("Software", "Helio")])
        png = os.path.splitext(path)[0] + ".png"
        surface_comparison_figure(rois[0], rois[1], sa, sb, p.name, png)
        return [path, png]

    # ── precision: the accuracy check and heights from the lateral scale ──
    def _z_mm(self):
        """(mm per pixel of height, where it came from) or (None, '')."""
        p = self.project
        if p.z_mm_per_unit:
            return p.z_mm_per_unit, ("the lateral scale" if p.z_from_scale else "the height scale")
        if p.scale_mm_per_px:
            return p.scale_mm_per_px, "the lateral scale (no height scale set)"
        return None, ""

    def accuracy_check(self):
        """Flatness of box 1; with box 2, the step from box 1's plane, against the known height."""
        from helio.application.heights import height_px
        from helio.core.metrology import flatness, path_height_sigma, step_height
        r, p = self.project.result, self.project
        if r is None or not self.acc_boxes:
            self.acc_rows = []
            self.refresh()
            return
        H = height_px(r)
        z, z_from = self._z_mm()
        um = (lambda px: f"{px * z * 1000:.1f} µm") if z else (lambda px: f"{px:.3g} px of height")
        rows = []
        try:
            f = flatness(H, self.acc_boxes[0])
        except ValueError as e:
            self.acc_rows = [("Box 1", str(e))]
            self.refresh()
            return
        rows += [("Flatness of box 1  RMS", um(f["rms"])), ("  peak to valley (99%)", um(f["p99"])),
                 ("  tilt of its plane", f"{f['tilt_deg']:.2f}°")]
        acc = dict(flat_rms_mm=f["rms"] * z if z else None, flat_p99_mm=f["p99"] * z if z else None,
                   tilt_deg=f["tilt_deg"], z_from=z_from)
        if len(self.acc_boxes) >= 2:
            st = step_height(H, self.acc_boxes[0], self.acc_boxes[1])
            (ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2) = self.acc_boxes[:2]
            sig = path_height_sigma(getattr(self.win.render_data, "qc_sigma_full", None),
                                    (ax1 + ax2) / 2, (ay1 + ay2) / 2, (bx1 + bx2) / 2, (by1 + by2) / 2)
            rows.append(("Step measured", um(st["step"]) if not z else f"{st['step'] * z:.4f} mm"))
            if sig is not None:
                rows.append(("  ± from the normals (1σ)", um(sig)))
            known = float(self.acc_known.value())
            if z and known > 0:
                err = st["step"] * z - known
                rows += [("Known step", f"{known:.4f} mm"),
                         ("Error", f"{err * 1000:+.1f} µm  ({err / known * 100:+.2f}%)")]
                acc.update(step_mm=st["step"] * z, known_mm=known, error_mm=err,
                           error_pct=err / known * 100,
                           sigma_mm=None if sig is None else sig * z)
            elif not z:
                rows.append(("Error", "set a scale first: the step is in pixels of height"))
        if z:
            rows.append(("Heights from", z_from))
        self.acc_rows = rows
        if z:
            p.accuracy = acc
        self.refresh()
        self.win.update_bottom()

    def z_from_lateral(self):
        """The height scale = the lateral scale: for a calibrated, linear capture."""
        p = self.project
        if not p.scale_mm_per_px:
            self.win.flash("Set the scale first: the height scale is taken from it.")
            return
        p.z_mm_per_unit = float(p.scale_mm_per_px)
        p.z_from_scale = True
        p.changed.emit()
        self.win.flash(f"Heights from the lateral scale: {p.z_mm_per_unit * 1000:.2f} µm per pixel of "
                       f"height. Check it on a reference with the accuracy check.", 10)

    def _ruler(self, x1, y1, x2, y2):
        """Auto-detect a ruler's tick spacing (MET-010): the geometry is
        proposed, the unit is the person's to give."""
        from helio.core.paper_analysis import detect_scale_ticks
        r = self.project.result
        src = r.albedo_grey_full if r is not None else None
        src_is_result = src is not None
        if src is None:
            import cv2
            src = read_frame(self.project.images[self.project.current], cv2.IMREAD_GRAYSCALE)
            box = (int(x1), int(y1), int(x2), int(y2))
        else:
            box = self.result_box(x1, y1, x2, y2)
        try:
            res = detect_scale_ticks(src, roi=box)
        except Exception as e:
            self.scale_err.setText(f"No tick spacing found: {e}")
            self.refresh()
            return
        c = res["confidence"]
        quality = "strong" if c >= 0.6 else ("weak — check it" if c < 0.35 else "moderate")
        # the ticks were found in the result's pixels when the solved picture was used: say them in the photograph's
        self.drawn_px = float(res["period_px"]) / (self._ws() if src_is_result else 1.0)
        self.tool = "scale"
        self.viewport.set_tool("line")
        self.scale_err.setText("")
        self.length.setValue(1.0)
        self.unit.setCurrentText("mm")
        self.line_len.setText(
            f"Tick spacing {self.drawn_px:.2f} px, about {res['n_ticks']} ticks at "
            f"{res['angle_deg']:.0f}°; confidence {c:.2f} ({quality}). How long is ONE tick? "
            "(Helio does not guess the unit: a wrong guess would spoil every measurement.)")
        self.win.update_bottom()

    def _set_zscale(self):
        from helio.core.metrology import z_scale
        try:
            self.project.z_mm_per_unit = z_scale(self.z_known.value(), self.z_diff)
            self.project.z_from_scale = False
        except ValueError as e:
            self.z_err.setText(str(e))
            return
        self.project.changed.emit()
        self.use("profile")

    def clear_zscale(self):
        self.project.z_mm_per_unit = None
        self.project.z_from_scale = False
        self.z_diff = 0.0
        self.project.changed.emit()
        self.win.flash("Height scale cleared: heights are in pixels of height again.")

    def _save_profile(self):
        from PySide6.QtWidgets import QFileDialog
        if not self.profile:
            return
        path, _ = QFileDialog.getSaveFileName(self.win, "Save the profile", "profile.csv",
                                              "CSV (*.csv)")
        if path:
            self.save_profile(path)

    def save_profile(self, path: str) -> str:
        from helio.core.metrology import profile_csv
        p = self.profile
        return profile_csv(path, p["x_axis"], p["z_plot"], p["label"], p["x_label"], p["z_unit"],
                           (p["Ra"], p["Rq"], p["Rz"], p["Rp"], p["Rv"]), iso=p.get("iso"))

    def _save_region_csv(self):
        import csv
        from PySide6.QtWidgets import QFileDialog
        if not self.region_rows:
            return
        if self.tool == "compare":
            if not self.compare_stats:
                return
            path, _ = QFileDialog.getSaveFileName(self.win, "Save the comparison",
                                                  "surface_comparison.csv", "CSV (*.csv)")
            if path:
                files = self.save_comparison(path)
                self.win.flash(f"Saved {os.path.basename(files[0])} and its chart.")
            return
        path, _ = QFileDialog.getSaveFileName(self.win, "Save the figures",
                                              "surface_statistics.csv", "CSV (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["parameter", "value"])
                w.writerows(self.region_rows)

    def _set_scale(self):
        from helio.core.metrology import scale_from_line
        try:
            mm_pp, ppi = scale_from_line(self.drawn_px, self.length.value(), self.unit.currentText())
        except ValueError as e:
            self.scale_err.setText(str(e))
            return
        self.ppi = ppi
        self.project.set_scale_photo(mm_pp)          # drawn in the photograph's pixels, on any of the pictures
        self.project.changed.emit()
        self.use("distance")

    # ── the picture ───────────────────────────────────────────────────
    def _show_image(self):
        import numpy as np
        p = self.project
        mode = self.show_seg.current()
        D = self.win.render_data
        if mode == "photo" or D is None:
            key = ("photo", p.images[p.current] if p.images else "")
            if self._shown_key != key:
                self._shown_key = key
                self.viewport.item.setPos(0, 0)
                self.viewport.set_image(key[1])
            return
        key = (mode, id(p.result))
        if self._shown_key == key:
            return
        self._shown_key = key
        if mode == "depth":
            arr = (np.clip(D.depth_proxy_p, 0, 1) * 255).astype(np.uint8)
            arr = np.repeat(arr[..., None], 3, axis=2)
        else:
            from helio.application.viewing import Display, screen_image
            from helio.domain.view_controls import from_angles, light_settings
            st = dict(self.win.engine_settings.get("Standard PBR", {}),
                      **light_settings(*from_angles(315, 25)), gain=0.6)
            arr = screen_image("Standard PBR", D, st, Display(layer="Shading"))
        r = p.result
        ws = self._ws()
        # the relief at the photograph's size and place, so a point means the same on every picture
        self.viewport.show_array(arr, full_size=(D.w / ws, D.h / ws))
        if r is not None and r.crop:
            self.viewport.item.setPos(r.crop[0] / ws, r.crop[1] / ws)

    # ── state → screen ────────────────────────────────────────────────
    def refresh(self):
        self.tools.clear()
        for name, aid in self.TOOLS:
            it = QListWidgetItem(("›  " if self.TOOL_OF.get(aid) == self.tool else "    ") + name)
            why = self.reg.reason(aid)
            if why:
                from helio_qt.workspaces import _disable
                _disable(it, why)
            self.tools.addItem(it)
        fit_list(self.tools)
        p = self.project
        self.show_seg.setVisible(p.solved)
        if not p.solved and self.show_seg.current() != "photo":
            self.show_seg.set_current("photo")
        self._show_image()
        mm_pp = p.photo_scale()
        lines = ([f"{mm_pp:.5f} mm per pixel  ·  {25.4 / mm_pp:.1f} PPI",
                  f"{mm_pp * 1000:.2f} µm per pixel"] if mm_pp
                 else ["No scale yet: lengths are in pixels."])
        if mm_pp and self._ws() < 1.0:
            lines.append(f"(the photographs' pixels. Solved at {self._ws() * 100:.0f}% size: each pixel of the "
                         f"result is {p.scale_mm_per_px:.5f} mm.)")
        if p.z_mm_per_unit:
            line = f"Height: {p.z_mm_per_unit * 1000:.3f} µm per pixel of height"
            if p.scale_mm_per_px:
                line += (f" ({p.z_mm_per_unit / p.scale_mm_per_px:.2f}× what the normals alone give: "
                         "1× when the photographs measure height exactly)")
            lines.append(line)
        else:
            lines.append("Height: no height scale yet; heights in pixels of height, from the normals.")
        self.scale.setText("\n".join(lines))
        self.btn_clear_z.setVisible(bool(p.z_mm_per_unit))
        t = self.tool
        self.scale_box.setVisible(t in ("scale", "ruler"))
        if t == "scale" and not self.line_len.text().startswith("Tick spacing"):
            self.line_len.setText(f"Line: {self.drawn_px:.1f} px. How long is it really?"
                                  if self.drawn_px else "Drag along a ruler or scale bar.")
        if t == "ruler":
            self.line_len.setText("Drag a long, thin box along the ruler's ticks: 20 mm or more of them.")
        self.btn_set.setEnabled(self.drawn_px > 0 and t == "scale")
        self.z_box.setVisible(t == "zscale")
        self.z_line.setText(f"Height change along the line: {self.z_diff:.4g} px of height"
                            + (f" (about {self.z_diff * p.scale_mm_per_px:.4g} mm from the normals alone)"
                               if p.scale_mm_per_px else "")
                            + ". How high is the feature really?" if self.z_diff else
                            "Drag from the flat base to the top of something of known height "
                            "(a coin, a shim, a step).")
        self.btn_z.setEnabled(self.z_diff > 0)
        self.btn_z_lateral.setEnabled(bool(p.scale_mm_per_px))
        self.dist_box.setVisible(t == "distance" or bool(self.measured))
        if self.measured:
            ls = [f"{mm:.2f} mm  ({px:.0f} px)" if mm is not None else f"{px:.0f} px (no scale)"
                  for px, mm in self.measured[-10:]]
            self.results.setText("\n".join(reversed(ls)))
        self.profile_box.setVisible(t == "profile")
        if self.profile:
            pr = self.profile
            self.chart.set_profile(pr["x_axis"], pr["z_plot"], pr["x_label"], pr["z_unit"])
            u = pr["z_unit"] if pr["z_unit"] != "rel" else "relative units"
            sig = pr.get("sigma_px")
            sig_line = ""
            if sig is not None:
                zm = p.z_mm_per_unit
                sig_line = ("\nEnd-to-end height ±" + (f"{sig * zm * 1000:.1f} µm" if zm else
                                                       f"{sig:.3g} px of height")
                            + " (1σ, from the normals' precision; calibration error not included)")
            iso = pr.get("iso")
            iso_line = ""
            if iso:
                xu = iso.get("x_unit", "px")
                iso_line = (f"\nFiltered at λc {iso['cutoff']:.3g} {xu}"
                            + ("" if iso["standard"] else " (a fifth of the line, not a standard cut-off)")
                            + f": roughness Ra {iso['Ra']:.4g}  ·  Rq {iso['Rq']:.4g}  ·  "
                              f"Rz {iso['Rz']:.4g}  ·  Rt {iso['Rt']:.4g};  "
                              f"waviness Wa {iso['Wa']:.4g}  ·  Wt {iso['Wt']:.4g}   (ISO 16610-21)")
            self.rough.setText(f"Ra {pr['Ra']:.4g}  ·  Rq {pr['Rq']:.4g}  ·  Rz {pr['Rz']:.4g}\n"
                               f"Rp {pr['Rp']:.4g}  ·  Rv {pr['Rv']:.4g}   ({u}, ISO 4287, about the "
                               f"whole profile's mean line)" + iso_line + sig_line)
        else:
            self.chart.set_profile(None, None, "", "")
            self.rough.setText("")
        self.btn_csv.setEnabled(bool(self.profile))
        self.region_box.setVisible(t in ("volume", "stats", "compare"))
        self.region_title.setText({"volume": "Area and volume",
                                   "stats": "Surface statistics (ISO 25178)",
                                   "compare": "Two regions compared (ISO 25178)"}.get(t, ""))
        self.btn_whole.setVisible(t != "compare")
        self.region_text.setText("\n".join(f"{n:<34} {v}" for n, v in self.region_rows)
                                 if self.region_rows else
                                 "Drag a rectangle over the area to measure, or use the whole image.")
        if t != "stats" or not self.region_rows:
            self.bearing.setVisible(False)
        self.btn_region_csv.setEnabled(bool(self.region_rows))
        self.acc_box.setVisible(t == "accuracy")
        self.acc_text.setText("\n".join(f"{n:<30} {v}" for n, v in self.acc_rows) if self.acc_rows else
                              "Drag box 1 over the reference surface.")
        self.tool_note.setText({
            "scale": "Drag along something of known length. Esc cancels.",
            "ruler": "Drag a long, thin box along a ruler's ticks: 20 mm or more of them.",
            "distance": "Drag between two points. Esc cancels.",
            "zscale": "Drag from the base to the top of a feature. Esc cancels.",
            "profile": "Drag across the surface. Esc cancels.",
            "volume": "Drag a rectangle a little larger than the feature: its border is the base.",
            "stats": "Drag a rectangle over the surface to characterise.",
            "compare": "Drag region A over a feature, then region B over the background.",
            "accuracy": "Box 1 on the reference surface, then box 2 on the step's top."
            }.get(t, ""))

    def next_step(self):
        if self.tool == "scale" and self.drawn_px:
            return "Set scale", "measure.apply_scale"
        if self.tool == "zscale" and self.z_diff:
            return "Set height scale", "measure.apply_zscale"
        if self.tool == "profile":
            return "Draw another profile", "measure.profile"
        if not self.project.scale_mm_per_px:
            return "Set a scale…", "measure.scale"
        return "Measure a distance", "measure.distance"

    def status(self):
        p = self.project
        if not p.images:
            return "idle", "No images yet", "Drop a folder of captures anywhere in this window"
        if self.tool == "scale":
            return "info", "Setting the scale", self.line_len.text()
        if self.tool == "zscale":
            return "info", "Setting the height scale", self.z_line.text()
        if self.tool == "profile" and self.profile:
            pr = self.profile
            return "ok", f"Rz {pr['Rz']:.3g} {pr['z_unit']}", pr["label"]
        if self.tool == "accuracy" and self.acc_rows:
            err = dict(self.acc_rows).get("Error")
            return ("ok", f"Error {err}" if err else "Flatness checked",
                    " · ".join(f"{n.strip()} {v}" for n, v in self.acc_rows[:2]))
        if self.tool in ("volume", "stats", "compare") and self.region_rows:
            return "ok", self.region_title.text(), " · ".join(v for _n, v in self.region_rows[1:3])
        if self.measured:
            px, mm = self.measured[-1]
            return "ok", (f"{mm:.2f} mm" if mm is not None else f"{px:.0f} px"), \
                   f"last distance · {len(self.measured)} measured"
        if not p.scale_mm_per_px:
            return "warn", "No scale set", "Measurements are in pixels until a scale is set"
        return "ok", "Scale set", f"{p.scale_mm_per_px * 1000:.2f} µm per pixel"
