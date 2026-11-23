"""Measure in the new interface: set a scale, measure distances (MET-001).

Offscreen.  Draws lines with real mouse events on the image, sets the
scale from a typed length, measures a distance in millimetres, checks the
numbers are the shared helio.core.metrology arithmetic (the classic panel's),
and that the scale reaches what uses it: a life-size STL and the session.

    python tests/test_qt_measure.py
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.make_golden import test_images  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    import numpy as np
    print("=" * 60)
    print("Measure: scale and distance")
    print("=" * 60)
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    from helio.core.metrology import distance, line_length_px, scale_from_line

    print("\nNothing to measure yet")
    check("the tools say why", w.registry.reason("measure.scale") != "")

    w.open_paths([test_images(PASS, FAIL)])
    m = w.workspaces["Measure"]
    w.registry.trigger("measure.scale")
    app.processEvents()
    check("Set Scale opens Measure with the line tool", w.current == "Measure"
          and m.tool == "scale" and m.viewport.tool == "line")
    check("the next step asks for the line first", w.primary.text() == "Set a scale…")

    print("\nDraw the scale line with the mouse")
    vp = m.viewport
    got = []
    vp.lineDrawn.connect(lambda *a: got.append(a))
    a_pt = vp.mapFromScene(100.0, 600.0)
    b_pt = vp.mapFromScene(400.0, 640.0)
    QTest.mousePress(vp.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a_pt)
    QTest.mouseMove(vp.viewport(), b_pt)
    QTest.mouseRelease(vp.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b_pt)
    app.processEvents()
    check("a drag gives a line in image pixels", got and abs(got[-1][0] - 100) < 2
          and abs(got[-1][3] - 640) < 2, got)
    x1, y1, x2, y2 = got[-1]
    check(f"its length is the classic tool's ({m.drawn_px:.1f} px)",
          m.drawn_px == line_length_px(x1, y1, x2, y2))
    check("the next step is to set it", w.primary.text() == "Set scale")
    m.length.setValue(2.5)
    m.unit.setCurrentText("cm")
    w.registry.trigger("measure.apply_scale")
    mm_pp, ppi = scale_from_line(m.drawn_px, 2.5, "cm")
    check(f"the scale is the shared arithmetic ({mm_pp:.5f} mm/px, {ppi:.1f} PPI)",
          w.project.scale_mm_per_px == mm_pp and abs(m.ppi - ppi) < 1e-9)
    check("Measure then moves on to distances", m.tool == "distance")

    print("\nMeasure a distance")
    m._line_drawn(0, 0, 300, 400)
    px, mm = distance(0, 0, 300, 400, mm_pp)
    check(f"500 px is {mm:.2f} mm, shown and in the status", px == 500 and m.measured[-1] == (px, mm)
          and w.status_text.text() == f"{mm:.2f} mm")
    QTest.keyClick(vp, Qt.Key.Key_Escape)
    check("Esc clears the line", m.viewport._line is None or not m.viewport._line.isVisible())

    print("\nThe scale reaches what uses it")
    out = os.path.join(tempfile.mkdtemp(), "s.helio")
    w.write_session(out)
    from helio.io import session as S
    check("the session stores it", S.read(out).mm_per_px == mm_pp)
    w.registry.trigger("process.solver:Uncalibrated PCA")
    w.registry.trigger("process.run")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.01)
    from helio_qt import publish
    r = w.project.result
    src = publish.Source(result=r, lights=None, light_source="", images=w.project.images,
                         name="scale", mm_per_px=w.project.scale_mm_per_px)
    stl = publish.export_one("stl", src, tempfile.mkdtemp())[0]
    import trimesh
    ext = trimesh.load(stl).extents
    from helio.application.exports import shared_defaults, trim_box
    x1, y1, x2, y2 = trim_box(r.w, r.h, shared_defaults()["edge_trim"])
    want = np.array([(x2 - x1) * mm_pp, (y2 - y1) * mm_pp])
    err = np.abs(np.sort(ext[:2]) - np.sort(want)) / np.sort(want)
    check(f"the STL is life size: {ext[0]:.1f} × {ext[1]:.1f} mm for "
          f"{want[0]:.1f} × {want[1]:.1f} mm of object", (err < 0.02).all(), err)

    print("\nHeight scale and profiles (MET-002)")
    from helio.core.metrology import height_reference, profile_line, profile_stats, z_scale
    hs = [m.tools.item(i) for i in range(m.tools.count()) if "Height scale" in m.tools.item(i).text()]
    m.tools.itemClicked.emit(hs[0])                              # the list, as a click would
    check("clicking Height scale in the list picks it", m.tool == "zscale")
    m._line_drawn(300, 500, 600, 520)
    from helio.application.heights import height_px
    H = height_px(r)                                              # true heights: what Measure reads
    check("Measure reads true heights, not the display depth", H is not None and H.shape == r.depth_vis_p.shape)
    z, _e, num = profile_line(H, 300, 500, 600, 520)
    check(f"the height change along the line, in px of height ({m.z_diff:.4g})",
          m.z_diff == height_reference(z))
    m.z_known.setValue(0.35)
    w.registry.trigger("measure.apply_zscale")
    check("the height scale is the shared arithmetic",
          w.project.z_mm_per_unit == z_scale(0.35, height_reference(z)) and m.tool == "profile")
    m._line_drawn(150, 700, 800, 700)
    z2, _e2, num2 = profile_line(H, 150, 700, 800, 700)
    want = profile_stats(z2, num2, w.project.scale_mm_per_px, w.project.z_mm_per_unit)
    got = m.profile
    check(f"the profile and its roughness are the shared figures (Rz {got['Rz']:.3g} {got['z_unit']})",
          all(np.array_equal(got[k], want[k]) for k in ("x_axis", "z_plot"))
          and all(got[k] == want[k] for k in ("Ra", "Rq", "Rz", "Rp", "Rv", "label", "z_unit")))
    check("now in real units: millimetres along, µm or mm up", got["x_label"] == "Distance (mm)"
          and got["z_unit"] in ("um", "mm"))
    import csv as _csv
    csv_path = m.save_profile(os.path.join(tempfile.mkdtemp(), "p.csv"))
    rows = list(_csv.reader(open(csv_path)))
    check("Save profile as CSV: title, roughness, the filtered roughness and waviness, then one row per sample",
          rows[0][0] == got["label"] and rows[1][0].startswith("Ra=") and "ISO 16610-21" in rows[2][0]
          and rows[3][0] == "Roughness" and rows[4][0] == "Waviness" and len(rows[5]) == 4
          and len(rows) == 6 + num2 and len(rows[-1]) == 4)
    out2 = os.path.join(tempfile.mkdtemp(), "z.helio")
    w.write_session(out2)
    check("the session stores the height scale, and what it is measured on",
          S.read(out2).z_mm_per_unit == w.project.z_mm_per_unit
          and S.read(out2).height_basis == "true-height-px")

    print("\nArea, volume and surface statistics (MET-013…015)")
    from helio.core.metrology import areal, true_surface_area, volume
    w.registry.trigger("measure.volume")
    check("Area and volume asks for a rectangle", m.viewport.tool == "rect")
    m._rect_drawn(300, 400, 600, 700)
    rows = dict(m.region_rows)
    a = true_surface_area(r.normals_full[400:700, 300:600], w.project.scale_mm_per_px)
    v = volume(H[400:700, 300:600], w.project.scale_mm_per_px, w.project.z_mm_per_unit)
    check("the true area is the shared figure, in mm²",
          rows.get("True surface area") == f"{a['true_mm2']:,.2f} mm²")
    check("the volume above the region's base plane, in mm³",
          rows.get("Material above the base plane") == f"{v['above_mm3']:.4f} mm³")
    w.registry.trigger("measure.stats")
    m._whole_region()
    rows = dict(m.region_rows)
    A = areal(H, r.normals_full, w.project.z_mm_per_unit)
    check("ISO 25178 over the whole image: heights in mm, Str, Sdq, Sdr",
          rows.get("Sa  arithmetic mean height") == f"{A['Sa']:.5g} mm"
          and rows.get("Str texture aspect ratio") == f"{A['Str']:.4f}"
          and "Sdr developed area ratio" in rows and m.bearing.isVisibleTo(w))
    m.show_seg.chosen.emit("relief")
    app.processEvents()
    check("the solved relief can be shown to measure on", m.viewport.path == ""
          and not m.viewport.item.pixmap().isNull())
    m.show_seg.set_current("photo")
    m.show_seg.chosen.emit("photo")
    w.registry.trigger("measure.clear_zscale")
    check("the height scale can be cleared", w.project.z_mm_per_unit is None)
    w.registry.trigger("measure.ruler")
    check("Scale from a ruler asks for a rectangle over the ticks", m.viewport.tool == "rect"
          and m.tool == "ruler")

    print("\nCompare two regions (the classic Surface Comparison)")
    from helio.application.export_reports import region_stats
    w.registry.trigger("measure.compare")
    check("it asks for region A", m.viewport.tool == "rect" and m.tool == "compare"
          and "region A" in m.tool_note.text())
    m._rect_drawn(300, 400, 600, 700)
    check("…then region B", "region B" in dict(m.region_rows).get("Next", ""))
    m._rect_drawn(700, 400, 1000, 700)
    sa = region_stats(H[400:700, 300:600], "ROI A", z_mm_per_unit=w.project.z_mm_per_unit,
                      mm_per_px=w.project.scale_mm_per_px)
    rows = dict(m.region_rows)
    check("the table gives A, B and A − B from the shared figures",
          rows.get("Sa", "").split()[0] == f"{sa['Sa']:.5g}" and len(rows["Sa"].split()) == 3)
    out3 = os.path.join(tempfile.mkdtemp(), "cmp.csv")
    files = m.save_comparison(out3)
    crow = list(_csv.reader(open(out3)))
    check("the comparison saves the classic CSV and its chart",
          crow[0][:4] == ["Parameter", "ROI A", "ROI B", "Diff (A-B)"] and os.path.exists(files[1]))

    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
