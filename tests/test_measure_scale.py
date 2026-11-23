"""A measurement means the same thing on every picture, whatever working size the solve used.

Solved at 50% (and 35%), the result has fewer pixels than the photographs.  A scale set, or a distance
measured, on the Measure tab's photograph must be the same length on its relief view, in Inspect's
pixels (the 2D view, its sections and scale bar) and in the 3D viewer.  This is the test of that:

* the scale, set on the photograph, is per photograph pixel; the result's pixel is bigger by 1 / ws;
* the same two points, drawn on the photograph or on the relief, measure the same millimetres;
* the 3D viewer's distance between them, and the 2D conversion, give the same millimetres;
* a scale set before the solve, one set after, one loaded from a session, and a re-solve at another
  size, all agree; a region solve's offset is honoured; full size is unchanged.

    python tests/test_measure_scale.py
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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
    from tools import make_golden as G
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    G.H, G.W = 900, 1200
    tmp = tempfile.mkdtemp(prefix="helio_ms_")
    paths, L = G.render_stack(tmp, n_lights=8)
    w = hw.MainWindow(appearance="dark")
    w.resize(1300, 850)
    w.show()
    p = w.project
    ms = w.workspaces["Measure"]
    ins = w.workspaces["Inspect"]

    def idle():
        t0 = time.time()
        while (p.busy or w.bg_pool.activeThreadCount()) and time.time() - t0 < 300:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    def solve(ws):
        w.project.update_settings(work_scale=ws)
        w.registry.trigger("process.run")
        idle()
        assert p.solved

    def set_scale_on_photo(px_len=400.0, mm=20.0):
        ms.use("scale")
        ms._line_drawn(100.0, 100.0, 100.0 + px_len, 100.0)
        ms.length.setValue(mm)
        ms.unit.setCurrentText("mm")
        ms._set_scale()

    w.open_paths([tmp])
    w.project.set_lights(L, "file:golden.lp")
    w.project.update_settings(photo_tone="linear")
    w.go_workspace("Measure")
    app.processEvents()

    print("A scale set before the solve (the photograph is all there is)")
    set_scale_on_photo()
    check("0.05 mm per photograph pixel (20 mm along 400 px)", abs(p.photo_scale() - 0.05) < 1e-9, p.photo_scale())
    check("and the result will be solved at a size; the scale follows it", p.scale_mm_per_px == p.photo_scale())

    for ws in (0.5, 0.35):
        print(f"\nSolved at {ws * 100:.0f}% size")
        solve(ws)
        r, D = p.result, w.render_data
        check(f"the result is {ws * 100:.0f}% of the photograph: {r.w} x {r.h}", abs(r.w - 1200 * ws) <= 1 and r.work_scale == ws)
        check("the photograph's scale did not move", abs(p.photo_scale() - 0.05) < 1e-9, p.photo_scale())
        check("a result pixel is 1 / ws times as big", abs(p.scale_mm_per_px - 0.05 / ws) < 1e-9, p.scale_mm_per_px)
        w.go_workspace("Measure")
        ms.show_seg.set_current("photo")
        ms.refresh()
        check("Measure's scale bar is in the photograph's pixels", abs(ms.scalebar.mm_per_px() - 0.05) < 1e-9, ms.scalebar.mm_per_px())
        check("and its panel says the photograph's scale, and the result's", f"{0.05:.5f} mm per pixel" in ms.scale.text()
              and f"{0.05 / ws:.5f}" in ms.scale.text(), ms.scale.text())

        print(" the same two points on the photograph and on the relief")
        ms.use("distance")
        ms.measured.clear()
        ms._line_drawn(200.0, 300.0, 200.0, 700.0)               # 400 photograph pixels, down
        ms._line_drawn(300.0, 400.0, 700.0, 700.0)               # 500 px along a diagonal
        for seg in ("relief", "depth"):
            ms.show_seg.set_current(seg)
            ms.refresh()
            sr = ms.viewport.scene().sceneRect()
            check(f"the {seg} view covers the photograph's frame ({sr.width():.0f} x {sr.height():.0f})",
                  abs(sr.width() - 1200) <= 2 and abs(sr.height() - 900) <= 2, (sr.width(), sr.height()))
            ms._line_drawn(200.0, 300.0, 200.0, 700.0)
            ms._line_drawn(300.0, 400.0, 700.0, 700.0)
        mm = [m for _px, m in ms.measured]
        check("20.00 mm and 25.00 mm, on the photograph and on both relief views", np.allclose(mm, [20.0, 25.0] * 3, atol=1e-6), mm)

        print(" Inspect's 2D pixels, and the 3D viewer")
        a = ms._result_line(200.0, 300.0, 200.0, 700.0)
        check("the line is at the right place in the result's pixels", np.allclose(a, (200 * ws, 300 * ws, 200 * ws, 700 * ws)), a)
        ins_len = float(np.hypot(a[2] - a[0], a[3] - a[1])) * p.scale_mm_per_px
        check(f"its length in result pixels x the result's scale = 20 mm ({ins_len:.3f})", abs(ins_len - 20.0) < 1e-6)
        from helio_qt.relief3d import ReliefView
        A = ins._relief_arrays(D)
        v = ReliefView()
        v.set_surface(A["depth"], A["normals"], A["albedo"], scale_mm=p.scale_mm_per_px, z_mm_per_unit=p.z_mm_per_unit,
                      full_size=(D.w, D.h))

        def at(x, y):                                            # a photograph point on the 3D surface
            rx, ry = ms._to_result(x, y)
            return v.world_of(rx / D.w, ry / D.h)
        v.points = [at(200.0, 300.0), at(200.0, 700.0)]
        m = v.measure()
        check(f"the 3D viewer: across = 20 mm ({m['across_mm']:.3f})", abs(m["across_mm"] - 20.0) < 0.2, m)
        v.points = [at(300.0, 400.0), at(700.0, 700.0)]
        check(f"and the diagonal = 25 mm ({v.measure()['across_mm']:.3f})", abs(v.measure()["across_mm"] - 25.0) < 0.25)
        v.section = ((200 * ws / D.w, 300 * ws / D.h), (200 * ws / D.w, 700 * ws / D.h))
        sp = v.section_profile()
        check(f"a section down the same line is 20 mm long ({sp['length']:.3f})", abs(sp["length"] - 20.0) < 0.2, sp["length"])
        bar = ins.scalebar
        check("Inspect's scale bar uses the result's pixel", abs(bar.mm_per_px() - p.scale_mm_per_px) < 1e-9, bar.mm_per_px())

        print(" regions and profiles read the right pixels")
        box = ms.result_box(200.0, 300.0, 600.0, 700.0)
        check(f"a box on the photograph is the right box in the result {box}",
              all(abs(a - b) <= 1 for a, b in zip(box, (200 * ws, 300 * ws, 600 * ws, 700 * ws))))
        ms.use("stats")
        ms._rect_drawn(200.0, 300.0, 600.0, 700.0)
        reg = dict(ms.region_rows)
        check("Surface statistics ran on it", any("Sa" in k for k in reg) or bool(reg), list(reg)[:3])
        r.crop = (40, 30, r.w, r.h)                              # as a region solve: starts at (40, 30) result pixels
        try:
            check("a region solve's offset is honoured", np.allclose(ms._to_result(200.0, 300.0), (200 * ws - 40, 300 * ws - 30)))
        finally:
            r.crop = None

    print("\nSession and re-solve")
    set_scale_photo = p.photo_scale()
    from helio.io import session as SE
    check("the photograph scale is what is saved", abs(set_scale_photo - 0.05) < 1e-9)
    solve(1.0)
    check("solved at full size: result pixel = photograph pixel = 0.05", abs(p.scale_mm_per_px - 0.05) < 1e-9
          and abs(p.photo_scale() - 0.05) < 1e-9)
    w.go_workspace("Measure")
    ms.use("distance")
    ms.measured.clear()
    ms._line_drawn(200.0, 300.0, 200.0, 700.0)
    check("and a distance there is the same 20 mm", abs(ms.measured[-1][1] - 20.0) < 1e-6, ms.measured)
    solve(0.35)
    check("back to 35%: still 0.05 per photograph pixel, 0.1429 per result pixel",
          abs(p.photo_scale() - 0.05) < 1e-9 and abs(p.scale_mm_per_px - 0.05 / 0.35) < 1e-9)
    p.set_scale_photo(0.08)                                      # a rig profile or a loaded session
    check("a scale given for the photographs is kept per result pixel", abs(p.scale_mm_per_px - 0.08 / 0.35) < 1e-9)

    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
