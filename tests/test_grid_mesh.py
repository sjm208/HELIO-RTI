"""The 3D relief panel's mesh grid: Rust against NumPy, and the finer-than-1:1 density.

* helio_core_rs.grid_mesh gives the NumPy builder's arrays bit for bit, for whole and
  fractional steps, odd and even sizes, and is faster;
* a step below 1 puts more vertices than pixels, at the right places, and the panel falls back
  to 1 : 1 (and says so) when that would pass the point limit;
* the readout and the panel's names for the ratios.

    python tests/test_grid_mesh.py
"""
import os
import sys
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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
    from helio_qt import relief3d as R
    from helio.core._native import rs
    if rs is None:
        print("  skip helio_core_rs is not built (this test is about its grid builder)")
        return 0
    print("Rust and NumPy")
    check("the Rust grid builder is there", rs is not None and hasattr(rs, "grid_mesh"))
    cases = [(2, 2, 1), (7, 5, 1), (64, 48, 2), (101, 77, 3), (640, 480, 1), (333, 222, 4),
             (50, 40, 0.5), (65, 33, 0.25), (1000, 700, 0.5), (20, 20, 1.5), (3, 3, 2)]
    same = True
    for w, h, st in cases:
        gw, gh = R.grid_size(w, h, st)
        v1, t1, g = R.build_grid(w, h, st)
        v2, t2 = R._build_grid_numpy(w, h, st, gw, gh)
        ok = (g == (gw, gh) and v1.dtype == np.float32 and t1.dtype == np.uint32 and v1.shape == v2.shape
              and np.array_equal(v1, v2) and np.array_equal(t1, t2))
        if not ok:
            same = False
            print("     differs:", (w, h, st))
    check(f"identical to NumPy for {len(cases)} sizes and steps (whole, fractional, odd, even)", same)
    check("whole steps give the grid sizes they always did",
          all(R.grid_size(w, h, s) == (max(2, (w - 1) // s + 1), max(2, (h - 1) // s + 1))
              for w, h, s in [(2500, 1700, 1), (2500, 1700, 2), (1999, 1001, 3), (640, 480, 8), (9, 9, 10)]))
    w, h = 2500, 1700
    t = time.perf_counter(); R.build_grid(w, h, 1); tr = time.perf_counter() - t
    gw, gh = R.grid_size(w, h, 1)
    t = time.perf_counter(); R._build_grid_numpy(w, h, 1, gw, gh); tn = time.perf_counter() - t
    print(f"       {gw * gh / 1e6:.1f} M points: Rust {tr * 1000:.0f} ms, NumPy {tn * 1000:.0f} ms (x{tn / tr:.0f})")
    check("and quicker", tr < tn)
    bad = 0
    for args in ((1, 5, 8, 8, 1.0), (5, 5, 0, 8, 1.0), (5, 5, 8, 8, 0.0), (5, 5, 8, 8, -1.0)):
        try:
            rs.grid_mesh(*args)
            bad += 1
        except ValueError:
            pass
    check("an empty or backwards grid is refused", bad == 0)

    print("Finer than one vertex a pixel")
    v, tr_, (gw, gh) = R.build_grid(10, 10, 0.5)
    check("half steps: 19 vertices across 10 pixels, and the first sits half a pixel in",
          (gw, gh) == (19, 19) and abs(v[0, 0] - 0.05) < 1e-6 and abs(v[1, 0] - v[0, 0] - 0.05) < 1e-6)
    check("every index points at a vertex", int(tr_.max()) < gw * gh and tr_.size == 6 * 18 * 18)

    print("In the panel")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    view = R.ReliefView()

    def surface(n_w, n_h):
        n = np.dstack([np.zeros((n_h, n_w)), np.zeros((n_h, n_w)), np.ones((n_h, n_w))]).astype(np.float32)
        view.set_surface(np.zeros((n_h, n_w), np.float32), n, np.full((n_h, n_w, 3), 128, np.uint8))
    surface(800, 600)
    view.set_mesh_step(0.5)
    check("a small surface takes 2 : 1", view.mesh_step == 0.5 and view.mesh_used() == 0.5, view.mesh_used())
    view._readout()
    check("and says so", "mesh 2 : 1" in view._readout_text, view._readout_text)
    surface(2500, 1875)
    check("a 2500 px preview cannot hold it: 1 : 1, with the reason",
          view.mesh_used() == 1 and (view._readout(), "mesh 1 : 1 (2 : 1 would pass" in view._readout_text)[1],
          view._readout_text)
    surface(3600, 2700)
    view.set_mesh_step(0.5)
    check("too big even for 1 : 1: the next step that fits, as before", view.mesh_used() == 2, view.mesh_used())
    view.set_mesh_step(1)
    check("whole steps are untouched: 1, 2, 4, 8 and a stray 3.7 -> 3",
          [R.ReliefView._clean_step(s) for s in (1, 2, 4, 8, 3.7)] == [1, 2, 4, 8, 3]
          and R.ReliefView._clean_step(0.1) == 0.25 and R.ReliefView._clean_step(0.5) == 0.5)
    check("the ratios are named as the panel names them",
          (R.ReliefView.ratio(0.5), R.ReliefView.ratio(1), R.ReliefView.ratio(4)) == ("2 : 1", "1 : 1", "1 : 4"))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
