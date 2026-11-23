"""An exported model must be the size of the thing it was measured from.

Calibration existed and reached nothing: `mm_per_px` appeared zero times in
helio/core/processing.py, and export_stl_3dprint derived XY purely from the
print bed. A true 60 x 40 mm plate, calibrated at 0.1 mm/px, exported as
255.6 x 170.2 mm at the default 256 mm bed - a decorative print, and useless
for measurement.

On the sampling convention, which the numbers below depend on: vertices sit at
pixel *centres*, so a grid of n samples spans n-1 intervals. A 600 px width at
0.1 mm/px therefore models 59.9 mm, not 60.0 - the outer half-pixel at each
end is not represented. That is correct, and distances measured between
features inside the model are exact. Only the outer extent is short, by one
sample.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0
MM_PER_PX = 0.1
W, H = 600, 400
RIDGE_PX = 100


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def surface():
    d = np.zeros((H, W), np.float32)
    d[:, 250:250 + RIDGE_PX] = 1.0
    return d


def export(tmp, name, **over):
    import trimesh
    from helio.io.exporters import export_stl_3dprint
    st = {"bed_mm": 256.0, "relief_mm": 2.0, "base_mm": 3.0, "z_ex": 1.0,
          "max_res": 20000, "step": 1}
    st.update(over)
    p = os.path.join(tmp, name)
    export_stl_3dprint(surface(), None, None, st, p)
    return trimesh.load(p, file_type="stl")


def main():
    print("=" * 62)
    print("Physical scale")
    print("=" * 62)
    tmp = tempfile.mkdtemp(prefix="helio_scale_")

    print(f"\ncalibrated: {W}x{H} px at {MM_PER_PX} mm/px "
          f"= {W*MM_PER_PX:.1f} x {H*MM_PER_PX:.1f} mm, {RIDGE_PX}px ridge "
          f"= {RIDGE_PX*MM_PER_PX:.1f} mm")
    m = export(tmp, "true.stl", mm_per_px=MM_PER_PX)
    ext = m.bounds[1] - m.bounds[0]
    want_x = (W - 1) * MM_PER_PX
    want_y = (H - 1) * MM_PER_PX
    check("width is the measured width", abs(ext[0] - want_x) < 0.05,
          f"{ext[0]:.3f} mm, expected {want_x:.3f}")
    check("height is the measured height", abs(ext[1] - want_y) < 0.05,
          f"{ext[1]:.3f} mm, expected {want_y:.3f}")
    check("the aspect ratio is preserved",
          abs((ext[0] / ext[1]) - (want_x / want_y)) < 0.01,
          f"{ext[0]/ext[1]:.4f} vs {want_x/want_y:.4f}")

    v = m.vertices
    top = v[v[:, 2] > v[:, 2].max() - 0.01]
    ridge = top[:, 0].max() - top[:, 0].min()
    check("a 10 mm ridge measures 10 mm to within a sample",
          abs(ridge - (RIDGE_PX - 1) * MM_PER_PX) < 0.05,
          f"{ridge:.3f} mm")

    print("\nthe bed is ignored when true scale is asked for")
    for bed in (60.0, 256.0, 400.0):
        mm = export(tmp, f"b{bed:.0f}.stl", mm_per_px=MM_PER_PX, bed_mm=bed)
        e = mm.bounds[1] - mm.bounds[0]
        check(f"bed={bed:g} mm does not change the model",
              abs(e[0] - want_x) < 0.05, f"{e[0]:.3f} mm")

    print("\nscale survives decimation")
    for mx in (1200, 600):
        mm = export(tmp, f"r{mx}.stl", mm_per_px=MM_PER_PX, max_res=mx)
        e = mm.bounds[1] - mm.bounds[0]
        check(f"max_res={mx} keeps the measured size",
              abs(e[0] - want_x) < 0.6, f"{e[0]:.3f} mm vs {want_x:.3f}")

    print("\nwithout calibration it still fills the bed, as before")
    mm = export(tmp, "bed.stl", bed_mm=100.0)
    e = mm.bounds[1] - mm.bounds[0]
    check("the long axis fills the bed exactly", abs(e[0] - 100.0) < 0.01,
          f"{e[0]:.3f} mm on a 100 mm bed — the old code left it a pixel short")
    check("the short axis follows the aspect ratio",
          abs(e[1] - 100.0 * (H - 1) / (W - 1)) < 0.05, f"{e[1]:.3f} mm")

    print("\na zero or missing calibration falls back rather than collapsing")
    for bad in (0.0, None):
        mm = export(tmp, f"z{bad}.stl", mm_per_px=bad, bed_mm=100.0)
        e = mm.bounds[1] - mm.bounds[0]
        check(f"mm_per_px={bad!r} falls back to the bed", abs(e[0] - 100.0) < 0.01,
              f"{e[0]:.3f} mm")

    print("\n" + "=" * 62)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 62)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
