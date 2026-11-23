"""Heights measured against surfaces whose heights are known.

Measure, the probe, the ISO figures and the height exports read true heights
(helio.application.heights): the integrated depth in pixels of height.  The display
depth they used to read is the depth less a broad blur, clipped and stretched, so a
feature's height there depended on its width.  Checked here:

* true heights: a narrow bump, a broad one of the same height and a tall peak come
  out at their heights (the display depth gives the broad one a fraction of its
  height and clips the peak: shown, so the test says why it matters);
* the volume of a bump is the analytic one;
* ISO 25178 and ISO 4287 remove the form: a tilted flat region and a tilted
  straight profile are not rough;
* a height scale saved on the display depth is dropped when the session is opened,
  with a note; one saved on true heights comes back.

    python tests/test_measure_heights.py
"""
import json
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HELIO_QT_SETTINGS", os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"  {detail}"))


class Result:
    """What a solve leaves: the integrated depth, the normals, the display depth."""


def main():
    from helio.application.heights import height_px
    from helio.application.pipeline import integrate_depth
    from helio.core.frames import normals_from_slopes
    from helio.core.processing import depth_visualizations

    h, w = 1300, 2000
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    def gauss(cx, cy, s, A):
        g = A * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
        return g, g * -(xx - cx) / s ** 2, g * -(yy - cy) / s ** 2

    parts = [gauss(500, 650, 25, 10.0), gauss(1300, 650, 220, 10.0), gauss(1800, 250, 10, 30.0)]
    z = sum(p[0] for p in parts)
    n = normals_from_slopes(sum(p[1] for p in parts), sum(p[2] for p in parts), 1.0).astype(np.float32)
    r = Result()
    r.normals_full = n
    r.depth_poisson = np.asarray(integrate_depth(n, "Least-Squares (Free Boundary)"), np.float32)
    r.depth_vis_p = depth_visualizations(r.depth_poisson, n, clip_pct=0.5)["depth_vis_p"]
    H = height_px(r)

    def rise(m, cx, cy):
        return float(m[cy, cx] - np.median(m[50:150, 50:150]))       # above a flat patch

    print("True heights: bumps 10 px tall, 50 and 440 px across, and a peak 30 px tall")
    hn, hb, hp = rise(H, 500, 650), rise(H, 1300, 650), rise(H, 1800, 250)
    check(f"narrow bump 10 px (got {hn:.2f})", abs(hn - 10) < 0.2, hn)
    check(f"broad bump 10 px (got {hb:.2f})", abs(hb - 10) < 0.2, hb)
    check(f"peak 30 px (got {hp:.2f})", abs(hp - 30) < 0.6, hp)
    dv = r.depth_vis_p
    dn, db, dp = rise(dv, 500, 650), rise(dv, 1300, 650), rise(dv, 1800, 250)
    check(f"why: the display depth puts the broad bump at {100 * db / dn:.0f}% of the narrow one "
          f"and the peak at {dp / dn:.2f}x (true 100% and 3x)", db / dn < 0.8 or dp / dn < 2.5, (db / dn, dp / dn))

    print("Volume of the narrow bump: 2 pi sigma^2 A")
    from helio.core.metrology import volume
    v = volume(H[550:751, 400:601], mm_per_px=0.1, z_mm_per_unit=0.1)
    want = 2 * np.pi * 25 ** 2 * 10 * 0.1 ** 3                         # mm^3 at 0.1 mm a pixel
    check(f"{v['above_mm3']:.3f} mm³ (analytic {want:.3f})", abs(v["above_mm3"] / want - 1) < 0.03,
          (v["above_mm3"], want))

    print("Form removed: tilt is not roughness")
    from helio.core.metrology import iso25178, profile_stats
    plane = 0.3 * xx[:200, :300] - 0.2 * yy[:200, :300] + 5
    rows = {k: val for k, val, _ in iso25178(plane)[0]}
    check(f"a tilted flat region: Sa {rows['Sa']:.2e} px, Sz {rows['Sz']:.2e} px", rows["Sa"] < 1e-6 and rows["Sz"] < 1e-6)
    ps = profile_stats(np.linspace(0, 40, 400), 400)
    check(f"a tilted straight profile: Ra {ps['Ra']:.2e} px", ps["Ra"] < 1e-6 and ps["Rz"] < 1e-6)
    ripple = 0.5 * np.sin(2 * np.pi * np.arange(400) / 40.0)
    ps = profile_stats(np.linspace(0, 40, 400) + ripple, 400)
    check(f"…and on it a 0.5 px ripple: Ra {ps['Ra']:.3f} (2A/pi = {1 / np.pi:.3f})", abs(ps["Ra"] - 1 / np.pi) < 0.01)

    print("Height scales saved with a session")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    from helio.io import session as S
    win = hw.MainWindow(appearance="dark")
    d = tempfile.mkdtemp()
    old = os.path.join(d, "old.helio")
    state = S.encode(S.Session(object_id="old", mm_per_px=0.1, z_mm_per_unit=0.02))
    state.pop("height_basis", None)
    json.dump(state, open(old, "w"))
    win.load_session_file(old, asked=True)
    check("one measured on the display depth is dropped", win.project.z_mm_per_unit is None)
    check("…and the note says to set it again", "Height scale" in (win.project.notice or ""), win.project.notice)
    new = os.path.join(d, "new.helio")
    S.write(new, S.Session(object_id="new", mm_per_px=0.1, z_mm_per_unit=0.02, height_basis="true-height-px"))
    win.load_session_file(new, asked=True)
    check("one measured on true heights comes back", win.project.z_mm_per_unit == 0.02, win.project.z_mm_per_unit)
    win.close()
    app.processEvents()

    print(f"\n{'=' * 60}\nResults: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS) + len(FAIL)} tests\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
