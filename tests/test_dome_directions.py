"""The light dome lights the side it points to, in every view that follows it.

On a surface with known slopes, for each view that follows the dome (and the compass view):

* with the light at north, east, south and west (and the diagonals), the picture is brighter where the
  slope faces the light than where it faces away (correlation of brightness with "faces the light");
* a lower light gives more relief contrast than a high one (in the views that shade by the angle);
* the dome's own geometry: a dot at the top is a light from the north, at the right from the east;
* the same through the real window: the dome moved, Inspect redrawn, the picture on screen checked.

    python tests/test_dome_directions.py
"""
import math
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = FAIL = 0
MIN_CORR = 0.5                      # "faces the light" against brightness; measured 0.63 to 1.00
# Views that shade by N.L: a lower light shows the relief with more contrast.
SHADE_BY_ANGLE = ("Albedo-Suppressed Raking (Clay Mode)", "PS Normal-Map Lit (Single)", "Subsurface Scattering (SSS)")
# Blinn-Phong views: N.L raised to 1/gain (6.7 at the default gain 0.15), so as the classic app drew
# them they go dark under a low light.  "Steady brightness" (pbr_expose, on by default) compensates.
POWER_CURVE = ("Standard PBR", "Robust PBR (MVS)")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    import cv2
    import numpy as np
    from tools import make_golden as G
    from helio.application import pipeline as P
    from helio.application.viewing import Display, screen_image
    from helio.data_manager import DataManager
    from helio.domain import view_controls as VC
    from helio.domain.registry import RENDER_ENGINES, default_engine_settings
    from helio.domain.settings import ProcessingSettings

    print("The dome's geometry")
    cases = [((0.0, -0.8), 0), ((0.8, 0.0), 90), ((0.0, 0.8), 180), ((-0.8, 0.0), 270),
             ((0.55, -0.55), 45), ((0.55, 0.55), 135), ((-0.55, 0.55), 225), ((-0.55, -0.55), 315)]
    bad = []
    for (dx, dy), want in cases:
        lx, ly, lz = VC.from_dome(dx, dy)
        az, el = VC.angles(lx, ly, lz)
        if min(abs(az - want), 360 - abs(az - want)) > 0.5:
            bad.append(((dx, dy), want, round(az, 1)))
    check("a dot at the top is north, right east, bottom south, left west, and the diagonals", not bad, bad)
    check("angles and from_angles are inverses",
          all(abs(VC.angles(*VC.from_angles(a, e))[0] - a) < 1e-6 and abs(VC.angles(*VC.from_angles(a, e))[1] - e) < 1e-6
              for a in (0, 33, 90, 200, 359) for e in (10, 45, 80)))
    check("the dot nearer the edge is a lower light",
          VC.angles(*VC.from_dome(0.9, 0))[1] < VC.angles(*VC.from_dome(0.3, 0))[1])

    G.H, G.W = 480, 640
    paths, L = G.render_stack(tempfile.mkdtemp(prefix="helio_dome_"), n_lights=8)
    r = P.solve(P.SolveRequest(paths=paths, lights=L, settings=ProcessingSettings(photo_tone="linear")))
    D = DataManager()
    P.store(r, D)
    truth = G.analytic_surface()[1]
    check("the solved normals agree with the known surface's", float(np.mean(np.sign(r.normals_full[..., 1]) == np.sign(truth[..., 1]))) > 0.9)
    eng = default_engine_settings()
    n = r.normals_full
    lat = np.stack([n[..., 0], -n[..., 1]], -1)                 # image x right, image y down
    sel = np.linalg.norm(lat, axis=-1) > 0.08
    ph, pw = r.ph, r.pw
    latp = cv2.resize(lat, (pw, ph))
    selp = cv2.resize(sel.astype(np.uint8), (pw, ph)) > 0

    def faces(img, az):
        u = np.array([math.sin(math.radians(az)), -math.cos(math.radians(az))])
        g = np.asarray(img).astype(np.float32)
        g = g.mean(-1) if g.ndim == 3 else g
        v = g[selp]
        return float(np.corrcoef((latp @ u)[selp], v)[0, 1]) if v.std() > 1e-6 else float("nan")

    def lit(view, az, el):
        st = dict(eng[view])
        if VC.light_kind(view) == "compass":
            st.update(hs_azi=az, hs_alt=el)
        else:
            lx, ly, lz = VC.from_angles(az, el)
            st.update(lx=lx, ly=ly, lz=lz)
        return st

    dome_views = [v for v in RENDER_ENGINES if VC.light_kind(v)]
    print(f"Each of the {len(dome_views)} views that follow the dome, light at 8 compass points, 40° up")
    table = {}
    for v in dome_views:
        sc = [faces(screen_image(v, D, lit(v, az, 40.0), Display(layer="Shading")), az) for az in range(0, 360, 45)]
        table[v] = sc
        check(f"{v}: the lit side is brighter, all round (worst {min(sc):.2f})", min(sc) >= MIN_CORR, [round(x, 2) for x in sc])

    print("Raking light gives more relief than overhead")
    for v in SHADE_BY_ANGLE:
        if v not in dome_views:
            continue
        lo = np.asarray(screen_image(v, D, lit(v, 315, 25.0), Display(layer="Shading"))).astype(np.float32).std()
        hi = np.asarray(screen_image(v, D, lit(v, 315, 75.0), Display(layer="Shading"))).astype(np.float32).std()
        check(f"{v}: contrast at 25° ({lo:.1f}) > at 75° ({hi:.1f})", lo > hi)

    flat_px = cv2.resize(r.normals_full[..., 2].astype(np.float32), (pw, ph)) > 0.995
    for v in POWER_CURVE:
        def flat_level(el, expose):
            st = dict(lit(v, 315, el), pbr_expose=expose)
            g = np.asarray(screen_image(v, D, st, Display(layer="Shading"))).astype(np.float32).mean(-1)
            return float(g[flat_px].mean()), float(g.std())
        old = [flat_level(el, 0) for el in (15, 30, 45, 60, 75)]
        new = [flat_level(el, 1) for el in (15, 30, 45, 60, 75)]
        check(f"{v}: as the classic app, a power curve: flat ground {old[0][0]:.0f} -> {old[-1][0]:.0f} of 255 from 15° to 75°",
              all(a[0] < b[0] for a, b in zip(old, old[1:])) and old[0][0] < 5)
        check(f"{v}: steady brightness keeps flat ground between 100 and 240 at every height "
              f"({min(x[0] for x in new):.0f}-{max(x[0] for x in new):.0f})", all(100 <= x[0] <= 240 for x in new))
        check(f"{v}: and a low light shows the relief it used to lose (contrast {new[0][1]:.0f} against {old[0][1]:.1f} at 15°)",
              new[0][1] > 20 * max(old[0][1], 0.5) and new[0][1] > new[-1][1])
        up = np.asarray(screen_image(v, D, dict(lit(v, 315, 90), pbr_expose=1), Display(layer="Shading")))
        up0 = np.asarray(screen_image(v, D, dict(lit(v, 315, 90), pbr_expose=0), Display(layer="Shading")))
        check(f"{v}: with the light overhead the picture is exactly the classic one", np.array_equal(up, up0))
    from helio.core import shading_engines as SE
    real_rust = SE._rust
    for el in (20, 35, 55):
        a_ = np.asarray(screen_image("Standard PBR", D, dict(lit("Standard PBR", 200, el)), Display(layer="Shading"))).astype(np.float32)
        SE._rust = lambda *a, **k: False
        try:
            b_ = np.asarray(screen_image("Standard PBR", D, dict(lit("Standard PBR", 200, el)), Display(layer="Shading"))).astype(np.float32)
        finally:
            SE._rust = real_rust
        check(f"Standard PBR at {el}°: Rust and NumPy agree (largest difference {float(np.abs(a_ - b_).max()):.1f}/255)",
              float(np.abs(a_ - b_).max()) <= 2.0)

    print("Moving the light changes the picture, and moving it back restores it")
    drifts = []
    for v in dome_views:
        a = np.asarray(screen_image(v, D, lit(v, 90, 40.0), Display(layer="Shading"))).astype(np.float32)
        screen_image(v, D, lit(v, 270, 40.0), Display(layer="Shading"))
        b = np.asarray(screen_image(v, D, lit(v, 90, 40.0), Display(layer="Shading"))).astype(np.float32)
        if float(np.abs(a - b).max()) > 0:
            drifts.append(v)
    check("the same light always draws the same picture (nothing sticks from the last light)", not drifts, drifts)

    print("Through the real window")
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    w = hw.MainWindow(appearance="dark")
    w.resize(1300, 850)
    w.show()
    w.open_paths([os.path.dirname(paths[0])])
    w.project.set_lights(L, "file:golden.lp")
    w.project.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.01)
    ins = w.workspaces["Inspect"]
    w.go_workspace("Inspect")
    app.processEvents()
    ins.wait_render()
    ph2, pw2 = w.render_data.ph, w.render_data.pw
    n2 = w.project.result.normals_full
    lat2 = cv2.resize(np.stack([n2[..., 0], -n2[..., 1]], -1), (pw2, ph2))
    sel2 = cv2.resize((np.linalg.norm(np.stack([n2[..., 0], -n2[..., 1]], -1), axis=-1) > 0.08).astype(np.uint8), (pw2, ph2)) > 0
    w.display.layer = "Shading"                    # the grey shading: the colour layer would add the albedo's pattern
    for view in ("Standard PBR", "Image Quotient (IQ)", "Cast-Shadow Raking", "Specular Enhancement"):
        w.set_view(view)
        ins.wait_render()
        scs = []
        for az in (0, 90, 180, 270):
            lx, ly, lz = VC.from_angles(az, 40.0)
            ins._dome_moved(lx, ly, lz, False)
            ins.wait_render()
            img = np.asarray(ins._last_image).astype(np.float32)
            g = img.mean(-1) if img.ndim == 3 else img
            u = np.array([math.sin(math.radians(az)), -math.cos(math.radians(az))])
            scs.append(float(np.corrcoef((lat2 @ u)[sel2], g[sel2])[0, 1]))
        check(f"{view}: moving the dome in the window lights the side it points to ({min(scs):.2f} worst)",
              min(scs) >= MIN_CORR, [round(x, 2) for x in scs])
    print("\nCorrelation of brightness with 'the slope faces the light', per direction (N, NE, E, SE, S, SW, W, NW):")
    for v, sc in table.items():
        print(f"  {v:44s} " + " ".join(f"{x:5.2f}" for x in sc))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
