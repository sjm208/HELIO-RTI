"""No ball: the lights found from the photographs (helio.core.uncalibrated).

* A surface of uneven bumps under twelve equal lamps, solved with no lights
  given: the normals and the lights come back to within a few degrees, and
  it is called raised (from the surround: nothing casts a shadow here).
* The same bumps sunk into the surface, with their cast shadows drawn: the
  shadows say sunken, and the normals come back the right way in.
* Lamps 40% uneven, and four frames: the photographs do not bear the lamps
  out, so the rough solve is used and says why.
* A large frame: the lights are worked out on a reduced copy and the normals
  made at full size, the same answer.
* Through the pipeline (sRGB files, no lights): the record travels with the
  result, and Turn the relief inside out turns it.
* In the app: Capture's No ball card chooses the solver, shows itself chosen,
  and another light source puts the solver back.

    python tests/test_no_ball.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def surface(h, w, sign=1.0):
    """Uneven bumps (no symmetry to hide a mirror), raised (sign 1) or sunk."""
    import numpy as np
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = np.zeros((h, w))
    for cx, cy, r, a in ((0.30, 0.35, 0.14, 0.9), (0.68, 0.30, 0.10, 0.6), (0.55, 0.70, 0.18, 1.0),
                         (0.25, 0.75, 0.08, 0.5), (0.80, 0.62, 0.07, 0.4)):
        z += a * r * w * np.exp(-(((xx - cx * w) ** 2 + (yy - cy * h) ** 2) / (r * w) ** 2))
    z *= sign
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(gx)])            # y up: rows run down
    return z, n / np.linalg.norm(n, axis=-1, keepdims=True)


def lamps(k=12, uneven=0.0):
    import numpy as np
    L = []
    for i in range(k):
        az, el = 2 * np.pi * i / k + 0.3, np.radians(30 + 12 * (i % 3))
        L.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    s = np.random.default_rng(3).normal(1, uneven, k).clip(0.4, 1.8) if uneven else np.ones(k)
    return np.array(L), s


def render(z, n, L, s, shadows=False):
    import numpy as np
    h, w = z.shape
    I = np.clip(np.einsum("hwc,nc->nhw", n, L), 0, None) * s[:, None, None]
    if shadows:
        ys, xs = np.mgrid[0:h, 0:w]
        for i, (lx, ly, lz) in enumerate(L):
            flat = np.hypot(lx, ly)
            dx, dy, rise = lx / flat, ly / flat, lz / flat
            dark = np.zeros((h, w), bool)
            for k in range(1, 80):
                c = np.clip(np.round(xs + k * dx).astype(int), 0, w - 1)
                r = np.clip(np.round(ys - k * dy).astype(int), 0, h - 1)
                dark |= z[r, c] - z > k * rise
            I[i][dark] *= 0.03
    return (I * 0.8).reshape(len(L), -1)


def ang(a, b):
    import numpy as np
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))


def main():
    import numpy as np
    from helio.core.uncalibrated import solve_no_ball
    print("=" * 60)
    print("No ball: the lights found from the photographs")
    print("=" * 60)
    h, w = 120, 150
    inner = np.zeros((h, w), bool)
    inner[8:-8, 8:-8] = True

    print("\nRaised bumps, twelve equal lamps, no shadows cast")
    z, n = surface(h, w)
    L, s = lamps()
    Lf, N, info = solve_no_ball(render(z, n, L, s), h, w)
    e = float(ang(N.reshape(h, w, 3), n)[inner].mean())
    le = float(ang(Lf, L).mean())
    check(f"the lights are found ({le:.1f}° out)", info["used"] == "found" and le < 4, info)
    check(f"the normals come back ({e:.1f}° out)", e < 4, e)
    check("…called raised, from the surround", info["raised"] == "surround", info["raised"])
    check("…the lamps found even", info["spread"] < 0.05, info["spread"])

    print("\nThe bumps sunk in, their shadows cast")
    zs, ns = surface(h, w, -1.0)
    Lf, N, info = solve_no_ball(render(zs, ns, L, s, shadows=True), h, w)
    lit = np.isfinite(zs) & inner
    e = float(ang(N.reshape(h, w, 3), ns)[lit].mean())
    check("the shadows say sunken", info["raised"] == "shadows" and info["votes"][0] > info["votes"][1],
          info["votes"])
    check(f"…and the normals come back the right way in ({e:.1f}° out)", e < 8, e)

    print("\nWhen the photographs do not bear the lamps out")
    L2, s2 = lamps(uneven=0.4)
    _L, _N, info = solve_no_ball(render(z, n, L2, s2), h, w)
    check("lamps 40% uneven: the rough solve, saying the lamps differ",
          info["used"] == "rough" and "differ" in info["why"], info)
    L4, s4 = lamps(k=4)
    _L, N4, info = solve_no_ball(render(z, n, L4, s4), h, w)
    check("four frames: the rough solve, saying six are needed",
          info["used"] == "rough" and "six" in info["why"] and N4.shape == (h * w, 3), info)

    print("\nA large frame")
    import cv2
    H, W = 1500, 1875
    small = render(z, n, L, s).reshape(len(L), h, w)
    big = np.stack([cv2.resize(f.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
                    for f in small]).reshape(len(L), -1)
    import time
    t = time.time()
    Lb, Nb, info_b = solve_no_ball(big, H, W)
    took = time.time() - t
    nb = cv2.resize(Nb.reshape(H, W, 3), (w, h), interpolation=cv2.INTER_AREA)
    nb /= np.linalg.norm(nb, axis=-1, keepdims=True)
    e = float(ang(nb, n)[inner].mean())
    check(f"found on a reduced copy, made at full size ({e:.1f}° out, {took:.1f} s)",
          info_b["used"] == "found" and Nb.shape == (H * W, 3) and e < 5 and took < 60, (info_b, e))

    print("\nThrough the pipeline")
    import dataclasses
    from helio.application.pipeline import SolveRequest, solve
    from helio.core.radiometry import linear_to_srgb
    from helio.domain.settings import ProcessingSettings
    tmp = tempfile.mkdtemp(prefix="helio_noball_")
    paths = []
    for i, f in enumerate(render(z, n, L, s).reshape(len(L), h, w)):
        enc = np.round(linear_to_srgb(np.clip(f, 0, 1) * 255.0)).astype(np.uint8)
        p = os.path.join(tmp, f"f{i:02d}.png")
        cv2.imwrite(p, np.dstack([enc] * 3))
        paths.append(p)
    base = ProcessingSettings(normal_engine="Uncalibrated PCA", photo_tone="srgb")
    r = solve(SolveRequest(paths=paths, lights=None, settings=base))
    e = float(ang(r.normals_full, n)[inner].mean())
    check(f"no lights given: solved, the record kept ({e:.1f}° out)",
          r.no_ball is not None and r.no_ball["used"] == "found" and e < 5, (r.no_ball, e))
    r2 = solve(SolveRequest(paths=paths, lights=None,
                            settings=dataclasses.replace(base, no_ball_inside_out=True)))
    d = float((r.normals_full[..., :2] * r2.normals_full[..., :2]).sum(-1)[inner].mean())
    check("Turn the relief inside out turns it (the slopes reversed)",
          d < 0 and r2.no_ball.get("inside_out"), d)
    from helio.application import exports as X
    try:
        X.light_file(r.light_matrix, "estimated", os.path.join(tmp, "x.lp"))
        said = ""
    except X.NothingToExport as ex:
        said = str(ex)
    check("a .lp of the found lights is refused, saying why and what to shoot",
          "found from the photographs" in said and "chrome ball" in said, said)

    print("\nIn the app")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from helio_qt.window import MainWindow
    win = MainWindow()
    win.project.set_images(paths)
    win.project.update_settings(normal_engine="Shadow-Rejection LS")
    win.registry.trigger("capture.no_ball")
    app.processEvents()
    cap = win.workspaces["Capture"]
    cap.refresh()
    check("No ball chooses the solver that finds the lights",
          win.project.settings.normal_engine == "Uncalibrated PCA")
    check("…Capture shows it chosen, and what will happen",
          cap.method.current() == "none" and "found when you process" in cap.method_status.text(),
          (cap.method.current(), cap.method_status.text()))
    check("…the capture can be processed with no lights", not win.registry.reason("process.run"),
          win.registry.reason("process.run"))
    pw = win.workspaces["Process"]
    pw.refresh()
    check("…Process offers Turn the relief inside out", not pw.chk["no_ball_inside_out"].isHidden())
    win.leave_no_ball()
    pw.refresh()
    check("…and only for the no-ball solver", pw.chk["no_ball_inside_out"].isHidden())
    check("another light source puts the solver back",
          win.project.settings.normal_engine == "Shadow-Rejection LS", win.project.settings.normal_engine)
    win.close()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
