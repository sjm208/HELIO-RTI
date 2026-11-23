"""Solved reflectance: a RAW capture's colour with the shading taken out.

* on a painted, bumpy, matt surface the median colour keeps some shading and
  the solved one is nearer the true paint;
* a cast shadow and a highlight in some frames do not get through;
* the colour's proportions stay the median's;
* it is made only for linear frames left as they were;
* the box swaps the colour everything reads, and says which is in use.

    python tests/test_reflectance.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
import numpy as np  # noqa: E402

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


def surface(h=120, w=160, n=8):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # a slope that changes across the picture: ridges 40 px apart
    gx = 0.8 * np.sin(xx / 40 * 2 * np.pi)
    N = np.dstack([-gx, np.zeros_like(gx), np.ones_like(gx)])
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    az = np.arange(n) * 2 * np.pi / n
    # lamps more on one side than the other, as on a real rig, so the median is left with shading
    L = np.stack([np.cos(az) * 0.6 + 0.25, np.sin(az) * 0.6, np.full(n, 0.8)], 1)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    paint = 0.25 + 0.45 * (yy > h / 2)                    # two paints, by rows: not the ridges
    rho = paint * 200
    stack = np.stack([rho * np.clip(N @ l.astype(np.float32), 0, None) for l in L]).astype(np.float32)
    return stack, L, N, paint


def main():
    from helio.core.radiometry import linear_to_srgb
    from helio.core.reflectance import solved_level, solved_reflectance
    stack, L, N, paint = surface()
    grey_median = np.median(stack, axis=0)
    tint = np.array([1.2, 1.0, 0.7], np.float32)
    rgb8 = np.round(linear_to_srgb(np.clip(grey_median[..., None] * tint / 1.0, 0, 255))).astype(np.uint8)

    rho = solved_level(stack, L, N)
    ripple = lambda a: float(np.std(a[10:50] / a[10:50].mean()))     # one paint: all ripple is shading
    check(f"the shading goes: {ripple(grey_median):.3f} in the median, {ripple(rho):.4f} solved",
          ripple(rho) < 0.02 and ripple(grey_median) > 3 * ripple(rho))
    check("the two paints keep their ratio", abs(rho[100:].mean() / rho[:50].mean() - 0.7 / 0.25) < 0.05)

    s2 = stack.copy()
    s2[0, 20:40, 30:60] = 0.0                                          # a cast shadow in one frame
    s2[1, 20:40, 30:60] = 255.0                                        # a clipped highlight in another
    r2 = solved_level(s2, L, N)
    check("a cast shadow and a highlight do not get through",
          abs(float(np.median(r2[20:40, 30:60] / rho[20:40, 30:60])) - 1) < 0.1)

    out = solved_reflectance(stack, L, N, rgb8)
    from helio.core.radiometry import srgb_to_linear
    a, b = srgb_to_linear(out[15, 30]), srgb_to_linear(rgb8[15, 30])
    check("the colour's proportions stay the median's", out.dtype == np.uint8 and np.allclose(a / a[1], b / b[1], atol=0.05),
          (a, b))

    from types import SimpleNamespace as NS
    from helio.application.pipeline import _reflectance, use_solved_colour
    ps = NS(flatten_illumination=False, multi_norm=False, pre_clahe=False, adaptive_light=0.0)
    req = NS(frames=None)
    raw = [True] * len(L)
    check("made for a capture all in RAW", _reflectance(ps, req, raw, stack, L, N, rgb8) is not None)
    check("not for TIFFs or JPEGs, nor a mix", _reflectance(ps, req, [False] * len(L), stack, L, N, rgb8) is None
          and _reflectance(ps, req, [True] * (len(L) - 1) + [False], stack, L, N, rgb8) is None)
    check("not when the frames were altered for the solve",
          _reflectance(NS(**{**ps.__dict__, 'multi_norm': True}), req, raw, stack, L, N, rgb8) is None
          and _reflectance(NS(**{**ps.__dict__, 'flatten_illumination': True}), req, raw, stack, L, N, rgb8) is None)
    check("not without lights", _reflectance(ps, req, raw, stack, None, N, rgb8) is None)

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    import helio_qt.window as hw
    w = hw.MainWindow(appearance="dark")
    ins = w.workspaces["Inspect"]
    ins._sync_solved_colour()
    check("the box stays away without a solved reflectance", ins.chk_solved_colour.isHidden())
    D = w.render_data = NS(albedo_rgb_full=rgb8, albedo_proxy=rgb8[::2, ::2], median_colour=(rgb8, rgb8[::2, ::2]),
                           solved_colour=(out, out[::2, ::2]))
    w.refresh = lambda: None
    ins._sync_solved_colour()
    check("it shows with one, off", not ins.chk_solved_colour.isHidden() and not ins.chk_solved_colour.isChecked())
    ins.chk_solved_colour.setChecked(True)
    check("ticked, the colour everything reads is the solved one",
          D.albedo_rgb_full is out and D.albedo_proxy is D.solved_colour[1])
    ins.chk_solved_colour.setChecked(False)
    check("unticked, back to the median", D.albedo_rgb_full is rgb8)
    check("with nothing solved it says so", use_solved_colour(NS(), True) is False)
    w.render_data = None
    del app
    # the whole solve, on RAW frames (decoding stood in for: 16-bit linear files, read as a raw file would be)
    import cv2
    from helio.application.pipeline import SolveRequest, solve
    from helio.io import raw_loader
    d = tempfile.mkdtemp(prefix="helio_rf_")
    paths = []
    for i, f in enumerate(stack):
        p = os.path.join(d, f"f{i}.png")
        cv2.imwrite(p, np.dstack([np.clip(f * 257, 0, 65535).astype(np.uint16)] * 3))
        paths.append(p)
    was = raw_loader.is_raw, raw_loader.read_raw_linear
    raw_loader.is_raw = lambda _p: True
    raw_loader.read_raw_linear = lambda p: cv2.imread(p, cv2.IMREAD_UNCHANGED)
    try:
        r = solve(SolveRequest(paths=paths, lights=L))
    finally:
        raw_loader.is_raw, raw_loader.read_raw_linear = was
    r_tiff = solve(SolveRequest(paths=paths, lights=L))
    ok = r.reflectance_rgb
    band = (slice(10, 50), slice(40, 120))
    grey = lambda a: srgb_to_linear(a[band].mean(axis=2).astype(np.float32))
    spread = lambda a: float(np.std(a) / a.mean())
    check("a whole solve on RAW frames makes it, flatter than the median colour"
          + (f" ({spread(grey(r.albedo_rgb_full)):.3f} to {spread(grey(ok)):.3f})" if ok is not None else ""),
          ok is not None and ok.shape == r.albedo_rgb_full.shape
          and spread(grey(ok)) < 0.5 * spread(grey(r.albedo_rgb_full)))
    check("and not on other files", r_tiff.reflectance_rgb is None)
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
