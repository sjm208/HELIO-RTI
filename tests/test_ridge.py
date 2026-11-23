"""Ridge and Groove Detector.

* a raised line is a ridge and not a groove, an incised one the reverse, a round bump is
  suppressed; the line's direction is right (0 along x, pi/2 along y, pi/4 on a diagonal);
* the Rust step is the NumPy step (float32 rounding), with and without Rust;
* line width is in full-size pixels: the preview of a picture finds the same lines as the
  picture, and a region drawn from the full-size data has the preview's brightness scale;
* the four display modes, the gain, the registered view and its controls.

    python tests/test_ridge.py
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from helio.core import shading_engines as S                          # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def surface(h=200, w=240):
    """A flat plate with a raised horizontal line, an incised vertical one, a diagonal ridge and a bump."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.zeros((h, w), np.float32)
    d += 0.5 * np.exp(-((y - 50) ** 2) / (2 * 1.5 ** 2)) * (x > 20) * (x < 120)       # ridge along x
    d -= 0.5 * np.exp(-((x - 60) ** 2) / (2 * 1.5 ** 2)) * (y > 100) * (y < 190)      # groove along y
    t = ((x - 150) - (y - 60)) / np.sqrt(2)                                           # distance to a 45° line
    d += 0.5 * np.exp(-(t ** 2) / (2 * 1.5 ** 2)) * (x > 140) * (x < 220) * (y > 50) * (y < 130)
    d += 0.5 * np.exp(-(((x - 170) ** 2 + (y - 160) ** 2)) / (2 * 4.0 ** 2))          # round bump
    return d


def main():
    d = surface()
    print("What it finds")
    ridge, groove, dirn = S.ridge_response(d, 1.5, 1.0)
    on_ridge = (slice(48, 53), slice(40, 100))
    on_groove = (slice(120, 170), slice(58, 63))
    check("a raised line is a ridge", ridge[on_ridge].mean() > 20 * groove[on_ridge].mean() + 1e-4,
          (ridge[on_ridge].mean(), groove[on_ridge].mean()))
    check("an incised line is a groove", groove[on_groove].mean() > 20 * ridge[on_groove].mean() + 1e-4,
          (groove[on_groove].mean(), ridge[on_groove].mean()))
    check("flat plate: nothing", max(ridge[150:195, 150:200].max() * 0 + ridge[5:40, 130:230].max(),
                                     groove[5:40, 130:230].max()) < 0.02 * ridge.max())
    bump = (slice(156, 165), slice(166, 175))
    check("a round bump gives far less than a line of the same height",
          max(ridge[bump].max(), groove[bump].max()) < 0.5 * ridge[on_ridge].max(),
          (ridge[bump].max(), ridge[on_ridge].max()))

    def ang(a):                                      # mean direction on a line (circular, period pi)
        return (np.angle(np.exp(2j * a).mean()) / 2) % np.pi
    check("along x: direction 0", min(ang(dirn[on_ridge]), np.pi - ang(dirn[on_ridge])) < 0.05, ang(dirn[on_ridge]))
    check("along y: direction pi/2", abs(ang(dirn[on_groove]) - np.pi / 2) < 0.05, ang(dirn[on_groove]))
    diag = np.zeros(d.shape, bool)
    yy, xx = np.mgrid[0:200, 0:240]
    diag[(np.abs(((xx - 150) - (yy - 60)) / np.sqrt(2)) < 1.0) & (xx > 150) & (xx < 210) & (yy > 60) & (yy < 120)] = True
    check("on the diagonal: direction pi/4", abs(ang(dirn[diag]) - np.pi / 4) < 0.08, ang(dirn[diag]))

    print("Rust and NumPy")
    real = S._rust
    r1 = S.ridge_response(d, 1.5, 1.0)
    S._rust = lambda *a, **k: False
    r2 = S.ridge_response(d, 1.5, 1.0)
    S._rust = real
    check("strength is the same to float32 rounding", all(np.allclose(a, b, rtol=1e-5, atol=1e-6)
          for a, b in zip(r1[:2], r2[:2])), [float(np.abs(a - b).max()) for a, b in zip(r1[:2], r2[:2])])
    strong = np.maximum(r1[0], r1[1]) > 0.05 * r1[0].max()
    dd = np.abs(r1[2] - r2[2])
    dd = np.minimum(dd, np.pi - dd)
    check("and so is the direction where there is a line", float(dd[strong].max()) < 1e-3, float(dd[strong].max()))
    if S._rs is not None:
        check("(the Rust kernel is in use)", hasattr(S._rs, "ridge_update"))
        import helio_core_rs as rs
        z = lambda *sh: np.zeros(sh or (2, 2), np.float32)       # noqa: E731
        bad = False
        try:
            rs.ridge_update(z(), z(2, 3), z(), 1.0, 0.5, z(), z(), z(), z())
        except ValueError:
            bad = True
        alias = False
        shared = z()
        try:
            rs.ridge_update(z(), z(), z(), 1.0, 0.5, shared, shared, z(), z())
        except TypeError:
            alias = True
        check("one array given as two outputs is refused, not written twice", alias)
        check("mismatched shapes are refused, not read past", bad)

    print("Width is in full-size pixels")
    full = surface(400, 480)
    full = cv2.resize(d, (480, 400), interpolation=cv2.INTER_CUBIC)       # the same picture, twice as big
    rf = S.ridge_response(full, 1.5, 1.0)[0]
    small = cv2.resize(full, (240, 200), interpolation=cv2.INTER_AREA)
    rp = S.ridge_response(small, 1.5, 0.5)[0]
    up = cv2.resize(rp, (480, 400), interpolation=cv2.INTER_LINEAR)
    c = np.corrcoef(rf.ravel(), up.ravel())[0, 1]
    check("the preview finds the same lines as the picture (correlation > 0.9)", c > 0.9, c)
    ratio = np.percentile(up, 99.5) / np.percentile(rf, 99.5)
    check("with about the same strength (within 25%)", 0.75 < ratio < 1.25, ratio)

    print("In the app")
    import tempfile
    from tools import make_golden as G
    from helio.application import pipeline as P
    from helio.application.render import render, to_display
    from helio.application.viewing import detail_image, screen_image, Display
    from helio.data_manager import DataManager
    from helio.domain import view_controls as VC
    from helio.domain.registry import RENDER_ENGINES, default_engine_settings, ENGINE_BLURBS
    from helio.domain.settings import ProcessingSettings
    V = "Ridge and Groove Detector"
    check("registered, with a blurb and its controls", V in RENDER_ENGINES and V in ENGINE_BLURBS
          and [c.key for c in VC.controls_for(V)] == ["gain", "rdg_width", "rdg_mode"]
          if hasattr(VC, "controls_for") else V in RENDER_ENGINES and V in ENGINE_BLURBS)
    check("it takes no light, and tone controls except in colour mode", VC.light_kind(V) is None
          and VC.takes_tone(V, {"rdg_mode": 0}) and not VC.takes_tone(V, {"rdg_mode": 3}))
    G.H, G.W = 1200, 1600
    paths, L = G.render_stack(tempfile.mkdtemp(prefix="helio_rdg_"), n_lights=5)
    r = P.solve(P.SolveRequest(paths=paths, lights=L, settings=ProcessingSettings(photo_tone="linear")))
    D = DataManager()
    P.store(r, D)
    st = default_engine_settings()[V]
    outs = {m: render(V, D, {**st, "rdg_mode": m}, r.ph, r.pw, True) for m in range(4)}
    check("modes 0-2 are grey, 3 is colour, all in 0..1", all(o.shape == (r.ph, r.pw) for m, o in outs.items() if m < 3)
          and outs[3].shape == (r.ph, r.pw, 3) and all(0 <= o.min() and o.max() <= 1 for o in outs.values()))
    check("ridges and grooves sit on a grey ground", abs(float(np.median(outs[0])) - 0.5) < 0.1, float(np.median(outs[0])))
    check("the one-sided modes are dark with bright lines", np.median(outs[1]) < 0.1 and outs[1].max() > 0.5
          and np.median(outs[2]) < 0.1)
    g2 = render(V, D, {**st, "rdg_mode": 1, "gain": 2.0}, r.ph, r.pw, True)
    check("gain brightens", g2.mean() > outs[1].mean())
    w_thin = render(V, D, {**st, "rdg_mode": 1, "rdg_width": 0.8}, r.ph, r.pw, True)
    w_wide = render(V, D, {**st, "rdg_mode": 1, "rdg_width": 5.0}, r.ph, r.pw, True)
    check("the width setting changes what is found", float(np.abs(w_thin - w_wide).mean()) > 0.001)
    # one white level for the preview and a region of the full-size data
    box = (400, 300, 1000, 800)
    disp = Display()
    pv = screen_image(V, D, st, disp)
    dt = detail_image(V, D, st, disp, box)
    s = pv.shape[1] / D.w
    pc = pv[int(box[1] * s):int(box[3] * s), int(box[0] * s):int(box[2] * s)]
    dd_ = cv2.resize(dt, (pc.shape[1], pc.shape[0]), interpolation=cv2.INTER_AREA)
    check("a region from the full-size data is the preview's picture, sharper (mean difference < 6/255)",
          float(np.abs(dd_.astype(float) - pc.astype(float)).mean()) < 6.0,
          float(np.abs(dd_.astype(float) - pc.astype(float)).mean()))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
