"""Precision photometric stereo: the white-board light field, the lens
calibration and the accuracy measures, on synthetic captures whose truth is
known.

* Four lights 40° up, unequal (0.75-1.3×) and each falling off across the
  frame by up to 50%: a flat board then reads tilted.  With the light field
  measured on a white board under the same lights, the flat reads flat
  again and a 6-pixel step measures 6 pixels.
* A chessboard photographed 14 times through a lens with known barrel
  distortion: the calibration finds it, and undistorting straightens the
  board's lines.
* step_height, flatness and path_height_sigma on known surfaces.

    python tests/test_precision_ps.py
"""
import math
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + ("" if cond else f"  {detail}"))


H, W = 240, 320
STEP = 6.0


def lights4(el=40.0):
    out = []
    for az in (45, 135, 225, 315):
        a, e = math.radians(az), math.radians(el)
        out.append((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    return np.array(out)


def surface():
    """Heights in pixels: a flat base and a 6 px plateau with soft edges."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    from scipy.ndimage import gaussian_filter
    z = np.zeros((H, W))
    z[80:170, 170:280] = STEP
    z = gaussian_filter(z, 3.0)
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(z)])          # +Y up: rows run down
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return z, n


def fields(L):
    """Each light's strength and fall-off: brighter towards the light's side."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    u, v = (xx - W / 2) / (W / 2), ((H / 2) - yy) / (H / 2)            # +Y up
    strengths = (0.75, 1.0, 1.3, 0.9)
    out = []
    for s, l in zip(strengths, L):
        d = np.hypot(l[0], l[1])
        g = 1.0 + 0.25 * (u * l[0] / d + v * l[1] / d)                  # up to ±25 %
        out.append(s * g)
    return out


def render(n, L, E, albedo=0.55):
    return [np.clip(255 * albedo * e * np.clip((n * l).sum(2), 0, None), 0, 255).astype(np.uint8)
            for l, e in zip(L, E)]


def solve(frames, L, field=None):
    from helio.core.light_field import factor
    from helio.core.photometric_stereo import linear_least_squares_normals
    st = np.stack([f.astype(np.float32) for f in frames])
    if field is not None:
        st = np.stack([st[i] * factor(field, i, (W, H)) for i in range(len(st))])
    n = linear_least_squares_normals(L.astype(np.float32), st.reshape(len(st), -1)).reshape(H, W, 3)
    return n


def base_tilt(n):
    m = n[10:70, 10:150].reshape(-1, 3).mean(0)
    return math.degrees(math.acos(m[2] / np.linalg.norm(m)))


def light_field_checks():
    from helio.core import light_field as LF
    from helio.core.depth_integration import poisson_depth_fft
    from helio.core.metrology import flatness, step_height, true_shape_scale
    print("White-board light field")
    L = lights4()
    z, n = surface()
    E = fields(L)
    frames = render(n, L, E)
    flat_n = np.dstack([np.zeros((H, W)), np.zeros((H, W)), np.ones((H, W))])
    boards = [b.astype(np.float32) for b in render(flat_n, L, E, albedo=0.8)]
    field = LF.from_boards(boards, L)
    check("the field finds the lamps' spread (1.3 / 0.75 = 1.73)",
          abs(field["strength_spread"] - 1.3 / 0.75) < 0.08, field["strength_spread"])
    check("…and the fall-off (about 2x corner to corner)", 0.8 < field["falloff"] < 1.2, field["falloff"])
    n0, n1 = solve(frames, L), solve(frames, L, field)
    t0, t1 = base_tilt(n0), base_tilt(n1)
    check("uneven lamps tilt the flat base", t0 > 3.0, t0)
    check("…the light field takes the tilt out (under 0.5°)", t1 < 0.5, t1)
    path = os.path.join(tempfile.mkdtemp(), "lf.npz")
    LF.save(path, field)
    back = LF.load(path)
    check("the field saves and loads", np.allclose(back["maps"], field["maps"])
          and back["frame_size"] == (W, H))
    f_half = LF.factor(back, 1, (W // 2, H // 2), (10, 10, 60, 50))
    check("…and fits any working size and region", f_half.shape == (40, 50))
    check("the lights it was shot with are recognised", LF.check_lights(back, L) == "")
    L2 = L.copy()
    L2[0] = (0.2, 0.2, 0.96)
    check("…moved lights are not", "Light 1" in LF.check_lights(back, L2))
    try:
        LF.from_boards([np.full((H, W), 255, np.uint8)] * 4, L)
        check("a clipped board is refused", False)
    except ValueError:
        check("a clipped board is refused", True)
    d = poisson_depth_fft(n1)
    d = d * true_shape_scale(d, n1)
    st = step_height(d, (10, 10, 150, 230), (190, 100, 260, 150))
    # the same frames evened by the lamps' true fields: what perfect knowledge of the lamps gives
    exact = solve([np.clip(f.astype(np.float32) / e, 0, None) for f, e in zip(frames, E)], L)
    dx = poisson_depth_fft(exact)
    dx = dx * true_shape_scale(dx, exact)
    sx = step_height(dx, (10, 10, 150, 230), (190, 100, 260, 150))["step"]
    check(f"the step measures as with the lamps known exactly (within 1%), and within 4% of "
          f"{STEP:.0f} px (8-bit frames, the integrator)",
          abs(st["step"] - sx) / sx < 0.01 and abs(st["step"] - STEP) / STEP < 0.04, (st["step"], sx))
    d0 = poisson_depth_fft(n0)
    d0 = d0 * true_shape_scale(d0, n0)
    f0, f1 = flatness(d0, (10, 10, 150, 230)), flatness(d, (10, 10, 150, 230))
    check("the base comes out flat (RMS under 0.1 px of height) and level",
          f1["rms"] < 0.1 and f1["tilt_deg"] < 0.3 < f0["tilt_deg"], (f0, f1))


def metrology_checks():
    from helio.core.metrology import flatness, path_height_sigma, step_height
    print("\nAccuracy measures")
    yy, xx = np.mgrid[0:100, 0:200].astype(np.float64)
    Hm = 0.02 * xx - 0.01 * yy + 3.0
    Hm[30:70, 120:180] += 4.0
    st = step_height(Hm, (5, 5, 100, 95), (130, 40, 170, 60))
    check("a step on a tilted base is its height above the base's plane", abs(st["step"] - 4.0) < 1e-9)
    f = flatness(Hm, (5, 5, 100, 95))
    check("a tilted plane is flat, its tilt reported",
          f["rms"] < 1e-9 and abs(f["tilt_deg"] - math.degrees(math.atan(math.hypot(.02, .01)))) < 1e-6)
    s1 = path_height_sigma(np.full((100, 200), 1.0), 0, 50, 99, 50)
    s4 = path_height_sigma(np.full((100, 200), 1.0), 0, 50, 199, 50)
    check("height uncertainty grows as the root of the length", abs(s4 / s1 - math.sqrt(2)) < 0.02,
          (s1, s4))
    check("…and is none without a precision map", path_height_sigma(None, 0, 0, 10, 10) is None)


def lens_checks():
    from helio.core import lens
    print("\nLens calibration")
    Wc, Hc = 1200, 900
    K = np.array([[1100.0, 0, Wc / 2 + 7], [0, 1100.0, Hc / 2 - 5], [0, 0, 1]])
    dist = np.array([-0.12, 0.03, 0.0, 0.0, 0.0])
    sq = 40
    cols, rows = 9, 6
    board = np.full(((rows + 3) * sq, (cols + 3) * sq), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    rng = np.random.default_rng(4)
    # the distortion, as a map from each distorted pixel back to where it would be undistorted
    yy, xx = np.mgrid[0:Hc, 0:Wc].astype(np.float32)
    pts = np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(Hc, Wc, 2)
    greys = []
    bh, bw = board.shape
    # each view through a pinhole camera with focal length K: the board (its pixels as mm) at a
    # random pose, about 900 mm away, tilted up to 25 degrees
    S = np.array([[1.0, 0, -bw / 2], [0, 1.0, -bh / 2], [0, 0, 1]])
    for k in range(14):
        rv = rng.uniform(-0.45, 0.45, 3) * np.array([1, 1, 0.6])
        Rm, _ = cv2.Rodrigues(rv)
        t = np.array([rng.uniform(-110, 110), rng.uniform(-80, 80), rng.uniform(850, 1050)])
        Hm = K @ np.column_stack([Rm[:, 0], Rm[:, 1], t]) @ S
        straight = cv2.warpPerspective(board, Hm / Hm[2, 2], (Wc, Hc), borderValue=200)
        img = cv2.remap(straight, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=200)
        greys.append(img)
    cal = lens.calibrate(greys, (cols, rows))
    check("the chessboard is found in most views", cal["used"] >= 10, cal["used"])
    check("the fit is sub-pixel", cal["rms"] < 0.5, cal["rms"])
    check("the barrel distortion is found (k1 about -0.12)", abs(cal["dist"][0] + 0.12) < 0.03,
          cal["dist"][:3])
    check("…and the corners move by tens of pixels", cal["edge_px"] > 10, cal["edge_px"])
    # undistorting a frame straightens the board's edges: corners found again lie on a homography
    c_d = lens.find_corners(greys[0], (cols, rows))
    fixed = lens.undistort(greys[0], cal)
    c_u = lens.find_corners(fixed, (cols, rows))

    def bend(c):
        objp = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float32)
        Hh, _ = cv2.findHomography(objp, c.reshape(-1, 2))
        proj = cv2.perspectiveTransform(objp.reshape(-1, 1, 2), Hh).reshape(-1, 2)
        return float(np.sqrt(np.mean(np.sum((proj - c.reshape(-1, 2)) ** 2, axis=1))))
    b0, b1 = bend(c_d), bend(c_u)
    check("undistorting straightens the board (lines fit a plane's view)", b1 < 0.3 * b0, (b0, b1))
    p = os.path.join(tempfile.mkdtemp(), "lens.npz")
    lens.save(p, cal)
    back = lens.load(p)
    check("the calibration saves and loads", np.allclose(back["K"], cal["K"]) and back["size"] == (Wc, Hc))
    half = lens.undistort(cv2.resize(greys[0], (Wc // 2, Hc // 2)), back, scale=0.5)
    full = cv2.resize(fixed, (Wc // 2, Hc // 2))
    check("…and works at half size", float(np.mean(np.abs(half.astype(int) - full.astype(int)))) < 6)
    check("frames of another size are refused", "calibrated on" in lens.check_size(back, (100, 100)))
    from helio.application.precision import parse_pattern
    check("the corner count reads '9 x 6'", parse_pattern("9 x 6") == (9, 6)
          and parse_pattern("12,8") == (12, 8))


light_field_checks()
metrology_checks()
lens_checks()
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
