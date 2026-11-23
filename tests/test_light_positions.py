"""Each light where two balls place it, and the light calibration's own error
(helio.core.light_positions).

* two balls' directions meet at the light: exact on true highlights; rays
  nearly parallel (a far light) are left unplaced;
* the calibration's error: larger for a rougher rim fit, and carried into the
  normals at about the size of the lights' own error;
* lamps close to a flat board, at different distances: the near-field solve
  with one dome radius leaves the board tilted, with each lamp at its own
  position it comes out flat, also through the whole solve.

    python tests/test_light_positions.py
"""
import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def highlight(s, lamp):
    """Where lamp (photo pixels, y down, z up) shows on ball s, seen from straight above."""
    from helio.core.sphere import Highlight
    d = lamp - np.array([s.cx, s.cy, s.r])
    d /= np.linalg.norm(d)
    N = d + np.array([0.0, 0.0, 1.0])
    N /= np.linalg.norm(N)
    return Highlight(s.cx + N[0] * s.r, s.cy + N[1] * s.r, 5.0, True)


W, H = 400, 300


def lamps():
    """Eight lamps 260-420 px above the board, round it, at different distances."""
    out = []
    for k, a in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False)):
        rad = 300 + 60 * (k % 3)
        out.append([W / 2 + rad * math.cos(a), H / 2 + rad * math.sin(a), 260 + 25 * k])
    return np.array(out)


def main():
    from helio.core import light_positions as P
    from helio.core.sphere import Calibration, Sphere, light_from_highlight

    print("Two balls")
    s1 = Sphere(60, 150, 20, rms=0.3, inliers=150)
    s2 = Sphere(340, 150, 20, rms=0.3, inliers=150)
    lp = lamps()
    h1 = [highlight(s1, l) for l in lp]
    h2 = [highlight(s2, l) for l in lp]
    L1 = np.array([light_from_highlight(s1, h.x, h.y) for h in h1])
    cal = Calibration(s1, L1, h1, sphere2=s2)
    cal.highlights2 = h2
    r = P.positions_from_two_balls(cal)
    check("every lamp placed where the two directions meet (to 0.01 px)",
          r["placed"] == 8 and float(np.abs(r["positions"] - lp).max()) < 0.01,
          float(np.abs(r["positions"] - lp).max()))
    far = P.triangulate([0, 0, 0], [0, 0.3, 1], [100, 0, 0], [0, 0.3, 1])
    check("rays from a far light (parallel) are left unplaced", far is None)
    check("one ball: no positions", P.positions_from_two_balls(Calibration(s1, L1, h1)) is None)

    print("\nThe calibration's own error")
    e = P.calibration_error(cal)
    rough = Calibration(Sphere(60, 150, 20, rms=1.5, inliers=150), L1, h1)
    er = P.calibration_error(rough)
    check(f"a ball of 20 px radius measures each light to about {e['direction_deg']:.1f}°, "
          f"{e['normal_deg']:.1f}° in the normals", 0.2 < e["direction_deg"] < 5
          and 0.2 * e["direction_deg"] < e["normal_deg"] < 2 * e["direction_deg"], e)
    check(f"…a rougher rim, more ({er['direction_deg']:.1f}°)", er["direction_deg"] > e["direction_deg"])
    check("…no error, no error in the normals", P.normal_error_deg(L1, np.zeros(8)) < 1e-6)

    print("\nA flat board under close lamps")
    from helio.core.photometric_stereo import solve_near_field_normals
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    pts = np.dstack([xx, yy, np.zeros_like(xx)])
    imgs = []
    for l in lp:
        d = l - pts
        dist = np.linalg.norm(d, axis=2)
        imgs.append(200.0 * 300.0 ** 2 * d[..., 2] / dist ** 3)            # cosine / distance²
    imgs = np.array(imgs, np.float32)
    # Directions as one ball in the middle measures them (y up), and the dome at their mean distance
    mid = np.array([W / 2, H / 2, 0.0])
    Lm = np.array([[v[0], -v[1], v[2]] / np.linalg.norm(v) for v in (lp - mid)])
    R = float(np.mean(np.linalg.norm(lp - mid, axis=1)))
    pos = np.column_stack([lp[:, 0] - W / 2, H / 2 - lp[:, 1], lp[:, 2]])

    def tilt(n):
        n = n[20:-20, 20:-20].reshape(-1, 3)
        return float(np.degrees(np.arccos(np.clip(n[:, 2], -1, 1))).mean())
    t_dome = tilt(solve_near_field_normals(imgs, Lm, H, W, dome_radius_px=R))
    t_pos = tilt(solve_near_field_normals(imgs, Lm, H, W, dome_radius_px=R, positions=pos))
    check(f"one dome radius leaves the board tilted ({t_dome:.1f}°); each lamp at its own "
          f"position, flat ({t_pos:.2f}°)", t_pos < 0.3 and t_dome > 3 * t_pos + 0.5, (t_dome, t_pos))

    print("\nThrough the whole solve")
    import cv2
    from helio.application.pipeline import SolveRequest, solve
    from helio.domain.settings import ProcessingSettings
    tmp = tempfile.mkdtemp(prefix="helio_lpos_")
    paths = []
    top = float(imgs.max())
    for i, im in enumerate(imgs):
        p = os.path.join(tmp, f"b{i:02d}.png")
        cv2.imwrite(p, np.dstack([np.round(im / top * 240).astype(np.uint8)] * 3))
        paths.append(p)
    st = ProcessingSettings(photo_tone="linear", near_field=True, near_field_from="ball",
                            normal_engine="Linear Least Squares")
    base = dict(paths=paths, lights=Lm.astype(np.float32), light_origin=(W / 2, H / 2))
    r_dome = solve(SolveRequest(settings=st, **base))
    r_pos = solve(SolveRequest(settings=st, light_positions=lp, **base))
    a, b = tilt(r_dome.normals_full), tilt(r_pos.normals_full)
    check(f"Process with the positions: flat ({b:.2f}°, one radius {a:.1f}°)", b < 0.5 and b < a, (a, b))


main()
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
