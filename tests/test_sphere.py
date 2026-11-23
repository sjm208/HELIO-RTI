"""Chrome-sphere calibration gives the right lights, one per frame (KD-020, FIX-009).

Renders a mirror ball under known lights (a dark ball with a small bright
highlight where the light's reflection meets the camera, on grey foam) and
checks the finder measures the ball's own radius and recovers each light to
within a degree or two; that a frame without a highlight is reported in its
place instead of shifting its neighbours; that the perspective formula is the
classic one when the camera looks straight at the ball and exact when it
does not; and, on the real test capture, that the radius is the ball's.

    python tests/test_sphere.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.make_golden import test_images  # noqa: E402
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def light(az, el):
    """+Y-up unit light at compass azimuth *az* (0 = north/top) and elevation *el*."""
    a, e = math.radians(az), math.radians(el)
    return (math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e))


def render_ball(L, w=640, h=480, cx=470.3, cy=330.6, R=26.4, seed=0):
    """A frame: grey foam, a dark chrome ball, the light's highlight on it."""
    import numpy as np
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(118, 6, (h, w)), 0, 255)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # Anti-aliased disc: supersample the edge by distance.
    d = np.hypot(xx - cx, yy - cy)
    cover = np.clip(R + 0.5 - d, 0, 1)
    nx, ny = (xx - cx) / R, (cy - yy) / R
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    Rz = 2 * nz * nz - 1                           # reflection of V=(0,0,1): 2(N.V)N - V
    Rx, Ry = 2 * nz * nx, 2 * nz * ny
    cosang = Rx * L[0] + Ry * L[1] + Rz * L[2]
    hl = 255 * np.exp(-(1 - cosang) / 0.012)          # a lamp with some size
    body = 28 - 18 * (d / R) ** 4                  # darker towards the rim
    ball = np.clip(body + hl, 0, 255)
    out = img * (1 - cover) + ball * cover
    return out.astype(np.uint8)


def main():
    import cv2
    import numpy as np
    from helio.core.processing import auto_detect_sphere, detect_sphere_candidates
    from helio.core.sphere import (Sphere, calibrate, condition_number,
                                   light_from_highlight, view_vector)
    from helio.utils.file_utils import estimate_light_from_specular
    import glob
    print("=" * 60)
    print("Chrome-sphere calibration")
    print("=" * 60)

    truth = [light(az, el) for az, el in ((0, 30), (45, 25), (90, 40), (135, 35), (180, 50),
                                          (225, 20), (270, 45), (315, 60), (20, 70), (200, 15))]
    tmp = tempfile.mkdtemp(prefix="helio_sphere_")
    paths = []
    for i, L in enumerate(truth):
        p = os.path.join(tmp, f"f{i:02d}.png")
        cv2.imwrite(p, render_ball(L, seed=i))
        paths.append(p)

    print("\nThe ball")
    cands = detect_sphere_candidates(paths, 480, 640)
    check("found once", len(cands) == 1, cands)
    cx, cy, r = cands[0]
    check("its centre to within half a pixel", math.hypot(cx - 470.3, cy - 330.6) < 0.5,
          (cx, cy))
    check("its own radius to within 3% (was padded 43%)", abs(r - 26.4) / 26.4 < 0.03, r)

    print("\nThe lights")
    L = auto_detect_sphere(paths)
    check("one light per frame, in order", L is not None and len(L) == len(truth))
    err = [math.degrees(math.acos(max(-1, min(1, float(np.dot(a, b)))))) for a, b in zip(L, truth)]
    check("every light within 1.5°", max(err) < 1.5, [round(e, 2) for e in err])
    check("the raking ones too (15-25° up)", max(err[i] for i in (1, 5, 9)) < 1.5)

    print("\nA frame with no highlight stays in its place")
    dark = os.path.join(tmp, "f99_dark.png")
    cv2.imwrite(dark, (render_ball(truth[0]) * 0).astype(np.uint8))
    seq = paths[:3] + [dark] + paths[3:6]
    cal = calibrate(seq, Sphere(cx, cy, r))
    check("it is marked, with a reason", not cal.ok[3] and "black" in cal.highlights[3].reason,
          cal.highlights[3])
    check("the frames after it keep their own lights",
          all(math.degrees(math.acos(min(1, float(np.dot(cal.lights[i + 1], truth[i]))))) < 1.5
              for i in range(3, 6)))
    check("and the calibration says it is incomplete", not cal.complete()
          and "frame 4" in " ".join(cal.notes), cal.notes)
    check("the automatic path refuses it rather than misalign", auto_detect_sphere(seq) is None)
    check("near-identical lights are both kept (no merging)",
          len(calibrate([paths[0], paths[0], paths[1]], Sphere(cx, cy, r)).lights) == 3)

    print("\nThe formula")
    s = Sphere(100.0, 80.0, 30.0)
    same = all(np.allclose(light_from_highlight(s, hx, hy),
                           (lambda v: v / np.linalg.norm(v))(estimate_light_from_specular(100, 80, 30, hx, hy)),
                           atol=1e-9) for hx, hy in ((110, 70), (85, 95), (100, 80), (127, 81)))
    check("looking straight at the ball it is the classic formula", same)
    V = np.array(view_vector(900, 120, 1000, 800, 1200.0))
    # The ball is right of and above the axis, so the camera is left of and below it.
    check("the view points from the ball back towards the lens axis",
          V[0] < 0 and V[1] < 0 and abs(np.linalg.norm(V) - 1) < 1e-9, V)
    e1 = np.array([1.0, 0, 0]) - V[0] * V; e1 /= np.linalg.norm(e1)
    e2 = np.cross(V, e1)
    worst = 0.0
    for Lt in truth:
        Lt = np.array(Lt)
        N = (Lt + V) / np.linalg.norm(Lt + V)          # the ball's normal that mirrors L into V
        hx, hy = s.cx + s.r * float(N @ e1), s.cy - s.r * float(N @ e2)
        got = light_from_highlight(s, hx, hy, V)
        worst = max(worst, math.degrees(math.acos(min(1.0, float(got @ Lt)))))
    check("seen obliquely, the perspective formula recovers each light exactly", worst < 1e-6, worst)
    check("the condition number of the light set is reported",
          1.0 <= condition_number(L) < 10, condition_number(L))

    print("\nThe real test capture")
    import glob
    real = sorted(glob.glob(os.path.join(test_images() or "", "*.jpeg")))
    if len(real) >= 3:
        h, w = cv2.imread(real[0], cv2.IMREAD_GRAYSCALE).shape
        rc = detect_sphere_candidates(real, h, w)
        check("the ball's radius is its silhouette's, about 24.5 px (was 31.7)",
              rc and abs(rc[0][2] - 24.5) < 1.5, rc)
        RL = auto_detect_sphere(real)
        el = [math.degrees(math.asin(v[2])) for v in RL] if RL is not None else []
        check("eight lights, 20-55° up", len(el) == 8 and all(20 < e < 55 for e in el),
              [round(e, 1) for e in el])
        from helio.core.sphere import refine
        fit = refine(real, rc[0][0], rc[0][1], rc[0][2])
        check("the rim fit is tight (rms under 1 px, most rays used)",
              fit.rms < 1.0 and fit.inliers > fit.rays // 2, fit)
        from helio.application.capture_tools import light_frame_check, swap_north_south
        lf = light_frame_check(real, RL, balls=[rc[0]])
        check("the sphere's lights read the right way up (north at the top)",
              lf["verdict"] == "upright" and lf["ratio"] > 1.15, lf)
        lf2 = light_frame_check(real, swap_north_south(RL), balls=[rc[0]])
        check("…and the same lights with north and south swapped are caught (KD-054)",
              lf2["verdict"] == "upside down", lf2)
        print("\nTwo balls (KD-056)")
        from helio.core.sphere import Sphere, calibrate, with_second_ball
        two = detect_sphere_candidates(real, h, w, want_two=True)
        check("the search finds both balls: 24.5 px at the bottom, 33 px at the top",
              len(two) >= 2 and abs(two[0][2] - 24.5) < 1.5 and abs(two[1][2] - 33.1) < 1.5, two)
        one = calibrate(real, Sphere(*two[0]))
        both = with_second_ball(real, one, Sphere(*two[1]))
        ref = calibrate(real, Sphere(*two[0]), sphere2=Sphere(*two[1]))
        check("both are used: each frame's light is their mean, and the notes say how far they "
              "disagree", both is not None and np.allclose(both.lights, ref.lights)
              and any("Two balls found" in n for n in both.notes), both.notes if both else None)
        check("the classic automatic search gives the same two-ball lights",
              RL is not None and np.allclose(RL, np.asarray(both.lights, np.float32)))
        fake = with_second_ball(real, one, Sphere(150.0, 640.0, 26.0))     # a patch of page
        check("a second circle that is not a ball is left out", fake is None)
    else:
        print("  (test images not present; skipped)")

    print("\nA ball on black velvet (KD-053)")
    from helio.core.sphere import refine, ball_polarity, LOWEST_LIGHT_DEG

    def velvet_ball(Lv, w=360, h=360, cx=181.4, cy=176.7, R=112.0, seed=0):
        """Black cloth; the ball's limb a little brighter than the cloth; inside it the
        dome's reflection, a brighter disc to 0.67 R; the lamp's highlight."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        d = np.hypot(xx - cx, yy - cy)
        img = np.clip(rng.normal(14, 2.5, (h, w)), 0, 255)
        cover = np.clip(R + 0.5 - d, 0, 1)
        inner = np.clip(0.67 * R + 0.5 - d, 0, 1)
        nx, ny = (xx - cx) / R, (cy - yy) / R
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        cosang = 2 * nz * nx * Lv[0] + 2 * nz * ny * Lv[1] + (2 * nz * nz - 1) * Lv[2]
        ball = 26 + 34 * inner + 255 * np.exp(-(1 - cosang) / 0.004)
        return np.clip(img * (1 - cover) + ball * cover, 0, 255).astype(np.uint8)
    vl = [light(az, 40) for az in range(0, 360, 45)]
    vpaths = []
    for i, Lv in enumerate(vl):
        pth = os.path.join(tmp, f"v{i:02d}.png")
        cv2.imwrite(pth, velvet_ball(Lv, seed=i))
        vpaths.append(pth)
    med = np.median([cv2.imread(q, cv2.IMREAD_GRAYSCALE) for q in vpaths], axis=0)
    check("the rim steps bright-to-dark on velvet (polarity -1), dark-to-bright on foam (+1)",
          ball_polarity(med, 181.4, 176.7, 110) == -1
          and ball_polarity(cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE), 470.3, 330.6, 26.4) == 1)
    fit = refine(vpaths, 181.4, 176.7, 104.0, drawn=True)
    check("a circle drawn near the ball is fitted to its rim, not the reflection inside it",
          abs(fit.r - 112.0) / 112.0 < 0.03, fit)
    fit2 = refine(vpaths, 181.4, 176.7, 80.0)
    check("a rough circle never settles on the inner reflection (0.67 R)",
          fit2.r > 0.85 * 112.0 or fit2.note, fit2)
    fx = os.path.join(ROOT, "tests", "fixtures", "chrome_ball_velvet")
    real_v = sorted(glob.glob(os.path.join(fx, "*.jpg"))) if os.path.isdir(fx) else []
    if real_v:
        kept = refine(real_v, 166, 165, 110, drawn=True)
        check("the real velvet ball: its faint rim is not replaced by the inner ring (was 75 px)",
              kept.r > 95, kept)
        bad = refine(real_v, 166, 165, 80)
        check("…and a fit that would put a light near the horizon is refused, with the reason",
              bad.r == 80 and ("horizon" in bad.note or "could not be seen" in bad.note), bad)
        cal = calibrate(real_v, Sphere(166, 165, 110))
        el = [a[1] for a in cal.angles() if a]
        check("lights from the drawn circle sit 30-50° up, as the dome's",
              len(el) == 8 and all(30 < e < 50 for e in el), [round(e, 1) for e in el])
        low = calibrate(real_v, Sphere(166.05, 165.6, 74.97))
        check(f"a circle drawn too small is flagged: lights under {LOWEST_LIGHT_DEG:.0f}° are named",
              any("horizon" in n for n in low.notes), low.notes)

    print("\nA ball on a light card beside gilding")
    # Gilt tooling on dark leather is a field of small specks, each lit or not
    # by each lamp; the ball sits on a light grey card.  The old search took
    # the specks (dark surroundings) and refused the ball (light ones).
    rng = np.random.default_rng(7)
    W, H, bx, by, bR = 1600, 1200, 1390.0, 905.0, 34.0
    specks = np.column_stack([rng.uniform(60, 1050, 500), rng.uniform(60, 1140, 500)])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    gpaths = []
    for i, Lg in enumerate(light(az, 35) for az in range(0, 360, 45)):
        img = np.full((H, W), 22.0)
        img[650:1160, 1220:1560] = 150.0                          # the grey card
        for (sx, sy) in specks[rng.random(len(specks)) < 0.6]:
            img[int(sy) - 1:int(sy) + 2, int(sx) - 1:int(sx) + 2] = 255.0
        d = np.hypot(xx - bx, yy - by)
        nx, ny = (xx - bx) / bR, (by - yy) / bR
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        cosang = 2 * nz * nx * Lg[0] + 2 * nz * ny * Lg[1] + (2 * nz * nz - 1) * Lg[2]
        ball = 45 + 255 * np.exp(-(1 - cosang) / 0.004)
        cover = np.clip(bR + 0.5 - d, 0, 1)
        img = img * (1 - cover) + ball * cover
        pth = os.path.join(tmp, f"g{i:02d}.png")
        cv2.imwrite(pth, np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8))
        gpaths.append(pth)
    gc = detect_sphere_candidates(gpaths, H, W)
    check("the ball is found, not the gilding", len(gc) == 1
          and math.hypot(gc[0][0] - bx, gc[0][1] - by) < 3 and abs(gc[0][2] - bR) / bR < 0.08, gc)
    from helio.core.processing import _ball_evidence
    gstack = [cv2.imread(q, cv2.IMREAD_GRAYSCALE) for q in gpaths]
    check("…because only the ball shows one moving highlight per frame",
          _ball_evidence(gstack, bx, by, bR) > 0.5
          and max(_ball_evidence(gstack, float(x), float(y), 30.0) for x, y in specks[:40]) == 0.0)
    from helio.core.sphere import reduction_for
    check("a 100 MP frame is read at 1/8 for the search, a small one whole",
          reduction_for((8585, 11444), 1000) == 8 and reduction_for((960, 1280), 1000) == 1)

    print("\nPhoto tone")
    from helio.domain.settings import ProcessingSettings
    from helio.core.radiometry import DEFAULT_TONE
    check("the default is automatic: decoded by the file's curve (DiLiGenT-Pi, 2026-11-12)",
          ProcessingSettings().photo_tone == "auto" and DEFAULT_TONE == "auto"
          and ProcessingSettings().resolved_photo_tone() == "srgb")

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
