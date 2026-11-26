"""Flat objects: a binding's low relief solved through the pipeline, with the truth known.

DiLiGenT (tools/benchmark_diligent.py) measures the solvers on curved objects under
distant lamps.  Most of what Helio is pointed at is flat - bindings, leaves,
prints - and on a flat object what goes wrong is the broad shape, not the
tooling: lamps close to the object bend it into a dish, unequal lamps tilt
it.  This renders a flat binding (border fillet, a tooled groove, stamps,
spine bands, grain) as a camera records it (8-bit sRGB) under such lamps,
solves it as the app does, and checks:

* the tooling comes back to within 2 degrees in every case;
* distant, equal lamps: flat, whichever solver (they agree on shallow relief);
* lamps close (1x-5x the frame's width): uncorrected, many degrees out;
  the lamp distance is found from the photographs (to 10%), and the
  near-field correction with it brings the binding back - also with the
  lights measured on a ball in the corner;
* distant lamps: found far away, and the correction changes nothing;
* unequal lamps: flat areas level them;
* a hand-held flash, near and uneven: the flat areas can be read (a plane
  in the logarithm), and with near-field they share one model of the lamps.

    python tests/test_flat_objects.py
"""
import dataclasses
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


H, W = 240, 320
MARGINS = "2,2,16,238;304,2,318,238"            # flat strips either side: no relief there


def binding():
    import cv2
    import numpy as np
    rng = np.random.default_rng(7)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    z = np.zeros((H, W))

    def ridge(x0, y0, x1, y1, width, height):
        d = np.minimum.reduce([np.abs(xx - x0), np.abs(xx - x1), np.abs(yy - y0), np.abs(yy - y1)])
        inside = (xx > x0 - width) & (xx < x1 + width) & (yy > y0 - width) & (yy < y1 + width)
        return np.where(inside, height * np.exp(-(d / width) ** 2), 0.0)
    z += ridge(47, 33, 273, 207, 4, 2.0)
    z -= ridge(63, 50, 257, 190, 1.7, 0.8)
    for cx in np.linspace(87, 233, 5):
        for cy in np.linspace(73, 167, 3):
            d2 = (xx - cx) ** 2 + (yy - cy) ** 2
            z += 1.1 * np.exp(-d2 / 36) - 0.5 * np.exp(-d2 / 6)
    for by in (60, 100, 140, 180):
        z += 2.7 * np.exp(-((yy - by) / 5) ** 2) * np.exp(-((xx - 27) / 7) ** 2)
    z += rng.normal(0, 0.02, z.shape)
    z = cv2.GaussianBlur(z, (0, 0), 0.8)
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(gx)])
    return z, n / np.linalg.norm(n, axis=-1, keepdims=True)


def lamps(k=12):
    import numpy as np
    out = []
    for i in range(k):
        az, el = 2 * np.pi * i / k + 0.2, np.radians(25 + 15 * (i % 3))
        out.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.array(out)


def render(z, n, L, dist=None, strength=None, origin=(0.0, 0.0)):
    import numpy as np
    strength = np.ones(len(L)) if strength is None else strength
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    X = np.dstack([xx - (W - 1) / 2, (H - 1) / 2 - yy, z])
    O = np.array([origin[0], origin[1], 0.0])
    out = []
    for i, lv in enumerate(L):
        if dist is None:
            d, fall = np.broadcast_to(lv, (H, W, 3)), 1.0
        else:
            P = O + lv * (dist[i] if np.ndim(dist) else dist)
            v = P - X
            r = np.linalg.norm(v, axis=-1, keepdims=True)
            d, fall = v / r, (np.linalg.norm(P - O) / r[..., 0]) ** 2
        out.append(0.55 * np.clip((n * d).sum(-1), 0, None) * fall * strength[i])
    return np.array(out)


def write(frames):
    import cv2
    import numpy as np
    from helio.core.radiometry import linear_to_srgb
    tmp = tempfile.mkdtemp(prefix="helio_flat_")
    scale = 0.9 / np.percentile(frames, 99.7)
    paths = []
    for i, f in enumerate(frames):
        enc = np.round(linear_to_srgb(np.clip(f * scale, 0, 1) * 255.0)).astype(np.uint8)
        p = os.path.join(tmp, f"f{i:02d}.png")
        cv2.imwrite(p, np.dstack([enc] * 3))
        paths.append(p)
    return paths


def main():
    import cv2
    import numpy as np
    from helio.application.pipeline import SolveRequest, solve
    from helio.domain.settings import ProcessingSettings
    print("=" * 60)
    print("Flat objects: a binding's low relief, the truth known")
    print("=" * 60)
    z, n = binding()
    L = lamps()
    k = len(L)
    inner = np.zeros((H, W), bool)
    inner[14:-14, 14:-14] = True

    def ang(a, b):
        return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))

    def hp(m):          # the tooling: each normal against its neighbourhood's
        b = np.dstack([cv2.GaussianBlur(m[..., c].astype(np.float64), (0, 0), 15) for c in range(3)])
        b /= np.linalg.norm(b, axis=-1, keepdims=True)
        p, q = -m[..., 0] / m[..., 2], -m[..., 1] / m[..., 2]
        pb, qb = -b[..., 0] / b[..., 2], -b[..., 1] / b[..., 2]
        g = np.dstack([-(p - pb), -(q - qb), np.ones_like(p)])
        return g / np.linalg.norm(g, axis=-1, keepdims=True)

    def go(frames, lights=L, origin=None, **kw):
        r = solve(SolveRequest(paths=write(frames), lights=lights.astype(np.float32),
                               settings=ProcessingSettings(photo_tone="srgb", **kw), light_origin=origin))
        N = np.asarray(r.normals_full, np.float64)
        m, mt = N[inner].mean(0), n[inner].mean(0)
        lean = float(np.degrees(np.arccos(np.clip(m @ mt / np.linalg.norm(m) / np.linalg.norm(mt), -1, 1))))
        return float(ang(N, n)[inner].mean()), float(ang(hp(N), hp(n))[inner].mean()), lean, r

    details = []

    print("\nDistant, equal lamps")
    f = render(z, n, L)
    errs = {}
    for eng in ("Linear Least Squares", "Shadow-Rejection LS", "Sparse-Outlier LS (RPCA)"):
        e, d, _lean, r = go(f, normal_engine=eng)
        errs[eng] = e
        details.append(d)
    check(f"every solver flat to within 2° ({', '.join(f'{v:.2f}' for v in errs.values())})",
          max(errs.values()) < 2.0, errs)
    check("…and they agree on shallow relief", max(errs.values()) - min(errs.values()) < 0.3, errs)
    check("the photographs show the lamps far away", r.lamp_distance and r.lamp_distance["far_away"],
          r.lamp_distance)
    e_on, d, _l, r = go(f, near_field=True)
    check(f"…so near-field correction changes nothing ({e_on:.2f}°)", e_on < 2.0 and
          r.lamp_distance["source"] == "far", (e_on, r.lamp_distance))

    print("\nLamps close to the object")
    for ratio in (1.0, 1.5, 3.0, 5.0):
        f = render(z, n, L, dist=ratio * W)
        e_off, d0, _l, r = go(f)
        e_nf, d1, _l, r2 = go(f, near_field=True)
        found = r.lamp_distance["ratio"]
        details += [d0, d1]
        check(f"{ratio}x away: uncorrected {e_off:.1f}° out; found {found:.2f}x; "
              f"near-field {e_nf:.2f}°", e_off > 4.0 and abs(found / ratio - 1) < 0.1
              and e_nf < 2.0 and r2.lamp_distance["source"] == "photographs",
              (e_off, found, e_nf, r2.lamp_distance))
    e_old, _d, _l, _r = go(render(z, n, L, dist=3.0 * W), near_field=True, near_field_distance="radius")
    check(f"…the old fixed guess (1.5x) at 3x: {e_old:.1f}° out", e_old > 4.0, e_old)
    bx, by = W - 20, H - 20                         # a ball in the corner: the lights measured there
    f = render(z, n, L, dist=1.6 * W, origin=(bx - (W - 1) / 2, (H - 1) / 2 - by))
    e_off, _d, _l, _r = go(f, origin=(bx, by))
    e_nf, d, _l, r = go(f, origin=(bx, by), near_field=True)
    check(f"lights measured on a ball in the corner: {e_off:.1f}° uncorrected, {e_nf:.2f}° corrected",
          e_off > 10 and e_nf < 2.5, (e_off, e_nf, r.lamp_distance))

    print("\nUnequal lamps and a hand-held flash")
    st = np.random.default_rng(11).normal(1, 0.2, k).clip(0.6, 1.5)
    f = render(z, n, L, strength=st)
    e_off, _d, lean_off, _r = go(f)
    e_fa, d, lean_fa, r = go(f, even_flat=True, flat_areas=MARGINS)
    details.append(d)
    check(f"lamps ±20%: leaning {lean_off:.1f}°; flat areas level it ({lean_fa:.2f}°)",
          lean_off > 1.5 and lean_fa < 0.3 and e_fa < 2.0, (lean_off, lean_fa, e_fa))
    f = render(z, n, L, dist=1.5 * W, strength=st)
    e_both, d, _l, r = go(f, near_field=True, even_flat=True, flat_areas=MARGINS)
    details.append(d)
    check(f"close and unequal: near-field and flat areas together {e_both:.2f}°", e_both < 2.0, e_both)
    dist = 1.2 * W * np.random.default_rng(5).normal(1, 0.15, k).clip(0.75, 1.3)
    f = render(z, n, L, dist=dist, strength=st)
    e_off, _d, _l, _r = go(f)
    e_fa, _d, _l, r_fa = go(f, even_flat=True, flat_areas=MARGINS)
    e_both, d, _l, _r = go(f, near_field=True, even_flat=True, flat_areas=MARGINS)
    details.append(d)
    check("hand-held flash: the flat areas can be read (they could not: a plane went through zero)",
          r_fa.flat_light is not None, r_fa.flat_light)
    check(f"…uncorrected {e_off:.1f}°, flat areas {e_fa:.1f}°, with near-field {e_both:.1f}°",
          e_both < 5.0 and e_both < e_fa < e_off, (e_off, e_fa, e_both))

    print("\nThe tooling")
    check(f"comes back to within 2° in every case (worst {max(details):.2f}°)", max(details) < 2.0, details)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
