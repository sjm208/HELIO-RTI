"""Every depth integrator against surfaces whose heights are known, and the scale tool's length.

Exact slopes of analytic surfaces go in; the heights that come out must have the
right scale (a least-squares fit z_out = a z + b) and the right shape:

* Frankot-Chellappa gives heights in pixels (it gave 2 pi times too tall: the 2 pi
  of d/dx = j 2 pi u was missing);
* Multi-Scale keeps the broad shape in proportion to the detail (its quarter-size
  solve was left in quarter-size pixels, 4x too low);
* Least-Squares, Fast Poisson and Simchony give heights in pixels;
* the scale line is measured to the fraction of a pixel drawn;
* the near-field solver recovers point-lamp normals.

    python tests/test_depth_scale.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"  {detail}"))


def fit(d, z):
    A = np.stack([z.ravel(), np.ones(z.size)], 1)
    (a, b), *_ = np.linalg.lstsq(A, np.asarray(d, np.float64).ravel(), rcond=None)
    res = np.asarray(d, np.float64) - (a * z + b)
    return a, float(np.sqrt((res ** 2).mean()) / (z.max() - z.min()))


def main():
    from helio.application.pipeline import integrate_depth
    from helio.core.frames import normals_from_slopes
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    def gauss(cx, cy, s, A):
        g = A * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))
        return g, g * -(xx - cx) / s ** 2, g * -(yy - cy) / s ** 2

    z, gx, gy = gauss(160, 120, 30, 20.0)                  # a bump 20 px tall
    n = normals_from_slopes(gx, gy, 1.0).astype(np.float32)
    print("Heights in pixels: a Gaussian bump 20 px tall, from its exact normals")
    for eng in ("Frankot-Chellappa", "Fast Poisson (FFT)", "Simchony DCT", "Least-Squares (Free Boundary)"):
        a, r = fit(integrate_depth(n, eng), z)
        check(f"{eng}: scale 1 (got {a:.3f}), shape within 0.1 % ({100 * r:.3f} %)",
              abs(a - 1) < 0.01 and r < 1e-3, (a, r))

    print("Multi-Scale keeps broad shape and detail in proportion")
    zb, gxb, gyb = gauss(160, 120, 45, 12.0)               # broad
    zf, gxf, gyf = gauss(110, 90, 4, 1.5)                  # fine
    zt = zb + zf
    nt = normals_from_slopes(gxb + gxf, gyb + gyf, 1.0).astype(np.float32)
    d = np.asarray(integrate_depth(nt, "Multi-Scale Integration"), np.float64)
    a, r = fit(d, zt)
    check(f"scale near 1 (got {a:.3f}; it was 0.3 before the fix)", 0.85 < a < 1.15, a)
    check(f"shape within 3 % of the range ({100 * r:.2f} %)", r < 0.03, r)

    print("The scale line")
    from helio.core.metrology import line_length_px
    check("measured to the fraction of a pixel drawn",
          abs(line_length_px(10.6, 20.9, 110.4, 20.1) - np.hypot(99.8, 0.8)) < 1e-9,
          line_length_px(10.6, 20.9, 110.4, 20.1))

    print("Near-field: point lamps 300 px away, the ball off-centre")
    from helio.core.photometric_stereo import solve_near_field_normals
    hh, ww = 120, 160
    Y, X = np.mgrid[0:hh, 0:ww].astype(np.float64)
    g = 8 * np.exp(-(((X - 80) ** 2 + (Y - 60) ** 2) / (2 * 20.0 ** 2)))
    nn = normals_from_slopes(g * -(X - 80) / 400, g * -(Y - 60) / 400, 1.0)
    az = np.radians(np.arange(0, 360, 30)); el = np.radians(np.where(np.arange(12) % 2, 50, 30))
    L = np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)], 1)
    R, origin = 300.0, (-40.0, 20.0)
    P = np.stack([(X - (ww - 1) / 2) - origin[0], ((hh - 1) / 2 - Y) - origin[1], np.zeros_like(X)], -1)
    imgs = []
    for lv in L:
        v = lv * R - P
        dd = np.linalg.norm(v, axis=-1, keepdims=True)
        imgs.append(np.maximum((nn * v / dd).sum(-1), 0) * (R / dd[..., 0]) ** 2 * 200)
    out = solve_near_field_normals(np.array(imgs, np.float32), L, hh, ww, dome_radius_px=R, origin=origin)
    err = np.degrees(np.arccos(np.clip((out * nn).sum(-1), -1, 1)))
    check(f"mean error under 0.05° (got {err.mean():.3f}°), worst under 0.1° ({err.max():.3f}°)",
          err.mean() < 0.05 and err.max() < 0.1, (err.mean(), err.max()))

    print(f"\n{'=' * 60}\nResults: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS) + len(FAIL)} tests\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
