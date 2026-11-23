"""The corrections of the November 2026 maths audit, each against a known answer.

* ``poisson_free_boundary`` gives a tilted, bumpy surface back exactly (the
  mirrored-edge integrators lose the tilt), and the least-squares methods that
  start from it still do;
* Multi-Scale counts each scale once (a mid-sized bump came out 10 % too tall);
* the height scale (``true_shape_scale``) is not pushed up by grain or noise;
* ISO 25178 Str, Sdq and Sdr do not change when the region is tilted;
* a slanted profile's distance axis ends at the line's true length;
* the ISO 16610-21 Gaussian filter passes half of a sine at its cut-off, keeps a
  straight line to its ends, and separates a known roughness from a known waviness;
* mean curvature of a paraboloid's summit is its true curvature.

    python tests/test_audit_maths.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"  {detail}"))


def scene(h, w, tilt=0.0, ripple=0.0):
    """(heights, +Y-up normals) of two Gaussian bumps, a tilt and a 6 px ripple."""
    from helio.core.frames import normals_from_slopes
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    g1 = 12 * np.exp(-((xx - 150) ** 2 + (yy - 110) ** 2) / (2 * 35 ** 2))
    g2 = -6 * np.exp(-((xx - 230) ** 2 + (yy - 170) ** 2) / (2 * 18 ** 2))
    k = 2 * np.pi / 6.0
    z = g1 + g2 + tilt * (xx - w / 2) + ripple * np.sin(k * xx)
    gx = g1 * (-(xx - 150) / 35 ** 2) + g2 * (-(xx - 230) / 18 ** 2) + tilt + ripple * k * np.cos(k * xx)
    gy = g1 * (-(yy - 110) / 35 ** 2) + g2 * (-(yy - 170) / 18 ** 2)
    n = normals_from_slopes(gx.astype(np.float32), gy.astype(np.float32), np.float32(1.0))
    return z, n.astype(np.float32)


def rms(d, z):
    e = (np.asarray(d, np.float64) - np.mean(d)) - (z - z.mean())
    return float(np.sqrt((e ** 2).mean()))


def main():
    import cv2
    from helio.application.pipeline import integrate_depth
    from helio.core import metrology as M
    from helio.core.depth_integration import poisson_free_boundary, poisson_solver
    from helio.core.frames import normal_slopes, normals_from_slopes
    h, w = 240, 320

    # ── integration ───────────────────────────────────────────────────
    z, n = scene(h, w, tilt=0.05)
    free = poisson_free_boundary(*normal_slopes(n))
    mirrored = poisson_solver(*normal_slopes(n))
    check("free-boundary DCT: a tilted surface comes back to 0.01 px", rms(free, z) < 0.01, rms(free, z))
    check("…where the mirrored-edge Poisson loses the tilt (the reason it exists)", rms(mirrored, z) > 2.0,
          rms(mirrored, z))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    plane = poisson_free_boundary(np.full((h, w), 0.2, np.float32), np.full((h, w), -0.1, np.float32))
    check("free-boundary DCT: constant slopes give their plane exactly",
          np.abs(plane - (0.2 * xx - 0.1 * yy) + (0.2 * xx - 0.1 * yy).mean()).max() < 1e-3)
    for m in ("Least-Squares (Free Boundary)", "Bilateral (Keeps Steps)"):
        d = integrate_depth(n, m)
        check(f"{m}: still exact, started from it", rms(d, z) < 0.01, rms(d, z))
    d = integrate_depth(n, "Free-Boundary Poisson (DCT)")
    from helio.domain.registry import DEPTH_BLURBS
    check("Free-Boundary Poisson (DCT) is offered as a depth method, and exact here",
          "Free-Boundary Poisson (DCT)" in DEPTH_BLURBS and rms(d, z) < 0.01, rms(d, z))
    from helio.core import bini
    zb, nb = scene(h, w, tilt=0.05)
    big = bini.integrate(nb, bilateral=False, max_px=h * w // 4)       # the large-map path
    check("Least-Squares on a map above its size limit: detail from the free-boundary solve",
          rms(big, zb) < 0.1, rms(big, zb))

    z0, n0 = scene(h, w)
    d = integrate_depth(n0, "Multi-Scale Integration").astype(np.float64)
    check("Multi-Scale: each scale counted once (range within 1 %)",
          abs(np.ptp(d) / np.ptp(z0) - 1) < 0.01, np.ptp(d) / np.ptp(z0))

    # ── the height scale ───────────────────────────────────────────────
    zr, nr = scene(h, w, ripple=0.5)
    dr = integrate_depth(nr, "Least-Squares (Free Boundary)")
    s = M.true_shape_scale(dr, nr)
    check("height scale on a rippled surface is 1 (was 1.34)", abs(s - 1) < 0.05, s)
    rng = np.random.default_rng(0)
    nn = n0 + rng.normal(0, 0.1, n0.shape).astype(np.float32)
    nn /= np.linalg.norm(nn, axis=-1, keepdims=True)
    s = M.true_shape_scale(integrate_depth(nn, "Frankot-Chellappa"), nn)
    check("height scale with 6° of noise in the normals is 1 (was 1.16)", abs(s - 1) < 0.02, s)
    s = M.true_shape_scale(integrate_depth(n0, "Directional Integration"), n0)
    check("height scale still brings a 0-1 depth to pixels", abs(s - np.ptp(z0)) / np.ptp(z0) < 0.05, s)

    # ── areal parameters ───────────────────────────────────────────────
    def iso_surface(hh, ww, sigma):
        a = rng.normal(size=(hh + 200, ww + 200)).astype(np.float32)
        return cv2.GaussianBlur(a, (0, 0), sigma)[100:-100, 100:-100].astype(np.float64)
    zi = iso_surface(300, 300, 6.0)
    xi = np.mgrid[0:300, 0:300][1]
    s0, _ = M.texture_aspect_ratio(zi)
    s1, _ = M.texture_aspect_ratio(zi + 0.02 * xi)
    check("Str of an isotropic surface is high, and the same when tilted", s0 > 0.75 and abs(s0 - s1) < 1e-6,
          (s0, s1))
    za = cv2.GaussianBlur(rng.normal(size=(500, 500)).astype(np.float32), (0, 0), sigmaX=12,
                          sigmaY=3)[100:-100, 100:-100].astype(np.float64)
    sa, da = M.texture_aspect_ratio(za)
    check("Str of a surface 4 times longer one way is near 1/4, Std along it", 0.15 < sa < 0.35 and da == 0.0,
          (sa, da))
    zs = iso_surface(300, 300, 4.0) * 5
    gy, gx = np.gradient(zs)
    true_sdq = float(np.sqrt((gx ** 2 + gy ** 2).mean()))
    a0 = M.areal(zs, normals_from_slopes(gx, gy, 1.0))
    a1 = M.areal(zs + 0.1 * xi, normals_from_slopes(gx + 0.1, gy, 1.0))
    check("Sdq is the surface's own gradient", abs(a0["Sdq"] - true_sdq) < 0.002, (a0["Sdq"], true_sdq))
    check("Sdq and Sdr do not change when the region is tilted",
          abs(a1["Sdq"] - a0["Sdq"]) < 0.002 and abs(a1["Sdr"] - a0["Sdr"]) < 0.01,
          (a0["Sdq"], a1["Sdq"], a0["Sdr"], a1["Sdr"]))

    # ── profiles ──────────────────────────────────────────────────────
    yy, xx = np.mgrid[0:400, 0:400].astype(float)
    zz = np.sin(2 * np.pi * (xx + yy) / np.sqrt(2) / 20.0)
    zp, ends, num = M.profile_line(zz, 50, 50, 350, 350)
    length = math.hypot(ends[2] - ends[0], ends[3] - ends[1])
    st = M.profile_stats(zp, num, 0.1, None, length_px=length)
    check("a slanted profile's distance axis ends at the line's length",
          abs(st["x_axis"][-1] - length * 0.1) < 1e-9, (st["x_axis"][-1], length * 0.1))
    check("…and without the length it is one sample a pixel, as before",
          M.profile_stats(zp, num)["x_axis"][-1] == num - 1)
    check("the plain figures are unchanged: Ra of a sine is 2/pi of its amplitude",
          abs(st["Ra"] - 2 / np.pi) < 0.01 and st["iso"] is not None, st["Ra"])

    dx = 0.01
    x = np.arange(0, 8.0001, dx)
    for lam in (0.4, 0.8, 1.6):
        wv = M.gaussian_mean_line(np.sin(2 * np.pi * x / lam), dx, 0.8)
        got = float(np.sqrt(2) * wv[200:-200].std())
        want = math.exp(-math.pi * (M.GAUSS_ALPHA * 0.8 / lam) ** 2)
        check(f"Gaussian filter, cut-off 0.8: a {lam} wave passes {want:.3f}", abs(got - want) < 0.005, got)
    line = 0.3 * x + 1
    check("Gaussian mean line of a straight slope is the slope, to its ends",
          np.abs(M.gaussian_mean_line(line, dx, 0.8) - line).max() < 1e-9)
    prof = 1.0 * np.sin(2 * np.pi * x / 0.1) + 10 * np.sin(2 * np.pi * x / 5.0 + 0.3) + 3.0 * x
    r = M.roughness_waviness(prof, dx, 0.8)
    check("roughness of a 1-high fine sine under a 10-high wave and a tilt: Ra 2/pi, Rz 2",
          abs(r["Ra"] - 2 / np.pi) < 0.02 and abs(r["Rz"] - 2.0) < 0.1 and r["Rt"] >= r["Rz"],
          (r["Ra"], r["Rz"], r["Rt"]))
    check("…and the wave is the waviness", 5.0 < r["Wa"] < 7.0 and r["n_lengths"] == 10, (r["Wa"], r["n_lengths"]))
    check("cut-off chosen by ISO 4288: 0.8 mm for Ra 0.6 µm on 8 mm; 0.25 on 2 mm; a fifth of a short line",
          M.default_cutoff(8.0, 0.6) == (0.8, True) and M.default_cutoff(2.0, 0.6) == (0.25, True)
          and M.default_cutoff(0.2, 0.6)[1] is False and M.default_cutoff(500, None, unit_mm=False) == (100.0, False))
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "p.csv")
    full = M.profile_stats(zp, num, 0.1, 0.01, length_px=length)
    M.profile_csv(path, full["x_axis"], full["z_plot"], full["label"], full["x_label"], full["z_unit"],
                  (full["Ra"], full["Rq"], full["Rz"], full["Rp"], full["Rv"]), iso=full["iso"])
    text = open(path).read()
    check("the profile CSV carries the filtered figures and both profiles",
          "ISO 16610-21" in text and "Waviness(" in text and "Roughness(" in text)

    # ── curvature ─────────────────────────────────────────────────────
    from helio.core import processing as P
    yy, xx = np.mgrid[-40:41, -40:41].astype(np.float32)
    bowl = (0.001 * (xx ** 2 + yy ** 2)).astype(np.float32)      # summit curvature 0.002 each way
    captured = {}
    orig = P._rescale_1_99
    P._rescale_1_99 = lambda a: captured.setdefault("H", a)
    try:
        P.surface_curvature_mean(bowl, z_factor=1.0 / 8.0, smooth_sigma=0.5)
    finally:
        P._rescale_1_99 = orig
    check("mean curvature at a paraboloid's summit is its curvature (was 0)",
          abs(float(captured["H"][40, 40]) - 0.002) < 2e-4, float(captured["H"][40, 40]))

    print(f"\n{'=' * 60}\nResults: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS) + len(FAIL)} tests\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
