"""The memory-lean depth solvers against the originals they replaced.

* poisson_solver (a DCT on the image itself) vs the padded-FFT solver: the same
  answer, and nearer a float64 reference than the old one is;
* frankot_chellappa (half spectrum, in place) vs the full-grid version, odd
  and even sizes (the Nyquist row / column);
* simchony_depth_integration vs the full-denominator version;
* remove_poly22 vs the (h*w, 6) least-squares matrix it replaced;
* peak memory (tracemalloc) of each, as bytes per pixel.

    python tests/test_depth_memory.py
"""
import os
import sys
import tracemalloc

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from helio.core import depth_integration as D                      # noqa: E402
from helio.core.frames import normal_slopes                         # noqa: E402
from tools import make_golden as G                                       # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def slopes(h, w, noise=0.03, seed=0, dtype=np.float32):
    G.H, G.W = h, w
    _, n, _ = G.analytic_surface()
    n = n + np.random.default_rng(seed).normal(size=n.shape).astype(np.float32) * noise
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    gx, gy = normal_slopes(n)
    return gx.astype(dtype), gy.astype(dtype), n


def old_simchony(normals):
    from scipy.fft import dctn, idctn
    nx_s, ny_s = normal_slopes(normals)
    h, w = nx_s.shape
    d = dctn(np.gradient(nx_s, axis=1) + np.gradient(ny_s, axis=0), type=2, norm="ortho")
    cx, cy = np.cos(np.pi * np.arange(w) / w), np.cos(np.pi * np.arange(h) / h)
    denom = 2. * (np.add.outer(cy, cx) - 2.)
    denom[0, 0] = 1.
    z = d / denom
    z[0, 0] = 0.
    return idctn(z, type=2, norm="ortho").astype(np.float32)


def old_poly22(depth_map, edge=lambda n: max(1, n - 1)):
    h, w = depth_map.shape
    yy, xx = np.mgrid[0:h, 0:w]
    xn, yn = (xx / edge(w)) * 2 - 1, (yy / edge(h)) * 2 - 1
    A = np.column_stack([xn.ravel() ** 2, yn.ravel() ** 2, (xn * yn).ravel(), xn.ravel(), yn.ravel(), np.ones(h * w)])
    c = np.linalg.lstsq(A, depth_map.ravel(), rcond=None)[0]
    return depth_map - (A @ c).reshape(h, w)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(float(np.ptp(b)), 1e-12))


def main():
    sizes = ((240, 320), (601, 799), (600, 800), (601, 800), (600, 799), (1500, 2000), (9, 11), (8, 8))
    print("Poisson")
    for h, w in sizes:
        gx, gy, _ = slopes(h, w)
        gx64, gy64 = gx.astype(np.float64), gy.astype(np.float64)
        truth = D._poisson_solver_fft(gx64, gy64).astype(np.float64)
        new, old = D.poisson_solver(gx, gy), D._poisson_solver_fft(gx, gy)
        check(f"{h}x{w}: agrees with the padded-FFT solver ({rel(new, old):.1e} of the range) and is "
              f"at least as near a float64 reference", rel(new, old) < 5e-6 and rel(new, truth) <= rel(old, truth) * 1.5 + 1e-9,
              (rel(new, old), rel(new, truth), rel(old, truth)))
        # The depth is returned as float32 (eps 1.2e-7), so a few 1e-9 of the range is below one unit
        # in its last place; FFT implementations (x86, Apple silicon) differ at that level.
        check(f"{h}x{w}: float64 slopes give the same depth to {rel(D.poisson_solver(gx64, gy64), truth):.0e}",
              rel(D.poisson_solver(gx64, gy64), truth) < 1e-8)
    check("returns float32 (h, w)", D.poisson_solver(*slopes(50, 60)[:2]).dtype == np.float32
          and D.poisson_solver(*slopes(50, 60)[:2]).shape == (50, 60))

    print("Frankot-Chellappa")
    for h, w in sizes[:7]:
        gx, gy, _ = slopes(h, w)
        gx64, gy64 = gx.astype(np.float64), gy.astype(np.float64)
        truth = D._frankot_chellappa_grid(gx64, gy64).astype(np.float64)
        check(f"{h}x{w}: matches the full-grid version (float64 slopes: {rel(D.frankot_chellappa(gx64, gy64), truth):.0e}; "
              f"float32: {rel(D.frankot_chellappa(gx, gy), truth):.0e})",
              rel(D.frankot_chellappa(gx64, gy64), truth) < 1e-9
              and rel(D.frankot_chellappa(gx, gy), truth) < 5 * max(rel(D._frankot_chellappa_grid(gx, gy), truth), 1e-7))

    print("Simchony DCT")
    for h, w in sizes[:6]:
        n = slopes(h, w)[2]
        check(f"{h}x{w}: matches the full-denominator version", rel(D.simchony_depth_integration(n), old_simchony(n)) < 1e-6)

    print("Polynomial baseline")
    r = np.random.default_rng(4)
    for h, w, dt in ((300, 400, np.float32), (301, 257, np.float64), (1000, 1500, np.float32), (5, 700, np.float32)):
        z = (r.random((h, w)) * 3 + 0.002 * np.add.outer(np.arange(h) ** 1.2, np.arange(w))).astype(dt)
        new = D.remove_poly22(z)
        check(f"{h}x{w} {np.dtype(dt).name}: same residual as the (h*w, 6) fit ({rel(new, old_poly22(z)):.1e}), same dtype",
              rel(new, old_poly22(z)) < 5e-6 and new.dtype == dt)
    from helio.application.pipeline import parabolic_baseline_correction
    from helio.core.processing import subtract_parabolic_baseline
    one = r.random((1, 50)).astype(np.float32)
    check("a one-row map is returned unchanged by the pipeline version", np.array_equal(parabolic_baseline_correction(one), one))
    check("a one-row map does not fail the processing version", subtract_parabolic_baseline(one).shape == (1, 50))
    flat = r.random((200, 300)).astype(np.float32)
    check("both public functions use the strip fit", np.allclose(parabolic_baseline_correction(flat), subtract_parabolic_baseline(flat), atol=1e-5))

    print("Peak memory (tracemalloc), bytes per pixel")
    h, w = 1500, 2000
    gx, gy, n = slopes(h, w)
    z = (r.random((h, w)) * 3).astype(np.float32)
    px = h * w

    def peak(fn):
        tracemalloc.start()
        fn()
        _, p = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return p / px

    for name, fn, old, limit in (
            ("poisson_solver", lambda: D.poisson_solver(gx, gy), lambda: D._poisson_solver_fft(gx, gy), 40),
            ("frankot_chellappa", lambda: D.frankot_chellappa(gx, gy), lambda: D._frankot_chellappa_grid(gx, gy), 50)):
        pn, po = peak(fn), peak(old)
        check(f"{name}: {pn:.0f} B/px (was {po:.0f}), under {limit}", pn < limit and pn < po, (pn, po))
    pn, po = peak(lambda: D.remove_poly22(z)) * px / 1e6, peak(lambda: old_poly22(z)) * px / 1e6
    check(f"remove_poly22: {pn:.0f} MB for a {px / 1e6:.0f} MP map whatever its size (was {po:.0f} MB, "
          f"growing with the image), under 200 MB", pn < 200 and pn < po, (pn, po))

    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
