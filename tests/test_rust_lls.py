"""helio_core_rs Linear Least Squares against the NumPy reference (Step 3).

Checks, to 1e-5, that the Rust normals match
  * a frozen NumPy result for the analytic surface (tests/fixtures/golden_lls_normals.npz,
    so the reference itself cannot drift unnoticed), and
  * the live NumPy solve (helio.core.photometric_stereo._lls_normals_numpy),
on the analytic surface in float32 and uint8, on real photographs from
tests/fixtures, and on edge cases: zero pixels, rank-deficient lights, a lone
pixel, ragged block sizes, non-contiguous and wrong-typed input, bad shapes.

    python tests/test_rust_lls.py                 verify
    python tests/test_rust_lls.py --update-golden rewrite the frozen NumPy result

Without the compiled module only the NumPy-vs-frozen check runs.
Needs: cd helio_core_rs && maturin develop --release
"""
import glob
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GOLDEN = os.path.join(ROOT, "tests", "fixtures", "golden_lls_normals.npz")
TOL = 1e-5

from helio.core import photometric_stereo as ps          # noqa: E402

try:
    import helio_core_rs as rs                            # noqa: E402
    rs.lls_normals
except (ImportError, AttributeError):
    rs = None

_results = []


def check(name, fn):
    try:
        fn()
        _results.append((name, None))
        print(f"  ok    {name}")
    except Exception as e:                                # noqa: BLE001
        _results.append((name, e))
        print(f"  FAIL  {name}: {e}")
        if not isinstance(e, AssertionError):
            traceback.print_exc()


def rust(L, flat):
    """The Rust path, whatever HELIO_NO_RUST says."""
    return ps._lls_normals_rust(rs, L, flat)


def numpy_ref(L, flat):
    return ps._lls_normals_numpy(L, flat)


def agree(a, b, tol=TOL):
    assert a.shape == b.shape, f"shape {a.shape} vs {b.shape}"
    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    assert d <= tol, f"max |diff| {d:.2e} > {tol:.0e}"
    return d


def ring_lights(n, elev=40.0):
    el = np.radians(elev)
    az = 2 * np.pi * np.arange(n) / n
    return np.stack([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az),
                     np.full(n, np.sin(el))], 1).astype(np.float32)


def analytic_stack(n_lights=8):
    from tools.make_golden import analytic_surface
    z, nrm, alb = analytic_surface()
    L = ring_lights(n_lights)
    grey = alb.astype(np.float32).mean(-1) / 255.0
    stack = np.clip(nrm.reshape(-1, 3) @ L.T, 0, None).T * (0.35 + 0.65 * grey.reshape(1, -1))
    return L, np.ascontiguousarray(stack, np.float32), nrm


def main():
    update = "--update-golden" in sys.argv
    L, I, true_n = analytic_stack()

    print("NumPy reference")
    if update:
        np.savez_compressed(GOLDEN, normals=numpy_ref(L, I).astype(np.float32))
        print(f"  wrote {os.path.relpath(GOLDEN, ROOT)}")
    gold = np.load(GOLDEN)["normals"]

    def numpy_vs_frozen():
        agree(numpy_ref(L, I), gold)
    check("NumPy solve matches the frozen golden result", numpy_vs_frozen)

    def lit_pixels_recover_truth():
        lit = (I > 0.05).sum(0) >= 4                       # enough unshadowed lights
        err = np.abs(gold[lit] - true_n.reshape(-1, 3)[lit]).max()
        assert err < 0.2, f"golden normals are {err:.3f} from the true surface"
    check("golden normals recover the analytic surface where lit", lit_pixels_recover_truth)

    if rs is None:
        print("helio_core_rs not built: Rust checks skipped")
        return finish(skipped=True)

    print("Rust vs NumPy")
    check("analytic surface, float32, vs frozen golden", lambda: agree(rust(L, I), gold))
    check("analytic surface, float32, vs live NumPy", lambda: agree(rust(L, I), numpy_ref(L, I)))

    def u8():
        q = np.round(np.clip(I, 0, 1) * 255).astype(np.uint8)
        agree(rust(L, q), numpy_ref(L, q))
    check("analytic surface, uint8 (strip conversion)", u8)

    def f64():
        agree(rust(L, I.astype(np.float64)), numpy_ref(L, I.astype(np.float64)))
    check("analytic surface, float64 input", f64)

    def public_entry():
        agree(ps.linear_least_squares_normals(L, I), numpy_ref(L, I))
    check("linear_least_squares_normals() entry point", public_entry)

    def real_photos():
        import cv2
        files = sorted(glob.glob(os.path.join(ROOT, "tests", "fixtures", "chrome_ball_velvet", "*.jpg")))
        assert len(files) >= 3, "fixture photographs missing"
        imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in files]
        flat = np.stack([i.reshape(-1) for i in imgs])
        Lr = ring_lights(len(files), 55.0)
        d = agree(rust(Lr, flat), numpy_ref(Lr, flat))
        d2 = agree(rust(Lr, flat.astype(np.float32) / 255.0),
                   numpy_ref(Lr, flat.astype(np.float32) / 255.0))
        print(f"          ({len(files)} photos, {flat.shape[1]} px; max diff {max(d, d2):.1e})")
    check("real photographs from tests/fixtures", real_photos)

    print("Edge cases")
    rng = np.random.default_rng(7)

    def zeros():
        X = np.abs(rng.normal(size=(8, 4097))).astype(np.float32)
        X[:, [0, 1024, 4096]] = 0
        out = rust(ring_lights(8), X)
        assert not out[[0, 1024, 4096]].any(), "zero pixels must give the zero vector"
        agree(out, numpy_ref(ring_lights(8), X))
    check("all-zero pixels give the zero vector", zeros)

    def rank_deficient():
        Ld = ring_lights(6)
        Ld[:, 1] = 0
        Ld /= np.linalg.norm(Ld, axis=1, keepdims=True)
        X = rng.random((6, 3000)).astype(np.float32)
        agree(rust(Ld, X), numpy_ref(Ld, X))
    check("rank-deficient lights (minimum-norm answer)", rank_deficient)

    def tiny():
        for m in (1, 2, 1023, 1024, 1025, 2049):
            X = rng.random((5, m)).astype(np.float32)
            agree(rust(ring_lights(5), X), numpy_ref(ring_lights(5), X))
    check("1 px and block-boundary sizes", tiny)

    def three_lights():
        X = rng.random((3, 999)).astype(np.float32)
        agree(rust(ring_lights(3), X), numpy_ref(ring_lights(3), X))
    check("exactly 3 lights", three_lights)

    def many_lights():
        X = rng.random((64, 5000)).astype(np.float32)
        agree(rust(ring_lights(64), X), numpy_ref(ring_lights(64), X))
    check("64 lights", many_lights)

    def noncontig():
        X = rng.random((8, 6000)).astype(np.float32)[:, ::2]
        assert not X.flags.c_contiguous
        agree(rust(ring_lights(8), X), numpy_ref(ring_lights(8), X))
    check("non-contiguous input", noncontig)

    def pinv_dtype():
        X = rng.random((8, 100)).astype(np.float32)
        P = np.linalg.pinv(ring_lights(8).astype(np.float64))
        out = rs.lls_normals(np.ascontiguousarray(P), X)
        assert out.dtype == np.float32 and out.shape == (100, 3) and out.flags.c_contiguous
    check("returns C-contiguous float32 (n_px, 3)", pinv_dtype)

    def bad_shapes():
        X = rng.random((8, 10)).astype(np.float32)
        for P in (np.zeros((3, 7)), np.zeros((4, 8))):
            try:
                rs.lls_normals(P, X)
            except ValueError:
                continue
            raise AssertionError(f"pinv {P.shape} was accepted")
        try:
            rs.lls_normals(np.zeros((3, 8)), X[:, ::2])
        except ValueError:
            return
        raise AssertionError("a non-contiguous stack was accepted by the raw kernel")
    check("bad shapes and layouts raise ValueError", bad_shapes)

    def threads_env():
        assert rs.threads() >= 1
    check("thread pool is sized", threads_env)

    print("PTM / HSH fits (lstsq_columns)")
    from helio.core import parallel
    from helio.core.processing import compute_ptm_hsh_coefficients
    from helio.core import _native

    def scattered_lights(n, seed=3):
        r = np.random.default_rng(seed)
        az, el = r.uniform(0, 2 * np.pi, n), np.radians(r.uniform(25, 70, n))
        return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)

    def both_ways(fn):
        saved = _native.rs
        try:
            _native.rs = None
            ref = fn()
            _native.rs = rs
            got = fn()
        finally:
            _native.rs = saved
        return ref, got

    def rel(a, b):
        return float(np.abs(a.astype(np.float64) - b).max() / max(float(np.abs(a).max()), 1e-12))

    for n, ldt, bdt in ((8, np.float32, np.float32), (8, np.float32, np.uint8), (8, np.float64, np.float32),
                        (12, np.float32, np.uint16), (5, np.float32, np.float32)):
        def case(n=n, ldt=ldt, bdt=bdt):
            L = scattered_lights(n).astype(ldt)
            B = (np.random.default_rng(n).random((n, 50_003)) * 255).astype(bdt)
            (rp, rh), (gp, gh) = both_ways(lambda: compute_ptm_hsh_coefficients(L, B))
            assert gp.shape == rp.shape == (6, 50_003) and gh.shape == rh.shape == (9, 50_003)
            assert gp.dtype == gh.dtype == np.float32
            assert rel(rp, gp) < 1e-5 and rel(rh, gh) < 1e-5, (rel(rp, gp), rel(rh, gh))
        check(f"PTM and HSH, {n} lights {np.dtype(ldt).name}, stack {np.dtype(bdt).name}", case)

    def strips():
        L = scattered_lights(8).astype(np.float32)
        B = (np.random.default_rng(1).random((8, 40_000)) * 255).astype(np.uint8)
        saved = parallel.STRIP_BYTES
        parallel.STRIP_BYTES = 1 << 16                       # force many strips
        try:
            ref, got = both_ways(lambda: parallel.lstsq_columns(L.astype(np.float64)[:, :3], B))
        finally:
            parallel.STRIP_BYTES = saved
        assert rel(ref, got) < 1e-5
    check("a stack converted strip by strip gives the same coefficients", strips)

    def deficient():
        L = scattered_lights(6)
        L[:, 1] = 0
        A = np.stack([L[:, 0] ** 2, L[:, 2] ** 2, L[:, 0] * L[:, 2], L[:, 0], L[:, 2], np.ones(6)], 1)
        B = (np.random.default_rng(2).random((6, 3000)) * 255).astype(np.float32)
        ref, got = both_ways(lambda: parallel.lstsq_columns(A, B))
        assert rel(ref, got) < 1e-4, rel(ref, got)
    check("rank-deficient system: the same minimum-norm answer", deficient)

    def fallbacks():
        A = scattered_lights(4)
        B64 = np.random.default_rng(5).random((4, 100))
        _native.rs = rs
        try:
            out = parallel.lstsq_columns(A, B64)                 # float64 stack: NumPy decides
        finally:
            _native.rs = rs
        assert out.dtype == np.float64
    check("a float64 stack still goes to NumPy", fallbacks)

    def bad_shapes():
        for P, X in ((np.zeros((6, 7)), np.zeros((8, 4), np.float32)), (np.zeros((0, 8)), np.zeros((8, 4), np.float32))):
            try:
                rs.apply_pinv(P, X)
            except ValueError:
                continue
            raise AssertionError(f"pinv {P.shape} was accepted")
    check("apply_pinv rejects mismatched shapes", bad_shapes)
    return finish()


def finish(skipped=False):
    bad = [n for n, e in _results if e is not None]
    print("\n" + "=" * 56)
    print(f"Results: {len(_results) - len(bad)} passed, {len(bad)} failed out of {len(_results)} tests"
          + ("  (Rust skipped)" if skipped else ""))
    print("=" * 56)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
