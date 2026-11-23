"""Terrain analysis must survive small inputs and never upsample them.

compute_terrain_analysis used to scale every capture to 1380 px on its long
edge with no upper clamp, so a 96x128 stack was analysed at 1035x1380 - 145x
the source pixels. That is wasted work and invented detail, and at very small
sizes the rounding could collapse an axis and raise "!ssize.empty()".
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from helio.core.processing import compute_terrain_analysis   # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def _surface(h, w):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    depth = (np.cos(d / max(2.0, min(h, w) / 6.0)) * 0.5 + 0.5).astype(np.float32)
    n = np.dstack([np.gradient(depth, axis=1), np.gradient(depth, axis=0),
                   np.ones_like(depth)]).astype(np.float32)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return depth, n


SIZES = [(8, 8), (12, 16), (24, 32), (96, 128), (200, 200), (384, 512)]

for (h, w) in SIZES:
    def _runs(h=h, w=w):
        depth, n = _surface(h, w)
        res = compute_terrain_analysis(depth, n, h, w)
        assert res, "no maps returned"
        for k, v in res.items():
            assert np.all(np.isfinite(v)), f"{k} contains non-finite values"
    check(f"terrain analysis runs at {h}x{w}", _runs)

    def _no_upsample(h=h, w=w):
        depth, n = _surface(h, w)
        res = compute_terrain_analysis(depth, n, h, w)
        for k, v in res.items():
            vh, vw = v.shape[:2]
            assert vh <= h and vw <= w, (
                f"{k} is {vh}x{vw}, larger than the {h}x{w} source - "
                f"terrain analysis is upsampling again")
    check(f"terrain maps never exceed the {h}x{w} source", _no_upsample)


def _large_still_downsamples():
    h, w = 2000, 2600
    depth, n = _surface(h, w)
    res = compute_terrain_analysis(depth, n, h, w)
    any_map = next(iter(res.values()))
    assert max(any_map.shape[:2]) <= 1380, (
        f"working resolution {any_map.shape[:2]} exceeds the 1380px cap")
check("large captures are still capped at 1380px", _large_still_downsamples)


def _empty_is_rejected():
    try:
        compute_terrain_analysis(np.zeros((0, 0), np.float32), None, 0, 0)
    except ValueError:
        return
    except Exception as e:
        raise AssertionError(f"expected ValueError, got {type(e).__name__}: {e}")
    raise AssertionError("empty depth map was accepted")
check("an empty depth map raises a clear error", _empty_is_rejected)

print("\n" + "=" * 56)
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
print("=" * 56)
sys.exit(1 if FAIL else 0)
