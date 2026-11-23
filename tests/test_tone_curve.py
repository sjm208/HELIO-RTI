"""Decoding by the file's own curve (helio.core.tone_curve).

* an sRGB ICC profile reads as the sRGB curve; Adobe RGB (1998)'s as gamma 563/256;
  a linear profile as itself; a file without a profile or EXIF mark as sRGB;
* the inverse round-trips;
* a dome photographed in Adobe RGB (gamma 2.2, profile embedded) and solved
  decoded: by its own curve the normals are as good as from linear light, and
  better than read as if sRGB; the result names the curve.

    python tests/test_tone_curve.py
"""
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


ADOBE_ICC = "/System/Library/ColorSync/Profiles/AdobeRGB1998.icc"


def adobe_icc():
    if os.path.exists(ADOBE_ICC):
        return open(ADOBE_ICC, "rb").read()
    return None


def main():
    from PIL import Image, ImageCms
    from helio.core import tone_curve as T
    x = np.arange(256, dtype=np.float32)

    print("Reading the curve")
    srgb_icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    c = T.curve_from_icc(srgb_icc, "sRGB")
    check("an sRGB profile reads as the sRGB curve (within half a level of 255)",
          c is not None and float(np.abs(c.to_linear(x) - T.SRGB.to_linear(x)).max()) < 0.5)
    icc = adobe_icc()
    if icc:
        a = T.curve_from_icc(icc, "Adobe RGB (1998)")
        check("Adobe RGB (1998)'s reads as gamma 563/256",
              a is not None and float(np.abs(a.to_linear(x) - 255 * (x / 255) ** T.ADOBE_GAMMA).max()) < 0.05)
    else:
        print("  skip no Adobe RGB profile on this machine")
    lin = T.gamma_curve(1.0, "linear")
    check("a linear curve decodes to itself", float(np.abs(lin.to_linear(x) - x).max()) < 1e-3)
    for cv in (T.SRGB, T.ADOBE_RGB):
        check(f"{cv.name}: back and forth to within 0.05 of 255",
              float(np.abs(cv.from_linear(cv.to_linear(x)) - x).max()) < 0.05)
    tmp = tempfile.mkdtemp(prefix="helio_tone_")
    plain = os.path.join(tmp, "plain.png")
    Image.fromarray(np.full((8, 8, 3), 128, np.uint8)).save(plain)
    T.file_curve.cache_clear()
    check("a file with no profile and no EXIF mark is sRGB", T.file_curve(plain).name == "sRGB")

    if not icc:
        return
    print("\nA dome shot in Adobe RGB")
    h = w = 160
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    u, v = (xx - w / 2) / (w / 2.2), (h / 2 - yy) / (h / 2.2)
    inside = u * u + v * v < 1
    nz = np.sqrt(np.clip(1 - u * u - v * v, 0, 1))
    n = np.dstack([u, v, nz])
    n[~inside] = (0, 0, 1)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    az = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    el = np.radians(45)
    L = np.column_stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.full(8, np.sin(el))])
    paths, lin_paths = [], []
    for i, l in enumerate(L):
        light = np.clip((n * l).sum(2), 0, 1) * 0.85                           # light, 0..1
        enc = np.round(255 * light ** (1 / T.ADOBE_GAMMA)).astype(np.uint8)
        p = os.path.join(tmp, f"a{i:02d}.png")
        Image.fromarray(np.dstack([enc] * 3)).save(p, icc_profile=icc)
        paths.append(p)
        q = os.path.join(tmp, f"l{i:02d}.png")
        Image.fromarray(np.dstack([np.round(255 * light).astype(np.uint8)] * 3)).save(q)
        lin_paths.append(q)
    T.file_curve.cache_clear()
    check("the photographs are read as Adobe RGB", T.capture_curve(paths).name.startswith("Adobe RGB"),
          T.capture_curve(paths).name)
    from helio.application.pipeline import SolveRequest, solve
    from helio.domain.settings import ProcessingSettings
    lit = (np.clip(n.reshape(-1, 3) @ L.T, 0, None) > 0.05).all(axis=1).reshape(h, w) & inside
    lit[:8], lit[-8:], lit[:, :8], lit[:, -8:] = False, False, False, False

    def err(r):
        m = r.normals_full / np.linalg.norm(r.normals_full, axis=2, keepdims=True)
        return float(np.degrees(np.arccos(np.clip((m * n).sum(2), -1, 1)))[lit].mean())
    L32 = L.astype(np.float32)
    floor = err(solve(SolveRequest(paths=lin_paths, lights=L32, settings=ProcessingSettings(photo_tone="linear"))))
    r = solve(SolveRequest(paths=paths, lights=L32, settings=ProcessingSettings(photo_tone="srgb")))
    own = err(r)
    real = T.capture_curve                       # the same photographs, read as if sRGB
    T.capture_curve = lambda _paths: T.SRGB
    try:
        as_srgb = err(solve(SolveRequest(paths=paths, lights=L32, settings=ProcessingSettings(photo_tone="srgb"))))
    finally:
        T.capture_curve = real
    stored = err(solve(SolveRequest(paths=paths, lights=L32, settings=ProcessingSettings(photo_tone="stored"))))
    print(f"      mean error: linear {floor:.2f}°, by its own curve {own:.2f}°, as if sRGB "
          f"{as_srgb:.2f}°, as stored {stored:.2f}°")
    check("decoded by its own curve: within 0.15° of linear light", own < floor + 0.15, (own, floor))
    check("…better than read as if sRGB, and than as stored", own < as_srgb and own < stored,
          (own, as_srgb, stored))
    check("the result names the curve", r.tone_curve.startswith("Adobe RGB"), r.tone_curve)


main()
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
