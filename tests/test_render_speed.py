"""Fluid relighting at the preview's full size (PERF-001).

The light dome redraws the view on every move.  At the preview's full size
(2500 px on the long edge) the views took 110-1100 ms a frame; they now take
roughly 10-50 ms, with the same pictures.  This checks:

* the colour and tone composite gives the classic composite's picture bit
  for bit (the classic code is kept below as the reference);
* parallel.quantiles and parallel.median equal NumPy's, bit for bit;
* the views LIGHT_KIND says the light reaches are exactly the ones that
  change when it swings round, so a drag over any other view can skip
  drawing (PS Normal-Map Lit (8-Dir) lights from every side and takes only
  the elevation, from its own slider: the dome is hidden for it);
* cast shadows fall on the side away from the light (KD-055) and agree with
  a plain ray march in the right direction;
* exaggerated shading's light-free precomputation matches the per-pixel
  light rotation it replaced;
* the cache of light-free work keeps to its budget and lets go of the
  arrays of an old solve;
* every view the light reaches draws a drag frame at 2500 px in under
  FRAME_MS (each time is printed).

    python tests/test_render_speed.py
"""
import gc
import math
import os
import sys
import tempfile
import time

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FRAME_MS = 150.0
try:
    from helio.core._native import rs as _rs
except Exception:
    _rs = None
if _rs is None:
    FRAME_MS = 500.0                         # the NumPy fallback is allowed a few times longer; Rust is held to 150
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


# ── the classic composite, as it was, for comparison ─────────────────────

def classic_tone(shading, albedo, rough, settings, view_mode, colour_opacity):
    sh, sw = shading.shape[:2]
    if view_mode == "Combined Colour":
        shading_rgb = np.full((sh, sw, 3), 255.0) * shading[..., None]
        colour_rgb = albedo.astype(float) * shading[..., None]
        out = shading_rgb * (1.0 - colour_opacity) + colour_rgb * colour_opacity
    elif view_mode == "Roughness Fusion":
        out = np.full((sh, sw, 3), 255.0) * (
            shading * (1 - settings['rough_opacity']) +
            (rough.astype(float) / 255.0) * settings['rough_opacity'])[..., None]
    else:
        out = np.full((sh, sw, 3), 255.0) * shading[..., None]
    if settings.get('micro', 0.0) > 0:
        gray = cv2.cvtColor(albedo, cv2.COLOR_RGB2GRAY).astype(float)
        out = np.clip(out + (gray - cv2.GaussianBlur(gray, (0, 0), 3))[..., None]
                      * settings['micro'] * 2.0, 0, 255)
    out = np.clip((out * settings['bright'] - 128) * (1 + settings['contrast'] / 100) + 128, 0, 255)
    return cv2.LUT(out.astype(np.uint8), np.array(
        [((i / 255.0) ** (1.0 / settings['gamma'])) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")) if settings.get('gamma', 1.0) != 1.0 else out.astype(np.uint8)


def classic_exaggerated(n, L, n_scales=4, base=2.0, gain=1.0):
    """Exaggerated shading as it was: the light turned at every pixel."""
    def rot(L, m):
        ax_x, ax_y = -m[..., 1], m[..., 0]
        c = np.clip(m[..., 2], -1.0, 1.0)
        s_len = np.sqrt(ax_x * ax_x + ax_y * ax_y)
        small = s_len < 1e-8
        inv = np.where(small, 0.0, 1.0 / np.maximum(s_len, 1e-12))
        kx, ky = ax_x * inv, ax_y * inv
        kdotv = kx * L[0] + ky * L[1]
        out = np.empty(m.shape, np.float32)
        out[..., 0] = L[0] * c + ky * L[2] * s_len + kx * kdotv * (1.0 - c)
        out[..., 1] = L[1] * c + -kx * L[2] * s_len + ky * kdotv * (1.0 - c)
        out[..., 2] = L[2] * c + (kx * L[1] - ky * L[0]) * s_len
        return np.where(small[..., None], L.reshape(1, 1, 3), out)
    h, w = n.shape[:2]
    acc = np.zeros((h, w), np.float32)
    total = 0.0
    for i in range(n_scales):
        sigma = base * (2.0 ** i)
        m = cv2.GaussianBlur(n, (0, 0), sigmaX=sigma, sigmaY=sigma)
        m = m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-8)
        sh = np.maximum(0.0, np.sum(n * rot(L, m), axis=-1))
        wgt = 1.0 / (i + 1.0)
        acc += sh * wgt
        total += wgt
    out = acc / max(total, 1e-6)
    med = float(np.median(out))
    spread = float(np.percentile(np.abs(out - med), 98))
    out = np.clip((out - med) / max(spread, 1e-6) * 0.42 + 0.5, 0, 1)
    return np.power(out, 1.0 / gain).astype(np.float32)


def march(z, ux, uy, tan_elev, steps):
    """Cast shadows the plain way: sample towards the light, step by step."""
    h, w = z.shape
    blocked = np.zeros((h, w), bool)
    for t in range(1, steps + 1):
        M = np.float64([[1, 0, ux * t], [0, 1, uy * t]])   # dst(p) = z(p + t·u)
        zs = cv2.warpAffine(z, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REPLICATE)
        blocked |= (zs - z) > (t * tan_elev)
    return blocked


def solved(h, w):
    """The golden fixture's analytic surface at h x w, solved by the pipeline."""
    from tools import make_golden as g
    from helio.application.pipeline import SolveRequest, solve, store
    from helio.data_manager import DataManager
    from helio.domain.registry import default_engine_settings
    from helio.domain.settings import ProcessingSettings
    g.H, g.W = h, w
    tmp = tempfile.mkdtemp(prefix="helio_speed_")
    paths, L = g.render_stack(tmp)
    r = solve(SolveRequest(paths=paths, lights=L,
                           settings=ProcessingSettings(photo_tone="linear")))
    D = DataManager()
    store(r, D)
    D.plate_paths = paths
    D.engine_settings = default_engine_settings()
    return D


def main():
    from helio.core import parallel as P
    from helio.application.render import _tone_rows
    print("=" * 60)
    print("Fluid relighting at full preview size")
    print("=" * 60)
    rng = np.random.default_rng(7)

    print("\nThe composite: the classic picture, bit for bit")
    h, w = 1203, 977
    albedo = (rng.random((h, w, 3)) * 256).astype(np.uint8)
    rough = (rng.random((h, w)) * 256).astype(np.uint8)
    base = dict(bright=1.0, contrast=0.0, gamma=1.0, micro=0.0, rough_opacity=0.35)
    tones = [base, dict(base, bright=1.15, contrast=-12.0), dict(base, gamma=1.4),
             dict(base, micro=0.6, contrast=25.0)]
    shadings = {"float32": (rng.random((h, w)) * 1.3 - 0.1).astype(np.float32),
                "float64": rng.random((h, w)) * 1.3 - 0.1}
    odd = shadings["float32"].copy()
    odd[5, 7], odd[600, 3], odd[1000, 900] = np.nan, np.inf, -np.inf
    shadings["with NaN and inf"] = odd
    bad = []
    with np.errstate(invalid="ignore", over="ignore"):   # NaN and inf, on purpose
        for sname, s in shadings.items():
            for mode in ("Combined Colour", "Shading", "Roughness Fusion"):
                for op in (1.0, 0.6):
                    for t in tones:
                        a = classic_tone(s, albedo, rough, t, mode, op)
                        b = _tone_rows(s, albedo, rough, t, mode, op, h, w)
                        if not np.array_equal(a, b):
                            bad.append((sname, mode, op, t, int((a != b).sum())))
    check("every layer, opacity and tone setting, float32 and float64 shading, "
          "NaN and infinity: identical", not bad, bad[:3])

    print("\nPercentiles and medians without sorting")
    mism = []
    for k in range(18):
        shape = [(1500, 1100), (1001, 1003), (2000, 700)][k % 3]
        kind = k % 6
        if kind == 0:
            a = rng.random(shape, dtype=np.float32)
        elif kind == 1:
            a = (rng.normal(size=shape) ** 3).astype(np.float32)
        elif kind == 2:
            a = (np.round(rng.random(shape) * 20) / 7).astype(np.float32)      # many ties
        elif kind == 3:
            a = rng.random(shape) * 1e-3                                        # float64
        elif kind == 4:
            a = np.full(shape, 0.25, np.float32)                                # constant
        else:
            a = rng.random(shape, dtype=np.float32)
            a[3, 3] = np.nan                                                    # NumPy decides
        qs = [0, 1, 2.5, 50, 97.3, 98, 99, 100]
        got = P.quantiles(a, qs)
        ref = [float(v) for v in np.percentile(a, qs)]
        same = all((g == r) or (math.isnan(g) and math.isnan(r)) for g, r in zip(got, ref))
        m1, m2 = P.median(a), float(np.median(a))
        same = same and ((m1 == m2) or (math.isnan(m1) and math.isnan(m2)))
        if not same:
            mism.append((k, got, ref, m1, m2))
    check("quantiles and median equal NumPy's to the last bit (ties, float64, constant, NaN)",
          not mism, mism[:2])

    print("\nWhich views the light reaches")
    from helio.application.render import render
    from helio.domain.registry import RENDER_ENGINES
    from helio.domain.view_controls import LIGHT_KIND, light_settings
    D = solved(240, 320)
    es = D.engine_settings
    ph, pw = D.normals_proxy.shape[:2]
    wrong = []
    for eng in RENDER_ENGINES:
        outs = []
        for lx, ly in ((-0.6, -0.3), (0.3, 0.6)):          # same elevation, swung round
            lz = math.sqrt(max(0.0, 1 - lx * lx - ly * ly))
            st = dict(es[eng])
            st.update(light_settings(lx, ly, lz))
            outs.append(np.asarray(render(eng, D, st, ph, pw, True), np.float64))
        changes = not np.array_equal(outs[0], outs[1])
        if changes != (eng in LIGHT_KIND):
            wrong.append((eng, "changes" if changes else "does not change"))
    check(f"the {len(LIGHT_KIND)} views in LIGHT_KIND change as the light swings round and "
          f"the other {len(RENDER_ENGINES) - len(LIGHT_KIND)} do not", not wrong, wrong)

    print("\nCast shadows")
    from helio.core.shading_engines import _cast_shadow_mask, _engine_cast_shadow_raking
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
    ridge = np.exp(-((xx - 100) / 3.0) ** 2).astype(np.float32)
    flat = np.zeros((200, 200, 3), np.float32)
    flat[..., 2] = 1.0
    sides = {}
    for name, light in (("west", (-1.0, 0.0, 0.25)), ("east", (1.0, 0.0, 0.25)),
                        ("north", (0.0, 1.0, 0.25))):
        o = _engine_cast_shadow_raking(None, dict(shadow_relief=0.08, shadow_steps=48,
                                                  shadow_ambient=0.18), 200, 200,
                                       dict(n_std=flat, depth_p=ridge, light=np.array(light),
                                            gain=1.0))
        sides[name] = (float(o[100, 40:95].mean()), float(o[100, 105:160].mean()))
    check("a north-south ridge lit from the west throws its shadow east, and from the "
          "east, west (KD-055)", sides["west"][1] < 0.5 * sides["west"][0]
          and sides["east"][0] < 0.5 * sides["east"][1], sides)
    check("…and lit from the north it throws none across", abs(sides["north"][0] - sides["north"][1]) < 1e-6,
          sides["north"])
    zf = cv2.resize(D.depth_proxy_p.astype(np.float32), (1250, 940))
    zf = ((zf - zf.min()) / max(np.ptp(zf), 1e-8) * 0.08 * 1250).astype(np.float32)
    worst = 0.0
    for az in (0, 37, 90, 151, 222, 300):
        for el in (10, 25):
            ux, uy = math.sin(math.radians(az)), -math.cos(math.radians(az))
            tan = math.tan(math.radians(el))
            a = _cast_shadow_mask(zf, ux, uy, tan, 48)[50:-50, 50:-50]
            b = march(zf, ux, uy, tan, 48)[50:-50, 50:-50]
            worst = max(worst, float((a != b).mean()))
    check(f"the shadows agree with a ray march towards the light on {100 * (1 - worst):.2f}% "
          "of pixels or more, every direction (only soft edges differ)", worst < 0.005, worst)

    print("\nExaggerated shading")
    from helio.core.shading_engines import _engine_exaggerated_shading
    n = D.normals_proxy
    Lv = np.array([-0.5, 0.35, 0.79], np.float32)
    Lv /= np.linalg.norm(Lv)
    old = classic_exaggerated(n, Lv)
    new = _engine_exaggerated_shading(D, dict(exag_scales=4, exag_sigma=2.0), ph, pw,
                                      dict(n_std=n, light=Lv.astype(np.float64), gain=1.0))
    d = np.abs(old.astype(np.float64) - new) * 255
    check(f"turning the normals once instead of the light at every pixel changes no pixel by "
          f"a visible amount (largest {d.max():.3f} of 255)", d.max() < 0.5, d.max())

    print("\nThe cache of light-free work")
    c = P.LightFreeCache(max_bytes=12 << 20)
    srcs = [np.zeros((100, 100), np.float32) for _ in range(5)]
    for i, s in enumerate(srcs):
        c.get("t", (s,), (), lambda: np.zeros((1000, 1000), np.float32))        # 4 MB each
    check("it keeps to its budget, dropping the least recently used",
          c.stats()["mb"] <= 12.0 and c.stats()["entries"] <= 3, c.stats())
    calls = []
    s = np.ones(10, np.float32)
    c.get("t", (s,), (1,), lambda: calls.append(1) or np.zeros(3))
    c.get("t", (s,), (1,), lambda: calls.append(1) or np.zeros(3))
    c.get("t", (s.copy(),), (1,), lambda: calls.append(1) or np.zeros(3))
    check("a value is worked out once per source array; an equal copy is another array",
          len(calls) == 2)
    del srcs, s
    gc.collect()
    c.get("t", (np.zeros(2),), (), lambda: np.zeros(2))
    check("entries whose source arrays are gone are dropped", c.stats()["entries"] <= 2, c.stats())

    print(f"\nA drag frame at the preview's full size (2500 x 1875), {P.workers()} threads")
    from helio.application.viewing import Display, screen_image
    D2 = solved(2500, 1875)
    es2 = D2.engine_settings
    disp = Display(layer="Combined Colour")
    slow = []
    for eng in LIGHT_KIND:
        times = []
        for k in range(7):
            ang = k / 7 * 2 * math.pi
            lx, ly = 0.75 * math.sin(ang), -0.75 * math.cos(ang)
            st = dict(es2[eng])
            st.update(light_settings(lx, ly, math.sqrt(1 - lx * lx - ly * ly)))
            t0 = time.perf_counter()
            screen_image(eng, D2, st, disp)
            times.append((time.perf_counter() - t0) * 1000)
        frame = sorted(times[1:])[len(times[1:]) // 2]       # the first frame fills the cache
        print(f"       {eng:32s} {frame:6.1f} ms a frame (first {times[0]:.0f} ms)")
        if frame > FRAME_MS:
            slow.append((eng, round(frame)))
    check(f"every view the light reaches draws a drag frame in under {FRAME_MS:.0f} ms", not slow, slow)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
