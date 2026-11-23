"""The two views added in 2026-10: Multi-Directional Hillshading (Blend / PCA)
and Albedo-Suppressed Raking (Clay Mode).

    python tests/test_new_views.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from helio.core import shading_engines as se                          # noqa: E402
from helio.data_manager import DataManager                            # noqa: E402
from helio.domain import registry, view_controls as vc                # noqa: E402

MDH = "Multi-Directional Hillshading (Blend / PCA)"
CLAY = "Albedo-Suppressed Raking (Clay Mode)"
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def surface(h, w, seed=1):
    """Normals of a few ridges and a groove running in different directions."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = (3 * np.sin(xx / 17.0) + 2 * np.sin(yy / 11.0 + 1) + 2.5 * np.sin((xx + yy) / 23.0)
         + np.random.default_rng(seed).normal(size=(h, w)) * 0.15).astype(np.float32)
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(z)]).astype(np.float32)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def data_for(n):
    d = DataManager()
    d.normals_full = d.normals_proxy = n
    d.h, d.w = n.shape[:2]
    return d


def render(name, n, s, light=(.4, .3, .8), gain=1.0, dm=None):
    d = dm or data_for(n)
    ctx = dict(n_std=n, depth_p=None, light=np.array(light, np.float64), gain=gain, use_proxy=True,
               crop_coords=None, res_ref=1.0, res_res=1.0, _is_real_crop=False)
    return se.get_shading_engine(name, d, dict(registry.default_engine_settings()[name], **s), *n.shape[:2], ctx=ctx)


def main():
    print("Registration")
    check("both are offered in the view menu", MDH in registry.RENDER_ENGINES and CLAY in registry.RENDER_ENGINES)
    check("both have a blurb", MDH in registry.ENGINE_BLURBS and CLAY in registry.ENGINE_BLURBS)
    check("both are implemented", {MDH, CLAY} <= se.implemented_engines())
    check("each has its own controls", vc.VIEW_SETTINGS[MDH] == ("gain", "mdh_alt", "mdh_mode")
          and vc.VIEW_SETTINGS[CLAY] == ("gain", "clay_floor", "clay_micro"))
    check("every control they use exists, and the defaults agree with the engine defaults",
          all(k in vc.CONTROLS for k in vc.VIEW_SETTINGS[MDH] + vc.VIEW_SETTINGS[CLAY])
          and all(registry.default_engine_settings()[v][k] == vc.CONTROLS[k].default
                  for v in (MDH, CLAY) for k in vc.VIEW_SETTINGS[v] if k != "gain"))
    check("Clay Mode follows the dome light; the hillshading does not",
          vc.light_kind(CLAY) == "dome" and vc.light_kind(MDH) is None)
    check("they sit in the menu groups", se.engine_meta(MDH)["family"] == "Relief and terrain"
          and se.engine_meta(CLAY)["family"] == "Relighting")
    check("tone controls are hidden for false colour and for Clay Mode, shown for the grey blend",
          not vc.takes_tone(CLAY) and not vc.takes_tone(MDH, {"mdh_mode": 1}) and vc.takes_tone(MDH, {"mdh_mode": 0}))

    n = surface(240, 320)
    print("Multi-Directional Hillshading")
    blend = render(MDH, n, dict(mdh_mode=0))
    check("blend: a grey picture, 0..1", blend.shape == (240, 320) and blend.dtype == np.float32
          and 0.0 <= blend.min() and blend.max() <= 1.0 and blend.std() > 0.05, (blend.shape, blend.std()))
    ref = np.mean([np.maximum(0, n.reshape(-1, 3) @ L) for L in se._mdh_lights(35.0)], axis=0).reshape(240, 320)
    lo, hi = np.percentile(ref, [2, 98])
    check("blend equals the mean of eight hillshades, stretched 2-98%",
          float(np.abs(blend - np.clip((ref - lo) / (hi - lo), 0, 1)).max()) < 2e-3, float(np.abs(blend - np.clip((ref - lo) / (hi - lo), 0, 1)).max()))
    pca = render(MDH, n, dict(mdh_mode=1))
    check("PCA composite: RGB 0..1, three distinct channels", pca.shape == (240, 320, 3)
          and 0.0 <= pca.min() and pca.max() <= 1.0 and min(np.corrcoef(pca[..., i].ravel(), pca[..., j].ravel())[0, 1] ** 2
                                                              for i, j in ((0, 1), (0, 2), (1, 2))) < 0.25)
    X = np.maximum(n.reshape(-1, 3) @ se._mdh_lights(35.0).T, 0)
    X = X - X.mean(0)
    vals = np.sort(np.linalg.eigvalsh((X.T @ X) / (len(X) - 1)))[::-1]
    pc1 = (X @ np.linalg.eigh((X.T @ X))[1][:, -1]).reshape(240, 320)
    lo1, hi1 = np.percentile(pc1, [2, 98])
    ref1 = np.clip((pc1 - lo1) / (hi1 - lo1), 0, 1)
    ref1 = ref1 if np.corrcoef(ref1.ravel(), pca[..., 0].ravel())[0, 1] > 0 else 1 - np.clip((pc1 - lo1) / (hi1 - lo1), 0, 1)
    if abs(np.corrcoef(ref1.ravel(), pca[..., 0].ravel())[0, 1]) < 0.99:       # the other sign, stretched the other way
        lo1, hi1 = np.percentile(-pc1, [2, 98])
        ref1 = np.clip((-pc1 - lo1) / (hi1 - lo1), 0, 1)
    check("channel 1 is the first principal component, stretched 2-98% (matches an independent PCA)",
          float(np.abs(ref1 - pca[..., 0]).max()) < 5e-3, float(np.abs(ref1 - pca[..., 0]).max()))
    check("the first component carries most of the variance", vals[0] / vals.sum() > 0.4, vals / vals.sum())
    check("the colours do not flip between two renders (fixed sign convention)",
          np.array_equal(pca, render(MDH, n, dict(mdh_mode=1))) and np.array_equal(pca, render(MDH, n, dict(mdh_mode=1), dm=data_for(n.copy()))))
    check("the light slider changes nothing", np.array_equal(render(MDH, n, dict(mdh_mode=0), light=(-.5, .2, .4)), blend))
    check("altitude changes the picture", not np.array_equal(render(MDH, n, dict(mdh_alt=60.0)), blend))
    check("gain changes the picture", not np.array_equal(render(MDH, n, {}, gain=0.5), blend)
          and render(MDH, n, dict(mdh_mode=1), gain=0.5).shape == pca.shape)
    flat = np.zeros((60, 80, 3), np.float32)
    flat[..., 2] = 1.0
    check("a flat surface does not fail", np.isfinite(render(MDH, flat, dict(mdh_mode=1))).all()
          and np.isfinite(render(MDH, flat, dict(mdh_mode=0))).all())

    print("Clay Mode")
    clay = render(CLAY, n, {})
    check("RGB, grey (three equal channels), 0..1", clay.shape == (240, 320, 3) and clay.dtype == np.float32
          and np.array_equal(clay[..., 0], clay[..., 1]) and np.array_equal(clay[..., 1], clay[..., 2])
          and 0.0 <= clay.min() and clay.max() <= 1.0)
    graze = render(CLAY, n, {}, light=(.95, .1, .05))
    check("in shadow the picture is exactly the shadow floor (default 0.06)", abs(float(graze.min()) - 0.06) < 1e-6, graze.min())
    plain = render(CLAY, n, dict(clay_micro=0.0, clay_floor=0.0))
    L = np.array([.4, .3, .8])
    L = (L / np.linalg.norm(L)).astype(np.float32)
    lam = np.maximum(0, n.reshape(-1, 3) @ L).reshape(240, 320)
    check("with no floor and no micro-roughness it is plain Lambert shading",
          float(np.abs(plain[..., 0] - np.clip(lam, 0, 1)).max()) < 1e-6)
    check("micro-roughness emphasis changes the fine detail", float(np.abs(render(CLAY, n, dict(clay_micro=2.0, clay_floor=0.0))[..., 0] - plain[..., 0]).max()) > 0.01)
    check("the light moves the shading", not np.array_equal(render(CLAY, n, {}, light=(-.5, -.2, .5)), clay))
    check("a raised floor lifts the shadows", render(CLAY, n, dict(clay_floor=0.3)).min() >= 0.3 - 1e-6)
    check("gain changes it", not np.array_equal(render(CLAY, n, {}, gain=0.5), clay))
    saved = se._rs
    se._rs = None
    ref_np = render(CLAY, n, dict(clay_micro=1.2, clay_floor=0.1), gain=0.8)
    se._rs = saved
    if saved is not None:
        got = render(CLAY, n, dict(clay_micro=1.2, clay_floor=0.1), gain=0.8)
        check("Rust and NumPy give the same Clay Mode picture", float(np.abs(ref_np - got).max()) < 1e-6, float(np.abs(ref_np - got).max()))

    print("Albedo cannot reach the picture")
    from helio.application.render import composite
    d = data_for(n)
    d.albedo_rgb_full = d.albedo_proxy = np.random.default_rng(3).integers(0, 255, (240, 320, 3)).astype(np.uint8)
    d.roughness_proxy = None
    d.albedo_grey_full = d.albedo_rgb_full[..., 1]
    tone = dict(bright=1.0, contrast=0.0, gamma=1.0, micro=0.0, rough_opacity=0.5)
    pics = [composite(CLAY, clay, d, tone, 240, 320, True, view_mode=m, colour_opacity=1.0)
            for m in ("Combined Colour", "Shading", "Roughness Fusion")]
    check("Combined Colour, Shading and Roughness Fusion show the same grey clay picture",
          all(np.array_equal(pics[0], p) for p in pics[1:]) and np.array_equal(pics[0][..., 0], pics[0][..., 2]))
    d.albedo_rgb_full = d.albedo_proxy = 255 - d.albedo_proxy
    other = composite(CLAY, clay, d, tone, 240, 320, True, view_mode="Combined Colour", colour_opacity=1.0)
    check("and a different albedo makes no difference", np.array_equal(pics[0], other))

    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
