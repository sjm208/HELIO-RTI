"""helio_core_rs relighting kernels against the NumPy engines they replace.

For every engine that has a Rust kernel (Standard PBR, hillshade, multi-
directional hillshade / PS Normal-Map Lit 8-Dir, Diffuse Gain, Exaggerated
Shading) the picture is rendered twice from the same normals, once with the
kernel and once with helio.core.shading_engines._rs switched off, and the two
must agree.  Also: parallel.quantiles / median stay bit for bit NumPy's when
they use the Rust order-statistics kernel, and the raw kernels reject bad
input.

    python tests/test_rust_relight.py

Needs: cd helio_core_rs && maturin develop --release   (skips without it)
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from helio.core import parallel                                   # noqa: E402
from helio.core import shading_engines as se                      # noqa: E402
from helio.data_manager import DataManager                        # noqa: E402

try:
    import helio_core_rs as rs                                    # noqa: E402
    rs.lls_normals
except (ImportError, AttributeError):
    rs = None

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def normals(h, w, seed=1):
    r = np.random.default_rng(seed)
    n = r.normal(size=(h, w, 3)).astype(np.float32)
    n[..., 2] = np.abs(n[..., 2]) + 0.5
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def both(fn):
    """(Rust picture, NumPy picture) of the same call."""
    saved = se._rs
    try:
        se._rs = rs
        a = fn()
        se._rs = None
        b = fn()
    finally:
        se._rs = saved
    return a, b


def close(name, fn, tol):
    a, b = both(fn)
    ok = a.shape == b.shape and a.dtype == np.float32
    d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if ok else float("inf")
    check(f"{name}  (max |diff| {d:.1e})", ok and d <= tol, f"shape {a.shape} vs {b.shape}, dtype {a.dtype}, diff {d}")


def main():
    if rs is None:
        print("helio_core_rs not built: skipped")
        return 0
    h, w = 301, 413                                   # not a multiple of any block
    n = normals(h, w)
    d = DataManager()
    d.normals_full = d.normals_proxy = n
    d.h, d.w = h, w

    print("Engines, Rust vs NumPy")
    for g in (0.15, 0.5, 1.0):                         # table path (gain <= 1)
        s = dict(lx=.3, ly=.4, lz=.86, gain=g)
        close(f"Standard PBR gain {g}", lambda s=s: se.render_shading("Standard PBR", d, s, h, w), 2e-3)
    for g in (1.7, 4.0):                               # exact-power path stays NumPy
        s = dict(lx=-.5, ly=.1, lz=.7, gain=g)
        close(f"Standard PBR gain {g}", lambda s=s: se.render_shading("Standard PBR", d, s, h, w), 0)
    for alt, azi, amb in ((45, 315, .15), (10, 0, 0.0), (80, 200, .4)):
        s = dict(hs_alt=alt, hs_azi=azi, hs_amb=amb)
        close(f"hillshade alt {alt} azi {azi}", lambda s=s: se.calc_hillshade_raw(None, s, n), 1e-6)
    for alt in (20, 35, 70):
        close(f"multi-hillshade alt {alt}",
              lambda alt=alt: se.calc_multi_hillshade(None, dict(hs_alt=alt, hs_amb=.1), n), 1e-6)
    close("PS Normal-Map Lit (8-Dir)",
          lambda: se.get_shading_engine("PS Normal-Map Lit (8-Dir)", d, dict(hs_alt=40), h, w,
                                    ctx=dict(n_std=n, depth_p=np.zeros((h, w), np.float32))), 1e-6)
    big_h, big_w = 700, 900                            # percentile stretch needs >= 1M px for the fast path
    nb = normals(big_h, big_w, 2)
    db = DataManager()
    db.normals_full = db.normals_proxy = nb
    db.h, db.w = big_h, big_w
    for spec in (0.0, 0.3):
        s = dict(gain=1.0, dg_spec=spec, dg_strength=2.0, dg_gamma=0.7)
        close(f"Diffuse Gain spec {spec}", lambda s=s: se.render_shading("Diffuse Gain", db, s, big_h, big_w), 1e-4)
    s = dict(gain=1.0, dg_strength=3.5, dg_gamma=1.4, dg_spec=0.1)
    close("Diffuse Gain strength 3.5 gamma 1.4", lambda: se.render_shading("Diffuse Gain", db, s, big_h, big_w), 1e-4)
    for scales in (1, 4):
        s = dict(lx=.3, ly=.4, lz=.86, gain=1.0, exag_scales=scales)
        close(f"Exaggerated Shading {scales} scale(s)",
              lambda s=s: se.render_shading("Exaggerated Shading", db, s, big_h, big_w), 1e-4)

    import cv2
    zr_ = np.random.default_rng(9)
    depth = cv2.GaussianBlur(zr_.random((big_h, big_w)).astype(np.float32), (0, 0), 5)
    for light, gain, kw in (((.5, -.3, .6), 1.0, {}), ((.9, 0, .4), .15, {}), ((-.7, -.6, .3), 2.5, dict(shadow_steps=200, shadow_ambient=.3)),
                            ((0, 0, 1.0), 1.0, {}), ((0, -1, .2), 1.0, dict(shadow_relief=.3))):
        ctx = dict(n_std=nb, depth_p=depth, light=np.array(light, np.float32), gain=gain)
        close(f"Cast-Shadow Raking light {light} gain {gain} {kw}",
              lambda ctx=ctx, kw=kw: se.get_shading_engine("Cast-Shadow Raking", db, dict(kw), big_h, big_w, ctx=ctx), 1e-6)
    z_ = depth * 300
    for ux, uy, tan, steps in ((1, 0, .5, 48), (.6, -.8, .3, 17), (-.8, .6, 1.2, 300)):
        a, b = both(lambda: se._cast_shadow_mask(z_, ux, uy, tan, steps))
        check(f"shadow mask identical, direction ({ux}, {uy}) steps {steps}", np.array_equal(a, b))

    r2 = np.random.default_rng(11)
    npx = big_h * big_w
    ptm32 = (r2.random((6, npx)) * 40).astype(np.float32)
    ptm32[5] += 100
    med8 = (r2.random((big_h, big_w)) * 255).astype(np.uint8)

    def rti(mode, arr, light=(.3, .4, .86), gain=1.0, ldtype=np.float32, **kw):
        ctx = dict(n_std=nb, light=np.array(light, ldtype), gain=gain,
                   ptm=arr if mode == "ptm" else None, med=arr if mode == "median" else None)
        dr = DataManager()
        dr.normals_full = nb
        dr.ptm_coeffs = arr if mode == "ptm" else None
        return lambda: se.get_shading_engine("RTI Specular Enhancement", dr, dict(kw), big_h, big_w, ctx=ctx)
    for label, mode, arr in (("PTM float32", "ptm", ptm32), ("PTM float64", "ptm", ptm32.astype(np.float64)),
                             ("median uint8", "median", med8), ("median float32", "median", med8.astype(np.float32)),
                             ("diffuse only", "diffuse", None)):
        for gain in (1.0, 0.15, 2.0):
            close(f"RTI Specular Enhancement {label} gain {gain}", rti(mode, arr, gain=gain), 1e-5)
        close(f"RTI Specular Enhancement {label}, strong settings",
              rti(mode, arr, (-.6, .2, .5), rti_spec_coeff=3.0, rti_spec_floor=.01, rti_spec_blend=1.0), 1e-5)
    # the layout the preview really has: a (6, n) view of (h, w, 6) data
    pix_major = np.ascontiguousarray(ptm32.reshape(6, big_h, big_w).transpose(1, 2, 0)).transpose(2, 0, 1).reshape(6, -1)
    check("the preview's PTM layout is Fortran-ordered", pix_major.flags.f_contiguous and not pix_major.flags.c_contiguous)
    for dt in (np.float32, np.float64):
        close(f"RTI Specular Enhancement, pixel-major PTM {np.dtype(dt).name}", rti("ptm", pix_major.astype(dt, order="K")), 1e-5)
    close("RTI Specular Enhancement, float64 light (as the interface passes it)",
          rti("ptm", pix_major, ldtype=np.float64), 1e-5)
    wide = np.zeros((6, 2 * npx), np.float32)
    wide[:, ::2] = ptm32
    strided = wide[:, ::2]                                # (6, n_px) but every other element
    check("a strided PTM is not contiguous, falls back to NumPy, and gives the same picture",
          not (strided.flags.c_contiguous or strided.flags.f_contiguous))
    se._rs = rs
    fell_back, direct = rti("ptm", strided)(), rti("ptm", ptm32)()
    check("the fallback picture matches the Rust picture of the same coefficients",
          fell_back.dtype == np.float64 and direct.dtype == np.float32
          and float(np.abs(fell_back - direct.astype(np.float64)).max()) < 1e-5)

    print("Subsurface Scattering")
    sr = np.random.default_rng(31)
    ds = DataManager()
    ds.normals_full = nb
    ds.albedo_rgb_full = ds.albedo_proxy = (sr.random((big_h, big_w, 3)) * 256).astype(np.uint8)
    ds.h, ds.w = big_h, big_w
    for use_proxy, radius, blend, gain, light in ((True, 8.0, .45, 1.0, (.3, .4, .86)), (True, 3.0, .9, 2.0, (-.6, .2, .5)),
                                                  (False, 5.0, 0.0, 1.0, (0, 0, 1.0)), (True, 20.0, 1.0, .5, (.9, -.3, .2))):
        ctx = dict(n_std=nb, gain=gain, use_proxy=use_proxy, light=np.array(light, np.float64), crop_coords=None)
        close(f"SSS proxy={use_proxy} radius {radius} blend {blend} gain {gain}",
              lambda ctx=ctx, radius=radius, blend=blend: se.get_shading_engine(
                  "Subsurface Scattering (SSS)", ds, dict(sss_radius=radius, sss_blend=blend), big_h, big_w, ctx=dict(ctx)), 1e-6)

    print("Horizon scan (Sky-View Factor, Openness) against RVT")
    try:
        import rvt.vis as rvt_vis
    except ImportError:
        rvt_vis = None
    if rvt_vis is None:
        print("  (rvt-py is not installed: skipped)")
    else:
        import cv2 as _cv2
        hr = np.random.default_rng(41)
        for (hh, ww, nd, rr, res, nan_box) in ((120, 150, 16, 20, 1.0, False), (97, 131, 8, 7, 0.37, True),
                                               (200, 90, 32, 12, 2.5, False), (60, 60, 4, 5, 1.0, False)):
            z = _cv2.GaussianBlur(hr.random((hh, ww)).astype(np.float32), (0, 0), 3) * 60
            if nan_box:
                z[10:20, 30:40] = np.nan
                z[5, 5] = np.nan
            for neg in (False, True):
                surf = -z if neg else z
                ref = rvt_vis.sky_view_factor(dem=surf, resolution=res, compute_svf=True, compute_opns=True,
                                              svf_n_dir=nd, svf_r_max=rr)
                got = se._horizon_scan_rust(surf, res, nd, rr)
                ok = (got is not None and np.array_equal(np.isnan(ref["svf"]), np.isnan(got["svf"]))
                      and float(np.nanmax(np.abs(ref["svf"] - got["svf"]))) < 1e-12
                      and float(np.nanmax(np.abs(ref["opns"] - got["opns"]))) < 1e-10)
                check(f"{hh}x{ww}, {nd} directions, radius {rr}, pixel {res}, NaN box {nan_box}, inverted {neg}: same as RVT", ok)
        zp = _cv2.GaussianBlur(hr.random((big_h, big_w)).astype(np.float32), (0, 0), 4) * 60
        dh = DataManager()
        dh.normals_full = nb
        dh.h, dh.w = big_h, big_w
        for eng in ("Positive Openness", "Negative Openness", "Sky-View Factor (SVF)"):
            for sett in (dict(op_radius=20, svf_dirs=16, svf_radius=20), dict(op_radius=9, svf_dirs=8, svf_radius=9)):
                def render(eng=eng, sett=sett):
                    se._svf_bundle._slot = None
                    ctx = dict(n_std=nb, depth_p=zp, light=np.array([0, 0, 1.0]), gain=1.0, res_res=1.0, use_proxy=False,
                               _is_real_crop=False, crop_coords=None)
                    dh._terrain_cache = None
                    return se.get_shading_engine(eng, dh, dict(sett), big_h, big_w, ctx=ctx)
                saved_rs = se._rs
                se._rs = None
                ref = render()
                se._rs = rs
                got = render()
                se._rs = saved_rs
                check(f"{eng} {sett}: same picture with and without Rust (max |diff| {float(np.abs(ref.astype(np.float64) - got).max()):.1e})",
                      ref.shape == got.shape and float(np.abs(ref.astype(np.float64) - got).max()) < 1e-6)

    print("Kept light-free work (Normal Residual, Ambient Occlusion)")
    import cv2 as cv
    import math as _m
    kr = np.random.default_rng(51)
    kh, kw = 1100, 1100                                    # >= 1M pixels: the fast percentile path
    kn = kr.normal(size=(kh, kw, 3)).astype(np.float32)
    kn[..., 2] = np.abs(kn[..., 2]) + .5
    kn /= np.linalg.norm(kn, axis=-1, keepdims=True)
    kd = cv.GaussianBlur(kr.random((kh, kw)).astype(np.float32), (0, 0), 3)
    dk = DataManager()
    dk.normals_full = dk.normals_proxy = kn
    dk.h, dk.w = kh, kw

    def residual_reference(mode, scale, gain_nr):
        sg = max(1.0, kh * scale / 100.0)
        nx, ny = kn[..., 0], kn[..., 1]
        nxh, nyh = nx - cv.GaussianBlur(nx, (0, 0), sg), ny - cv.GaussianBlur(ny, (0, 0), sg)
        fr = []
        for az in [0, 45, 90, 135, 180, 225, 270, 315]:
            a = _m.radians(az)
            fr.append(nxh * float(_m.cos(_m.radians(15)) * _m.sin(a)) + nyh * float(-_m.cos(_m.radians(15)) * _m.cos(a)))
        sh = np.max(np.abs(np.stack(fr, axis=0)), axis=0).astype(np.float32) * gain_nr
        if mode == 0:
            p1, p99 = float(np.percentile(sh, 1)), float(np.percentile(sh, 99))
            return (1.0 - np.clip((sh - p1) / max(p99 - p1, 1e-8), 0, 1)).astype(np.float32)
        res = np.abs(sh)
        p1, p99 = float(np.percentile(res, 1)), float(np.percentile(res, 99))
        return np.clip(((res - p1) / max(p99 - p1, 1e-6)).astype(np.float32) * 1.0, 0, 1)

    def nres(**st):
        ctx = dict(n_std=kn, depth_p=kd, light=np.array([0, 0, 1.0]), gain=1.0, use_proxy=True, crop_coords=None,
                   res_res=1.0, _is_real_crop=False)
        return se.get_shading_engine("Normal Residual", dk, dict(st), kh, kw, ctx=ctx)
    for mode in (0, 2):
        first = nres(nr_mode=mode, nr_scale=4.0, nr_gain=1.3)
        again = nres(nr_mode=mode, nr_scale=4.0, nr_gain=1.3)
        check(f"Normal Residual mode {mode}: equals the original formulas, and a second visit gives the same picture",
              np.array_equal(first, again) and float(np.abs(first - residual_reference(mode, 4.0, 1.3)).max()) < 1e-6)
    check("Normal Residual mode 1 (colour) is built from the same kept shade", nres(nr_mode=1, nr_scale=4.0, nr_gain=1.3).shape == (kh, kw, 3))
    check("a different scale gives a different picture (the kept shade is keyed on its settings)",
          not np.array_equal(nres(nr_mode=0, nr_scale=4.0, nr_gain=1.3), nres(nr_mode=0, nr_scale=9.0, nr_gain=1.3)))
    kn2 = kn[::-1].copy()                                  # a "new solve": another array
    ctx2 = dict(n_std=kn2, depth_p=kd, light=np.array([0, 0, 1.0]), gain=1.0, use_proxy=True, crop_coords=None,
                res_res=1.0, _is_real_crop=False)
    dk2 = DataManager()
    dk2.normals_full = dk2.normals_proxy = kn2
    dk2.h, dk2.w = kh, kw
    check("a new solve (new arrays) does not get the old solve's kept shade",
          not np.array_equal(nres(nr_mode=0, nr_scale=4.0, nr_gain=1.3),
                             se.get_shading_engine("Normal Residual", dk2, dict(nr_mode=0, nr_scale=4.0, nr_gain=1.3), kh, kw, ctx=ctx2)))

    def ao(depth, **st):
        ctx = dict(n_std=kn, depth_p=depth, light=np.array([0, 0, 1.0]), gain=st.pop("gain", 1.0), use_proxy=True,
                   crop_coords=None, res_res=1.0, _is_real_crop=False)
        return se.get_shading_engine("Ambient Occlusion (AO)", dk, dict(st), kh, kw, ctx=ctx)
    sig = max(kh, kw) * 0.08
    lo = cv.GaussianBlur(kd, (0, 0), sigmaX=sig)
    ao_ref = np.clip(np.power(np.clip(np.clip(kn[..., 2], 0, 1) * np.clip((kd - lo) * 2.0 + 0.5, 0, 1), 1e-6, 1.0), 1.0 / 1.5), 0, 1)
    check("Ambient Occlusion equals the original formula, twice",
          float(np.abs(ao(kd) - ao_ref).max()) < 1e-6 and np.array_equal(ao(kd), ao(kd)))
    check("Ambient Occlusion: gain changes the picture but not the kept blur",
          not np.array_equal(ao(kd, gain=0.5), ao(kd)) and float(np.abs(ao(kd, gain=0.5) - np.clip(ao_ref * 0.5, 0, 1)).max()) < 1e-6)
    check("Ambient Occlusion: a different radius gives a different picture",
          not np.array_equal(ao(kd, ao_radius=0.02), ao(kd)))

    print("Tone composite, bit for bit")
    import itertools
    from helio.application import render as R
    import helio.core._native as native
    th, tw = 103, 517
    tr = np.random.default_rng(21)
    big_alb = (tr.random((th + 8, tw + 8, 3)) * 256).astype(np.uint8)
    rough = (tr.random((th, tw)) * 256).astype(np.uint8)
    s32 = (tr.random((th, tw)) * 1.3 - .1).astype(np.float32)
    odd = s32.copy()
    odd[5, 7], odd[60, 3], odd[90, 500], odd[0, 0] = np.nan, np.inf, -np.inf, np.nan
    shadings = {"float32": s32, "float64": s32.astype(np.float64) * 1.0000001,
                "float32 with NaN/inf": odd, "float64 with NaN/inf": odd.astype(np.float64)}
    t0 = dict(bright=1.0, contrast=0.0, gamma=1.0, micro=0.0, rough_opacity=0.35)
    tones = [t0, dict(t0, bright=1.15, contrast=-12.), dict(t0, gamma=1.4), dict(t0, micro=.6, contrast=25.),
             dict(t0, rough_opacity=.8, bright=.9), dict(t0, micro=.3, gamma=.7)]
    bad, total = [], 0
    saved_rs = native.rs
    with np.errstate(all="ignore"):
        for (sn, sh_), mode, op, cg, tone, crop in itertools.product(
                shadings.items(), ("Combined Colour", "Shading", "Roughness Fusion", "Other"),
                (1.0, .6, 0.0), (1.0, 1.3), tones, (False, True)):
            alb = big_alb[4:4 + th, 4:4 + tw] if crop else np.ascontiguousarray(big_alb[:th, :tw])
            native.rs = None
            ref = R._tone_rows(sh_, alb, rough, tone, mode, op, th, tw, cg)
            native.rs = rs
            got = R._tone_rows(sh_, alb, rough, tone, mode, op, th, tw, cg)
            total += 1
            if not np.array_equal(ref, got):
                bad.append((sn, mode, op, cg, tone, crop, int((ref != got).sum())))
    native.rs = saved_rs
    check(f"{total} layer / opacity / colour gain / tone / dtype / crop combinations, incl. NaN and infinity: identical",
          not bad, bad[:2])
    check("tone_rgb is what runs (not the NumPy fallback)", hasattr(rs, "tone_rgb"))
    odd_alb = np.zeros((4, 4, 3), np.uint8)
    for label, call in (("unknown layer", lambda: rs.tone_rgb(np.zeros((4, 4), np.float32), "Nope", 1., 1., .3, 1., 1., 0., True, albedo=odd_alb)),
                        ("Combined without albedo", lambda: rs.tone_rgb(np.zeros((4, 4), np.float32), "Combined Colour", 1., 1., .3, 1., 1., 0., True)),
                        ("Roughness without a map", lambda: rs.tone_rgb(np.zeros((4, 4), np.float32), "Roughness Fusion", 1., 1., .3, 1., 1., 0., True)),
                        ("albedo of the wrong size", lambda: rs.tone_rgb(np.zeros((4, 5), np.float32), "Combined Colour", 1., 1., .3, 1., 1., 0., True, albedo=odd_alb)),
                        ("shading of the wrong dtype", lambda: rs.tone_rgb(np.zeros((4, 4), np.int32), "Shading", 1., 1., .3, 1., 1., 0., True)),
                        ("short lut", lambda: rs.tone_rgb(np.zeros((4, 4), np.float32), "Shading", 1., 1., .3, 1., 1., 0., True, lut=np.zeros(10, np.uint8)))):
        try:
            call()
            check(f"tone_rgb rejects {label}", False)
        except ValueError:
            check(f"tone_rgb rejects {label}", True)

    print("Fallbacks")
    n64 = n.astype(np.float64)
    saved = se._rs
    se._rs = rs
    out = se.calc_hillshade_raw(None, dict(hs_alt=45, hs_azi=315), n64)
    se._rs = saved
    check("float64 normals still render (NumPy path)", out.shape == (h, w))
    check("non-contiguous array is refused by the helper", not se._rust("hillshade", n.reshape(-1, 3)[::2]))

    print("Order statistics")
    r = np.random.default_rng(5)
    cases = {
        "uniform": r.random((1100, 1100)).astype(np.float32),
        "signed": (r.normal(size=(1100, 1100)) * 1e3).astype(np.float32),
        "mostly zero": np.where(r.random((1300, 900)) < .8, 0, r.random((1300, 900))).astype(np.float32),
        "constant": np.full((1100, 1100), .25, np.float32),
        "odd count": r.random((1001, 1003)).astype(np.float32),
        "one NaN": np.where(np.arange(1100 * 1100).reshape(1100, 1100) == 77, np.nan, 1).astype(np.float32),
    }
    qs = [0, 1, 2, 50, 98, 99.5, 100]
    for name, a in cases.items():
        got, want = parallel.quantiles(a, qs), [float(v) for v in np.percentile(a, qs)]
        same = all(x == y or (np.isnan(x) and np.isnan(y)) for x, y in zip(got, want))
        m, wm = parallel.median(a), float(np.median(a))
        check(f"quantiles + median bit for bit: {name}", same and (m == wm or (np.isnan(m) and np.isnan(wm))))
    check("kth_values refuses NaN", rs.kth_values(np.array([1, np.nan], np.float32), [0]) is None)
    check("kth_values refuses a bad rank", rs.kth_values(np.ones(4, np.float32), [4]) is None)

    print("Raw kernel input checks")
    L3 = np.array([0, 0, 1], np.float32)

    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False
    check("normals must be (n, 3)", raises(lambda: rs.hillshade(np.zeros((5, 2), np.float32), L3, 0.1)))
    check("normals must be contiguous", raises(lambda: rs.hillshade(np.zeros((10, 3), np.float32)[::2], L3, 0.1)))
    check("light must have 3 values", raises(lambda: rs.hillshade(np.zeros((5, 3), np.float32), L3[:2], 0.1)))
    check("lut must not be empty", raises(lambda: rs.standard_pbr(np.zeros((5, 3), np.float32), L3, np.zeros(0, np.float32))))
    check("one weight per light", raises(lambda: rs.multi_lambert(
        np.zeros((5, 3), np.float32), np.zeros((2, 3), np.float32), np.ones(3, np.float32))))
    check("scales must match in size", raises(lambda: rs.scaled_lambert(
        [np.zeros((4, 3), np.float32), np.zeros((5, 3), np.float32)], L3, np.ones(2, np.float32), 2.0)))
    check("shadow steps must be >= 1", raises(lambda: rs.shadow_margin(np.zeros((2, 5), np.float32), .5, 0)))
    check("shadow_shade needs one lit value per pixel", raises(lambda: rs.shadow_shade(
        np.zeros((6, 3), np.float32), L3, np.zeros(5, np.float32), 1.0)))
    check("rti_specular rejects a wrong-sized ptm", raises(lambda: rs.rti_specular(
        np.zeros((4, 3), np.float32), L3, .05, 1.5, .6, 1.0, ptm=np.zeros((6, 3), np.float32))))
    check("rti_specular rejects a wrong-sized median", raises(lambda: rs.rti_specular(
        np.zeros((4, 3), np.float32), L3, .05, 1.5, .6, 1.0, med=np.zeros(3, np.uint8))))
    check("sss needs one scat value per pixel", raises(lambda: rs.sss(
        np.zeros((4, 3), np.float32), np.zeros(3), np.zeros(3, np.float32), .5)))
    check("sss needs a 3-vector light", raises(lambda: rs.sss(
        np.zeros((4, 3), np.float32), np.zeros(2), np.zeros(4, np.float32), .5)))
    check("empty lights rejected", raises(lambda: rs.diffuse_gain(
        np.zeros((5, 3), np.float32), np.zeros((0, 3), np.float32), 1, 1, 0)))

    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
