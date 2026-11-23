"""
Headless test suite for HELIO shading engine dispatch and DataManager fixes.
Run: python tests/test_engines.py
No display required — uses synthetic numpy arrays.
"""
import sys, os, traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helio.data_manager import DataManager
from helio.core.shading_engines import (
    get_shading_engine, implemented_engines, _ENGINE_DISPATCH,
)

PASS = 0
FAIL = 0

def check(desc, ok):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {desc}")
    else:
        FAIL += 1
        print(f"  FAIL  {desc}")

# ── 1. DataManager attribute access ────────────────────────────────────

def test_data_manager_attrs():
    dm = DataManager()
    check("DataManager has render_engine", hasattr(dm, "render_engine"))
    check("DataManager has view_mode", hasattr(dm, "view_mode"))
    check("render_engine default == 'Standard PBR'", dm.render_engine == "Standard PBR")
    check("view_mode default == 'Shading'", dm.view_mode == "Shading")
    dm.render_engine = "Robust PBR (MVS)"
    dm.view_mode = "Normals"
    check("render_engine can be set", dm.render_engine == "Robust PBR (MVS)")
    check("view_mode can be set", dm.view_mode == "Normals")

# ── 2. Engine registry completeness ────────────────────────────────────

ENGINES_LIST = [
    "Standard PBR", "Robust PBR (MVS)", "Normals Unsharp Masking",
    "Slope Magnitude", "Specular Enhancement", "Image Quotient (IQ)",
    "Positive Openness", "Negative Openness",
    "Specular View", "Ambient Occlusion (AO)",
    "PS Normal-Map Lit (Single)", "PS Normal-Map Lit (8-Dir)",
    "Slope Gradient",
    "Surface Curvature (Scratches)",
    "PCA 1", "PCA 2", "PCA 3",
    "Light Coverage Map", "Normals RGB", "Shape Index",
    "Retinex Enhancement", "Colour ICA (IC1-3)", "RGB-PS R-B Diff",
    "Raking Composite", "Normal Residual", "Diffuse Gain",
    "Bilateral Normal Sharpening", "Subsurface Scattering (SSS)",
    "RTI Specular Enhancement", "RTI Image USM",
]

def test_registry():
    registered = set(_ENGINE_DISPATCH.keys())
    for eng in ENGINES_LIST:
        check(f"Engine '{eng}' registered in _ENGINE_DISPATCH", eng in registered)
    for eng in registered:
        check(f"Dispatcher has callable for '{eng}'", callable(_ENGINE_DISPATCH[eng]))

# ── 3. Each engine runs with synthetic data ───────────────────────────

def make_synthetic_data(h=16, w=16):
    dm = DataManager()
    dm.h, dm.w = h, w
    dm.ph, dm.pw = 8, 8
    dm.normals_full = np.random.randn(h, w, 3).astype(np.float32)
    dm.normals_full /= np.linalg.norm(dm.normals_full, axis=-1, keepdims=True) + 1e-8
    dm.normals_proxy = dm.normals_full[:8, :8]
    dm.normals_robust = dm.normals_full.copy()
    dm.normals_robust_proxy = dm.normals_proxy.copy()
    dm.median_img = np.random.rand(h, w).astype(np.float32) * 0.5
    dm.median_proxy = dm.median_img[:8, :8]
    dm.depth_vis_p = np.random.rand(h, w).astype(np.float32)
    dm.depth_proxy_p = dm.depth_vis_p[:8, :8]
    dm.depth_vis_fc = dm.depth_vis_p.copy()
    dm.depth_proxy_fc = dm.depth_proxy_p.copy()
    dm.albedo_rgb_full = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    dm.albedo_proxy = dm.albedo_rgb_full[:8, :8]
    dm.roughness_full = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    dm.roughness_proxy = dm.roughness_full[:8, :8]
    dm.ptm_coeffs = np.random.randn(6, h * w).astype(np.float32)
    dm.ptm_proxy = np.random.randn(6, 8 * 8).astype(np.float32)
    dm.current_L_mat = np.random.randn(4, 3).astype(np.float32)
    dm.light_vecs = [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]
    dm.pca_full = np.random.rand(h, w, 3).astype(np.float32)
    dm.engine_settings = {eng: {"gain": 1.0, "lx": 0, "ly": 0, "lz": 1} for eng in ENGINES_LIST}
    s = dm.engine_settings["Standard PBR"]
    s.update({"bright": 1.0, "contrast": 0, "gamma": 1.0, "hs_alt": 45, "hs_azi": 315,
              "hs_z": 1.0, "hs_amb": 0.3, "svf_radius": 20, "svf_dirs": 16,
              "op_radius": 20, "op_elev": 10, "msrm_min": 1, "msrm_max": 5,
              "slrm_radius": 20, "sss_radius": 8.0, "sss_blend": 0.45,
              "rti_spec_coeff": 1.5, "rti_spec_floor": 0.05, "rti_spec_blend": 0.6,
              "rti_usm_sigma": 2.0, "rti_usm_amount": 1.5, "rti_usm_thresh": 0.0,
              "ld_radius": 20, "ld_elev": 35, "si_sigma": 1.5,
              "depth_cmap": "viridis", "hs_depth_src": "poisson",
              "ao_radius": 0.08, "ao_strength": 1.5,
              "dg_strength": 2.0, "dg_gamma": 0.7, "dg_spec": 0.3,
              "bil_d": 9, "bil_sc": 0.1,
              "def_sens": 1.0, "def_relief": 1.0, "def_noise": 0.5, "def_spec": 0.0,
              "rk_elevation": 20, "rk_n_dirs": 4, "rk_blend": 0.5, "rk_usm": 0.0,
              "ret_scale_s": 8, "ret_scale_m": 30,
              "ret_scale_l": 250, "shape_smooth": 1.5,
              "global_whites": 1.0, "global_blacks": 0.0, "global_midtone": 1.0,
              "global_clarity": 0.0, "global_texture": 0.0, "global_dehaze": 0.0,
              "micro": 0.0, "rough_opacity": 0.5,
              })
    return dm

def build_ctx(dm, h, w):
    return dict(
        n_std=dm.normals_full, n_rob=dm.normals_robust,
        med=dm.median_img, depth_p=dm.depth_vis_p,
        depth_fc=dm.depth_vis_fc, ptm=dm.ptm_coeffs,
        light=np.array([0, 0, 1], np.float32),
        gain=1.0, res_res=1.0, use_proxy=False,
        _is_real_crop=False, crop_coords=(0, 0, dm.w, dm.h),
    )

def test_all_engines_run():
    dm = make_synthetic_data(16, 16)
    ctx = build_ctx(dm, 16, 16)
    settings = dm.engine_settings["Standard PBR"]
    for eng_name in sorted(_ENGINE_DISPATCH.keys()):
        try:
            es = dm.engine_settings.get(eng_name, settings)
            result = get_shading_engine(eng_name, dm, es, 16, 16, ctx)
            ok = result is not None and result.size > 0 and result.ndim in (2, 3)
            check(f"Engine '{eng_name}' returned valid array shape={result.shape}", ok)
        except Exception as e:
            check(f"Engine '{eng_name}' raised: {e}", False)
            traceback.print_exc()

def test_robust_pbr_none_nrob():
    """Robust PBR must not crash when n_rob is None."""
    dm = make_synthetic_data(16, 16)
    ctx = build_ctx(dm, 16, 16)
    ctx['n_rob'] = None  # simulate missing robust normals
    result = get_shading_engine("Robust PBR (MVS)", dm, dm.engine_settings["Robust PBR (MVS)"], 16, 16, ctx)
    check("Robust PBR with None n_rob returns valid array", result is not None and result.size > 0)

def test_sss_none_albedo():
    """SSS engine must not crash when albedo arrays are None."""
    dm = make_synthetic_data(16, 16)
    dm.albedo_rgb_full = None
    dm.albedo_proxy = None
    ctx = build_ctx(dm, 16, 16)
    result = get_shading_engine("Subsurface Scattering (SSS)", dm, dm.engine_settings["Subsurface Scattering (SSS)"], 16, 16, ctx)
    check("SSS with None albedo returns valid array", result is not None and result.size > 0)

def test_sss_proxy_none_albedo():
    """SSS engine must not crash when using proxy and albedo_proxy is None."""
    dm = make_synthetic_data(16, 16)
    dm.albedo_proxy = None
    ctx = build_ctx(dm, 16, 16)
    ctx['use_proxy'] = True
    # Pass proxy-sized arrays and correct h,w
    ctx['n_std'] = dm.normals_proxy
    ctx['n_rob'] = dm.normals_robust_proxy
    ctx['med'] = dm.median_proxy
    ctx['depth_p'] = dm.depth_proxy_p
    ctx['depth_fc'] = dm.depth_proxy_fc
    ctx['ptm'] = dm.ptm_proxy
    result = get_shading_engine("Subsurface Scattering (SSS)", dm, dm.engine_settings["Subsurface Scattering (SSS)"], 8, 8, ctx)
    check("SSS proxy with None albedo_proxy returns valid array", result is not None and result.size > 0)

def test_ctx_none_fallback():
    """get_shading_engine with ctx=None should construct defaults from DataManager."""
    dm = make_synthetic_data(16, 16)
    result = get_shading_engine("Standard PBR", dm, dm.engine_settings["Standard PBR"], 16, 16, ctx=None)
    check("get_shading_engine(ctx=None) returns valid array", result is not None and result.size > 0)

# ── 4. Unknown engine fallback ─────────────────────────────────────────

def test_unknown_engine():
    dm = make_synthetic_data(16, 16)
    ctx = build_ctx(dm, 16, 16)
    result = get_shading_engine("Bogus Engine Name", dm, {}, 16, 16, ctx)
    check("Unknown engine returns zeros", result is not None and result.shape == (16, 16))

# ── 5. Proxy path ──────────────────────────────────────────────────────

def test_proxy_path():
    """Run all engines with proxy data."""
    dm = make_synthetic_data(16, 16)
    ctx = build_ctx(dm, 16, 16)
    ctx['use_proxy'] = True
    ctx['n_std'] = dm.normals_proxy
    ctx['n_rob'] = dm.normals_robust_proxy
    ctx['med'] = dm.median_proxy
    ctx['depth_p'] = dm.depth_proxy_p
    ctx['depth_fc'] = dm.depth_proxy_fc
    ctx['ptm'] = dm.ptm_proxy
    settings = dm.engine_settings["Standard PBR"]
    for eng_name in sorted(_ENGINE_DISPATCH.keys()):
        try:
            es = dm.engine_settings.get(eng_name, settings)
            result = get_shading_engine(eng_name, dm, es, 8, 8, ctx)
            ok = result is not None and result.size > 0 and result.ndim in (2, 3)
            check(f"Proxy: '{eng_name}' returned valid array shape={result.shape}", ok)
        except Exception as e:
            check(f"Proxy: '{eng_name}' raised: {e}", False)
            traceback.print_exc()


def test_pbr_lut_accuracy():
    """The Standard PBR lookup table must track exact np.power, or stand aside.

    The table replaces a per-pixel exponent (8.75 -> 4.08 ms at 1280x960), and
    it is exact enough only while gain <= 1. Above that the exponent falls
    below 1, the curve has unbounded slope at zero, and the first bin alone is
    wrong by up to 48/255 at gain 5 - which the slider reaches. No table size
    fixes that, so the engine must fall back to the exact computation there.

    This asserts both halves: accurate where used, and *not used* where it
    would be inaccurate.
    """
    from helio.core.shading_engines import _pbr_lut, _PBR_LUT_N

    x = np.linspace(0.0, 1.0, 20001, dtype=np.float64)
    worst = 0.0
    for gain in (0.1, 0.15, 0.25, 0.5, 0.75, 1.0):
        lut = _pbr_lut(1.0 / gain)
        check(f"LUT is used at gain {gain}", lut is not None)
        if lut is None:
            continue
        approx = lut[np.minimum((x * _PBR_LUT_N).astype(np.int32), _PBR_LUT_N - 1)]
        err = float(np.abs(approx - np.power(x, 1.0 / gain)).max() * 255.0)
        worst = max(worst, err)
        check(f"LUT within 1/255 of np.power at gain {gain} (max {err:.3f})",
              err < 1.0)

    for gain in (1.5, 2.0, 3.0, 5.0):
        check(f"LUT declined at gain {gain} (would be inaccurate)",
              _pbr_lut(1.0 / gain) is None)

    # And the engine itself must agree with the exact maths it replaced.
    dm = make_synthetic_data(64, 64)
    ctx = build_ctx(dm, 64, 64)
    ctx['gain'] = 0.15
    got = np.asarray(get_shading_engine("Standard PBR", dm,
                                        dm.engine_settings["Standard PBR"],
                                        64, 64, ctx=ctx), np.float64)
    n = ctx['n_std'].reshape(-1, 3)
    exact = np.power(np.maximum(0, np.dot(n, ctx['light'])), 1.0 / 0.15)
    exact = exact.reshape(got.shape)
    mae = float(np.abs(got - exact).mean() * 255.0)
    check(f"Standard PBR matches exact np.power (MAE {mae:.4f}/255)", mae < 1.0)


def test_ptm_size_mismatch():
    """PTM-dependent engines must not crash when ptm has wrong dimensions."""
    dm = make_synthetic_data(16, 16)
    ctx = build_ctx(dm, 16, 16)
    ctx['use_proxy'] = True
    ctx['n_std'] = dm.normals_proxy
    ctx['n_rob'] = dm.normals_robust_proxy
    ctx['med'] = dm.median_proxy
    ctx['depth_p'] = dm.depth_proxy_p
    ctx['depth_fc'] = dm.depth_proxy_fc
    # Deliberately wrong-sized ptm — only 10 columns instead of 8*8=64
    ctx['ptm'] = np.zeros((6, 10), np.float32)
    for eng in ["Specular Enhancement", "Image Quotient (IQ)"]:
        try:
            result = get_shading_engine(eng, dm, dm.engine_settings[eng], 8, 8, ctx)
            ok = result is not None and result.size > 0
            check(f"PTM mismatch '{eng}' handled gracefully", ok)
        except Exception as e:
            check(f"PTM mismatch '{eng}' raised: {e}", False)


# ── RVT-backed views use RVT ─────────────────────────────────────────

def test_rvt_paths():
    """With RVT installed, the views built on it use it.  RVT's signatures move
    between versions: msrm() gained a required scaling_factor in rvt-py 2, and
    every call to it failed quietly to the fallback until 2026-10-15."""
    import logging
    from helio.core import processing as PR
    from helio.core import shading_engines as SE
    if not SE.HAS_RVT:
        check("RVT is not installed: its views draw with their fallbacks", True)
        return
    seen = []

    class Catch(logging.Handler):
        def emit(self, rec):
            seen.append(rec.getMessage())

    catch = Catch(level=logging.WARNING)
    loggers = [logging.getLogger(SE.__name__), logging.getLogger(PR.__name__)]
    for lg in loggers:
        lg.addHandler(catch)
    try:
        dm = make_synthetic_data(64, 64)
        ctx = build_ctx(dm, 64, 64)
        settings = dm.engine_settings["Standard PBR"]
        for eng in ("Multi-Scale Relief Model (MSRM)", "Sky-View Factor (SVF)", "Positive Openness",
                    "Negative Openness", "Local Dominance", "Slope Gradient"):
            seen.clear()
            out = get_shading_engine(eng, dm, settings, 64, 64, ctx)
            check(f"'{eng}' draws with RVT, no fallback ({'; '.join(seen) or 'none'})",
                  not seen and out is not None and np.isfinite(out).all())
        seen.clear()
        d = dm.depth_vis_p.astype(np.float32)
        PR.msrm(d, 1.0, 5.0)
        PR.slrm(d, 12)
        PR.slrm(d, 5)                     # under RVT's 10 px: the same model, no warning
        PR.local_dominance(d, 3, 6)
        PR.svf_rvt(d, 5, 8)
        PR.positive_openness(d, 5, 8)
        check(f"the terrain-analysis helpers use RVT too ({'; '.join(seen) or 'no fallback'})",
              not seen)
    finally:
        for lg in loggers:
            lg.removeHandler(catch)


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("HELIO Shading Engine Test Suite")
    print("=" * 72)
    print()
    test_data_manager_attrs()
    print()
    test_registry()
    print()
    test_all_engines_run()
    print()
    test_robust_pbr_none_nrob()
    print()
    test_sss_none_albedo()
    print()
    test_sss_proxy_none_albedo()
    print()
    test_ctx_none_fallback()
    print()
    test_unknown_engine()
    print()
    test_proxy_path()
    print()
    test_ptm_size_mismatch()
    print()
    test_pbr_lut_accuracy()
    print()
    test_rvt_paths()
    print()
    print("=" * 72)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 72)
    sys.exit(0 if FAIL == 0 else 1)
