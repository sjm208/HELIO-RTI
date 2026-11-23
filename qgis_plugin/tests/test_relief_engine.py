"""Unit tests for relief_engine - run with:  python3 -m pytest tests/"""

import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "helio_relief"))
import relief_engine as eng  # noqa: E402


def synthetic_surface(h=192, w=256, seed=0):
    """Tilted sheet with a broad dome, an incised line pattern and grain."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 0.02 * x + 0.01 * y
    z += 8.0 * np.exp(-((x - 90) ** 2 + (y - 100) ** 2) / (2 * 30.0 ** 2))
    z += 1.5 * np.sin(x / 5.0) * np.exp(-((y - 45) / 12.0) ** 2)
    z += rng.normal(0, 0.03, (h, w))
    return z.astype(np.float32)


def test_bandlimited_interpolator_is_partition_of_unity():
    f = np.linspace(0, 1, 101, dtype=np.float32)
    s = eng._bandlimited_spline(f)
    assert s[0] == pytest.approx(1.0)
    assert s[-1] == pytest.approx(0.0)
    np.testing.assert_allclose(s + eng._bandlimited_spline(1 - f), 1.0, atol=1e-6)
    # Brown: "very nearly matches the cubic spline"
    fs = f[1:-1].astype(np.float64)
    spline = 3 * np.sinc(fs) ** 4 / (2 + np.cos(2 * np.pi * fs))
    assert np.max(np.abs(spline - s[1:-1])) < 0.01


def test_texture_shading_zero_detail_is_identity():
    z = synthetic_surface()
    np.testing.assert_allclose(eng.texture_shade(z, detail=0.0), z, atol=1e-3)


def test_texture_shading_is_zero_mean_and_ridge_positive():
    z = synthetic_surface()
    t = eng.texture_shade(z, detail=0.5)
    assert abs(float(t.mean())) < 0.08 * float(np.abs(t).mean())   # zero-mean on the mirror-padded domain
    assert t[100, 90] > 0          # dome top is higher than its surroundings
    flat = np.full((64, 64), 5.0, np.float32)
    assert np.abs(eng.texture_shade(flat, detail=0.8)).max() < 1e-4


def test_frequency_split_reconstructs_exactly():
    z = synthetic_surface()
    bands = eng.frequency_split(z, [64, 32, 16, 8])
    np.testing.assert_allclose(sum(bands.values()), z, atol=2e-3)


def test_eq_with_unit_gains_is_identity():
    spec = eng.Spectrum(np.zeros((40, 50), np.float32))
    m = eng.eq_multiplier(spec.fy, spec.fx, [30, 10, 5, 2], [1, 1, 1, 1, 1])
    np.testing.assert_allclose(m, 1.0, atol=1e-6)


def test_dct_gaussian_matches_scipy():
    from scipy.ndimage import gaussian_filter
    z = synthetic_surface()
    ref = gaussian_filter(z, 12, mode="reflect")
    np.testing.assert_allclose(eng.gaussian_blur(z, 12), ref, atol=5e-3)


def test_hillshade_orientation():
    y, x = np.mgrid[0:32, 0:32].astype(np.float32)
    rises_east = x * 0.5                      # faces west
    assert eng.hillshade(rises_east, azimuth=270)[16, 16] > eng.hillshade(rises_east, azimuth=90)[16, 16]
    rises_south = y * 0.5                     # row 0 is north, so this faces north
    assert eng.hillshade(rises_south, azimuth=0)[16, 16] > eng.hillshade(rises_south, azimuth=180)[16, 16]


def _gdal_reference_hillshade(z, multi=False):
    """gdaldem hillshade of z via osgeo (QGIS Python) or the gdaldem CLI."""
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "dem.tif"), os.path.join(tmp, "hs.tif")
        try:
            from osgeo import gdal
            gdal.UseExceptions()
            ds = gdal.GetDriverByName("GTiff").Create(src, z.shape[1], z.shape[0], 1, gdal.GDT_Float32)
            ds.SetGeoTransform((0, 1, 0, z.shape[0], 0, -1))
            ds.GetRasterBand(1).WriteArray(z)
            ds = None
            opts = dict(altitude=45, multiDirectional=True) if multi else dict(azimuth=315, altitude=45)
            gdal.DEMProcessing(dst, src, "hillshade", **opts)
            return gdal.Open(dst).ReadAsArray().astype(np.float32)
        except ImportError:
            pass
        rasterio = pytest.importorskip("rasterio")
        from rasterio.transform import from_origin
        if shutil.which("gdaldem") is None:
            pytest.skip("neither osgeo nor gdaldem available")
        with rasterio.open(src, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1,
                           dtype="float32", transform=from_origin(0, z.shape[0], 1, 1)) as ds:
            ds.write(z, 1)
        if subprocess.run(["gdaldem", "hillshade", src, dst, "-az", "315", "-alt", "45", "-q"]
                          + (["-multidirectional"] if multi else [])).returncode:
            pytest.skip("gdaldem CLI is broken on this machine")
        with rasterio.open(dst) as ds:
            return ds.read(1).astype(np.float32)


def test_hillshade_matches_gdaldem():
    z = synthetic_surface() * 4.0
    ref = _gdal_reference_hillshade(z)
    mine = 1.0 + 254.0 * eng.hillshade(z, azimuth=315, altitude=45)
    inner = (slice(2, -2), slice(2, -2))
    assert np.max(np.abs(mine[inner] - ref[inner])) <= 1.01


def test_sky_view_factor():
    flat = np.zeros((48, 48), np.float32)
    np.testing.assert_allclose(eng.sky_view_factor(flat, radius_px=8), 1.0)
    pit = flat.copy()
    pit[20:28, 20:28] = -5.0
    svf = eng.sky_view_factor(pit, radius_px=8)
    assert svf[24, 24] < 0.9 and svf[2, 2] == pytest.approx(1.0)


@pytest.mark.parametrize("mode", list(eng.RENDER_MODES))
def test_render_modes_produce_valid_images(mode):
    z = synthetic_surface()
    img, stats = eng.render(z, params={"mode": mode, "occlusion": 0.3, "flatten_sigma": 20})
    assert img.shape[:2] == z.shape
    assert np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
    assert img.std() > 0.01


def test_studio_flat_surface_renders_at_base_tone():
    flat = np.full((64, 64), 3.0, np.float32)
    img, _ = eng.render(flat, params={"mode": "studio", "base_tone": 0.6, "occlusion": 0.5})
    np.testing.assert_allclose(img, 0.6, atol=1e-4)


def test_clean_relief_flat_surface_renders_at_its_brightness():
    flat = np.full((64, 80), 3.0, np.float32)
    img, _ = eng.render(flat, params={"mode": "clean", "clean_mid": 0.6})
    np.testing.assert_allclose(img, 0.6, atol=1e-4)


def test_clean_relief_ignores_tilt_and_height_units():
    z = synthetic_surface()
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float32)
    base, _ = eng.render(z, params={"mode": "clean"})
    tilted, _ = eng.render(z + 0.3 * xx - 0.2 * yy, params={"mode": "clean"})
    scaled, _ = eng.render(z * 40.0, params={"mode": "clean"})
    assert np.abs(tilted - base).max() < 0.01
    assert np.abs(scaled - base).max() < 0.01


def test_clean_relief_removes_integration_bowl():
    # integrating photometric-stereo normals bends a depth map into a bowl far taller than its relief
    z = synthetic_surface()
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bowl = 0.02 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    p = {"mode": "clean", "clean_flatten": 12}
    base, st = eng.render(z, params=p)
    bowled, _ = eng.render(z + bowl, params=p, stats=st)
    plain, _ = eng.render(z + bowl, params=dict(p, clean_flatten=0), stats=st)
    m = 36
    inner = (slice(m, -m), slice(m, -m))
    assert np.abs(bowled - base)[inner].mean() < 0.01
    assert np.abs(plain - base)[inner].mean() > 5 * np.abs(bowled - base)[inner].mean()


def test_clean_relief_never_black_or_white():
    z = synthetic_surface() * 50.0
    img, _ = eng.render(z, params={"mode": "clean", "clean_relief": 75, "clean_contrast": 3.0})
    assert img.min() > 0.1 and img.max() < 0.9 and img.std() > 0.05


def test_clean_relief_highlight_gain_brightens_lit_faces_only():
    z = synthetic_surface()
    base, st = eng.render(z, params={"mode": "clean", "clean_highlight": 1.0})
    bright, _ = eng.render(z, params={"mode": "clean", "clean_highlight": 2.0}, stats=st)
    lit, dark = base > 0.6 + 1e-3, base < 0.6 - 1e-3
    assert lit.mean() > 0.05 and (bright[lit] > base[lit]).mean() > 0.99
    np.testing.assert_allclose(bright[dark], base[dark], atol=1e-6)


def test_clean_relief_lit_from_every_side_is_direction_free():
    z = synthetic_surface(96, 96)
    p = {"mode": "clean", "clean_allround": 1.0, "clean_lights": 16, "azimuth": 0.0}
    img, st = eng.render(z, params=p)
    rot, _ = eng.render(np.ascontiguousarray(np.rot90(z)), params=p, stats=st)
    # the outermost ring is gdaldem's edge extrapolation, which is not rotation-symmetric at the corners
    np.testing.assert_allclose(np.rot90(img)[2:-2, 2:-2], rot[2:-2, 2:-2], atol=1e-4)


def test_clean_relief_tiles_match_whole_render():
    z = synthetic_surface(160, 320)
    params = {"mode": "clean", "clean_flatten": 12, "clean_smooth": 2.0, "clean_fine": 0.5}
    plane = eng.fit_plane(z)
    whole, stats = eng.render(z, params=params, plane=plane)
    margin = eng.spectral_margin_px(params)
    mid = 160
    left, _ = eng.render(z[:, :mid + margin], params=params, stats=stats, plane=plane)
    right, _ = eng.render(z[:, mid - margin:], params=params, stats=stats, plane=plane, origin=(mid - margin, 0))
    stitched = np.hstack([left[:, :mid], right[:, margin:]])
    assert np.max(np.abs(stitched - whole)[4:-4]) < 0.02


def test_clean_relief_smoothing_removes_noise_keeps_walls():
    rng = np.random.default_rng(1)
    xx = np.mgrid[0:96, 0:96][1]
    z = (5.0 * ((xx // 16) % 2)).astype(np.float32)          # tooled walls every 16 px
    noisy = z + rng.normal(0, 0.3, z.shape).astype(np.float32)
    p = {"mode": "clean", "clean_flatten": 0, "clean_fine": 0.0}
    rough, _ = eng.render(noisy, params=dict(p, clean_smooth=0))
    smooth, _ = eng.render(noisy, params=dict(p, clean_smooth=3.0, clean_edge=2.0))
    plateau = (slice(8, -8), slice(20, 28))
    assert smooth[plateau].std() < 0.5 * rough[plateau].std()
    wall = smooth[8:-8, 31:33].mean()
    assert abs(wall - smooth[plateau].mean()) > 0.1


def test_nodata_is_masked_not_propagated():
    z = synthetic_surface()
    z[10:20, 10:20] = np.nan
    img, _ = eng.render(z, params={"mode": "studio"})
    assert np.isfinite(img).all() and (img[12:18, 12:18] == 0).all()


def test_tiled_render_matches_whole_render():
    z = synthetic_surface(160, 320)
    params = {"mode": "studio", "flatten_sigma": 10, "texture_mix": 0.4, "occlusion": 0.3, "svf_radius": 8,
              "shadows": 0.5, "shadow_reach": 20, "altitude": 20, "clarity": 0.3, "clarity_radius": 4,
              "curvature": 0.5, "slope_shade": 0.4, "detail_boost": 1.0, "detail_radius": 2}
    plane = eng.fit_plane(z)
    whole, stats = eng.render(z, params=params, plane=plane)
    margin = eng.spectral_margin_px(params)
    mid = 160
    left, _ = eng.render(z[:, :mid + margin], params=params, stats=stats, plane=plane)
    right, _ = eng.render(z[:, mid - margin:], params=params, stats=stats, plane=plane, origin=(mid - margin, 0))
    stitched = np.hstack([left[:, :mid], right[:, margin:]])
    assert np.max(np.abs(stitched - whole)[4:-4]) < 0.02


def test_auto_z_factor_hits_target_slope():
    z = synthetic_surface()
    k = eng.auto_z_factor(z, target_deg=30, percentile=95)
    gx, gy = eng.horn_gradient(z * k)
    slope = np.degrees(np.arctan(np.percentile(np.hypot(gx, gy), 95)))
    assert slope == pytest.approx(30, abs=0.5)


def test_post_split_without_treatment_is_identity_and_ramp_overlay_shape():
    z = synthetic_surface()
    settings = dict(sigmas=[64, 32, 16, 8], median=0, preblur=0, detrend=False, denoise=0,
                    treat={k: dict(mode="None", radius=0, threshold=0) for k in eng.BAND_KEYS})
    zc = eng.condition_surface(z, settings)
    bands = eng.post_split(eng.frequency_split(zc, settings["sigmas"]), settings)
    np.testing.assert_allclose(sum(bands.values()), z, atol=2e-3)
    stops = [(0.0, (200, 30, 30, 170)), (0.5, (255, 255, 255, 0)), (1.0, (30, 160, 60, 170))]
    rgb = eng.ramp_overlay(bands["Mid_Freq"], np.full(z.shape, 0.5, np.float32), stops)
    assert rgb.shape == z.shape + (3,) and 0 <= rgb.min() and rgb.max() <= 1


def test_studio_effects_leave_a_flat_surface_at_base_tone():
    flat = np.full((64, 64), 3.0, np.float32)
    img, _ = eng.render(flat, params={"mode": "studio", "base_tone": 0.6, "occlusion": 0.5, "shadows": 0.8,
                                      "wrap": 0.3, "clarity": 0.5, "specular": 0.3, "curvature": 0.8,
                                      "slope_shade": 0.7, "detail_boost": 2.0})
    np.testing.assert_allclose(img, 0.6, atol=1e-4)


def test_cast_shadows_fall_away_from_the_light():
    z = np.zeros((64, 64), np.float32)
    z[:, 30:34] = 10.0                                   # north-south wall
    lit = eng.cast_shadows(z, 1.0, 1.0, azimuth=270, altitude=10, reach_px=60)   # light from the west
    assert lit[32, 45] < 0.5                             # east of the wall: in shadow
    assert lit[32, 15] == pytest.approx(1.0)             # west of the wall: lit


def test_multidirectional_matches_gdal():
    z = synthetic_surface() * 4.0
    ref = _gdal_reference_hillshade(z, multi=True)
    lights = [(a, 45, 1.0, eng.WHITE) for a in (225, 270, 315, 0)]
    lum, _ = eng._multi_light(eng.surface_normals(z), lights, "adaptive")
    inner = (slice(2, -2), slice(2, -2))
    assert np.max(np.abs(1.0 + 254.0 * lum[inner] - ref[inner])) <= 1.01


@pytest.mark.parametrize("preset", list(eng.MULTI_PRESETS))
def test_multi_shade_presets_are_neutral_on_a_flat_surface(preset):
    method, lights = eng.MULTI_PRESETS[preset]
    params = {"mode": "multi", "multi_method": method, "multi_lights": lights, "base_tone": 0.6, "texture_mix": 0.3}
    flat = np.full((48, 48), 1.0, np.float32)
    img, _ = eng.render(flat, params=params)
    np.testing.assert_allclose(img, 0.6, atol=1e-4)
    shaded, _ = eng.render(synthetic_surface(), params=params)
    assert shaded.std() > 0.01
    assert (shaded.ndim == 3) == eng.is_rgb(params)


def test_exaggerated_shading_is_neutral_flat_and_reveals_detail():
    params = {"mode": "exaggerated", "base_tone": 0.6}
    img, _ = eng.render(np.full((64, 64), 2.0, np.float32), params=params)
    np.testing.assert_allclose(img, 0.6, atol=1e-4)
    shaded, _ = eng.render(synthetic_surface(), params=params)
    assert shaded.std() > 0.03


def test_exaggerated_tiles_match_whole():
    z = synthetic_surface(160, 320)
    params = {"mode": "exaggerated", "exag_scales": 4, "curvature": 0.5}
    plane = eng.fit_plane(z)
    whole, stats = eng.render(z, params=params, plane=plane)
    m = eng.spectral_margin_px(params)
    left, _ = eng.render(z[:, :160 + m], params=params, stats=stats, plane=plane)
    right, _ = eng.render(z[:, 160 - m:], params=params, stats=stats, plane=plane, origin=(160 - m, 0))
    assert np.max(np.abs(np.hstack([left[:, :160], right[:, m:]]) - whole)[4:-4]) < 0.02


def test_detail_boost_amplifies_fine_relief_only():
    rng = np.random.default_rng(4)
    y, x = np.mgrid[0:192, 0:192].astype(np.float32)
    z = 5.0 * np.exp(-((x - 96) ** 2 + (y - 96) ** 2) / (2 * 40.0 ** 2)) + rng.normal(0, 0.05, (192, 192))
    p0 = eng.merged_params({"mode": "hillshade", "detrend": False})
    s0 = eng.prepare_surface(z.astype(np.float32), 1, 1, p0)[0]
    s1 = eng.prepare_surface(z.astype(np.float32), 1, 1, dict(p0, detail_boost=2.0, detail_radius=2.0))[0]
    fine = lambda a: a - eng.gaussian_blur(a, 1.0)            # noqa: E731
    assert fine(s1).std() > 1.8 * fine(s0).std()             # fine relief lifted...
    np.testing.assert_allclose(eng.gaussian_blur(s1, 12), eng.gaussian_blur(s0, 12), atol=0.05)  # form kept (1 %)


def test_curvature_marks_ridges_bright_and_grooves_dark():
    z = np.zeros((64, 64), np.float32)
    z[:, 20:23] = 1.0                      # ridge
    z[:, 40:43] = -1.0                     # groove
    prep = eng.prepare(z, params={"mode": "studio", "curvature": 1.0, "detrend": False})
    k = eng.shade(prep, {"mode": "studio", "curvature": 1.0}, component="curvature")
    assert k[32, 21] > 0.6 and k[32, 41] < 0.4


@pytest.mark.parametrize("component", list(eng.COMPONENTS))
def test_components_are_valid_grey_images(component):
    params = {"mode": "studio", "texture_mix": 0.3, "occlusion": 0.3, "curvature": 0.5, "slope_shade": 0.3}
    prep = eng.prepare(synthetic_surface(), params=params)
    img = eng.shade(prep, params, component=component)
    assert img.shape == (192, 256) and np.isfinite(img).all() and 0 <= img.min() and img.max() <= 1


def test_openness_is_90_on_flat_and_higher_on_ridges():
    flat = np.zeros((48, 48), np.float32)
    svf, op = eng.horizon_metrics(flat, radius_px=8, openness=True)
    np.testing.assert_allclose(op, 90.0, atol=1e-3)
    np.testing.assert_allclose(svf, 1.0)
    ridge = flat.copy()
    ridge[:, 24] = 3.0                                    # sharp one-pixel ridge
    _, op = eng.horizon_metrics(ridge, radius_px=8, openness=True)
    assert op[24, 24] > 95.0 and op[24, 30] < 90.0


def test_vat_blend_renders_and_marks_incisions_dark():
    z = np.zeros((64, 64), np.float32)
    z[:, 30:33] = -2.0                                    # incision
    img, _ = eng.render(z, params={"mode": "vat", "detrend": False, "svf_radius": 8})
    assert img[32, 31] < img[32, 10] - 0.1
    for variant in eng.VAT_VARIANTS:
        out, _ = eng.render(synthetic_surface(), params={"mode": "vat", "vat_variant": variant})
        assert np.isfinite(out).all() and out.std() > 0.02


@pytest.mark.parametrize("preset", list(eng.BAND_PRESETS))
def test_band_shading_presets_are_neutral_flat_and_shade_relief(preset):
    rows = eng.BAND_PRESETS[preset]
    params = {"mode": "band" + "s", "band_types": [r[0] for r in rows], "band_weights": [r[1] for r in rows],
              "band_altitudes": [r[2] for r in rows], "base_tone": 0.6, "texture_mix": 0.0}
    img, _ = eng.render(np.full((64, 64), 1.5, np.float32), params=params)
    np.testing.assert_allclose(img, 0.6, atol=1e-4)
    shaded, _ = eng.render(synthetic_surface(), params=params)
    assert shaded.std() > 0.01


def test_band_components_and_tiles():
    z = synthetic_surface(160, 320)
    params = {"mode": "bands", "eq_sigmas": [24, 12, 6, 3], "texture_mix": 0.0}
    prep = eng.prepare(z, params=params)
    for i, t in enumerate(eng.DEFAULT_PARAMS["band_types"]):
        img = eng.shade(prep, params, component=f"band{i}")
        assert img.shape == z.shape and 0 <= img.min() and img.max() <= 1
    plane = eng.fit_plane(z)
    whole, stats = eng.render(z, params=params, plane=plane)
    m = eng.spectral_margin_px(params)
    left, _ = eng.render(z[:, :160 + m], params=params, stats=stats, plane=plane)
    right, _ = eng.render(z[:, 160 - m:], params=params, stats=stats, plane=plane, origin=(160 - m, 0))
    assert np.max(np.abs(np.hstack([left[:, :160], right[:, m:]]) - whole)[4:-4]) < 0.02


def test_band_autolevel_evens_out_band_contrast():
    z = synthetic_surface()
    base = {"mode": "bands", "band_types": ["off", "off", "hillshade", "off", "off"], "band_weights": [0, 0, 1, 0, 0],
            "texture_mix": 0.0}
    on, _ = eng.render(z * 0.01, params=dict(base, band_autolevel=True))
    off, _ = eng.render(z * 0.01, params=dict(base, band_autolevel=False))
    assert on.std() > 3 * off.std()                       # a faint band is lifted to readable contrast


def test_band_constants_are_reused_across_shade_types():
    z = synthetic_surface()
    a = {"mode": "bands", "band_types": ["hillshade", "off", "multi", "curvature", "texture"], "texture_mix": 0.0}
    b = dict(a, band_types=["texture", "curvature", "hillshade", "slope", "multi"])
    _, stats_a = eng.render(z, params=a)
    fresh_b, _ = eng.render(z, params=b)
    reused_b, _ = eng.render(z, params=b, stats=stats_a)
    np.testing.assert_allclose(reused_b, fresh_b, atol=1e-5)


GIS_PLAIN = {"mode": "gis", "gis_ambient": 0.0, "gis_shadow": 0.0, "gis_highlight": 0.0}


def test_gis_ring_spaces_lights_evenly_from_the_azimuth():
    assert [az for az, _ in eng.gis_ring(4, 45, 30)] == [45.0, 135.0, 225.0, 315.0]
    assert eng.gis_ring(1, 315, 30) == [(315.0, 30.0)]


def test_gis_single_plain_light_is_the_gdal_hillshade():
    z = synthetic_surface()
    gis, _ = eng.render(z, params=GIS_PLAIN)
    hs, _ = eng.render(z, params={"mode": "hillshade"})
    np.testing.assert_allclose(gis, hs, atol=1e-6)


def test_gis_flat_surface_sits_at_the_ambient_lifted_level():
    flat = np.full((32, 32), 1.0, np.float32)
    img, _ = eng.render(flat, params={"mode": "gis", "altitude": 40, "gis_ambient": 0.2, "gis_shadow": 0.1,
                                      "gis_highlight": 0.3})
    np.testing.assert_allclose(img, 0.2 + 0.8 * math.sin(math.radians(40)), atol=2e-3)
    white, _ = eng.render(flat, params=dict(GIS_PLAIN, gis_slope=1.0))
    np.testing.assert_allclose(white, 1.0, atol=1e-6)        # slope-enhanced: level ground washes out


def test_gis_blends_are_ordered_and_shadow_cap_and_highlight_work():
    z = synthetic_surface()
    base = dict(GIS_PLAIN, gis_lights=4)
    mx, _ = eng.render(z, params=dict(base, gis_blend="max"))
    av, _ = eng.render(z, params=dict(base, gis_blend="average"))
    mn, _ = eng.render(z, params=dict(base, gis_blend="min"))
    assert (mx >= av - 1e-6).all() and (av >= mn - 1e-6).all() and mx.mean() > mn.mean() + 0.01
    capped, _ = eng.render(z, params=dict(GIS_PLAIN, altitude=15, gis_shadow=0.4))
    assert capped.min() >= 0.4 - 1e-4
    plain, _ = eng.render(z, params=GIS_PLAIN)
    lit, _ = eng.render(z, params=dict(GIS_PLAIN, gis_highlight=0.8))
    assert (lit >= plain - 1e-6).all() and lit.max() > plain.max() + 0.01


def test_specular_enhancement_overhead_whitens_level_and_darkens_slopes():
    z = synthetic_surface()
    flat_img, _ = eng.render(np.zeros((32, 32), np.float32), params={"mode": "specular", "altitude": 90})
    np.testing.assert_allclose(flat_img, 1.0, atol=1e-5)
    img, _ = eng.render(z, params={"mode": "specular", "altitude": 90, "spec_depth": 3.0})
    deeper, _ = eng.render(z, params={"mode": "specular", "altitude": 90, "spec_depth": 6.0})
    assert img.min() < 0.2 and deeper.mean() < img.mean()


def test_gis_brightness_and_exaggeration():
    z = synthetic_surface()
    plain, _ = eng.render(z, params=GIS_PLAIN)
    dim, _ = eng.render(z, params=dict(GIS_PLAIN, gis_brightness=0.75))
    np.testing.assert_allclose(dim, 0.75 * plain, atol=1e-5)
    steep, _ = eng.render(z, params=dict(GIS_PLAIN, gis_ve=4.0))
    assert steep.std() > 1.5 * plain.std()


def test_smoothing_removes_terracing_of_an_8_bit_height_map():
    z = synthetic_surface()
    terraced = np.round(z * 4.0) / 4.0                   # coarse quantisation, like an 8-bit export
    raw, _ = eng.render(terraced, params=GIS_PLAIN)
    smooth, _ = eng.render(terraced, params=dict(GIS_PLAIN, smooth_sigma=2.0))
    truth, _ = eng.render(z, params=GIS_PLAIN)
    assert np.abs(smooth - truth).mean() < 0.6 * np.abs(raw - truth).mean()


def test_fit_recovers_the_gis_settings_of_a_reference_render():
    z = synthetic_surface()
    prep = eng.prepare(z, 1.0, 1.0, {"mode": "gis"})
    truth = dict(GIS_PLAIN, azimuth=45.0, altitude=52.0, gis_ve=2.5, gis_ambient=0.25, gis_brightness=0.78)
    ref = eng.shade(prep, truth)
    fit = eng.fit_gis_to_reference(prep.normals, ref)
    assert fit["r2"] > 0.999
    assert abs((fit["azimuth"] - 45.0 + 180) % 360 - 180) < 2.0 and abs(fit["altitude"] - 52.0) < 3.0
    assert abs(fit["gis_ambient"] - 0.25) < 0.05 and abs(fit["gis_brightness"] - 0.78) < 0.03
    again = eng.shade(prep, dict(GIS_PLAIN, **{k: v for k, v in fit.items() if k != "r2"}))
    assert np.abs(again - ref).max() < 0.01


def _gdaldem(z, opts, res=(1.0, 1.0)):
    """gdaldem hillshade of z with a gdaldem option string, via osgeo."""
    gdal = pytest.importorskip("osgeo.gdal")
    gdal.UseExceptions()
    h, w = z.shape
    ds = gdal.GetDriverByName("MEM").Create("", w, h, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((0, res[0], 0, h * res[1], 0, -res[1]))
    ds.GetRasterBand(1).WriteArray(z.astype(np.float32))
    out = gdal.DEMProcessing("/vsimem/test_hillshade.tif", ds, "hillshade", options=opts)
    a = out.GetRasterBand(1).ReadAsArray().astype(np.float32)
    out = None
    gdal.Unlink("/vsimem/test_hillshade.tif")
    return a


GDAL_CASES = [
    ("-az 315 -alt 45", {}),
    ("-az 70 -alt 25", {"azimuth": 70, "altitude": 25}),
    ("-alg ZevenbergenThorne -az 315 -alt 45", {"gdal_zt": True}),
    ("-combined -az 120 -alt 30", {"gdal_variant": "combined", "azimuth": 120, "altitude": 30}),
    ("-multidirectional -alt 45", {"gdal_variant": "multidirectional"}),
    ("-alg ZevenbergenThorne -multidirectional -alt 30", {"gdal_variant": "multidirectional", "gdal_zt": True,
                                                         "altitude": 30}),
    ("-igor -az 200", {"gdal_variant": "igor", "azimuth": 200}),
    ("-z 3 -s 2 -az 315 -alt 45", {"gdal_z": 3.0, "gdal_scale": 2.0}),
    ("-compute_edges -az 315 -alt 45", {"gdal_edges": True}),
    ("-compute_edges -combined -az 315 -alt 45", {"gdal_edges": True, "gdal_variant": "combined"}),
    ("-compute_edges -multidirectional -alt 45", {"gdal_edges": True, "gdal_variant": "multidirectional"}),
    ("-compute_edges -igor -az 315", {"gdal_edges": True, "gdal_variant": "igor"}),
]


@pytest.mark.parametrize("opts,params", GDAL_CASES)
def test_gdal_mode_matches_gdaldem_exactly(opts, params):
    z = synthetic_surface() * 4.0
    ref = _gdaldem(z, opts)
    p = dict({"mode": "gdal", "gdal_display": "raw", "azimuth": 315, "altitude": 45}, **params)
    img, _ = eng.render(z, params=p)
    assert np.abs(np.round(img * 255.0) - ref).max() <= 1.0


def test_gdal_mode_handles_non_square_pixels_and_ignores_surface_preparation():
    z = synthetic_surface() * 4.0
    ref = _gdaldem(z, "-az 315 -alt 45", res=(0.5, 2.0))
    p = {"mode": "gdal", "gdal_display": "raw", "azimuth": 315, "altitude": 45, "detrend": True,
         "flatten_sigma": 20, "z_factor": 3.0, "smooth_sigma": 2.0}
    img, _ = eng.render(z, 0.5, 2.0, p)
    assert np.abs(np.round(img * 255.0) - ref).max() <= 1.0


def test_gdal_display_limits_are_qgis_min_max_or_cumulative_cut():
    z = synthetic_surface() * 4.0
    raw, _ = eng.render(z, params={"mode": "gdal", "gdal_display": "raw"})
    dn = np.round(raw * 255.0)
    v = dn[dn > 0]
    assert eng.gdal_stretch_limits(dn) == (v.min(), v.max())            # QGIS's default: stretch to min/max
    lo, hi = eng.gdal_stretch_limits(dn, "cut")
    np.testing.assert_allclose((lo, hi), np.percentile(v, [2, 98]))
    assert len(eng.qgis_sample_rows(500, 400)) == 400                   # small: QGIS asks for exact statistics
    rows = eng.qgis_sample_rows(2000, 1500)                             # gdaldem strips of 4 rows, every 19th
    assert rows[:5].tolist() == [0, 1, 2, 3, 76] and len(rows) == 80
    shown, _ = eng.render(z, params={"mode": "gdal", "gdal_display": "cut", "gdal_limits": (lo, hi)})
    inner = (slice(1, -1), slice(1, -1))
    np.testing.assert_allclose(shown[inner], np.clip((dn[inner] - lo) / (hi - lo), 0, 1), atol=1e-6)


LAB = {"mode": "lab", "lab_dirs": 8}


@pytest.mark.parametrize("output", list(eng.LAB_OUTPUTS))
def test_lab_outputs_are_valid_images(output):
    z = synthetic_surface() * 4.0
    img, _ = eng.render(z, params=dict(LAB, lab_output=output, lab_weights=[0.2, 1, 1, 1, 1]))
    assert img.shape[:2] == z.shape and np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
    assert img.std() > 0.01 and (img.ndim == 3) == (output == "pcrgb")


def test_lab_hillshade_of_every_band_is_the_qgis_hillshade():
    z = synthetic_surface() * 4.0
    lab, _ = eng.render(z, params=dict(LAB, lab_display="raw", azimuth=300, lab_alt=40))
    qgis, _ = eng.render(z, params={"mode": "gdal", "gdal_display": "raw", "gdal_edges": True, "azimuth": 300,
                                    "altitude": 40})
    assert np.abs(lab - qgis).max() * 255.0 <= 1.0


def test_lab_pca_of_hillshades():
    z = synthetic_surface() * 4.0
    prep = eng.prepare(z, 1.0, 1.0, dict(LAB, lab_output="pcmag", lab_dirs=16))
    evr = prep.lab["pca"]["evr"]
    assert evr[0] >= evr[1] >= evr[2] > 0 and evr[0] + evr[1] > 0.8        # hillshades of a ring: two main axes
    slope = np.hypot(*eng.horn_gradient(z))
    mag = np.hypot(prep.lab["pcs"][0], prep.lab["pcs"][1])
    assert np.corrcoef(mag[4:-4, 4:-4].ravel(), slope[4:-4, 4:-4].ravel())[0, 1] > 0.9   # direction-free strength
    turned = eng.prepare(np.rot90(z).copy(), 1.0, 1.0, dict(LAB, lab_output="mean", lab_dirs=16))
    a, b = eng.prepare(z, 1.0, 1.0, dict(LAB, lab_output="mean", lab_dirs=16)).lab["mean"], np.rot90(turned.lab["mean"], -1)
    assert np.corrcoef(a[4:-4, 4:-4].ravel(), b[4:-4, 4:-4].ravel())[0, 1] > 0.99      # the mean ignores direction


def test_lab_tiles_match_whole_render():
    z = synthetic_surface(160, 320) * 4.0
    params = dict(LAB, lab_output="pc1", lab_weights=[0, 1, 1, 1, 1], eq_sigmas=[16.0, 8.0, 4.0, 2.0],
                  lab_smoothing="edge", lab_smooth_radius=2.0)
    whole, stats = eng.render(z, params=params)
    margin = eng.spectral_margin_px(params)
    mid = 160
    left, _ = eng.render(z[:, :mid + margin], params=params, stats=stats)
    right, _ = eng.render(z[:, mid - margin:], params=params, stats=stats, origin=(mid - margin, 0))
    stitched = np.hstack([left[:, :mid], right[:, margin:]])
    assert margin < mid and np.max(np.abs(stitched - whole)[4:-4]) < 0.02


def test_edge_preserving_smoothing_keeps_a_step_and_removes_noise():
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(1)
    step = np.zeros((64, 64), np.float32)
    step[:, 32:] = 1.0
    noisy = step + rng.normal(0, 0.05, step.shape).astype(np.float32)
    kept = eng.guided_smooth(noisy, 3, 0.1 ** 2)
    blurred = gaussian_filter(noisy, 1.5)
    assert (kept - step)[:, 4:26].std() < 0.5 * (noisy - step)[:, 4:26].std()
    assert (kept[:, 33] - kept[:, 30]).mean() > (blurred[:, 33] - blurred[:, 30]).mean() + 0.1


def test_lab_multiscale_ring_keeps_fine_relief():
    from scipy.ndimage import gaussian_filter
    z = synthetic_surface() * 4.0
    fine = lambda a: (a - gaussian_filter(a, 3))[8:-8, 8:-8].std() / a[8:-8, 8:-8].std()
    p = dict(LAB, lab_output="pc1", lab_pca_bands=True)
    mixed = eng.prepare(z, 1.0, 1.0, dict(p, lab_weights=[0.3, 0.3, 1, 1, 1])).lab["pcs"][0]
    broad = eng.prepare(z, 1.0, 1.0, dict(p, lab_weights=[0, 1, 0, 0, 0])).lab["pcs"][0]
    assert fine(mixed) > 1.5 * fine(broad)
