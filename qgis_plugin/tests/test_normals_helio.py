"""Tests for shading from a normal map and for recognising HELIO export folders."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "helio_relief"))
import helio  # noqa: E402
import relief_engine as eng  # noqa: E402
from test_relief_engine import synthetic_surface  # noqa: E402


def normal_map(z, green="up"):
    """The RGB normal map (0..1) of a height map, R = x, G = y (image up for 'up'), B = z."""
    gr, gc = np.gradient(z.astype(np.float64))
    n = np.dstack([-gc, gr if green == "up" else -gr, np.ones_like(gc)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return ((n + 1.0) / 2.0).astype(np.float32)


def test_integrated_slopes_give_back_the_surface():
    z = synthetic_surface(160, 200).astype(np.float64)
    gr, gc = np.gradient(z)
    back = eng.integrate_slopes(gc, gr, flatten=12)
    ref = z - eng.gaussian_blur(z.astype(np.float32), 12)
    inner = (slice(20, -20), slice(20, -20))
    assert np.corrcoef(back[inner].ravel(), ref[inner].ravel())[0, 1] > 0.97


def test_clean_relief_from_normals_matches_the_height_map_render():
    z = synthetic_surface(192, 256)
    p = {"mode": "clean", "clean_flatten": 30, "clean_smooth": 0.0, "clean_fine": 0.0}
    from_z, _ = eng.render(z, params=p)
    from_n, _ = eng.render(z, params=dict(p, clean_source="normals"), normals=normal_map(z))
    inner = (slice(16, -16), slice(16, -16))
    assert np.corrcoef(from_z[inner].ravel(), from_n[inner].ravel())[0, 1] > 0.95


def test_directx_normal_maps_and_their_detection():
    z = synthetic_surface(192, 256)
    up, down = normal_map(z, "up"), normal_map(z, "down")
    assert eng.detect_normals_convention(up, z)[0] == "up"
    got, agreement = eng.detect_normals_convention(down, z)
    assert got == "down" and agreement > 0.9
    p = {"mode": "clean", "clean_source": "normals", "clean_flatten": 30}
    a, st = eng.render(z, params=dict(p, normals_green="up"), normals=up)
    b, _ = eng.render(z, params=dict(p, normals_green="down"), normals=down, stats=st)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_normals_source_tiles_match_whole_render():
    z = synthetic_surface(160, 320)
    nm = normal_map(z)
    params = {"mode": "clean", "clean_source": "normals", "clean_flatten": 12, "clean_smooth": 2.0, "clean_fine": 0.5}
    whole, stats = eng.render(z, params=params, normals=nm)
    margin = eng.spectral_margin_px(params)
    mid = 160
    left, _ = eng.render(z[:, :mid + margin], params=params, stats=stats, normals=nm[:, :mid + margin])
    right, _ = eng.render(z[:, mid - margin:], params=params, stats=stats, origin=(mid - margin, 0),
                          normals=nm[:, mid - margin:])
    stitched = np.hstack([left[:, :mid], right[:, margin:]])
    assert np.max(np.abs(stitched - whole)[4:-4]) < 0.02


def test_empty_normal_pixels_face_up():
    rgb = np.zeros((4, 4, 3), np.float32)
    nx, ny, nz = eng.decode_normals(rgb)
    assert (nx == 0).all() and (ny == 0).all() and (nz == 1).all()


GIS_README = """HELIO Pro — GIS DEM Export Package
Object ID : helio_dem
Images captured : 4
CALIBRATION
  XY pixel size : 1.000000 mm/pixel (estimated — calibrate scale bar for accuracy)
  Z values      : mm above local datum (calibrated)
"""

MS_README = """HELIO Pro V106 — MorphoSource Deposit Package
Object ID  : HELIO_OBJECT
Physical   : uncalibrated
Pixels     : 9479×6687
Images     : 8 PS capture frames
    R=X, G=Y, B=Z in OpenGL convention.
    32-bit float depth map. Relative units — calibrate Z in HELIO before depositing for physical values.
"""


def _touch(folder, *names):
    for n in names:
        with open(os.path.join(folder, n), "w") as f:
            f.write("x")


def test_gis_export_is_recognised(tmp_path):
    d = str(tmp_path)
    _touch(d, "helio_dem_DEM_Poisson.tif", "helio_dem_DEM_FrankotChellappa.tif",
           "helio_dem_DEM_FrankotChellappa_MSRM_F_M11.0-11.0_S2.tif", "helio_dem_DEM_Poisson.tif.aux.xml",
           "albedo_16bit.tif", "helio_dem_Albedo_RGB.tif", "helio_dem_DEM_Poisson.tfw")
    with open(os.path.join(d, "README_GIS.txt"), "w", encoding="utf-8") as f:
        f.write(GIS_README)
    s = helio.scan(d)
    assert s["kind"] == "gis" and s["object_id"] == "helio_dem"
    assert [k for k, _ in s["dems"]] == ["Poisson", "Frankot-Chellappa"]
    assert os.path.basename(s["dems"][1][1]) == "helio_dem_DEM_FrankotChellappa.tif"   # not the visualisation
    assert os.path.basename(s["albedo"]) == "albedo_16bit.tif" and s["normals"] is None
    r = s["readme"]
    assert r["xy_size"] == 1.0 and r["xy_unit"] == "mm" and r["xy_estimated"] and r["z_unit"] == "mm"
    assert r["z_calibrated"] and r["images"] == 4
    assert s["uncertainty"] is None and s["metal"] is None


def test_gis_export_uncertainty_and_metal_layers_are_found(tmp_path):
    d = str(tmp_path)
    _touch(d, "obj_DEM_Poisson.tif", "obj_DEM_FrankotChellappa.tif", "obj_NormalMap.tif", "obj_uncertainty.tif",
           "obj_metal_mask.tif", "obj_metal_mask.tfw")
    s = helio.scan(d)
    assert s["normals"].endswith("obj_NormalMap.tif")
    assert s["uncertainty"].endswith("obj_uncertainty.tif") and s["metal"].endswith("obj_metal_mask.tif")


def test_morphosource_package_is_recognised(tmp_path):
    d = str(tmp_path)
    _touch(d, "HELIO_OBJECT_depth.tif", "HELIO_OBJECT_albedo.jpg", "HELIO_OBJECT_normals.png",
           "HELIO_OBJECT_roughness.png", "HELIO_OBJECT.glb")
    with open(os.path.join(d, "README.txt"), "w", encoding="utf-8") as f:
        f.write(MS_README)
    s = helio.scan(d)
    assert s["kind"] == "morphosource" and s["object_id"] == "HELIO_OBJECT"
    assert [k for k, _ in s["dems"]] == ["Depth"]
    assert s["normals"].endswith("HELIO_OBJECT_normals.png") and s["albedo"].endswith("HELIO_OBJECT_albedo.jpg")
    r = s["readme"]
    assert r["normals_convention"] == "up" and r["z_calibrated"] is False and r["pixels"] == (9479, 6687)
