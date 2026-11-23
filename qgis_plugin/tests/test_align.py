"""Tests for align - registering a colour / albedo image onto a height map (needs GDAL)."""

import math
import os
import sys

import numpy as np
import pytest
from scipy import ndimage as ndi

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "helio_relief"))
gdal = pytest.importorskip("osgeo.gdal")
import align  # noqa: E402


def _scene(h=900, w=1200, seed=3):
    """A height map with grain, scattered tool stamps and the object's outline against its background."""
    rng = np.random.default_rng(seed)
    z = ndi.gaussian_filter(rng.normal(size=(h, w)), 3) * 20
    stamps = np.zeros((h, w))
    stamps[rng.integers(0, h, 900), rng.integers(0, w, 900)] = 1.0
    z -= 40.0 * ndi.gaussian_filter(stamps, 4)                     # round punched hollows, everywhere
    z[:60], z[-60:], z[:, :60], z[:, -60:] = -5.0, -5.0, -5.0, -5.0
    return z.astype(np.float32)


def _colour_of(z):
    """An albedo sharing the height map's structure: darker in the hollows, paler on the raised areas."""
    g = z - ndi.gaussian_filter(z, 8)
    return 0.55 + 0.35 * np.tanh(g / (3 * g.std()))


def _write(path, arr, gt=None):
    bands = arr.shape[2] if arr.ndim == 3 else 1
    dtype = gdal.GDT_UInt16 if arr.dtype == np.uint16 else gdal.GDT_Float32
    ds = gdal.GetDriverByName("GTiff").Create(path, arr.shape[1], arr.shape[0], bands, dtype)
    if gt:
        ds.SetGeoTransform(gt)
    for i in range(bands):
        ds.GetRasterBand(i + 1).WriteArray(arr[..., i] if arr.ndim == 3 else arr)
    ds = None


def _affine(sx, sy, rot_deg, tx, ty):
    t = math.radians(rot_deg)
    return np.array([[sx * math.cos(t), -sy * math.sin(t), tx], [sx * math.sin(t), sy * math.cos(t), ty]])


# (scale x, scale y, rotation deg, offset x, offset y): the same pixels; a shift; padding with a resample; and
# the user's HELIO export (albedo inside the depth map, turned by ~0.3 deg)
CASES = [(1.0, 1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 7.3, -4.6), (1.0301, 1.0321, 0.0, 37.4, 23.2),
         (1.0, 1.0, 0.3, 40.2, 35.7)]


@pytest.mark.parametrize("case", CASES)
def test_colour_image_is_found_on_the_height_map(tmp_path, case):
    z = _scene()
    H, W = z.shape
    A = _affine(*case)
    identity = case == (1.0, 1.0, 0.0, 0.0, 0.0)
    IW, IH = (W, H) if identity else (int((W - 90) / case[0]), int((H - 90) / case[1]))
    yy, xx = np.mgrid[0:IH, 0:IW].astype(np.float64)
    lum_ref = _colour_of(z)
    lum = ndi.map_coordinates(lum_ref, [A[1, 0] * xx + A[1, 1] * yy + A[1, 2], A[0, 0] * xx + A[0, 1] * yy + A[0, 2]],
                              order=3, mode="nearest")
    rgb = (np.clip(np.dstack([lum, lum * 0.9, lum * 0.7]), 0, 1) * 60000).astype(np.uint16)
    ref, img = str(tmp_path / "dem.tif"), str(tmp_path / "albedo.tif")
    _write(ref, z, (0.0, 1.0, 0.0, float(H), 0.0, -1.0))
    _write(img, rgb)                          # no georeferencing, as photometric-stereo albedos come
    got = align.estimate(ref, img)
    B = np.array([[got["a"], got["b"], got["c"]], [got["d"], got["e"], got["f"]]])
    corners = np.array([[0, IW - 1, 0, IW - 1], [0, 0, IH - 1, IH - 1], [1, 1, 1, 1]], dtype=np.float64)
    assert np.abs((B - A) @ corners).max() < 0.75         # every corner placed to within a pixel
    assert got["identity"] == identity and got["rotation"] == pytest.approx(case[2], abs=0.02)

    out = str(tmp_path / "aligned.vrt")
    align.write_vrt(ref, img, got, out)
    ds = gdal.Open(out)
    assert (ds.RasterXSize, ds.RasterYSize, ds.RasterCount) == (W, H, 3)
    np.testing.assert_allclose(ds.GetGeoTransform(), (0.0, 1.0, 0.0, float(H), 0.0, -1.0), atol=1e-9)
    assert float(ds.GetMetadataItem("HPSS_WHITE")) == pytest.approx(np.percentile(rgb[rgb > 0], 99.8), rel=0.05)
    back = ds.GetRasterBand(1).ReadAsArray().astype(np.float64) / 60000
    inner = (slice(80, H - 80), slice(80, W - 80))
    inside = ndi.binary_erosion(back > 0, iterations=4)[inner]      # the colour image's footprint (nodata 0 outside)
    assert inside.mean() > 0.8
    assert np.corrcoef(back[inner][inside], lum_ref[inner][inside])[0, 1] > 0.98


def test_unrelated_image_is_refused(tmp_path):
    z = _scene()
    rng = np.random.default_rng(9)
    noise = (rng.random((700, 1000, 3)) * 60000).astype(np.uint16)
    ref, img = str(tmp_path / "dem.tif"), str(tmp_path / "noise.tif")
    _write(ref, z, (0.0, 1.0, 0.0, float(z.shape[0]), 0.0, -1.0))
    _write(img, noise)
    with pytest.raises(RuntimeError):
        align.estimate(ref, img)
