"""
relief_engine - numerical core of Helio Relief.

Pure NumPy/SciPy: no QGIS or Qt imports, so everything here can be unit tested
outside QGIS and called safely from a QgsTask worker thread.

Conventions
    * Heights are 2-D float32 arrays, row 0 at the top (north) of the image.
    * dx, dy are pixel sizes in the same units as the (z-scaled) heights.
    * Azimuth is degrees clockwise from image-up (north); altitude is degrees
      above the horizon - the same convention as ``gdaldem hillshade``.
    * Shaded images are float32 in [0, 1].

Texture shading follows Leland Brown, "Texture Shading: A New Technique for
Depicting Terrain Relief" (2014): a discretised fractional Laplacian computed
in the DCT domain (mirror boundaries), using his band-limited interpolator and
a smooth S-curve contrast stretch instead of hard clipping.

Rendering modes include a studio relief model (key / fill / ambient lighting
with soft wrap, soft cast shadows, Brown texture shading, sky-view occlusion
and clarity) and multi-shade hillshading (several weighted, optionally coloured
lights; the aspect-adaptive method reproduces GDAL's multidirectional
hillshade).

Rendering is split in two: ``prepare`` does everything that does not depend
on the light (surface filtering, normals, texture shading, sky-view factor)
and ``shade`` applies lights and tone. Moving the light only re-runs
``shade``, which keeps the live preview interactive.
"""

import math

import numpy as np

try:
    from scipy import fft as _sfft
    from scipy import ndimage as _ndi
    HAS_SCIPY = True
except ImportError:  # pragma: no cover - QGIS bundles scipy on the platforms we target
    _sfft = _ndi = None
    HAS_SCIPY = False


class Cancelled(Exception):
    """Raised (usually by a progress callback) to abort a long computation."""


def _noop(*_args, **_kwargs):
    pass


# ----------------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------------

def fill_nodata(z):
    """Return (filled, valid_mask). NaNs are replaced with the nearest valid
    height so gradients and spectra stay well defined; valid_mask is None when
    the array had no gaps."""
    z = np.asarray(z, dtype=np.float32)
    valid = np.isfinite(z)
    if valid.all():
        return z, None
    if not valid.any():
        return np.zeros_like(z), valid
    if HAS_SCIPY:
        idx = _ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = z[tuple(idx)]
    else:
        filled = np.where(valid, z, np.float32(np.nanmean(z)))
    return filled.astype(np.float32, copy=False), valid


def robust_scale(x, max_samples=400_000):
    """Robust standard deviation (1.4826 x MAD) of a subsample of x."""
    step = max(1, int(math.sqrt(x.size / max_samples)))
    s = x[::step, ::step] if x.ndim == 2 else x[::step]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 1.0
    med = np.median(s)
    sd = float(np.median(np.abs(s - med)) * 1.4826)
    if sd <= 0:
        sd = float(s.std())
    return sd if sd > 0 else 1.0


def texture_scale(x, max_samples=400_000):
    """Normalising scale for texture-shading values: the 95th percentile of
    |x - median| / 1.96 (= 1 sigma for Gaussian data). Unlike the MAD it is not
    collapsed by large flat areas (blank parchment, bare plate), which would
    otherwise drive every feature into saturation."""
    step = max(1, int(math.sqrt(x.size / max_samples)))
    s = x[::step, ::step]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(s - np.median(s)), 95.0)) / 1.96
    return v if v > 0 else robust_scale(x)


def fit_plane(z, max_samples=250_000):
    """Least-squares plane z ~ a*col + b*row + c, fitted on a subsample.
    Coefficients are in the array's own pixel coordinates."""
    h, w = z.shape
    step = max(1, int(math.sqrt(z.size / max_samples)))
    rows, cols = np.mgrid[0:h:step, 0:w:step]
    zs = z[::step, ::step]
    ok = np.isfinite(zs)
    if ok.sum() < 3:
        return (0.0, 0.0, 0.0)
    a = np.c_[cols[ok], rows[ok], np.ones(ok.sum())]
    coef = np.linalg.lstsq(a, zs[ok].astype(np.float64), rcond=None)[0]
    return tuple(float(c) for c in coef)


def plane_surface(plane, shape, origin=(0.0, 0.0), step=1.0):
    """Evaluate a fitted plane on a grid whose pixel (0, 0) sits at full-resolution
    pixel `origin` (col, row), with `step` full-resolution pixels per grid pixel."""
    a, b, c = plane
    h, w = shape
    cols = (origin[0] + np.arange(w, dtype=np.float64) * step)[None, :]
    rows = (origin[1] + np.arange(h, dtype=np.float64) * step)[:, None]
    return (a * cols + b * rows + c).astype(np.float32)


def to_uint8(img):
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_uint16(img):
    return (np.clip(img, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


# ----------------------------------------------------------------------------
# Frequency domain (DCT = mirror boundaries, as Brown recommends)
# ----------------------------------------------------------------------------

class Spectrum:
    """Mirror-boundary spectrum of a 2-D array.

    Reflecting the surface across its edges avoids wrap-around seams; a
    DCT-II gives exactly that boundary condition with no extra work. Filters
    are arrays built from ``fy`` / ``fx``: the non-negative spatial frequency
    in cycles per pixel (0 .. 0.5), shaped to broadcast over the image.
    """

    def __init__(self, arr, reach=0):
        arr = np.asarray(arr, dtype=np.float32)
        self.shape = arr.shape
        ny, nx = arr.shape
        if HAS_SCIPY:
            # Mirror-pad to FFT-friendly lengths: a transform whose length has a large prime
            # factor (e.g. 922 = 2 x 461) is many times slower. The pad is at least `reach`
            # pixels (the widest filter's extent), so everything a filter can see past the edge
            # is an exact mirror; results are cropped back to the original size.
            r = int(max(0, reach))
            py = _sfft.next_fast_len(ny + min(r, ny), real=True)
            px = _sfft.next_fast_len(nx + min(r, nx), real=True)
            if (py, px) != (ny, nx):
                arr = np.pad(arr, ((0, py - ny), (0, px - nx)), mode="symmetric")
            self._coef = _sfft.dctn(arr, type=2, norm="ortho", workers=-1)
            self.fy = (np.arange(py, dtype=np.float32) / np.float32(2 * py))[:, None]
            self.fx = (np.arange(px, dtype=np.float32) / np.float32(2 * px))[None, :]
        else:
            ext = np.pad(arr, ((0, ny), (0, nx)), mode="symmetric")
            self._coef = np.fft.rfft2(ext)
            self.fy = np.abs(np.fft.fftfreq(2 * ny)).astype(np.float32)[:, None]
            self.fx = np.fft.rfftfreq(2 * nx).astype(np.float32)[None, :]

    def filtered(self, multiplier):
        if HAS_SCIPY:
            out = _sfft.idctn(self._coef * multiplier, type=2, norm="ortho", workers=-1)
            return np.ascontiguousarray(out[:self.shape[0], :self.shape[1]], dtype=np.float32)
        ny, nx = self.shape
        out = np.fft.irfft2(self._coef * multiplier, s=(2 * ny, 2 * nx))
        return out[:ny, :nx].astype(np.float32)


def gaussian_multiplier(fy, fx, sigma):
    """Spectrum of a Gaussian blur with standard deviation `sigma` pixels."""
    if sigma <= 0:
        return np.float32(1.0)
    k = np.float32(-2.0 * math.pi ** 2 * sigma ** 2)
    return np.exp(k * (fy * fy + fx * fx)).astype(np.float32)


def eq_multiplier(fy, fx, sigmas, gains):
    """Five-band spatial-frequency equaliser as one spectral multiplier.

    Four Gaussian crossovers (any order) split the spectrum into
    very-low / low / mid / high / ultra-high bands, exactly as the Frequency
    Studio does; with all gains at 1.0 the multiplier is identically 1."""
    s = sorted(sigmas, reverse=True)
    prev = gaussian_multiplier(fy, fx, s[0])
    m = np.float32(gains[0]) * prev
    for i in range(1, 4):
        cur = gaussian_multiplier(fy, fx, s[i])
        m = m + np.float32(gains[i]) * (cur - prev)
        prev = cur
    return m + np.float32(gains[4]) * (1.0 - prev)


def _bandlimited_spline(f):
    """Brown eq. 6: band-limited stand-in for the cubic-spline interpolator.

    Zero for |f| >= 1 and S(f) + S(1 - f) = 1, so it is a true interpolator;
    it tracks the cubic spline's 3 sinc^4(f) / (2 + cos 2 pi f) to ~0.5 %."""
    f = np.minimum(np.abs(f), np.float32(1.0))
    a = (1.0 - f) ** 4
    b = f ** 4
    return a / (a + b)


def fractional_laplacian_multiplier(fy, fx, detail, dx=1.0, dy=1.0):
    """Brown eqs. 3-5: spline-interpolate, apply the fractional Laplacian
    (2 pi |k|)^alpha, then re-sample by summing the four alias terms that
    survive the band limit. `detail` is alpha: 0.5 = Brown's "50 %"."""
    alpha = float(detail)
    total = None
    for ay in (fy, fy - np.float32(1.0)):
        sy = _bandlimited_spline(ay)
        ky = (ay / np.float32(dy)) ** 2
        for ax in (fx, fx - np.float32(1.0)):
            sx = _bandlimited_spline(ax)
            kx = (ax / np.float32(dx)) ** 2
            term = (sy * sx) * np.power(ky + kx, np.float32(alpha / 2.0))
            total = term if total is None else total + term
    return total * np.float32((2.0 * math.pi) ** alpha)


def texture_shade(z, dx=1.0, dy=1.0, detail=0.5, spectrum=None):
    """Raw (signed) texture-shading values: positive = higher than the
    surrounding surface (ridges), negative = lower (valleys, incisions)."""
    spec = spectrum if spectrum is not None else Spectrum(z, 32)
    return spec.filtered(fractional_laplacian_multiplier(spec.fy, spec.fx, detail, dx, dy))


def s_curve(x, contrast=1.0, midtone=0.0, curve="tanh"):
    """Brown eqs. 7 / 8: smooth S-curve mapping signed values to [0, 1] that
    saturates gradually, so outliers (the sharpest ridges and cuts) keep some
    contrast instead of clipping flat."""
    t = np.float32(contrast) * x + np.float32(midtone)
    y = np.tanh(t) if curve == "tanh" else t / np.sqrt(1.0 + t * t)
    return (0.5 * (y + 1.0)).astype(np.float32)


def texture_to_image(tex, contrast=1.0, midtone=0.0, curve="tanh", scale=None):
    scale = scale or texture_scale(tex)
    return s_curve(tex / np.float32(scale), contrast, midtone, curve)


def _reach(*sigmas):
    """Pixels past an edge that a set of Gaussian filters can see (4 sigma)."""
    return int(4.0 * max([0.0] + [float(v) for v in sigmas])) + 2


def gaussian_blur(z, sigma):
    """Mirror-boundary Gaussian blur. Uses the DCT for large sigmas, where it is
    far faster than a spatial kernel (cost does not grow with sigma)."""
    if sigma <= 0:
        return np.asarray(z, dtype=np.float32)
    if HAS_SCIPY and sigma < 6:
        return _ndi.gaussian_filter(np.asarray(z, dtype=np.float32), sigma, mode="reflect")
    spec = Spectrum(z, _reach(sigma))
    return spec.filtered(gaussian_multiplier(spec.fy, spec.fx, sigma))


def frequency_split(z, sigmas, progress=_noop):
    """Split a surface into five band-pass layers that sum back to z exactly.
    One forward DCT plus four inverse transforms, whatever the sigmas."""
    s = sorted(sigmas, reverse=True)
    spec = Spectrum(z, _reach(*s))
    blurs = []
    for i, sig in enumerate(s):
        blurs.append(spec.filtered(gaussian_multiplier(spec.fy, spec.fx, sig)))
        progress(int(100 * (i + 1) / 4), f"Crossover {i + 1}/4 (sigma {sig:g} px)")
    z = np.asarray(z, dtype=np.float32)
    return {
        "Very_Low_Freq": blurs[0],
        "Low_Freq": blurs[1] - blurs[0],
        "Mid_Freq": blurs[2] - blurs[1],
        "High_Freq": blurs[3] - blurs[2],
        "Ultra_High_Freq": z - blurs[3],
    }


def treat_band(arr, mode, radius, threshold=0.0):
    """Per-band treatment used by the Frequency Studio."""
    out = arr
    if radius > 0:
        if mode == "Gaussian smooth":
            out = gaussian_blur(arr, radius)
        elif mode == "Median despeckle" and HAS_SCIPY:
            out = _ndi.median_filter(arr, size=max(3, int(round(radius)) * 2 + 1))
        elif mode == "Local contrast":
            out = arr + (arr - gaussian_blur(arr, radius))
    if threshold > 0:
        out = np.where(np.abs(out) < threshold, np.float32(0.0), out)
    return np.asarray(out, dtype=np.float32)


BAND_KEYS = ("Very_Low_Freq", "Low_Freq", "Mid_Freq", "High_Freq", "Ultra_High_Freq")


def condition_surface(z, s, px_scale=1.0, plane=None, origin=(0.0, 0.0)):
    """Frequency Studio pre-split conditioning (detrend, median despeckle,
    Gaussian pre-blur). Kernel sizes in `s` are full-resolution px."""
    if s["detrend"] and plane is not None:
        z = z - plane_surface(plane, z.shape, origin, px_scale)
    k = int(round(s["median"] / px_scale)) // 2 * 2 + 1
    if s["median"] >= 3 and k >= 3 and HAS_SCIPY:
        z = _ndi.median_filter(z, size=k)
    if s["preblur"] > 0:
        z = gaussian_blur(z, s["preblur"] / px_scale)
    return z


def post_split(bands, s, px_scale=1.0):
    """Ultra-high denoise and the per-band treatments, applied after splitting."""
    if s["denoise"] > 0 and 1.0 / px_scale >= 0.3:
        u, a = bands["Ultra_High_Freq"], np.float32(s["denoise"] / 100.0)
        bands["Ultra_High_Freq"] = u * (1 - a) + gaussian_blur(u, 1.0 / px_scale) * a
    for key, t in s["treat"].items():
        bands[key] = treat_band(bands[key], t["mode"], t["radius"] / px_scale, t["threshold"])
    return bands


def ramp_overlay(band, grey, stops):
    """Diverging colour ramp [(fraction, (r, g, b, a)), ...] composited over a
    grey relief; the ramp's midpoint sits on the band's zero crossing."""
    v = s_curve(band / np.float32(robust_scale(band)), 0.6)
    if not stops:
        return v
    fr = [s[0] for s in stops]
    cols = np.array([s[1] for s in stops], dtype=np.float32) / 255.0
    rgba = np.stack([np.interp(v, fr, cols[:, c]) for c in range(4)], axis=-1).astype(np.float32)
    a = rgba[..., 3:]
    return grey[..., None] * (1.0 - a) + rgba[..., :3] * a


# ----------------------------------------------------------------------------
# Gradients, normals and lighting
# ----------------------------------------------------------------------------

def horn_gradient(z, dx=1.0, dy=1.0):
    """Horn (1981) 3x3 gradient, as used by gdaldem. Returns
    (dz/d_east, dz/d_north) with row 0 at the north edge."""
    zp = np.pad(np.asarray(z, dtype=np.float32), 1, mode="edge")
    top, mid, bot = zp[:-2], zp[1:-1], zp[2:]
    gx = ((top[:, 2:] + 2 * mid[:, 2:] + bot[:, 2:]) -
          (top[:, :-2] + 2 * mid[:, :-2] + bot[:, :-2])) / np.float32(8 * dx)
    gy = ((top[:, :-2] + 2 * top[:, 1:-1] + top[:, 2:]) -
          (bot[:, :-2] + 2 * bot[:, 1:-1] + bot[:, 2:])) / np.float32(8 * dy)
    return gx, gy


def surface_normals(z, dx=1.0, dy=1.0, z_factor=1.0):
    """Unit normals (east, north, up) of a height field."""
    gx, gy = horn_gradient(z, dx, dy)
    if z_factor != 1.0:
        gx *= np.float32(z_factor)
        gy *= np.float32(z_factor)
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    return -gx * inv, -gy * inv, inv


def light_vector(azimuth, altitude):
    az, alt = math.radians(azimuth), math.radians(altitude)
    return (math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt))


def lambert(normals, light):
    nx, ny, nz = normals
    lx, ly, lz = (np.float32(v) for v in light)
    return np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)


def hillshade(z, dx=1.0, dy=1.0, azimuth=315.0, altitude=45.0, z_factor=1.0):
    """Single-light Lambertian hillshade in [0, 1] (matches gdaldem hillshade)."""
    return lambert(surface_normals(z, dx, dy, z_factor), light_vector(azimuth, altitude))


def _gdal_windows(z, edges):
    """The nine 3x3 neighbours of every pixel as gdaldem reads them (a b c /
    d e f / g h i, north row first). Without -compute_edges the outer ring
    stays NaN; with it, missing neighbours are extrapolated the way GDAL does:
    2a - b along each line, and on the first and last lines from the clamped
    column."""
    h, w = z.shape
    zp = np.full((h + 2, w + 2), np.nan)
    zp[1:-1, 1:-1] = z
    win = [zp[r:r + h, c:c + w].copy() for r in range(3) for c in range(3)]
    if not edges or h < 2 or w < 2:
        return win
    for k in (0, 3, 6):
        m = np.isnan(win[k])
        win[k][m] = (2.0 * win[k + 1] - win[k + 2])[m]
    for k in (2, 5, 8):
        m = np.isnan(win[k])
        win[k][m] = (2.0 * win[k - 1] - win[k - 2])[m]
    zc = np.pad(z, ((0, 0), (1, 1)), mode="edge")
    up, dn = 2.0 * zc[0] - zc[1], 2.0 * zc[-1] - zc[-2]
    for k, off in ((0, 0), (1, 1), (2, 2)):
        win[k][0] = up[off:off + w]
    for k, off in ((6, 0), (7, 1), (8, 2)):
        win[k][-1] = dn[off:off + w]
    win[3][0], win[5][0] = zc[0][0:w], zc[0][2:2 + w]
    win[6][0], win[8][0] = zc[1][0:w], zc[1][2:2 + w]
    win[3][-1], win[5][-1] = zc[-1][0:w], zc[-1][2:2 + w]
    win[0][-1], win[2][-1] = zc[-2][0:w], zc[-2][2:2 + w]
    return win


def gdal_slopes(z, dx=1.0, dy=1.0, zt=False, edges=False):
    """gdaldem's slopes: x = west - east, y = north - south (Horn's 3x3 or
    Zevenbergen-Thorne), NaN on the outer ring unless edges are computed."""
    w = _gdal_windows(np.asarray(z, dtype=np.float64), bool(edges))
    if zt:
        return (w[3] - w[5]) / (2.0 * dx), (w[1] - w[7]) / (2.0 * dy)
    return (((w[0] + 2 * w[3] + w[6]) - (w[2] + 2 * w[5] + w[8])) / (8.0 * dx),
            ((w[0] + 2 * w[1] + w[2]) - (w[6] + 2 * w[7] + w[8])) / (8.0 * dy))


def gdal_hillshade(z, dx=1.0, dy=1.0, params=None):
    """gdaldem hillshade, reproduced exactly (verified against GDAL 3.12 to
    within one grey level for every option): standard, combined,
    multidirectional or Igor shading, Horn or Zevenbergen-Thorne gradient,
    -z, -s and -compute_edges. Returns gdaldem's grey levels as float32:
    1-255, and 0 where gdaldem writes nodata (the outer ring without
    -compute_edges)."""
    p = merged_params(params)
    zf = float(p["gdal_z"]) / (float(p["gdal_scale"]) or 1.0)
    x, y = gdal_slopes(z, dx, dy, p["gdal_zt"], p["gdal_edges"])
    variant = p["gdal_variant"]
    az, alt = math.radians(float(p["azimuth"])), math.radians(float(p["altitude"]))
    s2 = x * x + y * y
    with np.errstate(invalid="ignore", divide="ignore"):
        if variant == "igor":
            slope_deg = np.degrees(np.arctan(np.sqrt(s2) * zf))
            aspect = np.arctan2(-y, x)
            diff = np.abs((aspect - (1.5 * math.pi - az) + math.pi) % (2 * math.pi) - math.pi)
            v = 255.0 * (1.0 - (slope_deg / 90.0) * (1.0 - diff / math.pi))
        elif variant == "multidirectional":
            sa, ca, s225 = math.sin(alt), math.cos(alt) * zf, math.sin(math.radians(225.0))
            v225 = np.maximum(0.0, sa + (x - y) * ca * s225)
            v270 = np.maximum(0.0, sa - x * ca)
            v315 = np.maximum(0.0, sa + (x + y) * ca * s225)
            v360 = np.maximum(0.0, sa - y * ca)
            w225 = 0.5 * s2 - x * y
            c = (w225 * v225 + x * x * v270 + (s2 - w225) * v315 + y * y * v360) / s2 / 2.0
            c = np.where(s2 == 0, sa, c)
            v = 1.0 + 254.0 * c / np.sqrt(1.0 + zf * zf * s2)
        else:
            c = (math.sin(alt) - (y * math.cos(az) - x * math.sin(az)) * math.cos(alt) * zf) / np.sqrt(1.0 + zf * zf * s2)
            if variant == "combined":
                c = 1.0 - np.arccos(np.clip(c, -1.0, 1.0)) * np.arctan(np.sqrt(zf * zf * s2)) / (math.pi / 2) ** 2
            v = np.where(c <= 0, 1.0, 1.0 + 254.0 * c)
    v = np.floor(v + 0.5)
    return np.where(np.isfinite(v), np.clip(v, 0.0, 255.0), 0.0).astype(np.float32)


def gdal_stretch_limits(dn, method="qgis"):
    """Display limits for gdaldem grey levels (nodata 0 left out). 'qgis':
    what QGIS does with a new single-band layer - stretch between the minimum
    and maximum of a sample of the pixels (pass a sample of about the 250 000
    pixels QGIS reads); 'cut': the 2 % and 98 % points of the cumulative
    histogram."""
    v = np.asarray(dn)
    v = v[v > 0]
    if v.size == 0:
        return 0.0, 255.0
    lo, hi = (v.min(), v.max()) if method == "qgis" else np.percentile(v, [2.0, 98.0])
    return float(lo), float(max(hi, lo + 1.0))


def qgis_sample_rows(width, height, sample_size=250000):
    """Rows QGIS's default 'estimated' statistics read from a gdaldem
    hillshade: QGIS asks GDAL for approximate statistics, which read every
    n-th block of the file (n = sqrt(number of blocks)), and gdaldem writes
    strips of max(1, 8192 // width) rows. Every row when the raster is small
    enough (<= 2 x sample_size pixels) for QGIS to ask for exact statistics."""
    if width * height <= 2 * sample_size:
        return np.arange(height)
    rps = max(1, 8192 // max(int(width), 1))
    nblocks = -(-int(height) // rps)
    rate = max(1, int(math.sqrt(float(nblocks))))
    return np.concatenate([np.arange(b * rps, min(height, (b + 1) * rps)) for b in range(0, nblocks, rate)])


def guided_smooth(z, radius, eps):
    """Edge-preserving smoothing: the guided filter (He, Sun & Tang 2013) with
    the surface as its own guide. Where the relief within `radius` px varies
    by much less than sqrt(eps) it is averaged away (noise, grain); where it
    varies by much more - the walls of incisions, raised lines - it is kept."""
    from scipy.ndimage import uniform_filter
    size = 2 * max(1, int(round(radius))) + 1
    z = np.asarray(z, dtype=np.float64)
    m = uniform_filter(z, size, mode="reflect")
    var = np.maximum(uniform_filter(z * z, size, mode="reflect") - m * m, 0.0)
    a = var / (var + float(eps))
    b = m - a * m
    return (uniform_filter(a, size, mode="reflect") * z + uniform_filter(b, size, mode="reflect")).astype(np.float32)


def _local_relief_scale(z, radius):
    """Typical local relief: the median standard deviation of heights within `radius` px."""
    from scipy.ndimage import uniform_filter
    size = 2 * max(1, int(round(radius))) + 1
    z = np.asarray(z, dtype=np.float64)
    m = uniform_filter(z, size, mode="reflect")
    sd = np.sqrt(np.maximum(uniform_filter(z * z, size, mode="reflect") - m * m, 0.0))
    return float(max(np.median(sd[::3, ::3]), 1e-9))


def _lab_surface(z, dx, dy, p, px_scale, plane, origin, stats):
    """The Shade Lab surface: optional tilt removal and smoothing, then the
    Frequency-tab bands (one DCT) re-weighted by lab_weights. Returns
    (surface, the five band surfaces or None when no output needs them)."""
    zf = np.asarray(z, dtype=np.float32)
    if p["lab_detrend"]:
        zf = zf - (plane_surface(fit_plane(zf), zf.shape) if plane is None else
                   plane_surface(plane, zf.shape, origin, px_scale))
    method, rad = p["lab_smoothing"], float(p["lab_smooth_radius"]) / px_scale
    if method == "edge" and rad >= 0.5:
        scale = stats.setdefault("lab_edge_scale", _local_relief_scale(zf, rad))
        zf = guided_smooth(zf, rad, (float(p["lab_edge"]) * scale) ** 2)
    weights = [float(w) for w in p["lab_weights"]]
    need_bands = p["lab_output"].startswith("band") or (p["lab_pca_bands"] and p["lab_output"] in LAB_STACK_OUTPUTS)
    gauss = method == "gaussian" and rad >= 0.3
    if not (need_bands or gauss or any(w != 1.0 for w in weights)):
        return zf, None
    sig = sorted(p["eq_sigmas"], reverse=True)
    spec = Spectrum(zf, _reach(*([v / px_scale for v in sig] + ([rad] if gauss else []))))
    g = [gaussian_multiplier(spec.fy, spec.fx, v / px_scale) for v in sig]
    mults = [g[0], g[1] - g[0], g[2] - g[1], g[3] - g[2], 1.0 - g[3]]
    smooth = gaussian_multiplier(spec.fy, spec.fx, rad) if gauss else np.float32(1.0)
    out = p["lab_output"]
    pca_bands = p["lab_pca_bands"] and out in LAB_STACK_OUTPUTS
    surface = None if (out.startswith("band") or pca_bands) else \
        spec.filtered(sum(np.float32(w) * m for w, m in zip(weights, mults)) * smooth)
    bands = None
    if need_bands:
        want = {int(out[4:])} if out.startswith("band") else {i for i, w in enumerate(weights) if w > 0}
        if p["lab_autolevel"] and "lab_levels" not in stats:
            want = set(range(5))                 # the shared levels need every band once (on the reference crop)
        bands = [spec.filtered(mults[i] * smooth) if i in want else None for i in range(5)]
    return surface, bands


def _lab_hillshade(src, prep, p, azimuth, zf, variant):
    """gdaldem hillshade grey levels (1-255) of one lab surface, edges computed."""
    return gdal_hillshade(src, prep.dx, prep.dy, dict(p, azimuth=azimuth, altitude=p["lab_alt"], gdal_z=zf,
                                                         gdal_scale=1.0, gdal_zt=p["lab_zt"], gdal_edges=True,
                                                         gdal_variant=variant))


def _lab_slopes(src, prep, p, zf):
    """gdaldem's standard hillshade factored so each direction costs a few
    array operations: (base, xs, ys) with
    hillshade(az) = 1 + 254 * max(0, base - ys cos(az) + xs sin(az))."""
    alt = math.radians(float(p["lab_alt"]))
    x, y = gdal_slopes(src, prep.dx, prep.dy, p["lab_zt"], True)
    inv = 1.0 / np.sqrt(1.0 + zf * zf * (x * x + y * y))
    return ((math.sin(alt) * inv).astype(np.float32), (x * (math.cos(alt) * zf) * inv).astype(np.float32),
            (y * (math.cos(alt) * zf) * inv).astype(np.float32))


def _lab_ring(prep, p, azs):
    """One hillshade per direction round the compass (gdaldem's standard
    formula, grey levels) - of the lab surface, or, multi-scale, the weighted
    mix of every included band's auto-levelled hillshade from that direction,
    so fine and broad relief both read in every image of the ring."""
    flat = np.float32(1.0 + 254.0 * math.sin(math.radians(float(p["lab_alt"]))))
    parts = []
    if p["lab_pca_bands"] and prep.bands is not None:
        parts = [(float(w), _lab_slopes(band, prep, p, float(lv) * float(p["lab_z"])))
                 for band, lv, w in zip(prep.bands, prep.lab["levels"], p["lab_weights"])
                 if float(w) > 0 and band is not None]
    if not parts:
        src = prep.surface if prep.surface is not None else np.zeros(prep.shape, np.float32)
        parts = [(1.0, _lab_slopes(src, prep, p, float(p["lab_z"]) * prep.lab.get("comb_level", 1.0)))]
    norm = np.float32(1.0 / max(1.0, sum(w for w, _ in parts)))
    for az in azs:
        ca, sa = np.float32(math.cos(math.radians(az))), np.float32(math.sin(math.radians(az)))
        acc = None
        for w, (base, xs, ys) in parts:
            c = base - (ys * ca - xs * sa)
            dev = np.float32(w) * (np.where(c <= 0, np.float32(1.0), np.float32(1.0) + np.float32(254.0) * c) - flat)
            acc = dev if acc is None else acc + dev
        yield flat + norm * acc


def _lab_stack(prep, p):
    """Hillshades from lab_dirs directions round the compass (_lab_ring),
    reduced one at a time to their mean and first three principal components
    (PCA of hillshades: Devereux, Amable & Crow 2008). The basis, explained
    variance and display limits come from the reference crop (stats), so
    previews and tiles agree."""
    n = int(p["lab_dirs"])
    azs = [360.0 * i / n for i in range(n)]
    basis = prep.stats.get("lab_pca")
    if basis is None:
        x = np.stack([h[::3, ::3].ravel().astype(np.float64) for h in _lab_ring(prep, p, azs)])
        mean = x.mean(axis=1)
        vals, vecs = np.linalg.eigh(np.cov(x))
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order][:, :3]

        def near(target):
            return min(range(n), key=lambda i: abs((azs[i] - target + 180.0) % 360.0 - 180.0))

        # Fixed signs, so the components always read the same way: PC1 like a hillshade lit from the north-west,
        # PC2 like one lit from the north-east, PC3 positive on average.
        for j, ref in enumerate((vecs[near(315.0), 0], vecs[near(45.0), 1], vecs[:, 2].sum())):
            if ref < 0:
                vecs[:, j] = -vecs[:, j]
        proj = vecs.T @ (x - mean[:, None])
        basis = {"mean": mean.tolist(), "vecs": vecs.tolist(),
                 "evr": (vals[:3] / max(float(vals.sum()), 1e-12)).tolist(),
                 "pc_lim": [np.percentile(proj[j], [0.5, 99.5]).tolist() for j in range(proj.shape[0])],
                 "mag_lim": np.percentile(np.hypot(proj[0], proj[1]), [0.5, 99.5]).tolist(),
                 "mean_lim": np.percentile(x.mean(axis=0), [0.5, 99.5]).tolist()}
        prep.stats["lab_pca"] = basis
    mean, vecs = np.asarray(basis["mean"]), np.asarray(basis["vecs"])
    pcs = np.zeros((vecs.shape[1],) + prep.shape, np.float32)
    acc = np.zeros(prep.shape, np.float32)
    for i, h in enumerate(_lab_ring(prep, p, azs)):
        acc += h
        d = h - np.float32(mean[i])
        for j in range(vecs.shape[1]):
            pcs[j] += np.float32(vecs[i, j]) * d
    prep.lab.update(mean=acc / np.float32(n), pcs=pcs, pca=basis)


def _lab_image(prep, p):
    """The selected Shade Lab output as a display image (grey, or RGB for pcrgb)."""
    out, auto = p["lab_output"], p["lab_display"] == "auto"

    def fit(v, lim):
        lo, hi = float(lim[0]), float(lim[1])
        return np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0).astype(np.float32)

    if out == "hillshade" or out.startswith("band"):
        if out == "hillshade":
            src, zf = prep.surface, float(p["lab_z"]) * prep.lab.get("comb_level", 1.0)
        else:
            i = int(out[4:])
            if prep.bands is None:
                return np.zeros(prep.shape, np.float32)
            src, zf = prep.bands[i], float(prep.lab["levels"][i]) * float(p["lab_z"])
        dn = _lab_hillshade(src, prep, p, p["azimuth"], zf, p["lab_variant"])
        if not auto:
            return (dn / np.float32(255.0)).astype(np.float32)
        lim = p["lab_limits"] or np.percentile(dn[::2, ::2], [0.5, 99.5])
        return fit(dn, lim)
    lab = prep.lab
    if "pcs" not in lab:
        return np.zeros(prep.shape, np.float32)
    if out == "mean":
        return fit(lab["mean"], lab["pca"]["mean_lim"]) if auto else (lab["mean"] / np.float32(255.0))
    if out == "pcmag":
        return fit(np.hypot(lab["pcs"][0], lab["pcs"][1]), lab["pca"]["mag_lim"])
    if out == "pcrgb":
        return np.stack([fit(lab["pcs"][j], lab["pca"]["pc_lim"][j]) for j in range(3)], axis=-1)
    j = int(out[2]) - 1
    return fit(lab["pcs"][j], lab["pca"]["pc_lim"][j])


CLEAN_FINE_SIGMA = 6.0       # full-resolution px: relief finer than this is what "Fine detail" lifts
CLEAN_REF_SLOPE = 35.0       # degrees: the slope the clean-relief tone scale is calibrated on


def decode_normals(rgb, green="up"):
    """Unit normals (nx, ny, nz) from a normal map scaled to 0..1 (R = x, G = y, B = z). green 'up' = OpenGL (the
    photometric-stereo standard, HELIO), 'down' = DirectX. Pixels with no data (all 0) come back facing up."""
    rgb = np.asarray(rgb, dtype=np.float32)
    n = rgb * np.float32(2.0) - np.float32(1.0)
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    if green == "down":
        ny = -ny
    empty = rgb.max(axis=-1) <= 0
    nz = np.maximum(nz, np.float32(0.05))            # grazing normals would give unbounded slopes
    if empty.any():
        nx, ny, nz = np.where(empty, 0, nx), np.where(empty, 0, ny), np.where(empty, 1, nz)
    return nx.astype(np.float32), ny.astype(np.float32), nz.astype(np.float32)


def detect_normals_convention(rgb, z):
    """'up' (OpenGL) or 'down' (DirectX): which way the normal map's green channel must be read to agree with the
    height map's own slopes, plus that agreement (correlation, 0..1). Labels are not always right - a HELIO
    normals.png marked OpenGL agrees with its DEM only read the DirectX way."""
    nx, ny, nz = decode_normals(rgb, "up")
    zf = np.asarray(z, dtype=np.float64)
    zf = zf - gaussian_blur(zf, 16)
    gr, gc = np.gradient(zf)
    q = (ny / nz).astype(np.float64)
    q = q - gaussian_blur(q, 16)
    p = (-nx / nz).astype(np.float64)
    p = p - gaussian_blur(p, 16)
    cq = float(np.corrcoef(q.ravel(), gr.ravel())[0, 1])
    cp = float(np.corrcoef(p.ravel(), gc.ravel())[0, 1])
    return ("up" if cq >= 0 else "down"), float(min(abs(cq), abs(cp)))


def integrate_slopes(p, q, flatten=0.0, fine=0.0, fine_sigma=None):
    """Heights from gradients p = dz/dcolumn, q = dz/drow (height units per pixel): the least-squares surface - the
    Poisson equation with mirror boundaries, solved with one DCT - optionally high-passed (drift broader than
    `flatten` px removed) and with relief finer than `fine_sigma` px lifted by `fine`, in the same transform.
    Integrating each window on its own is safe for any render that removes the drift: what differs between
    windows is only the broad form that the high-pass takes away."""
    fine_sigma = CLEAN_FINE_SIGMA if fine_sigma is None else fine_sigma
    p = np.asarray(p, dtype=np.float32) - np.float32(np.median(p))       # no overall tilt
    q = np.asarray(q, dtype=np.float32) - np.float32(np.median(q))
    div = np.gradient(p, axis=1) + np.gradient(q, axis=0)
    spec = Spectrum(div, _reach(flatten, fine_sigma if fine > 0 else 0.0))
    lap = (np.float32(2.0) * np.cos(np.float32(2.0 * math.pi) * spec.fy) +
           np.float32(2.0) * np.cos(np.float32(2.0 * math.pi) * spec.fx) - np.float32(4.0))
    lap[0, 0] = np.float32(1.0)
    mult = np.float32(1.0) / lap
    mult[0, 0] = np.float32(0.0)
    if flatten > 0:
        mult = mult * (np.float32(1.0) - gaussian_multiplier(spec.fy, spec.fx, flatten))
    if fine > 0:
        mult = mult * (np.float32(1.0) + np.float32(fine) * (np.float32(1.0) - gaussian_multiplier(spec.fy, spec.fx,
                                                                                                     fine_sigma)))
    return spec.filtered(mult)


def _clean_surface(z, p, px_scale, plane, origin, stats, normals=None, dx=1.0, dy=1.0):
    """The clean-relief surface. Tilt and broad drift are removed (integrating
    photometric-stereo normals bends a depth map into a ramp or bowl far taller
    than the relief on it) and the finest relief is optionally lifted, both in
    one spectral filter; then edge-preserving smoothing averages away noise and
    grain - including what the lift amplified - but keeps the walls of tooling
    and incisions. With clean_source 'normals' the surface is integrated from the
    normal map in that same filter instead - every detail the normals hold, and
    no drift to begin with."""
    flat, fine = float(p["clean_flatten"]) / px_scale, float(p["clean_fine"])
    if normals is not None and p.get("clean_source") == "normals":
        nx, ny, nz = decode_normals(normals, p.get("normals_green", "up"))
        zf = integrate_slopes(-nx / nz * np.float32(dx), ny / nz * np.float32(dy), flat, fine,
                              CLEAN_FINE_SIGMA / px_scale)
        flat = fine = 0.0
    else:
        zf = np.asarray(z, dtype=np.float32)
        zf = zf - (plane_surface(fit_plane(zf), zf.shape) if plane is None else
                   plane_surface(plane, zf.shape, origin, px_scale))
    if flat > 0 or fine > 0:
        fs = CLEAN_FINE_SIGMA / px_scale
        spec = Spectrum(zf, _reach(flat, fs if fine > 0 else 0.0))
        mult = np.float32(1.0)
        if flat > 0:
            mult = np.float32(1.0) - gaussian_multiplier(spec.fy, spec.fx, flat)
        if fine > 0:
            mult = mult * (np.float32(1.0) + np.float32(fine) * (np.float32(1.0) - gaussian_multiplier(spec.fy, spec.fx, fs)))
        zf = spec.filtered(mult)
    rad = float(p["clean_smooth"]) / px_scale
    if rad >= 0.5:
        scale = stats.setdefault("clean_edge_scale", _local_relief_scale(zf, rad))
        zf = guided_smooth(zf, rad, (float(p["clean_edge"]) * scale) ** 2)
    return np.asarray(zf, dtype=np.float32)


def _clean_image(prep, p):
    """Clean relief: a key light blended with a ring of lights round the compass
    (gdaldem's hillshade formula, each clipped at its shadow line), so every
    edge reads whichever way it runs while the key light keeps the sense of
    depth. The grey is a soft S-curve about the flat level: a flat surface sits
    at clean_mid and the relief swings either side without reaching black or
    white - the matte grey of scanner shaded renders."""
    x, y = prep.normals
    k = np.float32(math.tan(math.radians(min(max(float(p["clean_relief"]), 0.5), 85.0))))
    xs, ys = x * k, y * k
    inv = np.float32(1.0) / np.sqrt(np.float32(1.0) + xs * xs + ys * ys)
    a = min(max(float(p["clean_allround"]), 0.0), 1.0)
    alt, ralt = math.radians(float(p["altitude"])), math.radians(float(p["clean_ring_alt"]))
    out = np.zeros(x.shape, np.float32)

    def add(weight, t, azimuths):
        base, xi, yi = np.float32(math.sin(t)) * inv, xs * (np.float32(math.cos(t)) * inv), ys * (np.float32(math.cos(t)) * inv)
        for az in azimuths:
            az = math.radians(az)
            c = base - yi * np.float32(math.cos(az)) + xi * np.float32(math.sin(az))
            out.__iadd__(np.float32(weight) * np.maximum(c, np.float32(0.0)))

    if a < 1.0:
        add(1.0 - a, alt, [float(p["azimuth"])])
    if a > 0.0:
        n = max(1, int(p["clean_lights"]))
        add(a / n, ralt, [float(p["azimuth"]) + 360.0 * i / n for i in range(n)])
    flat = (1.0 - a) * math.sin(alt) + a * math.sin(ralt)
    th = math.radians(CLEAN_REF_SLOPE)
    ref = abs((1.0 - a) * (max(math.sin(alt - th), 0.0) - math.sin(alt)) + a * math.sin(ralt) * (math.cos(th) - 1.0))
    dev = out - np.float32(flat)
    # A face turned to the light brightens far less than one turned away darkens, so the light side has its own
    # gain; fitted pixel by pixel to archive shaded renders of tooled bindings (their normal maps vs their renders)
    # it is 1.4-1.8, and with it this tone curve fits them as well as any monotone curve can.
    gain = np.where(dev > 0, np.float32(max(float(p["clean_highlight"]), 0.0)), np.float32(1.0))
    t = np.tanh(dev * gain * np.float32(1.2 * float(p["clean_contrast"]) / max(ref, 1e-3)))
    mid = min(max(float(p["clean_mid"]), 0.05), 0.95)
    hi, lo = mid + 0.7 * (1.0 - mid), 0.2 * mid
    return np.where(t > 0, np.float32(mid) + np.float32(hi - mid) * t, np.float32(mid) + np.float32(mid - lo) * t
                    ).astype(np.float32)


def ring_lights(n, altitude, offset=0.0):
    return [((offset + i * 360.0 / n) % 360.0, altitude) for i in range(int(n))]


def light_dome(normals, lights, weights=None):
    """Weighted mean of Lambertian shading over a set of (azimuth, altitude)."""
    weights = weights or [1.0] * len(lights)
    acc = np.zeros_like(normals[2])
    for (az, alt), w in zip(lights, weights):
        acc += np.float32(w) * lambert(normals, light_vector(az, alt))
    return acc / np.float32(sum(weights))


def light_stack_chunks(surface, dx, dy, lights, rows=256):
    """Yield (row0, row1, stack (N, r, W), normals) slab by slab, so an N-light
    stack of a gigapixel surface never has to exist in memory at once. Each
    slab carries a one-row halo, so results match an unchunked computation."""
    h = surface.shape[0]
    vecs = [light_vector(az, alt) for az, alt in lights]
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        a0, a1 = max(0, r0 - 1), min(h, r1 + 1)
        n = surface_normals(surface[a0:a1], dx, dy)
        n = tuple(c[r0 - a0:r0 - a0 + (r1 - r0)] for c in n)
        yield r0, r1, np.stack([lambert(n, v) for v in vecs]), n


def normal_map_rgb(normals, convention="OpenGL"):
    """Tangent-space normal map as uint8 RGB (H, W, 3). OpenGL = green up,
    DirectX = green down."""
    nx, ny, nz = normals
    if convention.lower().startswith("directx"):
        ny = -ny
    return np.dstack([to_uint8(0.5 * c + 0.5) for c in (nx, ny, nz)])


def horizon_metrics(z, dx=1.0, dy=1.0, radius_px=24, n_dirs=16, progress=_noop, openness=False):
    """Horizon scan in n_dirs directions up to radius_px (radii sampled
    geometrically, ~n_dirs x 12 array passes). Returns the sky-view factor
    (Zaksek et al. 2011: 1 - mean sin(horizon elevation), 1 = open sky) and,
    if asked, positive openness in degrees (Yokoyama et al. 2002: mean zenith
    angle of the horizon; 90 = flat, higher on ridges, lower in hollows)."""
    z = np.asarray(z, dtype=np.float32)
    radius_px = max(1, int(round(radius_px)))
    radii = sorted({max(1, int(round(r))) for r in np.geomspace(1, radius_px, num=min(radius_px, 12))})
    h, w = z.shape
    zp = np.pad(z, radius_px, mode="edge")
    svf_acc = np.zeros((h, w), np.float32)
    open_acc = np.zeros((h, w), np.float32) if openness else None
    max_tan = np.empty((h, w), np.float32)
    n_dirs = int(n_dirs)
    for k in range(n_dirs):
        theta = 2.0 * math.pi * k / n_dirs
        ex, ny_ = math.sin(theta), math.cos(theta)
        max_tan.fill(-np.inf)
        seen = set()
        for r in radii:
            ox, oy = int(round(r * ex)), int(round(r * ny_))
            if (ox, oy) in seen or (ox == 0 and oy == 0):
                continue
            seen.add((ox, oy))
            dist = np.float32(math.hypot(ox * dx, oy * dy))
            shifted = zp[radius_px - oy:radius_px - oy + h, radius_px + ox:radius_px + ox + w]
            np.maximum(max_tan, (shifted - z) / dist, out=max_tan)
        up = np.maximum(max_tan, 0.0)
        svf_acc += up / np.sqrt(1.0 + up * up)
        if openness:
            open_acc += 90.0 - np.degrees(np.arctan(max_tan))
        progress(int(100 * (k + 1) / n_dirs), f"Horizon scan: direction {k + 1}/{n_dirs}")
    svf = 1.0 - svf_acc / np.float32(n_dirs)
    return (svf, open_acc / np.float32(n_dirs)) if openness else svf


def sky_view_factor(z, dx=1.0, dy=1.0, radius_px=24, n_dirs=16, progress=_noop):
    """Horizon-based sky-view factor (see horizon_metrics)."""
    return horizon_metrics(z, dx, dy, radius_px, n_dirs, progress)


def topographic_position_index(z, radius_px=1):
    """TPI: height minus the mean of its neighbourhood (ridges > 0 > valleys)."""
    size = 2 * max(1, int(radius_px)) + 1
    z = np.asarray(z, dtype=np.float32)
    if HAS_SCIPY:
        return z - _ndi.uniform_filter(z, size=size, mode="reflect")
    return z - gaussian_blur(z, size / 2.0)


def auto_z_factor(z, dx=1.0, dy=1.0, target_deg=30.0, percentile=95.0):
    """Z-factor that brings the given slope percentile to `target_deg`, so a
    height map in any units renders with sensible relief."""
    gx, gy = horn_gradient(z, dx, dy)
    g = np.hypot(gx, gy)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return 1.0
    ref = float(np.percentile(g, percentile))
    return math.tan(math.radians(target_deg)) / ref if ref > 0 else 1.0


def tone(img, base=0.5, contrast=1.0, gamma=1.0, soft=True):
    """Contrast around `base` with a soft shoulder and toe (Brown: roll values
    off smoothly instead of hard clipping), followed by display gamma."""
    img = np.asarray(img, dtype=np.float32)
    if soft or contrast != 1.0:
        base = min(max(float(base), 1e-3), 1.0 - 1e-3)
        d = (img - np.float32(base)) * np.float32(contrast)
        up = np.float32(1.0 - base) * np.tanh(d / np.float32(1.0 - base))
        down = np.float32(base) * np.tanh(d / np.float32(base))
        img = np.float32(base) + np.where(d > 0, up, down)
    img = np.clip(img, 0.0, 1.0)
    if gamma != 1.0:
        img = img ** np.float32(1.0 / gamma)
    return img.astype(np.float32, copy=False)


def apply_albedo(shade, albedo, strength=1.0, base=0.62, relief=1.0, gamma=1.0, exposure=1.0):
    """Colour composite: exposure x albedo^gamma x (shading / flat grey)^relief - a flat surface keeps its colour,
    the relief lightens and darkens it. Fitted to archive colour composites of tooled bindings (their albedo,
    shaded render and composite, pixel by pixel): gamma 0.53-0.59, exposure 0.76-0.82, relief 0.5-1.8, R^2
    0.88 / 0.98 against 0.59 / 0.70 for the plain product. `strength` fades the colour towards grey. Pixels
    outside the colour image (all bands 0, nodata) keep the grey render."""
    shade = np.asarray(shade, dtype=np.float32)
    rel = np.power(np.maximum(shade / np.float32(base), np.float32(0.0)), np.float32(relief))
    flat = shade[..., None] if shade.ndim == 2 else shade      # coloured lights: already three channels
    if shade.ndim == 2:
        rel = rel[..., None]
    alb = np.asarray(albedo, dtype=np.float32)
    inside = alb.max(axis=-1, keepdims=True) > 0
    if gamma != 1.0:
        alb = np.power(np.clip(alb, 0.0, 1.0), np.float32(gamma))
    col = 1.0 - np.float32(strength) * (1.0 - alb)
    out = np.clip(np.float32(exposure) * col * rel, 0.0, 1.0)
    return np.where(inside, out, flat).astype(np.float32)


def colour_composite(shade, albedo, params, strength=1.0):
    """apply_albedo with a render's own flat grey and colour-composite settings."""
    p = merged_params(params)
    return apply_albedo(shade, albedo, strength, albedo_base(p), float(p["albedo_relief"]), float(p["albedo_gamma"]),
                        float(p["albedo_exposure"]))


def albedo_base(params):
    """Grey of a flat surface in this render - what the colour composite divides the shading by."""
    p = merged_params(params)
    return float(p["clean_mid"]) if p["mode"] == "clean" else float(p["base_tone"])


def tint_image(grey, tint):
    if tuple(tint) == (1.0, 1.0, 1.0):
        return grey
    return np.clip(grey[..., None] * np.asarray(tint, dtype=np.float32), 0.0, 1.0)


# ----------------------------------------------------------------------------
# Relief rendering
# ----------------------------------------------------------------------------

RENDER_MODES = {
    "clean": "Clean relief (all-round hillshade)",
    "gdal": "QGIS hillshade (gdaldem)",
    "lab": "Shade Lab (bands × hillshade × PCA)",
    "studio": "Studio relief render",
    "gis": "GIS hillshade (Global Mapper-style)",
    "multi": "Multi-shade (multi-directional hillshade)",
    "exaggerated": "Exaggerated multi-scale shading",
    "specular": "Specular enhancement (RTI-style)",
    "vat": "Archaeological blend (VAT)",
    "bands": "Frequency band shading",
    "hillshade": "Hillshade (single light)",
    "dome": "Light dome (diffuse slope shading)",
    "texture": "Texture shading (Brown)",
    "svf": "Sky-view factor",
    "normals": "RGB normal map",
}
WHITE = (1.0, 1.0, 1.0)

MULTI_METHODS = {
    "adaptive": "Aspect-adaptive (GDAL / Mark 1992)",
    "average": "Weighted average",
    "max": "Brightest light",
}

# gdaldem hillshade's shading methods (QGIS: Raster > Analysis > Hillshade).
GDAL_VARIANTS = {
    "standard": "Standard (one light)",
    "combined": "Combined shading (slope + oblique)",
    "multidirectional": "Multidirectional shading",
    "igor": "Igor shading",
}

# Shade Lab outputs: QGIS-formula hillshades of chosen frequency bands, and the mean and principal components of
# hillshades lit from every direction round the compass (PCA of hillshades: Devereux, Amable & Crow 2008).
LAB_OUTPUTS = {
    "hillshade": "Hillshade of the chosen bands",
    "band0": "Very-low band hillshade",
    "band1": "Low band hillshade",
    "band2": "Mid band hillshade",
    "band3": "High band hillshade",
    "band4": "Ultra-high band hillshade",
    "mean": "Mean of all directions (direction-free)",
    "pc1": "PCA 1 - the most informative light",
    "pc2": "PCA 2 - the light across it",
    "pc3": "PCA 3 - what is left (fine detail)",
    "pcmag": "PCA 1+2 strength (direction-free)",
    "pcrgb": "PCA 1-2-3 as colour",
}
LAB_STACK_OUTPUTS = ("mean", "pc1", "pc2", "pc3", "pcmag", "pcrgb")
LAB_SMOOTHING = {"none": "None", "gaussian": "Gaussian", "edge": "Edge-preserving (guided filter)"}

# How a gdaldem hillshade is displayed. QGIS styles a new single-band layer with 'stretch to min/max' between the
# minimum and maximum estimated from ~250 000 sampled pixels.
GDAL_DISPLAY = {
    "qgis": "As QGIS shows it (stretch to estimated min/max)",
    "cut": "Punchier: 2-98 % cumulative cut",
    "raw": "gdaldem grey levels 1-255, no stretch",
}

# Global Mapper's ways of combining several light sources (Dynamic Hill Shading).
GIS_BLENDS = {
    "weighted": "Weighted average (key light counts most)",
    "average": "Average",
    "max": "Maximum (brightest light)",
    "min": "Minimum (darkest light)",
}

# (method, [(azimuth, altitude, weight, (r, g, b)), ...])
MULTI_PRESETS = {
    "Aspect-adaptive 4 lights (GDAL multidirectional)": (
        "adaptive", [(225, 45, 1.0, WHITE), (270, 45, 1.0, WHITE), (315, 45, 1.0, WHITE), (0, 45, 1.0, WHITE)]),
    "Raking ring, 8 lights (inscriptions)": ("adaptive", [(a, 15, 1.0, WHITE) for a in range(0, 360, 45)]),
    "Balanced 6 lights (NW-weighted)": (
        "average", [(315, 40, 1.0, WHITE), (270, 40, 0.7, WHITE), (0, 40, 0.7, WHITE), (225, 40, 0.35, WHITE),
                    (45, 40, 0.35, WHITE), (135, 40, 0.1, WHITE)]),
    "RGB orientation (3 coloured lights)": (
        "average", [(0, 35, 1.0, (1.0, 0.0, 0.0)), (120, 35, 1.0, (0.0, 1.0, 0.0)), (240, 35, 1.0, (0.0, 0.0, 1.0))]),
    "Warm key / cool fill": ("average", [(315, 35, 1.0, (1.0, 0.86, 0.66)), (135, 25, 0.45, (0.62, 0.76, 1.0))]),
    "Brightest of 4 lights": ("max", [(a, 30, 1.0, WHITE) for a in (0, 90, 180, 270)]),
}

DEFAULT_PARAMS = {
    "mode": "studio",
    # surface preparation
    "z_factor": 1.0,
    "detrend": True,
    "flatten_sigma": 0.0,        # full-resolution px; 0 = off
    "eq_enabled": False,
    "eq_sigmas": [64.0, 32.0, 16.0, 8.0],
    "eq_gains": [1.0, 1.0, 1.0, 1.0, 1.0],
    "detail_boost": 0.0,         # unsharp-mask gain on the heights (drypoint burr, fine incisions)
    "detail_radius": 2.0,        # full-resolution px
    "smooth_sigma": 0.0,         # full-resolution px; low-pass the heights first (8-bit / noisy height maps)
    # key light, fill, ambient
    "azimuth": 315.0,
    "altitude": 35.0,
    "key": 1.0,
    "fill": 0.2,
    "fill_offset": 180.0,
    "fill_altitude": 30.0,
    "ambient": 0.3,
    "wrap": 0.0,                 # soft-light wrap: 0 = hard Lambert, 0.5 = very soft
    # cast shadows (studio key light)
    "shadows": 0.0,
    "shadow_softness": 3.0,      # apparent size of the light, degrees
    "shadow_reach": 96,          # full-resolution px searched for occluders
    # multi-shade
    "multi_lights": list(MULTI_PRESETS["Aspect-adaptive 4 lights (GDAL multidirectional)"][1]),
    "multi_method": "adaptive",
    # QGIS / gdaldem hillshade, reproduced exactly
    "gdal_z": 1.0,               # -z: vertical exaggeration
    "gdal_scale": 1.0,           # -s: ratio of vertical units to horizontal (111120 for degrees)
    "gdal_variant": "standard",
    "gdal_zt": False,            # Zevenbergen-Thorne gradient instead of Horn's
    "gdal_edges": False,         # -compute_edges
    "gdal_display": "qgis",      # 'qgis': as QGIS shows a new layer; 'cut': 2-98 % cumulative cut; 'raw'
    "gdal_limits": None,         # (low, high) grey levels of that stretch, from whole-surface samples
    # Clean relief: drift removed, edge-preserving smoothing, a key light plus a ring of lights, soft grey tone
    "clean_flatten": 150.0,      # full-resolution px: drift / warp broader than this removed (Gaussian high-pass)
    "clean_smooth": 3.0,         # full-resolution px: edge-preserving smoothing radius (guided filter); 0 = off
    "clean_edge": 0.5,           # relief steps above this share of the typical local relief stay sharp
    "clean_fine": 0.3,           # lift of the relief finer than CLEAN_FINE_SIGMA px
    "clean_relief": 45.0,        # degrees: the steepest 5 % shaded as if this steep (archive shaded renders: 44-45)
    "clean_allround": 0.3,       # share of the ring of lights round the compass (0 = the key light alone)
    "clean_ring_alt": 30.0,      # degrees: height of the ring of lights
    "clean_lights": 16,          # lights in the ring
    "clean_mid": 0.6,            # grey of the flat surface (archive shaded renders: 0.57-0.62)
    "clean_contrast": 1.0,
    "clean_highlight": 1.6,      # extra gain for faces turned to the light (archive shaded renders: 1.4-1.8)
    "clean_source": "height",    # 'height': the height map; 'normals': the photometric-stereo normal map itself
    "normals_green": "up",       # normal-map green channel: 'up' (OpenGL, HELIO) or 'down' (DirectX)
    # colour composite (albedo x relief), fitted to archive colour composites: exposure x albedo^gamma x (S/flat)^relief
    "albedo_relief": 1.0,
    "albedo_gamma": 0.55,
    "albedo_exposure": 0.8,
    # Shade Lab: frequency bands x QGIS-formula hillshades x PCA of many directions
    "lab_output": "hillshade",
    "lab_weights": [1.0, 1.0, 1.0, 1.0, 1.0],   # share of each Frequency-tab band in the lab surface (0 = left out)
    "lab_detrend": False,
    "lab_smoothing": "none",
    "lab_smooth_radius": 1.5,    # full-resolution px
    "lab_edge": 0.3,             # edge-preserving: relief below this share of the typical local relief is smoothed
    "lab_variant": "standard",   # gdaldem shading of the single-light outputs
    "lab_z": 1.0,
    "lab_alt": 45.0,
    "lab_zt": False,
    "lab_autolevel": True,       # band hillshades, and a re-weighted band mix: steepest 5 % raised to 30 degrees
    "lab_dirs": 16,              # directions round the compass for the mean and the PCA
    "lab_pca_bands": False,      # multi-scale ring: each light shades every band, mixed by weight
    "lab_display": "auto",       # 'auto': contrast from the whole surface; 'raw': gdaldem grey levels
    "lab_limits": None,          # (low, high) grey levels for 'auto', from the reference crop
    # GIS hillshade (Global Mapper-style 'Daylight' shader with hill shading)
    "gis_lights": 1,             # light sources spaced evenly round the circle from the azimuth
    "gis_blend": "weighted",
    "gis_ambient": 0.15,         # ambient lighting: lifts every tone
    "gis_shadow": 0.2,           # shadow darkness cap: 0 = shadows may go to black
    "gis_highlight": 0.15,       # highlight from direct light: whitens faces turned to the light
    "gis_slope": 0.0,            # slope-enhanced relief: low slopes washed out, steep slopes keep the shading
    "gis_slope_range": 10.0,     # degrees of slope at which the full shading returns
    "gis_ve": 1.0,               # vertical exaggeration for the hill shading (on top of the Z-factor)
    "gis_brightness": 1.0,       # surface colour: grey of a face turned full to the light (Daylight shader)
    # specular enhancement (RTI-style)
    "spec_depth": 1.0,           # 'normal depth': multiplies every slope before the highlight is computed
    # light dome
    "dome_lights": 16,
    "dome_altitude": 35.0,
    # Brown texture shading
    "texture_detail": 0.6,
    "texture_mix": 0.35,
    "texture_contrast": 1.0,
    "texture_midtone": 0.0,
    "texture_curve": "tanh",
    # occlusion
    "occlusion": 0.0,
    "svf_radius": 24,
    "svf_dirs": 16,
    # material and tone
    "specular": 0.0,
    "shininess": 30.0,
    "clarity": 0.0,              # local contrast
    "clarity_radius": 12.0,      # full-resolution px
    # line / edge enhancement
    "curvature": 0.0,            # radiance-scaling style: ridges brighter, incisions darker
    "curvature_radius": 1.5,     # full-resolution px
    "slope_shade": 0.0,          # steep walls (incision sides) darkened
    "slope_range": 45.0,         # degrees of slope that reach full darkening
    # exaggerated multi-scale shading
    "exag_contrast": 8.0,
    "exag_scales": 5,
    "exag_detail": 0.5,          # > 0 favours fine scales
    "exag_base": 0.2,            # share of ordinary shading of the coarsest surface
    # archaeological blend (VAT)
    "vat_variant": "general",
    # frequency band shading (one shade per Frequency-tab band)
    "band_types": ["hillshade", "hillshade", "multi", "curvature", "texture"],
    "band_weights": [0.5, 0.6, 0.8, 0.6, 0.3],
    "band_altitudes": [45.0, 35.0, 25.0, 30.0, 30.0],
    "band_enabled": [True, True, True, True, True],
    "band_autolevel": True,
    "base_tone": 0.62,
    "contrast": 1.0,
    "gamma": 1.0,
    "tint": WHITE,
    "normal_convention": "OpenGL",
}

# Parameters that change what `prepare` computes; everything else is shading.
PREPARE_KEYS = ("mode", "z_factor", "detrend", "flatten_sigma", "eq_enabled", "eq_sigmas", "eq_gains",
                "detail_boost", "detail_radius", "smooth_sigma", "texture_detail", "svf_radius", "svf_dirs",
                "curvature_radius", "exag_scales", "band_types", "band_enabled", "band_autolevel",
                "lab_output", "lab_weights", "lab_detrend", "lab_smoothing", "lab_smooth_radius", "lab_edge", "lab_z",
                "lab_alt", "lab_zt", "lab_autolevel", "lab_dirs", "lab_pca_bands", "clean_flatten", "clean_smooth",
                "clean_edge", "clean_fine", "clean_source", "normals_green")

COMPONENTS = {
    "final": "Final render",
    "lighting": "Lighting only",
    "shadows": "Cast-shadow mask",
    "texture": "Texture shading",
    "occlusion": "Occlusion (sky view)",
    "curvature": "Curvature (ridges / incisions)",
    "slope": "Slope",
}

EXAG_SIGMA0 = 1.0                        # finest exaggerated-shading scale, full-resolution px

# VAT (Kokalj & Somrak 2019, Relief Visualization Toolbox "VAT - Archaeological"):
# hillshade (normal 100 %) <- slope 0-50 deg, inverted (luminosity 50 %) <- positive
# openness 68-93 deg (overlay 50 %) <- sky-view factor 0.7-1.0 (multiply 25 %).
# "low-relief" narrows the stretches for very flat surfaces.
VAT_VARIANTS = {
    "general": {"slope": (0.0, 50.0), "openness": (68.0, 93.0), "svf": (0.7, 1.0)},
    "low-relief": {"slope": (0.0, 20.0), "openness": (80.0, 92.0), "svf": (0.9, 1.0)},
}

TINTS = {
    "Neutral grey": WHITE,
    "Warm plaster": (1.0, 0.965, 0.9),
    "Cool silver": (0.93, 0.965, 1.0),
    "Parchment": (1.0, 0.94, 0.82),
    "Bronze": (1.0, 0.82, 0.62),
}

LIT_MODES = ("studio", "multi", "exaggerated", "bands")   # modes with texture, occlusion, lines, clarity

BAND_SHADE_TYPES = {
    "off": "Off",
    "hillshade": "Hillshade",
    "multi": "Multi-directional (GDAL)",
    "texture": "Texture shading",
    "curvature": "Curvature",
    "slope": "Slope",
}

# Per band (very low, low, mid, high, ultra-high): (shade type, strength, light altitude).
BAND_PRESETS = {
    "Balanced (form, marks, lines, grain)": [("hillshade", 0.5, 45), ("hillshade", 0.6, 35), ("multi", 0.8, 25),
                                            ("curvature", 0.6, 30), ("texture", 0.3, 30)],
    "Soft form, crisp detail (resolution bumping)": [("hillshade", 0.7, 45), ("hillshade", 0.5, 35),
                                                     ("hillshade", 0.5, 25), ("hillshade", 0.35, 20),
                                                     ("off", 0.0, 30)],
    "Drypoint lines": [("off", 0.0, 45), ("hillshade", 0.25, 35), ("multi", 0.5, 20), ("curvature", 0.8, 30),
                       ("curvature", 0.5, 30)],
    "Inscription (lit from every side)": [("off", 0.0, 45), ("hillshade", 0.3, 35), ("multi", 0.9, 15),
                                          ("multi", 0.6, 12), ("slope", 0.3, 30)],
    "Form only (warp, cockling)": [("hillshade", 1.0, 40), ("hillshade", 0.8, 35), ("off", 0.0, 30),
                                   ("off", 0.0, 30), ("off", 0.0, 30)],
    "Surface texture only": [("off", 0.0, 45), ("off", 0.0, 35), ("texture", 0.6, 30), ("texture", 0.6, 30),
                             ("curvature", 0.4, 30)],
}


def merged_params(params):
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    return p


def needs_texture(p):
    return p["mode"] == "texture" or (p["mode"] in LIT_MODES and p["texture_mix"] > 0)


def needs_svf(p):
    return p["mode"] in ("svf", "vat") or (p["mode"] in LIT_MODES and p["occlusion"] > 0)


def needs_curvature(p):
    return p["mode"] in LIT_MODES and p["curvature"] > 0


def is_rgb(params):
    """True when a render with these parameters produces colour output."""
    p = merged_params(params)
    if p["mode"] == "normals":
        return True
    if p["mode"] == "lab":
        return p["lab_output"] == "pcrgb"
    if p["mode"] == "multi" and any(tuple(c) != WHITE for *_, c in p["multi_lights"]):
        return True
    return tuple(p["tint"]) != WHITE


def spectral_margin_px(params, precise=True):
    """Overlap a tile needs so filters, shadows and SVF match an untiled render.
    precise=False (live previews) trims the band-shading margin for speed."""
    p = merged_params(params)
    sig = [p["flatten_sigma"]] + (list(p["eq_sigmas"]) if p["eq_enabled"] else [])
    margin = 3.0 * max(sig) if sig else 0.0
    if needs_svf(p):
        margin = max(margin, p["svf_radius"] + 2)
    if needs_texture(p):
        margin = max(margin, 128)
    if p["mode"] == "studio" and p["shadows"] > 0:
        margin = max(margin, p["shadow_reach"] + 2)
    if p["mode"] in LIT_MODES and p["clarity"] > 0:
        margin = max(margin, 3.0 * p["clarity_radius"] + 2)
    if p["detail_boost"] > 0:
        margin = max(margin, 3.0 * p["detail_radius"] + 2)
    if p["smooth_sigma"] > 0:
        margin = max(margin, 3.0 * p["smooth_sigma"] + 2)
    if needs_curvature(p):
        margin = max(margin, 3.0 * p["curvature_radius"] + 8)
    if p["mode"] == "exaggerated":
        margin = max(margin, 3.0 * EXAG_SIGMA0 * 2 ** int(p["exag_scales"]) + 2)
    if p["mode"] == "lab":
        split = (any(float(w) != 1.0 for w in p["lab_weights"]) or p["lab_output"].startswith("band")
                 or (p["lab_pca_bands"] and p["lab_output"] in LAB_STACK_OUTPUTS))
        if split:
            margin = max(margin, (4.0 if precise else 3.0) * max(p["eq_sigmas"]) + 2)
        if p["lab_smoothing"] != "none":
            margin = max(margin, 3.0 * p["lab_smooth_radius"] + 4)
    if p["mode"] == "clean":
        margin = max(margin, (4.0 if precise else 3.0) * float(p["clean_flatten"]) + 2,
                     2.0 * float(p["clean_smooth"]) + (4.0 * CLEAN_FINE_SIGMA if p["clean_fine"] > 0 else 0.0) + 4)
    if p["mode"] == "bands":             # auto-level amplifies the broad bands, so tiles need a deep overlap
        margin = max(margin, (5.0 if precise else 3.0) * max(p["eq_sigmas"]) + 2)
    return int(min(max(margin, 32), 1024))


def prepare_surface(z, dx, dy, p, px_scale=1.0, plane=None, origin=(0.0, 0.0), want_texture=False,
                    want_curvature=False):
    """Z-scale, detrend, flatten, equalise and detail-boost a height map.

    Returns (surface, texture_raw_or_None, curvature_or_None). `px_scale` is
    the number of full-resolution pixels per pixel of `z` (preview renders
    pass > 1 so all pixel-based radii keep their full-resolution meaning)."""
    zf = np.asarray(z, dtype=np.float32)
    if p["detrend"]:
        if plane is None:
            zf = zf - plane_surface(fit_plane(zf), zf.shape)
        else:
            zf = zf - plane_surface(plane, zf.shape, origin, px_scale)
    if p["z_factor"] != 1.0:
        zf = zf * np.float32(p["z_factor"])

    boost = p["detail_boost"] > 0 and p["detail_radius"] > 0
    smooth = p["smooth_sigma"] > 0
    need_filter = p["flatten_sigma"] > 0 or p["eq_enabled"] or boost or smooth
    if not (need_filter or want_texture or want_curvature):
        return zf, None, None

    sig = [p["flatten_sigma"]] + (list(p["eq_sigmas"]) if p["eq_enabled"] else [])
    sig += [p["detail_radius"]] if boost else []
    sig += [p["smooth_sigma"]] if smooth else []
    sig += [p["curvature_radius"]] if want_curvature else []
    spec = Spectrum(zf, max(_reach(*[v / px_scale for v in sig]), 32 if want_texture else 0))
    mult = np.float32(1.0)
    if p["eq_enabled"]:
        mult = eq_multiplier(spec.fy, spec.fx, [s / px_scale for s in p["eq_sigmas"]], p["eq_gains"])
    if p["flatten_sigma"] > 0:
        mult = mult * (1.0 - gaussian_multiplier(spec.fy, spec.fx, p["flatten_sigma"] / px_scale))
    if boost:
        fine = 1.0 - gaussian_multiplier(spec.fy, spec.fx, p["detail_radius"] / px_scale)
        mult = mult * (1.0 + np.float32(p["detail_boost"]) * fine)
    if smooth:
        # removes the terracing of 8-bit height maps and photometric-stereo noise before anything is shaded
        mult = mult * gaussian_multiplier(spec.fy, spec.fx, p["smooth_sigma"] / px_scale)
    surface = spec.filtered(mult) if need_filter else zf
    tex = curv = None
    if want_texture:
        tex = spec.filtered(mult * fractional_laplacian_multiplier(spec.fy, spec.fx, p["texture_detail"], dx, dy))
    if want_curvature:
        # -Laplacian of the Gaussian-smoothed surface: > 0 on ridges / burr, < 0 in incisions.
        k2 = np.float32(4.0 * math.pi ** 2) * ((spec.fy / np.float32(dy)) ** 2 + (spec.fx / np.float32(dx)) ** 2)
        smooth = gaussian_multiplier(spec.fy, spec.fx, max(p["curvature_radius"] / px_scale, 0.3))
        curv = spec.filtered(mult * k2 * smooth)
    return surface, tex, curv


class Prepared:
    """Light-independent intermediate products of a render."""

    def __init__(self, mode, shape, valid):
        self.mode = mode
        self.shape = shape
        self.valid = valid
        self.normals = None
        self.surface = None      # kept for cast shadows (studio mode)
        self.tex = None
        self.svf = None
        self.openness = None     # positive openness, degrees (VAT)
        self.curv = None         # curvature (line / edge enhancement)
        self.scales = None       # multi-scale normals (exaggerated shading)
        self.shadow_cache = None # (light key, visible mask) - reused while only non-light settings change
        self.scale_sigmas = None
        self.bands = None        # per-band shading inputs (frequency band shading); band surfaces (Shade Lab)
        self.lab = {}            # Shade Lab: band levels, direction mean / spread, principal components
        self.stats = {}
        self.dx = self.dy = self.px_scale = 1.0


def prepare(z, dx=1.0, dy=1.0, params=None, px_scale=1.0, stats=None, plane=None,
            origin=(0.0, 0.0), progress=_noop, texture=None, occlusion=None, curvature=None, normals=None):
    """Compute everything that does not depend on the lights.

    `stats` carries global normalisation constants (texture scale, SVF
    stretch, curvature scale); pass the stats of a reference render back in so
    tiles and previews share one tone mapping. `texture` / `occlusion` /
    `curvature` force those components on or off (default: only when the
    parameters use them)."""
    p = merged_params(params)
    want_tex = needs_texture(p) if texture is None else texture
    want_svf = needs_svf(p) if occlusion is None else occlusion
    want_curv = needs_curvature(p) if curvature is None else curvature
    z, valid = fill_nodata(z)
    if p["mode"] == "gdal":              # exactly gdaldem: the raw heights, no surface preparation at all
        prep = Prepared("gdal", z.shape, valid)
        prep.dx, prep.dy, prep.px_scale = float(dx), float(dy), float(px_scale)
        prep.stats = dict(stats or {})
        prep.surface = np.asarray(z, dtype=np.float64)
        return prep
    if p["mode"] == "lab":
        prep = Prepared("lab", z.shape, valid)
        prep.dx, prep.dy, prep.px_scale = float(dx), float(dy), float(px_scale)
        prep.stats = dict(stats or {})
        progress(10, "Frequency bands")
        prep.surface, prep.bands = _lab_surface(z, dx, dy, p, px_scale, plane, origin, prep.stats)
        levels = [1.0] * 5
        if prep.bands is not None and p["lab_autolevel"]:
            levels = prep.stats.get("lab_levels")
            if levels is None:
                levels = []
                for band in prep.bands:
                    gx, gy = horn_gradient(band, dx, dy)
                    ref = float(np.percentile(np.hypot(gx, gy)[::2, ::2], 95.0))
                    levels.append(math.tan(math.radians(30.0)) / ref if ref > 1e-12 else 1.0)
                prep.stats["lab_levels"] = levels
        prep.lab["levels"] = list(levels)
        comb = 1.0          # with every band at 1 the lab surface is the true relief (exactly QGIS's hillshade)
        if p["lab_autolevel"] and prep.surface is not None and any(float(w) != 1.0 for w in p["lab_weights"]):
            comb = prep.stats.get("lab_comb_level")
            if comb is None:
                gx, gy = horn_gradient(prep.surface, dx, dy)
                ref = float(np.percentile(np.hypot(gx, gy)[::2, ::2], 95.0))
                comb = math.tan(math.radians(30.0)) / ref if ref > 1e-12 else 1.0
                prep.stats["lab_comb_level"] = comb
        prep.lab["comb_level"] = float(comb)
        if p["lab_output"] in LAB_STACK_OUTPUTS:
            progress(50, "Hillshades from every direction")
            _lab_stack(prep, p)
        return prep
    if p["mode"] == "clean":
        prep = Prepared("clean", z.shape, valid)
        prep.dx, prep.dy, prep.px_scale = float(dx), float(dy), float(px_scale)
        prep.stats = dict(stats or {})
        progress(20, "Removing drift, smoothing")
        surface = _clean_surface(z, p, px_scale, plane, origin, prep.stats, normals, dx, dy)
        x, y = gdal_slopes(surface, dx, dy, False, True)
        level = prep.stats.get("clean_level")
        if level is None:            # slopes in units of the steepest 5 %, so any height units shade alike
            ref = float(np.percentile(np.hypot(x, y)[::2, ::2], 95.0))
            level = 1.0 / ref if ref > 1e-12 else 1.0
            prep.stats["clean_level"] = level
        prep.normals = ((x * level).astype(np.float32), (y * level).astype(np.float32))
        return prep
    progress(5, "Preparing surface")
    surface, tex, curv = prepare_surface(z, dx, dy, p, px_scale, plane, origin, want_tex, want_curv)
    prep = Prepared(p["mode"], surface.shape, valid)
    prep.dx, prep.dy, prep.px_scale = float(dx), float(dy), float(px_scale)
    prep.stats = dict(stats or {})
    if p["mode"] == "studio":
        prep.surface = surface
    if tex is not None:
        prep.tex = tex
        prep.stats.setdefault("texture_scale", texture_scale(tex))
    if curv is not None:
        prep.curv = curv
        prep.stats.setdefault("curv_scale", texture_scale(curv))
    if want_svf:
        progress(30, "Sky-view factor")
        scan = horizon_metrics(surface, dx, dy, max(1.0, p["svf_radius"] / px_scale), p["svf_dirs"],
                               lambda pc, m: progress(30 + pc // 2, m), openness=p["mode"] == "vat")
        svf = scan[0] if p["mode"] == "vat" else scan
        if p["mode"] == "vat":
            prep.openness = scan[1]
        prep.svf = svf
        prep.stats.setdefault("svf_low", float(np.percentile(svf[::4, ::4], 1.0)))
    if p["mode"] not in ("texture", "svf"):
        progress(85, "Surface normals")
        prep.normals = surface_normals(surface, dx, dy)
        if p["mode"] == "dome":
            # A full ring of lights cancels directional shading and leaves slope
            # magnitude (1 - n_z); this light-independent scale stretches it.
            steep = float(np.percentile((1.0 - prep.normals[2])[::2, ::2], 99.0))
            prep.stats.setdefault("slope_scale", max(steep, 1e-6))
        if p["mode"] == "exaggerated":
            progress(90, "Multi-scale normals")
            prep.scales, prep.scale_sigmas = exaggerated_normals(surface, dx, dy, p["exag_scales"], px_scale)
        if p["mode"] == "bands":
            progress(88, "Frequency bands")
            prep.bands = _prepare_bands(surface, dx, dy, p, px_scale, prep.stats)
    return prep


def _prepare_bands(surface, dx, dy, p, px_scale, stats):
    """Split the prepared surface into the five Frequency-tab bands (one DCT)
    and compute what each band's shading type needs. With auto-level each band
    is scaled so its steepest 5 % reaches 30 degrees, so fine bands shade as
    clearly as broad ones; levels and scales come from the reference crop when
    `stats` has them, so previews, thumbnails and tiles share one tone."""
    sig = sorted(p["eq_sigmas"], reverse=True)
    spec = Spectrum(surface, _reach(*[v / px_scale for v in sig + [p["curvature_radius"]]]))
    fy, fx = spec.fy, spec.fx
    g = [gaussian_multiplier(fy, fx, v / px_scale) for v in sig]
    mults = [g[0], g[1] - g[0], g[2] - g[1], g[3] - g[2], 1.0 - g[3]]
    types, enabled = list(p["band_types"]), list(p["band_enabled"])
    have = "band_levels" in stats
    levels = list(stats.get("band_levels") or [1.0] * 5)
    tex_scales = list(stats.get("band_tex_scale") or [None] * 5)
    curv_scales = list(stats.get("band_curv_scale") or [None] * 5)
    out = [None] * 5
    # When the shared constants are being computed (no stats yet), do it for every band and every
    # kind of shade, so switching a band's shade type later reuses them instead of recomputing.
    fl = k2s = None
    for i in range(5):
        t = types[i]
        active = enabled[i] and t != "off"
        if not active and have:
            continue
        band = tex = curv = None
        if (active and t in ("hillshade", "multi", "slope")) or (not have and p["band_autolevel"]):
            band = spec.filtered(mults[i])
            if p["band_autolevel"] and not have:
                gx, gy = horn_gradient(band, dx, dy)
                ref = float(np.percentile(np.hypot(gx, gy)[::2, ::2], 95.0))
                levels[i] = math.tan(math.radians(30.0)) / ref if ref > 1e-12 else 1.0
        if (active and t == "texture") or not have:
            if fl is None:
                fl = fractional_laplacian_multiplier(fy, fx, p["texture_detail"], dx, dy)
            tex = spec.filtered(mults[i] * fl)
            if tex_scales[i] is None:
                tex_scales[i] = texture_scale(tex)
        if (active and t == "curvature") or not have:
            if k2s is None:
                k2s = (np.float32(4.0 * math.pi ** 2) * ((fy / np.float32(dy)) ** 2 + (fx / np.float32(dx)) ** 2)
                       * gaussian_multiplier(fy, fx, max(p["curvature_radius"] / px_scale, 0.3)))
            curv = spec.filtered(mults[i] * k2s)
            if curv_scales[i] is None:
                curv_scales[i] = texture_scale(curv)
        if not active:
            continue
        entry = {"type": t}
        if t in ("hillshade", "multi", "slope"):
            lv = np.float32(levels[i] if p["band_autolevel"] else 1.0)
            entry["normals"] = surface_normals(band * lv, dx, dy)
        elif t == "texture":
            entry["tex"], entry["scale"] = tex, tex_scales[i]
        else:
            entry["curv"], entry["scale"] = curv, curv_scales[i]
        out[i] = entry
    stats.setdefault("band_levels", levels)
    stats.setdefault("band_tex_scale", tex_scales)
    stats.setdefault("band_curv_scale", curv_scales)
    return out


def _band_image(entry, p, altitude):
    """One band's shading as a grey image; a level surface gives 0.5."""
    t = entry["type"]
    if t == "hillshade":
        lam = lambert(entry["normals"], light_vector(p["azimuth"], altitude))
        return np.clip(0.5 * lam / np.float32(max(math.sin(math.radians(altitude)), 1e-3)), 0.0, 1.0)
    if t == "multi":
        rot = p["azimuth"] - 315.0
        lights = [((a + rot) % 360.0, altitude, 1.0, WHITE) for a in (225, 270, 315, 0)]
        lum, flat = _multi_light(entry["normals"], lights, "adaptive")
        return np.clip(0.5 * lum / np.float32(max(flat, 1e-3)), 0.0, 1.0)
    if t == "texture":
        return texture_to_image(entry["tex"], p["texture_contrast"], 0.0, p["texture_curve"], entry["scale"])
    if t == "curvature":
        k = np.tanh(np.float32(0.8) * entry["curv"] / np.float32(max(entry["scale"], 1e-12)))
        return (0.5 + 0.5 * k).astype(np.float32)
    slope = np.degrees(np.arccos(np.clip(entry["normals"][2], 0.0, 1.0)))
    return (0.5 - 0.5 * np.clip(slope / np.float32(max(p["slope_range"], 1e-3)), 0.0, 1.0)).astype(np.float32)


def _band_shading(prep, p):
    """Mix the bands' shades: 0.5 + sum_i w_i * (shade_i - 0.5), where the
    strengths act as mix weights - if they add up to more than 1 they are
    scaled down together so the combination does not saturate (overall punch
    is the Contrast control's job)."""
    active = [(i, e) for i, e in enumerate(prep.bands or []) if e is not None and float(p["band_weights"][i]) > 0]
    total = sum(float(p["band_weights"][i]) for i, _ in active)
    norm = np.float32(1.0 / max(1.0, total))
    acc = np.full(prep.shape, 0.5, np.float32)
    for i, entry in active:
        acc += norm * np.float32(p["band_weights"][i]) * (_band_image(entry, p, float(p["band_altitudes"][i])) - 0.5)
    return acc, 0.5


def exaggerated_normals(surface, dx, dy, n_scales=5, px_scale=1.0):
    """Normals of the surface smoothed at sigma = 2^i full-resolution px,
    i = 0 .. n_scales (one DCT shared by every scale)."""
    spec = Spectrum(surface, _reach(EXAG_SIGMA0 * 2 ** int(n_scales) / px_scale))
    scales, sigmas = [], []
    for i in range(int(n_scales) + 1):
        sig = EXAG_SIGMA0 * 2 ** i
        eff = sig / px_scale
        zs = surface if eff < 0.35 else spec.filtered(gaussian_multiplier(spec.fy, spec.fx, eff))
        scales.append(surface_normals(zs, dx, dy))
        sigmas.append(sig)
    return scales, sigmas


def _exaggerated(prep, light, p):
    """Exaggerated shading (Rusinkiewicz, Burns & DeCarlo 2006). At every scale
    the light is made grazing by projecting it onto the plane perpendicular to
    the next-coarser smoothed normal; clamp(a * n_i . l_i+1) is summed with
    weights sigma_i^-detail, so detail at every scale reads regardless of the
    overall slope. A level surface gets 0.5."""
    scales, sigmas = prep.scales, prep.scale_sigmas
    lx, ly, lz = (np.float32(v) for v in light)
    a = np.float32(p["exag_contrast"])
    ks = np.array([s ** (-float(p["exag_detail"])) for s in sigmas[:-1]], dtype=np.float64)
    ks /= ks.sum()
    acc = np.zeros(prep.shape, np.float32)
    for i in range(len(scales) - 1):
        nx, ny, nz = scales[i]
        cx, cy, cz = scales[i + 1]
        d = lx * cx + ly * cy + lz * cz
        px, py, pz = lx - d * cx, ly - d * cy, lz - d * cz
        inv = 1.0 / np.sqrt(px * px + py * py + pz * pz + np.float32(1e-12))
        acc += np.float32(ks[i]) * np.clip(a * (nx * px + ny * py + nz * pz) * inv, -1.0, 1.0)
    e = 0.5 + 0.5 * acc
    base = float(p["exag_base"])
    if base > 0:
        coarse = lambert(scales[-1], light) / np.float32(max(math.sin(math.radians(p["altitude"])), 1e-3))
        e = (1.0 - np.float32(base)) * e + np.float32(base * 0.5) * coarse
    return e, 0.5


def _wrap_lambert(normals, light, wrap=0.0):
    """Lambert with optional 'wrap' - the softer roll-off of a large, diffused
    light on a matte surface such as plaster."""
    if wrap <= 0:
        return lambert(normals, light)
    nx, ny, nz = normals
    lx, ly, lz = (np.float32(v) for v in light)
    w = np.float32(wrap)
    return np.clip((nx * lx + ny * ly + nz * lz + w) / (1.0 + w), 0.0, 1.0)


def _flat_level(altitude, wrap=0.0):
    return (math.sin(math.radians(altitude)) + wrap) / (1.0 + wrap)


def cast_shadows(z, dx, dy, azimuth, altitude, reach_px, softness_deg=3.0):
    """Fraction of the light visible from each pixel (1 = lit, 0 = shadow).

    Scans the surface towards the light for the highest horizon; the light is
    treated as a disc `softness_deg` across, so shadow edges get a penumbra."""
    z = np.asarray(z, dtype=np.float32)
    h, w = z.shape
    reach = max(1, int(round(reach_px)))
    radii = sorted({max(1, int(round(r))) for r in np.geomspace(1, reach, num=min(reach, 24))})
    a = math.radians(azimuth)
    ex, ny_ = math.sin(a), math.cos(a)
    zp = np.pad(z, reach, mode="edge")
    max_tan = np.zeros((h, w), np.float32)
    seen = set()
    for r in radii:
        ox, oy = int(round(r * ex)), int(round(r * ny_))
        if (ox, oy) in seen or (ox == 0 and oy == 0):
            continue
        seen.add((ox, oy))
        dist = np.float32(math.hypot(ox * dx, oy * dy))
        shifted = zp[reach - oy:reach - oy + h, reach + ox:reach + ox + w]
        np.maximum(max_tan, (shifted - z) / dist, out=max_tan)
    horizon = np.degrees(np.arctan(max_tan))
    half = max(float(softness_deg), 0.2) / 2.0
    return np.clip((np.float32(altitude) - horizon + half) / np.float32(2.0 * half), 0.0, 1.0)


def _multi_light(normals, lights, method="adaptive", wrap=0.0):
    """Combine several lights. Returns (lum, flat): lum is (H, W) - or (H, W, 3)
    when any light is coloured - and flat is the value a level surface gets.

    'adaptive' weights each light per pixel by how squarely it faces the slope,
    (n_h . l_h)^2 (Mark 1992); with lights at 225/270/315/360 degrees this is
    GDAL's multidirectional hillshade. 'average' uses fixed weights and 'max'
    keeps the brightest light."""
    lights = [(float(az), float(alt), float(w), tuple(float(v) for v in c))
              for az, alt, w, c in lights if w > 0] or [(315.0, 45.0, 1.0, WHITE)]
    colour = any(c != WHITE for *_, c in lights)
    nx, ny, _ = normals
    ch = 3 if colour else 1
    acc = np.zeros(nx.shape + (ch,), np.float32)
    den = np.zeros(nx.shape + (ch,), np.float32) if method == "adaptive" else np.zeros(ch, np.float32)
    flat_num, flat_den, flat_max = np.zeros(ch), np.zeros(ch), np.zeros(ch)
    for az, alt, w, c in lights:
        cv = np.asarray(c if colour else (1.0,), dtype=np.float32)
        lam = _wrap_lambert(normals, light_vector(az, alt), wrap)
        fl = _flat_level(alt, wrap)
        if method == "max":
            np.maximum(acc, (np.float32(w) * lam)[..., None] * cv, out=acc)
            flat_max = np.maximum(flat_max, w * fl * cv)
            continue
        if method == "adaptive":
            s, co = np.float32(math.sin(math.radians(az))), np.float32(math.cos(math.radians(az)))
            wk = np.float32(w) * ((nx * s + ny * co) ** 2 + np.float32(1e-6))
            acc += (wk * lam)[..., None] * cv
            den += wk[..., None] * cv
        else:
            acc += (np.float32(w) * lam)[..., None] * cv
            den += np.float32(w) * cv
        flat_num += w * fl * cv
        flat_den += w * cv
    if method == "max":
        lum, flat = acc, flat_max
    else:
        lum = acc / np.maximum(den, np.float32(1e-12))
        flat = flat_num / np.maximum(flat_den, 1e-12)
    if not colour:
        return lum[..., 0], float(flat[0])
    return lum, flat.astype(np.float32)


def gis_ring(n, azimuth, altitude):
    """`n` lights spaced evenly round a circle, starting at `azimuth` - how
    Global Mapper places its 'multiple light sources' (4 lights from 45 deg:
    45, 135, 225, 315)."""
    n = max(1, int(n))
    return [((float(azimuth) + 360.0 * i / n) % 360.0, float(altitude)) for i in range(n)]


def _smooth_floor(x, floor, knee=0.04):
    """max(x, floor) with a rounded corner, so capped shadows keep a trace of
    their relief instead of flattening to one grey."""
    if floor <= 0:
        return x
    f, k = np.float32(floor), np.float32(knee)
    return 0.5 * (x + f + np.sqrt((x - f) ** 2 + k * k))


def _gis_hillshade(normals, p):
    """Hill shading as in GIS packages such as Global Mapper (the 'Daylight'
    shader: every height the same colour, relief from the light alone).

    Lambert shading for `gis_lights` lights spaced evenly from the azimuth,
    all at the key altitude, combined by weighted average (the key light
    counts as much as all the others together), average, maximum or minimum.
    Then 'ambient lighting' lifts every tone, 'shadow darkness' caps how dark
    shadows may get (a soft floor, so it bites only on steep relief and leaves
    low relief untouched), 'highlight from direct light' whitens faces turned
    squarely to the key light, and 'slope-enhanced' relief (Brown, Utah
    Geological Survey) screens a slope mask over the shade so near-level
    ground turns white while steep walls keep their shading.
    Returns (image, the value a level surface gets)."""
    ve = float(p["gis_ve"])
    if ve != 1.0:
        nx, ny, nz = normals
        v = np.float32(ve)
        inv = 1.0 / np.sqrt((v * nx) ** 2 + (v * ny) ** 2 + nz * nz)
        normals = (v * nx * inv, v * ny * inv, nz * inv)
    alt = float(p["altitude"])
    ring = gis_ring(p["gis_lights"], p["azimuth"], alt)
    n = len(ring)
    blend = p["gis_blend"] if n > 1 else "average"
    weights = [0.5] + [0.5 / (n - 1)] * (n - 1) if blend == "weighted" and n > 1 else [1.0 / n] * n
    acc = key = None
    for (az, a), wt in zip(ring, weights):
        lam = lambert(normals, light_vector(az, a))
        if key is None:
            key = lam
        if acc is None:
            acc = lam if blend in ("max", "min") else np.float32(wt) * lam
        elif blend == "max":
            acc = np.maximum(acc, lam)
        elif blend == "min":
            acc = np.minimum(acc, lam)
        else:
            acc = acc + np.float32(wt) * lam
    flat0 = math.sin(math.radians(alt))              # every light shares the altitude
    amb = np.float32(p["gis_ambient"])
    img, flat = amb + (1.0 - amb) * acc, float(amb + (1.0 - amb) * flat0)
    if p["gis_highlight"] > 0 and flat0 < 0.999:
        t = np.clip((key - np.float32(flat0)) / np.float32(1.0 - flat0), 0.0, 1.0)
        img = img + (1.0 - img) * np.float32(p["gis_highlight"]) * t * t
    floor = float(p["gis_shadow"])
    img = _smooth_floor(img, floor)
    flat = float(_smooth_floor(np.float32(flat), floor))
    if p["gis_slope"] > 0:
        slope = np.degrees(np.arccos(np.clip(normals[2], 0.0, 1.0)))
        wash = np.float32(p["gis_slope"]) * np.clip(1.0 - slope / np.float32(max(p["gis_slope_range"], 0.1)),
                                                    0.0, 1.0)
        img = 1.0 - (1.0 - img) * (1.0 - wash)          # screen blend
        flat = 1.0 - (1.0 - flat) * (1.0 - float(p["gis_slope"]))
    bright = float(p["gis_brightness"])
    return (img * np.float32(bright)).astype(np.float32), flat * bright


def fit_gis_to_reference(normals, ref, mask=None, max_samples=200_000):
    """Find the GIS hill shading that best reproduces a reference grey render
    of the same surface - for example the 'shaded render' a photometric-stereo
    scanner delivers with its depth map. Fits the light azimuth and altitude,
    the hill-shading exaggeration and the tone (grey = brightness * (ambient +
    (1 - ambient) * lambert)) by least squares.

    `normals` come from `prepare` of the height map; `ref` is the reference
    for the same pixels (grey or RGB; 8-bit, 16-bit or 0-1). Returns GIS
    parameters plus 'r2', the share of the reference's variance reproduced."""
    from scipy import optimize

    y = np.asarray(ref, dtype=np.float64)
    if y.ndim == 3:
        y = y[..., :3].mean(axis=2)
    top = np.nanmax(y)
    if top > 1.5:
        y = y / (65535.0 if top > 255.5 else 255.0)
    m = np.isfinite(y) & (y > 0.004) & (y < 0.996)
    if mask is not None:
        m &= mask
    idx = np.flatnonzero(m)
    if idx.size < 500:
        raise ValueError("Too few usable pixels in the reference (is it the same area as the height map?)")
    if idx.size > max_samples:
        idx = np.random.default_rng(0).choice(idx, max_samples, replace=False)
    px, py, pz = (np.asarray(a, dtype=np.float64).ravel()[idx] for a in normals)
    ys = y.ravel()[idx]

    def lam(az, alt, lk):
        k = math.exp(lk)
        l = light_vector(az, min(max(alt, 0.5), 90.0))
        inv = 1.0 / np.sqrt((k * px) ** 2 + (k * py) ** 2 + pz * pz)
        return np.clip((k * px * l[0] + k * py * l[1] + pz * l[2]) * inv, 0.0, 1.0)

    def solve(sh):
        a = np.stack([np.ones_like(sh), sh], 1)
        c, *_ = np.linalg.lstsq(a, ys, rcond=None)
        return c, float(((a @ c - ys) ** 2).mean())

    def cost(v):
        return solve(lam(*v))[1]

    start = min(((cost((az, alt, lk)), (az, alt, lk)) for az in range(0, 360, 30) for alt in (20, 35, 50, 65, 80)
                 for lk in (-1.0, 0.0, 1.0, 2.0, 3.0)), key=lambda t: t[0])[1]
    res = optimize.minimize(cost, start, method="Nelder-Mead",
                            options={"xatol": 0.02, "fatol": 1e-11, "maxiter": 1000})
    az, alt, lk = res.x
    alt = min(max(alt, 0.5), 90.0)
    (a, b), mse = solve(lam(az, alt, lk))
    bright = a + b
    amb = a / bright if abs(bright) > 1e-6 else 0.0
    return {"azimuth": float(az % 360.0), "altitude": float(alt), "gis_ve": float(math.exp(lk)),
            "gis_ambient": float(min(max(amb, 0.0), 1.0)), "gis_brightness": float(min(max(bright, 0.05), 1.5)),
            "gis_lights": 1, "gis_shadow": 0.0, "gis_highlight": 0.0, "gis_slope": 0.0,
            "r2": float(1.0 - mse / max(float(ys.var()), 1e-12))}


def _specular_enhancement(normals, p):
    """Specular enhancement (Malzbender et al. 2006) for photometric-stereo
    recordings: a Blinn-Phong material with a black diffuse colour and a white highlight
    (shininess 30, three.js normalisation), lit by the key light on normals
    whose slope is multiplied by `spec_depth`. Only the glints remain, so the
    slightest change of slope flashes light or dark; with an overhead light a
    level surface is white and every incision a dark line.
    Returns (image, the value a level surface gets)."""
    nx, ny, nz = normals
    d = np.float32(max(float(p["spec_depth"]), 1e-3))
    inv = 1.0 / np.sqrt((d * nx) ** 2 + (d * ny) ** 2 + nz * nz)
    ex = (d * nx * inv, d * ny * inv, nz * inv)
    lv = light_vector(p["azimuth"], p["altitude"])
    hv = np.array([lv[0], lv[1], lv[2] + 1.0])
    hv /= np.linalg.norm(hv)
    sh = float(p["shininess"])
    k = float(p["key"]) * (0.5 * sh + 1.0) / math.pi
    img = np.float32(k) * lambert(ex, lv) * lambert(ex, hv) ** np.float32(sh)
    flat = k * float(lv[2]) * float(hv[2]) ** sh
    return np.minimum(img, np.float32(1.0)), min(flat, 1.0)


def _clarity(lum, amount, radius_px):
    """Local contrast (unsharp mask at a mid radius) for crisp surface texture."""
    if amount <= 0 or radius_px < 0.3:
        return lum
    if lum.ndim == 3:
        base = np.stack([gaussian_blur(lum[..., c], radius_px) for c in range(lum.shape[2])], axis=-1)
    else:
        base = gaussian_blur(lum, radius_px)
    return lum + np.float32(amount) * (lum - base)


def _modulate(img, factor):
    return img * (factor[..., None] if img.ndim == 3 else factor)


def _studio(prep, p, component=None):
    """Lit relief model shared by the studio, multi-shade and exaggerated modes:
    lighting (key / fill / ambient with soft wrap and soft cast shadows, a
    multi-shade rig, or exaggerated multi-scale shading), normalised so a level
    surface sits at `base_tone`; then Brown texture shading, sky-view
    occlusion, slope darkening, curvature enhancement (radiance-scaling style)
    and clarity; soft tone mapping. With `component` set, returns that single
    ingredient as a grey image (None when it is not available)."""
    normals, wrap, amb = prep.normals, float(p["wrap"]), float(p["ambient"])
    lit = None
    if prep.mode == "multi":
        lum, flat = _multi_light(normals, p["multi_lights"], p["multi_method"], wrap)
        lum = lum + np.float32(amb)
        flat = flat + amb
    elif prep.mode == "exaggerated":
        lum, flat = _exaggerated(prep, light_vector(p["azimuth"], p["altitude"]), p)
    elif prep.mode == "bands":
        if component and component.startswith("band"):
            entry = (prep.bands or [None] * 5)[int(component[4:])]
            return None if entry is None else _band_image(entry, p, float(p["band_altitudes"][int(component[4:])]))
        lum, flat = _band_shading(prep, p)
    else:
        az, alt = p["azimuth"], p["altitude"]
        key_l = light_vector(az, alt)
        lum = np.full(prep.shape, np.float32(amb), dtype=np.float32)
        flat = amb
        if (p["shadows"] > 0 or component == "shadows") and prep.surface is not None and alt < 89.5:
            skey = (round(float(az), 4), round(float(alt), 4), float(p["shadow_reach"]), float(p["shadow_softness"]))
            if prep.shadow_cache is not None and prep.shadow_cache[0] == skey:
                visible = prep.shadow_cache[1]
            else:
                visible = cast_shadows(prep.surface, prep.dx, prep.dy, az, alt,
                                       max(1.0, p["shadow_reach"] / prep.px_scale), p["shadow_softness"])
                prep.shadow_cache = (skey, visible)
            if component == "shadows":
                return visible
            lit = 1.0 - np.float32(p["shadows"]) * (1.0 - visible)
        if p["key"] > 0:
            key = _wrap_lambert(normals, key_l, wrap)
            lum += np.float32(p["key"]) * (key * lit if lit is not None else key)
            flat += p["key"] * _flat_level(alt, wrap)
        if p["fill"] > 0:
            fa = p["fill_altitude"]
            lum += np.float32(p["fill"]) * _wrap_lambert(normals, light_vector(az + p["fill_offset"], fa), wrap)
            flat += p["fill"] * _flat_level(fa, wrap)
        if p["specular"] > 0:
            hv = np.array([key_l[0], key_l[1], key_l[2] + 1.0])
            hv /= np.linalg.norm(hv)
            spec = lambert(normals, hv) ** np.float32(p["shininess"])
            lum += np.float32(p["specular"]) * (spec * lit if lit is not None else spec)
            flat += p["specular"] * hv[2] ** p["shininess"]
    lum = lum * (np.float32(p["base_tone"]) / np.maximum(np.asarray(flat, dtype=np.float32), np.float32(1e-6)))
    if component == "lighting":
        return tone(lum if lum.ndim == 2 else lum.mean(axis=2), p["base_tone"], 1.0, 1.0, soft=True)
    if component == "shadows":
        return np.ones(prep.shape, np.float32)          # this mode casts no shadows

    st = prep.stats
    if prep.tex is not None and (p["texture_mix"] > 0 or component == "texture"):
        t_img = texture_to_image(prep.tex, p["texture_contrast"], p["texture_midtone"], p["texture_curve"],
                                 st.get("texture_scale"))
        if component == "texture":
            return t_img
        mix = np.float32(p["texture_mix"])
        lum = _modulate(lum, (1.0 - mix) + 2.0 * mix * t_img)
    elif component == "texture":
        return None
    if prep.svf is not None and (p["occlusion"] > 0 or component == "occlusion"):
        low = st.get("svf_low", 0.0)
        dark = np.clip((1.0 - prep.svf) / max(1e-6, 1.0 - low), 0.0, 1.0)
        if component == "occlusion":
            return (1.0 - dark).astype(np.float32)
        lum = _modulate(lum, 1.0 - np.float32(p["occlusion"]) * dark)
    elif component == "occlusion":
        return None
    if p["slope_shade"] > 0 or component == "slope":
        slope = np.degrees(np.arccos(np.clip(normals[2], 0.0, 1.0)))
        steep = np.clip(slope / np.float32(max(p["slope_range"], 1e-3)), 0.0, 1.0)
        if component == "slope":
            return (1.0 - steep).astype(np.float32)
        lum = _modulate(lum, 1.0 - np.float32(p["slope_shade"]) * steep)
    if prep.curv is not None and (p["curvature"] > 0 or component == "curvature"):
        k = np.tanh(np.float32(0.8) * prep.curv / np.float32(max(st.get("curv_scale", 1.0), 1e-12)))
        if component == "curvature":
            return (0.5 + 0.5 * k).astype(np.float32)
        lum = _modulate(lum, 1.0 + np.float32(p["curvature"]) * k)
    elif component == "curvature":
        return None
    lum = _clarity(lum, p["clarity"], p["clarity_radius"] / prep.px_scale)
    return tone(lum, p["base_tone"], p["contrast"], p["gamma"], soft=True)


def _vat(prep, p):
    """Visualisation for Archaeological Topography - the four-layer blend of
    the Relief Visualization Toolbox (see VAT_VARIANTS)."""
    v = VAT_VARIANTS.get(p["vat_variant"], VAT_VARIANTS["general"])

    def stretch(a, lo_hi):
        lo, hi = lo_hi
        return np.clip((a - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0)

    normals = prep.normals
    out = lambert(normals, light_vector(p["azimuth"], p["altitude"]))                 # hillshade, normal 100 %
    slope = np.degrees(np.arccos(np.clip(normals[2], 0.0, 1.0)))
    out = 0.5 * out + 0.5 * (1.0 - stretch(slope, v["slope"]))                         # slope, luminosity 50 %
    op = stretch(prep.openness, v["openness"])
    ov = np.where(out < 0.5, 2.0 * out * op, 1.0 - 2.0 * (1.0 - out) * (1.0 - op))
    out = 0.5 * out + 0.5 * ov                                                         # openness, overlay 50 %
    out = 0.75 * out + 0.25 * out * stretch(prep.svf, v["svf"])                        # SVF, multiply 25 %
    return tone(out, 0.5, p["contrast"], p["gamma"], soft=False)


def shade(prep, params=None, component=None):
    """Apply lights, texture / occlusion mixing and tone to a Prepared surface.
    `component` (a COMPONENTS key) returns one ingredient of a lit render on
    its own; it falls back to the final render when not available."""
    p = merged_params(params)
    mode = prep.mode
    st = prep.stats

    if mode == "gdal":
        dn = gdal_hillshade(prep.surface, prep.dx, prep.dy, p)
        lim = p["gdal_limits"] if p["gdal_display"] != "raw" else None
        if lim:
            lo, hi = float(lim[0]), float(lim[1])
            img = np.where(dn > 0, np.clip((dn - lo) / max(hi - lo, 1e-6), 0.0, 1.0), 0.0)
        else:
            img = dn / np.float32(255.0)
        return _finish(img.astype(np.float32), prep.valid, p, tint=False)

    if mode == "lab":
        return _finish(_lab_image(prep, p), prep.valid, p, tint=False)

    if mode == "clean":
        return _finish(_clean_image(prep, p), prep.valid, p, tint=False)

    if mode == "texture":
        img = texture_to_image(prep.tex, p["texture_contrast"], p["texture_midtone"], p["texture_curve"],
                               st.get("texture_scale"))
        return _finish(tone(img, 0.5, 1.0, p["gamma"], soft=False), prep.valid, p)

    if mode == "svf":
        low = st.get("svf_low", 0.0)
        img = np.clip((prep.svf - low) / max(1e-6, 1.0 - low), 0.0, 1.0)
        return _finish(tone(img, 0.5, p["contrast"], p["gamma"], soft=False), prep.valid, p)

    normals = prep.normals
    if mode == "normals":
        rgb = normal_map_rgb(normals, p["normal_convention"]).astype(np.float32) / 255.0
        return _finish(rgb, prep.valid, p, tint=False)

    if mode == "hillshade":
        img = lambert(normals, light_vector(p["azimuth"], p["altitude"]))
        img = tone(img, math.sin(math.radians(p["altitude"])), p["contrast"], p["gamma"], soft=False)
        return _finish(img, prep.valid, p)

    if mode == "dome":
        img = light_dome(normals, ring_lights(p["dome_lights"], p["dome_altitude"], p["azimuth"]))
        rel = img / np.float32(max(math.sin(math.radians(p["dome_altitude"])), 1e-3))
        dark = np.clip(1.0 - rel, 0.0, None) / np.float32(st.get("slope_scale", 0.05))
        img = 0.9 - 0.75 * np.tanh(dark)          # flat = near white, steepest ~ dark grey
        img = tone(img, 0.6, p["contrast"], p["gamma"], soft=False)
        return _finish(img, prep.valid, p)

    if mode == "vat":
        return _finish(_vat(prep, p), prep.valid, p)

    if mode == "gis":
        img, flat = _gis_hillshade(normals, p)
        return _finish(tone(img, flat, p["contrast"], p["gamma"], soft=False), prep.valid, p)

    if mode == "specular":
        img, flat = _specular_enhancement(normals, p)
        return _finish(tone(img, flat, p["contrast"], p["gamma"], soft=False), prep.valid, p)

    if component and component != "final" and mode in LIT_MODES:
        img = _studio(prep, p, component)
        if img is not None:
            return _finish(np.asarray(img, dtype=np.float32), prep.valid, p, tint=False)
    return _finish(_studio(prep, p), prep.valid, p)


def render(z, dx=1.0, dy=1.0, params=None, px_scale=1.0, stats=None, plane=None,
           origin=(0.0, 0.0), progress=_noop, normals=None):
    """prepare + shade in one call. Returns (image, stats). `normals`: the normal map's window (RGB scaled
    0..1) for clean_source 'normals'."""
    prep = prepare(z, dx, dy, params, px_scale, stats, plane, origin, progress, normals=normals)
    progress(92, "Shading")
    return shade(prep, params), prep.stats


def _finish(img, valid, p, tint=True):
    if tint and img.ndim == 2:
        img = tint_image(img, p["tint"])
    if valid is not None:
        if img.ndim == 3:
            img = img.copy()
            img[~valid] = 0.0
        else:
            img = np.where(valid, img, np.float32(0.0))
    return img


# ----------------------------------------------------------------------------
# Light-stack analysis helpers
# ----------------------------------------------------------------------------

def numpy_pca(stack, n_components=3):
    """PCA across the first axis of an (N, H, W) stack.
    Returns (components (k, H, W), explained-variance ratios)."""
    n_bands, rows, cols = stack.shape
    x = stack.reshape(n_bands, -1).astype(np.float64)
    x -= x.mean(axis=1, keepdims=True)
    cov = (x @ x.T) / max(1, x.shape[1] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    k = min(n_components, n_bands)
    comps = (eigvecs[:, :k].T @ x).astype(np.float32).reshape(k, rows, cols)
    explained = eigvals[:k] / max(eigvals.sum(), 1e-12)
    return comps, explained
