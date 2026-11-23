"""
Register a colour / albedo image to a height map.

Photometric-stereo packages export the albedo and the depth map from the same camera, but not always on the same
pixel grid - integration may pad, resample or slightly rotate the depth map (the user's HELIO export: 9101 x 6421
albedo inside a 9375 x 6627 depth map, turned by ~0.3 deg) - and usually without georeferencing, so QGIS draws them
apart. This finds where the colour image sits on the height map's pixels:

1. coarse - phase correlation of edge maps on overviews over a range of scales (scale + offset);
2. fine - phase correlation of the fine detail both show (tooling, cracks, grain: high-passed heights against
   high-passed colour, correlated or anti-correlated alike) in a grid of full-resolution patches, and a
   least-squares affine fit (scale, rotation, shear, offset) with outliers rejected, iterated.

It then writes a VRT that resamples the colour image onto the height map's exact grid, georeferenced like it, so
QGIS draws it right over the height map and the plugin reads it pixel for pixel. Nothing is copied; 16-bit values
are kept. An image that shares no structure with the height map is refused.
"""

import numpy as np
from osgeo import gdal
from scipy import ndimage as ndi

PATCH = 512             # full-resolution px of each matching patch
GRID = (6, 5)           # patches across, down
MIN_STRENGTH = 8.0      # phase-correlation peak / surface sd below which a patch match is not trusted
MIN_MATCHES = 6         # agreeing patches needed to trust the fit
MAX_RMS = 3.0           # px: residual above which the fit is not trusted


def _noop(*_args, **_kwargs):
    pass


def _read_lum(ds, win, out, resample=gdal.GRIORA_Average):
    """Mean of the first three bands (or the only band) over win = (x, y, w, h), resampled to out = (w, h)."""
    n = min(3, ds.RasterCount)
    acc = None
    for i in range(n):
        a = ds.GetRasterBand(i + 1).ReadAsArray(int(win[0]), int(win[1]), int(win[2]), int(win[3]),
                                                buf_xsize=int(out[0]), buf_ysize=int(out[1]),
                                                resample_alg=resample).astype(np.float64)
        acc = a if acc is None else acc + a
    acc /= n
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        acc[acc == nodata] = np.nan
    return acc


def _fill(x):
    x = np.asarray(x, dtype=np.float64)
    bad = ~np.isfinite(x)
    return np.where(bad, np.nanmedian(x) if (~bad).any() else 0.0, x) if bad.any() else x


def _edges(x, hp):
    """Edge strength of the structure, whatever the image measures: high-passed, gradient magnitude,
    log-compressed so the object's outline does not swamp the tooling, standardised."""
    x = _fill(x)
    x = x - ndi.gaussian_filter(x, hp)
    g = np.hypot(*np.gradient(ndi.gaussian_filter(x, 1.0)))
    g = np.log1p(g / (np.median(g) + 1e-12))
    return (g - g.mean()) / (g.std() + 1e-12)


def _detail(x, hp):
    """Fine detail with its sign: high-passed, lightly smoothed, standardised."""
    x = _fill(x)
    x = ndi.gaussian_filter(x - ndi.gaussian_filter(x, hp), 1.0)
    return (x - x.mean()) / (x.std() + 1e-12)


def _phase(a, b):
    """(dx, dy, strength) with a(u) ~ +/- b(u - d): what sits at pixel j of b sits at j + d in a. The strongest
    peak of either sign (colour may be darker or lighter where the relief is raised), sub-pixel (parabolic);
    strength = |peak| / sd of the correlation surface."""
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    cross = np.fft.rfft2(a * win) * np.conj(np.fft.rfft2(b * win))
    r = np.fft.irfft2(cross / (np.abs(cross) + 1e-12), s=(h, w))
    iy, ix = np.unravel_index(int(np.argmax(np.abs(r))), r.shape)
    r = r * np.sign(r[iy, ix])

    def vertex(c0, cm, cp):
        d = cm - 2.0 * c0 + cp
        return 0.0 if d >= 0 else 0.5 * (cm - cp) / d

    fy = vertex(r[iy, ix], r[iy - 1, ix], r[(iy + 1) % h, ix])
    fx = vertex(r[iy, ix], r[iy, ix - 1], r[iy, (ix + 1) % w])
    dy = (iy if iy <= h // 2 else iy - h) + fy
    dx = (ix if ix <= w // 2 else ix - w) + fx
    return dx, dy, float(r[iy, ix] / (r.std() + 1e-12))


def _coarse(rds, ids, progress, max_dim=1024):
    """Scale and offset from overviews - phase correlation of edge maps at a range of scales, a coarse sweep then
    a finer one round the best. Returns the 2 x 3 affine (colour px -> height-map px) and the peak strength."""
    RW, RH, IW, IH = rds.RasterXSize, rds.RasterYSize, ids.RasterXSize, ids.RasterYSize
    k = max(1.0, max(RW, RH, IW, IH) / float(max_dim))
    rw, rh = max(16, int(round(RW / k))), max(16, int(round(RH / k)))
    iw, ih = max(16, int(round(IW / k))), max(16, int(round(IH / k)))
    progress(3, "Reading overviews")
    fr = _edges(_read_lum(rds, (0, 0, RW, RH), (rw, rh)), 16)
    fi = _edges(_read_lum(ids, (0, 0, IW, IH), (iw, ih)), 16)
    kr, ki = RW / float(rw), IW / float(iw)

    def trial(s):
        z = fi if s == 1.0 else ndi.zoom(fi, s * ki / kr, order=1)
        canvas = np.zeros((rh, rw))
        hh, ww = min(rh, z.shape[0]), min(rw, z.shape[1])
        canvas[:hh, :ww] = z[:hh, :ww]
        dx, dy, strength = _phase(fr, canvas)
        return strength, s, dx, dy

    if (IW, IH) == (RW, RH):
        sweep = [1.0]
    else:
        sweep = list(np.arange(0.80, 1.2501, 0.02)) + [1.0]
    best = max(trial(s) for s in sweep)
    progress(20, "Finding the colour image's scale and offset")
    if len(sweep) > 1:
        best = max([best] + [trial(s) for s in np.arange(best[1] - 0.02, best[1] + 0.0201, 0.004)])
    strength, s, dx, dy = best
    progress(30, "Finding the colour image's scale and offset")
    return np.array([[s, 0.0, dx * kr], [0.0, s, dy * kr]]), strength


def _fit_affine(obs):
    """Least-squares affine from (x, y, u, v) rows, outliers (> 3 x median residual, at least 1.5 px) dropped."""
    o = np.asarray(obs, dtype=np.float64)
    A, res = None, None
    for _ in range(4):
        M = np.c_[o[:, 0], o[:, 1], np.ones(len(o))]
        cu, *_ = np.linalg.lstsq(M, o[:, 2], rcond=None)
        cv, *_ = np.linalg.lstsq(M, o[:, 3], rcond=None)
        A = np.vstack([cu, cv])
        res = np.hypot(o[:, 2] - M @ cu, o[:, 3] - M @ cv)
        keep = res <= max(3.0 * np.median(res), 1.5)
        if keep.all() or keep.sum() < 4:
            break
        o = o[keep]
    return A, len(o), float(np.sqrt(np.mean(res ** 2)))


def _fine(rds, ids, A, progress):
    """Refine the affine from sub-pixel matches of GRID full-resolution patches (iterated three times)."""
    RW, RH, IW, IH = rds.RasterXSize, rds.RasterYSize, ids.RasterXSize, ids.RasterYSize
    matched, total, rms = 0, 0, float("inf")
    for it in range(3):
        corners = A @ np.array([[0, IW - 1, 0, IW - 1], [0, 0, IH - 1, IH - 1], [1, 1, 1, 1]], dtype=np.float64)
        u0, u1 = max(0.0, corners[0].min()), min(RW - 1.0, corners[0].max())
        v0, v1 = max(0.0, corners[1].min()), min(RH - 1.0, corners[1].max())
        P = int(min(PATCH, 0.9 * (u1 - u0) / GRID[0] * 2, 0.9 * (v1 - v0) / GRID[1] * 2))
        if P < 48:
            break
        inv = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
        sx, sy = float(np.hypot(A[0, 0], A[1, 0])), float(np.hypot(A[0, 1], A[1, 1]))
        obs, total = [], 0
        for gy in range(GRID[1]):
            for gx in range(GRID[0]):
                total += 1
                uc = u0 + (u1 - u0) * (gx + 0.5) / GRID[0]
                vc = v0 + (v1 - v0) * (gy + 0.5) / GRID[1]
                ru, rv = int(round(uc - (P - 1) / 2.0)), int(round(vc - (P - 1) / 2.0))
                wi, hi = max(8, int(round(P / sx))), max(8, int(round(P / sy)))
                xc, yc = inv @ np.array([ru + (P - 1) / 2.0, rv + (P - 1) / 2.0, 1.0])
                xi, yi = int(round(xc - (wi - 1) / 2.0)), int(round(yc - (hi - 1) / 2.0))
                if (ru < 0 or rv < 0 or ru + P > RW or rv + P > RH or xi < 0 or yi < 0 or xi + wi > IW
                        or yi + hi > IH):
                    continue
                a = _detail(_read_lum(rds, (ru, rv, P, P), (P, P), gdal.GRIORA_Bilinear), 12)
                b = _detail(_read_lum(ids, (xi, yi, wi, hi), (P, P), gdal.GRIORA_Bilinear), 12)
                dx, dy, strength = _phase(a, b)
                if strength < MIN_STRENGTH or abs(dx) > P / 3 or abs(dy) > P / 3:
                    continue
                j = (P - 1) / 2.0                                        # patch centre, buffer pixels
                xb, yb = xi + (j + 0.5) * wi / P - 0.5, yi + (j + 0.5) * hi / P - 0.5   # ... in colour pixels
                obs.append((xb, yb, ru + j + dx, rv + j + dy))            # ... and where it lies on the height map
            progress(30 + int(20 * (it + (gy + 1) / GRID[1])), "Matching full-resolution patches")
        if len(obs) < 4:
            matched, rms = len(obs), float("inf")
            break
        A, matched, rms = _fit_affine(obs)
    return A, matched, total, rms


def estimate(ref_path, img_path, progress=_noop):
    """Where the colour image sits on the height map: the affine (colour px -> height-map px, pixel centres)
    u = a x + b y + c, v = d x + e y + f, with its scale, rotation and offset, how many patches matched and the
    rms residual (px). Raises RuntimeError when the image cannot be placed."""
    rds, ids = gdal.Open(ref_path, gdal.GA_ReadOnly), gdal.Open(img_path, gdal.GA_ReadOnly)
    if rds is None or ids is None:
        raise RuntimeError("Cannot open the height map or the colour image")
    A, strength = _coarse(rds, ids, progress)
    A, matched, total, rms = _fine(rds, ids, A, progress)
    if matched < MIN_MATCHES or not rms <= MAX_RMS:
        raise RuntimeError(f"The colour image could not be placed on the height map ({matched} of {total} places "
                           f"matched) - is it the same object, seen by the same camera?")
    ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    same = (ids.RasterXSize, ids.RasterYSize) == (rds.RasterXSize, rds.RasterYSize)
    corners = np.array([[0, ids.RasterXSize, 0, ids.RasterXSize], [0, 0, ids.RasterYSize, ids.RasterYSize],
                        [1, 1, 1, 1]], dtype=np.float64)
    if same and np.abs((A - ident) @ corners).max() < 0.3:
        A = ident                                                        # the same pixels: nothing to move
    a, b, c, d, e, f = (float(v) for v in A.ravel())
    return dict(a=a, b=b, c=c, d=d, e=e, f=f, sx=float(np.hypot(a, d)), sy=float(np.hypot(b, e)),
                rotation=float(np.degrees(np.arctan2(d, a))), tx=c, ty=f, matched=matched, patches=total, rms=rms,
                strength=strength, identity=bool((A == ident).all()))


def reference_geotransform(ds):
    """The height map's geotransform - or where QGIS draws a raster without one (top-left at 0, 0, y up)."""
    gt = ds.GetGeoTransform(can_return_null=True)
    return tuple(gt) if gt else (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)


def white_level(ds):
    """Level treated as white for a colour image: 255 for 8-bit; otherwise the 99.8th percentile of a decimated
    read of its bands (16-bit albedos rarely use the full range)."""
    b = ds.GetRasterBand(1)
    if gdal.GetDataTypeName(b.DataType) == "Byte":
        return 255.0
    k = max(1, int(max(ds.RasterXSize, ds.RasterYSize) / 1024))
    w, h = max(1, ds.RasterXSize // k), max(1, ds.RasterYSize // k)
    v = np.concatenate([ds.GetRasterBand(i + 1).ReadAsArray(buf_xsize=w, buf_ysize=h).astype(np.float64).ravel()
                        for i in range(min(3, ds.RasterCount))])
    v = v[np.isfinite(v) & (v > 0)]
    return float(max(np.percentile(v, 99.8), 1e-9)) if v.size else 1.0


def write_vrt(ref_path, img_path, fit, out_path):
    """Write out_path: the colour image resampled (cubic) onto the height map's exact grid, georeferenced like
    it, nodata 0 outside the colour image, and its white level in the metadata (HPSS_WHITE). A second small VRT
    (<out>_georef.vrt) holds the colour image georeferenced at its own resolution - the first one's source."""
    rds = gdal.Open(ref_path, gdal.GA_ReadOnly)
    ids = gdal.Open(img_path, gdal.GA_ReadOnly)
    gt, srs = reference_geotransform(rds), rds.GetProjection()
    W, H = rds.RasterXSize, rds.RasterYSize
    a, b, c, d, e, f = (fit[k] for k in "abcdef")
    # pixel centres -> pixel corners (geotransforms address corners): u' = a x' + b y' + c + (1 - a - b) / 2
    c, f = c + 0.5 * (1.0 - a - b), f + 0.5 * (1.0 - d - e)
    igt = (gt[0] + gt[1] * c + gt[2] * f, gt[1] * a + gt[2] * d, gt[1] * b + gt[2] * e,
           gt[3] + gt[4] * c + gt[5] * f, gt[4] * a + gt[5] * d, gt[4] * b + gt[5] * e)
    geo_path = out_path[:-4] + "_georef.vrt"
    geo = gdal.Translate(geo_path, ids, format="VRT")
    if geo is None:
        raise RuntimeError(f"Could not write {geo_path}")
    geo.SetGeoTransform(igt)
    if srs:
        geo.SetProjection(srs)
    geo = None
    xs = sorted((gt[0], gt[0] + W * gt[1]))
    ys = sorted((gt[3], gt[3] + H * gt[5]))
    wk = dict(format="VRT", outputBounds=[xs[0], ys[0], xs[1], ys[1]], width=W, height=H, resampleAlg="cubic",
              dstNodata=0, multithread=True)
    if srs:
        wk.update(srcSRS=srs, dstSRS=srs)
    out = gdal.Warp(out_path, geo_path, **wk)
    if out is None:
        raise RuntimeError(f"Could not write {out_path}")
    out.SetMetadataItem("HPSS_WHITE", repr(white_level(ids)))
    out = None
    return out_path


def align_to(ref_path, img_path, out_path, progress=_noop):
    """estimate + write_vrt. Returns the estimate plus path = the aligned VRT."""
    fit = estimate(ref_path, img_path, progress)
    progress(90, "Writing the aligned colour VRT")
    write_vrt(ref_path, img_path, fit, out_path)
    progress(100, "Aligned")
    return dict(fit, path=out_path)
