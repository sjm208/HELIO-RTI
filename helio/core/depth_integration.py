"""
Depth-from-normal integration solvers.

All functions return a depth map of shape (H, W), float32.

Functions that take *normals* read them +Y up, the frame calibration and the
shading engines use; functions that take slopes take (dz/dcol, dz/drow).
helio.core.frames converts between the two (KD-009, FIX-003).
"""

import logging
import numpy as np

try:
    # Every core: the same transforms (pocketfft, as NumPy's), about 6 times
    # faster on a 46-megapixel solve; the results agree with NumPy's to about
    # 1e-11 of their size, far below anything a picture or a measurement shows.
    # Its threads are shut down whenever the program starts another process,
    # and a transform caught at that moment fails ("Work item submitted after
    # shutdown"): that one is done again on one core, so a solve never fails.
    from scipy import fft as _sfft

    def _fft2(a):
        try:
            return _sfft.fft2(a, workers=-1)
        except RuntimeError:
            return np.fft.fft2(a)

    def _ifft2(a):
        try:
            return _sfft.ifft2(a, workers=-1)
        except RuntimeError:
            return np.fft.ifft2(a)
except ImportError:                    # pragma: no cover - SciPy is part of Helio
    _fft2, _ifft2 = np.fft.fft2, np.fft.ifft2

from helio.core.frames import depth_slopes, normal_slopes

_log = logging.getLogger(__name__)


def _poisson_solver_fft(nx, ny):
    """The padded-FFT solver ``poisson_solver`` replaced: kept as the reference
    its tests compare with, and as the fallback when SciPy's DCT is missing."""
    div_pad = np.pad(np.gradient(nx, axis=1) + np.gradient(ny, axis=0),
                     ((0, nx.shape[0]), (0, nx.shape[1])), mode='symmetric')
    u, v = np.meshgrid(2 * np.pi * np.fft.fftfreq(div_pad.shape[1]),
                       2 * np.pi * np.fft.fftfreq(div_pad.shape[0]))
    denom = (u**2 + v**2); denom[0,0] = 1.0
    z_hat = _fft2(div_pad) / -denom; z_hat[0,0] = 0.0
    return np.real(_ifft2(z_hat))[:nx.shape[0], :nx.shape[1]].astype(np.float32)


def _dct(fn, a):
    """``scipy.fft.dctn`` / ``idctn`` (type 2) on every core, in place; one core
    if the thread pool was shut down at that moment (see _fft2)."""
    try:
        return fn(a, type=2, overwrite_x=True, workers=-1)
    except RuntimeError:
        return fn(a, type=2, overwrite_x=True, workers=1)


def _divide_by_laplacian(c, wy2, wx2, sign=-1.0):
    """In place: ``c[i, j] /= sign * (wy2[i] + wx2[j])``, ``c[0, 0] = 0``, a strip
    of rows at a time on every core, so the full-size denominator the grid
    version built (and its float64 temporaries) never exists."""
    from helio.core.parallel import by_rows

    def band(y0, y1):
        den = wy2[y0:y1, None] + wx2[None, :]
        if y0 == 0:
            den[0, 0] = 1.0
        c[y0:y1] /= sign * den
    by_rows(band, c.shape[0], c.shape[1], parts=max(1, min(c.shape[0], c.size // 2_000_000)))
    c[0, 0] = 0.0


def poisson_solver(nx, ny):
    """Depth whose gradient is (nx, ny): the Poisson equation with the surface
    mirrored at the frame's edges (zero slope across them).

    That mirrored problem is exactly what a type-2 DCT diagonalises, so it is
    solved here on the image itself: the padded-FFT solver this replaces
    mirrored the image into 4x the pixels and ran a complex FFT over them
    (``_poisson_solver_fft``; the two agree to rounding, ~1e-12 of the depth
    range).  About a tenth of the work, and the peak memory is a float64 copy
    of the image instead of several complex ones four times the size - the
    difference between fitting and swapping on a 16 GB machine.
    """
    try:
        from scipy.fft import dctn, idctn
    except ImportError:                                  # pragma: no cover - SciPy is part of Helio
        return _poisson_solver_fft(nx, ny)
    h, w = nx.shape
    d = np.gradient(nx, axis=1)
    g = np.gradient(ny, axis=0)
    d = d.__iadd__(g) if d.dtype == g.dtype else d + g
    del g
    c = _dct(dctn, d.astype(np.float64))
    del d
    # the DCT's frequencies: pi k / n (the mirrored image's 2 pi k / 2n)
    wy2 = (np.pi * np.arange(h) / h) ** 2
    wx2 = (np.pi * np.arange(w) / w) ** 2
    _divide_by_laplacian(c, wy2, wx2)
    return _dct(idctn, c).astype(np.float32)


def poisson_free_boundary(nx, ny):
    """Depth whose gradient is (nx, ny) by least squares, nothing assumed at
    the frame's edges (the natural boundary condition), in one DCT.

    Minimises sum (z[r, c+1] - z[r, c] - p)^2 + (z[r+1, c] - z[r, c] - q)^2
    over the whole grid, p and q the slopes midway between the two pixels.
    Its normal equations are the 5-point Laplacian with the slopes' flux
    stopped at the edge, which the type-2 DCT diagonalises exactly
    (eigenvalues 2 cos(pi k / n) - 2): Quéau, Durou and Aujol, "Normal
    Integration: a Survey", JMIV 2018, section 3.4 (after Simchony et al.
    1990).  ``poisson_solver`` differs only in dropping the edge flux, which
    is what bends its surface when the slope at the edge is not zero: a
    tilted object comes out level, relief running off the frame bows the
    whole map.  Here a tilted plane comes back as that plane (to rounding)
    and heights in pixels are exact for slopes that are a surface's.
    Mean zero, float32; as fast as ``poisson_solver``."""
    from scipy.fft import dctn, idctn
    h, w = nx.shape
    f = np.zeros((h, w), np.float64)
    if w > 1:
        ph = nx[:, 1:].astype(np.float64)
        ph += nx[:, :-1]
        ph *= 0.5                                        # slope between columns c and c+1
        f[:, :-1] += ph
        f[:, 1:] -= ph
        del ph
    if h > 1:
        qh = ny[1:, :].astype(np.float64)
        qh += ny[:-1, :]
        qh *= 0.5
        f[:-1, :] += qh
        f[1:, :] -= qh
        del qh
    c = _dct(dctn, f)
    del f
    wy2 = 2.0 - 2.0 * np.cos(np.pi * np.arange(h) / h)
    wx2 = 2.0 - 2.0 * np.cos(np.pi * np.arange(w) / w)
    _divide_by_laplacian(c, wy2, wx2)
    return _dct(idctn, c).astype(np.float32)


def poisson_depth_fft(normals, mask=None, scale=1.0):
    gx, gy = normal_slopes(normals)
    if mask is not None:
        gx *= mask; gy *= mask
    return poisson_solver(gx, gy)


def simchony_depth_integration(normals):
    nx_s, ny_s = normal_slopes(normals)
    h, w = nx_s.shape
    try:
        from scipy.fft import dctn, idctn
        D = dctn(np.gradient(nx_s, axis=1) + np.gradient(ny_s, axis=0), type=2, norm="ortho")
        iD = lambda x: idctn(x, type=2, norm="ortho")
    except ImportError:
        from scipy.fftpack import dct, idct
        D = dct(dct(np.gradient(nx_s, axis=1) + np.gradient(ny_s, axis=0), type=2, norm="ortho", axis=0), type=2, norm="ortho", axis=1)
        iD = lambda x: idct(idct(x, type=2, norm="ortho", axis=1), type=2, norm="ortho", axis=0)
    del nx_s, ny_s
    cx = np.cos(np.pi * np.arange(w) / w)
    cy = np.cos(np.pi * np.arange(h) / h)
    # Z = D / (2 (cy + cx - 2)), Z[0, 0] = 0: a strip of rows at a time, into a
    # float64 array, without the full-size denominator and quotient temporaries.
    from helio.core.parallel import by_rows
    Z = np.empty((h, w), np.float64)

    def band(y0, y1):
        den = 2. * (np.add.outer(cy[y0:y1], cx) - 2.)
        if y0 == 0:
            den[0, 0] = 1.
        Z[y0:y1] = D[y0:y1] / den
    by_rows(band, h, w, parts=max(1, min(h, h * w // 2_000_000)))
    del D
    Z[0, 0] = 0.
    return iD(Z).astype(np.float32)


def _frankot_chellappa_grid(nx, ny):
    """The full-grid version ``frankot_chellappa`` replaced (its test reference)."""
    u, v = np.meshgrid(np.fft.fftfreq(nx.shape[1]), np.fft.fftfreq(nx.shape[0]))
    denom = 2 * np.pi * (u**2 + v**2); denom[0, 0] = 1.0
    Z_hat = (-1j * u * _fft2(nx) - 1j * v * _fft2(ny)) / (denom + 1e-10)
    Z_hat[0,0] = 0.0
    return np.real(_ifft2(Z_hat)).astype(np.float32)


def frankot_chellappa(nx, ny):
    """Fourier-basis integration of the slopes (nx, ny) (Frankot & Chellappa 1988).

    With u, v in cycles per pixel (``fftfreq``), d/dx is j 2 pi u, so
    Z = -j (u P + v Q) / (2 pi (u^2 + v^2)): heights in pixels, as the other
    integrators give.  The 2 pi was missing until 2026-11-05, and every height
    came out 2 pi times too tall.

    The input is real, so only half the spectrum is formed (``rfft2``), the
    frequency grids are never built (a row and a column of them, broadcast a
    strip at a time), and the spectrum is worked on in place.  The same values
    as ``_frankot_chellappa_grid`` to rounding, in about a third of the memory."""
    try:
        from scipy import fft as sf
    except ImportError:                                  # pragma: no cover
        return _frankot_chellappa_grid(nx, ny)
    h, w = nx.shape

    def rfft2(a):
        try:
            return sf.rfft2(a, workers=-1)
        except RuntimeError:
            return sf.rfft2(a, workers=1)
    fx, fy = rfft2(nx), rfft2(ny)
    u = np.fft.rfftfreq(w)                               # columns: 0 .. 0.5
    v = np.fft.fftfreq(h)                                # rows
    # At the Nyquist column / row (even sizes) the derivative has no sign: the
    # full-spectrum version's np.real() averaged +0.5 and -0.5 to zero there.
    un, vn = u.copy(), v.copy()
    if w % 2 == 0:
        un[-1] = 0.0
    if h % 2 == 0:
        vn[h // 2] = 0.0
    from helio.core.parallel import by_rows
    Z = np.empty(fx.shape, np.complex128)

    def band(y0, y1):
        vv = v[y0:y1, None]
        denom = 2 * np.pi * (u[None, :] ** 2 + vv ** 2)
        if y0 == 0:
            denom[0, 0] = 1.0
        Z[y0:y1] = (-1j * un[None, :] * fx[y0:y1] - 1j * vn[y0:y1, None] * fy[y0:y1]) / (denom + 1e-10)
    by_rows(band, h, fx.shape[1], parts=max(1, min(h, fx.size // 1_000_000)))
    del fx, fy
    Z[0, 0] = 0.0
    try:
        z = sf.irfft2(Z, s=(h, w), workers=-1)
    except RuntimeError:
        z = sf.irfft2(Z, s=(h, w), workers=1)
    return z.astype(np.float32)


def frankot_chellappa_depth(normals, mask=None):
    gx, gy = normal_slopes(normals)
    return frankot_chellappa(gx, gy)


def iterative_normal_integration(normals_in, n_iters=4, ps_weight=0.85):
    normals = normals_in.copy().astype(np.float32)
    for _ in range(n_iters):
        p, q = normal_slopes(normals)
        depth = frankot_chellappa(p, q)
        gy, gx = np.gradient(depth)
        grad_n = np.stack([-gx, gy, np.ones_like(gx)], axis=-1).astype(np.float32)   # +Y up
        mag = np.linalg.norm(grad_n, axis=-1, keepdims=True)
        grad_n /= np.maximum(mag, 1e-9)
        blended = normals_in * ps_weight + grad_n * (1.0 - ps_weight)
        mag_b = np.linalg.norm(blended, axis=-1, keepdims=True)
        normals = blended / np.maximum(mag_b, 1e-9)
    return normals


def directional_depth_integration(normals):
    H, W = normals.shape[:2]
    nx = normals[..., 0].astype(np.float32)
    ny = normals[..., 1].astype(np.float32)
    nz = np.maximum(normals[..., 2].astype(np.float32), 0.001)
    # Slopes, as every other integrator takes them.  This used to pass the raw
    # nx, ny - the negated gradient - so the surface came back inside out:
    # bumps as holes.  Inverting relief is now a separate, explicit choice
    # (helio.core.invert_relief, "Bumps <-> holes" in the Compute Engine panel).
    depth_base = poisson_solver(*depth_slopes(nx, ny, nz))
    _sigma_lf = max(3.0, min(H, W) * 0.02)
    import cv2
    depth_lf = cv2.GaussianBlur(depth_base, (0, 0), sigmaX=_sigma_lf)
    depth_hf = depth_base - depth_lf
    try:
        import numpy.fft as _fft
        _F = _fft.fft2(depth_hf.astype(np.float64))
        _F = _fft.fftshift(_F)
        cy, cx = H // 2, W // 2
        fy = (np.arange(H) - cy).reshape(-1, 1).astype(np.float32)
        fx = (np.arange(W) - cx).reshape(1, -1).astype(np.float32)
        _ang = np.arctan2(np.abs(fy), np.abs(fx))
        _boost = (np.cos(2 * _ang)**4 + 0.2).astype(np.float32)
        _F_dir = _F * _boost
        _F_dir = _fft.ifftshift(_F_dir)
        depth_hf_dir = np.real(_fft.ifft2(_F_dir)).astype(np.float32)
    except Exception:
        _log.debug("Directional FFT depth enhancement failed, using non-directional")
        depth_hf_dir = depth_hf
    strength = 0.3
    depth_final = depth_lf + depth_hf * (1.0 - strength) + depth_hf_dir * strength
    lo, hi = float(depth_final.min()), float(depth_final.max())
    if hi - lo > 1e-9:
        depth_final = (depth_final - lo) / (hi - lo)
    return depth_final.astype(np.float32)


def confidence_weighted_depth_integration(normals, confidence, method="poisson"):
    gx, gy = normal_slopes(normals)
    gx = gx * confidence
    gy = gy * confidence
    if method == "dct":
        try:
            from scipy.fft import dctn, idctn
            div = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
            H, W = div.shape
            div_pad = np.pad(div, ((0, H), (0, W)), mode='reflect')
            u = np.cos(np.pi * np.arange(div_pad.shape[0]) / div_pad.shape[0])
            v = np.cos(np.pi * np.arange(div_pad.shape[1]) / div_pad.shape[1])
            denom = (u[:, None] + v[None, :] - 2) * 2
            denom[0, 0] = 1.0
            Z = idctn(dctn(div_pad) / denom)[:H, :W]
            return Z.astype(np.float32)
        except Exception:
            _log.debug("DCT confidence-weighted integration failed, falling back to Poisson")
    div_pad = np.pad(
        np.gradient(gx, axis=1) + np.gradient(gy, axis=0),
        ((0, normals.shape[0]), (0, normals.shape[1])), mode='symmetric')
    u, v = np.meshgrid(
        2 * np.pi * np.fft.fftfreq(div_pad.shape[1]),
        2 * np.pi * np.fft.fftfreq(div_pad.shape[0]))
    denom = u**2 + v**2; denom[0, 0] = 1.0
    z_hat = _fft2(div_pad) / -denom; z_hat[0, 0] = 0.0
    return np.real(_ifft2(z_hat))[:normals.shape[0], :normals.shape[1]].astype(np.float32)


def multiscale_depth_integration(normals):
    """Broad shape solved at quarter size plus fine detail solved at full size.

    The two parts are complementary: the broad shape is integrated from the
    smoothed normals' slopes, the detail from what the smoothing took out of
    the slopes (slopes of the normals less slopes of the smoothed normals),
    so together they are the whole slope field once.  Until 2026-11-10 the
    broad part was solved from the unsmoothed normals, so everything between
    the blur's scale and the quarter-size grid's was counted in both parts:
    a 35 px-wide bump came out 10 % too tall."""
    import cv2
    H, W = normals.shape[:2]
    scale = 0.25
    nH, nW = max(4, int(H * scale)), max(4, int(W * scale))
    n_smooth = np.empty_like(normals)
    for ch in range(3):
        n_smooth[..., ch] = cv2.GaussianBlur(normals[..., ch], (0, 0),
                                              sigmaX=max(3.0, min(H, W) * 0.04))
    mag_s = np.linalg.norm(n_smooth, axis=-1, keepdims=True)
    n_smooth = np.divide(n_smooth, mag_s, out=np.zeros_like(n_smooth), where=mag_s > 1e-9)
    gx_s, gy_s = normal_slopes(n_smooth)
    # Slopes are the same number at any size; heights solved at 1/4 size are in
    # 1/4-size pixels: back to this image's pixels before the detail is added.
    z_low_small = poisson_solver(cv2.resize(gx_s, (nW, nH), interpolation=cv2.INTER_AREA),
                                 cv2.resize(gy_s, (nW, nH), interpolation=cv2.INTER_AREA))
    z_low_small *= np.float32(0.5 * (W / nW + H / nH))
    z_low = cv2.resize(z_low_small, (W, H), interpolation=cv2.INTER_CUBIC)
    gx, gy = normal_slopes(normals)
    z_high = poisson_solver(gx - gx_s, gy - gy_s)
    combined = z_low + z_high
    return combined.astype(np.float32)


def compute_hybrid_depth_blend(depth_global, depth_flat, detail_mix=0.12):
    import cv2
    mix = float(np.clip(detail_mix, 0.0, 1.0))
    g = depth_global.astype(np.float32)
    f = depth_flat.astype(np.float32)
    if g.shape != f.shape:
        f = cv2.resize(f, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    def _norm01(arr):
        lo, hi = arr.min(), arr.max()
        return ((arr - lo) / (hi - lo)).astype(np.float32) if hi > lo + 1e-9 else np.zeros_like(arr)
    g_n = _norm01(g)
    f_n = _norm01(f)
    blended = g_n * (1.0 - mix) + f_n * mix
    return _norm01(blended)


def gradient_smoothed_poisson(normals, albedo_guide=None, radius_frac=0.03, epsilon=0.05):
    import cv2
    H, W = normals.shape[:2]
    nz_safe = np.maximum(normals[..., 2], 0.001)
    gx, gy = (g.astype(np.float32) for g in
              depth_slopes(normals[..., 0], normals[..., 1], nz_safe))
    radius = max(3, int(min(H, W) * radius_frac))
    try:
        if albedo_guide is not None:
            if albedo_guide.ndim == 3:
                guide = cv2.cvtColor(albedo_guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            else:
                guide = albedo_guide.astype(np.float32) / 255.0
            if guide.shape != (H, W):
                guide = cv2.resize(guide, (W, H), interpolation=cv2.INTER_LINEAR)
            try:
                gx_s = cv2.ximgproc.guidedFilter(guide, gx, radius, epsilon ** 2)
                gy_s = cv2.ximgproc.guidedFilter(guide, gy, radius, epsilon ** 2)
            except AttributeError:
                gx_s = cv2.bilateralFilter(gx, d=max(5, radius*2+1),
                                           sigmaColor=epsilon * 2, sigmaSpace=float(radius))
                gy_s = cv2.bilateralFilter(gy, d=max(5, radius*2+1),
                                           sigmaColor=epsilon * 2, sigmaSpace=float(radius))
        else:
            sigma = max(1.0, radius * 0.4)
            gx_s = cv2.GaussianBlur(gx, (0, 0), sigmaX=sigma)
            gy_s = cv2.GaussianBlur(gy, (0, 0), sigmaX=sigma)
    except Exception as _smooth_err:
        _log.warning("Gradient smoothing failed, integrating unsmoothed gradients: %s",
                     _smooth_err, exc_info=True)
        gx_s, gy_s = gx, gy
    try:
        div_pad = np.pad(
            np.gradient(gx_s, axis=1) + np.gradient(gy_s, axis=0),
            ((0, H), (0, W)), mode='symmetric')
        u, v = np.meshgrid(
            2 * np.pi * np.fft.fftfreq(div_pad.shape[1]),
            2 * np.pi * np.fft.fftfreq(div_pad.shape[0]))
        denom = u ** 2 + v ** 2
        denom[0, 0] = 1.0
        z_hat = _fft2(div_pad) / -denom
        z_hat[0, 0] = 0.0
        depth = np.real(_ifft2(z_hat))[:H, :W].astype(np.float32)
        return depth
    except Exception:
        _log.debug("FFT directional depth solver failed, falling back to Poisson")
        return poisson_solver(gx, gy)


def directional_engraving_enhance(height_map, tool_angle_deg=45.0, strength=1.5,
                                   blend_mode="additive", soften_px=0.8,
                                   enhance_mode="single",
                                   tool_angle_b_deg=90.0, blend_mix=0.5):
    import cv2
    arr = height_map.astype(np.float32)
    if soften_px > 0.3:
        arr = cv2.GaussianBlur(arr, (0, 0), float(soften_px))
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
    def _single_enhance(base, grad_component):
        if blend_mode == "multiplicative":
            return base * (1.0 + float(strength) * np.abs(grad_component))
        else:
            return base + float(strength) * grad_component
    def _perp_grad(angle_deg):
        perp_rad = np.radians(float(angle_deg) + 90.0)
        return gx * float(np.cos(perp_rad)) + gy * float(np.sin(perp_rad))
    if enhance_mode == "omni":
        grad_mag = np.sqrt(gx * gx + gy * gy)
        if blend_mode == "multiplicative":
            enhanced = arr * (1.0 + float(strength) * grad_mag)
        else:
            enhanced = arr + float(strength) * grad_mag
    elif enhance_mode == "dual":
        mix = float(np.clip(blend_mix, 0.0, 1.0))
        enh_a = _single_enhance(arr, _perp_grad(tool_angle_deg))
        enh_b = _single_enhance(arr, _perp_grad(tool_angle_b_deg))
        enhanced = enh_a * (1.0 - mix) + enh_b * mix
    else:
        enhanced = _single_enhance(arr, _perp_grad(tool_angle_deg))
    return enhanced.astype(np.float32)


def remove_poly22(depth_map, *, guard_single_row: bool = False):
    """``depth_map`` minus its best-fit ``a x^2 + b y^2 + c xy + d x + e y + f``
    (coordinates scaled to [-1, 1]), in the input's dtype.

    The least-squares fit is the same as the textbook one that builds an
    (h*w, 6) float64 matrix (4.8 GB at 100 MP, plus the products made from it);
    here the 6x6 normal equations are accumulated a strip of rows at a time in
    float64 and the surface is subtracted a strip at a time, so the peak is a
    few MB beyond the map itself.  ``guard_single_row`` makes a 1-pixel-wide
    or -high map raise (the pipeline's version returns the map unchanged then);
    otherwise a single row or column is treated as coordinate 0.
    """
    h, w = depth_map.shape
    if guard_single_row and (h < 2 or w < 2):
        raise ValueError("needs at least 2 rows and 2 columns")
    xn = (np.arange(w) / max(1, w - 1)) * 2 - 1
    yn = (np.arange(h) / max(1, h - 1)) * 2 - 1
    step = max(1, 250_000 // max(1, w))

    def design(a, b):
        y = yn[a:b, None]
        x = xn[None, :]
        return np.stack([np.broadcast_to(x * x, (b - a, w)), np.broadcast_to(y * y, (b - a, w)),
                         x * y, np.broadcast_to(x, (b - a, w)), np.broadcast_to(y, (b - a, w)),
                         np.ones((b - a, w))], axis=-1).reshape(-1, 6)
    ata = np.zeros((6, 6))
    atz = np.zeros(6)
    for a in range(0, h, step):
        b = min(h, a + step)
        A = design(a, b)
        z = depth_map[a:b].astype(np.float64).ravel()
        ata += A.T @ A
        atz += A.T @ z
    coeffs = np.linalg.lstsq(ata, atz, rcond=None)[0]
    out = np.empty_like(depth_map, dtype=np.result_type(depth_map.dtype, np.float32))
    for a in range(0, h, step):
        b = min(h, a + step)
        bowl = (design(a, b) @ coeffs).reshape(b - a, w)
        out[a:b] = depth_map[a:b] - bowl
    return out
