"""Solves for what least squares gets wrong: unequal lamps, gold and metal, GUI-free.

Three tools, each with a Rust twin in helio_core_rs (``robust.rs``) that
``helio.core._native`` picks when it is built; the NumPy code here stays the
reference the Rust is tested against (tests/test_robust_ps.py).

**Semi-calibrated photometric stereo** (``light_strengths``).  The light
directions are known (ball, light file, dome) but not how strong each lamp
was.  LED domes are rarely even (10-30% between lamps is common) and the
solve reads a brighter lamp as the surface turning towards it.  Holding the
directions fixed, the surface and the strengths are found together by
alternating least squares (Cho, Matsushita, Tai and Kweon, "Semi-calibrated
photometric stereo", TPAMI 2018): each pixel's scaled normal from the
strengths, then each lamp's strength from every pixel, until they agree.  No
grey card is needed; the flash and flat-area corrections, which measure the
same thing on a marked reference, take precedence when they are on.

**Sparse-outlier least squares** (``sparse_outlier_normals``): the per-pixel
form of robust PCA (Wu, Ganesh, Shi, Matsushita, Wang and Ma, ACCV 2010;
Ikehata, Wipf, Matsushita and Aizawa, CVPR 2012).  Each pixel's readings are
the Lambertian model plus a sparse error: most readings are right, a few
(the highlight on gold, a cast shadow) are wildly wrong.  Minimises
``1/2 |I - L b - e|^2 + lam |e|_1`` by alternating the least-squares normal
and a soft-thresholded error, so a highlight is set aside however bright it
is, where weighted least squares only turns it down.

**Where the surface shines** (``specular_excess``, ``flare``, ``metal_mask``):
two cues.  The excess is how far a pixel's brightest reading rises above
what a matte surface with its normal could give: it needs lamps to spare,
because with four the fit takes a glint in.  The flare needs no normal: the
brightest reading over the median, against the matte about it, on pixels
that are not dark.  On a four-lamp velvet binding it finds the embroidered
gold, the gilt frame and the corner pieces, and leaves the velvet, the grey
card and a parchment sheet's lamp-side edge alone.  The mask picks the
pixels the metal-aware solve re-solves with a robust solver
(pipeline._metal_aware).
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from helio.core._native import rs

_log = logging.getLogger(__name__)

# Pixels used to estimate the lamp strengths: an even sample, plenty for a
# handful of unknowns and the same for the Rust and the NumPy code.
STRENGTH_SAMPLE = 400_000
_CHUNK = 65_536          # pixels per block in the NumPy solvers
SHADOW_FRACTION = 0.02   # a reading below this share of the pixel's brightest is shadow
CONTINUATION_START, CONTINUATION_RATE = 2.0, 0.75


def _sample_columns(m: int, k: int) -> np.ndarray:
    """*k* evenly spaced column indices of *m* (all of them when m <= k)."""
    if m <= k:
        return np.arange(m)
    return np.linspace(0, m - 1, k).astype(np.int64)


def strength_window(I: np.ndarray) -> tuple[float, float]:
    """The readings trusted for the strengths: above the shadows, below clipping."""
    top = float(np.percentile(I, 99.9)) if I.size else 1.0
    return 0.02 * top, 0.97 * float(I.max(initial=0.0))


def _semi_cal_step_numpy(L, e, I, lo, hi):
    """One alternation: each pixel's scaled normal from strengths *e*, then the
    (numerator, denominator) of every lamp's least-squares strength."""
    L = np.asarray(L, np.float64)
    W = ((I > lo) & (I < hi)).astype(np.float64)                 # (n, m)
    Le = L * e[:, None]
    A = np.einsum("jm,ja,jb->mab", W, Le, Le)                     # (m, 3, 3)
    r = np.einsum("jm,ja,jm->ma", W, Le, I.astype(np.float64))    # (m, 3)
    ok = np.linalg.det(A) > 1e-12
    b = np.zeros_like(r)
    b[ok] = np.linalg.solve(A[ok], r[ok][..., None])[..., 0]
    s = L @ b.T                                                   # (n, m): shading without e
    v = W * (s > 0) * ok[None, :]
    num = (v * I * s).sum(axis=1)
    den = (v * s * s).sum(axis=1)
    return num, den


def light_strengths(I: np.ndarray, L: np.ndarray, iters: int = 30, tol: float = 1e-4,
                    sample: int = STRENGTH_SAMPLE) -> dict | None:
    """Each lamp's relative strength from the stack *I* (n, h·w) in linear
    light and the light directions *L* (n, 3).

    Returns ``dict(strength, spread, used, iters)`` with the strengths scaled
    to a geometric mean of 1, or None when too few pixels can be read (fewer
    than four lamps, or a dark or clipped stack)."""
    I = np.asarray(I)
    n = I.shape[0]
    L = np.asarray(L, np.float64).reshape(-1, 3)
    if n < 4 or len(L) != n:
        return None
    flat = I.reshape(n, -1)
    cols = _sample_columns(flat.shape[1], sample)
    S = np.ascontiguousarray(flat[:, cols], np.float32)
    lo, hi = strength_window(S)
    good = ((S > lo) & (S < hi)).sum(axis=0) >= 4               # pixels with four readings
    S = np.ascontiguousarray(S[:, good])
    if S.shape[1] < 64:
        return None
    e = np.ones(n)
    it = 0
    for it in range(1, iters + 1):
        if rs is not None and hasattr(rs, "semi_cal_step"):
            num, den = rs.semi_cal_step(np.ascontiguousarray(L), np.ascontiguousarray(e), S,
                                        float(lo), float(hi))
            num, den = np.asarray(num), np.asarray(den)
        else:
            num, den = _semi_cal_step_numpy(L, e, S, lo, hi)
        if not np.all(den > 0):
            return None
        new = num / den
        if not np.all(np.isfinite(new)) or np.any(new <= 0):
            return None
        new /= np.exp(np.mean(np.log(new)))
        moved = float(np.max(np.abs(new / e - 1.0)))
        e = new
        if moved < tol:
            break
    return dict(strength=[float(v) for v in e], spread=float(e.max() / e.min()),
                used=int(S.shape[1]), iters=it)


def continuation(lam: float, k: int) -> float:
    """The threshold at iteration *k*: from twice the pixel's mean down to *lam*
    by a quarter each step, so the grossest outliers are set aside first and
    a burnt-out highlight cannot drag the readings beside it out with it."""
    return max(lam, CONTINUATION_START * CONTINUATION_RATE ** k)


def _sparse_block_numpy(L, P, I, lam, iters, tol):
    """Normals (m, 3) of one block of pixels *I* (n, m) by sparse-outlier LS."""
    I = I.astype(np.float32)
    scale = np.maximum(I.mean(axis=0), 1e-6)                      # (m,)
    shadow = I <= SHADOW_FRACTION * I.max(axis=0, initial=0.0)[None, :]
    e = np.zeros_like(I)
    for k in range(iters):
        frac = continuation(lam, k)
        thr = (frac * scale)[None, :]
        b = P @ (I - e)
        res = I - L @ b
        new = np.sign(res) * np.maximum(np.abs(res) - thr, 0.0)
        new[shadow] = res[shadow]                 # a shadow is missing, not a reading
        moved = float(np.max(np.abs(new - e) / scale[None, :])) if new.size else 0.0
        e = new
        if frac <= lam and moved < tol:
            break
    b = _refit_inliers(L, I, e, P @ (I - e))
    mag = np.linalg.norm(b, axis=0)
    out = np.zeros((I.shape[1], 3), np.float32)
    nz = mag > 0
    out[nz] = (b[:, nz] / mag[nz]).T
    return out


def _refit_inliers(L, I, e, b):
    """Least squares again on each pixel's readings the sparse error left alone
    (and above the shadows): the soft threshold leaves *lam* of every outlier
    in, which tilts the normal, so the support is kept and the bias dropped.
    Pixels with fewer than three such readings keep *b*."""
    floor = SHADOW_FRACTION * I.max(axis=0, initial=0.0)[None, :]
    W = ((e == 0) & (I > floor)).astype(np.float64)
    L64 = L.astype(np.float64)
    A = np.einsum("jm,ja,jb->mab", W, L64, L64)
    r = np.einsum("jm,ja,jm->ma", W, L64, I.astype(np.float64))
    ok = (W.sum(axis=0) >= 3) & (np.linalg.det(A) > 1e-12)
    out = b.astype(np.float64)
    if ok.any():
        out[:, ok] = np.linalg.solve(A[ok], r[ok][..., None])[..., 0].T
    return out.astype(np.float32)


def sparse_outlier_normals(L: np.ndarray, I: np.ndarray, h: int, w: int, lam: float = 0.3,
                           iters: int = 100, tol: float = 1e-4) -> np.ndarray:
    """Unit normals (h, w, 3) from the stack *I* (n, h·w) and lights *L* (n, 3),
    with each reading allowed a sparse error (robust PCA, per pixel).

    *lam* is the outlier threshold as a fraction of the pixel's mean reading:
    a reading further than that from the matte model is set aside by the
    amount it exceeds it.  Lower sets aside more.  The normal is then fitted
    again, by plain least squares, to the readings that were not set aside."""
    L32 = np.asarray(L, np.float32).reshape(-1, 3)
    P = np.linalg.pinv(L32.astype(np.float64)).astype(np.float32)        # (3, n)
    flat = np.asarray(I).reshape(len(L32), -1)
    if rs is not None and hasattr(rs, "sparse_ps"):
        src = np.ascontiguousarray(flat, np.float32)
        out = rs.sparse_ps(np.ascontiguousarray(L32), np.ascontiguousarray(P), src,
                           float(lam), int(iters), float(tol))
        return np.asarray(out).reshape(h, w, 3)
    m = flat.shape[1]
    out = np.empty((m, 3), np.float32)
    for a in range(0, m, _CHUNK):
        z = min(m, a + _CHUNK)
        out[a:z] = _sparse_block_numpy(L32, P, flat[:, a:z], lam, iters, tol)
    return out.reshape(h, w, 3)


def _excess_block_numpy(L, N, I):
    I = I.astype(np.float32)
    s = np.maximum(L @ N.T, 0.0)                                  # (n, m)
    den = (s * s).sum(axis=0)
    rho = np.where(den > 0, (I * s).sum(axis=0) / np.maximum(den, 1e-12), 0.0)
    mean = np.maximum(I.mean(axis=0), 1e-6)
    return (np.max(I - rho[None, :] * s, axis=0) / mean).astype(np.float32)


def specular_excess(L: np.ndarray, normals: np.ndarray, I: np.ndarray) -> np.ndarray:
    """How far each pixel's brightest reading rises above the matte model with
    its own normal and best albedo, over its mean reading: (h, w) float32.

    About 0.1-0.4 on paper, parchment and leather (noise, interreflection);
    1 and well above on gilding, metal and varnish catching the light."""
    h, w = normals.shape[:2]
    L32 = np.asarray(L, np.float32).reshape(-1, 3)
    N = np.ascontiguousarray(normals.reshape(-1, 3), np.float32)
    flat = np.asarray(I).reshape(len(L32), -1)
    if rs is not None and hasattr(rs, "specular_excess"):
        out = rs.specular_excess(np.ascontiguousarray(L32), N, np.ascontiguousarray(flat, np.float32))
        return np.asarray(out).reshape(h, w)
    m = flat.shape[1]
    out = np.empty(m, np.float32)
    for a in range(0, m, _CHUNK):
        z = min(m, a + _CHUNK)
        out[a:z] = _excess_block_numpy(L32, N[a:z], flat[:, a:z])
    return out.reshape(h, w)


def excess_threshold(sensitivity: float) -> float:
    """The specular excess above which a pixel counts as metal: sensitivity
    0 -> 1.6 (only bright gold), 0.5 -> 0.9, 1 -> 0.2 (any sheen)."""
    return 1.6 - 1.4 * float(np.clip(sensitivity, 0.0, 1.0))


def flare(I: np.ndarray, L: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """The appearance cue, which needs no normal: (flare, brightness), each (h, w).

    *flare* is a pixel's brightest reading over its median, divided by what a
    flat matte surface facing the camera gives under the same lights (the
    brightest lamp's height over the median lamp's).  About 1 on paper,
    parchment and wax; well above on gold and metal catching the light.  It
    is what tells metal with only four lamps, where every reading is needed
    for the fit and the residual cannot show a highlight (the fit takes it in).

    *brightness* is the median reading over the capture's 99th percentile:
    black velvet and shadow flare too, and are told apart by being dark.

    The flare is then taken relative to the matte about it (``_local_floor``):
    lamps close to the object light one edge of the sheet far more than the
    other, so the flare there rises with no metal at all, and a grey card at
    the edge of the frame flares with it."""
    from helio.core.parallel import frames_median
    n = len(L)
    stack = np.asarray(I).reshape(n, h, w)
    med = frames_median(stack).astype(np.float32)
    mx = stack.max(axis=0).astype(np.float32)
    lz = np.clip(np.asarray(L, np.float64)[:, 2], 0.05, 1.0)
    geo = float(lz.max() / np.median(lz))
    out = mx / np.maximum(med, 1e-3) / geo
    top = float(np.percentile(med, 99)) or 1.0
    bright = med / top
    out /= _local_floor(out, bright)
    return out.astype(np.float32), bright.astype(np.float32)


FLOOR_TILES = 8          # tiles along the longer side for the local matte flare
FLOOR_PERCENTILE = 10    # the low end of the bright pixels in a tile: its matte, even where metal is most of it


def _local_floor(fl: np.ndarray, bright: np.ndarray) -> np.ndarray:
    """The flare of the matte about each pixel: in coarse tiles, the low end of
    the bright pixels' flare (dark ones are not trusted) pooled over the tile
    and its neighbours, smoothed and brought back to full size.  Pooling over
    the neighbours measures a gilt panel that fills its tile against the matte
    beside it; pooling rather than taking the lowest neighbour keeps a sheet's
    lamp-side edge measured against itself.  Tiles with too few bright pixels
    take the median of the others."""
    h, w = fl.shape
    ty = max(1, round(FLOOR_TILES * h / max(h, w)))
    tx = max(1, round(FLOOR_TILES * w / max(h, w)))
    ys = np.linspace(0, h, ty + 1).astype(int)
    xs = np.linspace(0, w, tx + 1).astype(int)
    step = max(1, int(np.sqrt(h * w / 4e6)))           # a sample is plenty for a percentile
    grid = np.full((ty, tx), np.nan, np.float32)
    for i in range(ty):
        for j in range(tx):
            a, b = ys[max(0, i - 1)], ys[min(ty, i + 2)]
            c, d = xs[max(0, j - 1)], xs[min(tx, j + 2)]
            f = fl[a:b:step, c:d:step]
            ok = bright[a:b:step, c:d:step] > BRIGHT_GATE
            if ok.sum() >= max(16, 0.05 * ok.size):
                grid[i, j] = np.percentile(f[ok], FLOOR_PERCENTILE)
    if np.isnan(grid).all():
        return np.ones((h, w), np.float32)
    grid[np.isnan(grid)] = np.nanmedian(grid)
    if min(ty, tx) > 2:
        grid = cv2.GaussianBlur(grid, (0, 0), 0.7)
    return np.maximum(cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR), 1e-3)


def flare_threshold(sensitivity: float) -> float:
    """The flare over the matte about it above which a bright pixel counts as
    metal: sensitivity 0 -> 1.6, 0.5 -> 1.35, 1 -> 1.1."""
    return 1.6 - 0.5 * float(np.clip(sensitivity, 0.0, 1.0))


BRIGHT_GATE = 0.2       # a pixel darker than this share of the capture's bright end is not metal


def metal_mask(excess: np.ndarray, sensitivity: float = 0.3, feather_px: float = 1.5,
               flare_map: np.ndarray | None = None, bright: np.ndarray | None = None) -> np.ndarray:
    """Blend weights (h, w) float32 in 0..1: 1 on the shining areas, 0 on matte.

    A pixel shines when its readings break the matte model (*excess*: needs
    enough lamps to show it) or, given *flare_map* and *bright*, when it
    flares far above what matte would under these lamps and is not dark.
    The raw threshold is opened (stray noisy pixels go), closed (a tooled
    line or a couched thread is one region, not a dotted one), freed of
    specks smaller than a few pixels, and feathered so the two solves meet
    without a seam."""
    raw = np.nan_to_num(excess, nan=0.0) > excess_threshold(sensitivity)
    close = 5
    if flare_map is not None and bright is not None:
        f = cv2.medianBlur(np.ascontiguousarray(flare_map, np.float32), 5)
        raw |= (f > flare_threshold(sensitivity)) & (bright > BRIGHT_GATE)
        close = 9                                # thread and tooling: join the glints into the motif
    raw = raw.astype(np.uint8)
    if not raw.any():
        return np.zeros(excess.shape, np.float32)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))
    m = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k3)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc)
    min_area = max(9, int(excess.size * 2e-6))
    n_lab, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_lab > 1:
        keep = np.zeros(n_lab, np.uint8)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        m = keep[lab]
    m = cv2.dilate(m, k3)                          # the rim of a highlight shines too
    out = m.astype(np.float32)
    if feather_px > 0 and out.any():
        out = cv2.GaussianBlur(out, (0, 0), float(feather_px))
        out = np.maximum(out, m.astype(np.float32) * 0.999)
    return np.clip(out, 0.0, 1.0)


def blend_normals(base: np.ndarray, other: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """*base* with *other* blended in by *weight* (h, w), renormalised, in place."""
    sel = weight > 0
    if not sel.any():
        return base
    wv = weight[sel][:, None]
    mix = base[sel] * (1.0 - wv) + other[sel] * wv
    mag = np.linalg.norm(mix, axis=1, keepdims=True)
    base[sel] = np.where(mag > 0, mix / np.maximum(mag, 1e-12), base[sel])
    return base
