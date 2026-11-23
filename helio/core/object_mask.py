"""The object apart from a dark surround, and depth integrated on the object alone.

A small object on black velvet (a steel punch, a coin, a seal) leaves most of
the frame too dark to read a surface direction from: there the solved normals
are noise.  Every whole-frame integrator (Poisson, Frankot-Chellappa, DCT)
sums that noise into the object's depth as broad, blotchy hills, and a solver
that leaves out "shadowed" frames makes it far worse, because to it the whole
surround is in shadow.

``object_mask`` finds the object: the pixels bright enough to read (a tenth of
the 99th-percentile albedo, as process_tools.readable), cleaned of speckle,
holes filled, small islands dropped.  None when the object fills the frame (no
surround to leave out) or when nothing coherent is found.

``masked_depth`` integrates on the object alone:

* the broad shape: least squares over the object's pixels only (free boundary,
  Quéau, Durou and Aujol 2018), the surround held flat and coupled to the
  object only very weakly, solved on a map of at most *max_px* pixels;
* the fine detail: the chosen integrator's own solution, with the surround's
  normals set straight up, high-passed above that scale.

Measured on a 50-light capture of a steel type punch on black (9% of the
frame is punch): the depth's own slopes disagree with the normals by 10° median
over the punch with whole-frame Poisson, 6° on the object alone.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

FILL_LIMIT = 0.85        # an object over this fraction of the frame has no surround to leave out
MIN_FRACTION = 0.003     # below this, nothing coherent was found
SURROUND_WEIGHT = 1e-3   # how weakly the flat surround holds the object
# A small object on its surround (a punch, a coin: a few per cent of the frame) gains
# from the mask.  A large one (a binding at 64 %) loses: held flat all round, it comes
# out a raised pillow with lifted edges and its tooling softened (2026-11-20), so
# Set Up for Measurement turns the mask on only below this.
SMALL_OBJECT = 0.40


def small_object(albedo) -> bool | None:
    """Whether *albedo* shows a small object on a dark surround (``object_mask``
    finds one, under ``SMALL_OBJECT`` of the frame); None when there is no albedo."""
    if albedo is None:
        return None
    m = object_mask(albedo)
    return m is not None and float(m.mean()) < SMALL_OBJECT


MASK_SIDE = 2000         # the mask is found at most this long a side, then scaled back up


def object_mask(albedo) -> np.ndarray | None:
    """Bool (h, w): the object, apart from a dark surround; None if there is none to leave out.

    Found on a copy at most MASK_SIDE pixels long and scaled back up: at full size
    (63 MP) the closing alone, a 135-pixel ellipse, took 4 s on the main thread and
    the window stopped answering when a click asked for it more than once."""
    if albedo is None:
        return None
    a = np.asarray(albedo)
    if a.ndim < 2:
        return None
    h, w = a.shape[:2]
    k = MASK_SIDE / max(h, w)
    if k < 1.0 and min(h, w) * k >= 16:
        small = cv2.resize(a if a.dtype in (np.uint8, np.float32) else a.astype(np.float32),
                           (max(16, round(w * k)), max(16, round(h * k))), interpolation=cv2.INTER_AREA)
        m = _object_mask(small)
        if m is None:
            return None
        return cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return _object_mask(a)


def _object_mask(albedo) -> np.ndarray | None:
    a = np.asarray(albedo, np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2 or min(a.shape) < 16:
        return None
    top = float(np.percentile(a, 99))
    if not np.isfinite(top) or top <= 0:
        return None
    m = (a >= 0.10 * top).astype(np.uint8)
    if m.mean() > FILL_LIMIT:
        return None
    k = max(3, int(round(min(a.shape) / 150)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3 * k, 3 * k)))
    n, lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.002 * m.size]
    if not big:
        return None
    keep = np.isin(lab, big).astype(np.uint8)
    # fill holes: whatever the outside cannot reach
    outside = np.pad(1 - keep, 1, constant_values=1)
    flood = np.zeros((outside.shape[0] + 2, outside.shape[1] + 2), np.uint8)
    cv2.floodFill(outside, flood, (0, 0), 2)
    keep = (outside[1:-1, 1:-1] != 2).astype(bool)
    frac = float(keep.mean())
    if frac < MIN_FRACTION or frac > FILL_LIMIT:
        return None
    return keep


def flatten_surround(normals, mask) -> np.ndarray:
    """*normals* with the surround's set straight up (no slope)."""
    n = np.array(normals, np.float32, copy=True)
    n[~mask] = (0.0, 0.0, 1.0)
    return n


def _least_squares_on(n, m):
    """Depth (small-grid pixels) on mask *m* from normals *n*: forward differences,
    weight 1 inside the object, SURROUND_WEIGHT elsewhere."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg
    from helio.core.frames import normal_slopes
    h, w = m.shape
    gx, gy = normal_slopes(n)
    idx = np.arange(h * w).reshape(h, w)
    wx = np.where(m[:, :-1] & m[:, 1:], 1.0, SURROUND_WEIGHT).ravel()
    wy = np.where(m[:-1, :] & m[1:, :], 1.0, SURROUND_WEIGHT).ravel()
    bx = (0.5 * (gx[:, :-1] + gx[:, 1:])).ravel()
    by = (0.5 * (gy[:-1, :] + gy[1:, :])).ravel()
    a0, a1 = idx[:, :-1].ravel(), idx[:, 1:].ravel()
    b0, b1 = idx[:-1, :].ravel(), idx[1:, :].ravel()
    nx_, ny_ = len(a0), len(b0)
    rows = np.r_[np.arange(nx_), np.arange(nx_), nx_ + np.arange(ny_), nx_ + np.arange(ny_)]
    cols = np.r_[a1, a0, b1, b0]
    vals = np.r_[np.ones(nx_), -np.ones(nx_), np.ones(ny_), -np.ones(ny_)]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(nx_ + ny_, h * w))
    W = np.r_[wx, wy]
    AtW = A.T.multiply(W).tocsr()
    M = (AtW @ A + sp.identity(h * w, format="csr") * 1e-9).tocsr()
    rhs = AtW @ np.r_[bx, by]
    # a close start: the Poisson solution of the same slopes
    from helio.core.depth_integration import poisson_solver
    x0 = poisson_solver(gx, gy).astype(np.float64).ravel()
    x0 -= x0.mean()
    D = sp.diags(1.0 / np.clip(M.diagonal(), 1e-9, None))
    z, _info = cg(M, rhs, x0=x0, M=D, maxiter=6000, rtol=1e-6)
    z = z.reshape(h, w)
    sur = ~m
    return z - (float(np.median(z[sur])) if sur.any() else float(np.median(z)))


def masked_depth(normals, mask, integrate, *, max_px: int = 1_000_000) -> np.ndarray:
    """Depth in pixels on the object alone (see the module notes).

    *integrate(normals) -> depth* is the chosen whole-frame integrator; it gives
    the fine detail.  The surround comes out flat at zero."""
    from helio.core.metrology import true_shape_scale
    n = flatten_surround(normals, mask)
    h, w = mask.shape
    z_full = np.asarray(integrate(n), np.float32)
    z_full = z_full * np.float32(true_shape_scale(z_full, n))          # into pixels
    ys, xs = np.nonzero(mask)
    pad = max(8, int(0.05 * max(h, w)))
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    ch, cw = y1 - y0, x1 - x0
    s = min(1.0, math.sqrt(max_px / float(ch * cw)))
    sh, sw = max(8, int(round(ch * s))), max(8, int(round(cw * s)))
    nc = n[y0:y1, x0:x1]
    mc = mask[y0:y1, x0:x1]
    if s < 1.0:
        ns = cv2.resize(nc, (sw, sh), interpolation=cv2.INTER_AREA)
        ns /= np.maximum(np.linalg.norm(ns, axis=-1, keepdims=True), 1e-8)
        ms = cv2.resize(mc.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.5
    else:
        ns, ms = nc, mc
    zs = _least_squares_on(ns, ms) * (cw / float(sw))                  # back to full-size pixels
    coarse = zs if s >= 1.0 else cv2.resize(zs.astype(np.float32), (cw, ch), interpolation=cv2.INTER_CUBIC)
    # The surround is flat at zero; the object stands on it, levelled so the
    # ring just outside its edge is at zero.
    ring = cv2.dilate(mc.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~mc
    coarse = coarse - (float(np.median(coarse[ring])) if ring.any() else float(np.median(coarse[mc])))
    out = np.zeros((h, w), np.float32)
    out[y0:y1, x0:x1] = np.where(mc, coarse, 0.0)
    if s < 1.0:                                                        # the detail finer than the solve's grid
        lw, lh = max(8, int(round(w * s))), max(8, int(round(h * s)))
        low = cv2.resize(cv2.resize(z_full, (lw, lh), interpolation=cv2.INTER_AREA), (w, h),
                         interpolation=cv2.INTER_CUBIC)
        detail = z_full - low
        out += np.where(mask, detail, 0.0).astype(np.float32)
    # The surround carries on from the object's rim instead of sitting at zero.  A
    # large object's rim is not at the surround's level (a binding's rounded boards
    # end 40 % of the relief's range below it), and a surround pinned at zero stood
    # up from it as a cliff all round: the 3D relief's Height made a wall of it, and
    # its detail-vs-shape high-pass a ridge inside the edge and a trench outside
    # (2026-11-20).  A small object whose rim meets the surround is as it was.
    return np.where(mask, out, extend_from(out, mask)).astype(np.float32)


def extend_from(z, known) -> np.ndarray:
    """*z* where *known*, carried smoothly into the rest with no step at the edge.

    A pixel *d* away from the known part takes the mean of the known rim (the band
    just inside the edge) under a Gaussian about 2 *d* wide: at the edge that is the
    rim itself, farther out the rim averaged along its length, so a ragged rim (a
    binding's worn edge, tooling running off it) is not drawn out into streaks.
    Far off, the surround settles to the rim's median level."""
    z = np.asarray(z, np.float32)
    kn = np.asarray(known, bool)
    if kn.all() or not kn.any():
        return z.copy()
    h, w = z.shape
    big = max(h, w)
    # exact distances: the 5x5 approximation differed by half a pixel from one
    # column to the next, which a raking light drew as streaks across the surround
    d_out = cv2.distanceTransform((~kn).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_in = cv2.distanceTransform(kn.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    rim = kn & (d_in <= max(4.0, 0.01 * big))           # only the rim speaks for the surround
    del d_in
    wz = np.where(rim, z, 0.0).astype(np.float32)
    wk = rim.astype(np.float32)
    level = float(np.median(z[rim]))
    smax = max(4.0, big / 4.0)
    sigmas = [2.0]
    while sigmas[-1] < smax:
        sigmas.append(sigmas[-1] * 2.0)
    pos = np.log2(np.clip(2.0 * d_out, 2.0, sigmas[-1]))  # which scale (twice the distance), between two
    acc = np.zeros((h, w), np.float32)
    tot = np.zeros((h, w), np.float32)
    for i, sg in enumerate(sigmas):
        wt = np.clip(1.0 - np.abs(pos - (i + 1)), 0.0, 1.0).astype(np.float32)
        if not wt.any():
            continue
        f = max(1.0, sg / 4.0)                          # blur on a reduced copy: same result, faster
        size = (max(1, int(round(w / f))), max(1, int(round(h / f))))
        num = cv2.GaussianBlur(cv2.resize(wz, size, interpolation=cv2.INTER_AREA), (0, 0), sg / f)
        den = cv2.GaussianBlur(cv2.resize(wk, size, interpolation=cv2.INTER_AREA), (0, 0), sg / f)
        val = np.where(den > 1e-6, num / np.maximum(den, 1e-6), level).astype(np.float32)
        if f > 1.0:
            val = cv2.resize(val, (w, h), interpolation=cv2.INTER_LINEAR)
        acc += wt * val
        tot += wt
    fill = np.where(tot > 0, acc / np.maximum(tot, 1e-6), level).astype(np.float32)
    del acc, tot, pos
    settle = big * 0.08
    fill = level + (fill - level) * np.exp(-d_out / settle).astype(np.float32)
    return np.where(kn, z, fill).astype(np.float32)


def depth_agreement(normals, depth, mask=None, *, side: int = 800) -> float | None:
    """How far the depth's own slopes disagree with the normals, in degrees (median
    over *mask*, or the whole frame): 0 when the depth is exactly the surface the
    normals describe.  None when there is nothing to judge."""
    from helio.core.frames import normal_slopes, normals_from_slopes
    if normals is None or depth is None:
        return None
    n = np.asarray(normals, np.float32)
    d = np.asarray(depth, np.float32)
    if n.ndim != 3 or d.shape != n.shape[:2] or min(d.shape) < 8:
        return None
    step = max(1, int(math.ceil(max(d.shape) / side)))
    m = np.ones(d.shape, bool) if mask is None else np.asarray(mask, bool)
    if step > 1:                                                       # a block mean keeps the slopes honest
        hh, ww = (d.shape[0] // step) * step, (d.shape[1] // step) * step
        d = d[:hh, :ww].reshape(hh // step, step, ww // step, step).mean(axis=(1, 3)) / step
        n = n[:hh, :ww].reshape(hh // step, step, ww // step, step, 3).mean(axis=(1, 3))
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        m = m[:hh, :ww].reshape(hh // step, step, ww // step, step).mean(axis=(1, 3)) > 0.5
    m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (n[..., 2] > 0.2)
    if np.count_nonzero(m) < 50:
        return None
    gy, gx = np.gradient(d.astype(np.float64))
    px, py = normal_slopes(n)
    g = np.concatenate([gx[m], gy[m]])
    t = np.concatenate([px[m], py[m]])
    gg = float(g @ g)
    k = float(g @ t) / gg if gg > 1e-12 else 1.0
    if not np.isfinite(k) or k <= 0:
        k = 1.0
    dn = normals_from_slopes(gx * k, gy * k, 1.0)
    ang = np.degrees(np.arccos(np.clip(np.sum(dn * n, axis=-1), -1.0, 1.0)))
    return float(np.median(ang[m]))
