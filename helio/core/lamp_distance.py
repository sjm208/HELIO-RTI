"""How far the lamps were from the object, from the photographs themselves.

A lamp close to the object lights each point from a slightly different
direction, and the near side more brightly than the far side.  The
near-field solve (photometric_stereo.solve_near_field_normals) undoes both,
but only if it knows how far away the lamps were.  On a flat object that
matters more than anything else: with lamps 1.5 frame-widths away and no
correction, a flat binding solved as a dish twenty times deeper than its
tooling; with the correction at the wrong distance it was no better, and at
a much shorter assumed distance than the true one, worse.

The distance shows itself in the photographs.  At the right distance every
pixel's brightnesses, one per lamp, are what a matte surface facing one way
would give under those lamps (a least-squares fit leaves little over); at
the wrong distance, the directions and brightnesses assumed for each lamp
do not match what reached the pixel, and more is left over.  So the fit is
tried at a range of distances, on a reduced copy of the frames, and the one
that leaves least is taken - if it is clearly better than the lamps being
far away (the fit at a great distance), and if the curve has a clear
lowest point.  Otherwise the lamps are taken to be far enough away not to
matter, or the estimate is reported as uncertain.

Shadows, highlights and clipped values do not follow the matte model and
are left out of each pixel's fit (the darkest and brightest reading at each
pixel, and anything clipped).
"""
from __future__ import annotations

import numpy as np

__all__ = ["estimate", "RATIOS"]

# Distances tried, in multiples of the frame's longer side (from the light
# origin: the ball, or the frame's centre).
RATIOS = (0.6, 0.75, 0.9, 1.1, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.5, 8.5, 12.0)
FAR = 1000.0             # "far away": the lamps as distant lights
WORK_SIDE = 240          # the frames are reduced to this for the trial fits
CLEAR = 0.85             # the best fit must leave at most this share of the far fit's misfit


def _misfit(I, ok, L, R, xs, ys, origin, positions=None, falloff=True):
    """Mean squared misfit, as a share of the pixels' brightness, of the
    per-pixel matte fit with the lamps *R* pixels from *origin*."""
    ox, oy = origin
    S = np.asarray(L, np.float64) * R
    norm2 = np.full(len(S), R * R)
    if positions is not None:
        P0 = np.asarray(positions, np.float64) - np.array([ox, oy, 0.0])
        good = np.isfinite(P0).all(axis=1)
        S[good] = P0[good] * (R / max(float(np.mean(np.linalg.norm(P0[good], axis=1))), 1e-9))
    P = np.stack([xs - ox, ys - oy, np.zeros_like(xs)], 1)           # (m, 3)
    Lp = S[:, None, :] - P[None, :, :]                                # (n, m, 3)
    d2 = np.einsum("kmi,kmi->km", Lp, Lp)
    Lf = (Lp * (norm2[:, None] / (d2 * np.sqrt(d2)))[..., None] if falloff
          else Lp / np.sqrt(d2)[..., None])
    W = ok.astype(np.float64)                                         # (n, m)
    A = np.einsum("km,kmi,kmj->mij", W, Lf, Lf) + np.eye(3) * 1e-9
    b = np.einsum("km,kmi,km->mi", W, Lf, I)
    G = np.linalg.solve(A, b[..., None])[..., 0]
    pred = np.einsum("kmi,mi->km", Lf, G)
    res = ((I - pred) ** 2 * W).sum(0) / np.maximum(W.sum(0), 1)
    scale = (I ** 2 * W).sum(0) / np.maximum(W.sum(0), 1)
    return float(np.mean(res / np.maximum(scale, 1e-12)))


def estimate(stack, lights, origin=(0.0, 0.0), positions=None, mask=None, falloff=True):
    """The lamps' distance, from *stack* (n, h, w) in linear light and the
    measured *lights* (n, 3, unit, +Y up), as seen from *origin* (pixels
    from the frame's centre, +Y up: the ball).

    Returns {"ratio": distance over the frame's longer side, or None when
    the lamps are far enough away not to matter; *falloff* False when the
    frames' fall-off has been evened already (flat areas): the direction
    alone is fitted; "radius_px": the same in
    the stack's pixels (None likewise); "sure": the fit clearly prefers it
    (else it is a guess, and None is safer); "curve": [(ratio, misfit)],
    "far": the misfit with distant lamps; "gain": best over far}."""
    import cv2
    I = np.asarray(stack, np.float32)
    n, h, w = I.shape
    L = np.asarray(lights, np.float64)
    use = np.isfinite(L).all(axis=1)
    I, L = I[use], L[use]
    if positions is not None:
        positions = np.asarray(positions, np.float64)[use]
    n = len(L)
    out = {"ratio": None, "radius_px": None, "sure": False, "curve": [], "far": None, "gain": None,
           "far_away": False}
    if n < 5:
        return out
    k = min(1.0, WORK_SIDE / float(max(h, w)))
    hs, ws = max(8, int(round(h * k))), max(8, int(round(w * k)))
    small = np.stack([cv2.resize(f, (ws, hs), interpolation=cv2.INTER_AREA) for f in I])
    keep = np.ones((hs, ws), bool) if mask is None else \
        cv2.resize(np.asarray(mask, np.float32), (ws, hs), interpolation=cv2.INTER_AREA) > 0.99
    keep[:2], keep[-2:], keep[:, :2], keep[:, -2:] = False, False, False, False
    yy, xx = np.nonzero(keep)
    if len(yy) < 200:
        return out
    if len(yy) > 20000:
        pick = np.random.default_rng(0).choice(len(yy), 20000, replace=False)
        yy, xx = yy[pick], xx[pick]
    Iv = small[:, yy, xx].astype(np.float64)                          # (n, m)
    top = float(np.percentile(Iv, 99.9))
    ok = (Iv < 0.98 * top) & (Iv > 0.02 * top)
    lo, hi = np.argmin(Iv, axis=0), np.argmax(Iv, axis=0)
    cols = np.arange(Iv.shape[1])
    ok[lo, cols] = False                                              # the likeliest shadow
    ok[hi, cols] = False                                              # the likeliest highlight
    sel = ok.sum(0) >= 4
    Iv, ok, yy, xx = Iv[:, sel], ok[:, sel], yy[sel], xx[sel]
    if Iv.shape[1] < 200:
        return out
    # pixel centres in the full stack's pixels, from its centre, +Y up
    xs = (xx + 0.5) / k - w / 2.0
    ys = h / 2.0 - (yy + 0.5) / k
    side = float(max(h, w))
    far = _misfit(Iv, ok, L, FAR * side, xs, ys, origin, positions, falloff)
    curve = [(r, _misfit(Iv, ok, L, r * side, xs, ys, origin, positions, falloff)) for r in RATIOS]
    out["curve"], out["far"] = curve, far
    j = int(np.argmin([m for _r, m in curve]))
    best_r, best_m = curve[j]
    if 0 < j < len(curve) - 1:                                        # a parabola in log distance
        x = np.log([curve[j - 1][0], best_r, curve[j + 1][0]])
        y = np.array([curve[j - 1][1], best_m, curve[j + 1][1]])
        a, b, _c = np.polyfit(x, y, 2)
        if a > 0:
            best_r = float(np.exp(np.clip(-b / (2 * a), x[0], x[2])))
    out["gain"] = best_m / max(far, 1e-12)
    interior = 0 < j < len(curve) - 1
    out["sure"] = bool(interior and out["gain"] <= CLEAR)
    if out["gain"] > CLEAR or j == len(curve) - 1:
        out["far_away"] = out["gain"] >= 0.97 or j == len(curve) - 1  # no near distance fits better
        return out                                                    # far enough not to matter
    out["ratio"] = float(best_r)
    out["radius_px"] = float(best_r * side)
    return out
