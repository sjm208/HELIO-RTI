"""Where each light is, and how well the chrome ball measured it.

**Positions from two balls.**  One chrome ball gives each light's direction
from where the ball stands.  Two balls give two directions from two places;
where the two rays meet is the light (triangulation: the midpoint of the
shortest segment between them).  The near-field solve can then put every
light at its own position, rather than all of them on one dome of an assumed
radius.  Rays that are nearly parallel (under ``PARALLEL_DEG`` apart) say the
light is too far away for the two balls to tell; that light is left distant.

Coordinates are the photograph's pixels: x right, y down from the top-left,
z towards the camera, in the same pixels (the camera taken as looking
straight down).  A ball's centre stands a radius above the board, and so do
the rays; positions come out above the board on the same scale.

**How well the directions are known.**  A light's direction comes from the
ball's centre, its radius and the highlight's place on it.  Each is uncertain
by a fraction of a pixel: the centre and radius by the rim fit's residual
over the square root of the points it used, the highlight by
``HIGHLIGHT_SIGMA_PX``.  ``direction_sigma_deg`` carries those through to
degrees, and ``normal_error_deg`` carries the lights' errors through a solve
to the normals (Monte Carlo on matte normals up to 60° from the camera).
That is the error the calibration alone puts into every normal, before
shine, shadow or noise; the uncertainty map does not include it.
"""
from __future__ import annotations

import math

import numpy as np

PARALLEL_DEG = 1.0              # rays closer than this: the light is too far to triangulate
HIGHLIGHT_SIGMA_PX = 0.5        # a highlight's centre, found to sub-pixel
ASSUMED_RIM_PX = 1.0            # a ball outline drawn by hand, not fitted to its rim


def triangulate(c1, d1, c2, d2):
    """(position (3,), miss in px) where the rays c1 + t d1 and c2 + t d2 come
    closest, or None when they are nearly parallel or meet behind the balls."""
    c1, d1, c2, d2 = (np.asarray(v, np.float64) for v in (c1, d1, c2, d2))
    d1, d2 = d1 / np.linalg.norm(d1), d2 / np.linalg.norm(d2)
    b = float(d1 @ d2)
    if math.degrees(math.acos(max(-1.0, min(1.0, b)))) < PARALLEL_DEG:
        return None
    w0 = c1 - c2
    d, e = float(d1 @ w0), float(d2 @ w0)
    den = 1.0 - b * b
    t1, t2 = (b * e - d) / den, (e - b * d) / den
    if t1 <= 0 or t2 <= 0:
        return None
    p1, p2 = c1 + t1 * d1, c2 + t2 * d2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))


def _to_photo(L):
    """A light direction (+Y up) in the photograph's axes (y down)."""
    return np.array([L[0], -L[1], L[2]], np.float64)


def positions_from_two_balls(cal) -> dict | None:
    """Each light's position from a two-ball calibration (helio.core.sphere.Calibration).

    Returns dict(positions (n, 3) with rows of nan where a light could not be
    placed, miss (n,) px, placed: count, distance_px: median distance from the
    balls' midpoint), or None when the calibration has no second ball."""
    from helio.core.sphere import light_from_highlight
    s1, s2 = cal.sphere, getattr(cal, "sphere2", None)
    h1, h2 = cal.highlights, getattr(cal, "highlights2", None)
    if s2 is None or h2 is None:
        return None
    view = cal.view
    n = len(h1)
    pos = np.full((n, 3), np.nan)
    miss = np.full(n, np.nan)
    c1 = np.array([s1.cx, s1.cy, s1.r])
    c2 = np.array([s2.cx, s2.cy, s2.r])
    for i in range(n):
        a, b = h1[i], h2[i]
        if not (a.ok and b.ok) or a.reason == "from the second ball only":
            continue
        d1 = _to_photo(light_from_highlight(s1, a.x, a.y, view))
        d2 = _to_photo(light_from_highlight(s2, b.x, b.y, view))
        t = triangulate(c1, d1, c2, d2)
        if t is not None:
            pos[i], miss[i] = t
    ok = np.isfinite(pos).all(axis=1)
    mid = 0.5 * (c1 + c2)
    dist = float(np.median(np.linalg.norm(pos[ok] - mid, axis=1))) if ok.any() else float("nan")
    return dict(positions=pos, miss=miss, placed=int(ok.sum()), distance_px=dist)


def direction_sigma_deg(sphere, highlights, view=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Each light's 1σ direction error (degrees) from the ball's fit and the highlight's place."""
    from helio.core.sphere import Sphere, light_from_highlight
    rim = sphere.rms if (sphere.rms == sphere.rms and sphere.rms > 0) else ASSUMED_RIM_PX
    pts = max(int(sphere.inliers or 0), 1)
    s_c = rim / math.sqrt(max(pts / 2.0, 1.0))          # each centre coordinate
    s_r = rim / math.sqrt(pts)
    out = np.full(len(highlights), np.nan)
    for i, hl in enumerate(highlights):
        if not hl.ok:
            continue
        L0 = light_from_highlight(sphere, hl.x, hl.y, view)
        var = 0.0
        for dcx, dcy, dr, dhx, dhy, s in ((1, 0, 0, 0, 0, s_c), (0, 1, 0, 0, 0, s_c),
                                          (0, 0, 1, 0, 0, s_r), (0, 0, 0, 1, 0, HIGHLIGHT_SIGMA_PX),
                                          (0, 0, 0, 0, 1, HIGHLIGHT_SIGMA_PX)):
            sp = Sphere(sphere.cx + dcx * s, sphere.cy + dcy * s, sphere.r + dr * s)
            L1 = light_from_highlight(sp, hl.x + dhx * s, hl.y + dhy * s, view)
            var += math.degrees(math.acos(max(-1.0, min(1.0, float(L0 @ L1))))) ** 2
        out[i] = math.sqrt(var)
    return out


def normal_error_deg(L, sigma_deg, *, trials: int = 64, seed: int = 0) -> float:
    """The median angle (degrees) by which lights *L* (n, 3), each off by *sigma_deg*
    (n,) in a random direction, put least-squares normals out (matte, up to 60° tilt)."""
    L = np.asarray(L, np.float64)
    sig = np.radians(np.nan_to_num(np.asarray(sigma_deg, np.float64), nan=0.0))
    ok = np.isfinite(L).all(axis=1)
    L, sig = L[ok], sig[ok]
    if len(L) < 3:
        return float("nan")
    rng = np.random.default_rng(seed)
    m = 400
    th = np.arccos(rng.uniform(math.cos(math.radians(60)), 1.0, m))
    ph = rng.uniform(0, 2 * math.pi, m)
    N = np.column_stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    I = np.clip(L @ N.T, 0, None)                                      # (n, m)
    errs = []
    for _ in range(trials):
        axis = rng.normal(size=L.shape)
        axis -= (axis * L).sum(1, keepdims=True) * L                  # perpendicular to each light
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)
        ang = rng.normal(size=len(L)) * sig
        Lp = L * np.cos(ang)[:, None] + axis * np.sin(ang)[:, None]
        G = np.linalg.lstsq(Lp, I, rcond=None)[0].T
        G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
        errs.append(np.degrees(np.arccos(np.clip((G * N).sum(1), -1, 1))))
    return float(np.median(np.concatenate(errs)))


def calibration_error(cal) -> dict | None:
    """dict(direction_deg: median 1σ of the lights' directions, normal_deg: what that puts
    into the normals, assumed: the ball's outline was not fitted) for a sphere calibration."""
    s = cal.sphere
    sig = direction_sigma_deg(s, cal.highlights, cal.view)
    if getattr(cal, "sphere2", None) is not None and getattr(cal, "highlights2", None):
        sig2 = direction_sigma_deg(cal.sphere2, cal.highlights2, cal.view)
        both = np.isfinite(sig) & np.isfinite(sig2)
        sig = np.where(both, np.sqrt(sig ** 2 + sig2 ** 2) / 2.0, np.where(np.isfinite(sig), sig, sig2))
    if not np.isfinite(sig).any():
        return None
    return dict(direction_deg=float(np.nanmedian(sig)),
                normal_deg=normal_error_deg(cal.lights, sig),
                assumed=not (s.rms == s.rms and s.rms > 0))
