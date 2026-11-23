"""A synthetic capture for the Qt tests: a known surface and twelve lights.

The same surface and lights as Helio 3's docs/pyside6-migration/tools/
capture_baseline.py (only the parts a test needs, without Helio Classic):
a tilted plane with a bump, a ridge and an L-shaped incision, lit from twelve
directions at 25-65 degrees.
"""
from __future__ import annotations

import numpy as np

H, W, NLIGHTS = 96, 128, 12


def surface(h=H, w=W):
    """Height field in pixels-of-depth units; asymmetric in x, y and sign."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = 0.020 * xx + 0.045 * yy                                   # tilted plane, |b| > |a|
    z += 9.0 * np.exp(-(((xx - 0.30 * w) / 10.0) ** 2 + ((yy - 0.32 * h) / 8.0) ** 2))  # bump UL
    d = ((xx - 0.55 * w) + (yy - 0.70 * h)) / np.sqrt(2)           # ridge along x = -y line
    z += 3.0 * np.exp(-(d / 4.0) ** 2) * (xx > 0.45 * w)
    groove = ((np.abs(yy - 0.15 * h) < 1.5) & (xx > 0.60 * w) & (xx < 0.90 * w)) | \
             ((np.abs(xx - 0.60 * w) < 1.5) & (yy > 0.15 * h) & (yy < 0.40 * h))
    z -= 2.0 * groove                                               # L-shaped incision, upper right
    from scipy.ndimage import gaussian_filter
    z = gaussian_filter(z, 0.8)                                     # keep it differentiable
    zy, zx = np.gradient(z)                                         # zy: slope down the rows
    n = np.dstack([-zx, zy, np.ones_like(z)])                       # +Y up (helio.core.frames)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return z.astype(np.float32), n.astype(np.float32)


def lights(n=NLIGHTS):
    out = []
    for i in range(n):
        az = 2 * np.pi * i / n + 0.17                       # no light on an axis
        el = np.radians(25 + 40 * ((i * 7) % n) / n)        # elevations 25-65 deg, mixed
        out.append([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)])   # +Y up
    L = np.array(out, np.float32)
    return L / np.linalg.norm(L, axis=1, keepdims=True)


