"""Solved reflectance: the surface's own colour, with the shading taken out.

The colour Helio shows by default is the median of the frames.  It looks right
and rejects highlights and shadows, but a slope turned from most of the lamps
stays a little darker.  Photometric stereo offers better: each frame says
I = ρ N·L, so with the normals solved ρ is I / (N·L), the same in every frame
if the surface is matt.

That holds for light, so this needs linear frames: RAW.  Per pixel, ρ is the
median over the frames that can see it (N·L not grazing, not clipped), so cast
shadows and highlights are rejected as in the colour picture.  The colour's
proportions (R:G:B) come from the median in linear light, the level from ρ:
the proportions hardly change with the light, and no colour frames need
keeping.

Relative, not absolute: without a calibrated white the scale is free, so the
level is set to the median colour's.  GUI-free.
"""
from __future__ import annotations

import numpy as np

from helio.core.radiometry import linear_to_srgb, srgb_to_linear

GRAZING = 0.15          # N·L below this: too near the shadow edge to divide by
CLIPPED = 250.0         # stored 0-255: at the sensor's top
LUMA = np.array([0.299, 0.587, 0.114], np.float32)     # as the loader's grey


def solved_level(stack, lights, normals, *, rows: int = 512) -> np.ndarray:
    """ρ (h, w) float32, from *stack* (n, h, w) linear, 0-255, *lights* (n, 3)
    and *normals* (h, w, 3).  NaN where fewer than three frames see the pixel."""
    L = np.asarray(lights, np.float32)
    n, h, w = stack.shape
    out = np.empty((h, w), np.float32)
    for y in range(0, h, rows):
        I = np.asarray(stack[:, y:y + rows], np.float32)
        shade = np.moveaxis(np.asarray(normals[y:y + rows], np.float32) @ L.T, -1, 0)
        ok = (shade > GRAZING) & (I < CLIPPED)
        r = np.where(ok, I / np.maximum(shade, GRAZING), np.inf)
        r.sort(axis=0)
        m = ok.sum(axis=0)
        v = np.take_along_axis(r, ((np.maximum(m, 1) - 1) // 2)[None], axis=0)[0]
        v2 = np.take_along_axis(r, (np.maximum(m, 1) // 2)[None], axis=0)[0]
        out[y:y + rows] = np.where(m >= 3, (v + v2) / 2, np.nan)
    return out


def solved_reflectance(stack, lights, normals, median_rgb8) -> np.ndarray:
    """(h, w, 3) uint8 sRGB: the median colour's proportions at the solved level."""
    lin = srgb_to_linear(median_rgb8, 255.0)
    luma = np.maximum(lin @ LUMA, 1e-3)
    rho = solved_level(stack, lights, normals)
    good = np.isfinite(rho)
    k = float(np.median(luma[good]) / max(np.median(rho[good]), 1e-6)) if good.any() else 1.0
    rho = np.where(good, rho * k, luma)         # no frame sees it: the median colour stands
    return np.round(linear_to_srgb(lin * (rho / luma)[..., None], 255.0)).astype(np.uint8)
