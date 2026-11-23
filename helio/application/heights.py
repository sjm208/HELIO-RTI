"""Heights to measure with: the solve's surface in pixels of true height.

Photometric stereo measures slopes, and the integrated depth (``depth_poisson``)
is those slopes summed: heights in the photographs' own pixels.  Every height
Helio gives in millimetres is read from this map and the height scale, which is
*millimetres per pixel of height* (``Project.z_mm_per_unit``, HEIGHT_BASIS).

Not the display depth (``depth_vis_p``): that is the depth less a broad blur,
clipped at its 0.5 % extremes and stretched to 0-1, so a feature's height there
depends on its width (a bump 1,000 px across read 14 % of an equally tall one
80 px across) and the tallest peaks are cut flat.  It is for looking, not for
measuring (checked against known heights in tests/test_measure_heights.py).

The depth is put into pixels with ``true_shape_scale``: the scale that makes the
map's slopes agree with the normals, both smoothed alike so that grain and
noise do not sway it.  Least-squares, Poisson, Simchony and
Frankot-Chellappa depths are in pixels already (scale 1); the other integrators
are brought to it.
"""
from __future__ import annotations

import logging

import numpy as np

_log = logging.getLogger(__name__)

# What z_mm_per_unit means, saved with a session: a height scale saved without it
# was measured on the display depth and cannot be carried over.
HEIGHT_BASIS = "true-height-px"

_held: dict = {}


def height_px(D) -> np.ndarray | None:
    """Heights in pixels above the lowest point (float32, the solve's full size), or
    None without a depth.

    *D* is a SolveResult or the render data: anything with ``depth_poisson`` and
    ``normals_full``.  Made once per solve and kept."""
    depth = getattr(D, "depth_poisson", None)
    if depth is None:
        return None
    normals = getattr(D, "normals_full", None)
    hit = _held.get("h")
    if hit is not None and hit[0] is depth and hit[1] is normals:
        return hit[2]
    from helio.core.metrology import true_shape_scale
    d = np.asarray(depth, np.float32)
    if normals is not None and np.shape(normals)[:2] == d.shape:
        s = true_shape_scale(d, normals)
    else:
        _log.warning("height_px: no normals the depth's size; the depth taken as pixels of height")
        s = 1.0
    out = d * np.float32(s)
    out -= out.min()                                  # above the lowest point: 0 there
    _held["h"] = (depth, normals, out)
    return out


def mm_factor(z_mm_per_unit) -> float | None:
    """Millimetres per pixel of height, or None when no height scale is set."""
    try:
        z = float(z_mm_per_unit or 0.0)
    except (TypeError, ValueError):
        return None
    return z if z > 0 else None
