"""Bumps <-> holes: turn a reconstructed surface inside out.

The inside-out surface of z(x, y) is -z(x, y).  Its slopes are negated, so its
unit normals are (-nx, -ny, nz): the sideways tilt flips and the component
towards the camera does not.  Every depth integrator in HELIO is linear in
the slopes, so integrating the flipped normals gives exactly the negated
depth - which is why inverting the stored result in place agrees with
processing the inverted normals from scratch.

The 0..1 display maps (depth_vis_p, depth_normalised_full, the proxies) are
percentile-clipped min-max normalisations.  The clip is symmetric, so the map
of the negated depth is 1 - map.  Applied twice it returns the original to
within float rounding (one ulp); normals and raw depth come back bit for bit.

Arrays are changed in place - they can be full resolution, and a copy of each
would double the memory for a comparison toggle.  Integer arrays are treated
as encoded (0..max standing for -1..1, or for 0..1), where inversion is
max - value; negating them would wrap around.
"""
from __future__ import annotations

import numpy as np


def _flip(channel: np.ndarray) -> None:
    """Negate a signed quantity in place, respecting integer encodings."""
    if np.issubdtype(channel.dtype, np.integer):
        np.subtract(np.iinfo(channel.dtype).max, channel, out=channel)
    else:
        np.negative(channel, out=channel)


def invert_normals_inplace(normals: np.ndarray) -> np.ndarray:
    """Negate nx and ny of an (h, w, 3) normal field, in place."""
    _flip(normals[..., 0])
    _flip(normals[..., 1])
    return normals


def invert_depth_inplace(depth: np.ndarray) -> np.ndarray:
    """Negate a raw (integration-unit) depth map, in place."""
    np.negative(depth, out=depth)
    return depth


def invert_unit_depth_inplace(depth01: np.ndarray) -> np.ndarray:
    """Turn a 0..1 normalised depth map into 1 - map, in place."""
    if np.issubdtype(depth01.dtype, np.integer):
        _flip(depth01)                      # 0..max encoding: max - value
    else:
        np.negative(depth01, out=depth01)
        depth01 += 1
    return depth01
