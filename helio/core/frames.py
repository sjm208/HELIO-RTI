"""Which way is up: where normals meet image rows (contracts.md §3).

Normals and calibrated light vectors are +Y up (north).  Sphere calibration
and .lp files give lights that way, so the solver returns normals that way,
and every shading engine lights them that way.  Images, depth maps and the
mesh grid are indexed by row, and rows run down (south).  A normal's Y and
its surface's slope down the rows therefore have opposite signs:

    dz/dcol = -nx / nz          dz/drow = +ny / nz

Code that crosses between the two uses these helpers, or says why not, so
the sign lives here.  Until FIX-003 every integrator read +Y-up normals as if
Y ran down the rows (KD-009): each round bump came back as a saddle, and the
depth map, mesh and DEM of every calibrated capture were wrong in Y.

Each helper does the same arithmetic as the rows-down code it replaced with
one sign changed, and IEEE negation is exact, so solving with +Y-up lights
now gives bit-for-bit the depth that rows-down lights used to.
"""
from __future__ import annotations

import cv2
import numpy as np

NZ_MIN = 0.001   # floor on nz before dividing: the integrators' long-standing value


def depth_slopes(nx, ny, nz):
    """Slopes (dz/dcol, dz/drow) from +Y-up normal components.

    *nz* must already be floored; callers differ in the floor they use."""
    return -nx / nz, ny / nz


def normal_slopes(normals, nz_min: float = NZ_MIN):
    """Slopes (dz/dcol, dz/drow) of the surface with these +Y-up normals."""
    nz = np.maximum(normals[..., 2], nz_min)
    return depth_slopes(normals[..., 0], normals[..., 1], nz)


def normals_from_slopes(gx, gy, gz):
    """+Y-up unit normals of a surface with slopes gx = dz/dcol, gy = dz/drow.

    *gz* is the vertical scale of the slopes (a scalar or an array)."""
    gz = np.broadcast_to(np.asarray(gz, dtype=gx.dtype), gx.shape)
    length = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return np.stack([-gx / length, gy / length, gz / length], axis=-1)


def normal_divergence(nx, ny, sobel: bool = True):
    """Divergence of +Y-up normals: positive on a dome, negative in a hollow.

    dNx/dx + dNy/dy with y up, i.e. dNx/dcol - dNy/drow.  The old rows-down
    sum, dNx/dcol + dNy/drow, cancels on a round dome and reads a saddle.
    *sobel* picks 3x3 Sobel derivatives (float32); otherwise np.gradient."""
    if sobel:
        return (cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=3)
                - cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=3))
    return np.gradient(nx, axis=1) - np.gradient(ny, axis=0)
