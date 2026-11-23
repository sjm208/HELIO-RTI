"""Least-squares and bilateral normal integration, GUI-free (REC-020, REC-021).

Two depth methods from the recent literature, for when the depth is to be
measured rather than looked at.

**Least squares, free boundary.**  Every pixel's depth is fitted to its
neighbours' through the slope the normal gives, with forward and backward
differences, and nothing assumed at the frame edge.  The FFT and DCT
integrators solve the same equations under an assumed boundary (periodic,
or zero slope across the edge), which bends the surface near the edges
(Quéau, Durou and Aujol, "Normal Integration: a Survey", JMIV 2018, and
"Variational Methods for Normal Integration", JMIV 2018).

**Bilateral (keeps steps).**  Cao, Santo, Shi, Okura and Matsushita,
"Bilateral Normal Integration", ECCV 2022.  The surface is taken as only
one-sidedly smooth: at each pixel the forward and backward slope equations
are weighted by which side the depth changes less on, re-estimated
iteratively, so a step - a coin's rim, a page edge, a deep cut - stays a
step instead of being smeared into a ramp that tilts the flat ground beside
it.  The first iteration, with equal weights, is the least-squares method.

Measured on a synthetic raised disc with a steep, noisy bevel
(tests/test_bini.py): background flatness error 0.06 px against 2.0 px for the
DCT and FFT integrators, relief height 42 against 49 (true 40).

Frame (helio.core.frames, contracts.md §3): normals +Y up, depth indexed
by rows, so dz/dcol = -nx/nz and dz/drow = +ny/nz.  Each equation is
multiplied through by nz, as in the paper, so steep pixels do not dominate.

Large maps are solved at up to *max_px* pixels and the fine detail above
that scale added back from the full-resolution free-boundary DCT solution
(helio.core.depth_integration.poisson_free_boundary), so a
100 MP capture integrates in about a minute rather than an hour.
"""
from __future__ import annotations

import math

import numpy as np


def _difference_rows(h: int, w: int, nz: np.ndarray):
    """The four nz-weighted difference operators (forward/backward along
    columns, then rows) and where each has a neighbour."""
    import scipy.sparse as sp
    N = h * w
    idx = np.arange(N).reshape(h, w)
    nzf = nz.ravel()

    def op(a, b, valid):
        """Rows i (at valid pixels): nz_i * (z[b_i] - z[a_i])."""
        i = idx[valid]
        rows = np.r_[i, i]
        cols = np.r_[b[valid], a[valid]]
        vals = np.r_[nzf[i], -nzf[i]]
        return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    right = np.zeros((h, w), bool); right[:, :-1] = True
    left = np.zeros((h, w), bool); left[:, 1:] = True
    down = np.zeros((h, w), bool); down[:-1, :] = True
    up = np.zeros((h, w), bool); up[1:, :] = True
    nxt_c = np.zeros_like(idx); nxt_c[:, :-1] = idx[:, 1:]
    prv_c = np.zeros_like(idx); prv_c[:, 1:] = idx[:, :-1]
    nxt_r = np.zeros_like(idx); nxt_r[:-1, :] = idx[1:, :]
    prv_r = np.zeros_like(idx); prv_r[1:, :] = idx[:-1, :]
    A1 = op(idx, nxt_c, right)        # z[c+1] - z[c]
    A2 = op(prv_c, idx, left)         # z[c] - z[c-1]
    A3 = op(idx, nxt_r, down)         # z[r+1] - z[r]
    A4 = op(prv_r, idx, up)           # z[r] - z[r-1]
    return (A1, A2, A3, A4), tuple(m.ravel() for m in (right, left, down, up))


def _solve_full(normals, bilateral: bool, k: float, max_iter: int, tol: float,
                x0=None, check=None):
    import scipy.sparse as sp
    from scipy.sparse.linalg import cg
    n = np.asarray(normals, np.float64)
    h, w = n.shape[:2]
    nx, ny = n[..., 0], n[..., 1]
    nz = np.maximum(n[..., 2], 1e-3)
    (A1, A2, A3, A4), (m1, m2, m3, m4) = _difference_rows(h, w, nz)
    b = np.r_[np.where(m1, -nx.ravel(), 0.0), np.where(m2, -nx.ravel(), 0.0),
              np.where(m3, ny.ravel(), 0.0), np.where(m4, ny.ravel(), 0.0)]
    A = sp.vstack([A1, A2, A3, A4]).tocsr()
    N = h * w
    wts = np.full(4 * N, 0.5)
    z = np.zeros(N) if x0 is None else np.asarray(x0, np.float64).ravel() - float(np.mean(x0))
    reg = sp.identity(N, format="csr") * 1e-9        # pins the free constant

    def energy(z, wts):
        r = A @ z - b
        return float(r @ (wts * r))

    E = energy(z, wts)
    for _ in range(max_iter if bilateral else 1):
        if check:
            check()
        AtW = A.T.multiply(wts).tocsr()
        M = (AtW @ A + reg).tocsr()
        rhs = AtW @ b
        D = sp.diags(1.0 / np.clip(M.diagonal(), 1e-8, None))
        z, _info = cg(M, rhs, x0=z, M=D, maxiter=4000, rtol=1e-5)
        if not bilateral:
            break
        a1, a2, a3, a4 = A1 @ z, A2 @ z, A3 @ z, A4 @ z
        wu = 1.0 / (1.0 + np.exp(-k * (a2 ** 2 - a1 ** 2)))
        wv = 1.0 / (1.0 + np.exp(-k * (a4 ** 2 - a3 ** 2)))
        wts = np.r_[wu, 1.0 - wu, wv, 1.0 - wv]
        E_old, E = E, energy(z, wts)
        if abs(E - E_old) / max(E_old, 1e-12) < tol:
            break
    z = z.reshape(h, w)
    return (z - z.mean()).astype(np.float32)


def integrate(normals, bilateral: bool = True, *, k: float = 2.0, max_iter: int = 60,
              tol: float = 1e-4, max_px: int = 1_500_000, check=None) -> np.ndarray:
    """Depth from +Y-up unit normals (h, w, 3), in pixel units, mean zero.

    *bilateral* False is the plain least-squares (free boundary) solution.
    Above *max_px* pixels the solve runs on a reduced map and the detail
    finer than that scale comes from the full-resolution free-boundary DCT solution."""
    import cv2
    from helio.core.depth_integration import poisson_free_boundary
    from helio.core.frames import normal_slopes
    n = np.asarray(normals, np.float32)
    h, w = n.shape[:2]
    # The start, and above max_px the fine detail: the unweighted free-boundary
    # least-squares surface, exact in one DCT.  (Until 2026-11-10 both came from
    # the mirrored-edge integrators: the start was far from the answer wherever
    # the edge slopes, 10 s of CG against 0.2 s on a tilted 0.5 MP map, and the
    # detail carried their bend at the edges.)
    if h * w <= max_px:
        x0 = poisson_free_boundary(*normal_slopes(n))
        return _solve_full(n, bilateral, k, max_iter, tol, x0=x0, check=check)
    s = math.sqrt(max_px / float(h * w))
    sh, sw = max(8, int(round(h * s))), max(8, int(round(w * s)))
    small = cv2.resize(n, (sw, sh), interpolation=cv2.INTER_AREA)
    small /= np.maximum(np.linalg.norm(small, axis=-1, keepdims=True), 1e-8)
    z_small = _solve_full(small, bilateral, k, max_iter, tol,
                          x0=poisson_free_boundary(*normal_slopes(small)), check=check)
    # Depth is in pixels of the grid it was solved on: back to full-size pixels.
    coarse = cv2.resize(z_small * (w / float(sw)), (w, h), interpolation=cv2.INTER_CUBIC)
    zp = poisson_free_boundary(*normal_slopes(n))
    zp_low = cv2.resize(cv2.resize(zp, (sw, sh), interpolation=cv2.INTER_AREA), (w, h),
                        interpolation=cv2.INTER_CUBIC)
    z = coarse + (zp - zp_low)
    return (z - z.mean()).astype(np.float32)
