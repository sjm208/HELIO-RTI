"""Radial-basis relighting from the photographs themselves, GUI-free.

PTM and HSH fit each pixel's brightness with a low-order polynomial of the
light direction: right for matte paper, wrong for anything that shines or
glows.  Gold tooling flares in a narrow lobe the polynomial smears into a
dull bloom; vellum, horn and wax scatter light the normal map cannot
describe.  Radial-basis interpolation (Pintus, Dulecha, Ciortan, Gobbetti
and Giachetti, "Objective and subjective evaluation of virtual relighting
from reflectance transformation imaging data", GCH 2018; the RBF mode of
Relight/OpenLIME) relights by interpolating the photographs themselves
across the light directions, so a highlight is where it was photographed and
as sharp, and a light between two frames gives an image between them.

The interpolation is linear in the photographs: a new light *l* gives
weights ``w(l)`` over the frames, one set for the whole picture, and the
relit picture is the weighted sum.  The frames are kept compressed as a
mean and a few principal components (``build``), so a relit frame costs a
dozen planes, not every photograph.

The kernel is a Gaussian of the distance between light directions, its
width a multiple of the lights' typical spacing, with a linear polynomial
term so a light outside the dome extrapolates smoothly instead of fading to
black.  *smooth* regularises the fit: 0 passes through every photograph
exactly, higher trades fidelity for a smoother response between lights.
"""
from __future__ import annotations

import cv2
import numpy as np

COMPONENTS = 16                       # principal components kept (all frames when fewer)
FULL_LIMIT_BYTES = 400 * 1024 ** 2    # the full-size planes are kept only below this


def _pca(X: np.ndarray, k: int):
    """Mean (P,) and (U (n, k), C (k, P)) with X - mean ~ U @ C, from the n x n Gram matrix."""
    mean = X.mean(axis=0)
    Xc = X - mean
    G = Xc @ Xc.T
    vals, vecs = np.linalg.eigh(G.astype(np.float64))
    order = np.argsort(vals)[::-1][:k]
    U = vecs[:, order].astype(np.float32)
    return mean.astype(np.float32), U


def build(stack: np.ndarray, lights: np.ndarray, proxy_hw: tuple[int, int], keep_full: bool = True,
          k: int = COMPONENTS) -> dict | None:
    """The compressed frames for relighting.

    *stack* is the (n, h, w) grey stack as stored (what the eye sees, not the
    solve's linear light), *lights* (n, 3) +Y up.  The components are found on
    the preview-size frames and applied to the full-size ones, a band of rows
    at a time, when *keep_full* and they fit in FULL_LIMIT_BYTES."""
    n = len(stack)
    L = np.asarray(lights, np.float32).reshape(-1, 3)
    if n < 4 or len(L) != n:
        return None
    h, w = stack.shape[1:3]
    ph, pw = proxy_hw
    k = int(min(k, n))
    small = np.stack([cv2.resize(np.ascontiguousarray(f, np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
                      for f in stack]).reshape(n, -1)
    mean_p, U = _pca(small, k)
    comps_p = (U.T @ (small - mean_p)).astype(np.float16).reshape(k, ph, pw)
    out = dict(lights=L, U=U, mean_proxy=mean_p.reshape(ph, pw), comps_proxy=comps_p,
               mean_full=None, comps_full=None, h=h, w=w)
    if keep_full and (h, w) != (ph, pw) and k * h * w * 2 <= FULL_LIMIT_BYTES:
        mean_f = np.empty((h, w), np.float32)
        comps_f = np.empty((k, h, w), np.float16)
        step = max(1, (8 << 20) // max(1, n * w * 4))
        for y0 in range(0, h, step):
            y1 = min(h, y0 + step)
            band = stack[:, y0:y1].astype(np.float32).reshape(n, -1)
            m = band.mean(axis=0)
            mean_f[y0:y1] = m.reshape(y1 - y0, w)
            comps_f[:, y0:y1] = (U.T @ (band - m)).reshape(k, y1 - y0, w)
        out.update(mean_full=mean_f, comps_full=comps_f)
    elif (h, w) == (ph, pw):              # the preview is the full size: the same planes, no copy
        out.update(mean_full=out["mean_proxy"], comps_full=comps_p)
    return out


def spacing(lights: np.ndarray) -> float:
    """The lights' typical spacing: the median distance to the nearest other light."""
    L = np.asarray(lights, np.float64)
    d = np.linalg.norm(L[:, None, :] - L[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def clamp_to_dome(lights: np.ndarray, light) -> np.ndarray:
    """*light* (unit, +Y up) held to the elevations the lamps were at: the
    photographs say nothing about a light higher or lower than any of them,
    and interpolation there only averages them into a flat picture."""
    L = np.asarray(lights, np.float64).reshape(-1, 3)
    l = np.asarray(light, np.float64).reshape(3)
    l = l / max(np.linalg.norm(l), 1e-12)
    els = np.arcsin(np.clip(L[:, 2] / np.maximum(np.linalg.norm(L, axis=1), 1e-12), -1, 1))
    el = float(np.arcsin(np.clip(l[2], -1, 1)))
    lo, hi = float(els.min()), float(els.max())
    if lo - 1e-6 <= el <= hi + 1e-6:
        return l
    el = min(max(el, lo), hi)
    h = float(np.hypot(l[0], l[1]))
    if h < 1e-9:                                   # straight overhead: no azimuth of its own
        k = int(np.argmax(L[:, 2]))
        h_dir = L[k, :2] / max(float(np.hypot(*L[k, :2])), 1e-12)
    else:
        h_dir = l[:2] / h
    return np.array([np.cos(el) * h_dir[0], np.cos(el) * h_dir[1], np.sin(el)])


def weights(lights: np.ndarray, light: np.ndarray, width: float = 1.2, smooth: float = 0.05) -> np.ndarray:
    """The frames' weights (n,) for a light *light* (3,), +Y up.

    *width*: the kernel's width in light spacings.  *smooth*: the
    regularisation, 0 = through every photograph."""
    L = np.asarray(lights, np.float64).reshape(-1, 3)
    n = len(L)
    sigma = max(1e-3, float(width) * max(spacing(L), 1e-3))
    l = clamp_to_dome(L, light)
    Phi = np.exp(-(np.linalg.norm(L[:, None] - L[None], axis=-1) / sigma) ** 2)
    phi = np.exp(-(np.linalg.norm(L - l, axis=-1) / sigma) ** 2)
    P = np.hstack([np.ones((n, 1)), L]) if n >= 5 else np.ones((n, 1))
    q = P.shape[1]
    M = np.zeros((n + q, n + q))
    M[:n, :n] = Phi + float(smooth) * np.eye(n)
    M[:n, n:] = P
    M[n:, :n] = P.T
    rhs = np.concatenate([phi, [1.0, *l][:q]])
    sol = np.linalg.lstsq(M, rhs, rcond=None)[0]
    return sol[:n].astype(np.float32)


def relight(rbf: dict, light, width: float = 1.2, smooth: float = 0.05, full: bool = False,
            box=None) -> np.ndarray:
    """The grey picture (h, w) float32, 0-255 scale, lit from *light* (+Y up).

    *full*: the full-size planes (when kept); *box* (x1, y1, x2, y2) in the
    planes' own pixels cuts them first."""
    wts = weights(rbf["lights"], light, width, smooth)
    a = (wts @ rbf["U"]).astype(np.float32)               # (k,)
    mean = rbf["mean_full"] if full else rbf["mean_proxy"]
    comps = rbf["comps_full"] if full else rbf["comps_proxy"]
    if box is not None:
        x1, y1, x2, y2 = box
        mean, comps = mean[y1:y2, x1:x2], comps[:, y1:y2, x1:x2]
    out = mean * np.float32(wts.sum())
    for c in range(len(a)):
        out += a[c] * comps[c].astype(np.float32)
    return np.maximum(out, 0.0)


def crop(rbf: dict | None, box, h: int, w: int) -> dict | None:
    """*rbf* cut to *box* (x1, y1, x2, y2) of the full (h, w) solve, for a
    cropped result: the full planes sliced, the preview planes made from them
    (or cut from the old preview when the full ones were not kept)."""
    if rbf is None:
        return None
    x1, y1, x2, y2 = box
    out = dict(rbf, h=y2 - y1, w=x2 - x1)
    from helio.application.pipeline import proxy_dims
    ph, pw = proxy_dims(y2 - y1, x2 - x1)
    if rbf.get("comps_full") is not None:
        mf, cf = rbf["mean_full"][y1:y2, x1:x2], rbf["comps_full"][:, y1:y2, x1:x2]
        out.update(mean_full=mf, comps_full=cf)
        src_m, src_c = mf, cf
    else:
        oph, opw = rbf["mean_proxy"].shape
        sy, sx = oph / float(h), opw / float(w)
        a, b = int(y1 * sy), max(int(y1 * sy) + 1, int(round(y2 * sy)))
        c, d = int(x1 * sx), max(int(x1 * sx) + 1, int(round(x2 * sx)))
        src_m, src_c = rbf["mean_proxy"][a:b, c:d], rbf["comps_proxy"][:, a:b, c:d]
    out["mean_proxy"] = cv2.resize(np.ascontiguousarray(src_m, np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
    out["comps_proxy"] = np.stack([cv2.resize(np.ascontiguousarray(p, np.float32), (pw, ph),
                                              interpolation=cv2.INTER_AREA) for p in src_c]).astype(np.float16)
    return out
