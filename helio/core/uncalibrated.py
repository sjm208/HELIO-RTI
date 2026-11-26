"""Photometric stereo with no ball: the lights found from the photographs.

The brightness of every pixel under every light, I (n lights x P pixels), is
close to rank 3: I ~ L B, with L the lights (scaled by strength) and B the
normals (scaled by albedo).  A rank-3 SVD finds L and B only up to an unknown
3x3 matrix A, since (L A^-1)(A B) gives the same photographs.  The old solve
(photometric_stereo.photometric_stereo_pca) only turned the result to face the
camera, so the relief could come out leaning, stretched or flattened.

Three things narrow A down, each from the photographs alone:

1. **Equal lamps** (Hayakawa 1994).  With one lamp moved round, or a dome of
   like LEDs, every light is about as strong.  Asking that every row of L A^-1
   be one long fixes all of A but a rotation (and a mirror).
2. **A real surface** (integrability).  Normals that come from a surface have
   no curl: the slope across, differentiated down, equals the slope down,
   differentiated across.  Turning the normals about the camera's axis, or
   mirroring them, puts curl in, so the turn with the least curl is taken.
   This leaves one ambiguity no shading can settle: the surface turned half
   round is the same surface inside out (raised for sunken), lit from the
   opposite side.
3. **Raised or sunken.**  A cast shadow lies on the side away from the light,
   behind something higher.  Each solution's height map is checked against
   the dark pixels that face their light yet are dark, and the one whose
   heights explain more of them is taken.  With too few cast shadows to
   decide, the object is taken to stand up from its surround.

When the photographs do not bear the lamps out - the strengths it finds
still differ by more than SPREAD_MAX, or every light comes out on one side
of the object (LEAN_MAX), as a hand-held flash at changing distances or a
flat object with few frames can give - the result is not trusted, and the
rough solve (the old one, facing the camera, good for flat objects but not
to scale) is used instead and said to be rough.

On DiLiGenT with the lamps evened (ten objects, 96 lights), the mean error
was 15.3 degrees against 15.5 for least squares with the true lights and 48.4
for the rough solve; with the lamps 10% uneven 15.6, 20% uneven 19.7
(tools/benchmark_no_ball.py, docs/BENCHMARK_DILIGENT.md).

What is left is the lights' overall strength (irrelevant to the normals) and
the accuracy of the equal-lamp assumption: lamps that differ by 20% leave a
few degrees of lean.  Measurement still needs a ball (see QUICKSTART).
"""
from __future__ import annotations

import numpy as np

__all__ = ["solve_no_ball", "SPREAD_MAX", "LEAN_MAX"]

SPREAD_MAX = 0.2   # the strengths found may still differ by this much (sd / mean)
LEAN_MAX = 0.75    # how far to one side the lights may lie (their mean direction across, 0-1)


def _rot_z(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotation_taking(a, b):
    from helio.core.photometric_stereo import _rotation_taking as rt
    return rt(np.asarray(a, np.float64), np.asarray(b, np.float64))


def equal_lamps(L0: np.ndarray):
    """M with every row of L0 @ M one long (C = M M^T, solved linearly for its
    six entries), or None when the lights do not pin it down (fewer than six,
    or a C that is not positive definite even after a mild repair)."""
    if len(L0) < 6:
        return None
    a, b, c = L0[:, 0], L0[:, 1], L0[:, 2]
    D = np.stack([a * a, b * b, c * c, 2 * a * b, 2 * a * c, 2 * b * c], 1)
    x, *_ = np.linalg.lstsq(D, np.ones(len(L0)), rcond=None)
    C = np.array([[x[0], x[3], x[4]], [x[3], x[1], x[5]], [x[4], x[5], x[2]]])
    w, V = np.linalg.eigh(C)
    if w[-1] <= 0:
        return None
    w = np.maximum(w, w[-1] * 1e-3)
    C = (V * w) @ V.T
    return np.linalg.cholesky(C)


def _slopes(n, keep):
    """Slopes p = dz/dx (x right), q = dz/dy (y up) where the normal faces the camera."""
    nz = np.where(keep, n[..., 2], 1.0)
    return -n[..., 0] / nz, -n[..., 1] / nz


def curl(n: np.ndarray, keep: np.ndarray) -> float:
    """Median |curl| of the slope field (per pixel), over pixels whose four
    neighbours are kept: 0 for normals that come from a surface."""
    p, q = _slopes(n, keep)
    dp_dy = p[:-1, :-1] - p[1:, :-1]          # y up: the row above minus this row
    dq_dx = q[:-1, 1:] - q[:-1, :-1]
    ok = keep[:-1, :-1] & keep[1:, :-1] & keep[:-1, 1:]
    if ok.sum() < 50:
        return float("inf")
    return float(np.median(np.abs(dp_dy - dq_dx)[ok]))


def _small(a, step):
    return a[::step, ::step]


def best_turn(B: np.ndarray, keep: np.ndarray, h: int, w: int):
    """The rotation about the camera's axis (and mirror) whose normals have the
    least curl: (T, score, runner-up score).  T and T turned 180 degrees score
    the same (the surface inside out); T is the one in [0, 180)."""
    step = max(1, int(round(max(h, w) / 300)))
    n = (B / np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-12)).T.reshape(h, w, 3)
    n, k = _small(n, step), _small(keep, step)
    k = k & (n[..., 2] > 0.2)
    best = []
    for mirror in (False, True):
        Mi = np.diag([-1.0, 1.0, 1.0]) if mirror else np.eye(3)
        for deg in np.arange(0.0, 180.0, 2.0):
            T = _rot_z(deg) @ Mi
            best.append((curl(n @ T.T, k), deg, mirror))
    best.sort()
    s0, deg0, mir0 = best[0]
    Mi = np.diag([-1.0, 1.0, 1.0]) if mir0 else np.eye(3)
    fine = [(curl(n @ (_rot_z(d) @ Mi).T, k), d) for d in np.arange(deg0 - 2.0, deg0 + 2.01, 0.25)]
    s, d = min(fine)
    other = min(b[0] for b in best if b[2] != mir0 or abs(((b[1] - deg0 + 90) % 180) - 90) > 20)
    return _rot_z(d) @ Mi, s, other


def _heights(n: np.ndarray, keep: np.ndarray) -> np.ndarray:
    from helio.core.depth_integration import poisson_depth_fft
    nn = np.where(keep[..., None], n, np.array([0.0, 0.0, 1.0]))
    return np.asarray(poisson_depth_fft(nn.astype(np.float32)), np.float64)


def cast_shadow_vote(I: np.ndarray, n: np.ndarray, L: np.ndarray, z: np.ndarray,
                     keep: np.ndarray, reach: int = 40):
    """How many cast shadows height map *z* (with lights *L*, normals *n*)
    explains, and the same for the surface inside out (-z, lights mirrored
    through the camera's axis): (explained, explained inside out, shadows).

    A cast shadow here is a pixel that faces its light (n.l > 0.3) yet is
    much darker under it than its own typical brightness.  It is explained
    when, walking from it towards the light across the image, the height
    rises above the line climbing at the light's elevation."""
    h, w = z.shape
    nl = np.einsum("hwc,nc->nhw", n, L)
    Ip = I.reshape(len(L), h, w)
    typical = np.median(Ip, axis=0)
    yes = [0, 0]
    total = 0
    rng = np.random.default_rng(0)
    for i in range(len(L)):
        lx, ly, lz = L[i] / max(np.linalg.norm(L[i]), 1e-12)
        flat = np.hypot(lx, ly)
        if flat < 0.15 or lz <= 0.05:              # a light from straight above casts little
            continue
        dark = keep & (nl[i] > 0.3) & (Ip[i] < 0.25 * typical) & (typical > 0)
        ys, xs = np.nonzero(dark)
        if len(ys) == 0:
            continue
        if len(ys) > 4000:
            pick = rng.choice(len(ys), 4000, replace=False)
            ys, xs = ys[pick], xs[pick]
        total += len(ys)
        rise = lz / flat                            # height gained per pixel towards the light
        for side, sgn in ((0, 1.0), (1, -1.0)):     # as solved; inside out
            dx, dy = sgn * lx / flat, sgn * ly / flat
            zz = sgn * z
            hit = np.zeros(len(ys), bool)
            for k in range(1, reach + 1):
                c = np.round(xs + k * dx).astype(int)
                r = np.round(ys - k * dy).astype(int)  # y up: rows go the other way
                inside = (c >= 0) & (c < w) & (r >= 0) & (r < h)
                cc, rr = np.clip(c, 0, w - 1), np.clip(r, 0, h - 1)
                hit |= inside & (zz[rr, cc] - zz[ys, xs] > k * rise)
            yes[side] += int(hit.sum())
    return yes[0], yes[1], total


def stands_up(z: np.ndarray, keep: np.ndarray) -> float:
    """How far the object stands above its edge (the median height inside
    minus the median round the border of what is kept), in pixels."""
    k = keep.copy()
    edge = k & ~(np.roll(k, 1, 0) & np.roll(k, -1, 0) & np.roll(k, 1, 1) & np.roll(k, -1, 1))
    edge[0, :] |= k[0, :]
    edge[-1, :] |= k[-1, :]
    edge[:, 0] |= k[:, 0]
    edge[:, -1] |= k[:, -1]
    if edge.sum() < 10 or k.sum() < 100:
        return 0.0
    return float(np.median(z[k]) - np.median(z[edge]))


WORK_SIDE = 600     # the lights are worked out on frames this long at most


def _shrink(I: np.ndarray, h: int, w: int, keep: np.ndarray):
    """The frames (n, h*w) and the kept pixels at most WORK_SIDE long (area
    averaging): the lights come out the same, in a fraction of the time."""
    import cv2
    k = min(1.0, WORK_SIDE / float(max(h, w)))
    if k >= 1.0:
        return np.asarray(I, np.float64), keep, h, w
    hs, ws = max(8, int(round(h * k))), max(8, int(round(w * k)))
    out = np.stack([cv2.resize(np.asarray(f, np.float32).reshape(h, w), (ws, hs),
                               interpolation=cv2.INTER_AREA) for f in I]).reshape(len(I), -1)
    ks = cv2.resize(keep.astype(np.float32), (ws, hs), interpolation=cv2.INTER_AREA) > 0.99
    return out.astype(np.float64), ks, hs, ws


def _kept(I: np.ndarray, h: int, w: int, mask) -> np.ndarray:
    keep = np.ones((h, w), bool) if mask is None else np.asarray(mask, bool).reshape(h, w)
    if not keep.any():
        return keep
    lit = np.asarray(I, np.float32).mean(axis=0).reshape(h, w)
    return keep & (lit > np.percentile(lit[keep], 5) * 0.5)


def solve_no_ball(I_flat: np.ndarray, h: int, w: int, mask: np.ndarray | None = None):
    """Normals and lights from photographs with no ball.

    *I_flat* (n, h*w), the frames' brightness in linear light (as for the
    other solvers); *mask* (h, w), the object, when it is known.  Returns
    (L (n, 3), normals (h*w, 3), info).  *info* says how it went: "used" is
    "found", or "rough" with "why" (the rough solve, when the photographs did
    not bear the lamps out); "spread", how much the strengths found still
    differ (0.1: 10%); "lean", how far to one side the lights lie; "raised",
    "shadows" or "surround", what settled raised against sunken, with the
    shadow "votes"; "curl" and "curl_other", the chosen turn's curl and the
    next best's.

    The lights are worked out on frames at most WORK_SIDE long; the normals
    are then made at full size in strips, from the one 3x3 answer."""
    keep = _kept(I_flat, h, w, mask)
    info = {"used": "", "why": "", "equal_lamps": False, "curl": None, "curl_other": None,
            "raised": "", "votes": None, "spread": None, "lean": None}
    I, ks, hs, ws = _shrink(I_flat, h, w, keep)
    U, S, _Vt = np.linalg.svd(I[:, ks.ravel()], full_matrices=False)
    L0 = U[:, :3] * np.sqrt(S[:3])
    P0 = np.linalg.pinv(L0)                       # B = P0 I, for any pixel
    M = equal_lamps(L0)
    if M is None:
        info["why"] = "too few photographs to find the lights (six or more are needed)"
        return _rough(I_flat, keep, info)
    info["equal_lamps"] = True
    A = np.linalg.inv(M)                          # B1 = A B0, L1 = L0 A^-1
    B1 = A @ (P0 @ I)
    # face the camera: the mean normal (over what is kept) to +Z
    nb = B1 / np.maximum(np.linalg.norm(B1, axis=0, keepdims=True), 1e-12)
    mean = nb[:, ks.ravel()].mean(axis=1)
    if np.linalg.norm(mean) > 1e-9:
        A = _rotation_taking(mean / np.linalg.norm(mean), [0.0, 0.0, 1.0]) @ A
    T, s, other = best_turn(A @ (P0 @ I), ks, hs, ws)
    A = T @ A
    info["curl"], info["curl_other"] = s, other
    B1 = A @ (P0 @ I)
    L1 = L0 @ np.linalg.inv(A)
    n = (B1 / np.maximum(np.linalg.norm(B1, axis=0, keepdims=True), 1e-12)).T.reshape(hs, ws, 3)
    # raised or sunken
    z = _heights(n, ks)
    a, b, tot = cast_shadow_vote(I, n, L1, z, ks)
    info["votes"] = (a, b, tot)
    flip = False
    # Enough explained one way, and clearly more than the other: a small
    # object casts few shadows, but a few dozen that all agree settle it.
    if max(a, b) >= max(25, 0.0002 * hs * ws) and abs(a - b) > 0.1 * max(a, b):
        flip, info["raised"] = b > a, "shadows"
    else:
        up = stands_up(z, ks)
        if abs(up) > 1e-6:
            flip, info["raised"] = up < 0, "surround"
    if flip:
        A = _rot_z(180.0) @ A
        L1 = L0 @ np.linalg.inv(A)
    strengths = np.linalg.norm(L1, axis=1)
    info["spread"] = float(np.std(strengths) / max(np.mean(strengths), 1e-12))
    Lu = L1 / np.maximum(np.linalg.norm(L1, axis=1, keepdims=True), 1e-12)
    across = Lu[:, :2] / np.maximum(np.linalg.norm(Lu[:, :2], axis=1, keepdims=True), 1e-12)
    info["lean"] = float(np.linalg.norm(across.mean(axis=0)))
    if info["spread"] > SPREAD_MAX:
        info["why"] = (f"the lamps would have to differ by {info['spread'] * 100:.0f}% for this "
                       "answer: they were probably not equally strong")
        return _rough(I_flat, keep, info)
    if info["lean"] > LEAN_MAX:
        info["why"] = "the lights it found all lie on one side, which the photographs do not bear out"
        return _rough(I_flat, keep, info)
    info["used"] = "found"
    Q = (A @ P0).astype(np.float32)               # full-size normals, a strip at a time
    src = np.asarray(I_flat)
    N = np.empty((h * w, 3), np.float32)
    step = 1 << 20
    for i in range(0, h * w, step):
        b = Q @ np.asarray(src[:, i:i + step], np.float32)
        N[i:i + step] = (b / np.maximum(np.linalg.norm(b, axis=0, keepdims=True), 1e-12)).T
    N[~keep.ravel()] = (0.0, 0.0, 1.0)              # the surround: no light reached it to read
    return Lu.astype(np.float32), N, info


def _rough(I_flat, keep, info):
    """The rough solve (photometric_stereo_pca): facing the camera, the rest
    of the ambiguity left as it falls."""
    from helio.core.photometric_stereo import photometric_stereo_pca
    L, N = photometric_stereo_pca(np.asarray(I_flat, np.float32))
    N = np.asarray(N, np.float32)
    N[~keep.ravel()] = (0.0, 0.0, 1.0)
    info["used"] = "rough"
    return np.asarray(L, np.float32), N, info
