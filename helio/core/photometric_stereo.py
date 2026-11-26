"""
Photometric stereo solvers.

All functions are **pure**: they accept NumPy arrays and return NumPy arrays.
No UI or file I/O is performed here.
"""

import gc
import logging
from typing import Optional

import numpy as np

_log = logging.getLogger(__name__)


def _rotation_taking(a, b):
    """Smallest rotation matrix carrying unit vector *a* onto unit vector *b*."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        # Parallel already, or antiparallel: a 180-degree turn about any
        # perpendicular axis will do.
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            perp = np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))


def photometric_stereo_pca(I_flat):
    """Uncalibrated photometric stereo by rank-3 SVD.

    The factorisation ``I ~ L N`` is only defined up to an invertible 3x3
    transform, and ``np.linalg.svd`` orders its rows by singular value, not by
    geometric axis.  The dominant singular vector of a photometric-stereo
    matrix is essentially mean brightness - which is the *camera-facing*
    direction - so the raw factorisation lands it in row 0, i.e. the X channel.
    Every consumer downstream (relighting, depth integration, the viewer
    exports) reads channel 2 as "towards the camera", so leaving the basis
    unresolved yields normals that are unit length and entirely wrong: on the
    real test capture the raw solve gave a mean normal of (0.75, 0.01, 0.06)
    with only 56% of pixels facing the camera, and N.L came out *negative*, so
    a relit view rendered almost black.

    The rotational half of that ambiguity is resolved here against the prior
    HELIO is built around: the subject is a broadly planar object facing the
    camera, so its mean normal is +Z.  Rotating the mean onto +Z also fixes
    the global sign, since a basis pointing away from the camera has a mean
    that rotates back to front.  *L_pseudo* receives the same rotation, so
    ``L N`` - and therefore the fitted radiance - is unchanged.

    The remaining generalised bas-relief (shear/scale) freedom needs an
    integrability constraint and is *not* resolved here; this engine is
    documented as an approximation for when calibration is impossible.
    """
    U, S, Vt = np.linalg.svd(I_flat, full_matrices=False)
    L_pseudo = U[:, :3] * np.sqrt(S[:3])
    N_pseudo = Vt[:3, :] * np.sqrt(S[:3].reshape(3,1))
    mag = np.linalg.norm(N_pseudo.T, axis=1, keepdims=True)
    normals = np.divide(N_pseudo.T, mag, out=np.zeros_like(N_pseudo.T), where=mag!=0)

    # Orient the arbitrary SVD basis so that "towards the camera" is +Z.
    valid = (mag > 0).ravel()
    if valid.any():
        mean_n = normals[valid].mean(axis=0).astype(np.float64)
        norm = float(np.linalg.norm(mean_n))
        if norm > 1e-9:
            R = _rotation_taking(mean_n / norm, np.array([0.0, 0.0, 1.0]))
            normals = (normals @ R.T).astype(normals.dtype)
            # Keep I ~ L N exact: (L R^T)(R N) == L N.
            L_pseudo = (L_pseudo @ R.T).astype(L_pseudo.dtype)
            # Renormalise: the rotation is orthonormal, but float32 drifts.
            m2 = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = np.divide(normals, m2, out=np.zeros_like(normals), where=m2 > 1e-9)

    del U, S, Vt, N_pseudo, mag
    gc.collect()
    return L_pseudo, normals


def solve_near_field_normals(images, L_mat, h, w, dome_radius_px=None, origin=(0.0, 0.0),
                             positions=None, equal_power=True, falloff=True):
    """Normals under point lights at a finite distance (inverse-square falloff).

    The lights sit on a dome 1.5 image-widths away; each pixel sees each
    light from its own direction, attenuated by distance squared.  The dome
    is centred on *origin* (x, y in pixels from the image's centre, +Y up):
    where the light directions were measured, the chrome ball.  A ball at
    the side of the frame measured the lights from there, not from the
    middle; centred there, the far side of a frame no longer comes out
    tilted (on a binding with the ball at the left, the cover's lean went
    from 11-21 degrees to about 4).  Solved
    per pixel by least squares, in batches: the classic loop solved one
    pixel at a time in Python, which takes minutes on a large capture
    (same normal equations; tests/test_solvers.py checks the two agree).

    *positions* (n, 3), when given, puts each light where it was measured
    (helio.core.light_positions: two balls), in this image's pixels from its
    centre, +Y up, z above the board; a row of nan keeps that light on the
    dome.  With *equal_power* the lamps are taken as equally strong, so one
    nearer the object lights it more (1/d²); otherwise (the frames already
    evened to one strength, by a grey patch, flat areas or semi-calibrated)
    each is 1 where the lights were measured (*origin*), as on the dome.
    
    With *falloff* False the frames' fall-off across the frame has already
    been evened (flat areas, helio.core.flash): only each light's direction
    changes from pixel to pixel, or the fall-off would be taken out twice.
    """
    dome_radius = float(dome_radius_px) if dome_radius_px else max(h, w) * 1.5
    # Pixel positions in the lights' frame: +Y up, so the top row is +h/2
    # (helio.core.frames; this used rows-down y, KD-009).
    ox, oy = (float(v) for v in (origin or (0.0, 0.0)))
    # pixel centres, one pixel apart (linspace(-w/2, w/2, w) spaced them w/(w-1))
    xs = (np.arange(w) - (w - 1) / 2.0) - ox
    ys = ((h - 1) / 2.0 - np.arange(h)) - oy
    S = np.asarray(L_mat, np.float64) * dome_radius                  # (n, 3)
    norm2 = np.full(len(S), dome_radius ** 2)                        # |light - origin|^2: strength 1 there
    if positions is not None:
        P0 = np.asarray(positions, np.float64) - np.array([ox, oy, 0.0])
        ok = np.isfinite(P0).all(axis=1)
        S[ok] = P0[ok]
        d2o = np.einsum("ki,ki->k", P0[ok], P0[ok])
        norm2[ok] = float(np.mean(d2o)) if equal_power else d2o
    I_flat = images.reshape(images.shape[0], -1)
    n_px = h * w
    normals_flat = np.zeros((n_px, 3), dtype=np.float32)
    batch = 65536
    for start in range(0, n_px, batch):
        end = min(start + batch, n_px)
        idx = np.arange(start, end)
        P = np.zeros((end - start, 3))
        P[:, 0] = xs[idx % w]
        P[:, 1] = ys[idx // w]
        Lp = S[:, None, :] - P[None, :, :]                             # (n, b, 3)
        d2 = np.einsum('kbi,kbi->kb', Lp, Lp)
        if falloff:
            Lf = Lp * (norm2[:, None] / (d2 * np.sqrt(d2)))[..., None]     # unit * R^2/d^2
        else:                                   # the fall-off already evened (flat areas): direction only
            Lf = Lp / np.sqrt(d2)[..., None]
        AtA = np.einsum('kbi,kbj->bij', Lf, Lf)
        Atb = np.einsum('kbi,kb->bi', Lf, I_flat[:, start:end].astype(np.float64))
        try:
            G = np.linalg.solve(AtA, Atb[..., None])[..., 0]
        except np.linalg.LinAlgError:
            G = np.stack([np.linalg.lstsq(AtA[j], Atb[j], rcond=None)[0]
                          for j in range(len(AtA))])
        mag = np.linalg.norm(G, axis=1, keepdims=True)
        normals_flat[start:end] = np.divide(G, mag, out=np.zeros_like(G), where=mag > 0)
    del I_flat, S
    gc.collect()
    return normals_flat.reshape(h, w, 3)


def irls_photometric_stereo(L_mat, I_mat, h, w, n_iters=4, sigma=40.0, I_ref=None):
    """Iteratively reweighted least squares.

    *I_ref*, when given, is the stack as the camera encoded it, and the
    first weights (well-exposed mid-tones count most) are judged on it; the
    residual scale *sigma* is then carried into *I_mat*'s units by the ratio
    of the two stacks' medians (helio.core.radiometry, FIX-010)."""
    n_lights, n_pixels = I_mat.shape
    L = L_mat.astype(np.float32)
    I = I_mat.astype(np.float32)
    batch = 131072
    eye3 = np.eye(3, dtype=np.float32) * 1e-6
    ref = I if I_ref is None else np.asarray(I_ref, np.float32)
    weights = np.maximum(np.exp(-((ref - 128.0) ** 2) / (2.0 * sigma ** 2)), 0.01)
    if I_ref is not None:
        sigma = sigma * float(np.median(I)) / max(float(np.median(ref)), 1e-6)
        del ref
    G = np.zeros((n_pixels, 3), dtype=np.float32)
    for iteration in range(n_iters):
        G_new = np.zeros((n_pixels, 3), dtype=np.float32)
        w_new = np.zeros_like(weights) if iteration < n_iters - 1 else None
        for start in range(0, n_pixels, batch):
            end = min(start + batch, n_pixels)
            w_b = weights[:, start:end]
            I_b = I[:, start:end]
            wL = w_b[:, :, np.newaxis] * L[:, np.newaxis, :]
            AtA = np.einsum('kbi,kj->bij', wL, L) + eye3
            Atb = np.einsum('kbi,kb->bi', wL, I_b)
            # Atb[..., None]: NumPy 2 reads a 2-D right-hand side as one
            # (batch, 3) matrix, not a stack of 3-vectors (tests/test_solvers.py).
            G_new[start:end] = np.linalg.solve(AtA, Atb[..., None])[..., 0]
            if iteration < n_iters - 1:
                mag_b = np.linalg.norm(G_new[start:end], axis=1, keepdims=True)
                N_b = np.divide(G_new[start:end], mag_b,
                               out=np.zeros_like(G_new[start:end]), where=mag_b > 1e-9)
                alb_b = mag_b.flatten()
                pred_b = alb_b[np.newaxis, :] * (L @ N_b.T)
                resid_b = np.abs(I_b - pred_b)
                w_new[:, start:end] = np.maximum(
                    1.0 / (1.0 + (resid_b / sigma) ** 2), 0.01)
        if iteration < n_iters - 1:
            weights = w_new
        else:
            G = G_new
    mag = np.linalg.norm(G, axis=1, keepdims=True)
    normals = np.divide(G, mag, out=np.zeros_like(G), where=mag > 1e-9)
    del G, mag, L, I, weights, eye3, G_new, w_new
    gc.collect()
    return normals.reshape(h, w, 3).astype(np.float32)


def _weighted_ls(L_mat, I_mat, weights, h, w):
    """Per-pixel weighted least squares, in memory-safe batches.

    AtA[p] = sum_k w[k,p] L[k] (x) L[k]   and   Atb[p] = sum_k w[k,p] I[k,p] L[k]
    """
    n_imgs, n_pixels = I_mat.shape
    G = np.zeros((n_pixels, 3), dtype=np.float32)
    batch_sz = 32768
    eye3 = np.eye(3, dtype=np.float32) * 1e-6          # numerical safety
    for start in range(0, n_pixels, batch_sz):
        end = min(start + batch_sz, n_pixels)
        w_b = weights[:, start:end]                              # (n_imgs, batch)
        I_b = I_mat[:, start:end]                                # (n_imgs, batch)
        wL = w_b[:, :, np.newaxis] * L_mat[:, np.newaxis, :]     # (n_imgs, batch, 3)
        AtA = np.einsum('kbi,kj->bij', wL, L_mat)                # (batch, 3, 3)
        Atb = np.einsum('kbi,kb->bi', wL, I_b)                   # (batch, 3)
        AtA += eye3
        # Atb[..., None]: NumPy 2 reads a 2-D right-hand side as one
        # (batch, 3) matrix, not a stack of 3-vectors (tests/test_solvers.py).
        G[start:end] = np.linalg.solve(AtA, Atb[..., None])[..., 0]
    mag = np.linalg.norm(G, axis=1, keepdims=True)
    return np.divide(G, mag, out=np.zeros_like(G), where=mag > 1e-9
                     ).reshape(h, w, 3).astype(np.float32)


SHADOW_PER_UNIT = 1.0 / 120.0     # the Shadow threshold setting (5-40, 12 by default) as a share


def shadow_fraction(thresh) -> float:
    """The Shadow threshold setting as the share of a pixel's own bright frames below which a frame is
    shadow there: 12 (the default) is a tenth."""
    return float(thresh) * SHADOW_PER_UNIT


def shadow_rejection_ls(L_mat, I_mat, h, w, thresh, I_ref=None):
    """Least squares that leaves out each pixel's shadowed frames.

    A frame is shadow at a pixel when it is darker than shadow_fraction(*thresh*) of that pixel's own
    bright frames (its 90th percentile), judged on *I_ref* (the values as stored) when the solve runs
    in linear light (FIX-010); a pixel lit in fewer than 3 frames falls back to all of them.

    The cut was a fixed pixel value (*thresh* of 255) until 2026-11-12.  On a glossy or dark object,
    exposed so its highlights do not clip, the ordinary lit frames sit below such a value (a median of
    8 and of 2 on two DiLiGenT-Pi objects): most of them were thrown out and the fit made from the
    brightest, 30 degrees off where least squares is 7 (tools/benchmark_diligent_pi.py).  Relative to the
    pixel, the cut means the same at any exposure and on any albedo."""
    L_mat = L_mat.astype(np.float32)
    I_mat = I_mat.astype(np.float32)
    ref = I_mat if I_ref is None else I_ref
    n = ref.shape[0]
    q = int(round(0.9 * (n - 1)))
    frac = shadow_fraction(thresh)
    shadow_w = np.empty(I_mat.shape, np.float32)
    batch = 262144                                   # the cut made a batch at a time: no second stack in memory
    for start in range(0, I_mat.shape[1], batch):
        end = min(start + batch, I_mat.shape[1])
        r = np.asarray(ref[:, start:end], np.float32)
        bright = np.partition(r, q, axis=0)[q]
        lit = r >= frac * bright[None, :]
        lit[:, lit.sum(axis=0) < 3] = True           # fallback for fully-shadowed pixels
        shadow_w[:, start:end] = lit
    return _weighted_ls(L_mat, I_mat, shadow_w, h, w)


def intensity_weighted_ls(L_mat, I_mat, h, w, sigma=42.0, I_ref=None):
    """Mid-Tone Weighted LS: mid-tones count most.

    Each observation gets a Gaussian weight peaking at 128 and falling to
    ~0.01 at 0 and 255, floored at 0.01 so no pixel's system is singular.
    sigma = 128 / sqrt(2 ln 100) ~ 42.  (Moved verbatim from
    RTIViewerPro._run_processing_body.)
    """
    L_mat = L_mat.astype(np.float32)
    I_mat = I_mat.astype(np.float32)
    ref = I_mat if I_ref is None else np.asarray(I_ref, np.float32)   # FIX-010
    weights = np.exp(-((ref - 128.0) ** 2) / (2.0 * sigma ** 2))
    weights = np.maximum(weights, 0.01)
    return _weighted_ls(L_mat, I_mat, weights, h, w)


def sfs_refinement(normals_ps, master_img_grey, L_mat, albedo_grey=None,
                   n_iters=5, lambda_prior=0.70, step_size=0.06):
    import cv2
    N = normals_ps.copy().astype(np.float32)
    h, w = N.shape[:2]
    I = master_img_grey.astype(np.float32)
    if I.max() > 1.5:
        np.divide(I, 255.0, out=I)
    L_mean = L_mat.astype(np.float32).mean(axis=0)
    norm_L = float(np.linalg.norm(L_mean))
    if norm_L > 1e-8:
        np.divide(L_mean, norm_L, out=L_mean)
    else:
        L_mean = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if albedo_grey is None:
        rho = np.clip(I, 0.05, 1.0)
    else:
        rho = albedo_grey.astype(np.float32)
        if rho.max() > 1.5:
            np.divide(rho, 255.0, out=rho)
        rho = np.clip(rho, 0.05, 1.0)
    N_ps = normals_ps.astype(np.float32)
    for _ in range(n_iters):
        NdotL = np.clip((N * L_mean).sum(axis=-1), 0.0, None)
        I_pred = rho * NdotL
        residual = I - I_pred
        grad = (-2.0 * residual * rho)[..., np.newaxis] * L_mean \
               + 2.0 * lambda_prior * (N - N_ps)
        g_max = float(np.abs(grad).max())
        if g_max > 1e-9:
            grad *= step_size / g_max
        np.subtract(N, grad, out=N)
        mag = np.linalg.norm(N, axis=-1, keepdims=True)
        N = np.divide(N, mag, out=N, where=mag > 1e-9)
        for _ch in range(2):
            N[..., _ch] = cv2.bilateralFilter(
                N[..., _ch], d=5, sigmaColor=0.04, sigmaSpace=3.0)
        mag = np.linalg.norm(N, axis=-1, keepdims=True)
        N = np.divide(N, mag, out=N, where=mag > 1e-9)
    return N.astype(np.float32)


def iterative_specular_separation_ps(L_mat, I_mat, h, w, n_iters=2):
    """Normals for glossy surfaces: highlights found and set aside, then solved.

    The first pass is robust: per pixel, the best of a few random subsets of
    lights by median residual.  Each light then gets a weight: 0.05 where it
    reads brighter than the matte fit by more than three quarters of the
    lights do (a highlight), or the surface faces away from it (shadow),
    else 1.  Every later pass refits by weighted least squares over all the
    lights, (Lᵀ W L) g = Lᵀ W i, and weighs again.

    Until 2026-10-15 the weights were computed and never used: every pass
    redrew random subsets, so passes after the first changed nothing but the
    draw (three passes measured worse than one).  On a synthetic glossy
    surface (Lambert plus Blinn-Phong, 16 lights) the mean error went from
    1.98° to 0.73°; on a matte one it is 0.31° against plain least squares' 0.28°.
    """
    n_lights, n_pixels = I_mat.shape
    L = L_mat.astype(np.float32)
    I = I_mat.astype(np.float32)
    eye3 = np.eye(3, dtype=np.float32) * 1e-6
    LL = (L[:, :, None] * L[:, None, :]).reshape(n_lights, 9)      # each light's L Lᵀ
    batch = 65536
    min_subset = max(3, n_lights // 3)
    rng = np.random.default_rng(42)
    G = np.zeros((n_pixels, 3), dtype=np.float32)
    weights = np.ones((n_lights, n_pixels), dtype=np.float32)
    for iteration in range(max(1, n_iters)):
        G_new = np.zeros((n_pixels, 3), dtype=np.float32)
        w_new = np.zeros((n_lights, n_pixels), dtype=np.float32)
        for start in range(0, n_pixels, batch):
            end = min(start + batch, n_pixels)
            nb = end - start
            I_b = I[:, start:end]
            if iteration == 0:
                best_G = np.zeros((nb, 3), np.float32)
                best_score = np.full(nb, np.inf, np.float32)
                for _ in range(max(4, 2 * n_iters)):
                    idx = rng.choice(n_lights, min_subset, replace=False)
                    L_sub = L[idx]
                    I_sub = I_b[idx]
                    AtA = L_sub.T @ L_sub + eye3
                    Atb = L_sub.T @ I_sub
                    try:
                        G_trial = np.linalg.solve(AtA, Atb).T
                    except np.linalg.LinAlgError:
                        continue
                    mag = np.linalg.norm(G_trial, axis=1, keepdims=True)
                    N_tr = np.divide(G_trial, mag,
                                     out=np.zeros_like(G_trial), where=mag > 1e-9)
                    alb = mag[:, 0]
                    pred = alb[np.newaxis, :] * np.clip(L @ N_tr.T, 0, None)
                    resid = np.median(np.abs(I_b - pred), axis=0)
                    better = resid < best_score
                    best_G[better] = G_trial[better]
                    best_score[better] = resid[better]
            else:
                w_b = weights[:, start:end]
                A = (w_b.T @ LL).reshape(nb, 3, 3) + eye3
                b = (w_b * I_b).T @ L
                best_G = np.linalg.solve(A, b[..., None])[..., 0].astype(np.float32)
            G_new[start:end] = best_G
            mag2 = np.linalg.norm(best_G, axis=1, keepdims=True)
            N_b = np.divide(best_G, mag2,
                            out=np.zeros_like(best_G), where=mag2 > 1e-9)
            facing = L @ N_b.T
            diff2 = mag2[:, 0][np.newaxis, :] * np.clip(facing, 0, None)
            spec2 = np.clip(I_b - diff2, 0, None)
            thresh = np.percentile(spec2, 75, axis=0, keepdims=True)
            w_new[:, start:end] = (np.where(spec2 > thresh, 0.05, 1.0)
                                   * np.where(facing > 0.02, 1.0, 0.05))
        weights = np.maximum(w_new, 0.01)
        G = G_new
    mag = np.linalg.norm(G, axis=1, keepdims=True)
    N = np.divide(G, mag, out=np.zeros_like(G), where=mag > 1e-9)
    return N.reshape(h, w, 3).astype(np.float32)


def generate_lighting_L(light_vecs):
    return np.array(light_vecs, dtype=np.float32)


def _lls_normals_numpy(L_mat, flat):
    """The reference: strip-wise ``lstsq`` then unit length, as the pipeline
    did before helio_core_rs.  (N, 3) in the type NumPy gives the solve."""
    from helio.core.parallel import lstsq_columns
    G = lstsq_columns(L_mat, flat).T
    mag = np.linalg.norm(G, axis=1, keepdims=True)
    return np.divide(G, mag, out=np.zeros_like(G), where=mag != 0)


def _lls_normals_rust(rs, L_mat, flat):
    """helio_core_rs path.  The (tiny) pseudo-inverse is taken here with the
    same rank cut-off ``lstsq`` uses, so rank-deficient light sets agree; the
    per-pixel product and normalisation run in Rust.  The stack goes in as
    float32 without a copy when it already is; otherwise (uint8 captures) it
    is converted a strip at a time so a 46-megapixel stack is never doubled."""
    from helio.core.parallel import parts_for
    L = np.asarray(L_mat, np.float64)
    n, n_px = flat.shape
    P = np.ascontiguousarray(np.linalg.pinv(L, rcond=np.finfo(np.float64).eps * max(L.shape)))
    if flat.dtype == np.float32 and flat.flags.c_contiguous:
        return rs.lls_normals(P, flat)
    out = np.empty((n_px, 3), np.float32)
    step = -(-n_px // parts_for(n_px, n_px * n * 4))
    for a in range(0, n_px, step):
        b = min(a + step, n_px)
        out[a:b] = rs.lls_normals(P, np.ascontiguousarray(flat[:, a:b], dtype=np.float32))
    return out


def linear_least_squares_normals(L_mat, flat):
    """Unit surface normals, (n_pixels, 3), from the Linear Least Squares fit
    of ``flat`` (n_lights, n_pixels) to the lights ``L_mat`` (n_lights, 3).

    Uses helio_core_rs when it is built (float32 result), NumPy otherwise.
    Pixels whose fit is exactly zero get the zero vector either way."""
    from helio.core._native import rs
    L_mat = np.asarray(L_mat)
    if rs is not None and L_mat.ndim == 2 and L_mat.shape == (flat.shape[0], 3):
        return _lls_normals_rust(rs, L_mat, flat)
    return _lls_normals_numpy(L_mat, flat)
