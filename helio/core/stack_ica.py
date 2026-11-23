"""Independent-component analysis of the light stack (the forensic ICA views):
PCA through the Gram matrix, whitening, FastICA, and a sign convention.

Moved out of helio.core.processing (November 2026), which still re-exports every name.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np


_log = logging.getLogger(__name__)



def skew_sign(values: np.ndarray, loadings: np.ndarray | None = None) -> float:
    """+1 or -1: which way up a principal or independent component is shown.

    A component is only defined up to its sign: the maths library may return it or its
    negative, and the same photographs then give a picture and its photographic negative.
    Fixed here so that the same data always gives the same picture: the rare, strongly
    different pixels (a mark, a highlight, a stain) are the bright ones, on a darker
    ground, i.e. the component's skewness is positive.  Measured on a fixed-stride sample
    (up to 2 million pixels), so it is cheap and the same on every machine.  A symmetric
    component (skewness ~ 0) is oriented so that its largest loading is positive instead.
    """
    v = np.asarray(values).reshape(-1)
    step = max(1, v.size // 2_000_000)
    x = v[::step].astype(np.float64)
    sd = x.std()
    if sd > 0:
        sk = float((((x - x.mean()) / sd) ** 3).mean())
        if abs(sk) > 1e-3:
            return 1.0 if sk > 0 else -1.0
    if loadings is not None and np.size(loadings):
        a = np.asarray(loadings).reshape(-1)
        return 1.0 if a[int(np.argmax(np.abs(a)))] >= 0 else -1.0
    return 1.0



def pca_via_gram(I_c: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA on a (n_rows, n_pixels) matrix via its (n_rows x n_rows) Gram
    matrix instead of a full SVD — same speedup rationale as
    :func:`whiten_pca_gram`, but returns raw (non-whitened) principal
    component rows and their explained-variance fractions, matching
    ``np.linalg.svd(I_c, full_matrices=False)``'s ``(Vt, S)`` contract
    (verified to match full SVD to ~1e-7 on synthetic data).

    Returns ``(components, variance)`` where ``components`` has shape
    (n_components, n_pixels) and ``variance`` are the fractions of total
    variance explained, descending.
    """
    gram = I_c @ I_c.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    total_var = float(np.sum(np.maximum(eigvals, 0)))
    order = np.argsort(eigvals)[::-1][:n_components]
    ev_k = np.maximum(eigvals[order], 1e-12)
    U_k = eigvecs[:, order]
    S_k = np.sqrt(ev_k)
    components = ((U_k.T @ I_c) / S_k[:, None]).astype(np.float32)
    variance = ev_k / max(total_var, 1e-12)
    return components, variance



def whiten_pca_gram(I_c: np.ndarray, n_components: int) -> np.ndarray:
    """PCA-whiten a (n_rows, n_pixels) matrix via its (n_rows x n_rows) Gram
    matrix instead of a full SVD.

    ``np.linalg.svd`` on a wide matrix (a handful of rows, tens/hundreds of
    millions of columns) does not exploit the small row count — on a
    16 x 100M matrix a full economy SVD takes ~80s vs ~6s for this
    Gram-matrix eigendecomposition (n_rows is at most ~50, so ``eigh`` on the
    n_rows x n_rows Gram matrix is essentially free; the O(n_rows x n_pixels)
    matrix multiply to build it is the only pass over the full image data).

    Returns whitened data ``K`` of shape (n_components, n_pixels), each row
    scaled to unit variance (the standard PCA pre-whitening step for ICA).
    """
    n_rows = I_c.shape[0]
    gram = (I_c @ I_c.T) / max(I_c.shape[1] - 1, 1)
    ev, evec = np.linalg.eigh(gram)
    idx = np.argsort(ev)[::-1][:n_components]
    ev_k = np.maximum(ev[idx], 1e-10)
    evec_k = evec[:, idx]
    s_k_sqrt = np.sqrt(ev_k) * np.sqrt(max(n_rows - 1, 1))
    K = (evec_k.T @ I_c) / (s_k_sqrt[:, None] + 1e-10)
    return K.astype(np.float32)



def fastica_fit(
    K: np.ndarray,
    n_components: int,
    max_iter: int = 200,
    tol: float = 1e-4,
    subsample: int | None = 400_000,
    seed: int = 42,
    progress_cb: Callable[[str, float], None] | None = None,
) -> np.ndarray:
    """Fit a FastICA (log-cosh) unmixing matrix on whitened data ``K``.

    ``K`` has shape (n_components, n_pixels). The fixed-point update matrix
    ``W`` is only n_components x n_components, so the iterative optimisation
    only needs a representative sample of pixels to estimate reliably — not
    every pixel in a 100MP image. Running the loop on the full pixel count
    (the previous behaviour) made each iteration take seconds, and with
    200-300 iterations of a non-converging fit that is the 15-20 minute hang
    this function replaces. With ``subsample`` set, the loop runs on a random
    subset of pixels; the caller then projects the *full* ``K`` through the
    fitted matrix once (a single fast matmul) for full-resolution output.

    Returns the fitted (n_components, n_components) unmixing matrix.
    """
    n_pix = K.shape[1]
    if subsample is not None and n_pix > subsample:
        rng = np.random.default_rng(seed)
        cols = rng.integers(0, n_pix, size=subsample)
        K_fit = K[:, cols]
    else:
        K_fit = K

    rng2 = np.random.RandomState(seed)
    W_ica = rng2.randn(n_components, n_components).astype(np.float32)
    for i in range(n_components):
        W_ica[i] /= np.linalg.norm(W_ica[i]) + 1e-9

    for it in range(max_iter):
        W_old = W_ica.copy()
        wx = W_ica @ K_fit
        g = np.tanh(wx)
        g_d = 1.0 - g**2
        W_new = (g @ K_fit.T) / K_fit.shape[1] - g_d.mean(axis=1, keepdims=True) * W_ica
        eig_vals, eig_vecs = np.linalg.eigh(W_new @ W_new.T)
        eig_vals = np.maximum(eig_vals, 1e-12)
        W_ica = (eig_vecs / np.sqrt(eig_vals) @ eig_vecs.T) @ W_new
        lim = float(np.max(np.abs(np.abs(np.diag(W_ica @ W_old.T)) - 1)))
        if progress_cb and (it % 5 == 0 or it == max_iter - 1):
            progress_cb(f"FastICA iteration {it + 1}/{max_iter} (delta={lim:.2e})",
                        it / max_iter)
        if lim < tol:
            if progress_cb:
                progress_cb(f"FastICA converged at iteration {it + 1}", 1.0)
            break

    return W_ica
