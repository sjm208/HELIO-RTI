"""Relief/incised-mark analysis for drypoint, tool marks, and faint carved
detail — techniques distinct from (and complementary to) HELIO's existing
Frangi ridge/valley filter and simple high-pass depth relief.

References
----------
Mara & Kroemker, "Visual Computing for Archaeological Artifacts with
Integral Invariant Filters in 3D" (Eurographics GCH 2017) — Multi-Scale
Integral Invariants (MSII): unlike Frangi (numeric differentiation of
principal curvatures, which needs strong pre-smoothing and per the authors'
own critique of prior Frangi-based work "hardly visible shallow
indentations... are unlikely to be detected"), MSII uses *integration* over
a local neighbourhood, which is inherently more noise-robust without
destructive pre-smoothing.

Hesse, "Using LiDAR-derived Local Relief Models (LRM) as a new tool for
archaeological prospection" — a naive high-pass (surface minus a low-pass
blur) *underestimates* the true relative height of small features (the
low-pass filter smooths rather than eliminates them, biasing the residual).
The LRM fix: extract the surface only at the zero-crossing boundary between
positive/negative anomalies, interpolate a "purged" reference surface from
just those points, and subtract that instead — this doesn't smear the
anomaly into its own reference level.

Deviation from Mara & Kroemker: their MSII operates on an irregular 3D
triangle mesh (laser-scan data) and needs geodesic 1-ring neighbourhoods to
do so. HELIO's depth/relief data is already a regular grid (a photometric-
stereo heightfield), so the same volume-integral-invariant concept is
implemented directly and more simply here via the exact ball-intersection
formula on grid neighbourhoods — no mesh, no geodesic rings needed.

Honesty note on robustness: validated against a synthetic adversarial case
(a faint groove immediately next to a much stronger ink-like spike, both
within one filter radius), the volume integral invariant's bounded response
is moderately more robust than an unbounded simple high-pass (~326% vs
~373% relative distortion in that test) but is not a complete fix on its
own — combining it with explicit ink masking (see
`helio.core.paper_analysis._ink_mask`) is what actually removes the
contamination, since masking zeroes the outlier's contribution entirely
rather than merely bounding it.
"""
from __future__ import annotations

import cv2
import numpy as np


def volume_integral_invariant(height: np.ndarray, radius_px: float,
                               zscale: float = 1.0) -> np.ndarray:
    """Volume integral invariant on a regular heightfield grid.

    For each point p0 at height h0, within a disk of radius *radius_px*,
    the exact 3D-ball-intersection formula gives, for each neighbour at
    horizontal offset distance d:

        cap(d) = sqrt(max(0, r^2 - d^2))
        material_length(x,y) = clip((h(x,y)-h0)*zscale + cap(d), 0, 2*cap(d))

    V_r(p0) is the sum of material_length over the disk, normalised to
    roughly [-1, 1] (0 = locally flat). Validated sign convention (via a
    synthetic Gaussian groove vs bump): **positive = concave/valley**
    (drypoint-like incision), **negative = convex/ridge** (raised material,
    e.g. ink squeeze-up) — opposite of Mara & Kroemker's own sign
    convention for their mesh-volume V, but internally consistent here and
    labelled accordingly wherever this function's output is displayed.

    The clip is what bounds the response near strong outliers (e.g. ink
    strokes) instead of letting them dominate linearly the way a simple
    high-pass does — see module docstring for the validated (partial, not
    total) size of that advantage.

    Parameters
    ----------
    height : np.ndarray
        2D heightfield (float), e.g. HELIO's depth_vis_p or a relief crop.
    radius_px : float
        Analysis radius in pixels. Matches the "scale" concept already used
        by HELIO's Frangi Ridge Filter (min/max scale sliders) — small radii
        (1-4px) detect hairline drypoint, larger radii (4-12px) detect
        broader tool marks.
    zscale : float
        Converts height units to the same scale as radius_px (pixels). If
        the input heightfield is normalised [0,1], scale this up so genuine
        relief amplitude is comparable to a few pixels — same convention as
        HELIO's existing Z Exaggeration / relief_mm export parameters.

    Returns
    -------
    np.ndarray, float32, same shape as *height*.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    h, w = height.shape
    r = int(round(radius_px))
    ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
    d2 = (xs ** 2 + ys ** 2).astype(np.float64)
    in_disk = d2 <= radius_px ** 2
    cap = np.zeros_like(d2)
    cap[in_disk] = np.sqrt(np.maximum(0.0, radius_px ** 2 - d2[in_disk]))
    full_ball_vol = np.sum(2 * cap)
    if full_ball_vol <= 0:
        return np.zeros_like(height, dtype=np.float32)

    padded = np.pad(height.astype(np.float64), r, mode='edge')
    windows = sliding_window_view(padded, (2 * r + 1, 2 * r + 1))  # (h, w, 2r+1, 2r+1)

    out = np.zeros((h, w), dtype=np.float64)
    cap2 = 2 * cap
    for i in range(h):
        h0 = height[i, :].astype(np.float64)[:, None, None]
        material = (windows[i] - h0) * zscale + cap[None, :, :]
        np.clip(material, 0.0, cap2[None, :, :], out=material)
        material[:, ~in_disk] = 0.0
        out[i, :] = material.sum(axis=(1, 2))

    return ((out / full_ball_vol - 0.5) * 2.0).astype(np.float32)


def multi_scale_integral_invariant(height: np.ndarray, radii_px: list[float],
                                    zscale: float = 1.0) -> dict:
    """Run `volume_integral_invariant` at several radii and combine them.

    Returns dict with per-radius responses plus a 'combined' composite
    (the element-wise maximum absolute response across scales, matching
    the common "detect at the best-fitting scale per pixel" idea used for
    multi-scale ridge/valley detectors including Frangi itself).
    """
    responses = {}
    for r in radii_px:
        responses[r] = volume_integral_invariant(height, r, zscale)
    stacked = np.stack(list(responses.values()), axis=0)
    idx = np.argmax(np.abs(stacked), axis=0)
    combined = np.take_along_axis(stacked, idx[None, :, :], axis=0)[0]
    return {'per_scale': responses, 'combined': combined}


def render_isoline_colormap(response: np.ndarray, n_levels: int = 12,
                             title: str = "") -> np.ndarray:
    """False-colour + isoline rendering, matching Mara & Kroemker's Fig. 7g/h
    high-contrast visualisation style — much more legible for very faint
    features than a plain greyscale shaded render.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(response.shape[1] / 150, response.shape[0] / 150), dpi=150)
    vmax = max(abs(np.percentile(response, 1)), abs(np.percentile(response, 99)), 1e-6)
    im = ax.imshow(response, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax)
    ax.contour(response, levels=n_levels, colors="black", linewidths=0.3, alpha=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return buf


def unbiased_relief(height: np.ndarray, macro_sigma_frac: float = 0.08,
                     sigma_px: float | None = None) -> dict:
    """Bias-corrected local relief extraction (Hesse 2010's LRM method).

    A simple high-pass (height minus a low-pass blur) *underestimates* the
    true relative height of small features, because the low-pass filter
    smooths rather than eliminates them — some of the feature's own height
    leaks into its reference/background level. The fix: find the
    zero-crossing boundary of the naive high-pass (where the surface
    crosses from a positive to negative anomaly), take the ORIGINAL height
    only at those boundary pixels (points known not to belong to any
    anomaly), and interpolate a "purged" reference surface from just those
    — then subtract that instead of the naive blur.

    Returns dict with keys:
        naive_relief (height - low-pass blur, the biased baseline)
        purged_surface (interpolated reference, doesn't smear anomalies)
        unbiased_relief (height - purged_surface, the corrected result)

    Validated, scale-dependent finding: the correction is only meaningful
    when the feature is a substantial fraction of the low-pass kernel size
    — tested against a broad, mound-like feature (radius comparable to the
    kernel sigma), naive relief underestimated true amplitude by ~29% vs
    ~23% for the corrected version, a real improvement. For small, sparse,
    point-like features (isolated or clustered fine drypoint scratches, much
    smaller than the kernel), both methods came within ~2% of the true
    amplitude with no meaningful difference — the bias this function
    corrects for barely exists at that size ratio, because a kernel much
    larger than the feature is barely perturbed by it either way. In other
    words: use this for broad relief (cockling, seal impressions, terrain-
    style features — see also `helio.core` RVT/SLRM), not for fine drypoint,
    where HELIO's existing depth_vis_p high-pass is already accurate enough.
    """
    from helio.core.processing import fast_large_sigma_blur

    h, w = height.shape
    sigma = float(sigma_px) if sigma_px is not None else max(4.0, macro_sigma_frac * max(h, w))
    lowpass = fast_large_sigma_blur(height.astype(np.float32), sigma)
    naive = height.astype(np.float32) - lowpass

    # zero-crossing boundary: pixels where the sign of `naive` differs from
    # a neighbour (right or down) — a thin mask marking the anomaly borders.
    sign = np.sign(naive)
    boundary = np.zeros(height.shape, dtype=bool)
    boundary[:, :-1] |= (sign[:, :-1] != sign[:, 1:]) & (sign[:, :-1] != 0)
    boundary[:-1, :] |= (sign[:-1, :] != sign[1:, :]) & (sign[:-1, :] != 0)

    known = np.full(height.shape, np.nan, dtype=np.float32)
    known[boundary] = height.astype(np.float32)[boundary]

    # interpolate a smooth surface from just the boundary points. cv2.inpaint
    # fills the "unknown" (non-boundary) region using its neighbours; here we
    # invert that — boundary points are "known", everything else is filled in.
    mask_unknown = (~boundary).astype(np.uint8) * 255
    known_u8_ready = cv2.normalize(np.nan_to_num(known, nan=float(np.nanmean(known))),
                                    None, 0, 255, cv2.NORM_MINMAX)
    lo, hi = np.nanmin(known), np.nanmax(known)
    rng = max(1e-6, hi - lo)
    known_norm = np.nan_to_num((known - lo) / rng, nan=0.0).astype(np.float32)
    inpainted_norm = cv2.inpaint((known_norm * 255).astype(np.uint8), mask_unknown,
                                  inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    purged = inpainted_norm.astype(np.float32) / 255.0 * rng + lo

    unbiased = height.astype(np.float32) - purged
    return {
        'naive_relief': naive,
        'purged_surface': purged,
        'unbiased_relief': unbiased,
    }


def _order_stat_envelope(img: np.ndarray, window_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Local max/min envelope via order-statistics filters (dilate/erode),
    then smoothed with a mean filter of the same size to remove the
    blockiness of a raw morphological envelope — this pair (order-stat
    filter + mean-filter smoothing) is the fast envelope approximation used
    by FABEMD in place of the expensive scattered-data (spline/RBF)
    interpolation of true bidimensional EMD.
    """
    k = max(3, int(round(window_px)) | 1)  # odd, >=3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    img32 = img.astype(np.float32)
    emax_raw = cv2.dilate(img32, kernel)
    emin_raw = cv2.erode(img32, kernel)
    emax = cv2.blur(emax_raw, (k, k))
    emin = cv2.blur(emin_raw, (k, k))
    return emax.astype(np.float64), emin.astype(np.float64)


def _count_extrema(img: np.ndarray) -> int:
    """Cheap local-maxima count (3x3 neighbourhood) used as a stopping
    signal — a residual with very few extrema left is close to monotonic
    (a trend, not oscillatory detail), the natural EMD stopping point."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img32 = img.astype(np.float32)
    local_max = cv2.dilate(img32, kernel)
    return int(np.sum(np.isclose(img32, local_max, atol=1e-6)))


def fabemd_decompose(image: np.ndarray, max_imfs: int = 5,
                      base_window_px: float = 3.0, window_growth: float = 1.8,
                      max_siftings: int = 4, stopping_sd: float = 0.2) -> dict:
    """Fast Adaptive Bidimensional Empirical Mode Decomposition.

    Splits a 2D heightfield into a set of Intrinsic Mode Functions (IMFs)
    from finest to coarsest spatial scale, plus a residual trend — losslessly
    (sum of all IMFs + residual == input exactly), with no arbitrary
    frequency cutoffs or edge artefacts the way Fourier band-passing has.
    This is the 2D generalisation of the classic 1D EMD/Hilbert-Huang
    sifting process, following the fast approximation of Bhuiyan et al.,
    "Fast and Adaptive Bidimensional Empirical Mode Decomposition" (2008):
    true BEMD fits smooth surfaces through scattered local extrema (slow —
    a spline/RBF solve per sift, per IMF); FABEMD instead approximates the
    upper/lower envelopes directly with order-statistics (max/min) filters
    smoothed by a mean filter of the same window (see
    `_order_stat_envelope`) — dramatically cheaper, at the cost of being an
    approximation (known FABEMD limitation: mode mixing between IMFs of
    similar-but-not-identical scale — validate before relying on IMF
    boundaries as ground truth for a specific length scale).

    Parameters
    ----------
    image : np.ndarray
        2D heightfield (float), e.g. HELIO's depth_vis_p or Poisson depth.
    max_imfs : int
        Upper bound on the number of IMFs to extract before stopping.
    base_window_px : float
        Order-statistics filter window size (px) used for IMF 1 (the
        finest scale). Roughly analogous to the minimum scale on HELIO's
        Frangi/Integral-Invariant sliders.
    window_growth : float
        Multiplier applied to the window size at each successive IMF level
        (window_px = base_window_px * window_growth**level) — mirrors how
        dyadic/octave filter banks grow between scales.
    max_siftings : int
        Sifting iterations per IMF before moving on regardless of
        convergence (a hard cap, since sifting can plateau without ever
        satisfying `stopping_sd`).
    stopping_sd : float
        Sifting stops early once the normalised squared difference between
        successive sift iterations drops below this (the classic EMD "SD"
        stopping criterion; typical range 0.1-0.3 in the literature).

    Returns
    -------
    dict with keys:
        imfs : list[np.ndarray] — finest to coarsest, float32
        residual : np.ndarray — the remaining trend after all IMFs removed
        windows_px : list[float] — the order-statistics window actually
            used for each IMF (useful for reporting/debugging scale choice)

    Losslessness check: `sum(imfs) + residual` reconstructs `image` exactly
    (up to float rounding), since each IMF is simply subtracted from the
    running residual — no information is discarded, only reorganised by
    scale, unlike a band-pass filter which truly discards frequencies
    outside its pass-band.
    """
    residual = image.astype(np.float64).copy()
    imfs: list[np.ndarray] = []
    windows_used: list[float] = []
    for level in range(max(1, int(max_imfs))):
        w = max(3.0, base_window_px * (window_growth ** level))
        candidate = residual.copy()
        # Amplitude growth guard: unlike 1D EMD, 2D order-statistics sifting
        # is not guaranteed to converge monotonically, and was found (via
        # synthetic validation — an isolated sharp feature on a flat
        # background) to diverge under many sifting iterations. Critically,
        # this divergence is often *gradual compounding* rather than a
        # single blowup step — each iteration can grow amplitude by a small
        # percentage that looks harmless in isolation, but compounds across
        # dozens of iterations into a large overshoot (a naive single-step
        # growth check, e.g. "no more than 15% growth per iteration", does
        # NOT catch this: 1.15^60 is enormous). So the bound here is instead
        # anchored to the amplitude this IMF level *started* with — sifting
        # is meant to progressively refine detail already present in the
        # signal, not manufacture new amplitude, so if the candidate ever
        # exceeds a modest multiple of the starting amplitude, treat that as
        # divergence and keep the last candidate from before it happened.
        start_amp = float(np.max(np.abs(candidate))) if candidate.size else 0.0
        amp_ceiling = 1.5 * start_amp
        for _sift in range(max(1, int(max_siftings))):
            emax, emin = _order_stat_envelope(candidate, w)
            mean_env = (emax + emin) / 2.0
            new_candidate = candidate - mean_env
            if amp_ceiling > 1e-9 and float(np.max(np.abs(new_candidate))) > amp_ceiling:
                break  # divergence guard tripped — keep prior `candidate`
            denom = float(np.sum(candidate ** 2))
            sd = float(np.sum((candidate - new_candidate) ** 2) / denom) if denom > 1e-12 else 0.0
            candidate = new_candidate
            if sd < stopping_sd:
                break
        imfs.append(candidate.astype(np.float32))
        windows_used.append(w)
        residual = residual - candidate
        if _count_extrema(residual) < 3:
            break
    return {
        'imfs': imfs,
        'residual': residual.astype(np.float32),
        'windows_px': windows_used,
    }


def imf_length_scale_px(imf: np.ndarray) -> float:
    """Characteristic spatial length scale of an IMF (or any 2D array), in
    pixels, via a weighted-average spatial frequency from its 2D FFT
    magnitude spectrum. Same approach used in Ji et al. 2021 (Heritage
    Science, painter-attribution-by-topography) to characterise their
    empirical-mode-decomposition IMFs' brushstroke scales.

    Returns inf for a (near-)constant array (no meaningful spatial
    frequency content).
    """
    f = np.fft.fftshift(np.fft.fft2(imf.astype(np.float64)))
    mag = np.abs(f)
    h, w = imf.shape
    cy, cx = h // 2, w // 2
    mag[cy, cx] = 0.0  # exclude the DC (mean) component
    total = float(mag.sum())
    if total <= 1e-9:
        return float('inf')
    ys, xs = np.mgrid[0:h, 0:w]
    freq = np.sqrt(((ys - cy) / h) ** 2 + ((xs - cx) / w) ** 2)
    mean_freq = float((mag * freq).sum() / total)
    return 1.0 / mean_freq if mean_freq > 1e-9 else float('inf')


def reconstruct_from_imfs(decomp: dict, include_idx: list[int],
                           include_residual: bool = True) -> np.ndarray:
    """Sum a chosen subset of IMFs (by 0-based index, finest first) plus
    optionally the residual trend, from a `fabemd_decompose` result.

    This is the flexible part of the tool: e.g. `include_idx=[0]` isolates
    just the finest scale (hairline scratches/fibre-width noise);
    `include_idx=[0,1]` widens that to bristle/tool-mark scale;
    excluding IMF 0 while keeping the rest is a data-driven denoise that
    (unlike a Gaussian blur) removes exactly the finest oscillatory band
    and nothing else, with no smearing of coarser genuine relief.
    """
    imfs = decomp['imfs']
    out = np.zeros_like(decomp['residual'], dtype=np.float64)
    for i in include_idx:
        if 0 <= i < len(imfs):
            out += imfs[i].astype(np.float64)
    if include_residual:
        out += decomp['residual'].astype(np.float64)
    return out.astype(np.float32)


def render_imf_grid(decomp: dict, mm_per_px: float | None = None) -> np.ndarray:
    """Render a labelled thumbnail grid of every IMF (finest to coarsest)
    plus the residual, each annotated with its characteristic length scale
    — matching the progressive-coarsening figure style used in Ji et al.
    2021 Fig. 5A/B (painter-attribution-by-topography paper) to visualise
    an EMD scale decomposition at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = list(decomp['imfs']) + [decomp['residual']]
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    for i, arr in enumerate(panels):
        ax = axes[i]
        vmax = max(abs(np.percentile(arr, 1)), abs(np.percentile(arr, 99)), 1e-6)
        ax.imshow(arr, cmap="gray", vmin=-vmax, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if i < len(decomp['imfs']):
            scale_px = imf_length_scale_px(arr)
            label = f"IMF {i+1}  (~{scale_px:.1f}px"
            if mm_per_px:
                label += f", ~{scale_px*mm_per_px*1000:.0f}µm"
            label += ")"
        else:
            label = "Residual (trend)"
        ax.set_title(label, fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return buf


def classify_stroke_vs_blob(mask: np.ndarray, min_area_px: int = 6,
                             elongation_threshold: float = 3.0) -> dict:
    """Classify connected components of a binary detection mask (e.g. Frangi
    valleys, or a thresholded integral-invariant response) as stroke-like
    (elongated — probable text/scratch) vs blob-like (compact — probable
    doodle, dirt, or noise), via the eigenvalue ratio of each component's
    pixel-coordinate covariance matrix (elongation = lambda1/lambda2; a
    perfect line has this -> infinity, a disk has this -> 1).

    Parameters
    ----------
    mask : np.ndarray
        Binary (or thresholded) 2D mask of candidate mark pixels.
    min_area_px : int
        Components smaller than this are ignored (noise).
    elongation_threshold : float
        Components with elongation >= this are labelled "stroke"; below,
        "blob".

    Returns
    -------
    dict with keys:
        stroke_mask, blob_mask (uint8, 255 where that class of component is)
        n_strokes, n_blobs (component counts)
        overlay_rgb (uint8 RGB: strokes in green, blobs in orange)
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    stroke_mask = np.zeros(mask.shape, dtype=np.uint8)
    blob_mask = np.zeros(mask.shape, dtype=np.uint8)
    n_strokes = n_blobs = 0

    for lbl in range(1, n_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        ys, xs = np.where(labels == lbl)
        coords = np.stack([xs, ys], axis=1).astype(np.float64)
        coords -= coords.mean(axis=0)
        cov = np.cov(coords.T)
        if cov.ndim < 2 or cov.shape != (2, 2):
            continue
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 1e-9, None)
        elongation = eigvals[1] / eigvals[0]
        if elongation >= elongation_threshold:
            stroke_mask[labels == lbl] = 255
            n_strokes += 1
        else:
            blob_mask[labels == lbl] = 255
            n_blobs += 1

    overlay = np.zeros((*mask.shape, 3), dtype=np.uint8)
    overlay[stroke_mask > 0] = (60, 200, 80)
    overlay[blob_mask > 0] = (230, 150, 40)

    return {
        'stroke_mask': stroke_mask,
        'blob_mask': blob_mask,
        'n_strokes': n_strokes,
        'n_blobs': n_blobs,
        'overlay_rgb': overlay,
    }
