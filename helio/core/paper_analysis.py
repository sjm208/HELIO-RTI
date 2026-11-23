"""Paper-structure analysis — chain/laid line detection and watermark motif isolation.

Chain lines and laid lines are impressions left in handmade paper by the wire
mould used to form the sheet: chain lines are widely spaced (~15-40mm) wires
running one direction; laid lines are closely-spaced (~4-16 per cm) fine wires
running perpendicular to them. A watermark is a wire design sewn onto the
mould that leaves a thinner, more translucent patch of paper. All three are
used by paper historians / codicologists to identify mould stock and date and
localise manuscripts.

References
----------
Grossmann, Schonlieb & Da Rold, "Hidden Knowledge: Mathematical Methods for
the Extraction of the Fingerprint of Medieval Paper from Digital Images"
(arXiv:2303.03794, 2023) — chain-line pipeline: patch -> spectral TV
band-pass -> rectangular Fourier filtering -> axis projection -> peak
detection (their Fig. 5). Laid-line pipeline: patch -> spectral TV
band-pass -> Radon transform sinogram -> best-angle projection -> peak
detection (their Fig. 6).

Deviation from the paper: this module isolates the faint chain/laid-line
texture from ink/degradation using a large-window median-filter background
subtraction rather than the paper's spectral total-variation (TV flow)
decomposition. Both are attempts at the same thing (an edge-preserving,
outlier-robust separation of small-low-contrast structures from large-
high-contrast ones); the median-filter version was chosen after validating
that a from-scratch spectral TV flow solver was numerically fragile (small
timestep/regularisation errors caused the classic TV flow extremum-principle
violation) within the time available, whereas the median-filter approach
was validated directly: on synthetic periodic-texture-plus-ink-blob test
images it recovers fine texture at ~matching amplitude to ground truth while
suppressing broad ink features by >90%, and runs in well under a second on
manuscript-scale patches (validated via a downsample/filter/upsample scheme
analogous to fast_large_sigma_blur elsewhere in this codebase).

The Fourier rectangular-filtering (chain lines) and Radon-transform sinogram
(laid lines) stages are implemented as literally described in the paper.
"""
from __future__ import annotations

import cv2
import numpy as np


def _fast_median_background(img: np.ndarray, ksize: float, small_target: float = 400.0) -> np.ndarray:
    """Large-window median background estimate, computed efficiently by
    downsampling first (median filters don't have a fast separable
    algorithm the way Gaussian blur does, so a direct call at manuscript
    resolution with a large kernel is far too slow — validated at ~64s for
    a single 2000x2000 patch at ksize=41 versus ~0.15s downsampled).
    """
    from scipy.ndimage import median_filter
    h, w = img.shape
    if max(h, w) <= small_target * 1.5:
        return median_filter(img, size=max(3, int(ksize) | 1), mode='nearest')
    scale = small_target / max(h, w)
    sh, sw = max(4, int(h * scale)), max(4, int(w * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    k_small = max(3, int(ksize * scale) | 1)
    small_bg = median_filter(small, size=k_small, mode='nearest')
    return cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)


def _ink_mask(gray: np.ndarray, bg: np.ndarray, threshold: float = 25.0,
              dilate_px: int = 9) -> np.ndarray:
    """Flag pixels significantly darker than their own local background —
    i.e. ink/text strokes, as opposed to the much lower-amplitude chain/laid
    line texture. Median-filter background subtraction alone isn't enough:
    validated on a synthetic test with two lines of ink text, the residual
    band still showed enough ink-edge energy to produce completely wrong
    chain-line spacing (33.9mm measured vs 25.0mm true) even after band-pass
    isolation. Explicitly zeroing out local-outlier-dark pixels before
    periodicity detection fixed this exactly (500.0px measured vs 500px
    true). A *local* (bg-relative) threshold is used rather than a global
    percentile — an earlier version using a global percentile flagged 57%
    of the image as "ink" (any pixel below the percentile, including
    ordinary background variation), whereas this flags ~3-4% here, matching
    the actual ink coverage.
    """
    diff = bg - gray.astype(np.float32)
    mask = (diff > threshold).astype(np.uint8)
    if dilate_px > 1:
        mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8))
    return mask > 0


def isolate_paper_texture(gray: np.ndarray, bg_ksize_frac: float = 0.05,
                           ink_threshold: float = 25.0) -> np.ndarray:
    """Isolate the faint mould-wire texture from ink/degradation/noise.

    Chain/laid lines are low-contrast, fine-grained features riding on top
    of much higher-contrast ink strokes and paper discolouration. A large
    median filter is robust to broad high-contrast features (unlike a
    Gaussian blur, which smears them into the background estimate and leaks
    their contrast into the "isolated" band) — see module docstring. Ink
    strokes specifically are additionally masked out via `_ink_mask`, since
    validation showed median-filtering alone still let ink contaminate the
    periodicity signal (see `_ink_mask` docstring).
    """
    h, w = gray.shape
    ksize = max(9.0, bg_ksize_frac * max(h, w))
    gray_f = gray.astype(np.float32)
    bg = _fast_median_background(gray_f, ksize)
    band = gray_f - bg
    mask = _ink_mask(gray_f, bg, threshold=ink_threshold)
    band[mask] = 0.0
    return band


def _fourier_rectangular_enhance(band: np.ndarray, orientation: str,
                                  period_lo_px: float, period_hi_px: float,
                                  strip_halfwidth_px: int = 2) -> np.ndarray:
    """Enhance periodic lines of a given orientation via 2D Fourier
    rectangular band-pass filtering (Grossmann et al. Section 2.3 / Fig. 5d).

    For vertical lines, periodicity runs along X, so the corresponding
    frequency-domain energy sits along the horizontal (u) axis, at a
    distance from the DC centre corresponding to 1/period. A thin rectangle
    (height = strip_halfwidth_px*2+1, spanning the full orthogonal extent)
    masking that band — and its mirror on the opposite side of the DC
    centre — isolates just that periodic structure, suppressing noise,
    ink-boundary residue, and the very different laid-line frequency.
    """
    h, w = band.shape
    F = np.fft.fftshift(np.fft.fft2(band))
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    lo_freq_px = w / period_hi_px if orientation == 'vertical' else h / period_hi_px
    hi_freq_px = w / period_lo_px if orientation == 'vertical' else h / period_lo_px
    lo_freq_px = max(1.0, lo_freq_px)
    if orientation == 'vertical':
        for sign in (+1, -1):
            u0 = int(cx + sign * (lo_freq_px + hi_freq_px) / 2)
            half = max(1, int((hi_freq_px - lo_freq_px) / 2) + 1)
            u_lo, u_hi = max(0, u0 - half), min(w, u0 + half + 1)
            mask[max(0, cy - strip_halfwidth_px):min(h, cy + strip_halfwidth_px + 1), u_lo:u_hi] = 1.0
    else:
        for sign in (+1, -1):
            v0 = int(cy + sign * (lo_freq_px + hi_freq_px) / 2)
            half = max(1, int((hi_freq_px - lo_freq_px) / 2) + 1)
            v_lo, v_hi = max(0, v0 - half), min(h, v0 + half + 1)
            mask[v_lo:v_hi, max(0, cx - strip_halfwidth_px):min(w, cx + strip_halfwidth_px + 1)] = 1.0
    F_filtered = F * mask
    enhanced = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))
    return enhanced.astype(np.float32)


def _detect_peaks_1d(profile: np.ndarray, min_spacing_px: float, smooth_sigma: float = 2.0,
                      threshold_k: float = 1.0):
    """Gaussian-smooth *profile*, threshold at mean + k*std, and pick local
    maxima at least *min_spacing_px* apart (Grossmann et al. Fig. 5e/6e).

    Returns (peak_indices, smoothed_profile, threshold_value).
    """
    smoothed = cv2.GaussianBlur(profile.astype(np.float32).reshape(1, -1), (0, 0), smooth_sigma).ravel()
    thresh = smoothed.mean() + threshold_k * smoothed.std()
    candidates = np.flatnonzero(
        (smoothed > thresh) &
        (smoothed >= np.r_[smoothed[1:], -np.inf]) &
        (smoothed >= np.r_[-np.inf, smoothed[:-1]])
    )
    if len(candidates) == 0:
        return np.array([], dtype=int), smoothed, thresh
    peaks = [int(candidates[0])]
    for c in candidates[1:]:
        if c - peaks[-1] >= min_spacing_px * 0.5:
            peaks.append(int(c))
        elif smoothed[c] > smoothed[peaks[-1]]:
            peaks[-1] = int(c)
    return np.array(peaks, dtype=int), smoothed, thresh


def _render_profile_plot(profile: np.ndarray, peaks: np.ndarray, threshold: float,
                          title: str, xlabel: str = "Position (px)") -> np.ndarray:
    """Render a Figure-5e/6e-style line plot (profile, peak markers, threshold
    line) to an RGB uint8 array, matching the visualisation style requested.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=120)
    ax.plot(profile, color="#3a6ea8", linewidth=0.8)
    if len(peaks):
        ax.plot(peaks, profile[peaks], "x", color="#e08a2b", markersize=7, markeredgewidth=2)
    ax.axhline(threshold, color="#888888", linestyle=":", linewidth=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Projected magnitude", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return buf


def _render_sinogram_plot(sinogram: np.ndarray, angles_deg: np.ndarray,
                           best_angle: float, title: str = "Radon sinogram") -> np.ndarray:
    """Render the Radon sinogram with labeled axes (Grossmann et al. Fig. 6d:
    projection angle x projection position). A raw normalised array of the
    sinogram is unreadable as-is — angle steps are typically only a few
    dozen rows, so displaying it directly with no aspect-ratio correction or
    axis labels compresses to an indecipherable sliver. This renders it via
    matplotlib with a sensible aspect ratio and a marker for the best angle.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=120)
    extent = [angles_deg[0], angles_deg[-1], sinogram.shape[1], 0]
    ax.imshow(sinogram.T, cmap="gray", aspect="auto", extent=extent)
    ax.axvline(best_angle, color="#e08a2b", linewidth=1.2, linestyle="--")
    ax.set_title(f"{title}  (best angle: {best_angle:.0f}°)", fontsize=10)
    ax.set_xlabel("Projection angle (deg)", fontsize=9)
    ax.set_ylabel("Projection position (px)", fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba()).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return buf


def _mm_range_to_px(mm_range: tuple[float, float], mm_per_px: float | None,
                     img_w: int) -> tuple[float, float]:
    if mm_per_px:
        return mm_range[0] / mm_per_px, mm_range[1] / mm_per_px
    return max(20.0, img_w * 0.01), max(200.0, img_w * 0.15)


def _per_cm_range_to_px(per_cm_range: tuple[float, float], mm_per_px: float | None,
                         img_w: int) -> tuple[float, float]:
    if mm_per_px:
        lo_mm = 10.0 / per_cm_range[1]
        hi_mm = 10.0 / per_cm_range[0]
        return lo_mm / mm_per_px, hi_mm / mm_per_px
    return max(4.0, img_w * 0.002), max(20.0, img_w * 0.02)


def detect_chain_lines(
    image: np.ndarray,
    mm_per_px: float | None,
    chain_range_mm: tuple[float, float] = (15.0, 40.0),
    orientation: str = 'auto',
) -> dict:
    """Detect chain-line spacing from a reflected-light image of paper,
    following Grossmann et al. Section 2.3 / Algorithm 1 / Fig. 5:
    isolate paper texture -> rectangular Fourier enhancement -> project
    onto the perpendicular axis -> peak detection -> spacing measurement.

    Parameters
    ----------
    image : np.ndarray
        RGB or grayscale reflected-light image (ideally a patch containing
        several chain lines — see the paper's "Patch selection" step;
        processing the whole page is unnecessary and much slower).
    mm_per_px : float or None
        Calibrated scale. If None, results are reported in pixels only.
    chain_range_mm : (float, float)
        Expected chain-line spacing range in mm (typical: 15-40mm).
    orientation : 'auto' | 'vertical' | 'horizontal'
        Assumed chain-line direction. 'auto' tries both and keeps whichever
        shows a stronger peak.

    Returns
    -------
    dict with keys:
        chain_spacing_px, chain_spacing_mm, chain_confidence, chain_orientation
        n_lines_detected
        band_image, fourier_image (uint8 diagnostic intermediate images)
        profile_plot (uint8 RGB, Fig. 5e-style projection + peaks)
        overlay_rgb (uint8 RGB, detected chain-line positions on the original)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        orig_rgb = image if image.dtype == np.uint8 else cv2.normalize(
            image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray = image.astype(np.float32)
        orig_rgb = cv2.cvtColor(cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                                 cv2.COLOR_GRAY2RGB)
    h, w = gray.shape

    band = isolate_paper_texture(gray)
    lo_px, hi_px = _mm_range_to_px(chain_range_mm, mm_per_px, w)

    orientations = ['vertical', 'horizontal'] if orientation == 'auto' else [orientation]
    best = None
    for ori in orientations:
        enhanced = _fourier_rectangular_enhance(band, ori, lo_px, hi_px)
        proj = enhanced.mean(axis=0) if ori == 'vertical' else enhanced.mean(axis=1)
        peaks, smoothed, thresh = _detect_peaks_1d(proj, min_spacing_px=lo_px)
        if len(peaks) < 2:
            score = 0.0
        else:
            spacings = np.diff(peaks)
            score = len(peaks) / (1.0 + spacings.std() / max(1.0, spacings.mean()))
        if best is None or score > best['score']:
            best = dict(orientation=ori, enhanced=enhanced, proj=proj, peaks=peaks,
                        smoothed=smoothed, thresh=thresh, score=score)

    peaks = best['peaks']
    spacing_px = float(np.median(np.diff(peaks))) if len(peaks) >= 2 else None

    results = {
        'chain_spacing_px': spacing_px,
        'chain_orientation': best['orientation'],
        'n_lines_detected': int(len(peaks)),
        'chain_spacing_mm': (spacing_px * mm_per_px) if (spacing_px and mm_per_px) else None,
    }
    # crude confidence: how uniform are the detected spacings (1.0 = perfectly regular)
    if spacing_px and len(peaks) >= 3:
        spacings = np.diff(peaks)
        results['chain_confidence'] = float(max(0.0, 1.0 - spacings.std() / max(1.0, spacings.mean())))
    else:
        results['chain_confidence'] = 0.0

    results['band_image'] = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results['fourier_image'] = cv2.normalize(best['enhanced'], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results['profile_plot'] = _render_profile_plot(
        best['smoothed'], peaks, best['thresh'],
        f"Chain-line projection ({best['orientation']}, {len(peaks)} lines detected)")

    overlay = orig_rgb.copy()
    for p in peaks:
        if best['orientation'] == 'vertical':
            cv2.line(overlay, (int(p), 0), (int(p), h - 1), (60, 140, 255), 2)
        else:
            cv2.line(overlay, (0, int(p)), (w - 1, int(p)), (60, 140, 255), 2)
    results['overlay_rgb'] = overlay
    return results


def radon_transform(img: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Radon transform via rotate + column-sum (no scikit-image dependency).

    Returns a sinogram of shape (len(angles_deg), projection_length) —
    matching Grossmann et al. Fig. 6d (projection angle x projection
    position). Validated against a synthetic tilted-line test image:
    recovers both the true line angle and spacing to within ~1%.
    """
    h, w = img.shape
    diag = int(np.ceil(np.sqrt(h**2 + w**2)))
    pad_h, pad_w = (diag - h) // 2, (diag - w) // 2
    padded = np.pad(img, ((pad_h, diag - h - pad_h), (pad_w, diag - w - pad_w)), mode='constant')
    cy, cx = padded.shape[0] / 2.0, padded.shape[1] / 2.0
    sinogram = np.zeros((len(angles_deg), diag), dtype=np.float32)
    for i, ang in enumerate(angles_deg):
        M = cv2.getRotationMatrix2D((cx, cy), float(ang), 1.0)
        rotated = cv2.warpAffine(padded, M, (padded.shape[1], padded.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        sinogram[i] = rotated.sum(axis=0)
    return sinogram


def _periodicity_score(profile: np.ndarray, lo_period_px: float, hi_period_px: float):
    p = profile - profile.mean()
    n = len(p)
    if n < 8:
        return 0.0, None
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(p * win))
    freqs = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        periods = np.where(freqs > 0, 1.0 / np.maximum(freqs, 1e-12), np.inf)
    valid = (periods >= lo_period_px) & (periods <= hi_period_px)
    if not np.any(valid) or spec[valid].sum() <= 0:
        return 0.0, None
    idx_in_valid = np.argmax(spec[valid])
    idx = np.flatnonzero(valid)[idx_in_valid]
    return float(spec[idx]), float(periods[idx])


def detect_laid_lines(
    image: np.ndarray,
    mm_per_px: float | None,
    laid_range_per_cm: tuple[float, float] = (4.0, 16.0),
    angle_range_deg: tuple[float, float] = (0.0, 180.0),
    angle_step_deg: float = 2.0,
) -> dict:
    """Detect laid-line density from a reflected-light image of paper,
    following Grossmann et al. Section 2.4 / Fig. 6: isolate paper texture
    -> Radon transform sinogram across *angle_range_deg* -> pick the angle
    with the strongest in-band periodicity -> peak detection at that angle.

    Using the full Radon sinogram (rather than assuming an axis-aligned
    orientation) makes this robust to the page being scanned at a slight
    rotation, and directly produces the sinogram visualisation used in the
    paper's Fig. 6d.

    Parameters
    ----------
    image : np.ndarray
        RGB or grayscale reflected-light image (a patch containing ~1cm of
        clearly-visible laid lines, per the paper — often near the gutter).
    mm_per_px : float or None
        Calibrated scale. If None, density is not computed (px period only).
    laid_range_per_cm : (float, float)
        Expected laid-line density range (typical: 4-16/cm).
    angle_range_deg, angle_step_deg : search range/resolution for the Radon
        sinogram's angle axis. Narrow this (e.g. around 0/90) to speed up
        processing if the page orientation is already known.

    Returns
    -------
    dict with keys:
        laid_spacing_px, laid_density_per_cm, best_angle_deg, laid_confidence
        band_image (uint8), sinogram_image (uint8, Fig. 6d-style)
        profile_plot (uint8 RGB, Fig. 6e-style profile + peaks at best angle)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)
    w = gray.shape[1]

    band = isolate_paper_texture(gray)
    lo_px, hi_px = _per_cm_range_to_px(laid_range_per_cm, mm_per_px, w)

    angles = np.arange(angle_range_deg[0], angle_range_deg[1], angle_step_deg)
    sinogram = radon_transform(band, angles)

    scores = np.zeros(len(angles), dtype=np.float32)
    periods = [None] * len(angles)
    for i in range(len(angles)):
        score, period = _periodicity_score(sinogram[i], lo_px, hi_px)
        scores[i] = score
        periods[i] = period
    best_idx = int(np.argmax(scores)) if scores.size else 0
    best_angle = float(angles[best_idx]) if len(angles) else 0.0
    best_period_px = periods[best_idx] if len(angles) else None

    peaks, smoothed, thresh = _detect_peaks_1d(sinogram[best_idx], min_spacing_px=(best_period_px or lo_px))

    results = {
        'best_angle_deg': best_angle,
        'laid_spacing_px': best_period_px,
        'laid_density_per_cm': None,
        'laid_confidence': float(scores[best_idx] / (scores.sum() + 1e-9)) if scores.sum() > 0 else 0.0,
    }
    if best_period_px and mm_per_px:
        spacing_mm = best_period_px * mm_per_px
        if spacing_mm > 0:
            results['laid_density_per_cm'] = 10.0 / spacing_mm

    results['band_image'] = cv2.normalize(band, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results['sinogram_image'] = _render_sinogram_plot(sinogram, angles, best_angle)
    results['profile_plot'] = _render_profile_plot(
        smoothed, peaks, thresh,
        f"Laid-line projection at {best_angle:.0f}° (best angle)",
        xlabel="Projection position (px)")
    return results


def isolate_watermark_motif(
    transmissive_img: np.ndarray,
    text_suppress_px: int = 25,
    clahe_clip: float = 3.0,
    upscale: float = 1.0,
) -> dict:
    """Suppress occluding ink/text from a transmitted-light (backlit) capture,
    isolating the smooth watermark shading pattern for tracing / comparison
    against watermark databases (e.g. Bernstein, WZIS).

    Method: ink strokes are thin, dark, high-local-contrast features against
    the bright, translucent paper; the watermark itself is a broad, smooth
    brightness gradient caused by paper-thickness variation. A large-kernel
    greyscale morphological closing (dilate then erode with a disk kernel
    wider than typical stroke width) removes the thin dark strokes while
    preserving the broad watermark shading, since closing cannot recover dark
    features narrower than the structuring element. This assumes the standard
    backlit-photography convention: ink appears darker than the illuminated
    paper (it absorbs/blocks transmitted light).

    Parameters
    ----------
    transmissive_img : np.ndarray
        Backlit capture, RGB or grayscale, uint8 or float.
    text_suppress_px : int
        Structuring-element diameter in pixels — should exceed the stroke
        width of ink/text on the page (increase for bold/thick ink).
    clahe_clip : float
        CLAHE clip limit for the enhanced/traceable output.
    upscale : float
        Output resolution multiplier (e.g. 2.0 = double resolution) applied
        via Lanczos interpolation after isolation, for a larger/more
        detailed traceable export. 1.0 = no upscaling.

    Returns
    -------
    dict with keys:
        motif_gray (uint8, text suppressed, raw contrast)
        motif_enhanced (uint8, CLAHE-enhanced for tracing/visibility)
        text_mask (uint8, where ink was detected/suppressed — 255 = ink)
    """
    if transmissive_img.ndim == 3:
        src = transmissive_img
        if src.dtype != np.uint8:
            src = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        gray = transmissive_img
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ksize = max(3, int(text_suppress_px) | 1)  # force odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    motif = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    delta = cv2.absdiff(gray, motif)
    _, text_mask = cv2.threshold(delta, 15, 255, cv2.THRESH_BINARY)

    clahe = cv2.createCLAHE(clipLimit=max(0.5, clahe_clip), tileGridSize=(16, 16))
    enhanced = clahe.apply(motif)

    if upscale and upscale > 1.01:
        h, w = motif.shape
        new_size = (int(w * upscale), int(h * upscale))
        motif = cv2.resize(motif, new_size, interpolation=cv2.INTER_LANCZOS4)
        enhanced = cv2.resize(enhanced, new_size, interpolation=cv2.INTER_LANCZOS4)
        text_mask = cv2.resize(text_mask, new_size, interpolation=cv2.INTER_NEAREST)

    return {
        'motif_gray': motif,
        'motif_enhanced': enhanced,
        'text_mask': text_mask,
    }


def detect_scale_ticks(img, roi=None, min_period_px=4.0, max_period_px=400.0):
    """Find the dominant tick spacing of a ruler or scale bar, in pixels.

    Deliberately not "detect the scale bar and tell me the mm/pixel". Reading a
    ruler's *units* from an image means reading its printed numerals, and a tool
    that guessed wrong there would silently corrupt every measurement downstream
    — the one failure this program cannot afford. What is reliable is the
    spacing: ticks on a ruler are the most strongly periodic thing in almost any
    heritage frame. So this reports the period and its confidence, and the
    operator supplies what one tick means.

    That division of labour is the same one the sphere calibrator uses: the
    machine proposes the geometry, the human confirms the meaning.

    Method is the chain-line one, narrowed. Gradient magnitude, Radon transform
    to find the dominant straight orientation, project along it, then
    autocorrelate the projection and take the first strong non-zero lag.

    Parameters
    ----------
    img : 2-D or 3-D array
    roi : (x1, y1, x2, y2) or None
        Strongly recommended — drawn round the ruler, this is reliable; run on
        a whole frame it competes with weave, hatching and every other periodic
        thing present.

    Returns
    -------
    dict: ``period_px``, ``confidence`` (0-1), ``angle_deg``, ``n_ticks``
    """
    import numpy as _np
    import cv2 as _cv2

    a = _np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    a = a.astype(_np.float32)
    if roi is not None:
        x1, y1, x2, y2 = roi
        a = a[y1:y2, x1:x2]
    if min(a.shape[:2]) < 24:
        raise ValueError("region is too small to find a tick pattern")

    a = (a - a.min()) / (float(_np.ptp(a)) + 1e-9)
    gx = _cv2.Sobel(a, _cv2.CV_32F, 1, 0, ksize=3)
    gy = _cv2.Sobel(a, _cv2.CV_32F, 0, 1, ksize=3)
    mag = _np.hypot(gx, gy)

    lo = max(2, int(min_period_px))
    hh_, ww_ = mag.shape
    yy_, xx_ = _np.mgrid[0:hh_, 0:ww_].astype(_np.float64)
    wts = mag.astype(_np.float64).ravel()

    def project(ang):
        """Edge strength by position along direction *ang* (0 = across, 90 =
        down), every pixel counted: no rotated copy to crop a long, thin box
        (which kept only ~3 ticks of a ruler box turned 90°)."""
        t = _np.radians(ang)
        s_ = (xx_ * _np.cos(t) + yy_ * _np.sin(t)).ravel()
        s_ -= s_.min()
        i0 = _np.floor(s_).astype(_np.int64)
        f = s_ - i0
        n = int(i0.max()) + 2
        num = (_np.bincount(i0, wts * (1 - f), n) + _np.bincount(i0 + 1, wts * f, n))
        den = (_np.bincount(i0, 1 - f, n) + _np.bincount(i0 + 1, f, n))
        return (num / _np.maximum(den, 1e-9)).astype(_np.float32)

    def periodic(proj):
        """(period, strength): the first strong peak of the projection's
        autocorrelation, after broad changes (a card's edge, a patch boundary)
        are taken out.  The pitch is the first peak; its harmonics follow."""
        n = proj.size
        hi_ = min(n // 2, int(max_period_px))
        if hi_ <= lo + 2:
            return None, -1.0
        base = _cv2.GaussianBlur(proj.reshape(1, -1).astype(_np.float32), (0, 0),
                                 max(2.0, hi_ / 3.0)).ravel()
        p = proj - base
        p = p - p.mean()
        denom = float((p * p).sum()) or 1.0
        ac = _np.correlate(p, p, mode="full")[n - 1:] / denom
        peaks = [k for k in range(max(lo, 1), hi_) if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1]]
        if not peaks:
            return None, -1.0
        top = max(ac[k] for k in peaks)
        if top <= 0.05:
            return None, -1.0
        k = next(k for k in peaks if ac[k] >= 0.6 * top)
        period = float(k)
        y0, y1_, y2_ = float(ac[k - 1]), float(ac[k]), float(ac[k + 1])
        d = y0 - 2 * y1_ + y2_
        if abs(d) > 1e-9:
            period += 0.5 * (y0 - y2_) / d                 # sub-pixel, by a parabola
        # A ruler is read along many ticks: across a long, thin box only a few
        # repeats fit, and chance structure there must not win.
        repeats = n / max(period, 1e-9)
        return period, float(ac[k]) * min(1.0, repeats / 8.0)

    # The ruler's direction: the projection that is most periodic.  (The one
    # that merely varied most was the card's own edge, not its ticks.)
    # A box drawn along a ruler is long and thin: the ruler runs its length.
    # Searching only near that keeps JPEG's 8-pixel block pattern, which is
    # periodic in every direction, from winning across a narrow box.
    if max(hh_, ww_) >= 2 * min(hh_, ww_):
        along = 90.0 if hh_ > ww_ else 0.0
        angles = _np.arange(along - 30.0, along + 30.01, 2.0)
    else:
        angles = _np.arange(0.0, 180.0, 2.0)
    best = (None, -1.0, None, None)
    for ang in angles:
        proj = project(ang)
        per, strength = periodic(proj)
        if per is not None and strength > best[1]:
            best = (ang, strength, proj, per)
    angle, conf, proj, period = best
    if proj is None or period is None:
        raise ValueError("no tick pattern could be found in that region")

    # Then to a quarter of a degree, by sharpness: projected exactly along the
    # tick marks, each tick stays a narrow peak; a little off, it smears.  (How
    # periodic the projection looked chose angles up to 16° off the ruler.)
    def sharpness(pr):
        hp = pr - _cv2.GaussianBlur(pr.reshape(1, -1).astype(_np.float32), (0, 0),
                                    max(2.0, period * 2.0)).ravel()
        return float(_np.var(hp))
    best_s = sharpness(proj)
    for ang in _np.arange(angle - 8.0, angle + 8.01, 0.25):
        if angles.size < 90 and not (angles.min() <= ang <= angles.max()):
            continue
        pr = project(ang)
        sh = sharpness(pr)
        if sh > best_s:
            best_s, angle, proj = sh, float(ang), pr
    per, strength = periodic(proj)
    if per is not None:
        period, conf = per, strength
    # Then every tick itself: each tick is a peak of edge strength along the
    # ruler (both of its edges, merged by a blur of a fifth of the pitch);
    # numbered by its distance from the first in pitches and fitted with a
    # straight line, the slope is the pitch from all of them at once.  On a
    # QPcard's millimetre scale this agrees with ticks fitted by hand to 0.1 %.
    pr = proj - _cv2.GaussianBlur(proj.reshape(1, -1).astype(_np.float32), (0, 0),
                                  max(2.0, period * 2.0)).ravel()
    pr = _cv2.GaussianBlur(pr.reshape(1, -1).astype(_np.float32), (0, 0),
                           max(0.8, period * 0.2)).ravel().astype(_np.float64)
    half = max(1, int(period * 0.35))
    cand = [i for i in range(half, pr.size - half)
            if pr[i] == pr[i - half:i + half + 1].max() and pr[i] > 0]
    if len(cand) >= 4:
        strong = _np.percentile([pr[i] for i in cand], 30)
        cand = [i for i in cand if pr[i] >= strong and 0 < i < pr.size - 1]
        pos = []
        for i in cand:
            d = pr[i - 1] - 2 * pr[i] + pr[i + 1]
            pos.append(i + (0.5 * (pr[i - 1] - pr[i + 1]) / d if abs(d) > 1e-12 else 0.0))
        pos = _np.asarray(pos)
        if pos.size >= 4:
            idx = _np.round((pos - pos[0]) / period)
            keep = _np.abs(pos - pos[0] - idx * period) < 0.3 * period
            if keep.sum() >= 4 and _np.ptp(idx[keep]) >= 3:
                A = _np.vstack([idx[keep], _np.ones(int(keep.sum()))]).T
                period = float(_np.linalg.lstsq(A, pos[keep], rcond=None)[0][0])
    conf = float(_np.clip(conf, 0.0, 1.0))

    n_ticks = int(max(0, proj.size // max(period, 1e-6)))
    return {"period_px": period, "confidence": conf,
            "angle_deg": float(angle), "n_ticks": n_ticks}
