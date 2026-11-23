"""Quality-control products derived from data the solve already produced.

Nothing here computes a new reconstruction. Each function takes arrays that
`run_processing` has already stored and asks a question about them that the
program could always have answered and never did.
"""
from __future__ import annotations

import numpy as np


def integrator_disagreement(depth_a: np.ndarray, depth_b: np.ndarray):
    """How far two independently integrated height fields disagree.

    ``run_processing`` solves the surface twice — Poisson (``depth_vis_p``)
    and Frankot-Chellappa (``depth_vis_fc``) — stores both, and never compares
    them. Where two independent integrators of the same normal field disagree
    is where the relief is least trustworthy, and that is a different question
    from the existing uncertainty map, which measures how well each *pixel*
    fits the lighting model rather than how well the *integration* is
    determined.

    **The affine fit is the whole trick.** Both maps are normalised to 0–1
    independently, and an integrated height field is only defined up to an
    additive constant anyway, so subtracting them raw measures the
    normalisation and the arbitrary datum, not the disagreement — on a clean
    capture that produces a large, smooth, entirely meaningless difference.
    Least squares removes the gain and offset first, leaving only the shape
    the two solvers genuinely differ on.

    Returns ``(diff, stats)``. ``diff`` is the signed residual in the units of
    *depth_a*; ``stats`` carries ``rms``, ``p95``, ``max`` and the fitted
    ``gain``/``offset``, so a caller can report the fit as well as the result.
    """
    a = np.asarray(depth_a, dtype=np.float64)
    b = np.asarray(depth_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    good = np.isfinite(a) & np.isfinite(b)
    if not good.any():
        raise ValueError("no finite pixels common to both height maps")

    av, bv = a[good], b[good]
    # Solve  av ≈ gain * bv + offset  in the least-squares sense.
    denom = float(((bv - bv.mean()) ** 2).sum())
    if denom < 1e-12:
        # The second map is flat: no gain is recoverable, so match the datum
        # only and let the difference speak for itself.
        gain, offset = 1.0, float(av.mean() - bv.mean())
    else:
        gain = float(((bv - bv.mean()) * (av - av.mean())).sum() / denom)
        offset = float(av.mean() - gain * bv.mean())

    diff = np.full(a.shape, np.nan, dtype=np.float64)
    diff[good] = av - (gain * bv + offset)

    d = diff[good]
    stats = {
        "rms": float(np.sqrt(np.mean(d ** 2))),
        "p95": float(np.percentile(np.abs(d), 95)),
        "max": float(np.abs(d).max()),
        "gain": gain,
        "offset": offset,
        "pixels": int(good.sum()),
        # Fraction of the *range of the surface itself*, which is what makes
        # the number interpretable: 0.5% of relief is agreement, 20% is not.
        "range": float(av.max() - av.min()),
    }
    stats["rms_frac"] = (stats["rms"] / stats["range"]) if stats["range"] > 1e-12 else 0.0
    return diff, stats


def interframe_drift(frames, roi=None, max_frames=64):
    """Measure camera or object movement between frames of a stack.

    Photometric stereo assumes the camera and object are fixed and only the
    light moves. Every solver here relies on it, the Read Me instructs it, and
    nothing has ever checked it — so a tripod nudged between frame 3 and 4
    produces smeared normals that look like a bad surface rather than a bad
    capture, and the only way to find out was to distrust the result.

    Each frame is phase-correlated against the stack's median frame (three
    or more frames; against the first for two), on edge strength, and the
    shift all frames share is removed. A small move reads low (a 3 px move as
    about 2 px), so the magnitude is a guide; the frame it names is reliable.
    Phase correlation is used
    rather than feature matching because it is global, sub-pixel, and indifferent
    to the thing that actually changes between these frames: the lighting. A
    Hann window suppresses the edge discontinuity that would otherwise dominate
    the transform.

    Parameters
    ----------
    frames : sequence of 2-D arrays
        Greyscale frames, all the same shape.
    roi : (x1, y1, x2, y2) or None
        Restrict to a region. Worth using when part of the scene legitimately
        moves — a hand, a turning light — or to concentrate on high-contrast
        detail where correlation is sharpest.
    max_frames : int
        Cap for very long stacks; frames are sampled evenly across the run so a
        drift that develops late is still seen.

    Returns
    -------
    dict with:
        ``shifts``   list of (index, dx, dy, magnitude) in pixels
        ``worst``    (index, magnitude) of the largest displacement
        ``median``   median magnitude across the stack
        ``verdict``  'clean' | 'marginal' | 'moved'
    """
    import numpy as _np
    try:
        import cv2 as _cv2
    except Exception:                       # pragma: no cover
        raise RuntimeError("OpenCV is required for drift detection")

    if frames is None or len(frames) < 2:
        raise ValueError("need at least two frames to compare")

    idx = list(range(len(frames)))
    if len(idx) > max_frames:
        step = len(idx) / float(max_frames)
        idx = sorted({int(i * step) for i in range(max_frames)})

    def _prep(a):
        f = _np.asarray(a)
        if f.ndim == 3:
            f = f[..., :3].mean(axis=2)
        f = f.astype(_np.float32)
        if roi is not None:
            x1, y1, x2, y2 = roi
            f = f[y1:y2, x1:x2]
        # Compare edges, not brightness.  Under raking light the shading and
        # the shadows move with the lamp, and raw phase correlation read that
        # as the camera moving: 46 px "drift" on a gilt binding whose frames
        # are within 6 px.  Edge strength follows the object's structure,
        # which stays put while the light moves.
        f = _cv2.GaussianBlur(f, (0, 0), 1.0)
        f = _np.hypot(_cv2.Sobel(f, _cv2.CV_32F, 1, 0), _cv2.Sobel(f, _cv2.CV_32F, 0, 1))
        f = _cv2.GaussianBlur(f, (0, 0), 1.5)
        f -= float(f.mean())
        s = float(f.std())
        return f / s if s > 1e-6 else f

    prepped = {i: _prep(frames[i]) for i in idx}
    first = prepped[idx[0]]
    if min(first.shape[:2]) < 16:
        raise ValueError("region is too small to correlate")
    for i in idx:
        if prepped[i].shape != first.shape:
            raise ValueError(f"frame {i} is {prepped[i].shape}, first frame is {first.shape}")
    win = _cv2.createHanningWindow((first.shape[1], first.shape[0]), _cv2.CV_32F)
    # Each frame against the stack's median, which is lit from every side,
    # rather than against frame 1, lit from one: that read a side light's
    # shadows as movement.  A shift every frame shares is the median's own
    # offset, not movement, and is taken out.
    if len(idx) >= 3:
        ref = _np.median(_np.stack([prepped[i] for i in idx]), axis=0).astype(_np.float32)
        raw = [(i, *_cv2.phaseCorrelate(ref, prepped[i], win)[0]) for i in idx]
        mx = float(_np.median([r[1] for r in raw]))      # median: one moved frame
        my = float(_np.median([r[2] for r in raw]))      # does not drag the rest
        raw = [(i, dx - mx, dy - my) for i, dx, dy in raw]
    else:
        raw = [(idx[0], 0.0, 0.0)] + [(i, *_cv2.phaseCorrelate(first, prepped[i], win)[0])
                                      for i in idx[1:]]
    shifts = [(i, float(dx), float(dy), float(_np.hypot(dx, dy))) for i, dx, dy in raw]

    mags = [m for _, _, _, m in shifts] or [0.0]
    worst = max(shifts, key=lambda s: s[3]) if shifts else (0, 0.0, 0.0, 0.0)
    med = float(_np.median(mags))

    # Thresholds in pixels. Below half a pixel nothing downstream can tell;
    # beyond about two, normals visibly smear at full resolution. The band
    # between is worth a look rather than an alarm.
    peak = worst[3]
    verdict = "clean" if peak < 0.5 else ("marginal" if peak < 2.0 else "moved")

    return {"shifts": shifts, "worst": (worst[0], worst[3]),
            # frames compared beside the first: callers add one for the frame count
            "median": med, "verdict": verdict, "n_compared": len(idx) - 1}
