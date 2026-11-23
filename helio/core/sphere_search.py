"""Finding the chrome ball(s) in a capture automatically: candidate circles from the
frames, scored by a highlight that moves with the light (helio.core.sphere does the
rest: the rim fit, the highlights, the calibration).

Moved out of helio.core.processing (November 2026), which still re-exports every name.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from helio.core.frame_read import read_frame

_log = logging.getLogger(__name__)



def detect_sphere_candidates(
    img_paths: list[str],
    full_h: int,
    full_w: int,
    want_two: bool = False,
) -> list[tuple[float, float, float]]:
    """Locate chrome calibration spheres. Returns [(cx, cy, r), ...] full-res.

    A chrome sphere's signature is a compact, near-saturated SPECULAR HIGHLIGHT
    sitting in DARK surroundings (black background or grey foam) — unlike the
    large bright object being captured, whose highlights are embedded in bright
    surroundings. We seed on those isolated highlights and fit the ball disc
    locally around each seed.

    The search radius is HARD-CAPPED to a small fraction of the image, so no
    giant spurious circle can be produced. That cap is the whole point: the
    previous implementation ran Hough over the entire frame with maxRadius at
    40% of the image and then took ``max()`` of every radius estimate, so a
    spurious circle always won. On a 960x1280 manuscript capture it returned
    r=445 for a sphere whose true radius is 28 px, and the resulting ROI
    covered the blown-out page instead of the ball.

    Two more faults showed on a gilt binding photographed with the ball on a
    light grey card (100 MP frames): the card failed the dark-surroundings
    test, the gold tooling on dark leather passed it, and a patch of tooling
    1100 px across was returned as the ball; and every frame was decoded
    whole, three times over, which took 20 s.  Now:

    * frames are decoded at reduced size (JPEG's own DCT scaling) wherever
      the ball stays at least 40 px across;
    * circles are also looked for in the median frame (a ball is a clean
      disc in every frame), not only round isolated highlights;
    * every candidate must behave as a chrome ball across the frames
      (``_ball_evidence``): in most frames one compact highlight inside the
      disc, few separate specks around it, and the highlight in a different
      place from frame to frame as the light moves.  Gilding, rosettes and
      glints fail this; a candidate that fails is never returned, so a
      capture with no ball says so instead of producing wrong lights.
    """
    from helio.core.sphere import read_many, reduction_for, refine
    DET = 1000.0
    sc = min(1.0, DET / max(full_h, full_w))
    k_det = reduction_for((full_w, full_h), int(DET))
    dw_, dh_ = int(full_w * sc), int(full_h * sc)

    # Max-projection across a few frames: the highlight sweeps the sphere over
    # the sequence, reinforcing the ball region.
    stack = []
    for g in read_many(img_paths[:min(len(img_paths), 8)], k_det):
        if g is None:
            continue
        if g.shape[:2] != (dh_, dw_):
            g = cv2.resize(g, (dw_, dh_), interpolation=cv2.INTER_AREA)
        stack.append(g)
    if not stack:
        return []
    d = np.maximum.reduce(stack).astype(np.uint8)
    dh, dw = d.shape
    md = min(dh, dw)
    r_max = int(md * 0.10)               # hard cap — balls are small
    yy, xx = np.ogrid[:dh, :dw]

    # ── 1. Seed on isolated specular highlights ──────────────────────────
    thr = max(235, int(np.percentile(d, 99.7)))
    _, bw = cv2.threshold(d, thr, 255, cv2.THRESH_BINARY)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    seeds = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 3 or area > (r_max * r_max * math.pi):
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        brad = max(2.0, max(stats[i, cv2.CC_STAT_WIDTH],
                            stats[i, cv2.CC_STAT_HEIGHT]) / 2.0)
        x0, x1_ = max(0, int(cx - brad * 4) - 1), min(dw, int(cx + brad * 4) + 2)
        y0, y1_ = max(0, int(cy - brad * 4) - 1), min(dh, int(cy + brad * 4) + 2)
        inner = (xx[:, x0:x1_] - cx) ** 2 + (yy[y0:y1_, :] - cy) ** 2
        ring = (inner <= (brad * 4)**2) & (inner > (brad * 2)**2)
        ring_mean = float(d[y0:y1_, x0:x1_][ring].mean()) if ring.sum() > 0 else 255.0
        # Isolated in dark surroundings (black bg ~30-60, foam ~90-110);
        # the bright object's highlights have ring means ~180+ → rejected.
        if ring_mean <= 125:
            seeds.append((ring_mean, cx, cy))

    # ── 2. Fit the ball disc locally around each seed ────────────────────
    def _fit_disc(cx, cy):
        win = int(md * 0.11)
        y1, y2 = max(0, int(cy - win)), min(dh, int(cy + win))
        x1, x2 = max(0, int(cx - win)), min(dw, int(cx + win))
        sub = d[y1:y2, x1:x2]
        if sub.size == 0:
            return None
        med = float(np.median(sub))
        fg = (sub > max(med + 30, 110)).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        nn, ll, _ss, cc = cv2.connectedComponentsWithStats(fg, 8)
        if nn <= 1:
            return None
        sy = min(max(int(cy - y1), 0), sub.shape[0] - 1)
        sx = min(max(int(cx - x1), 0), sub.shape[1] - 1)
        lbl = int(ll[sy, sx])
        if lbl == 0:                     # seed not on a fg pixel — take nearest
            best, bd = 0, 1e9
            for k in range(1, nn):
                dd = math.hypot(cc[k][0] - sx, cc[k][1] - sy)
                if dd < bd:
                    bd, best = dd, k
            lbl = best
        if lbl == 0:
            return None
        cnts, _h = cv2.findContours((ll == lbl).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        (ex, ey), er = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
        return ex + x1, ey + y1, er

    # (score, cx, cy, search radius, disc radius); lower score is better
    cands = []
    for (ring_mean, cx, cy) in seeds[:400]:
        fit = _fit_disc(cx, cy)
        if fit is not None:
            ex, ey, er = fit
            rr = min(max(er * 1.3, md * 0.03), r_max)
            ocx, ocy = ex, ey
        else:
            rr = md * 0.045
            ocx, ocy = cx, cy
            er = rr
        # Rank: darker surroundings = more ball-like; a larger fitted disc
        # gently breaks ties in favour of a real ball over a tiny glint.
        score = ring_mean - min(er, r_max) * 0.15
        cands.append((score, ocx, ocy, rr, min(er, r_max)))

    # ── 3. Circles in the median frame: the ball on any background ───────
    med_img = np.median(np.stack(stack), axis=0).astype(np.uint8)
    r_min = max(5, int(md * 0.008))
    circles = cv2.HoughCircles(cv2.GaussianBlur(med_img, (0, 0), 1.2), cv2.HOUGH_GRADIENT,
                               dp=1.0, minDist=2 * r_min, param1=90, param2=20,
                               minRadius=r_min, maxRadius=r_max)
    # A circle in the median frame with highlight seeds inside it is that
    # ball: judged as the circle, ranked by its best seed.  Seeds outside
    # every circle are judged on their own disc and search window.
    n_seeded = len(cands)
    owner = [None] * n_seeded
    hough = []
    if circles is not None:
        circ = sorted((tuple(map(float, c)) for c in circles[0][:300]), key=lambda c: c[2])
        for (hx, hy, hr) in circ:                       # smallest first: a seed's own ball
            inside = [j for j in range(n_seeded) if owner[j] is None
                      and math.hypot(hx - cands[j][1], hy - cands[j][2]) < hr]
            for j in inside:
                owner[j] = len(hough)
            best = min((cands[j][0] for j in inside), default=200.0)
            hough.append((best, hx, hy, hr))

    # ── 4. Keep only what behaves as a chrome ball ───────────────────────
    judged = []
    for (best, hx, hy, hr) in hough:
        if _ball_evidence(stack, hx, hy, hr) >= 0.3:
            judged.append((1.0, best, hx, hy, hr * 1.43 / 1.1))      # 1: a circle, near
    for j, (score, ocx, ocy, rr, er) in enumerate(cands):
        if owner[j] is not None:
            continue
        if max(_ball_evidence(stack, ocx, ocy, r) for r in (er, rr)) >= 0.3:
            judged.append((0.0, score, ocx, ocy, rr))
    # Of two circles that pass, one round the other, the inner is the ball:
    # a wider circle about it passes only because the ball is inside it.
    judged = [t for t in judged
              if not any(o is not t and o[4] < t[4] / 1.3
                         and math.hypot(o[2] - t[2], o[3] - t[3]) < 0.5 * t[4]
                         for o in judged)]
    judged.sort(key=lambda t: t[1])

    want = 2 if want_two else 1
    found: list[tuple[float, float, float]] = []
    near: list[bool] = []
    for (kind, _score, ocx, ocy, rr) in judged:
        ox, oy, orr = ocx / sc, ocy / sc, (rr / sc) * 1.1   # 10% padding
        if any(math.hypot(ox - fx, oy - fy) < (orr + fr) * 0.9
               for (fx, fy, fr) in found):
            continue
        found.append((ox, oy, orr))
        near.append(kind == 1.0)
        if len(found) >= want:
            break
    detect_sphere_candidates.notes = []
    if not found:
        return []
    # Each circle so far is a search window: the blob's enclosing circle
    # padded by 1.3 x 1.1.  Used as the ball's radius it put every light
    # 6-22 degrees too high (KD-020), so measure the ball's own rim and
    # return that (helio.core.sphere.refine).  The frames are read once, at
    # the smallest size that keeps the smallest ball 40 px across.
    r_small = min(orr / 1.43 for (_x, _y, orr) in found)
    k_fit = 1
    while k_fit < 8 and r_small / (k_fit * 2) >= 40:
        k_fit *= 2
    frames = read_many(list(img_paths)[:8], k_fit)
    fits = [refine(img_paths, ox, oy, orr / 1.43, reduce=k_fit, frames=frames, near=nr)
            for (ox, oy, orr), nr in zip(found, near)]
    # What the rim fit had to say about each ball ("check the circle"), for
    # callers that show it; the return value stays (cx, cy, r) tuples.
    detect_sphere_candidates.notes = [f.note for f in fits]
    return [f.as_tuple() for f in fits]



def _ball_evidence(stack, cx: float, cy: float, r: float) -> float:
    """How far the frames behave as a chrome ball of radius *r* at (cx, cy),
    0 (not a ball) to 1.

    A mirror ball shows each lamp as one small highlight, which moves as the
    lamp moves; what surrounds the ball has no such moving point.  So in a
    frame that counts, the disc holds one or two compact blobs at 85% of
    its peak and a small share of its area, and the ring round it holds
    few separate specks (a bright page is one large region, which is
    allowed; gilding is many specks, which is not).  Over the frames, the
    highlights must spread: one glint in a fixed place is not a ball.  And
    the disc must be a body, darker or lighter than its surroundings in the
    median frame: a speck on plain leather is neither."""
    if r < 2.0:
        return 0.0
    h, w = stack[0].shape[:2]
    R = r * 1.9
    x1, x2 = max(0, int(cx - R)), min(w, int(cx + R) + 1)
    y1, y2 = max(0, int(cy - R)), min(h, int(cy + R) + 1)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    yy, xx = np.mgrid[y1:y2, x1:x2]
    dist = np.hypot(xx - cx, yy - cy)
    disc = dist <= r * 0.92
    ring = (dist >= r * 1.15) & (dist <= r * 1.9)
    n_disc = int(disc.sum())
    if n_disc < 12 or not ring.any():
        return 0.0
    # The ball is a body: in the median frame (highlights washed out) it is
    # darker or lighter than what is round it.  A speck on plain leather is not.
    med = np.median(np.stack([g[y1:y2, x1:x2] for g in stack]), axis=0)
    if abs(float(np.median(med[disc])) - float(np.median(med[ring]))) < 8.0:
        return 0.0
    disc_u8 = disc.astype(np.uint8)
    good, pos = 0, []
    for g in stack:
        sub = g[y1:y2, x1:x2]
        vals = sub[disc]
        vmax, vmed = float(vals.max()), float(np.median(vals))
        if vmax < 150 or vmax < vmed + 60:
            continue                                   # no highlight in this frame
        t = max(vmax * 0.85, vmed + 0.6 * (vmax - vmed))
        hi = (sub >= t).astype(np.uint8)
        nd, _l, st, ce = cv2.connectedComponentsWithStats(hi * disc_u8, 8)
        if nd <= 1 or nd - 1 > 2:
            continue
        if int(st[1:, cv2.CC_STAT_AREA].sum()) > 0.05 * n_disc:
            continue
        nr, _l2, st2, _c2 = cv2.connectedComponentsWithStats(hi * ring.astype(np.uint8), 8)
        specks = int(np.sum(st2[1:, cv2.CC_STAT_AREA] <= max(2.0, 0.05 * n_disc))) if nr > 1 else 0
        if specks > 2:
            continue
        b = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        pos.append((ce[b][0], ce[b][1]))
        good += 1
    n = len(stack)
    if good < max(3, int(math.ceil(0.5 * n))):
        return 0.0
    P = np.asarray(pos, np.float64)
    spread = float(np.sqrt(((P - P.mean(0)) ** 2).sum(1).mean())) / r
    if spread < 0.06:
        return 0.0
    return (good / n) * min(1.0, spread / 0.15)



def auto_detect_sphere(
    img_paths: list[str],
    roi: tuple[int, int, int] | None = None,
) -> np.ndarray | None:
    """Auto-detect chrome sphere and extract LP vectors.

    If *roi* (cx, cy, r) is provided, uses that directly (RTIBuilder-style).
    Otherwise falls back to the automatic grid-patch + Hough algorithm.

    Parameters
    ----------
    img_paths : list of str
        Paths to the captured images (at least 3 required).
    roi : (cx, cy, r) tuple or None
        User-drawn sphere ROI in full-resolution coordinates.

    Returns
    -------
    np.ndarray or None
        (N, 3) float32 array of normalised light vectors, or None if detection fails.
    """
    import sys
    import cv2 as _cv2
    from helio.core.sphere import Sphere, calibrate

    IMG_EXT = ('.jpg', '.jpeg', '.tif', '.tiff', '.png', '.bmp')
    paths = [p for p in img_paths if p.lower().endswith(IMG_EXT)]
    if len(paths) < 3:
        return None
    if roi is not None:
        # RTIBuilder-style: the circle the user drew on the ball's outline.
        sphere = Sphere(*(float(v) for v in roi))
    else:
        first = read_frame(paths[0], _cv2.IMREAD_GRAYSCALE)
        if first is None:
            return None
        full_h, full_w = first.shape
        cands = detect_sphere_candidates(paths, full_h, full_w, want_two=True)
        if not cands:
            print("[AutoSphere] No chrome sphere found in any sample frame", file=sys.stderr)
            return None
        sphere = Sphere(*cands[0])
        print(f"[AutoSphere] Sphere (full-res): cx={sphere.cx:.1f} cy={sphere.cy:.1f} "
              f"r={sphere.r:.1f}  image={full_w}x{full_h}", file=sys.stderr)
    # One light per frame, in order (helio.core.sphere.calibrate).  A frame
    # without a usable highlight used to be skipped, and lights within 5
    # degrees merged, so the list came back short and later frames took
    # their neighbours' lights (KD-020).  Now an incomplete calibration is
    # refused rather than misaligned.
    cal = calibrate(paths, sphere)
    if roi is None and len(cands) > 1:
        # Two real balls: each frame's light is their mean (KD-056), as the
        # new interface's search does.
        from helio.core.sphere import with_second_ball
        cal = with_second_ball(paths, cal, Sphere(*cands[1])) or cal
    if not cal.complete():
        print(f"[AutoSphere] {'; '.join(cal.notes) or 'calibration incomplete'}", file=sys.stderr)
        return None
    print(f"[AutoSphere] Extracted {len(cal.lights)} light vectors", file=sys.stderr)
    return cal.lights.astype(np.float32)
