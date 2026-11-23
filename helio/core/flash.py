"""Even out a hand-held flash from a grey reference in the shot (CAL-003).

Photometric stereo assumes every light is equally strong.  A flash carried
round the object by hand is not: on six bindings shot that way the
strongest frame was 1.8 to 2.6 times the weakest, and the solve read the
brighter frames as the surface turning towards them.  Flat grey cards came
out tilted 7-18°.

The chrome ball gives each light's direction but not its strength.  A flat,
matte, grey reference lying in the shot gives the strength: Lambertian, its
brightness in frame i is  strength_i · albedo · (n · L_i)  with n straight
up, so  strength_i ∝ brightness_i / L_i,z.  Each frame is then scaled to the
weakest flash's level (never brighter, so nothing clips).  On the bindings
the cards came out within 1-3° of flat and five of six book covers flatter.

The reference must lie flat, face the camera, be lit by every flash and
never clip: a grey card (an X-Rite or QPcard grey patch, or any matte
mid-grey) placed beside the object.  ``find_grey_reference`` looks for one;
a person can also mark it.
"""
from __future__ import annotations

from helio.core.frame_read import read_frame

import math

import numpy as np

MIN_LZ = 0.05            # a light this close to the horizon says nothing about strength


def box_text(box) -> str:
    return ",".join(str(int(v)) for v in box)


def parse_box(text: str):
    """(x1, y1, x2, y2) from 'x1,y1,x2,y2', or None."""
    try:
        v = [int(float(t)) for t in str(text).split(",")]
    except (TypeError, ValueError):
        return None
    if len(v) != 4 or v[2] - v[0] < 4 or v[3] - v[1] < 4:
        return None
    return tuple(v)


def find_grey_reference(paths, target: int = 1000) -> dict | None:
    """The largest flat, matte, neutral mid-grey patch present in every frame.

    Returns {'box': (x1, y1, x2, y2) in full-size pixels, 'level' 0-1,
    'chroma', 'side' px} or None.  The patch must be grey (no colour),
    mid-toned (12-80 %), even inside (a matte card, not texture), never
    clipped and never in shadow in any frame."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    from helio.core.sphere import image_size, reduction_for
    paths = list(paths)[:8]
    if len(paths) < 3:
        return None
    size = image_size(paths[0])
    if not size:
        return None
    k = reduction_for(size, target)
    flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
            4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[k]
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as ex:
        frames = [f for f in ex.map(lambda p: read_frame(p, flag), paths) if f is not None]
    if len(frames) < 3:
        return None
    h, w = frames[0].shape[:2]
    frames = [f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
              for f in frames]
    F = np.stack(frames).astype(np.float32) / 255.0                 # (n, h, w, 3) BGR
    grey_each = F.mean(axis=3)                                        # (n, h, w)
    med = np.median(F, axis=0)
    g = med.mean(axis=2)
    chroma = (med.max(axis=2) - med.min(axis=2)) / (g + 1e-3)
    r = max(5, int(round(min(h, w) * 0.012)))
    mean = cv2.blur(g, (r, r))
    sq = cv2.blur(g * g, (r, r))
    cov = np.sqrt(np.maximum(sq - mean * mean, 0)) / (mean + 1e-3)
    mask = ((cov < 0.03) & (chroma < 0.10) & (g > 0.12) & (g < 0.80)
            & (grey_each.max(axis=0) < 0.97) & (grey_each.min(axis=0) > 0.03))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = _card_like(mask, g, cv2)
    if not mask.any():
        return None
    # The frame's edge bounds a patch too (a card cut off by the frame).
    dt = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    # Large, and near mid-grey: furthest from clipping under a strong flash
    # and from the shadows under a weak one.
    score = dt * np.clip(1.0 - np.abs(g - 0.45) / 0.45, 0.2, 1.0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    d = float(dt[y, x])
    if d < max(4.0, 0.006 * min(h, w)):
        return None
    half = int(max(3, min(d * 0.7, 0.03 * min(h, w))))
    x1, y1, x2, y2 = x - half, y - half, x + half, y + half
    # Even inside in (nearly) every frame, not only on average.
    covs = []
    for gi in grey_each:
        p = gi[y1:y2, x1:x2]
        covs.append(float(p.std() / (p.mean() + 1e-3)))
    if np.mean(np.asarray(covs) < 0.06) < 0.75:
        return None
    return {"box": (x1 * k, y1 * k, x2 * k, y2 * k), "level": float(g[y, x]),
            "chroma": float(chroma[y, x]), "side": int(2 * half * k)}


# A flat area must be lit well above the dark: black velvet round a binding
# (about 6 % of white) reflects nothing like a matte surface, and on Binding 2
# it made the cover 4.2° out where the grey card alone left 1.7°.
FLAT_MIN_LEVEL = 0.08


def check_reference(paths, box, lights=None, min_level: float = 0.0) -> dict:
    """Is a hand-marked patch usable as the grey reference?

    *box* (x1, y1, x2, y2) in full-size pixels.  Each frame is read reduced
    (never so far that the patch is under 8 px across) and the patch judged:
    it must not clip or sink into shadow in any frame, since the strength
    is read from it in each.  Returns {'ok', 'problem' (why not), 'warning'
    (usable, but…), 'spread' when *lights* match the frames}."""
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    paths = list(paths)
    x1, y1, x2, y2 = (int(v) for v in box)
    side = min(x2 - x1, y2 - y1)
    if side < 8:
        return {"ok": False, "problem": "the box is too small"}
    k = next(r for r in (8, 4, 2, 1) if side / r >= 8)
    flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
            4: cv2.IMREAD_REDUCED_GRAYSCALE_4, 8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[k]

    def patch(p):
        g = read_frame(p, flag)
        if g is None:
            return None
        h, w = g.shape[:2]
        a, b, c, d = x1 // k, y1 // k, min(w, -(-x2 // k)), min(h, -(-y2 // k))
        if c - a < 4 or d - b < 4:
            return None
        return g[b:d, a:c].astype(np.float32)
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as ex:
        patches = list(ex.map(patch, paths))
    if any(q is None for q in patches):
        return {"ok": False, "problem": "the box is outside some of the photographs"}
    n = len(patches)
    clipped = [i + 1 for i, q in enumerate(patches) if np.mean(q >= 0.97 * 255) > 0.5]
    dark = [i + 1 for i, q in enumerate(patches) if np.mean(q <= 0.02 * 255) > 0.5]

    def frames(ix):
        return ("frame " if len(ix) == 1 else "frames ") + ", ".join(map(str, ix))
    if len(clipped) + len(dark) > n - max(3, math.ceil(0.75 * n)):
        bits = ([f"it is white (clipped) in {frames(clipped)}"] if clipped else []) + \
               ([f"it is black in {frames(dark)}"] if dark else [])
        return {"ok": False, "problem": " and ".join(bits)}
    level = float(np.median([np.median(q) for q in patches])) / 255.0
    if min_level and level < min_level:
        return {"ok": False, "problem": f"it is too dark to measure the light on ({level * 100:.0f} % "
                                        f"of white; black cloth and shadow reflect too little): "
                                        f"use a lit, matte, mid-toned surface"}
    cov = [float(q.std() / (q.mean() + 1e-3)) for q in patches]
    out = {"ok": True, "problem": "", "warning": ""}
    if np.median(cov) > 0.08:
        out["warning"] = ("The patch is not even: if it is not a flat, matte grey card, the "
                          "correction may be off.")
    elif clipped or dark:
        out["warning"] = f"Not readable in {frames(clipped + dark)}: those frames are left as they are."
    if lights is not None and len(lights) == n:
        stack = np.stack([np.full((4, 4), float(np.median(q)), np.float32) for q in patches])
        st = strengths(stack, (0, 0, 4, 4), lights)
        if st is not None:
            out["spread"] = st["spread"]
    return out


def _card_like(mask, g, cv2):
    """Keep only the even grey regions that look like a card or a chart's
    patch: a modest, nearly rectangular region, clearly lighter or darker
    than what surrounds it.  A flat, featureless stretch of the object is
    none of these, and is not taken for a reference."""
    h, w = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, bool)
    side_min = 2 * max(4.0, 0.006 * min(h, w))
    for i in range(1, n):
        x, y, ww, hh, area = (int(v) for v in stats[i])
        if area < side_min ** 2 or area > 0.15 * h * w:
            continue
        comp = (lab[y:y + hh, x:x + ww] == i).astype(np.uint8)
        cnts, _h = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        (_cx, _cy), (rw, rh), _a = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
        if rw < 1 or rh < 1 or area / (rw * rh) < 0.8 or max(rw, rh) / min(rw, rh) > 8:
            continue
        pad = max(3, int(0.15 * min(ww, hh)))
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(w, x + ww + pad), min(h, y + hh + pad)
        ring = lab[y0:y1, x0:x1] != i
        inside = float(np.median(g[lab == i]))
        outside = float(np.median(g[y0:y1, x0:x1][ring])) if ring.any() else inside
        if abs(outside - inside) / max(inside, 1e-3) < 0.12:
            continue                                  # no edge: part of a larger surface
        keep[i] = True
    return keep[lab].astype(np.uint8)


def strengths(stack, box, lights) -> dict | None:
    """Each frame's light strength from the reference in *box* (stack pixels).

    *stack* (n, h, w) as the solve sees it; *lights* (n, 3), +Z towards the
    camera.  Returns {'strength' (n,) relative to their median, 'factor' (n,)
    to multiply each frame by (the weakest flash's level, all <= 1),
    'spread' strongest/weakest, 'used' frames measured} or None when the
    reference cannot be read (outside the frame, clipped, in shadow)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    n, h, w = stack.shape[:3]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    L = np.asarray(lights, np.float64)
    if L.shape[0] != n:
        return None
    top = 255.0 if float(np.max(stack[:, y1:y2, x1:x2])) > 1.5 else 1.0
    vals = np.full(n, np.nan)
    for i in range(n):
        p = np.asarray(stack[i, y1:y2, x1:x2], np.float64).ravel()
        ok = (p < 0.97 * top) & (p > 0.02 * top)
        if ok.mean() > 0.5:
            vals[i] = float(np.median(p[ok]))
    lz = L[:, 2]
    good = np.isfinite(vals) & np.isfinite(lz) & (lz > MIN_LZ)
    if good.sum() < max(3, math.ceil(0.75 * n)):
        return None
    s = np.ones(n)
    s[good] = vals[good] / lz[good]
    s[good] /= float(np.median(s[good]))
    s[~good] = 1.0                       # unmeasurable frames are left as they are
    factor = float(s.min()) / s
    return {"strength": s, "factor": factor, "spread": float(s.max() / s.min()),
            "used": int(good.sum())}


# ── flat areas: the flash's fall-off across the frame (CAL-004) ─────────
#
# One grey card gives each flash's strength, not the way its light falls
# off across the frame: a flash held to the left lights the left of the
# object more, and the solve reads that as the surface tilting left.  An
# area known to be flat (the board round a book, the card, a margin) is lit
# by every flash at the same angle across it, so whatever changes across it
# from frame to frame is the flash.  Each frame's light is modelled as a
# plane of brightness over the frame, fitted on the marked areas, and divided
# out before the solve.  Several areas, on either side of the object, pin
# the gradient across it.

def boxes_text(boxes) -> str:
    return ";".join(box_text(b) for b in boxes)


def parse_boxes(text: str) -> list:
    """[(x1, y1, x2, y2), …] from 'x1,y1,x2,y2;…'."""
    out = []
    for part in str(text or "").split(";"):
        b = parse_box(part)
        if b is not None:
            out.append(b)
    return out


def flat_area_planes(stack, boxes, lights, frame_size, offset=(0, 0),
                     max_samples: int = 200_000) -> dict | None:
    """Each frame's light as a plane over the frame, from areas known flat.

    *stack* (n, h, w) as the solve reads it (linear when it is linear);
    *boxes* in the photographs' pixels; *frame_size* (w, h) of the
    photographs and *offset* where *stack* starts in them (a region solve).
    Returns {'coef' (n, 3): g = c0 + c1·u + c2·v with u, v = x/W - 0.5,
    y/H - 0.5; 'factor_min' the scale that keeps every corrected value at or
    below the original (nothing is brightened, so nothing clips);
    'falloff' the largest change of a frame's light across the frame, as a
    share; 'spread' strongest over weakest frame; 'used'; 'axes' the
    gradients that could be measured}, or None."""
    S = np.asarray(stack)
    n, h, w = S.shape[:3]
    L = np.asarray(lights, np.float64)
    if L.shape[0] != n or not boxes:
        return None
    W, H = float(frame_size[0]), float(frame_size[1])
    ox, oy = offset
    ys_all, xs_all = [], []
    for x1, y1, x2, y2 in boxes:
        a, b = max(0, int(x1) - ox), max(0, int(y1) - oy)
        c, d = min(w, int(x2) - ox), min(h, int(y2) - oy)
        if c - a < 4 or d - b < 4:
            continue
        yy, xx = np.mgrid[b:d, a:c]
        ys_all.append(yy.ravel())
        xs_all.append(xx.ravel())
    if not ys_all:
        return None
    ys, xs = np.concatenate(ys_all), np.concatenate(xs_all)
    if len(ys) > max_samples:
        pick = np.random.default_rng(0).choice(len(ys), max_samples, replace=False)
        ys, xs = ys[pick], xs[pick]
    u = (xs + ox) / W - 0.5
    v = (ys + oy) / H - 0.5
    vals = S[:, ys, xs].astype(np.float64)                     # (n, m)
    top = 255.0 if float(vals.max()) > 1.5 else 1.0
    ok = (vals < 0.97 * top) & (vals > FLAT_MIN_LEVEL * top)
    lz = L[:, 2]
    good = np.isfinite(lz) & (lz > MIN_LZ) & (ok.mean(axis=1) > 0.5)
    if good.sum() < max(3, math.ceil(0.75 * n)):
        return None
    q = vals / np.where(lz > MIN_LZ, lz, 1.0)[:, None]
    qm = np.where(ok & good[:, None], q, np.nan)
    ref = np.nanmedian(qm, axis=0)                             # each pixel's own albedo
    keep_px = np.isfinite(ref) & (ref > 0)
    # Gradients that the areas can measure: a strip across the frame says
    # nothing about the gradient along it.
    axes = [True, float(np.std(u[keep_px])) > 0.04, float(np.std(v[keep_px])) > 0.04]
    A_full = np.stack([np.ones_like(u), u, v], axis=1)[:, axes]
    coef = np.zeros((n, 3))
    coef[:, 0] = 1.0
    for i in range(n):
        if not good[i]:
            continue
        m = keep_px & ok[i]
        if m.sum() < 50:
            good[i] = False
            continue
        r = q[i, m] / ref[m]
        A = A_full[m]
        sel = np.ones(len(r), bool)
        for _ in range(3):                                     # robust: drop the far misfits
            c, *_ = np.linalg.lstsq(A[sel], r[sel], rcond=None)
            res = r - A @ c
            mad = 1.4826 * float(np.median(np.abs(res[sel] - np.median(res[sel])))) + 1e-9
            sel = np.abs(res) < 3.5 * mad
        full = np.zeros(3)
        full[np.array(axes)] = c
        coef[i] = full
    if good.sum() < max(3, math.ceil(0.75 * n)):
        return None
    coef[~good] = 0.0
    coef[~good, 0] = float(np.median(coef[good, 0]))
    corners = np.array([[1, cu, cv] for cu in (-0.5, 0.5) for cv in (-0.5, 0.5)])
    g_corner = corners @ coef.T                                 # (4, n)
    if float(g_corner.min()) <= 0:
        return None                                             # a plane through zero: not light
    lo = float(g_corner.min())
    falloff = float(np.max((g_corner.max(axis=0) - g_corner.min(axis=0)) / g_corner.mean(axis=0)))
    centre = coef[:, 0]
    return {"coef": coef, "factor_min": lo, "falloff": falloff,
            "spread": float(centre[good].max() / centre[good].min()), "used": int(good.sum()),
            "axes": ("level",) + tuple(a for a, on in zip(("left-right", "top-bottom"), axes[1:]) if on)}


def flat_area_factor(planes: dict, i: int, h: int, w: int, frame_size, offset=(0, 0)):
    """The multiplier for frame *i* over an (h, w) stack: lo / g_i(x, y),
    at most 1 everywhere."""
    W, H = float(frame_size[0]), float(frame_size[1])
    c = planes["coef"][i]
    u = ((np.arange(w, dtype=np.float32) + offset[0]) / W - 0.5)
    v = ((np.arange(h, dtype=np.float32) + offset[1]) / H - 0.5)
    g = np.float32(c[0]) + np.float32(c[1]) * u[None, :] + np.float32(c[2]) * v[:, None]
    return np.float32(planes["factor_min"]) / np.maximum(g, np.float32(1e-6))


# ── when the references cannot be trusted across the object ─────────────

DISAGREE_LIMIT = 1.25          # two flat areas this far apart on a frame: fall-off


def area_disagreement(stack, boxes, lights, offset=(0, 0)) -> float | None:
    """How far apart the marked areas read the flashes: the largest ratio,
    over frames, between two areas' strength for the same flash (each area's
    strengths relative to its own median).  1 when they agree; above
    DISAGREE_LIMIT the flash's light changes across the frame, and an area
    on one side cannot stand for the other.  None with fewer than two
    readable areas."""
    S = np.asarray(stack)
    reads = []
    for x1, y1, x2, y2 in boxes:
        st = strengths(S, (x1 - offset[0], y1 - offset[1], x2 - offset[0], y2 - offset[1]), lights)
        if st is not None:
            reads.append(np.asarray(st["strength"], np.float64))
    if len(reads) < 2:
        return None
    R = np.stack(reads)
    return float(np.max(R.max(axis=0) / np.maximum(R.min(axis=0), 1e-9)))


def coverage(boxes, frame_size) -> dict:
    """Where the marked areas lie: {'x_span', 'y_span' (of their centres,
    shares of the frame), 'side' ('left', 'right', 'top', 'bottom' or ''
    when they are all to one side of the middle), 'offcentre' (the
    furthest centre from the middle, a share of the frame's half-diagonal)}."""
    W, H = float(frame_size[0]), float(frame_size[1])
    if not boxes or W <= 0 or H <= 0:
        return {"x_span": 0.0, "y_span": 0.0, "side": "", "offcentre": 0.0}
    cx = np.array([(b[0] + b[2]) / 2.0 / W for b in boxes])
    cy = np.array([(b[1] + b[3]) / 2.0 / H for b in boxes])
    side = ("left" if cx.max() < 0.4 else "right" if cx.min() > 0.6 else
            "top" if cy.max() < 0.4 else "bottom" if cy.min() > 0.6 else "")
    off = float(np.max(np.hypot(cx - 0.5, cy - 0.5)) / math.hypot(0.5, 0.5))
    return {"x_span": float(np.ptp(cx)), "y_span": float(np.ptp(cy)), "side": side,
            "offcentre": off}
