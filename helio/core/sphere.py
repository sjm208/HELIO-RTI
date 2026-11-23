"""Light directions from a chrome sphere, GUI-free (CAL-001, FIX-009).

A mirror sphere shows each light as a highlight.  Where the highlight sits
on the ball gives the ball's surface normal there, and the light is that
normal's mirror reflection of the view direction:

    N = ((hx - cx) / r, (cy - hy) / r, nz)          (+Y up, contracts.md §3)
    L = 2 (N · V) N - V

Three things decide the accuracy, and each had a fault here (KD-020):

* **The radius.**  ``N`` is the highlight's offset divided by the radius,
  so a radius 40% too large pulls every normal towards the centre and every
  light towards the zenith: on the test capture a 25° light came out at
  46°.  The automatic finder padded the radius it measured by 1.3 × 1.1
  for a search window and then used the padded value in the formula.
  ``fit_rim`` measures the ball's silhouette instead: the strongest dark-
  to-bright step along 180 rays from the centre, fitted with a least-squares
  circle (Kåsa) that rejects outliers such as the cast shadow.
* **One light per frame.**  Frames whose highlight was not found were
  skipped, and lights within 5° of each other merged, so the light list
  could come back shorter than the image list and every later frame took
  its neighbour's light.  ``calibrate`` returns a light or a reason for
  every frame, in order.
* **The view direction.**  The classic formula assumes the camera looks
  straight down at the ball (V = (0, 0, 1)).  A ball near the edge of a
  frame shot with a short lens is seen obliquely; with the focal length
  (from the photos' EXIF) the view ray to the ball is known and used.

The highlight itself is the centroid of the brightest blob (at 85% of the
peak), as RTIBuilder and the classic app locate it.
"""
from __future__ import annotations

from helio.core.frame_read import read_frame

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Sphere:
    cx: float
    cy: float
    r: float
    rms: float = float("nan")      # rim-fit residual, px (nan: not fitted)
    inliers: int = 0               # rim points used
    rays: int = 0                  # rim points looked for
    note: str = ""                 # why a rim fit was not used, when it was refused

    def as_tuple(self) -> tuple[float, float, float]:
        return float(self.cx), float(self.cy), float(self.r)


@dataclass(frozen=True)
class Highlight:
    x: float                       # full-resolution image pixels
    y: float
    quality: float                 # peak / mean brightness inside the ball
    ok: bool
    reason: str = ""


@dataclass
class Calibration:
    """One entry per frame, in the frames' order."""
    sphere: Sphere
    lights: np.ndarray             # (N, 3) unit vectors, +Y up; rows of nan where not found
    highlights: list[Highlight]
    view: tuple[float, float, float] = (0.0, 0.0, 1.0)
    focal_px: float | None = None
    sphere2: Sphere | None = None
    notes: list[str] = field(default_factory=list)
    highlights2: list | None = None
    crops: object = None

    @property
    def ok(self) -> np.ndarray:
        return np.isfinite(self.lights).all(axis=1)

    def complete(self) -> bool:
        return bool(self.ok.all()) and len(self.lights) >= 3

    def angles(self) -> list[tuple[float, float] | None]:
        """(azimuth from north clockwise, elevation) in degrees, per frame."""
        out = []
        for lv in self.lights:
            if not np.isfinite(lv).all():
                out.append(None)
                continue
            az = math.degrees(math.atan2(lv[0], lv[1])) % 360
            el = math.degrees(math.asin(max(-1.0, min(1.0, float(lv[2])))))
            out.append((az, el))
        return out


# ── reading ──────────────────────────────────────────────────────────────

def read_grey(path: str, reduce: int = 1):
    """A frame as 8-bit grey, as the highlight search has always read it.

    *reduce* 2, 4 or 8 reads it at that fraction of its size.  A JPEG is
    then decoded at the smaller size directly (its DCT scaling), several
    times faster than decoding a 100 MP frame whole and shrinking it."""
    import cv2
    k = int(reduce)
    if k in (2, 4, 8):
        flag = {2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
                8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[k]
        return read_frame(path, flag)
    return read_frame(path, cv2.IMREAD_GRAYSCALE)


def read_many(paths, reduce: int = 1) -> list:
    """read_grey for each path, several at once (OpenCV decodes without
    holding Python's lock).  Whole 100 MP frames take 100 MB each, so at
    most four are decoded together; reduced ones eight."""
    from concurrent.futures import ThreadPoolExecutor
    import os
    paths = list(paths)
    if len(paths) <= 1:
        return [read_grey(p, reduce) for p in paths]
    workers = max(1, min(len(paths), os.cpu_count() or 2, 8 if reduce > 1 else 4))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda p: read_grey(p, reduce), paths))


def image_size(path: str) -> tuple[int, int] | None:
    """(width, height) from the file's header, without decoding it."""
    from helio.io.raw_loader import is_raw, raw_size
    if is_raw(path):
        try:
            return raw_size(path)
        except Exception:
            return None
    try:
        import warnings
        from PIL import Image
        # A 100 MP header is not a decompression bomb.  A filter for that one
        # warning, set once: catch_warnings is not safe across the threads
        # that read frames together.
        warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
        with Image.open(path) as im:
            w, h = im.size
            o = (im.getexif() or {}).get(0x0112, 1)
        return (h, w) if o in (5, 6, 7, 8) else (w, h)
    except Exception:
        import cv2
        g = read_frame(path, cv2.IMREAD_GRAYSCALE)
        return None if g is None else (g.shape[1], g.shape[0])


def reduction_for(size: tuple[int, int], longest: int) -> int:
    """The largest of 1, 2, 4, 8 that keeps the frame's long side at least *longest*."""
    k = 1
    while k < 8 and max(size) / (k * 2) >= longest:
        k *= 2
    return k


def brightest_view(paths, longest: int = 2400):
    """The whole frame as each pixel's brightest over every photograph,
    stretched and lifted to be seen: for marking a ball by hand when it is
    too dark to find.  The flash's highlight moves round the ball from frame
    to frame, so here they trace its face even when each photograph shows
    only a black disc.  Returns (uint8 RGB at a reduced size, (w, h) of the
    photographs), or (None, None) when the frames cannot be read."""
    import cv2
    paths = list(paths)
    size = image_size(paths[0]) if paths else None
    if not size:
        return None, None
    k = reduction_for(size, longest)
    frames = [g for g in read_many(paths, k) if g is not None]
    if not frames:
        return None, None
    h, w = frames[0].shape[:2]
    frames = [g if g.shape[:2] == (h, w) else cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
              for g in frames]
    m = np.maximum.reduce(frames).astype(np.float32)
    lo, hi = float(np.percentile(m, 1.0)), float(np.percentile(m, 99.8))
    v = np.clip((m - lo) / max(hi - lo, 1.0), 0.0, 1.0) ** 0.6     # shadows lifted
    g8 = (v * 255).astype(np.uint8)
    return np.repeat(g8[..., None], 3, axis=2), size


def hand_sphere(paths, cx: float, cy: float, r: float, *, frames=None) -> Sphere:
    """A ball outlined by hand: fitted to its rim when the rim is found close
    to the circle drawn (centre within a quarter of the radius, radius within
    a quarter), else the circle exactly as drawn.  A dark ball's rim often
    cannot be seen, and a fit that wandered off would be worse than the hand."""
    try:
        if frames is None:
            # A dark shot's rim is too faint to fit; one common gain brings
            # it back without moving anything (the search itself is unchanged).
            fr = read_many(list(paths)[:8])
            if all(g is not None for g in fr):
                top = float(np.percentile(np.maximum.reduce(fr), 99.9))
                k = 255.0 / max(top, 1.0)
                frames = fr if k <= 1.05 else [
                    np.clip(g.astype(np.float32) * k, 0, 255).astype(np.uint8) for g in fr]
        f = refine(paths, cx, cy, r, drawn=True, frames=frames)
    except Exception:
        f = None
    if (f is not None and math.isfinite(f.rms) and not f.note
            and math.hypot(f.cx - cx, f.cy - cy) < 0.25 * r and abs(f.r - r) < 0.25 * r):
        return f
    return Sphere(float(cx), float(cy), float(r))


def max_projection(paths, box=None, limit: int | None = 8):
    """The brightest value of each pixel over the frames (the highlights sweep
    the ball, so their union outlines it).  *box* (x1, y1, x2, y2) crops."""
    import cv2
    out = None
    for p in list(paths)[:limit] if limit else list(paths):
        g = read_frame(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        if box is not None:
            x1, y1, x2, y2 = box
            g = g[y1:y2, x1:x2]
        out = g.copy() if out is None else np.maximum(out, g[:out.shape[0], :out.shape[1]])
    return out


# ── the ball's outline ───────────────────────────────────────────────────

def kasa_circle(x, y):
    """Least-squares circle through points (Kåsa): (cx, cy, r)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    c, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = float(c[0]), float(c[1])
    return cx, cy, math.sqrt(max(float(c[2]) + cx * cx + cy * cy, 0.0))


def ball_polarity(img, cx: float, cy: float, r0: float) -> int:
    """+1 when the ball is darker than what surrounds it (a chrome ball on
    foam or paper: its rim is a dark-inside, bright-outside step), -1 when it
    is brighter (on black velvet, where even its limb outshines the cloth)."""
    a = np.asarray(img, np.float32)
    yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    d = np.hypot(xx - cx, yy - cy)
    inner = a[d < 0.7 * r0]
    ring = a[(d > 1.35 * r0) & (d < 1.9 * r0)]
    if inner.size == 0 or ring.size == 0:
        return 1
    return 1 if float(np.median(ring)) >= float(np.median(inner)) else -1


def _ring_mean(img, cx: float, cy: float, a: float, b: float) -> float:
    """Mean brightness of the ring a <= d < b about (cx, cy)."""
    x1, x2 = max(0, int(cx - b) - 1), min(img.shape[1], int(cx + b) + 2)
    y1, y2 = max(0, int(cy - b) - 1), min(img.shape[0], int(cy + b) + 2)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    d = np.hypot(xx - cx, yy - cy)
    v = img[y1:y2, x1:x2][(d >= a) & (d < b)]
    return float(v.mean()) if v.size else float("nan")


def edge_radius(img, cx: float, cy: float, lo: float, hi: float, polarity: int,
                rays: int = 90) -> tuple[float, float]:
    """The radius of the strongest step of sign *polarity* (+1 dark to bright
    going outwards) between *lo* and *hi*, along *rays* rays, combined
    robustly: (radius, share of rays that agree within 4%)."""
    import cv2
    a = cv2.GaussianBlur(np.asarray(img, np.float32), (0, 0), max(1.0, hi / 150.0))
    rs = np.arange(lo, hi, 0.5, dtype=np.float32)
    if rs.size < 4:
        return float("nan"), 0.0
    t = np.linspace(0, 2 * np.pi, rays, endpoint=False, dtype=np.float32)
    mx = (cx + np.outer(np.cos(t), rs)).astype(np.float32)
    my = (cy + np.outer(np.sin(t), rs)).astype(np.float32)
    prof = cv2.remap(a, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    g = np.diff(prof, axis=1) * float(polarity)
    found = rs[np.argmax(g, axis=1)] + 0.25
    m = float(np.median(found))
    ok = np.abs(found - m) <= 0.04 * m
    return (float(found[ok].mean()) if ok.any() else m), float(ok.mean())


def _ball_in_a_hole(med, cx: float, cy: float, r: float, pol: int) -> float | None:
    """A better rim than *r*, or None, for a ball whose rim found on foam
    (+1) is not the ball's own.

    A ball set in a hole in its card, or on its own shadow, has a dark ring
    round it, with the card lighter beyond.  The strongest rising step is
    then either the hole's outer edge (a narrow dark band just inside it)
    or the edge of the bright reflection inside the ball (a bright band
    just outside it, ending in a falling step).  In both the ball's
    silhouette is that falling step: its limb against the dark ring."""
    if pol != 1:
        return None
    ring = lambda a, b: _ring_mean(med, cx, cy, a * r, b * r)            # noqa: E731
    gap, limb, card = ring(0.90, 0.98), ring(0.76, 0.86), ring(1.02, 1.10)
    if limb - gap >= 6 and card - gap >= 6:                      # the hole's edge
        r2, agree = edge_radius(med, cx, cy, 0.78 * r, 0.995 * r, -1)
        return r2 if agree >= 0.5 else None
    band, inside = ring(1.06, 1.20), ring(0.86, 0.96)
    if band - inside >= 6:                                       # a bright band outside
        r2, agree = edge_radius(med, cx, cy, 1.08 * r, 1.45 * r, -1)
        if agree >= 0.5:
            beyond = _ring_mean(med, cx, cy, r2 * 1.02, r2 * 1.08)
            if band - beyond >= 6:
                return r2
    return None


def _hole_rim(med, cx: float, cy: float, r0: float) -> float | None:
    """The ball's rim near a rough circle when the rim fit saw none: the
    falling step from its limb into a dark ring that has the card lighter
    beyond it (a ball set in a hole, or on its shadow), or None."""
    r2, agree = edge_radius(med, cx, cy, 0.7 * r0, 1.6 * r0, -1)
    if not np.isfinite(r2) or agree < 0.6:
        return None
    ring = lambda a, b: _ring_mean(med, cx, cy, a * r2, b * r2)          # noqa: E731
    limb, gap, card = ring(0.88, 0.97), ring(1.01, 1.06), ring(1.12, 1.30)
    return r2 if (limb - gap >= 6 and card - gap >= 6) else None


def fit_rim(img, cx: float, cy: float, r0: float, rays: int = 180,
            band: tuple[float, float] = (0.5, 2.0), polarity: int = 1) -> Sphere | None:
    """The ball's silhouette near (cx, cy, r0), fitted as a circle.

    Along each ray from the centre the rim is a step in brightness: dark
    inside to bright outside for a ball on foam or paper (*polarity* +1: it
    reflects the room behind the camera), bright inside to dark outside for
    a ball on black velvet (-1, ``ball_polarity`` tells which).  Every
    strong step of that sign is a candidate: on foam at least 35% of that
    ray's strongest; on velvet any step four times the ray's noise, since
    there the rim is the faintest step on the ray.  A trial circle through three candidates on different rays is
    supported by the rays that have a step on it (within 1.5 px; RANSAC, 800
    trials, fixed seed).  On foam the best-supported circle is the rim.  On
    velvet the dome's own reflection makes a bright-to-dark ring *inside*
    the outline too, so of the circles nearly as well supported as the best
    (60%) and concentric with it, the outermost is taken.  HELIO 2026-10
    looked for the foam step only and on a ball on velvet fitted that inner
    reflection, a third inside the rim: every light came out far too low
    (KD-053).  The points are then fitted by least squares (Kåsa) without
    those more than 2.5 median residuals off.  None if the rim cannot be
    seen."""
    import cv2
    from scipy.ndimage import map_coordinates
    if r0 < 2:
        return None
    im = cv2.GaussianBlur(np.asarray(img, np.float32), (0, 0), 1.0)
    rs = np.linspace(band[0] * r0, band[1] * r0, int(r0 * (band[1] - band[0]) * 4) + 8)
    cand = []                                   # (x, y, ray)
    for k, a in enumerate(np.linspace(0, 2 * np.pi, rays, endpoint=False)):
        xs, ys = cx + rs * np.cos(a), cy + rs * np.sin(a)
        prof = map_coordinates(im, [ys, xs], order=1, mode="nearest")
        if polarity >= 0:
            d = np.gradient(prof)                    # per sample, as it always was
            top = float(d.max())
            if top <= 2.0:
                continue
            floor = 0.35 * top
        else:
            # On velvet the rim is the faintest step on the ray, a few grey levels
            # per pixel beside the dome's reflection: a step counts when it stands
            # well clear of this ray's own noise (levels per pixel).
            d = -np.gradient(prof, rs)
            top = float(d.max())
            if top < 1.5:
                continue
            noise = 1.4826 * float(np.median(np.abs(d - np.median(d))))
            floor = max(1.0, 4.0 * noise, 0.12 * top)
        peaks = np.flatnonzero((d[1:-1] >= d[:-2]) & (d[1:-1] >= d[2:])
                               & (d[1:-1] >= floor)) + 1
        for j in peaks:
            cand.append((xs[j], ys[j], k))
    if len(cand) < 12:
        return None
    C = np.asarray(cand, np.float64)
    ray = C[:, 2].astype(int)
    rng = np.random.default_rng(12345)
    min_support = max(12, rays // 6)
    trials = []
    for _ in range(800):
        i = rng.choice(len(C), 3, replace=False)
        if len(set(ray[i])) < 3:
            continue
        try:
            tcx, tcy, tr = kasa_circle(C[i, 0], C[i, 1])
        except Exception:
            continue
        if not (band[0] * r0 * 0.95 <= tr <= band[1] * r0) or math.hypot(tcx - cx, tcy - cy) > 1.5 * r0:
            continue
        res = np.abs(np.hypot(C[:, 0] - tcx, C[:, 1] - tcy) - tr)
        tol = max(1.5, 0.03 * tr)
        score = len(set(ray[res < tol]))
        if score >= min_support:
            trials.append((score, tcx, tcy, tr))
    if not trials:
        return None
    best = max(trials)
    if polarity >= 0:
        _score, tcx, tcy, tr = best
    else:
        near = [t for t in trials if t[0] >= 0.6 * best[0]
                and math.hypot(t[1] - best[1], t[2] - best[2]) <= max(2.0, 0.12 * best[3])]
        _score, tcx, tcy, tr = max(near, key=lambda t: (t[3], t[0]))
    # One point per ray: its candidate nearest the chosen circle.
    res = np.abs(np.hypot(C[:, 0] - tcx, C[:, 1] - tcy) - tr)
    pick = {}
    for idx in np.argsort(res):
        if res[idx] > max(1.5, 0.03 * tr) * 2:
            break
        pick.setdefault(ray[idx], idx)
    P = C[list(pick.values())]
    x, y = P[:, 0], P[:, 1]
    keep = np.ones(len(x), bool)
    fcx, fcy, fr = tcx, tcy, tr
    for _ in range(5):
        if keep.sum() < 8:
            return None
        fcx, fcy, fr = kasa_circle(x[keep], y[keep])
        r_ = np.abs(np.hypot(x - fcx, y - fcy) - fr)
        keep = r_ < max(1.0, 2.5 * float(np.median(r_[keep])))
    if keep.sum() < 8:
        return None
    r_ = np.hypot(x[keep] - fcx, y[keep] - fcy) - fr
    return Sphere(fcx, fcy, fr, float(np.sqrt(np.mean(r_ ** 2))), int(keep.sum()), rays)


# The lowest a light can sit and still be believed (degrees above the
# horizon), and the highlight's offset on the ball, as a fraction of its
# radius, that puts a light there: offset = sin((90 - elevation) / 2).
LOWEST_LIGHT_DEG = 5.0
MIN_RIM_SUPPORT = 0.4          # share of the rays that must find the rim for a fit to count
MAX_HIGHLIGHT_OFFSET = math.sin(math.radians((90.0 - LOWEST_LIGHT_DEG) / 2.0))


def highlight_offsets(crops, s: Sphere) -> list[float]:
    """Each frame's highlight distance from the centre, as a fraction of the
    radius (nan where none is found).  *crops* are grey images in the same
    coordinates as *s*."""
    out = []
    for img in crops:
        h = find_highlight(img, s)
        out.append(math.hypot(h.x - s.cx, h.y - s.cy) / s.r if h.ok else float("nan"))
    return out


def refine(paths, cx: float, cy: float, r0: float, *, drawn: bool = False,
           reduce: int = 1, frames=None, near: bool = False) -> Sphere:
    """A rough circle made exact by its rim, at full resolution.

    *drawn*: the circle is one a person placed on the ball, so the rim is
    looked for close to it (0.75-1.3 r0).  Otherwise the circle is the
    finder's rough one, which seeds on the highlights inside the ball: the
    rim is searched for first in a wider band (0.6-1.8 r0), then in a
    narrow one about the circle just found.

    A fitted rim is refused, and the circle kept as it was (with a note),
    when it would put any frame's light lower than LOWEST_LIGHT_DEG: the
    highlights on a ball sit inside sin(42.5°) = 0.68 of its radius unless
    a lamp was all but level with the object, and a fit that says otherwise
    has found a ring inside the ball (KD-053).  Falls back to the rough
    circle (rms nan) when the rim cannot be seen.

    *reduce* fits the rim on frames read at 1/2, 1/4 or 1/8 size (a large
    ball keeps its precision; the frames decode several times faster), and
    *frames* passes frames already read at that size, so finding two balls
    reads each photograph once.  *near*: the circle is already about the
    ball's outline (a circle found in the frames, not a highlight seed), so
    the rim is looked for close to it (0.75-1.3 r0), as for a drawn one,
    and the search cannot walk out to the edge of the card the ball sits on."""
    k = max(1, int(reduce))
    if k > 1:
        if frames is None:
            frames = read_many(list(paths)[:8], k)
        t = refine(paths, (cx + 0.5) / k - 0.5, (cy + 0.5) / k - 0.5, r0 / k, drawn=drawn,
                   frames=frames, near=near)
        return Sphere((t.cx + 0.5) * k - 0.5, (t.cy + 0.5) * k - 0.5, t.r * k,
                      t.rms * k, t.inliers, t.rays, t.note)
    pad = int(r0 * 3.5) + 6
    x1, y1 = max(0, int(cx - pad)), max(0, int(cy - pad))
    box = (x1, y1, int(cx + pad), int(cy + pad))
    crops = []
    for g in (frames[:8] if frames is not None else read_many(list(paths)[:8])):
        if g is not None:
            crops.append(g[box[1]:box[3], box[0]:box[2]])
    mx = None
    for g in crops:
        mx = g.copy() if mx is None else np.maximum(mx, g[:mx.shape[0], :mx.shape[1]])
    seed = Sphere(float(cx), float(cy), float(r0))
    if mx is None or not mx.size:
        return seed
    c = c0 = (cx - x1, cy - y1, r0)
    # Which way the rim steps is read from the median frame, where the
    # highlights (one place per frame) wash out and the ball is its body colour.
    med = np.median(np.stack([g[:mx.shape[0], :mx.shape[1]] for g in crops]), axis=0)
    pol = ball_polarity(med, c[0], c[1], r0)
    s = None
    bands = ((0.75, 1.3),) if (drawn or near) else ((0.4, 3.0), (0.75, 1.3))
    for band in bands:
        t = fit_rim(mx, c[0], c[1], c[2], band=band, polarity=pol)
        if t is None or not (0.3 * r0 <= t.r <= 3.5 * r0):
            break
        s, c = t, (t.cx, t.cy, t.r)
    if s is None or s.inliers < MIN_RIM_SUPPORT * s.rays:
        r2 = None if drawn else _hole_rim(med, c0[0], c0[1], r0)
        if r2 is not None:
            return Sphere(seed.cx, seed.cy, r2, note=(
                f"the ball sits in a hole or on its shadow: its rim is the edge of its limb "
                f"against the dark ring ({r2:.1f} px)"))
        return Sphere(seed.cx, seed.cy, seed.r, note="the ball's rim could not be seen clearly, "
                                                     "so the circle was kept")
    frames_ = [g[:mx.shape[0], :mx.shape[1]] for g in crops]

    def plausible(t):
        o = [v for v in highlight_offsets(frames_, t) if np.isfinite(v)]
        return not o or max(o) <= MAX_HIGHLIGHT_OFFSET
    if not drawn:
        r2 = _ball_in_a_hole(med, s.cx, s.cy, s.r, pol)
        if r2 is not None:
            t = Sphere(s.cx, s.cy, r2, s.rms, s.inliers, s.rays,
                       note=f"the ball sits in a hole or on its shadow: its rim is the edge "
                            f"of its limb against the dark ring ({r2:.1f} px), not the "
                            f"{s.r:.1f} px edge found first")
            if plausible(t):             # else the rim found first stands
                s = t
    offs = [o for o in highlight_offsets(frames_, s) if np.isfinite(o)]
    if offs and max(offs) > MAX_HIGHLIGHT_OFFSET:
        low = 90.0 - 2.0 * math.degrees(math.asin(min(1.0, max(offs))))
        return Sphere(seed.cx, seed.cy, seed.r,
                      note=(f"a rim at {s.r:.1f} px would put a light {low:.1f}° above the "
                            f"horizon, so it is a ring inside the ball; the circle was kept"))
    return Sphere(s.cx + x1, s.cy + y1, s.r, s.rms, s.inliers, s.rays, s.note)


# ── the highlight ────────────────────────────────────────────────────────

def ball_crop(img, s: Sphere):
    """The square around the ball, everything outside the disc set to 0, and
    the ball's centre in that crop."""
    h, w = img.shape[:2]
    x1, y1 = max(0, int(s.cx - s.r)), max(0, int(s.cy - s.r))
    x2, y2 = min(w, int(s.cx + s.r)), min(h, int(s.cy + s.r))
    roi = img[y1:y2, x1:x2].copy()
    lcx, lcy = float(s.cx - x1), float(s.cy - y1)
    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    roi = np.where((xx - lcx) ** 2 + (yy - lcy) ** 2 <= s.r ** 2, roi, 0).astype(np.uint8)
    return roi, lcx, lcy, x1, y1


def find_highlight(img, s: Sphere, robust: bool = False) -> Highlight:
    """The highlight on the ball in one 8-bit grey frame, in image pixels."""
    from helio.utils.file_utils import _extract_highlight_robust, _extract_highlight_subpixel
    if img is None:
        return Highlight(float("nan"), float("nan"), 0.0, False, "the frame could not be read")
    roi, lcx, lcy, x1, y1 = ball_crop(img, s)
    if roi.size == 0:
        return Highlight(float("nan"), float("nan"), 0.0, False, "the ball is outside the frame")
    if float(roi.max()) < 5:
        return Highlight(float("nan"), float("nan"), 0.0, False, "the ball is black in this frame")
    if robust:
        hx, hy, q, valid = _extract_highlight_robust(roi, lcx, lcy, s.r)
        if not valid:
            hx, hy, q = _extract_highlight_subpixel(roi, lcx, lcy, s.r)
    else:
        hx, hy, q = _extract_highlight_subpixel(roi, lcx, lcy, s.r)
    if math.hypot(hx - lcx, hy - lcy) > s.r:
        return Highlight(hx + x1, hy + y1, q, False, "the highlight is off the ball")
    return Highlight(hx + x1, hy + y1, float(q), True)


# ── geometry ─────────────────────────────────────────────────────────────

def focal_px_from_exif(path: str) -> float | None:
    """The lens focal length in pixels of this frame, from its EXIF, or None.

    From the 35 mm-equivalent focal length over the 43.27 mm full-frame
    diagonal, else from the focal length and the focal-plane resolution."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            ex = im.getexif()
            sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
            tags = {**dict(ex), **dict(sub)}
        f35 = tags.get(41989)                      # FocalLengthIn35mmFilm
        if f35:
            return float(f35) * math.hypot(w, h) / 43.267
        f = tags.get(37386)                        # FocalLength (mm)
        xres, unit = tags.get(41486), tags.get(41488, 2)   # FocalPlaneXResolution
        if f and xres:
            per_mm = float(xres) / (25.4 if int(unit) == 2 else 10.0 if int(unit) == 3 else 1.0)
            # The resolution is the sensor's, and a resized copy keeps it
            # while its width shrinks, so the pair is only trusted when the
            # sensor they imply is a real one (10-70 mm wide).  A 960 px
            # copy of a 100 MP back implies a 4.4 mm sensor: refused, rather
            # than a focal length twelve times too long.
            sensor_mm = max(w, h) / per_mm
            if 10.0 <= sensor_mm <= 70.0:
                return float(f) * per_mm
    except Exception:
        return None
    return None


def focal35_from_exif(path: str) -> float | None:
    """The lens's 35 mm-equivalent focal length from the frame's EXIF, or None
    when it cannot be known (see focal_px_from_exif for the plausibility rule)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            ex = im.getexif()
            sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
            tags = {**dict(ex), **dict(sub)}
        if tags.get(41989):
            return float(tags[41989])
        fpx = focal_px_from_exif(path)
        return None if fpx is None else fpx * 43.267 / math.hypot(w, h)
    except Exception:
        return None


def view_vector(cx: float, cy: float, w: int, h: int, focal_px: float | None):
    """The direction from the ball to the camera, +Y up.  (0, 0, 1) without a focal length."""
    if not focal_px:
        return (0.0, 0.0, 1.0)
    v = np.array([-(cx - w / 2.0), (cy - h / 2.0), float(focal_px)])
    v /= np.linalg.norm(v)
    return tuple(float(t) for t in v)


def light_from_highlight(s: Sphere, hx: float, hy: float, view=(0.0, 0.0, 1.0)) -> np.ndarray:
    """The unit light direction (+Y up) for a highlight at (hx, hy).

    With view (0, 0, 1) this is exactly the classic
    ``estimate_light_from_specular``.  Otherwise the ball is seen along
    *view*: the highlight's offset on the disc is taken in the plane
    perpendicular to it."""
    nx = (hx - s.cx) / s.r
    ny = (s.cy - hy) / s.r
    d = nx * nx + ny * ny
    if d > 1.0:
        k = 1.0 / math.sqrt(d)
        nx, ny, d = nx * k, ny * k, 1.0
    nz = math.sqrt(max(0.0, 1.0 - d))
    V = np.asarray(view, np.float64)
    V = V / np.linalg.norm(V)
    if abs(V[2] - 1.0) < 1e-12:
        N = np.array([nx, ny, nz])
    else:
        e1 = np.array([1.0, 0.0, 0.0]) - V[0] * V
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(V, e1)
        N = nx * e1 + ny * e2 + nz * V
    L = 2.0 * float(N @ V) * N - V
    return L / max(np.linalg.norm(L), 1e-12)


# ── the whole calibration ────────────────────────────────────────────────

class FrameCrops:
    """Each frame read once, kept as the square around the ball(s).

    Moving or resizing the circle then re-finds every highlight from memory
    instead of decoding every photograph again (a 100 MP frame takes a
    second or more to read)."""

    def __init__(self, paths, spheres, margin: float = 2.2):
        import cv2
        self.paths = list(paths)
        self.boxes = []
        self.crops: list[list] = []
        first = read_grey(self.paths[0]) if self.paths else None
        self.h, self.w = (first.shape[:2] if first is not None else (0, 0))
        for s in spheres:
            pad = int(s.r * margin) + 4
            self.boxes.append((max(0, int(s.cx - pad)), max(0, int(s.cy - pad)),
                               min(self.w, int(s.cx + pad)), min(self.h, int(s.cy + pad))))

        def crop(i_p):
            i, p = i_p
            img = first if i == 0 else read_grey(p)
            if img is not None and img.shape[:2] != (self.h, self.w):
                img = cv2.resize(img, (self.w, self.h))
            return [None if img is None else img[y1:y2, x1:x2].copy()
                    for (x1, y1, x2, y2) in self.boxes]
        # Several frames decoded at once (OpenCV releases Python's lock);
        # each is cropped as it arrives, so at most four whole frames are held.
        from concurrent.futures import ThreadPoolExecutor
        import os
        with ThreadPoolExecutor(max_workers=max(1, min(4, os.cpu_count() or 2))) as ex:
            self.crops = list(ex.map(crop, enumerate(self.paths)))

    def covers(self, k: int, s: Sphere) -> bool:
        if k >= len(self.boxes):
            return False
        x1, y1, x2, y2 = self.boxes[k]
        return (s.cx - s.r >= x1 and s.cy - s.r >= y1 and s.cx + s.r <= x2 and s.cy + s.r <= y2)

    def frame(self, i: int, k: int, s: Sphere):
        """(crop, sphere in crop coordinates, (x0, y0))."""
        x1, y1, _x2, _y2 = self.boxes[k]
        return self.crops[i][k], Sphere(s.cx - x1, s.cy - y1, s.r), (x1, y1)


def _find_all(crops: FrameCrops, k: int, s: Sphere, robust: bool) -> list[Highlight]:
    """Every frame's highlight on ball *k*, in image pixels.

    Robust is the classic Robust Mode (extract_lights_from_sphere): blob-
    filtered highlights, then a consensus - frames whose highlight is more
    than 30% of a radius from the median of the good ones (quality above 2)
    are re-read the standard way, since a static reflection does not move
    with the light."""
    from helio.utils.file_utils import _extract_highlight_robust, _extract_highlight_subpixel
    out: list[Highlight] = []
    raw = []
    for i in range(len(crops.paths)):
        img, ls, (x0, y0) = crops.frame(i, k, s)
        if img is None:
            out.append(Highlight(float("nan"), float("nan"), 0.0, False,
                                 "the frame could not be read"))
            raw.append(None)
            continue
        roi, lcx, lcy, x1, y1 = ball_crop(img, ls)
        if roi.size == 0:
            out.append(Highlight(float("nan"), float("nan"), 0.0, False,
                                 "the ball is outside the frame"))
            raw.append(None)
            continue
        if float(roi.max()) < 5:
            out.append(Highlight(float("nan"), float("nan"), 0.0, False,
                                 "the ball is black in this frame"))
            raw.append(None)
            continue
        if robust:
            hx, hy, q, valid = _extract_highlight_robust(roi, lcx, lcy, s.r)
        else:
            (hx, hy, q), valid = _extract_highlight_subpixel(roi, lcx, lcy, s.r), True
        raw.append((roi, lcx, lcy, hx, hy, q, valid))
        out.append(Highlight(hx + x1 + x0, hy + y1 + y0, float(q), True))
    if robust:
        good = [(h.x, h.y) for h, r in zip(out, raw) if r is not None and r[6] and r[5] > 2.0]
        if len(good) >= 3:
            mx = float(np.median([g[0] for g in good]))
            my = float(np.median([g[1] for g in good]))
            tol = s.r * 0.30
            for i, (h, r) in enumerate(zip(out, raw)):
                if r is None:
                    continue
                if math.hypot(h.x - mx, h.y - my) > tol:
                    roi, lcx, lcy = r[0], r[1], r[2]
                    hx, hy, q = _extract_highlight_subpixel(roi, lcx, lcy, s.r)
                    dx, dy = h.x - r[3], h.y - r[4]
                    out[i] = Highlight(hx + dx, hy + dy, float(q), True)
    for i, h in enumerate(out):
        if h.ok and math.hypot(h.x - s.cx, h.y - s.cy) > s.r:
            out[i] = Highlight(h.x, h.y, h.quality, False, "the highlight is off the ball")
    return out


def calibrate(paths, sphere: Sphere, *, robust: bool = False, sphere2: Sphere | None = None,
              focal_px: float | None = None, progress=None, crops: FrameCrops | None = None,
              ) -> Calibration:
    """A light for every frame, from the highlight on *sphere* (and *sphere2*).

    Frames whose highlight cannot be found get a row of nan and a reason,
    never a neighbour's light.  With two balls each frame's light is the
    normalised mean of the two, where both were found (the classic dual-ball
    mode).  *robust* is the classic Robust Mode.  Pass *crops* (FrameCrops)
    to re-use frames already read."""
    paths = list(paths)
    spheres = [sphere] + ([sphere2] if sphere2 is not None else [])
    if crops is None or not all(crops.covers(k, s) for k, s in enumerate(spheres)):
        crops = FrameCrops(paths, spheres)
    h, w = crops.h, crops.w
    view = view_vector(sphere.cx, sphere.cy, w, h, focal_px)
    his = _find_all(crops, 0, sphere, robust)
    his2 = _find_all(crops, 1, sphere2, robust) if sphere2 is not None else None
    view2 = view_vector(sphere2.cx, sphere2.cy, w, h, focal_px) if sphere2 else None
    lights = np.full((len(paths), 3), np.nan, np.float32)
    for i, hl in enumerate(his):
        L = None
        if hl.ok:
            L = light_from_highlight(sphere, hl.x, hl.y, view)
        if his2 is not None and his2[i].ok:
            L2 = light_from_highlight(sphere2, his2[i].x, his2[i].y, view2)
            L = L2 if L is None else (L + L2) / max(np.linalg.norm(L + L2), 1e-12)
        if L is not None:
            if L[2] > 0:
                lights[i] = L
                if not hl.ok:
                    his[i] = Highlight(hl.x, hl.y, hl.quality, True, "from the second ball only")
            else:
                his[i] = Highlight(hl.x, hl.y, hl.quality, False,
                                   "the light it gives is below the horizon: check the outline")
        if progress:
            progress(i + 1, len(paths))
    bad = [i + 1 for i, hl in enumerate(his) if not np.isfinite(lights[i]).all()]
    notes: list[str] = []
    if bad:
        notes.append(f"No usable highlight in frame{'s' if len(bad) > 1 else ''} "
                     + ", ".join(map(str, bad)))
    low = [i + 1 for i in range(len(lights)) if np.isfinite(lights[i]).all()
           and math.degrees(math.asin(min(1.0, float(lights[i][2])))) < LOWEST_LIGHT_DEG]
    if low:
        notes.append(f"Frame{'s' if len(low) > 1 else ''} {', '.join(map(str, low))}: the light "
                     f"comes out within {LOWEST_LIGHT_DEG:.0f}° of the horizon. Check the circle "
                     "sits on the ball's edge: one drawn too small puts every light too low.")
    if focal_px:
        notes.append(f"Perspective-corrected: lens focal length {focal_px:.0f} px")
    cal = Calibration(sphere, lights, his, view, focal_px, sphere2, notes)
    cal.highlights2 = his2
    cal.crops = crops
    try:                                            # what the ball's own precision puts into the normals
        from helio.core.light_positions import calibration_error
        err = calibration_error(cal)
        if err is not None and err["normal_deg"] == err["normal_deg"]:
            notes.append(f"The ball measures each light to about ±{err['direction_deg']:.1f}°"
                         + (" (outline drawn, not fitted to the rim: assumed 1 px)" if err["assumed"] else "")
                         + f"; that alone puts about {err['normal_deg']:.1f}° into the surface "
                           "directions. A larger ball in the frame measures them better.")
    except Exception:                               # noqa: BLE001 - a note, never a failure
        pass
    return cal


def with_second_ball(paths, cal: Calibration, s2: Sphere) -> Calibration | None:
    """Both balls, when a search finds two real ones (KD-056).

    With the lights close to the object their direction changes across it,
    so one ball in a corner sees each light from its own side.  Each frame's
    light becomes the mean of the two balls' (calibrate's dual-ball mode),
    and the calibration notes say how far they disagree.  A second circle
    that is not a ball (a button, a ring) finds no highlight in most frames,
    or lights nowhere near the first ball's: then None, and the first ball
    is used alone.  So is a second circle overlapping the first: the same
    ball found twice.
    """
    if math.hypot(s2.cx - cal.sphere.cx, s2.cy - cal.sphere.cy) < (s2.r + cal.sphere.r):
        return None
    other = calibrate(paths, s2)
    a, b = np.asarray(cal.lights, np.float64), np.asarray(other.lights, np.float64)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if ok.sum() < max(3, int(0.75 * len(a))):
        return None
    ang = np.degrees(np.arccos(np.clip(np.sum(a[ok] * b[ok], axis=1), -1.0, 1.0)))
    if float(np.median(ang)) > 35.0:
        return None
    both = calibrate(paths, cal.sphere, sphere2=s2)
    both.notes.append(
        f"Two balls found and both used: each frame's light is their mean. They disagree by "
        f"{float(np.median(ang)):.0f}° (up to {float(ang.max()):.0f}°) because the lights are close "
        f"to the object, so the light's direction changes across it.")
    from helio.core.light_positions import positions_from_two_balls
    pos = positions_from_two_balls(both)
    if pos is not None and pos["placed"]:
        both.notes.append(
            f"Each light placed where the two balls' directions meet ({pos['placed']} of "
            f"{len(both.lights)}, about {pos['distance_px']:.0f} px away): Near-field correction "
            f"then uses every light's own position instead of one dome radius.")
    return both


def dome_coverage(L) -> dict:
    """The classic Dome Coverage Score (QC-004), from the usable lights.

    Azimuthal uniformity over twelve 30° sectors (35%), elevation balance
    between raking (<20°), mid and steep (>=60°) lights (35%), lights near
    the zenith (15%) and the number of lights, full marks at 24 (15%).
    The classic computed the azimuth with the light matrix's y negated, so
    the empty sectors it listed were mirrored north-south; here they are
    compass directions (KD-021)."""
    L = np.asarray(L, np.float64)
    L = L[np.isfinite(L).all(axis=1)]
    n = len(L)
    if n < 2:
        return {"n": n, "total": 0, "grade": "F", "gaps": []}
    Ln = L / (np.linalg.norm(L, axis=1, keepdims=True) + 1e-9)
    az = np.degrees(np.arctan2(Ln[:, 0], Ln[:, 1])) % 360
    el = np.degrees(np.arcsin(np.clip(Ln[:, 2], -1, 1)))
    az_bins = np.histogram(az, bins=12, range=(0, 360))[0]
    az_fill = (az_bins > 0).sum() / 12.0
    az_cv = az_bins.std() / (az_bins.mean() + 1e-9)
    az_score = max(0, az_fill * 100 - az_cv * 20)
    el_lo = (el < 20).sum() / n
    el_mid = ((el >= 20) & (el < 60)).sum() / n
    el_hi = (el >= 60).sum() / n
    el_score = (min(el_lo, 0.3) / 0.3 * 34 + min(el_mid, 0.4) / 0.4 * 33
                + min(el_hi, 0.3) / 0.3 * 33)
    polar_score = min((el >= 70).sum() / n / 0.1, 1.0) * 100
    count_score = min(n / 24.0, 1.0) * 100
    total = int(0.35 * az_score + 0.35 * el_score + 0.15 * polar_score + 0.15 * count_score)
    grade = "A" if total >= 90 else "B" if total >= 75 else "C" if total >= 60 else \
        "D" if total >= 45 else "F"
    return {"n": n, "total": total, "grade": grade, "az_score": az_score, "el_score": el_score,
            "polar_score": polar_score, "count_score": count_score,
            "sectors_filled": int((az_bins > 0).sum()), "raking": int((el < 20).sum()),
            "mid": int(((el >= 20) & (el < 60)).sum()), "steep": int((el >= 60).sum()),
            "gaps": [(i * 30, (i + 1) * 30) for i, c in enumerate(az_bins) if c == 0],
            "elevations": (float(el.min()), float(el.max()))}


def condition_number(L) -> float:
    """cond(L) of the usable lights: how well they pin down a normal (lower is better).

    A light matrix whose lights all sit at one elevation or in one quarter
    of the sky amplifies image noise into normal error by this factor."""
    L = np.asarray(L, np.float64)
    L = L[np.isfinite(L).all(axis=1)]
    if len(L) < 3:
        return float("inf")
    s = np.linalg.svd(L, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-12))
