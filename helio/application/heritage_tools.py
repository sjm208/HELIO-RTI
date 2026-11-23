"""Tools for coins, seals, gems, inscriptions and finds, GUI-free.

* legend unwrap: the band of lettering round a coin, seal, stamp, binding
  centrepiece or volvelle unrolled into a strip that reads left to right.
  Circles, ovals and pointed ovals (the vesica of ecclesiastical and women's
  seals); straightened by fitting the outline to the borders; read from the
  colour, the view on screen, or a raking light that falls the same way on
  every letter; a part arc for a fragment; the whole face as rows for a
  volvelle, astrolabe or rota; with a degree and millimetre scale.
* virtual_impression: what a seal matrix or intaglio would stamp (mirrored
  and turned inside out), or an inscription's letters raised (turned inside
  out only), lit by one raking light.
* relief_match: how alike two registered surfaces are, for die studies.
* line_drawing_svg: grooves and crests traced as vector lines, the
  archaeological and epigraphic line drawing, in millimetres once a scale is
  set.

The research tools (helio_qt.research) and Publish (helio_qt.publish) call
these.
"""
from __future__ import annotations

import math
from xml.sax.saxutils import escape

import cv2
import numpy as np


# ── round objects: the rim and the legend ────────────────────────────────

SHAPES = {"circle": "Circle", "oval": "Oval", "vesica": "Pointed oval", "free": "Free shape"}


def find_rim(rgb, *, size: int = 640) -> tuple[float, float, float]:
    """(cx, cy, r) in *rgb*'s pixels: the largest circle that reads as the rim.

    Found on a reduced copy with a Hough transform; when none is found, the
    circle centred in the picture that just fits inside it."""
    img = np.asarray(rgb)
    h, w = img.shape[:2]
    grey = _grey8(img)
    k = size / float(max(h, w))
    small = cv2.resize(grey, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.0)
    m = min(small.shape[:2])
    found = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m,
                             param1=80, param2=30, minRadius=int(m * 0.25),
                             maxRadius=int(m * 0.55))
    if found is not None and len(found[0]):
        cx, cy, r = (float(v) / k for v in found[0][0])
        return cx, cy, r
    return w / 2.0, h / 2.0, min(w, h) * 0.47


def _grey8(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float32).mean(axis=2)
    return cv2.normalize(a.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def find_outline(rgb, shape: str = "circle", *, size: int = 800, near=None) -> dict:
    """The object's edge as an outline dict (cx, cy, a, b, turn, shape) in
    *rgb*'s pixels: a, b the half-width and half-height before the turn
    (degrees clockwise).  A circle by the Hough transform (find_rim).  An
    oval or pointed oval from there: along 180 rays out from that centre, the
    strongest edge near the circle, and the ellipse through those points with
    the stray ones dropped.  Pass a relief picture (a tilt or raking view) for
    a worn or paper-covered impression, whose colour hardly shows its edge.

    *near* (cx, cy, r): start from there instead of the Hough circle, as
    when an object found on the page (find_objects) is refined; a circle is
    then the oval fit made round."""
    if shape == "free":
        return find_free(rgb)
    cx, cy, r = near if near is not None else find_rim(rgb)
    out = dict(cx=cx, cy=cy, a=r, b=r, turn=0.0, shape=shape)
    if shape == "circle" and near is None:
        return out
    img = _grey8(rgb)
    h, w = img.shape[:2]
    k = min(1.0, size / float(max(h, w)))
    small = cv2.GaussianBlur(cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                                        interpolation=cv2.INTER_AREA), (0, 0), 1.5).astype(np.float32)
    mag = np.hypot(cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3))
    c0, r0 = np.array([cx, cy]) * k, r * k
    th = np.radians(np.arange(0, 360, 2.0))
    # from a ring found on the page, stay near it: a seal's paper tab or the
    # next border out is a strong edge too
    lo, hi = (0.75, 1.25) if near is not None else (0.6, 1.45)
    rad = np.linspace(lo * r0, hi * r0, 96)
    xs = c0[0] + np.sin(th)[:, None] * rad[None, :]
    ys = c0[1] - np.cos(th)[:, None] * rad[None, :]
    prof = cv2.remap(mag, xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    best = prof.argmax(axis=1)
    ok = prof[np.arange(len(th)), best] > 0
    pts = np.stack([xs[np.arange(len(th)), best], ys[np.arange(len(th)), best]], axis=1)[ok]
    if len(pts) < 20:
        return out
    for _ in range(3):                                # fit, drop the points far off it, fit again
        (ex, ey), (ew, eh), ang = cv2.fitEllipse(pts.astype(np.float32))
        t = math.radians(ang)
        x, y = pts[:, 0] - ex, pts[:, 1] - ey
        u, v = x * math.cos(t) + y * math.sin(t), -x * math.sin(t) + y * math.cos(t)
        res = np.abs(np.hypot(u / max(ew / 2, 1e-6), v / max(eh / 2, 1e-6)) - 1)
        keep = res < max(0.03, 2.5 * np.median(res))
        if keep.all() or keep.sum() < 20:
            break
        pts = pts[keep]
    # fitEllipse's angle turns the first axis (ew) from x; ours turns the height from upright
    a, b, turn = ew / 2 / k, eh / 2 / k, float(ang)
    if a > b:
        a, b, turn = b, a, turn + 90.0
    turn = (turn + 90.0) % 180.0 - 90.0
    if shape == "circle":
        r = math.sqrt(a * b)
        return dict(cx=ex / k, cy=ey / k, a=r, b=r, turn=0.0, shape=shape)
    return dict(cx=ex / k, cy=ey / k, a=a, b=b, turn=turn, shape=shape)


def find_objects(*pictures, size: int = 900, most: int = 8) -> list[dict]:
    """Every seal, coin or stamp on a page: a list of oval outline dicts
    (cx, cy, a, b, turn, shape='oval'), in reading order.

    *pictures* are the same page in one or more forms (the colour, the
    relief).  Each is compared with its own surroundings, not with the page
    as a whole (a black copy-stand or a colour chart beside the page would
    swamp that): a colour picture by how far each part's colour is from the
    page round it (a wax seal on parchment), a grey one by how busy it is
    with relief (a worn or paper-covered impression the colour hides).  The
    differences are cut at several levels, so a faint seal and a strong one
    are both found, and a region is kept when it is a solid, roughly round
    or oval blob, neither a speck nor a sliver nor most of the page; the same
    object found more than once is kept once.  Refine one with find_outline
    on the part round it (object_crop)."""
    if not pictures:
        return []
    h, w = np.asarray(pictures[0]).shape[:2]
    k = min(1.0, size / float(max(h, w)))
    sw, sh = max(1, int(w * k)), max(1, int(h * k))
    big = max(3, int(0.15 * max(sw, sh))) | 1
    maps = []
    for pic in pictures:
        a = np.asarray(pic)
        if a.ndim == 3 and a.shape[2] >= 3 and float(np.ptp(a[..., :3].astype(np.float32).mean(2))) > 0:
            lab = cv2.cvtColor(cv2.resize(_as_rgb8(a), (sw, sh), interpolation=cv2.INTER_AREA),
                               cv2.COLOR_RGB2LAB)
            # the surroundings: a wide median, which steps over an object rather than into it
            bg = cv2.GaussianBlur(cv2.medianBlur(lab, min(big, 255)).astype(np.float32), (0, 0), big / 12)
            near = cv2.GaussianBlur(lab.astype(np.float32), (0, 0), max(2.0, 0.006 * max(sw, sh)))
            maps.append(np.linalg.norm(near - bg, axis=2))
        else:
            g = cv2.resize(_grey8(a), (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
            sig = max(2.0, 0.006 * max(sw, sh))
            mean = cv2.GaussianBlur(g, (0, 0), sig)
            busy = np.sqrt(np.maximum(cv2.GaussianBlur(g * g, (0, 0), sig) - mean * mean, 0))
            maps.append(cv2.GaussianBlur(busy, (0, 0), 2 * sig))
    found = []
    page = sw * sh
    close = max(3, int(0.01 * max(sw, sh))) | 1
    for m in maps:
        for q in (75, 85, 90, 94, 97):
            mask = (m > np.percentile(m, q)).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 0.003 * page or area > 0.2 * page or len(c) < 12:
                    continue
                solid = area / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
                (ex, ey), (ew, eh), ang = cv2.fitEllipse(c)
                fill = area / max(math.pi * ew * eh / 4.0, 1.0)           # 1 for a true ellipse
                if solid < 0.88 or not 0.82 < fill < 1.12 or min(ew, eh) < 0.4 * max(ew, eh):
                    continue
                inside = np.zeros((sh, sw), np.uint8)
                cv2.drawContours(inside, [c], -1, 255, -1)
                ring = cv2.dilate(inside, np.ones((close * 2 + 1,) * 2, np.uint8)) & ~inside
                contrast = float(m[inside > 0].mean()) / (float(m[ring > 0].mean()) + 1e-6) \
                    if ring.any() else 1.0
                ea, eb, turn = ew / 2 / k, eh / 2 / k, float(ang)
                if ea > eb:
                    ea, eb, turn = eb, ea, turn + 90.0
                turn = (turn + 90.0) % 180.0 - 90.0
                score = solid * (1.0 - abs(1.0 - fill)) * min(contrast, 4.0) * (area / page) ** 0.25
                found.append(dict(cx=ex / k, cy=ey / k, a=ea, b=eb, turn=turn, shape="oval", score=score))
    found.sort(key=lambda o: -o["score"])
    kept = []
    for o in found:                    # the same object again: a centre inside one already kept
        if any(math.hypot(o["cx"] - q["cx"], o["cy"] - q["cy"]) < max(min(q["a"], q["b"]), min(o["a"], o["b"]))
               for q in kept):
            continue
        kept.append(o)
        if len(kept) >= most:
            break
    # reading order: top to bottom by rows, then left to right
    row = max(1.0, float(np.median([o["b"] for o in kept]))) if kept else 1.0
    kept.sort(key=lambda o: (round(o["cy"] / (2 * row)), o["cx"]))
    return kept


def object_crop(o: dict, w: int, h: int, margin: float = 0.25) -> tuple[int, int, int, int]:
    """(x, y, width, height): the part of a w x h picture round outline *o*,
    with *margin* of its size beside it - what find_outline refines it in."""
    r = max(o["a"], o["b"]) * (1 + margin)
    x0, y0 = int(max(0, o["cx"] - r)), int(max(0, o["cy"] - r))
    x1, y1 = int(min(w, o["cx"] + r)), int(min(h, o["cy"] + r))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def draw_objects(rgb, objects: list[dict], chosen: int = -1, *, max_side: int = 1600) -> np.ndarray:
    """The page with each object found ringed and numbered (the chosen one
    yellow), reduced to *max_side*."""
    img = _as_rgb8(rgb).copy()
    h, w = img.shape[:2]
    k = min(1.0, max_side / float(max(h, w)))
    if k < 1:
        img = cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    t = max(2, int(round(max(img.shape[:2]) / 400)))
    for i, o in enumerate(objects):
        col = (255, 210, 0) if i == chosen else (0, 200, 255)
        P = (outline_points(o, 240) * k).astype(np.int32)
        cv2.polylines(img, [P], True, (0, 0, 0), t + 2, cv2.LINE_AA)
        cv2.polylines(img, [P], True, col, t, cv2.LINE_AA)
        fs = max(0.6, max(img.shape[:2]) / 900)
        txt = str(i + 1)
        (tw, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
        cx, cy = int(o["cx"] * k - tw / 2), int(o["cy"] * k + th_ / 2)
        cv2.putText(img, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), t + 3, cv2.LINE_AA)
        cv2.putText(img, txt, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, col, t, cv2.LINE_AA)
    return img


FREE_ANCHORS = 16


def free_from(o: dict, n: int = FREE_ANCHORS) -> dict:
    """Any outline as a free shape of *n* anchors spaced evenly round it."""
    return free_outline(outline_points(o, n), shape="free")


def free_outline(anchors, shape: str = "free") -> dict:
    """A free shape through *anchors* (picture pixels): centred on their box,
    a and b its half-width and half-height, the anchors kept as fractions of
    those (so Edge, Proportions and Turn work on it as on an oval)."""
    p = np.asarray(anchors, np.float64)
    (x0, y0), (x1, y1) = p.min(axis=0), p.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    a, b = max((x1 - x0) / 2, 1.0), max((y1 - y0) / 2, 1.0)
    return dict(cx=cx, cy=cy, a=a, b=b, turn=0.0, shape=shape,
                points=[[float((x - cx) / a), float((y - cy) / b)] for x, y in p])


def free_anchors(o: dict) -> np.ndarray:
    """The free shape's anchors on the picture (n, 2)."""
    u = np.asarray(o["points"], np.float64) * np.array([o["a"], o["b"]])
    th = math.radians(float(o.get("turn", 0.0)))
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return u @ rot.T + np.array([o["cx"], o["cy"]])


def with_anchors(o: dict, anchors) -> dict:
    """*o* with its anchors moved to *anchors* (picture pixels), its centre,
    size and turn kept, so dragging one point moves only that point."""
    th = math.radians(float(o.get("turn", 0.0)))
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    u = (np.asarray(anchors, np.float64) - np.array([o["cx"], o["cy"]])) @ rot / np.array([o["a"], o["b"]])
    return dict(o, points=u.tolist())


def _spline_closed(p: np.ndarray, per: int = 64) -> np.ndarray:
    """A closed Catmull-Rom curve through *p*: it passes through every anchor."""
    n = len(p)
    t = np.linspace(0, 1, per, endpoint=False)[:, None]
    out = []
    for i in range(n):
        p0, p1, p2, p3 = p[(i - 1) % n], p[i], p[(i + 1) % n], p[(i + 2) % n]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.concatenate(out)


def find_free(rgb, *, size: int = 800, n: int = FREE_ANCHORS) -> dict:
    """A free shape round the object: the largest closed region near the
    middle that stands out from its surround (a lozenge centrepiece, a shield,
    a cartouche), its outline as *n* anchors.  Else the oval fit's."""
    img = _grey8(rgb)
    h, w = img.shape[:2]
    k = min(1.0, size / float(max(h, w)))
    small = cv2.GaussianBlur(cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k))),
                                        interpolation=cv2.INTER_AREA), (0, 0), 3.0)
    sh, sw = small.shape[:2]
    best, best_score = None, 0.0
    _, th = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (th, 255 - th):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, cw, ch = cv2.boundingRect(c)
            if area < 0.04 * sw * sh or cw > 0.97 * sw or ch > 0.97 * sh:
                continue
            mx, my = x + cw / 2, y + ch / 2
            centred = 1.0 - min(1.0, math.hypot(mx - sw / 2, my - sh / 2) / (0.5 * max(sw, sh)))
            solid = area / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
            score = area * centred * solid
            if score > best_score:
                best, best_score = c, score
    if best is None:
        return free_from(find_outline(rgb, "oval", size=size), n)
    pts = _resample_closed(best[:, 0, :].astype(np.float64), n) / k
    return free_outline(pts)


def relief_picture(normals) -> np.ndarray:
    """How steeply the surface tilts (uint8): the edges of an impression, its
    borders and letters, plain even where the colour shows nothing."""
    n = np.asarray(normals, np.float32)
    t = cv2.GaussianBlur(np.hypot(n[..., 0], n[..., 1]), (0, 0), 1.5)
    hi = float(np.percentile(t, 99.5)) or 1.0
    return np.clip(t / hi * 255, 0, 255).astype(np.uint8)


def outline_points(o: dict, n: int = 4096) -> np.ndarray:
    """(n, 2) points round the outline, clockwise on screen from the top of
    the shape, evenly spaced along it.  A pointed oval (vesica) is two arcs of
    one circle size meeting at its top and bottom points."""
    a, b = float(o["a"]), float(o["b"])
    m = 8 * n
    if o.get("shape") == "free" and o.get("points"):
        pts = _spline_closed(np.asarray(o["points"], np.float64) * np.array([a, b]),
                             per=max(16, m // max(len(o["points"]), 1)))
        x, y = pts[:, 0], pts[:, 1]
        if float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0:
            pts = pts[::-1]                              # clockwise on screen, as the others run
    elif o.get("shape") == "vesica" and b > a:
        R = (a * a + b * b) / (2 * a)
        xc = a - R
        p0, p1 = math.atan2(-b, -xc), math.atan2(b, -xc)
        ph = np.linspace(p0, p1, m // 2, endpoint=False)
        right = np.stack([xc + R * np.cos(ph), R * np.sin(ph)], axis=1)
        left = -right                                   # the same arc turned half round: bottom to top
        pts = np.concatenate([right, left])
    else:
        t = np.linspace(0, 2 * np.pi, m, endpoint=False)
        pts = np.stack([a * np.sin(t), -b * np.cos(t)], axis=1)
    th = math.radians(float(o.get("turn", 0.0)))
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    pts = pts @ rot.T + np.array([o["cx"], o["cy"]])
    return _resample_closed(pts, n)


def _resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    q = np.linspace(0, s[-1], n, endpoint=False)
    closed = np.vstack([pts, pts[:1]])
    return np.stack([np.interp(q, s, closed[:, 0]), np.interp(q, s, closed[:, 1])], axis=1)


def _screen_angle(o: dict, p: np.ndarray) -> np.ndarray:
    """Degrees clockwise from 12 o'clock of points *p* about the outline's centre."""
    return np.degrees(np.arctan2(p[..., 0] - o["cx"], -(p[..., 1] - o["cy"]))) % 360.0


def legend_maps(o: dict, width: float, *, start_deg: float = 0.0, span_deg: float = 360.0,
                clockwise: bool = True, height: int = 0, whole_face: bool = False) -> dict:
    """Where each pixel of the strip comes from: dict(map_x, map_y (H, W)
    float32 for cv2.remap; along, up (W, 2): the strip's right and up as
    directions on the picture; angles (W,): each column's clock angle in
    degrees; length: the strip's length in picture pixels).

    The band is *width* pixels deep inside the outline, offset along the
    outline's normal so it keeps its width all round (an oval's or a pointed
    oval's lettering is as tall at the ends as at the sides).  *whole_face*
    instead shrinks the outline to its centre, so every ring of a volvelle,
    an astrolabe or a rota becomes a row; the outer ring keeps its size.

    Clockwise legends put the edge at the top of the strip; anticlockwise
    ones (letters' feet to the edge) the inside, so both read upright.  The
    strip begins at *start_deg* clockwise from 12 o'clock and runs *span_deg*
    of the way round (less than 360 for a fragment).  Columns are spread
    evenly along the band's middle, so letters keep their size."""
    span = float(np.clip(span_deg, 1.0, 360.0))
    n = 4096
    P = outline_points(o, n)
    T = np.roll(P, -2, axis=0) - np.roll(P, 2, axis=0)           # clockwise tangent
    T /= np.maximum(np.hypot(T[:, 0], T[:, 1]), 1e-9)[:, None]
    N = np.stack([T[:, 1], -T[:, 0]], axis=1)                    # outward
    ang = _screen_angle(o, P)
    i0 = int(np.argmin(np.abs((ang - start_deg + 180.0) % 360.0 - 180.0)))
    order = (i0 + np.arange(n)) % n if clockwise else (i0 - np.arange(n)) % n
    keep = order[:max(2, int(round(n * span / 360.0)) + (0 if span >= 360 else 1))]
    P, T, N, ang = P[keep], T[keep], N[keep], ang[keep]
    if not clockwise:
        T = -T
    C = np.array([o["cx"], o["cy"]])
    if whole_face:
        depth = float(np.mean(np.hypot(*(P - C).T)))
        mid = P
    else:
        depth = float(max(1.0, width))
        mid = P - 0.5 * depth * N
    length = float(np.hypot(*np.diff(mid, axis=0).T).sum())
    if span >= 360:
        length += float(np.hypot(*(mid[0] - mid[-1])))
    W = max(8, int(round(length)))
    H = int(height) if height else max(2, int(round(depth)))
    q = np.linspace(0, len(P) - (0 if span >= 360 else 1), W, endpoint=False if span >= 360 else True)

    def at(arr, qq):
        idx = np.arange(len(arr), dtype=np.float64)
        if span >= 360:
            arr = np.vstack([arr, arr[:1]])
            idx = np.arange(len(arr), dtype=np.float64)
        return np.stack([np.interp(qq, idx, arr[:, k]) for k in range(arr.shape[1])], axis=1)
    Pw, Nw, Tw = at(P, q), at(N, q), at(T, q)
    Nw /= np.maximum(np.hypot(Nw[:, 0], Nw[:, 1]), 1e-9)[:, None]
    Tw /= np.maximum(np.hypot(Tw[:, 0], Tw[:, 1]), 1e-9)[:, None]
    f = np.arange(H, dtype=np.float64) / max(H - 1, 1)            # 0 at the top row
    if not clockwise:
        f = 1.0 - f
    if whole_face:
        s = (1.0 - f)[:, None, None]
        pts = C + s * (Pw[None, :, :] - C)
    else:
        pts = Pw[None, :, :] - (f * depth)[:, None, None] * Nw[None, :, :]
    up = Nw if clockwise else -Nw
    angles = np.unwrap(np.radians(at(ang[:, None], q)[:, 0]))
    return dict(map_x=pts[..., 0].astype(np.float32), map_y=pts[..., 1].astype(np.float32),
                along=Tw, up=up, angles=np.degrees(angles) % 360.0, length=length, depth=depth)


def mirrored(o: dict, w: int) -> dict:
    """The outline as it falls on the picture flipped left to right (*w* wide)."""
    out = dict(o, cx=w - 1 - o["cx"], turn=-float(o.get("turn", 0.0)))
    if o.get("points"):
        out["points"] = [[-u, v] for u, v in o["points"]]
    return out


def unwrap(img, maps: dict) -> np.ndarray:
    """*img* (any channels) resampled into the strip *maps* describes."""
    a = np.ascontiguousarray(img)
    return cv2.remap(a, maps["map_x"], maps["map_y"], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def even_raking(normals, maps: dict, *, light_deg: float = 0.0, elevation: float = 25.0,
                flatten: bool = True) -> np.ndarray:
    """The strip lit by one raking light that turns with the legend, so every
    letter is lit from the same side of itself (uint8 grey).  With a fixed
    light the letters facing it read and those opposite flatten or invert;
    here the surface normals are turned into the strip's own frame column by
    column first.  *light_deg* is where the light comes from on the strip,
    clockwise from its top (0: from the edge for a clockwise legend);
    *elevation* above the surface.  *flatten* takes away the object's broad
    shape so the letters are left.  Normals are +Y up (Helio's convention)."""
    n = unwrap(np.asarray(normals, np.float32), maps)
    nx, ny_img, nz = n[..., 0], -n[..., 1], n[..., 2]               # to picture axes (y down)
    al, up = maps["along"].astype(np.float32), maps["up"].astype(np.float32)
    nt = nx * al[None, :, 0] + ny_img * al[None, :, 1]
    nu = nx * up[None, :, 0] + ny_img * up[None, :, 1]
    a, e = math.radians(light_deg), math.radians(elevation)
    shade = nt * math.sin(a) * math.cos(e) + nu * math.cos(a) * math.cos(e) + nz * math.sin(e)
    valid = np.hypot(np.hypot(nx, ny_img), nz) > 0.1
    if flatten:
        # the object's own dome or dish lights one edge of the band and darkens
        # the other; taken away at the band's scale, the letters are left
        sig = max(2.0, shade.shape[0] / 3.0)
        shade = shade - cv2.GaussianBlur(np.where(valid, shade, 0).astype(np.float32), (0, 0), sig)
    shade = np.where(valid, shade, np.nan)
    lo, hi = np.nanpercentile(shade, 1), np.nanpercentile(shade, 99)
    out = np.clip((np.nan_to_num(shade, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def straighten(img, o: dict, width: float, *, steps: int = 80) -> dict:
    """The outline moved (and, for an oval, reshaped and turned) until the
    rings round the legend run flat along the strip: a centre a few pixels
    out makes every border and baseline wave once round.  Its size is kept.

    The fit maximises the contrast of the strip's average across its height
    (sharp borders give a sharp profile), over a band a little wider than
    the legend, on a small strip so it takes a moment."""
    grey = _grey8(img)
    small = cv2.GaussianBlur(grey, (0, 0), max(0.5, min(grey.shape[:2]) / 1500.0))
    base = dict(o)
    margin = 0.35 * width

    def score(p):
        q = dict(base, cx=p[0], cy=p[1], a=p[2], b=p[3], turn=p[4])
        q2 = dict(q, a=q["a"] + margin, b=q["b"] + margin)
        m = legend_maps(q2, width + 2 * margin, height=64)
        k = 1440.0 / max(m["map_x"].shape[1], 1)
        if k < 1:
            m = dict(m, map_x=cv2.resize(m["map_x"], (1440, 64)), map_y=cv2.resize(m["map_y"], (1440, 64)))
        strip = cv2.remap(small, m["map_x"], m["map_y"], cv2.INTER_LINEAR).astype(np.float32)
        prof = strip.mean(axis=1)
        return float(np.var(prof) + 4.0 * np.mean(np.diff(prof) ** 2))
    from scipy.optimize import minimize
    p0 = np.array([o["cx"], o["cy"], o["a"], o["b"], float(o.get("turn", 0.0))], np.float64)
    round_ = o.get("shape", "circle") in ("circle", "free")    # a free shape only moves
    size = math.sqrt(o["a"] * o["b"])

    def full(x):
        # a circle moves only its centre; an oval also its proportions (its
        # area kept, so the band stays where the user put it) and its turn
        if round_:
            return np.array([x[0], x[1], p0[2], p0[3], p0[4]])
        f = math.exp(x[2])
        return np.array([x[0], x[1], p0[2] * f, p0[3] / f, x[3]])
    x0 = np.array([p0[0], p0[1]] if round_ else [p0[0], p0[1], 0.0, p0[4]])
    best = (score(p0), x0)
    # a simplex search moves every setting at once: one at a time stalled on
    # a turned oval, trading the centre against the turn (2026-10-29)
    for scale in (0.06, 0.02):
        d = [scale * size, scale * size] + ([] if round_ else [scale * 2, scale * 60.0])
        simplex = [best[1]] + [best[1] + np.eye(len(x0))[i] * d[i] for i in range(len(x0))]
        res = minimize(lambda x: -score(full(x)), best[1], method="Nelder-Mead",
                       options=dict(initial_simplex=np.array(simplex), maxfev=steps * 6,
                                    xatol=0.05, fatol=1e-3))
        if -res.fun > best[0]:
            best = (-res.fun, res.x)
    p = full(best[1])
    return dict(base, cx=float(p[0]), cy=float(p[1]), a=float(p[2]), b=float(p[3]),
                turn=float((p[4] + 90.0) % 180.0 - 90.0) if not round_ else float(base.get("turn", 0.0)))


def legend_scale(maps: dict, *, mm_per_px: float = 0.0, height: int = 0) -> np.ndarray:
    """A band to set under the strip: the clock angle every 10° (numbered
    every 30°) and, once a scale is set, millimetres along the band's middle."""
    Hs, W = maps["map_x"].shape
    u = float(np.clip(Hs / 30.0, 0.6, max(0.6, W / 2400.0)))     # follows the strip's height
    H = height or int(round((44 if mm_per_px else 26) * u))
    out = np.full((H, W, 3), 245, np.uint8)
    fs, th = 0.42 * u, max(1, int(round(u)))
    ang = maps["angles"]
    for x in range(1, W):
        a0, a1 = ang[x - 1], ang[x]
        for t in range(0, 360, 10):
            d0, d1 = (a0 - t + 180) % 360 - 180, (a1 - t + 180) % 360 - 180
            if d0 * d1 <= 0 and abs(d0 - d1) < 90:
                big = t % 30 == 0
                cv2.line(out, (x, 0), (x, int((10 if big else 5) * u)), (60, 60, 60), th)
                if big:
                    cv2.putText(out, str(t), (x + 2, int(20 * u)),
                                cv2.FONT_HERSHEY_SIMPLEX, fs, (60, 60, 60), th, cv2.LINE_AA)
    if mm_per_px:
        y = H - int(4 * u)
        step = 5.0
        total = W * mm_per_px
        while total / step > 60:
            step *= 2
        for i in range(int(total / step) + 1):
            x = int(round(i * step / mm_per_px))
            big = i % 2 == 0
            cv2.line(out, (x, y), (x, y - int((9 if big else 5) * u)), (30, 30, 160), th)
            if big:
                cv2.putText(out, f"{i * step:g}", (x + 2, y - int(10 * u)), cv2.FONT_HERSHEY_SIMPLEX,
                            fs, (30, 30, 160), th, cv2.LINE_AA)
        cv2.putText(out, "mm", (W - int(30 * u), y - int(10 * u)), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (30, 30, 160), th, cv2.LINE_AA)
    return out


def outline_record(o: dict, width: float, *, mm_per_px: float = 0.0, **settings) -> dict:
    """What the strip was made from, for the sidecar beside it: the outline
    and band in pixels and, with a scale, the dimensions a seal catalogue
    gives (a circle's diameter; an oval's height x width)."""
    rec = dict(shape=SHAPES.get(o.get("shape", "circle"), "Circle"),
               centre_px=[round(o["cx"], 1), round(o["cy"], 1)],
               half_width_px=round(o["a"], 1), half_height_px=round(o["b"], 1),
               turn_deg=round(float(o.get("turn", 0.0)), 1), band_px=round(float(width), 1), **settings)
    if mm_per_px:
        rec["mm_per_px"] = mm_per_px
        rec["band_mm"] = round(width * mm_per_px, 2)
        if o.get("shape", "circle") == "circle":
            rec["diameter_mm"] = round((o["a"] + o["b"]) * mm_per_px, 2)
        else:
            rec["height_mm"] = round(2 * o["b"] * mm_per_px, 2)
            rec["width_mm"] = round(2 * o["a"] * mm_per_px, 2)
    return rec


def unwrap_legend(rgb, cx: float, cy: float, r_outer: float, r_inner: float, *,
                  start_deg: float = 0.0, clockwise: bool = True, height: int = 0) -> np.ndarray:
    """The ring between *r_inner* and *r_outer* round (cx, cy), unrolled into a
    strip (legend_maps for a circle).  The strip's width is the ring's middle
    circumference, so letters keep their size."""
    r_outer, r_inner = float(max(r_outer, r_inner)), float(min(r_outer, r_inner))
    if r_outer - r_inner < 2:
        raise ValueError("The ring is too narrow: make the outer radius larger than the inner.")
    o = dict(cx=cx, cy=cy, a=r_outer, b=r_outer, turn=0.0, shape="circle")
    return unwrap(rgb, legend_maps(o, r_outer - r_inner, start_deg=start_deg, clockwise=clockwise,
                                   height=height))


def draw_outline(rgb, o: dict, width: float, *, start_deg: float = 0.0, span_deg: float = 360.0,
                 clockwise: bool = True, whole_face: bool = False, max_side: int = 1600) -> np.ndarray:
    """A copy of *rgb* (reduced to *max_side*) with the band, where the strip
    starts (red) and, for a part arc, where it ends (blue), for the preview."""
    img = np.asarray(rgb)
    h, w = img.shape[:2]
    k = min(1.0, max_side / float(max(h, w)))
    out = cv2.resize(_as_rgb8(img), (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_AREA)
    t = max(2, int(max(out.shape[:2]) / 500))
    P = outline_points(o, 720) * k
    cv2.polylines(out, [P.astype(np.int32)], True, (255, 200, 0), t, cv2.LINE_AA)
    m = legend_maps(o, width, start_deg=start_deg, span_deg=span_deg, clockwise=clockwise, height=2,
                    whole_face=whole_face)
    inner = np.stack([m["map_x"][1 if clockwise else 0], m["map_y"][1 if clockwise else 0]], axis=1) * k
    if not whole_face:
        cv2.polylines(out, [inner[::max(1, len(inner) // 720)].astype(np.int32)], span_deg >= 360,
                      (255, 200, 0), t, cv2.LINE_AA)
    outer = np.stack([m["map_x"][0 if clockwise else 1], m["map_y"][0 if clockwise else 1]], axis=1) * k
    for x, col in ((0, (255, 60, 60)), (-1, (60, 140, 255))):
        if x == -1 and span_deg >= 360:
            break
        cv2.line(out, tuple(int(v) for v in inner[x]), tuple(int(v) for v in outer[x]), col, t + 1, cv2.LINE_AA)
    if o.get("shape") == "free" and o.get("points"):
        for x, y in free_anchors(o) * k:
            cv2.circle(out, (int(x), int(y)), 3 * t, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), 3 * t, (255, 200, 0), max(1, t // 2 + 1), cv2.LINE_AA)
    return out


def legend_figure(picture, o: dict, strip, *, margin: float = 0.25, gap: int = 0) -> np.ndarray:
    """The object above its strip, as one picture for a catalogue or a slide:
    the object cropped round its outline at its own size (smaller only when
    it is wider than the strip), centred on white over the strip."""
    pic = _as_rgb8(picture)
    h, w = pic.shape[:2]
    P = outline_points(o, 360)
    (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
    m = margin * max(x1 - x0, y1 - y0)
    x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
    x1, y1 = int(min(w, x1 + m)), int(min(h, y1 + m))
    top = pic[y0:y1, x0:x1]
    strip = _as_rgb8(strip)
    W = strip.shape[1]
    if top.shape[1] > W:
        top = cv2.resize(top, (W, max(1, int(round(top.shape[0] * W / top.shape[1])))),
                         interpolation=cv2.INTER_AREA)
    gap = gap or max(8, strip.shape[0] // 3)
    left = (W - top.shape[1]) // 2
    row = np.full((top.shape[0], W, 3), 255, np.uint8)
    row[:, left:left + top.shape[1]] = top
    return np.vstack([np.full((gap, W, 3), 255, np.uint8), row, np.full((gap, W, 3), 255, np.uint8), strip])


def _as_rgb8(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    a = a[..., :3]
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    a = a.astype(np.float32)
    hi = 1.0 if a.max() <= 1.5 else (255.0 if a.max() <= 255.5 else 65535.0)
    return np.clip(a / hi * 255.0, 0, 255).astype(np.uint8)


def draw_ring(rgb, cx, cy, r_outer, r_inner, start_deg=0.0) -> np.ndarray:
    """*rgb* with the ring and the strip's start marked, for the preview."""
    out = np.ascontiguousarray(np.asarray(rgb)[..., :3]).copy()
    t = max(2, int(max(out.shape[:2]) / 400))
    for r in (r_outer, r_inner):
        cv2.circle(out, (int(cx), int(cy)), int(r), (255, 200, 0), t, cv2.LINE_AA)
    a = math.radians(start_deg)
    p0 = (int(cx + r_inner * math.sin(a)), int(cy - r_inner * math.cos(a)))
    p1 = (int(cx + r_outer * math.sin(a)), int(cy - r_outer * math.cos(a)))
    cv2.line(out, p0, p1, (255, 60, 60), t + 1, cv2.LINE_AA)
    return out


# ── virtual impression ───────────────────────────────────────────────────

IMPRESSION_COLOURS = {
    "Plaster": (236, 232, 222), "Red wax": (168, 40, 32), "Natural wax": (196, 150, 82),
    "Clay": (176, 120, 86), "Lead": (128, 132, 138), "Paper squeeze": (226, 220, 204),
}


def virtual_impression(normals, *, mirror: bool = True, invert: bool = True,
                       azimuth: float = 315.0, elevation: float = 35.0,
                       colour=(236, 232, 222), ambient: float = 0.18) -> np.ndarray:
    """The surface as its impression would look, lit by one light.

    *invert* turns it inside out (bumps <-> hollows: a cut letter stands up);
    *mirror* turns it left for right, as a stamp prints.  A matrix or an
    intaglio wants both; a squeeze-like reading of an inscription wants only
    *invert*.  Normals are +Y up (toward the top of the image).  The light's
    *azimuth* is clockwise from the top of the picture, so 315 is the upper
    left, where readers expect it."""
    n = np.asarray(normals, np.float32)
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    if invert:
        nx, ny = -nx, -ny
    if mirror:
        nx, ny, nz = -nx[:, ::-1], ny[:, ::-1], nz[:, ::-1]
    a, e = math.radians(azimuth), math.radians(elevation)
    lx, ly, lz = math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), math.sin(e)
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    shade = ambient + (1.0 - ambient) * shade
    rgb = shade[..., None] * (np.asarray(colour, np.float32)[None, None, :] / 255.0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ── die studies ──────────────────────────────────────────────────────────

def relief_match(a, b, *, sigma: float = 2.0) -> float:
    """How alike two registered pictures of relief are, 0 to 1 (1: the same).

    The correlation of their fine detail (the picture less its blur at a few
    times *sigma*), where both have something: colour and broad lighting
    count for little, cut detail for much.  For one die against another it is
    a pointer, not a verdict: wear and a different light lower it."""
    def detail(img):
        g = np.asarray(img, np.float32)
        if g.ndim == 3:
            g = g[..., :3].mean(axis=-1)
        g = cv2.GaussianBlur(g, (0, 0), sigma) - cv2.GaussianBlur(g, (0, 0), sigma * 4)
        return g
    da, db = detail(a), detail(b)
    if da.shape != db.shape:
        raise ValueError("Register the two pictures first: they are not the same size.")
    ga = np.asarray(a)
    valid = (ga.max(axis=-1) if ga.ndim == 3 else ga) > 0      # where A was warped to
    valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if valid.sum() < 100:
        return 0.0
    x, y = da[valid], db[valid]
    x, y = x - x.mean(), y - y.mean()
    den = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return max(0.0, float((x * y).sum()) / den) if den > 0 else 0.0


def match_words(score: float) -> str:
    if score >= 0.6:
        return "close: worth examining as a possible die link"
    if score >= 0.35:
        return "partial: some shared detail"
    return "low: little shared detail"


# ── line drawing ─────────────────────────────────────────────────────────

def _curve_lines(dep, sign: int, sig: float, threshold: float) -> np.ndarray:
    """One-pixel lines where the surface curves (sign +1: grooves' floors, -1: crests)."""
    from skimage.morphology import skeletonize
    lap = cv2.Laplacian(cv2.GaussianBlur(dep, (0, 0), sig), cv2.CV_32F, ksize=3) * sign
    g = np.maximum(lap, 0)
    top = float(np.percentile(g[g > 0], 99.5)) if np.any(g > 0) else 0.0
    if top <= 0:
        return np.zeros(dep.shape, bool)
    mask = g / top > float(threshold)
    return skeletonize(mask)


_NB = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def trace_lines(skel, min_len: int = 12) -> list[np.ndarray]:
    """A one-pixel skeleton as polylines [(x, y), ...]: each run between ends and junctions."""
    sk = np.pad(np.asarray(skel, bool), 1)
    ys, xs = np.nonzero(sk)
    nbr = sum(np.roll(np.roll(sk, -dy, 0), -dx, 1).astype(np.int8) for dy, dx in _NB)
    seen = np.zeros_like(sk)
    lines = []

    def walk(y, x):
        pts = [(x, y)]
        seen[y, x] = True
        while True:
            nxt = None
            for dy, dx in _NB:
                yy, xx = y + dy, x + dx
                if sk[yy, xx] and not seen[yy, xx]:
                    nxt = (yy, xx)
                    break
            if nxt is None:
                break
            y, x = nxt
            seen[y, x] = True
            pts.append((x, y))
            if nbr[y, x] > 2:          # a junction ends the run
                break
        return pts
    order = sorted(zip(ys, xs), key=lambda p: nbr[p] != 1)      # ends first, then loops
    for y, x in order:
        if seen[y, x]:
            continue
        pts = walk(y, x)
        if len(pts) >= min_len:
            a = np.asarray(pts, np.float32) - 1.0                  # the padding
            a = cv2.approxPolyDP(a.reshape(-1, 1, 2), 0.8, False).reshape(-1, 2)
            lines.append(a)
    return lines


def line_drawing(depth, *, detail: float = 0.15, smooth: float | None = None,
                 crests: bool = True, min_len: int = 12) -> dict:
    """Grooves (cut lines) and crests (raised edges) of a surface as polylines."""
    from helio.application.research import groove_smoothing
    dep = np.asarray(depth, np.float32)
    sig = float(smooth) if smooth else groove_smoothing(dep.shape[1], dep.shape[0])
    out = {"grooves": trace_lines(_curve_lines(dep, +1, sig, detail), min_len),
           "crests": trace_lines(_curve_lines(dep, -1, sig, detail), min_len) if crests else [],
           "w": dep.shape[1], "h": dep.shape[0]}
    return out


def draw_lines(lines: dict, background=None) -> np.ndarray:
    """The drawing as a picture, for the preview: black grooves, grey crests, on white."""
    h, w = lines["h"], lines["w"]
    img = (np.full((h, w, 3), 255, np.uint8) if background is None
           else (255 - (255 - np.asarray(background)[..., :3]) * 0.25).astype(np.uint8))
    t = max(1, int(max(h, w) / 1500))
    for key, col in (("crests", (150, 150, 150)), ("grooves", (0, 0, 0))):
        for a in lines[key]:
            cv2.polylines(img, [a.astype(np.int32).reshape(-1, 1, 2)], False, col, t, cv2.LINE_AA)
    return img


def bar_length(most: float) -> float:
    """The longest round length (1, 2 or 5 times a power of ten) no longer than *most*."""
    if most <= 0:
        return 1.0
    e = math.floor(math.log10(most))
    return max(s * 10.0 ** e for s in (1, 2, 5) if s * 10.0 ** e <= most * 1.0000001)


def line_drawing_svg(path: str, lines: dict, *, mm_per_px: float = 0.0, title: str = "",
                     background_png: str = "") -> str:
    """The drawing as SVG: grooves and crests in their own layers (groups), a scale
    bar, and sized in millimetres once a scale is set (else in pixels).  With
    *background_png*, that picture sits under the lines in a hidden layer."""
    w, h = lines["w"], lines["h"]
    k = float(mm_per_px) if mm_per_px else 1.0
    unit = "mm" if mm_per_px else "px"
    W, H = w * k, h * k
    stroke = max(W, H) / 900.0

    def poly(a):
        return " ".join(f"{x * k:.3f},{y * k:.3f}" for x, y in a)
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
           f'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
           f'width="{W:.3f}{unit}" height="{H:.3f}{unit}" viewBox="0 0 {W:.3f} {H:.3f}">',
           f"<title>{escape(title or 'Line drawing')}</title>",
           "<desc>Traced by Helio from the photometric-stereo surface: grooves are where the "
           "surface curves into a cut, crests where it curves over an edge.</desc>"]
    if background_png:
        out.append(f'<g id="photograph" inkscape:groupmode="layer" inkscape:label="Photograph" '
                   f'style="display:none"><image xlink:href="{escape(background_png)}" x="0" y="0" '
                   f'width="{W:.3f}" height="{H:.3f}"/></g>')
    for key, label, col, sw in (("crests", "Crests", "#888888", stroke * 0.6),
                                ("grooves", "Grooves", "#000000", stroke)):
        out.append(f'<g id="{key}" inkscape:groupmode="layer" inkscape:label="{label}" fill="none" '
                   f'stroke="{col}" stroke-width="{sw:.4f}" stroke-linecap="round" '
                   f'stroke-linejoin="round">')
        out += [f'<polyline points="{poly(a)}"/>' for a in lines[key]]
        out.append("</g>")
    if mm_per_px:
        bar = bar_length(W / 5.0)
        x0, y0 = W * 0.04, H * 0.96
        out.append(f'<g id="scale" inkscape:groupmode="layer" inkscape:label="Scale">'
                   f'<rect x="{x0:.3f}" y="{y0 - stroke * 3:.3f}" width="{bar:.3f}" '
                   f'height="{stroke * 3:.3f}" fill="#000"/>'
                   f'<text x="{x0:.3f}" y="{y0 - stroke * 6:.3f}" font-family="Helvetica, Arial" '
                   f'font-size="{max(W, H) / 60:.3f}">{bar:g} mm</text></g>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path
