"""The publication sheet: one page that presents a capture (REP-011).

Top to bottom: the title and the overall score; two boxes of details (the
object, the capture); a large view of the object with the section line and
any measured lines drawn on it, and the scale bar under it, beside a grid of
the views chosen, each captioned under its picture; the section's profile;
the details (each a row of the same crop under several views); pictures
from the 3D views; and a footer with the lights and how the result was
made.  An optional second page shows the surface lit from eight directions
round its normal map (the light compass).

No words are set over a picture: captions and figures go under or beside
it.  The only marks on a picture are the lines themselves, A and B at a
section's ends and a number at each measured line, which the caption
spells out.  The views are cut to the object (the largest bright region of
the colour picture), so a black cloth, a ball and a ruler do not take the
panels; each grey view's contrast is evened out over the object, and views in
colour print as the screen draws them.  With Sheet.turn the object is mirrored
and turned as the screen shows it.

GUI-free: the pictures come in already drawn (Sheet), at the preview's
size; positions are in the photographs' pixels (full_size).  Written as a
JPEG and a PDF at 300 dpi.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

PAPER, CARD = (250, 249, 246), (255, 255, 255)
INK, INK2, INK3 = (30, 31, 34), (77, 80, 87), (122, 125, 132)
RULE, ACCENT = (214, 210, 203), (185, 120, 26)
SECTION, MEASURE = (52, 120, 200), (214, 70, 60)
FAMILY_COLOURS = {"Relighting": (196, 120, 40), "Relief and terrain": (60, 120, 170),
                  "Decomposition": (120, 90, 160), "Quality": (90, 140, 90)}
PAGE_W = 3508                         # A3 portrait at 300 dpi: 3508 x 4961
PAGE_H = 4961
M = 130                               # the page's margin
DPI = 300

# The views a sheet shows unless others are chosen: two that relight, the
# relief readers that suit tooling and incision, and one that says how sure.
DEFAULT_VIEWS = ("Standard PBR", "Raking Composite", "Exaggerated Shading",
                 "Surface Curvature (Scratches)", "Multi-Scale Relief Model (MSRM)",
                 "Local Dominance", "Specular Enhancement", "Normal Uncertainty (QC)")
DEFAULT_DETAIL_VIEWS = ("Standard PBR", "Exaggerated Shading", "Surface Curvature (Scratches)",
                        "Multi-Scale Relief Model (MSRM)")
HERO_VIEW = "Raking Composite"
# Views lit from overhead by default that read flat on a page: the sheet lights
# them from the side that shows the most relief, at this height (degrees),
# unless their light has been moved.
SIDE_LIT = {"Exaggerated Shading": 45.0}
COLOURED = ("False-Colour Depth", "Colour ICA (IC1-3)", "Normal Uncertainty (QC)", "Normals RGB")
SHORT = {"Surface Curvature (Scratches)": "Surface Curvature", "Multi-Scale Relief Model (MSRM)": "MSRM",
         "Sky-View Factor (SVF)": "Sky-View Factor", "Colour ICA (IC1-3)": "Colour ICA",
         "Normal Uncertainty (QC)": "Normal Uncertainty", "Image Quotient (IQ)": "Image Quotient",
         "PS Normal-Map Lit (Single)": "Normal-map lit"}


@dataclass
class View:
    name: str
    picture: np.ndarray                   # uint8 RGB (or grey, or 0..1 float), preview size
    blurb: str = ""
    family: str = ""


@dataclass
class Sheet:
    """Everything the sheet shows; positions in the photographs' pixels."""
    title: str
    full_size: tuple[int, int]            # (w, h) of the solve
    albedo: np.ndarray                    # the colour picture, preview size: finds the object
    views: list[View]
    hero: View | None = None
    subtitle: str = ""
    object_rows: list[tuple[str, str]] = field(default_factory=list)
    capture_rows: list[tuple[str, str]] = field(default_factory=list)
    score: dict | None = None             # process_tools.overall_score
    lights: np.ndarray | None = None      # (N, 3), +Y up
    mm_per_px: float | None = None
    section: dict | None = None           # {"a_px", "b_px", "dist", "z", "unit", "length", "length_unit", "rq"}
    sections: list[dict] = field(default_factory=list)   # several, A–B, C–D, …; else [section]
    measures: list[tuple[float, float, float, float, str]] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)       # annotations (helio.application.annotations), in the solve's pixels
    details: list[tuple[str, tuple[int, int, int, int]]] = field(default_factory=list)
    detail_views: list[View] = field(default_factory=list)
    shots: list[tuple[str, np.ndarray]] = field(default_factory=list)   # 3D pictures, captioned
    shots_heading: str = "In 3D"
    tools: list[tuple[str, np.ndarray]] = field(default_factory=list)   # the tools' results, captioned
    compass: dict | None = None           # {azimuth: picture} for the second page
    normals: np.ndarray | None = None     # for the compass's centre
    footer: list[str] = field(default_factory=list)
    paper: str = "A3"                     # "A3", "A4" (the A3 page scaled: the same shape) or
                                          # "Letter" (scaled to fit, centred on the paper colour)
    png: bool = False                     # also a lossless PNG of each page
    quality: str = "print"                # "print": 300 dpi; "screen": 150 dpi, a quarter of the pixels
    turn: object = None                   # viewing.ScreenTurn: the screen's mirror and turn; None: as stored


# ── drawing helpers ────────────────────────────────────────────────────

def _sections(s: Sheet) -> list[tuple[str, str, dict]]:
    """Each section with its two end letters: A–B, C–D, E–F, …"""
    secs = list(s.sections) or ([s.section] if s.section else [])
    return [(chr(65 + 2 * i), chr(66 + 2 * i), sec) for i, sec in enumerate(secs[:13])]


def _fonts():
    from PIL import ImageFont
    for path in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(path):
            return lambda size, bold=False: ImageFont.truetype(path, size, index=1 if bold else 0)
    try:
        import matplotlib
        path = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        reg, bold_ = os.path.join(path, "DejaVuSans.ttf"), os.path.join(path, "DejaVuSans-Bold.ttf")
        return lambda size, bold=False: ImageFont.truetype(bold_ if bold else reg, size)
    except Exception:                                     # pragma: no cover
        return lambda size, bold=False: ImageFont.load_default()


F = _fonts()


def _rgb(a) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.uint8:
        f = np.nan_to_num(a.astype(np.float32))
        f = f / (255.0 if f.max() > 1.5 else 1.0)
        a = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return np.ascontiguousarray(a[..., :3])


def stretch(a, lo: float = 1.0, hi: float = 99.5, mid: float = 0.5) -> np.ndarray:
    """A view balanced as a print would be: its contrast spread from the 1st
    to the 99.5th percentile, then its mid-tone set so the typical part of
    the object sits near mid-grey (a shading view that is mostly lit came out
    nearly white, its shadows a few dark pixels)."""
    f = _rgb(a).astype(np.float32)
    lum = f.mean(axis=2)
    l1, l2 = np.percentile(lum, lo), np.percentile(lum, hi)
    g = np.clip((f - l1) / max(l2 - l1, 1.0), 0, 1)
    m = float(np.median(g.mean(axis=2)))
    if 0.02 < m < 0.98:
        gamma = float(np.clip(math.log(mid) / math.log(m), 0.5, 3.0))
        g = np.power(g, gamma)
    return (g * 255).astype(np.uint8)


def trim(a, tol: float = 12.0, pad: int = 24) -> np.ndarray:
    """A picture cut to what differs from its corners' colour (a 3D view's
    background), with a margin."""
    a = _rgb(a)
    corners = np.array([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]], np.float32).mean(axis=0)
    diff = np.abs(a.astype(np.float32) - corners).max(axis=2) > tol
    ys, xs = np.nonzero(diff)
    if len(xs) < 50:
        return a
    h, w = a.shape[:2]
    return a[max(0, ys.min() - pad):min(h, ys.max() + pad), max(0, xs.min() - pad):min(w, xs.max() + pad)]


def object_box(albedo) -> tuple[int, int, int, int]:
    """The object alone, in the picture's pixels: the largest bright region
    of the colour picture, with a margin (the cloth, the ball and the ruler
    are smaller, or apart from it)."""
    g = cv2.cvtColor(_rgb(albedo), cv2.COLOR_RGB2GRAY)
    H, W = g.shape
    _t, m = cv2.threshold(cv2.GaussianBlur(g, (0, 0), 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return 0, 0, W, H
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = (int(v) for v in stats[k, :4])
    if w * h < 0.08 * W * H:                       # nothing stands out: the whole frame
        return 0, 0, W, H
    pad = int(0.02 * max(w, h))
    return max(0, x - pad), max(0, y - pad), min(W, x + w + pad), min(H, y + h + pad)


def _fit(img: np.ndarray, w: int, h: int, bg=PAPER):
    """*img* scaled to fit w x h; returns (PIL tile, (x, y, w, h) of the picture in it)."""
    from PIL import Image
    im = Image.fromarray(img)
    k = min(w / im.width, h / im.height)
    im = im.resize((max(1, int(im.width * k)), max(1, int(im.height * k))), Image.LANCZOS)
    tile = Image.new("RGB", (w, h), bg)
    x, y = (w - im.width) // 2, (h - im.height) // 2
    tile.paste(im, (x, y))
    return tile, (x, y, im.width, im.height)


def _wrap(d, text: str, font, width: float) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for wd in words:
        t = (cur + " " + wd).strip()
        if d.textlength(t, font=font) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def _short(name: str) -> str:
    return SHORT.get(name, name)


class _Page:
    """A3 pages laid out top to bottom: a part that does not fit what is left
    of a page starts the next one, so nothing is cut across a page break
    (a part taller than a whole page lengthens its page instead)."""

    def __init__(self, height: int = PAGE_H):
        from PIL import Image, ImageDraw
        self.done = []
        self.im = Image.new("RGB", (PAGE_W, height), PAPER)
        self.d = ImageDraw.Draw(self.im)
        self.y = M

    def need(self, h: int):
        from PIL import Image, ImageDraw
        if self.y + h + M <= self.im.height:
            return
        if self.y > M + 10 and h + 2 * M <= PAGE_H:          # a new page
            self.done.append(self.im)
            self.im = Image.new("RGB", (PAGE_W, PAGE_H), PAPER)
            self.d = ImageDraw.Draw(self.im)
            self.y = M
            return
        bigger = Image.new("RGB", (PAGE_W, self.y + h + M + 200), PAPER)
        bigger.paste(self.im, (0, 0))
        self.im, self.d = bigger, ImageDraw.Draw(bigger)

    def grow(self, h: int):
        """This page *h* px taller (scaled onto the paper with the rest of it)."""
        from PIL import Image, ImageDraw
        bigger = Image.new("RGB", (PAGE_W, self.im.height + int(h)), PAPER)
        bigger.paste(self.im, (0, 0))
        self.im, self.d = bigger, ImageDraw.Draw(bigger)

    def pages(self):
        last = self.im.crop((0, 0, PAGE_W, max(PAGE_H, int(self.y + M))))
        return self.done + [last]

    def heading(self, text: str, room: int = 90):
        """A heading, kept on the page with the first *room* pixels of what follows."""
        self.need(room)
        self.d.text((M, self.y), text.upper(), font=F(30, True), fill=INK3)
        self.d.line((M, self.y + 46, PAGE_W - M, self.y + 46), fill=RULE, width=2)
        self.y += 72


# ── the parts of the page ──────────────────────────────────────────────

def _header(pg: _Page, s: Sheet):
    d = pg.d
    d.text((M, pg.y), s.title, font=F(72, True), fill=INK)
    if s.subtitle:
        for k, line in enumerate(_wrap(d, s.subtitle, F(36), PAGE_W - 2 * M - 420)[:2]):
            d.text((M, pg.y + 100 + k * 46), line, font=F(36), fill=INK2)
    if s.score:
        x, y = PAGE_W - M - 340, pg.y
        d.rounded_rectangle((x, y, x + 340, y + 150), 20, fill=CARD, outline=RULE, width=3)
        d.text((x + 28, y + 14), str(s.score["score"]), font=F(88, True), fill=INK)
        d.text((x + 175, y + 32), s.score["word"], font=F(36, True), fill=ACCENT)
        d.text((x + 175, y + 82), "score of 100", font=F(26), fill=INK3)
    pg.y += 210


def _boxes(pg: _Page, s: Sheet):
    """The two boxes of details, side by side, rows only where there is something to say."""
    boxes = [("The object", s.object_rows), ("The capture", s.capture_rows)]
    boxes = [(t, [(k, v) for k, v in rows if str(v).strip()]) for t, rows in boxes]
    boxes = [(t, rows) for t, rows in boxes if rows]
    if not boxes:
        return
    gap = 60
    bw = (PAGE_W - 2 * M - gap * (len(boxes) - 1)) // len(boxes)
    lab_w = 360
    f_lab, f_val = F(28, True), F(28)
    heights = []
    layouts = []
    for _t, rows in boxes:
        lines = []
        for k, v in rows:
            vl = _wrap(pg.d, v, f_val, bw - lab_w - 70) or [""]
            lines.append((k, vl))
        layouts.append(lines)
        heights.append(90 + sum(40 * len(vl) + 10 for _k, vl in lines) + 20)
    h = max(heights)
    pg.need(h + 50)
    for i, ((title, _rows), lines) in enumerate(zip(boxes, layouts)):
        x = M + i * (bw + gap)
        pg.d.rounded_rectangle((x, pg.y, x + bw, pg.y + h), 16, fill=CARD, outline=RULE, width=2)
        pg.d.text((x + 34, pg.y + 26), title.upper(), font=F(28, True), fill=ACCENT)
        yy = pg.y + 86
        for k, vl in lines:
            pg.d.text((x + 34, yy), k, font=f_lab, fill=INK2)
            for j, v in enumerate(vl):
                pg.d.text((x + 34 + lab_w, yy + j * 40), v, font=f_val, fill=INK)
            yy += 40 * len(vl) + 10
    pg.y += h + 60


def _scale_bar(d, x: int, y: int, px_per_mm: float, max_w: int):
    """A scale bar of a round number of millimetres, under a picture."""
    for mm in (500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1):
        if px_per_mm * mm <= max_w * 0.35:
            break
    L = px_per_mm * mm
    d.rectangle((x, y, x + L, y + 14), fill=INK)
    d.rectangle((x + L / 2, y, x + L, y + 14), fill=CARD, outline=INK, width=2)
    d.text((x + L + 20, y - 10), f"{mm:g} mm", font=F(28), fill=INK2)


NOTE_TYPES = ("arrow", "ellipse", "line", "measure", "polygon", "freehand", "dot", "number", "text")
NOTES_MAX = 14


def _hex(c, default=ACCENT):
    c = str(c or "").lstrip("#")
    try:
        return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) if len(c) >= 6 else default
    except ValueError:
        return default


def _note_list(s: Sheet) -> list[dict]:
    """The annotations the sheet shows: every kind but the rectangles (those are the boxed details)."""
    return [a for a in s.notes if a.get("type") in NOTE_TYPES and a.get("pts")][:NOTES_MAX]


def _draw_notes(d, s: Sheet, to, legend: list):
    """The annotation layer on the large view: each shape in its own colour, and a tag n1, n2 … beside every
    one that has a caption, which the legend then spells out."""
    from helio.application.annotations import length_label
    k = 0
    for a in _note_list(s):
        t, col = a["type"], _hex(a.get("color"))
        pts = [to(x, y) for x, y in a["pts"]]
        p, q = pts[0], pts[-1]
        anchor = p
        if t == "arrow":
            d.line((*p, *q), fill=col, width=8)
            ang = math.atan2(q[1] - p[1], q[0] - p[0])
            for da in (2.6, -2.6):
                d.line((*q, q[0] + 52 * math.cos(ang + da), q[1] + 52 * math.sin(ang + da)), fill=col, width=8)
        elif t == "ellipse":
            d.ellipse((min(p[0], q[0]), min(p[1], q[1]), max(p[0], q[0]), max(p[1], q[1])), outline=col, width=7)
            anchor = (min(p[0], q[0]), min(p[1], q[1]))
        elif t in ("line", "measure"):
            d.line((*p, *q), fill=col, width=7)
            anchor = ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 - 30)
            if t == "measure":
                lab = a.get("label") or length_label(a["pts"][0], a["pts"][-1], s.mm_per_px)
                d.text((anchor[0] + 24, anchor[1] - 18), lab, font=F(26, True), fill=col)
        elif t in ("polygon", "freehand") and len(pts) > 1:
            d.line(pts + ([pts[0]] if t == "polygon" else []), fill=col, width=7, joint="curve")
        elif t in ("dot", "number"):
            d.ellipse((p[0] - 13, p[1] - 13, p[0] + 13, p[1] + 13), fill=col, outline=CARD, width=3)
            if t == "number":
                d.text((p[0] + 20, p[1] - 18), str(a.get("text") or ""), font=F(28, True), fill=col)
                continue
        elif t == "text":
            anchor = q
            d.ellipse((q[0] - 8, q[1] - 8, q[0] + 8, q[1] + 8), fill=col)
        cap = str(a.get("text") or "").strip() if t != "measure" else ""
        if not cap:
            continue
        k += 1
        _marker(d, anchor[0] + (30 if t in ("dot", "text") else 0), anchor[1] - (30 if t in ("dot", "text") else 0),
                f"n{k}", col, r=26)
        legend.append((f"n{k}", col, cap))
    more = sum(1 for a in s.notes if a.get("type") in NOTE_TYPES and a.get("pts")) - len(_note_list(s))
    if more > 0:
        legend.append(("…", INK3, f"and {more} more notes not shown"))


def _marker(d, x, y, text, colour, r=22):
    d.ellipse((x - r, y - r, x + r, y + r), fill=colour, outline=CARD, width=4)
    f = F(26, True)
    tw = d.textlength(text, font=f)
    d.text((x - tw / 2, y - 16), text, font=f, fill=CARD)


def _tile(pg: _Page, v: View, box, x: int, y: int, w: int, h_pic: int, letter: str):
    """One view: its picture, and under it its name and what it shows."""
    x1, y1, x2, y2 = box
    img = _rgb(v.picture)[y1:y2, x1:x2]
    img = balanced(v.name, img)
    tile, _ = _fit(img, w, h_pic)
    pg.im.paste(tile, (x, y))
    cy = y + h_pic + 14
    col = FAMILY_COLOURS.get(v.family, RULE)
    pg.d.rectangle((x, cy + 4, x + 10, cy + 40), fill=col)
    pg.d.text((x + 24, cy), f"{letter}  {_short(v.name)}", font=F(28, True), fill=INK)
    if v.blurb:
        for j, line in enumerate(_wrap(pg.d, v.blurb, F(22), w - 30)[:2]):
            pg.d.text((x + 24, cy + 40 + j * 27), line, font=F(22), fill=INK2)


CAPTION_H = 120


def _hero_and_grid(pg: _Page, s: Sheet, box):
    """The large view (with its lines) on the left, four views beside it in
    two columns, and the rest of the views in rows of four under both."""
    x1, y1, x2, y2 = box
    aspect = (y2 - y1) / max(1, x2 - x1)
    hero = s.hero or (s.views[0] if s.views else None)
    hw_ = int((PAGE_W - 2 * M) * 0.56)
    gx = M + hw_ + 70
    gw = PAGE_W - M - gx
    pic = balanced(hero.name, _rgb(hero.picture)[y1:y2, x1:x2]) if hero is not None else None
    hh = int(min(hw_ * aspect, 2100))
    cw = (gw - 40) // 2
    ch_pic = int(cw * aspect)
    beside = min(len(s.views), 2 * max(1, (hh + 260) // (ch_pic + CAPTION_H + 30)))
    pg.need(hh + 420)
    top = pg.y
    legend = []
    if pic is not None:
        tile, (ox, oy, tw, th) = _fit(pic, hw_, hh)
        pg.im.paste(tile, (M, top))
        pw, ph = s.full_size
        aw, ah = s.albedo.shape[1], s.albedo.shape[0]
        to = lambda X, Y: (M + ox + (X * aw / pw - x1) * tw / max(1, x2 - x1),          # noqa: E731
                           top + oy + (Y * ah / ph - y1) * th / max(1, y2 - y1))
        d = pg.d
        for i, (name, (dx1, dy1, dx2, dy2)) in enumerate(s.details, 1):
            p, q = to(dx1, dy1), to(dx2, dy2)
            d.rectangle((*p, *q), outline=ACCENT, width=6)
            _marker(d, p[0], p[1], str(i), ACCENT)
        if s.details:
            legend.append(("1–" + str(len(s.details)) if len(s.details) > 1 else "1", ACCENT,
                           "details, shown close up below"))
        for a_, b_, sec in _sections(s):
            for cp in (sec.get("series") or []):                       # the parallel copies, fainter
                if abs(float(cp.get("offset", 0.0))) < 1e-9 or "a_px" not in cp:
                    continue
                d.line((*to(*cp["a_px"]), *to(*cp["b_px"])), fill=_lighter(SECTION), width=5)
            (ax, ay), (bx, by) = sec["a_px"], sec["b_px"]
            p, q = to(ax, ay), to(bx, by)
            d.line((*p, *q), fill=SECTION, width=8)
            _marker(d, *p, a_, SECTION)
            _marker(d, *q, b_, SECTION)
            ncp = len(sec.get("series") or [])
            legend.append((f"{a_}–{b_}", SECTION, f"section, {sec['length']:.3g} "
                                                 f"{sec['length_unit']} (profile below)"
                                                 + (f", with {ncp - 1} parallel copies" if ncp > 1 else "")))
        _draw_notes(d, s, to, legend)
        for i, (mx1, my1, mx2, my2, lab) in enumerate(s.measures, 1):
            p, q = to(mx1, my1), to(mx2, my2)
            d.line((*p, *q), fill=MEASURE, width=7)
            for e in (p, q):
                d.ellipse((e[0] - 9, e[1] - 9, e[0] + 9, e[1] + 9), fill=MEASURE)
            _marker(d, (p[0] + q[0]) / 2, (p[1] + q[1]) / 2 - 36, f"m{i}", MEASURE, r=24)
            legend.append((f"m{i}", MEASURE, f"measured {lab}"))
        cy = top + hh + 26
        d.text((M, cy), _short(hero.name), font=F(34, True), fill=INK)
        if hero.blurb:
            for j, line in enumerate(_wrap(d, hero.blurb, F(26), hw_ - 420)[:2]):
                d.text((M, cy + 46 + j * 32), line, font=F(26), fill=INK2)
        if s.mm_per_px:
            px_per_mm = (tw / max(1, x2 - x1)) * (aw / pw) / s.mm_per_px
            _scale_bar(d, M + hw_ - 380, cy + 14, px_per_mm, 380 * 2.8)
        ly = cy + 130
        for tag, col, text in legend:
            d.rounded_rectangle((M, ly, M + 80, ly + 36), 8, fill=col)
            tw2 = d.textlength(tag, font=F(24, True))
            d.text((M + 40 - tw2 / 2, ly + 4), tag, font=F(24, True), fill=CARD)
            lines = _wrap(d, text, F(26), hw_ - 120)[:3] or [""]
            for j, line in enumerate(lines):
                d.text((M + 100, ly + 2 + j * 32), line, font=F(26), fill=INK2)
            ly += 46 + 32 * (len(lines) - 1)
        left_bottom = ly
    else:
        left_bottom = top
    right_bottom = top
    for i, v in enumerate(s.views[:beside]):
        c, r = i % 2, i // 2
        y = top + r * (ch_pic + CAPTION_H + 30)
        _tile(pg, v, box, gx + c * (cw + 40), y, cw, ch_pic, chr(97 + i))
        right_bottom = max(right_bottom, y + ch_pic + CAPTION_H)
    pg.y = max(left_bottom, right_bottom) + 50
    rest = s.views[beside:]
    if rest:
        cols = 4
        w4 = (PAGE_W - 2 * M - 40 * (cols - 1)) // cols
        h4 = int(w4 * aspect)
        for start in range(0, len(rest), cols):
            pg.need(h4 + CAPTION_H + 30)
            for k, v in enumerate(rest[start:start + cols]):
                _tile(pg, v, box, M + k * (w4 + 40), pg.y, w4, h4, chr(97 + beside + start + k))
            pg.y += h4 + CAPTION_H + 40
    pg.y += 20


def _lighter(c, k: float = 0.55):
    return tuple(int(v + (255 - v) * k) for v in c)


def _nice_step(x: float) -> float:
    if x <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(x))
    return next((m * p for m in (5, 2, 1) if m * p <= x), p)


SNIP_FRACTION = 0.36                  # the snippet's share of the page's width; the graph has the rest


def _snip_layout(snip):
    """(graph's left edge, graph's width, snippet's left edge, snippet's width) for a section block: the picture
    of the area on one side and the graph on the other, or the graph alone across the page."""
    full = PAGE_W - 2 * M
    if snip is None:
        return M, full, M, 0
    sw, gap = int(full * SNIP_FRACTION), 60
    cw = full - sw - gap
    return (M + sw + gap, cw, M, sw) if snip[0] == "L" else (M, cw, M + cw + gap, sw)


def _paste_snip(pg: _Page, snip, y0: int, h: int):
    """The area picture beside the graph, at the block's top."""
    _cl, _cw, sx, sw = _snip_layout(snip)
    th = int(min(h - 40, 820, sw * 0.95))
    tile = snip[1](sw, th)
    pg.im.paste(tile, (sx, y0))
    pg.d.rectangle((sx, y0, sx + sw, y0 + th), outline=RULE, width=2)
    if len(snip) > 2 and snip[2]:
        pg.d.text((sx, y0 + th + 14), snip[2], font=F(24), fill=INK3)


def _snippet(s: Sheet, sec: dict, a_: str, b_: str):
    """A builder for the picture of the area a section runs through: the large view cut round the line (and its
    copies), with the line and its ends A and B drawn on it and a scale bar, to fit a box of any shape."""
    hero = s.hero or (s.views[0] if s.views else None)
    if hero is None or "a_px" not in sec:
        return None
    pic = _rgb(hero.picture)
    ph_, pw_ = pic.shape[:2]
    W, H = s.full_size
    kx, ky = pw_ / float(max(W, 1)), ph_ / float(max(H, 1))
    lines = [(sec["a_px"], sec["b_px"])] + [(c["a_px"], c["b_px"]) for c in (sec.get("series") or []) if "a_px" in c]

    def build(bw: int, bh: int):
        from PIL import Image, ImageDraw
        xs = [p[0] * kx for ln in lines for p in ln]
        ys = [p[1] * ky for ln in lines for p in ln]
        aspect = bw / float(max(bh, 1))
        cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        half_w = max((max(xs) - min(xs)) / 2.0 * 1.45, (max(ys) - min(ys)) / 2.0 * 1.45 * aspect, 24.0)
        half_h = half_w / aspect
        half_w, half_h = min(half_w, pw_ / 2.0), min(half_h, ph_ / 2.0)
        if half_h * aspect > half_w:
            half_w = half_h * aspect
        cx = min(max(cx, half_w), pw_ - half_w)
        cy = min(max(cy, half_h), ph_ - half_h)
        x1, x2, y1, y2 = int(cx - half_w), int(cx + half_w), int(cy - half_h), int(cy + half_h)
        crop = pic[max(0, y1):max(y1 + 2, y2), max(0, x1):max(x1 + 2, x2)]
        crop = balanced(hero.name, crop)
        tile = Image.fromarray(np.ascontiguousarray(cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_CUBIC)))
        d = ImageDraw.Draw(tile)
        sc = bw / float(max(x2 - x1, 1))
        to = lambda p: ((p[0] * kx - x1) * sc, (p[1] * ky - y1) * sc)             # noqa: E731
        for (pa, pb) in lines[1:]:
            d.line((*to(pa), *to(pb)), fill=_lighter(SECTION), width=4)
        pa, pb = to(sec["a_px"]), to(sec["b_px"])
        d.line((*pa, *pb), fill=SECTION, width=8)
        _marker(d, *pa, a_, SECTION)
        _marker(d, *pb, b_, SECTION)
        if s.mm_per_px:
            ppm = sc * kx / s.mm_per_px                                             # tile pixels per millimetre
            mm = next((m for m in (500, 200, 100, 50, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1) if ppm * m <= bw * 0.9 * 0.35), 0.1)
            plate_w = int(ppm * mm + 24 + d.textlength(f"{mm:g} mm", font=F(28)) + 40)
            d.rounded_rectangle((14, bh - 70, 14 + plate_w, bh - 14), 12, fill=(255, 255, 255))
            _scale_bar(d, 28, bh - 50, ppm, int(bw * 0.9))
        return tile
    return build


def _section_charts(pg: _Page, s: Sheet):
    secs = _sections(s)
    for i, (a_, b_, sec) in enumerate(secs):
        build = _snippet(s, sec, a_, b_)
        snip = (("L" if i % 2 == 0 else "R"), build, f"Where {a_}–{b_} was taken") if build is not None else None  # alternates sides
        if len(sec.get("series") or []) > 1:
            _series_chart(pg, sec, a_, b_, snip)
        else:
            _section_chart(pg, sec, a_, b_, 520 if len(secs) == 1 and snip is None else 460, snip)


def _series_chart(pg: _Page, sec: dict, a_: str, b_: str, snip=None):
    """The section and its parallel copies stacked one above another (a ridgeline plot), each at its own
    baseline, the offset written at the right and one height bar for all."""
    from helio.core.sections import stack_gap
    profs = [p for p in sec["series"] if p.get("dist") is not None and len(p["dist"]) > 1]
    if len(profs) < 2:
        return _section_chart(pg, sec, a_, b_, 520, snip)
    n = len(profs)
    h = int(min(1700, 420 + 62 * n))
    pg.heading(f"Section {a_}–{b_} and {n - 1} parallel copies", room=h + 190)
    pg.need(h + 60)
    cl, cw, _sx, _sw = _snip_layout(snip)
    x0, y0, w = cl + 150, pg.y, cw - 330
    d = pg.d
    if snip is not None:
        _paste_snip(pg, snip, y0, h)
    gap = stack_gap(profs)
    zmin = min(float(np.min(p["z"])) for p in profs)
    zmax = max(float(np.max(p["z"])) for p in profs) + gap * (n - 1)
    pad = (zmax - zmin) * 0.06 or 1.0
    lo, hi = zmin - pad, zmax + pad
    xmax = float(profs[0]["dist"][-1]) or 1.0
    top, bot = y0 + 20, y0 + h - 70
    X = lambda v: x0 + v / xmax * w                                              # noqa: E731
    Y = lambda v, k: bot - ((v - lo) + k * gap) / (hi - lo) * (bot - top)       # noqa: E731
    d.rectangle((x0, top - 10, x0 + w, bot + 10), outline=RULE, width=2)
    for k in range(n - 1, -1, -1):                                               # the back ones first
        p = profs[k]
        dist, z = np.asarray(p["dist"], float), np.asarray(p["z"], float)
        pts = [(X(a), Y(b, k)) for a, b in zip(dist, z)]
        base = Y(zmin, k)
        d.line((x0, base, x0 + w, base), fill=RULE, width=1)
        d.polygon(pts + [(pts[-1][0], base), (pts[0][0], base)], fill=CARD)
        main = abs(float(p.get("offset", 0.0))) < 1e-9
        d.line(pts, fill=SECTION if main else _lighter(SECTION, 0.15), width=5 if main else 3)
        d.text((x0 + w + 16, pts[-1][1] - 14), ("0" if main else f"{float(p.get('offset', 0.0)):+.3g}"), font=F(22, main), fill=INK2)
    lu = sec.get("length_unit", "")
    zu = str(sec.get("unit", "")).split()[0]
    bar = _nice_step(gap * 1.2)
    d.line((x0 - 24, Y(zmin, 0), x0 - 24, Y(zmin + bar, 0)), fill=INK2, width=4)
    d.text((cl, Y(zmin + bar / 2, 0) - 14), f"{bar:g} {zu}", font=F(22), fill=INK3)
    d.text((x0, y0 + h - 40), f"0 {lu}", font=F(24), fill=INK3)
    end = f"{xmax:.3g} {lu}"
    d.text((x0 + w - d.textlength(end, font=F(24)), y0 + h - 40), end, font=F(24), fill=INK3)
    d.text((x0 + w + 16, top - 40), f"{lu} off", font=F(22), fill=INK3)
    note = (f"{n} parallel sections {sec.get('spacing', 0):g} {lu} apart; offsets in {lu}, positive to the right "
            f"looking from {a_} to {b_}. Heights in {zu}, each at its own baseline (one height bar for all). "
            f"{a_}–{b_}: {sec['length']:.3g} {sec['length_unit']} long, Rq {sec.get('rq', 0):.3g} {zu}.")
    for j, line in enumerate(_wrap(d, note, F(26), cw)[:4]):
        d.text((cl, y0 + h + 12 + j * 34), line, font=F(26), fill=INK2)
    pg.y = y0 + h + 150


def _section_chart(pg: _Page, sec: dict, a_: str, b_: str, h: int, snip=None):
    if not sec or sec.get("dist") is None or len(sec["dist"]) < 2:
        return
    pg.heading(f"Section {a_}–{b_}", room=h + 170)
    pg.need(h + 60)
    cl, cw, _sx, _sw = _snip_layout(snip)
    x0, y0, w = cl + 120, pg.y, cw - 160
    d = pg.d
    if snip is not None:
        _paste_snip(pg, snip, y0, h)
    dist, z = np.asarray(sec["dist"], float), np.asarray(sec["z"], float)
    zl, zh = float(z.min()), float(z.max())
    if zh - zl < 1e-12:
        zh = zl + 1.0
    X = lambda v: x0 + (v - dist[0]) / max(dist[-1] - dist[0], 1e-12) * w        # noqa: E731
    Y = lambda v: y0 + h - 60 - (v - zl) / (zh - zl) * (h - 120)                   # noqa: E731
    d.rectangle((x0, y0, x0 + w, y0 + h - 40), outline=RULE, width=2)
    pts = [(X(a), Y(b)) for a, b in zip(dist, z)]
    d.polygon(pts + [(pts[-1][0], y0 + h - 40), (pts[0][0], y0 + h - 40)], fill=(222, 233, 246))
    d.line(pts, fill=SECTION, width=5)
    _marker(d, pts[0][0], pts[0][1], a_, SECTION)
    _marker(d, pts[-1][0], pts[-1][1], b_, SECTION)
    zu = str(sec.get("unit", "")).split()[0]
    for k in range(5):
        v = zl + (zh - zl) * k / 4
        nd = max(0, 2 - int(math.floor(math.log10(max(zh - zl, 1e-12)))))          # the decimals the range deserves
        d.text((cl, Y(v) - 14), f"{round(v, nd) + 0.0:g}", font=F(22), fill=INK3)
    d.text((x0, y0 + h - 30), f"0 {sec['length_unit']}", font=F(24), fill=INK3)
    end = f"{dist[-1]:.3g} {sec['length_unit']}"
    d.text((x0 + w - d.textlength(end, font=F(24)), y0 + h - 30), end, font=F(24), fill=INK3)
    note = (f"Height in {zu} along the line {a_}–{b_} drawn on the large view: {sec['length']:.3g} "
            f"{sec['length_unit']} long, roughness Rq {sec.get('rq', 0):.3g} {zu}.")
    lines = _wrap(d, note, F(26), cw)[:3]
    for j, line in enumerate(lines):
        d.text((cl, y0 + h + 10 + j * 34), line, font=F(26), fill=INK2)
    pg.y = y0 + h + 80 + 34 * (len(lines) - 1)


def _details(pg: _Page, s: Sheet, box):
    if not s.details or not s.detail_views:
        return
    pg.heading("Details", room=500)
    x1, y1, x2, y2 = box
    pw, ph = s.full_size
    name_w = 300
    gx = M + name_w
    n = len(s.detail_views)
    cw = (PAGE_W - M - gx - 20 * (n - 1)) // n
    pg.need(60)
    for j, v in enumerate(s.detail_views):
        pg.d.text((gx + j * (cw + 20), pg.y), _short(v.name), font=F(28, True), fill=INK)
    pg.y += 56
    for i, (name, (dx1, dy1, dx2, dy2)) in enumerate(s.details, 1):
        ah, aw = s.albedo.shape[:2]
        c = [int(dx1 * aw / pw), int(dy1 * ah / ph), int(dx2 * aw / pw), int(dy2 * ah / ph)]
        if c[2] - c[0] < 4 or c[3] - c[1] < 4:
            continue
        rh = int(min(560, cw * (c[3] - c[1]) / max(1, c[2] - c[0])))
        rh = max(rh, 220)
        pg.need(rh + 40)
        _marker(pg.d, M + 30, pg.y + 30, str(i), ACCENT)
        for k, line in enumerate(_wrap(pg.d, name, F(26, True), name_w - 40)[:3]):
            pg.d.text((M, pg.y + 70 + k * 32), line, font=F(26, True), fill=INK2)
        for j, v in enumerate(s.detail_views):
            part = _rgb(v.picture)[c[1]:c[3], c[0]:c[2]]
            part = balanced(v.name, part)
            tile, _ = _fit(part, cw, rh, bg=(238, 236, 232))
            pg.im.paste(tile, (gx + j * (cw + 20), pg.y))
        pg.y += rh + 30
    pg.y += 30


def _shots(pg: _Page, s: Sheet):
    if not s.shots:
        return
    pg.heading(s.shots_heading, room=700)
    n = min(3, len(s.shots))
    cw = (PAGE_W - 2 * M - 40 * (n - 1)) // n
    for start in range(0, len(s.shots), n):
        row = [(c, trim(im)) for c, im in s.shots[start:start + n]]
        hs = [min(int(cw * im.shape[0] / max(1, im.shape[1])), 900) for _c, im in row]
        h = max(hs)
        pg.need(h + 110)
        for j, (cap, im) in enumerate(row):
            bg = tuple(int(v) for v in _rgb(im)[0, 0])          # the 3D view's own background
            tile, _ = _fit(_rgb(im), cw, h, bg=bg)
            pg.im.paste(tile, (M + j * (cw + 40), pg.y))
            pg.d.text((M + j * (cw + 40), pg.y + h + 14), cap, font=F(26), fill=INK2)
        pg.y += h + 80


def _tools(pg: _Page, s: Sheet):
    """The tools' results (a legend unrolled, a line drawing, the zones ...):
    a long strip across the page on its own, others three to a row."""
    if not s.tools:
        return
    pg.heading("From the tools", room=500)
    W = PAGE_W - 2 * M
    i = 0
    while i < len(s.tools):
        cap, im = s.tools[i]
        im = _rgb(im)
        if im.shape[1] > 2.5 * im.shape[0]:                    # a strip: the full width
            row = [(cap, im)]
            i += 1
        else:
            row = []
            while i < len(s.tools) and len(row) < 3 and not (
                    _rgb(s.tools[i][1]).shape[1] > 2.5 * _rgb(s.tools[i][1]).shape[0]):
                row.append((s.tools[i][0], _rgb(s.tools[i][1])))
                i += 1
        n = len(row)
        cw = (W - 40 * (n - 1)) // n
        h = max(min(int(cw * r.shape[0] / max(1, r.shape[1])), 900) for _c, r in row)
        pg.need(h + 110)
        for j, (c, r) in enumerate(row):
            tile, (_x, _y, _w, ih) = _fit(r, cw, h)
            pg.im.paste(tile, (M + j * (cw + 40), pg.y))
            for k, line in enumerate(_wrap(pg.d, c, F(26), cw)[:2]):
                pg.d.text((M + j * (cw + 40), pg.y + h + 14 + 34 * k), line, font=F(26), fill=INK2)
        pg.y += h + 110


def _light_dome(size: int, lights):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (size, size), CARD)
    d = ImageDraw.Draw(im)
    c, r = size / 2, size / 2 - 16
    for k in (1.0, 0.66, 0.33):
        d.ellipse((c - r * k, c - r * k, c + r * k, c + r * k), outline=RULE, width=2)
    d.line((c - r, c, c + r, c), fill=RULE, width=1)
    d.line((c, c - r, c, c + r), fill=RULE, width=1)
    for lx, ly, _lz in np.asarray(lights, float):
        if np.isfinite(lx) and np.isfinite(ly):
            px, py = c + lx * r, c - ly * r
            d.ellipse((px - 7, py - 7, px + 7, py + 7), fill=ACCENT)
    d.text((c - 8, 0), "N", font=F(20, True), fill=INK3)
    return im


FOOTER_STRETCH = 0.12        # a page this much taller, rather than a page for the footer alone


def _footer(pg: _Page, s: Sheet):
    lit = s.lights is not None and len(s.lights)
    h = max(30 + 240 if lit else 0, 30 + (100 if lit else 0) + 38 * len(s.footer)) + 20
    short = pg.y + h + M - pg.im.height
    if 0 < short <= FOOTER_STRETCH * PAGE_H and pg.y > M + 10:
        pg.grow(short)            # kept with what it closes: the page scaled a little onto the paper
    else:
        pg.need(h)
    pg.d.line((M, pg.y, PAGE_W - M, pg.y), fill=RULE, width=3)
    y = pg.y + 30
    x = M
    if lit:
        pg.im.paste(_light_dome(240, s.lights), (M, y))
        pg.d.text((M + 270, y), "How it was lit", font=F(28, True), fill=INK)
        pg.d.text((M + 270, y + 40), f"{len(s.lights)} lights, seen from above", font=F(24), fill=INK2)
        x = M + 270
        y += 100
    for k, line in enumerate(s.footer):
        pg.d.text((x, y + k * 38), line, font=F(26), fill=INK2 if k else INK)
    pg.y = max(pg.y + h, y + 38 * len(s.footer) + 20)


def _compass(s: Sheet):
    """The second page: lit from eight directions round the normal map."""
    if not s.compass:
        return None
    pg = _Page()
    pg.d.text((M, pg.y), s.title + "  ·  lit from eight directions", font=F(60, True), fill=INK)
    pg.y += 130
    box = object_box(s.albedo)
    x1, y1, x2, y2 = box
    # on an oval, so eight portrait or landscape pictures fill the page
    cx, cy, Rx, Ry = PAGE_W // 2, 2620, 1200, 1700
    t = 1150
    if s.normals is not None:
        nm = np.clip((np.asarray(s.normals, float) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        tile, (_ix, iy, _iw, ih) = _fit(nm[y1:y2, x1:x2], t, t)
        pg.im.paste(tile, (cx - t // 2, cy - t // 2))
        # captions go under the picture, not under the square it is fitted in: a wide
        # object left them hundreds of pixels below it, beside the next picture
        lab = "surface directions"
        pg.d.text((cx - pg.d.textlength(lab, font=F(28, True)) / 2, cy - t // 2 + iy + ih + 12), lab,
                  font=F(28, True), fill=INK2)
    names = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}
    size = 780
    for az, img in sorted(s.compass.items()):
        a = math.radians(az)
        x, y = cx + Rx * math.sin(a), cy - Ry * math.cos(a)
        tile, (_ix, iy, _iw, ih) = _fit(stretch(_rgb(img)[y1:y2, x1:x2]), size, size)
        pg.im.paste(tile, (int(x - size / 2), int(y - size / 2)))
        lab = f"lit from the {names.get(az, f'bearing {az}°')}"
        pg.d.text((x - pg.d.textlength(lab, font=F(28, True)) / 2, y - size / 2 + iy + ih + 8), lab,
                  font=F(28, True), fill=INK2)
    pg.y = cy + Ry + size / 2 + 90
    pg.d.text((M, pg.y), "Each view is the surface relit from that side, low (25° up), from its "
                         "normals: marks that run across the light show, marks along it fade.",
              font=F(28), fill=INK2)
    return pg.im


PAPER_SIZES_MM = {"A3": (297, 420), "A4": (210, 297), "Letter": (215.9, 279.4)}


def to_paper(im, paper: str, dpi: int = DPI):
    """An A3 page (PAGE_W x PAGE_H) on *paper* at *dpi*: A3 and A4 are the same
    shape, so it is scaled; US Letter is squatter, so the page is scaled to fit
    and centred on the paper colour."""
    from PIL import Image
    w_mm, h_mm = PAPER_SIZES_MM.get(paper, PAPER_SIZES_MM["A3"])
    W, H = round(w_mm / 25.4 * dpi), round(h_mm / 25.4 * dpi)
    if paper not in ("Letter",) and im.size == (W, H):
        return im
    k = min(W / im.width, H / im.height)
    fitted = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    if fitted.size == (W, H):
        return fitted
    out = Image.new("RGB", (W, H), PAPER)
    out.paste(fitted, ((W - fitted.width) // 2, (H - fitted.height) // 2))
    return out


def turned(s: Sheet) -> Sheet:
    """*s* as the screen shows the object (s.turn, a viewing.ScreenTurn): every
    picture of the object mirrored and turned, the section and measured lines,
    notes and boxed details moved with it, the lights turned.  The 3D pictures and
    the tools' results are taken as they are."""
    import copy
    t = s.turn
    if t is None or not t.any:
        return s
    W, H = s.full_size
    out = copy.copy(s)
    out.turn = None
    out.full_size = t.size(W, H)

    def pt(p):
        return t.point(float(p[0]), float(p[1]), W, H)

    def view(v):
        return None if v is None else View(v.name, t.image(_rgb(v.picture)), v.blurb, v.family)

    def section(sec):
        if not sec:
            return sec
        sec = dict(sec)
        for k in ("a_px", "b_px"):
            if k in sec:
                sec[k] = pt(sec[k])
        if sec.get("series"):
            sec["series"] = [section(c) for c in sec["series"]]
        return sec
    out.albedo = t.image(_rgb(s.albedo))
    out.views = [view(v) for v in s.views]
    out.hero = view(s.hero)
    out.detail_views = [view(v) for v in s.detail_views]
    out.section = section(s.section)
    out.sections = [section(c) for c in s.sections]
    out.measures = [(*pt((m[0], m[1])), *pt((m[2], m[3])), *m[4:]) for m in s.measures]
    out.notes = [dict(a, pts=[pt(q) for q in a.get("pts") or []]) for a in s.notes]
    out.details = [(name, t.box(b, W, H)) for name, b in s.details]
    out.lights = None if s.lights is None else t.vectors(np.asarray(s.lights, np.float64))
    if s.normals is not None:
        out.normals = t.image(t.vectors(np.asarray(s.normals, np.float32)), (0.0, 0.0, 1.0))
    if s.compass:
        out.compass = {int(round(t.bearing(az))) % 360: t.image(_rgb(img)) for az, img in s.compass.items()}
    return out


def _in_colour(a) -> bool:
    """A picture in colour (the photograph's, or a view drawn over it) rather than a
    grey reading of the surface: the contrast stretch that suits a reading made a
    photograph's colour vivid and hard."""
    f = _rgb(a)
    if f.ndim != 3 or f.shape[2] < 3:
        return False
    small = f[::max(1, f.shape[0] // 200), ::max(1, f.shape[1] // 200)].astype(np.int16)
    return float(np.abs(small[..., 0] - small[..., 2]).mean()) > 8.0


def balanced(name: str, a) -> np.ndarray:
    """A view as the sheet prints it: grey readings stretched (stretch), pictures in
    colour as the screen draws them."""
    return _rgb(a) if name in COLOURED or _in_colour(a) else stretch(a)


def make(s: Sheet, path_base: str) -> list[str]:
    """The sheet as *path_base*.jpg and .pdf (and .png when asked), on A3 or
    A4, with the compass as a second page when asked.  Returns the files
    written."""
    s = turned(s)
    box = object_box(s.albedo)
    pg = _Page()
    _header(pg, s)
    _boxes(pg, s)
    pg.heading("The object and its views", room=1500)
    _hero_and_grid(pg, s, box)
    _section_charts(pg, s)
    _details(pg, s, box)
    _tools(pg, s)
    _shots(pg, s)
    _footer(pg, s)
    pages = pg.pages()
    second = _compass(s)
    dpi = DPI if s.quality != "screen" else DPI // 2
    pages = [to_paper(im, s.paper, dpi) for im in pages]
    second = None if second is None else to_paper(second, s.paper, dpi)
    files = []

    from helio.io.colour_space import srgb_profile_bytes
    icc = srgb_profile_bytes()                     # says what its numbers mean, wherever it is opened

    def write(im, base):
        im.save(base + ".jpg", quality=92 if dpi == DPI else 85, dpi=(dpi, dpi), icc_profile=icc)
        files.append(base + ".jpg")
        if s.png:
            im.save(base + ".png", dpi=(dpi, dpi), icc_profile=icc)
            files.append(base + ".png")
    for k, im in enumerate(pages, 1):
        write(im, path_base if k == 1 else f"{path_base}_page{k}")
    if second is not None:
        pages.append(second)
        write(second, path_base + "_light_compass")
    pdf = path_base + ".pdf"
    pages[0].save(pdf, "PDF", resolution=dpi, save_all=True, append_images=pages[1:])
    files.append(pdf)
    return files
