"""Two or three views in one sweep or 3D recording: side by side, every view
moving together, or one after another, a full turn each.

A view here is (engine, layer): the engine as Inspect names it ("Standard
PBR", "Normals Unsharp Masking"...) and the layer it is drawn in, Colour
("Combined Colour") or Grey ("Shading").  The frames themselves are drawn by
the callers (helio_qt.publish.light_sweep, MainWindow.record_relief_video);
this lays them out and labels them."""
from __future__ import annotations

import numpy as np

LAYOUTS = {"side": ("Side by side", "Every view at once, moving together: the views compared "
                                    "frame by frame."),
           "after": ("One after another", "A full turn in each view in turn, the view's name in "
                                          "the corner: as long as the turns put together.")}
COLOUR, GREY = "Combined Colour", "Shading"
LAYER_NAMES = {COLOUR: "colour", GREY: "grey"}
MAX_VIEWS = 3
GAP = 0.012             # between panels, as a part of the frame's long side


def label_of(engine: str, layer: str) -> str:
    """"Standard PBR, colour"; a view drawn as on screen keeps its own name."""
    from helio.domain.registry import view_name
    name = view_name(engine)
    return f"{name}, {LAYER_NAMES[layer]}" if layer in LAYER_NAMES else name


def grid(n: int, w: int, h: int, aspect: float = 16 / 9) -> tuple[int, int]:
    """(rows, columns) for *n* panels *w* x *h*: the arrangement whose whole
    is nearest *aspect* (a slide's 16:9), so a tall object goes in a row and a
    wide one in a column; an empty cell counts against a grid."""
    best = None
    for rows in range(1, n + 1):
        cols = -(-n // rows)
        a = (cols * w) / float(rows * h)
        score = abs(np.log(a / aspect)) + 1.0 * (rows * cols - n)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    return best[1], best[2]


def _font(px: int):
    from PIL import ImageFont
    for name in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:                                   # noqa: BLE001
            continue
    return ImageFont.load_default(size=px)


def caption(img: np.ndarray, text: str) -> np.ndarray:
    """*img* with *text* in its lower left corner, white on a dark band, sized to
    the picture so it reads on a slide."""
    from PIL import Image, ImageDraw
    if not text:
        return img
    h, w = img.shape[:2]
    px = max(11, int(round(min(w, h) * 0.045)))
    pic = Image.fromarray(np.ascontiguousarray(img[..., :3]))
    d = ImageDraw.Draw(pic, "RGBA")
    f = _font(px)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    pad = max(4, px // 3)
    bx, by = pad, h - (y1 - y0) - 3 * pad
    d.rectangle((bx - pad // 2, by - pad, bx + (x1 - x0) + pad + pad // 2, h - pad), fill=(0, 0, 0, 150))
    d.text((bx + pad // 2 - x0, by - y0), text, font=f, fill=(255, 255, 255, 255))
    return np.asarray(pic)


def tile(pictures, labels=(), background=(0, 0, 0), rows: int = 0, cols: int = 0) -> np.ndarray:
    """*pictures* (RGB, any sizes) laid out in a grid of equal cells (by
    grid(), unless *rows* and *cols* are given), each fitted to its cell and
    centred on *background*, each labelled with its entry of *labels*."""
    import cv2
    pics = [np.asarray(p)[..., :3] for p in pictures]
    if len(pics) == 1 and not labels:
        return pics[0]
    cw = max(p.shape[1] for p in pics)
    ch = max(p.shape[0] for p in pics)
    if not rows:
        rows, cols = grid(len(pics), cw, ch)
    gap = max(2, int(round(GAP * max(cw * cols, ch * rows))))
    W, H = cols * cw + (cols - 1) * gap, rows * ch + (rows - 1) * gap
    W, H = W + W % 2, H + H % 2
    out = np.empty((H, W, 3), np.uint8)
    out[...] = np.asarray(background, np.uint8)
    for i, p in enumerate(pics):
        r, c = divmod(i, cols)
        ph, pw = p.shape[:2]
        k = min(cw / pw, ch / ph)
        if abs(k - 1) > 1e-3:
            p = cv2.resize(p, (max(1, int(round(pw * k))), max(1, int(round(ph * k)))),
                           interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
            ph, pw = p.shape[:2]
        if i < len(labels):
            p = caption(p, labels[i])
        x = c * (cw + gap) + (cw - pw) // 2
        y = r * (ch + gap) + (ch - ph) // 2
        out[y:y + ph, x:x + pw] = p
    return out


def even(img: np.ndarray, size) -> np.ndarray:
    """*img* fitted into *size* (w, h), centred on black: the frames of one
    after another made one size for the video."""
    import cv2
    w, h = size
    ph, pw = img.shape[:2]
    if (pw, ph) == (w, h):
        return img
    k = min(w / pw, h / ph)
    p = cv2.resize(img, (max(1, int(round(pw * k))), max(1, int(round(ph * k)))),
                   interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - p.shape[0]) // 2, (w - p.shape[1]) // 2
    out[y:y + p.shape[0], x:x + p.shape[1]] = p[..., :3]
    return out
