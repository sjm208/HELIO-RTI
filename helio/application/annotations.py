"""Annotations on a render, GUI-free (SCH annotation layer, REP scholar set).

One model for the classic Annotation Layer (arrows, text, shapes, freehand,
polygons, measurements) and the Scholar Annotation export (named layers, a
three-panel publication set): each annotation is a dict in full-image pixel
coordinates with its layer's name.  ``burn`` draws them onto a picture as
the classic export did; ``svg`` writes them as vector overlays, grouped by
layer; ``scholar_set`` writes the clean / annotated / albedo panels, JSON
and SVG.

Fixed from HELIO 2: a measurement used ``_px_scale_mm``, which nothing ever
set, so every length assumed 0.1 mm per pixel (KD-044); the SVG's arrows
pointed to an arrowhead marker that was never defined, and its text was not
escaped; the scholar layers had no way to hold an annotation, so the
"annotated" panel and SVG carried only layer names (KD-045).
"""
from __future__ import annotations

import json
import math
import os
from xml.sax.saxutils import escape

import numpy as np

TOOLS = ("arrow", "text", "ellipse", "rect", "line", "freehand", "polygon", "measure", "dot",
         "number")
COLOURS = (("#ff0000", "Red"), ("#ffff00", "Yellow"), ("#00ffff", "Cyan"), ("#ffffff", "White"))
LAYER_PRESETS = (("Letter identifications", "#4488ff", "ellipse"),
                 ("Musical notes", "#ff8844", "ellipse"),
                 ("Pinholes / pricking", "#ffdd44", "dot"),
                 ("Structural damage", "#ff4444", "polygon"),
                 ("Uncertain readings", "#cc88ff", "line"),
                 ("Custom", "#88ffcc", "arrow"))
DEFAULT_LAYER = "Annotations"


def make(kind: str, pts, *, color: str = "#ff0000", text: str = "", fsize: int = 18,
         fill: bool = False, opacity: float = 0.6, width: int = 3,
         layer: str = DEFAULT_LAYER) -> dict:
    return {"type": kind, "pts": [(float(x), float(y)) for x, y in pts], "color": color,
            "text": text, "fsize": int(fsize), "fill": bool(fill), "opacity": float(opacity),
            "width": int(width), "layer": layer}


def length_label(p0, p1, mm_per_px=None) -> str:
    """A measurement's label: millimetres with a scale, else pixels (never a
    guessed scale)."""
    d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return f"{d * mm_per_px:.1f} mm" if mm_per_px else f"{d:.0f} px"


def _rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _font(size: int):
    from PIL import ImageFont
    for name in ("Helvetica", "Arial", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, max(6, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def burn(img, anns, mm_per_px=None, *, visible_layers=None):
    """The annotations drawn onto *img* (uint8 RGB): solid strokes, filled
    shapes at their opacity, as the classic Export PNG.  Returns uint8 RGB."""
    from PIL import Image, ImageDraw
    base = Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for n, a in enumerate(anns, 1):
        if visible_layers is not None and a.get("layer") not in visible_layers:
            continue
        t, lw = a["type"], max(1, int(a.get("width", 3)))
        rgb = _rgb(a["color"])
        solid, translucent = rgb + (255,), rgb + (int(255 * a.get("opacity", 0.6)),)
        pts = [(int(round(x)), int(round(y))) for x, y in a["pts"]]
        if t == "freehand" and len(pts) > 1:
            draw.line(pts, fill=solid, width=lw, joint="curve")
            continue
        if t == "polygon" and len(pts) > 2:
            if a.get("fill"):
                draw.polygon(pts, fill=translucent, outline=solid)
            else:
                draw.line(pts + [pts[0]], fill=solid, width=lw)
            continue
        if t in ("dot", "number"):
            (x, y) = pts[0]
            r = max(3, lw * 2)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=solid)
            if t == "number":
                draw.text((x + r + 2, y - r), str(a.get("text") or n), fill=solid,
                          font=_font(a.get("fsize", 18)))
            continue
        if len(pts) < 2:
            continue
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        if t == "arrow":
            draw.line([(x1, y1), (x2, y2)], fill=solid, width=lw + 1)
            dx, dy = x2 - x1, y2 - y1
            mag = max(1.0, math.hypot(dx, dy))
            ux, uy = dx / mag, dy / mag
            pl, pw = 14 + 2 * lw, 6 + lw
            draw.polygon([(x2, y2), (int(x2 - ux * pl + uy * pw), int(y2 - uy * pl - ux * pw)),
                          (int(x2 - ux * pl - uy * pw), int(y2 - uy * pl + ux * pw))], fill=solid)
        elif t == "text":
            draw.text((x2, y2), a.get("text", ""), fill=solid, font=_font(a.get("fsize", 18)))
        elif t in ("ellipse", "rect"):
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            fn = draw.ellipse if t == "ellipse" else draw.rectangle
            if a.get("fill"):
                fn(box, fill=translucent, outline=solid, width=lw)
            else:
                fn(box, outline=solid, width=lw)
        elif t == "measure":
            draw.line([(x1, y1), (x2, y2)], fill=solid, width=lw)
            ls = int(a.get("lsize", 14))
            draw.text(((x1 + x2) // 2, (y1 + y2) // 2 - ls - 2),
                      a.get("label") or length_label(pts[0], pts[-1], mm_per_px),
                      fill=solid, font=_font(ls))
        else:
            draw.line([(x1, y1), (x2, y2)], fill=solid, width=lw)
    return np.asarray(Image.alpha_composite(base, overlay).convert("RGB"))


def svg(anns, w: int, h: int, mm_per_px=None, *, title: str = "", layers=None,
        picture: str = "") -> str:
    """The annotations as SVG in image pixels, one group per layer, with the
    arrowhead marker defined and every text escaped; *picture*, a JPEG as
    base64, goes underneath so the file shows the view it annotates."""
    order = []
    for a in anns:
        if a.get("layer", DEFAULT_LAYER) not in order:
            order.append(a.get("layer", DEFAULT_LAYER))
    colours = {ly["name"]: ly["color"] for ly in (layers or [])}
    for ly in layers or []:
        if ly["name"] not in order:
            order.append(ly["name"])
    out = ["<?xml version='1.0' encoding='UTF-8'?>",
           f"<svg xmlns='http://www.w3.org/2000/svg' width='{int(w)}' height='{int(h)}' "
           f"viewBox='0 0 {int(w)} {int(h)}'>"]
    if title:
        out.append(f"  <title>{escape(title)}</title>")
    if picture:
        out.append(f"  <image id='view' x='0' y='0' width='{int(w)}' height='{int(h)}' "
                   f"href='data:image/jpeg;base64,{picture}'/>")
    out.append("  <defs><marker id='ah' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' "
               "markerHeight='6' orient='auto-start-reverse'><path d='M0,0 L10,5 L0,10 z' "
               "fill='context-stroke'/></marker></defs>")
    for name in order:
        gid = "layer_" + "".join(c if c.isalnum() else "_" for c in name)
        stroke = colours.get(name, "")
        out.append(f"  <g id='{gid}' data-name='{escape(name, {chr(39): '&apos;'})}'"
                   + (f" stroke='{stroke}'" if stroke else "") + ">")
        for n, a in enumerate([a for a in anns if a.get("layer", DEFAULT_LAYER) == name], 1):
            t, col, lw = a["type"], a["color"], max(1, int(a.get("width", 3)))
            fill = (f"fill='{col}' fill-opacity='{a.get('opacity', 0.6):.2f}'" if a.get("fill")
                    else "fill='none'")
            pts = [(round(x, 1), round(y, 1)) for x, y in a["pts"]]
            ps = " ".join(f"{x},{y}" for x, y in pts)
            if t == "freehand":
                out.append(f"    <polyline points='{ps}' fill='none' stroke='{col}' stroke-width='{lw}'/>")
            elif t == "polygon":
                out.append(f"    <polygon points='{ps}' {fill} stroke='{col}' stroke-width='{lw}'/>")
            elif t in ("dot", "number"):
                x, y = pts[0]
                out.append(f"    <circle cx='{x}' cy='{y}' r='{max(3, lw * 2)}' fill='{col}'/>")
                if t == "number":
                    out.append(f"    <text x='{x + 8}' y='{y}' fill='{col}' font-size='{a.get('fsize', 18)}'>"
                               f"{escape(str(a.get('text') or n))}</text>")
            elif len(pts) >= 2:
                (x1, y1), (x2, y2) = pts[0], pts[-1]
                if t == "arrow":
                    out.append(f"    <line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' "
                               f"stroke-width='{lw + 1}' marker-end='url(#ah)'/>")
                elif t == "text":
                    out.append(f"    <text x='{x2}' y='{y2}' fill='{col}' font-size='{a.get('fsize', 18)}' "
                               f"font-weight='bold'>{escape(a.get('text', ''))}</text>")
                elif t == "ellipse":
                    out.append(f"    <ellipse cx='{(x1 + x2) / 2}' cy='{(y1 + y2) / 2}' "
                               f"rx='{abs(x2 - x1) / 2}' ry='{abs(y2 - y1) / 2}' {fill} stroke='{col}' "
                               f"stroke-width='{lw}'/>")
                elif t == "rect":
                    out.append(f"    <rect x='{min(x1, x2)}' y='{min(y1, y2)}' width='{abs(x2 - x1)}' "
                               f"height='{abs(y2 - y1)}' {fill} stroke='{col}' stroke-width='{lw}'/>")
                elif t == "measure":
                    out.append(f"    <line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' "
                               f"stroke-width='{lw}' stroke-dasharray='6,3'/>")
                    out.append(f"    <text x='{(x1 + x2) / 2}' y='{(y1 + y2) / 2 - 8}' fill='{col}' "
                               f"font-size='{a.get('lsize', 12)}' font-weight='bold'>"
                               f"{escape(a.get('label') or length_label(pts[0], pts[-1], mm_per_px))}"
                               "</text>")
                else:
                    out.append(f"    <line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{col}' "
                               f"stroke-width='{lw}'/>")
        out.append("  </g>")
    out.append("</svg>")
    return "\n".join(out)


def save_json(path: str, anns, *, layers=(), notes=(), meta: dict | None = None) -> str:
    with open(path, "w") as f:
        json.dump({"helio_annotations": 1, "meta": meta or {}, "layers": list(layers),
                   "notes": list(notes), "annotations": list(anns)}, f, indent=2)
    return path


def load_json(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    anns = [make(a["type"], a["pts"], color=a.get("color", "#ff0000"), text=a.get("text", ""),
                 fsize=a.get("fsize", 18), fill=a.get("fill", False), opacity=a.get("opacity", 0.6),
                 width=a.get("width", 3), layer=a.get("layer", DEFAULT_LAYER))
            for a in d.get("annotations", [])]
    return {"annotations": anns, "layers": d.get("layers", []), "notes": d.get("notes", []),
            "meta": d.get("meta", {})}


EXPORT_SIDES = {"2,000 px": 2000, "4,000 px": 4000, "Full size": 0}
SCREEN_FIT = 1200            # the long side, in screen pixels, of a picture fitted to the screen


def scaled(anns, k: float, s: float, mm_per_px=None) -> list[dict]:
    """The annotations for a picture *k* times the image's size, drawn *s*
    times their on-screen size: positions times *k*; line widths, text and
    dots times *s*.  A measurement keeps its true length as its label."""
    out = []
    for a in anns:
        b = dict(a)
        b["pts"] = [(x * k, y * k) for x, y in a["pts"]]
        b["width"] = max(1.0, float(a.get("width", 3)) * s)
        b["fsize"] = max(8, int(round(float(a.get("fsize", 18)) * s)))
        b["lsize"] = max(8, int(round(14 * s)))
        if a["type"] == "measure" and len(a["pts"]) >= 2:
            b["label"] = length_label(a["pts"][0], a["pts"][-1], mm_per_px)
        out.append(b)
    return out


def export_size(img_shape, long_side: int = 4000, screen_long: float | None = None):
    """(k, s) for an export *long_side* pixels on its long side (0: full size):
    *k* image pixels to export pixels, *s* screen pixels to export pixels for
    the annotations' widths and text, so they look as on screen with the
    whole picture in view (*screen_long* px on its long side there)."""
    h, w = img_shape[:2]
    k = 1.0 if not long_side else min(1.0, long_side / float(max(h, w)))
    s = max(1.0, max(h, w) * k / float(screen_long or SCREEN_FIT))
    return k, s


def annotated_render(img, anns, path: str, mm_per_px=None, title: str = "", *,
                     long_side: int = 4000, screen_long: float | None = None,
                     quality: int = 92, layers=None, image_size=None) -> list[str]:
    """The annotated view: JPEG (or PNG or TIFF, by the file's name) at
    *long_side* pixels on its long side (0: full size), the annotations drawn
    at the size they show on screen; and an SVG beside it with the view
    underneath and each layer's annotations in a group of its own.
    *image_size* (w, h): the pixels the annotations are in, when *img* is
    a different size (the on-screen render, when that is big enough).

    It used to write the full-size render as an uncompressed PNG (280 MB
    for a binding) with the annotations at their screen widths, a few
    pixels on a 100-megapixel picture, and an SVG of those lines alone."""
    import base64
    import cv2
    a = np.asarray(img)
    full = image_size or (a.shape[1], a.shape[0])           # the pixels the annotations are in
    k, s = export_size((full[1], full[0]), long_side, screen_long)
    tw, th = max(1, int(round(full[0] * k))), max(1, int(round(full[1] * k)))
    if (a.shape[1], a.shape[0]) != (tw, th):
        a = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA if a.shape[1] > tw else cv2.INTER_LINEAR)
    a = np.ascontiguousarray(a[..., :3] if a.ndim == 3 else np.repeat(a[..., None], 3, axis=2))
    sa = scaled(anns, k, s, mm_per_px)
    out = burn(a, sa, mm_per_px)
    ext = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else []
    cv2.imwrite(path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR), params)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    sp = os.path.splitext(path)[0] + ".svg"
    h, w = a.shape[:2]
    with open(sp, "w") as f:
        f.write(svg(sa, w, h, mm_per_px, title=title, layers=layers,
                    picture=base64.b64encode(jpg.tobytes()).decode() if ok else ""))
    return [path, sp]


def scholar_set(out_dir: str, safe: str, depth, albedo, anns, layers, *, notes: str = "",
                include_albedo: bool = True, mm_per_px=None, obj_id: str = "", solver: str = "",
                stamp: str = "", long_side: int = 4000, screen_long: float | None = None
                ) -> list[str]:
    """The three-panel publication set: the clean relief render (equalised
    depth), the annotated render (visible layers' annotations, a legend and
    the notes), the albedo overlay, the JSON record and the SVG."""
    import datetime
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = np.asarray(depth, np.float64)
    dn = ((d - d.min()) / (np.ptp(d) + 1e-9) * 255).astype(np.uint8)
    base = cv2.cvtColor(cv2.equalizeHist(dn), cv2.COLOR_GRAY2RGB)
    # At *long_side* pixels (0: full size), the annotations at their screen
    # size: the panels were full size and drawn at twice that (17,000 px
    # across for a binding), the annotations a few pixels wide on them.
    H0, W0 = base.shape[:2]
    record = anns                                   # the JSON keeps the image's own pixels
    k, s = export_size(base.shape, long_side, screen_long)
    if k < 1.0:
        base = cv2.resize(base, (max(1, int(round(base.shape[1] * k))),
                                 max(1, int(round(base.shape[0] * k)))), interpolation=cv2.INTER_AREA)
    anns = scaled(anns, k, s, mm_per_px)
    H, W = base.shape[:2]
    visible = [ly for ly in layers if ly.get("visible", True)]
    names = {ly["name"] for ly in visible}
    os.makedirs(out_dir, exist_ok=True)
    clean = os.path.join(out_dir, f"{safe}_{ts}_render_clean.png")
    cv2.imwrite(clean, cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

    def panel(img, path):
        fig, ax = plt.subplots(figsize=(W / 100, H / 100), facecolor="black")
        ax.imshow(burn(img, anns, mm_per_px, visible_layers=names))
        ax.axis("off")
        for i, ly in enumerate(visible):
            ax.text(10 * s, (10 + i * 22) * s, f"■ {ly['name']}", color=ly["color"], fontsize=7 * s,
                    fontweight="bold", bbox=dict(facecolor="black", alpha=0.5, pad=1))
        if notes and path.endswith("_render_annotated.png"):
            ax.text(W * 0.02, H * 0.96, notes, color="white", fontsize=5 * s, verticalalignment="bottom",
                    bbox=dict(facecolor="black", alpha=0.6, pad=3))
        fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="black", pad_inches=0)
        plt.close(fig)
        return path
    files = [clean, panel(base, os.path.join(out_dir, f"{safe}_{ts}_render_annotated.png"))]
    if include_albedo and albedo is not None:
        alb = np.asarray(albedo)
        if alb.shape[:2] != (H, W):
            alb = cv2.resize(alb, (W, H))
        files.append(panel(alb, os.path.join(out_dir, f"{safe}_{ts}_albedo_overlay.png")))
    recs = [{"name": ly["name"], "color": ly["color"], "type": ly.get("type", "ellipse"),
             "annotations": [a for a in record if a.get("layer") == ly["name"]]} for ly in visible]
    jp = os.path.join(out_dir, f"{safe}_{ts}_annotations.json")
    with open(jp, "w") as f:
        json.dump({"object_id": obj_id, "timestamp": ts, "solver": solver, "layers": recs,
                   "notes": notes.split("\n") if notes else [], "image_size": [W0, H0]}, f, indent=2)
    sp = os.path.join(out_dir, f"{safe}_{ts}_annotations.svg")
    with open(sp, "w") as f:
        f.write(svg([a for a in anns if a.get("layer") in names], W, H, mm_per_px,
                    title=f"{obj_id} — Scholar Annotations", layers=visible))
    return files + [jp, sp]
