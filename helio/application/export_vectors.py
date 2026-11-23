"""Sections and annotations as GeoJSON, for the GIS package.

What was marked on the object in Helio should arrive on the map in QGIS with the
height map, not be drawn again.  The package's rasters have one unit per pixel
with the origin at the top left of the map (y = height at the top row), so a point
in the solve's pixels goes there directly.  The rasters' CRS is a placeholder for
a flat object, so the files carry none: whoever loads them gives them the raster's.
GUI-free.
"""
from __future__ import annotations

import json
import math
import os

from helio.application.annotations import length_label

_POINTS = ("dot", "number", "text")
_AREAS = ("polygon", "rect", "ellipse")


def to_map(h: int, k: float = 1.0, crop=None):
    """(x, y) in the solve's pixels to map units, for a raster *h* rows high made *k* × the
    solve's size, cut from *crop* (x1, y1, ...)."""
    ox, oy = (crop[0], crop[1]) if crop else (0, 0)
    return lambda x, y: [round((float(x) - ox) * k, 3), round(h - (float(y) - oy) * k, 3)]


def section_features(sections, where, mm_per_px=None) -> list[dict]:
    """*sections*: dicts of name ('A–B'), a, b (the solve's pixels), copies."""
    out = []
    for s in sections:
        (ax, ay), (bx, by) = s["a"], s["b"]
        out.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=[where(ax, ay), where(bx, by)]),
                        properties=dict(name=s["name"], length=length_label((ax, ay), (bx, by), mm_per_px),
                                        copies=int(s.get("copies", 0)))))
    return out


def _shape(a, where):
    t, pts = a["type"], a.get("pts") or []
    if not pts:
        return None
    if t in _POINTS:
        return dict(type="Point", coordinates=where(*pts[0]))
    if t in _AREAS:
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        if t == "polygon":
            ring = [tuple(p) for p in pts] + [tuple(pts[0])]
        elif t == "rect":
            ring = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
        else:
            cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1) / 2, abs(y2 - y1) / 2
            ring = [(cx + rx * math.cos(i * math.pi / 36), cy + ry * math.sin(i * math.pi / 36)) for i in range(73)]
            ring[-1] = ring[0]
        return dict(type="Polygon", coordinates=[[where(x, y) for x, y in ring]]) if len(ring) > 3 else None
    return dict(type="LineString", coordinates=[where(x, y) for x, y in pts]) if len(pts) > 1 else None


def annotation_features(anns, where, mm_per_px=None) -> list[dict]:
    out = []
    for a in anns:
        g = _shape(a, where)
        if g is None:
            continue
        pts = a.get("pts") or []
        label = a.get("text") or a.get("label") or ""
        if a["type"] == "measure" and not label and len(pts) > 1:
            label = length_label(pts[0], pts[-1], mm_per_px)
        out.append(dict(type="Feature", geometry=g, properties=dict(
            kind=a["type"], layer=a.get("layer", ""), label=label, colour=a.get("color", "#ff0000"))))
    return out


def write(out_dir: str, stem: str, *, sections=(), annotations=(), h: int, k: float = 1.0, crop=None,
          mm_per_px=None) -> list[str]:
    """<stem>_sections.geojson and <stem>_annotations.geojson, for those there are."""
    where, files = to_map(h, k, crop), []
    for name, feats in (("sections", section_features(sections, where, mm_per_px)),
                        ("annotations", annotation_features(annotations, where, mm_per_px))):
        if feats:
            path = os.path.join(out_dir, f"{stem}_{name}.geojson")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dict(type="FeatureCollection", name=name, features=feats), f, ensure_ascii=False)
            files.append(path)
    return files
