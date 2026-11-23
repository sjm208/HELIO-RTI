"""The analyses the publication sheet can work out for itself, at their usual
settings, under "From the tools": tool marks, pits, a line drawing,
inscription zones and the impression.  Each is what its Research tool shows
(helio_qt.research), made here without opening the tool; a result made in
the tool with other settings and added to the sheet goes on as it is.

They are worked out at up to WORK_SIDE px on the long side: as sharp as a
quarter of an A3 page prints, and quick on a 100 MP capture."""
from __future__ import annotations

import numpy as np

WORK_SIDE = 3000

ANALYSES = {
    "toolmarks": ("Tool marks", "Grooves, scratches and tool marks traced as strokes (Toolpaths): "
                                "the colour is the way each runs, brighter is deeper."),
    "pits": ("Pits and stylus marks", "Pinholes, pricking and small pits found in the height "
                                      "(Micro-features), ringed."),
    "lines": ("Line drawing", "Cut lines (black) and raised edges (grey) traced from the height, "
                              "over the object faintly."),
    "zones": ("Inscription zones", "Where writing or worked decoration most likely is, numbered "
                                   "by how much there is."),
    "impression": ("Virtual impression", "The surface turned inside out and mirrored, as a seal "
                                         "or an intaglio would print."),
}


def _work(D):
    """(k, resize) for working at up to WORK_SIDE px: k is work px per full px."""
    import cv2
    w, h = int(D.w), int(D.h)
    k = min(1.0, WORK_SIDE / float(max(w, h)))
    size = (max(16, int(round(w * k))), max(16, int(round(h * k))))

    def resize(a, interp=cv2.INTER_AREA):
        if a is None:
            return None
        a = np.asarray(a)
        if a.shape[:2] == (size[1], size[0]):
            return a
        return cv2.resize(np.ascontiguousarray(a), size, interpolation=interp)
    return k, resize


def _normals(n):
    n = np.asarray(n, np.float32)
    return n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-6)


def make(key: str, D, *, mm_per_px: float = 0.0, z_mm_per_unit: float = 0.0, kind: str = ""):
    """(caption, RGB picture, in the solve's orientation) for the analysis *key*
    of ANALYSES, or None when it found nothing to show.  *kind* is the object
    profile's key (helio.domain.object_profiles), for the zones and impression."""
    import cv2
    title, blurb = ANALYSES[key]
    k, resize = _work(D)
    albedo = resize(D.albedo_rgb_full)
    albedo = None if albedo is None else np.asarray(albedo, np.uint8)[..., :3]
    mm = (mm_per_px / k) if mm_per_px else None
    if key == "toolmarks":
        from helio.application.heights import height_px
        from helio.application.research import groove_smoothing, stroke_colours, toolpaths
        hts = height_px(D)
        depth = resize(np.asarray(D.depth_vis_p if hts is None else hts, np.float32))
        if hts is not None:
            depth = depth * np.float32(k)              # px of height, at the working size
        normals = _normals(resize(D.normals_full, cv2.INTER_LINEAR))
        strokes, labels, skel = toolpaths(depth, normals=normals, use_normals=True,
                                          smooth=groove_smoothing(depth.shape[1], depth.shape[0]),
                                          mm_per_px=mm, z_mm_per_unit=(z_mm_per_unit or None)
                                          if hts is not None else None)
        if not strokes:
            return None
        img = (albedo if albedo is not None else np.full((*labels.shape, 3), 200, np.uint8))
        img = (img.astype(np.float32) * 0.55).astype(np.uint8)
        lut = np.zeros((int(labels.max()) + 1, 3), np.uint8)
        keep = np.zeros(len(lut), bool)
        for sid, rgb in stroke_colours(strokes, by_direction=True, by_depth=True).items():
            lut[sid] = rgb
            keep[sid] = True
        on = (cv2.dilate(skel, np.ones((2, 2), np.uint8)) > 0) & keep[labels]
        img[on] = lut[labels[on]]
        unit = "mm" if strokes[0].get("length_mm") is not None else "px"
        total = sum(s["length_mm"] if unit == "mm" else s["length_px"] for s in strokes)
        return (f"{title}: {len(strokes)} stroke{'s' if len(strokes) != 1 else ''}, {total:.0f} {unit} in all. Colour: the way each "
                "runs (red across, cyan up and down); brighter is deeper.", img)
    if key == "pits":
        from helio.application.research import groove_smoothing, micro_features
        depth = resize(np.asarray(D.depth_vis_p, np.float32))
        s = groove_smoothing(depth.shape[1], depth.shape[0])
        rows = micro_features(depth, min_r=round(2 * s), max_r=round(12 * s), mm_per_px=mm)
        if not rows:
            return None
        base = albedo if albedo is not None else cv2.cvtColor(
            (np.clip(depth, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        img = (base.astype(np.float32) * 0.7).astype(np.uint8)
        t = max(1, int(round(max(img.shape[:2]) / 1200)))
        for r in rows:
            cv2.circle(img, (int(r["x_px"]), int(r["y_px"])), max(3, int(round(r["radius_px"] * 1.6))),
                       (255, 68, 102), t, cv2.LINE_AA)
        return f"{title}: {len(rows)} found, ringed." if len(rows) > 1 else f"{title}: one found, ringed.", img
    if key == "lines":
        from helio.application.heritage_tools import draw_lines, line_drawing
        res = line_drawing(resize(np.asarray(D.depth_vis_p, np.float32)))
        if not res["grooves"] and not res["crests"]:
            return None
        bg = albedo if albedo is not None and albedo.shape[:2] == (res["h"], res["w"]) else None
        return (f"{title}: {len(res['grooves'])} cut lines (black), {len(res['crests'])} raised "
                "edges (grey).", draw_lines(res, bg))
    if key == "zones":
        from helio.application.research import INSCRIPTION_MODES, draw_zones, inscription_zones
        mode = {"coin": INSCRIPTION_MODES[1], "seal": INSCRIPTION_MODES[1],
                "inscription": INSCRIPTION_MODES[2]}.get(kind, INSCRIPTION_MODES[0])
        spec = getattr(D, "spec_var", None)
        boxes, _acc = inscription_zones(resize(np.asarray(D.depth_vis_p, np.float32)),
                                        _normals(resize(D.normals_full, cv2.INTER_LINEAR)),
                                        resize(spec) if spec is not None else None, albedo, mode=mode)
        if not boxes:
            return None
        return f"{title}: " + (f"{len(boxes)} zones, Z1 the densest." if len(boxes) > 1 else "one zone."), draw_zones(albedo, boxes)
    if key == "impression":
        from helio.application.heritage_tools import virtual_impression
        raised = kind == "inscription"
        img = virtual_impression(_normals(resize(D.normals_full, cv2.INTER_LINEAR)),
                                 mirror=not raised, invert=True)
        return (f"{title}: " + ("the cut letters standing up, as on a squeeze." if raised else
                                "as the object would print into wax or plaster."), img)
    raise KeyError(key)
