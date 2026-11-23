"""A quick look: the solve at about 1,000 pixels across, in seconds.

The photographs are read once at a small size and kept for the capture, so a
quick look, and a comparison of the light corrections, can be run again and
again without reading the full-size files.  Nothing here replaces the full
solve: the result is a preview, never stored in the project.

Every box the settings carry (the grey patch, the flat areas, the region) is
in full-size pixels; it is scaled to the small frames here.
"""
from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

QUICK_SIDE = 1000           # the long side of a quick look, in pixels
_cache: dict = {}           # one capture's small frames: {"key", "frames", "full", "scale"}


def _read_small(path: str, side: int):
    """(rgb, grey, is_raw, (full_w, full_h)) at about *side* pixels on the long side.
    JPEG decodes straight to a reduced size; other files are read and shrunk."""
    import cv2
    from helio.io import load_and_process_single
    from helio.io.raw_loader import is_raw
    raw = is_raw(path)
    if not raw:
        full = _size(path)
        if full is not None:
            red = max(1, min(8, max(full) // side))
            red = 8 if red >= 8 else 4 if red >= 4 else 2 if red >= 2 else 1
            flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                    4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[red]
            img = cv2.imread(path, flag)
            if img is not None:
                from helio.io.colour_space import to_srgb
                rgb = cv2.cvtColor(_shrink(img, side), cv2.COLOR_BGR2RGB)
                grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
                return to_srgb(rgb, path), cv2.GaussianBlur(grey, (0, 0), 0.7), False, full
    rgb, grey = load_and_process_single(path)
    if rgb is None:
        return None, None, raw, None
    full = (rgb.shape[1], rgb.shape[0])
    return _shrink(rgb, side), _shrink(grey, side), raw, full


def _size(path: str):
    from helio.io.raw_loader import is_raw, raw_size
    try:
        if is_raw(path):                         # the header PIL sees is the file's thumbnail
            return raw_size(path)
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            w, h = im.size
            turned = im.getexif().get(0x0112, 1) in (5, 6, 7, 8)    # read upright, as OpenCV does
            return (h, w) if turned else (w, h)
    except Exception:
        return None


def _shrink(a, side: int):
    import cv2
    h, w = a.shape[:2]
    s = side / max(h, w)
    if s >= 1:
        return a
    return cv2.resize(a, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def small_frames(paths, side: int = QUICK_SIDE, progress=None) -> dict:
    """The capture's frames at the small size: {"frames", "full": (w, h), "scale"}.
    Read once per capture and kept."""
    paths = list(paths)
    try:
        stamp = tuple(os.path.getmtime(p) for p in paths)
    except OSError:
        stamp = ()
    key = (tuple(paths), side, stamp)
    if _cache.get("key") == key:
        return _cache
    progress = progress or (lambda *a, **k: None)
    out = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futs = {ex.submit(_read_small, p, side): i for i, p in enumerate(paths)}
        for k, f in enumerate(futs):
            out[futs[f]] = f.result()
            progress(f"Reading the photographs small  {k + 1}/{len(paths)}")
    sizes = [o[3] for o in out if o[3] is not None]
    if not sizes:
        raise ValueError("None of the photographs could be read.")
    full = sizes[0]
    first = next(o for o in out if o[0] is not None)
    scale = first[0].shape[1] / float(full[0])
    _cache.clear()
    _cache.update(key=key, frames=[(o[0], o[1], o[2]) for o in out], full=full, scale=scale)
    return _cache


def forget():
    _cache.clear()


def _scale_box(box, s):
    return tuple(int(round(v * s)) for v in box)


def scaled_settings(ps, s: float):
    """The settings with every box scaled from full size by *s*."""
    from helio.core.flash import boxes_text, parse_box, parse_boxes
    kw = {}
    b = parse_box(ps.flash_ref) if ps.flash_ref else None
    if b is not None:
        kw["flash_ref"] = ",".join(str(v) for v in _scale_box(b, s))
    if ps.flat_areas:
        kw["flat_areas"] = boxes_text([_scale_box(b, s) for b in parse_boxes(ps.flat_areas)])
    return dataclasses.replace(ps, **kw) if kw else ps


def small_request(req, small: dict, settings=None):
    """*req*, the solve request, for the small frames."""
    import cv2
    s = small["scale"]
    full_w, full_h = small["full"]
    ps = scaled_settings(settings or req.settings, s)
    crop = None
    if req.crop is not None:
        rw, rh = req.crop_ref_size or (full_w, full_h)
        crop = _scale_box(req.crop, s * full_w / float(rw or full_w))
    flat = req.flat_ref_normals
    if flat is not None:
        fh, fw = flat.shape[:2]
        flat = cv2.resize(np.asarray(flat, np.float32), (max(1, round(fw * s)), max(1, round(fh * s))),
                          interpolation=cv2.INTER_AREA)
    origin = None if req.light_origin is None else tuple(v * s for v in req.light_origin)
    positions = None if req.light_positions is None else np.asarray(req.light_positions) * s
    return dataclasses.replace(req, settings=ps, crop=crop, crop_ref_size=None,
                               flat_ref_normals=flat, frames=small["frames"], light_origin=origin,
                               light_positions=positions,
                               mm_per_px=(req.mm_per_px / s) if req.mm_per_px else None)


def quick_solve(req, progress=None, settings=None):
    """The solve, small.  Returns (SolveResult, scale from full size)."""
    from helio.application import pipeline
    small = small_frames(req.paths, progress=progress)
    r = pipeline.solve(small_request(req, small, settings),
                       progress=(lambda st, f=None, d="": progress and progress(st or d))
                       if progress else None)
    return r, small["scale"]


def tilt_deg(normals, box) -> float | None:
    """How far a box's typical direction leans from straight up, in degrees."""
    x1, y1, x2, y2 = box
    h, w = normals.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    m = np.median(normals[y1:y2, x1:x2].reshape(-1, 3), axis=0)
    n = np.linalg.norm(m)
    if not np.isfinite(n) or n == 0:
        return None
    return float(np.degrees(np.arccos(np.clip(abs(m[2]) / n, 0, 1))))


# The corrections compared, each on its own: (name, settings to apply).
_OFF = dict(even_flash=False, even_flat=False, flatten_illumination=False, near_field=False)
VARIANTS = (
    ("None", dict(_OFF)),
    ("Grey patch", dict(_OFF, even_flash=True)),
    ("Flat areas", dict(_OFF, even_flat=True)),
    ("Flat object", dict(_OFF, flatten_illumination=True)),
    ("Near-field", dict(_OFF, near_field=True)),
)


def compare(req, progress=None) -> dict:
    """Each light correction on its own, solved small, and how flat the
    marked flat things come out: the grey patch and every flat area, and
    the picture as a whole.  {"rows": [...], "refs": [(name, box)], "full"}."""
    from helio.core.flash import parse_box, parse_boxes
    ps = req.settings
    small = small_frames(req.paths, progress=progress)
    s = small["scale"]
    refs = []
    if ps.flash_ref and parse_box(ps.flash_ref):
        refs.append(("Grey patch", parse_box(ps.flash_ref)))
    for i, b in enumerate(parse_boxes(ps.flat_areas) if ps.flat_areas else []):
        refs.append((f"Flat area {i + 1}", b))
    rows = []
    for name, kw in VARIANTS:
        if name == "Grey patch" and not ps.flash_ref:
            continue
        if name == "Flat areas" and not ps.flat_areas:
            continue
        if progress:
            progress(f"Solving small: {name.lower()}")
        st = dataclasses.replace(ps, **kw)
        r = pipeline_solve(req, small, st)
        n = r.normals_full
        off = (0, 0)
        sreq = small_request(req, small, st)
        if sreq.crop is not None:
            off = sreq.crop[:2]
        tilts = []
        for _nm, b in refs:
            sb = _scale_box(b, s)
            tilts.append(tilt_deg(n, (sb[0] - off[0], sb[1] - off[1],
                                      sb[2] - off[0], sb[3] - off[1])))
        known = [t for t in tilts if t is not None]
        whole = tilt_deg(n, (0, 0, n.shape[1], n.shape[0]))
        applied = (name == "None" or (name == "Grey patch" and r.flash is not None)
                   or (name == "Flat areas" and r.flat_light is not None)
                   or name in ("Flat object", "Near-field"))
        rows.append(dict(name=name, tilts=tilts, mean=(float(np.mean(known)) if known else None),
                         worst=(max(known) if known else None), whole=whole, applied=applied,
                         normals=n, depth=r.depth_normalised_full))
    return dict(rows=rows, refs=refs, full=small["full"], scale=s)


def pipeline_solve(req, small, settings):
    from helio.application import pipeline
    return pipeline.solve(small_request(req, small, settings))
