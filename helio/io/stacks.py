"""One navigable dataset instead of a folder of pictures.

Every other export here writes a single image, or a directory you then open
one file at a time. An ImageJ stack is a different kind of thing: forty render
engines in one window that you scrub through with the arrow keys, each slice
carrying the name of the engine that made it, with real millimetres attached
so Fiji's measuring tools give physical units without being told twice.

ImageJ and Fiji are what conservation and heritage science actually use, so
this is the format that lands in someone's normal working environment rather
than asking them to adopt ours.
"""
from __future__ import annotations

import numpy as np


def write_engine_stack(path: str, images, labels, mm_per_px: float = 0.0,
                       object_id: str = "") -> str:
    """Write an ImageJ-compatible multi-page TIFF.

    *images* are HxWx3 uint8 RGB frames, all the same shape; *labels* names
    each slice. When *mm_per_px* is known the TIFF carries a resolution in
    pixels-per-millimetre and ImageJ's unit is set to mm, so a measurement
    made in Fiji comes out in millimetres rather than pixels — which is the
    difference between a picture and a measurement.

    Returns the path written.
    """
    import tifffile

    frames = [np.ascontiguousarray(im) for im in images]
    if not frames:
        raise ValueError("no engine rendered — nothing to stack")
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"frames differ in shape: {sorted(shapes)}")
    if frames[0].ndim != 3 or frames[0].shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB frames, got {frames[0].shape}")
    if len(labels) != len(frames):
        raise ValueError(f"{len(labels)} labels for {len(frames)} frames")

    arr = np.stack(frames).astype(np.uint8)          # (slices, h, w, 3)

    meta = {"Labels": list(labels), "mode": "color"}
    kwargs = {}
    if mm_per_px and mm_per_px > 0:
        px_per_mm = 1.0 / float(mm_per_px)
        # TIFF stores a rational; ImageJ reads unit from its own metadata.
        kwargs["resolution"] = (px_per_mm, px_per_mm)
        meta["unit"] = "mm"
    if object_id:
        meta["Info"] = f"Helio engine stack\nObject: {object_id}"

    tifffile.imwrite(path, arr, imagej=True, photometric="rgb",
                     metadata=meta, **kwargs)
    return path


def write_light_cube(path: str, frames, labels, *, mm_per_px: float = 0.0,
                     object_id: str = "", info_lines=None) -> str:
    """Write the aligned source frames as an ImageJ data cube.

    Different in kind from :func:`write_engine_stack`, and the difference is
    the point. There the third axis is *which visualisation* — 8-bit, display
    stretched, forty pictures of one surface. Here it is **light direction**,
    a physical variable, and every slice is the same quantity measured under a
    different condition. That is what makes arithmetic down the Z axis mean
    something: in Fiji, ``Image ▸ Stacks ▸ Plot Z-axis Profile`` on a pixel
    plots its reflectance against light angle, which *is* the RTI measurement.
    Variance across the cube is a specularity map, the minimum is a shadow
    mask, the median is the albedo — and anyone can compute their own rather
    than take ours.

    Frames keep their source bit depth: promoting 8-bit JPEGs to 16 would
    imply precision that was never captured.
    """
    import tifffile

    fr = [np.ascontiguousarray(f) for f in frames]
    if not fr:
        raise ValueError("no frames to write")
    shapes = {f.shape for f in fr}
    if len(shapes) != 1:
        raise ValueError(f"frames differ in shape: {sorted(shapes)}")
    if fr[0].ndim != 2:
        raise ValueError(f"expected 2-D greyscale frames, got {fr[0].shape}")
    if len(labels) != len(fr):
        raise ValueError(f"{len(labels)} labels for {len(fr)} frames")

    dt = np.uint16 if any(f.dtype == np.uint16 for f in fr) else np.uint8
    arr = np.stack([f.astype(dt) for f in fr])

    meta = {"Labels": list(labels), "mode": "grayscale"}
    kwargs = {}
    if mm_per_px and mm_per_px > 0:
        px_per_mm = 1.0 / float(mm_per_px)
        kwargs["resolution"] = (px_per_mm, px_per_mm)
        meta["unit"] = "mm"
    info = [f"Helio light cube", f"Object: {object_id or '(unnamed)'}",
            f"Slices: {len(fr)} light directions",
            "Z axis is light direction, not time or depth.",
            "Plot Z-axis Profile on a pixel to see its reflectance function."]
    if info_lines:
        info += list(info_lines)
    meta["Info"] = "\n".join(info)

    tifffile.imwrite(path, arr, imagej=True, photometric="minisblack",
                     metadata=meta, **kwargs)
    return path
