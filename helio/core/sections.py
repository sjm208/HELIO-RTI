"""Parallel sections: a series of copies of one section line, evenly spaced across it.

A section is a line A-B across the relief and the profile of the heights along it.  A series repeats it at
equal steps to either side, so a tool mark, a furrow or a binding's tooling can be read as a set of profiles
rather than one.  Pure geometry, no Qt: the viewer samples the heights (``ReliefView.profile_between``).

Offsets are signed lengths across the line, in the units of the surface's pixels (or mm, as the caller
converts): **positive is to the right-hand side looking from A to B, as seen on screen** (an image has y
down, so for a line drawn left to right, positive is down the screen).
"""
from __future__ import annotations

import math


def offsets(count: int, spacing: float) -> list[float]:
    """*count* offsets *spacing* apart, centred on 0 (the drawn line is the middle one when count is odd)."""
    count = max(1, int(count))
    return [(i - (count - 1) / 2.0) * float(spacing) for i in range(count)]


def offset_lines(a, b, count: int, spacing: float, size_px: tuple[float, float]):
    """The lines of a series, for the line *a* -> *b* (each a (u, v) in 0..1 across a surface that is
    *size_px* = (W, H) pixels), *spacing* apart in those pixels.

    Returns (lines, dropped): ``lines`` is a list of dicts ``offset``, ``a``, ``b`` (the offset in pixels,
    and the ends in (u, v)), from the negative side to the positive; ``dropped`` counts those that would
    leave the surface (they are not clipped: a shorter line would not be a copy of the section).
    """
    (u1, v1), (u2, v2) = a, b
    W, H = float(size_px[0]), float(size_px[1])
    dx, dy = (u2 - u1) * W, (v2 - v1) * H
    length = math.hypot(dx, dy)
    if length <= 0 or W <= 0 or H <= 0:
        raise ValueError("the section has no length")
    nx, ny = -dy / length, dx / length                  # a unit step to the right of A -> B, in pixels
    out, dropped = [], 0
    for off in offsets(count, spacing):
        ou, ov = nx * off / W, ny * off / H
        pa, pb = (u1 + ou, v1 + ov), (u2 + ou, v2 + ov)
        if all(-1e-9 <= c <= 1 + 1e-9 for c in (*pa, *pb)):
            out.append(dict(offset=off, a=pa, b=pb))
        else:
            dropped += 1
    return out, dropped


def stack_gap(profiles: list[dict], factor: float = 0.6) -> float:
    """The vertical step between stacked profiles: *factor* of the tallest one's relief, so neighbours
    overlap a little rather than run together."""
    rng = max((float(max(p["z"]) - min(p["z"])) for p in profiles), default=0.0)
    return rng * factor if rng > 0 else 1.0


def nice_scale(length_mm: float, target_mm: float = 170.0) -> float:
    """A round drawing scale (paper mm per real mm: 1, 2, 5, 10, 20, 25, 50 ... or 1/2, 1/5 ...) that
    draws *length_mm* about *target_mm* wide, never wider."""
    if length_mm <= 0:
        return 1.0
    raw = target_mm / length_mm
    steps = [1, 2, 2.5, 5]
    best = None
    for e in range(-6, 7):
        for s in steps:
            v = s * 10.0 ** e
            if v <= raw * 1.0000001 and (best is None or v > best):
                best = v
    return best or raw


def scale_label(s: float) -> str:
    """1:1, 40:1 (enlarged) or 1:50 (reduced): a drawing scale as drawn on a sheet."""
    if s >= 1:
        return f"{s:g}:1"
    return f"1:{1 / s:g}"
