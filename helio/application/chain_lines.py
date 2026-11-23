"""Chain lines (and laid lines, and ruling) counted along the 3D relief's section.

The counting is helio.core.paper_lines (from MSI_Reveal).  This chooses the
picture it counts on and turns the section's ends and the strip's width into
that module's terms.

Chain and laid lines are where the mould's wires thinned the sheet, so they
show best in transmitted light: a backlit frame (Capture › Add Backlit
Image…), lined up with the solve, is used when there is one, with the albedo
as the reflected-light reference that masks the writing.  Without one the
lines are looked for in the albedo, or in the relief's height, where a laid
sheet's wires often stand as faint ridges in raking light, or in the view
Inspect is showing (a low raking relight, a high-pass view), as it is shown.
"""
from __future__ import annotations

import numpy as np

from helio.core.paper_lines import KINDS, LineCount, count_lines_strip, suggest_swath_mm, swath_strip

#: The pictures the lines can be counted on, with a reader's name.
SOURCES = {"backlit": "Backlit frame (transmitted)", "albedo": "Albedo (reflected)",
           "height": "Relief height", "view": "This view (as shown)"}
#: The kinds offered (helio.core.paper_lines.KINDS less "auto").
LINE_KINDS = {k: v for k, v in KINDS.items() if k != "auto"}
#: The strip's width without a scale, in pixels.
SWATH_PX_NO_SCALE = 60.0
TRANSMISSIVE_NOTE = ("Chain lines show best in transmitted light: add a backlit frame "
                     "(Capture › Add Backlit Image…) and count on it.")


def grey(a) -> np.ndarray | None:
    """One float band from a picture (RGB averaged), or None."""
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    return a.mean(axis=2) if a.ndim == 3 else a


def pictures(source: str, albedo, height, backlit, view=None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(picture to count on, reflected-light reference or None) for *source*.

    *albedo*, *height*, *backlit*: the solve's albedo, its depth and the lined-up
    backlit frame (or None), all in the same pixels; *view*: the picture Inspect
    shows, unmirrored (any size: the caller scales the section to it).  The
    reference is the albedo, and only for the backlit frame: in reflected light
    the writing is dark, so it can be masked out of the transmitted picture."""
    if source == "view":
        return grey(view), None
    if source == "backlit":
        return grey(backlit), (grey(albedo) if backlit is not None else None)
    if source == "height":
        return grey(height), None
    return grey(albedo), None


def count_along(image, a, b, *, kind: str = "chain", swath_mm: float | None = None,
                mm_per_px: float | None = None, reference=None, sensitivity: float = 1.0) -> LineCount:
    """The lines a section from *a* to *b* (pixels of *image*) crosses.

    The strip is *swath_mm* wide (the kind's usual width when None); without a
    scale it is SWATH_PX_NO_SCALE pixels.  Positions in the result are pixels
    along the section from *a*; spacings are square to the lines."""
    image = np.asarray(image, dtype=np.float64)
    if swath_mm is None:
        swath_mm = suggest_swath_mm(kind)
    width = float(swath_mm) / mm_per_px if mm_per_px else SWATH_PX_NO_SCALE
    width = max(3.0, width)
    dist, offsets, strip = swath_strip(image, a, b, width=width)
    ref = swath_strip(np.asarray(reference, np.float64), a, b, width=width)[2] if reference is not None else None
    found = count_lines_strip(dist, offsets, strip, kind=kind, mm_per_px=mm_per_px,
                              sensitivity=sensitivity, reference=ref)
    # MSI_Reveal asks for the capture's pixels per inch; Helio's scale is set in Measure (the caller says so).
    found.notes = [n for n in found.notes if "pixels per inch" not in n]
    return found


def along(found: LineCount, a, b) -> list[float]:
    """Each line's place along the section, 0 at *a* to 1 at *b*."""
    length = float(np.hypot(b[0] - a[0], b[1] - a[1])) or 1.0
    return [min(1.0, max(0.0, p / length)) for p in found.positions]
