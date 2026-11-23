"""Which files are captures, and in what order (GUI-free; both apps use it).

One definition of the capture file types, moved out of Helio_Pro.py so the
Qt app reads exactly the same set.  It was once spelled out in five places
with three different contents, and a .webp capture could be dropped onto the
window but not batch-processed.
"""
from __future__ import annotations

import os
import re

_DIGITS = re.compile(r"(\d+)")


def natural_key(path: str):
    """Sort key for capture frames in shot order: numbers by value, case ignored.

    Plain sorted() puts "x-10.JPG" before "x-3.JPG", which scrambles a
    capture whose frame numbers are not zero-padded, and with it every light
    paired by position (an RTIBuilder .lp of 01.jpg … 50.jpg against
    x-3.JPG … x-52.JPG).
    """
    name = os.path.basename(str(path)).lower()
    return [(0, int(t), t) if t.isdigit() else (1, 0, t) for t in _DIGITS.split(name)] + [str(path)]


def capture_extensions() -> frozenset[str]:
    """Lower-case extensions HELIO can load; raw formats only when a decoder is present."""
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    try:
        from helio.io.raw_loader import RAW_EXTS, available
        if available():
            exts |= set(RAW_EXTS)
    except Exception:
        pass
    return frozenset(exts)


def is_capture(path: str, exts: frozenset[str] | None = None) -> bool:
    name = os.path.basename(path)
    return (not name.startswith(".")
            and os.path.splitext(name)[1].lower() in (exts or capture_extensions()))


def captures_in(paths, exts: frozenset[str] | None = None) -> list[str]:
    """Expand folders and filter files, as the legacy drop handler does.

    A folder contributes the captures directly inside it, in natural
    (shot) order: x-2 before x-10.
    Files keep the order given.  Duplicates are dropped, first one kept.
    """
    exts = exts or capture_extensions()
    out: list[str] = []
    for item in paths:
        item = str(item).strip()
        if not item:
            continue
        if os.path.isdir(item):
            out.extend(sorted((os.path.join(item, f) for f in os.listdir(item)
                               if is_capture(f, exts)), key=natural_key))
        elif is_capture(item, exts):
            out.append(item)
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]
