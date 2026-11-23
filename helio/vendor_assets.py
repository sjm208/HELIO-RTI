"""Locate the vendored JavaScript that exported viewers inline.

Why these are vendored at all
-----------------------------
Three exports used to pull their libraries from a CDN: the 3D Web Viewer and
the HF-Normals viewer loaded three.js r128 from jsdelivr, and DeepZoom loaded
OpenSeadragon from cdnjs. That is fine for a page you post today and forget,
and wrong for everything this program is actually for. Those exports feed the
Heritage Archive and Zenodo deposits — packages whose entire promise is that
they still open in twenty years. A deposit that goes blank when a CDN retires
a 2021 build of three.js is a preservation failure, and it fails silently and
late, which is the worst way to fail.

The libraries are ~1 MB together, against exports that embed base64 meshes and
whole tile pyramids. They are well under 1% of what they are there to display.

Lookup follows the same candidate-path pattern as ``helio_help``'s
``help_images``: source tree first, then the places py2app puts data files.

Missing assets raise. The help_images comment in setup.py records what silence
costs — a bundle that shipped a help page of blank gaps, with nothing erroring,
unnoticed until a user mentioned it. A viewer with no three.js is a white
rectangle, so this says so at export time instead.
"""

from __future__ import annotations

import os
import sys

__all__ = ["vendor_dir", "read_text", "read_bytes", "osd_images_dir",
           "VendorAssetMissing"]

_HERE = os.path.dirname(os.path.abspath(__file__))


class VendorAssetMissing(FileNotFoundError):
    """A vendored library is not in the build."""


def _candidates() -> list:
    return [
        os.path.join(_HERE, "vendor"),
        os.path.join(os.path.dirname(_HERE), "vendor"),
        os.path.join(getattr(sys, "_MEIPASS", _HERE), "vendor"),
        os.path.join(os.path.dirname(os.path.dirname(_HERE)), "vendor"),
        # py2app drops DATA_FILES entries into Contents/Resources.
        os.path.join(os.path.dirname(sys.executable), "..", "Resources", "vendor"),
    ]


def vendor_dir() -> str:
    """The directory holding the vendored libraries."""
    for c in _candidates():
        if os.path.isdir(c):
            return os.path.normpath(c)
    raise VendorAssetMissing(
        "vendored JavaScript not found. The exported viewers inline three.js "
        "and OpenSeadragon so they work offline and inside archival deposits; "
        "without them an export would be a blank page. Looked in:\n  "
        + "\n  ".join(os.path.normpath(c) for c in _candidates()))


def read_text(name: str) -> str:
    """One vendored text asset, e.g. ``three.min.js``."""
    p = os.path.join(vendor_dir(), name)
    if not os.path.isfile(p):
        raise VendorAssetMissing(f"{name} is missing from {vendor_dir()}")
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def read_bytes(name: str) -> bytes:
    p = os.path.join(vendor_dir(), name)
    if not os.path.isfile(p):
        raise VendorAssetMissing(f"{name} is missing from {vendor_dir()}")
    with open(p, "rb") as f:
        return f.read()


def osd_images_dir() -> str:
    """OpenSeadragon's navigation button images.

    DeepZoom writes a folder rather than one file, so these travel beside the
    tiles as real files and the page keeps its default ``prefixUrl`` shape —
    simpler than 27 data URIs, and it keeps the HTML readable.
    """
    p = os.path.join(vendor_dir(), "osd_images")
    if not os.path.isdir(p):
        raise VendorAssetMissing(f"osd_images is missing from {vendor_dir()}")
    return p


def three_js_bundle() -> str:
    """three.js plus the two example modules the viewers use, concatenated.

    Order matters: OrbitControls and GLTFLoader both attach themselves to the
    THREE global, so the core has to be evaluated first.
    """
    return "\n".join([
        read_text("three.min.js"),
        read_text("OrbitControls.js"),
        read_text("GLTFLoader.js"),
    ])
