"""Camera raw files, decoded as a measurement rather than as a photograph.

Photometric stereo solves for a surface from how its brightness changes
between frames, so the only thing that may differ between them is the light.
A normal raw converter is built to do the opposite: it makes each picture look
good on its own. Two of its habits are fatal here, and both are off by
default in this module.

**Per-frame auto-brightness.** LibRaw brightens each file to a pleasing
exposure. Applied across a stack that is *deliberately* lit from different
angles, it flattens exactly the signal being measured - the dim frames come up
and the bright ones come down - and the solve sees a surface that is not
there. `no_auto_bright=True` is not a preference; without it the numbers are
wrong.

**Display gamma.** A ~2.2 curve makes mid-tones look right on a monitor and
makes intensities non-linear, so a pixel twice as bright no longer reads as
twice the value. Lambert's law is linear, so the fit is being asked to
describe a curve that is not physics. `gamma=(1, 1)` keeps the sensor's own
linear response.

The white balance is taken from the camera, which is a constant for a stack
shot in one session and so cannot introduce a per-frame difference.
"""
from __future__ import annotations

import logging
import os

import numpy as np

_log = logging.getLogger(__name__)

#: Extensions LibRaw handles. Kept explicit rather than "try it and see", so
#: a corrupt JPEG is never mistaken for a raw file and decoded into noise.
RAW_EXTS = frozenset({
    ".cr2", ".cr3", ".crw",           # Canon
    ".nef", ".nrw",                   # Nikon
    ".arw", ".srf", ".sr2",           # Sony
    ".dng",                           # Adobe / Leica / Pentax / phones
    ".raf",                           # Fujifilm
    ".orf",                           # Olympus
    ".rw2",                           # Panasonic
    ".pef", ".dng",                   # Pentax
    ".mrw",                           # Minolta
    ".x3f",                           # Sigma
    ".3fr", ".fff",                   # Hasselblad
    ".iiq",                           # Phase One
    ".erf",                           # Epson
    ".mos", ".mef",                   # Leaf / Mamiya
    ".dcr", ".kdc",                   # Kodak
    ".rwl",                           # Leica
})


def is_raw(path: str) -> bool:
    return os.path.splitext(str(path))[1].lower() in RAW_EXTS


def available() -> bool:
    """Is a raw decoder present? Raw support is optional by design."""
    try:
        import rawpy  # noqa: F401
        return True
    except Exception:
        return False


def read_raw_linear(path: str) -> np.ndarray:
    """Decode a raw file to linear 16-bit RGB.

    Returns ``(h, w, 3)`` uint16 in linear light. Raises if rawpy is absent or
    the file cannot be decoded - callers decide whether that is fatal.
    """
    import rawpy

    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(
            gamma=(1, 1),                       # linear: Lambert's law is linear
            no_auto_bright=True,                # never per-frame exposure
            output_bps=16,                      # keep the sensor's precision
            use_camera_wb=True,                 # constant across a session
            output_color=rawpy.ColorSpace.sRGB,
        )


def raw_size(path: str) -> tuple[int, int]:
    """(width, height) of the decoded frame, turned as it will be, without decoding it."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        s = raw.sizes
        return (s.iheight, s.iwidth) if s.flip in (5, 6) else (s.iwidth, s.iheight)


def embedded_preview(path: str) -> bytes | None:
    """The JPEG the camera put in the file, or None: for a thumbnail, no decoding."""
    import rawpy

    try:
        with rawpy.imread(str(path)) as raw:
            t = raw.extract_thumb()
        return bytes(t.data) if t.format == rawpy.ThumbFormat.JPEG else None
    except Exception:
        return None


def read_raw_sized(path: str, half: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    """Linear 16-bit RGB, decoded as read_raw_linear, and the full frame's (width,
    height).  *half*: at half size without demosaicing, for searching a frame."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        s = raw.sizes
        full = (s.iheight, s.iwidth) if s.flip in (5, 6) else (s.iwidth, s.iheight)
        return raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, use_camera_wb=True,
                               output_color=rawpy.ColorSpace.sRGB, half_size=half), full


def linear_to_srgb8(rgb16: np.ndarray) -> np.ndarray:
    """Linear 16-bit to display-encoded 8-bit, for the parts that are looked at.

    The solve keeps the linear data; only the albedo the eye sees needs the
    transfer curve, or every raw capture would look black.
    """
    x = np.clip(np.asarray(rgb16, np.float32) / 65535.0, 0.0, 1.0)
    out = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
