"""What colour space a photograph's numbers are in, and bringing them to sRGB.

Everything Helio shows and exports as colour is sRGB, but a heritage workflow's
TIFFs are often tagged Adobe RGB, ProPhoto or eciRGB.  Read as if they were
sRGB, their colours come out pale and shifted: on a charter's 8-bit Adobe RGB
frames (a low-chroma, parchment and wax subject) about 2.2 ΔE₀₀ on average and
6 levels of red, which is what a careful eye sees side by side.  So the colour
is converted by the file's own ICC profile as it is read.

Only the colour.  The grey the solve fits stays the file's stored numbers
(helio.core.radiometry): a colour conversion is not a measurement of light.
GUI-free.
"""
from __future__ import annotations

import functools
import io
import logging

import numpy as np

_log = logging.getLogger(__name__)


def embedded_profile(path: str) -> bytes | None:
    """The ICC profile in the file at *path*, or None.  Reads the header only."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None           # a heritage capture is rightly huge
        with Image.open(path) as im:
            return im.info.get("icc_profile") or None
    except Exception:
        return None


@functools.lru_cache(maxsize=8)
def _cms(icc: bytes):
    """(name, transform to sRGB or None) for this profile.  None: nothing to do
    (already sRGB, or not an RGB profile)."""
    from PIL import ImageCms
    src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
    name = (ImageCms.getProfileDescription(src) or "").strip() or "unnamed profile"
    if src.profile.xcolor_space.strip() != "RGB" or "srgb" in name.lower():
        return name, None
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    return name, ImageCms.buildTransform(src, srgb, "RGB", "RGB",
                                         renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC)


def profile_name(path: str) -> str:
    """'Adobe RGB (1998)', '' when the file has no profile."""
    icc = embedded_profile(path)
    if not icc:
        return ""
    try:
        return _cms(icc)[0]
    except Exception:
        return ""


def to_srgb(rgb8: np.ndarray, path: str) -> np.ndarray:
    """*rgb8* (uint8 RGB, read from *path*) in sRGB, by the file's own profile.
    Unchanged when it has none, is sRGB already, or the profile cannot be used."""
    icc = embedded_profile(path)
    if not icc:
        return rgb8
    try:
        x = _cms(icc)[1]
        if x is None:
            return rgb8
        from PIL import Image, ImageCms
        return np.asarray(ImageCms.applyTransform(Image.fromarray(np.ascontiguousarray(rgb8)), x))
    except Exception as ex:
        _log.warning("colour profile of %s not used: %s", path, ex)
        return rgb8


def srgb_profile_bytes() -> bytes:
    """An sRGB ICC profile, to embed in what Helio writes."""
    from PIL import ImageCms
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
