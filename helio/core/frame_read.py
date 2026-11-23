"""A frame from any file, for the readers that are not the solve's loader.

The drift check, the ball search, the flash check and the tools read frames
with cv2.imread, which knows no RAW: it finds nothing in a Sony or Phase One
file, and only the 256 px thumbnail in a DNG.  read_frame takes the same flags
and gives a RAW frame at the size the loader would: display-encoded for looking
for things in it, or linear (*linear*) for values read as the solve reads them.
Anything else is cv2.imread.
"""
from __future__ import annotations

import cv2
import numpy as np


def read_frame(path: str, flag: int = cv2.IMREAD_COLOR, *, linear: bool = False, sharp: bool = False):
    """cv2.imread(path, flag), RAW too.  A RAW frame is decoded at half size and
    scaled to the size asked for, several times faster and all a search needs;
    *sharp* decodes it whole.  *linear* (grey): float32 0-255, light."""
    from helio.io import raw_loader
    if not raw_loader.is_raw(path):
        return cv2.imread(path, flag)
    k = {cv2.IMREAD_REDUCED_GRAYSCALE_2: 2, cv2.IMREAD_REDUCED_COLOR_2: 2,
         cv2.IMREAD_REDUCED_GRAYSCALE_4: 4, cv2.IMREAD_REDUCED_COLOR_4: 4,
         cv2.IMREAD_REDUCED_GRAYSCALE_8: 8, cv2.IMREAD_REDUCED_COLOR_8: 8}.get(flag, 1)
    grey = flag in (cv2.IMREAD_GRAYSCALE, cv2.IMREAD_REDUCED_GRAYSCALE_2, cv2.IMREAD_REDUCED_GRAYSCALE_4,
                    cv2.IMREAD_REDUCED_GRAYSCALE_8)
    try:
        rgb16, (w, h) = raw_loader.read_raw_sized(path, half=not sharp)
    except Exception:
        return None
    size = ((w + k - 1) // k, (h + k - 1) // k)
    if rgb16.shape[1::-1] != size:
        rgb16 = cv2.resize(rgb16, size, interpolation=cv2.INTER_AREA if k > 2 else cv2.INTER_LINEAR)
    if grey and linear:
        return rgb16.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32) / 257.0
    rgb8 = raw_loader.linear_to_srgb8(rgb16)
    return cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY if grey else cv2.COLOR_RGB2BGR)
