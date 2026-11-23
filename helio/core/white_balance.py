"""White balance from a neutral patch: per-channel gains, applied in linear light.

The colour Helio shows and exports is 8-bit sRGB (the median of the frames, or
a RAW capture's solved reflectance).  A patch the user says is neutral - the
white of a scale bar, a grey card - gives three gains that make its mean
linear R, G and B equal, at the patch's own luminance (Rec. 709 Y), so the
picture is neither brightened nor darkened on average.  The gains multiply
linear light, never the encoded numbers: scaling sRGB codes per channel would
bend the hue of everything darker or lighter than the patch.

Only the colour.  The grey the solve reads is untouched: a white balance is a
statement about the light's colour, not a measurement of the surface.
GUI-free.
"""
from __future__ import annotations

import cv2
import numpy as np

from helio.core.radiometry import linear_to_srgb, srgb_to_linear

#: Rec. 709 / sRGB luminance weights.
Y709 = np.array([0.2126, 0.7152, 0.0722])
#: A patch with any channel at or above this code is clipped: its colour is not known.
CLIP = 250
#: Gains outside this range mean the patch was not neutral (or not lit).
GAIN_RANGE = (1 / 3, 3.0)


def neutral(gains) -> bool:
    """True for no white balance (None, or gains all 1)."""
    return gains is None or bool(np.allclose(np.asarray(gains, float), 1.0, atol=1e-4))


def gains_from_patch(rgb8: np.ndarray) -> tuple[float, float, float]:
    """(gR, gG, gB) that make the mean of *rgb8* (uint8 sRGB, any shape (..., 3))
    neutral at its own luminance.  Clipped pixels are left out; ValueError if
    the patch is too dark, all clipped, or so far from neutral that the gains
    would be absurd."""
    px = np.asarray(rgb8).reshape(-1, 3)
    if px.size == 0:
        raise ValueError("The patch is empty.")
    ok = px.max(axis=1) < CLIP
    if ok.sum() < max(1, 0.2 * len(px)):
        raise ValueError("The patch is clipped (pure white): pick a light grey that is not "
                         "burnt out, or a smaller patch inside it.")
    lin = srgb_to_linear(px[ok].astype(np.float64), 255.0) / 255.0      # 0..1 linear
    mean = lin.mean(axis=0, dtype=np.float64)
    if mean.min() < 0.01:
        raise ValueError("The patch is too dark to judge its colour: pick a white or light grey.")
    y = float(Y709 @ mean)
    g = y / mean
    if g.min() < GAIN_RANGE[0] or g.max() > GAIN_RANGE[1]:
        raise ValueError("That patch is strongly coloured, not a neutral grey or white.")
    return tuple(round(float(v), 5) for v in g)


def lut(gains) -> np.ndarray:
    """(256, 1, 3) uint8: code -> balanced code for each channel."""
    codes = srgb_to_linear(np.arange(256, dtype=np.float64), 255.0)
    out = np.empty((256, 1, 3), np.uint8)
    for c, g in enumerate(np.asarray(gains, float)):
        out[:, 0, c] = np.clip(np.round(linear_to_srgb(codes * g, 255.0)), 0, 255)
    return out


def apply(rgb8: np.ndarray | None, gains) -> np.ndarray | None:
    """*rgb8* (uint8 RGB) white-balanced by *gains*; the array itself when neutral."""
    if rgb8 is None or neutral(gains):
        return rgb8
    a = np.asarray(rgb8)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] < 3:
        return rgb8
    table = lut(gains)
    if a.shape[2] == 3:
        return cv2.LUT(np.ascontiguousarray(a), table)
    out = a.copy()
    out[..., :3] = cv2.LUT(np.ascontiguousarray(a[..., :3]), table)
    return out


def describe(gains) -> str:
    """'R ×1.08 · G ×0.97 · B ×1.21', or 'off'."""
    if neutral(gains):
        return "off"
    r, g, b = (float(v) for v in gains)
    return f"R ×{r:.2f} · G ×{g:.2f} · B ×{b:.2f}"
