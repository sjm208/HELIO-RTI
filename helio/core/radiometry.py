"""Pixel values to light, GUI-free (FIX-010, KD-052).

Photometric stereo fits I = ρ N·L, which holds for *light*: a pixel value
proportional to the radiance reaching the sensor.  A camera JPEG or TIFF
stores display-encoded values instead (sRGB: roughly the 2.2 root of the
light).  On a perfect synthetic surface, encoded exactly with the sRGB
curve and kept in floating point, decoding before the solve takes the
error from 2.5° to 0.1° (tests/test_radiometry.py), and from 7 October 2026
HELIO decoded by default.

Real captures say otherwise (KD-052).  On the 8-frame manuscript in
``test images/`` and on a copper plate shot with a chrome ball (8-bit
camera files), decoding made every slope about twice as steep: the real
relief, the noise and, worst, the false tilt that uneven lighting puts into
any capture (on the flat plate a median 11° became 22°).  Measured without
regard to steepness (curl over slope, band energies, ink leakage) the two
are alike, so decoding gains no precision here; what it does is double the
errors a raking light and the 3D view show.  A camera's rendering (tone
curve, black level, flare) is not the plain sRGB curve either, so decoded
values are no more proportional to light than stored ones.  HELIO solves on
the stored values by default again, as it always had; for measurement a
height scale set from a reference absorbs the factor on gentle relief.
Decoding stays available for files that really carry a plain sRGB curve at
16 bits; RAW frames are read linearly whatever is chosen
(helio.io.raw_loader).  The false tilt itself is reduced by Flatten
illumination (plate: 11° to 2.4°), for flat objects.

The DiLiGenT benchmark (docs/BENCHMARK_DILIGENT.md, November 2026) measured
the other side: with the lighting calibrated, 8-bit sRGB frames solved as
stored come out 2.3° worse on average and 10.5° worse on a matte ball (14.0°
against 3.5°), and decoding recovers the linear result.  The doubled false
tilt KD-052 saw comes from uneven lighting, which a white-board light field
or flat areas take out.  So the default became "auto": decoded when the
lighting is evened (``resolve_tone``), as stored otherwise.

DiLiGenT-Pi (docs/BENCHMARK_DILIGENT_PI.md, 12 November 2026): 30 near-flat
objects with fine relief, coins and plaques like a punch's face.  As stored,
the direction error is about the same (9.4° against 9.2°) but every normal
leans toward the camera, and the fine detail comes out at 60% of its depth;
decoded, 97%.  A colleague's punches came out shallow, and raising the 3D
Height could not put the detail back against the broad shape.  So "auto" now
decodes camera files whether or not the lighting is evened; the false tilt of
uneven lighting is a broad one, which Flattened (3D relief), Flatten
illumination and a white board take out.

Decoding uses the file's own curve (helio.core.tone_curve): its ICC
profile's, Adobe RGB (1998)'s gamma 2.2 when a camera marks it in the EXIF,
else sRGB.  Most heritage captures are Adobe RGB, whose curve has no linear
toe: decoded as sRGB, a dome's normals came out 0.26° off, by its own curve
0.06° (tests/test_tone_curve.py).

``PHOTO_TONES`` names the choices the processing settings offer.
"""
from __future__ import annotations

import numpy as np

PHOTO_TONES = {
    "auto": "Automatic: decoded to light by the file's curve (RAW read as linear)",
    "stored": "As stored (camera JPEG and TIFF)",
    "srgb": "Decode to linear light by the file's curve (Adobe RGB, sRGB or its ICC profile)",
    "linear": "Linear already (RAW-converted linear TIFF)",
}
DEFAULT_TONE = "auto"
LEGACY_TONE = "stored"          # what a session without the key was solved with


def resolve_tone(tone: str, *, evened: bool = False) -> str:
    """The tone a solve uses: "auto" decodes ("srgb", by the file's own curve; RAW
    frames stay linear); any other choice is kept as it is.  *evened* (a white-board
    light field or flat areas) no longer changes it: until 2026-11-12 "auto" solved
    as stored without it, and fine relief came out at 60% of its depth."""
    if tone == "auto":
        return "srgb"
    return tone


def srgb_to_linear(v, scale: float = 255.0) -> np.ndarray:
    """sRGB-encoded values on 0..*scale* to linear light on the same scale (float32)."""
    x = np.clip(np.asarray(v, np.float32) / scale, 0.0, 1.0)
    out = np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))
    return (out * scale).astype(np.float32)


def linear_to_srgb(v, scale: float = 255.0) -> np.ndarray:
    """Linear light on 0..*scale* to sRGB-encoded values on the same scale (float32)."""
    x = np.clip(np.asarray(v, np.float32) / scale, 0.0, 1.0)
    out = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (out * scale).astype(np.float32)


def subtract_ambient(frame, ambient, *, frame_raw: bool = False, ambient_raw: bool = False,
                     curve=None) -> np.ndarray:
    """*frame* less the light that was there with the lamps off (*ambient*), in linear light.

    Room light, daylight or a modelling lamp reaches every frame alike, so it
    tells the solve nothing about direction and pulls every normal toward the
    camera (the relief comes out shallower).  Light adds in linear units, so both
    are decoded from sRGB (unless RAW, which is linear already), subtracted,
    floored at 0 and encoded back as the frame was: everything after this step
    reads the frame as before.  Same shape and dtype as *frame* (grey or colour)."""
    dec = srgb_to_linear if curve is None else curve.to_linear       # *curve*: the file's own
    enc = linear_to_srgb if curve is None else curve.from_linear     # (helio.core.tone_curve)
    f = np.asarray(frame)
    lf = f.astype(np.float32) if frame_raw else dec(f)
    la = np.asarray(ambient, np.float32) if ambient_raw else dec(ambient)
    if la.ndim == 2 and lf.ndim == 3:
        la = la[..., None]
    out = np.clip(lf - la, 0.0, 255.0)
    if not frame_raw:
        out = enc(out)
    if np.issubdtype(f.dtype, np.integer):
        return np.round(out).astype(f.dtype)
    return out.astype(f.dtype)


def solve_intensities(grey_stack, tone: str, raw_frames=None, curve=None) -> np.ndarray:
    """The stack the normal solve fits: linear light, 0-255 scale.

    *grey_stack* is (n, h, w) as loaded (encoded for photos, linear for RAW
    frames, see helio.io.image_loader).  *raw_frames* marks the RAW frames,
    which are linear whatever *tone* says.  "srgb" decodes by *curve*
    (helio.core.tone_curve: the file's own, Adobe RGB's gamma 2.2 or sRGB),
    sRGB when None.  Returns *grey_stack* itself when nothing needs decoding."""
    if tone != "srgb":
        return grey_stack
    raw = np.zeros(len(grey_stack), bool) if raw_frames is None else np.asarray(raw_frames, bool)
    if raw.all():
        return grey_stack
    out = np.empty(np.shape(grey_stack), np.float32)
    for i in range(len(grey_stack)):
        out[i] = grey_stack[i] if raw[i] else (
            srgb_to_linear(grey_stack[i]) if curve is None else curve.to_linear(grey_stack[i]))
    return out
