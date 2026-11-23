"""The curve a photograph's numbers were encoded with, to decode them to light.

Photometric stereo fits light, so Process › Photo tone "decode" undoes the
encoding of a JPEG or TIFF before the solve (helio.core.radiometry).  Which
encoding depends on the file:

* **ICC profile in the file**: its red tone curve (``rTRC``, or ``kTRC``
  for grey): a single gamma (Adobe RGB (1998) is 563/256 = 2.199), a
  parametric curve (sRGB's), or a table.  A gamma of 1 (a linear profile)
  decodes to itself.
* **No profile**: the EXIF colour space.  A camera set to Adobe RGB writes
  ColorSpace "uncalibrated" (65535) with interoperability index ``R03``
  (the DCF rule) and usually embeds no profile.  Anything else is sRGB.

Colour conversion (helio.io.colour_space) is separate: it brings the colour
to sRGB for display.  This only says how the grey the solve fits relates to
light.  GUI-free.
"""
from __future__ import annotations

import functools
import struct
from dataclasses import dataclass

import numpy as np

ADOBE_GAMMA = 563.0 / 256.0            # Adobe RGB (1998) specification, 2.19921875


@dataclass(frozen=True)
class Curve:
    """A decoding curve: *table* maps evenly spaced code values 0..1 to light 0..1."""
    name: str
    table: tuple                       # floats, light at np.linspace(0, 1, len(table))

    def to_linear(self, v, scale: float = 255.0) -> np.ndarray:
        x = np.clip(np.asarray(v, np.float32) / scale, 0.0, 1.0)
        t = np.asarray(self.table, np.float32)
        return (np.interp(x, np.linspace(0.0, 1.0, len(t), dtype=np.float32), t) * scale
                ).astype(np.float32)

    def from_linear(self, v, scale: float = 255.0) -> np.ndarray:
        """The inverse: light back to code values (the table is monotonic)."""
        y = np.clip(np.asarray(v, np.float32) / scale, 0.0, 1.0)
        t = np.maximum.accumulate(np.asarray(self.table, np.float32))
        return (np.interp(y, t, np.linspace(0.0, 1.0, len(t), dtype=np.float32)) * scale
                ).astype(np.float32)


def _table(fn, n: int = 4096) -> tuple:
    x = np.linspace(0.0, 1.0, n)
    return tuple(float(v) for v in np.clip(fn(x), 0.0, 1.0))


def _srgb(x):
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


SRGB = Curve("sRGB", _table(_srgb))


def gamma_curve(g: float, name: str = "") -> Curve:
    return Curve(name or f"gamma {g:.2f}", _table(lambda x: np.power(x, g)))


ADOBE_RGB = gamma_curve(ADOBE_GAMMA, "Adobe RGB (1998), gamma 2.2")


def _para(kind: int, p: list[float]):
    """ICC parametricCurveType, functions 0-4."""
    g = p[0]

    def fn(x):
        if kind == 0:
            return np.power(x, g)
        a, b = p[1], p[2]
        lin = np.power(np.maximum(a * x + b, 0.0), g)
        if kind == 1:
            return np.where(x >= -b / a, lin, 0.0)
        if kind == 2:
            return np.where(x >= -b / a, lin + p[3], p[3])
        if kind == 3:
            return np.where(x >= p[4], lin, p[3] * x)
        return np.where(x >= p[4], lin + p[5], p[3] * x + p[6])
    return fn


def curve_from_icc(icc: bytes, name: str = "") -> Curve | None:
    """The profile's tone curve (rTRC, else kTRC), or None when it has none we can read."""
    try:
        n = struct.unpack(">I", icc[128:132])[0]
        tags = {}
        for k in range(n):
            sig, off, size = struct.unpack(">4sII", icc[132 + 12 * k:144 + 12 * k])
            tags[sig] = (off, size)
        for sig in (b"rTRC", b"kTRC", b"gTRC"):
            if sig in tags:
                off, size = tags[sig]
                break
        else:
            return None
        typ = icc[off:off + 4]
        if typ == b"curv":
            cnt = struct.unpack(">I", icc[off + 8:off + 12])[0]
            if cnt == 0:
                return gamma_curve(1.0, name or "linear")
            if cnt == 1:
                g = struct.unpack(">H", icc[off + 12:off + 14])[0] / 256.0
                return gamma_curve(g, name or f"gamma {g:.2f}")
            vals = np.frombuffer(icc[off + 12:off + 12 + 2 * cnt], ">u2").astype(np.float64) / 65535.0
            return Curve(name or "profile table", tuple(float(v) for v in vals))
        if typ == b"para":
            kind = struct.unpack(">H", icc[off + 8:off + 10])[0]
            npar = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(kind)
            if npar is None:
                return None
            p = [v / 65536.0 for v in struct.unpack(f">{npar}i", icc[off + 12:off + 12 + 4 * npar])]
            return Curve(name or "profile curve", _table(_para(kind, p)))
    except Exception:                                      # noqa: BLE001 - a broken profile: fall back
        return None
    return None


def _exif_adobe(path: str) -> bool:
    """A camera JPEG set to Adobe RGB: ColorSpace uncalibrated, interoperability R03."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            ex = im.getexif()
            cs = ex.get_ifd(0x8769).get(0xA001)
            interop = ex.get_ifd(0xA005).get(1)
        return cs == 65535 and str(interop).strip().upper() == "R03"
    except Exception:                                      # noqa: BLE001
        return False


@functools.lru_cache(maxsize=32)
def file_curve(path: str) -> Curve:
    """How *path*'s numbers were encoded (see the module notes); sRGB when nothing says otherwise."""
    from helio.io.colour_space import embedded_profile, profile_name
    icc = embedded_profile(path)
    if icc:
        c = curve_from_icc(icc, profile_name(path))
        if c is not None:
            return c
    if _exif_adobe(path):
        return ADOBE_RGB
    return SRGB


def capture_curve(paths) -> Curve:
    """The curve of a capture: its first photograph that is not RAW (all are shot alike)."""
    from helio.io.raw_loader import is_raw
    for p in paths or ():
        if p and not is_raw(p):
            return file_curve(p)
    return SRGB
