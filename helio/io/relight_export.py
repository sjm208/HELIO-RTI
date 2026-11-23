"""Write an RTI package in the CNR-ISTI relight / OpenLIME web format.

Relight (RelightLab, relight.js, OpenLIME) is the tool the field actually uses
for browser-based RTI now that CHI's RTIViewer is legacy. Its web format is a
directory holding info.json plus coefficient planes as JPEGs.

HELIO fits six *luminance* PTM coefficients plus an RGB albedo, which is
exactly relight's "lrgb" colorspace at 9 planes: three RGB planes followed by
the six polynomial terms, packed three-to-a-JPEG.

The polynomial convention matches on both sides:

    L(u, v) = c0*u^2 + c1*v^2 + c2*u*v + c3*u + c4*v + c5

Coefficients are stored as uint8 and recovered by the viewer as

    value = (stored/255 - bias) * scale

so each plane carries its own scale and bias in info.json.
"""
from __future__ import annotations

import json
import logging
import os

import cv2
import numpy as np

_log = logging.getLogger(__name__)

#: relight rejects any extension other than .jpg for source images - verified
#: against relight-cli 2026.01, which reports "Number of lights in dome needs
#: to be equal to the number of images" for a folder of .jpeg files.
RELIGHT_IMAGE_EXT = ".jpg"


def _pack_plane(values: list[np.ndarray]) -> tuple[np.ndarray, list, list]:
    """Pack up to 3 coefficient maps into one BGR uint8 image + scale/bias."""
    scales, biases, chans = [], [], []
    for v in values:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        scale = hi - lo
        if not np.isfinite(scale) or scale <= 1e-12:
            # A constant plane has no range to normalise against. Storing
            # zeros and scale=1/bias=0 would decode back to 0 and silently
            # throw the value away, so carry it in the bias instead:
            # (0 - (-lo)) * 1 == lo.
            scale, bias = 1.0, -lo if np.isfinite(lo) else 0.0
            stored = np.zeros_like(v, dtype=np.float32)
        else:
            bias = -lo / scale
            stored = v / scale + bias
        scales.append(round(float(scale), 6))
        biases.append(round(float(bias), 6))
        chans.append(np.clip(stored, 0.0, 1.0) * 255.0)
    while len(chans) < 3:
        chans.append(np.zeros_like(chans[0]))
        scales.append(1.0)
        biases.append(0.0)
    rgb = np.stack(chans, axis=-1).astype(np.uint8)
    return rgb, scales, biases


def export_relight_package(
    out_dir: str,
    ptm_coeffs: np.ndarray,
    albedo_rgb: np.ndarray,
    lights: np.ndarray | None,
    h: int,
    w: int,
    quality: int = 95,
) -> str:
    """Write info.json + plane_*.jpg. Returns the output directory.

    ptm_coeffs : (6, h*w) or (6, h, w) luminance coefficients
    albedo_rgb : (h, w, 3) uint8 or float
    lights     : (n, 3) unit light vectors, or None
    """
    os.makedirs(out_dir, exist_ok=True)

    c = np.asarray(ptm_coeffs, dtype=np.float32)
    if c.ndim == 2:
        c = c.reshape(6, h, w)
    if c.shape[0] != 6:
        raise ValueError(f"expected 6 PTM coefficients, got {c.shape[0]}")

    alb = np.asarray(albedo_rgb, dtype=np.float32)
    if alb.max() > 1.5:
        alb = alb / 255.0
    if alb.shape[:2] != (h, w):
        alb = cv2.resize(alb, (w, h))

    planes, all_scale, all_bias = [], [], []

    # Plane 0: RGB albedo, stored directly (scale 1, bias 0) - matches what
    # relight-cli itself writes for the first three lrgb planes.
    rgb = np.clip(alb, 0.0, 1.0) * 255.0
    planes.append(rgb.astype(np.uint8))
    all_scale += [1.0, 1.0, 1.0]
    all_bias += [0.0, 0.0, 0.0]

    # Planes 1..2: the six polynomial terms, three per image.
    for group in (c[0:3], c[3:6]):
        img, sc, bi = _pack_plane([group[i] for i in range(group.shape[0])])
        planes.append(img)
        all_scale += sc
        all_bias += bi

    for i, img in enumerate(planes):
        path = os.path.join(out_dir, f"plane_{i}{RELIGHT_IMAGE_EXT}")
        # cv2 writes BGR; our channel order is already the intended one, so
        # flip to keep R in the red channel of the file.
        # No chroma subsampling, as relight-cli: each channel is a different
        # coefficient, and OpenCV's default 4:2:0 blurs two of them together.
        cv2.imwrite(path, img[..., ::-1],
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
                     int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                     int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)])

    flat_lights = []
    if lights is not None:
        for v in np.asarray(lights, dtype=np.float32):
            flat_lights += [round(float(x), 3) for x in v[:3]]

    info = {
        "width": int(w),
        "height": int(h),
        "format": RELIGHT_IMAGE_EXT.lstrip("."),
        "type": "ptm",
        "colorspace": "lrgb",
        "lights": flat_lights,
        "nplanes": 9,
        "quality": int(quality),
        "materials": [{"scale": all_scale, "bias": all_bias}],
    }
    with open(os.path.join(out_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    _log.info("relight package written to %s (%d planes)", out_dir, len(planes))
    return out_dir


def decode_plane(path: str, scale: list, bias: list, idx0: int):
    """Recover the coefficient maps stored in one plane (for verification)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise OSError(f"could not read {path}")
    img = img[..., ::-1].astype(np.float32) / 255.0
    return [(img[..., k] - bias[idx0 + k]) * scale[idx0 + k] for k in range(3)]
