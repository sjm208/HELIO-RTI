"""Each light's strength and fall-off across the frame, from a white board (CAL-005).

Photometric stereo fits  I = rho * s * (n . L)  with every light of the same
strength s, the same everywhere in the frame.  A real lamp is neither: the
four heads of a fixed rig differ by some percent, and each lights the side
of the frame nearer to it more (inverse square, and the reflector's beam).
The solve reads both as the surface turning towards the brighter light:
false tilt, and on a flat object a false curve.

A flat, matte, evenly white board filling the frame, photographed once per
light from the rig's fixed positions, measures exactly that.  Lambertian and
facing the camera (n = +Z), the board's brightness under light i at x is

    B_i(x) = rho_b * E_i(x) * L_i,z

so the light field  E_i(x) = B_i(x) / L_i,z  (smoothed: the board's own grain
is not the light).  Dividing every frame of a capture by its light's E_i,
all normalised by one common level, evens both the fall-off and the
strengths at once, with the directions left as measured on the ball.

This is the classic flat-field calibration of a fixed photometric-stereo rig.
It replaces the near-field model (the board has measured what that model
assumes), the grey card and the semi-calibrated strengths; use it with the
plain distant-light solve.  It holds only while nothing moves: lamps, camera,
lens, focus, aperture, flash power.

Stored small (``GRID`` on the long side): the light changes slowly across the
frame.  GUI-free.
"""
from __future__ import annotations

import datetime
import os

import cv2
import numpy as np

#: The stored maps' long side, in cells.
GRID = 96
#: A light this low tells nothing reliable about a board facing the camera.
MIN_LZ = 0.05


def from_boards(boards, lights) -> dict:
    """The light field from one board frame per light.

    *boards* are (h, w) arrays in the solve's intensity units (linear for RAW,
    as the solve reads them), all one size; *lights* (n, 3), +Y up, in the
    same order.  Raises ValueError for a board that is clipped, dark, or
    for a light too low to use."""
    L = np.asarray(lights, np.float64)
    if len(boards) != len(L):
        raise ValueError(f"{len(boards)} board frames for {len(L)} lights: one board frame per "
                         f"light, in the capture's order.")
    if len(L) < 3:
        raise ValueError("At least three lights.")
    h, w = np.shape(boards[0])[:2]
    gw, gh = (GRID, max(2, round(GRID * h / w))) if w >= h else (max(2, round(GRID * w / h)), GRID)
    top = 255.0 if np.asarray(boards[0]).dtype == np.uint8 else float(
        max(np.percentile(np.asarray(b, np.float32)[::8, ::8], 99.9) for b in boards))
    maps, clipped = [], []
    for i, (b, l) in enumerate(zip(boards, L)):
        b = np.asarray(b, np.float32)
        if b.shape[:2] != (h, w):
            raise ValueError(f"Board frame {i + 1} is {b.shape[1]} x {b.shape[0]}, the first "
                             f"{w} x {h}.")
        if b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        lz = float(l[2] / max(np.linalg.norm(l), 1e-12))
        if lz < MIN_LZ:
            raise ValueError(f"Light {i + 1} is too low ({np.degrees(np.arcsin(lz)):.0f}° up) to "
                             f"measure on a board.")
        if np.asarray(boards[i]).dtype == np.uint8:
            clipped.append(float(np.mean(np.asarray(boards[i]) >= 250)))
        # Cell averages (exact for any smooth fall-off, up to the frame's edge), then a 3 x 3
        # median on the grid: a speck of dust or a fibre on the board is not the light.
        m = cv2.resize(b, (gw, gh), interpolation=cv2.INTER_AREA)
        m = cv2.medianBlur(m.astype(np.float32), 3) / lz
        if float(np.median(m)) <= 1e-6:
            raise ValueError(f"Board frame {i + 1} is black.")
        maps.append(m.astype(np.float32))
    if clipped and max(clipped) > 0.01:
        k = int(np.argmax(clipped))
        raise ValueError(f"Board frame {k + 1} is clipped ({clipped[k] * 100:.0f}% of it pure white): "
                         f"lower the flash power or the exposure and shoot the board again.")
    M = np.stack(maps)
    level = float(np.median(M))
    M /= level
    strengths = np.median(M.reshape(len(M), -1), axis=1)
    fall = [(float(m.max() / max(m.min(), 1e-6)) - 1.0) for m in M]
    return dict(maps=M, frame_size=(int(w), int(h)), lights=L.round(6).tolist(),
                strengths=strengths.round(4).tolist(),
                strength_spread=float(strengths.max() / max(strengths.min(), 1e-6)),
                falloff=float(max(fall)), top=top,
                created=datetime.datetime.now().isoformat(timespec="seconds"))


def save(path: str, field: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, maps=field["maps"], frame_size=np.asarray(field["frame_size"]),
                        lights=np.asarray(field["lights"]), strengths=np.asarray(field["strengths"]),
                        strength_spread=field["strength_spread"], falloff=field["falloff"],
                        created=str(field["created"]))
    return path


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return dict(maps=z["maps"].astype(np.float32), frame_size=tuple(int(v) for v in z["frame_size"]),
                    lights=z["lights"].tolist(), strengths=z["strengths"].tolist(),
                    strength_spread=float(z["strength_spread"]), falloff=float(z["falloff"]),
                    created=str(z["created"]))


def describe(field: dict) -> str:
    return (f"{len(field['maps'])} lights · strongest {field['strength_spread']:.2f}× the weakest · "
            f"fall-off up to {field['falloff'] * 100:.0f}% across the frame")


def factor(field: dict, i: int, base_size, crop=None) -> np.ndarray:
    """1 / E_i over the solve's frame: (h, w) float32 to multiply frame *i* by.

    *base_size* (w, h) is the photographs' size as the solve holds them (the
    working size); *crop* (x1, y1, x2, y2) in those pixels, or None.  The field
    is stretched over the whole photograph, so any working size or region
    lines up."""
    bw, bh = (int(v) for v in base_size)
    fw, fh = field["frame_size"]
    if abs(bw / bh - fw / fh) > 0.01:
        raise ValueError(f"The light field was measured on {fw} x {fh} frames; these are "
                         f"{bw} x {bh}.")
    E = cv2.resize(np.asarray(field["maps"][i], np.float32), (bw, bh), interpolation=cv2.INTER_CUBIC)
    if crop is not None:
        x1, y1, x2, y2 = (int(v) for v in crop)
        E = E[y1:y2, x1:x2]
    return (1.0 / np.maximum(E, 1e-3)).astype(np.float32)


def check_lights(field: dict, lights, tol_deg: float = 3.0) -> str:
    """'' when *lights* are the ones the board was shot with, else why not."""
    a = np.asarray(field["lights"], np.float64)
    b = np.asarray(lights, np.float64)
    if a.shape != b.shape:
        return f"The light field has {len(a)} lights; this capture {len(b)}."
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))
    if float(ang.max()) > tol_deg:
        k = int(np.argmax(ang))
        return (f"Light {k + 1} is {ang[k]:.1f}° from where it was when the board was shot: "
                f"has the rig moved, or the frames' order changed?")
    return ""
