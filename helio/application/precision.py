"""Precision photometric stereo: calibrating a fixed rig and saying how ready a
capture is to measure with.  GUI-free; the Qt app's Process › Precision
measurement and Measure › Accuracy check use it.

The pieces (each optional, each off unless set):

* the white-board light field (helio.core.light_field): every light's strength
  and fall-off, from one board frame per light;
* the lens correction (helio.core.lens): distortion from chessboard photographs;
* heights in millimetres from the lateral scale: with the light field on a
  calibrated, linear capture, the normals' slopes are true, so a pixel of
  height is a pixel of width;
* the accuracy check on references (helio.core.metrology.step_height,
  flatness): the error of this rig, measured, not assumed.
"""
from __future__ import annotations

import os

import numpy as np

#: The board is read at this fraction of full size: the field is smoothed far more.
BOARD_SCALE = 0.25


def board_field(paths, lights, photo_tone: str = "stored", progress=None) -> dict:
    """The light field from white-board frames (one per light, in the capture's
    order), read exactly as the solve reads frames (RAW linear, the same tone)."""
    from helio.application.pipeline import _load_scaled
    from helio.core import light_field as LF
    from helio.core.radiometry import solve_intensities
    frames, size = _load_scaled(list(paths), progress or (lambda *a, **k: None), BOARD_SCALE)
    bad = [os.path.basename(p) for p, f in zip(paths, frames) if f[1] is None]
    if bad:
        raise ValueError(f"Could not read {', '.join(bad)}.")
    greys = np.stack([np.asarray(f[1], np.float32) for f in frames])
    raw = [f[2] for f in frames]
    from helio.core.tone_curve import capture_curve
    lin = solve_intensities(greys, photo_tone, raw, capture_curve(paths))
    if np.asarray(frames[0][1]).dtype == np.uint8:
        top = [float(np.mean(np.asarray(f[1]) >= 250)) for f in frames]
        if max(top) > 0.01:
            k = int(np.argmax(top))
            raise ValueError(f"{os.path.basename(paths[k])} is clipped ({top[k] * 100:.0f}% pure white): "
                             f"lower the flash power or the exposure and shoot the board again.")
    field = LF.from_boards(list(lin), lights)
    field["frame_size"] = tuple(int(v) for v in size)        # the photographs' own size
    return field


def lens_from_photos(paths, pattern=(9, 6)) -> dict:
    """The lens calibration from chessboard photographs."""
    import cv2
    from helio.core import lens
    from helio.core.frame_read import read_frame
    greys = []
    for p in paths:
        g = read_frame(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise ValueError(f"Could not read {os.path.basename(p)}.")
        if g.dtype != np.uint8:
            g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        greys.append(g)
    return lens.calibrate(greys, pattern)


def parse_pattern(text: str) -> tuple[int, int]:
    """'9 x 6', '9x6', '9,6' -> (9, 6)."""
    import re
    nums = [int(v) for v in re.findall(r"\d+", str(text))]
    if len(nums) != 2 or min(nums) < 2:
        raise ValueError("Give the inner corners as columns x rows, for example 9 x 6.")
    return nums[0], nums[1]


def readiness(*, n_lights: int, conditioning: float, raw: bool, photo_tone: str,
              light_field: str, lens_calibration: str, near_field: bool, scale: bool,
              z_from_scale: bool, accuracy: dict | None) -> list[tuple[str, bool, str]]:
    """The checklist for measuring: [(item, met, what to do)].  Advice, not rules:
    any capture still solves."""
    out = [
        ("Four or more lights", n_lights >= 4,
         "Four lights leave one spare, so a pixel in shadow or glare in one frame can be "
         "judged; three only just pin a normal down."),
        ("Lights well spread (conditioning under 3)", conditioning < 3.0,
         "Set the lights first (the ball, or a rig profile)." if not np.isfinite(conditioning) else
         f"Conditioning is {conditioning:.1f}: spread the lights round the object and vary their "
         f"height."),
        ("Linear frames (RAW, or linear TIFF)", raw or photo_tone == "linear",
         "Shoot RAW: a camera JPEG's tone curve is not proportional to light, and the slopes "
         "come out scaled."),
        ("White-board light field", bool(light_field),
         "Shoot a matte white board filling the frame under each light, and calibrate."),

    ]
    if light_field:
        out.append(("Near-field model off (the board replaces it)", not near_field,
                    "Turn Near-field off: the board has measured the fall-off already."))
    out += [
        ("Lens corrected", bool(lens_calibration),
         "Photograph a chessboard 10-20 times across the frame, and calibrate the lens."),
        ("Scale set", scale, "Set the scale in Measure, on a ruler in the shot."),
        ("Heights from the lateral scale", z_from_scale,
         "Measure › Height scale › Use the lateral scale, once the capture is calibrated."),
        ("Accuracy checked on a reference", bool(accuracy),
         "Measure › Accuracy check on a gauge block or a flat plate shot on the same rig."),
    ]
    return out


def readiness_text(items, details: bool = False) -> str:
    """'n of m ready' and a line per item; with *details*, what to do for each one not met."""
    met = sum(1 for _n, ok, _d in items if ok)
    lines = [f"{met} of {len(items)} ready for measurement"]
    for name, ok, todo in items:
        lines.append(("✓ " if ok else "○ ") + name + ("" if ok or not details else f" — {todo}"))
    return "\n".join(lines)


# Process › Set Up for Measurement: the solve's choices for measuring, all at once.
# The normal solver is the one that measured best on the DiLiGenT benchmark (13.1°
# mean, better than least squares on 9 of 10 objects, docs/BENCHMARK_DILIGENT.md);
# the depth method is the one with no assumption at the frame's edge; the tone is
# decoded by the file's curve once the lighting is evened.  Everything that alters
# the photographs' light for looks rather than for physics is off.
MEASUREMENT_SETTINGS = dict(
    normal_engine="Sparse-Outlier LS (RPCA)",
    depth_engine="Least-Squares (Free Boundary)",
    photo_tone="auto",
    metal_aware=False,             # the whole solve is shine-resistant already
    pre_clahe=False,               # local contrast: not photometric
    multi_norm=False,              # median brightness: hides lamp strengths instead of measuring them
    adaptive_light=0.0,
    flatten_illumination=False,    # flattens real shape as well as false tilt
    sfs_refine=False,
    pre_smooth=0.0,
    parabolic_corr=False,          # takes out real curvature too
)
# Depth on the object alone (mask_surround) is not in the list: it suits a small object
# on a dark surround and spoils a large one (helio.core.object_mask.SMALL_OBJECT), so
# it follows the last solve's colour (``measurement_mask``).


def measurement_mask(albedo) -> bool | None:
    """mask_surround for measuring, from the last solve's *albedo*: on for a small
    object on a dark surround, off otherwise; None (left as it is) before a solve."""
    from helio.core.object_mask import small_object
    return small_object(albedo)


def measurement_changes(settings, albedo=None) -> dict:
    """The MEASUREMENT_SETTINGS that differ from *settings* (ProcessingSettings), and
    mask_surround as ``measurement_mask`` decides when *albedo* (the last solve's) is given."""
    ch = {k: v for k, v in MEASUREMENT_SETTINGS.items() if getattr(settings, k, v) != v}
    mask = measurement_mask(albedo)
    if mask is not None and bool(getattr(settings, "mask_surround", False)) != mask:
        ch["mask_surround"] = mask
    return ch
