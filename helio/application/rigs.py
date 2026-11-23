"""Rig profiles: what stays the same from capture to capture on a fixed rig or
a dome (the light directions, where the ball was, the dome radius and
near-field, the scale, how the photographs record light), saved once and
applied to the next capture of the same number of frames.

Kept as JSON in the preferences (``rig/profiles``): {name: profile}."""
from __future__ import annotations

import datetime
import json

import numpy as np

KEY = "rig/profiles"


def load(text: str) -> dict:
    try:
        d = json.loads(text or "{}")
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


def dump(profiles: dict) -> str:
    return json.dumps(profiles)


def make(lights, *, sphere=None, sphere2=None, dome_radius_mm=None, near_field=False,
         near_field_from="ball", mm_per_px=None, photo_tone="", note: str = "",
         light_field: str = "", lens_calibration: str = "") -> dict:
    """A profile from what a calibrated capture knows."""
    L = np.asarray(lights, np.float64)
    if L.ndim != 2 or L.shape[1] != 3 or len(L) < 3:
        raise ValueError("A rig profile needs at least three measured lights.")
    if not np.isfinite(L).all():
        raise ValueError("Some frames have no light; set every light before saving the rig.")
    tup = lambda s: None if s is None else [float(v) for v in tuple(s)[:3]]
    return dict(lights=L.round(6).tolist(), n=len(L), sphere=tup(sphere), sphere2=tup(sphere2),
                dome_radius_mm=None if dome_radius_mm is None else float(dome_radius_mm),
                near_field=bool(near_field), near_field_from=str(near_field_from),
                mm_per_px=None if not mm_per_px else float(mm_per_px),
                photo_tone=str(photo_tone or ""), note=str(note or ""),
                light_field=str(light_field or ""), lens_calibration=str(lens_calibration or ""),
                saved=datetime.date.today().isoformat())


def fitting(profiles: dict, n_frames: int) -> list[str]:
    """The profiles with one light for each of *n_frames* frames."""
    return sorted(k for k, p in profiles.items() if int(p.get("n", 0)) == int(n_frames))


def describe(p: dict) -> str:
    """A profile in a few words."""
    bits = [f"{p.get('n', 0)} lights"]
    if p.get("sphere"):
        bits.append("ball's place")
    if p.get("near_field"):
        bits.append(f"near-field {p.get('dome_radius_mm') or 0:.0f} mm")
    if p.get("mm_per_px"):
        bits.append(f"scale {p['mm_per_px']:.4f} mm/px")
    if p.get("light_field"):
        bits.append("white-board light field")
    if p.get("lens_calibration"):
        bits.append("lens corrected")
    return " · ".join(bits)
