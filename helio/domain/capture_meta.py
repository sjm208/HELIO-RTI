"""Capture details and points of interest, GUI-free (IMP-003, MET-012).

The fields of the classic Capture Metadata panel, in its groups and order;
they travel in the session's ``capture_metadata`` and fill the provenance
record and reports.  Points of interest are measured 3D positions (lights,
camera, origin) in millimetres for metrological traceability; the classic
app could export them but had lost the loader, so they could never be set.
"""
from __future__ import annotations

import csv
import json
import time

# (group, [(label, key, hint, tip)])
FIELDS = (
    ("Institution and operator", [
        ("Institution", "institution", "e.g. Helio", "The holding or capturing institution: credited in every export."),
        ("Department / collection", "collection", "e.g. Cultural Heritage Imaging Lab",
         "The department or collection the object belongs to."),
        ("Operator", "operator", "Name of the person capturing", "Who made the capture: credited as the creator."),
        ("Object title", "title", "e.g. Intaglio seal, limestone",
         "What the object is called: exports and files are named after it."),
        ("Date of capture", "capture_date", "YYYY-MM-DD", "When the photographs were taken."),
    ]),
    ("Camera and optics", [
        ("Camera model", "camera", "e.g. Phase One IQ4 100MP", "The camera body, for the processing record."),
        ("Lens", "lens", "e.g. Schneider 120mm Macro", "The lens and its focal length."),
        ("Sensor", "sensor", "e.g. 100MP CMOS", "The sensor: its type and size in pixels."),
        ("Pixel pitch (µm)", "pixel_pitch", "e.g. 3.76",
         "The sensor's pixel size: with the magnification it gives the scale."),
    ]),
    ("Capture settings", [
        ("Lighting", "lights", "e.g. LED x8, 4000K", "The lights: how many, what kind, their colour temperature."),
        ("Stand / rig", "stand", "e.g. Horizontal stand", "The stand, dome or rig the capture was made on."),
        ("Image count", "img_count", "", "Filled in from the frames."),
        ("Working distance (mm)", "working_dist", "e.g. 430",
         "From the lens to the object, in millimetres."),
        ("Scale (mm/px)", "scale_mmpx", "e.g. 0.042", "Filled in when a scale is measured."),
    ]),
    ("Notes and reference", [
        ("Notes", "notes", "Any departure from the usual protocol",
         "Anything unusual about this capture, written into the record."),
        ("DOI / reference", "doi", "e.g. 10.3390/heritage8040129",
         "A publication or dataset this capture belongs to."),
    ]),
)
KEYS = tuple(k for _g, fs in FIELDS for _l, k, _h, _t in fs)


def defaults(n_images: int = 0) -> dict:
    """The classic panel's Defaults (its rig, as the classic app ships it)."""
    return {"institution": "Helio", "collection": "Cultural Heritage Imaging Lab",
            "operator": "Scott Maloney", "camera": "Phase One IQ4 100MP",
            "sensor": "100MP CMOS (8682x11574)", "pixel_pitch": "3.76",
            "capture_date": time.strftime("%Y-%m-%d"), "img_count": str(n_images)}


def write_json(path: str, meta: dict) -> str:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


def read_json(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("Not a capture-details file (expected a JSON object).")
    return {str(k): str(v) for k, v in d.items()}


def write_csv(path: str, meta: dict) -> str:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Field", "Value"])
        for k, v in meta.items():
            w.writerow([k, v])
    return path


# ── points of interest ──────────────────────────────────────────────────

def read_poi(path: str) -> dict:
    """Points from a CSV of name, X_mm, Y_mm, Z_mm (a header row is optional)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 4 or not row[0].strip():
                continue
            try:
                xyz = tuple(float(v) for v in row[1:4])
            except ValueError:
                continue                       # the header
            out[row[0].strip()] = xyz
    if not out:
        raise ValueError("No points found: expected rows of name, X_mm, Y_mm, Z_mm.")
    return out


def write_poi(path: str, poi: dict, profile: str = "") -> str:
    """The classic Export PoI / 3D Positions file."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "X_mm", "Y_mm", "Z_mm", "notes"])
        for name, (x, y, z) in poi.items():
            note = ("acquisition plane origin" if name == "Origin" else
                    "camera position" if name == "Camera" else f"light source — {profile}")
            w.writerow([name, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", note])
    return path
