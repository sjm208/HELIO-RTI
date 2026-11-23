"""Read and write .helio session files, GUI-free (SES-001).

One codec for both apps.  The format is the classic one exactly
(contracts.md §6): plain JSON, two-space indent, keys in the classic order,
no version key.  Every key is optional; a missing key takes the classic
default; retired engine names map to today's; keys this version does not
know are kept and written back, so a session survives any app that opens it.

The light directions travel too since SES-003 (approved 2026-10-07): two
optional keys after the classic ones, written only when there are lights, so
a session without them is byte for byte what the classic app always wrote.
Older apps ignore them.  Since FIX-010 one more optional key, ``photo_tone``,
comes first after the classic keys: how the solve read the photos.  The optional
``work_scale`` (the working size, below) is written only when it is not full size.
``mm_per_px`` is always the size of a pixel of the photographs as they are on disk,
whatever working size a result was solved at.  Other format changes (a version key, relative paths)
need an announced, reviewed ticket (BLUEPRINT rule 8).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields, replace
from typing import Any

from helio.domain.settings import LEGACY_DEPTH_ALIASES, LEGACY_NORMAL_ALIASES

# (key, default) in the order the classic app writes them.
_KEYS: tuple[tuple[str, Any], ...] = (
    ("object_id", ""),
    ("plate_paths", []),
    ("last_image_folder", ""),
    ("render_engine", "Standard PBR"),
    ("normal_engine", "Linear Least Squares"),
    ("depth_engine", "Fast Poisson (FFT)"),
    ("bookmarks", {}),
    ("sharpen_params", {}),
    ("watertight", True),
    ("black_sides", True),
    ("vtx_color", True),
    ("sketchfab", False),
    ("pbr_ao", True),
    ("pbr_disp", True),
    ("normal_linear", True),
    ("normal_flip_y", True),
    ("bake_normals", False),
    ("bake_depth_pct", 5.0),            # percent (a fraction only inside export settings)
    ("invert_relief", False),           # added 2026-10-07
    ("export_hf", 0.05),
    ("export_amp", 1.0),
    ("hybrid_depth_enabled", False),
    ("hybrid_detail_mix", 0.12),
    ("crop_box", None),
    ("pub_format", "pdf"),
    ("pub_bg", False),
    ("mm_per_px", None),
    ("z_mm_per_unit", None),
    ("drawn_scale_px", None),
    ("calculated_ppi", None),
    ("near_field", False),
    ("flat_correction", False),
    ("scale_mm_val", ""),
    ("engine_settings", {}),
    ("capture_metadata", {}),
)
# SES-003: optional, written after the classic keys and only when present.
#   light_vectors: one [x, y, z] per image, unit length, +Y up (contracts §3.2)
#   light_source:  where they came from: "sphere", "file:<name>", "dome",
#                  or "measured" (the classic app cannot tell sphere from .lp)
#   photo_tone:    how the solve read the photos' values (FIX-010): "srgb",
#                  "linear" or "stored".  Absent = "stored", which is what every
#                  session written before 2026-10-07 was solved with.
#   ambient_frame: the lights-off photograph subtracted from every frame.
#   height_basis:  what z_mm_per_unit is measured on: "true-height-px" (mm per
#                  pixel of true height, helio.application.heights, from
#                  2026-11-05).  Absent with a height scale = measured on the
#                  display depth: it cannot be carried over, and is dropped.
#   even_flash, flash_ref, even_flat, flat_areas: the hand-held flash's
#                  corrections (CAL-003, CAL-004): the grey reference's box
#                  and the flat areas ("x1,y1,x2,y2;…"), the photographs' pixels.
_OPTIONAL: tuple[tuple[str, Any], ...] = (("photo_tone", ""), ("light_vectors", None),
                                          ("light_source", ""), ("even_flash", False),
                                          ("flash_ref", ""), ("even_flat", False),
                                          ("flat_areas", ""), ("near_field_from", "ball"),
                                          ("work_scale", 1.0), ("ambient_frame", ""),
                                          ("height_basis", ""))
_DEFAULTS = dict(_KEYS + _OPTIONAL)


def _default(name):
    v = _DEFAULTS[name]
    return field(default_factory=lambda v=v: type(v)()) if isinstance(v, (list, dict)) else field(default=v)


@dataclass
class Session:
    object_id: str = _default("object_id")
    plate_paths: list = _default("plate_paths")
    last_image_folder: str = _default("last_image_folder")
    render_engine: str = _default("render_engine")
    normal_engine: str = _default("normal_engine")
    depth_engine: str = _default("depth_engine")
    bookmarks: dict = _default("bookmarks")
    sharpen_params: dict = _default("sharpen_params")
    watertight: bool = _default("watertight")
    black_sides: bool = _default("black_sides")
    vtx_color: bool = _default("vtx_color")
    sketchfab: bool = _default("sketchfab")
    pbr_ao: bool = _default("pbr_ao")
    pbr_disp: bool = _default("pbr_disp")
    normal_linear: bool = _default("normal_linear")
    normal_flip_y: bool = _default("normal_flip_y")
    bake_normals: bool = _default("bake_normals")
    bake_depth_pct: float = _default("bake_depth_pct")
    invert_relief: bool = _default("invert_relief")
    export_hf: float = _default("export_hf")
    export_amp: float = _default("export_amp")
    hybrid_depth_enabled: bool = _default("hybrid_depth_enabled")
    hybrid_detail_mix: float = _default("hybrid_detail_mix")
    crop_box: Any = _default("crop_box")
    pub_format: str = _default("pub_format")
    pub_bg: bool = _default("pub_bg")
    mm_per_px: Any = _default("mm_per_px")
    z_mm_per_unit: Any = _default("z_mm_per_unit")
    drawn_scale_px: Any = _default("drawn_scale_px")
    calculated_ppi: Any = _default("calculated_ppi")
    near_field: bool = _default("near_field")
    flat_correction: bool = _default("flat_correction")
    scale_mm_val: str = _default("scale_mm_val")
    engine_settings: dict = _default("engine_settings")
    capture_metadata: dict = _default("capture_metadata")
    photo_tone: str = ""                           # FIX-010; "" = not recorded
    light_vectors: Any = None                      # SES-003
    light_source: str = ""
    even_flash: bool = False                       # CAL-003
    flash_ref: str = ""
    even_flat: bool = False                        # CAL-004
    flat_areas: str = ""
    near_field_from: str = "ball"                  # the near-field dome's centre: "ball" or "centre"
    work_scale: float = 1.0                        # the working size: 1 = full, 0 = automatic, else a fraction
    ambient_frame: str = ""                        # the lights-off photograph subtracted, or ""
    height_basis: str = ""                         # what z_mm_per_unit is measured on (application.heights)
    extra: dict = field(default_factory=dict)      # keys this version does not know

    def lights(self):
        """The stored light directions as an (N, 3) float32 array, or None."""
        if not self.light_vectors:
            return None
        import numpy as np
        return np.asarray(self.light_vectors, dtype=np.float32).reshape(-1, 3)

    def processing_settings(self, base):
        """*base* (a helio.domain.ProcessingSettings) with the session's solve choices."""
        from helio.core.radiometry import LEGACY_TONE, PHOTO_TONES
        tone = self.photo_tone if self.photo_tone in PHOTO_TONES else LEGACY_TONE
        return replace(base, normal_engine=self.normal_engine, depth_engine=self.depth_engine,
                       invert_relief=bool(self.invert_relief),
                       flat_correction=bool(self.flat_correction), photo_tone=tone,
                       near_field=bool(self.near_field), even_flash=bool(self.even_flash),
                       flash_ref=str(self.flash_ref or ""), even_flat=bool(self.even_flat),
                       flat_areas=str(self.flat_areas or ""),
                       near_field_from="centre" if self.near_field_from == "centre" else "ball",
                       work_scale=self._work_scale(), ambient_frame=str(self.ambient_frame or ""))

    def _work_scale(self) -> float:
        try:
            v = float(self.work_scale)
        except (TypeError, ValueError):
            return 1.0
        return v if v == 0.0 or 0.1 <= v < 0.995 else 1.0


KEYS = tuple(k for k, _ in _KEYS)
OPTIONAL = tuple(k for k, _ in _OPTIONAL)
assert KEYS + OPTIONAL == tuple(f.name for f in fields(Session) if f.name != "extra")


def locate(path: str, *folders: str) -> str:
    """*path* if it is there; else a file of the same name in one of *folders* (the
    photographs' folder, the session's), for a capture copied elsewhere with its
    files; else *path* as it was."""
    if not path or os.path.exists(path):
        return path
    name = os.path.basename(path)
    for f in folders:
        q = os.path.join(f, name) if f else ""
        if q and os.path.exists(q):
            return q
    return path


def decode(state: dict) -> Session:
    """A Session from a parsed .helio dict, as the classic reader takes it."""
    s = Session(**{k: state[k] for k in KEYS + OPTIONAL if k in state},
                extra={k: v for k, v in state.items() if k not in _DEFAULTS})
    s.normal_engine = LEGACY_NORMAL_ALIASES.get(s.normal_engine, s.normal_engine)
    s.depth_engine = LEGACY_DEPTH_ALIASES.get(s.depth_engine, s.depth_engine)
    return s


def encode(s: Session) -> dict:
    """The dict the classic writer produces: known keys in order, the lights if
    there are any (SES-003), then any kept extras."""
    out = {k: getattr(s, k) for k in KEYS}
    if s.photo_tone:
        out["photo_tone"] = s.photo_tone
    if s.light_vectors:
        out["light_vectors"] = [[float(c) for c in v] for v in s.light_vectors]
        out["light_source"] = s.light_source
    if s.flash_ref or s.even_flash:
        out["even_flash"], out["flash_ref"] = bool(s.even_flash), s.flash_ref
    if s.flat_areas or s.even_flat:
        out["even_flat"], out["flat_areas"] = bool(s.even_flat), s.flat_areas
    if s.near_field_from == "centre":
        out["near_field_from"] = "centre"
    if s.work_scale not in (1.0, None):
        out["work_scale"] = float(s.work_scale)
    if s.ambient_frame:
        out["ambient_frame"] = s.ambient_frame
    if s.z_mm_per_unit and s.height_basis:
        out["height_basis"] = s.height_basis
    for k, v in s.extra.items():
        out.setdefault(k, v)
    return out


def read(path: str) -> Session:
    with open(path) as f:
        return decode(json.load(f))


def with_lights(s: Session, lights, source: str) -> Session:
    """*s* carrying *lights* ((N, 3), +Y up) from *source*, or none if *lights* is None."""
    if lights is None or not len(lights):
        return replace(s, light_vectors=None, light_source="")
    return replace(s, light_vectors=[[float(c) for c in v] for v in lights],
                   light_source=source)


def write(path: str, s: Session) -> str:
    with open(path, "w") as f:
        json.dump(encode(s), f, indent=2)
    return path
