"""What the open project holds, and what is possible next.

A thin Qt holder around plain data.  Workspaces, menus and the ⌘K palette
read it to decide what is available and say why not; nothing here touches a
widget.  The solve result is a helio.application.pipeline.SolveResult.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QObject, Signal

from helio.domain import ProcessingSettings

# Why something is unavailable.  One wording per reason, everywhere.
NEED_IMAGES = "Import images first: File › Import Images…"
NEED_LIGHTS = "Set the lights first, in Capture (or choose the Uncalibrated PCA solver)"
NEED_SOLVE = "Process the images first"
NEED_SCALE = "Needs a measured scale: set one in Measure"
NOT_RUNNING = "Nothing is running"
NOT_EXPORTED = "Nothing exported yet"
BUSY = "Busy: wait for the current task to finish"
CLASSIC = "Not available in this version."


# What cameras call every picture: not a name for the object.
CAMERA_PREFIXES = ("img", "dsc", "dscf", "dscn", "_dsc", "p", "pxl", "gopr", "dji", "r", "l",
                   "image", "photo", "capture", "frame", "shot")


def shared_stem(paths) -> str:
    """The name a capture's photographs share, without the numbers that
    tell them apart (Seal_01.tif, Seal_02.tif: Seal); "" when they share
    nothing worth calling the object, or only a camera's own prefix."""
    stems = [os.path.splitext(os.path.basename(p))[0] for p in paths]
    if len(stems) < 2:
        return ""
    common = os.path.commonprefix(stems)
    common = common.rstrip("0123456789").rstrip(" _-.#(").strip()
    if len(common) < 3 or common.lower().lstrip("_") in CAMERA_PREFIXES:
        return ""
    return common


class Project(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.images: list[str] = []
        self.lights: np.ndarray | None = None      # (N, 3) +Y-up unit vectors
        self.light_source = ""                      # "sphere", "file:<name>"
        self.sphere: tuple[int, int, int] | None | bool = None   # False: searched, none found
        self.calibration = None                     # helio.core.sphere.Calibration, sphere lights
        self.light_frame: dict | None = None        # capture_tools.light_frame_check of the lights
        self.sphere_candidates: list = []
        self.excluded: set[int] = set()             # frames left out of the solve
        self.excluded_auto: set[int] = set()        # ...of those, the ones Helio left out itself
        self.exif_focal35: float | None = None
        self.capture_meta: dict = {}                # Capture details (session capture_metadata)
        self.drift: dict | None = None              # the frame drift check, run at import
        self.poi: dict = {}                         # points of interest {name: (x, y, z) mm}
        self.poi_source = ""
        self.settings = ProcessingSettings()
        self.result = None                          # SolveResult of the last finished solve
        self.solve_seconds = 0.0
        # The scale is the size of a pixel of the CURRENT RESULT (every measurement, plate and
        # export reads it that way).  A result solved at a reduced working size has bigger pixels,
        # so the scale is kept together with the working size its pixels are at (scale_basis:
        # 1 = the photographs' own pixels, which is also what it means before any solve).
        # photo_scale() is the size of a photograph's pixel, which is what a session stores and
        # what a solve is given; set_result() re-bases the scale when a result at another working
        # size arrives.
        self.scale_mm_per_px: float | None = None
        self.scale_basis: float = 1.0
        self.z_mm_per_unit: float | None = None     # height scale (MET-002)
        self.z_from_scale = False                   # the height scale is the lateral scale (a calibrated capture)
        self.accuracy: dict | None = None           # the last accuracy check on a reference (CAL-007)
        self.busy = ""                              # "sphere", "solve", or ""
        self.stage, self.fraction, self.detail = "", 0.0, ""
        self.message = ""                           # last outcome worth saying (cancelled, failed)
        self.notice = ""                            # one-off news: session opened or saved
        self.export_rows: dict[str, tuple[str, str, list]] = {}   # key → (state, note, files)
        self.last_export_dir = ""
        self.colour_check: dict | None = None       # the last ColorChecker reading (MET-010)
        # Per-channel gains (R, G, B) in linear light from a neutral patch
        # (helio.core.white_balance), or None: applied to the colour the views
        # and exports read, kept with the session.  ``white_patch`` is where it
        # was picked ((x1, y1, x2, y2), the solve's pixels), to say so.
        self.white_balance: tuple[float, float, float] | None = None
        self.white_patch: tuple | None = None
        self.current = 0                            # selected frame
        self.session_name = ""                      # the .helio's name, once opened or saved

    # ── facts ────────────────────────────────────────────────────────
    @property
    def name(self) -> str:
        """What the object is called, and so what files are named after: the
        title in Capture details, else the session's file name, else the
        name the photographs share (Seal_01.tif, Seal_02.tif: Seal), and only
        then the photographs' folder."""
        if not self.images:
            return "No project"
        title = str(dict(self.capture_meta).get("title") or "").strip()
        return (title or self.session_name or shared_stem(self.images)
                or self.folder_name)

    @property
    def folder_name(self) -> str:
        return os.path.basename(os.path.dirname(self.images[0])) or "Capture" if self.images else ""

    @property
    def solved(self) -> bool:
        return self.result is not None

    def lights_ready(self) -> bool:
        """A light for every frame that is used, and at least three used."""
        if self.lights is None or len(self.lights) != len(self.images):
            return False
        use = self.used_frames()
        return len(use) >= 3 and bool(np.isfinite(self.lights[use]).all())

    def used_frames(self) -> list[int]:
        return [i for i in range(len(self.images)) if i not in self.excluded]

    def solve_images(self) -> list[str]:
        return [self.images[i] for i in self.used_frames()]

    def solve_lights(self):
        if self.lights is None:
            return None
        return np.asarray(self.lights[self.used_frames()], np.float32)

    def set_excluded(self, i: int, out: bool):
        if out:
            self.excluded.add(i)
        else:
            self.excluded.discard(i)
        # The user has decided about this frame, so it is no longer Helio's to undo.
        self.excluded_auto.discard(i)
        self.result = None
        self.changed.emit()

    def settings_changed_since_solve(self) -> bool:
        return self.result is not None and not self.settings.solves_like(self.result.settings)

    def needs(self, need: str | None) -> str:
        """'' when *need* is met, else the reason it is not."""
        if need == "running":
            return "" if self.busy in ("solve", "export") else NOT_RUNNING
        if need == "exported":
            return "" if self.last_export_dir and not self.busy else NOT_EXPORTED
        if self.busy and need not in (None, "always"):
            return BUSY
        if need in (None, "always"):
            return ""
        if need == "classic":
            return CLASSIC
        if not self.images:
            return NEED_IMAGES
        if need == "images":
            return ""
        if need == "lights":
            return "" if self.lights_ready() else NEED_LIGHTS
        if need == "sphere":
            return "" if self.calibration is not None else "Find the chrome sphere first" 
        if need == "solvable":
            if len(self.images) < 3:
                return "Photometric stereo needs at least 3 images"
            pca = self.settings.resolved_normal_engine() == "Uncalibrated PCA"
            return "" if (pca or self.lights_ready()) else NEED_LIGHTS
        if not self.solved:
            return NEED_SOLVE
        if need == "solved":
            return ""
        if need == "scale":
            return "" if self.scale_mm_per_px else NEED_SCALE
        return CLASSIC

    # ── changes ──────────────────────────────────────────────────────
    def set_images(self, paths: list[str]):
        self.images = list(paths)
        self.lights, self.light_source, self.sphere = None, "", None
        self.calibration, self.sphere_candidates, self.excluded = None, [], set()
        self.excluded_auto = set()
        self.drift = None
        try:
            from helio.core.sphere import focal35_from_exif
            self.exif_focal35 = focal35_from_exif(self.images[0]) if self.images else None
        except Exception:
            self.exif_focal35 = None
        self.result, self.current, self.message, self.notice = None, 0, "", ""
        self.changed.emit()

    def set_lights(self, lights, source: str, sphere=None, calibration=None, sphere2=None):
        self.lights = None if lights is None else np.asarray(lights, np.float32)
        self.light_source, self.sphere = source, sphere
        self.sphere2 = sphere2 if sphere2 is not None else getattr(calibration, "sphere2", None)
        self.calibration = calibration
        self.light_frame = None
        # A frame with no light is left out; one the user left out stays out.
        #
        # Only by hand, though. An automatic exclusion belongs to the detection that
        # made it: carrying it into the next one left frames ticked off whose light had
        # since been found perfectly well, with no reason shown anywhere and nothing to
        # undo it but noticing. Re-reading the lights therefore clears Helio's own
        # exclusions and makes them again from what it just found.
        if self.lights is not None and len(self.lights) == len(self.images):
            bad = {i for i in range(len(self.lights)) if not np.isfinite(self.lights[i]).all()}
            by_hand = {i for i in self.excluded
                       if i < len(self.images) and i not in self.excluded_auto}
            self.excluded, self.excluded_auto = by_hand | bad, bad
        else:
            self.excluded, self.excluded_auto = set(), set()
        self.result = None                 # solved with other lights: no longer this capture's
        self.notice = ""
        self.changed.emit()

    def set_busy(self, what: str):
        self.busy = what
        if what:
            self.stage, self.fraction, self.detail, self.message = "", 0.0, "", ""
        self.changed.emit()

    def set_progress(self, stage: str, fraction: float, detail: str = ""):
        if stage:
            self.stage = stage
        self.fraction, self.detail = fraction, detail
        self.changed.emit()

    # ── the scale, and the working size its pixels are at ────────────
    def result_work_scale(self) -> float:
        """The working size of the current result (1 = full size; also 1 with no result)."""
        return float(getattr(self.result, "work_scale", 1.0) or 1.0) if self.result is not None else 1.0

    def photo_scale(self) -> float | None:
        """mm per pixel of the photographs as they are on disk, or None: what a session
        stores and a solve is given, whatever size the current result was solved at."""
        return self.scale_mm_per_px * self.scale_basis if self.scale_mm_per_px else None

    def set_scale_photo(self, mm_per_photo_px) -> None:
        """A scale given for the photographs' pixels (a session, a rig profile): kept in
        the pixels of the current result."""
        ws = self.result_work_scale()
        self.scale_mm_per_px = float(mm_per_photo_px) / ws if mm_per_photo_px else None
        self.scale_basis = ws

    def set_scale_measured(self, mm_per_result_px) -> None:
        """A scale measured on the picture being looked at: already the result's pixels."""
        self.scale_mm_per_px = float(mm_per_result_px) if mm_per_result_px else None
        self.scale_basis = self.result_work_scale()

    def clear_scale(self) -> None:
        self.scale_mm_per_px, self.scale_basis = None, 1.0

    def rebase_scale(self, new_work_scale: float) -> None:
        """The scale for pixels at *new_work_scale*, from the pixels it was in."""
        new = float(new_work_scale or 1.0)
        if self.scale_mm_per_px:
            self.scale_mm_per_px = self.scale_mm_per_px * self.scale_basis / new
        self.scale_basis = new

    def set_result(self, result, seconds: float):
        self.rebase_scale(getattr(result, "work_scale", 1.0))
        self.result, self.solve_seconds = result, seconds
        self.busy, self.message = "", ""
        self.changed.emit()

    def finish(self, message: str):
        """A run ended without a new result; the previous result is kept."""
        self.busy, self.message = "", message
        self.changed.emit()

    def set_export_rows(self, rows: dict):
        self.export_rows = dict(rows)
        self.changed.emit()

    def update_settings(self, **kw):
        self.settings = replace(self.settings, **kw)
        self.changed.emit()

    def select(self, i: int):
        if 0 <= i < len(self.images) and i != self.current:
            self.current = i
            self.changed.emit()
