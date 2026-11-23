"""Processing settings as one plain, immutable snapshot (DOM-001).

The legacy worker read eighteen Tk variables from its background thread, so a
setting changed mid-run could reach half a solve, and the reads could not run
without a live Tk interpreter at all.  ProcessingSettings is taken once, on
the thread that owns the widgets, and the worker reads only this.

Plain Python on purpose - no Tk, no Qt - so the Qt app, the CLI and tests can
build one without a window, and a job can carry it to a worker or a process.

Field defaults are the legacy app's live defaults (audit/runtime-tkvars.json,
7 Oct 2026).  Values are stored as the user chose them; nothing is clamped or
"improved" here, because the solve must match the legacy app exactly.  Legacy
engine names from old sessions are mapped by resolved_normal_engine /
resolved_depth_engine, as RTIViewerPro._normalise_*_engine does.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

# One list of each, in dropdown order: helio.domain.registry (it had drifted: this copy
# lacked the two depth methods added on 2026-10-07, so the full check never ran them).
from helio.domain.registry import DEPTH_METHODS as DEPTH_INTEGRATORS  # noqa: E402
from helio.domain.registry import NORMAL_SOLVERS  # noqa: E402,F401
LEGACY_NORMAL_ALIASES = {"PyTorch Deep PS": "Uncalibrated PCA",
                         # renamed 2026-11: it weights mid-tones; it is not a metal solver
                         "Robust Least Squares (Shiny Metal)": "Mid-Tone Weighted LS"}
LEGACY_DEPTH_ALIASES = {"PyTorch L1 Edge-Preserving": "Fast Poisson (FFT)"}


@dataclass(frozen=True)
class ProcessingSettings:
    """Everything the solve reads that a user can set."""

    normal_engine: str = "Linear Least Squares"
    depth_engine: str = "Fast Poisson (FFT)"
    # How the photos' pixel values relate to light (helio.core.radiometry):
    # "auto" (the default) decodes camera files by their own curve (from
    # 2026-11-12 whether or not the lighting is evened: as stored, fine relief
    # came out at 60% of its depth, docs/BENCHMARK_DILIGENT_PI.md);
    # "stored" solves on the values as they are, "srgb" decodes them to linear
    # light first, "linear" takes them as light already.
    photo_tone: str = "auto"
    # Hand-held flash (CAL-003): each frame scaled to one flash strength,
    # measured on a flat grey reference in the shot.  flash_ref is its box,
    # "x1,y1,x2,y2" in full-size pixels; "" for none.
    even_flash: bool = False
    flash_ref: str = ""
    # CAL-004: the flash's fall-off across the frame, measured on areas
    # known to be flat ('x1,y1,x2,y2;…', the photographs' pixels).  When on,
    # it does the grey card's work too (its level is each flash's strength).
    even_flat: bool = False
    flat_areas: str = ""
    # Semi-calibrated (helio.core.robust_ps.light_strengths): the lamps'
    # strengths found from the photographs themselves, with the directions
    # held fixed, and every frame evened to them.  Nothing to mark; the flat
    # areas and the grey patch take precedence when they are on.
    semi_calibrated: bool = False
    # Ambient (lights-off) frame: a photograph taken with the lamps off and
    # everything else as for the lit frames.  Its light, which every frame
    # shares (daylight, room light), is subtracted in linear light from each
    # before anything else reads them (helio.core.radiometry.subtract_ambient).
    # The photograph's path; "" for none.
    ambient_frame: str = ""
    # Precision measurement (CAL-005, CAL-006), for a fixed rig: each light's
    # strength and fall-off measured on a white board (helio.core.light_field,
    # a .npz path), and the lens's distortion from chessboard photographs
    # (helio.core.lens, a .npz path).  "" = not used.
    light_field: str = ""
    lens_calibration: str = ""
    # pre-processing
    flatten_illumination: bool = False
    # Retired 2026-11 (docs/OVERLAP_AUDIT.md): scaling every frame to one median
    # brightness also took out real shading, since a frame's median depends on
    # which way the surface faces its light.  Kept so old sessions load; the
    # solve no longer applies it (pipeline.solve).  The grey patch, flat areas
    # and semi-calibrated measure lamp strength properly.
    multi_norm: bool = False
    pre_clahe: bool = False
    pre_clahe_clip: float = 2.0
    pre_clahe_tile: int = 16
    auto_align: bool = False
    adaptive_light: float = 0.0
    # solver options
    shadow_thresh: int = 12             # Shadow-Rejection LS only
    # Sparse-Outlier LS (RPCA): the outlier threshold, a fraction of each
    # pixel's mean reading.  Lower sets aside more readings.
    rpca_lambda: float = 0.3
    # Metal-aware solve: the chosen solver everywhere, and where the surface
    # shines (gilding, clasps, metal point; robust_ps.metal_mask) metal_solver
    # instead, blended at the edges.  Sensitivity 0-1: higher marks more; 0.3 keeps
    # a parchment sheet's lamp-side edge and grey card out on a hand-lit capture.
    metal_aware: bool = False
    metal_solver: str = "Sparse-Outlier LS (RPCA)"
    metal_sensitivity: float = 0.3
    # Near-field light correction: lights at a finite distance, dome_radius_mm
    # from the object.  Until 2026-10-07 the classic checkbox and radius were
    # saved and reported but never read; now they are (FIX-012).
    near_field: bool = False
    dome_radius_mm: float = 200.0
    near_field_from: str = "ball"       # "ball": the dome centred where the lights were measured;
                                        # "centre": on the picture's middle (as before 2026-10-20)
    sfs_refine: bool = False
    sfs_iters: int = 5
    sfs_prior: float = 0.70
    flat_correction: bool = False       # needs a flat reference to take effect
    flat_corr_strength: float = 1.0
    # depth
    # Pre-Integration Smoothing, 0 (off) to 10: the classic slider
    # (depth_smooth_var).  Process ignored it until 2026-10-07 and Recalculate
    # Depth alone read it (KD-011); both now apply it, with one filter
    # (helio.application.process_tools.smooth_normals), to the integration only.
    pre_smooth: float = 0.0
    parabolic_corr: bool = False
    # A small object on a dark surround (a punch, a coin on velvet): the depth
    # is integrated on the object alone, so the surround's noise stays out of
    # it (helio.core.object_mask).  Does nothing when the object fills the frame.
    mask_surround: bool = False       # off by default (2026-11-11): switch it on for a small object on a dark surround
    invert_relief: bool = False
    # Working size (optional, for machines short of memory): the photographs are
    # reduced to this fraction of their size as they are read, so a capture that
    # would not fit in memory can be solved.  1.0 = full size (the default).
    # Everything in the result is then in the reduced pixels; the result records
    # the fraction (SolveResult.work_scale).  0 = automatic: full size when the
    # solve fits in free memory, else the largest reduced size that does.
    # HELIO_WORK_SCALE overrides it.
    work_scale: float = 1.0

    def __post_init__(self):
        # Coerce the types Tk hands back (IntVar for booleans, str/float mixes)
        # so equality and dict round trips are exact.  No range changes.
        for f in fields(self):
            v = getattr(self, f.name)
            want = {"str": str, "bool": bool, "int": int, "float": float}[f.type]
            if not isinstance(v, want) or (want is int and isinstance(v, bool)):
                object.__setattr__(self, f.name, want(v))
        # Old names, made current (docs/OVERLAP_AUDIT.md).  The near-field solver was
        # the Near-field correction checkbox under a second name (same code): it is
        # the checkbox now, with plain least squares.
        if self.normal_engine == "Near-Field Attenuation":
            object.__setattr__(self, "normal_engine", "Linear Least Squares")
            object.__setattr__(self, "near_field", True)
        for k in ("normal_engine", "metal_solver"):
            v = getattr(self, k)
            if v in LEGACY_NORMAL_ALIASES:
                object.__setattr__(self, k, LEGACY_NORMAL_ALIASES[v])

    def resolved_normal_engine(self) -> str:
        return LEGACY_NORMAL_ALIASES.get(self.normal_engine, self.normal_engine)

    def resolved_depth_engine(self) -> str:
        return LEGACY_DEPTH_ALIASES.get(self.depth_engine, self.depth_engine)

    def lighting_evened(self) -> bool:
        """A white-board light field (one that exists) or flat areas even the lighting."""
        import os
        return bool((self.light_field and os.path.exists(self.light_field))
                    or (self.even_flat and self.flat_areas))

    def resolved_photo_tone(self) -> str:
        """The tone the solve uses ("auto" made definite, helio.core.radiometry.resolve_tone)."""
        from helio.core.radiometry import resolve_tone
        return resolve_tone(self.photo_tone, evened=self.lighting_evened())

    def solves_like(self, solved: "ProcessingSettings") -> bool:
        """Would solving with these settings give the result *solved* records?

        The solve records what it used, not what was asked: "auto" tone made
        definite (decoded) and the retired frame-brightness levelling turned off.  Those are not changes
        the person made, so they do not ask for the images to be processed again."""
        from dataclasses import replace
        asked = self
        if asked.multi_norm:
            asked = replace(asked, multi_norm=False)
        if asked.photo_tone == "auto" and solved.photo_tone != "auto":
            # what Automatic gives now: a result solved as stored before it decoded is not this solve
            asked = replace(asked, photo_tone=asked.resolved_photo_tone())
        return asked == solved

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProcessingSettings":
        """Unknown keys are ignored and missing ones take the defaults, the
        same tolerance the session reader has."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
