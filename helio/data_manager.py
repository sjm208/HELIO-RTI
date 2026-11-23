"""
Central state holder for HELIO.

Owns all large arrays and shared-memory IPC objects.  Every module imports
from here to access the current scan data without passing ``self`` around.

Rules
-----
* The UI layer (``helio.ui``) creates one ``DataManager`` at startup and
  stores it as ``app.data``.
* Core / render / IO modules receive the manager as a constructor or function
  argument so they never import the UI layer.
* Large arrays live **only** on the manager; modules hold references during
  computation then release them.
* Call ``clear_arrays()`` between scans to free memory.
* Call ``cleanup()`` for a full teardown of arrays and shared memory.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing
from typing import Any, Optional

_log = logging.getLogger(__name__)

import numpy as np

from helio.config import DEFAULT_ENGINE_SETTINGS, SHARED_STATE_DEFAULTS


class DataManager:
    """Holds all scan data, engine settings, and inter-process shared state.

    Usage
    -----
    >>> dm = DataManager()
    >>> dm.plate_paths = ["/path/to/img1.tif", ...]
    >>> dm.run_processing()
    >>> dm.normals_full    # → np.ndarray
    >>> dm.cleanup()       # → del + GC
    """

    def __init__(self) -> None:
        # ── File paths ─────────────────────────────────────────────────
        self.plate_paths: list[str] = []
        self.ball_paths: list[str] = []
        self.last_folder: str = ""
        self.object_id: str = "helio_scan"

        # ── Image geometry (set after processing) ───────────────────────
        self.h: int = 0
        self.w: int = 0
        self.ph: int = 0   # proxy height
        self.work_scale: float = 1.0     # < 1: solved from reduced photographs (pipeline.SolveResult)
        self.source_size = None          # (w, h) of the photographs on disk
        self.solve_region = None         # a region solved: (x1, y1, x2, y2) as fractions of the photographs
        self.pw: int = 0   # proxy width

        # ── Calibration ─────────────────────────────────────────────────
        self.current_L_mat: Optional[np.ndarray] = None      # (n, 3) light matrix
        self.pseudo_L_mat: Optional[np.ndarray] = None       # pseudo-L for uncalibrated
        self.light_vecs: list[tuple[float, float, float]] = []
        self.n: int = 0                                      # number of lights
        self.selected: int = 0
        self.scale_mm: float = 0.0
        self.drawn_scale_px: int = 0
        self._mm_per_px: float = 0.0
        self._z_mm_per_unit: float = 0.0

        # ── Large processing arrays (populated by core solvers) ─────────
        self.normals_full: Optional[np.ndarray] = None        # (h, w, 3)  float32
        self.normals_proxy: Optional[np.ndarray] = None       # (ph, pw, 3) float32
        self.normals_robust: Optional[np.ndarray] = None      # robust MVS variant
        self.normals_robust_proxy: Optional[np.ndarray] = None
        self.median_img: Optional[np.ndarray] = None          # (h, w)
        self.median_proxy: Optional[np.ndarray] = None

        # Stack-derived quality/legibility maps. Computed once in
        # _run_processing_body while grey_stack is in scope (the whole light
        # stack is discarded after a run, so an engine cannot derive these
        # lazily) and held as uint8: they are display maps, and 1/255 is far
        # more precision than a per-pixel frame *fraction* needs. Four
        # float32 copies would have cost ~31 MB against the 392 MB the
        # memory work got the whole app down to.
        self.qc_resid_full: Optional[np.ndarray] = None        # (h, w) uint8
        self.qc_resid_proxy: Optional[np.ndarray] = None
        self.qc_sigma_full: Optional[np.ndarray] = None        # (h, w) float32, degrees
        self.qc_sigma_proxy: Optional[np.ndarray] = None
        self.ml_fusion_full: Optional[np.ndarray] = None       # (h, w) uint8
        # Where the metal-aware solve re-solved the normals (gilding, clasps):
        # 255 = re-solved.  None unless that option ran.
        self.metal_mask_full: Optional[np.ndarray] = None      # (h, w) uint8
        self.ml_fusion_proxy: Optional[np.ndarray] = None

        self.depth_vis_p: Optional[np.ndarray] = None         # Poisson depth (h, w)
        self.depth_poisson: Optional[np.ndarray] = None       # integrated depth, true heights (application.heights)
        self.depth_proxy_p: Optional[np.ndarray] = None
        self.depth_vis_fc: Optional[np.ndarray] = None        # Frankot-Chellappa depth
        self.depth_proxy_fc: Optional[np.ndarray] = None

        self.albedo_rgb_full: Optional[np.ndarray] = None     # (h, w, 3)  uint8
        self.albedo_proxy: Optional[np.ndarray] = None
        self.albedo_grey_full: Optional[np.ndarray] = None
        self.median_colour: Optional[tuple] = None            # (full, proxy): the median of the frames
        self.solved_colour: Optional[tuple] = None            # (full, proxy): RAW captures, else None
        self.roughness_full: Optional[np.ndarray] = None
        self.roughness_proxy: Optional[np.ndarray] = None

        self.pca_full: Optional[np.ndarray] = None            # (h, w, 3)  float32 0-255
        self.spec_var: Optional[np.ndarray] = None            # specular variance (h, w)

        # ── Transmissive (backlit) capture — optional, display-only ─────
        # A single extra photograph lit from underneath.  NEVER enters
        # plate_paths, grey_stack, current_L_mat, or any solver input —
        # it is a visualisation/analysis channel only.
        self.transmissive_path: str = ""                      # source file path
        self.transmissive_raw: Optional[np.ndarray] = None    # as loaded, untouched
        self.transmissive_aligned: Optional[np.ndarray] = None  # warped to solve space
        self.transmissive_transform: Optional[dict] = None    # stored alignment transform

        self.ptm_coeffs: Optional[np.ndarray] = None          # (6, h*w)  float32
        self.ptm_proxy: Optional[np.ndarray] = None
        self.hsh_coeffs: Optional[np.ndarray] = None
        # Tight machines keep only the previews; ensure_ptm_hsh fits the rest again.
        self.ptm_deferred: bool = False
        self.ptm_planes = None                                # six preview-size PTM planes
        self.rbf = None                                       # helio.core.rbf_relight.build: compressed frames
        self.ptm_refit = None

        # ── Cached per-engine renders (populated on demand) ────────────
        self._svf_full: Optional[np.ndarray] = None
        self._svf_proxy: Optional[np.ndarray] = None
        self._msrm_full: Optional[np.ndarray] = None
        self._msrm_proxy: Optional[np.ndarray] = None
        self._curv_full: Optional[np.ndarray] = None
        self._curv_proxy: Optional[np.ndarray] = None
        self._slope_full: Optional[np.ndarray] = None
        self._slope_proxy: Optional[np.ndarray] = None
        self._pos_opn_full: Optional[np.ndarray] = None
        self._neg_opn_full: Optional[np.ndarray] = None
        self._loc_dom_full: Optional[np.ndarray] = None
        self._loc_dom_proxy: Optional[np.ndarray] = None
        self._slrm_full: Optional[np.ndarray] = None
        self._slrm_proxy: Optional[np.ndarray] = None
        self._cica_cache: Optional[dict] = None
        self._rbd_cache: Optional[dict] = None
        self._rbd_plate_cache: Optional[dict] = None
        self._rbd_plate_cache_key: Optional[tuple] = None

        # ── Engine settings ────────────────────────────────────────────
        self.engine_settings: dict[str, dict] = {}
        self.render_engine: str = "Standard PBR"
        self.normal_engine: str = "Linear Least Squares"
        self.depth_engine: str = "Fast Poisson (FFT)"
        self.view_mode: str = "Shading"

        # ── Inter-process shared state (3D viewer ↔ main) ─────────────
        self.shared_state: dict[str, Any] = {}
        self._build_shared_state()

    def ensure_ptm_hsh(self, progress=None):
        """The PTM and HSH coefficients, fitted again from the photographs when this
        machine did not keep them (see pipeline.ensure_ptm_hsh).  Slow the first time."""
        if self.ptm_coeffs is None and self.ptm_deferred and self.ptm_refit is not None:
            self.ptm_coeffs, self.hsh_coeffs = self.ptm_refit(progress)
        return self.ptm_coeffs, self.hsh_coeffs

    # ── Shared-state factory ──────────────────────────────────────────

    def _build_shared_state(self) -> None:
        self.shared_state = {}
        for key, (typecode, default) in SHARED_STATE_DEFAULTS.items():
            self.shared_state[key] = multiprocessing.Value(typecode, default)

    # ── Array lifecycle ───────────────────────────────────────────────

    def clear_arrays(self, *, keep_calibration: bool = False) -> None:
        """Delete all large processing arrays and run GC.

        Parameters
        ----------
        keep_calibration:
            If True, preserve ``current_L_mat``, ``light_vecs``,
            ``scale_mm``, ``_mm_per_px`` so the user does not have to
            re-calibrate after clearing.
        """
        _ARRAY_ATTRS = [
            "normals_full", "normals_proxy",
            "normals_robust", "normals_robust_proxy",
            "median_img", "median_proxy",
            "qc_resid_full", "qc_resid_proxy", "qc_sigma_full", "qc_sigma_proxy",
            "ml_fusion_full", "ml_fusion_proxy", "metal_mask_full",
            "depth_vis_p", "depth_proxy_p",
            "depth_vis_fc", "depth_proxy_fc",
            "albedo_rgb_full", "albedo_proxy", "albedo_grey_full", "median_colour", "solved_colour",
            "roughness_full", "roughness_proxy",
            "pca_full", "spec_var",
            "transmissive_raw", "transmissive_aligned",
            "ptm_coeffs", "ptm_proxy", "hsh_coeffs",
            "_svf_full", "_svf_proxy",
            "_msrm_full", "_msrm_proxy",
            "_curv_full", "_curv_proxy",
            "_slope_full", "_slope_proxy",
            "_pos_opn_full", "_neg_opn_full",
            "_loc_dom_full", "_loc_dom_proxy",
            "_slrm_full", "_slrm_proxy",
            "_cica_cache", "_rbd_cache",
            "_rbd_plate_cache", "rbf",
        ]
        for attr in _ARRAY_ATTRS:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
            setattr(self, attr, None)
        self.h = self.w = self.ph = self.pw = 0
        self.work_scale, self.source_size = 1.0, None
        self.solve_region = None
        self.ptm_deferred, self.ptm_planes, self.ptm_refit = False, None, None
        self.plate_paths.clear()
        self.transmissive_path = ""
        self.transmissive_transform = None
        if not keep_calibration:
            self.current_L_mat = None
            self.pseudo_L_mat = None
            self.light_vecs.clear()
            self.scale_mm = 0.0
            self._mm_per_px = 0.0
        gc.collect()

    def cleanup(self) -> None:
        """Full teardown: arrays, shared memory, GC."""
        self.clear_arrays(keep_calibration=False)
        # Destroy shared-memory Values
        for key in list(self.shared_state.keys()):
            try:
                del self.shared_state[key]
            except Exception:
                _log.debug("Failed to delete shared_state key")
        self.shared_state.clear()
        gc.collect()
        try:
            from helio.utils import aggressive_cleanup
            aggressive_cleanup()
        except Exception:
            _log.debug("aggressive_cleanup failed")
