"""The photometric-stereo solve, GUI-free: images + lights + settings → result (PROC-001).

One pipeline for both apps.  ``RTIViewerPro._run_processing_body`` calls
``solve`` and copies the result into its own attributes; the Qt app runs it
through ``helio.application.jobs``.  The arithmetic is the legacy body's,
moved line for line, so both apps give bit-identical normals and depth
(checked against docs/pyside6-migration/baseline/scientific-baseline.npz).

``progress(stage, fraction, detail)`` reports the legacy stages and bar
positions; ``check()`` is called at the legacy cancel points and raises
``helio.application.jobs.Cancelled`` to stop.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import threading
from concurrent.futures import as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from helio.application.jobs import Cancelled
from helio.domain import ProcessingSettings

_log = logging.getLogger(__name__)

__all__ = ["SolveRequest", "SolveResult", "CalibrationRequired", "TooFewImages", "Cancelled",
           "solve", "integrate_depth", "parabolic_baseline_correction", "proxy_dims",
           "preview_side", "set_preview_side",
           "image_pool"]


class CalibrationRequired(Exception):
    """The chosen solver needs light directions and none were given."""

    def __init__(self, engine: str):
        super().__init__(f"The '{engine}' solver needs calibrated light directions.")
        self.engine = engine


class TooFewImages(Exception):
    def __init__(self, n: int):
        super().__init__(f"Need at least 3 valid images; {n} could be read.")
        self.n = n


def image_pool(max_workers=None):
    """A pool for per-image work: threads.

    The work is ``cv2.imread`` plus NumPy, both of which release the GIL, so
    threads run it in parallel; and each decoded frame stays where it is.
    Processes (used outside the app until 2026-10-20) sent every frame back
    through a pipe: reading 32 photographs of 46 megapixels took 41.5 s that
    way and 2.1 s with threads.  Inside the app a process pool never worked
    anyway (py2app's stub launcher does not understand the child's
    ``--multiprocessing-fork`` argument).
    """
    from concurrent.futures import ThreadPoolExecutor
    return ThreadPoolExecutor(max_workers=max_workers or (os.cpu_count() or 4))


PREVIEW_SIDE = 2500                 # the preview's long side when nothing has set it
PREVIEW_MAX = 4096                  # the largest a screen may ask for
_preview_side: int | None = None


def preview_side() -> int:
    """The preview's long side: HELIO_PREVIEW_SIDE if set, else what the app set
    at start (the screen's own pixels on a machine with the memory), else PREVIEW_SIDE."""
    raw = os.environ.get("HELIO_PREVIEW_SIDE", "").strip()
    if raw:
        try:
            return max(512, min(PREVIEW_MAX, int(float(raw))))
        except ValueError:
            pass
    return _preview_side or PREVIEW_SIDE


def set_preview_side(screen_long_px: int | None, total_memory: int) -> int:
    """Fit the preview to the screen: as many pixels as its long side shows (so a
    whole-picture view is never enlarged), up to PREVIEW_MAX, on a machine with
    24 GB or more; PREVIEW_SIDE otherwise.  Called once, before any solve, because
    every proxy of a solve must share one size."""
    global _preview_side
    side = PREVIEW_SIDE
    if screen_long_px and total_memory >= 24 * 2 ** 30:
        side = max(PREVIEW_SIDE, min(PREVIEW_MAX, int(screen_long_px)))
    _preview_side = side
    return side


def proxy_dims(h: int, w: int) -> tuple[int, int]:
    """The canonical preview size (height, width): never upsampled.

    Every proxy must share one size, because the engines multiply them
    together (contracts.md §3.1)."""
    h, w = int(h or 1), int(w or 1)
    sc = min(1.0, preview_side() / max(h, w))
    return int(h * sc), int(w * sc)


def integrate_depth(normals, engine: str, *, spec_var=None, light_count_conf=None,
                    albedo_rgb=None, mask=None):
    """Depth from +Y-up normals with the named engine (the depth-method menu).

    *spec_var* and *light_count_conf* feed Confidence-Weighted Poisson;
    *albedo_rgb* guides Gradient-Smoothed Poisson.  Any other name, legacy
    aliases included, is Fast Poisson.  *mask* (helio.core.object_mask): the
    object apart from a dark surround; the depth is then integrated on it
    alone, the engine giving the fine detail, and the surround is flat.
    """
    normals = repair_normals(normals)
    if mask is not None and np.shape(mask) == np.shape(normals)[:2]:
        from helio.core.object_mask import masked_depth
        try:
            return masked_depth(normals, mask, lambda n: _integrate_whole(
                n, engine, spec_var=spec_var, light_count_conf=light_count_conf, albedo_rgb=albedo_rgb))
        except Exception:                                  # noqa: BLE001 - the whole frame, as before
            _log.warning("depth on the object alone failed; integrating the whole frame", exc_info=True)
    return _integrate_whole(normals, engine, spec_var=spec_var, light_count_conf=light_count_conf,
                            albedo_rgb=albedo_rgb)


NZ_REPAIR = 0.05      # a normal this near edge-on, or facing away, cannot be a seen surface


def repair_normals(normals):
    """*normals* with the impossible ones (facing away from the camera, edge-on,
    or not finite) filled from their good neighbours, for integration only.

    A solver that leaves out frames per pixel (shadow-rejecting,
    shine-resistant) can leave a few pixels fitted on two or three frames,
    pointing away from the camera.  One such pixel is a slope of hundreds:
    on a steel punch, 2% of them made the depth span 16,800 px where the
    surface spans 150.  Returns *normals* itself when none needs it."""
    n = np.asarray(normals, np.float32)
    if n.ndim != 3 or n.shape[2] != 3:
        return normals
    bad = ~np.isfinite(n).all(axis=2) | (np.nan_to_num(n[..., 2], nan=0.0) < NZ_REPAIR)
    if not bad.any():
        return normals
    good = (~bad).astype(np.float32)
    out = np.where(bad[..., None], 0.0, np.nan_to_num(n)).astype(np.float32)
    fill = out.copy()
    k = 2
    todo = bad.copy()
    while todo.any() and k <= 64:                     # widen until every hole has good neighbours
        num = cv2.blur(out * good[..., None], (2 * k + 1, 2 * k + 1))
        den = cv2.blur(good, (2 * k + 1, 2 * k + 1))
        ok = todo & (den > 1e-3)
        fill[ok] = num[ok] / den[ok, None]
        todo &= ~ok
        k *= 2
    fill[todo] = (0.0, 0.0, 1.0)
    out = np.where(bad[..., None], fill, out)
    out /= np.maximum(np.linalg.norm(out, axis=2, keepdims=True), 1e-8)
    weak = out[..., 2] < NZ_REPAIR
    out[weak] = (0.0, 0.0, 1.0)
    return out


def surround_mask(ps, albedo):
    """The object mask for the depth (helio.core.object_mask), or None when the
    setting is off or the object fills the frame."""
    if not getattr(ps, "mask_surround", False) or albedo is None:
        return None
    try:
        from helio.core.object_mask import object_mask
        return object_mask(albedo)
    except Exception:                                      # noqa: BLE001
        _log.warning("object mask - skipping", exc_info=True)
        return None


def _integrate_whole(normals, engine: str, *, spec_var=None, light_count_conf=None, albedo_rgb=None):
    from helio.core.depth_integration import (
        confidence_weighted_depth_integration, directional_depth_integration,
        frankot_chellappa_depth, gradient_smoothed_poisson, multiscale_depth_integration,
        poisson_solver, simchony_depth_integration)
    from helio.core.frames import normal_slopes
    if engine == "Frankot-Chellappa":
        # Projects the gradient field onto the Fourier basis, a set of
        # integrable fields.  Keeps fine incised detail best; assumes the
        # surface meets itself at opposite frame edges.
        return frankot_chellappa_depth(normals)
    if engine == "Simchony DCT":
        return simchony_depth_integration(normals)
    if engine == "Confidence-Weighted Poisson":
        # Inverse specular variance x light-count fraction: down-weights
        # specular hotspots and shadow pixels independently.
        _spec_conf = 1.0 / (1.0 + cv2.normalize(
            spec_var.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32))
        _lc_conf = light_count_conf if light_count_conf is not None else np.ones_like(_spec_conf)
        _conf = cv2.GaussianBlur(_spec_conf * _lc_conf, (0, 0), sigmaX=2.0)
        return confidence_weighted_depth_integration(normals, _conf, method="poisson")
    if engine == "Multi-Scale Integration":
        # Low-frequency shape (downsampled) plus high-frequency detail.
        return multiscale_depth_integration(normals)
    if engine == "Gradient-Smoothed Poisson":
        # Edge-preserving smoothing of the gradients, guided by the albedo.
        return gradient_smoothed_poisson(normals, albedo_guide=albedo_rgb,
                                         radius_frac=0.03, epsilon=0.05)
    if engine == "Directional Integration":
        return directional_depth_integration(normals)
    if engine == "Free-Boundary Poisson (DCT)":
        from helio.core.depth_integration import poisson_free_boundary
        return poisson_free_boundary(*normal_slopes(normals))
    if engine in ("Least-Squares (Free Boundary)", "Bilateral (Keeps Steps)"):
        from helio.core.bini import integrate
        return integrate(normals, bilateral=engine == "Bilateral (Keeps Steps)")
    return poisson_solver(*normal_slopes(normals))


def parabolic_baseline_correction(depth_map):
    """Subtract the best-fit poly22 surface (Gaiani et al. 2025 §2.2C).

    z = a x^2 + b y^2 + c xy + d x + e y + f over coordinates scaled to
    [-1, 1]; removes bowl and saddle warp, keeps micro-relief.  Returns the
    input unchanged if the fit fails.
    """
    try:
        from helio.core.depth_integration import remove_poly22
        return remove_poly22(depth_map, guard_single_row=True)
    except Exception:
        return depth_map


@dataclass
class SolveRequest:
    paths: Sequence[str]
    lights: np.ndarray | None                  # (N, 3), +Y up; None only for Uncalibrated PCA
    settings: ProcessingSettings = field(default_factory=ProcessingSettings)
    crop: tuple[int, int, int, int] | None = None     # (x1, y1, x2, y2)
    crop_ref_size: tuple[int, int] | None = None      # (w, h) the crop was drawn on
    flat_ref_normals: np.ndarray | None = None
    depth_clip_pct: float = 0.5                    # the classic slider's default
    mm_per_px: float | None = None                 # the measured scale, for the near-field dome
    frames: list | None = None                     # [(rgb, grey, is_raw)] already read (quick look)
    light_origin: tuple[float, float] | None = None   # the ball's centre (x, y), photographs' pixels
    # Each light's position (n, 3) from two balls (helio.core.light_positions), photographs'
    # pixels (x right, y down, z above the board), one row per light solved; nan where not placed.
    light_positions: np.ndarray | None = None
    # A region (crop) solved from the photographs at full size, whatever the
    # working size, when the region fits: only the region's pixels are kept
    # as each photograph is read.  Process a Region asks for it.
    full_size_region: bool = False


@dataclass
class SolveResult:
    settings: ProcessingSettings
    n_images: int
    h: int
    w: int
    base_h: int
    base_w: int
    crop: tuple[int, int, int, int] | None
    light_matrix: np.ndarray | None            # the lights solved with (PCA: its own)
    albedo_rgb_full: np.ndarray
    albedo_grey_full: np.ndarray
    normals_full: np.ndarray
    normals_were_autoflipped: bool
    relief_inverted: bool
    ptm_coeffs: np.ndarray | None              # None while ptm_deferred (see ensure_ptm_hsh)
    hsh_coeffs: np.ndarray | None
    median_img: np.ndarray
    spec_var: np.ndarray
    roughness_full: np.ndarray
    light_count_conf: np.ndarray
    depth_poisson: np.ndarray
    depth_vis_p: np.ndarray
    depth_normalised_full: np.ndarray
    depth_vis_fc: np.ndarray
    pca_full: np.ndarray
    terrain: dict[str, np.ndarray]
    normals_robust: np.ndarray | None          # None: the same as normals_full
    quality: dict[str, np.ndarray] | None      # 'resid', 'fusion'; None if it failed
    proxy: dict[str, Any]                      # every preview, keyed by its legacy name
    ph: int = 0
    pw: int = 0
    alignment_shifts: Any = None
    generation: int = 0
    flash: dict | None = None                  # CAL-003: the flash strengths evened, or None
    flat_light: dict | None = None             # CAL-004: the flash fall-off evened, or None
    work_scale: float = 1.0                    # < 1: solved from photographs reduced to this fraction
    source_size: tuple[int, int] | None = None  # (w, h) of the photographs as they are on disk
    # Tight machines do not keep the PTM/HSH coefficients (60 bytes a pixel): the
    # previews are kept, and the coefficients are fitted again from the photographs
    # when an export or a full-size render asks (ensure_ptm_hsh).
    ptm_deferred: bool = False
    ptm_planes: Any = None                     # the six PTM planes at preview size, for the live views
    refit: Any = None                          # the SolveRequest, to fit them again
    light_strength: dict | None = None         # semi-calibrated: each lamp's strength, evened; or None
    ambient: dict | None = None                # the lights-off frame subtracted: its file and level; or None
    metal: dict | None = None                  # metal-aware: the solver, the share re-solved; or None
    metal_mask: np.ndarray | None = None       # (h, w) uint8, 255 = re-solved by the metal solver
    no_ball: dict | None = None                # no ball: how the lights were found (helio.core.uncalibrated)
    lamp_distance: dict | None = None          # how far the lamps were, from the photographs, and what was used
    rbf: dict | None = None                    # helio.core.rbf_relight.build: the frames, for RBF Relight
    reflectance_rgb: np.ndarray | None = None  # helio.core.reflectance: RAW captures only
    light_field: dict | None = None            # the white-board light field applied (CAL-005): what it did
    lens: dict | None = None                   # the lens correction applied (CAL-006)
    lights_placed: int = 0                     # near-field: lights at their own positions (two balls)
    tone_curve: str = ""                       # the curve decoded by (photo_tone "srgb"): "sRGB", "Adobe RGB (1998)…"
    raw_gain: float = 1.0                      # the one gain a RAW stack was brought up by


def _crop_rbf(rbf, box, h, w):
    from helio.core.rbf_relight import crop
    return crop(rbf, box, h, w)


def crop_result(r: SolveResult, box) -> SolveResult:
    """The same solve, cut to *box* (x1, y1, x2, y2 in *r*'s own pixels):
    what Inspect's selection exports and analyses, as the classic's crop did.

    Every full-size array is sliced (a view, not a copy) and the previews
    are made again from them; *crop* records where the box lies on the
    photographs, so everything that rereads them (the light stack, the
    PTM's frames) reads the same region.  Depth keeps the whole solve's
    normalisation: a part of the object is not re-stretched to look as
    deep as the whole."""
    from dataclasses import replace
    h, w = r.h, r.w
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        raise ValueError("The selection is too small.")
    ch, cw = y2 - y1, x2 - x1

    def cut(a):
        if not isinstance(a, np.ndarray):
            return a
        if a.ndim >= 2 and a.shape[:2] == (h, w):
            return a[y1:y2, x1:x2]
        if a.ndim == 2 and a.shape[1] == h * w:            # coefficients: (k, h·w)
            return np.ascontiguousarray(a.reshape(-1, h, w)[:, y1:y2, x1:x2]).reshape(len(a), -1)
        return a
    ox, oy = (r.crop[0], r.crop[1]) if r.crop else (0, 0)
    out = replace(
        r, h=ch, w=cw, crop=(ox + x1, oy + y1, ox + x2, oy + y2),
        albedo_rgb_full=cut(r.albedo_rgb_full), albedo_grey_full=cut(r.albedo_grey_full),
        normals_full=cut(r.normals_full), ptm_coeffs=cut(r.ptm_coeffs), hsh_coeffs=cut(r.hsh_coeffs),
        median_img=cut(r.median_img), spec_var=cut(r.spec_var), roughness_full=cut(r.roughness_full),
        light_count_conf=cut(r.light_count_conf), depth_poisson=cut(r.depth_poisson),
        depth_vis_p=cut(r.depth_vis_p), depth_normalised_full=cut(r.depth_normalised_full),
        depth_vis_fc=cut(r.depth_vis_fc), pca_full=cut(r.pca_full),
        terrain={k: cut(v) for k, v in (r.terrain or {}).items()},
        normals_robust=cut(r.normals_robust),
        quality=None if r.quality is None else {k: cut(v) for k, v in r.quality.items()},
        alignment_shifts=r.alignment_shifts, metal_mask=cut(r.metal_mask),
        rbf=_crop_rbf(r.rbf, (x1, y1, x2, y2), h, w), reflectance_rgb=cut(r.reflectance_rgb))
    ph, pw = proxy_dims(ch, cw)

    def small(a):
        return None if a is None else cv2.resize(np.ascontiguousarray(a), (pw, ph))
    P = dict(r.proxy)
    P.update(normals=small(out.normals_full), albedo=small(out.albedo_rgb_full),
             roughness=small(out.roughness_full), median=small(out.median_img),
             depth_p=small(out.depth_vis_p), depth_fc=small(out.depth_vis_fc),
             reflectance=small(out.reflectance_rgb))
    P["normals_robust"] = P["normals"] if out.normals_robust is None else small(out.normals_robust)
    for k, v in (out.quality or {}).items():
        P[f"quality_{k}"] = small(v)
    for k, v in out.terrain.items():
        if k in P:
            P[k] = small(v)
    planes = out.ptm_planes
    if out.ptm_coeffs is not None:
        P["ptm"] = cv2.resize(out.ptm_coeffs.reshape(6, ch, cw).transpose(1, 2, 0), (pw, ph)
                              ).transpose(2, 0, 1).reshape(6, -1)
        if out.ptm_deferred:
            planes = _ptm_planes(out.ptm_coeffs, ch, cw, pw, ph)
    elif out.ptm_deferred and r.ptm_planes is not None:
        # Not fitted: cut the same box out of the preview planes (the box in preview pixels).
        sy, sx = r.ph / float(h), r.pw / float(w)
        a, b = int(round(y1 * sy)), max(int(round(y1 * sy)) + 1, int(round(y2 * sy)))
        c, d = int(round(x1 * sx)), max(int(round(x1 * sx)) + 1, int(round(x2 * sx)))
        planes = tuple(cv2.resize(np.ascontiguousarray(pl[a:b, c:d]), (pw, ph),
                                  interpolation=cv2.INTER_AREA) for pl in r.ptm_planes)
        P["ptm"] = np.stack([pl.reshape(-1) for pl in planes])
    return replace(out, proxy=P, ph=ph, pw=pw, ptm_planes=planes)


def _ptm_planes(ptm, h: int, w: int, pw: int, ph: int):
    """The six PTM planes area-averaged to the preview size: what the live views relight."""
    return tuple(cv2.resize(np.ascontiguousarray(ptm[k], np.float32).reshape(h, w), (pw, ph),
                            interpolation=cv2.INTER_AREA) for k in range(6))


_refit_lock = threading.Lock()


def ensure_ptm_hsh(r: SolveResult, progress: Callable | None = None, check=None):
    """The PTM and HSH coefficients of *r*, fitted again from the photographs if a
    tight machine did not keep them.  Returns (ptm, hsh); the result keeps them."""
    with _refit_lock:
        if r.ptm_coeffs is None and r.ptm_deferred and r.refit is not None:
            import dataclasses
            req = r.refit
            if r.crop is not None:          # in the pixels of the (possibly reduced) result's own base
                req = dataclasses.replace(req, crop=r.crop, crop_ref_size=(r.base_w, r.base_h))
            (progress or (lambda *a, **k: None))("Fitting PTM and HSH again", 0.0)
            try:
                ptm, hsh = solve(req, progress, check, ptm_only=True)
            except (TooFewImages, OSError) as e:
                raise RuntimeError("The photographs could not be read again to fit the PTM and HSH "
                                   "(they were not kept in memory on this machine). Have they been "
                                   "moved or renamed?") from e
            if ptm is None or ptm.shape[1] != r.h * r.w:
                raise RuntimeError("The PTM could not be fitted again from the photographs "
                                   "(have they moved or changed?)")
            r.ptm_coeffs, r.hsh_coeffs = ptm, hsh
        return r.ptm_coeffs, r.hsh_coeffs


def region_fraction(crop, base_w: int, base_h: int):
    """*crop* (x1, y1, x2, y2 in the solve's base pixels) as fractions of the
    photographs, or None for a whole-frame solve.  What anything that rereads
    the photographs from disk cuts them to, whatever size it reads them at."""
    if crop is None or not base_w or not base_h:
        return None
    x1, y1, x2, y2 = crop
    return (x1 / base_w, y1 / base_h, x2 / base_w, y2 / base_h)


def store(r: SolveResult, D) -> None:
    """Put a result where the render engines read it: a helio.data_manager.DataManager.

    The one mapping both apps use.  The classic app keeps a few more arrays
    on its window (the multi-hillshade and openness proxies); no engine
    reads those.
    """
    D.h, D.w = r.h, r.w
    D.work_scale, D.source_size = r.work_scale, r.source_size
    D.solve_region = region_fraction(r.crop, r.base_w, r.base_h)
    D.albedo_rgb_full, D.albedo_grey_full = r.albedo_rgb_full, r.albedo_grey_full
    D.current_L_mat = r.light_matrix
    D.normals_full = r.normals_full
    D.ptm_coeffs, D.hsh_coeffs = r.ptm_coeffs, r.hsh_coeffs
    D.ptm_deferred, D.ptm_planes = r.ptm_deferred, r.ptm_planes
    D.rbf = getattr(r, "rbf", None)
    D.metal_mask_full = getattr(r, "metal_mask", None)
    D.ptm_refit = (lambda progress=None: ensure_ptm_hsh(r, progress)) if r.ptm_deferred else None
    D.median_img, D.spec_var, D.roughness_full = r.median_img, r.spec_var, r.roughness_full
    D.depth_vis_p, D.depth_vis_fc, D.pca_full = r.depth_vis_p, r.depth_vis_fc, r.pca_full
    D.depth_poisson = r.depth_poisson               # the integrated depth: true heights (helio.application.heights)
    D._cica_cache = None
    D._rbd_cache = None
    t, P = r.terrain, r.proxy
    D._svf_full = t["_svf"]
    D._msrm_full, D._curv_full, D._slope_full = t["_msrm"], t["_curv"], t["_slope"]
    D._loc_dom_full, D._pos_opn_full = t["_loc_dom"], t["_pos_opn"]
    D._neg_opn_full, D._slrm_full = t["_neg_opn"], t["_slrm"]
    D.ph, D.pw = r.ph, r.pw
    D.normals_proxy = P["normals"]
    if r.normals_robust is None:
        # Uncalibrated PCA, too few lights, or a failed solve: alias rather
        # than copy (identical data, half the memory).
        D.normals_robust = D.normals_full
        D.normals_robust_proxy = D.normals_proxy
    else:
        D.normals_robust = r.normals_robust
        D.normals_robust_proxy = P["normals_robust"]
    if r.quality is not None:
        for src, full, prox in (('resid', 'qc_resid_full', 'qc_resid_proxy'),
                                ('fusion', 'ml_fusion_full', 'ml_fusion_proxy')):
            setattr(D, full, r.quality.get(src))
            setattr(D, prox, P.get(f"quality_{src}"))
        # The normal's angular uncertainty, degrees (FIX-011).
        D.qc_sigma_full = r.quality.get("sigma_deg")
        D.qc_sigma_proxy = P.get("quality_sigma_deg")
    D.albedo_proxy, D.roughness_proxy = P["albedo"], P["roughness"]
    D.median_colour = (D.albedo_rgb_full, D.albedo_proxy)
    D.solved_colour = None if r.reflectance_rgb is None else (r.reflectance_rgb, P.get("reflectance"))
    D.median_proxy = P["median"]
    D.depth_proxy_p, D.depth_proxy_fc = P["depth_p"], P["depth_fc"]
    D.ptm_proxy = P["ptm"]
    D._svf_proxy, D._msrm_proxy = P["_svf"], P["_msrm"]
    D._curv_proxy = P["_curv"]
    D._slope_proxy, D._loc_dom_proxy = P["_slope"], P["_loc_dom"]
    D._slrm_proxy = P["_slrm"]


def _load(paths, progress, scale: float = 1.0):
    """[(rgb, grey, is_raw)] per path, in order (rgb None if unreadable).

    With *scale* < 1 each photograph is reduced (area-averaged) as soon as it is
    read, in its own worker, so the full-size pictures never pile up: that is
    the point of a reduced working size on a machine short of memory."""
    return _load_scaled(paths, progress, scale)[0]


def _reduce(a, scale: float):
    if scale >= 1.0 or a is None:
        return a
    h, w = a.shape[:2]
    return cv2.resize(a, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def _load_scaled(paths, progress, scale: float = 1.0, box=None):
    """(frames, (w, h) of the first readable photograph before reduction).

    *box* (x1, y1, x2, y2), the photographs' pixels: keep only that region of
    each, cut as soon as it is read, so the full-size frames never pile up."""
    from helio.io import load_and_process_single
    from helio.io.raw_loader import is_raw
    n = len(paths)
    res_map, sizes = {}, {}

    def cut(a):
        if a is None or box is None:
            return a
        x1, y1, x2, y2 = box
        return np.ascontiguousarray(a[y1:y2, x1:x2])

    def one(i, p):
        rgb, grey = load_and_process_single(p)
        if rgb is not None:
            sizes[i] = (rgb.shape[1], rgb.shape[0])
        return _reduce(cut(rgb), scale), _reduce(cut(grey), scale)
    with image_pool() as ex:
        futures = {ex.submit(one, i, p): i for i, p in enumerate(paths)}
        for fi, fut in enumerate(as_completed(futures)):
            res_map[futures[fut]] = fut.result()
            done = fi + 1
            progress(None, done / n * 0.18, f"Loading images  {done}/{n}")
    frames = [(*res_map[i], is_raw(paths[i])) for i in range(n)]
    return frames, (sizes[min(sizes)] if sizes else None)


def _reflectance(ps, req, raw_frames, stack, L, normals, median_rgb):
    """The solved reflectance, for a capture all in RAW whose stack is still light:
    what evens, flattens or equalises the frames for the solve spoils it as a
    measurement.  None otherwise, and never fails a solve."""
    if (req.frames is not None or L is None or len(L) != len(stack) or not raw_frames or not all(raw_frames)
            or ps.flatten_illumination or ps.multi_norm or ps.pre_clahe or ps.adaptive_light > 0.0):
        return None
    try:
        from helio.core.reflectance import solved_reflectance
        return solved_reflectance(stack, L, normals, median_rgb)
    except Exception:
        _log.warning("solved reflectance - skipping", exc_info=True)
        return None


RAW_TOP = 180.0          # where the brightest of a RAW stack is put, 0-255


def _raw_exposure(grey_stack, rgb_list, raw_frames):
    """A RAW capture brought up to a working level, with one gain for the whole stack.

    Decoded linearly a raw frame sits far down the scale (Capture One DNGs of a page: median
    0.2 of 255, the brightest 1 % about 45), where a camera or Capture One would have brightened
    it.  Every threshold in the solve and its quality maps is on a 0-255 scale, so most of the
    surface read as too dark and the colour map came out black.  The same factor on every frame
    keeps the ratios between frames, which are all the solve reads, so the measurement stands;
    per-frame brightening is what raw_loader forbids.  (grey_stack, rgb_list, gain)."""
    if not raw_frames or not all(raw_frames):
        return grey_stack, rgb_list, 1.0
    top = max(float(np.percentile(g[::8, ::8], 99.5)) for g in grey_stack)
    k = float(np.clip(RAW_TOP / max(top, 1e-3), 1.0, 64.0))
    if k < 1.05:
        return grey_stack, rgb_list, 1.0
    from helio.core.radiometry import linear_to_srgb, srgb_to_linear
    lut = np.round(linear_to_srgb(srgb_to_linear(np.arange(256), 255.0) * k, 255.0)).astype(np.uint8)
    grey_stack = np.asarray(grey_stack, np.float32) * np.float32(k)
    return grey_stack, [cv2.LUT(r, lut) for r in rgb_list], k


def use_solved_colour(D, on: bool) -> bool:
    """Show the solved reflectance as the colour (*on*) or the median of the frames: views,
    3D textures and exports all read the colour from the same place.  False if there is none."""
    chosen = getattr(D, "solved_colour", None) if on else getattr(D, "median_colour", None)
    if chosen is None or chosen[0] is None:
        return False
    D.albedo_rgb_full, D.albedo_proxy = chosen
    return True


def white_balance_base(r) -> dict:
    """The result's colour as solved, before any white balance: kept on the result
    the first time a white balance is applied, so changing or resetting it never
    compounds."""
    base = getattr(r, "_wb_base", None)
    if base is None:
        P = r.proxy or {}
        base = dict(albedo=r.albedo_rgb_full, albedo_p=P.get("albedo"),
                    refl=r.reflectance_rgb, refl_p=P.get("reflectance"))
        r._wb_base = base
    return base


def apply_white_balance(r, gains, D=None) -> None:
    """Give result *r* (and the DataManager *D* made from it, if any) the colour
    white-balanced by *gains* (helio.core.white_balance), or back as solved when
    *gains* is None.  Views, 3D textures and exports all read the colour from
    r.albedo_rgb_full / D.albedo_rgb_full, so this is the one place it changes;
    the median colour and a RAW capture's solved reflectance both, whichever is
    shown staying shown."""
    from helio.core import white_balance as WB
    base = white_balance_base(r)
    on = not WB.neutral(gains)
    r.albedo_rgb_full = WB.apply(base["albedo"], gains) if on else base["albedo"]
    r.reflectance_rgb = WB.apply(base["refl"], gains) if on else base["refl"]
    if r.proxy is not None:
        r.proxy["albedo"] = WB.apply(base["albedo_p"], gains) if on else base["albedo_p"]
        if "reflectance" in r.proxy or base["refl_p"] is not None:
            r.proxy["reflectance"] = WB.apply(base["refl_p"], gains) if on else base["refl_p"]
    r.white_balance = tuple(float(v) for v in gains) if on else None
    if D is None:
        return
    solved = getattr(D, "solved_colour", None)
    solved_shown = bool(solved) and D.albedo_rgb_full is solved[0]
    P = r.proxy or {}
    D.median_colour = (r.albedo_rgb_full, P.get("albedo"))
    D.solved_colour = None if r.reflectance_rgb is None else (r.reflectance_rgb, P.get("reflectance"))
    use_solved_colour(D, solved_shown)


def effective_work_scale(ps, full_size=None, n_photos: int = 0) -> float:
    """The fraction the photographs are reduced to: HELIO_WORK_SCALE if set, else
    the setting; 1.0 (full size) unless it is between 0.1 and 0.995.  0 (or
    HELIO_WORK_SCALE=auto) is automatic: the largest size that fits this machine's
    free memory, which needs the photographs' size *full_size* (w, h) and their count."""
    import os
    raw = os.environ.get("HELIO_WORK_SCALE", "").strip()
    try:
        s = 0.0 if raw.lower() == "auto" else float(raw) if raw else float(getattr(ps, "work_scale", 1.0))
    except ValueError:
        s = 1.0
    if s == 0.0:
        if full_size is None or n_photos < 1:
            return 1.0
        from helio.utils.memory_plan import auto_scale
        s = auto_scale(full_size[1], full_size[0], n_photos, ps)
    return s if 0.1 <= s < 0.995 else 1.0


def reduced_request(req, scale: float, full_size, base_size):
    """*req* for photographs reduced to *base_size* from *full_size* (w, h): every
    pixel-unit input converted, as quick_look.small_request does for a quick look."""
    from helio.application import quick_look as ql
    s = base_size[0] / float(full_size[0])
    ps = ql.scaled_settings(req.settings, s)
    flat = req.flat_ref_normals
    if flat is not None:
        fh, fw = flat.shape[:2]
        flat = cv2.resize(np.asarray(flat, np.float32), (max(1, round(fw * s)), max(1, round(fh * s))),
                          interpolation=cv2.INTER_AREA)
    import dataclasses
    return dataclasses.replace(
        req, settings=ps, flat_ref_normals=flat,
        # the crop stays in the pixels it was drawn on: solve() scales it by base / reference size
        crop_ref_size=req.crop_ref_size or (full_size if req.crop is not None else None),
        light_origin=None if req.light_origin is None else tuple(v * s for v in req.light_origin),
        light_positions=None if req.light_positions is None else np.asarray(req.light_positions) * s,
        mm_per_px=(req.mm_per_px / s) if req.mm_per_px else None)


def result_mm_per_px(r, mm_per_px_full):
    """The scale of result *r*'s pixels, from the scale of the photographs'
    pixels: a result solved at a reduced working size has bigger pixels."""
    if not mm_per_px_full:
        return mm_per_px_full
    return mm_per_px_full / r.work_scale if getattr(r, "work_scale", 1.0) not in (0, None) else mm_per_px_full


def near_field_radius_px(ps, mm_per_px, h: int, w: int, found=None):
    """The dome radius in this image's pixels, and where it came from.

    *found* is helio.core.lamp_distance.estimate's answer for these frames.
    With Lamp distance "From the photographs" (the default) a sure estimate
    is used ("photographs"), and lamps the photographs show to be far away
    are taken as distant ("far": the correction then changes nothing).
    Otherwise, or with "From the dome radius", the dome radius the user gave
    in millimetres over the measured scale ("measured"); with no scale, the
    classic guess, 1.5 times the image's longer side ("assumed")."""
    if getattr(ps, "near_field_distance", "photos") != "radius" and found:
        if found.get("sure") and found.get("radius_px"):
            return float(found["radius_px"]), "photographs"
        if found.get("far_away"):
            return max(h, w) * 1000.0, "far"
    if (ps.near_field or ps.normal_engine == "Near-Field Attenuation") and mm_per_px:
        return float(ps.dome_radius_mm) / float(mm_per_px), "measured"
    return max(h, w) * 1.5, "assumed"


def near_field_positions(light_positions, crop, h: int, w: int):
    """Each light's position (helio.core.light_positions, photographs' pixels, y down) in
    the processed image's frame: pixels from its centre, +Y up, z above the board."""
    if light_positions is None:
        return None
    P = np.array(light_positions, np.float64, copy=True)
    if crop is not None:
        P[:, 0] -= crop[0]
        P[:, 1] -= crop[1]
    return np.column_stack([P[:, 0] - w / 2.0, h / 2.0 - P[:, 1], P[:, 2]])


def near_field_origin(ps, light_origin, crop, h: int, w: int):
    """Where the near-field dome is centred, in pixels from the processed
    image's centre (+Y up): the ball, when the lights were measured on one
    and Lamps measured from is the ball; otherwise the picture's centre."""
    if ps.near_field_from != "ball" or light_origin is None:
        return (0.0, 0.0)
    bx, by = (float(v) for v in light_origin)
    if crop is not None:
        bx, by = bx - crop[0], by - crop[1]
    return (bx - w / 2.0, h / 2.0 - by)


def _undistort_frames(res, path, full_size, work_scale, load_box, progress):
    """Every loaded frame (rgb, grey, raw) with the lens's distortion taken out
    (helio.core.lens).  The camera matrix follows the working size and a
    region's offset.  A calibration for another camera or size is refused."""
    from helio.core import lens
    cal = lens.load(path)
    why = lens.check_size(cal, full_size)
    if why:
        raise ValueError(f"Lens correction: {why} Calibrate this camera's lens, or turn the "
                         f"correction off in Process.")
    off = (load_box[0], load_box[1]) if load_box is not None else (0, 0)
    progress(None, 0.2, "Taking out the lens's distortion")
    out = [(None if r is None else lens.undistort(r, cal, work_scale, off),
            None if g is None else lens.undistort(g, cal, work_scale, off), raw) for r, g, raw in res]
    return out, {"file": os.path.basename(path), "edge_px": round(cal["edge_px"], 2),
                 "rms_px": round(cal["rms"], 3)}


def _apply_light_field(ps, lights, grey_stack, raw_frames, base_size, crop, progress):
    """The grey stack divided by each light's white-board field (helio.core.light_field):
    fall-off and strengths evened in one step.  In the solve's intensity units: a RAW
    frame is linear; a decoded photograph is evened before decoding, as the flat areas
    are (factor ** 1/2.2); stored values are taken as the solve takes them."""
    from helio.core import light_field as LF
    field = LF.load(ps.light_field)
    why = LF.check_lights(field, lights)
    if why:
        raise ValueError(f"White-board light field: {why} Shoot the board again, or turn the "
                         f"light field off in Process.")
    progress(None, 0.22, "Evening the lights with the white-board calibration")
    raw = np.zeros(len(grey_stack), bool) if raw_frames is None else np.asarray(raw_frames, bool)
    decoded = ps.photo_tone == "srgb"
    out = np.empty(np.shape(grey_stack), np.float32 if grey_stack.dtype != np.uint8 else np.uint8)
    # Every frame scaled to the weakest point of the weakest light, never brighter, so
    # nothing clips; one factor for all frames changes no normal.
    top = float(np.max(1.0 / np.maximum(np.asarray(field["maps"]), 1e-3)))
    for i in range(len(grey_stack)):
        f = LF.factor(field, i, base_size, crop) / np.float32(top)
        if not raw[i] and decoded:
            f = np.power(f, np.float32(1 / 2.2))
        g = grey_stack[i].astype(np.float32) * f
        out[i] = np.clip(np.rint(g), 0, 255).astype(np.uint8) if out.dtype == np.uint8 else g
    if ps.near_field:
        _log.warning("light field: the near-field model is on as well; the board has measured "
                     "the fall-off already, so the two together over-correct")
    return out, {"file": os.path.basename(ps.light_field), "spread": round(field["strength_spread"], 3),
                 "falloff": round(field["falloff"], 3)}


def _solve_normals(engine, flat_grey, grey_stack, L, h, w, ps, flat_ref=None, mm_per_px=None,
                   origin=None, positions=None, equal_power=True, found=None, falloff=True):
    """*flat_grey*/*grey_stack* are the intensities solved (linear light,
    FIX-010); *flat_ref* the values as stored, which the robust solvers'
    weights and thresholds are defined on."""
    from helio.core.photometric_stereo import (
        intensity_weighted_ls, irls_photometric_stereo, iterative_specular_separation_ps,
        shadow_rejection_ls, solve_near_field_normals)
    if engine == "Uncalibrated PCA":
        from helio.core.uncalibrated import solve_no_ball
        L, normals, _info = solve_no_ball(flat_grey, h, w)
        return normals.reshape(h, w, 3), L
    if engine == "Near-Field Attenuation" or (ps.near_field and L is not None):
        radius, _src = near_field_radius_px(ps, mm_per_px, h, w, found)
        return solve_near_field_normals(grey_stack, L, h, w, dome_radius_px=radius,
                                        origin=origin or (0.0, 0.0), positions=positions,
                                        equal_power=equal_power, falloff=falloff), L
    if engine == "Sparse-Outlier LS (RPCA)":
        from helio.core.robust_ps import sparse_outlier_normals
        return sparse_outlier_normals(L, flat_grey, h, w, lam=ps.rpca_lambda), L
    if engine == "Shadow-Rejection LS":
        return shadow_rejection_ls(L, flat_grey, h, w, ps.shadow_thresh, I_ref=flat_ref), L
    if engine == "Mid-Tone Weighted LS":
        return intensity_weighted_ls(L, flat_grey, h, w, I_ref=flat_ref), L
    if engine == "IRLS (Metallic/Specular)":
        return irls_photometric_stereo(L, flat_grey.astype(np.float32), h, w,
                                       n_iters=5, sigma=40.0, I_ref=flat_ref), L
    if engine == "Iterative Specular Separation":
        return iterative_specular_separation_ps(L, flat_grey.astype(np.float32), h, w,
                                                n_iters=3), L
    from helio.core.photometric_stereo import linear_least_squares_normals
    return linear_least_squares_normals(L, flat_grey).reshape(h, w, 3), L   # Rust if built, else strips


def _lamp_distance(ps, stack, L, origin, positions, h, w, mm_per_px, raw_frames=None, curve=None):
    """How far the lamps were, read from the frames (helio.core.lamp_distance),
    whether or not the near-field correction is on: with it on, a sure answer
    sets its distance; with it off, Process can say when the lamps were close
    enough to bend a flat object.  None with no lights, or lights placed from
    two balls (their positions are measured already).  Never stops a solve."""
    if L is None or positions is not None or len(L) < 5:
        return None
    try:
        import cv2
        from helio.core.lamp_distance import WORK_SIDE, estimate
        from helio.core.radiometry import solve_intensities
        # a reduced copy, read as the solve reads light: frame by frame, so a
        # 100 MP capture is never decoded whole a second time
        k = min(1.0, 2.0 * WORK_SIDE / float(max(h, w)))
        size = (max(8, int(round(w * k))), max(8, int(round(h * k))))
        small = np.stack([cv2.resize(np.asarray(f, np.float32), size, interpolation=cv2.INTER_AREA)
                          for f in stack])
        small = np.asarray(solve_intensities(small, ps.photo_tone, raw_frames, curve), np.float32)
        found = estimate(small, L, origin=(origin[0] * k, origin[1] * k))
        for key in ("radius_px",):
            if found.get(key):
                found[key] = found[key] / k
    except Exception:
        _log.debug("lamp distance: not read", exc_info=True)
        return None
    near = ps.near_field or ps.normal_engine == "Near-Field Attenuation"
    radius, src = near_field_radius_px(ps, mm_per_px, h, w, found)
    rec = dict(ratio=found["ratio"], sure=found["sure"], far_away=found["far_away"],
               gain=found["gain"], used=bool(near), source=src if near else "",
               radius_px=radius if near else None)
    _log.info("lamp distance: " + (f"about {found['ratio']:.1f} x the frame's longer side"
                                   if found["ratio"] else "far" if found["far_away"] else "unclear")
              + (f" ({'sure' if found['sure'] else 'a guess'})" if found["ratio"] else "")
              + (f"; near-field radius {radius:.0f} px ({src})" if near else ""))
    return rec


def _subtract_ambient(ps, grey_stack, rgb_list, raw_frames, *, work_scale, load_box, region_hw, crop,
                      progress, curve=None):
    """The lights-off frame (ps.ambient_frame) read as the lit frames were - the same
    working size, region and crop - and subtracted from each in linear light.

    Returns (grey_stack, rgb_list, level): *level* is the ambient light as a fraction
    of the lit frames' (0.08: 8 %), or None when the frame could not be read, in
    which case the frames are returned as they were and the solve goes on."""
    from helio.core.radiometry import srgb_to_linear, subtract_ambient
    path = ps.ambient_frame
    if not os.path.exists(path):
        _log.warning("ambient frame %s not found: not subtracted", path)
        progress(None, 0.2, f"The lights-off frame {os.path.basename(path)} was not found: "
                            "solving without it")
        return grey_stack, rgb_list, None
    (res, _size) = _load_scaled([path], lambda *a, **k: None, work_scale, load_box)
    rgb_a, grey_a, raw_a = res[0] if res else (None, None, False)
    if grey_a is None:
        _log.warning("ambient frame %s could not be read: not subtracted", path)
        return grey_stack, rgb_list, None
    h, w = region_hw
    if grey_a.shape[:2] != (h, w):
        grey_a = cv2.resize(grey_a, (w, h), interpolation=cv2.INTER_AREA)
        rgb_a = cv2.resize(rgb_a, (w, h), interpolation=cv2.INTER_AREA)
    if load_box is None and crop is not None:
        x1, y1, x2, y2 = crop
        grey_a, rgb_a = grey_a[y1:y2, x1:x2], rgb_a[y1:y2, x1:x2]
    raw = [bool(r) for r in raw_frames] + [False] * (len(grey_stack) - len(raw_frames))
    # The grey is the file's own numbers (its curve); the colour has been brought to sRGB.
    dec = srgb_to_linear if curve is None else curve.to_linear
    lit = float(np.mean([dec(g).mean() if not r else float(np.mean(g))
                         for g, r in zip(grey_stack, raw)]))
    dark = float(np.mean(grey_a if raw_a else dec(grey_a)))
    level = dark / lit if lit > 0 else 0.0
    out = np.empty_like(grey_stack)
    for i in range(len(grey_stack)):
        out[i] = subtract_ambient(grey_stack[i], grey_a, frame_raw=raw[i], ambient_raw=bool(raw_a),
                                  curve=curve)
    rgb_list = [subtract_ambient(c, rgb_a, frame_raw=raw[i], ambient_raw=bool(raw_a))
                for i, c in enumerate(rgb_list)]
    progress(None, 0.2, f"Lights-off frame subtracted: the ambient light was {level * 100:.0f}% of "
                        "the lit frames'")
    return out, rgb_list, level


def _even_flash(ps, lights, grey_stack, rgb_list, raw_frames, crop, frame_size=(1, 1), curve=None):
    """Hand-held flash (CAL-003): scale each frame to the weakest flash's
    strength, measured on the grey reference (helio.core.flash).  The
    strengths are read as the solve reads light (photo_tone); the stored
    values are scaled to match.  Returns None, with a log line, when the
    reference cannot be read."""
    from helio.core.flash import parse_box, strengths
    from helio.core.radiometry import solve_intensities
    box = parse_box(ps.flash_ref)
    if box is None:
        return None
    if crop is not None:
        box = (box[0] - crop[0], box[1] - crop[1], box[2] - crop[0], box[3] - crop[1])
    x1, y1, x2, y2 = box
    n, h, w = grey_stack.shape[:3]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
        _log.warning("flash strength: the grey reference is outside the processed region")
        return None
    sub = solve_intensities(grey_stack[:, y1:y2, x1:x2], ps.photo_tone, raw_frames, curve)
    r = strengths(sub, (0, 0, x2 - x1, y2 - y1), lights)
    if r is None:
        _log.warning("flash strength: the grey reference could not be read (clipped or dark)")
        return None
    raw = np.zeros(n, bool) if raw_frames is None else np.asarray(raw_frames, bool)
    decoded = ps.photo_tone == "srgb"
    # Stored values: an sRGB-decoded solve scales linear light, so the stored
    # value scales by about factor^(1/2.2); otherwise exactly by the factor.
    f_stored = np.array([f if (raw[i] or not decoded) else f ** (1 / 2.2)
                         for i, f in enumerate(r["factor"])])
    dt = grey_stack.dtype
    gs = np.clip(grey_stack.astype(np.float32) * f_stored[:, None, None].astype(np.float32),
                 0, 255 if dt == np.uint8 else np.inf).astype(dt)
    rl = [np.clip(im.astype(np.float32) * float(f_stored[i]), 0,
                  255 if im.dtype == np.uint8 else np.inf).astype(im.dtype)
          for i, im in enumerate(rgb_list)]
    _log.info(f"flash strength evened: spread x{r['spread']:.2f} over {r['used']} frames")
    from helio.core.flash import coverage
    return dict(grey_stack=gs, rgb_list=rl, box=parse_box(ps.flash_ref),
                strength=[float(v) for v in r["strength"]], spread=r["spread"], used=r["used"],
                coverage=coverage([parse_box(ps.flash_ref)], frame_size))


def _even_flat_areas(ps, lights, grey_stack, rgb_list, raw_frames, crop, frame_size, curve=None,
                     near=None):
    """The flash's fall-off across the frame (CAL-004), measured on the
    areas marked flat and divided out of every frame: grey and colour, as
    the solve reads them (photo_tone) and as stored.  None, with a log line,
    when the areas cannot be read."""
    from helio.core.flash import (area_disagreement, coverage, flat_area_factor, flat_area_planes,
                                  parse_boxes)
    from helio.core.radiometry import solve_intensities
    boxes = parse_boxes(ps.flat_areas)
    if not boxes:
        return None
    offset = (crop[0], crop[1]) if crop is not None else (0, 0)
    n, h, w = grey_stack.shape[:3]
    lin = solve_intensities(grey_stack, ps.photo_tone, raw_frames, curve)
    model = None
    if near is not None:                      # (radius px, origin): the near-field solve's lamps
        model = _flat_light_model(lights, near[0], near[1], h, w)
    planes = flat_area_planes(lin, boxes, lights, frame_size, offset, model=model)
    disagree = area_disagreement(lin, boxes, lights, offset)
    del lin
    if planes is None:
        _log.warning("flat areas: they could not be read (clipped, dark or outside the frame)")
        return None
    raw = np.zeros(n, bool) if raw_frames is None else np.asarray(raw_frames, bool)
    decoded = ps.photo_tone == "srgb"
    gs = np.empty_like(grey_stack)
    rl = []
    for i in range(n):
        f = flat_area_factor(planes, i, h, w, frame_size, offset)
        fs = f if (raw[i] or not decoded) else np.power(f, np.float32(1 / 2.2))
        hi = 255 if grey_stack.dtype == np.uint8 else np.inf
        gs[i] = np.clip(grey_stack[i].astype(np.float32) * fs, 0, hi).astype(grey_stack.dtype)
        im = rgb_list[i]
        hi = 255 if im.dtype == np.uint8 else np.inf
        rl.append(np.clip(im.astype(np.float32) * fs[..., None], 0, hi).astype(im.dtype))
    _log.info(f"flat areas: fall-off up to {planes['falloff'] * 100:.0f}% evened over "
              f"{planes['used']} frames ({', '.join(planes['axes'])})")
    return dict(grey_stack=gs, rgb_list=rl, boxes=boxes, falloff=planes["falloff"],
                spread=planes["spread"], used=planes["used"], axes=list(planes["axes"]),
                coef=planes["coef"].tolist(), disagree=disagree,
                coverage=coverage(boxes, frame_size))


def _flat_light_model(lights, radius, origin, h: int, w: int):
    """The light each lamp gives a flat, facing surface at stack pixels
    (xs, ys), as the near-field solve models it (photometric_stereo.
    solve_near_field_normals: lamps *radius* px from *origin*, 1 there)."""
    S = np.asarray(lights, np.float64) * float(radius)
    ox, oy = origin

    def model(xs, ys):
        X = np.stack([(np.asarray(xs) - (w - 1) / 2.0) - ox, ((h - 1) / 2.0 - np.asarray(ys)) - oy,
                      np.zeros(len(xs))], 1)
        Lp = S[:, None, :] - X[None, :, :]
        d2 = np.einsum("kmi,kmi->km", Lp, Lp)
        return Lp[..., 2] * float(radius) ** 2 / (d2 * np.sqrt(d2))     # n = +Z: the z part
    return model


def region_box(crop, ref_size, full_size):
    """*crop* (drawn on an image *ref_size* (w, h), or on the photographs when
    None) in the photographs' pixels, *full_size* (w, h), inside the frame."""
    x1, y1, x2, y2 = (float(v) for v in crop)
    fw, fh = full_size
    rw, rh = ref_size or full_size
    sx, sy = fw / float(rw or fw), fh / float(rh or fh)
    x1, x2 = sorted((int(round(x1 * sx)), int(round(x2 * sx))))
    y1, y2 = sorted((int(round(y1 * sy)), int(round(y2 * sy))))
    return max(0, x1), max(0, y1), min(fw, x2), min(fh, y2)


def _even_strengths(ps, lights, grey_stack, rgb_list, raw_frames, curve=None):
    """Semi-calibrated (helio.core.robust_ps): each lamp's strength from the
    photographs, with the directions held, and every frame scaled to the
    geometric mean of them.  The strengths are read as the solve reads light
    (photo_tone) on an even sample of pixels; the stored values are scaled to
    match, as the grey patch's correction does.  None, with a log line, when
    there is not enough to go on."""
    from helio.core.radiometry import solve_intensities
    from helio.core.robust_ps import STRENGTH_SAMPLE, _sample_columns, light_strengths
    n = len(grey_stack)
    flat = grey_stack.reshape(n, -1)
    cols = _sample_columns(flat.shape[1], STRENGTH_SAMPLE)
    sub = solve_intensities(np.ascontiguousarray(flat[:, cols]), ps.photo_tone, raw_frames, curve)
    r = light_strengths(sub, lights)
    if r is None:
        _log.warning("semi-calibrated: the lamp strengths could not be found (too few lamps, or "
                     "the frames are dark or clipped)")
        return None
    raw = np.zeros(n, bool) if raw_frames is None else np.asarray(raw_frames, bool)
    decoded = ps.photo_tone == "srgb"
    f = [1.0 / v for v in r["strength"]]
    f_stored = [v if (raw[i] or not decoded) else v ** (1 / 2.2) for i, v in enumerate(f)]
    gs = np.empty_like(grey_stack)
    hi = 255 if grey_stack.dtype == np.uint8 else np.inf
    for i in range(n):                                   # a frame at a time: no float copy of the stack
        gs[i] = np.clip(grey_stack[i].astype(np.float32) * np.float32(f_stored[i]), 0, hi)
    rl = [np.clip(im.astype(np.float32) * np.float32(f_stored[i]), 0,
                  255 if im.dtype == np.uint8 else np.inf).astype(im.dtype)
          for i, im in enumerate(rgb_list)]
    _log.info(f"semi-calibrated: lamp strengths spread x{r['spread']:.2f} over {n} frames, "
              f"{r['used']} pixels, {r['iters']} rounds")
    return dict(grey_stack=gs, rgb_list=rl, **r)


def _metal_aware(ps, engine, normals, flat_solve, flat_ref, L, h, w):
    """Where the surface shines, the normals solved again with ps.metal_solver
    and blended in (helio.core.robust_ps).  Returns (normals, mask uint8 or
    None, record or None)."""
    from helio.core import robust_ps as R
    solver = ps.metal_solver
    if L is None or engine == "Uncalibrated PCA":
        _log.info("metal-aware: needs calibrated lights; skipped")
        return normals, None, None
    if solver == engine:
        _log.info(f"metal-aware: the whole surface is already solved with {engine}; skipped")
        return normals, None, None
    excess = R.specular_excess(L, normals, flat_solve)
    fl, bright = R.flare(flat_solve, L, h, w)
    mask = R.metal_mask(excess, ps.metal_sensitivity, flare_map=fl, bright=bright)
    del excess, fl, bright
    idx = np.flatnonzero(mask.reshape(-1) > 0)
    rec = dict(solver=solver, sensitivity=float(ps.metal_sensitivity),
               share=float(idx.size) / float(h * w), pixels=int(idx.size))
    if idx.size == 0:
        _log.info("metal-aware: nothing shines enough to re-solve")
        return normals, np.zeros((h, w), np.uint8), rec
    sub = np.ascontiguousarray(flat_solve[:, idx])
    ref = None if flat_ref is None else np.ascontiguousarray(flat_ref[:, idx])
    k = idx.size
    if solver == "Sparse-Outlier LS (RPCA)":
        other = R.sparse_outlier_normals(L, sub, k, 1, lam=ps.rpca_lambda)
    else:
        import dataclasses               # the re-solve is per pixel: no near-field geometry for it
        other, _ = _solve_normals(solver, sub, sub.reshape(len(sub), k, 1), L, k, 1,
                                  dataclasses.replace(ps, near_field=False), flat_ref=ref)
    flat_n = normals.reshape(-1, 3)
    part = flat_n[idx][:, None, :].copy()
    R.blend_normals(part, np.asarray(other, np.float32).reshape(k, 1, 3), mask.reshape(-1)[idx][:, None])
    flat_n[idx] = part[:, 0]
    _log.info(f"metal-aware: {rec['share'] * 100:.1f}% of the surface re-solved with {solver}")
    return normals, (mask * 255).astype(np.uint8), rec


def solve(req: SolveRequest, progress: Callable | None = None,
          check: Callable[[], None] | None = None, ptm_only: bool = False):
    """Run the whole solve.  Raises TooFewImages, CalibrationRequired or Cancelled."""
    progress = progress or (lambda *a, **k: None)
    check = check or (lambda: None)
    ps = req.settings
    if ps.multi_norm:                               # retired: see ProcessingSettings.multi_norm
        import dataclasses
        _log.info("frame-brightness levelling is retired: not applied")
        ps = dataclasses.replace(ps, multi_norm=False)
        req = dataclasses.replace(req, settings=ps)
    auto_tone = ps.photo_tone == "auto"
    if auto_tone:                                   # made definite: the result records what was used
        import dataclasses
        ps = dataclasses.replace(ps, photo_tone=ps.resolved_photo_tone())
        req = dataclasses.replace(req, settings=ps)
    from helio.core import parallel
    from helio.utils import stage_profile
    prof = stage_profile.StageProfile() if stage_profile.enabled() else None
    if prof is not None:
        progress = prof.wrap(progress)
    saved_strip = parallel.STRIP_BYTES
    try:
        return _solve(req, progress, check, ps, prof, ptm_only, auto_tone)
    finally:
        parallel.STRIP_BYTES = saved_strip
        if prof is not None:
            prof.emit()


def _photo_size(paths):
    """(w, h) of the first readable photograph, from its header - no pixels read."""
    from helio.application.quick_look import _size
    for p in paths:
        sz = _size(p)
        if sz:
            return sz
    return None


def _announce(plan, progress):
    if plan.message:
        _log.warning(plan.message)
        progress(None, 0.0, plan.message)


def _solve(req, progress, check, ps, prof, ptm_only=False, auto_tone=False):
    from helio.utils.memory_plan import describe_machine, plan_solve
    from helio.core.processing import (
        adaptive_light_weighting, apply_clahe_stack, apply_flat_correction_to_normals,
        auto_orient_normals, compute_depth_visualizations, compute_light_count_map,
        compute_ptm_hsh_coefficients, compute_terrain_analysis, flatten_illumination_greys,
        phase_align_stack,
        resize_to_common_shape, stack_quality_maps, trimmed_photometric_stereo)
    from helio.core.photometric_stereo import sfs_refinement

    work_scale, full_size, load_box = 1.0, None, None
    if req.frames is not None:
        res = list(req.frames)
    else:
        full_size = _photo_size(req.paths)
        work_scale = effective_work_scale(ps, full_size, len(req.paths))
        if req.full_size_region and req.crop is not None and full_size is not None:
            box = region_box(req.crop, req.crop_ref_size, full_size)
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bw >= 8 and bh >= 8 and plan_solve(bh, bw, len(req.paths), ps).level != "too_big":
                load_box, work_scale = box, 1.0
                progress(None, 0.0, f"Solving the region at full size: {bw} x {bh} pixels of the "
                                    f"photographs")
            else:
                _log.warning("region: too big to solve at full size on this machine; "
                             "solved at the working size")
        if full_size is not None:                         # the plan, before anything big is read
            early = (plan_solve(load_box[3] - load_box[1], load_box[2] - load_box[0], len(req.paths), ps)
                     if load_box is not None else
                     plan_solve(full_size[1] * work_scale, full_size[0] * work_scale, len(req.paths), ps))
            _announce(early, progress)
        res, read_size = _load_scaled(list(req.paths), progress, work_scale, load_box)
        full_size = read_size or full_size
    lens_note = None
    if ps.lens_calibration and full_size is not None:
        res, lens_note = _undistort_frames(res, ps.lens_calibration, full_size, work_scale, load_box,
                                           progress)
    rgbs, greys, raw_frames = [], [], []
    for r, g, raw in res:
        if r is not None:
            rgbs.append(r)
            greys.append(g)
            raw_frames.append(raw)
    del res
    if len(greys) < 3:
        raise TooFewImages(len(greys))

    # One grey stack: it was also kept as a list of frames and copied from
    # that into a second array, 5.9 GB more on a 32-frame, 46-megapixel capture.
    n_frames = len(greys)
    grey_stack, base_h, base_w = resize_to_common_shape(greys)
    region_hw = (base_h, base_w)
    if load_box is not None:                             # the frames are the region; the base is the photograph
        base_w, base_h = full_size
    user_settings = ps                                   # what the result records (full-size boxes)
    user_req = req
    if work_scale < 1.0 and full_size is not None:
        req = reduced_request(req, work_scale, full_size, (base_w, base_h))
        ps = req.settings
        progress(None, 0.19, f"Working at {work_scale * 100:.0f}% size: {base_w} x {base_h} pixels "
                             f"(the photographs are {full_size[0]} x {full_size[1]})")
    mem_plan = plan_solve(region_hw[0], region_hw[1], n_frames, ps)
    if prof is not None:
        prof.header = (f"{n_frames} images, {base_w * base_h / 1e6:.1f} MP each  ({describe_machine()}); "
                       f"expected peak {mem_plan.estimate / 1e9:.1f} GB, {mem_plan.level}"
                       + (f"; working size {work_scale * 100:.0f}%" if work_scale < 1.0 else ""))
    if mem_plan.level != "comfortable":
        from helio.core import parallel
        parallel.STRIP_BYTES = mem_plan.strip_bytes     # restored by solve()
    if req.frames is not None or full_size is None:      # not announced before loading
        _announce(mem_plan, progress)
    del greys
    for i in range(len(rgbs)):
        if rgbs[i].shape[:2] != region_hw:
            rgbs[i] = cv2.resize(rgbs[i], (region_hw[1], region_hw[0]))
    h, w = region_hw
    rgb_list = rgbs
    crop = None
    if load_box is not None:
        crop = tuple(int(v) for v in load_box)
    elif req.crop is not None:
        cx1, cy1, cx2, cy2 = req.crop
        # The crop is in the coordinates of the image it was drawn on; scale
        # it back to the source size when that differs.
        prev_w, prev_h = req.crop_ref_size or (base_w, base_h)
        if prev_w > 0 and prev_h > 0 and (prev_w != base_w or prev_h != base_h):
            sx = base_w / prev_w
            sy = base_h / prev_h
            cx1 = int(round(cx1 * sx)); cy1 = int(round(cy1 * sy))
            cx2 = int(round(cx2 * sx)); cy2 = int(round(cy2 * sy))
        cx1 = max(0, cx1); cy1 = max(0, cy1)
        cx2 = min(base_w, cx2); cy2 = min(base_h, cy2)
        grey_stack = grey_stack[:, cy1:cy2, cx1:cx2]
        rgb_list = [r[cy1:cy2, cx1:cx2] for r in rgb_list]
        h, w = grey_stack.shape[1], grey_stack.shape[2]
        crop = (cx1, cy1, cx2, cy2)

    # How the photographs were encoded (helio.core.tone_curve): their ICC profile's
    # curve, Adobe RGB's gamma 2.2 from the EXIF, or sRGB.  Read when decoding.
    from helio.core.tone_curve import capture_curve
    curve = capture_curve(req.paths) if req.paths else None   # quick looks keep the paths too
    ambient_level = None
    if ps.ambient_frame:
        grey_stack, rgb_list, ambient_level = _subtract_ambient(
            ps, grey_stack, rgb_list, raw_frames, work_scale=work_scale, load_box=load_box,
            region_hw=region_hw, crop=crop, progress=progress, curve=curve)
    grey_stack, rgb_list, raw_gain = _raw_exposure(grey_stack, rgb_list, raw_frames)
    # How far the lamps were, read before anything evens the frames (the
    # fall-off across the frame is half of what shows the distance).
    lamp_distance = None
    if req.lights is not None and ps.resolved_normal_engine() != "Uncalibrated PCA":
        h0, w0 = grey_stack.shape[1:3]
        nf_origin = near_field_origin(ps, req.light_origin, crop, h0, w0)
        nf_positions = near_field_positions(req.light_positions, crop, h0, w0)
        lamp_distance = _lamp_distance(ps, grey_stack, req.lights, nf_origin, nf_positions,
                                       h0, w0, req.mm_per_px, raw_frames=raw_frames, curve=curve)
    light_field = None
    if ps.light_field and req.lights is not None:
        grey_stack, light_field = _apply_light_field(ps, req.lights, grey_stack, raw_frames,
                                                     (base_w, base_h), crop, progress)
    flash = None
    flat_light = None
    if ps.even_flat and ps.flat_areas and req.lights is not None:
        near = None
        if ps.near_field and lamp_distance is not None and req.light_positions is None:
            hh, ww = grey_stack.shape[1:3]
            near = (near_field_radius_px(ps, req.mm_per_px, hh, ww, lamp_distance)[0],
                    near_field_origin(ps, req.light_origin, crop, hh, ww))
        flat_light = _even_flat_areas(ps, req.lights, grey_stack, rgb_list, raw_frames, crop,
                                      (base_w, base_h), curve, near=near)
        if flat_light is not None:
            grey_stack, rgb_list = flat_light.pop("grey_stack"), flat_light.pop("rgb_list")
    if flat_light is None and ps.even_flash and ps.flash_ref and req.lights is not None:
        flash = _even_flash(ps, req.lights, grey_stack, rgb_list, raw_frames, crop,
                            (base_w, base_h), curve)
        if flash is not None:
            grey_stack, rgb_list = flash.pop("grey_stack"), flash.pop("rgb_list")
    light_strength = None
    if ps.semi_calibrated and req.lights is not None:
        if flash is not None or flat_light is not None:
            _log.info("semi-calibrated: not needed, the lamps were evened on the marked reference")
        else:
            light_strength = _even_strengths(ps, req.lights, grey_stack, rgb_list, raw_frames, curve)
            if light_strength is not None:
                grey_stack, rgb_list = light_strength.pop("grey_stack"), light_strength.pop("rgb_list")
    if ps.flatten_illumination:
        # What the solve reads only.  The colour picture stays as
        # photographed: dividing each channel by its own blur, as this did
        # until 2026-10-20, lifted black cloth to a grey haze and flattened
        # the gold and the grey card's steps ("muddy" in Inspect).
        grey_stack = flatten_illumination_greys(grey_stack, w, h)
    if ps.pre_clahe:
        try:
            grey_stack = apply_clahe_stack(grey_stack, clip_limit=ps.pre_clahe_clip,
                                           tile_size=ps.pre_clahe_tile)
        except Exception:
            _log.debug("CLAHE skipped", exc_info=True)   # never abort the solve for it
    shifts = phase_align_stack(grey_stack, rgb_list) if ps.auto_align else None

    from helio.core.parallel import frames_median, frames_var
    albedo_rgb_full = frames_median(rgb_list).astype(np.uint8)
    rgb_list = None                    # nothing reads the colour frames after this: 4.4 GB freed
    grey_median = frames_median(grey_stack)
    albedo_grey_full = grey_median.astype(np.uint8)
    flat_grey = grey_stack.reshape(len(grey_stack), -1)
    progress("Solving normals", 0.35)
    check()
    engine = ps.resolved_normal_engine()
    L = req.lights
    if L is None and engine != "Uncalibrated PCA":
        raise CalibrationRequired(engine)
    if ps.adaptive_light > 0.0 and L is not None:
        grey_stack = adaptive_light_weighting(grey_stack, L, ps.adaptive_light)
        flat_grey = grey_stack.reshape(len(grey_stack), -1).astype(np.float32)

    if ptm_only:                      # the same stack the fit below reads, and nothing else
        return compute_ptm_hsh_coefficients(L, flat_grey)

    # Linear light for the solve (FIX-010): a temporary copy, decoded from the
    # camera's sRGB unless the photos are RAW or declared linear.  Everything
    # made for looking at (albedo, PTM/HSH, fusion) stays on the stored values.
    from helio.core.radiometry import solve_intensities
    solve_stack = solve_intensities(grey_stack, ps.photo_tone, raw_frames, curve)
    flat_solve = solve_stack.reshape(len(solve_stack), -1)
    no_ball = None
    if engine == "Uncalibrated PCA":
        # No ball: the lights found from the photographs (helio.core.uncalibrated),
        # whatever lights the capture has, as the solver has always done.
        from helio.core.uncalibrated import solve_no_ball
        L, n_flat, no_ball = solve_no_ball(flat_solve, h, w)
        normals = n_flat.reshape(h, w, 3)
        if ps.no_ball_inside_out:                  # raised for sunken: half a turn about the view
            normals[..., :2] *= -1.0
            L = np.asarray(L, np.float32).copy()
            L[:, :2] *= -1.0
            no_ball["inside_out"] = True
        _log.info(f"no ball: lights {no_ball['used']}"
                  + (f" ({no_ball['why']})" if no_ball["why"] else
                     f", lamps even to {no_ball['spread'] * 100:.0f}%, raised by {no_ball['raised']}"))
    else:
        nf_origin = near_field_origin(ps, req.light_origin, crop, h, w)
        nf_positions = near_field_positions(req.light_positions, crop, h, w)
        # The white-board light field has measured each lamp's fall-off pixel by
        # pixel and divided it out: the near-field solve then corrects the
        # direction only.  Flat areas even it with a plane, too coarse to stand
        # in for the lamps' own fall-off (on a flat binding under lamps 1.5
        # frame-widths away, direction only left 6 degrees, the full model 1.5).
        evened = light_field is not None
        normals, L = _solve_normals(engine, flat_solve, solve_stack, L, h, w, ps,
                                    flat_ref=None if solve_stack is grey_stack else flat_grey,
                                    mm_per_px=req.mm_per_px,
                                    origin=nf_origin, positions=nf_positions,
                                    found=lamp_distance, falloff=not evened,
                                    # strengths already evened on a reference: each lamp 1 there
                                    equal_power=(flash is None and flat_light is None
                                                 and light_strength is None))
    metal = metal_mask = None
    if ps.metal_aware:
        progress(None, 0.38, "Re-solving where the surface shines")
        normals, metal_mask, metal = _metal_aware(
            ps, engine, normals, flat_solve, None if solve_stack is grey_stack else flat_grey, L, h, w)

    reflectance = _reflectance(ps, req, raw_frames, solve_stack, L, normals, albedo_rgb_full)
    if ps.sfs_refine and L is not None and len(grey_stack) > 0:
        try:
            normals = sfs_refinement(normals, solve_stack[0], L, albedo_grey=None,
                                     n_iters=ps.sfs_iters, lambda_prior=ps.sfs_prior)
        except Exception as e:
            _log.warning(f"SfS refinement skipped: {e}")
    if ps.flat_correction and req.flat_ref_normals is not None:
        if light_field is not None:
            _log.warning("flat-field correction and the white-board light field are both on: they "
                         "correct the same distortion, so it is taken out twice")
        normals = apply_flat_correction_to_normals(normals, req.flat_ref_normals,
                                                   float(ps.flat_corr_strength))
    normals, flipped = auto_orient_normals(normals)
    # Bumps <-> holes, before anything is derived from the normals.
    if ps.invert_relief:
        from helio.core.invert_relief import invert_normals_inplace
        invert_normals_inplace(normals)

    progress(None, 0.4)
    check()
    ptm_coeffs, hsh_coeffs = compute_ptm_hsh_coefficients(L, flat_grey)
    # A machine the plan calls tight does not keep the coefficients: the previews are
    # made now and the fit is repeated if an export asks.  Plenty of memory: unchanged.
    ph, pw = proxy_dims(h, w)
    defer = (ptm_coeffs is not None and mem_plan.level != "comfortable" and req.frames is None
             and not os.environ.get("HELIO_KEEP_PTM"))
    ptm_planes = None
    if defer:
        proxy_ptm = cv2.resize(ptm_coeffs.reshape(6, h, w).transpose(1, 2, 0), (pw, ph)
                               ).transpose(2, 0, 1).reshape(6, -1)
        ptm_planes = _ptm_planes(ptm_coeffs, h, w, pw, ph)
        ptm_coeffs = hsh_coeffs = None
    # the same median as the albedo's, unless adaptive light weighting has
    # changed the stack since
    median_img = grey_median if ps.adaptive_light <= 0.0 else frames_median(grey_stack)
    del grey_median
    spec_var = frames_var(grey_stack)
    roughness_full = 255 - cv2.normalize(np.log1p(spec_var), None, 0, 255,
                                         cv2.NORM_MINMAX).astype(np.uint8)
    light_count_conf = compute_light_count_map(grey_stack)
    rbf = None                         # RBF relighting: the frames themselves, compressed
    if L is not None and len(L) == len(grey_stack) and mem_plan.level != "too_big":
        try:
            from helio.core import rbf_relight
            rbf = rbf_relight.build(grey_stack, L, (ph, pw), keep_full=mem_plan.level == "comfortable")
        except Exception:
            _log.warning("RBF relighting frames - skipping", exc_info=True)

    progress("Integrating depth", 0.70)
    check()
    # Pre-integration smoothing (0-10), the same filter Recalculate Depth
    # uses and on the integration only: the stored normals stay as solved.
    # (Process used to ignore the slider, and had a filter of its own: KD-011.)
    from helio.application.process_tools import smooth_normals
    n_int = smooth_normals(normals, int(round(ps.pre_smooth))) if ps.pre_smooth > 0.01 else normals
    obj_mask = surround_mask(ps, albedo_rgb_full)
    if obj_mask is not None and obj_mask.shape != n_int.shape[:2]:
        obj_mask = None
    depth = integrate_depth(n_int, ps.depth_engine, spec_var=spec_var,
                            light_count_conf=light_count_conf, albedo_rgb=albedo_rgb_full,
                            mask=obj_mask)
    if ps.parabolic_corr:
        depth = parabolic_baseline_correction(depth)
    dv = compute_depth_visualizations(depth, n_int, grey_stack, h, w,
                                      clip_pct=req.depth_clip_pct, mask=obj_mask)
    del n_int

    progress(None, 0.8)
    check()
    terrain = compute_terrain_analysis(dv["depth_vis_p"], normals, h, w)

    progress(None, 0.9)
    progress("Building previews", 0.95)
    proxy: dict[str, Any] = {}
    proxy["normals"] = cv2.resize(normals, (pw, ph))
    # Robust PBR (MVS) needs normals that differ from the standard solve: a
    # trimmed solve, while grey_stack is still here.
    nrob = None
    try:
        if L is not None and len(grey_stack) >= 5 and len(L) == len(grey_stack) \
                and grey_stack[0].shape[:2] == (h, w):
            nrob = trimmed_photometric_stereo(solve_stack, L, h, w)
    except Exception:
        _log.warning("robust normals - falling back to the standard solve", exc_info=True)
        nrob = None
    if nrob is not None and ps.invert_relief:
        from helio.core.invert_relief import invert_normals_inplace
        invert_normals_inplace(nrob)          # its own solve, so its own flip
    proxy["normals_robust"] = proxy["normals"] if nrob is None else cv2.resize(nrob, (pw, ph))
    quality = None
    try:
        qm = stack_quality_maps(grey_stack, L, h, w, solve_stack=solve_stack)
        quality = {k: qm.get(k) for k in ("resid", "fusion", "sigma_deg")}
        for k, v in quality.items():
            proxy[f"quality_{k}"] = None if v is None else cv2.resize(v, (pw, ph))
    except Exception:
        _log.warning("stack quality maps - skipping", exc_info=True)
    del solve_stack, flat_solve

    def to_proxy(d, ch):
        return cv2.resize(d.reshape(ch, h, w).transpose(1, 2, 0), (pw, ph)).transpose(2, 0, 1).reshape(ch, -1)
    proxy["albedo"] = cv2.resize(albedo_rgb_full, (pw, ph))
    proxy["reflectance"] = None if reflectance is None else cv2.resize(reflectance, (pw, ph))
    proxy["roughness"] = cv2.resize(roughness_full, (pw, ph))
    proxy["median"] = cv2.resize(median_img, (pw, ph))
    proxy["depth_p"] = cv2.resize(dv["depth_vis_p"], (pw, ph))
    proxy["depth_fc"] = cv2.resize(dv["depth_vis_fc"], (pw, ph))
    proxy["ptm"] = proxy_ptm if defer else to_proxy(ptm_coeffs, 6)
    for k in ("_multi_hs", "_svf", "_msrm", "_curv", "_slope",
              "_loc_dom", "_pos_opn", "_neg_opn", "_slrm"):
        proxy[k] = cv2.resize(terrain[k], (pw, ph))

    out = SolveResult(
        settings=user_settings, n_images=n_frames, h=h, w=w, base_h=base_h, base_w=base_w, crop=crop,
        light_matrix=L, albedo_rgb_full=albedo_rgb_full, albedo_grey_full=albedo_grey_full,
        normals_full=normals, normals_were_autoflipped=bool(flipped),
        relief_inverted=bool(ps.invert_relief), ptm_coeffs=ptm_coeffs, hsh_coeffs=hsh_coeffs,
        median_img=median_img, spec_var=spec_var, roughness_full=roughness_full,
        light_count_conf=light_count_conf, depth_poisson=depth,
        depth_vis_p=dv["depth_vis_p"], depth_normalised_full=dv["depth_normalised_full"],
        depth_vis_fc=dv["depth_vis_fc"], pca_full=dv["pca_full"], terrain=terrain,
        normals_robust=nrob, quality=quality, proxy=proxy, ph=ph, pw=pw,
        alignment_shifts=shifts, flash=flash, flat_light=flat_light,
        work_scale=work_scale if full_size is not None else 1.0, source_size=full_size,
        ptm_deferred=defer, ptm_planes=ptm_planes, refit=user_req if defer else None,
        light_strength=light_strength, metal=metal, metal_mask=metal_mask, rbf=rbf, no_ball=no_ball,
        lamp_distance=lamp_distance,
        reflectance_rgb=reflectance, raw_gain=raw_gain, light_field=light_field, lens=lens_note,
        tone_curve=(("sRGB" if curve is None else curve.name) if ps.photo_tone == "srgb" else ""),
        lights_placed=(int(np.isfinite(np.asarray(req.light_positions, float)).all(axis=1).sum())
                       if req.light_positions is not None and (ps.near_field or engine == "Near-Field Attenuation")
                       else 0),
        ambient=None if ambient_level is None else {"file": os.path.basename(ps.ambient_frame),
                                                    "level": round(float(ambient_level), 4)})
    del grey_stack, rgb_list, flat_grey
    gc.collect()
    return out
