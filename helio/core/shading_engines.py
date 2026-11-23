"""
Shading engine dispatcher.

``render_shading(engine_name, data_manager, engine_settings)`` is the single
entry point for all per-pixel rendering.  Engines dispatch via
``_ENGINE_DISPATCH`` (name -> function).

Adding a new engine
--------------------
1. Write a function ``def _engine_xxx(data, s, h, w, ctx) -> np.ndarray``.
2. Add an entry to ``_ENGINE_DISPATCH`` with name -> function reference.
3. Add the display name to ``RTIViewerPro.engines_list`` so it appears in the
   render-engine dropdown (that list is the source of truth for the UI).
4. Optionally add cache keys to ``invalidate_engine_cache()``.
"""

from __future__ import annotations

from helio.core.frame_read import read_frame

import math
import os
from typing import TYPE_CHECKING, Optional

import cv2
import logging
import numpy as np

from helio.core.frames import normal_divergence, normals_from_slopes
from helio.core.parallel import by_rows, cached, median, quantiles
from helio.core._native import rs as _rs

_log = logging.getLogger(__name__)


# ── Fluid relighting (PERF-001) ────────────────────────────────────────
#
# A drag on the light dome asks for a new picture every few milliseconds at
# the preview's full size (up to 2500 px on the long edge).  The engines
# that change with the light therefore do two things:
#
# * work that does not depend on the light (unit normals, smoothed normals,
#   blurred colour, the shadow height field) is kept in
#   helio.core.parallel's cache, keyed by the arrays it came from;
# * the per-pixel part runs band by band on every core (by_rows).  Each band
#   does exactly the arithmetic the whole-array code did at those pixels.
#
# Whole-image statistics (medians, percentiles) stay whole-image: they go
# through parallel.quantiles, which finds the same order statistics without
# sorting.  tests/test_render_speed.py checks every changed engine against the
# code it replaced, pixel for pixel, and times the drag.

def _unit_normals_flat(normals: np.ndarray) -> np.ndarray:
    """(h*w, 3) float32 unit normals: light-free, so worked out once per array."""
    def make():
        n = normals.astype(np.float32).reshape(-1, 3)
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    return cached("unit-normals", (normals,), (), make)


def _flat3(a: np.ndarray) -> np.ndarray:
    """An (h, w, 3) array as (h*w, 3) without copying when it can."""
    return a.reshape(-1, 3)


def _rust(kernel: str, *arrays) -> bool:
    """True when helio_core_rs has *kernel* and every array is the C-contiguous
    float32 it takes (so the call borrows NumPy's memory, no copy).  Otherwise
    the engine runs its NumPy code, which is the reference the kernel is
    tested against."""
    return (_rs is not None and hasattr(_rs, kernel)
            and all(a.dtype == np.float32 and a.flags.c_contiguous for a in arrays))


def _f32(v) -> np.ndarray:
    return np.ascontiguousarray(v, np.float32)

# Optional RVT (Relief Visualisation Toolbox)
try:
    import rvt.vis
    HAS_RVT = True
except ImportError:
    HAS_RVT = False

if TYPE_CHECKING:
    from helio.data_manager import DataManager


# ── Registry ──────────────────────────────────────────────────────────

def render_shading(engine_name: str, data: DataManager,
                   settings: dict, h: int, w: int) -> np.ndarray:
    """Render *engine_name* using *data* arrays.

    Returns (h, w, 3) uint8 or float32.
    Delegates to ``get_shading_engine`` for dispatch.
    """
    return get_shading_engine(engine_name, data, settings, h, w)


def invalidate_engine_cache(data: DataManager, engine: str) -> None:
    """Clear cached full/proxy arrays for *engine* in *data*."""
    _CACHE_MAP = {
        "SVF": "_svf_", "Multi-Scale Relief Map": "_msrm_",
        "Curvature": "_curv_", "Slope": "_slope_",
        "Opened Relief Map": ("_pos_opn_", "_neg_opn_"),
        "Local Dominant Orientation": "_loc_dom_",
        "Shallow-Light Relief Map": "_slrm_",
    }
    key = _CACHE_MAP.get(engine)
    if key is None:
        return
    if isinstance(key, str):
        setattr(data, key + "full", None)
        setattr(data, key + "proxy", None)
    elif isinstance(key, tuple):
        for k in key:
            setattr(data, k + "full", None)
            setattr(data, k + "proxy", None)


def compute_ptm_relighting(
    c: np.ndarray, u: float, v: float,
) -> np.ndarray:
    """PTM relighting: quadratic polynomial evaluation at (u, v)."""
    return c[0]*u**2 + c[1]*v**2 + c[2]*u*v + c[3]*u + c[4]*v + c[5]


def calc_hillshade_raw(
    d_map: np.ndarray,
    settings: dict,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Single-direction hillshade from PS normals or depth map.
    See ``RTIViewerPro.calc_hillshade_raw`` for docstring."""
    sa  = np.deg2rad(settings.get('hs_alt', 45))
    saz = np.deg2rad(settings.get('hs_azi', 315))
    amb = settings.get('hs_amb', 0.15)
    # Azimuth → light vector, expressed in the PS-NORMALS frame (+Y = up = North,
    # per estimate_light_from_specular's ny = (cy-hy)/r):
    #   az 0 = North (+Y), 90 = East (+X), 180 = South (−Y), 270 = West (−X).
    # (History: this first used lx=cos az, ly=sin az — a math convention rotated
    #  90° from the dome. That was corrected to sin/−cos to match the dome's
    #  SCREEN frame, but the dome's screen frame is itself Y-flipped relative to
    #  the normals, so North still lit South. ly is now +cos az, which puts the
    #  light in the same frame as the normals — see the note in
    #  RTIViewerPro.get_shading.)
    lx  = np.cos(sa) * np.sin(saz)
    ly  = np.cos(sa) * np.cos(saz)
    lz  = np.sin(sa)
    L   = np.array([lx, ly, lz], dtype=np.float32)

    if normals is not None and normals.ndim == 3 and normals.shape[2] == 3:
        h, w = normals.shape[:2]
        N = _unit_normals_flat(normals)
        if _rust("hillshade", N):
            return _rs.hillshade(N, L, np.float32(amb)).reshape(h, w)
        out = np.empty((h, w), np.float32)

        def band(y0, y1):
            hs_b = np.clip(np.abs(N[y0 * w:y1 * w] @ L), 0, 1)
            out[y0:y1] = np.clip(hs_b + amb, 0, 1).reshape(y1 - y0, w)
        by_rows(band, h, w)
        return out
    else:
        raw = d_map.astype(np.float32)
        _h, _w = raw.shape
        d_rng = float(np.ptp(raw))
        if d_rng < 1e-9:
            return np.full((_h, _w), amb, dtype=np.float32)
        hs_z = settings.get('hs_z', 1.0)
        dm = (raw - raw.min()) / d_rng * hs_z * min(_h, _w) / 4.0
        gx = cv2.Sobel(dm, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(dm, cv2.CV_32F, 0, 1, ksize=3)
        slope  = np.pi / 2.0 - np.arctan(np.sqrt(gx**2 + gy**2))
        aspect = np.arctan2(-gy, -gx)
        # saz − 90° shifts the classic GDAL azimuth (0 = +X/East) into the
        # app-canonical frame (0 = North/top) so this depth-only fallback lights
        # from the same compass direction as the normals branch above.
        hs = np.sin(sa)*np.sin(slope) + np.cos(sa)*np.cos(slope)*np.cos((saz - np.pi/2) - aspect)

    return np.clip(hs.astype(np.float32) + amb, 0, 1)


def calc_multi_hillshade(
    d_map: np.ndarray,
    settings: dict,
    normals: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-directional hillshade — Mark (1992) / USGS weighted algorithm.
    See ``RTIViewerPro.calc_multi_hillshade`` for docstring."""
    sa  = np.deg2rad(settings.get('hs_alt', 45))
    amb = settings.get('hs_amb', 0.1)
    lz  = float(np.sin(sa))
    r   = float(np.cos(sa))

    _az_weights = [
        (0,   1.000), (45,  0.707), (90,  1.000), (135, 0.707),
        (180, 1.000), (225, 0.707), (270, 1.000), (315, 0.707),
    ]
    _total_w = sum(w for _, w in _az_weights)

    if normals is not None and normals.ndim == 3 and normals.shape[2] == 3:
        h, w = normals.shape[:2]
        N = _unit_normals_flat(normals)
        lights = []
        for az, weight in _az_weights:
            az_r = np.radians(az)
            # +cos: light expressed in the PS-normals frame (+Y = North) — see
            # calc_hillshade_raw / RTIViewerPro.get_shading notes.
            lights.append((np.array([r * float(np.sin(az_r)), r * float(np.cos(az_r)), lz],
                                    dtype=np.float32), weight))
        if _rust("multi_lambert", N):
            multi, _ = _rs.multi_lambert(N, _f32([l for l, _ in lights]),
                                         _f32([wt for _, wt in lights]))
            multi /= np.float32(_total_w)
            multi += np.float32(amb)
            return np.clip(multi, 0, 1, out=multi).reshape(h, w)
        out = np.empty((h, w), np.float32)

        def band(y0, y1):
            nb = N[y0 * w:y1 * w]
            multi = np.zeros(nb.shape[0], dtype=np.float32)
            for L, weight in lights:
                multi += np.maximum(0.0, nb @ L) * weight
            out[y0:y1] = np.clip((multi / _total_w).reshape(y1 - y0, w) + amb, 0, 1)
        by_rows(band, h, w)
        return out

    hs_z = settings.get('hs_z', 1.0)
    raw  = d_map.astype(np.float32)
    _h, _w = raw.shape
    d_r = float(np.ptp(raw))
    if d_r < 1e-9:
        return np.full((_h, _w), 0.3, dtype=np.float32)
    dm = (raw - raw.min()) / d_r * hs_z * min(_h, _w) / 4.0
    gx = cv2.Sobel(dm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dm, cv2.CV_32F, 0, 1, ksize=3)
    slope  = np.pi / 2.0 - np.arctan(np.sqrt(gx**2 + gy**2))
    aspect = np.arctan2(-gy, -gx)
    multi  = np.zeros((_h, _w), dtype=np.float32)
    for az, weight in _az_weights:
        multi += np.clip(
            np.sin(sa) * np.sin(slope)
            + np.cos(sa) * np.cos(slope) * np.cos(np.deg2rad(az) - aspect),
            0, 1) * weight
    return np.clip(multi / _total_w + amb, 0, 1)


def compute_hsh_relighting(
    c: list[float],
    u: float,
    v: float,
    z: float,
) -> float:
    """HSH relighting — 9-coefficient hemispherical harmonics.
    Returns radiance for direction (u,v,z) given coefficients c[0..8].
    """
    return (c[0] + c[1]*u + c[2]*v + c[3]*z + c[4]*u*v
            + c[5]*u*z + c[6]*v*z + c[7]*(u**2 - v**2) + c[8]*(3*z**2 - 1))


def calc_svf(d_map: np.ndarray, settings: dict) -> np.ndarray:
    """Sky-View Factor using Yokoyama (2002) horizon scanning.
    See ``RTIViewerPro.calc_svf`` for docstring."""
    dem   = d_map.astype(np.float32)
    H, W  = dem.shape
    r_px  = max(5, int(settings.get('svf_radius', settings.get('op_radius', 20))))
    n_dir = max(4, int(settings.get('svf_dirs', 16)))

    yg, xg = np.mgrid[0:H, 0:W].astype(np.float32)
    sum_cos2 = np.zeros((H, W), np.float32)

    for d_i in range(n_dir):
        az  = 2.0 * np.pi * d_i / n_dir
        sdx = float(np.sin(az))
        sdy = -float(np.cos(az))
        max_tan = np.full((H, W), -9999.0, np.float32)
        for r in range(1, r_px + 1):
            tx = np.clip(xg + r * sdx, 0, W - 1)
            ty = np.clip(yg + r * sdy, 0, H - 1)
            target_z = cv2.remap(dem, tx, ty, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            tan_e = (target_z - dem) / float(r)
            max_tan = np.maximum(max_tan, tan_e)
        horizon = np.arctan(max_tan)
        sum_cos2 += np.cos(horizon) ** 2

    return np.clip(sum_cos2 / n_dir, 0.0, 1.0).astype(np.float32)


def _use_cache_fn(
    data: DataManager,
    attr_full: str,
    attr_proxy: str,
    compute_fn,
    use_proxy: bool,
    is_real_crop: bool,
    h: int,
    w: int,
) -> np.ndarray:
    """Return the cached proxy result for interactive (low-res) display;
    otherwise always compute fresh at the exact resolution requested.

    The "*_full" cache (e.g. data._curv_full) is set once in Helio_Pro.py
    from compute_terrain_analysis, which deliberately works at a fixed
    ~1380px cap for speed (see its `_work_sc`) — it is NOT native-resolution
    despite the name. A prior version of this function reused that cache
    for any non-cropped, non-proxy request by resizing it up to (h, w) via
    Lanczos, which silently upsampled a ~1380px computation to full
    manuscript resolution for whole-image renders/exports (e.g. "Export
    Current View" or "Export Mag Region" with no crop drawn) — soft output
    that didn't match the sharp result from zooming into a crop (which
    already always recomputes fresh, since is_real_crop=True bypasses both
    cache branches). Interactive whole-image display is unaffected: it
    always requests the proxy explicitly (use_proxy=True), so this branch
    is only ever reached by genuine full-resolution, non-cropped requests
    (i.e. exports), which should always be computed fresh at that
    resolution rather than upscaled from a lower-res cache.
    """
    if use_proxy and getattr(data, attr_proxy, None) is not None:
        return getattr(data, attr_proxy)
    return compute_fn()


ShadingCtx = dict


# Module-level dispatch table — populated below after all _engine_xxx functions
# are defined.  This is referenced by get_shading_engine() and by Helio_Pro.py
# to check if an engine has been migrated.
_ENGINE_DISPATCH: dict[str, callable] = {}


# ── Engine metadata ─────────────────────────────────────────────────────────
#
# What each engine reads, and which family it belongs to.
#
# This exists because a user swapped the depth engine, pressed Recalculate, and
# saw an identical picture.  That was mostly correct - only a minority of
# engines read the height map at all - but nothing in the program could say so,
# and a control that appears to do nothing reads as a broken control.
#
# 'consumes' is the honest answer to "will changing X affect this view?".  Note
# it could NOT be derived from source alone: Sky-View Factor and Negative
# Openness measurably respond to a depth change while reading depth only
# through a helper, one level below anything a regex over the function body can
# see.  So the table is checked *behaviourally* by tests/test_engine_meta.py, which
# renders each engine against two depth engines and asserts that anything which
# visibly moves is declared here.  Static inspection under-reports; measurement
# does not.
_ENGINE_META: dict[str, dict] = {
    'Ambient Occlusion (AO)': {'family': 'Relief and terrain', 'consumes': ['depth', 'normals']},
    'Bilateral Normal Sharpening': {'family': 'Normals', 'consumes': ['light', 'normals']},
    'Cast-Shadow Raking': {'family': 'Relief and terrain', 'consumes': ['depth', 'light', 'normals']},
    'Colour ICA (IC1-3)': {'family': 'Decomposition', 'consumes': ['plates']},
    'Curvedness': {'family': 'Relief and terrain', 'consumes': ['depth']},
    'Diffuse Gain': {'family': 'Relighting', 'consumes': ['normals']},
    'Exaggerated Shading': {'family': 'Relief and terrain', 'consumes': ['light', 'normals']},
    'False-Colour Depth': {'family': 'Depth', 'consumes': ['depth']},
    'Image Quotient (IQ)': {'family': 'Decomposition', 'consumes': ['light', 'lightmatrix', 'ptm']},
    'Light Coverage Map': {'family': 'Quality control', 'consumes': ['normals']},
    'Local Dominance': {'family': 'Relief and terrain', 'consumes': []},
    'Multi-Light Fusion': {'family': 'Relighting', 'consumes': []},
    'Multi-Scale Relief Model (MSRM)': {'family': 'Relief and terrain', 'consumes': ['depth']},
    'Negative Openness': {'family': 'Relief and terrain', 'consumes': ['depth']},
    'Normal Residual': {'family': 'Quality control', 'consumes': ['normals']},
    'Normal Uncertainty (QC)': {'family': 'Quality control', 'consumes': []},
    'Normals RGB': {'family': 'Normals', 'consumes': ['normals']},
    'Normals Unsharp Masking': {'family': 'Normals', 'consumes': ['light', 'normals']},
    'PCA 1': {'family': 'Decomposition', 'consumes': []},
    'PCA 2': {'family': 'Decomposition', 'consumes': []},
    'PCA 3': {'family': 'Decomposition', 'consumes': []},
    'PS Fit Residual (QC)': {'family': 'Quality control', 'consumes': []},
    'PS Normal-Map Lit (8-Dir)': {'family': 'Relighting', 'consumes': ['depth', 'normals']},
    'PS Normal-Map Lit (Single)': {'family': 'Relighting', 'consumes': ['depth', 'normals']},
    'Positive Openness': {'family': 'Relief and terrain', 'consumes': ['depth']},
    'RGB-PS R-B Diff': {'family': 'Decomposition', 'consumes': ['lightmatrix', 'plates']},
    'RTI Image USM': {'family': 'Relighting', 'consumes': ['light', 'normals']},
    'RBF Relight (Shiny / Translucent)': {'family': 'Relighting', 'consumes': ['light', 'plates']},
    'RTI Specular Enhancement': {'family': 'Relighting', 'consumes': ['light', 'normals', 'ptm']},
    'Raking Composite': {'family': 'Relief and terrain', 'consumes': ['normals']},
    'Multi-Directional Hillshading (Blend / PCA)': {'family': 'Relief and terrain', 'consumes': ['normals']},
    'Albedo-Suppressed Raking (Clay Mode)': {'family': 'Relighting', 'consumes': ['light', 'normals']},
    'Ridge and Groove Detector': {'family': 'Relief and terrain', 'consumes': ['depth']},
    'Retinex Enhancement': {'family': 'Decomposition', 'consumes': []},
    'Robust PBR (MVS)': {'family': 'Relighting', 'consumes': ['light', 'normals']},
    'Shape Index': {'family': 'Relief and terrain', 'consumes': ['depth']},
    'Sky-View Factor (SVF)': {'family': 'Relief and terrain', 'consumes': ['depth']},
    'Slope Gradient': {'family': 'Relief and terrain', 'consumes': ['depth']},
    'Slope Magnitude': {'family': 'Relief and terrain', 'consumes': ['depth', 'normals']},
    'Specular Enhancement': {'family': 'Relighting', 'consumes': ['light', 'ptm']},
    'Specular View': {'family': 'Relighting', 'consumes': ['lightmatrix']},
    'Standard PBR': {'family': 'Relighting', 'consumes': ['light', 'normals']},
    'Matte Relight (as 3D)': {'family': 'Relighting', 'consumes': ['light', 'normals']},
    'Subsurface Scattering (SSS)': {'family': 'Decomposition', 'consumes': ['light', 'normals']},
    'Surface Curvature (Scratches)': {'family': 'Relief and terrain', 'consumes': ['depth', 'normals']},
}


def engine_meta(engine: str) -> dict:
    """Family and inputs for *engine*; empty-ish defaults when unknown."""
    return _ENGINE_META.get(engine, {"family": "Other", "consumes": []})


# The order families appear in the view menu.  Not alphabetical: this is
# roughly the order a capture is looked at.  Relighting first because that is
# what the surface actually looks like and where Standard PBR lives, then the
# normal field it was solved from, then the decompositions that pull the
# surface apart, then relief and terrain, then the raw depth, and finally the
# quality-control views - which answer "should I trust this" rather than
# "what does it look like", and are the last thing you reach for.
_FAMILY_ORDER = ("Relighting", "Normals", "Decomposition",
                 "Relief and terrain", "Depth", "Quality control")

# Within Relighting, Standard PBR leads: it is the default engine and the one
# everything else is judged against.
_FAMILY_LEADS = {"Relighting": ("Standard PBR", "Matte Relight (as 3D)")}


def engine_families() -> dict[str, list[str]]:
    """Engine names grouped by family, in the order the menu shows them."""
    out: dict[str, list[str]] = {}
    for name in _ENGINE_DISPATCH:
        out.setdefault(engine_meta(name)["family"], []).append(name)

    def _rank(fam):
        return (_FAMILY_ORDER.index(fam) if fam in _FAMILY_ORDER
                else len(_FAMILY_ORDER), fam)

    ordered = {}
    for fam in sorted(out, key=_rank):
        leads = [n for n in _FAMILY_LEADS.get(fam, ()) if n in out[fam]]
        rest = sorted(n for n in out[fam] if n not in leads)
        ordered[fam] = leads + rest
    return ordered


def engines_consuming(what: str) -> set[str]:
    """Engines that read *what* - e.g. "depth", "ptm", "plates"."""
    return {n for n in _ENGINE_DISPATCH if what in engine_meta(n)["consumes"]}


def implemented_engines() -> set[str]:
    """Return the set of engine names that have been migrated to the module."""
    return set(_ENGINE_DISPATCH.keys())
"""Context dict for shading engines. Keys: n_std, n_rob, med, depth_p,
depth_fc, ptm, light, gain, res_res, use_proxy, _is_real_crop, crop_coords."""


def get_shading_engine(
    engine_name: str,
    data: DataManager,
    settings: dict,
    h: int,
    w: int,
    ctx: Optional[ShadingCtx] = None,
) -> np.ndarray:
    """Dispatch to the shading engine matching *engine_name*.

    This is the main entry point that replaces the original
    ``RTIViewerPro.get_shading()``.  Each ``if engine == …`` block from
    the monolithic method is now a separate pure function in this module.

    When *h* / *w* match the full-size dimensions the full-resolution
    cached array is updated; otherwise the proxy version is updated.

    When *ctx* is ``None`` the function builds a minimal context from
    ``data.*`` directly (no crop, no overrides).  When provided the
    caller (typically ``Helio_Pro.py``) controls the pre-computed arrays.

    Returns (h, w, 3) uint8 or float32.
    """
    if ctx is None:
        use_proxy = False
        is_real_crop = False
        crop_coords = (0, 0, data.w, data.h) if (data.w and data.h) else None
        # DataManager declares every one of these up-front as None, so a plain
        # getattr(..., default) default is unreachable — the attribute always
        # exists and the caller would get None.  Treat None as missing so a
        # cleared cache degrades to the intended zero-array instead of crashing.
        def _arr(name, default):
            v = getattr(data, name, None)
            return default if v is None else v

        n_std  = _arr('normals_full',    np.zeros((h, w, 3), np.float32))
        n_rob  = _arr('normals_robust',  n_std)
        med    = _arr('median_img',      np.zeros((h, w), np.float32))
        depth_p = _arr('depth_vis_p',    np.zeros((h, w), np.float32))
        depth_fc = _arr('depth_vis_fc',  depth_p)
        ptm    = _arr('ptm_coeffs',      np.zeros((6, h * w), np.float32))
        # Negate Y: lx/ly/lz are stored in SCREEN coords (+Y = down = South) but
        # the PS normals use +Y = up = North.  Mirrors the same conversion in
        # RTIViewerPro.get_shading — see the long note there.
        light  = np.array([settings.get('lx', 0), -settings.get('ly', 0),
                           settings.get('lz', 1)], dtype=np.float32)
        gain   = float(settings.get('gain', 1.0))
        res_res = 1.0
        ctx = dict(n_std=n_std, n_rob=n_rob, med=med, depth_p=depth_p,
                   depth_fc=depth_fc, ptm=ptm, light=light, gain=gain,
                   res_res=res_res, use_proxy=use_proxy,
                   _is_real_crop=is_real_crop, crop_coords=crop_coords)

    # ── Dispatch ────────────────────────────────────────────────────
    # Only engines with implemented _engine_xxx functions are listed.
    # Engines not yet migrated fall through to the zero-array fallback
    # (or are handled by Helio_Pro.py's inline fallback).
    DISPATCH = _ENGINE_DISPATCH
    fn = DISPATCH.get(engine_name)
    if fn is not None:
        return fn(data, settings, h, w, ctx)
    return np.zeros((h, w), dtype=np.float32)


# ── Individual engine implementations ─────────────────────────────────

# ── Standard PBR exponent lookup table ────────────────────────────────
#
# max(0, N.L) ** (1/gain) spends nearly all its time in the exponent, not the
# dot product: measured at 1200x1600, dot 0.9 ms against np.power 14.9 ms of a
# 15.2 ms frame. A table over the 0..1 domain removes that, roughly halving the
# engine (8.75 -> 4.08 ms at 1280x960).
#
# It is only sound for **gain <= 1**. Above that the exponent falls below 1, so
# the curve has infinite slope at zero and the first bin alone is badly wrong -
# measured max error against exact np.power:
#
#     gain 0.15 -> 0.41/255      gain 2.0 -> 3.96/255
#     gain 1.0  -> 0.06/255      gain 3.0 -> 15.9/255
#                                gain 5.0 -> 48.2/255
#
# No table size fixes it (the derivative is unbounded), and interpolating to
# fix it costs 8.94 ms - slower than the np.power it replaces. So above gain 1
# the engine simply does the exact thing. The gain slider runs to 5.0, so this
# is a case users reach.
_PBR_LUT_N = 4096
_PBR_LUT_CACHE: dict[float, np.ndarray] = {}


def _pbr_lut(inv_gain: float):
    """Table for ``x ** inv_gain`` over x in 0..1, or None if unsafe/unhelpful."""
    if not np.isfinite(inv_gain) or inv_gain < 1.0:
        return None                      # gain > 1: quantisation is visible
    key = round(float(inv_gain), 4)
    lut = _PBR_LUT_CACHE.get(key)
    if lut is None:
        if len(_PBR_LUT_CACHE) > 8:      # gain changes rarely; do not grow
            _PBR_LUT_CACHE.clear()
        lut = np.power(np.linspace(0.0, 1.0, _PBR_LUT_N, dtype=np.float32),
                       key).astype(np.float32)
        _PBR_LUT_CACHE[key] = lut
    return lut


_PBR_FLOOR = 0.45          # steady brightness: a flat area is never drawn darker than this
_PBR_KNEE = 0.62           # ...and highlights roll off smoothly above this instead of being cut at 1


def _pbr_exposure(light, inv_gain: float, on) -> float:
    """The factor that keeps a flat area from going dark as the light is lowered (1.0: as the classic app).

    Standard and Robust PBR raise N.L to the power 1 / gain (6.7 at the default gain 0.15), so a flat area is
    sin(elevation) ** 6.7: 0.23 with the light 53 degrees up, 0.003 at 25, black below.  With steady
    brightness on, a flat area reads at least ``_PBR_FLOOR`` whatever the elevation, and as the classic app
    draws it once the light is high enough to reach that (about 62 degrees at the default gain): it only lifts
    a low light, and never makes any picture brighter than the classic one.  Overhead the factor is exactly 1.
    Computed in float64: 1 / sin(el) ** power is large under a low light."""
    if not on:
        return 1.0
    nrm = float(np.linalg.norm(light)) or 1.0
    lz = min(1.0, max(0.05, float(light[2]) / nrm))
    return max(1.0, _PBR_FLOOR * lz ** (-float(inv_gain)))


def _pbr_shoulder(x):
    """Smooth the top of a lifted picture: unchanged up to ``_PBR_KNEE``, then easing towards 1 without
    ever reaching it, so bright slopes keep their gradation instead of a flat white plateau."""
    x = np.asarray(x, np.float64)
    over = np.maximum(x - _PBR_KNEE, 0.0) / (1.0 - _PBR_KNEE)
    return np.where(x <= _PBR_KNEE, x, _PBR_KNEE + (1.0 - _PBR_KNEE) * (1.0 - np.exp(-over)))


def _ctx_normals(ctx, data, h, w):
    """Resolve normals from ctx, falling back to data.normals_proxy during drag."""
    n = ctx.get('n_std')
    if n is not None and n.shape[:2] == (h, w):
        return n
    return getattr(data, 'normals_full', np.zeros((h, w, 3), np.float32))

def _engine_standard_pbr(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Standard PBR (energy-conserving Blinn-Phong)."""
    gain = ctx['gain']
    light = ctx['light']
    n_std = ctx['n_std']
    _g = max(0.01, gain)
    _ah, _aw = n_std.shape[:2]
    if _ah != h or _aw != w:
        h, w = _ah, _aw
    flat = _flat3(n_std)
    _lut = _pbr_lut(1.0 / _g)
    _k = _pbr_exposure(light, 1.0 / _g, s.get('pbr_expose', 1))
    if _lut is not None and _k != 1.0:
        _lut = _pbr_shoulder(_lut.astype(np.float64) * _k).astype(np.float32)      # a scaled copy: the cached table is shared
    if _lut is not None and _rust("standard_pbr", flat):
        return _rs.standard_pbr(flat, _f32(light), _lut).reshape(h, w)
    out = np.empty(h * w, np.float32 if _lut is not None else np.result_type(flat, light))

    def band(y0, y1):
        _d = np.dot(flat[y0 * w:y1 * w], light)
        np.maximum(_d, 0, out=_d)
        if _lut is not None:
            # Table lookup instead of a per-pixel pow. Measured at 1280x960:
            # 8.75 ms -> 4.08 ms, max error 0.41/255 at the default gain.
            out[y0 * w:y1 * w] = _lut[np.minimum((_d * _PBR_LUT_N).astype(np.int32),
                                                 _PBR_LUT_N - 1)]
        else:
            out[y0 * w:y1 * w] = _pbr_shoulder(np.power(_d, 1.0 / _g) * _k) if _k != 1.0 \
                else np.power(_d, 1.0 / _g)
    by_rows(band, h, w)
    return out.reshape(h, w)


def _engine_matte_relight(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """The 3D relief's lighting on the picture: ambient + diffuse * N.L, and a
    soft Blinn-Phong sheen seen from straight above (helio_qt.relief3d, its
    defaults 0.18, 0.95, 0.12 and 24).  Standard PBR raises N.L to 1 / gain
    (6.7 at 0.15), a tight pool of light that reads as shine on the colour;
    this keeps the photograph's colour as photographed, the light still
    movable.  The sheen multiplies the colour (the composite multiplies the
    shading by it), so it is the material's own tint, not white.  Scaled so a
    flat area lit from overhead is 1 and clipped there: every view returns
    0..1, and over 1 is read as a 0..255 picture."""
    light = np.asarray(ctx['light'], np.float64)
    n_std = ctx['n_std']
    h, w = n_std.shape[:2]
    flat = _flat3(n_std)
    nl = float(np.linalg.norm(light)) or 1.0
    l = (light / nl).astype(np.float32)
    half = l + np.array([0.0, 0.0, 1.0], np.float32)
    half /= float(np.linalg.norm(half)) or 1.0
    amb = float(s.get('mr_ambient', 0.18))
    dif = float(s.get('mr_diffuse', 0.95))
    shine = float(s.get('mr_shine', 0.12))
    power = max(1.0, float(s.get('mr_shininess', 24.0)))
    top = 1.0 / max(amb + dif + shine, 1e-3)       # a flat area under a light overhead is 1
    out = np.empty(h * w, np.float32)

    def band(y0, y1):
        n = flat[y0 * w:y1 * w]
        d = np.maximum(n @ l, 0.0)
        v = amb + dif * d
        if shine > 0.0:
            v = v + shine * np.where(d > 0.0, np.power(np.maximum(n @ half, 0.0), power), 0.0)
        np.clip(v * top, 0.0, 1.0, out=out[y0 * w:y1 * w])
    by_rows(band, h, w)
    return out.reshape(h, w)


def _engine_robust_pbr(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Robust PBR (MVS) using robust normals.

    The exponent is 1/gain, matching _engine_standard_pbr. It used to be gain
    itself, so the shared default of 0.15 (chosen to *prevent* blown
    highlights, see RTIViewerPro.__init__) raised N.L to the power 0.15 and
    blew the render out to near-uniform white instead: mean 254, std 0.5.
    One setting cannot mean opposite things in two sibling engines.
    """
    gain = ctx['gain']
    light = ctx['light']
    n_rob = ctx.get('n_rob')
    if n_rob is None:
        n_rob = ctx['n_std']
    _g = max(0.01, gain)
    _ah, _aw = n_rob.shape[:2]
    if _ah != h or _aw != w:
        h, w = _ah, _aw
    flat = _flat3(n_rob)
    out = np.empty((h, w), np.result_type(flat, light))
    _k = _pbr_exposure(light, 1.0 / _g, s.get('pbr_expose', 1))

    def band(y0, y1):
        v = np.power(np.clip(np.abs(np.dot(flat[y0 * w:y1 * w], light)
                                    .reshape(y1 - y0, w)), 0, 1), 1.0 / _g)
        out[y0:y1] = _pbr_shoulder(v * _k) if _k != 1.0 else v
    by_rows(band, h, w)
    return out


def _engine_slope_magnitude(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Slope Magnitude from normal map."""
    n_std = ctx['n_std']
    gain = ctx['gain']
    _ah, _aw = n_std.shape[:2]
    if _ah != h or _aw != w:
        h, w = _ah, _aw
    return np.clip(np.sqrt(n_std[..., 0]**2 + n_std[..., 1]**2).reshape(h, w) * gain * 2.0, 0, 1)


def _engine_normals_usm(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Normals Unsharp Masking."""
    n_std = ctx['n_std']
    light = ctx['light']
    gain = ctx['gain']
    _sigma = float(s.get('nr_scale', 5.0))

    def make():
        N_sharp = n_std + ((n_std - cv2.GaussianBlur(n_std, (0, 0), sigmaX=_sigma)) * gain * 2.0)
        norm = np.linalg.norm(N_sharp, axis=-1, keepdims=True)
        return np.divide(N_sharp, norm, out=np.zeros_like(N_sharp), where=norm != 0)
    N_sharp = cached("normals-usm", (n_std,), (_sigma, float(gain)), make)
    h, w = N_sharp.shape[:2]
    flat = _flat3(N_sharp)
    out = np.empty((h, w), np.result_type(flat, light))

    def band(y0, y1):
        out[y0:y1] = np.power(np.maximum(0, np.dot(flat[y0 * w:y1 * w], light)
                                         .reshape(y1 - y0, w)), 1.0)
    by_rows(band, h, w)
    return out


def _engine_spec_enhance(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Specular Enhancement via PTM coefficients."""
    ptm = ctx['ptm']
    light = ctx['light']
    gain = ctx['gain']
    # Proxy path: the relighting at preview size, which render() builds from
    # the PTM planes area-averaged to that size (the same as relighting at
    # full size and shrinking the result: both steps are linear).
    _relit_p = ctx.get('_ptm_relit_proxy')
    _quad_p  = ctx.get('_ptm_quad_proxy')
    _gamma = float(s.get('dg_gamma', 1.0))
    if _relit_p is not None and _quad_p is not None:
        ph, pw = _relit_p.shape[:2]
        out = np.empty((ph, pw), np.float32)

        def band(y0, y1):
            result = _relit_p[y0:y1] + _quad_p[y0:y1] * gain * 2.5
            if _gamma != 1.0:
                result = np.power(np.clip(result / 255.0, 0, 1), _gamma) * 255.0
            out[y0:y1] = np.clip(result / 255.0, 0, 1)
        by_rows(band, ph, pw)
        return out
    elif ptm is not None and ptm.ndim == 2 and ptm.shape[1] == h * w:
        result = (compute_ptm_relighting(ptm, light[0], light[1])
                  + (ptm[0] * light[0]**2 + ptm[1] * light[1]**2) * gain * 2.5).reshape(h, w)
    else:
        return np.zeros((h, w), np.float32)
    if _gamma != 1.0:
        result = np.power(np.clip(result / 255.0, 0, 1), _gamma) * 255.0
    return np.clip(result / 255.0, 0, 1).astype(np.float32)


def _engine_ambient_occlusion(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Normal-based Ambient Occlusion with depth prominence."""
    n_std = ctx['n_std']
    depth_p = ctx['depth_p']
    gain = ctx['gain']
    nz = np.clip(n_std[..., 2], 0, 1)
    ao_radius = float(s.get('ao_radius', 0.08))
    ao_strength = float(s.get('ao_strength', 1.5))
    sigma = max(h, w) * ao_radius
    # The wide blur (~3 s at 2500 px) depends only on the depth map and the
    # radius, not on the gain or the light: kept, so only the first visit pays.
    depth_lo = cached("ao-lowpass", (depth_p,), (float(sigma),),
                      lambda: cv2.GaussianBlur(depth_p.astype(np.float32), (0, 0), sigmaX=sigma))
    prominence = np.clip((depth_p - depth_lo) * 2.0 + 0.5, 0, 1)
    ao_raw = nz * prominence
    ao_gamma = np.power(np.clip(ao_raw, 1e-6, 1.0), 1.0 / max(0.1, ao_strength))
    return np.clip(ao_gamma * gain, 0, 1)


def _engine_false_colour_depth(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """False-Colour Depth map."""
    depth_p = ctx['depth_p']
    gain = ctx['gain']
    d = cv2.resize(depth_p, (w, h)).astype(np.float32)
    return np.clip(cv2.normalize(d, None, 0, 1, cv2.NORM_MINMAX) * gain, 0, 1)


def _engine_light_coverage(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Light Coverage Map from normals Z-component."""
    gain = ctx['gain']
    crop_coords = ctx['crop_coords']
    use_proxy = ctx['use_proxy']
    if getattr(data, 'normals_full', None) is None:
        return np.zeros((h, w))
    src = data.normals_proxy if use_proxy else data.normals_full
    if not use_proxy and crop_coords:
        x1, y1, x2, y2 = crop_coords
        src = src[y1:y2, x1:x2]
    if src.shape[:2] != (h, w):
        src = cv2.resize(src, (w, h))
    cov_map = np.clip(src[..., 2].astype(np.float32), 0, 1)
    return np.clip(cov_map * gain, 0, 1)


def _engine_normals_rgb(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Normals as a true RGB normal map: XYZ mapped to RGB.

    This used to build the colour map and then throw the colour away with
    COLOR_RGB2GRAY, so an engine called "Normals RGB" returned greyscale -
    on a near-flat page that is a featureless grey rectangle, which is why it
    read as blank. The framework handles 3-channel returns (Colour ICA has
    always used them), so the map is now returned as the lilac normal map
    everyone in photometric stereo expects.
    """
    gain = ctx['gain']
    crop_coords = ctx['crop_coords']
    use_proxy = ctx['use_proxy']
    if getattr(data, 'normals_full', None) is None:
        return np.zeros((h, w))
    src = data.normals_proxy if use_proxy else data.normals_full
    if not use_proxy and crop_coords:
        x1, y1, x2, y2 = crop_coords
        src = src[y1:y2, x1:x2]
    if src.shape[:2] != (h, w):
        src = cv2.resize(src.astype(np.float32), (w, h))
    rgb = (src.astype(np.float32) * 0.5 + 0.5).clip(0, 1)
    return np.clip(rgb * gain, 0, 1)


def _engine_shape_index(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Shape Index from principal curvatures."""
    depth_p = ctx['depth_p']
    gain = ctx['gain']
    if getattr(data, 'depth_vis_p', None) is None:
        return np.zeros((h, w))
    d = cv2.resize(depth_p, (w, h)).astype(np.float32)
    blur_r = max(1, int(s.get('shape_smooth', 1.5)))
    if blur_r > 0:
        d = cv2.GaussianBlur(d, (0, 0), sigmaX=blur_r)
    Lx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    Ly = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    Lxx = cv2.Sobel(Lx, cv2.CV_32F, 1, 0, ksize=3)
    Lyy = cv2.Sobel(Ly, cv2.CV_32F, 0, 1, ksize=3)
    Lxy = cv2.Sobel(Lx, cv2.CV_32F, 0, 1, ksize=3)
    H_curv = -(Lxx + Lyy) * 0.5
    K_curv = Lxx * Lyy - Lxy**2
    disc = np.sqrt(np.maximum(H_curv**2 - K_curv, 0))
    k1 = H_curv + disc
    k2 = H_curv - disc
    shape_idx = (2.0 / np.pi) * np.arctan((k1 + k2) / (k1 - k2 + 1e-9))
    return np.clip((shape_idx + 1.0) * 0.5 * gain, 0, 1)


def _engine_anisotropic_svf(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Anisotropic Sky View Factor with contrast stretch and directional
    strength control.

    Controllable settings (engine sliders):
        svf_radius   — search radius in pixels  (5–200)
        svf_dirs     — number of search directions  (4–32)
        hs_azi       — azimuth for anisotropy direction  (0–360°)
        hs_z         — directional strength  (0.1–10; 1 ≈ current default)
        gain         — brightness multiplier
        contrast     — percentile clip for contrast stretch (0=off)
    """
    depth_p = ctx['depth_p']
    gain = ctx['gain']
    _r = max(5, int(s.get('svf_radius', 20)))
    _nd = max(4, int(s.get('svf_dirs', 16)))
    _az = float(s.get('hs_azi', 315))
    _strength = float(s.get('hs_z', 1.0))
    _contrast = float(s.get('contrast', 0))

    # Compute base SVF and directional aspect together
    d = depth_p.astype(np.float32)

    # Try RVT for ASVF if available
    svf_arr = None
    w_dir = np.clip(np.cos(
        np.arctan2(
            -cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3),
            -cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        ) - np.deg2rad(_az)
    ) * 0.5 + 0.5, 0, 1)

    if HAS_RVT:
        try:
            svf_arr = rvt.vis.sky_view_factor(
                dem=depth_p, resolution=ctx['res_res'],
                compute_svf=False, compute_asvf=True,
                svf_n_dir=_nd, svf_r_max=_r,
                asvf_dir=_az, asvf_level=1)['asvf']
        except Exception:
            _log.warning("RVT sky-view factor failed, falling back to calc_svf")

    if svf_arr is None:
        svf_arr = calc_svf(d, dict(s, svf_radius=_r, svf_dirs=_nd))

    if svf_arr is None:
        return np.zeros((h, w), np.float32)

    # Blend SVF with directional weighting
    # hs_z = 0 → pure SVF, 1 → current default, 2+ → strong directional
    result = svf_arr * np.clip(
        1.0 - _strength * 0.5 + w_dir * _strength * 0.5, 0.1, 1.0)

    # Contrast stretch — spread the naturally narrow SVF range
    if _contrast > 0:
        _clip = min(_contrast, 49)
        p_low, p_high = np.percentile(result, [_clip, 100 - _clip])
        if p_high > p_low:
            result = np.clip((result - p_low) / (p_high - p_low + 1e-8), 0, 1)

    result = np.clip(result * gain, 0, 1)
    return result.astype(np.float32)


# Engines that need the _use_cache pattern:

def _terrain_cached(data: DataManager, ctx: ShadingCtx, tag: str,
                    h: int, w: int, params: tuple, compute):
    """Memoise a slow, light-independent engine for one view.

    SVF, Openness and Local Dominance march rays over the depth map. None of
    them depends on the light, yet they were recomputed on every repaint and
    every drag of the light dome - measured at **870 ms and 800 ms on a 1.2 MP
    test image**, so seconds apiece on a real 45 MP capture. That is the
    "engine X is unusably slow" report; RGB-PS R-B Diff, which gets the blame,
    is 2 ms once warm.

    The key carries the settings the engine reads *and* a cheap fingerprint of
    the depth map, so re-processing a capture invalidates it rather than
    quietly serving the previous surface.
    """
    d = ctx.get('depth_p')
    if d is None:
        return compute()
    da = np.asarray(d)
    fp = (da.shape, float(da[::37, ::37].sum()))
    key = (tag, params, fp, h, w, ctx['use_proxy'], ctx.get('crop_coords'),
           ctx.get('_is_real_crop'), ctx.get('_pretrimmed_crop_frac'))
    cache = getattr(data, '_terrain_cache', None)
    if cache is not None and cache.get('key') == key:
        return cache['img']
    img = compute()
    data._terrain_cache = {'key': key, 'img': img}
    return img


def _svf_bundle(dem, resolution, n_dir, r_max, negate=False):
    """One RVT horizon scan, serving both sky-view factor and openness.

    `rvt.vis.sky_view_factor` computes SVF and openness from the *same* ray
    march and returns both when asked. HELIO asked twice: the SVF engine
    called it with ``compute_svf=True`` and Positive Openness called it again,
    identically, with ``compute_opns=True``. Measured on a 960x1280 depth map
    at 16 directions: 636 ms and 572 ms separately, **664 ms for both
    together** - the second scan was very nearly free and we were paying full
    price for it.

    The key carries the radius and direction count, so the two engines share a
    scan only when they genuinely asked for the same one; different settings
    simply produce separate entries rather than a wrong answer. The depth
    fingerprint is the same cheap strided sum `_terrain_cached` uses, so a
    re-solve invalidates this too.
    """
    da = np.asarray(dem)
    key = (da.shape, float(da[::37, ::37].sum()), float(resolution),
           int(n_dir), int(r_max), bool(negate))
    cached = getattr(_svf_bundle, "_slot", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    surface = -da if negate else da
    res = _horizon_scan_rust(surface, resolution, n_dir, r_max) if _rs is not None else None
    if res is None:
        res = rvt.vis.sky_view_factor(
            dem=surface, resolution=resolution,
            compute_svf=True, compute_opns=True,
            svf_n_dir=int(n_dir), svf_r_max=int(r_max))
    _svf_bundle._slot = (key, res)
    return res


def _horizon_scan_rust(dem, resolution, n_dir, r_max):
    """``rvt.vis.sky_view_factor(dem, resolution, compute_svf=True,
    compute_opns=True, svf_n_dir=n_dir, svf_r_max=r_max)`` from helio_core_rs, or
    None when it cannot (the module is old, or the surface is not 2-D).

    RVT's preparation is repeated step for step - float32, NaN mask, divide by
    the pixel size, reflect-pad by the radius, its own table of search points -
    and only the scan itself (a full-image np.roll / np.fmax for every direction
    and radius, ~7 s at 2500 px) is replaced by one pass per pixel."""
    if not hasattr(_rs, "horizon_scan") or getattr(dem, "ndim", 0) != 2 or min(dem.shape) < 2:
        return None
    try:
        dem = np.asarray(dem).astype(np.float32)
        nan_mask = np.isnan(dem)
        dem = dem / resolution                              # RVT: float32 / Python float
        move = rvt.vis.horizon_shift_vector(num_directions=int(n_dir), radius_pixels=int(r_max),
                                            min_radius=1)
        shifts, dist, starts = [], [], [0]
        for key in move:
            shifts.extend(move[key]["shift"])
            dist.extend(move[key]["distance"])
            starts.append(len(dist))
        padded = np.ascontiguousarray(np.pad(dem, int(r_max), mode="reflect"))
        svf, opns = _rs.horizon_scan(
            padded, int(r_max), np.ascontiguousarray(np.asarray(shifts, np.int32).reshape(-1, 2)),
            np.asarray(dist, np.float64), np.asarray(starts, np.int64), True, True)
        svf[nan_mask] = np.nan
        opns[nan_mask] = np.nan
        return {"svf": svf, "opns": opns}
    except Exception:
        _log.warning("helio_core_rs horizon scan failed; using RVT", exc_info=True)
        return None


def _openness_keys(s: dict) -> tuple:
    return (s.get('op_radius'), s.get('svf_dirs'), s.get('op_smooth'), s.get('op_clip'),
            s.get('op_mode'))


def _engine_pos_openness(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    return _terrain_cached(data, ctx, 'posopen', h, w, _openness_keys(s),
                           lambda: _openness_view(data, s, h, w, ctx, negative=False))


def _engine_neg_openness(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    return _terrain_cached(data, ctx, 'negopen', h, w, _openness_keys(s),
                           lambda: _openness_view(data, s, h, w, ctx, negative=True))


def _stretch(a: np.ndarray, clip: float) -> np.ndarray:
    """*a* to 0..1 between its *clip* and 100 - *clip* percentiles (0: its
    minimum and maximum, as the terrain views always were)."""
    a = np.nan_to_num(np.asarray(a, np.float32))
    if clip > 0:
        lo, hi = (float(v) for v in np.percentile(a, [clip, 100 - clip]))
    else:
        lo, hi = float(a.min()), float(a.max())
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1).astype(np.float32)


def _openness_raw(depth, res_res, s, negative):
    """Openness of *depth* (RVT's angles, or calc_svf's 0..1 when RVT is missing)."""
    _r = max(5, int(s.get('op_radius', 20)))
    _nd = max(4, int(s.get('svf_dirs', 16)))
    if HAS_RVT:
        try:
            # Negative openness scans the inverted surface, so it cannot share
            # positive's bundle - but it can still keep its own.
            return _svf_bundle(depth, res_res, _nd, _r, negate=negative)['opns']
        except Exception:
            _log.warning("RVT openness failed, falling back to calc_svf")
    o = calc_svf(depth, dict(s, svf_radius=_r, svf_dirs=_nd))
    return 1.0 - o if negative else o


def _openness_view(data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
                   negative: bool) -> np.ndarray:
    """Positive or Negative Openness, with its display choices.

    *op_smooth* smooths the surface first (pixels), to quieten paper grain;
    *op_clip* stretches between percentiles rather than the extremes, so a
    deep hole or the object's rim does not leave the surface a flat grey;
    *op_mode* 1 shows positive minus negative openness, RVT's openness
    difference: ridges and raised strokes bright, grooves and hollows dark,
    in one picture."""
    depth_p = ctx['depth_p']
    res_res = ctx['res_res']
    use_proxy = ctx['use_proxy']
    is_real_crop = ctx['_is_real_crop']
    smooth = float(s.get('op_smooth', 0.0))
    clip = float(s.get('op_clip', 0.0))
    mode = int(s.get('op_mode', 0))

    def _compute():
        d = np.asarray(depth_p, np.float32)
        if smooth > 0:
            d = cv2.GaussianBlur(d, (0, 0), smooth)
        if mode == 1:
            pos = _openness_raw(d, res_res, s, False)
            neg = _openness_raw(d, res_res, s, True)
            return _stretch(np.asarray(pos, np.float32) - np.asarray(neg, np.float32), max(clip, 0.5))
        return _stretch(_openness_raw(d, res_res, s, negative), clip)

    if smooth or clip or mode:          # the stored preview is the plain view
        return _compute()
    return _use_cache_fn(data, '_neg_opn_full' if negative else '_pos_opn_full',
                         '_neg_opn_proxy' if negative else '_pos_opn_proxy', _compute,
                         use_proxy, is_real_crop, h, w)


def _engine_local_dominance(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    return _terrain_cached(data, ctx, 'locdom', h, w, (s.get('ld_radius'), s.get('ld_max')),
                           lambda: _engine_local_dominance_impl(data, s, h, w, ctx))


def _engine_local_dominance_impl(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Local Dominance."""
    depth_p = ctx['depth_p']
    use_proxy = ctx['use_proxy']
    is_real_crop = ctx['_is_real_crop']

    def _compute():
        _rmin = max(5, int(s.get('ld_radius', 10)))
        _rmax = max(10, int(s.get('ld_max', 20)))
        if HAS_RVT:
            try:
                _res = rvt.vis.local_dominance(dem=depth_p, min_rad=_rmin, max_rad=_rmax)
                # 2-98 %: min-max let the few extremes (the chrome ball, the frame's edge) take
                # the range, and the object itself came out one flat grey (2-98 % in 0.32-0.43)
                _lo, _hi = np.nanpercentile(_res, [2, 98])
                return np.clip((_res - _lo) / max(float(_hi - _lo), 1e-9), 0, 1).astype(np.float32)
            except TypeError:
                pass
        from scipy.ndimage import uniform_filter
        _lo = uniform_filter(depth_p.astype(np.float32), _rmin * 2 + 1)
        _hi = uniform_filter(depth_p.astype(np.float32), _rmax * 2 + 1)
        _d = depth_p.astype(np.float32) - (_lo + _hi) * 0.5
        return np.clip(0.5 + _d / (float(np.percentile(np.abs(_d), 98)) + 1e-9) * 0.5,
                       0, 1).astype(np.float32)

    return _use_cache_fn(data, '_loc_dom_full', '_loc_dom_proxy', _compute,
                         use_proxy, is_real_crop, h, w)


def _engine_sky_view_factor(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    return _terrain_cached(data, ctx, 'svf', h, w, (s.get('svf_radius'), s.get('svf_dirs')),
                           lambda: _engine_sky_view_factor_impl(data, s, h, w, ctx))


def _engine_sky_view_factor_impl(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    depth_p = ctx['depth_p']
    res_res = ctx['res_res']
    use_proxy = ctx['use_proxy']
    is_real_crop = ctx['_is_real_crop']

    def _compute_svf():
        _r  = max(5, int(s.get('svf_radius', 20)))
        _nd = max(4, int(s.get('svf_dirs', 16)))
        if HAS_RVT:
            try:
                return np.clip(
                    _svf_bundle(depth_p, res_res, _nd, _r)['svf'],
                    0, 1).astype(np.float32)
            except Exception:
                _log.warning("RVT sky-view factor failed, falling back to calc_svf")
        return calc_svf(depth_p, dict(s, svf_radius=_r, svf_dirs=_nd))
    # NOTE: no stale "_svf_full" cache reuse here — see _use_cache_fn's
    # docstring for why resizing a cached lower-working-resolution result
    # up to a full-resolution request silently produces soft output that
    # also goes stale the moment depth_p changes (e.g. an EMD depth
    # override). This branch is only reached for full-resolution,
    # non-cropped requests (exports), so recomputing fresh is correct.
    out = _compute_svf()

    # Display stretch. Sky-view factor is the *fraction of sky visible*, so on
    # a manuscript leaf or a book cover it sits at ~1.0 everywhere and renders
    # as a white rectangle: measured mean 254.0, std 0.2 out of 255 on a real
    # capture, which is a correct number and an unusable picture. The signal is
    # entirely in the small deviations, so stretch them to fill the range, the
    # way the other relief engines here already do.
    #
    # This is the *display* engine only. The GIS exports write the raw values
    # from data._svf_full, so nothing quantitative is touched by this.
    if bool(s.get('svf_stretch', True)):
        f = np.asarray(out, np.float32)
        lo, hi = np.percentile(f, (float(s.get('svf_lo_pct', 1.0)),
                                   float(s.get('svf_hi_pct', 99.0))))
        if hi - lo > 1e-6:
            out = np.clip((f - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return out


def _engine_ps_normals_lit_8dir(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    n_std = ctx['n_std']
    depth_p = ctx['depth_p']
    _nrm_use = n_std
    _depth_src = s.get('hs_depth_src', 'normals')
    if _depth_src in ('hf_depth', 'uhf_depth') and getattr(data, 'depth_vis_p', None) is not None:
        _dv = data.depth_vis_p.astype(np.float32)
        _sigma = max(0.5, h * 0.005) if _depth_src == 'uhf_depth' else max(0.5, h * 0.05)
        _d_hp = _dv - cv2.GaussianBlur(_dv, (0,0), _sigma)
        _gx = cv2.Sobel(_d_hp, cv2.CV_32F, 1, 0, ksize=3)
        _gy = cv2.Sobel(_d_hp, cv2.CV_32F, 0, 1, ksize=3)
        _gz = np.ones_like(_gx) * (h * 0.02)
        # +Y up, like the solved normals it is blended with below.
        _nrm_depth = normals_from_slopes(_gx, _gy, _gz).astype(np.float32)
        if _nrm_use is not None and _nrm_use.shape[:2] == _nrm_depth.shape[:2]:
            _nrm_use = _nrm_depth * 0.7 + _nrm_use * 0.3
            _nrm_len = np.linalg.norm(_nrm_use, axis=-1, keepdims=True)
            _nrm_use = _nrm_use / np.maximum(_nrm_len, 1e-8)
        else:
            _nrm_use = _nrm_depth
    return calc_multi_hillshade(depth_p, s, normals=_nrm_use)


def _engine_ps_normals_lit_single(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    n_std = ctx['n_std']
    depth_p = ctx['depth_p']
    depth_fc = ctx['depth_fc']
    src_depth = depth_fc if s.get('hs_depth_src', 'poisson') == 'frankot' else depth_p
    return calc_hillshade_raw(src_depth, s, normals=n_std)


def _engine_diffuse_gain(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    if getattr(data, 'normals_full', None) is None:
        return np.zeros((h, w), np.float32)
    dg_str  = float(s.get('dg_strength', 2.0))
    dg_gam  = max(0.1, float(s.get('dg_gamma',  0.7)))
    dg_spec = float(s.get('dg_spec',    0.3))
    gain = ctx['gain']
    n_std = ctx['n_std']
    _nrm = n_std
    if _nrm is None:
        return np.zeros((h, w), np.float32)
    if _nrm.shape[:2] != (h, w):
        _nrm = cv2.resize(_nrm.astype(np.float32), (w, h), cv2.INTER_LINEAR)
    if _nrm is n_std:
        N = _unit_normals_flat(_nrm)             # light-free: once per solve, shared with the hillshades
    else:
        N = _nrm.astype(np.float32).reshape(-1, 3)
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-8)
    dg_alt = float(np.clip(s.get('dg_alt', 35.0), 5.0, 80.0))
    lz = float(np.sin(np.radians(dg_alt)))
    r_ = float(np.cos(np.radians(dg_alt)))
    lights = []
    for az in [0, 45, 90, 135, 180, 225, 270, 315]:
        az_r = np.radians(az)
        lights.append(np.array([r_ * np.sin(az_r), -r_ * np.cos(az_r), lz], dtype=np.float32))
    if _rust("diffuse_gain", N):
        dg_out = _rs.diffuse_gain(N, np.stack(lights), np.float32(dg_str), np.float32(dg_gam),
                                  np.float32(dg_spec))
    else:
        shading_sum = np.zeros(h * w, dtype=np.float32)
        shading_max = np.zeros(h * w, dtype=np.float32)
        for L in lights:
            raw_shade = np.maximum(0.0, N @ L)
            shading_sum += raw_shade
            shading_max = np.maximum(shading_max, raw_shade)
        avg_shade = (shading_sum / 8.0).astype(np.float32)
        if dg_spec > 0.01:
            avg_shade = np.maximum(0, avg_shade - dg_spec * shading_max)
        dg_out = np.power(np.clip(avg_shade * dg_str, 0, 1), dg_gam)
    dg_out = dg_out.reshape(h, w)
    p2, p98 = quantiles(dg_out, [2, 98])          # np.percentile's values, without the sort
    if p98 - p2 > 1e-6:
        dg_out = np.clip((dg_out - p2) / (p98 - p2), 0, 1)
    return np.clip(dg_out.astype(np.float32) * gain, 0, 1)


def _engine_surface_curvature(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    n_std = ctx['n_std']
    depth_p = ctx['depth_p']
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    is_real_crop = ctx['_is_real_crop']
    smooth = float(s.get('curv_smooth', 0.0))

    def nxy():
        # the normals' X and Y, smoothed first when asked (quietens paper grain)
        nx, ny = n_std[..., 0].astype(np.float32), n_std[..., 1].astype(np.float32)
        if smooth > 0:
            nx, ny = cv2.GaussianBlur(nx, (0, 0), smooth), cv2.GaussianBlur(ny, (0, 0), smooth)
        return nx, ny

    def _compute_curv():
        if n_std is not None and n_std.ndim == 3:
            curv = normal_divergence(*nxy())
            curv = curv * gain * 8.0 + 0.5
            return np.clip(curv, 0, 1).astype(np.float32)
        d1 = depth_p.astype(np.float32)
        if d1.ndim == 3:
            d1 = cv2.cvtColor(d1, cv2.COLOR_RGB2GRAY) if d1.shape[2] == 3 else d1[..., 0]
        d_rng = float(np.ptp(d1))
        if d_rng > 1e-6:
            d1 = d1 / d_rng * min(d1.shape) / 4.0
        lap = cv2.Laplacian(d1, cv2.CV_32F)
        return np.clip(lap * gain * 2.0 + 0.5, 0, 1).astype(np.float32)
    if use_proxy and n_std is not None and n_std.ndim == 3:
        # On screen, at the size shown: the whole-picture view used the
        # terrain analysis's copy, made at about 1,380 px and stretched up,
        # which looked soft.  Scaled to the picture's own range (1st/99th
        # percentile), as that copy was, so it looks as it did, only sharp.
        # Exports and crops (and Helio Classic's renders) are unchanged.
        curv = normal_divergence(*nxy())
        half = max(abs(float(np.percentile(curv, 1))), abs(float(np.percentile(curv, 99))), 1e-8)
        return np.clip(curv / half * 0.5 * gain + 0.5, 0, 1).astype(np.float32)
    if smooth > 0:                          # the stored preview is the plain view
        return _compute_curv()
    return _use_cache_fn(data, '_curv_full', '_curv_proxy', _compute_curv,
                         use_proxy, is_real_crop, h, w)


def _engine_slope_gradient(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    depth_p = ctx['depth_p']
    res_res = ctx['res_res']
    use_proxy = ctx['use_proxy']
    is_real_crop = ctx['_is_real_crop']

    smooth = float(s.get('slope_smooth', 0.0))
    clip = float(s.get('slope_clip', 0.0))

    def _compute_slope():
        d = np.asarray(depth_p, np.float32)
        if smooth > 0:
            d = cv2.GaussianBlur(d, (0, 0), smooth)
        if HAS_RVT:
            try:
                _sa = rvt.vis.slope_aspect(dem=d,
                                            resolution_x=res_res, resolution_y=res_res,
                                            output_units='degree')
                return _stretch(_sa['slope'], clip)
            except Exception:
                _log.warning("RVT slope failed, falling back to Sobel-based slope")
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
        return _stretch(np.sqrt(gx**2 + gy**2), clip)
    # See _use_cache_fn's docstring — no stale "_slope_full" cache reuse;
    # this branch only fires for full-resolution exports, so always
    # recompute fresh from the current depth_p.
    return _compute_slope()


def _engine_msrm(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    depth_p = ctx['depth_p']
    res_res = ctx['res_res']
    use_proxy = ctx['use_proxy']
    is_real_crop = ctx['_is_real_crop']

    def _compute_msrm():
        _rmin = max(1, int(s.get('msrm_min', 1)))
        _rmax = max(_rmin+1, int(s.get('msrm_max', 5)))
        acc = None
        if HAS_RVT:
            try:
                # scaling_factor is required since rvt-py 2; 2 is RVT's own default.
                # Without it every call failed and the fallback below ran instead.
                acc = rvt.vis.msrm(dem=depth_p, resolution=res_res,
                                   feature_min=float(_rmin), feature_max=float(_rmax),
                                   scaling_factor=2).astype(np.float32)
            except Exception:
                _log.warning("RVT multi-scale relief model failed, falling back to Gaussian differences")
        if acc is None:
            d = depth_p.astype(np.float32)
            scales = np.linspace(_rmin, _rmax, 4)
            acc = np.zeros_like(d)
            for sig in scales:
                acc += d - cv2.GaussianBlur(d, (0,0), sigmaX=max(0.3, float(sig)))
            acc /= len(scales)
        # a percentile stretch (2-98 % by default): a min-max stretch lets the
        # object's rim set the contrast and leaves the surface a flat grey.
        return _stretch(acc, float(s.get('msrm_clip', 2.0)))
    # See _use_cache_fn's docstring — no stale "_msrm_full" cache reuse;
    # this branch only fires for full-resolution exports, so always
    # recompute fresh from the current depth_p.
    return _compute_msrm()


def _engine_image_quotient(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    ptm = ctx['ptm']
    med = ctx['med']
    light = ctx['light']
    gain = ctx['gain']
    # Proxy path: use pre-computed full-res relighting
    _relit_p = ctx.get('_ptm_relit_proxy')
    if _relit_p is not None and _relit_p.shape == (h, w):
        _ptm_r = _relit_p
    elif ptm is not None and ptm.ndim == 2 and ptm.shape[1] == h * w:
        _ptm_r = compute_ptm_relighting(ptm, light[0], light[1]).reshape(h, w).astype(np.float32)
    else:
        return np.zeros((h, w), np.float32)
    def med_norm():
        _med_r = med.reshape(h, w).astype(np.float32)
        return _med_r / (_med_r.max() + 1e-8)
    _med_n = cached("iq-median", (med,), (h, w), med_norm)
    _den = _ptm_r.max() + 1e-8
    _iq = np.empty((h, w), np.result_type(_ptm_r, _med_n))

    def quotient(y0, y1):
        _iq[y0:y1] = (_ptm_r[y0:y1] / _den) / (_med_n[y0:y1] + 0.05)
    by_rows(quotient, h, w)
    _p1, _p99 = quantiles(_iq, [1, 99])
    _k = max(0.1, gain)

    def stretch(y0, y1):
        t = np.clip((_iq[y0:y1] - _p1) / (_p99 - _p1 + 1e-8), 0, 1).astype(np.float32)
        _iq[y0:y1] = np.clip((t - 0.5) * _k + 0.5, 0, 1)
    by_rows(stretch, h, w)
    def_spec = float(s.get('def_spec', 0.0))
    if def_spec > 0.0 and getattr(data, 'spec_var', None) is not None:
        _spec_ch = data.spec_var.astype(np.float32)
        crop_coords = ctx.get('crop_coords')
        use_proxy = ctx['use_proxy']
        if crop_coords and not use_proxy:
            _qx1, _qy1, _qx2, _qy2 = crop_coords
            _spec_ch = _spec_ch[_qy1:_qy2, _qx1:_qx2]
        _spec_ch = cv2.normalize(_spec_ch, None, 0, 1, cv2.NORM_MINMAX)
        if _spec_ch.shape != (h, w):
            _spec_ch = cv2.resize(_spec_ch, (w, h), interpolation=cv2.INTER_LINEAR)
        _iq = np.clip(_iq * (1.0 - def_spec) + _spec_ch * def_spec, 0, 1)
    return _iq.astype(np.float32)


def _engine_specular_view(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    if getattr(data, 'spec_var', None) is None:
        return np.zeros((h, w))
    spec = data.spec_var.astype(np.float32)
    crop_coords = ctx.get('crop_coords')
    use_proxy = ctx['use_proxy']
    if crop_coords and not use_proxy:
        _sx1, _sy1, _sx2, _sy2 = crop_coords
        spec = spec[_sy1:_sy2, _sx1:_sx2]
    # Robust normalisation, not min-max. Specular variance has a long tail on
    # real captures - a handful of blown highlight pixels set the maximum and
    # squash everything else toward zero, after which the default noise floor
    # of 0.5 zeroes the lot. Measured on a real 8-frame manuscript: mean 1.0
    # out of 255, i.e. a black rectangle. The synthetic fixture has no such
    # outliers, which is why it looked correct there.
    _lo, _hi = np.percentile(spec, (1.0, 99.0))
    if _hi - _lo > 1e-8:
        spec = np.clip((spec - _lo) / (_hi - _lo), 0.0, 1.0).astype(np.float32)
    else:
        spec = cv2.normalize(spec, None, 0, 1, cv2.NORM_MINMAX)
    if spec.shape[0] != h or spec.shape[1] != w:
        spec = cv2.resize(spec, (w, h), interpolation=cv2.INTER_LINEAR)
    sensitivity  = float(s.get('def_sens',   1.0))
    noise_floor  = float(s.get('def_noise',  0.5))
    relief       = float(s.get('def_relief', 1.0))
    gain = ctx['gain']
    spec = np.clip(spec * sensitivity, 0, 1)
    spec = np.where(spec < noise_floor, 0.0, spec)
    spec = np.power(np.clip(spec, 1e-6, 1.0), 1.0 / max(0.1, relief))
    return np.clip(spec * gain, 0, 1)


def _engine_pca(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
    ch: int = 0,
) -> np.ndarray:
    if getattr(data, 'pca_full', None) is None:
        return np.zeros((h, w))
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    crop_coords = ctx.get('crop_coords')
    if use_proxy:
        src = data.pca_full[..., ch].astype(np.float32) / 255.0
        if src.shape != (h, w):
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        src = data.pca_full[..., ch].astype(np.float32) / 255.0
        if crop_coords:
            _px1, _py1, _px2, _py2 = crop_coords
            src = src[_py1:_py2, _px1:_px2]
        if src.shape != (h, w):
            src = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(src * gain, 0, 1)


def _engine_retinex(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    if getattr(data, 'albedo_rgb_full', None) is None:
        return np.zeros((h, w))
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    crop_coords = ctx.get('crop_coords')
    src_full = (data.albedo_proxy if use_proxy else data.albedo_rgb_full).astype(np.float32)
    if not use_proxy and crop_coords:
        x1r,y1r,x2r,y2r = crop_coords; src_full = src_full[y1r:y2r, x1r:x2r]
    if src_full.shape[:2] != (h, w): src_full = cv2.resize(src_full, (w, h))
    gray = cv2.cvtColor(np.clip(src_full, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    from helio.core.processing import fast_large_sigma_blur
    scale_pct = float(s.get('ret_scale_s', 15))
    sigma = max(3.0, h * scale_pct / 100.0)
    local_mean = fast_large_sigma_blur(gray, sigma)
    lcn = (gray + 1.0) / (local_mean + 1.0)
    scale_m_pct = float(s.get('ret_scale_m', 80))
    if abs(scale_m_pct - scale_pct) > 5:
        sigma_m = max(3.0, h * scale_m_pct / 100.0)
        local_mean_m = fast_large_sigma_blur(gray, sigma_m)
        lcn_m = (gray + 1.0) / (local_mean_m + 1.0)
        w_s = float(s.get('ret_scale_l', 250)) / 500.0
        w_s = max(0.0, min(1.0, w_s))
        lcn = lcn * (1.0 - w_s) + lcn_m * w_s
    p2, p98 = np.percentile(lcn, 2), np.percentile(lcn, 98)
    rng = p98 - p2
    if rng < 1e-6:
        return np.full((h, w), 0.5, dtype=np.float32)
    retinex = np.clip((lcn - p2) / rng, 0, 1).astype(np.float32)
    if abs(gain - 1.0) > 0.01:
        retinex = np.clip(0.5 + (retinex - 0.5) * gain, 0, 1)
    return retinex


def _engine_raking_composite(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    if getattr(data, 'normals_full', None) is None:
        return np.zeros((h, w), np.float32)
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    crop_coords = ctx.get('crop_coords')
    import math as _math
    el      = float(s.get('rk_elevation', 20))
    n_dirs  = max(2, int(s.get('rk_n_dirs', 4)))
    usm_str = float(s.get('rk_usm', 0.0))
    if use_proxy:
        normals = (getattr(data, 'normals_proxy', None)
                   if getattr(data, 'normals_proxy', None) is not None
                   else data.normals_full)
    else:
        normals = data.normals_full
        if crop_coords:
            x1c, y1c, x2c, y2c = crop_coords
            normals = normals[y1c:y2c, x1c:x2c]
    if normals.shape[:2] != (h, w):
        normals = cv2.resize(normals.astype(np.float32),
                             (w, h), interpolation=cv2.INTER_LINEAR)
    N = normals.astype(np.float32).reshape(-1, 3)
    N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-8)
    lz_ = _math.sin(_math.radians(el))
    r_  = _math.cos(_math.radians(el))
    shading = np.zeros(h * w, dtype=np.float32)
    for i in range(n_dirs):
        az = _math.radians(360.0 * i / n_dirs)
        lv = np.array([r_ * _math.sin(az), -r_ * _math.cos(az), lz_], dtype=np.float32)
        shading += np.maximum(0.0, N @ lv)
    composite = (shading / n_dirs).reshape(h, w)
    p2, p98 = np.percentile(composite, 2), np.percentile(composite, 98)
    if p98 - p2 > 1e-6:
        composite = np.clip((composite - p2) / (p98 - p2), 0, 1)
    if usm_str > 0.01:
        blurred = cv2.GaussianBlur(composite, (0, 0), max(1.0, h * 0.008))
        composite = np.clip(composite + (composite - blurred) * usm_str, 0, 1)
    return np.clip(composite.astype(np.float32) * gain, 0, 1)


# ── Multi-Directional Hillshading (blend / PCA) and Clay Mode (2026-10) ─────

_MDH_AZIMUTHS = tuple(range(0, 360, 45))


def _mdh_lights(alt_deg: float) -> np.ndarray:
    """Eight unit light vectors at one altitude, every 45 degrees, in the normals'
    frame (+Y up; azimuth 0 = north, 90 = east, as calc_hillshade_raw)."""
    el = np.radians(alt_deg)
    az = np.radians(_MDH_AZIMUTHS)
    return np.stack([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az),
                     np.full(len(az), np.sin(el))], axis=1).astype(np.float32)


def _stretch01(a: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    """*a* scaled so its lo..hi percentiles map to 0..1, clipped (unchanged if flat)."""
    lo, hi = quantiles(np.ascontiguousarray(a), [lo_pct, hi_pct])
    if hi - lo <= 1e-6:
        return np.clip(a, 0, 1).astype(np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1).astype(np.float32)


def _engine_multi_directional_hillshading(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Hillshade from eight compass directions at one altitude, as either

    * **blend** (``mdh_mode`` 0): the eight shaded pictures averaged, contrast-
      stretched - nothing is left in shadow, and relief of every orientation reads; or
    * **PCA composite** (``mdh_mode`` 1): the eight pictures are strongly
      correlated, so their principal components separate what they share
      (overall relief, PC1 - red) from how the shading changes with direction
      (PC2 - green, PC3 - blue).  False colour: ridges and grooves running in
      different directions take different hues.  (Devereux, Miller & Rowe 2008,
      "Multi-directional hillshades by PCA".)

    Independent of the light slider; ``mdh_alt`` is the altitude of all eight lights.
    """
    n = ctx['n_std']
    if n is None or n.ndim != 3:
        return np.zeros((h, w), np.float32)
    if n.shape[:2] != (h, w):
        h, w = n.shape[:2]
    alt = float(s.get('mdh_alt', 35.0))
    mode = int(s.get('mdh_mode', 0))
    gain = max(0.01, float(ctx['gain']))
    N = _unit_normals_flat(n)
    lights = _mdh_lights(alt)

    def make_blend():
        if _rust("multi_lambert", N):
            total, _ = _rs.multi_lambert(N, lights, np.ones(len(lights), np.float32))
        else:
            total = np.zeros(N.shape[0], np.float32)
            for L in lights:
                total += np.maximum(0.0, N @ L)
        total /= np.float32(len(lights))
        return _stretch01(total.reshape(h, w))

    def make_pca():
        X = np.maximum(N @ lights.T, 0.0)                      # (pixels, 8) hillshades
        X -= X.mean(axis=0, keepdims=True)
        cov = (X.T @ X).astype(np.float64) / max(1, X.shape[0] - 1)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1][:3]
        V = vecs[:, order]
        for k in range(V.shape[1]):                            # a fixed sign, so the colours do not flip
            ref = V[:, k].sum() if k == 0 else V[np.argmax(np.abs(V[:, k])), k]
            if ref < 0:
                V[:, k] = -V[:, k]
        pcs = (X @ V.astype(np.float32)).reshape(h, w, V.shape[1])
        out = np.empty((h, w, 3), np.float32)
        out[:] = 0.5
        for k in range(V.shape[1]):
            out[..., k] = _stretch01(pcs[..., k])
        return out

    # Light-free, so worked out once per solve and altitude; only the gain follows the slider.
    base = cached("mdh-pca" if mode == 1 else "mdh-blend", (n,), (alt, h, w),
                  make_pca if mode == 1 else make_blend)
    if gain == 1.0:
        return base
    return np.power(base, np.float32(1.0 / gain))


# ── Ridge and Groove Detector (2026-10) ───────────────────────────────────

_RIDGE_SCALES = (1.0, 1.6, 2.5)         # the line widths tried, times the width asked for
_RIDGE_BETA = 0.5                        # how round a blob may be before it stops counting as a line


def _gauss_kernels(sigma: float):
    """Gaussian and its first and second derivative, as 1-D float32 kernels."""
    r = max(2, int(np.ceil(3.5 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-x * x / (2.0 * sigma * sigma))
    g /= g.sum()
    return (g.astype(np.float32), (-x / (sigma * sigma) * g).astype(np.float32),
            ((x * x / sigma ** 4 - 1.0 / sigma ** 2) * g).astype(np.float32))


def ridge_response(depth: np.ndarray, width: float, scale: float = 1.0):
    """Lines in a height map, found where the surface curves sharply across one direction only.

    The Hessian of the height at three widths (``width`` x 1, 1.6, 2.5, in full-size pixels;
    ``scale`` is the size of *depth* relative to the full-size picture) gives, at each pixel,
    the curvature across and along the strongest bend.  A raised line (ridge) bends strongly
    down across it, a groove strongly up; a round bump or pit bends both ways and is
    suppressed (Frangi's blobness, beta 0.5).  Returns float32 ``ridge`` and ``groove``
    strength (the best over the widths, widths made comparable by sigma squared) and
    ``direction``, the line's orientation in radians, 0..pi, 0 along x.
    """
    d = np.ascontiguousarray(depth, np.float32)
    ridge = np.zeros(d.shape, np.float32)
    groove = np.zeros(d.shape, np.float32)
    th_r = np.zeros(d.shape, np.float32)
    th_g = np.zeros(d.shape, np.float32)
    for k in _RIDGE_SCALES:
        sg = max(0.6, float(width) * k * scale)
        g0, g1, g2 = _gauss_kernels(sg)
        hxx = cv2.sepFilter2D(d, cv2.CV_32F, g2, g0, borderType=cv2.BORDER_REFLECT)
        hyy = cv2.sepFilter2D(d, cv2.CV_32F, g0, g2, borderType=cv2.BORDER_REFLECT)
        hxy = cv2.sepFilter2D(d, cv2.CV_32F, g1, g1, borderType=cv2.BORDER_REFLECT)
        s2 = np.float32(sg * sg)
        if _rust("ridge_update", hxx, hxy, hyy, ridge, groove, th_r, th_g):
            _rs.ridge_update(hxx, hxy, hyy, float(s2), float(2.0 * _RIDGE_BETA ** 2), ridge, groove, th_r, th_g)
            continue
        m = (hxx + hyy) * np.float32(0.5)
        q = np.hypot((hxx - hyy) * np.float32(0.5), hxy)
        l1, l2 = m + q, m - q                                   # l1 >= l2
        theta = np.float32(0.5) * np.arctan2(np.float32(2.0) * hxy, hxx - hyy)   # the l1 axis
        for a, b, acc, th in ((-l2, l1, ridge, th_r), (l1, l2, groove, th_g)):
            a = a * s2
            b = np.abs(b) * s2
            ok = a > 0
            rb = b / np.where(ok, a, np.float32(1.0))
            resp = np.where(ok, a * np.exp(-(rb * rb) / np.float32(2.0 * _RIDGE_BETA ** 2)), np.float32(0.0))
            better = resp > acc
            acc[better] = resp[better]
            th[better] = theta[better]
    # the line runs along the l1 axis for a ridge, across it for a groove
    direction = np.where(ridge >= groove, th_r, th_g + np.float32(np.pi / 2))
    direction = np.mod(direction, np.float32(np.pi)).astype(np.float32)
    return ridge, groove, direction


def _ridge_norm(data, width: float) -> float:
    """The strength drawn as white: worked out once on the preview, so a region drawn at
    full size is not stretched to its own contents."""
    dp = getattr(data, 'depth_proxy_p', None)
    if dp is None:
        return 1.0
    sc = dp.shape[1] / float(max(getattr(data, 'w', dp.shape[1]), 1))

    def make():
        r, g, _ = ridge_response(dp, width, sc)
        v = np.maximum(r, g)
        return float(quantiles(v, [99.5])[0]) + 1e-12
    return cached("ridge-norm", (dp,), (round(width, 4), round(sc, 5)), make)


def _engine_ridge_detector(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Incised lines, scratches, tool marks and ruling found as lines, not as shading.

    * ``rdg_width`` - the width of the thinnest line looked for, in pixels of the full-size
      picture (the same at any zoom); two wider ones are tried as well.
    * ``rdg_mode`` - 0 ridges bright and grooves dark on grey; 1 grooves only (bright on
      black); 2 ridges only; 3 false colour: hue is the direction the line runs, brightness
      its strength.
    Independent of the light; only the gain and these follow the sliders.
    """
    depth_p = ctx['depth_p']
    gain = max(0.05, float(ctx['gain']))
    if getattr(data, 'depth_vis_p', None) is None or depth_p is None:
        return np.zeros((h, w), np.float32)
    width = float(np.clip(s.get('rdg_width', 1.5), 0.5, 12.0))
    mode = int(s.get('rdg_mode', 0))
    d = depth_p if depth_p.shape[:2] == (h, w) else cv2.resize(np.asarray(depth_p, np.float32), (w, h))
    scale = (w / float(max(data.w, 1))) if ctx['use_proxy'] else 1.0
    ridge, groove, direction = cached("ridge", (depth_p,), (round(width, 4), round(scale, 5), h, w),
                                      lambda: ridge_response(d, width, scale))
    norm = np.float32(_ridge_norm(data, width) / gain)
    if mode == 1:
        return np.clip(groove / norm, 0, 1).astype(np.float32)
    if mode == 2:
        return np.clip(ridge / norm, 0, 1).astype(np.float32)
    if mode == 3:
        v = np.clip(np.maximum(ridge, groove) / norm, 0, 1) ** np.float32(0.7)
        hsv = np.dstack([direction * np.float32(360.0 / np.pi), np.ones_like(v), v]).astype(np.float32)
        return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1).astype(np.float32)
    return np.clip(np.float32(0.5) + np.float32(0.5) * (ridge - groove) / norm, 0, 1).astype(np.float32)


def _engine_clay_mode(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Albedo-suppressed raking light: the surface as unpainted clay.

    Plain Lambertian shading of the solved normals under the dome's light, with a
    uniform white albedo - the picture has no colour at all, so stains,
    discoloured varnish, foxing and pigment cannot be mistaken for form.  It
    returns RGB (grey in three channels), which the compositor shows as it is,
    so the colour layer cannot bring the albedo back.

    * ``clay_floor`` - the darkest any pixel may be (0-0.4).  Raised, it keeps the
      micro-roughness in deep shadow from being lost to black.
    * ``clay_micro`` - micro-roughness emphasis: the shading's detail finer than
      ~0.3% of the long side is boosted by this factor (0 = plain Lambert).
    """
    n = ctx['n_std']
    if n is None or n.ndim != 3:
        return np.zeros((h, w, 3), np.float32)
    if n.shape[:2] != (h, w):
        h, w = n.shape[:2]
    L = np.asarray(ctx['light'], np.float64)
    L = (L / max(1e-6, float(np.linalg.norm(L)))).astype(np.float32)
    floor = float(np.clip(s.get('clay_floor', 0.06), 0.0, 0.4))
    micro = max(0.0, float(s.get('clay_micro', 0.6)))
    gain = max(0.01, float(ctx['gain']))
    N = _unit_normals_flat(n)
    if _rust("clay_lambert", N):
        d = _rs.clay_lambert(N, L).reshape(h, w)
    else:
        d = np.empty((h, w), np.float32)

        def band(y0, y1):
            d[y0:y1] = np.maximum(0.0, N[y0 * w:y1 * w] @ L).reshape(y1 - y0, w)
        by_rows(band, h, w)
    low = cv2.GaussianBlur(d, (0, 0), max(1.0, 0.003 * max(h, w))) if micro > 0.0 else None
    if _rust("clay_finish", d):
        return _rs.clay_finish(d, np.float32(micro), floor, np.float32(1.0 / gain), low=low)
    if low is not None:
        d = d + (d - low) * np.float32(micro)
    grey = np.clip(d, 0.0, 1.0)
    grey = floor + (1.0 - floor) * grey
    if gain != 1.0:
        grey = np.power(grey, np.float32(1.0 / gain))
    return np.repeat(grey[..., None], 3, axis=2).astype(np.float32, copy=False)


def _engine_rti_specular_enhancement(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    n_std = ctx['n_std']
    light = ctx['light']
    gain = ctx['gain']
    ptm = ctx.get('ptm')
    med = ctx.get('med')
    _sc   = float(s.get('rti_spec_coeff', 1.5))
    _sfl  = float(s.get('rti_spec_floor', 0.05))
    _sbl  = float(s.get('rti_spec_blend', 0.6))
    _g    = max(0.01, gain)
    flat = _flat3(n_std)
    mode = "diffuse"
    if ((getattr(data, 'ptm_coeffs', None) is not None or getattr(data, 'ptm_deferred', False))
            and ptm is not None):
        if ptm.ndim == 2 and ptm.shape[0] >= 6 and ptm.shape[1] == h * w:
            mode = "ptm"
        else:
            _log.warning("PTM relighting failed, falling back to diffuse only")
    elif med is not None and med.size > 0:
        mode = "median"
    if _rust("rti_specular", flat):
        # The PTM coefficients may be C- or Fortran-ordered (the preview's are
        # a transposed view); the kernel reads either without a copy.
        src = ptm if mode == "ptm" else med if mode == "median" else None
        if src is None or ((src.flags.c_contiguous or (mode == "ptm" and src.flags.f_contiguous))
                           and src.dtype in ((np.float32, np.float64) if mode == "ptm"
                                             else (np.uint8, np.float32))):
            return _rs.rti_specular(
                flat, _f32(light), np.float32(_sfl), np.float32(_sc), np.float32(_sbl), np.float32(1.0 / _g),
                ptm=ptm if mode == "ptm" else None, med=med if mode == "median" else None,
            ).reshape(h, w)
    out = np.empty((h, w), np.float64)
    u, v = light[0], light[1]

    def band(y0, y1):
        _diff = np.maximum(0, np.dot(flat[y0 * w:y1 * w], light)).reshape(y1 - y0, w)
        if mode == "ptm":
            _predicted = compute_ptm_relighting(
                ptm[:, y0 * w:y1 * w], u, v).reshape(y1 - y0, w).astype(np.float32)
            _predicted = np.clip(_predicted / 255.0, 0, 1)
        elif mode == "median":
            _predicted = np.clip(med[y0:y1].astype(np.float32) / 255.0, 0, 1)
        else:
            _predicted = _diff
        _spec_raw  = np.maximum(0.0, _predicted - _diff)
        _spec_mask = (_spec_raw > _sfl).astype(np.float32)
        _spec_amp  = _spec_raw * _spec_mask * _sc
        _out = _diff * (1.0 - _sbl) + np.clip(_diff + _spec_amp, 0, 1) * _sbl
        out[y0:y1] = np.power(np.clip(_out, 0, 1), 1.0 / _g)
    by_rows(band, h, w)
    return out


def _engine_rti_image_usm(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    n_std = ctx['n_std']
    light = ctx['light']
    gain = ctx['gain']
    _sig   = max(0.3, float(s.get('rti_usm_sigma',  2.0)))
    _amt   = float(s.get('rti_usm_amount', 1.5))
    _thr   = float(s.get('rti_usm_thresh', 0.0))
    _g     = max(0.01, gain)
    flat = _flat3(n_std)
    _base = np.empty((h, w), np.float32)

    def base(y0, y1):
        b = np.maximum(0, np.dot(flat[y0 * w:y1 * w], light)).reshape(y1 - y0, w)
        _base[y0:y1] = np.power(b, 1.0 / _g)
    by_rows(base, h, w)
    _blur  = cv2.GaussianBlur(_base, (0, 0), sigmaX=_sig)
    out = np.empty((h, w), np.float32)

    def sharpen(y0, y1):
        _edge = _base[y0:y1] - _blur[y0:y1]
        if _thr > 1e-4:
            _edge_mask = (np.abs(_edge) > _thr).astype(np.float32)
            _edge = _edge * _edge_mask
        out[y0:y1] = np.clip(_base[y0:y1] + _edge * _amt, 0, 1)
    by_rows(sharpen, h, w)
    return out


def _engine_bilateral_normal_sharpening(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    if getattr(data, 'normals_full', None) is None:
        return np.zeros((h, w))
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    crop_coords = ctx.get('crop_coords')
    bil_d = max(5, int(s.get('bil_d', 9)))
    bil_sc = max(0.01, s.get('bil_sc', 0.1))

    def make():
        src = (data.normals_proxy if use_proxy else data.normals_full).astype(np.float32)
        if not use_proxy and crop_coords:
            x1b,y1b,x2b,y2b = crop_coords; src = src[y1b:y2b, x1b:x2b]
        if src.shape[:2] != (h, w): src = cv2.resize(src, (w, h))
        sharp = np.zeros_like(src)
        for ch_i in range(3):
            smoothed = cv2.bilateralFilter(src[...,ch_i], d=bil_d, sigmaColor=bil_sc, sigmaSpace=5)
            sharp[...,ch_i] = src[...,ch_i] + (src[...,ch_i] - smoothed) * gain * 2.0
        norm = np.linalg.norm(sharp, axis=-1, keepdims=True)
        return np.divide(sharp, norm, out=np.zeros_like(sharp), where=norm>0)
    if use_proxy and getattr(data, 'normals_proxy', None) is not None:
        sharp = cached("bilateral-normals", (data.normals_proxy,),
                       (bil_d, float(bil_sc), float(gain), h, w), make)
    else:
        sharp = make()
    # ctx['light'] is +Y up like the normals; s['ly'] is the screen's +Y down,
    # which lit south-facing slopes for a light set at the top (KD-014).
    lv = np.asarray(ctx['light'], np.float64)
    flat = _flat3(sharp)
    out = np.empty((h, w), np.result_type(flat, lv))

    def band(y0, y1):
        out[y0:y1] = np.clip(np.maximum(0, np.dot(flat[y0 * w:y1 * w], lv)
                                        .reshape(y1 - y0, w)), 0, 1)
    by_rows(band, h, w)
    return out


def _engine_sss(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    if getattr(data, 'albedo_rgb_full', None) is None:
        return np.zeros((h, w), np.float32)
    n_std = ctx['n_std']
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    crop_coords = ctx.get('crop_coords')
    _sss_r = float(s.get('sss_radius', 8.0)) * gain
    _sss_b = float(s.get('sss_blend', 0.45))
    lv = np.asarray(ctx['light'], np.float64)       # +Y up, as the normals (KD-014)
    flat = _flat3(n_std)
    if use_proxy:
        _alb_src = data.albedo_proxy
    else:
        _ax1, _ay1, _ax2, _ay2 = crop_coords if crop_coords else (0,0,data.w,data.h)
        _alb_src = data.albedo_rgb_full[_ay1:_ay2, _ax1:_ax2]
    if _alb_src is None or _alb_src.size == 0:
        return np.clip(np.abs(np.dot(flat, lv).reshape(h, w)), 0, 1)
    _sig = max(1.0, _sss_r)

    def make():
        _alb_s = _alb_src.astype(np.float32) / 255.0
        if _alb_s.shape[:2] != (h, w):
            _alb_s = cv2.resize(_alb_s, (w, h), interpolation=cv2.INTER_LINEAR)
        _scattered = cv2.GaussianBlur(_alb_s, (0,0), sigmaX=_sig)
        return 0.299*_scattered[...,0] + 0.587*_scattered[...,1] + 0.114*_scattered[...,2]
    _scat_lum = (cached("sss-scatter", (_alb_src,), (_sig, h, w), make) if use_proxy
                 else make())
    if _rust("sss", flat, _scat_lum) and lv.shape == (3,):
        return _rs.sss(flat, lv, _scat_lum, _sss_b).reshape(h, w)
    out = np.empty((h, w), np.float32)

    def band(y0, y1):
        _diff = np.clip(np.abs(np.dot(flat[y0 * w:y1 * w], lv).reshape(y1 - y0, w)), 0, 1)
        _sss_shading = _diff * (1.0 - _sss_b) + _scat_lum[y0:y1] * _sss_b * _diff
        out[y0:y1] = np.clip(_sss_shading.astype(np.float32), 0, 1)
    by_rows(band, h, w)
    return out


def _engine_normal_residual(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    if getattr(data, 'normals_full', None) is None:
        return np.zeros((h, w), np.float32)
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    crop_coords = ctx.get('crop_coords')
    NR_MAX_PX = 3000
    nr_scale = float(s.get('nr_scale', 5.0))
    nr_gain  = float(s.get('nr_gain', 1.5))
    nr_mode  = int(s.get('nr_mode', 0))
    if use_proxy:
        normals_src = (getattr(data, 'normals_proxy', None)
                       if getattr(data, 'normals_proxy', None) is not None
                       else data.normals_full)
    else:
        normals_src = data.normals_full
        if crop_coords:
            x1c, y1c, x2c, y2c = crop_coords
            normals_src = normals_src[y1c:y2c, x1c:x2c]
    H_src, W_src = normals_src.shape[:2]
    cap_scale = min(1.0, NR_MAX_PX / max(H_src, W_src))
    if cap_scale < 0.99:
        H_cap = max(4, int(H_src * cap_scale))
        W_cap = max(4, int(W_src * cap_scale))
        normals_cap = cv2.resize(normals_src.astype(np.float32),
                                 (W_cap, H_cap), interpolation=cv2.INTER_AREA)
    else:
        H_cap, W_cap = H_src, W_src
        normals_cap = normals_src.astype(np.float32)
    def make_shade():
        sigma = max(1.0, H_cap * nr_scale / 100.0)
        Nx = normals_cap[..., 0]
        Ny = normals_cap[..., 1]
        Nx_hp = Nx - cv2.GaussianBlur(Nx, (0, 0), sigma)
        Ny_hp = Ny - cv2.GaussianBlur(Ny, (0, 0), sigma)
        import math as _math
        frames = []
        for az_d in [0, 45, 90, 135, 180, 225, 270, 315]:
            az_r = _math.radians(az_d)
            lx = float(_math.cos(_math.radians(15)) * _math.sin(az_r))
            ly = float(-_math.cos(_math.radians(15)) * _math.cos(az_r))
            frames.append(Nx_hp * lx + Ny_hp * ly)
        shade = np.max(np.abs(np.stack(frames, axis=0)), axis=0).astype(np.float32)
        shade = shade * nr_gain
        if (H_cap, W_cap) != (h, w):
            shade = cv2.resize(shade, (w, h), interpolation=cv2.INTER_LINEAR)
        return shade
    # Two wide blurs and eight directions (~4.5 s at 2500 px) that depend on the
    # solved normals and two settings only - not on the light, the gain or the
    # display mode - so they are done once per solve.  (A crop is a new view of
    # the normals each time, so it is not kept.)
    shade_2d = cached("normal-residual", (normals_src,), (nr_scale, nr_gain, h, w), make_shade)
    if nr_mode == 0:
        p1, p99 = quantiles(shade_2d, [1, 99])
        rng = max(p99 - p1, 1e-8)
        result = np.clip((shade_2d - p1) / rng, 0, 1)
        result = 1.0 - result
        return result.astype(np.float32)
    elif nr_mode == 1:
        p1, p99 = quantiles(shade_2d, [2, 98])
        half = max(abs(p1), abs(p99), 1e-6)
        shade_n = np.clip(shade_2d / half, -1.0, 1.0)
        pos = np.clip(shade_n, 0, 1)
        neg = np.clip(-shade_n, 0, 1)
        R = 0.5 + pos * 0.5 - neg * 0.3
        G = 0.5 - np.abs(shade_n) * 0.4
        B = 0.5 + neg * 0.5 - pos * 0.3
        rgb = np.stack([
            np.clip(R, 0, 1).astype(np.float32),
            np.clip(G, 0, 1).astype(np.float32),
            np.clip(B, 0, 1).astype(np.float32)
        ], axis=-1)
        return np.clip(rgb * gain, 0, 1)
    else:
        result = np.abs(shade_2d)
        p1r, p99r = quantiles(result, [1, 99])
        rng  = max(p99r - p1r, 1e-6)
        result = np.clip((result - p1r) / rng, 0, 1)
        return np.clip(result.astype(np.float32) * gain, 0, 1)


def _plate_frac(data: DataManager, ctx: ShadingCtx):
    """The part of each photograph a view that rereads them from disk should
    show, as fractions (x1, y1, x2, y2), or None for the whole frame.

    The magnifier's or selection's crop is in the solved picture's pixels; a
    region solve (Reprocess with a crop) is itself a part of the photographs,
    so the two are composed.  Without the second, these views showed the
    whole photograph after a region was solved."""
    crop_coords = ctx.get('crop_coords')
    pretrimmed = ctx.get('_pretrimmed_crop_frac')
    frac = None
    if not ctx['use_proxy'] and crop_coords:
        frac = (crop_coords[0] / max(data.w, 1), crop_coords[1] / max(data.h, 1),
                crop_coords[2] / max(data.w, 1), crop_coords[3] / max(data.h, 1))
    elif pretrimmed is not None:
        frac = tuple(pretrimmed)
    region = getattr(data, 'solve_region', None)
    if region is None:
        return frac
    if frac is None:
        return tuple(region)
    rx1, ry1, rx2, ry2 = region
    rw, rh = rx2 - rx1, ry2 - ry1
    return (rx1 + frac[0] * rw, ry1 + frac[1] * rh, rx1 + frac[2] * rw, ry1 + frac[3] * rh)


def _engine_colour_ica(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Independent components of the frames' colour, IC1-3 as R, G, B.

    Every pixel is the same linear mix of the frames' 3N colour channels
    (centred, whitened to unit variance in five principal components, then
    rotated by FastICA), so the mix is learned from a sample of pixels and
    applied to
    all of them at the size asked for.  It used to be learned and applied
    on a copy 400 px across and stretched up, which made the view soft and
    blocky, and to decode every photograph whole on each draw, which made it
    slow; the frames are now read at the size needed (JPEG's DCT scaling,
    several at once) and the result is kept."""
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    _cica_frac = _plate_frac(data, ctx)
    plate_paths = list(getattr(data, 'plate_paths', []))
    cache_key = ('cica', use_proxy, len(plate_paths), _cica_frac)
    if getattr(data, '_cica_cache', None) is not None:
        _cc = data._cica_cache
        if _cc.get('key') == cache_key and _cc.get('shape') == (h, w):
            return np.clip(_cc['lum'] * gain, 0, 1)

    def read(p):
        # The smallest JPEG reduction whose crop is still at least (w, h).
        k = 1
        try:
            from helio.core.sphere import image_size
            sz = image_size(p)
        except Exception:
            sz = None
        if sz:
            fw = (_cica_frac[2] - _cica_frac[0]) if _cica_frac else 1.0
            fh = (_cica_frac[3] - _cica_frac[1]) if _cica_frac else 1.0
            while k < 8 and sz[0] * fw / (k * 2) >= w and sz[1] * fh / (k * 2) >= h:
                k *= 2
        flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2,
                4: cv2.IMREAD_REDUCED_COLOR_4, 8: cv2.IMREAD_REDUCED_COLOR_8}[k]
        im = read_frame(p, flag)
        if im is None:
            return None
        if _cica_frac:
            ih_d, iw_d = im.shape[:2]
            im = im[int(_cica_frac[1]*ih_d):int(_cica_frac[3]*ih_d),
                    int(_cica_frac[0]*iw_d):int(_cica_frac[2]*iw_d)]
            if im.size == 0:
                return None
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        return im                                               # uint8 BGR, (h, w)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(plate_paths), os.cpu_count() or 2))) as ex:
        imgs = [im for im in ex.map(read, plate_paths) if im is not None]
    if len(imgs) < 3:
        return np.zeros((h, w), np.float32)
    n_px = h * w
    # (pixels, 3N) in RGB order per frame, as before
    X = np.empty((n_px, 3 * len(imgs)), np.float32)
    for f, im in enumerate(imgs):
        X[:, 3 * f:3 * f + 3] = im.reshape(-1, 3)[:, ::-1].astype(np.float32) * (1.0 / 255.0)
    del imgs
    rng = np.random.default_rng(42)
    sample = X[rng.choice(n_px, size=min(n_px, 200_000), replace=False)]
    Sc = sample - sample.mean(axis=1, keepdims=True)            # each pixel centred over channels
    n_pcs = min(5, Sc.shape[1])
    C = (Sc.T @ Sc) / max(Sc.shape[0] - 1, 1)
    ev2, evec2 = np.linalg.eigh(C)
    # eigenvectors have no sign of their own; fix it (largest loading positive) so the
    # whitened data, and so the whole fit that follows, is the same on every machine
    evec2 = evec2 * np.where(evec2[np.argmax(np.abs(evec2), axis=0), np.arange(evec2.shape[1])] >= 0, 1.0, -1.0).astype(evec2.dtype)
    ev2_k = np.maximum(ev2[::-1][:n_pcs], 1e-10)
    evec2_k = evec2.T[::-1][:n_pcs]
    Wh = (evec2_k / np.sqrt(ev2_k)[:, None]).astype(np.float32)  # (n_pcs, 3N) whitening
    K = Wh @ Sc.T
    np.random.seed(42)
    W_ica = np.random.randn(n_pcs, n_pcs).astype(np.float32)
    for i in range(n_pcs): W_ica[i] /= np.linalg.norm(W_ica[i]) + 1e-9
    for it in range(60):
        wx = W_ica @ K; g = np.tanh(wx); g_d = 1.0 - g**2
        W_new = (g @ K.T) / K.shape[1] - g_d.mean(axis=1, keepdims=True) * W_ica
        ev3, evec3 = np.linalg.eigh(W_new @ W_new.T)
        W_ica = (evec3 / np.sqrt(np.maximum(ev3, 1e-12)) @ evec3.T) @ W_new
        if it > 5 and max(abs(abs(np.diag(W_ica @ W_ica.T)) - 1)) < 5e-4:
            break
    # Which three to show.  The order FastICA returns is arbitrary, so pick
    # by meaning: leave out the component that follows the pixels' overall
    # brightness (the albedo, already the photograph), and of the rest show
    # the three most non-Gaussian (largest |excess kurtosis|): the ones with
    # the most distinct structure, strongest first as R, G, B.
    ics_all = (W_ica @ K)                                       # (n_pcs, sample)
    bright = sample.mean(axis=1)
    corr = [abs(float(np.corrcoef(ic, bright)[0, 1])) if np.std(ic) > 0 else 0.0
            for ic in ics_all]
    albedo = int(np.argmax(corr))
    z = (ics_all - ics_all.mean(axis=1, keepdims=True)) / (ics_all.std(axis=1, keepdims=True) + 1e-9)
    kurt = np.abs((z ** 4).mean(axis=1) - 3.0)
    ic_idx = [int(i) for i in np.argsort(-kurt) if int(i) != albedo][:3]
    while len(ic_idx) < 3: ic_idx.append(ic_idx[-1])
    M = (W_ica[ic_idx] @ Wh).astype(np.float32)                 # (3, 3N): the whole mix
    ics_s = (Sc @ M.T)                                          # the sample, for the stretch
    # An independent component has no sign of its own (the maths library may give it or its
    # negative): show each with its rare strong pixels bright, so the same photographs always
    # give the same picture.
    from helio.core.processing import skew_sign
    flip = np.array([skew_sign(ics_s[:, j], M[j]) for j in range(3)], np.float32)
    M = M * flip[:, None]
    ics_s = ics_s * flip[None, :]
    lo = np.percentile(ics_s, 1, axis=0)
    hi = np.percentile(ics_s, 99, axis=0)
    out = np.empty((n_px, 3), np.float32)
    step = 1 << 20
    for a in range(0, n_px, step):                              # in blocks: bounded memory
        xb = X[a:a + step]
        xb = xb - xb.mean(axis=1, keepdims=True)
        out[a:a + step] = (xb @ M.T - lo) / (hi - lo + 1e-9)
    del X
    ica_rgb = np.clip(out, 0, 1).reshape(h, w, 3)
    data._cica_cache = {'key': cache_key, 'shape': (h, w), 'lum': ica_rgb}
    return np.clip(ica_rgb * gain, 0, 1)


def _engine_rgb_ps_rb_diff(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    gain = ctx['gain']
    use_proxy = ctx['use_proxy']
    _rbd_frac = _plate_frac(data, ctx)
    plate_paths = getattr(data, 'plate_paths', [])
    _rbd_key = ('rbd', len(plate_paths), use_proxy, _rbd_frac)
    if getattr(data, '_rbd_cache', None) is not None:
        _rc = data._rbd_cache
        if _rc.get('key') == _rbd_key and _rc.get('shape') == (h, w):
            return np.clip(_rc['img'] * gain, 0, 1)
    if not plate_paths:
        return np.full((h, w), 0.5, dtype=np.float32)
    if getattr(data, 'current_L_mat', None) is None:
        return np.full((h, w), 0.3, dtype=np.float32)
    L = data.current_L_mat.astype(np.float32)
    eye3 = np.eye(3, dtype=np.float32) * 1e-6
    RBD_PX = 600 if use_proxy else min(1500, max(h, w))

    # Cache the plates WHOLE, never cropped.
    #
    # This used to cache each plate *after* cropping it to the requested
    # region, under a key that did not include the region. Every later request
    # then cropped the already-cropped image again, so dragging the magnifier
    # shrank the sampled area on each move until `im_rgb.size == 0`, every
    # plate was skipped, H2 stayed None and the engine returned a black tile.
    # That is the "magnifier shows a small dark patch, then nothing" report.
    #
    # Held as uint8 at up to RBD_CACHE_PX on the long side (~34 MB for eight
    # plates) rather than float32 at full size, and read from disk once per
    # capture instead of on every magnifier move - which is where this
    # engine's reputation for being CPU-heavy came from.
    RBD_CACHE_PX = 1400
    _plate_cache_key = (tuple(plate_paths), RBD_CACHE_PX)
    if getattr(data, '_rbd_plate_cache_key', None) != _plate_cache_key:
        data._rbd_plate_cache = {}
        data._rbd_plate_cache_key = _plate_cache_key
    cache = data._rbd_plate_cache

    def _read_small(p):
        im = read_frame(p, cv2.IMREAD_COLOR)
        if im is None:
            return None
        fh, fw = im.shape[:2]
        fsc = min(1.0, RBD_CACHE_PX / max(fh, fw))
        if fsc < 0.99:
            im = cv2.resize(im, (max(4, int(fw * fsc)), max(4, int(fh * fsc))),
                            interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)          # uint8, whole frame
    # The plates not yet held, read in parallel (the same pictures): one by one,
    # 32 photographs of 46 megapixels took 23 s the first time the view opened.
    missing = [p for p in plate_paths if cache.get(p) is None]
    if missing:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(missing))) as ex:
            for p, im in zip(missing, ex.map(_read_small, missing)):
                if im is not None:
                    cache[p] = im
    H2, W2 = None, None
    ch_imgs = {0: [], 2: []}
    for p in plate_paths:
        full = cache.get(p)
        if full is None:
            continue
        im_rgb = full
        if _rbd_frac:
            ih_d, iw_d = im_rgb.shape[:2]
            x1c = int(_rbd_frac[0] * iw_d); y1c = int(_rbd_frac[1] * ih_d)
            x2c = int(_rbd_frac[2] * iw_d); y2c = int(_rbd_frac[3] * ih_d)
            # Guard the degenerate crop rather than skipping the plate: a
            # sub-pixel magnifier region on a downscaled cache can round to an
            # empty slice, and skipping every plate is what produced black.
            x2c = max(x1c + 2, min(x2c, iw_d)); y2c = max(y1c + 2, min(y2c, ih_d))
            x1c = max(0, min(x1c, x2c - 2));    y1c = max(0, min(y1c, y2c - 2))
            im_rgb = im_rgb[y1c:y2c, x1c:x2c]
            if im_rgb.size == 0:
                continue
        ih, iw = im_rgb.shape[:2]
        sc = min(1.0, RBD_PX / max(ih, iw))
        if sc < 0.99:
            im_rgb = cv2.resize(im_rgb, (max(4, int(iw * sc)), max(4, int(ih * sc))),
                                interpolation=cv2.INTER_AREA)
        im_rgb = im_rgb.astype(np.float32)
        if H2 is None:
            H2, W2 = im_rgb.shape[:2]
        elif im_rgb.shape[:2] != (H2, W2):
            im_rgb = cv2.resize(im_rgb, (W2, H2), interpolation=cv2.INTER_AREA)
        ch_imgs[0].append(im_rgb[..., 0])
        ch_imgs[2].append(im_rgb[..., 2])
    if H2 is None: return np.zeros((h, w), np.float32)
    # One 3xN matrix multiply per channel instead of a Python loop solving the
    # same 3x3 system for every 65536-pixel block. The normal equations do not
    # depend on the pixels, so the pseudo-inverse is computed once.
    albedos = {}
    AtA = (L.T @ L) + eye3
    M = np.linalg.solve(AtA, L.T).astype(np.float32)     # (3, n_lights)
    for ci in [0, 2]:
        I_mat = np.stack([c.reshape(-1) for c in ch_imgs[ci]], axis=0).astype(np.float32)
        I_mat *= np.float32(1.0 / 255.0)
        G = M @ I_mat                                     # (3, n_pix)
        albedos[ci] = np.sqrt((G * G).sum(axis=0)).reshape(H2, W2)
    diff = albedos[0] - albedos[2]
    p2, p98 = np.percentile(diff, 2), np.percentile(diff, 98)
    rng = p98 - p2
    if rng < 1e-6: return np.zeros((h, w), np.float32)
    img_n = np.clip((diff - p2) / rng, 0, 1).astype(np.float32)
    blur = cv2.GaussianBlur(img_n, (0, 0), max(1.0, h * 0.01))
    img_n = np.clip(img_n + (img_n - blur) * 0.5, 0, 1)
    if (H2, W2) != (h, w):
        img_n = cv2.resize(img_n, (w, h), interpolation=cv2.INTER_LINEAR)
    data._rbd_cache = {'key': _rbd_key, 'shape': (h, w), 'img': img_n}
    return np.clip(img_n * gain, 0, 1)


# ── Engine registration (run after all _engine_xxx are defined) ──────

# ── Shape-depiction engines (2026-08-18) ──────────────────────────────
# These four come from the graphics literature rather than the terrain
# (RVT) tradition that most of the engines above follow. They depict form
# by re-lighting or modulating against local shape, not by deriving a
# terrain statistic from the depth map.

def _unrotate_normal(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``R(m)^T n``: each normal turned by the inverse of R(m), the rotation
    that takes +Z onto *m* (axis Z × m, cos θ = m_z).  Exaggerated shading
    lights each scale with R(m)L, the light re-expressed relative to that
    pixel's smoothed normal.

    Rodrigues by −θ about the same axis: v·cos θ − (k × v)·sin θ +
    k·(k·v)·(1 − cos θ), with k_z = 0.  Since n · R L == R^T n · L, lighting
    these with L gives exaggerated shading's per-scale term without turning
    the light at every pixel, which is what made it slow to drag.
    """
    ax_x, ax_y = -m[..., 1], m[..., 0]          # Z x m
    c = np.clip(m[..., 2], -1.0, 1.0)
    s_len = np.sqrt(ax_x * ax_x + ax_y * ax_y)
    small = s_len < 1e-8                         # m already parallel to Z
    inv = np.where(small, 0.0, 1.0 / np.maximum(s_len, 1e-12))
    kx, ky = ax_x * inv, ax_y * inv
    nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
    kdotn = kx * nx + ky * ny
    one_c = 1.0 - c
    out = np.empty(n.shape, np.float32)
    out[..., 0] = nx * c - (ky * nz) * s_len + kx * kdotn * one_c
    out[..., 1] = ny * c + (kx * nz) * s_len + ky * kdotn * one_c
    out[..., 2] = nz * c - (kx * ny - ky * nx) * s_len
    return np.where(small[..., None], n, out)


def _engine_exaggerated_shading(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Exaggerated Shading (Rusinkiewicz, Burns & DeCarlo, SIGGRAPH 2006).

    Shades the surface at several scales at once. At each scale the light is
    rotated into the frame of that scale's *smoothed* normal, so detail is lit
    as though its local neighbourhood were flat and facing the light. Fine
    scratches and broad undulation therefore stay legible in one image, where
    a single N.L render must sacrifice one for the other.

    Unlike Normals Unsharp Masking this is a lighting model, not a sharpening
    filter: it never manufactures contrast that the geometry does not contain.
    """
    n = ctx['n_std']
    if n is None or n.ndim != 3:
        return np.zeros((h, w), np.float32)
    if n.shape[:2] != (h, w):
        h, w = n.shape[:2]
    L = ctx['light'].astype(np.float32)
    L = L / max(1e-6, float(np.linalg.norm(L)))
    gain = max(0.01, ctx['gain'])
    n_scales = int(s.get('exag_scales', 4))
    base = float(s.get('exag_sigma', 2.0))
    scales = max(1, n_scales)
    # Flatten: the shading's broad tone (a page's curve, cockling, a warped
    # board, coarse scales lighting whole regions) taken out over this many
    # thousandths of the long side; 0 leaves the view as it was.  Turning the
    # normals first changes nothing here: each scale is already lit against
    # its own smoothed normal, so a common turn cancels.
    flat = float(s.get('exag_flatten', 0.0))

    def unrotated():
        # n . R(m)L == (R(m)^T n) . L, and R(m)^T n does not depend on the
        # light: keep it per scale, so a drag costs one dot product a scale.
        nn = n
        ps = []
        for i in range(scales):
            sigma = base * (2.0 ** i)
            m = cv2.GaussianBlur(nn, (0, 0), sigmaX=sigma, sigmaY=sigma)
            norm = np.linalg.norm(m, axis=-1, keepdims=True)
            m = m / np.maximum(norm, 1e-8)
            ps.append(np.ascontiguousarray(_unrotate_normal(nn, m), np.float32).reshape(-1, 3))
        return tuple(ps)
    ps = cached("exaggerated-unrotated", (n,), (scales, base), unrotated)

    if _rust("scaled_lambert", *ps):
        wgts = [1.0 / (i + 1.0) for i in range(len(ps))]     # finer scales dominate
        out = _rs.scaled_lambert(list(ps), _f32(L), _f32(wgts), np.float32(sum(wgts))).reshape(h, w)
    else:
        out = np.empty((h, w), np.float32)

        def shade(y0, y1):
            acc = np.zeros((y1 - y0) * w, np.float32)
            total = 0.0
            for i, p in enumerate(ps):
                sh = p[y0 * w:y1 * w] @ L
                np.maximum(sh, 0.0, out=sh)
                wgt = 1.0 / (i + 1.0)                # finer scales dominate
                acc += sh * wgt
                total += wgt
            out[y0:y1] = (acc / max(total, 1e-6)).reshape(y1 - y0, w)
        by_rows(shade, h, w)
    if flat > 0:
        sf = max(1.0, flat / 1000.0 * max(h, w))
        out -= cv2.GaussianBlur(out, (0, 0), sf) - float(np.mean(out))
    # A smooth region returns a *constant* here: where n == m exactly,
    # dot(n, R(m)L) == dot(Rz, RL) == L_z. That is the point of the method —
    # flat areas go mid-grey and only departures from local flatness show —
    # but it means a fixed contrast factor renders a mostly-flat page as
    # featureless grey (measured std 8.8 against Standard PBR's 43). Set the
    # stretch from the spread that is actually there, robustly, so the
    # deviation fills the range on any capture.
    med = median(out)
    dev = np.empty_like(out)

    def deviation(y0, y1):
        dev[y0:y1] = np.abs(out[y0:y1] - med)
    by_rows(deviation, h, w)
    spread = quantiles(dev, [98])[0]
    k = max(spread, 1e-6)

    def stretch(y0, y1):
        o = np.clip((out[y0:y1] - med) / k * 0.42 + 0.5, 0, 1)
        out[y0:y1] = np.power(o, 1.0 / gain)
    by_rows(stretch, h, w)
    return out


def _engine_radiance_scaling(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Radiance Scaling (Vergne et al. 2010) — curvature-modulated shading.

    Scales reflected radiance by a function of local curvature, brightening
    convexities and darkening concavities by an amount that depends on how
    curved they are rather than on how they happen to face the light. This is
    the shape-depiction operator MeshLab exposes, and it reads well on matte,
    low-relief surfaces where a plain N.L render goes flat.
    """
    n = ctx['n_std']
    if n is None or n.ndim != 3:
        return np.zeros((h, w), np.float32)
    if n.shape[:2] != (h, w):
        h, w = n.shape[:2]
    L = ctx['light'].astype(np.float32)
    L = L / max(1e-6, float(np.linalg.norm(L)))
    gain = max(0.01, ctx['gain'])
    strength = float(s.get('rs_strength', 6.0))
    sigma = float(s.get('rs_sigma', 1.5))

    shading = np.clip(np.sum(n * L.reshape(1, 1, 3), axis=-1), 0.0, 1.0)
    Nx = cv2.GaussianBlur(n[..., 0], (0, 0), sigmaX=sigma)
    Ny = cv2.GaussianBlur(n[..., 1], (0, 0), sigmaX=sigma)
    curv = normal_divergence(Nx, Ny)
    # Normalise curvature robustly to [-1, 1]; a few outliers must not set
    # the scale or the whole image collapses to mid-grey.
    k = float(np.percentile(np.abs(curv), 99))
    kh = np.clip(curv / max(k, 1e-8), -1.0, 1.0)
    # Vergne's reflectance transform: sigma = (g*k + 1) / (g*k*shading + 1)
    scaled = shading * (strength * kh + 1.0) / (strength * kh * shading + 1.0)
    return np.power(np.clip(scaled, 0, 1), 1.0 / gain).astype(np.float32)


def _engine_curvedness(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Curvedness (Koenderink & van Doorn 1992) — the partner to Shape Index.

    Shape Index says *what kind* of shape a point is (pit, ridge, saddle) and
    is deliberately scale-free: it throws the magnitude away. Curvedness is
    that discarded magnitude, sqrt((k1^2 + k2^2)/2) — how strongly curved,
    regardless of kind. The two together describe local shape completely;
    HELIO carried only the first.

    Log-compressed, because curvature magnitude on a real capture spans
    several decades and a linear map shows only the sharpest edge.
    """
    depth_p = ctx['depth_p']
    if depth_p is None:
        return np.zeros((h, w), np.float32)
    gain = max(0.01, ctx['gain'])
    d = cv2.resize(depth_p.astype(np.float32), (w, h))
    if d.ndim == 3:
        d = d[..., 0]
    sm = float(s.get('curvedness_smooth', 1.5))
    if sm > 0:
        d = cv2.GaussianBlur(d, (0, 0), sigmaX=sm)
    Lx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    Ly = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    Lxx = cv2.Sobel(Lx, cv2.CV_32F, 1, 0, ksize=3)
    Lyy = cv2.Sobel(Ly, cv2.CV_32F, 0, 1, ksize=3)
    Lxy = cv2.Sobel(Lx, cv2.CV_32F, 0, 1, ksize=3)
    H_curv = -(Lxx + Lyy) * 0.5
    K_curv = Lxx * Lyy - Lxy ** 2
    disc = np.sqrt(np.maximum(H_curv ** 2 - K_curv, 0))
    k1, k2 = H_curv + disc, H_curv - disc
    C = np.sqrt((k1 * k1 + k2 * k2) * 0.5)
    C = np.log1p(C / max(float(np.percentile(C, 99)), 1e-8) * 40.0) / np.log(41.0)
    # The log map already lands in 0..1. Multiplying by a further 6 clipped
    # almost everything to white (measured mean 247/255) - gain alone, and
    # inverted so high curvature reads dark like every other relief engine
    # here rather than as a white-out.
    return np.clip(1.0 - C * gain, 0, 1).astype(np.float32)


def _engine_cast_shadow_raking(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Raking light with real cast shadows over the depth map.

    Every other relighting engine here computes N.L, which is *attached*
    shadow only: a pixel facing away from the light goes dark, but nothing
    ever blocks light from reaching a pixel that faces it. Real raking light
    across an incised or embossed surface is dominated by the long shadow the
    relief throws, and that cue is what a conservator actually reads.

    A pixel is in shadow when the surface within ``shadow_steps`` pixels of
    it, towards the light, rises above the light's elevation: some t in
    1..steps with z(p + t·u) − z(p) > t·tan(elevation).  The height field is
    turned so the light runs along the rows; there that test is one running
    maximum per row (z − x·tan, maximum over the next *steps* pixels, a
    morphological dilation), then the result is turned back (KD-055).

    It used to march 48 shifted copies of the height field per frame, and
    shifted them the wrong way: it looked for blockers on the side away from
    the light, so a ridge lit from the west darkened its west flank and left
    the east one bright.
    """
    n = ctx['n_std']
    depth_p = ctx['depth_p']
    if n is None or n.ndim != 3 or depth_p is None:
        return np.zeros((h, w), np.float32)
    if n.shape[:2] != (h, w):
        h, w = n.shape[:2]
    L = ctx['light'].astype(np.float32)
    L = L / max(1e-6, float(np.linalg.norm(L)))
    gain = max(0.01, ctx['gain'])
    relief = float(s.get('shadow_relief', 0.08))   # depth range as frac of width
    steps = max(1, int(s.get('shadow_steps', 48)))
    ambient = float(s.get('shadow_ambient', 0.18))

    zscale = relief * max(h, w)                    # depth in pixel units

    def height_field():
        d = cv2.resize(depth_p.astype(np.float32), (w, h))
        if d.ndim == 3:
            d = d[..., 0]
        rng = float(np.ptp(d))
        if rng > 1e-8:
            d = (d - d.min()) / rng
        return d * zscale
    z = cached("shadow-heights", (depth_p,), (h, w, relief), height_field)

    lx, ly, lz = float(L[0]), float(L[1]), float(L[2])
    horiz = math.hypot(lx, ly)
    if horiz < 1e-6:                               # light straight down: no shadows
        lit = np.ones((h, w), np.float32)
    else:
        # Towards the light, in image axes (rows run south; ly is +Y north).
        ux, uy = lx / horiz, -ly / horiz
        margin = _cast_shadow_margin(z, ux, uy, lz / horiz, steps)
        if _rust("shadow_lit", margin):
            lit = _rs.shadow_lit(margin, np.float32(ambient))
        else:
            lit = np.where(margin > 0, ambient, 1.0).astype(np.float32)
        lit = cv2.GaussianBlur(lit, (0, 0), sigmaX=0.8)   # soften stair-steps
    flat_n = _flat3(n)
    if _rust("shadow_shade", flat_n, lit):
        return _rs.shadow_shade(flat_n, _f32(L), lit, np.float32(1.0 / gain)).reshape(h, w)
    out = np.empty((h, w), np.float32)
    L3 = L.reshape(1, 1, 3)

    def shade(y0, y1):
        shading = np.maximum(0.0, np.sum(n[y0:y1] * L3, axis=-1))
        out[y0:y1] = np.power(np.clip(shading * lit[y0:y1], 0, 1), 1.0 / gain)
    by_rows(shade, h, w)
    return out


def _cast_shadow_mask(z: np.ndarray, ux: float, uy: float, tan_elev: float,
                      steps: int) -> np.ndarray:
    """True where the height field *z* hides the light coming from (ux, uy)
    (``_cast_shadow_margin`` > 0)."""
    return _cast_shadow_margin(z, ux, uy, tan_elev, steps) > 0


def _cast_shadow_margin(z: np.ndarray, ux: float, uy: float, tan_elev: float,
                        steps: int) -> np.ndarray:
    """How far the height field *z* rises above the light's ray, per pixel
    (float32, positive = hidden from the light coming from (ux, uy)).

    (ux, uy) is the unit direction towards the light in image axes; the
    light stands *tan_elev* above the horizon.  Rotating by R = [[ux, uy],
    [−uy, ux]] puts that direction along +x, so with g(x) = z(x) − x·tan the
    test "some t in 1..steps with z(x+t) − z(x) > t·tan" is
    "max(g(x+1 .. x+steps)) > g(x)": a dilation.  The margin, not the mask,
    is turned back (bilinear), so shadow edges stay sub-pixel.
    """
    h, w = z.shape
    rw = int(math.ceil(abs(ux) * w + abs(uy) * h)) + 4
    rh = int(math.ceil(abs(uy) * w + abs(ux) * h)) + 4
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    qx, qy = (rw - 1) / 2.0, (rh - 1) / 2.0
    M = np.float64([[ux, uy, qx - (ux * cx + uy * cy)],
                    [-uy, ux, qy - (-uy * cx + ux * cy)]])
    zr = cv2.warpAffine(z, M, (rw, rh), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    if _rust("shadow_margin", zr):               # tilt, window maximum and margin in one pass
        margin = _rs.shadow_margin(zr, np.float32(tan_elev), steps)
    else:
        ramp = (np.arange(rw, dtype=np.float32) * np.float32(tan_elev))[None, :]
        g = np.empty_like(zr)

        def tilt(y0, y1):
            np.subtract(zr[y0:y1], ramp, out=g[y0:y1])
        by_rows(tilt, rh, rw)
        ahead = cv2.dilate(g, np.ones((1, steps), np.uint8), anchor=(0, 0),
                           borderType=cv2.BORDER_CONSTANT, borderValue=-np.inf)
        margin = np.empty_like(g)                 # > 0: something ahead is higher
        margin[:, :-1] = ahead[:, 1:] - g[:, :-1]
        margin[:, -1] = -1.0
    return cv2.warpAffine(margin, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


# ── Stack-derived quality / legibility engines (2026-08-18) ───────────
# These read the uint8 maps computed once in _run_processing_body from the
# whole light stack (see processing.stack_quality_maps). They are the only
# engines here that are independent of the interactive light position - the
# question they answer is about the *capture*, not about a lighting choice.

def _qc_map(data, full_attr: str, proxy_attr: str, h: int, w: int,
            ctx: ShadingCtx, scale: float = 255.0) -> Optional[np.ndarray]:
    """Fetch a stack-derived map at the requested size, or None if absent.

    None is a real state, not an error: the maps do not exist before the first
    run, and the residual map does not exist at all on an uncalibrated solve.
    """
    m = getattr(data, proxy_attr, None) if ctx.get('use_proxy') else None
    if m is None:
        m = getattr(data, full_attr, None)
    if m is None:
        return None
    m = np.asarray(m)
    # Honour an active crop before resizing. These maps are stored whole and,
    # unlike every other engine here, are not derived from the ctx arrays that
    # Helio_Pro has already cropped - so resizing the full map to the
    # requested tile scaled the whole page into it. That is exactly what made
    # the magnifier show the full image instead of zooming: it calls
    # get_shading(y2-y1, x2-x1, False, crop_coords=(x1,y1,x2,y2)) and only
    # crop-aware engines were reading those coordinates.
    if ctx.get('_is_real_crop') and ctx.get('crop_coords'):
        x1, y1, x2, y2 = (int(v) for v in ctx['crop_coords'])
        mh, mw = m.shape[:2]
        # crop_coords are full-resolution pixels; the stored map may be the
        # proxy, so scale the box into the map's own grid before slicing.
        sx = mw / float(max(getattr(data, 'w', 0) or mw, 1))
        sy = mh / float(max(getattr(data, 'h', 0) or mh, 1))
        cx1, cy1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        cx2, cy2 = min(mw, int(round(x2 * sx))), min(mh, int(round(y2 * sy)))
        if cx2 - cx1 >= 2 and cy2 - cy1 >= 2:
            m = m[cy1:cy2, cx1:cx2]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return m.astype(np.float32) / scale


def _engine_normal_uncertainty(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """How sure each normal is: its 1-sigma angular uncertainty (FIX-011).

    From the spread of each pixel's own observations about its least-squares
    fit (helio.core.processing.normal_uncertainty_deg), drawn in cividis on a
    fixed scale from 0 to *unc_max* degrees, so two captures can be compared
    by eye.  Dark is sure; yellow is doubtful (shadow, highlight, movement).
    Needs calibrated lights and at least four frames.
    """
    m = _qc_map(data, 'qc_sigma_full', 'qc_sigma_proxy', h, w, ctx, scale=1.0)
    if m is None:
        return np.full((h, w), 0.5, np.float32)
    vmax = max(0.5, float(s.get('unc_max', 10.0)))
    import matplotlib.cm as _cm
    return _cm.cividis(np.clip(m / vmax, 0, 1))[..., :3].astype(np.float32)


def _engine_rbf_relight(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """The photographs themselves interpolated to the light (helio.core.rbf_relight).

    For what shines or glows: gold, metal, vellum, wax.  A highlight is where
    it was photographed, and as sharp; nothing is fitted to the normals.
    Returned as the relit brightness over the pixel's median (0.7 = as bright as
    its median, clipped at 1), so the Colour layer is the object relit in its own
    colours and Shading the light alone.
    """
    rbf = getattr(data, 'rbf', None)
    if rbf is None:
        return np.full((h, w), 0.5, np.float32)
    from helio.core import rbf_relight
    width = float(s.get('rbf_width', 1.2))
    smooth = float(s.get('rbf_smooth', 0.05))
    light = np.asarray(ctx['light'], np.float64)
    full = not ctx.get('use_proxy') and rbf.get('comps_full') is not None
    box = None
    if ctx.get('_is_real_crop') and ctx.get('crop_coords'):
        x1, y1, x2, y2 = (int(v) for v in ctx['crop_coords'])
        if full:
            box = (x1, y1, x2, y2)
        else:                                  # the box in the preview planes' own pixels
            mh, mw = rbf['mean_proxy'].shape
            sx = mw / float(max(getattr(data, 'w', 0) or mw, 1))
            sy = mh / float(max(getattr(data, 'h', 0) or mh, 1))
            cx1, cy1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
            cx2, cy2 = min(mw, int(round(x2 * sx))), min(mh, int(round(y2 * sy)))
            if cx2 - cx1 >= 2 and cy2 - cy1 >= 2:
                box = (cx1, cy1, cx2, cy2)
    out = rbf_relight.relight(rbf, light, width, smooth, full=full, box=box)
    if out.shape != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    med = ctx.get('med')
    if med is None or np.shape(med) != (h, w):
        med = rbf['mean_full'] if full and box is None else None
        if med is None or med.shape != (h, w):
            med = np.full((h, w), max(1.0, float(np.median(out))), np.float32)
    gain = max(0.01, float(ctx.get('gain', 1.0)))
    # 0..1 like every relighting view: a pixel lit as brightly as its median reads 0.7,
    # leaving room for highlights up to about 1.4 times the median before they clip.
    ratio = out / np.maximum(np.asarray(med, np.float32), 2.0)
    return np.clip(ratio * np.float32(0.7 * gain), 0.0, 1.0).astype(np.float32)


def _engine_multi_light_fusion(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """Contrast-weighted fusion across every light direction.

    Faded ink is legible under some light directions and invisible under
    others, and any single relit view has to pick one. A median or mean over
    the stack averages the good frames away with the bad. This weights each
    observation by local contrast and by how well exposed it was, so each
    pixel is drawn from whichever frames actually showed something there.

    Independent of the light slider by design - it is a summary of the whole
    capture, not a relighting.
    """
    m = _qc_map(data, 'ml_fusion_full', 'ml_fusion_proxy', h, w, ctx)
    if m is None:
        return np.full((h, w), 0.5, np.float32)
    gain = max(0.01, ctx['gain'])
    return np.clip(m * gain, 0, 1).astype(np.float32)


def _engine_fit_residual_qc(
    data: DataManager, s: dict, h: int, w: int, ctx: ShadingCtx,
) -> np.ndarray:
    """RMS residual of the per-pixel Lambertian fit.

    How far the observations sit from the model photometric stereo assumes.
    High residual means specular, shadowed or interreflected - the places
    where the recovered normal is least believable, and in practice where
    gilding, metal point and glazes announce themselves.

    Needs calibrated light vectors; returns flat mid-grey after an
    uncalibrated (PCA) solve, where there is no light matrix to fit against.
    """
    m = _qc_map(data, 'qc_resid_full', 'qc_resid_proxy', h, w, ctx)
    if m is None:
        return np.full((h, w), 0.5, np.float32)
    gain = max(0.01, ctx['gain'])
    floor = float(np.clip(s.get('psr_floor', 0.05), 0.0, 0.95))
    amount = float(s.get('psr_gain', 1.5))
    gamma = max(0.05, float(s.get('psr_gamma', 1.0)))
    # Subtract the floor *before* the gain. On a clean capture most of the
    # frame sits in a low residual haze; removing it first separates genuinely
    # non-Lambertian pixels from the noise, where raising the gain alone just
    # amplifies both together.
    r = np.clip((m - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
    r = np.power(r, gamma) * amount * gain
    return np.clip(r, 0, 1).astype(np.float32)



def _plate_grey_stack(data: DataManager, ctx: ShadingCtx, h: int, w: int):
    """The capture's frames as a grey stack, at *h* x *w*.

    Shares the whole-plate cache RGB-PS R-B Diff builds, so opening one of
    these after that one costs no disk reads - and inherits the rule that a
    magnifier bug already paid for once: **cache the plates whole and crop per
    request**, never cache them cropped.
    """
    plate_paths = getattr(data, 'plate_paths', []) or []
    if not plate_paths:
        return None
    frac = _plate_frac(data, ctx)

    CACHE_PX = 1400
    key = (tuple(plate_paths), CACHE_PX)
    if getattr(data, '_rbd_plate_cache_key', None) != key:
        data._rbd_plate_cache = {}
        data._rbd_plate_cache_key = key
    cache = data._rbd_plate_cache

    out = []
    for pth in plate_paths:
        full = cache.get(pth)
        if full is None:
            im = read_frame(pth, cv2.IMREAD_COLOR)
            if im is None:
                continue
            fh, fw = im.shape[:2]
            sc = min(1.0, CACHE_PX / max(fh, fw))
            if sc < 0.99:
                im = cv2.resize(im, (max(4, int(fw * sc)), max(4, int(fh * sc))),
                                interpolation=cv2.INTER_AREA)
            full = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            cache[pth] = full
        g = full
        if frac:
            ih, iw = g.shape[:2]
            x1 = int(frac[0] * iw); y1 = int(frac[1] * ih)
            x2 = int(frac[2] * iw); y2 = int(frac[3] * ih)
            x2 = max(x1 + 2, min(x2, iw)); y2 = max(y1 + 2, min(y2, ih))
            x1 = max(0, min(x1, x2 - 2));  y1 = max(0, min(y1, y2 - 2))
            g = g[y1:y2, x1:x2]
            if g.size == 0:
                continue
        g = cv2.cvtColor(g, cv2.COLOR_RGB2GRAY)
        out.append(cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA))
    if len(out) < 2:
        return None
    return np.stack(out).astype(np.float32) / 255.0


_ENGINE_DISPATCH.update({
    "Standard PBR": _engine_standard_pbr,
    "Matte Relight (as 3D)": _engine_matte_relight,
    "Robust PBR (MVS)": _engine_robust_pbr,
    "Slope Magnitude": _engine_slope_magnitude,
    "Normals Unsharp Masking": _engine_normals_usm,
    "Specular Enhancement": _engine_spec_enhance,
    "Ambient Occlusion (AO)": _engine_ambient_occlusion,
    "False-Colour Depth": _engine_false_colour_depth,
    "Light Coverage Map": _engine_light_coverage,
    "Normals RGB": _engine_normals_rgb,
    "Normal Uncertainty (QC)": _engine_normal_uncertainty,
    "Shape Index": _engine_shape_index,
    "Positive Openness": _engine_pos_openness,
    "Negative Openness": _engine_neg_openness,
    "Local Dominance": _engine_local_dominance,
    "Sky-View Factor (SVF)": _engine_sky_view_factor,
    "PS Normal-Map Lit (8-Dir)": _engine_ps_normals_lit_8dir,
    "PS Normal-Map Lit (Single)": _engine_ps_normals_lit_single,
    "Diffuse Gain": _engine_diffuse_gain,
    "Surface Curvature (Scratches)": _engine_surface_curvature,
    "Slope Gradient": _engine_slope_gradient,
    "Multi-Scale Relief Model (MSRM)": _engine_msrm,
    "Image Quotient (IQ)": _engine_image_quotient,
    "Specular View": _engine_specular_view,
    "PCA 1": lambda d,s,h,w,c: _engine_pca(d,s,h,w,c,ch=0),
    "PCA 2": lambda d,s,h,w,c: _engine_pca(d,s,h,w,c,ch=1),
    "PCA 3": lambda d,s,h,w,c: _engine_pca(d,s,h,w,c,ch=2),
    "Retinex Enhancement": _engine_retinex,
    "Raking Composite": _engine_raking_composite,
    "Multi-Directional Hillshading (Blend / PCA)": _engine_multi_directional_hillshading,
    "Albedo-Suppressed Raking (Clay Mode)": _engine_clay_mode,
    "Ridge and Groove Detector": _engine_ridge_detector,
    "RTI Specular Enhancement": _engine_rti_specular_enhancement,
    "RTI Image USM": _engine_rti_image_usm,
    "RBF Relight (Shiny / Translucent)": _engine_rbf_relight,
    "Bilateral Normal Sharpening": _engine_bilateral_normal_sharpening,
    "Subsurface Scattering (SSS)": _engine_sss,
    "Normal Residual": _engine_normal_residual,
    "Colour ICA (IC1-3)": _engine_colour_ica,
    "RGB-PS R-B Diff": _engine_rgb_ps_rb_diff,
    "Exaggerated Shading": _engine_exaggerated_shading,
    "Curvedness": _engine_curvedness,
    "Cast-Shadow Raking": _engine_cast_shadow_raking,
    "Multi-Light Fusion": _engine_multi_light_fusion,
    "PS Fit Residual (QC)": _engine_fit_residual_qc,
})
