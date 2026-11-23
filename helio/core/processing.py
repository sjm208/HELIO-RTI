"""Terrain analysis and processing kernels.

Each function is a pure NumPy/OpenCV computation with no UI dependencies.
Used by Helio_Pro.py methods and callable from tests or headless scripts.
"""

from __future__ import annotations

from helio.core.frame_read import read_frame

import gc
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

import logging
_log = logging.getLogger(__name__)

from helio.utils import aggressive_cleanup
from helio.core.depth_integration import (
    frankot_chellappa,
    simchony_depth_integration,
)
from helio.core.frames import depth_slopes, normal_divergence, normal_slopes

from helio.config import HAS_RVT, MESH_MAX_RES

if HAS_RVT:
    import rvt.vis

# Moved to their own modules (November 2026); re-exported so every caller is unchanged.
from helio.core.mesh_kernels import (  # noqa: E402,F401
    fast_large_sigma_blur,
    subtract_parabolic_baseline,
    repair_nonfinite,
    compute_export_depth_grid,
    apply_export_filters,
    bake_normal_detail,
    build_grid_mesh,
    build_watertight_mesh_data,
    compute_hf_depth_from_normals,
    compute_vertex_normals_from_ps,
    pack_binary_stl)
from helio.core.stack_ica import (  # noqa: E402,F401
    skew_sign,
    pca_via_gram,
    whiten_pca_gram,
    fastica_fit)
from helio.core.sphere_search import (  # noqa: E402,F401
    detect_sphere_candidates,
    _ball_evidence,
    auto_detect_sphere)


# ── Hillshade ──────────────────────────────────────────────────────

def hillshade_from_normals(
    nrm: np.ndarray, az_deg: float, alt_deg: float,
) -> np.ndarray:
    """Hillshade from PS normals (most accurate for manuscript surfaces)."""
    N = nrm.astype(np.float32)
    if N.max() > 1.5:
        N = N / 127.5 - 1.0
    h, w = N.shape[:2]
    N = N.reshape(-1, 3)
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-8)
    az, al = np.radians(az_deg), np.radians(alt_deg)
    L = np.array([
        np.sin(az) * np.cos(al),
        -np.cos(az) * np.cos(al),
        np.sin(al),
    ], np.float32)
    return np.clip(np.maximum(0, (N @ L).reshape(h, w)), 0, 1)


def hillshade_from_gradient(
    dem: np.ndarray, gx: np.ndarray, gy: np.ndarray,
    az_deg: float, alt_deg: float,
) -> np.ndarray:
    """Hillshade from gradient (fallback when PS normals unavailable)."""
    az, al = np.radians(az_deg), np.radians(alt_deg)
    sl = np.arctan(np.sqrt(gx ** 2 + gy ** 2)).astype(np.float32)
    asp = (np.arctan2(-gy, gx) % (2 * np.pi)).astype(np.float32)
    return np.clip(
        np.cos(np.pi / 2 - al) * np.cos(sl)
        + np.sin(np.pi / 2 - al) * np.sin(sl) * np.cos(az - asp),
        0, 1,
    ).astype(np.float32)


def multi_hillshade(
    dem: np.ndarray,
    nrm: Optional[np.ndarray],
    n_dirs: int, alt_deg: float,
) -> np.ndarray:
    """Multi-directional hillshade (mean of N azimuths)."""
    gh, gw = dem.shape
    gy, gx = np.gradient(dem)
    gx = gx.astype(np.float32)
    gy = gy.astype(np.float32)
    acc = np.zeros((gh, gw), np.float32)
    for di in range(n_dirs):
        az = 360.0 * di / n_dirs
        if nrm is not None:
            acc += hillshade_from_normals(nrm, az, alt_deg)
        else:
            acc += hillshade_from_gradient(dem, gx, gy, az, alt_deg)
    return acc / n_dirs


# ── Slope ──────────────────────────────────────────────────────────

def slope(dem: np.ndarray, unit: str = "degree") -> np.ndarray:
    """Slope gradient from DEM."""
    gy, gx = np.gradient(dem.astype(np.float32))
    sl = np.arctan(np.sqrt(gx ** 2 + gy ** 2))
    if unit == "degree":
        return np.degrees(sl).astype(np.float32)
    elif unit == "percent":
        return (np.tan(sl) * 100).astype(np.float32)
    return sl.astype(np.float32)


# ── SVF: Yokoyama horizon scanning ─────────────────────────────────

def svf_scan(
    dem: np.ndarray, r_px: int, n_dir: int,
    positive_opns: bool = False,
    positive: bool = True,
) -> np.ndarray:
    """Yokoyama horizon scanning SVF / Openness.

    Parameters
    ----------
    dem : np.ndarray
        VE-scaled DEM (float32).
    r_px : int
        Search radius in pixels.
    n_dir : int
        Number of search directions.
    positive_opns : bool
        Compute positive openness instead of SVF.
    positive : bool
        Positive (True) or negative (False) openness.

    Returns
    -------
    np.ndarray
        SVF or openness map (float32).
    """
    h, w = dem.shape
    yg = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    xg = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
    acc = np.zeros((h, w), np.float32)
    for di in range(n_dir):
        az = 2.0 * np.pi * di / n_dir
        sdx = float(np.sin(az))
        sdy = -float(np.cos(az))
        best = np.full((h, w), -9999.0, np.float32)
        for r in range(1, r_px + 1):
            tx = np.clip(xg + r * sdx, 0, w - 1).astype(np.float32)
            ty = np.clip(yg + r * sdy, 0, h - 1).astype(np.float32)
            tz = cv2.remap(dem, tx, ty, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            tan_e = (tz - dem) / float(r)
            np.maximum(best, tan_e, out=best)
        horizon = np.arctan(best).astype(np.float32)
        if positive_opns:
            acc += (np.pi / 2 - horizon) if positive else -(np.pi / 2 - horizon)
        else:
            acc += np.cos(horizon) ** 2
    return (acc / n_dir).astype(np.float32)


# ── Simple Local Relief Model ──────────────────────────────────────

def slrm(dem: np.ndarray, radius: int) -> np.ndarray:
    """Simple Local Relief Model (SLRM): the height less its local mean.

    Uses RVT when available and *radius* is within the 10-50 px RVT accepts;
    otherwise the same model by a uniform filter.
    """
    if HAS_RVT and 10 <= radius <= 50:
        try:
            return rvt.vis.slrm(dem, radius_cell=radius)
        except Exception:
            _log.warning("RVT slrm failed, falling back to uniform filter")
    return dem - uniform_filter(dem, radius * 2 + 1).astype(np.float32)


# ── Multi-Scale Relief Model ───────────────────────────────────────

def msrm(dem: np.ndarray, rmin: float, rmax: float) -> np.ndarray:
    """Multi-Scale Relief Model (MSRM).

    Uses RVT when available, falls back to multi-scale Gaussian subtraction.
    """
    if HAS_RVT:
        try:
            # scaling_factor is required since rvt-py 2; 2 is RVT's own default.
            return rvt.vis.msrm(dem, 1.0, rmin, rmax, 2)
        except Exception:
            _log.warning("RVT msrm failed, falling back to multi-scale Gaussian subtraction")
    scales = np.linspace(max(0.5, rmin), rmax, 5)
    acc = np.zeros_like(dem)
    for s in scales:
        acc += dem - cv2.GaussianBlur(dem, (0, 0), float(s))
    return (acc / len(scales)).astype(np.float32)


# ── Local Dominance ────────────────────────────────────────────────

def local_dominance(dem: np.ndarray, min_rad: int, max_rad: int) -> np.ndarray:
    """Local Dominance (elevation percentile within an annulus).

    Uses RVT when available, falls back to uniform-filter annulus.
    """
    if HAS_RVT:
        try:
            return rvt.vis.local_dominance(dem, min_rad=min_rad, max_rad=max_rad)
        except Exception:
            _log.warning("RVT local_dominance failed, falling back to uniform filter approximation")
    lo = uniform_filter(dem, min_rad * 2 + 1)
    hi = uniform_filter(dem, max_rad * 2 + 1)
    diff = dem - (lo + hi) * 0.5
    p98 = float(np.percentile(np.abs(diff), 98)) + 1e-9
    return np.clip(0.5 + diff / p98 * 0.5, 0, 1).astype(np.float32)


# ── Surface Curvature ──────────────────────────────────────────────

def surface_curvature(dem: np.ndarray) -> np.ndarray:
    """Surface curvature (Laplacian of DEM)."""
    dx2 = cv2.Sobel(cv2.Sobel(dem, cv2.CV_32F, 1, 0), cv2.CV_32F, 1, 0)
    dy2 = cv2.Sobel(cv2.Sobel(dem, cv2.CV_32F, 0, 1), cv2.CV_32F, 0, 1)
    curv = (dx2 + dy2).astype(np.float32)
    p2 = float(np.percentile(curv, 2))
    p98 = float(np.percentile(curv, 98))
    return np.clip((curv - p2) / max(p98 - p2, 1e-9), 0, 1).astype(np.float32)


# ── SVF (RVT wrapper) ──────────────────────────────────────────────

def svf_rvt(
    dem: np.ndarray, r_px: int, n_dir: int,
) -> np.ndarray:
    """Sky-view factor via RVT, with pure-NumPy fallback."""
    if HAS_RVT:
        try:
            return np.clip(rvt.vis.sky_view_factor(
                dem, 1.0, compute_svf=True,
                svf_n_dir=n_dir, svf_r_max=r_px,
            )["svf"], 0, 1)
        except Exception:
            _log.warning("RVT sky_view_factor (SVF) failed, falling back to svf_scan")
    return np.clip(svf_scan(dem, r_px, n_dir), 0, 1)


def anisotropic_svf(
    dem: np.ndarray, r_px: int, n_dir: int,
    az_deg: float, aspect: np.ndarray,
    level: int = 1,
) -> np.ndarray:
    """Anisotropic sky-view factor.

    Uses RVT when available, falls back to directional weighting.
    """
    if HAS_RVT:
        try:
            aniso = 2 if level == 2 else 1
            return np.clip(rvt.vis.sky_view_factor(
                dem, 1.0, compute_svf=False, compute_asvf=True,
                svf_n_dir=n_dir, svf_r_max=r_px,
                asvf_dir=az_deg, asvf_level=aniso,
            )["asvf"], 0, 1)
        except Exception:
            _log.warning("RVT anisotropic sky_view_factor failed, falling back to directional SVF")
    svf = svf_scan(dem, r_px, n_dir)
    w_dir = np.clip(np.cos(aspect - np.radians(az_deg)) * 0.5 + 0.5, 0, 1)
    return np.clip(svf * (0.5 + w_dir * 0.5), 0, 1).astype(np.float32)


def positive_openness(
    dem: np.ndarray, r_px: int, n_dir: int,
) -> np.ndarray:
    """Positive openness.

    Uses RVT when available, falls back to SVF scan with positive_opns=True.
    """
    if HAS_RVT:
        try:
            return cv2.normalize(rvt.vis.sky_view_factor(
                dem, 1.0, compute_svf=False, compute_opns=True,
                svf_n_dir=n_dir, svf_r_max=r_px,
            )["opns"], None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        except Exception:
            _log.warning("RVT positive openness failed, falling back to svf_scan")
    result = svf_scan(dem, r_px, n_dir, positive_opns=True, positive=True)
    return cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)


def negative_openness(
    dem: np.ndarray, r_px: int, n_dir: int,
) -> np.ndarray:
    """Negative openness (inverted DEM)."""
    if HAS_RVT:
        try:
            return cv2.normalize(rvt.vis.sky_view_factor(
                -dem, 1.0, compute_svf=False, compute_opns=True,
                svf_n_dir=n_dir, svf_r_max=r_px,
            )["opns"], None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        except Exception:
            _log.warning("RVT negative openness failed, falling back to svf_scan on inverted DEM")
    result = svf_scan(-dem, r_px, n_dir, positive_opns=True, positive=True)
    return cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)


# ── Image pre-processing ───────────────────────────────────────────


def flatten_illumination_greys(
    grey_stack: np.ndarray,
    w: int, h: int,
) -> np.ndarray:
    """Equalise illumination across a stack of greyscale images.

    Each image is divided by a strong Gaussian blur of itself, then
    rescaled to the original mean brightness of its lit pixels.
    """
    out = grey_stack.copy().astype(np.float32)
    for i in range(len(out)):
        blur = cv2.resize(
            cv2.GaussianBlur(
                cv2.resize(out[i], (0, 0), fx=0.05, fy=0.05),
                (0, 0), sigmaX=15,
            ),
            (w, h),
        )
        mask = out[i] > 15
        target = float(np.mean(out[i][mask])) if np.any(mask) else 128.0
        out[i] = np.clip((out[i] / (blur + 1e-5)) * target, 0, 255)
    return out


def flatten_illumination_rgb(
    rgb_list: list[np.ndarray],
    w: int, h: int,
) -> list[np.ndarray]:
    """Equalise illumination across a list of RGB images."""
    out = []
    for img in rgb_list:
        blur = cv2.resize(
            cv2.GaussianBlur(
                cv2.resize(img.astype(np.float32), (0, 0), fx=0.05, fy=0.05),
                (0, 0), sigmaX=15,
            ),
            (w, h),
        )
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = grey > 15
        target = np.mean(img[mask], axis=0) if np.any(mask) else np.array([128, 128, 128])
        corrected = np.clip((img.astype(np.float32) / (blur + 1e-5)) * target, 0, 255)
        out.append(corrected.astype(np.uint8))
    return out


def normalise_median_brightness(grey_stack: np.ndarray) -> np.ndarray:
    """Normalise each frame so its median matches the stack-wide median."""
    medians = [float(np.median(g)) for g in grey_stack]
    target = float(np.median(medians))
    out = grey_stack.copy().astype(np.float32)
    for i in range(len(out)):
        out[i] = np.clip(out[i] * (target / (medians[i] + 1e-5)), 0, 255)
    return out


def apply_clahe_stack(grey_stack: np.ndarray, clip_limit: float = 2.0, tile_size: int = 16) -> np.ndarray:
    """Apply CLAHE to every frame in a greyscale stack."""
    tile_size = max(4, min(64, tile_size))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    out = grey_stack.copy().astype(np.float32)
    for i in range(len(out)):
        g8 = np.clip(out[i], 0, 255).astype(np.uint8)
        out[i] = clahe.apply(g8).astype(np.float32)
    return out


def adaptive_light_weighting(
    grey_stack: np.ndarray, L_mat: np.ndarray, alw: float,
) -> np.ndarray:
    """Down-weight grazing-angle lights based on Z component of light vectors."""
    lz = np.clip(L_mat[:, 2], 0.0, 1.0)
    wf = 1.0 - alw * (1.0 - np.clip(np.degrees(np.arcsin(lz)) / 45.0, 0.0, 1.0))
    return (grey_stack * wf[:, np.newaxis, np.newaxis]).astype(np.float32)


def resize_to_common_shape(
    images: list[np.ndarray],
) -> tuple[np.ndarray, int, int]:
    """Resize all images to match the first image's (h, w), return stack + h + w."""
    base_h, base_w = images[0].shape
    out = np.empty((len(images), base_h, base_w), dtype=np.float32)
    out[0] = images[0]
    for i in range(1, len(images)):
        img = images[i]
        if img.shape != (base_h, base_w):
            img = cv2.resize(img, (base_w, base_h))
        out[i] = img
    return out, base_h, base_w


def crop_stack_with_coord_transform(
    grey_stack: np.ndarray,
    rgb_list: list[np.ndarray],
    crop_override: tuple[int, int, int, int],
    prev_w: int, prev_h: int,
    base_w: int, base_h: int,
) -> tuple[np.ndarray, list[np.ndarray], int, int]:
    """Apply crop to image stack, transforming coords from previous space if needed."""
    cx1, cy1, cx2, cy2 = crop_override
    if prev_w > 0 and prev_h > 0 and (prev_w != base_w or prev_h != base_h):
        sx = base_w / prev_w
        sy = base_h / prev_h
        cx1 = int(round(cx1 * sx))
        cy1 = int(round(cy1 * sy))
        cx2 = int(round(cx2 * sx))
        cy2 = int(round(cy2 * sy))
    cx1 = max(0, cx1)
    cy1 = max(0, cy1)
    cx2 = min(base_w, cx2)
    cy2 = min(base_h, cy2)
    grey_out = grey_stack[:, cy1:cy2, cx1:cx2]
    rgb_out = [r[cy1:cy2, cx1:cx2] for r in rgb_list]
    return grey_out, rgb_out, grey_out.shape[2], grey_out.shape[1]


# ── Multi-scale topographic position ───────────────────────────────


def deviation_from_elevation(dem: np.ndarray, radius: float) -> np.ndarray:
    """Feature-preserving deviation-from-elevation (DEV) at a given scale."""
    blur = cv2.GaussianBlur(dem, (0, 0), float(radius))
    raw = dem - blur
    p2 = float(np.percentile(raw, 2))
    p98 = float(np.percentile(raw, 98))
    return np.clip((raw - p2) / (p98 - p2 + 1e-9), 0, 1).astype(np.float32)


def mstp_composite(
    dem: np.ndarray,
    r_fine: float, r_med: float, r_coarse: float,
    lightness: float = 0.9,
) -> np.ndarray:
    """Multi-Scale Topographic Position (MSTP) RGB composite.

    R = fine-scale DEV, G = meso-scale DEV, B = coarse-scale DEV.
    Returns (H, W, 3) uint8 array.
    """
    dev_f = deviation_from_elevation(dem, r_fine)
    dev_m = deviation_from_elevation(dem, r_med)
    dev_c = deviation_from_elevation(dem, r_coarse)
    rgb = np.stack([dev_f, dev_m, dev_c], axis=-1).astype(np.float32)
    lit = np.power(np.clip(rgb, 0, 1), 1.0 / max(0.1, lightness))
    return (lit * 255).clip(0, 255).astype(np.uint8)


import math as _math          # noqa: E402  (as the mesh kernels and the views below spell it)



def cap_img(arr: np.ndarray | None, cap: int = 1200) -> np.ndarray | None:
    """Downsample an image to at most *cap* pixels on its longest axis.

    Integer division is used to preserve exact Nyquist — avoids the
    half-pixel shift that ``INTER_AREA`` can introduce on odd-sized
    inputs, which causes visible 1-px seams in tiled contact sheets.
    """
    if arr is None:
        return None
    h, w = arr.shape[:2]
    if max(h, w) <= cap:
        return arr
    sc = cap / max(h, w)
    nh = max(1, int(h * sc))
    nw = max(1, int(w * sc))
    return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)


# ── Shape-from-Shading / Openness computation ─────────────────────

def openness_yokoyama(
    depth: np.ndarray,
    radius_px: float,
    n_dirs: int,
    n_steps: int,
    positive: bool,
    max_chunk_bytes: int = 256_000_000,
) -> np.ndarray:
    """Vectorised Yokoyama (2002) Terrain Openness.

    Parameters
    ----------
    depth : np.ndarray
        VE-scaled depth map (float32).
    radius_px : float
        Search radius in pixels.
    n_dirs : int
        Number of azimuth search directions.
    n_steps : int
        Number of radial steps (samples along each azimuth).
    positive : bool
        True = positive openness, False = negative openness.
    max_chunk_bytes : int
        Memory cap for the step-chunking loop.

    Returns
    -------
    np.ndarray
        Openness map (float32) — higher = more exposed.
    """
    fh, fw = depth.shape
    ys = np.arange(fh, dtype=np.float32)
    xs = np.arange(fw, dtype=np.float32)
    step_sz = np.float32(radius_px / max(n_steps, 1))
    max_chunk = max(1, int(max_chunk_bytes / (fh * fw * 4)))
    acc = np.zeros((fh, fw), np.float32)

    for di in range(n_dirs):
        az = np.float64(2.0 * np.pi * di / n_dirs)
        caz = np.float32(np.cos(az))
        saz = np.float32(np.sin(az))

        if positive:
            h_best = np.full((fh, fw), np.float32(-np.pi * 0.5), np.float32)
        else:
            h_best = np.full((fh, fw), np.float32(np.pi * 0.5), np.float32)

        for chunk_start in range(0, n_steps, max_chunk):
            chunk_end = min(chunk_start + max_chunk, n_steps)
            steps_idx = np.arange(chunk_start + 1, chunk_end + 1, dtype=np.float32)
            dists = steps_idx * step_sz
            sx_all = np.clip(
                (xs[np.newaxis, np.newaxis, :] + caz * dists[:, np.newaxis, np.newaxis]
                 ).astype(np.int32), 0, fw - 1)
            sy_all = np.clip(
                (ys[np.newaxis, :, np.newaxis] + saz * dists[:, np.newaxis, np.newaxis]
                 ).astype(np.int32), 0, fh - 1)
            samp = depth[sy_all, sx_all]
            h_ang = np.arctan2(samp - depth[np.newaxis], dists[:, np.newaxis, np.newaxis])
            del sx_all, sy_all, samp
            if positive:
                np.maximum(h_best, h_ang.max(axis=0), out=h_best)
            else:
                np.minimum(h_best, h_ang.min(axis=0), out=h_best)
            del h_ang

        if positive:
            acc += np.float32(np.pi * 0.5) - h_best
        else:
            acc += np.float32(np.pi * 0.5) + h_best
        del h_best

    return acc / n_dirs


def _rescale_1_99(arr: np.ndarray) -> np.ndarray:
    """Clip-and-stretch to [0, 1] using 1st/99th percentiles."""
    p1, p99 = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
    return np.clip((arr - p1) / (p99 - p1 + 1e-9), 0, 1).astype(np.float32)


def _norm_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]."""
    mn, mx = float(arr.min()), float(arr.max())
    return (arr - mn) / (mx - mn + 1e-9)


def hillshade_from_gradients(
    depth: np.ndarray,
    az_deg: float,
    alt_deg: float,
    z_factor: float = 1.0,
) -> np.ndarray:
    """QGIS-style hillshade from depth gradients."""
    az_rad = np.radians((360.0 - az_deg + 90.0) % 360.0)
    alt_rad = np.radians(alt_deg)
    ca, sa = np.float32(np.cos(az_rad)), np.float32(np.sin(az_rad))
    ce, se = np.float32(np.cos(alt_rad)), np.float32(np.sin(alt_rad))
    gy, gx = np.gradient(depth.astype(np.float32) * np.float32(z_factor))
    sx, sy = gx, -gy.astype(np.float32)
    dnorm = np.sqrt(np.float32(1.0) + sx ** 2 + sy ** 2)
    hs = np.clip((ce * (ca * (-sy) + sa * sx) + se) / dnorm, 0, 1).astype(np.float32)
    return _rescale_1_99(hs)


def md_hillshade_8dir(
    depth: np.ndarray,
    alt_deg: float = 45.0,
) -> np.ndarray:
    """Multi-directional hillshade (Mark 1992): 8 azimuths with cosine weights."""
    _mdh_az_wts = [
        (0, 1.000), (45, 0.707), (90, 1.000), (135, 0.707),
        (180, 1.000), (225, 0.707), (270, 1.000), (315, 0.707),
    ]
    total_w = sum(wt for _, wt in _mdh_az_wts)
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    slope = np.arctan(np.sqrt(gx ** 2 + gy ** 2)).astype(np.float32)
    aspect = np.arctan2(-gy, gx).astype(np.float32)
    alt_r = np.radians(alt_deg)
    acc = np.zeros_like(depth, dtype=np.float32)
    for maz, mwt in _mdh_az_wts:
        az_r = np.radians(float(maz))
        hs_dir = (np.sin(alt_r) * np.sin(slope)
                  + np.cos(alt_r) * np.cos(slope) * np.cos(az_r - aspect))
        acc += np.clip(hs_dir, 0, 1).astype(np.float32) * np.float32(mwt)
    return _rescale_1_99(acc / total_w)


def sky_diffuse_render(
    depth: np.ndarray,
    n_az: int = 16,
    alts: tuple[float, ...] = (15.0, 45.0, 75.0),
) -> np.ndarray:
    """Hemisphere sky-diffuse render (N azimuths x M elevations)."""
    h, w = depth.shape
    gy, gx = np.gradient(depth.astype(np.float32))
    sx, sy = gx.astype(np.float32), (-gy).astype(np.float32)
    dnorm = np.sqrt(np.float32(1.0) + sx ** 2 + sy ** 2)
    acc = np.zeros((h, w), np.float32)
    count = n_az * len(alts)
    for sa in alts:
        sa_r = np.radians(sa)
        ce = np.float32(np.cos(sa_r))
        se = np.float32(np.sin(sa_r))
        for ai in range(n_az):
            az_r = np.float32(2.0 * np.pi * ai / n_az)
            ca = np.float32(np.cos(az_r))
            sa2 = np.float32(np.sin(az_r))
            hs = np.clip((ce * (ca * (-sy) + sa2 * sx) + se) / dnorm, 0, 1)
            acc += hs.astype(np.float32)
    return _rescale_1_99(acc / count)


def surface_curvature_mean(
    depth: np.ndarray,
    z_factor: float = 1.0,
    smooth_sigma: float = 1.5,
) -> np.ndarray:
    """Mean surface curvature (κ1 + κ2) / 2 — positive = convex, negative = concave."""
    dep_z = depth.astype(np.float32) * np.float32(z_factor) * 8.0
    gy, gx = np.gradient(dep_z)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    fx = gx.astype(np.float32)
    fy = gy.astype(np.float32)
    fxx = gxx.astype(np.float32)
    fyy = gyy.astype(np.float32)
    fxy = gyx.astype(np.float32)
    denom = np.float32(2.0) * np.power(np.float32(1.0) + fx ** 2 + fy ** 2, np.float32(1.5))
    # H = ((1 + fx^2) fyy - 2 fx fy fxy + (1 + fy^2) fxx) / (2 (1 + fx^2 + fy^2)^1.5).
    # (The two "1 +" were missing until 2026-11-10: on level ground, where
    # fx = fy = 0, every curvature read zero.)
    one = np.float32(1.0)
    H = ((one + fx ** 2) * fyy - np.float32(2.0) * fx * fy * fxy + (one + fy ** 2) * fxx) / (denom + 1e-9)
    H_s = cv2.GaussianBlur(H, (0, 0), smooth_sigma)
    return _rescale_1_99(H_s)


def local_relief_slrm(
    depth: np.ndarray,
    radius: float,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Local Relief Model (fine-scale SLRM): Gaussian trend subtraction."""
    arr = depth.astype(np.float32) * np.float32(z_factor)
    trend = cv2.GaussianBlur(arr, (0, 0), float(radius))
    relief = arr - trend
    return _rescale_1_99(relief)


def post_process_composite(
    composite: np.ndarray,
    noise_supp: float = 0.0,
    clahe_clip: float = 0.0,
    unsharp: float = 0.0,
) -> np.ndarray:
    """Post-processing pipeline: NL-Means → CLAHE → Unsharp mask."""
    result = composite.copy()
    if noise_supp > 0.05:
        h_nlm = max(2, int(noise_supp * 5))
        c8 = (result * 255).clip(0, 255).astype(np.uint8)
        result = cv2.fastNlMeansDenoising(c8, h=h_nlm, templateWindowSize=7,
                                           searchWindowSize=21).astype(np.float32) / 255.0
    if clahe_clip > 0.1:
        c8 = (result * 255).clip(0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(16, 16))
        result = clahe.apply(c8).astype(np.float32) / 255.0
    if unsharp > 0.05:
        blurred = cv2.GaussianBlur(result, (0, 0), 1.5)
        result = np.clip(result + float(unsharp) * (result - blurred), 0, 1).astype(np.float32)
    return result


def compute_sfs_composite(
    depth: np.ndarray,
    normals: np.ndarray,
    albedo: np.ndarray,
    settings: dict,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict[str, np.ndarray]:
    """Full Shape-from-Shading composite generation pipeline.

    Parameters
    ----------
    depth : np.ndarray
        Depth map (float32, H×W).
    normals : np.ndarray
        PS normal map (float32, H×W×3, values in [-1, 1]).
    albedo : np.ndarray
        RGB albedo (uint8, H×W×3).
    settings : dict
        Keys: depth_src, quality, z_factor, az, alt, blend_mode,
              fine_wt, gamma, arc_width, hs_gamma, comp_mode,
              noise_supp, clahe_clip, unsharp, cmap, cond_blend,
              do_mirror, do_invert, do_screen.
    progress_cb : Callable or None
        Called with (message, fraction).

    Returns
    -------
    dict
        'composite', 'comp_8', 'comp_16', 'sfs_final', 'sfs_fine_norm',
        'heat_map_8', 'cond_map_8', 'col_multiply_8', 'col_overlay_8',
        'col_screen_8' (optional), 'mirror_8' (optional), 'invert_8' (optional).
    """
    _upd = progress_cb or (lambda msg, frac: None)
    h, w = depth.shape
    _z = max(0.001, min(2.000, float(settings.get("z_factor", 0.05))))
    _az = float(settings.get("az", 305.0)) % 360.0
    _alt = max(5.0, min(85.0, float(settings.get("alt", 45.0))))
    _blend = settings.get("blend_mode", "Multiply")
    _fw = max(0.0, min(1.0, float(settings.get("fine_wt", 0.60))))
    _gamma = max(0.2, min(3.0, float(settings.get("gamma", 0.88))))
    _q = settings.get("quality", "Standard")
    _dirs_override = int(settings.get("n_dirs", 0))
    _steps_override = int(settings.get("n_steps", 0))
    _arc_width = max(0.0, min(90.0, float(settings.get("arc_width", 0.0))))
    _hs_gamma = max(0.3, min(2.5, float(settings.get("hs_gamma", 1.0))))
    _comp_mode = settings.get("comp_mode", "Pure Openness")
    _noise_supp = float(settings.get("noise_supp", 0.0))
    _clahe_clip = float(settings.get("clahe_clip", 0.0))
    _unsharp = float(settings.get("unsharp", 0.0))
    _hm_cmap = settings.get("cmap", "Jet")
    _cond_blend = float(settings.get("cond_blend", 0.35))
    _do_mirror = bool(settings.get("do_mirror", False))
    _do_invert = bool(settings.get("do_invert", False))
    _do_screen = bool(settings.get("do_screen", True))

    # Quality presets
    if "Draft" in _q:
        n_dirs, n_steps, _half = 8, 8, True
    elif "Standard" in _q:
        n_dirs, n_steps, _half = 8, 12, False
    elif "Full" in _q:
        n_dirs, n_steps, _half = 16, 16, False
    else:
        n_dirs, n_steps, _half = 16, 24, False
    if _dirs_override > 0:
        n_dirs = _dirs_override
    if _steps_override > 0:
        n_steps = _steps_override

    # Depth scaling
    d_rng = float(np.ptp(depth))
    if d_rng < 1e-9:
        d_rng = 1.0
    scale_factor = min(h, w) / 8.0
    d_sc = (depth - depth.min()) / d_rng * scale_factor * _z

    d_work = cv2.resize(d_sc, (max(4, w // 2), max(4, h // 2)),
                        interpolation=cv2.INTER_AREA) if _half else d_sc
    _ch, _cw = d_work.shape

    _upd("Fine openness…", 0.05)
    r_fine = max(4.0, min(_ch, _cw) * 0.060)
    po_fine = openness_yokoyama(d_work, r_fine, n_dirs, n_steps, True)
    no_fine = openness_yokoyama(d_work, r_fine, n_dirs, n_steps, False)

    _upd("Coarse openness…", 0.50)
    r_coarse = max(8.0, min(_ch, _cw) * 0.18)
    po_coarse = openness_yokoyama(d_work, r_coarse, n_dirs, n_steps, True)
    no_coarse = openness_yokoyama(d_work, r_coarse, n_dirs, n_steps, False)

    if _half:
        _up = lambda a: cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        po_fine, no_fine = _up(po_fine), _up(no_fine)
        po_coarse, no_coarse = _up(po_coarse), _up(no_coarse)

    po_fine_n = _norm_01(po_fine)
    no_fine_n = _norm_01(no_fine)
    po_coarse_n = _norm_01(po_coarse)
    no_coarse_n = _norm_01(no_coarse)

    sfs_fine = (po_fine_n + (1.0 - no_fine_n)) * 0.5
    sfs_coarse = (po_coarse_n + (1.0 - no_coarse_n)) * 0.5
    sfs_blend = sfs_fine * _fw + sfs_coarse * (1.0 - _fw)

    _upd("SVF ambient layer…", 0.78)
    nz = np.clip(normals[..., 2], 0.0, 1.0)
    svf_approx = np.power(nz, 1.5)
    svf_blur = cv2.GaussianBlur(svf_approx, (0, 0), max(5.0, 0.03 * math.sqrt(h * h + w * w)))
    svf_norm = _norm_01(svf_approx - svf_blur * 0.5 + 0.5)
    sfs_final = sfs_blend * (0.85 + 0.15 * svf_norm)

    p2, p98 = float(np.percentile(sfs_final, 2)), float(np.percentile(sfs_final, 98))
    if p98 > p2:
        np.subtract(sfs_final, p2, out=sfs_final)
        np.divide(sfs_final, p98 - p2, out=sfs_final)
        np.clip(sfs_final, 0, 1, out=sfs_final)
    np.clip(sfs_final, 0, 1, out=sfs_final)
    sfs_final = np.power(sfs_final, _gamma).astype(np.float32)

    sfs_fine_norm = np.power(np.clip(
        _norm_01((po_fine_n + (1.0 - no_fine_n)) * 0.5), 0, 1), _gamma).astype(np.float32)

    _upd(f"Composite: {_comp_mode}…", 0.86)
    composite = sfs_final.copy()

    if _comp_mode == "Openness Screen":
        composite = np.clip(1.0 - (1.0 - sfs_fine_norm) * (1.0 - sfs_final), 0, 1).astype(np.float32)
        composite = _rescale_1_99(composite)

    elif _comp_mode == "MDH 8-dir":
        mdh = md_hillshade_8dir(d_work, _alt)
        if _half:
            mdh = cv2.resize(mdh, (w, h), interpolation=cv2.INTER_LINEAR)
        composite = _rescale_1_99(mdh)

    elif _comp_mode == "Sky Diffuse":
        sky = sky_diffuse_render(d_work)
        if _half:
            sky = cv2.resize(sky, (w, h), interpolation=cv2.INTER_LINEAR)
        composite = _rescale_1_99(sky)

    elif _comp_mode == "Sky × Hillshade":
        sky = sky_diffuse_render(d_work, n_az=8, alts=(30.0, 60.0))
        hs = hillshade_from_gradients(d_work, _az, _alt)
        if abs(_hs_gamma - 1.0) > 0.02:
            np.power(hs, np.float32(_hs_gamma), out=hs)
        blended = np.clip(1.0 - (1.0 - sky) * (1.0 - hs), 0, 1).astype(np.float32)
        if _half:
            blended = cv2.resize(blended, (w, h), interpolation=cv2.INTER_LINEAR)
        composite = _rescale_1_99(blended)

    elif _comp_mode == "Curvature":
        composite = surface_curvature_mean(d_work, z_factor=8.0)
        if _half:
            composite = cv2.resize(composite, (w, h), interpolation=cv2.INTER_LINEAR)

    elif _comp_mode == "Local Relief":
        _lr_sigma = max(3.0, r_fine * 2.0)
        composite = local_relief_slrm(d_work, _lr_sigma, z_factor=1.0)
        if _half:
            composite = cv2.resize(composite, (w, h), interpolation=cv2.INTER_LINEAR)

    elif _comp_mode == "Broad+Fine (Gaussian)":
        # Two-layer depth shading: a Gaussian blur of the depth map gives
        # the broad/macro layer, which records the gentle gradients over a
        # wider area, and that is blended with a high-passed fine layer.
        # Both are rendered with a single configurable directional light —
        # distinct from the openness/ambient-occlusion sweep every other
        # mode in this function uses.
        _bf_broad_pct = max(1.0, min(40.0, float(settings.get("bf_broad_pct", 8.0))))
        _bf_fine_pct  = max(0.1, min(10.0, float(settings.get("bf_fine_pct", 1.0))))
        _bf_fine_gain = max(1.0, min(20.0, float(settings.get("bf_fine_gain", 6.0))))
        _ref_dim = min(_ch, _cw)
        _broad_sigma = max(1.0, _ref_dim * _bf_broad_pct / 100.0)
        _fine_sigma  = max(0.5, _ref_dim * _bf_fine_pct / 100.0)
        d_broad = cv2.GaussianBlur(d_work, (0, 0), sigmaX=_broad_sigma)
        d_fine_hp = (d_work - cv2.GaussianBlur(d_work, (0, 0), sigmaX=_fine_sigma)) * _bf_fine_gain
        hs_broad = hillshade_from_gradients(d_broad, _az, _alt, z_factor=1.0)
        hs_fine  = hillshade_from_gradients(d_fine_hp, _az, _alt, z_factor=1.0)
        if abs(_hs_gamma - 1.0) > 0.02:
            np.clip(hs_broad, 0, 1, out=hs_broad); np.power(hs_broad, _hs_gamma, out=hs_broad)
            np.clip(hs_fine, 0, 1, out=hs_fine); np.power(hs_fine, _hs_gamma, out=hs_fine)
        a, b = hs_broad, hs_fine
        if _blend == "Multiply":
            comp_raw = a * b
        elif _blend == "Screen":
            comp_raw = 1.0 - (1.0 - a) * (1.0 - b)
        elif _blend == "Soft Light":
            comp_raw = (1.0 - 2.0 * b) * a ** 2 + 2.0 * b * a
        elif _blend == "Hard Light":
            mask = b < 0.5
            comp_raw = np.where(mask, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
        else:
            mask = a < 0.5
            comp_raw = np.where(mask, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
        # _fw ("Fine weight") mixes how much of the fine layer's blend result
        # is brought in over the pure broad layer — 0 = broad only, 1 = full
        # blend — same control the other modes already use for fine/coarse
        # mixing, reused here for consistency.
        comp_raw = a * (1.0 - _fw) + comp_raw * _fw
        if _half:
            comp_raw = cv2.resize(comp_raw, (w, h), interpolation=cv2.INTER_LINEAR)
        composite = _rescale_1_99(comp_raw.astype(np.float32))

    elif "Hillshade" in _comp_mode or _comp_mode == "Hillshade ×":
        def _qhs(arr, az, alt):
            return hillshade_from_gradients(arr, az, alt, z_factor=1.0)

        if _arc_width >= 1.0:
            n_arc = 5
            half_arc = _arc_width / 2.0
            offsets = np.linspace(-half_arc, half_arc, n_arc)
            weights = 0.5 + 0.5 * np.cos(np.pi * offsets / half_arc)
            weights /= weights.sum()

            def _arc_hs(arr):
                acc = np.zeros_like(arr, dtype=np.float32)
                for off, wt in zip(offsets, weights):
                    acc += np.float32(wt) * _qhs(arr, (_az + off) % 360.0, _alt)
                return acc

            hs_main = _arc_hs(d_work)
            hs_fine = _arc_hs(sfs_fine_norm * float(d_work.max() + 1e-9))
        else:
            hs_main = _qhs(d_work, _az, _alt)
            hs_fine = _qhs(sfs_fine_norm * float(d_work.max() + 1e-9), _az, _alt)

        if abs(_hs_gamma - 1.0) > 0.02:
            np.clip(hs_main, 0, 1, out=hs_main)
            np.power(hs_main, _hs_gamma, out=hs_main)
            np.clip(hs_fine, 0, 1, out=hs_fine)
            np.power(hs_fine, _hs_gamma, out=hs_fine)

        a, b = hs_main, hs_fine
        if _blend == "Multiply":
            comp_raw = a * b
        elif _blend == "Screen":
            comp_raw = 1.0 - (1.0 - a) * (1.0 - b)
        elif _blend == "Soft Light":
            comp_raw = (1.0 - 2.0 * b) * a ** 2 + 2.0 * b * a
        elif _blend == "Hard Light":
            mask = b < 0.5
            comp_raw = np.where(mask, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
        else:
            mask = a < 0.5
            comp_raw = np.where(mask, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
        composite = _rescale_1_99(comp_raw.astype(np.float32))

    _upd("Post-processing…", 0.89)
    composite = post_process_composite(composite, _noise_supp, _clahe_clip, _unsharp)

    _upd("Building outputs…", 0.91)
    comp_8 = (composite * 255).clip(0, 255).astype(np.uint8)
    comp_16 = (composite * 65535).clip(0, 65535).astype(np.uint16)
    sfs_8 = (sfs_final * 255).clip(0, 255).astype(np.uint8)
    fine_8 = (sfs_fine_norm * 255).clip(0, 255).astype(np.uint8)
    sfs_16 = (sfs_final * 65535).clip(0, 65535).astype(np.uint16)
    alb_f = albedo.astype(np.float32) / 255.0

    # albedo (and therefore alb_f) is RGB — cv2.imwrite always assumes BGR,
    # so every colour output built from it must be converted before going
    # into `result`, the same way cond_map_8 below already correctly does.
    # col_multiply_8/col_overlay_8/col_screen_8/alb_8 previously skipped this
    # conversion and saved with red/blue swapped (a blue cast on what should
    # be warm parchment tones).
    col_multiply = np.clip(alb_f * composite[..., np.newaxis], 0, 1)
    col_mult8 = cv2.cvtColor((col_multiply * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    lum3 = composite[..., np.newaxis]
    dark = 2.0 * lum3 * alb_f
    light = 1.0 - 2.0 * (1.0 - lum3) * (1.0 - alb_f)
    col_overlay8 = cv2.cvtColor(
        (np.clip(np.where(lum3 < 0.5, dark, light), 0, 1) * 255).clip(0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR)

    result = {
        "composite": composite,
        "comp_8": comp_8,
        "comp_16": comp_16,
        "sfs_final": sfs_final,
        "sfs_fine_norm": sfs_fine_norm,
        "sfs_8": sfs_8,
        "fine_8": fine_8,
        "sfs_16": sfs_16,
        "col_multiply_8": col_mult8,
        "col_overlay_8": col_overlay8,
        "r_fine": r_fine,
        "r_coarse": r_coarse,
        "n_dirs": n_dirs,
        "n_steps": n_steps,
    }

    if _do_screen:
        s3 = composite[..., np.newaxis]
        col_screen = np.clip(1.0 - (1.0 - s3) * (1.0 - alb_f), 0, 1)
        result["col_screen_8"] = cv2.cvtColor(
            (col_screen * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Elevation heat map
    _cmap_map = {
        "Jet": cv2.COLORMAP_JET, "Plasma": cv2.COLORMAP_PLASMA,
        "Viridis": cv2.COLORMAP_VIRIDIS, "Inferno": cv2.COLORMAP_INFERNO,
        "Magma": cv2.COLORMAP_MAGMA, "Turbo": cv2.COLORMAP_TURBO,
        "Spectral": cv2.COLORMAP_RAINBOW,
    }
    cv2_cmap = _cmap_map.get(_hm_cmap, cv2.COLORMAP_JET)
    dp = depth.astype(np.float32)
    dp_p1, dp_p99 = float(np.percentile(dp, 1)), float(np.percentile(dp, 99))
    dp_n = np.clip((dp - dp_p1) / (dp_p99 - dp_p1 + 1e-9), 0, 1)
    dp_8 = (dp_n * 255).clip(0, 255).astype(np.uint8)
    heat_8 = cv2.applyColorMap(dp_8, cv2_cmap)
    result["heat_map_8"] = heat_8

    # Condition map
    heat_f = heat_8.astype(np.float32) / 255.0
    alb_bgr_f = cv2.cvtColor((alb_f * 255).clip(0, 255).astype(np.uint8),
                              cv2.COLOR_RGB2BGR).astype(np.float32) / 255.0
    cond_map_8 = (np.clip(heat_f * np.float32(_cond_blend)
                          + alb_bgr_f * np.float32(1.0 - _cond_blend), 0, 1) * 255).clip(0, 255).astype(np.uint8)
    result["cond_map_8"] = cond_map_8

    if _do_mirror:
        result["mirror_8"] = cv2.flip(comp_8, 1)
    if _do_invert:
        inv_comp = np.float32(1.0) - sfs_final
        inv_comp = np.power(np.clip(inv_comp, 0, 1), _gamma).astype(np.float32)
        result["invert_8"] = (inv_comp * 255).clip(0, 255).astype(np.uint8)

    result["alb_8"] = cv2.cvtColor(albedo.astype(np.uint8), cv2.COLOR_RGB2BGR)

    _upd("SFS composite done.", 1.0)
    return result


# ── RVT multi-map computation ────────────────────────────────────

def compute_rvt_maps(
    depth: np.ndarray,
    normals: np.ndarray,
    albedo: np.ndarray,
    settings: dict,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict[str, np.ndarray]:
    """Compute the full RVT (Relief Visualisation Toolbox) suite of maps.

    Parameters
    ----------
    depth : np.ndarray
        Depth map (float32, H×W).
    normals : np.ndarray
        PS normal map (float32, H×W×3, values in [-1, 1]).
    albedo : np.ndarray
        RGB albedo (uint8, H×W×3).
    settings : dict
        Keys: vef, slrm_r, svf_r, svf_dirs, anis_dir_deg,
              ld_min, ld_max, mstp_light, blend_op, lf_r, scale,
              hs_az, hs_alt, hs_z, hs_blend, hs_weight,
              hs_clahe_clip, hs_out_gamma.
    progress_cb : Callable or None

    Returns
    -------
    dict
        'slrm', 'msrm', 'svf', 'asvf', 'ld', 'pos_open', 'neg_open',
        'dev_f', 'dev_m', 'dev_c', 'mstp_rgb', 'rvt_comp',
        'hrlc_map', 'hrlc_colour_8', 'hs_from_depth', plus *_8 scaled.
    """
    _upd = progress_cb or (lambda msg, frac: None)
    H, W = depth.shape

    vef = max(0.1, min(5.0, float(settings.get("vef", 1.0))))
    slrm_r = max(3, int(settings.get("slrm_r", 10)))
    svf_r = max(4, int(settings.get("svf_r", 10)))
    svf_dirs = int(settings.get("svf_dirs", 8))
    anis_dir_deg = float(settings.get("anis_dir_deg", 315.0))
    ld_min = max(2, int(settings.get("ld_min", 3)))
    ld_max = max(ld_min + 1, int(settings.get("ld_max", 11)))
    mstp_light = max(0.2, min(2.0, float(settings.get("mstp_light", 0.9))))
    blend_op = max(0.0, min(1.0, float(settings.get("blend_op", 0.40))))
    lf_r = max(10, int(settings.get("lf_r", 80)))
    _scale = settings.get("scale", "Auto")

    _hs_az = float(settings.get("hs_az", 305.0))
    _hs_alt = float(settings.get("hs_alt", 45.0))
    _hs_z = float(settings.get("hs_z", 0.05))
    _hs_blend = settings.get("hs_blend", "Multiply")
    _hs_weight = float(settings.get("hs_weight", 0.6))
    _hs_clahe_clip = float(settings.get("hs_clahe_clip", 1.5))
    _hs_out_gamma = float(settings.get("hs_out_gamma", 0.85))

    d_rng = float(np.ptp(depth))
    if d_rng < 1e-9:
        d_rng = 1.0
    d_sc = (depth - depth.min()) / d_rng * min(H, W) / 4.0 * vef
    short = min(H, W)

    if _scale == "Fine":
        r_f, r_m, r_c = (max(3, int(short * .015)), max(8, int(short * .05)),
                         max(15, int(short * .12)))
    elif _scale == "Medium":
        r_f, r_m, r_c = (max(5, int(short * .03)), max(15, int(short * .10)),
                         max(30, int(short * .22)))
    elif _scale == "Coarse":
        r_f, r_m, r_c = (max(10, int(short * .06)), max(25, int(short * .15)),
                         max(50, int(short * .35)))
    else:
        r_f, r_m, r_c = (max(5, int(short * .025)), max(15, int(short * .08)),
                         max(40, int(short * .20)))

    def _clip_norm(a, lo=1, hi=99):
        p1, p99 = float(np.percentile(a, lo)), float(np.percentile(a, hi))
        return np.clip((a - p1) / (p99 - p1 + 1e-9), 0, 1).astype(np.float32)

    _upd("SLRM…", 0.04)
    slrm_trend = cv2.GaussianBlur(d_sc, (0, 0), float(slrm_r))
    slrm = _clip_norm(d_sc - slrm_trend)

    _upd("MSRM…", 0.10)
    n_steps_msrm = 6
    msrm_acc = np.zeros((H, W), np.float32)
    log_rads = np.logspace(np.log10(max(3, r_f)), np.log10(r_c), n_steps_msrm)
    for _lr in log_rads:
        _blur = cv2.GaussianBlur(d_sc, (0, 0), float(_lr))
        msrm_acc += d_sc - _blur
    msrm = _clip_norm(msrm_acc / n_steps_msrm)

    _upd(f"SVF (r={svf_r}, {svf_dirs} dirs)…", 0.18)
    n_svf = int(svf_dirs)
    step_sz = max(1.0, svf_r / 10.0)
    n_steps_svf = max(5, int(svf_r / step_sz))
    ys = np.arange(H, dtype=np.float32)
    xs = np.arange(W, dtype=np.float32)
    svf_acc = np.zeros((H, W), np.float32)
    for _di in range(n_svf):
        az = np.float32(2.0 * np.pi * _di / n_svf)
        caz = np.float32(np.cos(az))
        saz = np.float32(np.sin(az))
        h_max = np.full((H, W), np.float32(-np.pi * 0.5), np.float32)
        for _st in range(1, n_steps_svf + 1):
            dist = np.float32(_st * step_sz)
            sx = np.clip((xs[np.newaxis, :] + caz * dist).astype(np.int32), 0, W - 1)
            sy = np.clip((ys[:, np.newaxis] + saz * dist).astype(np.int32), 0, H - 1)
            dz = d_sc[sy, sx] - d_sc
            np.maximum(h_max, np.arctan2(dz, dist), out=h_max)
        svf_acc += np.cos(np.maximum(h_max, np.float32(0)))
    svf = np.clip(svf_acc / n_svf, 0, 1).astype(np.float32)

    _upd(f"ASVF (dir={anis_dir_deg:.0f}°)…", 0.32)
    main_az_rad = np.radians(anis_dir_deg % 360.0)
    asvf_acc = np.zeros((H, W), np.float32)
    w_sum = 0.0
    for _di in range(n_svf):
        az = np.float32(2.0 * np.pi * _di / n_svf)
        diff = abs(float(az) - float(main_az_rad))
        diff = min(diff, 2 * np.pi - diff)
        w = np.float32(0.1 + 0.9 * (1 + np.cos(diff)) * 0.5)
        w_sum += w
        caz = np.float32(np.cos(az))
        saz = np.float32(np.sin(az))
        h_max = np.full((H, W), np.float32(-np.pi * 0.5), np.float32)
        for _st in range(1, n_steps_svf + 1):
            dist = np.float32(_st * step_sz)
            sx = np.clip((xs[np.newaxis, :] + caz * dist).astype(np.int32), 0, W - 1)
            sy = np.clip((ys[:, np.newaxis] + saz * dist).astype(np.int32), 0, H - 1)
            dz = d_sc[sy, sx] - d_sc
            np.maximum(h_max, np.arctan2(dz, dist), out=h_max)
        asvf_acc += w * np.cos(np.maximum(h_max, np.float32(0)))
    asvf = np.clip(asvf_acc / max(w_sum, 1e-6), 0, 1).astype(np.float32)

    _upd("Local Dominance…", 0.48)
    ld_acc = np.zeros((H, W), np.float32)
    n_ld = ld_max - ld_min + 1
    for _r in range(ld_min, ld_max + 1):
        local = cv2.GaussianBlur(d_sc, (0, 0), float(max(1, _r // 2)))
        nbhd = cv2.GaussianBlur(d_sc, (0, 0), float(_r))
        ld_acc += local / (nbhd + 1e-6)
    ld = _clip_norm(ld_acc / n_ld)

    _upd("Openness +/-…", 0.58)
    nz = np.clip(normals[..., 2], 0.0, 1.0)
    pos_open = _norm_01(np.power(nz, 0.8).astype(np.float32))
    neg_open = _norm_01(np.power(1.0 - nz, 0.8).astype(np.float32))

    _upd("DEV fine/med/coarse…", 0.64)
    dev_f = _clip_norm(d_sc - cv2.GaussianBlur(d_sc, (0, 0), float(r_f)))
    dev_m = _clip_norm(d_sc - cv2.GaussianBlur(d_sc, (0, 0), float(r_m)))
    dev_c = _clip_norm(d_sc - cv2.GaussianBlur(d_sc, (0, 0), float(r_c)))

    _upd("MSTP RGB…", 0.70)
    mstp_rgb = np.stack([dev_f, dev_m, dev_c], axis=-1).astype(np.float32)
    mstp_lit = np.power(np.clip(mstp_rgb, 0, 1), 1.0 / max(0.1, mstp_light))
    mstp_8 = (mstp_lit * 255).clip(0, 255).astype(np.uint8)

    _upd("RVT composite (SVF × Openness)…", 0.76)
    rvt_comp = _norm_01(svf * pos_open).astype(np.float32)

    _upd("HF+LF composite…", 0.82)
    slrm_hf = slrm
    lf_trend = cv2.GaussianBlur(d_sc, (0, 0), float(lf_r))
    slrm_lf = _clip_norm(d_sc - lf_trend)
    lf_weighted = slrm_lf * np.float32(blend_op)
    hrlc_map = np.clip(1.0 - (1.0 - slrm_hf) * (1.0 - lf_weighted), 0, 1)
    hrlc_map = _clip_norm(hrlc_map)
    p2s, p98s = float(np.percentile(hrlc_map, 2)), float(np.percentile(hrlc_map, 98))
    np.subtract(hrlc_map, p2s, out=hrlc_map)
    np.divide(hrlc_map, p98s - p2s + 1e-9, out=hrlc_map)
    np.clip(hrlc_map, 0, 1, out=hrlc_map)
    np.power(hrlc_map, 0.88, out=hrlc_map)
    alb_n = albedo / 255.0
    sel_col = np.clip(alb_n * hrlc_map[..., np.newaxis] * 1.4 * 0.55
                      + np.stack([hrlc_map] * 3, -1) * 0.45, 0, 1)
    hrlc_colour_8 = (sel_col * 255).clip(0, 255).astype(np.uint8)

    _upd("Hillshade from depth…", 0.88)
    hs_from_depth = hillshade_from_gradients(depth, _hs_az, _hs_alt, _hs_z)

    _upd("Done computation.", 1.0)
    return {
        "slrm": slrm,
        "msrm": msrm,
        "svf": svf,
        "asvf": asvf,
        "ld": ld,
        "pos_open": pos_open,
        "neg_open": neg_open,
        "dev_f": dev_f,
        "dev_m": dev_m,
        "dev_c": dev_c,
        "mstp_rgb": mstp_rgb,
        "mstp_8": mstp_8,
        "rvt_comp": rvt_comp,
        "hrlc_map": hrlc_map,
        "hrlc_colour_8": hrlc_colour_8,
        "hs_from_depth": hs_from_depth,
    }


# ── PBR texture map computation ──────────────────────────────────

def compute_detail_normals(
    normals: np.ndarray,
    depth: np.ndarray | None = None,
    step: int | None = None,
    sigma: float | None = None,
) -> np.ndarray:
    """Compute detail (high-frequency) normal map from full normals.

    Parameters
    ----------
    normals : (H, W, 3) float — PS normal map
    depth : (H, W) float or None — depth map (unused, kept for API compat)
    step : int or None — mesh step (unused, kept for API compat)
    sigma : float or None — Gaussian HP sigma; if None, derived from shape
    """
    n_f = normals.astype(np.float32)
    h, w = n_f.shape[:2]
    if sigma is None:
        sigma = max(3.0, 0.02 * math.sqrt(h * h + w * w))
    nm_hp_x = n_f[..., 0] - fast_large_sigma_blur(n_f[..., 0], sigma)
    nm_hp_y = n_f[..., 1] - fast_large_sigma_blur(n_f[..., 1], sigma)
    nm_hp_z = np.sqrt(np.clip(1.0 - (nm_hp_x ** 2 + nm_hp_y ** 2), 0.01, 1.0))
    nm_det = np.stack([nm_hp_x, nm_hp_y, nm_hp_z], axis=-1)
    return nm_det / (np.linalg.norm(nm_det, axis=-1, keepdims=True) + 1e-8)


def compute_ambient_occlusion(
    normals: np.ndarray,
    sigma: float | None = None,
) -> np.ndarray:
    """Compute ambient occlusion map from PS normal map."""
    n_f = normals.astype(np.float32)
    h, w = n_f.shape[:2]
    if sigma is None:
        sigma = max(5.0, 0.02 * math.sqrt(h * h + w * w))
    nz = np.clip(n_f[..., 2], 0.0, 1.0)
    ao = np.power(nz, 1.5)
    ao_blur = fast_large_sigma_blur(ao, sigma)
    return np.clip(ao - ao_blur * 0.35 + 0.35, 0, 1)


def compute_cavity_map(
    normals: np.ndarray,
    sigma: float | None = None,
) -> np.ndarray:
    """Compute cavity map from normal map divergence."""
    n_f = normals.astype(np.float32)
    h, w = n_f.shape[:2]
    if sigma is None:
        sigma = max(5.0, 0.02 * math.sqrt(h * h + w * w))
    div_n = normal_divergence(n_f[..., 0], n_f[..., 1], sobel=False)
    div_hp = div_n - fast_large_sigma_blur(div_n, sigma)
    return np.clip(-div_hp * 8.0 + 0.5, 0, 1)


def compute_curvature_rgb(
    normals: np.ndarray,
    sigma: float | None = None,
) -> np.ndarray:
    """Compute colour-coded curvature map (B=concave, G=flat, R=convex)."""
    n_f = normals.astype(np.float32)
    h, w = n_f.shape[:2]
    if sigma is None:
        sigma = max(2.0, 0.005 * math.sqrt(h * h + w * w))
    curv = normal_divergence(n_f[..., 0], n_f[..., 1], sobel=False)
    # Curvature's sigma is intentionally small/local (0.5% of diagonal, vs
    # 2-35% for the other PBR maps), so it falls under fast_large_sigma_blur's
    # default direct-blur cutoff — but "small sigma" isn't the same as "cheap":
    # on a 100MP image a direct blur at this sigma still measured ~15s.
    # A smaller small_target forces the fast downsample path here too
    # (verified: 15.3s -> 0.2s, correlation 0.9995 vs the direct blur).
    curv_sm = fast_large_sigma_blur(curv, max(1.0, sigma), small_target=16.0)
    p99 = float(np.percentile(np.abs(curv_sm), 99))
    curv_n = np.clip(curv_sm / p99 * 0.5 + 0.5, 0, 1)
    curv_r = np.clip((curv_n - 0.5) * 2.0, 0, 1)
    curv_g = np.clip((0.5 - curv_n) * 2.0, 0, 1)
    curv_b = 1.0 - np.abs(curv_n - 0.5) * 2.0
    return (np.stack([curv_b, curv_g, curv_r], axis=-1) * 255).clip(0, 255).astype(np.uint8)


def compute_depth_displacement(
    depth_or_normals: np.ndarray,
    sigma_hp: float | None = None,
) -> np.ndarray:
    """FFT depth integration → high-pass filtered displacement map.

    Parameters
    ----------
    depth_or_normals : (H, W) or (H, W, 3) float
        Either a pre-computed depth map (2D) OR a PS normal map (H, W, 3)
        from which depth is integrated via Frankot–Chellappa.
    sigma_hp : float or None
        Gaussian high-pass sigma; if None, derived from shape.
    """
    arr = depth_or_normals.astype(np.float32)
    if arr.ndim == 3:
        # Treat as normals → integrate via Frankot-Chellappa
        n_f = arr
        sh, sw = n_f.shape[:2]
        nz_c = np.maximum(n_f[..., 2], 0.001)
        p_fc, q_fc = depth_slopes(n_f[..., 0], n_f[..., 1], nz_c)
        depth_int = frankot_chellappa(p_fc, q_fc)
    else:
        depth_int = arr
        sh, sw = depth_int.shape
    if sigma_hp is None:
        sigma_hp = max(5.0, 0.35 * math.sqrt(sh * sh + sw * sw))
    disp_hp = depth_int - fast_large_sigma_blur(depth_int, sigma_hp)
    rng_d = disp_hp.max() - disp_hp.min()
    if rng_d > 1e-9:
        return ((disp_hp - disp_hp.min()) / rng_d).astype(np.float32)
    return (disp_hp * 0).astype(np.float32)


def compute_sds_maps(
    normals_crop: np.ndarray,
    alb_crop: np.ndarray,
    grey_stack: np.ndarray | None = None,
    l_mat: np.ndarray | None = None,
    spec_var: np.ndarray | None = None,
    median_img: np.ndarray | None = None,
    y1: int = 0, x1: int = 0,
    y2: int = 0, x2: int = 0,
) -> dict[str, np.ndarray]:
    """Specular/Diffuse Separation maps (Mallick et al. 2005).

    Returns dict with keys: diffuse_albedo_8/16, specular_8/32f,
    gloss_index_8/16, highlight_mask_8.
    """
    sh, sw = normals_crop.shape[:2]
    _n_f32 = normals_crop.astype(np.float32)
    _alb_f = alb_crop.astype(np.float32) / 255.0

    if l_mat is not None and grey_stack is not None and y2 > 0:
        _L32 = l_mat.astype(np.float32)
        _nflat = _n_f32.reshape(-1, 3)
        _ndotl = np.clip(_nflat @ _L32.T, 0, 1)
        _imgs_c = grey_stack[:, y1:y2, x1:x2].astype(np.float32) / 255.0
        _imgs_flat = _imgs_c.reshape(_imgs_c.shape[0], -1).T
        _num = np.sum(_imgs_flat * _ndotl, axis=1)
        _den = np.sum(_ndotl ** 2, axis=1) + 1e-8
        _rho = np.clip(_num / _den, 0, 1)
        _spec_res = np.clip(_imgs_flat - (_rho[:, np.newaxis] * _ndotl), 0, 1)
        _spec_res = np.max(_spec_res, axis=1).reshape(sh, sw)
        _diff_alb = _rho.reshape(sh, sw)
    elif median_img is not None:
        _med = median_img[y1:y2, x1:x2].astype(np.float32) / 255.0
        _sv = spec_var if spec_var is not None else np.zeros((sh, sw), np.float32)
        if _sv.ndim == 2 and _sv.shape[:2] == (sh, sw):
            # Already the crop size (caller pre-cropped, or no crop requested).
            _sv_c = _sv
        elif _sv.ndim == 2 and _sv.shape[0] >= y2 and _sv.shape[1] >= x2:
            # Full-resolution array (the normal case) — slice the crop region.
            # This branch used to be gated on _sv already matching the CROP
            # size instead of the SOURCE size, so a real crop always missed
            # it and fell through to resizing the whole frame down to crop
            # dimensions instead of slicing it — silently wrong whenever a
            # non-trivial crop was requested.
            _sv_c = _sv[y1:y2, x1:x2]
        else:
            _sv_c = cv2.resize(_sv.astype(np.float32), (sw, sh))
        _spec_res = np.clip(cv2.normalize(_sv_c.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX), 0, 1)
        _diff_alb = np.clip(_med - _spec_res * 0.5, 0, 1)
    else:
        _diff_alb = np.mean(_alb_f, axis=-1)
        _spec_res = np.zeros((sh, sw), np.float32)

    _gloss = np.clip(_spec_res / (_diff_alb + _spec_res + 1e-6), 0, 1)
    _gl_mu, _gl_sd = float(np.mean(_gloss)), float(np.std(_gloss))
    _hl_mask = ((_gloss > _gl_mu + 2.0 * _gl_sd).astype(np.uint8) * 255)
    _diff_ratio = np.clip(_diff_alb[:, :, np.newaxis] / (np.mean(_alb_f, axis=-1, keepdims=True) + 0.01), 0, 4)
    _diff_rgb = np.clip(_alb_f * _diff_ratio, 0, 1)

    return {
        "diffuse_albedo_8": (_diff_rgb * 255).clip(0, 255).astype(np.uint8),
        "diffuse_albedo_16": (_diff_rgb * 65535).clip(0, 65535).astype(np.uint16),
        "specular_8": (_spec_res * 255).clip(0, 255).astype(np.uint8),
        "specular_32f": _spec_res.astype(np.float32),
        "gloss_index_8": (_gloss * 255).clip(0, 255).astype(np.uint8),
        "gloss_index_16": (_gloss * 65535).clip(0, 65535).astype(np.uint16),
        "highlight_mask_8": _hl_mask,
    }


# ── Terrain analysis task dispatch ─────────────────────────────────

# ── Light-count confidence map ─────────────────────────────────────

def compute_light_count_map(grey_stack: np.ndarray) -> np.ndarray:
    """Count how many images contribute meaningful illumination at each pixel.

    Returns (H, W) float32 map in [0.05, 1.0].
    """
    from helio.core.parallel import by_rows, parts_for
    _gs_f32 = np.asarray(grey_stack, np.float32)       # no copy when it is already float32
    _per_img_med = np.median(_gs_f32, axis=(1, 2), keepdims=True)
    _thresh = np.maximum(_per_img_med * 0.05, 2.0)
    n, h = _gs_f32.shape[0], _gs_f32.shape[1]
    _lit = np.empty(_gs_f32.shape[1:], np.float32)

    def band(y0, y1):                  # in strips: the same counts, no whole-stack copies
        _lit[y0:y1] = (_gs_f32[:, y0:y1] > _thresh).sum(axis=0).astype(np.float32)
    by_rows(band, h, _gs_f32[0, :1].size * n, parts=parts_for(h, _gs_f32.size * 2))
    _n_total = float(max(1, grey_stack.shape[0]))
    return np.clip(_lit / _n_total, 0.05, 1.0).astype(np.float32)


# ── PTM / HSH coefficient solver ──────────────────────────────────

def compute_ptm_hsh_coefficients(
    L_mat: np.ndarray,
    flat_grey: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Solve PTM (6-coeff) and HSH (9-coeff) basis from calibrated lights.

    Returns (ptm_coeffs, hsh_coeffs), each (n_coeffs, n_pixels) float64.
    Either may be None if the solve fails.
    """
    if L_mat is None or len(L_mat) != flat_grey.shape[0]:
        return None, None
    U, V, Z = L_mat[:, 0], L_mat[:, 1], L_mat[:, 2]
    try:
        A_ptm = np.stack([U**2, V**2, U*V, U, V, np.ones_like(U)], axis=1)
        A_hsh = np.stack([np.ones_like(U), U, V, Z, U*V, U*Z, V*Z,
                          U**2 - V**2, 3*(Z**2) - 1], axis=1)
        # Every pixel is fitted on its own, so the pixels are fitted in strips,
        # in parallel: the same coefficients, without the whole stack turned to
        # float64 at once (12 GB on a 32-frame, 46-megapixel capture).
        from helio.core.parallel import lstsq_columns
        return lstsq_columns(A_ptm, flat_grey), lstsq_columns(A_hsh, flat_grey)
    except Exception:
        _log.warning("PTM fitting failed, returning None")
        return None, None


# ── Auto-orient normals ───────────────────────────────────────────

def auto_orient_normals(
    normals: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Ensure Nz (Z component) is positive — median check.

    Returns (normals, was_flipped).
    """
    _nz = normals[..., 2].astype(np.float32).flatten()
    _nz_median = float(np.median(_nz))
    _nz_p10 = float(np.percentile(_nz, 10))
    _needs_flip = _nz_median < 0.0 or _nz_p10 < -0.1
    if _needs_flip:
        return -normals, True
    return normals, False


# ── Depth post-processing pipeline ─────────────────────────────────

def _broad(depth, scale_d, mask=None):
    """The depth's broad shape (a 25 px blur on a map of at most 500 px), to
    take away for the high-pass view.  With *mask* (the object on a dark
    surround, helio.core.object_mask) the blur is of the object alone
    (normalised convolution), so its edge leaves no halo on the surround."""
    h_d, w_d = depth.shape
    size = (max(1, int(w_d * scale_d)), max(1, int(h_d * scale_d)))
    if mask is None:
        low = cv2.resize(depth, size, interpolation=cv2.INTER_AREA)
        return cv2.resize(cv2.GaussianBlur(low, (0, 0), sigmaX=25), (w_d, h_d),
                          interpolation=cv2.INTER_LINEAR)
    m = np.asarray(mask, np.float32)
    low_m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    low_d = cv2.resize(np.where(mask, depth, 0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    num = cv2.GaussianBlur(low_d, (0, 0), sigmaX=25)
    den = cv2.GaussianBlur(low_m, (0, 0), sigmaX=25)
    b = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-4)
    return cv2.resize(b, (w_d, h_d), interpolation=cv2.INTER_LINEAR)


def _stretch_hp(hp, lo_pct, mask=None):
    """The high-pass depth clipped at its extremes and stretched to 0-1; with
    *mask*, the extremes are the object's and the surround carries on from the
    object's rim (helio.core.object_mask.extend_from).  It sat at the middle of
    the range until 2026-11-20, a step up from a binding's rounded rim that the
    3D relief drew as a cliff all round the object."""
    if mask is None:
        p1, p99 = np.percentile(hp, lo_pct), np.percentile(hp, 100.0 - lo_pct)
        return cv2.normalize(np.clip(hp, p1, p99), None, 0, 1, cv2.NORM_MINMAX)
    from helio.core.object_mask import extend_from
    v = hp[mask]
    p1, p99 = float(np.percentile(v, lo_pct)), float(np.percentile(v, 100.0 - lo_pct))
    hp = np.where(mask, hp, extend_from(hp, mask)).astype(np.float32)
    span = max(p99 - p1, 1e-9)
    return np.clip((hp - p1) / span, 0, 1).astype(np.float32)


def depth_visualizations(
    depth_poisson: np.ndarray,
    normals_full: np.ndarray,
    clip_pct: float,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """High-pass and full-range normalised depth, and the Frankot-Chellappa
    alternative, from raw depth: what Process and Recalculate Depth both
    store (helio.application.process_tools).

    *mask* (helio.core.object_mask): the object on a dark surround, whose
    depth was integrated on it alone; the views then judge the object alone
    and show the surround flat.

    Returns dict with keys depth_vis_p, depth_normalised_full, depth_vis_fc."""
    from helio.core.depth_integration import frankot_chellappa

    h_d, w_d = depth_poisson.shape
    scale_d = min(1.0, 500.0 / max(h_d, w_d))
    if mask is not None and (np.shape(mask) != (h_d, w_d) or not np.any(mask)):
        mask = None

    # HP-filtered visualisation depth
    blur_p = _broad(depth_poisson, scale_d, mask)
    dp_hp = depth_poisson - blur_p
    _dcp = max(0.01, min(4.9, clip_pct))
    depth_vis_p = _stretch_hp(dp_hp, _dcp, mask)

    # Full-range (no HP) for 3D viewer
    _dp_full = depth_poisson.copy()
    _p1f = np.percentile(_dp_full, max(0.01, _dcp * 0.5))
    _p99f = np.percentile(_dp_full, 100.0 - max(0.01, _dcp * 0.5))
    depth_normalised_full = cv2.normalize(
        np.clip(_dp_full, _p1f, _p99f), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Frankot-Chellappa depth for alternate visualisation
    n_fc = normals_full
    if mask is not None:
        from helio.core.object_mask import flatten_surround
        n_fc = flatten_surround(normals_full, mask)
    depth_fc = frankot_chellappa(*normal_slopes(n_fc))
    blur_fc = _broad(depth_fc, scale_d, mask)
    dfc_hp = depth_fc - blur_fc
    depth_vis_fc = _stretch_hp(dfc_hp, 1, mask)
    del dp_hp, blur_p, _dp_full, depth_fc, blur_fc, dfc_hp
    return {"depth_vis_p": depth_vis_p, "depth_normalised_full": depth_normalised_full,
            "depth_vis_fc": depth_vis_fc}


def compute_depth_visualizations(
    depth_poisson: np.ndarray,
    normals_full: np.ndarray,
    grey_stack: np.ndarray,
    h: int, w: int,
    clip_pct: float,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """High-pass normalisation + Frankot-Chellappa + PCA from raw depth.

    Returns dict with keys:
        depth_vis_p, depth_normalised_full, depth_vis_fc, pca_full
    """
    out = depth_visualizations(depth_poisson, normals_full, clip_pct, mask)

    # PCA of greyscale stack
    # asarray: no copy of an already-float32 stack (astype copied all 5.9 GB)
    flat_gs = np.asarray(grey_stack.reshape(grey_stack.shape[0], -1), np.float32)
    gs_centered = flat_gs - np.mean(flat_gs, axis=0)
    evals, evecs = np.linalg.eigh(
        np.dot(gs_centered, gs_centered.T) / gs_centered.shape[1])
    evecs = evecs[:, np.argsort(evals)[::-1]]
    pca_comps = np.dot(gs_centered.T, evecs[:, :3])
    pca_img = np.zeros((gs_centered.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        comp = pca_comps[:, i]
        if i < evecs.shape[1] and skew_sign(comp, evecs[:, i]) < 0:
            comp = -comp
        c_min, c_max = np.min(comp), np.max(comp)
        if c_max > c_min:
            pca_img[:, i] = ((comp - c_min) / (c_max - c_min) * 255).astype(np.uint8)
    out["pca_full"] = pca_img.reshape(h, w, 3)

    del flat_gs, gs_centered, evals, evecs, pca_comps, pca_img
    gc.collect()
    return out


# ── Terrain analysis pipeline ─────────────────────────────────────

def compute_terrain_analysis(
    depth_vis_p: np.ndarray,
    normals_full: np.ndarray | None,
    h: int, w: int,
) -> dict[str, np.ndarray]:
    """Compute all terrain analysis maps at a fixed working resolution.

    Returns dict with all terrain maps resampled to working resolution.
    """
    if depth_vis_p is None or depth_vis_p.size == 0 or min(h, w) < 1:
        raise ValueError(
            f"compute_terrain_analysis: need a non-empty depth map, got "
            f"shape={None if depth_vis_p is None else depth_vis_p.shape} h={h} w={w}")
    # Clamp to 1.0: this used to UPSAMPLE any capture under 1380 px, so a
    # 96x128 stack had its terrain analysed at 1035x1380 - 145x the source
    # pixels, inventing detail that was never captured and making small inputs
    # pathologically slow. Same missing clamp as the proxy-scale bug.
    _work_sc = min(1.0, 1380.0 / max(h, w))
    # Never let rounding collapse an axis to zero: cv2.resize raises
    # "!ssize.empty()" on a 0-length side, which is the small-image crash.
    _work_h = max(2, int(round(h * _work_sc)))
    _work_w = max(2, int(round(w * _work_sc)))
    if (_work_h, _work_w) == depth_vis_p.shape[:2]:
        d_work = depth_vis_p.astype(np.float32, copy=True)
    else:
        d_work = cv2.resize(depth_vis_p, (_work_w, _work_h),
                            interpolation=cv2.INTER_AREA).astype(np.float32)

    result: dict[str, np.ndarray] = {}
    _aggressive_cleanup_after_terrain = True

    # Multi-directional hillshade — Horn's Sobel, no pre-blur
    _d_rng = float(np.ptp(d_work))
    if _d_rng < 1e-9:
        result["_multi_hs"] = np.full(d_work.shape, 0.3, dtype=np.float32)
    else:
        _dm = (d_work - d_work.min()) / _d_rng * min(_work_h, _work_w) / 4.0
        _gx = cv2.Sobel(_dm, cv2.CV_32F, 1, 0, ksize=3)
        _gy = cv2.Sobel(_dm, cv2.CV_32F, 0, 1, ksize=3)
        _sl = np.pi / 2 - np.arctan(np.sqrt(_gx**2 + _gy**2))
        _asp = np.arctan2(-_gy, -_gx)
        _sa = np.deg2rad(45)
        _mhs = np.zeros_like(d_work)
        for _az in [0, 45, 90, 135, 180, 225, 270, 315]:
            _mhs += np.clip(
                np.sin(_sa) * np.sin(_sl)
                + np.cos(_sa) * np.cos(_sl) * np.cos(np.deg2rad(_az) - _asp),
                0, 1)
        result["_multi_hs"] = np.clip(_mhs / 8.0 + 0.1, 0, 1).astype(np.float32)

    # Sky-View Factor (arctan formulation; scale-aware sigma)
    _svf_sigma = max(_work_h, _work_w) * 0.025
    _svf_ao = cv2.GaussianBlur(d_work, (0, 0), sigmaX=_svf_sigma) - d_work
    result["_svf"] = np.clip(0.5 + np.arctan(_svf_ao * 8.0) / np.pi,
                             0, 1).astype(np.float32)


    # MSRM — weighted HP sum at 4 scales
    _msrm = np.zeros_like(d_work)
    for _sig, _wt in [(2, 0.25), (5, 0.35), (15, 0.25), (40, 0.15)]:
        _msrm += (d_work - cv2.GaussianBlur(d_work, (0, 0), sigmaX=float(_sig))) * _wt
    cv2.normalize(_msrm, _msrm, 0, 1, cv2.NORM_MINMAX)
    result["_msrm"] = _msrm.astype(np.float32)

    # Surface Curvature — normal map divergence
    if normals_full is not None:
        _nf_c = cv2.resize(normals_full.astype(np.float32), (_work_w, _work_h),
                           interpolation=cv2.INTER_AREA)
        _curv_raw = normal_divergence(_nf_c[..., 0], _nf_c[..., 1])
        _p1c = float(np.percentile(_curv_raw, 1))
        _p99c = float(np.percentile(_curv_raw, 99))
        _half_range = max(abs(_p1c), abs(_p99c), 1e-8)
        result["_curv"] = np.clip(
            _curv_raw / _half_range * 0.5 + 0.5, 0, 1).astype(np.float32)
    else:
        result["_curv"] = np.clip(
            cv2.Laplacian(d_work / max(np.ptp(d_work), 1e-6)
                          * min(_work_h, _work_w) / 4.0, cv2.CV_32F) * 8.0 + 0.5,
            0, 1).astype(np.float32)

    # Slope Gradient
    _gx2 = cv2.Sobel(d_work, cv2.CV_32F, 1, 0, ksize=3)
    _gy2 = cv2.Sobel(d_work, cv2.CV_32F, 0, 1, ksize=3)
    result["_slope"] = cv2.normalize(
        np.sqrt(_gx2**2 + _gy2**2), None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Local Dominance
    _ld_sigma = max(_work_h, _work_w) * 0.04
    result["_loc_dom"] = cv2.normalize(
        d_work - cv2.GaussianBlur(d_work, (0, 0), sigmaX=_ld_sigma),
        None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    # Positive / Negative Openness
    _svf = result["_svf"]
    if HAS_RVT:
        import rvt.vis as _rvt_vis
        try:
            _opn_res = _rvt_vis.sky_view_factor(dem=d_work, resolution=1.0, compute_opns=True)
            result["_pos_opn"] = cv2.normalize(
                _opn_res["opns"], None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
            _opn_neg = _rvt_vis.sky_view_factor(dem=-d_work, resolution=1.0, compute_opns=True)
            result["_neg_opn"] = cv2.normalize(
                _opn_neg["opns"], None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        except Exception:
            _log.warning("RVT openness computation failed, using SVF approximation")
            result["_pos_opn"] = _svf.copy()
            result["_neg_opn"] = (1.0 - _svf).clip(0, 1)
    else:
        result["_pos_opn"] = _svf.copy()
        result["_neg_opn"] = (1.0 - _svf).clip(0, 1)

    # SLRM
    if HAS_RVT:
        import rvt.vis as _rvt_vis
        try:
            result["_slrm"] = cv2.normalize(
                _rvt_vis.slrm(dem=d_work, radius_cell=20),
                None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        except Exception:
            _log.warning("RVT SLRM failed, falling back to Gaussian blur subtraction")
            _slrm_s = max(_work_h, _work_w) * 0.015
            result["_slrm"] = cv2.normalize(
                (d_work - cv2.GaussianBlur(d_work, (0, 0), sigmaX=_slrm_s)) * 10 + 0.5,
                None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
    else:
        _slrm_s = max(_work_h, _work_w) * 0.015
        result["_slrm"] = cv2.normalize(
            (d_work - cv2.GaussianBlur(d_work, (0, 0), sigmaX=_slrm_s)) * 10 + 0.5,
            None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)

    if _aggressive_cleanup_after_terrain:
        del d_work, depth_vis_p, normals_full
        gc.collect()
    return result


TERRAIN_TASKS = {
    "Hillshade": ("hs_azi", "hs_alt"),
    "Multi-Dir Hillshade": ("n_dirs", "hs_alt"),
    "Slope Gradient": ("unit",),
    "SLRM": ("radius",),
    "MSRM": ("msrm_min", "msrm_max"),
    "SVF": ("svf_radius", "svf_dirs"),
    "Anisotropic SVF": ("svf_radius", "svf_dirs", "hs_azi", "aniso"),
    "Positive Openness": ("op_radius", "svf_dirs"),
    "Negative Openness": ("op_radius", "svf_dirs"),
    "Local Dominance": ("ld_radius", "ld_max"),
    "Surface Curvature": (),
}
"""Map of terrain task names to their parameter key names (for UI forms)."""


def phase_align_stack(
    grey_stack: np.ndarray | list[np.ndarray],
    rgb_list: list[np.ndarray],
) -> list[tuple[float, float]]:
    """Phase-correlation alignment for photometric stereo image stacks.

    Uses FFT phase spectrum (illumination-invariant) so dramatically
    different per-image lighting does not confuse the alignment.
    Reference = median stack (already the albedo approximation).
    Hanning window eliminates FFT edge-ringing artefacts.
    Shifts capped at ±5 % of image diagonal to avoid runaway frames.

    Near-dark frames (std < 2) are aligned using a foreground mask —
    the phase correlation is computed only over the bright region of the
    *reference* image, which avoids the large dark background biasing
    the alignment for those frames.

    Correction applied in-place to grey_stack and rgb_list via warpAffine
    (INTER_LINEAR + BORDER_REFLECT to avoid black borders).

    Returns list of (dx, dy) floats, one per image.
    """
    h, w = grey_stack[0].shape[:2]
    diag = (h**2 + w**2) ** 0.5
    max_sh = diag * 0.05
    ref_f = np.median(grey_stack, axis=0).astype(np.float64)
    win = cv2.createHanningWindow((w, h), cv2.CV_64F)

    _ref_norm = ref_f.copy()
    cv2.normalize(_ref_norm, _ref_norm, 0, 1, cv2.NORM_MINMAX)
    _fg_mask = (_ref_norm > 0.05).astype(np.float64)
    cv2.GaussianBlur(_fg_mask, (0, 0), sigmaX=max(2.0, w * 0.01), dst=_fg_mask)
    np.clip(_fg_mask, 0, 1, out=_fg_mask)

    shifts = []
    for i in range(len(grey_stack)):
        g = grey_stack[i].astype(np.float64)
        if g.std() >= 2.0:
            (dx, dy), _ = cv2.phaseCorrelate(ref_f * win, g * win)
        else:
            (dx, dy), _ = cv2.phaseCorrelate(
                ref_f * _fg_mask * win,
                g * _fg_mask * win)

        if abs(dx) > max_sh or abs(dy) > max_sh:
            shifts.append((0.0, 0.0))
            continue
        shifts.append((dx, dy))
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        grey_stack[i] = cv2.warpAffine(grey_stack[i], M, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
        if i < len(rgb_list):
            rgb_list[i] = cv2.warpAffine(rgb_list[i], M, (w, h),
                                          flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT)
    return shifts


def generate_dzi(
    image_array: np.ndarray,
    output_dir: str,
    name: str,
    tile_size: int = 254,
    overlap: int = 1,
    jpeg_q: int = 88,
) -> str:
    """Generate a DeepZoom Image pyramid (.dzi + _files/) from an RGB/Grey array.

    tile_size=254 + overlap=1 → 256×256 actual tiles (Relight / OpenSeadragon
    convention).  Using 254+1 avoids sub-pixel seam artefacts at tile boundaries
    because adjacent tiles share a 1-px border strip.
    jpeg_q: JPEG quality, 88 is a good balance of size vs quality for analytical images.

    Returns path to the .dzi manifest file.
    """
    import math as _math
    import os

    os.makedirs(output_dir, exist_ok=True)
    h, w = image_array.shape[:2]
    if image_array.ndim == 2:
        img3 = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        img3 = image_array[..., :3]
    else:
        img3 = image_array
    img3 = img3.astype(np.uint8)

    max_level = int(_math.ceil(_math.log2(max(w, h)))) if max(w, h) > 1 else 0
    files_dir = os.path.join(output_dir, f"{name}_files")
    os.makedirs(files_dir, exist_ok=True)

    curr = img3.copy()
    for level in range(max_level, -1, -1):
        ch, cw = curr.shape[:2]
        level_dir = os.path.join(files_dir, str(level))
        os.makedirs(level_dir, exist_ok=True)
        n_cols = int(_math.ceil(cw / tile_size))
        n_rows = int(_math.ceil(ch / tile_size))
        for row in range(n_rows):
            for col in range(n_cols):
                x0 = max(0, col * tile_size - overlap)
                y0 = max(0, row * tile_size - overlap)
                x1 = min(cw, (col + 1) * tile_size + overlap)
                y1 = min(ch, (row + 1) * tile_size + overlap)
                tile = curr[y0:y1, x0:x1]
                tile_path = os.path.join(level_dir, f"{col}_{row}.jpg")
                cv2.imwrite(tile_path,
                            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
        if cw > 1 or ch > 1:
            curr = cv2.resize(curr,
                              (max(1, cw // 2), max(1, ch // 2)),
                              interpolation=cv2.INTER_AREA)

    dzi_path = os.path.join(output_dir, f"{name}.dzi")
    with open(dzi_path, "w") as f:
        f.write(
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"'
            f' Format="jpg" Overlap="{overlap}" TileSize="{tile_size}">'
            f'<Size Height="{h}" Width="{w}"/>'
            f'</Image>'
        )
    del img3, curr, image_array
    gc.collect()
    return dzi_path


def lighten_hex(hex_color: str, amount: float = 0.15) -> str:
    """Return a slightly lighter hex colour for hover states."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = min(255, int(r + (255 - r) * amount))
        g = min(255, int(g + (255 - g) * amount))
        b = min(255, int(b + (255 - b) * amount))
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        _log.warning("lighten_color failed, returning original color")
        return hex_color


def default_ring_positions(n: int) -> list[tuple[float, float, float]]:
    """Generate default light ring positions for dome calibration.

    Returns *n* (x, y, z) vectors distributed across elevation rings:
      8 at 60°, 16 at 38°, 32 at 22°, remainder at zenith.
    """
    import math as _math
    vecs, remaining = [], n
    for cap, elev in [(8, 60), (16, 38), (32, 22)]:
        count = min(remaining, cap)
        r_xy = _math.cos(_math.radians(elev))
        lz = _math.sin(_math.radians(elev))
        for k in range(count):
            az = 2 * _math.pi * k / count
            vecs.append((r_xy * _math.cos(az), r_xy * _math.sin(az), lz))
        remaining -= count
        if remaining <= 0:
            break
    for _ in range(remaining):
        vecs.append((0.0, 0.0, 1.0))
    return vecs[:n]


def load_reference_normal_map(path: str, h: int, w: int) -> np.ndarray:
    """Load a normal map PNG/TIF and convert to float32 [-1, 1] format.

    Auto-detects 8-bit (0-255) and 16-bit (0-65535) images.
    Resizes to (h, w) if dimensions differ and re-normalises unit length.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.dtype == np.uint16:
        n = img.astype(np.float32) / 32767.5 - 1.0
    else:
        n = img.astype(np.float32) / 127.5 - 1.0
    if n.shape[:2] != (h, w):
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_LINEAR)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, mag, out=np.zeros_like(n), where=mag > 1e-9)
    return n.astype(np.float32)


def apply_flat_correction_to_normals(
    normals: np.ndarray,
    flat_ref: np.ndarray | None,
    strength: float,
) -> np.ndarray:
    """Subtract a systematic distortion field measured from a flat reference.

    Parameters
    ----------
    normals : (H, W, 3) float32
        Normal map to correct.
    flat_ref : (H, W, 3) float32 or None
        Flat-reference normal field. If None, returns *normals* unchanged.
    strength : float
        Correction strength in [0, 1].

    Returns
    -------
    (H, W, 3) float32 corrected normal map.
    """
    if flat_ref is None:
        return normals
    h, w = normals.shape[:2]
    flat = flat_ref
    if flat.shape[:2] != (h, w):
        flat = cv2.resize(flat, (w, h), interpolation=cv2.INTER_LINEAR)
    deviation = flat - np.array([0.0, 0.0, 1.0], dtype=np.float32)
    corrected = normals - strength * deviation
    mag = np.linalg.norm(corrected, axis=-1, keepdims=True)
    return np.divide(corrected, mag, out=np.zeros_like(corrected), where=mag > 1e-9)


def _trimmed_block(I, Lm, n_img: int, trim_high: int, trim_low: int) -> np.ndarray:
    """trimmed_photometric_stereo for one strip of pixels: (3, pixels) unit normals."""
    # Rank each pixel's observations by intensity, then keep the middle band.
    if trim_high or trim_low:
        order = np.argsort(I, axis=0)
        rank = np.empty_like(order)
        np.put_along_axis(
            rank, order,
            np.broadcast_to(np.arange(n_img, dtype=order.dtype)[:, None], I.shape),
            axis=0)
        W = ((rank >= trim_low) & (rank < n_img - trim_high)).astype(np.float32)
        del order, rank
    else:
        W = np.ones_like(I)

    # Weighted normal equations, per pixel: (L^T W L) g = L^T W i
    ATA = np.einsum('ka,kb,kp->abp', Lm, Lm, W, optimize=True)
    ATb = np.einsum('ka,kp->ap', Lm, W * I, optimize=True)
    del W

    a, b, c = ATA[0, 0], ATA[0, 1], ATA[0, 2]
    d, e, f = ATA[1, 1], ATA[1, 2], ATA[2, 2]
    del ATA
    # Analytic symmetric-3x3 inverse: a batched np.linalg.solve over millions
    # of pixels is far slower and raises on any singular pixel.
    c00, c01, c02 = d * f - e * e, c * e - b * f, b * e - c * d
    c11, c12, c22 = a * f - c * c, b * c - a * e, a * d - b * b
    det = a * c00 + b * c01 + c * c02
    ok = np.abs(det) > 1e-12
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=ok)

    G = np.empty((3, I.shape[1]), np.float32)
    G[0] = (c00 * ATb[0] + c01 * ATb[1] + c02 * ATb[2]) * inv
    G[1] = (c01 * ATb[0] + c11 * ATb[1] + c12 * ATb[2]) * inv
    G[2] = (c02 * ATb[0] + c12 * ATb[1] + c22 * ATb[2]) * inv
    G[:, ~ok] = 0.0

    mag = np.linalg.norm(G, axis=0)
    N = np.divide(G, mag, out=np.zeros_like(G), where=mag > 1e-8)
    return N


def trimmed_photometric_stereo(
    grey_stack, L, h: int, w: int, trim_high: int = 1, trim_low: int = 1,
) -> np.ndarray:
    """Photometric stereo that drops each pixel's brightest and darkest reads.

    Plain least squares assumes every observation is Lambertian. The brightest
    read at a pixel is usually a specular highlight and the darkest is usually a
    shadow or an occluded light; both break that assumption and drag the fitted
    normal off. This trims them *per pixel* rather than per frame, so every
    light keeps contributing wherever it is well behaved — which matters on
    parchment and gilding, where the specular pixel set moves with the light.

    Returns (h, w, 3) unit normals. Pixels with a degenerate light
    configuration come back as zeros.
    """
    I_all = np.asarray(grey_stack, dtype=np.float32).reshape(len(grey_stack), -1)
    Lm = np.asarray(L, dtype=np.float32)
    n_img = I_all.shape[0]
    if Lm.shape[0] != n_img:
        raise ValueError(f"L has {Lm.shape[0]} rows for {n_img} images")

    if n_img - trim_high - trim_low < 3:      # need 3 good reads to solve
        trim_high = trim_low = 0

    # Every pixel is solved on its own, so the pixels go in strips, in
    # parallel: the same normals, without the whole-stack rankings (two
    # 64-bit arrays, 12 GB each on a 32-frame, 46-megapixel capture).
    from helio.core.parallel import by_rows, parts_for
    n_px = I_all.shape[1]
    N = np.empty((3, n_px), np.float32)

    def band(p0, p1):
        N[:, p0:p1] = _trimmed_block(I_all[:, p0:p1], Lm, n_img, trim_high, trim_low)
    by_rows(band, n_px, n_img, parts=parts_for(n_px, n_px * n_img * 30))
    N = N.T.reshape(h, w, 3)
    # Same hemisphere convention the rest of the pipeline uses.
    flip = N[..., 2] < 0
    N[flip] *= -1.0
    return np.ascontiguousarray(N, dtype=np.float32)


def normal_uncertainty_deg(L, g, r2, n):
    """Per-pixel 1-sigma angular uncertainty of a least-squares normal, degrees.

    With noise estimated from each pixel's own fit, sigma^2 = |r|^2 / (n - 3),
    the scaled normal g = rho N has covariance C = sigma^2 (L^T L)^-1.  Only
    the part of C across the normal turns it, so the angular spread is
    sqrt(trace C - N^T C N) / rho.  A pixel where the lights agree is sharp;
    one where they disagree (shadow, specular, a moving object) is not.
    (Least-squares error propagation; see e.g. Woodham 1994, JOSA A 11(11).)
    """
    Ci = np.linalg.inv(np.asarray(L, np.float64).T @ np.asarray(L, np.float64)).astype(np.float32)
    rho = np.linalg.norm(g, axis=0)
    N = np.divide(g, rho, out=np.zeros_like(g), where=rho > 1e-6)
    s2 = (r2 / max(n - 3, 1)).astype(np.float32)
    nCn = np.einsum('ip,ij,jp->p', N, Ci, N, optimize=True)
    var = s2 * np.maximum(np.trace(Ci) - nCn, 0.0)
    sig = np.sqrt(var) / np.maximum(rho, 1e-3)
    sig = np.degrees(np.arctan(sig)).astype(np.float32)
    sig[rho <= 1e-3] = 90.0
    return sig


def stack_quality_maps(grey_stack, L_mat, h, w, solve_stack=None):
    """Per-pixel capture-quality and legibility maps from the whole light stack.

    Every render engine in HELIO answers "what does this surface look like".
    None of them answers "should I trust this pixel", which on a real capture
    is a different and often more urgent question: a pixel that was blown out
    in six of eight frames has a normal derived from two usable observations,
    and nothing on screen says so.

    Must be computed while the stack is in scope. HELIO keeps normals, depth,
    coefficients and terrain maps after a run but discards ``grey_stack``, so
    an engine cannot derive these lazily without re-running the whole
    preprocessing chain (crop, flatten, median-normalise, CLAHE) and risking
    divergence from the main path - the same reasoning that put
    ``trimmed_photometric_stereo`` here.

    Returns a dict of uint8 (h, w) maps:

    ``resid``   RMS residual of the Lambertian fit (absent if uncalibrated)
    ``fusion``  contrast-weighted fusion across the stack
    ``sigma_deg`` float32: the normal's 1-sigma angular uncertainty in degrees
                (absent if uncalibrated or with only 3 lights)

    The residual and the uncertainty are measured on *solve_stack* when it
    is given - the linear-light intensities the normals were solved from
    (helio.core.radiometry) - and the fusion on *grey_stack*, which is for
    looking at.

    The clipping and shadow-frequency maps that were here were removed on
    Scott's review: correct, but on a good capture they render as a flat green
    page and a flat black one, which is not worth a dropdown slot or the
    memory.
    """
    out = {}
    stack = np.asarray(grey_stack, dtype=np.float32)
    if stack.ndim != 3 or stack.shape[0] < 2:
        return out
    n = stack.shape[0]

    # ── Lambertian fit residual. Solve per-pixel albedo*normal by least
    # squares, then measure how far the observations actually sit from the
    # model. High residual means specular, shadowed, or interreflected - the
    # places where a photometric-stereo normal is least believable, and where
    # gilding and metal point show up.
    try:
        if L_mat is not None and len(L_mat) == n:
            Lm = np.asarray(L_mat, np.float32)
            src = stack if solve_stack is None else np.asarray(solve_stack, np.float32)
            flat = src.reshape(n, -1)
            # In strips: the same fit and residual per pixel, without the
            # whole-stack prediction and its difference held at once.
            from helio.core.parallel import by_rows, lstsq_columns, parts_for
            g = lstsq_columns(Lm, flat)                          # (3, h*w)
            n_px = flat.shape[1]
            r2 = np.empty(n_px, np.result_type(flat.dtype, Lm.dtype, g.dtype))

            def band(a, b):
                r2[a:b] = np.sum((flat[:, a:b] - Lm @ g[:, a:b]) ** 2, axis=0)
            by_rows(band, n_px, n, parts=parts_for(n_px, n_px * n * 4 * 3))
            resid = np.sqrt(r2 / n)
            scale = float(np.percentile(resid, 99))
            out['resid'] = np.clip(resid.reshape(h, w) / max(scale, 1e-6)
                                   * 255.0, 0, 255).astype(np.uint8)
            if n > 3:
                out['sigma_deg'] = normal_uncertainty_deg(Lm, g, r2, n).reshape(h, w)
    except Exception:
        _log.exception("Lambertian fit residual failed; skipping that map")

    # ── Contrast-weighted multi-light fusion. Faded ink is legible under some
    # light directions and invisible under others, and a median or mean over
    # the stack averages the good frames away. Weight each observation by local
    # contrast and by how well exposed it is, then combine - Mertens-style
    # exposure fusion, applied over light directions instead of exposures.
    try:
        wsum = np.zeros((h, w), np.float32)
        acc = np.zeros((h, w), np.float32)
        for i in range(n):
            f = stack[i]
            contrast = np.abs(cv2.Laplacian(f, cv2.CV_32F, ksize=3))
            # Well-exposedness: a Gaussian about mid-grey, so blown and
            # crushed observations are discounted rather than trusted equally.
            wexp = np.exp(-((f - 128.0) ** 2) / (2.0 * 60.0 ** 2))
            wgt = contrast * wexp + 1e-3
            acc += f * wgt
            wsum += wgt
        fused = acc / np.maximum(wsum, 1e-6)
        lo, hi = np.percentile(fused, (0.5, 99.5))
        fused = (fused - lo) / max(hi - lo, 1e-6)
        out['fusion'] = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        _log.exception("multi-light fusion failed; skipping that map")

    return out
