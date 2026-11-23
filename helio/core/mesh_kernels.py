"""Mesh-building kernels: the export depth grid, its filters, normal detail baked into
it, grid and watertight meshes, binary STL.  Pure NumPy/OpenCV, GUI-free.

Moved out of helio.core.processing (November 2026), which still re-exports every name.
"""

from __future__ import annotations

import logging
import math
import math as _math
import os
import struct as _struct
from typing import Callable

import cv2
import numpy as np

from helio.core.depth_integration import frankot_chellappa, simchony_depth_integration
from helio.core.frames import depth_slopes
from helio.config import MESH_MAX_RES

_log = logging.getLogger(__name__)



def fast_large_sigma_blur(img: np.ndarray, sigma: float, small_target: float = 64.0) -> np.ndarray:
    """Gaussian-blur ``img`` with a large ``sigma`` efficiently.

    ``cv2.GaussianBlur``'s kernel radius scales linearly with sigma, so when
    sigma is comparable to the image's own dimensions (as used for macro-scale
    high-pass filtering — e.g. removing global shape from a depth map before
    3D export) the direct convolution cost scales with image_size x
    kernel_size: a single such blur on a 2500x2500 export grid measured ~10s.

    A blur that wide only recovers *low*-frequency content, so downsampling
    first loses nothing: shrink the image until sigma maps to roughly
    ``small_target`` px, blur there (cheap), then upsample back. Measured
    ~370x faster with <0.3% relative error and >0.999 correlation against the
    direct blur on realistic depth data.
    """
    h, w = img.shape[:2]
    if sigma <= small_target * 1.5:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    scale = small_target / sigma
    sh, sw = max(4, int(h * scale)), max(4, int(w * scale))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    small_blur = cv2.GaussianBlur(small, (0, 0), small_target)
    return cv2.resize(small_blur, (w, h), interpolation=cv2.INTER_LINEAR)



def subtract_parabolic_baseline(depth_map: np.ndarray) -> np.ndarray:
    """Fit a bivariate quadratic (poly22) surface to *depth_map* and subtract it.

    Frankot-Chellappa / FFT-based normal integration accumulates a smooth
    low-order bowl/saddle bias from noisy real-world normal maps. A Gaussian
    high-pass (``depth - blur(depth, sigma)``) does *not* remove this: blurring
    a quadratic surface returns the same quadratic plus a constant offset, so
    the bowl survives the subtraction almost untouched. Explicitly fitting and
    removing the low-order polynomial term is required to flatten it.

    Model: z(x,y) = a*x^2 + b*y^2 + c*x*y + d*x + e*y + f
    """
    from helio.core.depth_integration import remove_poly22
    return remove_poly22(depth_map)



def repair_nonfinite(a: np.ndarray, name: str = "depth map",
                     max_fraction: float = 0.02) -> np.ndarray:
    """Fill NaN and inf with the nearest finite value, or refuse if there are
    too many.

    A single bad pixel used to destroy an entire export in silence. The flat-map
    guard downstream is ``if rng < 1e-9``, and ``NaN < 1e-9`` is False, so it
    never fired; the NaN then propagated through the Gaussian blurs until the
    whole surface was non-finite. Measured on a 200x300 map with one NaN pixel:
    the STL still wrote 12 MB, its Z extent collapsed from 6.00 mm to 0.00 mm,
    the mesh was not watertight, and the app reported "3D-print ready". A
    single inf did the same.

    Refusing outright would be worse than the disease - one dead pixel should
    not block a print - so a small number are repaired and reported, and only a
    genuinely broken map is refused.
    """
    a = np.asarray(a)
    if a.size == 0:
        return a
    bad = ~np.isfinite(a)
    n_bad = int(bad.sum())
    if n_bad == 0:
        return a
    frac = n_bad / a.size
    if frac > max_fraction:
        raise ValueError(
            f"{name} is {frac*100:.1f}% NaN or infinite ({n_bad:,} of "
            f"{a.size:,} samples). Re-solve before exporting - repairing this "
            "much would be inventing the surface rather than measuring it.")
    if n_bad == a.size:
        raise ValueError(f"{name} is entirely NaN or infinite.")

    out = np.array(a, dtype=np.float32, copy=True)
    try:
        from scipy import ndimage as _ndi
        # Index of the nearest finite sample for every bad one.
        idx = _ndi.distance_transform_edt(bad, return_distances=False,
                                          return_indices=True)
        out[bad] = out[tuple(i[bad] for i in idx)]
    except Exception:
        # Without scipy, the finite median is a poorer but safe stand-in.
        out[bad] = float(np.median(out[~bad]))
    _log.warning("%s: repaired %d non-finite sample(s) (%.4f%%) by nearest-"
                 "neighbour fill before export", name, n_bad, frac * 100)
    return out



def compute_export_depth_grid(
    depth_vis_p: np.ndarray,
    depth_normalised: np.ndarray | None,
    normals: np.ndarray | None,
    settings: dict,
    progress_cb: Callable[[str, float], None] | None = None,
) -> np.ndarray:
    """Produce a normalised [0,1] depth grid for 3D export.

    Supports three paths:

    * **HD** (``use_hd=True``) — Frankot–Chellappa integration from PS
      normals.  Bypasses *depth_vis_p* and preserves carved lines.

    * **Hybrid** (``use_hybrid=True``) — blends *depth_normalised* with
      *depth_vis_p* at the given *detail_mix* ratio.

    * **Standard** — blends *depth_normalised* and *depth_vis_p* via
      *shape_blend*.

    Parameters
    ----------
    depth_vis_p : np.ndarray
        High-pass filtered depth (HP detail layer).
    depth_normalised : np.ndarray or None
        Full-range normalised depth (global macro-shape).
    normals : np.ndarray or None
        PS normal map (H, W, 3), required for HD path.
    settings : dict
        Keys: ``use_hd``, ``use_hybrid``, ``detail_mix``, ``shape_blend``,
        ``hf_cutoff``, ``amp``, ``max_res``, ``fz``, ``mirror_x``,
        ``step``, ``hp_sigma``, ``smooth``, ``warp_rem``, ``micro``,
        ``hd_warp_pct``.
    progress_cb : callable or None
        ``(message, fraction)``.

    Returns
    -------
    np.ndarray
        Normalised depth grid (float32, values in [0, 1]).
    """
    # Repair on the way in, before any blur can spread a single bad sample
    # across the whole surface.
    depth_vis_p = repair_nonfinite(depth_vis_p, "depth map")
    if depth_normalised is not None:
        depth_normalised = repair_nonfinite(depth_normalised, "full-range depth")
    if normals is not None:
        normals = repair_nonfinite(normals, "normal map")

    use_hd = settings.get("use_hd", False)
    use_hybrid = settings.get("use_hybrid", False)
    detail_mix = settings.get("detail_mix", 0.12)
    sb = max(0.0, min(1.0, settings.get("shape_blend", 0.55)))
    hf_cutoff = settings.get("hf_cutoff", 0.05)
    amp = settings.get("amp", 1.0)
    max_res = settings.get("max_res", MESH_MAX_RES)
    fz = settings.get("fz", 1)
    mirror_x = settings.get("mirror_x", False)
    hd_warp_pct = settings.get("hd_warp_pct", 0.18)

    if use_hd and normals is not None:
        if progress_cb:
            progress_cb("HD: integrating PS normals (FC)...", 0.0)
        n_crop = normals.astype(np.float32)
        sh, sw = n_crop.shape[:2]
        nz_c = np.maximum(n_crop[..., 2], 0.15)
        nx_sm = cv2.GaussianBlur(n_crop[..., 0], (0, 0), sigmaX=1.5)
        ny_sm = cv2.GaussianBlur(n_crop[..., 1], (0, 0), sigmaX=1.5)
        p_fc, q_fc = (np.clip(g, -5.0, 5.0) for g in depth_slopes(nx_sm, ny_sm, nz_c))
        depth_fc = frankot_chellappa(p_fc, q_fc)
        _p1, _p99 = np.percentile(depth_fc, 1), np.percentile(depth_fc, 99)
        _rng = _p99 - _p1
        np.clip(depth_fc, _p1 - _rng * 0.5, _p99 + _rng * 0.5, out=depth_fc)
        sigma_macro = max(4.0, hd_warp_pct * _math.sqrt(sh**2 + sw**2))
        if sigma_macro > 4.0:
            base = depth_fc - fast_large_sigma_blur(depth_fc, sigma_macro)
        else:
            base = depth_fc
        # Gaussian high-pass alone leaves the FC bowl/saddle bias in place
        # (see subtract_parabolic_baseline docstring) — explicitly flatten it.
        base = subtract_parabolic_baseline(base)
        scale = min(1.0, max_res / max(sh, sw))
        gH = max(4, int(sh * scale))
        gW = max(4, int(sw * scale))
        if progress_cb:
            progress_cb(f"HD: building {gH}x{gW} grid...", 0.25)
        if scale < 1.0:
            base = cv2.resize(base, (gW, gH), interpolation=cv2.INTER_AREA)
    elif use_hybrid and depth_normalised is not None:
        _global_crop = depth_normalised.astype(np.float32)
        _flat_crop = depth_vis_p.astype(np.float32)
        base = _global_crop * (1.0 - detail_mix) + _flat_crop * detail_mix
        gH, gW = base.shape
    else:
        d_hp = depth_vis_p.astype(np.float32)
        if depth_normalised is not None:
            d_full = depth_normalised.astype(np.float32)
            base = d_full * (1.0 - sb) + d_hp * sb
        else:
            base = d_hp.copy()
        h_r, w_r = base.shape
        scale = min(1.0, max_res / max(h_r, w_r))
        gH = max(4, int(h_r * scale))
        gW = max(4, int(w_r * scale))
        if progress_cb:
            progress_cb(f"Building {gH}x{gW} grid...", 0.2)
        if scale < 1.0:
            base = cv2.resize(base, (gW, gH), interpolation=cv2.INTER_AREA)
        if sb > 0.1:
            sigma_sb = max(3.0, sb * 0.4 * _math.sqrt(gH**2 + gW**2))
            base = base - fast_large_sigma_blur(base, sigma_sb)

    if amp != 1.0:
        sigma_hf = max(2.0, hf_cutoff * _math.sqrt(gH * gH + gW * gW))
        hf_layer = base - fast_large_sigma_blur(base, sigma_hf)
        np.add(base, hf_layer * (amp - 1.0), out=base)

    smooth = settings.get("smooth", 0.0)
    if smooth > 0:
        cv2.GaussianBlur(base, (0, 0), float(smooth), dst=base)

    # Belt and braces: the input was repaired on the way in, but the filters
    # above can still produce a non-finite value of their own.
    base = repair_nonfinite(base, "export depth grid")

    rng = float(base.max() - base.min())
    if not np.isfinite(rng) or rng < 1e-9:
        raise ValueError("Depth map appears flat. Process images first.")
    depth_grid = ((base - base.min()) / rng).astype(np.float32)
    if fz < 0:
        np.subtract(1.0, depth_grid, out=depth_grid)
    if mirror_x:
        depth_grid = np.fliplr(depth_grid)
    return depth_grid



def apply_export_filters(
    z_raw: np.ndarray,
    settings: dict,
    progress_cb: Callable[[str, float], None] | None = None,
) -> np.ndarray:
    """Apply HP, smooth, warp-removal, and micro-relief filters to a depth grid.

    Parameters
    ----------
    z_raw : np.ndarray
        2D decimated depth grid.
    settings : dict
        Keys: ``hp_sigma``, ``smooth``, ``warp_rem``, ``micro``.
    progress_cb : callable or None

    Returns
    -------
    np.ndarray
        Filtered depth grid.
    """
    hp_sigma = settings.get("hp_sigma", 0.0)
    smooth = settings.get("smooth", 0.0)
    warp_rem = settings.get("warp_rem", 0.0)
    micro = settings.get("micro", 0.0)
    gh, gw = z_raw.shape

    if hp_sigma > 0.01:
        sigma_px = float(hp_sigma) * max(gh, gw) * 0.4
        z_raw = z_raw - fast_large_sigma_blur(z_raw, sigma_px)
    if smooth > 0:
        cv2.GaussianBlur(z_raw, (0, 0), sigmaX=float(smooth), dst=z_raw)
    if warp_rem > 0:
        z_raw = z_raw - fast_large_sigma_blur(z_raw, max(gh, gw) * (0.05 + warp_rem * 0.45))
    if micro > 0:
        base_lf = fast_large_sigma_blur(z_raw, max(gh, gw) * 0.05)
        np.subtract(z_raw, base_lf, out=z_raw)
        np.multiply(z_raw, 1.0 + micro, out=z_raw)
        np.add(z_raw, base_lf, out=z_raw)
    return z_raw



def bake_normal_detail(
    z_grid: np.ndarray,
    normals: np.ndarray | None,
    depth_pct: float = 0.05,
    cutoff: float = 0.02,
) -> np.ndarray:
    """Displace a depth grid by the relief the normal map holds and it does not.

    The exported mesh is decimated; the normal map is not.  Everything finer
    than the vertex spacing therefore survives only as a texture, and any
    viewer that ignores normal maps - MeshLab's default shading, CloudCompare,
    a slicer - shows a surface far smoother than the capture.  This integrates
    the normals into a height field, keeps the high-frequency band, and adds it
    to the grid, so the vertices carry the detail themselves.

    Parameters
    ----------
    z_grid : np.ndarray
        Decimated depth grid, normalised to roughly [0, 1].  Not modified.
    normals : np.ndarray or None
        PS normal map (H, W, 3), cropped to the same region as *z_grid* but at
        any resolution - it is area-resampled to match.  ``None`` is a no-op.
    depth_pct : float
        Peak-to-peak height of the baked band as a fraction of the grid's own
        robust depth range: 0.05 spans 5% of the model depth.  <= 0 is a no-op.
    cutoff : float
        High-pass cutoff as a fraction of the grid diagonal.  The macro shape
        is already in *z_grid*; the point of the band is to add back only what
        decimation removed, so this stays small.

    Returns
    -------
    np.ndarray
        A new float32 grid of the same shape as *z_grid*.

    Notes
    -----
    Bake at the resolution the mesh will actually have.  Handing this a grid
    that gets decimated again afterwards aliases the band just added.
    """
    if normals is None or depth_pct <= 0.0:
        return z_grid
    gh, gw = z_grid.shape[:2]
    if gh < 4 or gw < 4:
        return z_grid

    n = repair_nonfinite(np.asarray(normals, dtype=np.float32), "normal map")
    if n.ndim != 3 or n.shape[2] < 3:
        raise ValueError("bake_normal_detail needs an (H, W, 3) normal map")
    if n.shape[:2] != (gh, gw):
        n = cv2.resize(n[..., :3], (gw, gh), interpolation=cv2.INTER_AREA)
    mag = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.divide(n, mag, out=np.zeros_like(n), where=mag > 1e-9)

    # Clamp nz and cap the slopes before integrating: a grazing normal at the
    # rim of the solve otherwise contributes an unbounded gradient, and the
    # integrator spreads that one sample across the whole patch.
    nz = np.maximum(n[..., 2], 0.15)
    n = np.stack([np.clip(n[..., 0] / nz, -5.0, 5.0),        # rescaled to nz = 1,
                  np.clip(n[..., 1] / nz, -5.0, 5.0),        # frame unchanged
                  np.ones_like(nz)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    # DCT (Neumann boundaries), not Frankot-Chellappa.  FC assumes the patch
    # tiles periodically, and an object photographed against a background does
    # not: the seam error lands as a rim of false relief a few pixels wide.
    # Measured on a synthetic dome with known incised lines, FC put more than
    # half of the band's entire peak-to-peak in the outermost four pixels of
    # the grid.  Simchony's DCT solver has no such seam.
    detail = simchony_depth_integration(n)

    # The integrator still returns the surface with its bowl on it.  A Gaussian
    # high-pass alone does not remove that bias (see
    # subtract_parabolic_baseline) - the HD export path flattens it in two
    # steps for the same reason, and so does this.
    sigma = max(2.0, float(cutoff) * math.sqrt(gh * gh + gw * gw))
    detail = detail - fast_large_sigma_blur(detail, sigma)
    detail = subtract_parabolic_baseline(detail)
    detail = repair_nonfinite(detail, "baked normal detail")

    # Percentiles, not min/max: one bad pixel in the normal solve would
    # otherwise set the scale for the entire surface and flatten the rest.
    d_lo, d_hi = np.percentile(detail, (1.0, 99.0))
    d_rng = float(d_hi - d_lo)
    if not np.isfinite(d_rng) or d_rng < 1e-9:
        _log.warning("normal map carries no high-frequency relief to bake")
        return z_grid
    z_lo, z_hi = np.percentile(z_grid, (1.0, 99.0))
    z_rng = float(z_hi - z_lo)
    if not np.isfinite(z_rng) or z_rng < 1e-9:
        z_rng = 1.0

    out = z_grid.astype(np.float32, copy=True)
    out += (detail * ((float(depth_pct) * z_rng) / d_rng)).astype(np.float32)
    return out



def build_grid_mesh(
    depth_grid: np.ndarray,
    settings: dict,
    progress_cb: Callable[[str, float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build vertex, face, UV, and colour arrays from a depth grid.

    Parameters
    ----------
    depth_grid : np.ndarray
        2D depth map (float32).
    settings : dict
        Keys: ``height``, ``fx``, ``fy``, ``fz``, ``step``, ``center_origin``,
        ``vertex_colors``, ``export_brightness``.
    progress_cb : callable or None

    Returns
    -------
    (vertices, faces, colors, uvs) : tuple of np.ndarray
        ``vertices`` (N, 3), ``faces`` (M, 3), ``colors`` (N, 3) uint8,
        ``uvs`` (N, 2) float32.
    """
    height = settings.get("height", 300.0)
    fx = settings.get("fx", 1)
    fy = settings.get("fy", -1)
    fz = settings.get("fz", 1)
    step = settings.get("step", 2)
    center_origin = settings.get("center_origin", True)
    use_vtx_color = settings.get("vertex_colors", True)
    exp_bright = settings.get("export_brightness", 1.0)
    albedo = settings.get("albedo", None)

    h, w = depth_grid.shape
    y_s, x_s = np.mgrid[0:h:step, 0:w:step]
    grid_h, grid_w = len(y_s[:, 0]), len(x_s[0, :])

    z_raw = depth_grid[0:h:step, 0:w:step].copy()
    z_raw = apply_export_filters(z_raw, settings, progress_cb)

    verts = np.column_stack((
        ((x_s - w / 2) * fx).flatten(),
        ((y_s - h / 2) * fy).flatten(),
        ((z_raw * fz * height) - (height / 2)).flatten(),
    )).astype(np.float32)

    if center_origin:
        verts[:, 0] -= np.mean(verts[:, 0])
        verts[:, 1] -= np.mean(verts[:, 1])
        verts[:, 2] -= np.mean(verts[:, 2])

    rr, cc = np.meshgrid(np.arange(grid_h - 1), np.arange(grid_w - 1), indexing="ij")
    v0 = rr * grid_w + cc
    faces = np.vstack((
        np.column_stack((v0.flatten(), (v0 + grid_w).flatten(), (v0 + 1).flatten())),
        np.column_stack(((v0 + 1).flatten(), (v0 + grid_w).flatten(), (v0 + grid_w + 1).flatten())),
    )).astype(np.int32)
    if (fx * fy * fz) < 0:
        faces = faces[:, [0, 2, 1]]
    # Emit outward-facing geometry here rather than leaving each consumer to
    # correct it.  The winding above alone produced a surface facing -Z - into
    # the object - so every *single-sided* export was 100% back-facing:
    # measured -0.817 mean face Nz on a dome whose true normals point up.
    # doubleSided:true hid it in the bundled web viewer, while Blender,
    # MeshLab and Sketchfab showed it inside out.  The watertight builder used
    # to compensate with a reverse of its own; that is now unnecessary.
    faces = faces[:, [0, 2, 1]]

    if use_vtx_color and albedo is not None:
        alb = albedo[0:h:step, 0:w:step].astype(np.float32)
        if exp_bright != 1.0:
            alb = np.power(np.clip(alb / 255.0, 0, 1), 1.0 / max(0.1, exp_bright)) * 255.0
        colors = np.clip(alb.reshape(-1, 3), 0, 255).astype(np.uint8)
    else:
        colors = np.full((len(verts), 3), 150, dtype=np.uint8)

    uv_cols = np.tile(np.linspace(0, 1, grid_w), grid_h)
    uv_rows = np.repeat(np.linspace(0, 1, grid_h), grid_w)
    uvs = np.column_stack((uv_cols, uv_rows)).astype(np.float32)

    return verts, faces, colors, uvs



def build_watertight_mesh_data(
    verts_top: np.ndarray,
    faces_top: np.ndarray,
    colors_top: np.ndarray,
    uvs_top: np.ndarray,
    settings: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Extend a single-sided mesh into a closed solid.

    Adds a flat bottom, side walls with dedicated vertices (to avoid UV colour
    bleed), and returns combined arrays.  Side-wall and bottom vertices receive
    black UVs so they sample a black patch at texture coordinate (0,0).

    **What "watertight" means here.** The result is a closed volume, but the
    seam vertices where the top surface meets the side walls are *duplicated
    rather than shared* - deliberately, because a shared vertex cannot carry
    both the top surface's UV and the wall's black UV, and the colour would
    bleed down the wall.  So the mesh is geometrically closed while being
    topologically open as written: a reader that merges coincident vertices
    (trimesh by default, and every slicer) sees a watertight solid; a reader
    that does not sees unshared edges along the seam.

    Measured on a 48x64 grid: 440 exactly-coincident vertex pairs, watertight
    after merging, positive volume.  This is normal practice for a textured
    mesh and is not a defect - but the distinction matters if a repository
    validates the file as written, so ``tests/test_mesh.py`` pins both halves of it.

    Parameters
    ----------
    verts_top : np.ndarray
        (N, 3) top-surface vertices.
    faces_top : np.ndarray
        (M, 3) top-surface triangle indices.
    colors_top : np.ndarray
        (N, 3) top-surface vertex colours.
    uvs_top : np.ndarray
        (N, 2) top-surface UVs.
    settings : dict
        Keys: ``fx``, ``fy``, ``fz``.

    Returns
    -------
    (all_verts, all_faces, all_colors, all_uvs, n_top_verts) : tuple
    """
    fx = settings.get("fx", 1)
    fy = settings.get("fy", -1)
    fz = settings.get("fz", 1)
    flip = (fx * fy * fz) < 0
    N = len(verts_top)

    z_min = float(verts_top[:, 2].min())
    z_range = float(verts_top[:, 2].max() - z_min)
    verts_bottom = verts_top.copy()
    verts_bottom[:, 2] = z_min - (z_range * 0.2 + 2.0)

    # The solid used to be built with its walls facing out and both surfaces
    # facing in, so the signed volume read 10,812 against a true 38,036 - a
    # 3.5x error in anything computed from the mesh, and inconsistent normals
    # for anything that renders or slices it.
    # Measured per face group against the mesh centroid, at both fy=-1 and
    # fy=+1: the side walls come out 100% outward either way, while the top
    # and bottom come out 0% - so the surfaces need reversing unconditionally,
    # not on the mirror flag.  build_grid_mesh emits the grid wound for the
    # opposite convention and compensates for its own mirroring internally,
    # which is why `flip` is the wrong test here even though it is the right
    # one for the walls below.
    # build_grid_mesh now hands us outward-facing surfaces, so the reverse
    # that used to live here is gone.  The bottom is still the mirror of the
    # top, which is what makes it face down.
    faces_bottom = faces_top[:, [0, 2, 1]] + N

    gw = int(_math.isqrt(N))  # approximate — caller must ensure square-ish grid
    # Actually compute from UV pattern
    _u = uvs_top[:, 0]
    grid_w = int(round(1.0 / np.mean(np.diff(np.unique(_u)))) + 1) if len(np.unique(_u)) > 1 else 1

    top_edge = np.arange(grid_w)
    right_edge = np.arange(grid_w - 1, N, grid_w)
    bottom_edge = np.arange(N - 1, N - grid_w - 1, -1)
    left_edge = np.arange(N - grid_w, -1, -grid_w)
    perimeter = np.concatenate([top_edge[:-1], right_edge[:-1], bottom_edge[:-1], left_edge[:-1]])
    P = len(perimeter)

    verts_side_top = verts_top[perimeter].copy()
    verts_side_bot = verts_bottom[perimeter].copy()
    base_st = 2 * N
    base_sb = 2 * N + P

    side_faces = []
    for i in range(P):
        j = (i + 1) % P
        st0, st1 = base_st + i, base_st + j
        sb0, sb1 = base_sb + i, base_sb + j
        if flip:
            side_faces.extend([[st0, st1, sb0], [st1, sb1, sb0]])
        else:
            side_faces.extend([[st0, sb0, st1], [st1, sb0, sb1]])
    side_arr = np.array(side_faces, dtype=np.int32)

    all_verts = np.vstack([verts_top, verts_bottom, verts_side_top, verts_side_bot])
    all_faces = np.vstack([faces_top, faces_bottom, side_arr])

    # Mirroring Z turns the solid inside out: the bottom slab, placed below the
    # surface in output coordinates, ends up above it.  Winding stays
    # self-consistent either way, so nothing looks wrong - but the signed
    # volume goes negative and every normal points into the object.  This
    # belongs here rather than in build_grid_mesh: it is a property of closing
    # the surface into a solid, and applying it to the open grid as well broke
    # four of the eight sign combinations.
    if fz < 0:
        all_faces = all_faces[:, [0, 2, 1]]

    colors_black = np.zeros((2 * P, 3), dtype=np.uint8)
    all_colors = np.vstack([colors_top, np.full_like(colors_top, 30), colors_black])

    _bc = 0.002
    _cc = 0.5
    uv_bottom = np.full((N, 2), _bc, dtype=np.float32)
    uv_side_top = np.full((P, 2), _bc, dtype=np.float32)
    uv_side_bot = np.full((P, 2), _bc, dtype=np.float32)
    all_uvs = np.vstack([uvs_top, uv_bottom, uv_side_top, uv_side_bot])

    return all_verts, all_faces, all_colors, all_uvs, N



def compute_hf_depth_from_normals(
    normals: np.ndarray,
    hf_cutoff: float = 0.35,
    max_res: int = MESH_MAX_RES,
    progress_cb: Callable[[str, float], None] | None = None,
    denoise: bool = True,
) -> np.ndarray:
    """Frankot–Chellappa integration + aggressive HP filter for HF-only depth.

    Returns a normalised [0,1] depth grid containing only high-frequency
    surface detail (carved lines, brush strokes, tool marks).

    Parameters
    ----------
    normals : np.ndarray
        PS normal map (H, W, 3), float32.
    hf_cutoff : float
        Fraction of diagonal used as the HP Gaussian sigma (default 0.35).
    max_res : int
        Long-axis cap for the output grid (default 1200).
    progress_cb : callable or None
    denoise : bool
        Apply a mild edge-preserving bilateral pre-filter to the normals
        before FFT integration (default True). A high-pass filter can't
        distinguish genuine fine tool-mark detail from per-pixel
        photometric-stereo noise — both are "high frequency" by
        definition — so any noise in the source normal map is preserved/
        amplified right alongside real detail unless removed beforehand.
        Bilateral (not Gaussian) specifically because it preserves strong
        local gradients — i.e. real carved edges — while smoothing the
        low-amplitude jitter in flat regions.

    Returns
    -------
    np.ndarray
        Normalised HF depth grid (float32, [0, 1]).
    """
    n_crop = normals.astype(np.float32)
    sh, sw = n_crop.shape[:2]

    # Downsample BEFORE FFT — doing FFT at full res (e.g. 11000×8000) then
    # downsampling to max_res=1200 is ~85× slower than downsample-first.
    # HF detail lost by downsampling would be averaged away in the final
    # 1200px output anyway, so no information is lost.
    scale = min(1.0, max_res / max(sh, sw))
    if scale < 1.0:
        gW = max(4, int(sw * scale))
        gH = max(4, int(sh * scale))
        n_crop = cv2.resize(n_crop, (gW, gH), interpolation=cv2.INTER_AREA)
        sh, sw = gH, gW
        if progress_cb:
            progress_cb("Resampled normals to {}×{}…".format(sw, sh), 0.05)

    if denoise:
        if progress_cb:
            progress_cb("Denoising normals (edge-preserving)…", 0.1)
        n_crop = cv2.bilateralFilter(n_crop, d=0, sigmaColor=0.15, sigmaSpace=2.5)
        _nlen = np.linalg.norm(n_crop, axis=-1, keepdims=True)
        n_crop = n_crop / np.maximum(_nlen, 1e-6)

    if progress_cb:
        progress_cb("Computing Frankot–Chellappa FFT…", 0.15)

    nx_c = n_crop[..., 0]
    ny_c = n_crop[..., 1]
    nz_c = np.maximum(n_crop[..., 2], 0.001)
    p_fc, q_fc = depth_slopes(nx_c, ny_c, nz_c)
    depth_fc = frankot_chellappa(p_fc, q_fc)

    if progress_cb:
        progress_cb("High-pass filtering…", 0.55)

    sigma_macro = max(5.0, hf_cutoff * _math.sqrt(sh * sh + sw * sw))
    depth_hf = depth_fc - fast_large_sigma_blur(depth_fc, sigma_macro)
    # Gaussian high-pass alone leaves the FC bowl/saddle bias in place (see
    # subtract_parabolic_baseline docstring) — explicitly flatten it.
    depth_hf = subtract_parabolic_baseline(depth_hf)

    if progress_cb:
        progress_cb("Normalising depth grid…", 0.85)

    rng = float(depth_hf.max() - depth_hf.min())
    if rng > 1e-9:
        depth_grid = ((depth_hf - depth_hf.min()) / rng).astype(np.float32)
    else:
        depth_grid = np.zeros_like(depth_hf)

    return depth_grid



def compute_vertex_normals_from_ps(
    normals: np.ndarray,
    settings: dict,
) -> np.ndarray:
    """Resample PS normals onto the vertex grid and apply axis flips.

    Parameters
    ----------
    normals : np.ndarray
        Full-resolution PS normal map (H, W, 3).
    settings : dict
        Keys: ``crop`` (x1,y1,x2,y2), ``step``, ``fx``, ``fy``, ``fz``.

    Returns
    -------
    np.ndarray
        Vertex normals (N, 3), float32, unit-length.
    """
    x1, y1, x2, y2 = settings.get("crop", (0, 0, normals.shape[1], normals.shape[0]))
    step = settings.get("step", 2)
    fx = settings.get("fx", 1)
    fy = settings.get("fy", -1)
    fz = settings.get("fz", 1)
    h = y2 - y1
    w = x2 - x1

    n_grid = normals[y1:y2, x1:x2][0:h:step, 0:w:step].astype(np.float32)
    n_grid[:, :, 0] *= fx
    n_grid[:, :, 1] *= fy
    n_grid[:, :, 2] *= fz
    if np.mean(n_grid[:, :, 2]) < 0:
        np.negative(n_grid, out=n_grid)
    nlen = np.linalg.norm(n_grid, axis=-1, keepdims=True)
    nlen = np.where(nlen < 1e-9, 1.0, nlen)
    return (n_grid / nlen).reshape(-1, 3).astype(np.float32)



def pack_binary_stl(
    all_tris: np.ndarray,
    filepath: str,
) -> tuple[int, float]:
    """Write a binary STL file from a triangle soup.

    Parameters
    ----------
    all_tris : np.ndarray
        (N, 3, 3) float32 triangle vertices.
    filepath : str
        Output path.

    Returns
    -------
    (n_tri, size_mb) : tuple[int, float]
    """
    n_tri = len(all_tris)
    e1 = all_tris[:, 1] - all_tris[:, 0]
    e2 = all_tris[:, 2] - all_tris[:, 0]
    norms = np.cross(e1, e2).astype(np.float32)
    nlen = np.linalg.norm(norms, axis=1, keepdims=True)
    norms = np.divide(norms, nlen, out=np.zeros_like(norms), where=nlen > 1e-12)

    dt = np.dtype([("n", np.float32, 3), ("v0", np.float32, 3),
                   ("v1", np.float32, 3), ("v2", np.float32, 3),
                   ("attr", np.uint16)])
    buf = np.zeros(n_tri, dtype=dt)
    buf["n"] = norms
    buf["v0"] = all_tris[:, 0]
    buf["v1"] = all_tris[:, 1]
    buf["v2"] = all_tris[:, 2]

    with open(filepath, "wb") as f:
        # Binary STL requires an EXACTLY 80-byte header — every byte after
        # it (triangle count, then each 50-byte triangle record) is at a
        # fixed offset with no delimiters, so a short header misaligns the
        # whole rest of the file for any strict parser. This used to write
        # only 77 bytes (25-byte string + 52 zero bytes), silently
        # corrupting every export.
        _header = b"Helio 3D Print Export"
        f.write(_header + b"\x00" * (80 - len(_header)))
        f.write(_struct.pack("<I", n_tri))
        f.write(buf.tobytes())

    fsize_mb = os.path.getsize(filepath) / 1e6
    return n_tri, fsize_mb
