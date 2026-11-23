"""Image exports, GUI-free (EXP-001, second slice).

The classic Visuals and Science exports' writing, moved here so both apps
write the same files: every view as a PNG, the normal-map set, DeepZoom
tiles, the PCA, HSH and elevation maps, the 16-bit PBR suite, specular /
diffuse separation, the uncertainty map and a render with a scale bar.

Renders come in through a ``draw(engine) -> RGB uint8`` callable, so each
app draws with its own canvas while the files, names and maths are shared.
The classic methods keep their dialogs, progress windows and messages.
"""
from __future__ import annotations

import os
import shutil
from typing import Callable

import cv2
import numpy as np

from helio.application.exports import NothingToExport

_TIFF = [cv2.IMWRITE_TIFF_COMPRESSION, 1]


def rot(arr, k: int):
    """The canvas rotation, for display-only exports (never scientific ones)."""
    k = int(k) % 4
    return arr if k == 0 else np.rot90(arr, k=k)


def engine_file(eng: str) -> str:
    """The classic Snapshot All Views file name for an engine."""
    return (eng.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
            .replace(".", ""))


def load_stack(paths, w: int, h: int, *, crop=None, base=None, tone: str = "stored"):
    """The frames as an (N, h, w) 0–255 float32 grey stack at the solve's
    size (cropped first when the solve was a region); 'srgb' decodes to
    linear light as the solve does.  None if any frame cannot be read.

    *crop* is in the solve's base pixels, *base* (w, h): a solve at a reduced
    working size has a base smaller than the photographs, so the crop is
    scaled to each photograph as read."""
    from helio.core.frame_read import read_frame
    from helio.io.raw_loader import is_raw
    out = []
    for p in paths:
        g = read_frame(p, cv2.IMREAD_GRAYSCALE, linear=True, sharp=True)
        if g is None:
            return None
        if crop:
            kx, ky = ((g.shape[1] / base[0], g.shape[0] / base[1]) if base and base[0] and base[1]
                      else (1.0, 1.0))
            x0, y0, x1, y1 = crop
            g = g[int(round(y0 * ky)):int(round(y1 * ky)), int(round(x0 * kx)):int(round(x1 * kx))]
        if g.shape != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        if tone == "srgb" and not is_raw(p):          # RAW is linear already
            from helio.core.radiometry import srgb_to_linear
            g = srgb_to_linear(g, 255.0).astype(np.float32) * 255.0
        out.append(g)
    return np.stack(out, axis=0) if out else None


# ── every view ───────────────────────────────────────────────────────────

def all_views(draw: Callable, engines, out_dir: str, *, base=None, rot_k: int = 0,
              progress: Callable | None = None) -> tuple[list[str], int]:
    """Snapshot All Views: one PNG per engine, plus the albedo, normals and
    depth heatmap when *base* = (albedo, normals, depth_vis) is given.
    An engine that fails is skipped.  Returns (files, engines written)."""
    os.makedirs(out_dir, exist_ok=True)
    files, done = [], 0
    engines = list(engines)
    for i, eng in enumerate(engines):
        if progress:
            progress(f"Rendering {i + 1}/{len(engines)}: {eng}", (i + 1) / max(1, len(engines)))
        try:
            img = rot(draw(eng), rot_k)
            p = os.path.join(out_dir, f"{engine_file(eng)}.png")
            cv2.imwrite(p, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
            files.append(p)
            done += 1
        except Exception as ex:                     # the classic carries on
            print(f"[AllViews] {eng}: {ex}")
    if base is not None:
        alb, nrm, dep = base
        for name, arr in (("0_PureColour.png", cv2.cvtColor(alb, cv2.COLOR_RGB2BGR)),
                          ("0_Normals.png", cv2.cvtColor((((nrm + 1) / 2) * 255).astype(np.uint8),
                                                         cv2.COLOR_RGB2BGR)),
                          ("0_DepthHeatmap.png", cv2.applyColorMap(
                              (np.clip(dep, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO))):
            p = os.path.join(out_dir, name)
            cv2.imwrite(p, arr)
            files.append(p)
    return files, done


# ── the normal-map set (Combined Renders) ────────────────────────────────

def hillshade(depth, az_deg: float = 305.0, alt_deg: float = 35.0, z: float = 0.3):
    """Analytic hillshade of a height field (rows down, height up) lit from
    compass azimuth *az_deg* (0 = the top of the image, clockwise) at
    *alt_deg*, 1–99 % stretched with gamma 0.88 as the classic map.

    HELIO 2 mixed the GIS azimuth conversion (360 − az + 90) into a vector
    formula and lit the map from 215°, not the 305° in its file name
    (KD-029)."""
    d = np.asarray(depth, np.float32) * float(z)
    gy, gx = np.gradient(d)
    sx, sy = gx, -gy                                  # slopes east and north
    a, e = np.radians(az_deg), np.radians(alt_deg)
    lx, ly, lz = np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), np.sin(e)
    hs = np.clip((-sx * lx - sy * ly + lz) / np.sqrt(1.0 + sx * sx + sy * sy), 0, 1)
    p1, p99 = float(np.percentile(hs, 1)), float(np.percentile(hs, 99))
    hs = np.clip((hs - p1) / (p99 - p1 + 1e-9), 0, 1)
    return np.power(hs, 0.88).astype(np.float32)


def combined_renders(D, out_dir: str, obj_id: str = "export", *, crop=None, rot_k: int = 0,
                     z: float = 0.3, progress: Callable | None = None) -> list[str]:
    """Normal map, hillshade (Az 305° Alt 35°), albedo × hillshade hybrid,
    albedo and the high-frequency normal detail map, as the classic
    Combined Renders."""
    if getattr(D, "normals_full", None) is None:
        raise NothingToExport("Process scans first.")
    upd = progress or (lambda *_a: None)
    x1, y1, x2, y2 = crop if crop else (0, 0, D.w, D.h)
    H, W = y2 - y1, x2 - x1
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in (obj_id or "export"))
    os.makedirs(out_dir, exist_ok=True)
    out = []

    def save(nm, arr):
        p = os.path.join(out_dir, nm)
        cv2.imwrite(p, arr)
        out.append(p)

    upd("Saving normal map…", 0.1)
    nm = np.round((D.normals_full[y1:y2, x1:x2] * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    save(f"{safe}_normal_map.png", cv2.cvtColor(rot(nm, rot_k), cv2.COLOR_RGB2BGR))
    upd("Rendering hillshade…", 0.25)
    hs = hillshade(D.depth_vis_p[y1:y2, x1:x2], 305.0, 35.0, z)
    save(f"{safe}_hillshade_Az305_Alt35.png", rot((hs * 255).clip(0, 255).astype(np.uint8), rot_k))
    upd("Building albedo/hillshade hybrid…", 0.45)
    alb = D.albedo_rgb_full[y1:y2, x1:x2].astype(np.float32) / 255.0
    hs3 = np.stack([hs, hs, hs], axis=-1)
    hybrid = np.clip(alb * hs3 * 1.4, 0, 1)
    screen = np.clip(1.0 - (1.0 - alb) * (1.0 - hs3 * 0.35), 0, 1)
    h8 = ((hybrid * 0.65 + screen * 0.35) * 255).clip(0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(h8, (0, 0), 1.2)
    h8 = np.clip(h8.astype(np.float32) * 1.35 - blur.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
    save(f"{safe}_hybrid_albedo_hillshade.png", cv2.cvtColor(rot(h8, rot_k), cv2.COLOR_RGB2BGR))
    upd("Saving albedo map…", 0.60)
    save(f"{safe}_albedo.png", cv2.cvtColor(rot(D.albedo_rgb_full[y1:y2, x1:x2], rot_k),
                                            cv2.COLOR_RGB2BGR))
    upd("Computing HF normal deviation map…", 0.75)
    n = D.normals_full[y1:y2, x1:x2].astype(np.float32)
    lf = cv2.GaussianBlur(n, (0, 0), max(3.0, 0.012 * float(np.hypot(H, W))))
    hf = n - lf
    mag = np.sqrt(hf[..., 0] ** 2 + hf[..., 1] ** 2)
    p1, p99 = float(np.percentile(mag, 1)), float(np.percentile(mag, 99))
    u8 = (np.clip((mag - p1) / (p99 - p1 + 1e-9), 0, 1) * 255).clip(0, 255).astype(np.uint8)
    save(f"{safe}_hf_normal_detail.png",
         rot(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8), rot_k))
    upd("Done.", 1.0)
    return out


# ── single maps ──────────────────────────────────────────────────────────

def pca_tiff(pca_full, path: str) -> str:
    if pca_full is None:
        raise NothingToExport("No PCA map — process images first.")
    cv2.imwrite(path, cv2.cvtColor(pca_full, cv2.COLOR_RGB2BGR), _TIFF)
    return path


def elevation_heatmap(depth_vis, path: str) -> str:
    if depth_vis is None:
        raise NothingToExport("Process images first.")
    cv2.imwrite(path, cv2.applyColorMap((np.clip(depth_vis, 0, 1) * 255).astype(np.uint8),
                                        cv2.COLORMAP_TURBO))
    return path


def hsh_coefficients(hsh_coeffs, w: int, h: int, out_dir: str, *, f32: bool = True,
                     u16: bool = True, vis: bool = False) -> list[str]:
    """The nine HSH coefficient maps: float32, 16-bit range-mapped, and/or colour."""
    if hsh_coeffs is None:
        raise NothingToExport("No HSH coefficients — process images first.")
    out = []
    for i in range(9):
        c = hsh_coeffs[i].reshape(h, w).astype(np.float32)
        if f32:
            p = os.path.join(out_dir, f"hsh_coeff_{i}_f32.tif")
            cv2.imwrite(p, c, _TIFF)
            out.append(p)
        if u16:
            mn, mx = float(c.min()), float(c.max())
            p = os.path.join(out_dir, f"hsh_coeff_{i}_u16.tif")
            cv2.imwrite(p, ((c - mn) / max(mx - mn, 1e-9) * 65535).astype(np.uint16), _TIFF)
            out.append(p)
        if vis:
            u8 = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            p = os.path.join(out_dir, f"hsh_coeff_{i}_vis.tif")
            cv2.imwrite(p, cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), _TIFF)
            out.append(p)
    return out


def uncertainty_map(resid, path: str, *, crop=None) -> tuple[str, float, float]:
    """The per-pixel fit residual as a tagged GeoTIFF (0–1).  Returns (path,
    mean, 99th percentile)."""
    from helio.io.exporters import write_geotiff
    if resid is None:
        raise NothingToExport("No residual map. It is produced by a calibrated solve — an "
                              "uncalibrated (PCA) solve has no light matrix to fit against.")
    r = np.asarray(resid, dtype=np.float32)
    if crop:
        x1, y1, x2, y2 = crop
        r = r[y1:y2, x1:x2]
    if r.max() > 1.5:
        r = r / 255.0
    write_geotiff(path, r, tags={
        "HELIO_LAYER": "uncertainty",
        "HELIO_UNITS": "0-1 normalised RMS residual",
        "HELIO_UNCERTAINTY": "RMS residual of the per-pixel Lambertian fit; high = the "
                             "recovered normal is least believable"})
    return path, float(r.mean()), float(np.percentile(r, 99))


def scale_bar_render(rgb, mm_per_px: float, path: str) -> str:
    """A render with a calibrated scale bar burned in."""
    from helio.io.sidecars import draw_scale_bar
    if not mm_per_px or mm_per_px <= 0:
        raise NothingToExport("No measured scale. Set one in Measure first.")
    cv2.imwrite(path, cv2.cvtColor(draw_scale_bar(np.asarray(rgb), float(mm_per_px)),
                                   cv2.COLOR_RGB2BGR))
    return path


# ── specular / diffuse and the PBR suite ─────────────────────────────────

SDS_README = (
    "Helio — Specular/Diffuse Separation\n"
    + "=" * 52 + "\n\n"
    "METHOD\n"
    "Per-pixel Lambertian/Blinn-Phong decomposition.\n"
    "For each pixel: rho = ΣI_i·(N·L_i) / Σ(N·L_i)²  (least-squares diffuse albedo)\n"
    "Specular residual = max over lights of (I_i − ρ·N·L_i)\n"
    "Reference: Mallick et al. 2005; Tan et al. 2003 ICCV\n\n"
    "OUTPUT FILES\n"
    "─────────────────────────────────────────────────────\n"
    "sds_diffuse_albedo.png / _16bit.tif\n"
    "  Pure Lambertian reflectance. Specular highlights removed.\n"
    "  Colour hue preserved via per-channel ratio to mean albedo.\n\n"
    "sds_specular_component.png / _32f.tif\n"
    "  Specular residual: what cannot be explained by N·L alone.\n"
    "  Bright = strong specular return (gloss, glare, burr ridge).\n"
    "  Use 32f TIFF for photometric quantification in GIS/Fiji.\n\n"
    "sds_gloss_index.png / _16bit.tif\n"
    "  Gloss fraction = specular / (diffuse + specular).\n"
    "  0 = fully matte  |  1 = fully specular / mirror-like.\n\n"
    "sds_highlight_mask.png\n"
    "  Binary mask: pixels where gloss > mean + 2σ.\n"
    "  Use to exclude specular hot-spots from measurements.\n\n"
    "USE CASES\n"
    "─────────────────────────────────────────────────────\n"
    "  Corroded copper plates: diffuse map reveals\n"
    "    engraved detail hidden under specular surface glare.\n"
    "  Glazed ceramics: gloss index maps glaze distribution.\n"
    "  Metallic coins: diffuse removes die polish from field.\n"
    "  Varnished paintings: specular tracks varnish condition.\n"
    "  Drypoint prints: gloss index quantifies surviving burr\n"
    "    (raised metal ridge flanking the incised groove that\n"
    "     holds ink and gives drypoint its characteristic quality).\n")


def _write_sds(sds: dict, out_dir: str) -> list[str]:
    files = []
    for name, arr, ext_args in (
            ("sds_diffuse_albedo_16bit.tif", cv2.cvtColor(sds["diffuse_albedo_16"], cv2.COLOR_RGB2BGR), _TIFF),
            ("sds_diffuse_albedo.png", cv2.cvtColor(sds["diffuse_albedo_8"], cv2.COLOR_RGB2BGR), None),
            ("sds_specular_component_32f.tif", sds["specular_32f"], _TIFF),
            ("sds_specular_component.png", sds["specular_8"], None),
            ("sds_gloss_index_16bit.tif", sds["gloss_index_16"], _TIFF),
            ("sds_gloss_index.png", sds["gloss_index_8"], None),
            ("sds_highlight_mask.png", sds["highlight_mask_8"], None)):
        p = os.path.join(out_dir, name)
        cv2.imwrite(p, arr, ext_args) if ext_args else cv2.imwrite(p, arr)
        files.append(p)
    return files


def specular_diffuse(D, out_dir: str, *, lights=None, stack=None, crop=None,
                     progress: Callable | None = None) -> list[str]:
    """Specular/Diffuse Separation (Mallick et al. 2005): diffuse albedo,
    specular residual, gloss index and highlight mask, with a README.

    *stack* is the frames as (N, h, w) 0–255 (``load_stack``) matching
    *lights*; without them the classic variance proxy is used."""
    from helio.core.processing import compute_sds_maps
    if getattr(D, "normals_full", None) is None:
        raise NothingToExport("Process scans first.")
    upd = progress or (lambda *_a: None)
    x1, y1, x2, y2 = crop if crop else (0, 0, D.w, D.h)
    os.makedirs(out_dir, exist_ok=True)
    upd("Computing Lambertian fit…", 0.15)
    if stack is not None and lights is not None and len(stack) != len(lights):
        stack = None
    sds = compute_sds_maps(D.normals_full[y1:y2, x1:x2], D.albedo_rgb_full[y1:y2, x1:x2],
                           grey_stack=stack, l_mat=lights if stack is not None else None,
                           spec_var=getattr(D, "spec_var", None),
                           median_img=getattr(D, "median_img", None), y1=y1, x1=x1, y2=y2, x2=x2)
    upd("Saving outputs…", 0.75)
    files = _write_sds(sds, out_dir)
    p = os.path.join(out_dir, "SDS_README.txt")
    with open(p, "w") as f:
        f.write(SDS_README)
    upd("Done.", 1.0)
    return files + [p]


PBR_MAPS = (("albedo", "Albedo / Base Colour 16-bit TIFF"),
            ("normal_map", "Normal Map (primary) 16-bit TIFF"),
            ("roughness", "Roughness 16-bit TIFF"),
            ("metallic", "Metallic PNG (constant zero — dielectric)"),
            ("detail_normals", "Detail Normals PNG"),
            ("ao", "Ambient Occlusion PNG"),
            ("cavity", "Cavity Map PNG"),
            ("curvature", "Curvature RGB PNG"),
            ("displacement", "Displacement 16-bit PNG (FFT + high-pass)"),
            ("sds", "Specular/Diffuse Separation maps"))


def pbr_suite(D, out_dir: str, selected: dict | None = None, *, step: int = 1, lights=None,
              stack=None, progress: Callable | None = None) -> list[str]:
    """The 16-bit PBR suite: the maps in *selected* (all by default)."""
    from helio.core.processing import (compute_ambient_occlusion, compute_cavity_map,
                                       compute_curvature_rgb, compute_depth_displacement,
                                       compute_detail_normals, compute_sds_maps)
    if getattr(D, "normals_full", None) is None:
        raise NothingToExport("Process scans first.")
    sel = {k: True for k, _ in PBR_MAPS} if selected is None else dict(selected)
    if not any(sel.values()):
        raise NothingToExport("Tick at least one map to export.")
    os.makedirs(out_dir, exist_ok=True)
    n, alb, d = D.normals_full, D.albedo_rgb_full, getattr(D, "depth_vis_p", None)
    rough = getattr(D, "roughness_full", None)
    steps = [k for k, _ in PBR_MAPS if sel.get(k)
             and not (k == "roughness" and rough is None) and not (k == "displacement" and d is None)]
    out = []

    def w(name, arr, args=None):
        p = os.path.join(out_dir, name)
        cv2.imwrite(p, arr, args) if args else cv2.imwrite(p, arr)
        out.append(p)
    for i, k in enumerate(steps):
        if progress:
            progress(dict(PBR_MAPS)[k], i / max(1, len(steps)))
        if k == "albedo":
            w("albedo_16bit.tif", cv2.cvtColor(alb.astype(np.uint16) * 257, cv2.COLOR_RGB2BGR), _TIFF)
        elif k == "normal_map":
            n16 = (((n + 1.0) / 2.0) * 65535).clip(0, 65535).astype(np.uint16)
            w("normal_map_16bit.tif", cv2.cvtColor(n16, cv2.COLOR_RGB2BGR), _TIFF)
        elif k == "roughness":
            w("roughness_16bit.tif", rough.astype(np.uint16) * 257, _TIFF)
        elif k == "metallic":
            w("metallic.png", np.zeros(n.shape[:2], np.uint8))
        elif k == "detail_normals":
            dn = compute_detail_normals(n, d, step)
            w("detail_normals.png", cv2.cvtColor((((dn + 1.0) / 2.0) * 255).clip(0, 255)
                                                 .astype(np.uint8), cv2.COLOR_RGB2BGR))
        elif k == "ao":
            w("ambient_occlusion.png", (compute_ambient_occlusion(n) * 255).astype(np.uint8))
        elif k == "cavity":
            w("cavity.png", (compute_cavity_map(n) * 255).astype(np.uint8))
        elif k == "curvature":
            w("curvature_rgb.png", cv2.cvtColor(compute_curvature_rgb(n), cv2.COLOR_RGB2BGR))
        elif k == "displacement":
            disp = compute_depth_displacement(d)
            w("displacement.png", cv2.normalize(disp, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16))
        elif k == "sds":
            st = stack if (stack is not None and lights is not None
                           and len(stack) == len(lights)) else None
            sh, sw = n.shape[:2]
            sds = compute_sds_maps(n, alb, grey_stack=st, l_mat=lights if st is not None else None,
                                   spec_var=getattr(D, "spec_var", None),
                                   median_img=getattr(D, "median_img", None), y1=0, x1=0, y2=sh, x2=sw)
            out += _write_sds(sds, out_dir)
    return out


# ── DeepZoom ─────────────────────────────────────────────────────────────

def deepzoom_data(D, max_edge: int | None = None):
    """A copy of *D* at the DeepZoom resolution cap (every layer shares it,
    so layers line up when switched), PTM/HSH coefficients resized per
    channel.  *D* itself is left alone."""
    import copy
    from helio.config import DEEPZOOM_MAX_EDGE
    if getattr(D, "ptm_deferred", False):
        D.ensure_ptm_hsh()
    cap = max_edge or DEEPZOOM_MAX_EDGE
    h, w = D.h, D.w
    s = min(1.0, cap / max(h, w))
    if s >= 1.0:
        return D
    zh, zw = max(1, int(h * s)), max(1, int(w * s))
    Z = copy.copy(D)

    def rs(a):
        return None if a is None else cv2.resize(a, (zw, zh), interpolation=cv2.INTER_AREA)

    def rc(c):
        if c is None or c.ndim != 2 or c.shape[1] != h * w:
            return c
        o = np.empty((c.shape[0], zh * zw), np.float32)
        for i in range(c.shape[0]):
            o[i] = cv2.resize(c[i].reshape(h, w).astype(np.float32), (zw, zh),
                              interpolation=cv2.INTER_AREA).ravel()
        return o
    for k in ("normals_full", "albedo_rgb_full", "roughness_full", "depth_vis_p", "spec_var",
              "pca_full"):
        setattr(Z, k, rs(getattr(D, k, None)))
    Z.ptm_coeffs = rc(getattr(D, "ptm_coeffs", None))
    Z.hsh_coeffs = rc(getattr(D, "hsh_coeffs", None))
    Z.h, Z.w = zh, zw
    return Z


def deepzoom_layers(Z, draw: Callable) -> list[tuple[str, np.ndarray, str]]:
    """The classic DeepZoom layers from data at the cap: colour, normals,
    curvature, PBR, roughness, then specular and the PCA components when
    there are any.  *draw(engine)* renders from *Z*."""
    r = Z.roughness_full
    r8 = r.astype(np.uint8) if r.dtype == np.uint8 else (np.clip(r, 0, 1) * 255).astype(np.uint8)
    layers = [("albedo", Z.albedo_rgb_full, "Pure Colour"),
              ("normals", (((Z.normals_full + 1.0) / 2.0) * 255).astype(np.uint8), "Normal Map"),
              ("curvature", draw("Surface Curvature (Scratches)"), "Surface Curvature"),
              ("pbr", draw("Standard PBR"), "Standard PBR"),
              ("roughness", cv2.cvtColor(r8, cv2.COLOR_GRAY2RGB), "Roughness")]
    if getattr(Z, "spec_var", None) is not None:
        layers.append(("specular_view", draw("Specular View"), "Specular View"))
    if getattr(Z, "pca_full", None) is not None:
        for i in range(3):
            layers.append((f"pca{i + 1}", Z.pca_full[..., i:i + 1].repeat(3, axis=2), f"PCA {i + 1}"))
    return layers


def deepzoom_html(layers) -> str:
    """The DeepZoom viewer page; on a phone its layer buttons are one row
    that scrolls sideways (OpenSeadragon pans and pinches by itself)."""
    from helio.application.phone_web import phone_ready
    return phone_ready(_deepzoom_page(layers), css=(
        "#toolbar{flex-wrap:nowrap!important;overflow-x:auto;-webkit-overflow-scrolling:touch}"
        "#toolbar button{flex:0 0 auto}#toolbar #lbl{display:none}#osd{top:56px!important}"))


def _deepzoom_page(layers) -> str:
    btn = "".join('<button onclick="switchLayer(' + "'" + sl + "'" + ')">' + lbl + '</button>'
                  for sl, _, lbl in layers)
    labels = ",".join(f'"{sl}":"{lbl}"' for sl, _, lbl in layers)
    return f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Helio | DeepZoom Viewer</title>
<style>
  body{{margin:0;background:#111;font-family:sans-serif;}}
  #toolbar{{position:fixed;top:0;left:0;right:0;z-index:20;background:rgba(0,0,0,0.82);padding:6px;display:flex;flex-wrap:wrap;gap:4px;}}
  #toolbar button{{padding:6px 12px;background:#2c3e50;color:#eee;border:none;border-radius:4px;cursor:pointer;font-size:13px;}}
  #toolbar button:hover,#toolbar button.active{{background:#2980b9;color:#fff;}}
  #toolbar #lbl{{color:#aaa;font-size:13px;align-self:center;margin-left:8px;}}
  #osd{{position:fixed;top:44px;left:0;right:0;bottom:0;}}
</style>
<script src="openseadragon.min.js"></script>
</head><body>
<div id="toolbar">{btn}<span id="lbl">Layer: Pure Colour</span></div>
<div id="osd"></div>
<script>
var viewer=OpenSeadragon({{id:"osd",prefixUrl:"osd_images/",tileSources:"albedo.dzi",showNavigator:true,navigatorPosition:"BOTTOM_RIGHT",zoomPerClick:1.5,animationTime:0.4}});
var labels={{{labels}}};
var active=null;
// Keep the view when switching layers: every layer is written at the same
// resolution cap, so viewport coordinates mean the same thing in all of them.
function switchLayer(name){{
  if(active)active.classList.remove("active");
  active=document.querySelector("button[onclick*=\\""+name+"\\"]");
  if(active)active.classList.add("active");
  document.getElementById("lbl").textContent="Layer: "+labels[name];
  var keep=null;
  try{{ if(viewer.viewport && viewer.world.getItemCount()>0)
          keep=viewer.viewport.getBounds(true); }}catch(e){{}}
  var restore=function(){{
    viewer.removeHandler("open",restore);
    if(!keep)return;
    try{{ viewer.viewport.fitBounds(keep,true); }}catch(e){{}}   // true = no animation
  }};
  viewer.addHandler("open",restore);
  viewer.open(name+".dzi");
}}
document.querySelector("#toolbar button").classList.add("active");
</script></body></html>"""


def sharable_launchers(folder: str) -> list[str]:
    """1_Mac.command and 2_Win.bat: serve the folder with no-cache headers
    and open index.html, so a browser always reads fresh files."""
    import time
    cb = str(int(time.time()))
    mac = ('#!/bin/bash\n'
           'DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"\n'
           'cd "$DIR"\n'
           'lsof -ti:8000 | xargs kill -9 2>/dev/null; sleep 0.3\n'
           'python3 -c "\nimport http.server, os\n'
           'folder=os.getcwd()\n'
           'class H(http.server.SimpleHTTPRequestHandler):\n'
           '  def __init__(self,*a,**k): super().__init__(*a,directory=folder,**k)\n'
           '  def end_headers(self):\n'
           '    self.send_header(\'Cache-Control\',\'no-store,no-cache\')\n'
           '    self.send_header(\'Access-Control-Allow-Origin\',\'*\')\n'
           '    super().end_headers()\n'
           '  def log_message(self,f,*a): pass\n'
           'http.server.ThreadingHTTPServer((\'\',8000),H).serve_forever()\n" &\n'
           'PID=$!; sleep 1\n'
           'open "http://localhost:8000/index.html?v=' + cb + '"\n'
           'echo "WebRTI viewer open. Press Enter to close."; read\n'
           'kill $PID 2>/dev/null\n')
    mp = os.path.join(folder, "1_Mac.command")
    with open(mp, "w") as f:
        f.write(mac)
    os.chmod(mp, 0o755)
    wp = os.path.join(folder, "2_Win.bat")
    with open(wp, "w") as f:
        f.write('@echo off\ncd /d "%~dp0"\nstart http://localhost:8000/index.html?v=' + cb
                + '\npython -m http.server 8000\npause\n')
    return [mp, wp]


def write_deepzoom(layers, out_dir: str, *, cancelled: Callable | None = None,
                   progress: Callable | None = None) -> list[str]:
    """Write each layer's DZI pyramid, the viewer page and OpenSeadragon
    (bundled, so the folder opens without a network), and the launchers.
    Returns [] if cancelled before the end."""
    from helio.core.processing import generate_dzi
    from helio.vendor_assets import osd_images_dir, read_bytes
    os.makedirs(out_dir, exist_ok=True)
    for i, (slug, img, _lbl) in enumerate(layers):
        if cancelled and cancelled():
            return []
        if progress:
            progress(f"Building pyramid {i + 1}/{len(layers)}: {slug}…", (i + 1) / len(layers))
        generate_dzi(img, out_dir, slug, 254, 1, 88)
    page = os.path.join(out_dir, "index.html")
    with open(page, "w", encoding="utf-8") as f:
        f.write(deepzoom_html(layers))
    with open(os.path.join(out_dir, "openseadragon.min.js"), "wb") as f:
        f.write(read_bytes("openseadragon.min.js"))
    dst = os.path.join(out_dir, "osd_images")
    os.makedirs(dst, exist_ok=True)
    for im in os.listdir(osd_images_dir()):
        if im.lower().endswith(".png"):
            shutil.copy2(os.path.join(osd_images_dir(), im), os.path.join(dst, im))
    return [page] + sharable_launchers(out_dir)


# ── multi-light composite ────────────────────────────────────────────────

COMPOSITE_MODES = ("Mean", "Max", "Weighted (STD)")


def multi_light_composite(draw: Callable, lights, mode: str = "Mean",
                          progress: Callable | None = None) -> tuple[np.ndarray, int]:
    """The view relit from every captured light, combined: the mean (shadows
    cancelled), the maximum, or weighted by each pixel's spread across
    lights.  *lights* are +Y up (the light matrix); *draw(lx, ly, lz)* takes
    the screen frame the views use, so y is flipped once here.  HELIO 2 put
    the +Y-up vectors into the screen-frame settings unflipped and relit
    every frame from the mirrored side (KD-043).  Returns (image, lights used)."""
    frames = []
    L = np.asarray(lights, np.float64)
    for i, lv in enumerate(L):
        mag = float(np.linalg.norm(lv))
        if mag < 1e-6:
            continue
        x, y, z = lv / mag
        frames.append(np.asarray(draw(float(x), float(-y), float(z))).astype(np.uint8))
        if progress:
            progress(f"Light {i + 1}/{len(L)}", (i + 1) / len(L))
    if not frames:
        raise NothingToExport("At least 2 calibrated light directions required.")
    stack = np.array(frames)
    if mode == "Max":
        out = stack.max(axis=0)
    elif mode.startswith("Weighted"):
        wts = stack.std(axis=0).mean(axis=-1, keepdims=True) + 1e-6
        out = (stack * wts).sum(axis=0) / (wts.sum(axis=0) + 1e-6)
    else:
        out = stack.mean(axis=0)
    return out.clip(0, 255).astype(np.uint8), len(frames)


# ── frequency maps ───────────────────────────────────────────────────────

def frequency_maps(depth_vis, out_dir: str, *, normals=None, depth_poisson=None, depth_fc=None,
                   hp_sigma: float = 0.0, mm_per_px=None, z_exag: float = 200.0,
                   obj_id: str = "", crop=None) -> list[str]:
    """The raw Poisson and Frankot–Chellappa heights and the low, high and
    ultra-high frequency bands of the depth as 32-bit GeoTIFFs at one unit
    per pixel (the physical scale in the tags, so QGIS's RVT radii stay in
    cells).  The two raw heights are integrated from the normals when not
    given."""
    import datetime
    from helio.core.depth_integration import frankot_chellappa, poisson_solver
    from helio.core.frames import normal_slopes
    from helio.io.exporters import write_geotiff
    if depth_vis is None:
        raise NothingToExport("Process images first.")
    x1, y1, x2, y2 = crop if crop else (0, 0, depth_vis.shape[1], depth_vis.shape[0])
    fdp = depth_vis[y1:y2, x1:x2]
    sigma = hp_sigma * 100.0 + 5.0
    low = cv2.GaussianBlur(fdp, (0, 0), sigma)
    high = fdp - low
    mm = float(mm_per_px) if mm_per_px else 1.0

    def nz(a):
        a = a.astype(np.float32)
        return (a - a.min()) / (float(a.max() - a.min()) + 1e-9) * z_exag
    uhf = fdp.astype(np.float32) - cv2.GaussianBlur(fdp, (0, 0), max(0.8, sigma * 0.05))
    specs = [("low_freq_shape_32f.tif", nz(low), "Low-frequency shape (macro form)"),
             ("high_freq_detail_32f.tif", nz(high) * 0.1, "High-frequency detail (embossing/inscription)"),
             ("ultra_high_freq_texture_32f.tif", nz(uhf) * 0.02,
              "Ultra-high-frequency texture (scratches/grain)")]
    if depth_fc is None and normals is not None:
        depth_fc = frankot_chellappa(*normal_slopes(normals))
    if depth_poisson is None and normals is not None:
        depth_poisson = poisson_solver(*normal_slopes(normals))
    if depth_poisson is not None:
        specs.insert(0, ("1_height_map_primary_32f.tif",
                         depth_poisson[y1:y2, x1:x2].astype(np.float32),
                         "Primary (Poisson-integrated) raw height map"))
    if depth_fc is not None:
        specs.insert(1, ("2_height_map_FC_32f.tif", depth_fc[y1:y2, x1:x2].astype(np.float32),
                         "Frankot-Chellappa raw height map"))
    tags = dict(HELIO_RES_MM=str(mm), HELIO_Z_SCALE_UNITS=str(z_exag),
                HELIO_CRS="EPSG:32630 (pixel size = 1.0, physical scale in HELIO_RES_MM)",
                HELIO_OBJECT_ID=obj_id or "", HELIO_DATE=datetime.date.today().isoformat())
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for name, arr, desc in specs:
        p = os.path.join(out_dir, name)
        write_geotiff(p, arr, tags={**tags, "HELIO_LAYER": desc})
        files.append(p)
    return files
