"""
Mesh, publication-sheet, and terrain-map exporters.

Each function is a **pure worker** that receives NumPy arrays and file
paths and returns the path(s) written.  UI progress callbacks are optional.
"""

from __future__ import annotations

import csv
import html
import io
import logging
import os
import time
import zipfile
from typing import Callable, Optional

_log = logging.getLogger(__name__)

import cv2
import numpy as np

from helio.core.frames import normal_divergence

from helio.config import (
    HAS_RASTERIO, MESH_MAX_RES, ALBEDO_TEX_RES, NORMAL_TEX_RES, RTI_VIEWER_MAX_SIDE,
)

# Standard library — late import to avoid cost on helio import
import base64 as _b64

if HAS_RASTERIO:
    import rasterio
    from rasterio.transform import from_origin


# ── GeoTIFF writing ────────────────────────────────────────────────
#
# Every GeoTIFF HELIO produces goes through write_geotiff() below.  Before
# consolidation there were six separate write paths that disagreed with each
# other: some wrote LZW-compressed, tiled, tagged rasters and some wrote bare
# uncompressed ones; some emitted a .tfw world file and some did not; and the
# batch DEM path used a different world-file pixel size from every other path,
# so the same DEM landed at a different scale in QGIS depending on which
# button produced it.  Keep new export code calling this function.

_NODATA = -9999.0

# HELIO's world-file convention: pixel size is always 1.0 unit/pixel and the
# physical scale travels in the HELIO_RES_MM tag.  Writing the true sub-mm
# value here would make QGIS/RVT compute search neighbourhoods of 100,000+
# cells per pixel, which never finish.
_PIXEL_SIZE = 1.0


def write_tfw(tif_path: str, height: int, pixel_size: float = _PIXEL_SIZE) -> str:
    """Write the .tfw sidecar next to *tif_path*.  Returns the .tfw path.

    A world file is dumb but universal: it survives re-saves and tools that
    strip embedded GeoTIFF tags, so it is cheap insurance even when the
    raster itself is already georeferenced.
    """
    wld = os.path.splitext(tif_path)[0] + ".tfw"
    with open(wld, "w") as wf:
        wf.write(f"{float(pixel_size)}\n0.0\n0.0\n{-float(pixel_size)}\n"
                 f"0.0\n{float(height) * float(pixel_size)}\n")
    return wld


def write_geotiff(
    path: str,
    data: np.ndarray | None = None,
    *,
    rgb: np.ndarray | None = None,
    crs: str = "EPSG:32630",
    transform=None,
    nodata: float | None = _NODATA,
    tags: dict[str, str] | None = None,
    world_file: bool = True,
) -> str:
    """Write one GeoTIFF — the single canonical write path.

    Uses *rasterio* when installed (LZW + predictor, 256×256 tiles, nodata
    and metadata tags) and falls back to ``cv2.imwrite`` otherwise.  The
    .tfw sidecar is written in both cases, so georeferencing survives even
    without rasterio.

    Parameters
    ----------
    data : 2D array for a single-band raster, or None when *rgb* is given.
    rgb  : (H, W, 3) array for a 3-band raster.
    tags : extra metadata (e.g. ``HELIO_OBJECT_ID``); ignored without rasterio.

    Returns *path*.
    """
    if data is None and rgb is None:
        raise ValueError("write_geotiff needs either data= or rgb=")

    if rgb is not None:
        arr = np.asarray(rgb)
        height, width = arr.shape[:2]
        bands = arr.shape[2] if arr.ndim == 3 else 1
    else:
        arr = np.asarray(data)
        height, width = arr.shape[:2]
        bands = 1
    dtype = "uint8" if arr.dtype == np.uint8 else "float32"
    if dtype == "float32":
        arr = arr.astype(np.float32)

    if HAS_RASTERIO:
        if transform is None:
            transform = from_origin(0, float(height), _PIXEL_SIZE, _PIXEL_SIZE)
        kw: dict = dict(
            driver="GTiff", height=height, width=width,
            count=bands, dtype=dtype, crs=crs, transform=transform,
            compress="lzw", predictor=3 if dtype == "float32" else 2,
            tiled=True, blockxsize=256, blockysize=256,
        )
        if dtype == "float32" and nodata is not None:
            kw["nodata"] = nodata
        # LZW-compress the 256 px tiles on every core: the same file, ~3x sooner
        # (a 12 MP float32 DEM: 0.42 s -> 0.12 s; a 100 MP one is tens of seconds).
        kw["num_threads"] = "all_cpus"
        with rasterio.open(path, "w", **kw) as dst:
            if bands == 1:
                dst.write(arr if arr.ndim == 2 else arr[:, :, 0], 1)
            else:
                for i in range(bands):
                    dst.write(arr[:, :, i], i + 1)
            if tags:
                dst.update_tags(**{k: str(v) for k, v in tags.items()})
    else:
        # cv2 wants BGR channel order for colour; single-band writes as-is.
        out = arr
        if bands == 3:
            out = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, out, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    if world_file:
        write_tfw(path, height)
    return path


def save_geotiff(
    path: str,
    data: np.ndarray,
    height: int,
    width: int,
    res_mm: float = 1.0,
    crs: str = "EPSG:32630",
) -> str:
    """Write a single float32 GeoTIFF.  Thin wrapper over write_geotiff()."""
    return write_geotiff(path, data.astype(np.float32), crs=crs,
                         tags={"HELIO_RES_MM": f"{res_mm:.6f}"})


def save_geotiff_dict(
    dir_path: str,
    maps: dict[str, np.ndarray],
    height: int,
    width: int,
    res_mm: float = 1.0,
) -> list[str]:
    """Write multiple float32 GeoTIFFs to *dir_path*.

    Returns list of written paths.
    """
    os.makedirs(dir_path, exist_ok=True)
    written = []
    for name, data in maps.items():
        path = os.path.join(dir_path, name)
        save_geotiff(path, data, height, width, res_mm)
        written.append(path)
    return written


def write_geotiff_with_tags(
    path: str,
    data: np.ndarray | None,
    height: int,
    width: int,
    bands: int = 1,
    dtype: str = "float32",
    crs: str = "EPSG:32630",
    transform=None,
    nodata: float = _NODATA,
    tags: dict[str, str] | None = None,
    rgb: np.ndarray | None = None,
) -> str:
    """Backwards-compatible wrapper over :func:`write_geotiff`.

    Retained so existing callers keep working; new code should call
    ``write_geotiff`` directly.  Unlike the old implementation this no longer
    raises without rasterio — it degrades to a plain TIFF plus world file.
    """
    if rgb is not None and dtype == "uint8":
        rgb = np.asarray(rgb).astype(np.uint8)
    return write_geotiff(path, data, rgb=rgb, crs=crs, transform=transform,
                         nodata=nodata if dtype == "float32" else None,
                         tags=tags)


def save_dem_package(
    out_dir: str,
    obj_id: str,
    arr_poisson: np.ndarray,
    arr_fc: np.ndarray,
    albedo: np.ndarray | None = None,
    normals: np.ndarray | None = None,
    crs: str = "EPSG:32630",
    tags: dict[str, str] | None = None,
    mm_per_px: float = 1.0,
    write_world_file: bool = True,
) -> dict[str, str]:
    """Write a full DEM package GeoTIFF set.

    Returns dict of logical name → file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    h, w = arr_poisson.shape
    if tags is None:
        tags = {}
    if "HELIO_OBJECT_ID" not in tags:
        tags["HELIO_OBJECT_ID"] = obj_id

    paths = {}
    p_poisson = os.path.join(out_dir, f"{obj_id}_DEM_Poisson.tif")
    paths["poisson"] = p_poisson
    p_fc = os.path.join(out_dir, f"{obj_id}_DEM_FrankotChellappa.tif")
    paths["fc"] = p_fc

    tags = {**tags, "HELIO_RES_MM": f"{mm_per_px:.6f}"}
    write_geotiff(p_poisson, arr_poisson, crs=crs, tags=tags,
                  world_file=write_world_file)
    write_geotiff(p_fc, arr_fc, crs=crs, tags=tags,
                  world_file=write_world_file)

    if albedo is not None:
        p_alb = os.path.join(out_dir, f"{obj_id}_Albedo_RGB.tif")
        paths["albedo"] = p_alb
        if albedo.shape[:2] != (h, w):
            albedo = cv2.resize(albedo, (w, h), interpolation=cv2.INTER_AREA)
        write_geotiff(p_alb, rgb=albedo.astype(np.uint8), crs=crs, tags=tags,
                      world_file=write_world_file)

    if normals is not None:
        p_nrm = os.path.join(out_dir, f"{obj_id}_NormalMap.tif")
        paths["normals"] = p_nrm
        nrm = normals.astype(np.float32)
        if nrm.shape[:2] != (h, w):
            nrm = cv2.resize(nrm, (w, h), interpolation=cv2.INTER_LINEAR)
        write_geotiff(p_nrm, rgb=nrm, crs=crs, tags=tags,
                      world_file=write_world_file)

    return paths


# ── Terrain analysis maps ──────────────────────────────────────────

def save_terrain_maps(
    dir_path: str,
    maps: dict[str, np.ndarray],
    height: int,
    width: int,
    res_mm: float = 1.0,
    has_rvt: bool = False,
) -> list[str]:
    """Save a dict of terrain analysis maps as 32-bit float GeoTIFFs.

    Returns list of written paths.
    """
    return save_geotiff_dict(dir_path, maps, height, width, res_mm)


def save_hillshade_maps(
    dir_path: str,
    height_map: np.ndarray,
    height_fc: np.ndarray,
    hillshade_primary: np.ndarray,
    hillshade_fc: np.ndarray,
    multi_hs: np.ndarray,
    svf: np.ndarray,
    res_mm: float = 1.0,
) -> list[str]:
    """Save the standard 6-map hillshade export set.

    Returns list of written paths.
    """
    maps = {
        "1_height_map_primary_32f.tif": height_map,
        "2_height_map_FC_32f.tif": height_fc,
        "3_hillshade_primary_32f.tif": hillshade_primary,
        "4_hillshade_FC_32f.tif": hillshade_fc,
        "5_multi_directional_hillshade_32f.tif": multi_hs,
        "6_sky_view_factor_32f.tif": svf,
    }
    h, w = height_map.shape[:2]
    return save_geotiff_dict(dir_path, maps, h, w, res_mm)


def save_gis_analysis_maps(
    dir_path: str,
    pos_opn: np.ndarray,
    neg_opn: np.ndarray,
    msrm_arr: np.ndarray,
    curv: np.ndarray,
    slope_arr: np.ndarray,
    loc_dom: np.ndarray,
    svf: np.ndarray,
    res_mm: float = 1.0,
) -> list[str]:
    """Save the GIS analysis 7-map export set.

    Returns list of written paths.
    """
    maps = {
        "1_positive_openness_32f.tif": pos_opn,
        "2_negative_openness_32f.tif": neg_opn,
        "3_multi_scale_relief_msrm_32f.tif": msrm_arr,
        "4_surface_curvature_32f.tif": curv,
        "5_slope_gradient_32f.tif": slope_arr,
        "6_local_dominance_32f.tif": loc_dom,
        "7_sky_view_factor_32f.tif": svf,
    }
    h, w = svf.shape[:2]
    return save_geotiff_dict(dir_path, maps, h, w, res_mm)


# ── Heritage Archive (ZIP + GLB + textures + stats) ────────────────


def write_heritage_archive(
    out_path: str,
    albedo: np.ndarray,
    normals: np.ndarray,
    depth: np.ndarray,
    depth_poisson: np.ndarray,
    object_id: str = "HELIO_Object",
    crop_box: tuple[int, int, int, int] | None = None,
    export_step: int = 1,
    height_scale: float = 300.0,
    normal_engine: str = "",
    depth_engine: str = "",
    forensic_maps: dict[str, np.ndarray] | None = None,
    progress_cb: Callable[[str, float], None] | None = None,
) -> str:
    """Write a heritage archive ZIP (GLB + textures + stats + README).

    Parameters
    ----------
    out_path : str
        Destination ``.zip`` path.
    albedo : np.ndarray
        RGB albedo (H, W, 3) uint8.
    normals : np.ndarray
        PS normal map (H, W, 3) float32, values in [-1, 1].
    depth : np.ndarray
        Depth map (H, W) float32.
    depth_poisson : np.ndarray
        Poisson-integrated depth (H, W) float32.
    object_id : str
        Identifier used for filenames inside the archive.
    crop_box : tuple or None
        (x1, y1, x2, y2) crop region.  If None, uses full arrays.
    export_step : int
        Vertex decimation step for GLB mesh (1 = full res).
    height_scale : float
        Z-scale for mesh extrusion.
    normal_engine, depth_engine : str
        Label strings stored in README.
    forensic_maps : dict or None
        Optional extra forensic maps {name: array}.
    progress_cb : Callable or None
        Called with (message, fraction).

    Returns
    -------
    str
        *out_path* on success.
    """
    _upd = progress_cb or (lambda msg, frac: None)
    safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in object_id)
    date_str = time.strftime("%Y-%m-%d")

    if crop_box:
        x1, y1, x2, y2 = crop_box
    else:
        y1, y2, x1, x2 = 0, albedo.shape[0], 0, albedo.shape[1]

    _upd("Encoding albedo…", 0.05)
    alb_crop = albedo[y1:y2, x1:x2]
    _, alb_buf = cv2.imencode(".png", cv2.cvtColor(alb_crop, cv2.COLOR_RGB2BGR))

    _upd("Encoding normals…", 0.15)
    nm8 = np.round((normals[y1:y2, x1:x2] * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    _, nm_buf = cv2.imencode(".png", cv2.cvtColor(nm8, cv2.COLOR_RGB2BGR))

    _upd("Exporting GLB mesh…", 0.25)
    glb_bytes = None
    try:
        import trimesh

        h, w = y2 - y1, x2 - x1
        ys, xs = np.mgrid[0:h:export_step, 0:w:export_step]
        gh, gw = len(ys[:, 0]), len(xs[0, :])
        z = depth_poisson[y1:y2, x1:x2][0:h:export_step, 0:w:export_step]
        verts = np.column_stack((
            (xs - w / 2).flatten(),
            (ys - h / 2).flatten(),
            (z * height_scale - height_scale / 2).flatten(),
        )).astype(np.float32)
        rr, cc = np.meshgrid(np.arange(gh - 1), np.arange(gw - 1), indexing="ij")
        v0 = rr * gw + cc
        faces = np.vstack((
            np.column_stack((v0.flatten(), (v0 + gw).flatten(), (v0 + 1).flatten())),
            np.column_stack(((v0 + 1).flatten(), (v0 + gw).flatten(), (v0 + gw + 1).flatten())),
        )).astype(np.uint32)
        colors = alb_crop[0:h:export_step, 0:w:export_step].reshape(-1, 3)
        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces,
            vertex_colors=np.column_stack((colors, np.full(len(colors), 255))).astype(np.uint8),
            process=False,
        )
        buf = io.BytesIO()
        mesh.export(buf, file_type="glb")
        glb_bytes = buf.getvalue()
    except Exception:
        _log.warning("GLB mesh export failed, skipping")
        glb_bytes = None

    _upd("Computing surface statistics…", 0.55)
    d = depth[y1:y2, x1:x2].astype(np.float64)
    m = np.mean(d)
    sq = float(np.sqrt(np.mean((d - m) ** 2)))
    stats = {
        "Sa": round(float(np.mean(np.abs(d - m))), 5),
        "Sq": round(sq, 5),
        "Sz": round(float(d.max() - d.min()), 5),
        "Ssk": round(float(np.mean((d - m) ** 3) / (sq ** 3 + 1e-12)), 4),
        "Sku": round(float(np.mean((d - m) ** 4) / (sq ** 4 + 1e-12)), 4),
    }
    csv_buf = io.StringIO()
    cw = csv.writer(csv_buf)
    cw.writerow(["Parameter", "Value", "ISO 25178 Definition"])
    defs = {"Sa": "Areal mean height", "Sq": "RMS height", "Sz": "Max height",
            "Ssk": "Skewness", "Sku": "Kurtosis"}
    for k, v in stats.items():
        cw.writerow([k, v, defs.get(k, "")])

    _upd("Writing ZIP…", 0.75)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(f"{safe_id}_albedo.png", alb_buf.tobytes())
        zf.writestr(f"{safe_id}_normal.png", nm_buf.tobytes())
        if glb_bytes:
            zf.writestr(f"{safe_id}_model.glb", glb_bytes)
        zf.writestr(f"{safe_id}_surface_stats.csv", csv_buf.getvalue().encode())

        if forensic_maps:
            for fname, farr in forensic_maps.items():
                try:
                    _, fbuf = cv2.imencode(
                        ".tif",
                        cv2.normalize(farr.astype(np.float32), None, 0, 65535,
                                      cv2.NORM_MINMAX).astype(np.uint16),
                        [cv2.IMWRITE_TIFF_COMPRESSION, 1],
                    )
                    zf.writestr(fname, fbuf.tobytes())
                except Exception:
                    _log.warning("Failed to add forensic map %s to archive", fname)

        readme = (
            f"Helio — Heritage Archive\n{'='*50}\n"
            f"Object:  {object_id}\nDate:    {date_str}\n"
            f"Size:    {w} x {h} px\n"
            f"Normal engine: {normal_engine}\n"
            f"Depth engine:  {depth_engine}\n\n"
            f"FILES\n{'─'*40}\n"
            f"  {safe_id}_albedo.png        — colour photograph\n"
            f"  {safe_id}_normal.png        — PS normal map\n"
            f"  {safe_id}_model.glb         — 3D mesh (GLB)\n"
            f"  {safe_id}_surface_stats.csv — ISO 25178 stats\n"
            f"  MANIFEST.txt                — this file\n\n"
            f"Generated by Helio\n"
        )
        zf.writestr("MANIFEST.txt", readme.encode())

    _upd("Done.", 1.0)
    return out_path


# ── QGIS Plugin ZIP generation ─────────────────────────────────────


# Helio Relief: Scott Maloney's QGIS plugin, made alongside Helio (its Import
# HELIO reads Helio's GIS height-map package).  Its source is qgis_plugin/;
# qgis_plugin/dev/build_zip.sh builds this ZIP.  Helio only copies it out.
QGIS_PLUGIN_ZIP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "vendor", "helio_relief_v4.1.0.zip")


def write_qgis_plugin_zip(out_path: str, object_id: str = "HELIO") -> str:
    """Helio Relief's installable ZIP, copied to
    *out_path* (QGIS: Plugins > Manage and Install Plugins > Install from ZIP).
    *object_id* is unused; kept for the classic app's call."""
    import shutil
    if not os.path.isfile(QGIS_PLUGIN_ZIP):
        raise FileNotFoundError("Helio Relief (the QGIS plugin) is missing from this "
                                "copy of Helio (helio/vendor).")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shutil.copyfile(QGIS_PLUGIN_ZIP, out_path)
    return out_path


# ── 3D mesh exporters (extracted from Helio_Pro methods) ────────────

import io as _io
import json as _json
import math as _math
import os as _os
import struct as _struct
import zipfile as _zfm
from typing import Callable, Optional

import cv2 as _cv2
import numpy as _np
from PIL import Image as _Image

from helio.core.processing import (
    bake_normal_detail,
    build_grid_mesh,
    build_watertight_mesh_data,
    compute_export_depth_grid,
    compute_hf_depth_from_normals,
    compute_vertex_normals_from_ps,
    pack_binary_stl,
)


def _uv_to_trimesh(uv: _np.ndarray) -> _np.ndarray:
    """Convert image-convention UVs (v=0 = image TOP row) to trimesh's
    OpenGL-style convention (v=0 = image BOTTOM row).

    Every UV grid in this codebase is built in image convention. trimesh,
    however, assumes incoming UVs are OpenGL-convention and converts them
    to glTF's top-left origin on GLB export by writing ``1 - v``. Handing
    it image-convention UVs therefore double-flips: the written file tells
    every spec-compliant renderer (three.js, Blender, Sketchfab) to sample
    the texture vertically mirrored against the geometry. Proven by reading
    the raw TEXCOORD_0 accessor back from an exported GLB: the image-top
    vertex we assigned v=0 was stored as v=1. trimesh's own loader flips
    back on import, so a trimesh round-trip can NOT detect this — only a
    raw parse or a real renderer shows it.

    Call this on any image-convention UV array immediately before passing
    it to ``trimesh.visual.TextureVisuals``.
    """
    uv = _np.asarray(uv, dtype=_np.float32).copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    return uv


def _tex_resize(img_bgr: _np.ndarray, res: str) -> _np.ndarray:
    """Resize a texture to a square, centre-cropped size."""
    if res == "Full":
        return img_bgr
    sz = int(res)
    ih, iw = img_bgr.shape[:2]
    sc = sz / max(ih, iw)
    nh, nw = int(ih * sc), int(iw * sc)
    r = _cv2.resize(img_bgr, (nw, nh), interpolation=_cv2.INTER_LANCZOS4)
    c = _np.zeros((sz, sz, r.shape[2]), dtype=r.dtype)
    yo, xo = (sz - nh) // 2, (sz - nw) // 2
    c[yo:yo + nh, xo:xo + nw] = r
    return c


def _fix_glb_material(
    glb_path: str,
    metallic: float = 0.0,
    roughness: float = 0.65,
    double_sided: bool = True,
) -> None:
    """Post-process a GLB to set correct PBR material parameters.

    Opens the binary GLB, locates the JSON chunk, patches
    metallicFactor/roughnessFactor/doubleSided, and rewrites the file.
    """
    try:
        import json as _js
        with open(glb_path, "rb") as _f:
            _magic = _f.read(4)
            _ver = int.from_bytes(_f.read(4), "little")
            _len = int.from_bytes(_f.read(4), "little")
            _chunks = []
            while _f.tell() < _len:
                _clen = int.from_bytes(_f.read(4), "little")
                _ctype = int.from_bytes(_f.read(4), "little")
                _data = _f.read(_clen)
                _chunks.append((_ctype, _data))
        _gltf = _js.loads(_chunks[0][1])
        if "materials" in _gltf:
            for _mat in _gltf["materials"]:
                if "pbrMetallicRoughness" not in _mat:
                    _mat["pbrMetallicRoughness"] = {}
                _mat["pbrMetallicRoughness"]["metallicFactor"] = metallic
                _mat["pbrMetallicRoughness"]["roughnessFactor"] = roughness
                _mat["doubleSided"] = double_sided
        _new_json = _js.dumps(_gltf, separators=(",", ":")).encode()
        _pad = (4 - len(_new_json) % 4) % 4
        _new_json += b" " * _pad
        _bin = _chunks[1][1] if len(_chunks) > 1 else b""
        with open(glb_path, "wb") as _fw:
            _fw.write(b"glTF")
            _fw.write((2).to_bytes(4, "little"))
            _fw.write((12 + 8 + len(_new_json) + 8 + len(_bin)).to_bytes(4, "little"))
            _fw.write(len(_new_json).to_bytes(4, "little"))
            _fw.write((0x4E4F534A).to_bytes(4, "little"))  # JSON
            _fw.write(_new_json)
            if _bin:
                _fw.write(len(_bin).to_bytes(4, "little"))
                _fw.write((0x004E4942).to_bytes(4, "little"))  # BIN
                _fw.write(_bin)
    except Exception as _e:
        _log.warning("GLB material fix skipped: %s", _e, exc_info=True)


def _glb_textures_jpeg(glb_path: str, quality: int = 92) -> None:
    """Re-encode a GLB's PNG textures as JPEG at *quality* (trimesh writes PNG,
    lossless and three to four times the size), and rebuild its binary chunk.
    A photograph's colour and a smooth normal map lose nothing visible at 90+,
    so the bytes saved go to a larger texture and a finer mesh instead."""
    import io as _io
    import json as _js
    import struct as _st
    from PIL import Image as _Img
    with open(glb_path, "rb") as f:
        raw = f.read()
    jl = _st.unpack_from("<I", raw, 12)[0]
    gltf = _js.loads(raw[20:20 + jl])
    bin_at = 20 + jl
    bl = _st.unpack_from("<I", raw, bin_at)[0]
    blob = raw[bin_at + 8:bin_at + 8 + bl]
    views = gltf.get("bufferViews", [])
    swap = {}
    for im in gltf.get("images", []):
        if im.get("mimeType") != "image/png" or "bufferView" not in im:
            continue
        bv = views[im["bufferView"]]
        o = bv.get("byteOffset", 0)
        pic = _Img.open(_io.BytesIO(blob[o:o + bv["byteLength"]]))
        if pic.mode not in ("RGB", "L"):
            pic = pic.convert("RGB")
        buf = _io.BytesIO()
        pic.save(buf, "JPEG", quality=int(quality), subsampling=0 if quality >= 90 else 2,
                 optimize=True)
        swap[im["bufferView"]] = buf.getvalue()
        im["mimeType"] = "image/jpeg"
    if not swap:
        return
    out = bytearray()
    for i, bv in enumerate(views):                  # every view kept in order, 4-byte aligned
        o = bv.get("byteOffset", 0)
        data = swap.get(i, blob[o:o + bv["byteLength"]])
        out += b"\0" * ((4 - len(out) % 4) % 4)
        bv["byteOffset"], bv["byteLength"] = len(out), len(data)
        out += data
    out += b"\0" * ((4 - len(out) % 4) % 4)
    gltf["buffers"][0]["byteLength"] = len(out)
    js = _js.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(glb_path, "wb") as f:
        f.write(b"glTF" + _st.pack("<II", 2, 12 + 8 + len(js) + 8 + len(out)))
        f.write(_st.pack("<I", len(js)) + b"JSON" + js)
        f.write(_st.pack("<I", len(out)) + b"BIN\0" + bytes(out))


def export_stl_3dprint(
    depth_vis_p: _np.ndarray,
    depth_normalised: Optional[_np.ndarray],
    normals: Optional[_np.ndarray],
    settings: dict,
    filepath: str,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> str:
    """Export a watertight binary STL suitable for 3D printing.

    Builds the mesh from a depth grid, computes face normals explicitly,
    and writes a valid manifold STL with top, bottom, and side walls.

    Parameters
    ----------
    depth_vis_p : np.ndarray
        High-pass depth image.
    depth_normalised : np.ndarray or None
        Full-range normalised depth.
    normals : np.ndarray or None
        PS normals (for HD path).
    settings : dict
        Keys: ``relief_mm``, ``base_mm``, ``max_res``, ``mirror_x``,
        ``bed_mm``, ``z_ex``, ``fz``, ``shape_blend``, ``smooth``,
        ``step``, ``hp_sigma``, ``warp_rem``, ``micro``, ``use_hd``,
        ``hd_warp_pct``.
    filepath : str
        Output ``.stl`` path.
    progress_cb : callable or None
        ``(message, fraction)``

    Returns
    -------
    str
        *filepath* on success.
    """
    if progress_cb:
        progress_cb("Building depth grid...", 0.0)

    relief_mm = settings.get("relief_mm", 3.0)
    base_mm = settings.get("base_mm", 3.0)
    bed_mm = settings.get("bed_mm", 256.0)
    z_ex = settings.get("z_ex", 1.0)
    mirror_x = settings.get("mirror_x", False)

    depth_grid = compute_export_depth_grid(
        depth_vis_p, depth_normalised, normals,
        {**settings, "mirror_x": mirror_x},
        progress_cb,
    )
    gh, gw = depth_grid.shape

    if progress_cb:
        progress_cb(f"Building {gh * gw:,} vertices...", 0.3)

    y_s, x_s = _np.mgrid[0:gh, 0:gw]

    # Physical scale.
    #
    # "true_mm_per_px" is millimetres per *grid* pixel, supplied by the caller
    # when the capture has been calibrated.  Given it, the model comes out at
    # life size and a 10mm feature measures 10mm - which is the point of
    # calibrating in the first place.  Without it the model is scaled to fill
    # the print bed instead, which is fine for a decorative print and useless
    # for measurement: a true 60x40mm plate exported at the default 256mm bed
    # came out 255.6 x 170.2mm, and its 10mm ridge measured 42mm.
    #
    # The bed path also divides by (n - 1), not n.  A grid of n samples spans
    # n-1 intervals, so the old formula left the model exactly one pixel short
    # on each axis - 59.900mm for a 60.000mm object - and meant the bed size
    # was never actually reached.
    # "mm_per_px" is millimetres per *source* pixel, straight from the
    # metrology panel.  The grid here is a downsampled copy of that source, so
    # scale it by how much the grid shrank rather than making the caller work
    # that out - the caller cannot know the grid size, because this function
    # chooses it.
    _src_mm_px = float(settings.get("mm_per_px", 0.0) or 0.0)
    if _src_mm_px > 0:
        _src_h, _src_w = depth_vis_p.shape[:2]
        px_mm = _src_mm_px * (float(_src_w) / float(max(1, gw)))
    else:
        px_mm = bed_mm / max(1, max(gh, gw) - 1)
    X = (x_s.flatten() * px_mm).astype(_np.float32)
    Y = (y_s.flatten() * px_mm).astype(_np.float32)
    Z = (depth_grid.flatten() * relief_mm * z_ex + base_mm).astype(_np.float32)

    vt = _np.column_stack((X, Y, Z))
    vb = _np.column_stack((X, Y, _np.zeros(len(X), dtype=_np.float32)))

    if progress_cb:
        progress_cb("Building surface triangles...", 0.5)
    rr, cc = _np.meshgrid(_np.arange(gh - 1), _np.arange(gw - 1), indexing="ij")
    v00 = (rr * gw + cc).flatten()
    v01 = (rr * gw + cc + 1).flatten()
    v10 = ((rr + 1) * gw + cc).flatten()
    v11 = ((rr + 1) * gw + cc + 1).flatten()

    top_t1 = _np.stack([vt[v00], vt[v01], vt[v11]], axis=1)
    top_t2 = _np.stack([vt[v00], vt[v11], vt[v10]], axis=1)
    bot_t1 = _np.stack([vb[v00], vb[v11], vb[v01]], axis=1)
    bot_t2 = _np.stack([vb[v00], vb[v10], vb[v11]], axis=1)

    if progress_cb:
        progress_cb("Building side walls...", 0.7)
    p_top = [(0, c) for c in range(gw - 1)]
    p_right = [(r, gw - 1) for r in range(gh - 1)]
    p_bot = [(gh - 1, c) for c in range(gw - 1, 0, -1)]
    p_left = [(r, 0) for r in range(gh - 1, 0, -1)]
    perim = _np.array([r * gw + c for r, c in p_top + p_right + p_bot + p_left])
    nxt = _np.roll(perim, -1)

    st, sn = vt[perim], vt[nxt]
    sb_c, sb_n = vb[perim], vb[nxt]
    side_t1 = _np.stack([sn, st, sb_c], axis=1)
    side_t2 = _np.stack([sn, sb_c, sb_n], axis=1)

    all_tris = _np.concatenate(
        [top_t1, top_t2, bot_t1, bot_t2, side_t1, side_t2],
        axis=0,
    ).astype(_np.float32)

    if progress_cb:
        progress_cb("Writing STL file...", 0.9)
    n_tri, fsize_mb = pack_binary_stl(all_tris, filepath)

    if progress_cb:
        progress_cb(f"Done: {fsize_mb:.1f} MB, {n_tri:,} triangles", 1.0)
    print(f"[STL] {n_tri:,} triangles, {fsize_mb:.1f} MB -> {filepath}")
    return filepath


def export_3d_mesh(
    depth_vis_p: _np.ndarray,
    depth_normalised: Optional[_np.ndarray],
    normals: _np.ndarray,
    albedo_rgb: _np.ndarray,
    settings: dict,
    filepath: str,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> str:
    """Export a 3D mesh in GLB, OBJ, or PLY format with PBR textures.

    Supports GLB (with embedded albedo + normal textures), OBJ (with
    companion MTL + albedo/normal PNGs), and PLY point cloud.

    Parameters
    ----------
    depth_vis_p : np.ndarray
        High-pass depth (HP detail layer).
    depth_normalised : np.ndarray or None
        Full-range normalised depth.
    normals : np.ndarray
        Full-resolution PS normal map (H, W, 3).
    albedo_rgb : np.ndarray
        Full-resolution RGB albedo (H, W, 3), uint8.
    settings : dict
        Keys: ``step``, ``height``, ``fx``, ``fy``, ``fz``, ``use_hd``,
        ``use_hybrid``, ``detail_mix``, ``shape_blend``, ``hf_cutoff``,
        ``amp``, ``max_res``, ``hp_sigma``, ``smooth``, ``warp_rem``,
        ``micro``, ``edge_trim``, ``center_origin``, ``vertex_colors``,
        ``export_brightness``, ``watertight``, ``sketchfab_pbr``,
        ``normal_flip_y``, ``ao_blend``, ``hd_warp_pct``, ``crop``,
        ``bake_normals``, ``bake_depth_pct``.
    filepath : str
        Output path (``.glb``, ``.obj``, or ``.ply``).
    progress_cb : callable or None

    Returns
    -------
    str
        *filepath* on success.
    """
    import trimesh

    step = settings.get("step", 2)
    fx = settings.get("fx", 1)
    fy = settings.get("fy", -1)
    fz = settings.get("fz", 1)
    watertight = settings.get("watertight", False)
    use_vtx_color = settings.get("vertex_colors", True)
    exp_bright = settings.get("export_brightness", 1.0)
    ao_blend = settings.get("ao_blend", 0.0)
    normal_flip_y = settings.get("normal_flip_y", False)
    sketchfab_pbr = settings.get("sketchfab_pbr", False)

    x1, y1, x2, y2 = settings.get("crop", (0, 0, albedo_rgb.shape[1], albedo_rgb.shape[0]))

    if progress_cb:
        progress_cb("Building depth grid...", 0.0)

    depth_grid = compute_export_depth_grid(
        depth_vis_p[y1:y2, x1:x2],
        depth_normalised[y1:y2, x1:x2] if depth_normalised is not None else None,
        normals[y1:y2, x1:x2] if normals is not None else None,
        settings,
        progress_cb,
    )

    # Bake the normal map's high-frequency relief into the geometry, so the
    # detail survives in viewers that ignore normal textures.  Applied to the
    # grid before build_grid_mesh, which is why _build_export_settings holds
    # step at 1 for every caller that switches this on.
    if settings.get("bake_normals", False) and normals is not None:
        if progress_cb:
            progress_cb("Baking normal detail into geometry...", 0.25)
        depth_grid = bake_normal_detail(
            depth_grid,
            normals[y1:y2, x1:x2],
            depth_pct=float(settings.get("bake_depth_pct", 0.05)),
        )

    h, w = depth_grid.shape

    _crop_albedo = albedo_rgb[y1:y2, x1:x2]

    if progress_cb:
        progress_cb("Building mesh geometry...", 0.3)

    verts, faces, colors, uvs = build_grid_mesh(
        depth_grid,
        {**settings, "albedo": _crop_albedo},
        progress_cb,
    )

    # AO blend on vertex colours
    if ao_blend > 0.01 and use_vtx_color and filepath.endswith(".glb"):
        gh = int(_math.sqrt(len(verts)))  # approximate
        gw = len(verts) // gh
        if gh * gw == len(verts):
            z_reshaped = depth_grid[0:h:step, 0:w:step].copy()
            _gz, _gx = _np.gradient(z_reshaped)
            _ao = _np.power(_np.clip(1.0 / _np.sqrt(1.0 + _gx**2 + _gz**2), 0, 1), 2.5).flatten()
            _cf = colors.astype(_np.float32)
            colors = _np.clip(ao_blend * (_cf * _ao[:, _np.newaxis]) + (1.0 - ao_blend) * _cf, 0, 255).astype(_np.uint8)

    is_point_cloud = filepath.lower().endswith(".ply") and settings.get("point_cloud", False)

    if is_point_cloud:
        colors_rgba = _np.column_stack((colors, _np.full(len(colors), 255, dtype=_np.uint8)))
        trimesh.points.PointCloud(verts, colors=colors_rgba).export(filepath)
        return filepath

    is_glb = filepath.lower().endswith(".glb") or filepath.lower().endswith(".gltf")

    if watertight:
        if progress_cb:
            progress_cb("Building watertight mesh...", 0.5)
        all_verts, all_faces, all_colors, all_uvs, n_top = build_watertight_mesh_data(
            verts, faces, colors, uvs, settings
        )
        verts_final, faces_final, colors_final = all_verts, all_faces, all_colors
        _all_uvs = all_uvs
        _n_top_verts = n_top
    else:
        verts_final, faces_final, colors_final = verts, faces, colors
        _all_uvs = uvs
        _n_top_verts = len(verts)

    # Vertex normals from PS normals
    vertex_normals = compute_vertex_normals_from_ps(
        normals, {**settings, "crop": (x1, y1, x2, y2)}
    )
    if watertight:
        vn_flat = _np.array([0.0, 0.0, -1.0], dtype=_np.float32)
        n_bottom = _np.tile(vn_flat, (len(verts_final) - _n_top_verts, 1))
        all_vnormals = _np.vstack([vertex_normals, n_bottom])
    else:
        all_vnormals = vertex_normals

    if progress_cb:
        progress_cb("Exporting mesh...", 0.7)

    if is_glb:
        try:
            alb_grid = _crop_albedo[0:h:step, 0:w:step].astype(_np.uint8)
            if exp_bright != 1.0:
                alb_grid = _np.clip(
                    _np.power(_np.clip(alb_grid.astype(_np.float32) / 255.0, 0, 1),
                              1.0 / max(0.1, exp_bright)) * 255.0,
                    0, 255,
                ).astype(_np.uint8)
            albedo_texture_img = _Image.fromarray(alb_grid)

            n_tex = normals[y1:y2, x1:x2][0:h:step, 0:w:step].copy().astype(_np.float32)
            nm_rgb = _np.round((n_tex * 0.5 + 0.5) * 255.0).clip(0, 255).astype(_np.uint8)
            normal_texture_img = _Image.fromarray(nm_rgb)

            if watertight:
                _alb_patched = _np.array(albedo_texture_img)
                _patch_sz = max(4, _alb_patched.shape[0] // 64)
                _alb_patched[:_patch_sz, :_patch_sz] = 0
                albedo_texture_img = _Image.fromarray(_alb_patched)

            pbr_mat = trimesh.visual.material.PBRMaterial(
                name="helio_pbr",
                baseColorTexture=albedo_texture_img,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                normalTexture=normal_texture_img,
                metallicFactor=0.0,
                roughnessFactor=0.65,
                doubleSided=True,
            )
            tex_visual = trimesh.visual.TextureVisuals(uv=_uv_to_trimesh(_all_uvs), material=pbr_mat)
            mesh_final = trimesh.Trimesh(
                vertices=verts_final,
                faces=faces_final,
                vertex_normals=all_vnormals[:len(verts_final)],
                process=False,
            )
            mesh_final.visual = tex_visual
        except Exception as _tex_err:
            _log.warning("Textured GLB failed, falling back to untextured: %s", _tex_err, exc_info=True)
            mesh_final = trimesh.Trimesh(
                vertices=verts_final,
                faces=faces_final,
                vertex_normals=all_vnormals[:len(verts_final)],
                vertex_colors=_np.column_stack(
                    (colors_final, _np.full(len(colors_final), 255))
                ).astype(_np.uint8),
                process=False,
            )
    else:
        mesh_final = trimesh.Trimesh(
            vertices=verts_final,
            faces=faces_final,
            vertex_normals=all_vnormals[:len(verts_final)],
            vertex_colors=_np.column_stack(
                (colors_final, _np.full(len(colors_final), 255))
            ).astype(_np.uint8),
            process=False,
        )

    mesh_final.export(filepath)

    if is_glb:
        _fix_glb_material(filepath, metallic=0.0, roughness=0.65, double_sided=True)

    # OBJ companion files
    if filepath.lower().endswith(".obj") and use_vtx_color:
        try:
            _bn = _os.path.splitext(_os.path.basename(filepath))[0]
            _dir = _os.path.dirname(filepath)
            alb_grid = _crop_albedo[0:h:step, 0:w:step].astype(_np.uint8)
            if exp_bright != 1.0:
                alb_grid = _np.clip(
                    _np.power(_np.clip(alb_grid.astype(_np.float32) / 255.0, 0, 1),
                              1.0 / max(0.1, exp_bright)) * 255.0,
                    0, 255,
                ).astype(_np.uint8)
            _cv2.imwrite(_os.path.join(_dir, f"{_bn}_albedo.png"),
                         _cv2.cvtColor(alb_grid, _cv2.COLOR_RGB2BGR))

            n_obj = normals[y1:y2, x1:x2][0:h:step, 0:w:step].copy().astype(_np.float32)
            nm_obj = _np.round((n_obj * 0.5 + 0.5) * 255.0).clip(0, 255).astype(_np.uint8)
            if normal_flip_y:
                nm_obj = nm_obj.copy()
                nm_obj[..., 1] = 255 - nm_obj[..., 1]
            _cv2.imwrite(_os.path.join(_dir, f"{_bn}_normal.png"),
                         _cv2.cvtColor(nm_obj, _cv2.COLOR_RGB2BGR))

            with open(_os.path.join(_dir, f"{_bn}.mtl"), "w") as _m:
                _m.write(f"newmtl helio\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n")
                _m.write(f"map_Kd {_bn}_albedo.png\nnorm {_bn}_normal.png\n")

            with open(filepath, "w") as _of:
                _of.write(f"# Helio — Photometric Stereo 3D Export\n")
                _of.write(f"mtllib {_bn}.mtl\nusemtl helio\n")
                _CHUNK = 50000
                for _i0 in range(0, len(verts_final), _CHUNK):
                    _sl = slice(_i0, min(_i0 + _CHUNK, len(verts_final)))
                    _of.write("\n".join(
                        f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
                        for v in verts_final[_sl]) + "\n")
                _vn = all_vnormals[:len(verts_final)]
                for _i0 in range(0, len(_vn), _CHUNK):
                    _sl = slice(_i0, min(_i0 + _CHUNK, len(_vn)))
                    _of.write("\n".join(
                        f"vn {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
                        for v in _vn[_sl]) + "\n")
                _uvs_flip = _np.column_stack([_all_uvs[:, 0], 1.0 - _all_uvs[:, 1]])
                for _i0 in range(0, len(_uvs_flip), _CHUNK):
                    _sl = slice(_i0, min(_i0 + _CHUNK, len(_uvs_flip)))
                    _of.write("\n".join(
                        f"vt {u[0]:.6f} {u[1]:.6f}"
                        for u in _uvs_flip[_sl]) + "\n")
                _fi = faces_final.astype(_np.int32) + 1
                for _i0 in range(0, len(_fi), _CHUNK):
                    _sl = slice(_i0, min(_i0 + _CHUNK, len(_fi)))
                    _of.write("\n".join(
                        f"f {r[0]}/{r[0]}/{r[0]} {r[1]}/{r[1]}/{r[1]} {r[2]}/{r[2]}/{r[2]}"
                        for r in _fi[_sl]) + "\n")
        except Exception as _obj_err:
            _log.warning("OBJ UV rewrite warning: %s", _obj_err, exc_info=True)

    # Sketchfab PBR ZIP package — albedo + normal + roughness + AO +
    # displacement, all bundled with .obj + .mtl + sketchfab.json metadata.
    # Triggered by the "Sketchfab PBR Package (ZIP)" checkbox in the 3D
    # Exports tab.  Also works for non-Sketchfab use (any PBR viewer).
    if sketchfab_pbr:
        try:
            from helio.utils.file_utils import (
                _write_sketchfab_metadata, _write_obj_sketchfab_zip,
            )
            from helio.core.processing import (
                compute_ambient_occlusion, compute_depth_displacement,
            )
            if progress_cb:
                progress_cb("Building Sketchfab PBR ZIP...", 0.85)
            _dir = _os.path.dirname(filepath) or "."
            _bn = _os.path.splitext(_os.path.basename(filepath))[0]
            _zip_path = _os.path.join(_dir, f"{_bn}_sketchfab.zip")

            # Build PBR textures (downsampled to mesh grid for UV alignment)
            alb_grid = _crop_albedo[0:h:step, 0:w:step].astype(_np.uint8)
            if exp_bright != 1.0:
                alb_grid = _np.clip(
                    _np.power(_np.clip(alb_grid.astype(_np.float32) / 255.0, 0, 1),
                              1.0 / max(0.1, exp_bright)) * 255.0,
                    0, 255,
                ).astype(_np.uint8)
            alb_bgr = _cv2.cvtColor(alb_grid, _cv2.COLOR_RGB2BGR)

            n_tex = normals[y1:y2, x1:x2][0:h:step, 0:w:step].copy().astype(_np.float32)
            nm_rgb = _np.round((n_tex * 0.5 + 0.5) * 255.0).clip(0, 255).astype(_np.uint8)
            if normal_flip_y:
                nm_rgb = nm_rgb.copy()
                nm_rgb[..., 1] = 255 - nm_rgb[..., 1]
            nm_bgr = _cv2.cvtColor(nm_rgb, _cv2.COLOR_RGB2BGR)

            # Roughness from curvature (high curvature = rough)
            _curv = normal_divergence(n_tex[..., 0], n_tex[..., 1], sobel=False)
            _curv_sm = _cv2.GaussianBlur(_curv, (0, 0), 3.0)
            _rough = _np.clip(0.5 + _np.abs(_curv_sm) * 8.0, 0, 1)
            _rough_gray = (_rough * 255).astype(_np.uint8)

            # AO from PS normals
            _ao_arr = compute_ambient_occlusion(normals[y1:y2, x1:x2][0:h:step, 0:w:step])
            _ao_gray = (_ao_arr * 255).clip(0, 255).astype(_np.uint8)

            # Displacement (16-bit) from depth grid
            _disp = compute_depth_displacement(depth_grid)
            _disp_u16 = _cv2.normalize(_disp, None, 0, 65535, _cv2.NORM_MINMAX).astype(_np.uint16)

            obj_id = settings.get("obj_id", _bn)
            _write_sketchfab_metadata(_dir, obj_id,
                description=f"Photometric Stereo 3D export — Helio. Object: {obj_id}")
            _write_obj_sketchfab_zip(
                _zip_path, _bn,
                verts_final, faces_final,
                _all_uvs, all_vnormals[:len(verts_final)],
                alb_bgr, nm_bgr,
                roughness_gray=_rough_gray,
                ao_gray=_ao_gray,
                disp_u16=_disp_u16,
                jpeg=bool(settings.get("sketchfab_jpeg", False)))
            if progress_cb:
                progress_cb(f"Sketchfab ZIP saved: {_os.path.basename(_zip_path)}", 0.95)
        except Exception as _sf_err:
            _log.warning("Sketchfab PBR ZIP warning: %s", _sf_err, exc_info=True)

    if progress_cb:
        progress_cb(f"Done: {_os.path.basename(filepath)}", 1.0)
    return filepath


def export_3d_web_viewer(
    depth_vis_p: _np.ndarray,
    depth_normalised: Optional[_np.ndarray],
    normals: _np.ndarray,
    albedo_rgb: _np.ndarray,
    settings: dict,
    out_dir: str,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> dict[str, str]:
    """Generate a standalone HTML + GLB 3D web viewer.

    Writes ``document_model.glb`` (textured PBR mesh) and ``index.html``
    (Three.js viewer with raking light, controls, 360° sweep).

    Parameters
    ----------
    depth_vis_p : np.ndarray
        High-pass depth.
    depth_normalised : np.ndarray or None
        Full-range normalised depth.
    normals : np.ndarray
        PS normal map (H, W, 3).
    albedo_rgb : np.ndarray
        RGB albedo (H, W, 3), uint8.
    settings : dict
        Keys: ``step``, ``height``, ``fx``, ``fy``, ``fz``, ``shape_blend``,
        ``hp_sigma``, ``smooth``, ``warp_rem``, ``micro``, ``edge_trim``,
        ``center_origin``, ``vertex_colors``, ``export_brightness``,
        ``normal_flip_y``, ``current_engine``, ``crop``.
    out_dir : str
        Output directory.
    progress_cb : callable or None

    Returns
    -------
    dict[str, str]
        ``{"glb": ..., "html": ...}`` paths written.
    """
    import trimesh

    x1, y1, x2, y2 = settings.get("crop", (0, 0, albedo_rgb.shape[1], albedo_rgb.shape[0]))
    step = settings.get("step", 2)
    fx = settings.get("fx", 1)
    fy = settings.get("fy", -1)
    fz = settings.get("fz", 1)
    current_engine = settings.get("current_engine", "")
    analytical_engines = {
        "SLRM (Micro-Relief)", "Surface Curvature (Scratches)",
        "Positive Openness", "Negative Openness", "Local Dominance",
        "Normals Unsharp Masking", "Slope Magnitude", "Specular Enhancement",
        "Image Quotient (IQ)", "PS Normal-Map Lit (Single)",
        "PS Normal-Map Lit (8-Dir)", "Sky-View Factor (SVF)",
        "Slope Gradient",
        "Multi-Scale Relief Model (MSRM)",
    }

    if progress_cb:
        progress_cb("Building depth grid...", 0.0)

    depth_grid = compute_export_depth_grid(
        depth_vis_p[y1:y2, x1:x2],
        depth_normalised[y1:y2, x1:x2] if depth_normalised is not None else None,
        None,  # web viewer uses standard path only
        {**settings, "use_hd": False, "use_hybrid": False},
        progress_cb,
    )
    h, w = depth_grid.shape
    _crop_albedo = albedo_rgb[y1:y2, x1:x2]

    # Vertex-color fallback (build_grid_mesh) samples albedo at the *mesh*
    # grid's own resolution — resample to (h, w) rather than slicing the
    # full-res crop with those (much smaller) indices, which used to grab
    # only the top-left corner of the image instead of the whole extent.
    _mesh_albedo = (_crop_albedo if _crop_albedo.shape[:2] == (h, w)
                     else _cv2.resize(_crop_albedo, (w, h), interpolation=_cv2.INTER_AREA))

    if progress_cb:
        progress_cb("Building mesh...", 0.3)

    verts, faces, colors, uvs = build_grid_mesh(
        depth_grid,
        {**settings, "albedo": _mesh_albedo},
        progress_cb,
    )

    if progress_cb:
        progress_cb("Building GLB...", 0.6)

    # The PBR texture is sampled per-pixel by the GPU, independent of vertex
    # density, so give it a higher resolution than the (decimated) mesh grid
    # for a sharp result without inflating vertex/triangle count.
    #
    # Albedo and normal map are NOT resized the same way: a normal map
    # encodes a *direction* per pixel, and cv2.resize's area-averaging blends
    # neighbouring directions into a shorter, blunter vector — fine tool
    # marks/engraving detail (the whole point of this viewer) gets washed
    # out even at a nominally "high" resolution, because what matters is the
    # ratio of feature size to texture resolution, not the absolute pixel
    # count. Albedo tolerates the same blur fine (it carries colour, not
    # per-pixel lighting response), so it keeps the tighter cap; the normal
    # map gets a much more generous one and is renormalised after resizing.
    tex_res = int(settings.get("tex_res", ALBEDO_TEX_RES))
    normal_tex_res = int(settings.get("normal_tex_res", NORMAL_TEX_RES))
    th, tw = _crop_albedo.shape[:2]
    _tscale = min(1.0, tex_res / max(th, tw))
    tgH, tgW = max(4, int(th * _tscale)), max(4, int(tw * _tscale))
    _nscale = min(1.0, normal_tex_res / max(th, tw))
    ngH, ngW = max(4, int(th * _nscale)), max(4, int(tw * _nscale))
    _ncrop = normals[y1:y2, x1:x2]
    alb_grid = (_cv2.resize(_crop_albedo, (tgW, tgH), interpolation=_cv2.INTER_AREA)
                if _tscale < 1.0 else _crop_albedo).astype(_np.uint8)
    if _nscale < 1.0:
        n_web = _cv2.resize(_ncrop, (ngW, ngH), interpolation=_cv2.INTER_AREA).astype(_np.float32)
        _nlen = _np.linalg.norm(n_web, axis=-1, keepdims=True)
        n_web = n_web / _np.maximum(_nlen, 1e-6)
    else:
        n_web = _ncrop.astype(_np.float32)
    albedo_img = _Image.fromarray(alb_grid)

    nm_web = _np.round((n_web * 0.5 + 0.5) * 255.0).clip(0, 255).astype(_np.uint8)
    if settings.get("normal_flip_y", False):
        nm_web = nm_web.copy()
        nm_web[..., 1] = 255 - nm_web[..., 1]
    normal_img = _Image.fromarray(nm_web)

    glb_path = _os.path.join(out_dir, "document_model.glb")
    try:
        pbr_mat = trimesh.visual.material.PBRMaterial(
            name="helio_pbr",
            baseColorTexture=albedo_img,
            baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            normalTexture=normal_img,
            metallicFactor=0.0,
            roughnessFactor=0.65,
            doubleSided=True,
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=_uv_to_trimesh(uvs), material=pbr_mat)
    except Exception:
        _log.warning("PBR textured mesh failed, falling back to vertex colours")
        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces,
            vertex_colors=_np.column_stack(
                (colors, _np.full(colors.shape[0], 255))
            ).astype(_np.uint8),
            process=False,
        )
    mesh.export(glb_path)
    _fix_glb_material(glb_path, metallic=0.0, roughness=0.65, double_sided=True)
    if settings.get("tex_jpeg"):
        try:
            _glb_textures_jpeg(glb_path, int(settings["tex_jpeg"]))
        except Exception:
            _log.warning("GLB textures kept as PNG", exc_info=True)

    if progress_cb:
        progress_cb("Writing HTML...", 0.9)

    # Embed the GLB as a base64 data URI so the HTML is fully self-contained
    # — opens from file:// without CORS blocking the GLB fetch.
    glb_b64 = _b64.b64encode(open(glb_path, "rb").read()).decode("ascii")
    html_doc = _build_web_viewer_html(glb_b64=glb_b64, name=settings.get("obj_id", ""),
                                      look=settings.get("viewer_look"),
                                      tilt=settings.get("viewer_tilt", 0.0))
    html_doc = _phone_3d(html_doc)
    html_path = _os.path.join(out_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    if progress_cb:
        progress_cb("Done", 1.0)
    return {"glb": glb_path, "html": html_path}


def _phone_3d(html_doc: str) -> str:
    """The 3D viewers on a phone: the controls panel as a sheet along the
    bottom (the model turns with one finger and zooms with two, as it
    always has), its own Controls button and the keyboard rows hidden."""
    from helio.application.phone_web import phone_ready
    return phone_ready(html_doc, panel="#panel", hide=("#toggle", ".krow"),
                       tip="One finger turns the model \u00b7 two fingers zoom and pan")


def _build_web_viewer_html(glb_b64: str = "", name: str = "", look: dict | None = None,
                           tilt: float = 0.0) -> str:
    """Return the Three.js embedded HTML template for the standard web viewer.

    *look* sets sliders' starting values by id ({"s_exp": 0.9, ...}): the page
    opens with them, as if the reader had moved them.  *tilt* is the opening
    view's angle from overhead, from the object's front (F and R come back to it).

    When ``glb_b64`` is provided, the GLB is embedded as a base64 data URI
    so the viewer opens from file:// without CORS blocking the GLB fetch.
    """
    _glb_uri = f"data:model/gltf-binary;base64,{glb_b64}" if glb_b64 else "document_model.glb"
    # three.js is inlined rather than fetched from jsdelivr. The page already
    # said "The viewer is self-contained" in its own error message while
    # loading three libraries over the network — true of the model, false of
    # the viewer. It is now true of both.
    from helio.vendor_assets import three_js_bundle
    _three_js = three_js_bundle()
    _html = """<!DOCTYPE html>
<html><head><title>Helio | 3D Surface Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{overflow:hidden;background:#111;color:#eee;font-family:'Segoe UI',sans-serif;}
#c{position:fixed;top:0;left:0;width:100%;height:100%;}
#panel{position:fixed;top:0;right:0;width:272px;height:100%;background:rgba(13,13,18,0.94);
  border-left:1px solid #2a2a3a;padding:14px 12px;overflow-y:auto;z-index:10;
  transition:transform 0.25s;}
#panel.hidden{transform:translateX(272px);}
#toggle{position:fixed;top:12px;right:282px;z-index:11;background:#1a1a2e;border:1px solid #444;
  color:#ccc;padding:6px 10px;border-radius:6px;cursor:pointer;font-size:12px;transition:right 0.25s;}
#toggle.hidden{right:10px;}
h2{font-size:12px;color:#f39c12;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;
  padding-bottom:4px;border-bottom:1px solid #2a2a3a;}
.row{margin-bottom:9px;}
label{display:block;font-size:11px;color:#aaa;margin-bottom:3px;}
input[type=range]{width:100%;accent-color:#f39c12;cursor:pointer;}
.val{display:inline-block;font-size:11px;color:#f39c12;float:right;}
.hint{font-size:10px;color:#888;margin-top:8px;line-height:1.6;}
hr{border:none;border-top:1px solid #222230;margin:9px 0;}
.btn-row{display:flex;gap:5px;margin-bottom:8px;}
.hbtn{flex:1;padding:5px 4px;border:1px solid #2a3a4a;background:#0d1a26;color:#7ec8e3;
  font-size:11px;border-radius:5px;cursor:pointer;transition:background 0.15s;}
.hbtn:hover{background:#1a3a50;}
.hbtn.active{background:#1a5276;border-color:#5dade2;color:#fff;}
kbd{display:inline-block;background:#1a1a2e;border:1px solid #444;border-radius:3px;padding:1px 5px;font-size:10px;font-family:inherit;color:#f39c12;min-width:14px;text-align:center;}
.krow{display:flex;justify-content:space-between;font-size:10px;color:#888;margin-bottom:4px;}
#dte-overlay{display:none;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  background:rgba(10,10,20,0.96);border:1px solid #1a5276;border-radius:10px;
  padding:18px 24px;z-index:30;text-align:center;min-width:260px;}
#dte-overlay h3{color:#5dade2;font-size:14px;margin-bottom:10px;}
#dte-overlay p{font-size:11px;color:#aaa;margin-bottom:14px;line-height:1.6;}
#dte-cancel{padding:6px 20px;background:#333;border:none;color:#ccc;border-radius:5px;cursor:pointer;}
#err{display:none;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  background:#c0392b;padding:20px 30px;border-radius:8px;font-size:16px;text-align:center;z-index:20;}
#objname{position:fixed;top:10px;left:10px;max-width:46vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  background:rgba(13,20,22,0.72);color:#dfe9e6;border:1px solid #2a3a3a;padding:4px 10px;border-radius:12px;
  font-size:12px;font-weight:600;z-index:15;pointer-events:none;}
#objname:empty{display:none;}
</style></head>
<body>
<canvas id="c"></canvas>
<div id="objname">__OBJ_NAME__</div>
<button id="toggle" onclick="togglePanel()">&#9664; Controls</button>
<div id="panel">
  <h2>Lighting</h2>
  <div class="row"><label>Key Light <span class="val" id="v_key">2.2</span></label>
    <input type="range" id="s_key" min="0" max="6" step="0.1" value="2.2"></div>
  <div class="row"><label>Fill Light <span class="val" id="v_fill">0.9</span></label>
    <input type="range" id="s_fill" min="0" max="3" step="0.05" value="0.9"></div>
  <div class="row"><label>Ambient <span class="val" id="v_amb">1.4</span></label>
    <input type="range" id="s_amb" min="0" max="4" step="0.1" value="1.4"></div>
  <div class="row"><label>Raking Light <span class="val" id="v_rake">2.0</span></label>
    <input type="range" id="s_rake" min="0" max="6" step="0.1" value="2.0"></div>
  <div class="row"><label>Raking Elevation <span class="val" id="v_elev">45</span>&deg;</label>
    <input type="range" id="s_elev" min="1" max="89" step="1" value="45"></div>
  <div class="row"><label>Sweep Speed <span class="val" id="v_spd">1.0</span>&deg;/s</label>
    <input type="range" id="s_spd" min="0.2" max="8" step="0.2" value="1.0"></div>
  <hr>
  <h2>&#9654; Animation</h2>
  <div class="btn-row">
    <button class="hbtn" id="btn360" onclick="toggle360()">&#8635; 360&deg; Sweep</button>
    <button class="hbtn" id="btnDTE" onclick="startDTE()">Deep Time</button>
  </div>
  <div id="dte-status" style="font-size:12px;color:#7CC0EE;min-height:14px;margin-bottom:4px;"></div>
  <hr>
  <h2>Render</h2>
  <div class="row"><label>Exposure <span class="val" id="v_exp">1.1</span></label>
    <input type="range" id="s_exp" min="0.1" max="4" step="0.05" value="1.1"></div>
  <div class="row"><label>Roughness <span class="val" id="v_rough">0.65</span></label>
    <input type="range" id="s_rough" min="0" max="1" step="0.01" value="0.65"></div>
  <div class="row"><label>Metalness <span class="val" id="v_metal">0.0</span></label>
    <input type="range" id="s_metal" min="0" max="1" step="0.01" value="0.0"></div>
  <hr>
  <h2>Light Compass</h2>
  <div style="text-align:center;margin-bottom:6px;">
    <canvas id="compass" width="90" height="90" style="cursor:crosshair;border-radius:50%;"></canvas>
  </div>
  <div class="hint" style="margin-top:0;text-align:center;">Drag the dome to set light direction</div>
  <div class="row"><label>Background <span class="val" id="v_bg">0.1</span></label>
    <input type="range" id="s_bg" min="0" max="1" step="0.01" value="0.1"></div>
  <hr>
  <h2>Shortcuts</h2>
  <div class="krow"><span><kbd>R</kbd> Reset</span><span><kbd>S</kbd> Screenshot</span></div>
  <div class="krow"><span><kbd>F</kbd> Fit view</span><span><kbd>L</kbd> 360&deg; Sweep</span></div>
  <div class="krow"><span><kbd>&larr;</kbd><kbd>&rarr;</kbd> Azimuth</span><span><kbd>&uarr;</kbd><kbd>&darr;</kbd> Elevation</span></div>
  <div class="krow"><span><kbd>D</kbd> Deep Time</span><span></span></div>
  <div class="hint" style="margin-top:4px;">
    <kbd>LMB</kbd> Rotate &nbsp; <kbd>Space</kbd>/<kbd>&#8984;</kbd>+Drag Pan &nbsp; <kbd>RMB</kbd> Pan &nbsp; <kbd>Scroll</kbd> Zoom<br>
    <kbd>&#8679;</kbd>+Drag Sweep light
  </div>
</div>

<div id="dte-overlay">
  <h3>Deep Time Explore</h3>
  <p>Slowly sweeps raking light through 8 cardinal angles<br>
  at low elevation to maximise surface relief visibility.<br>
  Inspired by archaeological multi-directional RTI.</p>
  <button id="dte-cancel" onclick="stopDTE()">Stop</button>
</div>

<div id="err">Failed to load 3D model.<br>The viewer is self-contained — if you see this, the embedded GLB may be corrupted. Try re-exporting.</div>
<script>/*__THREE_JS__*/</script>
<script>
const canvas=document.getElementById('c');
const scene=new THREE.Scene();
const bgCol=new THREE.Color(0x1e1e1e);
scene.background=bgCol;
const camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.1,50000);
const renderer=new THREE.WebGLRenderer({antialias:true,canvas,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.outputEncoding=THREE.sRGBEncoding;
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=1.1;

const hemiLight=new THREE.HemisphereLight(0xfff0e0,0x203050,1.4);
scene.add(hemiLight);
const keyLight=new THREE.DirectionalLight(0xfff8f0,2.2);
keyLight.position.set(2,3,4); scene.add(keyLight);
const fillLight=new THREE.DirectionalLight(0xe8f4ff,0.9);
fillLight.position.set(-3,0,2); scene.add(fillLight);
const dirLight=new THREE.DirectionalLight(0xffffff,2.0);
dirLight.position.set(0,0,200); scene.add(dirLight);

const controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.enableDamping=true; controls.dampingFactor=0.06;

let loadedModel=null, modelCenter=new THREE.Vector3(), modelSize=1;
/* The opening view: straight from the front (the object's bottom edge nearest),
   tilted VIEW_TILT degrees from overhead, centred; 0 looks straight down. */
const VIEW_TILT=__VIEW_TILT__;
function homeView(){
  const box=loadedModel?new THREE.Box3().setFromObject(loadedModel):null;
  const c=box?box.getCenter(new THREE.Vector3()):modelCenter.clone();
  const s=box?box.getSize(new THREE.Vector3()).length():modelSize;
  const t=VIEW_TILT*Math.PI/180, d=s*(VIEW_TILT?1.05:1.2);
  controls.target.copy(c);
  camera.position.set(c.x,c.y-Math.sin(t)*d,c.z+Math.cos(t)*d);
  camera.lookAt(c);
  controls.update();
}
/* Pan: Space + drag, or Cmd / Ctrl + drag (OrbitControls), or the right button. */
window.addEventListener('keydown',e=>{if(e.code==='Space'&&!e.repeat){
  controls.mouseButtons.LEFT=THREE.MOUSE.PAN;renderer.domElement.style.cursor='grab';e.preventDefault();}});
window.addEventListener('keyup',e=>{if(e.code==='Space'){
  controls.mouseButtons.LEFT=THREE.MOUSE.ROTATE;renderer.domElement.style.cursor='';}});
new THREE.GLTFLoader().load('document_model.glb',gltf=>{
  loadedModel=gltf.scene;
  loadedModel.traverse(c=>{
    if(!c.isMesh) return;
    const hasTex = c.material && c.material.map;
    if(!hasTex) c.material.vertexColors = true;
    c.material.roughness = 0.65;
    c.material.metalness = 0.0;
    c.material.side = THREE.DoubleSide;
    c.material.needsUpdate = true;
  });
  scene.add(loadedModel);
  const box=new THREE.Box3().setFromObject(loadedModel);
  modelSize=box.getSize(new THREE.Vector3()).length();
  box.getCenter(modelCenter);
  homeView();
  controls.saveState();                        /* R comes back here */
  updateRakingLight();
},undefined,()=>{document.getElementById('err').style.display='block';});

window.addEventListener('resize',()=>{
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
});

function sl(id,valId,cb){
  const el=document.getElementById(id);
  el.addEventListener('input',()=>{document.getElementById(valId).textContent=parseFloat(el.value).toFixed(2);cb(parseFloat(el.value));});
}
sl('s_key','v_key',v=>{keyLight.intensity=v;});
sl('s_fill','v_fill',v=>{fillLight.intensity=v;});
sl('s_amb','v_amb',v=>{hemiLight.intensity=v;});
sl('s_rake','v_rake',v=>{dirLight.intensity=v;});
sl('s_exp','v_exp',v=>{renderer.toneMappingExposure=v;});
sl('s_rough','v_rough',v=>{if(loadedModel)loadedModel.traverse(c=>{if(c.isMesh){c.material.roughness=v;c.material.needsUpdate=true;}});});
sl('s_metal','v_metal',v=>{if(loadedModel)loadedModel.traverse(c=>{if(c.isMesh){c.material.metalness=v;c.material.needsUpdate=true;}});});
sl('s_bg','v_bg',v=>{bgCol.setRGB(v*0.13,v*0.13,v*0.13);scene.background=bgCol;});
sl('s_spd','v_spd',v=>{ sweepSpeed=v; });

let panelOpen=true;
function togglePanel(){
  panelOpen=!panelOpen;
  document.getElementById('panel').classList.toggle('hidden',!panelOpen);
  const t=document.getElementById('toggle');
  t.classList.toggle('hidden',!panelOpen);
  t.innerHTML=panelOpen?'&#9664; Controls':'&#9654; Controls';
}

let lightAz=0, lightEl=45;
function updateRakingLight(){
  const phi=THREE.MathUtils.degToRad(lightEl);
  const az=THREE.MathUtils.degToRad(lightAz);
  const dist=300;
  dirLight.position.set(
    dist*Math.cos(phi)*Math.cos(az),
    dist*Math.cos(phi)*Math.sin(az),
    dist*Math.sin(phi));
  dirLight.lookAt(modelCenter);
  document.getElementById('s_elev').value=lightEl;
  document.getElementById('v_elev').textContent=Math.round(lightEl);
  drawDome();
}
document.getElementById('s_elev').addEventListener('input',function(){
  lightEl=parseFloat(this.value);
  document.getElementById('v_elev').textContent=Math.round(lightEl);
  updateRakingLight();
});

// ── Light Compass — draggable virtual dome (ported from the WebRTI viewer) ──
const domeCv=document.getElementById('compass');
const domeCtx=domeCv.getContext('2d');
const domeR=domeCv.width/2;
let domeDrag=false;
function drawDome(){
  const azR=THREE.MathUtils.degToRad(lightAz), elR=THREE.MathUtils.degToRad(lightEl);
  const dlx=Math.cos(elR)*Math.sin(azR), dly=Math.cos(elR)*Math.cos(azR);
  domeCtx.clearRect(0,0,domeCv.width,domeCv.height);
  const g=domeCtx.createRadialGradient(domeR,domeR,0,domeR,domeR,domeR);
  g.addColorStop(0,'#1A1B1E'); g.addColorStop(1,'#1A1B1E');
  domeCtx.fillStyle=g; domeCtx.beginPath(); domeCtx.arc(domeR,domeR,domeR,0,Math.PI*2); domeCtx.fill();
  domeCtx.strokeStyle='#3E4046'; domeCtx.lineWidth=1;
  [0.33,0.67,1].forEach(r=>{domeCtx.beginPath();domeCtx.arc(domeR,domeR,r*domeR,0,Math.PI*2);domeCtx.stroke();});
  for(let i=0;i<8;i++){const a=i*Math.PI/4;domeCtx.beginPath();domeCtx.moveTo(domeR,domeR);domeCtx.lineTo(domeR+Math.cos(a)*domeR,domeR+Math.sin(a)*domeR);domeCtx.stroke();}
  const m=Math.sqrt(dlx*dlx+dly*dly), sc=m>0.95?0.95/m:1;
  const px=domeR+dlx*domeR*sc, py=domeR-dly*domeR*sc;
  domeCtx.beginPath(); domeCtx.arc(px,py,6,0,Math.PI*2);
  domeCtx.fillStyle='#E6A93E'; domeCtx.fill(); domeCtx.strokeStyle='#222326'; domeCtx.lineWidth=2; domeCtx.stroke();
}
function domeXY(px,py){
  const dlx=(px-domeR)/domeR, dly=-(py-domeR)/domeR;
  const r=Math.min(1,Math.sqrt(dlx*dlx+dly*dly));
  const elR=Math.max(3*Math.PI/180,Math.min(89*Math.PI/180,Math.acos(r)));
  const azR=Math.atan2(dlx,dly);
  lightAz=((azR*180/Math.PI)%360+360)%360;
  lightEl=elR*180/Math.PI;
  updateRakingLight();
}
domeCv.addEventListener('mousedown',e=>{e.preventDefault();domeDrag=true;domeXY(e.offsetX,e.offsetY);});
window.addEventListener('mousemove',e=>{if(!domeDrag)return;const r=domeCv.getBoundingClientRect();domeXY(e.clientX-r.left,e.clientY-r.top);});
window.addEventListener('mouseup',()=>domeDrag=false);
domeCv.addEventListener('touchstart',e=>{e.preventDefault();domeDrag=true;const r=domeCv.getBoundingClientRect(),t=e.touches[0];domeXY(t.clientX-r.left,t.clientY-r.top);},{passive:false});
domeCv.addEventListener('touchmove',e=>{e.preventDefault();const r=domeCv.getBoundingClientRect(),t=e.touches[0];domeXY(t.clientX-r.left,t.clientY-r.top);},{passive:false});
domeCv.addEventListener('touchend',()=>domeDrag=false);

let sweeping360=false, sweepSpeed=1.0, lastSweepT=null;
function toggle360(){
  sweeping360=!sweeping360;
  const b=document.getElementById('btn360');
  b.classList.toggle('active',sweeping360);
  b.textContent=sweeping360?'⏹ Stop Sweep':'⮔ 360° Sweep';
  if(sweeping360){lastSweepT=null; stopDTE();}
}

let dteActive=false, dteStage=0, dtePaused=0;
const DTE_AZIMUTHS=[0,45,90,135,180,225,270,315];
const DTE_ELEV=22;
const DTE_DWELL_MS=2200;
let dteSavedAz=0, dteSavedEl=45;
let dteTransT=null, dteStartAz=0, dteTargetAz=0, dteTransDur=900;

function startDTE(){
  if(dteActive){stopDTE();return;}
  dteSavedAz=lightAz; dteSavedEl=lightEl;
  dteActive=true; dteStage=0; dtePaused=0; dteTransT=null;
  sweeping360=false;
  document.getElementById('btn360').classList.remove('active');
  document.getElementById('btn360').textContent='⮔ 360° Sweep';
  document.getElementById('dte-overlay').style.display='block';
  document.getElementById('dte-status').textContent='Deep Time: stage 1/8';
  dteGotoStage(0);
}
function stopDTE(){
  if(!dteActive) return;
  dteActive=false;
  document.getElementById('dte-overlay').style.display='none';
  document.getElementById('dte-status').textContent='';
  lightAz=dteSavedAz; lightEl=dteSavedEl;
  updateRakingLight();
}
function dteGotoStage(s){
  dteStage=s;
  dteStartAz=lightAz;
  dteTargetAz=DTE_AZIMUTHS[s];
  lightEl=DTE_ELEV;
  dtePaused=DTE_DWELL_MS;
  dteTransT=null;
  document.getElementById('dte-status').textContent='Deep Time: angle '+(s+1)+'/8 — '+DTE_AZIMUTHS[s]+'°';
}

let prevT=null;
(function animate(ts){
  requestAnimationFrame(animate);
  controls.update();
  const dt=prevT!=null?Math.min(ts-prevT,80):0; prevT=ts;
  if(sweeping360){
    lightAz=(lightAz+sweepSpeed*(dt/1000)*360)%360;
    updateRakingLight();
  }
  if(dteActive){
    if(dteTransT===null){ dteTransT=0; }
    if(dteTransT<dteTransDur){
      dteTransT+=dt;
      const t=Math.min(dteTransT/dteTransDur,1);
      const ease=t<0.5?2*t*t:1-Math.pow(-2*t+2,2)/2;
      lightAz=dteStartAz+(dteTargetAz-dteStartAz)*ease;
      updateRakingLight();
    } else {
      lightAz=dteTargetAz;
      updateRakingLight();
      dtePaused-=dt;
      if(dtePaused<=0){
        const next=(dteStage+1)%DTE_AZIMUTHS.length;
        if(next===0){ stopDTE(); }
        else { dteGotoStage(next); }
      }
    }
  }
  renderer.render(scene,camera);
})();

function takeScreenshot(){
  renderer.render(scene,camera);
  const nm=(document.getElementById('objname').textContent||'relief').trim().replace(/[^\\w.-]+/g,'_')||'relief';
  canvas.toBlob(b=>{
    const a=document.createElement('a');
    a.href=URL.createObjectURL(b);
    a.download=nm+'_'+new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')+'.png';
    a.click();
  });
}
let shiftDown=false,isDragging=false,dragStartX=0,dragStartY=0;
window.addEventListener('keydown',e=>{
  if(e.key==='Shift'){shiftDown=true;controls.enabled=false;}
  if(e.key==='r'||e.key==='R') controls.reset();
  if(e.key==='f'||e.key==='F'){
    if(loadedModel) homeView();
  }
  if(e.key==='l'||e.key==='L') toggle360();
  if(e.key==='d'||e.key==='D') startDTE();
  if(e.key==='s'||e.key==='S'){e.preventDefault();takeScreenshot();}
  if(e.key==='ArrowLeft'){lightAz=(lightAz-5)%360;updateRakingLight();}
  if(e.key==='ArrowRight'){lightAz=(lightAz+5)%360;updateRakingLight();}
  if(e.key==='ArrowUp'){lightEl=Math.min(89,lightEl+5);updateRakingLight();}
  if(e.key==='ArrowDown'){lightEl=Math.max(1,lightEl-5);updateRakingLight();}
});
window.addEventListener('keyup',e=>{if(e.key==='Shift'){shiftDown=false;isDragging=false;controls.enabled=true;}});
window.addEventListener('mousedown',e=>{if(shiftDown){isDragging=true;dragStartX=e.clientX;dragStartY=e.clientY;}});
window.addEventListener('mouseup',()=>{isDragging=false;});
window.addEventListener('mousemove',e=>{
  if(isDragging){
    const dx=e.clientX-dragStartX, dy=e.clientY-dragStartY;
    dragStartX=e.clientX; dragStartY=e.clientY;
    lightAz=(lightAz+dx*0.5)%360;
    lightEl=Math.max(1,Math.min(89,lightEl-dy*0.3));
    sweeping360=false; dteActive=false;
    document.getElementById('btn360').classList.remove('active');
    document.getElementById('btn360').textContent='⮔ 360° Sweep';
    document.getElementById('dte-overlay').style.display='none';
    updateRakingLight();
  }
});
updateRakingLight();
</script></body></html>"""
    _html = _html.replace("'document_model.glb'", f"'{_glb_uri}'")
    _html = _html.replace("__VIEW_TILT__", f"{max(0.0, min(80.0, float(tilt or 0.0))):g}")
    if look:
        import json as _js
        _set = ("<script>(function(L){for(var k in L){var e=document.getElementById(k);"
                "if(e){e.value=L[k];e.dispatchEvent(new Event('input'));}}})("
                + _js.dumps({str(k): float(v) for k, v in look.items()}) + ");</script>")
        _html = _html.replace("</body></html>", _set + "</body></html>", 1)
    # Injected last: the replaces above search the whole document, and there is
    # no reason to make them walk 726 KB of minified three.js.
    _html = _html.replace("/*__THREE_JS__*/", _three_js)
    return _html.replace("__OBJ_NAME__", __import__("html").escape(name or ""))


def export_3d_web_viewer_hf_normals(
    normals: _np.ndarray,
    albedo_rgb: _np.ndarray,
    settings: dict,
    out_dir: str,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> dict[str, str]:
    """Generate an HF+normals 3D web viewer with embedded GLB + normal map.

    Uses Frankot–Chellappa integration + aggressive HP filter for the
    geometry, and exports a separate PS normal-map PNG that the Three.js
    viewer loads as a per-pixel normal texture.

    Parameters
    ----------
    normals : np.ndarray
        PS normal map (H, W, 3), float32.
    albedo_rgb : np.ndarray
        RGB albedo (H, W, 3), uint8.
    settings : dict
        Keys: ``step``, ``height``, ``fx``, ``fy``, ``fz``, ``smooth``,
        ``micro``, ``edge_trim``, ``center_origin``, ``normal_flip_y``,
        ``hf_cutoff``, ``max_res``, ``crop``.
    out_dir : str
        Output directory.
    progress_cb : callable or None

    Returns
    -------
    dict[str, str]
        ``{"glb": ..., "normal_map": ..., "html": ...}`` paths written.
    """
    import trimesh

    x1, y1, x2, y2 = settings.get("crop", (0, 0, albedo_rgb.shape[1], albedo_rgb.shape[0]))
    step = settings.get("step", 2)
    fx = settings.get("fx", 1)
    fy = settings.get("fy", -1)
    fz = settings.get("fz", 1)
    normal_flip_y = settings.get("normal_flip_y", False)
    center_origin = settings.get("center_origin", True)
    hf_cutoff = settings.get("hf_cutoff", 0.35)
    max_res = settings.get("max_res", MESH_MAX_RES)

    if progress_cb:
        progress_cb("Computing HF depth from normals...", 0.0)

    depth_grid = compute_hf_depth_from_normals(
        normals[y1:y2, x1:x2], hf_cutoff, max_res, progress_cb,
    )
    h_d, w_d = depth_grid.shape
    _crop_albedo = albedo_rgb[y1:y2, x1:x2]
    _crop_normals = normals[y1:y2, x1:x2]

    # Vertex-color fallback samples albedo at the mesh grid's own
    # resolution — resample rather than slicing the full-res crop with
    # those (much smaller) indices, which used to grab only the top-left
    # corner of the image instead of the whole extent.
    _mesh_albedo = (_crop_albedo if _crop_albedo.shape[:2] == (h_d, w_d)
                     else _cv2.resize(_crop_albedo, (w_d, h_d), interpolation=_cv2.INTER_AREA))

    if progress_cb:
        progress_cb("Building mesh...", 0.3)

    verts, faces, colors, uvs = build_grid_mesh(
        depth_grid,
        {**settings, "albedo": _mesh_albedo},
        progress_cb,
    )

    if progress_cb:
        progress_cb("Building GLB with PBR textures...", 0.6)

    # PBR texture is GPU-sampled per-pixel, independent of vertex density —
    # give it a higher resolution than the (decimated) mesh grid.
    #
    # Albedo and normal map are NOT resized the same way — see the matching
    # comment in export_3d_web_viewer. A normal map is a per-pixel direction;
    # cv2.resize's area-averaging blunts fine engraving/tool-mark detail even
    # at a nominally high resolution, because what matters is feature size
    # relative to texture resolution, not the absolute pixel count. This is
    # the HF+Normals viewer specifically — normal-map detail IS the point of
    # this mode — so it gets a much more generous cap and is renormalised
    # after resizing.
    tex_res = int(settings.get("tex_res", ALBEDO_TEX_RES))
    normal_tex_res = int(settings.get("normal_tex_res", NORMAL_TEX_RES))
    th, tw = _crop_albedo.shape[:2]
    _tscale = min(1.0, tex_res / max(th, tw))
    tgH, tgW = max(4, int(th * _tscale)), max(4, int(tw * _tscale))
    _nscale = min(1.0, normal_tex_res / max(th, tw))
    ngH, ngW = max(4, int(th * _nscale)), max(4, int(tw * _nscale))
    alb_grid = (_cv2.resize(_crop_albedo, (tgW, tgH), interpolation=_cv2.INTER_AREA)
                if _tscale < 1.0 else _crop_albedo).astype(_np.uint8)
    if _nscale < 1.0:
        nm = _cv2.resize(_crop_normals, (ngW, ngH), interpolation=_cv2.INTER_AREA).astype(_np.float32)
        _nlen = _np.linalg.norm(nm, axis=-1, keepdims=True)
        nm = nm / _np.maximum(_nlen, 1e-6)
    else:
        nm = _crop_normals.astype(_np.float32)
    alb_img = _Image.fromarray(alb_grid)

    nm_rgb = _np.round((nm * 0.5 + 0.5) * 255.0).clip(0, 255).astype(_np.uint8)
    if normal_flip_y:
        nm_rgb = nm_rgb.copy()
        nm_rgb[..., 1] = 255 - nm_rgb[..., 1]
    nm_img = _Image.fromarray(nm_rgb)

    glb_path = _os.path.join(out_dir, "hf_normals_model.glb")
    try:
        pbr = trimesh.visual.material.PBRMaterial(
            name="helio_pbr",
            baseColorTexture=alb_img,
            baseColorFactor=[1, 1, 1, 1],
            normalTexture=nm_img,
            metallicFactor=0.0,
            roughnessFactor=0.65,
            doubleSided=True,
        )
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual = trimesh.visual.TextureVisuals(uv=_uv_to_trimesh(uvs), material=pbr)
    except Exception:
        _log.warning("PBR textured mesh export failed, using vertex colours")
        mesh = trimesh.Trimesh(
            vertices=verts, faces=faces,
            vertex_colors=_np.column_stack(
                (colors, _np.full(len(colors), 255, dtype=_np.uint8))
            ).astype(_np.uint8),
            process=False,
        )
    mesh.export(glb_path)
    _fix_glb_material(glb_path, metallic=0.0, roughness=0.65, double_sided=True)

    if progress_cb:
        progress_cb("Writing normal map PNG...", 0.8)

    n_export = normals[y1:y2, x1:x2].copy().astype(_np.float32)
    nm_png = ((n_export * 0.5 + 0.5) * 255).clip(0, 255).astype(_np.uint8)
    # Apply the same green-channel flip as the GLB-embedded normal texture.
    # The viewer JS recomputes UVs with v=0 at the mesh top and calls
    # computeTangents(), so the bitangent points image-DOWN (-Y world);
    # the G channel must therefore encode -ny, exactly like the embedded
    # path above. This PNG used to be written raw (G=+ny), which inverted
    # the Y-relief response — combined with the flipY sampling mismatch
    # (fixed in the JS below) it produced the ghosted/relief-backwards look.
    if normal_flip_y:
        nm_png = nm_png.copy()
        nm_png[..., 1] = 255 - nm_png[..., 1]
    nm_bgr = _cv2.cvtColor(nm_png, _cv2.COLOR_RGB2BGR)
    nm_path = _os.path.join(out_dir, "hf_normal_map.png")
    _cv2.imwrite(nm_path, nm_bgr)

    if progress_cb:
        progress_cb("Writing HTML...", 0.9)

    # Embed the GLB + normal map as base64 data URIs directly in the HTML
    # so the viewer is fully self-contained — no CORS / file:// issues
    # when opening the HTML directly from disk in any browser.  Once the
    # HTML is generated it will load the 3D model without needing the
    # companion .glb/.png files to be reachable.
    glb_b64  = _b64.b64encode(open(glb_path,  "rb").read()).decode("ascii")
    nm_b64   = _b64.b64encode(open(nm_path,   "rb").read()).decode("ascii")
    html_doc = _build_hf_normals_html(glb_b64=glb_b64, normals_b64=nm_b64, name=settings.get("obj_id", ""))
    html_doc = _phone_3d(html_doc)
    html_path = _os.path.join(out_dir, "index_hf_normals.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    if progress_cb:
        progress_cb("Done", 1.0)
    return {"glb": glb_path, "normal_map": nm_path, "html": html_path}


def _build_hf_normals_html(glb_b64: str = "", normals_b64: str = "", name: str = "") -> str:
    """Return the Three.js embedded HTML template for the HF+normals viewer.

    When ``glb_b64`` and ``normals_b64`` are provided, the GLB and normal
    map PNG are embedded as base64 data URIs directly in the page so the
    viewer works when opened from ``file://`` without requiring a local
    HTTP server (browser CORS would otherwise block GLB fetch).
    """
    _glb_uri  = f"data:model/gltf-binary;base64,{glb_b64}"  if glb_b64  else "hf_normals_model.glb"
    _nm_uri   = f"data:image/png;base64,{normals_b64}"      if normals_b64 else "hf_normal_map.png"
    # Inlined for the same reason as the standard viewer: this page is written
    # into archival deposits, and a deposit that needs jsdelivr to still be
    # serving three.js r128 is not archived.
    from helio.vendor_assets import three_js_bundle
    _three_js = three_js_bundle()
    _html = """<!DOCTYPE html>
<html><head><title>Helio | HF+Normals 3D Viewer</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;}
body{overflow:hidden;background:#111;color:#eee;font-family:'Segoe UI',sans-serif;}
#c{position:fixed;top:0;left:0;width:100%;height:100%;}
#badge{position:fixed;top:10px;left:50%;transform:translateX(-50%);
  background:rgba(26,107,90,0.85);color:#fff;padding:4px 14px;border-radius:12px;
  font-size:11px;letter-spacing:0.5px;pointer-events:none;z-index:5;}
#panel{position:fixed;top:0;right:0;width:270px;height:100%;
  background:rgba(10,18,16,0.93);border-left:1px solid #1a6b5a;
  padding:14px 12px;overflow-y:auto;z-index:10;transition:transform 0.25s;}
#panel.hidden{transform:translateX(270px);}
#toggle{position:fixed;top:12px;right:280px;z-index:11;background:#0d2e28;
  border:1px solid #1a6b5a;color:#5df0c8;padding:6px 10px;border-radius:6px;
  cursor:pointer;font-size:12px;transition:right 0.25s;}
#toggle.hidden{right:10px;}
h2{font-size:12px;color:#5df0c8;text-transform:uppercase;letter-spacing:1px;
  margin-bottom:8px;padding-bottom:4px;border-bottom:1px solid #1a6b5a;}
.row{margin-bottom:9px;}
label{display:block;font-size:11px;color:#8ecec4;margin-bottom:2px;}
input[type=range]{width:100%;accent-color:#5df0c8;cursor:pointer;}
.val{display:inline-block;font-size:11px;color:#5df0c8;float:right;}
.note{font-size:10px;color:#5df0c8;margin-top:4px;line-height:1.5;}
kbd{display:inline-block;background:#0d2e28;border:1px solid #1a6b5a;border-radius:3px;padding:1px 5px;font-size:10px;font-family:inherit;color:#5df0c8;min-width:14px;text-align:center;}
.krow{display:flex;justify-content:space-between;font-size:10px;color:#8ecec4;margin-bottom:4px;}
hr{border:none;border-top:1px solid #1a2e2a;margin:8px 0;}
#err{display:none;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  background:#c0392b;padding:20px 30px;border-radius:8px;font-size:16px;text-align:center;z-index:20;}
#objname{position:fixed;top:10px;left:10px;max-width:46vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  background:rgba(13,20,22,0.72);color:#dfe9e6;border:1px solid #2a3a3a;padding:4px 10px;border-radius:12px;
  font-size:12px;font-weight:600;z-index:15;pointer-events:none;}
#objname:empty{display:none;}
</style></head>
<body>
<canvas id="c"></canvas>
<div id="objname">__OBJ_NAME__</div>
<div id="badge">Helio &middot; HF geometry + PS normal map</div>
<button id="toggle" onclick="togglePanel()">&#9664; Controls</button>
<div id="panel">
  <h2>Raking Light</h2>
  <div class="row"><label>Intensity <span class="val" id="v_rake">3.5</span></label>
    <input type="range" id="s_rake" min="0" max="8" step="0.1" value="3.5"></div>
  <div class="row"><label>Elevation <span class="val" id="v_elev">30</span>&deg;</label>
    <input type="range" id="s_elev" min="1" max="89" step="1" value="30"></div>
  <hr>
  <h2>Ambient / Fill</h2>
  <div class="row"><label>Ambient <span class="val" id="v_amb">0.6</span></label>
    <input type="range" id="s_amb" min="0" max="3" step="0.05" value="0.6"></div>
  <div class="row"><label>Fill Light <span class="val" id="v_fill">0.4</span></label>
    <input type="range" id="s_fill" min="0" max="3" step="0.05" value="0.4"></div>
  <hr>
  <h2>Colour</h2>
  <div class="row"><label>Photograph colour <span class="val" id="v_col">0.00</span></label>
    <input type="range" id="s_col" min="0" max="1" step="0.01" value="0" disabled></div>
  <div class="note" id="n_col">0 = the relief alone, 1 = the photograph's colour laid over it</div>
  <hr>
  <h2>Normal Map</h2>
  <div class="row"><label>Normal Scale <span class="val" id="v_nscale">1.5</span></label>
    <input type="range" id="s_nscale" min="0" max="5" step="0.05" value="1.5"></div>
  <div class="note">Higher = stronger per-pixel shading from PS normals</div>
  <hr>
  <h2>Render</h2>
  <div class="row"><label>Exposure <span class="val" id="v_exp">1.2</span></label>
    <input type="range" id="s_exp" min="0.2" max="4" step="0.05" value="1.2"></div>
  <div class="row"><label>Roughness <span class="val" id="v_rough">0.55</span></label>
    <input type="range" id="s_rough" min="0" max="1" step="0.01" value="0.55"></div>
  <div class="row"><label>Background <span class="val" id="v_bg">0.08</span></label>
    <input type="range" id="s_bg" min="0" max="1" step="0.01" value="0.08"></div>
  <hr>
  <h2>Virtual Depth of Field</h2>
  <div class="row"><label>Aperture (f-stop) <span class="val" id="v_dof_ap">0.0</span></label>
    <input type="range" id="s_dof_ap" min="0" max="1" step="0.01" value="0.0"></div>
  <div class="row"><label>Focal Plane <span class="val" id="v_dof_fp">0.50</span></label>
    <input type="range" id="s_dof_fp" min="0" max="1" step="0.01" value="0.50"></div>
  <div class="row"><label>Max Blur (px) <span class="val" id="v_dof_blur">8</span></label>
    <input type="range" id="s_dof_blur" min="2" max="24" step="1" value="8"></div>
  <div class="note">Simulates a virtual macro lens. Aperture 0 = off.<br>Focal Plane = normalised depth 0&rarr;1 (0=surface top).<br>Based on the thin-lens equation: blur = aperture &times; |z &minus; focal| / focal.</div>
  <hr>
  <h2>Light Compass</h2>
  <div style="text-align:center;margin-bottom:4px;">
    <canvas id="compass" width="84" height="84" style="cursor:crosshair;border-radius:50%;"></canvas>
  </div>
  <div class="note" style="margin-top:0;text-align:center;">Drag the dome to set light direction</div>
  <hr>
  <h2>Shortcuts</h2>
  <div class="krow"><span><kbd>R</kbd> Reset</span><span><kbd>S</kbd> Screenshot</span></div>
  <div class="krow"><span><kbd>F</kbd> Fit view</span><span></span></div>
  <div class="krow"><span><kbd>&larr;</kbd><kbd>&rarr;</kbd> Azimuth</span><span><kbd>&uarr;</kbd><kbd>&darr;</kbd> Elevation</span></div>
  <div class="note" style="margin-top:4px;">
    <kbd>LMB</kbd> Rotate &nbsp; <kbd>RMB</kbd> Pan &nbsp; <kbd>Scroll</kbd> Zoom<br>
    <kbd>&#8679;</kbd>+Drag Sweep light
  </div>
</div>
<div id="err">Failed to load model or normal map.<br>The viewer is self-contained — if you see this, the embedded GLB or normal map may be corrupted. Try re-exporting.</div>
<script>/*__THREE_JS__*/</script>
<script>
const canvas=document.getElementById('c');
const scene=new THREE.Scene();
const bgCol=new THREE.Color(0x1e1e1e);
scene.background=bgCol;
const camera=new THREE.PerspectiveCamera(45,innerWidth/innerHeight,0.1,50000);
const renderer=new THREE.WebGLRenderer({antialias:true,canvas,preserveDrawingBuffer:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.setSize(innerWidth,innerHeight);
renderer.outputEncoding=THREE.sRGBEncoding;
renderer.toneMapping=THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure=1.2;

const hemiLight=new THREE.HemisphereLight(0xfff0e0,0x101818,0.6);
scene.add(hemiLight);
const fillLight=new THREE.DirectionalLight(0xe8fffc,0.4);
fillLight.position.set(-3,1,2); scene.add(fillLight);
const rakingLight=new THREE.DirectionalLight(0xffffff,3.5);
rakingLight.position.set(200,0,100); scene.add(rakingLight);

const controls=new THREE.OrbitControls(camera,renderer.domElement);
controls.enableDamping=true; controls.dampingFactor=0.06;

let loadedMesh=null, modelCenter=new THREE.Vector3(), modelSize=1;
let normalMapTex=null;
const colourU={value:0.0};   // how much of the photograph's colour lies over the relief (0..1)

// Load normal map texture first, then GLB
new THREE.TextureLoader().load('hf_normal_map.png',
  tex=>{
    tex.encoding=THREE.LinearEncoding;  // normal maps must be linear
    tex.wrapS=tex.wrapT=THREE.RepeatWrapping;
    // The UVs recomputed below assign v=0 to the mesh TOP (image top row).
    // TextureLoader defaults flipY=true, which puts the PNG's BOTTOM row at
    // v=0 — sampling the normal map vertically mirrored against the mesh
    // (ghosted second image). flipY=false aligns v=0 with the PNG top row.
    tex.flipY=false;
    normalMapTex=tex;
    loadGlb();
  },
  undefined,
  ()=>{ loadGlb(); }  // proceed even if normal map fails
);

function loadGlb(){
  new THREE.GLTFLoader().load('hf_normals_model.glb',gltf=>{
    loadedMesh=gltf.scene;
    loadedMesh.traverse(c=>{
      if(!c.isMesh) return;
      c.geometry.computeVertexNormals();
      const pos=c.geometry.attributes.position;
      let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
      for(let i=0;i<pos.count;i++){
        const x=pos.getX(i),y=pos.getY(i);
        if(x<minX)minX=x;if(x>maxX)maxX=x;
        if(y<minY)minY=y;if(y>maxY)maxY=y;
      }
      const rX=maxX-minX||1, rY=maxY-minY||1;
      const uvArr=new Float32Array(pos.count*2);
      for(let i=0;i<pos.count;i++){
        uvArr[i*2  ]=(pos.getX(i)-minX)/rX;
        uvArr[i*2+1]=1.0-((pos.getY(i)-minY)/rY);
      }
      c.geometry.setAttribute('uv',new THREE.BufferAttribute(uvArr,2));
      c.geometry.attributes.uv.needsUpdate=true;
      if(c.geometry.index) c.geometry.computeTangents();
      const old=c.material;
      const albedoTex=(old && old.map)?old.map:null;     // the photograph's colour, from the model
      const mat=new THREE.MeshStandardMaterial({
        color: 0xffffff,
        vertexColors: !!c.geometry.attributes.color,
        roughness: 0.55,
        metalness: 0.0,
        side: THREE.DoubleSide,
      });
      if(albedoTex){
        mat.map=albedoTex;
        // The colour slider: the relief in neutral clay grey (dark enough not to blow out under the raking light), the photograph laid over it by uColour.
        mat.onBeforeCompile=shader=>{
          shader.uniforms.uColour=colourU;
          shader.fragmentShader=shader.fragmentShader
            .replace('void main() {','uniform float uColour;\\nvoid main() {')
            .replace('#include <map_fragment>','#include <map_fragment>\\n  diffuseColor.rgb=mix(vec3(0.12),diffuseColor.rgb,uColour);');
        };
        document.getElementById('s_col').disabled=false;
      }else{
        document.getElementById('n_col').textContent='This model has no colour image to lay over the relief.';
      }
      if(normalMapTex){
        mat.normalMap=normalMapTex;
        mat.normalScale=new THREE.Vector2(1.5,1.5);
      }
      c.material=mat;
      if(old && old.dispose) old.dispose();
    });
    scene.add(loadedMesh);
    const box=new THREE.Box3().setFromObject(loadedMesh);
    modelSize=box.getSize(new THREE.Vector3()).length();
    box.getCenter(modelCenter);
    controls.target.copy(modelCenter);
    camera.position.set(
      modelCenter.x + modelSize*0.15,
      modelCenter.y - modelSize*0.3,
      modelCenter.z + modelSize*1.1
    );
    camera.lookAt(modelCenter);
    updateRaking();
  },undefined,()=>{document.getElementById('err').style.display='block';});
}

window.addEventListener('resize',()=>{
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
});

let lightAz=0, lightEl=30;
function updateRaking(){
  const phi=THREE.MathUtils.degToRad(lightEl);
  const az=THREE.MathUtils.degToRad(lightAz);
  const d=300;
  rakingLight.position.set(d*Math.cos(phi)*Math.cos(az), d*Math.cos(phi)*Math.sin(az), d*Math.sin(phi));
  rakingLight.lookAt(modelCenter);
  document.getElementById('s_elev').value=lightEl;
  document.getElementById('v_elev').textContent=Math.round(lightEl);
  drawDome();
}

// ── Light Compass — draggable virtual dome (ported from the WebRTI viewer) ──
const domeCv=document.getElementById('compass');
const domeCtx=domeCv.getContext('2d');
const domeR=domeCv.width/2;
let domeDrag=false;
function drawDome(){
  const azR=THREE.MathUtils.degToRad(lightAz), elR=THREE.MathUtils.degToRad(lightEl);
  const dlx=Math.cos(elR)*Math.sin(azR), dly=Math.cos(elR)*Math.cos(azR);
  domeCtx.clearRect(0,0,domeCv.width,domeCv.height);
  const g=domeCtx.createRadialGradient(domeR,domeR,0,domeR,domeR,domeR);
  g.addColorStop(0,'#1A1B1E'); g.addColorStop(1,'#1A1B1E');
  domeCtx.fillStyle=g; domeCtx.beginPath(); domeCtx.arc(domeR,domeR,domeR,0,Math.PI*2); domeCtx.fill();
  domeCtx.strokeStyle='#3E4046'; domeCtx.lineWidth=1;
  [0.33,0.67,1].forEach(r=>{domeCtx.beginPath();domeCtx.arc(domeR,domeR,r*domeR,0,Math.PI*2);domeCtx.stroke();});
  for(let i=0;i<8;i++){const a=i*Math.PI/4;domeCtx.beginPath();domeCtx.moveTo(domeR,domeR);domeCtx.lineTo(domeR+Math.cos(a)*domeR,domeR+Math.sin(a)*domeR);domeCtx.stroke();}
  const m=Math.sqrt(dlx*dlx+dly*dly), sc=m>0.95?0.95/m:1;
  const px=domeR+dlx*domeR*sc, py=domeR-dly*domeR*sc;
  domeCtx.beginPath(); domeCtx.arc(px,py,6,0,Math.PI*2);
  domeCtx.fillStyle='#E6A93E'; domeCtx.fill(); domeCtx.strokeStyle='#222326'; domeCtx.lineWidth=2; domeCtx.stroke();
}
function domeXY(px,py){
  const dlx=(px-domeR)/domeR, dly=-(py-domeR)/domeR;
  const r=Math.min(1,Math.sqrt(dlx*dlx+dly*dly));
  const elR=Math.max(3*Math.PI/180,Math.min(89*Math.PI/180,Math.acos(r)));
  const azR=Math.atan2(dlx,dly);
  lightAz=((azR*180/Math.PI)%360+360)%360;
  lightEl=elR*180/Math.PI;
  updateRaking();
}
domeCv.addEventListener('mousedown',e=>{e.preventDefault();domeDrag=true;domeXY(e.offsetX,e.offsetY);});
window.addEventListener('mousemove',e=>{if(!domeDrag)return;const r=domeCv.getBoundingClientRect();domeXY(e.clientX-r.left,e.clientY-r.top);});
window.addEventListener('mouseup',()=>domeDrag=false);
domeCv.addEventListener('touchstart',e=>{e.preventDefault();domeDrag=true;const r=domeCv.getBoundingClientRect(),t=e.touches[0];domeXY(t.clientX-r.left,t.clientY-r.top);},{passive:false});
domeCv.addEventListener('touchmove',e=>{e.preventDefault();const r=domeCv.getBoundingClientRect(),t=e.touches[0];domeXY(t.clientX-r.left,t.clientY-r.top);},{passive:false});
domeCv.addEventListener('touchend',()=>domeDrag=false);

function sl(id,vid,cb){
  document.getElementById(id).addEventListener('input',function(){
    document.getElementById(vid).textContent=parseFloat(this.value).toFixed(2);
    cb(parseFloat(this.value));
  });
}
sl('s_rake','v_rake',v=>{rakingLight.intensity=v;});
sl('s_elev','v_elev',v=>{lightEl=parseFloat(v);updateRaking();});
sl('s_amb','v_amb',v=>{hemiLight.intensity=v;});
sl('s_fill','v_fill',v=>{fillLight.intensity=v;});
sl('s_col','v_col',v=>{colourU.value=v;});
sl('s_nscale','v_nscale',v=>{
  if(loadedMesh)loadedMesh.traverse(c=>{
    if(c.isMesh&&c.material.normalMap)c.material.normalScale.set(v,v);
  });
});
sl('s_exp','v_exp',v=>{renderer.toneMappingExposure=v;});
sl('s_rough','v_rough',v=>{
  if(loadedMesh)loadedMesh.traverse(c=>{if(c.isMesh){c.material.roughness=v;c.material.needsUpdate=true;}});
});
sl('s_bg','v_bg',v=>{bgCol.setRGB(v*0.13,v*0.13,v*0.13);scene.background=bgCol;});

function takeScreenshot(){
  renderer.render(scene,camera);
  const nm=(document.getElementById('objname').textContent||'relief').trim().replace(/[^\\w.-]+/g,'_')||'relief';
  canvas.toBlob(b=>{
    const a=document.createElement('a');
    a.href=URL.createObjectURL(b);
    a.download=nm+'_'+new Date().toISOString().slice(0,19).replace(/[:T]/g,'-')+'.png';
    a.click();
  });
}
let shiftDown=false,dragging=false,dsx=0,dsy=0;
window.addEventListener('keydown',e=>{
  if(e.key==='Shift'){shiftDown=true;controls.enabled=false;}
  if(e.key==='r'||e.key==='R')controls.reset();
  if((e.key==='f'||e.key==='F')&&loadedMesh){
    const b=new THREE.Box3().setFromObject(loadedMesh);
    const s=b.getSize(new THREE.Vector3()).length();
    const c2=b.getCenter(new THREE.Vector3());
    controls.target.copy(c2);
    camera.position.set(c2.x,c2.y,c2.z+s*1.2);
    camera.lookAt(c2);
  }
  if(e.key==='s'||e.key==='S'){e.preventDefault();takeScreenshot();}
  if(e.key==='ArrowLeft'){lightAz=(lightAz-5)%360;updateRaking();}
  if(e.key==='ArrowRight'){lightAz=(lightAz+5)%360;updateRaking();}
  if(e.key==='ArrowUp'){lightEl=Math.min(89,lightEl+5);updateRaking();}
  if(e.key==='ArrowDown'){lightEl=Math.max(1,lightEl-5);updateRaking();}
});
window.addEventListener('keyup',e=>{if(e.key==='Shift'){shiftDown=false;dragging=false;controls.enabled=true;}});
window.addEventListener('mousedown',e=>{if(shiftDown){dragging=true;dsx=e.clientX;dsy=e.clientY;}});
window.addEventListener('mouseup',()=>{dragging=false;});
window.addEventListener('mousemove',e=>{
  if(dragging){
    lightAz=(lightAz+(e.clientX-dsx)*0.5)%360;
    lightEl=Math.max(1,Math.min(89,lightEl-(e.clientY-dsy)*0.3));
    dsx=e.clientX;dsy=e.clientY;
    updateRaking();
  }
});

function togglePanel(){
  const p=document.getElementById('panel');
  const t=document.getElementById('toggle');
  const h=p.classList.toggle('hidden');
  t.classList.toggle('hidden',h);
  t.innerHTML=h?'&#9654; Controls':'&#9664; Controls';
}

let dofAperture=0, dofFocalPlane=0.50, dofMaxBlur=8;
let rtDepth=null, rtColor=null, dofMat=null, dofQuad=null, dofScene=null, dofCamera=null;

function initDoF(){
  const W=renderer.domElement.width, H=renderer.domElement.height;
  rtColor = new THREE.WebGLRenderTarget(W,H,{
    minFilter:THREE.LinearFilter, magFilter:THREE.LinearFilter,
    format:THREE.RGBAFormat, type:THREE.UnsignedByteType
  });
  rtDepth = new THREE.WebGLRenderTarget(W,H,{
    minFilter:THREE.LinearFilter, magFilter:THREE.LinearFilter,
    format:THREE.RGBAFormat, type:THREE.UnsignedByteType,
    depthBuffer:true, depthTexture: new THREE.DepthTexture(W,H)
  });
  dofCamera = new THREE.OrthographicCamera(-1,1,1,-1,0,1);
  dofScene  = new THREE.Scene();
  dofMat = new THREE.ShaderMaterial({
    uniforms:{
      tColor:{value:null}, tDepth:{value:null},
      uNear:{value:camera.near}, uFar:{value:camera.far},
      uAperture:{value:0.0}, uFocalPlane:{value:0.5}, uMaxBlur:{value:8.0},
      uRes:{value:new THREE.Vector2(W,H)}
    },
    vertexShader:'varying vec2 vUv; void main(){ vUv=uv; gl_Position=vec4(position,1.0); }',
    fragmentShader:'precision highp float; uniform sampler2D tColor, tDepth; uniform float uNear, uFar, uAperture, uFocalPlane, uMaxBlur; uniform vec2 uRes; varying vec2 vUv; float linearDepth(float d){ float z = d * 2.0 - 1.0; return (2.0 * uNear * uFar) / (uFar + uNear - z * (uFar - uNear)); } void main(){ if(uAperture < 0.001){ gl_FragColor = texture2D(tColor, vUv); return; } float rawD = texture2D(tDepth, vUv).r; float linD = linearDepth(rawD); float normD = clamp((linD - uNear) / (uFar - uNear), 0.0, 1.0); float coc = uAperture * abs(normD - uFocalPlane) / max(uFocalPlane, 0.001); float blur = min(coc * uMaxBlur, uMaxBlur); if(blur < 0.5){ gl_FragColor = texture2D(tColor, vUv); return; } vec4 col = vec4(0.0); float total = 0.0; vec2 px = 1.0 / uRes; float r = blur; for(int i=-3; i<=3; i++){ for(int j=-3; j<=3; j++){ if(abs(float(i))+abs(float(j)) > 4.0) continue; vec2 off = vec2(float(i),float(j)) * px * r; col += texture2D(tColor, vUv + off); total += 1.0; } } gl_FragColor = col / total; }'
  });
  const geo = new THREE.PlaneGeometry(2,2);
  dofQuad = new THREE.Mesh(geo, dofMat);
  dofScene.add(dofQuad);
}

function resizeDoF(){
  if(!rtColor) return;
  const W=renderer.domElement.width, H=renderer.domElement.height;
  rtColor.setSize(W,H); rtDepth.setSize(W,H);
  if(dofMat) dofMat.uniforms.uRes.value.set(W,H);
}

window.addEventListener('resize',()=>{
  camera.aspect=innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth,innerHeight);
  resizeDoF();
});

function slv(id,vid,digits,cb){
  const el=document.getElementById(id); if(!el)return;
  el.addEventListener('input',function(){
    const v=parseFloat(this.value);
    document.getElementById(vid).textContent=v.toFixed(digits);
    cb(v);
  });
}
slv('s_dof_ap','v_dof_ap',2,v=>{ dofAperture=v; if(!rtColor&&v>0) initDoF(); if(dofMat) dofMat.uniforms.uAperture.value=v; });
slv('s_dof_fp','v_dof_fp',2,v=>{ dofFocalPlane=v; if(dofMat) dofMat.uniforms.uFocalPlane.value=v; });
slv('s_dof_blur','v_dof_blur',0,v=>{ dofMaxBlur=v; if(dofMat) dofMat.uniforms.uMaxBlur.value=v; });

updateRaking();
(function animate(){
  requestAnimationFrame(animate);
  controls.update();
  if(dofAperture > 0.001 && rtColor){
    renderer.setRenderTarget(rtDepth);
    renderer.render(scene,camera);
    renderer.setRenderTarget(rtColor);
    renderer.render(scene,camera);
    renderer.setRenderTarget(null);
    dofMat.uniforms.tColor.value = rtColor.texture;
    dofMat.uniforms.tDepth.value = rtDepth.depthTexture;
    dofMat.uniforms.uNear.value  = camera.near;
    dofMat.uniforms.uFar.value   = camera.far;
    renderer.render(dofScene, dofCamera);
  } else {
    renderer.render(scene,camera);
  }
})();
</script></body></html>"""
    # Inject the GLB + normal map URIs (base64 data URI when available,
    # relative filename otherwise — useful for the legacy non-embedded path).
    _html = _html.replace("'hf_normal_map.png'",  f"'{_nm_uri}'")
    _html = _html.replace("'hf_normals_model.glb'", f"'{_glb_uri}'")
    _html = _html.replace("/*__THREE_JS__*/", _three_js)
    return _html.replace("__OBJ_NAME__", __import__("html").escape(name or ""))


# ── Legacy stubs (to be implemented) ───────────────────────────────

def export_glb_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
    path: str = "mesh.glb",
    vertex_colors: Optional[np.ndarray] = None,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> str:
    """Export a glTF 2.0 binary (.glb) mesh.

    Renders the supplied vertex/face buffers to a self-contained GLB file
    using *trimesh*. Optional vertex colours are packed as RGBA uint8.
    If ``uvs`` are provided, a PBR material placeholder is attached so
    downstream viewers (Sketchfab, Three.js) interpret the UVs correctly.

    Parameters
    ----------
    vertices : (V, 3) float32
    faces : (F, 3) int32 / uint32
    normals : optional (V, 3) float32 — recomputed from faces if None
    uvs : optional (V, 2) float32
    vertex_colors : optional (V, 3) or (V, 4) uint8
    path : output path (``.glb``)
    progress_cb : optional fractional-progress callback (0.0–1.0)

    Returns *path*.
    """
    import trimesh
    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    f_data = np.ascontiguousarray(faces, dtype=np.uint32)
    mesh = trimesh.Trimesh(vertices=verts, faces=f_data, process=False)
    if normals is not None:
        mesh.vertex_normals = np.ascontiguousarray(normals, dtype=np.float32)
    if vertex_colors is not None:
        vc = np.ascontiguousarray(vertex_colors, dtype=np.uint8)
        if vc.shape[-1] == 3:
            vc = np.column_stack([vc, np.full(len(vc), 255, dtype=np.uint8)])
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=vc)
    if uvs is not None:
        empty_tex = trimesh.visual.TextureVisuals(uv=np.ascontiguousarray(uvs, dtype=np.float32))
        mesh.visual = empty_tex
    if progress_cb: progress_cb(0.5)
    mesh.export(path)
    _fix_glb_material(path, metallic=0.0, roughness=0.65, double_sided=True)
    if progress_cb: progress_cb(1.0)
    return path


def export_obj_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    uvs: Optional[np.ndarray] = None,
    path: str = "mesh.obj",
    vertex_colors: Optional[np.ndarray] = None,
) -> str:
    """Wavefront OBJ export — vertex/face/normal/uv buffers only (no MTL).

    Returns *path*.
    """
    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    f_data = np.ascontiguousarray(faces, dtype=np.int32)
    with open(path, 'w') as f:
        # Bulk write vertices - much faster than loop
        v_lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts]
        f.writelines(v_lines)
        if normals is not None:
            n_arr = np.asarray(normals, dtype=np.float32)
            vn_lines = [f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n" for n in n_arr]
            f.writelines(vn_lines)
        if uvs is not None:
            # Flip V, as every other writer here does.  HELIO builds UVs with
            # the origin at the top-left to match image indexing, while OBJ
            # (and glTF, and trimesh) put it at the bottom-left - so writing
            # them through unchanged mirrors the texture against the geometry.
            # _uv_to_trimesh does this for the GLB paths and
            # _write_obj_sketchfab_zip does it for the Sketchfab OBJ; this
            # function was the one place that did not, so a texture applied to
            # its output came out upside down relative to every other export.
            uv_arr = np.asarray(uvs, dtype=np.float32)
            vt_lines = [f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n" for uv in uv_arr]
            f.writelines(vt_lines)
        has_vn = normals is not None
        has_vt = uvs is not None
        # Bulk write faces - much faster than loop
        f_lines = []
        for face in f_data:
            a, b, c = face[0] + 1, face[1] + 1, face[2] + 1
            def _v(idx):
                if has_vn and has_vt:
                    return f"{idx}/{idx}/{idx}"
                if has_vt:
                    return f"{idx}/{idx}"
                if has_vn:
                    return f"{idx}//{idx}"
                return str(idx)
            f_lines.append(f"f {_v(a)} {_v(b)} {_v(c)}\n")
        f.writelines(f_lines)
    return path


def export_stl_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: Optional[np.ndarray] = None,
    path: str = "mesh.stl",
) -> str:
    """Binary STL export via *trimesh*.

    ``normals`` are recomputed from face cross-products if not supplied.

    Returns *path*.
    """
    import trimesh
    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    f_data = np.ascontiguousarray(faces, dtype=np.uint32)
    mesh = trimesh.Trimesh(vertices=verts, faces=f_data, process=False)
    if normals is not None:
        mesh.face_normals = np.ascontiguousarray(normals, dtype=np.float32)
    # STL binary is the default trimesh output for .stl/.STL
    mesh.export(path, file_type='stl')
    return path


def export_point_cloud_ply(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    path: str = "pointcloud.ply",
) -> str:
    """PLY point-cloud export via *trimesh.PointCloud*.

    Returns *path*.
    """
    import trimesh
    pts = np.ascontiguousarray(points, dtype=np.float32)
    cloud = trimesh.points.PointCloud(pts)
    if colors is not None:
        cloud.colors = np.ascontiguousarray(colors, dtype=np.uint8)
    if normals is not None:
        # PointCloud stores vertex metadata under .metadata
        cloud.metadata['normals'] = np.ascontiguousarray(normals, dtype=np.float32)
    cloud.export(path, file_type='ply')
    return path


def export_publication_sheet(
    renders: dict[str, np.ndarray],
    path: str,
    title: str = "Helio Publication Sheet",
    dpi: int = 300,
    page_size: str = "A4",
    layout: str = "grid",
    progress_cb: Optional[Callable[[float], None]] = None,
) -> str:
    """Multi-image publication sheet (PDF/PNG).

    Not yet implemented in the package-level API. The full publication-sheet
    pipeline (engine renders + 3D captures + metrology + metadata + Abbott
    curve) lives on ``RTIViewerPro.generate_contact_sheet`` in ``Helio_Pro.py``
    — call that method directly; it has all the data plumbing.

    Returns *path*.
    """
    raise NotImplementedError(
        "Publication sheet not yet implemented in helio.io.exporters. "
        "Use RTIViewerPro.generate_contact_sheet() from Helio_Pro.py instead — "
        "it has all the data plumbing."
    )


def export_video_walkthrough(
    normals: np.ndarray,
    depth: np.ndarray,
    path: str,
    engine: str = "Standard PBR",
    total_frames: int = 72,
    framerate: int = 24,
    width: int = 1920,
    height: int = 1080,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> str:
    """360° turntable video of the 3D model.

    Not yet implemented in the package-level API. The interactive video
    walkthrough (framestepping + light assignment) lives on
    ``VideoWalkthroughWindow`` in ``Helio_Pro.py`` and the spin/sweep
    animation capture is wired into ``RTIViewerPro.export_sweep_gif``
    + ``export_spin_animation``.

    Returns *path*.
    """
    raise NotImplementedError(
        "Video walkthrough not yet implemented in helio.io.exporters. "
        "Use RTIViewerPro.export_sweep_gif() / .export_spin_animation() "
        "for turntable video from Helio_Pro.py instead."
    )


# ── RTI viewer HTML generators ─────────────────────────────────────

def export_ptm_selfcontained_viewer(
    out_path: str,
    ptm_coeffs: list[np.ndarray],
    albedo_rgb: np.ndarray,
    normals: np.ndarray,
    src_w: int,
    src_h: int,
    obj_id: str = "Object",
    crop_box: tuple[int, int, int, int] | None = None,
    max_edge: int = 1024,
    progress_cb: Callable[[str, float | None], None] | None = None,
) -> str:
    """Export a self-contained HTML5 PTM RTI viewer with embedded base64 textures.

    Encodes 6 PTM coefficient planes + albedo + normal map as base64 PNGs
    embedded in a single portable HTML file.  Resolution is capped to
    *max_edge* on the long side to keep the file size manageable.

    Returns *out_path*.
    """
    import base64

    if progress_cb is not None:
        progress_cb("Encoding PTM self-contained viewer…", None)

    if crop_box is not None:
        x1, y1, x2, y2 = crop_box
    else:
        trim = 0.0
        x1 = max(0, int(src_w * trim))
        y1 = max(0, int(src_h * trim))
        x2 = src_w - max(0, int(src_w * trim))
        y2 = src_h - max(0, int(src_h * trim))

    src_hc, src_wc = y2 - y1, x2 - x1
    if max(src_hc, src_wc) > max_edge:
        scale = max_edge / max(src_hc, src_wc)
        ph = max(1, int(src_hc * scale))
        pw = max(1, int(src_wc * scale))
    else:
        ph, pw = src_hc, src_wc

    coeff_b64: list[str] = []
    for i in range(6):
        ch_full = cv2.normalize(
            ptm_coeffs[i].reshape(src_h, src_w)[y1:y2, x1:x2],
            None, 0, 255, cv2.NORM_MINMAX,
        ).astype(np.uint8)
        ch = (
            cv2.resize(ch_full, (pw, ph), interpolation=cv2.INTER_AREA)
            if (pw != src_wc or ph != src_hc)
            else ch_full
        )
        _, buf = cv2.imencode(".png", ch, [cv2.IMWRITE_PNG_COMPRESSION, 7])
        coeff_b64.append(base64.b64encode(buf.tobytes()).decode())

    alb_full = cv2.cvtColor(
        albedo_rgb[y1:y2, x1:x2], cv2.COLOR_RGB2BGR
    )
    alb_res = (
        cv2.resize(alb_full, (pw, ph), interpolation=cv2.INTER_AREA)
        if (pw != src_wc or ph != src_hc)
        else alb_full
    )
    _, abuf = cv2.imencode(".png", alb_res, [cv2.IMWRITE_PNG_COMPRESSION, 7])
    alb_b64 = base64.b64encode(abuf.tobytes()).decode()

    nm_full = (
        np.round((normals[y1:y2, x1:x2] * 0.5 + 0.5) * 255)
        .clip(0, 255)
        .astype(np.uint8)
    )
    nm_full_bgr = cv2.cvtColor(nm_full, cv2.COLOR_RGB2BGR)
    nm_res = (
        cv2.resize(nm_full_bgr, (pw, ph), interpolation=cv2.INTER_LINEAR)
        if (pw != src_wc or ph != src_hc)
        else nm_full_bgr
    )
    _, nbuf = cv2.imencode(".png", nm_res, [cv2.IMWRITE_PNG_COMPRESSION, 7])
    nm_b64 = base64.b64encode(nbuf.tobytes()).decode()

    _safe_id = html.escape(obj_id)
    html_doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Helio RTI — {_safe_id}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:#0a0a0f;color:#ddd;font-family:'Segoe UI',sans-serif;overflow:hidden;}}
#wrap{{display:flex;width:100vw;height:100vh;}}
#cv{{flex:1;cursor:crosshair;image-rendering:auto;}}
#panel{{width:230px;min-width:230px;padding:14px 12px;background:rgba(13,13,20,0.96);
  border-left:1px solid #222;overflow-y:auto;}}
h3{{font-size:11px;color:#f39c12;letter-spacing:1px;text-transform:uppercase;
  margin-bottom:7px;padding-bottom:3px;border-bottom:1px solid #222;}}
.row{{margin-bottom:7px;font-size:11px;color:#aaa;}}
input[type=range]{{width:100%;accent-color:#f39c12;}}
select{{width:100%;background:#111;color:#ddd;border:1px solid #333;padding:4px 6px;
  font-size:11px;border-radius:4px;margin-bottom:8px;}}
.val{{color:#f39c12;float:right;}}
hr{{border:none;border-top:1px solid #1a1a2a;margin:8px 0;}}
</style></head><body>
<div id="wrap">
<canvas id="cv"></canvas>
<div id="panel">
  <h3>RTI Relighter</h3>
  <div class="row">Drag compass or move cursor over image to relight.</div>
  <div style="text-align:center;margin:8px 0;">
    <canvas id="compass" width="96" height="96" style="cursor:grab;border-radius:50%;background:transparent;border:1px solid #36383D;"></canvas>
  </div>
  <hr>
  <h3>Render Mode</h3>
  <select id="renderMode" onchange="render()">
    <option value="colour">Colour (albedo)</option>
    <option value="grey">Greyscale</option>
    <option value="greenShade">Green-Shade (RTI classic)</option>
    <option value="specEnh">Specular Enhanced</option>
    <option value="rakingOnly">Raking Light Only</option>
    <option value="normalVis">Normal Map Visualise</option>
  </select>
  <hr>
  <div class="row">Gain <span class="val" id="v_gain">1.0</span></div>
  <input type="range" id="s_gain" min="0.5" max="5" step="0.1" value="1.0">
  <div class="row" style="margin-top:6px;">Ambient <span class="val" id="v_amb">0.12</span></div>
  <input type="range" id="s_amb" min="0" max="1" step="0.02" value="0.12">
  <div class="row" style="margin-top:6px;">Specular Mix <span class="val" id="v_spec">0.5</span></div>
  <input type="range" id="s_spec" min="0" max="2" step="0.05" value="0.5">
  <hr>
  <div class="row" style="font-size:12px;color:#A0A3AA;">
    Object: {_safe_id}<br>
    Viewer size: {pw}×{ph} px<br>
    (Source: {src_wc}×{src_hc} px)<br>
    Generated by Helio
  </div>
</div>
</div>
<script>
const W={pw},H={ph};
const coeffB64=[{','.join(repr(s) for s in coeff_b64)}];
const albB64={repr(alb_b64)};
const nmB64={repr(nm_b64)};
let coeffData=new Array(6), albData=null, nmData=null;
let lx=0.5, ly=0.5, gain=1.0, ambient=0.12, specMix=0.5;
let loaded=0;
const cv=document.getElementById('cv');
const ctx=cv.getContext('2d');
cv.width=W; cv.height=H;

function decodeImg(b64, cb){{
  const img=new Image();
  img.onload=()=>{{
    const tc=document.createElement('canvas'); tc.width=W; tc.height=H;
    const tx=tc.getContext('2d'); tx.drawImage(img,0,0,W,H);
    cb(tx.getImageData(0,0,W,H).data);
  }};
  img.src='data:image/png;base64,'+b64;
}}

const TOTAL=8;
coeffB64.forEach((b,i)=>decodeImg(b,d=>{{coeffData[i]=d;loaded++;if(loaded>=TOTAL)render();}}));
decodeImg(albB64,d=>{{albData=d;loaded++;if(loaded>=TOTAL)render();}});
decodeImg(nmB64, d=>{{nmData=d; loaded++;if(loaded>=TOTAL)render();}});

function ptmVal(p){{
  const lxn=lx*2-1, lyn=ly*2-1;
  const lzn=Math.sqrt(Math.max(0,1-lxn*lxn-lyn*lyn));
  const u=[lxn*lxn, lyn*lyn, lxn*lyn, lxn*lzn, lyn*lzn, 1.0];
  let v=0;
  for(let k=0;k<6;k++) v+=u[k]*(coeffData[k][p*4]/255.);
  return Math.min(1,Math.max(0,v*gain+ambient));
}}

function render(){{
  if(loaded<TOTAL)return;
  const mode=document.getElementById('renderMode').value;
  const out=ctx.createImageData(W,H);
  for(let p=0;p<W*H;p++){{
    const i4=p*4;
    const lum=ptmVal(p);
    let r,g,b;
    if(mode==='colour'){{
      r=(albData[i4]/255.)*lum;
      g=(albData[i4+1]/255.)*lum;
      b=(albData[i4+2]/255.)*lum;
    }} else if(mode==='grey'){{
      const grey=(albData[i4]*0.299+albData[i4+1]*0.587+albData[i4+2]*0.114)/255.*lum;
      r=g=b=grey;
    }} else if(mode==='greenShade'){{
      r=lum*0.12; g=lum; b=lum*0.28;
    }} else if(mode==='specEnh'){{
      const base=lum; const spec=Math.pow(Math.max(0,base),4.0)*specMix;
      r=Math.min(1,(albData[i4]/255.)*base+spec);
      g=Math.min(1,(albData[i4+1]/255.)*base+spec);
      b=Math.min(1,(albData[i4+2]/255.)*base+spec);
    }} else if(mode==='rakingOnly'){{
      r=g=b=lum;
    }} else if(mode==='normalVis'){{
      r=nmData[i4]/255.; g=nmData[i4+1]/255.; b=nmData[i4+2]/255.;
    }}
    out.data[i4  ]=r*255;
    out.data[i4+1]=g*255;
    out.data[i4+2]=b*255;
    out.data[i4+3]=255;
  }}
  ctx.putImageData(out,0,0);
}}

function fitCanvas(){{
  // Beside the picture on a computer; a sheet along the bottom on a phone.
  const wrap=document.getElementById('wrap'), pnl=document.getElementById('panel');
  const side=getComputedStyle(pnl).position!=='fixed';
  const open=!document.body.classList.contains('hp-closed');
  const aw=wrap.clientWidth-(side?pnl.offsetWidth:0);
  const ah=wrap.clientHeight-(!side&&open?pnl.offsetHeight:0);
  const scale=Math.min(aw/W, ah/H);
  cv.style.width=(W*scale)+'px'; cv.style.height=(H*scale)+'px';
}}
window.addEventListener('resize', fitCanvas); fitCanvas();

cv.addEventListener('mousemove',e=>{{
  const r=cv.getBoundingClientRect();
  lx=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));
  ly=Math.max(0,Math.min(1,(e.clientY-r.top)/r.height));
  drawCompass(); render();
}});

const comp=document.getElementById('compass');
const cctx=comp.getContext('2d');
let cDrag=false;
function drawCompass(){{
  cctx.clearRect(0,0,96,96);
  cctx.fillStyle='#1A1B1E'; cctx.beginPath(); cctx.arc(48,48,42,0,Math.PI*2); cctx.fill();
  cctx.strokeStyle='#3E4046'; cctx.lineWidth=1.5;
  cctx.beginPath(); cctx.arc(48,48,42,0,Math.PI*2); cctx.stroke();
  [['N',48,10],['S',48,88],['W',10,52],['E',86,52]].forEach(([t,x,y])=>{{
    cctx.fillStyle='#B4B6BC'; cctx.font='600 11px system-ui,sans-serif';
    cctx.textAlign='center'; cctx.textBaseline='middle';
    cctx.fillText(t,x,y);
  }});
  cctx.beginPath(); cctx.arc(48+42*(lx*2-1),48+42*(ly*2-1),9,0,Math.PI*2);
  cctx.fillStyle='#E6A93E'; cctx.fill(); cctx.strokeStyle='#222326'; cctx.lineWidth=2; cctx.stroke();
}}
comp.addEventListener('mousedown',()=>cDrag=true);
window.addEventListener('mouseup',()=>cDrag=false);
comp.addEventListener('mousemove',e=>{{
  if(!cDrag)return;
  const r=comp.getBoundingClientRect();
  lx=Math.max(0,Math.min(1,(e.clientX-r.left)/96));
  ly=Math.max(0,Math.min(1,(e.clientY-r.top)/96));
  drawCompass(); render();
}});
drawCompass();

function sl(id,vid,setter){{
  document.getElementById(id).addEventListener('input',function(){{
    setter(parseFloat(this.value));
    document.getElementById(vid).textContent=parseFloat(this.value).toFixed(2);
    render();
  }});
}}
sl('s_gain','v_gain',v=>gain=v);
sl('s_amb','v_amb',v=>ambient=v);
sl('s_spec','v_spec',v=>specMix=v);
</script></body></html>"""

    from helio.application.phone_web import phone_ready
    html_doc = phone_ready(
        html_doc, panel="#panel", stage="#cv",
        hooks="window.HelioPhone={getLight:function(){var u=2*lx-1,v=2*ly-1;"
              "return [Math.atan2(u,v)*180/Math.PI,Math.acos(Math.min(1,Math.hypot(u,v)))*180/Math.PI];},"
              "setLight:function(a,e){var r=Math.cos(e*Math.PI/180),q=a*Math.PI/180;"
              "lx=(1+r*Math.sin(q))/2;ly=(1+r*Math.cos(q))/2;drawCompass();render();},"
              "reset:function(){lx=.5;ly=.5;drawCompass();render();}};\n",
        hooks_before="</script>",
        css="#wrap{justify-content:center;align-items:center;"
            "height:calc(100vh - var(--hp-sheet))!important}#cv{flex:0 0 auto}",
        tip="Drag the picture to move the light \u00b7 double-tap to reset")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    if progress_cb is not None:
        progress_cb("PTM self-contained viewer written", 1.0)

    return out_path


def export_combined_rti_viewer_html(
    out_path: str,
    normals: np.ndarray,
    albedo: np.ndarray,
    roughness: np.ndarray | None,
    spec_var: np.ndarray | None,
    h: int,
    w: int,
    crop_coords: tuple[int, int, int, int],
    obj_id: str = "Object",
    meta: dict | None = None,
    max_tex: int = 4096,
    progress_cb: Callable[[str, float | None], None] | None = None,
    extra_layers: "list[tuple[str, np.ndarray]] | None" = None,
) -> str:
    """Export a self-contained WebGL RTI viewer with embedded base64 textures.

    Embeds normal map, albedo, roughness, and specularity as base64
    data URIs in a single HTML file.  Uses WebGL with a multi-mode
    fragment shader (colour relight, greyscale, specular, normal map,
    USM, surface gradient, colour×specular).

    All textures are capped to *max_tex* on the longest side.

    Returns *out_path*.
    """
    import io as _io
    import base64 as _b64
    import PIL.Image as _PIL
    import html

    x1, y1, x2, y2 = crop_coords[:4]

    if progress_cb is not None:
        progress_cb("Encoding combined RTI viewer…", None)

    nrm = normals[y1:y2, x1:x2].astype(np.float32)
    alb = albedo[y1:y2, x1:x2]
    Hc, Wc = nrm.shape[:2]
    sc = min(1.0, max_tex / max(Hc, Wc))
    dh = max(1, int(Hc * sc))
    dw = max(1, int(Wc * sc))

    if progress_cb is not None:
        progress_cb("Encoding normal map…", 0.1)
    nrm_u8 = (
        (nrm / np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-8) * 0.5 + 0.5)
        * 255
    ).clip(0, 255).astype(np.uint8)
    if sc < 1.0:
        nrm_u8 = cv2.resize(nrm_u8, (dw, dh), cv2.INTER_AREA)
        alb = cv2.resize(alb, (dw, dh), cv2.INTER_AREA)

    def _b64_png(arr):
        buf = _io.BytesIO()
        _PIL.fromarray(arr).save(buf, "PNG", optimize=False)
        return _b64.b64encode(buf.getvalue()).decode()

    def _b64_jpg(arr, q=92):
        buf = _io.BytesIO()
        _PIL.fromarray(arr).save(buf, "JPEG", quality=q)
        return _b64.b64encode(buf.getvalue()).decode()

    if progress_cb is not None:
        progress_cb("Encoding textures…", 0.2)
    nrm_b64 = "data:image/png;base64," + _b64_png(nrm_u8)
    alb_b64 = "data:image/jpeg;base64," + _b64_jpg(alb)

    # Baked engine layers. The live shader can only relight from the normal
    # map; engines that are statistical (PCA), stack-derived (Multi-Light
    # Fusion) or expensive (MSII) cannot be recomputed in a browser, so they
    # travel as pre-rendered images the recipient can switch to.
    layer_b64: list = []
    if extra_layers:
        for _i, (_lbl, _img) in enumerate(extra_layers):
            try:
                a = np.asarray(_img)
                if a.dtype != np.uint8:
                    a = a.astype(np.float32)
                    if float(np.nanmax(a)) <= 1.0 + 1e-6:
                        a = a * 255.0
                    a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
                if a.ndim == 2:
                    a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
                a = a[..., :3]
                if a.shape[:2] != (h, w):
                    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
                a = a[y1:y2, x1:x2]
                if sc < 1.0:
                    a = cv2.resize(a, (dw, dh), interpolation=cv2.INTER_AREA)
                layer_b64.append((str(_lbl),
                                  "data:image/jpeg;base64," + _b64_jpg(a)))
            except Exception:
                _log.exception("baked layer %r could not be encoded", _lbl)
        if progress_cb is not None:
            progress_cb(f"Encoded {len(layer_b64)} engine layers…", 0.45)

    _layer_buttons = ""
    _baked_js = "var BAKED=[];"
    if layer_b64:
        # Flagged "static": these are decompositions of the whole light
        # stack, so they cannot follow the dome and a user is right to
        # expect an explanation rather than a dead-feeling button.
        _layer_buttons = ('\n  <div class="sep"></div>'
            '\n  <div class="lgrouphdr">Static maps &mdash; no relighting</div>\n  '
            + "\n  ".join(
                f'<button class="mbtn lbtn" id="L{i}" onclick="setLayer({i})" '
                f'title="{lbl} — computed from the whole light stack, so it does '
                f'not change with the light">{lbl}</button>'
                for i, (lbl, _u) in enumerate(layer_b64)))
        # NOTE: single braces. This string is *interpolated into* the
        # f-string template, it is not part of it - so the {{ }} doubling the
        # template needs would survive verbatim into the output and produce
        # `var BAKED=[{{src:'...'}}]`, a JavaScript syntax error that kills
        # the whole <script> block. The HTML and CSS still render, so the
        # symptom is a viewer with working menus and a blank canvas.
        _baked_js = ("var BAKED=["
            + ",".join("{src:'%s'}" % u for _l, u in layer_b64) + "];")

    if progress_cb is not None:
        progress_cb("Encoding roughness…", 0.5)
    if roughness is not None:
        rgh = roughness
        if rgh.ndim == 3:
            rgh = rgh[..., 0]
        rgh_full = cv2.resize(
            rgh.astype(np.uint8), (w, h), cv2.INTER_AREA
        )[y1:y2, x1:x2]
        if sc < 1.0:
            rgh_full = cv2.resize(rgh_full, (dw, dh), cv2.INTER_AREA)
        rgh_arr = (
            cv2.cvtColor(rgh_full, cv2.COLOR_GRAY2RGB)
            if rgh_full.ndim == 2
            else rgh_full
        )
        rgh_b64 = "data:image/jpeg;base64," + _b64_jpg(rgh_arr)
    else:
        rgh_b64 = "data:image/jpeg;base64," + _b64_jpg(
            np.full((dh, dw, 3), 128, np.uint8)
        )

    if progress_cb is not None:
        progress_cb("Encoding specular…", 0.65)
    if spec_var is not None:
        sp = (
            cv2.resize(
                spec_var.astype(np.float32), (w, h), cv2.INTER_AREA
            )[y1:y2, x1:x2]
            if spec_var.shape != (h, w)
            else spec_var[y1:y2, x1:x2]
        )
        sp_u8 = (np.clip(sp, 0, 1) * 255).astype(np.uint8)
        if sc < 1.0:
            sp_u8 = cv2.resize(sp_u8, (dw, dh), cv2.INTER_AREA)
        sp_b64 = "data:image/jpeg;base64," + _b64_jpg(
            cv2.cvtColor(sp_u8, cv2.COLOR_GRAY2RGB)
        )
    else:
        sp_b64 = "data:image/jpeg;base64," + _b64_jpg(
            np.full((dh, dw, 3), 40, np.uint8)
        )

    if meta is not None:
        ppmm = float(meta.get("scale_mmpx", 0) or 0)
    else:
        ppmm = 0.0
    scale_note = f"{1/ppmm:.1f} px/mm" if ppmm > 0 else "uncalibrated"

    if progress_cb is not None:
        progress_cb("Writing HTML viewer…", 0.85)

    _safe_id = html.escape(obj_id)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Helio RTI — {_safe_id}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{width:100%;height:100%;background:#080c12;overflow:hidden;font-family:Arial,sans-serif;color:#ddd}}
#wrap{{position:fixed;top:0;left:0;right:230px;bottom:0;display:flex;align-items:center;justify-content:center;overflow:hidden}}
#cv{{display:block;max-width:100%;max-height:100%;width:auto;height:auto;transform-origin:center center;cursor:crosshair}}
#bar{{position:fixed;top:0;right:0;bottom:0;width:230px;background:rgba(6,14,22,.96);border-left:1px solid #1a4a2a;overflow-y:auto;padding:10px 12px;z-index:100}}
h3{{color:#4ade80;font-size:12px;margin-bottom:4px;letter-spacing:.4px}}
.sub{{color:#3a6a5a;font-size:9px;margin-bottom:6px}}
#dome{{cursor:crosshair;border-radius:50%;border:1px solid #1a4a2a;display:block;margin:0 auto}}
#readout{{text-align:center;color:#f9e547;font-size:10px;margin:2px 0 5px}}
.mbtn{{display:block;width:100%;padding:4px 7px;margin:2px 0;background:#0d1a12;color:#8a9;border:1px solid #1a4a2a;border-radius:3px;cursor:pointer;font-size:10px;text-align:left}}
.mbtn.on,.mbtn:hover{{background:#1a4a2a;color:#4ade80;border-color:#4ade80}}
.sep{{border-top:1px solid #1a3a2a;margin:6px 0}}
.slg{{display:flex;align-items:center;gap:4px;margin:3px 0;font-size:10px;color:#8899aa}}
.slg label{{width:58px;flex-shrink:0}}
.slg input[type=range]{{flex:1;accent-color:#4ade80}}
.slg .val{{width:30px;text-align:right;color:#4ade80;font-weight:bold;font-size:10px}}
#zbar{{position:fixed;bottom:6px;left:0;right:230px;display:flex;align-items:center;justify-content:center;gap:5px;z-index:50;pointer-events:none}}
#zbar button{{background:rgba(6,14,22,.85);border:1px solid #1a4a2a;color:#4ade80;font-size:13px;padding:2px 8px;border-radius:11px;cursor:pointer;pointer-events:all}}
#titlebar{{position:absolute;left:14px;top:12px;font:600 13px/1.3 system-ui,-apple-system,sans-serif;
  color:#cfe6f5;background:rgba(6,14,22,.7);border:1px solid #1a4a2a;padding:4px 10px;
  border-radius:5px;pointer-events:none;max-width:52vw;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
#byline{{font:11px/1.35 system-ui,-apple-system,sans-serif;color:#7fd8a0;
  margin:-4px 0 8px;letter-spacing:.02em}}
.lgrouphdr{{font:10px/1.4 ui-monospace,Menlo,monospace;color:#6f8a7c;
  letter-spacing:.12em;text-transform:uppercase;margin:2px 0 3px}}
#zlbl{{color:#4ade80;font-size:10px;min-width:38px;text-align:center;background:rgba(6,14,22,.7);padding:1px 5px;border-radius:7px}}
#hint{{position:fixed;bottom:32px;right:238px;font-size:9px;color:#334;pointer-events:none}}
#expbtn{{width:100%;margin-top:6px;padding:5px;background:#12321e;color:#4ade80;border:1px solid #1a4a2a;border-radius:4px;cursor:pointer;font-size:10px}}
#expbtn:hover{{background:#1a4a2a}}
</style>
</head>
<body>
<div id="wrap"><canvas id="cv"></canvas></div>
<div id="bar">
  <h3>Helio RTI Viewer</h3>
  <div id="byline">by Scott Maloney</div>
  <div class="sub">{_safe_id} &nbsp;·&nbsp; {scale_note} &nbsp;·&nbsp; {dw}×{dh}px</div>
  <canvas id="dome" width="110" height="110"></canvas>
  <div id="readout">Az: 315° | El: 45°</div>
  <div class="sep"></div>
  <button class="mbtn on" id="m0" onclick="setMode(0)">Colour Relight</button>
  <button class="mbtn"    id="m1" onclick="setMode(1)">Greyscale RTI</button>
  <button class="mbtn"    id="m2" onclick="setMode(2)">Specular Enhance</button>
  <button class="mbtn"    id="m3" onclick="setMode(3)">Normal Map</button>
  <button class="mbtn"    id="m4" onclick="setMode(4)">Normals USM</button>
  <button class="mbtn"    id="m5" onclick="setMode(5)">Surface Gradient</button>
  <button class="mbtn"    id="m6" onclick="setMode(6)">Col × Specular</button>
  <button class="mbtn"    id="m7" onclick="setMode(7)">Roughness Fusion</button>
  <button class="mbtn"    id="m8" onclick="setMode(8)">RTI Spec Enhance</button>
  <div class="sep"></div>
  <button class="mbtn" id="bSplit" onclick="setSplit(this.className.indexOf('on')<0)"
          title="Show two modes side by side under one light — the left is the mode selected above">⬌ Split Compare</button>
  <div class="slg" id="cmpRow" style="display:none">
    <label>Right</label>
    <select id="cmpB" onchange="setModeB(this.value)"
            style="flex:1;background:#18191C;color:#ECEAE5;border:1px solid #36383D;border-radius:6px;font-size:12px;padding:4px 6px">
      <option value="0">Colour Relight</option><option value="1" selected>Greyscale RTI</option>
      <option value="2">Specular Enhance</option><option value="3">Normal Map</option>
      <option value="4">Normals USM</option><option value="5">Surface Gradient</option>
      <option value="6">Col × Specular</option><option value="7">Roughness Fusion</option>
      <option value="8">RTI Spec Enhance</option>
    </select>
  </div>
  <button class="mbtn" id="bSweep" onclick="toggleSweep()"
          title="Turn the light once around the object at the current elevation">▶ Raking Sweep</button>
  {_layer_buttons}
  <div class="sep"></div>
  <div class="slg"><label>Bright</label><input type="range" id="sBr" min="0.2" max="3" step="0.05" value="1" oninput="upd()"><span class="val" id="vBr">1.00</span></div>
  <div class="slg"><label>Contrast</label><input type="range" id="sCo" min="0.2" max="3" step="0.05" value="1" oninput="upd()"><span class="val" id="vCo">1.00</span></div>
  <div class="slg"><label>Gamma</label><input type="range" id="sGa" min="0.3" max="3" step="0.05" value="1" oninput="upd()"><span class="val" id="vGa">1.00</span></div>
  <div class="slg"><label>Specular</label><input type="range" id="sSp" min="0" max="4" step="0.05" value="1" oninput="upd()"><span class="val" id="vSp">1.00</span></div>
  <div class="slg"><label>Fill light</label><input type="range" id="sAm" min="0" max="0.6" step="0.01" value="0.12" oninput="upd()"><span class="val" id="vAm">0.12</span></div>
  <div class="slg"><label>Roughness mix</label><input type="range" id="sRo" min="0" max="1" step="0.02" value="0.5" oninput="upd()"><span class="val" id="vRo">0.50</span></div>
  <div class="slg"><label>Spec coeff</label><input type="range" id="sSc" min="0.2" max="5" step="0.05" value="1.5" oninput="upd()"><span class="val" id="vSc">1.50</span></div>
  <div class="slg"><label>Spec floor</label><input type="range" id="sSf" min="0" max="0.3" step="0.005" value="0.05" oninput="upd()"><span class="val" id="vSf">0.05</span></div>
  <div class="slg"><label>Spec blend</label><input type="range" id="sSb" min="0" max="1" step="0.02" value="0.6" oninput="upd()"><span class="val" id="vSb">0.60</span></div>
  <div class="slg"><label>Light Elev</label><input type="range" id="sEl" min="5" max="85" step="5" value="45" oninput="setElev(this.value)"><span class="val" id="vEl">45°</span></div>
  <div class="slg"><label>Light Az</label><input type="range" id="sAz" min="0" max="360" step="5" value="315" oninput="setAz(this.value)"><span class="val" id="vAz">315°</span></div>
  <button id="expbtn" onclick="exportPNG()">📷 Export PNG at this light</button>
</div>
<div id="zbar">
  <button onclick="doZoom(-1)">&#8722;</button>
  <span id="zlbl">100%</span>
  <button onclick="doZoom(1)">&#43;</button>
  <button onclick="resetView()" style="font-size:10px">Reset</button>
</div>
<div id="titlebar">{obj_id}</div>
<div id="hint">Drag dome = relight · Scroll = zoom · Space+drag = pan · Dbl-click = reset</div>
<script>
var DW={dw}, DH={dh};
var cv=document.getElementById('cv');
cv.width=DW; cv.height=DH;
var gl=cv.getContext('webgl')||cv.getContext('experimental-webgl');
if(!gl){{document.getElementById('wrap').innerHTML='<p style="color:#FF8F6E;padding:40px;font:16px system-ui,sans-serif">WebGL not supported</p>';}}
else{{ gl.getExtension('OES_standard_derivatives'); init(); }}

function init(){{
var VS='attribute vec2 a;varying vec2 uv;void main(){{uv=a*.5+.5;uv.y=1.-uv.y;gl_Position=vec4(a,0,1);}}';
{_baked_js}
var FS='#extension GL_OES_standard_derivatives : enable\\nprecision highp float;\\n'
+'varying vec2 uv;uniform sampler2D uA,uN,uR,uS;uniform vec3 uL;uniform float uMa,uBr,uCo,uGa,uSp,uAm,uRo,uSc,uSf,uSb;uniform sampler2D uB;\\n'
+'uniform float uM2,uSplit;\\n'
+'vec3 fc(float t){{t=clamp(t,0.,1.);if(t<.25)return mix(vec3(0,0,.5),vec3(0,.8,.8),t*4.);if(t<.5)return mix(vec3(0,.8,.8),vec3(0,.8,0),(t-.25)*4.);if(t<.75)return mix(vec3(0,.8,0),vec3(1,1,0),(t-.5)*4.);return mix(vec3(1,1,0),vec3(1,0,0),(t-.75)*4.);}}\\n'
+'void main(){{\\n'
+'  float uM = (uSplit>0.0 && uv.x>uSplit) ? uM2 : uMa;\\n'
+'  if(uv.x<0.||uv.x>1.||uv.y<0.||uv.y>1.){{gl_FragColor=vec4(.06,.08,.12,1.);return;}}\\n'
+'  vec3 n=normalize(texture2D(uN,uv).rgb*2.-1.),alb=texture2D(uA,uv).rgb;\\n'
+'  float rgh=texture2D(uR,uv).r,spc=texture2D(uS,uv).r;\\n'
+'  vec3 L=normalize(uL),H=normalize(L+vec3(0,0,1));\\n'
+'  float NL=max(dot(n,L),0.),NH=max(dot(n,H),0.);\\n'
+'  float sp=spc*pow(NH,mix(128.,2.,rgh))*uSp;\\n'
+'  float lum=dot(alb,vec3(.299,.587,.114));\\n'
+'  vec3 col;\\n'
+'  if(uM<.5){{col=alb*(NL*(1.-uAm)+uAm)+vec3(sp*.4);}}\\n'
+'  else if(uM<1.5){{col=vec3(lum*(NL*(1.-uAm)+uAm));}}\\n'
+'  else if(uM<2.5){{col=vec3(lum*.2+sp*2.+NL*.5);}}\\n'
+'  else if(uM<3.5){{col=n*.5+.5;}}\\n'
+'  else if(uM<4.5){{\\n'
+'    vec2 dx=dFdx(uv)*3.,dy=dFdy(uv)*3.;\\n'
+'    vec3 nb=normalize((texture2D(uN,uv+dx)+texture2D(uN,uv-dx)+texture2D(uN,uv+dy)+texture2D(uN,uv-dy)+texture2D(uN,uv)*4.).rgb*.125)*2.-1.;\\n'
+'    col=vec3(max(dot(normalize(n+(n-nb)*2.5),L),0.));}}\\n'
+'  else if(uM<5.5){{float slp=length(vec2(n.x,n.y));float az=atan(n.y,n.x)/6.28318+.5;col=fc(az)*(slp*.9+.1);col=mix(col,vec3(NL*.8+.1),.25);float cx=dFdx(n.z),cy=dFdy(n.z);col=clamp(col*(1.+clamp(sqrt(cx*cx+cy*cy)*60.,0.,.6)),0.,1.);}}\\n'
+'  else if(uM<6.5){{col=alb*(NL*.9+.1)+vec3(sp*2.)*mix(vec3(1.),alb,.3);}}\\n'
+'  else if(uM<7.5){{float shd=lum*(NL*(1.-uAm)+uAm);col=vec3(shd*(1.-uRo)+rgh*uRo);}}\\n'
+'  else if(uM<8.5){{\\n'
+'    float df=NL;\\n'
+'    float pr=lum;\\n'
+'    float sr=max(0.,pr-df);\\n'
+'    float am=sr*(sr>uSf?1.:0.)*uSc;\\n'
+'    float o=df*(1.-uSb)+clamp(df+am,0.,1.)*uSb;\\n'
+'    col=vec3(o);}}\\n'
+'  else{{col=texture2D(uB,uv).rgb;}}\\n'
+'  if(uSplit>0.0&&abs(uv.x-uSplit)<0.0016) col=vec3(0.95);\\n'
+'  col=clamp(col*uBr,0.,1.);col=clamp((col-.5)*uCo+.5,0.,1.);\\n'
+'  col=pow(col,vec3(1./max(uGa,.01)));gl_FragColor=vec4(col,1.);}}';

function mkSh(t,s){{var sh=gl.createShader(t);gl.shaderSource(sh,s);gl.compileShader(sh);
  if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))console.error(gl.getShaderInfoLog(sh));return sh;}}
var pg=gl.createProgram();
gl.attachShader(pg,mkSh(gl.VERTEX_SHADER,VS));
gl.attachShader(pg,mkSh(gl.FRAGMENT_SHADER,FS));
gl.linkProgram(pg);
if(!gl.getProgramParameter(pg,gl.LINK_STATUS)){{console.error(gl.getProgramInfoLog(pg));return;}}

var buf=gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER,buf);
gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
var aLoc=gl.getAttribLocation(pg,'a');
gl.enableVertexAttribArray(aLoc);
gl.vertexAttribPointer(aLoc,2,gl.FLOAT,false,0,0);

var uL_=gl.getUniformLocation(pg,'uL'),uM_=gl.getUniformLocation(pg,'uMa');
var uM2_=gl.getUniformLocation(pg,'uM2'),uSplit_=gl.getUniformLocation(pg,'uSplit');
var mode2=1, splitAt=0.0, sweepTimer=null, splitDrag=false;
var uBr_=gl.getUniformLocation(pg,'uBr'),uCo_=gl.getUniformLocation(pg,'uCo'),uGa_=gl.getUniformLocation(pg,'uGa');
var uSp_=gl.getUniformLocation(pg,'uSp'),uAm_=gl.getUniformLocation(pg,'uAm');
var uRo_=gl.getUniformLocation(pg,'uRo');
var uSc_=gl.getUniformLocation(pg,'uSc'),uSf_=gl.getUniformLocation(pg,'uSf'),uSb_=gl.getUniformLocation(pg,'uSb');

function mkTex(dataURL,unit,uname){{
  var tex=gl.createTexture();
  gl.activeTexture(gl.TEXTURE0+unit); gl.bindTexture(gl.TEXTURE_2D,tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([100,100,100,255]));
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.useProgram(pg); gl.uniform1i(gl.getUniformLocation(pg,uname),unit);
  var img=new Image();
  img.onload=function(){{
    gl.activeTexture(gl.TEXTURE0+unit); gl.bindTexture(gl.TEXTURE_2D,tex);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);
    rdr();}};
  img.onerror=function(){{console.warn('Tex failed:',uname);rdr();}};
  img.src=dataURL;}}

mkTex('{alb_b64}',0,'uA');
mkTex('{nrm_b64}',1,'uN');
mkTex('{rgh_b64}',2,'uR');
mkTex('{sp_b64}', 3,'uS');

var lx=0,ly=0,lz=1,mode=0,zoom=1,panX=0,panY=0;
var spaceDown=false,panDrag=false,dX=0,dY=0,pX0=0,pY0=0;

var dome=document.getElementById('dome');
var dc=dome.getContext('2d'),DR=dome.width/2,dDrag=false;

function drawDome(dlx,dly){{
  dc.clearRect(0,0,dome.width,dome.height);
  var g=dc.createRadialGradient(DR,DR,0,DR,DR,DR);
  g.addColorStop(0,'#1A1B1E');g.addColorStop(1,'#1A1B1E');
  dc.fillStyle=g;dc.beginPath();dc.arc(DR,DR,DR,0,Math.PI*2);dc.fill();
  dc.strokeStyle='#3E4046';dc.lineWidth=1;
  [.33,.67,1].forEach(function(r){{dc.beginPath();dc.arc(DR,DR,r*DR,0,Math.PI*2);dc.stroke();}});
  for(var i=0;i<8;i++){{var a=i*Math.PI/4;dc.beginPath();dc.moveTo(DR,DR);dc.lineTo(DR+Math.cos(a)*DR,DR+Math.sin(a)*DR);dc.stroke();}}
  var m=Math.sqrt(dlx*dlx+dly*dly),sc2=m>.95?.95/m:1;
  var px=DR+dlx*DR*sc2,py=DR-dly*DR*sc2;
  dc.beginPath();dc.arc(px,py,7,0,Math.PI*2);
  dc.fillStyle='#E6A93E';dc.fill();dc.strokeStyle='#222326';dc.lineWidth=2;dc.stroke();}}

function setLight(az,el){{
  var ce=Math.cos(el),se=Math.sin(el);
  lx=ce*Math.sin(az);ly=ce*Math.cos(az);lz=se;
  var len=Math.sqrt(lx*lx+ly*ly+lz*lz);if(len>0){{lx/=len;ly/=len;lz/=len;}}
  var azD=Math.round(((az*180/Math.PI)%360+360)%360),elD=Math.round(el*180/Math.PI);
  document.getElementById('readout').textContent='Az: '+azD+'° | El: '+elD+'°';
  var e=document.getElementById('sEl');if(e)e.value=elD;
  var ev=document.getElementById('vEl');if(ev)ev.textContent=elD+'°';
  var a=document.getElementById('sAz');if(a)a.value=azD;
  var av=document.getElementById('vAz');if(av)av.textContent=azD+'°';
  drawDome(lx,ly); rdr();}}

function domeXY(px,py){{
  var dlx=(px-DR)/DR,dly=-(py-DR)/DR;
  var r=Math.min(1,Math.sqrt(dlx*dlx+dly*dly));
  var el=Math.max(3*Math.PI/180,Math.min(87*Math.PI/180,Math.acos(r)));
  setLight(Math.atan2(dlx,dly),el);}}

dome.addEventListener('mousedown',function(e){{e.preventDefault();e.stopPropagation();dDrag=true;domeXY(e.offsetX,e.offsetY);}});
window.addEventListener('mousemove',function(e){{if(!dDrag)return;var r=dome.getBoundingClientRect();domeXY(e.clientX-r.left,e.clientY-r.top);}});
window.addEventListener('mouseup',function(){{dDrag=false;}});
dome.addEventListener('touchstart',function(e){{e.preventDefault();e.stopPropagation();dDrag=true;var r=dome.getBoundingClientRect(),t=e.touches[0];domeXY(t.clientX-r.left,t.clientY-r.top);}},{{passive:false}});
dome.addEventListener('touchmove',function(e){{e.preventDefault();var r=dome.getBoundingClientRect(),t=e.touches[0];domeXY(t.clientX-r.left,t.clientY-r.top);}},{{passive:false}});
dome.addEventListener('touchend',function(){{dDrag=false;}});

function setElev(v){{setLight(Math.atan2(lx,ly),parseFloat(v)*Math.PI/180);}}
function setAz(v){{setLight(parseFloat(v)*Math.PI/180,Math.asin(Math.min(1,Math.abs(lz))));}}
function rdr(){{
  gl.viewport(0,0,DW,DH); gl.useProgram(pg);
  gl.uniform3f(uL_,lx,ly,lz); gl.uniform1f(uM_,mode);
  gl.uniform1f(uM2_,mode2); gl.uniform1f(uSplit_,splitAt);
  gl.uniform1f(uBr_,parseFloat(document.getElementById('sBr').value));
  gl.uniform1f(uCo_,parseFloat(document.getElementById('sCo').value));
  gl.uniform1f(uGa_,parseFloat(document.getElementById('sGa').value));
  var _sp=document.getElementById('sSp'),_am=document.getElementById('sAm');
  gl.uniform1f(uSp_,_sp?parseFloat(_sp.value):1.0);
  gl.uniform1f(uAm_,_am?parseFloat(_am.value):0.12);
  var _ro=document.getElementById('sRo');
  gl.uniform1f(uRo_,_ro?parseFloat(_ro.value):0.5);
  var _sc=document.getElementById('sSc'),_sf=document.getElementById('sSf'),_sb=document.getElementById('sSb');
  gl.uniform1f(uSc_,_sc?parseFloat(_sc.value):1.5);
  gl.uniform1f(uSf_,_sf?parseFloat(_sf.value):0.05);
  gl.uniform1f(uSb_,_sb?parseFloat(_sb.value):0.6);
  gl.drawArrays(gl.TRIANGLE_STRIP,0,4);}}
function upd(){{
  var _s1=document.getElementById('sSp'),_v1=document.getElementById('vSp');
  if(_s1&&_v1)_v1.textContent=parseFloat(_s1.value).toFixed(2);
  var _s2=document.getElementById('sAm'),_v2=document.getElementById('vAm');
  if(_s2&&_v2)_v2.textContent=parseFloat(_s2.value).toFixed(2);
  var _s3=document.getElementById('sRo'),_v3=document.getElementById('vRo');
  if(_s3&&_v3)_v3.textContent=parseFloat(_s3.value).toFixed(2);
  [['sSc','vSc'],['sSf','vSf'],['sSb','vSb']].forEach(function(pr){{
    var a=document.getElementById(pr[0]),b=document.getElementById(pr[1]);
    if(a&&b)b.textContent=parseFloat(a.value).toFixed(2);}});
  document.getElementById('vBr').textContent=parseFloat(document.getElementById('sBr').value).toFixed(2);
  document.getElementById('vCo').textContent=parseFloat(document.getElementById('sCo').value).toFixed(2);
  document.getElementById('vGa').textContent=parseFloat(document.getElementById('sGa').value).toFixed(2);
  rdr();}}
function setMode(m){{mode=m;clearLayer();
  var lb=document.querySelectorAll('.lbtn');for(var k=0;k<lb.length;k++)lb[k].classList.remove('on');
  for(var i=0;i<=10;i++){{var b=document.getElementById('m'+i);if(b)b.className='mbtn'+(i===m?' on':'');}}
  var s2=document.getElementById('cmpB');if(s2)s2.value=String(mode2);rdr();}}
// Split comparison. One draw call: the shader picks a branch per pixel,
// so the two halves are always the same frame under the same light —
// which is the whole point of comparing them.
function setSplit(on){{splitAt=on?(splitAt>0?splitAt:0.5):0.0;
  var b=document.getElementById('bSplit');if(b)b.className='mbtn'+(on?' on':'');
  var r=document.getElementById('cmpRow');if(r)r.style.display=on?'flex':'none';rdr();}}
function setModeB(v){{mode2=parseInt(v,10);rdr();}}
// Raking sweep: one full turn of the light. The elevation stays where the
// user put it, because a sweep that also changed the angle of incidence
// would confound the two things it is meant to separate.
function toggleSweep(){{
  var b=document.getElementById('bSweep');
  if(sweepTimer){{clearInterval(sweepTimer);sweepTimer=null;if(b){{b.textContent='\u25b6 Raking Sweep';b.className='mbtn';}}return;}}
  if(b){{b.textContent='\u2759\u2759 Stop Sweep';b.className='mbtn on';}}
  // Driven through the azimuth slider's own handler, so the slider, the dome
  // and the readout all stay in step. The first version invented `az` and
  // `alt` locals, which belong to a different viewer in this file — they were
  // undefined here, the arithmetic produced NaN, and the sweep silently did
  // nothing while reporting itself as running.
  sweepTimer=setInterval(function(){{
    var s=document.getElementById('sAz');if(!s)return;
    var v=(parseFloat(s.value)+4)%360;
    s.value=String(v); setAz(v);}},70);}}
function setZL(){{document.getElementById('zlbl').textContent=Math.round(zoom*100)+'%';}}
function applyT(){{setZL();cv.style.transform='scale('+zoom+') translate('+panX/zoom+'px,'+panY/zoom+'px)';}}
function doZoom(d){{zoom=Math.max(.1,Math.min(20,zoom*Math.pow(1.25,d)));applyT();}}
function resetView(){{zoom=1;panX=0;panY=0;setLight(315*Math.PI/180,45*Math.PI/180);applyT();}}
function exportPNG(){{rdr();var a=document.createElement('a');a.download='RTI_export.png';a.href=cv.toDataURL('image/png');a.click();}}
document.addEventListener('keydown',function(e){{if(e.code==='Space'){{spaceDown=true;e.preventDefault();}}}});
document.addEventListener('keyup',function(e){{if(e.code==='Space'){{spaceDown=false;panDrag=false;}}}});
document.getElementById('wrap').addEventListener('wheel',function(e){{e.preventDefault();zoom=Math.max(.1,Math.min(20,zoom*(e.deltaY<0?1.12:1/1.12)));applyT();}},{{passive:false}});
cv.addEventListener('mousedown',function(e){{if(dDrag)return;if(e.button===1||spaceDown){{panDrag=true;dX=e.clientX;dY=e.clientY;pX0=panX;pY0=panY;e.preventDefault();}}}});
window.addEventListener('mousemove',function(e){{if(panDrag){{panX=pX0+(e.clientX-dX);panY=pY0+(e.clientY-dY);applyT();}}}});
window.addEventListener('mouseup',function(e){{if(e.button===1||spaceDown)panDrag=false;}});
function splitFromEvent(e){{var r=cv.getBoundingClientRect();
  splitAt=Math.max(.02,Math.min(.98,(e.clientX-r.left)/r.width));rdr();}}
cv.addEventListener('mousedown',function(e){{
  if(splitAt>0&&!spaceDown&&e.button===0&&!dDrag){{splitDrag=true;splitFromEvent(e);e.preventDefault();}}}});
window.addEventListener('mousemove',function(e){{if(splitDrag)splitFromEvent(e);}});
window.addEventListener('mouseup',function(){{splitDrag=false;}});
cv.addEventListener('dblclick',resetView);
var _bakedTex=null,_bakedLoaded=-1;
function setLayer(i){{
  // Upload the layer as a texture and switch the shader to mode 8. As an
  // <img> overlay it covered the whole window, so it ignored the canvas zoom
  // and pan and appeared at a different size to every other mode.
  if(!BAKED[i])return;
  var btns=document.querySelectorAll('.mbtn');
  for(var k=0;k<btns.length;k++)btns[k].classList.remove('on');
  var b=document.getElementById('L'+i); if(b)b.classList.add('on');
  if(!_bakedTex){{
    _bakedTex=gl.createTexture();
    gl.activeTexture(gl.TEXTURE0+4); gl.bindTexture(gl.TEXTURE_2D,_bakedTex);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([100,100,100,255]));
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.useProgram(pg); gl.uniform1i(gl.getUniformLocation(pg,'uB'),4);
  }}
  if(_bakedLoaded===i){{mode=9;rdr();return;}}
  var im=new Image();
  im.onload=function(){{
    gl.activeTexture(gl.TEXTURE0+4); gl.bindTexture(gl.TEXTURE_2D,_bakedTex);
    // Every other texture here goes through mkTex, which never touches
    // UNPACK_FLIP_Y_WEBGL and so uploads at the WebGL default of false.
    // This path used to set it true, and nothing on the Python side
    // compensated - the baked layers are written top-down exactly like
    // albedo and normals - so PCA 1-3 arrived vertically mirrored against
    // the very maps they exist to be compared with. Set explicitly rather
    // than left to the default, because the default is what made an
    // asymmetry between two upload paths invisible in the first place.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,im);
    _bakedLoaded=i; mode=9; rdr();
  }};
  im.src=BAKED[i].src;
}}
function clearLayer(){{}}
window.setLayer=setLayer;
window.setMode=setMode;window.setElev=setElev;window.setAz=setAz;
window.doZoom=doZoom;window.resetView=resetView;window.upd=upd;window.exportPNG=exportPNG;
// These are called from onclick/onchange attributes, which resolve in the
// global scope — everything here lives inside init()'s closure, so a
// handler that is not exported fails with "not defined" the first time a
// user clicks it, and only when a user clicks it.
window.setSplit=setSplit;window.setModeB=setModeB;window.toggleSweep=toggleSweep;
setLight(315*Math.PI/180,45*Math.PI/180);
}} // end init()
</script></body></html>"""

    from helio.application.phone_web import rti_viewer
    html_doc = rti_viewer(html_doc, "} // end init()")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    if progress_cb is not None:
        progress_cb("Combined RTI viewer written", 1.0)

    return out_path


def export_normalmap_rti_viewer_html(
    out_dir: str,
    normals: np.ndarray,
    albedo: np.ndarray,
    crop_coords: tuple[int, int, int, int],
    obj_id: str = "Object",
    meta: dict | None = None,
    max_side: int = RTI_VIEWER_MAX_SIDE,
    progress_cb: Callable[[str, float | None], None] | None = None,
) -> str:
    """Export a WebGL RTI viewer using PS normal map + albedo as standalone files.

    Writes *index.html*, *normals.png*, *albedo.png*, and *README.txt* to
    *out_dir*.  The HTML embeds both textures as base64 data URIs so it
    works from ``file://``.

    Returns the path to *index.html*.
    """
    import io as _io
    import base64 as _b64
    import PIL.Image as _PIL
    import html

    x1, y1, x2, y2 = crop_coords[:4]

    if progress_cb is not None:
        progress_cb("Exporting normal map…", 0.2)

    nrm = normals[y1:y2, x1:x2].astype(np.float32)
    nlen = np.linalg.norm(nrm, axis=-1, keepdims=True)
    nrm /= np.maximum(nlen, 1e-8)
    nrm_u8 = ((nrm * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

    alb = albedo[y1:y2, x1:x2]
    Hc, Wc = nrm.shape[:2]

    if max(Hc, Wc) > max_side:
        sc = max_side / max(Hc, Wc)
        dh = int(Hc * sc)
        dw = int(Wc * sc)
        nrm_u8 = cv2.resize(nrm_u8, (dw, dh), cv2.INTER_AREA)
        alb = cv2.resize(alb, (dw, dh), cv2.INTER_AREA)
    else:
        dh, dw = Hc, Wc

    if progress_cb is not None:
        progress_cb("Encoding textures as base64…", 0.5)

    def _to_b64_png(arr_u8):
        buf = _io.BytesIO()
        _PIL.fromarray(arr_u8).save(buf, format="PNG", optimize=False)
        return _b64.b64encode(buf.getvalue()).decode()

    nrm_b64 = _to_b64_png(nrm_u8)
    alb_b64 = _to_b64_png(alb)
    nrm_data_url = f"data:image/png;base64,{nrm_b64}"
    alb_data_url = f"data:image/png;base64,{alb_b64}"

    _PIL.fromarray(nrm_u8).save(os.path.join(out_dir, "normals.png"), optimize=False)
    _PIL.fromarray(alb).save(os.path.join(out_dir, "albedo.png"), optimize=False)

    if meta is not None:
        ppmm = float(meta.get("scale_mmpx", 0) or 0)
    else:
        ppmm = 0.0
    scale_note = f"{1/ppmm:.1f} px/mm" if ppmm > 0 else "uncalibrated"

    if progress_cb is not None:
        progress_cb("Writing viewer HTML…", 0.85)

    _safe_id = html.escape(obj_id)
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Helio — PS Normal-Map RTI: {_safe_id}</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ width:100%; height:100%; background:#080c12; overflow:hidden; user-select:none; }}
#wrap {{
  position:fixed; top:0; left:0; width:100%; height:100%;
  display:flex; align-items:center; justify-content:center;
  overflow:hidden;
}}
#c {{
  display:block;
  max-width:100%;
  max-height:100%;
  width:auto; height:auto;
  transform-origin: center center;
  cursor:grab;
}}
#c:active {{ cursor:grabbing; }}
#ui {{
  position:fixed; top:12px; right:12px; z-index:200;
  background:rgba(6,14,22,0.93);
  border:1px solid #1a4a2a; border-radius:10px;
  padding:14px 16px; width:250px;
  font-family: Arial, sans-serif;
}}
#ui h3 {{ color:#4ade80; font-size:13px; margin-bottom:8px; letter-spacing:.4px; }}
#dome-wrap {{ text-align:center; margin-bottom:6px; }}
#dome {{ cursor:crosshair; border-radius:50%; border:1px solid #1a4a2a; }}
#readout {{ text-align:center; color:#f9e547; font-size:11px; margin:3px 0 8px; }}
.row {{ display:flex; align-items:center; gap:6px; margin:4px 0; font-size:11px; color:#8899aa; }}
.row label {{ width:82px; flex-shrink:0; }}
.row input[type=range] {{ flex:1; accent-color:#4ade80; height:14px; cursor:pointer; }}
.row .val {{ width:32px; text-align:right; color:#4ade80; font-weight:bold; font-size:11px; }}
.sep {{ border-top:1px solid #1a3a2a; margin:8px 0; }}
#export-btn {{
  width:100%; margin-top:6px; padding:7px 0;
  background:#12321e; color:#4ade80;
  border:1px solid #1a4a2a; border-radius:5px;
  cursor:pointer; font-size:11px;
}}
#export-btn:hover {{ background:#1a4a2a; }}
.mode-btn {{
  flex:1 1 calc(50% - 4px); padding:5px 4px;
  background:#0d1a26; color:#7ec8e3; border:1px solid #1a3a4a;
  border-radius:5px; cursor:pointer; font-size:10px;
}}
.mode-btn:hover {{ background:#1a3a50; }}
.mode-btn.active {{ background:#1a5276; border-color:#5dade2; color:#fff; }}
#hints {{
  position:fixed; bottom:8px; left:10px;
  font-size:10px; color:#334; font-family:Arial,sans-serif;
  pointer-events:none;
}}
</style>
</head>
<body>

<div id="wrap"><canvas id="c"></canvas></div>

<div id="ui">
  <h3>PS Normal-Map RTI</h3>
  <div id="dome-wrap">
    <canvas id="dome" width="120" height="120"></canvas>
  </div>
  <div id="readout">Az: 315° | El: 45°</div>

  <div class="sep"></div>

  <div id="mode-row" style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px;">
    <button class="mode-btn active" data-mode="0">Relit</button>
    <button class="mode-btn" data-mode="1">Greyscale</button>
    <button class="mode-btn" data-mode="2">Specular</button>
    <button class="mode-btn" data-mode="3">Normal Map</button>
    <button class="mode-btn" data-mode="4">Albedo</button>
  </div>

  <div class="sep"></div>

  <div class="row"><label>Gain</label>
    <input type="range" id="s-gain" min="0.1" max="5" step="0.05" value="1.0">
    <span class="val" id="v-gain">1.00</span></div>

  <div class="row"><label>Specular</label>
    <input type="range" id="s-spec" min="0" max="3" step="0.05" value="0.0">
    <span class="val" id="v-spec">0.00</span></div>

  <div class="row"><label>Gamma</label>
    <input type="range" id="s-gam" min="0.3" max="2.5" step="0.05" value="1.0">
    <span class="val" id="v-gam">1.00</span></div>

  <div class="row"><label>Colour mix</label>
    <input type="range" id="s-cmix" min="0" max="1" step="0.01" value="0.0">
    <span class="val" id="v-cmix">0.00</span></div>

  <div class="sep"></div>

  <div class="row"><label>Zoom</label>
    <input type="range" id="s-zoom" min="0.2" max="10" step="0.1" value="1.0">
    <span class="val" id="v-zoom">1.0×</span></div>

  <button id="export-btn" onclick="exportPNG()">Export PNG at this light</button>
</div>

<div id="hints">
  Helio &nbsp;|&nbsp; {_safe_id} &nbsp;|&nbsp; {scale_note} &nbsp;|&nbsp; {dw}×{dh}px<br>
  Scroll = zoom &nbsp;·&nbsp; Drag = pan &nbsp;·&nbsp; Dbl-click = reset &nbsp;·&nbsp; Dome = light
</div>

<script>
const IMG_W = {dw}, IMG_H = {dh};

const canvas = document.getElementById('c');
canvas.width  = IMG_W;
canvas.height = IMG_H;

const gl = canvas.getContext('webgl', {{
  antialias: true,
  preserveDrawingBuffer: true,
}});
if (!gl) {{ document.body.innerHTML = '<p style="color:#FF8F6E;padding:40px;font:16px system-ui,sans-serif">WebGL not supported</p>'; }}

const VS = `
attribute vec2 aPos;
varying vec2 vUV;
void main() {{
  vUV = aPos * 0.5 + 0.5;
  vUV.y = 1.0 - vUV.y;
  gl_Position = vec4(aPos, 0.0, 1.0);
}}`;

const FS = `
precision highp float;
uniform sampler2D uNorm;
uniform sampler2D uAlb;
uniform vec3  uL;
uniform float uGain;
uniform float uSpec;
uniform float uGam;
uniform float uCmix;
uniform int   uMode;
varying vec2 vUV;

void main() {{
  vec3 N = normalize(texture2D(uNorm, vUV).rgb * 2.0 - 1.0);
  vec3 alb = texture2D(uAlb, vUV).rgb;

  // Mode 3: raw normal map visualisation, mode 4: plain albedo — no relighting math needed.
  if (uMode == 3) {{ gl_FragColor = vec4(N * 0.5 + 0.5, 1.0); return; }}
  if (uMode == 4) {{ gl_FragColor = vec4(alb, 1.0); return; }}

  float diff = max(0.0, dot(N, uL));
  float sp = 0.0;
  vec3 H = normalize(uL + vec3(0.0, 0.0, 1.0));
  sp = pow(max(0.0, dot(N, H)), 32.0) * max(uSpec, 0.6);

  if (uMode == 2) {{
    // Specular-only: isolate the highlight response, ignore diffuse term.
    gl_FragColor = vec4(vec3(clamp(sp * uGain, 0.0, 1.0)), 1.0);
    return;
  }}

  sp *= (uSpec > 0.001) ? 1.0 : 0.0;
  float I = clamp((diff + sp) * uGain, 0.0, 1.0);
  I = pow(I, 1.0 / max(uGam, 0.01));
  vec3 grey = vec3(I);

  if (uMode == 1) {{ gl_FragColor = vec4(grey, 1.0); return; }}

  vec3 col = mix(grey, alb * I, uCmix);
  gl_FragColor = vec4(col, 1.0);
}}`;

function compileShader(type, src) {{
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    console.error('Shader error:', gl.getShaderInfoLog(s));
  return s;
}}

const prog = gl.createProgram();
gl.attachShader(prog, compileShader(gl.VERTEX_SHADER,   VS));
gl.attachShader(prog, compileShader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
gl.useProgram(prog);

const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, 'aPos');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

const uL    = gl.getUniformLocation(prog, 'uL');
const uGain = gl.getUniformLocation(prog, 'uGain');
const uSpec = gl.getUniformLocation(prog, 'uSpec');
const uGam  = gl.getUniformLocation(prog, 'uGam');
const uCmix = gl.getUniformLocation(prog, 'uCmix');
const uMode = gl.getUniformLocation(prog, 'uMode');
const uNorm = gl.getUniformLocation(prog, 'uNorm');
const uAlb  = gl.getUniformLocation(prog, 'uAlb');

let gain = 1.0, spec = 0.0, gam = 1.0, cmix = 0.0, viewMode = 0;
let lx = 0.0, ly = 0.0, lz = 1.0;

function loadTex(dataURL, unit) {{
  const tex = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
                new Uint8Array([180, 100, 100, 255]));
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  const img = new Image();
  img.onload = () => {{
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.generateMipmap(gl.TEXTURE_2D);
    render();
  }};
  img.src = dataURL;
  return tex;
}}

loadTex('{nrm_data_url}', 0);
loadTex('{alb_data_url}', 1);
gl.uniform1i(uNorm, 0);
gl.uniform1i(uAlb,  1);

function render() {{
  gl.viewport(0, 0, IMG_W, IMG_H);
  gl.uniform3f(uL,    lx, ly, lz);
  gl.uniform1f(uGain, gain);
  gl.uniform1f(uSpec, spec);
  gl.uniform1f(uGam,  gam);
  gl.uniform1f(uCmix, cmix);
  gl.uniform1i(uMode, viewMode);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}}

document.querySelectorAll('.mode-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    viewMode = parseInt(btn.dataset.mode, 10);
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    render();
  }});
}});

let zoom = 1.0;
let panX = 0, panY = 0;

function applyTransform() {{
  canvas.style.transform =
    `scale(${{zoom}}) translate(${{panX / zoom}}px, ${{panY / zoom}}px)`;
}}

document.getElementById('wrap').addEventListener('wheel', e => {{
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.15 : 0.87;
  zoom = Math.max(0.2, Math.min(20, zoom * factor));
  const zs = document.getElementById('s-zoom');
  zs.value = Math.min(parseFloat(zs.max), zoom);
  document.getElementById('v-zoom').textContent = zoom.toFixed(1) + '×';
  applyTransform();
}}, {{ passive: false }});

let dragging = false, dragX0 = 0, dragY0 = 0, panX0 = 0, panY0 = 0;
canvas.addEventListener('mousedown', e => {{
  if (draggingDome) return;
  dragging = true; dragX0 = e.clientX; dragY0 = e.clientY;
  panX0 = panX; panY0 = panY;
}});
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  panX = panX0 + (e.clientX - dragX0);
  panY = panY0 + (e.clientY - dragY0);
  applyTransform();
}});
window.addEventListener('mouseup', () => dragging = false);

canvas.addEventListener('dblclick', () => {{
  zoom = 1; panX = 0; panY = 0;
  document.getElementById('s-zoom').value = 1;
  document.getElementById('v-zoom').textContent = '1.0×';
  applyTransform();
}});

let t0 = null, t1 = null, tz0 = 1;
canvas.addEventListener('touchstart', e => {{
  if (draggingDome) return;
  e.preventDefault();
  if (e.touches.length === 1) {{
    t0 = e.touches[0]; panX0 = panX; panY0 = panY;
  }} else if (e.touches.length === 2) {{
    t1 = e.touches[1]; tz0 = zoom;
  }}
}}, {{ passive:false }});
canvas.addEventListener('touchmove', e => {{
  if (draggingDome) return;
  e.preventDefault();
  if (e.touches.length === 1 && t0) {{
    panX = panX0 + e.touches[0].clientX - t0.clientX;
    panY = panY0 + e.touches[0].clientY - t0.clientY;
    applyTransform();
  }} else if (e.touches.length === 2 && t1) {{
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    const dx0= t0.clientX - t1.clientX, dy0= t0.clientY - t1.clientY;
    const dist  = Math.hypot(dx, dy);
    const dist0 = Math.hypot(dx0, dy0);
    zoom = Math.max(0.2, Math.min(20, tz0 * dist / (dist0 || 1)));
    applyTransform();
  }}
}}, {{ passive:false }});

const dome = document.getElementById('dome');
const dc   = dome.getContext('2d');
const DR   = dome.width / 2;
let draggingDome = false;

function drawDome(dlx, dly) {{
  dc.clearRect(0, 0, dome.width, dome.height);
  const g = dc.createRadialGradient(DR,DR,0, DR,DR,DR);
  g.addColorStop(0, '#1A1B1E'); g.addColorStop(1, '#1A1B1E');
  dc.fillStyle = g;
  dc.beginPath(); dc.arc(DR,DR,DR,0,Math.PI*2); dc.fill();
  dc.strokeStyle='#3E4046'; dc.lineWidth=1;
  [0.33,0.67,1.0].forEach(r => {{
    dc.beginPath(); dc.arc(DR,DR,r*DR,0,Math.PI*2); dc.stroke();
  }});
  for (let i=0;i<8;i++) {{
    const a=i*Math.PI/4;
    dc.beginPath(); dc.moveTo(DR,DR);
    dc.lineTo(DR+Math.cos(a)*DR, DR+Math.sin(a)*DR); dc.stroke();
  }}
  const px=DR+dlx*DR, py=DR+dly*DR;
  dc.beginPath(); dc.arc(px,py,8,0,Math.PI*2);
  dc.fillStyle='#E6A93E'; dc.fill();
  dc.strokeStyle='#222326'; dc.lineWidth=2; dc.stroke();
}}

function domeToLight(px, py) {{
  const dlx=(px-DR)/DR, dly=(py-DR)/DR;
  const r=Math.min(1, Math.sqrt(dlx*dlx+dly*dly));
  const el=Math.acos(r), az=Math.atan2(dly, dlx);
  lx = Math.sin(el)*Math.cos(az);
  ly = Math.sin(el)*Math.sin(az);
  lz = Math.cos(el);
  const len = Math.sqrt(lx*lx+ly*ly+lz*lz);
  lx/=len; ly/=len; lz/=len;
  drawDome(dlx, dly);
  const azDeg = ((Math.atan2(ly,lx)*180/Math.PI)+360)%360;
  const elDeg = Math.asin(Math.min(1,Math.abs(lz)))*180/Math.PI;
  document.getElementById('readout').textContent =
    `Az: ${{azDeg.toFixed(0)}}° | El: ${{elDeg.toFixed(0)}}°`;
  render();
}}

dome.addEventListener('mousedown', e => {{
  e.stopPropagation();
  draggingDome=true; domeToLight(e.offsetX,e.offsetY);
}});
window.addEventListener('mousemove', e => {{
  if (!draggingDome) return;
  const r=dome.getBoundingClientRect();
  domeToLight(e.clientX-r.left, e.clientY-r.top);
}});
window.addEventListener('mouseup', () => {{ draggingDome=false; }});
dome.addEventListener('touchstart', e => {{
  e.preventDefault(); e.stopPropagation();
  const t=e.touches[0], r=dome.getBoundingClientRect();
  draggingDome=true; domeToLight(t.clientX-r.left, t.clientY-r.top);
}}, {{passive:false}});
dome.addEventListener('touchmove', e => {{
  e.preventDefault();
  const t=e.touches[0], r=dome.getBoundingClientRect();
  domeToLight(t.clientX-r.left, t.clientY-r.top);
}}, {{passive:false}});

function wireSlider(sliderId, valId, setter, fmt) {{
  const sl = document.getElementById(sliderId);
  const lbl = document.getElementById(valId);
  lbl.textContent = fmt(parseFloat(sl.value));
  sl.addEventListener('input', () => {{
    const v = parseFloat(sl.value);
    setter(v);
    lbl.textContent = fmt(v);
    render();
  }});
}}

wireSlider('s-gain', 'v-gain', v => gain = v, v => v.toFixed(2));
wireSlider('s-spec', 'v-spec', v => spec = v, v => v.toFixed(2));
wireSlider('s-gam',  'v-gam',  v => gam  = v, v => v.toFixed(2));
wireSlider('s-cmix', 'v-cmix', v => cmix = v, v => v.toFixed(2));
wireSlider('s-zoom', 'v-zoom', v => {{
  zoom = v; panX = 0; panY = 0; applyTransform();
}}, v => v.toFixed(1) + '×');

function exportPNG() {{
  render();
  const a = document.createElement('a');
  a.download = 'RTI_export_{_safe_id}.png';
  a.href = canvas.toDataURL('image/png');
  a.click();
}}

const initAz = 315 * Math.PI / 180;
const initEl = 45  * Math.PI / 180;
drawDome(
  0.5 * Math.cos(initAz) * Math.sin(initEl + Math.PI/2),
  0.5 * Math.sin(initAz) * Math.sin(initEl + Math.PI/2)
);
lx = Math.sin(initEl)*Math.cos(initAz);
ly = Math.sin(initEl)*Math.sin(initAz);
lz = Math.cos(initEl);
render();
</script>
</body>
</html>"""

    from helio.application.viewer_skin import skin
    html_doc = skin(html_doc, "#ui{width:264px!important}.row label{width:76px!important}"
                              ".row .val{width:40px!important}.mode-btn{padding:6px 4px!important}"
                              "#dome{border-radius:50%}")
    html_path = os.path.join(out_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_doc)

    readme_path = os.path.join(out_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(
            "Helio — PS Normal-Map RTI Viewer\n"
            + "=" * 50 + "\n\n"
            + f"Object: {obj_id}\nResolution: {dw}×{dh}px  {scale_note}\n\n"
            + "USAGE:\n"
            + "  Open index.html in any modern browser (Chrome/Firefox/Safari).\n"
            + "  No server required — all three files must be in the same folder.\n\n"
            + "CONTROLS:\n"
            + "  View mode buttons      — Relit / Greyscale / Specular / Normal Map / Albedo\n"
            + "  Drag the dome circle   — move the virtual light\n"
            + "  Gain slider            — overall brightness\n"
            + "  Specular slider        — add specular highlights (0=diffuse only)\n"
            + "  Gamma                  — tone curve adjustment\n"
            + "  Colour mix             — 0=greyscale shading, 1=full colour\n"
            + "  Export PNG button      — save the current view as PNG\n\n"
            + "FILES:\n"
            + "  index.html   — the viewer\n"
            + "  normals.png  — PS normal map (tangent space, RGB encoded)\n"
            + "  albedo.png   — colour albedo texture\n\n"
            + "WHY BETTER THAN PTM:\n"
            + "  PTM uses a 6-coefficient biquadratic approximation fitted per pixel.\n"
            + "  This viewer uses the full per-pixel normal from all-image PS solve,\n"
            + "  giving exact relighting without ringing or low-frequency bias.\n"
        )

    if progress_cb is not None:
        progress_cb("Normal-map RTI viewer written", 1.0)

    return html_path


def export_hsh_deepzoom_relight_html(
    out_dir: str,
    normals: np.ndarray,
    albedo_rgb: np.ndarray,
    hsh_coeffs: np.ndarray,
    h: int,
    w: int,
    obj_id: str = "object",
    max_dim: int = 2048,
    progress_cb: Callable[[str, float | None], None] | None = None,
) -> str:
    """Export a WebGL HSH relighting viewer with flat image files.

    Writes *albedo.jpg*, *normals.jpg*, HSH coefficient planes (``c0.png``,
    ``c1.png``, …), *meta.json*, and *index.html* into a subdirectory
    ``rti_relight/`` under *out_dir*.

    Returns the path to *index.html*.
    """
    import json as _json

    if progress_cb is not None:
        progress_cb("Exporting HSH relight viewer…", None)

    pkg = os.path.join(out_dir, "rti_relight")
    os.makedirs(pkg, exist_ok=True)

    sf = min(1.0, max_dim / max(h, w))
    th = int(h * sf)
    tw = int(w * sf)

    alb_s = cv2.resize(albedo_rgb, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.imwrite(
        os.path.join(pkg, "albedo.jpg"),
        cv2.cvtColor(alb_s, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    nm = (((np.clip(normals, -1, 1) + 1.0) / 2.0) * 255).astype(np.uint8)
    nm_s = cv2.resize(nm, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.imwrite(
        os.path.join(pkg, "normals.jpg"),
        cv2.cvtColor(nm_s, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    # The fit is made on 0–255 grey (compute_ptm_hsh_coefficients); the
    # shader draws 0–1, so coefficients left on that scale drew pure white.
    # Overhead, the basis sums to c0 + c3 + 2·c8: that tells the scale.
    raw_coeffs = np.asarray(hsh_coeffs, dtype=np.float32)
    if raw_coeffs.shape[0] >= 9:
        overhead = raw_coeffs[0] + raw_coeffs[3] + 2.0 * raw_coeffs[8]
        if float(np.percentile(overhead, 99)) > 2.0:
            raw_coeffs = raw_coeffs / 255.0
    n_coeff = raw_coeffs.shape[0]
    n_planes = int(np.ceil(n_coeff / 3))
    scale_bias: list[list[float]] = []

    for p in range(n_planes):
        channels = []
        for c in range(3):
            ci = p * 3 + c
            ch_flat = raw_coeffs[ci] if ci < n_coeff else np.zeros(h * w, np.float32)
            ch = ch_flat.reshape(h, w).astype(np.float32)
            lo = float(ch.min())
            rng = float(ch.max() - ch.min())
            rng = max(rng, 1e-9)
            ch_n = np.clip(((ch - lo) / rng) * 255, 0, 255).astype(np.uint8)
            ch_s = cv2.resize(ch_n, (tw, th), interpolation=cv2.INTER_AREA)
            channels.append(ch_s)
            scale_bias.append([lo, rng])
        plane = np.stack(channels, axis=2)
        cv2.imwrite(
            os.path.join(pkg, f"c{p}.png"),
            cv2.cvtColor(plane, cv2.COLOR_RGB2BGR),
        )

    meta = {
        "oid": obj_id,
        "w": tw,
        "h": th,
        "n": n_coeff,
        "planes": n_planes,
        "sb": scale_bias,
        "lights": [],
    }
    with open(os.path.join(pkg, "meta.json"), "w") as jf:
        _json.dump(meta, jf)

    sb_vals: list[str] = []
    for lo, rng in scale_bias:
        sb_vals.append(f"[{lo:.6f},{rng:.6f}]")
    sb_js = "[" + ",".join(sb_vals) + "]"

    _safe_id = html.escape(obj_id)
    html_doc = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Helio RTI Relight</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0f1a;font-family:system-ui,sans-serif;overflow:hidden;width:100vw;height:100vh}
#bar{position:fixed;top:0;left:0;right:0;height:44px;z-index:30;
  background:rgba(8,14,26,.96);border-bottom:1px solid #1e3050;
  display:flex;align-items:center;gap:8px;padding:0 12px}
#bar h1{font-size:13px;font-weight:700;color:#7dd4f8;white-space:nowrap}
.sep{width:1px;height:26px;background:#1e3050;margin:0 4px}
label{font-size:11px;color:#5a8aaa;display:flex;align-items:center;gap:4px}
label input[type=range]{width:64px;accent-color:#4ade80}
label span{min-width:30px;color:#7dd4f8;font-size:11px;text-align:right}
#canvas{position:fixed;top:44px;left:0;width:calc(100vw - 310px);height:calc(100vh - 44px);display:block}
#panel{position:fixed;top:44px;right:0;width:310px;bottom:0;
  background:rgba(8,14,26,.97);border-left:1px solid #1e3050;
  display:flex;flex-direction:column;align-items:center;padding:14px 10px;gap:10px;overflow-y:auto}
#panel h2{color:#7dd4f8;font-size:12px;font-weight:700;align-self:flex-start}
#dome{width:240px;height:240px;cursor:crosshair;user-select:none}
#light-info{color:#4ade80;font-size:12px;font-weight:700;text-align:center}
.mrow{display:grid;grid-template-columns:1fr 1fr;gap:5px;width:100%}
.mbtn{padding:5px 8px;background:#1a2e44;color:#90c8e8;border:1px solid #2a4a68;
  border-radius:4px;cursor:pointer;font-size:11px;font-weight:600;text-align:center}
.mbtn:hover,.mbtn.on{background:#1a5276;color:#fff;border-color:#3a7aaa}
.crow{display:flex;align-items:center;gap:6px;width:100%;font-size:11px;color:#5a8aaa}
.crow label{min-width:52px}
.crow input[type=range]{flex:1;accent-color:#3a7aaa}
.crow span{min-width:32px;color:#7dd4f8;text-align:right}
#shot{width:100%;padding:7px;background:#1a5276;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:12px;font-weight:700}
#shot:hover{background:#2176ae}
#status{font-size:10px;color:#3a5a7a;text-align:center;word-break:break-all}
</style>
</head>
<body>
<div id="bar">
  <h1>Helio RTI Relight &middot; """ + _safe_id + """</h1>
  <div class="sep"></div>
  <label>Brightness<input type="range" id="sBr" min="0.2" max="3" step="0.05" value="1"><span id="vBr">1.00</span></label>
  <label>Contrast<input type="range" id="sCo" min="0.5" max="3" step="0.05" value="1"><span id="vCo">1.00</span></label>
  <label>Gamma<input type="range" id="sGa" min="0.3" max="2.5" step="0.05" value="1"><span id="vGa">1.00</span></label>
</div>

<canvas id="canvas"></canvas>

<div id="panel">
  <h2>Light Direction</h2>
  <svg id="dome" viewBox="0 0 240 240">
    <circle cx="120" cy="120" r="116" fill="#1A1B1E" stroke="#3E4046" stroke-width="1.5"/>
    <circle cx="120" cy="120" r="77" fill="none" stroke="#3E4046" stroke-width="1" stroke-dasharray="3 4"/>
    <circle cx="120" cy="120" r="38" fill="none" stroke="#3E4046" stroke-width="1" stroke-dasharray="3 4"/>
    <line x1="120" y1="4" x2="120" y2="236" stroke="#3E4046" stroke-width="1" stroke-dasharray="3 4"/>
    <line x1="4" y1="120" x2="236" y2="120" stroke="#3E4046" stroke-width="1" stroke-dasharray="3 4"/>
    <text x="120" y="16" text-anchor="middle" fill="#B4B6BC" font-size="12" font-weight="bold">N</text>
    <text x="120" y="234" text-anchor="middle" fill="#B4B6BC" font-size="12" font-weight="bold">S</text>
    <text x="12" y="124" text-anchor="middle" fill="#B4B6BC" font-size="12" font-weight="bold">W</text>
    <text x="230" y="124" text-anchor="middle" fill="#B4B6BC" font-size="12" font-weight="bold">E</text>
    <line id="spoke" x1="120" y1="120" x2="62" y2="62" stroke="#E6A93E80" stroke-width="2"/>
    <circle id="dot" cx="62" cy="62" r="11" fill="#E6A93E" stroke="#222326" stroke-width="3" style="filter:drop-shadow(0 1px 3px rgba(0,0,0,.5))"/>
  </svg>
  <div id="light-info">Az: 315&deg; | Alt: 55&deg;</div>
  <div style="width:100%;height:1px;background:#36383D"></div>
  <h2>Display Mode</h2>
  <div class="mrow">
    <div class="mbtn on" id="mRTI"    onclick="setMode(0)">RTI Relight</div>
    <div class="mbtn"   id="mCol"    onclick="setMode(1)">Colour</div>
    <div class="mbtn"   id="mNrm"    onclick="setMode(2)">Normals</div>
    <div class="mbtn"   id="mSpec"   onclick="setMode(3)">Specular</div>
    <div class="mbtn"   id="mAO"     onclick="setMode(4)">Diffuse AO</div>
    <div class="mbtn"   id="mMix"    onclick="setMode(5)">Mix RTI+Col</div>
  </div>
  <div class="crow"><label>Mix RTI</label><input type="range" id="sMix" min="0" max="1" step="0.05" value="0.5"><span id="vMix">0.50</span></div>
  <div style="width:100%;height:1px;background:#36383D"></div>
  <button id="shot" onclick="doShot()">Save Screenshot</button>
  <div id="status">Loading...</div>
</div>

<script>
var SB=""" + sb_js + """;
var N=""" + str(n_coeff) + """, NP=""" + str(n_planes) + """;
var W=""" + str(tw) + """, H=""" + str(th) + """;
var mode=0, lx=0, ly=0, lz=1, az=315, alt=45, dFixed=false;
var gl, prog, textures={}, uniforms={};
var loaded=0, needed=2+NP;

function setLight(a,b){
  az=a; alt=b;
  var ar=a*Math.PI/180, br=b*Math.PI/180;
  lx=Math.sin(ar)*Math.cos(br);
  ly=-Math.cos(ar)*Math.cos(br);
  lz=Math.sin(br);
  var rn=Math.cos(br), px=120+rn*Math.sin(ar)*116, py=120-rn*Math.cos(ar)*116;
  document.getElementById('dot').setAttribute('cx',px.toFixed(1));
  document.getElementById('dot').setAttribute('cy',py.toFixed(1));
  document.getElementById('spoke').setAttribute('x2',px.toFixed(1));
  document.getElementById('spoke').setAttribute('y2',py.toFixed(1));
  document.getElementById('light-info').textContent='Az: '+a.toFixed(0)+'° | Alt: '+b.toFixed(0)+'°';
  render();
}
setLight(315,45);

var dome=document.getElementById('dome'), dDown=false;
function domeXY(e){
  var r=dome.getBoundingClientRect();
  var dx=(e.clientX-r.left-r.width/2)/(r.width/2);
  var dy=(e.clientY-r.top-r.height/2)/(r.height/2);
  var rr=Math.min(Math.sqrt(dx*dx+dy*dy),0.98);
  var a=((Math.atan2(dx,-dy)*180/Math.PI)+360)%360;
  var b=90-rr*85;
  return [a,b];
}
dome.addEventListener('mousedown',function(e){dDown=true; dFixed=false;});
dome.addEventListener('mousemove',function(e){if(dDown){var ab=domeXY(e);setLight(ab[0],ab[1]);}});
dome.addEventListener('mouseup',function(e){dDown=false;});
dome.addEventListener('click',function(e){var ab=domeXY(e);setLight(ab[0],ab[1]);dFixed=true;});

var cv=document.getElementById('canvas');
cv.addEventListener('mousemove',function(e){
  if(dFixed||mode!==0) return;
  var r=cv.getBoundingClientRect();
  var dx=(e.clientX-r.left)/r.width*2-1;
  var dy=(e.clientY-r.top)/r.height*2-1;
  var rr=Math.min(Math.sqrt(dx*dx+dy*dy),0.98);
  var a=((Math.atan2(dx,-dy)*180/Math.PI)+360)%360;
  var b=90-rr*85;
  setLight(a,b);
});
cv.addEventListener('dblclick',function(){dFixed=!dFixed;});

var modeIds=['mRTI','mCol','mNrm','mSpec','mAO','mMix'];
function setMode(m){
  mode=m;
  modeIds.forEach(function(id){document.getElementById(id).classList.remove('on');});
  document.getElementById(modeIds[m]).classList.add('on');
  render();
}

function syncLabel(id,vId,dp){
  var el=document.getElementById(id);
  document.getElementById(vId).textContent=parseFloat(el.value).toFixed(dp);
  el.addEventListener('input',function(){
    document.getElementById(vId).textContent=parseFloat(el.value).toFixed(dp);
    render();
  });
}
syncLabel('sBr','vBr',2); syncLabel('sCo','vCo',2);
syncLabel('sGa','vGa',2); syncLabel('sMix','vMix',2);

function doShot(){render();var a=document.createElement('a');a.href=cv.toDataURL('image/png');a.download='helio_rti.png';a.click();}

var VERT=`attribute vec2 aP;varying vec2 vU;void main(){vU=aP*.5+.5;vU.y=1.-vU.y;gl_Position=vec4(aP,0.,1.);}`;
var FRAG=`precision mediump float;
varying vec2 vU;
uniform sampler2D uA,uN,uP0,uP1,uP2;
uniform float uLx,uLy,uLz,uM,uBr,uCo,uGa,uMix;
uniform vec2 uS0,uS1,uS2,uS3,uS4,uS5,uS6,uS7,uS8;
void main(){
  vec4 a0=texture2D(uP0,vU);
  vec4 a1=texture2D(uP1,vU);
  vec4 a2=texture2D(uP2,vU);
  vec4 ac=texture2D(uA,vU);
  vec4 an=texture2D(uN,vU);
  float c0=a0.r*uS0.y+uS0.x;
  float c1=a0.g*uS1.y+uS1.x;
  float c2=a0.b*uS2.y+uS2.x;
  float c3=a1.r*uS3.y+uS3.x;
  float c4=a1.g*uS4.y+uS4.x;
  float c5=a1.b*uS5.y+uS5.x;
  float c6=a2.r*uS6.y+uS6.x;
  float c7=a2.g*uS7.y+uS7.x;
  float c8=a2.b*uS8.y+uS8.x;
  float lx=uLx,ly=uLy,lz=uLz;
  float I=c0+c1*lx+c2*ly+c3*lz+c4*lx*ly+c5*lx*lz+c6*ly*lz+c7*(lx*lx-ly*ly)+c8*(3.0*lz*lz-1.0);
  I=clamp(I,0.0,1.0);
  vec3 nm=normalize(an.rgb*2.0-1.0);
  float nd=max(dot(nm,vec3(lx,ly,lz)),0.0);
  vec3 col;
  if(uM<0.5){col=vec3(I);}
  else if(uM<1.5){col=ac.rgb;}
  else if(uM<2.5){col=an.rgb;}
  else if(uM<3.5){col=vec3(clamp(I+pow(nd,16.0)*.5,0.,1.));}
  else if(uM<4.5){col=vec3(I)*(0.3+0.7*nd);}
  else{col=mix(ac.rgb,vec3(I),uMix);}
  col=col*uBr;
  col=(col-0.5)*uCo+0.5;
  col=clamp(col,0.0,1.0);
  col=pow(col,vec3(1.0/max(uGa,0.01)));
  gl_FragColor=vec4(col,1.0);
}`;

function mkShader(type,src){
  var s=gl.createShader(type);
  gl.shaderSource(s,src); gl.compileShader(s);
  if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)){
    var e=gl.getShaderInfoLog(s);
    console.error(type===gl.VERTEX_SHADER?'VERT':'FRAG',e);
    document.getElementById('status').textContent='Shader err: '+e.substring(0,120);
  }
  return s;
}

function initGL(){
  gl=cv.getContext('webgl')||cv.getContext('experimental-webgl');
  if(!gl){document.getElementById('status').textContent='No WebGL';return false;}
  prog=gl.createProgram();
  gl.attachShader(prog,mkShader(gl.VERTEX_SHADER,VERT));
  gl.attachShader(prog,mkShader(gl.FRAGMENT_SHADER,FRAG));
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog,gl.LINK_STATUS)){
    document.getElementById('status').textContent='Link err: '+gl.getProgramInfoLog(prog);
    return false;
  }
  gl.useProgram(prog);
  var buf=gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER,buf);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
  var ap=gl.getAttribLocation(prog,'aP');
  gl.enableVertexAttribArray(ap);
  gl.vertexAttribPointer(ap,2,gl.FLOAT,false,0,0);
  ['uA','uN','uP0','uP1','uP2'].forEach(function(n,i){gl.uniform1i(gl.getUniformLocation(prog,n),i);});
  uniforms.Lx=gl.getUniformLocation(prog,'uLx'); uniforms.Ly=gl.getUniformLocation(prog,'uLy');
  uniforms.Lz=gl.getUniformLocation(prog,'uLz'); uniforms.M=gl.getUniformLocation(prog,'uM');
  uniforms.Br=gl.getUniformLocation(prog,'uBr'); uniforms.Co=gl.getUniformLocation(prog,'uCo');
  uniforms.Ga=gl.getUniformLocation(prog,'uGa'); uniforms.Mix=gl.getUniformLocation(prog,'uMix');
  for(var i=0;i<9;i++){
    var loc=gl.getUniformLocation(prog,'uS'+i);
    if(loc&&SB[i]) gl.uniform2f(loc,SB[i][0],SB[i][1]);
  }
  return true;
}

function loadImg(url,unit,cb){
  document.getElementById('status').textContent='Loading '+url+'...';
  var img=new Image();
  img.crossOrigin='anonymous';
  img.onload=function(){
    if(!gl) return cb();
    gl.activeTexture(gl.TEXTURE0+unit);
    var t=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,t);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,img);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    textures[unit]=t; cb();
  };
  img.onerror=function(){
    document.getElementById('status').textContent='ERR 404: '+url;
    console.error('Cannot load',url); cb();
  };
  img.src=url;
}

// The canvas's own box (a phone puts it above the controls, not beside
// them), and the picture fitted inside it at its own shape: stretched to the
// box, a coin came out oval.
function resize(){
  var w = Math.max(1, cv.clientWidth), h = Math.max(1, cv.clientHeight);
  if(cv.width !== w || cv.height !== h){
    cv.width = w; cv.height = h;
  }
  var s = Math.min(w / W, h / H), vw = Math.round(W * s), vh = Math.round(H * s);
  if(gl) gl.viewport(Math.round((w - vw) / 2), Math.round((h - vh) / 2), vw, vh);
}

function render(){
  if(!gl||!prog) return;
  resize();
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.uniform1f(uniforms.Lx,lx); gl.uniform1f(uniforms.Ly,ly); gl.uniform1f(uniforms.Lz,lz);
  gl.uniform1f(uniforms.M,mode);
  gl.uniform1f(uniforms.Br,parseFloat(document.getElementById('sBr').value));
  gl.uniform1f(uniforms.Co,parseFloat(document.getElementById('sCo').value));
  gl.uniform1f(uniforms.Ga,parseFloat(document.getElementById('sGa').value));
  gl.uniform1f(uniforms.Mix,parseFloat(document.getElementById('sMix').value));
  gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
}

function onLoad(){
  loaded++;
  document.getElementById('status').textContent='Loaded '+loaded+' / '+needed+'...';
  if(loaded>=needed){
    document.getElementById('status').textContent='Ready — move dome or mouse over image';
    render();
  }
}

var imgQueue=[];
imgQueue.push(['albedo.jpg',0]);
imgQueue.push(['normals.jpg',1]);
for(var p=0;p<NP;p++) imgQueue.push(['c'+p+'.png',2+p]);

var rawImgs={}, rawLoaded=0;
imgQueue.forEach(function(item){
  var url=item[0], unit=item[1];
  var img=new Image();
  img.crossOrigin='anonymous';
  img.onload=function(){
    rawImgs[unit]=img; rawLoaded++;
    document.getElementById('status').textContent='Loading... '+rawLoaded+'/'+needed;
    if(rawLoaded>=needed){
      var ok=initGL();
      if(!ok){
        document.getElementById('status').textContent='WebGL unavailable — showing colour preview';
        var fc=document.createElement('canvas');
        fc.style.cssText='position:fixed;top:44px;left:0;right:310px;bottom:0;object-fit:contain';
        fc.width=W; fc.height=H;
        var ctx=fc.getContext('2d'); ctx.drawImage(rawImgs[0],0,0,W,H);
        cv.replaceWith(fc);
        return;
      }
      imgQueue.forEach(function(it){
        var uid=it[1];
        if(!rawImgs[uid]) return;
        gl.activeTexture(gl.TEXTURE0+uid);
        var t=gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D,t);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,rawImgs[uid]);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
        textures[uid]=t;
      });
      resize();
      document.getElementById('status').textContent='Ready — move dome or mouse over image';
      render();
      window.addEventListener('resize',function(){resize();render();});
    }
  };
  img.onerror=function(){
    console.error('Cannot load image: '+url);
    document.getElementById('status').textContent='ERROR: Cannot load '+url+'. Check server is running.';
    rawLoaded++;
    if(rawLoaded>=needed && Object.keys(rawImgs).length>0){
      var ok=initGL(); if(ok){
        imgQueue.forEach(function(it){ var uid=it[1]; if(!rawImgs[uid]) return;
          gl.activeTexture(gl.TEXTURE0+uid); var t=gl.createTexture();
          gl.bindTexture(gl.TEXTURE_2D,t);
          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,rawImgs[uid]);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
        }); resize(); render();
      }
    }
  };
  img.src=url;
});
</script>
</body>
</html>"""

    from helio.application.phone_web import phone_ready
    html_doc = phone_ready(
        html_doc, panel="#panel", stage="#canvas",
        hooks="window.HelioPhone={getLight:function(){return [az,alt];},"
              "setLight:function(a,e){dFixed=true;setLight(a,e);},"
              "reset:function(){dFixed=false;setLight(315,45);}};\n",
        hooks_before="</script>",
        css="#canvas{width:100vw!important;height:calc(100vh - 44px - var(--hp-sheet))!important}"
            "#bar{overflow-x:auto;-webkit-overflow-scrolling:touch}#bar h1{font-size:12px}"
            "#bar label{flex:0 0 auto}#dome{width:190px;height:190px}",
        tip="Drag the picture to move the light \u00b7 double-tap to reset")
    html_path = os.path.join(pkg, "index.html")
    with open(html_path, "w", encoding="utf-8") as hf:
        hf.write(html_doc)

    if progress_cb is not None:
        progress_cb("HSH relight viewer written", 1.0)

    return html_path


def export_engine_sweep_html(
    out_path: str,
    frames: "list[tuple[str, np.ndarray]]",
    obj_id: str = "Object",
    max_side: int = 1600,
    progress_cb: "Callable[[str, float | None], None] | None" = None,
) -> str:
    """One HTML file that slides through every render engine.

    The A/B Comparison Viewer wipes between two engines. Choosing *which* two
    is the hard part, and it is a choice made by flipping through the whole set
    — which until now could only be done inside HELIO. This carries the set, so
    the person you send it to can make that choice too, and can name the engine
    they mean because every frame is labelled.

    Frames are capped at *max_side* and stored as JPEG. Thirty-odd engines at
    full resolution would be a file nobody can email, which would defeat the
    only reason it is one file.

    Returns *out_path*.
    """
    import base64 as _b64

    if not frames:
        raise ValueError("no frames to write")

    # Engines disagree wildly about output range: some return 0..1, some 0..255,
    # some a signed field normalised to neither, and a few put their speculars
    # hard against white. Left alone, the sweep flickers between a blown frame
    # and a muddy one and the comparison it exists for becomes impossible.
    #
    # Balance is applied to *luminance*, with one gain and offset shared by all
    # three channels, so colour ratios survive — a per-channel stretch would
    # quietly shift hue and make two engines look like different materials.
    # The top of the range maps to 247, not 255: leaving headroom is what stops
    # a specular highlight becoming a flat white hole with its shape lost.
    #
    # A frame already well exposed is left untouched. The point is to stop the
    # outliers spoiling the sweep, not to restyle engines that were already
    # right.
    _TARGET_LO, _TARGET_HI = 4.0, 247.0

    def _balance(u8):
        f = u8.astype(np.float32)
        lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        lo = float(np.percentile(lum, 0.5))
        hi = float(np.percentile(lum, 99.5))
        if hi - lo < 1e-6:
            return u8, False
        blown = float((lum >= 254.0).mean())
        crushed = float((lum <= 1.0).mean())
        if (lo >= 2.0 and hi <= 250.0 and blown < 0.002 and crushed < 0.02
                and (hi - lo) > 90.0):
            return u8, False                      # already fine
        g = (_TARGET_HI - _TARGET_LO) / (hi - lo)
        return np.clip((f - lo) * g + _TARGET_LO, 0, 255).astype(np.uint8), True

    def _enc(arr):
        a = np.asarray(arr)
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        a = a[..., :3]
        if a.dtype != _np.uint8:
            a = a.astype(np.float32)
            if float(np.nanmax(a)) <= 1.0 + 1e-6:
                a = a * 255.0
            a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
        h, w = a.shape[:2]
        sc = min(1.0, max_side / float(max(h, w)))
        if sc < 1.0:
            a = cv2.resize(a, (max(1, int(w * sc)), max(1, int(h * sc))),
                           interpolation=cv2.INTER_AREA)
        a, _adj = _balance(a)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(a, cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), 86])
        if not ok:
            raise RuntimeError("frame could not be encoded")
        return ("data:image/jpeg;base64," + _b64.b64encode(buf.tobytes()).decode("ascii"),
                _adj)

    items, _n_adjusted = [], 0
    for i, (label, arr) in enumerate(frames):
        if progress_cb is not None:
            progress_cb(f"Encoding {label}…", 0.5 + 0.4 * (i / max(1, len(frames))))
        _src, _adj = _enc(arr)
        _n_adjusted += 1 if _adj else 0
        items.append((str(label), _src))

    # Said on the page, not just in the log. These frames are for choosing an
    # engine by eye; someone must not read a density off a stretched one
    # without being told it was stretched.
    _bal_note = (f" · {_n_adjusted} of {len(items)} frames exposure-balanced for viewing"
                 if _n_adjusted else "")
    _names = ",".join(_json.dumps(n) for n, _ in items)
    _srcs = "".join(
        f'<img class="fr" data-i="{i}" src="{s}" alt="{html.escape(n)}">'
        for i, (n, s) in enumerate(items))
    _title = html.escape(str(obj_id))

    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Helio — Engine Sweep — {_title}</title>
<style>
 html,body{{margin:0;height:100%;background:#0b0d12;color:#cfd8e3;
   font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;}}
 #stage{{position:fixed;inset:0 0 96px 0;display:flex;align-items:center;justify-content:center;overflow:hidden;}}
 .fr{{position:absolute;max-width:100%;max-height:100%;object-fit:contain;display:none;
   image-rendering:auto;user-select:none;-webkit-user-drag:none;}}
 .fr.on{{display:block;}}
 #bar{{position:fixed;left:0;right:0;bottom:0;height:96px;background:#12151d;
   border-top:1px solid #232a36;padding:10px 16px;box-sizing:border-box;}}
 #row1{{display:flex;align-items:center;gap:12px;}}
 #name{{font-size:15px;font-weight:600;color:#eaf1f8;flex:1;
   white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}}
 #count{{font-size:12px;color:#7b8798;font-variant-numeric:tabular-nums;}}
 button{{background:#1e2634;color:#cfd8e3;border:1px solid #2c3746;border-radius:6px;
   padding:6px 12px;font-size:13px;cursor:pointer;}}
 button:hover{{background:#283242;}}
 #slider{{width:100%;margin-top:12px;accent-color:#4b8fc7;cursor:pointer;}}
 #hint{{position:fixed;top:10px;left:0;right:0;text-align:center;font-size:11px;
   color:#5d6a7c;pointer-events:none;}}
</style></head><body>
<div id="hint">← → step · space plays · {_title}{_bal_note}</div>
<div id="stage">{_srcs}</div>
<div id="bar">
  <div id="row1">
    <button id="prev" title="Previous engine">◀</button>
    <button id="play" title="Play/pause">▶ Play</button>
    <button id="next" title="Next engine">▶</button>
    <div id="name"></div>
    <div id="count"></div>
  </div>
  <input id="slider" type="range" min="0" max="{len(items) - 1}" value="0" step="1">
</div>
<script>
var NAMES=[{_names}];
var frames=[].slice.call(document.querySelectorAll('.fr'));
var slider=document.getElementById('slider');
var nameEl=document.getElementById('name'), countEl=document.getElementById('count');
var i=0, timer=null;
function show(n){{
  n=Math.max(0,Math.min(NAMES.length-1,n));
  // Toggle a class rather than rebuild src: every frame is already decoded in
  // the document, so stepping is instant and does not flash white.
  if(frames[i]) frames[i].classList.remove('on');
  i=n; if(frames[i]) frames[i].classList.add('on');
  nameEl.textContent=NAMES[i];
  countEl.textContent=(i+1)+' / '+NAMES.length;
  if(slider.value!=i) slider.value=i;
}}
slider.addEventListener('input',function(){{show(parseInt(slider.value,10));}});
document.getElementById('prev').onclick=function(){{stop();show(i-1);}};
document.getElementById('next').onclick=function(){{stop();show(i+1);}};
function stop(){{if(timer){{clearInterval(timer);timer=null;
  document.getElementById('play').textContent='▶ Play';}}}}
document.getElementById('play').onclick=function(){{
  if(timer){{stop();return;}}
  this.textContent='❙❙ Pause';
  timer=setInterval(function(){{show((i+1)%NAMES.length);}},700);
}};
document.addEventListener('keydown',function(e){{
  if(e.key==='ArrowRight'){{stop();show(i+1);e.preventDefault();}}
  else if(e.key==='ArrowLeft'){{stop();show(i-1);e.preventDefault();}}
  else if(e.key===' '){{document.getElementById('play').click();e.preventDefault();}}
}});
// A phone: swipe the picture left or right to step.
var sx=null,stage=document.getElementById('stage');
stage.style.touchAction='pan-y';
stage.addEventListener('touchstart',function(e){{sx=e.touches[0].clientX;}},{{passive:true}});
stage.addEventListener('touchend',function(e){{
  if(sx===null)return; var dx=e.changedTouches[0].clientX-sx; sx=null;
  if(Math.abs(dx)>40){{stop();show(i+(dx<0?1:-1));}}
}},{{passive:true}});
show(0);
</script></body></html>"""

    from helio.application.phone_web import phone_ready
    doc = phone_ready(doc, css="#hint{font-size:10px!important;padding:0 10px}"
                               ".fr{width:100%;height:100%}",
                      tip="Swipe the picture to step through the views")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    if progress_cb is not None:
        progress_cb("Engine sweep viewer written", 1.0)
    return out_path
