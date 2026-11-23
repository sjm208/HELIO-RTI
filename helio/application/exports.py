"""Exports, GUI-free: what to write, with which settings (EXP-001, first slice).

The classic export methods were each a dialog, some settings gathered from
Tk, and a call into helio.io.  The gathering and the call live here now;
both apps use them, so a file comes out the same from either.  The classic
methods keep their dialogs and messages; tests/test_exports.py checks them, and
tests/test_qt_publish.py checks the new app's.

Moved so far: the export-settings builder, the GLB web viewer, STL for
printing, the .lp light file, the self-contained RTI viewer and the PTM
coefficient maps.  The rest stay in the classic app until moved the same way.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

# ── settings ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MeshOptions:
    """The export panel's mesh choices, with the classic app's defaults."""
    watertight: bool = True
    vertex_colors: bool = True
    sketchfab_pbr: bool = False
    sketchfab_jpeg: bool = False
    normal_flip_y: bool = True
    bake_normals: bool = False
    bake_depth_pct: float = 5.0        # percent here; a fraction in the settings dict
    use_hd: bool = False
    hd_warp_pct: float = 18.0          # percent here; a fraction in the settings dict


def trim_box(w: int, h: int, edge_trim_pct: float, crop_box=None):
    """The region an export covers: the drawn crop, else the edge-trimmed frame.

    The outermost rows of a photometric solve are the least trustworthy, which
    is why the trim exists; the 3D viewer, the STL and the web viewer all
    cover this same region (tests/test_export_region.py)."""
    if crop_box:
        return tuple(int(c) for c in crop_box)
    tp = edge_trim_pct / 100.0
    return (max(0, int(w * tp)), max(0, int(h * tp)),
            w - max(0, int(w * tp)), h - max(0, int(h * tp)))


# What the mesh exports write, per point of the mesh and per pixel of the
# picture (their texture maps), measured on the 8-frame test capture at
# steps 1 and 2 (2026-10-18).  The formats not listed have a resolution of
# their own: the export density does not change them.
MESH_BYTES = {
    "glb": (103.0, 4.7),          # GLB web viewer
    "glb_mesh": (114.0, 0.3),     # GLB model
    "obj": (460.0, 0.0),          # OBJ: text, the largest
    "ply": (108.0, 0.1),
    "sketchfab": (155.0, 8.8),
}


def mesh_estimate(w: int, h: int, step: int, edge_trim_pct: float, crop_box=None,
                  keys=None) -> dict:
    """The mesh an export would write: {'size': (w, h) covered, 'points',
    'triangles', 'bytes': {format: bytes}} for the formats in *keys* that
    follow the export density (all of MESH_BYTES by default).  As the
    classic's export estimate, with the bytes measured rather than assumed."""
    x1, y1, x2, y2 = trim_box(w, h, edge_trim_pct, crop_box)
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    st = max(1, int(step))
    gw, gh = (cw - 1) // st + 1, (ch - 1) // st + 1
    pts, tris = gw * gh, 2 * (gw - 1) * (gh - 1)
    keys = list(MESH_BYTES) if keys is None else [k for k in keys if k in MESH_BYTES]
    return {"size": (cw, ch), "points": pts, "triangles": tris,
            "bytes": {k: MESH_BYTES[k][0] * pts + MESH_BYTES[k][1] * cw * ch for k in keys}}


def build_export_settings(shared: dict, opts: MeshOptions, w: int, h: int, *, crop_box=None,
                          obj_id: str | None = None, fmt: str | None = None) -> dict:
    """The settings dict the mesh and web-viewer exporters read.

    *shared* is the 3D/export state (helio.config.SHARED_STATE_DEFAULTS keys,
    as plain values).  Moved from RTIViewerPro._build_export_settings."""
    settings = dict(shared)
    # Web viewers: step 1 by default, so full resolution; the export_step
    # slider can raise it for a smaller file.
    settings['step'] = max(1, settings.get('export_step', 1))
    # MESH_MAX_RES capped these exports far below native resolution; the step
    # alone controls density, as for the STL and OBJ exporters.
    settings['max_res'] = 20000
    settings['smooth'] = settings.get('mesh_smooth', 0.0)
    settings['micro'] = settings.get('micro_relief', 0.0)
    settings['hf_cutoff'] = settings.get('hf_normals_cutoff', 0.35)
    settings['use_hybrid'] = bool(settings.get('hybrid_enabled', 0))
    settings['detail_mix'] = settings.get('hybrid_mix', 0.12)
    settings['watertight'] = opts.watertight
    settings['vertex_colors'] = opts.vertex_colors
    settings['sketchfab_pbr'] = opts.sketchfab_pbr
    settings['sketchfab_jpeg'] = bool(opts.sketchfab_jpeg)
    settings['normal_flip_y'] = opts.normal_flip_y
    settings['bake_normals'] = bool(opts.bake_normals)
    settings['bake_depth_pct'] = float(opts.bake_depth_pct) / 100.0      # percent → fraction
    settings['use_hd'] = opts.use_hd
    settings['hd_warp_pct'] = opts.hd_warp_pct / 100.0
    if crop_box:
        settings['crop'] = tuple(crop_box)
    else:
        # No crop drawn: the edge-trimmed region, as the 3D viewer and the STL.
        try:
            tp = shared['edge_trim'] / 100.0           # this order, as trim_box: int(w * tp)
            tw = max(0, int(w * tp))
            th = max(0, int(h * tp))
            if tw or th:
                settings['crop'] = (tw, th, w - tw, h - th)
        except Exception:
            pass
    if obj_id is not None:             # the classic app writes it even when empty
        settings['obj_id'] = obj_id
    if fmt:
        settings['format'] = fmt
    return settings


# Publish's model choices that the 3D viewer does not share: plain values
# beside SHARED_STATE_DEFAULTS in the same dict, never sent to the viewer.
EXPORT_DEFAULTS = {
    "tex_size": "4096",          # helio.application.export_3d.TEXTURE_SIZES
    "tex_jpeg": 0,               # the colour textures as JPEG (normal maps stay PNG)
    "mesh_watertight": 1,        # a closed solid with a base
    "mesh_vertex_colours": 1,    # colour on every point (PLY, OBJ with its maps)
    "export_colour": "photo",    # the colour every export carries: "photo" as solved, "matte" as Matte Relight
}

# "Made for": one choice that sets the density and the model options together.
MESH_TARGETS = {
    "general": ("Blender, MeshLab, CloudCompare",       # Helio's starting values
                dict(export_step=4, tex_size="4096", tex_jpeg=0, mesh_watertight=1, mesh_vertex_colours=1)),
    "study": ("Close study (dense, 8K)",
              dict(export_step=2, tex_size="8192", tex_jpeg=0, mesh_watertight=1, mesh_vertex_colours=1)),
    "web": ("Web and Sketchfab (small)",
            dict(export_step=4, tex_size="4096", tex_jpeg=1, mesh_watertight=0, mesh_vertex_colours=1)),
    "phone": ("Phone and AR (smallest)",
              dict(export_step=8, tex_size="2048", tex_jpeg=1, mesh_watertight=0, mesh_vertex_colours=1)),
    "archive": ("Archive (everything, lossless)",
                dict(export_step=1, tex_size="full", tex_jpeg=0, mesh_watertight=1, mesh_vertex_colours=1)),
    "print": ("3D printing (closed, no colour)",
              dict(export_step=2, tex_size="mesh", tex_jpeg=0, mesh_watertight=1, mesh_vertex_colours=0)),
}


def mesh_target(shared: dict) -> str:
    """The MESH_TARGETS key the settings in *shared* match, or "" (custom)."""
    for key, (_name, vals) in MESH_TARGETS.items():
        if all(str(shared.get(k)) == str(v) for k, v in vals.items()):
            return key
    return ""


def texture_bytes(w: int, h: int, edge_trim_pct: float, tex_size: str, jpeg: bool) -> float:
    """About what the colour and normal textures add to a model at *tex_size*
    (measured: PNG colour about 1.6 bytes a texel, its normal map about 1.4;
    a JPEG colour about 0.4).  0 at the mesh's own density, counted in MESH_BYTES."""
    if tex_size in (None, "", "mesh"):
        return 0.0
    x1, y1, x2, y2 = trim_box(w, h, edge_trim_pct)
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    if tex_size != "full":
        k = min(1.0, int(tex_size) / float(max(cw, ch)))
        cw, ch = cw * k, ch * k
    return cw * ch * ((0.4 if jpeg else 1.6) + 1.4)


def shared_defaults() -> dict:
    """The 3D/export state as a fresh session has it (plain values)."""
    from helio.config import SHARED_STATE_DEFAULTS
    out = {}
    for k, (tc, v) in SHARED_STATE_DEFAULTS.items():
        # The classic app reads these through multiprocessing.Value; a 'f'
        # value comes back as the float32 it was stored as.
        out[k] = float(np.float32(v)) if tc == 'f' else v
    out.update(EXPORT_DEFAULTS)
    return out


# ── the exports ───────────────────────────────────────────────────────

class NothingToExport(Exception):
    """The export has no data to write (with the reason to show)."""


def web_viewer(depth_vis, depth_normalised, normals, albedo, settings: dict, out_dir: str,
               progress: Callable | None = None) -> dict:
    """GLB model + Three.js viewer (index.html) in *out_dir*.

    A mesh over four million vertices is decimated by raising the step, so
    the browser can still open it."""
    from helio.io.exporters import export_3d_web_viewer
    if depth_vis is None:
        raise NothingToExport("No scan data.")
    settings = dict(settings)
    if albedo is None:
        albedo = np.full((depth_vis.shape[0], depth_vis.shape[1], 3), 128, dtype=np.uint8)
    cx1, cy1, cx2, cy2 = settings.get("crop", (0, 0, albedo.shape[1], albedo.shape[0]))
    MAX_EXPORT_VERTS = 4_000_000
    est = ((cx2 - cx1) // settings['step']) * ((cy2 - cy1) // settings['step'])
    if est > MAX_EXPORT_VERTS:
        ratio = math.sqrt(est / MAX_EXPORT_VERTS)
        settings['step'] = max(settings['step'], int(math.ceil(settings['step'] * ratio)))
    return export_3d_web_viewer(depth_vis, depth_normalised, normals, albedo, settings, out_dir,
                                progress_cb=progress)


@dataclass(frozen=True)
class StlOptions:
    """The STL panel's choices, with the classic app's defaults."""
    relief_mm: float = 3.0
    base_mm: float = 3.0
    max_res: int = 200
    use_hd: bool = False
    mirror_x: bool = False
    bed_mm: float = 256.0
    z_ex: float = 1.0
    true_scale: bool = False
    hd_warp_pct: float = 18.0


def stl_print(depth_vis, depth_normalised, normals, w: int, h: int, shared: dict,
              opts: StlOptions, path: str, *, crop_box=None, mm_per_px: float = 0.0,
              normal_invert: bool = False, progress: Callable | None = None) -> str:
    """A watertight, print-ready STL of the region at *path*."""
    from helio.io.exporters import export_stl_3dprint
    if depth_vis is None:
        raise NothingToExport("No scan data. Run processing first.")
    max_res = opts.max_res
    # HD keeps carved lines the standard path smooths away; pointless if the
    # grid is then downsampled, so it is floored at 1200 px.
    if opts.use_hd and max_res < 1200:
        max_res = 1200
    fz = shared['fz']
    if normal_invert:          # the live Invert Normals override flips the mesh too
        fz = -fz
    x1, y1, x2, y2 = trim_box(w, h, shared['edge_trim'], crop_box)
    settings = {
        'relief_mm': opts.relief_mm, 'base_mm': opts.base_mm, 'max_res': max_res,
        'mirror_x': opts.mirror_x, 'bed_mm': max(50.0, opts.bed_mm), 'z_ex': max(1.0, opts.z_ex),
        # mm per source pixel when calibrated and true scale was asked for;
        # 0 scales the model to the print bed.
        'mm_per_px': float(mm_per_px or 0) if opts.true_scale else 0.0,
        'fz': fz, 'shape_blend': max(0.0, min(1.0, shared['shape_blend'])),
        'smooth': shared['mesh_smooth'],
        'step': 1, 'hp_sigma': 0, 'warp_rem': 0, 'micro': 0,
        'use_hd': opts.use_hd, 'hd_warp_pct': opts.hd_warp_pct / 100.0,
    }
    # Crop all three consistently, or a crop mixes full and cropped arrays.
    dn = depth_normalised[y1:y2, x1:x2] if depth_normalised is not None else None
    nf = normals[y1:y2, x1:x2] if normals is not None else None
    export_stl_3dprint(depth_vis[y1:y2, x1:x2], dn, nf, settings, path, progress_cb=progress)
    return path


def light_file(lights, source: str, path: str, image_paths=None) -> str:
    """A CHI .lp light file, each light named for its image.

    Lights only *estimated* by an Uncalibrated PCA solve are refused: they
    are known only up to a rotation, and a .lp would present them as measured."""
    from helio.io.sidecars import write_lp
    if lights is None or not len(lights):
        raise NothingToExport("No light vectors yet. Calibrate the lights, or import a .lp.")
    if source == "estimated":
        raise NothingToExport(
            "The only light directions here were recovered by the Uncalibrated PCA solve, "
            "which resolves them only up to a rotation. Writing them as a .lp would present "
            "an estimate as a measurement. Calibrate with a chrome sphere, place the lights "
            "on the dome, or import a .lp.")
    vecs = [tuple(float(c) for c in v) for v in lights]
    write_lp(path, vecs, image_paths)
    return path


def selfcontained_rti(ptm_coeffs, albedo, normals, w: int, h: int, shared: dict, path: str, *,
                      obj_id: str = "Object", crop_box=None) -> str:
    """One HTML file: move the light in any browser (long edge capped at 1024 px)."""
    from helio.io.exporters import export_ptm_selfcontained_viewer
    if ptm_coeffs is None or albedo is None:
        raise NothingToExport("Process scans first.")
    export_ptm_selfcontained_viewer(path, ptm_coeffs, albedo, normals, w, h,
                                    obj_id=obj_id or "Object",
                                    crop_box=trim_box(w, h, shared['edge_trim'], crop_box),
                                    max_edge=1024)
    return path


def ptm_coefficients(ptm_coeffs, w: int, h: int, out_dir: str, *, f32: bool = True,
                     u16: bool = True, vis: bool = False) -> list[str]:
    """The six PTM coefficient maps: float32, 16-bit range-mapped, and/or colour."""
    if ptm_coeffs is None:
        raise NothingToExport("No PTM coefficients — process images first.")
    out = []
    for i in range(6):
        c = ptm_coeffs[i].reshape(h, w).astype(np.float32)
        if f32:
            p = os.path.join(out_dir, f"ptm_coeff_{i}_f32.tif")
            cv2.imwrite(p, c, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            out.append(p)
        if u16:
            mn, mx = float(c.min()), float(c.max())
            p = os.path.join(out_dir, f"ptm_coeff_{i}_u16.tif")
            cv2.imwrite(p, ((c - mn) / max(mx - mn, 1e-9) * 65535).astype(np.uint16),
                        [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            out.append(p)
        if vis:
            u8 = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            p = os.path.join(out_dir, f"ptm_coeff_{i}_vis.tif")
            cv2.imwrite(p, cv2.applyColorMap(u8, cv2.COLORMAP_TURBO),
                        [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            out.append(p)
    return out


def anaglyph_image(depth_vis, albedo, path: str, *, crop_box=None, w: int = 0, h: int = 0) -> str:
    """Red-cyan stereo of the region (the drawn crop, else the whole frame).

    The separation scales with the image, so the effect is the same on a 6 MP
    and a 100 MP capture."""
    from helio.io.figures import anaglyph
    if depth_vis is None:
        raise NothingToExport("Process a capture first.")
    x1, y1, x2, y2 = crop_box if crop_box else (0, 0, w or depth_vis.shape[1], h or depth_vis.shape[0])
    d = depth_vis[y1:y2, x1:x2]
    c = albedo[y1:y2, x1:x2] if albedo is not None else None
    if c is None:
        c = np.dstack([np.clip(d * 255, 0, 255).astype(np.uint8)] * 3)
    sep = max(6, int(round((x2 - x1) * 0.012)))
    img = anaglyph(d, c, separation_px=sep)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


SWEEP_FRAMES, SWEEP_MS, SWEEP_RADIUS = 60, 50, 0.8      # 20 fps, light at 0.8 of the dome


def sweep_lights(n_frames: int = SWEEP_FRAMES, r: float = SWEEP_RADIUS):
    """The light positions of a sweep: once round, in screen frame (lx, ly, lz)."""
    out = []
    for theta in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        vx = float(np.cos(theta) * r)
        vy = float(np.sin(theta) * r)
        out.append((vx, vy, float(np.sqrt(max(0.001, 1.0 - vx ** 2 - vy ** 2)))))
    return out


def light_sweep_gif(draw: Callable, path: str, *, n_frames: int = SWEEP_FRAMES,
                    duration_ms: int = SWEEP_MS, progress: Callable | None = None) -> str:
    """An animated GIF of the light going once round the object.

    *draw(lx, ly, lz)* returns one frame as uint8 RGB; each app draws with its
    own renderer and view settings."""
    from PIL import Image
    frames = []
    lights = sweep_lights(n_frames)
    for i, (vx, vy, vz) in enumerate(lights):
        frames.append(Image.fromarray(draw(vx, vy, vz)).convert("P", dither=Image.FLOYDSTEINBERG))
        if progress:
            progress(f"Frame {i + 1} of {n_frames}", (i + 1) / n_frames)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms,
                   loop=0, optimize=False)
    return path


def point_cloud_e57(depth_vis, depth_normalised, albedo, w: int, h: int, shared: dict,
                    path: str, *, crop_box=None) -> str:
    """The surface as a coloured point cloud: E57, or a binary PLY beside it when
    pye57 is not installed.  Returns the file written."""
    if depth_vis is None:
        raise NothingToExport("Process images first.")
    step = max(1, shared['export_step'])
    sb = max(0., min(1., shared['shape_blend']))
    fx, fy, fz, hs = shared['fx'], shared['fy'], shared['fz'], shared['height']
    x1, y1, x2, y2 = crop_box if crop_box else (0, 0, w, h)
    if depth_normalised is not None and sb < 0.999:
        raw = (depth_normalised[y1:y2, x1:x2] * (1 - sb) +
               depth_vis[y1:y2, x1:x2] * sb).astype(np.float32)
    else:
        raw = depth_vis[y1:y2, x1:x2].astype(np.float32)
    hr, wr = raw.shape
    ys, xs = np.mgrid[0:hr:step, 0:wr:step]
    X = ((xs - wr / 2) * fx).flatten().astype(np.float64)
    Y = ((ys - hr / 2) * fy).flatten().astype(np.float64)
    Z = (raw[0:hr:step, 0:wr:step] * fz * hs - hs / 2).flatten().astype(np.float64)
    alb = albedo[y1:y2, x1:x2][0:hr:step, 0:wr:step]
    R = alb[:, :, 0].flatten().astype(np.uint8)
    G = alb[:, :, 1].flatten().astype(np.uint8)
    B = alb[:, :, 2].flatten().astype(np.uint8)
    try:
        import pye57
    except ImportError:
        import struct
        ply = path.replace(".e57", "_fallback.ply")
        hdr = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(X)}\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        with open(ply, "wb") as fp:
            fp.write(hdr.encode())
            for row in np.column_stack((X.astype(np.float32), Y.astype(np.float32),
                                        Z.astype(np.float32), R, G, B)):
                fp.write(struct.pack("<fffBBB", *row))
        return ply
    e57f = pye57.E57(path, mode="w")
    try:
        # pye57 >= 0.4 needs point_count; int32 colours suit every version.
        e57f.write_scan_raw({"cartesianX": X, "cartesianY": Y, "cartesianZ": Z,
                             "colorRed": R.astype(np.int32), "colorGreen": G.astype(np.int32),
                             "colorBlue": B.astype(np.int32)}, point_count=len(X))
    except TypeError:
        e57f.write_scan_raw({"cartesianX": X, "cartesianY": Y, "cartesianZ": Z,
                             "colorRed": R, "colorGreen": G, "colorBlue": B})
    e57f.close()
    return path


def lido_record(path: str, obj_id: str, normal_engine: str, depth_engine: str,
                n_images: int, when: str | None = None) -> str:
    """A LIDO XML record for museum collection systems.

    The object ID is escaped (an & or < once made the file invalid XML) and the
    image count is the capture's (it was always written as 0): KD-015."""
    import html
    import time
    oid = html.escape(obj_id or "HELIO_OBJECT")
    now_str = when or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<lido:lido xmlns:lido="http://www.lido-schema.org">
  <lido:lidoRecID lido:type="local">{oid}</lido:lidoRecID>
  <lido:descriptiveMetadata xml:lang="en">
    <lido:objectClassificationWrap>
      <lido:classificationWrap>
        <lido:classification>
          <lido:term>Photometric Stereo 3D Scan</lido:term>
        </lido:classification>
      </lido:classificationWrap>
    </lido:objectClassificationWrap>
    <lido:objectIdentificationWrap>
      <lido:titleWrap><lido:titleSet><lido:appellationValue>{oid}</lido:appellationValue></lido:titleSet></lido:titleWrap>
    </lido:objectIdentificationWrap>
  </lido:descriptiveMetadata>
  <lido:administrativeMetadata>
    <lido:recordWrap>
      <lido:recordID lido:type="local">{oid}</lido:recordID>
      <lido:recordType><lido:term>item</lido:term></lido:recordType>
      <lido:recordSource><lido:legalBodyName><lido:appellationValue>Helio</lido:appellationValue></lido:legalBodyName></lido:recordSource>
      <lido:recordMetadataDate>{now_str}</lido:recordMetadataDate>
    </lido:recordWrap>
    <lido:resourceWrap>
      <lido:resourceSet>
        <lido:resourceDescription>Normal solver: {html.escape(normal_engine)}. Depth: {html.escape(depth_engine)}. Image count: {int(n_images)}.</lido:resourceDescription>
      </lido:resourceSet>
    </lido:resourceWrap>
  </lido:administrativeMetadata>
</lido:lido>"""
    with open(path, 'w') as f:
        f.write(xml)
    return path


def iso_stats_csv(depth_vis, path: str, obj: str, *, crop_box=None, w: int = 0, h: int = 0,
                  z_mm_per_unit: float = 0.0) -> str:
    """The ISO 25178 areal parameters of the region as CSV, with their unit."""
    import csv
    from helio.core.metrology import iso25178
    if depth_vis is None:
        raise NothingToExport("Process images first.")
    x1, y1, x2, y2 = crop_box if crop_box else (0, 0, w or depth_vis.shape[1], h or depth_vis.shape[0])
    rows, unit = iso25178(np.asarray(depth_vis, np.float64)[y1:y2, x1:x2], z_mm_per_unit)
    z_mm = float(z_mm_per_unit or 0)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["# Helio — ISO 25178 areal parameters"])
        wr.writerow(["# object", obj])
        wr.writerow(["# region px", f"{x1},{y1},{x2},{y2}"])
        wr.writerow(["# height unit", unit])
        if z_mm <= 0:
            wr.writerow(["# NOTE", "Z is uncalibrated — heights are "
                         "relative. Calibrate in Metrology for mm."])
        wr.writerow([])
        wr.writerow(["parameter", "value", "unit", "meaning"])
        for name, val, mean in rows:
            wr.writerow([name, f"{val:.6g}", "" if "unitless" in mean else unit, mean])
    return path


PLATE_ENGINES = ("Standard PBR", "Raking Composite", "Normals RGB",
                 "Surface Curvature (Scratches)", "Sky-View Factor (SVF)", "PCA 1")


def plate_figure_export(draw: Callable, engines, path: str, *, title: str,
                        mm_per_px_preview: float | None = None) -> tuple[str, int]:
    """Several views, captioned and scaled, as one plate.

    *draw(engine)* returns that view as uint8 RGB at preview size, drawn the
    way the app draws it; *mm_per_px_preview* is the scale at that size (None:
    no scale bar).  Returns (path, number of panels)."""
    from helio.io.figures import plate_figure
    panels = []
    for eng in engines:
        try:
            panels.append((eng, np.asarray(draw(eng))))
        except Exception:
            import logging
            logging.getLogger(__name__).warning("plate figure: %s", eng, exc_info=True)
    if not panels:
        raise NothingToExport("No engine rendered.")
    sheet = plate_figure(panels, cols=3, tile_px=900, mm_per_px=mm_per_px_preview, title=title)
    cv2.imwrite(path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    return path, len(panels)


def calibration_report(path: str, *, obj: str, w: int, h: int, n_images: int,
                       mm_per_px: float, z_mm_per_unit: float, lights, light_source: str,
                       normal_engine: str, depth_engine: str, lights_from_matrix: bool,
                       when: str | None = None) -> str:
    """Everything known about how the capture was measured, as plain text."""
    import time
    mm_pp, z_mm = float(mm_per_px or 0), float(z_mm_per_unit or 0)
    lv = [tuple(float(c) for c in v) for v in (lights if lights is not None else [])]
    L = ["Helio — calibration report",
         f"Object      : {obj}",
         f"Written     : {when or time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
         f"Capture     : {w} x {h} px, {n_images} image(s)",
         "",
         "SPATIAL SCALE",
         f"  mm per pixel : {mm_pp:.6f}" if mm_pp > 0 else
         "  mm per pixel : NOT MEASURED — draw a scale bar and calculate PPI"]
    if mm_pp > 0:
        L.append(f"  field of view: {w * mm_pp:.2f} x {h * mm_pp:.2f} mm")
        L.append(f"  resolution   : {25.4 / mm_pp:.0f} ppi ({mm_pp * 1000:.1f} um/px)")
    L += ["", "HEIGHT SCALE",
          f"  mm per pixel of height : {z_mm:.6f} (heights from the normals, scale measured)" if z_mm > 0 else
          "  mm per pixel of height : NOT MEASURED — heights in pixels of height, from the normals"]
    L += ["", "LIGHTING",
          f"  lights       : {len(lv)}",
          "  source       : " + {
              "measured": "chrome sphere or imported .lp",
              "dome": "placed on the light dome",
              "estimated": "recovered by PCA — an estimate, not a measurement",
          }.get(light_source, "none")]
    if lv:
        el = [float(np.degrees(np.arcsin(max(-1.0, min(1.0, v[2]))))) for v in lv]
        az = sorted(float(np.degrees(np.arctan2(v[0], -v[1])) % 360) for v in lv)
        gaps = [((az[(i + 1) % len(az)] - a) % 360) for i, a in enumerate(az)]
        L += [f"  elevation    : {min(el):.1f}-{max(el):.1f} deg (mean {sum(el)/len(el):.1f})",
              f"  largest gap  : {max(gaps):.1f} deg in azimuth",
              f"  source       : {'imported .lp / matrix' if lights_from_matrix else 'in-session'}"]
        if len(lv) < 24:
            L.append(f"  NOTE         : {len(lv)} lights is below the "
                     "24-60 usually recommended for RTI")
        if max(gaps) > 90:
            L.append(f"  NOTE         : a {max(gaps):.0f} deg gap "
                     "leaves one side of the surface poorly lit")
    else:
        L.append("  NOTE         : no light vectors — uncalibrated solve")
    if light_source == "estimated":
        L.append("  NOTE         : PCA recovers directions only up to a "
                 "rotation; treat these as indicative, not measured")
    L += ["", "SOLVERS", f"  normals      : {normal_engine}", f"  depth        : {depth_engine}", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def heritage_archive(path: str, albedo, normals, depth_vis, w: int, h: int, shared: dict, *,
                     object_id: str, normal_engine: str, depth_engine: str, crop_box=None,
                     forensic_maps=None, progress: Callable | None = None) -> str:
    """The archival ZIP: GLB, textures, ISO 25178 statistics, metadata and a README."""
    from helio.io.exporters import write_heritage_archive
    if normals is None:
        raise NothingToExport("Process scans first.")
    write_heritage_archive(
        path, albedo=albedo, normals=normals, depth=depth_vis, depth_poisson=depth_vis,
        object_id=object_id or "HELIO_Object",
        crop_box=trim_box(w, h, shared['edge_trim'], crop_box),
        export_step=max(1, shared['export_step']), height_scale=shared['height'],
        normal_engine=normal_engine, depth_engine=depth_engine,
        forensic_maps=forensic_maps or None, progress_cb=progress)
    return path


PROVENANCE_REFERENCES = [
    {"citation": "Gaiani, M.; Angeletti, A.; Garagnani, S. Photometric Stereo Techniques for the 3D "
                 "Reconstruction of Paintings and Drawings Through the Measurement of Custom-Built "
                 "Repro Stands. Heritage 2025, 8, 129.",
     "doi": "10.3390/heritage8040129",
     "relevance": "Near-field PS model, measured PoI calibration, flat reference correction, "
                  "parabolic baseline"},
    {"citation": "Laurent, L. et al. Combining Geometric and Photometric 3D Reconstruction "
                 "Techniques for Cultural Heritage Documentation. J. Cult. Herit. 2025.",
     "doi": "hal-04972777v1",
     "relevance": "PG+PS calibration, low-frequency bias diagnostic"},
    {"citation": "ISO 25178-2:2021. Geometrical product specifications (GPS) — Surface texture: "
                 "Areal — Part 2.",
     "relevance": "Surface parameter definitions (Sa, Sq, Ssk, Sku, Sz)"},
]


def write_processing_record(path: str, record: dict) -> tuple[str, str]:
    """The reproducibility record as JSON, with a readable TXT companion beside it."""
    import json
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)
    txt_p = path.replace(".json", ".txt")
    with open(txt_p, 'w') as f:
        f.write(f"Helio — Processing Provenance Record\n{'=' * 55}\n\n")

        def section(title, d, indent=0):
            pad = "  " * indent
            f.write(f"\n{pad}{title}\n{pad}{'─' * 40}\n")
            for k, v in d.items():
                if isinstance(v, dict):
                    section(k, v, indent + 1)
                elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                    f.write(f"{pad}  {k:<30} {', '.join(v)}\n")
                else:
                    f.write(f"{pad}  {k:<30} {v}\n")
        for name, content in record.items():
            if isinstance(content, dict):
                section(name.upper().replace("_", " "), content)
            else:
                f.write(f"  {name:<30} {content}\n")
    return path, txt_p


GIS_MAPS = (("1_positive_openness_32f.tif", "_pos_opn"), ("2_negative_openness_32f.tif", "_neg_opn"),
            ("3_multi_scale_relief_msrm_32f.tif", "_msrm"), ("4_surface_curvature_32f.tif", "_curv"),
            ("5_slope_gradient_32f.tif", "_slope"), ("6_local_dominance_32f.tif", "_loc_dom"),
            ("7_sky_view_factor_32f.tif", "_svf"))


def gis_analysis_maps(out_dir: str, terrain: dict, w: int, h: int, *, obj_id: str = "",
                      mm_per_px: float = 0.0) -> list[str]:
    """Seven 32-bit terrain-analysis GeoTIFFs at full resolution.

    *terrain* is the solve's terrain dict (keys _svf, _pos_opn, ...).  The
    HELIO_RES_MM tag is the measured mm per pixel, or 1.0 when no scale is
    set (it used to divide the image width by the scale line's length: KD-016)."""
    from helio.io.exporters import write_geotiff
    svf = terrain.get("_svf")
    if svf is None:
        raise NothingToExport("The terrain analysis caches are empty — re-run processing.")

    def up(arr):
        if arr.shape != (h, w):
            return cv2.resize(arr, (w, h), interpolation=cv2.INTER_LANCZOS4)
        return arr
    zeros = np.zeros((h, w), np.float32)
    fallback = {"_pos_opn": svf, "_neg_opn": 1.0 - svf, "_loc_dom": svf}
    res_mm = float(mm_per_px) if mm_per_px and mm_per_px > 0 else 1.0
    tags = dict(HELIO_OBJECT_ID=obj_id or "", HELIO_RES_MM=f"{res_mm:.6f}",
                HELIO_CRS="EPSG:32630 (pixel size = 1.0, physical scale in HELIO_RES_MM)")
    out = []
    for name, key in GIS_MAPS:
        arr = terrain.get(key)
        arr = fallback.get(key, zeros) if arr is None else arr
        path = os.path.join(out_dir, name)
        write_geotiff(path, up(arr).astype(np.float32),
                      tags={**tags, "HELIO_LAYER": os.path.splitext(name)[0]})
        out.append(path)
    return out
