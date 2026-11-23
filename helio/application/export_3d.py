"""3D mesh exports, GUI-free: OBJ, PLY and GLB meshes, PLY point clouds and
the Sketchfab PBR package (EXM-001…003).

The classic _export_3d's worker, moved here unchanged apart from where it
reads its inputs, so both apps write the same mesh: the depth grid
(shape/detail blend, HD Frankot–Chellappa, hybrid blend, high-frequency
amplification, taco fix, smoothing, broad warp removal, fine detail boost,
baked normal detail), the watertight solid with consistent winding, the
textured GLB, the OBJ with its MTL and maps, the point cloud, and the PBR
package and its ZIP.

Two changes from HELIO 2: the AO blend setting was read with
``hasattr(dict, 'ao_blend')``, always False, so it never applied; and the
elevation-heatmap texture was lit from below in the normals' frame (KD-032).
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from helio.core.frames import depth_slopes
from helio.core.depth_integration import compute_hybrid_depth_blend
from helio.core.processing import fast_large_sigma_blur

_log = logging.getLogger(__name__)

# The export panel's choices, as the classic app starts.
MESH_DEFAULTS = {
    "hf_cutoff": 0.05, "amp": 1.0, "brightness": 1.0, "hybrid_depth": False,
    "vertex_colors": True, "watertight": True, "black_sides": True, "flip_y": True,
    "sketchfab": False, "sketchfab_jpeg": False, "pbr_ao": True, "pbr_disp": True,
    "pbr_render": True, "gamma_encode": False, "heatmap": False, "tex_res": "Full",
    "use_hd": False, "hd_warp_pct": 0.18, "detail_mix": 0.12, "center_origin": True,
    "bake_normals": False, "bake_depth_pct": 0.05, "render_engine": "", "view_mode": "Shading",
    # The colour and normal textures: "mesh" at the mesh's own density (as
    # before), else from the full-resolution photograph over what the mesh
    # covers, at most this many pixels on the longer side ("full": no limit).
    "tex_size": "mesh",
}
TEXTURE_SIZES = ("mesh", "2048", "4096", "8192", "full")


def texture_grid(img, h: int, w: int, step: int, size="mesh"):
    """The texture of a mesh sampled every *step* pixels of an (h, w) region.

    "mesh": the samples themselves, one texel a point.  Otherwise the image
    at full resolution over exactly the rows and columns the mesh spans (so
    its 0-1 UVs land on the same pixels), reduced to *size* on the longer
    side when larger ("full": as it is)."""
    if size in (None, "", "mesh"):
        return img[0:h:step, 0:w:step]
    gh, gw = (h - 1) // step + 1, (w - 1) // step + 1
    cov = img[:(gh - 1) * step + 1, :(gw - 1) * step + 1]
    if size != "full":
        cap = int(size)
        k = cap / float(max(cov.shape[:2]))
        if k < 1.0:
            cov = cv2.resize(np.ascontiguousarray(cov), (max(1, round(cov.shape[1] * k)),
                                                          max(1, round(cov.shape[0] * k))),
                             interpolation=cv2.INTER_AREA)
    return cov
FORMATS = {"glb": ".glb", "obj": ".obj", "ply": ".ply", "point": ".ply"}


def _fast_io():
    """helio_core_rs when it has the mesh writers, else None (the Python writers run)."""
    from helio.core._native import rs
    return rs if rs is not None and hasattr(rs, "write_obj") else None


def _u32_faces(faces) -> np.ndarray:
    return np.ascontiguousarray(faces, dtype=np.uint32)


def _f64(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _export_trimesh_obj(mesh, path: str) -> None:
    """``mesh.export(path)`` for an OBJ, the same text as trimesh writes
    (``v x y z [r g b]`` to 8 decimals, ``f a b c``) but formatted on every core
    when helio_core_rs is there - the 4-million-vertex mesh of a 12 MP capture
    takes minutes through trimesh's formatting.  Meshes with normals or UVs, and
    anything the writer cannot take, go through trimesh as before."""
    rs = _fast_io()
    vis = getattr(mesh, "visual", None)
    colours = getattr(vis, "vertex_colors", None) if getattr(vis, "kind", None) == "vertex" else None
    if (rs is None or "vertex_normals" in mesh._cache.cache or getattr(vis, "uv", None) is not None
            or (getattr(vis, "kind", None) not in (None, "vertex") and colours is None)):
        mesh.export(path)
        return
    try:
        rs.write_obj(path, _f64(mesh.vertices), _u32_faces(mesh.faces),
                     header="# https://github.com/mikedh/trimesh\n", decimals=8,
                     colors=None if colours is None else np.ascontiguousarray(colours[:, :3], np.uint8),
                     face_style="v", footer="\n")
    except Exception:
        mesh.export(path)


class _V:
    """A plain value where the classic read a multiprocessing value."""
    __slots__ = ("value",)

    def __init__(self, v):
        self.value = v


def mesh_export(D, filepath: str, file_format: str, *, ss: dict, options: dict | None = None,
                depth_normalised=None, crop=None, render: Callable | None = None,
                progress: Callable | None = None) -> dict:
    """Write the mesh (or point cloud) to *filepath*.

    *D* is the solve's data (normals_full, albedo_rgb_full, depth_vis_p,
    roughness_full, spec_var); *ss* the 3D settings as plain values (the
    classic shared state); *options* the export panel's choices
    (``MESH_DEFAULTS``); *render(engine, h, w, crop, view_mode)* draws the
    baked render for the PBR package.  With the Sketchfab option the mesh
    goes into a ``<name>_PBR_Package`` folder beside *filepath*, zipped.

    Returns {"path", "export_dir", "zip", "pbr_error"}."""
    import trimesh
    progress = progress or (lambda *_a, **_k: None)
    _ex = dict(MESH_DEFAULTS, **(options or {}))
    export_dir = ""
    if _ex["sketchfab"] and file_format != "point":
        export_dir = os.path.join(os.path.dirname(filepath),
                                  f"{os.path.splitext(os.path.basename(filepath))[0]}_PBR_Package")
        os.makedirs(export_dir, exist_ok=True)
        filepath = os.path.join(export_dir, os.path.basename(filepath))
    result = {"path": filepath, "export_dir": export_dir, "zip": "", "pbr_error": ""}
    _ss = {k: _V(v) for k, v in ss.items()}
    step, height = max(1, _ss['export_step'].value), _ss['height'].value
    fx, fy, fz = _ss['fx'].value, _ss['fy'].value, _ss['fz'].value
    trim_pct = _ss['edge_trim'].value / 100.0
    micro = _ss['micro_relief'].value

    x1, y1, x2, y2 = crop if crop else (max(0, int(D.w * trim_pct)), max(0, int(D.h * trim_pct)), D.w - max(0, int(D.w * trim_pct)), D.h - max(0, int(D.h * trim_pct)))
    _crop_w, _crop_h = x2 - x1, y2 - y1
    _MAX_EXPORT_VERTS = 4000000
    _est_verts = (_crop_w // step) * (_crop_h // step)
    if _est_verts > _MAX_EXPORT_VERTS:
        import math as _m2
        _ratio = _m2.sqrt(_est_verts / _MAX_EXPORT_VERTS)
        step = max(step, int(_m2.ceil(step * _ratio)))
        _est_verts = (_crop_w // step) * (_crop_h // step)
        progress(f'Auto-decimating to step={step} ({_est_verts:,} verts)…', 0.05)
    albedo = D.albedo_rgb_full[y1:y2, x1:x2]
    if albedo is None:
        albedo = np.full((y2-y1, x2-x1, 3), 128, dtype=np.uint8)

    hf_cutoff = _ex['hf_cutoff']
    amp       = _ex['amp']
    smooth    = _ss['mesh_smooth'].value
    warp_rem  = _ss['warp_rem'].value
    sb        = max(0.0, min(1.0, _ss['shape_blend'].value))
    hp_sigma  = _ss['hp_sigma'].value

    use_hd    = _ex['use_hd']

    if use_hd:
        n_crop = D.normals_full[y1:y2, x1:x2].astype(np.float32)
        sh, sw = n_crop.shape[:2]
        nz_c  = np.maximum(n_crop[..., 2], 0.15)
        nx_sm = cv2.GaussianBlur(n_crop[..., 0], (0, 0), sigmaX=1.5)
        ny_sm = cv2.GaussianBlur(n_crop[..., 1], (0, 0), sigmaX=1.5)
        p_fc, q_fc = (np.clip(g, -5.0, 5.0) for g in depth_slopes(nx_sm, ny_sm, nz_c))
        u_f, v_f = np.meshgrid(np.fft.fftfreq(sw), np.fft.fftfreq(sh))
        denom_f  = u_f**2 + v_f**2; denom_f[0, 0] = 1.0
        Z_hat    = (-1j*u_f*np.fft.fft2(p_fc) - 1j*v_f*np.fft.fft2(q_fc)) / (denom_f + 1e-10)
        Z_hat[0, 0] = 0.0
        depth_fc = np.real(np.fft.ifft2(Z_hat)).astype(np.float32)
        _p1_fc, _p99_fc = np.percentile(depth_fc, 1), np.percentile(depth_fc, 99)
        _range_fc = _p99_fc - _p1_fc
        depth_fc  = np.clip(depth_fc, _p1_fc - _range_fc * 0.5,
                                       _p99_fc + _range_fc * 0.5)
        _hd_warp_pct = _ex['hd_warp_pct']
        sigma_macro = max(4.0, _hd_warp_pct * math.sqrt(sh**2 + sw**2))
        if sigma_macro > 4.0:
            base = depth_fc - fast_large_sigma_blur(depth_fc, sigma_macro)
        else:
            base = depth_fc
    else:
        d_hp_crop = D.depth_vis_p[y1:y2, x1:x2].astype(np.float32)
        if depth_normalised is not None:
            d_full_crop = depth_normalised[y1:y2, x1:x2].astype(np.float32)
        else:
            d_full_crop = d_hp_crop.copy()
        base = d_full_crop * (1.0 - sb) + d_hp_crop * sb
        sh, sw = base.shape
        if sb > 0.1:
            _prog_sigma = max(3.0, sb * 0.40 * math.sqrt(sh**2 + sw**2))
            base = base - fast_large_sigma_blur(base, _prog_sigma)

    _use_hybrid = (
        _ex['hybrid_depth']
        and depth_normalised is not None
        and getattr(D, 'depth_vis_p', None) is not None
    )
    if _use_hybrid:
        _detail_mix = _ex['detail_mix']
        _global_crop = depth_normalised[y1:y2, x1:x2].astype(np.float32)
        _flat_crop   = D.depth_vis_p[y1:y2, x1:x2].astype(np.float32)
        base = compute_hybrid_depth_blend(_global_crop, _flat_crop, _detail_mix)
        sh, sw = base.shape

    if amp != 1.0:
        sigma_hf = max(2.0, hf_cutoff * math.sqrt(sh * sh + sw * sw))
        hf_layer = base - fast_large_sigma_blur(base, sigma_hf)
        base = base + hf_layer * (amp - 1.0)

    rng = base.max() - base.min()
    raw_depth = ((base - base.min()) / rng).astype(np.float32) if rng > 1e-9 else np.zeros_like(base)
    h, w = raw_depth.shape

    progress("Decimating mesh…", 0.15)
    y_s, x_s = np.mgrid[0:h:step, 0:w:step]
    grid_h, grid_w = len(y_s[:,0]), len(x_s[0,:])
    z_raw = raw_depth[0:h:step, 0:w:step].copy()
    if hp_sigma > 0.01:
        sigma_px = float(hp_sigma) * max(grid_h, grid_w) * 0.4
        z_raw = z_raw - fast_large_sigma_blur(z_raw, sigma_px)
    if smooth > 0:
        z_raw = cv2.GaussianBlur(z_raw, (0, 0), sigmaX=float(smooth))
    if warp_rem > 0:
        z_raw = z_raw - fast_large_sigma_blur(z_raw, max(grid_h, grid_w) * (0.05 + warp_rem * 0.45))
    if micro > 0:
        base_lf = fast_large_sigma_blur(z_raw, max(grid_h, grid_w) * 0.05)
        z_raw = base_lf + (z_raw - base_lf) * (1.0 + micro)

    # Last, on the grid that becomes vertices: anything the filters
    # above would smooth away has already run, and z_raw is at the
    # final mesh resolution, so nothing is baked that decimation would
    # then alias.
    if _ex['bake_normals'] and _ex['bake_depth_pct'] > 0:
        progress("Baking normal detail into geometry…", 0.22)
        from helio.core.processing import bake_normal_detail
        z_raw = bake_normal_detail(
            z_raw, D.normals_full[y1:y2, x1:x2],
            depth_pct=_ex['bake_depth_pct'])

    verts_top = np.column_stack((
        ((x_s - w/2) * fx).flatten(),
        ((y_s - h/2) * fy).flatten(),
        ((z_raw * fz * height) - (height / 2)).flatten()
    )).astype(np.float32)

    if _ex['center_origin']:
        verts_top[:, 0] -= np.mean(verts_top[:, 0]); verts_top[:, 1] -= np.mean(verts_top[:, 1]); verts_top[:, 2] -= np.mean(verts_top[:, 2])

    if len(albedo.shape) == 2: albedo = np.stack((albedo,)*3, axis=-1)
    _exp_bright = _ex['brightness']
    _alb_export = albedo[0:h:step, 0:w:step].astype(np.float32)
    if _exp_bright != 1.0:
        _alb_export = np.power(np.clip(_alb_export / 255.0, 0, 1), 1.0 / max(0.1, _exp_bright)) * 255.0
    colors_top_8bit = np.clip(_alb_export, 0, 255).reshape(-1, 3).astype(np.uint8) if _ex['vertex_colors'] else np.full((len(verts_top), 3), 150, dtype=np.uint8)
    _ao_e=_ss['ao_blend'].value if 'ao_blend' in _ss else 0.
    if _ao_e>0.01 and _ex['vertex_colors']:
        _gz_e,_gx_e=np.gradient(z_raw)
        _ao_m=np.power(np.clip(1./np.sqrt(1.+_gx_e**2+_gz_e**2),0.,1.),2.5).flatten()
        _cf=colors_top_8bit.astype(np.float32)
        colors_top_8bit=np.clip(
            _ao_e*(_cf*_ao_m[:,np.newaxis])+(1.-_ao_e)*_cf,0,255).astype(np.uint8)

    progress("Building geometry…", 0.30)
    if file_format == "point":
        trimesh.points.PointCloud(verts_top, colors=np.column_stack((colors_top_8bit, np.full(len(colors_top_8bit), 255))).astype(np.uint8)).export(filepath)
    else:
        rr, cc = np.meshgrid(np.arange(grid_h - 1), np.arange(grid_w - 1), indexing='ij')
        v0 = rr * grid_w + cc
        faces_top = np.vstack((np.column_stack((v0.flatten(), (v0+grid_w).flatten(), (v0+1).flatten())), np.column_stack(((v0+1).flatten(), (v0+grid_w).flatten(), (v0+grid_w+1).flatten())))).astype(np.int32)
        if (fx * fy * fz) < 0: faces_top = faces_top[:, [0, 2, 1]]
        # This block is a hand-inlined copy of build_watertight_mesh_data,
        # and the winding fix in commit 11908bd went into the library
        # version only - which Helio_Pro.py never calls.  So the GUI's
        # OBJ / PLY / GLB buttons kept exporting a solid whose surfaces
        # faced inward while its walls faced out: winding inconsistent
        # in all eight fx/fy/fz combinations, signed volume 10,835
        # against a true 38,036, and inside out whenever fz < 0.
        #
        # The two corrections the library carries, applied here:
        # reverse the surfaces unconditionally, and reverse everything
        # again when Z is mirrored (below, after the faces are stacked).
        faces_top = faces_top[:, [0, 2, 1]]

        uv_cols = np.tile(np.linspace(0, 1, grid_w), grid_h)
        uv_rows = np.repeat(np.linspace(0, 1, grid_h), grid_w)
        uv_top = np.column_stack((uv_cols, uv_rows)).astype(np.float32)

        if _ex['watertight']:
            verts_bottom = verts_top.copy(); verts_bottom[:, 2] = np.min(verts_top[:, 2]) - ((np.max(verts_top[:, 2]) - np.min(verts_top[:, 2])) * 0.2 + 2.0)
            faces_bottom = faces_top[:, [0, 2, 1]] + len(verts_top)
            top_edge, right_edge = np.arange(grid_w), np.arange(grid_w - 1, grid_h * grid_w, grid_w)
            bottom_edge, left_edge = np.arange(grid_h * grid_w - 1, (grid_h - 1) * grid_w - 1, -1), np.arange((grid_h - 1) * grid_w, -1, -grid_w)
            perimeter = np.concatenate([top_edge[:-1], right_edge[:-1], bottom_edge[:-1], left_edge[:-1]])
            N    = len(verts_top)
            P    = len(perimeter)
            flip = (fx * fy * fz) < 0

            verts_side_top = verts_top[perimeter].copy()
            verts_side_bot = verts_bottom[perimeter].copy()

            base_st = 2 * N
            base_sb = 2 * N + P

            # Vectorized side face generation - much faster than Python loop
            st0_arr = base_st + np.arange(P)
            st1_arr = base_st + np.arange(1, P + 1) % P
            sb0_arr = base_sb + np.arange(P)
            sb1_arr = base_sb + np.arange(1, P + 1) % P

            if flip:
                side_faces = np.column_stack([
                    np.column_stack([st0_arr, st1_arr, sb0_arr]),
                    np.column_stack([st1_arr, sb1_arr, sb0_arr])
                ]).reshape(-1, 3)
            else:
                side_faces = np.column_stack([
                    np.column_stack([st0_arr, sb0_arr, st1_arr]),
                    np.column_stack([st1_arr, sb0_arr, sb1_arr])
                ]).reshape(-1, 3)

            side_arr = side_faces.astype(np.int32)

            all_verts  = np.vstack([verts_top, verts_bottom,
                                    verts_side_top, verts_side_bot])
            all_faces  = np.vstack([faces_top, faces_bottom, side_arr])
            # Mirroring Z turns the solid inside out - the bottom slab
            # ends up above the surface.  Winding stays self-consistent,
            # so nothing looks wrong, but the signed volume goes
            # negative and every normal points into the object.
            if fz < 0:
                all_faces = all_faces[:, [0, 2, 1]]

            colors_black = np.zeros((2 * P, 3), dtype=np.uint8)
            all_colors = np.vstack([colors_top_8bit,
                                    np.full_like(colors_top_8bit, 30),
                                    colors_black])

            _black_sides_on = _ex['black_sides']
            _bc = 0.002
            _cc = 0.5

            uv_bottom_arr    = np.full((N, 2), _bc if _black_sides_on else _cc, dtype=np.float32)
            uv_side_top_arr  = np.full((P, 2), _bc if _black_sides_on else _cc, dtype=np.float32)
            uv_side_bot_arr  = np.full((P, 2), _bc if _black_sides_on else _cc, dtype=np.float32)
            all_uvs = np.vstack([uv_top, uv_bottom_arr,
                                 uv_side_top_arr, uv_side_bot_arr])

            verts_final  = all_verts
            faces_final  = all_faces
            colors_final = all_colors
            _n_top_verts = N
            _all_uvs     = all_uvs
        else:
            verts_final, faces_final, colors_final = verts_top, faces_top, colors_top_8bit
            _n_top_verts = len(verts_top)
            _all_uvs = uv_top

        n_grid = D.normals_full[y1:y2, x1:x2][0:h:step, 0:w:step].copy().astype(np.float32)
        n_grid[:, :, 0] *= fx
        n_grid[:, :, 1] *= fy
        n_grid[:, :, 2] *= fz
        if np.mean(n_grid[:, :, 2]) < 0:
            n_grid = -n_grid
        nlen = np.linalg.norm(n_grid, axis=-1, keepdims=True)
        nlen = np.where(nlen < 1e-9, 1.0, nlen)
        vertex_normals_top = (n_grid / nlen).reshape(-1, 3).astype(np.float32)

        _tsize = _ex.get('tex_size', 'mesh')
        _alb_grid = texture_grid(albedo, h, w, step, _tsize).astype(np.uint8)
        if _exp_bright != 1.0:
            _alb_grid = np.clip(
                np.power(np.clip(_alb_grid.astype(np.float32) / 255.0, 0, 1),
                         1.0 / max(0.1, _exp_bright)) * 255.0,
                0, 255).astype(np.uint8)
        albedo_texture_img = Image.fromarray(_alb_grid)
        if _ex['black_sides'] and _ex['watertight']:
            _alb_patched = np.array(albedo_texture_img)
            _patch_sz = max(4, _alb_patched.shape[0] // 64)
            _alb_patched[:_patch_sz, :_patch_sz] = 0
            albedo_texture_img = Image.fromarray(_alb_patched)

        n_tex = texture_grid(D.normals_full[y1:y2, x1:x2], h, w, step, _tsize).astype(np.float32)
        if _tsize not in (None, "", "mesh"):        # resampled: unit length again
            n_tex /= np.maximum(np.linalg.norm(n_tex, axis=-1, keepdims=True), 1e-9)
        nm_rgb = np.round((n_tex * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
        nm_rgb_embedded = nm_rgb.copy()
        nm_rgb_separate = nm_rgb.copy()
        if _ex['flip_y']:
            nm_rgb_separate = nm_rgb_separate.copy()
            nm_rgb_separate[..., 1] = 255 - nm_rgb_separate[..., 1]
        normal_texture_img = Image.fromarray(nm_rgb_embedded)

        is_glb = filepath.lower().endswith('.glb') or filepath.lower().endswith('.gltf')

        vn_flat = np.array([0., 0., -1.], dtype=np.float32)
        if _ex['watertight']:
            n_bottom = np.tile(vn_flat, (len(verts_final) - _n_top_verts, 1))
            all_vnormals = np.vstack([vertex_normals_top, n_bottom])
        else:
            all_vnormals = vertex_normals_top

        progress("Exporting mesh…", 0.50)
        if is_glb:
            try:
                pbr_mat = trimesh.visual.material.PBRMaterial(
                    name='helio_pbr',
                    baseColorTexture=albedo_texture_img,
                    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                    normalTexture=normal_texture_img,
                    metallicFactor=0.0,
                    roughnessFactor=0.65,
                    doubleSided=True,
                )
                # trimesh expects OpenGL-convention UVs (v=0 = image
                # bottom) and writes 1-v to glTF; ours are image-
                # convention (v=0 = top), so flip V or the texture
                # renders vertically mirrored — see _uv_to_trimesh
                # in helio/io/exporters.py for the full derivation.
                from helio.io.exporters import _uv_to_trimesh
                tex_visual = trimesh.visual.TextureVisuals(uv=_uv_to_trimesh(_all_uvs), material=pbr_mat)
                mesh_final = trimesh.Trimesh(
                    vertices=verts_final,
                    faces=faces_final,
                    vertex_normals=all_vnormals[:len(verts_final)],
                    process=False)
                mesh_final.visual = tex_visual
            except Exception as _tex_err:
                print(f"[Helio] Textured GLB failed ({_tex_err}), falling back to vertex colours")
                mesh_final = trimesh.Trimesh(
                    vertices=verts_final,
                    faces=faces_final,
                    vertex_normals=all_vnormals[:len(verts_final)],
                    vertex_colors=np.column_stack((colors_final, np.full(len(colors_final), 255))).astype(np.uint8),
                    process=False)
        else:
            mesh_final = trimesh.Trimesh(
                vertices=verts_final,
                faces=faces_final,
                vertex_normals=all_vnormals[:len(verts_final)],
                vertex_colors=np.column_stack((colors_final, np.full(len(colors_final), 255))).astype(np.uint8),
                process=False)

        # An OBJ with vertex colours is written below with its UVs, over whatever
        # trimesh would write here: writing it twice cost minutes at 4 M vertices.
        _obj_with_uvs = filepath.lower().endswith('.obj') and _ex['vertex_colors']
        if not _obj_with_uvs:
            mesh_final.export(filepath)

        if filepath.lower().endswith('.glb') or filepath.lower().endswith('.gltf'):
            from helio.io.exporters import _fix_glb_material
            _fix_glb_material(filepath, metallic=0.0, roughness=0.65, double_sided=True)

        if filepath.lower().endswith('.obj') and _ex['vertex_colors']:
            try:
                _bn  = os.path.splitext(os.path.basename(filepath))[0]
                _dir = os.path.dirname(filepath)
                _jpg = bool(_ex['sketchfab_jpeg'])     # the colour as JPEG; the normal map stays PNG
                _alb = f"{_bn}_albedo.{'jpg' if _jpg else 'png'}"
                cv2.imwrite(os.path.join(_dir, _alb), cv2.cvtColor(_alb_grid, cv2.COLOR_RGB2BGR),
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92] if _jpg else [])
                _nrm = f"{_bn}_normal.png"
                cv2.imwrite(os.path.join(_dir, _nrm),
                            cv2.cvtColor(nm_rgb_separate, cv2.COLOR_RGB2BGR))
                with open(os.path.join(_dir, f"{_bn}.mtl"), 'w') as _m:
                    _m.write(f"newmtl helio\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n")
                    _m.write(f"map_Kd {_alb}\nnorm {_nrm}\n")
                progress(f'Writing OBJ ({len(faces_final):,} faces)…', 0.65)
                _vn = all_vnormals[:len(verts_final)]
                _uvs_flip = np.column_stack([_all_uvs[:, 0], 1.0 - _all_uvs[:, 1]])
                _fi = faces_final.astype(np.int32) + 1
                progress(f'Writing {len(_fi):,} faces…', 0.75)
                _head = f"# Helio — Photometric Stereo 3D Export\nmtllib {_bn}.mtl\nusemtl helio\n"
                _rs_io = _fast_io()
                if _rs_io is not None:
                    _rs_io.write_obj(filepath, _f64(verts_final), _u32_faces(faces_final), header=_head,
                                     decimals=6, normals=_f64(_vn), normal_decimals=6,
                                     uvs=_f64(_uvs_flip), uv_decimals=6, face_style="v/vt/vn")
                else:
                    with open(filepath, 'w') as _of:
                        _of.write(_head)
                        # Bulk write vertices - much faster than np.savetxt
                        _of.writelines([f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts_final])
                        _of.writelines([f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n" for n in _vn])
                        _of.writelines([f"vt {uv[0]:.6f} {uv[1]:.6f}\n" for uv in _uvs_flip])
                        # Bulk write faces - much faster than np.savetxt
                        _of.writelines([f"f {f[0]}/{f[0]}/{f[0]} {f[1]}/{f[1]}/{f[1]} {f[2]}/{f[2]}/{f[2]}\n" for f in _fi])
            except Exception as _obj_err:
                print(f"[Helio] OBJ UV rewrite warning: {_obj_err}")
                mesh_final.export(filepath)          # the plain OBJ, so a failed rewrite still leaves a file

    try:
        if _ex['sketchfab'] and file_format != "point":
            progress("Building Sketchfab PBR package…", 0.85)
            base_name  = os.path.splitext(os.path.basename(filepath))[0]
            include_ao   = _ex['pbr_ao']
            include_disp = _ex['pbr_disp']
            gamma_encode = _ex['gamma_encode']
            flip_y       = _ex['flip_y']

            def _encode_for_srgb_renderer(nm_uint8_rgb):
                lf = nm_uint8_rgb.astype(np.float32) / 255.0
                encoded = np.power(np.clip(lf, 0, 1), 1.0 / 2.2) * 255.0
                return np.round(encoded).astype(np.uint8)

            def _tex_resize(img_bgr, res):
                # Roughness, specular and AO are single-channel, and
                # the old body indexed shape[2] unconditionally - which
                # is why they were the maps that never got resized.
                # The longer side at most *res*, the shape kept: the mesh's UVs
                # span the whole image, so padding it to a square (as this
                # did) drew the colour off the model.  Never enlarged.
                if res == 'Full':
                    return img_bgr
                sz = int(res); ih, iw = img_bgr.shape[:2]
                sc = sz / max(ih, iw)
                if sc >= 1.0:
                    return img_bgr
                nh, nw = max(1, round(ih * sc)), max(1, round(iw * sc))
                return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

            _tres = _ex['tex_res']
            if _ex.get('tex_size') not in (None, "", "mesh"):   # Publish's texture size
                _tres = 'Full' if _ex['tex_size'] == 'full' else str(_ex['tex_size'])

            # ── Texture format ──────────────────────────────────
            #
            # "JPEG textures" was drawn, given a tooltip stating
            # exactly what it should do, and never read on this path.
            # (_build_export_settings does read it, but that serves the
            # *web viewer* exports - a different function entirely, and
            # the reason this looked wired.) Every write below was a
            # hard-coded .png, so ticking it changed nothing.
            #
            # The rules are the tooltip's own: JPEG Q90 for colour-ish
            # maps, normal map always PNG because JPEG ringing corrupts
            # surface direction, and AO + displacement dropped entirely
            # for the smallest possible archive.
            _use_jpg = _ex['sketchfab_jpeg']
            _tex_ext = ".jpg" if _use_jpg else ".png"
            _tex_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90] if _use_jpg else []

            def _write_tex(name, img, force_png=False, params=None):
                """One texture, at the chosen resolution and format."""
                ext = ".png" if force_png else _tex_ext
                pth = os.path.join(export_dir, f"{base_name}_{name}{ext}")
                cv2.imwrite(pth, _tex_resize(img, _tres),
                            params if params is not None
                            else ([] if force_png else _tex_params))
                return os.path.basename(pth)

            _written = {}
            _written["baseColor"] = _write_tex(
                "baseColor", cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR))

            n_crop = D.normals_full[y1:y2, x1:x2]
            nm_8 = np.round((n_crop * 0.5 + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
            if flip_y:
                nm_8 = nm_8.copy()
                nm_8[..., 1] = 255 - nm_8[..., 1]
            if gamma_encode:
                nm_8 = _encode_for_srgb_renderer(nm_8)
            _written["normal"] = _write_tex(
                "normal", cv2.cvtColor(nm_8, cv2.COLOR_RGB2BGR), force_png=True)

            if getattr(D, 'roughness_full', None) is not None:
                rough_8 = (D.roughness_full[y1:y2, x1:x2].astype(np.float32)
                           / 255.0 * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rough_8 = np.full((y2 - y1, x2 - x1), 128, dtype=np.uint8)
            _written["roughness"] = _write_tex("roughness", rough_8)

            if getattr(D, 'spec_var', None) is not None:
                spec_8 = cv2.normalize(D.spec_var[y1:y2, x1:x2], None, 0, 255,
                                       cv2.NORM_MINMAX).astype(np.uint8)
            else:
                spec_8 = np.full((y2 - y1, x2 - x1), 128, dtype=np.uint8)
            _written["specular"] = _write_tex("specular", spec_8)

            if _ex['pbr_render']:
                try:
                    _reng  = _ex['render_engine']
                    _safe_eng = _reng.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
                    _rh, _rw  = y2 - y1, x2 - x1
                    # The caller's renderer, given the engine and view mode read
                    # before the worker started (a Tk read here would raise).
                    if render is None:
                        raise RuntimeError("no renderer for the baked render")
                    _rimg  = render(_reng, _rh, _rw, (x1, y1, x2, y2), _ex['view_mode'])
                    _rimg_bgr = cv2.cvtColor(_rimg, cv2.COLOR_RGB2BGR)
                    _written["render"] = _write_tex(
                        f"render_{_safe_eng}", _rimg_bgr)
                except Exception as _re:
                    # A bare print of the message hid *where* it came
                    # from for as long as this option has existed.
                    _log.exception("baked render export failed")
                    import traceback as _brt
                    _brt.print_exc()
                    print(f"[Helio] Baked render export skipped: {_re}")

            # AO and displacement are excluded in JPEG mode, which is
            # what the checkbox's tooltip promises and the reason the
            # mode exists: the smallest archive Sketchfab will take.
            if include_ao and not _use_jpg:
                n_f = n_crop.astype(np.float32)
                nz  = np.clip(n_f[..., 2], 0.0, 1.0)
                ao  = np.power(nz, 1.5)
                ao_blur = fast_large_sigma_blur(ao, max(5.0, 0.02 * math.sqrt(n_f.shape[0]**2 + n_f.shape[1]**2)))
                ao_contrast = np.clip(ao - ao_blur * 0.35 + 0.35, 0, 1)
                ao_8 = (ao_contrast * 255.0).clip(0, 255).astype(np.uint8)
                _written["ao"] = _write_tex("ao", ao_8, force_png=True)

            if include_disp and not _use_jpg:
                # 16-bit, so PNG regardless: JPEG cannot carry it.
                _written["displacement"] = f"{base_name}_displacement.png"
                cv2.imwrite(os.path.join(export_dir, _written["displacement"]),
                            (raw_depth * 65535).astype(np.uint16))

            if _ex['heatmap']:
                try:
                    _d_heat = np.clip(D.depth_vis_p[y1:y2, x1:x2], 0, 1)
                    _heat_bgr = cv2.applyColorMap((_d_heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
                    _heat_rgb = cv2.cvtColor(_heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    _n = D.normals_full[y1:y2, x1:x2].astype(np.float32)
                    # From the upper right in the normals' +Y-up frame; HELIO 2's
                    # -0.5 lit it from below, which reads relief as inverted (KD-032).
                    _lx_h, _ly_h, _lz_h = 0.5, 0.5, 0.7
                    _nl = np.sqrt(max(1e-6, _lx_h**2 + _ly_h**2 + _lz_h**2))
                    _diff = np.clip(_n[...,0]*(_lx_h/_nl) + _n[...,1]*(_ly_h/_nl) + _n[...,2]*(_lz_h/_nl), 0, 1)
                    _shade = (np.clip(_diff * 0.85 + 0.15, 0, 1))[..., np.newaxis]
                    _composite = np.clip(_heat_rgb * _shade, 0, 1)
                    _comp_bgr = cv2.cvtColor((_composite * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    _written["heatmap"] = _write_tex(
                        "elevation_heatmap", _comp_bgr)
                except Exception as _he:
                    print(f"[Helio] Heatmap composite export skipped: {_he}")

            # The .mtl must name the files that were actually written.
            # It used to hard-code ".png" for every map, so in JPEG mode
            # it would have pointed at files that do not exist - and in
            # PNG mode it still listed AO and displacement even when
            # those boxes were unticked and nothing had been written.
            _base_tex = _written.get("baseColor", f"{base_name}_baseColor.png")
            mtl_lines = (
                f"# Helio — Wavefront MTL\n"
                f"newmtl material_0\n"
                f"Ka 1.0 1.0 1.0\n"
                f"Kd 1.0 1.0 1.0\n"
                f"Ks 0.0 0.0 0.0\n"
                f"d 1.0\n"
                f"illum 1\n"
                f"map_Kd {_base_tex}\n"
            )
            if _written.get("normal"):
                mtl_lines += f"map_Bump {_written['normal']}\n"
            if _written.get("ao"):
                mtl_lines += f"map_Ka {_written['ao']}\n"
            else:
                mtl_lines += f"map_Ka {_base_tex}\n"
            if _written.get("displacement"):
                mtl_lines += f"disp {_written['displacement']}\n"
            with open(os.path.join(export_dir, f"{base_name}.mtl"), 'w') as f:
                f.write(mtl_lines)
            if file_format == "obj" and os.path.exists(filepath):
                with open(filepath, 'r') as f_in, open(filepath + ".tmp", 'w') as f_out:
                    f_out.write(f"mtllib {base_name}.mtl\nusemtl material_0\n")
                    shutil.copyfileobj(f_in, f_out)
                os.replace(filepath + ".tmp", filepath)

            # ── Actually produce the ZIP the checkbox promises ───
            #
            # The control is labelled "Sketchfab PBR Package (ZIP)" and
            # this function had no ZipFile in it anywhere: it wrote a
            # _PBR_Package *folder* of loose files and stopped. The
            # tooltip even said "upload the entire folder", contradicting
            # the label it sat under. Sketchfab takes a single archive,
            # so the folder was the one thing you could not upload.
            #
            # The folder is kept as well as zipped - it costs nothing,
            # and anyone who was working with the loose maps still can.
            progress("Zipping PBR package…", 0.95)
            _zip_path = export_dir + ".zip"
            try:
                import zipfile as _zf
                with _zf.ZipFile(_zip_path, "w", _zf.ZIP_DEFLATED) as _z:
                    for _root, _dirs, _files in os.walk(export_dir):
                        for _fn in sorted(_files):
                            if _fn.startswith("."):
                                continue
                            _full = os.path.join(_root, _fn)
                            _z.write(_full, os.path.relpath(_full, export_dir))
                # Verify rather than assume: a zip that cannot be read
                # back is worse than none, because it looks like success.
                result["zip"] = _zip_path
                with _zf.ZipFile(_zip_path) as _z:
                    _bad = _z.testzip()
                    if _bad:
                        raise RuntimeError(f"corrupt entry {_bad}")
                    _n_entries = len(_z.namelist())
                _log.info("Sketchfab PBR zip: %s (%d entries, %.1f MB)",
                         _zip_path, _n_entries,
                         os.path.getsize(_zip_path) / 1e6)
            except Exception as _ze:
                _log.exception("Sketchfab PBR zip failed")
                raise RuntimeError(
                    f"the texture maps were written to\n{export_dir}\n"
                    f"but the ZIP could not be created: {_ze}") from _ze
    except Exception as _pbr_err:
        _log.exception("mesh_export PBR package")
        result["pbr_error"] = str(_pbr_err)
    progress("Done.", 1.0)
    return result


# ── shared pieces of the packages ────────────────────────────────────────

def region(D, edge_trim_pct: float, crop=None):
    """The drawn crop, else the frame less the edge trim (as every 3D export)."""
    if crop:
        return tuple(int(c) for c in crop)
    t = edge_trim_pct / 100.0
    x1, y1 = max(0, int(D.w * t)), max(0, int(D.h * t))
    return x1, y1, D.w - x1, D.h - y1


def detail_depth(n_crop, sigma_frac: float, floor: float):
    """Frankot–Chellappa integration of +Y-up normals, high-passed at
    *sigma_frac* of the diagonal (the integration bowl removed), 0–1."""
    from helio.core.depth_integration import frankot_chellappa
    sh, sw = n_crop.shape[:2]
    nz = np.maximum(n_crop[..., 2], 0.001)
    p, q = depth_slopes(n_crop[..., 0], n_crop[..., 1], nz)
    d = frankot_chellappa(p, q)
    sigma = max(floor, sigma_frac * math.sqrt(sh * sh + sw * sw))
    hf = d - fast_large_sigma_blur(d, sigma)
    rng = hf.max() - hf.min()
    return ((hf - hf.min()) / rng).astype(np.float32) if rng > 1e-9 else np.zeros((sh, sw), np.float32), sigma


def _grid_faces(gH: int, gW: int, fx: int, fy: int, fz: int = 1) -> np.ndarray:
    """Two triangles per grid cell, wound so the surface faces up (+z) for
    any flip of X, Y or Z: the rule _export_3d arrived at.  The LOD and
    displacement meshes reversed the winding on the flip product alone and
    so, at the default flips, faced down (KD-034)."""
    rr, cc = np.meshgrid(np.arange(gH - 1), np.arange(gW - 1), indexing='ij')
    v0 = rr * gW + cc
    faces = np.vstack((np.column_stack((v0.ravel(), (v0 + gW).ravel(), (v0 + 1).ravel())),
                       np.column_stack(((v0 + 1).ravel(), (v0 + gW).ravel(), (v0 + gW + 1).ravel())))
                      ).astype(np.int64)
    if fx * fy * fz > 0:
        faces = faces[:, [0, 2, 1]]
    return faces


# ── 3D print package ─────────────────────────────────────────────────────

PRINT_DEFAULTS = {"size_mm": 100.0, "unit": "mm", "z_exp": 1.0, "base_mm": 3.0, "step": 2,
                  "fmt": "STL", "watertight": True, "colour": True, "z_scale": 2.0}


def print_mesh(D, out_path: str, *, edge_trim_pct: float = 2.0, crop=None,
               options: dict | None = None) -> dict:
    """The classic 3D Print export: high-passed Frankot–Chellappa relief on a
    flat base, sized in millimetres, the solve's normals as vertex normals,
    vertex colours, and a watertight base.

    HELIO 2's walls were wound the other way from its top and bottom and
    listed one corner twice, so the solid was neither consistent nor closed
    and its volume came out negative (KD-033)."""
    import trimesh
    o = dict(PRINT_DEFAULTS, **(options or {}))
    size_mm = float(o["size_mm"]) * {"cm": 10.0, "in": 25.4}.get(o["unit"], 1.0)
    step = max(1, int(o["step"]))
    x1, y1, x2, y2 = region(D, edge_trim_pct, crop)
    n_crop = D.normals_full[y1:y2, x1:x2].astype(np.float32)
    sh, sw = n_crop.shape[:2]
    from helio.core.depth_integration import frankot_chellappa
    p_fc, q_fc = depth_slopes(n_crop[..., 0], n_crop[..., 1], np.maximum(n_crop[..., 2], 0.001))
    depth_int = frankot_chellappa(p_fc, q_fc)
    base = depth_int - fast_large_sigma_blur(depth_int, max(5.0, 0.35 * math.sqrt(sh * sh + sw * sw)))
    if o["z_exp"] > 1.0:
        hf = base - fast_large_sigma_blur(base, max(2.0, 0.05 * math.sqrt(sh * sh + sw * sw)))
        base = base + hf * (o["z_exp"] - 1.0)
    rng = base.max() - base.min()
    raw = ((base - base.min()) / rng).astype(np.float32) if rng > 1e-9 else np.zeros_like(base)
    zd = raw[0:sh:step, 0:sw:step].copy()
    gh, gw = zd.shape
    mm_per_px = size_mm / max(gw, gh)
    relief_mm = min(size_mm * 0.08 * max(o["z_scale"], 0.3), size_mm * 0.4)
    z_height_mm = o["base_mm"] + relief_mm
    xs_g = np.arange(gw, dtype=np.float32) * mm_per_px
    ys_g = np.arange(gh, dtype=np.float32) * mm_per_px
    xg, yg = np.meshgrid(xs_g, ys_g)
    verts_top = np.column_stack(((xg.ravel() - xs_g[-1] / 2.0).astype(np.float32),
                                 (yg.ravel() - ys_g[-1] / 2.0).astype(np.float32),
                                 (o["base_mm"] + zd.ravel() * relief_mm).astype(np.float32)))
    rr2, cc2 = np.meshgrid(np.arange(gh - 1), np.arange(gw - 1), indexing='ij')
    v0 = rr2 * gw + cc2
    faces_top = np.vstack([np.column_stack([v0.ravel(), (v0 + 1).ravel(), (v0 + gw).ravel()]),
                           np.column_stack([(v0 + 1).ravel(), (v0 + gw + 1).ravel(), (v0 + gw).ravel()])
                           ]).astype(np.int32)
    n_grid = n_crop[0:sh:step, 0:sw:step].copy().astype(np.float32)
    nlen = np.linalg.norm(n_grid, axis=-1, keepdims=True)
    vn_top = (n_grid / np.where(nlen < 1e-9, 1.0, nlen)).reshape(-1, 3).astype(np.float32)
    colors_top = D.albedo_rgb_full[y1:y2, x1:x2][0:sh:step, 0:sw:step].reshape(-1, 3).astype(np.uint8)
    if o["watertight"]:
        verts_bot = verts_top.copy()
        verts_bot[:, 2] = 0.0
        nv = len(verts_top)
        faces_bot = faces_top[:, [0, 2, 1]] + nv
        perim = (list(range(gw)) + [r * gw + (gw - 1) for r in range(1, gh - 1)]
                 + list(range(gh * gw - 1, (gh - 1) * gw - 1, -1)) + [r * gw for r in range(gh - 2, 0, -1)])
        side = []
        for i in range(len(perim)):
            a, b = perim[i], perim[(i + 1) % len(perim)]
            side.extend([[a, b + nv, b], [a, a + nv, b + nv]])
        verts = np.vstack([verts_top, verts_bot])
        colors = np.vstack([colors_top, np.full_like(colors_top, 80)])
        faces = np.vstack([faces_top, faces_bot, np.array(side, dtype=np.int32)])
        vnorm = np.vstack([vn_top, np.tile([0., 0., -1.], (nv, 1)).astype(np.float32)])
    else:
        verts, colors, faces, vnorm = verts_top, colors_top, faces_top, vn_top
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=vnorm, process=False)
    if o["colour"]:
        mesh.visual.vertex_colors = np.hstack([colors, np.full((len(colors), 1), 255, np.uint8)])
    _rs_io = _fast_io()
    if _rs_io is not None and hasattr(_rs_io, "write_stl") and out_path.lower().endswith(".stl"):
        # binary STL (it holds no colours or vertex normals): normals and records
        # written on every core, instead of trimesh's face-normal pass and copies
        _rs_io.write_stl(out_path, _f64(verts), _u32_faces(faces), header="Helio 3D print export")
    else:
        mesh.export(out_path)
    return {"path": out_path, "grid": (gw, gh), "vertices": len(verts), "faces": len(faces),
            "size_mm": (gw * mm_per_px, gh * mm_per_px, z_height_mm),
            "region": "Crop box" if crop else "Full frame (trimmed)"}


# ── multi-LOD archival package ───────────────────────────────────────────

LODS = (("lod0_full", 1, "Archival — every pixel, up to 25 million vertices"),
        ("lod1_medium", 4, "Study — about 2 million vertices, fast to load in most tools"),
        ("lod2_web", 8, "Web/preview — about 300,000 vertices, for Three.js / Sketchfab"))
# The most vertices each level may have.  The steps above were set for 20 MP
# captures; on a 100 MP one they made a 98-million-vertex "full" mesh (a 12 GB
# OBJ most tools cannot open) and a 1.5-million-vertex "web" one.  A level's
# step is now its own step or the smallest that keeps it within this.
LOD_MAX_VERTS = {"lod0_full": 25_000_000, "lod1_medium": 2_000_000, "lod2_web": 300_000}


def lod_step(name: str, step: int, h: int, w: int) -> int:
    """The grid step a level uses on an h x w surface."""
    cap = LOD_MAX_VERTS.get(name)
    if not cap:
        return step
    return max(int(step), int(math.ceil(math.sqrt(h * w / float(cap)))))


def lod_package(D, out_dir: str, obj_id: str, *, ss: dict, crop=None) -> dict:
    """Three mesh LODs (OBJ + GLB), normal, displacement and albedo maps and a README."""
    import datetime
    import trimesh
    obj_id = (obj_id or "").strip() or "helio_object"
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in obj_id)
    height, fx, fy, fz = ss["height"], ss["fx"], ss["fy"], ss["fz"]
    x1, y1, x2, y2 = region(D, ss["edge_trim"], crop)
    n_crop = D.normals_full[y1:y2, x1:x2].astype(np.float32)
    raw, _sigma = detail_depth(n_crop, 0.35, 5.0)
    h_d, w_d = raw.shape
    alb = D.albedo_rgb_full[y1:y2, x1:x2]
    os.makedirs(out_dir, exist_ok=True)

    def build(step):
        y_s, x_s = np.mgrid[0:h_d:step, 0:w_d:step]
        gH, gW = y_s.shape
        z = raw[0:h_d:step, 0:w_d:step]
        verts = np.column_stack((((x_s - w_d / 2) * fx).ravel(), ((y_s - h_d / 2) * fy).ravel(),
                                 ((z * fz * height) - height / 2).ravel())).astype(np.float32)
        cols = alb[0:h_d:step, 0:w_d:step].reshape(-1, 3).astype(np.uint8)
        return trimesh.Trimesh(vertices=verts, faces=_grid_faces(gH, gW, fx, fy, fz).astype(np.uint32),
                               vertex_colors=np.column_stack((cols, np.full(len(cols), 255, np.uint8))),
                               process=False)
    files, sizes, steps = [], {}, {}
    for name, step, _d in LODS:
        step = steps[name] = lod_step(name, step, h_d, w_d)
        m = build(step)
        for fmt in ("glb", "obj"):
            p = os.path.join(out_dir, f"{safe}_{name}.{fmt}")
            _export_trimesh_obj(m, p) if fmt == "obj" else m.export(p)
            files.append(p)
        sizes[name] = (len(m.vertices), len(m.faces))
    for name, arr, args in (
            (f"{safe}_normal_map.png", cv2.cvtColor(((n_crop * 0.5 + 0.5) * 255).clip(0, 255)
                                                    .astype(np.uint8), cv2.COLOR_RGB2BGR), None),
            (f"{safe}_displacement_16bit.png", (raw * 65535).astype(np.uint16), None),
            (f"{safe}_albedo.png", cv2.cvtColor(alb, cv2.COLOR_RGB2BGR), None),
            (f"{safe}_displacement_32f.tif", raw, [cv2.IMWRITE_TIFF_COMPRESSION, 1])):
        p = os.path.join(out_dir, name)
        cv2.imwrite(p, arr, args) if args else cv2.imwrite(p, arr)
        files.append(p)
    rp = os.path.join(out_dir, f"{safe}_LOD_README.txt")
    with open(rp, "w") as r:
        r.write(f"Helio — Multi-LOD Archival Package\n{'=' * 44}\n")
        r.write(f"Object ID : {obj_id}\n")
        r.write(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        r.write("Meshes (OBJ + GLB for each LOD):\n")
        for name, _step, desc in LODS:
            v, f = sizes[name]
            step = steps[name]
            r.write(f"  {safe}_{name}.obj/.glb\n")
            r.write(f"    {desc}\n    {v:,} vertices, {f:,} faces  (step={step})\n\n")
        r.write("Texture maps:\n")
        r.write(f"  {safe}_normal_map.png         — PS normal map (8-bit RGB)\n")
        r.write(f"  {safe}_displacement_16bit.png — micro-relief (16-bit PNG)\n")
        r.write(f"  {safe}_displacement_32f.tif   — micro-relief (32-bit float TIFF)\n")
        r.write(f"  {safe}_albedo.png             — albedo (RGB)\n\n")
        r.write("Recommended use:\n")
        r.write("  Research / publication : lod0_full + normal_map + displacement_32f.tif\n")
        r.write("  Blender / Meshlab      : lod1_medium OBJ + displacement_16bit.png\n")
        r.write("  Web embed / Sketchfab  : lod2_web GLB + albedo + normal_map\n\n")
        r.write("Depth / displacement notes:\n")
        r.write("  HP-filtered (FC normal integration, sigma=35% of diagonal).\n")
        r.write("  Removes integration bowl. Only genuine micro-relief from PS normals survives.\n")
    files.append(rp)
    return {"files": files, "sizes": sizes, "safe_id": safe}


# ── displacement package ─────────────────────────────────────────────────

def displacement_package(D, out_dir: str, obj_id: str, *, ss: dict, depth_normalised=None,
                         crop=None, step: int = 4, progress: Callable | None = None) -> dict:
    """A coarse base mesh (OBJ), a 16-bit Frankot–Chellappa displacement map,
    the normal map and albedo, and a Blender guide."""
    upd = progress or (lambda *_a: None)
    obj_id = (obj_id or "").strip() or "helio"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in obj_id)
    os.makedirs(out_dir, exist_ok=True)
    upd("Resolving crop region…", 0.02)
    x1, y1, x2, y2 = region(D, ss["edge_trim"], crop)
    n_crop = D.normals_full[y1:y2, x1:x2].astype(np.float32)
    alb_crop = D.albedo_rgb_full[y1:y2, x1:x2]
    sh, sw = n_crop.shape[:2]
    upd("Frankot-Chellappa integration (this gives the detail)…", 0.08)
    disp, sigma_macro = detail_depth(n_crop, 0.18, 8.0)
    disp16 = (np.clip(disp, 0, 1) * 65535).astype(np.uint16)
    upd("Building flat base mesh…", 0.45)
    # The base is meant to be coarse: Blender subdivides it 7 levels and the
    # displacement map carries the detail.  At most 256 segments on its long
    # side (every 4th pixel of a 100 MP capture was 6 million vertices, an
    # 850 MB OBJ and 25 minutes, and could never be subdivided).
    step = max(2, int(step), int(math.ceil(max(sh, sw) / 256.0)))
    src = depth_normalised if depth_normalised is not None else D.depth_vis_p
    d_full = np.asarray(src)[y1:y2, x1:x2].astype(np.float32)
    sigma = max(8.0, 0.15 * math.sqrt(sh ** 2 + sw ** 2))
    k = min(1.0, 1024.0 / max(sh, sw))          # the broad shape: blurred on a small copy
    small = cv2.resize(d_full, (max(2, int(sw * k)), max(2, int(sh * k))),
                       interpolation=cv2.INTER_AREA) if k < 1.0 else d_full
    d_small = cv2.GaussianBlur(small, (0, 0), max(1.0, sigma * k))
    height, fx, fy = ss["height"], ss["fx"], ss["fy"]
    ys, xs = np.mgrid[0:sh:step, 0:sw:step]
    gH, gW = ys.shape
    z = cv2.remap(d_small, (xs * k).astype(np.float32), (ys * k).astype(np.float32),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    verts = np.column_stack((((xs - sw / 2) * fx).ravel(), ((ys - sh / 2) * fy).ravel(),
                             (z * height - height / 2).ravel())).astype(np.float32)
    faces = _grid_faces(gH, gW, fx, fy, 1)
    u_uv = (xs / sw).ravel().astype(np.float32)
    v_uv = (1.0 - ys / sh).ravel().astype(np.float32)
    upd("Writing OBJ base mesh…", 0.60)
    obj_path = os.path.join(out_dir, f"{safe}_flat_base.obj")
    mtl = f"{safe}_flat_base.mtl"
    _head = ("# Helio — flat base mesh for displacement\n"
             "# Load in Blender + Subdivision (level 7) + Displace modifier\n"
             f"mtllib {mtl}\n")
    _rs_io = _fast_io()
    try:
        if _rs_io is None:
            raise RuntimeError("no fast writer")
        _rs_io.write_obj(obj_path, _f64(verts), _u32_faces(faces), header=_head, decimals=4,
                         uvs=_f64(np.column_stack([np.asarray(u_uv, np.float64), np.asarray(v_uv, np.float64)])),
                         uv_decimals=6, mid="usemtl albedo\ng surface\n", face_style="v/vt")
    except Exception:
        with open(obj_path, "w") as f:
            f.write(_head)
            f.writelines(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n" for v in verts)
            f.writelines(f"vt {u:.6f} {vv:.6f}\n" for u, vv in zip(u_uv, v_uv))
            f.write("usemtl albedo\ng surface\n")
            f.writelines(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n" for a, b, c in faces)
    with open(os.path.join(out_dir, mtl), "w") as f:
        f.write(f"newmtl albedo\nmap_Kd {safe}_albedo.png\nmap_bump {safe}_normal_map.png\n")
    upd("Saving displacement TIFF and textures…", 0.75)
    disp_path = os.path.join(out_dir, f"{safe}_displacement_16bit.tif")
    cv2.imwrite(disp_path, disp16, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    norm_path = os.path.join(out_dir, f"{safe}_normal_map.png")
    cv2.imwrite(norm_path, cv2.cvtColor(np.clip((n_crop + 1.0) / 2.0 * 255, 0, 255).astype(np.uint8),
                                        cv2.COLOR_RGB2BGR))
    alb_path = os.path.join(out_dir, f"{safe}_albedo.png")
    cv2.imwrite(alb_path, cv2.cvtColor(alb_crop.astype(np.uint8), cv2.COLOR_RGB2BGR))
    upd("Writing Blender guide…", 0.92)
    readme = (
        f"Helio — High-Detail Displacement Package\n{'=' * 50}\nObject ID: {obj_id}\n\n"
        f"Files in this package:\n"
        f"  {safe}_flat_base.obj       — coarse base mesh\n"
        f"  {safe}_displacement_16bit.tif — 16-bit displacement map\n"
        f"  {safe}_normal_map.png       — RGB normal map\n"
        f"  {safe}_albedo.png           — colour texture\n\n"
        f"WHY THIS GIVES BETTER DETAIL THAN THE STANDARD MESH\n{'─' * 50}\n"
        f"The depth integration used to build standard OBJ/STL meshes\n"
        f"acts as a low-pass filter — fine carved detail (grooves <3px\n"
        f"wide) doesn't survive integration. This package exports the\n"
        f"PS normal map AS a displacement map, bypassing that loss.\n\n"
        f"BLENDER WORKFLOW (tested Blender 4.x)\n{'─' * 50}\n"
        f"1. File > Import > Wavefront OBJ → select flat_base.obj\n"
        f"2. Select the mesh. Add Properties > Modifier > Subdivision Surface\n"
        f"   Set Levels Viewport = 7, Render = 7\n"
        f"3. Add Properties > Modifier > Displace\n"
        f"   - Click 'New' to create a texture\n"
        f"   - In Texture Properties: Type = Image, Open displacement_16bit.tif\n"
        f"   - Back in Displace modifier: Strength = 0.05 to 0.20\n"
        f"     (adjust until carved lines are clearly visible)\n"
        f"   - Direction = Z\n"
        f"4. Add Material:\n"
        f"   - Base Color: connect albedo.png\n"
        f"   - Normal: Add Normal Map node, connect normal_map.png\n"
        f"   - Set Normal Map Strength = 1.5–3.0\n"
        f"5. Render. The Displace modifier gives actual geometry from the\n"
        f"   PS normals; the Normal Map adds sub-grid micro-detail.\n\n"
        f"MESHMIXER / ZBRUSH\n{'─' * 50}\n"
        f"ZBrush: Lightbox > Import > OBJ → DynaMesh → Apply Displacement\n"
        f"  (Surface > Applied Displacement, select 16-bit TIFF)\n"
        f"MeshMixer: Use 'Deform > Displace' with the 16-bit TIFF.\n\n"
        f"TECHNICAL NOTES\n{'─' * 50}\n"
        f"Displacement method: Frankot-Chellappa integration of PS normals\n"
        f"HP filter sigma: {sigma_macro:.1f}px (18% of image diagonal)\n"
        f"Displacement range: full [0,1] 16-bit scale\n"
        f"Base mesh step: {step}px between vertices\n")
    rp = os.path.join(out_dir, "README.txt")
    with open(rp, "w") as f:
        f.write(readme)
    return {"files": [obj_path, os.path.join(out_dir, mtl), disp_path, norm_path, alb_path, rp],
            "safe_id": safe}


# ── the HF + normals web viewer ──────────────────────────────────────────

def hf_normals_viewer(normals, albedo, settings: dict, out_dir: str,
                      progress: Callable | None = None) -> dict:
    """The 3D viewer built from the normals' high frequencies, with the same
    four-million-vertex safety net as the GLB web viewer."""
    from helio.io.exporters import export_3d_web_viewer_hf_normals
    settings = dict(settings)
    cx1, cy1, cx2, cy2 = settings.get("crop", (0, 0, normals.shape[1], normals.shape[0]))
    est = ((cx2 - cx1) // settings['step']) * ((cy2 - cy1) // settings['step'])
    if est > 4_000_000:
        settings['step'] = max(settings['step'], int(math.ceil(settings['step']
                                                             * math.sqrt(est / 4_000_000))))
    return export_3d_web_viewer_hf_normals(normals, albedo, settings, out_dir, progress_cb=progress)


# ── facsimile (physical reproduction) ────────────────────────────────────

FACSIMILE_TYPES = (("flat", "Flat original (manuscript, print, map)"),
                   ("plate", "Printing plate: relief inverted and mirrored, an impression tool"),
                   ("seal", "Seal / bulla: relief inverted and mirrored, for a rolled impression"),
                   ("painting", "Painting / textured surface"))
FACSIMILE_RES = (("draft", 256, "Draft: ~256 px grid, fast preview"),
                 ("balanced", 512, "Balanced: ~512 px grid, recommended"),
                 ("fine", 1024, "Fine: ~1024 px grid, high detail"),
                 ("maximum", 0, "Maximum: every pixel, archival / CNC"))
FACSIMILE_FORMATS = (("STL", "STL: universal 3D printing (binary)"),
                     ("OBJ", "OBJ + 16-bit displacement TIFF: elevated printing / CNC"),
                     ("TIFF", "16-bit displacement TIFF only: raster elevated printing"),
                     ("GLB", "Coloured GLB: albedo on the relief, physically scaled"))
FACSIMILE_DEFAULTS = {"otype": "flat", "invert": False, "mag": 1.0, "zmul": 1.0, "warp": 0.80,
                      "taper": 0.04, "res": "balanced", "custom": 768, "fmt": "OBJ"}
ASSUMED_MM_PER_PX = 0.1


def facsimile_grid(shape, res: str = "balanced", custom=768) -> tuple[int, int, int]:
    """(step, rows, cols) of the facsimile grid for a depth of *shape*."""
    H, W = shape[:2]
    try:
        target = dict((k, t) for k, t, _ in FACSIMILE_RES).get(res, int(custom or 512))
    except (TypeError, ValueError):
        target = 512
    target = max(64, min(int(target or max(H, W)), max(H, W)))
    step = max(1, max(H, W) // target)
    return step, len(range(0, H, step)), len(range(0, W, step))


def facsimile_relief(depth, *, otype: str = "flat", invert: bool = False, warp: float = 0.8,
                     taper: float = 0.04, zmul: float = 1.0) -> np.ndarray:
    """The facsimile's relief, lowest point 0, in the depth's units × *zmul*:
    inverted for a plate or seal (and/or by hand), the low-frequency bowl
    (a quadric fitted to a σ = 15 % blur) removed by *warp*, exaggerated,
    then a cosine taper over *taper* of each border (HELIO 2's steps)."""
    d = np.asarray(depth, np.float32).copy()
    if otype in ("plate", "seal"):
        d = d.max() - d
    if invert:
        d = d.max() - d
    if warp > 0.005:
        # The bowl is broad (a 15 % blur, then a quadric), so it is found on
        # a copy at most 1024 px across and evaluated at full size.  On a
        # 100 MP capture the full-size blur and a 100-million-row least
        # squares took 19 minutes and 5 GB.
        H, W = d.shape
        k = min(1.0, 1024.0 / max(H, W))
        small = (cv2.resize(d, (max(2, int(round(W * k))), max(2, int(round(H * k)))),
                            interpolation=cv2.INTER_AREA) if k < 1.0 else d)
        sh, sw = small.shape
        sig = max(1.0, max(H, W) * 0.15 * k)
        sm = cv2.GaussianBlur(small, (0, 0), sigmaX=sig, sigmaY=sig)
        yy, xx = np.mgrid[0:sh, 0:sw]
        xn = (xx / max(sw - 1, 1)) * 2 - 1
        yn = (yy / max(sh - 1, 1)) * 2 - 1
        A = np.column_stack([xn.ravel() ** 2, yn.ravel() ** 2, (xn * yn).ravel(), xn.ravel(),
                             yn.ravel(), np.ones(sh * sw)])
        c, *_ = np.linalg.lstsq(A, sm.ravel(), rcond=None)
        xf = (np.arange(W, dtype=np.float32) / max(W - 1, 1) * 2 - 1)[None, :]
        yf = (np.arange(H, dtype=np.float32) / max(H - 1, 1) * 2 - 1)[:, None]
        bowl = (c[0] * xf ** 2 + c[1] * yf ** 2 + c[2] * xf * yf + c[3] * xf + c[4] * yf
                + c[5]).astype(np.float32)
        d = d - bowl * warp
        d -= d.min()
    d = d * zmul
    if taper > 0.001:
        H, W = d.shape
        bh = min(max(1, int(H * taper)), H // 2 - 1)
        bw = min(max(1, int(W * taper)), W // 2 - 1)
        ry = np.ones(H, np.float32)
        rx = np.ones(W, np.float32)
        ry[:bh] = 0.5 - 0.5 * np.cos(np.pi * np.arange(bh) / bh)
        ry[H - bh:] = ry[:bh][::-1]
        rx[:bw] = 0.5 - 0.5 * np.cos(np.pi * np.arange(bw) / bw)
        rx[W - bw:] = rx[:bw][::-1]
        d = d * np.outer(ry, rx)
    return d - d.min()


def facsimile_mesh(relief, *, mm_per_px=None, z_mm_per_unit=None, mag: float = 1.0,
                   target: int = 512, mirror: bool = False):
    """Grid mesh of the relief in millimetres.

    X and Y are pixel × mm/px × magnification; HELIO 2 multiplied by the
    grid step a second time, so any mesh coarser than every pixel came out
    step × too wide and long, and it never had the measured scale, using
    0.1 mm/px always (KD-044).  Z is the relief × mm per pixel of height (or
    the relief itself, uncalibrated) × magnification; HELIO 2 re-stretched
    the relief to 0–1 after flattening, which overstated a calibrated
    height by the flattening (KD-050).  The image's top is +Y, as the
    object lies; HELIO 2 put it at −Y, a mirror image, which only a plate or
    seal (a tool) should be (*mirror*).  Returns (verts, faces, step,
    (rows, cols), mm/px used, whether that was assumed)."""
    H, W = relief.shape
    assumed = not mm_per_px
    px = float(mm_per_px or ASSUMED_MM_PER_PX)
    target = max(64, min(int(target or max(H, W)), max(H, W)))
    step = max(1, max(H, W) // target)
    ys = np.arange(0, H, step)
    xs = np.arange(0, W, step)
    yg, xg = np.meshgrid(ys, xs, indexing="ij")
    rH, rW = yg.shape
    zk = float(z_mm_per_unit) if z_mm_per_unit else 1.0
    yv = yg if mirror else (H - 1 - yg)
    verts = np.stack([xg.ravel() * px * mag, yv.ravel() * px * mag,
                      relief[yg, xg].ravel() * zk * mag], axis=1).astype(np.float64)
    faces = _grid_faces(rH, rW, 1, 1 if mirror else -1, 1)
    return verts, faces, step, (rH, rW), (yg, xg), px, assumed


def _write_stl(path: str, verts, faces, header: str):
    rs = _fast_io()
    if rs is not None and hasattr(rs, "write_stl"):
        try:
            rs.write_stl(path, _f64(verts), _u32_faces(faces), header=header)
            return
        except Exception:
            pass                                   # the NumPy writer below
    tri = verts[faces].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), [0, 0, 1]).astype(np.float32)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(header.encode()[:80].ljust(80, b" "))
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(rec.tobytes())


def _write_obj(path: str, verts, faces, header: str):
    rs = _fast_io()
    if rs is not None:
        try:
            rs.write_obj(path, _f64(verts), _u32_faces(faces), header=header, decimals=4,
                         blank_after_vertices=True, face_style="v")
            return
        except Exception:
            pass                                   # the NumPy writer below
    with open(path, "w") as fh:
        fh.write(header)
        np.savetxt(fh, verts, fmt="v %.4f %.4f %.4f")
        fh.write("\n")
        np.savetxt(fh, faces + 1, fmt="f %d %d %d")


def facsimile_preview(depth, options: dict, *, mm_per_px=None, z_mm_per_unit=None,
                      out_dir: str | None = None) -> str:
    """A draft (256 px) OBJ of the facsimile with every other setting as
    chosen, for Quick Look."""
    import tempfile
    o = dict(FACSIMILE_DEFAULTS, **(options or {}))
    rel = facsimile_relief(depth, otype=o["otype"], invert=o["invert"], warp=o["warp"],
                           taper=o["taper"], zmul=o["zmul"])
    v, f, *_ = facsimile_mesh(rel, mm_per_px=mm_per_px, z_mm_per_unit=z_mm_per_unit, mag=o["mag"],
                              target=256, mirror=o["otype"] in ("plate", "seal"))
    p = os.path.join(out_dir or tempfile.mkdtemp(prefix="helio_preview_"), "helio_preview.obj")
    _write_obj(p, v, f, f"# Helio Quick Look Preview\n# Z={o['zmul']}x warp={o['warp'] * 100:.0f}% "
                        f"taper={o['taper'] * 100:.1f}%{' [INVERTED]' if o['invert'] else ''}\n\n")
    return p


def facsimile_package(depth, out_dir: str, safe: str, options: dict | None = None, *,
                      mm_per_px=None, z_mm_per_unit=None, albedo=None, obj_id: str = "",
                      depth_engine: str = "", normal_engine: str = "", stamp: str = "") -> dict:
    """The classic Facsimile Export: STL, OBJ + displacement, displacement
    TIFF or coloured GLB, and a README of every setting."""
    import datetime
    o = dict(FACSIMILE_DEFAULTS, **(options or {}))
    ts = stamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    otype, fmt = o["otype"], o["fmt"]
    tool = otype in ("plate", "seal")
    rel = facsimile_relief(depth, otype=otype, invert=o["invert"], warp=o["warp"], taper=o["taper"],
                           zmul=o["zmul"])
    rng = float(rel.max())
    step0, *_ = facsimile_grid(rel.shape, o["res"], o.get("custom"))
    verts, faces, step, (rH, rW), (yg, xg), px, assumed = facsimile_mesh(
        rel, mm_per_px=mm_per_px, z_mm_per_unit=z_mm_per_unit, mag=o["mag"],
        target=max(rel.shape) // step0, mirror=tool)
    H, W = rel.shape
    files = []
    base = os.path.join(out_dir, f"{safe}_{ts}_facsimile")
    d16 = (rel / rng * 65535).astype(np.uint16) if rng > 1e-12 else np.zeros_like(rel, np.uint16)
    if tool:
        d16 = d16[::-1]          # the displacement map mirrored like the tool
    if fmt in ("TIFF", "OBJ"):
        p = base + "_displacement.tif"
        try:
            import tifffile
            tifffile.imwrite(p, d16)
        except ImportError:
            cv2.imwrite(p, d16)
        files.append(p)
    head = (f"Helio | warp={o['warp'] * 100:.0f}% taper={o['taper'] * 100:.1f}% "
            f"res={o['res']}")
    if fmt == "STL":
        files.insert(0, base + ".stl")
        _write_stl(files[0], verts, faces, head)
    elif fmt == "OBJ":
        files.insert(0, base + ".obj")
        _write_obj(files[0], verts, faces,
                   f"# Helio Facsimile — {obj_id}\n# warp={o['warp'] * 100:.0f}% "
                   f"taper={o['taper'] * 100:.1f}% res={o['res']} step={step}\n"
                   f"# vertices={len(verts):,} faces={len(faces):,}  units: mm\n\n")
    elif fmt == "GLB":
        import trimesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        if albedo is not None:
            alb = np.asarray(albedo)
            uy = np.clip((yg.ravel() / max(H - 1, 1) * (alb.shape[0] - 1)).astype(np.int32), 0, alb.shape[0] - 1)
            ux = np.clip((xg.ravel() / max(W - 1, 1) * (alb.shape[1] - 1)).astype(np.int32), 0, alb.shape[1] - 1)
            vc = alb[uy, ux][:, :3].astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=np.hstack([vc, np.full((len(vc), 1), 255, np.uint8)]))
        files.insert(0, base + ".glb")
        mesh.export(files[0])
        from helio.io.exporters import _fix_glb_material
        _fix_glb_material(files[0])
    zk = float(z_mm_per_unit) if z_mm_per_unit else None
    size = (W * px * o["mag"], H * px * o["mag"], rng * (zk or 1.0) * o["mag"])
    cal = (f"Z-calibrated: {zk * 1000:.1f} µm per pixel of height (true heights from the normals)" if zk else
           "Z not calibrated: relative units (shape right, height not to scale)")
    xy = (f"{px:.5f} mm/px, measured" if not assumed else
          f"NOT MEASURED: {px} mm/px assumed. Set the scale for a true-size facsimile.")
    readme = base + "_README.txt"
    with open(readme, "w", encoding="utf-8") as fh:
        fh.write("Helio - Facsimile Export\n" + "=" * 40 + "\n"
                 f"Object ID:       {obj_id}\n"
                 f"Object type:     {otype}\n"
                 f"Format:          {fmt}\n"
                 f"Magnification:   {o['mag']}x\n"
                 f"Z exaggeration:  {o['zmul']}x\n"
                 f"Warp flatten:    {o['warp'] * 100:.0f}%\n"
                 f"Edge taper:      {o['taper'] * 100:.1f}%\n"
                 f"Resolution:      {o['res']} (step={step}, grid {rW}x{rH})\n"
                 f"Mesh vertices:   {len(verts):,}\n"
                 f"Mesh faces:      {len(faces):,}\n"
                 f"Size:            {size[0]:.1f} x {size[1]:.1f} x {size[2]:.3f} "
                 f"{'mm' if zk else 'mm (height in relative units)'}\n"
                 f"XY scale:        {xy}\n"
                 f"Z calibration:   {cal}\n"
                 f"Depth inverted:  {'YES - impression tool' if tool else 'No'}"
                 f"{' (and inverted by hand)' if o['invert'] else ''}\n"
                 f"Orientation:     {'MIRRORED, as a tool must be to print the right way round' if tool else 'as the object, seen from above (+Y = top of the image)'}\n"
                 f"Depth solver:    {depth_engine}\n"
                 f"Normal engine:   {normal_engine}\n\n"
                 "Warp flatten removes Poisson integrator low-frequency drift\n"
                 "before Z exaggeration so only genuine relief is amplified.\n"
                 "Edge taper applies cosine window at boundary for flat print bed.\n")
    files.append(readme)
    return {"files": files, "grid": (rW, rH), "vertices": len(verts), "faces": len(faces),
            "step": step, "size_mm": size, "assumed_scale": assumed, "mirrored": tool}
