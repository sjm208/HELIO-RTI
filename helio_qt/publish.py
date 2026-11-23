"""Run the chosen exports as one job, item by item (UX-005 on EXP-001).

Every format calls a helio.application.exports function, the same one the
classic app's menu calls, with the classic app's default settings.  One
item failing is reported and the rest carry on; cancelling stops between
items.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from helio.application import exports as X
from helio_qt.publish_catalog import all_formats

_log = logging.getLogger(__name__)


@dataclass
class Source:
    """What the exports read: one solve, its lights and its name."""
    result: object                         # helio.application.pipeline.SolveResult
    lights: np.ndarray | None
    light_source: str                      # measured, estimated, or ""
    images: list[str]
    name: str
    mm_per_px: float = 0.0
    z_mm_per_unit: float = 0.0
    shared: dict = field(default_factory=X.shared_defaults)
    data: object = None                    # the DataManager Inspect renders from
    view: str = "Standard PBR"
    view_settings: dict = field(default_factory=dict)   # the view's settings, light included
    meta: dict = field(default_factory=dict)        # Capture details
    poi: dict = field(default_factory=dict)         # points of interest {name: (x, y, z) mm}
    poi_source: str = ""
    display: object = None                 # helio.application.viewing.Display; None: colour
    engine_settings: dict = field(default_factory=dict)   # every view's settings, light included
    tone: str = "srgb"                     # how the frames were decoded for the solve
    compare_view: str = ""                 # Inspect's Compare view, for the comparison page
    profile: tuple | None = None           # Measure's profile, as the classic current_profile_data
    facts: dict = field(default_factory=dict)      # the plate's extras: colour check, drift, notes, relief
    white_balance: tuple | None = None     # (R, G, B) linear gains, helio.core.white_balance; None: off
    turn: float = 0.0                      # Inspect's turn, degrees clockwise (display only)

    def screen(self):
        from helio.application.viewing import Display
        return self.display if self.display is not None else Display(layer="Combined Colour")


def file_stem(name: str) -> str:
    import re
    keep = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.strip())
    return re.sub(r"_{2,}", "_", keep).strip("_") or "helio"


def embed_meta(path: str, meta: dict, what: str = "") -> bool:
    """The capture details written into a saved picture, where Finder,
    Bridge, Lightroom and exiftool read them: a TIFF's ImageDescription,
    Artist and Copyright, a PNG's text, a JPEG's EXIF.  *what* says what the
    picture is (the view).  Blank details leave the file as it is."""
    from helio.domain.capture_meta import FIELDS
    meta = {k: str(v).strip() for k, v in dict(meta or {}).items() if str(v).strip()}
    if not meta:
        return False
    from helio.domain import object_profiles as OP
    labels = {**OP.labels(), **{k: lab for _g, fs in FIELDS for lab, k, _h, _t in fs},
              "object_type": "Object type"}
    if "object_type" in meta:
        meta["object_type"] = OP.get(meta["object_type"]).name
    lines = ([what] if what else []) + [f"{labels.get(k, k)}: {v}" for k, v in meta.items()
                                        if k in labels]
    desc = "\n".join(lines + ["Made with Helio"])
    artist = meta.get("operator", "")
    rights = " · ".join(x for x in (meta.get("institution", ""), meta.get("collection", "")) if x)
    try:
        from PIL import Image
        ext = os.path.splitext(path)[1].lower()
        with Image.open(path) as im:
            im.load()
            if ext in (".tif", ".tiff"):
                info = {270: desc, 305: "Helio"}
                if artist:
                    info[315] = artist
                if rights:
                    info[33432] = rights
                im.save(path, tiffinfo=info, compression=None)
            elif ext == ".png":
                from PIL.PngImagePlugin import PngInfo
                pi = PngInfo()
                pi.add_text("Title", meta.get("title", "") or what)
                pi.add_text("Description", desc)
                if artist:
                    pi.add_text("Author", artist)
                if rights:
                    pi.add_text("Copyright", rights)
                pi.add_text("Software", "Helio")
                im.save(path, pnginfo=pi)
            elif ext in (".jpg", ".jpeg"):
                ex = im.getexif()
                ex[270], ex[305] = desc, "Helio"
                if artist:
                    ex[315] = artist
                if rights:
                    ex[33432] = rights
                im.save(path, exif=ex, quality=95)
            else:
                return False
        return True
    except Exception:
        _log.exception("could not write the capture details into %s", path)
        return False


def export_one(key: str, src: Source, out_dir: str) -> list[str]:
    """Write one format into *out_dir*; returns the files written."""
    r, stem = src.result, file_stem(src.name)
    if key == "glb":
        folder = os.path.join(out_dir, f"{stem}_web_viewer")
        os.makedirs(folder, exist_ok=True)
        settings = X.build_export_settings(src.shared, X.MeshOptions(), r.w, r.h, obj_id=src.name)
        out = X.web_viewer(r.depth_vis_p, r.depth_normalised_full, r.normals_full,
                           r.albedo_rgb_full, settings, folder)
        return sorted(os.path.join(folder, f) for f in os.listdir(folder)) or list(out.values())
    if key == "stl":
        path = os.path.join(out_dir, f"{stem}_3dprint.stl")
        return [X.stl_print(r.depth_vis_p, r.depth_normalised_full, r.normals_full, r.w, r.h,
                            src.shared, X.StlOptions(true_scale=bool(src.mm_per_px)), path,
                            mm_per_px=src.mm_per_px)]
    if key == "png_views":
        return [current_view(src, out_dir)]
    if key == "e57":
        return [X.point_cloud_e57(r.depth_vis_p, r.depth_normalised_full, r.albedo_rgb_full,
                                  r.w, r.h, src.shared, os.path.join(out_dir, f"{stem}.e57"))]
    if key == "lido":
        return [X.lido_record(os.path.join(out_dir, f"{stem}_lido.xml"), src.name,
                              r.settings.normal_engine, r.settings.depth_engine, r.n_images)]
    if key == "iso_stats":
        from helio.application.heights import height_px
        return [X.iso_stats_csv(height_px(r), os.path.join(out_dir, f"{stem}_iso25178.csv"),
                                stem, z_mm_per_unit=src.z_mm_per_unit)]
    if key == "plate":
        return [plate(src, out_dir)]
    if key == "calibration":
        estimated = src.lights is None and r.light_matrix is not None
        lights = r.light_matrix if estimated else src.lights
        return [X.calibration_report(
            os.path.join(out_dir, f"{stem}_calibration.txt"), obj=stem, w=r.w, h=r.h,
            n_images=r.n_images, mm_per_px=src.mm_per_px, z_mm_per_unit=src.z_mm_per_unit,
            lights=lights,
            light_source="estimated" if estimated else ("measured" if lights is not None else ""),
            normal_engine=r.settings.normal_engine, depth_engine=r.settings.depth_engine,
            lights_from_matrix=r.light_matrix is not None)]
    if key == "heritage":
        return [X.heritage_archive(os.path.join(out_dir, f"{stem}_heritage.zip"),
                                   r.albedo_rgb_full, r.normals_full, r.depth_vis_p, r.w, r.h,
                                   src.shared, object_id=src.name,
                                   normal_engine=r.settings.normal_engine,
                                   depth_engine=r.settings.depth_engine)]
    if key == "provenance":
        return list(X.write_processing_record(os.path.join(out_dir, f"{stem}_provenance.json"),
                                              processing_record(src)))
    if key == "gis_maps":
        folder = os.path.join(out_dir, f"{stem}_gis_maps")
        os.makedirs(folder, exist_ok=True)
        return X.gis_analysis_maps(folder, r.terrain, r.w, r.h, obj_id=src.name,
                                   mm_per_px=src.mm_per_px)
    if key == "qgis":
        from helio.io.exporters import write_qgis_plugin_zip
        from helio.io.exporters import QGIS_PLUGIN_ZIP
        path = os.path.join(out_dir, os.path.basename(QGIS_PLUGIN_ZIP))
        write_qgis_plugin_zip(path, src.name or "HELIO")
        return [path]
    if key == "anaglyph":
        t = _screen_turn(src)                   # as the screen shows the object
        dep = np.asarray(r.depth_vis_p)
        return [X.anaglyph_image(t.image(dep, float(np.nanmin(dep))), t.image(np.asarray(r.albedo_rgb_full)),
                                 os.path.join(out_dir, f"{stem}_anaglyph.png"))]
    if key == "sweep_gif":
        return [sweep_gif(src, out_dir)]
    if key == "lp":
        return [X.light_file(src.lights, src.light_source, os.path.join(out_dir, f"{stem}.lp"),
                             src.images)]
    if key == "rti_viewer":
        _fitted(r)
        return [X.selfcontained_rti(r.ptm_coeffs, r.albedo_rgb_full, r.normals_full, r.w, r.h,
                                    src.shared, os.path.join(out_dir, f"{stem}_rti_viewer.html"),
                                    obj_id=src.name)]
    if key == "ptm":
        folder = os.path.join(out_dir, f"{stem}_ptm")
        os.makedirs(folder, exist_ok=True)
        _fitted(r)
        return X.ptm_coefficients(r.ptm_coeffs, r.w, r.h, folder)
    if key in IMAGE_EXPORTS:
        return IMAGE_EXPORTS[key](src, out_dir)
    if key in MESH_EXPORTS:
        return MESH_EXPORTS[key](src, out_dir)
    if key in WEB_EXPORTS:
        return WEB_EXPORTS[key](src, out_dir)
    if key in ARCHIVE_EXPORTS:
        return ARCHIVE_EXPORTS[key](src, out_dir)
    if key in REPORT_EXPORTS:
        return REPORT_EXPORTS[key](src, out_dir)
    raise X.NothingToExport("This one is not available in this version.")


# ── the image exports (helio.application.export_images) ──────────────────

def _layer(src) -> str:
    return "Shading" if src.screen().layer == "Shading" else "Combined Colour"


def _drawer(src, D):
    """draw(engine) at *D*'s full size, each view with its own settings."""
    from helio.application.render import composite, render
    from helio.domain.registry import default_engine_settings
    layer = _layer(src)
    defaults = default_engine_settings()

    def draw(eng):
        st = dict(src.engine_settings.get(eng) or defaults.get(eng) or default_engine_settings((eng,))[eng])
        img = render(eng, D, st, D.h, D.w, False)
        return composite(eng, img, D, st, D.h, D.w, False, view_mode=layer)
    return draw


def _need_data(src):
    if src.data is None:
        raise X.NothingToExport("Process the images first")
    return src.data


def _folder(out_dir, name):
    f = os.path.join(out_dir, name)
    os.makedirs(f, exist_ok=True)
    return f


def _all_views(src, out_dir):
    from helio.application.export_images import all_views
    from helio.domain.registry import RENDER_ENGINES
    D = _need_data(src)
    t = _screen_turn(src)                       # as the screen shows the object
    draw = _drawer(src, D)
    dep = np.asarray(D.depth_vis_p)
    files, _n = all_views(lambda eng: t.image(np.asarray(draw(eng))), RENDER_ENGINES,
                          _folder(out_dir, f"{file_stem(src.name)}_all_views"),
                          base=(t.image(np.asarray(D.albedo_rgb_full)),
                                t.image(t.vectors(np.asarray(D.normals_full)), (0.0, 0.0, 1.0)),
                                t.image(dep, float(np.nanmin(dep)))))
    return files


def _normal_map(src, out_dir):
    from helio.application.export_images import combined_renders
    return combined_renders(_need_data(src), _folder(out_dir, f"{file_stem(src.name)}_normal_maps"),
                            src.name)


def _deepzoom(src, out_dir):
    from helio.application.export_images import deepzoom_data, deepzoom_layers, write_deepzoom
    Z = deepzoom_data(_need_data(src))
    layers = deepzoom_layers(Z, _drawer(src, Z))
    t = _screen_turn(src)
    if t.any:                                   # as the screen shows the object
        moved = []
        for name, img, title in layers:
            if name == "normals":               # directions turned with the picture
                n = t.image(t.vectors(np.asarray(Z.normals_full)), (0.0, 0.0, 1.0))
                img = (((n + 1.0) / 2.0) * 255).astype(np.uint8)
            else:
                img = t.image(np.asarray(img))
            moved.append((name, img, title))
        layers = moved
    return write_deepzoom(layers, _folder(out_dir, f"{file_stem(src.name)}_deepzoom"))


def _scalebar(src, out_dir):
    from helio.application.export_images import scale_bar_render
    from helio.application.render import composite, render
    D = _need_data(src)
    view = "Standard PBR" if src.view == "Depth map" else src.view
    st = dict(src.view_settings) if view == src.view else dict(src.engine_settings.get(view, {}))
    rgb = composite(view, render(view, D, st, D.h, D.w, False), D, st, D.h, D.w, False,
                    view_mode=_layer(src))
    rgb = _screen_turn(src).image(np.asarray(rgb))       # as the screen shows the object
    return [scale_bar_render(rgb, src.mm_per_px, os.path.join(
        out_dir, f"{file_stem(src.name)}_{file_stem(view)[:24]}_scalebar.png"))]


def _heatmap(src, out_dir):
    from helio.application.export_images import elevation_heatmap
    dep = np.asarray(src.result.depth_vis_p)
    return [elevation_heatmap(_screen_turn(src).image(dep, float(np.nanmin(dep))),
                              os.path.join(out_dir, f"{file_stem(src.name)}_elevation_heatmap.jpg"))]


def _stack(src):
    from helio.application.export_images import load_stack
    r = src.result
    if r.light_matrix is None or len(src.images) != len(r.light_matrix):
        return None, None
    return r.light_matrix, load_stack(src.images, r.w, r.h, crop=r.crop, base=(r.base_w, r.base_h),
                                       tone=src.tone)


def _pbr(src, out_dir):
    from helio.application.export_images import pbr_suite
    L, st = _stack(src)
    return pbr_suite(_need_data(src), _folder(out_dir, f"{file_stem(src.name)}_pbr"),
                     step=max(1, int(src.shared.get("export_step", 1) or 1)), lights=L, stack=st)


def _sds(src, out_dir):
    from helio.application.export_images import specular_diffuse
    L, st = _stack(src)
    return specular_diffuse(_need_data(src), _folder(out_dir, f"{file_stem(src.name)}_specular_diffuse"),
                            lights=L, stack=st)


def _pca(src, out_dir):
    from helio.application.export_images import pca_tiff
    return [pca_tiff(src.result.pca_full, os.path.join(out_dir, f"{file_stem(src.name)}_pca.tif"))]


def _fitted(r):
    """PTM and HSH coefficients in *r*: fitted again from the photographs when a machine
    short of memory did not keep them (the first export that needs them waits for it)."""
    from helio.application.pipeline import ensure_ptm_hsh
    return ensure_ptm_hsh(r)


def _hsh(src, out_dir):
    from helio.application.export_images import hsh_coefficients
    r = src.result
    _fitted(r)
    return hsh_coefficients(r.hsh_coeffs, r.w, r.h, _folder(out_dir, f"{file_stem(src.name)}_hsh"))


def _uncertainty(src, out_dir):
    from helio.application.export_images import uncertainty_map
    D = _need_data(src)
    resid = getattr(D, "qc_resid_full", None)
    return [uncertainty_map(resid, os.path.join(out_dir, f"{file_stem(src.name)}_uncertainty.tif"))[0]]


# ── 3D models (helio.application.export_3d) ───────────────────────────────

def _mesh_runner(fmt: str, ext: str, suffix: str = "", **options):
    def run(src, out_dir):
        from helio.application.export_3d import mesh_export
        from helio.application.render import composite, render
        D, r = _need_data(src), src.result

        def draw(eng, h, w, cc, vm):
            st = dict(src.engine_settings.get(eng) or src.view_settings)
            return composite(eng, render(eng, D, st, h, w, False, cc), D, st, h, w, False, cc,
                             view_mode=vm)
        path = os.path.join(out_dir, f"{file_stem(src.name)}{suffix}{ext}")
        sh = src.shared
        model = dict(tex_size=str(sh.get("tex_size", X.EXPORT_DEFAULTS["tex_size"])),
                     sketchfab_jpeg=bool(sh.get("tex_jpeg", 0)),
                     watertight=bool(sh.get("mesh_watertight", 1)),
                     vertex_colors=bool(sh.get("mesh_vertex_colours", 1)))
        res = mesh_export(D, path, fmt, ss=src.shared,
                          options=dict(model, **options,
                                       render_engine=src.view if src.view != "Depth map"
                                       else "Standard PBR", view_mode=_layer(src)),
                          depth_normalised=r.depth_normalised_full, render=draw)
        if res["pbr_error"]:
            raise RuntimeError("The mesh was written but its PBR package failed: "
                               + res["pbr_error"])
        folder = os.path.dirname(res["path"])
        bn = os.path.splitext(os.path.basename(res["path"]))[0]
        files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(bn))
        return files + ([res["zip"]] if res["zip"] else [])
    return run


def _print(src, out_dir):
    from helio.application.export_3d import print_mesh
    D = _need_data(src)
    size = max(D.w, D.h) * src.mm_per_px if src.mm_per_px else 100.0
    return [print_mesh(D, os.path.join(out_dir, f"{file_stem(src.name)}_print.stl"),
                       edge_trim_pct=float(src.shared.get("edge_trim", 2.0)),
                       options={"size_mm": size})["path"]]


def _facsimile(src, out_dir):
    from helio.application.export_3d import facsimile_package
    D = _need_data(src)
    s = src.result.settings
    return facsimile_package(D.depth_vis_p, _folder(out_dir, f"{file_stem(src.name)}_facsimile"),
                             file_stem(src.name), {"fmt": "OBJ"}, mm_per_px=src.mm_per_px or None,
                             z_mm_per_unit=src.z_mm_per_unit or None, albedo=D.albedo_rgb_full,
                             obj_id=src.name, depth_engine=s.depth_engine,
                             normal_engine=s.resolved_normal_engine())["files"]


def _lod(src, out_dir):
    from helio.application.export_3d import lod_package
    return lod_package(_need_data(src), _folder(out_dir, f"{file_stem(src.name)}_lod"), src.name,
                       ss=src.shared)["files"]


def _displacement(src, out_dir):
    from helio.application.export_3d import displacement_package
    return displacement_package(_need_data(src), _folder(out_dir, f"{file_stem(src.name)}_displacement"),
                                src.name, ss=src.shared,
                                depth_normalised=src.result.depth_normalised_full)["files"]


def _hf_viewer(src, out_dir):
    from helio.application.export_3d import hf_normals_viewer
    r = src.result
    folder = _folder(out_dir, f"{file_stem(src.name)}_hf_viewer")
    settings = X.build_export_settings(src.shared, X.MeshOptions(), r.w, r.h, obj_id=src.name)
    hf_normals_viewer(r.normals_full, r.albedo_rgb_full, settings, folder)
    return sorted(os.path.join(folder, f) for f in os.listdir(folder))


def _usdz(src, out_dir):
    """USDZ for AR: one OpenUSD file at the object's true size, checked against Apple's AR rules."""
    from helio.application import export_usd as U
    from helio.domain.view_controls import light_of
    r = src.result
    path = os.path.join(out_dir, f"{file_stem(src.name)}.usdz")
    lx, ly, lz = light_of(src.view_settings or {}, src.view)
    try:
        from helio_qt import __version__ as v
    except Exception:                                    # noqa: BLE001
        v = ""
    out = U.export(path, depth=r.depth_vis_p, normals=r.normals_full, albedo=r.albedo_rgb_full,
                   mm_per_px=src.mm_per_px or None, lights=src.lights, view_light=(lx, -ly, lz),
                   meta=dict(object=src.name, helio=f"Helio {v}".strip(),
                             normals=r.settings.resolved_normal_engine(),
                             height=r.settings.resolved_depth_engine()))
    ok, problems = U.check(path)
    if not ok:
        _log.info("usdz: Apple's AR rules found problems: %s", "; ".join(problems))
    return [path] + [f for f in out.get("files", ()) if f != path]


MESH_EXPORTS = {"usdz": _usdz, "glb_mesh": _mesh_runner("glb", ".glb"), "obj": _mesh_runner("obj", ".obj"),
                "ply": _mesh_runner("ply", ".ply"),
                "point_cloud": _mesh_runner("point", ".ply", "_points"),
                "sketchfab": _mesh_runner("glb", ".glb", "_sketchfab", sketchfab=True),
                "print": _print, "facsimile": _facsimile, "lod": _lod, "displacement": _displacement,
                "hf_viewer": _hf_viewer}


# ── web viewers (helio.application.export_web) ─────────────────────────────

def _proxy_drawer(src, D):
    """draw(engine) at preview size, each view with its own settings."""
    from helio.application.render import composite, render
    from helio.domain.registry import default_engine_settings
    layer, defaults = _layer(src), default_engine_settings()

    def draw(eng):
        st = dict(src.engine_settings.get(eng) or defaults.get(eng) or default_engine_settings((eng,))[eng])
        img = render(eng, D, st, D.ph, D.pw, True)
        return composite(eng, img, D, st, *np.asarray(img).shape[:2], True, view_mode=layer)
    return draw


def _trim(src):
    from helio.application.export_3d import region
    return region(_need_data(src), float(src.shared.get("edge_trim", 2.0)))


def _webrti(src, out_dir):
    from helio.application.export_web import webrti
    D, r = _need_data(src), src.result
    try:
        hybrid = _proxy_drawer(src, D)("Standard PBR")
    except Exception:
        hybrid = None
    return webrti(D, _folder(out_dir, f"{file_stem(src.name)}_webrti"), hybrid, obj_id=src.name,
                  mm_per_px=src.mm_per_px or None, z_mm_per_unit=src.z_mm_per_unit or None,
                  normal_engine=r.settings.normal_engine, depth_engine=r.settings.depth_engine)


def _shown(src):
    """The solve's arrays for the web viewers, as Inspect shows them: inside the
    trimmed region, mirrored and turned like the screen (Mirror / Invert and the
    turn).  A SimpleNamespace with normals, albedo, roughness, spec_var, depth,
    depth_norm, h, w, box (the whole of what is returned) and layer(img), which
    moves a picture of the whole frame (any size) the same way."""
    from types import SimpleNamespace
    from helio.application.viewing import screen_surface
    r = src.result
    box = _trim(src)
    d = src.screen()
    turned = bool(float(src.turn or 0.0) % 360.0) or d.transformed()
    if not turned:
        return SimpleNamespace(normals=r.normals_full, albedo=r.albedo_rgb_full,
                               roughness=r.roughness_full, spec_var=r.spec_var,
                               depth=r.depth_vis_p, depth_norm=r.depth_normalised_full,
                               h=r.h, w=r.w, box=box, layer=lambda img: img)
    x1, y1, x2, y2 = box

    def cut(a):
        return None if a is None else np.asarray(a)[y1:y2, x1:x2]
    rough = cut(r.roughness_full)
    n, alb, maps, hs = screen_surface(
        d, src.turn, normals=cut(r.normals_full), colour=cut(r.albedo_rgb_full),
        maps=[m for m in (rough, cut(r.spec_var)) if m is not None],
        heights=(cut(r.depth_vis_p), cut(r.depth_normalised_full)))
    maps = list(maps)
    rough = maps.pop(0) if rough is not None else None
    spec = maps.pop(0) if r.spec_var is not None else None
    hh, ww = n.shape[:2]

    def layer(img):
        a = np.asarray(img)
        sy, sx = a.shape[0] / float(r.h), a.shape[1] / float(r.w)
        a = a[int(y1 * sy):max(int(y1 * sy) + 1, int(round(y2 * sy))),
              int(x1 * sx):max(int(x1 * sx) + 1, int(round(x2 * sx)))]
        return screen_surface(d, src.turn, colour=a)[1]
    return SimpleNamespace(normals=n, albedo=alb, roughness=rough, spec_var=spec,
                           depth=hs[0], depth_norm=hs[1], h=hh, w=ww, box=(0, 0, ww, hh),
                           layer=layer)


def _rti_combined(src, out_dir):
    from helio.application.export_web import COMBINED_BAKE, baked_frames, combined_rti_viewer
    D = _need_data(src)
    extra, _flat = baked_frames(_proxy_drawer(src, D), COMBINED_BAKE, 3.0)
    s = _shown(src)
    path = os.path.join(out_dir, f"{file_stem(src.name)}_rti_combined.html")
    combined_rti_viewer(path, s.normals, s.albedo, s.roughness, s.spec_var, s.h, s.w, s.box,
                        src.name, [(k, s.layer(a)) for k, a in extra], budget_mb=25.0)
    return [path]


def _rti_3d(src, out_dir):
    """Relight and 3D in one file: the combined RTI viewer and a GLB viewer
    behind two tabs (export_web.rti_3d_viewer), both turned and mirrored as
    Inspect shows the object."""
    import shutil
    import tempfile
    from helio.application import export_web as EW
    D = _need_data(src)
    stem = file_stem(src.name)
    work = tempfile.mkdtemp(prefix="helio_rti3d_")
    try:
        extra, _flat = EW.baked_frames(_proxy_drawer(src, D), EW.COMBINED_BAKE, 3.0)
        s = _shown(src)
        rti = os.path.join(work, "rti.html")
        EW.combined_rti_viewer(rti, s.normals, s.albedo, s.roughness, s.spec_var, s.h, s.w,
                               s.box, src.name, [(k, s.layer(a)) for k, a in extra],
                               budget_mb=EW.RTI_3D_RTI_MB)
        with open(rti, encoding="utf-8") as f:
            rti_page = f.read()
        room = EW.RTI_3D_MB - len(rti_page.encode("utf-8")) / 1e6
        base = X.build_export_settings(src.shared, X.MeshOptions(), s.w, s.h, obj_id=src.name,
                                       crop_box=s.box)
        x1, y1, x2, y2 = base.get("crop", s.box)
        os.makedirs(os.path.join(work, "model"), exist_ok=True)
        for verts, tex, quality in EW.RTI_3D_TIERS:      # the finest model that fits
            settings = dict(base, tex_res=tex, normal_tex_res=tex, tex_jpeg=quality,
                            viewer_look=EW.RTI_3D_LOOK, viewer_tilt=EW.RTI_3D_TILT)
            settings["step"] = max(int(base.get("step", 1)), EW.rti_3d_step(x2 - x1, y2 - y1, verts))
            model = X.web_viewer(s.depth, s.depth_norm, s.normals, s.albedo, settings,
                                 os.path.join(work, "model"))
            with open(model["html"], encoding="utf-8") as f:
                model_page = f.read()
            if len(model_page.encode("utf-8")) / 1e6 <= room:
                break
        path = os.path.join(out_dir, f"{stem}_relight_and_3d.html")
        EW.rti_3d_viewer(path, rti_page, model_page, src.name)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return [path]


def _relight(src, out_dir):
    from helio.application.export_web import app_like, relight_package
    r = src.result
    folder = _folder(out_dir, f"{file_stem(src.name)}_relight")
    prov = app_like(name=src.name, images=src.images, lights=r.light_matrix,
                    light_kind="file" if src.light_source == "measured" else src.light_source,
                    h=r.h, w=r.w, normal_engine=r.settings.normal_engine,
                    depth_engine=r.settings.depth_engine, engine=src.view,
                    engine_settings=src.engine_settings)
    _fitted(r)
    from helio.application.pipeline import region_fraction
    relight_package(folder, r.ptm_coeffs, r.albedo_rgb_full, r.light_matrix, r.h, r.w,
                    src.images, provenance_app=prov, region=region_fraction(r.crop, r.base_w, r.base_h))
    return sorted(os.path.join(dp, f) for dp, _d, fs in os.walk(folder) for f in fs)


def _openlime_hsh(src, out_dir):
    from helio.application.export_web import app_like, openlime_hsh_package
    from helio.application.pipeline import region_fraction
    r = src.result
    folder = _folder(out_dir, f"{file_stem(src.name)}_openlime_hsh")
    prov = app_like(name=src.name, images=src.images, lights=r.light_matrix,
                    light_kind="file" if src.light_source == "measured" else src.light_source,
                    h=r.h, w=r.w, normal_engine=r.settings.normal_engine,
                    depth_engine=r.settings.depth_engine, engine=src.view,
                    engine_settings=src.engine_settings)
    res = openlime_hsh_package(folder, src.images, r.light_matrix, normals=r.normals_full,
                               region=region_fraction(r.crop, r.base_w, r.base_h),
                               pixel_mm=src.mm_per_px or None, result_w=r.w, title=src.name,
                               provenance_app=prov, white_balance=src.white_balance)
    _openlime_maps(src, res["dir"])
    return sorted(os.path.join(dp, f) for dp, _d, fs in os.walk(res["dir"]) for f in fs)


#: The maps the OpenLIME viewer offers in its Views menu, in order: (id, label, file in maps/).
OPENLIME_VIEWS = (("colour", "Colour", "albedo_16bit.tif"),
                  ("specular", "Specular", "sds_specular_component.png"),
                  ("diffuse", "Diffuse colour", "sds_diffuse_albedo.png"),
                  ("cavity", "Cavity", "cavity.png"),
                  ("pca", "PCA 1-3 (RGB)", "pca_rgb.png"),
                  ("pca1", "PCA 1", "pca_1.png"),
                  ("pca2", "PCA 2", "pca_2.png"),
                  ("pca3", "PCA 3", "pca_3.png"))


def _openlime_views(src, folder, maps):
    """Browser copies of the maps (maps/web/, 8-bit tarzoom) and index.html
    rewritten with them in its Views menu, each scaled onto the RTI."""
    import json
    import cv2
    from helio.io import openlime_hsh as O
    with open(os.path.join(folder, "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    web = os.path.join(maps, "web")
    views = []
    for vid, label, name in OPENLIME_VIEWS:
        p = os.path.join(maps, name)
        if not os.path.isfile(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        url, layout, w = O.web_view(img, web, vid)
        views.append(dict(id=vid, label=label, url=f"maps/web/{url}", layout=layout,
                          z=round(float(info["width"]) / float(w), 6)))
    layout = "tarzoom" if os.path.isfile(os.path.join(folder, "plane_0.tzi")) else "image"
    has_normals = os.path.isfile(os.path.join(folder, "normals.tzi" if layout == "tarzoom" else "normals.jpg"))
    brdf = _openlime_brdf(maps, web, float(info["width"]))
    for name in ("sds_gloss_index.png", "sds_gloss_index_16bit.tif"):     # only the shine reads it
        try:
            os.remove(os.path.join(maps, name))
        except OSError:
            pass
    with open(os.path.join(folder, "index.html"), "w", encoding="utf-8") as f:
        f.write(O.viewer_html(layout, has_normals, src.name, views, brdf))


def _openlime_brdf(maps, web, rti_w):
    """The relightable maps' inputs for OpenLIME's BRDF layer, all one size and
    layout: kd the colour with the shine taken out (else the colour), ks the
    specular part, gloss the gloss index, normals (n + 1) / 2, +Y up.  None when
    the normal map or the colour is missing."""
    import cv2
    from helio.io import openlime_hsh as O

    def read(name, colour):
        p = os.path.join(maps, name)
        a = cv2.imread(p, cv2.IMREAD_UNCHANGED) if os.path.isfile(p) else None
        if a is None:
            return None
        if a.dtype == np.uint16:
            a = (a >> 8).astype(np.uint8)
        if a.ndim == 3:
            a = cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2RGB)
        elif colour:
            a = np.repeat(a[..., None], 3, axis=2)
        return a
    kd = read("sds_diffuse_albedo.png", True)
    if kd is None:
        kd = read("albedo_16bit.tif", True)
    nrm = read("normal_map_16bit.tif", True)
    if kd is None or nrm is None or kd.shape[:2] != nrm.shape[:2]:
        return None
    out = {}
    def stretch(a, lo_pct, hi_pct):
        # 0..1 over this capture's own range, so the shine looks alike from object to object
        if a is None:
            return None
        f = a[..., 0].astype(np.float32)
        lo, hi = np.percentile(f[::4, ::4], (lo_pct, hi_pct))
        g = np.clip((f - lo) / max(float(hi - lo), 1.0) * 255.0, 0, 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)
    ks = stretch(read("sds_specular_component.png", True), 0.0, 99.5)
    gloss = stretch(read("sds_gloss_index.png", True), 1.0, 99.0)
    for key, img in (("kd", kd), ("normals", nrm), ("ks", ks), ("gloss", gloss)):
        if img is None or img.shape[:2] != kd.shape[:2]:
            continue
        url, layout, w = O.web_view(img, web, "brdf_" + key, 95 if key == "normals" else 90)
        out[key] = f"maps/web/{url}"
        out["layout"] = layout
    out["z"] = round(rti_w / float(kd.shape[1]), 6)
    return out


def _openlime_maps(src, folder):
    """The surface maps beside the RTI, in maps/: colour and normals (16-bit),
    cavity, the specular / diffuse separation (specular, diffuse colour,
    highlight mask; gloss feeds the viewer's shine and is not kept) and the
    PCA of the frames, at the solve's size.  A map that cannot be made is named in maps/README.txt; the
    RTI itself is already written and stands."""
    from helio.application.export_images import PBR_MAPS, pbr_suite
    D = src.data
    maps = os.path.join(folder, "maps")
    note = ""
    if D is None or getattr(D, "normals_full", None) is None:
        note = "No maps: the capture is not processed."
    else:
        try:
            L, st = _stack(src)
        except Exception:
            L, st = None, None                 # the SDS maps fall back to the variance proxy
        try:
            pbr_suite(D, maps, {k: k in ("albedo", "normal_map", "cavity", "sds") for k, _ in PBR_MAPS},
                      step=max(1, int(src.shared.get("export_step", 1) or 1)), lights=L, stack=st)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("OpenLIME maps", exc_info=True)
            note = f"Some maps could not be made: {e}"
        pca = getattr(D, "pca_full", None)
        if pca is not None:
            try:
                import cv2
                os.makedirs(maps, exist_ok=True)
                cv2.imwrite(os.path.join(maps, "pca_rgb.png"), cv2.cvtColor(pca, cv2.COLOR_RGB2BGR))
                for i in range(3):
                    cv2.imwrite(os.path.join(maps, f"pca_{i + 1}.png"), pca[..., i])
            except Exception as e:
                note = (note + "\n" if note else "") + f"The PCA maps could not be made: {e}"
    os.makedirs(maps, exist_ok=True)
    try:
        _openlime_views(src, folder, maps)
    except Exception as e:                      # the RTI and the maps stand without the menu
        import logging
        logging.getLogger(__name__).warning("OpenLIME views", exc_info=True)
        note = (note + "\n" if note else "") + f"The viewer's map views could not be made: {e}"
    r = src.result
    with open(os.path.join(maps, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"""Surface maps of {src.name}
{"=" * (17 + len(src.name))}

{r.w} x {r.h} pixels: the size HELIO solved at (the RTI planes beside this folder
are at the photographs' full size).  Same orientation; +Y up in the normal maps.

  albedo_16bit.tif                 colour, sRGB{" (white balanced)" if src.white_balance else ""}
  normal_map_16bit.tif             photometric-stereo normals, (n + 1) / 2 in RGB, +Y up
  cavity.png                       hollows
  sds_specular_component*.tif/png  the specular (shiny) part of the reflection
  sds_diffuse_albedo*.tif/png      the colour with the specular part taken out
  sds_highlight_mask.png           where the highlights are
  pca_rgb.png, pca_1..3.png        the first three principal components of the frames
                                   (how each point's brightness changes with the light),
                                   each stretched to 0..255; pca_rgb holds 1, 2, 3 as R, G, B
{chr(10) + note + chr(10) if note else ""}""")



def _rti_deepzoom(src, out_dir):
    from helio.application.export_web import rti_deepzoom_relight
    r = src.result
    folder = _folder(out_dir, f"{file_stem(src.name)}_rti_deepzoom")
    _fitted(r)
    rti_deepzoom_relight(folder, r.normals_full, r.albedo_rgb_full, r.hsh_coeffs, r.h, r.w, src.name)
    return sorted(os.path.join(dp, f) for dp, _d, fs in os.walk(folder) for f in fs)[:200]


def _iiif(src, out_dir):
    from helio.application.export_web import iiif_manifest
    r = src.result
    return [iiif_manifest(os.path.join(out_dir, f"{file_stem(src.name)}_iiif3d.json"),
                          obj_id=src.name, w=r.w, h=r.h, n_images=r.n_images, lights=r.light_matrix,
                          normal_engine=r.settings.normal_engine,
                          depth_engine=r.settings.depth_engine, mm_per_px=src.mm_per_px or None,
                          z_mm_per_unit=src.z_mm_per_unit or None)["path"]]


# ── discipline records (helio.application.export_scholarly, heritage_tools) ───

def _object_images(src, out_dir, folder: str, most: int = 6) -> list[tuple[str, str, int, int]]:
    """The colour and the object profile's views as JPEGs in *folder*: (label, file, w, h)."""
    from helio.domain.object_profiles import of
    from helio.domain.registry import RENDER_ENGINES, view_name
    D = _need_data(src)
    draw = _drawer(src, D)
    shots = [("Colour", D.albedo_rgb_full)]
    views = [v for v in of(src.meta).views if v in RENDER_ENGINES] or ["Standard PBR"]
    if src.view in RENDER_ENGINES and src.view not in views:
        views.insert(0, src.view)
    shots += [(view_name(v), None) for v in views[:most - 1]]
    out = []
    for lab, img in shots:
        if img is None:
            img = draw(next(v for v in views if view_name(v) == lab))
        img = np.asarray(img)[..., :3]
        fname = f"{file_stem(src.name)}_{file_stem(lab).lower()[:32]}.jpg"
        cv2.imwrite(os.path.join(folder, fname), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        out.append((lab, fname, img.shape[1], img.shape[0]))
    return out


def _iiif_images(src, out_dir):
    from helio.application.export_scholarly import iiif_image_manifest
    folder = _folder(out_dir, f"{file_stem(src.name)}_iiif")
    shots = _object_images(src, out_dir, folder)
    files = [os.path.join(folder, f) for _l, f, _w, _h in shots]
    return files + iiif_image_manifest(folder, file_stem(src.name), src.meta, shots,
                                       base_url=str(src.shared.get("iiif_base_url", "") or ""),
                                       mm_per_px=src.mm_per_px or 0.0)


def _nomisma(src, out_dir):
    from helio.application.export_scholarly import nomisma_record
    D = _need_data(src)
    img = os.path.join(out_dir, f"{file_stem(src.name)}_colour.jpg")
    cv2.imwrite(img, cv2.cvtColor(np.asarray(D.albedo_rgb_full)[..., :3], cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    return [img] + nomisma_record(os.path.join(out_dir, f"{file_stem(src.name)}_nomisma"), src.meta,
                                  file_stem(src.name), images=[img])


def _epidoc(src, out_dir):
    from helio.application.export_scholarly import epidoc_record
    folder = _folder(out_dir, f"{file_stem(src.name)}_epidoc")
    shots = _object_images(src, out_dir, folder, most=3)
    r = src.result
    path = epidoc_record(os.path.join(folder, f"{file_stem(src.name)}.xml"), src.meta, file_stem(src.name),
                         images=[f for _l, f, _w, _h in shots], w=r.w, h=r.h,
                         mm_per_px=src.mm_per_px or 0.0)
    return [os.path.join(folder, f) for _l, f, _w, _h in shots] + [path]


def _line_drawing(src, out_dir):
    from helio.application.heritage_tools import line_drawing, line_drawing_svg
    D = _need_data(src)
    stem = file_stem(src.name)
    photo = os.path.join(out_dir, f"{stem}_drawing_photo.jpg")
    t = _screen_turn(src)                     # as the screen shows the object
    cv2.imwrite(photo, cv2.cvtColor(np.ascontiguousarray(t.image(np.asarray(D.albedo_rgb_full)[..., :3])),
                                    cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    dep = np.asarray(D.depth_vis_p)
    svg = line_drawing_svg(os.path.join(out_dir, f"{stem}_drawing.svg"),
                           line_drawing(t.image(dep, float(np.nanmin(dep)))),
                           mm_per_px=src.mm_per_px or 0.0, title=src.name,
                           background_png=os.path.basename(photo))
    return [svg, photo]


def _compare(src, out_dir):
    from helio.application.export_web import comparison_page
    D = _need_data(src)
    a = src.view if src.view != "Depth map" else "Standard PBR"
    b = src.compare_view if src.compare_view and src.compare_view != a else (
        "Surface Curvature (Scratches)" if a != "Surface Curvature (Scratches)" else "Standard PBR")
    draw = _drawer(src, D)
    return [comparison_page(draw(a), draw(b), a, b,
                            _folder(out_dir, f"{file_stem(src.name)}_comparison"))]


def _engine_sweep(src, out_dir):
    from helio.application.export_web import baked_frames
    from helio.domain.registry import RENDER_ENGINES
    from helio.io.exporters import export_engine_sweep_html
    frames, _skipped = baked_frames(_proxy_drawer(src, _need_data(src)), RENDER_ENGINES, 2.0)
    if not frames:
        raise X.NothingToExport("No view produced a usable frame.")
    path = os.path.join(out_dir, f"{file_stem(src.name)}_engine_sweep.html")
    export_engine_sweep_html(path, frames, obj_id=src.name)
    return [path]


def _rti_web(src, out_dir):
    """The RTI web package (helio.application.rti_web), with its default options."""
    from helio.application import rti_web
    D, r = _need_data(src), src.result
    fx = dict(src.facts or {})
    _fitted(r)
    opts = rti_web.Options(title=src.name, licence=fx.get("licence") or "CC BY 4.0",
                           creator=fx.get("creator", ""), institution=fx.get("institution", ""))
    res = rti_web.package(_folder(out_dir, f"{file_stem(src.name)}_web"), normals=r.normals_full,
                          albedo=r.albedo_rgb_full, lights=src.lights,
                          ptm_coeffs=getattr(r, "ptm_coeffs", None),
                          roughness=getattr(r, "roughness_full", None),
                          spec_var=getattr(r, "spec_var", None), images=src.images,
                          mm_per_px=src.mm_per_px or None, z_mm_per_unit=src.z_mm_per_unit or None,
                          opts=opts)
    return [res["index"]] + ([res["quick_view"]] if res["quick_view"] else [])


WEB_EXPORTS = {"rti_web": _rti_web, "webrti": _webrti, "rti_combined": _rti_combined,
               "rti_3d": _rti_3d,
               "relight_web": _relight, "openlime_hsh": _openlime_hsh,
               "rti_deepzoom": _rti_deepzoom, "iiif": _iiif, "compare": _compare,
               "engine_sweep": _engine_sweep}


# ── archive and data (helio.application.export_archive, export_dem) ─────────

def _prov(src):
    from helio.application.export_web import app_like
    r = src.result
    return app_like(name=src.name, images=src.images, lights=r.light_matrix,
                    light_kind="file" if src.light_source == "measured" else src.light_source,
                    h=r.h, w=r.w, normal_engine=r.settings.normal_engine,
                    depth_engine=r.settings.depth_engine, engine=src.view,
                    engine_settings=src.engine_settings)


def _zenodo(src, out_dir):
    from helio.application.export_archive import zenodo_package
    r = src.result
    return [zenodo_package(os.path.join(out_dir, f"{file_stem(src.name)}_zenodo.zip"), src.name,
                           normal_engine=r.settings.normal_engine,
                           depth_engine=r.settings.depth_engine, lights=r.light_matrix,
                           images=src.images, meta=src.meta, provenance_app=_prov(src))["path"]]


def _morphosource(src, out_dir):
    from helio.application.export_archive import morphosource_package
    D, r = _need_data(src), src.result
    path = os.path.join(out_dir, f"{file_stem(src.name)}_morphosource.zip")
    return [morphosource_package(D, path, file_stem(src.name), ss=src.shared,
                                 depth_normalised=r.depth_normalised_full,
                                 normal_engine=r.settings.normal_engine,
                                 depth_engine=r.settings.depth_engine,
                                 mm_per_px=src.mm_per_px or None,
                                 z_mm_per_unit=src.z_mm_per_unit or None, n_images=r.n_images,
                                 lights=r.light_matrix,
                                 institution=src.meta.get("institution", ""))]


def _cidoc(src, out_dir):
    from helio.application.export_archive import write_cidoc
    r = src.result
    return [write_cidoc(os.path.join(out_dir, f"{file_stem(src.name)}_cidoc.jsonld"),
                        obj_id=src.name, normal_engine=r.settings.normal_engine,
                        depth_engine=r.settings.depth_engine,
                        light_count=0 if r.light_matrix is None else len(r.light_matrix),
                        w=r.w, h=r.h)]


def _dublin_core(src, out_dir):
    from helio.application.export_archive import dublin_core_tiff
    D = _need_data(src)
    view = "Standard PBR" if src.view == "Depth map" else src.view
    return [dublin_core_tiff(_drawer(src, D)(view),
                             os.path.join(out_dir, f"{file_stem(src.name)}_dc.tif"), src.name, view)]


def _dem(src, out_dir):
    from helio.application.export_dem import dem_package
    from helio.domain.object_profiles import of as object_of
    D, r = _need_data(src), src.result
    folder = _folder(out_dir, f"{file_stem(src.name)}_GIS_DEM")
    lights = [tuple(float(c) for c in v) for v in r.light_matrix] if r.light_matrix is not None else []
    dem_package(D, folder, src.name, mm_per_px=src.mm_per_px or None,
                z_mm_per_unit=src.z_mm_per_unit or None,
                invert=bool(getattr(src.screen(), "depth_invert", False)),
                normal_engine=r.settings.normal_engine, lights=lights,
                light_kind="estimated" if src.light_source == "estimated" else "measured",
                provenance_app=_prov(src), object_type=object_of(src.meta).key,
                sections=(src.facts or {}).get("kept_sections", ()),
                annotations=(src.facts or {}).get("annotations", ()))
    return sorted(os.path.join(folder, f) for f in os.listdir(folder))


def _light_vectors(src, out_dir):
    from helio.application.export_archive import light_vectors_csv
    return [light_vectors_csv(src.result.light_matrix,
                              os.path.join(out_dir, f"{file_stem(src.name)}_light_vectors.csv"))]


def _light_cube(src, out_dir):
    from helio.application.export_archive import light_cube
    r = src.result
    s = r.settings
    notes = [n for flag, n in ((s.flatten_illumination, "illumination flattening"),
                               (s.multi_norm, "median brightness normalisation"),
                               (s.pre_clahe, "per-image CLAHE")) if flag]
    path = os.path.join(out_dir, f"{file_stem(src.name)}_light_cube.tif")
    res = light_cube(src.images, path, want_shape=(r.h, r.w), crop=r.crop,
                     shifts=r.alignment_shifts, mm_per_px=src.mm_per_px, obj=file_stem(src.name),
                     notes=notes, lights=r.light_matrix,
                     light_kind="estimated" if src.light_source == "estimated" else "measured",
                     base=(r.base_w, r.base_h))
    return [res["path"]] + ([res["lp"]] if res["lp"] else [])


def _engine_stack(src, out_dir):
    from helio.application.export_archive import engine_stack
    from helio.domain.registry import RENDER_ENGINES
    D = _need_data(src)
    path = os.path.join(out_dir, f"{file_stem(src.name)}_engine_stack.tif")
    return [engine_stack(_proxy_drawer(src, D), RENDER_ENGINES, path, mm_per_px=src.mm_per_px,
                         full_w=D.w, obj=file_stem(src.name))["path"]]


def _solved(src):
    """The solved reflectance of this capture, if it has one."""
    return getattr(src.result, "reflectance_rgb", None)


def _paradata(src, results, base):
    """The paradata of the files in *results* (key, files, problem), in *base*."""
    from helio.application import paradata
    r, D = src.result, src.data
    solved = getattr(D, "solved_colour", None)
    p = paradata.build(record=processing_record(src), images=src.images, lights=r.light_matrix,
                       light_source=src.light_source, settings=r.settings, tone=src.tone,
                       solved=bool(solved and D.albedo_rgb_full is solved[0]),
                       check=(src.facts or {}).get("colour"), results=results, base=base)
    return paradata.write(p, base, file_stem(src.name))


def _heights(r):
    from helio.application.heights import height_px
    return height_px(r)


def _linear_masters(src, out_dir):
    from helio.application.export_masters import linear_masters
    r = src.result
    return linear_masters(_folder(out_dir, f"{file_stem(src.name)}_exr"), file_stem(src.name),
                          colour=r.albedo_rgb_full, normals=r.normals_full, depth=_heights(r),
                          solved=_solved(src), z_mm_per_unit=src.z_mm_per_unit)


def _usd_layered(src, out_dir):
    from helio.application.export_masters import usd_layered
    from helio.application.paradata import frame_records
    r, stem = src.result, file_stem(src.name)
    rec = processing_record(src)
    return usd_layered(_folder(out_dir, f"{stem}_usd"), stem, depth=r.depth_poisson, normals=r.normals_full,
                       colour=r.albedo_rgb_full, mm_per_px=src.mm_per_px or None, solved=_solved(src),
                       meta={"title": src.meta.get("title", src.name)},
                       record=dict(object=rec["object"], software=rec["software"],
                                   normals=r.settings.normal_engine, heights=r.settings.depth_engine,
                                   frames=frame_records(src.images)),
                       lights=r.light_matrix, z_mm_per_unit=src.z_mm_per_unit,
                       sections=(src.facts or {}).get("kept_sections", ()),
                       annotations=(src.facts or {}).get("annotations", ()),
                       colour_check=(src.facts or {}).get("colour"))


def _colour_validation(src, out_dir):
    from helio.application import colour_check as C
    from helio.io import load_and_process_single
    D = _need_data(src)
    stages = []
    first = load_and_process_single(src.images[0])[0] if src.images else None
    if first is not None:
        stages.append(("Photograph", first))
    stages.append(("Colour map", src.result.albedo_rgb_full))
    if _solved(src) is not None:
        stages.append(("Solved reflectance", _solved(src)))
    stages.append((f"Render: {src.view}", np.asarray(_drawer(src, D)(src.view))[..., :3].astype(np.uint8)))
    res = C.validate(stages)
    if not any("mean" in s for s in res["stages"]):
        raise X.NothingToExport("No ColorChecker Classic (24 patches) in this capture to validate against.")
    return C.write_validation(res, os.path.join(out_dir, f"{file_stem(src.name)}_colour_validation"))


def _deposit(src, out_dir):
    """A folder a repository can take: masters and derivatives apart, the record beside them."""
    from helio.io.manifest import write_manifest
    top = _folder(out_dir, f"{file_stem(src.name)}_deposit")
    results = []
    for sub, keys in (("masters", ("linear_masters", "usd_layered")), ("derivatives/web", ("glb_mesh",)),
                      ("derivatives/print", ("stl",)),
                      ("metadata", ("lido", "dublin_core", "lp", "provenance", "colour_validation"))):
        where = os.path.join(top, sub)
        os.makedirs(where, exist_ok=True)
        for k in keys:
            try:
                results.append((k, export_one(k, src, where), ""))
            except Exception as e:                       # one piece missing is said, not fatal
                _log.info("deposit: %s left out: %s", k, e)
                results.append((k, [], str(e)))
    _paradata(src, results, top)
    missing = [f"{k}: {why}" for k, files, why in results if why]
    if missing:
        with open(os.path.join(top, "LEFT_OUT.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")
    write_manifest(top, object_id=src.name)
    return sorted(os.path.join(d, n) for d, _s, fs in os.walk(top) for n in fs)


ARCHIVE_EXPORTS = {"deposit": _deposit, "linear_masters": _linear_masters, "usd_layered": _usd_layered,
                   "colour_validation": _colour_validation, "nomisma": _nomisma, "epidoc": _epidoc, "iiif_images": _iiif_images,
                   "line_drawing": _line_drawing, "zenodo": _zenodo, "morphosource": _morphosource, "cidoc": _cidoc,
                   "dublin_core": _dublin_core, "geotiff": _dem, "light_vectors": _light_vectors,
                   "light_cube": _light_cube, "engine_stack": _engine_stack}


# ── reports (helio.application.export_reports) ─────────────────────────────

def _sheet(src, out_dir):
    from helio.application.export_reports import SHEET_DEFAULT, publication_sheet
    D, r = _need_data(src), src.result
    pd = _proxy_drawer(src, D)
    from helio.application.render import composite, render
    from helio.domain.registry import default_engine_settings
    defaults = default_engine_settings()

    def draw(eng, force_light=False):
        if not force_light:
            return pd(eng)
        st = dict(src.engine_settings.get(eng) or defaults.get(eng) or default_engine_settings((eng,))[eng], lx=0.0, ly=0.0, lz=1.0)
        img = render(eng, D, st, D.ph, D.pw, True)
        return composite(eng, img, D, st, *np.asarray(img).shape[:2], True, view_mode=_layer(src))
    path = os.path.join(out_dir, f"{file_stem(src.name)}_contact_sheet.jpg")   # the new sheet is _publication_sheet
    lights = r.light_matrix
    t = _screen_turn(src)
    if t.any:                       # as the screen shows the object
        from helio.application.viewing import TurnedData
        plain = draw
        draw = lambda eng, force_light=False: t.image(np.asarray(plain(eng, force_light)))   # noqa: E731
        lights = None if lights is None else t.vectors(np.asarray(lights, np.float64))
        D = TurnedData(D, t)
    res = publication_sheet(D, path, draw=draw, selected=SHEET_DEFAULT,
                            engine_settings=src.engine_settings, lights=lights, ss=src.shared,
                            n_images=r.n_images, capture_meta=src.meta, profile=src.profile,
                            mm_per_px=src.mm_per_px or None, z_mm_per_unit=src.z_mm_per_unit or None,
                            ppi=(25.4 / src.mm_per_px) if src.mm_per_px else None,
                            scale_val=(D.w * src.mm_per_px) if src.mm_per_px else 0.0,
                            obj=src.name, engine=src.view)
    return [res["path"]]


def _figures(src, out_dir):
    from helio.application.export_reports import FIGURE_ENGINES, publication_figures
    D, r = _need_data(src), src.result
    t = _screen_turn(src)
    pd = _proxy_drawer(src, D)
    nw, nh = t.size(r.w, r.h)
    return publication_figures(lambda eng: t.image(np.asarray(pd(eng))), FIGURE_ENGINES,
                               _folder(out_dir, f"{file_stem(src.name)}_figures"),
                               full_shape=(nh, nw), obj_id=src.name, mm_per_px=src.mm_per_px)


def _condition(src, out_dir):
    from helio.application.export_reports import condition_map
    r = src.result
    from helio.application.heights import height_px
    t = _screen_turn(src)
    hp = height_px(r)
    return condition_map(t.image(hp, float(np.nanmin(hp))), t.image(r.albedo_rgb_full),
                         _folder(out_dir, f"{file_stem(src.name)}_condition"), file_stem(src.name),
                         z_mm_per_unit=src.z_mm_per_unit or None, obj_id=src.name,
                         operator=src.meta.get("operator", ""), notes=src.meta.get("notes", ""),
                         depth_engine=r.settings.depth_engine)


def _rose(src, out_dir):
    from helio.io.figures import rose_diagram
    img = rose_diagram(src.result.normals_full, bins=36, title="Surface orientation — whole surface")
    path = os.path.join(out_dir, f"{file_stem(src.name)}_rose.png")
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return [path]


def _plate_a3(src, out_dir):
    """The publication plate (helio.application.plate_doc): A3, PDF and PNG."""
    from helio.application.plate_doc import Facts, plate
    from helio.application.render import composite, render
    from helio.domain.registry import default_engine_settings
    from helio.core.sphere import dome_coverage
    D, r = _need_data(src), src.result
    defaults = default_engine_settings()

    def draw(eng, overrides=None):
        st = dict(src.engine_settings.get(eng) or defaults.get(eng)
                  or default_engine_settings((eng,))[eng])
        st.update(overrides or {})
        img = render(eng, D, st, D.ph, D.pw, True)
        return composite(eng, img, D, st, *np.asarray(img).shape[:2], True, view_mode="Shading")
    fx = dict(src.facts or {})
    lights = src.lights if src.lights is not None else r.light_matrix
    try:
        from helio_qt import __version__ as v
    except Exception:
        v = ""
    st = r.settings
    facts = Facts(title=src.name, subtitle=fx.get("subtitle", "Photometric stereo · reflectance "
                                                               "transformation imaging"),
                  frames=len(src.images) or r.n_images, lights=lights,
                  light_note=fx.get("light_note", ""), mm_per_px=src.mm_per_px or None,
                  z_mm_per_unit=src.z_mm_per_unit or None,
                  normal_engine=st.resolved_normal_engine(), depth_engine=st.resolved_depth_engine(),
                  photo_tone=st.resolved_photo_tone(), solve_seconds=fx.get("solve_seconds", 0.0),
                  drift_px=fx.get("drift_px"), flash_spread=fx.get("flash_spread"),
                  colour=fx.get("colour"),
                  creator=fx.get("creator", ""), institution=fx.get("institution", ""),
                  licence=fx.get("licence", ""), helio=f"Helio {v}".strip(),
                  notes=list(fx.get("notes", [])))
    score = None
    if lights is not None and len(lights):
        try:
            score = dome_coverage(np.asarray(lights, np.float64))
        except Exception:
            score = None
    base = os.path.join(out_dir, f"{file_stem(src.name)}_publication_plate")
    section = fx.get("section")
    t = _screen_turn(src)
    if t.any:                       # as the screen shows the object: pictures, lights, section
        from helio.application.viewing import TurnedData
        plain = draw
        # a light the plate picks on the turned page is turned back to draw with
        draw = lambda eng, overrides=None: t.image(np.asarray(plain(eng, t.light_back(overrides))))   # noqa: E731
        if section:
            section = tuple(t.point(float(p[0]), float(p[1]), D.w, D.h) for p in section)
        if facts.lights is not None:
            facts.lights = t.vectors(np.asarray(facts.lights, np.float64))
        D = TurnedData(D, t)
    res = plate(base, D=D, facts=facts, draw=draw, relief=fx.get("relief"),
                section=section, dome_score=score)
    return [res["pdf"], res["png"]]


def _screen_turn(src):
    """The screen's mirror and turn (viewing.ScreenTurn), for the reports."""
    from helio.application.viewing import ScreenTurn
    return ScreenTurn(src.screen(), src.turn)


def _pub_sheet(src, out_dir):
    """The publication sheet (helio.application.pub_sheet): the object and its
    views, its details, the section and measured lines, 3D pictures; JPEG and PDF."""
    from helio.application import pub_sheet as PS
    from helio.application.render import composite, render
    from helio.core.shading_engines import _ENGINE_META
    from helio.domain.registry import ENGINE_BLURBS, default_engine_settings
    from helio.domain.view_controls import from_angles, light_settings
    D, r = _need_data(src), src.result
    defaults = default_engine_settings()
    fx = dict(src.facts or {})
    opt = dict(fx.get("sheet") or {})
    layer = _layer(src)

    def draw(eng, overrides=None, mode=None):
        st = dict(src.engine_settings.get(eng) or defaults.get(eng)
                  or default_engine_settings((eng,))[eng])
        st.update(overrides or {})
        if (mode or layer) == src.screen().layer:
            # in the screen's own layer: drawn as the screen draws it (its colour
            # brightness and image adjustments too), the mirror and turn left to the sheet
            from helio.application.viewing import screen_image
            return screen_image(eng, D, st, src.screen(), transforms=False)
        img = render(eng, D, st, D.ph, D.pw, True)
        return composite(eng, img, D, st, *np.asarray(img).shape[:2], True, view_mode=mode or layer)

    side = []

    def side_light(eng):
        """A side light for a view still lit from overhead (PS.SIDE_LIT)."""
        st = src.engine_settings.get(eng) or defaults.get(eng) or {}
        if eng not in PS.SIDE_LIT or float(st.get("lz", 1.0)) < 0.98:
            return None
        if not side:
            from helio.application.plate_doc import best_raking, from_above
            L = src.lights if src.lights is not None else r.light_matrix
            side.append(from_above(best_raking(L, D.normals_proxy)[0]))
        return light_settings(*from_angles(side[0], PS.SIDE_LIT[eng]))

    def view(eng):
        fam = _ENGINE_META.get(eng, {}).get("family", "")
        # A relief or decomposition view shows its own reading (its shading),
        # not tinted by the colour layer, which made them all look alike.
        mode = None if fam in ("Relighting",) or eng in PS.COLOURED else "Shading"
        try:
            pic = draw(eng, side_light(eng), mode=mode)
        except Exception:
            return None
        if pic is None:
            return None
        return PS.View(eng, np.asarray(pic), ENGINE_BLURBS.get(eng, ""), fam)
    views = [v for v in (view(e) for e in (opt.get("views") or PS.DEFAULT_VIEWS)) if v is not None]
    dviews = [v for v in (view(e) for e in (opt.get("detail_views") or PS.DEFAULT_DETAIL_VIEWS))
              if v is not None]
    hero = view(opt.get("hero") or PS.HERO_VIEW) or (views[0] if views else None)
    meta = dict(src.meta or {})
    st_ = r.settings
    lights = src.lights if src.lights is not None else r.light_matrix
    mm = src.mm_per_px or None
    try:
        from helio_qt import __version__ as v_
    except Exception:
        v_ = ""
    q = opt.get("quality") or {}
    capture = [("Photographs", f"{len(src.images) or r.n_images}"),
               ("Size", f"{r.w:,} × {r.h:,} px"
                        + (f"  ({r.w * mm:.1f} × {r.h * mm:.1f} mm)" if mm else "")),
               ("Camera", " · ".join(x for x in (meta.get("camera", ""), meta.get("lens", "")) if x)),
               ("Lights", fx.get("light_note", "") or ("measured" if lights is not None else "")),
               ("Solved with", f"{st_.resolved_normal_engine()} · {st_.resolved_depth_engine()}"),
               ("Scale", f"{mm * 1000:.2f} µm per pixel" if mm else "not set"),
               ("How sure", f"{q['med']:.2f}° median, 95% under {q['p95']:.1f}°" if q else ""),
               ("Light spread", f"{q['cond']:.2f}  (2 or under is good)" if q.get("cond") else ""),
               ("Captured", " · ".join(x for x in (meta.get("capture_date", ""), meta.get("operator", ""))
                                       if x))]
    obj = [("Title", meta.get("title", "") or src.name),
           ("Institution", meta.get("institution", "") or fx.get("institution", "")),
           ("Collection", meta.get("collection", "")),
           ("Notes", meta.get("notes", "")),
           ("Reference", meta.get("doi", "")),
           ("Credit", fx.get("creator", "")),
           ("Licence", fx.get("licence", ""))]
    compass = None
    if opt.get("compass"):
        compass = {}
        for az in range(0, 360, 45):
            lx, ly, lz = from_angles(az, 25)
            try:
                # this view reads its light as a compass bearing and height
                # (view_controls.light_kind "compass"); lx, ly, lz alone left
                # all eight lit alike
                compass[az] = draw("PS Normal-Map Lit (Single)",
                                   dict(lx=lx, ly=ly, lz=lz, hs_azi=float(az), hs_alt=25.0),
                                   mode="Shading")
            except Exception:
                compass = None
                break
    import datetime
    if opt.get("hide_capture"):
        capture = []
    # the analyses ticked for the sheet (helio.application.sheet_tools), worked out
    # now, turned as the screen shows the object; then the tools' own results
    from helio.application import sheet_tools as ST
    from helio.domain.object_profiles import of as _profile
    tools, tn = [], _screen_turn(src)
    for key in [k for k in (opt.get("analyses") or []) if k in ST.ANALYSES]:
        try:
            res = ST.make(key, D, mm_per_px=mm or 0.0, z_mm_per_unit=src.z_mm_per_unit or 0.0,
                          kind=_profile(meta).key)
        except Exception:                               # noqa: BLE001  one never stops the sheet
            import logging
            logging.getLogger(__name__).warning("sheet analysis %s", key, exc_info=True)
            continue
        if res is not None:
            tools.append((res[0], tn.image(np.asarray(res[1]))))
    tools += list(opt.get("tools") or [])
    sheet = PS.Sheet(
        title=str(opt.get("title") or "").strip() or meta.get("title", "") or src.name,
        full_size=(r.w, r.h),
        albedo=np.asarray(D.albedo_proxy), views=views, hero=hero,
        subtitle=str(opt.get("subtitle") or "").strip() or fx.get("subtitle", "")
        or " · ".join(x for x in (meta.get("collection", ""), meta.get("institution", "")) if x),
        object_rows=obj, capture_rows=capture,
        score=None if opt.get("hide_score") else opt.get("score"), lights=lights,
        mm_per_px=mm, section=fx.get("section_prof"), sections=_sheet_sections(fx, opt),
        shots_heading=opt.get("shots_heading") or "In 3D",
        measures=list(opt.get("measures") or []),
        notes=list(opt.get("notes") or []) if opt.get("notes_on", True) else [],
        details=list(opt.get("details") or []), detail_views=dviews,
        shots=list(opt.get("shots") or []), tools=tools, compass=compass,
        normals=np.asarray(D.normals_proxy) if compass else None,
        paper=opt.get("paper") if opt.get("paper") in ("A4", "Letter") else "A3",
        png=bool(opt.get("png")), quality="screen" if opt.get("quality") == "screen" else "print",
        turn=_screen_turn(src),
        footer=[f"Made with Helio {v_}".strip() + f" · {datetime.date.today():%d %B %Y}",
                "Photometric stereo: the surface's direction at every point, from photographs "
                "lit from many sides.",
                "Views cut to the object; grey views' contrast evened out over it, colour as "
                "on screen."])
    base = os.path.join(out_dir, f"{file_stem(src.name)}_publication_sheet")
    return PS.make(sheet, base)


def _sheet_sections(fx: dict, opt: dict) -> list[dict]:
    """The sections added to the sheet.  The one drawn now goes on only when
    none was added: it used to be added as well whenever it differed at all
    from the kept ones, so a line nudged after Add to sheet became a second,
    all but identical section nobody had asked for."""
    secs = [dict(p) for p in (opt.get("sections") or [])]
    live = fx.get("section_prof")
    if live and not secs:
        secs.append(live)
    return secs


REPORT_EXPORTS = {"sheet": _sheet, "figures": _figures, "condition": _condition, "rose": _rose,
                  "plate_a3": _plate_a3, "pub_sheet": _pub_sheet}


def _freq(src, out_dir):
    from helio.application.export_images import frequency_maps
    r = src.result
    return frequency_maps(r.depth_vis_p, _folder(out_dir, f"{file_stem(src.name)}_frequency_maps"),
                          normals=r.normals_full, depth_poisson=r.depth_poisson,
                          hp_sigma=float(src.shared.get("hp_sigma", 0.0)),
                          mm_per_px=src.mm_per_px or None, obj_id=src.name)


IMAGE_EXPORTS = {"freq_maps": _freq, "all_views": _all_views, "normal_map": _normal_map, "deepzoom": _deepzoom,
                 "scalebar": _scalebar, "heatmap": _heatmap, "pbr": _pbr, "sds": _sds,
                 "pca": _pca, "hsh": _hsh, "uncertainty": _uncertainty}


def current_view(src: Source, out_dir: str) -> str:
    """The Inspect view at full resolution, as the classic "Current Visual (TIFF)" draws it:
    the same layer, display transforms and settings as the screen
    (helio.application.viewing.screen_image)."""
    import cv2
    from helio.application.viewing import screen_image, turn_image
    r, stem = src.result, file_stem(src.name)
    if src.data is None:
        raise X.NothingToExport("Process the images first")
    d = src.screen()
    if src.view == "Depth map":
        # As on screen: 8-bit grey (the 32-bit heights are Publish's height
        # formats), light high, inverted when the screen is.
        from helio.application.viewing import depth_layer
        g = depth_layer(src.data, use_proxy=False)
        if d.depth_invert:
            g = 255 - g
        g = _screen_turn(src).image(g)                   # mirrored and turned as on screen
        path = os.path.join(out_dir, f"{stem}_depth.tif")
        cv2.imwrite(path, g, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        return path
    rgb = turn_image(screen_image(src.view, src.data, src.view_settings, d, use_proxy=False),
                     src.turn)                            # turned as Inspect shows it
    path = os.path.join(out_dir, f"{stem}_{file_stem(src.view)[:24]}.tif")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_TIFF_COMPRESSION, 1])
    return path


def processing_record(src: Source) -> dict:
    """The classic record's fields, filled from what this app knows.

    The stand profile, points of interest and occlusion fabric are classic-only
    tools: their fields are present and empty, so a reader sees the same shape."""
    import hashlib
    import platform
    import time
    from helio_qt import __version__
    r, ps = src.result, src.result.settings
    L = r.light_matrix
    return {
        "helio_version": __version__,
        "schema_version": "1.0",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "object": {"id": src.name, "title": src.meta.get("title", ""),
                   "institution": src.meta.get("institution", ""),
                   "collection": src.meta.get("collection", "")},
        "capture": {
            "operator": src.meta.get("operator", ""), "date": src.meta.get("capture_date", ""),
            "camera": src.meta.get("camera", ""), "lens": src.meta.get("lens", ""),
            "sensor": src.meta.get("sensor", ""), "pixel_pitch_um": src.meta.get("pixel_pitch", ""),
            "lighting": src.meta.get("lights", src.light_source),
            "stand": src.meta.get("stand", ""),
            "working_dist_mm": src.meta.get("working_dist", ""),
            "scale_mmpx": src.mm_per_px or "",
            "image_count": len(src.images),
            "image_paths": [os.path.basename(p) for p in src.images],
            "image_resolution": {"width": int(r.w), "height": int(r.h)},
        },
        "calibration": {
            "stand_profile": src.poi_source, "poi_loaded": bool(src.poi),
            "poi_source": src.poi_source,
            "poi_coords": {k: list(v) for k, v in src.poi.items()},
            "light_matrix_sha256": hashlib.sha256(np.asarray(L).tobytes()).hexdigest()[:16]
            if L is not None else "",
            "light_matrix_shape": list(np.asarray(L).shape) if L is not None else [],
            "occlusion_fabric": "",
        },
        "pre_processing": {
            "flatten_illumination": ps.flatten_illumination,
            "multi_light_normalise": ps.multi_norm,
            "near_field_correction": bool(ps.near_field),
            "dome_radius_mm": float(ps.dome_radius_mm),
            "photo_tone": ps.resolved_photo_tone(),
            "tone_curve": getattr(r, "tone_curve", ""),
            "lights_at_own_positions": int(getattr(r, "lights_placed", 0)),
            "flat_field_correction": ps.flat_correction,
            "flat_correction_strength": float(ps.flat_corr_strength),
            "flat_ref_image_count": 0,
        },
        "normal_computation": {"solver": ps.normal_engine, "shadow_threshold": ps.shadow_thresh,
                               "depth_smooth_passes": 0},
        "depth_integration": {"engine": ps.depth_engine, "parabolic_correction": ps.parabolic_corr,
                              "object_alone": bool(getattr(ps, "mask_surround", False)),
                              "hp_normalisation_sigma": 25, "hp_percentile_clip": "1–99"},
        "output": {"normal_map_dtype": "float32", "depth_poisson_dtype": "float32",
                   "ptm_coefficients": 6, "hsh_coefficients": 9},
        "software": {"name": "Helio", "version": __version__, "author": "Scott Maloney",
                     "platform": platform.system(), "python": platform.python_version()},
        "references": X.PROVENANCE_REFERENCES,
    }


def plate(src: Source, out_dir: str) -> str:
    """Five views on one plate, drawn as Inspect draws them."""
    from helio.application.render import composite, render
    from helio.domain.registry import RENDER_ENGINES
    D, r = src.data, src.result
    if D is None:
        raise X.NothingToExport("Process the images first")
    pick = [e for e in X.PLATE_ENGINES if e in RENDER_ENGINES][:6]
    lx, ly, lz = (src.view_settings.get(k, d) for k, d in (("lx", 0.0), ("ly", 0.0), ("lz", 1.0)))
    settings = getattr(D, "engine_settings", {}) or {}

    layer = "Shading" if src.screen().layer == "Shading" else "Combined Colour"

    t = _screen_turn(src)

    def draw(eng):
        st = dict(settings.get(eng, src.view_settings), lx=lx, ly=ly, lz=lz)
        img = render(eng, D, st, D.ph, D.pw, True)
        return t.image(np.asarray(composite(eng, img, D, st, D.ph, D.pw, True, view_mode=layer)))
    scaled = src.mm_per_px * (r.w / float(D.pw)) if src.mm_per_px else None
    path, _ = X.plate_figure_export(draw, pick, os.path.join(out_dir, f"{file_stem(src.name)}_plate.png"),
                                    title=f"{file_stem(src.name)}  —  engine plate",
                                    mm_per_px_preview=scaled)
    return path


def sweep_gif(src: Source, out_dir: str) -> str:
    """The light once round, in the current view, at preview resolution."""
    D = src.data
    if D is None:
        raise X.NothingToExport("Process the images first")
    return X.light_sweep_gif(_sweep_draw(src),
                             os.path.join(out_dir, f"{file_stem(src.name)}_light_sweep.gif"))


def _sweep_draw(src: Source, full: bool = False, long_side: int = 0, view: str = "", layer: str = ""):
    """draw(lx, ly, lz): the current view lit from there, as on screen (or at
    full resolution), fitted to *long_side* when given.  *view* and *layer*
    draw another view (an engine, in "Combined Colour" or "Shading") with its
    own settings, mirrored and turned as the screen is."""
    import copy
    import cv2
    from helio.application.viewing import screen_image, turn_image
    D = src.data
    d = copy.deepcopy(src.screen())
    if view:
        from helio.domain.registry import default_engine_settings
        settings = dict(src.engine_settings.get(view) or default_engine_settings((view,)).get(view) or {})
        d.layer = layer or d.layer
    else:
        view = src.view if src.view != "Depth map" else "Standard PBR"
        settings = dict(src.view_settings)
    if d.layer == "Depth Map (Greyscale)":
        d.layer = "Combined Colour"

    def draw(lx, ly, lz):
        from helio.domain.view_controls import light_settings
        img = screen_image(view, D, dict(settings, **light_settings(lx, ly, lz)), d,
                           use_proxy=not full)
        img = turn_image(img, src.turn)                   # turned as Inspect shows it
        h, w = img.shape[:2]
        if long_side and max(w, h) > long_side:
            k = long_side / float(max(w, h))
            img = cv2.resize(img, (max(2, int(w * k)), max(2, int(h * k))),
                             interpolation=cv2.INTER_AREA)
        return img
    return draw


def sweep_views(src: Source, views) -> list[tuple[str, str, str]]:
    """(engine, layer, label) for the light sweep's *views*: "" first for the view
    as on screen, then up to two more (engine, layer) pairs."""
    from helio.application.multiview import label_of
    out = []
    for eng, layer in views or [("", "")]:
        if eng:
            out.append((eng, layer, label_of(eng, layer)))
        else:
            e = src.view if src.view != "Depth map" else "Standard PBR"
            lay = src.screen().layer
            out.append(("", "", label_of(e, lay if lay != "Depth Map (Greyscale)" else "Combined Colour")))
    return out


def light_sweep(src: Source, path: str, *, elevation: float = 37.0, seconds: float = 4.0,
                fps: int = 20, start_az: float = 0.0, reverse: bool = False,
                full: bool = False, long_side: int = 0, progress=None, max_mb: float = 0.0,
                repeat: int = 1, views=None, layout: str = "side", labels: bool = True) -> str:
    """The light once round the current view at *elevation* degrees, starting
    from the compass bearing *start_az*, as a GIF or MP4 by *path*'s name.
    *full* draws every frame at full resolution (slow on a large capture);
    *max_mb* keeps the file under that many megabytes (viewer3d.write_video).

    *views*, [(engine, layer), ...] with ("", "") for the view on screen, puts
    two or three views in the one sweep (helio.application.multiview): *layout*
    "side" by side, every frame, or "after" one another, a full turn each,
    *seconds* apiece; *labels* names each view in its corner."""
    from helio.application import multiview as MV
    from helio.application.viewer3d import write_video
    from helio.domain.view_controls import from_angles
    if src.data is None:
        raise X.NothingToExport("Process the images first")
    D = src.data
    if full and long_side and max(int(D.pw or 0), int(D.ph or 0)) >= long_side:
        full = False                     # the preview already holds that size: no need to wait
    vs = sweep_views(src, views)[:MV.MAX_VIEWS]
    several = len(vs) > 1
    each = long_side
    if several and layout == "side" and long_side:   # the whole frame, not each panel, that long
        each = 0
    draws = [_sweep_draw(src, full, each, e, l) for e, l, _t in vs]
    names = [t for _e, _l, t in vs] if several and labels else []
    n = max(8, int(round(seconds * fps)))
    frames = []

    def fit(img):
        if long_side and max(img.shape[:2]) > long_side:
            import cv2
            k = long_side / float(max(img.shape[:2]))
            img = cv2.resize(img, (max(2, int(img.shape[1] * k)) // 2 * 2,
                                   max(2, int(img.shape[0] * k)) // 2 * 2), interpolation=cv2.INTER_AREA)
        return img
    total = n * (1 if layout == "side" else len(draws))
    for v in range(1 if layout == "side" else len(draws)):
        for i in range(n):
            t = (n - i) % n if reverse else i
            L = from_angles((start_az + 360.0 * t / n) % 360.0, elevation)
            if layout == "side" and several:
                frames.append(fit(MV.tile([dr(*L) for dr in draws], names)))
            else:
                img = draws[v](*L)
                frames.append(MV.caption(img, names[v]) if names else img)
            if progress:
                progress(len(frames), total)
    if several and layout != "side":                  # one size for the whole video
        size = (max(f.shape[1] for f in frames) // 2 * 2, max(f.shape[0] for f in frames) // 2 * 2)
        frames = [MV.even(f, size) for f in frames]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return write_video(frames, path, fps, max_mb=max_mb, repeat=repeat)


def job(keys: list[str], src: Source, out_dir: str):
    """A JobManager function: ``(key, files, problem)`` per item, in order."""
    names = {f.key: f.name for f in all_formats()}

    def run(ctx):
        os.makedirs(out_dir, exist_ok=True)
        results = []
        n = len(keys)
        for i, k in enumerate(keys):
            ctx.check()
            ctx.progress(f"Writing {names.get(k, k)}", i / n, k)
            try:
                results.append((k, export_one(k, src, out_dir), ""))
            except X.NothingToExport as e:
                results.append((k, [], str(e)))
            except Exception as e:                      # one item's problem is not the queue's
                _log.exception("export %s failed", k)
                results.append((k, [], f"{type(e).__name__}: {e}"))
            ctx.progress(None, (i + 1) / n, k)
        if any(files for _k, files, _p in results) and keys != ["deposit"]:   # a deposit has its own
            try:
                _paradata(src, results, out_dir)
            except Exception:                           # the record is never why an export failed
                _log.warning("paradata - skipping", exc_info=True)
        return results
    return run
