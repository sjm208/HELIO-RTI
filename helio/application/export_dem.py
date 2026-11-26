"""The GIS DEM package, GUI-free (EXR-012).

The classic export's writing, moved so both apps write the same package:
levelled Poisson and Frankot–Chellappa DEMs at one unit per pixel (the
physical scale in HELIO_ tags, so QGIS's RVT radii stay in cells), albedo,
normals, the uncertainty raster, the .lp, provenance, a .prj, a README and
a checksummed manifest.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

from helio.application.exports import NothingToExport
from helio.io.exporters import write_geotiff

_log = logging.getLogger(__name__)
try:
    import rasterio  # noqa: F401
    HAS_RASTERIO = True
except Exception:
    HAS_RASTERIO = False

LEVEL_DEFAULTS = {"level": True, "warp": False, "warp_frac": 0.15}


def dem_package(D, out_dir: str, obj_name: str, *, mm_per_px=None, z_mm_per_unit=None,
                z_exag: float = 200.0, invert: bool = False, crop=None, normal_engine: str = "",
                lights=None, light_kind: str = "", provenance_app=None, res_choice: str = "",
                crs_choice: str = "", level: bool = True, warp: bool = False,
                warp_frac: float = 0.15, object_type: str = "",
                sections=(), annotations=()) -> dict:
    """Write the package into *out_dir*; returns its key figures."""
    _do_level, _do_warp = bool(level), bool(warp)
    _warp_frac = max(0.05, min(0.40, float(warp_frac)))
    obj_id = (obj_name or "helio_dem").replace(" ", "_")

    # ── 1. Pixel size in METRES ───────────────────────────────────────
    # _mm_per_px is the authoritative source (set by calculate_ppi).
    # Fall back to scale_mm / drawn_scale_px if available.
    # Final fallback: 1.0 mm/px → 0.001 m/px (unpleasant but safe).
    xy_measured = bool(mm_per_px) and float(mm_per_px) > 0
    if not xy_measured:
        mm_per_px = 1.0   # fallback: 1 mm/px, and the README says it was assumed

    res_m = mm_per_px / 1000.0   # kept for metadata/README display only — NOT used in transform

    # ── 2. Z scale (in pixel-units; pixel size = 1.0) ─────────────────
    z_mpu_mm = z_mm_per_unit
    z_exag = float(z_exag)

    try:
        _z_valid = z_mpu_mm is not None and float(z_mpu_mm) > 0
    except (TypeError, ValueError):
        _z_valid = False

    if _z_valid:
        # Calibrated: true heights (pixels of height, helio.application.heights)
        # in pixel-units to preserve the real aspect ratio: z_mpu_mm mm per
        # pixel of height ÷ mm_per_px mm per pixel.
        z_scale_units = float(z_mpu_mm) / mm_per_px
        z_scale_m     = float(z_mpu_mm) / 1000.0   # metres — README display only
        z_calibrated  = True
        z_note = (f"Z calibrated: {float(z_mpu_mm):.6f} mm per pixel of height (true heights) "
                  f"→ {z_scale_units:.4f} pixel-units stored in GeoTIFF "
                  f"(real XY/Z aspect ratio preserved, pixel size = 1.0).")
    else:
        # Uncalibrated: Z = depth × z_exag so the maximum stored value
        # equals z_exag pixel-units.  The ratio z_exag/1.0 gives the
        # intended vertical exaggeration directly — no unit conversion.
        z_scale_units = z_exag
        z_scale_m     = res_m * z_exag   # metres — README display only
        z_calibrated  = False
        z_note = (f"Z NOT calibrated. Stored as depth [0,1] × {z_exag:.0f} pixel-units "
                  f"(vertical exaggeration = {z_exag:.0f}×, pixel size = 1.0). "
                  f"Use Metrology → Steps 3 & 4 for true Z calibration.")

    # ── 3. Build depth arrays ────────────────────────────────────────
    from helio.core import (subtract_parabolic_baseline,
                            fast_large_sigma_blur)
    inv_active = bool(invert)

    # Apply active crop (respects Reprocess Crop Region)
    _gdx1, _gdy1, _gdx2, _gdy2 = crop if crop else (0, 0, D.w, D.h)
    def _crop_d(a):
        return a[_gdy1:_gdy2, _gdx1:_gdx2] if a is not None else None

    def _get_d(key):
        """A clean cropped float32 depth array: [0,1], or with a height scale the true
        heights in pixels of height (not stretched: their range is the measurement)."""
        arr = None
        if _z_valid:
            from helio.application.heights import height_px
            arr = height_px(D)
            if arr is not None:
                a = _crop_d(arr).astype(np.float32)
                if _do_level:
                    a = subtract_parabolic_baseline(a)
                if _do_warp:
                    a = a - fast_large_sigma_blur(a, max(1.0, min(a.shape[:2]) * _warp_frac))
                a = a - float(a.min())
                return (float(a.max()) - a) if inv_active else a
        if key == 'poisson':
            arr = getattr(D, 'depth_vis_p', None)
        elif key == 'fc':
            fc = getattr(D, 'depth_vis_fc', None)
            arr = fc if fc is not None else getattr(D, 'depth_vis_p', None)
        if arr is None: return None
        a = _crop_d(arr).astype(np.float32)
        # Level *after* the crop and *before* the normalise.  Cropping
        # first means the fit sees only the region actually exported,
        # so a small crop off a large plate is levelled against its own
        # slope rather than against the whole sheet's.  Normalising
        # first would let the tilt set the range and squash the real
        # relief into a fraction of it.
        if _do_level:
            a = subtract_parabolic_baseline(a)
        if _do_warp:
            a = a - fast_large_sigma_blur(
                a, max(1.0, min(a.shape[:2]) * _warp_frac))
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo + 1e-9)
        if inv_active: a = 1.0 - a
        return a

    d_poisson = _get_d('poisson')
    d_fc      = _get_d('fc')
    if d_poisson is None:
        raise NothingToExport("Depth map not available.")

    # Scale Z to pixel-units (pixel size = 1.0 in transform)
    arr_poisson = (d_poisson * z_scale_units).astype(np.float32)
    arr_fc      = ((d_fc if d_fc is not None else d_poisson)
                   * z_scale_units).astype(np.float32)
    h, w = arr_poisson.shape   # now reflects crop dimensions

    # ── 4. Resolution cap (optional, for faster RVT processing) ──────
    res_str = res_choice or "Full resolution"
    _cap = None
    if '3000' in res_str: _cap = 3000
    elif '2000' in res_str: _cap = 2000
    elif '1500' in res_str: _cap = 1500

    if _cap and max(h, w) > _cap:
        scale_f  = _cap / max(h, w)
        h2 = max(4, int(h * scale_f))
        w2 = max(4, int(w * scale_f))
        arr_poisson = cv2.resize(arr_poisson, (w2, h2),
                                 interpolation=cv2.INTER_AREA)
        arr_fc      = cv2.resize(arr_fc,      (w2, h2),
                                 interpolation=cv2.INTER_AREA)
        # Update metadata to reflect the coarser physical resolution
        # after downsampling.  The transform pixel size stays 1.0.
        mm_per_px = mm_per_px * (h / h2)
        res_m     = mm_per_px / 1000.0   # keep in sync for README
        h, w      = h2, w2

    NODATA = -9999.0

    # ── 5. GeoTIFF transform ─────────────────────────────────────────
    # Built inside write_geotiff(): origin top-left, pixel size 1.0
    # (one unit per pixel).  Physical scale lives in the HELIO_ tags.
    # Using the real sub-mm value would make QGIS's RVT compute search
    # neighbourhoods of ~100,000+ cells per pixel (10 m radius ÷
    # 0.000085 m/px), which never finish.
    # This used to import rasterio at module scope here and build an
    # unused transform, so the whole export raised ImportError when
    # rasterio — an optional dependency — was not installed.

    # ── 6. CRS selection ─────────────────────────────────────────────
    _crs_str = crs_choice or "Local mm (metric, no real location)"
    if _crs_str.startswith("Local mm"):
        _crs     = "EPSG:32630"
        _crs_tag = "EPSG:32630 (local metric placeholder)"
    else:
        import re as _re
        _m = _re.search(r'EPSG:(\d+)', _crs_str)
        _epsg = int(_m.group(1)) if _m else 32630
        _crs  = f"EPSG:{_epsg}"
        _crs_tag = _crs

    # ── 7. Write GeoTIFFs ─────────────────────────────────────────────
    if _do_level and _do_warp:
        _lvl_tag = f"plane+bowl (poly22) then high-pass at {_warp_frac*100:.0f}% of short edge"
    elif _do_level:
        _lvl_tag = "plane+bowl (poly22)"
    elif _do_warp:
        _lvl_tag = f"high-pass at {_warp_frac*100:.0f}% of short edge"
    else:
        _lvl_tag = "none — raw solved depth, may carry overall tilt"
    from helio.io.exporters import save_dem_package
    _common_tags = dict(
        HELIO_OBJECT_ID    = obj_id,
        HELIO_XY_RES_MM    = f"{mm_per_px:.6f}",
        HELIO_XY_RES_M     = f"{res_m:.9f}",
        HELIO_Z_SCALE_M    = f"{z_scale_m:.9f}",
        HELIO_Z_CALIBRATED = "yes" if z_calibrated else "no",
        HELIO_CRS          = _crs_tag,
        HELIO_NODATA       = str(NODATA),
        HELIO_IMAGES       = str(len(getattr(D, 'plate_paths', []))),
        HELIO_SOLVER       = normal_engine,
        HELIO_LEVELLED     = _lvl_tag,
    )
    save_dem_package(
        out_dir, obj_id,
        arr_poisson, arr_fc,
        albedo=D.albedo_rgb_full if getattr(D, 'albedo_rgb_full', None) is not None else None,
        normals=D.normals_full if getattr(D, 'normals_full', None) is not None else None,
        crs=_crs, tags=_common_tags, mm_per_px=mm_per_px,
    )

    # What was marked on the object, over the height map
    try:
        from helio.application.export_vectors import write as write_vectors
        write_vectors(out_dir, obj_id, sections=sections, annotations=annotations, h=arr_poisson.shape[0],
                      k=arr_poisson.shape[0] / max(_gdy2 - _gdy1, 1), crop=(_gdx1, _gdy1),
                      mm_per_px=mm_per_px if xy_measured else None)
    except Exception:
        _log.exception("GIS DEM — sections and annotations")

    # ── 7b. Uncertainty, light directions and provenance ─────────────
    # A height map exported on its own is a picture: nothing in the
    # file distinguishes twelve well-spread lights on matte parchment
    # from four clustered ones on gilding.  Optical surface metrology
    # expects a measurement to arrive with a statement of its
    # uncertainty, and HELIO already computes the raw material - the
    # RMS residual of the per-pixel Lambertian fit, which is exactly
    # where shadowing, specularity and thin light coverage announce
    # themselves.  It was a view you could look at and close; now it
    # travels with the data.
    _resid = getattr(D, 'qc_resid_full', None)
    if _resid is not None:
        try:
            _r = _crop_d(np.asarray(_resid, dtype=np.float32))
            if _r is not None and _r.shape[:2] != arr_poisson.shape[:2]:
                _r = cv2.resize(_r, (arr_poisson.shape[1], arr_poisson.shape[0]),
                                interpolation=cv2.INTER_AREA)
            # Stored as uint8; write it as 0-1 so the number in the
            # file means something on its own rather than depending on
            # a scale the reader has to know about.
            if _r.dtype != np.float32 or _r.max() > 1.5:
                _r = _r.astype(np.float32) / 255.0
            write_geotiff(os.path.join(out_dir, f"{obj_id}_uncertainty.tif"),
                          _r.astype(np.float32), crs=_crs,
                          tags=dict(_common_tags,
                                    HELIO_LAYER="uncertainty",
                                    HELIO_UNITS="0-1 normalised RMS residual",
                                    HELIO_UNCERTAINTY="RMS residual of the "
                                    "per-pixel Lambertian fit; high = the "
                                    "recovered normal is least believable"))
        except Exception:
            _log.exception("GIS DEM — uncertainty raster")
    else:
        _log.info("GIS DEM: no residual map (uncalibrated solve) — "
                 "uncertainty raster not written")

    # Where the metal-aware solve re-solved the normals.  Height over
    # gilding is the least certain part of the map; Helio Relief loads
    # this beside the heights so the reader can see where that is.
    _metal = getattr(D, 'metal_mask_full', None)
    if _metal is not None:
        try:
            _m = _crop_d(np.asarray(_metal))
            if _m is not None and _m.shape[:2] != arr_poisson.shape[:2]:
                _m = cv2.resize(_m.astype(np.uint8), (arr_poisson.shape[1], arr_poisson.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            write_geotiff(os.path.join(out_dir, f"{obj_id}_metal_mask.tif"),
                          (np.asarray(_m) > 0).astype(np.uint8), crs=_crs,
                          tags=dict(_common_tags,
                                    HELIO_LAYER="metal_mask",
                                    HELIO_UNITS="1 = normals re-solved as metal (gilding, clasps), 0 = not"))
        except Exception:
            _log.exception("GIS DEM — metal mask")

    # Light directions, in the format every other RTI tool reads.  A
    # partner lab could not process our files for exactly this reason:
    # nothing in them said which image was lit from where.
    try:
        # Read through _light_matrix, not data.light_vecs directly:
        # that attribute belongs to the dome editor, so a capture
        # calibrated with a chrome sphere silently shipped a DEM
        # package with no .lp in it - the exact failure this block was
        # written to prevent.
        _lv, _lv_src = (lights, light_kind)
        # *lights* may be a list or an (N, 3) array: test its length, not its
        # truth (an array has none, and the sidecar was silently skipped).
        if _lv is not None and len(_lv) and _lv_src != "estimated":
            _names = [os.path.basename(x) for x in
                      getattr(D, 'plate_paths', [])]
            with open(os.path.join(out_dir, f"{obj_id}.lp"), "w") as _lf:
                _lf.write(f"{len(_lv)}\n")
                for _i, (_lx, _ly, _lz) in enumerate(_lv):
                    _nm = _names[_i] if _i < len(_names) else f"image_{_i:03d}"
                    _lf.write(f"{_nm} {_lx:.6f} {_ly:.6f} {_lz:.6f}\n")
        elif _lv_src == "estimated":
            _log.info("GIS DEM: the lights were found from the photographs (no ball), "
                      "not measured: .lp sidecar not written")
    except Exception:
        _log.exception("GIS DEM — .lp sidecar")

    try:
        from helio.io.provenance import write_provenance
        if provenance_app is not None:
            write_provenance(provenance_app, out_dir)
    except Exception:
        _log.exception("GIS DEM — provenance")

    # ── 8. PRJ sidecar ───────────────────────────────────────────────
    with open(os.path.join(out_dir, f"{obj_id}_DEM.prj"), 'w') as pf:
        pf.write('PROJCS["WGS 84 / UTM zone 30N",GEOGCS["WGS 84",DATUM["WGS_1984",'                         'SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],'                         'UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'                         'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",-3],'                         'PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],'                         'PARAMETER["false_northing",0],UNIT["metre",1]]')

    # ── 9. README ────────────────────────────────────────────────────
    # The first block is what Helio Relief (the QGIS plugin) reads
    # (its helio.parse_readme): keep these labels as they are.
    _n_imgs = len(lights) if lights is not None else 0
    _xy_line = (f"{mm_per_px:.6g} mm/pixel (measured with the ruler)" if xy_measured else
                f"{mm_per_px:.6g} mm/pixel (estimated: no scale was set, 1 mm per pixel is assumed)")
    _readme = f"""Helio — GIS DEM Package
Object ID     : {obj_id}
Object type   : {object_type or 'not given'}
XY pixel size : {_xy_line}
Z values      : {'mm (calibrated)' if z_calibrated else 'relative units (uncalibrated)'}
Pixels        : {w} × {h}
{f'Images captured : {_n_imgs}' if _n_imgs else ''}
Normal map    : Nx, Ny, Nz as 3-band float32, in OpenGL convention (+Y towards the top of the image)

COORDINATE SYSTEM
  Pixel size    : 1.0 (one unit per pixel — deliberate, see note below)
  Physical size : {f'{mm_per_px:.6f} mm/pixel  (stored in HELIO_ tags)' if xy_measured else 'not measured'}
  Image extent  : {w} × {h} pixel-units{f'  ({w * mm_per_px / 10:.1f} cm × {h * mm_per_px / 10:.1f} cm)' if xy_measured else ''}
  Z scale       : {f'{z_scale_units:.4f} pixel-units per pixel of height ({float(z_mpu_mm):.6f} mm per pixel of height)' if z_calibrated else f'{z_scale_units:.4f} pixel-units per unit of relative depth (exaggerated for viewing; not millimetres)'}
  Z calibrated  : {'YES — values reflect real physical depth' if z_calibrated else 'NO — relative depth, exaggerated for visualisation'}
  CRS           : {_crs_tag}
  Levelling     : {_lvl_tag}
  Uncertainty   : {'included — see _uncertainty.tif' if getattr(D,'qc_resid_full',None) is not None else 'not available (uncalibrated solve)'}
  Metal         : {'included — see _metal_mask.tif (normals re-solved over gilding / metal)' if getattr(D,'metal_mask_full',None) is not None else 'not used'}

WHY PIXEL SIZE = 1.0
  QGIS's Relief Visualisation Toolbox converts its search-radius setting from
  map-units into cell counts by dividing by the pixel size.  At the real
  physical resolution (~0.000085 m/px for a 300 dpi scan), a default 10-unit
  radius becomes ~117,000 cells per pixel — RVT would never finish, even on
  a small 30 MB file.  Using pixel size = 1.0 means a 10-unit radius = 10
  cells, which processes in seconds.  The physical scale is not lost — it is
  stored in the HELIO_XY_RES_MM and HELIO_XY_RES_M tags.

LOADING IN QGIS
  1. Drag the .tif files into QGIS
  2. Right-click layer → Zoom to Layer
  3. Layer Styling → Hillshade, Z Factor = 1.0  (detail should be visible)
  4. For Relief Visualisation Toolbox (RVT):
       - Search radius = 10–50 pixel-units  (this is now cells, as intended)
       - These settings work for any file size

INTERPRETING Z VALUES
  Z = 0.0  → lowest point in the scan
  Z = {z_scale_units:.2f}  → highest point{f' ({float(z_mpu_mm):.4f} mm)' if z_calibrated else ' (relative: not a measurement)'}
  {'Values reflect real physical depth — calibrated against known Z reference.' if z_calibrated
   else 'Values are relative only — not calibrated to real physical units.'}

FILES IN THIS PACKAGE
  *_DEM_Poisson.tif          — smooth depth map (good for topographic analysis)
  *_DEM_FrankotChellappa.tif — detail-preserving depth (good for inscription analysis)
  *_Albedo_RGB.tif           — photometric albedo (use as texture overlay in QGIS 3D)
  *_NormalMap.tif            — surface normals (Nx, Ny, Nz as 3-band float32)
  *_uncertainty.tif          — 0-1 fit residual: high = least believable (calibrated solves)
  *_sections.geojson         — the sections kept in Helio, as lines (A–B ...), in the height map's units
  *_annotations.geojson      — the annotations made in Helio, in the height map's units
  *_metal_mask.tif           — 1 where the normals were re-solved as metal (metal-aware solve only)
  *.lp                       — which photograph was lit from where (measured lights only)
  *.prj                      — CRS definition sidecar
  *.tfw                      — world file (pixel size/origin), written for every .tif
  README_GIS.txt             — this file

NOTES
  {z_note}
  Rasterio {'available — full GeoTIFF metadata embedded (plus .tfw world files for redundancy)' if HAS_RASTERIO else 'NOT available — georeferencing carried entirely by the .tfw world files'}
"""
    with open(os.path.join(out_dir, "README_GIS.txt"), "w",
              encoding="utf-8") as rf:
        rf.write(_readme)

    inv_note = " (Z-flip active)" if inv_active else ""
    # Written last, so it can hash everything else in the folder.
    try:
        from helio.io.manifest import write_manifest
        write_manifest(out_dir, object_id=obj_id,
                       note="GIS DEM package — see README for layer meanings")
    except Exception:
        _log.exception("GIS DEM — manifest")

    return {"out_dir": out_dir, "mm_per_px": mm_per_px, "w": w, "h": h,
            "z_scale_units": z_scale_units, "z_scale_m": z_scale_m, "z_calibrated": z_calibrated,
            "inv_note": inv_note}
