"""
Recognise a Helio export folder.

Helio writes two kinds of package:
- the GIS DEM export: README_GIS.txt, <id>_DEM_Poisson.tif and <id>_DEM_FrankotChellappa.tif (32-bit float depth,
  two integrations of the same normals), a colour image (<id>_Albedo_RGB.tif, or a 16-bit albedo), the normal map
  (<id>_NormalMap.tif), the fit residual (<id>_uncertainty.tif), where the normals were re-solved as metal
  (<id>_metal_mask.tif, only when that option ran), a .lp, .prj / .tfw;
- the MorphoSource package: README.txt, <id>_depth.tif, <id>_albedo.jpg, <id>_normals.png (R = X, G = Y, B = Z,
  OpenGL), roughness / specular maps, a GLB mesh.
`scan` finds the height maps, colour image and normal map and reads the README's object id, pixel size, units and
calibration notes. No Qt, no GDAL - just files and text.
"""

import os
import re

RASTER = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def parse_readme(text):
    """What a HELIO README says: object_id, xy_size + xy_unit (+ xy_estimated), z_unit, z_calibrated,
    images, pixels (w, h), normals_convention."""
    info = {}
    m = re.search(r"Object type\s*:\s*(\S+)", text)
    if m:
        info["object_type"] = m.group(1)
    m = re.search(r"Object ID\s*:\s*(\S+)", text)
    if m:
        info["object_id"] = m.group(1)
    m = re.search(r"XY pixel size\s*:\s*([0-9.eE+-]+)\s*(mm|µm|um|cm|m)\s*/\s*pix\w*([^\n]*)", text, re.I)
    if m:
        info["xy_size"] = float(m.group(1))
        info["xy_unit"] = {"um": "µm"}.get(m.group(2).lower(), m.group(2).lower() if m.group(2) != "µm" else "µm")
        info["xy_estimated"] = "estimat" in m.group(3).lower()
    m = re.search(r"Z values\s*:\s*([^\n]*)", text)
    if m:
        line = m.group(1).lower()
        info["z_unit"] = "mm" if "mm" in line else None
        info["z_calibrated"] = "calibrated" in line and "uncalibrated" not in line
    m = re.search(r"Physical\s*:\s*([^\n]*)", text)
    if m and "uncalibrated" in m.group(1).lower():
        info["z_calibrated"] = False
    if re.search(r"relative units", text, re.I) and "z_unit" not in info:
        info["z_unit"] = None
        info["z_calibrated"] = False
    m = re.search(r"Images(?: captured)?\s*:\s*(\d+)", text)
    if m:
        info["images"] = int(m.group(1))
    m = re.search(r"Pixels\s*:\s*(\d+)\s*[x×]\s*(\d+)", text)
    if m:
        info["pixels"] = (int(m.group(1)), int(m.group(2)))
    m = re.search(r"in\s+(OpenGL|DirectX)\s+convention", text, re.I)
    if m:
        info["normals_convention"] = "up" if m.group(1).lower() == "opengl" else "down"
    return info


def _pick(paths, prefer=()):
    """The plain export among candidates: preferred name parts first, then the shortest name (derived
    visualisations add suffixes, e.g. _DEM_FrankotChellappa_MSRM_F_M11.0-11.0_S2.tif)."""
    if not paths:
        return None
    def rank(p):
        name = os.path.basename(p).lower()
        return (min([i for i, k in enumerate(prefer) if k in name] or [len(prefer)]), len(name), name)
    return min(paths, key=rank)


def scan(folder):
    """The HELIO export in `folder`: dict(folder, object_id, kind ('gis' / 'morphosource' / 'unknown'),
    dems [(label, path)], albedo, normals, uncertainty, metal, readme {parse_readme}). Missing items are None /
    empty. uncertainty is Helio's 0-1 fit residual; metal marks where its metal-aware solve re-solved the normals."""
    inside = [d for d in os.listdir(folder) if d.endswith("_GIS_DEM") and os.path.isdir(os.path.join(folder, d))]
    if len(inside) == 1:                         # the folder exported into, not the package in it
        folder = os.path.join(folder, inside[0])
    files = sorted(f for f in os.listdir(folder) if not f.startswith(".") and not f.lower().endswith(".aux.xml"))
    paths = [os.path.join(folder, f) for f in files if os.path.isfile(os.path.join(folder, f))]

    def with_(part, exts=RASTER):
        return [p for p in paths if part in os.path.basename(p).lower() and p.lower().endswith(exts)]

    dems = []
    for label, part in (("Poisson", "_dem_poisson"), ("Frankot-Chellappa", "_dem_frankotchellappa")):
        pick = _pick(with_(part, (".tif", ".tiff")))
        if pick:
            dems.append((label, pick))
    if not dems:
        pick = _pick(with_("depth", (".tif", ".tiff")))
        if pick:
            dems.append(("Depth", pick))
    albedo = _pick(with_("albedo"), prefer=("16bit", ".tif", ".png", ".jpg"))
    normals = _pick([p for p in with_("normal") if not p.lower().endswith((".mtl", ".obj"))], prefer=("normals",))
    uncertainty = _pick(with_("_uncertainty", (".tif", ".tiff")))
    metal = _pick(with_("_metal_mask", (".tif", ".tiff")))
    sections = _pick(with_("_sections", (".geojson",)))
    annotations = _pick(with_("_annotations", (".geojson",)))
    readme_path = next((p for p in paths if os.path.basename(p).lower() in ("readme_gis.txt", "readme.txt")), None)
    readme = parse_readme(_text(readme_path)) if readme_path else {}
    kind = ("gis" if any(k in ("Poisson", "Frankot-Chellappa") for k, _ in dems) else
            "morphosource" if dems and (normals or "morphosource" in folder.lower()) else "unknown")
    return dict(folder=folder, object_id=readme.get("object_id") or os.path.basename(os.path.normpath(folder)),
                kind=kind, dems=dems, albedo=albedo, normals=normals, uncertainty=uncertainty, metal=metal,
                readme=readme, readme_path=readme_path, object_type=readme.get("object_type", ""),
                sections=sections, annotations=annotations)
