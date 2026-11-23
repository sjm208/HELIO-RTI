"""Masters: the surface as numbers, for keeping, not for looking at.

Scene-linear OpenEXR for the colour, normals and heights, and a layered USD
stage, as the VFX and museum 3D worlds hold their work: the geometry, the record
of how it was made and the colour are separate layers, so one can be changed
without rewriting the others (a flattened USDZ is the delivery form of the same).

The colour's numbers are linear with sRGB (Rec. 709) primaries.  They are the
colour map decoded, so 8-bit in origin: the height and normals are the
measurement here, and a colour EXR says no more than the map it came from.
GUI-free.
"""
from __future__ import annotations

import json
import os
import re

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")    # read when OpenEXR is first used

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def write_exr(path: str, arr, *, half: bool = False) -> str:
    """(h, w, 3) RGB or (h, w) float, OpenEXR with ZIP compression."""
    a = np.ascontiguousarray(arr, np.float32)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    args = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT]
    if not cv2.imwrite(path, a, args):
        raise OSError(f"could not write {os.path.basename(path)}")
    return path


def linear_masters(folder: str, stem: str, *, colour, normals, depth, solved=None,
                   z_mm_per_unit: float = 0.0) -> list[str]:
    """Colour (half float), normals and heights (float).  *depth* is true heights in
    pixels of height (helio.application.heights): written in mm once there is a height
    scale (mm per pixel of height), else in pixels of height."""
    from helio.core.radiometry import srgb_to_linear
    os.makedirs(folder, exist_ok=True)
    p = lambda name: os.path.join(folder, f"{stem}_{name}.exr")     # noqa: E731
    files = [write_exr(p("colour_linear"), srgb_to_linear(colour, 1.0 * 255) / 255.0, half=True),
             write_exr(p("normals"), normals)]
    h = np.asarray(depth, np.float32)
    files.append(write_exr(p("height_mm" if z_mm_per_unit else "height_px"), h * float(z_mm_per_unit or 1.0)))
    if solved is not None:
        files.append(write_exr(p("reflectance_linear"), srgb_to_linear(solved, 255.0) / 255.0, half=True))
    return files


def _identifier(text: str, fallback: str) -> str:
    from pxr import Tf
    name = Tf.MakeValidIdentifier(str(text or "")) or fallback
    return name if name[0].isalpha() else f"{fallback}_{name}"


def _surface_grid(surface: str, prim: str, w: int, h: int):
    """(point on the surface for an image pixel, height scale m per unit of the normalised map):
    the surface mesh read back, so the marks sit exactly on it."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(surface)
    pts = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(f"/{prim}/Surface")).GetPointsAttr().Get(), np.float64)
    gx = np.unique(np.round(pts[:, 0], 9))
    gz = np.unique(np.round(pts[:, 2], 9))
    grid = pts[:, 1].reshape(len(gz), len(gx))
    x0, x1, z0, z1 = gx[0], gx[-1], gz[0], gz[-1]

    def at(x, y):
        X = x0 + (x1 - x0) * float(x) / max(w - 1, 1)
        Z = z0 + (z1 - z0) * float(y) / max(h - 1, 1)
        fx = np.clip((X - x0) / max(x1 - x0, 1e-12) * (len(gx) - 1), 0, len(gx) - 1)
        fz = np.clip((Z - z0) / max(z1 - z0, 1e-12) * (len(gz) - 1), 0, len(gz) - 1)
        i, j = int(min(fx, len(gx) - 2)), int(min(fz, len(gz) - 2))
        tx, tz = fx - i, fz - j
        Y = (grid[j, i] * (1 - tx) * (1 - tz) + grid[j, i + 1] * tx * (1 - tz)
             + grid[j + 1, i] * (1 - tx) * tz + grid[j + 1, i + 1] * tx * tz)
        return (X, float(Y), Z)
    return at, float(np.ptp(pts[:, 1]))


def _hex(c: str):
    c = str(c or "#ff0000").lstrip("#")
    try:
        return tuple(int(c[k:k + 2], 16) / 255.0 for k in (0, 2, 4))
    except ValueError:
        return (1.0, 0.0, 0.0)


def _marks_layer(path, prim, at, lift, sections=(), annotations=(), mm_per_px=None) -> bool:
    """Kept sections and annotations as curves and points lying on the surface (lifted by *lift*)."""
    from pxr import Gf, Usd, UsdGeom, Vt
    from helio.application.annotations import length_label
    from helio.application.export_vectors import _AREAS, _POINTS
    stage = Usd.Stage.CreateNew(path)
    root = stage.OverridePrim(f"/{prim}")
    scope = UsdGeom.Scope.Define(stage, root.GetPath().AppendChild("Marks"))
    done = 0

    def lifted(x, y):
        X, Y, Z = at(x, y)
        return Gf.Vec3f(X, Y + lift, Z)

    def curve(name, pts_px, colour, closed=False, props=None, width=None):
        nonlocal done
        pts = [lifted(*p) for p in pts_px]
        if closed:
            pts.append(pts[0])
        if len(pts) < 2:
            return
        c = UsdGeom.BasisCurves.Define(stage, scope.GetPath().AppendChild(name))
        c.CreateTypeAttr(UsdGeom.Tokens.linear)
        c.CreatePointsAttr(Vt.Vec3fArray(pts))
        c.CreateCurveVertexCountsAttr(Vt.IntArray([len(pts)]))
        c.CreateWidthsAttr(Vt.FloatArray([width or lift * 2]))
        c.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
        for k, v in (props or {}).items():
            c.GetPrim().SetCustomDataByKey(k, v)
        done += 1
    for s in sections:
        (ax, ay), (bx, by) = s["a"], s["b"]
        line = [(ax + (bx - ax) * t, ay + (by - ay) * t) for t in np.linspace(0, 1, 65)]
        curve(_identifier("Section_" + s["name"], "Section"), line, (0.17, 0.42, 1.0),
              props=dict(name=s["name"], length=length_label((ax, ay), (bx, by), mm_per_px),
                         copies=int(s.get("copies", 0))), width=lift * 3)
    for i, a in enumerate(annotations, 1):
        t, pts = a.get("type"), [tuple(p) for p in (a.get("pts") or [])]
        if not pts:
            continue
        colour = _hex(a.get("color"))
        label = a.get("text") or a.get("label") or ""
        props = dict(kind=t, label=label, layer=a.get("layer", ""))
        name = _identifier(f"Note_{i}_{t}", "Note")
        if t in _POINTS:
            p = UsdGeom.Points.Define(stage, scope.GetPath().AppendChild(name))
            p.CreatePointsAttr(Vt.Vec3fArray([lifted(*pts[0])]))
            p.CreateWidthsAttr(Vt.FloatArray([lift * 8]))
            p.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colour)]))
            for k, v in props.items():
                p.GetPrim().SetCustomDataByKey(k, v)
            done += 1
        elif t in _AREAS and len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[-1]
            if t == "rect":
                ring = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            elif t == "ellipse":
                cx, cy, rx, ry = (x1 + x2) / 2, (y1 + y2) / 2, abs(x2 - x1) / 2, abs(y2 - y1) / 2
                ring = [(cx + rx * np.cos(k * np.pi / 36), cy + ry * np.sin(k * np.pi / 36)) for k in range(72)]
            else:
                ring = pts
            curve(name, ring, colour, closed=True, props=props)
        else:
            if t == "measure" and not label and len(pts) > 1:
                props["label"] = length_label(pts[0], pts[-1], mm_per_px)
            curve(name, pts, colour, props=props)
    if done:
        stage.GetRootLayer().Save()
    return bool(done)


def _looks_layer(path, prim, colour_file, normals_file, height_file, height_m, encoded_normals_file) -> None:
    """Two more ways to look at the surface: the measurement itself (the solve's normals and heights
    as float EXR textures, read raw, the heights as displacement) and OpenPBR, the MaterialX surface
    the Alliance for OpenUSD is moving to."""
    from pxr import Sdf, Usd, UsdShade
    stage = Usd.Stage.CreateNew(path)
    looks = stage.OverridePrim(f"/{prim}/Looks").GetPath()

    def texture(mat_path, name, file, space, out_type=Sdf.ValueTypeNames.Float3, out="rgb"):
        t = UsdShade.Shader.Define(stage, mat_path.AppendChild(name))
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"./{file}"))
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(space)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            UsdShade.Shader(stage.GetPrimAtPath(mat_path.AppendChild("Coords"))).ConnectableAPI(), "result")
        t.CreateOutput(out, out_type)
        return t

    m = UsdShade.Material.Define(stage, looks.AppendChild("Measured"))
    st = UsdShade.Shader.Define(stage, m.GetPath().AppendChild("Coords"))
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
    st.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    col = texture(m.GetPath(), "Colour", colour_file, "sRGB")
    nrm = texture(m.GetPath(), "Normals", normals_file, "raw")      # tangent space already: x, north, out
    hgt = texture(m.GetPath(), "Height", height_file, "raw", Sdf.ValueTypeNames.Float, "r")
    hgt.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((height_m, height_m, height_m, 1.0))
    sh = UsdShade.Shader.Define(stage, m.GetPath().AppendChild("Surface"))
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(col.ConnectableAPI(), "rgb")
    sh.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nrm.ConnectableAPI(), "rgb")
    sh.CreateInput("displacement", Sdf.ValueTypeNames.Float).ConnectToSource(hgt.ConnectableAPI(), "r")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    m.CreateDisplacementOutput().ConnectToSource(sh.ConnectableAPI(), "displacement")
    sh.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    sh.CreateOutput("displacement", Sdf.ValueTypeNames.Token)

    o = UsdShade.Material.Define(stage, looks.AppendChild("OpenPBR"))
    img = UsdShade.Shader.Define(stage, o.GetPath().AppendChild("BaseColour"))
    img.CreateIdAttr("ND_image_color3")
    img.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"./{colour_file}"))
    img.GetInput("file").GetAttr().SetColorSpace("srgb_texture")
    img.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
    # MaterialX's normal map takes the tangent-space normal encoded 0-1 (the 8-bit map the surface
    # already carries), not the raw -1..1 EXR, and gives the world normal OpenPBR takes
    nimg = UsdShade.Shader.Define(stage, o.GetPath().AppendChild("NormalImage"))
    nimg.CreateIdAttr("ND_image_vector3")
    nimg.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"./{encoded_normals_file}"))
    nimg.GetInput("file").GetAttr().SetColorSpace("raw")
    nimg.CreateOutput("out", Sdf.ValueTypeNames.Vector3f)
    nmap = UsdShade.Shader.Define(stage, o.GetPath().AppendChild("NormalMap"))
    nmap.CreateIdAttr("ND_normalmap_float")
    nmap.CreateInput("in", Sdf.ValueTypeNames.Vector3f).ConnectToSource(nimg.ConnectableAPI(), "out")
    nmap.CreateOutput("out", Sdf.ValueTypeNames.Vector3f)
    pbr = UsdShade.Shader.Define(stage, o.GetPath().AppendChild("Surface"))
    pbr.CreateIdAttr("ND_open_pbr_surface_surfaceshader")
    pbr.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(img.ConnectableAPI(), "out")
    pbr.CreateInput("geometry_normal", Sdf.ValueTypeNames.Vector3f).ConnectToSource(nmap.ConnectableAPI(), "out")
    pbr.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    pbr.CreateOutput("out", Sdf.ValueTypeNames.Token)
    o.CreateSurfaceOutput("mtlx").ConnectToSource(pbr.ConnectableAPI(), "out")
    stage.GetRootLayer().Save()


_USD_TO_MTLX = {"color3f": "color3", "float": "float", "vector3f": "vector3", "float3": "vector3",
                "asset": "filename", "string": "string", "token": "surfaceshader", "int": "integer",
                "float2": "vector2", "color4f": "color4"}


def check_materialx(stage_path: str) -> list[str]:
    """Every MaterialX node in the stage against MaterialX's own library: the node exists, each
    input it is given is one the node has, of that type, and each connection joins matching types.
    [] when right; a note when MaterialX is not installed (it is for development, not the app)."""
    try:
        import MaterialX as mx
    except ImportError:
        return ["MaterialX is not installed here: not checked"]
    from pxr import Usd, UsdShade
    lib = mx.createDocument()
    mx.loadLibraries(mx.getDefaultDataLibraryFolders(), mx.getDefaultDataSearchPath(), lib)
    stage = Usd.Stage.Open(stage_path)
    problems = []

    def mtype(t):
        return _USD_TO_MTLX.get(str(t).lower(), str(t).lower())
    for prim in stage.Traverse():
        sh = UsdShade.Shader(prim)
        if not sh:
            continue
        sid = sh.GetIdAttr().Get() or ""
        if not sid.startswith("ND_"):
            continue
        nd = lib.getNodeDef(sid)
        if nd is None:
            problems.append(f"{prim.GetPath()}: MaterialX has no node {sid}")
            continue
        ins = {i.getName(): i.getType() for i in nd.getActiveInputs()}
        outs = {o.getName(): o.getType() for o in nd.getActiveOutputs()}
        for inp in sh.GetInputs():
            n, t = inp.GetBaseName(), mtype(inp.GetTypeName())
            if n not in ins:
                problems.append(f"{prim.GetPath()}: {sid} has no input {n}")
            elif ins[n] != t:
                problems.append(f"{prim.GetPath()}: {sid}.{n} is {ins[n]}, given {t}")
            for src in inp.GetConnectedSources()[0]:
                so = UsdShade.Shader(src.source.GetPrim())
                sid2 = so.GetIdAttr().Get() or ""
                nd2 = lib.getNodeDef(sid2) if sid2.startswith("ND_") else None
                if nd2 is not None and nd2.getActiveOutput(src.sourceName) is not None \
                        and nd2.getActiveOutput(src.sourceName).getType() != ins.get(n):
                    problems.append(f"{prim.GetPath()}: {n} ({ins.get(n)}) fed a "
                                    f"{nd2.getActiveOutput(src.sourceName).getType()} from {sid2}")
        for out in sh.GetOutputs():
            if out.GetBaseName() not in outs:
                problems.append(f"{prim.GetPath()}: {sid} has no output {out.GetBaseName()}")
    return problems


def usd_layered(folder: str, stem: str, *, depth, normals, colour, mm_per_px=None, meta=None,
                record: dict | None = None, solved=None, lights=None, view_light=None,
                z_mm_per_unit: float = 0.0, sections=(), annotations=(), colour_check=None) -> list[str]:
    """A stage of layers: <stem>.usda holds the record and references the surface, with variant sets
    for the colour (median, and solved when there is one) and the material (the photograph, the
    measurement with displacement, OpenPBR); the capture's lights; the sections and annotations as
    curves on the surface; and the scale in the stage's own metadata.  The extra layers need
    OpenUSD's library (pxr); without it the stage is the surface, its record and the colour."""
    from helio.application import export_usd
    os.makedirs(folder, exist_ok=True)
    surface = os.path.join(folder, f"{stem}_surface.usda")
    export_usd.export(surface, depth=depth, normals=normals, albedo=colour, mm_per_px=mm_per_px, meta=meta or {},
                      lights=lights, view_light=view_light)
    text = open(surface, encoding="utf-8").read()
    prim = re.search(r'defaultPrim\s*=\s*"(\w+)"', text).group(1)
    up = re.search(r'upAxis\s*=\s*"(\w)"', text)
    mpu = re.search(r'metersPerUnit\s*=\s*([\d.e-]+)', text)
    head = f'    defaultPrim = "{prim}"\n'
    head += f'    upAxis = "{up.group(1)}"\n' if up else ""
    head += f"    metersPerUnit = {mpu.group(1)}\n" if mpu else ""
    rec_name = f"{stem}_record.usda"
    blob = json.dumps(record or {}, ensure_ascii=False, default=str).replace("\\", "\\\\").replace('"', '\\"')
    with open(os.path.join(folder, rec_name), "w", encoding="utf-8") as f:
        f.write(f'#usda 1.0\n(\n    customLayerData = {{\n        string helio_paradata = "{blob}"\n    }}\n{head})\n')
    files = [surface, os.path.join(folder, rec_name)]
    solved_over = ""
    if solved is not None:
        name = f"{stem}_reflectance.png"
        cv2.imwrite(os.path.join(folder, name), cv2.cvtColor(solved, cv2.COLOR_RGB2BGR))
        files.append(os.path.join(folder, name))
        solved_over = ('            over "Looks" {\n                over "Photograph" {\n'
                       '                    over "Colour" {\n'
                       f"                        asset inputs:file = @./{name}@\n"
                       "                    }\n                }\n            }\n")
    extra_layers, notes = [], []
    material_variants = ""
    try:
        import pxr  # noqa: F401
        have_pxr = True
    except ImportError:
        have_pxr = False
        notes.append("OpenUSD's library was not here: no measured or OpenPBR material, and no marks.")
    if have_pxr:
        h, w = np.asarray(depth).shape[:2]
        at, relief_m = _surface_grid(surface, prim, w, h)
        d = np.asarray(depth, np.float32)
        d01 = (d - np.nanmin(d)) / max(float(np.nanmax(d) - np.nanmin(d)), 1e-12)
        nfile, hfile = f"{stem}_normals.exr", f"{stem}_height.exr"
        write_exr(os.path.join(folder, nfile), normals)
        write_exr(os.path.join(folder, hfile), d01)
        files += [os.path.join(folder, nfile), os.path.join(folder, hfile)]
        looks_name = f"{stem}_looks.usda"
        _looks_layer(os.path.join(folder, looks_name), prim, "colour.jpg", nfile, hfile, relief_m, "normals.png")
        extra_layers.append(looks_name)
        files.append(os.path.join(folder, looks_name))
        marks_name = f"{stem}_marks.usda"
        lift = max(relief_m * 0.01, 2e-5)
        if _marks_layer(os.path.join(folder, marks_name), prim, at, lift, sections, annotations, mm_per_px):
            extra_layers.append(marks_name)
            files.append(os.path.join(folder, marks_name))
        material_variants = "\n".join(
            f'        "{v}" {{\n            over "Surface" {{\n'
            f"                rel material:binding = </{prim}/Looks/{m}>\n            }}\n        }}"
            for v, m in (("photograph", "Photograph"), ("measured", "Measured"), ("openpbr", "OpenPBR")))
    info = dict(made_with="Helio", mm_per_px=float(mm_per_px) if mm_per_px else None,
                heights_mm_per_px_of_height=float(z_mm_per_unit) if z_mm_per_unit else None,
                heights=("millimetres: true heights from the normals, height scale measured" if z_mm_per_unit
                         else "pixels of height, from the normals: no height scale was measured"),
                colour_check=colour_check or None, notes=notes)
    blob_info = json.dumps(info, ensure_ascii=False, default=str).replace("\\", "\\\\").replace('"', '\\"')
    head += f'    customLayerData = {{\n        string helio_scale = "{blob_info}"\n    }}\n'
    root = os.path.join(folder, f"{stem}.usda")
    # the surface is referenced, not sublayered: a variant is stronger than a reference, weaker than a sublayer
    meta_ = f'    references = @./{os.path.basename(surface)}@</{prim}>\n'
    sets, chosen, body = [], [], ""
    if solved is not None:
        sets.append("colour")
        chosen.append('        string colour = "median"')
        body += ('    variantSet "colour" = {\n        "median" {\n        }\n        "solved" {\n'
                 + solved_over + "        }\n    }\n")
    if material_variants:
        sets.append("material")
        chosen.append('        string material = "photograph"')
        body += '    variantSet "material" = {\n' + material_variants + "\n    }\n"
    if sets:
        meta_ += ("    variants = {\n" + "\n".join(chosen) + "\n    }\n"
                  + "    prepend variantSets = [" + ", ".join(f'"{s}"' for s in sets) + "]\n")
    subs = ",\n".join(f"        @./{n}@" for n in [rec_name, *extra_layers])
    with open(root, "w", encoding="utf-8") as f:
        f.write(f'#usda 1.0\n(\n{head}    subLayers = [\n{subs}\n    ]\n)\n\n'
                f'def "{prim}" (\n{meta_})\n{{\n{body}}}\n')
    return [root] + files + [os.path.join(folder, n) for n in os.listdir(folder)
                             if n not in {os.path.basename(x) for x in [root, *files]}]
