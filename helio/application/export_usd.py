"""OpenUSD and USDZ (EXP-012): the relief as a USD scene, and as one file
for AR at the object's true size.

Written by hand (USDA text, and the USDZ zip with each file's data on a
64-byte boundary as the format requires), so it needs no USD library; when
Pixar's ``pxr`` is installed the tests check the result with its own
compliance checker, ARKit rules included.

The scene follows the AOUSD core specification's conventions and Apple's
USDZ rules: Y up, metres (``metersPerUnit = 1``), one default prim, a
``UsdPreviewSurface`` with the colour photograph as its base colour and the
photometric normals as a tangent-space normal map (read raw, scaled 2 and
biased -1), PNG and JPEG textures only, no subdivision.  A flat object
lies on a table (``preliminary:anchoring:type = "plane"``, horizontal); a
wall object hangs (vertical).  The capture's lights travel with the scene
as UsdLux distant lights (off but for the view's light) and in the layer's
metadata, with the scale, the solve and HELIO's version, so the file says
where it came from.
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import time
import zipfile

import cv2
import numpy as np

MAX_VERTS = 250_000                 # AR viewers on phones stay smooth below this
MAX_TEX = 4096


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def mesh_from_depth(depth01: np.ndarray, normals: np.ndarray, *, width_m: float,
                    height_scale_m: float, step: int):
    """A grid mesh of the relief.

    Returns (points (N,3) in metres, Y up, the object lying in X-Z; face
    vertex indices; per-vertex st; per-vertex normals; grid (gw, gh))."""
    h, w = depth01.shape
    xs = np.arange(0, w, step)
    ys = np.arange(0, h, step)
    if xs[-1] != w - 1:
        xs = np.append(xs, w - 1)
    if ys[-1] != h - 1:
        ys = np.append(ys, h - 1)
    gw, gh = len(xs), len(ys)
    px = width_m / max(w - 1, 1)                       # metres per image pixel
    X, Y = np.meshgrid(xs, ys)
    Z = depth01[Y, X].astype(np.float64)
    # image x -> +X, image row (south) -> +Z (towards the viewer), height -> +Y
    pts = np.stack([(X - (w - 1) / 2.0) * px, (Z - Z.min()) * height_scale_m,
                    (Y - (h - 1) / 2.0) * px], -1).reshape(-1, 3)
    st = np.stack([X / max(w - 1, 1), 1.0 - Y / max(h - 1, 1)], -1).reshape(-1, 2)
    n = normals[Y, X].astype(np.float64)                # +Y up (north) in image terms
    nv = np.stack([n[..., 0], n[..., 2], -n[..., 1]], -1).reshape(-1, 3)   # to X, Y(up), Z(south)
    nv /= np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-9)
    idx = np.arange(gw * gh).reshape(gh, gw)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.stack([a, c, d, b], -1)                  # quads, counter-clockwise seen from above
    return pts, faces, st, nv, (gw, gh)


def _arr(name: str, kind: str, rows, per: int, fmt=_fmt, extra: str = "") -> str:
    vals = ", ".join("(" + ", ".join(fmt(v) for v in r) + ")" for r in rows) if per > 1 else \
        ", ".join(fmt(v) for v in rows)
    return f"        {kind} {name} = [{vals}]{extra}\n"


def usda(*, pts, faces, st, nrm, albedo_name: str, normal_name: str, meta: dict,
         lights=None, view_light=None, wall: bool = False, name: str = "Relief",
         with_lights: bool = True) -> str:
    """The scene as USDA text.  *with_lights* False for AR: Apple's rules
    allow no lights in a USDZ (the viewer lights it); the lights stay in the
    metadata."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    fv = faces.ravel()
    out = io.StringIO()
    out.write("#usda 1.0\n(\n")
    out.write(f'    defaultPrim = "{name}"\n    metersPerUnit = 1\n    upAxis = "Y"\n')
    out.write("    customLayerData = {\n")
    out.write(f"        string helio = {json.dumps(meta.get('helio', 'Helio'))}\n")
    out.write(f"        string helio_metadata = "
              f"{json.dumps(json.dumps(meta, ensure_ascii=False), ensure_ascii=False)}\n")
    out.write("    }\n)\n\n")
    anchor = "vertical" if wall else "horizontal"
    out.write(f'def Xform "{name}" (\n    kind = "component"\n    prepend apiSchemas = ["Preliminary_AnchoringAPI"]\n)\n{{\n')
    out.write('    token preliminary:anchoring:type = "plane"\n')
    out.write(f'    token preliminary:planeAnchoring:alignment = "{anchor}"\n')
    if wall:
        out.write("    float3 xformOp:rotateXYZ = (90, 0, 0)\n")
        out.write('    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]\n')
    out.write('\n    def Mesh "Surface" (\n        prepend apiSchemas = ["MaterialBindingAPI"]\n    )\n    {\n')
    out.write(f"        float3[] extent = [({_fmt(lo[0])}, {_fmt(lo[1])}, {_fmt(lo[2])}), "
              f"({_fmt(hi[0])}, {_fmt(hi[1])}, {_fmt(hi[2])})]\n")
    out.write("        int[] faceVertexCounts = [" + ", ".join(["4"] * len(faces)) + "]\n")
    out.write("        int[] faceVertexIndices = [" + ", ".join(str(int(i)) for i in fv) + "]\n")
    out.write(_arr("points", "point3f[]", pts, 3))
    out.write(_arr("normals", "normal3f[]", nrm, 3, extra=" (\n            interpolation = \"vertex\"\n        )"))
    out.write(_arr("primvars:st", "texCoord2f[]", st, 2,
                   extra=" (\n            interpolation = \"vertex\"\n        )"))
    out.write('        uniform token subdivisionScheme = "none"\n')
    out.write("        rel material:binding = </" + name + "/Looks/Photograph>\n")
    out.write("    }\n\n")
    out.write('    def Scope "Looks"\n    {\n')
    out.write('        def Material "Photograph"\n        {\n')
    out.write(f"            token outputs:surface.connect = </{name}/Looks/Photograph/Surface.outputs:surface>\n\n")
    out.write('            def Shader "Surface"\n            {\n')
    out.write('                uniform token info:id = "UsdPreviewSurface"\n')
    out.write(f"                color3f inputs:diffuseColor.connect = </{name}/Looks/Photograph/Colour.outputs:rgb>\n")
    out.write(f"                normal3f inputs:normal.connect = </{name}/Looks/Photograph/Normals.outputs:rgb>\n")
    out.write("                float inputs:roughness = 0.7\n                float inputs:metallic = 0\n")
    out.write("                token outputs:surface\n            }\n\n")
    out.write('            def Shader "Coords"\n            {\n')
    out.write('                uniform token info:id = "UsdPrimvarReader_float2"\n')
    out.write('                string inputs:varname = "st"\n                float2 outputs:result\n            }\n\n')
    for tex, file, extra in (("Colour", albedo_name, '                token inputs:sourceColorSpace = "sRGB"\n'),
                             ("Normals", normal_name,
                              '                token inputs:sourceColorSpace = "raw"\n'
                              "                float4 inputs:scale = (2, 2, 2, 1)\n"
                              "                float4 inputs:bias = (-1, -1, -1, 0)\n")):
        out.write(f'            def Shader "{tex}"\n            {{\n')
        out.write('                uniform token info:id = "UsdUVTexture"\n')
        out.write(f"                asset inputs:file = @{file}@\n")
        out.write(f"                float2 inputs:st.connect = </{name}/Looks/Photograph/Coords.outputs:result>\n")
        out.write('                token inputs:wrapS = "clamp"\n                token inputs:wrapT = "clamp"\n')
        out.write(extra)
        out.write("                float3 outputs:rgb\n            }\n\n")
    out.write("        }\n    }\n")
    if not with_lights:
        out.write("}\n")
        return out.getvalue()
    # The capture's lights, and the view's: they travel with the scene (UsdLux).
    out.write('\n    def Scope "Lights"\n    {\n')
    all_lights = []
    if view_light is not None:
        all_lights.append(("ViewLight", view_light, True))
    for i, L in enumerate(lights if lights is not None else []):
        all_lights.append((f"Capture{i + 1:02d}", L, False))
    for nm, L, on in all_lights:
        v = np.asarray(L, np.float64)
        v = v / max(np.linalg.norm(v), 1e-9)
        # light direction in scene axes: x east, y up (from the light's z), z south
        d = np.array([v[0], v[2], -v[1]])
        # a DistantLight shines down its -Z; aim -Z along -d
        yaw = math.degrees(math.atan2(d[0], d[2]))
        pitch = -math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))
        out.write(f'        def DistantLight "{nm}"\n        {{\n')
        out.write(f"            float inputs:intensity = {3 if on else 0}\n")
        out.write("            float inputs:angle = 0.53\n")
        out.write(f"            float3 xformOp:rotateXYZ = ({_fmt(pitch)}, {_fmt(yaw)}, 0)\n")
        out.write('            uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]\n')
        if not on:
            out.write('            token visibility = "invisible"\n')
        out.write("        }\n")
    out.write("    }\n}\n")
    return out.getvalue()


def _png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("could not encode a PNG")
    return buf.tobytes()


def _jpg(arr: np.ndarray, q: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("could not encode a JPEG")
    return buf.tobytes()


def write_usdz(path: str, files: list[tuple[str, bytes]]) -> str:
    """A USDZ: an uncompressed zip, the USD layer first, every file's data
    starting on a 64-byte boundary (the alignment field is extra data, ID 0x1986)."""
    with open(path, "wb") as raw:
        zf = zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_STORED, allowZip64=False)
        for name, data in files:
            zi = zipfile.ZipInfo(name, date_time=time.localtime(time.time())[:6])
            zi.compress_type = zipfile.ZIP_STORED
            header = raw.tell()
            base = header + 30 + len(name.encode("utf-8")) + 4     # 4: the extra field's own header
            pad = (-base) % 64
            zi.extra = struct.pack("<HH", 0x1986, pad) + b"\0" * pad
            zf.writestr(zi, data)
        zf.close()
    return path


def export(out_path: str, *, depth, normals, albedo, mm_per_px=None, exaggeration: float = 1.0,
           assumed_width_mm: float = 200.0, wall: bool = False, lights=None, view_light=None,
           meta: dict | None = None, also_usda: bool = False) -> dict:
    """USDZ (or .usda / .usdc-free USDA when *out_path* ends .usda).

    *depth* any float height map; *normals* +Y up; *albedo* uint8 RGB.
    Heights: the height map scaled so its slopes agree with the normals (the
    true shape), times *exaggeration*.  Size: *mm_per_px*, or, without one,
    *assumed_width_mm* across, said so in the notes.  Returns a dict of the
    file(s), sizes and notes."""
    from helio.core.metrology import true_shape_scale
    notes = []
    d = np.asarray(depth, np.float32)
    lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
    d01 = np.nan_to_num((d - lo) / (hi - lo if hi > lo else 1.0)).astype(np.float32)
    nrm = np.asarray(normals, np.float32)
    if nrm.shape[:2] != d01.shape:
        nrm = cv2.resize(nrm, (d01.shape[1], d01.shape[0]), interpolation=cv2.INTER_AREA)
    h, w = d01.shape
    if mm_per_px:
        width_m = w * float(mm_per_px) / 1000.0
    else:
        width_m = assumed_width_mm / 1000.0
        notes.append(f"No scale was set, so the object is {assumed_width_mm:.0f} mm across in AR. "
                     "Measure → Set Scale makes it its true size.")
    s_px = true_shape_scale(d01, nrm)                    # pixels of height per unit
    height_scale_m = s_px * (width_m / max(w, 1)) * float(exaggeration)
    step = max(1, int(math.ceil(math.sqrt(w * h / MAX_VERTS))))
    pts, faces, st, nv, grid = mesh_from_depth(d01, nrm, width_m=width_m,
                                               height_scale_m=height_scale_m, step=step)
    # textures, at most MAX_TEX on the long side
    alb = np.ascontiguousarray(albedo, np.uint8)
    k = min(1.0, MAX_TEX / max(alb.shape[:2]))
    if k < 1.0:
        alb = cv2.resize(alb, (int(alb.shape[1] * k), int(alb.shape[0] * k)), interpolation=cv2.INTER_AREA)
    nt = cv2.resize(nrm, (alb.shape[1], alb.shape[0]), interpolation=cv2.INTER_AREA)
    nt = nt / np.maximum(np.linalg.norm(nt, axis=-1, keepdims=True), 1e-8)
    # tangent space for a flat grid with st = (x, 1 - y): T = +x, B = +north, N = up
    n8 = np.clip(np.round((nt * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)
    meta = dict(meta or {})
    meta.setdefault("made", time.strftime("%Y-%m-%d %H:%M"))
    meta.update(mm_per_px=float(mm_per_px) if mm_per_px else None,
                width_mm=round(width_m * 1000, 3), exaggeration=float(exaggeration),
                height_from="the normals (true shape scale)",
                vertices=int(len(pts)), mesh_step=step)
    if lights is not None:
        meta["capture_lights"] = [[round(float(v), 6) for v in L] for L in np.asarray(lights)]
    if view_light is not None:
        meta["view_light"] = [round(float(v), 6) for v in view_light]
    args = dict(pts=pts, faces=faces, st=st, nrm=nv, albedo_name="colour.jpg",
                normal_name="normals.png", meta=meta, lights=lights, view_light=view_light, wall=wall)
    text = usda(**args)                              # the scene, its lights in it
    text_ar = usda(**args, with_lights=False)        # for AR: the viewer brings the light
    tex = [("colour.jpg", _jpg(alb[..., ::-1])), ("normals.png", _png(n8[..., ::-1]))]
    out = dict(path=out_path, vertices=int(len(pts)), faces=int(len(faces)), step=step,
               width_mm=round(width_m * 1000, 2), notes=notes, files=[], writer="")
    stem = os.path.splitext(os.path.basename(out_path))[0] or "relief"
    if out_path.lower().endswith(".usda"):
        folder = os.path.dirname(os.path.abspath(out_path))
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        for nm, data in tex:
            with open(os.path.join(folder, nm), "wb") as f:
                f.write(data)
        out["files"] = [out_path] + [os.path.join(folder, nm) for nm, _ in tex]
        out["writer"] = "text"
    else:
        if not _usdz_with_pxr(out_path, stem, text_ar, tex):
            # No OpenUSD library here: the text layer, packaged by hand.  Quick
            # Look and most viewers read it; Apple's rules prefer a binary root.
            write_usdz(out_path, [(f"{stem}.usda", text_ar.encode("utf-8"))] + tex)
            out["writer"] = "text"
            notes.append("Written without the OpenUSD library, as text inside the USDZ. "
                         "pip install usd-core gives the binary form Apple's rules prefer.")
        else:
            out["writer"] = "usdc"
        out["files"] = [out_path]
        if also_usda:
            folder = os.path.join(os.path.dirname(os.path.abspath(out_path)), f"{stem}_usda")
            os.makedirs(folder, exist_ok=True)
            p = os.path.join(folder, f"{stem}.usda")
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            for nm, data in tex:
                with open(os.path.join(folder, nm), "wb") as f:
                    f.write(data)
            out["files"].append(p)
    out["mb"] = round(sum(os.path.getsize(p) for p in out["files"] if os.path.isfile(p)) / 1e6, 2)
    return out


def _usdz_with_pxr(out_path: str, stem: str, text: str, tex) -> bool:
    """USDZ through OpenUSD: the layer converted to binary (usdc) and packaged
    by UsdUtils' ARKit packager.  False when pxr is not installed."""
    try:
        from pxr import Sdf, UsdUtils
    except Exception:
        return False
    import tempfile
    tmp = tempfile.mkdtemp(prefix="helio_usdz_")
    src = os.path.join(tmp, f"{stem}.usda")
    with open(src, "w", encoding="utf-8") as f:
        f.write(text)
    for nm, data in tex:
        with open(os.path.join(tmp, nm), "wb") as f:
            f.write(data)
    layer = Sdf.Layer.FindOrOpen(src)
    if layer is None:
        return False
    usdc = os.path.join(tmp, f"{stem}.usdc")
    if not layer.Export(usdc):
        return False
    if os.path.exists(out_path):
        os.remove(out_path)
    ok = UsdUtils.CreateNewARKitUsdzPackage(usdc, out_path)
    return bool(ok) and os.path.isfile(out_path)


def check(path: str, arkit: bool = True) -> tuple[bool, list[str]]:
    """Pixar's compliance checker on *path*, when pxr is installed: (passed, problems)."""
    try:
        from pxr import UsdUtils
    except Exception:
        return True, ["pxr is not installed here: not checked"]
    have_sdr = _shader_registry()        # asked first: pxr's first lookup may raise
    if not hasattr(UsdUtils, "ComplianceChecker"):
        return _check_validation(path, arkit, have_sdr)
    c = UsdUtils.ComplianceChecker(arkit=arkit, skipARKitRootLayerCheck=False)
    if not have_sdr:
        # usd-core from pip ships without the shader definitions (usdHydra's
        # shaderDefs.usda), so the shader-type rule cannot run here; in USD
        # 25.11 it raises from inside the checker instead of failing a check.
        rules = getattr(c, "_rules", None)
        if isinstance(rules, list):
            c._rules = [r for r in rules
                        if type(r).__name__ != "ShaderPropertyTypeConformanceChecker"]
    try:
        c.CheckCompliance(path)
    except Exception as e:                   # the checker's own failure is not the file's
        last = (str(e).strip().splitlines() or ["?"])[-1].strip()
        return False, [f"the compliance checker stopped: {last[:200]}"]
    problems = list(c.GetErrors()) + list(c.GetFailedChecks())
    if not have_sdr:
        problems = [p for p in problems if "ShaderPropertyTypeConformanceChecker" not in p]
    return not problems, problems


# Apple's AR rules, as UsdUtils.ComplianceChecker(arkit=True) had them.
_ARKIT_PRIM_TYPES = {"", "Scope", "Xform", "Camera", "Shader", "Material", "Mesh", "Sphere", "Cube",
                     "Cylinder", "Cone", "Capsule", "GeomSubset", "Points", "SkelRoot", "Skeleton",
                     "SkelAnimation", "BlendShape", "SpatialAudio"}
_ARKIT_SHADERS = {"UsdPreviewSurface", "UsdUVTexture", "UsdTransform2d"}
_ARKIT_FILES = {".usdc", ".usd", ".png", ".jpg", ".jpeg", ".m4a", ".mp3", ".wav"}
# Validator families that have nothing to say about a relief.
_NOT_FOR_RELIEF = ("usdPhysicsValidators:", "usdMediaValidators:", "usdSkelValidators:")


def _check_validation(path: str, arkit: bool, have_sdr: bool) -> tuple[bool, list[str]]:
    """check() for OpenUSD 26 and later, where the Python ComplianceChecker is
    gone: Pixar's UsdValidation validators (core, geometry, shading, lights,
    packaging), and Apple's AR rules, which UsdValidation does not carry, by
    hand.  Errors fail; warnings do not, as with the old checker."""
    from pxr import Usd, UsdValidation
    try:
        stage = Usd.Stage.Open(path)
    except Exception as e:
        return False, [f"OpenUSD could not open it: {str(e).strip()[:200]}"]
    if stage is None:
        return False, ["OpenUSD could not open it"]
    reg = UsdValidation.ValidationRegistry()
    names = [m.name for m in reg.GetAllValidatorMetadata()
             if not m.isSuite and not m.name.startswith(_NOT_FOR_RELIEF)
             and (have_sdr or m.name != "usdShadeValidators:ShaderSdrCompliance")]
    try:
        found = UsdValidation.ValidationContext(reg.GetOrLoadValidatorsByName(names)).Validate(stage)
    except Exception as e:                   # the validators' own failure is not the file's
        last = (str(e).strip().splitlines() or ["?"])[-1].strip()
        return False, [f"the USD validators stopped: {last[:200]}"]
    problems = [f"{e.GetName()}: {e.GetMessage()}" for e in found
                if e.GetType() == UsdValidation.ValidationErrorType.Error]
    if arkit:
        problems += _arkit_problems(path, stage)
    return not problems, problems


def _arkit_problems(path: str, stage) -> list[str]:
    out = []
    if path.lower().endswith(".usdz"):
        with zipfile.ZipFile(path) as z:
            members = [i.filename for i in z.infolist()]
        if not members or os.path.splitext(members[0])[1].lower() != ".usdc":
            out.append("ARKit: the package's root layer must be binary (.usdc)")
        bad = [m for m in members if os.path.splitext(m)[1].lower() not in _ARKIT_FILES]
        if bad:
            out.append(f"ARKit: files of a kind AR viewers do not read: {', '.join(bad[:5])}")
        layers = [m for m in members if m.lower().endswith((".usd", ".usda", ".usdc"))]
        if len(layers) > 1:
            out.append(f"ARKit: one USD layer per package, found {len(layers)}")
    for prim in stage.Traverse():
        t = prim.GetTypeName()
        if t not in _ARKIT_PRIM_TYPES:
            out.append(f"ARKit: prim type {t} is not allowed ({prim.GetPath()})")
        if t == "Shader":
            a = prim.GetAttribute("info:id")
            sid = str(a.Get()) if a and a.HasAuthoredValue() else ""
            if sid not in _ARKIT_SHADERS and not sid.startswith("UsdPrimvarReader"):
                out.append(f"ARKit: shader {sid or '(no id)'} is not a preview shader ({prim.GetPath()})")
    return out


def _shader_registry() -> bool:
    try:
        from pxr import Sdr
        return Sdr.Registry().GetShaderNodeByIdentifier("UsdPreviewSurface") is not None
    except Exception:
        return False


def read_mesh(path: str) -> dict:
    """The first mesh of a USD, USDA, USDC or USDZ file, for viewing: points
    (N, 3), triangles (M, 3), and the base colour texture's bytes if it has
    one.  Needs pxr; says so plainly when it is missing."""
    try:
        from pxr import Usd, UsdGeom
    except Exception:
        raise RuntimeError("Opening USD files needs OpenUSD (pip install usd-core).")
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"Could not open {os.path.basename(path)} as USD.")
    up = UsdGeom.GetStageUpAxis(stage)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
            counts = np.asarray(m.GetFaceVertexCountsAttr().Get(), np.int64)
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), np.int64)
            tris, k = [], 0
            for c in counts:
                f = idx[k:k + c]
                for j in range(1, c - 1):                       # fan
                    tris.append((f[0], f[j], f[j + 1]))
                k += c
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            M = np.array(xf, np.float64).T
            P = (np.c_[pts, np.ones(len(pts))] @ M.T)[:, :3]
            return dict(points=P, triangles=np.asarray(tris, np.int64), up=str(up),
                        name=prim.GetName(), meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage))
    raise RuntimeError(f"No mesh in {os.path.basename(path)}.")
