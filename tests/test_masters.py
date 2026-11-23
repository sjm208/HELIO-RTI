"""What is kept: paradata, deposit, linear masters, layered USD, colour validation.

    python tests/test_masters.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    from helio.application import colour_check as C, export_masters as M, paradata as P
    d = tempfile.mkdtemp(prefix="helio_m_")
    yy, xx = np.mgrid[0:60, 0:80].astype(np.float32)
    depth = np.sin(xx / 9) * 3
    n = np.dstack([np.zeros_like(xx), np.zeros_like(xx), np.ones_like(xx)])
    colour = np.full((60, 80, 3), (188, 100, 40), np.uint8)
    solved = np.full((60, 80, 3), (150, 90, 40), np.uint8)

    print("Linear masters")
    fs = M.linear_masters(d, "s", colour=colour, normals=n, depth=depth, solved=solved, z_mm_per_unit=0.5)
    c = cv2.imread(os.path.join(d, "s_colour_linear.exr"), cv2.IMREAD_UNCHANGED)
    check("the colour is linear, RGB in order (sRGB 188 reads about 0.50)",
          c is not None and abs(float(c[5, 5, 2]) - 0.503) < 0.01 and c[5, 5, 2] > c[5, 5, 0], c[5, 5])
    h = cv2.imread(os.path.join(d, "s_height_mm.exr"), cv2.IMREAD_UNCHANGED)
    check("heights in mm once scaled", abs(float(h.max()) - 1.5) < 0.01)
    nr = cv2.imread(os.path.join(d, "s_normals.exr"), cv2.IMREAD_UNCHANGED)
    check("normals keep Z in the blue slot", abs(float(nr[5, 5, 0]) - 1) < 1e-6 and len(fs) == 4)

    print("\nLayered USD")
    u = os.path.join(d, "usd")
    us = M.usd_layered(u, "s", depth=depth, normals=n, colour=colour, mm_per_px=0.1, solved=solved,
                       record=dict(note='a "quoted" value'))
    try:
        from pxr import Usd
        st = Usd.Stage.Open(us[0])
        layers = [os.path.basename(x.identifier) for x in st.GetUsedLayers()]
        check(f"the stage opens with its layers {layers}", "s_record.usda" in layers and "s_surface.usda" in layers)
        p = st.GetDefaultPrim()
        check("a colour variant, median by default", p.GetVariantSet("colour").GetVariantSelection() == "median")
        tex = lambda: [q for q in st.Traverse() if q.GetName() == "Colour"][0].GetAttribute("inputs:file").Get().path
        was = tex()
        p.GetVariantSet("colour").SetVariantSelection("solved")
        check("the solved variant swaps the texture", tex() != was and "reflectance" in tex(), tex())
    except ImportError:
        print("  --   (no pxr here)")

    print("\nOpenUSD: measured and OpenPBR materials, lights, scale, marks")
    from helio.application import annotations as A
    u2 = os.path.join(d, "usd2")
    L = np.array([[0.7, 0.0, 0.7], [0.0, 0.7, 0.7], [-0.7, 0.0, 0.7]])
    us2 = M.usd_layered(u2, "s", depth=depth, normals=n, colour=colour, mm_per_px=0.1, solved=solved, lights=L,
                        z_mm_per_unit=0.02, record=dict(note="x"), colour_check=dict(mean=1.8),
                        sections=[dict(name="A–B", a=(10, 30), b=(70, 30), copies=2)],
                        annotations=[A.make("rect", [(5, 5), (30, 25)], color="#00ff00", text="Stamp"),
                                     A.make("dot", [(40, 40)], text="Pin")])
    try:
        from pxr import Usd, UsdGeom, UsdLux, UsdUtils
        st = Usd.Stage.Open(us2[0])
        p = st.GetDefaultPrim()
        mat = p.GetVariantSet("material")
        check("a material variant: photograph, measured, OpenPBR", sorted(mat.GetVariantNames())
              == ["measured", "openpbr", "photograph"] and mat.GetVariantSelection() == "photograph")
        surf = st.GetPrimAtPath(p.GetPath().AppendChild("Surface"))
        from pxr import UsdShade
        bound = lambda: UsdShade.MaterialBindingAPI(surf).GetDirectBindingRel().GetTargets()[0].name
        got = []
        for v in ("measured", "openpbr", "photograph"):
            mat.SetVariantSelection(v)
            got.append(bound())
        check(f"…each binds its material ({got})", got == ["Measured", "OpenPBR", "Photograph"])
        meas = st.GetPrimAtPath(p.GetPath().AppendPath("Looks/Measured/Normals"))
        f = meas.GetAttribute("inputs:file").Get().resolvedPath
        ex = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        check("the measured normals are float EXR, read raw", ex is not None and ex.dtype == np.float32
              and meas.GetAttribute("inputs:sourceColorSpace").Get() == "raw")
        check("…and the heights drive displacement", st.GetPrimAtPath(p.GetPath().AppendPath(
              "Looks/Measured/Surface")).GetAttribute("inputs:displacement").HasAuthoredConnections())
        pbr = st.GetPrimAtPath(p.GetPath().AppendPath("Looks/OpenPBR/Surface"))
        check("OpenPBR through MaterialX nodes", pbr.GetAttribute("info:id").Get() == "ND_open_pbr_surface_surfaceshader"
              and pbr.GetAttribute("inputs:base_color").HasAuthoredConnections())
        lights = [q for q in st.Traverse() if q.IsA(UsdLux.DistantLight)]
        check(f"the capture's {len(lights)} lights are lights in the scene", len(lights) == 3)
        info = st.GetRootLayer().customLayerData.get("helio_scale", "")
        check("the scale and the colour check in the stage's metadata", '"mm_per_px": 0.1' in info
              and "millimetres: true heights from the normals" in info and '"mean": 1.8' in info)
        marks = [q for q in st.Traverse() if "Marks" in str(q.GetPath()) and q.GetTypeName() in ("BasisCurves", "Points")]
        sec = [q for q in marks if q.GetCustomDataByKey("name") == "A–B"]
        check(f"sections and notes as curves and points ({len(marks)})", len(marks) == 3 and sec
              and sec[0].GetCustomDataByKey("length") == "6.0 mm")
        pts = UsdGeom.BasisCurves(sec[0]).GetPointsAttr().Get()
        ys = [q[1] for q in pts]
        check("…lying on the surface, following its relief", max(ys) - min(ys) > 0)
        mx_problems = M.check_materialx(us2[0])
        check(f"the OpenPBR nodes are MaterialX's own, inputs and types right ({len(mx_problems)} problems)",
              not mx_problems or mx_problems == ["MaterialX is not installed here: not checked"], mx_problems)
        from helio.application.export_usd import check as usd_check
        passed, errors = usd_check(us2[0], arkit=False)
        check(f"OpenUSD's own checker passes it ({len(errors)} errors)", not errors, errors[:3])
    except ImportError:
        print("  --   (no pxr here)")

    print("\nParadata")
    fr = os.path.join(d, "f1.png")
    cv2.imwrite(fr, colour)
    from helio.domain.settings import ProcessingSettings
    out = os.path.join(d, "out.stl")
    open(out, "w").write("solid")
    p = P.build(record=dict(object=dict(id="Seal"), capture=dict(scale_mmpx=0.05),
                            software=dict(name="Helio", version="4")), images=[fr, "/gone/f2.png"],
                lights=np.eye(3), light_source="measured", settings=ProcessingSettings(), tone="stored",
                solved=False, check=dict(mean=1.8), results=[("stl", [out], ""), ("glb", [], "no")], base=d)
    check("each photograph with its checksum, a missing one without",
          len(p["frames"][0]["sha256"]) == 64 and p["frames"][1]["sha256"] is None)
    check("every setting", "normal_engine" in p["processing"] and "photo_tone" in p["processing"])
    check("each file with its role", p["outputs"][0]["role"] == "derivative: print"
          and len(p["outputs"]) == 1 and len(p["outputs"][0]["sha256"]) == 64)
    check("the colour said plainly", p["colour"]["colour_map"] == "median of the frames"
          and p["colour"]["input"] == "encoded files")
    js, txt = P.write(p, d, "s")
    check("JSON and a read-me", json.load(open(js, encoding="utf8"))["schema"] == P.SCHEMA
          and "out.stl" in open(txt, encoding="utf8").read())

    print("\nColour validation")
    good = C.synthetic_chart(900)[..., ::-1]
    res = C.validate([("Photograph", good), ("Render", np.clip(good.astype(int) + 25, 0, 255).astype(np.uint8)),
                      ("Blank", np.full((400, 400, 3), 128, np.uint8))])
    a, b, e = res["stages"]
    check(f"the chart through the stages ({a.get('mean')} then {b.get('mean')})",
          "mean" in a and "mean" in b and b["mean"] > a["mean"], res["change"])
    check("a stage with no chart says so", "problem" in e)
    check("grey patches out of order is no reading", not C.evaluate(np.random.default_rng(0).random((24, 3)))["reliable"])
    v = C.write_validation(res, os.path.join(d, "v"))
    check("a table and a summary", all(os.path.exists(x) for x in v) and "not a claim" in open(v[1], encoding="utf8").read())
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
