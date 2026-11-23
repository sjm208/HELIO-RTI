"""Publish's 3D formats write real, sound meshes (EXM-001…012).

Offscreen.  Solves the baseline fixture, exports the GLB model, OBJ (with
its material and maps), PLY, point cloud, Sketchfab PBR package, 3D print
solid, multi-LOD package, displacement package and the fine-detail web
viewer, and checks each: meshes load; the watertight mesh is closed,
consistently wound and of positive volume; the print solid too (KD-033);
every LOD faces up (KD-034); the PBR ZIP passes a CRC test and holds its
maps; the displacement map is 16-bit; the viewer page is well formed.

    python tests/test_qt_publish_3d.py
"""
import importlib.util
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
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
    import cv2
    import numpy as np
    import trimesh
    print("=" * 60)
    print("Publish: 3D models")
    print("=" * 60)
    import test_exports as TE
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    def idle(limit=900.0):
        t0 = time.time()
        while w.project.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    spec = importlib.util.spec_from_file_location(
        "cb", os.path.join(ROOT, "docs", "pyside6-migration", "tools", "capture_baseline.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    z_gt, n_gt = cb.surface()
    L = cb.lights()
    src_dir = tempfile.mkdtemp(prefix="helio_qt_fixture_")
    for i, lv in enumerate(L):
        img = np.clip(n_gt @ lv, 0, 1) * 235 + 10
        cv2.imwrite(os.path.join(src_dir, f"img_{i:02d}.png"),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    w.open_paths([src_dir])
    w.project.set_lights(L, "file:fixture.lp")
    w.project.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    idle()
    check("solved", w.project.result is not None)
    pub = w.workspaces["Publish"]
    w.go_workspace("Publish")
    keys = ("glb_mesh", "obj", "ply", "point_cloud", "sketchfab", "print", "lod", "displacement",
            "hf_viewer", "facsimile")
    pub.selected = []
    for k in keys:
        pub._toggle(k)
    check("all ten can be chosen", sorted(pub.chosen()) == sorted(keys), pub.chosen())
    out = tempfile.mkdtemp(prefix="helio_qt_3d_")
    w.publish_to(out)
    idle()
    rows = w.project.export_rows
    check("all ten written", all(v[0] == "done" for v in rows.values()) and len(rows) == 10,
          {k: v[:2] for k, v in rows.items() if v[0] != "done"})

    def files(k, end=""):
        return [f for f in rows.get(k, ("", "", []))[2] if f.endswith(end)]

    def valid(fn, path):
        try:
            return True, fn(path)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    print("\nMeshes")
    glb = files("glb_mesh", ".glb")
    ok, why = valid(TE._validate_mesh, glb[0]) if glb else (False, "none")
    check(f"GLB model loads ({why})", ok)
    m = trimesh.load(glb[0], force="mesh") if glb else None
    if m is not None:          # the walls carry their own vertices (for their UVs): weld them
        m.merge_vertices(merge_tex=True, merge_norm=True)
    check("…a closed solid, consistently wound, of positive volume",
          m is not None and m.is_watertight and m.is_winding_consistent and m.volume > 0,
          None if m is None else (m.is_watertight, m.is_winding_consistent, m.volume))
    obj = files("obj", ".obj")
    names = {os.path.basename(f) for f in rows["obj"][2]}
    ok, why = valid(TE._validate_mesh, obj[0]) if obj else (False, "none")
    check(f"OBJ loads ({why}) with its material and maps",
          ok and any(n.endswith(".mtl") for n in names) and any("albedo" in n for n in names))
    ply = files("ply", ".ply")
    ok, why = valid(TE._validate_mesh, ply[0]) if ply else (False, "none")
    check(f"PLY mesh loads ({why})", ok)
    pc = trimesh.load(files("point_cloud", ".ply")[0]) if files("point_cloud", ".ply") else None
    check("the point cloud has a coloured point per grid vertex",
          pc is not None and len(pc.vertices) > 1000 and getattr(pc, "colors", None) is not None)
    zips = files("sketchfab", ".zip")
    ok, why = valid(TE._validate_zip, zips[0]) if zips else (False, "no zip")
    import zipfile
    znames = zipfile.ZipFile(zips[0]).namelist() if ok else []
    check(f"Sketchfab package: a sound ZIP ({why}) with the GLB and its maps",
          ok and any(n.endswith(".glb") for n in znames)
          and all(any(t in n for n in znames) for t in ("baseColor", "normal", "roughness", "ao")),
          znames)

    print("\nPackages")
    stl = files("print", ".stl")
    sm = trimesh.load(stl[0]) if stl else None
    check("3D print: a closed, outward-wound solid (KD-033)",
          sm is not None and sm.is_watertight and sm.is_winding_consistent and sm.volume > 0,
          None if sm is None else (sm.is_watertight, sm.is_winding_consistent, sm.volume))
    ext = sm.extents if sm is not None else (0, 0, 0)
    check(f"…100 mm across without a scale ({ext[0]:.1f} × {ext[1]:.1f} mm)",
          sm is not None and abs(max(ext[0], ext[1]) - 100) < 2)
    lods = files("lod", ".glb")
    ups = []
    for f in lods:
        lm = trimesh.load(f, force="mesh")
        ups.append(float(lm.face_normals[:, 2].mean()))
    check("multi-LOD: three GLB levels, every one facing up (KD-034)",
          len(lods) == 3 and all(u > 0.5 for u in ups), ups)
    disp = [f for f in files("displacement") if f.endswith("_displacement_16bit.tif")]
    d16 = cv2.imread(disp[0], cv2.IMREAD_UNCHANGED) if disp else None
    check("displacement package: a 16-bit map, base mesh and guide",
          d16 is not None and d16.dtype == np.uint16 and files("displacement", ".obj")
          and files("displacement", "README.txt"))
    html = files("hf_viewer", ".html")
    ok, why = valid(TE._validate_html, html[0]) if html else (False, "none")
    check(f"fine-detail viewer: a well-formed page ({why})", ok)

    print("\nFacsimile")
    fobj = files("facsimile", ".obj")
    fm = trimesh.load(fobj[0], force="mesh") if fobj else None
    H, W = w.project.result.h, w.project.result.w
    readme = open(files("facsimile", "README.txt")[0]).read() if files("facsimile", "README.txt") else ""
    check("Publish: OBJ relief, 16-bit displacement and README; facing up",
          fm is not None and files("facsimile", "_displacement.tif") and fm.face_normals[:, 2].mean() > 0.5)
    check("…without a scale, 0.1 mm/px and the README says it was assumed (KD-044)",
          fm is not None and abs(fm.extents[0] - (W - 1) * 0.1) < 0.6 and "NOT MEASURED" in readme,
          None if fm is None else fm.extents)
    w.project.scale_mm_per_px = 0.05
    dlg = w.show_facsimile()
    dlg.otype.setCurrentIndex(1)                     # printing plate
    dlg.fmt.setCurrentIndex(0)                       # STL
    dlg.warp.set_value(0)
    dlg.taper.set_value(0)
    dlg.res.setCurrentIndex(3)                       # every pixel
    fo = tempfile.mkdtemp(prefix="helio_fac_")
    dlg.export(fo)
    w.wait_background()
    app.processEvents()
    res = dlg.result
    pm = trimesh.load(res["files"][0]) if res else None
    check("the Facsimile window: a plate is a mirrored tool, at the measured scale",
          res and res["mirrored"] and pm is not None and abs(pm.extents[0] - (W - 1) * 0.05) < 0.1
          and abs(pm.extents[1] - (H - 1) * 0.05) < 0.1, None if pm is None else pm.extents)
    check("…X and Y no longer multiplied by the grid step (KD-050)",
          res and res["step"] == 1 and len(pm.vertices) == H * W)
    os.environ["HELIO_NO_QUICKLOOK"] = "1"
    dlg.preview()
    check("…and a Quick Look draft", os.path.exists(getattr(dlg, "preview_path", "")))
    dlg.close()

    print("\nSketchfab")
    import json as _json
    import urllib.error
    sent = []

    class _R:
        def read(self):
            return _json.dumps({"uid": "abc123"}).encode()

    sk = w.show_sketchfab()
    sk.token.setText("TOKEN")
    sk.opener = lambda req, t: (sent.append(req), _R())[1]
    sk.upload()
    w.wait_background()
    app.processEvents()
    check("upload: a GLB posted with the token, the model page returned",
          sent and sent[-1].get_header("Authorization") == "Token TOKEN" and b"glTF" in sent[-1].data
          and sk.url == "https://sketchfab.com/models/abc123", sk.status.text())
    calls = []

    def bad(req, t):
        calls.append(req)
        raise urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    sk.opener = bad
    sk.upload()
    w.wait_background()
    app.processEvents()
    check("…a certificate failure stops once; never a retry unchecked (KD-048)",
          len(calls) == 1 and "certificate" in sk.status.text().lower(), sk.status.text())
    sk.close()

    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
