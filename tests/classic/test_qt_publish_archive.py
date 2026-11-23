"""Publish's archive and data formats, and the manifest tools (HER, EXR-012, GEO-016…018).

Offscreen.  Solves the baseline fixture, exports the Zenodo and MorphoSource
packages, the CIDOC-CRM record, a TIFF with Dublin Core, the GIS DEM
package, the light vectors, the light cube and the engine stack, and checks
each: sound ZIPs with their metadata; MorphoSource's GLB not mirrored and
facing out (KD-039) with its lights in front (KD-036); the Dublin Core XMP
really in the TIFF (KD-038); the DEM's rasters, .lp and checksums; a cube of
one slice per frame registered to the solve.  Then a folder manifest is
written, verified, and catches a changed file.

    python tests/test_qt_publish_archive.py
"""
import importlib.util
import json
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
    import zipfile
    print("=" * 60)
    print("Publish: archive, data and manifests")
    print("=" * 60)
    import test_exports as TE
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    infos = []
    hw.QMessageBox.information = staticmethod(lambda *a, **k: infos.append(a))
    hw.QMessageBox.critical = staticmethod(lambda *a, **k: infos.append(a))
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
    p = w.project
    p.set_lights(L, "file:fixture.lp")
    p.update_settings(photo_tone="linear")
    p.capture_meta = {"title": "Seal <A> & B", "institution": "Test Museum"}
    w.registry.trigger("process.run")
    idle()
    r = p.result
    check("solved", r is not None)
    w.go_workspace("Publish")
    pub = w.workspaces["Publish"]
    keys = ("zenodo", "morphosource", "cidoc", "dublin_core", "geotiff", "light_vectors",
            "light_cube", "engine_stack")
    pub.selected = []
    for k in keys:
        pub._toggle(k)
    check("all eight can be chosen", sorted(pub.chosen()) == sorted(keys), pub.chosen())
    out = tempfile.mkdtemp(prefix="helio_qt_arc_")
    w.publish_to(out)
    idle()
    rows = p.export_rows
    check("all eight written", all(v[0] == "done" for v in rows.values()) and len(rows) == 8,
          {k: v[:2] for k, v in rows.items() if v[0] != "done"})

    def files(k, end=""):
        return [f for f in rows.get(k, ("", "", []))[2] if f.endswith(end)]

    def zip_ok(path):
        try:
            TE._validate_zip(path)
            return True
        except Exception as e:
            print("   ", e)
            return False

    print("\nDeposits")
    z = files("zenodo", ".zip")
    zn = zipfile.ZipFile(z[0]).namelist() if z and zip_ok(z[0]) else []
    check("Zenodo: a sound ZIP with its deposit metadata and checksums",
          {"zenodo.json", "datacite.json", "manifest.csv", "light_vectors.csv",
           "helio_provenance.json"} <= set(zn), zn)
    ms = files("morphosource", ".zip")
    msz = zipfile.ZipFile(ms[0]) if ms and zip_ok(ms[0]) else None
    names = msz.namelist() if msz else []
    check("MorphoSource: mesh, maps, manifest and Darwin Core metadata",
          any(n.endswith(".glb") for n in names) and any(n.endswith("_morphosource_metadata.json")
                                                        for n in names))
    tmp = tempfile.mkdtemp()
    glb = [n for n in names if n.endswith(".glb")]
    m = None
    if glb:
        msz.extract(glb[0], tmp)
        m = trimesh.load(os.path.join(tmp, glb[0]), force="mesh")
    up = float(m.face_normals[:, 2].mean()) if m is not None else -1
    check(f"…its mesh faces out of the relief ({up:.2f}) (KD-039)", up > 0.5)
    if m is not None:
        v = np.asarray(m.vertices)
        # the top row of the image has the largest y: north up, not mirrored
        top = v[: v.shape[0] // 50]
        check("…and is not mirrored: the top of the image is +y", top[:, 1].mean() > v[:, 1].mean())
    mf = [n for n in names if n.endswith("_iiif_manifest.json")]
    lights = []
    if mf:
        man = json.loads(msz.read(mf[0]))
        ann = man["items"][0]["annotations"][0]["items"]
        lights = [a["body"]["position"] for a in ann if a["body"].get("type") == "PointLight"]
    check("…with every light in front of the model (KD-036)",
          len(lights) == len(L) and all(q["z"] > 0 for q in lights))
    ci = json.load(open(files("cidoc", ".jsonld")[0])) if files("cidoc", ".jsonld") else {}
    check("CIDOC-CRM: a JSON-LD record with the capture's parameters",
          ci.get("@type") == "crm:E22_Human-Made_Object"
          and ci["crmdig:L11_had_output"]["helio:capture_parameters"]["light_count"] == len(L))
    from PIL import Image
    dc = files("dublin_core", ".tif")
    xmp = Image.open(dc[0]).tag_v2.get(700) if dc else b""
    xmp = xmp.decode("utf-8") if isinstance(xmp, (bytes, bytearray)) else str(xmp or "")
    check("Dublin Core: the XMP is in the TIFF (KD-038)",
          "<dc:identifier>" in xmp and "HELIO Pro" in xmp, xmp[:80])
    import xml.etree.ElementTree as ET
    body = xmp[xmp.index("<x:xmpmeta"): xmp.index("</x:xmpmeta>") + len("</x:xmpmeta>")] if xmp else ""
    try:
        ET.fromstring(body)
        well = True
    except Exception:
        well = False
    check("…and is well-formed XML", well)

    print("\nData")
    dem = rows["geotiff"][2]
    names = {os.path.basename(f) for f in dem}
    check("GIS DEM: rasters, README, .lp and a checksum manifest",
          any(n.endswith("_DEM_Poisson.tif") for n in names) and "README_GIS.txt" in names
          and any(n.endswith(".lp") for n in names) and any("MANIFEST" in n.upper() for n in names),
          sorted(names))
    poi = cv2.imread([f for f in dem if f.endswith("_DEM_Poisson.tif")][0], cv2.IMREAD_UNCHANGED)
    check("…the DEM is 32-bit at full resolution", poi.dtype == np.float32
          and poi.shape[:2] == (r.h, r.w))
    import csv
    lv = list(csv.reader(open(files("light_vectors", ".csv")[0])))
    check("light vectors read back to the solve's",
          lv[0] == ["Image_Index", "L_X", "L_Y", "L_Z"] and len(lv) == len(L) + 1
          and np.allclose(np.array([row[1:] for row in lv[1:]], float), r.light_matrix, atol=1e-5))
    import tifffile
    cube = tifffile.imread(files("light_cube", ".tif")[0]) if files("light_cube", ".tif") else None
    check("light cube: one slice per frame, registered to the solve",
          cube is not None and cube.shape[0] == len(L) and cube.shape[-2:] == (r.h, r.w),
          None if cube is None else cube.shape)
    check("…with the .lp beside it", bool(files("light_cube", ".lp")))
    from helio.domain.registry import RENDER_ENGINES
    stack = tifffile.TiffFile(files("engine_stack", ".tif")[0]) if files("engine_stack", ".tif") else None
    n_sl = len(stack.pages) if stack is not None else 0
    check(f"engine stack: one slice per view ({n_sl})", n_sl >= len(RENDER_ENGINES) - 2)

    print("\nManifests")
    folder = os.path.dirname(dem[0])
    hw.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: folder)
    w.registry.trigger("share.manifest")
    infos.clear()
    w.registry.trigger("share.verify")
    check("a folder's manifest verifies", infos and "match" in infos[-1][2], infos[-1:] or None)
    with open(os.path.join(folder, "README_GIS.txt"), "a") as f:
        f.write("changed\n")
    infos.clear()
    w.registry.trigger("share.verify")
    check("…and a changed file is caught", infos and "CHANGED" in infos[-1][2])

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
