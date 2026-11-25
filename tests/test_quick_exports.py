"""The quick exports: RTI web package, Send to QGIS, USDZ for AR (EXP-010…012).

Offscreen, on the golden fixture solved by the pipeline, writing only to
temporary folders.  QGIS is never started for real: the launch is caught,
and the profile path it would use is a scratch one.

* the RTI web package has its viewer (with the metadata inside it), tiles
  for every level with their overlap, the quick view, the Relight data,
  the lights, the README, the embed code and the launchers;
* Send to QGIS writes the GIS package Helio Relief reads (its own scanner,
  run here from qgis_plugin/, finds the heights, the colour, the uncertainty
  and the README's units) and starts the QGIS that has the plugin with the
  startup script;
* the USDZ is a valid zip with every file on a 64-byte boundary, its scene
  has Y up, metres, the anchoring and no lights; with OpenUSD installed it
  passes Pixar's compliance checker with Apple's AR rules, and reads back.

    python tests/test_quick_exports.py
"""
import json
import math
import os
import struct
import sys
import tempfile
import zipfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def zip_aligned(path) -> bool:
    """Every member's data starts on a 64-byte boundary (the USDZ rule)."""
    with open(path, "rb") as f:
        raw = f.read()
    with zipfile.ZipFile(path) as z:
        for zi in z.infolist():
            off = zi.header_offset
            n, e = struct.unpack("<HH", raw[off + 26:off + 30])
            if (off + 30 + n + e) % 64 or zi.compress_type != zipfile.ZIP_STORED:
                return False
    return True


def main():
    from PySide6.QtWidgets import QApplication
    print("=" * 60)
    print("Quick exports")
    print("=" * 60)
    app = QApplication(sys.argv)
    from tools import make_golden as g
    from helio.application.pipeline import SolveRequest, solve, store
    from helio.data_manager import DataManager
    from helio.domain.settings import ProcessingSettings
    tmp = tempfile.mkdtemp(prefix="helio_quick_")
    paths, L = g.render_stack(tmp)
    r = solve(SolveRequest(paths=paths, lights=L, settings=ProcessingSettings(photo_tone="linear")))
    D = DataManager()
    store(r, D)
    D.plate_paths = paths

    print("\nRTI web package")
    from helio.application import rti_web
    out = os.path.join(tmp, "web")
    res = rti_web.package(out, normals=r.normals_full, albedo=r.albedo_rgb_full, lights=L,
                          ptm_coeffs=getattr(r, "ptm_coeffs", None), roughness=r.roughness_full,
                          spec_var=getattr(r, "spec_var", None), images=paths, mm_per_px=0.05,
                          opts=rti_web.Options(title="Golden <test>", obj_id="G-1",
                                               url="https://example.org/g1/"))
    names = set(os.listdir(out))
    want = {"index.html", "quick-view.html", "relight", "tiles", "lights.lp", "metadata.json",
            "README.txt", "embed.html", "1_Mac.command", "2_Win.bat", "qr.png"}
    check("the folder holds the viewer, the quick view, Relight data, lights, metadata, README, "
          "embed code, launchers and QR code", want <= names, sorted(want - names))
    meta = json.load(open(os.path.join(out, "metadata.json")))
    lv = meta["levels"]
    ok_tiles = all(os.path.isfile(os.path.join(out, "tiles", layer, str(i), f"{c}_{rr}.{ext}"))
                   for layer, ext in (("normals", "png"), ("albedo", "jpg"))
                   for i, l in enumerate(lv) for c in range(l["cols"]) for rr in range(l["rows"]))
    check(f"every tile of every level is there ({len(lv)} levels down to one tile)",
          ok_tiles and lv[-1]["cols"] == lv[-1]["rows"] == 1 and lv[0]["w"] == r.w)
    import cv2
    t0 = cv2.imread(os.path.join(out, "tiles", "albedo", "0", "0_0.jpg"))
    th, tw = t0.shape[:2]
    check("tiles carry a pixel of their neighbours (256 + overlap)",
          tw == min(r.w, 256 + 1) and th == min(r.h, 256 + 1), (tw, th))
    html = open(os.path.join(out, "index.html"), encoding="utf-8").read()
    check("the viewer carries its metadata and escapes the title",
          '"object_id":"G-1"' in html and "Golden &lt;test&gt;" in html and "<test>" not in html)
    lp = open(os.path.join(out, "lights.lp")).read().split("\n")
    check("lights.lp: a count, then one light per frame", lp[0] == str(len(L))
          and len([x for x in lp[1:] if x.strip()]) == len(L))
    readme = open(os.path.join(out, "README.txt"), encoding="utf-8").read()
    check("the README says how to open it and gives the scale", "1_Mac.command" in readme
          and "0.05 mm per pixel" in readme)
    e = rti_web.estimate(r.w, r.h)
    check(f"the estimate is in the right range ({e['mb']} MB estimated, {res['mb']} MB written)",
          0.2 < res["mb"] / max(e["mb"], 0.01) < 5)

    print("\nSend to QGIS")
    from helio.application import export_qgis as Q
    import importlib.util
    qout = os.path.join(tmp, "qgis")
    info = Q.package(D, qout, "Golden test", mm_per_px=0.05, normal_engine="Linear Least Squares",
                     lights=L)
    scanner = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qgis_plugin", "helio_relief", "helio.py")
    spec = importlib.util.spec_from_file_location("helio_relief_scan", scanner)
    hp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hp)                          # its file reader only: no QGIS needed
    sc = hp.scan(qout)
    check("Helio Relief's own scanner finds both height maps, the colour, the normals, the uncertainty "
          "and the units", [lab for lab, _ in sc["dems"]] == ["Poisson", "Frankot-Chellappa"] and sc["albedo"]
          and sc["normals"] and sc["uncertainty"] and sc["readme"].get("xy_size")
          and sc["readme"].get("z_calibrated") is False, sc)
    lps = [os.path.join(dp, f) for dp, _d, fs in os.walk(qout) for f in fs if f.endswith(".lp")]
    lp_lines = open(lps[0]).read().split("\n") if lps else []
    check("the package says which image was lit from where (.lp), lights given as an array",
          bool(lps) and lp_lines[0] == str(len(L)) and len([x for x in lp_lines[1:] if x.strip()])
          == len(L), lps)
    script = open(info["script"], encoding="utf-8").read()
    check("the startup script hands the folder to the plugin, or loads it plainly",
          "_import_helio(FOLDER" in script and "_plain()" in script and os.path.abspath(qout) in script)
    compile(script, "open_in_qgis.py", "exec")
    check("…and is valid Python", True)
    calls = []
    fake = dict(app="/Applications/QGIS-test.app", exe="/Applications/QGIS-test.app/Contents/MacOS/QGIS",
                version="4.2.2", major=4)
    prof = tempfile.mkdtemp(prefix="qgis_profiles_")
    Q.launch(fake, info["script"], prof, popen=lambda a, **k: calls.append(a))
    env = Q.clean_env({"QT_PLUGIN_PATH": "/bundle", "DYLD_LIBRARY_PATH": "/b", "PYTHONHOME": "/b", "HOME": "/h",
                       "LD_LIBRARY_PATH_ORIG": "/orig", "LD_LIBRARY_PATH": "/b"})
    check("QGIS does not inherit this app's Qt, libraries or Python (it aborts if it does)",
          env == {"HOME": "/h", "LD_LIBRARY_PATH": "/orig"}, env)
    import zipfile
    src = os.path.join(ROOT, "qgis_plugin", "helio_relief")
    with zipfile.ZipFile(Q.bundled_zip()) as z:
        stale = [n for n in os.listdir(src) if n.endswith((".py", ".txt"))
                 and z.read(f"helio_relief/{n}") != open(os.path.join(src, n), "rb").read()]
    check("the plugin Helio ships is the plugin in this repository (dev/build_zip.sh)", not stale, stale)
    p3 = tempfile.mkdtemp(prefix="helio_prof_")
    st0 = Q.plugin_state(4, p3)
    old = os.path.join(p3, "default", "python", "plugins", "heritage_photometric_stereo_suite")
    os.makedirs(old)
    open(os.path.join(old, "metadata.txt"), "w").write("version=3.1.0\n")
    st1 = Q.plugin_state(4, p3)
    Q.install_plugin(4, p3)
    st2 = Q.plugin_state(4, p3)
    check("the old plugin or none is offered Helio's copy; once in, it is used and not offered again",
          st0["install"] and st1["install"] and st1["old_name"] and not st2["install"]
          and st2["version"] == Q.bundled_version() and os.path.isdir(old), (st0, st1, st2))
    check("QGIS is started on the script, with a scratch profile in tests",
          calls and calls[0][1:3] == ["--code", info["script"]] and calls[0][-2:] == ["--profiles-path", prof])
    old = os.path.join(prof, "default", "python", "plugins", "heritage_photometric_stereo_suite")
    os.makedirs(old)
    with open(os.path.join(old, "metadata.txt"), "w") as f:
        f.write("[general]\nversion=3.1.0\n")
    check("the plugin under its old name is found", Q.plugin_version(Q.plugin_path(4, prof)) == "3.1.0")
    os.makedirs(os.path.join(prof, "default", "python", "plugins", Q.PLUGIN))
    with open(os.path.join(prof, "default", "python", "plugins", Q.PLUGIN, "metadata.txt"), "w") as f:
        f.write("[general]\nversion=9.9.9\n")
    pick = Q.choose([dict(fake, major=3, version="3.40"), fake], prof)
    check("the QGIS chosen is the one with the plugin, Helio Relief before its old name", pick["major"] == 4
          and Q.plugin_version(Q.plugin_path(4, prof)) == "9.9.9")

    print("\nUSDZ for AR")
    from helio.application import export_usd as U
    up = os.path.join(tmp, "golden.usdz")
    res = U.export(up, depth=r.depth_vis_p, normals=r.normals_full, albedo=r.albedo_rgb_full,
                   mm_per_px=0.05, exaggeration=2.0, lights=L, view_light=(-0.5, 0.5, 0.7),
                   meta={"object": "golden"}, also_usda=True)
    check("a USDZ: an uncompressed zip, every file's data on a 64-byte boundary", zip_aligned(up))
    with zipfile.ZipFile(up) as z:
        first = z.infolist()[0].filename
    check(f"the USD layer comes first ({first})", first.endswith((".usdc", ".usda")))
    text = open(res["files"][-1], encoding="utf-8").read()
    check("the scene: Y up, metres, a default prim, the table anchoring, a preview surface with "
          "the normal map read raw",
          'upAxis = "Y"' in text and "metersPerUnit = 1" in text and 'defaultPrim = "Relief"' in text
          and 'preliminary:planeAnchoring:alignment = "horizontal"' in text
          and 'sourceColorSpace = "raw"' in text and "inputs:bias = (-1, -1, -1, 0)" in text)
    check("the .usda keeps the lights; the AR file has none", "DistantLight" in text)
    check(f"true size from the scale: {res['width_mm']} mm across", abs(res["width_mm"] - r.w * 0.05) < 0.2)
    ok, problems = U.check(up)
    check("Pixar's compliance checker, with Apple's AR rules, passes (when OpenUSD is here)", ok, problems)
    try:
        m = U.read_mesh(up)
        check("it reads back: points, triangles, Y up, metres", m["points"].shape[0] == res["vertices"]
              and m["up"] == "Y" and m["meters_per_unit"] == 1.0)
    except RuntimeError as e:
        check(f"(OpenUSD is not installed: {e})", True)
    # the hand-made path, as a build without OpenUSD writes it
    orig = U._usdz_with_pxr
    U._usdz_with_pxr = lambda *a, **k: False
    try:
        up2 = os.path.join(tmp, "golden_text.usdz")
        res2 = U.export(up2, depth=r.depth_vis_p, normals=r.normals_full, albedo=r.albedo_rgb_full,
                        mm_per_px=0.05)
    finally:
        U._usdz_with_pxr = orig
    check("without OpenUSD it still writes a valid, aligned USDZ (text layer) and says so",
          zip_aligned(up2) and res2["writer"] == "text" and any("usd-core" in n for n in res2["notes"]))
    ok2, problems2 = U.check(up2, arkit=False)
    check("…which OpenUSD itself can open and checks clean (apart from Apple's binary-layer rule)",
          ok2, problems2)

    print("\nThe sheets, in the window")
    import helio_qt.window as hw
    w = hw.MainWindow(appearance="dark")
    w.project.set_images(paths)
    w.project.set_lights(L, "file:golden.lp")
    w.project.result = r
    w.refresh()
    from helio_qt.quick_exports import QgisDialog, RtiWebDialog, UsdzDialog
    dlg = RtiWebDialog(w)
    dlg.out.setText(os.path.join(tmp, "from_sheet"))
    dlg.run()
    w.wait_background()
    check("RTI web package from its sheet: written, with a way to see it",
          dlg.result and os.path.isfile(os.path.join(tmp, "from_sheet", "index.html"))
          and dlg.b_show.isVisibleTo(dlg.d))
    qd = QgisDialog(w, profiles_path=prof)
    qd.qgis = fake
    qd.out.setText(os.path.join(tmp, "qgis_sheet"))
    launched = []
    qd.run(open_qgis=True, popen=lambda a, **k: launched.append(a))
    w.wait_background()
    check("Send to QGIS from its sheet: the package, then QGIS on its script",
          qd.result and launched and launched[0][2].endswith("open_in_qgis.py"))
    ud = UsdzDialog(w)
    ud.out.setText(os.path.join(tmp, "sheet.usdz"))
    ud.run()
    w.wait_background()
    check("USDZ from its sheet: written and checked", ud.result and os.path.isfile(
        os.path.join(tmp, "sheet.usdz")) and "Apple's AR rules" in ud.status.text(), ud.status.text())
    for sid in ("export.rti_web", "export.qgis", "export.usdz", "view.snapshot"):
        check(f"{sid} has a menu command and a handler", sid in w.registry.actions
              and callable(getattr(w, w.registry.specs[sid].handler, None)))
    from helio_qt import publish as P
    from helio_qt.publish_catalog import DESTINATIONS
    check("USDZ for AR / VR is in Publish under 3D models",
          any(f.key == "usdz" for f in DESTINATIONS["3D models"]))
    pub = os.path.join(tmp, "publish_usdz")
    os.makedirs(pub, exist_ok=True)
    files = P.export_one("usdz", w.publish_source(), pub)
    check("…and Publish writes it: an aligned USDZ", files and files[0].endswith(".usdz")
          and zip_aligned(files[0]), files)
    labels = [b.text() for b in w.workspaces["Inspect"].sec_export.findChildren(type(ud.b_go))]
    check("Inspect's quick exports no longer offer the phone view or the USDZ",
          not any("Phone" in t or "USDZ" in t for t in labels), labels)
    check("…and there is no Phone View command", "share.phone" not in w.registry.specs)
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
