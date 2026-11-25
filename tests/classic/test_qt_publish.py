"""Publish in the new interface writes real, valid files (UX-005 on EXP-001).

Runs the Qt app offscreen, solves the baseline fixture, then exports every
format that has moved to helio.application.exports (the same functions the
classic menu now calls) and checks each file with tests/test_exports.py's own
validators, plus what each format must contain.  Also: a light file the PCA
solver only estimated is refused while the rest still export; cancelling
stops the queue; nothing else can start while it runs.

    python tests/test_qt_publish.py
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
    import numpy as np
    import cv2
    print("=" * 60)
    print("Publish: exports from the new interface")
    print("=" * 60)
    import test_exports as TE
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(   # the next capture: don't save
        lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    def run_until_idle(limit=600.0):
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

    print("\nBefore solving")
    check("exporting says why it cannot run", w.registry.reason("file.export_now") != "")
    w.registry.trigger("process.run")
    run_until_idle()
    r = w.project.result
    check("solved", r is not None)

    print("\nChoosing formats")
    pub = w.workspaces["Publish"]
    w.go_workspace("Publish")
    pub._preset(next(pub.presets.item(i) for i in range(pub.presets.count())
                     if pub.presets.item(i).text() == "Share on the web"))
    check("the Share on the web preset picks the 3D web viewer and the RTI web package",
          pub.chosen() == ["glb", "rti_web"], pub.chosen())
    for k in ("stl", "lp", "ptm", "png_views"):
        pub._toggle(k)
    from helio_qt.publish_catalog import all_formats, available
    left = [f.key for f in all_formats() if not available(f)]
    if left:                                      # one still classic-only: cannot be ticked
        pub._toggle(left[0])
        check("a classic-only format cannot be ticked", left[0] not in pub.chosen())
    check("six formats chosen, and the next step says so",
          sorted(pub.chosen()) == ["glb", "lp", "png_views", "ptm", "rti_web", "stl"]
          and w.primary.text() == "Export 6 formats…", (pub.chosen(), w.primary.text()))

    print("\nClear, and presets of your own")
    names = {}
    from PySide6.QtWidgets import QInputDialog
    real_get = QInputDialog.getText
    QInputDialog.getText = staticmethod(lambda *a_, **k: (names.get("next", ""), True))
    try:
        names["next"] = "My web and print"
        pub.save_preset()
    finally:
        QInputDialog.getText = real_get
    texts = [pub.presets.item(i).text() for i in range(pub.presets.count())]
    check("Save as preset keeps the ticked formats under your name, in the list",
          "My web and print" in texts and sorted(pub.own_presets()["My web and print"])
          == ["glb", "lp", "png_views", "ptm", "rti_web", "stl"], texts)
    pub.clear_selection()
    check("Clear unticks every format", pub.chosen() == [] and not any(c.property("selected") == "true" for c in pub.cards)
          and not pub.btn_clear.isEnabled())
    pub._preset(next(pub.presets.item(i) for i in range(pub.presets.count())
                     if pub.presets.item(i).text() == "My web and print"))
    check("…and your preset ticks them all again",
          sorted(pub.chosen()) == ["glb", "lp", "png_views", "ptm", "rti_web", "stl"], pub.chosen())
    pub.presets.setCurrentRow(texts.index("Museum archive"))
    check("a built-in preset cannot be deleted", not pub.btn_del_preset.isEnabled())
    pub.presets.setCurrentRow(texts.index("My web and print"))
    pub.delete_preset()
    pub.presets.setCurrentRow(-1)
    check("your own can, and is gone from the list",
          "My web and print" not in [pub.presets.item(i).text() for i in range(pub.presets.count())]
          and "My web and print" not in pub.own_presets())

    print("\nThe settings beside the formats")
    from helio.application.exports import mesh_estimate
    before_sel = list(pub.selected)
    before_step = w.shared.get("export_step", 1)
    pub.clear_selection()
    check("no mesh ticked: no 3D model settings", not pub.mesh_box.isVisibleTo(w))
    pub._toggle("ply")
    check("a mesh ticked: its density, height and edge trim, with the size it will be",
          pub.mesh_box.isVisibleTo(w) and "million points" in pub.mesh_estimate.text()
          and "PLY" in pub.mesh_estimate.text(), pub.mesh_estimate.text())
    pub.mesh_density.chosen.emit("2")
    check("…the density chosen here is the 3D settings' export density (one value)",
          w.shared["export_step"] == 2 and pub.mesh_density.current() == "2")
    e1 = mesh_estimate(8585, 11444, 1, 2.0, keys=["obj", "glb"])
    e2 = mesh_estimate(8585, 11444, 2, 2.0, keys=["obj", "glb"])
    check(f"a 98 MP binding at 1 : 1: {e1['points'] / 1e6:.1f} M points, OBJ about "
          f"{e1['bytes']['obj'] / 1e9:.0f} GB; 1 : 2 a quarter",
          85e6 < e1["points"] < 95e6 and 35e9 < e1["bytes"]["obj"] < 48e9
          and abs(e2["points"] / e1["points"] - 0.25) < 0.01)
    w.prefs.setValue("rti_web/creator", "S. Maloney")
    w.prefs.setValue("publish/licence", "CC BY-NC 4.0")
    pub.refresh()
    check("the credit and licence are shown, and reach the plate",
          "S. Maloney" in pub.credit_text.text() and "CC BY-NC 4.0" in pub.credit_text.text()
          and w._plate_facts()["licence"] == "CC BY-NC 4.0", pub.credit_text.text())
    check("the scale is stated where the exports are made", pub.scale_text.text() != "")
    w.model3d.set("export_step", 3)
    pub.refresh()
    check("a density the buttons do not offer (3, from the 3D settings) lights none of them",
          pub.mesh_density.current() == "", pub.mesh_density.current())
    pub.selected = before_sel                     # the six, for the export below
    pub._after_choice()
    w.model3d.set("export_step", before_step)

    print("\nExport")
    out = tempfile.mkdtemp(prefix="helio_qt_export_")
    w.publish_to(out)
    check("the queue runs on the job thread; other work waits",
          w.project.busy == "export" and w.registry.reason("process.run") != "")
    run_until_idle()
    rows = w.project.export_rows
    check("all six written", all(v[0] == "done" for v in rows.values()) and len(rows) == 6,
          {k: v[:2] for k, v in rows.items()})
    from helio_qt.publish import file_stem
    stem = file_stem(w.project.name)          # files are named for the project
    folder = os.listdir(out)

    def valid(fn, path):
        try:
            return True, fn(path)
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    glb = [f for f in rows["glb"][2] if f.endswith(".glb")]
    ok, why = valid(TE._validate_mesh, glb[0]) if glb else (False, "no .glb")
    check(f"GLB web viewer: the model loads ({why})", ok)
    html = [f for f in rows["glb"][2] if f.endswith(".html")]
    ok, why = valid(TE._validate_html, html[0]) if html else (False, "no index.html")
    check(f"...and its page is well formed ({why})", ok)
    stl = rows["stl"][2][0]
    ok, why = valid(TE._validate_stl, stl)
    check(f"STL: a complete binary STL ({why})", ok and os.path.basename(stl) == f"{stem}_3dprint.stl")
    import trimesh
    check("...and a watertight solid", trimesh.load(stl).is_watertight)
    lp = rows["lp"][2][0]
    from helio.io.lp import read_lp
    back = read_lp(lp)
    names = [ln.split()[0] for ln in open(lp).read().splitlines()[1:] if ln.strip()]
    check(".lp: reads back to the lights it was given, one line per image, named for it",
          back.shape == (12, 3) and np.allclose(back, L, atol=1e-6)
          and names == [os.path.basename(p) for p in w.project.images])
    web = rows["rti_web"][2]
    index = [f for f in web if f.endswith("index.html")]
    quick = [f for f in web if f.endswith("quick-view.html")]
    ok, why = valid(TE._validate_html, index[0]) if index else (False, "no index.html")
    ok2, why2 = valid(TE._validate_html, quick[0]) if quick else (False, "no quick-view.html")
    check(f"RTI web package: its page and its one-file quick view are well formed ({why}; {why2})",
          ok and ok2 and os.path.getsize(quick[0]) > 10000)
    ptm = rows["ptm"][2]
    c0 = cv2.imread([f for f in ptm if f.endswith("ptm_coeff_0_f32.tif")][0], cv2.IMREAD_UNCHANGED)
    check("PTM: 12 maps, and coefficient 0 is the solve's exactly",
          len(ptm) == 12 and np.array_equal(c0, r.ptm_coeffs[0].reshape(r.h, r.w).astype(np.float32)))
    tif = rows["png_views"][2][0]
    from helio.application.render import composite, render
    st = dict(w.engine_settings["Standard PBR"])          # its own light is in its settings
    want = composite("Standard PBR", render("Standard PBR", w.render_data, st, r.h, r.w, False),
                     w.render_data, st, r.h, r.w, False, view_mode="Combined Colour",
                     colour_gain=w.display.colour_gain)    # as the screen shows it
    got = cv2.imread(tif, cv2.IMREAD_COLOR)[..., ::-1]
    check("the current view: full resolution, the classic render and composite exactly",
          got.shape[:2] == (r.h, r.w) and np.array_equal(got, want))
    check(f"everything landed in the chosen folder ({sorted(folder)})",
          {f"{stem}_web_viewer", f"{stem}_3dprint.stl", f"{stem}.lp",
           f"{stem}_web", f"{stem}_ptm"} <= set(folder))
    check("the next step is Show in Finder, and the status says Exported",
          w.primary.text() == "Show in Finder" and w.status_text.text() == "Exported")
    snap = tempfile.mkdtemp(prefix="helio_snap_")
    w.save_view("image").save(os.path.join(snap, "view.jpg"))        # a name remembered as .jpg
    got_s = sorted(os.listdir(snap))
    im_s = cv2.imread(os.path.join(snap, "view.tif"), cv2.IMREAD_UNCHANGED) if got_s else None
    check("This view is saved as a .tif whatever the name typed, 8-bit",
          got_s == ["view.tif"] and im_s is not None and im_s.dtype == np.uint8, got_s)
    old_view = w.view
    w.set_view("Depth map")
    w.save_view("image").save(os.path.join(snap, "view.tif"))
    im_d = cv2.imread(os.path.join(snap, "view.tif"), cv2.IMREAD_UNCHANGED)
    check("…the Depth map too: 8-bit grey as on screen, not 32-bit",
          im_d is not None and im_d.dtype == np.uint8 and im_d.ndim == 2, None if im_d is None else im_d.dtype)
    w.set_view(old_view or "Standard PBR")

    print("\nImages: anaglyph and the moving-light GIF")
    pub.selected = []
    for k in ("anaglyph", "sweep_gif"):
        pub._toggle(k)
    out_i = tempfile.mkdtemp(prefix="helio_qt_export_")
    w.publish_to(out_i)
    run_until_idle()
    rows = w.project.export_rows
    from PIL import Image
    ana = cv2.imread(rows["anaglyph"][2][0], cv2.IMREAD_COLOR) if rows["anaglyph"][2] else None
    check("anaglyph: full size, with red and cyan views that differ",
          ana is not None and ana.shape[:2] == (r.h, r.w)
          and float(np.abs(ana[..., 2].astype(int) - ana[..., 1].astype(int)).mean()) > 1)
    gif = Image.open(rows["sweep_gif"][2][0]) if rows["sweep_gif"][2] else None
    check("moving-light GIF: 60 frames at 20 fps, and the frames differ",
          gif is not None and gif.n_frames == 60 and gif.info.get("duration") == 50
          and np.asarray(gif.convert("L")).std() > 0)

    print("\nReports, archive and data: E57, LIDO, ISO statistics, plate, calibration report")
    pub.selected = []
    for k in ("e57", "lido", "iso_stats", "plate", "calibration"):
        pub._toggle(k)
    out_r = tempfile.mkdtemp(prefix="helio_qt_export_")
    w.publish_to(out_r)
    run_until_idle()
    rows = w.project.export_rows
    check("all five written", all(v[0] == "done" for v in rows.values()) and len(rows) == 5,
          {k: v[:2] for k, v in rows.items()})
    e57 = rows["e57"][2][0] if rows["e57"][2] else ""
    try:
        import pye57
        _e = pye57.E57(e57)
        n_pts = int(_e.read_scan_raw(0)["cartesianX"].size)
    except ImportError:
        n_pts = int(open(e57, "rb").read(200).split(b"element vertex ")[1].split(b"\n")[0])
    from helio.application.exports import shared_defaults
    st = max(1, shared_defaults()["export_step"])            # every step-th pixel, as classic
    want_pts = len(range(0, r.h, st)) * len(range(0, r.w, st))
    check(f"E57: a point cloud of the surface ({n_pts:,} points, every {st}nd pixel)",
          n_pts == want_pts, f"{n_pts} vs {want_pts}")
    import xml.etree.ElementTree as ET
    lido = ET.parse(rows["lido"][2][0]).getroot()
    text = ET.tostring(lido, encoding="unicode")
    check("LIDO: well-formed, named for the object, with the real image count (KD-015)",
          w.project.name in text and "Image count: 12." in text)
    import csv as _csv
    iso = list(_csv.reader(open(rows["iso_stats"][2][0])))
    names = [row[0] for row in iso if row and row[0] in ("Sa", "Sq", "Ssk", "Sku", "Sp", "Sv", "Sz")]
    check("ISO 25178: all seven parameters", names == ["Sa", "Sq", "Ssk", "Sku", "Sp", "Sv", "Sz"])
    plate_img = cv2.imread(rows["plate"][2][0], cv2.IMREAD_COLOR)
    check(f"plate figure: one composed image ({plate_img.shape[1]}×{plate_img.shape[0]})",
          plate_img is not None and plate_img.shape[1] > 1500)
    rep = open(rows["calibration"][2][0]).read()
    check("calibration report: 12 measured lights and the solvers named",
          "lights       : 12" in rep and "chrome sphere or imported .lp" in rep
          and "Linear Least Squares" in rep)
    from helio.application.exports import lido_record
    odd = os.path.join(out_r, "odd.xml")
    lido_record(odd, "Box <3> & Sons", "LLS", "FFT", 4)
    check("an object name with < and & still makes valid XML (KD-015)",
          ET.parse(odd).getroot() is not None)

    print("\nThe publication sheet")
    pub.selected = []
    pub._toggle("pub_sheet")
    pub.refresh()
    check("ticking the publication sheet shows its choices", pub.sheet_box.isVisibleTo(w)
          and "views beside" in pub.sheet_text.text(), pub.sheet_text.text())
    ins_ = w.workspaces["Inspect"]
    meas_ = w.workspaces["Measure"]
    meas_.lines = [(r.w * 0.2, r.h * 0.8, r.w * 0.8, r.h * 0.8, "12.3 mm")]
    ins_.annotator.anns = [dict(type="rect", pts=[(r.w * 0.3, r.h * 0.3), (r.w * 0.6, r.h * 0.6)],
                                color="#FFD60A", width=3, layer="Notes", text="Centre")]
    w.set_sheet_prefs(views=["Standard PBR", "Surface Curvature (Scratches)", "PCA 1"],
                      detail_views=["Standard PBR", "Exaggerated Shading"], compass=True)
    w.sheet_shots = [("3D relief, 1", np.full((300, 400, 3), 40, np.uint8), "3d")]
    pub.refresh()
    check("…and says what it will hold: the views, a boxed detail, a measured line, a 3D picture",
          "3 views" in pub.sheet_text.text() and "1 boxed details" in pub.sheet_text.text()
          and "1 measured lines" in pub.sheet_text.text() and "1 added picture" in pub.sheet_text.text(),
          pub.sheet_text.text())
    check("the clay 3D picture is ticked by default", pub.chk_clay.isChecked())
    rv = ins_.relief
    if rv is not None and rv.gl_ok and rv.data is not None:
        keep = (rv.material, rv.cam.az, rv.cam.el, rv.cam.dist)
        sh = w._plate_facts()["sheet"]["shots"]
        check("…it comes first, made from the 3D relief, and leaves the 3D view as it was",
              sh and sh[0][0].startswith("Clay 3D") and sh[0][1].shape[:2] == (1600, 2400)
              and (rv.material, rv.cam.az, rv.cam.el, rv.cam.dist) == keep, [c for c, _ in sh])
    pic_dir = tempfile.mkdtemp(prefix="helio_qt_sheet_pic_")
    cv2.imwrite(os.path.join(pic_dir, "Raking photo.jpg"), np.full((3000, 4000, 3), 90, np.uint8))
    check("a picture from a file joins them, captioned by its name, and the heading says Pictures",
          w.add_sheet_pictures([os.path.join(pic_dir, "Raking photo.jpg")]) == 1
          and w._sheet_options()["shots"][-1][0] == "Raking photo"
          and max(w._sheet_options()["shots"][-1][1].shape[:2]) == 2000
          and w._sheet_options()["shots_heading"] == "Pictures")

    def fake_section(y):
        return dict(a_px=(r.w * 0.2, r.h * y), b_px=(r.w * 0.8, r.h * y),
                    dist=np.linspace(0, 10, 50), z=np.sin(np.linspace(0, 6, 50)), unit="mm",
                    length=10.0, length_unit="mm", rq=0.5, ra=0.4, rz=1.0, features=[])
    keep_prof, keep_of = getattr(ins_, "section_prof", None), ins_._relief_of
    ins_._relief_of = w.selected_result()[1]
    ins_.section_prof = fake_section(0.3)
    first = w.add_sheet_section("relief")
    again = w.add_sheet_section("relief")
    ins_.section_prof = fake_section(0.6)
    w.add_sheet_section("relief")
    ins_.section_prof = fake_section(0.8)                       # drawn, not added: still shown
    from helio_qt.publish import _sheet_sections
    secs = _sheet_sections(dict(section_prof=ins_.section_prof), w._sheet_options())
    pub.refresh()
    check("sections are kept one by one (not twice), and the one drawn now is shown too",
          first and not again and len(secs) == 3 and "3 sections" in pub.sheet_text.text(),
          (len(secs), pub.sheet_text.text()))
    ins_.section_prof, ins_._relief_of = keep_prof, keep_of
    out_s = tempfile.mkdtemp(prefix="helio_qt_sheet_")
    w.publish_to(out_s)
    run_until_idle()
    files = w.project.export_rows["pub_sheet"][2]
    pages = [f for f in files if f.endswith(".jpg")]
    first = cv2.imread(pages[0]) if pages else None
    check("written: A3 pages at 300 dpi, the light compass page, and one PDF",
          w.project.export_rows["pub_sheet"][0] == "done" and first is not None
          and first.shape[1] == 3508 and first.shape[0] >= 4961
          and any(f.endswith("_light_compass.jpg") for f in files)
          and any(f.endswith(".pdf") for f in files), files)
    from helio.application import pub_sheet as PSH
    alb = np.zeros((200, 300, 3), np.uint8)
    alb[40:160, 90:210] = 180                                  # an object on black cloth
    alb[80:100, 20:40] = 160                                   # a small prop beside it
    x1_, y1_, x2_, y2_ = PSH.object_box(alb)
    check("views are cut to the object: the largest bright region, not the prop beside it",
          85 <= x1_ <= 90 and 205 <= x2_ <= 215 and 35 <= y1_ <= 40 and 160 <= y2_ <= 165,
          (x1_, y1_, x2_, y2_))
    w.sheet_shots, w.sheet_sections = [], []
    meas_.lines = []
    ins_.annotator.anns = []

    print("\nArchive: heritage ZIP and processing record")
    pub.selected = []
    for k in ("heritage", "provenance"):
        pub._toggle(k)
    out_a = tempfile.mkdtemp(prefix="helio_qt_export_")
    w.publish_to(out_a)
    run_until_idle()
    rows = w.project.export_rows
    check("both written", all(v[0] == "done" for v in rows.values()) and len(rows) == 2,
          {k: v[:2] for k, v in rows.items()})
    ok, why = valid(TE._validate_zip, rows["heritage"][2][0]) if rows["heritage"][2] else (False, "none")
    import zipfile
    names = zipfile.ZipFile(rows["heritage"][2][0]).namelist() if ok else []
    check(f"heritage archive: a sound ZIP with a model and statistics ({why})",
          ok and any(n.endswith(".glb") for n in names) and any("iso" in n.lower() or "stat" in n.lower()
                                                                for n in names), names)
    import json
    rec = json.load(open([f for f in rows["provenance"][2] if f.endswith(".json")][0]))
    from helio.application.exports import PROVENANCE_REFERENCES
    check("processing record: the classic shape, this capture's facts",
          rec["capture"]["image_count"] == 12 and rec["normal_computation"]["solver"] ==
          "Linear Least Squares" and rec["references"] == PROVENANCE_REFERENCES
          and rec["calibration"]["light_matrix_shape"] == [12, 3]
          and any(f.endswith(".txt") for f in rows["provenance"][2]))

    print("\nGIS: terrain analysis maps and the QGIS plugin")
    w.project.scale_mm_per_px = 0.05
    pub.selected = []
    for k in ("gis_maps", "qgis"):
        pub._toggle(k)
    out_g = tempfile.mkdtemp(prefix="helio_qt_export_")
    w.publish_to(out_g)
    run_until_idle()
    rows = w.project.export_rows
    tifs = rows["gis_maps"][2]
    shapes = {cv2.imread(t, cv2.IMREAD_UNCHANGED).shape for t in tifs} if tifs else set()
    check("seven 32-bit maps at full resolution", len(tifs) == 7 and shapes == {(r.h, r.w)}, shapes)
    try:
        import rasterio
        tag = rasterio.open(tifs[0]).tags().get("HELIO_RES_MM")
        check(f"the resolution tag is the measured mm per pixel ({tag}) (KD-016)", tag == "0.050000")
    except ImportError:
        print("  skip resolution tag: rasterio not installed")
    ok, why = valid(TE._validate_zip, rows["qgis"][2][0]) if rows["qgis"][2] else (False, "none")
    check(f"QGIS plugin: a sound ZIP ({why})", ok)
    if rows["qgis"][2]:
        import zipfile
        with zipfile.ZipFile(rows["qgis"][2][0]) as z_:
            meta_ = z_.read("helio_relief/metadata.txt").decode()
        check("…it is Helio Relief, 4.0.0", "name=Helio Relief" in meta_ and "version=4.0.0" in meta_)
    w.project.scale_mm_per_px = None

    print("\nAn estimated light file is refused, the rest still export")
    w.open_paths([src_dir])
    w.registry.trigger("process.solver:Uncalibrated PCA")
    w.registry.trigger("process.run")
    run_until_idle()
    pub.selected = []
    for k in ("lp", "ptm"):
        pub._toggle(k)
    out2 = tempfile.mkdtemp(prefix="helio_qt_export_")
    w.publish_to(out2)
    run_until_idle()
    rows = w.project.export_rows
    check("the .lp is refused with the reason", rows["lp"][0] == "problem"
          and "estimate" in rows["lp"][1] and not os.path.exists(os.path.join(out2, f"{stem}.lp")))
    check("...and the PTM maps are still written", rows["ptm"][0] == "done" and len(rows["ptm"][2]) == 12)
    check("the status says some could not be written", w.status_text.text() == "Exported with problems")

    print("\nCancel")
    pub.selected = []
    for k in ("glb", "stl", "ptm", "rti_viewer"):
        pub._toggle(k)
    out3 = tempfile.mkdtemp(prefix="helio_qt_export_")
    w.publish_to(out3)
    w.registry.trigger("process.cancel")
    run_until_idle()
    rows = w.project.export_rows
    check("the queue stops between items and says so",
          not w.project.busy and "cancelled" in w.project.message.lower()
          and any(v[0] == "stopped" for v in rows.values()), {k: v[0] for k, v in rows.items()})

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
