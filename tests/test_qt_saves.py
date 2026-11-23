"""Saving from Inspect and the 3D relief, the names files get, and the
publication sheet's options.

Offscreen.  Solves the baseline fixture, then checks: files are named after
the object (its title, the session, the photographs' shared name) before the
folder; G opens the 2D light sweep, or Record video at its Light sweep when
the 3D relief shows; the 2D sweep writes a GIF and an MP4 at the height
asked; the Save menu beside Sweep; Record video's lengths, sizes, frame
rates, file type and direction; the 3D picture's sizes; the mesh densities
and their counts; the capture details written into saved pictures; the
publication sheet on A4, with a PNG, its own title and no score.

    python tests/test_qt_saves.py
"""
import importlib.util
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    print("=" * 60)
    print("Saving: names, light sweeps, 3D pictures, the sheet's options")
    print("=" * 60)
    from helio_qt.state import shared_stem
    check("photographs named Seal_01, Seal_02 share the name Seal",
          shared_stem(["/a/Seal_01.tif", "/a/Seal_02.tif"]) == "Seal")
    check("…a camera's IMG_ or DSC is no name for the object",
          shared_stem(["/a/IMG_0101.jpg", "/a/IMG_0102.jpg"]) == ""
          and shared_stem(["/a/DSC01.arw", "/a/DSC02.arw"]) == "")
    from helio_qt.publish import file_stem
    check("a title becomes a tidy file name", file_stem("Intaglio seal, limestone")
          == "Intaglio_seal_limestone", file_stem("Intaglio seal, limestone"))

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
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
        "cb", os.path.join(ROOT, "tests", "fixtures", "synthetic_capture.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    _z, n_gt = cb.surface()
    L = cb.lights()
    src_dir = os.path.join(tempfile.mkdtemp(prefix="helio_qt_fixture_"), "capture folder")
    os.makedirs(src_dir)
    for i, lv in enumerate(L):
        img = np.clip(n_gt @ lv, 0, 1) * 235 + 10
        cv2.imwrite(os.path.join(src_dir, f"Bronze_coin_{i:02d}.png"),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    w.open_paths([src_dir])
    p = w.project
    check("files are named after the photographs, not their folder",
          p.name == "Bronze_coin" and p.folder_name == "capture folder", p.name)
    p.set_lights(L, "file:fixture.lp")
    w.registry.trigger("process.run")
    idle()
    check("solved", p.result is not None)
    p.capture_meta = {"title": "Roman coin, bronze", "operator": "A. Tester",
                      "institution": "Test Museum"}
    check("…and after the title in Capture details", p.name == "Roman coin, bronze"
          and file_stem(p.name) == "Roman_coin_bronze")
    check("the default export folder is the object's", w.default_export_dir().endswith(
        "Roman_coin_bronze"), w.default_export_dir())

    print("\nThe 2D light sweep")
    ins = w.workspaces["Inspect"]
    w.go_workspace("Inspect")
    ins.set_centre_mode("image")
    d = w.save_sweep_gif()
    from helio_qt.sweep_dialog import SweepDialog
    check("G in the 2D view opens Save the light sweep", isinstance(d, SweepDialog))
    check("…at the light's height (80° at most: overhead, a sweep shows nothing) and bearing",
          abs(d.light_el.value() - min(80.0, ins.height_row.value())) < 0.51
          and abs(d.light_from.value() - ins.az_row.value()) < 0.51,
          (d.light_el.value(), ins.height_row.value(), d.light_from.value(), ins.az_row.value()))
    out = tempfile.mkdtemp(prefix="helio_saves_")
    d.light_el.set_value(20.0)
    st = d.settings()
    st.pop("fmt")
    gif = w.write_light_sweep(os.path.join(out, "sweep.gif"), wait=True, **st)
    from PIL import Image
    with Image.open(gif) as im:
        n_gif = im.n_frames
    check("…it writes a looping GIF, 20 frames a second, once round", n_gif == 80, n_gif)
    st.update(fps=30, seconds=3.0)
    mp4 = w.write_light_sweep(os.path.join(out, "sweep.mp4"), wait=True, **st)
    cap = cv2.VideoCapture(mp4)
    n_mp4 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    check("…and an MP4", n_mp4 == 90, n_mp4)
    from helio_qt import publish
    from helio.domain.view_controls import from_angles
    src = w.publish_source()
    draw = publish._sweep_draw(src)
    lo, hi = draw(*from_angles(90, 10)), draw(*from_angles(90, 80))
    check("the light's height changes the frames (it was fixed at 37°)",
          float(np.abs(lo.astype(float) - hi.astype(float)).mean()) > 2.0)
    d.close()
    acts = [a.text() for a in ins.light_save_menu.actions()]
    check("Save ▾ beside Sweep: the sweep, this view at full resolution, the loupe, 3D",
          len(acts) == 4 and any("full resolution" in a for a in acts)
          and ins.btn_light_save.isEnabled(), acts)

    print("\nThe full-resolution view, with the capture details in it")
    tif = os.path.join(out, "view.tif")
    hw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (tif, ""))
    w.snapshot()
    with Image.open(tif) as im:
        desc, artist, size = im.tag_v2.get(270, ""), im.tag_v2.get(315, ""), im.size
    check("This view at full resolution is the solve's full size",
          size == (p.result.w, p.result.h), size)
    check("…and carries the title and operator, as Classic's exports did",
          "Roman coin, bronze" in desc and artist == "A. Tester", (desc, artist))

    print("\nRecord video and the 3D picture")
    from helio_qt.record_dialog import PictureDialog, RecordDialog
    from helio.application.viewer3d import VIDEO_SIZES
    r = ins._ensure_relief()
    rd = RecordDialog(w, r, "light")
    rd.length.setCurrentIndex(rd.length.findData(6))
    rd.fps.setCurrentIndex(rd.fps.findData(60))
    rd.size.set_current("Portrait")
    rd.fmt.set_current("gif")
    rd.turn.set_current("ccw")
    st = rd.settings()
    check("Record video: 6 s at 60 a second, portrait, GIF, anticlockwise",
          st["seconds"] == 6 and st["fps"] == 60 and st["size"] == (1080, 1920)
          and st["fmt"] == "gif" and st["reverse"], st)
    check("…six sizes, from 720p to 4K, square and portrait", len(VIDEO_SIZES) == 6)
    check("…the light sweep starts where the light is", rd.light_from.isVisibleTo(rd)
          and abs(rd.light_from.value() - ins.az_row.value()) < 1e-6)
    rd.close()
    from helio.application.viewer3d import video_pose
    fw = [video_pose("light", i, 8, light_el=30)[3][0] for i in range(8)]
    bw = [video_pose("light", i, 8, light_el=30, reverse=True)[3][0] for i in range(8)]
    check("…anticlockwise goes the other way round", fw[1] == 45.0 and bw[1] == 315.0, (fw, bw))
    pd = PictureDialog(w, r)
    pd.size.set_current("8k")
    pd.bg.set_current("white")
    pd.fmt.set_current("tif")
    st = pd.settings()
    check("the 3D picture: up to 8K, on white, as TIFF",
          st == dict(long_side=7680, background="#FFFFFF", fmt="tif"), st)
    pd.close()
    ins.set_centre_mode("3d")
    app.processEvents()
    asked = []
    keep = w.open_record_dialog
    w.open_record_dialog = lambda style="turntable", relief=None: asked.append(style)
    w.save_sweep_gif()
    w.open_record_dialog = keep
    check("G with the 3D relief showing records the 3D light sweep", asked == ["light"], asked)

    print("\nMesh density")
    keys = list(ins.relief_mesh.buttons)
    check("seven densities, 2 : 1 to 1 : 8", keys == ["0.5", "1", "2", "3", "4", "6", "8"], keys)
    if r.data is not None:
        ins._relief_mesh_step("1")
        p1 = r.mesh_counts()[0]
        ins._relief_mesh_step("3")
        p3 = r.mesh_counts()[0]
        check("…each builds the mesh it says, and says how many points",
              r.mesh_used() == 3 and p1 > 8 * p3 and "M points" in ins.mesh_note.text(),
              (p1, p3, ins.mesh_note.text()))
    else:
        check("…the relief has no GL here: counted from the arrays instead",
              r.mesh_counts() == (0, 0))
    ins.chk_relief_wire.setChecked(True)
    check("Show the mesh draws its edges", r.wireframe)
    ins.chk_relief_wire.setChecked(False)
    ins.set_centre_mode("image")

    print("\nThe publication sheet's options")
    w.set_sheet_prefs(paper="A4", png=True, title="Coin, obverse", subtitle="Test hoard",
                      hide_score=True)
    w.go_workspace("Publish")
    pub = w.workspaces["Publish"]
    pub.selected = []
    pub._toggle("pub_sheet")
    pub.refresh()
    check("the sheet's choices show its options", pub.sheet_paper.current() == "A4"
          and pub.chk_png.isChecked() and not pub.chk_score.isChecked())
    sheet_dir = tempfile.mkdtemp(prefix="helio_sheet_")
    w.publish_to(sheet_dir, ["pub_sheet"])
    idle()
    got = sorted(os.listdir(sheet_dir))
    jpg = [f for f in got if f.endswith("_publication_sheet.jpg")]
    pngs = [f for f in got if f.endswith("_publication_sheet.png")]
    im = cv2.imread(os.path.join(sheet_dir, jpg[0])) if jpg else None
    check("an A4 sheet (2480 × 3508), with a PNG beside the JPEG and PDF",
          im is not None and im.shape[:2] == (3508, 2480) and pngs
          and any(f.endswith(".pdf") for f in got), (got, None if im is None else im.shape))
    check("…named after the object", bool(jpg) and jpg[0].startswith("Roman_coin_bronze"), got)

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
