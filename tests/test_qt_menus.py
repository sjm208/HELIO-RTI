"""Every menu item no other test reaches, chosen from its menu.

Offscreen.  Solves the baseline fixture, then triggers each item through the
registry (as the menu, ⌘K and the shortcut do) with the file dialogs and
message boxes answered, and checks what it did: the session saved and opened
again, photographs and lights imported, the quick exports (RTI web package,
USDZ, QGIS package) written, the colour check, the flat reference, every
normal solver chosen, the quality checks, the measure tools, the 3D items,
the view and window items, About.  An item that raises fails.

    python tests/test_qt_menus.py
"""
import importlib.util
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
os.environ["HELIO_PREFS"] = os.path.join(tempfile.mkdtemp(prefix="helio_prefs_"), "prefs.json")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = FAIL = 0
REACHED = set()


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
    print("Every menu item no other test reaches")
    print("=" * 60)
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMessageBox
    app = QApplication(sys.argv)
    import helio_qt.window as hw

    shown, opened, boxes = [], [], []
    answers = {}                                   # what the next file dialog returns

    def answer(kind):
        return answers.pop(kind, "")
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (answer("open"), ""))
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (answer("opens") or [], ""))
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (answer("save"), ""))
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: answer("dir"))
    for name in ("information", "warning", "critical", "about"):
        setattr(QMessageBox, name, staticmethod(
            lambda *a, _n=name, **k: boxes.append((_n, a[1] if len(a) > 1 else "")) or
            QMessageBox.StandardButton.Ok))
    QMessageBox.question = staticmethod(
        lambda *a, **k: boxes.append(("question", a[1] if len(a) > 1 else ""))
        or QMessageBox.StandardButton.Discard)
    QDialog.exec = lambda self: shown.append(type(self).__name__) or 0
    QDesktopServices.openUrl = staticmethod(lambda url: opened.append(url.toLocalFile()) or True)

    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    reg = w.registry

    def idle(limit=900.0):
        t0 = time.time()
        while w.project.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        w.wait_background()
        app.processEvents()

    def fire(sid):
        """The item as its menu chooses it; False when it raised."""
        REACHED.add(sid)
        try:
            reg.trigger(sid)
            app.processEvents()
            return True
        except Exception as e:                        # noqa: BLE001 - the failure is the finding
            import traceback
            traceback.print_exc()
            print(f"    {sid} raised {e!r}")
            return False

    spec = importlib.util.spec_from_file_location(
        "cb", os.path.join(ROOT, "tests", "fixtures", "synthetic_capture.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    _z, n_gt = cb.surface()
    L = cb.lights()
    src_dir = tempfile.mkdtemp(prefix="helio_qt_menus_")
    for i, lv in enumerate(L):
        img = np.clip(n_gt @ lv, 0, 1) * 235 + 10
        cv2.imwrite(os.path.join(src_dir, f"Tablet_{i:02d}.png"),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    lp = os.path.join(tempfile.mkdtemp(prefix="helio_qt_lp_"), "dome.lp")
    with open(lp, "w") as f:
        f.write(f"{len(L)}\n")
        for i, lv in enumerate(L):
            f.write(f"{os.path.join(src_dir, f'Tablet_{i:02d}.png')} {lv[0]:.6f} {lv[1]:.6f} {lv[2]:.6f}\n")
    out = tempfile.mkdtemp(prefix="helio_qt_menus_out_")
    p = w.project

    print("\nFile")
    answers["dir"] = src_dir
    check("Import Images… opens the folder chosen", fire("file.import") and len(p.images) == len(L),
          len(p.images))
    answers["open"] = lp
    check("Import Light Positions… reads the .lp", fire("file.import_lp") and p.lights is not None
          and np.allclose(np.asarray(p.lights), L, atol=1e-4))
    reg.trigger("process.run")
    idle()
    check("solved", p.result is not None)
    sess = os.path.join(out, "tablet.helio")
    answers["save"] = sess
    check("Save As… writes the session", fire("file.save_as") and os.path.exists(sess))
    answers["open"] = sess
    check("Open… opens it again", fire("file.open") and len(p.images) == len(L)
          and os.path.samefile(w.session_path, sess), w.session_path)
    if p.result is None:
        reg.trigger("process.run")
        idle()
    check("Export… goes to Publish", fire("file.export") and w.current == "Publish")

    print("\nFile › Quick Export")
    shown.clear()
    check("RTI Web Package… opens its sheet", fire("export.rti_web") and "QDialog" in shown
          and w._quick.__class__.__name__ == "RtiWebDialog", shown)
    q = w._quick
    q.out.setText(os.path.join(out, "Tablet_web"))
    q.run()
    idle()
    res = getattr(q, "result", None) or {}
    check("…and writes the package, its page ready for a phone",
          res and os.path.exists(res["index"])
          and "helio-phone" in open(res["index"], encoding="utf-8").read(), q.status.text())
    check("Show in Finder opens the folder just written",
          fire("file.reveal") and opened and os.path.samefile(opened[-1], res.get("dir", out)), opened)
    check("USDZ for AR… opens its sheet", fire("export.usdz")
          and w._quick.__class__.__name__ == "UsdzDialog")
    q = w._quick
    q.out.setText(os.path.join(out, "Tablet.usdz"))
    q.run()
    idle()
    usdz = os.path.join(out, "Tablet.usdz")
    check("…and writes a USDZ that passes Apple's AR rules",
          os.path.exists(usdz) and getattr(q, "result", {}).get("check", (False,))[0], q.status.text())
    check("Send to QGIS… opens its sheet", fire("export.qgis")
          and w._quick.__class__.__name__ == "QgisDialog")
    q = w._quick
    q.out.setText(os.path.join(out, "Tablet_gis"))
    q.run(open_qgis=False)
    idle()
    folder = (getattr(q, "result", None) or {}).get("folder", "")
    check("…and writes the GIS package (QGIS itself not started here)",
          folder and os.path.isdir(folder) and os.listdir(folder), q.status.text())

    print("\nEdit and Capture")
    QApplication.clipboard().clear()
    check("Copy puts the photograph on the clipboard", fire("edit.copy")
          and not QApplication.clipboard().image().isNull())
    shown.clear()
    check("Capture Details… opens", fire("capture.details") and "CaptureDetailsDialog" in shown, shown)
    check("Adjust the Ball Outline is offered only with a sphere",
          bool(reg.reason("capture.outline")) and fire("capture.outline"))
    boxes.clear()
    check("Check for Movement compares the frames", fire("capture.drift"))
    idle()
    check("…and says what it found", p.drift is not None and p.drift.get("verdict") in
          ("clean", "moved", "slight", "mixed") or bool(p.drift), p.drift)
    flat = os.path.join(tempfile.mkdtemp(prefix="helio_qt_flat_"), "flat")
    os.makedirs(flat)
    flats = []
    for i, lv in enumerate(L):
        img = np.full((96, 128), 10 + 235 * max(lv[2], 0), np.uint8)
        fp = os.path.join(flat, f"flat_{i:02d}.png")
        cv2.imwrite(fp, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        flats.append(fp)
    answers["opens"] = flats
    boxes.clear()
    check("Load a Flat Reference… reads the blank frames", fire("process.flat"), boxes)
    idle()
    answers["open"] = flats[0]
    check("Add Lights-Off Frame… keeps the photograph for the solve",
          fire("capture.ambient") and p.settings.ambient_frame == flats[0])
    check("Remove Lights-Off Frame forgets it", fire("capture.ambient_clear")
          and p.settings.ambient_frame == "")

    print("\nProcess")
    from helio.domain.registry import NORMAL_SOLVERS
    names = [sid.split(":", 1)[1] for sid in reg.specs if sid.startswith("process.solver:")]
    picked = []
    for name in names:
        fire(f"process.solver:{name}")
        picked.append(p.settings.normal_engine == name and reg.actions[f"process.solver:{name}"].isChecked())
    check(f"each normal solver ({len(names)}) can be chosen from the menu", all(picked) and names,
          [n for n, ok in zip(names, picked) if not ok])
    fire(f"process.solver:{names[0]}")
    check("Process Settings goes to Process", fire("process.settings") and w.current == "Process")
    ref = os.path.join(out, "reference_normals.png")
    nrm = ((p.result.normals_full * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(ref, cv2.cvtColor(nrm, cv2.COLOR_RGB2BGR))
    answers["open"] = ref
    shown.clear()
    check("Compare to a Reference… measures against the map chosen",
          fire("qc.compare") and shown, shown)
    answers["dir"] = out
    boxes.clear()
    before = set(os.listdir(out))
    ok = fire("qc.uncertainty")
    idle()
    new = set(os.listdir(out)) - before
    check("Uncertainty Maps… saves them (a calibrated solve has them)", ok and new, (boxes, new))

    print("\nMeasure")
    m = w.workspaces["Measure"]
    for sid, tool in (("measure.distance", "distance"), ("measure.profile", "profile"),
                      ("measure.zscale", "zscale")):
        check(f"{reg.specs[sid].text.rstrip('…')} picks its tool",
              fire(sid) and w.current == "Measure" and m.tool == tool, m.tool)
    from helio.application.colour_check import synthetic_chart
    chart = os.path.join(out, "chart.png")
    cv2.imwrite(chart, synthetic_chart())
    answers["open"] = chart
    shown.clear()
    check("Check Colour… finds the chart and shows the result",
          fire("measure.colour") and bool(p.colour_check) and bool(shown),
          (p.colour_check, shown, boxes[-1:]))

    print("\nView")
    w.go_workspace("Inspect")
    ins = w.workspaces["Inspect"]
    check("Histogram opens its section", fire("view.histogram"))
    boxes.clear()
    ins._loupe_at = None
    check("Save the Loupe's Region… asks for a spot first", fire("view.export_loupe")
          and boxes and "loupe" in boxes[-1][1].lower(), boxes)
    ins._loupe_at = (p.result.w // 2, p.result.h // 2)
    loupe = os.path.join(out, "loupe.tif")
    answers["save"] = loupe
    check("…and with one, saves its region at full resolution", fire("view.export_loupe")
          and os.path.exists(loupe) and cv2.imread(loupe, cv2.IMREAD_UNCHANGED) is not None, boxes)
    asked = []
    keep = w.open_record_dialog
    w.open_record_dialog = lambda style="turntable", relief=None: asked.append(style)
    ok = fire("view.relief_turntable") and fire("view.relief_sweep")
    w.open_record_dialog = keep
    check("Record a 3D Turntable / Light Sweep open Record video at each",
          ok and asked == ["turntable", "light"], asked)
    check("3D Model › Flattened shows the 3D relief", fire("view3d.open:flattened")
          and ins.centre_mode == "3d")
    ins.set_centre_mode("image")
    app.processEvents()
    rot = w.canvas_rotation
    check("Rotate Left turns the picture", fire("view.rotate_left")
          and w.canvas_rotation % 360 == (rot - 90) % 360)
    fire("view.rotate_reset") if "view.rotate_reset" in reg.specs else w._rotate(None)
    check("Turn View Right 1° turns it one degree, the slider with it", fire("view.rotate_right_fine")
          and w.canvas_rotation == 1 and abs(ins.rot_row.value() - 1.0) < 1e-6,
          (w.canvas_rotation, ins.rot_row.value()))
    check("Turn View Left 1° twice: minus one degree", fire("view.rotate_left_fine")
          and fire("view.rotate_left_fine") and w.canvas_rotation == -1, w.canvas_rotation)
    ins.rot_row.changed.emit("view_turn", 12.5, True)
    check("the Turn slider sets any angle", w.canvas_rotation == 12.5, w.canvas_rotation)
    fire("view.rotate_reset")
    check("Original Orientation: 0, the slider too", w.canvas_rotation == 0 and ins.rot_row.value() == 0)
    vp = w._viewport()
    z0 = vp.zoom()
    check("Zoom Out zooms out", fire("view.zoom_out") and vp.zoom() < z0, (z0, vp.zoom()))
    vis = w.left.isVisible()
    check("Show Sidebar hides and shows it", fire("view.sidebar") and w.left.isVisible() != vis
          and fire("view.sidebar") and w.left.isVisible() == vis)
    vis = w.insp_stack.isVisible()
    check("Show Inspector hides and shows it", fire("view.inspector")
          and w.insp_stack.isVisible() != vis and fire("view.inspector")
          and w.insp_stack.isVisible() == vis)

    print("\nWindow and Help")
    check("Minimize", fire("window.minimize"))
    w.showNormal()
    app.processEvents()
    check("Zoom fills the screen and back", fire("window.zoom") and fire("window.zoom"))
    boxes.clear()
    check("About Helio says its version", fire("app.about") and boxes
          and "Helio" in boxes[-1][1], boxes)

    print("\nWhat Helio 3 tests beside Helio Classic (tests_classic cannot run here)")
    # Helio 3 reaches these in test_qt_publish*, test_qt_help, test_qt_shell and test_qt_relief,
    # which need Classic beside them; here each is chosen from its menu with every file dialog
    # answered, and must not raise.
    spare = tempfile.mkdtemp(prefix="helio_menus_more_")
    for sid in ("export.rti_combined", "export.rti_3d", "export.hf_viewer", "file.export_now",
                "file.facsimile",
                "share.sketchfab", "share.qr", "share.embed", "share.manifest",
                "share.verify", "process.cancel", "ai.open:identify", "ai.open:report",
                "ai.open:caption", "view.relief_picture", "view.relief_flyover",
                "view3d.open:inverted", "view3d.open:heatmap", "view.fit",
                "view.appearance:light", "view.appearance:dark", "view.appearance:system",
                "view.scheme:graphite", "view.scheme:nightdome", "help.help", "help.search",
                "help.shortcuts", "help.welcome", "help.report"):
        answers.update(save=os.path.join(spare, sid.replace(":", "_").replace(".", "_")),
                       dir=spare, open="")
        ok = fire(sid)
        idle()
        check(f"{sid} runs", ok)
    answers.clear()
    if getattr(w, "phone_share", None) is not None:
        w.phone_share.stop()
        w.phone_share = None

    print("\nCoverage")
    import glob
    import re
    from helio_qt.actions import _specs
    others = "\n".join(open(f).read() for f in (glob.glob(os.path.join(ROOT, "test_qt_*.py"))
                                                               + glob.glob(os.path.join(ROOT, "tests", "test_qt_*.py")))
                       if not f.endswith("tests/test_qt_menus.py"))
    missed = []
    for s in _specs():
        if not s.handler:
            continue
        if (s.id in REACHED or f'"{s.id}"' in others or f"'{s.id}'" in others
                or re.search(r"\b%s\(" % re.escape(s.handler), others)):
            continue
        missed.append(s.id)
    check("every menu item is reached by a test", not missed, missed)

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
