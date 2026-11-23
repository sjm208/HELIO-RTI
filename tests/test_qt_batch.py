"""File › Batch Process Folders (the classic Overnight Batch).

Offscreen.  The shared scan and calibration chain on folders with known
contents: only subfolders with 3+ images count; a shared light file in the
parent is found; an object's own light file wins; an unreadable one is
logged and skipped, never leaving the last object's lights in place; a
.txt that is not a light file is ignored.  Then the window: three objects
solved and exported into their own folders, with a log; and a run stopped
part-way.

    python tests/test_qt_batch.py
"""
import os
import shutil
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
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from helio.application import batch as B
    from helio.io.sidecars import write_lp
    from tools.make_golden import render_stack
    print("=" * 60)
    print("Batch processing")
    print("=" * 60)

    parent = tempfile.mkdtemp(prefix="helio_batch_")
    os.makedirs(os.path.join(parent, "obj_a"))
    paths, L = render_stack(os.path.join(parent, "obj_a"))
    for name in ("obj_b", "obj_c"):
        shutil.copytree(os.path.join(parent, "obj_a"), os.path.join(parent, name))
    os.makedirs(os.path.join(parent, "too_few"))
    for p in paths[:2]:
        shutil.copy(p, os.path.join(parent, "too_few"))
    write_lp(os.path.join(parent, "lights.lp"), L, paths)
    with open(os.path.join(parent, "obj_b", "dome.lp"), "w") as fh:
        fh.write("this is not a light file\n")
    with open(os.path.join(parent, "obj_c", "notes.txt"), "w") as fh:
        fh.write("1 2 3\n4 5 6\n7 8 9\n")
    L_own = L.copy()
    L_own[:, 2] *= 0.9
    own = os.path.join(parent, "obj_a", "obj_a.lp")
    write_lp(own, L_own / np.linalg.norm(L_own, axis=1, keepdims=True), paths)

    print("\nScan and calibration")
    found, shared = B.scan(parent)
    check("three object folders (the one with 2 images left out), the shared light file found",
          [os.path.basename(f) for f, _n in found] == ["obj_a", "obj_b", "obj_c"]
          and os.path.basename(shared or "") == "lights.lp", (found, shared))

    def no_ball(_paths):
        return None
    imgs = B._captures(os.path.join(parent, "obj_a"))
    La, src_a, kind_a = B.calibration_for(os.path.join(parent, "obj_a"), imgs, shared_lp=shared,
                                          detect=no_ball)
    check("an object's own light file comes first", kind_a == "file" and "obj_a.lp" in src_a
          and np.allclose(La[0], (L_own / np.linalg.norm(L_own, axis=1, keepdims=True))[0], atol=1e-4))
    log = []
    Lb, src_b, kind_b = B.calibration_for(os.path.join(parent, "obj_b"), imgs, shared_lp=shared,
                                          detect=no_ball, log=log)
    check("an unreadable one is logged and skipped: the shared file, not the last object's lights",
          kind_b == "file" and "lights.lp" in src_b and any("dome.lp" in ln for ln in log)
          and np.allclose(Lb, L / np.linalg.norm(L, axis=1, keepdims=True), atol=1e-4), (src_b, log))
    Lc, src_c, kind_c = B.calibration_for(os.path.join(parent, "obj_c"), imgs, shared_lp=None,
                                          detect=no_ball)
    check("notes.txt is not a light file; with nothing else, uncalibrated", Lc is None and kind_c == "pca",
          src_c)
    Ls, src_s, kind_s = B.calibration_for(os.path.join(parent, "obj_c"), imgs,
                                          detect=lambda p: L, shared_lp=shared)
    check("a ball found in the folder comes before the shared file", kind_s == "sphere", src_s)
    Lr, src_r, kind_r = B.calibration_for(os.path.join(parent, "obj_c"), imgs, roi=(10, 10, 5),
                                          detect=no_ball, shared_lp=shared)
    check("the drawn region failing falls to uncalibrated, as the classic (not the shared file)",
          kind_r == "pca" and "drawn region" in src_r, src_r)

    print("\nThe window")
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1280, 860)
    w.show()
    acts = w.registry.actions
    check("File › Batch Process Folders is available (it was classic-only)", acts["file.batch"].isEnabled())
    dlg = w.show_batch()
    dlg.choose_parent(parent)
    check("the folders are listed and ticked; the shared light file is picked up",
          dlg.folders.count() == 3 and len(dlg.chosen_folders()) == 3
          and os.path.basename(dlg.shared_lp) == "lights.lp")
    dlg.chk_auto.setChecked(False)
    for k, c in dlg.export_chk.items():
        c.setChecked(k in ("heatmap", "gis_dem", "gis_hillshade", "sfs"))
    out = tempfile.mkdtemp(prefix="helio_batch_out_")
    dlg.choose_out(out)
    p = w.project
    t0 = time.time()
    dlg.start()
    w.wait_background()
    app.processEvents()
    res = dlg.result
    check("three objects solved and exported", res is not None and res["ok"] == 3,
          None if res is None else res["lines"])
    a = res["objects"].get("obj_a", {}) if res else {}
    names = [os.path.basename(f) for f in a.get("files", [])]
    check("…each into its own folder: heatmap, DEM package, hillshade, eight raking renders",
          a and a["out"] == os.path.join(out, "obj_a") and not a["failed"]
          and any("heatmap" in n.lower() for n in names)
          and any(n.endswith("_Hillshade_MultiDir.tif") for n in names)
          and sum("_SfS_" in n for n in names) == 8, (names, a.get("failed")))
    check("…obj_a with its own lights, obj_b with the shared file (its own unreadable)",
          "obj_a.lp" in res["objects"]["obj_a"]["calibration"]
          and "lights.lp" in res["objects"]["obj_b"]["calibration"])
    logtxt = open(res["log"]).read() if res and res["log"] else ""
    check("a log of every object and its calibration", "OK    obj_c" in logtxt and "dome.lp" in logtxt,
          logtxt[-400:])
    check("the open project is untouched", p.result is None and not p.images)
    print(f"  ({time.time() - t0:.0f} s)")

    calls = []

    def cancel_on_second():
        calls.append(1)
        if len(calls) > 1:
            from helio.application.jobs import Cancelled
            raise Cancelled()
    r2 = B.run([os.path.join(parent, n) for n in ("obj_a", "obj_b")], ["x"], out,
               solve=lambda *a: None, export=lambda *a: [], calibrate=lambda f, i, lg: (None, "pca", "pca"),
               check=cancel_on_second)
    check("stopping ends the run at the next object, and the log says so",
          r2["ok"] == 0 and any("STOP" in ln for ln in r2["lines"]), r2["lines"])
    dlg.close()
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
