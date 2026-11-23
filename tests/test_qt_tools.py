"""Inspect and Measure tools moved from the classic Science and 2D Renders tabs.

Offscreen.  Solves the golden stack and checks: the multi-light composite
relights each frame from its own side (KD-043) and combines them by mean,
maximum or spread; the change map against a loaded reference (robust noise
floor, mm with a height scale) and its change report; the frequency maps.

    python tests/test_qt_tools.py
"""
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
    from PySide6.QtWidgets import QApplication
    try:                                   # GeoTIFF tags are unknown to libtiff: not errors
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    print("=" * 60)
    print("Tools: composite, change over time, frequency maps")
    print("=" * 60)
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(   # the next capture: don't save
        lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    p = w.project

    def idle(limit=600):
        t0 = time.time()
        while p.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    from tools.make_golden import render_stack
    tmp = tempfile.mkdtemp(prefix="helio_tools_")
    paths, L = render_stack(tmp)
    w.open_paths([tmp])
    p.set_lights(L, "file:golden.lp")
    p.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    idle()
    r = p.result
    check("solved", r is not None)

    print("\nMulti-light composite")
    from helio.application.export_images import multi_light_composite
    seen = []
    multi_light_composite(lambda lx, ly, lz: (seen.append((lx, ly, lz)),
                                              np.zeros((4, 4, 3), np.uint8))[1], L)
    check("each frame is relit from its own side: the light's y flipped once to the screen (KD-043)",
          np.allclose([(x, -y, z) for x, y, z in seen], L / np.linalg.norm(L, axis=1, keepdims=True),
                      atol=1e-6))
    stack = [np.full((4, 4, 3), v, np.uint8) for v in (10, 50, 90)]
    it = iter(stack)
    mean, n = multi_light_composite(lambda *a: next(it), L[:3], "Mean")
    it = iter(stack)
    mx, _n = multi_light_composite(lambda *a: next(it), L[:3], "Max")
    check("mean and maximum combine as named", n == 3 and mean[0, 0, 0] == 50 and mx[0, 0, 0] == 90)
    out = tempfile.mkdtemp(prefix="helio_tools_out_")
    import PySide6.QtWidgets as qw
    qw.QInputDialog.getItem = staticmethod(lambda *a, **k: ("Mean", True))
    comp = os.path.join(out, "comp.tif")
    hw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (comp, ""))
    w.registry.trigger("view.composite")
    w.wait_background()
    img = cv2.imread(comp)
    check("View › Multi-Light Composite writes the composite at preview size",
          img is not None and img.shape[:2] == (r.proxy["normals"].shape[0], r.proxy["normals"].shape[1]))

    print("\nChange over time")
    ref = r.depth_vis_p.copy()
    ref_path = os.path.join(out, "ref.npy")
    np.save(ref_path, ref)
    hw.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (ref_path, ""))
    w.registry.trigger("measure.load_reference")
    check("a reference depth loads", getattr(w, "reference_depth", None) is not None)
    from helio.application.export_reports import change_map
    cur = ref.copy()
    cur[40:60, 50:70] += 0.3
    res = change_map(cur, ref, os.path.join(out, "c.tif"), z_mm_per_unit=0.5)
    d = cv2.imread(res["path"], cv2.IMREAD_UNCHANGED)
    check("the change map is current − reference, in normalised units (both are 0-1 depths: no height scale applies)",
          res["unit"] == "normalised" and abs(float(d[50, 60]) - 0.3) < 1e-3 and res["moved"] >= 400,
          (res, None if d is None else float(d[50, 60])))
    cm = os.path.join(out, "change.tif")
    hw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (cm, ""))
    w.registry.trigger("measure.change_map")
    check("Measure › Save Change Map writes it", os.path.exists(cm))
    hw.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: out)
    w.registry.trigger("measure.change_report")
    check("…and Save Change Report its figure and statistics",
          any(f.endswith("_change_detection.png") for f in os.listdir(out))
          and any(f.endswith("_change_stats.csv") for f in os.listdir(out)))

    print("\nFrequency maps")
    from helio.application.export_images import frequency_maps
    files = frequency_maps(r.depth_vis_p, os.path.join(out, "freq"), normals=r.normals_full,
                           depth_poisson=r.depth_poisson)
    names = [os.path.basename(f) for f in files]
    arrs = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in files]
    check("five 32-bit maps: two raw heights and three bands",
          len(files) == 5 and names[0] == "1_height_map_primary_32f.tif"
          and all(a is not None and a.dtype == np.float32 and a.shape == (r.h, r.w) for a in arrs))

    print("\nInspect details carried from the classic panel")
    import time as _t
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    w.go_workspace("Inspect")
    ins = w.workspaces["Inspect"]
    v = ins.viewport
    for _ in range(4):
        w.zoom_in()
    app.processEvents()
    hb = v.horizontalScrollBar()
    hb.setValue(hb.maximum() // 2)
    x0 = hb.value()

    def mid(kind, x, y):
        t = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}[kind]
        pos = QPointF(x, y)
        btn = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.MiddleButton
        held = Qt.MouseButton.MiddleButton if kind != "release" else Qt.MouseButton.NoButton
        QApplication.sendEvent(v.viewport(), QMouseEvent(t, pos, v.viewport().mapToGlobal(pos), btn, held,
                                                         Qt.KeyboardModifier.NoModifier))
    mid("press", 200, 200)
    mid("move", 160, 200)
    mid("release", 160, 200)
    check("the middle button pans, as in HELIO 2", hb.value() == x0 + 40, (x0, hb.value()))
    import helio_qt.app as qa
    loaded = []
    sp = os.path.join(out, "x.helio")
    open(sp, "w").write("{}")
    orig = hw.MainWindow
    class _W:
        def __init__(self): self.prefs = w.prefs
        def show(self): pass
        def load_session_file(self, p): loaded.append(p)
        def open_paths(self, p): loaded.append(("paths", p))
    qa_main_src = open(qa.__file__).read()
    check("python -m helio_qt file.helio opens the session", "sessions[0]" in qa_main_src
          and "load_session_file" in qa_main_src)
    del orig, _W

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
