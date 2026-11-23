"""Publish's "Colour in the exports", and the object mask at full size.

* As Matte Relight: the exports' colour is the Matte Relight view's (as the screen draws it,
  at its light and the Colour brightness), full size and preview; the solve keeps its own colour.
* As photographed (the default): the exports read the solve's colour itself.
* The object mask of a 63-megapixel picture takes well under a second (it took 4 s, on the main
  thread, and the window stopped answering), and agrees with the mask found at full size.

    python tests/test_export_colour.py
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
    from helio.core import object_mask as OM

    print("The object mask at full size")
    H, W = 6687, 9479
    a = np.full((H, W, 3), 8, np.uint8)
    cv2.rectangle(a, (900, 700), (8500, 5900), (170, 150, 110), -1)
    t = time.time()
    m = OM.object_mask(a)
    took = time.time() - t
    check(f"a 63-megapixel picture in under a second ({took:.2f} s)", m is not None and took < 1.0)
    ref = OM._object_mask(a[::4, ::4])
    ref = cv2.resize(ref.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    check("full size, and the same object", m is not None and m.shape == (H, W)
          and float((m == ref).mean()) > 0.995)

    print("\nColour in the exports")
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.show()
    from tools import make_golden as G
    paths, L = G.render_stack(tempfile.mkdtemp(prefix="helio_colour_"))
    w.open_paths([os.path.dirname(paths[0])])
    w.project.set_lights(L, "file:golden.lp")
    w.project.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 600:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    solved = w.project.result.albedo_rgb_full
    check("As photographed by default: the exports read the solve's colour",
          w.shared.get("export_colour", "photo") == "photo" and w.publish_source().result.albedo_rgb_full is solved)
    w.go_workspace("Publish")
    pub = w.workspaces["Publish"]
    pub.export_colour.chosen.emit("matte")
    app.processEvents()
    from helio.application.render import MATTE_VIEW, matte_colour
    src = w.publish_source()
    want = matte_colour(w.render_data, w.engine_settings[MATTE_VIEW], w.display.colour_gain)
    got = np.asarray(src.result.albedo_rgb_full)
    check("As Matte Relight: the exports' colour is the Matte Relight view's, full size",
          got.shape == solved.shape and np.array_equal(got, want) and np.array_equal(src.data.albedo_rgb_full, want))
    check("…and its preview too", src.data.albedo_proxy is not None
          and np.array_equal(src.result.proxy["albedo"], src.data.albedo_proxy)
          and src.data.albedo_proxy.shape == w.render_data.albedo_proxy.shape)
    check("…while the solve keeps its own colour", w.project.result.albedo_rgb_full is solved
          and w.render_data.albedo_rgb_full is not got)
    w.close()
    app.processEvents()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
