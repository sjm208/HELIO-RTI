"""The 3D relief is shaded with the solve's own resolution, not the mesh's.

* detail_side: 4,500 px normally, 8,000 with 24 GB free, less on a machine short of memory;
* detail_textures: unit normals and colour area-averaged from the full-size solve, only when
  they would be sharper than the mesh;
* the Inspect panel asks for them after the mesh is shown and hands them to the 3D view.

    python tests/test_relief_detail.py
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
    from helio.application import viewer3d as V
    from helio.application import pipeline as P
    from helio.data_manager import DataManager
    from helio.domain.settings import ProcessingSettings
    from tools import make_golden as G

    GB = 2 ** 30
    print("How sharp")
    check("plenty of memory: 4,500 px", V.detail_side(16 * GB) == 4500 and V.detail_side(12 * GB) == 4500)
    check("a great deal of memory: 8,000 px", V.detail_side(32 * GB) == 8000 and V.detail_side(24 * GB) == 8000)
    check("a 16 GB machine with other programs open: 3,200; a tight one: 2,400",
          V.detail_side(8 * GB) == 3200 and V.detail_side(4 * GB) == 2400)

    G.H, G.W = 480, 640
    paths, L = G.render_stack(tempfile.mkdtemp(prefix="helio_rd_"))
    r = P.solve(P.SolveRequest(paths=paths, lights=L, settings=ProcessingSettings(photo_tone="linear")))
    D = DataManager()
    P.store(r, D)
    print("The textures")
    out = V.detail_textures(D, mesh_w=320, side=640)
    check("sharper than the mesh: made, at the solve's size here", out is not None and out[2] == (640, 480)
          and out[0].shape == (480, 640, 3) and out[1].shape == (480, 640, 3) and out[0].dtype == np.float32)
    check("unit normals", abs(np.linalg.norm(out[0], axis=2) - 1).max() < 1e-4)
    out2 = V.detail_textures(D, mesh_w=320, side=320)
    ref = cv2.resize(D.normals_full.astype(np.float32), (320, 240), interpolation=cv2.INTER_AREA)
    ref /= np.maximum(np.linalg.norm(ref, axis=2, keepdims=True), 1e-8)
    check("smaller side: the area-average of every pixel, renormalised",
          out2 is None or np.allclose(out2[0], ref, atol=1e-6))
    check("no sharper than the mesh already is: nothing made", V.detail_textures(D, mesh_w=640, side=4500) is None)
    check("and the long side is capped", V.detail_textures(D, mesh_w=100, side=400)[2] == (400, 300))
    D2 = DataManager()
    check("no solve, nothing", V.detail_textures(D2, 100) is None)

    print("In Inspect")
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    w = hw.MainWindow(appearance="dark")
    w.resize(1200, 800)
    w.show()
    w.open_paths([os.path.dirname(paths[0])])
    w.project.set_lights(L, "file:golden.lp")
    w.project.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.01)
    ins = w.workspaces["Inspect"]
    Dw = w.render_data

    class Stub:
        data = {"w": 320}
        got = None

        def isVisible(self):
            return False

        def set_detail(self, n, a):
            Stub.got = (n, a)
    ins.relief, ins._relief_of = Stub(), Dw
    ins._sharpen_relief(Dw)
    w.wait_background()
    check("the panel asks for the sharper shading and hands it to the 3D view",
          Stub.got is not None and Stub.got[0].shape[:2] == (Dw.h, Dw.w) and Stub.got[1].shape[:2] == (Dw.h, Dw.w))
    Stub.got = None
    Stub.data = {"w": Dw.w}
    ins._sharpen_relief(Dw)
    w.wait_background()
    check("a mesh already as sharp as the solve asks for nothing", Stub.got is None)
    ins.relief = None
    ins._sharpen_relief(Dw)                       # no 3D view (no OpenGL): nothing happens
    check("and no 3D view is no error", True)
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
