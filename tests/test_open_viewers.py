"""The two web viewers asked for from Inspect (Combined RTI viewer, 3D viewer: fine detail) open in the
browser when they are written; an export from Publish does not.

    python tests/test_open_viewers.py
"""
import os
import sys
import tempfile

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


class Job:
    def __init__(self, result):
        self.result = result


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import helio_qt.window as hw
    w = hw.MainWindow()
    opened = []
    w.reveal = lambda path: opened.append(path)
    asked = []
    w.publish_to = lambda folder, keys=None, open_when_done=False: asked.append((folder, keys, open_when_done))
    hw.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "/tmp/out")
    w.export_hf_viewer()
    w.export_rti_combined()
    check("both Inspect buttons ask for the page to be opened when done",
          asked == [("/tmp/out", ["hf_viewer"], True), ("/tmp/out", ["rti_combined"], True)], asked)
    del w.publish_to                                                   # the real one again

    w.project.set_export_rows = lambda rows: None
    w.project.finish = lambda msg="": None
    w._open_when_done = ("hf_viewer",)
    w._export_done(Job([("hf_viewer", ["/o/model.glb", "/o/index_hf_normals.html", "/o/n.png"], "")]))
    check("the page is opened, not the model or the picture", opened == ["/o/index_hf_normals.html"], opened)
    w._export_done(Job([("hf_viewer", ["/o/index_hf_normals.html"], "")]))
    check("and only the once", len(opened) == 1, opened)
    w._open_when_done = ("rti_combined",)
    w._export_done(Job([("rti_combined", [], "Process the images first")]))
    check("nothing opens when it could not be written", len(opened) == 1, opened)
    w._open_when_done = ()
    w._export_done(Job([("glb", ["/o/index.html"], "")]))
    check("an export from Publish does not open anything", len(opened) == 1, opened)
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
