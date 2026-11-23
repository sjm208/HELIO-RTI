"""Annotating the view in Inspect (the classic Annotation Layer and Scholar Annotations).

Offscreen.  Solves the golden stack, then draws through the view's own mouse
events and the tool's methods: an arrow lands in image pixels; a polygon
closes on double-click; numbered points count; Delete and Undo remove; a
measurement is in pixels without a scale and millimetres with one (KD-044);
layers hold their annotations and can be hidden (KD-045); the set saves and
reloads; the annotated view exports with a well-formed SVG whose arrows have
heads; the scholar set writes its panels, JSON and SVG by layer.

    python tests/test_qt_annotate.py
"""
import json
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
    import xml.etree.ElementTree as ET
    import cv2
    import numpy as np
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    print("=" * 60)
    print("Annotate the view")
    print("=" * 60)
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    import helio_qt.annotate as an
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
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
    tmp = tempfile.mkdtemp(prefix="helio_ann_")
    _paths, L = render_stack(tmp)
    w.open_paths([tmp])
    p.set_lights(L, "file:golden.lp")
    p.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    idle()
    r = p.result
    w.go_workspace("Inspect")
    app.processEvents()
    ins = w.workspaces["Inspect"]
    A = ins.annotator
    v = ins.viewport
    check("solved, and the Annotate section is in Inspect", r is not None and A.btn_on.isEnabled())

    print("\nDrawing")
    A.btn_on.setChecked(True)
    check("annotating takes the view's mouse", v.tool == "annotate")

    def at(x, y):
        return v.mapFromScene(QPointF(x, y))

    def mouse(kind, x, y):
        t = {"press": QEvent.Type.MouseButtonPress, "move": QEvent.Type.MouseMove,
             "release": QEvent.Type.MouseButtonRelease}[kind]
        pos = QPointF(at(x, y))
        btn = Qt.MouseButton.NoButton if kind == "move" else Qt.MouseButton.LeftButton
        held = Qt.MouseButton.LeftButton if kind != "release" else Qt.MouseButton.NoButton
        QApplication.sendEvent(v.viewport(), QMouseEvent(t, pos, v.viewport().mapToGlobal(pos), btn,
                                                         held, Qt.KeyboardModifier.NoModifier))
    A.tools.chosen.emit("arrow")
    mouse("press", 50, 60)
    mouse("move", 120, 90)
    mouse("release", 150, 100)
    a0 = A.anns[-1] if A.anns else None
    check("an arrow drawn with the mouse lands in image pixels",
          a0 is not None and a0["type"] == "arrow" and abs(a0["pts"][0][0] - 50) < 3
          and abs(a0["pts"][-1][1] - 100) < 3, a0)
    A.tools.chosen.emit("polygon")
    for x, y in ((20, 20), (80, 20), (50, 70)):
        A.press(x, y)
        A.release(x, y)
    A.double_click()
    check("a polygon closes on double-click", A.anns[-1]["type"] == "polygon"
          and len(A.anns[-1]["pts"]) == 3)
    A.tools.chosen.emit("number")
    A.press(200, 50)
    A.press(220, 60)
    check("numbered points are placed", [a["type"] for a in A.anns[-2:]] == ["number", "number"])
    n = len(A.anns)
    A.tools.chosen.emit("delete")
    A.press(220, 60)
    check("Delete removes the annotation clicked", len(A.anns) == n - 1)
    A.undo()
    check("Undo removes the last", len(A.anns) == n - 2)
    A.tools.chosen.emit("measure")
    A.press(10, 10)
    A.move(40, 50, True)
    A.release(40, 50)
    from helio.application.annotations import length_label
    check("a measurement is in pixels without a scale (never a guessed 0.1 mm: KD-044)",
          length_label((10, 10), (40, 50), A.mm_per_px()) == "50 px")
    p.scale_mm_per_px = 0.05
    check("…and in millimetres with one", length_label((10, 10), (40, 50), A.mm_per_px()) == "2.5 mm")

    print("\nLayers")
    A.preset.setCurrentIndex(4)                       # Structural damage
    A._preset(4)
    check("a scholar layer preset becomes the layer drawn on",
          A.layer == "Structural damage" and A.colour == "#ff4444" and A.tool == "polygon")
    A.tools.chosen.emit("rect")
    A.press(100, 100)
    A.move(160, 140, True)
    A.release(160, 140)
    check("its annotations carry the layer", A.anns[-1]["layer"] == "Structural damage")
    A.set_layer_visible("Structural damage", False)
    check("hiding a layer hides its annotations", not A.items[-1].isVisible()
          and A.items[0].isVisible())
    A.set_layer_visible("Structural damage", True)

    print("\nFiles")
    out = tempfile.mkdtemp(prefix="helio_ann_out_")
    jp = A.save(os.path.join(out, "a.json"))
    saved = [dict(a) for a in A.anns]
    A.clear()
    A.load(jp)
    check("the set saves and reloads", [a["type"] for a in A.anns] == [a["type"] for a in saved]
          and len(A.layers) == 2)
    files = A.export_render(os.path.join(out, "annotated.png"))
    img = cv2.imread(files[0])
    svgt = open(files[1]).read()
    root = ET.fromstring(svgt.split("?>", 1)[1])
    check("the annotated view: full resolution, and a well-formed SVG",
          img is not None and img.shape[:2] == (r.h, r.w) and root.tag.endswith("svg"))
    check("…whose arrows have a defined head, and layers are groups",
          "id='ah'" in svgt and "marker-end='url(#ah)'" in svgt and "layer_Structural_damage" in svgt)
    jpg = A.export_render(os.path.join(out, "annotated.jpg"))
    shot = cv2.cvtColor(cv2.imread(jpg[0]), cv2.COLOR_BGR2RGB).astype(np.float32)
    clean = A.current_image().astype(np.float32)[..., :3]
    changed = float((np.abs(shot - clean).max(axis=2) > 60).mean())
    check(f"a JPEG export is small, and the annotations show in it ({changed * 100:.1f}% of "
          f"pixels drawn on)", os.path.getsize(jpg[0]) < os.path.getsize(files[0]) and changed > 0.001,
          (os.path.getsize(jpg[0]), changed))
    small = A.export_render(os.path.join(out, "small.jpg"), long_side=100)
    sm = cv2.cvtColor(cv2.imread(small[0]), cv2.COLOR_BGR2RGB).astype(np.float32)
    clean_sm = cv2.resize(clean, (sm.shape[1], sm.shape[0]), interpolation=cv2.INTER_AREA)
    check("…at a smaller size the picture shrinks and the annotations go with it, still visible",
          max(sm.shape[:2]) == 100 and float((np.abs(sm - clean_sm).max(axis=2) > 60).mean()) > 0.001)
    st = open(jpg[1]).read()
    check("the SVG shows the view under the annotations",
          "<image id='view'" in st and "data:image/jpeg;base64," in st and "layer_Structural_damage" in st)
    A.notes.setPlainText("1. A crack along the rim")
    sch = A.export_scholar(out)
    js = json.load(open([f for f in sch if f.endswith(".json")][0]))
    dmg = [ly for ly in js["layers"] if ly["name"] == "Structural damage"]
    check("the scholar set: clean, annotated and albedo panels, JSON and SVG",
          len(sch) == 5 and all(os.path.exists(f) for f in sch))
    check("…its layers hold their annotations (HELIO 2's were always empty: KD-045)",
          dmg and len(dmg[0]["annotations"]) == 1 and js["notes"] == ["1. A crack along the rim"])
    mine = [a_ for a_ in A.anns if a_.get("layer") == "Structural damage"]
    check("…in the image's own pixels, whatever the export size",
          dmg and mine and dmg[0]["annotations"][0]["pts"] == [list(p_) for p_ in mine[0]["pts"]]
          and js["image_size"] == [r.w, r.h], (dmg and dmg[0]["annotations"][0]["pts"], mine and mine[0]["pts"]))
    A._tool_chosen("rect")
    shape_opts = not A.fill_box.isHidden() and A.text_box.isHidden()
    A._tool_chosen("text")
    check("the panel shows the tool's own options: fill for shapes, the words for text",
          shape_opts and not A.text_box.isHidden() and A.fill_box.isHidden())
    A.btn_on.setChecked(False)
    check("stopping gives the mouse back", v.tool == "")

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
