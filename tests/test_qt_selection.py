"""Inspect's selection: the classic's Shift-drag crop, and a Select button.

A box drawn on the picture limits Publish and the Research tools to that
part of the object, until cleared.  HELIO Pro 3 had carried the gesture
over only as Process a Region (2026-10-17).
"""
import os
import sys
import tempfile
import time

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
    print("=" * 60)
    print("The selection")
    print("=" * 60)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("HELIO_QT_SETTINGS", os.path.join(tempfile.mkdtemp(), "q.ini"))
    os.environ.setdefault("HELIO_PREFS", os.path.join(tempfile.mkdtemp(), "p.json"))
    import numpy as np
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    import helio_qt.window as hw
    for nm in ("information", "warning", "critical", "question"):
        setattr(hw.QMessageBox, nm, staticmethod(lambda *a, **k: None))
    from test_qt_next_capture import capture
    folder = tempfile.mkdtemp(prefix="helio_sel_")
    paths, L = capture(folder, 4)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    def settle(limit=120):
        t = time.time()
        while w.project.busy and time.time() - t < limit:
            app.processEvents()
            time.sleep(0.02)
        w.wait_background()
        app.processEvents()
    w.open_paths(paths)
    settle()
    w.project.set_lights(L, "file:sel.lp")
    w.registry.trigger("process.run")
    settle()
    r = w.project.result
    check("processed", r is not None)
    w.go_workspace("Inspect")
    ins = w.workspaces["Inspect"]
    vp = ins.viewport
    w.set_view("Standard PBR")
    app.processEvents()
    check("a Select button beside the surround swatches, no selection yet",
          ins.btn_select.isVisibleTo(w) and ins.btn_select.isEnabled()
          and not ins.sel_note.isVisibleTo(w) and w.selection is None)

    print("\nShift-drag, as in the classic")
    vp.fit()
    app.processEvents()
    a = vp.mapFromScene(60, 40)
    b = vp.mapFromScene(220, 180)
    QTest.mousePress(vp.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, a)
    QTest.mouseMove(vp.viewport(), b)
    QTest.mouseRelease(vp.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, b)
    app.processEvents()
    sel = w.selection
    check("the box becomes the selection, in the solve's pixels (within a few px)",
          sel is not None and abs(sel[0] - 60) <= 3 and abs(sel[1] - 40) <= 3
          and abs(sel[2] - 220) <= 3 and abs(sel[3] - 180) <= 3, sel)
    check("…shown on the picture and named beside the button, with Clear",
          vp._sel.isVisible() and ins.sel_note.isVisibleTo(w) and "×" in ins.sel_note.text()
          and ins.btn_sel_clear.isVisibleTo(w), ins.sel_note.text())

    print("\nWhat uses it")
    src = w.publish_source()
    x1, y1, x2, y2 = sel
    check("Publish reads the selection alone: its size, its pixels, where it lies",
          src.result.w == x2 - x1 and src.result.h == y2 - y1
          and np.array_equal(src.result.normals_full, r.normals_full[y1:y2, x1:x2])
          and src.result.crop == (x1, y1, x2, y2) and src.data.w == x2 - x1,
          (src.result.w, src.result.h, src.result.crop))
    P = src.result.proxy
    check("…with previews of the selection, all one size",
          P["normals"].shape[:2] == P["albedo"].shape[:2] == P["depth_p"].shape[:2]
          == (src.result.ph, src.result.pw) and P["ptm"].shape[1] == src.result.ph * src.result.pw)
    from helio_qt import publish
    out = tempfile.mkdtemp(prefix="helio_sel_out_")
    files = publish.export_one("png_views", src, out)
    import cv2
    im = cv2.imread(files[0]) if files else None
    check("…and an export of it is the selection's size", im is not None
          and im.shape[:2] == (y2 - y1, x2 - x1), None if im is None else im.shape)
    dlg = w.open_research("toolpaths")
    D = dlg.data()
    check("the Research tools work on it too", D.w == x2 - x1 and D.h == y2 - y1, (D.w, D.h))
    dlg.close()
    whole = w.project.result
    check("the solve itself is untouched", whole is r and r.w == w.render_data.w)

    print("\nThe 3D relief shows the selected part")
    ins.set_centre_mode("3d")
    for _ in range(40):
        app.processEvents()
        time.sleep(0.02)
    r3 = ins.relief
    Ds = w.selected_result()[1]
    dh, dw = r3.data["depth"].shape[:2] if r3 is not None and r3.data else (0, 0)
    check("the relief is the selection's surface, at its own preview's detail",
          r3 is not None and r3.data is not None and (dh, dw) == Ds.depth_proxy_p.shape[:2]
          and r3.full_px == ((x2 - x1) / dw, (y2 - y1) / dh), (dh, dw, getattr(r3, "full_px", None)))
    w.set_selection(None)
    for _ in range(20):
        app.processEvents()
        time.sleep(0.02)
    check("…and the whole object again when the selection is cleared",
          r3.data["depth"].shape[:2] == w.render_data.depth_proxy_p.shape[:2])
    ins.set_centre_mode("image")
    w.set_selection(sel)
    app.processEvents()

    print("\nThe Select button, a click, Clear")
    ins.btn_select.setChecked(True)
    a, b = vp.mapFromScene(10, 10), vp.mapFromScene(110, 90)
    QTest.mousePress(vp.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    QTest.mouseMove(vp.viewport(), b)
    QTest.mouseRelease(vp.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    app.processEvents()
    check("with Select on, a plain drag selects, and Select turns itself off",
          w.selection is not None and abs(w.selection[2] - 110) <= 3
          and not ins.btn_select.isChecked() and not vp.select_mode, w.selection)
    c = vp.mapFromScene(150, 150)
    QTest.mouseClick(vp.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, c)
    app.processEvents()
    check("a Shift-click without a drag clears it, as in the classic", w.selection is None
          and not vp._sel.isVisible())
    w.set_selection((20, 20, 120, 100))
    ins.btn_sel_clear.click()
    app.processEvents()
    check("Clear goes back to the whole object", w.selection is None
          and w.publish_source().result is w.project.result)
    w.set_selection((20, 20, 30, 30))
    check("a box under 16 px is no selection", w.selection is None)

    print("\nOnly Inspect's picture selects")
    mv = w.workspaces["Measure"].viewport
    w.go_workspace("Measure")
    app.processEvents()
    if not mv.item.pixmap().isNull():
        a2, b2 = mv.mapFromScene(20, 20), mv.mapFromScene(120, 100)
        QTest.mousePress(mv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, a2)
        QTest.mouseMove(mv.viewport(), b2)
        QTest.mouseRelease(mv.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, b2)
        app.processEvents()
    check("Shift-drag in Measure draws no selection box there",
          not mv.can_select and getattr(mv, "_sel", None) is None)
    w.go_workspace("Inspect")
    app.processEvents()

    print("\nA new solve, the same selection")
    from dataclasses import replace as _rep
    w.set_selection((20, 20, 120, 100))
    first = w.selected_result()[0]
    w.project.result = _rep(w.project.result, normals_full=w.project.result.normals_full * -1)
    second = w.selected_result()[0]
    check("the cut follows the new solve (held by the result, not its id)",
          second is not first and np.array_equal(second.normals_full,
                                                  w.project.result.normals_full[20:100, 20:120]))
    w.project.result = r
    w.set_selection(None)

    print("\nOn a photograph")
    ins._show_frame(0)
    app.processEvents()
    ins._selection_drawn(30, 30, 130, 110)
    check("a box drawn on a photograph selects the same pixels",
          w.selection == (30, 30, 130, 110), w.selection)
    check("…and is drawn there", vp._sel.isVisible())

    print("\nThe next capture")
    folder2 = tempfile.mkdtemp(prefix="helio_sel2_")
    p2, _L2 = capture(folder2, 5)
    w.open_paths(p2)
    settle()
    check("opening other photographs clears it", w.selection is None and not ins.sel_note.isVisibleTo(w))
    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
