"""One binding after another in the same window (SES-005).

Finish a capture (processed, scaled, measured, annotated, saved), then open
the next set of photographs: nothing of the first may stay with the second.
Until 2026-10-17 Save after opening the second set wrote it over the first
binding's .helio, under the first binding's name, bookmarks and scale.
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


def capture(folder, seed):
    """Six frames of a small embossed panel, each lit from another side."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    h, w = 240, 320
    z = np.zeros((h, w), np.float32)
    for _ in range(40):
        cv2.circle(z, (int(rng.uniform(20, 300)), int(rng.uniform(20, 220))),
                   int(rng.uniform(4, 12)), 1.0, -1)
    z = cv2.GaussianBlur(z, (0, 0), 2.0) * 4.0
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    L = []
    paths = []
    for k in range(6):
        a, el = 2 * np.pi * k / 6, np.radians(40)
        l = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
        L.append(l)
        img = np.clip(0.6 * np.clip(n @ l, 0, 1) * 255, 0, 255).astype(np.uint8)
        p = os.path.join(folder, f"frame_{k}.png")
        cv2.imwrite(p, cv2.merge([img, img, img]))
        paths.append(p)
    return paths, np.array(L)


def main():
    print("=" * 60)
    print("The next capture in the same window")
    print("=" * 60)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("HELIO_QT_SETTINGS", os.path.join(tempfile.mkdtemp(), "q.ini"))
    os.environ.setdefault("HELIO_PREFS", os.path.join(tempfile.mkdtemp(), "p.json"))
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)
    import helio_qt.window as hw
    for nm in ("information", "warning", "critical"):
        setattr(hw.QMessageBox, nm, staticmethod(lambda *a, **k: None))
    asked = []
    answer = [QMessageBox.StandardButton.Discard]

    def question(*a, **k):
        asked.append(a[2] if len(a) > 2 else "")
        return answer[0]
    hw.QMessageBox.question = staticmethod(question)

    def settle(w, limit=120):
        t = time.time()
        while w.project.busy and time.time() - t < limit:
            app.processEvents()
            time.sleep(0.02)
        w.wait_background()
        app.processEvents()

    book1 = tempfile.mkdtemp(prefix="helio_book1_")
    book2 = tempfile.mkdtemp(prefix="helio_book2_")
    paths1, L1 = capture(book1, 1)
    paths2, L2 = capture(book2, 2)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    print("\nThe first binding, finished")
    w.open_paths(paths1)
    settle(w)
    w.project.set_lights(L1, "file:book1.lp")
    w.registry.trigger("process.run")
    settle(w)
    p = w.project
    check("it processed", p.solved)
    p.scale_mm_per_px, p.z_mm_per_unit = 0.03, 0.2
    p.poi = {"boss": (1.0, 2.0, 0.1)}
    p.capture_meta = {"Object": "Book one"}
    p.colour_check = {"mean": 2.0, "max": 4.0, "file": "chart.jpg", "verdict": "good", "worst": "A1"}
    p.export_rows = {"obj": ("done", "", ["/x/book1.obj"])}
    w.bookmarks = {"raking": {"x": 1}}
    ins = w.workspaces["Inspect"]
    ins.annotator.anns = [{"kind": "arrow"}]
    w.history.stack = [{"render_engine": "Standard PBR"}]
    meas = w.workspaces["Measure"]
    meas.measured = [(12.0, None)]
    sess1 = os.path.join(book1, "book1.helio")
    w.write_session(sess1)
    saved1 = open(sess1, "rb").read()
    check("…and was saved", w.session_path == sess1 and not w.unsaved_work())
    eng = w.engine
    w.engine_settings[eng]["gain"] = float(w.engine_settings[eng].get("gain", 1.0)) + 0.3
    check("moving a view's slider after saving is not unsaved work", not w.unsaved_work())
    p.scale_mm_per_px = 0.031
    check("a change after saving is unsaved work", w.unsaved_work())

    print("\nOpening the next one asks first, and Cancel keeps the first")
    asked.clear()
    answer[0] = QMessageBox.StandardButton.Cancel
    opened = w.open_paths(paths2)
    check("you are asked whether to save the first binding", len(asked) == 1, asked)
    check("Cancel leaves the first binding open, untouched",
          not opened and p.images == paths1 and p.solved and p.scale_mm_per_px == 0.031)

    print("\nDon't Save: the second binding starts clean")
    answer[0] = QMessageBox.StandardButton.Discard
    asked.clear()
    check("it opens", w.open_paths(paths2) and p.images == paths2)
    settle(w)
    check("no session file: Save will ask where", not w.session_path and w.session is None,
          w.session_path)
    check("the window no longer names the first session", w.windowFilePath() == "",
          w.windowFilePath())
    check("no lights, no result", p.lights is None and not p.solved and w.render_data is None)
    check("no scale or height scale", p.scale_mm_per_px is None and p.z_mm_per_unit is None)
    check("no points, capture details, colour check or export list",
          not p.poi and not p.capture_meta and p.colour_check is None and not p.export_rows)
    check("no bookmarks", not w.bookmarks)
    check("no annotations", not ins.annotator.anns)
    check("no measurements", not meas.measured and meas.profile is None)
    check("undo starts again: nothing of the first binding to step back to",
          not w.history.stack and not w.history.redo_stack, len(w.history.stack))
    check("the first binding's session file is as it was saved", open(sess1, "rb").read() == saved1)

    print("\nNew Project forgets the grey card")
    p.update_settings(even_flash=True, flash_ref="10,10,40,40")
    w.grey_ref = {"state": "found", "box": (10, 10, 40, 40)}
    w.new_project()
    check("its box and the correction are gone with the photographs",
          not p.settings.flash_ref and not p.settings.even_flash and w.grey_ref["state"] == "none")
    w.open_paths(paths2)
    settle(w)

    print("\nNothing to lose: no question")
    asked.clear()
    w.open_paths(paths1)
    settle(w)
    check("a capture not yet processed opens the next without asking", not asked, asked)

    print("\nSave first")
    w.open_paths(paths2)
    settle(w)
    w.project.set_lights(L2, "file:book2.lp")
    w.registry.trigger("process.run")
    settle(w)
    sess2 = os.path.join(book2, "book2.helio")
    w.write_session(sess2)
    w.project.scale_mm_per_px = 0.05
    answer[0] = QMessageBox.StandardButton.Save
    asked.clear()
    w.open_paths(paths1)
    settle(w)
    from helio.io import session as S
    check("Save writes the second binding's own session, then opens the next",
          len(asked) == 1 and abs((S.read(sess2).mm_per_px or 0) - 0.05) < 1e-9
          and p.images == paths1)
    check("…and the first binding's file is still its own",
          [os.path.basename(q) for q in S.read(sess1).plate_paths]
          == [os.path.basename(q) for q in paths1])

    print("\nThe menu")
    a = w.registry.actions.get("file.next")
    check("File › Start Next Capture… is there", a is not None and "Next" in a.text(),
          a and a.text())
    pub = w.workspaces["Publish"]
    check("Publish ends with a way on: Start the next capture",
          hasattr(pub, "btn_next") and "next capture" in pub.btn_next.text().lower())

    print("\nWhile processing")
    w.open_paths(paths2)
    settle(w)
    w.project.set_lights(L2, "file:book2.lp")
    w.registry.trigger("process.run")
    app.processEvents()
    answer[0] = QMessageBox.StandardButton.Cancel
    asked.clear()
    busy = bool(w.project.busy)
    w.open_paths(paths1)
    check("a running solve is not swapped out from under itself",
          not busy or (asked and p.images == paths2), (busy, asked))
    settle(w)

    print("\nQuitting")
    w.project.scale_mm_per_px = 0.07              # work not in any session
    answer[0] = QMessageBox.StandardButton.Cancel
    asked.clear()
    w.registry.trigger("app.quit")
    app.processEvents()
    check("Quit with unsaved work asks to save first, and Cancel stays",
          len(asked) == 1 and w.isVisible(), (asked, w.isVisible()))
    answer[0] = QMessageBox.StandardButton.Discard
    asked.clear()
    w.registry.trigger("app.quit")
    app.processEvents()
    check("…Don't Save quits", len(asked) == 1 and not w.isVisible(), (asked, w.isVisible()))
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
