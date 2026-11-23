"""Finding the way back, and seeing what is on: Inspect's photograph mode and the tool buttons.

* A thumbnail clicked over a solved view: "Back to <view>" on the left and right, a badge that
  says which frame and that Esc goes back, the bottom bar naming the frame.
* Esc goes back to the view; with a drawing tool on, the first Esc gives up the drawing and
  stays on the photograph; Esc is not taken from anything else while a view is up.
* The step tabs are wide enough for the chosen one's bold label; the light's buttons are wide
  enough for their words.
* Select, Section, Sweep and Annotate are marked as tools, so the theme outlines them when on.

    python tests/test_navigation.py
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"  {detail}"))


def main():
    from PySide6.QtGui import QFont, QFontMetrics
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    from tools import make_golden as G
    G.H, G.W = 600, 800
    tmp = tempfile.mkdtemp(prefix="helio_nav_")
    paths, L = G.render_stack(tmp, n_lights=6)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    w.open_paths([tmp])
    w.project.set_lights(L, "file:golden.lp")
    w.registry.trigger("process.run")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.01)
    w.go_workspace("Inspect")
    ins = w.workspaces["Inspect"]

    def settle():
        for _ in range(20):
            app.processEvents()
            time.sleep(0.01)
        ins.wait_render()
        app.processEvents()

    def click_frame(i):
        ins.strip.setCurrentRow(i)
        ins.strip.itemClicked.emit(ins.strip.item(i))
        settle()

    def esc():
        sc = next((s for s in getattr(ins, "_esc_backs", ()) if s.isEnabled()), None)
        if sc is not None:
            sc.activated.emit()
        settle()
        return sc is not None

    settle()
    view = w.view
    print("A photograph over the view")
    check("a view is up to begin with", ins.mode == "view" and view, (ins.mode, view))
    check("Esc is left alone while the view is up",
          not any(s.isEnabled() for s in getattr(ins, "_esc_backs", ())))
    click_frame(2)
    check("the thumbnail shows its photograph", ins.mode == "frame")
    name = ins.view_name(view)
    back, back2 = getattr(ins, "btn_back_view", None), getattr(ins, "btn_back_view2", None)
    check("Back to the view, on the left", back is not None and back.isVisible()
          and back.text() == f"◀ Back to {name}", back and back.text())
    check("... and on the right", back2 is not None and back2.isVisible())
    t = ins.badge.text()
    check("the badge names the frame and Esc", ins.badge.isVisible() and "Frame 3 of 6" in t
          and "Esc" in t and name in t, t)
    check("the bottom bar names the frame, not the view", w.status_text.text() == "Frame 3 of 6",
          w.status_text.text())

    print("Back")
    if back is not None:
        back.click()
    else:
        ins.set_view(view)                   # the old way back: choose the view again
    settle()
    check("Back returns to the same view", ins.mode == "view" and w.view == view)
    check("... and the back buttons and badge go", back is not None and not back.isVisible()
          and not back2.isVisible() and "Frame" not in ins.badge.text())
    check("... and the bottom bar names the view", w.status_text.text() == view, w.status_text.text())
    click_frame(4)
    check("Esc returns to the view", esc() and ins.mode == "view" and w.view == view)

    print("Esc with a drawing tool on")
    click_frame(1)
    ins.btn_section.setChecked(True)
    settle()
    check("Section is on over the photograph", ins.viewport.tool == "line")
    esc()
    check("the first Esc gives up the drawing and stays on the photograph",
          not ins.btn_section.isChecked() and not ins.viewport.tool and ins.mode == "frame")
    esc()
    check("the second goes back to the view", ins.mode == "view")

    print("Words that fit")
    for st, b in w.ws_bar.buttons.items():
        f = QFont(b.font())
        f.setWeight(QFont.Weight.DemiBold)
        need = QFontMetrics(f).horizontalAdvance(b.text())
        check(f"step tab {st}: room for its bold label", b.minimumWidth() >= need + 24,
              (b.minimumWidth(), need))
    w.go_workspace("Inspect")
    ins.sec_light.set_open(True)
    settle()
    for b in (ins.btn_raking, ins.btn_sweep, ins.btn_light_save):
        need = QFontMetrics(b.font()).horizontalAdvance(b.text())
        check(f"“{b.text()}” has room for its words", b.width() >= need + 12, (b.width(), need))

    print("Tools show when they are on")
    for b in (ins.btn_select, ins.btn_section, ins.btn_sweep, ins.annotator.btn_on):
        check(f"{b.text()} is marked as a tool", b.property("tool") == "true")
    check("the theme outlines a tool that is on", '[tool="true"]:checked' in app.styleSheet()
          or '[tool="true"]:checked' in w.styleSheet())

    w.project.busy = False
    w.close()
    print(f"\n{'=' * 60}\nResults: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS) + len(FAIL)} tests\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
