"""Every command and every control says what it does when the pointer rests on it.

* Every command in the menus and the ⌘K palette has a description
  (helio_qt.action_tips), shown as its tooltip, with the reason when it is
  unavailable; the menus show tooltips; no description names a command that
  no longer exists; ⌘K finds commands by what they do.
* Every button, box, choice and slider in the five workspaces, and in every
  dialog the menus open, has a tooltip (helio_qt.dialog_tips fills dialogs'),
  and no dialog tip is keyed to a window title nothing uses.

    python tests/test_qt_tooltips.py
"""
import importlib.util
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def untipped(root, seen=None):
    from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox,
                                   QLineEdit, QScrollBar)
    out = []
    for T in (QAbstractButton, QComboBox, QAbstractSlider, QAbstractSpinBox, QLineEdit):
        for c in root.findChildren(T):
            if isinstance(c, QScrollBar) or not c.isVisibleTo(root) or (seen is not None and id(c) in seen):
                continue
            if seen is not None:
                seen.add(id(c))
            tip, p, hops = c.toolTip(), c.parentWidget(), 0
            while not tip and p is not None and hops < 2:          # a spin box's own line edit, a combo's
                tip, p, hops = p.toolTip(), p.parentWidget(), hops + 1
            if not tip:
                out.append(f"{type(c).__name__} {getattr(c, 'text', lambda: '')() or c.accessibleName()!r}")
    return out


def main():
    from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QInputDialog, QMessageBox
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    from helio_qt.action_tips import TIPS, tip
    from helio_qt.actions import SPECS
    print("Commands")
    ids = {s.id for s in SPECS}
    no_tip = [s.id for s in SPECS if not tip(s.id)]
    check("every command has a description", not no_tip, no_tip)
    stale = sorted(set(TIPS) - ids)
    check("no description for a command that is gone", not stale, stale)
    for nm in ("warning", "information", "question", "critical"):
        setattr(QMessageBox, nm, staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    a = w.registry.actions["process.light_field"]
    check("a command's tooltip is its description", tip("process.light_field") in a.toolTip(), a.toolTip())
    check("…with why not, while it is unavailable", "Not available:" in a.toolTip(), a.toolTip())
    from PySide6.QtWidgets import QMenu
    menus = [m for m in w.menuBar().findChildren(QMenu) if m.title()]     # the named menus and submenus
    check("the menus show tooltips", menus and all(m.toolTipsVisible() for m in menus),
          [m.title() for m in menus if not m.toolTipsVisible()])
    nt = w.registry.native_tip(("Process", "Precision Measurement"), "Calibrate Lens from a Chessboard…")
    check("the Mac's own menu bar gets the same tooltip, by the item's place",
          tip("process.lens") in nt and w.registry.native_tip(("View",), "Process").startswith("Process:")
          and w.registry.native_tip(("Python",), "Quit Python") == w.registry.actions["app.quit"].toolTip(),
          nt)
    if sys.platform == "darwin" and os.environ.get("HELIO_NATIVE_MENU_TEST"):
        # a real window on screen: the native menu items carry the descriptions
        import subprocess
        code = (
            "import os,sys,tempfile;sys.path.insert(0,%r);os.environ['HELIO_QT_SETTINGS']="
            "os.path.join(tempfile.mkdtemp(),'p.ini')\n"
            "from PySide6.QtWidgets import QApplication\nfrom PySide6.QtCore import QTimer\n"
            "app=QApplication([]);import helio_qt.window as hw;w=hw.MainWindow();w.show()\n"
            "def go():\n from helio_qt import native_menu as N\n w.registry.sync_native_tooltips()\n"
            " print('TIP', N.tooltip_of(('Process','Precision Measurement',"
            "'Calibrate Lens from a Chessboard…'))[:30]);app.quit()\n"
            "QTimer.singleShot(2000,go);app.exec()\n") % ROOT
        env = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                             timeout=120).stdout
        check("…and they are on the native menu items", "TIP Precision:" in out, out[-200:])
    from helio_qt.palette import search
    found = [sid for sid, _why in search(w.registry, "chessboard distortion")]
    check("⌘K finds a command by what it does", "process.lens" in found, found[:5])

    print("\nWorkspaces")
    spec = importlib.util.spec_from_file_location(
        "cb", os.path.join(ROOT, "tests", "fixtures", "synthetic_capture.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    _z, n = cb.surface()
    L = cb.lights()
    d = os.path.join(tempfile.mkdtemp(prefix="helio_tips_"), "capture")
    os.makedirs(d)
    for i, lv in enumerate(L):
        img = (np.clip(n @ lv, 0, 1) * 235 + 10).astype(np.uint8)
        cv2.imwrite(os.path.join(d, f"Coin_{i:02d}.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    w.open_paths([d])
    w.project.set_lights(L, "file:x.lp")
    w.registry.trigger("process.run")
    t0 = time.time()
    while (w.project.busy or w.project.result is None) and time.time() - t0 < 600:
        app.processEvents()
        time.sleep(0.02)
    seen = set()
    for ws in ("Capture", "Process", "Inspect", "Measure", "Publish"):
        w.go_workspace(ws)
        for _ in range(10):
            app.processEvents()
        W = w.workspaces[ws]
        miss = []
        for root in (W.sidebar, W.center, W.inspector):
            miss += untipped(root, seen)
        check(f"{ws}: every control has a tooltip", not miss, miss)

    print("\nDialogs")
    QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: ("", ""))
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: ([], ""))
    QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: "")
    QInputDialog.getText = staticmethod(lambda *a, **k: ("", False))
    QInputDialog.getItem = staticmethod(lambda *a, **k: ("", False))
    QDialog.exec = lambda self: (self.show(), 0)[1]
    skip = ("app.quit", "file.new", "file.next", "process.run", "process.cancel", "process.recalc",
            "file.open", "window.minimize", "file.batch", "file.export_now", "view.sweep",
            "process.refine")
    before = {id(t) for t in app.topLevelWidgets()}
    titles, problems = set(), []
    for sid in list(w.registry.actions):
        if sid in skip or sid.startswith(("view.appearance", "view.scheme", "process.solver",
                                          "process.depth")):
            continue
        t1 = time.time()
        while w.project.busy and time.time() - t1 < 60:          # one command at a time
            app.processEvents()
            time.sleep(0.02)
        try:
            w.registry.trigger(sid)
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"{sid}: {type(e).__name__}")
        for _ in range(3):
            app.processEvents()
        for t in app.topLevelWidgets():
            if id(t) in before or not t.isVisible() or t is w:
                continue
            before.add(id(t))
            titles.add(t.windowTitle())
            miss = untipped(t)
            if miss:
                problems.append(f"{sid} › {t.windowTitle()}: {miss}")
            t.close()
    check("every dialog's controls have tooltips", not problems, "\n        ".join(problems))
    from helio_qt.dialog_tips import TIPS as DT
    stale = sorted(set(DT) - titles)
    check("no dialog tips for a window that is gone", not stale, stale)
    w.close()


main()
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
