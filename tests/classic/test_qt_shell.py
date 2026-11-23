"""The new interface's shell does what the approved design says (QT-001, UX-001..006).

Runs offscreen, so no window appears.  Checks the colour contract, the 12 pt
floor, the menus and shortcuts, the five workspaces and their next steps, the
⌘K search, importing the real capture, finding its sphere exactly as the
classic app does, light files, the theme switch, a tooltip on every control,
no shortcut used twice, and a clean shutdown.  It
also asserts the new app never loads Tk or the classic app.

    python tests/test_qt_shell.py
"""
import os
import re
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
CAPTURE = os.path.join(ROOT, "test images")

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def settle(app, seconds=0.3):
    t0 = time.time()
    while time.time() - t0 < seconds:
        app.processEvents()
        time.sleep(0.01)


def is_emoji(ch):
    o = ord(ch)
    return 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x26FF or o == 0xFE0F


def main():
    print("=" * 60)
    print("The new interface: shell, theme, menus, workspaces")
    print("=" * 60)
    from PySide6.QtCore import QThreadPool
    from PySide6.QtWidgets import QApplication, QPushButton, QWidget
    app = QApplication(sys.argv)
    from helio_qt import theme
    from helio_qt.actions import MENUS, SPECS
    from helio_qt.palette import search
    from helio_qt.window import MainWindow

    print("\nColour and type")
    for name, t in theme.THEMES.items():
        c = t.colours()
        worst = min((theme.contrast(c[a], c[s]), a, s) for a in theme.TEXT_TOKENS
                    for s in theme.SURFACES)
        check(f"{name}: every text colour is at least 4.5:1 on every surface "
              f"(worst {worst[0]:.2f}, {worst[1]} on {worst[2]})", worst[0] >= 4.5)
        check(f"{name}: text on the gold button is at least 4.5:1",
              theme.contrast(t.on_accent, t.accent) >= 4.5)
        sizes = [float(x) for x in re.findall(r"font-size:\s*([\d.]+)pt", theme.stylesheet(t))]
        check(f"{name}: no style sets text below 12 pt", min(sizes) >= 12, f"{min(sizes)}")

    w = MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    settle(app)
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in
                    ("tkinter", "_tkinter", "customtkinter", "Helio_Pro"))
    check("Tk and the classic app are never loaded", not leaked, ", ".join(leaked))

    small = []
    for ws in w.workspaces.values():
        for root in (ws.sidebar, ws.center, ws.inspector):
            for wd in [root] + root.findChildren(QWidget):
                wd.ensurePolished()
                pt = wd.font().pointSizeF()
                if 0 < pt < 12:
                    small.append(f"{type(wd).__name__}:{wd.objectName() or ''} {pt}")
    check("no widget's text is below 12 pt", not small, "; ".join(small[:5]))

    print("\nMenus and shortcuts")
    titles = [a.text() for a in w.menuBar().actions()]
    check(f"menus in Apple's order: {', '.join(titles)}", titles == list(MENUS))
    homes = {}
    for m in w.menuBar().findChildren(type(w.menuBar().actions()[0].menu())):
        for a in m.actions():
            if a.objectName():
                homes[a.objectName()] = homes.get(a.objectName(), 0) + 1
    check("every command has exactly one menu home",
          all(homes.get(s.id) == 1 for s in SPECS),
          [s.id for s in SPECS if homes.get(s.id) != 1][:5])
    keys = [s.shortcut for s in SPECS if s.shortcut]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    check("no two commands share a shortcut", not dup, dup)
    texts = [s.text for s in SPECS] + [b.text() for b in w.findChildren(QPushButton)]
    check("no emoji in any command or button", not any(is_emoji(ch) for t in texts for ch in t))
    unnamed = [b for b in w.findChildren(QPushButton) if not b.text().strip()
               and not b.accessibleName()]
    check("every icon-only button has an accessible name", not unnamed, len(unnamed))
    for i, name in enumerate(("Capture", "Process", "Inspect", "Measure", "Publish")):
        a = w.registry.actions[f"view.ws:{name}"]
        check(f"⌘{i + 1} is {name}", a.shortcut().toString() == f"Ctrl+{i + 1}")
        a.trigger()
        settle(app, 0.05)
        check("...and shows it", w.current == name and
              w.center_stack.currentWidget() is w.workspaces[name].center)

    print("\nBefore anything is imported")
    w.go_workspace("Capture")
    check("the next step is to import", w.primary.text() == "Import images…" and w.primary.isEnabled())
    check("processing says why it cannot run",
          w.registry.reason("process.run") != "" and not w.registry.actions["process.run"].isEnabled())

    print("\nImport the real capture")
    w.open_paths([CAPTURE])
    t0 = time.time()
    while len(w.thumbs.cache) < 8 and time.time() - t0 < 10:
        settle(app, 0.05)
    expected = sorted(os.path.join(CAPTURE, f) for f in os.listdir(CAPTURE)
                      if f.lower().endswith((".jpg", ".jpeg")) and not f.startswith("."))
    check("8 frames, in file order", w.project.images == expected, len(w.project.images))
    check("8 thumbnails loaded", len(w.thumbs.cache) == 8)
    strip = w.workspaces["Inspect"].strip
    check("the filmstrip shows them in the same order", strip.paths == expected and strip.count() == 8)
    check("next step: find the sphere", w.primary.text() == "Find the chrome sphere")

    print("\nFind the sphere, as the classic app does")
    w.registry.trigger("capture.sphere")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 120:
        settle(app, 0.05)
    settle(app, 0.1)
    import numpy as np
    from helio.core.processing import auto_detect_sphere
    want = auto_detect_sphere(expected, None)
    got = w.project.lights
    check("lights identical to the classic automatic detection",
          got is not None and want is not None and np.array_equal(got, np.asarray(want, np.float32)))
    check(f"sphere reported where it was found {w.project.sphere}",
          bool(w.project.sphere) and abs(w.project.sphere[0] - 906) < 3
          and abs(w.project.sphere[1] - 1167) < 3)
    check("Capture is marked done and points on to Process",
          w.project.lights_ready() and w.primary.text() == "Continue to Process →")
    w.registry.trigger("view.ws:Process")
    settle(app, 0.05)
    check("Process names the step and is ready to run (tests/test_qt_process.py runs it)",
          w.primary.text() == "Process 8 images" and w.primary.isEnabled()
          and not w.reason.isVisible())

    print("\nLight files")
    from helio.io.lp import parse_lp
    L = parse_lp("3\nimg1.jpg 0 0 2\nimg2.jpg 0.6 0.8 0\n0 0 0\nbad line here x\n1 0 0\n")
    check("RTIBuilder and plain lines read; count, zero and bad lines skipped",
          L.shape == (3, 3) and np.allclose(L, [[0, 0, 1], [0.6, 0.8, 0], [1, 0, 0]]))
    lp = os.path.join(tempfile.mkdtemp(), "dome.lp")
    with open(lp, "w") as f:
        f.write("8\n" + "".join(f"f{i}.jpg {v[0]} {v[1]} {v[2]}\n" for i, v in enumerate(got)))
    w.load_light_file(lp)
    check("a light file replaces the sphere's lights",
          w.project.light_source == "file:dome.lp" and np.allclose(w.project.lights, got, atol=1e-6))

    print("\nSettings follow the shared registry")
    from helio.domain.registry import DEPTH_METHODS, NORMAL_SOLVERS
    src = open(os.path.join(ROOT, "Helio_Pro.py"), encoding="utf-8").read()

    def dropdown(var):
        m = re.search(r"CTkOptionMenu\(f_ai, values=\[([^\]]*)\], variable=self\.data\." + var, src)
        return re.findall(r'"([^"]+)"', m.group(1)) if m else None
    check("the classic solver menu offers exactly the registry's solvers",
          dropdown("normal_engine") == list(NORMAL_SOLVERS), dropdown("normal_engine"))
    check("the classic depth menu offers exactly the registry's methods",
          dropdown("depth_engine") == list(DEPTH_METHODS), dropdown("depth_engine"))
    w.registry.trigger("process.depth:Frankot-Chellappa")
    check("choosing a depth method from the menu updates the settings and the inspector",
          w.project.settings.depth_engine == "Frankot-Chellappa"
          and w.workspaces["Process"].depth.currentText() == "Frankot-Chellappa")
    from helio_qt.publish_catalog import all_formats
    missing = [f.legacy or f.key for f in all_formats()
               if ((f"    def {f.legacy}(" not in src) if f.legacy else not f.impl)]
    check("every Publish format names a real classic export, or is new and has its own",
          not missing, missing)

    print("\n⌘K search")
    hits = search(w.registry, "depth")
    ids = [h[0] for h in hits]
    check("'depth' finds Recalculate Depth and every depth method",
          "process.recalc" in ids and all(f"process.depth:{m}" in ids for m in DEPTH_METHODS))
    check("unavailable commands come with their reason",
          all(why for sid, why in hits if not w.registry.actions[sid].isEnabled()))
    check("'sphere' finds Find Chrome Sphere first", search(w.registry, "sphere")[0][0] == "capture.sphere")
    w.show_palette()
    settle(app, 0.05)
    w.palette_dlg.query.setText("zoom fit")
    settle(app, 0.05)
    check("the palette lists what was typed", w.palette_dlg.list.count() >= 1)
    w.palette_dlg.close()

    print("\nAppearance")
    w.set_appearance("light", save=False)
    settle(app, 0.05)
    light_bg = app.palette().color(app.palette().ColorRole.Window).name().upper()
    w.set_appearance("dark", save=False)
    settle(app, 0.05)
    dark_bg = app.palette().color(app.palette().ColorRole.Window).name().upper()
    check(f"light and dark switch live ({light_bg} / {dark_bg})",
          light_bg == theme.PAPER.ground and dark_bg == theme.GRAPHITE.ground)
    w.set_scheme("nightdome", save=False)
    settle(app, 0.05)
    nd_bg = app.palette().color(app.palette().ColorRole.Window).name().upper()
    check("the Night Dome scheme is still a choice, live", nd_bg == theme.NIGHT_DOME.ground
          and w.registry.actions["view.scheme:nightdome"].isChecked())
    w.set_scheme("graphite", save=False)
    w.set_appearance("system", save=False)
    check("Match the System picks one of the two", app.property("helio_theme") in ("light", "dark"))

    print("\nCapture file rules shared with the classic app")
    from helio.io.capture import captures_in
    d = tempfile.mkdtemp()
    for n in ("b.JPG", "a.tif", ".hidden.jpg", "notes.txt", "c.png"):
        open(os.path.join(d, n), "w").close()
    check("a folder gives its captures sorted, hidden files and others left out",
          [os.path.basename(p) for p in captures_in([d])] == ["a.tif", "b.JPG", "c.png"])

    print("\nEvery control says what it does, every shortcut once")
    from PySide6.QtGui import QKeySequence
    from PySide6.QtWidgets import (QAbstractButton, QAbstractSlider, QAbstractSpinBox, QComboBox,
                                   QLineEdit, QScrollBar, QWidget)
    from helio_qt.actions import ALIASES, SPECS
    from helio_qt.controls import ViewPicker
    keys = {}
    for sp in SPECS:
        if sp.shortcut:
            keys.setdefault(QKeySequence(sp.shortcut).toString(), []).append(sp.id)
    for k, sid in ALIASES:
        keys.setdefault(QKeySequence(k).toString(), []).append(sid)
    dup = {k: v for k, v in keys.items() if len(v) > 1}
    check("no key is the shortcut of two commands", not dup, dup)
    bare = []
    for name, ws in w.workspaces.items():
        w.go_workspace(name)
        app.processEvents()
        for part in ("sidebar", "center", "inspector"):
            root = getattr(ws, part, None)
            for c in (root.findChildren(QWidget) if root is not None else ()):
                if not isinstance(c, (QAbstractButton, QAbstractSlider, QComboBox, QLineEdit,
                                      QAbstractSpinBox, ViewPicker)):
                    continue
                if isinstance(c, QScrollBar) or c.objectName().startswith("qt_") or \
                        isinstance(c.parent(), (QLineEdit, QComboBox)) or \
                        type(c.parent()).__name__ == "ParamRow" and isinstance(c, QAbstractSpinBox):
                    continue
                if not c.toolTip().strip():
                    text = c.text() if isinstance(c, QAbstractButton) else c.accessibleName()
                    bare.append(f"{name}: {type(c).__name__} {text!r}")
    check("every button, slider, field and list in the five workspaces has a tooltip",
          not bare, bare[:12])

    print("\nShut down")
    w.close()
    settle(app, 0.2)
    check("no background work left running",
          w.pool.activeThreadCount() == 0 and w.thumbs.pool.activeThreadCount() == 0
          and QThreadPool.globalInstance().activeThreadCount() == 0)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
