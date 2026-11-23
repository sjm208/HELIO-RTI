"""Help, Getting Started, Report a Problem, the crash note, and the menus.

Offscreen.  The help renders every section (no classic screenshots) and
search narrows it; Getting Started pages forward and back and remembers
"show at start"; the problem report carries the project's facts; an
uncaught error is logged and offers the report, once per burst; Help ›
Open HELIO Pro (classic) starts the classic script.  And every command in
every menu has a handler the window really has: nothing left saying
"in the classic app only".

    python tests/test_qt_help.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.abspath(__file__))
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
    from PySide6.QtWidgets import QApplication, QMessageBox
    print("=" * 60)
    print("Help and the menus")
    print("=" * 60)
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    from helio_qt.actions import SPECS
    w = hw.MainWindow(appearance="dark")
    w.show()

    print("\nEvery command is real")
    missing = [s.id for s in SPECS if not s.handler or not callable(getattr(w, s.handler, None))]
    check("every menu command has a handler the window has", not missing, missing)
    check("…so none is marked classic-only", not any(s.needs == "classic" for s in SPECS))
    keys = {(k, sid) for k, sid, _a in w.alias_actions}
    check("HELIO 2's single keys still work on the image: R, E, ⇧E, F, ?, and G for the GIF",
          {("R", "process.run"), ("E", "view.next"), ("Shift+E", "view.prev"), ("F", "view.fit"),
           ("?", "help.shortcuts")} <= keys and w.registry.specs["view.sweep_gif"].shortcut == "G")

    print("\nHelp")
    h = w.show_help()
    from helio_qt.help import sections
    secs = sections()
    texts = []
    for i in range(h.list.count()):
        h.list.setCurrentRow(i)
        texts.append(h.view.toPlainText())
    check(f"all {len(secs)} sections render, with no classic screenshots",
          len(texts) == len(secs) and all(len(t) > 80 for t in texts) and "<img" not in h.view.toHtml())
    check("…the glossary, the methods and references and every view are there",
          any("Chrome ball" in t for t in texts) and any("Woodham" in t for t in texts)
          and any("Standard PBR" in t for t in texts))
    from helio_qt.relief3d import KEYS
    relief = next((t for (name, _b), t in zip(secs, texts) if name == "Inspect: the 3D relief"), "")
    check("the 3D relief's page lists every key on its key card (one list, two places)",
          relief and all(k in relief and what in relief for _g, rows in KEYS for k, what in rows))
    titles = [name for name, _b in secs]
    check("pages for the quick exports, the colour check and the plate",
          "Quick exports" in titles and "Colour: checking it" in titles
          and any("publication plate" in t.lower() for t in texts)
          and any("Settings bars" in t for t in texts))
    h.search.setText("ollama")
    shown = [h.list.item(i).text() for i in range(h.list.count()) if not h.list.item(i).isHidden()]
    check("search narrows the sections", shown == ["Research and AI"], shown)
    h.close()

    print("\nGetting started")
    g = w.show_welcome()
    g.go(1)
    g.go(1)
    g.go(-1)
    check("pages forward and back", g.stack.currentIndex() == 1 and g.btn_back.isEnabled())
    for _ in range(g.stack.count()):
        g.go(1)
    check("…the last page's button starts; unticking \"show at start\" is remembered",
          not g.isVisible())
    g2 = w.show_welcome()
    g2.chk.setChecked(False)
    g2.reject()
    check("…and remembered", w.prefs.value("welcome_done", False, type=bool) is True)

    print("\nReport a problem")
    rp = os.path.join(tempfile.mkdtemp(prefix="helio_rep_"), "report.txt")
    d = w.report_problem(rp)
    txt = open(rp).read()
    check("the report: version, machine, this project's facts",
          "new interface" in txt and "machine" in txt and "solver    :" in txt and "images    : 0" in txt,
          txt[:300])
    d.close()

    print("\nThe crash note")
    from helio_qt.help import CrashNote
    shown_boxes = []
    orig_show = QMessageBox.show
    QMessageBox.show = lambda self: shown_boxes.append(self.text())
    note = CrashNote(w)
    try:
        raise ValueError("test failure")
    except ValueError:
        note.hook(*sys.exc_info())
    try:
        raise ValueError("again")
    except ValueError:
        note.hook(*sys.exc_info())
    QMessageBox.show = orig_show
    sys.excepthook = note.prev
    check("an uncaught error offers Report a Problem, once per burst",
          len(shown_boxes) == 1 and "test failure" in shown_boxes[0], shown_boxes)

    print("\nThe classic app")
    import subprocess
    launched = []
    orig = subprocess.Popen
    subprocess.Popen = lambda args, **k: launched.append(args) or "proc"
    w.registry.trigger("help.classic")
    subprocess.Popen = orig
    check("Help › Open HELIO Pro (classic) runs the classic script with this Python",
          launched and launched[0][0] == sys.executable and launched[0][1].endswith("Helio_Pro.py"),
          launched)

    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
