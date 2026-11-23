"""What was found and kept travels with the session: annotations (with their layers and captions), the sections
kept for the publication sheet (with their parallel copies), and the sheet's choices.  A session with none of
these is byte for byte what it was; the older format's reader still reads one that has them.

    python tests/test_kept_work.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
GOLDEN = os.path.join(ROOT, "tests", "fixtures", "session_golden.helio")
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
    import numpy as np
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    from helio.application import annotations as A
    from helio.io import session as S
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.load_session_file(GOLDEN)
    app.processEvents()
    ann = w.workspaces["Inspect"].annotator

    print("Nothing kept")
    plain = os.path.join(tempfile.mkdtemp(), "plain.helio")
    w.write_session(plain)
    check("a session with nothing kept has no extra key", "helio3" not in json.load(open(plain)))

    print("Annotations, sections, sheet choices")
    ann.anns = [A.make("arrow", [(10, 20), (110, 120)], color="#00ff00", text="Tool mark", layer="Annotations"),
                A.make("dot", [(300.5, 400.25)], text="Pinhole", layer="Pinholes")]
    ann.layers = [{"name": "Annotations", "color": "#ff0000", "type": "arrow", "visible": True},
                  {"name": "Pinholes", "color": "#ffdd44", "type": "dot", "visible": False}]
    d = np.linspace(0, 40, 50)
    prof = dict(dist=d, z=np.sin(d), unit="mm height", length_unit="mm", length=40.0, a_px=(10.0, 20.0),
                b_px=(410.0, 20.0), ra=0.5, rq=0.6, rz=2.0,
                series=[dict(dist=d, z=np.sin(d) + k, offset=0.5 * k, a_px=(10.0, 20 + 12 * k), b_px=(410.0, 20 + 12 * k))
                        for k in (-1, 0, 1)], spacing=0.5)
    w.sheet_sections = [(prof, w.render_data)]
    w.set_sheet_prefs(compass=True, notes_on=False)
    kept = os.path.join(tempfile.mkdtemp(), "kept.helio")
    w.write_session(kept)
    raw = json.load(open(kept))
    check("they are written under one key", set(raw["helio3"]) == {"annotations", "sections", "sheet"}, list(raw.get("helio3", {})))
    check("the classic keys are untouched and the older reader reads it", S.read(kept).normal_engine == "Shadow-Rejection LS"
          and list(raw)[:len(json.load(open(plain)))] == list(json.load(open(plain))))
    check("and writes the unknown key back (a session survives any app that opens it)", S.read(kept).extra.get("helio3"))

    print("Opened again, in a new window")
    w2 = hw.MainWindow(appearance="dark")
    w2.load_session_file(kept)
    app.processEvents()
    a2 = w2.workspaces["Inspect"].annotator
    check("the annotations are back, with their captions and positions",
          [(a["type"], a["text"], a["pts"][0]) for a in a2.anns] == [("arrow", "Tool mark", (10.0, 20.0)), ("dot", "Pinhole", (300.5, 400.25))],
          [(a["type"], a["text"]) for a in a2.anns])
    check("and drawn on the picture", len(a2.items) == 2)
    check("the layers are back, with which one was hidden", [(ly["name"], ly["visible"]) for ly in a2.layers] ==
          [("Annotations", True), ("Pinholes", False)])
    sec = w2.sheet_sections
    check("the kept section is back with its profile and its copies", len(sec) == 1 and
          np.allclose(sec[0][0]["z"], np.sin(d)) and len(sec[0][0]["series"]) == 3 and
          sec[0][0]["series"][2]["offset"] == 0.5 and sec[0][0]["a_px"] == (10.0, 20.0), sec[:1])
    import types
    w2.selected_result = lambda: (types.SimpleNamespace(w=2000, h=1500), w2.render_data)       # a solved surface
    w2.workspaces["Measure"].lines = []
    check("it counts for the sheet again on the surface it was kept on", len(w2._sheet_options()["sections"]) == 1)
    check("the sheet's choices are back", w2.sheet_prefs().get("compass") is True and w2.sheet_prefs().get("notes_on") is False)
    w2.write_session(os.path.join(tempfile.mkdtemp(), "again.helio"))
    check("saving it again keeps the kept section", len(json.load(open(w2.session_path))["helio3"]["sections"]) == 1)

    print("The sheet gets the visible layers' notes")
    w.sheet_prefs = lambda: {}
    w.selected_result = lambda: (types.SimpleNamespace(w=2000, h=1500), w.render_data)
    w.workspaces["Measure"].lines = []
    opt = w._sheet_options()
    check("the hidden layer's note is left out, the shown one's goes in", [n["text"] for n in opt["notes"]] == ["Tool mark"], opt["notes"])
    ann.layers[1]["visible"] = True
    check("shown, it goes in too", [n["text"] for n in w._sheet_options()["notes"]] == ["Tool mark", "Pinhole"])
    ann.anns.append(A.make("rect", [(1, 1), (9, 9)], text="Box", layer="Annotations"))
    check("a box is a detail, not a note", all(n["type"] != "rect" for n in w._sheet_options()["notes"]))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
