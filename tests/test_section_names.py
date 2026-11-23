"""The sections are named A–B, C–D, E–F … as they will be on the publication sheet: the card, the chart, the 3D line
and the line on the picture all use the same letters, and a kept section keeps its letters.

    python tests/test_section_names.py
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


def prof(x):
    import numpy as np
    d = np.linspace(0, 10, 30)
    return dict(dist=d, z=np.sin(d), unit="mm height", length_unit="mm", length=10.0, a_px=(x, 10.0), b_px=(x + 50.0, 10.0),
                ra=0.1, rq=0.1, rz=1.0, features=[])


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    w = hw.MainWindow(appearance="dark")
    ins = w.workspaces["Inspect"]
    card = ins._section_card()

    print("The letters")
    check("the first section is A–B", w.section_letters(prof(0)) == ("A", "B"))
    w.sheet_sections = [(prof(0), None), (prof(100), None)]
    check("a kept section keeps its letters", w.section_letters(prof(100)) == ("C", "D") and w.section_letters(prof(0)) == ("A", "B"))
    check("a new line is the next pair", w.section_letters(prof(300)) == ("E", "F"))
    w.sheet_sections = [(prof(10 * i), None) for i in range(13)]
    check("after thirteen it starts again rather than running off the alphabet", w.section_letters(prof(999)) == ("A", "B"))
    w.sheet_sections = []

    print("On the card, the chart and the line")
    ins.section_card.set_profile(prof(0))
    ins.section_prof = prof(0)
    ins.name_section()
    check("the card says Section A–B", ins.section_card.title.text() == "Section A–B")
    w.sheet_sections = [(prof(0), None)]
    ins.section_card.set_profile(prof(200))
    ins.section_prof = prof(200)
    ins.name_section()
    check("with one kept, the next line is Section C–D on the card", ins.section_card.title.text() == "Section C–D")
    check("and its chart", ins.section_card.chart.letters == ("C", "D"))
    check("and the picture's line", ins.viewport.section_letters == ("C", "D"))
    ins.section_prof = prof(0)
    ins.section_card.set_profile(prof(0))
    ins.name_section()
    check("the kept one is still A–B", ins.section_card.title.text() == "Section A–B")
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
