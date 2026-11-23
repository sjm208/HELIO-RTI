"""What is on the publication sheet can be seen, checked and changed.

A sheet came back with four sections on two seals (A–B and E–F one line a
few pixels apart, C–D and G–H the same) and two 3D pictures from the same
angle: nothing showed what was on it, only a line with exactly the same
ends was refused, and the line still drawn was added at export whenever it
differed at all from the kept ones.  So:

* a line close to a kept one (the same way, alongside it, within its band of
  copies) is recognised; another seal's, a crossing line or one far along
  is not;
* Add this section asks first: Replace keeps the letters, Keep both adds,
  Cancel leaves the sheet as it was; the same for a 3D picture taken twice;
* the line drawn now goes on the sheet only when none was kept;
* Publish lists everything, flags a near-duplicate kept before this check,
  and removes, moves (letters follow) and renames;
* the kept sections are drawn faintly on Inspect's picture, the chosen one
  bold;
* a tool's result goes on the sheet under From the tools: a strip across
  the page, the rest three to a row;
* the eight-direction page puts each caption under its picture.

    python tests/test_sheet_contents.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
import numpy as np  # noqa: E402

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


def sec(a, b, copies=0, spacing=20.0):
    """A section from a to b (pixels), with *copies* parallel copies each side."""
    d = np.linspace(0, 10, 30)
    p = dict(dist=d, z=np.sin(d), unit="mm height", length_unit="mm", length=27.8, a_px=tuple(a), b_px=tuple(b),
             ra=0.1, rq=0.1, rz=1.0, features=[])
    if copies:
        a_, b_ = np.array(a, float), np.array(b, float)
        u = (b_ - a_) / np.linalg.norm(b_ - a_)
        n = np.array([-u[1], u[0]])
        p["series"] = [dict(dist=d, z=np.sin(d), offset=k * spacing, a_px=tuple(a_ + k * spacing * n),
                            b_px=tuple(b_ + k * spacing * n)) for k in range(-copies, copies + 1)]
        p["spacing"] = spacing
    return p


def main():
    from helio.application import sheet_contents as SC

    print("Is it the same section?")
    ab = sec((160, 500), (520, 480), copies=3)                      # the left seal, with copies
    ef = sec((150, 515), (515, 500))                                # drawn again, a little lower
    cd = sec((780, 500), (980, 490))                                # the right seal
    cross = sec((340, 330), (340, 680))                             # across A–B
    far = sec((700, 470), (1060, 450))                              # A–B's direction, further along
    c = SC.section_closeness(ef, ab)
    check(f"a line redrawn a few pixels from A–B is the same ({c and round(c['apart'])} px, "
          f"{c and round(c['angle'])}°)", c is not None and c["apart"] < 25)
    check("…and within its band of parallel copies counts too",
          SC.section_closeness(sec((160, 560), (520, 540)), ab) is not None)
    check("the band is the copies' reach", abs(SC.band_half_width(ab) - 60) < 1)
    check("another seal's line, a crossing line and one further along are not",
          SC.section_closeness(cd, ab) is None and SC.section_closeness(cross, ab) is None
          and SC.section_closeness(far, ab) is None)
    check("the closest kept one is named", SC.close_section(ef, [cd, ab])[0] == 1)
    check("in millimetres once there is a scale", SC.closeness_words(c, 0.034).endswith("° apart in direction")
          and " mm apart" in SC.closeness_words(c, 0.034))
    check("a sheet kept before this check: E–F flagged as close to A–B", SC.duplicates([ab, cd, ef, cross]) == {2: 0})

    print("\nIs it the same picture?")
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:400, 0:600]
    a = (np.dstack([np.sin(xx / 40.0) * np.cos(yy / 30.0)] * 3) * 100 + 128).astype(np.uint8)
    a2 = np.clip(a.astype(int) + rng.normal(0, 4, a.shape), 0, 255).astype(np.uint8)
    b = (np.dstack([np.sin(yy / 17.0)] * 3) * 100 + 128).astype(np.uint8)
    check(f"the same 3D view taken twice is alike ({SC.pictures_alike(a, a2):.3f})", SC.close_picture(a2, [b, a]) == (1, SC.pictures_alike(a2, a)))
    check(f"a different view is not ({SC.pictures_alike(a, b):.3f})", SC.close_picture(b, [a]) is None)

    print("\nIn the window")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    import helio_qt.window as hw
    w = hw.MainWindow(appearance="dark")
    ins = w.workspaces["Inspect"]
    ins._section_card()
    answers = []
    w.ask_sheet_duplicate = lambda text: (answers.append(text), answers_next.pop(0))[1]
    answers_next = []

    def add(p):
        ins.section_prof = p
        ins.section_card.set_profile(p)
        return w.add_sheet_section("relief")
    w.flash = lambda *a, **k: None
    check("A–B is added", add(ab) and len(w.sheet_sections) == 1)
    answers_next[:] = ["cancel"]
    check("a line close to it asks first; Cancel leaves the sheet as it was",
          not add(ef) and len(w.sheet_sections) == 1 and "close to section A–B" in answers[-1], answers[-1:])
    answers_next[:] = ["replace"]
    add(ef)
    check("Replace puts it in A–B's place, keeping the letters", len(w.sheet_sections) == 1
          and w.sheet_sections[0][0]["a_px"] == ef["a_px"] and w.section_letters(ef) == ("A", "B"))
    answers_next[:] = ["add"]
    add(ab)
    check("Keep both adds it as C–D", len(w.sheet_sections) == 2 and w.section_letters(ab) == ("C", "D"))
    n_asked = len(answers)
    add(cd)
    check("the other seal's line goes on without a question, as E–F",
          len(answers) == n_asked and w.section_letters(cd) == ("E", "F"))

    from helio_qt.publish import _sheet_sections
    live = sec((300, 300), (400, 320))
    check("the line drawn now is not added when sections are kept",
          len(_sheet_sections(dict(section_prof=live), dict(sections=[ab, cd]))) == 2)
    check("…and is the sheet's section when none is", _sheet_sections(dict(section_prof=live), dict(sections=[]))
          == [live])

    print("\nThe kept lines on the picture")
    drawn = []
    real_show = ins.viewport.show_kept_sections
    ins.viewport.show_kept_sections = lambda items: drawn.append(list(items))
    w.project.result = w.project.result or type("R", (), {})()
    w.render_data = w.render_data or type("D", (), {"w": 1000, "h": 1000})()
    ins.btn_section.blockSignals(True)
    ins.btn_section.setChecked(False)
    ins.btn_section.blockSignals(False)
    ins._draw_kept_sections()
    check("with the Section tool off the picture has no kept lines", drawn[-1] == [])
    ins.btn_section.blockSignals(True)
    ins.btn_section.setChecked(True)
    ins.btn_section.blockSignals(False)
    real = ins._solve_to_view, ins._sel_offset, w.selected_result, getattr(ins, "section_prof", None)
    ins._solve_to_view, ins._sel_offset = (lambda x, y: (x, y)), (lambda: (0, 0))
    w.selected_result = lambda: (None, None)
    ins.section_prof = None
    ins._draw_kept_sections()
    check("with it on they show", len(drawn[-1]) == len(w.sheet_sections) and len(drawn[-1]) > 0, drawn[-1])
    ins.section_card.chk_share.setChecked(False)
    check("…and Show on the 2D image unticked takes them off", drawn[-1] == [])
    ins.section_card.chk_share.setChecked(True)
    ins._solve_to_view, ins._sel_offset, w.selected_result, ins.section_prof = real
    ins.btn_section.blockSignals(True)
    ins.btn_section.setChecked(False)
    ins.btn_section.blockSignals(False)
    ins.viewport.show_kept_sections = real_show
    w.project.result, w.render_data = None, None

    print("\nThe list in Publish")
    pub = w.workspaces["Publish"]
    w.sheet_shots = [("3D relief, 1", a, "3d")]
    answers_next[:] = ["cancel"]
    check("the same 3D view again asks first", not w._add_sheet_picture("3D relief, 2", a2, "3d")
          and len(w.sheet_shots) == 1 and "3D relief, 1" in answers[-1])
    w.add_sheet_tool("Legend unrolled: + SIGILLVM", np.full((40, 900, 3), 200, np.uint8))
    check("a tool's result is kept as one", w.sheet_shots[-1][2] == "tool"
          and w._sheet_options().get("tools", [None])[0][0] == "Legend unrolled: + SIGILLVM"
          if w.project.result is not None else w.sheet_shots[-1][2] == "tool")
    ins.section_prof = live
    pub._fill_sheet_list()
    rows = [pub.sheet_list.item(i).text() for i in range(pub.sheet_list.count())]
    check(f"it lists the sections with their letters, the line drawn now, and the pictures ({len(rows)} rows)",
          len(rows) == 6 and rows[0].startswith("A–B") and "⚠ close to A–B" in rows[1]
          and "drawn now, not on the sheet" in rows[3] and "(3D)" in rows[4] and "(from a tool)" in rows[5], rows)
    pub.sheet_list.setCurrentRow(1)
    pub._move_sheet_row(-1)
    check("Up moves C–D first, and the letters follow its place",
          w.sheet_sections[0][0]["a_px"] == ab["a_px"] and w.section_letters(ab) == ("A", "B")
          and pub.sheet_list.currentRow() == 0)
    pub.sheet_list.setCurrentRow(1)
    pub._remove_sheet_row()
    check("Remove takes it off", len(w.sheet_sections) == 2 and w.sheet_sections[1][0]["a_px"] == cd["a_px"])
    w.rename_sheet_picture(0, "Seal, raking from the left")
    check("a picture's caption can be changed", w.sheet_shots[0][0] == "Seal, raking from the left")

    print("\nOn the sheet")
    from PIL import Image
    from helio.application import pub_sheet as PS
    H, W = 300, 400
    pic = np.full((H, W, 3), 150, np.uint8)
    s = PS.Sheet(title="Test", full_size=(W, H), albedo=pic, views=[PS.View("Standard PBR", pic, "relit")],
                 hero=PS.View("Standard PBR", pic, "relit"),
                 tools=[("Legend unrolled", np.full((60, 1800, 3), 30, np.uint8)),
                        ("Line drawing", np.full((400, 300, 3), 90, np.uint8)),
                        ("Zones", np.full((400, 300, 3), 90, np.uint8))])
    pg = PS._Page()
    y0 = pg.y
    PS._tools(pg, s)
    im = np.asarray(pg.im)
    strip_rows = np.where((np.abs(im[:, PS.PAGE_W // 2].astype(int) - 30) < 4).all(axis=1))[0]
    check("From the tools: the strip runs across the page, the others side by side below it",
          len(strip_rows) > 0 and (np.abs(im[strip_rows[0], PS.M + 5].astype(int) - 30) < 4).all()
          and (np.abs(im[strip_rows[0], PS.PAGE_W - PS.M - 5].astype(int) - 30) < 4).all() and pg.y > y0 + 400)

    yy, xx = np.mgrid[0:200, 0:600]
    wide = (np.dstack([np.sin(xx / 9.0)] * 3) * 80 + 120).astype(np.uint8)
    s2 = PS.Sheet(title="Test", full_size=(600, 200), albedo=wide, views=[], hero=None,
                  compass={az: wide for az in range(0, 360, 45)}, normals=np.dstack([np.zeros((200, 600))] * 2
                                                                                    + [np.ones((200, 600))]))
    page = PS._compass(s2)
    g = np.asarray(page.convert("L")).astype(int)
    # the N picture: a 780 px square round (page centre, 920); a 3:1 picture fills 780 x 260 of it,
    # so its bottom edge is at 920 + 130 = 1050 - and its caption just under that, not at the square's foot
    band = g[:, PS.PAGE_W // 2 - 120:PS.PAGE_W // 2 + 120]
    text_rows = np.where((band < 150).sum(axis=1) > 3)[0]
    under = text_rows[(text_rows > 1052) & (text_rows < 1400)]
    check(f"the eight-direction page puts each caption just under its picture (text at {under[:1]})",
          len(under) > 0 and under[0] < 1100, under[:3])
    del Image, app
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
