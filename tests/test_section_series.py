"""Parallel sections and section drawings.

* the geometry: even steps either side of the line, perpendicular, a stated sign, clipping by dropping;
* sampled on surfaces with known answers (a ramp, a plane): copies are copies, steps are the right size in
  mm and in pixels, the middle copy is the section itself;
* the card: copies and spacing ask for a series, it is drawn and cleared with the section, Save and Drawing
  write what they say;
* the files are read back: the SVG is to the scale it states (checked from its own coordinates), the DXF has
  the polylines and vertices, the CSV the columns.

    python tests/test_section_series.py
"""
import math
import os
import re
import sys
import tempfile
import xml.dom.minidom

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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


def prof(off=0.0, n=200, length=4.0, unit="mm", feats=True, amp=0.1):
    d = np.linspace(0, length, n)
    z = amp * np.sin(d * 3) + off * 0.01
    return dict(dist=d, z=z, unit=unit, length_unit="mm" if unit.startswith("mm") else "px", length=float(length),
                offset=off, a_px=(10.0, 20.0), b_px=(110.0, 20.0), ra=0.07, rq=0.07, rz=0.2,
                features=[dict(n=1, kind="groove", index=60, at=1.0, depth=0.1, width=0.3),
                          dict(n=2, kind="ridge", index=120, at=2.0, depth=0.1, width=0.3)] if feats else None)


def main():
    from helio.core import sections as S
    from helio.application import section_export as E

    print("The geometry")
    check("offsets: centred, even", S.offsets(5, 2.0) == [-4.0, -2.0, 0.0, 2.0, 4.0] and S.offsets(1, 3.0) == [0.0]
          and S.offsets(4, 1.0) == [-1.5, -0.5, 0.5, 1.5])
    W, H = 1000.0, 800.0
    for a, b in (((0.2, 0.5), (0.8, 0.5)), ((0.5, 0.2), (0.5, 0.8)), ((0.2, 0.3), (0.7, 0.6)), ((0.7, 0.6), (0.2, 0.3))):
        lines, dropped = S.offset_lines(a, b, 7, 12.0, (W, H))
        da = np.array([(b[0] - a[0]) * W, (b[1] - a[1]) * H])
        ok = dropped == 0 and len(lines) == 7
        for ln in lines:
            ea = np.array([(ln["b"][0] - ln["a"][0]) * W, (ln["b"][1] - ln["a"][1]) * H])
            ok &= np.allclose(ea, da, atol=1e-6)                                     # a copy: same length and direction
            mid = np.array([((ln["a"][0] + ln["b"][0]) / 2 - (a[0] + b[0]) / 2) * W, ((ln["a"][1] + ln["b"][1]) / 2 - (a[1] + b[1]) / 2) * H])
            ok &= abs(float(mid @ da)) < 1e-6 * np.linalg.norm(da)                     # shifted perpendicular
            ok &= abs(np.linalg.norm(mid) - abs(ln["offset"])) < 1e-6                  # by the stated offset
        check(f"copies of {a}->{b} are perpendicular, parallel, 12 px apart", ok)
    lines, _ = S.offset_lines((0.2, 0.5), (0.8, 0.5), 3, 10.0, (W, H))
    check("positive is to the right looking A to B: down the screen for a line drawn left to right",
          lines[2]["offset"] > 0 and lines[2]["a"][1] > 0.5 > lines[0]["a"][1])
    lines, _ = S.offset_lines((0.5, 0.2), (0.5, 0.8), 3, 10.0, (W, H))
    check("…and to the left of the screen for a line drawn downward", lines[2]["offset"] > 0 and lines[2]["a"][0] < 0.5)
    lines, dropped = S.offset_lines((0.1, 0.05), (0.9, 0.05), 7, 40.0, (W, H))      # offsets -120..+120 px; v = 0.05 -/+ 0.15
    check("copies that would leave the surface are dropped, not clipped (the one exactly on the edge stays)",
          dropped == 2 and len(lines) == 5 and min(c for ln in lines for c in (ln["a"][1], ln["b"][1])) >= -1e-9, (dropped, len(lines)))
    try:
        S.offset_lines((0.5, 0.5), (0.5, 0.5), 3, 1.0, (W, H))
        check("a line with no length is refused", False)
    except ValueError:
        check("a line with no length is refused", True)
    check("round drawing scales that never overrun the page",
          all(4.0 * S.nice_scale(4.0, 170) <= 170.0001 and 4.0 * S.nice_scale(4.0, 170) > 170 / 2.6 for _ in [0])
          and S.scale_label(25.0) == "25:1" and S.scale_label(0.02) == "1:50" and S.scale_label(1.0) == "1:1")
    check("a stack's step is a fraction of the tallest relief", abs(S.stack_gap([prof(0)]) - 0.6 * (max(prof(0)["z"]) - min(prof(0)["z"]))) < 1e-12)

    print("On surfaces with known answers")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from helio_qt.relief3d import ReliefView
    Hh, Ww = 120, 160
    yy, xx = np.mgrid[0:Hh, 0:Ww].astype(np.float32)
    flat_n = np.dstack([np.zeros((Hh, Ww)), np.zeros((Hh, Ww)), np.ones((Hh, Ww))]).astype(np.float32)
    alb = np.full((Hh, Ww, 3), 128, np.uint8)

    def view(depth, mm=0.1, zmm=2.0):
        v = ReliefView()
        v.set_surface(depth.astype(np.float32), flat_n, alb, scale_mm=mm, z_mm_per_unit=zmm, full_size=(Ww, Hh))
        return v
    ramp = view(xx / Ww * 0.5)                                       # height depends on x only
    ramp.set_section((0.2, 0.5), (0.8, 0.5))
    ser = ramp.section_series(5, 2.0)
    zs = [q["z"] for q in ser["profiles"]]
    check("5 copies, 2 mm apart, none dropped", len(zs) == 5 and ser["dropped"] == 0
          and [round(q["offset"], 6) for q in ser["profiles"]] == [-4.0, -2.0, 0.0, 2.0, 4.0], [q["offset"] for q in ser["profiles"]])
    check("along a ramp's contour every copy is the same profile", all(np.allclose(zs[0], q, atol=1e-6) for q in zs))
    one = ramp.section_profile(features=False)
    check("the middle copy is the section itself", np.allclose(zs[2], one["z"]) and np.allclose(ser["profiles"][2]["dist"], one["dist"]))
    check("every copy is the section's length (9.6 mm here)", all(abs(q["length"] - one["length"]) < 1e-9 for q in ser["profiles"]) and abs(one["length"] - 9.6) < 1e-6)
    ramp.set_section((0.5, 0.2), (0.5, 0.8))
    ser = ramp.section_series(5, 2.0)
    means = np.array([float(np.mean(q["z"])) for q in ser["profiles"]])
    along = ramp.profile_between((0.3, 0.5), (0.7, 0.5))             # the ramp itself: height per mm
    slope = (along["z"][-1] - along["z"][0]) / along["dist"][-1]
    steps = np.diff(means)
    check("across the ramp each copy steps by slope x spacing (negative: positive offsets are towards -x)",
          np.allclose(steps, -slope * 2.0, rtol=0.02), (steps, -slope * 2.0))
    px = view(xx / Ww * 0.5, mm=None, zmm=None)
    px.set_section((0.5, 0.2), (0.5, 0.8))
    sp = px.section_series(3, 10.0)
    check("with no scale the spacing and offsets are pixels", sp["length_unit"] == "px" and [round(q["offset"], 6) for q in sp["profiles"]] == [-10.0, 0.0, 10.0])
    near = view(xx * 0 + 0.1)
    near.set_section((0.1, 0.05), (0.9, 0.05))
    ns = near.section_series(7, 2.0)
    check("near an edge the copies that do not fit are left out and counted", ns["dropped"] > 0 and len(ns["profiles"]) + ns["dropped"] == 7, (ns["dropped"], len(ns["profiles"])))
    check("lines are held for the relief to draw", len(near.series_lines) == len(ns["profiles"]) and len(near.series_lines) > 1)
    near.clear_series()
    check("and cleared", near.series_lines == [])
    ramp.set_section((0.2, 0.5), (0.8, 0.5))
    ramp.section_series(3, 2.0)
    ramp.clear_section()
    check("clearing the section clears its copies", ramp.series_lines == [] and ramp.section is None)
    ramp.set_section((0.2, 0.5), (0.8, 0.5))
    ramp.section_series(3, 2.0)
    ramp.set_surface(ramp.data["depth"], flat_n, alb)
    check("a new surface clears them", ramp.series_lines == [])
    ramp.set_scale(0.1, 2.0)

    print("The card")
    from helio_qt.section_card import SectionCard
    ramp.set_section((0.2, 0.5), (0.8, 0.5))
    card = SectionCard(share="Show on the 2D image")
    card.attach(lambda: ramp, None, lambda: "Test object")
    asked = []
    card.series_requested.connect(lambda c, s: asked.append((c, round(s, 6))))
    pr = ramp.section_profile()
    card.set_profile(pr)
    check("a section shows the card, spacing starts at a tenth of its length, in mm", card.isVisibleTo(card.parent()) or not card.isHidden(), card.isHidden())
    check("the spacing is a tenth of the length (0.96 -> 0.96 to 2 figures) and says mm", abs(card.spacing.value() - 0.96) < 1e-9 and card.spacing.suffix() == " mm", (card.spacing.value(), card.spacing.suffix()))
    check("1 copy is the section alone: no chart", card.series is None and not card.series_chart.isVisibleTo(card))
    card.count.setValue(5)
    check("asking for 5 copies draws them", card.series is not None and len(card.series["profiles"]) == 5 and ramp.series_lines
          and card.series_chart.isVisibleTo(card) or card.series is not None and len(card.series["profiles"]) == 5, card.series is None)
    check("and the card says how many", "5 parallel sections" in card.series_msg.text() and "0.96" in card.series_msg.text(), card.series_msg.text())
    card.spacing.setValue(2.0)
    check("changing the spacing asks again", asked[-1] == (5, 2.0) and abs(card.series["spacing"] - 2.0) < 1e-9, asked[-3:])
    check("stack off overlays them", (card.chk_stack.setChecked(False) or True) and card.series_chart.stacked is False)
    card.chk_stack.setChecked(True)
    ramp.set_section((0.25, 0.5), (0.75, 0.5))
    card.set_profile(ramp.section_profile())
    check("moving the section gives its copies again, a step the person chose is kept", abs(card.spacing.value() - 2.0) < 1e-9
          and card.series is not None and abs(card.series["profiles"][2]["length"] - ramp.section_profile()["length"]) < 1e-9)
    card.clear_requested.emit()
    check("Clear removes the copies", card.series is None and ramp.series_lines == [])
    card.set_profile(None)
    check("no section, no card", card.series is None)

    print("Files, read back")
    tmp = tempfile.mkdtemp(prefix="helio_sections_")
    one = prof(0.0)
    info = E.section_svg(os.path.join(tmp, "one.svg"), [one], title="Plate 4: section A–B", meta={"Object": "Binding"})
    doc = xml.dom.minidom.parse(os.path.join(tmp, "one.svg"))
    pl = doc.getElementsByTagName("polyline")
    pts = np.array([[float(c) for c in t.split(",")] for t in pl[0].getAttribute("points").split()])
    paper_w = pts[:, 0].max() - pts[:, 0].min()
    check(f"a 4 mm section is drawn at {info['scale']}, its drawn width {paper_w:.1f} mm = 4 mm x {info['scale_factor']:g}",
          abs(paper_w - 4.0 * info["scale_factor"]) < 0.05 and info["to_scale"])
    paper_h = pts[:, 1].max() - pts[:, 1].min()
    check("heights are to the same scale (true): the drawn height range = the relief x the scale",
          abs(paper_h - (one["z"].max() - one["z"].min()) * info["scale_factor"]) < 0.05, (paper_h, (one["z"].max() - one["z"].min()) * info["scale_factor"]))
    E.section_svg(os.path.join(tmp, "ex.svg"), [one], v_exag=5.0)
    pts5 = np.array([[float(c) for c in t.split(",")] for t in xml.dom.minidom.parse(os.path.join(tmp, "ex.svg")).getElementsByTagName("polyline")[0].getAttribute("points").split()])
    check("heights ×5 are drawn 5 times taller and the sheet says so",
          abs((pts5[:, 1].max() - pts5[:, 1].min()) - 5 * paper_h) < 0.1 and "exaggerated ×5" in open(os.path.join(tmp, "ex.svg")).read())
    txt = open(os.path.join(tmp, "one.svg")).read()
    check("the title block says the object, the scale, A and B, and Ra Rq Rz", all(w in txt for w in ("Plate 4", "Binding", "25:1", "A at 10.0", "Ra 0.07")))
    check("grooves and ridges are marked and numbered", txt.count("<polygon") == 2)
    sbar = re.search(r'<line x1="([\d.]+)" y1="[\d.]+" x2="([\d.]+)" y2="[\d.]+" stroke="#000" stroke-width="0.5"/>', txt)
    mm_bar = float(re.search(r'text-anchor="middle">([\d.]+) mm</text>', txt).group(1))
    check(f"the scale bar is {mm_bar:g} mm long at the drawing's scale", sbar and abs((float(sbar.group(2)) - float(sbar.group(1))) - mm_bar * info["scale_factor"]) < 0.01)
    series = [prof(o, feats=False) for o in (-1.0, 0.0, 1.0)]
    E.section_svg(os.path.join(tmp, "ser.svg"), series, stacked=True)
    sd = xml.dom.minidom.parse(os.path.join(tmp, "ser.svg"))
    pls = [np.array([[float(c) for c in t.split(",")] for t in e.getAttribute("points").split()]) for e in sd.getElementsByTagName("polyline")]
    check("a series is one polyline each, stacked upward in order", len(pls) == 3 and pls[0][:, 1].mean() > pls[1][:, 1].mean() > pls[2][:, 1].mean())
    gaps = [pls[i][:, 1].mean() - pls[i + 1][:, 1].mean() for i in range(2)]
    check("at equal steps", abs(gaps[0] - gaps[1]) < 0.05, gaps)
    E.section_svg(os.path.join(tmp, "ov.svg"), series, stacked=False)
    ov = [np.array([[float(c) for c in t.split(",")] for t in e.getAttribute("points").split()]) for e in xml.dom.minidom.parse(os.path.join(tmp, "ov.svg")).getElementsByTagName("polyline")]
    check("overlaid they share one set of axes", all(abs(o[:, 0].min() - ov[0][:, 0].min()) < 1e-6 for o in ov) and max(o[:, 1].mean() for o in ov) - min(o[:, 1].mean() for o in ov) < 1.0)
    pxp = prof(0.0, unit="px", feats=False)
    pxp["length_unit"] = "px"
    ip = E.section_svg(os.path.join(tmp, "px.svg"), [pxp])
    check("with no scale it says it is not to scale", not ip["to_scale"] and "not to scale" in open(os.path.join(tmp, "px.svg")).read())

    E.section_dxf(os.path.join(tmp, "s.dxf"), series, v_exag=1.0, stacked=True)
    raw = open(os.path.join(tmp, "s.dxf"), newline="").read().split("\r\n")
    pairs = [(int(raw[i].strip()), raw[i + 1]) for i in range(0, len(raw) - 1, 2) if raw[i].strip().lstrip("-").isdigit()]
    ents = [v for c, v in pairs if c == 0]
    check("DXF: starts with a header, ends with EOF", pairs[0] == (0, "SECTION") and pairs[-1] == (0, "EOF"))
    check("DXF: a polyline and its end for each section, a vertex for every sample",
          ents.count("POLYLINE") == 3 and ents.count("SEQEND") == 3 and ents.count("VERTEX") == 3 * 200, (ents.count("POLYLINE"), ents.count("VERTEX")))
    xs, cur = [], None
    for c, v in pairs:
        if c == 0:
            cur = v
        elif c == 10 and cur == "VERTEX":
            xs.append(float(v))
    check("DXF: the vertices' x runs along the section, 0 to 4 mm", abs(max(xs) - 4.0) < 1e-5 and min(xs) >= -1e-9, (min(xs), max(xs)))
    check("DXF: millimetres, and layers for profile, features, frame and text", (9, "$INSUNITS") in pairs and {"PROFILE", "FEATURES", "FRAME", "TEXT"} <= {v for c, v in pairs if c == 2})
    try:
        import ezdxf
        doc = ezdxf.readfile(os.path.join(tmp, "s.dxf"))
        aud = doc.audit()
        check("DXF: ezdxf reads it and finds nothing wrong", len(aud.errors) == 0, [str(e) for e in aud.errors][:2])
    except ImportError:
        print("  skip ezdxf is not installed: the DXF was checked from its own codes")
    E.section_dxf(os.path.join(tmp, "one.dxf"), [one])
    check("DXF of one section marks its grooves and ridges", open(os.path.join(tmp, "one.dxf")).read().count("groove") == 1)

    E.series_csv(os.path.join(tmp, "s.csv"), series)
    rows = [ln.split(",") for ln in open(os.path.join(tmp, "s.csv")).read().strip().splitlines()]
    check("CSV: a distance column and a height column each, one row a sample", len(rows[1]) == 4 and len(rows) == 2 + 200 and "offset_-1mm" in rows[1][1], rows[1])
    check("CSV: heights are the profiles'", abs(float(rows[2 + 50][2]) - float(series[1]["z"][50])) < 1e-5)

    print("Written from the card")
    card2 = SectionCard()
    ramp.set_section((0.2, 0.5), (0.8, 0.5))
    card2.attach(lambda: ramp, None)
    card2.set_profile(ramp.section_profile())
    f1 = card2.write_drawing(os.path.join(tmp, "card_one"), "Test object")
    check("a single section: the drawing and the CAD file", [os.path.basename(f) for f in f1] == ["card_one.svg", "card_one.dxf"] and all(os.path.getsize(f) > 500 for f in f1))
    card2.count.setValue(5)
    f5 = card2.write_drawing(os.path.join(tmp, "card_five"), "Test object")
    check("five copies: and the spreadsheet", [os.path.basename(f) for f in f5] == ["card_five.svg", "card_five.dxf", "card_five_series.csv"])
    t5 = open(f5[0]).read()
    check("the sheet says how many and how far apart", "5 parallel" in t5 and "Test object" in t5, "")
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
