"""The publication sheet carries your annotations and your parallel sections.

* the annotation layer (arrows, ellipses, lines, dots, numbers, text, ...) is drawn on the large view in its own
  colours; every one with a caption gets a tag n1, n2 ... and a legend line with the caption; rectangles stay the
  boxed details; measurements and numbered dots add no caption line; too many are summed up
* a section with copies is drawn with its parallel lines faintly on the large view, and its chart is the stacked
  series, with the offsets; a section alone is unchanged
* the whole sheet writes (JPEG and PDF)

    python tests/test_sheet_notes.py
"""
import os
import sys
import tempfile

import numpy as np

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


def profile(a, b, offset=0.0, n=120, amp=1.0):
    d = np.linspace(0, 40.0, n)
    z = amp * np.sin(d / 3.0 + offset) + 0.2 * offset
    return dict(dist=d, z=z, unit="mm height", length_unit="mm", length=40.0, a_px=a, b_px=b, rq=0.4, offset=offset)


def count(img, rgb, tol=60):
    a = np.asarray(img).astype(int)
    return int((np.abs(a - np.array(rgb)).max(axis=2) < tol).sum())


def main():
    from PIL import Image, ImageDraw
    from helio.application import annotations as A
    from helio.application import pub_sheet as PS

    H, W = 300, 400
    yy, xx = np.mgrid[0:H, 0:W]
    pic = np.dstack([(60 + 120 * np.sin(xx / 25.0) ** 2).astype(np.uint8)] * 3)
    albedo = np.full((H, W, 3), 200, np.uint8)
    albedo[40:260, 60:340] = (150, 120, 90)
    full = (W * 4, H * 4)
    GREEN, MAG = "#00ff00", "#ff00ff"

    def sheet(**kw):
        return PS.Sheet(title="Test", full_size=full, albedo=albedo,
                        views=[PS.View("Standard PBR", pic, "relit"), PS.View("Raking Composite", pic, "raking")],
                        hero=PS.View("Raking Composite", pic, "raking"), mm_per_px=0.05, **kw)

    print("The annotation layer")
    notes = [A.make("arrow", [(400, 400), (700, 600)], color=GREEN, text="Tool mark, upper left"),
             A.make("ellipse", [(800, 300), (1100, 500)], color=GREEN, text="Pricking"),
             A.make("dot", [(900, 800)], color=GREEN, text="Pinhole"),
             A.make("number", [(1000, 900)], color=GREEN, text="7"),
             A.make("measure", [(300, 900), (700, 900)], color=GREEN),
             A.make("text", [(1200, 200), (1250, 250)], color=GREEN, text="Later repair"),
             A.make("polygon", [(300, 1000), (500, 1000), (400, 1150)], color=GREEN, text="Loss"),
             A.make("rect", [(100, 100), (200, 200)], color=GREEN, text="Detail box")]
    s = sheet(notes=notes)
    check("the rectangle is left to the details; the rest are notes", [a["type"] for a in PS._note_list(s)] ==
          ["arrow", "ellipse", "dot", "number", "measure", "text", "polygon"])
    im = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(im)
    legend = []
    PS._draw_notes(d, s, lambda x, y: (x / 2.0, y / 2.0), legend)
    check("one legend line for each caption (not the measurement or the number)", [t for t, _c, _x in legend] ==
          ["n1", "n2", "n3", "n4", "n5"] and [x for _t, _c, x in legend] ==
          ["Tool mark, upper left", "Pricking", "Pinhole", "Later repair", "Loss"], legend)
    check("they are drawn in the colour they were given", count(im, (0, 255, 0)) > 800, count(im, (0, 255, 0)))
    many = sheet(notes=[A.make("dot", [(10 * i, 10)], text=f"n{i}") for i in range(20)])
    leg2 = []
    PS._draw_notes(ImageDraw.Draw(Image.new("RGB", (800, 600))), many, lambda x, y: (x, y), leg2)
    check("more than the limit are summed up", len(leg2) == PS.NOTES_MAX + 1 and "more" in leg2[-1][2], leg2[-1:])

    print("Parallel sections")
    main_p = profile((400, 400), (1400, 400), 0.0)
    copies = [profile((400, 400 + 120 * k), (1400, 400 + 120 * k), 0.5 * k, amp=1.0 + 0.1 * k) for k in (-2, -1, 0, 1, 2)]
    for c, k in zip(copies, (-2, -1, 0, 1, 2)):
        c["offset"] = 0.5 * k
    sec_alone = dict(main_p, rq=0.4)
    sec_series = dict(main_p, series=copies, spacing=0.5)
    LIGHT = PS._lighter(PS.SECTION)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, kw in (("plain", dict(sections=[sec_alone])), ("series", dict(sections=[sec_series])),
                         ("both", dict(sections=[sec_series], notes=notes))):
            files = PS.make(sheet(**kw), os.path.join(tmp, name))
            out[name] = (files, Image.open(files[0]).convert("RGB"), Image.open(files[0]).size)
            check(f"the {name} sheet writes a JPEG and a PDF", any(f.endswith(".pdf") for f in files) and os.path.exists(files[0]), files)
        check("the copies' faint lines are on the large view", count(out["series"][1], LIGHT, 25) > count(out["plain"][1], LIGHT, 25) + 300,
              (count(out["series"][1], LIGHT, 25), count(out["plain"][1], LIGHT, 25)))
        full_pages = lambda k: sum(1 for f in out[k][0] if f.endswith(".jpg"))      # noqa: E731
        check("the series chart is taller than the single profile (more page used)",
              (full_pages("series"), out["series"][2][1]) >= (full_pages("plain"), out["plain"][2][1]) and
              np.asarray(out["series"][1]).std() != np.asarray(out["plain"][1]).std())
        check("the notes appear on the sheet in their colour", count(out["both"][1], (0, 255, 0), 90) > 500 and
              count(out["plain"][1], (0, 255, 0), 90) < 50, (count(out["both"][1], (0, 255, 0), 90), count(out["plain"][1], (0, 255, 0), 90)))
    print("The area beside the graph")
    full_w = PS.PAGE_W - 2 * PS.M
    cl, cw, sx, sw = PS._snip_layout(("L", None))
    check("picture on the left: the graph starts after it, and together they fill the width",
          sx == PS.M and cl > sx + sw and cl + cw == PS.M + full_w and sw > 0, (cl, cw, sx, sw))
    cl, cw, sx, sw = PS._snip_layout(("R", None))
    check("picture on the right: the graph starts at the margin", cl == PS.M and sx > cl + cw and sx + sw == PS.M + full_w, (cl, cw, sx, sw))
    check("with no picture the graph has the whole width", PS._snip_layout(None) == (PS.M, full_w, PS.M, 0))
    s_ = sheet(sections=[sec_alone])
    build = PS._snippet(s_, sec_alone, "A", "B")
    tile = np.asarray(build(900, 500))
    check("the area picture is cut to the box it is given", tile.shape == (500, 900, 3), tile.shape)
    check("and carries the section's line and its ends", count(tile, PS.SECTION, 30) > 400, count(tile, PS.SECTION, 30))
    far = PS._snippet(s_, dict(sec_alone, a_px=(1500.0, 1100.0), b_px=(1560.0, 1100.0)), "C", "D")(600, 600)
    check("a line at the edge of the picture still gives a full tile", np.asarray(far).shape == (600, 600, 3))
    check("none can be made without a picture of the object", PS._snippet(PS.Sheet(title="x", full_size=full, albedo=albedo, views=[]), sec_alone, "A", "B") is None)
    with tempfile.TemporaryDirectory() as tmp:
        three = [dict(sec_alone, a_px=(400.0, 300.0 + 200 * i), b_px=(1400.0, 300.0 + 200 * i)) for i in range(3)]
        f3 = PS.make(sheet(sections=three), os.path.join(tmp, "three"))
        check("three sections write a sheet, the picture alternating sides", any(f.endswith(".pdf") for f in f3), f3)
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
