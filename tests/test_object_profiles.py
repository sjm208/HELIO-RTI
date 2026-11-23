"""Object profiles and the tools for coins, seals, gems, inscriptions and pages.

* Every profile names views, exports, materials and fields that exist.
* Die axis: medal alignment is 12 h, coin alignment 6 h, either way round.
* Legend unwrap finds a synthetic coin's rim and unrolls the ring at its size.
* Virtual impression mirrors and inverts the surface as a stamp prints.
* The line drawing traces a cut ring and line, as SVG in millimetres.
* Relief match: 1 for the same picture, near 0 for noise.
* Nomisma, EpiDoc and IIIF records carry the capture details, and parse.
* In the window: choosing a profile sets the solver, puts its views first and
  offers its exports; the new tools open, run and save on a real solve.

    python tests/test_object_profiles.py
"""
import json
import math
import os
import sys
import tempfile
import time
import xml.dom.minidom

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.make_golden import test_images  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def synthetic_coin(n=400):
    import cv2
    import numpy as np
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - 210, yy - 195)
    dep = np.where(r < 170, 1.0, 0.0).astype(np.float32)
    dep -= 0.3 * np.exp(-((r - 120) / 3) ** 2)                       # a cut ring
    dep -= 0.3 * np.exp(-((xx - 210) / 3) ** 2) * (r < 100)          # a cut line
    gy, gx = np.gradient(cv2.GaussianBlur(dep, (0, 0), 1.5) * 20)
    nrm = np.dstack([-gx, gy, np.ones_like(gx)])
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    rgb = (np.dstack([dep] * 3) * 200).clip(0, 255).astype(np.uint8)
    return dep, nrm.astype(np.float32), rgb


def pure():
    import numpy as np
    from helio.application import heritage_tools as T
    from helio.application.process_tools import PRESETS
    from helio.domain import object_profiles as OP
    from helio.domain.capture_meta import KEYS
    from helio.domain.registry import RENDER_ENGINES
    from helio.domain.settings import ProcessingSettings
    from helio_qt.publish_catalog import all_formats
    from helio_qt.research import RESEARCH

    print("\nProfiles")
    fmts = {f.key for f in all_formats()}
    for p in OP.PROFILES.values():
        bad = [v for v in p.views if v not in RENDER_ENGINES and v != "Depth map"]
        check(f"{p.key}: its views exist", not bad, bad)
        check(f"{p.key}: its exports exist", set(p.exports) <= fmts, set(p.exports) - fmts)
        check(f"{p.key}: its material and settings exist",
              (not p.material or p.material in PRESETS)
              and set(p.settings) <= set(ProcessingSettings().to_dict()))
        check(f"{p.key}: its tools exist", set(p.tools) <= set(RESEARCH), set(p.tools) - set(RESEARCH))
        keys = [k for _l, k, _h, _t in p.fields]
        check(f"{p.key}: its fields are its own", len(keys) == len(set(keys))
              and not set(keys) & set(KEYS) and "object_type" not in keys)
    check("an unknown or missing type is General", OP.of({}).key == "general"
          and OP.of({"object_type": "nonsense"}).key == "general")

    print("\nDie axis")
    check("upright both: 12 h", OP.die_axis_hours(0, 0) == 12)
    check("reverse upside down: 6 h", OP.die_axis_hours(0, 180) == 6 and OP.die_axis_hours(-90, 90) == 6)
    check("a quarter turn clockwise: 3 h, anticlockwise: 9 h",
          OP.die_axis_hours(10, 100) == 3 and OP.die_axis_hours(0, -90) == 9)

    print("\nLegend unwrap")
    dep, nrm, rgb = synthetic_coin()
    cx, cy, r = T.find_rim(rgb)
    check(f"the rim is found ({cx:.0f}, {cy:.0f}, r {r:.0f})",
          abs(cx - 210) < 4 and abs(cy - 195) < 4 and abs(r - 170) < 5)
    strip = T.unwrap_legend(rgb, cx, cy, 160, 100)
    check(f"the strip is the ring's size ({strip.shape[1]} x {strip.shape[0]})",
          strip.shape[0] == 60 and abs(strip.shape[1] - 2 * np.pi * 130) < 2)
    row = strip[int(round(160 - 120)), :, 0]                       # the cut ring's radius
    check("the cut ring runs straight along the strip", float(row.std()) < 6 and row.mean() < 150,
          (row.mean(), row.std()))
    try:
        T.unwrap_legend(rgb, cx, cy, 100, 99.5)
        check("a ring too narrow is refused", False)
    except ValueError:
        check("a ring too narrow is refused", True)

    print("\nVirtual impression")
    flat = np.zeros((4, 6, 3), np.float32)
    flat[..., 2] = 1
    flat[:, :3, 0] = 0.6                                           # the left half leans right
    flat /= np.linalg.norm(flat, axis=-1, keepdims=True)
    lit = T.virtual_impression(flat, mirror=False, invert=False, azimuth=90, elevation=30,
                               colour=(255, 255, 255), ambient=0)[..., 0].astype(int)
    check("as is: what leans to the light is brighter", lit[0, 0] > lit[0, 5])
    stamp = T.virtual_impression(flat, mirror=True, invert=True, azimuth=90, elevation=30,
                                 colour=(255, 255, 255), ambient=0)[..., 0].astype(int)
    check("an impression: the lean moves to the other side and still faces the light",
          stamp[0, 5] > stamp[0, 0] and stamp[0, 5] == lit[0, 0])
    raised = T.virtual_impression(flat, mirror=False, invert=True, azimuth=90, elevation=30,
                                  colour=(255, 255, 255), ambient=0)[..., 0].astype(int)
    check("raised letters: same side, turned away", raised[0, 0] < lit[0, 0])

    print("\nLine drawing")
    lines = T.line_drawing(dep)
    check(f"the cut ring and line are traced ({len(lines['grooves'])} groove lines)",
          len(lines["grooves"]) >= 2)
    pts = np.vstack(lines["grooves"])
    rr = np.hypot(pts[:, 0] - 210, pts[:, 1] - 195)
    on = (np.abs(rr - 120) < 4) | (np.abs(pts[:, 0] - 210) < 4) | (np.abs(rr - 171) < 4)
    check("…along the cuts and the rim's foot, nowhere else", float(np.mean(on)) > 0.95)
    out = tempfile.mkdtemp()
    svg = T.line_drawing_svg(os.path.join(out, "d.svg"), lines, mm_per_px=0.05, title="Coin & co")
    doc = xml.dom.minidom.parse(svg)
    root = doc.documentElement
    check("the SVG parses, sized in mm, a layer each and a scale bar",
          root.getAttribute("width") == "20.000mm" and
          {g.getAttribute("id") for g in doc.getElementsByTagName("g")} >= {"grooves", "crests", "scale"})
    check("bar lengths are round", T.bar_length(4.3) == 2 and T.bar_length(17) == 10 and T.bar_length(5) == 5)

    print("\nRelief match")
    check("the same picture matches 1", abs(T.relief_match(rgb, rgb.copy()) - 1) < 1e-6)
    noise = (np.random.default_rng(1).random(rgb.shape) * 255).astype(np.uint8)
    check("noise matches near 0", T.relief_match(rgb, noise) < 0.1)

    print("\nRecords")
    from helio.application.export_scholarly import epidoc_record, iiif_image_manifest, nomisma_record
    meta = {"object_type": "coin", "title": "Denarius of Hadrian", "denomination": "denarius",
            "weight_g": "3.21", "diameter_mm": "18.5", "die_axis": "6", "face": "Obverse",
            "mint": "Rome", "institution": "Museum & Co"}
    files = nomisma_record(os.path.join(out, "n"), meta, "den_1", images=["den_1_colour.jpg"])
    ld = json.load(open(files[1]))
    check("Nomisma: weight, diameter and axis typed", ld["nmo:hasWeight"]["@value"] == 3.21
          and ld["nmo:hasDiameter"]["@value"] == 18.5 and ld["nmo:hasAxis"]["@value"] == 6
          and ld["nmo:hasMint"] == "Rome" and "nmo:hasObverse" in ld)
    check("…and blanks left out", "nmo:hasLegend" not in ld)
    import csv
    rows = list(csv.reader(open(files[0])))
    check("Nomisma CSV: one header, one row", len(rows) == 2 and rows[1][rows[0].index("nmo:hasAxis")] == "6")
    ins = {"object_type": "inscription", "title": "Dedication <to> Hadrian", "material": "marble",
           "letter_height_mm": "20", "findspot": "Ephesos", "text_ref": "EDH HD000001"}
    ep = epidoc_record(os.path.join(out, "e.xml"), ins, "ded_1", images=["a.jpg", "b.jpg"], w=400, h=300,
                       mm_per_px=0.1)
    d = xml.dom.minidom.parse(ep)
    t = lambda tag: [n.firstChild.data if n.firstChild else "" for n in d.getElementsByTagName(tag)]  # noqa: E731
    check("EpiDoc parses, the title escaped", t("title") == ["Dedication <to> Hadrian"])
    check("…with material, letter height, findspot, corpus number and size",
          t("material") == ["marble"] and "20" in t("height") and t("placeName") == ["Ephesos"]
          and "EDH HD000001" in t("idno") and t("width") == ["40.0"])
    check("…the images as the facsimile", len(d.getElementsByTagName("graphic")) == 2)
    shots = [("Colour", "c.jpg", 400, 300), ("Relief", "r.jpg", 400, 300)]
    mf = iiif_image_manifest(out, "ded_1", ins, shots, mm_per_px=0.1)
    man = json.load(open(mf[0]))
    check("IIIF: a canvas per image, with its size and physical scale",
          len(man["items"]) == 2 and man["items"][1]["width"] == 400
          and man["items"][0]["service"][0]["physicalScale"] == 0.1
          and man["items"][1]["items"][0]["items"][0]["body"]["id"].endswith("/r.jpg"))
    check("…the placeholder address explained", "example.org" in open(mf[1]).read())


def window():
    import cv2
    import numpy as np
    from helio.application.heritage_tools import outline_points
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    import helio_qt.research as rs
    import helio_qt.window as hw
    from helio_qt import publish
    from helio_qt.dialogs import CaptureDetailsDialog
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.open_paths([test_images(PASS, FAIL)])

    print("\nChoosing a profile")
    d = CaptureDetailsDialog(w)
    d.kind.setCurrentIndex(d.kind.findData("coin"))
    check("the coin's fields appear", "die_axis" in d.disc_edits and "weight_g" in d.disc_edits)
    d.disc_edits["weight_g"].setText("3.21")
    d.kind.setCurrentIndex(d.kind.findData("seal"))
    check("…another type shows its own", "seal_kind" in d.disc_edits and "weight_g" not in d.disc_edits)
    d.kind.setCurrentIndex(d.kind.findData("coin"))
    check("…and coming back keeps what was typed", d.disc_edits["weight_g"].text() == "3.21")
    d.save()
    p = w.project
    check("saved in the capture details", p.capture_meta.get("object_type") == "coin"
          and p.capture_meta.get("weight_g") == "3.21")
    check("the solver is set for metal", p.settings.normal_engine == "Sparse-Outlier LS (RPCA)"
          and p.settings.metal_aware)
    ins = w.workspaces["Inspect"]
    check("its views come first in Inspect", not ins.suggested_top.isHidden()
          and ins.suggested_top.child(0).data(0, 256) == "Robust PBR (MVS)")
    check("…and stay in their families too", "Robust PBR (MVS)" in ins.items)
    pub = w.workspaces["Publish"]
    check("its exports are offered first in Publish",
          list(pub.all_presets())[0] == "For this object: Coin, medal or token")
    d = CaptureDetailsDialog(w)
    d.kind.setCurrentIndex(d.kind.findData("general"))
    d.save()
    check("General: no type stored, nothing first, the coin's fields kept",
          "object_type" not in p.capture_meta and ins.suggested_top.isHidden()
          and p.capture_meta.get("weight_g") == "3.21")
    meta = dict(p.capture_meta, object_type="inscription", material="marble")
    p.capture_meta = meta

    print("\nOn a solve")
    w.registry.trigger("process.solver:Uncalibrated PCA")
    w.registry.trigger("process.run")
    t0 = time.time()
    while p.busy and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.01)
    check("solved", p.solved)

    def run(dlg):
        dlg.run()
        t0 = time.time()
        while dlg.result is None and time.time() - t0 < 120:
            app.processEvents()
            time.sleep(0.01)
        return dlg.result
    out = tempfile.mkdtemp(prefix="helio_prof_")
    dlg = w.open_research("unwrap")
    app.processEvents()
    n = dlg.find_objects(quiet=True)
    if n > 1:
        page_shown = dlg._page
        j = n - 1
        ob = dlg.objects[j]
        m = dlg._map
        dlg._press(ob["cx"] * m["k"], ob["cy"] * m["k"], 1.0, False)
        check(f"Find objects rings {n} on the page; a click on one picks it",
              page_shown and not dlg._page and dlg.which.currentIndex() == j
              and math.hypot(dlg.outline["cx"] - ob["cx"], dlg.outline["cy"] - ob["cy"])
              < 0.35 * math.sqrt(ob["a"] * ob["b"]))
    else:
        check(f"Find objects on the page ({n} found) leaves an outline to unwrap", dlg.outline is not None)
    H_, W_ = dlg.img.shape[:2]
    dlg.objects = [dict(cx=W_ * x, cy=H_ * 0.5, a=W_ * 0.08, b=W_ * 0.1, turn=0.0, shape="oval", score=1.0)
                   for x in (0.3, 0.7)]
    dlg.which.clear()
    dlg.which.addItems(["Object 1", "Object 2"])
    dlg.show_page()
    m = dlg._map
    dlg._press(W_ * 0.7 * m["k"], H_ * 0.5 * m["k"], 1.0, False)
    check("on a page of several, a click on the second picks it and shows it alone",
          not dlg._page and dlg.which.currentIndex() == 1
          and abs(dlg.outline["cx"] - W_ * 0.7) < 0.03 * W_ and abs(dlg.outline["cy"] - H_ * 0.5) < 0.03 * W_)
    # no limit in the way: small as the object needs, one side of an oval alone, the depth by its line
    dlg.size.set_value(5)
    check("Edge goes down to 5%, and up to 300%", dlg.size.value() == 5 and dlg.size.spin.maximum() == 300)
    dlg.size.set_value(100)
    dlg.shape.set_current("oval")
    dlg.outline = dict(cx=W_ * 0.5, cy=H_ * 0.5, a=W_ * 0.2, b=H_ * 0.3, turn=0.0, shape="oval")
    dlg._ring()

    def on_screen(x, y):
        m = dlg._map
        return (x - m["x0"]) * m["k"], (y - m["y0"]) * m["k"]

    def drag(a, b, alt=False):
        dlg._press(*on_screen(*a), 1.0, alt)
        dlg._drag(*on_screen(*b))
        dlg._release()
    o0 = dict(dlg.outline)
    cx, cy = o0["cx"], o0["cy"]
    drag((cx + o0["a"], cy), (cx + 0.1 * o0["a"], cy))
    check("an oval's side dragged in moves that side alone, down to a tenth",
          abs(dlg.outline["a"] - 0.1 * o0["a"]) < 2 and abs(dlg.outline["b"] - o0["b"]) < 0.5, dlg.outline)
    a1, b1 = dlg.outline["a"], dlg.outline["b"]
    drag((cx, cy + b1), (cx, cy + 0.5 * b1), alt=True)
    check("Option-drag resizes both, in proportion",
          abs(dlg.outline["a"] - 0.5 * a1) < 2 and abs(dlg.outline["b"] - 0.5 * b1) < 2, dlg.outline)
    dlg.shape.set_current("circle")
    dlg.outline = dict(cx=cx, cy=cy, a=100.0, b=100.0, turn=0.0, shape="circle")
    dlg.width_.set_value(30)
    dlg._ring()
    drag((cx + 70, cy), (cx + 40, cy))
    check("the inner line dragged sets the band's depth", abs(dlg.width_.value() - 60) < 1, dlg.width_.value())
    dlg.outline = dict(cx=cx, cy=cy, a=20.0, b=20.0, turn=0.0, shape="circle")
    dlg._ring()
    drag((cx, cy), (cx + 30, cy))
    check("a tiny ring still moves when dragged from inside", abs(dlg.outline["cx"] - (cx + 30)) < 2, dlg.outline)
    dlg.width_.set_value(20)
    dlg.size.set_value(97)
    dlg.find()
    res = run(dlg)
    strip = res["strip"] if res else None
    check("Legend Unwrap makes a strip, read from the relief, with a scale beneath",
          strip is not None and strip.shape[1] > strip.shape[0] and res["band"] is not None
          and res["band"].shape[1] == strip.shape[1] and res["record"]["source"] == "Relief")
    w.display.layer = "Combined Colour"
    dlg.source.setCurrentText("View on screen")
    colour = dlg._shown().copy()
    w.display.layer = "Shading"
    dlg._refresh_view()
    clay = dlg._shown()
    check("View on screen follows the view: Clay after Colour",
          float(np.abs(clay.astype(np.float32) - colour.astype(np.float32)).mean()) > 2
          and float(np.std(clay.astype(np.float32), axis=2).mean()) < 1.0)
    dlg.source.setCurrentText("Another view…")
    dlg.other_view.setCurrentText("Positive Openness")
    dlg.other_layer.setCurrentText("Clay")
    dlg._draw_other()
    t0 = time.time()
    while "other" not in dlg.pics and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    dlg.result = None
    res = run(dlg)
    check("…and Another view reads from any view, here Positive Openness",
          res is not None and res["record"]["source"] == "Positive Openness")
    dlg.shape.set_current("oval")
    dlg._shape_changed()
    dlg.source.setCurrentText("Even raking light")
    dlg.mirror.setChecked(True)
    dlg.result = None
    res = run(dlg)
    check("…an oval, mirrored, in the even raking light",
          res is not None and res["record"]["shape"] == "Oval" and res["record"]["mirrored"]
          and res["record"]["light_deg"] == 90)
    from helio.application.heritage_tools import free_anchors

    def at(X, Y):                              # a point on the picture, where the preview shows it
        m = dlg._map
        return (X - m["x0"]) * m["k"] * m["r"] + m["ox"], (Y - m["y0"]) * m["k"] * m["r"] + m["oy"]
    dlg.shape.set_current("circle")
    dlg._shape_changed()
    o0 = dict(dlg.outline)
    dlg._press(*at(o0["cx"], o0["cy"]), 1.0, False)
    dlg._drag(*at(o0["cx"] + 10, o0["cy"] - 6))
    dlg._release()
    check("dragging inside the outline moves it",
          abs(dlg.outline["cx"] - o0["cx"] - 10) < 0.6 and abs(dlg.outline["cy"] - o0["cy"] + 6) < 0.6
          and dlg.result is None and not dlg.btn_save.isEnabled())
    o1, _w = dlg.ring()
    dlg._press(*at(o1["cx"] + o1["a"], o1["cy"]), 1.0, False)
    dlg._drag(*at(o1["cx"] + 1.1 * o1["a"], o1["cy"]))
    dlg._release()
    o2, _w = dlg.ring()
    check(f"dragging its edge resizes it ({o1['a']:.0f} -> {o2['a']:.0f} px)", abs(o2["a"] / o1["a"] - 1.1) < 0.02)
    dlg.shape.set_current("free")
    dlg._shape_changed()
    A = free_anchors(dlg.ring()[0])
    P2 = outline_points(dlg.ring()[0], 360)
    rad = np.hypot(*(P2 - [o2["cx"], o2["cy"]]).T)
    check(f"Free shape starts from the outline on screen ({len(A)} points, radius {rad.mean():.1f} "
          f"for {o2['a']:.1f}, centre {dlg.ring()[0]['cx'] - o2['cx']:+.1f})",
          len(A) == 16 and abs(rad.mean() - o2["a"]) < 1.5)
    dlg._press(*at(*A[2]), 1.0, False)
    dlg._drag(*at(A[2][0] + 8, A[2][1] + 5))
    dlg._release()
    B = free_anchors(dlg.ring()[0])
    check("dragging a point moves that point alone", np.allclose(B[2], A[2] + (8, 5), atol=0.6)
          and np.allclose(np.delete(B, 2, axis=0), np.delete(A, 2, axis=0), atol=0.6))
    dlg._press(*at(*B[5]), 1.0, True)
    dlg._release()
    C = free_anchors(dlg.ring()[0])
    mid = (C[7] + C[8]) / 2
    Pm = outline_points(dlg.ring()[0], 720)
    near = Pm[np.argmin(np.hypot(*(Pm - mid).T))]
    dlg._double(*at(*near), 1.0)
    check("Option-click removes a point, a double-click on the edge adds one",
          len(C) == 15 and len(free_anchors(dlg.ring()[0])) == 16)
    res = run(dlg)
    check("…and the free shape unwraps", res is not None and res["record"]["shape"] == "Free shape")
    dlg.with_pic.setChecked(True)
    dlg.reads.setText("+ SIGILLVM TEST")
    rs.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(out, "legend.png"), ""))
    dlg.save()
    rec = json.load(open(os.path.join(out, "legend.json"))) if os.path.exists(os.path.join(out, "legend.json")) else {}
    check("…and saves it with its record and the legend, which goes to Capture details",
          os.path.exists(os.path.join(out, "legend.png")) and rec.get("legend") == "+ SIGILLVM TEST"
          and p.capture_meta.get("legend") == "+ SIGILLVM TEST", rec)
    saved = cv2.imread(os.path.join(out, "legend.png"))
    composed = dlg._composed(dlg.result)
    check("…with the object above the strip when asked", saved is not None
          and saved.shape[1] == composed.shape[1] and saved.shape[0] > composed.shape[0] + 50)
    n0 = len(getattr(w, "sheet_shots", []))
    w.flash = lambda *a, **k: None
    dlg.add_to_sheet()
    check("Add to sheet puts it on the publication sheet, captioned with the legend",
          len(w.sheet_shots) == n0 + 1 and w.sheet_shots[-1][2] == "tool"
          and w.sheet_shots[-1][0] == "Legend unrolled: + SIGILLVM TEST"
          and w.sheet_shots[-1][1].shape[0] == saved.shape[0])
    dlg.close()
    dlg = w.open_research("impression")
    check("Virtual Impression starts on raised letters for an inscription", dlg.kind.current() == "raised")
    imp = run(dlg)
    check("…and makes one the solve's size", imp is not None and imp.shape[:2] == (p.result.h, p.result.w))
    dlg.close()
    dlg = w.open_research("linedraw")
    res = run(dlg)
    check("Line Drawing traces lines", res is not None and len(res["grooves"]) > 0)
    rs.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (os.path.join(out, "draw.svg"), ""))
    dlg.save()
    check("…and saves the SVG with its photograph", os.path.exists(os.path.join(out, "draw.svg"))
          and os.path.exists(os.path.join(out, "draw_photo.jpg")))
    dlg.close()
    dlg = w.open_research("recto")
    dlg.kind.set_current("coin")
    dlg._kind("coin")
    check("Two Faces: obverse and reverse, with the die axis", dlg.side.current() == "Obverse"
          and dlg.axis_box.isVisibleTo(dlg) and dlg.die_axis() is None)
    other = os.path.join(out, "legend.png")
    dlg.choose_partner(other)
    dlg.dial_rev.dial.setValue(180)
    check("…6 h when the reverse stands upside down", dlg.die_axis() == 6)
    dlg.record_axis()
    check("…recorded in Capture details", p.capture_meta.get("die_axis") == "6"
          and p.capture_meta.get("face") == "Obverse")
    dlg.close()

    print("\nPublish")
    src = w.publish_source()
    for key, want in (("iiif_images", "manifest.json"), ("epidoc", ".xml"), ("nomisma", ".jsonld"),
                      ("line_drawing", ".svg")):
        files = publish.export_one(key, src, out)
        check(f"{key} writes {want}", any(f.endswith(want) for f in files) and all(map(os.path.exists, files)),
              files)
    man = json.load(open(os.path.join(out, f"{publish.file_stem(src.name)}_iiif", "manifest.json")))
    check("the IIIF manifest has the colour and the inscription's views",
          man["items"][0]["label"]["en"] == ["Colour"] and len(man["items"]) >= 4)
    w.close()


def main():
    print("=" * 60)
    print("Object profiles and their tools")
    print("=" * 60)
    pure()
    window()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
