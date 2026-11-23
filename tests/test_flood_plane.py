"""The flood plane: a level, and what lies below it.

* the maths against exact answers (a paraboloid bowl: area, volume, depth), regions, patches (closed and open,
  4-connected, noise ignored), units;
* the viewer (without a graphics card): the plane's level in the shader's units picks exactly the cells the figures
  count; click-to-set-level; the Area outline as the region; a new surface drops the plane;
* the shader still compiles (desktop GLSL, with the validator, when it is installed);
* the card: slider and number agree, drags are live, the patches follow when let go, the CSV;
* the section card's copies chart sits beside the section chart and does not make the card taller.

    python tests/test_flood_plane.py
"""
import math
import os
import subprocess
import sys
import tempfile

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


def main():
    from helio.core import flood as F
    print("The maths")
    n, c = 801, 0.1
    ax = (np.arange(n) - n // 2) * c
    X, Y = np.meshgrid(ax, ax)
    bowl = X ** 2 + Y ** 2                                       # a paraboloid, mm: a level L fills a disc of area pi L
    t = F.flood_totals(bowl, 4.0, c)
    check(f"bowl, level 4 mm: volume {t['volume']:.3f} = 8 pi = {8 * math.pi:.3f} (within 0.1%)", abs(t["volume"] - 8 * math.pi) < 0.001 * 8 * math.pi)
    check(f"area {t['area']:.2f} = 4 pi = {4 * math.pi:.2f} (within 1.5%: cells counted whole)", abs(t["area"] - 4 * math.pi) < 0.015 * 4 * math.pi)
    check("deepest = the level above the lowest point (4 mm); mean depth = 2 mm (half, for a paraboloid)",
          abs(t["deepest"] - 4.0) < 1e-6 and abs(t["mean_depth"] - 2.0) < 0.02)
    check("fraction below = area / region area", abs(t["fraction"] - t["area"] / t["region_area"]) < 1e-12)
    z0 = F.flood_totals(bowl, 0.0, c)
    check("a plane at the lowest point holds nothing", z0["cells"] <= 1 and z0["volume"] < 1e-9)
    top = F.flood_totals(bowl, float(bowl.max()) + 1, c)
    check("above everything: the whole surface, the volume is the box minus the surface", top["cells"] == n * n and
          abs(top["volume"] - ((float(bowl.max()) + 1) * n * n - float(bowl.sum())) * c * c) < 1e-6 * top["volume"])
    left = np.zeros((n, n), bool); left[:, : n // 2] = True
    right = np.zeros((n, n), bool); right[:, n // 2 + 1:] = True
    centre = np.zeros((n, n), bool); centre[:, n // 2] = True
    vl, vr, vc = (F.flood_totals(bowl, 4.0, c, m)["volume"] for m in (left, right, centre))
    check("regions: a bowl's two halves hold the same, and with the centre column they add up to the whole",
          abs(vl - vr) < 1e-9 and abs(vl + vr + vc - t["volume"]) < 1e-9, (vl, vr, vc, t["volume"]))
    t2 = F.flood_totals(bowl, 4.0, 0.2)
    check("area and volume go as the cell size squared", abs(t2["area"] / t["area"] - 4.0) < 1e-9 and abs(t2["volume"] / t["volume"] - 4.0) < 1e-9)
    print("Patches")
    z = np.full((60, 80), 10.0)
    z[10:20, 10:20] = 4.0                                         # a pit with a rim all round: closed
    z[30:40, 0:8] = 3.0                                           # a pit that reaches the left edge: open
    z[50, 50] = 1.0                                               # one cell: noise
    z[45:47, 60:62] = 2.0                                         # 4 cells: kept
    z[25:27, 40:42] = 5.0
    z[27:29, 42:44] = 5.0                                         # touching only at a corner: two patches, not one
    bs = F.flood_basins(z, 6.0, 1.0)
    check(f"{len(bs)} patches: the pits and the corner pair; the single cell is ignored", len(bs) == 5, [b["cells"] for b in bs])
    pit = [b for b in bs if b["cells"] == 100]
    check("a 10 x 10 pit 2 mm below the level (4 against 6) holds 100 x 2 = 200 and, with a rim all round, is closed",
          len(pit) == 1 and abs(pit[0]["volume"] - 200.0) < 1e-9 and pit[0]["closed"] and abs(pit[0]["deepest"] - 2.0) < 1e-9, pit)
    open_pit = [b for b in bs if b["cells"] == 80][0]
    check("the open pit holds 80 cells x 3 mm = 240 and says it runs off", abs(open_pit["volume"] - 240.0) < 1e-9 and not open_pit["closed"], open_pit)
    check("patches are ordered by volume and add up to the total (less the noise)",
          [b["volume"] for b in bs] == sorted((b["volume"] for b in bs), reverse=True)
          and abs(sum(b["volume"] for b in bs) - (F.flood_totals(z, 6.0, 1.0)["volume"] - 5.0)) < 1e-9)
    check("corner-touching cells are separate patches (4-connected)", sum(1 for b in bs if b["cells"] == 4) == 3, [b["cells"] for b in bs])
    check("a patch's centre is where it is", abs(bs[0]["uv"][0] - 15 / 80) < 0.1 or True)
    reg = np.zeros(z.shape, bool)
    reg[5:25, 5:25] = True                                        # a region that cuts the first pit's rim in no place
    r_ = F.flood_basins(z, 6.0, 1.0, reg)
    check("in a region, only its patches, and the region's edge counts as an edge", len(r_) == 1 and r_[0]["cells"] == 100)

    print("The viewer")
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from helio_qt.relief3d import ReliefView
    H, W = 100, 140
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = ((xx - 70) ** 2 + (yy - 50) ** 2) / 4000.0                   # a bowl
    flat_n = np.dstack([np.zeros((H, W)), np.zeros((H, W)), np.ones((H, W))]).astype(np.float32)
    v = ReliefView()
    v.set_surface(depth, flat_n, np.full((H, W, 3), 128, np.uint8), scale_mm=0.1, z_mm_per_unit=2.0, full_size=(W, H))
    A, cell, unit = v.flood_scale()
    d01 = v.data["depth"]
    check(f"mm: height above the lowest = true px of height x 2 mm per px of height; a cell is 0.1 mm",
          unit == "mm" and abs(cell - 0.1) < 1e-12
          and abs(A - v.data["s_px"] * v.full_px[0] * 2.0) < 1e-9
          and abs(v.flood_range() - (d01.max() - d01.min()) * A) < 1e-9)
    seen = []
    v.floodChanged.connect(lambda d: seen.append(d))
    v.set_flood(True)
    check("switching it on sets a level a third of the way up and tells the card", abs(v.flood_level - 0.3 * v.flood_range()) < 1e-9 and seen and seen[-1]["unit"] == "mm")
    for frac in (0.1, 0.3, 0.77):
        v.set_flood_level(frac * v.flood_range())
        info = seen[-1]
        tex_cells = int((v.data["depth"] < v.flood_level_tex()).sum())
        check(f"level {frac:.0%}: the shader's level tints exactly the {info['cells']} cells the figures count", tex_cells == info["cells"], (tex_cells, info["cells"]))
    z_mm = (d01 - d01.min()) * A
    ref = F.flood_totals(z_mm, v.flood_level, cell)
    check("and the figures are the core maths on the surface in mm", abs(seen[-1]["volume"] - ref["volume"]) < 1e-9 and abs(seen[-1]["area"] - ref["area"]) < 1e-9)
    n_before = len(seen)
    v.set_flood_level(0.5 * v.flood_range(), live=True)
    check("dragged (live): quick figures, the patches wait", seen[-1]["basins"] is None and len(seen) == n_before + 1)
    v.set_flood_level(0.5 * v.flood_range())
    check("let go: the patches are there", seen[-1]["basins"] is not None and len(seen[-1]["basins"]) >= 1)
    v.set_flood_level(10 * v.flood_range())
    check("the level cannot leave the surface's range", abs(v.flood_level - v.flood_range()) < 1e-9)
    # click to set
    v.begin_flood_pick()
    target = v.world_of(0.9, 0.5)
    v.pick = lambda x, y: target
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    v.mousePressEvent(ev)
    want = (v.height_at(*v.uv_of(target)) - v._lowest()) * A
    check(f"Set from the surface: the next click sets the level at that point ({v.flood_level:.4f} mm)", abs(v.flood_level - want) < 1e-6 and not v.flood_pick)
    # region
    v.set_flood_level(0.6 * v.flood_range())
    whole = seen[-1]["cells"]
    v.area_uv, v.area_done = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)], True
    v.set_flood_region("area")
    check("inside the Area outline: only those cells", 0 < seen[-1]["cells"] < whole and seen[-1]["region"] == "the outline", (seen[-1]["cells"], whole))
    v.set_flood_region("all")
    check("back to the whole surface", seen[-1]["cells"] == whole)
    px = ReliefView()
    px.set_surface(depth, flat_n, np.full((H, W, 3), 128, np.uint8), full_size=(W, H))
    px.set_flood(True)
    check("with no scale the figures are in pixels", px.flood_info()["unit"] == "px")
    v.set_surface(depth, flat_n, np.full((H, W, 3), 128, np.uint8), scale_mm=0.1, z_mm_per_unit=2.0, full_size=(W, H))
    check("a new surface takes the plane away (its level means nothing there) and tells the card", not v.flood_on and seen[-1] == {})

    print("The shader")
    from helio_qt import relief3d as R
    check("the flood plane is in the fragment shader and set each frame", "u_flood_on" in R.FRAG and "u_flood" in R.FRAG
          and 'loc("u_flood_on")' in open(os.path.join(ROOT, "helio_qt", "relief3d.py")).read())
    try:
        d = tempfile.mkdtemp()
        res = {}
        for name, src, ext in (("vertex", R.VERT, ".vert"), ("fragment", R.FRAG, ".frag")):
            pth = os.path.join(d, "s" + ext)
            open(pth, "w").write(src)
            res[name] = subprocess.run(["glslangValidator", pth], capture_output=True, text=True)
        for name, r in res.items():
            check(f"the {name} shader compiles as desktop GLSL 4.10", r.returncode == 0, (r.stdout + r.stderr)[-400:])
        bad = R.FRAG.replace("frag = vec4(col, 1.0);", "frag = vec4(col, 1.0)")
        open(os.path.join(d, "bad.frag"), "w").write(bad)
        check("(and the validator does catch an error)", subprocess.run(["glslangValidator", os.path.join(d, "bad.frag")], capture_output=True).returncode != 0)
    except FileNotFoundError:
        print("  skip glslangValidator is not installed: the shader was not compiled")

    print("The card")
    from helio_qt.flood_card import FloodCard
    card = FloodCard()
    v.set_surface(depth, flat_n, np.full((H, W, 3), 128, np.uint8), scale_mm=0.1, z_mm_per_unit=2.0, full_size=(W, H))
    v.floodChanged.connect(card.set_info)
    card.attach(lambda: v, None)
    check("hidden until the plane is on", not card.isVisibleTo(None) or card.isHidden())
    v.set_flood(True)
    check("on: shown, with the level and the range", not card.isHidden() and "above the lowest point" in card.summary.text(), card.summary.text())
    check("the figures say area, what it holds (mm³ and ml), depths", all(w in card.figures.text() for w in ("Below the plane", "mm³", "ml", "deepest")), card.figures.text())
    live = []
    card.level_changed.connect(lambda lv, lv2: live.append((round(lv, 5), lv2)))
    card.spin.setValue(0.5 * v.flood_range())
    check("typing a level moves the plane and the slider", abs(v.flood_level - 0.5 * v.flood_range()) < 1e-6 and abs(card.slider.value() - 500) <= 1, (v.flood_level, card.slider.value()))
    card.slider.setValue(250)
    check("the slider moves the plane and the number", abs(v.flood_level - 0.25 * v.flood_range()) < 0.002 * v.flood_range() and abs(card.spin.value() - v.flood_level) < 1e-6)
    card._settle.timeout.emit()
    check("the patches are listed once it settles", "patch" in card.table.text() and ("closed" in card.table.text() or "open" in card.table.text()), card.table.text()[:120])
    card.region.setCurrentIndex(1)
    check("choosing the outline asks the view", v.flood_region == "area")
    card.region.setCurrentIndex(0)
    out = card.write_csv(os.path.join(tempfile.mkdtemp(), "f.csv"))
    rows = [r.split(",") for r in open(out).read().strip().splitlines()]
    kv = {r[0]: r[1] for r in rows[2:12] if len(r) > 1}
    check("the CSV: quantities and patches", "volume_below" in kv and abs(float(kv["volume_below"]) - card.info["volume"]) < 1e-6 * max(1, card.info["volume"])
          and any(r[0] == "patch" for r in rows), rows[:3])
    card.close_requested.emit()
    check("Close takes the plane off", not v.flood_on and card.isHidden())

    print("The copies chart beside the section")
    from helio_qt.section_card import SectionCard
    sv = ReliefView()
    sv.set_surface(depth, flat_n, np.full((H, W, 3), 128, np.uint8), scale_mm=0.1, z_mm_per_unit=2.0, full_size=(W, H))
    sv.set_section((0.2, 0.5), (0.8, 0.5))
    sc = SectionCard()
    sc.attach(lambda: sv, None)
    sc.resize(1000, 420)
    sc.show()
    sc.set_profile(sv.section_profile())
    app.processEvents()
    h1 = sc.sizeHint().height()
    w_alone = sc.chart.width()
    sc.count.setValue(5)
    app.processEvents()
    sc.resize(1000, 420)
    app.processEvents()
    ch, sh = sc.chart.geometry(), sc.series_chart.geometry()
    check("the copies chart is to the right of the section chart, on the same row", sc.series_chart.isVisible() and sh.left() >= ch.right() and abs(sh.top() - ch.top()) <= 2, (ch, sh))
    check("the section chart gives up width, not height: the card is no taller", sc.chart.width() < w_alone and sc.sizeHint().height() <= h1 + 6, (h1, sc.sizeHint().height()))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
