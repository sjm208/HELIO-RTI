"""Chain lines along the 3D relief's section (helio.application.chain_lines, helio.core.paper_lines).

* a made backlit sheet with chain lines 25 mm apart and writing over them: the section counts them, gets the
  spacing in mm, and the albedo as reference keeps the writing out;
* the picture chosen: the backlit frame with the albedo as its reference, the albedo alone, the height;
* the card: off until ticked, the backlit frame chosen when it comes, the count asked for when the line is let
  go (not while it is dragged), the lines marked on the chart and handed to the views, the transmissive note;
* Inspect's counter: the section's ends in the solve's pixels reach the picture, the scale gives mm, and the
  relief is told where the lines are.

The counting itself is tested in tests/test_paper_lines.py (copied with it from MSI_Reveal).

    python tests/test_chain_lines.py
"""
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile  # noqa: E402
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = FAIL = 0
MM = 0.1                       # mm per pixel (254 dpi)
CHAIN_MM = 25.0
FIRST_MM = 8.0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def sheet(h=700, w=1400, seed=1):
    """(backlit, albedo): chain lines lighter in transmitted light, laid lines across them, writing dark in both."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    period, first = CHAIN_MM / MM, FIRST_MM / MM
    d = np.mod(x - first + period / 2, period) - period / 2
    chain = np.exp(-0.5 * (d / (0.6 / MM / 2.355)) ** 2)
    laid = 0.5 + 0.5 * np.cos(2 * np.pi * y / (1.0 / MM))
    writing = np.zeros((h, w))
    for k in range(9):                                       # strokes of ink across the lines
        r0 = 80 + 60 * k
        writing[r0:r0 + 14, 100 + 30 * k: 1200 - 20 * k] = 1.0
    backlit = 0.6 + 0.05 * chain + 0.02 * laid + rng.normal(0, 0.01, (h, w))
    backlit = backlit * (1 - 0.7 * writing)
    albedo = 0.8 + rng.normal(0, 0.005, (h, w)) - 0.6 * writing
    truth = [first + i * period for i in range(int((w - first) / period) + 1) if first + i * period < w]
    return backlit, albedo, truth


def main():
    from helio.application.chain_lines import (LINE_KINDS, SOURCES, TRANSMISSIVE_NOTE, along, count_along,
                                               pictures)

    print("Counting on a made sheet")
    backlit, albedo, truth = sheet()
    h, w = backlit.shape
    a, b = (2.0, h / 2.0), (w - 3.0, h / 2.0)
    found = count_along(backlit, a, b, kind="chain", swath_mm=40.0, mm_per_px=MM, reference=albedo)
    inside = [t for t in truth if a[0] + 20 < t < b[0] - 20]
    got = [a[0] + p for p in found.positions]
    near = all(min(abs(g - t) for g in got) < 6 for t in inside) if got else False
    check(f"every chain line the section crosses is found ({len(inside)})", near and found.count == len(inside),
          (found.count, [round(g) for g in got], [round(t) for t in inside]))
    check("their spacing is 25 mm", found.mean_spacing is not None
          and abs(found.mean_spacing * MM - CHAIN_MM) < 0.5, found.mean_spacing and found.mean_spacing * MM)
    check("lighter than the sheet, in transmitted light", found.polarity == "light", found.polarity)
    places = along(found, a, b)
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    check("their places along A-B, 0 to 1", all(0 <= t <= 1 for t in places)
          and abs(places[0] * length - found.positions[0]) < 1e-6)
    blank = 0.6 + np.random.default_rng(3).normal(0, 0.01, (400, 900))
    none = count_along(blank, (5, 200), (890, 200), kind="chain", swath_mm=30.0, mm_per_px=MM)
    check("a sheet without chain lines is given none", none.count == 0, none.count)
    no_scale = count_along(backlit, a, b, kind="chain", swath_mm=None, mm_per_px=None, reference=albedo)
    check("without a scale it still counts, in pixels", no_scale.count >= 2 and no_scale.mm_per_px is None,
          no_scale.count)

    print("\nThe picture chosen")
    height = np.random.default_rng(4).normal(0, 1, (h, w))
    img, ref = pictures("backlit", albedo, height, backlit)
    check("the backlit frame, with the albedo as its reference", np.array_equal(img, backlit) and np.array_equal(ref, albedo))
    img, ref = pictures("backlit", albedo, height, None)
    check("…none without a backlit frame", img is None and ref is None)
    img, ref = pictures("albedo", np.dstack([albedo] * 3), height, backlit)
    check("the albedo alone, colour made grey", img.ndim == 2 and np.allclose(img, albedo) and ref is None)
    img, ref = pictures("height", albedo, height, backlit)
    check("the relief's height", np.array_equal(img, height) and ref is None)
    check("the kinds offered are chain, laid and ruling", list(LINE_KINDS) == ["chain", "laid", "ruling"])

    print("\nThe card")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from helio_qt.section_card import SectionCard
    card = SectionCard()
    check("not offered on a card whose view has no counter", not card.lines_box.isVisibleTo(card))
    card.enable_lines(False)
    check("offered once enabled, off until ticked", card.lines_box.isVisibleTo(card) and card.lines_options() is None)
    model = card.lines_source.model()
    check("without a backlit frame: the albedo, the backlit choice greyed",
          card.lines_source.currentData() == "albedo"
          and not model.item(card.lines_source.findData("backlit")).isEnabled())
    card.set_backlit(True)
    check("a backlit frame lined up: it is chosen", card.lines_source.currentData() == "backlit"
          and model.item(card.lines_source.findData("backlit")).isEnabled())
    card.lines_source.setCurrentIndex(card.lines_source.findData("height"))
    card.set_backlit(True)
    check("…and the person's own choice is kept after that", card.lines_source.currentData() == "height")
    card.lines_source.setCurrentIndex(card.lines_source.findData("backlit"))
    asked, marked = [], []
    card.lines_requested.connect(lambda: asked.append(1))
    card.lines_changed.connect(lambda v: marked.append(list(v)))
    prof = dict(dist=np.linspace(0, (w - 5) * MM, 200), z=np.zeros(200), unit="mm", length_unit="mm",
                length=(w - 5) * MM, a_px=a, b_px=b, ra=0.0, rq=0.0, rz=0.0, features=[])
    card.set_profile(prof)
    check("off: a section asks for no count", not asked)
    card.chk_lines.setChecked(True)
    check("ticked: the count is asked for", len(asked) == 1)
    opts = card.lines_options()
    check("the options: chain lines, the backlit frame, a 20 mm strip",
          opts == dict(kind="chain", source="backlit", swath_mm=20.0), opts)
    card.lines_kind.setCurrentIndex(card.lines_kind.findData("laid"))
    check("laid lines: a narrower strip, counted again", card.lines_swath.value() == 6.0 and len(asked) == 2)
    card.lines_swath.setValue(30.0)
    card.lines_kind.setCurrentIndex(card.lines_kind.findData("chain"))
    check("a strip set by hand is kept when the kind changes", card.lines_swath.value() == 30.0)
    card.set_line_count(found, places, "Counted on the backlit frame.")
    check("the lines are marked on the chart and handed to the views",
          card.chart.lines == places and marked[-1] == places)
    check("the card says how many, the spacing in mm and the picture",
          f"{found.count} chain lines" in card.lines_msg.text() and "mm" in card.lines_msg.text()
          and "backlit" in card.lines_msg.text(), card.lines_msg.text())
    n = len(asked)
    card.set_profile(dict(prof, features=None))
    check("an end being dragged: the old marks go, no count yet", card.chart.lines == [] and marked[-1] == []
          and len(asked) == n)
    card.set_profile(prof)
    check("let go: counted again", len(asked) == n + 1)
    card.chk_lines.setChecked(False)
    check("unticked: the marks go", card.chart.lines == [] and marked[-1] == [])

    print("\nInspect's counter")
    from helio_qt.inspect_ws import InspectWorkspace as WS
    card.chk_lines.setChecked(True)
    card.set_backlit(True)
    card.lines_source.setCurrentIndex(card.lines_source.findData("backlit"))
    r = SimpleNamespace(w=w, h=h, albedo_grey_full=albedo, depth_poisson=height)
    marks = []
    card.lines_changed.connect(lambda v: marks.append(v))
    from helio.application.viewing import Display
    shown = np.fliplr(backlit[::2, ::2])                         # drawn at half size, mirrored for viewing
    fake = SimpleNamespace(section_card=card, section_prof=prof, relief=SimpleNamespace(scale_mm=MM),
                           _relief_source=lambda: (r, None), _has_backlit=lambda: True,
                           _backlit_for=lambda rr: backlit, view_name=lambda v: v,
                           win=SimpleNamespace(view="Raking", display=Display(flip_h=True), selection=None),
                           project=SimpleNamespace(result=r), _last_image=shown)
    fake._view_for_count = lambda rr: WS._view_for_count(fake, rr)
    res = WS.count_section_lines(fake)
    check("counted on the backlit frame from the section's ends", res is not None and res.count == found.count,
          res and res.count)
    check("…in mm from the scale", res is not None and res.mm_per_px == MM)
    check("…and the views told where", marks and marks[-1] == along(res, a, b))
    fake._has_backlit, fake._backlit_for = (lambda: False), (lambda rr: None)
    res = WS.count_section_lines(fake)
    check("the backlit frame gone: counted on the albedo, with the transmissive note",
          card.lines_source.currentData() == "albedo" and "backlit" in card.lines_msg.text(),
          card.lines_msg.text())
    card.lines_source.setCurrentIndex(card.lines_source.findData("view"))
    res = WS.count_section_lines(fake)
    check("this view: named on the list, counted on the picture shown",
          card.lines_source.currentText() == "This view: Raking" and "Raking" in card.lines_msg.text()
          and res is not None and res.count == found.count, (card.lines_source.currentText(), res and res.count))
    check("…unmirrored, and at its own size: the lines in the same places",
          res is not None and all(min(abs(2 * p - q) for q in found.positions) < 8 for p in res.positions)
          and abs(res.mean_spacing * res.mm_per_px - CHAIN_MM) < 0.5,
          res and [round(p) for p in res.positions])
    for disp in (Display(flip_v=True), Display(rot180=True), Display(flip_h=True, flip_v=True, rot180=True)):
        from helio.application.viewing import apply_transforms
        fake.win.display, fake._last_image = disp, apply_transforms(albedo, disp)
        back = WS._view_for_count(fake, r)
        check(f"the view put back the solve's way round ({disp.flip_h, disp.flip_v, disp.rot180})",
              np.array_equal(back, albedo))
    fake.win.selection = (100, 50, 700, 450)
    fake._last_image = albedo[::2, ::2]
    fake.win.display = Display()
    part = WS._view_for_count(fake, SimpleNamespace(w=600, h=400))
    check("a selection: the view cut to it, at the view's size", part.shape == (200, 300)
          and np.array_equal(part, albedo[::2, ::2][25:225, 50:350]), part.shape)
    fake.win.selection, fake._last_image = None, None
    res = WS.count_section_lines(fake)
    check("no view drawn yet: said so", res is None and "Show a view" in card.lines_msg.text())
    fake._last_image = shown
    card.lines_source.setCurrentIndex(card.lines_source.findData("backlit"))
    half = SimpleNamespace(w=w * 2, h=h * 2, albedo_grey_full=albedo, depth_poisson=height)  # a picture at half size
    fake._relief_source = lambda: (half, None)
    fake.section_prof = dict(prof, a_px=(a[0] * 2, a[1] * 2), b_px=(b[0] * 2, b[1] * 2))
    fake.relief = SimpleNamespace(scale_mm=MM / 2)
    res = WS.count_section_lines(fake)
    check("a picture at another size: the ends and the scale follow it", res is not None
          and abs(res.mm_per_px - MM) < 1e-12)
    fake.relief = SimpleNamespace(scale_mm=None)
    fake._relief_source = lambda: (r, None)
    fake.section_prof = prof
    res = WS.count_section_lines(fake)
    check("without a scale: counted, and told to set one", res is not None and "Set Scale" in card.lines_msg.text())
    check("the sources have names", set(SOURCES) == {"backlit", "albedo", "height", "view"})

    print("\nThe backlit frame, lined up")
    raw = (np.dstack([backlit] * 3) * 255).astype(np.uint8)          # as loaded: 8-bit BGR
    whole = SimpleNamespace(w=w, h=h)
    holder = SimpleNamespace(win=SimpleNamespace(backlit={"raw": raw, "H": np.eye(3), "ref_shape": (h, w)},
                                                 selection=None),
                             project=SimpleNamespace(result=whole))
    holder._has_backlit = lambda: True
    got = WS._backlit_for(holder, whole)
    check("read as one grey band, in full precision, the solve's size",
          got.ndim == 2 and got.shape == (h, w) and got.dtype == np.float32
          and abs(float(got.mean()) - float(backlit.mean())) < 0.01, (got.shape, got.dtype))
    check("…and kept, not warped again", WS._backlit_for(holder, whole) is got)
    holder.win.selection = (100, 50, 700, 450)
    part = WS._backlit_for(holder, SimpleNamespace(w=600, h=400))
    check("cut to a selection", part.shape == (400, 600) and np.array_equal(part, got[50:450, 100:700]))

    print("\nRoom for the relief")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter, QWidget
    from helio_qt.inspect_ws import UnderCards
    split = QSplitter(Qt.Orientation.Vertical)
    centre = QWidget()
    centre.setMinimumHeight(260)
    split.addWidget(centre)
    card2 = SectionCard()
    card2.enable_lines(False)
    card2.btn_fold.setChecked(False)
    under = UnderCards(split, [card2])
    split.resize(900, 800)
    split.show()

    def settle():
        for _ in range(5):
            app.processEvents()
    settle()
    check("no section: the relief has the whole height", not under.isVisible() and split.sizes()[1] == 0)
    card2.set_profile(prof)
    settle()
    sizes = split.sizes()
    check("a section: the card below a divider, the relief keeping most of the room",
          under.isVisible() and sizes[0] >= 0.55 * sum(sizes) and sizes[0] >= 260, sizes)
    check("the card scrolls in its share rather than squeezing",
          under.area.widget().sizeHint().height() > sizes[1] or under.area.verticalScrollBar().maximum() >= 0)
    card2.btn_fold.setChecked(True)
    settle()
    folded = split.sizes()
    check("folded: the card takes one line, the relief the rest", folded[1] < 120 and folded[0] > sizes[0], folded)
    card2.btn_fold.setChecked(False)
    settle()
    split.moveSplitter(300, 1)
    settle()
    under._remember()
    kept = under.share()
    card2.set_profile(None)
    settle()
    card2.set_profile(prof)
    settle()
    again = split.sizes()
    check("the divider where it was dragged is kept for the next section",
          abs(again[1] / float(sum(again)) - kept) < 0.05 and abs(kept - UnderCards.SHARE) > 0.05, (kept, again))
    card2.set_profile(None)
    settle()
    check("the section cleared: the card goes, the relief has the room back", not under.isVisible())

    print(f"\n{PASS} passed, {FAIL} failed")
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
