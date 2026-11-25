"""Hand-held flash evened out by a grey card (CAL-003).

A flat object with a grey card beside it, photographed under eight flashes
of different strength (as a flash carried round by hand is): the card is
found, each flash's strength is measured on it, and the solve leaves the
flat object flat, where without the card it comes out tilted.
"""
import os
import sys
import tempfile
import time

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


W, H = 900, 700
CARD = (700, 250, 820, 420)                        # x1, y1, x2, y2: the grey card
TRUE_STRENGTH = [1.0, 1.6, 0.7, 1.3, 0.85, 1.9, 1.1, 0.75]


def lights():
    import numpy as np
    out = []
    for k in range(8):
        a, el = 2 * np.pi * k / 8, np.radians(30 + 5 * (k % 3))
        out.append([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
    return np.array(out)


def capture(folder):
    """Eight frames: an embossed flat panel, a matte grey card, black around."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(5)
    z = np.zeros((H, W), np.float32)
    for _ in range(140):
        x, y = rng.uniform(60, 640), rng.uniform(60, 640)
        cv2.circle(z, (int(x), int(y)), int(rng.uniform(5, 15)), 1.0, -1)
    z = cv2.GaussianBlur(z, (0, 0), 2.5) * 5.0
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(z)])            # +Y up, as the lights
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    alb = np.zeros((H, W), np.float32)
    alb[40:660, 40:660] = 0.55                            # the object
    x1, y1, x2, y2 = CARD
    alb[y1:y2, x1:x2] = 0.45                              # the card: flat, n = (0, 0, 1)
    n[y1:y2, x1:x2] = (0.0, 0.0, 1.0)
    # The object has a colour, as leather, parchment and stone do; the card is neutral.
    tint = np.ones((H, W, 3), np.float32)
    tint[40:660, 40:660] = (0.55, 0.8, 1.0)              # BGR: a warm brown
    tint[y1:y2, x1:x2] = 1.0
    paths = []
    for k, (L, s) in enumerate(zip(lights(), TRUE_STRENGTH)):
        shade = alb * s * np.clip(n @ L, 0, 1) * 1.35
        img = np.clip(shade[..., None] * tint, 0, 1)
        img = np.clip(img * 255 + rng.normal(0, 1.0, img.shape), 0, 255).astype(np.uint8)
        p = os.path.join(folder, f"f{k}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


# The fixtures write light straight into 8-bit files (no sRGB curve), so they
# are solved as stored; falloff_capture(srgb=True) checks the default, Automatic.
TONE = "stored"
FALLOFF = 0.5            # how much brighter a flash is on its own side of the frame


def falloff_capture(folder, srgb=False):
    """A flat, grainy, embossed panel filling the frame, under eight flashes
    of different strength that each light their own side more: a flash
    held by hand off to one side, close in.  Returns (paths, left margin,
    right margin), the margins plain flat board."""
    import cv2
    import numpy as np
    rng = np.random.default_rng(11)
    z = np.zeros((H, W), np.float32)
    for _ in range(90):
        x, y = rng.uniform(260, 640), rng.uniform(160, 540)
        cv2.circle(z, (int(x), int(y)), int(rng.uniform(5, 14)), 1.0, -1)
    z = cv2.GaussianBlur(z, (0, 0), 2.5) * 5.0
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    alb = np.clip(0.5 + rng.normal(0, 0.05, (H, W)), 0.2, 0.8).astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    u, v = xx / W - 0.5, yy / H - 0.5
    paths = []
    for k, (L, s) in enumerate(zip(lights(), TRUE_STRENGTH)):
        # brighter towards the flash: east is +u, north is -v (rows run south)
        g = s * (1.0 + FALLOFF * (u * L[0] - v * L[1]) / np.hypot(L[0], L[1]))
        img = np.clip(alb * g * np.clip(n @ L, 0, 1) * 1.1, 0, 1)
        if srgb:                                         # as a camera writes it
            from helio.core.radiometry import linear_to_srgb
            img = linear_to_srgb(img * 255.0) / 255.0
        img = np.clip(img * 255 + rng.normal(0, 0.8, img.shape), 0, 255).astype(np.uint8)
        p = os.path.join(folder, f"g{k}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths, (20, 60, 180, 640), (720, 60, 880, 640)


def tilt(normals, box):
    import numpy as np
    x1, y1, x2, y2 = box
    m = np.median(normals[y1:y2, x1:x2].reshape(-1, 3), axis=0)
    m /= np.linalg.norm(m)
    return float(np.degrees(np.arccos(np.clip(abs(m[2]), -1, 1))))


def main():
    import numpy as np
    print("=" * 60)
    print("Hand-held flash and the grey card")
    print("=" * 60)
    from helio.core.flash import find_grey_reference, strengths, box_text
    tmp = tempfile.mkdtemp(prefix="helio_flash_")
    paths = capture(tmp)
    ref = find_grey_reference(paths)
    inside = ref is not None and (CARD[0] <= ref["box"][0] and CARD[1] <= ref["box"][1]
                                  and ref["box"][2] <= CARD[2] and ref["box"][3] <= CARD[3])
    check("the grey card is found, and the box lies on it", inside, ref)
    import cv2
    stack = np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]).astype(np.float32)
    st = strengths(stack, ref["box"], lights()) if ref else None
    want = np.array(TRUE_STRENGTH) / np.median(TRUE_STRENGTH)
    err = float(np.max(np.abs(st["strength"] / want - 1))) if st else 1.0
    check(f"each flash's strength is read from the card to within 3 % ({err * 100:.1f} %)",
          err < 0.03, st and np.round(st["strength"], 3))
    check("…the spread is reported (x2.7 here) and no frame is brightened",
          st and abs(st["spread"] - max(TRUE_STRENGTH) / min(TRUE_STRENGTH)) < 0.1
          and float(st["factor"].max()) <= 1.0 + 1e-9, st and st["spread"])

    from helio.application.pipeline import SolveRequest, solve
    from helio.domain.settings import ProcessingSettings
    obj = (100, 100, 600, 600)
    plain = solve(SolveRequest(paths=paths, lights=lights(), settings=ProcessingSettings(photo_tone=TONE)))
    even = solve(SolveRequest(paths=paths, lights=lights(), settings=ProcessingSettings(photo_tone=TONE,
        even_flash=True, flash_ref=box_text(ref["box"]))))
    t0, t1 = tilt(plain.normals_full, obj), tilt(even.normals_full, obj)
    check(f"without the card the flat object comes out tilted ({t0:.1f}°)", t0 > 3.0, t0)
    check(f"with it, flat ({t1:.1f}°)", t1 < 1.0, t1)
    check("the result records what was evened", even.flash is not None
          and abs(even.flash["spread"] - st["spread"]) < 0.05 and plain.flash is None,
          even.flash and even.flash["spread"])
    off = solve(SolveRequest(paths=paths, lights=lights(), settings=ProcessingSettings(photo_tone=TONE,
        even_flash=True, flash_ref="0,0,3,3")))
    check("a reference that cannot be read leaves the solve as it was",
          off.flash is None and np.allclose(off.normals_full, plain.normals_full))

    print("\nFlat areas: the flash's fall-off across the frame (CAL-004)")
    falloff_dir = tempfile.mkdtemp(prefix="helio_falloff_")
    fpaths, fbox_l, fbox_r = falloff_capture(falloff_dir)
    from helio.core.flash import boxes_text, flat_area_planes
    import cv2 as _cv2
    fstack = np.stack([_cv2.imread(p_, _cv2.IMREAD_GRAYSCALE) for p_ in fpaths]).astype(np.float32)
    pl = flat_area_planes(fstack, [fbox_l, fbox_r], lights(), (W, H))
    check(f"two flat margins measure each flash's fall-off, both ways (up to "
          f"{pl['falloff'] * 100:.0f} % corner to corner)",
          pl is not None and pl["falloff"] > 0.4 and "left-right" in pl["axes"]
          and "top-bottom" in pl["axes"], pl and (pl["falloff"], pl["axes"]))
    flat_obj = (250, 150, 650, 550)
    f_plain = solve(SolveRequest(paths=fpaths, lights=lights(), settings=ProcessingSettings(photo_tone=TONE)))
    f_card = solve(SolveRequest(paths=fpaths, lights=lights(), settings=ProcessingSettings(photo_tone=TONE,
        even_flash=True, flash_ref=box_text(fbox_r))))
    f_flat = solve(SolveRequest(paths=fpaths, lights=lights(), settings=ProcessingSettings(photo_tone=TONE,
        even_flat=True, flat_areas=boxes_text([fbox_l, fbox_r]))))
    t_plain, t_card, t_flat = (tilt(r_.normals_full, flat_obj) for r_ in (f_plain, f_card, f_flat))
    f_obj = solve(SolveRequest(paths=fpaths, lights=lights(), settings=ProcessingSettings(photo_tone=TONE,
        flatten_illumination=True)))
    check("Flat object corrects what the solve reads, and leaves the colour as photographed",
          np.array_equal(f_obj.albedo_rgb_full, f_plain.albedo_rgb_full)
          and not np.allclose(f_obj.normals_full, f_plain.normals_full, atol=1e-3))
    check(f"a flash that falls off across the frame tilts the flat panel ({t_plain:.1f}°)",
          t_plain > 3.0, t_plain)
    check(f"…one grey card cannot see the fall-off ({t_card:.1f}°)", t_card > 2.0, t_card)
    check(f"…two flat areas even it out: flat ({t_flat:.1f}°)", t_flat < 1.0, t_flat)
    check("…and the result records it", f_flat.flat_light is not None
          and abs(f_flat.flat_light["falloff"] - pl["falloff"]) < 0.02 and f_flat.flash is None)
    # The default, Automatic, on camera-encoded (sRGB) frames: decoded once the flat
    # areas even the lighting, as stored when nothing evens it (KD-052).
    spaths, _l, _r = falloff_capture(tempfile.mkdtemp(prefix="helio_falloff_srgb_"), srgb=True)
    s_auto = solve(SolveRequest(paths=spaths, lights=lights(), settings=ProcessingSettings(
        even_flat=True, flat_areas=boxes_text([fbox_l, fbox_r]))))
    s_stored = solve(SolveRequest(paths=spaths, lights=lights(), settings=ProcessingSettings(
        photo_tone="stored", even_flat=True, flat_areas=boxes_text([fbox_l, fbox_r]))))
    t_auto, t_stored = tilt(s_auto.normals_full, flat_obj), tilt(s_stored.normals_full, flat_obj)
    check(f"sRGB frames, Automatic, flat areas: decoded, and flat ({t_auto:.1f}°, as stored "
          f"{t_stored:.1f}°)", s_auto.settings.photo_tone == "srgb" and t_auto < 1.0,
          (t_auto, t_stored))
    s_none = solve(SolveRequest(paths=spaths, lights=lights(), settings=ProcessingSettings()))
    s_bad = solve(SolveRequest(paths=spaths, lights=lights(), settings=ProcessingSettings(
        even_flat=True, flat_areas="0,0,3,3")))
    check("…nothing evened: decoded all the same (from 2026-11-12), also when the areas cannot be read",
          s_none.settings.photo_tone == "srgb" and s_bad.settings.photo_tone == "srgb"
          and np.allclose(s_bad.normals_full, s_none.normals_full))
    from helio.core.flash import DISAGREE_LIMIT, area_disagreement, coverage
    dis = area_disagreement(fstack, [fbox_l, fbox_r], lights())
    check(f"margins on either side read the flashes differently (×{dis:.2f}): the fall-off",
          dis is not None and dis > DISAGREE_LIMIT, dis)
    check("…areas on both sides span the frame; areas on one side say which",
          coverage([fbox_l, fbox_r], (W, H))["side"] == ""
          and coverage([fbox_l, (20, 400, 180, 650)], (W, H))["side"] == "left")
    f_bad = solve(SolveRequest(paths=fpaths, lights=lights(), settings=ProcessingSettings(photo_tone=TONE,
        even_flat=True, flat_areas="0,0,3,3")))
    check("areas that cannot be read leave the solve as it was",
          f_bad.flat_light is None and np.allclose(f_bad.normals_full, f_plain.normals_full))

    print("\nIn the app")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("HELIO_QT_SETTINGS", os.path.join(tempfile.mkdtemp(), "q.ini"))
    os.environ.setdefault("HELIO_PREFS", os.path.join(tempfile.mkdtemp(), "p.json"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    import helio_qt.window as hw
    for nm in ("information", "warning", "critical", "question"):
        setattr(hw.QMessageBox, nm, staticmethod(lambda *a, **k: None))
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    w.open_paths(paths)
    w.project.update_settings(photo_tone=TONE)
    t = time.time()
    while w.grey_ref.get("state") == "looking" and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.02)
    w.wait_background()
    s = w.project.settings
    check("opening the photographs finds the card and turns the correction on",
          w.grey_ref.get("state") == "found" and s.even_flash and bool(s.flash_ref),
          (w.grey_ref.get("state"), s.even_flash, s.flash_ref))
    w.go_workspace("Process")
    pw = w.workspaces["Process"]
    pw.refresh()
    check("Process says so", "found on a grey card" in pw.flash_note.text(), pw.flash_note.text())
    w.project.set_lights(lights(), "file:flash.lp")
    w.registry.trigger("process.run")
    t = time.time()
    while w.project.busy and time.time() - t < 300:
        app.processEvents()
        time.sleep(0.02)
    pw.refresh()
    check("after processing it says by how much the flash varied",
          "Evened" in pw.flash_note.text() and "×2.7" in pw.flash_note.text(), pw.flash_note.text())
    fx = w._plate_facts()
    check("the publication plate's method will say it", abs(fx.get("flash_spread", 0) - 2.71) < 0.1,
          fx.get("flash_spread"))
    from helio.application.plate_doc import Facts
    import inspect
    src = inspect.getsource(__import__("helio.application.plate_doc", fromlist=["x"]))
    check("…in the method's words", "grey card in the shot" in src and "evened against" in src)

    print("\nMarking the grey patch by hand")
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    ins = w.workspaces["Inspect"]
    vp = ins.viewport

    def wait_check():
        t = time.time()
        while w.grey_ref.get("state") == "checking" and time.time() - t < 30:
            app.processEvents()
            time.sleep(0.02)
        w.wait_background()
    pw.btn_grey.click()
    app.processEvents()
    check("Mark the grey patch… shows a photograph in Inspect, the box tool on",
          w.current == "Inspect" and ins.mode == "frame" and vp.tool == "rect",
          (w.current, ins.mode, vp.tool))
    x1, y1, x2, y2 = CARD
    vp.rectDrawn.emit(x1 + 15, y1 + 20, x2 - 15, y2 - 20)
    app.processEvents()
    check("after the box: back in Process, the tool off", w.current == "Process" and vp.tool == "",
          (w.current, vp.tool))
    wait_check()
    pw.refresh()
    s = w.project.settings
    check("the patch is checked in every frame and used, with the flash's spread",
          w.grey_ref.get("state") == "marked" and s.even_flash
          and s.flash_ref == box_text((x1 + 15, y1 + 20, x2 - 15, y2 - 20))
          and abs(w.grey_ref.get("spread", 0) - 2.71) < 0.1, (w.grey_ref, s.flash_ref))
    check("Process says it was marked by hand, and by how much the flash varies",
          "marked by hand" in pw.flash_note.text() and "×2.7" in pw.flash_note.text(),
          pw.flash_note.text())
    good = s.flash_ref

    pw.btn_grey.click()
    app.processEvents()
    vp.rectDrawn.emit(5, 5, 35, 35)                      # the black surround
    app.processEvents()
    wait_check()
    pw.refresh()
    check("a black patch is refused, says why, and the earlier patch stands",
          w.project.settings.flash_ref == good and w.grey_ref.get("state") == "marked"
          and "black" in w._flash[0], (w.project.settings.flash_ref, w._flash[0]))

    pw.btn_grey.click()
    app.processEvents()
    QTest.keyClick(vp, Qt.Key.Key_Escape)
    app.processEvents()
    check("Esc gives it up: back in Process, the tool off", w.current == "Process"
          and vp.tool == "", (w.current, vp.tool))
    vp.set_tool("rect")
    vp.rectDrawn.emit(x1 + 30, y1 + 30, x2 - 30, y2 - 30)  # a later box, for something else
    app.processEvents()
    w.wait_background()
    check("…and a later box is not taken for the grey patch", w.project.settings.flash_ref == good,
          w.project.settings.flash_ref)
    vp.set_tool("")

    pw.btn_grey.click()
    app.processEvents()
    w.go_workspace("Measure")
    vp.rectDrawn.emit(x1 + 30, y1 + 30, x2 - 30, y2 - 30)
    app.processEvents()
    w.wait_background()
    check("leaving Inspect gives it up too", vp.tool == "" and w.project.settings.flash_ref == good)

    from helio.core.flash import check_reference
    r = check_reference(paths, (x1 + 10, y1 + 10, x2 - 10, y2 - 10))
    check("check_reference: the card is fine, with no lights no spread",
          r["ok"] and not r["warning"] and "spread" not in r, r)
    r = check_reference(paths, (60, 60, 600, 600))
    check("…the embossed object is usable but flagged as not even", r["ok"] and r["warning"], r)

    print("\nMarking flat areas in the app")
    w.open_paths(fpaths)
    w.project.update_settings(photo_tone=TONE)
    w.wait_background()
    app.processEvents()
    w.go_workspace("Process")
    pw.refresh()
    from types import SimpleNamespace as NS
    from helio_qt.process_ws import strength_key
    old = lambda **k: NS(**{**dict(even_flat=False, flat_areas="", even_flash=False, flash_ref="",
                                   semi_calibrated=False), **k})
    check("Lamp strengths shows what the solve uses, for sessions with several boxes ticked",
          [strength_key(old(even_flat=True, flat_areas="1,1,9,9", even_flash=True, flash_ref="1,1,5,5")),
           strength_key(old(even_flat=True, even_flash=True, flash_ref="1,1,5,5")),
           strength_key(old(even_flash=True, semi_calibrated=True)),
           strength_key(old(semi_calibrated=True)), strength_key(old())]
          == ["flat", "grey", "grey", "photos", "off"])
    check("no flat areas yet: Process says so, Lamp strengths not on them",
          "No flat areas" in pw.flat_area_note.text() and pw.strength.current() != "flat")
    for k_, bx in enumerate((fbox_l, fbox_r)):
        pw.btn_flat_area.click()
        app.processEvents()
        vp.rectDrawn.emit(*bx)
        app.processEvents()
        w.wait_background()
        app.processEvents()
        if k_ == 0:
            pw.refresh()
            check("one area, on the left: a warning that the right is not measured",
                  "all on the left" in pw.flat_area_note.text()
                  and pw.flat_area_note.property("status") == "warn", pw.flat_area_note.text())
    s = w.project.settings
    pw.refresh()
    check("two areas marked: kept, turned on, said so, back in Process",
          s.even_flat and len(s.flat_areas.split(";")) == 2 and w.current == "Process"
          and "2 flat areas marked" in pw.flat_area_note.text(), (s.flat_areas, pw.flat_area_note.text()))
    w.project.set_lights(lights(), "file:falloff.lp")
    pw.refresh()
    check("Light and flash says what is on", "flat areas on" in pw.light_sec.badge.text(),
          pw.light_sec.badge.text())

    print("\nCompare corrections and a quick look")
    import helio_qt.compare_dialog as cdlg
    shown = []
    real = cdlg.CompareDialog.show
    cdlg.CompareDialog.show = lambda self_: shown.append(self_)
    pw.btn_compare.click()
    w.wait_background()
    app.processEvents()
    cdlg.CompareDialog.show = real
    out = getattr(pw, "last_compare", None)
    rows = {r_["name"]: r_ for r_ in (out or {}).get("rows", [])}
    check("Compare solves each correction small and opens its table",
          shown and set(rows) >= {"None", "Flat areas", "Flat object"}, list(rows))
    check(f"…the flat areas lean least with flat areas on ({rows['Flat areas']['worst']:.1f}°, "
          f"none {rows['None']['worst']:.1f}°)",
          rows["Flat areas"]["worst"] < 1.0 < rows["None"]["worst"])
    check("…and it says so beside the button", "Flattest: flat areas" in pw.compare_note.text(),
          pw.compare_note.text())
    from helio_qt.compare_dialog import best_row
    fake = [dict(name="None", worst=12.1, mean=12.1, whole=9.5, applied=True),
            dict(name="Near-field", worst=8.2, mean=8.2, whole=29.4, applied=True),
            dict(name="Flat object", worst=12.8, mean=12.8, whole=8.9, applied=True)]
    check("a correction that flattens the card but tilts the whole picture is not 'flattest'",
          best_row(fake)["name"] == "None")
    shown[0].use_buttons["Flat object"].click()
    app.processEvents()
    s = w.project.settings
    check("Use this turns that correction on alone, keeping the marked areas",
          s.flatten_illumination and not s.even_flat and s.flat_areas)
    shown[0].close()
    w.project.update_settings(even_flat=True, flatten_illumination=False)
    pw.btn_quick.click()
    w.wait_background()
    app.processEvents()
    check("Quick look shows a small solve, said to be a preview, and keeps nothing",
          pw.quick_head.isVisible() and "Quick look" in pw.quick_head.text()
          and pw.previews.isVisible() and w.project.result is None,
          (pw.quick_head.text(), w.project.result))
    check("…with an overall score under the pictures, and the look kept in the list",
          pw.score_big.isVisible() and pw.score_big.text().split()[0].isdigit()
          and len(pw.looks) == 1 and pw.looks_head.isVisible(), (pw.score_big.text(), pw.looks))
    first = pw.looks[0]
    w.project.update_settings(even_flat=False)
    pw.btn_quick.click()
    w.wait_background()
    app.processEvents()
    check("a second quick look with flat areas off scores lower (the margins lean), both listed",
          len(pw.looks) == 2 and pw.looks[1]["score"] < first["score"]
          and pw.looks[1]["summary"] == "no corrections" and "flat areas" in first["summary"],
          [(lk["score"], lk["summary"]) for lk in pw.looks])
    pw.use_look(first["settings"])
    check("Use these puts the best look's settings back", w.project.settings.even_flat)
    before_n = len(pw.looks)
    pw.btn_try.click()
    w.wait_background()
    app.processEvents()
    tried = pw.looks[before_n:]
    names = [lk["summary"] for lk in tried]
    best_tried = max(lk["score"] for lk in tried)
    first_n = tried[0]["n"]
    pw.show_look(first_n)
    check("Show puts an earlier quick look's pictures and score back at the top",
          pw._quick["result"] is tried[0]["result"] and pw.shown_look == first_n
          and pw.score_big.text().split()[0] == str(tried[0]["score"]))
    keep_looks = [dict(lk) for lk in pw.looks]
    for lk in pw.looks:
        lk["score"], lk["word"] = 77, "Good"
    pw._draw_looks()
    texts = [pw.looks_box.itemAt(i).widget().text() for i in range(pw.looks_box.count())
             if pw.looks_box.itemAt(i).widget() is not None]
    rows = [pw.looks_box.itemAt(i).layout() for i in range(pw.looks_box.count())
            if pw.looks_box.itemAt(i).layout() is not None]
    row_text = " ".join(rows[k].itemAt(j).widget().text() for k in range(len(rows))
                        for j in range(rows[k].count())
                        if hasattr(rows[k].itemAt(j).widget(), "text"))
    check("all scoring the same: no '✓ best' on every row, and a line saying why",
          texts and "score the same" in texts[0] and "✓" not in row_text, (texts, row_text[:200]))
    pw.looks = keep_looks
    pw._draw_looks()
    check("Try a few runs a quick look for each change worth trying, scored, the best shown",
          len(tried) >= 5 and "flat areas" in names and "no corrections" in names
          and "shine-resistant solver" in " ".join(names)
          and pw.quick_head.isVisible() and pw.score_big.text().split()[0] == str(best_tried)
          and "Tried" in pw.quick_note.text(), (names, pw.score_big.text(), pw.quick_note.text()))
    from helio.application.process_tools import overall_score
    check("the score: sure, well-lit and flat is 100; doubtful, bunched and tilted is weak",
          overall_score(0.8, 2.5, 1.5, [("g", 0.5)])["score"] == 100
          and overall_score(6, 20, 7, [("g", 12)])["word"] == "Weak"
          and [n for n, _p, _o in overall_score(3, 9, 2)["parts"]] == ["Sureness", "Lights"])
    w.registry.trigger("process.run")
    t = time.time()
    while w.project.busy and time.time() - t < 300:
        app.processEvents()
        time.sleep(0.02)
    pw.refresh()
    check("the full solve replaces the quick look", not pw.quick_head.isVisible())
    check("after processing: evened, and by how much it fell off",
          "Evened from 2 flat areas" in pw.flat_area_note.text(), pw.flat_area_note.text())
    check(f"…the flat panel solved flat in the app ({tilt(w.project.result.normals_full, flat_obj):.1f}°)",
          tilt(w.project.result.normals_full, flat_obj) < 1.0)
    pw.btn_flat_clear.click()
    app.processEvents()
    pw.refresh()
    check("Clear forgets the boxes, and Lamp strengths stays on Flat areas for new ones",
          not w.project.settings.flat_areas and pw.strength.current() == "flat")

    print("\nWithout a card")
    blank = tempfile.mkdtemp(prefix="helio_flash_none_")
    nocard = []
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        x1, y1, x2, y2 = CARD
        im[y1:y2, x1:x2] = 0
        q = os.path.join(blank, f"g{i}.png")
        cv2.imwrite(q, im)
        nocard.append(q)
    w.open_paths(nocard)
    w.project.update_settings(photo_tone=TONE)
    t = time.time()
    while w.grey_ref.get("state") == "looking" and time.time() - t < 30:
        app.processEvents()
        time.sleep(0.02)
    w.wait_background()
    w.go_workspace("Process")
    pw.refresh()
    check("no card: Process suggests placing one, and the option stays off",
          "No grey card" in pw.flash_note.text()
          and "place a flat, matte grey card" in pw.flash_note.toolTip()
          and pw.strength.current() != "grey", pw.flash_note.text())
    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
