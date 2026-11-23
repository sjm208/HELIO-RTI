"""The chrome-ball search on real bindings (CAL-002), when the drive is connected.

/Volumes/T7 1/BINDINGS PS holds six folders of eight 98 MP JPEGs: a gilt
binding on black, a grey card at the right, and the ball set in a hole in
it, near (7935, 7950).  Until 2026-10-15 the search took about 23 s and
returned a patch of gilding in the middle of the book.  Read only; skipped
when the drive is not there (HELIO_BINDINGS names another place).
"""
import glob
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DRIVE = os.environ.get("HELIO_BINDINGS") or next(
    iter(sorted(glob.glob("/Volumes/*/BINDINGS PS"))), "/Volumes/T7/BINDINGS PS")  # drives rename
BALL = (7935.0, 7950.0)          # where the ball sits in every book of this set
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def folders():
    out = []
    for d in sorted(glob.glob(os.path.join(DRIVE, "*", ""))):
        fs = sorted(f for f in glob.glob(os.path.join(d, "*.jpg"))
                    if not os.path.basename(f).startswith("._"))
        if len(fs) >= 3:
            out.append((os.path.basename(os.path.dirname(d)), fs))
    return out


def main():
    import math
    import numpy as np
    print("=" * 60)
    print("The chrome ball on real bindings")
    print("=" * 60)
    books = folders() if os.path.isdir(DRIVE) else []
    if not books:
        print(f"  (no captures at {DRIVE}; skipped)")
        print(f"Results: 0 passed, 0 failed out of 0 tests")
        return 0
    from helio.core.processing import detect_sphere_candidates
    from helio.core.sphere import Sphere, calibrate, image_size, with_second_ball
    for name, fs in books:
        print(f"\n{name} ({len(fs)} frames)")
        w, h = image_size(fs[0])
        t0 = time.time()
        c = detect_sphere_candidates(fs, h, w, want_two=True)
        dt = time.time() - t0
        notes = list(detect_sphere_candidates.notes)
        check(f"the search takes under 8 s on {w} × {h} frames ({dt:.1f} s; was about 18)",
              dt < 8.0, dt)
        check("the ball is found on its card, not in the gilding",
              bool(c) and math.hypot(c[0][0] - BALL[0], c[0][1] - BALL[1]) < 120, c[:1])
        if not c:
            continue
        r = c[0][2]
        check("its rim is the limb's edge (255-275 px) or it says to check the circle",
              255 <= r <= 275 or "could not be seen" in notes[0], (round(r, 1), notes[0][:70]))
        cal = calibrate(fs, Sphere(*c[0], note=notes[0]))
        el = np.degrees(np.arcsin(np.clip(np.asarray(cal.lights)[:, 2], -1, 1)))
        check("a light for every frame, all 10-75° up", len(el) == len(fs)
              and bool(np.all((el > 10) & (el < 75))), np.round(el, 1).tolist())
        if len(c) > 1:
            check("a second circle that is not a ball is left out",
                  with_second_ball(fs, cal, Sphere(*c[1])) is None, c[1])
        # The hand-held flash, evened by the QPcard on the right (CAL-003).
        from helio.core.flash import find_grey_reference, strengths
        from helio.core.sphere import read_many
        ref = find_grey_reference(fs)
        check("the grey card is found on the right of the frame",
              ref is not None and (ref["box"][0] + ref["box"][2]) / 2 > 0.85 * w, ref)
        if ref is not None:
            raw = np.stack([g.astype(np.float32) for g in read_many(fs, 8)])
            st = strengths(raw, [v // 8 for v in ref["box"]], cal.lights)

            def card_tilt(stack):
                k, hh, ww = stack.shape
                G = np.linalg.lstsq(np.asarray(cal.lights, np.float64), stack.reshape(k, -1),
                                    rcond=None)[0].T.reshape(hh, ww, 3)
                x1, y1, x2, y2 = [v // 8 for v in ref["box"]]
                m = np.median(G[y1:y2, x1:x2].reshape(-1, 3), axis=0)
                return math.degrees(math.acos(min(1.0, abs(m[2]) / np.linalg.norm(m))))
            before, after = card_tilt(raw), card_tilt(raw * st["factor"][:, None, None])
            check(f"evening the flash (it varied ×{st['spread']:.1f}) lays the card flat: "
                  f"{before:.1f}° → {after:.1f}°", after < 2.0 and after < before, (before, after))
            # Flat areas (CAL-004): the black cloth round the book is refused (it
            # made the cover worse: 4.2° against the card's 1.7°); the card is not.
            from helio.core.flash import FLAT_MIN_LEVEL, check_reference
            cloth = check_reference(fs, (80, 80, 880, 11300), min_level=FLAT_MIN_LEVEL)
            card_ok = check_reference(fs, ref["box"], min_level=FLAT_MIN_LEVEL)
            check("a flat area on the black cloth is refused as too dark; the card is accepted",
                  not cloth["ok"] and "too dark" in cloth["problem"] and card_ok["ok"],
                  (cloth.get("problem"), card_ok.get("problem")))

    print("\nThrough the window")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("HELIO_QT_SETTINGS", os.path.join(tempfile.mkdtemp(), "q.ini"))
    os.environ.setdefault("HELIO_PREFS", os.path.join(tempfile.mkdtemp(), "p.json"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    import helio_qt.window as hw
    for nm in ("information", "warning", "critical", "question"):
        setattr(hw.QMessageBox, nm, staticmethod(lambda *a, **k: None))
    win = hw.MainWindow(appearance="dark")
    win.resize(1440, 900)
    win.show()
    flagged = [fs for _n, fs in books
               if (lambda cc: cc and "could not be seen" in detect_sphere_candidates.notes[0])(
                   detect_sphere_candidates(fs, *image_size(fs[0])[::-1]))]
    fs = flagged[0] if flagged else books[0][1]
    win.open_paths(fs)
    t0 = time.time()
    while len(win.thumbs.cache) < len(fs) and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    win.registry.trigger("capture.sphere")
    t0 = time.time()
    while win.project.busy and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    win.go_workspace("Capture")
    app.processEvents()
    cap = win.workspaces["Capture"]
    check("Find Chrome Sphere sets a light for every frame",
          win.project.lights is not None and len(win.project.lights) == len(fs))
    if flagged:
        check("…and when the rim was faint, Capture says to check the circle",
              "Check the circle" in cap.sphere_lines.text(), cap.sphere_lines.text())
    win.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
