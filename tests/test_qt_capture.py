"""Capture in the new interface: the chrome sphere shown and adjustable (CAL-030…CAL-032).

Offscreen, on the real 8-frame test capture.  Finds the ball and checks
it is outlined at its own radius with every frame's highlight marked; that
moving, nudging, refitting and a second ball each re-read the lights; that
a frame can be left out and the solve then uses the rest; that robust mode
and perspective correction are wired; and that the lights' plot and figures
(conditioning, dome coverage) are shown.

    python tests/test_qt_capture.py
"""
import math
import os
import sys
import tempfile
import time

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


def main():
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    print("=" * 60)
    print("Capture: the chrome sphere")
    print("=" * 60)
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    cap = w.workspaces["Capture"]
    p = w.project

    def idle(limit=120):
        t0 = time.time()
        while p.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    w.open_paths([test_images(PASS, FAIL)])
    check("before searching: the three ways to set the lights", cap.cards_box.isVisibleTo(w))
    w.registry.trigger("capture.sphere")
    idle()
    cal = p.calibration
    print("\nThe ball")
    check("found, and the lights are ready", cal is not None and p.lights_ready()
          and p.light_source == "sphere")
    check(f"outlined at its own radius ({cal.sphere.r:.2f} px; the old finder said 31.7)",
          abs(cal.sphere.r - 24.6) < 1.5)
    check("the cards fold into one line once the lights have a source",
          not cap.cards_box.isVisibleTo(w) and cap.method_row.isVisibleTo(w)
          and cap.method.current() == "sphere")
    check("the ball is shown, with every frame's highlight marked",
          cap.work.currentIndex() == 1 and len(cap.view.markers) == len(p.images))
    check("…and the centre and radius in the inspector",
          abs(cap.fields["r"].value() - cal.sphere.r) < 0.006)        # shown to 0.01 px

    print("\nAdjusting the outline re-reads the lights")
    s = cal.sphere
    before = p.lights.copy()
    cap._circle_moved(0, s.cx, s.cy, s.r * 1.3, True)
    cap.wait()
    bigger = p.lights.copy()
    el = lambda L: [math.degrees(math.asin(v[2])) for v in L]   # noqa: E731
    check("a radius 30% too large makes every light look higher (the KD-020 bias)",
          np.mean(el(bigger)) > np.mean(el(before)) + 4, (np.mean(el(before)), np.mean(el(bigger))))
    cap.refit()
    cap.wait()
    check(f"Fit to the rim brings it back ({p.calibration.sphere.r:.2f} px, "
          f"{p.calibration.sphere.rms:.2f} px RMS)",
          abs(p.calibration.sphere.r - 24.6) < 1.5 and p.calibration.sphere.rms < 1.0
          and np.allclose(p.lights, before, atol=0.02))
    cap.view.setFocus()
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtCore import QEvent
    x0 = p.calibration.sphere.cx
    cap.view.step = 0.25
    QApplication.sendEvent(cap.view, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Right,
                                                Qt.KeyboardModifier.NoModifier))
    cap.wait()
    check("an arrow key nudges by the step (a quarter pixel)",
          abs(p.calibration.sphere.cx - x0 - 0.25) < 1e-6)

    print("\nFrames")
    check("one row per frame, each with its light", cap.frame_list.count() == len(p.images)
          and "° up" in cap.frame_list.item(0).text())
    cap.frame_list.item(2).setCheckState(Qt.CheckState.Unchecked)
    app.processEvents()
    check("unticking a frame leaves it out of the solve",
          p.excluded == {2} and len(p.solve_images()) == len(p.images) - 1
          and len(p.solve_lights()) == len(p.images) - 1 and p.lights_ready())
    req = w.solve_request()
    check("…the solve request has the other frames, each with its own light",
          req.paths == [q for i, q in enumerate(p.images) if i != 2]
          and np.array_equal(req.lights, p.lights[[0, 1, 3, 4, 5, 6, 7]]))
    cap.frame_list.item(2).setCheckState(Qt.CheckState.Checked)
    app.processEvents()

    print("\nThe lights' figures")
    t = cap.light_stats.text()
    check("conditioning and dome coverage are shown", "Conditioning" in t and "Dome coverage" in t, t)
    check("the dome plots the lights in use", len(cap.dome.captured) == len(p.images))

    print("\nOptions")
    cap.chk_robust.setChecked(True)
    cap.wait()
    check("robust mode re-reads the highlights", p.lights_ready())
    cap.chk_robust.setChecked(False)
    cap.wait()
    cap.chk_dual.setChecked(True)
    cap.wait()
    s2 = p.calibration.sphere2
    check(f"a second ball is found where the finder saw it ({s2.cx:.0f}, {s2.cy:.0f}, r {s2.r:.0f})",
          s2 is not None and math.hypot(s2.cx - 723, s2.cy - 71) < 6)
    dual = p.lights.copy()
    cap.chk_dual.setChecked(False)
    cap.wait()
    check("…and each light becomes the mean of both balls'",
          p.calibration.sphere2 is None and not np.allclose(dual, p.lights))
    plain = p.lights.copy()
    cap.chk_persp.setChecked(True)
    # This capture: a 120 mm lens on a 53.4 x 40 mm back, about 78 mm full-frame
    # equivalent, with the ball far into a corner.
    cap.focal.set_value(78)
    cap.focal.changed.emit("focal35", 78.0, True)
    cap.wait()
    turn = float(np.degrees(np.arccos(np.clip((plain * p.lights).sum(1), -1, 1))).max())
    check(f"perspective correction for the ball's corner position turns the lights ({turn:.1f}°)",
          0.5 < turn < 30)
    cap.chk_persp.setChecked(False)
    cap.wait()
    check("and switched off gives HELIO 2's lights again", np.allclose(p.lights, plain, atol=1e-6))

    print("\nThe session carries complete lights")
    out = os.path.join(tempfile.mkdtemp(), "c.helio")
    w.write_session(out)
    from helio.io import session as S
    check("every frame's light is saved", S.read(out).lights() is not None
          and np.allclose(S.read(out).lights(), p.lights, atol=1e-6))

    print("\nCapture details and points of interest")
    from helio_qt.dialogs import CaptureDetailsDialog
    d = CaptureDetailsDialog(w)
    d.reset()
    d.edits["operator"].setText("A. Tester")
    d.edits["lens"].setText("Schneider 120mm Macro")
    d.save()
    check("the details are kept with the project", p.capture_meta.get("operator") == "A. Tester"
          and p.capture_meta.get("img_count") == str(len(p.images)))
    poi_csv = os.path.join(tempfile.mkdtemp(), "poi.csv")
    with open(poi_csv, "w") as f:
        f.write("name,X_mm,Y_mm,Z_mm\nOrigin,0,0,0\nCamera,0,0,620\nLED1,300,0,200\n")
    hw.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (poi_csv, ""))
    d2 = CaptureDetailsDialog(w)
    d2.load_poi()
    check("points of interest load from a CSV", p.poi.get("Camera") == (0.0, 0.0, 620.0))
    out2 = os.path.join(tempfile.mkdtemp(), "poi_out.csv")
    hw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out2, ""))
    import helio_qt.dialogs as dg
    dg.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (out2, ""))
    d2.export_poi()
    check("…and export as the classic file", open(out2).read().startswith("name,X_mm,Y_mm,Z_mm,notes")
          and "camera position" in open(out2).read())
    sess2 = os.path.join(tempfile.mkdtemp(), "meta.helio")
    w.write_session(sess2)
    check("the session carries the details", S.read(sess2).capture_metadata.get("lens")
          == "Schneider 120mm Macro")

    print("\nDrift, alignment and calibration files")
    w.wait_background()
    check("the drift check ran at import, with a verdict in pixels",
          p.drift is not None and p.drift["verdict"] in ("clean", "marginal", "moved")
          and p.drift["n"] == len(p.images))
    import helio_qt.report as rp
    shown = []
    rp.ReportDialog.exec = lambda self: shown.append(self) or 0
    w.registry.trigger("capture.align")
    w.wait_background()
    check("Line Up Shifted Frames reports each frame's shift",
          shown and len(shown[-1].figures) == len(p.images) and "px" in shown[-1].figures[0][1])
    lp_out = os.path.join(tempfile.mkdtemp(), "cal.lp")
    hw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (lp_out, "RTIBuilder light file (*.lp)"))
    w.registry.trigger("capture.save_cal")
    hj_out = os.path.join(tempfile.mkdtemp(), "cal.hjson")
    hw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (hj_out, "HELIO calibration (*.hjson)"))
    w.registry.trigger("capture.save_cal")
    from helio.application.capture_tools import read_calibration
    La, _s, names = read_calibration(lp_out)
    Lb, _s2, _n2 = read_calibration(hj_out)
    used = p.solve_lights()
    check("the calibration saves as .lp and .hjson and reads back the same lights",
          np.allclose(La, used, atol=1e-5) and np.allclose(Lb, used, atol=1e-5)
          and names[0] == os.path.basename(p.solve_images()[0]))
    from helio.application.capture_tools import calibration_spheres
    ball = w._ball_tuples()[0]
    near = lambda a_, b_: a_ is not None and b_ is not None and all(
        abs(float(x) - float(y)) < 1e-3 for x, y in zip(a_, b_))
    check("where the ball was is saved: in the .hjson, and in a .hjson beside the .lp",
          ball is not None and near(calibration_spheres(hj_out)[0], ball)
          and near(calibration_spheres(lp_out)[0], ball)
          and os.path.exists(os.path.splitext(lp_out)[0] + ".hjson"), (ball, calibration_spheres(hj_out)))
    hw.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (hj_out, ""))
    w.registry.trigger("capture.load_cal")
    check("Load Calibration takes it back as the lights", p.lights_ready()
          and p.light_source == "file:cal.hjson")
    check("…with the ball's place, so near-field measures from it",
          ball is not None and near(w.light_origin(), ball[:2]), (w.light_origin(), ball))
    w.load_light_file(lp_out)
    check("a .lp with its Helio calibration beside it brings the ball's place too",
          ball is not None and near(w.light_origin(), ball[:2]), w.light_origin())
    w.wait_background()
    check("lights from a file are checked for north at the top: these read upright",
          (p.light_frame or {}).get("verdict") == "upright" and not cap.frame_warn.isVisibleTo(w),
          p.light_frame)
    # A light file written upside down, as HELIO's dome calibrator did before 2026-10 (KD-027)
    from helio.application.capture_tools import swap_north_south
    from helio.io.sidecars import write_lp
    lp_bad = os.path.join(tempfile.mkdtemp(), "upside_down.lp")
    good = np.array(p.lights, np.float32)
    write_lp(lp_bad, swap_north_south(good), p.images)
    w.load_light_file(lp_bad)
    w.wait_background()
    app.processEvents()
    check("an upside-down light file is caught, and Capture says so with a way to fix it (KD-054)",
          (p.light_frame or {}).get("verdict") == "upside down" and cap.frame_warn.isVisibleTo(w)
          and "upside down" in cap.frame_warn_text.text(), p.light_frame)
    w.swap_lights_north_south()
    w.wait_background()
    app.processEvents()
    check("…Swap north and south puts the lights back and the notice goes",
          np.allclose(p.lights, good, atol=1e-6) and not cap.frame_warn.isVisibleTo(w)
          and (p.light_frame or {}).get("verdict") == "upright")

    print("\nA search that finds no ball keeps the lights you have")
    L_had, src_had = p.lights.copy(), p.light_source
    w._sphere_done(None, [], "")
    app.processEvents()
    check("no ball found: lights from a file are kept, not set to none",
          p.lights is not None and np.allclose(p.lights, L_had) and p.light_source == src_had,
          p.light_source)
    w._sphere_done(None, [], "the photographs could not be read")
    app.processEvents()
    check("…and when the search fails outright", p.lights is not None
          and p.light_source == src_had)

    print("\nRig profiles")
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    p.update_settings(near_field=True, dome_radius_mm=450.0)
    p.scale_mm_per_px = 0.05
    L_saved, ball_saved = p.lights.copy(), w._ball_tuples()[0]
    check("Save as rig keeps the lights, the ball's place, near-field and the scale",
          w.save_rig_profile("Test rig") == "Test rig"
          and w.rig_profiles()["Test rig"]["n"] == len(p.images)
          and w.rig_profiles()["Test rig"]["dome_radius_mm"] == 450.0)
    p.set_lights(None, "")
    p.update_settings(near_field=False, dome_radius_mm=200.0)
    p.scale_mm_per_px = None
    cap.refresh()
    check("a capture with as many frames: Capture offers the rig",
          cap.rig_combo.count() == 1 and cap.btn_rig_use.isEnabled()
          and "fits these" in cap.rig_note.text(), cap.rig_note.text())
    cap.btn_rig_use.click()
    app.processEvents()
    check("Use sets the lights, the ball's place, near-field and the scale in one go",
          p.lights_ready() and np.allclose(p.lights, L_saved, atol=1e-5)
          and p.light_source == "rig:Test rig" and p.settings.near_field
          and p.settings.dome_radius_mm == 450.0 and p.scale_mm_per_px == 0.05
          and (ball_saved is None or w.light_origin() is not None))
    allp = w.rig_profiles()
    allp["Other"] = dict(allp["Test rig"], n=5)
    from helio.application import rigs as _rigs
    w.prefs.setValue(_rigs.KEY, _rigs.dump(allp))
    check("a rig with a different number of lights is refused",
          not w.apply_rig_profile("Other") and "Other" not in _rigs.fitting(allp, len(p.images)))
    w.delete_rig_profile("Other")
    w.delete_rig_profile("Test rig")
    p.update_settings(near_field=False)
    p.scale_mm_per_px = None

    print("\nThe backlit frame (TRN-001)")
    import cv2
    w.registry.trigger("process.run")
    t0 = time.time()
    while p.busy and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.01)
    bl = os.path.join(tempfile.mkdtemp(), "backlit.png")
    img = cv2.imread(p.images[0])
    cv2.imwrite(bl, cv2.flip(img, 1))                    # shot from below: mirrored
    hw.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (bl, ""))
    w.registry.trigger("capture.backlit")
    w.wait_background()
    b = w.backlit
    check(f"it is lined up with the surface, mirror found ({b.get('inliers')} features)",
          b.get("H") is not None and b.get("flip") == 1)
    w.set_view("Backlit frame")
    ins = w.workspaces["Inspect"]
    ins.wait_render()
    check("Inspect shows it as a view", not ins.items["Backlit frame"].isHidden()
          and not ins.error and not ins.viewport.item.pixmap().isNull())

    print("\nA ball too dark to find: marked by hand")
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    from PySide6.QtCore import QPoint as _QP
    from PySide6.QtCore import Qt as _Qt
    from PySide6.QtTest import QTest as _QT
    from helio.core.sphere import calibrate
    w.open_paths([test_images(PASS, FAIL)])
    w.wait_background()
    w.registry.trigger("capture.sphere")
    idle()
    true_s, true_L = p.calibration.sphere, np.array(p.lights)
    dark = tempfile.mkdtemp(prefix="helio_dark_ball_")
    for i, q in enumerate(p.images):
        im = cv2.imread(q).astype(np.float32) * 0.15        # a dark shot: the ball nearly black
        cv2.imwrite(os.path.join(dark, f"d{i}.png"), np.clip(im, 0, 255).astype(np.uint8))
    w.open_paths([dark])
    w.wait_background()
    app.processEvents()
    check("Capture › Mark the Chrome Ball by Hand… is there, and the Inspector offers it",
          "capture.mark_ball" in w.registry.actions and cap.btn_mark.isVisibleTo(w)
          and cap.btn_mark.isEnabled())
    w.registry.trigger("capture.mark_ball")
    w.wait_background()
    app.processEvents()
    check("marking shows every frame's brightest, stretched, at the photographs' size",
          cap.marking and cap.view.marking and not cap.view.item.pixmap().isNull()
          and cap.work.currentIndex() == 1 and "Marking" in cap.sphere_lines.text())
    vp = cap.view
    vp.zoom_to(true_s.cx, true_s.cy, true_s.r, margin=3)
    app.processEvents()
    c0 = vp.mapFromScene(true_s.cx + 1.5, true_s.cy - 1.0)        # a hand's centre, a little off
    e0 = vp.mapFromScene(true_s.cx + 1.5 + true_s.r * 1.06, true_s.cy - 1.0)
    _QT.mousePress(vp.viewport(), _Qt.MouseButton.LeftButton, _Qt.KeyboardModifier.NoModifier, c0)
    _QT.mouseMove(vp.viewport(), e0)
    from PySide6.QtCore import QEvent as _QE, QPointF as _QPF
    from PySide6.QtGui import QMouseEvent as _QME
    QApplication.sendEvent(vp.viewport(), _QME(_QE.Type.MouseMove, _QPF(e0), _QPF(e0),
                                               _Qt.MouseButton.NoButton, _Qt.MouseButton.LeftButton,
                                               _Qt.KeyboardModifier.NoModifier))
    _QT.mouseRelease(vp.viewport(), _Qt.MouseButton.LeftButton, _Qt.KeyboardModifier.NoModifier, e0)
    w.wait_background()
    cap.wait()
    app.processEvents()
    s = p.calibration.sphere if p.calibration is not None else None
    check("a plain drag from centre to edge outlines it, and marking ends",
          s is not None and not cap.marking and p.light_source == "sphere", s)
    off = math.hypot(s.cx - true_s.cx, s.cy - true_s.cy) if s else 99
    check(f"…close to the ball the search found on the normal photos "
          f"(centre {off:.2f} px off, radius {s.r if s else 0:.2f} for {true_s.r:.2f})",
          s is not None and off < 1.5 and abs(s.r - true_s.r) < 1.5)
    # The same photographs read with the ball where the search put it on the
    # normal ones: the hand outline should give the lights a found ball would.
    ref = np.array(calibrate(p.images, true_s).lights)
    L = np.array(p.lights) if p.lights is not None else None
    ang = (np.degrees(np.arccos(np.clip(np.sum(L * ref, axis=1), -1, 1)))
           if L is not None and L.shape == ref.shape else None)
    worst = float(np.nanmax(ang)) if ang is not None else float("nan")
    check(f"…so every frame's light is the one a found ball gives (worst {worst:.2f}°)",
          ang is not None and worst < 1.0, ang)
    w.registry.trigger("capture.mark_ball")
    w.wait_background()
    _QT.keyClick(vp, _Qt.Key.Key_Escape)
    app.processEvents()
    check("Esc cancels marking and keeps the outline already set",
          not cap.marking and p.calibration is not None and p.calibration.sphere == s)
    from helio.core.sphere import hand_sphere
    hs = hand_sphere(p.images, 40.0, 40.0, 20.0)
    check("a circle drawn where there is no ball is kept exactly as drawn",
          (hs.cx, hs.cy, hs.r) == (40.0, 40.0, 20.0))

    print("\nFrames left out, and getting them back")
    # A detection that fails on a frame leaves it out. Re-reading the lights used to
    # keep that exclusion for ever, by frame number, so a frame whose light was found
    # perfectly well the second time stayed ticked off with no reason shown.
    good = p.lights.copy()
    broken = good.copy()
    broken[2] = np.nan
    p.set_lights(broken, "sphere", sphere=p.sphere, calibration=p.calibration)
    app.processEvents()
    check("a frame with no light is left out, and said to be Helio's doing",
          p.excluded == {2} and p.excluded_auto == {2})
    check("…and the Frames warning says so", cap.left_out.isVisibleTo(w)
          and "No light on frame 3" in cap.left_out.text(), cap.left_out.text())
    p.set_lights(good, "sphere", sphere=p.sphere, calibration=p.calibration)
    app.processEvents()
    check("finding that light again puts the frame back in the solve",
          p.excluded == set() and p.excluded_auto == set()
          and len(p.solve_images()) == len(p.images))
    check("…and the warning goes with it", not cap.left_out.isVisibleTo(w))

    cap.frame_list.item(4).setCheckState(Qt.CheckState.Unchecked)
    app.processEvents()
    check("a frame left out by hand is not Helio's to undo",
          p.excluded == {4} and p.excluded_auto == set())
    check("…and the warning separates it from a missing light",
          cap.left_out.isVisibleTo(w) and "Frame 5 left out though the light was found"
          in cap.left_out.text(), cap.left_out.text())
    p.set_lights(good, "sphere", sphere=p.sphere, calibration=p.calibration)
    app.processEvents()
    check("…so re-reading the lights keeps it out", p.excluded == {4})
    proc = w.workspaces["Process"]
    w.go_workspace("Process")
    app.processEvents()
    check("Process warns before solving that a frame is missing",
          proc.left_out.isVisibleTo(w) and "without 1 of 8 frames (5)" in proc.left_out.text(),
          proc.left_out.text())
    cap.frame_list.item(4).setCheckState(Qt.CheckState.Checked)
    app.processEvents()
    check("ticking it back clears both warnings",
          not cap.left_out.isVisibleTo(w) and not proc.left_out.isVisibleTo(w))

    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
