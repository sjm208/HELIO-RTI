"""Precision photometric stereo in the app, offscreen, on a synthetic 4-light
capture with unequal, falling-off lamps and a 0.300 mm step.

The white-board light field is calibrated from board frames and applied by
Process; heights are taken from the lateral scale; Measure's accuracy check
finds the step within a few percent and the base flat; profiles carry a ±
from the normals' precision; the lens is calibrated from chessboard frames
and applied; the readiness checklist, the rig profile and the session carry
it all; and both calibrations turn off again.

    python tests/test_qt_precision.py
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

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


H, W = 480, 640
STEP_PX, MM = 6.0, 0.05            # a 6 px step at 0.05 mm per pixel: 0.300 mm


def lights4(el=40.0):
    return np.array([(math.cos(math.radians(el)) * math.cos(math.radians(az)),
                      math.cos(math.radians(el)) * math.sin(math.radians(az)), math.sin(math.radians(el)))
                     for az in (45, 135, 225, 315)])


def normals():
    from scipy.ndimage import gaussian_filter
    z = np.zeros((H, W))
    z[160:340, 360:560] = STEP_PX
    z = gaussian_filter(z, 4.0)
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(z)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def lamp_fields(L):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    u, v = (xx - W / 2) / (W / 2), (H / 2 - yy) / (H / 2)
    out = []
    for s, l in zip((0.75, 1.0, 1.3, 0.9), L):
        d = math.hypot(l[0], l[1])
        out.append(s * (1.0 + 0.25 * (u * l[0] / d + v * l[1] / d)))
    return out


def write_frames(folder, stem, n, L, E, albedo):
    os.makedirs(folder, exist_ok=True)
    paths = []
    for i, (l, e) in enumerate(zip(L, E)):
        img = np.clip(255 * albedo * e * np.clip((n * l).sum(2), 0, None), 0, 255).astype(np.uint8)
        p = os.path.join(folder, f"{stem}_{i + 1:02d}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        paths.append(p)
    return paths


def chessboards(folder):
    """12 views of a 9 x 6 chessboard through a lens with mild barrel distortion, at the capture's size."""
    os.makedirs(folder, exist_ok=True)
    K = np.array([[700.0, 0, W / 2], [0, 700.0, H / 2], [0, 0, 1]])
    dist = np.array([-0.08, 0.0, 0, 0, 0])
    sq, cols, rows = 24, 9, 6
    board = np.full(((rows + 3) * sq, (cols + 3) * sq), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    bh, bw = board.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    und = cv2.undistortPoints(np.stack([xx.ravel(), yy.ravel()], 1).reshape(-1, 1, 2), K, dist,
                              P=K).reshape(H, W, 2)
    S = np.array([[1.0, 0, -bw / 2], [0, 1.0, -bh / 2], [0, 0, 1]])
    rng = np.random.default_rng(7)
    paths = []
    for k in range(12):
        R, _ = cv2.Rodrigues(rng.uniform(-0.4, 0.4, 3) * np.array([1, 1, 0.5]))
        t = np.array([rng.uniform(-60, 60), rng.uniform(-40, 40), rng.uniform(560, 680)])
        Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
        img = cv2.remap(cv2.warpPerspective(board, Hm / Hm[2, 2], (W, H), borderValue=200),
                        und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=200)
        p = os.path.join(folder, f"chess_{k:02d}.png")
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: print("        (warning)", a[2] if len(a) > 2 else ""))
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    def idle(limit=600.0):
        t0 = time.time()
        while w.project.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    root = tempfile.mkdtemp(prefix="helio_precision_")
    L, n, E = lights4(), normals(), None
    E = lamp_fields(L)
    cap = write_frames(os.path.join(root, "capture"), "Step", n, L, E, 0.55)
    flat = np.dstack([np.zeros((H, W)), np.zeros((H, W)), np.ones((H, W))])
    boards = write_frames(os.path.join(root, "board"), "Board", flat, L, E, 0.8)
    chess = chessboards(os.path.join(root, "chess"))

    w.open_paths([os.path.dirname(cap[0])])
    p = w.project
    p.update_settings(photo_tone="linear")          # the fixture writes light, not sRGB
    p.set_lights(L.astype(np.float32), "file:rig.lp")
    w.registry.trigger("process.run")
    idle()
    r0 = p.result

    def tilt(r):
        m = r.normals_full[30:140, 30:300].reshape(-1, 3).mean(0)
        return math.degrees(math.acos(m[2] / np.linalg.norm(m)))
    check("without the light field the unequal lamps tilt the flat base", tilt(r0) > 3.0, tilt(r0))

    print("\nWhite-board light field")
    field = w.calibrate_light_field(boards)
    check("calibrated from the board frames", field is not None and p.settings.light_field.endswith(".npz"))
    check("…it finds the lamps unequal", field is not None and field["strength_spread"] > 1.5)
    pw = w.workspaces["Process"]
    pw.refresh()
    check("Process says what it is", "white" not in pw.lf_note.text().lower() or "lights" in pw.lf_note.text(),
          pw.lf_note.text())
    w.registry.trigger("process.run")
    idle()
    r1 = p.result
    check("the solve applies it and says so", r1 is not r0 and r1.light_field is not None, r1.light_field)
    check("…and the base comes out level (under 0.5°)", tilt(r1) < 0.5, tilt(r1))

    print("\nHeights and the accuracy check")
    p.set_scale_photo(MM)
    w.registry.trigger("measure.zscale_lateral")
    check("heights from the lateral scale", p.z_mm_per_unit == MM and p.z_from_scale)
    w.registry.trigger("measure.accuracy")
    ms = w.workspaces["Measure"]
    ms.acc_known.setValue(STEP_PX * MM)
    ms.viewport.rectDrawn.emit(30.0, 30.0, 300.0, 440.0)          # the base
    ms.viewport.rectDrawn.emit(400.0, 200.0, 520.0, 300.0)        # the step's top
    app.processEvents()
    acc = p.accuracy or {}
    check("the 0.300 mm step measures within 5%", acc.get("error_pct") is not None
          and abs(acc["error_pct"]) < 5.0, acc)
    check("…the base flat to under 5 µm RMS", (acc.get("flat_rms_mm") or 1) < 0.005, acc.get("flat_rms_mm"))
    check("…with a ± from the normals", acc.get("sigma_mm") is not None, acc)
    check("the panel shows the error", any(k == "Error" for k, _v in ms.acc_rows), ms.acc_rows)
    check("the plate notes it", any("reference step" in t for t in w._plate_facts().get("notes", [])))
    ms.use("profile")
    ms.viewport.lineDrawn.emit(50.0, 250.0, 600.0, 250.0)
    app.processEvents()
    check("a profile carries its ± from the normals' precision",
          ms.profile is not None and ms.profile.get("sigma_px") is not None
          and "±" in ms.rough.text(), ms.rough.text())

    print("\nLens")
    cal = w.calibrate_lens(chess, (9, 6))
    check("calibrated from the chessboards", cal is not None and p.settings.lens_calibration.endswith(".npz"),
          cal and cal.get("used"))
    check("…finding the barrel distortion", cal is not None and cal["dist"][0] < -0.04, cal and cal["dist"])
    w.registry.trigger("process.run")
    idle()
    check("the solve undistorts the frames and says so", p.result.lens is not None, p.result.lens)

    print("\nReadiness, rig profile, session")
    p.update_settings(photo_tone="auto")            # as a camera's frames would be: not linear
    pw.refresh()
    txt = pw.ready_note.text()
    check("the checklist counts what is ready", "of 9 ready" in txt and "✓ White-board light field" in txt
          and "✓ Lens corrected" in txt and "✓ Accuracy checked" in txt, txt)
    check("…and says the frames are not RAW", "○ Linear frames" in txt, txt)
    name = w.save_rig_profile("Precision rig")
    prof = w.rig_profiles().get(name) or {}
    check("the rig profile keeps both calibrations",
          prof.get("light_field") == p.settings.light_field and prof.get("lens_calibration")
          == p.settings.lens_calibration, prof)
    from helio.io import session as S
    st = S.encode(w.session_state())
    keep = st.get("helio3") or {}
    check("the session keeps the calibrations and the accuracy check",
          keep.get("solve", {}).get("light_field") == p.settings.light_field
          and keep.get("precision", {}).get("z_from_scale") is True
          and keep.get("precision", {}).get("accuracy"), keep.get("precision"))
    w.registry.trigger("process.light_field_clear")
    w.registry.trigger("process.lens_clear")
    check("both turn off again", not p.settings.light_field and not p.settings.lens_calibration)
    w.apply_rig_profile(name)
    check("…and come back with the rig profile", p.settings.light_field and p.settings.lens_calibration)

    print("\nSet up for measurement")
    from helio.application.precision import MEASUREMENT_SETTINGS
    p.update_settings(normal_engine="Linear Least Squares", pre_clahe=True, multi_norm=True,
                      photo_tone="stored")
    w.registry.trigger("process.measurement")
    check("one command sets every measuring choice",
          all(getattr(p.settings, k) == v for k, v in MEASUREMENT_SETTINGS.items()),
          {k: getattr(p.settings, k) for k in MEASUREMENT_SETTINGS})
    check("…and keeps the calibrations", p.settings.light_field and p.settings.lens_calibration)
    from helio.application.process_tools import accuracy_note
    check("beside the score: the measured error, or that it is a ranking",
          "measured on a reference" in accuracy_note(p.accuracy)
          and "not an accuracy" in accuracy_note(None), accuracy_note(p.accuracy))
    w.close()


main()
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
