"""The 3D relief inside the window (VIEW-004): what works without a graphics card.

Offscreen, so OpenGL is not there; everything the view does on the CPU is:

* without OpenGL it says so at once and points to the separate 3D window
  (a platform with no OpenGL never calls initializeGL, so it used to sit
  blank);
* its own keys stay with it while it has focus: S, C, F, ? and the rest
  reach the 3D tools, not the View menu's single-key commands for the 2D
  image, and the window's own shortcuts (⌥2, ⌘⌥S) still work.  ⌘⇧S, which
  the key card used to offer for a picture, is Save As;
* every key on the key card is one the view answers, and the Keyboard
  Shortcuts sheet lists them, with a filter;
* the camera and the gizmo agree on the compass; exaggeration 1 is the true
  shape (the scale fitted between the height map and the normals); picking
  finds the surface; measuring and sections give pixels, or millimetres with
  a scale.

    python tests/test_qt_relief.py
"""
import math
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def bumps(h=120, w=160, amp=6.0):
    """A height field in pixels, its normals (+Y up) and the height map scaled away."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = amp * np.exp(-(((xx - 0.4 * w) / 18) ** 2 + ((yy - 0.5 * h) / 15) ** 2))
    zy, zx = np.gradient(z)
    n = np.dstack([-zx, zy, np.ones_like(zx)])               # rows run down: +Y up flips zy
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return z, n.astype(np.float32)


def main():
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    from helio_qt import relief3d as R
    print("=" * 60)
    print("The 3D relief")
    print("=" * 60)

    print("\nThe sums")
    z, n = bumps()
    s_true = 37.0
    check("exaggeration 1 is the true shape: the fitted scale recovers the height map's",
          abs(R.true_shape_scale(z / s_true, n) - s_true) / s_true < 0.03,
          R.true_shape_scale(z / s_true, n))
    check("…and a flat or empty map leaves it at 1", R.true_shape_scale(np.zeros((40, 40)), n[:40, :40])
          == 1.0 and R.true_shape_scale(np.zeros((4, 4)), n[:4, :4]) == 1.0)
    cam = R.Camera()
    eyes = {}
    for az in (0.0, 90.0, 180.0, 270.0):
        cam.az, cam.el = az, 10.0
        eyes[az] = cam.eye()
    check("the camera's compass: az 0 looks north from the south, 90 from the east",
          eyes[0.0][1] < 0 and eyes[90.0][0] > 0 and eyes[180.0][1] > 0 and eyes[270.0][0] < 0)
    V = cam.view()[:3, :3]
    check("the view turns without stretching", np.allclose(V @ V.T, np.eye(3), atol=1e-9))

    print("\nWithout OpenGL")
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: ("", ""))
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    v = R.ReliefView()
    v.resize(900, 600)
    ok, why = R.gl_available()
    check("the offscreen platform is known to have no OpenGL", not ok and "offscreen" in why, why)
    check("…so the view says so at once, with the way to the separate 3D window",
          v.gl_ok is False and not v.fallback.isHidden()
          and "OpenGL 4.1" in v.fallback.text() and "View › 3D Model" in v.fallback.text())
    check("…and hides its tools and key card, which would do nothing",
          v.bar.isHidden() and v.card.isHidden())
    v._place_overlays()
    check("…even after a resize lays the overlays out again", v.card.isHidden())

    print("\nLengths in 3D are the photograph's, as in Measure")
    # The relief is shown at the preview's size; the scale is per pixel of the
    # photographs.  On a 100 MP capture (4.58x) every 3D length came out 4.58
    # times too short until 2026-10-17.
    ph, pw, full = 250, 188, (861, 1145)
    flat_n = np.dstack([np.zeros((ph, pw)), np.zeros((ph, pw)), np.ones((ph, pw))]).astype(np.float32)
    v.set_surface(np.zeros((ph, pw), np.float32), flat_n, np.full((ph, pw, 3), 128, np.uint8),
                  scale_mm=0.02917, full_size=full)
    W3, H3 = v.world_size()

    def at(u, vv):
        return np.array([(u - 0.5) * W3, (0.5 - vv) * H3, 0.0])
    v.points = [at(0.1, 0.5), at(0.9, 0.5)]
    m = v.measure()
    want_px = 0.8 * full[0]
    check(f"two points across are {want_px:.0f} of the photograph's pixels, not the preview's",
          abs(m["across_px"] - want_px) < 0.5, m["across_px"])
    check("…and the same millimetres as Measure would give",
          abs(m["across_mm"] - want_px * 0.02917) < 1e-3, m["across_mm"])
    v.section = ((0.5, 0.1), (0.5, 0.9))
    sp = v.section_profile()
    check("a section down the surface measures in the photograph's millimetres",
          abs(sp["length"] - 0.8 * full[1] * 0.02917) < 1e-3, sp["length"])
    v.set_scale(0.05, None)
    check("a scale set afterwards (Measure) is followed",
          abs(v.measure()["across_mm"] - want_px * 0.05) < 1e-3)

    print("\nMesh density")
    big_n = np.dstack([np.zeros((2500, 1875)), np.zeros((2500, 1875)), np.ones((2500, 1875))]).astype(np.float32)
    v.set_mesh_step(1)
    v.set_surface(np.zeros((2500, 1875), np.float32), big_n, np.full((2500, 1875, 3), 128, np.uint8))
    check("1 : 1 is really 1 : 1 on a 2500 px preview (it was turned into 1 : 2)",
          v.mesh_step == 1 and v.mesh_used() == 1, (v.mesh_step, v.mesh_used()))
    huge = np.dstack([np.zeros((3600, 2700)), np.zeros((3600, 2700)), np.ones((3600, 2700))]).astype(np.float32)
    v.set_surface(np.zeros((3600, 2700), np.float32), huge, np.full((3600, 2700, 3), 128, np.uint8))
    v._readout()
    check("…and when a surface is too big for it, the step used is shown, and why",
          v.mesh_used() == 2 and "1 : 2 (1 : 1 would pass" in v._readout_text, v._readout_text)
    v.set_mesh_step(2)
    del big_n, huge

    print("\nThe surface on the CPU")
    v.set_surface(z / s_true, n, np.full(z.shape + (3,), 180, np.uint8), scale_mm=0.05)
    v.set_exaggeration(1.0)
    v.cam.az, v.cam.el = 0.0, 89.0
    v.fit(animate=False)
    cx, cy = v.width() / 2, v.height() / 2
    q = v.pick(cx, cy)
    u, vv = v.uv_of(q) if q is not None else (None, None)
    check("a click in the middle of a top view lands in the middle of the relief",
          q is not None and abs(u - 0.5) < 0.02 and abs(vv - 0.5) < 0.02, (u, vv))
    want = v.height_at(u, vv) * v.zscale() if q is not None else None
    check("…on the surface, not above or below it", q is not None and abs(q[2] - want) < 1e-3,
          (None if q is None else q[2], want))
    v.points = [np.array([-0.5, 0.0, 0.0]), np.array([0.5, 0.0, 0.0])]
    m = v.measure()
    check("measuring: half the width apart is 80 px across, 4 mm at 0.05 mm per pixel",
          m and abs(m["across_px"] - 80.0) < 1e-6 and abs(m["across_mm"] - 4.0) < 1e-6
          and m["dh_from"] == "the normals", m)
    v.section = ((0.1, 0.5), (0.9, 0.5))
    prof = v.section_profile()
    check("a section across the bump peaks where the bump is, in mm from the normals",
          prof and prof["unit"].startswith("mm") and 0.25 < float(np.argmax(prof["z"])) / len(prof["z"]) < 0.5,
          None if not prof else (prof["unit"], int(np.argmax(prof["z"]))))
    pk = float(prof["z"].max() - prof["z"].min()) if prof else 0
    check(f"…and its height is the bump's: {pk:.3f} mm for {z.max() * 0.05:.3f}",
          abs(pk - z.max() * 0.05) / (z.max() * 0.05) < 0.06)
    v.set_light_screen(0.0, -1.0, 0.2)
    check("the 2D light (screen, +Y down) becomes the 3D light (+Y north)", v.light[1] > 0.9,
          v.light)

    print("\nIts keys, in the window")
    fixture = os.path.join(ROOT, "docs", "pyside6-migration", "tools", "capture_baseline.py")
    import importlib.util
    import cv2
    spec = importlib.util.spec_from_file_location("cb", fixture)
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    _zg, n_gt = cb.surface()
    L = cb.lights()
    src = tempfile.mkdtemp(prefix="helio_qt_relief_")
    for i, lv in enumerate(L):
        img = np.clip(n_gt @ lv, 0, 1) * 235 + 10
        cv2.imwrite(os.path.join(src, f"img_{i:02d}.png"),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    w.open_paths([src])
    w.project.set_lights(L, "file:fixture.lp")
    w.project.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 300:
        app.processEvents()
        time.sleep(0.01)
    w.go_workspace("Inspect")
    w.set_centre_mode("both")
    for _ in range(20):
        app.processEvents()
    ins = w.workspaces["Inspect"]
    r = ins.relief
    check("side by side: the 3D view is made beside the image", r is not None and not r.isHidden())
    class _Big:                                  # a 12 MP solve, with its 2500 px preview
        w, h, pw = 4000, 3000, 2500
        depth_vis_p = np.zeros((3000, 4000), np.float32)
        normals_full = np.dstack([np.zeros((3000, 4000)), np.zeros((3000, 4000)),
                                  np.ones((3000, 4000))]).astype(np.float32)
        albedo_rgb_full = np.zeros((3000, 4000, 3), np.uint8)
        depth_proxy_p = np.zeros((1875, 2500), np.float32)
        normals_proxy = np.zeros((1875, 2500, 3), np.float32)
        albedo_proxy = np.zeros((1875, 2500, 3), np.uint8)
    A_ = ins._relief_arrays(_Big())
    pts = A_["depth"].size
    check(f"1 : 1 is built from up to 7 million points of the full solve, not the 2500 px "
          f"preview ({pts / 1e6:.2f} M)", 6.8e6 < pts <= 7e6 and A_["normals"].shape[:2]
          == A_["depth"].shape and A_["albedo"].shape[:2] == A_["depth"].shape, A_["depth"].shape)
    del A_
    ins._rarr = None
    check("the 3D relief's Height starts at 0.6", abs(r.exaggeration - 0.6) < 1e-9
          and abs(ins.relief_rows["exaggeration"].value() - 0.6) < 1e-9,
          (r.exaggeration, ins.relief_rows["exaggeration"].value()))
    fired = []
    for sid, a in w.registry.actions.items():
        a.triggered.connect(lambda _c=False, sid=sid: fired.append(sid))
    for _key, _sid, a in getattr(w, "alias_actions", []):
        a.triggered.connect(lambda _c=False, k=_key: fired.append(f"alias {k}"))
    r.setFocus()
    app.processEvents()
    check("the 3D view takes focus", app.focusWidget() is r, type(app.focusWidget()).__name__)
    NM = Qt.KeyboardModifier.NoModifier
    for key, mods, label, test in (
            (Qt.Key.Key_S, NM, "S starts a section, not Save View", lambda: r.tool == "section"),
            (Qt.Key.Key_C, NM, "C toggles contours, not the 2D Clay view", lambda: r.contours),
            (Qt.Key.Key_M, NM, "M measures", lambda: r.tool == "measure"),
            (Qt.Key.Key_Escape, NM, "Esc leaves the tool", lambda: r.tool == ""),
            (Qt.Key.Key_BracketRight, NM, "] is more exaggeration", lambda: r.exaggeration > 0.6),
            (Qt.Key.Key_F, NM, "F fits the relief, not the image", lambda: True),
            (Qt.Key.Key_Question, Qt.KeyboardModifier.ShiftModifier,
             "? shows or hides the keys, not the shortcuts sheet", lambda: True)):
        fired.clear()
        QTest.keyClick(r, key, mods)
        app.processEvents()
        check(label, not fired and test(), fired)
    fired.clear()
    QTest.keyClick(r, Qt.Key.Key_2, Qt.KeyboardModifier.AltModifier)
    app.processEvents()
    check("⌥2 still reaches the window: 3D alone", "view.mode:3d" in fired and ins.centre_mode == "3d",
          fired)
    fired.clear()
    QTest.keyClick(r, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
    app.processEvents()
    check("⌘⌥S opens Save View from the 3D view", fired == ["view.snapshot"], fired)
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent

    def claims(key, mods=NM):
        ev = QKeyEvent(QEvent.Type.ShortcutOverride, key, mods)
        ev.ignore()
        QApplication.sendEvent(r, ev)
        return ev.isAccepted()
    check("it claims its own keys before any menu can (how macOS menus ask first)",
          claims(Qt.Key.Key_S) and claims(Qt.Key.Key_C) and claims(Qt.Key.Key_F)
          and claims(Qt.Key.Key_Question, Qt.KeyboardModifier.ShiftModifier))
    check("…and leaves ⌘ and ⌥ combinations to the window",
          not claims(Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
          and not claims(Qt.Key.Key_2, Qt.KeyboardModifier.AltModifier)
          and not claims(Qt.Key.Key_E))
    keys_on_card = {k for _g, rows in R.KEYS for k, _w in rows}
    check("the key card offers ⌘⌥S for the picture, and no longer ⌘⇧S (Save As)",
          "⌘⌥S" in keys_on_card and "⌘⇧S" not in keys_on_card)
    single = {"F": Qt.Key.Key_F, "Home": Qt.Key.Key_Home, "M": Qt.Key.Key_M, "S": Qt.Key.Key_S,
              "C": Qt.Key.Key_C, "T": Qt.Key.Key_T, "Esc": Qt.Key.Key_Escape, "?": Qt.Key.Key_Question}
    missing = [k for k, q in single.items() if k in keys_on_card and int(q) not in R.ReliefView.OWN_KEYS]
    check("every single key on the card is one the view keeps while it has focus", not missing, missing)

    print("\nThe section A–B and its card")
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtGui import QMouseEvent
    check("no card until a section is drawn", not ins.section_card.isVisibleTo(w))
    w.resize(1440, 1100)
    print(f"  (the relief {r.width()} × {r.height()} before the card)")                   # room for the relief and its card
    r.set_section((0.15, 0.5), (0.85, 0.5))
    for _ in range(10):
        app.processEvents()                # the card opens: the relief is laid out again
    prof = ins.section_prof
    check("drawing one opens the card under the 3D view, with its length and roughness",
          ins.section_card.isVisibleTo(w) and prof and "long" in ins.section_summary.text()
          and "Rq" in ins.section_summary.text() and ins.relief_profile.prof is prof,
          ins.section_summary.text())
    check("…the profile carries its ends in the photographs' pixels, and its grooves and ridges",
          prof and len(prof["a_px"]) == 2 and prof["b_px"][0] > prof["a_px"][0]
          and isinstance(prof["features"], list) and ins.section_features.text() != "")
    a, b = r.section_ends_screen()
    check("A and B are on screen at the line's ends", a is not None and b is not None and b.x() > a.x())
    chart = ins.relief_profile
    rr = chart.plot_rect()

    def move(widget, pt, buttons=Qt.MouseButton.NoButton, mods=NM):
        QApplication.sendEvent(widget, QMouseEvent(QEvent.Type.MouseMove, QPointF(pt), QPointF(pt),
                                                   Qt.MouseButton.NoButton, buttons, mods))
        app.processEvents()
    move(chart, QPoint(int(rr.left() + rr.width() * 0.5), int(rr.center().y())))
    check("moving along the chart puts a marker half way along the line in 3D",
          r.section_t is not None and abs(r.section_t - 0.5) < 0.03, r.section_t)
    mid = (a + b) / 2
    r._hover_section(mid)
    app.processEvents()
    check("…and pointing at the line in 3D moves the chart's pointer", chart.cursor_t is not None
          and abs(chart.cursor_t - 0.5) < 0.05, chart.cursor_t)
    for frac in (0.25, 0.75):
        QTest.mouseClick(chart, Qt.MouseButton.LeftButton, NM,
                         QPoint(int(rr.left() + rr.width() * frac), int(rr.center().y())))
    m2 = chart.measurement()
    check("two clicks on the chart measure between them (half the length apart)",
          m2 is not None and abs(m2[0] - prof["length"] * 0.5) < prof["length"] * 0.02, m2)
    ve = chart.exaggeration()
    check("the chart says how much it exaggerates the height (same units)",
          ve is None or ve > 0, ve)
    # Drag B, with Shift: it moves, and stays level with A.
    QTest.mousePress(r, Qt.MouseButton.LeftButton, NM, b.toPoint())
    target = QPoint(int(b.x() - 30), int(b.y() + 25))
    move(r, target, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier)
    QTest.mouseRelease(r, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ShiftModifier, target)
    app.processEvents()
    (ua, va), (ub, vb) = r.section
    check("B can be dragged to a new place, and Shift keeps the line straight across",
          abs(vb - va) < 1e-9 and abs(ub - 0.85) > 1e-3 and ins.section_prof is not prof,
          r.section)
    check("…its grooves and ridges are found again when the drag ends",
          isinstance(ins.section_prof.get("features"), list))
    r.set_tool("measure")
    before_sec = r.section
    a_now = r.section_ends_screen()[0]
    QTest.mouseClick(r, Qt.MouseButton.LeftButton, NM, a_now.toPoint())
    app.processEvents()
    check("with Measure on, a click by A places a measure point and leaves the section",
          r.section == before_sec and len(r.points) == 1, (r.section, len(r.points)))
    r.set_tool("")
    out = tempfile.mkdtemp(prefix="helio_section_")
    import helio_qt.inspect_ws as iw
    from PySide6.QtWidgets import QFileDialog
    real = QFileDialog.getSaveFileName
    from PySide6.QtWidgets import QWidget as _QW
    parents = []

    def fake_save(parent, *a_, **k):            # as Qt's: the parent must be a widget
        parents.append(parent)
        if parent is not None and not isinstance(parent, _QW):
            raise TypeError("getSaveFileName called with wrong argument types")
        return os.path.join(out, "sec.csv"), ""
    QFileDialog.getSaveFileName = staticmethod(fake_save)
    try:
        ins.save_section()
    finally:
        QFileDialog.getSaveFileName = real
    got = sorted(os.listdir(out))
    rows = open(os.path.join(out, "sec.csv")).read().splitlines() if "sec.csv" in got else []
    check("Save writes the profile CSV and the chart as SVG and PNG",
          {"sec.csv", "sec_chart.png", "sec_chart.svg"} <= set(got) and len(rows) > 100
          and rows[1].startswith("distance_"), got)
    fx = w._plate_facts()
    check("the publication plate is given this line for its section",
          fx.get("section") == (tuple(ins.section_prof["a_px"]), tuple(ins.section_prof["b_px"])),
          fx.get("section"))
    ins.set_centre_mode("image")
    w.set_selection((10, 10, 120, 90))
    app.processEvents()
    check("a selection changed while the 3D is hidden: the plate does not take the old line",
          "section" not in w._plate_facts())
    w.set_selection(None)
    ins.set_centre_mode("both")
    for _ in range(10):
        app.processEvents()
    r.set_section((0.15, 0.5), (0.85, 0.5))
    app.processEvents()
    r.clear_section()
    app.processEvents()
    check("Clear removes the line and closes the card", r.section is None
          and not ins.section_card.isVisibleTo(w) and "section" not in w._plate_facts())

    print("\nThe section on the 2D image")
    check("Clear takes the line off the picture too",
          not getattr(ins.viewport, "section_ends", None))
    ins.set_centre_mode("image")
    app.processEvents()
    D_ = w.render_data
    ins.btn_section.setChecked(True)
    check("Section turns on the image's line tool", ins.viewport.tool == "line")
    ins.viewport.lineDrawn.emit(D_.w * 0.2, D_.h * 0.5, D_.w * 0.8, D_.h * 0.5)
    for _ in range(10):
        app.processEvents()
    check("a line drawn on the image is the 3D relief's section, its card shown in Image mode",
          r.section is not None and abs(r.section[0][0] - 0.2) < 0.01
          and abs(r.section[1][0] - 0.8) < 0.01 and ins.section_card.isVisibleTo(w)
          and not ins.relief_box.isVisibleTo(w) and ins.viewport.tool == "", r.section)
    ends = ins.viewport.section_ends
    check("…and drawn on the picture with A and B where it was drawn",
          ends and abs(ends[0][0] - D_.w * 0.2) < 2 and abs(ends[1][0] - D_.w * 0.8) < 2, ends)
    ins.relief_profile.hovered.emit(0.5)
    dot = ins.viewport._secdot
    check("the chart's pointer shows on the picture's line, and in 3D",
          dot.isVisible() and abs(dot.pos().x() - D_.w * 0.5) < 2
          and r.section_t is not None and abs(r.section_t - 0.5) < 0.01, (dot.pos(), r.section_t))
    r.set_section((0.5, 0.1), (0.5, 0.9))
    app.processEvents()
    ends = ins.viewport.section_ends
    check("a line moved in 3D moves on the picture",
          ends and abs(ends[0][0] - D_.w * 0.5) < 2 and abs(ends[0][1] - D_.h * 0.1) < 2, ends)
    box = ins.section_card.chk_share
    check("Inspect's section card has Show on the 2D image too, ticked", box is not None
          and box.isChecked() and box.isVisibleTo(w))
    box.setChecked(False)
    app.processEvents()
    check("…unticked, the line leaves the picture (the 3D line and card stay)",
          not getattr(ins.viewport, "section_ends", None) and r.section is not None
          and ins.section_card.isVisibleTo(w))
    box.setChecked(True)
    app.processEvents()
    check("…ticked, it is back", bool(getattr(ins.viewport, "section_ends", None)))
    r.clear_section()
    app.processEvents()
    sel = (int(D_.w * 0.25), int(D_.h * 0.25), int(D_.w * 0.75), int(D_.h * 0.75))
    w.set_selection(sel)
    ins.set_centre_mode("both")
    for _ in range(10):
        app.processEvents()
    ins._push_relief()
    r.set_section((0.0, 0.5), (1.0, 0.5))
    app.processEvents()
    ends = ins.viewport.section_ends
    check("with a selection, a line across the selected part's relief lands across that part "
          "of the picture", ends and abs(ends[0][0] - sel[0]) < 2 and abs(ends[1][0] - sel[2]) < 2
          and abs(ends[0][1] - (sel[1] + sel[3]) / 2) < 2, (ends, sel))
    r.clear_section()
    w.set_selection(None)
    ins.set_centre_mode("image")
    app.processEvents()

    print("\nDouble-click: turn round that point")
    from PySide6.QtCore import QPoint as _QP
    from PySide6.QtTest import QTest as _QT
    ins.set_centre_mode("3d")
    for _ in range(10):
        app.processEvents()
    r.fit(animate=False)
    cx, cy = r.width() // 2 + 40, r.height() // 2 + 25
    q_ = r.pick(cx, cy)
    _QT.mouseDClick(r, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _QP(cx, cy))
    t0_ = time.time()
    while time.time() - t0_ < 1.0:
        app.processEvents()
        time.sleep(0.02)
    check("double-click on the surface: the camera turns round the point clicked",
          q_ is not None and np.allclose(r.cam.target, q_, atol=1e-6), (q_, r.cam.target))
    _QT.mouseDClick(r, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, _QP(3, 3))
    t0_ = time.time()
    while time.time() - t0_ < 1.0:
        app.processEvents()
        time.sleep(0.02)
    check("…on the background: back to the centre, the whole relief in view",
          np.allclose(r.cam.target, 0.0, atol=1e-6), r.cam.target)
    ins.set_centre_mode("image")
    app.processEvents()

    print("\nArea and volume")
    r.set_tool("area")
    check("A (and the Area button) turns on the area tool", r.tool == "area"
          and r.tool_buttons["area"].isChecked() and "click round" in r._readout_text)
    r.area_uv = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7)]
    r.finish_area()
    a_ = r.area_result
    full_w, full_h = w.render_data.w, w.render_data.h
    want = 0.4 * full_w * 0.4 * full_h * (w.project.scale_mm_per_px or 1.0) ** 2
    check("closing the outline gives its area, in the photographs' units, and the volume about "
          "the plane through its edge",
          a_ is not None and abs(a_["area"] / want - 1) < 0.05 and a_["surface"] >= a_["area"]
          and a_["below"] >= 0 and a_["above"] >= 0 and "Area" in r._readout_text
          and "volume below" in r._readout_text, (a_, want))
    r.set_tool("")
    check("leaving the tool keeps a closed outline; Esc-style leave lets an open one go",
          r.area_done and r.area_uv)
    r.set_tool("area")
    r.area_uv, r.area_done, r.area_result = [(0.1, 0.1)], False, None
    r.set_tool("")
    check("…an open outline is let go", not r.area_uv)

    print("\nThe high-detail 3D window")
    w.registry.trigger("view3d.window")
    for _ in range(20):
        app.processEvents()
    hd = w.hd3d
    hv = hd.view
    check("Open High-Detail 3D Window opens Helio's own 3D view in a window, built at full size "
          "up to 9 million points", hd.isVisible() and hv.data is not None
          and hv.MAX_MESH_POINTS == 9_000_000 and hv.mesh_used() == 1
          and (hv.data["w"], hv.data["h"]) == (w.render_data.w, w.render_data.h),
          None if hv.data is None else (hv.data["w"], hv.data["h"]))
    check("…with the same toolbar and key card as the 3D relief",
          set(hv.tool_buttons) == set(r.tool_buttons) and hv.card is not None)
    ins.relief_rows["exaggeration"].set_value(2.5)
    ins.relief_rows["exaggeration"].changed.emit("exaggeration", 2.5, True)
    ins.relief_clay.setCurrentText("Blue")
    ins.relief_material.set_current("clay")
    ins.relief_material.chosen.emit("clay")
    hd._follow()
    from helio.application.viewer3d import CLAY as _CL
    check("…following the 3D panel: Height, clay colour and material",
          abs(hv.exaggeration - 2.5) < 1e-9 and hv.clay == _CL["Blue"] and hv.material == "clay",
          (hv.exaggeration, hv.clay, hv.material))
    hv.lightMoved.emit(0.5, -0.5, 0.707, False)
    app.processEvents()
    lx, ly, lz = ins._current_light()
    check("…and dragging the light in it moves the one light everywhere",
          abs(lx - 0.5) < 0.05 and abs(ly + 0.5) < 0.05, (lx, ly, lz))
    check("the classic viewer is gone from Helio; Helio's own Flyover is in its place",
          "view3d.classic" not in w.registry.actions and "view3d.record:3" not in w.registry.actions
          and "view.relief_flyover" in w.registry.actions)
    hv.set_section((0.15, 0.5), (0.85, 0.5))
    for _ in range(5):
        app.processEvents()
    check("a section drawn in the window opens its own card at the foot, with its length",
          hd.card.isVisibleTo(hd) and "long" in hd.card.summary.text() and hd.card.prof is not None,
          hd.card.summary.text())
    hd.card.hovered.emit(0.5)
    check("…the card's pointer shows on the window's line", hv.section_t is not None
          and abs(hv.section_t - 0.5) < 0.01)
    ins.set_centre_mode("image")
    app.processEvents()
    hd.card.chk_share.setChecked(True)
    for _ in range(5):
        app.processEvents()
    ends = getattr(ins.viewport, "section_ends", None)
    check("Show on the 2D image: the window's line on Inspect's picture, with Inspect's card",
          ins.relief is not None and ins.relief.section == hv.section and ends is not None
          and ins.section_card.isVisibleTo(w), (ins.relief and ins.relief.section, hv.section, ends))
    hv.set_section((0.2, 0.3), (0.8, 0.7))
    for _ in range(5):
        app.processEvents()
    check("…and a line moved in the window moves there too",
          ins.relief.section == hv.section, (ins.relief.section, hv.section))
    hd.card.chk_share.setChecked(False)
    app.processEvents()
    check("unticked, it leaves the picture", ins.relief.section is None
          and not getattr(ins.viewport, "section_ends", None))
    hd.card.clear_requested.emit()
    app.processEvents()
    check("…and Clear removes the line and the card", hv.section is None and not hd.card.isVisibleTo(hd))
    kind = lambda: hd.windowFlags() & Qt.WindowType.WindowType_Mask
    check("the window keeps above Helio's main window by default (a floating panel)",
          hd.btn_top.isChecked() and kind() == Qt.WindowType.Tool and hd.isWindow())
    hd.btn_top.setChecked(False)
    app.processEvents()
    check("…Keep on top off: an ordinary window, still open and separate",
          kind() == Qt.WindowType.Window and hd.isWindow() and hd.isVisible())
    hd.btn_top.setChecked(True)
    app.processEvents()
    check("…and on again", kind() == Qt.WindowType.Tool and hd.isVisible())
    from PySide6.QtWidgets import QPushButton
    check("the window's keys go to its 3D view, and its buttons do not take them",
          hd.focusProxy() is hv and all(b_.focusPolicy() == Qt.FocusPolicy.NoFocus
                                        for b_ in hd.findChildren(QPushButton)
                                        if b_.parent() is hd))
    hd._follow()                                    # caught up, as its timer keeps it
    hv.set_light_screen(0.6, -0.3, 0.74)
    held = tuple(float(c) for c in hv.light)
    ins.relief_rows["exaggeration"].set_value(3.0)
    ins.relief_rows["exaggeration"].changed.emit("exaggeration", 3.0, True)
    hd._follow()
    check("a light moved in the window stays when another 3D setting changes",
          all(abs(a_ - b_) < 1e-6 for a_, b_ in zip(held, hv.light)), (held, hv.light))
    import helio_qt.relief_window as RW
    check("on a small capture the mesh is already full size: nothing sharper is made",
          hv.data.get("normals_tex") is None)
    mesh_w = hv.data["w"]
    hv.data["w"] = 16                               # as if the mesh were small beside the solve
    hd._detail_of = None
    hd.sharpen(w.selected_result()[1])
    w.wait_background()
    app.processEvents()
    hv.data["w"] = mesh_w
    hv._readout()
    nt = hv.data.get("normals_tex") if hv.data is not None else None
    long_side = max(w.render_data.w, w.render_data.h)
    check("the window shades the mesh with the solve's normals at up to 4,500 px, not the mesh's size",
          nt is not None and max(nt.shape[:2]) == min(RW.DETAIL_SIDE, long_side)
          and hv.data.get("albedo_tex") is not None and hv.data["albedo_tex"].shape[:2] == nt.shape[:2]
          and f"{nt.shape[1]:,} × {nt.shape[0]:,}" in hv._readout_text,
          None if nt is None else (nt.shape, hv.data["w"], hv._readout_text))
    hd.close()
    ins.relief_clay.setCurrentText("Grey")
    ins.relief_material.set_current("colour")
    ins.relief_material.chosen.emit("colour")

    print("\nThe 3D panel: one set of controls, in the window")
    w.set_centre_mode("image")
    w.registry.trigger("view3d.open:standard")
    app.processEvents()
    check("Open in 3D (⌘⇧3) shows the relief in the window, not a separate one",
          ins.centre_mode == "3d" and not r.isHidden() and not w.model3d.alive()
          and ins.sec_relief.isVisibleTo(w))

    def shaped():
        if hasattr(ins, "shape_renderer"):
            ins.shape_renderer.wait()
        for _ in range(5):
            app.processEvents()
        return r.data["depth"].copy()
    z0 = shaped()
    ins.shape_rows["shape_blend"].changed.emit("shape_blend", 0.0, True)
    z1 = shaped()
    check("a Shape slider reshapes the relief (Detail vs shape: the whole shape back)",
          z1.shape == z0.shape and float(np.abs(z1 - z0).mean()) > 1e-3
          and w.shared["shape_blend"] == 0.0, float(np.abs(z1 - z0).mean()))
    ins._relief_shape_reset()
    shaped()
    w.registry.trigger("view3d.open:inverted")
    z2 = shaped()
    check("Inverted turns the relief over",
          ins.relief_kind.current() == "inverted" and float(np.corrcoef(z2.ravel(), z0.ravel())[0, 1]) < -0.5)
    w.registry.trigger("view3d.open:heatmap")
    check("the heatmap is the relief coloured by height", r.material == "height"
          and ins.relief_kind.current() == "standard")
    ins.tone3d_rows["viewer_gamma"].changed.emit("viewer_gamma", 1.6, True)
    ins.relief_clay.setCurrentText("Mustard")
    ins.relief_rows["shine"].changed.emit("shine", 64.0, True)
    from helio.application.viewer3d import CLAY
    check("display tone, clay colour and shininess reach the 3D view",
          abs(r.tone["gamma"] - 1.6) < 1e-9 and r.clay == CLAY["Mustard"] and r.shine == 64.0)

    print("\nTilt correction, and the recordings' camera")
    ins.shape_rows["x_tilt"].changed.emit("x_tilt", 5.0, True)
    z_t = shaped()
    ins.shape_rows["x_tilt"].changed.emit("x_tilt", 0.0, True)
    z_0 = shaped()
    slope = float((z_t - z_0)[:, -1].mean() - (z_t - z_0)[:, 0].mean())
    check(f"Tilt left–right takes a slope out of the relief, as the classic viewer did "
          f"({slope:+.3f} across)", abs(slope + np.tan(np.radians(5.0))) < 0.01
          and w.shared["x_tilt"] == 0.0, slope)
    from helio.application.viewer3d import flyover_pose, turntable_pose, write_video
    n = 360
    poses = [flyover_pose(i, n, 30.0, 42.0, 3.2) for i in range(n + 1)]
    els = [p_[1] for p_ in poses]
    dists = [p_[2] for p_ in poses]
    step = max(abs(((poses[i + 1][0] - poses[i][0] + 180) % 360) - 180) for i in range(n))
    check("a flyover is one smooth loop: it ends where it began, dips low, moves in, never jumps",
          poses[0] == poses[n] and min(els) < 30 < 55 < max(els) and min(dists) < 0.75 * 3.2
          and step <= 1.01 and turntable_pose(90, 360, 0, 42, 3.2) == (90.0, 42, 3.2),
          (min(els), max(els), min(dists), step))
    from helio.application.viewer3d import video_pose
    sweep = [video_pose("light", i, 120, 30.0, 42.0, 3.2, light_el=25.0) for i in range(121)]
    check("a light sweep keeps the camera still and circles the light once at the height asked",
          all(p_[:3] == (30.0, 42.0, 3.2) for p_ in sweep) and sweep[0] == sweep[120]
          and {round(p_[3][1]) for p_ in sweep} == {25}
          and abs(sweep[60][3][0] - 180.0) < 1e-6)
    ov = [video_pose("overhead", i, 90, 0.0, 42.0, 3.2) for i in range(91)]
    check("an overhead orbit looks down from high and loops", ov[0] == ov[90]
          and all(p_[1] > 60 and p_[3] is None for p_ in ov))
    from helio_qt.record_dialog import RecordDialog
    dlg = RecordDialog(w, r, "flyover")
    check("Turntable… and Flyover… open Record video with that style, its note, a length and a size",
          dlg.style.current() == "flyover" and "loop" in dlg.blurb.text()
          and dlg.settings()["seconds"] == 12 and dlg.settings()["size"] == (1920, 1080)
          and not dlg.light_el.isVisibleTo(dlg))
    dlg.style.chosen.emit("light")
    dlg.style.set_current("light")
    dlg.light_el.set_value(20.0)
    dlg.size.set_current("4K")
    st_ = dlg.settings()
    check("…a light sweep asks the light's height; 4K is 3840 × 2160",
          dlg.light_el.isVisibleTo(dlg) and st_["style"] == "light" and st_["light_el"] == 20.0
          and st_["size"] == (3840, 2160), st_)
    dlg.close()
    vid = os.path.join(tempfile.mkdtemp(), "fly.mp4")
    frames_ = [np.full((108, 192, 3), (i * 4) % 255, np.uint8) for i in range(30)]
    write_video(frames_, vid)
    import cv2 as _cv2
    cap = _cv2.VideoCapture(vid)
    nfr = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    check("recordings are written as MP4 video (H.264 where the Mac has it)",
          os.path.getsize(vid) > 0 and nfr == 30, nfr)

    print("\nThe Keyboard Shortcuts sheet")
    from helio_qt.palette import ShortcutsSheet
    sheet = ShortcutsSheet(w)
    texts = [sheet.list.item(i).text() for i in range(sheet.list.count())]
    check("it lists the 3D view's keys, group by group",
          sum(t.startswith("IN THE 3D VIEW") for t in texts) == len(R.KEYS)
          and any("Section through the relief" in t for t in texts))
    check("…and the picture key with its menu command",
          any("Save View" in t for t in texts))
    sheet.filter.setText("contours")
    shown = [sheet.list.item(i).text() for i in range(sheet.list.count())
             if not sheet.list.item(i).isHidden()]
    check("typing narrows it to the rows that match, under their heading",
          any("contours" in t.lower() for t in shown) and len(shown) <= 4, shown)
    sheet.filter.clear()
    check("…and clearing shows everything again",
          all(not sheet.list.item(i).isHidden() for i in range(sheet.list.count())))

    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
