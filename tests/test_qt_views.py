"""Inspect in the new interface: every view as the classic app draws it (VIEW-001).

Runs the Qt app offscreen.  It processes the golden-image stack through the
job manager, then renders all 37 views through the same path Inspect uses
and compares each with its approved reference in tests/golden/ (the
tests/test_golden.py rule).  It also drives the workspace: the view keys, the
light dome (it must light the side it faces), "Show bumps as holes", the
frames, and how long a frame takes while the light is dragged on the real
manuscript.

    python tests/test_qt_views.py
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
MAE_TOL, MAXBAD_FRAC_TOL = 1.0, 0.02        # tests/test_golden.py's rule
# The golden stack's lamps all stand at one height round a ring, so its light stack has two
# principal components: the third's eigenvalue is about 1e-5 of the second's, rounding noise
# stretched to full contrast, and its pixels follow the order the BLAS adds in (MAE 1.8 between
# machines).  It is checked for being drawn, not against its picture.
NOISE_ONLY = {"PCA 3"}
FRAME_MS = 100                               # VIEW-001: no stall over 100 ms while dragging
try:
    from helio.core._native import rs as _rs
except Exception:
    _rs = None
if _rs is None:
    FRAME_MS = 200                           # the NumPy fallback is allowed twice the time; Rust is held to 100


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
    import cv2
    print("=" * 60)
    print("Inspect: the views of the solved surface")
    print("=" * 60)
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(   # the next capture: don't save
        lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    def run_until_idle(limit=600.0):
        t0 = time.time()
        while w.project.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    print("\nBefore anything is solved")
    check("the view keys say why they cannot run",
          all(w.registry.reason(a) for a in ("view.next", "view.depth", "view.clay")))

    print("\nThe golden stack, solved in the new app")
    from tools.make_golden import GOLDEN_DIR, H, W, render_stack, safe
    tmp = tempfile.mkdtemp(prefix="helio_qt_golden_")
    paths, L = render_stack(tmp)
    w.open_paths([tmp])
    w.project.set_lights(L, "file:golden.lp")
    w.project.update_settings(photo_tone="linear")      # the stack is radiance, not sRGB
    w.registry.trigger("process.run")
    run_until_idle()
    ins = w.workspaces["Inspect"]
    check("solved, and Inspect opens on the solved surface",
          w.project.solved and w.render_data is not None)
    w.go_workspace("Inspect")
    app.processEvents()
    ins.wait_render()
    check("the first view is Standard PBR, drawn",
          w.view == "Standard PBR" and ins.mode == "view"
          and not ins.viewport.item.pixmap().isNull())

    print("\nEvery view matches its approved reference")
    from helio.application.render import render, to_display
    from helio.domain.registry import RENDER_ENGINES
    bad = []
    for eng in RENDER_ENGINES:
        img = render(eng, w.render_data, dict(w.engine_settings[eng]), H, W, False)
        a = to_display(img)
        ref = cv2.imread(os.path.join(ROOT, GOLDEN_DIR, safe(eng) + ".png"), cv2.IMREAD_COLOR)
        if ref is None or ref.shape != a.shape:
            bad.append(f"{eng}: shape {a.shape} vs {None if ref is None else ref.shape}")
            continue
        if eng in NOISE_ONLY:
            if not np.isfinite(np.asarray(img, np.float32)).all() or a.max() == a.min():
                bad.append(f"{eng}: not drawn")
            continue
        diff = np.abs(a.astype(np.int16) - ref[..., ::-1].astype(np.int16))
        mae, badfrac = float(diff.mean()), float((diff.max(axis=2) > 32).mean())
        if mae > MAE_TOL or badfrac > MAXBAD_FRAC_TOL:
            bad.append(f"{eng}: MAE {mae:.2f}, {100 * badfrac:.1f}% far off")
    check(f"{len(RENDER_ENGINES) - len(bad)} of {len(RENDER_ENGINES)} views match "
          "tests/golden/", not bad, "; ".join(bad[:4]))

    print("\nView keys")
    def go(name):
        w.set_view(name)
        ins.wait_render()
    def pix():
        im = ins.viewport.item.pixmap().toImage().convertToFormat(QImage.Format.Format_RGB888)
        a = np.frombuffer(im.constBits(), np.uint8).reshape(im.height(), im.bytesPerLine())
        return a[:, :im.width() * 3].reshape(im.height(), im.width(), 3).astype(np.int16).copy()
    w.registry.trigger("view.next")
    ins.wait_render()
    check(f"→ moves to the next view ({w.view})", w.view == ins.order[1])
    w.registry.trigger("view.prev")
    ins.wait_render()
    check("← moves back", w.view == ins.order[0])
    w.registry.trigger("view.depth")
    ins.wait_render()
    check("D shows the depth map, with no light", w.view == ins.DEPTH_MAP
          and not ins.dome.isVisibleTo(w) and not ins.sec_light.isVisibleTo(w))
    w.registry.trigger("view.depth")
    ins.wait_render()
    check("D again goes back to the view before it", w.view == "Standard PBR")
    colour = pix()
    w.registry.trigger("view.clay")
    ins.wait_render()
    grey = pix()
    check("C shows clay (grey), with the light dome", w.display.layer == "Shading"
          and ins.dome.isVisibleTo(w) and float(np.abs(grey[..., 0] - grey[..., 2]).mean()) < 0.5)
    w.registry.trigger("view.clay")
    ins.wait_render()
    check("C again brings the colour back", w.display.layer == "Combined Colour"
          and float(np.abs(pix()[..., 0] - pix()[..., 2]).mean()) > 2)
    w.registry.trigger("view.clay")                 # clay for the light checks below
    ins.wait_render()

    print("\nThe light belongs to the view (on_engine_change)")
    def region_means():
        g = pix().mean(axis=2)
        # the golden dome is centred at (row 120, col 80); its north and south flanks
        return float(g[90:106, 70:91].mean()), float(g[134:150, 70:91].mean())
    cx, cy = ins.dome._centre()
    ins.dome._from_pos(cx, cy - ins.dome.R * 0.8, False)        # drag the dot to the north
    ins.wait_render()
    st = w.engine_settings["Standard PBR"]
    n_n, n_s = region_means()
    check(f"dragging north gives a north light ({ins.dome.caption()})",
          st["ly"] < -0.5 and abs(st["lx"]) < 0.05 and ins.dome.caption().startswith("Light 0°"))
    check("...stored as the classic dome stores it: lx, ly, lz and the compass angles",
          abs(st["hs_azi"]) < 1 or abs(st["hs_azi"] - 360) < 1)
    check(f"...which lights the dome's north flank ({n_n:.0f} vs south {n_s:.0f})",
          n_n > 3 * n_s and n_n - n_s > 10)
    ins.dome._from_pos(cx, cy + ins.dome.R * 0.8, False)        # and to the south
    ins.wait_render()
    s_n, s_s = region_means()
    check(f"a south light lights the south flank ({s_s:.0f} vs north {s_n:.0f})",
          s_s > 3 * s_n and s_s - s_n > 10)
    ins.height_row.set_value(20)
    ins._height_moved("light_height", 20, True)
    ins.wait_render()
    check("Light height moves the light without turning it",
          abs(ins.dome.angles()[1] - 20) < 1 and abs(ins.dome.angles()[0] - 180) < 1)
    pbr_light = (st["lx"], st["ly"], st["lz"])
    go("Exaggerated Shading")
    ex = w.engine_settings["Exaggerated Shading"]
    check("another view keeps its own light", (ex["lx"], ex["ly"], ex["lz"]) != pbr_light
          and abs(ins.dome.lx - ex["lx"]) < 1e-6)
    go("Standard PBR")
    check("and coming back restores this view's light",
          abs(ins.dome.lx - pbr_light[0]) < 1e-6 and abs(ins.dome.ly - pbr_light[1]) < 1e-6)
    ins.same_light_everywhere()
    check("“Use this light in every view” gives them all this one",
          all(abs(w.engine_settings[e]["ly"] - pbr_light[1]) < 1e-6 for e in RENDER_ENGINES))
    ins.az.setValue(90)
    ins.el.setValue(30)
    ins.wait_render()
    check("typing azimuth and elevation sets the light (the classic Go)",
          abs(ins.dome.angles()[0] - 90) < 1 and abs(ins.dome.angles()[1] - 30) < 1)
    go("PS Normal-Map Lit (Single)")
    ins.dome._from_pos(cx + ins.dome.R * 0.6, cy, False)
    ins.wait_render()
    ps = w.engine_settings["PS Normal-Map Lit (Single)"]
    check("a compass-lit view follows the dome through its angles",
          abs(ps["hs_azi"] - 90) < 1 and ins.dome.isVisibleTo(w))
    go("Retinex Enhancement")
    check("a view that does not take the light has no dome", not ins.dome.isVisibleTo(w)
          and not ins.sec_light.isVisibleTo(w))
    before_rx = dict(w.engine_settings["Retinex Enhancement"])
    ins._relight_drag(0.5, 0.0, False)
    check("…and Option-drag on it changes nothing", w.engine_settings["Retinex Enhancement"] == before_rx)

    print("\nEach view shows its own controls")
    from helio.domain.view_controls import controls_for, takes_tone
    wrong = []
    offered = [e for e in RENDER_ENGINES if e not in ins.HIDDEN]
    check("the eight-direction hillshade, the GIS Hillshade and the Shaded Render are not offered",
          "PS Normal-Map Lit (8-Dir)" not in ins.items and "Hillshade" not in ins.items
          and "Shaded Render" not in ins.items)
    fl = [c for c in controls_for("Exaggerated Shading") if c.key == "exag_flatten"]
    check("Exaggerated Shading has a Flatten slider, starting at 0 (the view as it was)",
          fl and fl[0].default == 0.0)
    for eng in offered:
        go(eng)
        want = [c.key for c in controls_for(eng, w.engine_settings[eng])]
        if list(ins.rows) != want or ins.sec_tone.isVisibleTo(w) != takes_tone(eng, w.engine_settings[eng]):
            wrong.append(eng)
        if ins.error:
            wrong.append(f"{eng}: {ins.error}")
    check(f"all {len(offered)} views: the measured controls, and tone where it applies",
          not wrong, wrong[:4])
    go("Retinex Enhancement")
    before = pix()
    ins.rows["ret_scale_s"].set_value(2)
    ins.rows["ret_scale_s"].changed.emit("ret_scale_s", 2.0, True)
    ins.wait_render()
    check("a control writes the classic key and redraws",
          w.engine_settings["Retinex Enhancement"]["ret_scale_s"] == 2
          and np.abs(pix() - before).mean() > 0.25)
    ins.reset_view_settings()
    ins.wait_render()
    check("Reset puts the view's settings back", w.engine_settings["Retinex Enhancement"]["ret_scale_s"] == 8)

    print("\nViews that need a lower light, and the sharp curvature view")
    from helio.domain.view_controls import light_of
    w.engine_settings.setdefault("Specular Enhancement", {}).update(lx=0.0, ly=0.0, lz=1.0)
    w.set_view("Specular Enhancement")
    app.processEvents()
    lx_, ly_, lz_ = light_of(w.engine_settings["Specular Enhancement"], "Specular Enhancement")
    check("Specular Enhancement chosen with the light overhead lowers it to 45° (it adds "
          "nothing from straight above)", abs(lz_ - math.sin(math.radians(45))) < 0.01, (lx_, ly_, lz_))
    from helio.core import shading_engines as SE
    import numpy as _np
    rng_ = _np.random.default_rng(3)
    n_ = rng_.normal(0, 0.2, (60, 80, 3)).astype(_np.float32)
    n_[..., 2] = 1.0
    n_ /= _np.linalg.norm(n_, axis=2, keepdims=True)

    class _D:
        _curv_proxy = _np.zeros((30, 40), _np.float32)  # the old, smaller copy
        _curv_full = None
    out_ = SE._engine_surface_curvature(_D(), {}, 60, 80, dict(
        n_std=n_, depth_p=_np.zeros((60, 80), _np.float32), gain=1.0, use_proxy=True,
        _is_real_crop=False))
    check("Surface Curvature on screen is worked out at the size shown, not stretched up from "
          "a smaller copy", out_.shape == (60, 80) and float(out_.std()) > 0.05, out_.shape)

    print("\nLayers, display transforms, split view")
    go("Standard PBR")
    for key in ("Combined Colour", "Roughness Fusion", "Confidence Overlay", "Shading"):
        w.registry.trigger(f"view.layer:{key}")
        ins.wait_render()
    check("every layer draws", not ins.error and w.display.layer == "Shading")
    base_img = pix()
    ins.chk_flip_h.setChecked(True)
    ins.wait_render()
    check("Mirror left-right mirrors the picture, and says so", np.array_equal(pix(), base_img[:, ::-1])
          and ins.badge.isVisibleTo(w) and "H-Flip" in ins.badge.text())
    w.registry.trigger("view.reset_transforms")
    ins.wait_render()
    check("Reset display transforms puts it back", np.array_equal(pix(), base_img)
          and not ins.badge.isVisibleTo(w))
    ins.chk_invert_normals.setChecked(True)
    ins.wait_render()
    check("Invert normals changes the view", not np.array_equal(pix(), base_img))
    ins.chk_invert_normals.setChecked(False)
    ins.wait_render()
    w.registry.trigger("view.compare")
    ins.wait_render()
    check("Split view shows a second view right of the divider", ins.viewport.comparing())
    w.registry.trigger("view.compare")
    ins.wait_render()
    check("and goes away again", not ins.viewport.comparing())
    w.registry.trigger("view.rotate_right")
    check("Turn view right turns the display, not the data", w.canvas_rotation == 90
          and abs(ins.viewport.transform().m12()) > 0)
    w.registry.trigger("view.rotate_reset")

    print("\nProbe, loupe, histogram")
    ins.mode = "view"
    ins._probed(80.0, 120.0)
    t = ins.probe.text()
    check("the probe reads the pixel: position, height, normal and its slope, ±",
          "x 80" in t and "height" in t and "slope" in t and "±" in t, t)
    check("the loupe starts closed", not ins.sec_loupe.is_open())
    ins.sec_loupe.set_open(True)
    ins._clicked(80.0, 120.0)
    ins.wait_render()
    check("clicking holds the loupe on that spot, at full resolution",
          ins.loupe.held and not ins.loupe.picture.pixmap().isNull())
    ins._double_clicked()
    check("double-click lets it follow again", not ins.loupe.held)
    check("the histogram shows the view's numbers", "mean" in ins.hist.stats)

    print("\nSaved states and undo")
    ins.save_state("north light")
    st_light = dict(w.engine_settings["Standard PBR"])
    ins.dome._from_pos(cx - ins.dome.R * 0.7, cy, False)
    ins.wait_render()
    ins.load_state("north light")
    ins.wait_render()
    check("a saved state brings back the view and its light",
          w.view == "Standard PBR" and abs(w.engine_settings["Standard PBR"]["lx"] - st_light["lx"]) < 1e-6)
    sess_path = os.path.join(tmp, "views.helio")
    w.write_session(sess_path)
    from helio.io import session as S
    check("saved states and each view's light travel with the session",
          "north light" in S.read(sess_path).bookmarks
          and abs(S.read(sess_path).engine_settings["Standard PBR"]["lx"] - st_light["lx"]) < 1e-6)
    w._undo_poll(); w._undo_poll()
    g0 = w.engine_settings["Standard PBR"]["gain"]
    ins.rows["gain"].changed.emit("gain", 0.9, True)
    w._undo_poll(); w._undo_poll()
    w.registry.trigger("edit.undo")
    ins.wait_render()
    check("⌘Z undoes a setting once it has settled (the classic history)",
          abs(w.engine_settings["Standard PBR"]["gain"] - g0) < 1e-9)
    w.registry.trigger("edit.redo")
    check("⇧⌘Z redoes it", abs(w.engine_settings["Standard PBR"]["gain"] - 0.9) < 1e-9)

    print("\nBest raking, sweep")
    w.registry.trigger("view.best_raking")
    ins.wait_render()
    check("Best raking sets a captured light and says which", "Best contrast" in w.status_detail.text())
    w.registry.trigger("view.sweep")
    for _ in range(10):
        app.processEvents()
        time.sleep(0.02)
    moving = ins.sweep_timer.isActive()
    w.registry.trigger("view.sweep")
    check("A sweeps the light round and stops again", moving and not ins.sweep_timer.isActive())

    def sweep_for(seconds):
        w.registry.trigger("view.sweep")
        t = time.time()
        while time.time() - t < seconds:
            app.processEvents()
            time.sleep(0.01)
        during = (ins.az_row.value(), ins.height_row.value())
        w.registry.trigger("view.sweep")
        app.processEvents()
        return during

    ins.write_light(0.0, 0.0, 1.0)                 # overhead, as a new capture opens
    ins._show_angles(0.0, 0.0, 1.0)
    during = sweep_for(0.6)
    from helio.domain.view_controls import light_of
    after = light_of(w.engine_settings[w.engine], w.engine)
    check("a light overhead sweeps low, at 30°, where the surface changes "
          "(it used to circle at 90° and show nothing)", abs(during[1] - 30) <= 1, f"{during}")
    check("…and Stop puts it back overhead", after[2] > 0.999, f"{after}")
    from helio.domain.view_controls import from_angles
    low = from_angles(100.0, 40.0)
    ins.write_light(*low)
    ins._show_angles(*low)
    during = sweep_for(0.6)
    after = light_of(w.engine_settings[w.engine], w.engine)
    check("a light already low sweeps at its own height (40°) and is left where it stops",
          abs(during[1] - 40) <= 1 and abs(after[2] - low[2]) < 0.01
          and abs(after[0] - low[0]) + abs(after[1] - low[1]) > 0.01, f"{during} {after}")
    high = from_angles(200.0, 75.0)
    ins.write_light(*high)
    ins._show_angles(*high)
    during = sweep_for(0.6)
    check("a light placed high on purpose (75°) sweeps there, not thrown out to 30° "
          "(the limit was 60° until 2026-11-20)", abs(during[1] - 75) <= 1, f"{during}")
    w.registry.trigger("view.sweep")
    for _ in range(5):
        app.processEvents()
        time.sleep(0.02)
    placed = from_angles(45.0, 70.0)
    ins._dome_moved(*placed, False)                # the user places the light on the dome
    t = time.time()
    while time.time() - t < 0.5:
        app.processEvents()
        time.sleep(0.01)
    still = ins.sweep_timer.isActive()
    el_now = ins.height_row.value()
    w.registry.trigger("view.sweep")
    app.processEvents()
    check("placing the light on the dome mid-sweep keeps it sweeping, at the new height (70°)",
          still and abs(el_now - 70) <= 1, f"active={still} el={el_now}")
    ins.write_light(0.0, 0.0, 1.0)
    ins._show_angles(0.0, 0.0, 1.0)
    w.registry.trigger("view.sweep")
    app.processEvents()
    ins._height_moved("h", 80.0, True)            # the height slider, mid-sweep
    t = time.time()
    while time.time() - t < 0.4:
        app.processEvents()
        time.sleep(0.01)
    el_now = ins.height_row.value()
    w.registry.trigger("view.sweep")
    app.processEvents()
    after = light_of(w.engine_settings[w.engine], w.engine)
    check("…and the height slider likewise (80°), and Stop then leaves the light where it was "
          "placed, not back overhead", abs(el_now - 80) <= 1 and after[2] < 0.999,
          f"el={el_now} after={after}")

    print("\nFrames")
    ins.strip.setCurrentRow(2)
    ins._show_frame(2)
    app.processEvents()
    check("clicking a frame shows that photograph",
          ins.mode == "frame" and ins.viewport.path == w.project.images[2]
          and not ins.dome.isVisibleTo(w),
          (ins.mode, ins.viewport.path, w.project.current, ins.dome.isVisibleTo(w)))
    w.set_view("Standard PBR")
    check("choosing a view goes back to the solved surface", ins.mode == "view")

    print("\nDrawing speed on the real manuscript")
    w.open_paths([test_images(PASS, FAIL)])
    w.registry.trigger("process.solver:Uncalibrated PCA")
    w.registry.trigger("process.run")
    run_until_idle()
    w.project.update_settings(photo_tone="stored")
    w.set_view("Standard PBR")
    app.processEvents()
    times = []
    ins.wait_render()
    for k in range(20):
        ang = k / 20 * 6.283
        ins.dome._from_pos(cx + ins.dome.R * 0.7 * np.sin(ang),
                           cy - ins.dome.R * 0.7 * np.cos(ang), True)
        ins.wait_render()
        times.append(ins.last_ms)
    D = w.render_data
    check(f"each frame of a drag draws in under {FRAME_MS} ms at {D.pw} × {D.ph} "
          f"(slowest {max(times):.0f} ms)", max(times) < FRAME_MS, times)

    print("\nFrames reach the screen while the light moves")
    # Until 2026-10-15 a frame that finished after the next mouse move was
    # dropped, so a continuous drag showed nothing until the mouse stopped.
    shown = []
    orig_show = ins.viewport.show_array

    def counting(a, full_size=None):
        shown.append(a.shape[:2])
        orig_show(a, full_size=full_size)
    ins.viewport.show_array = counting
    if os.environ.get("HELIO_TEST_OLD_DROP"):          # the behaviour before the fix, to see this fail
        ins.renderer.same_picture = lambda *_a: False
    t_end, k = time.time() + 1.0, 0
    while time.time() < t_end:
        ang = k * 0.21
        k += 1
        ins.dome._from_pos(cx + ins.dome.R * 0.7 * np.sin(ang),
                           cy - ins.dome.R * 0.7 * np.cos(ang), True)
        t_next = time.time() + 0.004      # faster than a frame draws: every one overlaps a move
        while time.time() < t_next:
            app.processEvents()
            time.sleep(0.001)
    during = len(shown)
    ins.dome._from_pos(cx + ins.dome.R * 0.5, cy, False)          # let go
    ins.wait_render()
    app.processEvents()
    ins.viewport.show_array = orig_show
    check(f"a drag shows frames as it goes, not only when it stops ({during} in a second "
          f"of {k} moves)", during >= 10, during)
    check("…and the frame it ends on is the full preview size",
          shown and shown[-1] == (D.ph, D.pw), shown[-3:])

    print("\nZoom: scrolling, pinching, and it stays where you put it")
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QNativeGestureEvent, QPointingDevice, QWheelEvent
    D, vp = w.render_data, ins.viewport
    w.set_view("Standard PBR")
    ins.wait_render()
    app.processEvents()
    vp.fit()
    mid = QPointF(vp.viewport().width() / 2, vp.viewport().height() / 2)

    def wheel(px, ang, mods=Qt.KeyboardModifier.NoModifier):
        ev = QWheelEvent(mid, QPointF(vp.viewport().mapToGlobal(mid.toPoint())), QPoint(*px),
                         QPoint(*ang), Qt.MouseButton.NoButton, mods, Qt.ScrollPhase.NoScrollPhase,
                         False)
        QApplication.sendEvent(vp.viewport(), ev)
    z0 = vp.zoom()
    wheel((0, 0), (0, 120))
    check("a mouse wheel notch zooms in by the classic step, ×1.15", abs(vp.zoom() / z0 - 1.15) < 0.01,
          vp.zoom() / z0)
    z1 = vp.zoom()
    wheel((0, 60), (0, 120))                       # a trackpad: it reports pixels as well
    check("two fingers sliding on a trackpad zoom too", vp.zoom() > z1 * 1.1, vp.zoom() / z1)
    z2 = vp.zoom()
    wheel((0, 60), (0, 120), Qt.KeyboardModifier.ShiftModifier)
    wheel((60, 0), (120, 0))
    check("Shift + scroll, or sliding sideways, pans instead", abs(vp.zoom() - z2) < 1e-9)
    dev = QPointingDevice.primaryPointingDevice()
    for typ, val in ((Qt.NativeGestureType.BeginNativeGesture, 0.0),
                     (Qt.NativeGestureType.ZoomNativeGesture, 0.5),
                     (Qt.NativeGestureType.EndNativeGesture, 0.0)):
        QApplication.sendEvent(vp.viewport(), QNativeGestureEvent(
            typ, dev, 2, mid, mid, QPointF(vp.viewport().mapToGlobal(mid.toPoint())), val,
            QPointF(0, 0), 1))
    check("a pinch zooms", abs(vp.zoom() / z2 - 1.5) < 0.01, vp.zoom() / z2)

    vp.centerOn(D.w * 0.3, D.h * 0.3)

    def place():
        c = vp.mapToScene(vp.viewport().rect().center())
        return round(vp.zoom(), 4), round(c.x(), 1), round(c.y(), 1)
    here = place()
    w.set_view("Exaggerated Shading")
    ins.wait_render()
    app.processEvents()
    check("another view keeps the zoom and the place", place() == here, (place(), here))
    ins.set_layer("Shading")
    ins.wait_render()
    app.processEvents()
    check("another layer keeps them", place() == here, (place(), here))
    ins.set_layer("Combined Colour")
    ins.wait_render()
    ins._show_frame(1)
    app.processEvents()
    check("a photograph keeps them", place() == here, (place(), here))
    w.set_view("Standard PBR")
    ins.wait_render()
    app.processEvents()
    check("…and so does going back to the view", place() == here, (place(), here))
    vp.show_array(np.zeros((D.ph // 2 + 1, D.pw // 2, 3), np.uint8), full_size=(D.w, D.h))
    check("a reduced frame while the light moves keeps them (its size rounded differently)",
          place() == here, (place(), here))
    ins.render_view(final=True)
    ins.wait_render()

    print("\nThe photographs under the image")
    n = len(w.project.images)
    check("solved without lights (PCA), the tiles carry no light and only show the photograph",
          all(v is None for v in ins.strip.lights)
          and "double-click" not in ins.strip.item(2).toolTip())
    az8 = np.radians(np.arange(n) * 360.0 / n)
    L8 = np.stack([np.sin(az8) * 0.6, np.cos(az8) * 0.6, np.full(n, 0.8)], 1)
    w.project.set_lights(L8, "file:views.lp")          # new lights: solved again with them
    w.registry.trigger("process.run")
    run_until_idle()
    w.go_workspace("Inspect")
    ins.wait_render()
    ins.refresh()
    check("each tile carries its frame's light, as a dot on a small compass",
          len(ins.strip.lights) == n and all(v is not None for v in ins.strip.lights))
    check("…and says what clicking does", "double-click" in ins.strip.item(2).toolTip()
          and "Lit from" in ins.strip.item(2).toolTip(), ins.strip.item(2).toolTip())
    ins._show_frame(2)
    ins.light_from_frame(2)
    ins.wait_render()
    lv, got = ins._frame_light(2), ins._current_light()
    check("double-clicking one lights the view as that photograph was lit",
          ins.mode == "view" and np.allclose(lv, got, atol=1e-6), (lv, got))
    check("the layer choice sits straight under the view picker, above the light",
          ins.view_box.layout().itemAt(0).widget() is ins.sec_layer)

    print("\nThe Colour layer's own brightness")
    from helio.application.viewing import Display, screen_image
    D = w.render_data
    st = dict(w.engine_settings["Standard PBR"])
    col1 = screen_image("Standard PBR", D, st, Display(layer="Combined Colour", colour_gain=1.0))
    col2 = screen_image("Standard PBR", D, st, Display(layer="Combined Colour", colour_gain=1.5))
    clay1 = screen_image("Standard PBR", D, st, Display(layer="Shading", colour_gain=1.0))
    clay2 = screen_image("Standard PBR", D, st, Display(layer="Shading", colour_gain=1.5))
    ratio = float(col2.mean()) / max(float(col1.mean()), 1e-6)
    check(f"Colour brightness lifts the Colour layer (×{ratio:.2f} for ×1.5)", 1.3 < ratio < 1.52, ratio)
    check("…and leaves Clay exactly as it was", np.array_equal(clay1, clay2))
    check("the new app starts the Colour layer brighter (×1.5), the classic stays at ×1",
          w.display.colour_gain == 1.5 and Display().colour_gain == 1.0)
    ins.set_layer("Combined Colour")
    check("its slider shows with the Colour layer", ins.colour_gain_row.isVisibleTo(w)
          and abs(ins.colour_gain_row.value() - 1.5) < 1e-9)
    ins.set_layer("Shading")
    check("…and hides with Clay", not ins.colour_gain_row.isVisibleTo(w))
    ins.set_layer("Combined Colour")

    print("\nA 4K picture of the centre")
    ins.set_centre_mode("image")
    pic = os.path.join(tmp, "centre.png")
    got = w.save_centre_picture(pic)
    a4k = cv2.imread(pic) if got else None
    vw, vh = ins.viewport.viewport().width(), ins.viewport.viewport().height()
    check("the image as shown, 3840 pixels on its long side, the shape of the view",
          a4k is not None and max(a4k.shape[:2]) == 3840
          and abs(a4k.shape[1] / a4k.shape[0] - vw / vh) < 0.01, None if a4k is None else a4k.shape)
    check("…with nothing of the zoom controls in it (a plain picture)",
          a4k is not None and float(a4k.std()) > 5)

    print("\nThe 3D relief can reach the screen")
    import inspect as _inspect
    import helio_qt.app as _A
    src = _inspect.getsource(_A.main)
    check("the app asks for OpenGL 4.1 core before it makes its QApplication (else macOS "
          "shows the window, upside down, where the relief should be)",
          src.index("use_core_profile()") < src.index("QApplication(argv)"))
    from PySide6.QtGui import QSurfaceFormat
    _A.use_core_profile()
    f = QSurfaceFormat.defaultFormat()
    check("…which sets 4.1 core as every context's format",
          (f.majorVersion(), f.minorVersion()) == (4, 1)
          and f.profile() == QSurfaceFormat.OpenGLContextProfile.CoreProfile)

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
