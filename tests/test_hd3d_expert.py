"""The high-detail 3D window's own tools, and the wire on its own.

Offscreen.  Solves the baseline fixture, then checks: the mesh's wire over
the surface or on its own, from Inspect's boxes and from X in the view (the
boxes follow); the high-detail window's shading detail (4.5K, 8K, full,
never more than the solve has or the graphics card takes, and said on the
window); its mesh and wire; the light set by number and raking from a
corner, moving the one light everywhere; the second light; fine detail,
cavity and specular enhancement; colour by slope or curvature; and Reset.
The shaders themselves need OpenGL, which the offscreen platform has not:
they are compiled on the graphics card where there is one.

    python tests/test_hd3d_expert.py
"""
import importlib.util
import math
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
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
    import cv2
    import numpy as np
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QApplication
    print("=" * 60)
    print("The high-detail 3D window's tools, and the wire on its own")
    print("=" * 60)
    from helio_qt.relief3d import FRAG, KEYS, ReliefView
    check("the shader has the high-detail window's uniforms",
          all(u in FRAG for u in ("u_enhance", "u_cavity", "u_specboost", "u_light2", "u_colourby",
                                  "u_slopemax")) and "u_wire == 2" in FRAG)
    check("X is on the key card and kept by the view",
          any(k == "X" for _g, rows in KEYS for k, _w in rows)
          and int(Qt.Key.Key_X) in ReliefView.OWN_KEYS)

    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    def idle(limit=900.0):
        t0 = time.time()
        while w.project.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    spec = importlib.util.spec_from_file_location(
        "cb", os.path.join(ROOT, "tests", "fixtures", "synthetic_capture.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    _z, n_gt = cb.surface()
    L = cb.lights()
    src_dir = os.path.join(tempfile.mkdtemp(prefix="helio_qt_fixture_"), "capture")
    os.makedirs(src_dir)
    for i, lv in enumerate(L):
        img = np.clip(n_gt @ lv, 0, 1) * 235 + 10
        cv2.imwrite(os.path.join(src_dir, f"Coin_{i:02d}.png"),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    w.open_paths([src_dir])
    w.project.set_lights(L, "file:fixture.lp")
    w.registry.trigger("process.run")
    idle()
    check("solved", w.project.result is not None)

    print("\nThe wire, in Inspect")
    ins = w.workspaces["Inspect"]
    w.go_workspace("Inspect")
    ins.set_centre_mode("3d")
    app.processEvents()
    r = ins.relief
    check("Wire only waits until the mesh is shown", not ins.chk_relief_wire_only.isEnabled())
    ins.chk_relief_wire.setChecked(True)
    check("Show the mesh: its edges over the surface",
          r.wire_mode == "over" and r.wireframe and ins.chk_relief_wire_only.isEnabled())
    ins.chk_relief_wire_only.setChecked(True)
    check("…Wire only: the surface hidden", r.wire_mode == "only" and r.wireframe)
    ins.chk_relief_wire.setChecked(False)
    check("…the mesh box off turns it all off", r.wire_mode == "off" and not r.wireframe
          and not ins.chk_relief_wire_only.isEnabled())
    ins.chk_relief_wire_only.setChecked(False)
    r.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier))
    check("X in the view: over the surface, and the box follows",
          r.wire_mode == "over" and ins.chk_relief_wire.isChecked()
          and not ins.chk_relief_wire_only.isChecked())
    r.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier))
    check("…X again: wire only, both boxes ticked", r.wire_mode == "only"
          and ins.chk_relief_wire.isChecked() and ins.chk_relief_wire_only.isChecked()
          and "wire only" in r._readout_text)
    r.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier))
    check("…and again: off", r.wire_mode == "off" and not ins.chk_relief_wire.isChecked())
    check("Inspect's relief has none of the window's tools on",
          r.enhance == r.cavity == r.spec_boost == 0.0 and r.light2() is None
          and r.colour_by == "surface")

    print("\nThe high-detail window")
    w.registry.trigger("view3d.window")
    for _ in range(20):
        app.processEvents()
    w.wait_background()
    hd = w.hd3d
    hv = hd.view
    check("it opens with its Expert tools showing", hd.isVisible() and hv.data is not None
          and hd.expert.isVisibleTo(hd) and hd.btn_expert.isChecked())
    hd.btn_expert.setChecked(False)
    check("…which hide, and the view keeps the room", not hd.expert.isVisibleTo(hd))
    hd.btn_expert.setChecked(True)

    D = hd._of
    full = max(D.w, D.h)
    hd.detail_seg.set_current("4500")
    check("4.5K: 4,500 px on the long side, or the solve's own when smaller",
          hd.detail_side() == min(4500, full), (hd.detail_side(), full))
    hd.detail_seg.set_current("full")
    check("Full: the solve's own size", hd.detail_side() == full, (hd.detail_side(), full))
    keep = hv.max_texture
    hv.max_texture = max(2, full // 2)
    check("…never more than the graphics card takes", hd.detail_side() == full // 2)
    hv.max_texture = keep
    hd.detail_seg.chosen.emit("full")
    w.wait_background()
    app.processEvents()
    check("the window says what the shading is made at",
          "Normals at" in hd.detail_note.text() and "shaded at" in hd.note.text(),
          (hd.detail_note.text(), hd.note.text()))

    from helio.application.viewer3d import detail_textures

    class Big:
        w, h = 400, 300
        normals_full = np.dstack([np.zeros((300, 400)), np.zeros((300, 400)), np.ones((300, 400))])
        albedo_rgb_full = np.zeros((300, 400, 3), np.uint8)
    out = detail_textures(Big(), 100, 400)
    check("the detail textures come at the side asked, larger than the mesh",
          out is not None and out[2] == (400, 300) and out[0].shape == (300, 400, 3))

    print("\nGeometry and shading")
    from helio_qt.relief_window import ADAPTIVE_BASE, REFINE_PX, fine_heights
    check("the window opens with the adaptive mesh, shaded smoothly when not by the normals",
          hv.adaptive and hv.mesh_step == ADAPTIVE_BASE and hv.smooth_shading
          and hd.geom_seg.current() == "adaptive" and hd.refine_seg.isEnabled())
    check("…Inspect's relief keeps its own fixed mesh and flat mesh shading",
          not r.adaptive and not r.smooth_shading)
    check("…the readout says so (no OpenGL here: none to divide)", "mesh" in hv._readout_text,
          hv._readout_text)
    hd.refine_seg.chosen.emit("finest")
    check("Finest: edges of about 1.5 px on screen", hv.tess_px == REFINE_PX["finest"])
    hd.chk_fine.setChecked(False)
    check("Fine geometry off", not hv.fine_heights)
    hd.chk_fine.setChecked(True)
    ins.chk_relief_nmap.setChecked(False)
    hd._look = None
    hd._follow()
    check("the 3D panel's normals box no longer turns this window's off", hv.normal_mapped)
    hd.chk_nmap.setChecked(False)
    check("…its own box does", not hv.normal_mapped and "shaded by the heights" in hv._readout_text,
          hv._readout_text)
    hd.chk_nmap.setChecked(True)
    ins.chk_relief_nmap.setChecked(True)
    hv.tess_ok = True
    hv.set_mesh_step(ADAPTIVE_BASE)
    keep_d = hv.cam.dist
    hv.cam.dist = keep_d * 40
    lv_far = hv.centre_level()
    hv.cam.dist = keep_d * 0.02
    lv_near = hv.centre_level()
    hv.cam.dist = keep_d
    check("the adaptive mesh divides more as the camera comes closer, to its limit",
          lv_near > lv_far and lv_near <= hv.tess_max(), (lv_far, lv_near, hv.tess_max()))
    hv.tess_ok = False
    check("…with no divided mesh here, the densest fixed one instead", hv.mesh_used() == 1)
    hv.tess_ok = None

    class Deep:
        depth_vis_p = (np.random.default_rng(1).random((300, 400)) * 0.1
                       + np.linspace(0, 1, 400)[None, :]).astype(np.float32)
    fh = fine_heights(Deep(), (400, 300), (100, 75), 2.0)
    smooth = fine_heights(type("S", (), {"depth_vis_p": np.tile(np.linspace(0, 1, 400, dtype=np.float32),
                                                                (300, 1))})(), (400, 300), (100, 75), 1.0)
    check("fine heights: what the full-size depth has beyond the mesh's heights, in their units",
          fh.shape == (300, 400) and abs(float(fh.mean())) < 0.01 and float(fh.std()) > 0.005
          and float(np.abs(smooth[5:-5, 5:-5]).max()) < 0.01, (fh.std(), np.abs(smooth).max()))
    check("…none without the depth", fine_heights(object(), (40, 30), (10, 7), 1.0) is None)

    hd.geom_seg.chosen.emit("4")
    check("Mesh 1 : 4 in the window", hv.mesh_used() == 4 and not hv.adaptive
          and not hd.refine_seg.isEnabled())
    hd.geom_seg.chosen.emit("adaptive")
    check("…and back to adaptive", hv.adaptive and hv.mesh_step == ADAPTIVE_BASE)
    hd.geom_seg.chosen.emit("4")
    hd.wire_seg.chosen.emit("only")
    check("Wire only in the window", hv.wire_mode == "only")
    hv.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_X, Qt.KeyboardModifier.NoModifier))
    check("…X steps on, and the window's choice follows",
          hv.wire_mode == "off" and hd.wire_seg.current() == "off")
    check("…Inspect's relief keeps its own", r.wire_mode == "off")

    print("\nDisplay and views")
    m_persp = hv.mvp()
    hd.chk_ortho.setChecked(True)
    m_ortho = hv.mvp()
    check("Orthographic: no perspective (the matrix's last row is 0 0 0 1)",
          hv.ortho and np.allclose(m_ortho[3], [0, 0, 0, 1], atol=1e-9)
          and not np.allclose(m_persp[3], [0, 0, 0, 1], atol=1e-9))
    q = hv.pick(hv.width() / 2, hv.height() / 2)
    check("…and picking still finds the surface", q is not None)
    hd.chk_ortho.setChecked(False)
    hv.cam.az = 0.0
    hd.chk_head.setChecked(True)
    a = hv.lit().copy()
    hv.cam.az = 90.0
    b = hv.lit().copy()
    check("Headlight: the light turns with the camera, the set light left as it was",
          hv.headlight and not np.allclose(a, b) and abs(np.linalg.norm(a) - 1) < 1e-9
          and np.allclose(hv.light, hv.light / np.linalg.norm(hv.light)))
    hd.chk_head.setChecked(False)
    check("…off: the one light", np.allclose(hv.lit(), hv.light))
    hd.chk_spin.setChecked(True)
    az0 = hv.cam.az
    hd._spin_step()
    check("Spin slowly turns the surface", hd.spin_timer.isActive() and hv.cam.az != az0)
    hd.chk_spin.setChecked(False)
    check("…and stops", not hd.spin_timer.isActive())
    hv.cam.az, hv.cam.el = 30.0, 40.0
    hd.save_view()
    hd.chk_ortho.setChecked(True)
    hv.cam.az, hv.cam.el = 200.0, 70.0
    hd.save_view()
    check("Save this view: a button for each", len(hd.views) == 2 and len(hd.view_buttons) == 2
          and hd.view_buttons[1].text() == "View 2")
    hd.go_view(0)
    hv.cam._anim = (0.0, 1e-6) + hv.cam._anim[2:]
    hv.cam.step()
    check("…a view comes back: its camera and its perspective",
          abs(hv.cam.az - 30.0) < 0.5 and abs(hv.cam.el - 40.0) < 0.5 and not hv.ortho
          and not hd.chk_ortho.isChecked(), (hv.cam.az, hv.cam.el, hv.ortho))
    for _ in range(12):
        hd.save_view()
    from helio_qt.relief_window import MAX_VIEWS
    check("…kept to the last nine", len(hd.views) == MAX_VIEWS and len(hd.view_buttons) == MAX_VIEWS)
    hd.clear_views()
    check("Clear forgets them", not hd.views and not hd.view_buttons)

    print("\nThe light")
    hd.set_light(45.0, 12.0)
    app.processEvents()
    az = math.degrees(math.atan2(hv.light[0], hv.light[1])) % 360.0
    el = math.degrees(math.asin(hv.light[2]))
    lx, ly, lz = ins._current_light()
    check("Rake NE: from the north-east, 12° high, in the window",
          abs(az - 45.0) < 0.5 and abs(el - 12.0) < 0.5, (az, el))
    check("…and everywhere: the one light moves", abs(lx - hv.light[0]) < 0.05
          and abs(ly + hv.light[1]) < 0.05, ((lx, ly, lz), hv.light))
    check("…the numbers on the panel follow", abs(hd.az_row.value() - 45) < 1
          and abs(hd.el_row.value() - 12) < 1, (hd.az_row.value(), hd.el_row.value()))
    hd.az_row.set_value(200)
    hd.el_row.set_value(30)
    hd._light_typed(True)
    az = math.degrees(math.atan2(hv.light[0], hv.light[1])) % 360.0
    check("the light by number", abs(az - 200) < 0.5
          and abs(math.degrees(math.asin(hv.light[2])) - 30) < 0.5)
    hd.light2_seg.chosen.emit("180")
    l2 = hv.light2()
    check("a second light, opposite, at the same height",
          l2 is not None and abs(l2[0] + hv.light[0]) < 1e-6 and abs(l2[1] + hv.light[1]) < 1e-6
          and abs(l2[2] - hv.light[2]) < 1e-6, (l2, hv.light))
    hd.light2_seg.chosen.emit("90")
    l2 = hv.light2()
    check("…or 90° round", abs(float(np.dot(l2[:2], hv.light[:2]))) < 1e-6
          and abs(np.linalg.norm(l2) - 1) < 1e-6)

    print("\nShading: standard, exaggerated, hillshade")
    from helio_qt.relief3d import FRAG as _F
    check("the shader has Exaggerated Shading and the eight-direction hillshade",
          "unrotate(" in _F and "u_exag_scales" in _F and "u_hs_alt" in _F)
    check("Standard to begin with, the other models' settings hidden",
          hv.shading == "standard" and hd.shading_seg.current() == "standard"
          and not hd.exag_scales_row.isVisibleTo(hd) and not hd.hs_alt_row.isVisibleTo(hd))
    hd.shading_seg.chosen.emit("exaggerated")
    check("Exaggerated: its Scales and Finest scale shown, the second light set aside",
          hv.shading == "exaggerated" and hd.exag_scales_row.isVisibleTo(hd)
          and hd.exag_sigma_row.isVisibleTo(hd) and not hd.hs_alt_row.isVisibleTo(hd)
          and not hd.light2_seg.isEnabled() and "exaggerated shading" in hv._readout_text,
          hv._readout_text)
    hd.exag_scales_row.changed.emit("exag_scales", 6, True)
    hd.exag_sigma_row.changed.emit("exag_sigma", 1.5, True)
    check("…its settings reach the view", hv.exag_scales == 6 and hv.exag_sigma == 1.5)
    hd.shading_seg.chosen.emit("hillshade")
    hd.hs_alt_row.changed.emit("hs_alt", 30.0, True)
    check("Hillshade ×8: its light height shown and used",
          hv.shading == "hillshade" and hv.hs_alt == 30.0 and hd.hs_alt_row.isVisibleTo(hd)
          and not hd.exag_scales_row.isVisibleTo(hd) and "8 directions at 30°" in hv._readout_text,
          hv._readout_text)
    check("…Inspect's relief stays standard", r.shading == "standard")

    print("\nEnhance and colour")
    hd.enhance_row.changed.emit("enhance", 1.5, True)
    hd.cavity_row.changed.emit("cavity", 0.6, True)
    hd.spec_row.changed.emit("spec", 0.4, True)
    check("fine detail, cavity and specular enhancement reach the view",
          (hv.enhance, hv.cavity, hv.spec_boost) == (1.5, 0.6, 0.4))
    check("…the slope's colour waits for Colour by slope", not hd.slope_row.isEnabled())
    hd.colour_seg.chosen.emit("slope")
    hd.slope_row.changed.emit("slope_max", 30.0, True)
    check("colour by slope, steepest at 30°", hv.colour_by == "slope" and hv.slope_max == 30.0
          and hd.slope_row.isEnabled())
    hd.colour_seg.chosen.emit("curvature")
    check("…or by curvature", hv.colour_by == "curvature")
    try:
        hv.set_expert(nonsense=1)
        check("an unknown setting is refused", False)
    except KeyError:
        check("an unknown setting is refused", True)
    hd.reset_expert()
    check("Reset: standard shading, one light, nothing enhanced, colour by surface, no wire",
          hv.shading == "standard" and hd.shading_seg.current() == "standard"
          and hd.light2_seg.isEnabled()
          and hv.enhance == hv.cavity == hv.spec_boost == 0.0 and hv.light2() is None
          and hv.colour_by == "surface" and hv.wire_mode == "off"
          and hd.colour_seg.current() == "surface" and hd.light2_seg.current() == "off"
          and hd.enhance_row.value() == 0.0)
    check("…and Inspect's relief never had them", r.enhance == r.cavity == 0.0)

    hd.close()
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
