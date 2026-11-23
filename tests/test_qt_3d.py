"""The 3D model in the new interface: the classic viewer, fed and driven alike (GEO, EXM-004…008).

Offscreen, with the viewer process replaced by a stand-in that records what
it was given (no window opens).  Checks the four views' arrays: the whole
frame for the viewer to trim, or an already-trimmed region that says so
(KD-030); the quadric gone from Auto-Flatten; Inverted; the heatmap's
colours; an analytical view's render as the mesh colour; the settings and
the Inspect light reaching the open viewer; a turntable recorded and
written as a GIF; a clean frame; the sharpening pipeline (strength 1.0 the
original, KD-031); and the image adjustments.

    python tests/test_qt_3d.py
"""
import os
import sys
import tempfile
import threading
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


class FakeProc:
    def __init__(self, inp, state):
        self.inp, self.state, self.on = inp, state, True

    def is_alive(self):
        return self.on

    def terminate(self):
        self.on = False

    def join(self, timeout=None):
        pass


def main():
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QApplication
    from helio.application import viewer3d as V3
    print("=" * 60)
    print("3D model: the viewer, its settings and recordings")
    print("=" * 60)
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    import helio_qt.model3d as m3
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    launched = []

    def fake_launch(inp, state, spec=None):
        p = FakeProc(inp, state)
        launched.append(p)
        return p
    m3.V3.launch = fake_launch
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    p = w.project
    M = w.model3d

    def idle(limit=600):
        t0 = time.time()
        while p.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    from tools.make_golden import render_stack
    tmp = tempfile.mkdtemp(prefix="helio_3d_")
    paths, L = render_stack(tmp)
    w.open_paths([tmp])
    p.set_lights(L, "file:golden.lp")
    p.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    idle()
    r = p.result
    D = w.render_data
    check("solved", r is not None)
    check("the 3D actions are in View › 3D Model; the classic viewer is no longer offered",
          all(sid in w.registry.specs for sid in ("view3d.open:standard", "view3d.window",
                                                   "view3d.settings"))
          and not any(sid in w.registry.specs for sid in ("view3d.classic", "view3d.record:1",
                                                          "view3d.clean")))

    print("\nOpening")
    w.set_view("Standard PBR") if hasattr(w, "set_view") else None
    w.shared["micro_relief"] = 1.1
    p.scale_mm_per_px = 0.05
    from helio.application.viewer3d import CLAY as _CLAY
    pre = w.workspaces["Inspect"]
    other_clay = next(n for n in _CLAY if n != "Grey")
    pre.relief_clay.setCurrentText(other_clay)
    pre.relief_material.set_current("clay")
    w.model3d.clay = "Grey"                       # as if the window had never been told
    w.shared["use_custom_clay"] = 0
    w.model3d.open("standard")                    # the code that remains, driven directly
    app.processEvents()
    pr = launched[-1]
    check("the window opens with the 3D panel's clay colour and its Clay surface",
          w.model3d.clay == other_clay and w.shared["use_custom_clay"] == 1
          and pr.state["clay_blend"].value > 0.5, (w.model3d.clay, pr.state["clay_blend"].value))
    check("the viewer is given the whole frame and trims it itself",
          pr.inp.rgb.shape[:2] == (r.h, r.w) and pr.inp.skip_trim == 0
          and V3.ViewerInput is type(pr.inp))
    check("…coloured by the albedo for a non-analytical view",
          np.array_equal(pr.inp.rgb, D.albedo_rgb_full))
    st = pr.state
    check("the settings reach the viewer", abs(st["micro_relief"].value - 1.1) < 1e-6)
    ins3 = w.workspaces["Inspect"]
    want = ins3._classic_height(ins3.relief_rows["exaggeration"].value())
    check("…its height is the 3D panel's, in its own pixels "
          f"({st['height'].value:.0f} for ×{ins3.relief_rows['exaggeration'].value():g})",
          abs(st["height"].value - want) < 1e-3)
    check("…and the 3D panel shows while it is open, saying so",
          ins3.sec_relief.isVisibleTo(w) and "drive it" in ins3.relief_window_note.text())
    ins3.relief_rows["exaggeration"].changed.emit("exaggeration", 9.0, True)
    check("raising the panel's Height raises the separate window's",
          abs(st["height"].value - ins3._classic_height(9.0)) < 1e-3 and st["height"].value > want)
    ins3.shape_rows["warp_rem"].changed.emit("warp_rem", 0.5, True)
    check("a Shape slider in the panel reaches it too (and the exports)",
          abs(st["warp_rem"].value - 0.5) < 1e-6 and w.shared["warp_rem"] == 0.5)
    check("…and the measured scale, in its pixels", abs(st["mm_per_px"].value - 0.05) < 1e-6
          and st["img_w"].value == r.w and st["viewer_skip_trim"].value == 0)
    w.engine_settings[w.engine].update(lx=0.3, ly=-0.4, lz=0.866)
    M._tick()
    check("the Inspect light follows into the viewer",
          abs(st["lx"].value - 0.3) < 1e-6 and abs(st["ly"].value + 0.4) < 1e-6)
    M.show_settings()
    M.rows3d["micro_relief"].changed.emit("micro_relief", 1.7, True)
    check("a slider in 3D settings changes the open viewer and what exports read",
          abs(st["micro_relief"].value - 1.7) < 1e-6 and w.shared["micro_relief"] == 1.7)
    M.flip_boxes["fy"].setChecked(True)
    check("flipping Y turns the default -1 to 1", st["fy"].value == 1 and w.shared["fy"] == 1)
    M.flip_boxes["fy"].setChecked(False)
    M.chk_clay.setChecked(True)
    M.clay_combo.setCurrentText("Red")
    check("custom clay sends its colour", st["use_custom_clay"].value == 1
          and abs(st["clay_r"].value - 0.8) < 1e-6)

    print("\nThe settings window and the 3D panel agree")
    M.dialog.hide()
    ins3.shape_rows["shape_blend"].changed.emit("shape_blend", 0.2, True)
    ins3.tone3d_rows["viewer_gamma"].changed.emit("viewer_gamma", 1.4, True)
    M.show_settings()
    check("opening the settings shows what the panel changed since",
          abs(M.rows3d["shape_blend"].value() - 0.2) < 1e-6
          and abs(M.rows3d["viewer_gamma"].value() - 1.4) < 1e-6
          and abs(M.rows3d["height"].value() - st["height"].value) <= 5)
    M.rows3d["micro_relief"].changed.emit("micro_relief", 2.2, True)
    check("a Shape slider in the settings moves the panel's",
          abs(ins3.shape_rows["micro_relief"].value() - 2.2) < 1e-6)
    M.rows3d["viewer_gamma"].changed.emit("viewer_gamma", 0.8, True)
    check("…a tone slider too, and the relief here takes it",
          abs(ins3.tone3d_rows["viewer_gamma"].value() - 0.8) < 1e-6
          and (ins3.relief is None or abs(ins3.relief.tone["gamma"] - 0.8) < 1e-6))
    M.rows3d["ambient_str"].changed.emit("ambient_str", 0.4, True)
    check("…and the lighting", abs(ins3.relief_rows["ambient"].value() - 0.4) < 1e-6
          and abs(st["ambient_str"].value - 0.4) < 1e-6)
    M.rows3d["height"].changed.emit("height", ins3._classic_height(6.0), True)
    check("the settings' Height shows as the panel's (×6)",
          abs(ins3.relief_rows["exaggeration"].value() - 6.0) < 0.05)

    print("\nThe other views")
    w.shared["edge_trim"] = 5.0
    M.open("flattened")
    pr2 = launched[-1]
    x1, y1, x2, y2 = V3.trim_box(r.w, r.h, 5.0)
    check("opening another view closes the first", not pr.is_alive())
    check("Auto-Flatten: the trimmed region, marked as trimmed (no second trim: KD-030)",
          pr2.inp.d_hp.shape == (y2 - y1, x2 - x1) and pr2.inp.skip_trim == 1
          and pr2.state["viewer_skip_trim"].value == 1)
    d = pr2.inp.d_hp.astype(np.float64)
    hh, ww = d.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    xf, yf = (xx / ww).ravel(), (yy / hh).ravel()
    A = np.column_stack([xf ** 2, yf ** 2, xf * yf, xf, yf, np.ones_like(xf)])
    c, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None)
    resid = d.ravel() - A @ c
    check("…the quadric is gone (what is left has no quadric in it)",
          np.allclose(c[:5], 0, atol=1e-6) and abs(d.min()) < 1e-6 and abs(d.max() - 1) < 1e-6,
          c[:5])
    del resid
    M.open("inverted")
    pr3 = launched[-1]
    want = V3.invert01(D.depth_vis_p.astype(np.float32))
    check("Inverted: the depth turned over", np.allclose(pr3.inp.d_hp, want, atol=1e-6)
          and pr3.inp.skip_trim == 0)
    M.open("heatmap")
    pr4 = launched[-1]
    check("Heatmap: coloured by height over the trimmed region, in colour mode",
          pr4.inp.skip_trim == 1 and pr4.inp.rgb.shape[2] == 3
          and pr4.inp.rgb.reshape(-1, 3).std(axis=0).sum() > 10 and pr4.state["clay_blend"].value == 0.0)
    w.engine = "Specular Enhancement"
    M.open("standard")
    pr5 = launched[-1]
    from helio.application.render import composite, render
    st5 = dict(w.engine_settings["Specular Enhancement"])
    layer = w.display.layer if w.display.layer in ("Shading", "Combined Colour",
                                                   "Roughness Fusion") else "Combined Colour"
    want = composite("Specular Enhancement", render("Specular Enhancement", D, st5, r.h, r.w,
                                                    False, (0, 0, r.w, r.h)),
                     D, st5, r.h, r.w, False, (0, 0, r.w, r.h), view_mode=layer)
    check("an analytical view colours the mesh with its render", np.array_equal(pr5.inp.rgb, want))
    w.engine = "Standard PBR"

    print("\nRecording and a clean frame")
    out = tempfile.mkdtemp(prefix="helio_3d_out_")
    gif = os.path.join(out, "turn.gif")
    m3.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (gif, ""))
    sd = V3.spin_dir()
    os.makedirs(sd, exist_ok=True)
    for f in os.listdir(sd):
        os.remove(os.path.join(sd, f))
    for i in range(36):
        im = np.zeros((40, 60, 4), np.uint8)
        im[..., 3] = 255
        im[:, : (i % 60) + 1, 1] = 200
        cv2.imwrite(os.path.join(sd, f"frm_{i:04d}.png"), im)
    M.record(1)
    st = launched[-1].state
    check("recording asks the viewer for frames of a turntable",
          st["spin_capture"].value == 1 and st["spin_total"].value >= 36)
    st["spin_frames_done"].value = st["spin_total"].value          # the viewer finished
    w.wait_background()
    app.processEvents()
    from PIL import Image
    ok = os.path.exists(gif)
    check("…and writes them as a GIF", ok and Image.open(gif).n_frames == 36)
    png = os.path.join(out, "clean.png")
    m3.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (png, ""))
    src = os.path.join(out, "viewer_frame.png")
    cv2.imwrite(src, np.full((30, 30, 3), 90, np.uint8))

    def viewer():                                        # answers the clean-export flag
        for _ in range(100):
            if st["clean_export"].value == 1:
                with open(os.path.join(tempfile.gettempdir(), "helio_last_clean_export.txt"), "w") as f:
                    f.write(src)
                return
            time.sleep(0.02)
    threading.Thread(target=viewer, daemon=True).start()
    M.clean_frame()
    check("a clean frame is copied where asked", os.path.exists(png))

    print("\nSharpening")
    n0 = r.normals_full.copy()
    same, _ = V3.sharpen(n0, None, dict(strength=1.0, smooth=2))
    check("strength 1.0 leaves the normals as they are (KD-031)", np.allclose(same, n0, atol=1e-6))
    nb = cv2.GaussianBlur(n0, (0, 0), 2)
    two, _ = V3.sharpen(n0, None, dict(strength=2.0, smooth=2))
    d1 = np.abs(n0 - nb).mean()
    d2 = np.abs(two - cv2.GaussianBlur(two, (0, 0), 2)).mean()
    check("strength 2 roughly doubles the fine detail", 1.6 < d2 / d1 < 2.4, d2 / d1)
    M.sharpen.update(strength=2.0, smooth=2.0, usm_amt=1.0)
    M.apply_sharpen()
    w.wait_background()
    idle()
    check("Apply sharpens the solve's normals and depth",
          not np.array_equal(p.result.normals_full, n0)
          and not np.array_equal(p.result.depth_vis_p, r.depth_vis_p))
    check("…and reopens the open viewer with them", launched[-1].is_alive()
          and launched[-1] is not pr5)

    print("\nImage adjustments")
    ins = w.workspaces["Inspect"]
    w.go_workspace("Inspect")
    from helio.application.viewing import screen_image
    base = screen_image(w.engine, w.render_data, dict(w.engine_settings[w.engine]), w.display)
    ins.adjust_rows["global_whites"].changed.emit("global_whites", 1.5, True)
    brighter = screen_image(w.engine, w.render_data, dict(w.engine_settings[w.engine]), w.display)
    check("Whites brightens the view", brighter.mean() > base.mean() + 2)
    ins._adjust_reset()
    again = screen_image(w.engine, w.render_data, dict(w.engine_settings[w.engine]), w.display)
    check("Reset returns it exactly", np.array_equal(again, base))

    M.close()
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
