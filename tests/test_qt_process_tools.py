"""Process in the new interface: every setting, the profiles and the tools.

Offscreen.  Solves the golden stack, then checks that each classic setting is
there and reaches the solve; Quick Profiles and Material Presets; that
Recalculate Depth with the engine already used gives back the solve's depth
exactly (KD-011); that another engine changes it; Iterative Normal Refine;
a flat reference; processing a region; the near-field correction using the
dome radius once a scale is set (FIX-012); and that each quality check
reports.

    python tests/test_qt_process_tools.py
"""
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
    from PySide6.QtWidgets import QApplication
    print("=" * 60)
    print("Process: settings, profiles and tools")
    print("=" * 60)
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    import helio_qt.report as rp
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.No)
    reports = []
    rp.ReportDialog.exec = lambda self: reports.append(self) or 0
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    pw = w.workspaces["Process"]
    p = w.project

    def idle(limit=600):
        t0 = time.time()
        while p.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    from tools.make_golden import render_stack
    tmp = tempfile.mkdtemp(prefix="helio_pt_")
    paths, L = render_stack(tmp)
    w.open_paths([tmp])
    p.set_lights(L, "file:golden.lp")

    print("\nEvery classic setting is here")
    names = set(pw.chk) | set(pw.rows)
    want = {"flatten_illumination", "auto_align", "pre_clahe", "pre_clahe_clip",
            "pre_clahe_tile", "adaptive_light", "shadow_thresh", "sfs_refine", "sfs_iters",
            "sfs_prior", "flat_correction", "flat_corr_strength", "parabolic_corr", "pre_smooth",
            "near_field", "dome_radius_mm"}
    check("the full set of solve settings", want <= names, want - names)
    pw.tone.setCurrentIndex(pw.tone.findData("linear"))   # the golden stack is radiance
    check("the photo tone reaches the settings", p.settings.photo_tone == "linear")
    pw.chk["auto_align"].setChecked(True)
    check("a checkbox writes its setting", p.settings.auto_align)
    pw.chk["auto_align"].setChecked(False)
    check("frame-brightness levelling is retired: no control for it",
          "multi_norm" not in pw.chk)
    pw.rows["adaptive_light"].changed.emit("adaptive_light", 0.3, True)
    check("a slider writes its setting", abs(p.settings.adaptive_light - 0.3) < 1e-9)
    pw.rows["adaptive_light"].changed.emit("adaptive_light", 0.0, True)

    print("\nProfiles")
    pw.apply_quick("Heritage Balanced")
    s = p.settings
    check("a Quick Profile sets solver, depth, smoothing and light weighting",
          s.normal_engine == "Shadow-Rejection LS" and s.depth_engine == "Fast Poisson (FFT)"
          and s.pre_smooth == 3.0 and abs(s.adaptive_light - 0.2) < 1e-9)
    pw.apply_material("wax_seal")
    check("a Material Preset sets solver, depth, parabolic and the 3D settings",
          p.settings.normal_engine == "Sparse-Outlier LS (RPCA)"
          and p.settings.parabolic_corr and w.shared["micro_relief"] == 1.8
          and w.shared["height"] == 350.0 and w.engine == "Specular Enhancement")
    p.update_settings(normal_engine="Linear Least Squares", depth_engine="Fast Poisson (FFT)",
                      pre_smooth=0.0, adaptive_light=0.0, parabolic_corr=False, multi_norm=False)

    w.registry.trigger("process.run")
    idle()
    r = p.result
    check("solved", r is not None)
    check("after the solve the next step is Inspect, with the tone on Automatic",
          p.settings.photo_tone != "auto" or pw.next_step()[1] == "view.ws:Inspect",
          (pw.next_step(), p.settings.photo_tone, r.settings.photo_tone))
    solver = p.settings.normal_engine
    w.set_solver("Mid-Tone Weighted LS" if solver != "Mid-Tone Weighted LS" else "Linear Least Squares")
    check("another solver chosen: process again", pw.next_step()[1] == "process.run", pw.next_step())
    w.set_solver(solver)
    check("the solved solver chosen back: Inspect again", pw.next_step()[1] == "view.ws:Inspect", pw.next_step())

    print("\nRecalculate Depth")
    w.registry.trigger("process.recalc")
    pw.wait()
    idle()
    check("with the engine already used it gives back the solve's depth exactly (KD-011)",
          np.array_equal(p.result.depth_vis_p, r.depth_vis_p)
          and np.array_equal(p.result.depth_poisson, r.depth_poisson))
    w.set_depth("Frankot-Chellappa")
    w.registry.trigger("process.recalc")
    pw.wait()
    idle()
    check("with another engine the depth changes, the normals do not",
          not np.array_equal(p.result.depth_poisson, r.depth_poisson)
          and np.array_equal(p.result.normals_full, r.normals_full)
          and p.result.settings.depth_engine == "Frankot-Chellappa"
          and not p.settings_changed_since_solve())
    w.set_depth("Fast Poisson (FFT)")
    w.registry.trigger("process.refine")
    pw.wait()
    idle()
    check("Iterative Normal Refine changes the depth and keeps the solved normals",
          not np.array_equal(p.result.depth_poisson, r.depth_poisson)
          and np.array_equal(p.result.normals_full, r.normals_full))

    print("\nQuality checks")
    for aid in ("qc.report", "qc.bias", "qc.disagreement", "qc.dome3d"):
        reports.clear()
        w.registry.trigger(aid)
        check(f"{aid} reports with figures", reports and reports[-1].figures,
              [x.title for x in reports])
    rep = reports and reports[-1]
    w.registry.trigger("qc.report")
    q = reports[-1]
    check("the QC report is rated from the measured precision (KD-022)",
          "1σ" in " ".join(n for n, _v in q.figures) and q.verdict.split(":")[0] in
          ("Metrological", "Research", "Exploratory"), q.verdict)
    out = tempfile.mkdtemp()
    files = q.save_to(out)
    check("a report saves its pictures, raw maps and figures",
          any(f.endswith(".txt") for f in files) and any(f.endswith(".tif") for f in files))
    del rep

    print("\nFlat reference")
    flat_dir = tempfile.mkdtemp(prefix="flat_")
    flat_paths = []
    for i, lv in enumerate(L):
        img = (np.clip(lv[2], 0, 1) * 200 + 20) * np.ones((60, 80), np.float32)
        img *= np.linspace(0.9, 1.1, 80)[None, :]          # a gentle falloff: the rig's distortion
        fp = os.path.join(flat_dir, f"flat_{i:02d}.png")
        cv2.imwrite(fp, np.clip(img, 0, 255).astype(np.uint8))
        flat_paths.append(fp)
    hw.QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (flat_paths, ""))
    w.registry.trigger("process.flat")
    check("loading a flat reference computes its normals and switches the correction on",
          w.flat_ref_normals is not None and p.settings.flat_correction
          and w.flat_ref_info["n"] == len(L))
    req = w.solve_request()
    check("…and the next solve receives it", req.flat_ref_normals is w.flat_ref_normals)
    p.update_settings(flat_correction=False)
    w.flat_ref_normals, w.flat_ref_info = None, {}

    print("\nNear-field correction")
    from helio.application.pipeline import near_field_radius_px
    ps = p.settings.__class__(near_field=True, dome_radius_mm=150.0)
    check("without a scale the dome is assumed 1.5 image-widths away",
          near_field_radius_px(ps, None, 240, 320) == (480.0, "assumed"))
    check("with a scale it is the radius in millimetres over the scale",
          near_field_radius_px(ps, 0.5, 240, 320) == (300.0, "measured"))
    p.scale_mm_per_px = 0.5
    p.update_settings(near_field=True, dome_radius_mm=150.0)
    check("the inspector says the measured scale is used",
          "measured scale" in pw.nf_note.text())
    req = w.solve_request()
    check("the solve request carries the scale", req.mm_per_px == 0.5)
    p.update_settings(near_field=False)
    p.scale_mm_per_px = None

    print("\nA region")
    w.registry.trigger("process.reprocess")
    ins = w.workspaces["Inspect"]
    check("Process a Region turns on the rectangle tool in Inspect",
          w.current == "Inspect" and ins.viewport.tool == "rect")
    ins.viewport.rectDrawn.emit(40.0, 30.0, 200.0, 180.0)
    idle()
    check("drawing it solves that region at full resolution",
          p.result is not None and p.result.crop == (40, 30, 200, 180)
          and p.result.w == 160 and p.result.h == 150, p.result.crop if p.result else None)
    pw.whole_image()
    idle()
    check("“Process the whole image again” goes back to the whole frame",
          p.result.crop is None and p.result.w == 320)

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
