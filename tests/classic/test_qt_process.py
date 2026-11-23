"""Process in the new interface gives exactly the classic app's result (PROC-001).

Runs the Qt app offscreen (no window) and processes through the real job
manager and the shared pipeline:

* the asymmetric fixture of the scientific baseline, Linear Least Squares +
  Fast Poisson: normals and depth bit-identical to
  docs/pyside6-migration/baseline/scientific-baseline.npz;
* the real 8-frame manuscript, Uncalibrated PCA + Fast Poisson: SHA-256
  identical to the baseline's real-capture hashes;
* progress arrives in stage order; Cancel keeps the previous result; a
  failure is reported, not swallowed; nothing is left running at close.

    python tests/test_qt_process.py
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
BASE = os.path.join(ROOT, "docs", "pyside6-migration", "baseline")

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def sha(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def depth_single_core(r):
    """The result's depth integrated again with NumPy's single-core FFT, as
    the classic app did (helio.core.depth_integration's transforms swapped)."""
    import numpy as np
    import helio.core.depth_integration as DI
    from helio.application.pipeline import integrate_depth
    keep = (DI._fft2, DI._ifft2)
    DI._fft2, DI._ifft2 = np.fft.fft2, np.fft.ifft2
    try:
        return integrate_depth(r.normals_full, r.settings.depth_engine, spec_var=r.spec_var,
                               light_count_conf=r.light_count_conf, albedo_rgb=r.albedo_rgb_full)
    finally:
        DI._fft2, DI._ifft2 = keep


def main():
    import numpy as np
    import cv2
    print("=" * 60)
    print("Process in the new interface")
    print("=" * 60)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    warnings = []
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(a[1:3]))
    hw.QMessageBox.question = staticmethod(   # the next capture: don't save
        lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    def run_until_idle(limit=600.0):
        stages, fracs = [], []
        t0 = time.time()
        while w.project.busy and time.time() - t0 < limit:
            app.processEvents()
            p = w.project
            if p.stage and (not stages or stages[-1] != p.stage):
                stages.append(p.stage)
            fracs.append(p.fraction)
            time.sleep(0.01)
        app.processEvents()
        return stages, fracs

    print("\nThe baseline fixture, as a calibrated capture")
    spec = importlib.util.spec_from_file_location(
        "cb", os.path.join(ROOT, "docs", "pyside6-migration", "tools", "capture_baseline.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    z_gt, n_gt = cb.surface()
    L = cb.lights()
    tmp = tempfile.mkdtemp(prefix="helio_qt_fixture_")
    for i, lv in enumerate(L):
        img = np.clip(n_gt @ lv, 0, 1) * 235 + 10
        cv2.imwrite(os.path.join(tmp, f"img_{i:02d}.png"),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    w.open_paths([tmp])
    check("Process says why it cannot run before the lights are set",
          w.registry.reason("process.run") != "" and not w.registry.actions["process.run"].isEnabled())
    w.project.set_lights(L, "file:fixture.lp")
    w.project.update_settings(photo_tone="linear")      # the fixture writes radiance (FIX-010)
    check("with lights, Process is ready and is the next step",
          not w.registry.reason("process.run") and w.workspaces["Process"].next_step()[1] == "process.run")
    seen_stages = []                 # every progress signal, in delivery order (not sampled)
    w._job_sig.progress.connect(lambda jid, st, f, d: st and (not seen_stages or seen_stages[-1] != st)
                                and seen_stages.append(st))
    w.registry.trigger("process.run")
    check("processing starts on the job thread; the window stays live",
          w.project.busy == "solve" and w.current == "Process")
    check("Cancel is offered while it runs", w.registry.actions["process.cancel"].isEnabled())
    stages, fracs = run_until_idle()
    r = w.project.result
    check(f"stages in order: {seen_stages}",
          [s for s in seen_stages if s in ("Solving normals", "Integrating depth", "Building previews")]
          == ["Solving normals", "Integrating depth", "Building previews"])
    check("the progress bar never goes backwards", fracs == sorted(fracs))
    base = np.load(os.path.join(BASE, "scientific-baseline.npz"))
    check("normals bit-identical to the classic app's (baseline)",
          r is not None and np.array_equal(r.normals_full, base["normals__Linear_Least_Squares"]))
    # Depth is integrated on every core since 2026-10-20 (SciPy's pocketfft):
    # the classic's own single-core transform still reproduces the baseline
    # bit for bit, and the multi-core depth lies within a millionth of it.
    d_base = base["depth__Linear_Least_Squares"]
    check("depth: the single-core transform reproduces the classic app's bit for bit",
          r is not None and np.array_equal(depth_single_core(r), d_base))
    check("…and the multi-core depth is within a millionth of its range of it",
          r is not None and float(np.abs(r.depth_poisson.astype(np.float64) - d_base).max())
          <= 1e-6 * float(np.ptp(d_base)))
    check("the result is shown: Solved, previews, and Inspect is next",
          w.workspaces["Process"].title.text() == "Solved"
          and w.workspaces["Process"].previews.isVisibleTo(w)
          and w.workspaces["Process"].next_step()[1] == "view.ws:Inspect")
    check("Capture and Process are marked done", w.project.solved and w.project.lights_ready())
    pw = w.workspaces["Process"]
    check("the figures are explained in plain words, folded away until asked for",
          pw.btn_explain.isVisibleTo(w) and not pw.explain.isVisibleTo(w)
          and "Light conditioning" in pw.explain.text() and "Exploratory" in pw.explain.text())
    pw.btn_explain.click()
    check("…and shown when asked", pw.explain.isVisibleTo(w))
    pw.btn_explain.click()
    from helio.application.process_tools import precision_rating, readable
    sd = r.quality["sigma_deg"]
    a = np.zeros(sd.shape, np.float32)
    a[: sd.shape[0] // 2] = 200.0                       # the lower half: a black surround
    m = readable(a, sd)
    check("the black surround is left out of the figures",
          m is not None and m[: sd.shape[0] // 2].all() and not m[sd.shape[0] // 2:].any()
          and precision_rating(sd, m)[1] == float(np.median(sd[: sd.shape[0] // 2])))
    import re
    cap = pw.prev_caps["sigma"].text()
    top = float(re.search(r"0–(\d+)°", cap).group(1))
    check(f"the map's scale reaches past most of the object, so it is never solid yellow ({cap})",
          top >= 10 and top >= min(45, precision_rating(sd, readable(r.albedo_grey_full, sd))[2]))

    print("\nCancel keeps the previous result")
    first = w.project.result
    w.registry.trigger("process.run")
    w.registry.trigger("process.cancel")
    run_until_idle()
    check("the run stops and says so", not w.project.busy and "Cancelled" in w.project.message)
    check("the previous result is untouched", w.project.result is first)

    print("\nChanging a setting after solving")
    w.registry.trigger("process.depth:Frankot-Chellappa")
    check("the result is kept but marked out of date, and Process is offered again",
          w.project.settings_changed_since_solve()
          and w.workspaces["Process"].next_step() == ("Process 12 images again", "process.run"))
    w.registry.trigger("process.depth:Fast Poisson (FFT)")

    print("\nThe real manuscript, as the baseline solved it")
    w.open_paths([os.path.join(ROOT, "test images")])
    w.registry.trigger("process.solver:Uncalibrated PCA")
    check("Uncalibrated PCA can run without lights", not w.registry.reason("process.run"))
    # "As stored" is HELIO 2's reading of the photos: an old result must be
    # reproducible bit for bit (FIX-010 keeps that path).
    w.project.update_settings(photo_tone="stored")
    w.registry.trigger("process.run")
    run_until_idle()
    r = w.project.result
    real = json.load(open(os.path.join(BASE, "scientific-baseline.json")))["real_capture"]
    check("as stored: normals SHA-256 identical to the classic app's",
          r is not None and sha(r.normals_full) == real["normals_sha256"])
    d1 = depth_single_core(r) if r is not None else None
    check("as stored: depth SHA-256 identical to the classic app's (single-core transform)",
          r is not None and sha(d1) == real["depth_sha256"])
    check("…the multi-core depth within a millionth of its range of that",
          r is not None and float(np.abs(r.depth_poisson.astype(np.float64) - d1).max())
          <= 1e-6 * float(np.ptp(d1)))
    stored = r
    w.project.update_settings(photo_tone="srgb")
    w.registry.trigger("process.run")
    run_until_idle()
    r = w.project.result
    check("decoded to linear light (the default), the same photos give different normals",
          r is not None and stored is not None
          and not np.array_equal(r.normals_full, stored.normals_full))

    print("\nA failure is reported")
    bad = tempfile.mkdtemp(prefix="helio_qt_bad_")
    for i in range(3):
        with open(os.path.join(bad, f"broken_{i}.png"), "wb") as f:
            f.write(b"not an image")
    good_result = w.project.result
    w.open_paths([bad])
    w.registry.trigger("process.solver:Uncalibrated PCA")
    warnings.clear()
    w.registry.trigger("process.run")
    run_until_idle()
    check("the user is told why", warnings and "3 valid images" in str(warnings[-1]), warnings)
    check("and the window is usable again", not w.project.busy and w.project.result is None
          and good_result is not None)

    print("\nClose")
    w.close()
    app.processEvents()
    left = [t for t in __import__("threading").enumerate() if t.name.startswith("helio-job-")
            and t.is_alive()]
    check("no job thread is left", not left, left)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    main()
