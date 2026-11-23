"""The Working size control and the scale that follows the working size (interface, offscreen).

    python tests/test_work_scale_ui.py
"""
import dataclasses
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
os.environ["HELIO_PREFS"] = os.path.join(tempfile.mkdtemp(prefix="helio_prefs_"), "prefs.json")
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
    os.environ.pop("HELIO_WORK_SCALE", None)
    os.environ.pop("HELIO_MEMORY_GB", None)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    from helio.application import pipeline as P
    from helio.domain.settings import ProcessingSettings
    from helio.io import session as S
    from tools import make_golden as G
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    w = hw.MainWindow(appearance="dark")
    w.show()
    proc = w.workspaces["Process"]
    p = w.project

    print("The control")
    items = [(proc.work.itemData(i), proc.work.itemText(i)) for i in range(proc.work.count())]
    check("it offers full size, 75/50/35/25% and automatic", [v for v, _ in items] == [1.0, 0.75, 0.5, 0.35, 0.25, 0.0], items)
    check("it starts at full size", proc.work.currentData() == 1.0 and p.settings.work_scale == 1.0)
    proc.work.setCurrentIndex(proc.work.findData(0.5))
    check("choosing 50% sets the project's working size", p.settings.work_scale == 0.5)
    proc.work.setCurrentIndex(proc.work.findData(0.0))
    check("choosing Automatic sets 0", p.settings.work_scale == 0.0)
    p.update_settings(work_scale=0.75)
    app.processEvents()
    check("a change made elsewhere (a session, a profile) shows in the control", proc.work.currentData() == 0.75)
    p.update_settings(work_scale=0.6)
    app.processEvents()
    check("a size that is not in the list is shown, not lost", abs(proc.work.currentData() - 0.6) < 1e-9
          and proc.work.currentText() == "60%")
    p.update_settings(work_scale=1.0)
    check("the accessible name and the explanation are set", proc.work.accessibleName() == "Working size"
          and "memory" in proc.work.toolTip())

    print("The memory note")
    G.H, G.W = 360, 480
    paths, L = G.render_stack(tempfile.mkdtemp(prefix="helio_wsui_"))
    check("with no photographs it explains the setting", "reduced pixels" in proc.work_note.text(), proc.work_note.text())
    w.project.set_images(paths) if hasattr(w.project, "set_images") else w.load_images(paths)
    p.set_lights(L, "file:test.lp") if hasattr(p, "set_lights") else None
    app.processEvents()
    os.environ["HELIO_MEMORY_GB"] = "16"
    proc._note_cache = None
    p.update_settings(work_scale=1.0)
    app.processEvents()
    t = proc.work_note.text()
    check("it says what the capture needs and what is free", "needs about" in t and "GB" in t and "free" in t, t)
    os.environ["HELIO_WORK_SCALE"] = "0.5"
    proc._note_cache = None
    p.update_settings(work_scale=0.75)
    app.processEvents()
    check("an HELIO_WORK_SCALE override is mentioned", "HELIO_WORK_SCALE=0.5 overrides" in proc.work_note.text(), proc.work_note.text())
    os.environ.pop("HELIO_WORK_SCALE")
    os.environ.pop("HELIO_MEMORY_GB")
    p.update_settings(work_scale=1.0)

    print("The scale follows the working size")
    from helio_qt.state import Project
    q = Project()                      # on its own: no window to draw the stand-in results
    q.clear_scale()
    check("no scale: nothing to convert", q.photo_scale() is None)
    q.set_scale_photo(0.04)
    check("before any solve a scale is in the photographs' pixels", q.scale_mm_per_px == 0.04 and q.scale_basis == 1.0 and q.photo_scale() == 0.04)

    class FakeResult:
        def __init__(self, ws):
            self.work_scale = ws
    q.set_result(FakeResult(0.5), 1.0)
    check("a result at 50% re-bases it: pixels twice as big, 0.08 mm each", abs(q.scale_mm_per_px - 0.08) < 1e-12 and q.scale_basis == 0.5)
    check("the photographs' pixel is still 0.04 mm", abs(q.photo_scale() - 0.04) < 1e-12)
    q.set_result(FakeResult(0.25), 1.0)
    check("then 25%: 0.16 mm per pixel", abs(q.scale_mm_per_px - 0.16) < 1e-12 and abs(q.photo_scale() - 0.04) < 1e-12)
    q.set_result(FakeResult(1.0), 1.0)
    check("and back to full size: 0.04", abs(q.scale_mm_per_px - 0.04) < 1e-12 and q.scale_basis == 1.0)
    q.set_result(FakeResult(0.5), 1.0)
    q.set_scale_measured(0.1)
    check("a scale measured on a 50% picture is already its own pixels: the photograph's is 0.05",
          q.scale_mm_per_px == 0.1 and abs(q.photo_scale() - 0.05) < 1e-12)
    q.set_result(FakeResult(1.0), 1.0)
    check("and at full size it is 0.05", abs(q.scale_mm_per_px - 0.05) < 1e-12)
    q.set_result(FakeResult(0.5), 1.0)
    q.set_scale_photo(0.06)
    check("a photograph-pixel scale given while a 50% result is open is shown for its pixels: 0.12",
          abs(q.scale_mm_per_px - 0.12) < 1e-12 and abs(q.photo_scale() - 0.06) < 1e-12)
    q.result = None
    q.clear_scale()
    check("clearing forgets both", q.scale_mm_per_px is None and q.scale_basis == 1.0)

    print("A reduced solve through the window")
    w.project.set_images(paths) if hasattr(w.project, "set_images") else None
    p.set_scale_photo(0.04)
    p.update_settings(work_scale=0.5, photo_tone="linear")
    req = w.solve_request()
    check("the solve is given the photographs' scale", abs(req.mm_per_px - 0.04) < 1e-12 and req.settings.work_scale == 0.5, req.mm_per_px)
    r = P.solve(dataclasses.replace(req, lights=L, paths=paths))
    p.set_result(r, 1.0)
    check("the result is at 50% and the project's scale is now for its pixels (0.08)",
          r.work_scale == 0.5 and (r.h, r.w) == (180, 240) and abs(p.scale_mm_per_px - 0.08) < 1e-12)
    check("a second solve, at full size, brings it back (0.04)", True)
    p.update_settings(work_scale=1.0)
    r2 = P.solve(dataclasses.replace(w.solve_request(), lights=L, paths=paths))
    p.set_result(r2, 1.0)
    check("…it does", r2.work_scale == 1.0 and abs(p.scale_mm_per_px - 0.04) < 1e-12 and (r2.h, r2.w) == (360, 480))

    print("Sessions")
    p.update_settings(work_scale=0.5)
    p.set_result(r, 1.0)                                # a 50% result is open: scale 0.08
    out = os.path.join(tempfile.mkdtemp(), "reduced.helio")
    w.write_session(out)
    saved = json.load(open(out))
    check("the session records the photographs' pixel scale (0.04), not the reduced one", abs(saved["mm_per_px"] - 0.04) < 1e-12, saved.get("mm_per_px"))
    check("and the working size", saved.get("work_scale") == 0.5, saved.get("work_scale"))
    s_new = S.read(out)
    check("it reads back", s_new.work_scale == 0.5 and s_new.processing_settings(ProcessingSettings()).work_scale == 0.5)
    p.update_settings(work_scale=1.0)
    p.set_result(r2, 1.0)
    out2 = os.path.join(tempfile.mkdtemp(), "full.helio")
    w.write_session(out2)
    saved2 = json.load(open(out2))
    check("a full-size session has no work_scale key (byte for byte what it was)", "work_scale" not in saved2 and abs(saved2["mm_per_px"] - 0.04) < 1e-12, saved2.keys())
    check("a session from before this existed reads at full size", S.decode({}).work_scale == 1.0
          and S.decode({}).processing_settings(ProcessingSettings()).work_scale == 1.0)
    check("a session with automatic keeps it", S.decode({"work_scale": 0.0}).processing_settings(ProcessingSettings()).work_scale == 0.0)
    check("a nonsense size is read as full size", S.decode({"work_scale": 7}).processing_settings(ProcessingSettings()).work_scale == 1.0
          and S.decode({"work_scale": "x"}).processing_settings(ProcessingSettings()).work_scale == 1.0)
    w.load_session_file(out)
    app.processEvents()
    check("opening the 50% session sets the control and the photographs' scale",
          p.settings.work_scale == 0.5 and proc.work.currentData() == 0.5 and abs(p.photo_scale() - 0.04) < 1e-12
          and p.scale_basis == 1.0, (p.settings.work_scale, p.photo_scale(), p.scale_basis))

    print("Rig profiles")
    q = Project()
    q.set_scale_photo(0.05)
    q.set_result(FakeResult(0.5), 1.0)
    from helio.application import rigs
    prof = rigs.make(L, mm_per_px=q.photo_scale())
    check("a rig profile stores the photographs' scale (0.05), not the reduced pixel's (0.10)",
          abs(prof["mm_per_px"] - 0.05) < 1e-12 and abs(q.scale_mm_per_px - 0.10) < 1e-12)

    print("The problem report")
    from helio_qt import help as H
    p.settings = dataclasses.replace(p.settings, work_scale=0.5)
    p.result = r
    txt = H.report_text(w) if hasattr(H, "report_text") else None
    if txt is None:
        for name in dir(H):
            fn = getattr(H, name)
            if callable(fn) and name.startswith(("report", "problem", "diagnostic")):
                try:
                    txt = fn(w)
                    break
                except TypeError:
                    continue
    check("it names the working size", txt is not None and "work size" in txt and "50%" in txt, (txt or "")[-300:])

    w.close()
    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
