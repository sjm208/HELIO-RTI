"""Publish's reports and figures (REP-002…006, KD-037, KD-040, KD-042).

Offscreen.  Solves the baseline fixture and exports the publication sheet,
the publication figures, the condition report, the orientation rose and
the A3 publication plate through Publish; then checks the shared functions directly: every panel the
picker offers is drawn, including the eight HELIO 2 could never draw
(KD-040); the condition statistics are levelled and labelled µm only with a
height scale (KD-042); the change map is current minus baseline; the rose
puts an eastward tilt at 90° and a northward one at 0° (KD-037).

    python tests/test_qt_publish_reports.py
"""
import importlib.util
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.abspath(__file__))
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
    import csv
    import cv2
    import numpy as np
    print("=" * 60)
    print("Publish: reports and figures")
    print("=" * 60)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
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
        "cb", os.path.join(ROOT, "docs", "pyside6-migration", "tools", "capture_baseline.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    z_gt, n_gt = cb.surface()
    L = cb.lights()
    src_dir = tempfile.mkdtemp(prefix="helio_qt_fixture_")
    for i, lv in enumerate(L):
        img = np.clip(n_gt @ lv, 0, 1) * 235 + 10
        cv2.imwrite(os.path.join(src_dir, f"img_{i:02d}.png"),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    w.open_paths([src_dir])
    p = w.project
    p.set_lights(L, "file:fixture.lp")
    p.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    idle()
    r = p.result
    check("solved", r is not None)
    p.scale_mm_per_px = 0.05
    w.go_workspace("Publish")
    pub = w.workspaces["Publish"]
    keys = ("sheet", "figures", "condition", "rose", "plate_a3")
    pub.selected = []
    for k in keys:
        pub._toggle(k)
    out = tempfile.mkdtemp(prefix="helio_qt_rep_")
    w.publish_to(out)
    idle(1800)
    rows = p.export_rows
    check("all five written", all(v[0] == "done" for v in rows.values()) and len(rows) == 5,
          {k: v[:2] for k, v in rows.items() if v[0] != "done"})

    def files(k, end=""):
        return [f for f in rows.get(k, ("", "", []))[2] if f.endswith(end)]

    sheet = cv2.imread(files("sheet", ".jpg")[0]) if files("sheet", ".jpg") else None
    check("the publication sheet is one large figure",
          sheet is not None and sheet.shape[0] > 3000, None if sheet is None else sheet.shape)
    figs = files("figures", ".tif")
    from helio.application.export_reports import FIGURE_ENGINES
    check(f"one figure per view ({len(figs)}) and a manifest",
          len(figs) >= len(FIGURE_ENGINES) - 2 and files("figures", "_figure_manifest.txt"))
    cond = files("condition", ".csv")
    stats = {row[0]: row for row in csv.reader(open(cond[0]))} if cond else {}
    check("condition report: levelled statistics in relative units without a height scale (KD-042)",
          stats.get("Sa", ["", "", ""])[2] == "relative" and "Levelling" in stats)
    check("the orientation rose is written", bool(files("rose", ".png")))
    plate_png = files("plate_a3", ".png")
    plate = cv2.imread(plate_png[0]) if plate_png else None
    check("the publication plate: an A3 PDF and a 300 dpi PNG (REP-010)",
          bool(files("plate_a3", ".pdf")) and plate is not None
          and abs(plate.shape[0] - 4962) <= 2 and abs(plate.shape[1] - 3507) <= 2,
          None if plate is None else plate.shape)

    print("\nThe shared functions")
    from helio.application.export_reports import (SHEET_PANELS, change_detection, condition_map,
                                                  publication_sheet)
    D = w.render_data
    tmp = tempfile.mkdtemp()

    def draw(eng, force_light=False):
        return np.full((D.ph, D.pw, 3), 120, np.uint8)
    res = publication_sheet(D, os.path.join(tmp, "s.jpg"), draw=draw,
                            selected=["21. Curvature", "29. False-Colour Depth", "26. Light Coverage"],
                            options={"metrology": False, "dpi": 50},
                            ss=dict(w.shared), lights=r.light_matrix, n_images=r.n_images)
    check("a picker numbered the HELIO 2 way still gets its panels (KD-040)",
          {"20. Curvature", "27. False-Colour Depth", "24. Light Coverage"} <= set(res["panels"]),
          res["panels"])
    res = publication_sheet(D, os.path.join(tmp, "all.jpg"), draw=draw, selected=SHEET_PANELS,
                            options={"dpi": 40}, ss=dict(w.shared), lights=r.light_matrix,
                            n_images=r.n_images)
    check(f"every panel offered is drawn ({len(res['panels'])} of {len(SHEET_PANELS)})",
          len(res["panels"]) == len(SHEET_PANELS))
    _img, c = condition_map(r.depth_poisson, r.albedo_rgb_full, tmp, "cal", z_mm_per_unit=0.002)
    st = {row[0]: row for row in csv.reader(open(c))}
    check("with a height scale the heights are in µm", st["Sa"][2] == "um")
    h, wd = 60, 80
    yy, xx = np.mgrid[0:h, 0:wd].astype(np.float32)
    tilt = 0.5 * xx + 0.2 * yy                                  # a pure plane
    _i, c2 = condition_map(tilt, None, tmp, "tilt")
    st2 = {row[0]: row for row in csv.reader(open(c2))}
    check("a tilted flat plate has no roughness once levelled", float(st2["Sa"][1]) < 1e-3, st2["Sa"])
    base = np.zeros((h, wd), np.float32)
    cur = base.copy()
    cur[20:30, 30:40] = 1.0
    _f, c3 = change_detection(cur, base, np.zeros((h, wd, 3), np.uint8), tmp, "chg")
    st3 = {row[0]: row for row in csv.reader(open(c3))}
    check("the change map is current minus baseline", abs(float(st3["Max_gain"][1]) - 1.0) < 1e-6)
    from helio.io.figures import orientation_histogram
    n = np.zeros((20, 20, 3), np.float32)
    n[..., 0], n[..., 2] = 0.3, 0.95
    e, f = orientation_histogram(n)
    k = int(np.argmax(f))
    east = np.degrees((e[k] + e[k + 1]) / 2)
    n[..., 0], n[..., 1] = 0.0, 0.3
    e, f = orientation_histogram(n)
    k = int(np.argmax(f))
    north = np.degrees((e[k] + e[k + 1]) / 2)
    check("the rose puts an eastward tilt at 90° and a northward one at 0° (KD-037)",
          abs(east - 90) <= 5 and north <= 5, (east, north))

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
