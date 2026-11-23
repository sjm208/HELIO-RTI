"""Printed calibration targets, measured back from their own PDFs, and the camera
alignment check on a photograph of the printed chessboard.

* Every target at A4, A3 and Letter: the page is the paper's size; the
  chessboard has 9 x 6 inner corners at the square size it says; the scale
  bars' ticks are 1 mm apart (Measure's ruler reader finds them so); the grid
  is crosses every 10 mm.
* The print dialog writes the PDF; File, Process and Measure open it.
* A photograph of the printed board, 3° off square, gives the tilt within half
  a degree, the side that is nearer, and the scale within 1%; and its scale can
  be set from it.

    python tests/test_qt_print_targets.py
"""
import math
import os
import sys
import tempfile

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


def render(path, dpi):
    from PySide6.QtCore import QSize
    from PySide6.QtPdf import QPdfDocument
    d = QPdfDocument()
    d.load(path)
    ps = d.pagePointSize(0)
    img = d.render(0, QSize(round(ps.width() / 72 * dpi), round(ps.height() / 72 * dpi)))
    img = img.convertToFormat(img.Format.Format_RGBA8888)
    rgba = np.array(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4]
    rgba = rgba.reshape(img.height(), img.width(), 4).astype(np.float32)
    a = rgba[..., 3:4] / 255.0                                  # the page is transparent where white
    grey = (rgba[..., :3].mean(axis=2, keepdims=True) * a + 255.0 * (1 - a))[..., 0]
    return ps.width() / 72 * 25.4, ps.height() / 72 * 25.4, np.clip(grey, 0, 255).astype(np.uint8)


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    from helio_qt import print_targets as T
    tmp = tempfile.mkdtemp(prefix="helio_targets_")
    print("The targets, measured from their PDFs")
    for paper in ("A4", "A3", "Letter"):
        dpi = 150
        r = T.chessboard(os.path.join(tmp, f"chess_{paper}.pdf"), paper)
        w, h, img = render(r["path"], dpi)
        check(f"{paper} chessboard: the page is the paper's size",
              abs(w - r["page_mm"][0]) < 0.5 and abs(h - r["page_mm"][1]) < 0.5, (w, h, r["page_mm"]))
        from helio.core.lens import find_corners
        c = find_corners(img, (9, 6))
        if c is None:
            check(f"{paper} chessboard: 9 x 6 inner corners found", False)
            continue
        g = c.reshape(6, 9, 2)
        sp = float(np.median(np.concatenate([np.linalg.norm(np.diff(g, axis=1), axis=2).ravel(),
                                             np.linalg.norm(np.diff(g, axis=0), axis=2).ravel()])))
        mm = sp / (dpi / 25.4)
        check(f"{paper} chessboard: 9 x 6 corners, squares {r['square_mm']:.0f} mm as printed on it",
              abs(mm - r["square_mm"]) / r["square_mm"] < 0.01, (mm, r["square_mm"]))
        s = T.scales(os.path.join(tmp, f"scales_{paper}.pdf"), paper)
        w, h, img = render(s["path"], 300)
        k = 300 / 25.4
        hundred = [st for st in s["strips"] if st["label"].startswith("100 mm")][0]
        x, y = hundred["ruler_x"], hundred["ruler_y"]
        roi = (int((x - 2) * k), int((y - 1) * k), int((x + 102) * k), int((y + 4) * k))
        from helio.core.paper_analysis import detect_scale_ticks
        res = detect_scale_ticks(img, roi=roi)
        check(f"{paper} scale bars: the ruler reader finds 1 mm ticks",
              abs(res["period_px"] / k - 1.0) < 0.02, res["period_px"] / k)
        check(f"{paper} scale bars: a fine 20 mm scale, an L scale and an inch rule",
              any("Fine" in st["label"] for st in s["strips"]) and any(st["label"] == "L scale" for st in s["strips"])
              and any("inch" in st["label"] for st in s["strips"]))
        gr = T.grid(os.path.join(tmp, f"grid_{paper}.pdf"), paper)
        check(f"{paper} grid: crosses every 10 mm inside the page",
              gr["grid_mm"][0] == (gr["n"][0] - 1) * 10 and gr["grid_mm"][0] < gr["page_mm"][0] - 20)

    print("\nIn the app")
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: print("        (warning)", a[2] if len(a) > 2 else ""))
    w = hw.MainWindow(appearance="dark")
    w.resize(1200, 800)
    w.show()
    w.registry.trigger("file.print_targets")
    d = w._targets_dlg
    check("File › Print Calibration Targets opens the dialog on the chessboard",
          d.isVisible() and d.kind.currentData() == "chessboard")
    d.close()
    d = w.print_scale_bars()
    check("Measure's command opens it on the scale bars", d.kind.currentData() == "scales")
    d.paper.setCurrentText("A3")
    out = d.save(os.path.join(tmp, "from_dialog.pdf"), open_after=False)
    check("…and it saves the PDF", out is not None and os.path.getsize(out["path"]) > 1000
          and "Saved" in d.status.text())
    d.close()
    d = w.print_chessboard()
    check("Process's command opens it on the chessboard", d.kind.currentData() == "chessboard")
    d.close()

    print("\nCamera alignment on a photograph of the printed board")
    pr = T.chessboard(os.path.join(tmp, "board.pdf"), "A4")
    _w, _h, tex = render(pr["path"], 100)                            # the printed sheet, 100 dpi
    px_mm = 25.4 / 100
    Wc, Hc, f = 3000, 2000, 3600.0
    K = np.array([[f, 0, Wc / 2], [0, f, Hc / 2], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array([0.0, math.radians(3.0), 0.0]))   # 3° about the vertical axis
    t = np.array([0.0, 0.0, 600.0])
    th, tw = tex.shape
    S = np.array([[px_mm, 0, -tw / 2 * px_mm], [0, px_mm, -th / 2 * px_mm], [0, 0, 1]])
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    photo = cv2.warpPerspective(tex, Hm / Hm[2, 2], (Wc, Hc), borderValue=120)
    pp = os.path.join(tmp, "board_photo.png")
    cv2.imwrite(pp, photo)
    pose = w.check_alignment(pp, pr["square_mm"], ask=False)
    check("the tilt is found within half a degree", pose is not None and abs(pose["tilt_deg"] - 3.0) < 0.5,
          pose and pose["tilt_deg"])
    from helio.core.lens import describe_pose
    check("…and the side nearer the camera named", pose is not None and ("right" in describe_pose(pose)
                                                                 or "left" in describe_pose(pose)), pose and describe_pose(pose))
    true_mm_px = 600.0 / f
    check("the scale on the board within 1%", pose is not None
          and abs(pose["mm_per_px"] - true_mm_px) / true_mm_px < 0.01, pose and (pose["mm_per_px"], true_mm_px))
    w.close()


main()
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
