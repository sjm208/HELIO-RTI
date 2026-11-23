"""Colour accuracy against a ColorChecker (MET-010).

* CIEDE2000 agrees with Sharma, Wu and Dalal's published test pairs;
* the reference colours survive the trip to sRGB and back (all but cyan,
  which lies outside sRGB);
* OpenCV's detector finds a drawn chart, and a faithful chart reads under
  0.5 ΔE₀₀ on every in-gamut patch, while a 6 % warm cast reads as
  noticeable;
* the CSV has every patch and the summary;
* Measure › Check Colour runs from the window and records the result.

    python tests/test_colour_check.py
"""
import csv
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import cv2

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


# Sharma, Wu and Dalal (2005), Table 1: (Lab 1, Lab 2, ΔE00).
SHARMA = [
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
]


def main():
    from helio.application import colour_check as C
    print("=" * 60)
    print("Colour accuracy")
    print("=" * 60)

    print("\nCIEDE2000")
    got = [float(C.delta_e_2000(a, b)) for a, b, _ in SHARMA]
    worst = max(abs(g - w) for g, (_, _, w) in zip(got, SHARMA))
    check(f"all {len(SHARMA)} of Sharma's test pairs, to 1e-4 (worst {worst:.1e})", worst < 1e-4)
    check("it is symmetric", abs(float(C.delta_e_2000(SHARMA[0][0], SHARMA[0][1]))
                                 - float(C.delta_e_2000(SHARMA[0][1], SHARMA[0][0]))) < 1e-9)

    print("\nThe reference")
    ref = np.array([r[1:] for r in C.REFERENCE])
    back = C.linear_to_lab_d50(C.srgb_to_linear(C.lab_d50_to_srgb(ref)))
    de = C.delta_e_2000(back, ref)
    check("every patch but cyan survives sRGB and back (under 0.05)",
          float(np.delete(de, 17).max()) < 0.05, de.round(3))
    check("…and cyan is the one outside sRGB", de[17] > 1.0)

    print("\nA chart in a photograph")
    img = C.synthetic_chart(900, noise=1.5, seed=3)
    found = C.find_chart(img)
    check("OpenCV's detector finds it", found is not None)
    res = C.evaluate(found["rgb"])
    ing = [r["de"] for r in res["rows"] if r["n"] != 18]
    check(f"a faithful chart: every in-gamut patch under 0.5 ΔE₀₀ (worst {max(ing):.2f}), "
          f"average {res['mean']:.2f}", max(ing) < 0.5 and res["mean"] < 0.6)
    rgb = np.clip(C.lab_d50_to_srgb(ref) * np.array([1.06, 1.0, 0.92]), 0, 1)
    warm = C.evaluate(C.find_chart(C.synthetic_chart(900, rgb=rgb, noise=1.5))["rgb"])
    check(f"a 6 % warm cast reads as noticeable (average {warm['mean']:.2f})",
          warm["mean"] > 3 and warm["verdict"].startswith("Noticeable"))
    check("a blank picture has no chart", C.find_chart(np.full((600, 800, 3), 128, np.uint8)) is None)

    tmp = tempfile.mkdtemp(prefix="helio_colour_")
    p = os.path.join(tmp, "chart.png")
    cv2.imwrite(p, img)
    out = C.write_csv(C.check_image(p), os.path.join(tmp, "check.csv"))
    rows = list(csv.reader(open(out)))
    check("the CSV: a header, 24 patches, then the summary",
          rows[0][0] == "patch" and len([r for r in rows[1:25] if r]) == 24
          and any(r and r[0] == "mean dE00" for r in rows))

    print("\nIn the window")
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    from helio_qt.quick_exports import ColourCheckDialog
    w = hw.MainWindow(appearance="dark")
    w.project.set_images([p])
    shown = []
    orig = ColourCheckDialog.exec
    ColourCheckDialog.exec = lambda self: shown.append(self) or 1
    try:
        r = w.colour_check(p)
    finally:
        ColourCheckDialog.exec = orig
    check("Measure › Check Colour finds the chart, shows the patches and remembers the result",
          r is not None and shown and w.project.colour_check["mean"] == r["mean"])
    csv2 = shown[0].save_csv(os.path.join(tmp, "from_sheet.csv"))
    check("…and saves the CSV from its sheet", os.path.isfile(csv2))
    ins = w.workspaces["Inspect"]
    ins._measure_summary()
    check("the right column says what was measured",
          "ΔE₀₀" in ins.m_rows["colour"].text() and "not set" in ins.m_rows["scale"].text())
    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
