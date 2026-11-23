"""A light file beside the photographs is found and loaded when they are opened.

* one .lp in the photographs' folder, with one light per photograph: found;
* a .lp in the folder above (a capture's JPG/ subfolder): found;
* a .lp naming the photographs, among others that do not fit: that one;
* two that fit equally: none chosen, and the user told why;
* a .lp with the wrong number of lights and no matching names: not used;
* opening the folder in the app loads it, paired in shot order (x-3 before x-10).

    python tests/test_lp_autodetect.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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


def lights(n):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    L = np.stack([0.6 * np.cos(a), 0.6 * np.sin(a), np.full(n, 0.8)], 1)
    return L / np.linalg.norm(L, axis=1, keepdims=True)


def write_lp(path, L, names=None):
    with open(path, "w") as f:
        f.write(f"{len(L)}\n")
        for i, v in enumerate(L):
            nm = names[i] if names else f"{i + 1:02d}.jpg"
            f.write(f"{nm} {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def frames(folder, n, start=3):
    import cv2
    os.makedirs(folder, exist_ok=True)
    out = []
    for i in range(n):
        p = os.path.join(folder, f"obj-{start + i}.jpg")
        cv2.imwrite(p, np.full((40, 60, 3), 60 + 3 * i, np.uint8))
        out.append(p)
    return out


def main():
    from helio.io.lp import find_light_file
    tmp = tempfile.mkdtemp(prefix="helio_lp_")
    n = 12
    L = lights(n)

    print("Finding it")
    a = os.path.join(tmp, "a")
    imgs = frames(a, n)
    write_lp(os.path.join(a, "dome.lp"), L)
    path, why = find_light_file(imgs)
    check("one .lp with one light per photograph, beside them", path.endswith("dome.lp"), (path, why))

    b = os.path.join(tmp, "b")
    imgs_b = frames(os.path.join(b, "JPG"), n)
    write_lp(os.path.join(b, "rig.lp"), L)
    check("…or in the folder above", find_light_file(imgs_b)[0].endswith("rig.lp"))

    c = os.path.join(tmp, "c")
    imgs_c = frames(c, n)
    write_lp(os.path.join(c, "other.lp"), lights(7))
    write_lp(os.path.join(c, "named.lp"), L[::-1], [os.path.basename(p) for p in imgs_c[::-1]])
    check("the one naming the photographs, not one that does not fit",
          find_light_file(imgs_c)[0].endswith("named.lp"))

    d = os.path.join(tmp, "d")
    imgs_d = frames(d, n)
    write_lp(os.path.join(d, "one.lp"), L)
    write_lp(os.path.join(d, "two.lp"), L)
    path, why = find_light_file(imgs_d)
    check("two that fit equally: neither chosen, and why said", path == "" and "2 light files" in why, why)

    e = os.path.join(tmp, "e")
    imgs_e = frames(e, n)
    write_lp(os.path.join(e, "wrong.lp"), lights(n + 5))
    check("the wrong number of lights, names not the photographs': not used", find_light_file(imgs_e)[0] == "")

    print("\nIn the app")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)  # noqa: F841
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    ok = w.open_paths([a], asked=True)
    p = w.project
    got = np.asarray(p.lights, np.float32) if p.lights is not None else None
    check("opening the folder loads the light file", ok and got is not None and len(got) == n
          and "dome.lp" in p.light_source, p.light_source)
    from helio.io.capture import natural_key
    order = sorted(range(n), key=lambda i: natural_key(p.images[i]))
    check("…paired in shot order: light 1 with obj-3, light 8 with obj-10",
          got is not None and all(np.allclose(got[i], L[k], atol=1e-5) for k, i in enumerate(order)))
    w.open_paths([e], asked=True)
    check("a folder with no light file that fits leaves the lights unset", p.lights is None)
    w.wait_background()
    w.close()


main()
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
