"""Photographs paired with a light file's lights: by name, else in shot order.

A Cambridge University Library capture has photographs x-3.JPG … x-52.JPG
and an RTIBuilder .lp naming 01.jpg … 50.jpg.  Helio once listed the
photographs alphabetically (x-10, x-11, …, x-19, x-20, …, x-3, x-30, …) and
paired the lights by position, so light 01 went to x-10: most frames got
another frame's light, which degrades the normals, depth and every relit
export without saying so.  Now the names are matched when they are the
photographs', and otherwise light k goes to the k-th frame in natural order
(x-3, x-4, …, x-52) with a warning.

    python tests/test_lp_pairing.py
"""
import os
import sys
import tempfile

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


def lights(n):
    """n distinct unit vectors; light k's x is k/100 so a row says which light it is."""
    import numpy as np
    out = []
    for k in range(1, n + 1):
        x, y = k / 100.0, 0.3
        out.append([x, y, (1 - x * x - y * y) ** 0.5])
    return np.array(out, np.float32)


def write_lp(path, names, L):
    with open(path, "w") as f:
        f.write(f"{len(L)}\n")
        for nm, (x, y, z) in zip(names, L):
            f.write(f"{nm} {x:.8f} {y:.8f} {z:.8f}\n")


def capture(folder, frames, lp_names, L, lp="CambridgeLib.lp"):
    import cv2
    import numpy as np
    os.makedirs(folder, exist_ok=True)
    im = np.full((8, 8, 3), 128, np.uint8)
    for f in frames:
        cv2.imwrite(os.path.join(folder, f), im)
    write_lp(os.path.join(folder, lp), lp_names, L)
    return folder


def light_no(v):
    return int(round(float(v[0]) * 100))


def main():
    import numpy as np
    from helio.io.capture import captures_in, natural_key
    from helio.io.lp import pair_lights, read_lp_named
    print("=" * 60)
    print("Photographs paired with their lights")
    print("=" * 60)
    tmp = tempfile.mkdtemp(prefix="helio_pair_")
    stem = "PR-BASKERVILLE-C-RLC-K-00016-003-00006"
    frames = [f"{stem}-{k}.JPG" for k in range(3, 53)]             # -3 … -52
    L = lights(50)
    lp_names = [f"{k:02d}.jpg" for k in range(1, 51)]              # 01.jpg … 50.jpg
    cam = capture(os.path.join(tmp, "cambridge"), frames, lp_names, L)

    print("\n— a folder is listed in shot order")
    imgs = captures_in([cam])
    base = [os.path.basename(p) for p in imgs]
    check("x-3 first, x-52 last", base[0] == f"{stem}-3.JPG" and base[-1] == f"{stem}-52.JPG",
          str(base[:3]))
    check("every frame in numeric order", base == frames, str(base[:12]))
    check("natural_key: numbers by value, case ignored",
          sorted(["a-10.jpg", "A-9.JPG", "a-2.jpg", "b-1.jpg"], key=natural_key)
          == ["a-2.jpg", "A-9.JPG", "a-10.jpg", "b-1.jpg"])

    print("\n— an .lp whose names are none of the photographs'")
    names, got = read_lp_named(os.path.join(cam, "CambridgeLib.lp"))
    check("the names are read", names[:2] == ["01.jpg", "02.jpg"] and len(names) == 50)
    out, how, warn = pair_lights(names, got, imgs)
    check("paired in shot order", how == "order")
    check("-3 gets light 01, -4 light 02, -52 light 50",
          [light_no(out[i]) for i in (0, 1, 49)] == [1, 2, 50],
          str([light_no(out[i]) for i in (0, 1, 49)]))
    check("-10 gets light 08, not 01 (alphabetical)", light_no(out[base.index(f"{stem}-10.JPG")]) == 8)
    check("the user is told the names matched nothing", "match none" in warn, warn)
    alpha = sorted(imgs)                       # the old, alphabetical listing
    out2, _how, _w = pair_lights(names, got, alpha)
    check("the same pairing whatever order the photographs come in",
          all(light_no(out2[alpha.index(p)]) == light_no(out[imgs.index(p)]) for p in imgs))

    print("\n— an .lp naming the photographs, in any order")
    rng = np.random.default_rng(3)
    perm = rng.permutation(50)
    named = [frames[j] for j in perm]
    out, how, warn = pair_lights([n.lower() for n in named], L[perm], imgs)
    check("by name", how == "name" and not warn, warn)
    check("each photograph gets its own line", np.allclose(out, L, atol=1e-6))
    stems = [os.path.splitext(f)[0] + ".tif" for f in named]           # another extension
    out, how, _w = pair_lights(stems, L[perm], imgs)
    check("by name, ignoring the extension", how == "name" and np.allclose(out, L, atol=1e-6))
    out, how, warn = pair_lights(named[:45], L[perm][:45], imgs)
    miss = [i for i in range(50) if not np.isfinite(out[i]).all()]
    check("a photograph the file does not name is left without a light",
          how == "name" and len(miss) == 5 and "45 of the 50" in warn, warn)

    print("\n— an .lp without names")
    out, how, warn = pair_lights([None] * 50, L, alpha)
    check("shot order, quietly", how == "order" and not warn)
    check("light 1 with -3", light_no(out[alpha.index(os.path.join(cam, f"{stem}-3.JPG"))]) == 1)

    print("\n— File › Import RTI Project Folder")
    from helio.application.dome_tools import read_rti_project
    r = read_rti_project(cam)
    check("imports all 50, not 'matched 0'", len(r["paths"]) == 50, str(len(r["paths"])))
    check("in shot order", [os.path.basename(p) for p in r["paths"]] == frames)
    check("light 01 with -3, 50 with -52",
          light_no(r["lights"][0]) == 1 and light_no(r["lights"][-1]) == 50)
    check("says it paired by order", r["paired"] == "order" and r["unmatched"])
    good = capture(os.path.join(tmp, "named"), frames, named, L[perm], lp="lights.lp")
    r = read_rti_project(good)
    check("names that match are matched", r["paired"] == "name" and not r["unmatched"]
          and all(light_no(v) == frames.index(os.path.basename(p)) + 1
                  for p, v in zip(r["paths"], r["lights"])))

    print("\n— batch")
    from helio.application import batch as B
    log = []
    Lb, src, kind = B.calibration_for(cam, B._captures(cam), log=log, detect=lambda p: None)
    check("the folder's .lp, in shot order", kind == "file" and light_no(Lb[0]) == 1
          and light_no(Lb[7]) == 8, src)
    check("the log says the names matched nothing", any("match none" in x for x in log), str(log))
    Lb, _src, _k = B.calibration_for(good, B._captures(good), detect=lambda p: None)
    check("by name in batch too", np.allclose(Lb, L, atol=1e-6))

    print("\n— the window: open the folder, Import Light Positions")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    try:
        w.check_drift = lambda *a, **k: None
        w.find_grey_reference = lambda *a, **k: None
        check("opened", w.open_paths([cam], asked=True))
        p = w.project
        check("the frames are in shot order", [os.path.basename(x) for x in p.images] == frames)
        check("light file read", w.load_light_file(os.path.join(cam, "CambridgeLib.lp")))
        check("-3 has light 01, -10 light 08, -52 light 50",
              [light_no(p.lights[i]) for i in (0, 7, 49)] == [1, 8, 50],
              str([light_no(p.lights[i]) for i in (0, 7, 49)]))
        check("the user is told", "match none" in (getattr(w, "_flash", ("",))[0] or ""))
        w.open_paths([good], asked=True)
        w.load_light_file(os.path.join(good, "lights.lp"))
        check("a named .lp is matched by name", np.allclose(w.project.lights, L, atol=1e-6))
    finally:
        w.close()
        app.processEvents()

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
