"""HELIO Pro 3 checking itself: `HELIO Pro 3 --selftest [folder]`.

The tests prove the source tree; this proves a built copy, run with that
copy's own interpreter and libraries.  No window opens and nothing of the
user's is touched: settings and preferences go to a scratch folder.  It makes
a small capture of a known surface, solves it through the window as a person
would, draws the views that lean on optional libraries, writes the exports
most likely to break in a bundle (the ones that use vendored files, data
files, OpenGL child processes, USD and OpenCV's chart finder), and writes
selftest.json.  Exit status 0 means every step passed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback


def _surface(h=96, w=128):
    import numpy as np
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = 9.0 * np.exp(-(((xx - 0.35 * w) / 12.0) ** 2 + ((yy - 0.4 * h) / 10.0) ** 2))
    z += 5.0 * np.exp(-(((xx - 0.7 * w) / 16.0) ** 2 + ((yy - 0.65 * h) / 13.0) ** 2))
    z += 0.02 * xx
    zy, zx = np.gradient(z)
    n = np.dstack([-zx, zy, np.ones_like(zx)])          # +Y up, as the solve returns
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def _lights(n=12):
    import numpy as np
    az = np.radians(np.arange(n) * 360.0 / n)
    el = np.radians(np.linspace(25, 55, n))
    return np.stack([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)], 1)


def run(out_dir: str = "") -> int:
    out_dir = os.path.abspath(out_dir or tempfile.mkdtemp(prefix="helio3_selftest_"))
    os.makedirs(out_dir, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix="helio3_selftest_scratch_")
    os.environ["QT_QPA_PLATFORM"] = os.environ.get("HELIO_SELFTEST_PLATFORM", "offscreen")
    os.environ["HELIO_QT_SETTINGS"] = os.path.join(scratch, "prefs.ini")
    os.environ["HELIO_PREFS"] = os.path.join(scratch, "prefs.json")
    steps: list[dict] = []

    def step(name, fn):
        t0 = time.time()
        try:
            detail = fn()
            steps.append(dict(step=name, ok=True, detail=str(detail or ""),
                              seconds=round(time.time() - t0, 2)))
        except Exception as e:
            steps.append(dict(step=name, ok=False, detail=f"{type(e).__name__}: {e}",
                              trace=traceback.format_exc(limit=6),
                              seconds=round(time.time() - t0, 2)))
        print(("  ok   " if steps[-1]["ok"] else "  FAIL ") + name
              + (f"  ({steps[-1]['detail']})" if steps[-1]["detail"] else ""), flush=True)

    print(f"Helio self-test → {out_dir}", flush=True)

    def libraries():
        import cv2
        import matplotlib
        import numpy
        import PIL
        import qrcode
        import scipy
        import skimage
        import tifffile
        import trimesh
        from PySide6.QtOpenGLWidgets import QOpenGLWidget  # noqa: F401
        assert hasattr(cv2, "mcc"), "OpenCV has no chart finder (mcc)"
        return (f"numpy {numpy.__version__}, scipy {scipy.__version__}, OpenCV {cv2.__version__}, "
                f"matplotlib {matplotlib.__version__}, Pillow {PIL.__version__}, "
                f"scikit-image {skimage.__version__}, trimesh {trimesh.__version__}, "
                f"tifffile {tifffile.__version__}, qrcode")
    step("libraries load", libraries)

    def kernels():
        # The Rust kernels: present and right, or (when the build meant to include them,
        # HELIO_REQUIRE_RUST=1) a failure - a bundle that silently lacks them is just slower.
        import numpy as np
        try:
            import helio_core_rs as rs
            rs.lls_normals              # the crate folder alone imports as an empty package
        except (ImportError, AttributeError):
            if os.environ.get("HELIO_REQUIRE_RUST") == "1":
                raise RuntimeError("helio_core_rs is not in this build (it was built, but not bundled)")
            return "NumPy kernels (helio_core_rs is not in this build)"
        L = np.array([[0.3, 0.1, 0.95], [-0.4, 0.2, 0.89], [0.1, -0.5, 0.86], [0.5, 0.4, 0.77]])
        n = np.array([0.2, -0.1, 0.97])
        n /= np.linalg.norm(n)
        I = np.maximum(L @ n, 0)[:, None].astype(np.float32).repeat(5, axis=1)
        got = rs.lls_normals(np.linalg.pinv(L), np.ascontiguousarray(I))
        if not np.allclose(got, n, atol=1e-5):
            raise RuntimeError(f"helio_core_rs gives wrong normals: {got[0]} for {n}")
        from helio.core._native import describe
        return describe()
    step("Rust kernels load and agree with NumPy", kernels)

    def optional():
        found = []
        for mod in ("rvt", "pxr", "OpenGL", "rawpy"):
            try:
                __import__(mod)
                found.append(mod)
            except Exception as e:
                found.append(f"{mod} MISSING ({type(e).__name__})")
        bad = [f for f in found if "MISSING" in f]
        if bad:
            raise RuntimeError(", ".join(bad))
        return ", ".join(found)
    step("optional libraries load (RVT, OpenUSD, OpenGL, LibRaw)", optional)

    def vendored():
        from helio.vendor_assets import read_bytes, vendor_dir
        n = len(read_bytes("three.min.js")) + len(read_bytes("openseadragon.min.js"))
        return f"{n // 1024} KB from {vendor_dir()}"
    step("vendored viewer libraries are in the bundle", vendored)

    import numpy as np
    from PySide6.QtWidgets import QApplication
    if QApplication.instance() is None:
        from helio_qt.app import use_core_profile
        use_core_profile()
    app = QApplication.instance() or QApplication([sys.argv[0]])
    import cv2
    import helio_qt.window as hw
    for name in ("information", "warning", "critical"):
        setattr(hw.QMessageBox, name, staticmethod(lambda *a, **k: None))
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Yes)
    ctx = {}

    def idle(limit=600):
        t0 = time.time()
        while ctx["w"].project.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    def capture_and_solve():
        n, L = _surface(), _lights()
        src = os.path.join(scratch, "capture")
        os.makedirs(src)
        for i, lv in enumerate(L):
            img = np.clip(n @ lv, 0, 1) * 235 + 10
            cv2.imwrite(os.path.join(src, f"img_{i:02d}.png"),
                        cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
        w = ctx["w"] = hw.MainWindow(appearance="dark")
        w.resize(1440, 900)
        w.show()
        w.open_paths([src])
        w.project.set_lights(L, "file:selftest.lp")
        w.project.update_settings(photo_tone="linear")
        w.registry.trigger("process.run")
        idle()
        r = w.project.result
        if r is None:
            raise RuntimeError(w.project.message or "no result")
        got = r.normals_full.reshape(-1, 3)
        err = np.degrees(np.arccos(np.clip(np.sum(got * n.reshape(-1, 3), 1), -1, 1)))
        if float(np.median(err)) > 3.0:
            raise RuntimeError(f"normals off by {float(np.median(err)):.1f}° (median)")
        return f"{len(L)} frames, normals within {float(np.median(err)):.2f}° (median)"
    step("a capture of a known surface solves", capture_and_solve)

    def views():
        import logging
        warned = []

        class Catch(logging.Handler):
            def emit(self, rec):
                if "fall" in rec.getMessage().lower():
                    warned.append(rec.getMessage())
        catch = Catch(level=logging.WARNING)
        logging.getLogger().addHandler(catch)
        try:
            w = ctx["w"]
            ins = w.workspaces["Inspect"]
            names = ("Standard PBR", "Positive Openness", "Negative Openness",
                     "Exaggerated Shading", "Surface Curvature (Scratches)")
            failed = []
            for v in names:
                if v not in ins.order:
                    failed.append(f"{v}: not a view")
                    continue
                w.set_view(v)
                ins.wait_render()
                app.processEvents()
                if ins.error:
                    failed.append(ins.error)
        finally:
            logging.getLogger().removeHandler(catch)
        if warned or failed:
            raise RuntimeError("; ".join((failed + warned)[:3]))
        return f"{len(names)} views drawn, RVT used where it should be"
    step("views draw, with RVT (no fallbacks)", views)

    def exports():
        from helio_qt import publish
        w = ctx["w"]
        w.project.scale_mm_per_px = 0.05
        src = w.publish_source()
        made, bad = [], []
        for key in ("plate_a3", "rti_web", "geotiff", "glb", "lp", "png_views", "sheet"):
            try:                                   # every export tried: one run shows every fault
                files = publish.export_one(key, src, out_dir)
                if not files or not all(os.path.exists(f) for f in files):
                    raise RuntimeError("wrote nothing")
                made.append(f"{key} ({len(files)})")
            except Exception as e:
                bad.append(f"{key}: {type(e).__name__}: {e}")
        if bad:
            raise RuntimeError("; ".join(bad))
        return ", ".join(made)
    step("exports: plate, RTI web, GIS, GLB, .lp, view, contact sheet", exports)

    def usdz():
        from helio.application import export_usd as U
        r = ctx["w"].project.result
        path = os.path.join(out_dir, "selftest.usdz")
        U.export(path, depth=r.depth_vis_p, normals=r.normals_full, albedo=r.albedo_rgb_full,
                 mm_per_px=0.05)
        ok, problems = U.check(path)
        if not ok:
            raise RuntimeError("; ".join(problems[:3]))
        return f"{os.path.getsize(path) // 1024} KB, ARKit checks passed"
    step("USDZ for AR, through OpenUSD's checker", usdz)

    def colour():
        from helio.application import colour_check as C
        found = C.find_chart(C.synthetic_chart(900, noise=1.5, seed=3))
        if found is None:
            raise RuntimeError("OpenCV did not find the drawn chart")
        res = C.evaluate(found["rgb"])
        return f"chart found, average ΔE00 {res['mean']:.2f}"
    step("colour check finds a ColorChecker", colour)

    ok = all(s["ok"] for s in steps)
    try:
        from helio_build import stamp
        build = stamp()
    except Exception:
        build = "unknown"
    report = dict(ok=ok, build=build, frozen=bool(getattr(sys, "frozen", False)),
                  executable=sys.executable, when=time.strftime("%Y-%m-%d %H:%M:%S"), steps=steps)
    with open(os.path.join(out_dir, "selftest.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"{'PASSED' if ok else 'FAILED'}: {sum(s['ok'] for s in steps)} of {len(steps)} steps "
          f"· build {build}", flush=True)
    if "w" in ctx:
        ctx["w"].close()
        app.processEvents()
    return 0 if ok else 1
