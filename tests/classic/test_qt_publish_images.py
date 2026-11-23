"""Publish's image and surface-map formats write real, valid files (EXP-001).

Offscreen.  Solves the baseline fixture, then exports every view, the normal
map set, DeepZoom tiles, a view with a scale bar, the elevation heatmap, the
16-bit PBR suite, specular/diffuse separation, the PCA, the HSH maps and the
uncertainty map through Publish, and checks each file: one image per view
and the Standard PBR one exactly as Inspect renders it; the hillshade lit
from the north-west its name promises (KD-029); a DeepZoom folder that opens
offline; 16-bit maps that are 16-bit; a specular residual near zero on this
matte fixture; HSH coefficient 0 equal to the solve's.

    python tests/test_qt_publish_images.py
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
    import cv2
    import numpy as np
    print("=" * 60)
    print("Publish: images and surface maps")
    print("=" * 60)
    import test_exports as TE
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
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
    w.project.set_lights(L, "file:fixture.lp")
    w.project.update_settings(photo_tone="linear")        # the fixture is radiance
    w.registry.trigger("process.run")
    idle()
    r = w.project.result
    check("solved", r is not None)
    pub = w.workspaces["Publish"]
    w.go_workspace("Publish")
    from helio_qt.publish_catalog import all_formats, available
    keys = ("all_views", "normal_map", "deepzoom", "scalebar", "heatmap", "pbr", "sds", "pca",
            "hsh", "uncertainty")
    fm = {f.key: f for f in all_formats()}
    check("each format is in the catalogue and runs here", all(k in fm and available(fm[k])
                                                               for k in keys))
    w.project.scale_mm_per_px = 0.05
    pub.selected = []
    for k in keys:
        pub._toggle(k)
    check("all ten can be chosen", sorted(pub.chosen()) == sorted(keys), pub.chosen())
    out = tempfile.mkdtemp(prefix="helio_qt_img_")
    w.publish_to(out)
    idle()
    rows = w.project.export_rows
    check("all ten written", all(v[0] == "done" for v in rows.values()) and len(rows) == 10,
          {k: v[:2] for k, v in rows.items() if v[0] != "done"})

    def files(k, end=""):
        return [f for f in rows.get(k, ("", "", []))[2] if f.endswith(end)]

    print("\nEvery view")
    from helio.domain.registry import RENDER_ENGINES
    pngs = files("all_views", ".png")
    check(f"one PNG per view plus colour, normals and heatmap ({len(pngs)})",
          len(pngs) == len(RENDER_ENGINES) + 3, len(pngs))
    from helio.application.render import composite, render
    from helio.application.export_images import engine_file
    st = dict(w.engine_settings["Standard PBR"])
    want = composite("Standard PBR", render("Standard PBR", w.render_data, st, r.h, r.w, False),
                     w.render_data, st, r.h, r.w, False, view_mode="Combined Colour")
    pbr_png = [f for f in pngs if os.path.basename(f) == engine_file("Standard PBR") + ".png"]
    got = cv2.imread(pbr_png[0], cv2.IMREAD_COLOR)[..., ::-1] if pbr_png else None
    check("…the Standard PBR one exactly as Inspect renders it",
          got is not None and np.array_equal(got, want))

    print("\nNormal map set")
    nm = files("normal_map", ".png")
    check("five maps", len(nm) == 5, [os.path.basename(f) for f in nm])
    from helio.application.export_images import hillshade
    hh, ww = 161, 161
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    bump = np.exp(-((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * 20.0 ** 2)) * 25
    hs = hillshade(bump)
    nw_side = hs[80 - 14, 80 - 14]                   # up-left of the summit: north-west
    se_side = hs[80 + 14, 80 + 14]
    check("its hillshade is lit from the north-west, as named (KD-029)", nw_side > se_side + 0.3,
          (nw_side, se_side))

    print("\nDeepZoom")
    dz = files("deepzoom")
    folder = os.path.dirname(dz[0]) if dz else ""
    ok = bool(dz)
    try:
        TE._validate_html(os.path.join(folder, "index.html"))
    except Exception as e:
        ok = False
        print("   ", e)
    check("a well-formed viewer page with its tiles and OpenSeadragon beside it",
          ok and os.path.exists(os.path.join(folder, "albedo.dzi"))
          and os.path.exists(os.path.join(folder, "openseadragon.min.js"))
          and os.path.isdir(os.path.join(folder, "osd_images"))
          and os.path.exists(os.path.join(folder, "1_Mac.command")))

    print("\nSingle images")
    sb = cv2.imread(files("scalebar")[0]) if files("scalebar") else None
    check("the view with a scale bar is full size", sb is not None and sb.shape[:2] == (r.h, r.w))
    hm = cv2.imread(files("heatmap")[0]) if files("heatmap") else None
    check("the heatmap is full size and in colour",
          hm is not None and hm.shape == (r.h, r.w, 3) and hm[..., 0].std() > 0)
    pca = cv2.imread(files("pca")[0], cv2.IMREAD_UNCHANGED) if files("pca") else None
    check("the PCA is a three-channel TIFF", pca is not None and pca.ndim == 3 and pca.shape[2] == 3)
    unc = cv2.imread(files("uncertainty")[0], cv2.IMREAD_UNCHANGED) if files("uncertainty") else None
    check("the uncertainty map is 32-bit and 0–1",
          unc is not None and unc.dtype == np.float32 and 0 <= float(unc.min())
          and float(unc.max()) <= 1.0001)

    print("\nSurface maps")
    names = {os.path.basename(f) for f in files("pbr")}
    want_names = {"albedo_16bit.tif", "normal_map_16bit.tif", "metallic.png", "detail_normals.png",
                  "ambient_occlusion.png", "cavity.png", "curvature_rgb.png", "displacement.png",
                  "sds_diffuse_albedo.png", "sds_gloss_index.png"}
    check("the PBR suite has every map", want_names <= names, want_names - names)
    a16 = cv2.imread([f for f in files("pbr") if f.endswith("albedo_16bit.tif")][0], cv2.IMREAD_UNCHANGED)
    check("…its 16-bit maps are 16-bit", a16.dtype == np.uint16)
    sds = files("sds")
    check("specular/diffuse: seven maps and the README", len(sds) == 8)
    spec = cv2.imread([f for f in sds if f.endswith("_32f.tif")][0], cv2.IMREAD_UNCHANGED)
    check("…a matte fixture has almost no specular residual (fitted per light)",
          float(np.mean(spec)) < 0.08, float(np.mean(spec)))
    hsh = files("hsh")
    c0 = cv2.imread([f for f in hsh if f.endswith("hsh_coeff_0_f32.tif")][0], cv2.IMREAD_UNCHANGED)
    check("HSH: 18 maps, coefficient 0 the solve's exactly",
          len(hsh) == 18 and np.array_equal(c0, r.hsh_coeffs[0].reshape(r.h, r.w).astype(np.float32)))

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
