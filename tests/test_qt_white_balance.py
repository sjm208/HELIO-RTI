"""White balance from a neutral patch: the colour only.

Checks the gains make a patch neutral in linear light and keep its
luminance, refuse clipped / dark / coloured patches, and leave greys grey;
then, offscreen in the app on a blue-cast synthetic capture with a white
card: a box drawn with Pick white makes the card neutral in the colour the
views and exports read, the solve (normals, depth, grey) is untouched, the
gains travel with the session and come back on open, a re-solve keeps them,
and Reset gives back the colour exactly as solved.

    python tests/test_qt_white_balance.py
"""
import os
import sys
import tempfile
import time

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


def spread(rgb) -> float:
    m = np.asarray(rgb, float).reshape(-1, 3).mean(0)
    return float(m.max() - m.min())


def unit():
    from helio.core import white_balance as W
    from helio.core.radiometry import srgb_to_linear
    print("Gains")
    patch = np.full((10, 10, 3), (150, 165, 190), np.uint8)          # a bluish white
    g = W.gains_from_patch(patch)
    out = W.apply(patch, g)
    check("the patch comes out neutral", spread(out) <= 1.0, out[0, 0])
    y0 = float(W.Y709 @ srgb_to_linear(np.array([150., 165, 190]), 255.0))
    y1 = float(W.Y709 @ srgb_to_linear(out[0, 0].astype(float), 255.0))
    check("…at its own luminance", abs(y0 - y1) / y0 < 0.02, (y0, y1))
    check("R raised, B lowered for a blue cast", g[0] > 1 > g[2], g)
    grey = np.full((4, 4, 3), 120, np.uint8)
    check("no gains: the very same array", W.apply(grey, None) is grey and W.apply(grey, (1, 1, 1)) is grey)
    check("neutral gains: a grey stays the same grey", np.array_equal(W.apply(grey, (1.0, 1.0, 1.0)), grey))
    lut = W.lut(g)
    check("each channel's curve rises", all(np.all(np.diff(lut[:, 0, c].astype(int)) >= 0) for c in range(3)))
    for name, bad in (("clipped", np.full((6, 6, 3), 255, np.uint8)),
                      ("too dark", np.full((6, 6, 3), 3, np.uint8)),
                      ("strongly coloured", np.full((6, 6, 3), (200, 20, 20), np.uint8))):
        try:
            W.gains_from_patch(bad)
            check(f"a {name} patch is refused", False)
        except ValueError:
            check(f"a {name} patch is refused", True)
    check("described", W.describe(g).startswith("R ×") and W.describe(None) == "off")


def app_checks():
    import cv2
    from PySide6.QtWidgets import QApplication
    print("\nIn the app")
    app = QApplication.instance() or QApplication(sys.argv)
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

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "cb", os.path.join(ROOT, "tests", "fixtures", "synthetic_capture.py"))
    cb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cb)
    _z, n_gt = cb.surface()
    L = cb.lights()
    h, wd = n_gt.shape[:2]
    cast = np.array([0.80, 0.90, 1.0])                 # a cool light, in linear terms
    alb = np.ones((h, wd, 3)) * np.array([0.55, 0.30, 0.18])      # leather
    card = (slice(10, 40), slice(10, 40))
    alb[card] = 0.8                                    # a white card
    src_dir = os.path.join(tempfile.mkdtemp(prefix="helio_wb_"), "capture")
    os.makedirs(src_dir)
    for i, lv in enumerate(L):
        shade = np.clip(n_gt @ lv, 0, 1)[..., None] * 0.85 + 0.05
        lin = np.clip(alb * cast * shade, 0, 1)
        img = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
        cv2.imwrite(os.path.join(src_dir, f"Card_{i:02d}.png"),
                    cv2.cvtColor((img * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
    w.open_paths([src_dir])
    p = w.project
    p.set_lights(L, "file:fixture.lp")
    w.registry.trigger("process.run")
    idle()
    check("solved", p.result is not None)
    r, D = p.result, w.render_data
    w.go_workspace("Inspect")
    ins = w.workspaces["Inspect"]
    ins.set_layer("Combined Colour")
    app.processEvents()
    before = r.albedo_rgb_full.copy()
    normals, depth, grey = r.normals_full, r.depth_vis_p, r.albedo_grey_full
    check("the card starts blue", before[card][..., 2].mean() > before[card][..., 0].mean() + 5,
          before[card].reshape(-1, 3).mean(0))
    check("Pick white shows with the Colour layer", not ins.wb_box.isHidden())

    w.registry.trigger("view.white_balance")
    app.processEvents()
    check("…and starts the box tool", ins.btn_wb_pick.isChecked() and ins.viewport.tool == "rect")
    ins.viewport.rectDrawn.emit(14.0, 14.0, 36.0, 36.0)
    app.processEvents()
    check("the gains are set", p.white_balance is not None and not ins.btn_wb_pick.isChecked(),
          p.white_balance)
    check("…and the panel says them", ins.wb_note.text().startswith("White balance: R ×")
          and ins.btn_wb_reset.isEnabled(), ins.wb_note.text())
    check("the card is neutral in the colour views and exports read",
          spread(r.albedo_rgb_full[card]) < 2.5 and spread(D.albedo_rgb_full[card]) < 2.5,
          r.albedo_rgb_full[card].reshape(-1, 3).mean(0))
    ph, pw = D.albedo_proxy.shape[:2]
    pc = (slice(int(14 * ph / h), int(36 * ph / h)), slice(int(14 * pw / wd), int(36 * pw / wd)))
    check("…and in the preview colour", spread(D.albedo_proxy[pc]) < 4,
          (D.albedo_proxy.shape, D.albedo_proxy[pc].reshape(-1, 3).mean(0)))
    check("the solve is untouched (normals, depth, grey)",
          r.normals_full is normals and r.depth_vis_p is depth and r.albedo_grey_full is grey)
    from helio.application.viewing import screen_image
    img = screen_image(w.view, D, w.engine_settings.get(w.view, {}), w.display)
    check("the screen draws the balanced colour", img is not None and img.ndim == 3)
    sel_r, _sel_d = w.selected_result()
    check("an export's Source carries the gains", w.publish_source().white_balance == p.white_balance)
    notes = w._plate_facts().get("notes", [])
    check("the plate notes it", any("white balanced" in n for n in notes), notes)

    print("\nThe OpenLIME HSH export: launcher and maps")
    from helio_qt import publish
    out = tempfile.mkdtemp(prefix="helio_hsh_")
    files = publish._openlime_hsh(w.publish_source(), out)
    names = {os.path.relpath(f, out).split(os.sep, 1)[-1] for f in files}
    check("the RTI, its viewer and the double-click launcher",
          {"info.json", "index.html", "serve.py", "Open viewer.command", "Open viewer.bat"} <= names, names)
    want = {"maps/albedo_16bit.tif", "maps/normal_map_16bit.tif", "maps/cavity.png",
            "maps/sds_specular_component.png", "maps/pca_rgb.png", "maps/pca_1.png", "maps/README.txt"}
    check("…and the maps: colour, normal, specular, cavity, PCA …",
          want <= names, sorted(want - names))
    gone = {"metallic.png", "curvature_rgb.png", "roughness_16bit.tif", "displacement.png",
            "ambient_occlusion.png", "detail_normals.png", "sds_gloss_index.png"}
    check("…not the maps left out (metallic, curvature, roughness, height, AO, detail normals, gloss)",
          not {"maps/" + g for g in gone} & names, sorted({"maps/" + g for g in gone} & names))
    html = open([f for f in files if f.endswith("index.html")][0]).read()
    check("…and the viewer's Views menu offers them",
          all(f'"id": "{v}"' in html for v in ("RTI", "normals", "colour", "specular", "cavity", "pca"))
          and not any(f'"id": "{v}"' in html for v in ("roughness", "gloss", "height", "ao", "detail_normals"))
          and "maps/web/cavity.tzi" in html and "maps/web/cavity.tzi" in names, html[:0])
    check("…with relightable shine and clay from the maps",
          all(f'"id": "{v}"' in html for v in ("relit_shine", "relit_clay")) and "relit_colour" not in html
          and {"maps/web/brdf_kd.tzi", "maps/web/brdf_ks.tzi", "maps/web/brdf_gloss.tzi",
               "maps/web/brdf_normals.tzi"} <= names)
    import cv2 as _cv
    mp = [f for f in files if f.endswith("maps/albedo_16bit.tif")][0]
    a16 = _cv.imread(mp, _cv.IMREAD_UNCHANGED)
    card16 = a16[card][..., ::-1].reshape(-1, 3).mean(0) / 257
    check("…the colour map is white balanced", float(card16.max() - card16.min()) < 3, card16)
    readme = open([f for f in files if f.endswith("maps/README.txt")][0]).read()
    check("…and says so", "white balanced" in readme and "could not" not in readme, readme[-300:])

    from helio.io import session as S
    path = os.path.join(src_dir, "card.helio")
    S.write(path, w.session_state())
    gains = p.white_balance
    w.registry.trigger("view.white_balance_reset")
    app.processEvents()
    check("Reset gives back the colour exactly as solved",
          p.white_balance is None and np.array_equal(r.albedo_rgb_full, before))
    check("the session keeps the gains",
          (S.read(path).extra.get("helio3") or {}).get("white_balance", {}).get("gains")
          == list(gains))
    w._restore_kept(S.read(path))
    check("…and they come back on open", p.white_balance == gains)
    w.registry.trigger("process.run")
    idle()
    check("a new solve gets the same white balance",
          p.result is not r and spread(w.render_data.albedo_rgb_full[card]) < 2.5,
          w.render_data.albedo_rgb_full[card].reshape(-1, 3).mean(0))
    ins.viewport.set_tool("")
    ins.btn_wb_pick.setChecked(True)
    ins.viewport.rectDrawn.emit(wd + 40.0, h + 40.0, wd + 60.0, h + 60.0)   # off the picture: refused
    app.processEvents()
    check("a bad patch only says so; the gains stay", p.white_balance == gains, (p.white_balance, gains))
    w.close()


unit()
app_checks()
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
