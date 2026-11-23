"""The turn in exports, the 3D relief's camera straight down, and the Relight + 3D file.

* The 3D relief's camera does not jump or flip as it passes straight down (it did,
  past 88.5°, facing any way but north).
* Turning normals with their picture lights the surface as before from a turned
  light: quarter turns, mirrors and any angle.
* The light sweep (GIF, MP4) and the Relight + 3D file come out turned as Inspect
  shows the object.
* The Relight + 3D file: JPEG textures at 4096, a finer mesh, under 50 MB, and the
  3D tab opening at its own, lower light.

    python tests/test_export_turn.py
"""
import base64
import json
import os
import re
import struct
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


def glb_of(page: str) -> dict:
    b = base64.b64decode(re.search(r"data:model/gltf-binary;base64,([A-Za-z0-9+/=]+)", page).group(1))
    jl = struct.unpack("<I", b[12:16])[0]
    return json.loads(b[20:20 + jl]), b


def main():
    import numpy as np
    from helio.application.viewing import Display, screen_surface, turn_image, turn_vectors

    print("The 3D relief's camera, straight down")
    from helio_qt.relief3d import Camera
    c, worst = Camera(), 0.0
    for az in range(0, 360, 5):
        c.az = az
        c.el = 88.4
        a = c.view()[:3, :3]
        c.el = 88.6
        worst = max(worst, float(np.abs(a - c.view()[:3, :3]).max()))
    check(f"no jump past 88.5°, facing any way ({worst:.4f})", worst < 0.01)

    print("\nTurning the surface")
    H, W = 90, 150
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    z = np.exp(-((xx - 45) ** 2 / 200 + (yy - 30) ** 2 / 90)) * 15 + xx * 0.05
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(z)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([0.6, 0.3, 0.74])
    L /= np.linalg.norm(L)
    for disp, deg in ((Display(), 90), (Display(), -90), (Display(flip_h=True), 0),
                      (Display(flip_v=True, rot180=True), 90), (Display(), 30), (Display(), -137.5)):
        fh, fv = disp.flip_h ^ disp.rot180, disp.flip_v ^ disp.rot180
        a = np.clip(n @ L, 0, 1)
        a = np.fliplr(a) if fh else a
        a = np.flipud(a) if fv else a
        a = turn_image(a, deg)
        inside = turn_image(np.ones((H, W), np.float32), deg) > 0.999
        Lt = L * [(-1 if fh else 1), (-1 if fv else 1), 1]
        nt = screen_surface(disp, deg, normals=n)[0]
        b = np.clip(nt @ turn_vectors(Lt[None], deg)[0], 0, 1)
        d = float(np.abs(a - b)[inside].max())
        check(f"lit the same turned {deg}°, mirrored {fh}/{fv} ({d:.3f})", d < 0.02)
    p = np.zeros((40, 60), np.uint8)
    p[2:6, 50:58] = 255                                     # top right
    q = turn_image(p, 90)
    ys, xs = np.nonzero(q)
    check("a quarter turn is clockwise, as Inspect's Turn right", q.shape == (60, 40)
          and ys.mean() > 40 and xs.mean() > 30)

    print("\nThe exports, turned")
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import time
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.show()
    from tools import make_golden as G
    paths, Ls = G.render_stack(tempfile.mkdtemp(prefix="helio_turn_"))
    w.open_paths([os.path.dirname(paths[0])])
    w.project.set_lights(Ls, "file:golden.lp")
    w.project.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 600:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    from helio_qt import publish as P
    from helio.application import export_web as EW
    flat = P._sweep_draw(w.publish_source())(0.3, 0.2, 0.93)
    w.rotate_right()
    src = w.publish_source()
    check("Publish reads Inspect's turn", src.turn == 90)
    turned = P._sweep_draw(src)(0.3, 0.2, 0.93)
    check("the light sweep's frames are turned", turned.shape[:2] == flat.shape[1::-1]
          and np.array_equal(turned, np.rot90(flat, -1)))
    from helio_qt.sweep_dialog import SIZES, SweepDialog
    check("the sweep comes in 720 px and 4K too", "720" in SIZES and SIZES["4k"][1] == 3840)
    dlg = SweepDialog(w)
    dlg.size.set_current("720")
    dlg.length.setCurrentIndex(0)
    st = dlg.settings()
    st.pop("fmt")
    gif = os.path.join(tempfile.mkdtemp(prefix="helio_sweep_"), "s.gif")
    w.write_light_sweep(gif, wait=True, **st)
    from PIL import Image as _I
    g = _I.open(gif)
    rw, rh = w.project.result.w, w.project.result.h
    check(f"Save the light sweep, as the dialog saves it: turned, 720 px or less ({g.size})",
          max(g.size) == min(720, max(rw, rh)) and (g.size[1] > g.size[0]) == (rw > rh))
    dlg.close()
    print("\nSeveral views in one sweep")
    from helio.application import multiview as MV
    check("panels go in a row for a tall object, a column for a wide one",
          MV.grid(3, 300, 900) == (1, 3) and MV.grid(2, 1600, 400) == (2, 1))
    three = [("", ""), ("Standard PBR", MV.COLOUR), ("Normals Unsharp Masking", MV.GREY)]
    vd = tempfile.mkdtemp(prefix="helio_views_")
    one = os.path.join(vd, "one.mp4")
    w.write_light_sweep(one, wait=True, seconds=1, fps=10, long_side=0)
    side = os.path.join(vd, "side.mp4")
    w.write_light_sweep(side, wait=True, seconds=1, fps=10, views=three, layout="side")
    after = os.path.join(vd, "after.gif")
    w.write_light_sweep(after, wait=True, seconds=1, fps=10, views=three, layout="after")
    import cv2

    def vinfo(p):
        c = cv2.VideoCapture(p)
        out = (int(c.get(cv2.CAP_PROP_FRAME_COUNT)), int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
               int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        c.release()
        return out
    n1, w1, h1 = vinfo(one)
    n3, w3, h3 = vinfo(side)
    check(f"side by side: three panels, the same length ({w1}x{h1} -> {w3}x{h3}, {n3} frames)",
          n3 == n1 and (w3 > 2.5 * w1 or h3 > 2.5 * h1))
    ga = _I.open(after)
    check(f"one after another: three turns ({ga.n_frames} frames, against {n1})", ga.n_frames == 3 * n1)
    f0 = np.asarray(ga.convert("RGB"))
    ga.seek(2 * n1)                                  # the third view, at the first light
    f1 = np.asarray(ga.convert("RGB"))
    d = float(np.abs(f0.astype(int) - f1.astype(int)).mean())
    check(f"…each in its own view ({d:.1f})", d > 3)
    lab = MV.caption(np.full((200, 300, 3), 128, np.uint8), "Standard PBR, colour")
    check("a view is named in its corner", lab[-30:, :120].std() > 5 and lab[:100].std() == 0)
    tif = P.current_view(src, tempfile.mkdtemp(prefix="helio_tif_"))
    t_img = _I.open(tif)
    check(f"This view at full resolution (TIFF): turned ({t_img.size})", t_img.size == (rh, rw))
    print("\nReports & figures, turned")
    rep = tempfile.mkdtemp(prefix="helio_reports_")
    for key in ("pub_sheet", "plate_a3", "figures", "condition", "sheet", "plate", "line_drawing"):
        try:
            files = P.export_one(key, src, rep)
            check(f"{key} written while turned", bool(files))
        except Exception as e:                              # noqa: BLE001
            check(f"{key} written while turned", False, f"{type(e).__name__}: {e}")
    import dataclasses
    import glob
    import cv2

    class Big:                                  # a capture over 2400 px, measured in mm
        def __init__(self, d):
            self._d = d

        def __getattr__(self, k):
            v = getattr(self._d, k)
            if k == "albedo_rgb_full":
                return cv2.resize(np.asarray(v), (3000, int(3000 * v.shape[0] / v.shape[1])))
            return v
    need = P._need_data
    P._need_data = lambda s: Big(need(s))
    try:
        files = P.export_one("plate_a3", dataclasses.replace(src, mm_per_px=0.05, z_mm_per_unit=0.05), rep)
        check("the A3 plate of a large, measured capture, turned", bool(files))
    except Exception as e:                                  # noqa: BLE001
        check("the A3 plate of a large, measured capture, turned", False, f"{type(e).__name__}: {e}")
    finally:
        P._need_data = need
    scaled = dataclasses.replace(src, mm_per_px=0.05)
    for key in ("png_views", "all_views", "deepzoom", "scalebar", "heatmap", "anaglyph"):
        try:
            files = P.export_one(key, scaled, rep)
            if key == "deepzoom":                   # the colour layer's size, from its .dzi
                dzi = glob.glob(os.path.join(rep, "*_deepzoom", "albedo.dzi"))
                txt = open(dzi[0]).read() if dzi else ""
                wh = re.search(r'Width="(\d+)"\s+Height="(\d+)"', txt) or re.search(
                    r'Height="(\d+)"\s+Width="(\d+)"', txt)
                if wh and wh.re.pattern.startswith("Height"):
                    sz = (int(wh.group(2)), int(wh.group(1)))
                else:
                    sz = (int(wh.group(1)), int(wh.group(2))) if wh else None
                check(f"{key}: turned ({sz})", sz is not None and (sz[1] > sz[0]) == (rw > rh))
                continue
            pics = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]
            sz = _I.open(pics[0]).size if pics else None
            check(f"{key}: turned ({sz})", sz is not None and (sz[1] > sz[0]) == (rw > rh))
        except Exception as e:                              # noqa: BLE001
            check(f"{key}: turned", False, f"{type(e).__name__}: {e}")
    import glob
    tifs = sorted(glob.glob(os.path.join(rep, "*_figures", "*.tif")))
    if tifs:
        fi = _I.open(tifs[0])
        check(f"a publication figure is turned ({fi.size})", (fi.height > fi.width) == (rw > rh))
    from helio.application import pub_sheet as PS
    from helio.application.viewing import ScreenTurn
    sh = PS.Sheet(title="t", full_size=(200, 100), albedo=np.zeros((50, 100, 3), np.uint8), views=[],
                  sections=[{"a_px": (150.0, 20.0), "b_px": (150.0, 80.0)}],
                  measures=[(10.0, 10.0, 20.0, 10.0, "m")], notes=[{"type": "line", "pts": [(150.0, 20.0)]}],
                  details=[("d", (0, 0, 40, 20))], lights=np.array([[1.0, 0.0, 0.0]]),
                  turn=ScreenTurn(None, 90))
    tt = PS.turned(sh)
    check("the sheet's lines, notes and boxes turn with it",
          tt.full_size == (100, 200) and np.allclose(tt.sections[0]["a_px"], (80.0, 150.0))
          and np.allclose(tt.notes[0]["pts"][0], (80.0, 150.0)) and tt.details[0][1] == (80, 0, 100, 40)
          and np.allclose(tt.lights[0], (0.0, -1.0, 0.0), atol=1e-9) and tt.albedo.shape[:2] == (100, 50))
    print("\nAnalyses on the publication sheet")
    from helio.application import sheet_tools as ST
    got = {}
    keep_make = PS.make
    PS.make = lambda s, b: got.update(s=s) or []
    w.set_sheet_prefs(analyses=list(ST.ANALYSES))
    try:
        P._pub_sheet(w.publish_source(), tempfile.mkdtemp(prefix="helio_an_"))
    finally:
        PS.make = keep_make
        w.set_sheet_prefs(analyses=[])
    caps = [c for c, _im in got["s"].tools]
    check(f"tool marks worked out for the sheet ({caps[0][:60] if caps else 'none'}…)",
          any(c.startswith("Tool marks:") and "strokes" in c for c in caps))
    check(f"…and the other analyses ({len(caps)} of {len(ST.ANALYSES)})", len(caps) >= 4, caps)
    tm = next((im for c, im in got["s"].tools if c.startswith("Tool marks")), None)
    check("…turned as the screen shows the object", tm is not None
          and (tm.shape[0] > tm.shape[1]) == (rw > rh), None if tm is None else tm.shape)
    from helio_qt.workspaces import _analyses_text
    check("the sheet's summary names them", _analyses_text(["toolmarks", "lines"]) ==
          "tool marks and line drawing")
    print("\nThe sheet's footer stays with it")
    foot = PS.Sheet(title="t", full_size=(100, 100), albedo=np.zeros((10, 10, 3), np.uint8), views=[],
                    lights=np.array([[0.5, 0.0, 0.86]] * 8), footer=["a", "b", "c"])
    for y in (4505, 4700, PS.PAGE_H - PS.M - 5):
        pg = PS._Page()
        pg.y = y
        PS._footer(pg, foot)
        pages = pg.pages()
        paper = PS.to_paper(pages[0], "A3")
        check(f"footer from y={y}: one page ({len(pages)}), A3 on paper {paper.size}",
              len(pages) == 1 and paper.size == (PS.PAGE_W, PS.PAGE_H)
              and pages[0].height <= PS.PAGE_H * (1 + PS.FOOTER_STRETCH))
    gold = np.zeros((40, 40, 3), np.uint8)
    gold[...] = (200, 150, 60)
    check("colour views print as drawn; grey views are stretched",
          np.array_equal(PS.balanced("Matte Relight (as 3D)", gold), gold)
          and not np.array_equal(PS.balanced("Exaggerated Shading", np.dstack([np.linspace(60, 120, 1600).reshape(40, 40)] * 3).astype(np.uint8)),
                                 np.dstack([np.linspace(60, 120, 1600).reshape(40, 40)] * 3).astype(np.uint8)))
    out = tempfile.mkdtemp(prefix="helio_rti3d_")
    path = P._rti_3d(src, out)[0]
    page = open(path, encoding="utf-8").read()
    mb = os.path.getsize(path) / 1e6
    check(f"Relight + 3D under 50 MB ({mb:.1f} MB)", mb < 50.0)
    inner = page.replace("\\u003c", "<")
    gltf, blob = glb_of(inner)
    check("its 3D textures are JPEG", {i.get("mimeType") for i in gltf["images"]} == {"image/jpeg"})
    from PIL import Image
    import io
    bv = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    o = 20 + struct.unpack("<I", blob[12:16])[0] + 8
    pic = Image.open(io.BytesIO(blob[o + bv.get("byteOffset", 0):o + bv.get("byteOffset", 0) + bv["byteLength"]]))
    check("…and turned: the colour is portrait where the capture is landscape",
          (pic.height > pic.width) == (w.project.result.w > w.project.result.h), pic.size)
    try:
        import trimesh
        m = trimesh.load(io.BytesIO(base64.b64decode(re.search(
            r"data:model/gltf-binary;base64,([A-Za-z0-9+/=]+)", inner).group(1))), file_type="glb")
        check("the GLB still opens", len(m.geometry) >= 1)
    except Exception as e:                                  # noqa: BLE001
        check("the GLB still opens", False, e)
    check("the 3D tab opens at its own light", all(f'\\"{k}\\":' in page or f'"{k}":' in page
                                                    for k in EW.RTI_3D_LOOK))
    check("…from the front, tilted, centred; and Space + drag pans",
          f"VIEW_TILT={EW.RTI_3D_TILT:g}" in page and "THREE.MOUSE.PAN" in page and "__VIEW_TILT__" not in page)
    w.rotate_reset()
    check("no turn: Publish reads none", w.publish_source().turn == 0)
    from helio.application.viewing import screen_image
    w.set_selection((40, 30, w.project.result.w - 60, w.project.result.h - 40))
    s_src = w.publish_source()
    ica = screen_image("Colour ICA (IC1-3)", s_src.data, {}, s_src.screen())
    check(f"a selection's Colour ICA is drawn from its photographs, not black ({ica.mean():.0f})",
          ica.mean() > 20)
    w.set_selection(None)
    w.close()
    app.processEvents()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    sys.exit(main())
