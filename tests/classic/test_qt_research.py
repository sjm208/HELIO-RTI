"""The Research menu and AI (the classic Research, Metrology and AI Tools windows).

Offscreen.  The shared maths on surfaces with known answers: a pit's depth
is measured below its surroundings, not as the height there (KD-046); a
groove's floor is traced, not its shoulders or the background, and its
length is the stroke's (KD-047); pyramid focus fusion is seamless and keeps
16 bits; the registration models recover a known transform with its RMS.
Then each window on the solved golden stack, and the AI dispatcher with the
network replaced: the Gemini key in a header, no retry without certificate
checks (KD-048), a local model with no key, located findings drawn.

    python tests/test_qt_research.py
"""
import json
import math
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "MISTRAL_API_KEY"):
    os.environ.pop(_k, None)
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


class Reply:
    def __init__(self, d):
        self.d = d

    def read(self):
        return json.dumps(self.d).encode()


def shared_maths(out):
    import cv2
    import numpy as np
    from helio.application import research as R
    print("\nShared maths")
    H, W = 200, 240
    yy, xx = np.mgrid[0:H, 0:W]
    d = 0.2 + 0.6 * xx / W
    for y, x, r, dep in ((60, 40, 5, 0.08), (100, 200, 5, 0.08)):
        d = d - dep * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * (r / 1.5) ** 2))
    rows = R.micro_features(d.astype(np.float32), threshold=0.01, min_depth=0.01, mm_per_px=0.05,
                            z_mm_per_unit=2.0)
    by_x = sorted(rows, key=lambda r: r["x_px"])
    check("two equal pits, one low and one high on a slope, read as equally deep (KD-046)",
          len(rows) == 2 and abs(by_x[0]["depth_norm"] - by_x[1]["depth_norm"]) < 0.01
          and by_x[0]["depth_norm"] > 0.04, rows)
    check("…positions in mm from the measured scale, depth in µm from the height scale (KD-044)",
          rows and abs(by_x[0]["x_mm"] - 2.0) < 0.15 and by_x[0]["depth_um"] > 50, rows)
    try:                    # a scale set: the mm values were float32, which JSON refused
        import tempfile as _tf
        R.export_micro_features(rows, d, _tf.mkdtemp(prefix="helio_micro_"), "t", "t")
        ok_, why_ = True, ""
    except TypeError as e:
        ok_, why_ = False, str(e)
    check("…and with a scale set, the GeoJSON export writes", ok_, why_)

    g = np.full((200, 200), 0.5, np.float32)
    g[50, 40:140] -= 0.2
    for i in range(80):
        g[100 + i, 30 + i] -= 0.2
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    strokes, labels, skel = R.toolpaths(g, min_len=20, threshold=0.15, mm_per_px=0.1)
    st = sorted(strokes, key=lambda s: s["angle_deg"])
    check("grooves are traced along their floors: two strokes, not the background (KD-047)",
          len(strokes) == 2 and all(s["mean_groove_depth"] > 0.01 for s in strokes), strokes)
    check("…a 100 px groove measures ~99 px, an 80-step diagonal ~113 (√2), not the dilated area",
          len(st) == 2 and abs(st[0]["length_px"] - 99) <= 2 and abs(st[1]["length_px"] - 80 * math.sqrt(2)) < 6
          and abs(st[0]["length_mm"] - 9.9) < 0.3, st)
    check("…with their directions (0° east–west, 45°), bearings from north",
          len(st) == 2 and st[0]["angle_deg"] == 0 and abs(st[1]["angle_deg"] - 45) < 2
          and st[0]["bearing_deg"] == 90, st)
    gy, gx = np.gradient(g.astype(np.float64) * 40)
    n = np.dstack([-gx, gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    sn, _l, _s = R.toolpaths(g, normals=n.astype(np.float32), use_normals=True, min_len=20)
    check("…and the same strokes from the normals' slopes (+Y north), without integrating",
          len(sn) == 2, sn)
    files = R.export_toolpaths(strokes, labels, skel, np.full((200, 200, 3), 128, np.uint8), out, "T")
    svg = open(files[0]).read()
    check("the toolpath SVG draws each stroke as a polyline along it", svg.count("<polyline") >= 2)
    by_dir = R.stroke_colours(st, by_direction=True)
    check("Colour by direction: the east–west stroke red, the 45° one green",
          by_dir[st[0]["id"]][0] > 200 > by_dir[st[0]["id"]][1]
          and by_dir[st[1]["id"]][1] > 200 > by_dir[st[1]["id"]][2], by_dir)
    deep, shallow = sorted(st, key=lambda s: -s["mean_groove_depth"])
    both = R.stroke_colours(st, by_direction=True, by_depth=True)
    check("…shaded by depth too: the deeper stroke brighter, its hue kept",
          max(both[deep["id"]]) > max(both[shallow["id"]]), both)
    plain = R.stroke_colours(st)
    check("…neither ticked: the eight colours in turn, as before",
          plain[st[0]["id"]] == (255, 0x44, 0x44), plain)
    check("a key for each colouring, none for the plain colours",
          R.colour_key(True, False).shape == (22, 280, 3) and R.colour_key(False, False) is None)
    files = R.export_toolpaths(strokes, labels, skel, np.full((200, 200, 3), 128, np.uint8), out, "Tdir",
                               colours=by_dir)
    check("…and Export draws them in those colours",
          ("#%02x%02x%02x" % by_dir[st[0]["id"]]) in open(files[0]).read())

    # Two blocks of writing on grainy paper: on a real binding the grain and a
    # textured backdrop passed the fixed cut everywhere, and no zone was left.
    def page(scale):
        H, W = 900 * scale, 700 * scale
        rng = np.random.default_rng(7)
        alb = np.clip(200 + rng.normal(0, 14, (H, W)), 0, 255).astype(np.uint8)   # grainy paper
        for (x0, y0) in ((100, 120), (100, 560)):
            for line in range(6):
                for ch in range(18):
                    cx, cy = (x0 + 28 * ch) * scale, (y0 + 36 * line) * scale
                    cv2.putText(alb, "e", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, 40,
                                max(1, 2 * scale))
        return np.full((H, W), 0.5, np.float32), np.dstack([alb] * 3)
    import time
    for scale in (1, 6):
        dep, alb = page(scale)
        t0 = time.time()
        zb, _acc = R.inscription_zones(dep, None, None, alb, mode="Book / Manuscript", sens=0.7)
        dt = time.time() - t0
        H, W = dep.shape
        on_text = [b for b in zb if b[2] * b[3] < 0.4 * H * W
                   and any(abs(b[1] + b[3] / 2 - yc * scale) < 120 * scale for yc in (210, 650))]
        check(f"two blocks of writing on grainy paper are two zones, not the page "
              f"({W}×{H}: {len(zb)} zones, {dt:.1f} s)",
              len(zb) == 2 and len(on_text) == 2 and dt < 20, zb[:4])

    # A large capture with thousands of strokes: it ran for hours, a pass of
    # the whole image per stroke (2026-10-17, a 98 MP binding).
    import time
    big = np.full((4000, 4000), 0.5, np.float32)
    for yy in range(40, 3960, 40):
        for xx in range(40, 3960, 160):
            big[yy, xx:xx + 100] -= 0.2
    big = cv2.GaussianBlur(big, (0, 0), 2.0)
    t0 = time.time()
    many, ml, ms = R.toolpaths(big, min_len=20, threshold=0.15)
    dt = time.time() - t0
    check(f"thousands of strokes on a 16 MP surface in seconds ({len(many)} in {dt:.1f} s)",
          len(many) > 2000 and dt < 30, (len(many), dt))
    t0 = time.time()
    R.export_toolpaths(many, ml, ms, np.full((4000, 4000, 3), 128, np.uint8), out, "Big")
    check(f"…and exported in seconds ({time.time() - t0:.1f} s)", time.time() - t0 < 60)
    # A wide groove with a speckled floor: one line along it, not a chain of loops.
    wide = np.full((300, 600), 0.5, np.float32)
    cv2.line(wide, (50, 150), (550, 150), 0.3, 24)
    wide = (cv2.GaussianBlur(wide, (0, 0), 4.0)
            + np.random.default_rng(1).normal(0, 0.004, wide.shape).astype(np.float32))
    one, _l, _s = R.toolpaths(wide, min_len=100, threshold=0.15, smooth=4.0)
    check("a wide, speckled groove is one stroke along its length when smoothed",
          len(one) == 1 and one[0]["length_px"] > 400, [(s["length_px"]) for s in one])
    check("the smoothing follows the capture's size (about 3 px at 100 MP)",
          R.groove_smoothing(8585, 11444) == 3.3 and R.groove_smoothing(200, 200) == 1.0)

    rng = np.random.default_rng(0)
    tex = (rng.random((128, 128, 3)) * 65535).astype(np.uint16)
    blur = cv2.GaussianBlur(tex, (0, 0), 4)
    a = tex.copy()
    a[:, 64:] = blur[:, 64:]
    b = tex.copy()
    b[:, :64] = blur[:, :64]
    paths = []
    for li in range(3):
        for k, im in enumerate((a, b)):
            p = os.path.join(out, f"L{li}_f{k}.tif")
            cv2.imwrite(p, im)
            paths.append(p)
    fused = R.focus_stack(paths, 3, 2, align=False)
    err = np.abs(fused[0].astype(float) - tex[..., ::-1]).mean() / 65535
    check("pyramid focus fusion recovers the sharp image across the seam, 16-bit kept",
          len(fused) == 3 and fused[0].dtype == np.uint16 and err < 0.02, err)

    pa = np.array([[10, 10], [100, 20], [30, 90], [120, 110], [60, 60]], float)
    th = math.radians(10)
    Rm = 1.2 * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    pb = pa @ Rm.T + [5, -3]
    for m in R.MODELS:
        M, rms = R.register_points(pa, pb, m)
        check(f"{m} recovers a turn-and-scale with ~0 px RMS", rms < 0.05 and
              (M.shape == (3, 3) if m == "Projective" else M.shape == (2, 3)), rms)
    shear = pa @ np.array([[1.0, 0.4], [0.0, 1.0]]).T
    _M, rs = R.register_points(pa, shear, "Similarity")
    _M, ra = R.register_points(pa, shear, "Affine")
    check("a shear fits as Affine, not as Similarity (HELIO 2's \"Affine\" was the similarity)",
          ra < 0.05 < rs, (rs, ra))
    Mm, rm = R.register_points(pa, np.column_stack([(199 - pa[:, 0]), pa[:, 1]]), "Similarity",
                               mirror=True, width_a=200)
    check("mirroring A reflects about its last column (w − 1 − x)", rm < 0.05, rm)


def main():
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QApplication
    print("=" * 60)
    print("Research tools and AI")
    print("=" * 60)
    out = tempfile.mkdtemp(prefix="helio_research_out_")
    shared_maths(out)

    app = QApplication(sys.argv)
    import helio_qt.window as hw
    import helio_qt.research as rs
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.information = staticmethod(lambda *a, **k: None)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()
    p = w.project

    def idle(limit=600):
        t0 = time.time()
        while p.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    print("\nThe menu")
    acts = w.registry.actions
    check("a Research menu with its tools and an AI submenu",
          "Research" in [s.split("/")[0] for s in (sp.menu for sp in w.registry.specs.values())]
          and "research.open:toolpaths" in acts and "ai.open:identify" in acts)
    check("…unavailable until solved (focus stacking is not: it makes the photographs)",
          not acts["research.open:micro"].isEnabled() and acts["research.open:focus"].isEnabled())

    from tools.make_golden import render_stack
    tmp = tempfile.mkdtemp(prefix="helio_research_")
    _paths, L = render_stack(tmp)
    w.open_paths([tmp])
    p.set_lights(L, "file:golden.lp")
    p.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    idle()
    r = p.result
    w.go_workspace("Inspect")
    app.processEvents()
    check("solved; the Research tools are available", r is not None and acts["research.open:micro"].isEnabled())

    def run(dlg):
        dlg.run()
        w.wait_background()
        app.processEvents()
        return dlg.result

    print("\nReflectance unmixing")
    dlg = w.open_research("unmix")
    res = run(dlg)
    ph, pw = r.proxy["normals"].shape[:2]
    check("three components at preview size, variance shares ≤ 1",
          res is not None and res[0].shape == (3, ph, pw) and 0 < sum(res[1]) <= 1.0001, dlg.status.text())
    base = os.path.join(out, "u.tif")
    hw.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (base, ""))
    rs.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (base, ""))
    dlg.save_as.setCurrentIndex(3)                       # maps + 32-bit values
    dlg.save()
    raw = cv2.imread(os.path.join(out, "u_component_1_32f.tif"), cv2.IMREAD_UNCHANGED)
    check("…saved as colour maps and 32-bit values, with the C1-C3 composite",
          raw is not None and raw.dtype == np.float32
          and os.path.exists(os.path.join(out, "u_component_3.png"))
          and os.path.exists(os.path.join(out, "u_composite_C1-C3.png")))
    dlg.save_as.setCurrentIndex(2)                       # maps + 16-bit greyscale
    dlg.save()
    g16 = cv2.imread(os.path.join(out, "u_component_2_16bit.tif"), cv2.IMREAD_UNCHANGED)
    check("…or with 16-bit greyscale TIFFs for an image editor",
          g16 is not None and g16.dtype == np.uint16 and int(g16.max()) == 65535)
    dlg.save_as.setCurrentIndex(1)                       # JPEG maps only
    base = os.path.join(out, "j.jpg")
    rs.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (base, ""))
    dlg.save()
    check("…or as JPEG maps alone",
          os.path.exists(os.path.join(out, "j_component_1.jpg"))
          and not any(f.startswith("j_") and f.endswith(".tif") for f in os.listdir(out)))
    from helio.application.research import unmix
    rng = np.random.default_rng(3)
    a1, a2 = rng.random((2, 900, 900)).astype(np.float32)
    mixes = rng.random((6, 2)) + 0.2
    st = np.stack([np.clip((m[0] * a1 + m[1] * a2) * 120, 0, 255) for m in mixes])
    full = unmix(st, 2, "ICA")[0]
    c = [max(abs(np.corrcoef(full[i].ravel(), s.ravel())[0, 1]) for s in (a1, a2)) for i in range(2)]
    check("ICA learnt from a sample still separates two mixed sources (|r| > 0.98)",
          min(c) > 0.98, c)
    dlg.close()

    print("\nMicro-features")
    dlg = w.open_research("micro")
    dlg.thr.set_value(0.01)
    dlg.min_d.set_value(0.0)
    res = run(dlg)
    check("detection runs on the solved surface", res is not None and isinstance(res[1], list),
          dlg.status.text())
    rs.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: out)
    if res and res[1]:
        dlg.save()
        check("…exports CSV, GeoJSON and the annotated image",
              all(any(f.endswith(s) for f in os.listdir(out))
                  for s in ("_micro_features.csv", "_micro_features.geojson", "_micro_features_annotated.png")))
    else:
        check("…(nothing found on the golden surface: Export stays off)", not dlg.btn_save.isEnabled())
    dlg.close()

    print("\nToolpaths")
    dlg = w.open_research("toolpaths")
    res = run(dlg)
    check("tracing runs on the solved surface", res is not None and len(res) == 4, dlg.status.text())
    before = None if dlg.preview.img is None else dlg.preview.img.copy()
    dlg.chk_colour.setChecked(True)
    check("ticking Colour by direction and depth redraws at once, with its key, without tracing again",
          dlg.result is res and dlg.key.isVisibleTo(dlg) and "cyan" in dlg.key_text.text()
          and "deeper" in dlg.key_text.text()
          and (before is None or not res[0] or not np.array_equal(before, dlg.preview.img)))
    dlg.chk_colour.setChecked(False)
    check("…and unticked, the key goes", not dlg.key.isVisibleTo(dlg))
    if res and res[0]:
        dlg.save()
        check("…exports SVG, PNG and statistics",
              any(f.endswith("_toolpath_stats.csv") for f in os.listdir(out)))
    dlg.close()

    print("\nInscription zones")
    dlg = w.open_research("zones")
    dlg.mode.setCurrentText("Stone / Carved Relief")
    res = run(dlg)
    check("zones are found and drawn on the current view at full resolution",
          res is not None and res[2].shape[:2] == (r.h, r.w), dlg.status.text())
    zp = os.path.join(out, "zones.png")
    rs.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (zp, ""))
    if res and res[0]:
        dlg.save()
        check("…saved with the signal map and the zone list",
              os.path.exists(zp) and os.path.exists(os.path.join(out, "zones_signal.png"))
              and open(os.path.join(out, "zones.csv")).read().startswith("zone,x_px"))
    dlg.close()

    print("\nRegister two images")
    dlg = w.open_research("register")
    B = dlg.img_b
    h, wd = B.shape[:2]
    th = math.radians(8)
    M = cv2.getRotationMatrix2D((wd / 2, h / 2), math.degrees(th), 0.9)
    A = cv2.warpAffine(B, M, (wd, h))
    dlg.set_a(A)
    Minv = cv2.invertAffineTransform(M)
    for x, y in ((0.2, 0.2), (0.8, 0.25), (0.3, 0.75), (0.75, 0.8), (0.5, 0.5)):
        bx, by = x * wd, y * h
        ax, ay = M @ [bx, by, 1]
        dlg.add_point("A", ax, ay)
        dlg.add_point("B", bx, by)
    del Minv
    dlg.model.setCurrentText("Similarity")
    dlg.register()
    check("A is warped onto B; the points fit to < 0.5 px", dlg.rms is not None and dlg.rms < 0.5,
          dlg.status.text())
    diff = np.abs(dlg.warped.astype(float) - B.astype(float))[h // 4:3 * h // 4, wd // 4:3 * wd // 4].mean()
    check("…and the warped image lies on B", diff < 12, diff)
    rp = os.path.join(out, "reg.png")
    rs.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (rp, ""))
    dlg.save()
    tj = json.load(open(os.path.join(out, "reg_transform.json")))
    check("…exported with its transform and RMS", tj["model"] == "Similarity" and tj["rms_px"] < 0.5)
    dlg.close()

    print("\nFocus stacking")
    dlg = w.open_research("focus")
    dlg.lights.setValue(3)
    dlg.frames.setValue(2)
    dlg.paths = [os.path.join(out, f"L{li}_f{k}.tif") for li in range(3) for k in range(2)]
    res = run(dlg)
    check("three merged photographs from six", res is not None and len(res) == 3, dlg.status.text())
    fo = tempfile.mkdtemp(prefix="helio_focus_")
    rs.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: fo)
    dlg.save()
    check("…saved, with Open them offered next", sorted(os.listdir(fo))[0] == "light_01_stacked.tif"
          and dlg.btn_open is not None)
    dlg.close()

    print("\nRecto and verso")
    dlg = w.open_research("recto")
    dlg.choose_partner(os.path.join(out, "other_side.helio"))
    lp = os.path.join(out, "link.json")
    rs.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (lp, ""))
    dlg.side.set_current("Verso")
    dlg.save_link()
    link = json.load(open(lp))
    check("the link names both sides and the solver", link["this_surface"] == "Verso"
          and link["partner_surface"] == "Recto" and link["partner_file"].endswith("other_side.helio"))
    ro = tempfile.mkdtemp(prefix="helio_rv_")
    rs.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: ro)
    dlg.export()
    check("…and this side exports named for its surface",
          any(f.endswith("_verso_depth.tif") for f in os.listdir(ro)))
    dlg.close()

    print("\nAI")
    from helio.application import ai
    sent = []

    def fake(reply):
        def opener(req, timeout):
            sent.append(req)
            return Reply(reply)
        return opener
    dlg = w.open_ai("identify")
    dlg.vendor.setCurrentText("Gemini 2.5 Flash (Google)")
    dlg.key.setText("SECRETKEY")
    dlg.detail.set_current("Comprehensive")
    dlg.opener = fake({"candidates": [{"content": {"parts": [{"text": "```json\n" + json.dumps({
        "artifact_id": "x", "material_profile": "copper alloy",
        "detections": [{"label": "abrasion", "box_2d": [100, 100, 400, 500], "severity": 2,
                        "description": "worn rim"}]}) + "\n```"}]}}]})
    dlg.ask()
    w.wait_background()
    app.processEvents()
    req = sent[-1]
    body = json.loads(req.data)
    check("Gemini: the key travels in a header, not the URL",
          "SECRETKEY" not in req.full_url and req.get_header("X-goog-api-key") == "SECRETKEY")
    check("…the view and the levelled ISO 25178 statistics are sent (KD-049)",
          body["contents"][0]["parts"][0].get("inline_data") is not None
          and "Sa=" in body["contents"][0]["parts"][1]["text"]
          and "plane-levelled" in body["contents"][0]["parts"][1]["text"])
    check("…located findings are listed and drawn as numbered boxes",
          "abrasion" in dlg.text.toPlainText() and dlg.data is not None
          and not np.array_equal(dlg.preview.img, dlg.image))
    ap = os.path.join(out, "ai.txt")
    dlg.save(ap)
    check("…saved with the caveat, the JSON and the annotated view",
          ai.CAVEAT in open(ap).read() and os.path.exists(os.path.join(out, "ai.json"))
          and os.path.exists(os.path.join(out, "ai_annotated.png")))
    check("the key is kept for this session only, not in the settings file",
          w.ai_keys.get("Gemini 2.5 Flash (Google)") == "SECRETKEY"
          and "SECRETKEY" not in open(os.environ["HELIO_QT_SETTINGS"]).read()
          if os.path.exists(os.environ["HELIO_QT_SETTINGS"]) else True)

    import urllib.error
    calls = []

    def bad_cert(req, timeout):
        calls.append(req)
        raise urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    dlg.vendor.setCurrentText("Claude (Anthropic)")
    dlg.key.setText("k")
    dlg.opener = bad_cert
    dlg.ask()
    w.wait_background()
    app.processEvents()
    check("a certificate failure stops, once, with the fix; never a retry unchecked (KD-048)",
          len(calls) == 1 and "certificate" in dlg.status.text().lower(), dlg.status.text())

    dlg.vendor.setCurrentText("Ollama (this computer)")
    check("a local model needs no key", not dlg.key.isVisibleTo(dlg) and dlg.btn_find.isVisibleTo(dlg))
    dlg.opener = fake({"models": [{"name": "gemma3:4b"}]})
    dlg.find_models()
    w.wait_background()
    app.processEvents()
    check("…its loaded models are listed", dlg.model.currentText() == "gemma3:4b")
    dlg.task.set_current("caption")
    dlg._task_changed("caption")
    dlg.opener = fake({"message": {"content": "Fig. X. A relief."}})
    dlg.ask()
    w.wait_background()
    app.processEvents()
    body = json.loads(sent[-1].data)
    check("…and answers from this computer, image included",
          sent[-1].full_url.startswith("http://localhost:11434") and body["model"] == "gemma3:4b"
          and body["messages"][0].get("images") and dlg.text.toPlainText() == "Fig. X. A relief.")
    dlg.close()

    print("\nThe classic app calls the same code")
    src = open(os.path.join(ROOT, "Helio_Pro.py"), encoding="utf-8").read()
    imported = " ".join(ln for ln in src.splitlines()
                        if "from helio.application.research import" in ln).replace(",", " ").split()
    for fn in ("unmix", "micro_features", "export_micro_features", "toolpaths", "export_toolpaths",
               "focus_stack", "register_points", "recto_verso_link", "recto_verso_export",
               "inscription_zones"):
        check(f"classic calls research.{fn}", fn in imported)
    check("classic AI goes through helio.application.ai, and the unchecked retry is gone",
          "ai.call(vendor_name" in src and "CERT_NONE" not in src and "?key=" not in src)

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
