"""Publish's web viewers and the sharing tools write real, valid files (WEB, HER).

Offscreen.  Solves the baseline fixture, exports WebRTI, the combined RTI
viewer, the relight / OpenLIME package, the RTI DeepZoom relight, the IIIF
3D manifest, the comparison page and the engine sweep, and checks each:
pages well formed, the combined viewer within its e-mail budget, the relight
source frames named .jpg in a lights.lp that reads back to the solve's
lights, every IIIF light in front of the model (KD-036), the provenance
note's azimuth formula (KD-035).  Then a labelled QR code that decodes back
to its address, and an embed snippet.

    python tests/test_qt_publish_web.py
"""
import importlib.util
import json
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
    print("Publish: web viewers and sharing")
    print("=" * 60)
    import test_exports as TE
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    import helio_qt.dialogs as dg
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
        cv2.imwrite(os.path.join(src_dir, f"img_{i:02d}.jpeg"),
                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
    w.open_paths([src_dir])
    w.project.set_lights(L, "file:fixture.lp")
    w.project.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    idle()
    r = w.project.result
    check("solved", r is not None)
    w.go_workspace("Publish")
    pub = w.workspaces["Publish"]
    keys = ("webrti", "rti_combined", "relight_web", "rti_deepzoom", "iiif", "compare", "engine_sweep")
    pub.selected = []
    for k in keys:
        pub._toggle(k)
    check("all seven can be chosen", sorted(pub.chosen()) == sorted(keys), pub.chosen())
    out = tempfile.mkdtemp(prefix="helio_qt_web_")
    w.publish_to(out)
    idle()
    rows = w.project.export_rows
    check("all seven written", all(v[0] == "done" for v in rows.values()) and len(rows) == 7,
          {k: v[:2] for k, v in rows.items() if v[0] != "done"})

    def files(k, end=""):
        return [f for f in rows.get(k, ("", "", []))[2] if f.endswith(end)]

    def html_ok(path):
        try:
            TE._validate_html(path)
            return True
        except Exception as e:
            print("   ", e)
            return False

    print("\nViewers")
    page = [f for f in files("webrti", "index.html")]
    check("WebRTI: page, six PTM planes and info.json",
          page and html_ok(page[0]) and len(files("webrti", ".jpg")) >= 8
          and json.load(open(files("webrti", "info.json")[0]))["format"] == "ptm")
    comb = files("rti_combined", ".html")
    mb = os.path.getsize(comb[0]) / 1e6 if comb else 99
    check(f"combined RTI viewer: one page within the 25 MB e-mail budget ({mb:.1f} MB)",
          comb and html_ok(comb[0]) and mb <= 25)
    lp = [f for f in files("relight_web") if f.endswith("lights.lp")]
    from helio.io.lp import read_lp
    names = [ln.split()[0] for ln in open(lp[0]).read().splitlines()[1:]] if lp else []
    check("relight: frames re-saved as .jpg (relight rejects .jpeg), lights.lp the solve's",
          lp and all(n.endswith(".jpg") for n in names) and len(names) == len(L)
          and np.allclose(read_lp(lp[0]), r.light_matrix, atol=1e-5))
    info = [f for f in files("relight_web") if f.endswith(os.path.join("relight_rti", "info.json"))]
    check("…and the coefficient package RelightLab reads", bool(info))
    prov = [f for f in files("relight_web") if f.endswith("helio_provenance.json")]
    note = json.load(open(prov[0]))["note"] if prov else ""
    check("…with a provenance record whose azimuth note matches its +Y-up lights (KD-035)",
          "atan2(x, y)" in note, note)
    dz = files("rti_deepzoom", ".html")
    check("RTI DeepZoom relight: a page", dz and html_ok(dz[0]))
    man = json.load(open(files("iiif", ".json")[0])) if files("iiif", ".json") else {}
    ann = man["items"][0]["annotations"][0]["items"] if man else []
    lights = [a["body"]["position"] for a in ann if a["body"]["type"] == "PointLight"]
    check("IIIF: a manifest with one light per frame", len(lights) == len(L))
    check("…every light in front of the model (z > 0) and north stays up (KD-036)",
          all(p["z"] > 0 for p in lights)
          and all(np.sign(round(p["y"], 3)) == np.sign(round(float(v[1]), 3))
                  for p, v in zip(lights, r.light_matrix)))
    cmp_ = files("compare", ".html")
    check("comparison page: well formed, both views embedded",
          cmp_ and html_ok(cmp_[0]) and open(cmp_[0]).read().count("data:image/jpeg;base64,") == 2)
    sw = files("engine_sweep", ".html")
    check("every view on one page", sw and html_ok(sw[0]) and os.path.getsize(sw[0]) > 50_000)

    print("\nOn a phone")
    more = ("rti_web", "glb", "deepzoom", "rti_viewer")
    out2 = tempfile.mkdtemp(prefix="helio_qt_web_phone_")
    w.publish_to(out2, list(more))
    idle()
    pages = [f for k in keys + more for f in files(k, ".html") if not f.endswith("embed.html")]
    pages += [os.path.join(dp, f) for dp, _d, fs in os.walk(out2) for f in fs
              if f.endswith(".html") and f != "embed.html"]
    check("the other four web exports written", len(pages) == 10, pages)
    bare = []
    for f in pages:
        t = open(f, encoding="utf-8").read()
        if not ('name="viewport"' in t and "charset" in t[:3000].lower() and "helio-phone" in t):
            bare.append(os.path.basename(f))
    check(f"every web page ({len(pages)}) fits a phone: viewport, charset, the phone layer",
          not bare, bare)
    rti = open(comb[0], encoding="utf-8").read() if comb else ""
    check("…the RTI viewers hand their light and zoom to the touch gestures",
          "window.HelioPhone={getLight" in rti and "touchstart" in rti)
    from helio.application.phone_web import phone_ready
    check("…and a page is made ready once", phone_ready(rti) == rti)
    import re as _re
    sb = json.loads(_re.search(r"var SB=(\[\[.*?\]\]);", open(dz[0]).read()).group(1)) if dz else []
    top = (sb[0][0] + sb[0][1]) + (sb[3][0] + sb[3][1]) + 2 * (sb[8][0] + sb[8][1]) if sb else 99
    check("RTI DeepZoom relight: coefficients on the 0–1 scale it draws (it drew pure white)",
          len(sb) == 9 and top < 4.0, sb[:1])

    print("\nSharing")
    png = os.path.join(out, "qr.png")
    dg.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (png, ""))
    d = dg.QrDialog(w)
    d.url.setText("https://example.org/helio/object-12")
    d.save()
    img = cv2.imread(png)
    txt = cv2.QRCodeDetector().detectAndDecode(img)[0] if img is not None else ""
    check("a labelled QR code that decodes to its address",
          txt == "https://example.org/helio/object-12" and img.shape[1] >= 520, txt)
    d.url.setText("https://https://example.org/helio/object-12")      # pasted after the https://
    d.save()
    txt = cv2.QRCodeDetector().detectAndDecode(cv2.imread(png))[0]
    check("…an address pasted after https:// is mended, so a phone opens it",
          txt == "https://example.org/helio/object-12", txt)
    check("…and the code is on screen to scan", d.code.pixmap() is not None
          and not d.code.pixmap().isNull())
    d.url.setText("localhost:8000/index.html")
    check("…localhost is called out: on a phone it is the phone", "phone" in d.status.text(),
          d.status.text())
    d.url.setText("/Users/someone/viewer/index.html")
    check("…and a file path makes no code", d.code.pixmap().isNull() and "file" in d.status.text())
    site = tempfile.mkdtemp(prefix="helio_share_")
    with open(os.path.join(site, "index.html"), "w") as fh:
        fh.write("<html><body>relief</body></html>")
    from helio.application.phone_share import Share
    import urllib.request
    sh = Share(site, host="127.0.0.1")
    try:
        body = urllib.request.urlopen(sh.url, timeout=5).read().decode()
    except Exception as ex:
        body = str(ex)
    sh.stop()
    check("a web export is shared from this Mac, its index.html at the address",
          "relief" in body and sh.url.endswith("/index.html"), body[:80])
    w.phone_share = None
    e = dg.EmbedDialog(w)
    e.url.setText("https://museum.org/v/index.html")
    s = e.text.toPlainText()
    check("an embed snippet for the address", s.startswith("<iframe") and "museum.org/v/index.html" in s)
    check("both are under File › Share", all(sid in w.registry.specs for sid in ("share.qr", "share.embed")))

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
