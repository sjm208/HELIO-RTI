"""The OpenLIME HSH export matches Relight's format and OpenLIME's reader.

Checks the HSH fit recovers known coefficients, the quantisation decodes
the way OpenLIME's shader does ((stored/255 - bias) * scale), the planes are
4:4:4 JPEGs, the tarzoom tiles sit where OpenLIME's LayoutTiles looks for
them (index, offsets, sizes), the low-memory banded fit equals the one-pass
fit, and - when RELIGHT_CLI points at relight-cli - that a synthetic capture
gives the same relit images as RelightLab's own HSH.

    python tests/test_openlime_hsh.py
    RELIGHT_CLI=/path/to/relight-cli python tests/test_openlime_hsh.py
"""
import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helio.io import openlime_hsh as O  # noqa: E402

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:                       # noqa: BLE001
        FAIL.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def dome(n_ring=12, elevations=(20, 40, 60)):
    out = []
    for j, el in enumerate(elevations):
        for k in range(n_ring):
            az, e = math.radians(k * 360 / n_ring + 15 * (j % 2)), math.radians(el)
            out.append((math.cos(e) * math.cos(az), math.cos(e) * math.sin(az), math.sin(e)))
    return np.array(out)


def capture(H=120, W=200, seed=0):
    """Lambertian, coloured, bumpy plate: frames (8-bit sRGB), lights, normals."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    z = 12 * np.exp(-((xx - 70) ** 2 + (yy - 60) ** 2) / (2 * 22 ** 2)) + 3 * np.sin(xx / 9.0)
    n = np.dstack([-np.gradient(z, axis=1), np.gradient(z, axis=0), np.ones_like(z)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    alb = np.clip(np.dstack([0.6 + 0.3 * np.sin(xx / 30), 0.5 + 0.2 * np.cos(yy / 20),
                             0.4 + 0.2 * np.sin((xx + yy) / 40)]), 0.05, 0.95)
    L = dome()
    frames = []
    for l in L:
        img = np.clip(alb * np.clip((n * l).sum(2), 0, None)[..., None] * 0.95 + 0.03, 0, 1)
        frames.append(np.rint(img ** (1 / 2.2) * 255).astype(np.uint8))
    return frames, L, n.astype(np.float32)


def openlime_qbox(idx):
    """LayoutTiles.initBoxes: tiles per level, level 0 the smallest."""
    w, h, out = idx["width"], idx["height"], [None] * idx["nlevels"]
    for lv in range(idx["nlevels"] - 1, -1, -1):
        out[lv] = (-(-w // idx["tilesize"]), -(-h // idx["tilesize"]))
        w >>= 1
        h >>= 1
    return out


def decode_planes(d, info, layout):
    m = info["materials"][0]
    sc, bi = np.array(m["scale"]).reshape(9, 3), np.array(m["bias"]).reshape(9, 3)
    out = []
    for p in range(9):
        if layout == "image":
            im = cv2.imread(os.path.join(d, f"plane_{p}.jpg"))[..., ::-1]
        else:
            im = stitch_top(os.path.join(d, f"plane_{p}"))
        out.append((im.astype(np.float64) / 255.0 - bi[p]) * sc[p])     # OpenLIME's decode
    return np.array(out)                                               # value / 255


def stitch_top(base):
    idx = json.load(open(base + ".tzi"))
    data = open(base + ".tzb", "rb").read()
    W, H, ts = idx["width"], idx["height"], idx["tilesize"]
    nx, ny = -(-W // ts), -(-H // ts)
    i = len(idx["offsets"]) - 1 - nx * ny
    out = np.zeros((H, W, 3), np.uint8)
    for y in range(ny):
        for x in range(nx):
            t = cv2.imdecode(np.frombuffer(data[idx["offsets"][i]:idx["offsets"][i + 1]], np.uint8), 1)
            out[y * ts:y * ts + t.shape[0], x * ts:x * ts + t.shape[1]] = t[..., ::-1]
            i += 1
    return out


frames, LIGHTS, NORMALS = capture()
H, W = frames[0].shape[:2]
TMP = tempfile.mkdtemp()


def _fit_exact():
    """Frames made from known HSH coefficients come back as those coefficients."""
    rng = np.random.default_rng(1)
    c = rng.normal(0, 30, (9, 8, 10, 3))
    c[0] += 300
    A = O.hsh_weights(LIGHTS)
    stack = np.tensordot(A, c, axes=(1, 0))            # (n, 8, 10, 3)
    got = O._accumulate(lambda k: stack[k], O.projector(LIGHTS))
    assert np.allclose(got, c, atol=1e-2), float(np.abs(got - c).max())


def _weights_relight():
    """Relight's lweights: the constant term, and theta clamped to pi/2 - 0.01."""
    w = O.hsh_weights(np.array([[0, 0, 1.0], [1.0, 0, 0]]))
    assert abs(w[0, 0] - 1 / math.sqrt(2 * math.pi)) < 1e-12
    assert np.allclose(w[0, [1, 3, 4, 5, 7, 8]], 0, atol=1e-6)            # overhead: no azimuth
    cT = math.cos(math.pi / 2 - 0.01)
    assert abs(w[1, 2] - math.sqrt(3 / (2 * math.pi)) * (2 * cT - 1)) < 1e-9


out_img = os.path.join(TMP, "img")
res_img = O.export(out_img, lambda k: frames[k], LIGHTS, H, W, layout="image", normals=NORMALS)
INFO = res_img["info"]


def _schema():
    assert {"width", "height", "format", "type", "colorspace", "lights", "nplanes", "quality",
            "materials"} <= set(INFO)
    assert (INFO["type"], INFO["colorspace"], INFO["nplanes"], INFO["format"]) == ("hsh", "rgb", 27, "jpg")
    assert len(INFO["materials"][0]["scale"]) == 27 and len(INFO["materials"][0]["bias"]) == 27
    assert len(INFO["lights"]) == 3 * len(LIGHTS)
    assert "normals" not in INFO          # a layer option in OpenLIME, not an info.json key


def _relights_frames():
    """Decoded as OpenLIME decodes, the fit gives back each photograph closely."""
    c = decode_planes(out_img, INFO, "image")
    for k in (0, 17, 30):
        r = np.tensordot(O.hsh_weights(LIGHTS[k:k + 1])[0], c, 1) * 255
        err = float(np.abs(r - frames[k]).mean())
        assert err < 3.0, f"frame {k}: {err:.2f}"


def _no_subsampling():
    from PIL import Image, JpegImagePlugin
    for p in ("plane_0.jpg", "plane_4.jpg", "normals.jpg"):
        assert JpegImagePlugin.get_sampling(Image.open(os.path.join(out_img, p))) == 0, p


def _normals_plane():
    n = cv2.imread(os.path.join(out_img, "normals.jpg"))[..., ::-1].astype(np.float64) / 127.5 - 1
    assert float(np.abs(n - NORMALS).mean()) < 0.02
    # +Y up: the bump's upper flank (row 40) faces up, its lower flank down.
    assert n[40, 70, 1] > 0.15 and n[80, 70, 1] < -0.15


out_tz = os.path.join(TMP, "tz")
res_tz = O.export(out_tz, lambda k: frames[k], LIGHTS, H, W, layout="tarzoom", normals=NORMALS)


def _tarzoom_index():
    for (w, h) in [(W, H), (256, 256), (257, 100), (513, 513), (3000, 2000), (4097, 1025)]:
        img = (np.random.default_rng(2).random((h, w, 3)) * 255).astype(np.uint8)
        idx, tiles = O.tarzoom_bytes(img, 85)
        qb = openlime_qbox(idx)
        assert sum(a * b for a, b in qb) == len(tiles) == len(idx["offsets"]) - 1, (w, h)
        assert idx["nlevels"] == O.n_levels(w, h)
        i = 0
        for lv, (tx, ty) in enumerate(qb):           # each tile the size OpenLIME maps it to
            il = idx["nlevels"] - 1 - lv
            side = idx["tilesize"] << il
            for y in range(ty):
                for x in range(tx):
                    t = cv2.imdecode(np.frombuffer(tiles[i], np.uint8), 1)
                    i += 1
                    assert abs(t.shape[1] - min(side, w - side * x) / (1 << il)) < 1, (w, h, lv, x, y)
                    assert abs(t.shape[0] - min(side, h - side * y) / (1 << il)) < 1, (w, h, lv, x, y)


def _tarzoom_files():
    for name in [f"plane_{p}" for p in range(9)] + ["normals"]:
        idx = json.load(open(os.path.join(out_tz, name + ".tzi")))
        assert idx["offsets"][-1] == os.path.getsize(os.path.join(out_tz, name + ".tzb")), name
        assert (idx["width"], idx["height"], idx["overlap"], idx["format"]) == (W, H, 0, "jpg")


def _tarzoom_same_planes():
    a = decode_planes(out_img, INFO, "image")
    b = decode_planes(out_tz, res_tz["info"], "tarzoom")
    assert float(np.abs(a - b).mean()) * 255 < 0.6


def _too_thin():
    try:
        O.tarzoom_bytes(np.zeros((1, 1000, 3), np.uint8), 90)
    except ValueError:
        return
    raise AssertionError("a 1000 x 1 tarzoom should be refused")


def _banded_equals_single():
    a, mn_a, mx_a = O.fit_quantised(lambda k: frames[k], LIGHTS, H, W, budget=1 << 40)
    b, mn_b, mx_b = O.fit_quantised(lambda k: frames[k], LIGHTS, H, W, budget=200_000)
    assert np.allclose(mn_a, mn_b, rtol=0.05, atol=0.5) and np.allclose(mx_a, mx_b, rtol=0.05, atol=0.5)
    assert float(np.abs(a.astype(int) - b.astype(int)).mean()) < 1.5


def _clustered_lights_refused():
    try:
        O.projector(np.tile([[0.1, 0.1, 0.99]], (12, 1)))
    except ValueError:
        return
    raise AssertionError("twelve identical lights should be refused")


# Nine lights in one ring, 32-48 degrees up, as a chrome-sphere capture of a
# binding: the exact nine-term fit has a condition number in the thousands.
NINE = np.array([[-0.633, -0.23, 0.739], [-0.311, -0.726, 0.614], [0.176, -0.82, 0.545],
                 [0.591, -0.595, 0.545], [0.825, -0.197, 0.529], [0.642, 0.49, 0.59],
                 [0.081, 0.703, 0.706], [-0.428, 0.679, 0.597], [-0.771, -0.118, 0.626]])


def _nine_images():
    """Nine photographs export, decode to sane values and relight like the scene."""
    assert O.is_regularised(NINE)
    alb = np.array([0.55, 0.35, 0.2])
    shade = lambda l: np.clip((NORMALS * (np.asarray(l) / np.linalg.norm(l))).sum(2), 0, None)
    f9 = [np.rint(np.clip(alb * shade(l)[..., None], 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
          for l in NINE]
    d = os.path.join(TMP, "nine")
    info = O.export(d, lambda k: f9[k], NINE, H, W, layout="image")["info"]
    sc = np.array(info["materials"][0]["scale"])
    assert np.all(np.isfinite(sc)) and sc.max() < 20, sc.max()   # not 1000s, as before
    c = decode_planes(d, info, "image")
    for k in range(9):                          # each photograph comes back
        r = np.tensordot(O.hsh_weights(NINE[k:k + 1])[0], c, 1) * 255
        err = float(np.abs(r - f9[k]).mean())
        assert err < 6.0, f"frame {k}: {err:.2f}"
    for l in ([0.0, 0.0, 1.0], [0.5, 0.5, 0.71], [-0.6, 0.2, 0.77]):   # and lights between
        r = np.tensordot(O.hsh_weights(np.array([l]))[0], c, 1) * 255
        want = np.clip(alb * shade(l)[..., None], 0, 1) ** (1 / 2.2) * 255
        assert 20 < r.mean() < 250, f"light {l}: mean {r.mean():.1f}"
        assert float(np.abs(r - want).mean()) < 25, f"light {l}: {np.abs(r - want).mean():.1f}"


def _dome_not_regularised():
    assert not O.is_regularised(LIGHTS)


def _launcher():
    """The folder opens itself: serve.py answers range requests; the page explains a black screen."""
    import socket
    import threading
    import urllib.request
    for f in ("serve.py", "Open viewer.command", "Open viewer.bat"):
        assert os.path.isfile(os.path.join(out_tz, f)), f
    assert os.access(os.path.join(out_tz, "Open viewer.command"), os.X_OK)
    html = open(os.path.join(out_tz, "index.html")).read()
    assert "location.protocol === 'file:'" in html and "r.status !== 206" in html
    ns = {"__name__": "serve"}
    exec(compile(open(os.path.join(out_tz, "serve.py")).read(), "serve.py", "exec"), ns)
    import functools
    httpd = ns["http"].server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(ns["Handler"], directory=out_tz))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        tzb = os.path.join(out_tz, "plane_0.tzb")
        req = urllib.request.Request(f"http://127.0.0.1:{port}/plane_0.tzb", headers={"Range": "bytes=10-19"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206 and r.read() == open(tzb, "rb").read()[10:20]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/info.json") as r:
            assert r.status == 200 and json.loads(r.read())["type"] == "hsh"
    finally:
        httpd.shutdown()


def _views_menu():
    """The viewer offers the RTI, the normals and each map, one at a time, maps scaled onto the RTI."""
    small = (np.random.default_rng(3).random((H // 2, W // 2)) * 65535).astype(np.uint16)
    url, layout, w = O.web_view(small, os.path.join(TMP, "web"), "rough")
    assert layout == "tarzoom" and url == "rough.tzi" and w == W // 2, (layout, url, w)
    assert O.web_view(np.zeros((1, 1000), np.uint8), os.path.join(TMP, "web"), "thin")[1] == "image"   # too thin to tile: one JPEG
    html = O.viewer_html("tarzoom", True, "T", [dict(id="rough", label="Roughness", url="maps/web/" + url,
                                                     layout=layout, z=W / w)])
    views = json.loads(html.split("const VIEWS = ", 1)[1].split(";\n", 1)[0])
    assert [v["id"] for v in views] == ["RTI", "normals", "rough"], views
    assert views[2]["z"] == 2.0 and views[1]["url"] == "normals.tzi"
    assert "label: v.label" in html and "ui.actions.layers.display = VIEWS.length > 1" in html
    assert ".openlime-mode-entry" in html                     # the RTI's broken modes are not offered
    b = O.viewer_html("tarzoom", True, "T", [], dict(kd="maps/web/brdf_kd.tzi", normals="maps/web/brdf_normals.tzi",
                                                   ks="maps/web/brdf_ks.tzi", gloss="maps/web/brdf_gloss.tzi",
                                                   layout="tarzoom", z=1.0))
    bv = json.loads(b.split("const VIEWS = ", 1)[1].split(";\n", 1)[0])
    assert [(v["id"], v.get("mode")) for v in bv if v.get("brdf")] == [
        ("relit_shine", "specular"), ("relit_clay", "monochrome")], bv
    assert "l.shader.innerCode = BRDF_CODE[v.mode]" in b and "pow(nh, ex)" in b and "pow(NdotL, 4.0)" in b
    nos = O.viewer_html("tarzoom", True, "T", [], dict(kd="a.tzi", normals="n.tzi", layout="tarzoom", z=1.0))
    assert "relit_shine" not in nos and "relit_clay" in nos          # no specular map: no shine view
    plain = O.viewer_html("tarzoom", False)
    assert "ui.actions.layers.display = VIEWS.length > 1" in plain and '"normals"' not in plain


def _relight_cli():
    cli = os.environ.get("RELIGHT_CLI", "")
    if not cli:
        print("        (RELIGHT_CLI not set: skipped the comparison with relight-cli)")
        return
    src, ref = os.path.join(TMP, "src"), os.path.join(TMP, "ref")
    os.makedirs(src)
    with open(os.path.join(src, "lights.lp"), "w") as f:
        f.write(f"{len(LIGHTS)}\n")
        for k, (img, l) in enumerate(zip(frames, LIGHTS)):
            nm = f"f{k:03d}.jpg"
            cv2.imwrite(os.path.join(src, nm), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
            f.write(f"{nm} {l[0]:.6f} {l[1]:.6f} {l[2]:.6f}\n")
    subprocess.run([cli, "-b", "hsh", "-p", "27", src, ref], check=True, capture_output=True)
    rinfo = json.load(open(os.path.join(ref, "info.json")))
    assert set(rinfo) == set(INFO), set(rinfo) ^ set(INFO)
    rc = decode_planes(ref, rinfo, "image")
    oc = decode_planes(out_img, INFO, "image")
    for l in ([0.3, -0.2, 0.93], [-0.6, 0.5, 0.62], [0.8, 0.1, 0.59]):
        w = O.hsh_weights(np.array([l]))[0]
        d = float(np.abs(np.tensordot(w, oc, 1) - np.tensordot(w, rc, 1)).mean() * 255)
        assert d < 1.5, f"light {l}: ours and relight-cli differ by {d:.2f}/255"


for name, fn in [("HSH fit recovers known coefficients", _fit_exact),
                 ("Relight's HSH basis", _weights_relight),
                 ("info.json is Relight's hsh/rgb schema", _schema),
                 ("decoded planes give back the photographs", _relights_frames),
                 ("planes are 4:4:4 JPEGs", _no_subsampling),
                 ("normals plane is (n+1)/2, +Y up", _normals_plane),
                 ("tarzoom tiles where OpenLIME looks", _tarzoom_index),
                 ("tarzoom offsets match the .tzb", _tarzoom_files),
                 ("tarzoom and single-image planes agree", _tarzoom_same_planes),
                 ("too-thin tarzoom refused", _too_thin),
                 ("banded fit equals one-pass fit", _banded_equals_single),
                 ("clustered lights refused", _clustered_lights_refused),
                 ("nine photographs in a ring relight", _nine_images),
                 ("a full dome keeps Relight's exact fit", _dome_not_regularised),
                 ("the folder opens itself (range server, launcher, a reason not black)", _launcher),
                 ("a Views menu: RTI, relit maps, normals and the maps, scaled onto the RTI", _views_menu),
                 ("same relit images as relight-cli", _relight_cli)]:
    check(name, fn)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
