"""
Background renderer for the live preview.

The canvas shows up to three layers: a whole-surface overview (context while
panning), a detail render of exactly what is on screen - at up to 1:1
full-resolution pixels, magnified beyond that - and, while you adjust
settings, a quick half-resolution draft of the view that appears first. Both are computed here on one worker thread.
Requests are coalesced (only the newest per layer is rendered) and NumPy/SciPy
release the GIL, so QGIS stays responsive while you drag, zoom and pan.
Results come back through the `ready` signal, delivered on the main thread.
"""

import json
import math
import threading
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from qgis.PyQt.QtCore import QObject, pyqtSignal

from . import detail
from . import raster_io as rio
from . import relief_engine as eng

OVERVIEW_DIM = 1200            # long edge of the whole-surface overview, px
DETAIL_MAX_PIXELS = 3_000_000  # cap on one detail render (HiDPI screens), px


class _LRU:
    """Small thread-safe LRU cache. Values are computed outside the lock."""

    def __init__(self, size):
        self.size, self._d, self._lock = size, OrderedDict(), threading.Lock()

    def get(self, key, make):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        value = make()
        with self._lock:
            self._d[key] = value
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return value


def prep_key(p):
    return json.dumps({k: p.get(k, eng.DEFAULT_PARAMS.get(k)) for k in eng.PREPARE_KEYS}, sort_keys=True)


def uses_normals(p):
    """True when a render with these parameters is shaded from the normal map."""
    return p.get("mode") == "clean" and p.get("clean_source") == "normals"


class PreviewRenderer(QObject):
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    busyChanged = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sync = False              # tests render inline
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="helio-relief-preview")
        self._lock = threading.Lock()
        self._pending = {}
        self._busy = False
        self._reads = _LRU(10)
        self._preps = _LRU(6)
        self._stats = _LRU(32)
        self._strip_preps = _LRU(40)

    # ---- scheduling
    def submit(self, req):
        if self.sync:
            self._run(req)
            return
        with self._lock:
            self._pending[req["kind"]] = req
            if self._busy:
                return
            self._busy = True
        self.busyChanged.emit(True)
        self._next()

    def _next(self, _future=None):
        with self._lock:
            req = (self._pending.pop("draft", None) or self._pending.pop("detail", None)
                   or self._pending.pop("overview", None) or self._pending.pop("lines", None)
                   or self._pending.pop("strip", None))
            if req is None:
                self._busy = False
        if req is None:
            self.busyChanged.emit(False)
            return
        try:
            self._pool.submit(self._run, req).add_done_callback(self._next)
        except RuntimeError:           # pool already shut down
            with self._lock:
                self._busy = False

    def _run(self, req):
        try:
            run = {"strip": self.compute_strip, "lines": self.compute_lines}.get(req["kind"], self.compute)
            self.ready.emit(run(req))
        except Exception:
            try:
                self.failed.emit(traceback.format_exc())
            except RuntimeError:
                pass

    def shutdown(self):
        with self._lock:
            self._pending.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- shared normalisation
    def stats(self, layer_key, ref, ref_origin, plane, dx, dy, p, texture, occlusion, curvature=False,
              normals_ref=None, normals_key=None):
        """Global tone constants from a full-resolution reference crop, shared by
        previews and final renders so every view gets identical tones. With a
        clean relief shaded from the normal map, normals_ref is that crop of it."""
        if not (texture or occlusion or curvature or p["mode"] in ("dome", "lab", "clean")):
            return {}
        # band-shading constants are computed for every band and shade kind, so they
        # do not depend on which shade each band currently uses
        pk = dict(p, band_types=None, band_enabled=None) if p["mode"] == "bands" else p
        nm = normals_ref if uses_normals(p) else None
        key = (layer_key, prep_key(pk), texture, occlusion, curvature, dx, dy, normals_key if nm is not None else None)
        return self._stats.get(key, lambda: eng.prepare(ref, dx, dy, p, 1.0, None, plane, ref_origin,
                                                        texture=texture, occlusion=occlusion,
                                                        curvature=curvature, normals=nm).stats)

    # ---- rendering (worker thread)
    def compute(self, req):
        t0 = time.perf_counter()
        W, H = req["full"]
        vx, vy, vw, vh = req["view"]
        pxs = float(req["px_scale"])
        p, tab, stack = req["params"], req["tab"], req.get("stack")
        if tab == "stack":
            p = dict(p, mode="hillshade", azimuth=stack["azimuth"], altitude=stack["altitude"], contrast=1.0,
                     gamma=1.0, tint=(1.0, 1.0, 1.0))
            if stack["raw"]:
                p.update(detrend=False, flatten_sigma=0.0, eq_enabled=False)
        p_full = p                                               # tone constants always from full settings
        if req["kind"] == "draft":
            p = dict(p, svf_dirs=min(8, int(p["svf_dirs"])))     # drafts trade a little accuracy for speed
        if req["kind"] == "overview":
            margin = 0
        elif tab == "frequency":
            margin = int(min(1024, 3 * max(req["split"]["sigmas"]) + 8))
        elif tab == "separate":
            margin = detail.margin_px(req["detail"])
        else:
            margin = eng.spectral_margin_px(p, precise=False)

        # Read the view plus a margin of context so filters match a full render.
        wx0, wy0 = max(0, vx - margin), max(0, vy - margin)
        wx1, wy1 = min(W, vx + vw + margin), min(H, vy + vh + margin)
        win = (int(wx0), int(wy0), int(wx1 - wx0), int(wy1 - wy0))
        out = (max(1, int(round(win[2] / pxs))), max(1, int(round(win[3] / pxs))))
        sx, sy = win[2] / float(out[0]), win[3] / float(out[1])
        z = self._reads.get(("z", req["layer_key"], win, out), lambda: rio.read_array(req["path"], win, out)[0])
        dx, dy = req["dx"] * sx, req["dy"] * sy
        origin = (win[0], win[1])

        spectrum = None
        if tab == "frequency":
            img = self._frequency(req, z, dx, dy, sx, origin)
        elif tab == "separate":
            d = detail.params(req["detail"])
            band, removed, st = self._band(req, z, sx, origin)
            img = detail.render(band, removed, d, p, dx, dy, st, sx)
            if req["kind"] != "draft":
                spectrum = self.spectrum(req)
        else:
            lit = p["mode"] in eng.LIT_MODES
            tex, svf, curv = lit or p["mode"] == "texture", eng.needs_svf(p), lit
            npath = req.get("normals") if uses_normals(p) else None
            nm = (self._reads.get(("n", npath, win, out), lambda: rio.read_normals(npath, req["full"], win, out))
                  if npath else None)
            st = self.stats(req["layer_key"], req["ref"], req["ref_origin"], req["plane"], req["dx"], req["dy"],
                            p_full, tex, svf, curv, req.get("ref_normals"), npath)
            key = ("prep", req["layer_key"], win, out, prep_key(p), tex, svf, curv, dx, dy, npath)
            prep = self._preps.get(key, lambda: eng.prepare(z, dx, dy, p, sx, st, req["plane"], origin,
                                                            texture=tex, occlusion=svf, curvature=curv, normals=nm))
            img = eng.shade(prep, p, component=req.get("component"))
            if tab == "stack":
                if stack["tone"] == 1:
                    img = np.maximum(img, np.float32(stack["lift"]))
                elif stack["tone"] == 2:
                    img = eng.tone(img, math.sin(math.radians(stack["altitude"])), 1.25, 1.0, soft=True)
            if req.get("albedo") and req.get("colour"):
                grey = img if img.ndim == 2 else img.mean(axis=2)
                img = eng.colour_composite(grey, self._colour(req, win, out), p, req["albedo_strength"])
        cmp = self._compare(req, z, dx, dy, sx, win, out, origin)

        # Crop the margin away again.
        ox = min(max(int(round((vx - win[0]) / sx)), 0), out[0] - 1)
        oy = min(max(int(round((vy - win[1]) / sy)), 0), out[1] - 1)
        ow = max(1, min(int(round(vw / sx)), out[0] - ox))
        oh = max(1, min(int(round(vh / sy)), out[1] - oy))
        sl = (slice(oy, oy + oh), slice(ox, ox + ow))
        img = np.ascontiguousarray(img[sl])
        grey = img if img.ndim == 2 else img.mean(axis=2)
        return dict(kind=req["kind"], gen=req["gen"], img=img,
                    cmp=np.ascontiguousarray(cmp[sl]) if cmp is not None else None,
                    probe=np.ascontiguousarray(z[sl]), rect=(win[0] + ox * sx, win[1] + oy * sy, ow * sx, oh * sy),
                    px_scale=sx, clip=float(((grey <= 0.5 / 255) | (grey >= 254.5 / 255)).mean() * 100.0),
                    spectrum=spectrum, ms=(time.perf_counter() - t0) * 1000.0)

    # ---- the Detail tab and live line tracing
    def _band(self, req, z, sx, origin):
        """The Detail tab's bands of a window (band, removed, stats), stats from the full-resolution reference."""
        d = detail.params(req["detail"])
        skey = json.dumps({k: v for k, v in d.items() if k not in ("display", "gain")}, sort_keys=True)
        model = self._stats.get(("detail_model", req["layer_key"]), lambda: detail.reference_model(req["ref"]))
        st = self._stats.get(("detail", req["layer_key"], skey, req["dx"], req["dy"]),
                             lambda: detail.detail_stats(req["ref"], d, req["dx"], req["dy"], req["plane"],
                                                         req["ref_origin"], model))
        band, removed = detail.isolate(z, d, sx, st, req["plane"], origin)
        return band, removed, st

    def spectrum(self, req):
        """(sizes px, power, noise model, crop) of a full-resolution crop at the middle of the view (up to 1024 px
        square), so the chart follows what you look at."""
        W, H = req["full"]
        vx, vy, vw, vh = req["view"]
        n = int(min(1024, W, H))
        x0 = int(min(max(round((vx + vw / 2.0 - n / 2.0) / 64.0) * 64, 0), W - n))
        y0 = int(min(max(round((vy + vh / 2.0 - n / 2.0) / 64.0) * 64, 0), H - n))

        def make():
            crop = rio.read_array(req["path"], (x0, y0, n, n))[0]
            s, pw = detail.radial_spectrum(crop)
            return s, pw, detail.fit_noise(s, pw), (x0, y0, n)
        return self._stats.get(("spectrum", req["layer_key"], x0, y0, n), make)

    def compute_lines(self, req):
        """Lines traced over the view (plus a margin), for the live overlay: [(kind, [(x, y) full-res px])]."""
        from . import tools
        t0 = time.perf_counter()
        W, H = req["full"]
        vx, vy, vw, vh = req["view"]
        tr = req["trace"]
        k = max(1.0, math.sqrt(vw * vh / float(tr["max_px"])))
        m = int(4 * tr["width"] + 8 + (detail.margin_px(req["detail"]) if req.get("detail") else 0))
        wx0, wy0 = max(0, int(vx - m)), max(0, int(vy - m))
        win = (wx0, wy0, int(min(W, vx + vw + m)) - wx0, int(min(H, vy + vh + m)) - wy0)
        out = (max(2, int(round(win[2] / k))), max(2, int(round(win[3] / k))))
        kx, ky = win[2] / float(out[0]), win[3] / float(out[1])
        z = self._reads.get(("z", req["layer_key"], win, out), lambda: rio.read_array(req["path"], win, out)[0])
        if req.get("detail"):
            z = self._band(req, z, kx, (win[0], win[1]))[0]
        lines, _ = tools.trace_lines(z, tr["width"] / kx, tr["sensitivity"], tr["min_length"] / kx, tr["kind"],
                                     tr["simplify"] / kx, bridge_px=tr["bridge"] / kx, smooth=tr["smooth"],
                                     min_strength=tr["min_strength"])
        overlay = [(l["kind"], [(win[0] + (x + 0.5) * kx, win[1] + (y + 0.5) * ky) for x, y in l["points"]])
                   for l in lines]
        inside = sum(1 for _k, pts in overlay if any(vx <= x <= vx + vw and vy <= y <= vy + vh for x, y in pts))
        return dict(kind="lines", gen=req["gen"], lines=overlay, count=inside, k=kx,
                    ms=(time.perf_counter() - t0) * 1000.0)

    def compute_strip(self, req):
        """Thumbnails of the current view, one per item (a look, a band or a
        light). Each is normalised on its own thumbnail - fast, and plenty for
        choosing between shades."""
        t0 = time.perf_counter()
        win, out = tuple(req["view"]), tuple(req["out"])
        z = self._reads.get(("z", req["layer_key"], win, out), lambda: rio.read_array(req["path"], win, out)[0])
        sx, sy = win[2] / float(out[0]), win[3] / float(out[1])
        dx, dy = req["dx"] * sx, req["dy"] * sy
        items = []
        for it in req["items"]:
            p = it["params"]
            if it.get("detail") is not None:          # the Detail tab: the band each way it can be shown
                d = detail.params(it["detail"])
                bkey = ("strip_band", req["layer_key"], win, out,
                        json.dumps({k: v for k, v in d.items() if k not in ("display", "gain")}, sort_keys=True))
                band, removed, st = self._strip_preps.get(
                    bkey, lambda d=d: self._band(dict(req, detail=d), z, sx, (win[0], win[1])))
                items.append((it["key"], it["label"],
                              np.ascontiguousarray(detail.render(band, removed, d, p, dx, dy, st, sx))))
                continue
            lit = p["mode"] in eng.LIT_MODES
            npath = req.get("normals") if uses_normals(p) and req.get("full") else None
            nm = (self._reads.get(("n", npath, win, out), lambda: rio.read_normals(npath, req["full"], win, out))
                  if npath else None)
            key = ("strip", req["layer_key"], win, out, prep_key(p), dx, dy, npath)
            prep = self._strip_preps.get(key, lambda p=p, lit=lit, nm=nm: eng.prepare(
                z, dx, dy, p, sx, None, req["plane"], (win[0], win[1]), texture=lit or p["mode"] == "texture",
                curvature=lit, normals=nm))
            items.append((it["key"], it["label"], np.ascontiguousarray(eng.shade(prep, p, component=it.get("component")))))
        return dict(kind="strip", gen=req["gen"], items=items, ms=(time.perf_counter() - t0) * 1000.0)

    def _colour(self, req, win, out):
        return self._reads.get(("rgb", req["colour"], win, out),
                               lambda: rio.read_rgb(req["colour"], req["full"], win, out))

    def _compare(self, req, z, dx, dy, sx, win, out, origin):
        what = req.get("compare")
        if what == "colour":
            return self._colour(req, win, out) if req.get("colour") else None
        if what == "hillshade":
            p = dict(req["params"], mode="hillshade", detrend=False, flatten_sigma=0.0, eq_enabled=False,
                     contrast=1.0, gamma=1.0, tint=(1.0, 1.0, 1.0))
            key = ("prep", req["layer_key"], win, out, prep_key(p), False, False, dx, dy)
            prep = self._preps.get(key, lambda: eng.prepare(z, dx, dy, p, sx, {}, None, origin,
                                                            texture=False, occlusion=False))
            return eng.shade(prep, p)
        if what == "height":
            zf = eng.fill_nodata(z)[0]
            if req["plane"] is not None:
                zf = zf - eng.plane_surface(req["plane"], zf.shape, origin, sx)
            lo, hi = np.percentile(zf[::2, ::2], [1.0, 99.0])
            return np.clip((zf - lo) / max(hi - lo, 1e-12), 0.0, 1.0).astype(np.float32)
        return None

    def _frequency(self, req, z, dx, dy, sx, origin):
        s = req["split"]
        zc = eng.condition_surface(eng.fill_nodata(z)[0], s, sx, req["plane"], origin)
        bands = eng.post_split(eng.frequency_split(zc, [v / sx for v in s["sigmas"]]), s, sx)
        shade_p = dict(req["params"], mode="studio", texture_mix=0.0, occlusion=0.0, flatten_sigma=0.0,
                       eq_enabled=False, detrend=False, tint=(1.0, 1.0, 1.0))
        band = req["band"]
        if band == 0:
            surf = sum((bands[k] * np.float32(s["gains"][k]) for k in s["enabled"]), np.zeros_like(zc))
            return eng.render(surf, dx, dy, shade_p, sx)[0]
        key = eng.BAND_KEYS[band - 1]
        b = bands[key] * np.float32(s["gains"][key])
        return eng.ramp_overlay(b, eng.render(b, dx, dy, shade_p, sx)[0], req["ramp_stops"])
