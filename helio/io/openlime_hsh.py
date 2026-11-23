"""RGB HSH RTI in the Relight / OpenLIME web format, single-image or tarzoom.

This is what RelightLab writes with ``relight-cli -b hsh -p 27`` and what the
OpenLIME ``rti`` layer reads (``type: 'rti', layout: 'tarzoom'``): nine
second-order hemispherical-harmonic coefficients per colour channel, 27 in
all, packed as nine RGB JPEGs - plane k holds coefficient k for R, G and B.

The files must be read by RelightLab and OpenLIME exactly as their own, so
each detail below was checked against Relight's public source
(cnr-isti-vclab/relight, whose core is LGPL-3.0) and is written here in
HELIO's own code; nothing is a translation of theirs:

* basis    - the nine second-order hemispherical harmonics (Gautron, Krivanek,
             Pattanaik and Bouatouch, EGSR 2004), with the normalisation and
             the zenith clamp (pi/2 - 0.01) Relight's HSH files use; fitted
             by least squares on the 8-bit sRGB frames - except that lights
             too few or too alike in height for nine terms (a nine-frame
             ring, say) get a ridge-regularised fit instead; see
             :func:`projector`.
* quantise - each of the 27 plane-channels gets its own min/max from the
             0.25% and 99.75% quantiles, then ``scale = (max - min) / 255``
             and ``bias = -min / (max - min)``, so the viewer recovers
             ``value/255 = (stored/255 - bias) * scale`` (info.json).
* JPEG     - no chroma subsampling: each channel is a different
             coefficient, and 4:2:0 would blur two of them together.
* tarzoom  - 256 px tiles, no overlap, levels halved (floor) until both
             sides fit one tile, a 1-4-6-4-1 binomial filter down to each
             level, tiles stored smallest level first and row by row, and
             ``plane_N.tzi`` holding the cumulative byte ``offsets``.

The format: Ponchio, Corsini and Scopigno, "RELIGHT: a compact and accurate
RTI representation for the web", Graphical Models 105 (2019).

Lights are +Y up, as in a .lp file and in HELIO; normals, when given, are
written as Relight's ``normals`` plane: (n + 1) / 2 in R, G, B, +Y up.
"""
from __future__ import annotations

import json
import math
import os
from typing import Callable

import cv2
import numpy as np

#: Relight's fit-side clamp on the light's zenith angle (src/rti.cpp).
THETA_CLAMP = math.pi / 2.0 - 0.01
#: Relight's default quantile for the coefficient range (RtiBuilder::rangeQuantile).
RANGE_QUANTILE = 0.995
#: Above this condition number of the light matrix the fit is regularised.
COND_LIMIT = 30.0
TILE_SIZE = 256
OVERLAP = 0
LAYOUTS = ("tarzoom", "image")


# ── basis and fit ────────────────────────────────────────────────────

def hsh_weights(lights: np.ndarray) -> np.ndarray:
    """(n, 9) HSH weights of each light: the basis HSH files are written in."""
    L = np.asarray(lights, np.float64)[:, :3]
    L = L / np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-12)
    lx, ly = L[:, 0], L[:, 1]
    lz = np.sqrt(np.clip(1.0 - lx * lx - ly * ly, 0.0, 1.0))
    phi = np.arctan2(ly, lx)
    phi = np.where(phi < 0.0, phi + 2.0 * math.pi, phi)
    theta = np.minimum(np.arccos(np.clip(lz, -1.0, 1.0)), THETA_CLAMP)
    cP, sP, cT = np.cos(phi), np.sin(phi), np.cos(theta)
    cT2 = cT * cT
    r = np.sqrt(np.clip(cT - cT2, 0.0, None))
    pi = math.pi
    return np.stack([
        np.full_like(lx, 1.0 / math.sqrt(2.0 * pi)),
        math.sqrt(6.0 / pi) * cP * r,
        math.sqrt(3.0 / (2.0 * pi)) * (-1.0 + 2.0 * cT),
        math.sqrt(6.0 / pi) * r * sP,
        math.sqrt(30.0 / pi) * np.cos(2.0 * phi) * (-cT + cT2),
        math.sqrt(30.0 / pi) * cP * (-1.0 + 2.0 * cT) * r,
        math.sqrt(5.0 / (2.0 * pi)) * (1.0 - 6.0 * cT + 6.0 * cT2),
        math.sqrt(30.0 / pi) * (-1.0 + 2.0 * cT) * r * sP,
        math.sqrt(30.0 / pi) * (-cT + cT2) * np.sin(2.0 * phi),
    ], axis=1)


def projector(lights: np.ndarray) -> np.ndarray:
    """(9, n): coefficients = projector @ samples.

    Relight's plain least squares, (A^T A)^-1 A^T, when the lights pin all
    nine terms down (condition number up to ``COND_LIMIT``).  Otherwise -
    nine photographs in one ring, or lights at only two or three heights -
    the terms that depend only on the light's height (0, 2 and 6) are
    nearly the same column, the exact fit cancels coefficients in the
    hundreds of thousands against each other, and eight-bit planes of those
    relight to black or noise.  Then the fit is ridge-regularised:
    (A^T A + lam D)^-1 A^T, with D leaving the constant term free and lam
    = (largest singular value / COND_LIMIT)^2, which keeps the photographs
    to about a level of 1-2 in 255 and relights sensibly in between.
    """
    A = hsh_weights(lights)
    if len(A) < 9:
        raise ValueError(f"HSH needs at least 9 photographs, got {len(A)}")
    if np.linalg.cond(A[:, [0, 1, 3]]) > 1e3:
        raise ValueError("The lights are too few or too clustered to fit HSH.")
    s = np.linalg.svd(A, compute_uv=False)
    AtA = A.T @ A
    if s[-1] > 0 and s[0] / s[-1] <= COND_LIMIT:
        return np.linalg.solve(AtA, A.T)
    lam = (s[0] / COND_LIMIT) ** 2
    return np.linalg.solve(AtA + lam * np.diag([0.0] + [1.0] * 8), A.T)


def is_regularised(lights: np.ndarray) -> bool:
    """True when :func:`projector` has to regularise these lights."""
    s = np.linalg.svd(hsh_weights(lights), compute_uv=False)
    return not (s[-1] > 0 and s[0] / s[-1] <= COND_LIMIT)


def relight(coeffs: np.ndarray, light) -> np.ndarray:
    """Render (9, h, w, 3) coefficients under one light, in 0..255 units."""
    w = hsh_weights(np.asarray(light, np.float64).reshape(1, 3))[0]
    return np.tensordot(w, coeffs, axes=(0, 0))


def _accumulate(read: Callable[[int], np.ndarray], P: np.ndarray, rows=None,
                progress=None, frac=(0.0, 1.0), check=None) -> np.ndarray:
    """(9, h, w, 3) float32 coefficients over *rows* (a slice), 0..255 units."""
    n = P.shape[1]
    acc = tmp = None
    for k in range(n):
        if check is not None:
            check()
        img = read(k)
        if rows is not None:
            img = img[rows]
        img = img.astype(np.float32)
        if acc is None:
            acc = np.zeros((9,) + img.shape, np.float32)
            tmp = np.empty_like(img)
        for p in range(9):
            np.multiply(img, np.float32(P[p, k]), out=tmp)
            acc[p] += tmp
        if progress:
            progress("Fitting HSH…", frac[0] + (frac[1] - frac[0]) * (k + 1) / n)
    return acc


def _ranges(coeffs: np.ndarray, quantile: float = RANGE_QUANTILE, most: int = 4_000_000):
    """Per plane-channel (min, max): the 0.25% and 99.75% quantiles, (27,), (27,)."""
    flat = coeffs.reshape(9, -1, 3)
    npx = flat.shape[1]
    if npx > most:
        idx = np.random.default_rng(0).choice(npx, most, replace=False)
        idx.sort()
        flat = flat[:, idx]
    n = flat.shape[1]
    lo_i = min(int((1.0 - quantile) / 2.0 * n), n - 1)
    hi_i = min(int((1.0 + quantile) / 2.0 * n), n - 1)
    mins, maxs = np.empty(27), np.empty(27)
    for p in range(9):
        for c in range(3):
            v = np.partition(flat[p, :, c], (lo_i, hi_i))
            lo, hi = float(v[lo_i]), float(v[hi_i])
            if not (hi > lo):
                hi = lo + 1e-6
            mins[p * 3 + c], maxs[p * 3 + c] = lo, hi
    return mins, maxs


def _quantise(coeffs: np.ndarray, mins: np.ndarray, maxs: np.ndarray, out: np.ndarray) -> None:
    """Coefficients (9, h, w, 3) -> uint8 into *out*, by the format's scale and bias."""
    span = (maxs - mins).reshape(9, 3)
    lo = mins.reshape(9, 3)
    for p in range(9):
        q = (coeffs[p] - lo[p]) * (255.0 / span[p])
        np.clip(np.rint(q, out=q), 0, 255, out=q)
        out[p] = q


def scale_bias(mins: np.ndarray, maxs: np.ndarray) -> tuple[list, list]:
    span = maxs - mins
    return ([round(float(s / 255.0), 7) for s in span],
            [round(float(-m / s), 6) for m, s in zip(mins, span)])


def fit_quantised(read: Callable[[int], np.ndarray], lights: np.ndarray, h: int, w: int,
                  budget: int | None = None, progress=None, check=None):
    """Fit and quantise.  Returns (planes uint8 (9, h, w, 3), mins, maxs).

    One pass over the photographs when the float coefficients fit in
    *budget* bytes; otherwise a sampling pass for the ranges and then the
    image in bands, each band reading the photographs again.
    """
    P = projector(lights)
    need = 9 * h * w * 3 * 4 + 2 * h * w * 3 * 4
    if budget is None:
        from helio.utils.memory_plan import system_memory
        budget = int(system_memory()[1] * 0.5)
    planes = np.empty((9, h, w, 3), np.uint8)
    if need <= budget:
        acc = _accumulate(read, P, None, progress, (0.0, 0.85), check)
        mins, maxs = _ranges(acc)
        _quantise(acc, mins, maxs, planes)
        return planes, mins, maxs
    step = max(1, int(math.ceil(h * w / 2_000_000)))
    sample = _accumulate(read, P, slice(0, h, step), progress, (0.0, 0.15), check)
    mins, maxs = _ranges(sample)
    del sample
    band = max(16, int(budget // max(1, need // h)))
    starts = list(range(0, h, band))
    for i, y0 in enumerate(starts):
        y1 = min(h, y0 + band)
        f0 = 0.15 + 0.7 * i / len(starts)
        acc = _accumulate(read, P, slice(y0, y1), progress,
                          (f0, f0 + 0.7 / len(starts)), check)
        _quantise(acc, mins, maxs, planes[:, y0:y1])
        del acc
    return planes, mins, maxs


# ── images and tiles ─────────────────────────────────────────────────

def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """RGB uint8 -> JPEG bytes, 4:4:4 (no chroma subsampling)."""
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[..., ::-1]),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality),
                            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                            int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)])
    if not ok:
        raise OSError("JPEG encoding failed")
    return buf.tobytes()


def n_levels(w: int, h: int, tile: int = TILE_SIZE) -> int:
    """The pyramid's levels: the full image, then halvings (floor) until both
    sides fit one tile.  k halvings fit when max(w, h) >> k <= tile, that is
    max(w, h) // (tile + 1) < 2**k."""
    return 1 + (max(int(w), int(h)) // (int(tile) + 1)).bit_length()


def halve(img: np.ndarray) -> np.ndarray:
    """Half size (floor): a 1-4-6-4-1 binomial filter, sampled at 2x+1, edges clamped."""
    h, w = img.shape[:2]
    if w < 2 or h < 2:
        return img[:max(1, h >> 1), :max(1, w >> 1)].copy()
    k = np.array([1, 4, 6, 4, 1], np.float32) / 16.0
    f = cv2.sepFilter2D(img.astype(np.float32), -1, k, k, borderType=cv2.BORDER_REPLICATE)
    out = f[1::2, 1::2][:h >> 1, :w >> 1]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def tarzoom_bytes(rgb: np.ndarray, quality: int, tile: int = TILE_SIZE,
                  overlap: int = OVERLAP) -> tuple[dict, list[bytes]]:
    """The .tzi index and the tiles, in .tzb order (smallest level first)."""
    h, w = rgb.shape[:2]
    nl = n_levels(w, h, tile)
    if (w >> (nl - 1)) == 0 or (h >> (nl - 1)) == 0:
        # OpenLIME counts a level's tiles from w >> k with no floor of 1, so
        # a side this thin has no tiles at the smallest level and every
        # offset after it would be wrong.
        raise ValueError(f"A {w} x {h} image is too thin for tarzoom; use the single-image layout.")
    levels = [rgb]
    for _ in range(nl - 1):
        levels.append(halve(levels[-1]))
    levels.reverse()                                    # 0 = smallest
    tiles: list[bytes] = []
    for img in levels:
        lh, lw = img.shape[:2]
        tx, ty = max(1, -(-lw // tile)), max(1, -(-lh // tile))
        for row in range(ty):
            y0, y1 = max(0, row * tile - overlap), min(lh, (row + 1) * tile + overlap)
            for col in range(tx):
                x0, x1 = max(0, col * tile - overlap), min(lw, (col + 1) * tile + overlap)
                tiles.append(encode_jpeg(img[y0:y1, x0:x1], quality))
    offsets = [0]
    for t in tiles:
        offsets.append(offsets[-1] + len(t))
    index = {"tilesize": tile, "overlap": overlap, "format": "jpg", "width": w,
             "height": h, "nlevels": nl, "offsets": offsets}
    return index, tiles


def write_image(rgb: np.ndarray, out_dir: str, name: str, layout: str, quality: int) -> list[str]:
    """plane_N.jpg, or plane_N.tzi + plane_N.tzb."""
    if layout == "image":
        p = os.path.join(out_dir, name + ".jpg")
        with open(p, "wb") as f:
            f.write(encode_jpeg(rgb, quality))
        return [p]
    if layout != "tarzoom":
        raise ValueError(f"unknown layout {layout!r}")
    index, tiles = tarzoom_bytes(rgb, quality)
    pb, pi = os.path.join(out_dir, name + ".tzb"), os.path.join(out_dir, name + ".tzi")
    with open(pb, "wb") as f:
        for t in tiles:
            f.write(t)
    with open(pi, "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    return [pi, pb]


def web_view(img: np.ndarray, out_dir: str, name: str, quality: int = 90) -> tuple[str, str, int]:
    """A map as the browser can show it: 8-bit RGB, tarzoom (single JPEG when
    too thin to tile).  Returns (file name relative to *out_dir*, layout, width)."""
    a = np.asarray(img)
    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    elif a.dtype != np.uint8:
        f = np.nan_to_num(a.astype(np.float32))
        lo, hi = float(np.percentile(f, 0.5)), float(np.percentile(f, 99.5))
        a = np.clip((f - lo) / max(hi - lo, 1e-12) * 255.0, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    a = np.ascontiguousarray(a[..., :3])
    os.makedirs(out_dir, exist_ok=True)
    try:
        write_image(a, out_dir, name, "tarzoom", quality)
        return name + ".tzi", "tarzoom", a.shape[1]
    except ValueError:
        write_image(a, out_dir, name, "image", quality)
        return name + ".jpg", "image", a.shape[1]


def normals_rgb(normals: np.ndarray, h: int, w: int) -> np.ndarray:
    """Unit normals (+Y up) -> Relight's normals image, resized to (h, w)."""
    n = np.nan_to_num(np.asarray(normals, np.float32)[..., :3])
    if n.shape[:2] != (h, w):
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_AREA
                       if n.shape[1] > w else cv2.INTER_LINEAR)
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-8)
    return np.clip(np.rint((n + 1.0) * 0.5 * 255.0), 0, 255).astype(np.uint8)


# ── the package ──────────────────────────────────────────────────────

def export(out_dir: str, read: Callable[[int], np.ndarray], lights: np.ndarray, h: int, w: int,
           *, normals: np.ndarray | None = None, layout: str = "tarzoom", quality: int = 95,
           pixel_mm: float | None = None, title: str = "", progress=None, check=None,
           budget: int | None = None) -> dict:
    """Write info.json, plane_0..8 and normals in *layout*, a viewer and a read-me.

    *read(k)* returns photograph k as (h, w, 3) uint8 sRGB, matching *lights[k]*.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    os.makedirs(out_dir, exist_ok=True)
    lights = np.asarray(lights, np.float64)[:, :3]
    planes, mins, maxs = fit_quantised(read, lights, h, w, budget, progress, check)
    files = []
    for k in range(9):
        if progress:
            progress("Writing the planes…", 0.85 + 0.12 * k / 9)
        files += write_image(planes[k], out_dir, f"plane_{k}", layout, quality)
    if normals is not None:
        files += write_image(normals_rgb(normals, h, w), out_dir, "normals", layout, quality)
    scale, bias = scale_bias(mins, maxs)
    unit = lights / np.maximum(np.linalg.norm(lights, axis=1, keepdims=True), 1e-12)
    info = {"width": int(w), "height": int(h), "format": "jpg", "type": "hsh",
            "colorspace": "rgb",
            "lights": [round(float(v), 3) for v in unit.reshape(-1)],
            "nplanes": 27, "quality": int(quality),
            "materials": [{"scale": scale, "bias": bias}]}
    if pixel_mm:
        info["pixelSizeInMM"] = float(pixel_mm)
    p = os.path.join(out_dir, "info.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    files.append(p)
    for name, text in (("index.html", viewer_html(layout, normals is not None, title)),
                       ("README.txt", readme(layout, normals is not None, w, h, len(lights))),
                       ("serve.py", SERVE_PY),
                       ("Open viewer.command", OPEN_COMMAND),
                       ("Open viewer.bat", OPEN_BAT)):
        q = os.path.join(out_dir, name)
        with open(q, "w", encoding="utf-8", newline="\r\n" if name.endswith(".bat") else "\n") as f:
            f.write(text)
        if name.endswith(".command"):
            os.chmod(q, 0o755)                 # double-clickable in the Finder
        files.append(q)
    if progress:
        progress("Done", 1.0)
    return {"dir": out_dir, "files": files, "info": info}


OPENLIME = "https://cdn.jsdelivr.net/npm/openlime@1.2.6/dist"

#: A local server that answers HTTP range requests (tarzoom needs them;
#: python -m http.server does not), then opens the viewer.  Standard library only.
SERVE_PY = r'''"""Serve this folder with HTTP range requests and open the viewer.

    python3 serve.py [port]

Ctrl-C (or closing the window) stops it.
"""
import functools, http.server, os, re, sys, threading, webbrowser


class Handler(http.server.SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if not m or not os.path.isfile(path) or not (m.group(1) or m.group(2)):
            self._left = None
            return super().send_head()
        size = os.path.getsize(path)
        if m.group(1):
            a = int(m.group(1))
            b = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:
            a, b = max(0, size - int(m.group(2))), size - 1
        if a > b or a >= size:
            self.send_error(416)
            return None
        f = open(path, "rb")
        f.seek(a)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.send_header("Content-Length", str(b - a + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._left = b - a + 1
        return f

    def copyfile(self, src, dst):
        n = getattr(self, "_left", None)
        if n is None:
            return super().copyfile(src, dst)
        while n > 0:
            chunk = src.read(min(n, 1 << 16))
            if not chunk:
                break
            dst.write(chunk)
            n -= len(chunk)

    def end_headers(self):
        # A re-export over the same folder must show, not yesterday's page from the cache.
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        pass


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    port = int(nums[0]) if nums else 0
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port),
                                            functools.partial(Handler, directory=here))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    print(f"Serving {here}\n  {url}\nCtrl-C to stop.")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
'''

OPEN_COMMAND = """#!/bin/bash
# Double-click: serves this folder and opens the RTI viewer in your browser.
cd "$(dirname "$0")" && exec python3 serve.py
"""

OPEN_BAT = """@echo off
rem Double-click: serves this folder and opens the RTI viewer in your browser.
cd /d "%~dp0"
python serve.py || py serve.py
pause
"""


def viewer_html(layout: str, has_normals: bool, title: str = "", maps=(), brdf=None) -> str:
    """OpenLIME's own RTI layer on this folder, light mode, and a Views menu:
    the relit RTI, the normals plane when there is one, and *maps* - dicts
    with ``id``, ``label``, ``url`` (a .tzi or .jpg, relative), ``layout``
    ("tarzoom" or "image") and ``z`` (the RTI's width over the map's, so a
    map solved at a reduced size lies exactly on the RTI).

    OpenLIME 1.2.6's normals and diffuse modes are wrong (the normals are
    shown through the sRGB curve and, without a normals file, estimated from
    three fixed lights; diffuse takes the quantised first coefficient as its
    colour), so the RTI's mode menu is hidden here; the normals are shown
    from HELIO's own normals plane instead.

    *brdf* (keys kd, normals and optionally ks, gloss: urls; layout; z) adds
    relightable views on OpenLIME's BRDF layer, under the RTI's light control:
    the shine (a highlight whose strength is the specular map, 0..1, and whose
    sharpness is the gloss map, 0..1: Blinn-Phong exponent 15..150) over a
    faint shape, and the shape in clay (cos^4 of the light's angle, so tooling
    and creases read as under a raking light).  Both replace the shader's own
    mode code: OpenLIME 1.2.6's specular mode is a broad Ward lobe that reads
    as dull shading, and its gamma and pi handling leave everything dark or flat."""
    t = (title or "RTI").replace("&", "&amp;").replace("<", "&lt;")
    views = [dict(id="RTI", label="Relight (RTI)", url="info.json", layout=layout, z=1.0, rti=True)]
    if has_normals:
        views.append(dict(id="normals", label="Normals",
                          url="normals.tzi" if layout == "tarzoom" else "normals.jpg",
                          layout=layout, z=1.0))
    if brdf and brdf.get("kd") and brdf.get("normals"):
        ch = {k: brdf[k] for k in ("kd", "ks", "normals", "gloss") if brdf.get(k)}
        for vid, label, mode in (("relit_shine", "Relit shine (specular)", "specular"),
                                 ("relit_clay", "Relit shape (clay)", "monochrome")):
            if mode == "specular" and ("ks" not in ch or "gloss" not in ch):
                continue
            views.append(dict(id=vid, label=label, brdf=ch, mode=mode, layout=brdf.get("layout", layout),
                              z=float(brdf.get("z", 1.0))))
    views += [dict(m) for m in maps]
    js_views = json.dumps([{k: v for k, v in m.items()} for m in views])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t}</title>
<link rel="stylesheet" href="{OPENLIME}/css/skin.css">
<style>html,body{{margin:0;padding:0;height:100%;background:#000}}#viewer{{position:relative;height:100%}}
/* The Views menu: a small see-through card top left, clear of the scale bar (bottom left). */
.openlime-layers-menu{{top:12px;left:-200px;width:172px;max-width:172px;height:auto;
  max-height:calc(100% - 130px);border-radius:8px;background:rgba(18,18,18,.42);
  -webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px);border:1px solid rgba(255,255,255,.08);
  box-shadow:none;transition:left .25s ease-in-out}}
.openlime-layers-menu.open{{left:12px}}
.openlime-layers-menu.closing{{left:-200px}}
.openlime-layers-header{{padding:5px 6px 5px 10px;background:transparent;border-bottom:1px solid rgba(255,255,255,.08)}}
.openlime-layers-header h2{{font-size:11px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;opacity:.75}}
.openlime-layers-close-btn{{font-size:16px;width:20px;height:20px;opacity:.7}}
.openlime-layers-content{{padding:4px 6px 6px}}
.openlime-entry{{margin:1px 0;padding:4px 8px;border-radius:4px;font-size:12px;font-weight:400}}
.openlime-layer-entry{{background:transparent}}
.openlime-layer-entry:hover{{background:rgba(255,255,255,.12)}}
.openlime-layer-entry.active{{background:rgba(233,180,76,.38)}}
.openlime-layer-icon{{width:8px;height:8px;margin-right:7px;border-radius:50%;opacity:.5}}
.openlime-layer-status{{font-size:11px;color:#E9B44C}}
</style>
</head><body>
<div id="viewer" class="openlime"></div>
<script src="{OPENLIME}/js/openlime.min.js"></script>
<script>
const lime = new OpenLIME.Viewer('#viewer', {{ background: 'black' }});
// Every view lies on the RTI: a map solved at a reduced size is scaled up by z about the centre,
// where OpenLIME puts every layer.
const VIEWS = {js_views};
const BRDF_CODE = {{
  specular: `float nh = max(dot(N, H), 0.0);
    float ex = mix(15.0, 150.0, clamp(gloss, 0.0, 1.0));
    float hl = pow(nh, ex) * (ex + 8.0) / 25.0;
    vec3 linearColor = vec3(clamp(2.0 * ks.x * hl * NdotL + 0.05 * NdotL, 0.0, 1.0));
    applyGamma = false;`,
  monochrome: `vec3 linearColor = vec3(0.88 * pow(NdotL, 4.0) + 0.03);
    applyGamma = false;`
}};
let layer = null;
for (const v of VIEWS) {{
  const tf = {{ x: 0, y: 0, z: v.z, a: 0 }};
  const opts = v.rti ? {{ type: 'rti', layout: v.layout, url: v.url, normals: false, label: v.label }}
             : v.brdf ? {{ type: 'brdf', layout: v.layout, channels: v.brdf, visible: false, label: v.label,
                          colorspaces: {{ kd: 'srgb', ks: 'linear' }}, transform: tf }}
             : {{ type: 'image', layout: v.layout, url: v.url, visible: false, label: v.label, transform: tf }};
  const l = new OpenLIME.Layer(opts);
  if (v.brdf) {{
    l.setMode(v.mode);                        // picks the textures the view reads
    l.shader.innerCode = BRDF_CODE[v.mode];   // ...and HELIO's shading of them (see viewer_html)
    l.shader.needsUpdate = true;
  }}
  lime.canvas.addLayer(v.id, l);
  if (v.rti) layer = l;
}}
OpenLIME.Skin.setUrl('{OPENLIME}/skin/skin.svg');
// OpenLIME lists every layer itself (one shown at a time); the RTI's own modes are left out of
// the list (see above), and the panel is called Views.
const ui = new OpenLIME.UIBasic(lime, {{ skin: '{OPENLIME}/skin/skin.svg', showLightDirections: true }});
ui.actions.light.active = true;
ui.actions.layers.display = VIEWS.length > 1;
(function tidyMenu(n) {{
  const m = ui.layerMenu;
  if (!m) {{ if (n < 200) setTimeout(() => tidyMenu(n + 1), 50); return; }}
  const h = m.querySelector('.openlime-layers-header h2');
  if (h) h.textContent = 'Views';
  m.querySelectorAll('.openlime-mode-entry, .openlime-section').forEach(e => e.remove());
  m.querySelectorAll('.openlime-list').forEach(e => {{ if (!e.querySelector('.openlime-layer-entry')) e.remove(); }});
}})(0);
ui.actions.zoomin.display = true;
ui.actions.zoomout.display = true;
lime.camera.maxFixedZoom = 4;
window.lime = lime; window.rtiLayer = layer;
// A black canvas with no word of why is what a file:// page, or a server that
// ignores HTTP range requests (python -m http.server), gives: say so instead.
function helioSay(t) {{
  const d = document.createElement('div');
  d.style.cssText = 'position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
    + 'color:#ddd;font:15px/1.5 -apple-system,system-ui,sans-serif;padding:24px;text-align:center;z-index:9';
  d.innerHTML = '<div style="max-width:34em">' + t + '</div>';
  document.body.appendChild(d);
}}
if (location.protocol === 'file:') {{
  helioSay('This viewer has to be served over HTTP: a browser will not load its files from disk.<br>'
    + 'Double-click <b>Open viewer.command</b> (Mac) or <b>Open viewer.bat</b> (Windows) in this folder.');
}} else if ('{layout}' === 'tarzoom') {{
  fetch('plane_0.tzb', {{ headers: {{ Range: 'bytes=0-1' }} }}).then(r => {{
    if (r.status !== 206)
      helioSay('This server sends whole files, not the pieces the tiled planes need (HTTP range '
        + 'requests), so the picture stays black.<br>Open it with <b>Open viewer.command</b> / '
        + '<b>Open viewer.bat</b> in this folder, or put the folder on an ordinary web server.');
    try {{ r.body && r.body.cancel(); }} catch (e) {{}}
  }}).catch(() => {{}});
}}
</script>
</body></html>
"""


def readme(layout: str, has_normals: bool, w: int, h: int, n: int) -> str:
    files = ("plane_0.tzi/.tzb ... plane_8.tzi/.tzb" if layout == "tarzoom"
             else "plane_0.jpg ... plane_8.jpg")
    norm = ("normals" + (".tzi/.tzb" if layout == "tarzoom" else ".jpg")
            + "   HELIO's photometric-stereo normals, (n+1)/2 in RGB, +Y up\n") if has_normals else ""
    return f"""RGB HSH RTI for OpenLIME and RelightLab
=======================================

{w} x {h} pixels, fitted from {n} photographs.

  info.json   type "hsh", colorspace "rgb", 27 coefficients, scale and bias
  {files}
              coefficient k of R, G and B in plane k (JPEG, no chroma subsampling)
  {norm}  index.html  OpenLIME 1.2.6 showing this folder (relighting)

This is the format RelightLab writes with  relight-cli -b hsh -p 27, and the
one OpenLIME's RTI layer reads:

  new OpenLIME.Layer({{ type: 'rti', layout: '{layout}', url: '.../info.json' }})

Add  normals: true  to that call to use the normals file.

maps/ (when exported from HELIO's Publish) holds the surface maps: colour,
normal, roughness, specular, displacement and more; see maps/README.txt.

To look at it here: double-click  Open viewer.command  (Mac) or  Open viewer.bat
(Windows). It runs serve.py, a small local web server, and opens the browser.
Double-clicking index.html itself shows nothing: browsers will not load its
files from a file:// address.
{"Tarzoom needs a server that answers HTTP range requests (any normal web server does;" + chr(10) + "python -m http.server does not, and leaves the picture black; serve.py does)." if layout == "tarzoom" else ""}

OpenLIME 1.2.6 (openlime.min.js, the current release) relights this
correctly. Its other modes are not to be trusted for measurement:

  normals   without a normals file it estimates the normal from the HSH fit
            under three fixed 45-degree lights, which flattens relief; with or
            without one, it passes the result through the sRGB display curve,
            so a level surface shows about (188, 188, 255), not (128, 128, 255).
            With normals: true it reads this folder's normals correctly; only
            that display curve is wrong.
  diffuse   uses plane 0 (the first HSH coefficient, still quantised) as the
            colour, not an albedo.

Use HELIO's own normal map for measurement.

Credits
-------
OpenLIME (GPL-3.0, CNR-ISTI Visual Computing Lab,
https://github.com/cnr-isti-vclab/openlime) is loaded by index.html from the
jsDelivr CDN when the page opens; it is not included in this folder. Copying
openlime.min.js in, to view offline, means distributing GPL-3 code under that
licence.
The format is Relight's: Ponchio, F., Corsini, M. & Scopigno, R. (2019),
"RELIGHT: a compact and accurate RTI representation for the web", Graphical
Models 105. The HSH basis: Gautron, P., Krivanek, J., Pattanaik, S. &
Bouatouch, K. (2004), "A novel hemispherical basis for accurate and efficient
rendering", Eurographics Symposium on Rendering.
"""
