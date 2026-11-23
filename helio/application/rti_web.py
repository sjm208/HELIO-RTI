"""The RTI web package (EXP-010): one folder that shows a capture on the web.

What it holds, so it can go on a website, on a USB stick or into an archive
without anything else:

``index.html``
    A relighting viewer at full resolution: the normals and the colour as
    deep-zoom tiles (256 px, 1 px overlap, a level per halving), relit on
    the graphics card as the light dome, the mouse or the arrow keys move
    the light.  Colour, shading, normals and specular enhancement; pan and
    zoom down to single pixels; a scale bar when the scale is known; every
    key listed on the page (?).  Plain WebGL 2, no library, no network.
``quick-view.html``
    The whole thing in one file (textures inside it, held under 25 MB):
    for e-mail, and for opening straight from a disk.  Browsers will not
    hand pictures loaded from a disk to WebGL, so index.html needs the
    folder served (the two launchers do it) or put on a website.
``relight/``
    The Relight / OpenLIME data (info.json and coefficient planes, what
    RelightLab and the relight.js / OpenLIME viewers read) and the frames
    with their lights, ready to fit again.
``lights.lp``, ``metadata.json``, ``README.txt``, ``embed.html``,
``1_Mac.command``, ``2_Win.bat``, and ``qr.png`` when a web address is given.

``package`` builds it; ``estimate`` says how big it will be first.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

TILE = 256
OVERLAP = 1


@dataclass
class Options:
    title: str = ""
    obj_id: str = ""
    licence: str = "CC BY 4.0"
    creator: str = ""
    institution: str = ""
    url: str = ""                       # where it will live: for the QR code and embed code
    quick_view: bool = True
    relight_data: bool = True
    frames: bool = True                 # the frames re-saved with lights.lp, to fit again
    zip: bool = False
    jpeg_quality: int = 90
    max_side: int = 0                   # 0: full size
    notes: list[str] = field(default_factory=list)


# ── tiles ────────────────────────────────────────────────────────────────

def levels_for(w: int, h: int) -> list[dict]:
    """The pyramid: level 0 full size, each next half, down to one tile."""
    out = []
    lw, lh = int(w), int(h)
    while True:
        out.append(dict(w=lw, h=lh, cols=math.ceil(lw / TILE), rows=math.ceil(lh / TILE)))
        if max(lw, lh) <= TILE:
            break
        lw, lh = max(1, math.ceil(lw / 2)), max(1, math.ceil(lh / 2))
    return out


def _write_tiles(img: np.ndarray, folder: str, ext: str, quality: int, pool) -> int:
    """One level's tiles, each with a pixel of its neighbours round it."""
    lh, lw = img.shape[:2]
    os.makedirs(folder, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext == "jpg" else \
        [int(cv2.IMWRITE_PNG_COMPRESSION), 6]
    jobs = []
    for r in range(math.ceil(lh / TILE)):
        for c in range(math.ceil(lw / TILE)):
            x0, y0 = c * TILE, r * TILE
            x1, y1 = min(lw, x0 + TILE), min(lh, y0 + TILE)
            ox0, oy0 = max(0, x0 - OVERLAP), max(0, y0 - OVERLAP)
            ox1, oy1 = min(lw, x1 + OVERLAP), min(lh, y1 + OVERLAP)
            tile = np.ascontiguousarray(img[oy0:oy1, ox0:ox1][..., ::-1])     # RGB -> BGR
            path = os.path.join(folder, f"{c}_{r}.{ext}")
            jobs.append(pool.submit(cv2.imwrite, path, tile, params))
    for j in jobs:
        if not j.result():
            raise OSError(f"could not write a tile in {folder}")
    return len(jobs)


def _normals_rgb(n: np.ndarray) -> np.ndarray:
    """+Y-up unit normals as 8-bit RGB (n * 0.5 + 0.5), the usual encoding."""
    return np.clip(np.round((np.nan_to_num(n) * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)


def _halve_normals(n: np.ndarray, w: int, h: int) -> np.ndarray:
    m = cv2.resize(n, (w, h), interpolation=cv2.INTER_AREA)
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-8)


def write_pyramids(normals: np.ndarray, albedo: np.ndarray, tiles_dir: str, quality: int = 90,
                   progress=None) -> dict:
    """tiles/normals/<level>/<col>_<row>.png and tiles/albedo/... .jpg."""
    h, w = albedo.shape[:2]
    if normals.shape[:2] != (h, w):
        normals = _halve_normals(np.asarray(normals, np.float32), w, h)
    levels = levels_for(w, h)
    n = np.asarray(normals, np.float32)
    a = np.ascontiguousarray(albedo, np.uint8)
    count = 0
    with ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4)))) as pool:
        for i, lv in enumerate(levels):
            if i:
                n = _halve_normals(n, lv["w"], lv["h"])
                a = cv2.resize(a, (lv["w"], lv["h"]), interpolation=cv2.INTER_AREA)
            count += _write_tiles(_normals_rgb(n), os.path.join(tiles_dir, "normals", str(i)), "png",
                                  quality, pool)
            count += _write_tiles(a, os.path.join(tiles_dir, "albedo", str(i)), "jpg", quality, pool)
            if progress:
                progress(f"Tiles: level {i + 1} of {len(levels)}", (i + 1) / len(levels))
    return dict(levels=levels, tiles=count)


# ── the package ──────────────────────────────────────────────────────────

def estimate(w: int, h: int, opts: Options | None = None) -> dict:
    """Rough size and file count before writing (JPEG ~0.18 B/px at q90, PNG normals ~0.9)."""
    opts = opts or Options()
    if opts.max_side and max(w, h) > opts.max_side:
        k = opts.max_side / max(w, h)
        w, h = int(w * k), int(h * k)
    px = sum(lv["w"] * lv["h"] for lv in levels_for(w, h))
    tiles = 2 * sum(lv["cols"] * lv["rows"] for lv in levels_for(w, h))
    mb = px * (0.18 + 0.9) / 1e6
    if opts.quick_view:                 # four maps, base64, at most 4096 px, under 25 MB
        k = min(1.0, 4096 / max(w, h))
        mb += min(25.0, w * h * k * k * 1.7 * 1.37 / 1e6)
    if opts.relight_data:
        mb += w * h * 9 * 0.18 / 1e6
    return dict(mb=round(mb, 1), files=tiles + 12, width=w, height=h)


def package(out_dir: str, *, normals, albedo, lights=None, ptm_coeffs=None, roughness=None,
            spec_var=None, images=(), mm_per_px=None, z_mm_per_unit=None, solve: dict | None = None,
            opts: Options | None = None, provenance_app=None, progress=None, region=None) -> dict:
    """Write the RTI web package into *out_dir* (made if missing).

    *normals* (h, w, 3) +Y up and *albedo* (h, w, 3) uint8 RGB at full size;
    *lights* (N, 3); *ptm_coeffs* (6, h*w) for the Relight data; *images* the
    frames (cut to *region*, a region solve's fractions).
    Returns {dir, index, quick_view, files, mb, levels, notes}."""
    opts = opts or Options()
    upd = progress or (lambda *_a: None)
    t0 = time.time()
    normals = np.asarray(normals, np.float32)
    albedo = np.ascontiguousarray(albedo, np.uint8)
    albedo_full = albedo                      # the Relight planes keep the full size
    h, w = albedo.shape[:2]
    if opts.max_side and max(w, h) > opts.max_side:
        k = opts.max_side / max(w, h)
        w2, h2 = max(1, int(round(w * k))), max(1, int(round(h * k)))
        albedo = cv2.resize(albedo, (w2, h2), interpolation=cv2.INTER_AREA)
        normals = _halve_normals(normals, w2, h2)
        if mm_per_px:
            mm_per_px = mm_per_px * w / w2
        h, w = h2, w2
    os.makedirs(out_dir, exist_ok=True)
    notes = list(opts.notes)
    title = opts.title or opts.obj_id or "Helio capture"

    upd("Writing the deep-zoom tiles…", 0.05)
    pyr = write_pyramids(normals, albedo, os.path.join(out_dir, "tiles"), opts.jpeg_quality,
                         progress=lambda s, f: upd(s, 0.05 + 0.6 * f))
    L = None if lights is None else np.asarray(lights, np.float64)
    meta = {
        "title": title, "object_id": opts.obj_id, "creator": opts.creator,
        "institution": opts.institution, "licence": opts.licence, "url": opts.url,
        "width": w, "height": h, "tile": TILE, "overlap": OVERLAP, "levels": pyr["levels"],
        "layers": {"normals": {"dir": "tiles/normals", "ext": "png", "encoding":
                               "rgb = n * 0.5 + 0.5, +Y up (north = top of the image)"},
                   "albedo": {"dir": "tiles/albedo", "ext": "jpg"}},
        "mm_per_px": float(mm_per_px) if mm_per_px else None,
        "z_mm_per_unit": float(z_mm_per_unit) if z_mm_per_unit else None,
        "lights": None if L is None else [[round(float(v), 6) for v in row] for row in L],
        "solve": solve or {},
        "made": time.strftime("%Y-%m-%d %H:%M"),
    }
    try:
        from helio_qt import __version__ as _v
        meta["helio"] = f"Helio {_v}"
    except Exception:
        meta["helio"] = "Helio"
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    upd("Writing the viewer…", 0.7)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(viewer_html(meta))

    if L is not None and len(L):
        with open(os.path.join(out_dir, "lights.lp"), "w") as f:
            names = [os.path.basename(p) for p in images] if len(images) == len(L) else \
                [f"frame_{i + 1:03d}.jpg" for i in range(len(L))]
            f.write(f"{len(L)}\n")
            for nm, v in zip(names, L):
                n = float(np.linalg.norm(v)) or 1.0
                f.write(f"{nm} {v[0] / n:.6f} {v[1] / n:.6f} {v[2] / n:.6f}\n")

    quick = ""
    if opts.quick_view:
        upd("Writing the one-file quick view…", 0.75)
        try:
            from helio.application.export_web import combined_rti_viewer
            quick = os.path.join(out_dir, "quick-view.html")
            r8 = np.full((h, w), 128, np.uint8)
            if roughness is not None:
                r8 = roughness if roughness.dtype == np.uint8 else \
                    (np.clip(roughness, 0, 1) * 255).astype(np.uint8)
                if r8.shape[:2] != (h, w):
                    r8 = cv2.resize(r8, (w, h), interpolation=cv2.INTER_AREA)
            sv = np.zeros((h, w), np.float32) if spec_var is None else np.asarray(spec_var, np.float32)
            if sv.shape[:2] != (h, w):
                sv = cv2.resize(sv, (w, h), interpolation=cv2.INTER_AREA)
            mb, used = combined_rti_viewer(quick, normals, albedo, r8, sv, h, w, (0, 0, w, h),
                                           opts.obj_id or title, budget_mb=25.0)
            if used < 4096:
                notes.append(f"quick-view.html holds the textures at {used} px to stay under 25 MB.")
        except Exception as e:                                   # the rest still stands
            notes.append(f"quick-view.html was not written: {e}")
            quick = ""

    if opts.relight_data and ptm_coeffs is not None:
        upd("Writing the Relight / OpenLIME data…", 0.85)
        try:
            from helio.application.export_web import relight_package
            relight_package(os.path.join(out_dir, "relight"), ptm_coeffs, albedo_full, lights,
                            *(_ptm_hw(ptm_coeffs, albedo_full)),
                            images=images if opts.frames else (), provenance_app=None, region=region)
        except Exception as e:
            notes.append(f"The Relight / OpenLIME data was not written: {e}")

    upd("Writing the read-me, embed code and launchers…", 0.92)
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(readme_text(meta, bool(quick), notes))
    from helio.application.export_web import embed_snippet
    with open(os.path.join(out_dir, "embed.html"), "w", encoding="utf-8") as f:
        url = (opts.url.rstrip("/") + "/index.html") if opts.url else "index.html"
        f.write("<!-- Paste this where the viewer should appear on a web page. -->\n"
                + embed_snippet(url, obj=title) + "\n")
    from helio.application.export_images import sharable_launchers
    sharable_launchers(out_dir)
    if opts.url:
        try:
            from helio.application.export_web import qr_png
            qr_png(opts.url, os.path.join(out_dir, "qr.png"), obj=title,
                   institution=opts.institution,
                   scale=f"{mm_per_px:.4g} mm/px" if mm_per_px else "")
        except Exception as e:
            notes.append(f"No QR code: {e}")
    if provenance_app is not None:
        try:
            from helio.io.provenance import write_provenance
            write_provenance(provenance_app, out_dir)
        except Exception:
            pass
    files, size = 0, 0
    for root, _d, fs in os.walk(out_dir):
        for fn in fs:
            files += 1
            size += os.path.getsize(os.path.join(root, fn))
    zip_path = ""
    if opts.zip:
        upd("Zipping…", 0.97)
        zip_path = shutil.make_archive(out_dir.rstrip(os.sep), "zip", out_dir)
    upd("Done", 1.0)
    return dict(dir=out_dir, index=os.path.join(out_dir, "index.html"), quick_view=quick,
                files=files, mb=round(size / 1e6, 1), levels=len(pyr["levels"]), notes=notes,
                zip=zip_path, seconds=round(time.time() - t0, 1))


def _ptm_hw(ptm, albedo) -> tuple[int, int]:
    c = np.asarray(ptm)
    if c.ndim == 3:
        return int(c.shape[1]), int(c.shape[2])
    h, w = albedo.shape[:2]
    if c.shape[1] == h * w:
        return h, w
    raise ValueError("the PTM planes and the colour are not the same size")


def readme_text(meta: dict, quick: bool, notes) -> str:
    mm = meta.get("mm_per_px")
    lines = [
        f"{meta['title']}",
        "=" * max(8, len(meta['title'])),
        "",
        "A relightable image (reflectance transformation imaging) made with "
        f"{meta.get('helio', 'Helio')} on {meta['made']}.",
        "",
        "HOW TO OPEN IT",
        "  On a website: upload this whole folder and open index.html.",
        "  On this computer: double-click 1_Mac.command (Mac) or 2_Win.bat (Windows).",
        "  They serve the folder and open the viewer; browsers will not relight",
        "  pictures read straight from a disk." + (" Or open quick-view.html: it is one" if quick else ""),
    ]
    if quick:
        lines.append("  file with everything inside, and opens from anywhere.")
    lines += [
        "",
        "IN THE VIEWER",
        "  Drag the dot on the dome, or hold Shift and drag on the image, to move the light.",
        "  Scroll or pinch to zoom, drag to move about, double-click to zoom in.",
        "  Arrow keys turn and raise the light; 1-4 change the look; F fits; ? lists every key.",
        "",
        "WHAT IS HERE",
        f"  index.html         the viewer ({meta['width']} x {meta['height']} pixels, "
        f"{len(meta['levels'])} zoom levels)",
        "  tiles/             the normals (PNG) and colour (JPEG) as deep-zoom tiles",
    ]
    if quick:
        lines.append("  quick-view.html    the whole viewer in one file")
    lines += [
        "  relight/           Relight / OpenLIME data (info.json + planes) and the frames to refit",
        "  lights.lp          the light directions, one per frame (x y z, +Y towards the top)",
        "  metadata.json      the object, scale, lights and how it was solved",
        "  embed.html         the code to put the viewer on another page",
        "",
        "SCALE",
        f"  {mm:.4g} mm per pixel" if mm else "  Not measured: distances are in pixels.",
        "",
        "RIGHTS",
        f"  {meta.get('licence') or 'Not stated'}"
        + (f" · {meta['creator']}" if meta.get("creator") else "")
        + (f" · {meta['institution']}" if meta.get("institution") else ""),
        "",
        "Normals are encoded as RGB = n * 0.5 + 0.5, with +Y towards the top of the image.",
    ]
    if notes:
        lines += ["", "NOTES"] + [f"  {n}" for n in notes]
    return "\n".join(lines) + "\n"


def viewer_html(meta: dict) -> str:
    """The deep-zoom relighting viewer, with its metadata inside it."""
    # "<" as \u003c: nothing inside the script can open or close a tag.
    blob = json.dumps(meta, separators=(",", ":")).replace("<", "\\u003c")
    page = VIEWER.replace("__META__", blob).replace("__TITLE__", _html(meta["title"]))
    # On a phone the settings go in a sheet along the bottom (they covered the
    # title); the page has its own drag, pinch and dome, so no gestures added.
    from helio.application.phone_web import phone_ready
    return phone_ready(page, panel="#tools", skin=False, css=(
        "#tools{display:flex!important;border-radius:14px 14px 0 0!important}"
        "#title{max-width:calc(100vw - 24px)!important}"
        "#dome,#busy{bottom:calc(var(--hp-sheet) + 12px + env(safe-area-inset-bottom))!important}"
        "#scale{bottom:calc(var(--hp-sheet) + 70px)!important}"
        "input[type=range]{width:55vw}"))


def _html(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


VIEWER = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>__TITLE__</title>
<style>
:root { --bg:#1B1C1F; --panel:rgba(34,35,38,.92); --line:#3E4046; --text:#ECEAE5; --text2:#B4B6BC;
        --accent:#E6A93E; color-scheme: dark; }
* { box-sizing: border-box; }
html, body { margin:0; height:100%; background:#2B2B2B; color:var(--text);
             font:13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; overflow:hidden; }
canvas#gl { position:fixed; inset:0; width:100%; height:100%; touch-action:none; cursor:grab; }
canvas#gl.dragging { cursor:grabbing; }
.panel { position:fixed; background:var(--panel); border:1px solid var(--line); border-radius:10px;
         padding:10px 12px; backdrop-filter: blur(6px); }
#title { top:12px; left:12px; right:auto; max-width:min(60vw, 560px); }
#title h1 { margin:0; font-size:15px; font-weight:600; }
#title p { margin:2px 0 0; color:var(--text2); font-size:12px; }
#dome { bottom:12px; left:12px; padding:8px; }
#dome canvas { display:block; width:150px; height:150px; cursor:pointer; touch-action:none; }
#dome .angles { text-align:center; color:var(--text2); font:12px ui-monospace, Menlo, monospace; margin-top:4px; }
#tools { top:12px; right:12px; width:232px; display:flex; flex-direction:column; gap:8px; }
.seg { display:grid; grid-template-columns:repeat(2, 1fr); gap:3px; padding:3px; border:1px solid var(--line); border-radius:7px; }
.seg button { border:0; border-radius:5px; background:transparent; color:var(--text2); padding:5px 4px; font:inherit; cursor:pointer; }
.seg button[aria-pressed="true"] { background:#33353A; color:var(--text); font-weight:600; }
label.row { display:flex; justify-content:space-between; align-items:center; gap:8px; color:var(--text2); font-size:12px; }
input[type=range] { width:120px; accent-color:var(--accent); }
button.plain { border:1px solid var(--line); border-radius:6px; background:transparent; color:var(--text); padding:5px 8px; font:inherit; cursor:pointer; }
button.plain:hover, .seg button:hover { color:var(--text); }
#scale { bottom:12px; left:50%; transform:translateX(-50%); padding:6px 10px; font:12px ui-monospace, Menlo, monospace; display:none; }
#scale .bar { height:4px; background:var(--text); margin-bottom:4px; }
#keys { inset:0; margin:auto; width:min(440px, 92vw); height:fit-content; display:none; z-index:5; }
#keys h2 { margin:0 0 8px; font-size:14px; }
#keys table { width:100%; border-collapse:collapse; font-size:12px; }
#keys td { padding:3px 4px; color:var(--text2); }
#keys kbd { font:12px ui-monospace, Menlo, monospace; border:1px solid var(--line); border-bottom-width:2px; border-radius:4px; padding:0 5px; background:#33353A; color:var(--text); }
#notice { inset:0; margin:auto; width:min(520px, 92vw); height:fit-content; display:none; z-index:6; }
#notice h2 { margin:0 0 6px; font-size:15px; }
#notice a { color:var(--accent); }
#busy { bottom:12px; right:12px; font-size:12px; color:var(--text2); display:none; }
@media (max-width: 640px) { #tools { width:200px; } #dome canvas { width:118px; height:118px; } }
</style>
</head>
<body>
<canvas id="gl" aria-label="Relightable image: drag to move about, Shift-drag to move the light"></canvas>
<div id="title" class="panel"><h1>__TITLE__</h1><p id="sub"></p></div>
<div id="dome" class="panel"><canvas id="domeCanvas" width="300" height="300" aria-label="Light direction: drag the dot"></canvas><div class="angles" id="angles"></div></div>
<div id="tools" class="panel" role="group" aria-label="Viewer settings">
  <div class="seg" role="group" aria-label="Look">
    <button data-mode="0" aria-pressed="true" title="1">Colour</button>
    <button data-mode="1" aria-pressed="false" title="2">Shading</button>
    <button data-mode="3" aria-pressed="false" title="3">Specular</button>
    <button data-mode="2" aria-pressed="false" title="4">Normals</button>
  </div>
  <label class="row">Brightness <input id="gain" type="range" min="0.2" max="2.5" step="0.01" value="1.0"></label>
  <label class="row">Ambient <input id="amb" type="range" min="0" max="0.8" step="0.01" value="0.15"></label>
  <label class="row">Shine <input id="spec" type="range" min="0" max="1.5" step="0.01" value="0.3"></label>
  <div style="display:flex; gap:6px; flex-wrap:wrap">
    <button class="plain" id="sweep" title="S">Sweep</button>
    <button class="plain" id="fit" title="F">Fit</button>
    <button class="plain" id="save" title="P">Save view</button>
    <button class="plain" id="help" title="?">Keys</button>
  </div>
</div>
<div id="scale" class="panel"><div class="bar" id="scalebar"></div><span id="scaletext"></span></div>
<div id="busy" class="panel">Loading tiles…</div>
<div id="keys" class="panel" role="dialog" aria-label="Keys">
  <h2>Keys</h2>
  <table>
    <tr><td><kbd>drag</kbd></td><td>Move about the image</td></tr>
    <tr><td><kbd>Shift</kbd> <kbd>drag</kbd></td><td>Move the light (or drag the dot on the dome)</td></tr>
    <tr><td><kbd>scroll</kbd> / pinch</td><td>Zoom in and out</td></tr>
    <tr><td><kbd>double-click</kbd></td><td>Zoom in there</td></tr>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>Turn the light</td></tr>
    <tr><td><kbd>↑</kbd> <kbd>↓</kbd></td><td>Raise and lower the light</td></tr>
    <tr><td><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> <kbd>4</kbd></td><td>Colour · shading · specular · normals</td></tr>
    <tr><td><kbd>+</kbd> <kbd>−</kbd></td><td>Zoom in and out</td></tr>
    <tr><td><kbd>F</kbd></td><td>Fit the whole image</td></tr>
    <tr><td><kbd>S</kbd></td><td>Sweep the light round</td></tr>
    <tr><td><kbd>P</kbd></td><td>Save the view as a picture</td></tr>
    <tr><td><kbd>?</kbd></td><td>Show or hide this list</td></tr>
  </table>
</div>
<div id="notice" class="panel" role="alert"></div>
<script type="application/json" id="meta">__META__</script>
<script>
"use strict";
const META = JSON.parse(document.getElementById("meta").textContent);
const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", {antialias: false, preserveDrawingBuffer: true});
const notice = document.getElementById("notice");
function say(html) { notice.innerHTML = html; notice.style.display = "block"; }
if (!gl) {
  say("<h2>This browser cannot show the viewer</h2><p>It needs WebGL 2, which every current browser has. " +
      "Try another browser, or open <a href='quick-view.html'>quick-view.html</a>.</p>");
  throw new Error("no WebGL 2");
}
document.getElementById("sub").textContent = [
  META.width + " × " + META.height + " px",
  META.mm_per_px ? (META.mm_per_px.toPrecision(3) + " mm per pixel") : "scale not measured",
  META.licence || ""].filter(Boolean).join(" · ");

// ── shaders ──────────────────────────────────────────────────────────────
const VS = `#version 300 es
in vec2 a_pos; in vec2 a_uv;
uniform vec2 u_scale; uniform vec2 u_offset; uniform vec2 u_view;
out vec2 v_uv;
void main() {
  vec2 p = a_pos * u_scale + u_offset;            // image pixels -> screen pixels
  gl_Position = vec4(p.x / u_view.x * 2.0 - 1.0, 1.0 - p.y / u_view.y * 2.0, 0.0, 1.0);
  v_uv = a_uv;
}`;
const FS = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_alb; uniform sampler2D u_nrm;
uniform vec3 u_light; uniform float u_gain; uniform float u_amb; uniform float u_spec; uniform int u_mode;
out vec4 o;
void main() {
  vec3 n = normalize(texture(u_nrm, v_uv).rgb * 2.0 - 1.0);
  vec3 a = texture(u_alb, v_uv).rgb;
  float d = max(dot(n, u_light), 0.0);
  vec3 h = normalize(u_light + vec3(0.0, 0.0, 1.0));
  float nh = max(dot(n, h), 0.0);
  vec3 col;
  if (u_mode == 0) col = a * (u_amb + d * u_gain) + vec3(u_spec * pow(nh, 24.0) * 0.5);
  else if (u_mode == 1) col = vec3(u_amb + d * u_gain) + vec3(u_spec * pow(nh, 24.0) * 0.5);
  else if (u_mode == 2) col = n * 0.5 + 0.5;
  else col = a * (u_amb + d * u_gain) * 0.7 + vec3((0.4 + u_spec) * pow(nh, 70.0));
  o = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;
function compile(type, src) {
  const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
const U = n => gl.getUniformLocation(prog, n);
const loc = { pos: gl.getAttribLocation(prog, "a_pos"), uv: gl.getAttribLocation(prog, "a_uv") };
const quad = gl.createBuffer();
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.enableVertexAttribArray(loc.pos); gl.vertexAttribPointer(loc.pos, 2, gl.FLOAT, false, 16, 0);
gl.enableVertexAttribArray(loc.uv); gl.vertexAttribPointer(loc.uv, 2, gl.FLOAT, false, 16, 8);

// ── state ────────────────────────────────────────────────────────────────
const S = { scale: 1, ox: 0, oy: 0, az: 315, el: 35, mode: 0, gain: 1.0, amb: 0.15, spec: 0.3, sweep: false };
const TILE = META.tile, OV = META.overlap, LV = META.levels;
function lightVec() {
  const a = S.az * Math.PI / 180, e = S.el * Math.PI / 180;
  return [Math.sin(a) * Math.cos(e), Math.cos(a) * Math.cos(e), Math.sin(e)];   // +Y north
}
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(innerWidth * dpr); canvas.height = Math.round(innerHeight * dpr);
  draw();
}
function fit() {
  const k = Math.min(canvas.width / META.width, canvas.height / META.height) * 0.92;
  S.scale = k; S.ox = (canvas.width - META.width * k) / 2; S.oy = (canvas.height - META.height * k) / 2;
  draw();
}

// ── tiles: loaded on demand, kept while there is room ────────────────────
const cache = new Map();      // key -> {alb, nrm, ready, used}
let loading = 0, broken = false;
function texFrom(img) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
  return t;
}
function loadImg(src) {
  return new Promise((ok, bad) => { const i = new Image(); i.onload = () => ok(i); i.onerror = bad; i.src = src; });
}
function tile(level, c, r) {
  const key = level + "/" + c + "_" + r;
  let t = cache.get(key);
  if (t) { t.used = performance.now(); return t; }
  t = { ready: false, used: performance.now() };
  cache.set(key, t);
  loading++; document.getElementById("busy").style.display = "block";
  const L = META.layers;
  Promise.all([loadImg(L.albedo.dir + "/" + key + "." + L.albedo.ext),
               loadImg(L.normals.dir + "/" + key + "." + L.normals.ext)]).then(([a, n]) => {
    try { t.alb = texFrom(a); t.nrm = texFrom(n); t.w = a.width; t.h = a.height; t.ready = true; }
    catch (e) { broken = true; blocked(); }
  }).catch(() => { t.failed = true; if (location.protocol === "file:") blocked(); })
    .finally(() => { loading--; if (!loading) document.getElementById("busy").style.display = "none"; draw(); trim(); });
  return t;
}
function blocked() {
  say("<h2>Open it through a web server</h2><p>Browsers will not relight pictures read straight " +
      "from a disk. Double-click <b>1_Mac.command</b> (Mac) or <b>2_Win.bat</b> (Windows) in this " +
      "folder, or put the folder on a website" +
      (META.quick ? "" : "") + ". Or open <a href='quick-view.html'>quick-view.html</a>, which " +
      "works from anywhere.</p>");
}
function trim() {
  if (cache.size < 260) return;
  const keys = [...cache.entries()].filter(([, t]) => t.ready).sort((a, b) => a[1].used - b[1].used);
  for (const [k, t] of keys.slice(0, cache.size - 200)) {
    gl.deleteTexture(t.alb); gl.deleteTexture(t.nrm); cache.delete(k);
  }
}
function levelFor(scale) {
  // level 0 is full size; choose the level whose pixels are about one screen pixel
  let l = Math.floor(Math.log2(1 / Math.max(scale, 1e-6)));
  return Math.max(0, Math.min(LV.length - 1, l));
}
function drawLevel(l, onlyReady) {
  const lv = LV[l], f = Math.pow(2, l);
  const x0 = Math.max(0, Math.floor((-S.ox / S.scale) / f / TILE)), y0 = Math.max(0, Math.floor((-S.oy / S.scale) / f / TILE));
  const x1 = Math.min(lv.cols - 1, Math.floor(((canvas.width - S.ox) / S.scale) / f / TILE));
  const y1 = Math.min(lv.rows - 1, Math.floor(((canvas.height - S.oy) / S.scale) / f / TILE));
  let missing = false;
  for (let r = y0; r <= y1; r++) for (let c = x0; c <= x1; c++) {
    const t = onlyReady ? cache.get(l + "/" + c + "_" + r) : tile(l, c, r);
    if (!t || !t.ready) { missing = true; continue; }
    // the tile's own pixels (inside the overlap), in image pixels at full size
    const px0 = c * TILE, py0 = r * TILE;
    const pw = Math.min(TILE, lv.w - px0), ph = Math.min(TILE, lv.h - py0);
    const ix = c > 0 ? OV : 0, iy = r > 0 ? OV : 0;
    const u0 = ix / t.w, v0 = iy / t.h, u1 = (ix + pw) / t.w, v1 = (iy + ph) / t.h;
    const X0 = px0 * f, Y0 = py0 * f, X1 = (px0 + pw) * f, Y1 = (py0 + ph) * f;
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([X0, Y0, u0, v0, X1, Y0, u1, v0, X0, Y1, u0, v1,
                                                     X1, Y1, u1, v1]), gl.STREAM_DRAW);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, t.alb);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, t.nrm);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  return missing;
}
let pending = false;
function draw() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.169, 0.169, 0.169, 1); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(prog); gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.uniform2f(U("u_scale"), S.scale, S.scale); gl.uniform2f(U("u_offset"), S.ox, S.oy);
    gl.uniform2f(U("u_view"), canvas.width, canvas.height);
    gl.uniform1i(U("u_alb"), 0); gl.uniform1i(U("u_nrm"), 1);
    gl.uniform3fv(U("u_light"), lightVec()); gl.uniform1f(U("u_gain"), S.gain);
    gl.uniform1f(U("u_amb"), S.amb); gl.uniform1f(U("u_spec"), S.spec); gl.uniform1i(U("u_mode"), S.mode);
    const want = levelFor(S.scale);
    // coarser levels first, already loaded, so nothing is blank while finer tiles arrive
    for (let l = LV.length - 1; l > want; l--) drawLevel(l, true);
    drawLevel(want, false);
    drawDome(); drawScale();
  });
}

// ── the light dome ───────────────────────────────────────────────────────
const dome = document.getElementById("domeCanvas"), dc = dome.getContext("2d");
function drawDome() {
  const W = dome.width, R = W * 0.42, cx = W / 2, cy = W / 2;
  dc.clearRect(0, 0, W, W);
  dc.fillStyle = "#1A1B1E"; dc.beginPath(); dc.arc(cx, cy, R, 0, 7); dc.fill();
  dc.strokeStyle = "#3E4046"; dc.lineWidth = 2;
  for (const k of [1, 0.66, 0.33]) { dc.beginPath(); dc.arc(cx, cy, R * k, 0, 7); dc.stroke(); }
  dc.beginPath(); dc.moveTo(cx - R, cy); dc.lineTo(cx + R, cy); dc.moveTo(cx, cy - R); dc.lineTo(cx, cy + R); dc.stroke();
  dc.fillStyle = "#B4B6BC"; dc.font = "bold 22px -apple-system, sans-serif"; dc.textAlign = "center";
  dc.fillText("N", cx, cy - R - 8);
  for (const L of (META.lights || [])) {
    const n = Math.hypot(L[0], L[1], L[2]) || 1, x = L[0] / n, y = L[1] / n;
    dc.fillStyle = "#7CC0EE"; dc.beginPath(); dc.arc(cx + x * R, cy - y * R, 5, 0, 7); dc.fill();
  }
  const v = lightVec();
  dc.fillStyle = "#E6A93E"; dc.strokeStyle = "#222326"; dc.lineWidth = 4;
  dc.beginPath(); dc.arc(cx + v[0] * R, cy - v[1] * R, 13, 0, 7); dc.fill(); dc.stroke();
  document.getElementById("angles").textContent = Math.round(S.az) + "° · " + Math.round(S.el) + "° up";
}
function domeAt(e) {
  const r = dome.getBoundingClientRect(), W = dome.width, R = W * 0.42;
  const x = ((e.clientX - r.left) / r.width * W - W / 2) / R, y = -((e.clientY - r.top) / r.height * W - W / 2) / R;
  const d = Math.min(1, Math.hypot(x, y));
  S.az = (Math.atan2(x, y) * 180 / Math.PI + 360) % 360;
  S.el = Math.max(2, Math.acos(d) * 180 / Math.PI);
  draw();
}
let domeDrag = false;
dome.addEventListener("pointerdown", e => { domeDrag = true; dome.setPointerCapture(e.pointerId); domeAt(e); });
dome.addEventListener("pointermove", e => { if (domeDrag) domeAt(e); });
dome.addEventListener("pointerup", () => { domeDrag = false; });

// ── the scale bar ────────────────────────────────────────────────────────
function drawScale() {
  const box = document.getElementById("scale");
  if (!META.mm_per_px) { box.style.display = "none"; return; }
  box.style.display = "block";
  const dpr = canvas.width / innerWidth;
  const mmPerCss = META.mm_per_px / S.scale * dpr, target = 110 * mmPerCss;
  const p = Math.pow(10, Math.floor(Math.log10(target))), nice = [1, 2, 5, 10].map(k => k * p).filter(v => v <= target).pop() || p;
  document.getElementById("scalebar").style.width = (nice / mmPerCss) + "px";
  document.getElementById("scaletext").textContent = nice >= 1 ? nice + " mm" : (nice * 1000).toFixed(0) + " µm";
}

// ── moving about ─────────────────────────────────────────────────────────
const pointers = new Map();
let last = null, lightDrag = false;
function zoomAt(k, x, y) {
  const dpr = canvas.width / innerWidth, px = x * dpr, py = y * dpr;
  const ns = Math.max(0.01, Math.min(16, S.scale * k));
  S.ox = px - (px - S.ox) * ns / S.scale; S.oy = py - (py - S.oy) * ns / S.scale; S.scale = ns; draw();
}
canvas.addEventListener("pointerdown", e => {
  canvas.setPointerCapture(e.pointerId); pointers.set(e.pointerId, e);
  last = e; lightDrag = e.shiftKey; canvas.classList.add("dragging");
});
canvas.addEventListener("pointermove", e => {
  if (!pointers.has(e.pointerId)) return;
  const prev = pointers.get(e.pointerId); pointers.set(e.pointerId, e);
  if (pointers.size === 2) {                     // pinch
    const [a, b] = [...pointers.values()];
    const other = [...pointers.entries()].find(([id]) => id !== e.pointerId)[1];
    const d0 = Math.hypot(prev.clientX - other.clientX, prev.clientY - other.clientY);
    const d1 = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    if (d0 > 0) zoomAt(d1 / d0, (a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2);
    return;
  }
  const dpr = canvas.width / innerWidth;
  if (lightDrag) {
    S.az = (S.az + (e.clientX - prev.clientX) * 0.6 + 360) % 360;
    S.el = Math.max(2, Math.min(89, S.el - (e.clientY - prev.clientY) * 0.3));
  } else { S.ox += (e.clientX - prev.clientX) * dpr; S.oy += (e.clientY - prev.clientY) * dpr; }
  draw();
});
for (const ev of ["pointerup", "pointercancel"]) canvas.addEventListener(ev, e => {
  pointers.delete(e.pointerId); if (!pointers.size) canvas.classList.remove("dragging");
});
canvas.addEventListener("wheel", e => { e.preventDefault(); zoomAt(Math.pow(0.9985, e.deltaY), e.clientX, e.clientY); }, {passive: false});
canvas.addEventListener("dblclick", e => zoomAt(2, e.clientX, e.clientY));

// ── controls and keys ────────────────────────────────────────────────────
const modeButtons = [...document.querySelectorAll(".seg button")];
function setMode(m) { S.mode = m; modeButtons.forEach(b => b.setAttribute("aria-pressed", String(+b.dataset.mode === m))); draw(); }
modeButtons.forEach(b => b.addEventListener("click", () => setMode(+b.dataset.mode)));
for (const [id, key] of [["gain", "gain"], ["amb", "amb"], ["spec", "spec"]])
  document.getElementById(id).addEventListener("input", e => { S[key] = +e.target.value; draw(); });
function sweep() {
  S.sweep = !S.sweep; document.getElementById("sweep").textContent = S.sweep ? "Stop" : "Sweep";
  const step = () => { if (!S.sweep) return; S.az = (S.az + 1.2) % 360; draw(); requestAnimationFrame(step); };
  step();
}
function save() {
  canvas.toBlob(b => {
    const a = document.createElement("a"); a.href = URL.createObjectURL(b);
    a.download = (META.object_id || "view") + "_relit.png"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  });
}
function keys() { const k = document.getElementById("keys"); k.style.display = k.style.display === "block" ? "none" : "block"; }
document.getElementById("sweep").onclick = sweep;
document.getElementById("fit").onclick = fit;
document.getElementById("save").onclick = save;
document.getElementById("help").onclick = keys;
addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key;
  if (k === "ArrowLeft") S.az = (S.az + 355) % 360; else if (k === "ArrowRight") S.az = (S.az + 5) % 360;
  else if (k === "ArrowUp") S.el = Math.min(89, S.el + 3); else if (k === "ArrowDown") S.el = Math.max(2, S.el - 3);
  else if ("1234".includes(k)) setMode([0, 1, 3, 2][+k - 1]);
  else if (k === "+" || k === "=") zoomAt(1.25, innerWidth / 2, innerHeight / 2);
  else if (k === "-" || k === "_") zoomAt(0.8, innerWidth / 2, innerHeight / 2);
  else if (k === "f" || k === "F") fit(); else if (k === "s" || k === "S") sweep();
  else if (k === "p" || k === "P") save(); else if (k === "?" || k === "/") keys();
  else if (k === "Escape") document.getElementById("keys").style.display = "none";
  else return;
  e.preventDefault(); draw();
});
addEventListener("resize", () => { resize(); });
resize(); fit();
</script>
</body>
</html>
"""
