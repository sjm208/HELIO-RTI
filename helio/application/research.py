"""Research tools, GUI-free (SCH: unmixing, micro-features, toolpaths, focus
stacking, cross-object registration, recto/verso, inscription zones).

The classic tool windows' computations, moved so both apps compute the same
results.  Fixed on the way (each named in its function):

* micro-features reported the surface's height at a feature's centre as its
  "depth" and filtered on that, and never had a scale for millimetres
  (KD-046, KD-044);
* a toolpath's length was the pixel count of its *dilated* skeleton, about
  three times the stroke at the default dilation, and its "depth" the height
  of the surface along it (KD-047);
* focus stacking said "Laplacian pyramid" but picked each pixel from one
  frame, which leaves seams; it now fuses Laplacian pyramids (Burt &
  Kolczynski 1993), the per-pixel choice kept as an option;
* registration's "Affine" was a similarity (4 degrees of freedom); all three
  models are offered by name, with the control-point residual reported.
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Callable

import cv2
import numpy as np


# ── reflectance unmixing ─────────────────────────────────────────────────

def unmix(stack, n: int = 3, method: str = "PCA", progress: Callable | None = None):
    """Decompose how each pixel answers the lights into *n* independent
    channels (PCA, or FastICA) of the (frames, h, w) 0–255 stack
    (export_images.load_stack, decoded as the solve decodes it).  Different
    materials, layers or coatings respond differently to the light, so they
    separate.  This works on the light stack itself: HELIO 2 called it
    "HSH" unmixing, but no HSH or PTM basis is involved.  Returns
    (components (n, h, w), share of variance per component)."""
    S = np.asarray(stack, np.float32) / 255.0
    if S.ndim != 3 or S.shape[0] < n:
        raise ValueError(f"Need at least {n} readable images.")
    N, h, w = S.shape
    X = S.reshape(N, -1).T
    if progress:
        progress(0.5)
    # The mix is learnt from a sample of pixels and applied to all of them:
    # the components are a fixed linear mix of the frames, so a few hundred
    # thousand pixels fix it as well as millions, at a fraction of the time
    # and memory.
    fit = X
    if len(X) > 300_000:
        fit = X[np.random.default_rng(0).choice(len(X), 300_000, replace=False)]
    if method == "ICA":
        from sklearn.decomposition import FastICA
        model = FastICA(n_components=n, max_iter=500, random_state=0).fit(fit)
    else:
        from sklearn.decomposition import PCA
        model = PCA(n_components=n, whiten=True).fit(fit)
    comps = np.empty((len(X), n), np.float32)
    for a in range(0, len(X), 1_000_000):
        comps[a:a + 1_000_000] = model.transform(X[a:a + 1_000_000])
    if method == "ICA":
        # ICA has no order of its own: largest share of the signal first
        v = comps.var(axis=0)
        order = np.argsort(-v)
        comps = comps[:, order]
        var = [float(x) for x in (v[order] / v.sum())]
    else:
        var = [float(v) for v in model.explained_variance_ratio_]
    return np.stack([comps[:, i].reshape(h, w) for i in range(n)]), var


# Perceptually uniform maps for the component images (the classic used jet,
# hot, cool, spring and winter, whose bands invent edges: Crameri et al. 2020).
UNMIX_MAPS = (cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_MAGMA, cv2.COLORMAP_CIVIDIS,
              cv2.COLORMAP_INFERNO, cv2.COLORMAP_PLASMA)


# How the components can be saved: (label, maps' format, values).  Values:
# None, "grey16" (each component stretched to 16-bit grey, for any image
# editor) or "f32" (the values themselves, 32-bit float, for analysis).
UNMIX_SAVE = (("Colour maps (PNG)", ".png", None),
              ("Colour maps (JPEG, smaller)", ".jpg", None),
              ("Colour maps + 16-bit greyscale TIFFs (for Photoshop)", ".png", "grey16"),
              ("Colour maps + 32-bit value TIFFs (for analysis)", ".png", "f32"))


def _stretch(ch):
    """0-1, from the 0.5th to the 99.5th percentile: one hot pixel does not
    flatten the rest (the plain min-max did)."""
    lo, hi = (float(v) for v in np.percentile(ch, (0.5, 99.5)))
    return np.clip((ch - lo) / (hi - lo + 1e-9), 0, 1)


def save_components(comps, base: str, ext: str = ".png", values: str | None = "f32",
                    composite: bool = True) -> list[str]:
    """Each component as a false-colour map (*ext*: .png, .jpg or .tif), with
    its values when asked (*values*: see UNMIX_SAVE), and with three or more
    components a false-colour composite: the first three as red, green, blue."""
    out = []
    for i, ch in enumerate(comps):
        s = _stretch(ch)
        p = f"{base}_component_{i + 1}{ext}"
        cv2.imwrite(p, cv2.applyColorMap((s * 255).astype(np.uint8), UNMIX_MAPS[i % len(UNMIX_MAPS)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == ".jpg" else [])
        out.append(p)
        if values == "f32":
            raw = f"{base}_component_{i + 1}_32f.tif"
            cv2.imwrite(raw, np.asarray(ch, np.float32), [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            out.append(raw)
        elif values == "grey16":
            g = f"{base}_component_{i + 1}_16bit.tif"
            cv2.imwrite(g, (s * 65535).astype(np.uint16))
            out.append(g)
    if composite and len(comps) >= 3:
        rgb = np.dstack([_stretch(comps[k]) for k in (2, 1, 0)])      # BGR for OpenCV
        c = f"{base}_composite_C1-C3{ext}"
        cv2.imwrite(c, (rgb * 255).astype(np.uint8),
                    [cv2.IMWRITE_JPEG_QUALITY, 92] if ext == ".jpg" else [])
        out.append(c)
    return out


# ── micro-features (pinholes, stylus marks, micro-depressions) ───────────

def micro_features(depth, *, min_r: float = 2.0, max_r: float = 12.0, threshold: float = 0.02,
                   min_depth: float = 0.01, depressions_only: bool = True, mm_per_px=None,
                   z_mm_per_unit=None, heights=None) -> list[dict]:
    """Laplacian-of-Gaussian blobs in a 0–1 height map, each with its local
    depth: how far its centre lies below (or a bump above) the ring around
    it, from r to 2r.  HELIO 2 used the surface's height at the centre as
    the depth (KD-046) and could not give positions in mm (KD-044).

    Found on *depth* (0-1, the thresholds are on it); the depth in µm is read
    from *heights* (true heights in pixels of height, helio.application.heights)
    with *z_mm_per_unit* (mm per pixel of height), and is None without both."""
    hts = None if heights is None else np.asarray(heights, np.float32)
    from skimage.feature import blob_log
    raw = np.asarray(depth, np.float32)
    if min_r >= max_r:
        raise ValueError("The minimum radius must be smaller than the maximum.")
    work = (raw.max() - raw) if depressions_only else raw
    blobs = blob_log(work, min_sigma=min_r / math.sqrt(2), max_sigma=max_r / math.sqrt(2),
                     num_sigma=max(5, int((max_r - min_r) / 2)), threshold=threshold,
                     exclude_border=True)
    H, W = raw.shape
    yy, xx = np.mgrid[0:H, 0:W]
    rows = []
    for y, x, sigma in blobs:
        yi, xi = int(round(y)), int(round(x))
        if not (0 <= yi < H and 0 <= xi < W):
            continue
        r = sigma * math.sqrt(2)
        R = int(math.ceil(2 * r)) + 1
        y0, y1, x0, x1 = max(0, yi - R), min(H, yi + R + 1), max(0, xi - R), min(W, xi + R + 1)
        d = np.hypot(yy[y0:y1, x0:x1] - yi, xx[y0:y1, x0:x1] - xi)
        patch = raw[y0:y1, x0:x1]
        ring = patch[(d >= r) & (d <= 2 * r)]
        core = patch[d <= max(1.0, r * 0.5)]
        if ring.size == 0 or core.size == 0:
            continue
        local = float(np.median(ring) - np.mean(core))
        if not depressions_only:
            local = abs(local)
        if local < min_depth:
            continue
        true_px = None
        if hts is not None and z_mm_per_unit:
            hp = hts[y0:y1, x0:x1]
            true_px = float(np.median(hp[(d >= r) & (d <= 2 * r)]) - np.mean(hp[d <= max(1.0, r * 0.5)]))
            if not depressions_only:
                true_px = abs(true_px)
        rows.append({
            "id": len(rows) + 1, "x_px": xi, "y_px": yi, "radius_px": round(float(r), 2),
            "depth_norm": round(local, 4),
            "depth_um": round(true_px * float(z_mm_per_unit) * 1000, 2) if true_px is not None else None,
            "x_mm": round(xi * float(mm_per_px), 3) if mm_per_px else None,
            "y_mm": round(yi * float(mm_per_px), 3) if mm_per_px else None,
            "radius_mm": round(float(r) * float(mm_per_px), 4) if mm_per_px else None})
    return rows


def export_micro_features(rows, depth, out_dir: str, safe: str, title: str) -> list[str]:
    """CSV, GeoJSON (mm when scaled) and an annotated PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    cp = os.path.join(out_dir, f"{safe}_micro_features.csv")
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        w.writeheader()
        w.writerows(rows)
    feats = [{"type": "Feature",
              "geometry": {"type": "Point",
                           "coordinates": [r["x_mm"] if r["x_mm"] is not None else r["x_px"],
                                           r["y_mm"] if r["y_mm"] is not None else r["y_px"]]},
              "properties": dict(r)} for r in rows]
    gp = os.path.join(out_dir, f"{safe}_micro_features.geojson")
    with open(gp, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f, indent=2)
    d = np.asarray(depth)
    fig, ax = plt.subplots(figsize=(12, 12 * d.shape[0] / d.shape[1]), facecolor="#111111")
    ax.imshow(d, cmap="gray", vmin=0, vmax=1)
    for r in rows:
        ax.add_patch(mp.Circle((r["x_px"], r["y_px"]), r["radius_px"], color="#ff4466", fill=False,
                               linewidth=0.7, alpha=0.85))
        ax.text(r["x_px"] + r["radius_px"] + 1, r["y_px"], str(r["id"]), color="#ff4466",
                fontsize=4.5, va="center")
    ax.set_title(title, color="white", fontsize=9)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    pp = os.path.join(out_dir, f"{safe}_micro_features_annotated.png")
    fig.savefig(pp, dpi=200, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    return [cp, gp, pp]


# ── toolpaths (scratch and groove networks) ──────────────────────────────

def groove_smoothing(w: int, h: int) -> float:
    """The curvature's smoothing for a capture this size, px: about a third
    of a tooled groove's width.  Unsmoothed, a 100 MP binding's grooves
    (about 30 px wide) break into speckle and chains of little loops."""
    return round(max(1.0, max(w, h) / 3500.0), 1)


def toolpaths(depth, *, normals=None, use_normals: bool = False, min_len: int = 20,
              threshold: float = 0.15, dilation: int = 1, mm_per_px=None,
              z_mm_per_unit=None, smooth: float | None = None
              ) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Grooves traced as strokes: where the surface curves up into a groove
    more than *threshold* × the strongest groove, thinned to one pixel, joined by *dilation* px, each stroke
    with its length, groove depth, width and direction.

    Length is counted on the one-pixel skeleton (√2 for diagonal steps);
    HELIO 2 counted the dilated pixels, about three times the stroke
    (KD-047).  Groove depth is how far the stroke lies below the surface
    around it (a σ = 6 px reference), not the surface's height.  Returns
    (strokes, labels, skeleton)."""
    from skimage.morphology import skeletonize
    dep = np.asarray(depth, np.float32)
    sig = float(smooth) if smooth else groove_smoothing(dep.shape[1], dep.shape[0])
    if use_normals and normals is not None:
        # the surface's slopes straight from the normals (+Y toward the top of the
        # image, so a row step down is −y), and their divergence: no integration
        n = np.asarray(normals, np.float32)
        nz = np.maximum(n[..., 2], 0.05)
        gx, gy = -n[..., 0] / nz, n[..., 1] / nz
        if sig > 1.0:
            gx, gy = cv2.GaussianBlur(gx, (0, 0), sig), cv2.GaussianBlur(gy, (0, 0), sig)
        lap = cv2.Sobel(gx, cv2.CV_32F, 1, 0) + cv2.Sobel(gy, cv2.CV_32F, 0, 1)
    else:
        lap = cv2.Laplacian(cv2.GaussianBlur(dep, (0, 0), sig), cv2.CV_32F, ksize=3)
    # a groove's floor curves up (positive Laplacian of the height).  HELIO 2
    # min–max scaled the negated Laplacian and kept everything above the
    # threshold: the flat background (≈0.7 on that scale) and the groove
    # shoulders, never the floors (KD-047).  The threshold is now a fraction
    # of the strongest groove curvature.
    g = np.maximum(lap, 0)
    top = float(np.percentile(g[g > 0], 99.5)) if np.any(g > 0) else 0.0
    mask = (g / top > float(threshold)) if top > 0 else np.zeros_like(dep, bool)
    # pinholes in a groove's band thin into loops round them: filled first
    holes = int(4 * sig * sig + 16)
    inv = (~mask).astype(np.uint8)
    nh, hl, hs, _c = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = np.zeros(nh, bool)
    small[1:] = hs[1:, cv2.CC_STAT_AREA] <= holes
    mask = mask | small[hl]
    skel = skeletonize(mask).astype(np.uint8) * 255
    joined = cv2.dilate(skel, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
                        iterations=int(dilation)) if dilation > 0 else skel
    n, labels, _stats, cents = cv2.connectedComponentsWithStats(joined, connectivity=8)
    ref = cv2.GaussianBlur(dep, (0, 0), 6.0)
    groove = ref - dep
    on = skel > 0
    # diagonal neighbours count √2: length ≈ orthogonal steps + √2 × diagonal steps
    # Each stroke is looked at inside its own bounding box: comparing the
    # whole image once per stroke took hours on a 100 MP tooled binding.
    from scipy import ndimage
    boxes = ndimage.find_objects(labels)
    strokes = []
    for lbl in range(1, n):
        sl = boxes[lbl - 1] if lbl - 1 < len(boxes) else None
        if sl is None:
            continue
        own = labels[sl] == lbl
        ys, xs = np.where(on[sl] & own)
        if ys.size < 2:
            continue
        ys, xs = ys + sl[0].start, xs + sl[1].start
        sub = np.zeros((ys.max() - ys.min() + 3, xs.max() - xs.min() + 3), np.uint8)
        sub[ys - ys.min() + 1, xs - xs.min() + 1] = 1
        orth = int((sub[:, 1:] & sub[:, :-1]).sum() + (sub[1:, :] & sub[:-1, :]).sum())
        diag = int((sub[1:, 1:] & sub[:-1, :-1]).sum() + (sub[1:, :-1] & sub[:-1, 1:]).sum())
        length = orth + math.sqrt(2) * max(0, (ys.size - 1) - orth) if diag else float(orth)
        length = max(float(length), float(ys.size - 1))
        if length < min_len:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        _, _, vt = np.linalg.svd(pts - pts.mean(axis=0), full_matrices=False)
        vx, vy = float(vt[0, 0]), float(vt[0, 1])
        angle = round(math.degrees(math.atan2(vy, vx)) % 180, 1) % 180
        gd = groove[ys, xs]
        k = z_mm_per_unit * 1000 if z_mm_per_unit else None
        width = float((mask[sl] & own).sum()) / max(length, 1.0)
        strokes.append({
            "id": lbl, "length_px": round(length, 1),
            "length_mm": round(length * mm_per_px, 3) if mm_per_px else None,
            "mean_groove_depth": round(float(gd.mean()), 5),
            "max_groove_depth": round(float(gd.max()), 5),
            "mean_groove_depth_um": round(float(gd.mean()) * k, 2) if k else None,
            "width_px": round(width, 2),
            "angle_deg": round(angle, 1),
            "bearing_deg": round((90.0 + angle) % 180, 1),
            "cx": round(float(cents[lbl][0]), 1), "cy": round(float(cents[lbl][1]), 1)})
    return strokes, labels, skel


STROKE_COLOURS = ('#ff4444', '#ff8800', '#ffee00', '#44ff44', '#00aaff', '#aa44ff', '#ff44aa',
                  '#ffffff')


def stroke_colours(strokes, by_direction: bool = False, by_depth: bool = False) -> dict:
    """{stroke id: (r, g, b)}.  Neither: the eight colours in turn, to tell
    neighbouring strokes apart.  By direction: the hue is the way the line
    runs (0-180 degrees round the colour wheel, so a line and its reverse
    match).  By depth: deeper grooves brighter, ranked within this result.
    Both: hue for direction, brightness for depth."""
    import colorsys
    if not strokes:
        return {}
    if by_depth:
        d = np.array([s["mean_groove_depth"] for s in strokes], np.float64)
        rank = np.argsort(np.argsort(d)) / max(len(d) - 1, 1)          # 0 shallowest .. 1 deepest
    out = {}
    for i, s in enumerate(strokes):
        if by_direction:
            v = 0.30 + 0.70 * float(rank[i]) if by_depth else 1.0
            rgb = colorsys.hsv_to_rgb((float(s["angle_deg"]) % 180.0) / 180.0, 0.85, v)
        elif by_depth:
            rgb = _depth_ramp(float(rank[i]))
        else:
            h = STROKE_COLOURS[i % len(STROKE_COLOURS)].lstrip("#")
            out[s["id"]] = tuple(int(h[k:k + 2], 16) for k in (0, 2, 4))
            continue
        out[s["id"]] = tuple(int(round(c * 255)) for c in rgb)
    return out


def _depth_ramp(t: float):
    """Shallow to deep: dim violet through orange to bright yellow (inferno)."""
    import matplotlib.cm as cm
    return cm.inferno(0.25 + 0.75 * min(max(t, 0.0), 1.0))[:3]


def colour_key(by_direction: bool, by_depth: bool, width: int = 280, height: int = 22):
    """A strip saying what the colours mean, uint8 RGB, or None for the plain colours."""
    import colorsys
    if not (by_direction or by_depth):
        return None
    t = np.linspace(0.0, 1.0, width)
    if by_direction:
        row = np.array([colorsys.hsv_to_rgb(x, 0.85, 1.0) for x in t])
    else:
        row = np.array([_depth_ramp(x) for x in t])
    img = np.repeat((row * 255).astype(np.uint8)[None], height, axis=0)
    if by_direction and by_depth:
        img = (img.astype(np.float32) * np.linspace(0.30, 1.0, height)[:, None, None]).astype(np.uint8)
    return img


def export_toolpaths(strokes, labels, skel, albedo, out_dir: str, safe: str,
                     colours: dict | None = None) -> list[str]:
    """An SVG of the strokes over the albedo, a PNG preview and the CSV.
    *colours*: {stroke id: (r, g, b)} (stroke_colours); the eight in turn by default."""
    import base64
    import io
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    H, W = labels.shape
    buf = io.BytesIO()
    Image.fromarray(np.asarray(albedo).astype(np.uint8)).save(buf, format="PNG")
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
             f'<image href="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" '
             f'width="{W}" height="{H}"/>']
    vis = cv2.cvtColor(np.asarray(albedo).astype(np.uint8), cv2.COLOR_RGB2BGR).copy()
    on = skel > 0
    from scipy import ndimage
    boxes = ndimage.find_objects(labels)
    for i, s in enumerate(strokes):
        sl = boxes[s["id"] - 1]
        y0, x0 = sl[0].start, sl[1].start
        sk = (on[sl] & (labels[sl] == s["id"])).astype(np.uint8)
        ys, xs = np.where(sk)
        ys, xs = ys + y0, xs + x0
        col = (("#%02x%02x%02x" % colours[s["id"]]) if colours
               else STROKE_COLOURS[i % len(STROKE_COLOURS)])
        # the outline of a one-pixel line runs along it and back: drawn open, it is the line
        # (HELIO 2 joined the pixels in raster order, a zig-zag across the stroke)
        cs, _ = cv2.findContours(sk, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            pts = " ".join(f"{x + 0.5 + x0},{y + 0.5 + y0}" for x, y in c.reshape(-1, 2))
            lines.append(f'<polyline class="stroke_{s["id"]}" points="{pts}" stroke="{col}" '
                         f'stroke-width="1.5" fill="none" opacity="0.85"/>')
        h = col.lstrip("#")
        vis[ys, xs] = tuple(int(h[k:k + 2], 16) for k in (4, 2, 0))
    lines.append("</svg>")
    sp = os.path.join(out_dir, f"{safe}_toolpaths.svg")
    with open(sp, "w") as f:
        f.write("\n".join(lines))
    pp = os.path.join(out_dir, f"{safe}_toolpaths.png")
    cv2.imwrite(pp, vis)
    cp = os.path.join(out_dir, f"{safe}_toolpath_stats.csv")
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(strokes[0].keys()) if strokes else ["id"])
        w.writeheader()
        w.writerows(strokes)
    return [sp, pp, cp]


# ── focus stacking ───────────────────────────────────────────────────────

def _align_ecc(imgs):
    ref = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = [imgs[0]]
    for img in imgs[1:]:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        try:
            M = np.eye(2, 3, dtype=np.float32)
            _, M = cv2.findTransformECC(ref, g, M, cv2.MOTION_TRANSLATION)
            out.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP))
        except cv2.error:
            out.append(img)
    return out


def _select(imgs_f):
    """The classic per-pixel choice: each pixel from the sharpest frame."""
    maps = [cv2.GaussianBlur(cv2.Laplacian(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY), cv2.CV_32F) ** 2,
                             (15, 15), 0) for i in imgs_f]
    best = np.argmax(np.stack(maps), axis=0)
    arr = np.stack(imgs_f)
    H, W = best.shape
    r, c = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return arr[best, r, c]


def _pyramid(imgs_f, levels: int = 5):
    """Laplacian-pyramid fusion (Burt & Kolczynski 1993): at each detail
    level keep the coefficient with the largest magnitude, average the
    coarsest level.  No seams where the sharpest frame changes."""
    H, W = imgs_f[0].shape[:2]
    levels = max(1, min(levels, int(math.log2(max(2, min(H, W)) / 8))))

    def lap_pyr(img):
        g = [img]
        for _ in range(levels):
            g.append(cv2.pyrDown(g[-1]))
        pyr = [g[i] - cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
               for i in range(levels)]
        return pyr + [g[-1]]
    pyrs = [lap_pyr(i) for i in imgs_f]
    fused = []
    for lv in range(levels):
        stack = np.stack([p[lv] for p in pyrs])
        mag = np.abs(stack).sum(axis=-1) if stack.ndim == 4 else np.abs(stack)
        idx = np.argmax(mag, axis=0)
        r, c = np.meshgrid(np.arange(idx.shape[0]), np.arange(idx.shape[1]), indexing="ij")
        fused.append(stack[idx, r, c])
    fused.append(np.mean([p[-1] for p in pyrs], axis=0))
    out = fused[-1]
    for lv in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(fused[lv].shape[1], fused[lv].shape[0])) + fused[lv]
    return out


def focus_stack(paths, n_lights: int, n_frames: int, *, align: bool = True,
                method: str = "pyramid", progress: Callable | None = None) -> list[np.ndarray]:
    """One sharp image per light from its focus bracket.  *paths* sorted, in
    light-major order (light 1's frames, then light 2's …).  Keeps the
    source bit depth (8 or 16 bit)."""
    from helio.io.capture import natural_key
    paths = sorted(paths, key=natural_key)        # shot order: x-2 before x-10
    if len(paths) != n_lights * n_frames:
        raise ValueError(f"Expected {n_lights * n_frames} images, got {len(paths)}.")
    merged = []
    for li in range(n_lights):
        bracket, depth = [], 255.0
        for fi in range(n_frames):
            p = paths[li * n_frames + fi]
            from helio.core.frame_read import read_frame
            from helio.io.raw_loader import is_raw
            raw = read_frame(p, cv2.IMREAD_COLOR, sharp=True) if is_raw(p) else cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"Cannot read {os.path.basename(p)}")
            if raw.ndim == 2:
                raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            depth = 65535.0 if raw.dtype == np.uint16 else 255.0
            bracket.append(cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB))
        if align:
            bracket = _align_ecc(bracket)
        f = [b.astype(np.float32) / depth for b in bracket]
        out = _select(f) if method == "select" else _pyramid(f)
        merged.append((np.clip(out, 0, 1) * depth).astype(np.uint16 if depth > 255 else np.uint8))
        if progress:
            progress((li + 1) / n_lights)
    return merged


# ── cross-object registration ────────────────────────────────────────────

MODELS = ("Similarity", "Affine", "Projective")


def register_points(pts_a, pts_b, model: str = "Similarity", *, mirror: bool = False,
                    width_a: int = 0):
    """The transform taking control points on image A to image B, and the
    RMS distance (px) between B's points and A's transformed points.
    Similarity: rotation, uniform scale, shift (the classic "Affine");
    Affine: six degrees of freedom; Projective: a homography (RANSAC)."""
    src = np.array(pts_a, np.float32)
    dst = np.array(pts_b, np.float32)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError(f"Need at least 4 matching points on each (A={len(src)}, B={len(dst)}).")
    if mirror:
        src[:, 0] = (width_a - 1) - src[:, 0]
    if model == "Projective":
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        if M is None:
            raise ValueError("No projective fit for these points.")
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), M).reshape(-1, 2)
    else:
        fn = cv2.estimateAffine2D if model == "Affine" else cv2.estimateAffinePartial2D
        M, _ = fn(src, dst)
        if M is None:
            raise ValueError("No fit for these points.")
        proj = src @ M[:, :2].T + M[:, 2]
    rms = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
    return M, rms


def warp_to(img_a, M, shape_b, *, mirror: bool = False):
    a = np.fliplr(img_a) if mirror else img_a
    h, w = shape_b[:2]
    if M.shape == (3, 3):
        return cv2.warpPerspective(np.ascontiguousarray(a), M, (w, h))
    return cv2.warpAffine(np.ascontiguousarray(a), M, (w, h))


def blend(warped, img_b, alpha: float):
    b = np.asarray(img_b, np.float32) / 255.0
    if b.ndim == 2:
        b = np.stack([b] * 3, axis=-1)
    a = np.asarray(warped, np.float32) / 255.0
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return (np.clip(a * alpha + b * (1 - alpha), 0, 1) * 255).astype(np.uint8)


def export_registration(path: str, composite, warped, img_b, M=None, rms=None, model="") -> list[str]:
    base = path.rsplit(".", 1)[0]
    cv2.imwrite(path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    cv2.imwrite(base + "_A_registered.png", cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    cv2.imwrite(base + "_B_original.png", cv2.cvtColor(np.asarray(img_b), cv2.COLOR_RGB2BGR))
    files = [path, base + "_A_registered.png", base + "_B_original.png"]
    if M is not None:
        with open(base + "_transform.json", "w") as f:
            json.dump({"model": model, "matrix": np.asarray(M).tolist(),
                       "rms_px": None if rms is None else float(rms)}, f, indent=2)
        files.append(base + "_transform.json")
    return files


# ── recto / verso ────────────────────────────────────────────────────────

# The two faces of one object: a leaf, a coin or medal, a seal and its matrix.
FACE_PAIRS = {"Recto": "Verso", "Verso": "Recto", "Obverse": "Reverse", "Reverse": "Obverse",
              "Matrix": "Impression", "Impression": "Matrix"}


def recto_verso_link(path: str, *, object_id: str, this_surface: str, partner_file: str,
                     auto_flip_partner: bool, solver: str, normal_engine: str,
                     die_axis: int | None = None) -> dict:
    import datetime
    link = {"object_id": object_id or "unknown", "this_surface": this_surface,
            "partner_surface": FACE_PAIRS.get(this_surface, "Recto"),
            "partner_file": partner_file, "auto_flip_partner": bool(auto_flip_partner),
            "linked_at": datetime.datetime.now().isoformat(), "solver": solver,
            "normal_engine": normal_engine}
    if die_axis:
        link["die_axis_h"] = int(die_axis)
    with open(path, "w") as f:
        json.dump(link, f, indent=2)
    return link


def recto_verso_export(out_dir: str, link: dict, albedo, depth) -> list[str]:
    """This side's albedo and depth, named for its surface, with the link."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in link["object_id"])
    side = link["this_surface"].lower()
    files = []
    if albedo is not None:
        p = os.path.join(out_dir, f"{safe}_{side}_albedo.png")
        cv2.imwrite(p, cv2.cvtColor(np.asarray(albedo), cv2.COLOR_RGB2BGR))
        files.append(p)
    d = np.asarray(depth, np.float64)
    p = os.path.join(out_dir, f"{safe}_{side}_depth.tif")
    cv2.imwrite(p, ((d - d.min()) / (np.ptp(d) + 1e-9) * 255).astype(np.uint8))
    files.append(p)
    pair = "recto_verso" if link["this_surface"] in ("Recto", "Verso") else "two_faces"
    p = os.path.join(out_dir, f"{safe}_{pair}_metadata.json")
    with open(p, "w") as f:
        json.dump(link, f, indent=2)
    return files + [p]


# ── inscription and feature zones ────────────────────────────────────────

INSCRIPTION_MODES = ("Book / Manuscript", "Coin / Medal / Wax Seal", "Stone / Carved Relief",
                     "Metal / Engraved Plate", "Auto (all sources)")


ZONES_SIZE = 2500           # px, the long side the zones are found at


def inscription_zones(depth, normals=None, spec=None, albedo=None, *,
                      mode: str = "Book / Manuscript", sens: float = 0.7,
                      max_zones: int = 16) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Zones likely to hold inscription or worked decoration: edges and
    curvature in the depth and normals, specular contrast and dark-on-light
    albedo, weighted for the object type, merged into letters, lines and
    blocks, scored by density.  Image processing, not a learned model.
    Returns (boxes x, y, w, h best first, the 0–255 signal map), both at the
    size given.

    Found at a long side of ZONES_SIZE: the filters' sizes are in pixels, so
    a 100 MP capture saw only noise-scale edges, took minutes, and merged
    everything into one blob (2026-10-17, a tooled binding: no zones)."""
    d_raw = np.asarray(depth)
    dh, dw = d_raw.shape[:2]
    k = min(1.0, ZONES_SIZE / max(dh, dw))
    if k >= 1.0:
        return _zones(d_raw, normals, spec, albedo, mode=mode, sens=sens, max_zones=max_zones)
    sw, sh = max(1, round(dw * k)), max(1, round(dh * k))

    def small(a):
        if a is None:
            return None
        a = np.asarray(a)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        return cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
    boxes, acc = _zones(small(d_raw), small(normals), small(spec), small(albedo), mode=mode,
                        sens=sens, max_zones=max_zones)
    fx, fy = dw / sw, dh / sh
    boxes = [(int(x * fx), int(y * fy), int(round(bw * fx)), int(round(bh * fy)))
             for x, y, bw, bh in boxes]
    return boxes, cv2.resize(acc, (dw, dh), interpolation=cv2.INTER_LINEAR)


def _marks(img, md: int, k: float) -> np.ndarray:
    """0/1 where *img* stands out from its local background by more than
    *k* × its own noise: ink, tooling, gilding or a cut edge do; paper grain,
    leather grain and a textured backdrop do not.  The noise is the median
    absolute fine detail (a robust sigma), the background a wide median."""
    a = cv2.GaussianBlur(np.asarray(img, np.float32), (0, 0), 1.0)
    hp = a - cv2.GaussianBlur(a, (0, 0), 1.5)
    sigma = 1.4826 * float(np.median(np.abs(hp - np.median(hp)))) + 1e-6
    win = min(255, max(15, int(md * 0.03)) | 1)
    # a wide median wants 8 bits: the 0.5-99.5 % range mapped onto them
    lo, hi = float(np.percentile(a, 0.5)), float(np.percentile(a, 99.5))
    step = (hi - lo + 1e-9) / 255.0
    a8 = np.clip((a - lo) / step, 0, 255).astype(np.uint8)
    bg = cv2.medianBlur(a8, win).astype(np.float32) * step + lo
    return (np.abs(a - bg) > k * sigma).astype(np.float32)


def _to8(a) -> np.ndarray:
    """0-255 by the 99.5th percentile (above it, 255)."""
    a = np.asarray(a, np.float32)
    top = float(np.percentile(a, 99.5))
    return np.clip(a * (255.0 / top if top > 0 else 0.0), 0, 255).astype(np.uint8)


def _zones(depth, normals, spec, albedo, *, mode, sens, max_zones):
    d_raw = np.asarray(depth)
    dh, dw = d_raw.shape
    ti = lambda a, b: int(np.interp(sens, [0.1, 1.0], [a, b]))  # noqa: E731
    t_lo, t_hi, t_grad, t_lap = ti(40, 8), ti(120, 25), ti(30, 6), ti(40, 8)
    t_nz, t_spec, t_alb = ti(30, 8), ti(60, 15), ti(20, 5)
    min_area_f = np.interp(sens, [0.1, 1.0], [0.003, 0.0003])
    accum = np.zeros((dh, dw), np.float32)
    d8 = cv2.normalize(d_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    d8b = cv2.bilateralFilter(d8, 7, 25, 25)
    if mode != "Book / Manuscript":
        ec = cv2.Canny(d8b, t_lo, t_hi).astype(np.float32) / 255.0
        accum += ec * {"Coin / Medal / Wax Seal": 2.0, "Stone / Carved Relief": 2.5,
                       "Metal / Engraved Plate": 2.0}.get(mode, 1.2)
        gx = cv2.Sobel(d_raw.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d_raw.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        # Scaled by the 99.5th percentile, not the maximum: one strong edge (a page's
        # edge, or the step where a depth worked out on the object alone meets its
        # flat surround) otherwise sets the scale and every mark falls under it.
        gm = _to8(np.sqrt(gx ** 2 + gy ** 2))
        _, gt = cv2.threshold(gm, t_grad, 255, cv2.THRESH_BINARY)
        accum += gt.astype(np.float32) / 255.0 * 1.5
        lap8 = _to8(np.abs(cv2.Laplacian(d8b, cv2.CV_32F)))
        _, lt = cv2.threshold(lap8, t_lap, 255, cv2.THRESH_BINARY)
        accum += lt.astype(np.float32) / 255.0 * 1.2
    k_noise = float(np.interp(sens, [0.1, 1.0], [8.0, 3.5]))     # × the noise: a mark
    md0 = min(dh, dw)
    if normals is not None:
        # the surface turning: tooling, incision and relief, against the grain
        n_raw = np.asarray(normals, np.float32)
        accum += _marks(n_raw[..., 0], md0, k_noise) * 1.6 + _marks(n_raw[..., 1], md0, k_noise) * 1.6
    if spec is not None:
        sp8 = cv2.normalize(np.asarray(spec, np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        sp8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(sp8)
        _, st = cv2.threshold(sp8, t_spec, 255, cv2.THRESH_BINARY)
        sgm = cv2.normalize(np.sqrt(cv2.Sobel(sp8, cv2.CV_32F, 1, 0, ksize=3) ** 2
                                    + cv2.Sobel(sp8, cv2.CV_32F, 0, 1, ksize=3) ** 2), None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)
        _, sgt = cv2.threshold(sgm, t_spec // 2, 255, cv2.THRESH_BINARY)
        ws = {"Book / Manuscript": 2.5, "Metal / Engraved Plate": 2.0}.get(mode, 1.5)
        accum += st.astype(np.float32) / 255.0 * ws + sgt.astype(np.float32) / 255.0 * ws * 0.8
    if albedo is not None:
        ag = cv2.cvtColor(np.asarray(albedo).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # ink, gilding, paint: lighter or darker than the surface around it
        # by more than its grain (HELIO 2's CLAHE and adaptive threshold
        # marked half of any grainy surface, writing or not)
        wa = {"Book / Manuscript": 3.0}.get(mode, 1.0)
        accum += _marks(ag, md0, k_noise) * wa
    if accum.max() > 0:
        accum = (accum / accum.max() * 255).astype(np.uint8)
    else:
        accum = accum.astype(np.uint8)
    # A mark is a pixel in the signal's own upper tail, not above a fixed
    # level: where every source adds a little (paper grain, a backdrop) a
    # fixed 15-60 was passed by nearly the whole frame, and the zones merged
    # into one the size of the image, drawn on its border (2026-10-29).
    pct = max(ti(60, 15), int(np.percentile(accum, np.interp(sens, [0.1, 1.0], [95.0, 80.0]))))
    md = min(dh, dw)
    # Where the marks are dense, not each mark: single pixels of grain or of
    # a textured backdrop pass any fixed cut, and on a binding they covered
    # the whole frame.  The local density is cut at its own upper quantile,
    # wider with the sensitivity, and never below the classic's level.
    dens = cv2.GaussianBlur((accum > pct).astype(np.float32), (0, 0), max(2.0, md * 0.008))
    keep = float(np.interp(sens, [0.1, 1.0], [8.0, 35.0]))           # % of the frame at most
    cut = max(float(np.percentile(dens, 100.0 - keep)), 0.15)
    binary = cv2.morphologyEx(((dens >= cut) * 255).astype(np.uint8), cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    odd = lambda v: v if v % 2 else v + 1  # noqa: E731
    k_s = max(3, int(md * 0.008))
    k_l = max(7, int(md * np.interp(sens, [0.1, 1.0], [0.015, 0.030])))
    k_p = max(15, int(md * np.interp(sens, [0.1, 1.0], [0.035, 0.060])))
    m1 = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((odd(k_s), odd(k_s)), np.uint8), iterations=2)
    m2 = cv2.dilate(m1, np.ones((odd(k_l), odd(k_l)), np.uint8), iterations=2)
    m3 = cv2.dilate(m2, np.ones((odd(k_p), odd(k_p)), np.uint8), iterations=1)
    lo, hi = dh * dw * min_area_f, dh * dw * 0.80

    def too_big(c):
        # a zone whose box takes in most of the frame says nothing about
        # where to look, however thin the region inside it
        bx, by, bw, bh = cv2.boundingRect(c)
        return cv2.contourArea(c) >= hi or bw * bh >= dh * dw * 0.7
    # Blocks, else lines, else letters: on an object worked all over (a tooled
    # binding, both pages of an opening) the blocks merge into one region
    # larger than any zone, and the finer grouping is the one that says where
    # to look.  A level is used only when nothing in it merged that far: the
    # first level with any zone at all kept the specks beside the rejected
    # merge and lost the text block itself (2026-10-29).
    cnts = []
    for m in (m3, m2, m1):
        found, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in found if lo < cv2.contourArea(c) and not too_big(c)]
        if cnts and not any(too_big(c) for c in found):
            break

    def score(c):
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 4 or bh < 4:
            return 0
        roi = accum[by:by + bh, bx:bx + bw]
        dens = float(np.count_nonzero(roi > pct)) / max(1, bw * bh)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        return dens * cv2.contourArea(c) * (1.3 if 1.2 < aspect < 12 else 1.0)
    rects = [tuple(int(v) for v in cv2.boundingRect(c)) for c in sorted(cnts, key=score, reverse=True)]

    def inside(r, o):
        return o is not r and o[0] <= r[0] and o[1] <= r[1] and r[0] + r[2] <= o[0] + o[2] \
            and r[1] + r[3] <= o[1] + o[3]
    # a hole's speck inside a zone is already in that zone
    rects = [r for r in rects if not any(inside(r, o) for o in rects)]
    return rects[:max_zones], accum


ZONE_PALETTE = ((255, 220, 0), (0, 255, 200), (255, 80, 80), (80, 200, 255), (200, 80, 255),
                (0, 255, 80), (255, 140, 0), (80, 120, 255), (255, 200, 80), (80, 255, 140),
                (255, 80, 180), (180, 255, 80))


def draw_zones(render, boxes):
    """The boxes and their Z numbers drawn on *render*, sized to it: a fixed 2 px
    line vanished when a 9000 px capture was shown in a 600 px preview."""
    out = np.ascontiguousarray(np.asarray(render).copy())
    u = max(1.0, max(out.shape[:2]) / 1000.0)             # 1 at 1000 px
    th = max(2, int(round(2 * u)))
    fs = 0.55 * u
    for i, (bx, by, bw, bh) in enumerate(boxes):
        col = ZONE_PALETTE[i % len(ZONE_PALETTE)]
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), col, th)
        tag = f"Z{i + 1}"
        (tw, tht), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        pad = int(4 * u)
        ty = by if by - tht - 2 * pad >= 0 else by + tht + 2 * pad   # inside the box at the top edge
        cv2.rectangle(out, (bx, ty - tht - 2 * pad), (bx + tw + 2 * pad, ty), (0, 0, 0), -1)
        cv2.putText(out, tag, (bx + pad, ty - pad), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
    return out
