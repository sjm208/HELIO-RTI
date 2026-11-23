"""Composed figures — the images that go in a paper, not another loose PNG.

HELIO writes plenty of pictures. What it has not written is a *figure*: several
renders arranged, captioned and scaled so the result can be dropped into a
publication without an hour in a page layout program first. These three are the
ones that were being assembled by hand.

Every function here returns a uint8 RGB array and touches no UI, so they are
testable without a window and reusable from headless mode.
"""

from __future__ import annotations

import numpy as np

__all__ = ["plate_figure", "rose_diagram", "anaglyph"]


def _as_rgb8(a):
    a = np.asarray(a)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    a = a[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if float(np.nanmax(a)) <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
    return a


def _font(px):
    from PIL import ImageFont
    for nm in ("/System/Library/Fonts/Supplemental/Arial.ttf",
               "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(nm, px)
        except Exception:
            continue
    return ImageFont.load_default()


def plate_figure(panels, cols=3, tile_px=900, mm_per_px=None,
                 title="", background=(255, 255, 255), caption_px=26):
    """Several renders as one captioned plate.

    ``panels`` is a sequence of ``(label, image)``. Tiles are letterboxed rather
    than cropped or stretched: a plate exists to be compared across, and a tile
    that has been cropped to fit is no longer the same field of view as the one
    beside it.

    A scale bar is drawn on the first tile only when ``mm_per_px`` is given —
    once per plate, because repeating it on every tile implies the tiles might
    be at different scales, and they are not.
    """
    from PIL import Image, ImageDraw

    if not panels:
        raise ValueError("no panels to compose")
    cols = max(1, int(cols))
    rows = (len(panels) + cols - 1) // cols

    tiles = []
    for _lbl, img in panels:
        a = _as_rgb8(img)
        h, w = a.shape[:2]
        sc = min(tile_px / float(w), tile_px / float(h))
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        import cv2
        t = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        pad = np.full((tile_px, tile_px, 3), background, np.uint8)
        y0, x0 = (tile_px - nh) // 2, (tile_px - nw) // 2
        pad[y0:y0 + nh, x0:x0 + nw] = t
        tiles.append((pad, sc, (x0, y0, nw, nh)))

    cell_h = tile_px + caption_px + 10
    title_h = (caption_px + 22) if title else 0
    W = cols * tile_px + (cols + 1) * 12
    H = title_h + rows * cell_h + 12
    sheet = Image.new("RGB", (W, H), tuple(background))
    d = ImageDraw.Draw(sheet)
    f_cap = _font(caption_px)
    f_ttl = _font(caption_px + 8)

    if title:
        d.text((14, 10), title, fill=(0, 0, 0), font=f_ttl)

    for i, ((tile, sc, geom), (lbl, _img)) in enumerate(zip(tiles, panels)):
        r, c = divmod(i, cols)
        x = 12 + c * (tile_px + 12)
        y = title_h + 6 + r * cell_h
        sheet.paste(Image.fromarray(tile), (x, y))

        if i == 0 and mm_per_px:
            # Sized to a round number of millimetres nearest a fifth of the
            # tile, so the bar reads as a measurement and not as a decoration.
            x0, y0, nw, nh = geom
            target_mm = (nw * (mm_per_px / sc)) / 5.0
            nice = min([1, 2, 5, 10, 20, 50, 100, 200, 500],
                       key=lambda v: abs(v - target_mm))
            bar_px = int(nice / (mm_per_px / sc))
            if 20 < bar_px < nw:
                bx = x + x0 + 18
                by = y + y0 + nh - 34
                d.rectangle([bx, by, bx + bar_px, by + 9], fill=(0, 0, 0))
                d.rectangle([bx, by, bx + bar_px // 2, by + 9], fill=(255, 255, 255))
                d.rectangle([bx, by, bx + bar_px, by + 9], outline=(0, 0, 0), width=2)
                d.text((bx, by - caption_px - 4), f"{nice} mm",
                       fill=(0, 0, 0), font=f_cap)

        d.text((x + 4, y + tile_px + 6), str(lbl), fill=(0, 0, 0), font=f_cap)

    return np.asarray(sheet)


def orientation_histogram(normals, bins=36, mask=None):
    """(bin edges in radians, weight fraction per bin) of the surface's tilt
    bearings: 0 = north (the top of the image), clockwise; normals +Y up.
    Each pixel votes with hypot(nx, ny), so flat noise counts for little.

    HELIO 2 took atan2(ny, nx), an angle from east, and drew it on north-up
    clockwise axes, so a surface tilted east was drawn at north (KD-037)."""
    n = np.asarray(normals, dtype=np.float32)
    if n.ndim != 3 or n.shape[2] < 3:
        raise ValueError("normals must be (h, w, 3)")
    nx, ny = n[..., 0], n[..., 1]
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        nx, ny = nx[m], ny[m]
    nx, ny = nx.ravel(), ny.ravel()
    w = np.hypot(nx, ny)
    keep = w > 1e-4
    if not np.any(keep):
        raise ValueError("surface is flat — no orientation to report")
    ang = np.arctan2(nx[keep], ny[keep]) % (2 * np.pi)
    edges = np.linspace(0, 2 * np.pi, bins + 1)
    hist, _ = np.histogram(ang, bins=edges, weights=w[keep])
    return edges, hist / (float(hist.sum()) or 1.0)


def rose_diagram(normals, bins=36, mask=None, title="Surface orientation",
                 size_px=1000):
    """Polar histogram of which way the surface faces.

    Weighted by the horizontal component of each normal, which matters more
    than it looks. A flat pixel has a normal of about (0, 0, 1); its azimuth is
    ``atan2`` of two numbers that are almost zero, so it is pure noise. Counting
    every pixel equally therefore buries a real tool direction under a fog
    contributed by the flattest, least informative parts of the surface.
    Weighting by ``hypot(nx, ny)`` lets each pixel vote in proportion to how
    much of a direction it actually has.

    Returns a uint8 RGB array of the plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges, frac = orientation_histogram(normals, bins, mask)

    dpi = 100.0
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    centres = (edges[:-1] + edges[1:]) / 2
    ax.bar(centres, frac, width=(2 * np.pi / bins) * 0.92,
           color="#3b6ea5", edgecolor="#1d3f63", linewidth=0.6, align="center")
    ax.set_yticklabels([])
    ax.set_title(title, pad=18, fontsize=15)

    # The dominant direction, stated rather than left to be eyeballed off the
    # petals — and its opposite marked, because a scratch has no head or tail
    # and 40° and 220° are the same stroke.
    k = int(np.argmax(frac))
    dom = np.degrees(centres[k])
    ax.annotate(f"dominant {dom:.0f}°  ({frac[k] * 100:.0f}% of weight)",
                xy=(0.5, -0.09), xycoords="axes fraction",
                ha="center", fontsize=11, color="#333333")
    for a_deg in (dom, (dom + 180) % 360):
        ax.plot([np.radians(a_deg)] * 2, [0, float(frac.max())],
                color="#c2410c", lw=1.4, ls="--", alpha=0.9)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def anaglyph(depth, colour, separation_px=14, invert=False):
    """Red–cyan stereo pair from the depth map.

    Two views are synthesised by shifting each pixel horizontally in proportion
    to its height, then the left eye's red channel is combined with the right
    eye's green and blue.

    Holes are filled forward rather than left black: a disocclusion is a place
    the synthesis has no information about, and a black gash reads as a crack in
    the object — inventing surface detail there would be worse, but so is
    inventing damage.
    """
    import cv2

    d = np.asarray(depth, np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
    dn = (d - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(d)
    if invert:
        dn = 1.0 - dn

    c = _as_rgb8(colour)
    if c.shape[:2] != dn.shape[:2]:
        c = cv2.resize(c, (dn.shape[1], dn.shape[0]), interpolation=cv2.INTER_AREA)

    h, w = dn.shape
    sep = float(separation_px)
    xs = np.arange(w, dtype=np.float32)[None, :].repeat(h, 0)
    ys = np.arange(h, dtype=np.float32)[:, None].repeat(w, 1)

    def _warp(sign):
        mx = xs + sign * (dn - 0.5) * sep
        out = cv2.remap(c, mx, ys, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        return out

    left = _warp(+1.0)
    right = _warp(-1.0)
    out = np.empty_like(c)
    out[..., 0] = left[..., 0]
    out[..., 1] = right[..., 1]
    out[..., 2] = right[..., 2]
    return out
