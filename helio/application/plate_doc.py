"""The publication plate (REP-010): one A3 page that presents a capture.

A figure plate in the manner of a conservation or archaeology report: the
object, how it was lit, what the surface shows under the views that read it
best, its height and a section through it, and how far the result can be
trusted, each panel lettered and captioned with what it shows and how it
was made.  A key-facts block heads the page and a methods paragraph closes
it, so the page stands on its own when it is printed, pinned up or sent.

Written as PDF (text stays text) and as a 300 dpi PNG.  GUI-free: the views
come in through ``draw(engine, overrides)``, which renders a view at
preview size with some of its settings changed (the light, mostly), and an
optional ``relief`` picture from the 3D view.

Units are stated as they are: heights are relative unless a height scale
was measured, and a missing scale is said, not guessed.
"""
from __future__ import annotations

import math
import os
import time
import textwrap
from dataclasses import dataclass, field

import cv2
import numpy as np

A3 = (11.69, 16.54)                       # inches, portrait
INK, INK2, INK3 = "#1E1F22", "#4D5057", "#7A7D84"
RULE = "#D6D2CB"
ACCENT = "#B9781A"
PAPER = "#FFFFFF"
# CIEDE2000.  As mathtext in the page's own face: Helvetica Neue's subscript
# digits rasterise blank at 9 pt and below.
DE00 = "ΔE$_{00}$"


@dataclass
class Facts:
    """What the plate says about the capture."""
    title: str = "Untitled capture"
    subtitle: str = ""
    frames: int = 0
    lights: np.ndarray | None = None       # (N, 3) +Y up
    light_note: str = ""                    # "chrome sphere", "two balls", "file: x.lp"
    mm_per_px: float | None = None
    z_mm_per_unit: float | None = None
    normal_engine: str = ""
    depth_engine: str = ""
    photo_tone: str = ""
    solve_seconds: float = 0.0
    drift_px: float | None = None
    flash_spread: float | None = None      # CAL-003: strongest/weakest flash, evened by a grey card
    colour: dict | None = None             # colour_check summary: mean, max, worst
    creator: str = ""
    institution: str = ""
    licence: str = ""
    helio: str = "Helio"
    notes: list = field(default_factory=list)


def _font_family() -> str:
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for want in ("Helvetica Neue", "Helvetica", "Arial", "Source Sans 3", "DejaVu Sans"):
        if want in names:
            return want
    return "DejaVu Sans"


def _ang(L) -> tuple[np.ndarray, np.ndarray]:
    L = np.asarray(L, np.float64)
    n = np.maximum(np.linalg.norm(L, axis=1), 1e-12)
    az = np.degrees(np.arctan2(L[:, 0], L[:, 1])) % 360.0
    el = np.degrees(np.arcsin(np.clip(L[:, 2] / n, -1, 1)))
    return az, el


def best_raking(L, normals, el: float = 20.0) -> tuple[float, float]:
    """The captured light direction that shows the most relief, lowered to *el*."""
    if L is None or not len(L):
        return 300.0, el
    from helio.application.viewing import best_raking_light
    try:
        _i, _l, az, _el = best_raking_light(np.asarray(L, np.float64), normals)
        return float(az), el
    except Exception:
        return 300.0, el


def from_above(az: float) -> float:
    """*az* turned, if need be, to come from the top half of the picture.

    Relief lit from below reads inside out to most eyes, cuts as ridges (the
    light-from-above prior: Ramachandran 1988), so a raking light found to
    show the most relief from the south is drawn from the north instead.
    Along the same line, it catches the same edges."""
    az %= 360.0
    return (az + 180.0) % 360.0 if 90.0 < az < 270.0 else az


def compass(az: float) -> str:
    """'north', 'north-east', … for a bearing in degrees."""
    names = ("north", "north-east", "east", "south-east", "south", "south-west", "west",
             "north-west")
    return names[int(((az % 360.0) + 22.5) // 45.0) % 8]


def nice_bar(width_mm: float) -> float:
    """A round scale-bar length near a fifth of the width."""
    target = max(width_mm / 5.0, 1e-6)
    p = 10 ** math.floor(math.log10(target))
    for k in (5, 2, 1):
        if k * p <= target:
            return k * p
    return p


def section_profile(depth, p0, p1, mm_per_px=None, z_mm_per_unit=None, n: int = 400) -> dict:
    """Heights along p0 → p1 (image pixels): distance, height, units and roughness."""
    d = np.asarray(depth, np.float32)
    (x0, y0), (x1, y1) = p0, p1
    t = np.linspace(0, 1, n)
    xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
    z = cv2.remap(d, xs.astype(np.float32).reshape(1, -1), ys.astype(np.float32).reshape(1, -1),
                  cv2.INTER_LINEAR).ravel().astype(np.float64)
    length_px = math.hypot(x1 - x0, y1 - y0)
    dist = t * (length_px * mm_per_px if mm_per_px else length_px)
    if z_mm_per_unit:
        z = (z - z.min()) * z_mm_per_unit
        unit = "mm"
    else:
        z = z - z.min()
        unit = "relative"
    dz = z - np.linspace(z[0], z[-1], n)
    zm = dz - dz.mean()
    return dict(dist=dist, z=z, unit=unit, dist_unit="mm" if mm_per_px else "px",
                Ra=float(np.mean(np.abs(zm))), Rq=float(np.sqrt(np.mean(zm ** 2))),
                Rz=float(np.ptp(dz)))


def quality_notes(facts: Facts, dome_score: dict | None) -> list[str]:
    """The quality line's parts: dome coverage (``sphere.dome_coverage``), frame
    drift, the colour check and any notes."""
    qual = []
    if dome_score:
        qual.append(f"dome coverage {dome_score.get('total', 0):.0f}/100 "
                    f"(grade {dome_score.get('grade', '?')})")
    if facts.drift_px is not None:
        qual.append(f"frame drift {facts.drift_px:.1f} px")
    if facts.colour:
        qual.append(f"colour {DE00} {facts.colour['mean']:.2f} average, worst {facts.colour['max']:.1f} "
                    f"({facts.colour.get('worst', '')})")
    return qual + [n for n in facts.notes if n]


def plate(out_base: str, *, D, facts: Facts, draw, relief=None, section=None,
          dome_score: dict | None = None, dpi: int = 300, progress=None) -> dict:
    """Write ``out_base``.pdf and .png; returns {pdf, png, panels, boxes, margin}.

    *boxes* holds every picture, caption and text block as (kind, name,
    (x0, y0, x1, y1)) in page fractions, so a test can see that none overlap.

    *draw(engine, overrides)* renders a view (uint8 RGB) at preview size;
    *relief* an RGB picture of the 3D relief, or None for a shaded-height
    stand-in; *section* ((x0, y0), (x1, y1)) in full-size pixels, or the
    horizontal centre line.

    Drawn on a plain Figure, not pyplot, with its settings scoped to the
    page: Publish runs it off the main thread, and nothing else the program
    draws should take on the plate's type."""
    import matplotlib
    fam = _font_family()
    rc = {"font.family": fam, "font.size": 8.5, "axes.edgecolor": RULE,
          "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
          "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
          "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold",
          "mathtext.default": "regular"}
    with matplotlib.rc_context(rc):
        return _plate(out_base, D=D, facts=facts, draw=draw, relief=relief, section=section,
                      dome_score=dome_score, dpi=dpi, progress=progress)


def _plate(out_base, *, D, facts, draw, relief, section, dome_score, dpi, progress) -> dict:
    import matplotlib
    from matplotlib import patheffects
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrow, Rectangle
    upd = progress or (lambda *_a: None)
    fig = Figure(figsize=A3, dpi=dpi, facecolor=PAPER)
    FigureCanvasAgg(fig)
    W, H = A3
    M = 0.55 / W                                    # side margin, as a fraction of the width
    panels = []
    boxes = []                                      # (kind, name, (x0, y0, x1, y1)), figure units
    h, w = D.h, D.w
    mm = facts.mm_per_px

    def caption(x, y, wfrac, letter, text):
        """A lettered caption with its top at *y*, wrapped to *wfrac* of the page
        width; returns where its last line ends (figure units)."""
        import textwrap
        chars = max(16, int((wfrac - 0.022) * W * 72 / 3.9))
        t0 = fig.text(x, y, f"({letter})", fontsize=8.0, fontweight="bold", color=INK, va="top")
        t = fig.text(x + 0.02, y, textwrap.fill(text, chars), fontsize=7.6, color=INK2, va="top",
                     ha="left", linespacing=1.25)
        x0_, y0_, x1_, y1_ = extent(t0, t)
        boxes.append(("caption", letter, (x0_, y0_, x1_, y1_)))
        return y0_

    def extent(*artists):
        """The box round *artists*, in figure units."""
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        pts = []
        for a in artists:
            bb = a.get_tightbbox(r) if hasattr(a, "get_tightbbox") else a.get_window_extent(r)
            pts += [inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))]
        pts = np.array(pts)
        return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()),
                float(pts[:, 1].max()))

    def picture(rect, img, letter, text, cmap=None, frame=True, side=None, clim=None,
                cap_w=None):
        """An image panel.  Its caption goes under it, wrapped to *cap_w* (the
        picture's width by default), or with *side* = (x, width) beside it,
        top-aligned.  Returns (axes, where the caption ends)."""
        ax = fig.add_axes(rect)
        boxes.append(("image", letter, (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])))
        ax.imshow(img, cmap=cmap, interpolation="lanczos" if img.ndim == 3 else "antialiased")
        if clim is not None:
            ax.images[0].set_clim(*clim)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(frame)
            sp.set_color(RULE)
            sp.set_linewidth(0.6)
        if side is not None:
            end = caption(side[0], rect[1] + rect[3], side[1], letter, text)
        else:
            end = caption(rect[0], rect[1] - 0.006, cap_w or rect[2], letter, text)
        panels.append(letter)
        return ax, end

    def fit_rect(x, y, bw, bh, aspect):
        """The largest box of *aspect* (w/h, in page inches) inside (x, y, bw, bh):
        flush left and flush top."""
        bw_in, bh_in = bw * W, bh * H
        if bw_in / bh_in > aspect:
            return [x, y, bh_in * aspect / W, bh]
        nh = bw_in / aspect / H
        return [x, y + (bh - nh), bw, nh]

    BAR = 0.032                                  # a colour bar and its numbers

    def caption_height(text, wfrac):
        """How tall caption() would set *text* at *wfrac* wide, without setting it."""
        import textwrap
        chars = max(16, int((wfrac - 0.022) * W * 72 / 3.9))
        t = fig.text(0, 0.5, textwrap.fill(text, chars), fontsize=7.6, va="top", linespacing=1.25)
        x0_, y0_, x1_, y1_ = extent(t)
        t.remove()
        return y1_ - y0_

    def row_plan(img_h, items):
        """Sizes for pictures side by side across the page, *img_h* high at most,
        spread so the outer ones meet the margins; *items* (image, letter,
        caption, kw), where kw may carry ``bar`` (the colour bar's unit) and
        ``bar_ends`` (words for its two ends).  Returns the rects (x, width and
        height set), their footprints, the gap between them and the row's height
        down to the end of its lowest caption."""
        n = len(items)
        slot = (1 - 2 * M - (n - 1) * 0.012) / n
        rects = []
        for img, _l, _t, kw in items:
            a = img.shape[1] / max(img.shape[0], 1)
            rects.append(fit_rect(0, 0, slot - (BAR if kw.get("bar") else 0), img_h, a))
        foot = [r[2] + (BAR if kw.get("bar") else 0) for r, (*_x, kw) in zip(rects, items)]
        gap = (1 - 2 * M - sum(foot)) / max(n - 1, 1)
        x = M
        for r, f in zip(rects, foot):
            r[0] = x
            x += f + gap
        # A caption may run a little into the gap beside its picture, never to
        # the next one; the last stops at the margin.
        cap_w = [f + (min(gap - 0.014, 0.4 * f) if i < n - 1 else 0.0)
                 for i, f in enumerate(foot)]
        height = max(r[3] + 0.006 + caption_height(text, cw)
                     for r, cw, (_i, _l, text, _k) in zip(rects, cap_w, items))
        return dict(rects=rects, cap_w=cap_w, height=height)

    def row(top_y, plan, items):
        """Draw a planned row with its top at *top_y*; returns the axes and where
        the lowest caption ends."""
        axes, ends = [], []
        for (img, letter, text, kw), r, cw in zip(items, plan["rects"], plan["cap_w"]):
            kw = dict(kw)
            unit, ends_ = kw.pop("bar", None), kw.pop("bar_ends", None)
            r = [r[0], top_y - r[3], r[2], r[3]]
            ax, end = picture(r, img, letter, text, cap_w=cw, **kw)
            if unit:
                cax = fig.add_axes([r[0] + r[2] + 0.004, r[1], 0.007, r[3]])
                cb = fig.colorbar(ax.images[0], cax=cax)
                cb.ax.tick_params(labelsize=6, length=2, pad=1.5, colors=INK2)
                cb.outline.set_linewidth(0.4)
                cb.outline.set_edgecolor(RULE)
                if ends_:
                    lo_, hi_ = ax.images[0].get_clim()
                    cb.set_ticks([lo_, hi_])
                    cb.set_ticklabels(list(ends_))
                cb.set_label(unit, fontsize=6, color=INK3, labelpad=2)
                boxes.append(("image", letter + "-bar", extent(cax)))
            axes.append(ax)
            ends.append(end)
        return axes, min(ends)

    aspect = w / max(h, 1)

    # ── header ────────────────────────────────────────────────────────
    upd("The header…", 0.05)
    fig.text(M, 0.972, facts.title, fontsize=22, fontweight="bold", color=INK, va="top")
    sub = facts.subtitle or "Photometric stereo"
    fig.text(M, 0.948, sub, fontsize=10.5, color=INK2, va="top")
    facts_rows = []
    if facts.frames:
        facts_rows.append(("Frames", f"{facts.frames}"))
    if facts.lights is not None and len(facts.lights):
        _az, el = _ang(facts.lights)
        facts_rows.append(("Lights", f"{len(el)}, {el.min():.0f}–{el.max():.0f}° up"
                           + (f" · {facts.light_note}" if facts.light_note else "")))
    facts_rows.append(("Scale", f"{mm:.4g} mm per pixel" if mm else "not measured"))
    if mm:
        facts_rows.append(("Size", f"{w * mm:.1f} × {h * mm:.1f} mm"))
    facts_rows.append(("Heights", f"{facts.z_mm_per_unit * 1000:.4g} µm per pixel of height" if facts.z_mm_per_unit
                       else "relative (no height scale)"))
    if facts.normal_engine:
        facts_rows.append(("Solve", f"{facts.normal_engine} · {facts.depth_engine}"))
    if facts.photo_tone:
        facts_rows.append(("Photos", {"stored": "values as stored (not linearised)",
                                      "srgb": "sRGB decoded to linear light",
                                      "linear": "linear light"}.get(facts.photo_tone,
                                                                    facts.photo_tone)))
    facts_rows.append(("Colour", f"{DE00} {facts.colour['mean']:.2f} average (ColorChecker)"
                       if facts.colour else "not checked against a chart"))
    x_k, x_v, y = 0.60, 0.69, 0.975
    for k, v in facts_rows[:9]:
        fig.text(x_k, y, k.upper(), fontsize=6.6, color=INK3, va="top", fontweight="bold")
        fig.text(x_v, y, v, fontsize=8.0, color=INK, va="top")
        y -= 0.0108
    rule_y = min(0.905, y - 0.004)
    boxes.append(("text", "header", (M, rule_y, 1 - M, 0.975)))
    fig.add_artist(Line2D([M, 1 - M], [rule_y, rule_y], color=ACCENT, linewidth=1.2,
                              transform=fig.transFigure))

    # ── row A: the object, how it was lit, its normals ────────────────
    upd("The object and its light…", 0.15)
    top = rule_y - 0.014
    bottom_a = 0.60
    a_rect = fit_rect(M, bottom_a + 0.032, 0.56, top - bottom_a - 0.032, aspect)
    albedo = np.asarray(D.albedo_rgb_full)
    if max(albedo.shape[:2]) > 2400:
        k = 2400 / max(albedo.shape[:2])
        albedo = cv2.resize(albedo, (int(albedo.shape[1] * k), int(albedo.shape[0] * k)),
                            interpolation=cv2.INTER_AREA)
    sx = albedo.shape[1] / w
    sec = section or ((w * 0.08, h * 0.5), (w * 0.92, h * 0.5))
    ax, _end = picture(a_rect, albedo, "a", "The colour of the surface, without the light's "
                       "shading (albedo). A–A′ marks the section in (j).",
                       cap_w=max(a_rect[2], 0.25))
    (x0, y0), (x1, y1) = sec
    ax.plot([x0 * sx, x1 * sx], [y0 * sx, y1 * sx], color=ACCENT, linewidth=1.4)
    for (xx, yy), lab in (((x0, y0), "A"), ((x1, y1), "A′")):
        ax.text(xx * sx, yy * sx - albedo.shape[0] * 0.012, lab, color=ACCENT, fontsize=8.5,
                fontweight="bold", ha="center", va="bottom",
                path_effects=[patheffects.withStroke(linewidth=2.2, foreground="white")])
    ih, iw = albedo.shape[:2]
    if mm:
        bar = nice_bar(w * mm)
        bpx = bar / mm * sx
        bx, by = iw * 0.05, ih * 0.95
        ax.add_patch(Rectangle((bx - iw * 0.012, by - ih * 0.035), bpx + iw * 0.024, ih * 0.05,
                               facecolor="white", alpha=0.85, edgecolor="none"))
        ax.plot([bx, bx + bpx], [by, by], color=INK, linewidth=2.2, solid_capstyle="butt")
        ax.text(bx + bpx / 2, by - ih * 0.008, f"{bar:g} mm", ha="center", va="bottom", fontsize=7.5,
                color=INK)
    ax.add_patch(FancyArrow(iw * 0.06, ih * 0.13, 0, -ih * 0.07, width=iw * 0.006,
                            head_width=iw * 0.025, head_length=ih * 0.025, color="white",
                            ec=INK, linewidth=0.6, length_includes_head=True))
    ax.text(iw * 0.06, ih * 0.155, "N", ha="center", va="top", fontsize=8, fontweight="bold",
            color="white", path_effects=[patheffects.withStroke(linewidth=2, foreground=INK)])

    # (b) and (c) share what the object leaves of the row: side by side beside
    # a tall object, one above the other beside a wide one.
    rx0 = a_rect[0] + a_rect[2] + 0.035
    rw = 1 - M - rx0
    h_a = top - bottom_a
    lab = 0.24                                     # inches kept round the dome for N E S W
    if rw * W >= 5.5:
        b_box = (rx0, bottom_a, rw * 0.42, h_a)
        c_box = (rx0 + rw * 0.42 + 0.03, bottom_a, rw * 0.58 - 0.03, h_a)
        b_under = True
    else:
        half = (h_a - 0.016) / 2
        b_box = (rx0, bottom_a + half + 0.016, rw, half)
        c_box = (rx0, bottom_a, rw, half)
        b_under = False

    # (b) the light dome
    bx, by, bw_, bh_ = b_box
    if b_under:
        d_in = min(bw_ * W - 2 * lab, bh_ * H - 2 * lab - 0.6, 3.0)
        d_x = bx + (bw_ - d_in / W) / 2
    else:
        d_in = min(bh_ * H - 2 * lab, bw_ * W * 0.5 - 2 * lab, 3.0)
        d_x = bx + lab / W
    d_y = by + bh_ - lab / H - d_in / H
    axb = fig.add_axes([d_x, d_y, d_in / W, d_in / H], projection="polar")
    axb.set_theta_zero_location("N")
    axb.set_theta_direction(-1)
    axb.set_rlim(0, 90)
    axb.set_rticks([15, 30, 45, 60, 75])
    axb.set_yticklabels(["75°", "60°", "45°", "30°", "15°"], fontsize=6, color=INK3)
    axb.set_xticks(np.radians([0, 90, 180, 270]))
    axb.set_xticklabels(["N", "E", "S", "W"], fontsize=7.5, color=INK2)
    axb.grid(color=RULE, linewidth=0.6)
    axb.spines["polar"].set_color(RULE)
    if facts.lights is not None and len(facts.lights):
        az, el = _ang(facts.lights)
        axb.scatter(np.radians(az), 90 - el, s=26, c=el, cmap="cividis", vmin=0, vmax=90,
                    edgecolors=INK, linewidths=0.5, zorder=3)
        if len(az) <= 16 and d_in >= 1.6:                   # numbers only while they can be read
            for i, (a_, e_) in enumerate(zip(az, el)):
                axb.text(np.radians(a_), 90 - e_ + 9, str(i + 1), fontsize=5.5, color=INK2,
                         ha="center", va="center")
        ltxt = (f"Where the {len(az)} lights were, seen from above: the rim is the horizon, the "
                f"centre overhead ({el.min():.0f}–{el.max():.0f}° up).")
    else:
        ltxt = "The light positions are not known (estimated from the photographs)."
    boxes.append(("image", "b", extent(axb)))
    if b_under:
        caption(bx, d_y - lab / H - 0.002, bw_, "b", ltxt)
    else:
        s_x = d_x + (d_in + lab) / W + 0.01
        caption(s_x, by + bh_, bx + bw_ - s_x, "b", ltxt)
    panels.append("b")

    # (c) the normals, the words beside them and the colour key under the words
    cx, cy, cw_, ch_ = c_box
    n_rect = fit_rect(cx, cy, cw_ * 0.56, ch_, aspect)
    nrm = np.asarray(D.normals_proxy if getattr(D, "normals_proxy", None) is not None else D.normals_full)
    ncol = np.clip((np.nan_to_num(nrm) * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    s_x = n_rect[0] + n_rect[2] + 0.012
    _axc, c_end = picture(n_rect, ncol, "c", "Which way the surface faces (the normals): red "
                          "towards east, green towards north, blue straight up. The dome is the key: "
                          "each direction in its colour.", side=(s_x, cx + cw_ - s_x))
    n_k = 320
    yy, xx = np.mgrid[-1:1:n_k * 1j, -1:1:n_k * 1j]
    rr = np.sqrt(xx ** 2 + yy ** 2)
    nz = np.sqrt(np.clip(1 - rr ** 2, 0, 1))
    key = np.empty((n_k, n_k, 4))
    key[..., :3] = np.stack([xx, -yy, nz], -1) * 0.5 + 0.5
    key[..., 3] = np.clip((1 - rr) * n_k / 2, 0, 1)        # a one-pixel soft rim
    k_in = min(1.0, (cx + cw_ - s_x) * W - 0.5, (c_end - cy) * H - 0.45)
    if k_in >= 0.45:                               # room under the words
        k_rect = [s_x + 0.02 + 0.16 / W, c_end - (0.22 + k_in) / H, k_in / W, k_in / H]
    else:                                          # else inside the picture's corner
        k_in = min(0.6, n_rect[2] * W * 0.3)
        k_rect = [n_rect[0] + n_rect[2] - (k_in + 0.06) / W, n_rect[1] + 0.06 / H, k_in / W, k_in / H]
    ka = fig.add_axes(k_rect)
    ka.imshow(np.clip(key, 0, 1), interpolation="antialiased")
    ka.axis("off")
    for t_, (kx, ky) in (("N", (0.5, 1.1)), ("E", (1.1, 0.5)), ("S", (0.5, -0.1)), ("W", (-0.1, 0.5))):
        ka.text(kx, ky, t_, transform=ka.transAxes, ha="center", va="center", fontsize=6.5,
                color=INK3)
    boxes.append(("image", "c-key", extent(ka)))
    panels.append("c-key")

    # ── footer, set from the bottom up: the rows above fill what it leaves ──
    upd("The methods…", 0.3)
    method = (f"Method. {facts.frames or 'Several'} photographs from a fixed camera, each lit from one "
              "direction" + (f" ({facts.light_note})" if facts.light_note else "")
              + (f"; each flash's strength (it varied ×{facts.flash_spread:.1f}) evened against a "
                 "grey card in the shot" if facts.flash_spread else "") + ". Surface "
              "normals and colour by photometric stereo (Woodham 1980)"
              + (f", solved by {facts.normal_engine}" if facts.normal_engine else "")
              + "; height by integrating the normals"
              + (f" ({facts.depth_engine})" if facts.depth_engine else "") + ". "
              + ("Distances use the measured scale. " if mm else "No scale was measured: distances "
                 "are in pixels. ")
              + ("Heights use the measured height scale." if facts.z_mm_per_unit else
                 "Heights are relative: integration recovers shape, not size, without a height "
                 "reference."))
    qual = quality_notes(facts, dome_score)
    width_chars = int((1 - 2 * M) * W * 72 / (7.2 * 0.5)) - 6

    def top_of(t):
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        return float(fig.transFigure.inverted().transform((0, bb.y1))[1])

    credit = " · ".join(p for p in (facts.creator, facts.institution, facts.licence) if p)
    fig.text(M, 0.022, credit, fontsize=7.0, color=INK3, va="bottom")
    fig.text(1 - M, 0.022, f"{facts.helio} · {time.strftime('%d %B %Y')}", fontsize=7.0, color=INK3,
             va="bottom", ha="right")
    y_f = 0.04
    if qual:
        tq = fig.text(M, y_f, textwrap.fill("Quality: " + " · ".join(qual) + ".", width_chars),
                      fontsize=7.2, color=INK2, va="bottom", linespacing=1.3)
        y_f = top_of(tq) + 0.005
    tm = fig.text(M, y_f, textwrap.fill(method, width_chars), fontsize=7.2, color=INK2, va="bottom",
                  linespacing=1.3)
    foot = top_of(tm) + 0.009
    boxes.append(("text", "footer", (M, 0.022, 1 - M, foot)))
    fig.add_artist(Line2D([M, 1 - M], [foot, foot], color=RULE, linewidth=0.8,
                              transform=fig.transFigure))
    floor = foot + 0.012

    # ── rows B to D: gathered, planned to fill what is left, then drawn ──
    upd("Reading the surface…", 0.35)
    rak_az, rak_el = best_raking(facts.lights, D.normals_proxy if getattr(D, "normals_proxy", None)
                                 is not None else D.normals_full)
    rak_az = from_above(rak_az)
    from helio.domain.view_controls import from_angles, light_settings
    views = (
        # Lambert (gain 1), shown through a display gamma as a photograph would be
        ("Standard PBR", dict(light_settings(*from_angles(rak_az, rak_el)), gain=1.0, gamma=2.2),
         f"Raking light from the {compass(rak_az)}, {rak_el:.0f}° up: low relief, tool marks "
         f"and impressions stand out.", "d"),
        ("Exaggerated Shading", dict(light_settings(*from_angles(rak_az, 35.0))),
         "Exaggerated shading: fine and broad relief lit together (Rusinkiewicz, Burns and "
         "DeCarlo 2006).", "e"),
        ("Multi-Scale Relief Model (MSRM)", {},
         "Multi-scale relief, no light: raised light, sunken dark, from fine to broad "
         "(Orengo and Petrie 2018).", "f"),
        ("Surface Curvature (Scratches)", {},
         "Curvature: ridges light, grooves dark, whatever the light. Scratches and "
         "incisions.", "g"),
    )
    items_b = []
    for eng, over, text, letter in views:
        try:
            img = np.asarray(draw(eng, over))
        except Exception as e:                           # a missing view never stops the plate
            img = np.full((max(1, D.ph), max(1, D.pw), 3), 235, np.uint8)
            text += f" (not drawn: {e})"
        items_b.append((img, letter, text, {}))

    upd("Height and quality…", 0.55)
    dep = np.asarray(D.depth_proxy_p if getattr(D, "depth_proxy_p", None) is not None else D.depth_vis_p,
                     np.float32)
    lo_h, hi_h = (float(v) for v in np.nanpercentile(dep, [1, 99]))
    if relief is None:
        from matplotlib.colors import LightSource
        ls = LightSource(azdeg=rak_az, altdeg=30)
        relief = (ls.shade(dep, cmap=matplotlib.colormaps["Greys_r"],
                           vert_exag=max(1.0, 0.25 * dep.shape[1]),
                           blend_mode="soft")[..., :3] * 255).astype(np.uint8)
        rtext = "The relief, shaded from the height alone (the 3D view was not open)."
    else:
        rtext = "The relief in 3D, lit with the photometric normals (HELIO's 3D view)."
    sig = getattr(D, "qc_sigma_proxy", None)
    if sig is None:
        sig = getattr(D, "qc_sigma_full", None)
    from helio.application.heights import height_px
    true_h = height_px(D)
    z_mm = facts.z_mm_per_unit if true_h is not None else None    # mm only from true heights
    if z_mm:
        hm = cv2.resize(true_h, (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_AREA)
        lo_m, hi_m = (float(v) for v in np.nanpercentile(hm, [1, 99]))
        dep_show = (hm - lo_m) * facts.z_mm_per_unit
        clim_h, bar_h = (0.0, max((hi_m - lo_m) * facts.z_mm_per_unit, 1e-9)), dict(bar="mm")
    else:
        dep_show = dep
        clim_h, bar_h = (lo_h, hi_h if hi_h > lo_h else lo_h + 1e-6), dict(bar="relative",
                                                                            bar_ends=("low", "high"))
    items_c = [(dep_show, "h", "Height, integrated from the normals: dark low, light high"
                + (" (mm above the lowest point shown)." if z_mm else
                   " (relative: no height scale)."),
                dict(cmap="cividis", clim=clim_h, **bar_h)),
               (np.asarray(relief), "i", rtext, {})]
    if sig is not None:
        sg = np.asarray(sig, np.float32)
        items_c.append((sg, "k", f"How sure the solve is: each pixel's normal uncertainty, median "
                        f"{float(np.nanmedian(sg)):.1f}° (brighter is less sure).",
                        dict(cmap="magma", clim=(0, float(np.nanpercentile(sg, 98)) or 1.0),
                             bar="degrees")))
    else:
        items_c.append((np.asarray(draw("Light Coverage Map", {})), "k", "How well each pixel was "
                        "lit: light coverage from the whole stack of photographs.", {}))

    prof_src = true_h if z_mm else D.depth_vis_p                     # true heights for mm
    prof = section_profile(np.asarray(prof_src, np.float32), sec[0], sec[1], mm, z_mm)
    u = " mm" if prof["unit"] == "mm" else ""
    j_text = (f"Section A–A′: the height along the line in (a). Roughness about the straight line "
              f"between its ends: Ra {prof['Ra']:.3g}{u} (mean), Rq {prof['Rq']:.3g}{u} "
              f"(root-mean-square), Rz {prof['Rz']:.3g}{u} (peak to valley)"
              + ("." if u else ", in the height's relative units."))

    # The plan: the pictures of rows B and C as tall as the page allows with
    # the section about 0.07 of the page; pictures held by their width (a wide
    # object) cannot grow, so then the section takes up to 0.12 and any room
    # still over is shared as space between the rows.
    top_b = bottom_a - 0.014
    room = top_b - floor
    fixed = 0.018 + 0.026 + 0.026 + caption_height(j_text, 1 - 2 * M)   # gaps, A labels, axis, words
    img_h = max(0.08, min(0.17, (room - fixed - 0.07 - 2 * 0.05) / 2))
    for _ in range(8):
        plan_b, plan_c = row_plan(img_h, items_b), row_plan(img_h, items_c)
        sec_h = room - plan_b["height"] - plan_c["height"] - fixed
        tall = any(abs(r[3] - img_h) < 1e-9 for r in plan_b["rects"] + plan_c["rects"])
        if sec_h < 0.045 and img_h > 0.06:
            img_h -= (0.045 - sec_h) / 2 + 0.002
        elif sec_h > 0.075 and tall and img_h < 0.2:
            img_h = min(0.2, img_h + (sec_h - 0.07) / 2)    # height-bound pictures can grow
        else:
            break
    spare = max(0.0, sec_h - 0.12) / 3
    sec_h = max(0.045, min(0.12, sec_h))

    _axes, end_b = row(top_b - spare, plan_b, items_b)
    _axes, end_c = row(end_b - 0.018 - spare, plan_c, items_c)

    # ── row D: the section ────────────────────────────────────────────
    upd("The section…", 0.8)
    rowD_top = end_c - 0.026 - spare
    axp = fig.add_axes([M + 0.045, rowD_top - sec_h, 1 - 2 * M - 0.06, sec_h])
    axp.fill_between(prof["dist"], prof["z"], prof["z"].min(), color="#E9E2D2", linewidth=0)
    axp.plot(prof["dist"], prof["z"], color=ACCENT, linewidth=1.1)
    axp.set_xlim(prof["dist"][0], prof["dist"][-1])
    axp.set_xlabel(f"Distance along A–A′ ({prof['dist_unit']})", fontsize=7.5)
    axp.set_ylabel("Height (mm)" if prof["unit"] == "mm" else "Height (relative)", fontsize=7.5)
    axp.tick_params(labelsize=6.5)
    for side_ in ("top", "right"):
        axp.spines[side_].set_visible(False)
    axp.grid(color="#EEEAE3", linewidth=0.5)
    axp.text(0.0, 1.04, "A", transform=axp.transAxes, fontsize=8.5, fontweight="bold",
             color=ACCENT, va="bottom", ha="left")
    axp.text(1.0, 1.04, "A′", transform=axp.transAxes, fontsize=8.5, fontweight="bold",
             color=ACCENT, va="bottom", ha="right")
    boxes.append(("image", "j", extent(axp)))
    caption(M, rowD_top - sec_h - 0.026, 1 - 2 * M, "j", j_text)
    panels.append("j")

    upd("Writing the files…", 0.95)
    os.makedirs(os.path.dirname(os.path.abspath(out_base)) or ".", exist_ok=True)
    pdf, png = out_base + ".pdf", out_base + ".png"
    fig.savefig(pdf, dpi=dpi, facecolor=PAPER)
    fig.savefig(png, dpi=dpi, facecolor=PAPER)
    return dict(pdf=pdf, png=png, panels=panels, boxes=boxes, margin=M)
