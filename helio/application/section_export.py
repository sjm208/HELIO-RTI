"""Section drawings and series, as files: SVG for a report, DXF for CAD, CSV for a spreadsheet.

``profiles`` are dicts as ``ReliefView.section_profile`` gives them (``dist``, ``z``, ``unit``, ``length_unit``,
``length``, ``a_px``, ``b_px``, ``ra``/``rq``/``rz``, ``features``), one for a single section, several for a series
(each with its ``offset``).  When the length is in mm the drawing is to a stated round scale; in pixels it is
drawn the same way and says it is not to scale.
"""
from __future__ import annotations

import datetime
import html

import numpy as np

from helio.core.sections import nice_scale, scale_label, stack_gap

SOFTWARE = "Helio"


def _mm_lengths(profiles) -> bool:
    return all(p.get("length_unit") == "mm" for p in profiles)


def _z_in_length_units(p) -> bool:
    """Height and distance in the same unit, so the drawing can be to one scale in both directions."""
    return p["unit"].split()[0] == p["length_unit"]


# ── CSV ──────────────────────────────────────────────────────────────────────

def series_csv(path: str, profiles: list[dict], offsets: list[float] | None = None) -> str:
    """One distance column and one height column per section (the profiles share their distances)."""
    import csv
    p0 = profiles[0]
    zu, lu = p0["unit"].split()[0], p0["length_unit"]
    offs = offsets if offsets is not None else [p.get("offset", 0.0) for p in profiles]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# {len(profiles)} parallel sections, {p0['length']:.6g} {lu} long; "
                    f"offsets in {lu}, positive to the right looking A to B; height in {p0['unit']}"])
        w.writerow([f"distance_{lu}"] + [f"height_{zu}_offset_{o:+.4g}{lu}" for o in offs])
        d = np.asarray(p0["dist"], float)
        zs = [np.interp(d, np.asarray(p["dist"], float), np.asarray(p["z"], float)) for p in profiles]
        for i in range(len(d)):
            w.writerow([f"{d[i]:.6g}"] + [f"{z[i]:.6g}" for z in zs])
    return path


# ── SVG ──────────────────────────────────────────────────────────────────────

GREYS = ["#111111", "#2a6fbb", "#b5651d", "#2e8b57", "#8b2e8b", "#a09000", "#007a7a", "#a02020"]


def _ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(1, n)
    mag = 10.0 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 5, 10) if m * mag >= raw)
    start = np.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-6:
        out.append(round(float(v), 10))
        v += step
    return out


def section_svg(path: str, profiles: list[dict], *, title: str = "", meta: dict | None = None,
                v_exag: float = 1.0, stacked: bool = True, page_width_mm: float = 200.0) -> dict:
    """A section drawing: the profile (or a series) to a stated scale, axes in mm, scale bars, the grooves
    and ridges marked, and a title block.  One SVG unit is one millimetre of paper.  Returns what was drawn
    (``scale`` as a label, ``v_exag``, the page size) so a caller can say it."""
    meta = meta or {}
    n = len(profiles)
    p0 = profiles[0]
    true_mm = _mm_lengths(profiles) and all(_z_in_length_units(p) for p in profiles)
    length = float(p0["length"])
    lu, zu = p0["length_unit"], p0["unit"].split()[0]
    margin_l, margin_r, margin_t = 22.0, 12.0, 14.0
    plot_w = page_width_mm - margin_l - margin_r
    s = nice_scale(length, plot_w) if true_mm else plot_w / max(length, 1e-12)   # paper mm per real mm (or per px)
    draw_w = length * s
    gap_real = stack_gap(profiles) if (n > 1 and stacked) else 0.0
    zmin = min(float(np.min(p["z"])) for p in profiles)
    zmax = max(float(np.max(p["z"])) for p in profiles)
    z_span = (zmax - zmin) + gap_real * (n - 1)
    sv = s * float(v_exag)                                 # paper mm per unit of height
    plot_h = max(z_span * sv, 6.0)
    block_h = 34.0
    page_h = margin_t + plot_h + 20.0 + block_h + 8.0
    x0 = margin_l + (plot_w - draw_w) / 2.0 if draw_w < plot_w else margin_l
    ybase = margin_t + plot_h                              # paper y of the lowest height

    def X(dv):
        return x0 + dv * s

    def Y(zv, i=0):
        return ybase - ((zv - zmin) + i * gap_real) * sv

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_width_mm:g}mm" height="{page_h:.1f}mm" '
           f'viewBox="0 0 {page_width_mm:g} {page_h:.2f}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect x="0" y="0" width="{page_width_mm:g}" height="{page_h:.2f}" fill="#ffffff"/>']
    # frame, grid and axes
    out.append(f'<rect x="{x0:.2f}" y="{margin_t:.2f}" width="{draw_w:.2f}" height="{plot_h:.2f}" '
               f'fill="none" stroke="#999" stroke-width="0.2"/>')
    for v in _ticks(0.0, length, 8):
        x = X(v)
        out.append(f'<line x1="{x:.2f}" y1="{margin_t:.2f}" x2="{x:.2f}" y2="{ybase:.2f}" stroke="#ddd" stroke-width="0.12"/>')
        out.append(f'<text x="{x:.2f}" y="{ybase + 5:.2f}" font-size="3" text-anchor="middle" fill="#444">{v:g}</text>')
    out.append(f'<text x="{x0 + draw_w:.2f}" y="{ybase + 9:.2f}" font-size="3" text-anchor="end" fill="#444">distance ({lu})</text>')
    z_hi = zmin + z_span
    if n == 1 or not stacked:
        for v in _ticks(zmin, z_hi, 5):
            y = Y(v)
            out.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x0 + draw_w:.2f}" y2="{y:.2f}" stroke="#ddd" stroke-width="0.12"/>')
            out.append(f'<text x="{x0 - 1.5:.2f}" y="{y + 1:.2f}" font-size="3" text-anchor="end" fill="#444">{v:g}</text>')
        out.append(f'<text x="{x0 - 1.5:.2f}" y="{margin_t - 3:.2f}" font-size="3" text-anchor="end" fill="#444">height ({zu})</text>')
    # the profiles
    for i, p in enumerate(profiles):
        d, z = np.asarray(p["dist"], float), np.asarray(p["z"], float)
        k = i if (n > 1 and stacked) else 0
        pts = " ".join(f"{X(a):.3f},{Y(b, k):.3f}" for a, b in zip(d, z))
        col = GREYS[i % len(GREYS)] if (n > 1 and not stacked) else GREYS[0]
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="0.35" '
                   f'stroke-linejoin="round" stroke-linecap="round"/>')
        if n > 1:
            off = p.get("offset", 0.0)
            ly = Y(float(z[-1]), k)
            out.append(f'<text x="{X(d[-1]) + 1.2:.2f}" y="{ly + 1:.2f}" font-size="2.6" fill="{col}">'
                       f'{off:+.3g} {lu}</text>')
    # grooves and ridges of a single section
    if n == 1:
        for f in p0.get("features") or []:
            i = min(len(p0["z"]) - 1, max(0, int(f["index"])))
            cx, cy = X(float(p0["dist"][i])), Y(float(p0["z"][i]))
            up = f["kind"] == "ridge"
            ty = cy - 4 if up else cy + 4
            tri = (f"{cx:.2f},{ty - 1.6:.2f} {cx - 1.5:.2f},{ty + 1.1:.2f} {cx + 1.5:.2f},{ty + 1.1:.2f}" if up
                   else f"{cx:.2f},{ty + 1.6:.2f} {cx - 1.5:.2f},{ty - 1.1:.2f} {cx + 1.5:.2f},{ty - 1.1:.2f}")
            out.append(f'<polygon points="{tri}" fill="{"#2e8b57" if up else "#c07000"}"/>')
            out.append(f'<text x="{cx:.2f}" y="{(ty - 3) if up else (ty + 5):.2f}" font-size="2.8" text-anchor="middle" '
                       f'fill="#222">{f["n"]}</text>')
    # A and B
    out.append(f'<text x="{x0 - 2:.2f}" y="{margin_t - 3:.2f}" font-size="4" font-weight="bold" text-anchor="end">A</text>')
    out.append(f'<text x="{x0 + draw_w + 2:.2f}" y="{margin_t - 3:.2f}" font-size="4" font-weight="bold">B</text>')
    # scale bars: a round length, in real units, drawn at the drawing's scale
    if true_mm:
        bar = next((b for b in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100) if b * s >= 18.0), 100)
        by = ybase + 14.0
        out.append(f'<line x1="{x0:.2f}" y1="{by:.2f}" x2="{x0 + bar * s:.2f}" y2="{by:.2f}" stroke="#000" stroke-width="0.5"/>')
        for xx in (x0, x0 + bar * s):
            out.append(f'<line x1="{xx:.2f}" y1="{by - 1.2:.2f}" x2="{xx:.2f}" y2="{by + 1.2:.2f}" stroke="#000" stroke-width="0.4"/>')
        out.append(f'<text x="{x0 + bar * s / 2:.2f}" y="{by - 1.8:.2f}" font-size="3" text-anchor="middle">{bar:g} mm</text>')
    # title block
    ty0 = ybase + 20.0
    out.append(f'<rect x="{margin_l - 10:.2f}" y="{ty0:.2f}" width="{page_width_mm - margin_l - margin_r + 20:.2f}" '
               f'height="{block_h:.1f}" fill="none" stroke="#000" stroke-width="0.3"/>')
    scale_txt = scale_label(s) if true_mm else "not to scale (no scale set: pixels)"
    vtxt = "true" if abs(v_exag - 1.0) < 1e-9 else f"exaggerated ×{v_exag:g}"
    lines = [f'<tspan font-size="4.2" font-weight="bold">{html.escape(title or "Section A–B")}</tspan>']
    rows = [f"Section A–B · {length:.4g} {lu} long" + (f" · {n} parallel sections {abs(profiles[1]['offset'] - profiles[0]['offset']):.3g} {lu} apart" if n > 1 else ""),
            f"Scale {scale_txt} horizontal · heights {vtxt} · heights in {p0['unit']}",
            f"A at {p0['a_px'][0]:.1f}, {p0['a_px'][1]:.1f} px · B at {p0['b_px'][0]:.1f}, {p0['b_px'][1]:.1f} px"]
    if n == 1 and "ra" in p0:
        rows.append(f"Ra {p0['ra']:.3g} · Rq {p0['rq']:.3g} · Rz {p0['rz']:.3g} {zu}")
    for k, v in meta.items():
        rows.append(f"{k}: {v}")
    rows.append(f"Drawn {datetime.date.today().isoformat()} by {SOFTWARE}")
    xt = margin_l - 8
    out.append(f'<text x="{xt:.2f}" y="{ty0 + 6:.2f}">{lines[0]}</text>')
    for j, row in enumerate(rows[:7]):
        out.append(f'<text x="{xt:.2f}" y="{ty0 + 11.5 + j * 3.6:.2f}" font-size="2.9" fill="#222">{html.escape(row)}</text>')
    out.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return dict(scale=scale_txt, scale_factor=s if true_mm else None, v_exag=float(v_exag),
                page_mm=(page_width_mm, page_h), to_scale=bool(true_mm))


# ── DXF (R12, ASCII: opens in every CAD program) ──────────────────────────────

def section_dxf(path: str, profiles: list[dict], *, v_exag: float = 1.0, stacked: bool = True,
                title: str = "") -> dict:
    """The profiles as polylines in model space, in the section's own unit (mm for a scaled section), x along
    the section and y the height ×*v_exag*; a series stacked one above another when *stacked*.  Layers:
    PROFILE, FEATURES (a point and a number at each groove and ridge), FRAME (the baseline), TEXT."""
    n = len(profiles)
    gap = stack_gap(profiles) if (n > 1 and stacked) else 0.0
    zmin = min(float(np.min(p["z"])) for p in profiles)
    g: list[tuple[int, object]] = []

    def add(code, value):
        g.append((code, value))

    def layer_table():
        add(0, "SECTION"); add(2, "TABLES"); add(0, "TABLE"); add(2, "LAYER"); add(70, 4)
        for name, colour in (("PROFILE", 7), ("FEATURES", 3), ("FRAME", 8), ("TEXT", 7)):
            add(0, "LAYER"); add(2, name); add(70, 0); add(62, colour); add(6, "CONTINUOUS")
        add(0, "ENDTAB"); add(0, "ENDSEC")

    add(0, "SECTION"); add(2, "HEADER"); add(9, "$ACADVER"); add(1, "AC1009")
    add(9, "$INSUNITS"); add(70, 4); add(0, "ENDSEC")                 # 4 = millimetres
    layer_table()
    add(0, "SECTION"); add(2, "ENTITIES")
    for i, p in enumerate(profiles):
        d, z = np.asarray(p["dist"], float), np.asarray(p["z"], float)
        k = i if (n > 1 and stacked) else 0
        y = ((z - zmin) + k * gap) * float(v_exag)
        add(0, "POLYLINE"); add(8, "PROFILE"); add(66, 1); add(70, 0)
        for xx, yy in zip(d, y):
            add(0, "VERTEX"); add(8, "PROFILE"); add(10, round(float(xx), 6)); add(20, round(float(yy), 6)); add(30, 0.0)
        add(0, "SEQEND"); add(8, "PROFILE")
        add(0, "LINE"); add(8, "FRAME")                              # this profile's baseline
        add(10, float(d[0])); add(20, round(k * gap * float(v_exag), 6)); add(30, 0.0)
        add(11, float(d[-1])); add(21, round(k * gap * float(v_exag), 6)); add(31, 0.0)
        if n == 1:
            for f in p.get("features") or []:
                j = min(len(z) - 1, max(0, int(f["index"])))
                add(0, "POINT"); add(8, "FEATURES"); add(10, float(d[j])); add(20, round(float(y[j]), 6)); add(30, 0.0)
                add(0, "TEXT"); add(8, "TEXT"); add(10, float(d[j])); add(20, round(float(y[j]) + 0.5, 6)); add(30, 0.0)
                add(40, max(float(d[-1]) / 60.0, 1e-3)); add(1, f"{f['n']} {f['kind']}")
        else:
            add(0, "TEXT"); add(8, "TEXT"); add(10, float(d[-1]) * 1.01); add(20, round(k * gap * float(v_exag), 6)); add(30, 0.0)
            add(40, max(float(d[-1]) / 60.0, 1e-3)); add(1, f"{p.get('offset', 0.0):+.4g} {p['length_unit']}")
    if title:
        add(0, "TEXT"); add(8, "TEXT"); add(10, 0.0); add(20, -float(profiles[0]["length"]) / 15.0); add(30, 0.0)
        add(40, max(float(profiles[0]["length"]) / 40.0, 1e-3)); add(1, title)
    add(0, "ENDSEC"); add(0, "EOF")
    with open(path, "w", encoding="ascii", errors="replace", newline="\r\n") as f:
        for code, value in g:
            f.write(f"{code:>3}\n{value}\n")
    return dict(entities=sum(1 for c, v in g if c == 0 and v in ("POLYLINE", "LINE", "TEXT", "POINT")),
                profiles=n, unit=profiles[0]["unit"])
