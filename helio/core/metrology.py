"""Scale and distance on the image, GUI-free (MET-001).

The arithmetic of the classic Metrology panel's "1. Draw Scale" and
"2. Set PPI", shared with the Qt Measure workspace so both give the same
numbers (contracts.md §4: millimetres per full-resolution pixel; PPI is
derived from it).
"""
from __future__ import annotations

import math

import numpy as np

UNITS_TO_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4, "inch": 25.4, "inches": 25.4}


def line_length_px(x1: float, y1: float, x2: float, y2: float) -> float:
    """Length of a drawn line, to the fraction of a pixel it was drawn to.  (The classic
    cut the ends to whole pixels: up to 1.4 px off, 1.4 % of the scale on a 100 px line.)"""
    return float(math.hypot(float(x2) - float(x1), float(y2) - float(y1)))


def to_mm(value: float, unit: str) -> float:
    """A length in mm, cm or inches, in millimetres."""
    unit = unit.lower()
    return value * 10.0 if unit == "cm" else (value * 25.4 if unit in ("in", "inches", "inch")
                                              else value)


def scale_from_line(drawn_px: float, value: float, unit: str = "mm") -> tuple[float, float]:
    """(mm per pixel, PPI) from a line *drawn_px* long that measures *value* *unit*.

    Raises ValueError for a non-positive length or line."""
    mm = to_mm(float(value), unit)
    if mm <= 0 or not drawn_px or drawn_px <= 0:
        raise ValueError("Enter a length greater than zero, and draw the line first.")
    ppi = drawn_px / (mm / 25.4)
    mm_per_px = mm / drawn_px
    return mm_per_px, ppi


def distance(x1: float, y1: float, x2: float, y2: float, mm_per_px: float | None):
    """(pixels, millimetres or None) between two image points."""
    px = math.hypot(x2 - x1, y2 - y1)
    return px, (px * mm_per_px if mm_per_px else None)


def level_plane(depth):
    """*depth* less its least-squares plane: the form removal (F-operator, a plane)
    ISO 25178 applies before its height parameters, so a tilted region is not read
    as rough."""
    import numpy as np
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    step = max(1, (h * w) // 400_000)
    A = np.column_stack([xx.ravel()[::step], yy.ravel()[::step], np.ones(len(xx.ravel()[::step]))])
    c, *_ = np.linalg.lstsq(A, d.ravel()[::step], rcond=None)
    return d - (c[0] * xx + c[1] * yy + c[2])


def level_line(z):
    """*z* less its least-squares line: the mean line of ISO 4287."""
    import numpy as np
    z = np.asarray(z, np.float64)
    x = np.arange(len(z), dtype=np.float64)
    a, b = np.polyfit(x, z, 1)
    return z - (a * x + b)


def iso25178(depth, z_mm_per_unit: float = 0.0):
    """The seven ISO 25178 areal height parameters of *depth*, its plane removed.

    *depth* is in pixels of height (helio.application.heights); heights are in mm
    when *z_mm_per_unit* (mm per pixel of height) is set, else in pixels of height.
    Returns (rows, unit): rows are (name, value, meaning)."""
    import numpy as np
    d = level_plane(depth)
    d = d - d.mean()
    f = d.flatten()
    Sq = float(np.sqrt(np.mean(f ** 2)))
    z_mm = float(z_mm_per_unit or 0)
    unit = "mm" if z_mm > 0 else "px of height"
    k = z_mm if z_mm > 0 else 1.0
    rows = [
        ("Sa", float(np.mean(np.abs(f))) * k, "arithmetic mean height"),
        ("Sq", Sq * k, "root mean square height"),
        ("Ssk", float(np.mean(f ** 3) / (Sq ** 3 + 1e-12)), "skewness (unitless)"),
        ("Sku", float(np.mean(f ** 4) / (Sq ** 4 + 1e-12)), "kurtosis (unitless)"),
        ("Sp", float(f.max()) * k, "maximum peak height"),
        ("Sv", float(abs(f.min())) * k, "maximum pit depth"),
        ("Sz", float(f.max() - f.min()) * k, "maximum height"),
    ]
    return rows, unit


# ── Height (MET-002) ───────────────────────────────────────────────────

def profile_line(depth, x1: float, y1: float, x2: float, y2: float):
    """Heights along a line, as the classic profile tool samples them.

    Ends are kept 2% inside the frame (the edge of a solve is its least
    trustworthy part); one sample per pixel of length, bilinear.  Returns
    (heights or None if shorter than 2 px, (ix1, iy1, ix2, iy2), samples)."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    h, w = depth.shape[:2]
    th, tw = int(h * 0.02), int(w * 0.02)
    ix1 = int(np.clip(x1, tw, w - tw)); iy1 = int(np.clip(y1, th, h - th))
    ix2 = int(np.clip(x2, tw, w - tw)); iy2 = int(np.clip(y2, th, h - th))
    num = int(round(np.hypot(ix2 - ix1, iy2 - iy1))) + 1     # samples one pixel apart, both ends in
    if num < 3:
        return None, (ix1, iy1, ix2, iy2), num
    z = map_coordinates(depth, [np.linspace(iy1, iy2, num), np.linspace(ix1, ix2, num)], order=1)
    return z, (ix1, iy1, ix2, iy2), num


def height_reference(z) -> float:
    """The depth difference along a line drawn from base to top of a known feature."""
    import numpy as np
    return float(np.max(z) - np.min(z))


def z_scale(known_mm: float, depth_diff: float) -> float:
    """mm per pixel of height from a feature *known_mm* high that rises *depth_diff*
    pixels of height (helio.application.heights)."""
    if known_mm <= 0:
        raise ValueError("Height must be > 0 mm.")
    if depth_diff < 1e-9:
        raise ValueError("The line shows no depth change: draw it from the flat base to the "
                         "top of the reference feature.")
    return known_mm / depth_diff


def profile_stats(z, num: int, mm_per_px=None, z_mm_per_unit=None, length_px=None,
                  cutoff=None) -> dict:
    """A height profile, with ISO 4287 parameters about its least-squares mean line
    (Ra, Rq, Rz, Rp, Rv: of the whole profile as drawn, form and waviness in,
    which the standard calls the primary profile's Pa, Pq, Pt, Pp, Pv), and,
    under ``iso``, the roughness and waviness the standard's Gaussian filter
    separates (``roughness_waviness``).

    *z* in pixels of height (helio.application.heights), sampled evenly along
    a line *length_px* long (default: one sample a pixel).  Distance in mm when
    *mm_per_px* is set, else pixels; height in mm (or µm when under 0.5 mm)
    when *z_mm_per_unit* (mm per pixel of height) is set, else pixels of
    height.  *cutoff*: the filter's cut-off in the distance unit; None picks
    one (``default_cutoff``)."""
    import numpy as np
    # The samples' own spacing: a slanted line's ends are whole pixels and its
    # length is not, so its samples are a little under or over a pixel apart.
    step_px = (float(length_px) / (num - 1)) if (length_px and num > 1) else 1.0
    if mm_per_px:
        x_axis = np.arange(num) * (step_px * float(mm_per_px)); x_label = "Distance (mm)"
    else:
        x_axis = np.arange(num, dtype=np.float64) * step_px; x_label = "Distance (px)"
    if z_mm_per_unit:
        z_abs = (z - np.min(z)) * z_mm_per_unit
        use_um = (float(np.max(z_abs) - np.min(z_abs))) < 0.5
        z_plot = z_abs * 1000 if use_um else z_abs
        z_unit = "um" if use_um else "mm"
    else:
        z_plot = z - np.min(z); z_unit = "px"
    r = level_line(z_plot) if len(z_plot) >= 3 else z_plot - np.mean(z_plot)   # about the mean line
    Ra = float(np.mean(np.abs(r)))
    Rq = float(np.sqrt(np.mean(r ** 2)))
    Rz = float(np.max(r) - np.min(r))
    Rp = float(np.max(r)); Rv = float(-np.min(r))
    iso = None
    if len(z_plot) >= 16:
        try:
            iso = roughness_waviness(z_plot, float(x_axis[1] - x_axis[0]), cutoff,
                                     ra_um=(Ra * (1.0 if z_unit == "um" else 1000.0)
                                            if (mm_per_px and z_mm_per_unit) else None),
                                     x_mm=bool(mm_per_px))
            iso["x_unit"] = "mm" if mm_per_px else "px"
        except Exception:                                  # noqa: BLE001 - the plain figures stand
            iso = None
    if mm_per_px and z_mm_per_unit:
        label = f"Profile {float(x_axis[-1]):.3f}mm  |  Ra={Ra:.2f} Rq={Rq:.2f} Rz={Rz:.2f} {z_unit}"
    elif z_mm_per_unit:
        label = f"Profile ({num}px)  Ra={Ra:.2f} Rz={Rz:.2f} {z_unit}"
    else:
        label = (f"Profile ({num}px)  Ra={Ra:.3g} Rz={Rz:.3g} px of height  "
                 "(heights in mm once a height scale is set)")
    return dict(x_axis=x_axis, z_plot=z_plot, label=label, x_label=x_label, z_unit=z_unit,
                Ra=Ra, Rq=Rq, Rz=Rz, Rp=Rp, Rv=Rv, iso=iso)


# ── Roughness and waviness: the ISO 16610-21 Gaussian filter (MET-016) ───

GAUSS_ALPHA = math.sqrt(math.log(2.0) / math.pi)        # 0.4697: 50 % transmission at the cut-off
STANDARD_CUTOFFS_MM = (0.08, 0.25, 0.8, 2.5, 8.0)        # ISO 4288 / ISO 21920-3


def gaussian_mean_line(z, dx: float, cutoff: float):
    """The Gaussian mean line of profile *z* (samples *dx* apart) at *cutoff*,
    both in the same unit: ISO 16610-21's weighting function
    exp(-pi (x / (alpha cutoff))^2) / (alpha cutoff), alpha = sqrt(ln 2 / pi),
    which passes half of a sine whose wavelength is the cut-off.

    Near the ends, where the weighting function runs off the profile, the
    weights that remain are renormalised and their first moment taken out (a
    local straight-line fit under the Gaussian weight: ISO 16610-28's
    regression filter of degree 1), so the whole length is usable and a tilted
    straight profile is its own mean line to its ends."""
    z = np.asarray(z, np.float64)
    n = len(z)
    s = GAUSS_ALPHA * float(cutoff) / float(dx)          # in samples
    half = int(min(n - 1, math.ceil(3.0 * s) + 1))
    k = np.arange(-half, half + 1, dtype=np.float64)
    g = np.exp(-math.pi * (k / s) ** 2)
    one, x = np.ones(n), np.arange(n, dtype=np.float64)
    # The full convolution's centre: "same" returns the longer input's length, and the kernel is longer
    # than the profile once the cut-off passes about a third of it.
    conv = lambda a: np.convolve(a, g, mode="full")[half:half + n]
    # Weighted straight-line fit about each point i: moments of (j - i) under g.
    m0 = conv(one)
    m1 = conv(x) - x * m0
    m2 = conv(x * x) - 2 * x * conv(x) + x * x * m0
    b0 = conv(z)
    b1 = conv(x * z) - x * b0
    det = m0 * m2 - m1 * m1
    line = np.where(det > 1e-9 * m0 * m0 + 1e-300, (m2 * b0 - m1 * b1) / np.where(det == 0, 1, det),
                    b0 / np.maximum(m0, 1e-300))
    return line


def default_cutoff(length: float, ra_um=None, unit_mm: bool = True) -> tuple[float, bool]:
    """(cut-off, standard?) for a profile *length* long.

    In millimetres, ISO 4288's choice from the roughness (Ra in µm: up to 0.02
    -> 0.08 mm, to 0.1 -> 0.25, to 2 -> 0.8, to 10 -> 2.5, above -> 8), brought
    down to the largest standard cut-off five of which fit the line when the
    line is shorter than the standard's five sampling lengths.  With no scale,
    or a line shorter than five of the smallest, a fifth of the line: not a
    standard value, and said so."""
    if unit_mm:
        want = 0.8
        if ra_um is not None:
            want = (0.08 if ra_um <= 0.02 else 0.25 if ra_um <= 0.1 else 0.8 if ra_um <= 2.0
                    else 2.5 if ra_um <= 10.0 else 8.0)
        fits = [c for c in STANDARD_CUTOFFS_MM if 5.0 * c <= length * (1 + 1e-9) and c <= want]
        if fits:
            return fits[-1], True
    return length / 5.0, False


def roughness_waviness(z, dx: float, cutoff=None, ra_um=None, x_mm=None) -> dict:
    """ISO 4287 roughness (R) and waviness (W) parameters of a profile, split by
    the ISO 16610-21 Gaussian filter at *cutoff* (lambda-c).

    *z* heights, *dx* the sample spacing, *cutoff* in dx's unit (None:
    ``default_cutoff``; *x_mm* says dx is in millimetres, so a standard
    cut-off can be chosen, and *ra_um*, the roughness in µm, which one).  The
    profile's least-squares line is removed first (the primary profile); the
    Gaussian mean line of that is the waviness, and what is left the
    roughness.  Returns, in z's unit: Ra, Rq, Rsk, Rku, Rp, Rv, Rz (the mean,
    over the sampling lengths of one cut-off each that fit the profile, of
    each one's highest peak to deepest valley: the standard's Rz), Rt (highest
    peak to deepest valley of the whole profile), Wa, Wq, Wt; and cutoff,
    standard (the cut-off is one of the standard series, with five sampling
    lengths in the line), n_lengths, and the two profiles (roughness,
    waviness)."""
    p = level_line(np.asarray(z, np.float64))
    n = len(p)
    length = dx * (n - 1)
    standard = False
    if cutoff is None:
        cutoff, standard = default_cutoff(length, ra_um, unit_mm=(ra_um is not None) if x_mm is None else bool(x_mm))
    cutoff = float(min(max(cutoff, 2.5 * dx), length))
    wav = gaussian_mean_line(p, dx, cutoff)
    r = p - wav
    r = r - r.mean()
    per = max(2, int(round(cutoff / dx)))
    k = max(1, min(n // per, 5 if standard else n // per))
    seg = [r[i * per:(i + 1) * per] for i in range(k)] if n >= per else [r]
    Rq = float(np.sqrt(np.mean(r ** 2)))
    w0 = wav - wav.mean()
    return dict(
        cutoff=cutoff, standard=bool(standard), n_lengths=len(seg),
        Ra=float(np.mean(np.abs(r))), Rq=Rq,
        Rsk=float(np.mean(r ** 3) / Rq ** 3) if Rq > 0 else 0.0,
        Rku=float(np.mean(r ** 4) / Rq ** 4) if Rq > 0 else 0.0,
        Rp=float(np.mean([s_.max() for s_ in seg])), Rv=float(np.mean([-s_.min() for s_ in seg])),
        Rz=float(np.mean([s_.max() - s_.min() for s_ in seg])),
        Rt=float(r.max() - r.min()),
        Wa=float(np.mean(np.abs(w0))), Wq=float(np.sqrt(np.mean(w0 ** 2))),
        Wt=float(w0.max() - w0.min()),
        roughness=r, waviness=wav)


def profile_csv(path: str, x_axis, z_plot, title: str, x_label: str = "Distance",
                z_unit: str = "rel", stats=None, iso=None) -> str:
    """A profile as CSV: title, the mean-line figures, then distance and elevation.
    With *iso* (``roughness_waviness``): the filtered roughness and waviness
    figures too, and the two profiles as further columns (the waviness column
    is about the profile's least-squares line)."""
    import csv
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow([title])
        if stats:
            Ra, Rq, Rz, Rp, Rv = stats
            wr.writerow([f"Ra={Ra:.4f}", f"Rq={Rq:.4f}", f"Rz={Rz:.4f}", f"Rp={Rp:.4f}",
                         f"Rv={Rv:.4f}", z_unit])
        if iso:
            xu = iso.get("x_unit", "")
            wr.writerow([f"Gaussian filter (ISO 16610-21) cut-off={iso['cutoff']:.4g} {xu}",
                         "standard cut-off" if iso["standard"] else "not a standard cut-off (a fifth of the line)",
                         f"sampling lengths={iso['n_lengths']}"])
            wr.writerow(["Roughness"] + [f"{k}={iso[k]:.4f}" for k in
                                         ("Ra", "Rq", "Rz", "Rt", "Rp", "Rv", "Rsk", "Rku")] + [z_unit])
            wr.writerow(["Waviness"] + [f"{k}={iso[k]:.4f}" for k in ("Wa", "Wq", "Wt")] + [z_unit])
        head = [x_label, f"Elevation({z_unit})"]
        if iso:
            head += [f"Waviness({z_unit})", f"Roughness({z_unit})"]
        wr.writerow(head)
        for i in range(len(x_axis)):
            row = [x_axis[i], z_plot[i]]
            if iso:
                row += [iso["waviness"][i], iso["roughness"][i]]
            wr.writerow(row)
    return path


# ── Area, volume and areal texture (MET-013…015) ─────────────────────────

def true_surface_area(normals, mm_per_px=None, nz_floor: float = 0.05) -> dict:
    """The 3D area of a surface from its normals: sum of 1 / n_z per pixel.

    Each pixel's patch of surface is tilted by the angle whose cosine is
    n_z, so its true area is the pixel's area over n_z (n_z floored at
    *nz_floor*, 87°, where the normals stop being trustworthy).  In mm² once
    a scale is set (the classic read a capture-metadata field the scale tool
    never fills, so it stayed in px² after calibration: KD-024)."""
    import numpy as np
    n = np.asarray(normals, np.float32)
    nl = np.linalg.norm(n, axis=-1)
    nz = np.clip(np.divide(n[..., 2], np.maximum(nl, 1e-8)), nz_floor, 1.0)
    h, w = n.shape[:2]
    true_px = float(np.sum(1.0 / nz))
    proj_px = float(h * w)
    out = dict(true_px2=true_px, projected_px2=proj_px, excess_pct=(true_px / proj_px - 1) * 100,
               w=w, h=h, mm_per_px=mm_per_px or None)
    if mm_per_px:
        a = float(mm_per_px) ** 2
        out.update(true_mm2=true_px * a, projected_mm2=proj_px * a)
    return out


def base_plane(depth, border: int = 3):
    """The least-squares plane through a region's border: the surface the
    region sits on, for volumes and step heights."""
    import numpy as np
    d = np.asarray(depth, np.float64)
    h, w = d.shape
    b = max(1, min(border, h // 4, w // 4))
    mask = np.zeros((h, w), bool)
    mask[:b, :] = mask[-b:, :] = mask[:, :b] = mask[:, -b:] = True
    yy, xx = np.nonzero(mask)
    A = np.c_[xx, yy, np.ones(len(xx))]
    c, *_ = np.linalg.lstsq(A, d[mask], rcond=None)
    Y, X = np.mgrid[0:h, 0:w]
    return c[0] * X + c[1] * Y + c[2]


def volume(depth, mm_per_px=None, z_mm_per_unit=None, base: str = "plane") -> dict:
    """Material above and space below a base, over a region.

    *base* "plane" is the least-squares plane through the region's border
    (draw the region a little larger than the feature); "min" is the
    region's lowest point.  In mm³ only with both scales set: the classic
    rescaled the depth to 0-1 and used the pixel size as the height, so its
    "mm³" had no physical meaning (KD-023).  Without them, in relative
    units, said so."""
    import numpy as np
    d = np.asarray(depth, np.float64)
    ref = base_plane(d) if base == "plane" else np.full_like(d, d.min())
    h = d - ref
    above, below = float(np.sum(h[h > 0])), float(-np.sum(h[h < 0]))
    out = dict(base=base, above_units=above, below_units=below, w=d.shape[1], h=d.shape[0],
               max_height_units=float(h.max()), max_depth_units=float(-h.min()))
    if mm_per_px and z_mm_per_unit:
        k = float(mm_per_px) ** 2 * float(z_mm_per_unit)
        out.update(above_mm3=above * k, below_mm3=below * k, net_mm3=(above - below) * k,
                   max_height_mm=out["max_height_units"] * float(z_mm_per_unit),
                   max_depth_mm=out["max_depth_units"] * float(z_mm_per_unit))
    return out


def texture_aspect_ratio(depth, threshold: float = 0.2, directions: int = 72) -> tuple[float, float]:
    """ISO 25178 Str and Std: the autocorrelation's fastest-decay length over
    its slowest (Str, 0-1: near 0 strongly directional, near 1 isotropic),
    and the direction of the slowest decay (Std, degrees from the x axis).

    The decay length in a direction is the first lag where the normalised
    autocorrelation of the mean-removed surface falls to *threshold* (0.2,
    the standard's default).  (The classic took the nearest and farthest of
    every lag below 0.2 in one quadrant, which is not this: KD-025.)"""
    import numpy as np
    from scipy.ndimage import map_coordinates
    # On the levelled surface, as the height parameters are: a tilt left in
    # is one long slope, correlated across the whole region, and was read as
    # a strong lay (an isotropic surface tilted 1 in 50 gave Str 0.56 for
    # 0.82, until 2026-11-10).
    d = level_plane(np.asarray(depth, np.float64))
    d = d - d.mean()
    h, w = d.shape
    f = np.fft.fft2(d, s=(2 * h, 2 * w))                  # zero-padded: no wrap-around
    ac = np.fft.fftshift(np.fft.ifft2(np.abs(f) ** 2).real)
    # Each lag is a sum over the pixels the shifted copies share: divided by
    # how many, or a lag along the region's short side decays faster than
    # the same lag along its long side for no reason of the surface's.
    ac /= np.maximum(h - np.abs(np.arange(-h, h)), 1)[:, None] * np.maximum(w - np.abs(np.arange(-w, w)), 1)[None, :]
    ac /= ac[h, w] + 1e-30
    rmax = min(h, w) / 2.0
    rs = np.linspace(0, rmax, int(rmax * 2) + 2)
    lengths = []
    for a in np.linspace(0, np.pi, directions, endpoint=False):
        xs, ys = w + rs * np.cos(a), h + rs * np.sin(a)
        prof = map_coordinates(ac, [ys, xs], order=1)
        below = np.flatnonzero(prof <= threshold)
        lengths.append(rs[below[0]] if len(below) else rmax)
    lengths = np.asarray(lengths)
    str_ = float(lengths.min() / max(lengths.max(), 1e-12))
    std = float(np.degrees(np.linspace(0, np.pi, directions, endpoint=False)[int(np.argmax(lengths))]))
    return str_, std


def areal(depth, normals=None, z_mm_per_unit=None, mm_per_px=None) -> dict:
    """ISO 25178 areal parameters of a region.

    Heights (Sa, Sq, Sp, Sv, Sz) in mm with a height scale, else relative;
    Ssk, Sku, Str unitless; Std in degrees.  With the normals, the two
    hybrid parameters photometric stereo measures directly: Sdq, the RMS
    surface gradient, and Sdr, the developed interfacial area ratio (%), each
    from the normals rather than by differentiating a reconstructed height.
    The Abbott-Firestone (bearing area) curve is returned as (material %,
    height)."""
    import numpy as np
    rows, unit = iso25178(depth, z_mm_per_unit or 0.0)
    out = {name: value for name, value, _m in rows}
    out["unit"] = unit
    out["Str"], out["Std"] = texture_aspect_ratio(depth)
    if normals is not None:
        n = np.asarray(normals, np.float64)
        nz = np.clip(n[..., 2], 0.05, 1.0)
        # Slopes about the region's mean slope: its best plane removed, as
        # for the heights.  (Left in until 2026-11-10: a region tilted 1 in
        # 10 read Sdq 0.131 for 0.085, Sdr 0.86 % for 0.36 %.)
        sx, sy = n[..., 0] / nz, n[..., 1] / nz
        g2 = (sx - sx.mean()) ** 2 + (sy - sy.mean()) ** 2
        out["Sdq"] = float(np.sqrt(np.mean(g2)))
        out["Sdr"] = float(np.mean(np.sqrt(1.0 + g2)) - 1.0) * 100.0
    d = level_plane(depth)
    flat = np.sort((d - d.mean()).ravel())[::-1] * (z_mm_per_unit or 1.0)
    out["bearing"] = (np.linspace(0, 100, 200), np.interp(np.linspace(0, 100, 200),
                                                          np.linspace(0, 100, len(flat)), flat))
    return out


def true_shape_scale(depth: np.ndarray, normals: np.ndarray) -> float:
    """Pixels of height per unit of *depth*: the scale that makes the height
    map's slopes agree with the normals' (least squares on the well-lit,
    sloping pixels).  1.0 when there is nothing to fit."""
    d = np.asarray(depth, np.float64)
    n = np.asarray(normals, np.float64)
    if d.ndim != 2 or n.ndim != 3 or d.shape != n.shape[:2] or min(d.shape) < 8:
        return 1.0
    step = max(1, int(round(max(d.shape) / 800)))
    d, n = d[::step, ::step], n[::step, ::step]
    from helio.core.frames import depth_slopes
    gx = np.gradient(d, axis=1) / step                      # dz/dcol
    gy = np.gradient(d, axis=0) / step                      # dz/drow
    px, py = depth_slopes(n[..., 0], n[..., 1], np.maximum(n[..., 2], 1e-3))   # the same, from the normals
    ok = (n[..., 2] > 0.2) & np.isfinite(px) & np.isfinite(py)
    # Compared on the broad shape, both smoothed alike.  A difference of the
    # height map between neighbours reads fine texture and noise weaker than
    # the normals give them (a 6 px ripple: 0.83 of its slope), so fitted on
    # every pixel the scale came out too large wherever the surface is
    # grained or the normals noisy: 1.34 on a rippled test surface, 1.16 with
    # 6 degrees of noise, and every height with it (until 2026-11-10).
    if min(d.shape) >= 24:
        import cv2
        lim = 5.0                                           # nz > 0.2 is a slope under 4.9
        blur = lambda a: cv2.GaussianBlur(np.clip(np.nan_to_num(a), -lim, lim), (0, 0), 2.5)
        gx, gy, px, py = blur(gx), blur(gy), blur(px), blur(py)
    g = np.concatenate([gx[ok], gy[ok]])
    t = np.concatenate([px[ok], py[ok]])
    gg = float(g @ g)
    if gg < 1e-12:
        return 1.0
    s = float(g @ t) / gg
    return s if np.isfinite(s) and s > 0 else 1.0


def profile_features(dist, z, *, min_share: float = 0.08, noise_k: float = 7.0) -> list[dict]:
    """The grooves and ridges a section crosses, each with its depth (or
    height), width and position, in the profile's own units.

    Measured about the straight line between the section's ends, so a tilted
    surface does not count as one long slope.  A feature must stand out by
    *min_share* of the profile's range and by *noise_k* × its noise (the
    median absolute step, a robust sigma), so grain is not read as tooling.
    Depth is the feature's prominence (below the shoulders either side);
    width is taken at half that depth.  For a tooled binding: the depth and
    width of each tool impression the line crosses."""
    from scipy.signal import find_peaks, peak_widths
    d = np.asarray(dist, np.float64)
    zz = np.asarray(z, np.float64)
    n = len(zz)
    if n < 8 or not np.all(np.isfinite(zz)):
        return []
    dz = zz - np.linspace(zz[0], zz[-1], n)
    steps = np.diff(dz)
    noise = 1.4826 * float(np.median(np.abs(steps - np.median(steps)))) / math.sqrt(2) + 1e-12
    span = float(np.ptp(dz))
    if span <= 0:
        return []
    need = max(min_share * span, noise_k * noise)
    step = float(d[-1] - d[0]) / max(n - 1, 1)
    # The surface's own level: a flat stretch between two grooves stands
    # above both floors, but it is the surface, not a ridge.
    level = float(np.median(dz))
    out = []
    for kind, sig in (("groove", -dz), ("ridge", dz)):
        idx, props = find_peaks(sig, prominence=need)
        keep = sig[idx] - (-level if kind == "groove" else level) >= 0.5 * need
        idx = idx[keep]
        props = {k: v[keep] for k, v in props.items()}
        if not len(idx):
            continue
        base = -level if kind == "groove" else level
        for j, i in enumerate(idx):
            pr = float(props["prominences"][j])
            # below (or above) the surface around it, never past it: a ridge
            # between two grooves is not as tall as the grooves are deep
            depth = min(pr, float(sig[i] - base))
            wd = peak_widths(sig, [i], rel_height=0.5 * depth / pr, prominence_data=(
                props["prominences"][j:j + 1], props["left_bases"][j:j + 1],
                props["right_bases"][j:j + 1]))[0][0]
            out.append(dict(kind=kind, at=float(d[i]), depth=depth, width=float(wd * step),
                            index=int(i)))
    out.sort(key=lambda f: f["at"])
    for k, f in enumerate(out, 1):
        f["n"] = k
    return out


def region_volume(z: np.ndarray, mask: np.ndarray, cell: float) -> dict | None:
    """Area and volume of a region of a height map, about the plane through
    its outline (a loss, a blind-tooled hollow, a raised boss).

    *z* heights and *cell* the side of one grid cell, in the same unit (mm
    or px); *mask* the region (bool, z's shape).  The reference plane is the
    least-squares plane through the region's edge, so the result does not
    depend on the object's tilt.  Returns {area, surface, below, above,
    deepest, highest, n} (areas in unit², volumes in unit³, heights in
    unit), or None for a region too small to fit a plane to."""
    import cv2
    m = np.asarray(mask, bool)
    if m.sum() < 9:
        return None
    m8 = m.astype(np.uint8)
    edge = (m8 - cv2.erode(m8, np.ones((3, 3), np.uint8))).astype(bool)
    ys, xs = np.nonzero(edge)
    if len(xs) < 3:
        return None
    zf = np.asarray(z, np.float64)
    A = np.stack([xs * cell, ys * cell, np.ones_like(xs, np.float64)], axis=1)
    coef = np.linalg.lstsq(A, zf[ys, xs], rcond=None)[0]
    yy, xx = np.nonzero(m)
    plane = coef[0] * xx * cell + coef[1] * yy * cell + coef[2]
    d = zf[yy, xx] - plane
    gy, gx = np.gradient(zf, cell)
    slope = np.sqrt(1.0 + gx[yy, xx] ** 2 + gy[yy, xx] ** 2)
    a = cell * cell
    return dict(area=float(len(d) * a), surface=float(slope.sum() * a),
                below=float(np.clip(-d, 0, None).sum() * a), above=float(np.clip(d, 0, None).sum() * a),
                deepest=float(max(0.0, -d.min())), highest=float(max(0.0, d.max())), n=int(len(d)))


# ── accuracy on references (CAL-007) ─────────────────────────────────

def _plane_fit(z: "np.ndarray"):
    """Least-squares plane through *z* (h, w): (coefficients a, b, c of
    a*x + b*y + c, residuals (h, w)), x and y in pixels from the box's corner."""
    import numpy as np
    h, w = z.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
    ok = np.isfinite(z.ravel())
    coef, *_ = np.linalg.lstsq(A[ok], z.ravel()[ok].astype(np.float64), rcond=None)
    return coef, (z - (A @ coef).reshape(h, w))


def flatness(H, box) -> dict:
    """How flat a region of the height map *H* (pixels of height) is: the
    residual about its own best plane, RMS and peak-to-valley, in pixels of
    height, and the plane's tilt in degrees.  On a flat reference (a glass
    plate, a gauge block's face) every figure is error."""
    import numpy as np
    x1, y1, x2, y2 = (int(v) for v in box)
    z = np.asarray(H, np.float64)[y1:y2, x1:x2]
    if z.shape[0] < 4 or z.shape[1] < 4:
        raise ValueError("Draw a larger box: at least 4 x 4 pixels.")
    coef, r = _plane_fit(z)
    tilt = float(np.degrees(np.arctan(np.hypot(coef[0], coef[1]))))
    return dict(rms=float(np.sqrt(np.mean(r ** 2))), pv=float(np.max(r) - np.min(r)),
                p99=float(np.percentile(r, 99.5) - np.percentile(r, 0.5)), tilt_deg=tilt,
                n=int(z.size))


def step_height(H, box_ref, box_top) -> dict:
    """A step measured as gauge blocks are: a plane fitted to the reference
    surface (*box_ref*) and extended under the top (*box_top*); the step is the
    top's mean height above that plane, in pixels of height (and its spread)."""
    import numpy as np
    H = np.asarray(H, np.float64)
    x1, y1, x2, y2 = (int(v) for v in box_ref)
    zr = H[y1:y2, x1:x2]
    if zr.shape[0] < 4 or zr.shape[1] < 4:
        raise ValueError("Draw a larger box on the reference surface.")
    coef, rr = _plane_fit(zr)
    a, b, c = coef
    X1, Y1, X2, Y2 = (int(v) for v in box_top)
    zt = H[Y1:Y2, X1:X2]
    if zt.shape[0] < 2 or zt.shape[1] < 2:
        raise ValueError("Draw a larger box on the top of the step.")
    yy, xx = np.mgrid[Y1:Y2, X1:X2].astype(np.float64)
    above = zt - (a * (xx - x1) + b * (yy - y1) + c)
    sep = float(np.hypot((X1 + X2) / 2 - (x1 + x2) / 2, (Y1 + Y2) / 2 - (y1 + y2) / 2))
    return dict(step=float(np.mean(above)), spread=float(np.std(above)),
                ref_rms=float(np.sqrt(np.mean(rr ** 2))), separation_px=sep)


def path_height_sigma(sigma_deg, x1, y1, x2, y2) -> float | None:
    """1-sigma uncertainty, in pixels of height, of the height difference
    between two points, from the normals' own precision along the line
    (helio.core.processing.normal_uncertainty_deg).

    Each pixel's slope error along the line is about tan(sigma)/sqrt(2) (half
    the normal's angular variance lies across the line); summed as a random
    walk over the samples.  An upper estimate for random error: the integrator
    averages over many paths, which only shrinks it.  Systematic error (light
    calibration, lens) is not in it: that is what the accuracy check on a
    reference measures."""
    import numpy as np
    from scipy.ndimage import map_coordinates
    if sigma_deg is None:
        return None
    s = np.asarray(sigma_deg, np.float32)
    num = int(round(np.hypot(x2 - x1, y2 - y1))) + 1
    if num < 2:
        return 0.0
    v = map_coordinates(s, [np.linspace(y1, y2, num), np.linspace(x1, x2, num)], order=1)
    slope = np.tan(np.radians(np.clip(v, 0.0, 60.0))) / np.sqrt(2.0)
    return float(np.sqrt(np.sum(slope.astype(np.float64) ** 2)))
