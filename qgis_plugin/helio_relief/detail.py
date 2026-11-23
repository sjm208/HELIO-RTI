"""
Separating detail from noise - the numerical side of the Detail tab (no Qt).

A height map holds relief at every feature size, from the object's form down to single pixels. This module:
- measures how much relief there is at each size (radially averaged power spectrum) and fits it with two power laws
  - the relief, rising steeply with size, and noise, flatter (white pixel noise is flat; noise in normals that were
  integrated to heights rises as size squared) - which gives the size below which noise outweighs relief, if any;
- isolates chosen bands of feature sizes (smooth band-passes in log size, one DCT), each with its own gain, and
  optionally separates detail from noise inside them: the Wiener gain S / (S + N) from that model, a gate that
  keeps strong local structure (tool marks, strokes) and damps faint uniform texture (grain), and edge-preserving
  smoothing;
- shows the result many ways - signed relief in grey or colour, hillshaded, as Clean relief, as line strength, as a
  map of where the band is strong, or what the separation removed (which should look structureless).
Scale constants come from a full-resolution reference crop (detail_stats), so previews and exports agree.
"""

import math

import numpy as np

try:
    from . import relief_engine as eng
    from . import tools
except ImportError:                     # tests import the modules directly
    import relief_engine as eng
    import tools

DISPLAYS = {
    "signed": "Signed relief (grey: raised light, cut dark)",
    "signed_colour": "Signed relief (colour: raised red, cut blue)",
    "hillshade": "Hillshade of the band (the key light)",
    "clean": "Clean relief of the band (lit from every side)",
    "lines": "Line strength (grooves dark, like a drawing)",
    "energy": "Where the band is strong",
    "noise": "What the separation removed",
}
N_BANDS = 3
BAND_DEFAULTS = [dict(on=True, fine=2.0, broad=64.0, gain=1.0), dict(on=False, fine=1.5, broad=8.0, gain=1.0),
                 dict(on=False, fine=16.0, broad=128.0, gain=1.0)]
DEFAULTS = dict(bands=BAND_DEFAULTS[:1], soft=0.5, wiener=0.0, gate=0.0, clean=0.0, display="signed", gain=1.0)
PRESETS = {                               # fine / broad in full-resolution px; 0 = no limit, -1 = at the noise crossover
    "Tooling and strokes (2-64 px)": dict(fine=2.0, broad=64.0),
    "Fine detail only (1.5-12 px)": dict(fine=1.5, broad=12.0),
    "Main marks (6-48 px)": dict(fine=6.0, broad=48.0),
    "Form without the detail (32 px and up)": dict(fine=32.0, broad=0.0),
    "Everything above the noise": dict(fine=-1.0, broad=0.0),
    "Noise only (below the crossover)": dict(fine=0.0, broad=-1.0),
}


def params(d=None):
    out = dict(DEFAULTS)
    d = dict(d or {})
    if "bands" not in d and ("fine" in d or "broad" in d):
        d["bands"] = [dict(on=True, fine=d.pop("fine", 2.0), broad=d.pop("broad", 0.0), gain=1.0)]
    out.update(d)
    return out


def _detrended(z, plane=None, origin=(0.0, 0.0), px_scale=1.0):
    z, _valid = eng.fill_nodata(z)
    z = np.asarray(z, dtype=np.float32)
    if plane is None:
        return z - eng.plane_surface(eng.fit_plane(z), z.shape)
    return z - eng.plane_surface(plane, z.shape, origin, px_scale)


def radial_spectrum(z, nbins=48):
    """(feature sizes px, power): the radially averaged power spectrum of a height window, feature size = 1 /
    spatial frequency (px per cycle), from 2 px (the finest a grid can hold) to the window size. The window is
    detrended and tapered (Hann) so its edges do not leak."""
    z = _detrended(z).astype(np.float64)
    h, w = z.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    power = np.abs(np.fft.rfft2(z * win)) ** 2 / max(float((win ** 2).sum()), 1e-12)
    f = np.sqrt(np.fft.fftfreq(h)[:, None] ** 2 + np.fft.rfftfreq(w)[None, :] ** 2)
    edges = np.geomspace(2.0 / max(h, w), 0.5, nbins + 1)
    idx = np.digitize(f.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    counts = np.bincount(idx[ok], minlength=nbins)
    total = np.bincount(idx[ok], power.ravel()[ok], minlength=nbins)
    keep = counts > 0
    centres = np.sqrt(edges[:-1] * edges[1:])[keep]
    sizes, pw = 1.0 / centres, (total[keep] / counts[keep])
    order = np.argsort(sizes)
    return sizes[order], pw[order]


def signal_power(model, size):
    return model["amp"] * np.asarray(size, float) ** model["slope"]


def noise_power(model, size):
    return model["noise"] * np.asarray(size, float) ** model["noise_slope"]


def fit_noise(sizes, power):
    """Fit power = relief + noise, each a power law in feature size: relief amp x size^slope, steep; noise
    noise x size^noise_slope with noise_slope 0-2.2 (0: white pixel noise; 2: noise in normals, integrated).
    crossover = the size where they are equal (below it noise wins); detected = the noise is resolvable (the
    crossover is at 1.6 px or more). Fitted in log space with a robust loss, so a few odd bins do not steer it."""
    s, p = np.asarray(sizes, float), np.asarray(power, float)
    ok = p > 0
    top = max(8.0, min(128.0, float(s.max()) / 3.0))
    use = ok & (s <= top)
    ls, lp = np.log(s[use]), np.log(p[use])
    lo = math.log(max(float(p[use].min()), 1e-300))
    wt = float(s[use].min()) / s[use]     # a bin's log power scatters as 1 / sqrt(its sample count), ~ its size
    mid = s[use] >= 6.0
    slope0, la0 = (np.polyfit(ls[mid], lp[mid], 1, w=wt[mid]) if mid.sum() >= 3 else (2.0, float(lp.max())))
    slope0 = float(min(max(slope0, 0.6), 8.0))
    # Noise shows as power at the finest sizes above the relief's power law. A clean power law sits ~0.25 (log)
    # above its own straight-line fit there (tapering and binning); integrated maps sit far below it.
    fine = s[use] <= 3.0
    excess = float(np.median(lp[fine] - (la0 + slope0 * ls[fine]))) if fine.any() else 0.0
    if excess < 0.35:
        return dict(amp=math.exp(la0), slope=slope0, noise=math.exp(la0) * 1e-9, noise_slope=0.0, crossover=1.0,
                    detected=False, excess=excess)

    def resid(v):                  # v = log amp, slope - noise slope, log noise, noise slope
        return (np.logaddexp(v[0] + (v[1] + v[3]) * ls, v[2] + v[3] * ls) - lp) * wt

    try:
        from scipy.optimize import least_squares
        best = None
        for b0 in (0.0, 2.0):
            for f0 in (lo - 2 * b0 - 4.0, lo - 2 * b0):
                x0 = [la0, max(slope0 - b0, 0.4), f0, b0]
                r = least_squares(resid, x0, loss="soft_l1", f_scale=0.3,
                                  bounds=([-np.inf, 0.3, -np.inf, 0.0], [np.inf, 9.0, np.inf, 2.2]))
                if best is None or r.cost < best.cost:
                    best = r
        la, da, lf, b = (float(v) for v in best.x)
    except Exception:              # no scipy: a straight line above white noise at the finest sizes' level
        la, da, lf, b = float(la0), slope0, lo, 0.0
    cross = math.exp((lf - la) / da)
    cross = float(np.clip(cross, 1.0, float(s.max())))
    return dict(amp=math.exp(la), slope=da + b, noise=math.exp(lf), noise_slope=b, crossover=cross,
                detected=bool(cross >= 1.6), excess=excess)


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def resolve_band(band, model):
    """(fine, broad) in px with the presets' -1 ('at the crossover') replaced by the model's crossover size."""
    fine, broad = float(band["fine"]), float(band["broad"])
    cross = float(model["crossover"]) if model else 2.0
    return (cross if fine < 0 else fine), (cross if broad < 0 else broad)


def active_bands(d, model=None):
    """[(fine, broad, gain)] of the bands that are on, sizes resolved."""
    return [resolve_band(b, model) + (float(b.get("gain", 1.0)),) for b in params(d)["bands"] if b.get("on", True)]


def finest(d, model=None):
    fines = [f for f, _b, _g in active_bands(d, model) if f > 0]
    return min(fines) if fines else 1.0


def band_multiplier(fy, fx, fine_px, broad_px, soft=0.5, px_scale=1.0):
    """Spectral gain keeping feature sizes between fine_px and broad_px (full-resolution px), rolling off smoothly
    over `soft` octaves either side (small = a hard cut). fine_px 0: no lower limit; broad_px 0: no upper limit."""
    f = np.sqrt(fy * fy + fx * fx).astype(np.float64)
    with np.errstate(divide="ignore"):
        size = np.where(f > 0, px_scale / np.maximum(f, 1e-12), np.inf)
    L = np.log2(size)
    soft = max(float(soft), 0.05)
    m = np.ones(np.broadcast(fy, fx).shape, np.float64)
    if fine_px > 0:
        m = m * _smoothstep((L - math.log2(fine_px)) / soft + 0.5)
    if broad_px > 0:
        m = m * np.where(np.isfinite(L), _smoothstep((math.log2(broad_px) - L) / soft + 0.5), 0.0)
    return m.astype(np.float32)


def wiener_multiplier(fy, fx, model, strength=1.0, px_scale=1.0):
    """S / (S + strength x N) at every frequency, from the fitted relief and noise power laws."""
    f = np.sqrt(fy * fy + fx * fx).astype(np.float64)
    size = px_scale / np.maximum(f, 1e-12)
    S, N = signal_power(model, size), noise_power(model, size)
    return (S / (S + max(float(strength), 0.0) * N)).astype(np.float32)


def response(sizes, d, model=None):
    """The linear part of the separation at each feature size (the band gains, soft edges and noise suppression):
    what the chart draws as 'kept'."""
    d = params(d)
    f = 1.0 / np.asarray(sizes, float)
    z = np.zeros_like(f)
    m = sum((g * band_multiplier(f, z, a, b, d["soft"]) for a, b, g in active_bands(d, model)), np.zeros_like(f))
    if float(d["wiener"]) > 0 and model:
        m = m * wiener_multiplier(f, z, model, d["wiener"])
    return np.asarray(m, float)


def _gate_sigma(d, model=None):
    return max(2.0 * finest(d, model), 3.0)


def margin_px(d):
    """Overlap (full-resolution px) a window needs so the result matches an untiled one."""
    d = params(d)
    reach = 8.0
    for b in d["bands"]:
        if b.get("on", True):
            fine, broad = abs(float(b["fine"])), abs(float(b["broad"]))
            reach = max(reach, 1.5 * broad if broad > 0 else 4.0 * max(fine, 2.0))
    if float(d["gate"]) > 0:
        reach += 3.0 * _gate_sigma(d)
    return int(min(1024, reach + 4.0 * float(d["clean"]) + 16))


def _local_energy(band, sigma):
    return np.sqrt(np.maximum(eng.gaussian_blur(band * band, sigma), 0.0))


def isolate(z, d, px_scale=1.0, stats=None, plane=None, origin=(0.0, 0.0)):
    """The chosen bands of a height window, separated from noise as set. Returns (band, removed): removed is what
    the separation (noise suppression, texture gate, clean-up) took out of the band-passed relief."""
    d = params(d)
    stats = {} if stats is None else stats
    zf = _detrended(z, plane, origin, px_scale)
    model = stats.get("model") or fit_noise(*radial_spectrum(zf))
    bands = active_bands(d, model)
    if not bands:
        zero = np.zeros_like(zf)
        return zero, zero.copy()
    reach = max((0.5 * b / px_scale if b > 0 else 2.0 * max(a, 1.0) / px_scale) for a, b, _g in bands)
    spec = eng.Spectrum(zf, eng._reach(reach))
    bm = sum((np.float32(g) * band_multiplier(spec.fy, spec.fx, a, b, d["soft"], px_scale) for a, b, g in bands),
             np.zeros((spec.fy.shape[0], spec.fx.shape[1]), np.float32))
    raw = spec.filtered(bm)
    band = (spec.filtered(bm * wiener_multiplier(spec.fy, spec.fx, model, d["wiener"], px_scale))
            if float(d["wiener"]) > 0 else raw)
    if float(d["gate"]) > 0:
        e = _local_energy(band, _gate_sigma(d, model) / px_scale)
        ref = float(stats.get("energy_ref") or np.median(e))
        t2 = np.float32((float(d["gate"]) * ref) ** 2)
        band = band * (e * e / (e * e + t2 + np.float32(1e-30)))
    rad = float(d["clean"]) / px_scale
    if rad >= 0.5:
        scale = stats.setdefault("detail_edge_scale", eng._local_relief_scale(band, rad))
        band = eng.guided_smooth(band, rad, (0.5 * scale) ** 2)
    return band.astype(np.float32), (raw - band).astype(np.float32)


def reference_model(ref):
    """The noise model of a full-resolution reference crop (it does not depend on the bands chosen)."""
    return fit_noise(*radial_spectrum(ref))


def detail_stats(ref, d, dx=1.0, dy=1.0, plane=None, origin=(0.0, 0.0), model=None):
    """Display constants and the noise model from a full-resolution reference crop. The constants come from its
    middle 512 px (plenty for percentiles, and quick enough to redo while a handle is dragged)."""
    d = params(d)
    st = {"model": model or reference_model(ref)}
    h, w = np.shape(ref)
    if h > 512 or w > 512:
        y0, x0 = max(0, (h - 512) // 2), max(0, (w - 512) // 2)
        ref = np.asarray(ref)[y0:y0 + 512, x0:x0 + 512]
        origin = (origin[0] + x0, origin[1] + y0)
    plain, _ = isolate(ref, dict(d, gate=0.0, clean=0.0), 1.0, st, plane, origin)
    st["energy_ref"] = float(max(np.median(_local_energy(plain, _gate_sigma(d, st["model"]))), 1e-30))
    band, removed = isolate(ref, d, 1.0, st, plane, origin)
    st["scale"] = float(max(np.percentile(np.abs(band), 99.5), 1e-12))
    x, y = eng.gdal_slopes(band, dx, dy, False, True)
    st["level"] = 1.0 / float(max(np.percentile(np.hypot(x, y)[::2, ::2], 95.0), 1e-12))
    ls, _ = tools.line_strength(band, max(1.5 * finest(d, st["model"]), 3.0))
    st["line_scale"] = float(max(np.percentile(ls, 99.5), 1e-12))
    st["noise_scale"] = float(max(np.percentile(np.abs(removed), 99.5), 0.05 * st["scale"]))
    return st


class _Prep:
    """Just enough of relief_engine.Prepared for _clean_image."""
    normals = None


def render(band, removed, d, p, dx, dy, stats, px_scale=1.0):
    """The band shown the chosen way, as a float image in 0..1 (RGB for the colour display). p: render parameters
    (the hillshade and Clean relief displays use its light and tone)."""
    d = params(d)
    mode, gain = d["display"], float(d["gain"])
    sc = float(stats.get("scale", 1.0))
    fine = finest(d, stats.get("model"))
    if mode in ("hillshade", "clean"):
        x, y = eng.gdal_slopes(band, dx, dy, False, True)
        pr = _Prep()
        lv = np.float32(stats.get("level", 1.0) * gain)
        pr.normals = ((x * lv).astype(np.float32), (y * lv).astype(np.float32))
        mp = eng.merged_params(p)
        if mode == "hillshade":
            mp = dict(mp, clean_allround=0.0)
        return eng._clean_image(pr, mp)
    if mode == "lines":
        ls, _ = tools.line_strength(band, max(1.5 * fine / px_scale, 1.5))
        return (1.0 - np.clip(ls / stats.get("line_scale", 1.0) * gain, 0.0, 1.0)).astype(np.float32)
    if mode == "energy":
        e = _local_energy(band, max(2.0 * fine, 4.0) / px_scale)
        return np.clip(e / (0.45 * sc) * gain, 0.0, 1.0).astype(np.float32)
    if mode == "noise":
        return (0.5 + 0.45 * np.tanh(removed / stats.get("noise_scale", sc) * gain)).astype(np.float32)
    t = np.tanh(band / sc * 1.5 * gain)
    if mode == "signed_colour":
        base = np.float32(0.96)
        red, blue = np.array([0.80, 0.18, 0.12], np.float32), np.array([0.12, 0.32, 0.80], np.float32)
        a = np.abs(t)[..., None]
        col = np.where((t > 0)[..., None], red, blue)
        return (base + a * (col - base)).astype(np.float32)
    return (0.5 + 0.45 * t).astype(np.float32)
