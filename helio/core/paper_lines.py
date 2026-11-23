"""Counting the lines of a sheet along a section: chain lines, laid lines, ruling.

(Copied from MSI_Reveal core/paper_lines.py on 2026-11-11, unchanged but for this note.)

The 3D Relief view's section tool draws a line A-B across the page. This module
turns the picture under that line into a *profile* and finds, numbers and
measures every line of the sheet's structure the section crosses:

* **chain lines** -- the widely spaced lines of a paper mould (12-45 mm apart,
  typically 20-30 mm). Their intervals are the moldmate fingerprint.
* **laid lines** -- the close-set wires at right angles to them (0.6-1.7 mm,
  6-17 per cm). Their density, and how it varies along the sheet, is the other.
* **ruling** -- for parchment, which is skin and has no mould lines: the ruled
  (drypoint, plummet or ink) lines of the page layout, 3-20 mm apart.

A section is drawn *across* the lines to be counted, so chain lines are counted
on a section that runs along the laid lines, and laid lines on one that runs
along the chain lines.

**A swath, not a line.** One row of pixels is mostly fibre noise. The profile
is the average of a strip of the given width centred on A-B, taken along the
direction the lines run, so the lines add up and the noise, ink and fibres
average away. If the section is not drawn quite square to the lines the
average smears them; :func:`swath_profile` can slant the averaging direction a
few degrees to whichever angle keeps the lines sharpest (``square=True``).

**Light or dark.** Mould wires thin the sheet, so in transmitted light laid and
chain lines are usually *lighter* than the paper; chain lines often also carry
darker "shadows" of pulp either side, and in reflected or raking light the
lines can read either way. ``polarity="auto"`` decides from the profile itself:
sparse lines make the detrended profile skewed towards them.

**A strip, not just its average.** On a real sheet a chain line is a 2-3 %
change of brightness under writing, stains and the cloudy formation of the
pulp, each several times stronger. What sets the line apart is that it *runs on*:
it crosses the whole strip, at one angle, while ink and cloud do not.
:func:`count_lines_strip` therefore keeps the strip as a picture
(:func:`swath_strip`), filters every row of it at the width of a line, masks
the writing (from reflected light when there is some), finds the angle that
lines the rows up, and adds them with the extremes left out. Each line is then
checked against the strip cut into bands -- "seen in 7 of 8" -- and, for chain
lines and ruling, against the rhythm of the others, which is allowed to place
a faint line between two clear ones but never to invent a run of them.

**Spacings are perpendicular.** A section that is not square to the lines
crosses them further apart than they are. Positions stay where the section
crosses each line; spacings are multiplied by cos(slant) once the slant is
known, so they are the distance between the lines themselves.

Pure numpy: no Qt, no SciPy. Positions are in the pixels of the image sampled;
the caller converts to capture pixels and millimetres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = [
    "KINDS",
    "SPACING_MM",
    "LineCount",
    "swath_profile",
    "swath_profile_tiled",
    "swath_strip",
    "swath_strip_tiled",
    "count_lines",
    "count_lines_strip",
    "suggest_swath_mm",
    "survey_page",
    "edit_lines",
]

#: The kinds of line that can be counted, with a reader's name.
KINDS = {"auto": "Auto (chain or laid)", "chain": "Chain lines", "laid": "Laid lines",
         "ruling": "Ruling (parchment)"}
#: Plausible spacing of each kind, in mm.
SPACING_MM = {"chain": (12.0, 45.0), "laid": (0.6, 1.7), "ruling": (3.0, 20.0)}
#: Without a resolution, the spacing searched, as fractions of the section's length
#: (chain, ruling) or in pixels (laid).
_FALLBACK_FRACTION = {"chain": (1 / 40, 1 / 1.6), "ruling": (1 / 80, 1 / 3)}
_FALLBACK_LAID_PX = (3.0, 40.0)
#: Autocorrelation height (0-1) above which a laid rhythm is believed in "auto".
_LAID_AUTO_CONFIDENCE = 0.30


# ---------------------------------------------------------------------- #
# Small signal tools
# ---------------------------------------------------------------------- #
def _gauss1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur of a 1-D signal, edges reflected, NaN-free input."""
    values = np.asarray(values, dtype="float64")
    if sigma <= 0.3 or values.size < 3:
        return values.copy()
    radius = int(min(math.ceil(3.0 * sigma), values.size - 1))
    x = np.arange(-radius, radius + 1, dtype="float64")
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Odd reflection keeps a straight slope straight past the ends: a light box
    # brighter on one side must not bend into a V at the edges and read as a line.
    padded = np.pad(values, radius, mode="reflect", reflect_type="odd")
    return np.convolve(padded, kernel, mode="valid")


def _robust_spread(values: np.ndarray) -> float:
    centre = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - centre)))
    return mad if mad > 0 else float(np.std(values)) or 1e-12


def _fill_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    index = np.arange(values.size)
    return np.interp(index, index[finite], values[finite])


def _bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """*image* at (xs, ys) (pixel centres at integers), NaN outside."""
    h, w = image.shape
    inside = (xs >= -0.5) & (xs <= w - 0.5) & (ys >= -0.5) & (ys <= h - 0.5)
    x = np.clip(xs, 0, w - 1)
    y = np.clip(ys, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(np.int64), w - 1)
    y0 = np.minimum(np.floor(y).astype(np.int64), h - 1)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    out = top * (1 - fy) + bottom * fy
    return np.where(inside, out, np.nan)


# ---------------------------------------------------------------------- #
# Sampling the swath
# ---------------------------------------------------------------------- #
def _sample(image: np.ndarray, a, b, width: float, step: float, slant_deg: float = 0.0,
            origin=(0.0, 0.0), count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Distances along A-B and the swath mean at each, for one slant.

    *count* fixes the number of samples (the tiled reader needs exactly the
    whole section's); otherwise one every *step* pixels.
    """
    dist, _offsets, values = _sample_strip(image, a, b, width, step, slant_deg, origin, count)
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(values).sum(axis=1)
        mean = np.where(counts > 0, np.nansum(values, axis=1) / np.maximum(counts, 1), np.nan)
    return dist, mean


def _across_offsets(width: float, step: float, rows: int | None = None) -> np.ndarray:
    half = max(0.0, width / 2.0)
    if rows is not None:
        return np.linspace(-half, half, max(1, int(rows))) if half > 0 and rows > 1 else np.zeros(1)
    n_across = max(1, int(round(width / max(step, 0.5))) + 1) if half > 0 else 1
    return np.linspace(-half, half, n_across) if n_across > 1 else np.zeros(1)


def _sample_strip(image: np.ndarray, a, b, width: float, step: float, slant_deg: float = 0.0,
                  origin=(0.0, 0.0), count: int | None = None,
                  rows: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Distances along A-B, offsets across it, and the strip ``[along, across]`` itself."""
    (ax, ay), (bx, by) = a, b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length < 1e-9 and count is None:
        raise ValueError("the section has no length")
    tx, ty = (dx / length, dy / length) if length >= 1e-9 else (1.0, 0.0)   # along the section
    nx, ny = -ty, tx                                 # across it
    shear = math.tan(math.radians(slant_deg))
    # The averaging direction: across the section, slanted along it by the shear.
    vx, vy = nx + shear * tx, ny + shear * ty
    n_along = count if count is not None else max(2, int(math.floor(length / step + 1e-9)) + 1)
    dist = np.linspace(0.0, length, n_along)
    offsets = _across_offsets(width, step, rows)
    xs = ax + dist[:, None] * tx + offsets[None, :] * vx - origin[0]
    ys = ay + dist[:, None] * ty + offsets[None, :] * vy - origin[1]
    values = _bilinear(np.asarray(image, dtype="float64"), xs, ys)
    return dist, offsets, values


def _sharpness(profile: np.ndarray, sigma: float) -> float:
    """Line-scale energy of a profile: what a well-squared average maximises."""
    p = _fill_nan(profile)
    fine = _gauss1d(p, max(0.6, sigma / 6.0)) - _gauss1d(p, max(1.5, sigma * 1.5))
    return float(np.mean(fine ** 2))


def swath_profile(image: np.ndarray, a, b, *, width: float = 8.0, step: float = 1.0,
                  square: bool = False, scale_hint: float | None = None,
                  max_slant_deg: float = 8.0) -> tuple[np.ndarray, np.ndarray, float]:
    """The profile along A-B of the strip *width* pixels wide centred on it.

    *a*, *b* are (x, y) in *image* pixels. Returns ``(distance, profile,
    slant_deg)``: distance along the section in pixels, the strip's mean at
    each, and the slant applied to the averaging direction (0 unless
    *square*). With *square*, slants up to ``max_slant_deg`` either way are
    tried and the one that keeps line-scale detail sharpest is kept --
    *scale_hint* is the expected line spacing in pixels, if known.
    """
    image = np.asarray(image, dtype="float64")
    if image.ndim != 2:
        raise ValueError("the image must be one band (2-D)")
    dist, profile = _sample(image, a, b, width, step)
    slant = 0.0
    if square and width > 2 * step:
        sigma = scale_hint or max(4.0, width / 2.0)
        best = _sharpness(profile, sigma)
        for angle in np.linspace(-max_slant_deg, max_slant_deg, 17):
            if abs(angle) < 1e-9:
                continue
            _d, candidate = _sample(image, a, b, width, step, angle)
            score = _sharpness(candidate, sigma)
            if score > best * 1.02:                 # only slant for a real gain
                best, profile, slant = score, candidate, float(angle)
    return dist, profile, slant


def swath_profile_tiled(read_rect: Callable[[int, int, int, int], np.ndarray], a, b, *,
                        width: float, step: float = 1.0, image_size: tuple[int, int],
                        max_pixels: int = 4_000_000, slant_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """:func:`swath_profile` read piece by piece, for a capture too big to hold.

    *read_rect(x, y, w, h)* returns that rectangle of the image (full
    resolution, one band). The section is cut into stretches whose bounding
    boxes stay under *max_pixels*, each read with a margin and sampled in the
    capture's own coordinates, so the result matches sampling the whole image.
    *image_size* is ``(width, height)``.
    """
    (ax, ay), (bx, by) = a, b
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-9:
        raise ValueError("the section has no length")
    full_w, full_h = int(image_size[0]), int(image_size[1])
    reach = width / 2.0 * math.hypot(1.0, math.tan(math.radians(slant_deg))) + 2.0
    # Longest stretch whose box (diagonal at worst) stays under the cap.
    side = max(64.0, math.sqrt(max_pixels) - 2 * reach)
    pieces = max(1, int(math.ceil(length / side)))
    n_total = max(2, int(math.floor(length / step + 1e-9)) + 1)
    dist = np.linspace(0.0, length, n_total)
    profile = np.full(n_total, np.nan)
    tx, ty = (bx - ax) / length, (by - ay) / length
    edges = np.linspace(0.0, length, pieces + 1)
    for i in range(pieces):
        lo, hi = edges[i], edges[i + 1]
        idx = np.nonzero((dist >= lo - 1e-9) & ((dist < hi) if i < pieces - 1 else (dist <= hi + 1e-9)))[0]
        if idx.size == 0:
            continue
        p0 = (ax + tx * dist[idx[0]], ay + ty * dist[idx[0]])
        p1 = (ax + tx * dist[idx[-1]], ay + ty * dist[idx[-1]])
        x0 = int(max(0, math.floor(min(p0[0], p1[0]) - reach)))
        y0 = int(max(0, math.floor(min(p0[1], p1[1]) - reach)))
        x1 = int(min(full_w, math.ceil(max(p0[0], p1[0]) + reach) + 1))
        y1 = int(min(full_h, math.ceil(max(p0[1], p1[1]) + reach) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        tile = np.asarray(read_rect(x0, y0, x1 - x0, y1 - y0), dtype="float64")
        n_local = int(idx.size)
        p1_used = p1 if n_local > 1 else p0
        _d, values = _sample(tile, p0, p1_used, width, step, slant_deg, origin=(x0, y0), count=n_local)
        profile[idx] = values
    return dist, profile


# ---------------------------------------------------------------------- #
# Counting
# ---------------------------------------------------------------------- #
@dataclass
class LineCount:
    """The lines found along one section, in the sampled image's pixels."""

    kind: str                                         # "chain", "laid" or "ruling"
    distance: np.ndarray                              # (n,) along the section, px
    profile: np.ndarray                               # (n,) the swath profile as sampled
    signal: np.ndarray                                # (n,) detrended, lines positive
    positions: list[float] = field(default_factory=list)   # px along the section
    polarity: str = "light"                           # "light": lines brighter; "dark": darker
    period_px: float | None = None                    # the rhythm, if the profile repeats
    confidence: float = 0.0                           # how strongly it repeats (0-1)
    mm_per_px: float | None = None
    slant_deg: float = 0.0
    edited: bool = False                              # lines added or removed by hand
    notes: list[str] = field(default_factory=list)
    #: Per line, the share (0-1) of the strip's bands that show it on their own; empty for a plain profile.
    support: list[float] = field(default_factory=list)
    #: Per line, True if it was too faint alone and was accepted because it sits on the others' rhythm.
    weak: list[bool] = field(default_factory=list)
    bands: int = 0                                    # how many bands the strip was cut into (0: a profile)

    # -- numbers --------------------------------------------------------
    @property
    def count(self) -> int:
        return len(self.positions)

    @property
    def perpendicular(self) -> float:
        """cos(slant): what a distance along the section is worth square to the lines."""
        return math.cos(math.radians(self.slant_deg)) if self.slant_deg else 1.0

    @property
    def spacings(self) -> list[float]:
        """Distances between neighbouring lines, square to them (px)."""
        p, k = self.positions, self.perpendicular
        return [(b - a) * k for a, b in zip(p, p[1:])]

    @property
    def mean_spacing(self) -> float | None:
        s = self.spacings
        return float(np.mean(s)) if s else None

    @property
    def sd_spacing(self) -> float | None:
        s = self.spacings
        return float(np.std(s, ddof=1)) if len(s) >= 2 else None

    @property
    def lines_per_cm(self) -> float | None:
        mean = self.mean_spacing
        if not mean or not self.mm_per_px:
            return None
        return 10.0 / (mean * self.mm_per_px)

    def irregular(self) -> list[int]:
        """Indices of spacings that break the rhythm (a missed line, a fold, an extra mark)."""
        s = np.asarray(self.spacings, dtype="float64")
        if s.size < 3:
            return []
        typical = float(np.median(s))
        return [i for i, v in enumerate(s) if v > 1.45 * typical or v < 0.6 * typical]

    def length(self, px: float) -> str:
        return f"{px * self.mm_per_px:.2f} mm" if self.mm_per_px else f"{px:.1f} px"

    def summary(self) -> str:
        name = {"chain": "chain line", "laid": "laid line", "ruling": "ruled line"}[self.kind]
        n = self.count
        if n == 0:
            text = f"No {name}s found along this section."
        else:
            text = f"{n} {name}{'s' if n != 1 else ''}"
            mean = self.mean_spacing
            if mean is not None:
                text += f" · mean spacing {self.length(mean)}"
                sd = self.sd_spacing
                if sd is not None:
                    text += f" ± {self.length(sd)}"
            per_cm = self.lines_per_cm
            if per_cm is not None and self.kind != "chain":
                text += f" · {per_cm:.1f} per cm"
            text += f" · {'lighter' if self.polarity == 'light' else 'darker'} than the sheet"
            faint = sum(1 for w in self.weak if w)
            if faint:
                text += f" · {faint} faint, placed by the rhythm"
            if self.edited:
                text += " · edited by hand"
        return text

    def rows(self) -> list[dict]:
        """One row per line: number, position, spacing to the previous one."""
        out = []
        k = self.perpendicular
        for i, p in enumerate(self.positions):
            row = {"n": i + 1, "position_px": p,
                   "spacing_px": (p - self.positions[i - 1]) * k if i else None}
            if self.mm_per_px:
                row["position_mm"] = p * self.mm_per_px
                row["spacing_mm"] = row["spacing_px"] * self.mm_per_px if i else None
            if i < len(self.support) and math.isfinite(self.support[i]):
                row["support"] = self.support[i]
            if i < len(self.weak):
                row["faint"] = bool(self.weak[i])
            out.append(row)
        return out


def _spacing_range(kind: str, mm_per_px: float | None, length_px: float,
                   spacing_mm: tuple[float, float] | None = None) -> tuple[float, float]:
    if mm_per_px:
        lo, hi = spacing_mm if spacing_mm else SPACING_MM[kind]
        return lo / mm_per_px, hi / mm_per_px
    if kind == "laid":
        return _FALLBACK_LAID_PX
    f_lo, f_hi = _FALLBACK_FRACTION[kind]
    return max(6.0, length_px * f_lo), max(12.0, length_px * f_hi)


def _rhythm(signal: np.ndarray, low: float, high: float) -> tuple[float | None, float]:
    """Repeat distance of *signal* within [low, high] samples by autocorrelation, and its strength."""
    s = signal - signal.mean()
    energy = float(s @ s)
    if energy <= 0 or s.size < 3 * max(low, 1.0):
        return None, 0.0
    n = s.size
    # FFT autocorrelation, unbiased so a long lag is not penalised for overlapping less.
    size = 1 << int(math.ceil(math.log2(2 * n)))
    spectrum = np.fft.rfft(s, size)
    full = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n]
    full = full / energy * n / np.maximum(n - np.arange(n), 1)
    lo, hi = int(max(1, math.floor(low))), int(min(math.ceil(high), n // 2))
    if hi <= lo + 1:
        return None, 0.0
    window = full[lo:hi + 1]
    peaks = [i for i in range(1, window.size - 1) if window[i] >= window[i - 1] and window[i] >= window[i + 1]]
    if not peaks:
        return None, 0.0
    best = max(peaks, key=lambda i: window[i])
    value = float(window[best])
    # Prefer the fundamental: the shortest repeat the best one is a whole multiple of. A profile that
    # repeats every P also repeats every 2P, 3P..., and the unbiased estimate is noisiest -- so often
    # highest -- at the longest of them; taking that for the spacing would call four lines in five extra.
    for i in peaks:
        lag = lo + i
        multiple = (lo + best) / lag
        if (lag < (lo + best) * 0.6 and window[i] >= 0.5 * value and window[i] > 0.1
                and abs(multiple - round(multiple)) < 0.12):
            best, value = i, float(window[i])
            break
    lag = lo + best
    # Parabolic refinement of the peak.
    if 0 < best < window.size - 1:
        y0, y1, y2 = window[best - 1], window[best], window[best + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            lag += 0.5 * (y0 - y2) / denom
    return float(lag), float(max(0.0, min(1.0, value)))


def _local_maxima(signal: np.ndarray) -> np.ndarray:
    s = signal
    if s.size < 3:
        return np.array([], dtype=int)
    return np.nonzero((s[1:-1] >= s[:-2]) & (s[1:-1] > s[2:]))[0] + 1


def _prominence(signal: np.ndarray, index: int, reach: int) -> float:
    lo, hi = max(0, index - reach), min(signal.size, index + reach + 1)
    left = float(np.min(signal[lo:index + 1]))
    right = float(np.min(signal[index:hi]))
    return float(signal[index] - max(left, right))


def _suppress(candidates: list[int], score: np.ndarray, min_sep: float) -> list[int]:
    kept: list[int] = []
    for i in sorted(candidates, key=lambda i: -score[i]):
        if all(abs(i - k) >= min_sep for k in kept):
            kept.append(i)
    return sorted(kept)


def _refine(signal: np.ndarray, i: int) -> float:
    if 0 < i < signal.size - 1:
        y0, y1, y2 = signal[i - 1], signal[i], signal[i + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            return i + float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return float(i)


def _skew(values: np.ndarray) -> float:
    v = values - values.mean()
    sd = v.std()
    return float(np.mean(v ** 3) / sd ** 3) if sd > 0 else 0.0


def count_lines(distance: np.ndarray, profile: np.ndarray, *, kind: str = "auto",
                mm_per_px: float | None = None, polarity: str = "auto",
                sensitivity: float = 1.0, reference: np.ndarray | None = None,
                slant_deg: float = 0.0) -> LineCount:
    """Find, number and measure the lines a section crosses.

    *distance*, *profile*: a section's swath profile (:func:`swath_profile`),
    distance evenly spaced in pixels. *kind*: ``"chain"``, ``"laid"``,
    ``"ruling"`` or ``"auto"`` (laid if the profile has a clear laid-line
    rhythm, else chain). *mm_per_px*: the sampled image's pixel size, for the
    plausible spacings; without it they are guessed from the section's
    length. *polarity*: ``"light"``, ``"dark"`` or ``"auto"``.
    *sensitivity*: above 1 finds fainter lines, below 1 only clear ones.
    *reference*: the same section's profile in reflected light: the page's
    writing and stains, divided out so they are not counted.
    """
    distance = np.asarray(distance, dtype="float64")
    raw = _fill_nan(profile)
    notes: list[str] = []
    n = raw.size
    if n < 8:
        return LineCount(kind if kind != "auto" else "chain", distance, raw, np.zeros(n),
                         mm_per_px=mm_per_px, notes=["The section is too short to count anything."])
    step = float(distance[1] - distance[0]) if n > 1 else 1.0
    length = float(distance[-1] - distance[0])
    values = raw
    if reference is not None:
        ref = _fill_nan(reference)
        trend = _gauss1d(ref, max(4.0, n / 20))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(trend) > 1e-12, ref / trend, 1.0)
        values = values / np.clip(ratio, 0.2, 5.0)

    if kind == "auto":
        lo, hi = _spacing_range("laid", mm_per_px, length)
        laid_period, laid_conf = _rhythm(_bandpass(values, lo / step, hi / step), lo / step, hi / step)
        kind = "laid" if laid_period and laid_conf >= _LAID_AUTO_CONFIDENCE else "chain"
        notes.append(f"Counted as {KINDS[kind].lower()}: "
                     + ("the profile repeats at laid-line spacing." if kind == "laid"
                        else "no laid-line rhythm along this section."))
    if kind not in SPACING_MM:
        raise ValueError(f"unknown kind of line {kind!r}")

    low_px, high_px = _spacing_range(kind, mm_per_px, length)
    low, high = low_px / step, high_px / step         # in samples
    signal = _bandpass(values, low, high, kind=kind)

    # Polarity: sparse lines skew the detrended profile towards themselves.
    if polarity == "auto":
        skew = _skew(signal)
        polarity = "light" if skew >= 0 else "dark"
        if kind == "laid":
            notes.append("Laid lines are nearly even stripes, so light or dark gives the same count.")
    if polarity == "dark":
        signal = -signal

    period, confidence = _rhythm(signal, low, min(high, n / 2))
    sensitivity = float(np.clip(sensitivity, 0.2, 5.0))
    if kind == "laid":
        positions = _pick_laid(signal, period, low, sensitivity)
    else:
        # A clear rhythm says how far apart the lines really are: closer
        # candidates are fibres and specks, whatever the plausible range allows.
        separation = max(low, 0.6 * period) if period and confidence >= 0.35 else low
        positions = _pick_sparse(signal, separation, sensitivity)
        if len(positions) < 2 and period and confidence >= 0.35:
            positions = _comb(signal, period, floor=1.5 * _robust_spread(signal) / sensitivity)
            if positions:
                notes.append("Lines placed from the rhythm of the whole profile, not from individually "
                             "sharp lines.")
    positions = [float(distance[0] + p * step) for p in positions]

    if mm_per_px and kind == "laid" and low_px < 4.0:
        notes.append(f"Laid lines here are only {low_px:.1f}-{high_px:.1f} px apart: count at full resolution.")
    if not mm_per_px:
        notes.append("Resolution unknown: spacings are in pixels and the plausible range is a guess. "
                     "Give the capture's pixels per inch for millimetres.")
    if kind == "chain" and length < 2 * low_px:
        notes.append("The section spans less than two chain-line spacings; draw it longer.")
    result = LineCount(kind, distance, raw, signal, positions, polarity,
                       period * step if period else None, confidence, mm_per_px, slant_deg, notes=notes)
    bad = result.irregular()
    if bad:
        result.notes.append(f"{len(bad)} spacing{'s' if len(bad) != 1 else ''} break the rhythm "
                            "(a missed or extra line, a fold, the edge): check them on the chart.")
    return result


def _bandpass(values: np.ndarray, low: float, high: float, *, kind: str = "laid") -> np.ndarray:
    """Keep detail at line scale: lose the lighting trend and the pixel noise."""
    trend = _gauss1d(values, max(2.0, high * (1.0 if kind == "laid" else 0.75)))
    med_level = float(np.median(np.abs(trend))) or 1.0
    local_level = np.maximum(_gauss1d(np.abs(values), max(4.0, high)), med_level * 0.25)
    detail = (values - trend) / np.maximum(local_level, 1e-9)   # local relative contrast across the sheet
    smooth = max(0.5, low / (6.0 if kind == "laid" else 32.0))
    return _gauss1d(detail, smooth)


def _pick_sparse(signal: np.ndarray, low: float, sensitivity: float) -> list[float]:
    """Pronounced peaks at least ~0.6 x the smallest spacing apart (chain lines, ruling)."""
    centre = float(np.median(signal))
    spread = _robust_spread(signal)
    threshold = centre + (2.9 / sensitivity) * spread
    reach = max(2, int(round(low * 0.5)))
    candidates = [int(i) for i in _local_maxima(signal)
                  if signal[i] >= threshold and _prominence(signal, int(i), reach) >= (1.95 / sensitivity) * spread]
    kept = _suppress(candidates, signal, 0.6 * low)
    return [_refine(signal, i) for i in kept]


def _pick_laid(signal: np.ndarray, period: float | None, low: float, sensitivity: float) -> list[float]:
    """Every crest of a close, nearly even pattern."""
    sep = 0.6 * (period or low)
    reach = max(1, int(round((period or low) * 0.5)))
    maxima = [int(i) for i in _local_maxima(signal)]
    if not maxima:
        return []
    prom = np.array([_prominence(signal, i, reach) for i in maxima])
    # Noise: what is left at the finest scale.
    noise = _robust_spread(np.diff(signal)) / math.sqrt(2.0)
    typical = float(np.median(np.sort(prom)[len(prom) // 2:])) if len(prom) else 0.0
    need = max((0.3 / sensitivity) * typical, (2.0 / sensitivity) * noise)
    candidates = [i for i, p in zip(maxima, prom) if p >= need]
    kept = _suppress(candidates, signal, sep)
    return [_refine(signal, i) for i in kept]


def _comb(signal: np.ndarray, period: float, *, min_ratio: float = 1.5, floor: float = 0.0) -> list[float]:
    """A comb of *period* slid to sit on the crests, kept only if it clearly fits.

    *floor*: how high the comb's teeth must sit on average, so a rhythm found
    in noise does not place lines on nothing.
    """
    step = max(1, int(round(period)))
    if signal.size < 2 * step:
        return []
    scores = np.array([signal[np.rint(np.arange(phase, signal.size - 1, period)).astype(int)].mean()
                       for phase in range(step)])
    best = int(np.argmax(scores))
    spread = float(np.std(scores)) or 1e-12
    if scores[best] <= max(0.0, floor) or (scores[best] - float(np.mean(scores))) < min_ratio * spread:
        return []
    return [float(p) for p in np.arange(best, signal.size - 1, period)]


def edit_lines(result: LineCount, *, add: float | None = None, remove: float | None = None,
               snap: bool = True) -> LineCount:
    """Add a line near *add* or remove the one nearest *remove* (px along the section).

    An added line snaps to the strongest crest of the signal within a third of
    the typical spacing, so a rough click lands on the line itself.
    """
    positions = list(result.positions)
    count = len(positions)
    support = list(result.support) if len(result.support) == count else [math.nan] * count
    weak = list(result.weak) if len(result.weak) == count else [False] * count
    step = float(result.distance[1] - result.distance[0]) if result.distance.size > 1 else 1.0
    origin = float(result.distance[0])
    if remove is not None and positions:
        nearest = min(range(len(positions)), key=lambda i: abs(positions[i] - remove))
        for values in (positions, support, weak):
            values.pop(nearest)
    if add is not None:
        where = float(add)
        if snap and result.signal.size:
            spacing = result.mean_spacing or result.period_px or 10 * step
            reach = max(1, int(round(spacing / 3.0 / step)))
            centre = int(round((where - origin) / step))
            lo, hi = max(0, centre - reach), min(result.signal.size, centre + reach + 1)
            if hi > lo:
                i = lo + int(np.argmax(result.signal[lo:hi]))
                where = origin + _refine(result.signal, i) * step
        positions.append(where)
        support.append(math.nan)                      # placed by hand: no measure of its own
        weak.append(False)
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    keep_lists = bool(result.support or result.weak)
    out = LineCount(result.kind, result.distance, result.profile, result.signal,
                    [positions[i] for i in order],
                    result.polarity, result.period_px, result.confidence, result.mm_per_px,
                    result.slant_deg, edited=True, notes=list(result.notes),
                    support=[support[i] for i in order] if keep_lists else [],
                    weak=[weak[i] for i in order] if keep_lists else [],
                    bands=result.bands)
    return out


# ---------------------------------------------------------------------- #
# The strip as a picture: lines that run on, against ink and cloud that do not
# ---------------------------------------------------------------------- #
#: Most samples kept across a strip. Lines run on across it, so a wide strip is thinned, not lengthened.
MAX_ACROSS = 128
#: Swath worth using for each kind, mm: wide enough to outvote writing and pulp, narrow enough to stay straight.
_SWATH_MM = {"auto": 15.0, "chain": 20.0, "laid": 6.0, "ruling": 12.0}


#: Line half-widths (Gaussian sigma) the strip is filtered at for chain lines and ruling: mm, or px unscaled.
_LINE_SIGMA_MM = (0.22, 0.4, 0.8)
_LINE_SIGMA_PX = (1.2, 2.2, 4.0)


def suggest_swath_mm(kind: str) -> float:
    """A swath width (mm) that suits *kind* on a real sheet."""
    return _SWATH_MM.get(kind, _SWATH_MM["auto"])


def _strip_rows(width: float, step: float, max_across: int) -> int:
    return int(max(1, min(max_across, int(round(width / max(step, 0.5))) + 1)))


def swath_strip(image: np.ndarray, a, b, *, width: float = 8.0, step: float = 1.0,
                max_across: int = MAX_ACROSS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The strip *width* pixels wide centred on A-B, as a picture.

    Returns ``(distance, offsets, strip)``: distance along the section (n,),
    the offsets across it (m,), both in pixels, and ``strip[n, m]`` -- NaN
    outside the image. For :func:`count_lines_strip`.
    """
    image = np.asarray(image, dtype="float64")
    if image.ndim != 2:
        raise ValueError("the image must be one band (2-D)")
    return _sample_strip(image, a, b, width, step, rows=_strip_rows(width, step, max_across))


def swath_strip_tiled(read_rect: Callable[[int, int, int, int], np.ndarray], a, b, *,
                      width: float, step: float = 1.0, image_size: tuple[int, int],
                      max_pixels: int = 4_000_000, max_across: int = MAX_ACROSS,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:func:`swath_strip` read piece by piece, for a capture too big to hold (see :func:`swath_profile_tiled`)."""
    (ax, ay), (bx, by) = a, b
    length = math.hypot(bx - ax, by - ay)
    if length < 1e-9:
        raise ValueError("the section has no length")
    full_w, full_h = int(image_size[0]), int(image_size[1])
    reach = width / 2.0 + 2.0
    side = max(64.0, math.sqrt(max_pixels) - 2 * reach)
    pieces = max(1, int(math.ceil(length / side)))
    n_total = max(2, int(math.floor(length / step + 1e-9)) + 1)
    rows = _strip_rows(width, step, max_across)
    dist = np.linspace(0.0, length, n_total)
    offsets = _across_offsets(width, step, rows)
    strip = np.full((n_total, offsets.size), np.nan)
    tx, ty = (bx - ax) / length, (by - ay) / length
    edges = np.linspace(0.0, length, pieces + 1)
    for i in range(pieces):
        lo, hi = edges[i], edges[i + 1]
        idx = np.nonzero((dist >= lo - 1e-9) & ((dist < hi) if i < pieces - 1 else (dist <= hi + 1e-9)))[0]
        if idx.size == 0:
            continue
        p0 = (ax + tx * dist[idx[0]], ay + ty * dist[idx[0]])
        p1 = (ax + tx * dist[idx[-1]], ay + ty * dist[idx[-1]])
        x0 = int(max(0, math.floor(min(p0[0], p1[0]) - reach)))
        y0 = int(max(0, math.floor(min(p0[1], p1[1]) - reach)))
        x1 = int(min(full_w, math.ceil(max(p0[0], p1[0]) + reach) + 1))
        y1 = int(min(full_h, math.ceil(max(p0[1], p1[1]) + reach) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        tile = np.asarray(read_rect(x0, y0, x1 - x0, y1 - y0), dtype="float64")
        if idx.size > 1:
            _d, _o, values = _sample_strip(tile, p0, p1, width, step, origin=(x0, y0), count=int(idx.size),
                                           rows=rows)
        else:                                             # one sample: a direction is still needed
            _d, _o, values = _sample_strip(tile, p0, (p0[0] + tx, p0[1] + ty), width, step,
                                           origin=(x0, y0), count=2, rows=rows)
            values = values[:1]
        strip[idx] = values
    return dist, offsets, strip


def _gauss_axis0(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur down the columns of a 2-D array (along the section), edges mirrored."""
    n = values.shape[0]
    if sigma <= 0.3 or n < 3:
        return values.copy()
    radius = int(min(math.ceil(3.0 * sigma), n - 1))
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="reflect")
    size = padded.shape[0]
    freq = np.fft.rfftfreq(size)
    transfer = np.exp(-2.0 * (math.pi * sigma * freq) ** 2)
    out = np.fft.irfft(np.fft.rfft(padded, axis=0) * transfer[:, None], size, axis=0)
    return out[radius:radius + n]


def _masked_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Blur that ignores masked samples (normalised convolution): the blur, and how much it stood on."""
    den = _gauss_axis0(weight, sigma)
    num = _gauss_axis0(values * weight, sigma)
    return num / np.maximum(den, 1e-6), den


def _strip_detail(values: np.ndarray, valid: np.ndarray, trend_sigma: float, smooth: float) -> np.ndarray:
    """Every row of the strip filtered to line scale, as local relative contrast; NaN where unknown."""
    weight = valid.astype("float64")
    filled = np.where(valid, values, 0.0)
    # A straight ramp (an uneven light box) is taken out first, so mirrored edges do not bend into a line.
    n = values.shape[0]
    x = np.arange(n, dtype="float64")[:, None]
    sw = np.maximum(weight.sum(axis=0), 1.0)
    mx = (weight * x).sum(axis=0) / sw
    my = (weight * filled).sum(axis=0) / sw
    var = (weight * (x - mx) ** 2).sum(axis=0)
    slope = np.where(var > 0, (weight * (x - mx) * (filled - my)).sum(axis=0) / np.maximum(var, 1e-12), 0.0)
    level = float(np.median(np.abs(my))) or 1.0
    flat = np.where(valid, filled - (my + slope * (x - mx)), 0.0)
    trend, _ = _masked_blur(flat, weight, trend_sigma)
    local = np.maximum(np.abs(my + slope * (x - mx) + trend), 0.25 * level)
    detail = np.where(valid, (flat - trend) / local, 0.0)
    smoothed, stood = _masked_blur(detail, weight, smooth)
    return np.where(stood >= 0.3, smoothed, np.nan)


def _dark_marks(values: np.ndarray, valid: np.ndarray, depth: float = 0.85) -> np.ndarray:
    """Where the strip is much darker than the sheet around it: writing, not the sheet's own lines.

    A mould line changes the light by a few per cent; ink takes a quarter or more. Anything below
    *depth* x the local level is marked -- unless it runs on across most of the strip, which writing
    does not do and a ruled or drawn line does.
    """
    weight = valid.astype("float64")
    sigma = max(4.0, values.shape[0] / 20.0)
    level, _ = _masked_blur(np.where(valid, values, 0.0), weight, sigma)
    dark = valid & (values < depth * level)
    # Again with the marks left out, so heavy writing does not drag its own yardstick down.
    level, stood = _masked_blur(np.where(valid & ~dark, values, 0.0), (valid & ~dark).astype("float64"), sigma)
    dark = valid & (stood > 0.05) & (values < depth * level)
    across = dark.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    dark[across > 0.6] = False                            # a line right across the strip is a line
    grown = dark.copy()
    grown[1:] |= dark[:-1]
    grown[:-1] |= dark[1:]
    grown[:, 1:] |= dark[:, :-1]
    grown[:, :-1] |= dark[:, 1:]
    return grown


def _shear_strip(detail: np.ndarray, offsets: np.ndarray, shear: float) -> np.ndarray:
    """The strip with each across-position slid along the section so lines at that slant stand upright.

    *offsets* in samples. ``out[d, j] = detail[d + offsets[j] * shear, j]``.
    """
    if abs(shear) < 1e-9:
        return detail
    n = detail.shape[0]
    index = np.arange(n, dtype="float64")
    out = np.empty_like(detail)
    for j in range(detail.shape[1]):
        out[:, j] = np.interp(index + offsets[j] * shear, index, detail[:, j], left=np.nan, right=np.nan)
    return out


def _best_shear(detail: np.ndarray, offsets: np.ndarray, max_slant_deg: float) -> float:
    """The slant (degrees) at which the strip's rows add up to the sharpest profile; 0 unless clearly better."""
    spread = _robust_spread(detail[np.isfinite(detail)]) if np.isfinite(detail).any() else 1.0
    clipped = np.clip(np.nan_to_num(detail), -4.0 * spread, 4.0 * spread)
    if clipped.shape[0] > 4000:                           # the angle does not need every sample
        take = int(math.ceil(clipped.shape[0] / 4000))
        clipped = clipped[::take]
        offsets = offsets / take
    n = clipped.shape[0]
    index = np.arange(n, dtype="float64")
    margin = int(min(n // 4, math.ceil(np.abs(offsets).max() * math.tan(math.radians(max_slant_deg))) + 1))
    inner = slice(margin, n - margin)

    def score(angle: float) -> float:
        shear = math.tan(math.radians(angle))
        total = np.zeros(n)
        for j in range(clipped.shape[1]):
            total += np.interp(index + offsets[j] * shear, index, clipped[:, j])
        total /= clipped.shape[1]
        return float(np.mean(total[inner] ** 2))

    base = score(0.0)
    coarse = np.linspace(-max_slant_deg, max_slant_deg, 33)
    scores = [score(float(a)) for a in coarse]
    k = int(np.argmax(scores))
    width = float(coarse[1] - coarse[0])
    fine = np.linspace(coarse[k] - width, coarse[k] + width, 9)
    fine_scores = [score(float(a)) for a in fine]
    k2 = int(np.argmax(fine_scores))
    best, angle = fine_scores[k2], float(fine[k2])
    if 0 < k2 < len(fine) - 1:                            # parabolic refinement
        y0, y1, y2 = fine_scores[k2 - 1], fine_scores[k2], fine_scores[k2 + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-18:
            angle += float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)) * float(fine[1] - fine[0])
    if best <= base * 1.05 or abs(angle) < 0.05:
        return 0.0
    return float(np.clip(angle, -max_slant_deg, max_slant_deg))


def _midmean(values: np.ndarray) -> np.ndarray:
    """Across each row, the mean of the middle half (NaNs left out): writing and specks do not vote."""
    n, m = values.shape
    ordered = np.sort(values, axis=1)                     # NaNs last
    k = np.isfinite(ordered).sum(axis=1)
    cums = np.concatenate([np.zeros((n, 1)), np.cumsum(np.nan_to_num(ordered), axis=1)], axis=1)
    lo = np.floor(k * 0.25).astype(int)
    hi = np.maximum(np.ceil(k * 0.75).astype(int), lo + 1)
    hi = np.minimum(hi, np.maximum(k, 1))
    lo = np.minimum(lo, hi - 1)
    rows = np.arange(n)
    out = (cums[rows, hi] - cums[rows, lo]) / np.maximum(hi - lo, 1)
    return np.where(k >= max(2, m // 4), out, np.nan)


def _band_profiles(aligned: np.ndarray, bands: int) -> np.ndarray:
    """The strip cut into *bands* side by side, each one's own profile: ``[n, bands]``."""
    groups = np.array_split(np.arange(aligned.shape[1]), bands)
    return np.stack([_nanmean_rows(aligned[:, g]) for g in groups], axis=1)


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    counts = np.isfinite(values).sum(axis=1)
    return np.where(counts > 0, np.nansum(values, axis=1) / np.maximum(counts, 1), np.nan)


def _support(band_z: np.ndarray, index: int, height: float, reach: int) -> float:
    """Share of the bands that show a crest of their own at *index* (within *reach* samples)."""
    lo, hi = max(0, index - reach), min(band_z.shape[0], index + reach + 1)
    window = band_z[lo:hi]
    known = np.isfinite(window).any(axis=0)
    if not known.any():
        return 0.0
    with np.errstate(invalid="ignore"):
        peak = np.nanmax(np.where(np.isfinite(window), window, -np.inf), axis=0)
    return float(np.mean(peak[known] >= 0.4 * height))


def _lattice_period(strong: list[int], low: float, high: float, rhythm: float | None,
                    rhythm_conf: float) -> float | None:
    """The lines' own spacing: from the clear ones if they agree, else from the profile's rhythm."""
    if len(strong) >= 3:
        gaps = np.diff(np.asarray(sorted(strong), dtype="float64"))
        gaps = gaps[gaps >= 0.6 * low]
        if gaps.size >= 2:
            unit = float(np.min(gaps))
            if rhythm and rhythm_conf >= 0.3 and 0.6 * unit <= rhythm <= 1.6 * unit:
                unit = rhythm
            folded = gaps / np.maximum(np.rint(gaps / unit), 1.0)
            period = float(np.median(folded))
            scatter = float(np.median(np.abs(folded - period))) / max(period, 1e-9)
            if scatter <= 0.12 and low * 0.8 <= period <= high * 1.2:
                return period
    if rhythm and rhythm_conf >= 0.3:
        return float(rhythm)
    return None


def _pick_lattice(z: np.ndarray, band_z: np.ndarray | None, low: float, high: float,
                  sensitivity: float) -> tuple[list[int], list[bool], float | None, float]:
    """Clear crests, plus faint ones that sit on the clear ones' rhythm. Indices into *z*."""
    n = z.size
    strong_z, faint_z = 3.6 / sensitivity, 1.9 / sensitivity
    reach = max(2, int(round(low * 0.5)))
    near = max(2, int(round(low / 24.0)))
    period, confidence = _rhythm(z, low, min(high, n / 2))
    candidates = []
    for i in _local_maxima(z):
        i = int(i)
        if z[i] < faint_z or _prominence(z, i, reach) < 0.7 * faint_z:
            continue
        candidates.append(i)
    support = {i: (_support(band_z, i, float(z[i]), near) if band_z is not None else 1.0) for i in candidates}
    separation = max(low, 0.6 * period) if period and confidence >= 0.35 else low
    strong = _suppress([i for i in candidates if z[i] >= strong_z and support[i] >= 0.5], z, 0.6 * separation)
    spacing = _lattice_period(strong, low, high, period, confidence)
    accepted = {i: False for i in strong}                 # index -> faint?
    if spacing and len(strong) >= 2:
        pool = [i for i in candidates if i not in accepted and support[i] >= 0.4]
        tolerance = 0.15 * spacing

        def take(expect: float) -> int | None:
            close = [i for i in pool if abs(i - expect) <= tolerance
                     and all(abs(i - k) >= 0.6 * spacing for k in accepted)]
            return max(close, key=lambda i: z[i]) if close else None

        ordered = sorted(accepted)
        for p, q in zip(ordered, ordered[1:]):            # between two clear lines
            m = int(round((q - p) / spacing))
            if m >= 2 and abs((q - p) / m - spacing) <= 0.2 * spacing:
                for k in range(1, m):
                    found = take(p + k * (q - p) / m)
                    if found is not None:
                        accepted[found] = True
        for direction in (-1, 1):                         # and on past the first and the last
            edge = float(min(accepted) if direction < 0 else max(accepted))
            while True:
                found = take(edge + direction * spacing)
                if found is None:
                    break
                accepted[found] = True
                edge = float(found)
    order = sorted(accepted)
    return order, [accepted[i] for i in order], period, confidence


def _lattice_quality(z: np.ndarray, picks: list[int]) -> float:
    """How good a set of lines is: how far they stand out, and how evenly they are spaced."""
    if not picks:
        return 0.0
    height = float(np.sum(np.minimum(z[picks], 8.0)))
    if len(picks) >= 3:
        gaps = np.diff(np.asarray(picks, dtype="float64"))
        evenness = float(np.std(gaps) / max(np.mean(gaps), 1e-9))
    else:
        evenness = 0.5
    return height / (1.0 + 2.0 * evenness)


def count_lines_strip(distance: np.ndarray, offsets: np.ndarray, strip: np.ndarray, *, kind: str = "auto",
                      mm_per_px: float | None = None, polarity: str = "auto", sensitivity: float = 1.0,
                      reference: np.ndarray | None = None, square: bool = True,
                      max_slant_deg: float = 12.0, spacing_mm: tuple[float, float] | None = None,
                      line_width_mm: float | None = None, mask_marks: bool = True) -> LineCount:
    """Find, number and measure the lines a section crosses, from the strip as a picture.

    *distance*, *offsets*, *strip*: from :func:`swath_strip`. *reference*: the
    same strip in reflected light -- its dark places are writing, masked out
    (and its stains divided out) before anything is counted. *square*: find
    the slant at which the strip's rows line up (up to ``max_slant_deg`` either
    way) and measure spacings square to the lines. *spacing_mm*: ``(least,
    most)`` spacing to look for, in mm, instead of the kind's usual range
    (used only when the resolution is known). *line_width_mm*: filter chain
    lines and ruling at this one line width instead of trying three.
    *mask_marks*: leave writing and stains out (from *reference*, or the
    strip's own dark marks); off, every pixel counts. The rest as
    :func:`count_lines`, which is what this falls back to for a strip too
    narrow to treat as a picture.
    """
    distance = np.asarray(distance, dtype="float64")
    strip = np.asarray(strip, dtype="float64")
    if strip.ndim != 2 or strip.shape[0] != distance.size:
        raise ValueError("the strip must be [along, across] and match the distances")
    offsets = np.asarray(offsets, dtype="float64")
    n, m = strip.shape
    raw = _nanmean_rows(strip)
    if m < 6 or n < 8:
        ref_profile = _nanmean_rows(np.asarray(reference, dtype="float64")) if reference is not None else None
        return count_lines(distance, raw, kind=kind, mm_per_px=mm_per_px, polarity=polarity,
                           sensitivity=sensitivity, reference=ref_profile)
    notes: list[str] = []
    step = float(distance[1] - distance[0])
    length = float(distance[-1] - distance[0])
    sensitivity = float(np.clip(sensitivity, 0.2, 5.0))
    max_slant_deg = float(np.clip(max_slant_deg, 0.0, 45.0))
    if spacing_mm is not None:
        lo_mm, hi_mm = sorted(float(v) for v in spacing_mm)
        spacing_mm = (max(lo_mm, 0.05), max(hi_mm, lo_mm * 1.2, 0.1)) if lo_mm > 0 else None
    valid = np.isfinite(strip)
    values = np.where(valid, strip, 0.0)

    if reference is not None:
        ref = np.asarray(reference, dtype="float64")
        if ref.shape != strip.shape:
            raise ValueError("the reflected-light strip must match the strip")
        known = np.isfinite(ref)
        fill = float(np.median(ref[known])) if known.any() else 1.0
        ref = np.where(known, ref, fill)
        trend = _gauss_axis0(ref, max(4.0, n / 20))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(np.abs(trend) > 1e-12, ref / trend, 1.0)
        values = values / np.clip(ratio, 0.2, 5.0)
        ink = ratio < 0.85
        grown = ink.copy()                                # a stroke's soft edge is still ink
        grown[1:] |= ink[:-1]
        grown[:-1] |= ink[1:]
        grown[:, 1:] |= ink[:, :-1]
        grown[:, :-1] |= ink[:, 1:]
        masked = float(np.mean(grown[valid])) if valid.any() else 0.0
        if mask_marks and masked < 0.7:                   # a page that is nearly all ink leaves nothing to mask
            valid = valid & ~grown
            if masked >= 0.02:
                notes.append(f"Writing masked from reflected light ({masked * 100:.0f}% of the strip).")
    else:
        dark = _dark_marks(values, valid) if mask_marks else np.zeros_like(valid)
        masked = float(np.mean(dark[valid])) if valid.any() else 0.0
        if 0.0 < masked < 0.5:
            valid = valid & ~dark
            if masked >= 0.02:
                notes.append(f"Dark marks (writing, stains) left out: {masked * 100:.0f}% of the strip. With a "
                             "reflected-light band they are masked more exactly.")

    offsets_s = offsets / step                            # in samples

    def measure(detail: np.ndarray):
        """Line the strip's rows up, add them, and scale to the sheet's own texture."""
        slant_ = _best_shear(detail, offsets_s, max_slant_deg) if square else 0.0
        aligned_ = _shear_strip(detail, offsets_s, math.tan(math.radians(slant_)))
        stack_ = _fill_nan(_midmean(aligned_))
        stack_ = stack_ - float(np.median(stack_))
        spread_ = _robust_spread(stack_)
        return slant_, stack_ / spread_, _band_profiles(aligned_, n_bands) / spread_

    n_bands = int(np.clip(m // 6, 3, 8))
    if kind == "auto":
        low_px, high_px = _spacing_range("laid", mm_per_px, length)
        low, high = low_px / step, high_px / step
        laid_try = measure(_strip_detail(values, valid, max(2.0, high), max(0.5, low / 6.0)))
        laid_period, laid_conf = _rhythm(laid_try[1], low, high)
        kind = "laid" if laid_period and laid_conf >= _LAID_AUTO_CONFIDENCE else "chain"
        notes.append(f"Counted as {KINDS[kind].lower()}: "
                     + ("the strip repeats at laid-line spacing." if kind == "laid"
                        else "no laid-line rhythm along this section."))
    if kind not in SPACING_MM:
        raise ValueError(f"unknown kind of line {kind!r}")
    if spacing_mm and not mm_per_px:
        notes.append("The spacing range you set needs the resolution (Calibrate ▸ Set Scale from a Ruler); "
                     "the usual range was searched.")
    elif spacing_mm:
        notes.append(f"Searched for spacings of {spacing_mm[0]:g}-{spacing_mm[1]:g} mm, as set.")
    low_px, high_px = _spacing_range(kind, mm_per_px, length, spacing_mm)
    low, high = low_px / step, high_px / step

    weak: list[bool] = []
    support: list[float] = []
    if kind == "laid":
        slant, z, band_z = measure(_strip_detail(values, valid, max(2.0, high), max(0.5, low / 6.0)))
        if polarity == "auto":
            polarity = "light" if _skew(z) >= 0 else "dark"
            notes.append("Laid lines are nearly even stripes, so light or dark gives the same count.")
        if polarity == "dark":
            z, band_z = -z, -band_z
        period, confidence = _rhythm(z, low, min(high, n / 2))
        picks = _pick_laid(z, period, low, sensitivity)
    else:
        # A chain or ruled line is under a millimetre or two wide whatever its spacing, and how wide is
        # not known beforehand. The strip is filtered at three line widths -- matched to the line, so
        # the pulp's cloud between lines is not kept as signal -- and the width whose lines stand out
        # most, and most evenly, is the one counted.
        if line_width_mm and mm_per_px:
            # A line's full width is about 2.5 sigma of the filter that matches it.
            widths = [max(0.8, (line_width_mm / 2.5) / (mm_per_px * step))]
        elif mm_per_px:
            widths = [max(0.8, w_mm / (mm_per_px * step)) for w_mm in _LINE_SIGMA_MM]
        else:
            widths = [max(0.8, w / step) for w in _LINE_SIGMA_PX]
        if line_width_mm and not mm_per_px:
            notes.append("The line width you set needs the resolution; three widths were tried.")
        widths = [w for w in dict.fromkeys(round(w, 3) for w in widths) if w <= low / 4.0] or [max(0.8, low / 8.0)]
        chosen = None
        for width_s in widths:
            slant_w, z_w, band_w = measure(_strip_detail(values, valid, max(4.0, 5.5 * width_s), width_s))
            for sign, name in ((1.0, "light"), (-1.0, "dark")):
                if polarity not in ("auto", name):
                    continue
                order, faint, period_w, conf_w = _pick_lattice(sign * z_w, sign * band_w, low, high, sensitivity)
                quality = _lattice_quality(sign * z_w, order)
                # The first tried stands unless another is clearly better: light before dark, fine before broad.
                if chosen is None or quality > chosen[0] * 1.15:
                    chosen = (quality, name, sign * z_w, sign * band_w, order, faint, period_w, conf_w, slant_w,
                              width_s)
        _q, polarity, z, band_z, order, weak, period, confidence, slant, width_s = chosen
        near = max(1, int(round(2.0 * width_s)))
        support = [_support(band_z, i, float(z[i]), near) for i in order]
        picks = [_refine(z, i) for i in order]
        if len(picks) < 2 and period and confidence >= 0.35:
            combed = _comb(z, period, floor=1.5 / sensitivity)
            if combed:
                picks, weak = combed, [True] * len(combed)
                support = [_support(band_z, int(round(p)), max(float(z[int(round(p))]), 1.0), near)
                           for p in combed]
                notes.append("Lines placed from the rhythm of the whole strip, not from individually "
                             "sharp lines.")
    positions = [float(distance[0] + p * step) for p in picks]

    if slant:
        notes.append(f"The lines run {abs(slant):.1f}° off square to the section; spacings are measured "
                     "square to the lines.")
    if mm_per_px and kind == "laid" and low_px < 4.0:
        notes.append(f"Laid lines here are only {low_px:.1f}-{high_px:.1f} px apart: count at full resolution.")
    if not mm_per_px:
        notes.append("Resolution unknown: spacings are in pixels and the plausible range is a guess. "
                     "Give the capture's pixels per inch for millimetres.")
    if kind == "chain" and length < 2 * low_px:
        notes.append("The section spans less than two chain-line spacings; draw it longer.")
    if support and positions:
        notes.append(f"Each line was checked in {n_bands} bands of the strip; the weakest shows in "
                     f"{min(support) * 100:.0f}% of them.")
    if kind != "laid" and not positions:
        notes.append("Nothing stands clear of the sheet's texture here. Try a wider swath (the lines add up "
                     "across it), a longer section, higher sensitivity, or a transmitted-light band.")
    result = LineCount(kind, distance, raw, z, positions, polarity,
                       period * step if period else None, confidence, mm_per_px, slant, notes=notes,
                       support=support, weak=list(weak) if weak else [False] * len(positions) if support else [],
                       bands=n_bands)
    bad = result.irregular()
    if bad:
        result.notes.append(f"{len(bad)} spacing{'s' if len(bad) != 1 else ''} break the rhythm "
                            "(a missed or extra line, a fold, the edge): check them on the chart.")
    return result


def survey_page(image: np.ndarray, *, kind: str = "chain", mm_per_px: float | None = None,
                reference: np.ndarray | None = None, polarity: str = "auto", sensitivity: float = 1.0,
                **options) -> tuple[tuple[float, float], tuple[float, float], float, LineCount] | None:
    """Lay the section for the reader: across the page each way, keeping whichever finds the lines.

    A section is tried left to right and top to bottom through the middle of *image*, each with a
    strip half the page wide -- as much of every line's length as the page offers, which is what
    makes faint lines countable -- and the one whose lines stand out most and most evenly wins.
    Returns ``(a, b, width_px, count)`` with *a*, *b* in image pixels, or ``None`` if neither way
    finds two lines. Meant for chain lines and ruling on the preview; laid lines need the
    capture's full resolution and a section drawn along a chain line. *options* go to
    :func:`count_lines_strip` (``spacing_mm``, ``line_width_mm``, ``max_slant_deg``, ``mask_marks``).
    """
    image = np.asarray(image, dtype="float64")
    if image.ndim != 2:
        raise ValueError("the image must be one band (2-D)")
    h, w = image.shape
    if h < 32 or w < 32:
        return None
    best = None
    for a, b, width in (((0.05 * w, h / 2.0), (0.95 * w, h / 2.0), 0.5 * h),
                        ((w / 2.0, 0.05 * h), (w / 2.0, 0.95 * h), 0.5 * w)):
        dist, offsets, strip = swath_strip(image, a, b, width=width)
        ref = swath_strip(reference, a, b, width=width)[2] if reference is not None else None
        found = count_lines_strip(dist, offsets, strip, kind=kind, mm_per_px=mm_per_px, polarity=polarity,
                                  sensitivity=sensitivity, reference=ref, **options)
        if found.count < 2:
            continue
        step = float(dist[1] - dist[0])
        picks = [int(np.clip(round((p - dist[0]) / step), 0, found.signal.size - 1)) for p in found.positions]
        quality = (_lattice_quality(found.signal, picks) if found.kind != "laid"
                   else found.confidence * found.count)
        if best is None or quality > best[0]:
            best = (quality, a, b, float(width), found)
    return None if best is None else best[1:]
