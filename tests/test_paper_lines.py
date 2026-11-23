"""Counting chain, laid and ruled lines along a section (core.paper_lines)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from helio.core.paper_lines import count_lines, edit_lines, swath_profile, swath_profile_tiled

DPI = 300.0
MM = 25.4 / DPI                       # mm per pixel
CHAIN_MM = 24.0
LAID_MM = 1.0


def _sheet(h=900, w=1400, *, chain_mm=CHAIN_MM, laid_mm=LAID_MM, tilt_deg=0.0, polarity=1.0,
           noise=0.01, chain_depth=0.08, laid_depth=0.03, first_chain_mm=10.0, seed=0, writing=False):
    """Transmitted-light laid paper: chain lines vertical, laid lines horizontal."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    t = math.radians(tilt_deg)
    xr = x * math.cos(t) + y * math.sin(t)           # across the chain lines
    yr = -x * math.sin(t) + y * math.cos(t)          # across the laid lines
    chain_px, laid_px = chain_mm / MM, laid_mm / MM
    first = first_chain_mm / MM
    d = np.mod(xr - first + chain_px / 2, chain_px) - chain_px / 2
    chain = np.exp(-0.5 * (d / (0.6 / MM / 2.355)) ** 2)            # 0.6 mm wide lines
    laid = 0.5 + 0.5 * np.cos(2 * np.pi * yr / laid_px)
    sheet = 1.0 + polarity * (chain_depth * chain + laid_depth * laid)
    sheet *= 1.0 + 0.15 * (x / w)                                   # uneven light box
    sheet += rng.normal(0, noise, sheet.shape)
    if writing:
        sheet[400:440, :] *= 0.5                                    # a line of text across
    expected = [first + k * chain_px for k in range(int(w / chain_px) + 2) if 0 <= first + k * chain_px < w]
    return sheet.astype("float32"), expected


def _across_chain(sheet, y=450.0, x0=20.0, x1=None):
    x1 = sheet.shape[1] - 20.0 if x1 is None else x1
    return (x0, y), (x1, y)


def test_chain_lines_are_counted_and_measured():
    sheet, expected = _sheet()
    a, b = _across_chain(sheet)
    dist, prof, _ = swath_profile(sheet, a, b, width=4 / MM)
    result = count_lines(dist, prof, kind="chain", mm_per_px=MM)
    inside = [e - a[0] for e in expected if a[0] + 5 < e < b[0] - 5]
    assert result.count == len(inside)
    assert np.allclose(result.positions, inside, atol=3.0)
    assert abs(result.mean_spacing * MM - CHAIN_MM) < 0.3
    assert result.polarity == "light"
    assert "chain line" in result.summary()


def test_dark_chain_lines_are_found_with_auto_polarity():
    sheet, expected = _sheet(polarity=-1.0)
    a, b = _across_chain(sheet)
    dist, prof, _ = swath_profile(sheet, a, b, width=4 / MM)
    result = count_lines(dist, prof, kind="chain", mm_per_px=MM)
    assert result.polarity == "dark"
    assert result.count == len([e for e in expected if a[0] + 5 < e < b[0] - 5])


def test_laid_lines_are_counted_along_a_chain_line():
    sheet, _ = _sheet(h=700, w=400)
    a, b = (200.0, 10.0), (200.0, 690.0)                 # vertical: across the horizontal laid lines
    dist, prof, _ = swath_profile(sheet, a, b, width=2 / MM)
    result = count_lines(dist, prof, kind="laid", mm_per_px=MM)
    span_mm = (b[1] - a[1]) * MM
    assert abs(result.count - span_mm / LAID_MM) <= 2
    assert abs(result.lines_per_cm - 10.0 / LAID_MM) < 0.3


def test_auto_tells_laid_from_chain():
    sheet, _ = _sheet(h=700, w=1400)
    d, p, _ = swath_profile(sheet, (200.0, 10.0), (200.0, 690.0), width=2 / MM)
    assert count_lines(d, p, kind="auto", mm_per_px=MM).kind == "laid"
    d, p, _ = swath_profile(sheet, (20.0, 450.0), (1380.0, 450.0), width=4 / MM)
    assert count_lines(d, p, kind="auto", mm_per_px=MM).kind == "chain"


def test_a_skewed_section_is_squared_to_the_lines():
    sheet, _ = _sheet(tilt_deg=5.0, noise=0.005)
    a, b = _across_chain(sheet)
    width = 25 / MM                                       # a wide swath smears tilted lines
    d, p, _ = swath_profile(sheet, a, b, width=width)
    plain = count_lines(d, p, kind="chain", mm_per_px=MM)
    d, p, slant = swath_profile(sheet, a, b, width=width, square=True, scale_hint=0.6 / MM)
    squared = count_lines(d, p, kind="chain", mm_per_px=MM)
    assert abs(abs(slant) - 5.0) <= 1.5
    assert squared.signal.max() > plain.signal.max()
    assert squared.count >= plain.count


def test_writing_is_not_counted_when_divided_by_reflected_light():
    sheet, expected = _sheet(writing=True)
    reflected = np.ones_like(sheet)
    reflected[400:440, :] = 0.5
    a, b = (20.0, 420.0), (1380.0, 420.0)
    d, p, _ = swath_profile(sheet, a, b, width=4 / MM)
    d, r, _ = swath_profile(reflected, a, b, width=4 / MM)
    result = count_lines(d, p, kind="chain", mm_per_px=MM, reference=r)
    assert result.count == len([e for e in expected if a[0] + 5 < e < b[0] - 5])


def test_blank_paper_has_no_lines():
    rng = np.random.default_rng(3)
    sheet = (1.0 + rng.normal(0, 0.01, (600, 1200))).astype("float32")
    d, p, _ = swath_profile(sheet, (10.0, 300.0), (1190.0, 300.0), width=4 / MM)
    assert count_lines(d, p, kind="chain", mm_per_px=MM).count == 0


def test_tiled_sampling_matches_the_whole_image():
    sheet, _ = _sheet(h=600, w=900)
    a, b = (30.5, 50.25), (870.0, 560.0)
    d1, p1, _ = swath_profile(sheet, a, b, width=30.0)

    def read(x, y, w, h):
        return sheet[y:y + h, x:x + w]

    d2, p2 = swath_profile_tiled(read, a, b, width=30.0, image_size=(900, 600), max_pixels=40_000)
    assert np.allclose(d1, d2)
    assert np.allclose(p1, p2, atol=1e-5, equal_nan=True)


def test_lines_can_be_added_and_removed_by_hand():
    sheet, _ = _sheet()
    d, p, _ = swath_profile(sheet, *_across_chain(sheet), width=4 / MM)
    result = count_lines(d, p, kind="chain", mm_per_px=MM)
    removed = edit_lines(result, remove=result.positions[1])
    assert removed.count == result.count - 1 and removed.edited
    added = edit_lines(removed, add=result.positions[1] + 6.0)     # a rough click snaps to the line
    assert added.count == result.count
    assert abs(added.positions[1] - result.positions[1]) < 2.0


def test_without_resolution_it_still_counts_and_says_so():
    sheet, expected = _sheet()
    d, p, _ = swath_profile(sheet, *_across_chain(sheet), width=12.0)
    result = count_lines(d, p, kind="chain")
    assert result.count == len([e for e in expected if 25 < e < sheet.shape[1] - 25])
    assert any("Resolution unknown" in n for n in result.notes)
    assert result.lines_per_cm is None


def test_a_zero_length_section_is_refused():
    with pytest.raises(ValueError):
        swath_profile(np.ones((10, 10)), (3, 3), (3, 3))


# ---------------------------------------------------------------------- #
# The strip as a picture: a sheet as it really is
# ---------------------------------------------------------------------- #
from helio.core.paper_lines import count_lines_strip, survey_page, swath_strip, swath_strip_tiled  # noqa: E402

REAL_DPI = 150.0
REAL_MM = 25.4 / REAL_DPI


def _blur(noise, sigma):
    import cv2

    return cv2.GaussianBlur(noise, (0, 0), sigma)


def _real_sheet(h=900, w=1500, *, chain_mm=25.0, tilt_deg=1.5, chain_depth=0.05, cloud=0.04, noise=0.015,
                writing=True, polarity=1.0, seed=0, vertical=True):
    """A sheet as transmitted light shows it: chain lines a few per cent strong, a little off square and
    unevenly spaced, under cloudy pulp three scales deep, sensor noise, laid lines, and lines of writing
    ten times darker than any of it. Returns (transmitted, reflected, chain crossings of the row y = h/2)."""
    import cv2

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(float)
    t = math.radians(tilt_deg)
    xr = x * math.cos(t) + y * math.sin(t)
    yr = -x * math.sin(t) + y * math.cos(t)
    lines = [8.0 / REAL_MM]
    while lines[-1] < w * 1.2:
        lines.append(lines[-1] + (chain_mm + rng.uniform(-0.8, 0.8)) / REAL_MM)
    sigma = 0.7 / REAL_MM / 2.355
    chain = np.zeros_like(x)
    for p in lines:
        chain += np.exp(-0.5 * ((xr - p) / sigma) ** 2) - 0.3 * np.exp(-0.5 * ((xr - p) / (3 * sigma)) ** 2)
    sheet = 1.0 + polarity * chain_depth * chain + 0.012 * np.cos(2 * np.pi * yr / (1.1 / REAL_MM))
    for scale_mm, amount in ((1.5, cloud * 0.5), (4.0, cloud), (12.0, cloud)):
        blob = _blur(rng.normal(0, 1, (h, w)), scale_mm / REAL_MM)
        sheet += amount * blob / blob.std()
    sheet *= 1.0 + 0.25 * (x / w) - 0.15 * (y / h)
    reflected = np.ones((h, w))
    if writing:
        ink = np.zeros((h, w), np.uint8)
        for row in range(int(15 / REAL_MM), h - 10, int(7.0 / REAL_MM)):
            cx = int(12 / REAL_MM)
            while cx < w - int(12 / REAL_MM):
                wide, tall = int(rng.uniform(0.6, 2.2) / REAL_MM), int(rng.uniform(1.5, 3.0) / REAL_MM)
                if rng.random() < 0.85:
                    cv2.line(ink, (cx, row), (cx + int(rng.integers(-3, 4)), row - tall), 255, 2)
                    if rng.random() < 0.5:
                        cv2.line(ink, (cx, row - tall // 2), (cx + wide, row - tall // 2), 255, 2)
                cx += wide + int(rng.uniform(0.2, 1.0) / REAL_MM)
        ink = _blur(ink.astype(float) / 255.0, 0.7)
        sheet *= 1.0 - 0.55 * ink
        reflected *= 1.0 - 0.75 * ink
    sheet += rng.normal(0, noise, sheet.shape)
    crossings = [(p - (h / 2.0) * math.sin(t)) / math.cos(t) for p in lines]
    if not vertical:                                      # chain lines running across the page instead
        return sheet.T.copy().astype("float32"), reflected.T.copy().astype("float32"), crossings
    return sheet.astype("float32"), reflected.astype("float32"), crossings


def _matches(found, truth, tolerance=6.0):
    hits = sum(1 for t in truth if any(abs(f - t) <= tolerance for f in found))
    extra = sum(1 for f in found if not any(abs(f - t) <= tolerance for t in truth))
    return hits, extra


@pytest.fixture(scope="module")
def real_sheet():
    return _real_sheet()


def _section(sheet):
    h, w = sheet.shape
    return (30.0, h / 2.0), (w - 30.0, h / 2.0)


def test_faint_chain_lines_under_writing_are_counted_from_the_strip(real_sheet):
    """The case the single profile cannot do: 5 % lines under 4 % cloud and 55 % ink, 1.5 degrees off square."""
    sheet, reflected, crossings = real_sheet
    a, b = _section(sheet)
    truth = [c - a[0] for c in crossings if a[0] + 8 < c < b[0] - 8]
    ends = [c - a[0] for c in crossings if a[0] <= c <= b[0]]

    d, p, slant = swath_profile(sheet, a, b, width=4 / REAL_MM, square=True)
    old = count_lines(d, p, kind="chain", mm_per_px=REAL_MM, slant_deg=slant)
    assert _matches(old.positions, truth)[0] <= len(truth) // 3          # what it was

    for use_reflected in (False, True):
        d, o, strip = swath_strip(sheet, a, b, width=20 / REAL_MM)
        ref = swath_strip(reflected, a, b, width=20 / REAL_MM)[2] if use_reflected else None
        result = count_lines_strip(d, o, strip, kind="chain", mm_per_px=REAL_MM, reference=ref)
        hits, _ = _matches(result.positions, truth)
        assert hits == len(truth)
        assert _matches(result.positions, ends)[1] == 0                  # and nothing that is not a line
        assert result.polarity == "light"
        assert abs(result.mean_spacing * REAL_MM - 25.0) < 0.6
        assert abs(abs(result.slant_deg) - 1.5) < 0.7
        assert len(result.support) == result.count and min(result.support) >= 0.5
        assert result.bands >= 3


def test_the_strip_counts_without_a_resolution(real_sheet):
    sheet, _reflected, crossings = real_sheet
    a, b = _section(sheet)
    truth = [c - a[0] for c in crossings if a[0] + 8 < c < b[0] - 8]
    d, o, strip = swath_strip(sheet, a, b, width=60.0)
    result = count_lines_strip(d, o, strip, kind="chain")
    assert _matches(result.positions, truth)[0] == len(truth)
    assert abs(result.mean_spacing - 25.0 / REAL_MM) < 5.0               # the spacing, not five times it
    assert any("Resolution unknown" in n for n in result.notes)


def test_dark_chain_lines_are_told_from_dark_writing():
    sheet, _reflected, crossings = _real_sheet(polarity=-1.0, seed=2)
    a, b = _section(sheet)
    truth = [c - a[0] for c in crossings if a[0] + 8 < c < b[0] - 8]
    d, o, strip = swath_strip(sheet, a, b, width=20 / REAL_MM)
    result = count_lines_strip(d, o, strip, kind="chain", mm_per_px=REAL_MM)
    assert result.polarity == "dark"
    assert _matches(result.positions, truth)[0] == len(truth)
    assert abs(result.mean_spacing * REAL_MM - 25.0) < 0.6


def test_a_wider_strip_finds_fainter_lines():
    sheet, _reflected, crossings = _real_sheet(chain_depth=0.028, cloud=0.05, noise=0.02, seed=1, h=1100)
    a, b = _section(sheet)
    truth = [c - a[0] for c in crossings if a[0] + 8 < c < b[0] - 8]
    found = []
    for width_mm in (8.0, 70.0):
        d, o, strip = swath_strip(sheet, a, b, width=width_mm / REAL_MM)
        result = count_lines_strip(d, o, strip, kind="chain", mm_per_px=REAL_MM)
        found.append(_matches(result.positions, truth)[0])
    assert found[1] > found[0]
    assert found[1] >= len(truth) - 1


def test_a_sheet_without_chain_lines_is_not_given_any():
    extras = 0
    for seed in range(3):
        sheet, reflected, _ = _real_sheet(chain_depth=0.0, seed=seed)
        a, b = _section(sheet)
        d, o, strip = swath_strip(sheet, a, b, width=20 / REAL_MM)
        ref = swath_strip(reflected, a, b, width=20 / REAL_MM)[2]
        extras += count_lines_strip(d, o, strip, kind="chain", mm_per_px=REAL_MM, reference=ref).count
    assert extras <= 2                                    # wove paper, three pages: at most a stray mark


def test_spacings_are_square_to_the_lines_on_a_skewed_section():
    sheet, _ = _sheet(tilt_deg=10.0, noise=0.005, h=900, w=1400)
    a, b = _across_chain(sheet)
    d, o, strip = swath_strip(sheet, a, b, width=20 / MM)
    result = count_lines_strip(d, o, strip, kind="chain", mm_per_px=MM)
    assert abs(abs(result.slant_deg) - 10.0) < 0.6
    along = float(np.mean(np.diff(result.positions))) * MM
    assert abs(along - CHAIN_MM / math.cos(math.radians(10.0))) < 0.15    # the section crosses them further apart
    assert abs(result.mean_spacing * MM - CHAIN_MM) < 0.15                # the lines themselves are 24 mm apart
    assert abs(result.rows()[1]["spacing_mm"] - CHAIN_MM) < 0.4


def test_laid_lines_from_the_strip_and_auto_kind():
    sheet, _ = _sheet(h=700, w=1400, tilt_deg=2.0)
    a, b = (200.0, 10.0), (200.0, 690.0)
    d, o, strip = swath_strip(sheet, a, b, width=6 / MM)
    result = count_lines_strip(d, o, strip, kind="auto", mm_per_px=MM)
    assert result.kind == "laid"
    assert abs(result.lines_per_cm - 10.0 / LAID_MM) < 0.3
    d, o, strip = swath_strip(sheet, (20.0, 350.0), (1380.0, 350.0), width=20 / MM)
    assert count_lines_strip(d, o, strip, kind="auto", mm_per_px=MM).kind == "chain"


def test_ruled_ink_lines_are_lines_not_writing():
    """A dark line right across the strip is ruling; it must not be masked as ink."""
    rng = np.random.default_rng(5)
    page = 1.0 + rng.normal(0, 0.01, (600, 900))
    rows = [80 + 60 * k for k in range(8)]
    for r in rows:
        page[r - 1:r + 2, :] *= 0.6
    d, o, strip = swath_strip(page, (450.0, 20.0), (450.0, 580.0), width=120.0)
    result = count_lines_strip(d, o, strip, kind="ruling", mm_per_px=MM)
    assert result.polarity == "dark"
    assert result.count == len(rows)
    assert np.allclose(result.positions, [r - 20.0 for r in rows], atol=1.5)


def test_the_tiled_strip_matches_the_whole_image():
    sheet, _ = _sheet(h=600, w=900)
    a, b = (30.5, 50.25), (870.0, 560.0)
    d1, o1, s1 = swath_strip(sheet, a, b, width=30.0)

    def read(x, y, w, h):
        return sheet[y:y + h, x:x + w]

    d2, o2, s2 = swath_strip_tiled(read, a, b, width=30.0, image_size=(900, 600), max_pixels=40_000)
    assert np.allclose(d1, d2) and np.allclose(o1, o2)
    assert np.allclose(s1, s2, atol=1e-5, equal_nan=True)
    assert np.allclose(np.nanmean(s1, axis=1), swath_profile(sheet, a, b, width=30.0)[1], equal_nan=True)


def test_hand_edits_keep_each_lines_own_measures(real_sheet):
    sheet, _reflected, _ = real_sheet
    a, b = _section(sheet)
    d, o, strip = swath_strip(sheet, a, b, width=20 / REAL_MM)
    result = count_lines_strip(d, o, strip, kind="chain", mm_per_px=REAL_MM)
    removed = edit_lines(result, remove=result.positions[2])
    assert removed.count == result.count - 1 == len(removed.support) == len(removed.weak)
    added = edit_lines(removed, add=result.positions[2] + 4.0)
    assert added.count == result.count == len(added.support)
    assert abs(added.positions[2] - result.positions[2]) < 2.0
    assert math.isnan(added.support[2])                   # placed by hand: no measure of its own
    assert "support" not in added.rows()[2] and "support" in added.rows()[1]


@pytest.mark.parametrize("vertical", [True, False])
def test_the_page_survey_lays_the_section_across_the_lines(vertical):
    sheet, reflected, _ = _real_sheet(seed=3, vertical=vertical)
    found = survey_page(sheet, kind="chain", mm_per_px=REAL_MM, reference=reflected)
    assert found is not None
    a, b, width, result = found
    across_x = abs(b[0] - a[0]) > abs(b[1] - a[1])
    assert across_x == vertical                           # chain lines down the page: a section across it
    assert result.count >= 8
    assert abs(result.mean_spacing * REAL_MM - 25.0) < 0.6
    assert width > 100


def test_the_page_survey_refuses_a_sheet_without_lines():
    rng = np.random.default_rng(11)
    assert survey_page(1.0 + rng.normal(0, 0.01, (400, 500)), kind="chain", mm_per_px=REAL_MM) is None


def test_the_rhythm_is_the_spacing_not_a_multiple_of_it():
    from helio.core.paper_lines import _rhythm

    x = np.arange(1640.0)
    signal = sum(np.exp(-0.5 * ((x - p) / 2.0) ** 2) for p in np.arange(40.0, 1640.0, 148.0))
    signal = signal + np.random.default_rng(0).normal(0, 0.05, x.size)
    period, _confidence = _rhythm(signal, 41.0, 820.0)
    assert abs(period - 148.0) < 3.0


def test_strip_options_narrow_the_search():
    """A spacing range, one line width and no masking are honoured and said in the notes."""
    from helio.core.paper_lines import count_lines_strip, swath_strip

    rng = np.random.default_rng(3)
    h, w = 120, 600
    image = 1.0 + 0.01 * rng.standard_normal((h, w))
    xs = np.arange(w)
    for centre in range(40, w, 50):                   # lines every 50 px = 25 mm at 0.5 mm/px
        image += 0.05 * np.exp(-0.5 * ((xs - centre) / 1.5) ** 2)[None, :]
    dist, offsets, strip = swath_strip(image, (0, h / 2), (w - 1, h / 2), width=80)
    base = count_lines_strip(dist, offsets, strip, kind="chain", mm_per_px=0.5)
    narrowed = count_lines_strip(dist, offsets, strip, kind="chain", mm_per_px=0.5, spacing_mm=(22, 28),
                                 line_width_mm=1.0, max_slant_deg=3.0, mask_marks=False)
    assert narrowed.count == base.count == 12
    assert any("22-28 mm" in note for note in narrowed.notes)
    assert abs(narrowed.mean_spacing - 50) < 1.0
    no_scale = count_lines_strip(dist, offsets, strip, kind="chain", spacing_mm=(22, 28))
    assert any("needs the resolution" in note for note in no_scale.notes)
