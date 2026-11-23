"""Tests for detail - the spectrum, the noise model, band isolation and its displays."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "helio_relief"))
import detail  # noqa: E402
import relief_engine as eng  # noqa: E402


def power_law_surface(n=256, slope=3.0, seed=0):
    """Relief whose power rises with feature size as size^slope (natural-looking surfaces: 2-4)."""
    rng = np.random.default_rng(seed)
    f = np.sqrt(np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2)
    amp = np.where(f > 0, np.maximum(f, 1e-9) ** (-slope / 2.0), 0.0)
    spec = amp * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    z = np.real(np.fft.ifft2(spec))
    return (z / z.std()).astype(np.float32)


def test_spectrum_of_white_noise_is_flat_and_of_relief_rises():
    rng = np.random.default_rng(1)
    s, p = detail.radial_spectrum(rng.normal(size=(256, 256)))
    assert s[0] < 2.5 and s[-1] > 60
    assert p[s > 20].mean() == pytest.approx(p[s < 5].mean(), rel=0.35)
    s, p = detail.radial_spectrum(power_law_surface())
    slope = np.polyfit(np.log(s[(s > 6) & (s < 60)]), np.log(p[(s > 6) & (s < 60)]), 1)[0]
    assert slope == pytest.approx(3.0, abs=0.6)


def test_noise_crossover_moves_with_the_noise_level():
    rng = np.random.default_rng(2)
    z = power_law_surface()
    crossings = []
    for sigma in (0.0, 0.05, 0.2):
        m = detail.fit_noise(*detail.radial_spectrum(z + rng.normal(0, sigma, z.shape).astype(np.float32)))
        crossings.append(m["crossover"])
        assert m["detected"] == (sigma > 0)
        if sigma >= 0.2:
            assert m["noise_slope"] < 0.8                           # clear white noise is found flat
    assert crossings[0] < crossings[1] < crossings[2]
    assert crossings[2] > 2.5


def test_no_noise_is_reported_on_an_already_smooth_map():
    # integrated heights (like HELIO's) have almost no power at the finest sizes: nothing to call noise
    z = eng.gaussian_blur(power_law_surface(), 1.2)
    m = detail.fit_noise(*detail.radial_spectrum(z))
    assert not m["detected"]


def test_integrated_normal_noise_is_found_rising_with_size():
    rng = np.random.default_rng(5)
    z = power_law_surface(256, slope=4.5)
    n = rng.normal(0, 1, z.shape)
    f2 = np.fft.fftfreq(256)[:, None] ** 2 + np.fft.fftfreq(256)[None, :] ** 2
    integrated = np.real(np.fft.ifft2(np.fft.fft2(n) / np.sqrt(np.maximum(f2, 1e-4))))   # white slopes, integrated
    noisy = z + 0.02 * integrated.astype(np.float32) / integrated.std()
    m = detail.fit_noise(*detail.radial_spectrum(noisy))
    assert m["detected"] and m["noise_slope"] > 1.2


def test_band_multiplier_keeps_the_band_only():
    sizes = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 256.0])
    f = 1.0 / sizes
    m = detail.band_multiplier(f, np.zeros_like(f), 4.0, 32.0, soft=0.5)
    assert np.all(m[(sizes >= 6) & (sizes <= 20)] > 0.99)
    assert m[0] < 0.01 and m[-1] < 0.01
    assert np.all(detail.band_multiplier(f, np.zeros_like(f), 0.0, 0.0) == 1.0)


def test_isolate_separates_two_gratings():
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    fine, broad = np.sin(2 * np.pi * xx / 8.0), np.sin(2 * np.pi * yy / 64.0)
    band, _ = detail.isolate(fine + broad, dict(fine=4.0, broad=16.0, wiener=0.0))
    inner = (slice(32, -32), slice(32, -32))
    assert np.corrcoef(band[inner].ravel(), fine[inner].ravel())[0, 1] > 0.97
    assert abs(np.corrcoef(band[inner].ravel(), broad[inner].ravel())[0, 1]) < 0.1


def test_wiener_brings_the_band_closer_to_the_true_relief():
    rng = np.random.default_rng(3)
    truth = power_law_surface()
    noise = rng.normal(0, 0.1, truth.shape).astype(np.float32)
    d = dict(fine=0.0, broad=40.0)
    ref, _ = detail.isolate(truth, dict(d, wiener=0.0))
    noise_band, _ = detail.isolate(noise, dict(d, wiener=0.0))
    plain, _ = detail.isolate(truth + noise, dict(d, wiener=0.0))
    clean, removed = detail.isolate(truth + noise, dict(d, wiener=1.0))
    assert np.std(clean - ref) < 0.85 * np.std(plain - ref)
    assert np.corrcoef(removed.ravel(), noise_band.ravel())[0, 1] > 0.6       # what it removed is mostly the noise


def test_gate_keeps_strong_marks_and_damps_faint_texture():
    rng = np.random.default_rng(6)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    marks = -1.0 * np.exp(-(((xx % 64) - 32) / 2.5) ** 2)
    grain = 0.06 * eng.gaussian_blur(rng.normal(0, 1, marks.shape).astype(np.float32), 1.5) / 0.19
    d = dict(fine=2.0, broad=24.0)
    plain, _ = detail.isolate(marks + grain, dict(d, gate=0.0))
    gated, removed = detail.isolate(marks + grain, dict(d, gate=1.0))
    between = (slice(20, -20), slice(4, 20))
    assert gated[between].std() < 0.6 * plain[between].std()
    depth = lambda b: float(np.median(b[20:-20, 24:40].max(axis=1) - b[20:-20, 32]))
    assert depth(gated) > 0.8 * depth(plain)


def test_bands_add_with_their_gains():
    z = power_law_surface(128)
    one, _ = detail.isolate(z, dict(bands=[dict(on=True, fine=2.0, broad=8.0, gain=1.0)]))
    two, _ = detail.isolate(z, dict(bands=[dict(on=True, fine=16.0, broad=64.0, gain=1.0)]))
    both, _ = detail.isolate(z, dict(bands=[dict(on=True, fine=2.0, broad=8.0, gain=2.0),
                                             dict(on=True, fine=16.0, broad=64.0, gain=0.5),
                                             dict(on=False, fine=1.0, broad=0.0, gain=1.0)]))
    inner = (slice(24, -24), slice(24, -24))
    assert np.allclose(both[inner], (2.0 * one + 0.5 * two)[inner], atol=2e-3 * np.abs(both).max())
    r = detail.response(np.array([4.0, 11.0, 32.0]), dict(bands=[dict(on=True, fine=2.0, broad=8.0, gain=2.0),
                                                                  dict(on=True, fine=16.0, broad=64.0, gain=0.5)]))
    assert r[0] == pytest.approx(2.0, abs=0.01) and r[2] == pytest.approx(0.5, abs=0.01) and r[1] < 0.6


def test_every_display_renders():
    z = power_law_surface(160) + np.random.default_rng(7).normal(0, 0.05, (160, 160)).astype(np.float32)
    d = dict(fine=2.0, broad=32.0, wiener=1.0, gate=0.5)
    st = detail.detail_stats(z, d)
    band, removed = detail.isolate(z, d, 1.0, st)
    for mode in detail.DISPLAYS:
        img = detail.render(band, removed, dict(d, display=mode), {}, 1.0, 1.0, st)
        assert img.shape[:2] == z.shape and np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
        assert img.std() > 0.005, mode
        assert (img.ndim == 3) == (mode == "signed_colour")


def test_tiles_match_the_whole_band():
    z = power_law_surface(192)[:, :]
    z = np.hstack([z, z[:, ::-1]])                                  # 192 x 384
    d = dict(fine=2.0, broad=16.0, wiener=1.0, gate=0.7, clean=2.0)
    plane = eng.fit_plane(z)
    st = detail.detail_stats(z, d, plane=plane)
    whole, _ = detail.isolate(z, d, 1.0, st, plane)
    m, mid = detail.margin_px(d), 192
    left, _ = detail.isolate(z[:, :mid + m], d, 1.0, st, plane)
    right, _ = detail.isolate(z[:, mid - m:], d, 1.0, st, plane, origin=(mid - m, 0))
    stitched = np.hstack([left[:, :mid], right[:, m:]])
    inner = (slice(16, -16), slice(16, -16))                        # the seam, away from the image's own edges
    assert np.max(np.abs(stitched - whole)[inner]) < 0.02 * st["scale"]


def test_presets_resolve_at_the_crossover():
    model = dict(crossover=2.7, noise=1.0, noise_slope=0.0, slope=3.0, amp=1.0)
    assert detail.resolve_band(dict(fine=-1.0, broad=0.0), model) == (2.7, 0.0)
    assert detail.resolve_band(dict(fine=0.0, broad=-1.0), model) == (0.0, 2.7)
    for name, preset in detail.PRESETS.items():
        assert detail.active_bands(dict(fine=preset["fine"], broad=preset["broad"]), model), name
