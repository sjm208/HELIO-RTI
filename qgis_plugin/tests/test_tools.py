"""Tests for tools - measuring, line tracing and comparing integrations."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "helio_relief"))
import tools  # noqa: E402


def test_profile_finds_a_grooves_depth_and_width():
    yy, xx = np.mgrid[0:80, 0:200].astype(np.float64)
    z = 0.01 * xx + 5.0 - 2.0 * np.exp(-((xx - 100.0) / 4.0) ** 2)       # a 2-unit groove on a tilted sheet
    d, h = tools.profile(z, 10.0, 40.0, 190.0, 40.0, step=0.25)
    s = tools.profile_stats(d, h)
    assert s["depth"] == pytest.approx(2.0, rel=0.08)
    assert s["depth_at"] == pytest.approx(90.0, abs=1.0)                   # distance from the start
    assert s["width"] == pytest.approx(2 * 4.0 * math.sqrt(math.log(2)), rel=0.15)
    assert s["length"] == pytest.approx(180.0)


def test_area_stats_measure_a_pit():
    yy, xx = np.mgrid[0:120, 0:120].astype(np.float64)
    pit = 3.0 * np.exp(-((xx - 60) ** 2 + (yy - 60) ** 2) / (2 * 8.0 ** 2))
    z = 0.02 * xx - 0.01 * yy + 10.0 - pit
    s = tools.area_stats(z)
    assert s["hollow_volume"] == pytest.approx(pit.sum(), rel=0.1)
    assert s["relief"] > 2.0 and s["pixels"] == z.size


def grooves(h=220, w=260, seed=0):
    """Three straight grooves (0.8 deep, ~3 px wide) in a grainy sheet, and where they run."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = rng.normal(0, 0.03, (h, w))
    lines = [((30, 40), (230, 40)), ((40, 90), (40, 200)), ((80, 120), (220, 190))]
    for (x0, y0), (x1, y1) in lines:
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy)
        t = np.clip(((xx - x0) * dx + (yy - y0) * dy) / L ** 2, 0, 1)
        dist = np.hypot(xx - (x0 + t * dx), yy - (y0 + t * dy))
        z -= 0.8 * np.exp(-(dist / 1.5) ** 2)
    return z, lines


def test_traced_lines_follow_the_grooves():
    z, truth = grooves()
    lines, strength = tools.trace_lines(z, width_px=3.0, sensitivity=0.5, min_length_px=20)
    assert strength.shape == z.shape
    long = [l for l in lines if l["length_px"] > 60]
    assert 3 <= len(long) <= 6
    for (x0, y0), (x1, y1) in truth:
        L = math.hypot(x1 - x0, y1 - y0)
        near = 0.0
        for l in long:
            pts = np.asarray(l["points"])
            t = np.clip(((pts[:, 0] - x0) * (x1 - x0) + (pts[:, 1] - y0) * (y1 - y0)) / L ** 2, 0, 1)
            off = np.hypot(pts[:, 0] - (x0 + t * (x1 - x0)), pts[:, 1] - (y0 + t * (y1 - y0)))
            if np.median(off) < 1.5:
                near += l["length_px"]
        assert near > 0.8 * L                               # most of each groove is traced, on the groove


def test_tracing_ignores_height_offset_and_tilt():
    # depth maps sit at thousands of units on a ramp; the lines must come out the same as on the bare grooves
    z, _ = grooves()
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float64)
    lifted = z + 8000.0 + 2.0 * xx - 1.2 * yy
    a, _ = tools.trace_lines(z, width_px=3.0, sensitivity=0.5, min_length_px=20)
    b, _ = tools.trace_lines(lifted, width_px=3.0, sensitivity=0.5, min_length_px=20)
    assert sum(l["length_px"] for l in b) == pytest.approx(sum(l["length_px"] for l in a), rel=0.05)
    s, _ = tools.line_strength(lifted, 3.0)
    assert (s > 0).mean() > 0.2                          # not zeroed by a false curvature from the offset


def test_raised_lines_are_traced_like_grooves_upside_down():
    z, _ = grooves()
    incised, _ = tools.trace_lines(z, width_px=3.0, sensitivity=0.5, min_length_px=20)
    raised, _ = tools.trace_lines(-z, width_px=3.0, sensitivity=0.5, min_length_px=20, kind="raised")
    a, b = sum(l["length_px"] for l in incised), sum(l["length_px"] for l in raised)
    assert b == pytest.approx(a, rel=0.05)
    assert all(l["kind"] == "raised" for l in raised)


def test_bridging_joins_a_broken_groove():
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:80, 0:300].astype(np.float64)
    z = rng.normal(0, 0.02, xx.shape) - 0.8 * np.exp(-((yy - 40) / 1.5) ** 2)
    for gap in range(30, 280, 50):                              # worn patches break the groove every 50 px
        z[:, gap:gap + 6] = rng.normal(0, 0.02, (80, 6))
    broken, _ = tools.trace_lines(z, 3.0, 0.5, 10)
    joined, _ = tools.trace_lines(z, 3.0, 0.5, 10, bridge_px=12)
    assert len(broken) >= 4
    longest = max(l["length_px"] for l in joined)
    assert longest > 0.85 * 290 and len(joined) < len(broken)


def test_line_attributes_and_smoothing():
    z, truth = grooves()
    lines, _ = tools.trace_lines(z, 3.0, 0.5, 60, smooth=2)
    assert lines and all(0.4 < l["depth"] < 1.0 for l in lines)            # the grooves are 0.8 deep
    horizontal = [l for l in lines if abs(l["points"][0][1] - 40) < 3 and abs(l["points"][-1][1] - 40) < 3]
    assert horizontal and all(min(l["orientation"], 180 - l["orientation"]) < 5 for l in horizontal)
    vertical = [l for l in lines if abs(l["points"][0][0] - 40) < 3 and abs(l["points"][-1][0] - 40) < 3]
    assert vertical and all(abs(l["orientation"] - 90) < 5 for l in vertical)
    pts = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0)]
    sm = tools._chaikin(pts, 2)
    assert sm[0] == pts[0] and sm[-1] == pts[-1] and len(sm) > len(pts)


def test_thinning_and_chains():
    m = np.zeros((20, 30), bool)
    m[8:11, 3:25] = True                                    # a 3-px-wide bar
    thin = tools._thin(m)
    assert thin.sum() < 30 and (thin.sum(axis=0)[5:20] == 1).all()
    chains = tools._chains(thin)
    assert len(chains) == 1 and len(chains[0]) >= 15


def test_comparing_and_fusing_integrations():
    rng = np.random.default_rng(1)

    def blurred(sigma, amp):
        return tools.eng.gaussian_blur(rng.normal(0, 1, (320, 320)).astype(np.float32), sigma) * amp

    base = blurred(12, 60) + blurred(3, 4)                                          # form + tooling, in both
    a = base + 0.01 * rng.normal(0, 1, base.shape).astype(np.float32)              # smooth: little fine detail
    b = 5.0 * (base + blurred(1.5, 1.5) + 0.3 * rng.normal(0, 1, base.shape).astype(np.float32))  # detail, noise, 5x
    m = tools.compare_integrations(a, b)
    assert m["b"]["detail"] > m["a"]["detail"] and m["b"]["noise"] > m["a"]["noise"]
    assert m["agreement"] > 0.5
    choice, why = tools.recommend([m], ("Poisson", "Frankot-Chellappa"))
    assert choice in ("a", "fuse") and why
    g = tools.fuse_gain(a, b, 3.0)
    assert g == pytest.approx(0.2, rel=0.15)                                     # b's heights are 5x a's
    fused = tools.fuse_heights(a, b, 3.0, g)
    assert np.corrcoef(tools.eng.gaussian_blur(fused, 4).ravel(), tools.eng.gaussian_blur(a, 4).ravel())[0, 1] > 0.95
