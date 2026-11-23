"""Semi-calibrated strengths, the sparse-outlier solver and the metal mask
(helio.core.robust_ps), and their Rust twins (helio_core_rs, robust.rs).

On a synthetic matte sphere under eight known lights:
  * the lamp strengths are found again when the lamps are unequal, and the
    solve with them evened is as good as with equal lamps;
  * the sparse-outlier solver keeps its normals under a burnt-out highlight
    that throws plain least squares off by tens of degrees, and agrees with
    least squares on a clean matte surface;
  * the specular excess is near zero on matte and high on a gilded patch,
    and the metal mask covers the patch and nothing else;
  * the Rust kernels agree with the NumPy code;
  * the pipeline: Semi-calibrated, Sparse-Outlier LS and Metal-aware solve
    through helio.application.pipeline.solve, and record what they did.

    python tests/test_robust_ps.py
"""
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from helio.core import robust_ps as R                    # noqa: E402

RS = R.rs if (R.rs is not None and hasattr(R.rs, "sparse_ps")) else None
_results = []


def check(name, fn):
    try:
        fn()
        _results.append((name, None))
        print(f"  ok    {name}")
    except Exception as e:                                # noqa: BLE001
        _results.append((name, e))
        print(f"  FAIL  {name}: {e}")
        if not isinstance(e, AssertionError):
            traceback.print_exc()


class numpy_only:
    """Run the block with the NumPy code, whatever is built."""
    def __enter__(self):
        self.saved, R.rs = R.rs, None

    def __exit__(self, *a):
        R.rs = self.saved


def lights(n=8):
    out = []
    for k in range(n):
        az = k * 2 * np.pi / n
        el = 0.6 if k % 2 == 0 else 1.0
        out.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.array(out, np.float32)


def sphere(side=40):
    u = (np.arange(side) + 0.5) / side * 1.6 - 0.8
    U, V = np.meshgrid(u, u)
    Z = np.sqrt(np.maximum(1 - U ** 2 - V ** 2, 0.05))
    N = np.stack([U, V, Z], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return N.astype(np.float32)


def render(L, N, gain=None, albedo=180.0):
    s = np.maximum(L @ N.reshape(-1, 3).T, 0.0)
    g = np.ones(len(L)) if gain is None else np.asarray(gain)
    return (albedo * g[:, None] * s).astype(np.float32)


def angle(a, b):
    a = a.reshape(-1, 3)
    b = b.reshape(-1, 3)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


def lls(L, I, side):
    b = np.linalg.pinv(L) @ I
    return (b / np.maximum(np.linalg.norm(b, axis=0), 1e-12)).T.reshape(side, side, 3)


GAIN = np.array([1.0, 1.3, 0.8, 1.1, 0.9, 1.25, 1.0, 0.85])


def t_strengths_found():
    L, N = lights(), sphere()
    I = render(L, N, GAIN)
    r = R.light_strengths(I, L)
    assert r is not None, "no answer"
    want = GAIN / np.exp(np.mean(np.log(GAIN)))
    err = np.max(np.abs(np.array(r["strength"]) - want))
    assert err < 0.01, f"strengths off by {err:.4f}: {np.round(r['strength'], 3)} vs {np.round(want, 3)}"
    assert abs(r["spread"] - GAIN.max() / GAIN.min()) < 0.02, r["spread"]


def t_strengths_even_lamps():
    L, N = lights(), sphere()
    r = R.light_strengths(render(L, N), L)
    assert r is not None and r["spread"] < 1.01, r


def t_strengths_fix_the_solve():
    L, N = lights(), sphere()
    I = render(L, N, GAIN)
    before = np.median(angle(lls(L, I, 40), N))
    e = np.array(R.light_strengths(I, L)["strength"])
    after = np.median(angle(lls(L, I / e[:, None], 40), N))
    equal = np.median(angle(lls(L, render(L, N), 40), N))   # least squares' own error (rim shadows)
    assert after < equal + 0.05 and before > equal + 1.0, \
        f"median error {before:.2f}° -> {after:.2f}° (equal lamps {equal:.2f}°)"


def t_strengths_refuse():
    L = lights()
    assert R.light_strengths(np.zeros((8, 400), np.float32), L) is None
    assert R.light_strengths(np.ones((3, 400), np.float32), L[:3]) is None


def t_sparse_highlight():
    L, N = lights(), sphere(24)
    I = render(L, N)
    I[2] += 400.0                                        # a burning highlight in frame 3
    rng = np.random.default_rng(0)
    I = np.maximum(I + rng.normal(0, 2, I.shape), 0).astype(np.float32)
    plain = angle(lls(L, I, 24), N)
    sparse = angle(R.sparse_outlier_normals(L, I, 24, 24), N)
    assert np.percentile(sparse, 99) < 3.0 and np.median(plain) > 20.0, \
        f"sparse p99 {np.percentile(sparse, 99):.2f}°, plain median {np.median(plain):.1f}°"


def t_sparse_clean():
    L, N = lights(), sphere(24)
    out = R.sparse_outlier_normals(L, render(L, N), 24, 24)
    assert angle(out, N).max() < 0.1, angle(out, N).max()


def t_sparse_dark_pixels():
    L = lights()
    out = R.sparse_outlier_normals(L, np.zeros((8, 30), np.float32), 5, 6)
    assert out.shape == (5, 6, 3) and np.isfinite(out).all()


def gilded():
    """A matte sphere with a gilded band: a sharp mirror lobe on top of the matte."""
    L, N = lights(), sphere(48)
    I = render(L, N)
    flat = N.reshape(-1, 3)
    band = np.zeros((48, 48), bool)
    band[20:28, 6:42] = True
    view = np.array([0, 0, 1.0])
    for j, l in enumerate(L):
        h = (l + view) / np.linalg.norm(l + view)
        lobe = np.clip(flat @ h, 0, 1) ** 60
        I[j] += np.where(band.reshape(-1), 900.0 * lobe, 0.0)
    return L, N, I, band


def t_excess_and_mask():
    L, N, I, band = gilded()
    ex = R.specular_excess(L, N, I)
    assert np.median(ex[~band]) < 0.05, np.median(ex[~band])
    lit = ex[band] > R.excess_threshold(0.5)
    assert lit.mean() > 0.5, f"only {lit.mean():.0%} of the band seen"
    m = R.metal_mask(ex, 0.5)
    import cv2
    near = cv2.dilate(band.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)   # the feathered rim
    outside = (m[~near] > 0.5).sum() / max(1, (~near).sum())
    assert outside < 0.03, f"{outside:.1%} of the matte marked"
    assert R.metal_mask(np.zeros((10, 10), np.float32)).max() == 0.0


def t_sensitivity_monotone():
    L, N, I, _ = gilded()
    ex = R.specular_excess(L, N, I)
    a = R.metal_mask(ex, 0.2).sum()
    b = R.metal_mask(ex, 0.8).sum()
    assert b >= a, (a, b)


def t_flare_four_lamps():
    """Four lamps, as on a hand-lit cover: the fit takes a glint in, so only the flare can see it."""
    L = lights(8)[[0, 2, 5, 7]]
    N = np.tile(np.array([0, 0, 1.0], np.float32), (60, 60, 1))
    I = render(L, N, albedo=150.0)
    gold = np.zeros((60, 60), bool)
    gold[20:40, 20:40] = True
    I[1, gold.reshape(-1)] *= 2.6                              # the gold catches lamp 2
    ex = R.specular_excess(L, R.sparse_outlier_normals(L, I, 60, 60), I)
    fl, bright = R.flare(I, L, 60, 60)
    m = R.metal_mask(ex, 0.5, flare_map=fl, bright=bright)
    core = np.zeros_like(gold)
    core[24:36, 24:36] = True
    assert m[core].mean() > 0.9, f"gold marked {m[core].mean():.0%}"
    far = np.ones_like(gold)
    far[12:48, 12:48] = False
    assert m[far].max() == 0.0, "matte marked"


def t_flare_ignores_falloff_and_dark():
    """A lamp close to one edge (light falling off across the sheet) and dark cloth are not metal."""
    L = lights(8)
    N = np.tile(np.array([0, 0, 1.0], np.float32), (80, 80, 1))
    I = render(L, N, albedo=150.0).reshape(8, 80, 80)
    ramp = np.linspace(1.0, 2.2, 80)[None, :]
    I[0] *= ramp                                               # lamp 1 near the right edge
    I[:, :, :15] *= 0.03                                       # black velvet down the left
    I[3, :, :15] *= 3.0                                        # ...with a sheen of its pile
    I = I.reshape(8, -1)
    fl, bright = R.flare(I, L, 80, 80)
    m = R.metal_mask(np.zeros((80, 80), np.float32), 0.5, flare_map=fl, bright=bright)
    assert m.max() == 0.0, f"{(m > 0).mean():.1%} marked"


def t_blend():
    a = np.tile(np.array([0, 0, 1.0], np.float32), (4, 4, 1))
    b = np.tile(np.array([1.0, 0, 0], np.float32), (4, 4, 1))
    w = np.zeros((4, 4), np.float32)
    w[0, 0], w[1, 1] = 1.0, 0.5
    out = R.blend_normals(a.copy(), b, w)
    assert np.allclose(out[0, 0], [1, 0, 0]) and np.allclose(out[2, 2], [0, 0, 1])
    assert np.allclose(np.linalg.norm(out[1, 1]), 1.0) and out[1, 1, 0] > 0.6


def t_rust_parity():
    if RS is None:
        print("        (helio_core_rs not built with robust.rs: skipped)")
        return
    L, N, I, _ = gilded()
    I[3] += 250.0
    a = R.sparse_outlier_normals(L, I, 48, 48)
    with numpy_only():
        b = R.sparse_outlier_normals(L, I, 48, 48)
    d = angle(a, b)
    assert np.percentile(d, 99) < 0.05 and d.max() < 1.0, f"sparse: p99 {np.percentile(d, 99):.4f}°, max {d.max():.3f}°"
    ea = R.specular_excess(L, N, I)
    with numpy_only():
        eb = R.specular_excess(L, N, I)
    assert np.allclose(ea, eb, atol=1e-4), np.abs(ea - eb).max()
    J = render(L, sphere(), GAIN)
    sa = R.light_strengths(J, L)
    with numpy_only():
        sb = R.light_strengths(J, L)
    assert np.allclose(sa["strength"], sb["strength"], atol=1e-6), (sa, sb)
    assert sa["iters"] == sb["iters"]


def t_rust_bad_shapes():
    if RS is None:
        return
    L = lights()
    for call in (lambda: RS.sparse_ps(L[:4], np.zeros((3, 8), np.float32), np.zeros((8, 5), np.float32), 0.3, 5, 1e-4),
                 lambda: RS.specular_excess(L, np.zeros((4, 3), np.float32), np.zeros((8, 5), np.float32)),
                 lambda: RS.semi_cal_step(L.astype(np.float64), np.ones(7), np.zeros((8, 5), np.float32), 0.0, 1.0)):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("a wrong shape was accepted")


# ── the pipeline ───────────────────────────────────────────────────────

def frames_from(I, side):
    """[(rgb, grey, is_raw)] as the loader gives them, uint8."""
    out = []
    for row in I:
        g = np.clip(row.reshape(side, side), 0, 255).astype(np.uint8)
        out.append((np.repeat(g[..., None], 3, axis=2), g, False))
    return out


def solve(frames, L, **settings):
    """The frames are rendered in linear light: said so, as Automatic decodes camera files (2026-11-12)."""
    from helio.application.pipeline import SolveRequest, solve as run
    from helio.domain.settings import ProcessingSettings
    settings.setdefault("photo_tone", "linear")
    return run(SolveRequest(paths=[], lights=L, settings=ProcessingSettings(**settings), frames=frames))


def t_pipeline_semi_calibrated():
    L, N = lights(), sphere(64)
    I = render(L, N, GAIN, albedo=150.0)
    frames = frames_from(I, 64)
    plain = solve(frames, L)
    semi = solve(frames, L, semi_calibrated=True)
    assert semi.light_strength is not None and plain.light_strength is None
    e0 = np.median(angle(plain.normals_full, N))
    e1 = np.median(angle(semi.normals_full, N))
    assert e1 < e0 / 3, f"median {e0:.2f}° -> {e1:.2f}°"
    assert abs(semi.light_strength["spread"] - GAIN.max() / GAIN.min()) < 0.05, semi.light_strength


def t_pipeline_semi_off_with_flash():
    L, N = lights(), sphere(32)
    frames = frames_from(render(L, N, GAIN), 32)
    r = solve(frames, L, semi_calibrated=True, even_flash=True, flash_ref="0,0,8,8")
    # the grey reference wins when it can be read; either way the record says what was done
    assert (r.flash is None) or (r.light_strength is None)


def t_pipeline_sparse_solver():
    L, N = lights(), sphere(48)
    I = render(L, N, albedo=120.0)
    I[2] = np.minimum(I[2] + 200.0, 255.0)
    frames = frames_from(I, 48)
    a = solve(frames, L)
    b = solve(frames, L, normal_engine="Sparse-Outlier LS (RPCA)")
    ea, eb = np.median(angle(a.normals_full, N)), np.median(angle(b.normals_full, N))
    assert eb < ea / 3, f"median {ea:.2f}° -> {eb:.2f}°"


def t_pipeline_metal_aware():
    L, N, I, band = gilded()
    I = np.clip(I * 0.25, 0, 255)
    frames = frames_from(I, 48)
    a = solve(frames, L)
    b = solve(frames, L, metal_aware=True)
    assert b.metal is not None and b.metal["share"] > 0.0, b.metal
    assert b.metal_mask is not None and b.metal_mask[band].mean() / 255 > 0.3, b.metal_mask[band].mean() / 255
    ea, eb = np.median(angle(a.normals_full[band], N[band])), np.median(angle(b.normals_full[band], N[band]))
    assert eb < ea, f"gilded band median {ea:.2f}° -> {eb:.2f}°"
    # Steep matte flares too and may be marked; re-solved, it must come out no worse
    # (the sparse solver treats the rim's attached shadows better than least squares).
    off = ~band
    ma, mb = np.median(angle(a.normals_full[off], N[off])), np.median(angle(b.normals_full[off], N[off]))
    assert mb <= ma + 0.05, f"the matte got worse: {ma:.2f}° -> {mb:.2f}°"


def t_settings_round_trip():
    from helio.domain.settings import NORMAL_SOLVERS, ProcessingSettings
    from helio.domain.registry import NORMAL_SOLVERS as REG
    assert "Sparse-Outlier LS (RPCA)" in NORMAL_SOLVERS and NORMAL_SOLVERS == REG
    s = ProcessingSettings(semi_calibrated=True, metal_aware=True, metal_solver="IRLS (Metallic/Specular)",
                           metal_sensitivity=0.7, rpca_lambda=0.4)
    assert ProcessingSettings.from_dict(s.to_dict()) == s


def main():
    print("robust_ps:", "Rust and NumPy" if RS is not None else "NumPy only")
    for name, fn in [
        ("lamp strengths found again", t_strengths_found),
        ("equal lamps stay equal", t_strengths_even_lamps),
        ("evened strengths fix the solve", t_strengths_fix_the_solve),
        ("too little to go on: no answer", t_strengths_refuse),
        ("sparse-outlier solve keeps the shape under a highlight", t_sparse_highlight),
        ("sparse-outlier solve on a clean matte surface", t_sparse_clean),
        ("dark pixels: finite normals", t_sparse_dark_pixels),
        ("specular excess and metal mask find the gilding", t_excess_and_mask),
        ("more sensitivity marks no less", t_sensitivity_monotone),
        ("four lamps: the flare finds a glint the fit took in", t_flare_four_lamps),
        ("fall-off across the sheet and dark cloth are not metal", t_flare_ignores_falloff_and_dark),
        ("normals blend and stay unit length", t_blend),
        ("Rust kernels agree with NumPy", t_rust_parity),
        ("Rust kernels refuse wrong shapes", t_rust_bad_shapes),
        ("pipeline: semi-calibrated", t_pipeline_semi_calibrated),
        ("pipeline: grey reference and semi-calibrated", t_pipeline_semi_off_with_flash),
        ("pipeline: Sparse-Outlier LS (RPCA)", t_pipeline_sparse_solver),
        ("pipeline: metal-aware solve", t_pipeline_metal_aware),
        ("settings: new keys round-trip", t_settings_round_trip),
    ]:
        check(name, fn)
    bad = [n for n, e in _results if e is not None]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
