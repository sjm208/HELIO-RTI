"""RBF Relight (helio.core.rbf_relight and the view of that name).

  * at a photograph's own light, with no smoothing and every component kept,
    the relit picture is that photograph;
  * between two lights it lies between their photographs, and the weights
    sum to about one;
  * a sharp highlight stays sharp, where the PTM fit smears it;
  * the solve keeps the compressed frames, the view renders at preview and
    full size, follows the light, and every control moves it;
  * a cropped result keeps them, cut to the crop.

    python tests/test_rbf_relight.py
"""
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from helio.core import rbf_relight as B                  # noqa: E402

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


def dome(n=12):
    out = []
    for k in range(n):
        az = k * 2 * np.pi / n
        el = (0.45, 0.8, 1.15)[k % 3]
        out.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.array(out, np.float32)


def scene(L, side=64):
    """A matte sphere with a polished band: matte shading plus a sharp mirror lobe."""
    u = (np.arange(side) + 0.5) / side * 1.6 - 0.8
    U, V = np.meshgrid(u, u)
    Z = np.sqrt(np.maximum(1 - U ** 2 - V ** 2, 0.05))
    N = np.stack([U, V, Z], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    band = np.zeros((side, side), bool)
    band[28:36] = True
    stack = []
    for l in L:
        h = (l + np.array([0, 0, 1.0])) / np.linalg.norm(l + np.array([0, 0, 1.0]))
        img = 140 * np.maximum(N @ l, 0) + np.where(band, 110 * np.clip(N @ h, 0, 1) ** 80, 0)
        stack.append(np.clip(img, 0, 255))
    return np.array(stack, np.float32), N, band


def t_reproduces_photographs():
    L = dome()
    S, _, _ = scene(L)
    rbf = B.build(S, L, S.shape[1:], keep_full=False)
    for j in (0, 5, 11):
        out = B.relight(rbf, L[j], smooth=0.0)
        err = np.abs(out - S[j]).max()
        assert err < 0.6, f"frame {j}: off by {err:.2f}"


def t_between_lights():
    L = dome()
    S, _, band = scene(L)
    mid = (L[0] + L[3]) / np.linalg.norm(L[0] + L[3])
    w = B.weights(L, mid)
    assert 0.8 < w.sum() < 1.2, w.sum()
    out = B.relight(B.build(S, L, S.shape[1:], keep_full=False), mid)
    truth, _, _ = scene(mid[None, :])                   # the matte surface under that light
    err = np.abs(out - truth[0])[~band].mean()
    assert err < 8.0, f"matte areas off by {err:.1f} of 255 between lights"


def t_highlight_sharper_than_ptm():
    from helio.core.processing import compute_ptm_hsh_coefficients
    from helio.core.shading_engines import compute_ptm_relighting
    L = dome()
    S, _, band = scene(L)
    j = 4
    rbf = B.relight(B.build(S, L, S.shape[1:], keep_full=False), L[j], smooth=0.0)
    ptm, _ = compute_ptm_hsh_coefficients(L, S.reshape(len(S), -1))
    p = compute_ptm_relighting(ptm, float(L[j][0]), float(L[j][1])).reshape(S.shape[1:])
    e_rbf = np.abs(rbf - S[j])[band].mean()
    e_ptm = np.abs(p - S[j])[band].mean()
    assert e_rbf < e_ptm / 3, f"on the polished band: RBF {e_rbf:.2f}, PTM {e_ptm:.2f}"


def t_full_and_proxy():
    L = dome()
    S, _, _ = scene(L, 96)
    rbf = B.build(S, L, (48, 48), keep_full=True)
    assert rbf["comps_full"] is not None and rbf["comps_proxy"].shape[1:] == (48, 48)
    a = B.relight(rbf, L[2], full=True)
    b = B.relight(rbf, L[2], full=False)
    import cv2
    assert np.abs(cv2.resize(a, (48, 48), interpolation=cv2.INTER_AREA) - b).mean() < 1.0
    assert B.build(S[:3], L[:3], (48, 48)) is None


def t_crop():
    L = dome()
    S, _, _ = scene(L, 96)
    rbf = B.build(S, L, (48, 48), keep_full=True)
    c = B.crop(rbf, (10, 20, 70, 60), 96, 96)
    a = B.relight(c, L[1], full=True)
    assert a.shape == (40, 60)
    assert np.abs(a - B.relight(rbf, L[1], full=True)[20:60, 10:70]).max() < 1e-3
    no_full = dict(rbf, comps_full=None, mean_full=None)
    assert B.crop(no_full, (10, 20, 70, 60), 96, 96)["comps_proxy"].shape[0] == len(rbf["U"][0])


def solved():
    from helio.application.pipeline import SolveRequest, solve, store
    from helio.data_manager import DataManager
    L = dome()
    S, _, _ = scene(L, 64)
    frames = [(np.repeat(f.astype(np.uint8)[..., None], 3, 2), f.astype(np.uint8), False) for f in S]
    r = solve(SolveRequest(paths=[], lights=L, frames=frames))
    D = DataManager()
    store(r, D)
    return r, D


def t_pipeline_and_view():
    from helio.application.render import render
    from helio.domain.view_controls import controls_for, light_kind
    name = "RBF Relight (Shiny / Translucent)"
    r, D = solved()
    assert r.rbf is not None and D.rbf is r.rbf
    assert light_kind(name) == "dome"
    base = dict(lx=0.3, ly=0.2, lz=0.93, gain=1.0, rbf_width=1.2, rbf_smooth=0.05)
    a = render(name, D, base, D.ph, D.pw, True)
    assert a.shape == (D.ph, D.pw) and np.isfinite(a).all() and a.std() > 0.01
    b = render(name, D, dict(base, lx=-0.6, ly=0.0, lz=0.8), D.ph, D.pw, True)
    assert np.abs(a - b).mean() > 0.01, "the light does not move it"
    full = render(name, D, base, D.h, D.w, False, (0, 0, D.w, D.h))
    assert full.shape == (D.h, D.w)
    part = render(name, D, base, 20, 30, False, (10, 12, 40, 32))
    assert part.shape == (20, 30) and np.abs(part - full[12:32, 10:40]).max() < 1e-3
    for c in controls_for(name):
        lo = render(name, D, dict(base, **{c.key: c.lo}), D.ph, D.pw, True)
        hi = render(name, D, dict(base, **{c.key: c.hi}), D.ph, D.pw, True)
        assert np.abs(lo - hi).mean() > 1e-4, f"{c.key} moves nothing"


def t_crop_result():
    from helio.application.pipeline import crop_result
    r, _ = solved()
    c = crop_result(r, (8, 8, 40, 48))
    assert c.rbf is not None and c.rbf["mean_full"].shape == (40, 32)


def main():
    for name, fn in [
        ("a photograph's own light gives that photograph", t_reproduces_photographs),
        ("between two lights: between their photographs", t_between_lights),
        ("a polished highlight: sharper than the PTM", t_highlight_sharper_than_ptm),
        ("full size and preview agree", t_full_and_proxy),
        ("a crop keeps the frames", t_crop),
        ("the solve keeps them; the view renders, follows the light and every control", t_pipeline_and_view),
        ("crop_result keeps them", t_crop_result),
    ]:
        check(name, fn)
    bad = [n for n, e in _results if e is not None]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
