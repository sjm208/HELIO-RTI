"""PCA 1-3 and Colour ICA give the same picture however the maths library signs its answer.

A principal or independent component is only defined up to its sign, so the same photographs
could give a picture or its photographic negative.  Each is now shown with its rare strong pixels
bright on a darker ground (skewness > 0).  This test makes the library return randomly signed
eigenvectors and requires the same pictures.

    python tests/test_pca_ica_determinism.py
"""
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    from tools import make_golden as G
    from helio.application import pipeline as P
    from helio.application.viewing import Display, screen_image
    from helio.core import parallel
    from helio.core.processing import skew_sign
    from helio.data_manager import DataManager
    from helio.domain.registry import default_engine_settings
    from helio.domain.settings import ProcessingSettings

    print("The sign rule")
    rng = np.random.default_rng(1)
    x = rng.gamma(2.0, 1.0, 50_000) - 2.0
    check("a component with a long bright tail keeps its sign", skew_sign(x) == 1.0)
    check("and its negative is turned over", skew_sign(-x) == 1.0 and (-x * skew_sign(-x)).tolist() == (-x * -1).tolist() or skew_sign(-x) == -1.0)
    sym = rng.standard_normal(50_000)
    sym = np.concatenate([sym, -sym])
    check("a symmetric one falls back to its largest loading",
          skew_sign(sym, np.array([0.1, -0.9, 0.2])) == -1.0 and skew_sign(sym, np.array([0.1, 0.9, 0.2])) == 1.0)
    check("a flat one is +1", skew_sign(np.zeros(10)) == 1.0)

    G.H, G.W = 240, 320
    paths, L = G.render_stack(tempfile.mkdtemp(prefix="helio_det_"))
    ps = ProcessingSettings(photo_tone="linear")
    real = np.linalg.eigh

    def signed(seed):
        r = np.random.default_rng(seed)

        def eigh(a, *args, **kw):
            w, v = real(a, *args, **kw)
            return w, v * r.choice([-1.0, 1.0], size=v.shape[1]).astype(v.dtype)
        return eigh

    outs = []
    for seed in (0, 1, 2, 3):
        np.linalg.eigh = signed(seed)
        try:
            res = P.solve(P.SolveRequest(paths=paths, lights=L, settings=ps))
        finally:
            np.linalg.eigh = real
        outs.append(res.pca_full.copy())
    print("PCA 1-3")
    check("four random signings give the same three pictures", all(np.array_equal(outs[0], o) for o in outs[1:]),
          [int(np.abs(outs[0].astype(int) - o.astype(int)).max()) for o in outs[1:]])
    res = P.solve(P.SolveRequest(paths=paths, lights=L, settings=ps))
    check("and the same as with the library's own signs", np.array_equal(res.pca_full, outs[0]))
    sk = []
    for i in range(3):
        c = res.pca_full[..., i].astype(np.float64).ravel()
        sk.append(float((((c - c.mean()) / c.std()) ** 3).mean()))
    check("each has its rare pixels bright (skewness >= 0)", all(s >= -0.05 for s in sk), sk)

    print("Colour ICA")
    D = DataManager()
    P.store(res, D)
    D.plate_paths = list(paths)
    st = default_engine_settings()["Colour ICA (IC1-3)"]
    pics = []
    for seed in (None, 11, 12):
        D._cica_cache = None
        parallel.CACHE.clear()
        if seed is not None:
            np.linalg.eigh = signed(seed)
        try:
            pics.append(np.asarray(screen_image("Colour ICA (IC1-3)", D, st, Display(layer="Shading"))).astype(np.float32))
        finally:
            np.linalg.eigh = real
    d = [float(np.abs(pics[0] - p).mean()) for p in pics[1:]]
    check("randomly signed eigenvectors give the same picture (mean difference < 3/255)", max(d) < 3.0, d)
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
