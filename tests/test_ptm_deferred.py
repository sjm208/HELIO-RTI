"""PTM / HSH on demand, on machines the memory plan calls tight.

* a roomy machine keeps the coefficients, as before;
* a tight one drops them (60 bytes a pixel) and keeps the previews, and every
  Inspect view of them looks exactly as it did;
* the coefficients fitted again from the photographs are the ones that were dropped,
  for a plain solve, a cropped one and one at a reduced working size;
* exports and full-size renders that need them ask for them.

    python tests/test_ptm_deferred.py
"""
import dataclasses
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import make_golden as G                                         # noqa: E402
from helio.application import pipeline as P                           # noqa: E402
from helio.application.render import render                           # noqa: E402
from helio.data_manager import DataManager                            # noqa: E402
from helio.domain.registry import default_engine_settings             # noqa: E402
from helio.domain.settings import ProcessingSettings                  # noqa: E402
from helio.utils import memory_plan as M                              # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def stored(r):
    D = DataManager()
    P.store(r, D)
    return D


def main():
    for k in ("HELIO_MEMORY_GB", "HELIO_KEEP_PTM", "HELIO_WORK_SCALE"):
        os.environ.pop(k, None)
    G.H, G.W = 480, 640
    paths, L = G.render_stack(tempfile.mkdtemp(prefix="helio_ptm_"))
    ps = ProcessingSettings(photo_tone="linear")
    req = P.SolveRequest(paths=paths, lights=L, settings=ps)
    eng = default_engine_settings()

    print("A roomy machine")
    os.environ["HELIO_MEMORY_GB"] = "64"
    eager = P.solve(req)
    check("keeps the coefficients", eager.ptm_coeffs is not None and eager.hsh_coeffs is not None
          and not eager.ptm_deferred and eager.refit is None)

    print("A tight machine")
    os.environ["HELIO_MEMORY_GB"] = "0.5"
    lvl = M.plan_solve(480, 640, len(paths), ps).level
    check("the plan calls it tight or worse", lvl != "comfortable", lvl)
    lazy = P.solve(req)
    check("drops them and says so", lazy.ptm_coeffs is None and lazy.hsh_coeffs is None and lazy.ptm_deferred
          and lazy.refit is req and len(lazy.ptm_planes) == 6)
    check("the preview is the one a roomy machine makes", np.array_equal(lazy.proxy["ptm"], eager.proxy["ptm"]))
    check("everything else is the same", np.array_equal(lazy.normals_full, eager.normals_full)
          and np.array_equal(lazy.depth_vis_p, eager.depth_vis_p))
    os.environ["HELIO_KEEP_PTM"] = "1"
    check("HELIO_KEEP_PTM keeps them anyway", P.solve(req).ptm_coeffs is not None)
    os.environ.pop("HELIO_KEEP_PTM")

    print("Inspect looks the same")
    De, Dl = stored(eager), stored(lazy)
    check("the store knows it is deferred", Dl.ptm_deferred and Dl.ptm_coeffs is None and Dl.ptm_refit is not None
          and not De.ptm_deferred)
    ph, pw = eager.ph, eager.pw
    for name in ("Specular Enhancement", "Image Quotient (IQ)", "RTI Specular Enhancement"):
        s = dict(eng[name])
        a = render(name, De, s, ph, pw, True)
        b = render(name, Dl, s, ph, pw, True)
        check(f"{name}: preview identical", np.allclose(a, b, atol=1e-6, rtol=0), np.abs(a - b).max())
    check("nothing was fitted to draw them", Dl.ptm_coeffs is None)

    print("Fitted again")
    D = stored(P.solve(req))
    ptm, hsh = D.ensure_ptm_hsh()
    check("the same PTM", np.allclose(ptm, eager.ptm_coeffs, atol=1e-6), np.abs(ptm - eager.ptm_coeffs).max())
    check("the same HSH", np.allclose(hsh, eager.hsh_coeffs, atol=1e-6), np.abs(hsh - eager.hsh_coeffs).max())
    check("kept after the first time", D.ensure_ptm_hsh()[0] is ptm)
    D2 = stored(P.solve(req))
    for name in ("Specular Enhancement", "RTI Specular Enhancement"):
        a = render(name, De, dict(eng[name]), 480, 640, False)
        b = render(name, D2, dict(eng[name]), 480, 640, False)
        check(f"{name}: a full-size render fits them and matches", np.allclose(a, b, atol=1e-6), np.abs(a - b).max())
    check("and keeps them", D2.ptm_coeffs is not None)
    seen = []
    r3 = P.solve(req)
    P.ensure_ptm_hsh(r3, lambda *a, **k: seen.append(a))
    check("a progress message says what is happening", seen and "again" in str(seen[0]), seen[:1])

    print("A crop")
    creq = dataclasses.replace(req, crop=(100, 80, 420, 330))
    os.environ["HELIO_MEMORY_GB"] = "64"
    ce = P.solve(creq)
    os.environ["HELIO_MEMORY_GB"] = "0.5"
    cl = P.solve(creq)
    p, _ = P.ensure_ptm_hsh(cl)
    check("fitted again inside the crop", p.shape == ce.ptm_coeffs.shape and np.allclose(p, ce.ptm_coeffs, atol=1e-6))

    print("A selection cut from a tight result")
    sel = P.crop_result(lazy, (100, 80, 420, 330))
    cut_eager = P.crop_result(eager, (100, 80, 420, 330))
    check("still deferred, planes cut, preview the right size",
          sel.ptm_deferred and sel.ptm_coeffs is None and sel.proxy["ptm"].shape == (6, sel.ph * sel.pw)
          and sel.ptm_planes[0].shape == (sel.ph, sel.pw))
    p, _ = P.ensure_ptm_hsh(sel)
    check("fits the selection's own pixels", p.shape == (6, sel.h * sel.w)
          and np.allclose(p, cut_eager.ptm_coeffs, atol=1e-6), np.abs(p - cut_eager.ptm_coeffs).max())
    sel2 = P.crop_result(P.solve(creq), (10, 10, 200, 150))
    p2, _ = P.ensure_ptm_hsh(sel2)
    ref2 = P.crop_result(ce, (10, 10, 200, 150)).ptm_coeffs
    check("a selection inside a crop", np.allclose(p2, ref2, atol=1e-6))
    full_cut = P.crop_result(lazy, (0, 0, 640, 480))
    check("the whole picture as a selection", full_cut.ptm_deferred and full_cut.ptm_planes[0].shape == (full_cut.ph, full_cut.pw))

    print("At a reduced working size")
    hreq = dataclasses.replace(req, settings=dataclasses.replace(ps, work_scale=0.5))
    os.environ["HELIO_MEMORY_GB"] = "64"
    he = P.solve(hreq)
    os.environ["HELIO_MEMORY_GB"] = "0.5"
    hl = P.solve(hreq)
    p, _ = P.ensure_ptm_hsh(hl)
    check("fitted again at the same reduced size", hl.ptm_deferred and p.shape == he.ptm_coeffs.shape
          and np.allclose(p, he.ptm_coeffs, atol=1e-6))

    print("The photographs have gone")
    gone = P.solve(req)
    os.rename(paths[0], paths[0] + ".moved")
    try:
        P.ensure_ptm_hsh(gone)
        check("a clear error, nothing half-stored", False)
    except Exception as e:
        check("a clear error, nothing half-stored", gone.ptm_coeffs is None, f"{type(e).__name__}: {e}")
    os.rename(paths[0] + ".moved", paths[0])

    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
