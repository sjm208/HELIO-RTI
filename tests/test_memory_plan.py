"""Memory awareness: reading the machine's memory and planning a solve.

    python tests/test_memory_plan.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from helio.utils import memory_plan as M                              # noqa: E402
from helio.domain.settings import ProcessingSettings                  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def machine(gb):
    os.environ["HELIO_MEMORY_GB"] = str(gb)


def main():
    saved = os.environ.pop("HELIO_MEMORY_GB", None)
    total, avail = M.system_memory()
    check("the machine's memory is read (total >= available > 0)", total >= avail > 0, (total, avail))
    machine(16)
    check("HELIO_MEMORY_GB sets the machine (80% of it free)", M.system_memory() == (16 * 10**9, int(16 * 10**9 * 0.8))
          and "16 GB" in M.describe_machine())
    os.environ["HELIO_MEMORY_GB"] = "nonsense"
    check("a bad HELIO_MEMORY_GB is ignored", M.system_memory()[0] > 0)

    print("The estimate, against what was measured (net of start-up, plus the app baseline, with margin)")
    for mp, n, measured in ((6, 8, 1.46), (12, 8, 2.47), (12, 4, 1.95), (12, 16, 3.11), (24, 8, 4.62)):
        est = M.estimate_solve_bytes(int(mp * 1e3), 1000, n) / 1e9
        floor = measured + M.APP_BASELINE / 1e9
        check(f"{mp} MP x {n} photos: estimate {est:.1f} GB covers the measured {floor:.1f} GB, within 45% over",
              floor <= est <= floor * 1.45, (est, floor))
    base = M.estimate_solve_bytes(10000, 10000, 8)
    check("100 MP x 8 photos needs roughly 20-30 GB", 20e9 < base < 30e9, base / 1e9)
    check("more photographs need more memory",
          M.estimate_solve_bytes(5000, 5000, 16) > M.estimate_solve_bytes(5000, 5000, 8))
    s = ProcessingSettings()
    on = ProcessingSettings(flatten_illumination=True, pre_clahe=True)
    check("options that hold another stack raise the estimate",
          M.estimate_solve_bytes(5000, 5000, 8, on) > M.estimate_solve_bytes(5000, 5000, 8, s))

    print("The plan, by machine")
    cases = ((96, "comfortable"), (64, "comfortable"), (32, "tight"), (16, "too_big"))
    for gb, want in cases:
        machine(gb)
        p = M.plan_solve(10000, 10000, 8)
        check(f"100 MP x 8 photos on a {gb} GB machine: {p.level}", p.level == want, p)
    machine(16)
    small = M.plan_solve(4000, 3000, 8)
    check("12 MP x 8 photos is comfortable on 16 GB, and says nothing", small.level == "comfortable" and small.message == "")
    machine(32)
    t = M.plan_solve(10000, 10000, 8)
    check("a tight plan says how much it needs and that the result is not reduced",
          "GB" in t.message and "not reduced" in t.message and t.strip_bytes < M.plan_solve(1000, 1000, 3).strip_bytes, t.message)
    machine(16)
    b = M.plan_solve(10000, 10000, 8)
    check("a too-big plan says it will swap and what would avoid it", "swap" in b.message and "crop" in b.message, b.message)
    check("smaller strips as memory gets shorter", b.strip_bytes <= t.strip_bytes < 128 << 20)

    print("In a solve")
    import tempfile
    import numpy as np
    from tools import make_golden as G
    from helio.application.pipeline import SolveRequest, solve
    from helio.core import parallel
    G.H, G.W = 240, 320
    paths, L = G.render_stack(tempfile.mkdtemp())
    seen = []
    machine(0.5)                                                         # a machine far too small for even this
    default_strip = parallel.STRIP_BYTES
    solve(SolveRequest(paths=paths, lights=L, settings=ProcessingSettings(photo_tone="linear")),
          progress=lambda stage, frac=None, detail="": seen.append(detail))
    check("a solve on a tiny machine reports the memory plan", any("Memory:" in (d or "") for d in seen), seen[:3])
    check("and puts the strip size back afterwards", parallel.STRIP_BYTES == default_strip)
    seen.clear()
    machine(64)
    r = solve(SolveRequest(paths=paths, lights=L, settings=ProcessingSettings(photo_tone="linear")),
              progress=lambda stage, frac=None, detail="": seen.append(detail))
    check("a solve on a roomy machine says nothing about memory", not any("Memory:" in (d or "") for d in seen))
    check("and still gives normals", r.normals_full.shape == (240, 320, 3))
    os.environ["HELIO_PROFILE"] = "1"
    import io
    import contextlib
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        solve(SolveRequest(paths=paths, lights=L, settings=ProcessingSettings(photo_tone="linear")))
    os.environ.pop("HELIO_PROFILE")
    rep = err.getvalue()
    check("HELIO_PROFILE=1 prints a per-stage timeline with machine and expected peak",
          "Solve profile:" in rep and "peak GB" in rep and "Solving normals" in rep and "expected peak" in rep, rep[:300])

    if saved is None:
        os.environ.pop("HELIO_MEMORY_GB", None)
    else:
        os.environ["HELIO_MEMORY_GB"] = saved
    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
