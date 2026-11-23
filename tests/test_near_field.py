"""The near-field solver, on lamps truly near.

The full check's known surface is lit by distant lamps, which this solver does
not model, so it read 24.7° there.  On what it is for it recovers the surface,
and choosing the solver uses the dome radius, as ticking the box does.

    python tests/test_near_field.py
"""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


H, W = 120, 160
yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
gx = 0.3 * np.sin(xx / 15)
TRUTH = np.dstack([-gx, 0.2 * np.cos(yy / 11), np.ones_like(gx)])
TRUTH /= np.linalg.norm(TRUTH, axis=2, keepdims=True)
az = np.arange(8) * np.pi / 4
LIGHTS = np.stack([np.cos(az) * 0.75, np.sin(az) * 0.75, np.full(8, 0.66)], 1)
LIGHTS /= np.linalg.norm(LIGHTS, axis=1, keepdims=True)


def lit(radius, origin=(0.0, 0.0)):
    X, Y = np.meshgrid(np.linspace(-W / 2, W / 2, W) - origin[0], np.linspace(H / 2, -H / 2, H) - origin[1])
    P = np.dstack([X, Y, np.zeros_like(X)])
    out = []
    for l in LIGHTS:
        v = l * radius - P
        d = np.linalg.norm(v, axis=2)
        out.append(200 * (radius / d) ** 2 * np.clip((TRUTH * (v / d[..., None])).sum(-1), 0, None))
    return np.array(out, np.float32)


def err(n):
    return float(np.degrees(np.arccos(np.clip((n * TRUTH).sum(-1), -1, 1))).mean())


def main():
    from helio.application.pipeline import near_field_radius_px
    from helio.core.photometric_stereo import solve_near_field_normals as solve
    e = err(solve(lit(240), LIGHTS, H, W, dome_radius_px=240))
    check(f"near lamps: the surface comes back ({e:.2f}°)", e < 0.5)
    e = err(solve(lit(240, (40, -20)), LIGHTS, H, W, dome_radius_px=240, origin=(40, -20)))
    check(f"with the ball off centre ({e:.2f}°)", e < 0.5)
    wrong = err(solve(lit(800), LIGHTS, H, W, dome_radius_px=240))
    check(f"the wrong radius shows ({wrong:.1f}°): it matters", wrong > 3)

    ps = NS(near_field=False, normal_engine="Near-Field Attenuation", dome_radius_mm=150.0)
    check("the solver chosen, with a scale: the radius given", near_field_radius_px(ps, 0.5, H, W) == (300.0, "measured"))
    check("no scale: the guess, and says so", near_field_radius_px(ps, None, H, W) == (240.0, "assumed"))
    ps.normal_engine, ps.near_field = "Linear Least Squares", True
    check("the box ticked: the same", near_field_radius_px(ps, 0.5, H, W)[1] == "measured")
    ps.near_field = False
    check("neither: not used", near_field_radius_px(ps, 0.5, H, W)[1] == "assumed")

    # Retired names (docs/OVERLAP_AUDIT.md): old sessions still mean what they meant.
    from helio.domain.registry import NORMAL_SOLVERS
    from helio.domain.settings import ProcessingSettings
    old = ProcessingSettings(normal_engine="Near-Field Attenuation")
    check("the near-field solver is the checkbox now: an old session gets least squares with it on",
          old.normal_engine == "Linear Least Squares" and old.near_field
          and "Near-Field Attenuation" not in NORMAL_SOLVERS)
    old = ProcessingSettings(normal_engine="Robust Least Squares (Shiny Metal)",
                             metal_solver="Robust Least Squares (Shiny Metal)")
    check("'Shiny Metal' is Mid-Tone Weighted LS, as solver and as metal solver",
          old.normal_engine == old.metal_solver == "Mid-Tone Weighted LS")
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
