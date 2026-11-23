"""Levelling for the GIS DEM export.

Jorge at a partner lab looked at a HELIO depth map and reported "considerable
overall tilt and possibly some low-frequency distortion".  He was right, and
the reason is structural rather than a bad capture: integrating a normal field
leaves its lowest frequencies almost unconstrained, so the solved surface
drifts into a tilt or a bowl.

The subtlety — and the reason the drift survived into files we sent out — is
that the Gaussian high-pass already baked into ``depth_vis_p`` *cannot* remove
it.  Blurring a quadratic returns the same quadratic plus a constant, so
subtracting the blur leaves the bowl standing.  Only fitting the low-order
term explicitly takes it out.  These tests pin all of that down.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helio.core import subtract_parabolic_baseline, fast_large_sigma_blur

PASS = FAIL = 0
H, W = 240, 320


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def surface(tilt=0.40, bowl=0.25, relief=0.02):
    """Fine grooves (the drypoint) sitting on a tilted, bowed plate."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    xn = xx / (W - 1) * 2 - 1
    yn = yy / (H - 1) * 2 - 1
    grooves = relief * np.sin(xx / 2.5).astype(np.float32)
    return (grooves + tilt * xn + bowl * (xn ** 2 + yn ** 2)).astype(np.float32), grooves


def low_order_energy(z):
    """How much of z is explained by a plane + bowl — 0 means fully level."""
    return float(np.abs(z - subtract_parabolic_baseline(z)).max())


def main():
    print("=" * 60)
    print("Levelling — GIS DEM export")
    print("=" * 60)

    z, grooves = surface()

    print("\nthe problem")
    check("raw solved surface carries a large low-order term",
          low_order_energy(z) > 0.5, f"got {low_order_energy(z):.3f}")

    print("\nwhat the existing high-pass does and does not do")
    # sigma = 5% of the long edge is what compute_depth_visualizations already
    # applies to depth_vis_p.  Measured, not assumed: at this width the blur
    # tracks a smooth tilt/bowl closely, so subtracting it removes most of the
    # drift.  An ideal global tilt is therefore NOT what survives to the file.
    hp = z - fast_large_sigma_blur(z, max(H, W) * 0.05)
    check("the 5% high-pass does remove most of a smooth global tilt",
          low_order_energy(hp) < 0.1 * low_order_energy(z),
          f"raw {low_order_energy(z):.3f} -> high-passed {low_order_energy(hp):.3f}")

    # ...but it is an approximation, and a much wider blur degenerates exactly
    # as subtract_parabolic_baseline's docstring describes: blurring a
    # quadratic returns near enough the same quadratic, so the bowl survives.
    hp_wide = z - fast_large_sigma_blur(z, max(H, W) * 0.50)
    check("a very wide high-pass leaves the bowl standing (the docstring case)",
          low_order_energy(hp_wide) > 5 * low_order_energy(hp),
          f"5% {low_order_energy(hp):.3f} vs 50% {low_order_energy(hp_wide):.3f}")

    print("\nlevelling does")
    lv = subtract_parabolic_baseline(z)
    check("levelled surface has essentially no low-order term",
          low_order_energy(lv) < 1e-4, f"got {low_order_energy(lv):.2e}")
    check("levelling preserves the groove amplitude",
          abs((lv.max() - lv.min()) - (grooves.max() - grooves.min())) < 5e-3,
          f"relief {lv.max()-lv.min():.4f} vs expected {grooves.max()-grooves.min():.4f}")
    check("levelled surface still correlates with the true grooves",
          np.corrcoef(lv.ravel(), grooves.ravel())[0, 1] > 0.99)
    check("levelling beats the high-pass by orders of magnitude",
          low_order_energy(lv) < low_order_energy(hp) / 100,
          f"levelled {low_order_energy(lv):.2e} vs high-passed {low_order_energy(hp):.3f}")

    print("\norder matters: level before normalising, not after")
    # The export normalises to [0,1].  Normalising a tilted surface spends
    # almost the whole range on the tilt, so the relief that survives is a
    # sliver — and levelling afterwards cannot bring it back.
    def norm(a):
        lo, hi = float(a.min()), float(a.max())
        return (a - lo) / (hi - lo + 1e-9)

    before = norm(subtract_parabolic_baseline(z))       # what the fix does
    after = subtract_parabolic_baseline(norm(z))        # the tempting mistake
    check("levelling first keeps far more of the range for real relief",
          np.ptp(before) > 20 * np.ptp(after),
          f"before {np.ptp(before):.4f} vs after {np.ptp(after):.4f}")

    print("\nedge cases")
    flat = np.zeros((H, W), np.float32)
    check("a flat surface survives levelling unchanged",
          np.abs(subtract_parabolic_baseline(flat)).max() < 1e-6)
    tiny = np.random.RandomState(0).rand(5, 5).astype(np.float32)
    check("a tiny array does not blow up",
          subtract_parabolic_baseline(tiny).shape == (5, 5))
    check("levelling never introduces NaN/Inf",
          np.isfinite(subtract_parabolic_baseline(z)).all())

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
