"""Depth on the object alone, and the score's Depth part (helio.core.object_mask).

A raised dome on a black surround whose normals are noise, as a steel punch on
velvet gives:

* the object is found (and nothing is masked when the object fills the frame);
* integrated on the object alone, the depth agrees with the normals far better
  than the whole frame does, the dome's height is right, the surround is flat;
* the depth views show the surround flat, with no halo at the object's edge;
* depth_agreement is near 0 for an exact surface and large for noise;
* the score's Depth part ranks a noisy-depth solve below a clean one.

    python tests/test_object_mask.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def scene(h=300, w=400, R=60.0, H=30.0, seed=1):
    """A spherical cap (radius R px, height H px) on a black surround with noise normals."""
    from helio.core.frames import normals_from_slopes
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r2 = (x - w / 2) ** 2 + (y - h / 2) ** 2
    rho = (R * R + H * H) / (2 * H)                         # the cap's sphere
    inside = r2 < R * R
    z = np.where(inside, np.sqrt(np.maximum(rho ** 2 - r2, 0)) - (rho - H), 0.0)
    gy, gx = np.gradient(z)
    n = normals_from_slopes(gx, gy, 1.0).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(h, w, 3)).astype(np.float32)
    noise[..., 2] = np.abs(noise[..., 2]) + 0.3
    noise /= np.linalg.norm(noise, axis=-1, keepdims=True)
    n[~inside] = noise[~inside]
    albedo = np.where(inside, 180, 4).astype(np.uint8) + rng.integers(0, 6, (h, w)).astype(np.uint8)
    return n, np.dstack([albedo] * 3), inside, z


def main():
    from helio.application.pipeline import integrate_depth
    from helio.application.process_tools import overall_score
    from helio.core.object_mask import depth_agreement, object_mask
    from helio.core.processing import depth_visualizations
    n, albedo, inside, z_true = scene()

    print("The object")
    m = object_mask(albedo)
    check("found on the dark surround", m is not None and abs(m.mean() - inside.mean()) < 0.01,
          m is not None and (m.mean(), inside.mean()))
    full = np.full_like(albedo, 150)
    check("nothing masked when the object fills the frame", object_mask(full) is None)

    print("\nIts depth")
    eng = "Fast Poisson (FFT)"
    z_whole = integrate_depth(n, eng)
    z_obj = integrate_depth(n, eng, mask=m)
    a_whole, a_obj = depth_agreement(n, z_whole, m), depth_agreement(n, z_obj, m)
    check(f"the object's depth agrees with its normals: {a_obj:.1f}° alone, {a_whole:.1f}° whole frame",
          a_obj < 3.0 and a_obj < 0.5 * a_whole)
    top = float(z_obj[inside].max() - np.median(z_obj[~inside]))
    check(f"the dome's height is right: {top:.1f} px (true 30)", abs(top - 30.0) < 3.0)
    check("the surround is flat: it carries on from the dome's rim, which meets it "
          f"(spread {float(np.std(z_obj[~m])):.3f} px against a 30 px dome)",
          float(np.std(z_obj[~m])) < 0.1)
    for e in ("Frankot-Chellappa", "Least-Squares (Free Boundary)"):
        check(f"…with {e} too", depth_agreement(n, integrate_depth(n, e, mask=m), m) < 3.0)

    print("\nImpossible normals")
    from helio.application.pipeline import repair_normals
    _n, _a, _in, _z = scene(seed=2)
    nb = _n.copy()
    rng = np.random.default_rng(5)
    ys, xs = np.nonzero(_in)
    pick = rng.choice(len(ys), int(0.02 * len(ys)), replace=False)
    nb[ys[pick], xs[pick]] = (0.0, 0.3, -0.95)            # facing away: 2% of the dome
    zb = integrate_depth(nb, eng, mask=object_mask(_a))
    zg = integrate_depth(_n, eng, mask=object_mask(_a))
    span_b, span_g = float(np.ptp(zb[_in])), float(np.ptp(zg[_in]))
    check(f"2% facing away no longer wrecks the depth: spans {span_b:.1f} px, clean {span_g:.1f}",
          abs(span_b - span_g) < 0.1 * span_g)
    check("normals with none impossible are passed through untouched", repair_normals(_n) is _n)

    print("\nThe views")
    v = depth_visualizations(z_obj, n, 0.5, m)["depth_vis_p"]
    ring = (~m) & (np.hypot(*np.mgrid[0:m.shape[0], 0:m.shape[1]] -
                            np.array(m.shape)[:, None, None] / 2) < 90)
    check("the high-pass view: no halo round the object, the surround near one grey",
          float(np.ptp(v[ring])) < 0.05 and float(np.ptp(v[~m])) < 0.05,
          f"{float(np.ptp(v[ring])):.3f} {float(np.ptp(v[~m])):.3f}")

    print("\nA large object whose rim is not at the surround's level (a binding)")
    from helio.core.object_mask import extend_from
    yy, xx = np.mgrid[0:300, 0:400].astype(np.float32)
    slab = (yy > 40) & (yy < 260) & (xx > 50) & (xx < 350)
    d_edge = np.minimum.reduce([yy - 40, 260 - yy, xx - 50, 350 - xx])
    z_slab = np.where(slab, -20.0 + 20.0 * np.clip(d_edge / 30.0, 0, 1), 0.0).astype(np.float32)
    ext = extend_from(z_slab, slab)
    inner = slab & ~np.pad(slab[1:-1, 1:-1] & slab[:-2, 1:-1] & slab[2:, 1:-1]
                           & slab[1:-1, :-2] & slab[1:-1, 2:], 1)
    outer = (~slab) & np.pad(slab[1:-1, :-2] | slab[1:-1, 2:] | slab[:-2, 1:-1] | slab[2:, 1:-1], 1)
    jump = abs(float(np.median(ext[outer]) - np.median(z_slab[inner])))
    check(f"the surround meets a rim 20 px below it with no step ({jump:.2f} px; it stood "
          "20 px up from it as a cliff all round, which the 3D relief drew as a wall)", jump < 1.0)
    check("…keeps the object exactly", np.array_equal(ext[slab], z_slab[slab]))
    gy, gx = np.gradient(ext)
    check("…and is smooth beyond it", float(np.abs(np.hypot(gx, gy)[~slab]).max()) < 2.0)

    print("\nThe score")
    check("an exact surface agrees to under half a degree",
          depth_agreement(n, z_true, inside) < 0.5, depth_agreement(n, z_true, inside))
    rng = np.random.default_rng(3)
    check("a depth of noise does not", depth_agreement(n, rng.normal(size=z_true.shape), inside) > 20)
    clean = overall_score(2.0, 5.0, 1.5, None, a_obj)
    noisy = overall_score(2.0, 5.0, 1.5, None, 40.0)
    check("the score has a Depth part out of 20", ("Depth", 20) in [(k, o) for k, _p, o in clean["parts"]])
    check(f"…which puts a blotchy depth below a clean one ({noisy['score']} < {clean['score']})",
          noisy["score"] < clean["score"])
    check("without a depth the score is as it was",
          [k for k, _p, _o in overall_score(3, 9, 2)["parts"]] == ["Sureness", "Lights"])

    print("\nThe setting")
    from helio.domain.settings import ProcessingSettings
    check("off by default", ProcessingSettings().mask_surround is False)

    print("\nSet Up for Measurement: the mask for a small object only")
    from helio.application.precision import measurement_changes
    from helio.core.object_mask import SMALL_OBJECT, small_object
    check(f"the dome ({inside.mean():.0%} of the frame) is a small object", small_object(albedo) is True)
    _n2, big_alb, big_in, _z2 = scene(R=170.0, H=40.0)         # a binding: most of the frame
    check(f"a large object ({big_in.mean():.0%} of the frame, over {SMALL_OBJECT:.0%}) is not "
          "(held flat all round it came out a pillow with lifted edges)",
          big_in.mean() > SMALL_OBJECT and small_object(big_alb) is False, f"{big_in.mean():.2f}")
    full = np.full((300, 400, 3), 150, np.uint8)
    check("nor is an object filling the frame", small_object(full) is False)
    check("before a solve, nothing to judge by", small_object(None) is None)
    off, on = ProcessingSettings(), ProcessingSettings(mask_surround=True)
    check("the preset turns the mask on for the small object",
          measurement_changes(off, albedo).get("mask_surround") is True)
    check("…turns it off for the large one", measurement_changes(on, big_alb).get("mask_surround") is False)
    check("…and before a solve leaves it as it is",
          "mask_surround" not in measurement_changes(on) and "mask_surround" not in measurement_changes(off))


main()
print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
