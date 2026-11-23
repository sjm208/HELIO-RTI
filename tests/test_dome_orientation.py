"""The dome's "Check which way round": a hand-placed dome mirrored east-west or north-south.

A mirrored light set fits every frame exactly as well (``score`` cannot see it), so the check
reads the photographs alone.  On a surface lit by near lamps, with the lights placed right and
mirrored each way:

* placed right, it suggests nothing;
* mirrored east-west it suggests "h", north-south "v";
* lamps far away: the broad shading of the surface still says which side each frame is lit from;
* the user's capture PR-F-00156, when present: the order as photographed passes, mirrored is caught.

    python tests/test_dome_orientation.py
"""
import glob
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS = FAIL = 0
REAL = os.path.expanduser("~/Pictures/ORDERS/Output/PS IMAGES/PR-F-00156-D-00001-00002")


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def ring(n, el, start_az=180.0, step=-45.0):
    e = math.radians(el)
    return [(math.cos(e) * math.sin(math.radians(start_az + step * k)),
             math.cos(e) * math.cos(math.radians(start_az + step * k)), math.sin(e)) for k in range(n)]


def render(lights, near=True):
    import numpy as np
    from tools import make_golden as G
    G.H, G.W = 240, 320
    _z, n, albedo = G.analytic_surface()
    rho = albedo.astype(np.float32).mean(axis=2) / 255.0
    yy, xx = np.mgrid[0:G.H, 0:G.W].astype(np.float32)
    u, v = (xx - G.W / 2) / G.W, (G.H / 2 - yy) / G.W          # +Y up, in picture widths
    out = []
    for l in lights:
        l = np.asarray(l, np.float32)
        shade = np.maximum(0.0, n @ l)
        if near:                                               # a lamp about a picture away
            lamp = l * 1.2
            d2 = (lamp[0] - u) ** 2 + (lamp[1] - v) ** 2 + lamp[2] ** 2
            shade = shade * (1.44 / d2)
        out.append(np.clip(rho * shade * 0.8, 0, 1).astype(np.float32))
    return out


def main():
    from helio.application import dome_tools as dt

    print("Near lamps, eight in a ring at 30°")
    L = ring(8, 30)
    plates = render(L)
    r = dt.orientation_check(L, plates)
    check("placed right: nothing suggested", r["suggest"] is None, r)
    r = dt.orientation_check(dt.mirror(L, "h"), plates)
    check("mirrored east-west: suggests h", r["suggest"] == "h", r)
    r = dt.orientation_check(dt.mirror(L, "v"), plates)
    check("mirrored north-south: suggests v", r["suggest"] == "v", r)
    r = dt.orientation_check(dt.mirror(dt.mirror(L, "h"), "v"), plates)
    check("turned half round (the relief inside out, which no test can tell): nothing suggested",
          r["suggest"] is None, r)

    print("Lamps far away: only the surface's own broad shading to go by")
    plates = render(L, near=False)
    r = dt.orientation_check(dt.mirror(L, "h"), plates)
    check("mirrored east-west: still suggests h", r["suggest"] == "h", r)
    check("placed right: nothing suggested", dt.orientation_check(L, plates)["suggest"] is None)
    check("too few frames: nothing suggested", dt.orientation_check(L[:3], plates[:3])["suggest"] is None)

    paths = sorted(glob.glob(os.path.join(REAL, "*.jpg")))
    if len(paths) == 8:
        import cv2
        print("PR-F-00156: eight lights from the bottom, round by the right (as photographed)")
        plates = []
        for p in paths:
            g = cv2.imread(p, cv2.IMREAD_REDUCED_GRAYSCALE_8)
            plates.append(g.astype("float32") / 255.0)
        L = ring(8, 30)
        check("as photographed: nothing suggested", dt.orientation_check(L, plates)["suggest"] is None)
        r = dt.orientation_check(ring(8, 30, step=45.0), plates)
        check("round by the left instead: suggests h", r["suggest"] == "h", r)
    else:
        print("PR-F-00156 not on this machine: the real capture skipped")

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
