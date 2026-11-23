"""Legend unwrap: circles, ovals and pointed ovals read straight.

Synthetic seals drawn with OpenCV's own circle and ellipse (so the outline
code is checked against something it did not draw):

* an oval seal's edge is found, and its proportions and turn;
* Straighten brings an off-centre circle, and a round guess at a turned
  oval, onto the borders, so they run flat along the strip;
* a pointed oval (vesica) has its points where they should be and a border
  drawn inside it at a constant width unrolls flat;
* the even raking light reads a ring of identical letters alike all round,
  where one fixed light reads them very differently;
* a part arc is that part's length; the whole face has a row per radius;
  anticlockwise legends put the inside at the top; mirroring an outline
  maps it onto the flipped picture;
* a free shape through anchor points: made from any outline it follows it;
  a lozenge is found; moving one point moves only that one; it mirrors;
* the object above its strip as one figure;
* several seals and a coin on one written page are each found, in reading
  order, and one is refined in the part round it;
* the scale band is the strip's width, and the sidecar record gives a
  circle's diameter and an oval's height x width in millimetres.

    python tests/test_legend.py
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helio.application import heritage_tools as T  # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def flatness(strip, rows):
    """How much the given rows of a grey strip vary along it (0 = flat)."""
    g = strip.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    return float(np.mean([g[r].std() for r in rows]))


def oval_seal(n=700, cx=360, cy=340, a=150, b=230, turn=20):
    """A wax-coloured oval seal: an outer and an inner border, beads between."""
    img = np.full((n, n, 3), 210, np.uint8)
    cv2.ellipse(img, (cx, cy), (a, b), turn, 0, 360, (150, 110, 60), -1, cv2.LINE_AA)
    for d in (8, 46):
        cv2.ellipse(img, (cx, cy), (a - d, b - d), turn, 0, 360, (60, 40, 20), 3, cv2.LINE_AA)
    return img


def main():
    print("Finding an oval")
    img = oval_seal()
    o = T.find_outline(img, "oval")
    check(f"the oval's centre, size and turn ({o['cx']:.0f}, {o['cy']:.0f}; {o['a']:.0f} x {o['b']:.0f}; "
          f"{o['turn']:.0f}°)", abs(o["cx"] - 360) < 6 and abs(o["cy"] - 340) < 6
          and abs(o["a"] - 150) < 12 and abs(o["b"] - 230) < 12 and abs(o["turn"] - 20) < 5)

    print("\nStraighten")
    disc = np.full((600, 600, 3), 200, np.uint8)
    cv2.circle(disc, (300, 290), 240, (120, 120, 120), -1, cv2.LINE_AA)
    for r in (230, 190):
        cv2.circle(disc, (300, 290), r, (30, 30, 30), 3, cv2.LINE_AA)
    off = dict(cx=309.0, cy=283.0, a=240.0, b=240.0, turn=0.0, shape="circle")
    good = T.straighten(disc, off, 60)
    before = flatness(T.unwrap(disc, T.legend_maps(off, 60)), [10, 50])
    after = flatness(T.unwrap(disc, T.legend_maps(good, 60)), [10, 50])
    check(f"an off-centre circle comes onto the borders (centre {good['cx']:.1f}, {good['cy']:.1f})",
          abs(good["cx"] - 300) < 1.5 and abs(good["cy"] - 290) < 1.5, good)
    check(f"…and the borders run flat ({before:.1f} -> {after:.1f})", after < 0.3 * before)
    # as far off as a Find on a worn or partly hidden seal might leave it
    guess = dict(cx=364.0, cy=337.0, a=150 * 1.03, b=230 * 0.97, turn=14.0, shape="oval")
    fit = T.straighten(img, guess, 50)
    check(f"a turned oval is straightened ({fit['cx']:.0f}, {fit['cy']:.0f}; {fit['a']:.0f} x {fit['b']:.0f}, "
          f"{fit['turn']:.1f}°)",
          abs(fit["cx"] - 360) < 3 and abs(fit["cy"] - 340) < 3 and abs(fit["turn"] - 20) < 3
          and abs(fit["b"] / fit["a"] - 230 / 150) < 0.06, fit)

    print("\nPointed oval")
    v = dict(cx=300.0, cy=300.0, a=120.0, b=220.0, turn=0.0, shape="vesica")
    P = T.outline_points(v, 2000)
    top, bottom = P[np.argmin(P[:, 1])], P[np.argmax(P[:, 1])]
    check("its points are at the top and bottom", np.allclose(top, (300, 80), atol=1.0)
          and np.allclose(bottom, (300, 520), atol=1.0), (top, bottom))
    check("it is as wide as asked", abs(P[:, 0].max() - 420) < 1.0 and abs(P[:, 0].min() - 180) < 1.0)
    canvas = np.full((600, 600), 230, np.uint8)
    m = T.legend_maps(v, 40, height=41)
    ring = np.stack([m["map_x"][20], m["map_y"][20]], axis=1)          # 20 px inside, all round
    cv2.polylines(canvas, [ring.astype(np.int32)], True, 0, 2, cv2.LINE_AA)
    strip = T.unwrap(canvas, m)
    sides = strip[:, int(0.15 * strip.shape[1]):int(0.35 * strip.shape[1])]
    check("a border drawn at a constant width unrolls flat",
          abs(int(np.argmin(sides.mean(axis=1))) - 20) <= 1 and flatness(sides, [20]) < 25)

    print("\nEven raking light")
    n = 500
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    r = np.hypot(xx - 250, yy - 250)
    dep = np.exp(-((r - 180) / 6) ** 2)                                # a raised ring all round
    gy, gx = np.gradient(cv2.GaussianBlur(dep, (0, 0), 1.0) * 30)
    nrm = np.dstack([-gx, gy, np.ones_like(gx)])
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    circ = dict(cx=250.0, cy=250.0, a=210.0, b=210.0, turn=0.0, shape="circle")
    m = T.legend_maps(circ, 60)
    even = T.even_raking(nrm, m, light_deg=0, elevation=25)
    fixed_light = np.array([0.0, 0.906, 0.423])                        # from the top of the picture
    fixed = np.clip(nrm @ fixed_light, 0, 1)
    fixed8 = (fixed * 255).astype(np.uint8)
    fx = T.unwrap(fixed8, m)
    lit_even, lit_fixed = even[24:30].mean(axis=0), fx[24:30].mean(axis=0)        # the ring's rim side
    check(f"the ring reads alike all round in the even light (spread {lit_even.std():.1f} vs "
          f"{lit_fixed.std():.1f} under one fixed light)", lit_even.std() < 0.25 * lit_fixed.std())

    print("\nSpan, whole face, direction, mirror")
    full = T.legend_maps(circ, 40)
    half = T.legend_maps(circ, 40, span_deg=180)
    check(f"half the span is half the strip ({half['map_x'].shape[1]} of {full['map_x'].shape[1]})",
          abs(half["map_x"].shape[1] - full["map_x"].shape[1] / 2) < 3)
    wf = T.legend_maps(circ, 40, whole_face=True)
    check("the whole face has a row per pixel of radius, the outer ring at the top",
          wf["map_x"].shape[0] == 210 and abs(float(wf["map_y"][0, 0]) - 40) < 1
          and abs(float(wf["map_y"][-1, 0]) - 250) < 1.5)
    acw = T.legend_maps(circ, 40, clockwise=False)
    check("anticlockwise puts the inside at the top and runs the other way",
          abs(float(acw["map_y"][0, 0]) - 80) < 1.5 and acw["map_x"][20, 10] < 250 - 1)
    pic = oval_seal()
    mo = T.mirrored(o, pic.shape[1])
    flip = pic[:, ::-1]
    a1 = T.unwrap(pic, T.legend_maps(o, 40))
    a2 = T.unwrap(flip, T.legend_maps(mo, 40, clockwise=False))
    check("a mirrored outline lies on the flipped picture",
          abs(mo["cx"] - (pic.shape[1] - 1 - o["cx"])) < 1e-6
          and abs(float(a1.mean()) - float(a2.mean())) < 3)

    print("\nFree shape")
    fr = T.free_from(circ)
    Pf = T.outline_points(fr, 720)
    rr = np.hypot(Pf[:, 0] - 250, Pf[:, 1] - 250)
    check(f"made from a circle, it follows it ({len(fr['points'])} points, radius {rr.min():.1f}-{rr.max():.1f})",
          len(fr["points"]) == T.FREE_ANCHORS and abs(rr.min() - 210) < 1.5 and abs(rr.max() - 210) < 1.5)
    ang = T._screen_angle(fr, Pf)
    check("…and runs clockwise like the others", float(np.median(np.diff(np.unwrap(np.radians(ang))))) > 0)
    loz = np.full((600, 600, 3), 60, np.uint8)
    diamond = np.array([[300, 90], [470, 300], [300, 510], [130, 300]], np.int32)
    cv2.fillPoly(loz, [diamond], (220, 190, 90), cv2.LINE_AA)
    lf = T.find_outline(loz, "free")
    A = T.free_anchors(lf)

    def to_diamond(p):
        x, y = abs(p[0] - 300) / 170.0, abs(p[1] - 300) / 210.0
        return abs(x + y - 1) * 150
    off = max(to_diamond(p) for p in A)
    check(f"a lozenge is found as a free shape (every point within {off:.1f} px of its edge)", off < 6)
    B = A.copy()
    B[3] += (12, -7)
    moved = T.free_anchors(T.with_anchors(lf, B))
    check("moving one point moves only that point", np.allclose(moved[3], A[3] + (12, -7), atol=1e-6)
          and np.allclose(np.delete(moved, 3, axis=0), np.delete(A, 3, axis=0), atol=1e-6))
    mf = T.mirrored(lf, 600)
    check("a free shape mirrors with the picture",
          np.allclose(T.free_anchors(mf)[:, 0], 599 - A[:, 0], atol=1e-6))
    sl = T.unwrap(loz, T.legend_maps(lf, 30))
    check("its band unrolls into a strip", sl.shape[0] == 30 and sl.shape[1] > 900)

    print("\nFigure")
    fig = T.legend_figure(loz, lf, sl)
    check(f"the object above its strip, one picture the strip's width ({fig.shape[1]} x {fig.shape[0]})",
          fig.shape[1] == sl.shape[1] and fig.shape[0] > sl.shape[0] + 400
          and np.array_equal(fig[-sl.shape[0]:], sl))

    print("\nSeveral objects on a page")
    rng = np.random.default_rng(1)
    page = np.clip(np.full((1400, 1800, 3), (225, 215, 190), np.float32)
                   + rng.normal(0, 6, (1400, 1800, 3)), 0, 255).astype(np.uint8)
    for i in range(30):
        cv2.putText(page, "lorem ipsum dolor sit amet consectetur", (60, 60 + i * 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 50, 40), 2)
    cv2.rectangle(page, (1650, 0), (1800, 1400), (15, 15, 15), -1)            # a black copy-stand
    truth = [(400, 1000, 170, 170), (950, 1050, 130, 200), (1450, 1100, 90, 90)]
    cv2.circle(page, (400, 1000), 170, (40, 40, 150), -1)
    cv2.circle(page, (400, 1000), 140, (20, 20, 90), 4)
    cv2.ellipse(page, (950, 1050), (130, 200), 15, 0, 360, (30, 70, 140), -1)
    cv2.circle(page, (1450, 1100), 90, (150, 150, 150), -1)
    objs = T.find_objects(page)
    hit = [any(abs(o["cx"] - x) < 0.2 * r and abs(o["cy"] - y) < 0.2 * r for o in objs) for x, y, r, _b in truth]
    check(f"two seals and a coin on a written page, beside a black stand: {len(objs)} found, "
          f"each one ringed", len(objs) == 3 and all(hit), [(round(o["cx"]), round(o["cy"])) for o in objs])
    check("…in reading order, left to right", [round(o["cx"], -2) for o in objs] == [400, 900, 1400]
          or objs[0]["cx"] < objs[1]["cx"] < objs[2]["cx"])
    o = objs[1]
    x, y, cw, ch = T.object_crop(o, page.shape[1], page.shape[0])
    f = T.find_outline(page[y:y + ch, x:x + cw], "oval",
                       near=(o["cx"] - x, o["cy"] - y, math.sqrt(o["a"] * o["b"])))
    check(f"…and the oval refined in the part round it ({f['a']:.0f} x {f['b']:.0f}, {f['turn']:.0f}°)",
          abs(f["a"] - 130) < 10 and abs(f["b"] - 200) < 12 and abs(f["turn"] - 15) < 5)
    blank = np.full((600, 800, 3), 200, np.uint8)
    check("a blank page has none", T.find_objects(blank) == [])
    drawn = T.draw_objects(page, objs, 1)
    check("they are drawn ringed and numbered", drawn.shape[1] == 1600 and not np.array_equal(
        drawn, cv2.resize(page, (1600, drawn.shape[0]), interpolation=cv2.INTER_AREA)))

    print("\nScale and record")
    band = T.legend_scale(full, mm_per_px=0.05)
    check("the scale band is the strip's width", band.shape[1] == full["map_x"].shape[1])
    rc = T.outline_record(circ, 40, mm_per_px=0.1, start_deg=180)
    ro = T.outline_record(dict(v, shape="oval"), 40, mm_per_px=0.1)
    check(f"a circle's diameter ({rc['diameter_mm']} mm), an oval's height x width "
          f"({ro['height_mm']} x {ro['width_mm']} mm)", rc["diameter_mm"] == 42.0
          and ro["height_mm"] == 44.0 and ro["width_mm"] == 24.0 and rc["start_deg"] == 180)

    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
