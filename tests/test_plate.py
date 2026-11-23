"""The publication plate (REP-010): one A3 page that stands on its own.

GUI-free, on a made-up surface, tall and wide:

* the PDF and PNG are written, A3 portrait, every panel lettered;
* no caption runs over a picture or another caption, and everything keeps
  inside the margins: the normals' words sit beside the map, not under it
  where they ran into the row below;
* the raking light comes from the top half of the picture (relief lit from
  below reads inside out), at gain 1 through a display gamma;
* the quality line reads the dome score's total, not a key it lacks.

    python tests/test_plate.py
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


class Surface:
    """What plate() reads from a solve, for a bumpy plaque *w* × *h*."""

    def __init__(self, w, h, sigma=True):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        z = (np.sin(xx / 23.0) * np.cos(yy / 31.0) * 4 + 0.002 * (xx - w / 2) ** 2 / w * 10)
        gy, gx = np.gradient(z)
        n = np.dstack([-gx, gy, np.ones_like(gx)])          # +Y up: north
        n /= np.linalg.norm(n, axis=2, keepdims=True)
        self.h = self.ph = h
        self.w = self.pw = w
        self.albedo_rgb_full = np.dstack([np.linspace(60, 200, w)[None].repeat(h, 0)] * 3
                                         ).astype(np.uint8)
        self.normals_full = n.astype(np.float32)
        self.normals_proxy = None
        self.depth_vis_p = z.astype(np.float32)
        self.depth_proxy_p = None
        self.qc_sigma_proxy = (np.abs(gx) * 20 + 1).astype(np.float32) if sigma else None


def ring(n=8, el=(30, 45)):
    az = np.radians(np.arange(n) * 360.0 / n)
    e = np.radians(np.linspace(el[0], el[1], n))
    return np.stack([np.sin(az) * np.cos(e), np.cos(az) * np.cos(e), np.sin(e)], 1)


def overlaps(a, b, slack=0.0015):
    return not (a[2] <= b[0] + slack or b[2] <= a[0] + slack
                or a[3] <= b[1] + slack or b[3] <= a[1] + slack)


def layout_problems(res):
    """Every caption against every picture and every other caption, and the margins."""
    M = res["margin"]
    out = []
    boxes = res["boxes"]
    caps = [b for b in boxes if b[0] == "caption"]
    solid = [b for b in boxes if b[0] in ("image", "text")]
    for i, (_k, name, box) in enumerate(caps):
        for _k2, other, ob in solid:
            if overlaps(box, ob):
                out.append(f"caption ({name}) over {other}")
        for _k2, other, ob in caps[i + 1:]:
            if overlaps(box, ob):
                out.append(f"caption ({name}) over caption ({other})")
    for _k, name, (x0, y0, x1, y1) in boxes:
        if x0 < M - 0.006 or x1 > 1 - M + 0.006 or y0 < 0.012 or y1 > 0.99:
            out.append(f"{name} outside the margins {x0:.3f}–{x1:.3f}, {y0:.3f}–{y1:.3f}")
    return out


def main():
    from helio.application import plate_doc as P
    print("=" * 60)
    print("The publication plate")
    print("=" * 60)

    print("\nLight from above")
    check("a raking light found from the south is drawn from the north",
          abs(P.from_above(179.0) - 359.0) < 1e-9)
    check("…and one from the north-west is left alone", P.from_above(300.0) == 300.0)
    check("the compass words", (P.compass(359), P.compass(135), P.compass(292)) ==
          ("north", "south-east", "west"))
    from helio.domain.view_controls import from_angles
    check("north is up the picture in the screen frame the views draw in",
          from_angles(P.from_above(179.0), 20.0)[1] < 0)

    tmp = tempfile.mkdtemp(prefix="helio_plate_")
    for label, (w, h), facts, sigma in (
        ("a tall object, relative heights",
         (420, 600), P.Facts(title="Folio 12r", frames=8, lights=ring(), light_note="chrome ball",
                             mm_per_px=0.02, normal_engine="Linear Least Squares",
                             depth_engine="Fast Poisson (FFT)", photo_tone="stored",
                             notes=["the two balls disagree by 4°"]), True),
        ("a wide object, measured heights, a colour check, no uncertainty map",
         (900, 420), P.Facts(title="Inscribed stone, fragment B", frames=24, lights=ring(24, (15, 60)),
                             mm_per_px=0.08, z_mm_per_unit=0.05, photo_tone="srgb",
                             colour=dict(mean=1.8, max=4.2, worst="cyan"), drift_px=0.4), False),
        ("no lights known, no scale",
         (640, 640), P.Facts(title="Seal impression"), True),
    ):
        print(f"\n{label.capitalize()}")
        D = Surface(w, h, sigma)
        calls = []

        def draw(eng, over, D=D, calls=calls):
            calls.append((eng, dict(over)))
            g = np.clip(D.normals_full[..., 2] * 255, 0, 255).astype(np.uint8)
            return np.dstack([g] * 3)

        base = os.path.join(tmp, label.split(",")[0].replace(" ", "_"))
        res = P.plate(base, D=D, facts=facts, draw=draw, dpi=60,
                      dome_score=dict(total=72.4, grade="B") if facts.lights is not None else None)
        import cv2
        png = cv2.imread(res["png"])
        check("PDF and PNG written, A3 portrait",
              os.path.getsize(res["pdf"]) > 10_000 and png is not None
              and abs(png.shape[0] / png.shape[1] - 16.54 / 11.69) < 0.01,
              None if png is None else png.shape)
        want = {"a", "b", "c", "c-key", "d", "e", "f", "g", "h", "i", "k", "j"}
        check("every panel lettered (a to k, and the section j)", want <= set(res["panels"]),
              sorted(want - set(res["panels"])))
        probs = layout_problems(res)
        check("no caption over a picture or another caption; all inside the margins", not probs,
              probs[:6])
        cap_c = [b for k, n, b in res["boxes"] if k == "caption" and n == "c"][0]
        img_c = [b for k, n, b in res["boxes"] if k == "image" and n == "c"][0]
        check("the normals' caption sits beside the map, its top level with the map's",
              cap_c[0] >= img_c[2] and abs(cap_c[3] - img_c[3]) < 0.01, (cap_c, img_c))
        rake = [o for e, o in calls if e == "Standard PBR"]
        check("the raking view: gain 1, display gamma, a low light from the top half",
              rake and rake[0].get("gain") == 1.0 and rake[0].get("gamma") == 2.2
              and rake[0]["ly"] <= 1e-9 and rake[0]["lz"] < 0.5, rake)
        check("the four readings: raking, exaggerated, multi-scale relief, curvature",
              [e for e, _o in calls][:4] == ["Standard PBR", "Exaggerated Shading",
                                             "Multi-Scale Relief Model (MSRM)",
                                             "Surface Curvature (Scratches)"],
              [e for e, _o in calls])
        if not sigma:
            check("without an uncertainty map the light coverage stands in",
                  any(e == "Light Coverage Map" for e, _o in calls))

    import matplotlib
    check("the plate's type and settings stay on the plate (the rest of HELIO's figures keep theirs)",
          matplotlib.rcParams["mathtext.fontset"] != "custom"
          and "Helvetica Neue" not in matplotlib.rcParams["font.family"],
          (matplotlib.rcParams["mathtext.fontset"], matplotlib.rcParams["font.family"]))

    print("\nThe quality line")
    q = P.quality_notes(P.Facts(drift_px=0.4), dict(total=72.4, grade="B"))
    check("the dome score's total and grade, not 0/100", "dome coverage 72/100 (grade B)" in q, q)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
