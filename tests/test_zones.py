"""Inscription zones: where to look, never the whole frame.

On the manuscript opening in test images/ (eight photographs, solved with
Uncalibrated PCA), in every surface type:

* no zone is (most of) the frame.  The 2026-10-29 bug: every source added a
  little everywhere, a fixed mark level was passed by nearly the whole frame,
  the blocks merged into one, and the specks left beside it were kept instead
  of the finer grouping - so the one zone was the image itself, drawn on its
  border, and the user saw "no difference";
* the right-hand page's block of writing is inside a zone;
* no zone sits inside another.

And the boxes are drawn heavier on a larger picture, so they still show when
a 9000 px capture is shown in a small preview.

    python tests/test_zones.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from tools.make_golden import test_images  # noqa: E402

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    from helio.application.pipeline import SolveRequest, solve, store
    from helio.application.research import INSCRIPTION_MODES, draw_zones, inscription_zones
    from helio.data_manager import DataManager
    from helio.domain.settings import ProcessingSettings
    print("Inscription zones")
    paths = sorted(glob.glob(os.path.join(test_images(PASS, FAIL), "*.jpeg")))
    r = solve(SolveRequest(paths=paths, lights=None,
                           settings=ProcessingSettings(normal_engine="Uncalibrated PCA")))
    D = DataManager()
    store(r, D)
    H, W = D.depth_vis_p.shape
    # the right-hand page's writing, as a fraction of the frame (x0, y0, x1, y1)
    tx0, ty0, tx1, ty1 = (int(0.20 * W), int(0.22 * H), int(0.80 * W), int(0.80 * H))

    def covers(b):
        ix = max(0, min(b[0] + b[2], tx1) - max(b[0], tx0))
        iy = max(0, min(b[1] + b[3], ty1) - max(b[1], ty0))
        return ix * iy >= 0.8 * (tx1 - tx0) * (ty1 - ty0)

    for mode in INSCRIPTION_MODES:
        boxes, _ = inscription_zones(D.depth_vis_p, D.normals_full, getattr(D, "spec_var", None),
                                     D.albedo_rgb_full, mode=mode, sens=0.7, max_zones=24)
        whole = [b for b in boxes if b[2] * b[3] >= 0.7 * W * H]
        nested = [b for b in boxes for o in boxes if o is not b and o[0] <= b[0] and o[1] <= b[1]
                  and b[0] + b[2] <= o[0] + o[2] and b[1] + b[3] <= o[1] + o[3]]
        check(f"{mode}: {len(boxes)} zones, none the frame", boxes and not whole, boxes[:3])
        check(f"{mode}: the page's writing is inside a zone", any(covers(b) for b in boxes), boxes[:3])
        check(f"{mode}: no zone inside another", not nested, nested[:2])

    small = draw_zones(np.zeros((500, 500, 3), np.uint8), [(100, 100, 200, 200)])
    big = draw_zones(np.zeros((8000, 8000, 3), np.uint8), [(1600, 1600, 3200, 3200)])
    t_small = int((small[200, 80:120].max(axis=1) > 0).sum())             # the left edge, mid-height
    t_big = int((big[3200, 1500:1700].max(axis=1) > 0).sum())
    check(f"the box line grows with the picture ({t_small} px at 500, {t_big} px at 8000)",
          t_small >= 2 and t_big >= 5 * t_small, (t_small, t_big))

    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
