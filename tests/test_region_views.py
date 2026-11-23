"""Views and exports that reread the photographs show a region solve's region.

Reprocessing with a crop solves a part of the photographs.  A few views read
the photographs again from disk (Colour ICA, RGB-PS R-B Diff), and they used
to cut them only to the magnifier's crop, so after a region solve they showed
the whole photograph again.  On the manuscript opening in test images/, at
full and at half working size:

* each of those views looks like the region, not the whole frame;
* the magnifier inside a region solve looks at the right part of it;
* load_stack (Unmix, the engine stacks) cuts the photographs to the region
  when the solve was at a reduced size, where the crop is in smaller pixels.

    python tests/test_region_views.py
"""
import glob
import os
import sys

import cv2
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


def grey(a):
    a = np.asarray(a, np.float32)
    return a.mean(axis=2) if a.ndim == 3 else a


def likeness(a, b):
    """How alike two pictures' edges are (|correlation| of gradient size),
    after bringing *b* to *a*'s size: signs and tones of a view do not matter."""
    a, b = grey(a), grey(b)
    b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    ga = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1)).ravel()
    gb = cv2.magnitude(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)).ravel()
    return abs(float(np.corrcoef(ga, gb)[0, 1]))


def main():
    from helio.application.export_images import load_stack
    from helio.application.pipeline import SolveRequest, solve, store
    from helio.application.render import render
    from helio.data_manager import DataManager
    from helio.domain.settings import ProcessingSettings
    paths = sorted(glob.glob(os.path.join(test_images(PASS, FAIL), "*.jpeg")))
    photo = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
    H, W = photo.shape[:2]
    box = (int(0.55 * W), int(0.30 * H), int(0.85 * W), int(0.70 * H))     # part of the right-hand page
    region = photo[box[1]:box[3], box[0]:box[2]]
    light = dict(lx=0.3, ly=-0.3, lz=0.9, gain=1.0)

    for scale in (1.0, 0.5):
        print(f"\nA region solved at {scale * 100:.0f}% working size")
        ps = ProcessingSettings(normal_engine="Uncalibrated PCA", work_scale=scale)
        r = solve(SolveRequest(paths=paths, lights=None, settings=ps, crop=box, crop_ref_size=(W, H)))
        D = DataManager()
        store(r, D)
        D.plate_paths = list(paths)
        check(f"the solve is the region ({r.w} x {r.h})", abs(r.w / r.h - region.shape[1] / region.shape[0]) < 0.02)
        for view in ("Colour ICA (IC1-3)", "RGB-PS R-B Diff"):
            out = render(view, D, light, r.h, r.w, False, (0, 0, r.w, r.h))
            on_region, on_whole = likeness(out, region), likeness(out, photo)
            check(f"{view} shows the region ({on_region:.2f} alike, the whole frame {on_whole:.2f})",
                  on_region > on_whole + 0.1, (on_region, on_whole))
        # the magnifier: the region's right half
        half = (r.w // 2, 0, r.w, r.h)
        out = render("RGB-PS R-B Diff", D, light, r.h, r.w - r.w // 2, False, half)
        right = region[:, region.shape[1] // 2:]
        left = region[:, :region.shape[1] // 2]
        a, b = likeness(out, right), likeness(out, left)
        check(f"the magnifier on a region looks at its own part ({a:.2f} vs the other half {b:.2f})",
              a > b + 0.1, (a, b))
        stack = load_stack(paths[:1], r.w, r.h, crop=r.crop, base=(r.base_w, r.base_h))
        a, b = likeness(stack[0], region), likeness(stack[0], photo)
        check(f"load_stack cuts the photographs to the region ({a:.2f} vs the whole frame {b:.2f})",
              a > b + 0.1, (a, b))

    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
