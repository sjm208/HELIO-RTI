"""The optional reduced working size (ProcessingSettings.work_scale / HELIO_WORK_SCALE).

    python tests/test_work_scale.py
"""
import dataclasses
import os
import sys
import tempfile
import tracemalloc

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools import make_golden as G                                         # noqa: E402
from helio.application import pipeline as P                           # noqa: E402
from helio.domain.settings import ProcessingSettings                  # noqa: E402
from helio.data_manager import DataManager                            # noqa: E402
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


def angle_deg(a, b):
    d = np.clip((a * b).sum(-1), -1, 1)
    return np.degrees(np.arccos(d))


def main():
    os.environ.pop("HELIO_WORK_SCALE", None)
    os.environ.pop("HELIO_MEMORY_GB", None)
    G.H, G.W = 480, 640
    paths, L = G.render_stack(tempfile.mkdtemp(prefix="helio_ws_"))
    truth = G.analytic_surface()[1]
    ps = ProcessingSettings(photo_tone="linear")

    print("Settings")
    check("work_scale defaults to 1.0 and is in the settings dict", ProcessingSettings().work_scale == 1.0
          and ProcessingSettings().to_dict()["work_scale"] == 1.0)
    check("an old settings dict without it loads at full size", ProcessingSettings.from_dict({"normal_engine": "Linear Least Squares"}).work_scale == 1.0)
    check("it round-trips", ProcessingSettings.from_dict(ProcessingSettings(work_scale=0.5).to_dict()).work_scale == 0.5)
    check("effective scale: the setting, clamped to 0.1..1 (>= 0.995 means full size)",
          P.effective_work_scale(ProcessingSettings(work_scale=0.5)) == 0.5 and P.effective_work_scale(ProcessingSettings(work_scale=1.0)) == 1.0
          and P.effective_work_scale(ProcessingSettings(work_scale=0.05)) == 1.0 and P.effective_work_scale(ProcessingSettings(work_scale=0.999)) == 1.0)
    os.environ["HELIO_WORK_SCALE"] = "0.25"
    check("HELIO_WORK_SCALE overrides the setting", P.effective_work_scale(ProcessingSettings(work_scale=0.5)) == 0.25)
    os.environ["HELIO_WORK_SCALE"] = "banana"
    check("a bad HELIO_WORK_SCALE means full size", P.effective_work_scale(ProcessingSettings()) == 1.0)
    os.environ.pop("HELIO_WORK_SCALE")

    print("A reduced solve")
    full = P.solve(P.SolveRequest(paths=paths, lights=L, settings=ps))
    tracemalloc.start()
    half = P.solve(P.SolveRequest(paths=paths, lights=L, settings=dataclasses.replace(ps, work_scale=0.5)))
    _, peak_half = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    P.solve(P.SolveRequest(paths=paths, lights=L, settings=ps))
    _, peak_full = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    check("full size is unchanged: 480 x 640, work_scale 1, source size recorded", (full.h, full.w) == (480, 640)
          and full.work_scale == 1.0 and full.source_size == (640, 480))
    check("at 50% the result is 240 x 320 and says so", (half.h, half.w) == (240, 320) and half.work_scale == 0.5
          and half.source_size == (640, 480) and half.normals_full.shape == (240, 320, 3))
    check(f"every array is at the reduced size (albedo, depth, PTM, previews)", half.albedo_rgb_full.shape[:2] == (240, 320)
          and half.depth_poisson.shape == (240, 320) and half.ptm_coeffs.shape == (6, 240 * 320))
    check(f"peak memory falls to {peak_half / peak_full:.2f} of full size (under 0.45)", peak_half < 0.45 * peak_full, (peak_half, peak_full))
    check("the result keeps the user's settings (not the reduced ones)", half.settings.work_scale == 0.5 and half.settings.flash_ref == "")
    small_truth = cv2.resize(truth, (320, 240), interpolation=cv2.INTER_AREA)
    small_truth /= np.linalg.norm(small_truth, axis=-1, keepdims=True)
    err_half = float(np.median(angle_deg(half.normals_full, small_truth)))
    err_full = float(np.median(angle_deg(full.normals_full, truth)))
    check(f"its normals match the surface as well as the full solve does (median error {err_half:.2f} vs {err_full:.2f} degrees)",
          err_half < err_full + 1.0 and err_half < 4.0, (err_half, err_full))
    d = angle_deg(cv2.resize(full.normals_full, (320, 240), interpolation=cv2.INTER_AREA) /
                  np.maximum(np.linalg.norm(cv2.resize(full.normals_full, (320, 240), interpolation=cv2.INTER_AREA), axis=-1, keepdims=True), 1e-9),
                  half.normals_full)
    check(f"and agree with the full solve reduced to the same size (median {float(np.median(d)):.2f} degrees)", float(np.median(d)) < 3.0)
    D = DataManager()
    P.store(half, D)
    check("store() puts the working size on the DataManager", D.work_scale == 0.5 and D.source_size == (640, 480) and (D.h, D.w) == (240, 320))
    D.reset() if hasattr(D, "reset") else None
    check("result_mm_per_px: a reduced pixel is twice as big", abs(P.result_mm_per_px(half, 0.05) - 0.10) < 1e-12
          and P.result_mm_per_px(full, 0.05) == 0.05 and P.result_mm_per_px(half, None) is None)

    print("Pixel-unit inputs are converted")
    base = P.SolveRequest(paths=paths, lights=L, settings=dataclasses.replace(ps, flash_ref="100,200,300,400", flat_areas="10,20,110,120;200,200,400,300",
                                                                           near_field=True),
                          crop=(100, 100, 300, 200), mm_per_px=0.04, light_origin=(320.0, 240.0),
                          flat_ref_normals=np.dstack([np.zeros((480, 640)), np.zeros((480, 640)), np.ones((480, 640))]).astype(np.float32))
    red = P.reduced_request(base, 0.5, (640, 480), (320, 240))
    check("flash box and flat areas are halved", red.settings.flash_ref == "50,100,150,200" and red.settings.flat_areas == "5,10,55,60;100,100,200,150", (red.settings.flash_ref, red.settings.flat_areas))
    check("the light origin is halved and the scale doubled", red.light_origin == (160.0, 120.0) and abs(red.mm_per_px - 0.08) < 1e-12)
    check("the flat reference is resized to the reduced image", red.flat_ref_normals.shape[:2] == (240, 320))
    check("the crop keeps the full-size pixels it was drawn in, with that size as its reference", red.crop == (100, 100, 300, 200) and red.crop_ref_size == (640, 480))
    check("the original request is not changed", base.mm_per_px == 0.04 and base.settings.flash_ref == "100,200,300,400")
    cropped = P.solve(P.SolveRequest(paths=paths, lights=L, settings=dataclasses.replace(ps, work_scale=0.5), crop=(100, 100, 300, 200)))
    check("a crop drawn on the photograph lands on the same region at half size (result 100 x 50, crop (50, 50, 150, 100))",
          (cropped.w, cropped.h) == (100, 50) and cropped.crop == (50, 50, 150, 100), (cropped.w, cropped.h, cropped.crop))

    print("The plan, before anything is read")
    seen = []
    os.environ["HELIO_MEMORY_GB"] = "0.9"
    G.H, G.W = 1000, 1400
    big_paths, big_L = G.render_stack(tempfile.mkdtemp(prefix="helio_ws_big_"), n_lights=4)
    P.solve(P.SolveRequest(paths=big_paths, lights=big_L, settings=ps),
            progress=lambda stage, frac=None, detail="": seen.append((frac, detail)))
    first_msgs = [d for f, d in seen if d and d.startswith("Memory:")]
    check("a too-small machine is told before loading starts (the first progress call)", seen and seen[0][1].startswith("Memory:"), seen[:2])
    p = M.plan_solve(10000, 10000, 8)
    os.environ["HELIO_MEMORY_GB"] = "32"
    p32 = M.plan_solve(10000, 10000, 8)
    os.environ["HELIO_MEMORY_GB"] = "16"
    p16 = M.plan_solve(10000, 10000, 8)
    check("100 MP x 8 on 32 GB: tight, and 75% size is offered", p32.level == "tight" and p32.suggested_scale == 0.75, (p32, p32.suggested_scale))
    check("100 MP x 8 on 16 GB: too big, and a smaller size is offered that fits", p16.level == "too_big" and p16.suggested_scale in (0.5, 0.35, 0.25)
          and M.estimate_solve_bytes(int(10000 * p16.suggested_scale), int(10000 * p16.suggested_scale), 8) <= 0.85 * p16.available, p16.suggested_scale)
    check("the message names the setting and the variable and warns about units",
          "HELIO_WORK_SCALE" in p16.message and "reduced pixels" in p16.message, p16.message)
    os.environ["HELIO_MEMORY_GB"] = "64"
    check("a roomy machine suggests nothing", M.plan_solve(10000, 10000, 8).suggested_scale is None)
    os.environ.pop("HELIO_MEMORY_GB")

    print("Quick look still works (its frames are already small)")
    from helio.application import quick_look as QL
    try:
        r, s = QL.quick_solve(P.SolveRequest(paths=paths, lights=L, settings=dataclasses.replace(ps, work_scale=0.5)))
        check("quick_solve ignores the working size (the frames are already reduced)", r.work_scale == 1.0 and max(r.h, r.w) <= 1000)
    finally:
        QL.forget()

    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
