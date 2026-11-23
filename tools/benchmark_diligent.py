"""Helio's normal solvers on the DiLiGenT benchmark (Shi et al., CVPR 2016 / TPAMI 2019).

DiLiGenT: ten objects, 96 calibrated lights each, 16-bit linear photographs and
ground-truth normals from a laser scan.  The usual measure is the mean angular
error (degrees) between the solved and true normals inside each object's mask.

    python tools/benchmark_diligent.py [path/to/DiLiGenT]     (default ~/Downloads/DiLiGenT)

Writes a Markdown table to stdout and the figures to docs/benchmark_diligent.json.
What is run:

* Calibrated: the given light directions and intensities, the photographs in
  linear light (as Helio reads a RAW frame), each solver Helio offers.
* Semi-calibrated: the directions only; Helio estimates each lamp's strength
  from the photographs (robust_ps.light_strengths), then least squares.
* Tone: the same photographs saved as 8-bit sRGB, as a camera JPEG would be,
  solved as stored (Helio's default) and decoded to linear first.
* The published results that ship with DiLiGenT (estNormalNonLambert), scored
  here with the same mask and measure, for comparison.

Every image is used (all 96 lights), as in the published table.  Each frame is
divided by its light's RGB intensity and averaged to grey; the stack is scaled
so the 99.5th percentile inside the mask reads 230 of 255 (a well-exposed
frame), which is where the robust solvers' thresholds are defined.
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io  # noqa: E402

OBJECTS = ["ball", "cat", "pot1", "bear", "pot2", "buddha", "goblet", "reading", "cow", "harvest"]
SOLVERS = ["Linear Least Squares", "Shadow-Rejection LS", "Mid-Tone Weighted LS",
           "IRLS (Metallic/Specular)", "Iterative Specular Separation", "Sparse-Outlier LS (RPCA)"]
PUBLISHED = {"l2": "Least squares (Woodham 1980)", "ICCV05Goldman": "Goldman 2005",
             "CVPR08Alldrin": "Alldrin 2008", "CVPR10Higo": "Higo 2010", "ACCV10Wu": "Wu 2010",
             "ECCV12Shi": "Shi 2012", "CVPR12Shi": "Shi 2012b", "CVPR12Ikehata": "Ikehata 2012",
             "CVPR14Ikehata": "Ikehata 2014"}


def load(base, obj):
    d = os.path.join(base, "pmsData", f"{obj}PNG")
    names = open(os.path.join(d, "filenames.txt")).read().split()
    L = np.loadtxt(os.path.join(d, "light_directions.txt")).astype(np.float32)
    E = np.loadtxt(os.path.join(d, "light_intensities.txt")).astype(np.float32)   # R G B
    mask = cv2.imread(os.path.join(d, "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    gt = scipy.io.loadmat(os.path.join(d, "Normal_gt.mat"))["Normal_gt"].astype(np.float32)
    ys, xs = np.nonzero(mask)
    cal = np.empty((len(names), len(ys)), np.float32)       # intensities divided out
    unc = np.empty_like(cal)                                 # as photographed
    for i, nm in enumerate(names):
        im = cv2.imread(os.path.join(d, nm), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32)
        px = im[ys, xs]                                      # (k, 3) RGB
        cal[i] = (px / E[i][None, :]).mean(axis=1)
        unc[i] = px.mean(axis=1)
    for a in (cal, unc):
        a *= 230.0 / max(float(np.percentile(a, 99.5)), 1e-6)
    return L, cal, unc, gt[ys, xs], mask, (ys, xs)


def ang_err(n, gt):
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(n * gt, axis=-1), -1.0, 1.0)))


def solve(engine, L, I, I_ref=None):
    from helio.application.pipeline import _solve_normals
    from helio.domain.settings import ProcessingSettings
    k = I.shape[1]
    n, _ = _solve_normals(engine, I, I.reshape(len(I), k, 1), L, k, 1, ProcessingSettings(),
                          flat_ref=I if I_ref is None else I_ref)
    return np.asarray(n, np.float32).reshape(k, 3)


def main(base):
    from helio.core.radiometry import linear_to_srgb, srgb_to_linear
    from helio.core.robust_ps import light_strengths
    rows, timing = {}, {}
    for obj in OBJECTS:
        t0 = time.time()
        L, cal, unc, gt, mask, (ys, xs) = load(base, obj)
        r = rows.setdefault(obj, {})
        for s in SOLVERS:
            t = time.time()
            e = ang_err(solve(s, L, cal), gt)
            r[s] = (float(e.mean()), float(np.median(e)))
            timing.setdefault(s, 0.0)
            timing[s] += time.time() - t
        # semi-calibrated: the lamps' strengths from the photographs
        st = light_strengths(unc, L)
        if st is not None:
            semi = unc / np.asarray(st["strength"], np.float32)[:, None]
            e = ang_err(solve("Linear Least Squares", L, semi), gt)
            r["Semi-calibrated LS"] = (float(e.mean()), float(np.median(e)))
        e = ang_err(solve("Linear Least Squares", L, unc), gt)
        r["LS, strengths ignored"] = (float(e.mean()), float(np.median(e)))
        # tone: an 8-bit sRGB "JPEG" of the calibrated stack
        enc = np.round(linear_to_srgb(np.clip(cal, 0, 255))).astype(np.uint8)
        e = ang_err(solve("Linear Least Squares", L, enc.astype(np.float32)), gt)
        r["8-bit sRGB, as stored"] = (float(e.mean()), float(np.median(e)))
        e = ang_err(solve("Linear Least Squares", L, srgb_to_linear(enc)), gt)
        r["8-bit sRGB, decoded"] = (float(e.mean()), float(np.median(e)))
        # published
        for key, label in PUBLISHED.items():
            p = os.path.join(base, "estNormalNonLambert", f"{obj}PNG_Normal_{key}.mat")
            if os.path.exists(p):
                m = scipy.io.loadmat(p)
                arr = next(v for k, v in m.items() if not k.startswith("__"))
                e = ang_err(np.asarray(arr, np.float32)[ys, xs], gt)
                r[f"[published] {label}"] = (float(e.mean()), float(np.median(e)))
        print(f"  {obj}: {mask.sum()} px, {time.time() - t0:.0f} s", file=sys.stderr)
    methods = list(rows[OBJECTS[0]].keys())
    out = ["| Method | " + " | ".join(OBJECTS) + " | **Mean** |",
           "|---|" + "---:|" * (len(OBJECTS) + 1)]
    for m in methods:
        vals = [rows[o][m][0] for o in OBJECTS if m in rows[o]]
        cells = [f"{rows[o][m][0]:.2f}" if m in rows[o] else "–" for o in OBJECTS]
        out.append(f"| {m} | " + " | ".join(cells) + f" | **{np.mean(vals):.2f}** |")
    print("\n".join(out))
    print("\nSeconds per solver over all ten objects: " +
          ", ".join(f"{s} {t:.0f}" for s, t in timing.items()))
    with open(os.path.join(ROOT, "docs", "benchmark_diligent.json"), "w") as f:
        json.dump(dict(mean_median_deg=rows, seconds=timing), f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/DiLiGenT"))
