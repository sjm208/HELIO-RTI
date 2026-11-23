"""Helio's normals and depth on DiLiGenT-Pi (Wang et al., ICCV 2023): near-flat objects with fine relief.

DiLiGenT-Pi: 30 objects (coins, medals, carvings: the scale and kind of a punch or a seal), 100 calibrated
lights, 16-bit linear photographs, normals from a scan.  What is measured:

* Normals: mean angular error inside the mask (degrees), for each solver, calibrated and linear; and
  "flatness": the solve's mean tilt from the camera over the true mean tilt (1.0 = as steep as the
  surface; below it, a relief that comes out shallower than it is, whatever the Height slider says).
* Tone: the photographs as an 8-bit sRGB camera JPEG would hold them, solved as stored and decoded.
* Depth: every depth method on the true normals and on the least-squares ones, the frame whole and the
  object alone (its mask).  The reference is the true normals integrated by least squares with a free
  boundary on the object alone.  After taking out the best plane: "relief" = the depth's spread over the
  reference's (1.0 = as deep), "detail" = the same for what is finer than 8 px (the letter forms of a
  punch), "corr" = how alike the shapes are (1.0 = the same).

    python tools/benchmark_diligent_pi.py [path/to/DiLiGenT-Pi] [--objects=A,B] [--scale=0.5] [--quick]

--scale: the photographs reduced (0.5: a quarter of the pixels, the same ranking); --quick: least squares
and the depth on the object alone only.

Writes a Markdown report to stdout and the figures to docs/benchmark_diligent_pi.json.
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

SOLVERS = ["Linear Least Squares", "Shadow-Rejection LS", "Mid-Tone Weighted LS",
           "IRLS (Metallic/Specular)", "Iterative Specular Separation", "Sparse-Outlier LS (RPCA)"]
DETAIL_PX = 8.0


def objects(base):
    d = os.path.join(base, "DiLiGenT-Pi_release_png")
    return sorted(o for o in os.listdir(d) if os.path.isdir(os.path.join(d, o)))


def load(base, obj, scale=1.0):
    """(L, calibrated stack, as-photographed stack, true normals (h, w, 3), mask)."""
    d = os.path.join(base, "DiLiGenT-Pi_release_png", obj)
    names = open(os.path.join(d, "filenames.txt")).read().split()
    L = np.loadtxt(os.path.join(d, "light_directions.txt")).astype(np.float32)
    E = np.loadtxt(os.path.join(d, "light_intensities.txt")).astype(np.float32)
    g = os.path.join(base, "DiLiGenT-Pi_gt", obj)
    mask = cv2.imread(os.path.join(g, "mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    gt = scipy.io.loadmat(os.path.join(g, "Normal_gt.mat"))["Normal_gt"].astype(np.float32)
    if scale != 1.0:
        h, w = mask.shape
        size = (int(round(w * scale)), int(round(h * scale)))
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
        gt = cv2.resize(gt, size, interpolation=cv2.INTER_AREA)
    gt /= np.maximum(np.linalg.norm(gt, axis=2, keepdims=True), 1e-9)
    h, w = mask.shape
    cal = np.empty((len(names), h, w), np.float32)
    unc = np.empty_like(cal)
    for i, nm in enumerate(names):
        im = cv2.imread(os.path.join(d, nm), cv2.IMREAD_UNCHANGED)[..., ::-1].astype(np.float32)
        if scale != 1.0:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        cal[i] = (im / E[i][None, None, :]).mean(axis=2)
        unc[i] = im.mean(axis=2)
    for a in (cal, unc):
        a *= 230.0 / max(float(np.percentile(a[:, mask], 99.5)), 1e-6)
    return L, cal, unc, gt, mask


def solve(engine, L, stack):
    """Normals (h, w, 3) from a stack (n, h, w), as Process solves them."""
    from helio.application.pipeline import _solve_normals
    from helio.domain.settings import ProcessingSettings
    n_, h, w = stack.shape
    flat = stack.reshape(n_, h * w)
    nrm, _ = _solve_normals(engine, flat, stack.reshape(n_, h, w), L, h, w, ProcessingSettings(), flat_ref=flat)
    return np.asarray(nrm, np.float32).reshape(h, w, 3)


def ang_err(n, gt, mask):
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip(np.sum(n[mask] * gt[mask], axis=-1), -1.0, 1.0)))


def tilt(n, mask):
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    return float(np.degrees(np.arccos(np.clip(n[mask][:, 2], -1, 1))).mean())


def flat_outside(n, mask):
    out = n.copy()
    out[~mask] = (0.0, 0.0, 1.0)
    return out


def depth(n, engine, mask=None):
    from helio.application.pipeline import integrate_depth
    return np.asarray(integrate_depth(n, engine, mask=mask), np.float64)


def deplane(z, mask):
    ys, xs = np.nonzero(mask)
    A = np.c_[xs, ys, np.ones_like(xs)].astype(np.float64)
    c, *_ = np.linalg.lstsq(A, z[ys, xs], rcond=None)
    return z[ys, xs] - A @ c


def compare(z, ref, mask):
    """(relief, detail, corr) of depth *z* against *ref* on *mask*, both with their best plane out."""
    a, b = deplane(z, mask), deplane(ref, mask)
    relief = float(np.std(a) / max(np.std(b), 1e-12))
    corr = float(np.corrcoef(a, b)[0, 1])

    def fine(v):
        full = np.zeros(mask.shape)
        full[mask] = v
        wgt = cv2.GaussianBlur(mask.astype(np.float64), (0, 0), DETAIL_PX)
        low = cv2.GaussianBlur(full, (0, 0), DETAIL_PX) / np.maximum(wgt, 1e-6)
        inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return (full - low)[inner]
    fa, fb = fine(a), fine(b)
    detail = float(np.std(fa) / max(np.std(fb), 1e-12))
    return relief, detail, corr


DEPTH_SUBSET = ("Fast Poisson (FFT)", "Frankot-Chellappa", "Free-Boundary Poisson (DCT)",
                "Least-Squares (Free Boundary)", "Bilateral (Keeps Steps)")


def main(base, only=None, quick=False, scale=1.0):
    from helio.core.radiometry import linear_to_srgb, srgb_to_linear
    objs = only or objects(base)
    solvers = SOLVERS[:1] if quick else SOLVERS
    report = {}
    for obj in objs:
        t0 = time.time()
        L, cal, unc, gt, mask = load(base, obj, scale)
        r = report.setdefault(obj, {"normals": {}, "depth": {}})
        r["true_tilt"] = tilt(gt, mask)
        solved = {}
        for s in solvers:
            n = solve(s, L, cal)
            solved[s] = n
            e = ang_err(n, gt, mask)
            r["normals"][s] = dict(mean=float(e.mean()), median=float(np.median(e)),
                                   flatness=tilt(n, mask) / r["true_tilt"])
        enc = np.round(linear_to_srgb(np.clip(cal, 0, 255))).astype(np.uint8).astype(np.float32)
        for name, stack in (("8-bit sRGB, as stored", enc), ("8-bit sRGB, decoded", srgb_to_linear(enc))):
            n = solve("Linear Least Squares", L, stack)
            solved[name] = n
            e = ang_err(n, gt, mask)
            r["normals"][name] = dict(mean=float(e.mean()), median=float(np.median(e)),
                                      flatness=tilt(n, mask) / r["true_tilt"])
        # depth: the reference, then each method on the true and on the solved normals
        ref = depth(flat_outside(gt, mask), "Least-Squares (Free Boundary)", mask)
        for src, n in (("true normals", gt), ("least squares", solved["Linear Least Squares"]),
                       ("8-bit sRGB, as stored", solved["8-bit sRGB, as stored"])):
            nn = flat_outside(n, mask) if src == "true normals" else n
            for eng in DEPTH_SUBSET:
                for where, m in (("whole frame", None), ("object alone", mask)):
                    if quick and where == "whole frame" and src != "least squares":
                        continue
                    t = time.time()
                    try:
                        z = depth(nn, eng, m)
                        rel, det, cor = compare(z, ref, mask)
                    except Exception as exc:                     # noqa: BLE001
                        r["depth"][f"{src} | {eng} | {where}"] = dict(error=str(exc))
                        continue
                    r["depth"][f"{src} | {eng} | {where}"] = dict(relief=rel, detail=det, corr=cor,
                                                                  seconds=time.time() - t)
        print(f"  {obj}: {int(mask.sum())} px, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    write(report)
    return report


def write(report):
    objs = list(report)
    out = ["## Normals: mean angular error (degrees) and flatness (solved tilt / true tilt)", "",
           "| Method | mean error | median error | flatness |", "|---|---:|---:|---:|"]
    methods = list(report[objs[0]]["normals"])
    for m in methods:
        e = [report[o]["normals"][m]["mean"] for o in objs]
        md = [report[o]["normals"][m]["median"] for o in objs]
        f = [report[o]["normals"][m]["flatness"] for o in objs]
        out.append(f"| {m} | {np.mean(e):.2f} | {np.mean(md):.2f} | {np.mean(f):.3f} |")
    out += ["", f"True mean tilt over the objects: {np.mean([report[o]['true_tilt'] for o in objs]):.1f}°", "",
            "## Depth against the true shape (mean over objects)", "",
            "| Normals | Depth method | Where | relief | detail | corr |", "|---|---|---|---:|---:|---:|"]
    keys = list(report[objs[0]]["depth"])
    for k in keys:
        rows = [report[o]["depth"].get(k) for o in objs]
        rows = [x for x in rows if x and "relief" in x]
        if not rows:
            out.append(f"| {' | '.join(k.split(' | '))} | failed | | |")
            continue
        out.append(f"| {' | '.join(k.split(' | '))} | {np.mean([x['relief'] for x in rows]):.2f} | "
                   f"{np.mean([x['detail'] for x in rows]):.2f} | {np.mean([x['corr'] for x in rows]):.3f} |")
    print("\n".join(out))
    with open(os.path.join(ROOT, "docs", "benchmark_diligent_pi.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--objects=")), None)
    sc = next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--scale=")), 1.0)
    main(args[0] if args else os.path.expanduser("~/projects/DiLiGenT-Pi"), only, "--quick" in sys.argv, sc)
