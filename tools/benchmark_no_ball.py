"""Helio's no-ball solve (helio.core.uncalibrated) on the DiLiGenT benchmark.

The lights are hidden from the solve: it gets the photographs alone, with the
lamps evened (each frame divided by its light's intensity, as one lamp moved
round would give), and again with them made 10% and 20% uneven (a fixed
random scale per frame).  Each is scored as tools/benchmark_diligent.py scores the
calibrated solvers: the mean angle to the true normals inside the mask.

    python tools/benchmark_no_ball.py [path/to/DiLiGenT]     (default ~/Downloads/DiLiGenT)

Prints a Markdown table: the rough solve (the old one), the no-ball solve at
each unevenness (with "rough" where it fell back), and least squares with
the true lights (from docs/benchmark_diligent.json) for comparison.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import benchmark_diligent as BD  # noqa: E402
from helio.core.photometric_stereo import photometric_stereo_pca  # noqa: E402
from helio.core.uncalibrated import solve_no_ball  # noqa: E402

UNEVEN = (0.0, 0.1, 0.2)


def main(base):
    known = {}
    try:
        with open(os.path.join(ROOT, "docs", "benchmark_diligent.json")) as f:
            known = json.load(f)["mean_median_deg"]
    except (OSError, KeyError, ValueError):
        pass
    print("| Object | Least squares, true lights | Rough (old) | No ball, even | 10% uneven | 20% uneven |")
    print("|---|---|---|---|---|---|")
    cols = {k: [] for k in ("ls", "rough", *UNEVEN)}
    for obj in BD.OBJECTS:
        L, cal, _unc, gt, mask, (ys, xs) = BD.load(base, obj)
        h, w = mask.shape
        rough = float(BD.ang_err(photometric_stereo_pca(cal)[1], gt).mean())
        cells = []
        for sig in UNEVEN:
            g = np.random.default_rng(1).normal(1, sig, len(cal)).clip(0.5, 1.6)[:, None] if sig else 1.0
            full = np.zeros((len(cal), h * w), np.float32)
            full[:, ys * w + xs] = cal * g
            _L, N, info = solve_no_ball(full, h, w, mask)
            e = float(BD.ang_err(N[ys * w + xs], gt).mean())
            cols[sig].append(e)
            cells.append(f"{e:.1f}" + (" (rough)" if info["used"] == "rough" else ""))
        ls = known.get(obj, {}).get("Linear Least Squares", [float("nan")])[0]
        cols["ls"].append(ls)
        cols["rough"].append(rough)
        print(f"| {obj} | {ls:.1f} | {rough:.1f} | " + " | ".join(cells) + " |", flush=True)
    print(f"| **mean** | {np.nanmean(cols['ls']):.1f} | {np.mean(cols['rough']):.1f} | "
          + " | ".join(f"{np.mean(cols[s]):.1f}" for s in UNEVEN) + " |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/DiLiGenT"))
