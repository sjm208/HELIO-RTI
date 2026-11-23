"""Small standalone exports that used to be buried inside larger packages.

Each of these existed only as a step inside something bigger - the .lp inside
the GIS DEM package, the checksum manifest inside the same, the uncertainty
raster likewise. They are useful on their own, and burying them meant the only
way to get a light file was to export a whole DEM.
"""
import os

import numpy as np


def write_lp(path: str, light_vecs, image_names=None) -> str:
    """Write a CHI/RTIBuilder .lp light file.

    The format every other RTI tool reads: a count, then one line per image
    naming it and its unit light vector. A partner lab could not process files
    we sent precisely because the direction information did not travel with
    them.
    """
    lv = [] if light_vecs is None else list(light_vecs)    # a list or an (N, 3) array
    if not lv:
        raise ValueError("No light vectors to write. Calibrate the lights first.")
    names = [os.path.basename(n) for n in (image_names or [])]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(lv)}\n")
        for i, v in enumerate(lv):
            lx, ly, lz = (float(v[0]), float(v[1]), float(v[2]))
            nm = names[i] if i < len(names) else f"image_{i:03d}"
            fh.write(f"{nm} {lx:.6f} {ly:.6f} {lz:.6f}\n")
    return path


def draw_scale_bar(img: np.ndarray, mm_per_px: float, *,
                   target_frac: float = 0.25, margin: int = 24) -> np.ndarray:
    """Burn a calibrated scale bar into a copy of *img*.

    Picks a round length - 1, 2, 5, 10, 20, 50... mm - closest to
    *target_frac* of the image width, so the bar reads as a real measurement
    rather than an arbitrary rule. Returns a new array; the input is untouched.
    """
    import cv2
    if mm_per_px is None or mm_per_px <= 0:
        raise ValueError("No measured scale. Set mm/pixel in Metrology first.")

    out = np.array(img, copy=True)
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if out.dtype != np.uint8:
        lo, hi = float(np.nanmin(out)), float(np.nanmax(out))
        out = ((out - lo) / max(hi - lo, 1e-9) * 255).astype(np.uint8)

    h, w = out.shape[:2]
    want_mm = (w * target_frac) * mm_per_px
    nice = [1, 2, 5]
    mag = 10 ** int(np.floor(np.log10(max(want_mm, 1e-6))))
    steps = sorted({n * mag for n in nice} | {n * mag * 10 for n in nice} |
                   {n * mag / 10 for n in nice})
    bar_mm = min(steps, key=lambda s: abs(s - want_mm))
    bar_px = int(round(bar_mm / mm_per_px))
    bar_px = max(8, min(bar_px, w - 2 * margin))

    thick = max(3, h // 200)
    x2, y2 = w - margin, h - margin
    x1 = x2 - bar_px
    # A dark plate behind the bar keeps it readable on a light surface.
    cv2.rectangle(out, (x1 - 10, y2 - thick - 30), (x2 + 10, y2 + 10),
                  (20, 20, 20), -1)
    cv2.rectangle(out, (x1, y2 - thick), (x2, y2), (255, 255, 255), -1)
    label = f"{bar_mm:g} mm" if bar_mm >= 1 else f"{bar_mm * 1000:g} um"
    scale = max(0.4, h / 1400.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(out, label, (x1 + (bar_px - tw) // 2, y2 - thick - 8),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out
