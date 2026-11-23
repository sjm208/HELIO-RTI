"""Colour accuracy against a ColorChecker (MET-010).

Find a 24-patch ColorChecker Classic in a photograph (OpenCV's mcc
detector), read each patch, and say how far each is from the chart's
published colour: CIEDE2000 (ΔE₀₀) in CIELAB under D50, the measure the
heritage imaging guidelines use (ISO 19264-1, FADGI, Metamorfoze).

The photograph is brought to sRGB by its own ICC profile and read as sRGB,
unless it says otherwise (``linear=True`` for a linear TIFF), converted to XYZ, adapted from D65 to D50 (Bradford),
and to L*a*b*.  The grey patches also give the tone response: how far each
step's lightness is from the chart's.

The Met's account of its colour work (Perspectives, "On Color Fidelity")
reports an average ΔE₀₀ of about 2 across a 140-patch target as its
working practice, and around 3 as where a difference becomes noticeable
side by side.  The verdict here uses those two figures, and says that is
what they are, rather than claiming any one guideline's star rating.
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np

# X-Rite ColorChecker Classic, after November 2014: CIELAB, D50, 2° observer.
REFERENCE = (
    ("Dark skin", 37.54, 14.37, 14.92), ("Light skin", 64.66, 19.27, 17.50),
    ("Blue sky", 49.32, -3.82, -22.54), ("Foliage", 43.46, -12.74, 22.72),
    ("Blue flower", 54.94, 9.61, -24.79), ("Bluish green", 70.48, -32.26, -0.37),
    ("Orange", 62.73, 35.83, 56.50), ("Purplish blue", 39.43, 10.75, -45.17),
    ("Moderate red", 50.57, 48.64, 16.67), ("Purple", 30.10, 22.54, -20.87),
    ("Yellow green", 71.77, -24.13, 58.19), ("Orange yellow", 71.51, 18.24, 67.37),
    ("Blue", 28.37, 15.42, -49.80), ("Green", 54.38, -39.72, 32.27),
    ("Red", 42.43, 51.05, 28.62), ("Yellow", 81.80, 2.67, 80.41),
    ("Magenta", 50.63, 51.28, -14.12), ("Cyan", 49.57, -29.71, -28.32),
    ("White 9.5", 95.19, -1.03, 2.93), ("Neutral 8", 81.29, -0.57, 0.44),
    ("Neutral 6.5", 66.89, -0.75, -0.06), ("Neutral 5", 50.76, -0.13, 0.14),
    ("Neutral 3.5", 35.63, -0.46, -0.48), ("Black 2", 20.64, 0.07, -0.46),
)
GREYS = range(18, 24)

_M_SRGB = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]])
_W65 = np.array([0.95047, 1.0, 1.08883])
_W50 = np.array([0.96422, 1.0, 0.82521])
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_lab_d50(rgb_lin) -> np.ndarray:
    """Linear sRGB (D65) to CIELAB under D50, Bradford-adapted."""
    xyz = np.asarray(rgb_lin, np.float64) @ _M_SRGB.T
    cone = (_BRADFORD @ _W50) / (_BRADFORD @ _W65)
    A = np.linalg.inv(_BRADFORD) @ np.diag(cone) @ _BRADFORD
    t = (xyz @ A.T) / _W50
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def delta_e_2000(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    """CIEDE2000 (Sharma, Wu and Dalal's implementation notes, 2005)."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, np.float64), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(C1p * C2p == 0, 0.0, np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh)))
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp = (L1 + L2) / 2
    Cbp = (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2,
                            np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dtheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / (kL * Sl)) ** 2 + (dCp / (kC * Sc)) ** 2 + (dHp / (kH * Sh)) ** 2
                   + Rt * (dCp / (kC * Sc)) * (dHp / (kH * Sh)))


def find_chart(img_bgr: np.ndarray):
    """The chart's 24 patches as mean sRGB (0..1, chart order) and their
    centres, or None.  OpenCV's mcc detector, ColorChecker Classic."""
    img = np.ascontiguousarray(img_bgr)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    det = cv2.mcc.CCheckerDetector_create()
    try:                                       # OpenCV 5: the chart type is set, not passed
        det.setColorChartType(cv2.mcc.MCC24)
        ok = det.process(img, 1)
    except (AttributeError, cv2.error):        # OpenCV 4.x
        ok = det.process(img, cv2.mcc.MCC24, 1)
    if not ok:
        return None
    best = det.getBestColorChecker()
    if best is None:
        return None
    stats = np.asarray(best.getChartsRGB(), np.float64)       # rows: patch × channel
    if stats.ndim != 2 or stats.shape[0] < 72:
        return None
    means = stats[:, 1].reshape(-1, 3)[:24] / 255.0             # p_size, average, std, max, min
    try:
        box = np.asarray(best.getBox(), np.float32)
    except Exception:
        box = None
    return dict(rgb=means, box=box, cost=float(best.getCost()) if hasattr(best, "getCost") else None)


def evaluate(rgb_measured, *, linear: bool = False) -> dict:
    """ΔE₀₀ per patch against the reference, and the summary."""
    rgb = np.clip(np.asarray(rgb_measured, np.float64), 0, 1)
    lab = linear_to_lab_d50(rgb if linear else srgb_to_linear(rgb))
    ref = np.array([r[1:] for r in REFERENCE])
    de = delta_e_2000(lab, ref)
    dL = lab[list(GREYS), 0] - ref[list(GREYS), 0]
    mean = float(de.mean())
    verdict = ("Within the Met's working practice (about 2 on average)" if mean <= 2.0 else
               "Close: a careful eye may see it side by side (2 to 3)" if mean <= 3.0 else
               "Noticeable (over 3): profile the camera, or check the light and white balance")
    rows = [dict(n=i + 1, name=REFERENCE[i][0], ref=ref[i].round(2).tolist(),
                 measured=lab[i].round(2).tolist(), de=round(float(de[i]), 2)) for i in range(24)]
    steps = np.diff(lab[list(GREYS), 0])
    return dict(reliable=bool((steps < 0).all()), rows=rows, mean=round(mean, 2), median=round(float(np.median(de)), 2),
                max=round(float(de.max()), 2), worst=REFERENCE[int(de.argmax())][0],
                p90=round(float(np.percentile(de, 90)), 2), grey_dL=dL.round(2).tolist(),
                grey_mean_dL=round(float(np.abs(dL).mean()), 2), verdict=verdict,
                encoding="linear" if linear else "sRGB")


NOT_A_CLASSIC = ("Something like a chart was found, but its grey patches are not in order, so this is not a "
                 "reading. Only the 24-patch ColorChecker Classic can be read: a Digital SG cannot.")


def validate(stages) -> dict:
    """The chart through the stages of the work: *stages* (name, uint8 RGB picture, sRGB), as
    the ISO 19264 check on a photograph, carried on to what was made from it (the colour map,
    a render).  Each: its evaluate(), or a reason it could not be read."""
    rows = []
    for name, rgb in stages:
        found = find_chart(np.ascontiguousarray(rgb)[..., ::-1])
        res = None if found is None else evaluate(found["rgb"])
        if res is None or not res["reliable"]:
            rows.append(dict(stage=name, problem="no ColorChecker Classic read here"))
        else:
            rows.append(dict(stage=name, mean=res["mean"], max=res["max"], worst=res["worst"],
                             grey_dL=res["grey_mean_dL"], verdict=res["verdict"], patches=res["rows"]))
    good = [r for r in rows if "mean" in r]
    return dict(stages=rows, change=round(good[-1]["mean"] - good[0]["mean"], 2) if len(good) > 1 else None)


def write_validation(res: dict, base: str) -> list[str]:
    """<base>.csv, every patch at every stage, and <base>.txt, the summary."""
    good = [r for r in res["stages"] if "mean" in r]
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patch", "name", "L ref", "a ref", "b ref"] + [f"dE00 {r['stage']}" for r in good])
        for i in range(24):
            p = good[0]["patches"][i]
            w.writerow([p["n"], p["name"], *p["ref"]] + [r["patches"][i]["de"] for r in good])
    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write("Colour validation: ColorChecker Classic, CIEDE2000 under D50\n"
                "The method of ISO 19264-1's colour measure; not a claim of conformance to it.\n\n")
        for r in res["stages"]:
            f.write(f"{r['stage']}: {r['problem']}\n" if "problem" in r else
                    f"{r['stage']}: mean {r['mean']}, worst {r['max']} ({r['worst']}), grey steps "
                    f"{r['grey_dL']} L*. {r['verdict']}\n")
        if res["change"] is not None:
            f.write(f"\nFirst stage to last: {res['change']:+.2f} ΔE00\n")
    return [base + ".csv", base + ".txt"]


def check_image(path: str, *, linear: bool = False) -> dict:
    """Find the chart in the photograph at *path* and evaluate it."""
    from helio.core.frame_read import read_frame
    from helio.io.raw_loader import is_raw
    raw = is_raw(path)
    img = read_frame(path, cv2.IMREAD_COLOR, sharp=True) if raw else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {os.path.basename(path)}.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype == np.uint16:
        img8 = (img / 257.0).astype(np.uint8)
    else:
        img8 = img
    profile = ""
    if not linear:
        from helio.io.colour_space import profile_name, to_srgb
        profile = profile_name(path)
        img8 = to_srgb(img8[..., ::-1], path)[..., ::-1]
    found = find_chart(img8)
    if found is None:
        raise LookupError("No ColorChecker Classic (24 patches) was found in this photograph. "
                          "It needs to be flat, evenly lit, and fill at least a twentieth of the frame.")
    rgb = found["rgb"]
    if img.dtype == np.uint16:                   # read the patches again at full depth
        rgb = rgb                                 # the detector's means are 8-bit; close enough
    out = evaluate(rgb, linear=linear)
    if not out["reliable"]:
        raise LookupError(NOT_A_CLASSIC)
    if profile and "srgb" not in profile.lower():
        out["encoding"] = f"{profile}, converted to sRGB"
    out.update(path=path, box=None if found["box"] is None else found["box"].tolist())
    return out


def write_csv(res: dict, path: str) -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patch", "name", "L ref", "a ref", "b ref", "L", "a", "b", "dE00"])
        for r in res["rows"]:
            w.writerow([r["n"], r["name"], *r["ref"], *r["measured"], r["de"]])
        w.writerow([])
        w.writerow(["mean dE00", res["mean"]])
        w.writerow(["median dE00", res["median"]])
        w.writerow(["90th percentile dE00", res["p90"]])
        w.writerow(["max dE00", res["max"], res["worst"]])
        w.writerow(["grey steps mean |dL*|", res["grey_mean_dL"]])
        w.writerow(["read as", res["encoding"]])
    return path


def synthetic_chart(size: int = 900, rgb=None, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A ColorChecker drawn from the reference (or *rgb*), on a dark mat: for tests."""
    ref = np.array([r[1:] for r in REFERENCE])
    if rgb is None:
        rgb = lab_d50_to_srgb(ref)
    cols, rows = 6, 4
    pw = size // (cols + 1)
    gap = pw // 6
    W, H = cols * pw + (cols + 1) * gap, rows * pw + (rows + 1) * gap
    img = np.full((H + 2 * pw, W + 2 * pw, 3), 40, np.uint8)
    cv2.rectangle(img, (pw - gap, pw - gap), (pw + W + gap, pw + H + gap), (20, 20, 20), -1)
    rng = np.random.default_rng(seed)
    for i in range(24):
        r, c = divmod(i, cols)
        x0, y0 = pw + gap + c * (pw + gap), pw + gap + r * (pw + gap)
        col = np.clip(np.asarray(rgb[i]) * 255.0, 0, 255)
        patch = np.zeros((pw, pw, 3)) + col[::-1]
        if noise:
            patch += rng.normal(0, noise, patch.shape)
        img[y0:y0 + pw, x0:x0 + pw] = np.clip(patch, 0, 255).astype(np.uint8)
    return img


def lab_d50_to_srgb(lab) -> np.ndarray:
    """CIELAB D50 to sRGB (0..1), for drawing the reference."""
    lab = np.asarray(lab, np.float64)
    fy = (lab[..., 0] + 16) / 116
    fx, fz = fy + lab[..., 1] / 500, fy - lab[..., 2] / 200
    d = 6 / 29

    def finv(f):
        return np.where(f > d, f ** 3, 3 * d * d * (f - 4 / 29))
    xyz50 = np.stack([finv(fx), finv(fy), finv(fz)], -1) * _W50
    cone = (_BRADFORD @ _W65) / (_BRADFORD @ _W50)
    A = np.linalg.inv(_BRADFORD) @ np.diag(cone) @ _BRADFORD
    lin = (xyz50 @ A.T) @ np.linalg.inv(_M_SRGB).T
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
