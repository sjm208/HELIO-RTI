"""Capture tools, GUI-free: frame drift, re-alignment, calibration profiles
and the backlit (transmissive) frame (CAL-001…003, IMP-002, TRN-001).

The computations of the classic Capture panel's buttons, moved so both apps
run them: the drift check (helio.core.qc.interframe_drift at 2400 px, scaled
back to the capture's pixels), the alignment report (phase correlation
against the median frame), calibration profiles in RTIBuilder .lp, HELIO
.hjson and CSV, and the ORB + RANSAC registration of a backlit frame onto
the solve (trying its mirror images, since a shot from below is often
flipped).
"""
from __future__ import annotations

from helio.core.frame_read import read_frame

import datetime
import json
import os

import cv2
import numpy as np


# ── frame drift ──────────────────────────────────────────────────────────

def drift_report(paths, target: float = 2400.0) -> dict:
    """Has the camera or object moved between frames?  Frames are compared
    at *target* px; the shifts are reported in the capture's own pixels."""
    from helio.core.qc import interframe_drift
    from helio.core.sphere import image_size, read_many, reduction_for
    frames, scale = [], 1.0
    paths = list(paths)
    size = image_size(paths[0]) if paths else None
    k = reduction_for(size, int(target)) if size else 1     # JPEG decoded small, not whole
    for i, im in enumerate(read_many(paths, k)):
        if im is None:
            continue
        if not frames:
            full = max(im.shape[:2]) * k
            scale = min(1.0, target / float(full))
            dsize = (max(1, int(round(im.shape[1] * k * scale))),
                     max(1, int(round(im.shape[0] * k * scale))))
        if im.shape[1::-1] != dsize:
            im = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)
        frames.append(im)
    if len(frames) < 2:
        return {"verdict": "clean", "peak_px": 0.0, "median_px": 0.0, "worst": "",
                "n": len(frames), "shifts": []}
    r = interframe_drift(frames)
    inv = 1.0 / scale if scale > 1e-9 else 1.0
    wi = r["worst"][0]
    return {"verdict": r["verdict"], "peak_px": r["worst"][1] * inv, "median_px": r["median"] * inv,
            "worst": os.path.basename(paths[wi]) if wi < len(paths) else f"frame {wi}",
            "worst_index": wi, "n": r["n_compared"] + 1,
            "shifts": [(i, dx * inv, dy * inv, m * inv) for i, dx, dy, m in r["shifts"]]}


def alignment_report(paths) -> list[tuple[float, float]]:
    """Re-Align Images Now: each frame's shift against the median, in pixels
    ((0, 0) where it was skipped for low contrast)."""
    from helio.core.processing import phase_align_stack
    from helio.io import load_and_process_single
    rgbs, greys = [], []
    for p in paths:
        r, g = load_and_process_single(p)
        if r is not None:
            rgbs.append(r)
            greys.append(g)
    if len(greys) < 2:
        raise ValueError("Need at least 2 readable images.")
    return phase_align_stack(np.array(greys), rgbs)


# ── calibration profiles ─────────────────────────────────────────────────

def read_calibration(path: str):
    """(lights (N, 3) +Y up, source label, file names) from a .lp, .hjson/.json or .csv.

    .lp as Import Light Positions reads it (helio.io.lp); .hjson and .csv as
    the classic Load Calibration does (unit vectors, z floored at 0.001)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".lp", ".dome", ".txt"):
        from helio.io.lp import read_lp
        L = read_lp(path)
        names = []
        for line in open(path).read().splitlines():
            t = line.split()
            if len(t) >= 4:
                try:
                    float(t[0])
                except ValueError:
                    names.append(" ".join(t[:-3]))
        return L, "RTIBuilder .lp", names
    L, names = [], []
    if ext in (".hjson", ".json"):
        with open(path) as f:
            data = json.load(f)
        if "light_vectors" not in data:
            raise ValueError("Not a Helio calibration file (no light_vectors).")
        for v in data["light_vectors"]:
            lx, ly, lz = float(v["lx"]), float(v["ly"]), float(v["lz"])
            n = max((lx * lx + ly * ly + lz * lz) ** 0.5, 1e-9)
            L.append([lx / n, ly / n, max(lz / n, 0.001)])
            names.append(v.get("filename", ""))
        src = f"Helio JSON ({data.get('created', '')[:10]})"
    elif ext == ".csv":
        import csv
        with open(path) as f:
            for row in csv.reader(f):
                if not row or row[0].strip().lower() in ("image_index", "index", "#"):
                    continue
                try:
                    nums = [float(x) for x in row if x.strip()]
                except ValueError:
                    continue
                if len(nums) >= 3:
                    lx, ly, lz = nums[-3], nums[-2], nums[-1]
                    n = max((lx * lx + ly * ly + lz * lz) ** 0.5, 1e-9)
                    L.append([lx / n, ly / n, max(lz / n, 0.001)])
        src = "CSV"
    else:
        raise ValueError(f"Unsupported calibration format: {ext}")
    if len(L) < 3:
        raise ValueError(f"Only {len(L)} light vectors found; at least 3 are needed.")
    return np.array(L, np.float32), src, names


def write_calibration_hjson(path: str, lights, image_paths=(), object_id: str = "",
                            ppi=None, source: str = "sphere_or_vdc", sphere=None,
                            sphere2=None) -> str:
    """The classic HELIO calibration JSON, with where the chrome ball(s) were
    (*sphere*, *sphere2*: cx, cy, r in the photographs' pixels) when known:
    near-field centres its dome there (helio.application.pipeline)."""
    L = np.asarray(lights, np.float64)
    paths = list(image_paths)
    profile = {
        "helio_calibration_version": "1.0",
        "created": datetime.datetime.now().isoformat(),
        "object_id": object_id or "",
        "n_lights": len(L),
        "ppi": float(ppi) if ppi else None,
        "calibration_source": source,
        "image_filenames": [os.path.basename(p) for p in paths],
        "light_vectors": [{"index": i,
                           "filename": os.path.basename(paths[i]) if i < len(paths)
                           else f"image_{i + 1:03d}.jpg",
                           "lx": float(L[i][0]), "ly": float(L[i][1]), "lz": float(L[i][2])}
                          for i in range(len(L))]}
    for key, s in (("sphere", sphere), ("sphere2", sphere2)):
        if s is not None:
            cx, cy, r = (float(v) for v in tuple(s)[:3])
            profile[key] = {"cx": cx, "cy": cy, "r": r}
    with open(path, "w") as f:
        json.dump(profile, f, indent=2)
    return path


def calibration_spheres(path: str):
    """(sphere, sphere2) as (cx, cy, r) tuples, or None each, for a calibration
    file: from a Helio .hjson/.json itself, or for a .lp from the .hjson of
    the same name beside it (Helio writes the two together)."""
    stem, ext = os.path.splitext(path)
    src = path if ext.lower() in (".hjson", ".json") else stem + ".hjson"
    try:
        with open(src) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None, None

    def one(k):
        s = data.get(k)
        try:
            return (float(s["cx"]), float(s["cy"]), float(s["r"])) if s else None
        except (KeyError, TypeError, ValueError):
            return None
    return one("sphere"), one("sphere2")


# ── the backlit frame ────────────────────────────────────────────────────

def to_grey_u8(img) -> np.ndarray:
    """Any loaded image (8/16-bit, grey/BGR/BGRA) to uint8 grey."""
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float32)
    mx = float(a.max()) if a.size else 0.0
    if mx > 255.0:
        a = a / 257.0
    elif 0 < mx <= 1.0:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def reference_grey(albedo) -> np.ndarray:
    """The solve's albedo as uint8 grey: what the backlit frame is registered to."""
    a = np.asarray(albedo)
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if a.dtype != np.uint8:
        mx = float(a.max()) if a.size else 0.0
        a = (np.clip(a / mx, 0, 1) * 255).astype(np.uint8) if mx > 1.5 \
            else (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return a


def flip_matrix(flip_code, h: int, w: int) -> np.ndarray:
    F = np.eye(3, dtype=np.float64)
    if flip_code in (1, -1):
        F = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], np.float64) @ F
    if flip_code in (0, -1):
        F = np.array([[1, 0, 0], [0, -1, h - 1], [0, 0, 1]], np.float64) @ F
    return F


def register_backlit(raw_grey, ref_grey):
    """ORB + RANSAC homography from the backlit frame to the solve, trying the
    frame and its three mirror images.  Returns (H full-res raw → ref,
    inliers, flip code) or (None, inliers, None) when under 15 inliers."""
    MAX_DIM = 1600.0
    s_raw = min(1.0, MAX_DIM / max(raw_grey.shape))
    s_ref = min(1.0, MAX_DIM / max(ref_grey.shape))
    raw_s = cv2.resize(raw_grey, None, fx=s_raw, fy=s_raw, interpolation=cv2.INTER_AREA) \
        if s_raw < 1.0 else raw_grey
    ref_s = cv2.resize(ref_grey, None, fx=s_ref, fy=s_ref, interpolation=cv2.INTER_AREA) \
        if s_ref < 1.0 else ref_grey
    orb = cv2.ORB_create(nfeatures=4000)
    k_ref, d_ref = orb.detectAndCompute(ref_s, None)
    if d_ref is None or len(k_ref) < 12:
        return None, 0, None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best = (None, 0, None)
    for flip_code in (None, 1, 0, -1):
        var = raw_s if flip_code is None else cv2.flip(raw_s, flip_code)
        k_v, d_v = orb.detectAndCompute(var, None)
        if d_v is None or len(k_v) < 12:
            continue
        matches = bf.match(d_v, d_ref)
        if len(matches) < 12:
            continue
        matches = sorted(matches, key=lambda m: m.distance)[:500]
        p_v = np.float32([k_v[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        p_ref = np.float32([k_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(p_v, p_ref, cv2.RANSAC, 5.0)
        if H is None or mask is None:
            continue
        inl = int(mask.sum())
        if inl > best[1]:
            best = (H, inl, flip_code)
    H_small, inliers, flip_code = best
    if H_small is None or inliers < 15:
        return None, inliers, None
    hh, ww = raw_grey.shape[:2]
    F = flip_matrix(flip_code, hh, ww) if flip_code is not None else np.eye(3)
    H_full = np.diag([1.0 / s_ref, 1.0 / s_ref, 1.0]) @ H_small @ np.diag([s_raw, s_raw, 1.0]) @ F
    return H_full, inliers, flip_code


def warp_backlit(raw, H, ref_shape, out_shape=None) -> np.ndarray:
    """The backlit frame in the solve's space, float32 RGB 0-1 (rescaled if
    the solve's size changed since registration)."""
    h, w = out_shape or ref_shape
    rh, rw = ref_shape
    H = np.asarray(H, np.float64)
    if (h, w) != (rh, rw):
        H = np.diag([w / float(rw), h / float(rh), 1.0]) @ H
    src = np.asarray(raw).astype(np.float32)
    if src.ndim == 3:
        src = src[..., :3][..., ::-1].copy()
    mx = float(src.max()) if src.size else 0.0
    src = src / (65535.0 if mx > 255.0 else 255.0 if mx > 1.0 else 1.0)
    return cv2.warpPerspective(src, H, (w, h), flags=cv2.INTER_LINEAR).astype(np.float32)


# ── are the lights the right way up? (KD-027, KD-054) ────────────────────

def light_frame_check(paths, lights, *, long_edge: int = 512, balls=()) -> dict:
    """Whether a set of lights has north at the top of the photographs.

    HELIO's lights are +Y up (north is the top of the image).  Light files
    written by its dome calibrator before October 2026 kept +Y down (KD-027),
    and other tools differ, so a file can arrive with north and south
    swapped.  Solved with such lights the normals' north–south slopes are
    reversed against the east–west ones, so they no longer belong to any
    surface: a real surface's slopes have no curl.  This solves least-
    squares normals from the photographs (stored values, long edge
    *long_edge*), compares the median |curl| of the slopes as given and with
    north and south swapped, and says which reading is the consistent one.
    *balls* (cx, cy, r) in full-resolution pixels are left out (a mirror
    ball's normals mean nothing here).

    Returns {"verdict": "upright" | "upside down" | "unclear",
             "curl_as_given", "curl_swapped", "ratio", "pixels"}."""
    L = np.asarray(lights, np.float64)
    ok = np.isfinite(L).all(axis=1)
    paths = [p for p, k in zip(paths, ok) if k]
    L = L[ok]
    if len(L) < 3 or len(paths) != len(L):
        return {"verdict": "unclear", "reason": "fewer than 3 usable lights"}
    first = read_frame(paths[0], cv2.IMREAD_GRAYSCALE)
    if first is None:
        return {"verdict": "unclear", "reason": "the first frame could not be read"}
    h0, w0 = first.shape
    k = min(1.0, long_edge / float(max(h0, w0)))
    size = (max(8, int(round(w0 * k))), max(8, int(round(h0 * k))))
    frames = []
    for p in paths:
        g = read_frame(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            return {"verdict": "unclear", "reason": f"{os.path.basename(p)} could not be read"}
        if g.shape != (h0, w0):
            g = cv2.resize(g, (w0, h0))
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        frames.append(cv2.GaussianBlur(g, (0, 0), 1.0) / 255.0)
    I = np.stack(frames, -1)
    G = I.reshape(-1, len(L)) @ np.linalg.pinv(L).T
    N = (G / np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-9)).reshape(I.shape[:2] + (3,))
    nz = np.maximum(N[..., 2], 1e-3)
    m = (I.mean(-1) > 0.06) & (N[..., 2] > 0.2)
    m[:4] = m[-4:] = False
    m[:, :4] = m[:, -4:] = False
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    for cx, cy, r in balls or ():
        m &= (xx - cx * k) ** 2 + (yy - cy * k) ** 2 > (1.4 * r * k) ** 2
    if m.sum() < 500:
        return {"verdict": "unclear", "reason": "too little lit surface to judge", "pixels": int(m.sum())}
    p_ = -N[..., 0] / nz
    q_up = N[..., 1] / nz                    # dz/drow for +Y-up normals
    c_up = np.abs(np.gradient(p_, axis=0) - np.gradient(q_up, axis=1))[m]
    c_dn = np.abs(np.gradient(p_, axis=0) + np.gradient(q_up, axis=1))[m]
    a, b = float(np.median(c_up)), float(np.median(c_dn))
    ratio = b / max(a, 1e-12)
    verdict = "upright" if ratio > 1.15 else "upside down" if ratio < 1 / 1.15 else "unclear"
    return {"verdict": verdict, "curl_as_given": a, "curl_swapped": b, "ratio": ratio,
            "pixels": int(m.sum())}


def swap_north_south(lights) -> np.ndarray:
    """The same lights with north and south exchanged (y negated)."""
    L = np.array(lights, np.float32, copy=True)
    L[:, 1] *= -1.0
    return L
