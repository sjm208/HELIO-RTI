"""What is on the publication sheet, and whether two things on it are the same.

The sheet is built over a long session: a section kept here, a 3D picture
there.  A sheet came back with four sections on two seals - A–B and E–F one
line a few pixels apart, C–D and G–H the same - and two 3D pictures from the
same angle, because nothing showed what was already on it and the only check
was for a line with exactly the same ends.  These say when a new section or
picture is close to one already kept, in words, so the person can replace
it, keep both or leave it.  GUI-free.
"""
from __future__ import annotations

import math

import numpy as np


def _line(p: dict):
    (ax, ay), (bx, by) = p["a_px"], p["b_px"]
    return np.array([ax, ay], float), np.array([bx, by], float)


def band_half_width(p: dict) -> float:
    """How far the section's parallel copies reach to either side of it, in
    pixels (0 for a section alone)."""
    a, b = _line(p)
    d = b - a
    L = float(np.hypot(*d)) or 1.0
    n = np.array([-d[1], d[0]]) / L
    far = 0.0
    for c in p.get("series") or []:
        if "a_px" in c and "b_px" in c:
            ca, cb = np.asarray(c["a_px"], float), np.asarray(c["b_px"], float)
            far = max(far, abs(float(np.dot((ca + cb) / 2 - a, n))))
    return far


def section_closeness(p: dict, q: dict) -> dict | None:
    """How *p* lies against *q*, or None when they are plainly different
    sections: {'apart': px between them, 'angle': degrees, 'overlap': the share
    of p that runs alongside q, 0..1}.

    Two sections are the same when they run the same way (within 12°), most
    of one runs alongside the other (over half), they are about as long (2:3
    either way) and the new one lies within the other's band of parallel
    copies - or, alone, within a twentieth of its length."""
    pa, pb = _line(p)
    qa, qb = _line(q)
    dp, dq = pb - pa, qb - qa
    Lp, Lq = float(np.hypot(*dp)), float(np.hypot(*dq))
    if Lp < 1 or Lq < 1:
        return None
    ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(dp, dq))) / (Lp * Lq))))
    if ang > 12 or not 2 / 3 <= Lp / Lq <= 3 / 2:
        return None
    u = dq / Lq
    n = np.array([-u[1], u[0]])
    apart = abs(float(np.dot((pa + pb) / 2 - qa, n)))
    if apart > band_half_width(q) + 0.05 * Lq:
        return None
    s0, s1 = sorted((float(np.dot(pa - qa, u)), float(np.dot(pb - qa, u))))
    overlap = max(0.0, min(s1, Lq) - max(s0, 0.0)) / max(s1 - s0, 1e-9)
    if overlap < 0.5:
        return None
    return dict(apart=apart, angle=ang, overlap=overlap)


def close_section(new: dict, kept: list[dict]) -> tuple[int, dict] | None:
    """The kept section *new* is closest to, as (index, section_closeness),
    or None when it is unlike all of them."""
    best = None
    for i, q in enumerate(kept):
        c = section_closeness(new, q)
        if c is not None and (best is None or c["apart"] < best[1]["apart"]):
            best = (i, c)
    return best


def closeness_words(c: dict, mm_per_px: float = 0.0) -> str:
    """'0.6 mm apart and 2° apart', in millimetres once there is a scale."""
    gap = f"{c['apart'] * mm_per_px:.1f} mm" if mm_per_px else f"{c['apart']:.0f} px"
    return f"{gap} apart, {c['angle']:.0f}° apart in direction"


def _thumb(img) -> np.ndarray:
    import cv2
    a = np.asarray(img)
    if a.ndim == 3:
        a = a[..., :3].astype(np.float32).mean(axis=2)
    a = cv2.resize(a.astype(np.float32), (64, 48), interpolation=cv2.INTER_AREA)
    return (a - a.mean()) / (a.std() + 1e-6)


def pictures_alike(a, b) -> float:
    """How alike two pictures are, 0..1 (1: the same picture), by their
    correlation at a small size, so the same 3D view taken twice reads as
    alike although its pixels differ a little."""
    ta, tb = _thumb(a), _thumb(b)
    return max(0.0, float((ta * tb).mean()))


ALIKE = 0.97              # pictures_alike above this: the same view taken again


def close_picture(new, kept: list) -> tuple[int, float] | None:
    """The kept picture *new* is (all but) the same as, as (index, likeness),
    or None.  *kept* are pictures."""
    best = None
    for i, k in enumerate(kept):
        s = pictures_alike(new, k)
        if s >= ALIKE and (best is None or s > best[1]):
            best = (i, s)
    return best


def duplicates(kept: list[dict]) -> dict[int, int]:
    """For a list already kept (a session saved before this check): each
    section that is close to an earlier one, as {later index: earlier index}."""
    out = {}
    for j in range(1, len(kept)):
        for i in range(j):
            if section_closeness(kept[j], kept[i]) is not None:
                out[j] = i
                break
    return out
