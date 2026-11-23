"""The lens's distortion, from photographs of a chessboard (CAL-006).

A measurement assumes a straight line on the object is straight in the
photograph.  Most lenses bow it a little (barrel or pincushion), most at the
edges: a 0.5 % distortion is 25 px at the edge of a 10,000 px frame, and the
lengths and areas measured there are off by as much.

OpenCV's camera calibration (Zhang's method) on 10-20 photographs of a flat
printed chessboard, at different places and tilts in the frame, gives the
lens's focal length, centre and distortion coefficients.  Every frame of a
capture is then undistorted before the solve: same size, same scale at the
centre, straight lines straight.  It holds for that lens at that focal length
and focus distance (a zoom or a refocus needs its own).

GUI-free.
"""
from __future__ import annotations

import datetime
import os

import cv2
import numpy as np

#: Long side the chessboard is searched at (corners then refined at full size).
SEARCH_SIDE = 2000


def find_corners(grey: np.ndarray, pattern: tuple[int, int]):
    """The chessboard's inner corners in *grey* (uint8), full-size pixels, or None."""
    h, w = grey.shape[:2]
    s = min(1.0, SEARCH_SIDE / max(h, w))
    small = cv2.resize(grey, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else grey
    ok, c = cv2.findChessboardCornersSB(small, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        ok, c = cv2.findChessboardCorners(small, pattern,
                                          flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            return None
    c = (c.reshape(-1, 2) / s).astype(np.float32).reshape(-1, 1, 2)
    win = max(5, int(round(4 / s)))
    cv2.cornerSubPix(grey, c, (win, win), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01))
    return c


def calibrate(greys, pattern=(9, 6), square_mm: float = 1.0) -> dict:
    """Calibration from chessboard photographs (uint8 greys, one size).

    *pattern* is the inner corners (columns, rows): a board of 10 x 7 squares
    has 9 x 6.  Raises ValueError with fewer than 5 usable views."""
    pattern = (int(pattern[0]), int(pattern[1]))
    h, w = greys[0].shape[:2]
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * float(square_mm)
    objs, imgs, used = [], [], []
    for k, g in enumerate(greys):
        if g.shape[:2] != (h, w):
            raise ValueError(f"Photograph {k + 1} is {g.shape[1]} x {g.shape[0]}, the first {w} x {h}.")
        c = find_corners(g, pattern)
        if c is not None:
            objs.append(obj)
            imgs.append(c)
            used.append(k)
    if len(used) < 5:
        raise ValueError(f"The chessboard ({pattern[0]} x {pattern[1]} inner corners) was found in "
                         f"{len(used)} of {len(greys)} photographs; at least 5 are needed. Check the "
                         f"corner count (squares less one each way) and that the whole board is in view.")
    # k3 held at 0: on the moderate lenses of copy work it only fits noise (a 12-view test
    # lens with k1 = -0.08 came back with k3 = 9.5 and a worse correction).
    rms, K, dist, _r, _t = cv2.calibrateCamera(objs, imgs, (w, h), None, None,
                                               flags=cv2.CALIB_FIX_K3)
    # How far the distortion moves a corner of the frame, in pixels.
    # Where the lens puts the frame's corners and edge midpoints, against where a perfect lens
    # would (the forward model: inverting it can fail to converge at the very corners).
    ideal = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, 0], [0, h / 2]], np.float64)
    norm = np.column_stack([(ideal[:, 0] - K[0, 2]) / K[0, 0], (ideal[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(ideal))])
    proj, _ = cv2.projectPoints(norm, np.zeros(3), np.zeros(3), K, dist)
    edge = float(np.max(np.linalg.norm(proj.reshape(-1, 2) - ideal, axis=1)))
    return dict(K=K, dist=dist.reshape(-1), size=(int(w), int(h)), rms=float(rms),
                used=len(used), of=len(greys), pattern=pattern, edge_px=edge,
                created=datetime.datetime.now().isoformat(timespec="seconds"))


def save(path: str, cal: dict) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, K=cal["K"], dist=cal["dist"], size=np.asarray(cal["size"]),
                        rms=cal["rms"], used=cal["used"], of=cal["of"],
                        pattern=np.asarray(cal["pattern"]), edge_px=cal["edge_px"],
                        created=str(cal["created"]))
    return path


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return dict(K=z["K"].astype(np.float64), dist=z["dist"].astype(np.float64),
                    size=tuple(int(v) for v in z["size"]), rms=float(z["rms"]), used=int(z["used"]),
                    of=int(z["of"]), pattern=tuple(int(v) for v in z["pattern"]),
                    edge_px=float(z["edge_px"]), created=str(z["created"]))


def describe(cal: dict) -> str:
    return (f"{cal['used']} of {cal['of']} views · fit {cal['rms']:.2f} px RMS · "
            f"up to {cal['edge_px']:.1f} px of distortion at the corners")


def camera_for(cal: dict, scale: float = 1.0, offset=(0, 0)) -> np.ndarray:
    """The camera matrix for frames held at *scale* of the calibrated size,
    cut from (x0, y0) of the full frame (a region solve)."""
    K = np.asarray(cal["K"], np.float64).copy()
    K[0, 2] -= float(offset[0])
    K[1, 2] -= float(offset[1])
    K[:2] *= float(scale)
    return K


def undistort(img: np.ndarray, cal: dict, scale: float = 1.0, offset=(0, 0)) -> np.ndarray:
    """*img* (any channels, uint8 or float) with the lens's distortion taken out:
    the same size, the same scale at the centre."""
    K = camera_for(cal, scale, offset)
    return cv2.undistort(img, K, np.asarray(cal["dist"], np.float64), None, K)


def check_size(cal: dict, size) -> str:
    """'' when frames of *size* (w, h) are the calibrated camera's, else why not."""
    cw, ch = cal["size"]
    w, h = (int(v) for v in size)
    if (cw, ch) != (w, h):
        return f"The lens was calibrated on {cw} x {ch} photographs; these are {w} x {h}."
    return ""


# ── is the camera square to the baseboard? (CAL-008) ─────────────────

def _focal_from_homography(Hm: np.ndarray, w: int, h: int) -> float | None:
    """The focal length (px) one view of a plane implies, the principal point at the
    centre and square pixels (Zhang's two constraints, solved for f alone).  None when
    the view is too square-on to tell (the constraints vanish as the tilt does)."""
    T = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1]])
    Hc = T @ Hm
    h1, h2 = Hc[:, 0], Hc[:, 1]
    # with K = diag(f, f, 1):  h1' W h2 = 0,  h1' W h1 = h2' W h2,  W = diag(1/f², 1/f², 1)
    a = [h1[0] * h2[0] + h1[1] * h2[1], (h1[0] ** 2 + h1[1] ** 2) - (h2[0] ** 2 + h2[1] ** 2)]
    b = [-h1[2] * h2[2], -(h1[2] ** 2 - h2[2] ** 2)]
    A, B = np.array(a), np.array(b)
    den = float(A @ A)
    if den < 1e-18:
        return None
    inv_f2 = float(A @ B) / den
    if inv_f2 <= 0:
        return None
    return float(1.0 / np.sqrt(inv_f2))


def board_pose(grey: np.ndarray, pattern=(9, 6), square_mm: float | None = None, cal: dict | None = None,
               focal_px: float | None = None) -> dict:
    """How the camera looks at a flat chessboard lying on the baseboard.

    Returns tilt_deg (the camera's axis off square to the board), tilt_towards_deg
    (the compass direction, in the photograph, of the board edge that is nearer the
    camera: 0 = top of the frame, 90 = right), roll_deg (the board's rows against the
    frame's), mm_per_px (with *square_mm*: the scale on the board, at its centre),
    and where the focal length came from.  The focal length is the lens calibration's,
    else *focal_px* (EXIF), else estimated from the board itself."""
    import math
    h, w = grey.shape[:2]
    c = find_corners(grey, pattern)
    if c is None:
        raise ValueError(f"No {pattern[0]} x {pattern[1]} chessboard found in the photograph: the "
                         f"whole board in view, sharp, and lit without glare.")
    pts = c.reshape(-1, 2).astype(np.float64)
    if cal is not None:
        cw, ch = cal["size"]
        K = camera_for(cal, w / float(cw))
        dist = np.asarray(cal["dist"], np.float64)
        src = "the lens calibration"
    else:
        dist = np.zeros(5)
        f = focal_px
        src = "the photograph's EXIF"
        if not f:
            obj2 = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2).astype(np.float64)
            Hm, _ = cv2.findHomography(obj2, pts)
            f = _focal_from_homography(Hm, w, h) if Hm is not None else None
            src = "the board itself (an estimate)"
        if not f:
            f = 1.2 * max(w, h)                             # a normal lens: tilt still close
            src = "an assumed normal lens (rough)"
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]])
    obj = np.zeros((len(pts), 3))
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * float(square_mm or 1.0)
    ok, rvec, tvec = cv2.solvePnP(obj, pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("The board's pose could not be solved.")
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]                                             # the board's normal, camera axes (y down)
    if n[2] > 0:
        n = -n
    tilt = math.degrees(math.acos(min(1.0, abs(n[2]))))
    # the board's nearer side: the normal leans away from it, seen in the image (x right, y down)
    towards = (math.degrees(math.atan2(-n[0], n[1])) + 360.0) % 360.0
    gx = pts.reshape(pattern[1], pattern[0], 2)
    row = gx[:, -1] - gx[:, 0]
    roll = math.degrees(math.atan2(float(np.mean(row[:, 1])), float(np.mean(row[:, 0]))))
    roll = ((roll + 90.0) % 180.0) - 90.0                   # a board found from its other end is the same
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts) ** 2, axis=1))))
    out = dict(tilt_deg=tilt, tilt_towards_deg=towards, roll_deg=roll, focal_from=src,
               focal_px=float(K[0, 0]), reproj_px=err, centre_px=tuple(pts.mean(0)))
    if square_mm:
        # the board's spacing at its middle, along rows and columns
        dx = np.linalg.norm(np.diff(gx, axis=1), axis=2)
        dy = np.linalg.norm(np.diff(gx, axis=0), axis=2)
        mid = float(np.median(np.concatenate([dx.ravel(), dy.ravel()])))
        out["mm_per_px"] = float(square_mm) / mid
        out["distance_mm"] = float(np.linalg.norm(tvec))
    return out


def describe_pose(p: dict) -> str:
    sides = ["top", "top-right", "right", "bottom-right", "bottom", "bottom-left", "left", "top-left"]
    side = sides[int(((p["tilt_towards_deg"] + 22.5) % 360) // 45)]
    if p["tilt_deg"] < 0.3:
        verdict = "square to the board"
    elif p["tilt_deg"] < 1.0:
        verdict = f"{p['tilt_deg']:.1f}° off square: good for measurement"
    else:
        verdict = f"{p['tilt_deg']:.1f}° off square: the {side} of the frame is nearer the camera"
    return verdict
