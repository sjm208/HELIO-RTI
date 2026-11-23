"""The Virtual Dome Calibrator's arithmetic, GUI-free (CAL-005…CAL-029).

For captures without a chrome sphere: each frame's light is placed by hand
on a dome seen from above, estimated from a shadow, read from EXIF, merged
from another light file, then checked and nudged against a solve.

Lights here are +Y-up unit vectors, the light-matrix frame (north, the top of
the photograph, is +Y; helio.core.frames), so what is placed goes to the
solve, an .lp file and the score without conversion.  A dome on screen flips
y once to draw (``to_dome_point`` / ``from_dome_point``).  The classic dome
kept its vectors in screen frame (+Y down) and handed them to the solve, the
.lp export and the score unconverted, which mirrored every hand-placed light
north–south (KD-027); it now converts at those edges with ``from_screen``.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np

# The classic Templates Library: (lights in the ring, elevation in degrees).
TEMPLATES = {
    "RTI Dome — 8 light": [(8, 45)],
    "RTI Dome — 16 light": [(16, 40)],
    "RTI Dome — 24 light": [(8, 60), (16, 35)],
    "RTI Dome — 32 light": [(8, 65), (16, 45), (8, 25)],
    "CHI Gold Standard (64)": [(8, 70), (16, 55), (24, 38), (16, 22)],
    "PTM Standard (2 rings)": [(8, 55), (8, 28)],
    "Raking Survey (outer)": [(1, 20)],
    "Overhead Survey": [(1, 75)],
    "Museum Compact (12)": [(6, 55), (6, 30)],
}

# The classic Arrange Helpers.
ARRANGE = {"ring60": "Ring at the elevation", "ring35": "Ring 35°",
           "two_rings": "Two rings", "hemisphere": "Hemisphere"}


# ── angles and the dome on screen ────────────────────────────────────────

# Named for the lights' frame (+Y up, north = +Y): helio.domain.view_controls has
# angles / from_angles for the screen's frame, whose y runs the other way.  The two
# shared names until 2026-11 (docs/OVERLAP_AUDIT.md).
def light_angles(v) -> tuple[float, float]:
    """Azimuth (0 = north, clockwise) and elevation in degrees, of a light (+Y up)."""
    x, y, z = (float(c) for c in v[:3])
    return math.degrees(math.atan2(x, y)) % 360.0, math.degrees(math.asin(max(0.0, min(1.0, z))))


def light_from_angles(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """The light (+Y up) at a compass azimuth and elevation: the inverse of light_angles."""
    r, a = math.cos(math.radians(el_deg)), math.radians(az_deg)
    return (r * math.sin(a), r * math.cos(a), math.sin(math.radians(el_deg)))


def to_dome_point(v) -> tuple[float, float]:
    """Where a light sits on a unit dome drawn on screen (y down)."""
    return float(v[0]), -float(v[1])


def from_dome_point(dx: float, dy: float) -> tuple[float, float, float]:
    """The classic _canvas_to_vec: a point on a unit dome drawn on screen (y
    down; outside the rim it is pulled onto it) to a light."""
    d = math.hypot(dx, dy)
    if d > 1.0:
        dx, dy = dx / d, dy / d
    return (dx, -dy, math.sqrt(max(0.001, 1.0 - dx * dx - dy * dy)))


def from_screen(vecs) -> list[tuple[float, float, float]]:
    """Screen-frame (+Y down) vectors, as the classic dome stores them, to lights."""
    return [(float(x), -float(y), float(z)) for x, y, z in vecs]


to_screen = from_screen          # the same flip, the other way


def as_lights(vecs) -> np.ndarray:
    """The classic Confirm: unit vectors, z floored at 0.001 (N, 3) float32."""
    out = []
    for x, y, z in vecs:
        n = math.sqrt(x * x + y * y + z * z) or 1.0
        out.append([x / n, y / n, max(0.001, z / n)])
    return np.array(out, np.float32)


# ── arranging ────────────────────────────────────────────────────────────

def ring(n: int, el_deg: float) -> list[tuple[float, float, float]]:
    """*n* lights evenly round one elevation, the first in the east and the
    rest clockwise, as the classic dome laid them out."""
    r, z = math.cos(math.radians(el_deg)), math.sin(math.radians(el_deg))
    return [(r * math.cos(2 * math.pi * k / n), -r * math.sin(2 * math.pi * k / n), z)
            for k in range(n)]


def hemisphere(n: int) -> list[tuple[float, float, float]]:
    """The classic default: 8 at 60°, 16 at 38°, 32 at 22°, the rest overhead."""
    from helio.core.processing import default_ring_positions
    return to_screen(default_ring_positions(n))


def arrange(mode: str, n: int, el_deg: float = 35.0) -> list[tuple[float, float, float]]:
    if mode == "ring60":
        return ring(n, el_deg)
    if mode == "ring35":
        return ring(n, 35.0)
    if mode == "two_rings":
        h = n // 2
        return ring(h, 55.0) + ring(n - h, 30.0)
    if mode == "hemisphere":
        return hemisphere(n)
    raise ValueError(f"Unknown arrangement: {mode}")


def template(name: str, n: int) -> list[tuple[float, float, float]]:
    """A Templates Library pattern, repeated or cut to *n* lights."""
    raw = []
    for cap, el in TEMPLATES[name]:
        raw += ring(cap, el)
    return [raw[j % len(raw)] for j in range(n)]


# ── editing ──────────────────────────────────────────────────────────────

def radius_scale(original, scale: float) -> list[tuple[float, float, float]]:
    """Every light pulled toward the zenith (scale < 1), keeping its azimuth;
    always from the positions before the first scaling."""
    out = []
    for x, y, z in original:
        r = math.hypot(x, y)
        if r < 1e-9:
            out.append((x, y, z))
            continue
        nr, a = min(r * scale, 0.9999), math.atan2(x, y)
        out.append((nr * math.sin(a), nr * math.cos(a), math.sqrt(max(0.0, 1.0 - nr * nr))))
    return out


def symmetric_move(vecs, i: int, target) -> list[tuple[float, float, float]]:
    """Symmetry Lock: dragging light *i* turns it to the pointer's azimuth and
    keeps its elevation, which every light then shares."""
    ox, oy, _ = vecs[i]
    r = math.hypot(ox, oy)
    a = math.atan2(target[0], target[1])
    z = math.sqrt(max(0.001, 1.0 - r * r))
    out = [(r * math.sin(math.atan2(x, y)), r * math.cos(math.atan2(x, y)), z) for x, y, _ in vecs]
    out[i] = (r * math.sin(a), r * math.cos(a), z)
    return out


def mirror(vecs, axis: str, index: int | None = None) -> list[tuple[float, float, float]]:
    """'h' swaps east and west, 'v' north and south; one light or all."""
    def flip(v):
        x, y, z = v
        return (-x, y, z) if axis == "h" else (x, -y, z)
    return [flip(v) if index is None or j == index else tuple(v) for j, v in enumerate(vecs)]


def copy_elevation(vecs, src: int) -> list[tuple[float, float, float]]:
    """Every light at light *src*'s elevation, each keeping its azimuth."""
    zs = vecs[src][2]
    rn = math.sqrt(max(0.0, 1.0 - zs * zs))
    out = []
    for x, y, _z in vecs:
        r = math.hypot(x, y)
        out.append((0.0, 0.0, zs) if r < 1e-6 else (x * rn / r, y * rn / r, zs))
    return out


def rotate(vecs, deg: float) -> list[tuple[float, float, float]]:
    """Turn every light *deg* degrees clockwise seen from above (azimuth + deg)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [(c * x + s * y, -s * x + c * y, z) for x, y, z in vecs]


def nudge(v, dx: float, dy: float, radius_px: float, step_px: float = 4.0):
    """An arrow key: the light moved *step_px* screen pixels on a dome of *radius_px*."""
    px, py = to_dome_point(v)
    return from_dome_point(px + dx * step_px / radius_px, py + dy * step_px / radius_px)


def shadow_azimuth(x0: float, y0: float, x1: float, y1: float) -> float:
    """The light's azimuth from an arrow drawn along a shadow on the
    photograph (image y down, its top north): the light is opposite."""
    shadow = math.degrees(math.atan2(x1 - x0, -(y1 - y0))) % 360.0
    return (shadow + 180.0) % 360.0


def read_vectors(path: str) -> list[tuple[float, float, float]]:
    """Import .lp / Session: a .helio session's stored lights, or from any
    other file the last three numbers of each line holding three or more
    (so "name x y z", "x y z" and "index x y z" all read).  +Y up, unit.

    HELIO 2 parsed a .helio line by line too, and a session is indented JSON
    with one number to a line, so it never found its lights."""
    vecs = []
    if path.lower().endswith(".helio"):
        from helio.io.session import read
        raw = read(path).light_vectors or []
    else:
        raw = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                nums = []
                for part in line.split():
                    try:
                        nums.append(float(part))
                    except ValueError:
                        pass
                if len(nums) >= 3:
                    raw.append(nums[-3:])
    for v in raw:
        x, y, z = (float(c) for c in v[:3])
        n = math.sqrt(x * x + y * y + z * z)
        if n > 0.01:
            vecs.append((x / n, y / n, z / n))
    if not vecs:
        raise ValueError("No light vectors found in that file.")
    return vecs


def merge(current, imported, how: str) -> list[tuple[float, float, float]]:
    """Multi-session consensus: 'average' or 'replace' by index; lights past
    the end of either list are left as they were."""
    out = [tuple(v) for v in current]
    for i in range(min(len(out), len(imported))):
        ix, iy, iz = (float(c) for c in imported[i][:3])
        if how == "average":
            cx, cy, cz = out[i]
            mx, my, mz = (ix + cx) / 2, (iy + cy) / 2, (iz + cz) / 2
            n = math.sqrt(mx * mx + my * my + mz * mz) or 1.0
            out[i] = (mx / n, my / n, mz / n)
        else:
            out[i] = (ix, iy, iz)
    return out


# ── EXIF ────────────────────────────────────────────────────────────────

def _exif(path):
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS
    with Image.open(path) as im:
        raw = im._getexif() if hasattr(im, "_getexif") else None
        if not raw:
            ex = im.getexif()
            raw = dict(ex)
            try:
                gps = ex.get_ifd(0x8825)
                if gps:
                    raw[0x8825] = dict(gps)
            except Exception:
                pass
    if not raw:
        return {}, {}
    exif = {TAGS.get(k, k): v for k, v in raw.items()}
    gps = exif.get("GPSInfo")
    gps_d = {GPSTAGS.get(k, k): v for k, v in gps.items()} if isinstance(gps, dict) else {}
    return exif, gps_d


def exif_estimate(paths, vecs, el_deg: float = 35.0):
    """The classic Auto-Estimate from EXIF: a GPS image bearing gives the
    azimuth; failing that, a fired flash spaces the frames evenly round the
    dome.  Elevation 35°.  Returns (lights, how many were set)."""
    out, found = [tuple(v) for v in vecs], 0
    n = len(out)
    for i, p in enumerate(paths[:n]):
        try:
            exif, gps = _exif(p)
        except Exception:
            continue
        az = None
        d = gps.get("GPSImgDirection")
        if d is not None:
            try:
                az = d[0] / d[1] if isinstance(d, tuple) else float(d)
            except (TypeError, ValueError, ZeroDivisionError):
                az = None
        flash = exif.get("Flash", 0)
        if az is None and isinstance(flash, int) and flash & 0x1:
            az = 360.0 * i / n
        if az is not None:
            out[i] = light_from_angles(az, el_deg)
            found += 1
    return out, found


# ── checking against a solve ─────────────────────────────────────────────

def load_plates(paths, shape, tone: str = "stored"):
    """Each frame as 0–1 grey at the normals' size (None where unreadable);
    'srgb' decodes to linear light, as the solve does."""
    import cv2
    h, w = shape
    from helio.core.frame_read import read_frame
    from helio.io.raw_loader import is_raw
    out = []
    for p in paths:
        g = read_frame(p, cv2.IMREAD_GRAYSCALE, linear=True, sharp=True)
        if g is None:
            out.append(None)
            continue
        g = cv2.resize(g, (w, h)).astype(np.float32)
        if tone == "srgb" and not is_raw(p):
            from helio.core.radiometry import srgb_to_linear
            g = srgb_to_linear(g, 255.0).astype(np.float32)
        else:
            g /= 255.0
        out.append(g)
    return out


def _basis(normals, albedo):
    N = np.asarray(normals, np.float32).reshape(-1, 3)
    ok = np.isfinite(N).all(axis=1) & (np.linalg.norm(N, axis=1) > 0.5)
    rho = None
    if albedo is not None:
        a = np.asarray(albedo, np.float32)
        if a.ndim == 3:
            a = a.mean(axis=2)
        rho = a.reshape(-1)
        rho = rho / (float(np.percentile(rho[ok], 99)) or 1.0) if ok.any() else rho
        ok &= np.isfinite(rho)
    return N, ok, rho


def score(lights, normals, plates, albedo=None) -> list[float]:
    """How well each light explains its frame: the correlation of the
    Lambertian prediction ρ·max(0, n·l) with the photograph, mapped to 0–1
    (0.5 where it cannot be told).  Without an albedo, ρ = 1 as in HELIO 2."""
    N, ok, rho = _basis(normals, albedo)
    out = []
    for i, l in enumerate(lights):
        pl = plates[i] if i < len(plates) else None
        if pl is None:
            out.append(0.5)
            continue
        pred = np.maximum(0.0, N @ np.asarray(l, np.float32))
        if rho is not None:
            pred = pred * rho
        a, b = pred[ok], pl.reshape(-1)[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else float("nan")
        out.append(min(1.0, max(0.0, (c + 1) / 2)) if math.isfinite(c) else 0.5)
    return out


ORIENT_SIDE = 400                         # the frames' long side for the check, in pixels


def _integrability(L, I):
    """How far the Lambertian normals solved from *I* with lights *L* are from a
    surface: median |curl| of the gradient field over its median size.  A light set
    mirrored on one axis flips one slope and leaves a field no surface has."""
    n, h, w = I.shape
    X = np.linalg.lstsq(L, I.reshape(n, -1), rcond=None)[0].T.reshape(h, w, 3)
    nz = np.maximum(X[..., 2], 1e-3 * float(np.abs(X).max() or 1.0))
    p = np.clip(-X[..., 0] / nz, -5, 5)            # dz/dx, image x right
    q = np.clip(X[..., 1] / nz, -5, 5)             # dz/dy, image y down
    py, px = np.gradient(p)
    qy, qx = np.gradient(q)
    size = float(np.median(np.abs(px) + np.abs(py) + np.abs(qx) + np.abs(qy)))
    return float(np.median(np.abs(py - qx))) / max(size, 1e-9)


def _falloff(L, I):
    """Each frame's broad brightening, over the mean frame (the object's own pattern
    divided out), against its light's direction across the picture: the side nearer
    a lamp is lit more.  Mean cosine, -1 to 1."""
    import cv2
    m = I.mean(axis=0) + 1e-3
    sig = max(I.shape[1:]) / 12.0
    cos = []
    for l, f in zip(L, I):
        r = cv2.GaussianBlur(f / m, (0, 0), sig)
        gx = float(np.gradient(r, axis=1).mean())
        gy = -float(np.gradient(r, axis=0).mean())   # +Y up, as the lights
        a = math.hypot(l[0], l[1])
        g = math.hypot(gx, gy)
        if a > 1e-6 and g > 1e-9:
            cos.append((gx * l[0] + gy * l[1]) / (a * g))
    return float(np.mean(cos)) if cos else 0.0


def orientation_plates(paths):
    """The frames for ``orientation_check``: grey 0-1, about ORIENT_SIDE pixels on the long
    side, area-averaged (a plain resize of a 9000-pixel frame aliases its fine texture and
    hides the mirror in the integrability).  None where unreadable."""
    import cv2
    from helio.core.frame_read import read_frame
    out = []
    for p in paths:
        g = read_frame(p, cv2.IMREAD_REDUCED_GRAYSCALE_4)
        if g is None:
            out.append(None)
            continue
        g = np.asarray(g, np.float32)
        h, w = g.shape[:2]
        k = min(1.0, ORIENT_SIDE / max(h, w))
        if k < 1.0:
            g = cv2.resize(g, (max(8, int(w * k)), max(8, int(h * k))), interpolation=cv2.INTER_AREA)
        out.append(g / 255.0)
    return out


def orientation_check(lights, plates) -> dict:
    """Do the photographs agree with which way round the lights are?

    A hand-placed dome is easily the photographer's mirror image (placed facing the
    object instead of from the camera): the solve then fits every frame exactly as
    well, so ``score`` cannot see it, but the normals are no longer a surface, the
    relief reads wrong under a raking light and the depth comes out poor.  Two
    witnesses, from the frames alone: how integrable the normals are with the lights
    as given and mirrored east-west or north-south (a mirror on both axes is the
    same surface turned inside out, which no test can tell), and on which side
    each frame is brighter.  ``suggest`` is "h" or "v" only when both agree that
    mirror is clearly better, else None."""
    import cv2
    L = np.asarray(lights, np.float64)
    keep = [i for i in range(len(L)) if i < len(plates) and plates[i] is not None]
    out = {"suggest": None, "curl": {}, "falloff": {}}
    if len(keep) < 4:
        return out
    L = L[keep]
    I = [np.asarray(plates[i], np.float32) for i in keep]
    h, w = I[0].shape[:2]
    s = min(1.0, ORIENT_SIDE / max(h, w))
    if s < 1.0:
        I = [cv2.resize(f, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
             for f in I]
    if any(f.shape != I[0].shape for f in I):
        I = [cv2.resize(f, (I[0].shape[1], I[0].shape[0]), interpolation=cv2.INTER_AREA) for f in I]
    I = np.stack(I)
    for k, M in (("given", (1, 1, 1)), ("h", (-1, 1, 1)), ("v", (1, -1, 1))):
        Lk = L * np.asarray(M, np.float64)
        out["curl"][k] = _integrability(Lk, I)
        out["falloff"][k] = _falloff(Lk, I)
    c, f = out["curl"], out["falloff"]
    for k in ("h", "v"):
        if c[k] < 0.75 * c["given"] and f[k] > f["given"] + 0.3 and f[k] > 0.5:
            out["suggest"] = k
    return out


def _residual(pred, img):
    """Mean squared error after the best gain: each frame's own exposure and
    lamp brightness are unknown, so only the shading pattern is compared."""
    pp = (pred * pred).sum(axis=0)
    k = np.where(pp > 1e-12, (pred * img[:, None]).sum(axis=0) / np.maximum(pp, 1e-12), 0.0)
    return ((k * pred - img[:, None]) ** 2).mean(axis=0)


def refine(lights, normals, plates, iters: int = 20, albedo=None, step: float = 0.04,
           decay: float = 0.92, progress=None):
    """The classic Vector Refinement: each light tries ±step along x, y and z
    (renormalised, kept above the horizon), keeps a move that fits its frame
    better, and the step shrinks by *decay* each round.

    The fit is the shading pattern after the best gain (``_residual``) rather
    than raw brightness: HELIO 2 compared max(0, n·l) with the photograph
    directly, so a light was pulled up or down to match the frame's exposure.
    Returns (lights, moves made, RMS before, RMS after), RMS per light."""
    N, ok, rho = _basis(normals, albedo)
    Nk = N[ok]
    rk = rho[ok] if rho is not None else None
    L = [np.asarray(v, np.float64) / (np.linalg.norm(v) or 1.0) for v in lights]
    imgs = [None if (i >= len(plates) or plates[i] is None) else plates[i].reshape(-1)[ok]
            for i in range(len(L))]

    def err(C, i):
        pred = np.maximum(0.0, Nk @ C.astype(np.float32))
        if rk is not None:
            pred = pred * rk[:, None]
        return _residual(pred, imgs[i])

    before = [float(np.sqrt(err(l[:, None], i)[0])) if imgs[i] is not None else float("nan")
              for i, l in enumerate(L)]
    moves = 0
    for it in range(int(iters)):
        for i, l in enumerate(L):
            if imgs[i] is None:
                continue
            cands = [l]
            for dim in range(3):
                for sign in (1.0, -1.0):
                    c = l.copy()
                    c[dim] += sign * step
                    n = np.linalg.norm(c)
                    if n < 0.01:
                        continue
                    c = c / n
                    if c[2] < 0.01:
                        continue
                    cands.append(c)
            e = err(np.stack(cands, axis=1), i)
            j = int(np.argmin(e))
            if j > 0 and e[j] < e[0]:
                L[i] = cands[j]
                moves += 1
        step *= decay
        if progress:
            progress((it + 1) / max(1, int(iters)))
    after = [float(np.sqrt(err(l[:, None], i)[0])) if imgs[i] is not None else float("nan")
             for i, l in enumerate(L)]
    return [tuple(float(c) for c in l) for l in L], moves, before, after


# ── an RTIBuilder / Relight project folder ───────────────────────────────

def read_rti_project(folder: str) -> dict:
    """The classic Import RTI Project Folder: the first .lp in the folder (or
    one level down), its vectors matched to the photographs by file name
    (case-insensitive, then ignoring the extension), or in natural (shot)
    order when the .lp names none of them (x-3 before x-10, not alphabetical).  Raises ValueError with the reason when it cannot.

    Returns {"lp": path, "paths": [...], "lights": (N, 3) +Y up,
    "unmatched": [...], "entries": count, "paired": "name" or "order"}."""
    from helio.io.capture import capture_extensions, natural_key
    lp_path = None
    for pattern in (os.path.join(folder, "*.lp"), os.path.join(folder, "*", "*.lp")):
        hits = glob.glob(pattern)
        if hits:
            lp_path = sorted(hits, key=natural_key)[0]
            break
    if lp_path is None:
        raise ValueError(f"No .lp light-position file in {folder}. Expected a file like "
                         "'sphere.lp' or 'lights.lp' in the project folder.")
    entries = []
    with open(lp_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].startswith("#"):
                continue
            if len(parts) == 1:
                continue
            try:
                x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            except (ValueError, IndexError):
                continue
            m = math.sqrt(x * x + y * y + z * z)
            if m <= 0:
                continue
            name = " ".join(parts[:-3]) if len(parts) >= 4 else None
            entries.append((name, x / m, y / m, z / m))
    if len(entries) < 4:
        raise ValueError(f"Only {len(entries)} light vector(s) in {os.path.basename(lp_path)}; "
                         "at least 4 are needed.")
    exts = {e.lower() for e in capture_extensions()}
    dirs = [folder] + ([os.path.dirname(lp_path)] if os.path.dirname(lp_path) != folder else [])
    imgs = sorted({os.path.join(d, f) for d in dirs for f in os.listdir(d)
                   if os.path.splitext(f)[1].lower() in exts}, key=natural_key)
    if not imgs:
        raise ValueError(f"No photographs in {folder} beside the .lp file.")
    by_base = {os.path.basename(p).lower(): p for p in imgs}
    paths, vecs, unmatched = [], [], []
    by_name = any(e[0] is not None for e in entries)
    if by_name:
        def named(nm):
            key = os.path.basename(nm or "").lower()
            return key in by_base or any(os.path.splitext(k)[0] == os.path.splitext(key)[0]
                                         for k in by_base)
        # Names that are none of the photographs' (RTIBuilder's 01.jpg … beside a
        # camera's x-3.JPG …): pair in shot order below, not "matched 0".
        by_name = any(named(e[0]) for e in entries if e[0] is not None)
    if by_name:
        for name, x, y, z in entries:
            key = os.path.basename(name or "").lower()
            hit = by_base.get(key)
            if hit is None:
                stem = os.path.splitext(key)[0]
                hit = next((v for k, v in by_base.items() if os.path.splitext(k)[0] == stem), None)
            if hit is None:
                unmatched.append(name)
            else:
                paths.append(hit)
                vecs.append([x, y, z])
    else:
        n = min(len(entries), len(imgs))
        paths = imgs[:n]
        vecs = [list(e[1:]) for e in entries[:n]]
        if any(e[0] is not None for e in entries):
            unmatched = [f"(no name in the .lp is a photograph's: paired in shot order, "
                         f"{os.path.basename(entries[0][0] or '')} with "
                         f"{os.path.basename(imgs[0])})"]
        if len(entries) != len(imgs):
            unmatched.append(f"(order mismatch: {len(entries)} vectors, {len(imgs)} images)")
    if len(paths) < 4:
        raise ValueError(f"Matched {len(paths)} images to light vectors (at least 4 are "
                         f"needed). Check that the names in {os.path.basename(lp_path)} match "
                         "the photographs." + (f" Unmatched: {', '.join(map(str, unmatched[:5]))}"
                                               if unmatched else ""))
    return {"lp": lp_path, "paths": paths, "lights": np.array(vecs, np.float32),
            "unmatched": unmatched, "entries": len(entries),
            "paired": "name" if by_name else "order"}
