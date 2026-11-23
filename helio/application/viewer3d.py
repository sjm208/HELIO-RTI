"""The 3D viewer, GUI-free: what it is given and how it is driven (GEO-001…014, EXM-004…008).

The viewer itself is helio.render.viewer.run_opengl_viewer, a separate
process that polls a dict of multiprocessing values (the "shared state")
every 50 ms: height, relief, clay, lighting, tone, spin and capture flags.
Both apps launch the same viewer with the same inputs from here:

* ``prepare`` builds the colour, depth and normal arrays for the four
  windows (the standard view, Auto-Flatten, Inverted and Heatmap), capped at
  9 MP as the classic did;
* ``calibrate`` puts the measured scale into the viewer's pixels;
* ``start_spin`` / ``wait_spin`` / ``assemble_spin`` record a turntable or
  flyover and write it as a GIF or MP4;
* ``clean_render`` asks the viewer for a frame without its HUD;
* ``sharpen`` is the Detail Sharpening Pipeline.

Auto-Flatten and Heatmap cut the edge trim out themselves but did not tell
the viewer, which trimmed again: their meshes lost twice the margin (KD-030).
They now say the data is already trimmed.
"""
from __future__ import annotations

import math
import multiprocessing
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

# Engines whose render, not the albedo, colours the mesh (the classic lists, joined).
ANALYTICAL = ("Positive Openness", "Negative Openness", "Local Dominance",
              "Normals Unsharp Masking", "Slope Magnitude", "Specular Enhancement",
              "Image Quotient (IQ)", "Specular View", "Ambient Occlusion (AO)",
              "PS Normal-Map Lit (Single)", "PS Normal-Map Lit (8-Dir)", "Sky-View Factor (SVF)",
              "Slope Gradient", "Multi-Scale Relief Model (MSRM)", "PCA 1", "PCA 2", "PCA 3",
              "Colour ICA (IC1-3)", "RGB-PS R-B Diff", "Retinex Enhancement")
PIXEL_CAP = 9_000_000
KINDS = ("standard", "flattened", "inverted", "heatmap")
SPIN_SPEEDS = {"Slow": 0.25, "Normal": 0.5, "Fast": 1.2, "Very Fast": 2.5}
SPIN_FRAMES = (36, 48, 72, 120)
SPIN_MODES = {1: ("Turntable", ".gif"), 2: ("Turntable", ".mp4"), 3: ("Flyover", ".gif"),
              4: ("Flyover", ".mp4")}
CLAY = {"Grey": (0.6, 0.6, 0.6), "Red": (0.8, 0.2, 0.2), "Blue": (0.2, 0.4, 0.8),
        "Green": (0.2, 0.7, 0.3), "Mustard": (0.8, 0.7, 0.2)}
SHARPEN_DEFAULTS = {"strength": 1.0, "smooth": 0, "usm_amt": 0.0, "usm_rad": 1.0,
                    "hf_boost": 0.0, "hf_rad": 2.0, "clahe": 0.0}


# ── the shared state ────────────────────────────────────────────────────

def make_state(values: dict | None = None) -> dict:
    """A fresh shared state (multiprocessing values), set from *values*."""
    from helio.config import SHARED_STATE_DEFAULTS
    st = {k: multiprocessing.Value(tc, v) for k, (tc, v) in SHARED_STATE_DEFAULTS.items()}
    sync(st, values or {})
    return st


def write(state: dict, key: str, value) -> None:
    v = state.get(key)
    if v is None:
        return
    v.value = int(value) if isinstance(v.value, int) else float(value)


def sync(state: dict, values: dict) -> None:
    for k, val in values.items():
        try:
            write(state, k, val)
        except (TypeError, ValueError):
            pass


def set_clay(state: dict, name: str, on: bool) -> None:
    write(state, "use_custom_clay", 1 if on else 0)
    r, g, b = CLAY.get(name, CLAY["Grey"])
    write(state, "clay_r", r)
    write(state, "clay_g", g)
    write(state, "clay_b", b)


# ── what the viewer is given ────────────────────────────────────────────

@dataclass
class ViewerInput:
    rgb: np.ndarray
    d_full: np.ndarray
    d_hp: np.ndarray
    normals: np.ndarray
    skip_trim: int            # 1: the arrays are already the region; the viewer must not trim
    live: bool = True         # the depth may be updated in place (the standard view only)


def trim_box(w: int, h: int, edge_trim_pct: float, crop=None):
    """The export region (helio.application.exports.trim_box: one definition for every export)."""
    from helio.application.exports import trim_box as region
    return region(w, h, edge_trim_pct, crop)


def quadric_flatten(depth) -> np.ndarray:
    """Auto-Flatten: subtract the best-fitting quadric surface (x², y², xy,
    x, y, 1 on normalised coordinates) and rescale to 0–1."""
    d = np.asarray(depth, np.float32)
    h, w = d.shape
    # Six coefficients: fitted on at most ~250 000 points (every k-th pixel),
    # then the quadric evaluated everywhere.  Fitting all 4.7 million points
    # of a preview took 0.4 s for the same six numbers.
    k = max(1, int(np.ceil(np.sqrt(h * w / 250_000))))
    ys, xs = np.mgrid[0:h:k, 0:w:k]

    def design(xx, yy):
        xf = xx.astype(np.float64).ravel() / max(w, 1)
        yf = yy.astype(np.float64).ravel() / max(h, 1)
        return np.column_stack([xf ** 2, yf ** 2, xf * yf, xf, yf, np.ones_like(xf)])
    coeffs, *_ = np.linalg.lstsq(design(xs, ys), d[::k, ::k].ravel().astype(np.float64), rcond=None)
    yy, xx = np.mgrid[0:h, 0:w]
    xf, yf = xx.astype(np.float32) / max(w, 1), yy.astype(np.float32) / max(h, 1)
    c = coeffs.astype(np.float32)
    flat = d - (c[0] * xf * xf + c[1] * yf * yf + c[2] * xf * yf + c[3] * xf + c[4] * yf + c[5])
    flat -= flat.min()
    mx = flat.max()
    return flat / mx if mx > 1e-9 else flat


def invert01(d) -> np.ndarray:
    d2 = -(np.asarray(d, np.float32))
    lo, hi = d2.min(), d2.max()
    return (d2 - lo) / (hi - lo + 1e-9)


def _blur(z, sigma: float):
    """A Gaussian blur; a very wide one on a reduced copy.  These blurs keep
    only broad shape (a warp, a bow, the bowl integration leaves), which the
    reduction does not touch, and at full size a 1000 px sigma takes seconds."""
    z = np.asarray(z, np.float32)
    if sigma <= 0:
        return z.copy()
    if sigma <= 24:
        return cv2.GaussianBlur(z, (0, 0), float(sigma))
    h, w = z.shape
    k = 12.0 / sigma
    sw, sh = max(4, int(round(w * k))), max(4, int(round(h * k)))
    small = cv2.resize(z, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), float(sigma) * sw / w)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def relief_surface(d_full, d_hp, *, shape_blend: float = 0.55, hp_sigma: float = 0.0,
                   mesh_smooth: float = 0.0, warp_rem: float = 0.0, micro: float = 0.4,
                   depth_clip_pct: float = 0.5, hybrid: bool = False, hybrid_mix: float = 0.12,
                   kind: str = "standard", smooth_scale: float = 1.0, x_tilt: float = 0.0,
                   y_tilt: float = 0.0) -> np.ndarray:
    """The height a 3D view meshes, from the full depth and the processed one.

    The classic viewer's own steps (helio.render.viewer), in its order and
    with its settings' names, so the 3D in the window and the separate
    window show the same surface: the detail-vs-shape blend (with its
    progressive high-pass above 0.4), the warp flatten, smoothing, the broad
    warp removal, the fine-detail boost, the percentile clip and the tilt
    correction (*x_tilt*, *y_tilt* degrees: a copy stand that is not level).  *kind*
    "flattened" takes out the best-fitting quadric first, "inverted" turns
    the relief over.  *smooth_scale* converts the smoothing's pixels from the
    classic's mesh to these arrays.  Heights stay in the depth's own units
    (not stretched to 0-1), so a scale fitted to them still means something."""
    from helio.core.parallel import quantiles
    d_full = np.asarray(d_full, np.float32)
    d_hp = np.asarray(d_hp, np.float32)
    if d_full.shape != d_hp.shape:
        d_full = cv2.resize(d_full, (d_hp.shape[1], d_hp.shape[0]), interpolation=cv2.INTER_AREA)
    if kind == "flattened":
        d_full = d_hp = quadric_flatten(d_hp)
    elif kind == "inverted":
        d_full, d_hp = invert01(d_full), invert01(d_hp)
    h, w = d_hp.shape
    if hybrid:
        def n01(a):
            lo, hi = float(a.min()), float(a.max())
            return ((a - lo) / (hi - lo)).astype(np.float32) if hi > lo + 1e-9 else np.zeros_like(a)
        z = n01(n01(d_full) * (1.0 - hybrid_mix) + n01(d_hp) * hybrid_mix)
    else:
        sb = max(0.0, min(1.0, float(shape_blend)))
        z = d_full * (1.0 - sb) + d_hp * sb
        if sb > 0.4:
            z = z - _blur(z, max(3.0, (sb - 0.4) * 0.20 * float(np.hypot(h, w))))
    if hp_sigma > 0.01:
        z = z - _blur(z, float(hp_sigma) * max(h, w) * 0.4)
    if mesh_smooth > 0:
        z = cv2.GaussianBlur(z, (0, 0), float(mesh_smooth) * max(smooth_scale, 1e-3))
    if warp_rem > 0:
        z = z - _blur(z, max(h, w) * (0.05 + float(warp_rem) * 0.45))
    if micro > 0:
        base = _blur(z, max(h, w) * 0.05)
        z = base + (z - base) * (1.0 + float(micro))
    c = max(0.01, min(4.9, float(depth_clip_pct)))
    lo, hi = quantiles(z, [c, 100.0 - c])                # percentages, as np.percentile
    z = np.clip(z, lo, hi)
    if abs(x_tilt) > 0.001 or abs(y_tilt) > 0.001:
        xg, yg = np.meshgrid(np.linspace(-0.5, 0.5, w, dtype=np.float32),
                             np.linspace(-0.5, 0.5, h, dtype=np.float32))
        z = z - (xg * np.float32(np.tan(np.radians(x_tilt)))
                 + yg * np.float32(np.tan(np.radians(y_tilt))))
    return z.astype(np.float32)


# ── recordings of Helio's own 3D view ───────────────────────────────────

FLYOVER_FRAMES = 360          # 12 s at 30 fps
VIDEO_SIZE = (1920, 1080)


def flyover_pose(i: int, n: int, az0: float = 0.0, el0: float = 42.0, dist0: float = 3.2):
    """(azimuth, elevation, distance) for frame *i* of *n* of a flyover.

    One smooth loop, the last frame running into the first so the video
    repeats without a jump: once round at a steady pace, dipping low on the
    far side (where the light rakes across the relief) and rising again,
    moving in towards the middle to show the detail and back out.  The
    classic flyover circled, rose to straight overhead and stopped there."""
    t = (i % max(n, 1)) / float(max(n, 1))
    az = (az0 + 360.0 * t) % 360.0
    el = max(14.0, min(80.0, el0 + 20.0 * math.cos(2.0 * math.pi * t)))   # high at the start, low halfway
    dist = dist0 * (1.0 - 0.28 * math.sin(math.pi * t) ** 2)               # closest a quarter and three quarters round
    return az, el, dist


def turntable_pose(i: int, n: int, az0: float = 0.0, el0: float = 42.0, dist0: float = 3.2):
    """(azimuth, elevation, distance) for frame *i* of *n*: once round, level."""
    return (az0 + 360.0 * (i % max(n, 1)) / float(max(n, 1))) % 360.0, el0, dist0


VIDEO_STYLES = {
    "turntable": ("Turntable", "Once round, level: the plainest record of the whole object."),
    "flyover": ("Flyover loop", "Once round, dipping low where the light rakes and moving in on "
                                "the detail; ends where it began, so it loops."),
    "overhead": ("Overhead orbit", "Looking down from high above as it circles: the whole face "
                                   "always in view, the relief turning under you."),
    "light": ("Light sweep", "The camera still, the light circling low round the object, as in "
                             "RTI: every mark catches the light in turn."),
}
VIDEO_LENGTHS = (4, 6, 8, 12, 20, 30, 60, 90, 120)    # seconds for one turn: the speed
VIDEO_SIZES = {"720p": (1280, 720), "HD": (1920, 1080), "4K": (3840, 2160),
               "Square": (1080, 1080), "Portrait": (1080, 1920), "Social 4:5": (1080, 1350)}
VIDEO_FPS = 30
VIDEO_RATES = (24, 30, 60)                    # frames a second


def video_pose(style: str, i: int, n: int, az0: float = 0.0, el0: float = 42.0,
               dist0: float = 3.2, light_el: float = 30.0, reverse: bool = False,
               light_az0: float = 0.0):
    """(azimuth, elevation, distance, light) for frame *i* of *n* of a
    recording in *style* (VIDEO_STYLES); *light* is (azimuth, elevation) of
    the light in degrees for the light sweep, None otherwise (the light as
    set).  *reverse* turns the other way round; the light sweep starts at
    *light_az0*.  Every style ends where it began, so a video loops."""
    if reverse:
        i = (n - i) % max(n, 1)
    if style == "flyover":
        return (*flyover_pose(i, n, az0, el0, dist0), None)
    if style == "overhead":
        t = (i % max(n, 1)) / float(max(n, 1))
        return (az0 + 360.0 * t) % 360.0, 72.0, dist0 * 0.95, None
    if style == "light":
        t = (i % max(n, 1)) / float(max(n, 1))
        return az0, el0, dist0, ((light_az0 + 360.0 * t) % 360.0, float(light_el))
    return (*turntable_pose(i, n, az0, el0, dist0), None)


def object_extent(pictures, background, tol: float = 14.0):
    """(x0, y0, x1, y1) as fractions of the frame: the box round the object in every
    one of *pictures* (frames drawn on a plain *background*, RGB 0-255), or None
    when nothing differs from it."""
    bg = np.asarray(background, np.float32).reshape(1, 1, 3)
    box = None
    for p in pictures:
        a = np.asarray(p, np.float32)
        ys, xs = np.nonzero(np.abs(a - bg).max(axis=2) > tol)
        if len(xs) < 20:
            continue
        h, w = a.shape[:2]
        b = (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]),
                                     max(box[2], b[2]), max(box[3], b[3]))
    return box


def fitted_frame(box, size, margin: float = 0.04, max_side: int = 8192):
    """How to draw a video that the object fills: (draw_w, draw_h, crop, out_w,
    out_h).  *box* (object_extent) is where the object goes in a frame of the
    chosen *size*'s shape; the frames are drawn larger by as much as the crop
    cuts away, then cut to (x, y, w, h), whose long side is *size*'s long side.
    Even sizes throughout, for the video encoder."""
    W, H = size
    x0, y0, x1, y1 = box
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    x0, y0, x1, y1 = max(0.0, x0 - mx), max(0.0, y0 - my), min(1.0, x1 + mx), min(1.0, y1 + my)
    fw, fh = max(1e-3, x1 - x0), max(1e-3, y1 - y0)
    long_out = max(W, H)
    aspect = (fw * W) / (fh * H)                       # the object's shape, in pixels
    out_w, out_h = (long_out, long_out / aspect) if aspect >= 1 else (long_out * aspect, long_out)
    k = min(out_w / (fw * W), out_h / (fh * H), max_side / float(max(W, H)))
    dw, dh = int(round(W * k / 2)) * 2, int(round(H * k / 2)) * 2
    cw, ch = int(fw * dw) // 2 * 2, int(fh * dh) // 2 * 2
    cx, cy = int(round(x0 * dw)), int(round(y0 * dh))
    cx, cy = min(cx, dw - cw), min(cy, dh - ch)
    return dw, dh, (cx, cy, cw, ch), cw, ch


REPEATS = (1, 2, 3, 4, 5)       # times round in one MP4

SIZE_LIMITS = {0: ("No limit", "As large as the size and length make it."),
               20: ("Under 20 MB", "For PowerPoint and Keynote: the file is kept under 20 MB, "
                                   "made smaller (and a GIF less smooth) only as far as it must be."),
               10: ("Under 10 MB", "For e-mail: the file is kept under 10 MB.")}


def write_video(frames, path: str, fps: int = 30, max_mb: float = 0.0, repeat: int = 1) -> str:
    """RGB uint8 frames as an MP4 (H.264, which browsers, QuickTime and
    PowerPoint play; MPEG-4 where H.264 is not to hand) or, by the file's
    name, a GIF at half the frames (all of them at 20 a second or fewer).

    With *max_mb*, the file is kept under that many megabytes, the length
    unchanged: an MP4 is encoded to fit where ffmpeg is installed, else its
    frames are made smaller; a GIF first drops to half its frames (not below
    10 a second), then its frames are made smaller.

    *repeat* writes an MP4's frames that many times over, so it plays round
    that often where the player does not loop it (a GIF loops by itself)."""
    if not frames:
        raise RuntimeError("No frames were recorded.")
    if repeat > 1 and not path.lower().endswith(".gif"):
        frames = list(frames) * int(repeat)
    limit = int(max_mb * 1e6) if max_mb else 0
    gif = path.lower().endswith(".gif")
    step = 1 if fps <= 20 else 2
    scale = 1.0
    for _ in range(10):
        fr = frames
        if scale < 0.999:
            h, w = frames[0].shape[:2]
            size = (max(16, int(w * scale) // 2 * 2), max(16, int(h * scale) // 2 * 2))
            fr = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]
        if gif:
            _write_gif(fr, path, fps, step)
        else:
            _write_mp4(fr, path, fps, limit)
        got = os.path.getsize(path)
        if not limit or got <= limit:
            return path
        if gif and fps / step / 2 >= 10:
            step *= 2
            continue
        scale *= max(0.5, min(0.95, math.sqrt(limit / got) * 0.95))
    raise RuntimeError(f"It could not be made smaller than {max_mb:g} MB: try a shorter length "
                       f"or a smaller size.")


def _write_gif(frames, path: str, fps: int, step: int):
    from PIL import Image
    ims = [Image.fromarray(f).convert("P", palette=Image.Palette.ADAPTIVE) for f in frames[::step]]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 * step / fps), loop=0)


def _ffmpeg() -> str:
    import shutil
    for p in (shutil.which("ffmpeg"), "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if p and os.path.isfile(p):
            return p
    return ""


def _write_mp4(frames, path: str, fps: int, limit: int = 0):
    h, w = frames[0].shape[:2]
    h, w = h - h % 2, w - w % 2
    ff = _ffmpeg() if limit else ""
    if ff:                              # H.264 at the bitrate that fits the limit
        import subprocess
        rate = int(limit * 8 * 0.92 / max(len(frames) / float(fps), 0.1))
        cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-preset", "slow", "-b:v", str(rate),
               "-maxrate", str(int(rate * 1.1)), "-bufsize", str(rate * 2),
               "-movflags", "+faststart", path]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            for f in frames:
                p.stdin.write(np.ascontiguousarray(f[:h, :w, :3]).tobytes())
            p.stdin.close()
            if p.wait() == 0 and os.path.getsize(path) > 0:
                return
        except Exception:                       # noqa: BLE001  OpenCV below, as without a limit
            pass
    h, w = h - h % 2, w - w % 2
    vw = None
    for code in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), fps, (w, h))
        if vw.isOpened():
            break
        vw.release()
    if vw is None or not vw.isOpened():
        raise RuntimeError("The video could not be written.")
    for f in frames:
        vw.write(cv2.cvtColor(np.ascontiguousarray(f[:h, :w]), cv2.COLOR_RGB2BGR))
    vw.release()
    return path


def _cap(inp: ViewerInput) -> ViewerInput:
    h, w = inp.rgb.shape[:2]
    if h * w <= PIXEL_CAP:
        return inp
    s = math.sqrt(PIXEL_CAP / (h * w))
    nw, nh = int(w * s), int(h * s)
    return ViewerInput(cv2.resize(inp.rgb, (nw, nh), interpolation=cv2.INTER_AREA),
                       cv2.resize(inp.d_full, (nw, nh), interpolation=cv2.INTER_AREA),
                       cv2.resize(inp.d_hp, (nw, nh), interpolation=cv2.INTER_AREA),
                       cv2.resize(inp.normals, (nw, nh), interpolation=cv2.INTER_LINEAR),
                       inp.skip_trim, inp.live)


def prepare(kind: str, D, *, engine: str, colour: Callable, depth_normalised=None,
            edge_trim_pct: float = 2.0, crop=None, pretrimmed: bool = False) -> ViewerInput:
    """The arrays for one of the four viewer windows.

    *colour(engine, h, w, crop_coords)* renders an analytical engine for the
    mesh's colour; other engines colour it with the albedo.  The standard and
    inverted views give the viewer the whole frame and let it apply the edge
    trim (unless a crop or a pre-trimmed region is in force); Auto-Flatten
    and Heatmap work on the trimmed region and say so."""
    if kind not in KINDS:
        raise ValueError(kind)
    W, H = D.w, D.h
    dn = depth_normalised if depth_normalised is not None else D.depth_vis_p
    if kind in ("standard", "inverted"):
        if crop or pretrimmed:
            x1, y1, x2, y2 = crop if crop else (0, 0, W, H)
            skip = 1
        else:
            x1, y1, x2, y2 = 0, 0, W, H
            skip = 0
    else:
        x1, y1, x2, y2 = trim_box(W, H, edge_trim_pct, crop)
        skip = 1
    region = (x1, y1, x2, y2)
    if kind == "heatmap":
        d = np.clip(D.depth_vis_p[y1:y2, x1:x2], 0, 1)
        rgb = cv2.cvtColor(cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                           cv2.COLOR_BGR2RGB)
    elif engine in ANALYTICAL:
        rgb = colour(engine, y2 - y1, x2 - x1, region)
    else:
        rgb = D.albedo_rgb_full[y1:y2, x1:x2].copy()
    normals = D.normals_full[y1:y2, x1:x2].copy()
    d_hp = D.depth_vis_p[y1:y2, x1:x2].astype(np.float32)
    d_full = np.asarray(dn)[y1:y2, x1:x2].astype(np.float32)
    if kind == "inverted":
        d_hp, d_full = invert01(d_hp), invert01(d_full)
    elif kind == "flattened":
        d_hp = d_full = quadric_flatten(d_hp)
    return _cap(ViewerInput(np.ascontiguousarray(rgb), d_full, d_hp, normals, skip,
                            live=kind == "standard"))


def calibrate(state: dict, orig_w: int, inp: ViewerInput, mm_per_px: float = 0.0,
              z_mm_per_unit: float = 0.0) -> None:
    """The measured scale in the viewer's pixels (it may be given a smaller copy)."""
    ph, pw = inp.rgb.shape[:2]
    s = pw / max(1.0, float(orig_w or 1))
    mm = float(mm_per_px or 0.0)
    write(state, "mm_per_px", (mm / s) if s > 0 else mm)
    write(state, "z_mm_per_unit", float(z_mm_per_unit or 0.0))
    write(state, "img_w", int(pw))
    write(state, "img_h", int(ph))
    write(state, "viewer_skip_trim", inp.skip_trim)


def launch(inp: ViewerInput, state: dict, spec=None):
    """Start the viewer process on *inp*; *spec* is a live depth channel."""
    from helio.render.viewer import run_opengl_viewer
    args = (inp.rgb, inp.d_full, inp.d_hp, inp.normals, state) + ((spec,) if spec is not None else ())
    proc = multiprocessing.Process(target=run_opengl_viewer, args=args)
    proc.start()
    return proc


# ── turntable and flyover ───────────────────────────────────────────────

def spin_dir() -> str:
    return os.path.join(tempfile.gettempdir(), "helio3d_spin")


def start_spin(state: dict, mode: int, frames: int = 72) -> int:
    """Ask the open viewer to record *frames* (a flyover at least 288)."""
    total = int(frames)
    if mode in (3, 4) and total < 120:
        total = 288
    write(state, "spin_total", total)
    write(state, "spin_capture", mode)
    write(state, "spin_frames_done", 0)
    return total


def wait_spin(state: dict, total: int, timeout: float = 120.0,
              alive: Callable | None = None) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = state.get("spin_frames_done")
        if v is not None and v.value >= total:
            return True
        if alive is not None and not alive():
            return False
        time.sleep(0.15)
    return False


def assemble_spin(mode: int, out_path: str, frames_dir: str | None = None) -> tuple[str, int]:
    """The captured frames as a GIF (18 fps) or MP4 (24 fps)."""
    import glob
    paths = sorted(glob.glob(os.path.join(frames_dir or spin_dir(), "frm_*.png")))
    frames = []
    for fp in paths:
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        frames.append(img)
    if not frames:
        raise RuntimeError("No frames were captured.")
    if mode in (1, 3):
        from PIL import Image
        pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        pil[0].save(out_path, save_all=True, append_images=pil[1:], loop=0,
                    duration=int(1000 / 18), optimize=False)
    else:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
        for f in frames:
            vw.write(f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)))
        vw.release()
    return out_path, len(frames)


# ── a clean frame ───────────────────────────────────────────────────────

def clean_render(state: dict, out_path: str, timeout: float = 8.0) -> str:
    """The viewer's current frame without its HUD, copied to *out_path*."""
    import shutil
    marker = os.path.join(tempfile.gettempdir(), "helio_last_clean_export.txt")
    if os.path.exists(marker):
        os.remove(marker)
    write(state, "clean_export", 1)
    t0 = time.time()
    while time.time() - t0 < timeout and not os.path.exists(marker):
        time.sleep(0.1)
    if not os.path.exists(marker):
        raise TimeoutError("The viewer did not respond in time. Click the viewer window "
                           "first to give it focus.")
    with open(marker) as f:
        src = f.read().strip()
    if not os.path.exists(src):
        raise FileNotFoundError(f"Render file not found: {src}")
    shutil.copy2(src, out_path)
    return out_path


# ── the Detail Sharpening Pipeline ──────────────────────────────────────

def sharpen(normals, depth, params: dict):
    """Normal strength, then unsharp mask, high-frequency boost and CLAHE on
    the depth.  Returns new (normals, depth).

    Strength scales the normals' detail above the Normal Smooth scale: the
    smoothed normals plus *strength* times what smoothing removed, so 1.0 is
    the original, 2 doubles the detail and 0.5 halves it, as the control
    says.  HELIO 2 added strength × the detail on top of the original, so
    1.0 doubled it (KD-031).  With Normal Smooth at 0 there is no detail
    scale and the normals are left as they are."""
    p = dict(SHARPEN_DEFAULTS, **params)
    n = np.asarray(normals, np.float32).copy()
    d = None if depth is None else np.asarray(depth, np.float32).copy()
    if p["smooth"] > 0 and p["strength"] != 1.0:
        nb = cv2.GaussianBlur(n, (0, 0), p["smooth"])
        n = nb + (n - nb) * p["strength"]
        norm = np.linalg.norm(n, axis=-1, keepdims=True)
        n = np.divide(n, norm, out=np.zeros_like(n), where=norm != 0)
    if d is not None and p["usm_amt"] > 0:
        blur = cv2.GaussianBlur(d, (0, 0), p["usm_rad"])
        d = cv2.addWeighted(d, 1.0 + p["usm_amt"], blur, -p["usm_amt"], 0)
    if d is not None and p["hf_boost"] > 0:
        d = d + (d - cv2.GaussianBlur(d, (0, 0), p["hf_rad"])) * p["hf_boost"]
    if d is not None and p["clahe"] > 0:
        cl = cv2.createCLAHE(clipLimit=p["clahe"], tileGridSize=(8, 8))
        d = cl.apply((np.clip(d, 0, 1) * 65535.0).astype(np.uint16)).astype(np.float32) / 65535.0
    return n, d


DETAIL_SIDE = 4500            # the long side of the normals and colour a mesh is shaded with
DETAIL_SIDE_LARGE = 8000      # the same with 24 GB or more available


def detail_side(available_bytes: int | None = None) -> int:
    """How sharp to shade the 3D relief: DETAIL_SIDE, less on a machine short of memory
    (the normals are float32 on the CPU: 12 bytes a pixel, 15 million pixels at 4,500 px)."""
    if available_bytes is None:
        from helio.utils.memory_plan import system_memory
        available_bytes = system_memory()[1]
    gb = available_bytes / 2 ** 30
    return DETAIL_SIDE_LARGE if gb >= 24 else DETAIL_SIDE if gb >= 12 else 3200 if gb >= 6 else 2400


def detail_textures(D, mesh_w: int, side: int | None = None):
    """(normals float32 unit vectors, colour uint8, (w, h)) of the solve at up to *side* on the
    long side, to shade and colour a mesh that is only *mesh_w* wide; None when the mesh
    already has all the solve has (or the solve has no normals).  Area-averaged from the
    full-size arrays, so every one of the photographs' pixels contributes."""
    import cv2
    full_n = getattr(D, "normals_full", None)
    if full_n is None:
        return None
    side = side or detail_side()
    k = min(1.0, side / float(max(D.w, D.h)))
    tw, th = max(1, int(round(D.w * k))), max(1, int(round(D.h * k)))
    if tw <= mesh_w:
        return None
    n = cv2.resize(np.asarray(full_n, np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-8)
    albedo = getattr(D, "albedo_rgb_full", None)
    a = None if albedo is None else cv2.resize(np.asarray(albedo), (tw, th), interpolation=cv2.INTER_AREA)
    return n, a, (tw, th)
