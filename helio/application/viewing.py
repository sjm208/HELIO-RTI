"""What the Inspect screen shows, and the tools around it, GUI-free (VIEW-002).

``screen_image`` is the image path of the classic ``update_main_view`` (the
layer, the engine render, the composite, then the Mirror / Invert display
transforms).  The classic window and the Qt Inspect workspace both call it,
so the two screens agree.  The rest is the state behind the classic Inspect
panel that has no business living in a widget: the best raking light, the
sweep, bookmarks ("Safe States") and the settle-then-commit undo history.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# The classic "visual layer" menu, in its order.
LAYERS = ("Shading", "Combined Colour", "Roughness Fusion", "Depth Map (Greyscale)",
          "Confidence Overlay")
CONF_COLOURS = ("Red", "Green", "Blue", "Yellow", "Cyan", "Magenta")
CONF_BLENDS = ("Overlay", "Multiply", "Additive")


def default_conf() -> dict:
    """The confidence overlay's settings as a fresh classic window has them."""
    return dict(colour="Red", opacity=0.6, blend="Overlay", shadow=20, highlight=235, mask=False)


GLOBAL_ADJUSTMENTS = (("global_dehaze", 0.0), ("global_whites", 1.0), ("global_blacks", 0.0),
                      ("global_midtone", 1.0), ("global_clarity", 0.0), ("global_texture", 0.0))


@dataclass
class Display:
    """Everything besides the view's own settings that changes the screen."""
    layer: str = "Shading"
    colour_opacity: float = 1.0
    # The Colour layer's own brightness (it multiplies a dark photograph's
    # colour), apart from the view's gain, which every layer shares.
    colour_gain: float = 1.0
    conf: dict = field(default_factory=default_conf)
    normal_invert: bool = False          # "Invert Normals"
    normal_flip_x: bool = False
    depth_invert: bool = False           # Mirror / Invert: "Invert Depth (Z)"
    flip_h: bool = False                 # Mirror / Invert: horizontal flip
    flip_v: bool = False
    rot180: bool = False
    adjustments: dict = field(default_factory=lambda: dict(GLOBAL_ADJUSTMENTS))

    def transformed(self) -> bool:
        return self.flip_h or self.flip_v or self.rot180

    def transform_badge(self) -> str:
        """The classic red badge: which display transforms are on, or ''."""
        parts = [n for on, n in ((self.flip_h, "H-Flip"), (self.flip_v, "V-Flip"),
                                 (self.rot180, "180°"), (self.depth_invert, "Z-Inv")) if on]
        return " · ".join(parts)


def conf_for(display: Display) -> dict | None:
    """The confidence settings composite() takes, or None (composite_final_image)."""
    if display.layer == "Confidence Overlay" or display.conf.get("mask"):
        return dict(display.conf)
    return None


def apply_transforms(img: np.ndarray, display: Display) -> np.ndarray:
    """The Mirror / Invert display transforms, in the classic order."""
    if display.flip_h:
        img = np.fliplr(img)
    if display.flip_v:
        img = np.flipud(img)
    if display.rot180:
        img = np.rot90(img, 2)
    return np.ascontiguousarray(img)


def turn_matrix(w: int, h: int, deg: float):
    """(M, new_w, new_h): the 2 x 3 map of a w x h picture turned *deg* clockwise
    onto a canvas that holds all of it (turn_image, turn_point)."""
    import cv2
    a = float(deg or 0.0) % 360.0
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -a, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    if a % 90.0 == 0.0:
        nw, nh = (w, h) if a % 180.0 == 0.0 else (h, w)
    else:
        nw, nh = int(math.ceil(h * s + w * c)), int(math.ceil(h * c + w * s))
    M[0, 2] += nw / 2.0 - w / 2.0
    M[1, 2] += nh / 2.0 - h / 2.0
    return M, nw, nh


def turn_image(img: np.ndarray, deg: float, fill=0) -> np.ndarray:
    """*img* turned *deg* clockwise, as Inspect's turn shows it.  Quarter turns are
    exact; any other angle grows the canvas to hold the whole picture, the corners
    *fill*."""
    import cv2
    a = float(deg or 0.0) % 360.0
    if a == 0.0:
        return img
    if a % 90.0 == 0.0:
        return np.ascontiguousarray(np.rot90(img, -int(a // 90)))
    h, w = img.shape[:2]
    M, nw, nh = turn_matrix(w, h, a)
    src = np.ascontiguousarray(img)
    if src.dtype == np.bool_:
        src = src.astype(np.uint8)
    border = fill if np.ndim(fill) == 0 else tuple(float(v) for v in fill)
    return cv2.warpAffine(src, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def turn_vectors(n: np.ndarray, deg: float) -> np.ndarray:
    """Normals (+Y up) of a picture turned *deg* clockwise: each one's in-plane
    part turned with it, so the surface is lit as before from a turned light."""
    a = math.radians(float(deg or 0.0) % 360.0)
    if a == 0.0:
        return n
    c, s = math.cos(a), math.sin(a)
    out = np.array(n, np.float32, copy=True)
    x, y = n[..., 0], n[..., 1]
    out[..., 0] = x * c + y * s
    out[..., 1] = -x * s + y * c
    return out


def screen_surface(display, turn: float, *, normals=None, colour=None, maps=(), heights=()):
    """The solve's arrays as the screen shows them: Mirror / Invert, then the
    turn.  Normals have their directions mirrored and turned too; *maps* (any
    per-pixel picture) and *heights* (a height per pixel) are moved like the
    colour.  Outside a turned picture the normals face up, the colour is black
    and the heights take their lowest value.  Returns (normals, colour, maps,
    heights), None where None was given."""
    fh = bool(getattr(display, "flip_h", False))
    fv = bool(getattr(display, "flip_v", False))
    if getattr(display, "rot180", False):
        fh, fv = not fh, not fv

    def move(a, fill=0):
        if a is None:
            return None
        if fh:
            a = np.fliplr(a)
        if fv:
            a = np.flipud(a)
        return turn_image(np.ascontiguousarray(a), turn, fill)
    n = None
    if normals is not None:
        n = np.array(normals, np.float32, copy=True)
        if fh:
            n[..., 0] = -n[..., 0]
        if fv:
            n[..., 1] = -n[..., 1]
        n = turn_vectors(n, turn)
        n = move(n, (0.0, 0.0, 1.0))
        if float(turn or 0.0) % 90.0:
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6)
    hs = []
    for a in heights:
        hs.append(None if a is None else move(a, float(np.nanmin(a)) if np.size(a) else 0.0))
    return n, move(colour), [move(m) for m in maps], hs


class ScreenTurn:
    """The screen's Mirror / Invert and turn, for things drawn away from the screen
    (the publication sheet, the plate, figures): pictures, points in a picture's
    pixels, sizes and light directions (+Y up), each moved as Inspect shows them."""

    def __init__(self, display=None, turn: float = 0.0):
        fh = bool(getattr(display, "flip_h", False))
        fv = bool(getattr(display, "flip_v", False))
        if getattr(display, "rot180", False):
            fh, fv = not fh, not fv
        self.fh, self.fv, self.turn = fh, fv, float(turn or 0.0) % 360.0

    @property
    def any(self) -> bool:
        return self.fh or self.fv or bool(self.turn)

    def image(self, img, fill=0):
        if img is None or not self.any:
            return img
        a = np.asarray(img)
        if self.fh:
            a = np.fliplr(a)
        if self.fv:
            a = np.flipud(a)
        return turn_image(np.ascontiguousarray(a), self.turn, fill)

    def size(self, w: int, h: int) -> tuple[int, int]:
        _M, nw, nh = turn_matrix(int(w), int(h), self.turn)
        return nw, nh

    def point(self, x: float, y: float, w: int, h: int) -> tuple[float, float]:
        """(x, y) in a w x h picture, where it lands in the moved picture."""
        if self.fh:
            x = w - x
        if self.fv:
            y = h - y
        if not self.turn:
            return float(x), float(y)
        M, _nw, _nh = turn_matrix(int(w), int(h), self.turn)
        return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]),
                float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))

    def box(self, b, w: int, h: int) -> tuple[int, int, int, int]:
        """A box (x1, y1, x2, y2): the box round its moved corners."""
        x1, y1, x2, y2 = b
        pts = [self.point(x, y, w, h) for x in (x1, x2) for y in (y1, y2)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))

    def vectors(self, v):
        """Directions (lights, normals; +Y up) as the moved picture sees them."""
        if v is None or not self.any:
            return v
        out = np.array(v, np.float64 if np.asarray(v).dtype == np.float64 else np.float32, copy=True)
        if self.fh:
            out[..., 0] = -out[..., 0]
        if self.fv:
            out[..., 1] = -out[..., 1]
        return turn_vectors(out, self.turn).astype(out.dtype)

    def unvectors(self, v):
        """The inverse of vectors: a direction on the moved picture, back as stored."""
        if v is None or not self.any:
            return v
        out = turn_vectors(np.array(v, np.float64, copy=True), -self.turn)
        if self.fh:
            out[..., 0] = -out[..., 0]
        if self.fv:
            out[..., 1] = -out[..., 1]
        return out

    def light_back(self, overrides: dict | None) -> dict | None:
        """A view's light settings chosen on the moved picture (lx, ly, lz and the
        compass angles), back to the stored picture's frame, to draw with."""
        if not overrides or "lx" not in overrides or not self.any:
            return overrides
        from helio.domain.view_controls import light_settings
        lx, ly, lz = self.unvectors(np.array([[overrides["lx"], overrides["ly"],
                                               overrides.get("lz", 0.0)]]))[0]
        return dict(overrides, **light_settings(float(lx), float(ly), float(lz)))

    def bearing(self, az: float) -> float:
        """A compass bearing (0 north, clockwise) on the moved picture."""
        a = math.radians(az)
        v = self.vectors(np.array([[math.sin(a), math.cos(a), 0.0]]))[0]
        return math.degrees(math.atan2(v[0], v[1])) % 360.0


class TurnedData:
    """*data* (a DataManager) as the screen shows it, for the reports that read its
    pictures directly: every array the size of the photographs or of the preview
    comes back moved by *t* (ScreenTurn), normals with their directions moved
    too; w, h, pw, ph are the moved sizes.  Arrays are moved once and kept."""

    def __init__(self, data, t: ScreenTurn):
        object.__setattr__(self, "_d", data)
        object.__setattr__(self, "_t", t)
        object.__setattr__(self, "_kept", {})

    def __getattr__(self, name):
        d, t = object.__getattribute__(self, "_d"), object.__getattribute__(self, "_t")
        if name in ("w", "h"):
            nw, nh = t.size(d.w, d.h)
            return nw if name == "w" else nh
        if name in ("pw", "ph"):
            nw, nh = t.size(d.pw, d.ph)
            return nw if name == "pw" else nh
        v = getattr(d, name)
        if not isinstance(v, np.ndarray) or v.ndim not in (2, 3) or v.shape[:2] not in (
                (int(d.h), int(d.w)), (int(d.ph), int(d.pw))):
            return v
        kept = object.__getattribute__(self, "_kept")
        hit = kept.get(name)
        if hit is not None and hit[0] is v:
            return hit[1]
        if "normal" in name and v.ndim == 3 and v.shape[2] == 3:
            out = t.image(t.vectors(v), (0.0, 0.0, 1.0))
        elif v.ndim == 2 and v.dtype.kind == "f":
            out = t.image(v, float(np.nanmin(v)) if v.size else 0.0)
        else:
            out = t.image(v)
        kept[name] = (v, out)
        return out

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_d"), name, value)


def image_point(x: float, y: float, w: int, h: int, display: Display) -> tuple[float, float]:
    """A point on the transformed screen image, back in the stored image's pixels."""
    if display.rot180:
        x, y = w - x, h - y
    if display.flip_v:
        y = h - y
    if display.flip_h:
        x = w - x
    return x, y


def depth_layer(data, use_proxy: bool = True, crop=None) -> np.ndarray:
    """The "Depth Map (Greyscale)" layer: the Poisson height, 0-255."""
    if use_proxy:
        arr = data.depth_proxy_p
    else:
        arr = data.depth_vis_p
        if crop:
            x1, y1, x2, y2 = crop
            arr = arr[y1:y2, x1:x2]
    return (np.asarray(arr) * 255).astype(np.uint8)


class ScaledData:
    """The preview-size arrays of *data*, *scale* times the size, for the
    frames drawn while the light is being dragged.

    Every attribute that is a preview-size array (``normals_proxy``,
    ``albedo_proxy``, ``ptm_proxy`` laid out as rows of pixels, ...) comes back
    resized once and kept; anything else is read from *data* as it is, so the
    engines need not know.  Normals are resized and made unit again.  The
    frame a drag ends on is always drawn from *data* itself, at full size."""

    def __init__(self, data, scale: float):
        object.__setattr__(self, "_d", data)
        pn = getattr(data, "normals_proxy", None)
        ph, pw = (pn.shape[:2] if pn is not None and getattr(pn, "ndim", 0) == 3
                  else (int(data.ph), int(data.pw)))
        nh, nw = max(8, int(round(ph * scale))), max(8, int(round(pw * scale)))
        object.__setattr__(self, "_src", (ph, pw))
        object.__setattr__(self, "_dst", (nh, nw))
        object.__setattr__(self, "_kept", {})
        object.__setattr__(self, "scale", nw / max(pw, 1))

    def __getattr__(self, name):
        import cv2
        d = object.__getattribute__(self, "_d")
        v = getattr(d, name)
        ph, pw = object.__getattribute__(self, "_src")
        nh, nw = object.__getattribute__(self, "_dst")
        if name == "ph":
            return nh
        if name == "pw":
            return nw
        if not isinstance(v, np.ndarray) or v.ndim < 2:
            return v
        kept = object.__getattribute__(self, "_kept")
        hit = kept.get(name)
        if hit is not None and hit[0] is v:
            return hit[1]
        if v.shape[:2] == (ph, pw) and v.ndim in (2, 3):
            src = v if v.dtype != np.bool_ else v.astype(np.uint8)
            out = cv2.resize(np.ascontiguousarray(src), (nw, nh), interpolation=cv2.INTER_AREA)
            if v.dtype == np.bool_:
                out = out.astype(bool)
            if v.ndim == 3 and v.shape[2] == 3 and v.dtype.kind == "f" and "normal" in name:
                n = np.linalg.norm(out, axis=2, keepdims=True)
                out = np.divide(out, n, out=np.zeros_like(out), where=n > 1e-9)
        elif v.ndim == 2 and v.shape[1] == ph * pw:
            k = v.shape[0]
            planes = v.reshape(k, ph, pw)
            out = np.stack([cv2.resize(np.ascontiguousarray(p), (nw, nh), interpolation=cv2.INTER_AREA)
                            for p in planes]).reshape(k, nh * nw)
        else:
            return v
        out = np.ascontiguousarray(out)
        kept[name] = (v, out)
        return out

    def __setattr__(self, name, value):
        object.__getattribute__(self, "__dict__")[name] = value


def screen_image(view: str, data, settings: dict, display: Display, *, use_proxy: bool = True,
                 crop=None, res_res: float = 1.0, pretrimmed_crop_frac=None,
                 transforms: bool = True) -> np.ndarray:
    """What the screen shows for *view*: uint8 RGB.

    At preview size (``use_proxy``) this is ``update_main_view``'s image; with
    ``use_proxy=False`` and *crop* (x1, y1, x2, y2) it is the magnifier's
    full-resolution cut.  Raises what the engine raises: the caller reports it.
    """
    from helio.application.render import composite, render
    if display.layer == "Depth Map (Greyscale)":
        g = depth_layer(data, use_proxy, crop)
        out = np.repeat(g[..., None], 3, axis=2)
    else:
        if use_proxy:
            pn = getattr(data, "normals_proxy", None)
            h, w = (pn.shape[:2] if pn is not None and pn.ndim == 3 else (data.ph, data.pw))
        else:
            x1, y1, x2, y2 = crop if crop else (0, 0, data.w, data.h)
            h, w = y2 - y1, x2 - x1
        shading = render(view, data, settings, h, w, use_proxy, crop,
                         normal_invert=display.normal_invert,
                         normal_flip_x=display.normal_flip_x,
                         depth_invert=display.depth_invert, res_res=res_res,
                         pretrimmed_crop_frac=pretrimmed_crop_frac)
        if shading is None:
            shading = np.zeros((h, w), np.float32)
        sh, sw = shading.shape[:2]
        out = composite(view, shading, data, settings, sh, sw, use_proxy, crop,
                        view_mode=display.layer,
                        colour_opacity=display.colour_opacity
                        if display.layer == "Combined Colour" else 1.0,
                        colour_gain=display.colour_gain,
                        conf=conf_for(display), adjustments=display.adjustments)
    out = np.asarray(out)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    out = out[..., :3]
    return apply_transforms(out, display) if transforms else np.ascontiguousarray(out)


# ── The part on screen at full resolution (Inspect draws it over the preview) ──────

# Measured (tests/test_live_detail.py) by drawing a region from the full-size arrays and from the
# preview and comparing them: most views are the same picture at both sizes; some stretch
# their output to the range of whatever they are given, so a region comes out lighter or
# darker than the preview of it; and some size their filters in pixels, so a region is a
# different picture.  The second group is matched to the preview's tone; the third is not
# drawn this way (a jump in look when the sharp part appears would be worse than a soft one).
DETAIL_TONE_MATCH = frozenset({
    "Image Quotient (IQ)", "Negative Openness", "Positive Openness", "Slope Gradient",
    "Diffuse Gain", "Multi-Directional Hillshading (Blend / PCA)", "Raking Composite",
    "Specular View", "Ridge and Groove Detector",
})
DETAIL_SKIP = frozenset({
    "Ambient Occlusion (AO)", "Retinex Enhancement", "Cast-Shadow Raking",
    "Surface Curvature (Scratches)", "Normal Residual", "Curvedness", "Shape Index",
    "Exaggerated Shading",
    # its 2-98 % stretch is taken over whatever region it is drawn on
    "Multi-Scale Relief Model (MSRM)",
    # a decomposition worked out again on whatever region it is given
    "Colour ICA (IC1-3)", "RGB-PS R-B Diff",
})


def tone_match(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """*img* (uint8 RGB) with a gain and offset per channel that best put it onto
    *reference* (the same region, smaller): the tone of the preview, the sharpness of *img*."""
    import cv2
    small = cv2.resize(img, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)
    step = max(1, small.shape[0] * small.shape[1] // 200_000)
    out = np.empty_like(img)
    for c in range(3):
        x = small[::step, ::step, c].astype(np.float64).ravel()
        y = reference[::step, ::step, c].astype(np.float64).ravel()
        var = x.var()
        if var < 1.0:                                         # flat: only the level can be matched
            lut = np.clip(np.arange(256) + (y.mean() - x.mean()), 0, 255)
        else:
            a = ((x - x.mean()) * (y - y.mean())).mean() / var
            lut = np.clip(a * (np.arange(256) - x.mean()) + y.mean(), 0, 255)
        out[..., c] = np.round(lut).astype(np.uint8)[img[..., c]]
    return out


def detail_image(view: str, data, settings: dict, display: Display, box) -> np.ndarray:
    """*view* of the region *box* (x1, y1, x2, y2, stored pixels) from the full-size arrays,
    uint8 RGB, display transforms applied, in the tone of the preview where the view needs it."""
    out = screen_image(view, data, settings, display, use_proxy=False, crop=box, transforms=False)
    if view in DETAIL_TONE_MATCH:
        pv = screen_image(view, data, settings, display, transforms=False)
        sy, sx = pv.shape[0] / float(data.h), pv.shape[1] / float(data.w)
        x1, y1, x2, y2 = box
        ref = pv[int(y1 * sy):max(int(y1 * sy) + 1, int(y2 * sy)), int(x1 * sx):max(int(x1 * sx) + 1, int(x2 * sx))]
        if ref.size:
            out = tone_match(out, np.ascontiguousarray(ref))
    return apply_transforms(out, display)


# ── The light tools of the classic dome panel ───────────────────────────

def best_raking_light(L: np.ndarray, normals: np.ndarray):
    """The captured light that gives the most surface contrast ("Best Raking").

    Scans every light of the light matrix (+Y up, as the normals) for the
    one whose N·L shading has the largest standard deviation.  Returns
    (index, (lx, ly, lz) in screen frame, azimuth, elevation).

    The classic button set the screen light to the matrix row as it was,
    so the y component went in with the wrong sign and the view lit from
    the mirror of the light it reported (KD-019); here it is converted.
    """
    flat_n = np.asarray(normals, np.float32).reshape(-1, 3)
    best_var, best_idx = -1.0, 0
    for i in range(L.shape[0]):
        lv = L[i] / (np.linalg.norm(L[i]) + 1e-9)
        v = float(np.std(np.maximum(0, flat_n @ lv)))
        if v > best_var:
            best_var, best_idx = v, i
    lv = L[best_idx] / (np.linalg.norm(L[best_idx]) + 1e-9)
    screen = (float(lv[0]), float(-lv[1]), float(lv[2]))
    from helio.domain.view_controls import angles
    az, el = angles(*screen)
    return best_idx, screen, az, el


SWEEP_MAX_EL = 85.0      # a light this near overhead (as a capture opens, at 90°) hardly moves…
SWEEP_RAKE_EL = 30.0     # …so the sweep comes down to a raking height instead.  Until
                         # 2026-11-20 this was 60°: a light placed at 70° on purpose was
                         # thrown out to the 30° ring


@dataclass
class Sweep:
    """The Auto-Sweep: the light circling at its current elevation (``sweep_step``),
    or at ``SWEEP_RAKE_EL`` when that is higher than ``SWEEP_MAX_EL``: the light
    starts overhead (90°), where a sweep shows nothing changing at all.  With
    *lower* False (a light the user has just placed) it keeps its height."""
    theta: float = 0.0
    vz: float = math.sqrt(0.5)
    r: float = math.sqrt(0.5)
    lowered: bool = False

    @classmethod
    def starting_at(cls, light, lower: bool = True) -> "Sweep":
        vx, vy, vz = (float(v) for v in light)
        n = math.sqrt(vx * vx + vy * vy + vz * vz) or 1.0
        vz = float(np.clip(vz / n, 0.05, 0.999))
        lowered = lower and math.degrees(math.asin(vz)) > SWEEP_MAX_EL
        if lowered:
            vz = math.sin(math.radians(SWEEP_RAKE_EL))
        theta = float(np.arctan2(vy, vx)) if abs(vx) + abs(vy) > 1e-6 else 0.0
        return cls(theta=theta, vz=vz, r=float(np.sqrt(max(0.001, 1.0 - vz ** 2))),
                   lowered=lowered)

    def step(self, speed: float = 1.0) -> tuple[float, float, float]:
        self.theta += 0.05 * speed
        vx, vy, vz = np.cos(self.theta) * self.r, np.sin(self.theta) * self.r, self.vz
        n = math.sqrt(vx * vx + vy * vy + vz * vz)
        return float(vx / n), float(vy / n), float(vz / n)


# ── Safe States (bookmarks) ──────────────────────────────────────────────

def bookmark(engine: str, layer: str, light, settings: dict, shared: dict,
             mag: float = 0.5, mag_size: int = 350) -> dict:
    """A bookmark as ``save_bookmark`` writes it (and sessions carry it)."""
    return {"engine": engine, "layer": layer, "lx": float(light[0]), "ly": float(light[1]),
            "lz": float(light[2]), "mag": float(mag), "mag_size": int(mag_size),
            "settings": dict(settings), "shared": dict(shared)}


def bookmark_light(b: dict) -> tuple[float, float, float]:
    """The light a bookmark restores (``load_bookmark``)."""
    return float(b["lx"]), float(b["ly"]), float(b["lz"])


# ── Undo ─────────────────────────────────────────────────────────────────

class History:
    """The classic undo: watch a snapshot, commit it once it has settled.

    ``observe`` is called on a timer (the classic polls every 400 ms).  A
    value still changing between two polls is a drag in progress, so a
    drag of a slider or the dome becomes one undo step, not thirty.
    """
    LIMIT = 50

    def __init__(self, state=None):
        self.stack: list = []
        self.redo_stack: list = []
        self.last = self.pending = state

    def observe(self, state) -> bool:
        """Returns True when a settled change was committed."""
        if state is None:
            return False
        if state != self.pending:
            self.pending = state
            return False
        if state != self.last:
            if self.last is not None:
                self.stack.append(self.last)
                del self.stack[:-self.LIMIT]
            self.redo_stack.clear()
            self.last = state
            return True
        return False

    def undo(self, current):
        """The state to go back to, or None."""
        if not self.stack:
            return None
        prev = self.stack.pop()
        if current is not None:
            self.redo_stack.append(current)
        return prev

    def redo(self, current):
        if not self.redo_stack:
            return None
        nxt = self.redo_stack.pop()
        if current is not None:
            self.stack.append(current)
        return nxt

    def settled(self, state):
        """After applying an undo or redo: that is now the baseline."""
        self.pending = self.last = state


def undo_state(engine: str, normal_engine: str, depth_engine: str, layer: str,
               settings: dict) -> dict:
    """The classic ``_undo_state`` snapshot: the view, solver choices and the view's scalars."""
    return {"render_engine": engine, "normal_engine": normal_engine,
            "depth_engine": depth_engine, "view_mode": layer,
            "settings": {k: v for k, v in (settings or {}).items()
                         if isinstance(v, (int, float, str, bool))}}
