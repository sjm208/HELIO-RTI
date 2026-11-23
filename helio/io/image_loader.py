"""
Image loading / processing / headless-render helpers.
"""

from __future__ import annotations
from typing import Optional
import logging
import numpy as np

_log = logging.getLogger(__name__)


def load_and_process_single(path):
    """Read one capture frame. Returns ``(rgb_uint8, grey_float32)``.

    Raw files are decoded linearly (see `helio.io.raw_loader`) and the extra
    precision is kept where it is actually used: the *grey* channel, which is
    what the photometric solve fits. It is returned on the same 0-255 scale as
    an 8-bit frame so every downstream threshold keeps its meaning, but as
    float32 carrying fractional values a JPEG could never hold. The RGB is
    display-encoded to 8-bit because it becomes the albedo people look at.

    Other files keep their stored numbers for the grey, at the file's depth (a
    16-bit TIFF's fractions are kept the same way), and their colour is brought
    to sRGB by the file's own ICC profile (helio.io.colour_space).
    """
    import cv2
    cv2.setNumThreads(0)

    try:
        from . import raw_loader
        _is_raw = raw_loader.is_raw(path)
    except Exception:
        _is_raw = False

    if _is_raw:
        try:
            rgb16 = raw_loader.read_raw_linear(path)
        except Exception as exc:
            _log.warning("raw decode failed for %s: %s", path, exc)
            return None, None
        # Rec.601 luma on LINEAR data - the weights describe how the eye sums
        # primaries, and applying them before a transfer curve is what keeps
        # the result proportional to radiance.
        lin = rgb16.astype(np.float32)
        grey = (0.299 * lin[..., 0] + 0.587 * lin[..., 1] + 0.114 * lin[..., 2])
        grey = grey / 257.0                     # 0-65535 -> 0-255, still float
        grey_smooth = cv2.GaussianBlur(grey, (0, 0), sigmaX=1.5)
        return raw_loader.linear_to_srgb8(rgb16), grey_smooth

    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    if img is not None and img.dtype not in (np.uint8, np.uint16):
        img = cv2.imread(path)                  # floating point: as it always was
    if img is None:
        return None, None
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if img.dtype == np.uint16:
        grey /= 257.0                           # 0-65535 -> 0-255, still float
        img = np.round(img / 257.0).astype(np.uint8)
    grey_smooth = cv2.GaussianBlur(grey, (0, 0), sigmaX=1.5)
    from .colour_space import to_srgb
    return to_srgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), path), grey_smooth


def load_plates(paths, h=None, w=None, normalise=True, max_workers=None):
    """Load and optionally normalise a stack of plate images.

    Parameters
    ----------
    paths : list of str
        Image file paths.
    h, w : int or None
        Target height/width. Images are resized to (h, w) if provided
        (using the first image's dimensions if only one is set).
    normalise : bool
        If True, divide each image by its mean brightness.
    max_workers : int or None
        Number of parallel workers (not yet implemented; always serial).

    Returns
    -------
    stack : np.ndarray, shape (n, h, w), float32
    grey_smooth : np.ndarray, shape (n, h, w), float32
    """
    import cv2
    if not paths:
        return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0, 0, 0), dtype=np.float32)

    images = []
    greys = []
    target = None
    for p in paths:
        rgb, grey = load_and_process_single(p)
        if rgb is None:
            continue
        _h, _w = grey.shape
        if target is None:
            target = (_h, _w) if h is None else (h, w if w else _w)
        if grey.shape != target:
            grey = cv2.resize(grey, (target[1], target[0]), interpolation=cv2.INTER_AREA)
            rgb = cv2.resize(rgb, (target[1], target[0]), interpolation=cv2.INTER_AREA)
        images.append(grey.astype(np.float32))
        greys.append(grey.astype(np.float32))

    stack = np.stack(images, axis=0)
    grey_smooth = np.stack(greys, axis=0)

    if normalise:
        mean_per_img = stack.reshape(stack.shape[0], -1).mean(axis=1, keepdims=True)
        mean_per_img = mean_per_img.reshape(-1, 1, 1)
        stack = stack / np.maximum(mean_per_img, 1e-8)

    return stack, grey_smooth


def headless_render_worker(verts, faces, colors, out_path,
                           bg_color='#ffffff', elevation=25, azimuth=-55):
    import numpy as np
    import cv2
    try:
        from vispy import app, scene
        from vispy.scene import visuals
        app.use_app('glfw')
        canvas = scene.SceneCanvas(keys=None, show=False, size=(3600, 2400), bgcolor=bg_color)
        view = canvas.central_widget.add_view()
        v_data = np.ascontiguousarray(verts, dtype=np.float32)
        f_data = np.ascontiguousarray(faces, dtype=np.uint32)
        c_data = np.ascontiguousarray(colors, dtype=np.float32)
        mesh = visuals.Mesh(vertices=v_data, faces=f_data, vertex_colors=c_data, shading=None)
        view.add(mesh)
        _bb_min = v_data.min(axis=0)
        _bb_max = v_data.max(axis=0)
        _diag = float(np.linalg.norm(_bb_max - _bb_min))
        distance = _diag * 1.05
        camera = scene.cameras.TurntableCamera(elevation=elevation, azimuth=azimuth,
                                               distance=distance, fov=58)
        view.camera = camera
        try:
            camera.set_range(margin=0.01)
        except TypeError:
            camera.set_range()
        img = canvas.render()
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
        canvas.close()
    except Exception as e:
        _log.error("Headless render failed: %s", e, exc_info=True)


def export_image(image, path, fmt="png", dpi=300):
    import cv2
    ext = fmt.lower()
    if ext == "jpg":
        ext = "jpeg"
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image)
    return path
