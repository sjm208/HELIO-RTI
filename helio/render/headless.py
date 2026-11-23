"""
Headless / off-screen rendering for high-res captures and publication sheets.
"""

import cv2
import numpy as np

from helio.io.image_loader import headless_render_worker


def headless_render_main(
    engine=None, width=3600, height=2400,
    normals=None, depth=None, settings=None, output_path="",
):
    """Off-screen render worker (subprocess entry point).

    Delegates to ``headless_render_worker`` for the actual VisPy off-screen
    rendering. This wrapper exists so the headless path can be extended with
    engine-based shading before dispatch.
    """
    if normals is None and depth is None:
        raise ValueError("headless_render_main: either normals or depth must be provided")

    h, w = normals.shape[:2] if normals is not None else depth.shape[:2]
    from helio.core.shading_engines import render_shading
    from helio.io.image_loader import headless_render_worker
    from helio.data_manager import DataManager

    dm = DataManager()
    dm.h, dm.w = h, w
    dm.normals_full = normals
    dm.depth_vis_p = depth if depth is not None else np.zeros((h, w), np.float32)

    if engine and engine != "Standard PBR":
        shaded = render_shading(engine, dm, settings or {}, h, w)
    else:
        shaded = normals if normals is not None else depth

    # Build a simple mesh from the depth/normal data
    from helio.render.viewer import build_viewer_mesh
    # build_viewer_mesh returns (vertices, faces) - two values, not three.
    # Unpacking three here meant headless rendering raised ValueError on every
    # run, so this path had never worked end to end.
    verts, faces = build_viewer_mesh(normals, depth)
    colors = None
    if shaded is not None:
        # Colour the vertices from the render that was just computed, so the
        # headless output matches what the engine draws rather than plain grey.
        try:
            c = np.asarray(shaded, np.float32)
            if c.ndim == 3:
                c = c.mean(axis=-1)
            if float(np.nanmax(c)) > 1.0 + 1e-6:
                c = c / 255.0
            ch, cw = c.shape[:2]
            n_v = verts.shape[0]
            side = int(np.sqrt(n_v)) or 1
            samp = cv2.resize(np.nan_to_num(c), (side, side)) if (ch, cw) != (side, side) else c
            flat = np.clip(samp.reshape(-1), 0.0, 1.0)
            if flat.size >= n_v:
                colors = np.repeat(flat[:n_v, None], 3, axis=1).astype(np.float32)
        except Exception:
            colors = None
    if colors is None:
        colors = np.ones((verts.shape[0], 3), dtype=np.float32) * 0.8

    headless_render_worker(
        verts, faces, colors, output_path,
        bg_color=settings.get("bg_color", "#ffffff") if settings else "#ffffff",
    )


def compute_publication_sheet_geometry(
    num_images, page_w=3508, page_h=2480, margin=80,
):
    """Compute (x, y, w, h) grid positions for *num_images* on a sheet."""
    cols = int(num_images ** 0.5)
    rows = (num_images + cols - 1) // cols
    cell_w = (page_w - 2 * margin) // cols
    cell_h = (page_h - 2 * margin) // rows
    positions = []
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= num_images:
                break
            x = margin + c * cell_w
            y = margin + r * cell_h
            positions.append((x, y, cell_w, cell_h))
    return positions
