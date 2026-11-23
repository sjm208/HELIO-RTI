"""
Interactive 3D viewer (VisPy + GLFW), launched as a subprocess.

This is a direct extraction of the original ``run_opengl_viewer()``
module-level function from the monolith.  No UI imports are allowed
here — the viewer communicates via shared memory only.
"""

import os
import sys
import time
import logging
import numpy as np

_log = logging.getLogger(__name__)


def _auto_crop(img: np.ndarray, pad: int = 8) -> np.ndarray:
    """Return a tightly-cropped copy of *img* (H×W×4 RGBA uint8) with uniform borders removed.

    Uses the alpha channel when it is informative (i.e. some transparent
    pixels exist), otherwise falls back to a luminance threshold against
    the background colour.  Threshold is intentionally aggressive
    (0.5 % of dynamic range above the background floor) so the crop is
    tight — only a thin ``pad``-pixel rim of background is retained.
    """
    # Prefer alpha channel when the image contains transparency — this is
    # the case when screenshots are rendered with a transparent canvas bg.
    if img.shape[-1] == 4:
        alpha = img[..., 3]
        if alpha.min() < 32:  # genuinely transparent pixels exist
            mask = alpha > 24
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if rows.any() and cols.any():
                y1, y2 = np.where(rows)[0][[0, -1]]
                x1, x2 = np.where(cols)[0][[0, -1]]
                y1 = max(0, y1 - pad); y2 = min(img.shape[0], y2 + 1 + pad)
                x1 = max(0, x1 - pad); x2 = min(img.shape[1], x2 + 1 + pad)
                if (y2 - y1) < img.shape[0] or (x2 - x1) < img.shape[1]:
                    return img[y1:y2, x1:x2].copy()
            return img

    # Luminance-based fallback (non-transparent background)
    rgb = img[..., :3].astype(np.float32)
    lo, hi = float(rgb.min()), float(rgb.max())
    if hi - lo < 5:
        return img
    # Aggressive threshold — 0.5 % of dynamic range above background floor.
    # Background #000000 (or #101010) gives a tight mask; only content
    # brighter than 1-2 grey levels above bg is kept.
    bg_thr = lo + (hi - lo) * 0.005
    mask = np.any(rgb > bg_thr, axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    y1 = max(0, y1 - pad)
    y2 = min(img.shape[0], y2 + 1 + pad)
    x1 = max(0, x1 - pad)
    x2 = min(img.shape[1], x2 + 1 + pad)
    if (y2 - y1) < img.shape[0] or (x2 - x1) < img.shape[1]:
        return img[y1:y2, x1:x2].copy()
    return img


def close_viewer_safely(canvas=None, timer=None) -> None:
    """Safely tear down a VisPy canvas + GLFW window + free GPU resources.

    This function:
      1. Stops the VisPy timer.
      2. Closes/destroys the canvas and its native window.
      3. Calls ``gc.collect`` + ``aggressive_cleanup`` to release GPU‑side
         buffers (VisPy internally allocates ``GLFWwindow`` + Metal textures).
      4. Destroys any remaining GLFW windows.

    Call this when the user closes the 3D viewer or switches away from
    the 3D tab.

    Parameters
    ----------
    canvas : vispy.scene.SceneCanvas or None
        The canvas to tear down.  If *None*, tries to close all GLFW windows.
    timer : vispy.app.Timer or None
        The per‑frame update timer.  Stops it before closing the canvas.
    """
    try:
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                _log.debug("Failed to stop timer")
            timer = None
    except Exception:
        _log.debug("Failed to clear timer reference")

    try:
        if canvas is not None:
            try:
                canvas.close()
            except Exception:
                _log.debug("Failed to close canvas")
            try:
                canvas.native.close()
            except Exception:
                _log.debug("Failed to close native window")
            canvas = None
    except Exception:
        _log.debug("Failed to clear canvas reference")

    # Destroy any remaining GLFW windows (in case the subprocess lingers)
    try:
        import glfw as _glfw
        if _glfw.init():
            _wins = _glfw.get_current_context()
            if _wins:
                _glfw.destroy_window(_wins)
            _glfw.terminate()
    except Exception:
        _log.debug("Failed to destroy GLFW windows")

    # Aggressive Python-level cleanup
    try:
        from helio.utils import aggressive_cleanup
        aggressive_cleanup()
    except Exception:
        _log.debug("Failed to run aggressive cleanup")

    _log.debug("close_viewer_safely: canvas and GPU resources released")


def run_opengl_viewer(img_rgb, raw_depth_full, raw_depth_hp, normal_map, shared,
                      depth_spec=None):
    """Open the 3D viewer.

    *depth_spec*, when given, names shared-memory buffers holding the depth
    arrays.  Mapping them lets the parent replace the depth map in place and
    bump ``shared['depth_gen']``, which the per-frame poll below already
    watches - so Recalculate Depth updates the mesh without the viewer being
    terminated and relaunched underneath the user's camera.  Falling back to
    the arrays passed by value is always safe; it simply means a relaunch.
    """
    import cv2, traceback as _tb
    _shm_handles = []
    if depth_spec:
        try:
            from helio.render.live_depth import attach
            _mapped, _shm_handles = attach(depth_spec)
            # Keep our own copies for the first draw, then read the live views.
            raw_depth_full = _mapped["d_full"]
            raw_depth_hp = _mapped["d_hp"]
        except Exception:
            _log.debug("could not map the live depth buffers; "
                       "falling back to the arrays passed by value", exc_info=True)
            _shm_handles = []
    try:
        from vispy import app, scene
        from vispy.scene import visuals
        app.use_app('glfw')
    except Exception as _e:
        _log.error("VisPy/GLFW init failed: %s", _e, exc_info=True)
        return
    try:
        canvas = scene.SceneCanvas(keys='interactive', show=True, size=(1280, 800),
                                   title="Helio | 3D Mesh Viewer", bgcolor='#101010')
        try:
            import glfw as _glfw
            # VisPy's GLFW backend wraps the raw GLFWwindow in a CanvasBackend,
            # which keeps it in ._id (vispy 0.16; .window in some older builds).
            # Handing over the wrapper itself failed every time, so the window
            # never floated and dropped behind HELIO at the first click there.
            _nat = canvas.native
            _glfw_win = getattr(_nat, '_id', None) or getattr(_nat, 'window', None) or _nat
            _glfw.set_window_attrib(_glfw_win, _glfw.FLOATING, True)
            try:
                _glfw.set_window_attrib(_glfw_win, _glfw.FOCUS_ON_SHOW, False)
            except Exception:
                _log.warning("Failed to set GLFW FOCUS_ON_SHOW", exc_info=True)
            _glfw.set_window_pos(_glfw_win, 100, 50)
        except Exception:
            _log.warning("Failed to set GLFW window attributes", exc_info=True)
        try:
            from AppKit import NSApp, NSFloatingWindowLevel
            import time as _t
            _t.sleep(0.15)
            for _w in NSApp.windows():
                try:
                    if "3D Mesh" in str(_w.title()) or "HELIO" in str(_w.title()):
                        _w.setLevel_(NSFloatingWindowLevel)
                except Exception:
                    _log.debug("Failed to set AppKit window level")
        except Exception:
            _log.debug("Failed to import/bind AppKit")
        view = canvas.central_widget.add_view()
        camera = scene.cameras.TurntableCamera(elevation=30, azimuth=0)
        view.camera = camera
        mesh = visuals.Mesh(shading=None)
        view.add(mesh)
        hud = None
        hud_counter = None
        try:
            from vispy.scene import widgets
            w_c, h_c = canvas.size
            hud = visuals.Text(
                '[S] Screenshot  |  [R] Reset  |  [C] Clay  |  Shift+Drag: Pan  |  Arrow Keys: Relight  |  Scroll: Zoom',
                color=(0.85, 0.85, 0.85, 0.9), font_size=8, anchor_x='center', anchor_y='bottom',
                pos=(w_c // 2, h_c - 24), parent=canvas.scene)
            init_count = int(shared.get('screenshot_count', 0).value if hasattr(shared.get('screenshot_count'), 'value') else 0)
            hud_counter = visuals.Text(
                f'📸 {init_count}/20 captures', color=(1.0, 0.85, 0.2, 0.95), font_size=12, bold=True,
                anchor_x='left', anchor_y='top',
                pos=(24, 24), parent=canvas.scene)
        except Exception:
            _log.warning("Failed to create HUD text elements", exc_info=True)
    except Exception as _e:
        _log.error("Canvas creation failed: %s", _e, exc_info=True)
        _tb.print_exc()
        return

    img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
    raw_depth_full = np.ascontiguousarray(raw_depth_full, dtype=np.float32)
    raw_depth_hp = np.ascontiguousarray(raw_depth_hp, dtype=np.float32)
    normal_map = np.ascontiguousarray(normal_map.astype(np.float32))
    if normal_map.max() > 1.5:
        normal_map = (normal_map / 127.5) - 1.0

    state = {
        'step': -1, 'height': -1, 'warp_rem': -1, 'mesh_smooth': -1,
        'fx': 1, 'fy': 1, 'fz': 1,
        'lx': -99, 'ly': -99, 'lz': -99, 'clay_blend': -1,
        'ambient': -1, 'diffuse': -1, 'specular': -1, 'shininess': -1,
        'edge_trim': -1, 'hp_sigma': -1, 'micro': -1, 'shape_blend': -1,
        'points': None, 'faces': None, 'camera_set': False,
        'auto_spin': -1, 'bg_light': -1, 'bake_3d': -1, 'center_origin': -1,
        'clean_export': -1,
        'viewer_gamma': -99., 'viewer_saturation': -99., 'shadow_lift': -99.,
        'highlight_clip': -99.,
        'z_slice_lo': -99., 'z_slice_hi': -99.,
        'use_custom_clay': -1, 'clay_r': -1, 'clay_g': -1, 'clay_b': -1,
    }
    h_orig, w_orig = img_rgb.shape[:2]

    pan_state = {'active': False, 'last_x': 0, 'last_y': 0, 'shift_held': False}

    @canvas.events.resize.connect
    def on_resize(event):
        try:
            w_c, h_c = canvas.size
            if hud:
                hud.pos = (w_c // 2, h_c - 32)
            if hud_counter:
                hud_counter.pos = (24, 24)
        except Exception:
            _log.warning("Failed to update HUD positions on resize", exc_info=True)

    try:
        w_c, h_c = canvas.size
        if hud:
            hud.pos = (w_c // 2, h_c - 32)
        if hud_counter:
            hud_counter.pos = (24, 24)
    except Exception:
        _log.warning("Failed to set initial HUD positions", exc_info=True)

    @canvas.events.key_press.connect
    def on_key_press(event):
        try:
            if event.key.name == 'S':
                _hud_was_visible = hud.visible if hud else False
                _counter_was_visible = hud_counter.visible if hud_counter else False
                if hud:
                    hud.visible = False
                if hud_counter:
                    hud_counter.visible = False
                canvas.update()
                # NOTE: SceneCanvas has no .draw() method — the previous
                # version called canvas.draw() which raised AttributeError
                # and silently aborted every keypress of 'S'. canvas.update()
                # above already marks the canvas dirty so canvas.render()
                # below captures the current frame with HUD hidden.
                # ----------------------------------------------------------------
                # Use the same proven black-white background rendering technique
                # as the Clean Export button — render the scene twice (black bg
                # then white bg), compute the alpha mask from the difference,
                # and de-embed the background colour to get a clean RGBA frame
                # with a valid alpha channel. This is more reliable than trying
                # to force canvas.bgcolor to transparent (which depends on GPU
                # framebuffer alpha support and was producing blank screenshots
                # on some macOS configurations).
                # ----------------------------------------------------------------
                _orig_bg = canvas.bgcolor
                try:
                    canvas.bgcolor = (0.0, 0.0, 0.0, 1.0)
                    canvas.update()
                    _frame_black = canvas.render().astype(np.float32)
                    canvas.bgcolor = (1.0, 1.0, 1.0, 1.0)
                    canvas.update()
                    _frame_white = canvas.render().astype(np.float32)
                    canvas.bgcolor = _orig_bg
                    canvas.update()
                    if hud:
                        hud.visible = _hud_was_visible
                    if hud_counter:
                        hud_counter.visible = _counter_was_visible
                    canvas.update()
                    fb = _frame_black[..., :3]
                    fw = _frame_white[..., :3]
                    diff = np.clip(fw - fb, 0, 255)
                    alpha_f = 1.0 - (diff / 255.0)
                    alpha_ch = np.max(alpha_f, axis=2)
                    a_safe = np.where(alpha_ch > 0.01, alpha_ch, 1.0)
                    rgb_out = np.clip(fb / a_safe[..., np.newaxis], 0, 255).astype(np.uint8)
                    alpha_out = (np.clip(alpha_ch, 0, 1) * 255).astype(np.uint8)
                    clean_rgba = np.concatenate([rgb_out, alpha_out[..., np.newaxis]], axis=-1)
                except Exception:
                    # Fall back to a single-frame render if the dual-render fails
                    _log.warning("Dual-render screenshot failed, falling back to single render", exc_info=True)
                    try: canvas.bgcolor = _orig_bg
                    except Exception: pass
                    if hud:
                        hud.visible = _hud_was_visible
                    if hud_counter:
                        hud_counter.visible = _counter_was_visible
                    canvas.update()
                    clean_rgba = canvas.render()
                # Tight auto-crop (alpha-aware): trims the background
                # to just a thin rim around the mesh silhouette.
                clean_frame = _auto_crop(clean_rgba, pad=8)
                # Composite over a dark BG (#101010) for a consistent visual
                # result — matches the 3D viewer's on-screen appearance.
                if clean_frame.shape[-1] == 4 and clean_frame[..., 3].min() < 32:
                    _bg = np.array([16, 16, 16, 255], dtype=np.uint8)
                    rgb = clean_frame[..., :3]
                    a = (clean_frame[..., 3:4].astype(np.float32) / 255.0)
                    composited = (rgb.astype(np.float32) * a +
                                  _bg[:3].astype(np.float32) * (1.0 - a)
                                  ).clip(0, 255).astype(np.uint8)
                    clean_frame = np.concatenate([composited, np.full_like(composited[..., :1], 255)], axis=-1)
                fname = f"HELIO_3D_Screenshot_{int(time.time())}.png"
                cv2.imwrite(fname, cv2.cvtColor(clean_frame, cv2.COLOR_RGBA2BGRA))
                print(f"[Helio] 3D screenshot saved: {fname}")
                import tempfile
                pub_dir = os.path.join(tempfile.gettempdir(), 'helio_3d_pub_shots')
                os.makedirs(pub_dir, exist_ok=True)
                existing = sorted([f for f in os.listdir(pub_dir)
                                   if f.startswith('pubshot_') and f.endswith('.png')])
                idx = len(existing)
                if idx < 20:
                    pub_path = os.path.join(pub_dir, f'pubshot_{idx:02d}.png')
                    cv2.imwrite(pub_path, cv2.cvtColor(clean_frame, cv2.COLOR_RGBA2BGRA))
                    shared['screenshot_count'].value = idx + 1
                    count_display = idx + 1
                    print(f"[Helio] Publication capture {count_display}/20 saved")
                    if hud_counter:
                        hud_counter.text = f'📸 {count_display}/20 captures'
                        canvas.update()
                else:
                    print("[Helio] Publication capture slots full (20/20). Screenshot saved to disk only.")
                    if hud_counter:
                        hud_counter.text = '📸 20/20 full — screenshot saved'
                        canvas.update()
            elif event.key.name == 'C':
                current = shared['clay_blend'].value
                shared['clay_blend'].value = 0.0 if current > 0.5 else 1.0
                if hud:
                    mode_name = "Clay" if shared['clay_blend'].value > 0.5 else "Colour"
                    hud.text = f'[C] → {mode_name} mode'
                    canvas.update()
            elif event.key.name == 'R':
                camera.center = (0, 0, 0)
                camera.set_range(margin=0.30)
                if hud:
                    hud.text = '[R] View reset'
                    canvas.update()
            elif event.key.name == 'Shift':
                pan_state['shift_held'] = True
            # Arrow keys for relighting
            elif event.key.name == 'Up':
                lx_new = max(-1.0, min(1.0, shared['lx'].value))
                ly_new = max(-1.0, min(1.0, shared['ly'].value + 0.05))
                lz_new = max(0.1, np.sqrt(max(0, 1.0 - lx_new**2 - ly_new**2)))
                shared['lx'].value = lx_new
                shared['ly'].value = ly_new
                shared['lz'].value = lz_new
                if hud:
                    hud.text = '↑ Relight'
                    canvas.update()
            elif event.key.name == 'Down':
                lx_new = max(-1.0, min(1.0, shared['lx'].value))
                ly_new = max(-1.0, min(1.0, shared['ly'].value - 0.05))
                lz_new = max(0.1, np.sqrt(max(0, 1.0 - lx_new**2 - ly_new**2)))
                shared['lx'].value = lx_new
                shared['ly'].value = ly_new
                shared['lz'].value = lz_new
                if hud:
                    hud.text = '↓ Relight'
                    canvas.update()
            elif event.key.name == 'Left':
                lx_new = max(-1.0, min(1.0, shared['lx'].value - 0.05))
                ly_new = max(-1.0, min(1.0, shared['ly'].value))
                lz_new = max(0.1, np.sqrt(max(0, 1.0 - lx_new**2 - ly_new**2)))
                shared['lx'].value = lx_new
                shared['ly'].value = ly_new
                shared['lz'].value = lz_new
                if hud:
                    hud.text = '← Relight'
                    canvas.update()
            elif event.key.name == 'Right':
                lx_new = max(-1.0, min(1.0, shared['lx'].value + 0.05))
                ly_new = max(-1.0, min(1.0, shared['ly'].value))
                lz_new = max(0.1, np.sqrt(max(0, 1.0 - lx_new**2 - ly_new**2)))
                shared['lx'].value = lx_new
                shared['ly'].value = ly_new
                shared['lz'].value = lz_new
                if hud:
                    hud.text = '→ Relight'
                    canvas.update()
        except Exception:
            _log.warning("Failed in key press handler", exc_info=True)

    @canvas.events.key_release.connect
    def on_key_release(event):
        try:
            if event.key.name == 'Shift':
                pan_state['shift_held'] = False
                pan_state['active'] = False
        except Exception:
            _log.warning("Failed in key release handler", exc_info=True)

    @canvas.events.mouse_press.connect
    def on_mouse_press(event):
        try:
            if (pan_state['shift_held'] and event.button == 1) or event.button == 2:
                pan_state['active'] = True
                pan_state['last_x'], pan_state['last_y'] = event.pos[0], event.pos[1]
                event.handled = True
        except Exception:
            _log.warning("Failed in mouse press handler", exc_info=True)

    @canvas.events.mouse_double_click.connect
    def on_mouse_double_click(event):
        try:
            camera.center = (0, 0, 0)
            camera.set_range(margin=0.30)
            canvas.update()
        except Exception:
            _log.warning("Failed in mouse double-click handler", exc_info=True)

    @canvas.events.mouse_release.connect
    def on_mouse_release(event):
        try:
            if pan_state['active']:
                pan_state['active'] = False
                event.handled = True
        except Exception:
            _log.warning("Failed in mouse release handler", exc_info=True)

    @canvas.events.mouse_move.connect
    def on_mouse_move(event):
        try:
            if pan_state['active'] and len(event.buttons) > 0:
                dx = event.pos[0] - pan_state['last_x']
                dy = event.pos[1] - pan_state['last_y']
                pan_state['last_x'], pan_state['last_y'] = event.pos[0], event.pos[1]
                if camera.distance is None:
                    return
                scale = camera.distance * 0.0015
                cx, cy, cz = camera.center
                camera.center = (cx - dx * scale, cy + dy * scale, cz)
                event.handled = True
                canvas.update()
        except Exception:
            _log.warning("Failed in mouse move handler", exc_info=True)

    _float_tick = [0]

    def update(ev):
        try:
            if shared['clean_export'].value == 1:
                try:
                    shared['clean_export'].value = 0
                    if hud:
                        hud.visible = False
                    if hud_counter:
                        hud_counter.visible = False
                    canvas.update()
                    _saved_bg = canvas.bgcolor
                    canvas.bgcolor = (0.0, 0.0, 0.0, 1.0)
                    canvas.update()
                    _frame_black = canvas.render().astype(np.float32)
                    canvas.bgcolor = (1.0, 1.0, 1.0, 1.0)
                    canvas.update()
                    _frame_white = canvas.render().astype(np.float32)
                    canvas.bgcolor = _saved_bg
                    if hud:
                        hud.visible = True
                    if hud_counter:
                        hud_counter.visible = True
                    canvas.update()
                    fb = _frame_black[..., :3]
                    fw = _frame_white[..., :3]
                    diff = np.clip(fw - fb, 0, 255)
                    alpha_f = 1.0 - (diff / 255.0)
                    alpha_ch = np.max(alpha_f, axis=2)
                    a_safe = np.where(alpha_ch > 0.01, alpha_ch, 1.0)
                    rgb_out = np.clip(fb / a_safe[..., np.newaxis], 0, 255).astype(np.uint8)
                    alpha_out = (np.clip(alpha_ch, 0, 1) * 255).astype(np.uint8)
                    bgra = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
                    bgra_4 = np.dstack([bgra, alpha_out])
                    import tempfile as _tmpf
                    _out = os.path.join(_tmpf.gettempdir(),
                                        f"HELIO_3D_CleanRender_{int(time.time())}.png")
                    cv2.imwrite(_out, bgra_4)
                    _marker = os.path.join(_tmpf.gettempdir(), "helio_last_clean_export.txt")
                    with open(_marker, "w") as _mf:
                        _mf.write(_out)
                    print(f"[Helio] Clean export saved (transparent PNG): {_out}")
                except Exception as _ce:
                    _log.error("Clean export error: %s", _ce, exc_info=True)

            if shared['raise_3d'].value == 1:
                try:
                    import glfw as _glfw
                    _glfw.focus_window(canvas.native)
                    _glfw.set_window_attrib(canvas.native, _glfw.FLOATING, True)
                except Exception:
                    _log.debug("Failed to focus GLFW window")
                shared['raise_3d'].value = 0
            _float_tick[0] += 1
            if _float_tick[0] >= 30:
                _float_tick[0] = 0
                try:
                    import glfw as _glfw
                    _glfw.set_window_attrib(canvas.native, _glfw.FLOATING, True)
                except Exception:
                    _log.debug("Failed to set GLFW floating attribute")
                try:
                    from AppKit import NSApp, NSFloatingWindowLevel
                    for _w in NSApp.windows():
                        try:
                            if "3D Mesh" in str(_w.title()) or "HELIO" in str(_w.title()):
                                _w.setLevel_(NSFloatingWindowLevel)
                        except Exception:
                            _log.debug("Failed to set NSWindow level")
                except Exception:
                    _log.debug("Failed to import AppKit")
            import math as _math
            if shared['auto_spin'].value:
                camera.azimuth += shared['spin_speed'].value
            _sc = getattr(update, '_spin_cap', {'mode': 0, 'frame': 0, 'total': 72, 'dir': ''})
            update._spin_cap = _sc
            if shared['spin_capture'].value > 0 and _sc['mode'] == 0:
                import tempfile as _tf
                _m = shared['spin_capture'].value
                _tot = shared['spin_total'].value
                _sc.update({'mode': _m, 'total': _tot, 'frame': 0,
                            'dir': os.path.join(_tf.gettempdir(), 'helio3d_spin')})
                os.makedirs(_sc['dir'], exist_ok=True)
                [os.remove(os.path.join(_sc['dir'], f)) for f in os.listdir(_sc['dir'])
                 if f.startswith('frm_') and f.endswith('.png')]
                camera.azimuth = 0.0
                camera.elevation = 30.0
                shared['auto_spin'].value = 1
            if _sc['mode'] > 0:
                t = _sc['frame'] / max(1, _sc['total'])
                if _sc['mode'] in (3, 4):
                    n = _sc['total']
                    p1 = int(n * 0.50)
                    p2 = int(n * 0.75)
                    fr = _sc['frame']
                    if fr < p1:
                        camera.azimuth = (fr / max(1, p1)) * 360.0
                        camera.elevation = 30.0
                    elif fr < p2:
                        t_r = (fr - p1) / max(1, p2 - p1)
                        ease = 0.5 - 0.5 * _math.cos(t_r * _math.pi)
                        camera.azimuth = 360.0 + t_r * 120.0
                        camera.elevation = 30.0 + ease * 55.0
                    else:
                        p3a = int(n * 0.88)
                        if fr < p3a:
                            t_s = (fr - p2) / max(1, p3a - p2)
                            ease = 0.5 - 0.5 * _math.cos(t_s * _math.pi)
                            camera.azimuth = 480.0 * (1.0 - ease) + 360.0 * ease
                            camera.elevation = 85.0 + ease * 4.9
                        else:
                            camera.azimuth = 360.0
                            camera.elevation = 89.9
                canvas.update()
                try:
                    import cv2 as _cv2
                    _cv2.imwrite(
                        os.path.join(_sc['dir'], f"frm_{_sc['frame']:04d}.png"),
                        _cv2.cvtColor(canvas.render(), _cv2.COLOR_RGBA2BGRA),
                    )
                except Exception:
                    _log.warning("Failed to save spin capture frame", exc_info=True)
                _sc['frame'] += 1
                if _sc['frame'] >= _sc['total']:
                    shared['spin_frames_done'].value = _sc['total']
                    shared['spin_capture'].value = 0
                    shared['auto_spin'].value = 0
                    _sc['mode'] = 0
            bg = shared['bg_light'].value
            if bg != state['bg_light']:
                state['bg_light'] = bg
                canvas.bgcolor = '#ffffff' if bg == 1 else '#101010'
                canvas.update()

            step = max(1, shared['step'].value)
            height_scale = shared['height'].value
            warp_rem = shared['warp_rem'].value
            smooth = shared['mesh_smooth'].value
            fx, fy, fz = shared['fx'].value, shared['fy'].value, shared['fz'].value
            lx, ly, lz = shared['lx'].value, shared['ly'].value, shared['lz'].value
            clay = shared['clay_blend'].value
            edge_trim = shared['edge_trim'].value
            amb_str = shared['ambient_str'].value
            diff_str = shared['diffuse_str'].value
            spec_str = shared['specular_str'].value
            shiny = max(1.0, shared['shininess'].value)
            v_gamma = max(0.1, shared['viewer_gamma'].value) if 'viewer_gamma' in shared else 1.0
            z_slice_lo = shared['z_slice_lo'].value if 'z_slice_lo' in shared else 0.0
            z_slice_hi = shared['z_slice_hi'].value if 'z_slice_hi' in shared else 1.0
            hq_normals = shared['hq_normals'].value if 'hq_normals' in shared else 0
            cross_sec = shared['cross_section'].value if 'cross_section' in shared else 0
            cross_sec_y = shared['cross_section_y'].value if 'cross_section_y' in shared else 0.5
            vertex_cap = shared['vertex_cap'].value if 'vertex_cap' in shared else 0
            v_sat = max(0.0, shared['viewer_saturation'].value) if 'viewer_saturation' in shared else 1.0
            v_shadow = max(0.0, shared['shadow_lift'].value) if 'shadow_lift' in shared else 0.0
            v_hilite = max(0.5, shared['highlight_clip'].value) if 'highlight_clip' in shared else 1.0
            g_whites = shared['global_whites'].value if 'global_whites' in shared else 1.0
            g_blacks = shared['global_blacks'].value if 'global_blacks' in shared else 0.0
            g_midtone = max(0.1, shared['global_midtone'].value) if 'global_midtone' in shared else 1.0
            g_clarity = shared['global_clarity'].value if 'global_clarity' in shared else 0.0
            g_texture = shared['global_texture'].value if 'global_texture' in shared else 0.0
            g_dehaze = shared['global_dehaze'].value if 'global_dehaze' in shared else 0.0
            bake_3d = shared['bake_3d'].value
            center_origin = shared['center_origin'].value
            use_c_clay = shared['use_custom_clay'].value
            clay_r, clay_g, clay_b = shared['clay_r'].value, shared['clay_g'].value, shared['clay_b'].value
            hp_sigma = shared['hp_sigma'].value
            micro = shared['micro_relief'].value
            shape_blend = shared['shape_blend'].value
            hybrid_enabled = shared['hybrid_enabled'].value if 'hybrid_enabled' in shared else 0
            hybrid_mix = max(0.0, min(1.0, shared['hybrid_mix'].value)) if 'hybrid_mix' in shared else 0.12
            depth_clip_v = max(0.01, min(4.9, shared['depth_clip_pct'].value))
            x_tilt_v = shared['x_tilt'].value
            y_tilt_v = shared['y_tilt'].value
            depth_gen_v = shared['depth_gen'].value if 'depth_gen' in shared else 0

            recalc_geom = (
                depth_gen_v != state.get('depth_gen', 0)
                or
                fx != state['fx'] or fy != state['fy'] or fz != state['fz']
                or step != state['step'] or height_scale != state['height']
                or edge_trim != state['edge_trim'] or warp_rem != state['warp_rem']
                or smooth != state['mesh_smooth'] or center_origin != state['center_origin']
                or hp_sigma != state['hp_sigma'] or micro != state['micro']
                or shape_blend != state['shape_blend']
                or hybrid_enabled != state.get('hybrid_enabled', 0)
                or hybrid_mix != state.get('hybrid_mix', 0.12)
                or x_tilt_v != state.get('x_tilt', 0.)
                or y_tilt_v != state.get('y_tilt', 0.)
                or depth_clip_v != state.get('depth_clip', 0.5)
                or hq_normals != state.get('hq_normals', 0)
                or cross_sec != state.get('cross_sec', 0)
                or cross_sec_y != state.get('cross_sec_y', 0.5)
                or vertex_cap != state.get('vertex_cap', 0)
            )
            recalc_light = (
                lx != state['lx'] or ly != state['ly'] or lz != state['lz']
                or clay != state['clay_blend'] or amb_str != state['ambient']
                or diff_str != state['diffuse'] or spec_str != state['specular']
                or shiny != state['shininess'] or bake_3d != state['bake_3d']
                or use_c_clay != state['use_custom_clay']
                or clay_r != state['clay_r'] or clay_g != state['clay_g']
                or clay_b != state['clay_b']
                or v_gamma != state.get('v_gamma', 1.) or v_sat != state.get('v_sat', 1.)
                or v_shadow != state.get('v_shadow', 0.)
                or v_hilite != state.get('v_hilite', 1.)
                or g_whites != state.get('g_whites', 1.)
                or g_blacks != state.get('g_blacks', 0.)
                or g_midtone != state.get('g_midtone', 1.)
                or g_clarity != state.get('g_clarity', 0.)
                or g_texture != state.get('g_texture', 0.)
                or g_dehaze != state.get('g_dehaze', 0.)
                or z_slice_lo != state.get('z_slice_lo', 0.)
                or z_slice_hi != state.get('z_slice_hi', 1.)
            )

            if recalc_geom or state['points'] is None:
                state.update({
                    'fx': fx, 'fy': fy, 'fz': fz, 'step': step,
                    'height': height_scale, 'edge_trim': edge_trim,
                    'warp_rem': warp_rem, 'mesh_smooth': smooth,
                    'center_origin': center_origin, 'hp_sigma': hp_sigma,
                    'micro': micro, 'shape_blend': shape_blend,
                    'hybrid_enabled': hybrid_enabled, 'hybrid_mix': hybrid_mix,
                })
                trim_h = max(1, int(h_orig * (edge_trim / 100.0))) if edge_trim > 0 else 0
                trim_w = max(1, int(w_orig * (edge_trim / 100.0))) if edge_trim > 0 else 0
                if shared.get('viewer_skip_trim') and shared['viewer_skip_trim'].value == 1:
                    trim_h = trim_w = 0
                y1, y2, x1, x2 = trim_h, h_orig - trim_h, trim_w, w_orig - trim_w
                _eff_step = step
                if vertex_cap > 0:
                    _vis_h = max(1, (y2 - y1))
                    _vis_w = max(1, (x2 - x1))
                    _est_verts = (_vis_h // _eff_step) * (_vis_w // _eff_step)
                    if _est_verts > vertex_cap:
                        while _est_verts > vertex_cap and _eff_step < 32:
                            _eff_step += 1
                            _est_verts = (_vis_h // _eff_step) * (_vis_w // _eff_step)
                step = _eff_step
                y_s, x_s = np.mgrid[y1:y2:step, x1:x2:step]
                h_curr, w_curr = len(y_s[:, 0]), len(x_s[0, :])
                d_full_crop = raw_depth_full[y1:y2:step, x1:x2:step]
                d_hp_crop = raw_depth_hp[y1:y2:step, x1:x2:step]
                if hybrid_enabled:
                    def _norm01(a):
                        lo, hi = a.min(), a.max()
                        return ((a - lo) / (hi - lo)).astype(np.float32) if hi > lo + 1e-9 else np.zeros_like(a)
                    z_raw = _norm01(d_full_crop) * (1.0 - hybrid_mix) + _norm01(d_hp_crop) * hybrid_mix
                    z_raw = _norm01(z_raw)
                else:
                    sb = max(0.0, min(1.0, shape_blend))
                    z_raw = (d_full_crop * (1.0 - sb) + d_hp_crop * sb).copy()
                    if sb > 0.4:
                        _diag = np.sqrt(float(h_curr)**2 + float(w_curr)**2)
                        _prog_sigma = max(3.0, (sb - 0.4) * 0.20 * _diag)
                        z_raw = z_raw - cv2.GaussianBlur(z_raw, (0, 0), _prog_sigma)
                if hp_sigma > 0.01:
                    sigma_px = float(hp_sigma) * max(h_curr, w_curr) * 0.4
                    z_raw = z_raw - cv2.GaussianBlur(z_raw, (0, 0), sigma_px)
                if smooth > 0:
                    z_raw = cv2.GaussianBlur(z_raw, (0, 0), sigmaX=float(smooth))
                if warp_rem > 0:
                    z_raw = z_raw - cv2.GaussianBlur(z_raw, (0, 0), max(h_curr, w_curr) * (0.05 + warp_rem * 0.45))
                if micro > 0:
                    base_lf = cv2.GaussianBlur(z_raw, (0, 0), max(h_curr, w_curr) * 0.05)
                    z_raw = base_lf + (z_raw - base_lf) * (1.0 + micro)
                _lo = np.percentile(z_raw, depth_clip_v)
                _hi = np.percentile(z_raw, 100. - depth_clip_v)
                z_raw = np.clip(z_raw, _lo, _hi)
                if abs(x_tilt_v) > 0.001 or abs(y_tilt_v) > 0.001:
                    _XG, _YG = np.meshgrid(np.linspace(-.5, .5, w_curr),
                                           np.linspace(-.5, .5, h_curr))
                    z_raw -= (_XG * np.tan(np.radians(x_tilt_v)) +
                              _YG * np.tan(np.radians(y_tilt_v)))
                state.update({
                    'x_tilt': x_tilt_v, 'y_tilt': y_tilt_v,
                    'depth_gen': depth_gen_v,
                    'depth_clip': depth_clip_v, '_z_ao': z_raw.copy(),
                })
                x_c = (x_s - w_orig / 2) * fx
                y_c = (y_s - h_orig / 2) * fy
                z_vals = (z_raw * fz * height_scale) - (height_scale / 2)
                points = np.column_stack((
                    x_c.flatten(), y_c.flatten(), z_vals.flatten(),
                )).astype(np.float32)
                if center_origin == 1:
                    points[:, 0] -= np.mean(points[:, 0])
                    points[:, 1] -= np.mean(points[:, 1])
                    points[:, 2] -= np.mean(points[:, 2])
                rr, cc = np.meshgrid(np.arange(h_curr - 1), np.arange(w_curr - 1), indexing='ij')
                v0 = rr * w_curr + cc
                faces_top = np.vstack((
                    np.column_stack((
                        v0.flatten(), (v0 + w_curr).flatten(), (v0 + 1).flatten(),
                    )),
                    np.column_stack((
                        (v0 + 1).flatten(), (v0 + w_curr).flatten(),
                        (v0 + w_curr + 1).flatten(),
                    )),
                )).astype(np.uint32)
                state.update({
                    'points': points, 'faces': faces_top,
                    'y1': y1, 'y2': y2, 'x1': x1, 'x2': x2,
                    'step_used': step, 'hq_normals': hq_normals,
                    'cross_sec': cross_sec, 'cross_sec_y': cross_sec_y,
                    'vertex_cap': vertex_cap,
                })
                recalc_light = True

            if recalc_light and state['points'] is not None:
                state.update({
                    'lx': lx, 'ly': ly, 'lz': lz, 'clay_blend': clay,
                    'ambient': amb_str, 'diffuse': diff_str, 'specular': spec_str,
                    'shininess': shiny, 'bake_3d': bake_3d,
                    'use_custom_clay': use_c_clay,
                    'clay_r': clay_r, 'clay_g': clay_g, 'clay_b': clay_b,
                })
                y1, y2, x1, x2 = state['y1'], state['y2'], state['x1'], state['x2']
                st = state['step_used']
                pure_colour = img_rgb[y1:y2:st, x1:x2:st].astype(np.float32) / 255.0
                clay_fill = (
                    np.array([clay_r, clay_g, clay_b], dtype=np.float32).reshape(1, 1, 3)
                    if use_c_clay == 1 else np.float32(0.6)
                )
                base_colour = (clay_fill * clay) + (pure_colour * (1.0 - clay))
                if state.get('hq_normals', 0) == 1 and normal_map is not None:
                    try:
                        nm_crop = normal_map[y1:y2:st, x1:x2:st]
                        nm_flat = nm_crop.reshape(-1, 3).astype(np.float32)
                        nm_len = np.linalg.norm(nm_flat, axis=1, keepdims=True)
                        _n_hq = np.divide(nm_flat, nm_len,
                                          out=np.zeros_like(nm_flat), where=nm_len > 1e-8)
                    except Exception:
                        _log.warning("Failed to compute HQ normals", exc_info=True)
                        _n_hq = None
                else:
                    _n_hq = None
                if bake_3d == 1:
                    lit_rgb = np.clip(
                        base_colour.reshape(-1, 3) * (amb_str + diff_str), 0.0, 1.0)
                else:
                    l_vec = np.array([lx, ly, lz], dtype=np.float32)
                    l_norm = np.linalg.norm(l_vec)
                    if l_norm < 1e-6:
                        l_vec = np.array([0., 0., 1.], dtype=np.float32)
                    else:
                        l_vec /= l_norm
                    n_grid = normal_map[y1:y2:st, x1:x2:st].copy()
                    if _n_hq is not None:
                        try:
                            _n_hq_grid = _n_hq.reshape(n_grid.shape[0], n_grid.shape[1], 3)
                            n_grid = _n_hq_grid.copy()
                        except Exception:
                            _log.warning("Failed to reshape HQ normal grid", exc_info=True)
                    n_grid[:, :, 0] *= fx
                    n_grid[:, :, 1] *= fy
                    n_grid[:, :, 2] *= fz
                    _n_inv = shared['normal_invert'].value if 'normal_invert' in shared else 0
                    _n_fx = shared['normal_flip_x'].value if 'normal_flip_x' in shared else 0
                    if _n_inv:
                        n_grid[:, :, 0] = -n_grid[:, :, 0]
                        n_grid[:, :, 1] = -n_grid[:, :, 1]
                    elif _n_fx:
                        n_grid[:, :, 0] = -n_grid[:, :, 0]
                    _nlen2 = np.linalg.norm(n_grid, axis=-1, keepdims=True)
                    n_grid = np.divide(n_grid, np.maximum(_nlen2, 1e-8))
                    diff_l = np.clip(np.maximum(0.0, np.dot(n_grid, l_vec)), 0.0, 1.0) * diff_str
                    h_vec = l_vec + np.array([0., 0., 1.], dtype=np.float32)
                    h_norm = np.linalg.norm(h_vec)
                    if h_norm > 1e-6:
                        h_vec /= h_norm
                    spec_l = np.power(
                        np.clip(np.maximum(0.0, np.dot(n_grid, h_vec)), 0.0, 1.0),
                        shiny,
                    ) * spec_str
                    total = amb_str + diff_l + spec_l
                    lit_rgb = np.clip(
                        base_colour.reshape(-1, 3) * total.reshape(-1, 1), 0.0, 1.0)
                state.update({
                    'v_gamma': v_gamma, 'v_sat': v_sat,
                    'v_shadow': v_shadow, 'v_hilite': v_hilite,
                    'g_whites': g_whites, 'g_blacks': g_blacks,
                    'g_midtone': g_midtone, 'g_clarity': g_clarity,
                    'g_texture': g_texture, 'g_dehaze': g_dehaze,
                })
                _apply_tone = (
                    v_shadow > 0.001 or v_hilite < 0.999 or v_gamma != 1.0 or v_sat != 1.0
                    or abs(g_whites - 1) > 0.01 or g_blacks > 0.01
                    or abs(g_midtone - 1) > 0.01 or g_clarity > 0.01
                    or g_texture > 0.01 or g_dehaze > 0.01
                )
                if _apply_tone:
                    _lr = lit_rgb.copy()
                    if v_shadow > 0.001 or v_hilite < 0.999:
                        _lr = np.clip(_lr * (v_hilite - v_shadow) + v_shadow, 0, 1)
                    if v_gamma != 1.0:
                        _lr = np.power(np.clip(_lr, 1e-6, 1.0), 1.0 / v_gamma)
                    if v_sat != 1.0:
                        _lum = 0.299 * _lr[:, 0] + 0.587 * _lr[:, 1] + 0.114 * _lr[:, 2]
                        _lr = np.clip(
                            _lum[:, np.newaxis] + (_lr - _lum[:, np.newaxis]) * v_sat, 0, 1)
                    if abs(g_whites - 1) > 0.01 or g_blacks > 0.01:
                        _lr = np.clip(_lr * g_whites + g_blacks, 0, 1)
                    if abs(g_midtone - 1) > 0.01:
                        _lr = np.power(
                            np.clip(_lr, 1e-6, 1.0), 1.0 / max(0.1, g_midtone))
                    if g_dehaze > 0.01:
                        for _ci in range(_lr.shape[1]):
                            _col = _lr[:, _ci]
                            _p1, _p99 = float(np.percentile(_col, 1)), float(np.percentile(_col, 99))
                            if _p99 - _p1 > 0.01:
                                _stretched = np.clip((_col - _p1) / (_p99 - _p1), 0, 1)
                                _lr[:, _ci] = _col * (1 - g_dehaze) + _stretched * g_dehaze
                    for _g_val, _sigma_frac in [(g_clarity, 0.15), (g_texture, 0.03)]:
                        if _g_val > 0.01:
                            _lum2 = _lr.mean(axis=1, keepdims=True)
                            _mean_lum = float(_lum2.mean())
                            _lr = np.clip(_lr + (_lum2 - _mean_lum) * _g_val * 0.5, 0, 1)
                    lit_rgb = _lr
                state.update({'z_slice_lo': z_slice_lo, 'z_slice_hi': z_slice_hi})
                _z_norm = state.get('_z_ao')
                if _z_norm is not None and (z_slice_lo > 0.001 or z_slice_hi < 0.999):
                    _zf = _z_norm.flatten()
                    _z_min, _z_max = _zf.min(), _zf.max()
                    _z_range = max(_z_max - _z_min, 1e-6)
                    _z_01 = (_zf - _z_min) / _z_range
                    _lo = min(z_slice_lo, z_slice_hi)
                    _hi = max(z_slice_lo, z_slice_hi)
                    _alpha = np.where((_z_01 >= _lo) & (_z_01 <= _hi), 1.0, 0.0).astype(np.float32)
                else:
                    _alpha = np.ones(len(lit_rgb), dtype=np.float32)
                if state.get('cross_sec', 0) == 1 and _z_norm is not None:
                    _pts = state.get('points')
                    if _pts is not None:
                        _y_pts = _pts[:, 1]
                        _y_min, _y_max = _y_pts.min(), _y_pts.max()
                        _y_range = max(_y_max - _y_min, 1e-6)
                        _y_norm = (_y_pts - _y_min) / _y_range
                        _cut = float(state.get('cross_sec_y', 0.5))
                        _xs_mask = (_y_norm <= _cut).astype(np.float32)
                        _alpha = _alpha * _xs_mask
                lit_rgba = np.ascontiguousarray(
                    np.column_stack((lit_rgb, _alpha)), dtype=np.float32)
                mesh.set_data(
                    vertices=np.ascontiguousarray(state['points']),
                    faces=np.ascontiguousarray(state['faces']),
                    vertex_colors=lit_rgba,
                )
                canvas.update()
                if not state['camera_set']:
                    try:
                        camera.set_range(margin=0.30)
                        try:
                            if hasattr(camera, 'fov'):
                                camera.fov = 45.0
                        except Exception:
                            _log.debug("Failed to set camera FOV")
                    except TypeError:
                        camera.set_range()
                        camera.distance *= 1.35
                    state['camera_set'] = True
        except Exception:
            _log.warning("Unhandled error in viewer update loop", exc_info=True)

    timer = app.Timer(0.05, connect=update, start=True)
    app.run()


def build_viewer_mesh(normals, depth, step=2, edge_trim=2.0):
    """Build the vertex and face arrays for the 3D mesh.

    Returns ``(vertices, faces)``. It has never returned colours - the
    headless renderer unpacked three values from it and raised on every run.
    """
    h_orig, w_orig = depth.shape
    trim_h = max(1, int(h_orig * (edge_trim / 100.0))) if edge_trim > 0 else 0
    trim_w = max(1, int(w_orig * (edge_trim / 100.0))) if edge_trim > 0 else 0
    y1, y2, x1, x2 = trim_h, h_orig - trim_h, trim_w, w_orig - trim_w
    y_s, x_s = np.mgrid[y1:y2:step, x1:x2:step]
    h_curr, w_curr = len(y_s[:, 0]), len(x_s[0, :])
    z_crop = depth[y1:y2:step, x1:x2:step]
    x_c = (x_s - w_orig / 2)
    y_c = (y_s - h_orig / 2)
    z_vals = z_crop
    points = np.column_stack((
        x_c.flatten(), y_c.flatten(), z_vals.flatten(),
    )).astype(np.float32)
    rr, cc = np.meshgrid(np.arange(h_curr - 1), np.arange(w_curr - 1), indexing='ij')
    v0 = rr * w_curr + cc
    faces = np.vstack((
        np.column_stack((
            v0.flatten(), (v0 + w_curr).flatten(), (v0 + 1).flatten(),
        )),
        np.column_stack((
            (v0 + 1).flatten(), (v0 + w_curr).flatten(),
            (v0 + w_curr + 1).flatten(),
        )),
    )).astype(np.uint32)
    return points, faces
