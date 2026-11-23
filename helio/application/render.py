"""Render a view of the solved surface, GUI-free (VIEW-001).

``render`` is the body of the classic ``RTIViewerPro.get_shading``, moved:
it picks the full-resolution or preview arrays, applies the display-only
overrides, converts the light from screen frame to the normals' +Y-up frame
(the one place that happens, contracts.md §3.2) and dispatches to the engine.
Both apps call it, so a view looks the same in each.
"""
from __future__ import annotations

import cv2
import numpy as np

from helio.core.parallel import by_rows, cached, map_rows
from helio.core.shading_engines import get_shading_engine, implemented_engines


def render(engine: str, data, settings: dict, h: int, w: int, use_proxy: bool,
           crop_coords=None, *, normal_invert: bool = False, normal_flip_x: bool = False,
           depth_invert: bool = False, res_res: float = 1.0, pretrimmed_crop_frac=None):
    """The view *engine* of *data* (a helio.data_manager.DataManager).

    *settings* are the engine's settings, light ``lx, ly, lz`` in screen frame
    (+Y down, as the light dome and bookmarks store it).  *use_proxy* renders
    the preview-size arrays; otherwise the full arrays, cut to *crop_coords*.
    Returns the engine's array, or None for an engine with no implementation.
    """
    if getattr(data, 'normals_full', None) is None:
        return np.zeros((h, w), dtype=np.float32)
    gain = settings.get('gain', 1.0)
    # Screen frame (+Y down: the dome dot is drawn at py = cy + vy*r) to the
    # normals' frame (+Y up: ny = (cy - hy)/r).  Negate Y here, and nowhere
    # else, so saved bookmarks and engine settings keep working unchanged.
    light = np.array([settings.get('lx', 0),
                      -settings.get('ly', 0),
                      settings.get('lz', 1)])

    # A real crop, not just full-image fallback coordinates.
    is_real_crop = False
    if crop_coords and not use_proxy:
        cx1, cy1, cx2, cy2 = crop_coords
        if cx1 > 0 or cy1 > 0 or cx2 < data.w or cy2 < data.h:
            is_real_crop = True

    if use_proxy:
        pnf = getattr(data, 'normals_proxy', None)
        pnr = getattr(data, 'normals_robust_proxy', None)
        pmi = getattr(data, 'median_proxy', None)
        pdp = getattr(data, 'depth_proxy_p', None)
        pdfc = getattr(data, 'depth_proxy_fc', None)
        ppt = getattr(data, 'ptm_proxy', None)
        n_std = pnf
        n_rob = pnr if pnr is not None else pnf
        med = pmi if pmi is not None else np.zeros(
            (pnf.shape[0], pnf.shape[1]) if pnf is not None else (h, w), np.float32)
        depth_p = pdp if pdp is not None else np.zeros_like(med)
        depth_fc = pdfc if pdfc is not None else depth_p
        ptm = ppt if ppt is not None else np.zeros((6, med.size), np.float32)
        # Always take h, w from the proxy arrays themselves: a stored ph/pw can
        # go stale, and every reshape(h, w) crashes if h*w != the array size.
        src = n_std if (n_std is not None and n_std.ndim == 3) else depth_p
        if src is not None:
            h, w = int(src.shape[0]), int(src.shape[1])
        # PTM engines reshape the coefficients to (h, w): rebuild if they disagree.
        if ptm is not None and ptm.ndim == 2 and ptm.shape[1] != h * w:
            ptm = np.zeros((6, h * w), np.float32)
    else:
        x1, y1, x2, y2 = crop_coords if crop_coords else (0, 0, data.w, data.h)
        # Any of these can be None if processing was interrupted.
        nf = getattr(data, 'normals_full', None)
        if nf is None:
            nf = np.zeros((data.h, data.w, 3), np.float32)
        nr = getattr(data, 'normals_robust', None)
        if nr is None:
            nr = nf
        mi = getattr(data, 'median_img', None)
        if mi is None:
            mi = np.zeros((data.h, data.w), np.float32)
        dvp = getattr(data, 'depth_vis_p', None)
        if dvp is None:
            dvp = np.zeros((data.h, data.w), np.float32)
        dvfc = getattr(data, 'depth_vis_fc', None)
        if dvfc is None:
            dvfc = dvp
        pt = getattr(data, 'ptm_coeffs', None)
        if pt is None and getattr(data, 'ptm_deferred', False) and engine in _PTM_ENGINES:
            pt = data.ensure_ptm_hsh()[0]            # a tight machine: fitted again, once
        if pt is None:
            pt = np.zeros((6, data.h * data.w), np.float32)
        n_std, n_rob, med, depth_p, depth_fc, ptm = (
            nf[y1:y2, x1:x2], nr[y1:y2, x1:x2], mi[y1:y2, x1:x2],
            dvp[y1:y2, x1:x2], dvfc[y1:y2, x1:x2],
            pt.reshape(6, data.h, data.w)[:, y1:y2, x1:x2].reshape(6, -1))
        h, w = int(n_std.shape[0]), int(n_std.shape[1])
        if n_std.size == 0 or n_std.shape[0] == 0 or n_std.shape[1] == 0:
            return np.zeros((max(1, h), max(1, w)), dtype=np.float32)

    # Display-only overrides: the stored normals and depth are untouched.  At
    # preview size the overridden copy is kept, so a drag does not copy the
    # normals every frame and the engines' own caches still recognise them.
    if normal_invert or normal_flip_x:
        src_n = n_std

        def flipped():
            n2 = src_n.copy().astype(np.float32)
            if normal_invert:
                n2[..., 0] = -n2[..., 0]
                n2[..., 1] = -n2[..., 1]
            elif normal_flip_x:
                n2[..., 0] = -n2[..., 0]
            return n2
        n_std = (cached("normals-display", (src_n,), (bool(normal_invert), bool(normal_flip_x)),
                        flipped) if use_proxy else flipped())
    if depth_invert and depth_p is not None:
        if use_proxy:
            dp, dfc = depth_p, depth_fc
            depth_p = cached("depth-inverted", (dp,), (), lambda: dp.max() - dp)
            depth_fc = cached("depth-inverted", (dfc,), (), lambda: dfc.max() - dfc)
        else:
            depth_p = depth_p.max() - depth_p
            depth_fc = depth_fc.max() - depth_fc

    if engine not in implemented_engines():
        return None
    ctx = dict(n_std=n_std, n_rob=n_rob, med=med, depth_p=depth_p, depth_fc=depth_fc,
               ptm=ptm, light=light, gain=gain, res_res=res_res, use_proxy=use_proxy,
               _is_real_crop=is_real_crop, crop_coords=crop_coords,
               _pretrimmed_crop_frac=pretrimmed_crop_frac)
    # PTM at preview size.  These views once relit the full-size coefficients
    # every frame and shrank the result, because the preview's own PTM planes
    # are resampled differently.  Relighting is linear in the coefficients and
    # so is area-averaging, so averaging the six planes down once and
    # relighting those gives the same picture, without touching every pixel
    # of a 60-megapixel capture on each move of the light.
    if use_proxy and engine in ("Specular Enhancement", "Image Quotient (IQ)"):
        full_ptm = getattr(data, 'ptm_coeffs', None)
        fh, fw = data.h, data.w
        planes = getattr(data, 'ptm_planes', None)
        if full_ptm is None and planes is not None and planes[0].shape == (h, w):
            c = planes                                # kept in place of the coefficients
            u, v = light[0], light[1]
            relit = np.empty((h, w), np.float32)
            quad = np.empty((h, w), np.float32)

            def relight(y0, y1):
                q = c[0][y0:y1]*u*u + c[1][y0:y1]*v*v
                quad[y0:y1] = q
                relit[y0:y1] = (q + c[2][y0:y1]*u*v + c[3][y0:y1]*u
                                + c[4][y0:y1]*v + c[5][y0:y1])
            by_rows(relight, h, w)
            ctx['_ptm_relit_proxy'] = relit
            ctx['_ptm_quad_proxy'] = quad
        elif (full_ptm is not None and full_ptm.ndim == 2 and full_ptm.shape[0] >= 6
                and fh and fw and full_ptm.shape[1] == fh * fw):
            c = _ptm_area_planes(full_ptm, fh, fw, w, h)
            u, v = light[0], light[1]
            relit = np.empty((h, w), np.float32)
            quad = np.empty((h, w), np.float32)

            def relight(y0, y1):
                q = c[0][y0:y1]*u*u + c[1][y0:y1]*v*v
                quad[y0:y1] = q
                relit[y0:y1] = (q + c[2][y0:y1]*u*v + c[3][y0:y1]*u
                                + c[4][y0:y1]*v + c[5][y0:y1])
            by_rows(relight, h, w)
            ctx['_ptm_relit_proxy'] = relit
            ctx['_ptm_quad_proxy'] = quad
    return get_shading_engine(engine, data, settings, h, w, ctx=ctx)


_PTM_ENGINES = ("Specular Enhancement", "Image Quotient (IQ)", "RTI Specular Enhancement")


def _ptm_area_planes(full_ptm, fh: int, fw: int, w: int, h: int):
    """The six PTM coefficient planes area-averaged to (h, w): light-free, kept."""
    def make():
        return tuple(cv2.resize(np.ascontiguousarray(full_ptm[k], np.float32).reshape(fh, fw),
                                (w, h), interpolation=cv2.INTER_AREA) for k in range(6))
    return cached("ptm-area-planes", (full_ptm,), (fh, fw, w, h), make)


CONF_COLOURS = {"Red": [255, 0, 0], "Green": [0, 255, 0], "Blue": [0, 0, 255],
                "Yellow": [255, 255, 0], "Cyan": [0, 255, 255], "Magenta": [255, 0, 255]}

_GAMMA_LUTS: dict[float, np.ndarray] = {}


def _gamma_lut(gamma: float) -> np.ndarray:
    lut = _GAMMA_LUTS.get(gamma)
    if lut is None:
        if len(_GAMMA_LUTS) > 16:
            _GAMMA_LUTS.clear()
        lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
                       ).astype("uint8")
        _GAMMA_LUTS[gamma] = lut
    return lut


def _micro_detail(albedo) -> np.ndarray:
    """The albedo's fine detail (grey minus its 3 px blur): light-free, so kept."""
    def make():
        gray = cv2.cvtColor(albedo, cv2.COLOR_RGB2GRAY).astype(float)
        return gray - cv2.GaussianBlur(gray, (0, 0), 3)
    return cached("micro-detail", (albedo,), (), make)


def _tone_native(shading, albedo, rough, view_mode, colour_opacity, colour_gain, rop,
                 bright, k, micro, detail, lut, plain):
    """The tone composite from helio_core_rs, one pass; None when it is not built
    or the arrays are not the C-contiguous types it takes (the NumPy code then
    runs).  Brightness and contrast are skipped under the same condition as the
    NumPy code skips them: a plain setting on a layer whose values they cannot
    change."""
    from helio.core._native import rs
    if rs is None or not hasattr(rs, "tone_rgb"):
        return None
    if view_mode not in ("Combined Colour", "Shading", "Roughness Fusion"):
        view_mode = "Shading"                    # the NumPy code's else-branch
    if shading.ndim != 2 or shading.dtype not in (np.float32, np.float64) \
            or not shading.flags.c_contiguous:
        return None
    if view_mode == "Combined Colour":
        if albedo.dtype != np.uint8 or albedo.ndim != 3 or albedo.shape[2] != 3:
            return None
        albedo = np.ascontiguousarray(albedo)    # a crop is a view: copied once per frame
    if view_mode == "Roughness Fusion" and (rough.dtype != np.uint8 or not rough.flags.c_contiguous):
        return None
    skip = plain and (view_mode == "Shading"
                      or (view_mode == "Combined Colour" and colour_opacity == 1.0 and colour_gain == 1.0))
    return rs.tone_rgb(
        shading, view_mode, float(colour_opacity), float(colour_gain), float(rop), float(bright),
        float(k), float(micro), skip,
        albedo=albedo if view_mode == "Combined Colour" else None,
        rough=rough if view_mode == "Roughness Fusion" else None,
        detail=detail, lut=lut)


def _tone_rows(shading, albedo, rough, settings: dict, view_mode: str, colour_opacity: float,
               sh: int, sw: int, colour_gain: float = 1.0) -> np.ndarray:
    """Layer, micro-detail, brightness, contrast and gamma: uint8 RGB.

    The arithmetic is the classic composite's, float64 throughout, and the
    picture is identical to it bit for bit (tests/test_render_speed.py).  What
    changed is how much of it is done:

    * every step is per pixel, so it runs band by band on every core;
    * the Shading and Roughness layers are grey, so one channel is worked
      and copied to three, instead of the same sums three times over;
    * in place where the classic made a fresh image per operation;
    * with the colour at full strength, ``255·s·0 + albedo·s·1`` is exactly
      ``albedo·s`` for any finite shading, and with brightness 1 and
      contrast 0, ``(o − 128)·1 + 128`` truncates exactly as ``o`` does
      whenever ``o`` is a float32 shading times a byte (32 significant bits,
      which float64 holds exactly), so those steps are skipped.
    """
    bright, k = settings['bright'], (1 + settings['contrast'] / 100)
    micro = settings.get('micro', 0.0)
    detail = _micro_detail(albedo) if micro > 0 else None
    gamma = settings.get('gamma', 1.0)
    lut = _gamma_lut(settings['gamma']) if gamma != 1.0 else None
    rop = settings['rough_opacity'] if view_mode == "Roughness Fusion" else 0.0
    plain = bright == 1.0 and k == 1.0 and detail is None and shading.dtype == np.float32
    native = _tone_native(shading, albedo, rough, view_mode, colour_opacity, colour_gain, rop,
                          bright, k, micro, detail, lut, plain)
    if native is not None:
        return native
    out = np.empty((sh, sw, 3), np.uint8)

    def band(y0, y1):
        s = shading[y0:y1]
        finite = bool(np.isfinite(s).all())
        exact = False                      # o is (float32 shading) x (byte), held exactly
        if view_mode == "Combined Colour":
            if colour_opacity == 1.0 and finite:
                o = np.multiply(albedo[y0:y1], s[..., None], dtype=np.float64)
                exact = True
            else:
                s3 = s[..., None]
                o = (np.full((y1 - y0, sw, 3), 255.0) * s3 * (1.0 - colour_opacity)
                     + albedo[y0:y1].astype(float) * s3 * colour_opacity)
            if colour_gain != 1.0:          # the Colour layer's own brightness
                o *= colour_gain
                exact = False
        elif view_mode == "Roughness Fusion":
            o = 255.0 * (s * (1 - rop) + (rough[y0:y1].astype(float) / 255.0) * rop)
        else:
            o = np.multiply(s, 255.0, dtype=np.float64)
            exact = finite
        if detail is not None:
            d = detail[y0:y1] * micro * 2.0
            o = np.clip(o + (d[..., None] if o.ndim == 3 else d), 0, 255)
        if not (plain and exact):
            o *= bright
            o -= 128
            o *= k
            o += 128
        np.clip(o, 0, 255, out=o)
        u8 = o.astype(np.uint8)
        if lut is not None:
            u8 = cv2.LUT(u8, lut)
        out[y0:y1] = u8 if u8.ndim == 3 else u8[..., None]

    by_rows(band, sh, sw)
    return out


def composite(engine: str, shading, data, settings: dict, h: int, w: int, use_proxy: bool,
              crop_coords=None, *, view_mode: str = "Shading", colour_opacity: float = 1.0,
              colour_gain: float = 1.0,
              conf: dict | None = None, adjustments: dict | None = None) -> np.ndarray:
    """What the screen shows: the view combined with colour, as uint8 RGB.

    The classic ``composite_final_image``, moved.  *view_mode* "Shading" is
    the grey view, "Combined Colour" relights the albedo, "Roughness Fusion"
    mixes in roughness.  *conf* (keys mask, overlay, shadow, highlight,
    colour, opacity, blend) is the confidence overlay; *adjustments* the
    global sliders (global_dehaze, global_whites, global_blacks,
    global_midtone, global_clarity, global_texture).
    """
    if getattr(data, 'albedo_rgb_full', None) is None:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if use_proxy:
        albedo = data.albedo_proxy
        if albedo is not None and albedo.ndim >= 2:     # the proxy's real size
            h, w = albedo.shape[:2]
        rough = data.roughness_proxy if data.roughness_proxy is not None else \
            np.full((h, w), 128, dtype=np.uint8)
    else:
        x1, y1, x2, y2 = crop_coords if crop_coords else (0, 0, data.w, data.h)
        albedo = data.albedo_rgb_full[y1:y2, x1:x2]
        if data.roughness_full is not None:
            rough = data.roughness_full[y1:y2, x1:x2]
        else:
            rough = np.full((y2 - y1, x2 - x1), 128, dtype=np.uint8)

    # The shading's size is the output size, whatever h, w say.
    sh, sw = shading.shape[:2]
    if albedo is None or albedo.size == 0 or albedo.shape[0] == 0 or albedo.shape[1] == 0:
        return np.zeros((sh, sw, 3), dtype=np.uint8)
    if rough is None or rough.size == 0 or rough.shape[0] == 0 or rough.shape[1] == 0:
        rough = np.full((sh, sw), 128, dtype=np.uint8)
    if albedo.shape[:2] != (sh, sw):
        albedo = cv2.resize(albedo, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if rough.shape[:2] != (sh, sw):
        rough = cv2.resize(rough, (sw, sh), interpolation=cv2.INTER_LINEAR)

    # False-colour engines (Colour ICA and the like) are already RGB.
    if shading.ndim == 3 and shading.shape[2] == 3:
        out_rgb = np.empty(shading.shape, np.uint8)
        map_rows(lambda s: np.clip(s * 255.0, 0, 255).astype(np.uint8), shading, out_rgb)
        if out_rgb.shape[:2] != (sh, sw):
            out_rgb = cv2.resize(out_rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        return out_rgb

    out = _tone_rows(shading, albedo, rough, settings, view_mode, colour_opacity, sh, sw,
                     colour_gain if view_mode == "Combined Colour" else 1.0)

    # Engines whose picture is their data, not a shading.
    if engine == "False-Colour Depth" and getattr(data, 'depth_vis_p', None) is not None:
        src = data.depth_proxy_p if use_proxy else data.depth_vis_p
        if crop_coords and not use_proxy:
            x1c, y1c, x2c, y2c = crop_coords
            src = src[y1c:y2c, x1c:x2c]
        d8 = cv2.normalize(cv2.resize(src, (w, h)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        import matplotlib.cm as _cm
        cmap_fn = getattr(_cm, settings.get('depth_cmap', 'viridis'), _cm.viridis)
        out = (cmap_fn(d8.astype(np.float32) / 255.0) * 255).astype(np.uint8)[:, :, :3]
    elif engine == "Normals RGB" and getattr(data, 'normals_full', None) is not None:
        src = data.normals_proxy if use_proxy else data.normals_full
        if crop_coords and not use_proxy:
            x1c, y1c, x2c, y2c = crop_coords
            src = src[y1c:y2c, x1c:x2c]
        n = cv2.resize(src.astype(np.float32), (w, h))
        out = np.clip((n * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)

    conf = conf or {}
    needs_conf = (view_mode == "Confidence Overlay") or conf.get("mask", False)
    if needs_conf and getattr(data, 'albedo_grey_full', None) is not None:
        shadow_cut, highlight_cut = conf.get("shadow", 0.1), conf.get("highlight", 0.9)
        if use_proxy:
            grey_src = cv2.cvtColor(data.albedo_proxy, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            grey_src = data.albedo_grey_full[y1:y2, x1:x2].astype(np.float32)
        conf_raw = np.where((grey_src >= shadow_cut) & (grey_src <= highlight_cut),
                            1.0, 0.0).astype(np.float32)
        conf_mask = cv2.GaussianBlur(conf_raw, (7, 7), 0)
        if view_mode == "Confidence Overlay":
            overlay = np.full_like(out, CONF_COLOURS.get(conf.get("colour", "Red"), [255, 0, 0]))
            alpha = (1.0 - conf_mask)[..., None] * conf.get("opacity", 1.0)
            blend = conf.get("blend", "Overlay")
            if blend == "Overlay":
                out = (out * (1.0 - alpha) + overlay * alpha).astype(np.uint8)
            elif blend == "Multiply":
                out = (out * (1.0 - alpha) + (out * overlay / 255.0) * alpha).astype(np.uint8)
            else:
                out = np.clip(out + overlay * alpha, 0, 255).astype(np.uint8)
        if conf.get("mask", False):
            cm = conf_mask
            if cm.shape[:2] != out.shape[:2]:
                cm = cv2.resize(cm.astype(np.float32), (out.shape[1], out.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            out = np.clip(out.astype(np.float32) * cm[..., None], 0, 255).astype(np.uint8)

    # Global image adjustments (the 3D tab's sliders).
    adj = adjustments or {}
    dehaze, whites = adj.get('global_dehaze', 0.0), adj.get('global_whites', 1.0)
    blacks, midtone = adj.get('global_blacks', 0.0), adj.get('global_midtone', 1.0)
    clarity, texture = adj.get('global_clarity', 0.0), adj.get('global_texture', 0.0)
    if any(abs(v - d) > 0.01 for v, d in [(dehaze, 0), (whites, 1), (blacks, 0), (midtone, 1),
                                          (clarity, 0), (texture, 0)]):
        f = out.astype(np.float32) / 255.0
        if dehaze > 0.01:
            p_lo, p_hi = float(np.percentile(f, 1)), float(np.percentile(f, 99))
            if p_hi - p_lo > 0.01:
                f = f * (1 - dehaze) + np.clip((f - p_lo) / (p_hi - p_lo), 0, 1) * dehaze
        if abs(whites - 1) > 0.01 or blacks > 0.01:
            f = np.clip(f * whites + blacks, 0, 1)
        if abs(midtone - 1) > 0.01:
            f = np.clip(f ** (1.0 / max(0.1, midtone)), 0, 1)
        for strength, sig_pct in [(clarity, 0.08), (texture, 0.015)]:
            if strength > 0.01:
                sig = max(0.5, f.shape[0] * sig_pct)
                if f.ndim == 3:
                    for c in range(3):
                        b = cv2.GaussianBlur(f[..., c], (0, 0), sig)
                        f[..., c] = np.clip(f[..., c] + (f[..., c] - b) * strength, 0, 1)
                else:
                    b = cv2.GaussianBlur(f, (0, 0), sig)
                    f = np.clip(f + (f - b) * strength, 0, 1)
        out = np.clip(f * 255, 0, 255).astype(np.uint8)
    return out


def to_display(img) -> np.ndarray:
    """An engine's output as uint8 RGB for the screen.

    Engines return 0..1 floats (most) or 0..255 (a few), grey or colour; the
    range decides the scale, as the golden-image tool does."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        finite = a[np.isfinite(a)]
        hi = float(finite.max()) if finite.size else 0.0
        if hi <= 1.0 + 1e-6:
            a = a * 255.0
        a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    return np.ascontiguousarray(a[..., :3])


MATTE_VIEW = "Matte Relight (as 3D)"


def matte_colour(data, settings: dict, colour_gain: float = 1.0, use_proxy: bool = False) -> np.ndarray:
    """The colour as the Matte Relight view draws it (uint8 RGB): the photographs'
    colour lit as the 3D relief lights it, at *settings*' light and with the Colour
    layer's brightness *colour_gain*.  Publish's "Colour in the exports" puts it in
    the albedo's place, so every export carries the colour as seen on screen."""
    if use_proxy:
        h, w = np.asarray(data.albedo_proxy).shape[:2]
    else:
        h, w = data.h, data.w
    shading = render(MATTE_VIEW, data, dict(settings), h, w, use_proxy)
    sh, sw = np.asarray(shading).shape[:2]
    return composite(MATTE_VIEW, shading, data, dict(settings), sh, sw, use_proxy,
                     view_mode="Combined Colour", colour_gain=float(colour_gain))
