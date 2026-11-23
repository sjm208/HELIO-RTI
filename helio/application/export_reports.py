"""Report and figure exports, GUI-free (REP-002…006).

The classic publication sheet (Generate Publication Sheet), publication
figures, condition report and change detection, moved so both apps write
the same figures.  Renders come in through a draw callable.

Fixed on the way: the sheet's panel picker numbered its panels differently
from the panels themselves, so eight of them (Curvature, MSRM, Image
Quotient, SLRM, Light Coverage, Local Dominance, Shape Index, False-Colour
Depth) could never appear (KD-040); its 3D raking renders were lit from the
mirrored side (KD-041); the condition report called uncalibrated heights
micrometres and drew them with the rainbow "jet" map (KD-042).
"""
from __future__ import annotations

import logging
import math
import multiprocessing
import os
import time
from typing import Callable

import cv2
import numpy as np

_log = logging.getLogger(__name__)

# The sheet's panels, in order: what a panel picker offers.
SHEET_PANELS = (
    "1. Light Dome", "2. Light Vec. Dist.", "3. Pure Colour", "4. Normals", "5. Depth",
    "6. Roughness", "7. Specular Variance", "8. Ink Threshold", "9. PCA", "10. Heatmap",
    "11. Standard PBR", "12. Robust PBR", "13. Primary Hillshade", "14. Multi-Dir Hillshade",
    "15. SVF", "16. Slope Gradient", "17. PTM Base", "18. Pos Openness", "19. Neg Openness",
    "20. Curvature", "21. MSRM", "22. Image Quotient", "23. SLRM", "24. Light Coverage",
    "25. Local Dominance", "26. Shape Index", "27. False-Colour Depth",
    "T1. Pure Openness", "T2. MDH 8-dir", "T3. Curvature", "T5. Neg Openness",
    "T6. Sky View Factor", "T7. Hillshade (raking)", "T8. MSRM",
    "F1. Colour ICA", "F2. RGB-PS R-B Diff", "F3. Normal Residual", "F4. Raking Composite",
    "F5. Diffuse Gain", "F6. Slope Gradient")


# The terrain (T) and forensic (F) panels that only repeat a numbered one: a
# sheet with everything selected shows each picture once.
SHEET_REPEATS = ("T1. Pure Openness", "T2. MDH 8-dir", "T3. Curvature", "T5. Neg Openness",
                 "T6. Sky View Factor", "T7. Hillshade (raking)", "T8. MSRM", "F6. Slope Gradient")
SHEET_DEFAULT = tuple(t for t in SHEET_PANELS if t not in SHEET_REPEATS)


def _pkey(title: str) -> str:
    """A panel's name without its number, so a picker numbered differently still matches."""
    t = str(title)
    return t.split(". ", 1)[1].strip().lower() if ". " in t else t.strip().lower()


SHEET_DEFAULTS = {"fmt": "jpg", "white": False, "depth": True, "normals": True, "scale_bar": True,
                  "metadata": True, "capture_meta": True, "histogram": False, "metrology": True,
                  "shots": False, "dpi": 300}


def publication_sheet(D, filepath: str, *, draw: Callable, selected, options: dict | None = None,
                      crop=None, engine_settings: dict | None = None, lights=None, ss: dict,
                      n_images: int = 0, shots=(), capture_meta: dict | None = None, profile=None,
                      mm_per_px=None, z_mm_per_unit=None, ppi=None, scale_val: float = 0.0,
                      obj: str = "", engine: str = "", progress: Callable | None = None) -> dict:
    """The publication sheet: the selected panels, a profile, raking renders,
    3D renders, metrology, screenshots and capture metadata, as one figure.

    *draw(engine, force_light)* renders a view at preview size (the light
    straight down when *force_light*).  Returns {"path", "panels"}."""
    from matplotlib.colors import LightSource
    from helio.io import headless_render_worker
    progress = progress or (lambda *_a: None)
    o = dict(SHEET_DEFAULTS, **(options or {}))
    fmt = o["fmt"]
    bg_col = 'white' if o["white"] else '#111111'
    txt_col = 'black' if o["white"] else 'white'
    opt_depth, opt_normals, opt_scale_bar = o["depth"], o["normals"], o["scale_bar"]
    opt_metadata, opt_capture_meta = o["metadata"], o["capture_meta"]
    opt_histogram, opt_metrology, opt_3d_shots = o["histogram"], o["metrology"], o["shots"]
    dpi_out = int(o["dpi"])
    opt_scale_val, opt_obj_str, opt_engine_str = float(scale_val or 0.0), obj, engine or "Standard PBR"
    engine_settings = engine_settings or {}
    capture_meta = capture_meta or {}
    use_crop = crop is not None
    cx1, cy1, cx2, cy2 = crop if use_crop else (0, 0, D.w, D.h)
    _ph, _pw = cy2 - cy1, cx2 - cx1
    _selected_keys = {_pkey(t) for t in selected}
    # Use Agg backend inside the thread — plt.figure() must NOT use
    # a GUI backend (TkAgg) from a background thread; it will hang/crash.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as _plt_agg
    import matplotlib.pyplot as plt  # rebind module-level plt for this scope
    progress(("Setting up..."))
    print("[Publisher] Assembling Publication Graphic...")

    def _c(arr):
        if arr is None: return None
        return arr[cy1:cy2, cx1:cx2] if arr.ndim >= 2 else arr

    def add_panel(ax, img, title_str, use_color=False):
        # Downsample to max 1200px before passing to imshow —
        # matplotlib only renders subplots at ~1800px at 300 DPI/6-inch
        # so passing full 11574px images wastes 40× memory and time.
        if img is not None:
            _h, _w = img.shape[:2]
            _cap = 1800
            if max(_h, _w) > _cap:
                _sc = _cap / max(_h, _w)
                _nh, _nw = max(1, int(_h*_sc)), max(1, int(_w*_sc))
                img = cv2.resize(img, (_nw, _nh), interpolation=cv2.INTER_AREA)
        ax.imshow(img, cmap='gray' if img is not None and (img.ndim==2 or not use_color) else None)
        ax.set_title(title_str, fontsize=16, fontweight='bold', color=txt_col)
        ax.axis('off'); ax.set_facecolor(bg_col)

    # ── Static panel arrays ────────────────────────────────────────────
    progress(("Building static panels..."))

    def _cap_img(arr, cap=1800):
        """Downsample to max cap px — panels at 6.5in/300DPI need ~1950px."""
        if arr is None: return None
        _h, _w = arr.shape[:2]
        if max(_h, _w) <= cap: return arr
        _sc = cap / max(_h, _w)
        return cv2.resize(arr, (max(1,int(_w*_sc)), max(1,int(_h*_sc))),
                          interpolation=cv2.INTER_AREA)

    norm_vis      = _cap_img((((_c(D.normals_full) + 1.0) / 2.0) * 255).astype(np.uint8))
    pca_settings  = engine_settings.get("Standard PBR")
    if not pca_settings:
        from helio.domain.registry import default_engine_settings
        pca_settings = default_engine_settings()["Standard PBR"]
    pca_res       = np.clip((_c(D.pca_full).astype(float) * pca_settings['bright'] - 128) * (1 + pca_settings['contrast']/100) + 128, 0, 255).astype(np.uint8)
    if pca_settings.get('gamma', 1.0) != 1.0:
        pca_res = cv2.LUT(pca_res, np.array([((i/255.0)**(1.0/pca_settings['gamma']))*255 for i in np.arange(256)]).astype("uint8"))
    _ag = _c(D.albedo_grey_full)
    ao_8bit = cv2.adaptiveThreshold(
        _ag.astype(np.uint8) if _ag is not None else np.zeros((_ph,_pw),np.uint8),
        255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    heat  = _cap_img(cv2.cvtColor(cv2.applyColorMap(
        (np.clip(_c(D.depth_vis_p), 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB))

    # ── Engine render helpers — use proxy for speed ───────────────────
    # Panels need only ~1200px at 300 DPI/6-inch subplots.
    # Proxy (~1380px) is 50-100× faster than full-res per engine.
    _render_memo = {}
    def get_render(eng, force_light=False):
        key = (eng, force_light)
        if key in _render_memo: return _render_memo[key]
        try:
            # The caller renders at preview size, with the light straight
            # down when force_light (the classic swapped it in and back).
            _render_memo[key] = _cap_img(draw(eng, force_light))
            return _render_memo[key]
        except Exception as _e:
            print(f"[Publisher] {eng}: {_e}")
            return np.zeros((D.ph, D.pw, 3), dtype=np.uint8)

    def gr_gray(eng, f_l=False):
        try: return cv2.cvtColor(get_render(eng, f_l), cv2.COLOR_RGB2GRAY)
        except Exception:
            return np.zeros((D.ph, D.pw), dtype=np.uint8)
    def gr_color(eng, f_l=False):
        try: return get_render(eng, f_l)
        except Exception:
            return np.zeros((D.ph, D.pw, 3), dtype=np.uint8)

    spec_var_norm = cv2.normalize(_c(D.spec_var), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # ── LAZY panel definitions — only renders selected panels ──────────
    # Each panel has a thunk (zero-arg lambda) that is only called if
    # the panel title is in _selected_set. Deselecting 20 panels means
    # those 20 engines are never computed at all.
    #
    # The engine rendering thunks use get_shading() which internally
    # caches results — if two panels share an engine (e.g. Hillshade
    # appears as both panel 13 and T7) the second call returns instantly.

    def _slrm_render():
        _dp2 = _c(D.depth_vis_p).astype(np.float32)
        _sig = max(3.0, min(_ph, _pw) * 0.02)
        _s2 = _dp2 - cv2.GaussianBlur(_dp2,(0,0),sigmaX=_sig)
        _p2,_p98 = float(np.percentile(_s2,2)), float(np.percentile(_s2,98))
        return (np.clip((_s2-_p2)/max(_p98-_p2,1e-8),0,1)*255).astype(np.uint8)

    def _ptm_render():
        if getattr(D, "ptm_deferred", False):
            D.ensure_ptm_hsh()
        return cv2.normalize(
            D.ptm_coeffs[0].reshape(D.h,D.w)[cy1:cy2,cx1:cx2],
            None,0,255,cv2.NORM_MINMAX).astype(np.uint8) if D.ptm_coeffs is not None else np.zeros((_ph,_pw),np.uint8)

    # Panel catalogue: (title, colour, thunk_or_value)
    # thunk = callable → evaluated lazily when panel is selected
    # value = array   → already computed (cheap, static data)
    _panel_defs = [
        ("1. Light Dome",           False, None),
        ("2. Light Vec. Dist.",      False, None),
        ("3. Pure Colour",           True,  lambda: _c(D.albedo_rgb_full)),
        ("4. Normals",               True,  lambda: norm_vis if opt_normals else None),
        ("5. Depth",                 False, lambda: _c(D.depth_vis_p) if opt_depth else None),
        ("6. Roughness",             False, lambda: _c(D.roughness_full)),
        ("7. Specular Variance",     False, lambda: spec_var_norm),
        ("8. Ink Threshold",               False, lambda: ao_8bit),
        ("9. PCA",                   True,  lambda: pca_res),
        ("10. Heatmap",              True,  lambda: heat),
        ("11. Standard PBR",         True,  lambda: (progress(("Rendering Standard PBR...")), get_render("Standard PBR"))[1]),
        ("12. Robust PBR",           True,  lambda: (progress(("Rendering Robust PBR...")), get_render("Robust PBR (MVS)"))[1]),
        ("13. Primary Hillshade",    False, lambda: (progress(("Rendering Hillshade...")), gr_gray("PS Normal-Map Lit (Single)"))[1]),
        ("14. Multi-Dir Hillshade",  False, lambda: (progress(("Rendering MDH...")), gr_gray("PS Normal-Map Lit (8-Dir)"))[1]),
        ("15. SVF",                  False, lambda: (progress(("Rendering SVF...")), gr_gray("Sky-View Factor (SVF)"))[1]),
        ("16. Slope Gradient",       False, lambda: (progress(("Rendering Slope...")), gr_gray("Slope Gradient"))[1]),
        ("17. PTM Base",             False, _ptm_render),
        ("18. Pos Openness",         False, lambda: (progress(("Rendering Pos Openness...")), gr_gray("Positive Openness"))[1]),
        ("19. Neg Openness",         False, lambda: (progress(("Rendering Neg Openness...")), gr_gray("Negative Openness"))[1]),
        ("20. Curvature",            False, lambda: (progress(("Rendering Curvature...")), gr_gray("Surface Curvature (Scratches)"))[1]),
        ("21. MSRM",                 False, lambda: (progress(("Rendering MSRM...")), gr_gray("Multi-Scale Relief Model (MSRM)"))[1]),
        ("22. Image Quotient",       False, lambda: (progress(("Rendering Image Quotient...")), gr_gray("Image Quotient (IQ)",True))[1]),
        ("23. SLRM",                 False, _slrm_render),
        ("24. Light Coverage",       False, lambda: (progress(("Rendering Light Coverage...")), gr_gray("Light Coverage Map"))[1]),
        ("25. Local Dominance",      False, lambda: (progress(("Rendering Local Dominance...")), gr_gray("Local Dominance"))[1]),
        ("26. Shape Index",          False, lambda: (progress(("Rendering Shape Index...")), gr_gray("Shape Index"))[1]),
        ("27. False-Colour Depth",   True,  lambda: (progress(("Rendering False-Colour Depth...")), gr_color("False-Colour Depth"))[1]),
        ("T1. Pure Openness",        False, lambda: gr_gray("Positive Openness")),
        ("T2. MDH 8-dir",            False, lambda: gr_gray("PS Normal-Map Lit (8-Dir)")),
        ("T3. Curvature",            False, lambda: gr_gray("Surface Curvature (Scratches)")),
        ("T5. Neg Openness",         False, lambda: gr_gray("Negative Openness")),
        ("T6. Sky View Factor",      False, lambda: gr_gray("Sky-View Factor (SVF)")),
        ("T7. Hillshade (raking)",   False, lambda: gr_gray("PS Normal-Map Lit (Single)")),
        ("T8. MSRM",                 False, lambda: gr_gray("MSRM (Multi-Scale Relief)") if "MSRM (Multi-Scale Relief)" in engine_settings else gr_gray("Multi-Scale Relief Model (MSRM)")),
        ("F1. Colour ICA",           True,  lambda: (progress(("Rendering Colour ICA...")),
                                                      gr_color("Colour ICA (IC1-3)"))[1]),
        ("F2. RGB-PS R-B Diff",      False, lambda: (progress(("Rendering RGB-PS R-B Diff...")),
                                                      gr_gray("RGB-PS R-B Diff"))[1]),
        ("F3. Normal Residual",      False, lambda: (progress(("Rendering Normal Residual...")),
                                                      gr_gray("Normal Residual"))[1]),
        ("F4. Raking Composite",     False, lambda: (progress(("Rendering Raking Composite...")),
                                                      gr_gray("Raking Composite"))[1]),
        ("F5. Diffuse Gain",         True,  lambda: (progress(("Rendering Diffuse Gain...")),
                                                      gr_color("Diffuse Gain"))[1]),
        ("F6. Slope Gradient",       False, lambda: (progress(("Rendering Slope Gradient...")),
                                                      gr_gray("Slope Gradient"))[1]),
    ]

    # ── Evaluate only selected panels ─────────────────────────────────
    _n_sel = sum(1 for t,_,_ in _panel_defs if _pkey(t) in _selected_keys)
    progress((f"Computing {_n_sel} selected panels..."))
    panels = []
    for title, use_color, thunk in _panel_defs:
        if _pkey(title) not in _selected_keys: continue
        if title in ("1. Light Dome", "2. Light Vec. Dist."):
            panels.append((None, title, use_color)); continue
        try:
            img = thunk() if callable(thunk) else thunk
        except Exception as _pe:
            print(f"[Publisher] panel '{title}' failed: {_pe}")
            img = None
        if img is not None:
            panels.append((img, title, use_color))

    n_extra_rows = 1 if opt_histogram else 0
    rows = max(1, math.ceil(len(panels) / 4))
    _met_enabled  = opt_metrology
    _pub_shots    = list(shots or []) if opt_3d_shots else []
    _shot_rows    = math.ceil(len(_pub_shots) / 2) * 2 if _pub_shots else 0
    _persp_3d_rows = 2          # crop perspective 3D (always)
    _full_3d_rows  = 2 if use_crop else 0  # full-image perspective 3D
    # Capture metadata panel needs 1 row; +2 offset means we need 3 extra rows
    _capture_meta_rows = 3 if (opt_capture_meta
                                and capture_meta) else 0
    base_rows = rows + 2 + n_extra_rows + 2 + _persp_3d_rows + _full_3d_rows + _shot_rows + _capture_meta_rows

    # Figure height cap: prevent multi-hour saves on huge figures.
    # At 300 DPI, a 24×120 in figure is 7200×36000 px — extremely slow.
    _PANEL_H_IN = 6.5
    _fig_h_raw = max(14, base_rows * _PANEL_H_IN)
    _fig_h_cap = 120  # max inches — raised so panels stay legible
    if _fig_h_raw > _fig_h_cap:
        print(f"[Publisher] Capping figure height {_fig_h_raw:.0f}→{_fig_h_cap} in")
        _fig_h_raw = _fig_h_cap
    # If tall figure at high DPI, reduce DPI to keep save time tolerable
    _fig_dpi = dpi_out
    if _fig_h_raw > 90 and dpi_out > 150:
        _fig_dpi = 150
        print(f"[Publisher] Tall figure ({_fig_h_raw:.0f}in) — reducing DPI {dpi_out}→150")
    elif _fig_h_raw > 110 and dpi_out > 100:
        _fig_dpi = 100
        print(f"[Publisher] Very tall figure ({_fig_h_raw:.0f}in) — reducing DPI {dpi_out}→100")

    progress((f"Drawing {len(panels)} panels..."))
    fig = plt.figure(figsize=(24, _fig_h_raw), dpi=_fig_dpi)
    fig.patch.set_facecolor(bg_col)
    gs = fig.add_gridspec(base_rows, 4)

    for i, (img, t, c) in enumerate(panels):
        if "Light Dome" in t:
            ax = fig.add_subplot(gs[i//4, i%4], projection='3d')
            if lights is not None:
                ax.scatter(lights[:,0], lights[:,1],
                           lights[:,2], c=lights[:,2],
                           cmap='plasma', s=100, edgecolors='black')
            ax.set_zlim(0,1); ax.set_title(t, fontsize=16, fontweight='bold', color=txt_col)
            ax.set_facecolor(bg_col)
        elif "Light Vec. Dist." in t:
            ax = fig.add_subplot(gs[i//4, i%4]); ax.set_facecolor(bg_col)
            if lights is not None:
                sc = ax.scatter(lights[:,0], lights[:,1],
                                c=lights[:,2], cmap='plasma', s=80,
                                edgecolors='gray', alpha=0.9, linewidths=0.5)
                theta = np.linspace(0, 2*np.pi, 200)
                ax.plot(np.cos(theta), np.sin(theta), color='gray', linewidth=0.8, alpha=0.5)
                plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='Elevation (Z)')
            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.set_aspect('equal')
            ax.set_xlabel('X', color=txt_col, fontsize=9); ax.set_ylabel('Y', color=txt_col, fontsize=9)
            ax.tick_params(colors=txt_col, labelsize=7)
            [s.set_edgecolor(txt_col) for s in ax.spines.values()]
            ax.set_title(t, fontsize=16, fontweight='bold', color=txt_col)
        else:
            add_panel(fig.add_subplot(gs[i//4, i%4]), img, t, c)

    # ── Profile reference image ────────────────────────────────────────
    ax21 = fig.add_subplot(gs[rows, 1:3])
    alb_display = _c(D.albedo_rgb_full).copy()
    scale_val = opt_scale_val
    _rh, _rw = alb_display.shape[:2]
    if opt_scale_bar and ppi:
        ppi    = ppi
        _mm_pp = (mm_per_px or 1.0 / ppi * 25.4)
        _img_mm = _mm_pp * _rw
        bar_mm  = 10 if _img_mm>=40 else (5 if _img_mm>=20 else (2 if _img_mm>=8 else 1))
        bar_px  = int(bar_mm / _mm_pp)
        bar_frac = max(0.005, min(0.4, bar_px / max(_rw, 1)))
        ax21.set_title(f"Profile Reference  |  Scale bar = {bar_mm} mm  |  PPI: {ppi:.0f}",
                       fontsize=16, fontweight='bold', color=txt_col)
    ax21.imshow(alb_display)
    _trim_pub = max(5, int(_rw * (ss['edge_trim'] / 100.0)))
    ax21.axhline(y=_rh * 0.5, color='#FFD700', linewidth=2.5, linestyle='-', alpha=0.85, zorder=10)
    ax21.axvline(x=_trim_pub, color='#FFD700', linewidth=1.2, linestyle=':', alpha=0.55, zorder=10)
    ax21.axvline(x=_rw-_trim_pub, color='#FFD700', linewidth=1.2, linestyle=':', alpha=0.55, zorder=10)
    import matplotlib.patches as _mp
    if opt_scale_bar and ppi:
        _sbx,_sby,_sbh = 0.05,0.06,0.025
        ax21.add_patch(_mp.FancyBboxPatch((_sbx-.006,_sby-.006),bar_frac+.012,_sbh+.012,
            boxstyle='square,pad=0',transform=ax21.transAxes,facecolor='black',alpha=.55,zorder=9))
        ax21.add_patch(_mp.Rectangle((_sbx,_sby),bar_frac,_sbh,
            transform=ax21.transAxes,facecolor='white',edgecolor='#444',linewidth=1.2,zorder=10))
        ax21.text(_sbx+bar_frac/2,_sby+_sbh+.013,f"{bar_mm} mm",
            transform=ax21.transAxes,ha='center',va='bottom',color='white',
            fontsize=11,fontweight='bold',zorder=11,
            bbox=dict(facecolor='black',alpha=.55,boxstyle='round,pad=1.5',edgecolor='none'))
    if not (opt_scale_bar and scale_val > 0):
        ax21.axhline(y=_rh//2, color='cyan', linewidth=4, linestyle='--')
    ax21.axis('off'); ax21.set_facecolor(bg_col)

    # ── Profile graph ─────────────────────────────────────────────────
    ax22 = fig.add_subplot(gs[rows+1, :])
    if profile:
        _cpd = profile
        x_axis,plot_z,p_title = _cpd[0],_cpd[1],_cpd[2]
        x_label = _cpd[3] if len(_cpd)>3 else "Distance"
        z_unit  = _cpd[4] if len(_cpd)>4 else "rel"
        _pstats = _cpd[5:10] if len(_cpd)>5 else None
        z_lbl   = f"Elevation ({z_unit})" if z_unit!="rel" else "Elevation (relative)"
        ax22.fill_between(x_axis, plot_z, float(np.min(plot_z)), color='#1a5a8a', alpha=0.5)
        ax22.plot(x_axis, plot_z, color='#4fc3f7', linewidth=1.8)
        ax22.axhline(float(np.mean(plot_z)), color='#ffb300', linewidth=1, linestyle='--', alpha=0.8)
        _im = int(np.argmax(plot_z)); _in = int(np.argmin(plot_z))
        ax22.annotate(f"{plot_z[_im]:.3f}", xy=(x_axis[_im],plot_z[_im]), xytext=(0,7),
            textcoords='offset points', color='#80ff80', fontsize=9, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->',color='#80ff80',lw=0.8))
        ax22.annotate(f"{plot_z[_in]:.3f}", xy=(x_axis[_in],plot_z[_in]), xytext=(0,-14),
            textcoords='offset points', color='#ff8080', fontsize=9, ha='center', fontweight='bold',
            arrowprops=dict(arrowstyle='->',color='#ff8080',lw=0.8))
        if _pstats:
            Ra,Rq,Rz,Rp,Rv = _pstats
            ax22.text(0.99,0.97,f"Ra={Ra:.3f}  Rq={Rq:.3f}  Rz={Rz:.3f} {z_unit}",
                transform=ax22.transAxes,ha='right',va='top',fontsize=11,color=txt_col,
                bbox=dict(facecolor='#0a1020' if bg_col!='white' else '#f0f4f8',alpha=0.8,edgecolor='#334455',boxstyle='round,pad=0.4'))
        ax22.set_xlabel(x_label, fontsize=14, color=txt_col)
        ax22.set_ylabel(z_lbl, fontsize=14, color=txt_col)
        ax22.set_title(p_title, fontsize=16, fontweight='bold', color=txt_col)
    else:
        from helio.application.heights import height_px
        _depth_crop = _c(height_px(D)) if z_mm_per_unit else _c(D.depth_vis_p)   # true heights for mm
        _dch, _dcw = _depth_crop.shape[:2]
        trim = max(5, int(_dcw * (ss['edge_trim'] / 100.0)))
        z_line = _depth_crop[_dch//2, trim:-trim] if trim < _dcw//2 else _depth_crop[_dch//2, :]
        n_pts  = len(z_line)
        _mm_pp = mm_per_px
        _z_mpu = z_mm_per_unit
        x_axis = np.linspace(0, n_pts*_mm_pp, n_pts) if _mm_pp else np.linspace(0, n_pts, n_pts)
        x_lbl  = "Distance (mm)" if _mm_pp else "Distance (px)"
        if _z_mpu:
            z_prof = (z_line.astype(np.float32) - float(np.min(z_line))) * _z_mpu
            if float(np.max(z_prof)-np.min(z_prof)) < 0.5:
                z_prof *= 1000; z_lbl="Elevation (µm)"; _zu="µm"
            else:
                z_lbl="Elevation (mm)"; _zu="mm"
        else:
            z_prof = z_line.astype(np.float32) - float(np.min(z_line))
            z_lbl="Elevation (relative — calibrate Z for mm)"; _zu="rel"
        zm = float(np.mean(z_prof))
        Ra_a=float(np.mean(np.abs(z_prof-zm))); Rq_a=float(np.sqrt(np.mean((z_prof-zm)**2)))
        Rz_a=float(np.max(z_prof)-np.min(z_prof))
        ax22.fill_between(x_axis, z_prof, float(np.min(z_prof)), color='#1a5a8a', alpha=0.5)
        ax22.plot(x_axis, z_prof, color='#4fc3f7', linewidth=1.8)
        ax22.axhline(zm, color='#ffb300', linewidth=1, linestyle='--', alpha=0.8)
        _im2=int(np.argmax(z_prof)); _in2=int(np.argmin(z_prof))
        ax22.annotate(f"{z_prof[_im2]:.3f}",xy=(x_axis[_im2],z_prof[_im2]),xytext=(0,7),
            textcoords='offset points',color='#80ff80',fontsize=9,ha='center',fontweight='bold',
            arrowprops=dict(arrowstyle='->',color='#80ff80',lw=0.8))
        ax22.annotate(f"{z_prof[_in2]:.3f}",xy=(x_axis[_in2],z_prof[_in2]),xytext=(0,-14),
            textcoords='offset points',color='#ff8080',fontsize=9,ha='center',fontweight='bold',
            arrowprops=dict(arrowstyle='->',color='#ff8080',lw=0.8))
        ax22.text(0.99,0.97,f"Ra={Ra_a:.3f}  Rq={Rq_a:.3f}  Rz={Rz_a:.3f} {_zu}",
            transform=ax22.transAxes,ha='right',va='top',fontsize=10,color=txt_col,
            bbox=dict(facecolor='#0a1020' if bg_col!='white' else '#f0f4f8',alpha=0.8,edgecolor='#334455',boxstyle='round,pad=0.4'))
        ax22.set_xlabel(x_lbl, fontsize=14, color=txt_col)
        ax22.set_ylabel(z_lbl, fontsize=14, color=txt_col)
        ax22.set_title("Cross-Section Profile (Auto Centre-Line)", fontsize=16, fontweight='bold', color=txt_col)
    ax22.grid(True, color='#444' if bg_col!='white' else '#ddd')
    ax22.set_facecolor(bg_col if bg_col=='white' else '#1a1a1a')
    ax22.tick_params(colors=txt_col, labelcolor=txt_col)
    [sp.set_edgecolor(txt_col) for sp in ax22.spines.values()]

    extra_row_idx = rows + 2

    # ── Histogram (optional) ──────────────────────────────────────────
    if opt_histogram:
        ax_hist = fig.add_subplot(gs[extra_row_idx, :])
        ax_hist.set_facecolor(bg_col if bg_col=='white' else '#1a1a1a')
        try:
            alb_f = _c(D.albedo_rgb_full).astype(np.float32)
            for ch,col,lbl in zip(range(3),['#e74c3c','#2ecc71','#3498db'],['Red','Green','Blue']):
                hist,bins = np.histogram(alb_f[:,:,ch].ravel(), bins=256, range=(0,255))
                ax_hist.plot(bins[:-1], hist, color=col, alpha=0.7, linewidth=1.2, label=lbl)
            lum = cv2.cvtColor(_c(D.albedo_rgb_full), cv2.COLOR_RGB2GRAY).astype(np.float32)
            hist_l,bins_l = np.histogram(lum.ravel(), bins=256, range=(0,255))
            ax_hist.plot(bins_l[:-1], hist_l,
                         color='white' if bg_col!='white' else 'black',
                         linewidth=1.8, label='Luminance', linestyle='--')
            ax_hist.legend(fontsize=10, framealpha=0.3)
            ax_hist.set_xlabel("Pixel Value (0–255)", fontsize=13, color=txt_col)
            ax_hist.set_ylabel("Count", fontsize=13, color=txt_col)
            ax_hist.tick_params(colors=txt_col)
            [s.set_edgecolor(txt_col) for s in ax_hist.spines.values()]
        except Exception as he:
            ax_hist.text(0.5,0.5,f"Histogram error: {he}",transform=ax_hist.transAxes,
                         ha='center',va='center',color=txt_col)
        ax_hist.set_title("Albedo RGB + Luminance Histogram", fontsize=16, fontweight='bold', color=txt_col)
        extra_row_idx += 1

    # ── Raking light renders ─────────────────────────────────────────────
    # Scale depth to match V96's headless render approach:
    # depth_vis_p is [0,1]; scale by height_scale*0.15 so vert_exag
    # produces a strong, visible raking effect — same as the 3D viewer.
    progress(("Rendering raking light views..."))
    _3d_label   = " (Crop)" if use_crop else ""
    _height_sc  = ss['height'] * 0.15
    # Downsample depth/albedo/normals for raking renders — 1200px is sufficient
    _RAKE_CAP = 1200
    _rsc = min(_RAKE_CAP / max(1, _ph), _RAKE_CAP / max(1, _pw))
    _rh2, _rw2 = max(1, int(_ph*_rsc)), max(1, int(_pw*_rsc))
    _depth_raw  = cv2.resize(_c(D.depth_vis_p).astype(np.float32), (_rw2, _rh2))
    _rake_z     = _depth_raw * _height_sc
    _rake_rgb   = cv2.resize(_c(D.albedo_rgb_full), (_rw2, _rh2)).astype(np.float64) / 255.0

    # 2D Raking panels — same LightSource.shade_rgb() as V96 + 3D renders
    ls_sw   = LightSource(azdeg=225, altdeg=45)
    ls_nw   = LightSource(azdeg=315, altdeg=45)
    # Colour raking: shade_rgb blends albedo with hillshade in overlay mode
    _col_rgb = ls_sw.shade_rgb(_rake_rgb.astype(np.float32),
                               _rake_z, vert_exag=3.0, blend_mode='overlay')
    _col_8   = (np.clip(_col_rgb ** 1.15 * 0.92, 0, 1) * 255).astype(np.uint8)
    # Clay raking: pure greyscale hillshade using hillshade()
    _clay_hs = ls_nw.hillshade(_rake_z, vert_exag=3.0)
    _clay_8  = (np.clip(_clay_hs * 0.88 + 0.04, 0, 1) * 255).astype(np.uint8)

    ax23a = fig.add_subplot(gs[extra_row_idx:extra_row_idx+2, 0:2])
    ax23a.imshow(_col_8); ax23a.axis('off')
    ax23a.set_title(f"Colour Raking (SW 45° hillshade){_3d_label}",
                    fontsize=16, fontweight='bold', color=txt_col)
    ax23a.set_facecolor(bg_col)

    ax23b = fig.add_subplot(gs[extra_row_idx:extra_row_idx+2, 2:4])
    ax23b.imshow(_clay_8, cmap='gray'); ax23b.axis('off')
    ax23b.set_title(f"Normal Raking — NW 45° (PS normals){_3d_label}",
                    fontsize=16, fontweight='bold', color=txt_col)
    ax23b.set_facecolor(bg_col)
    extra_row_idx += 2

    # ── 3D Perspective renders via headless_render_worker (VisPy/GLFW) ──
    # Same GPU-rendered quality as the live 3D viewer.
    # All 4 renders are launched IN PARALLEL to minimise wait time.
    progress(("Launching 3D renders in parallel (may take ~30s)..."))

    def _build_mesh(depth_arr, height_sc, step=None):
        """Build (verts, faces) from a depth array."""
        h3d, w3d = depth_arr.shape
        _step = step or max(1, max(h3d, w3d) // 600)
        z3 = depth_arr[::_step, ::_step].astype(np.float32) * height_sc
        ys, xs = np.mgrid[0:h3d:_step, 0:w3d:_step]
        gh, gw = ys.shape
        verts = np.column_stack(
            ((xs - w3d/2).flatten(),
             ((ys - h3d/2) * -1).flatten(),
             (z3 - z3.mean()).flatten())
        ).astype(np.float32)
        rr, cc = np.meshgrid(np.arange(gh-1), np.arange(gw-1), indexing='ij')
        v0 = rr*gw+cc
        faces = np.vstack((
            np.column_stack((v0.flatten(), (v0+gw).flatten(), (v0+1).flatten())),
            np.column_stack(((v0+1).flatten(), (v0+gw).flatten(), (v0+gw+1).flatten()))
        )).astype(np.uint32)
        return verts, faces, gh, gw, _step

    def _build_colors(albedo_arr, depth_arr, normals_arr,
                      gh, gw, step, azdeg, altdeg, clay=False):
        """Per-vertex RGBA colours using PS normal-map PBR lighting.

        Matches the 3D viewer appearance exactly:
        - Uses the PS normal map for per-pixel lighting (not depth gradients)
        - Same dot-product shading as the live OpenGL viewer
        - Clay mode: normals-based diffuse with clay-white albedo
        - Colour mode: albedo × diffuse + ambient

        This replaces the old matplotlib LightSource approach which
        was using the depth map as a DEM and gave a hillshade look,
        not the actual 3D viewer PBR appearance.
        """
        import math as _math
        # Sample at grid step
        rgb_s  = albedo_arr[::step, ::step].astype(np.float32) / 255.0
        N_s    = normals_arr[::step, ::step].astype(np.float32)  # (gh, gw, 3)
        # Normalise normals
        _nlen = np.linalg.norm(N_s, axis=-1, keepdims=True)
        N_s   = N_s / np.maximum(_nlen, 1e-8)

        # Build light vector from azimuth and altitude (same convention as 3D viewer)
        _alt_r = _math.radians(altdeg)
        _az_r  = _math.radians(azdeg)
        lx = float(_math.cos(_alt_r) * _math.sin(_az_r))
        # Normals are +Y up (north), so a compass light is (sin az, cos az):
        # HELIO 2 negated y and lit each render from the mirrored side (KD-041).
        ly = float(_math.cos(_alt_r) * _math.cos(_az_r))
        lz = float(_math.sin(_alt_r))
        L  = np.array([lx, ly, lz], dtype=np.float32)

        # Diffuse = max(0, N · L)
        N_flat  = N_s.reshape(-1, 3)
        diffuse = np.maximum(0.0, N_flat @ L).reshape(gh, gw)

        # Ambient: 15% fill light
        amb = 0.15
        shading = np.clip(diffuse + amb, 0, 1)

        if clay:
            # Clay-white render: neutral albedo, normal-map diffuse
            clay_col = np.array([0.82, 0.80, 0.78], np.float32)  # warm off-white
            lit = shading[..., None] * clay_col[None, None, :]
            # Slight specular highlight
            spec = np.power(np.clip(diffuse, 0, 1), 12.0) * 0.3
            lit  = np.clip(lit + spec[..., None], 0, 1)
        else:
            # Colour mode: albedo × shading (matching viewer _shade_rgb_pbr)
            lit = rgb_s * shading[..., None]
            # Gamma 0.9 to match viewer display (slightly lifted)
            lit = np.clip(lit ** 0.9, 0, 1)

        ones = np.ones((gh * gw,), np.float32)
        return np.column_stack((lit.reshape(-1, 3), ones)).astype(np.float32)

    _hsc      = ss['height'] * 0.15
    # _depth_raw is already capped to 1200px (from raking section above).
    # _alb_c and _nrm_c MUST match its dimensions exactly — _build_mesh
    # derives step from _depth_c then _build_colors slices all three arrays
    # at that step; shape mismatch causes the "cannot reshape" crash.
    _depth_c  = _depth_raw
    _alb_c    = cv2.resize(_c(D.albedo_rgb_full),
                           (_depth_c.shape[1], _depth_c.shape[0]),
                           interpolation=cv2.INTER_AREA)
    _nrm_c    = cv2.resize(_c(D.normals_full).astype(np.float32),
                           (_depth_c.shape[1], _depth_c.shape[0]),
                           interpolation=cv2.INTER_AREA)

    verts_c, faces_c, gh_c, gw_c, step_c = _build_mesh(_depth_c, _hsc)
    col_c  = _build_colors(_alb_c,  _depth_c, _nrm_c, gh_c, gw_c, step_c, 225, 45, clay=False)
    clay_c = _build_colors(_alb_c,  _depth_c, _nrm_c, gh_c, gw_c, step_c, 315, 45, clay=True)

    import tempfile, uuid
    _tmp = tempfile.gettempdir()
    def _tp(name): return os.path.join(_tmp, f"helio_pub_{uuid.uuid4().hex[:8]}_{name}.png")

    p_cc  = _tp("crop_col");  p_ck  = _tp("crop_clay")
    p_fc  = _tp("full_col");  p_fk  = _tp("full_clay")

    # Launch all renders in parallel — total wait ≈ time of slowest render
    procs = []
    procs.append(multiprocessing.Process(
        target=headless_render_worker,
        args=(verts_c, faces_c, col_c,  p_cc, bg_col, 35, -60)))
    procs.append(multiprocessing.Process(
        target=headless_render_worker,
        args=(verts_c, faces_c, clay_c, p_ck, bg_col, 35, -60)))

    if use_crop:
        _depth_f_raw = D.depth_vis_p.astype(np.float32)
        _fsc = min(_RAKE_CAP / max(1, _depth_f_raw.shape[0]),
                   _RAKE_CAP / max(1, _depth_f_raw.shape[1]))
        _depth_f = cv2.resize(_depth_f_raw,
                              (max(1,int(_depth_f_raw.shape[1]*_fsc)),
                               max(1,int(_depth_f_raw.shape[0]*_fsc))),
                              interpolation=cv2.INTER_AREA)
        _alb_f   = cv2.resize(D.albedo_rgb_full,
                              (_depth_f.shape[1], _depth_f.shape[0]),
                              interpolation=cv2.INTER_AREA)
        _nrm_f   = cv2.resize(D.normals_full.astype(np.float32),
                              (_depth_f.shape[1], _depth_f.shape[0]),
                              interpolation=cv2.INTER_AREA)
        verts_f, faces_f, gh_f, gw_f, step_f = _build_mesh(_depth_f, _hsc)
        col_f  = _build_colors(_alb_f, _depth_f, _nrm_f, gh_f, gw_f, step_f, 225, 45, clay=False)
        clay_f = _build_colors(_alb_f, _depth_f, _nrm_f, gh_f, gw_f, step_f, 315, 45, clay=True)
        procs.append(multiprocessing.Process(
            target=headless_render_worker,
            args=(verts_f, faces_f, col_f,  p_fc, bg_col, 35, -60)))
        procs.append(multiprocessing.Process(
            target=headless_render_worker,
            args=(verts_f, faces_f, clay_f, p_fk, bg_col, 35, -60)))

    for p in procs: p.start()
    # 10s timeout per render — headless VisPy/GLFW can hang indefinitely
    # on some macOS configurations. If it times out the panel shows
    # "Render unavailable" rather than blocking the whole export.
    for p in procs: p.join(timeout=10)
    for p in procs:
        if p.is_alive():
            print("[Publisher] 3D render timed out — skipping")
            p.terminate(); p.join(timeout=3)

    # Insert rendered images into figure
    def _add_render(ax, path, title):
        if os.path.exists(path):
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGRA2RGBA if path.endswith('.png') else cv2.COLOR_BGR2RGB)
            ax.imshow(img); os.remove(path)
        else:
            ax.text(0.5, 0.5, "Render unavailable\n(VisPy/GLFW not installed)",
                    transform=ax.transAxes, ha='center', va='center',
                    color=txt_col, fontsize=12)
        ax.axis('off'); ax.set_facecolor(bg_col)
        ax.set_title(title, fontsize=16, fontweight='bold', color=txt_col)

    ax3d_ca = fig.add_subplot(gs[extra_row_idx:extra_row_idx+2, 0:2])
    _add_render(ax3d_ca, p_cc,  f"3D Colour Raking (45° SW){_3d_label}")
    ax3d_cb = fig.add_subplot(gs[extra_row_idx:extra_row_idx+2, 2:4])
    _add_render(ax3d_cb, p_ck,  f"3D Clay Raking (45° NW){_3d_label}")
    extra_row_idx += 2

    if use_crop:
        ax3d_fa = fig.add_subplot(gs[extra_row_idx:extra_row_idx+2, 0:2])
        _add_render(ax3d_fa, p_fc, "3D Colour Raking (Full Image)")
        ax3d_fb = fig.add_subplot(gs[extra_row_idx:extra_row_idx+2, 2:4])
        _add_render(ax3d_fb, p_fk, "3D Clay Raking (NW, Full Image)")
        extra_row_idx += 2

    # ── Metrology ─────────────────────────────────────────────────────
    _met_enabled = opt_metrology
    if _met_enabled:
        ax_met_L = fig.add_subplot(gs[rows, 0]); ax_met_L.axis('off')
        ax_met_R = fig.add_subplot(gs[rows, 3]); ax_met_R.axis('off')
        ax_met_L.set_facecolor(bg_col if bg_col=='white' else '#111820')
        ax_met_R.set_facecolor(bg_col if bg_col=='white' else '#111820')
        try:
            from helio.application.heights import height_px
            from helio.core.metrology import level_plane
            _d_met = level_plane(_c(height_px(D)) if z_mm_per_unit else _c(D.depth_vis_p))   # form removed
            _d_met = _d_met - np.mean(_d_met)
            _h_met, _w_met = _d_met.shape
            Sq  = float(np.sqrt(np.mean(_d_met**2)))
            Sa  = float(np.mean(np.abs(_d_met)))
            Sz  = float(np.max(_d_met) - np.min(_d_met))
            Sp  = float(np.max(_d_met)); Sv = float(np.abs(np.min(_d_met)))
            Ssk = float(np.mean(_d_met**3) / (Sq**3 + 1e-15))
            Sku = float(np.mean(_d_met**4) / (Sq**4 + 1e-15))
            _z_mpu2 = z_mm_per_unit
            _ppi_v  = ppi
            _zp = _d_met[_h_met//2, :]
            Ra_m=float(np.mean(np.abs(_zp-np.mean(_zp))))
            Rq_m=float(np.sqrt(np.mean((_zp-np.mean(_zp))**2)))
            Rz_m=float(np.max(_zp)-np.min(_zp))
            _mfc='#101828' if bg_col!='white' else '#f0f4f8'
            _mec='#2a4060' if bg_col!='white' else '#aabbcc'
            if _z_mpu2:
                sc=_z_mpu2; u='mm'
                L_text = (
                    f"ISO 25178 Areal\n(calibrated, {_z_mpu2*1000:.2f} µm/px of height)\n─────────────────\n"
                    f"Sa  = {Sa*sc:.5f} {u}\nSq  = {Sq*sc:.5f} {u}\nSz  = {Sz*sc:.5f} {u}\n"
                    f"Sp  = {Sp*sc:.5f} {u}\nSv  = {Sv*sc:.5f} {u}\nSsk = {Ssk:.4f}\nSku = {Sku:.4f}"
                )
                R_text = (
                    f"Profile (centre-line)\n─────────────────\n"
                    f"Ra = {Ra_m*sc:.5f} {u}\nRq = {Rq_m*sc:.5f} {u}\nRz = {Rz_m*sc:.5f} {u}\n"
                    + (f"\nRegion: {_w_met}×{_h_met} px\nPPI: {_ppi_v:.1f}" if _ppi_v else f"\nRegion: {_w_met}×{_h_met} px")
                )
            else:
                L_text = (
                    f"ISO 25178 Areal\n(relative — uncalibrated)\n─────────────────\n"
                    f"Sa  = {Sa:.6f}\nSq  = {Sq:.6f}\nSz  = {Sz:.6f}\n"
                    f"Sp  = {Sp:.6f}\nSv  = {Sv:.6f}\nSsk = {Ssk:.4f}\nSku = {Sku:.4f}"
                )
                R_text = (
                    f"Profile (centre-line)\n─────────────────\n"
                    f"Ra = {Ra_m:.6f}\nRq = {Rq_m:.6f}\nRz = {Rz_m:.6f}\n"
                    f"\nRegion: {_w_met}×{_h_met} px\nUse Steps 3&4 for mm"
                )
            _mkw=dict(fontsize=9,fontfamily='monospace',va='top',color=txt_col,
                      bbox=dict(boxstyle='round,pad=0.5',facecolor=_mfc,alpha=0.92,edgecolor=_mec))
            ax_met_L.text(0.04,0.97,L_text,transform=ax_met_L.transAxes,**_mkw)
            ax_met_L.set_title("Surface Metrology",fontsize=11,fontweight='bold',color=txt_col,pad=4)
            ax_met_R.text(0.04,0.97,R_text,transform=ax_met_R.transAxes,**_mkw)
            ax_met_R.set_title("Profile Stats",fontsize=11,fontweight='bold',color=txt_col,pad=4)
        except Exception as _me:
            ax_met_L.text(0.1,0.5,f"Met. error:\n{_me}",transform=ax_met_L.transAxes,
                          fontsize=8,color=txt_col,va='center')

    # ── 3D viewer screenshots ─────────────────────────────────────────
    if _pub_shots:
        for shot_i, shot_path in enumerate(_pub_shots):
            try:
                shot_img = cv2.cvtColor(cv2.imread(shot_path), cv2.COLOR_BGR2RGB)
                r_idx = extra_row_idx + (shot_i//2)*2; c_idx = (shot_i%2)*2
                # Guard: only draw if rows exist in GridSpec
                if r_idx + 2 > base_rows: continue
                ax_shot = fig.add_subplot(gs[r_idx:r_idx+2, c_idx:c_idx+2])
                ax_shot.imshow(shot_img); ax_shot.axis('off')
                ax_shot.set_title(f"3D Viewer Capture {shot_i+1}", fontsize=16, fontweight='bold', color=txt_col)
                ax_shot.set_facecolor(bg_col)
            except Exception as _e: _log.debug("suppressed exception: %s", _e)

        # If odd number of screenshots, fill the last empty slot with a
        # centre-detail crop of the albedo render rather than blank space.
        if len(_pub_shots) % 2 == 1:
            try:
                _fill_r = extra_row_idx + (len(_pub_shots)//2)*2
                _fill_c = 2   # right column
                if _fill_r + 2 <= base_rows:
                    ax_fill = fig.add_subplot(gs[_fill_r:_fill_r+2, _fill_c:_fill_c+4])
                    _alb_fill = cv2.cvtColor(
                        cv2.resize(D.albedo_rgb_full,
                                   (512, 512)), cv2.COLOR_BGR2RGB)
                    # Centre crop at 50 %
                    _fh, _fw = _alb_fill.shape[:2]
                    _q1, _q2 = _fh//4, 3*_fh//4
                    _p1, _p2 = _fw//4, 3*_fw//4
                    ax_fill.imshow(_alb_fill[_q1:_q2, _p1:_p2])
                    ax_fill.axis('off')
                    ax_fill.set_title("Centre Detail (Albedo)", fontsize=14,
                                      fontweight='bold', color=txt_col, alpha=0.7)
                    ax_fill.set_facecolor(bg_col)
            except Exception as _e: _log.debug("suppressed exception: %s", _e)

        extra_row_idx += math.ceil(len(_pub_shots)/2)*2

    # ── Capture Metadata Sheet panel ─────────────────────────────────
    if opt_capture_meta:
        _cm = capture_meta or {}
        if _cm:
            # Add a full-width row at the bottom for the metadata table
            # Guard: _meta_row must be within allocated GridSpec rows
            _meta_row = min(extra_row_idx + 2, base_rows - 1)
            ax_meta = fig.add_subplot(gs[_meta_row, :])
            _opt_bg = (bg_col == 'white')
            ax_meta.set_facecolor("#0d1520" if not _opt_bg else "#f8f8f8")
            ax_meta.axis("off")
            _txt_col = "#c8ddf0" if not _opt_bg else "#1a1a2a"
            _lbl_col = "#5dade2" if not _opt_bg else "#1a5a8a"
            # Field display pairs
            _fields = [
                ("Institution",         _cm.get("institution","—")),
                ("Department",          _cm.get("collection","—")),
                ("Operator",            _cm.get("operator","—")),
                ("Object Title",        _cm.get("title","—")),
                ("Date of Capture",     _cm.get("capture_date","—")),
                ("Camera",              _cm.get("camera","—")),
                ("Lens",                _cm.get("lens","—")),
                ("Sensor",              _cm.get("sensor","—")),
                ("Lighting",            _cm.get("lights","—")),
                ("Stand / Rig",         _cm.get("stand","—")),
                ("Working Dist (mm)",   _cm.get("working_dist","—")),
                ("Scale (mm/px)",       _cm.get("scale_mmpx","—")),
                ("Image Count (PS)",    _cm.get("img_count",str(n_images) if n_images else "—")),
                ("Notes",               _cm.get("notes","—")),
                ("DOI / Reference",     _cm.get("doi","—")),
            ]
            # The object profile's own fields, those filled in (helio.domain.object_profiles)
            from helio.domain import object_profiles as _OP
            _prof = _OP.of(_cm)
            if _prof.key != "general":
                _fields.append(("Object Type", _prof.name))
                _fields += [(_l, _cm[_k]) for _l, _k, _h, _t in _prof.fields
                            if str(_cm.get(_k, "")).strip()][:9]
            # Layout in 3 columns
            _cols = 3
            _per_col = math.ceil(len(_fields) / _cols)
            ax_meta.set_title("Capture Metadata", fontsize=18, fontweight="bold",
                              color=_lbl_col, loc="left", pad=6)
            for _fi, (_lbl, _val) in enumerate(_fields):
                _col_i = _fi // _per_col
                _row_i = _fi % _per_col
                _x = 0.01 + _col_i * 0.335
                _y = 0.88 - _row_i * (0.82 / _per_col)
                ax_meta.text(_x, _y, f"{_lbl}:", fontsize=12,
                             color=_lbl_col, transform=ax_meta.transAxes,
                             fontweight="bold", ha="left", va="top")
                ax_meta.text(_x + 0.10, _y, _val, fontsize=12,
                             color=_txt_col, transform=ax_meta.transAxes,
                             ha="left", va="top")
            # Subtle border line at top of panel
            ax_meta.plot([0, 1], [0.97, 0.97], color=_lbl_col,
                         linewidth=0.8, alpha=0.5,
                         transform=ax_meta.transAxes)

    # ── Metadata suptitle ─────────────────────────────────────────────
    title_str = "Helio  |  Advanced Photometric Stereo • Quantitative RTI • Archival Metrology"
    if opt_metadata:
        meta_parts = [p for p in [opt_obj_str, time.strftime("%Y-%m-%d"),
                       opt_engine_str,
                       f"{n_images} images" if n_images else ""] if p]
        if scale_val > 0:
            meta_parts.append(f"Width: {scale_val:.2f} mm")
            if ppi: meta_parts.append(f"PPI: {ppi:.1f}")
        if meta_parts: title_str += "\n" + "  •  ".join(meta_parts)
        if use_crop: title_str += f"\nCrop: ({cx1},{cy1})–({cx2},{cy2})  [{_pw}×{_ph} px]"
    elif scale_val > 0:
        title_str += f"\nWidth: {scale_val:.2f} mm" + (f" | PPI: {ppi:.1f}" if ppi else "")
    fig.suptitle(title_str, fontsize=36, fontweight='bold', color=txt_col)

    progress((f"Saving {fmt.upper()} at {_fig_dpi} DPI..."))
    # tight_layout and bbox_inches='tight' both re-render the entire figure
    # and run a constraint solver — on a 24×144 inch, 40-subplot figure at
    # 300 DPI this takes 10–30 minutes. Use subplots_adjust instead.
    plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.03,
                        hspace=0.40, wspace=0.15)
    fig.savefig(filepath, dpi=_fig_dpi, format=fmt, facecolor=bg_col)
    plt.close(fig)

    progress("Done!")
    return {"path": filepath, "panels": [t for _img, t, _c in panels]}


# ── publication figures ──────────────────────────────────────────────────

FIGURE_ENGINES = (
    ("Standard PBR", "01_standard_pbr"), ("PS Normal-Map Lit (Single)", "02_hillshade"),
    ("PS Normal-Map Lit (8-Dir)", "03_multidirectional_hs"), ("Raking Composite", "04_raking_composite"),
    ("Positive Openness", "05_positive_openness"), ("Negative Openness", "06_negative_openness"),
    ("Sky-View Factor (SVF)", "07_svf"), ("Multi-Scale Relief Model (MSRM)", "08_msrm"),
    ("Surface Curvature (Scratches)", "09_curvature"), ("Ambient Occlusion (AO)", "10_ao"),
    ("PCA 1", "11_pca1"), ("PCA 2", "12_pca2"), ("PCA 3", "13_pca3"),
    ("False-Colour Depth", "14_falsecolour_depth"), ("Shape Index", "15_shape_index"),
    ("Retinex Enhancement", "16_retinex"), ("Normals RGB", "17_normals_rgb"),
    ("Specular Enhancement", "18_specular"), ("Local Dominance", "19_local_dominance"),
    ("Bilateral Normal Sharpening", "20_bilateral_sharp"))
TRANSMITTANCE = ("Captured Transmittance (photographed)", "21_captured_transmittance")


def publication_figures(draw: Callable, selected, out_dir: str, *, full_shape, obj_id: str = "",
                        mm_per_px: float = 0.0, dpi: int = 300, transmittance=None,
                        progress: Callable | None = None) -> list[str]:
    """Print-resolution TIFF figures, one per view, with a 10 mm scale bar
    once a scale is set, a caption, and a manifest.  *draw(engine)* renders
    at preview size; each is resized to the capture's *full_shape*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    obj_id = obj_id or "HELIO_OBJECT"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in obj_id)
    H, W = full_shape
    today = time.strftime("%Y-%m-%d")
    os.makedirs(out_dir, exist_ok=True)
    selected = list(selected)
    files, exported = [], []
    for n, (engine, fname) in enumerate(selected, 1):
        if progress:
            progress(f"Rendering {n}/{len(selected)}: {engine}", n / len(selected))
        try:
            if fname == TRANSMITTANCE[1]:
                if transmittance is None:
                    continue
                img = (np.clip(transmittance, 0, 1) * 255).astype(np.uint8)
            else:
                img = np.asarray(draw(engine))
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LANCZOS4)
        except Exception as e:
            print(f"[PubFig] {engine}: {e}")
            continue
        fig, ax = plt.subplots(figsize=(8, 10), facecolor="#0e1520")
        ax.set_facecolor("#0e1520")
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.axis("off")
        if mm_per_px and mm_per_px > 0:
            bar_mm = 10.0
            bar_px = bar_mm / mm_per_px
            bx, by = 0.04, 0.05
            ax.axhline(y=H * (1 - by - 0.01), xmin=bx, xmax=bx + bar_px / W, color="white", lw=2.5,
                       solid_capstyle="butt")
            ax.text(bx * W + bar_px / 2, H * (1 - by - 0.035), f"{bar_mm:.0f} mm", color="white",
                    fontsize=7, ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, ec="none"))
        fig.text(0.5, 0.005, f"Fig. {n:02d}  —  {engine}  —  {obj_id}  |  Helio  |  {today}",
                 color="#aabbcc", fontsize=7, ha="center", va="bottom")
        p = os.path.join(out_dir, f"{safe}_{fname}.tif")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.03)
        fig.savefig(p, dpi=dpi, facecolor="#0e1520", format="tiff")
        plt.close(fig)
        exported.append(fname)
        files.append(p)
    mp = os.path.join(out_dir, f"{safe}_figure_manifest.txt")
    with open(mp, "w") as f:
        f.write(f"Helio — Publication Figure Manifest\n{'=' * 50}\n")
        f.write(f"Object ID : {obj_id}\nDate : {today}\nDPI : {dpi}\n\n")
        for i, (e, fn) in enumerate(selected, 1):
            if fn in exported:
                f.write(f"  Fig. {i:02d}  {e:<42} → {safe}_{fn}.tif\n")
    return files + [mp]


# ── condition report ─────────────────────────────────────────────────────

def _level(d):
    """The least-squares plane removed (the minimal ISO 25178-2 F-operator)."""
    d = np.asarray(d, np.float64)
    h, w = d.shape
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(h * w)])
    c, *_ = np.linalg.lstsq(A, d.ravel(), rcond=None)
    return d - (A @ c).reshape(h, w)


def condition_map(depth, albedo, out_dir: str, safe: str, *, blend: float = 0.45,
                  z_mm_per_unit=None, operator: str = "", notes: str = "", obj_id: str = "",
                  depth_engine: str = "", cmap: str = "viridis") -> list[str]:
    """Mode A: height coloured over the albedo, and the areal statistics.

    Heights are levelled (plane removed) before Sa…Sz and reported in µm
    only when a height scale exists; otherwise in relative units, said so.
    The colour map is perceptually uniform: HELIO 2 used "jet", whose bands
    invent edges that are not in the surface (Borland & Taylor 2007;
    Crameri et al. 2020), and wrote "um" beside uncalibrated numbers (KD-042)."""
    import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from helio.core.metrology import iso25178
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = np.asarray(depth, np.float32)
    if albedo is None:
        albedo = np.zeros((*d.shape, 3), np.uint8)
    dn = (d - d.min()) / (np.ptp(d) + 1e-9)
    heat = (plt.get_cmap(cmap)(dn)[..., :3] * 255).astype(np.uint8)
    out = (np.clip(heat.astype(np.float32) / 255 * blend
                   + np.asarray(albedo).astype(np.float32) / 255 * (1 - blend), 0, 1) * 255
           ).astype(np.uint8)
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"{safe}_{ts}_condition_map.png")
    cv2.imwrite(img_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    z = float(z_mm_per_unit or 0.0)
    rows, _unit = iso25178(_level(d), z / 1.0 if z else 0.0)
    unit = "um" if z else "relative"
    k = 1000.0 if z else 1.0                   # mm → µm when calibrated
    csv_path = os.path.join(out_dir, f"{safe}_{ts}_surface_stats.csv")
    with open(csv_path, "w") as f:
        for name, value, _m in rows:
            if name in ("Sa", "Sq", "Sz", "Sp", "Sv"):
                f.write(f"{name},{value * k:.4f},{unit}\n")
            else:
                f.write(f"{name},{value:.4f},\n")
        f.write(f"Levelling,least-squares plane removed,\nColour map,{cmap},\n"
                f"Operator,{operator},\nNotes,{notes},\nObjectID,{obj_id},\nTimestamp,{ts},\n"
                f"Solver,{depth_engine},\n")
    return [img_path, csv_path]


def change_detection(depth, baseline, albedo, out_dir: str, safe: str, *, operator: str = "",
                     baseline_name: str = "") -> list[str]:
    """Mode B: this surface minus a baseline depth (resized to it): the
    albedo, the signed change and its distribution, and the statistics."""
    import datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = np.asarray(depth, np.float32)
    b = np.asarray(baseline, np.float32)
    if b.ndim == 3:
        b = b[..., 0]
    if b.shape != d.shape:
        b = cv2.resize(b, (d.shape[1], d.shape[0]))
    delta = d - b
    vmax = float(np.percentile(np.abs(delta), 98)) or 1e-9
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#111111")
    axes[0].imshow(albedo)
    axes[0].set_title("Albedo", color="white")
    im = axes[1].imshow(delta, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Depth Delta (Blue=Loss, Red=Gain)", color="white")
    plt.colorbar(im, ax=axes[1])
    axes[2].hist(delta.flatten(), bins=100, color="#4488cc", edgecolor="none")
    axes[2].set_title("Delta Distribution", color="white")
    axes[2].set_facecolor("#1a1a2a")
    for ax in axes[:2]:
        ax.axis("off")
    axes[2].tick_params(colors="white")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"{safe}_{ts}_change_detection.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    csv_path = os.path.join(out_dir, f"{safe}_{ts}_change_stats.csv")
    with open(csv_path, "w") as f:
        f.write(f"Mean_delta,{float(delta.mean()):.4f},depth_units\n"
                f"Std_delta,{float(delta.std()):.4f},depth_units\n"
                f"Max_gain,{float(delta.max()):.4f},depth_units\n"
                f"Max_loss,{float(delta.min()):.4f},depth_units\n"
                f"RMS_change,{float(np.sqrt(np.mean(delta ** 2))):.4f},depth_units\n"
                f"Operator,{operator},\nBaseline,{baseline_name},\nTimestamp,{ts},\n")
    return [fig_path, csv_path]


# ── comparing two regions ────────────────────────────────────────────────

ISO_DEFS = {"Sa": "Areal mean abs height (primary roughness)", "Sq": "Areal RMS height",
            "Ssk": "Skewness (neg=valley-heavy)", "Sku": "Kurtosis (Gaussian=3.0)",
            "Sz": "Peak-to-valley height", "Sp": "Max peak above mean",
            "Sv": "Max valley below mean"}


def region_stats(roi, label: str, *, z_mm_per_unit=None, mm_per_px=None) -> dict:
    """The classic Surface Comparison's ISO 25178 height figures for one
    region of true heights (helio.application.heights), its plane removed:
    µm when a height scale is set, else pixels of height."""
    from helio.core.metrology import level_plane
    d = level_plane(roi)
    if z_mm_per_unit:
        d = (d - d.min()) * z_mm_per_unit * 1000
        unit = "µm (calibrated)"
    else:
        d = d - d.min()
        unit = "px of height"
    m = float(np.mean(d))
    sq = float(np.sqrt(np.mean((d - m) ** 2)))
    return {"ROI": label, "Width (px)": d.shape[1], "Height (px)": d.shape[0], "Pixels": d.size,
            "Unit": unit, "Mean z": round(m, 5), "Sa": round(float(np.mean(np.abs(d - m))), 5),
            "Sq": round(sq, 5), "Ssk": round(float(np.mean((d - m) ** 3) / (sq ** 3 + 1e-12)), 4),
            "Sku": round(float(np.mean((d - m) ** 4) / (sq ** 4 + 1e-12)), 4),
            "Sz": round(float(np.max(d) - np.min(d)), 5), "Sp": round(float(np.max(d) - m), 5),
            "Sv": round(float(m - np.min(d)), 5)}


def surface_comparison_csv(path: str, sa: dict, sb: dict, meta_rows) -> str:
    import csv
    with open(path, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["Parameter", "ROI A", "ROI B", "Diff (A-B)", "ISO 25178 definition"])
        for k in [k for k in sa if k != "ROI"]:
            av, bv = sa[k], sb[k]
            try:
                diff = round(float(av) - float(bv), 5)
            except (TypeError, ValueError):
                diff = "—"
            cw.writerow([k, av, bv, diff, ISO_DEFS.get(k, "")])
        cw.writerow([])
        for row in meta_rows:
            cw.writerow(row)
    return path


def surface_comparison_figure(roi_a, roi_b, sa: dict, sb: dict, obj_id: str = "",
                              path: str | None = None):
    """The two regions and their figures side by side; PNG at *path*, and
    returned as an RGB array."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="#1a2030")
    for ax in axes:
        ax.set_facecolor("#0e1520")
        ax.tick_params(colors="#aabbcc", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
    for i, (roi, s, cmap) in enumerate(((roi_a, sa, cv2.COLORMAP_VIRIDIS),
                                        (roi_b, sb, cv2.COLORMAP_PLASMA))):
        disp = cv2.applyColorMap((np.clip(roi, 0, 1) * 255).astype(np.uint8), cmap)
        axes[i].imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{s['ROI']}  Sa={s['Sa']}  Sz={s['Sz']}", color="white", fontsize=9)
    keys = ["Sa", "Sq", "Ssk", "Sku", "Sz", "Sp", "Sv"]
    xs = list(range(len(keys)))
    axes[2].bar([x - 0.2 for x in xs], [float(sa[k]) for k in keys], 0.38, color="#5599cc",
                label="ROI A")
    axes[2].bar([x + 0.2 for x in xs], [float(sb[k]) for k in keys], 0.38, color="#ee8844",
                label="ROI B")
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(keys, fontsize=8)
    leg = axes[2].legend(fontsize=8)
    for t in leg.get_texts():
        t.set_color("#dddddd")
    axes[2].set_title("ISO 25178 Comparison", color="white", fontsize=9)
    fig.suptitle(f"Surface Statistics Comparison  |  {obj_id}", color="white", fontsize=10)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = np.asarray(Image.open(buf).convert("RGB"))
    if path:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


# ── change map against a reference ───────────────────────────────────────

def load_reference_depth(path: str) -> np.ndarray:
    """An earlier depth map of the same object (.npy, 16-bit PNG or TIFF), 0–1."""
    if path.lower().endswith(".npy"):
        ref = np.load(path).astype(np.float32)
    else:
        ref = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if ref is None:
            raise ValueError(f"Could not read {os.path.basename(path)}.")
        ref = ref.astype(np.float32)
    return cv2.normalize(ref, None, 0, 1, cv2.NORM_MINMAX)


def change_map(current, reference, path: str, z_mm_per_unit=None) -> dict:
    """This surface minus the reference (resized to it) as a tagged GeoTIFF,
    in normalised units: the reference is an earlier depth map stretched to
    0-1, so no height scale applies to it.  Changes within three robust standard
    deviations (1.4826 × MAD) of the difference are noise, not change."""
    from helio.io.exporters import write_geotiff
    cur = np.asarray(current, np.float32)
    r = np.asarray(reference, np.float32)
    if r.shape != cur.shape:
        r = cv2.resize(r, (cur.shape[1], cur.shape[0]), interpolation=cv2.INTER_AREA)
    diff = cur - r
    unit = "normalised"                      # z_mm_per_unit is mm per pixel of height: not these units
    mad = float(np.median(np.abs(diff - np.median(diff))))
    noise = 1.4826 * mad
    moved = int((np.abs(diff) > 3 * noise).sum())
    write_geotiff(path, diff.astype(np.float32), tags={
        "HELIO_LAYER": "change", "HELIO_UNITS": unit, "HELIO_NOISE_FLOOR": f"{noise:.6g}",
        "HELIO_CHANGED_PX": str(moved)})
    return {"path": path, "unit": unit, "noise": noise, "moved": moved,
            "moved_pct": 100.0 * moved / diff.size}
