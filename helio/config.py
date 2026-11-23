"""Application-wide constants, paths, and default values."""

import os
import multiprocessing
import tempfile

# ── Optional dependencies ──────────────────────────────────────────
try:
    import rvt.vis  # noqa: F401   (not rvt.blend_func: it breaks on new matplotlib and nothing uses it)
    HAS_RVT = True
except ImportError:
    HAS_RVT = False

try:
    import rasterio  # noqa: F401
    from rasterio.transform import from_origin  # noqa: F401
    import warnings as _w_rio
    _w_rio.filterwarnings(
        "ignore",
        category=getattr(rasterio.errors, "NotGeoreferencedWarning", UserWarning),
        module="rasterio",
    )
    del _w_rio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# ── Paths ────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(tempfile.gettempdir(), "helio_logs")
SESSION_DIR = os.path.join(tempfile.gettempdir(), "helio_sessions")
PUB_SHOT_DIR = os.path.join(tempfile.gettempdir(), "helio_3d_pub_shots")
DEFAULT_LOG_PATH = os.path.join(LOG_DIR, "helio.log")

# ── Processing limits ────────────────────────────────────────────────
MAX_VIEWER_PX = 9_000_000        # downscale arrays >9 MP before sending to 3D viewer
NR_MAX_PX = 3000                 # long-axis cap for Normal Residual engine
RBD_MAX_PX = 1500                # cap for RGB-PS R-B Diff engine
CICA_MAX_PX = 1200               # cap for Colour ICA engine
CONTACT_SHEET_DPI = 300
HEADLESS_RENDER_TIMEOUT = 10     # seconds per headless render process
PUB_SHEET_TIMEOUT = 600          # 10 min timeout for publication sheet

# ── 3D export resolution/quality defaults ─────────────────────────────
# Single source of truth for "how much detail to keep" across every mesh/
# texture export path. Previously each exporter hardcoded its own number
# independently (six different constants scattered across three files,
# picked at different times with no shared rationale) — that inconsistency
# is what produced visibly different quality between, e.g., the GLB and
# HF+Normals 3D viewers for the same capture. All defaults remain
# overridable per-call via each function's own ``settings``/parameter dict;
# this only centralizes what they fall back to when not overridden.
MESH_MAX_RES = 1200              # long-axis cap for exported mesh geometry (vertex grid)
ALBEDO_TEX_RES = 2048            # long-axis cap for albedo/colour PBR textures
NORMAL_TEX_RES = 8192            # long-axis cap for normal-map PBR textures — kept much
                                  # higher than albedo because normal maps encode a
                                  # per-pixel direction; area-averaging them during resize
                                  # blunts fine tool-mark/engraving detail even at a
                                  # nominally "high" resolution (see fast_large_sigma_blur-
                                  # adjacent normal-map fix in helio/io/exporters.py).
RTI_VIEWER_MAX_SIDE = 6000       # long-axis cap for the standalone PS Normal-Map RTI viewer
DEEPZOOM_MAX_EDGE = 8000         # long-axis cap shared by every DeepZoom pyramid layer
STL_DEFAULT_RES = 500            # initial value of the STL export resolution slider (FDM tier).
                                  # Was 200 — a ~30-40x downsample of typical 6000-8000px RTI
                                  # captures that crushed fine surface relief (drypoint scratches,
                                  # tool marks) before it ever reached the mesh. 500 keeps FDM
                                  # export fast while preserving meaningfully more high-frequency
                                  # detail; confirmed against real exported STLs that ~1000px
                                  # retains ~80 microns of fine-scale relief vs near-zero at 200px.

# ── 3D Viewer defaults ───────────────────────────────────────────────
VIEWER_FOV = 58
VIEWER_BG = "#1a1a2e"
VIEWER_SIZE = (1380, 900)
VIEWER_HEADLESS_SIZE = (3600, 2400)

# ── Default engine settings ──────────────────────────────────────────
DEFAULT_ENGINE_SETTINGS = {
    "gain": 1.0, "bright": 1.0, "contrast": 0, "gamma": 1.0,
    "micro": 0.0, "rough_opacity": 0.5,
    "hs_alt": 45, "hs_azi": 315, "hs_z": 1.0, "hs_amb": 0.3,
    "ao_radius": 0.08, "ao_strength": 1.5,
    "lx": 0.0, "ly": 0.0, "lz": 1.0,
    "def_sens": 1.0, "def_relief": 1.0, "def_noise": 0.5, "def_spec": 0.0,
    "shape_smooth": 1.5,
    "bil_d": 9, "bil_sc": 0.1,
    "ret_scale_s": 8, "ret_scale_m": 30, "ret_scale_l": 250,
    "hs_depth_src": "poisson",
    "depth_cmap": "viridis",
    "rk_elevation": 20, "rk_n_dirs": 4,
    "rk_blend": 0.5, "rk_usm": 0.0,
    "global_whites": 1.0, "global_blacks": 0.0, "global_midtone": 1.0,
    "global_clarity": 0.0, "global_texture": 0.0, "global_dehaze": 0.0,
    "dg_strength": 2.0, "dg_gamma": 0.7, "dg_spec": 0.3,
    "svf_radius": 20, "svf_dirs": 16,
    "op_radius": 20, "op_elev": 10,
    "msrm_min": 1, "msrm_max": 5,
    "slrm_radius": 20,
    "sss_radius": 8.0, "sss_blend": 0.45,
    "rti_spec_coeff": 1.5, "rti_spec_floor": 0.05, "rti_spec_blend": 0.6,
    "rti_usm_sigma": 2.0, "rti_usm_amount": 1.5, "rti_usm_thresh": 0.0,
    "ld_radius": 20, "ld_elev": 35,
    "si_sigma": 1.5,
}

# ── Shared-state typecode map (for multiprocessing.Value) ────────────
SHARED_STATE_DEFAULTS = {
    "step": ("i", 2), "export_step": ("i", 4), "edge_trim": ("f", 2.0),
    "warp_rem": ("f", 0.0), "mesh_smooth": ("f", 0.0),
    "clay_blend": ("f", 0.0), "height": ("f", 90.0),
    "fx": ("i", 1), "fy": ("i", -1), "fz": ("i", 1),
    "normal_invert": ("i", 0), "normal_flip_x": ("i", 0),
    "lx": ("f", 0.0), "ly": ("f", 0.0), "lz": ("f", 1.0),
    "ambient_str": ("f", 0.25), "diffuse_str": ("f", 1.0),
    "specular_str": ("f", 0.4), "shininess": ("f", 32.0),
    "auto_spin": ("i", 0), "bg_light": ("i", 0),
    "bake_3d": ("i", 0), "center_origin": ("i", 1),
    "hp_sigma": ("f", 0.0),
    "use_custom_clay": ("i", 0),
    "clay_r": ("f", 0.6), "clay_g": ("f", 0.6), "clay_b": ("f", 0.6),
    "micro_relief": ("f", 0.4), "shape_blend": ("f", 0.55),
    "hybrid_enabled": ("i", 0), "hybrid_mix": ("f", 0.12),
    "export_brightness": ("f", 1.0), "screenshot_count": ("i", 0),
    "raise_3d": ("i", 0), "clean_export": ("i", 0),
    "export_w": ("i", 0), "export_h": ("i", 0),
    "spin_speed": ("f", 0.5), "spin_capture": ("i", 0),
    "spin_frames_done": ("i", 0), "spin_total": ("i", 72),
    "ao_blend": ("f", 0.0), "normal_bump": ("f", 0.5),
    "normal_sharp": ("f", 0.0), "cavity_depth": ("f", 0.0),
    "ridge_bright": ("f", 0.0), "depth_clip_pct": ("f", 0.5),
    "x_tilt": ("f", 0.0), "y_tilt": ("f", 0.0),
    "viewer_skip_trim": ("i", 0),
    "mm_per_px": ("f", 0.0), "z_mm_per_unit": ("f", 0.0),
    "img_w": ("i", 1380), "img_h": ("i", 900),
    "viewer_gamma": ("f", 1.0), "viewer_saturation": ("f", 1.0),
    "shadow_lift": ("f", 0.0), "highlight_clip": ("f", 1.0),
    "global_whites": ("f", 1.0), "global_blacks": ("f", 0.0),
    "global_midtone": ("f", 1.0), "global_clarity": ("f", 0.0),
    "global_texture": ("f", 0.0), "global_dehaze": ("f", 0.0),
    "z_slice_lo": ("f", 0.0), "z_slice_hi": ("f", 1.0),
    "hq_normals": ("i", 0), "cross_section": ("i", 0),
    "cross_section_y": ("f", 0.5), "vertex_cap": ("i", 0),
    "hf_normals_cutoff": ("f", 0.35), "depth_gen": ("i", 0),
}
