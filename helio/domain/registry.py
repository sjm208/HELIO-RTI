"""The normal solvers and depth methods HELIO offers, in dropdown order (REG-001, part).

One list for both apps.  The legacy dropdowns must offer exactly these, in
this order (checked by tests/test_qt_shell.py against the live legacy window's
values); the blurbs are the one-line descriptions shown under each choice.
Labels are the stable identifiers used in sessions and ProcessingSettings.
"""
from __future__ import annotations

SOLVER_BLURBS = {
    "Linear Least Squares": "Default. Fast and right for most matte objects — parchment, paper, plaster, stone.",
    "Mid-Tone Weighted LS": "Counts well-exposed mid-tones most, burnt-out highlights and deep shadows least. Not a metal solver: for metal and gold use Sparse-Outlier LS (RPCA), which measured better on the DiLiGenT benchmark.",
    "IRLS (Metallic/Specular)": "Heavier outlier rejection for coins, seals and metal point.",
    "Iterative Specular Separation": "Splits shine from matte before solving. Glazed, varnished or waxed surfaces.",
    "Shadow-Rejection LS": "Drops shadowed frames per pixel. Best when relief is deep enough to cast hard shadows.",
    "Uncalibrated PCA": "No light directions needed. Rough shape when calibration is impossible.",
    "Sparse-Outlier LS (RPCA)": "Sets aside the few readings a matte surface could not give (a highlight on gold, a cast shadow), however bright, and solves from the rest. Gold tooling, metal, glossy media: the best of Helio's solvers on the DiLiGenT benchmark. Slower.",
}
# Offered after the others, under "More solvers": on the DiLiGenT benchmark
# none beat Sparse-Outlier LS (RPCA) on shiny objects (docs/BENCHMARK_DILIGENT.md),
# and Mid-Tone Weighted was worse than plain least squares on the metal one.
# Still chosen by name in sessions and by the metal-aware solve.
MORE_SOLVERS = ("Mid-Tone Weighted LS", "IRLS (Metallic/Specular)", "Iterative Specular Separation")
MAIN_SOLVERS = tuple(n for n in SOLVER_BLURBS if n not in MORE_SOLVERS)
# The solvers the metal-aware solve can use where the surface shines.
METAL_SOLVERS = ("Sparse-Outlier LS (RPCA)", "IRLS (Metallic/Specular)",
                 "Mid-Tone Weighted LS")
DEPTH_BLURBS = {
    "Fast Poisson (FFT)": "Default. Fast and reliable — start here.",
    "Frankot-Chellappa": "Keeps fine incised detail best. Assumes the surface meets itself at opposite edges.",
    "Simchony DCT": "Cosine-transform integration. Smoother on gently curved surfaces.",
    "Confidence-Weighted Poisson": "Down-weights untrustworthy pixels. Use when highlights distort the depth.",
    "Multi-Scale Integration": "Solves coarse-to-fine. Better on large, low-frequency warp such as cockling.",
    "Gradient-Smoothed Poisson": "Smooths gradients first. Quietest result on noisy captures.",
    "Directional Integration": "Integrates along one axis. Diagnostic rather than general-purpose.",
    # 2026-10-07, from the literature (helio.core.bini):
    # 2026-11-10 (helio.core.depth_integration.poisson_free_boundary):
    "Free-Boundary Poisson (DCT)": "As fast as Fast Poisson, with nothing assumed at the frame edge: a tilted object stays tilted, and relief running off the frame does not bow the whole map. Heights in pixels as they are (Quéau et al. 2018).",
    "Least-Squares (Free Boundary)": "Most accurate for measuring: no assumption at the frame edge (Quéau et al. 2018). Slower.",
    "Bilateral (Keeps Steps)": "Keeps steps sharp — coin rims, page edges, deep cuts — without tilting the ground beside them (Cao et al. 2022). Slowest.",
}

NORMAL_SOLVERS = tuple(SOLVER_BLURBS)
DEPTH_METHODS = tuple(DEPTH_BLURBS)


# The render engines the view menu offers, in menu order (moved from
# RTIViewerPro.__init__).  Engines with working implementations that are not
# offered here stay callable by name for exports: see shading_engines.
RENDER_ENGINES = (
    "Standard PBR", "Matte Relight (as 3D)", "Robust PBR (MVS)", "Normals Unsharp Masking",
    # Shape-depiction engines from the graphics literature, added
    # 2026-08-18.  (An MSII relief engine was written then too, never
    # registered, and removed 2026-10-07 (KD-007); the Integral
    # Invariant Filter in the Forensic panel is the MSII tool.)
    "Exaggerated Shading", "Curvedness",
    "Cast-Shadow Raking",
    # Stack-derived: computed from the whole light stack during the
    # solve, so unlike every other engine they do not respond to the
    # light slider. Fusion answers "where is the ink legible";
    # the residual answers "where is this normal believable".
    "Multi-Light Fusion", "PS Fit Residual (QC)", "Normal Uncertainty (QC)",
    "Slope Magnitude", "Specular Enhancement", "Image Quotient (IQ)",
    "Positive Openness", "Negative Openness", "Multi-Scale Relief Model (MSRM)",
    "Specular View", "Ambient Occlusion (AO)",
    "PS Normal-Map Lit (Single)",
    "PS Normal-Map Lit (8-Dir)",
    "Slope Gradient",
    "Surface Curvature (Scratches)",
    "PCA 1", "PCA 2", "PCA 3",
    "Light Coverage Map",
    "Normals RGB",
    "Shape Index",
    "Retinex Enhancement",
    "Colour ICA (IC1-3)",
    "RGB-PS R-B Diff",
    "Raking Composite",
    "Multi-Directional Hillshading (Blend / PCA)",
    "Albedo-Suppressed Raking (Clay Mode)",
    "Ridge and Groove Detector",
    "Normal Residual",
    "Diffuse Gain",
    "Bilateral Normal Sharpening",
    "Subsurface Scattering (SSS)",
    "RTI Specular Enhancement",
    "RTI Image USM",
    "RBF Relight (Shiny / Translucent)",
)

# Every engine starts from these settings; both PBR engines then lower their
# gain, which they apply as an exponent of 1/gain, to keep parchment and paper
# from blowing out.
_ENGINE_DEFAULTS = {
    "gain": 1.0, "bright": 1.0, "contrast": 0, "gamma": 1.0, "pbr_expose": 0,
    "micro": 0.0, "rough_opacity": 0.5,
    "hs_alt": 45, "hs_azi": 315, "hs_z": 1.0, "hs_amb": 0.3,
    "ao_radius": 0.08, "ao_strength": 1.5,
    "lx": 0.0, "ly": 0.0, "lz": 1.0,
    "def_sens": 1.0, "def_relief": 1.0, "def_noise": 0.5, "def_spec": 0.0,
    # Shape Index
    "shape_smooth": 1.5,
    # Bilateral Normal Sharpening
    "bil_d": 9, "bil_sc": 0.1,
    # Retinex
    "ret_scale_s": 8,  "ret_scale_m": 30, "ret_scale_l": 250,
    # Hillshade (merged) — depth source: "poisson" or "frankot"
    "hs_depth_src": "poisson",
    # False-Colour Depth (merged) — colormap name
    "depth_cmap": "viridis",
    # Multi-Directional Hillshading (blend / PCA)
    "mdh_alt": 35.0, "mdh_mode": 0,
    # Albedo-Suppressed Raking (Clay Mode)
    "clay_floor": 0.06, "clay_micro": 0.6,
    # Ridge and Groove Detector
    "rdg_width": 1.5, "rdg_mode": 0,
    # Raking Composite
    "rk_elevation": 20, "rk_n_dirs": 4,
    "rk_blend": 0.5,    "rk_usm": 0.0,
    # Global Image Adjustments (3D tab)
    "global_whites": 1.0, "global_blacks": 0.0, "global_midtone": 1.0,
    "global_clarity": 0.0, "global_texture": 0.0, "global_dehaze": 0.0,
    # Diffuse Gain
    "dg_strength": 2.0, "dg_gamma": 0.7, "dg_spec": 0.3, "dg_alt": 35.0,
    # PS Fit Residual (QC)
    "psr_floor": 0.05, "psr_gain": 1.5, "psr_gamma": 1.0,
    # Normal Uncertainty (QC): top of the colour scale, degrees
    "unc_max": 10.0,
    # SVF
    "svf_radius": 20, "svf_dirs": 16,
    # Openness +/-
    "op_radius": 20, "op_elev": 10, "op_smooth": 0.0, "op_clip": 0.0, "op_mode": 0,
    # Slope Gradient, Surface Curvature
    "slope_smooth": 0.0, "slope_clip": 0.0, "curv_smooth": 0.0,
    # MSRM
    "msrm_min": 1, "msrm_max": 5, "msrm_clip": 2.0,
    # SLRM
    "slrm_radius": 20,
    # Subsurface Scattering
    "sss_radius": 8.0, "sss_blend": 0.45,
    # RTI Specular Enhancement
    "rti_spec_coeff": 1.5, "rti_spec_floor": 0.05, "rti_spec_blend": 0.6,
    # Matte Relight (as 3D): the 3D relief's own lighting defaults
    "mr_ambient": 0.18, "mr_diffuse": 0.95, "mr_shine": 0.12, "mr_shininess": 24,
    # RBF Relight (Shiny / Translucent)
    "rbf_width": 1.2, "rbf_smooth": 0.05,
    # RTI Image USM
    "rti_usm_sigma": 2.0, "rti_usm_amount": 1.5, "rti_usm_thresh": 0.0,
    # Local Dominance
    "ld_radius": 20, "ld_elev": 35,
    # Shape Index
    "si_sigma": 1.5,
    # Normal Residual (also used by Normals Unsharp Masking)
    "nr_scale": 5.0, "nr_gain": 1.5, "nr_mode": 0,
}


def default_engine_settings(engines=RENDER_ENGINES) -> dict:
    """A fresh settings dict per engine, exactly as the classic app starts."""
    out = {eng: dict(_ENGINE_DEFAULTS) for eng in engines}
    for _pbr in ("Standard PBR", "Robust PBR (MVS)"):
        if _pbr in out:
            out[_pbr]["gain"] = 0.15
    for eng, own in VIEW_DEFAULTS.items():
        if eng in out:
            out[eng].update(own)
    return out


# A view's own starting values where it differs from the shared default
# (the slider's reset and its changed-mark follow it: view_controls).
VIEW_DEFAULTS: dict[str, dict] = {
    # RBF Relight interpolates the photographs, so it starts lit from among them
    # (north-west, 45° up, screen frame) rather than from overhead, where a dome
    # whose lamps share one height has no photograph to interpolate.
    "RBF Relight (Shiny / Translucent)": {"lx": -0.5, "ly": -0.5, "lz": 0.7071,
                                          "hs_azi": 315.0, "hs_alt": 45.0},
}


# ── One-line descriptions for the render engines ─────────────────────────
# Forty entries in a dropdown, most of them named after the maths rather than
# the job. A first-time user has no way to tell "Anisotropic Sky View Factor"
# from "Local Dominance" without hovering each one in turn, so the panel shows
# the description of whatever is selected, always visible, under the dropdown.
# Written as "what it shows you", not "what it computes".
ENGINE_BLURBS = {
    "Standard PBR": "Photographic relight. The everyday view — start here.",
    "Matte Relight (as 3D)": "The photograph's own colour, lit as the 3D relief lights it: soft and matte, no shiny pool of light. Move the light as usual.",
    "Robust PBR (MVS)": "Like Standard PBR but solved from outlier-rejecting normals. Better on shiny or metallic surfaces.",
    "Exaggerated Shading": "Lights every scale at once, so fine tooling and broad form stay readable together.",
    "Curvedness": "How strongly curved each point is, regardless of shape. Pairs with Shape Index.",
    "Cast-Shadow Raking": "Raking light with real cast shadows — the cue you get from tilting an object to the window.",
    "Multi-Light Fusion": "Best legibility for faded ink. Combines every frame by local contrast; ignores the light slider.",
    "PS Fit Residual (QC)": "Where the surface disobeys the maths — specular, shadowed, gilded. A trust map, not a picture.",
    "Normal Uncertainty (QC)": "How sure each normal is, in degrees: dark is sure, yellow is doubtful. Measured from each pixel's own lights.",
    "Normals Unsharp Masking": "Sharpens the normal field before lighting it. Fast way to lift fine scratches.",
    "Bilateral Normal Sharpening": "Edge-preserving sharpen — detail without haloes around strong edges.",
    "Normals RGB": "The raw normal map as colour. Diagnostic: flat colour means flat surface.",
    "Normal Residual": "Detail left after removing broad form. Good for stylus marks and hairline scratches.",
    "Slope Magnitude": "How steep each point is. Flat areas dark, walls bright.",
    "Slope Gradient": "Direction and steepness together, as a gradient map.",
    "Surface Curvature (Scratches)": "Tuned to scratches, stylus marks and drypoint burr — marks carrying no pigment.",
    "Shape Index": "What kind of shape each point is: pit, ridge, saddle or dome.",
    "Specular Enhancement": "Boosts the shine. Reveals polish, glaze and metal leaf.",
    "Specular View": "The specular component alone, with the diffuse removed.",
    "RTI Specular Enhancement": "RTIViewer's specular enhancement, for results comparable with that tool.",
    "RTI Image USM": "RTIViewer's unsharp masking, applied in relit image space.",
    "RBF Relight (Shiny / Translucent)": "Relights from the photographs themselves, interpolated between the lights: gold flares where it was photographed, and as sharp. For gilding, metal, vellum and wax, where PTM and HSH smear the highlights.",
    "Diffuse Gain": "RTI diffuse gain — amplifies shallow relief. Lights from eight sides at the chosen altitude (no single light to drag).",
    "Image Quotient (IQ)": "Divides out the lighting, leaving reflectance. Good for ink against parchment.",
    "Retinex Enhancement": "Evens out illumination across the frame. Rescues unevenly lit captures.",
    "Subsurface Scattering (SSS)": "Simulates light travelling inside the material — waxes, alabaster, skin.",
    "Ambient Occlusion (AO)": "Darkens crevices by how enclosed they are. Reads like a clay model.",
    "Sky-View Factor (SVF)": "How much sky each point can see. Shows shallow hollows a raking light flattens.",
    "Positive Openness": "How exposed each point is. Ridges and raised edges bright.",
    "Negative Openness": "The inverse — incisions, grooves and pits bright.",
    "Local Dominance": "How much each point stands above its surroundings.",
    "Multi-Scale Relief Model (MSRM)": "Relief at several scales combined. Good general-purpose survey view.",
    "Raking Composite": "Several raking directions merged into one image, so no detail hides in shadow.",
    "Multi-Directional Hillshading (Blend / PCA)": "Eight compass directions at once: averaged, or as a PCA false-colour composite where relief running different ways takes different hues.",
    "Ridge and Groove Detector": "Finds incised lines, scratches, tool marks and ruling as lines: raised ones bright, grooves dark, or colour-coded by the direction they run.",
    "Albedo-Suppressed Raking (Clay Mode)": "The surface as unpainted clay: shading only, no colour. Stains, discoloured varnish and pigment cannot be mistaken for form.",
    "PS Normal-Map Lit (Single)": "Hillshade from one light on the PS normals. Classic raking view.",
    "PS Normal-Map Lit (8-Dir)": "Hillshade from eight directions at once — nothing left in shadow.",
    "Light Coverage Map": "Diagnostic: how well your lights covered the surface. Not a picture of the object.",
    "False-Colour Depth": "Height as colour. Read it like a contour map.",
    "PCA 1": "Strongest statistical component across the light stack.",
    "PCA 2": "Second component — often separates ink from substrate.",
    "PCA 3": "Third component — subtle material differences.",
    "Colour ICA (IC1-3)": "Independent components as RGB. Separates overlapping materials.",
    "RGB-PS R-B Diff": "Red minus blue channel. Picks out pigment differences invisible in grey.",
}


# What a view is called on screen, where its engine's name says less.  The
# engine names stay the keys (sessions, exports, the classic app).
VIEW_NAMES: dict[str, str] = {}


def view_name(key: str) -> str:
    """A view's name on screen."""
    return VIEW_NAMES.get(key, key)
