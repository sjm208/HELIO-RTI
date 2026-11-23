"""What each view can be adjusted by, GUI-free (VIEW-002).

The classic Inspect panel decides which sliders to show by matching engine
names (``RTIViewerPro.update_dynamic_sliders``).  Several of those names no
longer match any engine ("SVF", "Openness +", "MSRM"), so some sliders never
appear; others appear for views that never read them ("HS Altitude" on
Local Dominance, "Residual Gain" on Surface Curvature), and three engines
read settings no slider offers (Exaggerated Shading, Curvedness,
Cast-Shadow Raking).  A control that moves nothing reads as a broken
control, and a setting nobody can reach is a missing one.

So this table is the measured answer instead: for every view, the settings
that change its picture.  ``tests/test_view_controls.py`` renders each view with
each of its controls at both ends of the range and fails if one does not
move the picture, and fails if a setting it does not list does.  Keys,
ranges, defaults and wording are the classic ones (``slider_configs`` in
Helio_Pro.py); where the classic app has no slider the range is new and
the default is the engine's own.

The light is screen frame (+Y down), as the classic dome stores it, and is
kept per view in ``engine_settings[view]['lx', 'ly', 'lz']`` with the
compass angles ``hs_azi``/``hs_alt`` beside it - the same keys, written the
same way, as ``RTIViewerPro.update_light``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Control:
    key: str
    label: str
    lo: float
    hi: float
    default: float
    step: float = 0.01
    tip: str = ""
    choices: tuple = ()          # (value, label) pairs: a choice, not a slider
    unit: str = ""

    @property
    def integer(self) -> bool:
        return self.step >= 1 and not self.choices

    @property
    def decimals(self) -> int:
        if self.integer or self.choices:
            return 0
        return 2 if self.step < 0.1 else 1


def _c(key, label, lo, hi, default, step=0.01, tip="", unit="", choices=()):
    return Control(key, label, lo, hi, default, step, tip, tuple(choices), unit)


CONTROLS: dict[str, Control] = {c.key: c for c in (
    # The classic "Analysis Adjustments", in its order and words.
    _c("gain", "Exposure / gain", 0.1, 5.0, 1.0, 0.01,
       "Intensity multiplier for the current view."),
    _c("bright", "Brightness", 0.4, 1.8, 1.0, 0.01, "Overall brightness adjustment."),
    _c("contrast", "Contrast", 0, 50, 0, 1, "Contrast enhancement (0 = none)."),
    _c("gamma", "Gamma", 0.1, 3.0, 1.0, 0.01,
       "Gamma curve. 1.0 = linear, below 1 lightens the midtones, above 1 darkens them."),
    _c("micro", "Micro-detail boost", 0.0, 2.0, 0.0, 0.01, "Adds texture detail from the albedo."),
    _c("rough_opacity", "Roughness opacity", 0.0, 1.0, 0.5, 0.01,
       "How much of the roughness map is blended in."),
    _c("hs_amb", "Ambient", 0.0, 1.0, 0.3, 0.01,
       "Minimum brightness, so shadows never go fully black."),
    _c("hs_alt", "Light height", 5, 90, 45, 1, "Elevation of the light above the horizon.", "°"),
    _c("mr_ambient", "Ambient", 0.0, 1.0, 0.18, 0.01,
       "Light from everywhere: how bright the side facing away from the lamp stays. "
       "As the 3D relief's Ambient."),
    _c("mr_diffuse", "Diffuse", 0.0, 2.0, 0.95, 0.01,
       "The lamp's light on the colour, by how squarely each point faces it. "
       "As the 3D relief's Diffuse."),
    _c("mr_shine", "Shine", 0.0, 1.0, 0.12, 0.01,
       "A soft sheen where the surface turns the lamp towards the camera, in the "
       "material's own colour. 0 is fully matte. As the 3D relief's Shine."),
    _c("mr_shininess", "Shininess", 2, 128, 24, 1,
       "How tight the sheen is: low spreads it over the slopes, high keeps it to small "
       "glints. As the 3D relief's Shininess."),
    _c("shape_smooth", "Pre-smooth", 0.0, 8.0, 1.5, 0.5,
       "Smoothing before the curvature is measured. Higher = smoother, loses fine detail."),
    _c("bil_d", "Bilateral diameter", 3, 21, 9, 1,
       "Neighbourhood for the bilateral filter. Larger = stronger smoothing on flat areas.", "px"),
    _c("bil_sc", "Bilateral sigma", 0.01, 0.5, 0.1, 0.01,
       "Edge sensitivity. Low = only smooth areas are blurred; high = blur across edges too."),
    _c("ret_scale_s", "Local contrast scale S", 1, 40, 8, 1,
       "Primary neighbourhood, as % of image height. 1% = fine detail, 20% = broad illumination.",
       "%"),
    _c("ret_scale_m", "Local contrast scale M", 5, 80, 30, 1,
       "Secondary neighbourhood (% of image height), blended with scale S. Keep above S.", "%"),
    _c("ret_scale_l", "S/M blend", 0, 500, 250, 1, "0 = scale S only (fine), 500 = scale M only (broad)."),
    _c("rk_elevation", "Raking elevation", 5, 45, 20, 1,
       "Height of the raking lights. 5° = extreme grazing, 20° = standard, 45° = soft.", "°"),
    _c("rk_n_dirs", "Directions", 2, 8, 4, 1,
       "Lights evenly round 360°. 2 = east and west, 4 = cardinal, 8 = smoothest."),
    _c("rk_usm", "Sharpen", 0.0, 3.0, 0.0, 0.05,
       "Unsharp mask on the composite. 0 = off."),
    _c("mdh_alt", "Light altitude", 5, 80, 35, 1,
       "Altitude of all eight lights. Low = raking (fine relief), high = flatter (broad form).", "°"),
    _c("mdh_mode", "Show", 0, 1, 0, 1,
       "Blend averages the eight directions into one grey picture. PCA composite is false colour: "
       "red is the relief all directions share, green and blue how the shading changes with direction.",
       choices=((0, "8-direction blend"), (1, "PCA composite (false colour)"))),
    _c("pbr_expose", "Light height", 0, 1, 0, 1,
       "These views raise the lighting to a power (1 / gain), so as the classic app draws them they go dark as "
       "the light is lowered. Steady lifts a low light so a flat area is never darker than a mid grey and the "
       "relief stays readable, with the highlights eased off instead of clipped; a high light, and an overhead "
       "one, is drawn exactly as the classic app does.",
       choices=((0, "As the classic app"), (1, "Keep brightness steady"))),
    _c("rdg_width", "Line width", 0.5, 12.0, 1.5, 0.1,
       "The thinnest line looked for, in pixels of the full-size picture (the same at any zoom). "
       "Two wider ones are tried as well. Raise it for broad grooves; keep it low for fine scratches.", "px"),
    _c("rdg_mode", "Show", 0, 3, 0, 1,
       "Both: raised lines bright, grooves dark. Grooves or Ridges: only those, bright on black. "
       "Direction: false colour, the hue is the way each line runs and the brightness how strong it is.",
       choices=((0, "Ridges and grooves"), (1, "Grooves only"), (2, "Ridges only"),
                (3, "Direction (false colour)"))),
    _c("clay_floor", "Shadow floor", 0.0, 0.4, 0.06, 0.01,
       "The darkest any pixel may be. Raise it to keep micro-roughness in deep shadow from going black."),
    _c("clay_micro", "Micro-roughness", 0.0, 3.0, 0.6, 0.05,
       "Boosts the shading's finest detail (about 0.3% of the long side). 0 = plain Lambert shading."),
    _c("op_radius", "Search radius", 5, 200, 20, 1,
       "Openness search radius in pixels. Smaller = fine features, larger = broad terrain.", "px"),
    _c("op_smooth", "Pre-smooth", 0.0, 8.0, 0.0, 0.5,
       "Smooths the surface before openness is measured, in pixels. Raise it to quieten paper "
       "grain and noise; 0 keeps every detail.", "px"),
    _c("op_clip", "Range clip", 0.0, 10.0, 0.0, 0.5,
       "Stretches the picture between this percentage at each end instead of the extremes, so "
       "one deep hole or the object's rim no longer leaves the surface a flat grey.", "%"),
    _c("op_mode", "Show", 0, 1, 0, 1,
       "This openness alone, or positive minus negative openness: ridges and raised strokes "
       "bright, grooves and hollows dark, in one picture.",
       choices=((0, "This openness"), (1, "Ridges and hollows (difference)"))),
    _c("slope_smooth", "Pre-smooth", 0.0, 8.0, 0.0, 0.5,
       "Smooths the surface before the slope is measured, in pixels.", "px"),
    _c("slope_clip", "Range clip", 0.0, 10.0, 0.0, 0.5,
       "Stretches between this percentage at each end instead of the steepest edge, so the "
       "object's rim does not leave the surface dark.", "%"),
    _c("curv_smooth", "Pre-smooth", 0.0, 8.0, 0.0, 0.5,
       "Smooths the normals before curvature is measured, in pixels. Raise it to keep only "
       "real scratches and lose the grain.", "px"),
    _c("msrm_clip", "Range clip", 0.0, 10.0, 2.0, 0.5,
       "Stretches between this percentage at each end. Lower keeps the strongest relief from "
       "clipping; higher brings up faint relief.", "%"),
    _c("msrm_min", "Smallest feature", 1, 20, 1, 1,
       "The finest relief kept, in pixels. Raise it to quieten paper grain and noise.", "px"),
    _c("msrm_max", "Largest feature", 2, 60, 5, 1,
       "The broadest relief kept, in pixels. Raise it to bring out wide tooling and cockling.",
       "px"),
    _c("svf_dirs", "Directions", 4, 32, 16, 1, "Search directions. 8 = fast, 16 = standard, 32 = finest."),
    _c("psr_floor", "Residual floor", 0.0, 0.9, 0.05, 0.01,
       "Residual treated as zero. 0 = show everything, 0.1–0.2 = only the real outliers."),
    _c("psr_gain", "Residual gain", 0.2, 6.0, 1.5, 0.05,
       "Brightness after the floor. If the frame goes white, raise the floor instead."),
    _c("psr_gamma", "Residual gamma", 0.2, 2.5, 1.0, 0.05,
       "Below 1 lifts faint residuals, above 1 keeps only the strongest."),
    _c("unc_max", "Scale top", 1.0, 45.0, 10.0, 0.5,
       "The uncertainty shown brightest. Keep it fixed to compare captures.", "°"),
    _c("ao_radius", "Occlusion radius", 0.01, 0.5, 0.08, 0.01,
       "Sampling radius, as a fraction of the image size."),
    _c("ao_strength", "Occlusion strength", 0.1, 5.0, 1.5, 0.05,
       "Multiplier for the computed occlusion."),
    _c("dg_alt", "Light altitude", 5, 80, 35, 1,
       "Altitude of the eight lights it averages. Low = raking (fine, shallow relief), high = "
       "flatter (broad form). 35° is the long-standing default.", "°"),
    _c("dg_strength", "Diffuse gain strength", 0.5, 8.0, 2.0, 0.05,
       "Gain on the diffuse (N·L) signal. 2× = standard, 4–6× = strong for paper and parchment."),
    _c("dg_gamma", "Diffuse gamma", 0.2, 2.0, 0.7, 0.01,
       "Gamma after the gain. 0.4–0.7 lifts shadows to reveal faint marks; 1.0 = linear."),
    _c("dg_spec", "Specular suppress", 0.0, 1.0, 0.3, 0.01,
       "Subtracts the estimated shine before the gain. 0.3 = semi-gloss, 0.8–1 = vellum, leather."),
    _c("rbf_width", "Smoothness between lights", 0.4, 3.0, 1.2, 0.05,
       "How far each photograph's influence reaches, in light spacings. Low keeps highlights "
       "sharp but can show steps between lights; high blends them smoothly."),
    _c("rbf_smooth", "Fit tolerance", 0.0, 0.5, 0.05, 0.01,
       "0 passes through every photograph exactly; higher smooths noise and flicker between "
       "lights at some loss of sparkle."),
    _c("rti_spec_coeff", "Specular coefficient", 0.2, 5.0, 1.5, 0.05,
       "RTIViewer's specular enhancement level. 1 = no change, 2–3 = moderate, 4–5 = strong."),
    _c("rti_spec_floor", "Specular floor", 0.0, 0.5, 0.05, 0.01,
       "Specular response below this is suppressed. Raise to 0.1–0.2 against noise."),
    _c("rti_spec_blend", "Specular / diffuse blend", 0.0, 1.0, 0.6, 0.01,
       "0 = pure diffuse, 0.6 = recommended, 1 = specular only."),
    _c("rti_usm_sigma", "Sharpen radius", 0.5, 8.0, 2.0, 0.1,
       "Blur radius of the unsharp mask, in pixels: small for scratches, large for toolmarks.", "px"),
    _c("rti_usm_amount", "Sharpen amount", 0.1, 5.0, 1.5, 0.05,
       "Unsharp mask strength. 1 = moderate, 2–3 = strong."),
    _c("rti_usm_thresh", "Sharpen threshold", 0.0, 0.3, 0.0, 0.01,
       "Minimum edge contrast before sharpening. 0 = sharpen everything, noise included."),
    _c("nr_scale", "Detail scale", 0.5, 20.0, 5.0, 0.1,
       "Smoothing radius as % of image height: what is removed before the residual is shown. "
       "5 for stylus marks, lower for hairline scratches.", "%"),
    _c("nr_mode", "Show", 0, 2, 0, 1,
       "Greyscale is best for scratches; false colour separates raised (red) from pressed "
       "(blue); absolute shows all relief bright.",
       choices=((0, "Greyscale"), (1, "False colour: raised / pressed"), (2, "Absolute"))),
    _c("def_sens", "Specular sensitivity", 0.1, 5.0, 1.0, 0.05,
       "Sensitivity of the specular highlight extraction."),
    _c("def_relief", "Specular relief", 0.0, 3.0, 1.0, 0.05,
       "Relief exaggeration for the specular surface view."),
    _c("def_noise", "Specular noise floor", 0.0, 1.0, 0.5, 0.01,
       "Raise to suppress low-frequency noise."),
    _c("def_spec", "Specular blend", 0.0, 1.0, 0.0, 0.01,
       "Blend the specular component back in. 0 = pure quotient, 1 = full specular."),
    # Read by the engines but never offered by the classic app.
    _c("sss_radius", "Scattering radius", 1.0, 30.0, 8.0, 0.5,
       "How far light travels under the surface, in pixels (scaled by the gain).", "px"),
    _c("sss_blend", "Scattering blend", 0.0, 1.0, 0.45, 0.01,
       "Share of the scattered light in the result."),
    _c("exag_scales", "Scales", 1, 8, 4, 1, "How many scales are lit at once."),
    _c("exag_flatten", "Flatten", 0.0, 200.0, 0.0, 5.0,
       "Take out the broad shape (a page's curve, cockling, a warped board) over this share of "
       "the picture, in thousandths of its long side, so only the surface's texture is shaded. "
       "0 keeps the whole shape."),
    _c("exag_sigma", "Finest scale", 0.5, 8.0, 2.0, 0.1,
       "Smoothing of the finest scale, in pixels; each further scale doubles it.", "px"),
    _c("curvedness_smooth", "Pre-smooth", 0.5, 8.0, 1.5, 0.1,
       "Smoothing before the curvature is measured.", "px"),
    _c("shadow_relief", "Relief", 0.01, 0.3, 0.08, 0.01,
       "Height of the surface, as a fraction of its width: how long the shadows are."),
    _c("shadow_steps", "Shadow steps", 8, 128, 48, 1,
       "Samples along each shadow ray: more = smoother, slower."),
    _c("shadow_ambient", "Shadow fill", 0.0, 0.6, 0.18, 0.01,
       "Light left inside a shadow, so it never goes fully black."),
)}

# The colour-strength slider of the Combined Colour layer is one window
# setting in the classic app (colour_opacity_var), not a per-view one.
COLOUR_GAIN = _c("colour_gain", "Colour brightness", 0.5, 4.0, 1.5, 0.05,
                 "Brightens the Colour layer only: a dark binding's colour, without "
                 "washing out Clay, Roughness or Confidence (they keep the view's gain).")
COLOUR_OPACITY = _c("colour_opacity", "Colour strength", 0.0, 1.0, 1.0, 0.01,
                    "0 = shape only, 1 = the photograph's colour.")

# Every view's own settings, in the order the panel lists them.  The tone
# controls (TONE) and the light come on top, where the view takes them.
VIEW_SETTINGS: dict[str, tuple[str, ...]] = {
    "Standard PBR": ("gain", "pbr_expose"),
    "Matte Relight (as 3D)": ("mr_ambient", "mr_diffuse", "mr_shine", "mr_shininess"),
    "Robust PBR (MVS)": ("gain", "pbr_expose"),
    "Normals Unsharp Masking": ("gain", "nr_scale"),
    "Exaggerated Shading": ("gain", "exag_scales", "exag_sigma", "exag_flatten"),
    "Curvedness": ("gain", "curvedness_smooth"),
    "Cast-Shadow Raking": ("gain", "shadow_relief", "shadow_steps", "shadow_ambient"),
    "Multi-Light Fusion": ("gain",),
    "PS Fit Residual (QC)": ("gain", "psr_floor", "psr_gain", "psr_gamma"),
    "Normal Uncertainty (QC)": ("unc_max",),
    "Slope Magnitude": ("gain",),
    "Specular Enhancement": ("gain", "dg_gamma"),
    "Image Quotient (IQ)": ("gain", "def_spec"),
    "Positive Openness": ("op_mode", "op_radius", "svf_dirs", "op_smooth", "op_clip"),
    "Negative Openness": ("op_mode", "op_radius", "svf_dirs", "op_smooth", "op_clip"),
    "Multi-Scale Relief Model (MSRM)": ("msrm_min", "msrm_max", "msrm_clip"),
    "Specular View": ("gain", "def_sens", "def_relief", "def_noise"),
    "Ambient Occlusion (AO)": ("gain", "ao_radius", "ao_strength"),
    "PS Normal-Map Lit (Single)": ("hs_amb",),
    "PS Normal-Map Lit (8-Dir)": ("hs_alt", "hs_amb"),
    "Slope Gradient": ("slope_smooth", "slope_clip"),
    "Surface Curvature (Scratches)": ("gain", "curv_smooth"),
    "PCA 1": ("gain",),
    "PCA 2": ("gain",),
    "PCA 3": ("gain",),
    "Light Coverage Map": ("gain",),
    "Normals RGB": ("gain",),
    "Shape Index": ("gain", "shape_smooth"),
    "Retinex Enhancement": ("gain", "ret_scale_s", "ret_scale_m", "ret_scale_l"),
    "Colour ICA (IC1-3)": ("gain",),
    "RGB-PS R-B Diff": ("gain",),
    "Raking Composite": ("gain", "rk_elevation", "rk_n_dirs", "rk_usm"),
    "Multi-Directional Hillshading (Blend / PCA)": ("gain", "mdh_alt", "mdh_mode"),
    "Albedo-Suppressed Raking (Clay Mode)": ("gain", "clay_floor", "clay_micro"),
    "Ridge and Groove Detector": ("gain", "rdg_width", "rdg_mode"),
    "Normal Residual": ("nr_mode", "nr_scale", "gain"),
    "Diffuse Gain": ("gain", "dg_alt", "dg_strength", "dg_gamma", "dg_spec"),
    "Bilateral Normal Sharpening": ("gain", "bil_d", "bil_sc"),
    "Subsurface Scattering (SSS)": ("gain", "sss_radius", "sss_blend"),
    "RTI Specular Enhancement": ("gain", "rti_spec_coeff", "rti_spec_floor", "rti_spec_blend"),
    "RTI Image USM": ("gain", "rti_usm_sigma", "rti_usm_amount", "rti_usm_thresh"),
    "RBF Relight (Shiny / Translucent)": ("gain", "rbf_width", "rbf_smooth"),
}

# Settings the classic panel shows for a view although they cannot change
# it, with the reason - so the omission is a decision on record, and the
# test can prove each one really is inert.
INERT: dict[str, dict[str, str]] = {
    "PS Normal-Map Lit (Single)": {
        "hs_z": "applies only to the depth fallback; this view lights the solved normals",
        "hs_depth_src": "as hs_z: the depth source is unused while normals exist"},
    "PS Normal-Map Lit (8-Dir)": {
        "hs_z": "applies only to the depth fallback; this view lights the solved normals",
        "hs_depth_src": "the classic choices (Poisson, Frankot) leave the normals unchanged"},
    "Normal Residual": {"nr_gain": "the residual is rescaled to its own percentiles afterwards"},
    "Surface Curvature (Scratches)": {"nr_gain": "the engine reads gain, not nr_gain"},
    "Slope Gradient": {"nr_gain": "the engine has no gain"},
    "Colour ICA (IC1-3)": {"nr_gain": "the engine has no settings"},
    "RGB-PS R-B Diff": {"nr_gain": "the engine has no settings"},
    "Shape Index": {"si_sigma": "the engine reads shape_smooth, not si_sigma"},
    "Positive Openness": {"op_elev": "openness is computed without an elevation threshold"},
    "Negative Openness": {"op_elev": "openness is computed without an elevation threshold"},
}

TONE = ("bright", "contrast", "gamma", "micro")

# Views whose picture is colour already: the tone controls and the
# colour/clay layer do not apply (render.composite returns them as they are).
RGB_VIEWS = frozenset({"Normals RGB", "Colour ICA (IC1-3)", "Normal Uncertainty (QC)",
                       "Albedo-Suppressed Raking (Clay Mode)"})

# How a view takes the light.  "dome": lx, ly, lz.  "compass": the compass
# angles hs_azi and hs_alt.  "elevation": hs_alt alone, from every side.
LIGHT_KIND: dict[str, str] = {
    **{e: "dome" for e in (
        "Standard PBR", "Matte Relight (as 3D)", "Robust PBR (MVS)", "Normals Unsharp Masking",
        "Exaggerated Shading",
        "Cast-Shadow Raking", "Specular Enhancement", "Image Quotient (IQ)",
        "Bilateral Normal Sharpening", "Subsurface Scattering (SSS)",
        "RTI Specular Enhancement", "RTI Image USM", "Albedo-Suppressed Raking (Clay Mode)",
        "RBF Relight (Shiny / Translucent)")},
    "PS Normal-Map Lit (Single)": "compass",
}


def light_kind(view: str) -> str | None:
    """'dome', 'compass' or None: whether moving the light changes *view*."""
    return LIGHT_KIND.get(view)


def controls_for(view: str, settings: dict | None = None) -> list[Control]:
    """The view's own controls, in panel order.

    Normal Residual's gain only acts on its colour and absolute modes; it
    is left out of greyscale, where the residual is rescaled afterwards."""
    from dataclasses import replace
    from helio.domain.registry import VIEW_DEFAULTS
    keys = list(VIEW_SETTINGS.get(view, ()))
    if view == "Normal Residual" and int((settings or {}).get("nr_mode", 0)) == 0:
        keys = [k for k in keys if k != "gain"]
    own = VIEW_DEFAULTS.get(view, {})
    # the view's own default, so the slider's reset and its changed-mark agree
    return [replace(CONTROLS[k], default=own[k]) if k in own else CONTROLS[k] for k in keys]


def takes_tone(view: str, settings: dict | None = None) -> bool:
    """Whether Brightness, Contrast, Gamma and Micro-detail change *view*."""
    if view in RGB_VIEWS:
        return False
    if view == "Normal Residual" and int((settings or {}).get("nr_mode", 0)) == 1:
        return False
    if view == "Multi-Directional Hillshading (Blend / PCA)" and int((settings or {}).get("mdh_mode", 0)) == 1:
        return False                                       # the PCA composite is false colour
    if view == "Ridge and Groove Detector" and int((settings or {}).get("rdg_mode", 0)) == 3:
        return False                                       # the direction picture is false colour
    return True


def value(settings: dict, key: str):
    """A setting as the engine sees it: stored, else the control's default."""
    c = CONTROLS.get(key)
    v = settings.get(key, c.default if c else None)
    return v


# ── The light, as the classic dome writes it ───────────────────────────

def angles(lx: float, ly: float, lz: float) -> tuple[float, float]:
    """(azimuth from north, clockwise; elevation) in degrees, of a screen-frame light."""
    az = math.degrees(math.atan2(lx, -ly)) % 360
    el = math.degrees(math.asin(max(0.0, min(1.0, lz))))
    return az, el


def from_angles(az_deg: float, el_deg: float) -> tuple[float, float, float]:
    """The screen-frame light at a compass azimuth and elevation (``_set_light_preset``)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    return math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)


def from_dome(dx: float, dy: float) -> tuple[float, float, float]:
    """The light for a point on the dome, unit radius, +Y down (``update_light``)."""
    d = math.hypot(dx, dy)
    if d > 1.0:
        dx, dy = dx / d, dy / d
    return dx, dy, math.sqrt(max(0.001, 1.0 - dx * dx - dy * dy))


def light_settings(lx: float, ly: float, lz: float) -> dict:
    """Every key the classic dome writes for one light: lx, ly, lz, hs_azi, hs_alt."""
    az, el = angles(lx, ly, lz)
    return dict(lx=float(lx), ly=float(ly), lz=float(lz), hs_azi=az, hs_alt=el)


def light_of(settings: dict, view: str | None = None) -> tuple[float, float, float]:
    """The light a view is drawn with, in screen frame.

    A compass view reads its angles, so its dome shows those; any other view
    reads lx, ly, lz (``on_engine_change``)."""
    if view is not None and light_kind(view) == "compass":
        return from_angles(float(settings.get("hs_azi", 315)), float(settings.get("hs_alt", 45)))
    return (float(settings.get("lx", 0.0)), float(settings.get("ly", 0.0)),
            float(settings.get("lz", 1.0)))


def at_elevation(lx: float, ly: float, el_deg: float) -> tuple[float, float, float]:
    """The light moved to *el_deg*, keeping its azimuth (``_on_elevation_slider``)."""
    mag = math.hypot(lx, ly)
    if mag < 0.001:
        lx, ly, mag = 0.0, 0.707, 0.707
    e = math.radians(el_deg)
    s = math.cos(e) / mag
    return lx * s, ly * s, math.sin(e)
