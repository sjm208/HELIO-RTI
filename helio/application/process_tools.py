"""The tools around a solve, GUI-free: shared by the classic Process panel and
the new app's Process workspace (REC-002…004, QC-001…008, ENG-018…025, IMP-004).

Each function is the computation of a classic button, moved out of
RTIViewerPro so both apps run the same code; what stays in each app is the
dialog around it.  Three were changed on the way, each recorded:

* Recalculate Depth (KD-011) now builds the depth exactly as Process does -
  the same clip percentile and the parabolic correction when it is on - so
  re-integrating with the engine you already used gives back the picture
  you had.  It used fixed 1-99% clipping and skipped the parabolic step.
* The Normal QC report (KD-022) rated a capture "Metrological / Research /
  Exploratory" from how far its normals spread around their mean.  That
  spread is the object's shape, not measurement error: a perfectly measured
  coin rates "Exploratory".  The rating now comes from the normals' own
  1-sigma uncertainty (helio.core.processing.normal_uncertainty_deg); the
  spread is still reported, and named for what it is.
* Flat references (IMP-004) are solved in the same light units as the
  capture (FIX-010), or the correction would subtract a gamma artefact.
"""
from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np


# ── Recalculate Depth ────────────────────────────────────────────────────

def smooth_normals(normals, smooth: int) -> np.ndarray:
    """The classic Pre-Integration Smoothing (0 = off, 1-10): an edge-keeping
    bilateral filter on the normals, then back to unit length."""
    n_work = np.asarray(normals, np.float32).copy()
    if smooth and smooth > 0:
        sigma = float(smooth) * 8.0          # 8-80 across the slider range
        n_work = cv2.bilateralFilter(n_work, d=9, sigmaColor=sigma / 127.5, sigmaSpace=sigma)
        mag = np.linalg.norm(n_work, axis=-1, keepdims=True)
        n_work = np.divide(n_work, mag, out=np.zeros_like(n_work), where=mag > 1e-9)
    return n_work


def depth_from_normals(normals, depth_engine: str, *, smooth: int = 0, clip_pct: float = 0.5,
                       parabolic: bool = False, spec_var=None, light_count_conf=None,
                       albedo_rgb=None, mask=None) -> dict:
    """Integrate *normals* into every depth field a result holds, as Process does.

    Returns depth_poisson, depth_vis_p, depth_normalised_full, depth_vis_fc."""
    from helio.application.pipeline import integrate_depth, parabolic_baseline_correction
    from helio.core.processing import depth_visualizations
    n_work = smooth_normals(normals, smooth)
    depth = integrate_depth(n_work, depth_engine, spec_var=spec_var,
                            light_count_conf=light_count_conf, albedo_rgb=albedo_rgb, mask=mask)
    if parabolic:
        depth = parabolic_baseline_correction(depth)
    if mask is not None and np.shape(mask) != np.shape(depth):
        mask = None
    out = depth_visualizations(depth, n_work, clip_pct, mask)
    out["depth_poisson"] = depth
    return out


def with_new_depth(r, depth_engine: str, *, smooth: int = 0, clip_pct: float = 0.5,
                   parabolic: bool | None = None, normals=None):
    """A copy of solve result *r* re-integrated with *depth_engine* (Recalculate Depth).

    The normals are *r*'s unless *normals* is given (Iterative Normal Refine
    integrates refined normals but keeps the solved ones for the views).
    The terrain maps and every depth preview are rebuilt from the new depth."""
    from helio.application.pipeline import proxy_dims, surround_mask
    from helio.core.processing import compute_terrain_analysis
    para = r.settings.parabolic_corr if parabolic is None else parabolic
    f = depth_from_normals(r.normals_full if normals is None else normals, depth_engine,
                           smooth=smooth, clip_pct=clip_pct, parabolic=para, spec_var=r.spec_var,
                           light_count_conf=r.light_count_conf, albedo_rgb=r.albedo_rgb_full,
                           mask=surround_mask(r.settings, r.albedo_rgb_full))
    terrain = compute_terrain_analysis(f["depth_vis_p"], r.normals_full, r.h, r.w)
    ph, pw = proxy_dims(r.h, r.w)
    proxy = dict(r.proxy)
    proxy["depth_p"] = cv2.resize(f["depth_vis_p"], (pw, ph))
    proxy["depth_fc"] = cv2.resize(f["depth_vis_fc"], (pw, ph))
    for k in ("_multi_hs", "_svf", "_msrm", "_curv", "_slope", "_loc_dom", "_pos_opn",
              "_neg_opn", "_slrm"):
        if k in terrain:
            proxy[k] = cv2.resize(terrain[k], (pw, ph))
    return replace(r, settings=replace(r.settings, depth_engine=depth_engine, parabolic_corr=para,
                                       pre_smooth=float(smooth)),
                   depth_poisson=f["depth_poisson"], depth_vis_p=f["depth_vis_p"],
                   depth_normalised_full=f["depth_normalised_full"], depth_vis_fc=f["depth_vis_fc"],
                   terrain=terrain, proxy=proxy)


def with_sharpened(r, params: dict):
    """A copy of solve result *r* through the Detail Sharpening Pipeline
    (helio.application.viewer3d.sharpen): new normals and display depth, and
    their previews; the classic "Apply & Push 3D"."""
    from helio.application.pipeline import proxy_dims
    from helio.application.viewer3d import sharpen
    n, d = sharpen(r.normals_full, r.depth_vis_p, params)
    ph, pw = proxy_dims(r.h, r.w)
    proxy = dict(r.proxy)
    proxy["normals"] = cv2.resize(n, (pw, ph))
    if d is not None:
        proxy["depth_p"] = cv2.resize(d, (pw, ph))
    return replace(r, normals_full=n, depth_vis_p=d if d is not None else r.depth_vis_p, proxy=proxy)


def refine_normals(normals, n_iters: int = 4, ps_weight: float = 0.85) -> np.ndarray:
    """Iterative Normal Refine: normals made more integrable, alternately
    integrating and differentiating, weighted back towards the solve."""
    from helio.core.depth_integration import iterative_normal_integration
    return iterative_normal_integration(normals, n_iters=n_iters, ps_weight=ps_weight)


# ── Flat reference ───────────────────────────────────────────────────────

def flat_reference_normals(paths, lights=None, photo_tone: str = "stored") -> tuple[np.ndarray, dict]:
    """The normal field of a blank flat surface under the capture's lights.

    The systematic distortion of the rig (lamp falloff, lens, lights not
    quite where calibrated) shows up as normals that are not (0, 0, 1);
    Apply flat-field correction later removes it from the object's normals.
    Solved like the capture: decoded to linear light unless the photos are
    RAW or declared linear.  Without the capture's lights, eight lights at
    45° evenly round the compass are assumed (as the classic does)."""
    from helio.core.radiometry import solve_intensities
    from helio.core.frame_read import read_frame
    from helio.io.raw_loader import is_raw
    greys, raw = [], []
    from helio.io.capture import natural_key
    for p in sorted(paths, key=natural_key):      # paired with the capture's lights by position
        img = read_frame(p, cv2.IMREAD_GRAYSCALE, linear=True, sharp=True)
        if img is not None:
            greys.append(img.astype(np.float32))
            raw.append(is_raw(p))
    if len(greys) < 3:
        raise ValueError("Need at least 3 readable flat reference images.")
    h0, w0 = greys[0].shape
    greys = [cv2.resize(g, (w0, h0)) if g.shape != (h0, w0) else g for g in greys]
    from helio.core.tone_curve import capture_curve
    stack = solve_intensities(np.array(greys), photo_tone, raw, capture_curve(paths))
    if lights is not None and len(lights) >= len(greys):
        L = np.asarray(lights, np.float32)[:len(greys)]
    else:
        n = len(greys)
        th = np.linspace(0, 2 * np.pi, n, endpoint=False)
        el = np.deg2rad(45)
        L = np.column_stack([np.cos(el) * np.cos(th), np.cos(el) * np.sin(th),
                             np.full(n, np.sin(el))]).astype(np.float32)
    G = np.linalg.lstsq(L, stack.reshape(len(greys), -1), rcond=None)[0].T
    mag = np.linalg.norm(G, axis=1, keepdims=True)
    flat = np.divide(G, mag, out=np.zeros_like(G), where=mag > 1e-9).reshape(h0, w0, 3).astype(np.float32)
    info = {"n": len(greys), "w": w0, "h": h0,
            "mean_abs_x": float(np.mean(np.abs(flat[..., 0]))),
            "mean_abs_y": float(np.mean(np.abs(flat[..., 1])))}
    return flat, info


# ── Quick Profiles (the Compute panel's four) ────────────────────────────

# label: (normal solver, depth method, pre-integration smoothing, adaptive light weight)
QUICK_PROFILES = {
    "Fine Detail": ("IRLS (Metallic/Specular)", "Gradient-Smoothed Poisson", 2, 0.0,
                    "Shallow incised marks, engraving, fine inscription."),
    "Smooth Surface": ("Linear Least Squares", "Simchony DCT", 6, 0.0,
                       "Smooth, matte, low relief: painted panels, vellum, porcelain."),
    "Heritage Balanced": ("Shadow-Rejection LS", "Fast Poisson (FFT)", 3, 0.2,
                          "A good start for most objects."),
    "Metallic / Specular": ("IRLS (Metallic/Specular)", "Confidence-Weighted Poisson", 1, 0.0,
                            "Metal lettering, seals, gilding, glazed ceramics."),
}


def quick_profile(label: str) -> dict:
    n, d, smooth, adapt, _tip = QUICK_PROFILES[label]
    return dict(normal_engine=n, depth_engine=d, pre_smooth=float(smooth),
                adaptive_light=float(adapt))


# ── Material Presets ─────────────────────────────────────────────────────

# The classic _apply_preset table, in its order and with its values:
# ne normal solver, de depth engine, par parabolic correction, mn even out
# brightness; then the 3D settings: sb shape_blend, hp hp_sigma, wr
# warp_rem, mr micro_relief, ms mesh_smooth, dc depth_clip_pct, ao ao_blend,
# ht height, xt x_tilt, yt y_tilt.
PRESETS = {
    "manuscript": ("Manuscript", dict(ne="Linear Least Squares", de="Simchony DCT", par=True, mn=True,
                   sb=0.9, hp=0.2, wr=0.2, mr=0.5, ms=0.5, dc=1.5, ao=0.3, ht=200., xt=0., yt=0.)),
    "coin": ("Coin / Medal", dict(ne="Sparse-Outlier LS (RPCA)", de="Fast Poisson (FFT)",
             par=False, mn=True, sb=1.0, hp=0.0, wr=0.0, mr=1.5, ms=0.0, dc=0.2, ao=0.5, ht=400.,
             xt=0., yt=0.)),
    "leather": ("Leather Binding", dict(ne="Linear Least Squares", de="Simchony DCT", par=True, mn=True,
                sb=0.6, hp=0.15, wr=0.15, mr=0.8, ms=0.5, dc=1.0, ao=0.4, ht=250., xt=0., yt=0.)),
    "stone": ("Stone Inscription", dict(ne="Shadow-Rejection LS", de="Fast Poisson (FFT)", par=False,
              mn=False, sb=0.4, hp=0.0, wr=0.1, mr=1.2, ms=0.0, dc=0.5, ao=0.6, ht=600., xt=0.,
              yt=0.)),
    "ceramic": ("Glazed Ceramic", dict(ne="Sparse-Outlier LS (RPCA)", de="Fast Poisson (FFT)",
                par=False, mn=True, sb=0.3, hp=0.1, wr=0.1, mr=0.4, ms=1.0, dc=0.5, ao=0.2, ht=300.,
                xt=0., yt=0.)),
    "painting": ("Painting/Panel", dict(ne="Linear Least Squares", de="Fast Poisson (FFT)", par=False,
                 mn=True, sb=0.8, hp=0.0, wr=0.0, mr=0.3, ms=0.5, dc=0.5, ao=0.0, ht=200., xt=0.,
                 yt=0.)),
    "wax_seal": ("Wax Seal", dict(ne="Sparse-Outlier LS (RPCA)", de="Simchony DCT", par=True,
                 mn=True, sb=0.7, hp=0.10, wr=0.05, mr=1.8, ms=0.3, dc=0.5, ao=0.4, ht=350., xt=0.,
                 yt=0.)),
    "textile": ("Textile / Weave", dict(ne="Linear Least Squares", de="Simchony DCT", par=True, mn=True,
                sb=0.95, hp=0.0, wr=0.0, mr=2.0, ms=0.0, dc=2.0, ao=0.2, ht=150., xt=0., yt=0.)),
}
_SHARED_KEYS = (("sb", "shape_blend"), ("hp", "hp_sigma"), ("wr", "warp_rem"),
                ("mr", "micro_relief"), ("ms", "mesh_smooth"), ("dc", "depth_clip_pct"),
                ("ao", "ao_blend"), ("ht", "height"), ("xt", "x_tilt"), ("yt", "y_tilt"))


def preset(name: str):
    """(processing-setting changes, 3D-setting changes, view to show or None) for a profile."""
    _label, p = PRESETS[name]
    # p["mn"] (frame-brightness levelling) is retired: ProcessingSettings.multi_norm
    settings = dict(normal_engine=p["ne"], depth_engine=p["de"], parabolic_corr=bool(p["par"]))
    shared = {sk: float(p[k]) for k, sk in _SHARED_KEYS}
    view = "Specular Enhancement" if name == "wax_seal" else None
    return settings, shared, view


# ── Quality control ──────────────────────────────────────────────────────

def angle_map(n1, n2) -> np.ndarray:
    """Per-pixel angle between two unit normal maps, degrees (the classic _compute_mae_map)."""
    dot = np.clip((n1 * n2).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot)).astype(np.float32)


def lowpass_normals(n, sigma: float) -> np.ndarray:
    lp = np.stack([cv2.GaussianBlur(n[..., c], (0, 0), sigma) for c in range(3)],
                  axis=-1).astype(np.float32)
    mag = np.linalg.norm(lp, axis=-1, keepdims=True)
    return np.divide(lp, mag, out=np.zeros_like(lp), where=mag > 1e-9)


def to_rgb_normals(n) -> np.ndarray:
    return ((np.asarray(n) * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def colour(values, cmap=cv2.COLORMAP_VIRIDIS, vmax=None) -> np.ndarray:
    v = np.asarray(values, np.float32)
    if vmax is None:
        g = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        g = (np.clip(v / max(vmax, 1e-9), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(g, cmap), cv2.COLOR_BGR2RGB)


def bias_diagnostic(normals, sigma: float, reference=None) -> dict:
    """Low-Frequency Bias Diagnostic (Laurent et al. 2025): the normals split
    at scale *sigma* into the broad shape (where calibration error lives) and
    the fine detail; against a reference normal map, the error of each."""
    n = np.asarray(normals, np.float32)
    n_lp = lowpass_normals(n, sigma)
    hp_angle = angle_map(n, n_lp)
    out = dict(sigma=sigma, full_rgb=to_rgb_normals(n), lp_rgb=to_rgb_normals(n_lp),
               hp_rgb=colour(hp_angle), hp_angle=hp_angle,
               hp_mean=float(np.mean(hp_angle)), hp_p95=float(np.percentile(hp_angle, 95)))
    if reference is not None:
        ref_lp = lowpass_normals(reference, sigma)
        mae_overall = float(angle_map(n, reference).mean())
        mae_lp = float(angle_map(n_lp, ref_lp).mean())
        out.update(ref_lp_rgb=to_rgb_normals(ref_lp), mae_overall=mae_overall, mae_lp=mae_lp,
                   mae_hp=max(0.0, mae_overall - mae_lp),
                   verdict=("good calibration: low-frequency bias under 3°" if mae_lp < 3.0 else
                            "moderate low-frequency bias: consider recalibrating" if mae_lp < 6.0
                            else "high low-frequency bias: recalibration strongly recommended"))
    return out


def compare_to_reference(normals, reference, sigma: float) -> dict:
    """Normal Map MAE vs Reference (Laurent et al. 2025, Tables 1-2)."""
    n = np.asarray(normals, np.float32)
    mae_map = angle_map(n, reference)
    mae_lp_map = angle_map(lowpass_normals(n, sigma), lowpass_normals(reference, sigma))
    mae = float(mae_map.mean())
    lp = float(mae_lp_map.mean())
    return dict(mae_map=mae_map, mae_lp_map=mae_lp_map, mean=mae, median=float(np.median(mae_map)),
                p95=float(np.percentile(mae_map, 95)), lp=lp, hp=max(0.0, mae - lp), sigma=sigma,
                heat_rgb=colour(mae_map, cv2.COLORMAP_INFERNO),
                lp_rgb=colour(mae_lp_map, cv2.COLORMAP_PLASMA), ref_rgb=to_rgb_normals(reference),
                verdict=("excellent calibration" if lp < 2.0 else "good calibration" if lp < 3.5
                         else "moderate low-frequency bias" if lp < 7.0
                         else "high low-frequency bias: consider photogrammetric recalibration"))


def readable(albedo, sigma_deg=None):
    """Where the photographs are bright enough to read a surface direction
    from: a black surround or a deep shadow has no normal to speak of, and
    its uncertainty (up to 90°) says nothing about the object's.  None when
    *albedo* is missing or does not match *sigma_deg*."""
    if albedo is None:
        return None
    a = np.asarray(albedo, np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if sigma_deg is not None and a.shape != np.shape(sigma_deg):
        return None
    top = float(np.percentile(a, 99))
    return a >= 0.10 * top if top > 0 else None


def precision_rating(sigma_deg, mask=None) -> tuple[str, float, float]:
    """(rating, median, 95th percentile) of a 1-sigma uncertainty map in
    degrees, over *mask* (helio.application.process_tools.readable) when given."""
    sd = np.asarray(sigma_deg, np.float32)
    if mask is not None and np.count_nonzero(mask) > 100:
        sd = sd[mask]
    med, p95 = float(np.median(sd)), float(np.percentile(sd, 95))
    rating = ("Metrological: the photographs agree to better than 1° (95% under 3°); "
              "the accuracy check measures the error"
              if med < 1.0 and p95 < 3.0 else
              "Research: normals known to better than 3° (95% under 8°)"
              if med < 3.0 and p95 < 8.0 else
              "Exploratory: normals uncertain by more than 3° in places")
    return rating, med, p95


def accuracy_note(acc) -> str:
    """Beside the ranking score: the accuracy check's measured error on a reference, or a
    reminder that the score is not one (the score is worked out from the photographs alone)."""
    if not acc:
        return "a ranking, not an accuracy: Measure › Accuracy Check measures that"
    bits = []
    if acc.get("step_mm") is not None and acc.get("known_mm"):
        bits.append(f"step {acc['step_mm']:.3f} mm against {acc['known_mm']:.3f} "
                    f"({abs(acc.get('error_pct', 0.0)):.1f}% off)")
    if acc.get("flat_rms_mm") is not None:
        bits.append(f"flat to {acc['flat_rms_mm'] * 1000:.1f} µm RMS")
    return "measured on a reference: " + (", ".join(bits) if bits else "checked")


def reference_leans(r) -> list[tuple[str, float]]:
    """How far each thing marked flat leans in solve *r*, in degrees: the
    grey patch and every flat area, read from the normals (flat is 0)."""
    from helio.core.flash import parse_box, parse_boxes
    ps = r.settings
    boxes = []
    if ps.flash_ref and parse_box(ps.flash_ref):
        boxes.append(("grey patch", parse_box(ps.flash_ref)))
    for i, b in enumerate(parse_boxes(ps.flat_areas) if ps.flat_areas else []):
        boxes.append((f"flat area {i + 1}", b))
    ox, oy = (r.crop[0], r.crop[1]) if r.crop else (0, 0)
    n = np.asarray(r.normals_full, np.float32)
    h, w = n.shape[:2]
    out = []
    for name, (x1, y1, x2, y2) in boxes:
        x1, x2 = max(0, x1 - ox), min(w, x2 - ox)
        y1, y2 = max(0, y1 - oy), min(h, y2 - oy)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        m = np.median(n[y1:y2, x1:x2].reshape(-1, 3), axis=0)
        k = float(np.linalg.norm(m))
        if np.isfinite(k) and k > 0:
            out.append((name, float(np.degrees(np.arccos(np.clip(abs(m[2]) / k, 0, 1))))))
    return out


SCORE_WORDS = ((80, "Very good"), (65, "Good"), (50, "Fair"), (0, "Weak"))


DEPTH_FULL, DEPTH_NONE = 4.0, 20.0     # degrees: the depth's slopes against the normals


def depth_agreement_of(r) -> float | None:
    """How far solve *r*'s depth disagrees with its own normals, in degrees
    (helio.core.object_mask.depth_agreement), over the object when it stands on
    a dark surround, else over what is bright enough to read."""
    from helio.core.object_mask import depth_agreement, object_mask
    n, d = getattr(r, "normals_full", None), getattr(r, "depth_poisson", None)
    if n is None or d is None:
        return None
    a = getattr(r, "albedo_rgb_full", None)
    m = object_mask(a) if a is not None and np.shape(a)[:2] == np.shape(d) else None
    if m is None:
        g = getattr(r, "albedo_grey_full", None)
        m = readable(g, d) if g is not None else None
    try:
        return depth_agreement(n, d, m)
    except Exception:                                    # noqa: BLE001 - the score goes without it
        return None


def overall_score(med: float, p95: float, cond: float | None = None,
                  leans: list | None = None, depth_deg: float | None = None) -> dict:
    """One figure out of 100 for a solve, to choose between quick looks.

    Sureness (60): how well the photographs agree about each direction,
    from the normal uncertainty's median and 95th percentile (1 degree
    and 3 full marks, about 8 none).  Lights (20): the light
    conditioning (2 or under full marks, 6 none).  Flatness (20): how far
    the worst thing marked flat leans (1 degree full marks, 10 none), only
    when something is marked.  Depth (20): how far the integrated depth's
    own slopes disagree with the normals (helio.core.object_mask.depth_agreement,
    DEPTH_FULL degrees full marks, DEPTH_NONE none), when *depth_deg* is
    given: the one part that tells solvers apart, since Sureness is measured
    on the photographs, not on the solver's answer.  What is not measured is
    left out and the rest scaled to 100.
    Returns {score, word, parts: [(name, points, out_of)]}."""
    e = 0.6 * float(med) + 0.4 * float(p95) / 3.0               # p95/3 is on the median's scale
    sure = 60.0 * float(np.clip((8.0 - e) / 7.0, 0.0, 1.0))
    parts = [("Sureness", sure, 60)]
    if cond is not None and np.isfinite(cond):
        parts.append(("Lights", 20.0 * float(np.clip((6.0 - cond) / 4.0, 0.0, 1.0)), 20))
    if leans:
        worst = max(v for _n, v in leans)
        parts.append(("Flatness", 20.0 * float(np.clip((10.0 - worst) / 9.0, 0.0, 1.0)), 20))
    if depth_deg is not None and np.isfinite(depth_deg):
        parts.append(("Depth", 20.0 * float(np.clip((DEPTH_NONE - depth_deg) / (DEPTH_NONE - DEPTH_FULL),
                                                    0.0, 1.0)), 20))
    got, of = sum(p for _n, p, _o in parts), sum(o for _n, _p, o in parts)
    score = int(round(100.0 * got / of)) if of else 0
    word = next(w for t, w in SCORE_WORDS if score >= t)
    return dict(score=score, word=word, parts=[(n, int(round(p)), o) for n, p, o in parts])


SOLVER_SHORT = {"Sparse-Outlier LS (RPCA)": "shine-resistant solver",
                "Mid-Tone Weighted LS": "mid-tone solver",
                "IRLS (Metallic/Specular)": "metal solver",
                "Iterative Specular Separation": "shine-separating solver",
                "Shadow-Rejection LS": "shadow-rejecting solver",
                "Uncalibrated PCA": "uncalibrated solver"}


def settings_summary(ps) -> str:
    """The choices that set one solve apart from another, in a few words."""
    on = [w for k, w in (("even_flat", "flat areas"), ("even_flash", "grey patch"),
                         ("flatten_illumination", "flat object"), ("near_field", "near-field"),
                         ("sfs_refine", "refined"), ("pre_clahe", "CLAHE"))
          if getattr(ps, k, False)]
    solver = SOLVER_SHORT.get(getattr(ps, "normal_engine", ""), "")
    if solver:
        on.append(solver)
    return ", ".join(on) if on else "no corrections"


def try_variants(ps, has_ball: bool) -> list[tuple[str, dict]]:
    """What Try a few runs: the settings as they are, and each one change
    worth trying from them (a correction alone, near-field when the ball's
    place is known, the shine- and shadow-resistant solvers), without
    repeating one already covered.  [(name, settings to change)]."""
    off = dict(even_flash=False, even_flat=False, flatten_illumination=False, near_field=False)
    out = [("as set", {}), ("no corrections", dict(off)),
           ("flat object", dict(off, flatten_illumination=True))]
    if ps.flash_ref:
        out.append(("grey patch", dict(off, even_flash=True)))
    if ps.flat_areas:
        out.append(("flat areas", dict(off, even_flat=True)))
    if has_ball:
        out.append(("near-field", dict(off, near_field=True)))
    for solver in ("Sparse-Outlier LS (RPCA)", "Shadow-Rejection LS"):
        if ps.normal_engine != solver:
            out.append((SOLVER_SHORT[solver], dict(normal_engine=solver)))
    seen, kept = [], []
    for name, kw in out:
        key = tuple(sorted((k, getattr(ps, k) if k not in kw else kw[k])
                           for k in set(kw) | set(off) | {"normal_engine"}))
        if key not in seen:
            seen.append(key)
            kept.append((name, kw))
    return kept


def normal_qc(normals, sigma: float, sigma_deg=None, flat_ref=None) -> dict:
    """The Normal QC report, honest about what it measures (KD-022).

    *Precision* is the normals' own 1-sigma uncertainty, from each pixel's
    lights (sigma_deg); its median and 95th percentile give the rating:
    under 1° and 3° metrological, under 3° and 8° research, else exploratory.
    *Surface variation* is the classic figure - the spread of the normals
    about their mean, split into broad and fine at *sigma* - reported for
    what it is, the object's relief.  With a flat reference applied, the
    deviation from the corrected flat is a true error, and is reported."""
    n = np.asarray(normals, np.float32)
    h, w = n.shape[:2]
    mean_n = np.mean(n.reshape(-1, 3), axis=0)
    mean_n /= (np.linalg.norm(mean_n) + 1e-9)
    dev = np.degrees(np.arccos(np.clip(np.einsum('hwc,c->hw', n, mean_n), -1.0, 1.0)))
    lp = cv2.GaussianBlur(dev.astype(np.float32), (0, 0), sigmaX=sigma)
    hp = np.abs(dev - lp)
    out = dict(sigma=sigma, variation=dev, variation_lp=lp,
               variation_stats={"mean": float(dev.mean()), "median": float(np.median(dev)),
                                "p95": float(np.percentile(dev, 95)),
                                "broad": float(lp.mean()), "fine": float(hp.mean())})
    if flat_ref is not None:
        flat = cv2.resize(flat_ref, (w, h))
        flat_ang = np.degrees(np.arccos(np.clip(flat[..., 2], -1.0, 1.0)))
        resid = np.abs(dev - flat_ang)
        out["flat_residual"] = {"mean": float(resid.mean()), "p95": float(np.percentile(resid, 95))}
    if sigma_deg is not None:
        sd = np.asarray(sigma_deg, np.float32)
        rating, med, p95 = precision_rating(sd)
        out.update(precision=sd, precision_stats={"median": med, "p95": p95,
                                                   "over_1": float(np.mean(sd > 1.0) * 100),
                                                   "over_3": float(np.mean(sd > 3.0) * 100)},
                   rating=rating)
    else:
        out["rating"] = ("No precision rating: it needs calibrated lights and at least four "
                         "frames. The figures below describe the surface, not the error.")
    return out


def light_dome_figure(L, size=(560, 560), dark=True) -> np.ndarray:
    """The classic View 3D Light Dome as an RGB image: every light on a wire
    hemisphere, coloured by height, seen from 30° up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    bg = "#10162A" if dark else "#FBF8F1"
    fg = "#ECE7DA" if dark else "#1F2433"
    fig = plt.Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor=bg)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg)
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi / 2:15j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="#6CC4F0", alpha=0.15)
    L = np.asarray(L, np.float64)
    L = L[np.isfinite(L).all(axis=1)]
    ax.scatter(L[:, 0], L[:, 1], L[:, 2], c=L[:, 2], cmap='plasma', s=50, edgecolors='black',
               depthshade=True)
    ax.scatter([0], [0], [0], color='red', s=100, marker='x')
    ax.set_xlabel('X (east)', color=fg)
    ax.set_ylabel('Y (north)', color=fg)
    ax.set_zlabel('Z (up)', color=fg)
    ax.tick_params(colors=fg)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])
    ax.view_init(elev=30, azim=45)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    return img
