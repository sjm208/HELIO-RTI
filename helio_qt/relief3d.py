"""The 3D relief inside the main window (VIEW-004).

An OpenGL 4.1 view of the solved surface that sits beside, or instead of,
the 2D view.  The mesh is the height map, but every pixel is shaded with the
photometric normals at the preview's full size, so the detail the
photographs recorded is all there even on a light mesh, and moving the
light only changes a few numbers sent to the graphics card: it follows the
light dome at full frame rate.

* **Exaggeration 1 is the true shape.**  The height map is only known up to
  a scale; the scale that makes its slopes agree with the normals is found
  once (least squares), so at 1 the relief has the proportions it really
  has, and at 4 every slope, in the geometry and in the shading, is four
  times steeper.
* **Surfaces**: the colour photograph, clay, roughness, trust (where the
  solve is unsure), height as colour, or any 2D view draped on the relief.
* **Reading tools**: height contours, a section line with its profile, and
  a two-point measure whose numbers are in millimetres when the scale is
  set (heights from the normals unless a height scale is measured).
* **Camera**: drag turns, ⇧-drag or Space-drag slides, the wheel comes
  closer, ⌥ makes any of them fine; 7 1 3 5 look from the top, front, side
  and along the light; F fits, Home resets; the gizmo's axes are buttons.
* **Every key is on screen** on the key card (? hides it), grouped as the
  research on 3D viewers suggests: always, camera, while held, tools.

Picking is done on the CPU against the height field (a ray marched through
it), so it needs nothing from the graphics card and works the same at any
anti-aliasing setting.  Without OpenGL 4.1 (a remote session, the offscreen
test platform) the view says so and offers the separate 3D window.
"""
from __future__ import annotations

import math
import time

import numpy as np

from helio.core.metrology import true_shape_scale  # noqa: F401  (shared; used here)
from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QSurfaceFormat
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QToolButton,
                               QVBoxLayout, QWidget)

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except Exception:                                    # pragma: no cover - Qt without OpenGL
    QOpenGLWidget = QWidget

MATERIALS = (("colour", "Colour"), ("clay", "Clay"), ("rough", "Rough"), ("trust", "Trust"),
             ("height", "Height"), ("view", "View"))
MATERIAL_TIPS = {
    "colour": "The colour photograph (albedo), lit.",
    "clay": "Shape only: an even clay colour, lit.",
    "rough": "The roughness map, lit: shiny is dark, rough is light.",
    "trust": "Where the solve is unsure, in colour: blue sure, yellow unsure.",
    "height": "Height as colour, lit: low blue to high yellow.",
    "view": "The 2D view on screen, draped on the relief as it is.",
}
PRESETS = {"top": (0.0, 89.0), "front": (0.0, 8.0), "side": (90.0, 8.0)}

# The key card: (group, [(keys, what)]), as on screen and in Help.
KEYS = (
    ("ALWAYS", (("⌥1 ⌥2 ⌥3", "Image · 3D · side by side"), ("F", "Fit the relief in the view"),
                ("Home", "Reset the camera, north up"), ("⌘⌥S", "Save this view as a picture"),
                ("?", "Show or hide these keys"))),
    ("CAMERA", (("drag", "Turn round the relief"), ("⇧ drag", "Slide the view"),
                ("double-click", "Turn round that point; the background: centre"),
                ("scroll", "Closer or further"), ("⌥", "Hold for slow, fine moves"),
                ("7 1 3 5", "Top · front · side · along the light"))),
    ("WHILE HELD", (("L drag", "Move the light"), ("Space drag", "Slide the view"))),
    ("TOOLS", (("M", "Measure between two points"), ("S", "Section through the relief"),
               ("A", "Area and volume: click round, double-click"),
               ("C", "Height contours on or off"), ("W", "Flood plane: what lies below a height"),
               ("X", "Mesh: off · over the surface · wire only"),
               ("[  ]", "Less or more exaggeration"),
               ("T", "Turntable: record a turn"), ("Esc", "Leave the tool"))),
)

VERT = """
#version 410 core
layout(location = 0) in vec2 a_uv;
uniform sampler2D u_height;
uniform mat4 u_mvp;
uniform float u_zscale;
uniform vec2 u_size;
out vec2 v_uv;
out vec3 v_pos;
void main() {
    float h = texture(u_height, a_uv).r;
    vec3 p = vec3((a_uv.x - 0.5) * u_size.x, (0.5 - a_uv.y) * u_size.y, h * u_zscale);
    v_uv = a_uv;
    v_pos = p;
    gl_Position = u_mvp * vec4(p, 1.0);
}
"""

# The high-detail window's adaptive mesh: each triangle of a lighter mesh
# divided on the graphics card, finer where it is large on screen (close to
# the camera), its new points lifted by the heights and by the fine heights
# from the full-size depth.  An edge's division depends on its two ends
# alone, so neighbours always agree and no cracks open.
TESS_VERT = """
#version 410 core
layout(location = 0) in vec2 a_uv;
out vec2 c_uv;
void main() { c_uv = a_uv; }
"""

TESS_CTRL = """
#version 410 core
layout(vertices = 3) out;
in vec2 c_uv[];
out vec2 e_uv[];
patch out float p_level;
uniform sampler2D u_height;
uniform mat4 u_mvp;
uniform float u_zscale;
uniform vec2 u_size;
uniform vec2 u_viewport;
uniform float u_tess_px;
uniform float u_tess_max;

vec4 clip_of(vec2 uv) {
    float h = textureLod(u_height, uv, 0.0).r;
    return u_mvp * vec4((uv.x - 0.5) * u_size.x, (0.5 - uv.y) * u_size.y, h * u_zscale, 1.0);
}

float level(vec4 a, vec4 b) {
    if (a.w <= 1e-6 || b.w <= 1e-6) return 1.0;
    vec2 d = (a.xy / a.w - b.xy / b.w) * 0.5 * u_viewport;
    return clamp(length(d) / u_tess_px, 1.0, u_tess_max);
}

void main() {
    e_uv[gl_InvocationID] = c_uv[gl_InvocationID];
    if (gl_InvocationID == 0) {
        vec4 p0 = clip_of(c_uv[0]), p1 = clip_of(c_uv[1]), p2 = clip_of(c_uv[2]);
        // wholly off one side of the screen (with a margin for the lift): not drawn
        float m = 1.15;
        bool off = (p0.x > m * p0.w && p1.x > m * p1.w && p2.x > m * p2.w)
                || (p0.x < -m * p0.w && p1.x < -m * p1.w && p2.x < -m * p2.w)
                || (p0.y > m * p0.w && p1.y > m * p1.w && p2.y > m * p2.w)
                || (p0.y < -m * p0.w && p1.y < -m * p1.w && p2.y < -m * p2.w);
        float l0 = off ? 0.0 : level(p1, p2);
        float l1 = off ? 0.0 : level(p2, p0);
        float l2 = off ? 0.0 : level(p0, p1);
        gl_TessLevelOuter[0] = l0;
        gl_TessLevelOuter[1] = l1;
        gl_TessLevelOuter[2] = l2;
        gl_TessLevelInner[0] = max(l0, max(l1, l2));
        p_level = max(l0, max(l1, l2));
    }
}
"""

TESS_EVAL = """
#version 410 core
layout(triangles, fractional_odd_spacing, ccw) in;
in vec2 e_uv[];
patch in float p_level;
uniform sampler2D u_height;
uniform sampler2D u_hdetail;
uniform int u_hd_on;
uniform float u_fine_per_edge;
uniform mat4 u_mvp;
uniform float u_zscale;
uniform vec2 u_size;
out vec2 v_uv;
out vec3 v_pos;
void main() {
    vec2 uv = gl_TessCoord.x * e_uv[0] + gl_TessCoord.y * e_uv[1] + gl_TessCoord.z * e_uv[2];
    float h = textureLod(u_height, uv, 0.0).r;
    // the fine heights at the level the points are spaced at: far off,
    // where the mesh is not divided, their average, so nothing shimmers
    float lod = log2(max(1.0, u_fine_per_edge / max(p_level, 1.0)));
    if (u_hd_on == 1) h += textureLod(u_hdetail, uv, lod).r;
    vec3 p = vec3((uv.x - 0.5) * u_size.x, (0.5 - uv.y) * u_size.y, h * u_zscale);
    v_uv = uv;
    v_pos = p;
    gl_Position = u_mvp * vec4(p, 1.0);
}
"""

FRAG = """
#version 410 core
in vec2 v_uv;
in vec3 v_pos;
uniform sampler2D u_height;
uniform sampler2D u_normals;
uniform sampler2D u_albedo;
uniform sampler2D u_view;
uniform sampler2D u_trust;
uniform sampler2D u_rough;
uniform sampler2D u_hdetail;
uniform int u_hd_on;
uniform int u_smooth;
uniform vec2 u_size;
uniform int u_material;
uniform int u_normalmap;
uniform float u_slope;
uniform vec3 u_light;
uniform vec3 u_eye;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shine;
uniform vec3 u_clay;
uniform int u_contours;
uniform float u_cstep;
uniform float u_zscale;
uniform int u_section;
uniform vec2 u_sa;
uniform vec2 u_sb;
uniform vec2 u_px;
uniform vec2 u_hrange;
uniform int u_flood_on;
uniform float u_flood;
uniform float u_gamma;
uniform float u_sat;
uniform float u_lift;
uniform float u_clip;
uniform int u_wire;
uniform float u_enhance;
uniform float u_cavity;
uniform float u_specboost;
uniform int u_light2_on;
uniform vec3 u_light2;
uniform int u_colourby;
uniform float u_slopemax;
uniform int u_shading;
uniform int u_exag_scales;
uniform float u_exag_sigma;
uniform float u_hs_alt;
out vec4 frag;

vec3 ramp(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 a = vec3(0.00, 0.13, 0.30), b = vec3(0.26, 0.36, 0.45), c = vec3(0.56, 0.56, 0.48);
    vec3 d = vec3(0.88, 0.80, 0.36), e = vec3(1.00, 0.92, 0.27);
    if (t < 0.25) return mix(a, b, t / 0.25);
    if (t < 0.5) return mix(b, c, (t - 0.25) / 0.25);
    if (t < 0.75) return mix(c, d, (t - 0.5) / 0.25);
    return mix(d, e, (t - 0.75) / 0.25);
}

vec3 bwr(float t) {
    t = clamp(t, -1.0, 1.0);
    vec3 mid = vec3(0.93, 0.92, 0.90);
    return t < 0.0 ? mix(mid, vec3(0.13, 0.36, 0.78), -t) : mix(mid, vec3(0.80, 0.22, 0.16), t);
}

// R(m)^T n: the normal turned as the light would be turned into the frame of
// the smoothed normal m (helio.core.shading_engines._unrotate_normal)
vec3 unrotate(vec3 n, vec3 m) {
    vec2 ax = vec2(-m.y, m.x);
    float c = clamp(m.z, -1.0, 1.0);
    float s = length(ax);
    if (s < 1e-6) return n;
    vec2 k = ax / s;
    float kn = k.x * n.x + k.y * n.y;
    float oc = 1.0 - c;
    return vec3(n.x * c - k.y * n.z * s + k.x * kn * oc,
                n.y * c + k.x * n.z * s + k.y * kn * oc,
                n.z * c - (k.x * n.y - k.y * n.x) * s);
}

float hgt(vec2 uv) {
    float h = texture(u_height, uv).r;
    if (u_hd_on == 1) h += texture(u_hdetail, uv).r;
    return h;
}

void main() {
    vec3 n;
    vec3 pn = texture(u_normals, v_uv).xyz;
    float curv = 0.0;
    if (u_enhance > 0.0 || u_cavity > 0.0 || u_colourby == 2) {
        // the normals a few texels round (or a pixel and a half on screen,
        // when farther off): their mean sharpens the detail (unsharp
        // masking), their spread across is the curvature
        vec2 ts = max(2.5 / vec2(textureSize(u_normals, 0)), 1.5 * fwidth(v_uv));
        vec3 nr = texture(u_normals, v_uv + vec2(ts.x, 0.0)).xyz;
        vec3 nl = texture(u_normals, v_uv - vec2(ts.x, 0.0)).xyz;
        vec3 nu = texture(u_normals, v_uv - vec2(0.0, ts.y)).xyz;
        vec3 nd = texture(u_normals, v_uv + vec2(0.0, ts.y)).xyz;
        curv = (nr.x - nl.x + nu.y - nd.y) * u_slope;
        pn += u_enhance * (pn - 0.25 * (nr + nl + nu + nd));
    }
    if (u_normalmap == 1) {
        n = normalize(vec3(pn.xy * u_slope, max(pn.z, 0.02)));
    } else if (u_smooth == 1) {
        // shaded by the heights themselves, smoothly: their slope a texel
        // either side, not the flat face of each triangle
        vec2 e = max(1.0 / vec2(u_hd_on == 1 ? textureSize(u_hdetail, 0) : textureSize(u_height, 0)),
                     fwidth(v_uv));                  // a pixel's width at least: no shimmer far off
        float dx = (hgt(v_uv + vec2(e.x, 0.0)) - hgt(v_uv - vec2(e.x, 0.0))) * u_zscale
                   / (2.0 * e.x * u_size.x);
        float dy = (hgt(v_uv - vec2(0.0, e.y)) - hgt(v_uv + vec2(0.0, e.y))) * u_zscale
                   / (2.0 * e.y * u_size.y);
        n = normalize(vec3(-dx, -dy, 1.0));
    } else {
        n = normalize(cross(dFdx(v_pos), dFdy(v_pos)));
        if (n.z < 0.0) n = -n;
    }
    vec3 L = normalize(u_light);
    vec3 V = normalize(u_eye - v_pos);
    float diff = max(dot(n, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = diff > 0.0 ? pow(max(dot(n, H), 0.0), u_shine) : 0.0;
    if (u_shading == 1) {
        // exaggerated shading (Rusinkiewicz, Burns & DeCarlo 2006): each
        // scale lit as if its smoothed neighbourhood faced the light; the
        // smoothed normals are the normals' own coarser mipmap levels, and
        // never finer than the screen shows
        float lod0 = textureQueryLod(u_normals, v_uv).y;
        float acc = 0.0, tot = 0.0;
        for (int i = 0; i < 8; i++) {
            if (i >= u_exag_scales) break;
            float lod = max(log2(2.0 * u_exag_sigma * exp2(float(i))), lod0 + 1.0);
            vec3 pm = textureLod(u_normals, v_uv, lod).xyz;
            vec3 m = normalize(vec3(pm.xy * u_slope, max(pm.z, 0.02)));
            float wgt = 1.0 / (float(i) + 1.0);            // finer scales lead
            acc += max(dot(unrotate(n, m), L), 0.0) * wgt;
            tot += wgt;
        }
        diff = acc / max(tot, 1e-6);
    } else if (u_shading == 2) {
        // multi-directional hillshade (Mark 1992, USGS weights): eight
        // lights round the compass at one height, the diagonals at 0.707
        float ca = cos(radians(u_hs_alt)), sa = sin(radians(u_hs_alt));
        float acc = 0.0;
        for (int i = 0; i < 8; i++) {
            float az = radians(45.0 * float(i));
            acc += max(dot(n, vec3(ca * sin(az), ca * cos(az), sa)), 0.0) * (i % 2 == 0 ? 1.0 : 0.707);
        }
        diff = acc / 6.828;
        spec = 0.0;
    }
    vec3 base;
    if (u_material == 0) base = texture(u_albedo, v_uv).rgb;
    else if (u_material == 1) base = u_clay;
    else if (u_material == 2) base = vec3(texture(u_rough, v_uv).r);
    else if (u_material == 3) base = ramp(texture(u_trust, v_uv).r);
    else base = ramp((texture(u_height, v_uv).r - u_hrange.x) / max(u_hrange.y - u_hrange.x, 1e-6));
    if (u_colourby == 1) base = ramp(acos(clamp(n.z, -1.0, 1.0)) / max(radians(u_slopemax), 1e-3));
    else if (u_colourby == 2) base = bwr(6.0 * curv);
    vec3 col;
    if (u_material == 5 && u_colourby == 0) {
        // The 2D picture carries its own light; the 3D light shades the relief
        // across it, relative to a light from straight above, so moving the
        // light shows (it used to do nothing on this surface).
        float flat_lit = u_ambient + u_diffuse * max(L.z, 0.25);
        col = texture(u_view, v_uv).rgb
              * clamp((u_ambient + u_diffuse * diff) / max(flat_lit, 1e-3), 0.0, 1.6);
    } else {
        float diff2 = (u_light2_on == 1 && u_shading == 0) ? 0.6 * max(dot(n, normalize(u_light2)), 0.0) : 0.0;
        col = base * (u_ambient + u_diffuse * (diff + diff2)) + vec3(u_specular * spec);
    }
    if (u_specboost > 0.0) {
        // specular enhancement, as in RTI: the colour quietened, a tight
        // highlight from the normals alone, so every stroke catches the light
        float luma0 = dot(col, vec3(0.299, 0.587, 0.114));
        col = mix(col, vec3(luma0), 0.6 * u_specboost);
        col += vec3(u_specboost * (diff > 0.0 ? pow(max(dot(n, H), 0.0), 72.0) : 0.0));
    }
    if (u_cavity > 0.0) col *= 1.0 + u_cavity * 0.7 * clamp(6.0 * curv, -1.0, 1.0);
    if (u_contours == 1 && u_cstep > 0.0) {
        float d = (texture(u_height, v_uv).r - u_hrange.x)
                  / max(u_hrange.y - u_hrange.x, 1e-6) / u_cstep;
        float w = fwidth(d);
        float line = 1.0 - smoothstep(0.0, 1.4 * w, abs(fract(d - 0.5) - 0.5));
        col = mix(col, vec3(0.06), 0.65 * line);
    }
    if (u_flood_on == 1) {
        // the flood plane: the surface below the level tinted, and its shoreline
        float hh = texture(u_height, v_uv).r;
        float fw = max(fwidth(hh), 1e-6);
        float under = 1.0 - smoothstep(u_flood - fw, u_flood + fw, hh);
        float shore = 1.0 - smoothstep(0.0, 1.8 * fw, abs(hh - u_flood));
        col = mix(col, vec3(0.08, 0.42, 0.95), 0.58 * under);
        col = mix(col, vec3(0.85, 0.95, 1.0), 0.9 * shore);
    }
    if (u_section == 1) {
        vec2 ab = u_sb - u_sa;
        vec2 q = (v_uv - u_sa) / u_px;
        vec2 abp = ab / u_px;
        float t = clamp(dot(q, abp) / max(dot(abp, abp), 1e-6), 0.0, 1.0);
        float dist = length(q - abp * t);
        float aa = max(fwidth(dist), 1e-4);
        float m = 1.0 - smoothstep(0.8 * aa, 2.2 * aa, dist);
        col = mix(col, vec3(0.43, 0.72, 1.0), m);
    }
    // display tone, as the classic viewer: shadow lift and highlight clip,
    // then gamma, then saturation about the luma
    col = clamp(col * (u_clip - u_lift) + u_lift, 0.0, 1.0);
    col = pow(max(col, vec3(1e-6)), vec3(1.0 / max(u_gamma, 0.05)));
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    col = clamp(vec3(luma) + (col - vec3(luma)) * u_sat, 0.0, 1.0);
    // the mesh's own edges, drawn over it: how dense the mesh is; on their
    // own (wire only), lit like the surface they outline
    if (u_wire == 1) col = mix(col, vec3(0.43, 0.72, 1.0), 0.75);
    else if (u_wire == 2) col = mix(col, vec3(0.70, 0.85, 1.0), 0.25);
    frag = vec4(col, 1.0);
}
"""


_GL_PROBE: tuple[bool, str] | None = None


def grid_size(w: int, h: int, st) -> tuple[int, int]:
    """Vertices across and down for a *w* x *h* surface sampled every *st* pixels."""
    def n(length):
        return max(2, int((length - 1) / st + 1e-9) + 1)
    return n(w), n(h)


def build_grid(w: int, h: int, st):
    """The mesh over a *w* x *h* surface sampled every *st* pixels (below 1: more vertices than
    pixels, filled in by the height texture's linear filtering): vertex positions (n, 2) float32
    in 0..1, triangle indices (6 per cell) uint32, and the grid's size.  Rust builds it in one
    parallel pass; the NumPy code is the reference it is checked against."""
    gw, gh = grid_size(w, h, st)
    from helio.core._native import rs
    if rs is not None and hasattr(rs, "grid_mesh"):
        verts, tris = rs.grid_mesh(gw, gh, int(w), int(h), float(st))
        return verts, tris, (gw, gh)
    return (*_build_grid_numpy(w, h, st, gw, gh), (gw, gh))


def _build_grid_numpy(w: int, h: int, st, gw: int, gh: int):
    u = (np.arange(gw, dtype=np.float32) * st + 0.5) / w
    v = (np.arange(gh, dtype=np.float32) * st + 0.5) / h
    uu, vv = np.meshgrid(np.minimum(u, 1.0), np.minimum(v, 1.0))
    verts = np.ascontiguousarray(np.stack([uu, vv], -1).reshape(-1, 2), np.float32)
    idx = np.arange(gw * gh, dtype=np.uint32).reshape(gh, gw)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.ascontiguousarray(np.stack([a, c, b, b, c, d], -1).ravel(), np.uint32)
    return verts, tris


def gl_available() -> tuple[bool, str]:
    """(True, "") when an OpenGL 4.1 core context can be made here, else
    (False, why).  Asked once: a platform with no OpenGL (the offscreen test
    platform, some remote sessions) never calls initializeGL, so without the
    question the view would sit blank instead of saying why."""
    global _GL_PROBE
    if _GL_PROBE is None:
        from PySide6.QtGui import QGuiApplication, QOpenGLContext
        try:
            plat = QGuiApplication.platformName()
            if plat in ("offscreen", "minimal", "vnc"):
                raise RuntimeError(f"the {plat} display has no OpenGL")
            fmt = QSurfaceFormat()
            fmt.setVersion(4, 1)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            ctx = QOpenGLContext()
            ctx.setFormat(fmt)
            if not ctx.create():
                raise RuntimeError("no OpenGL context could be made")
            got = ctx.format()
            if (got.majorVersion(), got.minorVersion()) < (4, 1):
                raise RuntimeError(f"OpenGL {got.majorVersion()}.{got.minorVersion()} only")
            _GL_PROBE = (True, "")
        except Exception as e:
            _GL_PROBE = (False, str(e))
    return _GL_PROBE


def _perspective(fov_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), np.float64)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _ortho(right: float, top: float, near: float, far: float) -> np.ndarray:
    m = np.zeros((4, 4), np.float64)
    m[0, 0] = 1.0 / max(right, 1e-9)
    m[1, 1] = 1.0 / max(top, 1e-9)
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    m[3, 3] = 1.0
    return m


def _look_at(eye, target, up) -> np.ndarray:
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-12)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(f, np.array([0.0, 1.0, 0.0]))
    s /= max(np.linalg.norm(s), 1e-12)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ eye, -u @ eye, f @ eye
    return m


class Camera:
    """A turntable camera round a target: azimuth (0 = looking north, from
    the south), elevation above the surface, distance; all eased on change."""

    def __init__(self):
        self.az, self.el, self.dist = 0.0, 55.0, 3.2
        self.target = np.zeros(3)
        self.fov = 30.0
        self._anim = None

    def eye(self) -> np.ndarray:
        a, e = math.radians(self.az), math.radians(self.el)
        d = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
        return self.target + d * self.dist

    def view(self) -> np.ndarray:
        # Straight down, "up" (z) is lost, so the screen's top is the way the camera
        # faces: away from the eye, (-sin az, cos az).  It is the same direction z
        # projects to just below, so looking past the top neither jumps nor flips.
        up = np.array([0.0, 0.0, 1.0]) if self.el < 88.5 else np.array(
            [-math.sin(math.radians(self.az)), math.cos(math.radians(self.az)), 0.0])
        return _look_at(self.eye(), self.target, up)

    def animate_to(self, az=None, el=None, dist=None, target=None, ms: float = 260.0):
        a0 = (self.az, self.el, self.dist, self.target.copy())
        az1 = self.az if az is None else az
        d = (az1 - self.az + 180.0) % 360.0 - 180.0           # the short way round
        a1 = (self.az + d, self.el if el is None else el, self.dist if dist is None else dist,
              self.target.copy() if target is None else np.asarray(target, np.float64))
        self._anim = (time.perf_counter(), ms / 1000.0, a0, a1)

    def step(self) -> bool:
        """Advance an animation; True while one is running."""
        if self._anim is None:
            return False
        t0, dur, a0, a1 = self._anim
        k = min(1.0, (time.perf_counter() - t0) / max(dur, 1e-6))
        k = k * k * (3 - 2 * k)                               # ease in and out
        self.az = a0[0] + (a1[0] - a0[0]) * k
        self.el = a0[1] + (a1[1] - a0[1]) * k
        self.dist = a0[2] + (a1[2] - a0[2]) * k
        self.target = a0[3] + (a1[3] - a0[3]) * k
        if k >= 1.0:
            self.az %= 360.0
            self._anim = None
        return True


class KeyCard(QFrame):
    """Every key of the 3D view, grouped, on the view itself (? hides it)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("hud", "true")
        self.setObjectName("KeyCard")
        self.setAccessibleName("Keys in the 3D view")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        t = QLabel("Keys in the 3D view")
        f = t.font()
        f.setWeight(QFont.Weight.DemiBold)
        t.setFont(f)
        head.addWidget(t, 1)
        self.hide_btn = QToolButton()
        self.hide_btn.setText("Hide  ·  press ?")
        self.hide_btn.setToolTip("Hide these keys (or press ?). The Keys button below brings "
                                 "them back.")
        self.hide_btn.setAccessibleName("Hide the keys")
        self.hide_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        head.addWidget(self.hide_btn)
        lay.addLayout(head)
        for group, rows in KEYS:
            g = QLabel(group)
            g.setProperty("role", "heading")
            lay.addWidget(g)
            grid = QGridLayout()
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(3)
            for r, (keys, what) in enumerate(rows):
                k = QLabel(keys)
                k.setObjectName("KeyCap")
                grid.addWidget(k, r, 0)
                w = QLabel(what)
                w.setProperty("role", "secondary")
                grid.addWidget(w, r, 1)
            grid.setColumnStretch(1, 1)
            lay.addLayout(grid)
        self.adjustSize()


class ReliefView(QOpenGLWidget):
    """The solved surface in 3D, shaded by its normals (module docstring)."""
    lightMoved = Signal(float, float, float, bool)      # lx, ly, lz (screen frame), dragging
    measured = Signal(dict)
    areaMeasured = Signal(dict)           # the area tool's region: helio.core.metrology.region_volume
    floodChanged = Signal(dict)           # the flood plane's figures (flood_info), when on and when it moves
    sectioned = Signal(dict)
    sectionHover = Signal(float)          # 0..1 along A-B under the pointer, or -1
    section_letters = ("A", "B")          # what the ends of the section line are called (the card names them)
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(4, 1)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        try:
            self.setFormat(fmt)
        except Exception:
            pass
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setMinimumSize(240, 200)
        self.setAccessibleName("3D relief")
        self.cam = Camera()
        self.gl_ok = None                 # None: not tried yet
        self.gl_error = ""
        self._gl = None
        self.data = None                  # dict of the arrays shown
        self._dirty = set()
        self.material = "colour"
        self.exaggeration = 0.6           # Height: starts just under the true shape, as asked
        self.normal_mapped = True
        self.ambient, self.diffuse, self.specular, self.shine = 0.18, 0.95, 0.12, 24.0
        self.clay = (0.6, 0.6, 0.6)       # Grey, as the classic viewer's clay
        self.tone = dict(gamma=1.0, sat=1.0, lift=0.0, clip=1.0)   # display only, as the classic
        self.hrange = (0.0, 1.0)          # the heights' extent: the height colours and contours
        self.contours, self.contour_step = False, 0.02
        self.light = np.array([-0.5, 0.5, 0.707])          # world: x east, y north, z up
        self.mesh_step = 2
        self.wireframe = False            # the mesh's edges drawn (over it, or on their own)
        self.wire_mode = "off"            # "off", "over" the surface, or "only" the wire
        # The high-detail window's own shading (all off here): fine detail
        # (unsharp masking on the normals), cavity shading from their
        # curvature, RTI-style specular enhancement, a second light some
        # degrees round from the first, and colour by slope or curvature.
        self.enhance, self.cavity, self.spec_boost = 0.0, 0.0, 0.0
        self.light2_offset = None         # degrees round from the light, or None
        self.colour_by, self.slope_max = "surface", 45.0
        self.shading = "standard"         # or "exaggerated", or "hillshade" from eight directions
        self.exag_scales, self.exag_sigma, self.hs_alt = 4, 2.0, 45.0
        self.max_texture = 16384          # the graphics card's largest texture, once it is asked
        # The high-detail window's geometry and camera (all off here): the
        # adaptive mesh (divided on the graphics card where it is close),
        # its target edge on screen in pixels, the fine heights from the
        # full-size depth, smooth shading from the heights when the normals
        # are not used, a camera without perspective, and a light that
        # follows the camera.
        self.adaptive, self.tess_px, self.fine_heights = False, 4.0, True
        self.smooth_shading, self.ortho, self.headlight = False, False, False
        self.tess_ok = None               # None: not tried yet
        self.tess_error = ""
        self.max_tess = 64
        self.tool = ""                    # "", "measure", "section", "area"
        self.points: list[np.ndarray] = []                 # measure: world points
        self.area_uv: list[tuple[float, float]] = []       # area: the outline, (u, v)
        self.area_done, self.area_result = False, None
        self.section = None                                # ((u, v), (u, v))
        self.line_marks: list[float] = []                  # chain lines found along it, 0..1 (section card)
        self.series_lines: list = []                       # the section's parallel copies, ((u, v), (u, v)) each
        self.flood_on = False                              # the flood plane (a level, tinting what lies below it)
        self.flood_level = 0.0                             # its height above the lowest point, in flood_scale()'s unit
        self.flood_region = "all"                          # "all" of the surface, or the "area" outline's inside
        self.flood_pick = False                            # the next click on the relief sets the level there
        self.section_t = None             # 0..1 along it: the chart's pointer, shown on the line
        self.keys_visible = self._stored_keys()    # the key card: until hidden once, then as left
        self._keys_asked = False          # shown on request even when the view is narrow
        self._clean = False               # drawing a picture: no gizmo
        self._readout_text = ""
        self.surround = None              # the image well's colour, when chosen
        self.scale_mm = None              # mm per pixel of the full-size photographs
        self.full_px = (1.0, 1.0)         # full-size pixels per pixel shown, across and down
        self.z_mm_per_unit = None         # mm per pixel of height (application.heights), when measured
        self._press = None
        self._held = set()
        self._fps = 0.0
        self._frames = []
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        self._build_overlays()
        ok, why = gl_available()
        if not ok:
            self._no_gl(why)

    def _no_gl(self, why: str):
        """Say why there is no 3D here and where the separate window is."""
        self.gl_ok, self.gl_error = False, why
        self.fallback.setText("The 3D view needs OpenGL 4.1, which this session does not "
                              f"offer ({why}).\nView › 3D Model opens the separate 3D window.")
        self.fallback.show()
        self.bar.hide()
        self.card.hide()

    # Keys the view keeps while it has focus.  Accepting the ShortcutOverride
    # is how a widget claims a key in Qt, and on macOS it is also what stops
    # a menu's key equivalent firing: without it, C or S here could reach the
    # 2D view's Clay or Save View through the View menu.
    OWN_KEYS = frozenset(int(k) for k in (
        Qt.Key.Key_F, Qt.Key.Key_Home, Qt.Key.Key_7, Qt.Key.Key_1, Qt.Key.Key_3, Qt.Key.Key_5,
        Qt.Key.Key_C, Qt.Key.Key_M, Qt.Key.Key_S, Qt.Key.Key_T, Qt.Key.Key_L, Qt.Key.Key_Space,
        Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight, Qt.Key.Key_Question, Qt.Key.Key_Slash,
        Qt.Key.Key_Escape, Qt.Key.Key_A, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_X))

    def event(self, e):
        if e.type() == QEvent.Type.ShortcutOverride and int(e.key()) in self.OWN_KEYS:
            held = e.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                    | Qt.KeyboardModifier.MetaModifier
                                    | Qt.KeyboardModifier.AltModifier)
            if not held:                  # ⇧ is part of ? and of nothing else here
                e.accept()
                return True
        return super().event(e)

    # ── overlays: the key card, the tools, the readout ─────────────────
    def _build_overlays(self):
        self.card = KeyCard(self)
        self.card.hide_btn.clicked.connect(self.toggle_keys)
        self.card.move(14, 14)
        self.bar = QFrame(self)
        self.bar.setProperty("floatbar", "true")
        bl = QHBoxLayout(self.bar)
        bl.setContentsMargins(4, 4, 4, 4)
        bl.setSpacing(2)
        self.tool_buttons = {}
        for key, text, tip in (("turn", "Turn", "Drag to turn round the relief"),
                               ("light", "Light", "L: hold and drag to move the light"),
                               ("measure", "Measure", "M: click two points on the relief"),
                               ("section", "Section", "S: drag a line across the relief"),
                               ("area", "Area", "A: click round a region, double-click or Enter "
                                                "to close it: its area and volume"),
                               ("contours", "Contours", "C: height contours on or off"),
                               ("flood", "Flood", "W: a flat plane at a height: what lies below it, its "
                                                  "area, and what it would hold"),
                               ("fit", "Fit", "F: the whole relief in view"),
                               ("keys", "Keys", "?: show or hide the keys")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.setAccessibleName(text)
            b.setCheckable(key in ("light", "measure", "section", "area", "contours", "flood", "keys"))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.clicked.connect(lambda _c=False, k=key: self._tool_clicked(k))
            bl.addWidget(b)
            self.tool_buttons[key] = b
        self.tool_buttons["keys"].setChecked(True)
        self.readout = QLabel(self)
        self.readout.setProperty("hud", "true")
        self.readout.setAccessibleName("3D readout")
        self.fallback = QLabel(self)
        self.fallback.setWordWrap(True)
        self.fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.fallback.hide()
        self._place_overlays()

    def _place_overlays(self):
        w, h = self.width(), self.height()
        self.bar.adjustSize()
        self.bar.move((w - self.bar.width()) // 2, h - self.bar.height() - 14)
        # The readout keeps clear of the gizmo (right) and, when it shows, the key card.
        roomy = w >= 760 or self._keys_asked
        card_on = (self.gl_ok is not False and self.keys_visible and roomy
                   and h > self.card.height() + 80)
        self.card.setVisible(card_on)
        if hasattr(self, "tool_buttons"):
            self.tool_buttons["keys"].setChecked(card_on)
        left = (self.card.width() + 28) if card_on else 14
        avail = max(60, w - left - 140)
        fm = self.readout.fontMetrics()
        self.readout.setText(fm.elidedText(self._readout_text, Qt.TextElideMode.ElideRight, avail - 22))
        self.readout.adjustSize()
        self.readout.move(left + max(0, (avail - self.readout.width()) // 2), 14)
        self.fallback.setGeometry(40, h // 2 - 60, max(80, w - 80), 120)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_overlays()

    def _tool_clicked(self, key: str):
        if key == "fit":
            self.fit()
        elif key == "keys":
            self.toggle_keys()
        elif key == "contours":
            self.set_contours(not self.contours)
        elif key == "flood":
            self.set_flood(not self.flood_on)
        elif key in ("measure", "section", "light", "area"):
            self.set_tool("" if self.tool == key else key)
        elif key == "turn":
            self.set_tool("")

    def set_tool(self, tool: str):
        self.tool = tool
        self.points = [] if tool != "measure" else self.points[-2:]
        for k in ("light", "measure", "section", "area"):
            self.tool_buttons[k].setChecked(tool == k)
        if tool != "area" and not self.area_done:
            self.area_uv = []                                # an outline not closed is let go
        self.setCursor(Qt.CursorShape.CrossCursor if tool in ("measure", "section", "area")
                       else Qt.CursorShape.ArrowCursor)
        self._readout()
        self.update()

    KEYS_KEY = "relief/keys_shown"

    @classmethod
    def _stored_keys(cls) -> bool:
        """Whether the key card shows: yes until it is hidden, then as it was left (it covers a third
        of a small view)."""
        try:
            from helio_qt.window import settings
            return str(settings().value(cls.KEYS_KEY, True)).lower() in ("true", "1")
        except Exception:
            return True

    def toggle_keys(self):
        showing = self.card.isVisible()
        self.keys_visible = not showing
        self._keys_asked = self.keys_visible
        self.tool_buttons["keys"].setChecked(self.keys_visible)
        try:
            from helio_qt.window import settings
            settings().setValue(self.KEYS_KEY, self.keys_visible)
        except Exception:
            pass
        self._place_overlays()

    def set_contours(self, on: bool):
        self.contours = bool(on)
        self.tool_buttons["contours"].setChecked(self.contours)
        self.update()

    # ── the flood plane ─────────────────────────────────────────────────
    def flood_scale(self):
        """(A, cell, unit): a height above the lowest point is (depth - lowest) x A in *unit* (mm when a
        scale is set, else pixels), and a cell of the surface is *cell* across in the same unit."""
        D = self.data
        kx = self.full_px[0]
        mm = self.scale_mm
        if mm and self.z_mm_per_unit:
            return D["s_px"] * kx * self.z_mm_per_unit, kx * mm, "mm"      # px of height x mm per px of height
        if mm:
            return D["s_px"] * kx * mm, kx * mm, "mm"
        return D["s_px"] * kx, kx, "px"

    def _lowest(self) -> float:
        d = self.data["depth"]
        c = getattr(self, "_dmin_cache", None)
        if c is None or c[0] is not d:
            self._dmin_cache = (d, float(d.min()))
        return self._dmin_cache[1]

    def flood_range(self) -> float:
        """The surface's height from its lowest to its highest point, in flood_scale()'s unit."""
        if self.data is None:
            return 0.0
        return float((float(self.data["depth"].max()) - self._lowest()) * self.flood_scale()[0])

    def flood_level_tex(self) -> float:
        """The level in the height texture's own units, for the shader."""
        if self.data is None:
            return 0.0
        A = self.flood_scale()[0]
        return self._lowest() + (self.flood_level / A if A else 0.0)

    def flood_mask(self):
        """The cells the figures are for: inside the Area tool's outline when asked and one is closed; else all."""
        if self.flood_region != "area" or not (self.area_done and self.area_uv) or self.data is None:
            return None
        import cv2
        D = self.data
        poly = np.array([[u * D["w"] - 0.5, v * D["h"] - 0.5] for u, v in self.area_uv], np.float32)
        m = np.zeros((D["h"], D["w"]), np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        return m.astype(bool)

    def flood_info(self, basins: bool = True) -> dict | None:
        """What lies below the plane: area, volume, depths (and its patches), in mm or pixels."""
        if self.data is None:
            return None
        from helio.core.flood import flood_basins, flood_totals
        A, cell, unit = self.flood_scale()
        z = (self.data["depth"] - self._lowest()) * A
        mask = self.flood_mask()
        out = flood_totals(z, self.flood_level, cell, mask)
        out.update(unit=unit, level=float(self.flood_level), range=self.flood_range(), cell=float(cell),
                   region="the outline" if mask is not None else "the whole surface",
                   basins=flood_basins(z, self.flood_level, cell, mask) if basins else None)
        return out

    def set_flood(self, on: bool):
        self.flood_on = bool(on) and self.data is not None
        self.tool_buttons["flood"].setChecked(self.flood_on)
        if self.flood_on and self.flood_level <= 0.0:
            self.flood_level = 0.3 * self.flood_range()        # a first look: a third of the way up
        if not self.flood_on:
            self.flood_pick = False
        self.update()
        self.floodChanged.emit(self.flood_info() if self.flood_on else {})

    def set_flood_level(self, level: float, live: bool = False):
        """Move the plane (height above the lowest point). *live*: a slider being dragged, so only the
        quick figures follow; the patches are worked out when it is let go."""
        if self.data is None:
            return
        self.flood_level = float(min(max(level, 0.0), self.flood_range()))
        self.update()
        if self.flood_on:
            self.floodChanged.emit(self.flood_info(basins=not live))

    def set_flood_region(self, which: str):
        self.flood_region = "area" if which == "area" else "all"
        if self.flood_on:
            self.floodChanged.emit(self.flood_info())

    def begin_flood_pick(self):
        """The next click on the relief sets the level at that point (click a hollow's rim to fill to it)."""
        self.flood_pick = True
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.message.emit("flood-pick")

    # ── data ─────────────────────────────────────────────────────────
    def set_surface(self, depth, normals, albedo, roughness=None, trust=None, view=None,
                    scale_mm=None, z_mm_per_unit=None, full_size=None):
        """The arrays to show, at the preview's size.  *depth* any float map;
        *normals* +Y up; *albedo* uint8 RGB; *trust* 0 sure .. 1 unsure."""
        d = np.ascontiguousarray(depth, np.float32)
        lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
        rng = hi - lo if hi > lo else 1.0
        d01 = np.nan_to_num((d - lo) / rng).astype(np.float32)
        h, w = d01.shape
        s = true_shape_scale(d01, normals)                  # pixels of height per unit
        self.data = dict(depth=d01, normals=np.ascontiguousarray(normals, np.float32),
                         albedo=np.ascontiguousarray(albedo, np.uint8),
                         rough=None if roughness is None else np.ascontiguousarray(roughness, np.uint8),
                         trust=None if trust is None else np.ascontiguousarray(trust, np.float32),
                         view=None if view is None else np.ascontiguousarray(view, np.uint8),
                         w=w, h=h, s_px=s, depth_range=rng)
        self.hrange = (0.0, 1.0)
        self.scale_mm, self.z_mm_per_unit = scale_mm, z_mm_per_unit
        # The surface is shown at the preview's size; lengths are given in the
        # photographs' own pixels, which the scale (Measure) is measured in.
        # Without this, every 3D length came out preview/full times too short
        # (4.6x on a 100 MP capture shown at 1875 px).
        self.full_px = ((float(full_size[0]) / w, float(full_size[1]) / h)
                        if full_size and w and h else (1.0, 1.0))
        self._dirty = {"height", "normals", "albedo", "rough", "trust", "view", "mesh", "hdetail"}
        self.mesh_step = self._clean_step(self.mesh_step)
        self.area_uv, self.area_done, self.area_result = [], False, None
        had = self.section is not None
        self.section, self.points, self.section_t, self.series_lines = None, [], None, []
        if self.flood_on:                               # another surface: the plane's height means nothing on it
            self.flood_on, self.flood_pick, self.flood_level = False, False, 0.0
            self.tool_buttons["flood"].setChecked(False)
            self.floodChanged.emit({})
        if had:
            self.sectioned.emit({})                  # another surface: its old line means nothing
        self.fit(animate=False)
        self._readout()
        self.update()

    def set_detail(self, normals=None, albedo=None):
        """Sharper pictures to shade and colour the mesh with than the mesh's
        own size (the high-detail window's 4,500 px normals): the shaders read
        them by position, so any size fits.  None keeps the mesh's own."""
        if self.data is None:
            return
        self.data["normals_tex"] = None if normals is None else np.ascontiguousarray(normals, np.float32)
        self.data["albedo_tex"] = None if albedo is None else np.ascontiguousarray(albedo, np.uint8)
        self._dirty |= {"normals", "albedo"}
        self._readout()
        self.update()

    def set_fine_heights(self, residual=None):
        """The heights' fine detail at the shading's size (what the full-size
        depth has that the mesh's heights do not, in their units), for the
        adaptive mesh to be lifted by.  None: none."""
        if self.data is None:
            return
        self.data["hdetail"] = None if residual is None else np.ascontiguousarray(residual, np.float32)
        self._dirty.add("hdetail")
        self._readout()
        self.update()

    def tess_max(self) -> float:
        """How far one triangle of the adaptive mesh may be divided: to half
        the finest heights' texel, and no more than the graphics card allows."""
        D = self.data
        if D is None:
            return 1.0
        hd = D.get("hdetail")
        fine_w = hd.shape[1] if (hd is not None and self.fine_heights) else D["w"]
        lv = 2.0 * float(self.mesh_used()) * fine_w / max(D["w"], 1)
        return float(max(1.0, min(lv, self.max_tess)))

    def lit(self) -> np.ndarray:
        """The light the relief is drawn with: the light, or with the
        headlight on, one from just over the viewer's shoulder."""
        if not self.headlight:
            return self.light
        a = math.radians(180.0 - self.cam.az - 35.0)
        e = math.radians(min(80.0, self.cam.el + 25.0))
        return np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])

    def set_scale(self, scale_mm=None, z_mm_per_unit=None):
        """The measured scales, when Measure sets or changes them."""
        self.scale_mm, self.z_mm_per_unit = scale_mm, z_mm_per_unit
        self._readout()
        self.update()

    def set_heights(self, z):
        """New heights for the surface, from the shape controls
        (helio.application.viewer3d.relief_surface), in the depth's own
        units.  The mesh, the true-shape scale, the height colours, the
        contours and the picking follow; the shading keeps the photometric
        normals."""
        if self.data is None or z is None:
            return
        import cv2
        D = self.data
        z = np.ascontiguousarray(z, np.float32)
        if z.shape != D["depth"].shape:
            z = cv2.resize(z, (D["w"], D["h"]), interpolation=cv2.INTER_AREA)
        z = np.nan_to_num(z / max(float(D["depth_range"]), 1e-9))
        D["depth"] = z
        D["s_px"] = true_shape_scale(z, D["normals"])
        self.hrange = (float(z.min()), float(z.max()))
        self._dirty.add("height")
        if self.flood_on:                               # heights changed shape: the figures follow
            self.floodChanged.emit(self.flood_info())
        self._readout()
        self.update()

    def set_view_image(self, rgb):
        """The 2D view as it is on screen, for the View surface."""
        if self.data is None or rgb is None:
            return
        self.data["view"] = np.ascontiguousarray(rgb, np.uint8)
        self._dirty.add("view")
        if self.material == "view":
            self.update()

    def set_light_screen(self, lx, ly, lz):
        """The 2D light (screen frame, +Y down) as the 3D light (+Y north)."""
        v = np.array([lx, -ly, lz], np.float64)
        n = np.linalg.norm(v)
        if n > 1e-9:
            self.light = v / n
            self.update()

    def world_size(self) -> tuple[float, float]:
        if self.data is None:
            return 2.0, 2.0
        w, h = self.data["w"], self.data["h"]
        return 2.0, 2.0 * h / max(w, 1)

    def zscale(self) -> float:
        """World height per unit of the normalised height map, exaggeration included."""
        if self.data is None:
            return 0.0
        return self.data["s_px"] * (2.0 / max(self.data["w"], 1)) * self.exaggeration

    def set_exaggeration(self, e: float):
        self.exaggeration = float(max(0.1, min(50.0, e)))
        self._readout()
        self.update()

    # Points the mesh may have: 1 : 1 of a 2500 px preview is 4.7 million
    # (9.4 million triangles, 150 MB), well within an Apple GPU.  It was 1.6
    # million, which turned 1 : 1 into 1 : 2 without a word.
    MAX_MESH_POINTS = 7_000_000      # the separate high-detail window takes 9 million

    def mesh_used(self) -> int:
        """The step the mesh is built at: the one chosen, unless that would
        pass MAX_MESH_POINTS (then the next that fits, said in the readout)."""
        st = self._clean_step(self.mesh_step)
        if self.adaptive and self.tess_ok is False:
            st = 1                                   # no adaptive mesh here: the densest instead
        if self.data is None:
            return st
        w, h = self.data["w"], self.data["h"]

        def points(s):
            gw, gh = grid_size(w, h, s)
            return gw * gh
        if st < 1 and points(st) > self.MAX_MESH_POINTS:
            st = 1                                   # denser than it can hold: one vertex a pixel
        while points(st) > self.MAX_MESH_POINTS:
            st += 1
        return st

    @staticmethod
    def _clean_step(step):
        """1, 2, 3 ... or a fraction down to 1/4 (0.5: four vertices to every pixel)."""
        step = float(step)
        return max(0.25, step) if step < 1 else max(1, int(step))

    @staticmethod
    def ratio(st) -> str:
        """The step as the panel names it: 1 : 2 is every second pixel, 2 : 1 two vertices a pixel."""
        return f"{1 / st:g} : 1" if st < 1 else f"1 : {st:g}"

    def mesh_counts(self, step=None) -> tuple[int, int]:
        """(points, triangles) of the mesh at *step* (the one built, by default)."""
        if self.data is None:
            return 0, 0
        st = self.mesh_used() if step is None else self._clean_step(step)
        gw, gh = grid_size(self.data["w"], self.data["h"], st)
        return gw * gh, 2 * (gw - 1) * (gh - 1)

    WIRE_MODES = ("off", "over", "only")

    def set_wireframe(self, on: bool, only: bool = False):
        self.set_wire_mode(("only" if only else "over") if on else "off")

    def set_wire_mode(self, mode: str):
        """The mesh's edges: "off", "over" the surface, or "only" the wire
        with the surface hidden (its hidden edges still hidden)."""
        self.wire_mode = mode if mode in self.WIRE_MODES else "off"
        self.wireframe = self.wire_mode != "off"
        self._readout()
        self.update()

    def cycle_wire(self):
        """X: off, over the surface, wire only, off."""
        i = self.WIRE_MODES.index(self.wire_mode)
        self.set_wire_mode(self.WIRE_MODES[(i + 1) % len(self.WIRE_MODES)])

    def light2(self):
        """The second light (world), *light2_offset* degrees round from the
        light at its height, or None."""
        if self.light2_offset is None:
            return None
        a = math.radians(float(self.light2_offset))
        x, y, z = (float(c) for c in self.lit())
        return np.array([x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a), z])

    def set_expert(self, **kw):
        """The high-detail window's shading: enhance, cavity, spec_boost,
        light2_offset, colour_by, slope_max, shading (standard, exaggerated,
        hillshade), exag_scales, exag_sigma, hs_alt."""
        for k, v in kw.items():
            if k not in ("enhance", "cavity", "spec_boost", "light2_offset", "colour_by", "slope_max",
                         "shading", "exag_scales", "exag_sigma", "hs_alt"):
                raise KeyError(k)
            setattr(self, k, v)
        self._readout()
        self.update()

    def set_mesh_step(self, step):
        self.mesh_step = self._clean_step(step)
        self._dirty.add("mesh")
        self._readout()
        self.update()

    def set_material(self, key: str):
        self.material = key
        self.update()

    def fit(self, animate: bool = True):
        W, H = self.world_size()
        dist = 1.25 * max(W, H) / (2 * math.tan(math.radians(self.cam.fov) / 2))
        if animate:
            self.cam.animate_to(dist=dist, target=np.zeros(3))
            self._timer.start()
        else:
            self.cam.dist, self.cam.target = dist, np.zeros(3)
        self.update()

    def look(self, preset: str):
        """7 top, 1 front, 3 side, 5 along the light (raking), Home north up."""
        if preset == "light":
            az = math.degrees(math.atan2(self.light[0], self.light[1])) % 360.0
            self.cam.animate_to(az=(az + 180.0) % 360.0, el=18.0)
        elif preset == "home":
            self.cam.animate_to(az=0.0, el=55.0, target=np.zeros(3))
        else:
            az, el = PRESETS[preset]
            self.cam.animate_to(az=az, el=el)
        self._timer.start()

    # ── OpenGL ───────────────────────────────────────────────────────
    def initializeGL(self):
        try:
            import OpenGL
            OpenGL.ERROR_CHECKING = False         # errors are checked where they matter, below
            from OpenGL import GL
            self._GL = GL
            for _ in range(16):                    # errors left in the context by others
                if GL.glGetError() == GL.GL_NO_ERROR:
                    break
            ver = GL.glGetString(GL.GL_VERSION)
            ver = ver.decode() if isinstance(ver, bytes) else str(ver)
            major = int(ver.split(".")[0]) if ver[:1].isdigit() else 0
            if major < 4 and "4." not in ver.split(" ")[0]:
                raise RuntimeError(f"OpenGL 4.1 is needed; this context is {ver}")
            self._program = self._link(VERT, FRAG)
            try:
                self._tess_program = self._link(TESS_VERT, FRAG, TESS_CTRL, TESS_EVAL)
                self.max_tess = int(GL.glGetIntegerv(GL.GL_MAX_TESS_GEN_LEVEL))
                self.tess_ok = True
            except Exception as e:                       # the plain mesh does instead
                self._tess_program, self.tess_ok, self.tess_error = None, False, str(e)
                for _ in range(16):
                    if GL.glGetError() == GL.GL_NO_ERROR:
                        break
            self._vao = GL.glGenVertexArrays(1)
            self._vbo, self._ibo = GL.glGenBuffers(2)
            self._tex = {}
            self._n_index = 0
            GL.glEnable(GL.GL_DEPTH_TEST)
            try:
                self.max_texture = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))
            except Exception:
                pass
            self.gl_ok = True
            if self.data is not None:      # a new context: everything goes up again
                self._dirty = {"height", "normals", "albedo", "rough", "trust", "view", "mesh", "hdetail"}
        except Exception as e:                                   # no usable OpenGL here
            self._no_gl(str(e))

    def _link(self, vs: str, fs: str, tcs: str | None = None, tes: str | None = None):
        GL = self._GL
        prog = GL.glCreateProgram()
        stages = [(vs, GL.GL_VERTEX_SHADER), (fs, GL.GL_FRAGMENT_SHADER)]
        if tcs and tes:
            stages += [(tcs, GL.GL_TESS_CONTROL_SHADER), (tes, GL.GL_TESS_EVALUATION_SHADER)]
        for src, kind in stages:
            sh = GL.glCreateShader(kind)
            GL.glShaderSource(sh, src)
            GL.glCompileShader(sh)
            if not GL.glGetShaderiv(sh, GL.GL_COMPILE_STATUS):
                raise RuntimeError(GL.glGetShaderInfoLog(sh).decode(errors="replace"))
            GL.glAttachShader(prog, sh)
        GL.glLinkProgram(prog)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            raise RuntimeError(GL.glGetProgramInfoLog(prog).decode(errors="replace"))
        return prog

    def _texture(self, name: str, arr, kind: str, mip: bool = False):
        GL = self._GL
        if name not in self._tex:
            self._tex[name] = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex[name])
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        a = np.ascontiguousarray(arr)
        h, w = a.shape[:2]
        if kind == "r32f":
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32F, w, h, 0, GL.GL_RED, GL.GL_FLOAT,
                            a.astype(np.float32))
        elif kind == "rgb16f":
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB16F, w, h, 0, GL.GL_RGB, GL.GL_FLOAT,
                            a.astype(np.float32))
        elif kind == "r16f":
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R16F, w, h, 0, GL.GL_RED, GL.GL_FLOAT,
                            a.astype(np.float32))
        elif kind == "r8":
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R8, w, h, 0, GL.GL_RED, GL.GL_UNSIGNED_BYTE,
                            a.astype(np.uint8))
        else:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                            a.astype(np.uint8))
        # The normals and colour are often far larger than the screen shows
        # them: mipmapped, they do not shimmer when the relief is far off.
        if mip:
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR if mip else GL.GL_LINEAR),
                     (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)   # Qt's own uploads (the glyph cache) expect 4

    def _upload(self):
        GL, D = self._GL, self.data
        if "height" in self._dirty:
            self._texture("height", D["depth"], "r32f", mip=True)
        if "normals" in self._dirty:
            nt = D.get("normals_tex")
            self._texture("normals", D["normals"] if nt is None else nt, "rgb16f", mip=True)
        if "albedo" in self._dirty:
            at = D.get("albedo_tex")
            self._texture("albedo", D["albedo"] if at is None else at, "rgb8", mip=True)
        if "rough" in self._dirty:
            r = D["rough"] if D["rough"] is not None else np.full((2, 2), 128, np.uint8)
            self._texture("rough", r, "r8")
        if "trust" in self._dirty:
            t = D["trust"] if D["trust"] is not None else np.zeros((2, 2), np.float32)
            self._texture("trust", t, "r32f")
        if "view" in self._dirty:
            v = D["view"] if D["view"] is not None else D["albedo"]
            self._texture("view", v, "rgb8")
        if "hdetail" in self._dirty:
            hd = D.get("hdetail")
            self._texture("hdetail", np.zeros((2, 2), np.float32) if hd is None else hd, "r16f", mip=True)
        if "mesh" in self._dirty:
            verts, tris, (gw, gh) = build_grid(D["w"], D["h"], self.mesh_used())
            GL.glBindVertexArray(self._vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, verts.nbytes, verts, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 8, None)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ibo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, tris.nbytes, tris, GL.GL_STATIC_DRAW)
            GL.glBindVertexArray(0)
            self._n_index = int(tris.size)
            self._grid = (gw, gh)
        self._dirty.clear()

    def mvp(self, size=None) -> np.ndarray:
        """The camera's matrix for the view, or for a picture of *size* (w, h) pixels."""
        if size is None:
            dpr = self.devicePixelRatioF()
            w, h = max(1, int(self.width() * dpr)), max(1, int(self.height() * dpr))
        else:
            w, h = max(1, int(size[0])), max(1, int(size[1]))
        W, H = self.world_size()
        far = self.cam.dist + 4 * max(W, H) + 2
        if self.ortho:
            # no perspective: the half-height the perspective camera shows at
            # the target, so turning it on keeps the view's size
            top = self.cam.dist * math.tan(math.radians(self.cam.fov) / 2.0)
            proj = _ortho(top * w / h, top, -(4 * max(W, H) + 2), far)
        else:
            proj = _perspective(self.cam.fov, w / h, max(0.01, self.cam.dist / 200.0), far)
        return proj @ self.cam.view()

    def paintGL(self):
        t0 = time.perf_counter()
        bg = QColor(getattr(self, "surround", None) or "#2B2B2B")
        if not self.gl_ok:
            return
        GL = self._GL
        GL.glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if self.data is None:
            return
        try:
            dpr = self.devicePixelRatioF()
            self._draw_gl(int(self.width() * dpr), int(self.height() * dpr), None)
        except Exception as e:                                  # reported, never a crash
            self.gl_ok, self.gl_error = False, str(e)
            self.fallback.setText(f"The 3D view stopped: {e}")
            self.fallback.show()
            return
        self._paint_overlay()
        dt = time.perf_counter() - t0
        self._frames = (self._frames + [dt])[-30:]

    def _draw_gl(self, w: int, h: int, size):
        """The relief into the bound framebuffer, *w* x *h* pixels; *size*
        is the picture's (w, h) when drawing off screen, else None."""
        GL = self._GL
        if self._dirty:
            self._upload()
        GL.glViewport(0, 0, w, h)
        GL.glEnable(GL.GL_DEPTH_TEST)
        tess = bool(self.adaptive and self.tess_ok and getattr(self, "_tess_program", None))
        P = self._tess_program if tess else self._program
        prim = GL.GL_PATCHES if tess else GL.GL_TRIANGLES
        GL.glUseProgram(P)
        loc = lambda n: GL.glGetUniformLocation(P, n)                  # noqa: E731
        if tess:
            GL.glPatchParameteri(GL.GL_PATCH_VERTICES, 3)
            GL.glUniform2f(loc("u_viewport"), float(w), float(h))
            GL.glUniform1f(loc("u_tess_px"), float(self.tess_px))
            GL.glUniform1f(loc("u_tess_max"), self.tess_max())
            hd = self.data.get("hdetail")
            GL.glUniform1f(loc("u_fine_per_edge"), float(self.mesh_used()) * (
                hd.shape[1] if hd is not None else self.data["w"]) / max(self.data["w"], 1))
        GL.glUniformMatrix4fv(loc("u_mvp"), 1, GL.GL_TRUE, self.mvp(size).astype(np.float32))
        W, H = self.world_size()
        GL.glUniform2f(loc("u_size"), W, H)
        GL.glUniform1f(loc("u_zscale"), self.zscale())
        for unit, name in enumerate(("height", "normals", "albedo", "view", "trust", "rough", "hdetail")):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex[name])
            GL.glUniform1i(loc(f"u_{name}"), unit)
        mat = [k for k, _t in MATERIALS].index(self.material) if self.material in dict(MATERIALS) else 0
        GL.glUniform1i(loc("u_material"), mat)
        GL.glUniform1i(loc("u_normalmap"), 1 if self.normal_mapped else 0)
        GL.glUniform1f(loc("u_slope"), self.exaggeration)
        GL.glUniform3f(loc("u_light"), *[float(x) for x in self.lit()])
        eye = self.cam.eye()
        if self.ortho:                     # parallel rays: the eye far off along them
            eye = self.cam.target + (eye - self.cam.target) * 50.0
        GL.glUniform3f(loc("u_eye"), *[float(x) for x in eye])
        GL.glUniform1i(loc("u_hd_on"), 1 if (self.fine_heights and tess
                                             and self.data.get("hdetail") is not None) else 0)
        GL.glUniform1i(loc("u_smooth"), 1 if self.smooth_shading else 0)
        GL.glUniform1f(loc("u_ambient"), self.ambient)
        GL.glUniform1f(loc("u_diffuse"), self.diffuse)
        GL.glUniform1f(loc("u_specular"), self.specular)
        GL.glUniform1f(loc("u_shine"), self.shine)
        GL.glUniform3f(loc("u_clay"), *self.clay)
        GL.glUniform1i(loc("u_contours"), 1 if self.contours else 0)
        GL.glUniform1f(loc("u_cstep"), float(self.contour_step))
        sec = self.section
        GL.glUniform1i(loc("u_section"), 1 if sec else 0)
        if sec:
            GL.glUniform2f(loc("u_sa"), *sec[0])
            GL.glUniform2f(loc("u_sb"), *sec[1])
        GL.glUniform2f(loc("u_px"), 1.0 / self.data["w"], 1.0 / self.data["h"])
        GL.glUniform2f(loc("u_hrange"), float(self.hrange[0]), float(self.hrange[1]))
        GL.glUniform1i(loc("u_flood_on"), 1 if self.flood_on else 0)
        GL.glUniform1f(loc("u_flood"), float(self.flood_level_tex()))
        t = self.tone
        GL.glUniform1f(loc("u_gamma"), float(t["gamma"]))
        GL.glUniform1f(loc("u_sat"), float(t["sat"]))
        GL.glUniform1f(loc("u_lift"), float(t["lift"]))
        GL.glUniform1f(loc("u_clip"), float(t["clip"]))
        GL.glUniform1f(loc("u_enhance"), float(self.enhance))
        GL.glUniform1f(loc("u_cavity"), float(self.cavity))
        GL.glUniform1f(loc("u_specboost"), float(self.spec_boost))
        l2 = self.light2()
        GL.glUniform1i(loc("u_light2_on"), 0 if l2 is None else 1)
        GL.glUniform3f(loc("u_light2"), *([0.0, 0.0, 1.0] if l2 is None else [float(x) for x in l2]))
        GL.glUniform1i(loc("u_colourby"), {"slope": 1, "curvature": 2}.get(self.colour_by, 0))
        GL.glUniform1f(loc("u_slopemax"), float(self.slope_max))
        GL.glUniform1i(loc("u_shading"), {"exaggerated": 1, "hillshade": 2}.get(self.shading, 0))
        GL.glUniform1i(loc("u_exag_scales"), int(max(1, min(8, self.exag_scales))))
        GL.glUniform1f(loc("u_exag_sigma"), float(self.exag_sigma))
        GL.glUniform1f(loc("u_hs_alt"), float(self.hs_alt))
        GL.glUniform1i(loc("u_wire"), 0)
        GL.glBindVertexArray(self._vao)
        only = self.wire_mode == "only"
        if self.wireframe:
            # the surface pushed back a little, so its edges draw over it
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glPolygonOffset(1.0, 1.0)
        if only:
            # wire only: the surface into the depth buffer alone, so the
            # edges behind it stay hidden and the background shows between
            GL.glColorMask(GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE, GL.GL_FALSE)
        GL.glDrawElements(prim, self._n_index, GL.GL_UNSIGNED_INT, None)
        if only:
            GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        if self.wireframe:
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
            GL.glUniform1i(loc("u_wire"), 2 if only else 1)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glDrawElements(prim, self._n_index, GL.GL_UNSIGNED_INT, None)
            GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
            GL.glDepthFunc(GL.GL_LESS)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        # Leave the context as QPainter expects it: no depth test, unit 0.
        GL.glDisable(GL.GL_DEPTH_TEST)
        for unit in range(7):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)

    def centre_level(self) -> float:
        """How finely the adaptive mesh is divided where the camera looks (a
        primitives query would say exactly, but Apple's OpenGL stops on one
        round a divided mesh): its edge there on screen, over the target."""
        if self.data is None:
            return 1.0
        dpr = self.devicePixelRatioF()
        h_px = max(1.0, self.height() * dpr)
        edge = float(self.mesh_used()) * 2.0 / max(self.data["w"], 1)        # world units
        top = self.cam.dist * math.tan(math.radians(self.cam.fov) / 2.0)
        px = edge * (h_px / 2.0) / max(top, 1e-9)
        return float(max(1.0, min(px / max(self.tess_px, 0.5), self.tess_max())))

    def picture_at(self, w: int, h: int) -> np.ndarray:
        """The relief drawn afresh at *w* x *h* pixels (a 4K picture, say),
        off screen, without the gizmo: uint8 RGB.  The camera keeps its
        view; a different shape of picture shows more or less round it."""
        from PySide6.QtOpenGL import QOpenGLFramebufferObject, QOpenGLFramebufferObjectFormat
        if not self.gl_ok or self.data is None:
            raise RuntimeError(self.gl_error or "the 3D view has nothing to draw")
        self.makeCurrent()
        try:
            fmt = QOpenGLFramebufferObjectFormat()
            fmt.setAttachment(QOpenGLFramebufferObject.Attachment.CombinedDepthStencil)
            fmt.setSamples(4)
            fbo = QOpenGLFramebufferObject(int(w), int(h), fmt)
            if not fbo.isValid():
                raise RuntimeError(f"a {w} x {h} picture is too large for this graphics card")
            fbo.bind()
            GL = self._GL
            bg = QColor(getattr(self, "surround", None) or "#2B2B2B")
            GL.glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            self._draw_gl(int(w), int(h), (int(w), int(h)))
            fbo.release()
            img = fbo.toImage().convertToFormat(QImage.Format.Format_RGB888)
        finally:
            self.doneCurrent()
        bpl = img.bytesPerLine()
        a = np.frombuffer(img.constBits(), np.uint8, count=bpl * img.height()).reshape(img.height(), bpl)
        return a[:, :img.width() * 3].reshape(img.height(), img.width(), 3).copy()

    def clay_picture(self, w: int = 2400, h: int = 1600, light_az: float = 315.0,
                     light_el: float = 25.0) -> np.ndarray:
        """The relief in clay, seen at an angle and a little closer than the
        whole, lit low from *light_az* (a compass bearing): the publication
        sheet's 3D picture.  The view on screen is left as it was."""
        c = self.cam
        keep = (self.material, c.az, c.el, c.dist, c.target.copy(), self.light.copy(), c._anim,
                self.section)
        try:
            W, H = self.world_size()
            fit = 1.25 * max(W, H) / (2 * math.tan(math.radians(c.fov) / 2))
            self.material, self.section = "clay", None
            c._anim = None
            c.az, c.el, c.dist, c.target = -20.0, 40.0, 0.7 * fit, np.zeros(3)
            a, e = math.radians(light_az), math.radians(light_el)
            self.light = np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e),
                                   math.sin(e)])
            return self.picture_at(w, h)
        finally:
            (self.material, c.az, c.el, c.dist, c.target, self.light, c._anim,
             self.section) = keep

    # ── what is drawn over the relief with QPainter ─────────────────
    def project(self, p) -> QPointF | None:
        m = self.mvp() @ np.array([p[0], p[1], p[2], 1.0])
        if m[3] <= 1e-9:
            return None
        x, y = m[0] / m[3], m[1] / m[3]
        return QPointF((x * 0.5 + 0.5) * self.width(), (0.5 - y * 0.5) * self.height())

    def _paint_overlay(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if not self._clean:
            self._paint_gizmo(p)
        if self.points:
            pen = QPen(QColor("#E6A93E"), 2)
            p.setPen(pen)
            pts = [self.project(q) for q in self.points]
            for q in pts:
                if q is not None:
                    p.setBrush(QColor("#E6A93E"))
                    p.drawEllipse(q, 4.5, 4.5)
            if len(pts) == 2 and None not in pts:
                p.drawLine(pts[0], pts[1])
        if self.section and self.data is not None and not self._clean:
            self._paint_section(p)
        if self.area_uv and self.data is not None and not self._clean:
            self._paint_area(p)
        p.end()

    SECTION_BLUE = "#6FB7FF"

    def world_of(self, u: float, v: float):
        W, H = self.world_size()
        return np.array([(u - 0.5) * W, (0.5 - v) * H, self.height_at(u, v) * self.zscale()])

    def section_ends_screen(self):
        """Where A and B are on screen, or None for an end behind the camera."""
        if not self.section or self.data is None:
            return None, None
        return tuple(self.project(self.world_of(*uv)) for uv in self.section)

    def _paint_section(self, p: QPainter):
        """A and B at the ends, as a section drawing reads (A–A′), and the
        chart's pointer as a marker on the line."""
        a, b = self.section_ends_screen()
        if self.series_lines:                              # the parallel copies, on the surface
            pen = QPen(QColor(self.SECTION_BLUE), 1.4, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            for (ua, va), (ub, vb) in self.series_lines:
                pts = [self.project(self.world_of(ua + (ub - ua) * t, va + (vb - va) * t))
                       for t in np.linspace(0.0, 1.0, 24)]
                pts = [q for q in pts if q is not None]
                if len(pts) > 1:
                    p.drawPolyline(pts)
        f = p.font()
        f.setPointSizeF(10.0)
        f.setWeight(QFont.Weight.Bold)
        p.setFont(f)
        for q, name in ((a, self.section_letters[0]), (b, self.section_letters[1])):
            if q is None:
                continue
            p.setPen(QPen(QColor("#0E1B2C"), 1.5))
            p.setBrush(QColor(self.SECTION_BLUE))
            p.drawEllipse(q, 10.5, 10.5)
            p.setPen(QColor("#0E1B2C"))
            p.drawText(QRectF(q.x() - 10.5, q.y() - 10.5, 21, 21), Qt.AlignmentFlag.AlignCenter, name)
        if self.line_marks:                                # chain lines found along it, numbered from A
            (u1, v1), (u2, v2) = self.section
            violet = QColor("#B48CF0")
            for k, tl in enumerate(self.line_marks, 1):
                m = self.project(self.world_of(u1 + (u2 - u1) * tl, v1 + (v2 - v1) * tl))
                if m is None:
                    continue
                p.setPen(QPen(QColor("#0E1B2C"), 1.2))
                p.setBrush(violet)
                p.drawEllipse(m, 7.0, 7.0)
                p.setPen(QColor("#0E1B2C"))
                p.drawText(QRectF(m.x() - 9, m.y() - 9, 18, 18), Qt.AlignmentFlag.AlignCenter, str(k))
            p.setFont(f)
        t = self.section_t
        if t is not None and 0.0 <= t <= 1.0:
            (u1, v1), (u2, v2) = self.section
            m = self.project(self.world_of(u1 + (u2 - u1) * t, v1 + (v2 - v1) * t))
            if m is not None:
                p.setPen(QPen(QColor("#FFFFFF"), 2))
                p.setBrush(QColor("#E6A93E"))
                p.drawEllipse(m, 6.5, 6.5)

    # ── area and volume ─────────────────────────────────────────────
    def mouseDoubleClickEvent(self, e):
        """Double-click: the area tool closes its outline; otherwise the camera
        turns round the point clicked (on the background: the centre, the
        whole relief in view, as the classic viewer's double-click did)."""
        if self.tool == "area":
            self.finish_area()
            return
        if e.button() != Qt.MouseButton.LeftButton or self.data is None:
            super().mouseDoubleClickEvent(e)
            return
        pos = e.position()
        q = self.pick(pos.x(), pos.y())
        if q is None:
            self.fit()
        else:
            self.cam.animate_to(target=np.asarray(q, float))
            self._timer.start()
        self.message.emit("changed")

    def finish_area(self):
        """Close the outline: the region's area and volume about the plane
        through its edge (helio.core.metrology.region_volume), in mm with a
        scale, else in the photographs' pixels."""
        if self.tool != "area" or self.data is None:
            return
        pts = list(dict.fromkeys(self.area_uv))       # a double-click adds its point twice
        if len(pts) < 3:
            return
        self.area_uv = pts
        self.area_done = True
        self.area_result = self.area_measure()
        if self.area_result:
            self.areaMeasured.emit(self.area_result)
        self._readout()
        self.update()

    def area_measure(self) -> dict | None:
        import cv2
        from helio.core.metrology import region_volume
        D = self.data
        h, w = D["h"], D["w"]
        poly = np.array([[u * w - 0.5, v * h - 0.5] for u, v in self.area_uv], np.float32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        kx = self.full_px[0]                              # the photographs' pixels per cell
        mm = self.scale_mm
        if mm and self.z_mm_per_unit:
            z, cell, unit = D["depth"] * D["s_px"] * kx * self.z_mm_per_unit, kx * mm, "mm"
        elif mm:
            z, cell, unit = D["depth"] * D["s_px"] * kx * mm, kx * mm, "mm"
        else:
            z, cell, unit = D["depth"] * D["s_px"] * kx, kx, "px"
        out = region_volume(z, mask.astype(bool), cell)
        if out is not None:
            # The flat area from the outline itself (exact), not by counting
            # cells, which counts the edge's cells whole; the surface in step.
            x, y = poly[:, 0], poly[:, 1]
            exact = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * cell * cell
            out["surface"] *= exact / max(out["area"], 1e-12)
            out["area"] = exact
            out["unit"] = unit
        return out

    def _paint_area(self, p: QPainter):
        from PySide6.QtGui import QPolygonF
        pts = [self.project(self.world_of(u, v)) for u, v in self.area_uv]
        pts = [q for q in pts if q is not None]
        if not pts:
            return
        col = QColor("#7FD18B")
        p.setPen(QPen(col, 2))
        if self.area_done and len(pts) >= 3:
            fill = QColor(col)
            fill.setAlpha(60)
            p.setBrush(fill)
            p.drawPolygon(QPolygonF(pts))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPolyline(QPolygonF(pts))
        p.setBrush(col)
        for q in pts:
            p.drawEllipse(q, 3.5, 3.5)

    def set_section_cursor(self, t):
        """The chart's pointer, 0..1 from A to B, or None."""
        t = None if t is None or t < 0 else float(t)
        if t != self.section_t:
            self.section_t = t
            self.update()

    def set_section(self, a, b):
        """The section from (u, v) *a* to *b*, as if drawn."""
        self.section = (tuple(map(float, a)), tuple(map(float, b)))
        self._section_changed()

    def set_line_marks(self, places):
        """Chain lines (or laid lines, ruling) found along the section, 0..1 from A: marked on it."""
        self.line_marks = [float(v) for v in places or ()]
        self.update()

    def clear_section(self):
        self.section, self.section_t, self.series_lines, self.line_marks = None, None, [], []
        self.sectioned.emit({})
        self._readout()
        self.update()

    def _section_changed(self, live: bool = False):
        prof = self.section_profile(features=not live)
        if prof:
            self.sectioned.emit(prof)
        self._readout()
        self.update()

    def _straight(self, uv, other):
        """Shift: straight across or straight down from the other end."""
        D = self.data
        du, dv = abs(uv[0] - other[0]) * D["w"], abs(uv[1] - other[1]) * D["h"]
        return (uv[0], other[1]) if du >= dv else (other[0], uv[1])

    def picture(self) -> np.ndarray:
        """The relief as drawn, without the gizmo, as uint8 RGB: for saved
        pictures, turntables and the publication plate.  The key card, the
        tools and the readout are widgets over the view, never in its
        framebuffer, so they need no hiding."""
        self._clean = True
        try:
            img = self.grabFramebuffer().convertToFormat(QImage.Format.Format_RGB888)
        finally:
            self._clean = False
        w, h, bpl = img.width(), img.height(), img.bytesPerLine()
        a = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
        return a[:, :w * 3].reshape(h, w, 3).copy()

    def gizmo_axes(self):
        """(label, screen offset, colour, (az, el) it looks along) for E, N, Z and their opposites."""
        V = self.cam.view()[:3, :3]
        out = []
        # Clicking an axis puts the camera out along it, looking back at the relief.
        for name, v, col, look in (("E", (1, 0, 0), "#E0675B", (90.0, 5.0)),
                                   ("N", (0, 1, 0), "#46C79A", (180.0, 5.0)),
                                   ("Z", (0, 0, 1), "#6FA8F5", (0.0, 89.0)),
                                   ("W", (-1, 0, 0), "#E0675B", (270.0, 5.0)),
                                   ("S", (0, -1, 0), "#46C79A", (0.0, 5.0))):
            d = V @ np.array(v, np.float64)
            out.append((name, QPointF(d[0], -d[1]), col, look, d[2]))
        return out

    def _gizmo_centre(self) -> QPointF:
        return QPointF(self.width() - 64, 64)

    def _paint_gizmo(self, p: QPainter):
        c = self._gizmo_centre()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(30, 31, 34, 220))
        p.drawEllipse(c, 50, 50)
        f = p.font()
        f.setPointSizeF(9.5)
        f.setWeight(QFont.Weight.Bold)
        p.setFont(f)
        for name, d, col, _look, depth in sorted(self.gizmo_axes(), key=lambda a: a[4]):
            end = c + d * 36
            if name in ("E", "N", "Z"):
                p.setPen(QPen(QColor(col), 2))
                p.drawLine(c, end)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(col))
                p.drawEllipse(end, 9, 9)
                p.setPen(QColor("#1B1C1F"))
                p.drawText(QRectF(end.x() - 9, end.y() - 9, 18, 18), Qt.AlignmentFlag.AlignCenter, name)
            else:
                p.setPen(QPen(QColor(col), 1.5))
                p.setBrush(Qt.BrushStyle.NoBrush)
                p.drawEllipse(end, 6, 6)

    def _gizmo_hit(self, pos: QPointF):
        c = self._gizmo_centre()
        for name, d, _col, look, _depth in self.gizmo_axes():
            end = c + d * 36
            if (end - pos).manhattanLength() < 13:
                return name, look
        return None

    # ── picking: a ray through the height field ─────────────────────
    def ray(self, x: float, y: float):
        inv = np.linalg.inv(self.mvp())
        nx, ny = 2.0 * x / max(1, self.width()) - 1.0, 1.0 - 2.0 * y / max(1, self.height())
        a = inv @ np.array([nx, ny, -1.0, 1.0])
        b = inv @ np.array([nx, ny, 1.0, 1.0])
        a, b = a[:3] / a[3], b[:3] / b[3]
        d = b - a
        return a, d / max(np.linalg.norm(d), 1e-12)

    def height_at(self, u: float, v: float) -> float:
        D = self.data
        x = min(max(u * D["w"] - 0.5, 0.0), D["w"] - 1.0)
        y = min(max(v * D["h"] - 0.5, 0.0), D["h"] - 1.0)
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, D["w"] - 1), min(y0 + 1, D["h"] - 1)
        fx, fy = x - x0, y - y0
        d = D["depth"]
        return float((d[y0, x0] * (1 - fx) + d[y0, x1] * fx) * (1 - fy)
                     + (d[y1, x0] * (1 - fx) + d[y1, x1] * fx) * fy)

    def pick(self, x: float, y: float):
        """The world point under the screen point (x, y), or None."""
        if self.data is None:
            return None
        o, d = self.ray(x, y)
        W, H = self.world_size()
        zs = self.zscale()
        zmin = zs * float(self.hrange[0]) - 1e-6
        zmax = zs * float(self.hrange[1]) + 1e-6
        # The slab the relief lives in: z from zmin to zmax.
        ts = []
        if abs(d[2]) > 1e-9:
            ts = sorted([(zmin - o[2]) / d[2], (zmax - o[2]) / d[2]])
        else:
            ts = [0.0, 50.0]
        t0, t1 = max(ts[0], 0.0), ts[1]
        if t1 <= t0:
            return None
        n = 400
        prev = None
        for i in range(n + 1):
            t = t0 + (t1 - t0) * i / n
            q = o + d * t
            u, v = q[0] / W + 0.5, 0.5 - q[1] / H
            if not (0 <= u <= 1 and 0 <= v <= 1):
                prev = None
                continue
            below = q[2] - self.height_at(u, v) * zs
            if prev is not None and prev[1] > 0 >= below:
                lo, hi = prev[0], t                         # bisect between the two samples
                for _ in range(24):
                    mid = (lo + hi) / 2
                    qm = o + d * mid
                    um, vm = qm[0] / W + 0.5, 0.5 - qm[1] / H
                    if qm[2] - self.height_at(um, vm) * zs > 0:
                        lo = mid
                    else:
                        hi = mid
                q = o + d * hi
                return q
            prev = (t, below)
        return None

    def uv_of(self, q) -> tuple[float, float]:
        W, H = self.world_size()
        return float(q[0] / W + 0.5), float(0.5 - q[1] / H)

    def measure(self) -> dict | None:
        """The two measure points: distance across, height difference, along the surface."""
        if len(self.points) != 2 or self.data is None:
            return None
        D = self.data
        (u1, v1), (u2, v2) = (self.uv_of(q) for q in self.points)
        kx, ky = self.full_px                             # in the photographs' pixels
        dx, dy = (u2 - u1) * D["w"] * kx, (v2 - v1) * D["h"] * ky
        h1, h2 = self.height_at(u1, v1), self.height_at(u2, v2)
        dh_px = (h2 - h1) * D["s_px"] * kx                # height in pixels, from the normals
        out = dict(across_px=math.hypot(dx, dy), dh_px=dh_px,
                   three_d_px=math.sqrt(dx * dx + dy * dy + dh_px * dh_px))
        mm = self.scale_mm
        if mm:
            out["across_mm"] = out["across_px"] * mm
            out["three_d_mm"] = out["three_d_px"] * mm
            if self.z_mm_per_unit:
                out["dh_mm"] = dh_px * self.z_mm_per_unit
                out["dh_from"] = "height scale"
            else:
                out["dh_mm"] = dh_px * mm
                out["dh_from"] = "the normals"
        return out

    def section_profile(self, n: int = 256, features: bool = True) -> dict | None:
        """Heights along the section line: distances and heights (mm when known)."""
        if not self.section or self.data is None:
            return None
        return self.profile_between(*self.section, n=n, features=features)

    def profile_between(self, a, b, n: int = 256, features: bool = False) -> dict | None:
        """Heights along any line from (u, v) *a* to *b*: the same dict a section gives."""
        if self.data is None:
            return None
        D = self.data
        (u1, v1), (u2, v2) = a, b
        ts = np.linspace(0, 1, n)
        hs = np.array([self.height_at(u1 + (u2 - u1) * t, v1 + (v2 - v1) * t) for t in ts])
        kx, ky = self.full_px                             # in the photographs' pixels
        k = kx
        length_px = math.hypot((u2 - u1) * D["w"] * kx, (v2 - v1) * D["h"] * ky)
        mm = self.scale_mm
        if mm and self.z_mm_per_unit:
            z = hs * D["s_px"] * k * self.z_mm_per_unit
            unit = "mm"
        elif mm:
            z = hs * D["s_px"] * k * mm
            unit = "mm (from the normals)"
        else:
            z = hs * D["s_px"] * k
            unit = "px"
        dist = ts * (length_px * mm if mm else length_px)
        dz = z - np.linspace(z[0], z[-1], n)               # about the straight line
        from helio.core.metrology import profile_features
        W, H = D["w"] * kx, D["h"] * ky                     # the photographs' pixels
        return dict(dist=dist, z=z, unit=unit, length=float(dist[-1]),
                    length_unit="mm" if mm else "px",
                    rq=float(np.sqrt(np.mean(dz ** 2))), rz=float(np.ptp(dz)),
                    ra=float(np.mean(np.abs(dz))),
                    a_px=(u1 * W, v1 * H), b_px=(u2 * W, v2 * H),
                    features=profile_features(dist, z) if features else None)

    def section_series(self, count: int, spacing: float, n: int = 256) -> dict | None:
        """The section and *count* - 1 copies of it, *spacing* apart (mm when a scale is set, else pixels),
        evenly to both sides of it: a list of profiles (each with its ``offset``, in that unit, positive to
        the right looking A to B), the lines drawn on the relief, and how many fell off the surface."""
        if not self.section or self.data is None:
            return None
        from helio.core.sections import offset_lines
        D = self.data
        kx, ky = self.full_px
        mm = self.scale_mm
        spacing_px = float(spacing) / mm if mm else float(spacing)
        lines, dropped = offset_lines(*self.section, int(count), spacing_px, (D["w"] * kx, D["h"] * ky))
        profs = []
        for ln in lines:
            pr = self.profile_between(ln["a"], ln["b"], n=n, features=False)
            pr["offset"] = ln["offset"] * mm if mm else ln["offset"]
            pr["a_uv"], pr["b_uv"] = ln["a"], ln["b"]
            profs.append(pr)
        self.series_lines = [(ln["a"], ln["b"]) for ln in lines] if len(lines) > 1 else []
        self.update()
        return dict(profiles=profs, dropped=dropped, count=int(count), spacing=float(spacing),
                    length_unit="mm" if mm else "px", unit=profs[0]["unit"] if profs else "")

    def clear_series(self):
        if self.series_lines:
            self.series_lines = []
            self.update()

    # ── interaction ──────────────────────────────────────────────────
    def mousePressEvent(self, e):
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        pos = e.position()
        hit = self._gizmo_hit(pos)
        if hit is not None:
            name, (az, el) = hit
            self.cam.animate_to(az=az, el=el)
            self._timer.start()
            return
        if self.flood_pick and e.button() == Qt.MouseButton.LeftButton:
            q = self.pick(pos.x(), pos.y())
            if q is not None:
                u, v = self.uv_of(q)
                self.flood_pick = False
                self.setCursor(Qt.CursorShape.CrossCursor if self.tool in ("measure", "section", "area")
                               else Qt.CursorShape.ArrowCursor)
                self.set_flood_level((self.height_at(u, v) - self._lowest()) * self.flood_scale()[0])
            return
        if self.tool == "area" and e.button() == Qt.MouseButton.LeftButton:
            q = self.pick(pos.x(), pos.y())
            if q is not None:
                if self.area_done:                           # a new region
                    self.area_uv, self.area_done, self.area_result = [], False, None
                self.area_uv.append(self.uv_of(q))
                self._readout()
                self.update()
            return
        if self.tool == "measure" and e.button() == Qt.MouseButton.LeftButton:
            q = self.pick(pos.x(), pos.y())
            if q is not None:
                self.points = (self.points + [q])[-2:] if len(self.points) < 2 else [q]
                m = self.measure()
                if m:
                    self.measured.emit(m)
                self._readout()
                self.update()
            return
        if self.section and self.tool in ("", "section") and e.button() == Qt.MouseButton.LeftButton:
            # an end of the section: drag it (Measure keeps its clicks)
            for i, q in enumerate(self.section_ends_screen()):
                if q is not None and (q - pos).manhattanLength() < 16:
                    self._press = ("section_end", i)
                    return
        if self.tool == "section" and e.button() == Qt.MouseButton.LeftButton:
            q = self.pick(pos.x(), pos.y())
            if q is not None:
                a = self.uv_of(q)
                self.section = (a, a)
                self._press = ("section", pos)
            return
        self._press = ("drag", pos, self.cam.az, self.cam.el, self.cam.target.copy())

    def mouseMoveEvent(self, e):
        pos = e.position()
        shift = bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if self._press is None:
            self._hover_section(pos)
            return
        if self._press[0] in ("section", "section_end"):
            q = self.pick(pos.x(), pos.y())
            if q is not None and self.section:
                i = 1 if self._press[0] == "section" else self._press[1]
                uv = self.uv_of(q)
                if shift:
                    uv = self._straight(uv, self.section[1 - i])
                ends = list(self.section)
                ends[i] = uv
                self.section = tuple(ends)
                if self._press[0] == "section_end":
                    self._section_changed(live=True)     # the chart follows; features on release
                else:
                    self.update()
            return
        _k, p0, az0, el0, tgt0 = self._press
        dx, dy = pos.x() - p0.x(), pos.y() - p0.y()
        mods = e.modifiers()
        k = 0.15 if mods & Qt.KeyboardModifier.AltModifier else 1.0
        if int(Qt.Key.Key_L) in self._held or self.tool == "light":
            # the light: drag sideways turns it, up and down raises it
            az = math.degrees(math.atan2(self.light[0], self.light[1])) % 360.0
            el = math.degrees(math.asin(max(-1.0, min(1.0, self.light[2]))))
            az = (az + dx * 0.5 * k) % 360.0
            el = max(3.0, min(89.0, el - dy * 0.3 * k))
            self._press = ("drag", pos, self.cam.az, self.cam.el, self.cam.target.copy())
            a, ee = math.radians(az), math.radians(el)
            self.light = np.array([math.sin(a) * math.cos(ee), math.cos(a) * math.cos(ee), math.sin(ee)])
            self.lightMoved.emit(float(self.light[0]), float(-self.light[1]), float(self.light[2]), True)
            self.update()
            return
        if mods & Qt.KeyboardModifier.ShiftModifier or int(Qt.Key.Key_Space) in self._held:
            V = self.cam.view()[:3, :3]
            right, up = V[0], V[1]
            s = self.cam.dist * 0.0018 * k
            self.cam.target = tgt0 - right * dx * s + up * dy * s
        else:
            self.cam.az = (az0 - dx * 0.35 * k) % 360.0
            self.cam.el = max(2.0, min(89.5, el0 + dy * 0.3 * k))
        self.update()

    def _hover_section(self, pos):
        """The pointer near the section line: where along it, for the chart."""
        if not self.section or self.data is None:
            return
        a, b = self.section_ends_screen()
        t = -1.0
        if a is not None and b is not None:
            ab = b - a
            L2 = ab.x() ** 2 + ab.y() ** 2
            if L2 > 1:
                tt = ((pos - a).x() * ab.x() + (pos - a).y() * ab.y()) / L2
                near = a + ab * max(0.0, min(1.0, tt))
                if 0.0 <= tt <= 1.0 and (near - pos).manhattanLength() < 14:
                    t = float(tt)
        self.set_section_cursor(t)
        self.sectionHover.emit(t)

    def mouseReleaseEvent(self, e):
        if self._press and self._press[0] in ("section", "section_end"):
            prof = self.section_profile()
            if prof:
                self.sectioned.emit(prof)
            self._readout()
        elif int(Qt.Key.Key_L) in self._held or self.tool == "light":
            self.lightMoved.emit(float(self.light[0]), float(-self.light[1]), float(self.light[2]), False)
        self._press = None

    def wheelEvent(self, e):
        k = 0.15 if e.modifiers() & Qt.KeyboardModifier.AltModifier else 1.0
        steps = e.angleDelta().y() / 120.0
        self.cam.dist = float(max(0.05, min(40.0, self.cam.dist * (0.88 ** (steps * k)))))
        self.update()
        if self.adaptive:
            self._readout()                       # how finely it is divided here

    def keyPressEvent(self, e):
        k = int(e.key())
        K = Qt.Key
        if k in (int(K.Key_L), int(K.Key_Space)):
            if not e.isAutoRepeat():
                self._held.add(k)
            return
        mods = e.modifiers()
        actions = {K.Key_F: self.fit, K.Key_Home: lambda: self.look("home"),
                   K.Key_7: lambda: self.look("top"), K.Key_1: lambda: self.look("front"),
                   K.Key_3: lambda: self.look("side"), K.Key_5: lambda: self.look("light"),
                   K.Key_C: lambda: self.set_contours(not self.contours),
                   K.Key_W: lambda: self.set_flood(not self.flood_on),
                   K.Key_X: self._wire_key,
                   K.Key_M: lambda: self.set_tool("" if self.tool == "measure" else "measure"),
                   K.Key_S: lambda: self.set_tool("" if self.tool == "section" else "section"),
                   K.Key_A: lambda: self.set_tool("" if self.tool == "area" else "area"),
                   K.Key_Return: self.finish_area, K.Key_Enter: self.finish_area,
                   K.Key_BracketLeft: lambda: self.set_exaggeration(self.exaggeration / 1.25),
                   K.Key_BracketRight: lambda: self.set_exaggeration(self.exaggeration * 1.25),
                   K.Key_Question: self.toggle_keys, K.Key_Slash: self.toggle_keys,
                   K.Key_T: lambda: self.message.emit("turntable"),
                   K.Key_Escape: lambda: self.set_tool("")}
        fn = {int(key): f for key, f in actions.items()}.get(k)
        if fn is not None and not (mods & (Qt.KeyboardModifier.ControlModifier
                                           | Qt.KeyboardModifier.MetaModifier)):
            fn()
            self.message.emit("changed")
            return
        super().keyPressEvent(e)

    def _wire_key(self):
        self.cycle_wire()
        self.message.emit("wire")                 # the panel's mesh boxes follow

    def keyReleaseEvent(self, e):
        if not e.isAutoRepeat():
            self._held.discard(int(e.key()))
        super().keyReleaseEvent(e)

    def _tick(self):
        if not self.cam.step():
            self._timer.stop()
        self.update()

    def _readout(self):
        if self.data is None:
            self._readout_text = "Solve a capture to see it in 3D"
        else:
            D = self.data
            nt = D.get("normals_tex")
            th, tw = (D["normals"] if nt is None else nt).shape[:2]
            if self.adaptive and self.tess_ok:
                k = self.centre_level()
                mesh = (f"adaptive mesh: {self.mesh_counts()[0] / 1e6:.1f} M points, "
                        + (f"×{k:.1f} finer here" if k > 1.05 else "come closer to refine")
                        + f" (up to ×{self.tess_max():.0f})")
            elif self.adaptive and self.tess_ok is False:
                mesh = f"mesh 1 : 1 (no adaptive mesh here: {self.tess_error[:60]})"
            elif self.mesh_used() == self.mesh_step:
                mesh = f"mesh {self.ratio(self.mesh_used())}, {self.mesh_counts()[0] / 1e6:.1f} M points"
            else:
                mesh = None
            bits = [f"shaded by the normals at {tw:,} × {th:,}" if self.normal_mapped
                    else "shaded by the heights" if self.smooth_shading else "shaded by the mesh",
                    (mesh if mesh else
                     f"mesh {self.ratio(self.mesh_used())} ({self.ratio(self.mesh_step)} would pass "
                     f"{self.MAX_MESH_POINTS // 1_000_000} million points)"),
                    f"exaggeration ×{self.exaggeration:.1f}"]
            if self.wire_mode == "only":
                bits.insert(2, "wire only")
            if self.shading == "exaggerated":
                bits.insert(1, f"exaggerated shading, {int(self.exag_scales)} scales")
            elif self.shading == "hillshade":
                bits.insert(1, f"hillshade from 8 directions at {self.hs_alt:.0f}°")
            if self.tool == "measure":
                m = self.measure()
                if m:
                    if "across_mm" in m:
                        bits = [f"{m['three_d_mm']:.2f} mm along · {m['across_mm']:.2f} mm across · "
                                f"Δh {m['dh_mm']:+.3f} mm (from {m['dh_from']})"]
                    else:
                        bits = [f"{m['three_d_px']:.1f} px · Δh {m['dh_px']:+.2f} px (set a scale for mm)"]
                else:
                    bits = ["Measure: click two points on the relief"]
            elif self.tool == "section" and not self.section:
                bits = ["Section: drag a line across the relief"]
            elif self.tool == "area":
                a = self.area_result
                if a:
                    u = a["unit"]
                    bits = [f"Area {a['area']:.4g} {u}² (surface {a['surface']:.4g})",
                            f"volume below {a['below']:.4g} {u}³, above {a['above']:.4g} {u}³",
                            f"deepest {a['deepest']:.3g} {u}"
                            + ("" if u == "mm" else "  (set a scale for mm)")]
                else:
                    bits = [f"Area: click round a region ({len(self.area_uv)} points), "
                            "double-click or Enter to close it"]
            self._readout_text = "  ·  ".join(bits)
        self._place_overlays()

    def fps(self) -> float:
        if not self._frames:
            return 0.0
        return 1.0 / max(1e-6, float(np.mean(self._frames)))
