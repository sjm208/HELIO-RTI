"""The high-detail 3D window: Helio's own 3D view, built large.

The 3D relief in Inspect is built from up to 7 million points so it turns
smoothly beside everything else; this window is the same view (toolbar,
gizmo, key card, measuring, and a section A–B with its card at the foot)
built from up to 9 million, shaded and coloured by the solve's normals and
colour at 4.5K, 8K or the solve's full size (as large as the graphics card
takes), for close looking, pictures, turntables and flyovers.  It follows the
3D panel: Height, light, material, clay, tone, contours, shading and shape,
whatever moves them; dragging the light in it moves the one light everywhere.

Its geometry is adaptive: a lighter mesh divided on the graphics card
(tessellation) wherever it is large on screen, so it stays tight however
close you come, its new points lifted by the fine heights from the full-size
depth; far off it costs no more than the mesh.  Its own panel, Expert
tools, is what the 3D relief in Inspect does not have: the geometry and
shading detail, its own choice of shading by the normals (smooth when off),
the wire on its own, an orthographic camera, a headlight, a slow spin, the
light set by number or raking from a corner, a second light, fine detail
sharpened from the normals, cavity shading, RTI-style specular enhancement,
colour by slope or curvature, and saved views.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QScrollArea, QVBoxLayout, QWidget

from helio_qt.widgets import button, label

HIGH_DETAIL_POINTS = 9_000_000
HIGH_DETAIL_POINTS_LARGE = 16_000_000    # 32 million triangles, about 500 MB: 32 GB of memory or more


def high_detail_points() -> int:
    """The points the high-detail window's mesh may have: more on a machine with the memory."""
    from helio.utils.memory_plan import system_memory
    return HIGH_DETAIL_POINTS_LARGE if system_memory()[0] >= 32 * 2 ** 30 else HIGH_DETAIL_POINTS

DETAIL_SIDE = 4500            # the normals and colour the mesh is shaded with, long side, at 4.5K
DETAILS = (("4500", "4.5K"), ("8000", "8K"), ("full", "Full"))
DETAIL_TIPS = {"4500": "The normals and colour at 4,500 px on the long side: light on memory.",
               "8000": "At 8,000 px on the long side: the fine tool marks and fibres.",
               "full": "At the solve's own full size, every photographed pixel a normal, as large "
                       "as the graphics card takes (16,384 px on most). Needs memory: about "
                       "20 bytes a pixel."}
ADAPTIVE_BASE = 3            # the adaptive mesh: every third point, divided where it is close
GEOMETRY_TIPS = {"adaptive": "A lighter mesh divided on the graphics card wherever it is close: "
                             "tight geometry close up, quick far off. Up to 64 times finer.",
                 "1": "Every point a vertex, everywhere (up to 9 million; 16 million with 32 GB of memory)",
                 "2": "Every second point", "4": "A light mesh: the wire reads clearly"}
REFINE = (("finest", "Finest", 1.5), ("fine", "Fine", 3.0), ("fast", "Fast", 6.0))
REFINE_PX = {k: px for k, _t, px in REFINE}
MAX_VIEWS = 9
EXPERT = dict(enhance=0.0, cavity=0.0, spec_boost=0.0, light2_offset=None, colour_by="surface",
              slope_max=45.0, shading="standard")
SHADINGS = (("standard", "Standard"), ("exaggerated", "Exaggerated"), ("hillshade", "Hillshade ×8"))
SHADING_TIPS = {
    "standard": "The one light (and a second, if chosen), as the 3D panel sets it",
    "exaggerated": "Exaggerated Shading (Rusinkiewicz et al. 2006), as in the 2D views: each scale "
                   "lit as if its neighbourhood faced the light, so fine scratches and broad "
                   "undulation read together. Follows the light as you move it.",
    "hillshade": "Multi-directional hillshade (Mark 1992, the USGS weights): eight lights round the "
                 "compass at one height, so no direction hides a feature. The light's direction "
                 "does not matter; its height is set below."}
RAKING = (("NW", 315.0), ("NE", 45.0), ("SW", 225.0), ("SE", 135.0))


def fine_heights(D, size, mesh_size, depth_range):
    """What the full-size depth has that the mesh's heights do not, at *size*
    (w, h): the depth there, less the mesh's heights brought up to it, in the
    mesh's height units (depth / *depth_range*).  None without the depth."""
    import cv2
    dv = getattr(D, "depth_vis_p", None)
    if dv is None:
        return None
    tw, th = int(size[0]), int(size[1])
    hi = cv2.resize(np.asarray(dv, np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    lo = cv2.resize(cv2.resize(hi, (int(mesh_size[0]), int(mesh_size[1])), interpolation=cv2.INTER_AREA),
                    (tw, th), interpolation=cv2.INTER_LINEAR)
    hi -= lo
    hi /= max(float(depth_range), 1e-9)
    return np.nan_to_num(hi, copy=False)


def first_detail() -> str:
    """Every pixel on a machine with room for a 100 MP capture (about 2 GB of it), 8K on one
    with the memory for that, else 4.5K."""
    from helio.utils.memory_plan import system_memory
    total = system_memory()[1]
    return "full" if total >= 32 * 2 ** 30 else "8000" if total >= 12 * 2 ** 30 else "4500"


class HighDetail3D(QWidget):
    def __init__(self, win):
        super().__init__(win, Qt.WindowType.Window)
        from helio_qt.relief3d import ReliefView
        self.win = win
        self.setObjectName("HighDetail3D")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.resize(1400, 900)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 12)
        lay.setSpacing(8)
        head = QHBoxLayout()
        head.setSpacing(8)
        self.title = label("High-detail 3D", "section")
        head.addWidget(self.title)
        self.note = label("", "caption")
        head.addWidget(self.note, 1)
        for text, tip, fn in (
                ("Picture…", "This 3D view as a picture: as on screen, or drawn afresh at "
                 "4K, 6K or 8K.",
                 lambda: self.win.save_relief_picture(relief=self.view)),
                ("Turntable…", "The surface turning once round, as MP4 or GIF.",
                 lambda: self.win.record_relief_turntable(relief=self.view)),
                ("Flyover…", "One smooth loop round the surface, dipping low where the light "
                 "rakes and moving in on the detail; loops without a jump. MP4 or GIF.",
                 lambda: self.win.record_relief_flyover(relief=self.view)),
                ("Light sweep…", "The camera still, the light circling the surface as in RTI. "
                 "MP4 or GIF.", lambda: self.win.record_relief_sweep(relief=self.view)),
                ("Fit", "The whole surface in view.  F", lambda: self.view.fit())):
            b = button(text, "quiet", tip)
            b.clicked.connect(fn)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)        # the keys stay with the 3D view
            head.addWidget(b)
        # Above Helio's main window, even when that is clicked (a floating
        # panel: it steps aside for other programs and comes back with Helio).
        self.btn_expert = button("Expert tools", "quiet", "Show or hide this window's own panel: "
                                 "geometry and shading detail, wire, camera, light, enhancement, "
                                 "colour by slope or curvature, and saved views.")
        self.btn_expert.setCheckable(True)
        self.btn_expert.setChecked(True)
        self.btn_expert.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        head.addWidget(self.btn_expert)
        self.btn_top = button("Keep on top", "quiet", "Keep this window above Helio's main "
                              "window, even when you click the main window.")
        self.btn_top.setCheckable(True)
        self.btn_top.setChecked(True)
        self.btn_top.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_top.toggled.connect(self.keep_on_top)
        head.addWidget(self.btn_top)
        lay.addLayout(head)
        self.keep_on_top(True)
        self.view = ReliefView()
        self.view.MAX_MESH_POINTS = high_detail_points()    # this one may be larger
        self.view.mesh_step = ADAPTIVE_BASE
        self.view.adaptive = True
        self.view.tess_px = REFINE_PX["fine"]
        self.view.smooth_shading = True                     # never the flat face of a triangle
        self.spin_timer = QTimer(self)
        self.spin_timer.setInterval(16)
        self.spin_timer.timeout.connect(self._spin_step)
        ins = self.win.workspaces["Inspect"]
        self.view.lightMoved.connect(ins._relief_light)      # the one light, from here too
        self.view.message.connect(self._view_message)
        self.view.lightMoved.connect(lambda *_a: self._show_light())
        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self.view, 1)
        self.expert = self._build_expert()
        body.addWidget(self.expert)
        self.btn_expert.toggled.connect(self.expert.setVisible)
        lay.addLayout(body, 1)
        # The key card's keys (L, M, S, C, F, ?) go to the 3D view from the
        # moment the window opens: with nothing focused they went nowhere,
        # and an L-drag turned the model instead of moving the light.
        self.setFocusProxy(self.view)
        # This window's own section: drawn here, its card at the foot.
        from helio_qt.section_card import SectionCard
        self.card = SectionCard(share="Show on the 2D image")
        self.card.chk_share.setToolTip("The same line on Inspect's 2D picture and its 3D relief, "
                                       "with its own card there, while this is ticked.")
        self.card.share_toggled.connect(self._share_section)
        self.view.sectioned.connect(self._section_moved)
        self.card.hovered.connect(self.view.set_section_cursor)
        self.card.clear_requested.connect(self.view.clear_section)
        self.card.save_requested.connect(lambda: self.card.save(self.win, self.win.project.name))
        self.card.sheet_requested.connect(lambda: self.win.add_sheet_section("hd"))
        self.card.attach(lambda: self.view, self.win, lambda: self.win.project.name)
        self.view.sectioned.connect(self.card.set_profile)
        self.view.sectioned.connect(lambda _p: self.name_section())
        self.view.sectionHover.connect(self.card.set_cursor)
        lay.addWidget(self.card)
        from helio_qt.flood_card import FloodCard
        self.flood = FloodCard()
        self.flood.attach(lambda: self.view, self.win, lambda: self.win.project.name)
        self.view.floodChanged.connect(self.flood.set_info)
        lay.addWidget(self.flood)
        self._detail_of = None
        self._detail_side = (0, 0)
        self._of = None
        self._look = None
        self._panel_light = None
        self._shape_gen = 0
        self.timer = QTimer(self)
        self.timer.setInterval(120)
        self.timer.timeout.connect(self._follow)

    # ── the surface ───────────────────────────────────────────────────
    def arrays(self):
        ins = self.win.workspaces["Inspect"]
        _r, D = ins._relief_source()
        return D, (ins._relief_arrays(D, high_detail_points()) if D is not None else None)

    def load(self) -> bool:
        """The solved surface (the selection's, when there is one), built large."""
        ins = self.win.workspaces["Inspect"]
        D, A = self.arrays()
        if A is None:
            return False
        trust = None if A["sigma"] is None else np.clip(np.asarray(A["sigma"], np.float32) / 10.0, 0, 1)
        p = self.win.project
        self.view.set_surface(A["depth"], A["normals"], A["albedo"], roughness=A["rough"],
                              trust=trust, view=ins._view_for_relief(getattr(ins, "_last_image", None)),
                              scale_mm=p.scale_mm_per_px, z_mm_per_unit=p.z_mm_per_unit,
                              full_size=(D.w, D.h))
        self._geometry(self.geom_seg.current() or "adaptive")
        self._of, self._look, self._panel_light = D, None, None
        self._detail_of = None
        h, w = A["depth"].shape[:2]
        sel = " · the selection" if self.win.selection else ""
        self.setWindowTitle(f"Helio — high-detail 3D · {p.name}")
        self.title.setText(f"High-detail 3D · {p.name}{sel}")
        self._built = (f"{w * h / 1e6:.1f} M points" if w * h >= 1e6 else f"{w * h:,} points") \
            + f" ({w:,} × {h:,})"
        self.note.setText(f"{self._built} · the 3D panel's settings, live")
        self._follow()
        self.reshape()
        self.sharpen(D)
        return True

    def detail_side(self, D=None) -> int:
        """The long side the shading is made at, for the detail chosen:
        never more than the solve has or the graphics card takes."""
        D = self._of if D is None else D
        full = max(D.w, D.h) if D is not None else 0
        key = self.detail_seg.current() or "4500"
        side = full if key == "full" else int(key)
        return int(max(1, min(side, full or side, self.view.max_texture)))

    def sharpen(self, D):
        """The solve's normals and colour at the detail chosen (4.5K, 8K or
        full) on the long side, and the fine heights at the same size (what
        the full-size depth has that the mesh's heights do not), made off the
        main thread, for the mesh to be shaded with and lifted by."""
        from helio.application.viewer3d import detail_textures
        if getattr(D, "normals_full", None) is None or self.view.data is None:
            return
        side = self.detail_side(D)
        if self._detail_of is D and self._detail_side[0] == side:
            return
        self._detail_of = D
        self._detail_side = (side, self._detail_side[1] + 1)
        gen = self._detail_side[1]
        V = self.view.data
        mesh_w, mesh_h, rng = V["w"], V["h"], float(V["depth_range"])
        self.detail_note.setText(f"Making the normals and fine heights at {side:,} px on the long side…")

        def make():
            out = detail_textures(D, mesh_w, side)
            return out, (None if out is None else fine_heights(D, out[2], (mesh_w, mesh_h), rng))

        def done(res, err):
            if gen != self._detail_side[1] or self._of is not D or self.view.data is None:
                return
            if err:
                self.detail_note.setText(f"Could not make that detail: {err}")
                return
            out, fine = res
            if out is None:                   # the mesh already has all the solve has
                self.view.set_detail(None, None)
            else:
                self.view.set_detail(out[0], out[1])
            self.view.set_fine_heights(fine)
            self._say_detail()
        self.win.run_background(make, done)

    def _say_detail(self):
        D = self.view.data
        if D is None:
            return
        nt = D.get("normals_tex")
        th, tw = (D["normals"] if nt is None else nt).shape[:2]
        fine = D.get("hdetail") is not None
        self.detail_note.setText(f"Normals at {tw:,} × {th:,}"
                                 + ("" if nt is not None else ", the mesh's own (the solve has no more)")
                                 + (" · fine heights at the same size" if fine else ""))
        self.note.setText(f"{getattr(self, '_built', '')} · shaded at {tw:,} × {th:,} · "
                          "the 3D panel's settings, live")

    def reshape(self):
        """The heights for the panel's shape settings, at this window's size,
        off the main thread; the newest request is the one that lands."""
        if self.view.data is None:
            return
        import cv2
        from helio.application.viewer3d import relief_surface
        ins = self.win.workspaces["Inspect"]
        D, A = self.arrays()
        if A is None:
            return
        r = ins._relief_source()[0]
        d_hp = A["depth"]
        full = getattr(r, "depth_normalised_full", None)
        d_full = d_hp if full is None else cv2.resize(np.asarray(full, np.float32),
                                                       (d_hp.shape[1], d_hp.shape[0]),
                                                       interpolation=cv2.INTER_AREA)
        params = dict(ins._shape_params(), smooth_scale=max(1.0, d_hp.shape[1] / 1500.0))
        self._shape_gen += 1
        gen = self._shape_gen

        def done(out, err):
            if err or out is None or gen != self._shape_gen or self.view.data is None:
                return
            self.view.set_heights(out)
        self.win.run_background(lambda: relief_surface(d_full, d_hp, **params), done)

    # ── the panel: this window's own tools ────────────────────────────
    def _build_expert(self) -> QScrollArea:
        from helio.domain.view_controls import Control
        from helio_qt.controls import ParamRow, Section, Segmented
        from PySide6.QtWidgets import QCheckBox
        box = QWidget()
        box.setObjectName("ExpertTools")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 10, 4)
        lay.setSpacing(10)
        store = getattr(self.win, "prefs", None)
        sec = None

        def section(title, key, tip, open_=True):
            nonlocal sec
            sec = Section(title, f"hd3d_{key}", store, open_=open_, tip=tip)
            lay.addWidget(sec)

        def caption(text):
            sec.add(label(text, "caption"))

        def row(key, text, lo, hi, val, step, tip, unit=""):
            r = ParamRow(Control(key, text, lo, hi, val, step, tip, unit=unit))
            sec.add(r)
            return r

        def check(text, tip, on, fn):
            c = QCheckBox(text)
            c.setToolTip(tip)
            c.setChecked(on)
            c.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            c.toggled.connect(fn)
            sec.add(c)
            return c

        section("Quality", "quality", "How finely the surface is built and shaded.")
        caption("Geometry")
        self.geom_seg = Segmented([("adaptive", "Adaptive"), ("1", "1 : 1"), ("2", "1 : 2"),
                                   ("4", "1 : 4")], GEOMETRY_TIPS)
        self.geom_seg.set_current("adaptive")
        self.geom_seg.chosen.connect(self._geometry)
        sec.add(self.geom_seg)
        self.refine_seg = Segmented([(k, t) for k, t, _px in REFINE], {
            k: f"Divide the mesh until its edges are about {px:g} px on screen" for k, _t, px in REFINE})
        self.refine_seg.set_current("fine")
        self.refine_seg.setToolTip("How finely the adaptive mesh is divided close up: Finest is "
                                   "the tightest geometry, Fast the lightest on the graphics card.")
        self.refine_seg.chosen.connect(self._refine)
        sec.add(self.refine_seg)
        self.chk_fine = check("Fine geometry from the full-size depth",
                              "The adaptive mesh lifted by the heights at the shading's size (4.5K, "
                              "8K or full), not only the mesh's own: tool marks and fibres in the "
                              "shape itself, close up.", True, self._fine_toggled)
        caption("Shading detail")
        self.detail_seg = Segmented(list(DETAILS), DETAIL_TIPS)
        self.detail_seg.set_current(first_detail())
        self.detail_seg.chosen.connect(lambda _k: self._of is not None and self.sharpen(self._of))
        sec.add(self.detail_seg)
        self.chk_nmap = check("Shade with the photometric normals",
                              "Every pixel lit with its own measured normal: the finest detail. "
                              "Off, it is shaded by the shape of the mesh, smoothly. This "
                              "window's own choice: the 3D panel's box does not change it.",
                              True, self._nmap_toggled)
        self.detail_note = label("", "caption", wrap=True)
        sec.add(self.detail_note)

        section("Display", "display", "The mesh's wire, the camera, and a slow turn.")
        self.wire_seg = Segmented([("off", "Surface"), ("over", "+ Wire"), ("only", "Wire only")],
                                  {"off": "The surface alone",
                                   "over": "The mesh's edges drawn over the surface",
                                   "only": "Only the mesh's edges: the surface hidden (edges "
                                           "behind it stay hidden)"})
        self.wire_seg.set_current("off")
        self.wire_seg.setToolTip("X steps through these in the view. With the adaptive mesh the "
                                 "wire shows it dividing as you come closer.")
        self.wire_seg.chosen.connect(self.view.set_wire_mode)
        sec.add(self.wire_seg)
        self.chk_ortho = check("Orthographic: no perspective",
                               "Parallel lines stay parallel and every part is at the same scale, "
                               "as a measured drawing: for plans and elevations.", False,
                               lambda on: self._set_view(ortho=on))
        self.chk_head = check("Headlight: the light follows the camera",
                              "A light from just over your shoulder as you turn, so whatever you "
                              "face is lit. Off, the one light stays where it is set.", False,
                              lambda on: self._set_view(headlight=on))
        self.chk_spin = check("Spin slowly", "The surface turning slowly round, to look at it from "
                              "every side (Turntable… records a turn).", False, self._spin)

        section("Light", "light", "How the surface is lit: the shading, the light by number, "
                "raking from a corner, and a second light.")
        caption("Shading")
        self.shading_seg = Segmented(list(SHADINGS), SHADING_TIPS)
        self.shading_seg.set_current("standard")
        self.shading_seg.chosen.connect(self._shading)
        sec.add(self.shading_seg)
        sh = getattr(self.win, "shared", {}) or {}
        self.exag_scales_row = row("exag_scales", "Scales", 1, 8, int(sh.get("exag_scales", 4)), 1,
                                   "How many scales are lit at once, each twice as broad as the last.")
        self.exag_sigma_row = row("exag_sigma", "Finest scale", 0.5, 8.0, float(sh.get("exag_sigma", 2.0)),
                                  0.1, "Smoothing of the finest scale, in the normals' pixels; each "
                                  "further scale doubles it.", "px")
        self.hs_alt_row = row("hs_alt", "Light height", 5, 90, 45, 1, "The height of the eight "
                              "hillshade lights above the horizon: low for the faintest relief.", "°")
        self.exag_scales_row.changed.connect(lambda _k, v, _f: self.view.set_expert(exag_scales=int(v)))
        self.exag_sigma_row.changed.connect(lambda _k, v, _f: self.view.set_expert(exag_sigma=float(v)))
        self.hs_alt_row.changed.connect(lambda _k, v, _f: self.view.set_expert(hs_alt=float(v)))
        self.view.exag_scales = int(self.exag_scales_row.value())
        self.view.exag_sigma = float(self.exag_sigma_row.value())
        for r in (self.exag_scales_row, self.exag_sigma_row, self.hs_alt_row):
            r.setVisible(False)
        caption("Direction")
        self.az_row = row("az", "Azimuth", 0, 359, 315, 1, "The compass direction the light "
                          "comes from: 0° north, 90° east.  Moves the one light everywhere.", "°")
        self.el_row = row("el", "Elevation", 3, 89, 45, 1, "The light's height: low rakes across "
                          "the surface and shows the faintest relief.", "°")
        for r in (self.az_row, self.el_row):
            r.changed.connect(lambda _k, _v, final: self._light_typed(final))
        grid = QGridLayout()
        grid.setSpacing(4)
        for i, (name, az) in enumerate(RAKING):
            b = button(f"Rake {name}", "quiet", f"A raking light from the {name}, 12° high.")
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _c=False, a=az: self.set_light(a, 12.0))
            grid.addWidget(b, i // 2, i % 2)
        sec.add_layout(grid)
        caption("Second light")
        self.light2_seg = Segmented([("off", "Off"), ("90", "90° round"), ("180", "Opposite")],
                                    {"off": "One light",
                                     "90": "A second light at the same height, 90° round, at 60%: "
                                           "strokes running both ways show at once",
                                     "180": "A second light opposite the first, at 60%: both "
                                            "sides of every groove lit"})
        self.light2_seg.set_current("off")
        self.light2_seg.chosen.connect(
            lambda k: self.view.set_expert(light2_offset=None if k == "off" else float(k)))
        sec.add(self.light2_seg)

        section("Enhance", "enhance", "Make the finest marks easier to read.")
        self.enhance_row = row("enhance", "Fine detail", 0.0, 4.0, 0.0, 0.05,
                               "Sharpens the normals themselves (unsharp masking): the finest "
                               "marks stand out without the shape changing.")
        self.cavity_row = row("cavity", "Cavity", 0.0, 1.0, 0.0, 0.01,
                              "Darkens hollows and lightens ridges by the surface's curvature, "
                              "whatever the light: incisions read as drawn lines.")
        self.spec_row = row("spec", "Specular enhancement", 0.0, 1.0, 0.0, 0.01,
                            "As in RTI: the colour quietened and a tight highlight from the "
                            "normals alone, so every stroke catches the light.")
        for r, attr in ((self.enhance_row, "enhance"), (self.cavity_row, "cavity"),
                        (self.spec_row, "spec_boost")):
            r.changed.connect(lambda _k, v, _f, a=attr: self.view.set_expert(**{a: float(v)}))
        caption("Colour by")
        self.colour_seg = Segmented([("surface", "Surface"), ("slope", "Slope"),
                                     ("curvature", "Curvature")],
                                    {"surface": "The 3D panel's surface (colour, clay, …)",
                                     "slope": "How steep the surface is, from the normals: flat "
                                              "blue to steep yellow",
                                     "curvature": "Hollows blue, ridges red, from the normals"})
        self.colour_seg.set_current("surface")
        self.colour_seg.chosen.connect(self._colour_by)
        sec.add(self.colour_seg)
        self.slope_row = row("slope_max", "Steepest colour", 5.0, 89.0, 45.0, 1.0,
                             "The slope that takes the top colour.", "°")
        self.slope_row.changed.connect(lambda _k, v, _f: self.view.set_expert(slope_max=float(v)))
        self.slope_row.setEnabled(False)
        reset = button("Reset enhancement", "quiet", "Back to the 3D panel's look: standard "
                       "shading, one light, no enhancement, colour by surface, no wire.")
        reset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        reset.clicked.connect(self.reset_expert)
        sec.add(reset)

        section("Views", "views", "Camera positions to come back to: the same view for every "
                "picture of an object.")
        vb = QHBoxLayout()
        vb.setSpacing(4)
        save = button("Save this view", "quiet", "Keep the camera, its perspective and the light, "
                      "to come back to (while Helio is open).")
        save.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        save.clicked.connect(self.save_view)
        vb.addWidget(save)
        clear = button("Clear", "quiet", "Forget the saved views.")
        clear.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        clear.clicked.connect(self.clear_views)
        vb.addWidget(clear)
        sec.add_layout(vb)
        self.views_grid = QGridLayout()
        self.views_grid.setSpacing(4)
        sec.add_layout(self.views_grid)
        self.views: list[dict] = []
        self.view_buttons: list = []

        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(box)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedWidth(310)
        return scroll

    def _shading(self, key: str):
        """Standard, Exaggerated or the eight-direction hillshade, with the
        settings each has."""
        self.exag_scales_row.setVisible(key == "exaggerated")
        self.exag_sigma_row.setVisible(key == "exaggerated")
        self.hs_alt_row.setVisible(key == "hillshade")
        self.light2_seg.setEnabled(key == "standard")
        self.view.set_expert(shading=key)

    def _geometry(self, key: str):
        """Adaptive: a lighter mesh divided on the graphics card where it is
        close; else a fixed mesh of every, every second or fourth point."""
        v = self.view
        adaptive = key == "adaptive"
        v.adaptive = adaptive
        self.refine_seg.setEnabled(adaptive)
        self.chk_fine.setEnabled(adaptive)
        v.set_mesh_step(ADAPTIVE_BASE if adaptive else float(key))

    def _refine(self, key: str):
        self.view.tess_px = REFINE_PX.get(key, 3.0)
        self.view._readout()
        self.view.update()

    def _fine_toggled(self, on: bool):
        self.view.fine_heights = bool(on)
        self.view._readout()
        self.view.update()

    def _nmap_toggled(self, on: bool):
        self.view.normal_mapped = bool(on)
        self.view._readout()
        self.view.update()

    def _set_view(self, **kw):
        for k, val in kw.items():
            setattr(self.view, k, bool(val))
        self.view._readout()
        self.view.update()

    def _spin(self, on: bool):
        if on:
            self.spin_timer.start()
        else:
            self.spin_timer.stop()

    def _spin_step(self):
        v = self.view
        if not self.isVisible() or v.data is None:
            return
        v.cam.az = (v.cam.az + 0.25) % 360.0
        v.update()

    def save_view(self):
        """The camera (and its perspective, and the light) kept, as a button."""
        if len(self.views) >= MAX_VIEWS:
            self.views.pop(0)
        c, v = self.view.cam, self.view
        self.views.append(dict(az=c.az, el=c.el, dist=c.dist, target=c.target.copy(),
                               ortho=v.ortho, light=v.light.copy()))
        self._view_buttons()

    def clear_views(self):
        self.views = []
        self._view_buttons()

    def _view_buttons(self):
        for b in self.view_buttons:
            self.views_grid.removeWidget(b)
            b.deleteLater()
        self.view_buttons = []
        for i, kept in enumerate(self.views):
            b = button(f"View {i + 1}", "quiet", f"Back to view {i + 1}: from {kept['az']:.0f}°, "
                       f"{kept['el']:.0f}° up" + (", orthographic" if kept["ortho"] else ""))
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _c=False, k=i: self.go_view(k))
            self.views_grid.addWidget(b, i // 3, i % 3)
            self.view_buttons.append(b)

    def go_view(self, i: int):
        import math
        kept = self.views[i]
        v = self.view
        self.chk_ortho.setChecked(kept["ortho"])
        v.cam.animate_to(az=kept["az"], el=kept["el"], dist=kept["dist"], target=kept["target"])
        v._timer.start()
        L = kept["light"]
        self.set_light(math.degrees(math.atan2(L[0], L[1])) % 360.0,
                       math.degrees(math.asin(max(-1.0, min(1.0, float(L[2]))))))

    def _colour_by(self, key: str):
        self.slope_row.setEnabled(key == "slope")
        self.view.set_expert(colour_by=key)

    def reset_expert(self):
        self.view.set_expert(**EXPERT)
        self.view.set_wire_mode("off")
        for r in (self.enhance_row, self.cavity_row, self.spec_row, self.slope_row):
            r.set_value(r.c.default)
        for seg, key in ((self.light2_seg, "off"), (self.colour_seg, "surface"), (self.wire_seg, "off"),
                         (self.shading_seg, "standard")):
            seg.set_current(key)
        self._shading("standard")
        self.slope_row.setEnabled(False)

    def set_light(self, az: float, el: float, final: bool = True):
        """The light from compass bearing *az*, *el* degrees high: here and,
        as a drag in the view would, everywhere."""
        import math
        a, e = math.radians(az), math.radians(el)
        v = self.view
        v.light = np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])
        v.update()
        v.lightMoved.emit(float(v.light[0]), float(-v.light[1]), float(v.light[2]), not final)

    def _light_typed(self, final: bool):
        self.set_light(float(self.az_row.value()), float(self.el_row.value()), final)

    def _show_light(self):
        """The light's numbers on the panel, wherever it was moved from."""
        import math
        L = self.view.light
        self.az_row.set_value(round(math.degrees(math.atan2(L[0], L[1])) % 360.0))
        self.el_row.set_value(round(min(89.0, max(3.0, math.degrees(math.asin(
            max(-1.0, min(1.0, float(L[2])))))))))

    def _view_message(self, what: str):
        if what == "turntable":
            self.win.record_relief_turntable(relief=self.view)
        elif what == "wire":
            self.wire_seg.set_current(self.view.wire_mode)

    def keep_on_top(self, on: bool):
        """Float above the main window (on), or be an ordinary window (off).
        The window type is set whole: Tool and Window share bits, so
        clearing Tool alone could clear Window too."""
        want = Qt.WindowType.Tool if on else Qt.WindowType.Window
        flags = (self.windowFlags() & ~Qt.WindowType.WindowType_Mask) | want
        if flags == self.windowFlags():
            return
        shown, geo = self.isVisible(), self.geometry()
        self.setWindowFlags(flags)                        # re-made: shown again below
        if shown:
            self.setGeometry(geo)
            self.show()
            self.raise_()
            self.activateWindow()

    def name_section(self):
        """The line's letters (A–B, C–D …, as it will be on the sheet), on its card, chart and 3D line."""
        a, b = self.win.section_letters(self.card.prof)
        self.card.set_letters(a, b)
        self.view.section_letters = (a, b)
        self.view.update()

    # ── the section, shown in Inspect too when asked ──────────────────
    def _section_moved(self, prof: dict):
        """The window's line, sent to Inspect when Show on the 2D image is
        ticked: once it is let go (a drag's live updates stay here)."""
        if self.card.chk_share.isChecked() and (not prof or prof.get("features") is not None):
            self._send_section()

    def _share_section(self, on: bool):
        ins = self.win.workspaces["Inspect"]
        if on:
            self._send_section()
        elif ins.relief is not None and getattr(self, "_sent", False):
            ins.relief.clear_section()
            self._sent = False

    def _send_section(self):
        """Inspect's relief (and so its 2D picture and card) given this line.
        Both views show the same surface, so the same (u, v) ends."""
        ins = self.win.workspaces["Inspect"]
        if self.view.section is None:
            if ins.relief is not None and getattr(self, "_sent", False):
                ins.relief.clear_section()
            self._sent = False
            return
        rv = ins._ensure_relief()
        ins._push_relief()
        if ins.centre_mode == "image":
            ins.relief_box.setVisible(False)
        rv.set_section(*self.view.section)
        self._sent = True

    # ── following the 3D panel ────────────────────────────────────────
    def _look_now(self) -> tuple:
        from helio.application.viewer3d import CLAY
        ins = self.win.workspaces["Inspect"]
        rr = ins.relief_rows
        tone = tuple(float(ins.tone3d_rows[k].value()) for k in ins._TONE_NAMES) \
            if hasattr(ins, "tone3d_rows") else ()
        return (float(rr["exaggeration"].value()), float(rr["ambient"].value()),
                float(rr["diffuse"].value()), float(rr["specular"].value()),
                float(rr["shine"].value()), float(rr["contour"].value()),
                CLAY.get(ins.relief_clay.currentText(), (0.80, 0.76, 0.68)),
                ins.relief_material.current() or "colour", tone,
                ins.chk_relief_contours.isChecked(), ins.chk_relief_nmap.isChecked(),
                ins.viewport.surround)

    def _follow(self):
        """Every setting on the 3D panel, applied here when it changes; a new
        solve or selection rebuilds the surface."""
        if not self.isVisible() or self.win.project.result is None:
            return
        ins = self.win.workspaces["Inspect"]
        _r, D = ins._relief_source()
        if D is not self._of:
            self.load()
            return
        v = self.view
        # The panel's light, only when it changes: a light dragged here is not
        # put back because some other setting moved.
        light = tuple(round(float(c), 4) for c in ins._current_light())
        if light != self._panel_light:
            self._panel_light = light
            v.set_light_screen(*light)
            v.update()
            self._show_light()
        look = self._look_now()
        if look == self._look:
            return
        self._look = look
        (ex, amb, dif, spec, shine, step, clay, mat, tone, contours, nmap, surround) = look
        v.set_exaggeration(ex)
        v.ambient, v.diffuse, v.specular, v.shine = amb, dif, spec, shine
        v.contour_step, v.clay, v.surround = step, clay, surround    # the normals: this window's own
        for name, val in zip(ins._TONE_NAMES.values(), tone):
            v.tone[name] = val
        v.set_material(mat)
        v.set_contours(contours)
        v._readout()
        v.update()

    def showEvent(self, e):
        super().showEvent(e)
        self.timer.start()
        if self.chk_spin.isChecked():
            self.spin_timer.start()
        self.view.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def hideEvent(self, e):
        self.timer.stop()
        self.spin_timer.stop()
        super().hideEvent(e)
