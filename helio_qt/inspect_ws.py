"""Inspect: every view of the solved surface, with the classic panel's tools (VIEW-002).

What it does, as the classic Inspect panel does it (Helio_Pro.py):

* **The light belongs to the view.**  Each view keeps its own light in
  ``engine_settings[view]`` (``lx, ly, lz`` and the compass angles
  ``hs_azi, hs_alt``), restored when the view is chosen
  (``on_engine_change``) and written by the dome as ``update_light`` writes
  it.  Views lit by compass angles (PS Normal-Map Lit) follow the dome too.
  "Use this light in every view" copies it across.
* **The controls are the view's own**, from the measured table in
  helio.domain.view_controls: each moves the picture, none is dead.
* **Layers** (Colour, Clay, Roughness, Confidence, and the depth map),
  colour strength, the confidence overlay's settings, **display
  transforms** (invert normals; mirror, 180°, invert depth), view rotation,
  **split view** against a second view, the **loupe** at full resolution,
  **Safe States**, best raking light, the auto-sweep and undo.
* The picture is drawn by helio.application.viewing.screen_image, which the
  classic window's image path is checked against (tests/test_qt_views.py).

Rendering runs on one worker thread with stale requests dropped, so a slow
view (openness, retinex) never freezes the window; while a slider or the
dome is being dragged a view slower than 150 ms waits for the release.
"""
from __future__ import annotations

import math
import os

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QInputDialog,
                               QLineEdit, QListWidget, QListWidgetItem, QSplitter,
                               QStackedWidget, QToolButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from helio_qt.controls import Histogram, Loupe, ParamRow, Section, Segmented, row_for
from helio_qt.widgets import (FrameList, LightDome, Viewport, button, label, panel,
                              pixmap_from_array, rule)

# While the light is dragged, each view is drawn at the largest scale that
# keeps a frame near DRAG_TARGET_MS (its own measured time decides), and the
# frame the drag ends on at full size.  The scales are fixed steps, so the
# resized arrays for each are made once and kept.
DRAG_TARGET_MS = 16.0
DRAG_SCALES = (1.0, 0.85, 0.7, 0.6, 0.5, 0.42, 0.35)
# Zoomed in past the preview's own resolution, the part on screen is drawn again from
# the full-size arrays and laid over the preview.  It follows the light while a drag
# frame of it (drawing it AND putting it on screen, which the GUI thread does) takes
# no longer than DETAIL_LIVE_MS, and is drawn once the drag ends otherwise; DETAIL_PX
# caps the region (smaller on a machine with little memory).  Measured: a frame of the
# drag costs ~19 ms, and a 0.8 MP detail added 13-30 ms to every one of them, so only a
# detail that is nearly free follows the light; the rest arrives when the light is let go.
DETAIL_LIVE_MS = 8.0
DETAIL_PX = (4_000_000, 9_000_000, 16_000_000)   # under 24 GB of memory, under 48 GB, 48 GB or more
DETAIL_ZOOM_OVER = 1.1                       # the preview must be enlarged this much on screen
LAYER_KEYS = (("Combined Colour", "Colour"), ("Shading", "Clay"),
              ("Roughness Fusion", "Roughness"), ("Confidence Overlay", "Confidence"))
LAYER_TIPS = {
    "Combined Colour": "The view lit over the photograph's colour.",
    "Shading": "Shape only, as grey clay.  C",
    "Roughness Fusion": "The view mixed with the surface's roughness map.",
    "Confidence Overlay": "Marks pixels too dark or too bright in the photographs to trust.",
}


# ── rendering off the main thread ────────────────────────────────────────

class _Signals(QObject):
    done = Signal(str, int, object, float, str)     # channel, generation, result, ms, error


class _Job(QRunnable):
    def __init__(self, owner, channel, gen, fn):
        super().__init__()
        self.owner, self.channel, self.gen, self.fn = owner, channel, gen, fn

    def run(self):
        import time
        if self.owner.gen.get(self.channel) != self.gen:
            return                                   # superseded before it started
        t0 = time.perf_counter()
        try:
            out, err = self.fn(), ""
        except Exception as e:                       # reported on screen, never swallowed
            import logging
            logging.getLogger(__name__).exception("render %s failed", self.channel)
            out, err = None, f"{type(e).__name__}: {e}"
        self.owner.sig.done.emit(self.channel, self.gen, out,
                                 (time.perf_counter() - t0) * 1000.0, err)


class Renderer:
    """One worker; the newest request per channel goes next.

    Each request carries a *tag* naming the picture it draws (the view, the
    data, the layer, the display transforms), so a frame that finishes after
    the light has moved on can still be shown: same picture, a slightly older
    light, newer than what is on screen."""

    def __init__(self):
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(1)
        self.gen: dict[str, int] = {}
        self.tags: dict[tuple, object] = {}
        self.sig = _Signals()

    def request(self, channel: str, fn, tag=None, priority: int = 0) -> int:
        g = self.gen.get(channel, 0) + 1
        self.gen[channel] = g
        self.tags[(channel, g)] = tag
        self.tags.pop((channel, g - 64), None)       # only recent ones are ever asked about
        self.pool.start(_Job(self, channel, g, fn), priority)    # queued work: higher goes first
        return g

    def latest(self, channel: str, gen: int) -> bool:
        return self.gen.get(channel) == gen

    def same_picture(self, channel: str, gen: int) -> bool:
        """Whether frame *gen* draws what the newest request draws (only the light differs)."""
        newest = self.tags.get((channel, self.gen.get(channel, 0)))
        return newest is not None and self.tags.get((channel, gen)) == newest

    def wait(self, ms: int = 120000):
        self.pool.waitForDone(ms)


# ── the workspace ────────────────────────────────────────────────────────

class UnderCards(QWidget):
    """The cards under Inspect's centre (flood, section), in the lower pane of *splitter*: scrolled in their
    share when they need more room, shown only while one of them is, and the share the person drags to
    kept (prefs ``inspect/under_share``).  A folded card takes only its height."""
    KEY = "inspect/under_share"
    SHARE = 0.38                                  # the cards' share of the centre until the divider is moved

    def __init__(self, splitter: QSplitter, cards):
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QScrollArea
        super().__init__()
        self.splitter, self.cards = splitter, list(cards)
        self._events = (QEvent.Type.ShowToParent, QEvent.Type.HideToParent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QScrollArea.Shape.NoFrame)
        self.area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        for c in self.cards:
            v.addWidget(c)
            c.installEventFilter(self)
            fold = getattr(c, "btn_fold", None)
            if fold is not None:                      # folded or unfolded: the pane follows
                fold.toggled.connect(lambda _on: QTimer.singleShot(0, self.place))
        v.addStretch(1)
        self.area.setWidget(inner)
        lay.addWidget(self.area)
        self.setMinimumHeight(70)
        splitter.addWidget(self)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.splitterMoved.connect(lambda _pos, _i: self._remember())
        self.setVisible(False)

    def eventFilter(self, obj, e):
        if e.type() in self._events:
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self.place)          # after the card has its new size
        return False

    def _folded(self) -> bool:
        shown = [c for c in self.cards if c.isVisibleTo(self)]
        return bool(shown) and all(getattr(c, "btn_fold", None) is not None and c.btn_fold.isChecked()
                                   for c in shown)

    def share(self) -> float:
        try:
            from helio_qt.window import settings
            v = float(settings().value(self.KEY, self.SHARE))
        except Exception:
            v = self.SHARE
        return min(0.75, max(0.12, v))

    def _remember(self):
        sizes = self.splitter.sizes()
        if self.isVisible() and not self._folded() and sum(sizes) > 0:
            try:
                from helio_qt.window import settings
                settings().setValue(self.KEY, sizes[1] / float(sum(sizes)))
            except Exception:
                pass

    def place(self):
        """Shown while a card is, at the remembered share (a folded card: just its height)."""
        on = any(c.isVisibleTo(self) for c in self.cards)
        if on != self.isVisible():
            self.setVisible(on)
        if not on:
            return
        total = sum(self.splitter.sizes()) or self.splitter.height()
        if total <= 0:
            return
        if self._folded():
            shown = [c for c in self.cards if c.isVisibleTo(self)]
            for c in shown:
                c.layout().activate()                 # the body just hidden: measure the card as it is now
            want = sum(c.sizeHint().height() for c in shown) + 2
        else:
            want = int(total * self.share())
        want = max(self.minimumHeight(), min(want, total - self.splitter.widget(0).minimumHeight()))
        self.splitter.setSizes([total - want, want])


class InspectWorkspace:
    name = "Inspect"
    FAMILY_ORDER = ("Relighting", "Relief and terrain", "Normals", "Depth", "Decomposition",
                    "Quality control")
    DEPTH_MAP = "Depth map"          # the depth itself, as grey: the classic depth layer
    BACKLIT = "Backlit frame"        # the transmitted-light photograph, lined up (TRN-001)
    SPECIAL = (DEPTH_MAP, BACKLIT)
    # Engines kept for the classic app and old sessions, not offered here:
    # the eight-direction hillshade.
    HIDDEN = frozenset({"PS Normal-Map Lit (8-Dir)"})

    def __init__(self, win):
        self.win = win
        self.project = win.project
        self.reg = win.registry
        self.sidebar, self.center, self.inspector = self.build()
        # Esc with a photograph up: back to the view (a drawing tool on is given up first)
        from PySide6.QtGui import QKeySequence, QShortcut
        self._esc_backs = []
        for host in (self.center, self.sidebar):
            sc = QShortcut(QKeySequence(Qt.Key.Key_Escape), host)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda: self.viewport.cancel_tool() or self.back_to_view())
            sc.setEnabled(False)
            self._esc_backs.append(sc)

    # ── layout ────────────────────────────────────────────────────────
    def build(self):
        from collections import OrderedDict
        from helio.core.shading_engines import _ENGINE_META
        from helio.domain.registry import ENGINE_BLURBS, RENDER_ENGINES, view_name
        self.view_name = view_name
        from helio.domain.view_controls import COLOUR_GAIN, COLOUR_OPACITY, CONTROLS, TONE
        from helio.application.viewing import CONF_BLENDS, CONF_COLOURS
        from helio_qt.state import NEED_SOLVE
        from helio_qt.workspaces import INSPECTOR_W, SIDE_W
        self.blurbs = dict(ENGINE_BLURBS)
        self.blurbs[self.DEPTH_MAP] = "Height as grey: light is high, dark is low.  D"
        self.blurbs[self.BACKLIT] = ("The backlit photograph, lined up with the surface: "
                                     "watermarks, chain lines, repairs, thin places.")
        fams = OrderedDict((f, []) for f in self.FAMILY_ORDER)
        for e in RENDER_ENGINES:
            if e in self.HIDDEN:
                continue
            fams.setdefault(_ENGINE_META.get(e, {}).get("family", "Other"), []).append(e)
        fams["Depth"].insert(0, self.DEPTH_MAP)
        fams["Decomposition"].insert(0, self.BACKLIT)
        self.families = fams
        self.family_of = {e: f for f, es in fams.items() for e in es}
        self.order = [e for es in fams.values() for e in es]
        store = self.win.prefs

        # sidebar: which view, then everything that changes it (VIEW-003).  The
        # list of views is a dropdown that searches, so the light and the
        # view's settings sit right under it in one scrolling column.
        side, sl = panel("Sidebar", SIDE_W)
        sl.setContentsMargins(0, 0, 0, 0)
        lpage = QWidget()
        ll = QVBoxLayout(lpage)
        ll.setContentsMargins(12, 12, 10, 12)
        ll.setSpacing(12)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a view")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Find a view")
        self.search.setToolTip("Type part of a view's name, or what it shows: scratches, height, "
                               "colour…")
        self.search.textChanged.connect(self._filter)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setAccessibleName("Views")
        self.tree.setRootIsDecorated(True)
        self.items, self.tops = {}, {}
        for fam, es in fams.items():
            top = QTreeWidgetItem([f"{fam}   {len(es)}"])
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(top)
            self.tops[fam] = top
            for e in es:
                it = QTreeWidgetItem([view_name(e)])
                it.setData(0, Qt.ItemDataRole.UserRole, e)
                it.setToolTip(0, self.blurbs.get(e, ""))
                top.addChild(it)
                self.items[e] = it
        # The object profile's views (helio.domain.object_profiles), first and open; the same
        # views stay in their families too.  Empty and hidden for General.
        self.suggested_top = QTreeWidgetItem(["For this object"])
        self.suggested_top.setFlags(self.suggested_top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.tree.insertTopLevelItem(0, self.suggested_top)
        self.suggested_top.setHidden(True)
        self.tree.itemClicked.connect(self._picked)
        self.tree.itemActivated.connect(self._picked)
        from helio_qt.controls import ViewPicker
        self.picker = ViewPicker(self.search, self.tree)
        ll.addWidget(label("VIEW", "heading"))
        ll.addWidget(self.picker)
        self.views_note = label(NEED_SOLVE + ". Until then you can look at the frames.",
                                "caption", wrap=True)
        ll.addWidget(self.views_note)
        self.btn_back_view = button("◀ Back to the view", "primary",
                                    "A photograph is shown. Back to the solved view, as you left it (Esc).")
        self.btn_back_view.clicked.connect(self.back_to_view)
        self.btn_back_view.hide()
        ll.addWidget(self.btn_back_view)
        self.states = QListWidget()
        self.states.setAccessibleName("Saved states")
        self.states.setMaximumHeight(110)
        self.states.itemActivated.connect(lambda it: self.load_state(it.text()))
        self.states.itemClicked.connect(lambda it: self.load_state(it.text()))
        sl.addWidget(self.states)
        self.btn_save_state = button("Save this state…", "quiet",
                                     "The view, its light and settings, the layer and the "
                                     "3D settings, under a name. Saved with the session.")
        self.btn_save_state.clicked.connect(lambda: self.reg.trigger("view.save_state"))

        # centre: the image, the dome, the readouts, the filmstrip
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(self._centre_bar())
        self.viewport = Viewport("Import images to see them here")
        self.dome = LightDome(150)
        self.dome.changed.connect(self._dome_moved)
        self.dome.hide()
        self.probe = label("", "probe")
        self.probe.setProperty("hud", "true")
        self.probe.setAccessibleName("Value under the pointer")
        self.viewport.add_overlay(self.probe, "tl")
        self.probe.hide()
        self.badge = label("", "badge")
        self.badge.setProperty("badge", "true")
        self.badge.setToolTip("Display transforms are on: the picture is mirrored, turned or "
                              "inverted for viewing. Display › Reset to undo.")
        self.viewport.add_overlay(self.badge, "tr")
        self.badge.hide()
        from helio_qt.widgets import ScaleBar
        self.scalebar = ScaleBar(self.viewport, lambda: self.project.scale_mm_per_px)
        self.viewport.add_overlay(self.scalebar, "bc")
        self.viewport.probed.connect(self._probed)
        self.viewport.clicked.connect(self._clicked)
        self.viewport.can_select = True
        self.viewport.selected.connect(self._selection_drawn)
        self.viewport.escaped.connect(lambda: self.btn_select.setChecked(False))
        self.viewport.escaped.connect(lambda: self.btn_section.setChecked(False))
        self.viewport.escaped.connect(lambda: self.btn_wb_pick.setChecked(False))
        self.viewport.rectDrawn.connect(self._wb_rect)
        self.viewport.lineDrawn.connect(self._section_drawn)
        self.viewport.doubleClicked.connect(self._double_clicked)
        self.viewport.relight.connect(self._relight_drag)
        self.viewport.wipeMoved.connect(lambda f: self._wipe_row.set_value(f))
        # The image and the 3D relief share the centre: either, or side by side.
        self.split = QSplitter(Qt.Orientation.Horizontal)
        self.split.setChildrenCollapsible(False)
        self.split.addWidget(self.viewport)
        self.relief = None                  # helio_qt.relief3d.ReliefView, made when first shown
        self._relief_of = None
        self.centre_mode = "image"
        # Under the centre: the flood and section cards, whichever of the image and the 3D relief the line
        # was drawn on.  A divider between them (dragged, and remembered), and the cards scroll in their
        # share rather than squeeze the relief: a section with chain lines and a long list of grooves
        # must not leave the relief too small to turn.
        self.vsplit = QSplitter(Qt.Orientation.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        self.split.setMinimumHeight(260)
        self.vsplit.addWidget(self.split)
        self.under = UnderCards(self.vsplit, [self._flood_card(), self._section_card()])
        cl.addWidget(self.vsplit, 1)
        self.strip = FrameList(self.win.thumbs, QSize(56, 72))
        self.strip.setToolTip("The photographs. Click one to see it; double-click to light the "
                              "view from the same direction, to compare the two.")
        self._strip_lights = None
        self.strip.itemClicked.connect(lambda _it: self._show_frame(self.strip.currentRow()))
        self.strip.itemDoubleClicked.connect(lambda it: self.light_from_frame(self.strip.row(it)))
        self.strip.itemSelectionChanged.connect(
            lambda: self.project.select(self.strip.currentRow()))
        cl.addWidget(self.strip)

        # inspector
        insp, il = panel("Inspector", INSPECTOR_W)
        self.insp_stack = QStackedWidget()
        il.addWidget(self.insp_stack, 1)
        fp = QWidget()                                     # page 0: a photograph
        fl = QVBoxLayout(fp)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(8)
        self.frame_title = label("No frame", "section")
        self.frame_info = label("", "secondary", wrap=True)
        fl.addWidget(self.frame_title)
        fl.addWidget(self.frame_info)
        fl.addWidget(rule())
        self.frame_hint = label("Relighting, depth and clay views appear here after processing.",
                                "caption", wrap=True)
        fl.addWidget(self.frame_hint)
        self.btn_back_view2 = button("◀ Back to the view", "primary",
                                     "Back to the solved view, as you left it (Esc).")
        self.btn_back_view2.clicked.connect(self.back_to_view)
        self.btn_back_view2.hide()
        fl.addWidget(self.btn_back_view2)
        fl.addStretch(1)
        self.insp_stack.addWidget(fp)

        page = QWidget()                                   # page 1: a view, in a scroll
        vl = QVBoxLayout(page)
        vl.setContentsMargins(0, 0, 4, 0)
        vl.setSpacing(12)
        self.view_title = label("", "section", wrap=True)          # named by the picker
        self.view_info = button("i", "info")
        self.view_info.setAccessibleName("About this view")
        self.view_family = label("", "chip")
        for wdg in (self.view_title, self.view_info, self.view_family):
            wdg.hide()
        self.view_blurb = label("", "caption", wrap=True)
        ll.addWidget(self.view_blurb)
        self.view_box = QWidget()                 # shown while a view (not a photograph) is up
        vb = QVBoxLayout(self.view_box)
        vb.setContentsMargins(0, 0, 0, 0)
        vb.setSpacing(12)
        ll.addWidget(self.view_box)

        # Light
        self.sec_light = Section("Light", "light", store)
        self.sec_light.add(self.dome)
        self.sec_light.body_lay.setAlignment(self.dome, Qt.AlignmentFlag.AlignHCenter)
        lr = QHBoxLayout()
        lr.setSpacing(6)
        self.az = QDoubleSpinBox()
        self.az.setRange(0, 359.9)
        self.az.setDecimals(0)
        self.az.setWrapping(True)
        self.az.setSuffix("°")
        self.az.setAccessibleName("Light azimuth")
        self.az.setToolTip("Azimuth: the compass direction the light comes from. "
                           "0° = north (top of the image), 90° = east.")
        self.el = QDoubleSpinBox()
        self.el.setRange(5, 85)
        self.el.setDecimals(0)
        self.el.setSuffix("°")
        self.el.setAccessibleName("Light elevation")
        self.el.setToolTip("Elevation above the horizon, 5–85°.")
        for sp in (self.az, self.el):
            sp.setKeyboardTracking(False)
            sp.valueChanged.connect(self._typed_angles)
        self.az.hide()                     # typed angles: the bars below drive these
        self.el.hide()
        self.sec_light.add(self.az)
        self.sec_light.add(self.el)
        from helio.domain.view_controls import Control
        self.az_row = ParamRow(Control("light_az", "Azimuth", 0, 359, 315, 1,
                                       "The compass direction the light comes from: 0° north "
                                       "(the top of the image), 90° east.", unit="°"))
        self.az_row.changed.connect(lambda _k, v, final: self._az_moved(v, final))
        self.sec_light.add(self.az_row)
        self.height_row = ParamRow(Control("light_height", "Elevation", 5, 90, 45, 1,
                                           "Raise or lower the light, keeping its direction: "
                                           "low for raking, high for colour.", unit="°"))
        self.height_row.changed.connect(self._height_moved)
        self.sec_light.add(self.height_row)
        br = QHBoxLayout()
        br.setSpacing(6)
        self.btn_raking = button("Best raking", "", "Light from the captured direction that shows "
                                 "the most surface contrast.")
        self.btn_raking.clicked.connect(lambda: self.reg.trigger("view.best_raking"))
        self.btn_sweep = button("Sweep", "", "Circle the light at its height.  A")
        self.btn_sweep.setCheckable(True)
        self.btn_sweep.setProperty("tool", "true")
        self.btn_sweep.clicked.connect(lambda: self.reg.trigger("view.sweep"))
        br.addWidget(self.btn_raking)
        br.addWidget(self.btn_sweep)
        # Save, beside the sweep: the sweep itself, and this view at full size.
        from PySide6.QtWidgets import QMenu
        self.btn_light_save = button("Save sweep or view", "", "Save the light sweep (GIF or MP4), or this "
                                     "view at full resolution.  G · S")
        self.light_save_menu = QMenu(self.btn_light_save)
        for text, act in (("Light sweep, GIF or MP4…", "view.sweep_gif"),
                          ("Picture of this view, full resolution…", "view.snapshot"),
                          ("The loupe's region at full resolution…", "view.export_loupe"),
                          ("3D light sweep video…", "view.relief_sweep")):
            self.light_save_menu.addAction(text, lambda a=act: self.reg.trigger(a))
        self.btn_light_save.setMenu(self.light_save_menu)
        self.sec_light.add_layout(br)
        br2 = QHBoxLayout()                       # its own row: three did not fit the sidebar's width
        br2.addWidget(self.btn_light_save)
        self.sec_light.add_layout(br2)
        self.sweep_speed = ParamRow(Control("sweep_speed", "Sweep speed", 0.2, 3.0, 1.0, 0.1,
                                            "How fast the sweep turns."))
        self.sec_light.add(self.sweep_speed)
        self.btn_same_light = button("Use this light in every view", "quiet",
                                     "Each view remembers its own light. This gives them all "
                                     "this one.")
        self.btn_same_light.clicked.connect(self.same_light_everywhere)
        self.sec_light.add(self.btn_same_light)
        self.chk_captured = QCheckBox("Show the capture's lights")
        self.chk_captured.setToolTip("Blue dots on the dome: where the photographs were lit from.")
        self.chk_captured.setChecked(True)
        self.chk_captured.toggled.connect(self._show_captured)
        self.sec_light.add(self.chk_captured)
        self.light_caption = label("Option-drag on the image to move the light.", "caption", wrap=True)
        self.sec_light.add(self.light_caption)
        vb.addWidget(self.sec_light)

        # The view's own settings
        self.sec_settings = Section("Settings", "settings", store)
        self.settings_box = QWidget()
        self.settings_lay = QVBoxLayout(self.settings_box)
        self.settings_lay.setContentsMargins(0, 0, 0, 0)
        self.settings_lay.setSpacing(8)
        self.sec_settings.add(self.settings_box)
        self.no_settings = label("This view has no settings of its own.", "caption", wrap=True)
        self.sec_settings.add(self.no_settings)
        self.btn_reset = button("Reset this view's settings", "quiet",
                                "Put this view's settings back as they first were.")
        self.btn_reset.clicked.connect(self.reset_view_settings)
        self.sec_settings.add(self.btn_reset)
        vb.addWidget(self.sec_settings)
        self.rows: dict[str, object] = {}

        # Tone
        self.sec_tone = Section("Tone", "tone", store, open_=False)
        self.tone_rows = {}
        for k in TONE:
            r = row_for(CONTROLS[k])
            r.changed.connect(self._setting_changed)
            self.sec_tone.add(r)
            self.tone_rows[k] = r
        vb.addWidget(self.sec_tone)

        # Image adjustments: the classic Global Image Adjustments, on the screen and its exports
        from helio_qt.controls import ParamRow as _PR
        self.sec_adjust = Section("Image adjustments", "adjust", store, open_=False)
        self.adjust_rows = {}
        for k, t, lo, hi, step, tip in (
                ("global_whites", "Whites", 0.0, 2.0, 0.01, "Above 1 pulls highlights brighter "
                 "(worn stone, stained parchment); below 1 rolls them off."),
                ("global_blacks", "Blacks", 0.0, 1.0, 0.01, "Lifts the black point to show detail "
                 "in deep shadow."),
                ("global_midtone", "Midtones", 0.5, 2.0, 0.01, "Above 1 lifts the midtones, below "
                 "1 darkens them."),
                ("global_clarity", "Clarity", 0.0, 3.0, 0.01, "Mid-scale local contrast: text "
                 "legibility, tool marks."),
                ("global_texture", "Texture", 0.0, 2.0, 0.01, "Fine-scale local contrast: fibres, "
                 "grain."),
                ("global_dehaze", "Haze removal", 0.0, 1.0, 0.01, "Stretches the view to fill the "
                 "range (auto levels).")):
            from helio.domain.view_controls import Control as _C
            r = _PR(_C(k, t, lo, hi, dict(self.win.display.adjustments).get(k, 0.0), step, tip))
            r.set_value(dict(self.win.display.adjustments).get(k, 0.0))
            r.changed.connect(self._adjust_changed)
            self.sec_adjust.add(r)
            self.adjust_rows[k] = r
        b_reset = button("Reset the adjustments", "quiet", "Set every adjustment back to "
                         "neutral.")
        b_reset.clicked.connect(self._adjust_reset)
        self.sec_adjust.add(b_reset)
        vb.addWidget(self.sec_adjust)

        # Layer
        self.sec_layer = Section("Layer", "layer", store)
        self.layer_seg = Segmented(list(LAYER_KEYS), LAYER_TIPS, columns=2)
        self.layer_seg.chosen.connect(self.set_layer)
        self.sec_layer.add(self.layer_seg)
        self.colour_row = ParamRow(COLOUR_OPACITY)
        self.colour_row.changed.connect(self._colour_changed)
        self.sec_layer.add(self.colour_row)
        self.colour_gain_row = ParamRow(COLOUR_GAIN)
        self.colour_gain_row.changed.connect(self._colour_gain_changed)
        self.sec_layer.add(self.colour_gain_row)
        # White balance: the colour only (never the solve, the shading or a measurement).
        self.wb_box = QWidget()
        wbl = QHBoxLayout(self.wb_box)
        wbl.setContentsMargins(0, 0, 0, 0)
        self.btn_wb_pick = QToolButton()
        self.btn_wb_pick.setText("Pick white")
        self.btn_wb_pick.setCheckable(True)
        self.btn_wb_pick.setProperty("tool", "true")
        self.btn_wb_pick.setToolTip("Click a white or grey that should be neutral (the white of a "
                                    "scale bar, a grey card), or drag a box over it. Only the "
                                    "colour changes: the shading, the solve and measurements do not.")
        self.btn_wb_pick.setAccessibleName("Pick a white balance point")
        self.btn_wb_pick.toggled.connect(self._wb_pick_toggled)
        wbl.addWidget(self.btn_wb_pick)
        self.btn_wb_reset = QToolButton()
        self.btn_wb_reset.setText("Reset")
        self.btn_wb_reset.setToolTip("The colour as photographed: no white balance.")
        self.btn_wb_reset.clicked.connect(self.reset_white_balance)
        wbl.addWidget(self.btn_wb_reset)
        wbl.addStretch(1)
        self.sec_layer.add(self.wb_box)
        self.wb_note = label("White balance: off", "caption", wrap=True)
        self.sec_layer.add(self.wb_note)
        self.chk_solved_colour = QCheckBox("Solved reflectance")
        self.chk_solved_colour.setToolTip(
            "A RAW capture's colour with the shading taken out by the solve, in place of the median "
            "of the frames, in the views, 3D textures and exports. Relative: not calibrated to a white.")
        self.chk_solved_colour.setVisible(False)
        self.chk_solved_colour.toggled.connect(self._solved_colour_toggled)
        self.sec_layer.add(self.chk_solved_colour)
        self.rough_row = row_for(CONTROLS["rough_opacity"])
        self.rough_row.changed.connect(self._setting_changed)
        self.sec_layer.add(self.rough_row)
        self.conf_box = QWidget()
        cb = QVBoxLayout(self.conf_box)
        cb.setContentsMargins(0, 0, 0, 0)
        cb.setSpacing(8)
        crow = QHBoxLayout()
        self.conf_colour = QComboBox()
        self.conf_colour.addItems(CONF_COLOURS)
        self.conf_colour.setAccessibleName("Highlight colour")
        self.conf_colour.setToolTip("The colour the confidence highlight is drawn in.")
        self.conf_blend = QComboBox()
        self.conf_blend.addItems(CONF_BLENDS)
        self.conf_blend.setAccessibleName("Blend")
        self.conf_blend.setToolTip("Overlay = balanced; Multiply darkens doubtful areas; "
                                   "Additive highlights them.")
        crow.addWidget(self.conf_colour, 1)
        crow.addWidget(self.conf_blend, 1)
        cb.addLayout(crow)
        self.conf_rows = {
            "opacity": ParamRow(Control("opacity", "Overlay opacity", 0.0, 1.0, 0.6, 0.01,
                                        "How strongly doubtful pixels are marked.")),
            "shadow": ParamRow(Control("shadow", "Shadow cutoff", 0, 100, 20, 1,
                                       "Pixels darker than this (0-255) are marked as shadowed.")),
            "highlight": ParamRow(Control("highlight", "Highlight cutoff", 150, 255, 235, 1,
                                          "Pixels brighter than this (0-255) are marked as blown.")),
        }
        for k, r in self.conf_rows.items():
            r.changed.connect(lambda _k, v, final, kk=k: self._conf_changed(kk, v, final))
            cb.addWidget(r)
        self.conf_mask = QCheckBox("Mask the view by confidence")
        self.conf_mask.setToolTip("Black out pixels outside the cutoffs, in every layer.")
        self.conf_mask.toggled.connect(lambda on: self._conf_changed("mask", bool(on), True))
        cb.addWidget(self.conf_mask)
        self.conf_colour.currentTextChanged.connect(lambda t: self._conf_changed("colour", t, True))
        self.conf_blend.currentTextChanged.connect(lambda t: self._conf_changed("blend", t, True))
        self.sec_layer.add(self.conf_box)
        vb.insertWidget(0, self.sec_layer)       # straight under the view picker, above the light

        # Display
        self.sec_display = Section("Display", "display", store, open_=False)
        self.chk_invert_normals = QCheckBox("Invert normals (bumps look like holes)")
        self.chk_invert_normals.setToolTip("Display only: the solved surface is unchanged.")
        self.chk_flip_h = QCheckBox("Mirror left ↔ right")
        self.chk_flip_h.setToolTip("Correct for most printing plates.")
        self.chk_flip_v = QCheckBox("Mirror top ↕ bottom")
        self.chk_flip_v.setToolTip("For plates viewed from the back.")
        self.chk_rot180 = QCheckBox("Turn 180°")
        self.chk_rot180.setToolTip("Show the picture upside down: for an object photographed "
                                   "the other way up.")
        self.chk_depth_inv = QCheckBox("Invert depth (Z)")
        self.chk_depth_inv.setToolTip("Negate the relief: embossed becomes debossed. Also applies "
                                      "to 3D outputs while on.")
        for chk, attr in ((self.chk_invert_normals, "normal_invert"), (self.chk_flip_h, "flip_h"),
                          (self.chk_flip_v, "flip_v"), (self.chk_rot180, "rot180"),
                          (self.chk_depth_inv, "depth_invert")):
            chk.toggled.connect(lambda on, a=attr: self._display_flag(a, on))
            self.sec_display.add(chk)
        rot = QHBoxLayout()
        rot.setSpacing(6)
        self.btn_rot_l = button("⟲ 90°", "", "Turn the view left (display only).")
        self.btn_rot_0 = button("0°", "outline", "Back to the captured orientation.")
        self.btn_rot_r = button("⟳ 90°", "", "Turn the view right (display only).")
        self.btn_rot_l.clicked.connect(lambda: self.reg.trigger("view.rotate_left"))
        self.btn_rot_r.clicked.connect(lambda: self.reg.trigger("view.rotate_right"))
        self.btn_rot_0.clicked.connect(lambda: self.reg.trigger("view.rotate_reset"))
        for b in (self.btn_rot_l, self.btn_rot_0, self.btn_rot_r):
            rot.addWidget(b)
        self.sec_display.add_layout(rot)
        from helio.domain.view_controls import Control
        self.rot_row = ParamRow(Control(
            "view_turn", "Turn", -180.0, 180.0, 0.0, step=0.1, unit="°",
            tip="Turn the view by any angle, to straighten a line of text or an edge "
                "(display only). ⇧-drag for tenths; ⌘⇧[ and ⌘⇧] turn it 1°; two fingers "
                "on the trackpad turn it too."))
        self.rot_row.changed.connect(lambda _k, v, _f: self.win.set_rotation_angle(v, keep_view=True))
        self.sec_display.add(self.rot_row)
        self.viewport.rotated.connect(lambda a: self.win.set_rotation_angle(a, keep_view=True))
        self.btn_reset_display = button("Reset display transforms", "quiet",
                                        "Undo every turn and mirror of the picture on screen.")
        self.btn_reset_display.clicked.connect(lambda: self.reg.trigger("view.reset_transforms"))
        self.sec_display.add(self.btn_reset_display)
        vb.addWidget(self.sec_display)

        # Compare
        self.sec_compare = Section("Compare", "compare", store, open_=False)
        self.chk_compare = QCheckBox("Split view")
        self.chk_compare.setToolTip("This view on the left, another on the right. Drag the "
                                    "divider on the image.")
        self.chk_compare.toggled.connect(lambda on: self.reg.trigger("view.compare")
                                         if on != self.win.compare_on else None)
        self.sec_compare.add(self.chk_compare)
        self.compare_combo = QComboBox()
        self.compare_combo.setAccessibleName("Right-hand view")
        self.compare_combo.setToolTip("The view shown on the right of the slider.")
        for e in self.order:
            if e not in self.SPECIAL:
                self.compare_combo.addItem(e)
        self.compare_combo.currentTextChanged.connect(self._compare_view)
        self.sec_compare.add(self.compare_combo)
        self.chk_compare_colour = QCheckBox("Show the plain colour photograph on the right")
        self.chk_compare_colour.setToolTip("Compare with the colour photograph instead of a "
                                           "second view.")
        self.chk_compare_colour.toggled.connect(self._compare_colour)
        self.sec_compare.add(self.chk_compare_colour)
        self._wipe_row = ParamRow(Control("wipe", "Divider", 0.05, 0.95, 0.5, 0.01,
                                          "Where the divider is, across the image."))
        self._wipe_row.changed.connect(lambda _k, v, _f: self.viewport.set_wipe(v))
        self.sec_compare.add(self._wipe_row)
        vb.addWidget(self.sec_compare)
        self.sec_states = Section("Saved states", "states", store, open_=False)
        self.sec_states.add(self.states)
        self.sec_states.add(self.btn_save_state)
        ll.addWidget(self.sec_states)
        ll.addStretch(1)
        from helio_qt.controls import VScroll as _VS
        self.side_scroll = _VS(lpage)
        sl.addWidget(self.side_scroll)

        # The quick exports, at the top of the right column (EXP-010…012)
        self.sec_export = Section("Export", "quick_export", store)
        from PySide6.QtWidgets import QGridLayout as _Grid
        wrow = _Grid()
        wrow.setHorizontalSpacing(6)
        for i, (text, sid, tip) in enumerate((
                # Short words: the full names were cut off in the 320 px column.
                ("RTI viewer", "export.rti_combined",
                 "Combined RTI viewer: one web page to relight with the mouse, offline in any "
                 "browser."),
                ("Fine-detail 3D", "export.hf_viewer",
                 "3D viewer, fine detail: a web page of the 3D relief with the fine detail drawn "
                 "from the normals, and a slider for the photograph's colour."))):
            b = button(text, "", tip)
            b.setObjectName("QuickWeb")
            b.clicked.connect(lambda _c=False, i=sid: self.reg.trigger(i))
            wrow.addWidget(b, 0, i)
        b = button("Relight + 3D", "", "Relight and 3D viewer: one web page with two tabs, the "
                   "relightable image and a light 3D model to turn. For a talk or an e-mail.")
        b.setObjectName("QuickWeb")
        b.clicked.connect(lambda _c=False: self.reg.trigger("export.rti_3d"))
        wrow.addWidget(b, 1, 0, 1, 2)                # its own row: a third did not fit across
        wrow.setVerticalSpacing(6)
        self.sec_export.add_layout(wrow)
        qrow = _Grid()
        qrow.setHorizontalSpacing(6)
        qrow.setVerticalSpacing(6)
        for i, (text, sid, tip) in enumerate((
                ("Send to QGIS", "export.qgis",
                 "The heights, colour and normals in QGIS, with the Helio "
                 "Relief plugin.  ⇧⌘G"),
                ("This view", "view.snapshot",
                 "Save View: the image at full resolution as a TIFF, the 3D relief, or "
                 "both as the centre shows them.  S"))):
            b = button(text, "", tip)
            b.clicked.connect(lambda _c=False, i=sid: self.reg.trigger(i))
            qrow.addWidget(b, i // 2, i % 2)
        self.sec_export.add_layout(qrow)
        b_all = button("Every export, by what it is for…  ⌘5", "quiet",
                       "Images, 3D, web, metrology, archives and reports.")
        b_all.clicked.connect(lambda: self.reg.trigger("view.ws:Publish"))
        self.sec_export.add(b_all)
        vl.addWidget(self.sec_export)

        # What has been measured, at a glance, with the way to each tool
        self.sec_measure = Section("Measure", "measure_summary", store)
        self.m_rows = {}
        for key, sid, act in (("scale", "measure.scale", "Set Scale…"),
                              ("height", "measure.zscale", "Set Height Scale…"),
                              ("colour", "measure.colour", "Check a Chart…")):
            row = QHBoxLayout()
            row.setSpacing(6)
            txt = label("", "secondary", wrap=True)
            b = button(act, "quiet", self.reg.specs[sid].text)
            b.clicked.connect(lambda _c=False, i=sid: self.reg.trigger(i))
            row.addWidget(txt, 1)
            row.addWidget(b)
            self.sec_measure.add_layout(row)
            self.m_rows[key] = txt
        b_m = button("Every measuring tool…  ⌘4", "quiet",
                     "The Measure step: every measuring tool on the image")
        b_m.clicked.connect(lambda: self.reg.trigger("view.ws:Measure"))
        self.sec_measure.add(b_m)
        vl.addWidget(self.sec_measure)

        # The 3D relief's own settings, shown while it is on screen (VIEW-004)
        from helio_qt.controls import ProfileChart, Segmented as _Seg
        from helio_qt.relief3d import MATERIAL_TIPS, MATERIALS
        from helio.application.viewer3d import CLAY as V3_CLAY
        # One panel for the 3D: the relief in the window and, when it is open,
        # the separate window too.  The shape and tone settings are the
        # classic 3D tab's own (win.shared), so the exports follow them as
        # they always did.
        sh = self.win.shared
        self.sec_relief = Section("3D relief", "relief", store)
        self.relief_kind = _Seg([("standard", "As captured"), ("flattened", "Flattened"),
                                 ("inverted", "Inverted")],
                                {"standard": "The surface as it was solved.",
                                 "flattened": "The best-fitting curved surface taken out: warp "
                                              "and bow gone, the marks kept.",
                                 "inverted": "Turned over: an intaglio, seal or die reads as "
                                             "relief."})
        self.relief_kind.set_current("standard")
        self.relief_kind.chosen.connect(lambda _k: self._relief_shape_changed())
        self.sec_relief.add(self.relief_kind)
        self.relief_material = _Seg(list(MATERIALS), MATERIAL_TIPS, columns=3)
        self.relief_material.set_current("colour")
        self.relief_material.chosen.connect(lambda k: self.relief and self.relief.set_material(k))
        self.sec_relief.add(self.relief_material)
        self.sec_relief.add(label("Mesh density", "secondary"))   # above Height, as asked
        self.relief_mesh = _Seg([("0.5", "2 : 1"), ("1", "1 : 1"), ("2", "1 : 2"), ("3", "1 : 3"),
                                 ("4", "1 : 4"), ("6", "1 : 6"), ("8", "1 : 8")],
                                {"0.5": "Two vertices to every pixel, between the pixels' heights: the "
                                        "smoothest edges, for a region or a small capture (a large "
                                        "surface uses 1 : 1 and says so)",
                                 "1": "Every pixel a vertex: one vertex for each height measured",
                                 "2": "Every second pixel: the usual balance",
                                 "3": "Every third pixel",
                                 "4": "A light mesh: fast on any machine",
                                 "6": "Lighter still",
                                 "8": "A sketch"}, columns=3)
        self.relief_mesh.setToolTip("How finely the 3D shape is built. The shading comes from the "
                                    "normals at full preview size whatever you choose (Shading by "
                                    "the normals), so the difference shows at the silhouette and "
                                    "in low raking light, or with that box off.")
        self.relief_mesh.set_current("2")
        self.relief_mesh.chosen.connect(self._relief_mesh_step)
        self.sec_relief.add(self.relief_mesh)
        self.mesh_note = label("", "caption", wrap=True)
        self.sec_relief.add(self.mesh_note)
        self.chk_relief_wire = QCheckBox("Show the mesh (wireframe)")
        self.chk_relief_wire.setToolTip("The mesh's edges drawn over the relief, so you can see "
                                        "how dense it is. Zoom in: at 1 : 1 they are finer than "
                                        "the screen's pixels.")
        self.chk_relief_wire.toggled.connect(self._relief_wire)
        self.sec_relief.add(self.chk_relief_wire)
        self.chk_relief_wire_only = QCheckBox("Wire only: hide the surface")
        self.chk_relief_wire_only.setToolTip("Only the mesh's edges, the surface itself hidden (its "
                                             "edges behind still hidden). A lighter mesh (1 : 4, "
                                             "1 : 8) shows the wire most clearly.  X steps through "
                                             "off, over the surface and wire only.")
        self.chk_relief_wire_only.setEnabled(False)
        self.chk_relief_wire_only.toggled.connect(self._relief_wire)
        under = QWidget()                                # set in under the box it depends on
        ul = QHBoxLayout(under)
        ul.setContentsMargins(22, 0, 0, 0)
        ul.addWidget(self.chk_relief_wire_only)
        self.sec_relief.add(under)
        self.relief_rows = {}
        for key, text, lo, hi, val, step, tip in (
                ("exaggeration", "Height", 0.5, 30.0, 0.6, 0.1,
                 "1 is the true shape: the heights scaled so their slopes agree with the "
                 "normals.  Higher makes every slope steeper, in the mesh and the shading.  [ ]"),
                ("ambient", "Ambient", 0.0, 1.0, 0.18, 0.01, "Light on every surface."),
                ("diffuse", "Diffuse", 0.0, 2.0, 0.95, 0.01, "How strongly the surface answers "
                 "the light."),
                ("specular", "Shine", 0.0, 1.0, 0.12, 0.01, "The glossy highlight."),
                ("shine", "Shininess", 1.0, 128.0, 24.0, 1.0, "The highlight's size: 1 broad, "
                 "128 a tight spot."),
                ("contour", "Contour step", 0.005, 0.2, 0.02, 0.005,
                 "Height between contour lines, as a share of the whole height range.  C")):
            r = ParamRow(Control(key, text, lo, hi, val, step, tip))
            r.changed.connect(lambda k, v, _f: self._relief_setting(k, v))
            self.sec_relief.add(r)
            self.relief_rows[key] = r
        crow = QHBoxLayout()
        crow.addWidget(label("Clay", "secondary"))
        self.relief_clay = QComboBox()
        self.relief_clay.addItems(["Warm"] + list(V3_CLAY))
        self.relief_clay.setCurrentText("Grey")          # the default; not a custom choice yet
        self.relief_clay.setAccessibleName("Clay colour")
        self.relief_clay.setToolTip("The colour of the Clay surface (and of the separate "
                                    "window's custom clay).")
        self.relief_clay.currentTextChanged.connect(self._relief_clay)
        crow.addWidget(self.relief_clay, 1)
        self.sec_relief.add_layout(crow)

        self.sec_shape = Section("Shape", "relief_shape", store, open_=False,
                                 tip="How the heights are made from the depth.")
        self.shape_rows = {}
        for key, text, lo, hi, step, tip in (
                ("shape_blend", "Detail vs shape", 0.0, 1.0, 0.01, "0 keeps the whole shape "
                 "(buckle, warp); 1 keeps the fine detail on a flat base."),
                ("hp_sigma", "Warp flatten", 0.0, 1.0, 0.01, "Takes out the bowl integration "
                 "can leave (the taco fix)."),
                ("warp_rem", "Broad warp remove", 0.0, 1.0, 0.01, "Takes out broad curvature, "
                 "bow or tilt."),
                ("micro_relief", "Fine detail boost", 0.0, 3.0, 0.05, "Lifts fine texture: 1-2 "
                 "for shallow coins and engravings."),
                ("mesh_smooth", "Smoothing", 0.0, 5.0, 0.1, "Smooths the heights; 0.5-1.5 "
                 "is usually enough."),
                ("x_tilt", "Tilt left–right", -15.0, 15.0, 0.1, "Takes out a left–right slope: "
                 "a copy stand that is not level. Degrees."),
                ("y_tilt", "Tilt front–back", -15.0, 15.0, 0.1, "Takes out a front–back slope. "
                 "Degrees.")):
            r = ParamRow(Control(key, text, lo, hi, float(sh.get(key, max(lo, 0.0))), step, tip,
                                 unit="°" if key.endswith("_tilt") else ""))
            r.changed.connect(lambda k, v, final: self._relief_shape(k, v, final))
            self.sec_shape.add(r)
            self.shape_rows[key] = r
        b_shape = button("Reset the shape", "quiet", "Back to the defaults: detail vs shape "
                         "0.55, fine detail 0.4, the rest off.")
        b_shape.clicked.connect(self._relief_shape_reset)
        self.sec_shape.add(b_shape)
        self.sec_relief.add(self.sec_shape)

        self.sec_tone3d = Section("Display tone", "relief_tone", store, open_=False,
                                  tip="How the 3D picture is shown, not the surface.")
        self.tone3d_rows = {}
        for key, text, lo, hi, step, tip in (
                ("viewer_gamma", "Gamma", 0.3, 3.0, 0.01, "Above 1 lifts the mid-tones."),
                ("viewer_saturation", "Saturation", 0.0, 2.0, 0.01, "0 grey, 1 natural."),
                ("shadow_lift", "Shadow lift", 0.0, 0.4, 0.01, "Keeps the darkest parts "
                 "from going black."),
                ("highlight_clip", "Highlight clip", 0.5, 1.0, 0.01, "Rolls off the "
                 "brightest highlights.")):
            r = ParamRow(Control(key, text, lo, hi, float(sh.get(key, 1.0 if "clip" in key
                                                                    or key.startswith("viewer")
                                                                    else 0.0)), step, tip))
            r.changed.connect(lambda k, v, _f: self._relief_tone(k, v))
            self.sec_tone3d.add(r)
            self.tone3d_rows[key] = r
        self.sec_relief.add(self.sec_tone3d)

        self.chk_relief_nmap = QCheckBox("Shade with the photometric normals")
        self.chk_relief_nmap.setChecked(True)
        self.chk_relief_nmap.setToolTip("Every pixel lit with its own normal, at the preview's full "
                                        "size, whatever the mesh density.  Off: the mesh's own facets.")
        self.chk_relief_nmap.toggled.connect(lambda on: self._relief_setting("nmap", on))
        self.sec_relief.add(self.chk_relief_nmap)
        self.chk_relief_contours = QCheckBox("Height contours  C")
        self.chk_relief_contours.setToolTip("Lines of equal height on the 3D relief (C in the 3D "
                                            "view).")
        self.chk_relief_contours.toggled.connect(lambda on: self.relief and self.relief.set_contours(on))
        self.sec_relief.add(self.chk_relief_contours)
        self.sec_relief.add(label("Section: S in the 3D view, then drag across the relief. Its "
                                  "profile opens under the 3D view; drag A or B to move it, "
                                  "Shift to keep it straight.", "caption", wrap=True))
        from PySide6.QtWidgets import QGridLayout
        rb = QGridLayout()
        rb.setHorizontalSpacing(6)
        rb.setVerticalSpacing(6)
        b_pic = button("Picture…", "", "Save View: this 3D view as a picture (as on screen, or "
                       "drawn afresh at 4K, 6K or 8K for print), or the image and the relief "
                       "side by side as the centre shows them.  ⌘⌥S")
        b_pic.clicked.connect(lambda: self.win.save_view(
            "centre" if self.centre_mode == "both" else "3d"))
        b_turn = button("Turntable…", "", "Record the relief turning once round, as MP4 or GIF.  T")
        b_turn.clicked.connect(lambda: self.win.record_relief_turntable())
        b_fly = button("Flyover…", "", "One smooth loop round the relief, dipping low where the "
                       "light rakes and moving in on the detail; loops without a jump. MP4 or GIF.")
        b_fly.clicked.connect(lambda: self.win.record_relief_flyover())
        b_light = button("Light sweep…", "", "The camera still, the light circling the relief "
                         "as in RTI, as MP4 or GIF.  G")
        b_light.clicked.connect(lambda: self.win.record_relief_sweep())
        rb.addWidget(b_pic, 0, 0)
        rb.addWidget(b_turn, 0, 1)
        rb.addWidget(b_fly, 1, 0)
        rb.addWidget(b_light, 1, 1)
        self.sec_relief.add_layout(rb)
        self.relief_window_note = label("", "caption", wrap=True)
        self.sec_relief.add(self.relief_window_note)
        b_win = button("Open high-detail 3D window", "quiet",
                       "This 3D view in its own window, built from up to 9 million points (here: "
                       "7 million), for close looking, pictures, turntables and flyovers. These "
                       "controls drive it.")
        b_win.clicked.connect(lambda: self.reg.trigger("view3d.window"))
        self.sec_relief.add(b_win)
        self.sec_relief.hide()
        vl.addWidget(self.sec_relief)

        # Loupe
        # Closed to start (the view is sharp when zoomed in; the loupe is for a pinned 1 : 1 look),
        # once over a remembered open, then as left.
        if store is not None and store.value("sections/loupe_closed_once") is None:
            store.setValue("sections/loupe", False)
            store.setValue("sections/loupe_closed_once", True)
        self.sec_loupe = Section("Loupe", "loupe", store, open_=False)
        self.loupe = Loupe(INSPECTOR_W - 60)
        self.loupe.zoom_changed.connect(lambda _z: self._loupe_request())
        self.chk_detail = QCheckBox("Sharp at full size when zoomed in")
        self.chk_detail.setChecked(True)
        self.chk_detail.setToolTip("Enlarged past the preview's resolution, the part on screen is drawn "
                                   "again from the full-size data and follows the light as far as the "
                                   "machine keeps up.  Off: the preview is only enlarged.")
        self.chk_detail.toggled.connect(self._set_live_detail)
        self.sec_display.add(self.chk_detail)        # the main view's own sharpness: not hidden with the loupe
        self.sec_loupe.add(self.loupe)
        self.btn_loupe_export = button("Save this region at full resolution…", "quiet",
                                       "The pixels in the loupe, written as an image at full "
                                       "resolution.")
        self.btn_loupe_export.clicked.connect(lambda: self.reg.trigger("view.export_loupe"))
        self.sec_loupe.add(self.btn_loupe_export)
        vl.addWidget(self.sec_loupe)

        # Histogram
        self.sec_hist = Section("Histogram", "histogram", store)
        self.hist = Histogram()
        self.sec_hist.add(self.hist)
        self.sec_hist.toggled.connect(self._hist_opened)
        vl.addWidget(self.sec_hist)
        # The 3D model (helio_qt.model3d)
        self.sec_3d = self.win.model3d.section(store)
        vl.addWidget(self.sec_3d)

        # Annotations (helio_qt.annotate)
        from helio_qt.annotate import Annotator
        self.annotator = Annotator(self.win, self.viewport)
        vl.addWidget(self.annotator.section(store))

        vl.addStretch(1)

        from helio_qt.controls import VScroll
        scroll = VScroll(page)
        self.scroll = scroll
        self.insp_stack.addWidget(scroll)

        self.mode = "frame"
        self.last_ms = 0.0
        self.view_ms: dict[str, float] = {}
        self.error = ""
        self._shown_result = None
        self._loupe_at = None
        self._probe_at = None
        self._pending_final = False
        self._shown_gen = 0
        self.drag_scale: dict[str, float] = {}       # per view: the scale a drag frame uses
        self._scaled: dict = {}                      # (id(data), scale) -> ScaledData
        self._req_scale: dict[int, float] = {}       # request -> the scale it was drawn at
        self._warmed: set = set()                    # (data, view, scale) drawn once in the background
        self.live_detail = True
        self._detail_ms: dict[str, float] = {}       # per view: how long its last full-size region took
        self._main_tag_seen = None
        self._detail_timer = QTimer()
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(140)
        self._detail_timer.timeout.connect(self._detail_request)
        self.viewport.viewChanged.connect(self._view_moved)
        self.renderer = Renderer()
        self.renderer.sig.done.connect(self._rendered)
        self.sweep_timer = QTimer()
        self.sweep_timer.setInterval(16)
        self.sweep_timer.timeout.connect(self._sweep_step)
        self.sweep = None
        return side, center, insp

    # ── the centre: the image, the 3D relief, or both (VIEW-004) ─────────
    SURROUNDS = (("theme", "Theme", None), ("black", "Black", "#000000"),
                 ("dark", "Dark grey", "#2B2B2B"), ("mid", "Mid grey (ISO 12646)", "#777777"),
                 ("white", "White", "#FFFFFF"))

    def _centre_bar(self) -> QWidget:
        from helio_qt.controls import Segmented
        bar = QWidget()
        bar.setObjectName("Toolbar")
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(8, 6, 8, 6)
        bl.setSpacing(8)
        self.mode_seg = Segmented([("image", "Image"), ("3d", "3D relief"), ("both", "Side by side")],
                                  {"image": "The view, or a photograph.  ⌥1",
                                   "3d": "The surface in 3D, lit by the same light.  ⌥2",
                                   "both": "The view and the 3D relief together.  ⌥3"})
        self.mode_seg.set_current("image")
        self.mode_seg.chosen.connect(self.set_centre_mode)
        self.mode_seg.setMaximumWidth(300)
        bl.addWidget(self.mode_seg, 1)
        bl.addSpacing(6)
        self.surround_lbl = label("Surround", "caption")
        bl.addWidget(self.surround_lbl)
        self.surround_buttons = {}
        sw = QHBoxLayout()                 # the swatches as one group, close together
        sw.setSpacing(4)
        bl.addLayout(sw)
        for key, name, colour in self.SURROUNDS:
            b = QToolButton()
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setFixedSize(24, 24)
            b.setToolTip(f"Surround: {name}" + ("" if colour else " (the neutral well)"))
            b.setAccessibleName(f"Surround {name}")
            b.setStyleSheet(f"QToolButton {{ background: {colour or 'transparent'}; border: 1px solid "
                            "#5A5D63; border-radius: 4px; } QToolButton:checked { border: 2px solid "
                            "#E6A93E; }")
            b.clicked.connect(lambda _c=False, c=colour: self.set_surround(c))
            sw.addWidget(b)
            self.surround_buttons[key] = b
        self.surround_buttons["theme"].setChecked(True)
        # The selection (the classic's Shift-drag crop): what Publish exports
        # and the Research tools analyse, until cleared.
        bl.addSpacing(8)
        self.btn_select = QToolButton()
        self.btn_select.setText("Select")
        self.btn_select.setCheckable(True)
        self.btn_select.setProperty("tool", "true")      # on: outlined in gold (theme)
        self.btn_select.setToolTip("Drag a box on the picture: Publish exports and the Research "
                                   "tools then work on that part alone. Shift-drag does the same "
                                   "at any time.")
        self.btn_select.setAccessibleName("Select a region")
        self.btn_select.toggled.connect(self._select_toggled)
        bl.addWidget(self.btn_select)
        self.sel_note = label("", "caption")
        bl.addWidget(self.sel_note)
        self.btn_sel_clear = QToolButton()
        self.btn_sel_clear.setText("Clear")
        self.btn_sel_clear.setToolTip("Back to the whole object.")
        self.btn_sel_clear.clicked.connect(lambda: self.win.set_selection(None))
        bl.addWidget(self.btn_sel_clear)
        self.btn_section = QToolButton()
        self.btn_section.setText("Section")
        self.btn_section.setCheckable(True)
        self.btn_section.setProperty("tool", "true")      # on: outlined in gold (theme)
        self.btn_section.setToolTip("Drag a line across the picture: its height profile shows "
                                    "below, and the same line on the 3D relief.")
        self.btn_section.setAccessibleName("Draw a section line")
        self.btn_section.toggled.connect(self._section_toggled)
        bl.addWidget(self.btn_section)
        bl.addStretch(1)
        self.chk_assess = QCheckBox("Colour assessment")
        self.chk_assess.setToolTip(
            "ISO 12646 viewing: a mid-grey surround as the tone reference and a thin white "
            "border as the white reference, the image held well inside.  A dark surround makes "
            "an image look brighter, less colourful and flatter than it is.")
        self.chk_assess.toggled.connect(lambda on: self.viewport.set_assessment(on))
        bl.addWidget(self.chk_assess)
        # Narrow (a small window, or side by side): the words give way, the
        # controls stay.  At 1100 px the switch and "Surround" were drawn
        # over each other.
        from PySide6.QtCore import QObject as _QO

        class _Fit(_QO):
            def eventFilter(fit, obj, ev):
                from PySide6.QtCore import QEvent
                if ev.type() == QEvent.Type.Resize:
                    narrow = obj.width() < 700
                    self.surround_lbl.setVisible(not narrow)
                    self.chk_assess.setText("Assess colour" if narrow else "Colour assessment")
                    for k, full, short in (("image", "Image", "Image"), ("3d", "3D relief", "3D"),
                                           ("both", "Side by side", "Both")):
                        self.mode_seg.buttons[k].setText(short if narrow else full)
                return False
        self._bar_fit = _Fit(bar)
        bar.installEventFilter(self._bar_fit)
        return bar

    # ── the section line on the image (the 3D relief's own, drawn here) ──
    def _section_toggled(self, on: bool):
        if on:
            if not (self.project.solved and self.win.render_data is not None):
                self.btn_section.setChecked(False)
                self.win.flash("A section needs a solved surface: process the capture first.")
                return
            self.btn_select.setChecked(False)
            self.win.flash("Drag a line across the picture. Esc to cancel.", 12)
        self.viewport.set_tool("line" if on else "")
        self._draw_kept_sections()

    def _sel_offset(self):
        """Where the 3D relief's surface starts in the solve's pixels."""
        sel, D = self.win.selection, self.win.render_data
        if sel and self.win.selected_result()[1] is not D:
            return sel[0], sel[1]
        return 0, 0

    def _solve_to_view(self, x, y):
        """A point in the solve's pixels, on the picture shown (and back:
        both mappings are their own inverse, as for the selection)."""
        r, D = self.project.result, self.win.render_data
        if self.mode == "frame":
            ox, oy = (r.crop[0], r.crop[1]) if r.crop else (0, 0)
            return x + ox, y + oy
        from helio.application.viewing import image_point
        return image_point(x, y, D.w, D.h, self.win.display)

    def _view_to_solve(self, x, y):
        r = self.project.result
        if self.mode == "frame":
            ox, oy = (r.crop[0], r.crop[1]) if r.crop else (0, 0)
            return x - ox, y - oy
        return self._solve_to_view(x, y)

    def _section_drawn(self, x1, y1, x2, y2):
        """A line drawn on the picture: the 3D relief's section, made there."""
        self.btn_section.setChecked(False)
        self.viewport.clear_line()
        r, D = self.project.result, self.win.render_data
        if r is None or D is None or math.hypot(x2 - x1, y2 - y1) < 8:
            return
        _r, SD = self._relief_source()
        rv = self._ensure_relief()
        self._push_relief()
        if self.centre_mode == "image":
            self.relief_box.setVisible(False)
        ox, oy = self._sel_offset()
        uv = []
        for x, y in ((x1, y1), (x2, y2)):
            sx, sy = self._view_to_solve(x, y)
            uv.append((min(max((sx - ox) / SD.w, 0.0), 1.0), min(max((sy - oy) / SD.h, 0.0), 1.0)))
        rv.set_section(*uv)

    def _draw_section(self):
        """The section line on the picture, wherever it was drawn."""
        prof = getattr(self, "section_prof", None)
        shown = getattr(self, "section_card", None) is None or self.section_card.chk_share.isChecked()
        self._draw_kept_sections()
        if not prof or not shown or self.project.result is None or self.win.render_data is None:
            self.viewport.clear_section()
            return
        ox, oy = self._sel_offset()
        (ax, ay), (bx, by) = prof["a_px"], prof["b_px"]
        self.viewport.show_section(*self._solve_to_view(ax + ox, ay + oy),
                                   *self._solve_to_view(bx + ox, by + oy))

    def highlight_kept_section(self, i):
        """Draw kept section *i* bold (the one chosen in Publish's list); None: none."""
        self._kept_highlight = i
        self._draw_kept_sections()

    def _draw_kept_sections(self):
        """The sections on the sheet, faint, on the surface they were kept on: only while the
        Section tool is on and Show on the 2D image is ticked, so the picture is otherwise clean."""
        w = self.win
        items = []
        card = getattr(self, "section_card", None)
        shown = self.btn_section.isChecked() and (card is None or card.chk_share.isChecked())
        if shown and self.project.result is not None and w.render_data is not None:
            here = w.selected_result()[1]
            live = getattr(self, "section_prof", None)
            ox, oy = self._sel_offset()
            strong = getattr(self, "_kept_highlight", None)
            for i, (p_, of) in enumerate(getattr(w, "sheet_sections", [])):
                if not (of is here or (of is None and here is w.render_data)):
                    continue
                if live and (tuple(live["a_px"]), tuple(live["b_px"])) == (tuple(p_["a_px"]), tuple(p_["b_px"])) \
                        and i != strong:
                    continue                          # drawn already, as the line in hand
                (ax, ay), (bx, by) = p_["a_px"], p_["b_px"]
                a, b = w.section_letters(p_)
                items.append((*self._solve_to_view(ax + ox, ay + oy), *self._solve_to_view(bx + ox, by + oy),
                              f"{a}–{b}", i == strong))
        self.viewport.show_kept_sections(items)

    def _select_toggled(self, on: bool):
        self.viewport.select_mode = bool(on)
        self.viewport.viewport().setCursor(Qt.CursorShape.CrossCursor if on
                                           else Qt.CursorShape.OpenHandCursor)
        if on:
            self.win.flash("Drag a box over the part to work on. Esc to cancel.", 12)

    def _selection_drawn(self, x1, y1, x2, y2):
        """A box on the picture, into the solve's own pixels."""
        self.btn_select.setChecked(False)
        r, D = self.project.result, self.win.render_data
        if r is None or D is None or abs(x2 - x1) < 16 or abs(y2 - y1) < 16:
            self.win.set_selection(None)              # a click: back to the whole object
            return
        if self.mode == "frame":
            ox, oy = (r.crop[0], r.crop[1]) if r.crop else (0, 0)
            box = (x1 - ox, y1 - oy, x2 - ox, y2 - oy)
        else:
            from helio.application.viewing import image_point
            ax, ay = image_point(x1, y1, D.w, D.h, self.win.display)
            bx, by = image_point(x2, y2, D.w, D.h, self.win.display)
            box = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
        self.win.set_selection(box)

    def _draw_selection(self):
        sel, r, D = self.win.selection, self.project.result, self.win.render_data
        has = bool(sel) and r is not None and D is not None
        self.sel_note.setVisible(has)
        self.btn_sel_clear.setVisible(has)
        self.btn_select.setEnabled(r is not None)
        if not has:
            self.viewport.clear_selection()
            return
        x1, y1, x2, y2 = sel
        self.sel_note.setText(f"Selection {int(x2 - x1)} × {int(y2 - y1)} px")
        if self.mode == "frame":
            ox, oy = (r.crop[0], r.crop[1]) if r.crop else (0, 0)
            self.viewport.show_selection(x1 + ox, y1 + oy, x2 + ox, y2 + oy)
        else:
            from helio.application.viewing import image_point
            ax, ay = image_point(x1, y1, D.w, D.h, self.win.display)
            bx, by = image_point(x2, y2, D.w, D.h, self.win.display)
            self.viewport.show_selection(ax, ay, bx, by)

    def set_surround(self, colour):
        self.viewport.set_surround(colour)
        if self.relief is not None:
            self.relief.surround = colour
            self.relief.update()

    def set_centre_mode(self, mode: str):
        """'image', '3d' or 'both' (⌥1, ⌥2, ⌥3)."""
        if mode not in ("image", "3d", "both"):
            return
        if mode != "image" and not (self.project.solved and self.win.render_data is not None):
            self.win.flash("The 3D relief needs a solved surface: process the capture first.")
            self.mode_seg.set_current(self.centre_mode)
            return
        self.centre_mode = mode
        self.mode_seg.set_current(mode)
        if mode != "image":
            self._ensure_relief()
            self._push_relief()
        self.viewport.setVisible(mode != "3d")
        if self.relief is not None:
            self.relief_box.setVisible(mode != "image")
            if mode == "both":
                w = max(2, self.split.width())
                self.split.setSizes([w // 2, w - w // 2])
            if mode != "image":
                self.relief.setFocus(Qt.FocusReason.OtherFocusReason)
        self.update_relief_panel()

    def centre_picture(self, long_side: int = 3840):
        """The centre as it is now (the image, the 3D relief, or both side by
        side) as one uint8 RGB picture *long_side* pixels on its longer side.
        Each part is drawn afresh at that size: the image from its rendered
        view (the part on screen, at the zoom shown), the relief by OpenGL
        off screen; the gizmo, the key card and the zoom controls are left out."""
        import numpy as np
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QColor, QImage, QPainter
        parts = []
        if self.viewport.isVisible():
            parts.append(("image", self.viewport.viewport()))
        if self.relief is not None and self.relief.isVisible():
            parts.append(("3d", self.relief))
        if not parts:
            raise RuntimeError("nothing is shown in the centre")
        gap = 4 if len(parts) > 1 else 0
        sw = sum(wd.width() for _k, wd in parts) + gap * (len(parts) - 1)
        sh = max(wd.height() for _k, wd in parts)
        k = long_side / max(sw, sh)
        W, H = int(round(sw * k)), int(round(sh * k))
        out = np.zeros((H, W, 3), np.uint8)
        bg = QColor(self.viewport.surround or self.viewport._well)
        out[:] = (bg.red(), bg.green(), bg.blue())
        x = 0
        for kind, wd in parts:
            pw = int(round(wd.width() * k)) if kind != parts[-1][0] else W - x
            if kind == "image":
                img = QImage(pw, H, QImage.Format.Format_RGB888)
                img.fill(bg)
                p = QPainter(img)
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                self.viewport.render(p, QRectF(0, 0, pw, H), self.viewport.viewport().rect())
                p.end()
                bpl = img.bytesPerLine()
                a = np.frombuffer(img.constBits(), np.uint8, count=bpl * H).reshape(H, bpl)
                tile = a[:, :pw * 3].reshape(H, pw, 3)
            else:
                tile = self.relief.picture_at(pw, H)
            out[:, x:x + pw] = tile[:, :min(pw, W - x)]
            x += pw + int(round(gap * k))
        return out

    def _ensure_relief(self):
        if self.relief is not None:
            return self.relief
        from helio_qt.relief3d import ReliefView
        self.relief = ReliefView()
        self.relief.lightMoved.connect(self._relief_light)
        self.relief.sectioned.connect(self._relief_section)
        self.relief.floodChanged.connect(self.flood_card.set_info)
        self.relief.message.connect(self._relief_message)
        self.relief.surround = self.viewport.surround
        # The relief, and under it the section card when a line is drawn.
        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(6)
        bl.addWidget(self.relief, 1)
        self.relief_box = box
        self.split.addWidget(box)
        lx, ly, lz = self._current_light()
        self.relief.set_light_screen(lx, ly, lz)
        return self.relief

    def _current_light(self):
        from helio.domain.view_controls import light_of
        return light_of(self.settings(), self.win.engine)

    def _relief_source(self):
        """(result, render data) the 3D relief shows: the selection's alone
        when there is one, at the preview's full detail."""
        r, D = self.win.selected_result()
        return r, D

    def _view_for_relief(self, a):
        """The 2D picture laid on the relief, cut to the selection."""
        sel, D = self.win.selection, self.win.render_data
        if a is None or not sel or D is None or self.win.selected_result()[1] is D:
            return a
        from helio.application.viewing import image_point
        h, w = a.shape[:2]
        ax, ay = image_point(sel[0], sel[1], D.w, D.h, self.win.display)
        bx, by = image_point(sel[2], sel[3], D.w, D.h, self.win.display)
        kx, ky = w / D.w, h / D.h
        x1, x2 = sorted((int(ax * kx), int(bx * kx)))
        y1, y2 = sorted((int(ay * ky), int(by * ky)))
        return a[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)]

    def _push_relief(self):
        """The solved surface into the 3D view, once per solve (and per
        selection: a selected part is shown alone, in finer detail)."""
        import numpy as np
        _r, D = self._relief_source()
        if self.relief is None or D is None or self._relief_of is D:
            return
        A = self._relief_arrays(D)
        trust = None
        sg = A["sigma"]
        if sg is not None:
            trust = np.clip(np.asarray(sg, np.float32) / 10.0, 0, 1)
        self.relief.set_surface(A["depth"], A["normals"], A["albedo"],
                                roughness=A["rough"], trust=trust,
                                view=self._view_for_relief(getattr(self, "_last_image", None)),
                                scale_mm=self.project.scale_mm_per_px,
                                z_mm_per_unit=self.project.z_mm_per_unit,
                                full_size=(D.w, D.h))
        self._relief_of = D
        self._sharpen_relief(D)
        self._relief_wire()
        self._mesh_note()
        lx, ly, lz = self._current_light()
        self.relief.set_light_screen(lx, ly, lz)
        if hasattr(self, "relief_rows"):
            self.relief_rows["exaggeration"].set_value(self.relief.exaggeration)
            for k in ("ambient", "diffuse", "specular", "shine"):
                self._relief_setting(k, self.relief_rows[k].value())
            for k, row in self.tone3d_rows.items():
                self._relief_tone(k, float(self.win.shared.get(k, 1.0 if k in (
                    "viewer_gamma", "viewer_saturation", "highlight_clip") else 0.0)))
        self._relief_shape_changed()

    def _sharpen_relief(self, D):
        """Shade the mesh with the solve's normals and colour at up to 4,500 px (8,000 with 24 GB free, less on a machine
        short of memory), not at the mesh's own size: the mesh is held to a few million points,
        and the shading is what shows fine detail.  Made off the main thread; the mesh is
        shown at once and sharpens when this lands."""
        from helio.application.viewer3d import detail_textures
        r = self.relief
        if r is None or r.data is None or getattr(D, "normals_full", None) is None:
            return
        mesh_w = r.data["w"]

        def make():
            return detail_textures(D, mesh_w)

        def done(out, err):
            if err or out is None or self._relief_of is not D or self.relief is None or self.relief.data is None:
                return
            self.relief.set_detail(out[0], out[1])
        self.win.run_background(make, done)

    # ── the 3D's shape, tone and clay (shared with the separate window) ──
    def _relief_arrays(self, D, max_points: int | None = None) -> dict:
        """The surface the 3D relief is built from: up to the mesh's limit
        (ReliefView.MAX_MESH_POINTS, 6 million), made from the full-size
        solve, so 1 : 1 has that many points; the 2500 px preview held it to
        4.7 million.  The preview itself when that is as large.  Once per
        solve (and selection)."""
        import cv2
        import numpy as np
        from helio_qt.relief3d import ReliefView
        cap = int(max_points or ReliefView.MAX_MESH_POINTS)
        held = getattr(self, "_rarr", None)
        if held is not None and held[0] is D and held[2] == cap:
            return held[1]
        cache = getattr(self, "_rarr_more", {})       # the high-detail window's size, kept apart
        if cap != ReliefView.MAX_MESH_POINTS and cache.get("D") is D and cap in cache:
            return cache[cap]
        k = min(1.0, (cap / float(max(1, D.w * D.h))) ** 0.5)
        tw, th = max(1, int(D.w * k)), max(1, int(D.h * k))
        full_n = getattr(D, "normals_full", None)
        if abs(tw - D.depth_proxy_p.shape[1]) <= 1 or full_n is None:
            out = dict(depth=D.depth_proxy_p, normals=D.normals_proxy, albedo=D.albedo_proxy,
                       rough=getattr(D, "roughness_proxy", None),
                       sigma=getattr(D, "qc_sigma_proxy", None))
        else:
            def small(a, dtype=None):
                if a is None:
                    return None
                a = np.asarray(a) if dtype is None else np.asarray(a, dtype)
                return cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)
            n = small(full_n, np.float32)
            n /= np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-8)
            out = dict(depth=small(D.depth_vis_p, np.float32), normals=n,
                       albedo=small(D.albedo_rgb_full), rough=small(getattr(D, "roughness_full", None)),
                       sigma=small(getattr(D, "qc_sigma_full", None), np.float32))
        if cap == ReliefView.MAX_MESH_POINTS:
            self._rarr = (D, out, cap)
        else:
            self._rarr_more = {"D": D, cap: out}
        return out

    def _full_depth_proxy(self, D):
        """The full, unprocessed depth at the preview's size (the classic
        viewer's d_full), made once per solve."""
        import cv2
        import numpy as np
        r = self._relief_source()[0]
        full = getattr(r, "depth_normalised_full", None) if r is not None else None
        held = getattr(self, "_dfull", None)
        if held is None or held[0][0] is not D or held[0][1] is not full:
            hp = self._relief_arrays(D)["depth"]
            if full is None:
                arr = hp
            else:
                arr = cv2.resize(np.asarray(full, np.float32), (hp.shape[1], hp.shape[0]),
                                 interpolation=cv2.INTER_AREA)
            self._dfull = ((D, full), arr)
        return self._dfull[1]

    def _shape_params(self) -> dict:
        sh = self.win.shared
        D = self._relief_source()[1]
        w = self._relief_arrays(D)["depth"].shape[1] if D is not None else 1500
        return dict(shape_blend=float(sh.get("shape_blend", 0.55)),
                    hp_sigma=float(sh.get("hp_sigma", 0.0)),
                    mesh_smooth=float(sh.get("mesh_smooth", 0.0)),
                    warp_rem=float(sh.get("warp_rem", 0.0)),
                    micro=float(sh.get("micro_relief", 0.4)),
                    depth_clip_pct=float(sh.get("depth_clip_pct", 0.5)),
                    hybrid=bool(sh.get("hybrid_enabled", 0)),
                    hybrid_mix=float(sh.get("hybrid_mix", 0.12)),
                    kind=self.relief_kind.current() if hasattr(self, "relief_kind") else "standard",
                    x_tilt=float(sh.get("x_tilt", 0.0)), y_tilt=float(sh.get("y_tilt", 0.0)),
                    smooth_scale=max(1.0, w / 1500.0))

    def _relief_shape(self, key, v, final=True):
        self.win.model3d.set(key, float(v))      # the shared setting: exports and the separate window
        self._relief_shape_changed()

    def _relief_shape_changed(self):
        """The relief's heights again, on their own worker: a slider stays
        smooth, and the newest setting is always the one that lands."""
        hd = getattr(self.win, "hd3d", None)
        if hd is not None and hd.isVisible():
            hd.reshape()                         # the high-detail window, at its own size
        D, r = self._relief_source()[1], self.relief
        if D is None or r is None or r.data is None:
            return
        from helio.application.viewer3d import relief_surface
        d_full, d_hp, params = (self._full_depth_proxy(D), self._relief_arrays(D)["depth"],
                                self._shape_params())
        if not hasattr(self, "shape_renderer"):
            self.shape_renderer = Renderer()
            self.shape_renderer.sig.done.connect(self._shape_done)
            self._shape_shown = 0
        self.shape_renderer.request("shape", lambda: relief_surface(d_full, d_hp, **params),
                                    tag=id(D))

    def _shape_done(self, channel, gen, out, ms, err):
        if err or out is None or self.relief is None or gen <= self._shape_shown:
            return
        if not self.shape_renderer.latest(channel, gen) and \
                not self.shape_renderer.same_picture(channel, gen):
            return
        self._shape_shown = gen
        self.relief.set_heights(out)

    def _relief_shape_reset(self):
        from helio.config import SHARED_STATE_DEFAULTS
        for key, row in self.shape_rows.items():
            v = float(SHARED_STATE_DEFAULTS[key][1])
            row.set_value(v)
            self.win.model3d.set(key, v)
        self._relief_shape_changed()

    _TONE_NAMES = {"viewer_gamma": "gamma", "viewer_saturation": "sat", "shadow_lift": "lift",
                   "highlight_clip": "clip"}

    def _relief_tone(self, key, v):
        self.win.model3d.set(key, float(v))
        r = self.relief
        if r is None:
            return
        r.tone[self._TONE_NAMES[key]] = float(v)
        r.update()

    def _relief_clay(self, name):
        from helio.application.viewer3d import CLAY
        m = self.win.model3d
        m.clay = name if name in CLAY else "Grey"
        if name in CLAY:
            self.win.shared["use_custom_clay"] = 1
            if m.state is not None:
                from helio.application import viewer3d as V3
                V3.set_clay(m.state, name, True)
        r = self.relief
        if r is not None:
            r.clay = CLAY.get(name, (0.80, 0.76, 0.68))
            r.update()

    def _relief_mesh_step(self, k):
        if self.relief is not None:
            self.relief.set_mesh_step(float(k))
        self.win.model3d.set("step", max(1, int(float(k))))      # the exports and the separate window: whole pixels
        self._mesh_note()

    def _mesh_note(self):
        """The mesh's size now, under the density choice.  The shading is made from the solve's
        own normals at up to detail_side() px, finer than the mesh, so density shows at the edges and
        in low light."""
        r = self.relief
        if r is None or r.data is None:
            from helio.application.viewer3d import detail_side
            self.mesh_note.setText("The shading is made from the solve's own normals at up to "
                                   f"{detail_side():,} px, so detail stays sharp at any density. Density "
                                   "changes the shape (edges, low light).")
            return
        pts, tris = r.mesh_counts()
        self.mesh_note.setText(f"{r.ratio(r.mesh_used())} · {pts / 1e6:.1f} M points, "
                               f"{tris / 1e6:.1f} M triangles, from {r.data['w']:,} × "
                               f"{r.data['h']:,} (held to {r.MAX_MESH_POINTS // 1_000_000} million "
                               f"points). Tick Show the mesh to see it.")

    def _height_per_exaggeration(self):
        """The classic window's height for exaggeration 1: the true-shape
        scale in its pixels (it may be given a smaller copy of the arrays)."""
        D, r = self._relief_source()[1], self.relief
        if D is None:
            return None
        if r is not None and r.data is not None:
            s_px = float(r.data["s_px"]) * float(r.data["depth_range"])
        else:
            from helio_qt.relief3d import true_shape_scale
            A = self._relief_arrays(D)
            s_px = true_shape_scale(A["depth"], A["normals"])
        m = self.win.model3d
        img_w = m.state["img_w"].value if m.state is not None and "img_w" in m.state else D.w
        # s_px is in the pixels of the surface the relief was built from
        built_w = (r.data["w"] if r is not None and r.data is not None
                   else self._relief_arrays(D)["depth"].shape[1])
        return s_px * (float(img_w or D.w) / max(1, built_w))

    def sync_to_window(self):
        """What this panel shows, into the shared 3D values the separate
        window opens with: its height, material, clay and surface.  Before,
        only a setting moved since opening reached it, so it opened at its own
        defaults (a warm clay, its own height)."""
        if not hasattr(self, "relief_rows"):
            return
        from helio.application.viewer3d import CLAY
        m = self.win.model3d
        for k in ("exaggeration", "ambient", "diffuse", "specular", "shine"):
            self._relief_setting(k, self.relief_rows[k].value())
        name = self.relief_clay.currentText()
        m.clay = name if name in CLAY else "Grey"
        self.win.shared["use_custom_clay"] = 1
        mat = self.relief_material.current() if hasattr(self, "relief_material") else "colour"
        self.win.shared["clay_blend"] = 1.0 if mat == "clay" else 0.0

    def _classic_height(self, e: float):
        """The classic window's height for exaggeration *e*."""
        k = self._height_per_exaggeration()
        return None if k is None else max(10.0, min(5000.0, e * k))

    # The separate window's settings edit the same shared values as this panel.
    _SHAPE_KEYS = ("shape_blend", "hp_sigma", "warp_rem", "micro_relief", "mesh_smooth",
                   "depth_clip_pct", "hybrid_enabled", "hybrid_mix", "x_tilt", "y_tilt")
    _MATERIAL_ROWS = {"ambient_str": "ambient", "diffuse_str": "diffuse",
                      "specular_str": "specular", "shininess": "shine"}

    def shared_3d_changed(self, key: str):
        """*key* in ``win.shared`` was changed in the separate window's
        settings: show it on this panel and in the relief here, so the two
        never disagree.  Nothing is sent back."""
        v = self.win.shared.get(key)
        if v is None:
            return
        r = self.relief
        if key in self._SHAPE_KEYS:
            if key in self.shape_rows:
                self.shape_rows[key].set_value(v)
            self._relief_shape_changed()
        elif key in self.tone3d_rows:
            self.tone3d_rows[key].set_value(v)
            if r is not None:
                r.tone[self._TONE_NAMES[key]] = float(v)
                r.update()
        elif key in self._MATERIAL_ROWS:
            k = self._MATERIAL_ROWS[key]
            self.relief_rows[k].set_value(v)
            if r is not None:
                setattr(r, k, float(v))
                r.update()
        elif key == "height":
            per = self._height_per_exaggeration()
            if per:
                row = self.relief_rows["exaggeration"]
                row.set_value(float(v) / per)
                if r is not None:
                    r.set_exaggeration(row.value())

    def update_relief_panel(self):
        """The 3D panel shows while either 3D view is: here, or the separate window."""
        if not hasattr(self, "sec_relief"):
            return
        hd = getattr(self.win, "hd3d", None)
        alive = self.win.model3d.alive() or (hd is not None and hd.isVisible())
        self.sec_relief.setVisible(self.centre_mode != "image" or alive)
        self.relief_window_note.setText("The 3D window is open: these controls drive it "
                                        "too." if alive else "")
        self.relief_window_note.setVisible(alive)

    def _relief_light(self, lx, ly, lz, dragging):
        """L-drag in 3D moves the one light: the dome and the 2D view follow."""
        from helio.domain.view_controls import light_kind
        if light_kind(self.win.engine) is None:
            self.dome.set_light(lx, ly, lz)
            return
        self.write_light(lx, ly, lz)
        self.dome.set_light(lx, ly, lz)
        self._show_angles(lx, ly, lz)
        if self.centre_mode == "both" and self._light_matters():
            self.schedule_render(final=not dragging)

    def _relief_setting(self, key, v):
        m = self.win.model3d                         # the separate window, when it is open
        classic = {"ambient": "ambient_str", "diffuse": "diffuse_str", "specular": "specular_str",
                   "shine": "shininess"}
        if key in classic:
            m.set(classic[key], float(v))
        elif key == "exaggeration":
            h = self._classic_height(float(v))
            if h is not None:
                m.set("height", h)
        r = self.relief
        if r is None:
            return
        if key == "exaggeration":
            r.set_exaggeration(float(v))
        elif key == "shine":
            r.shine = float(v)
            r.update()
        elif key == "contour":
            r.contour_step = float(v)
            if not r.contours:
                r.set_contours(True)
                self.chk_relief_contours.setChecked(True)
            r.update()
        elif key == "nmap":
            self.mesh_note.setVisible(bool(v))
            r.normal_mapped = bool(v)
            r._readout()
            r.update()
        elif key in ("ambient", "diffuse", "specular"):
            setattr(r, key, float(v))
            r.update()

    # ── the section card, under the 3D relief ─────────────────────────
    def _flood_card(self):
        """Under the 3D relief while the Flood tool is on: its level and what lies below it."""
        from helio_qt.flood_card import FloodCard
        card = FloodCard()
        card.attach(lambda: self.relief, self.win, lambda: self.project.name)
        self.flood_card = card
        return card

    def _section_card(self):
        from helio_qt.section_card import SectionCard
        card = SectionCard(share="Show on the 2D image")
        card.chk_share.setChecked(True)
        card.chk_share.setToolTip("The section line on the 2D picture too, with A and B and "
                                  "the chart's pointer.")
        card.share_toggled.connect(lambda _on: self._draw_section())
        card.hovered.connect(self._section_hover)
        card.clear_requested.connect(lambda: self.relief and self.relief.clear_section())
        card.save_requested.connect(self.save_section)
        card.sheet_requested.connect(lambda: self.win.add_sheet_section("relief"))
        self.section_card = card
        card.attach(lambda: self.relief, self.win, lambda: self.project.name)
        # chain lines along the section: off until ticked on the card
        card.enable_lines(self._has_backlit())
        card.lines_requested.connect(self.count_section_lines)
        card.lines_changed.connect(lambda places: self.relief and self.relief.set_line_marks(places))
        # the card's parts, by the names the rest of Inspect (and its tests) use
        self.section_summary, self.section_features = card.summary, card.features
        self.relief_profile, self.btn_section_big = card.chart, card.btn_big
        return card

    def name_section(self):
        """The section's letters (A–B, C–D …, as it will be on the sheet) on the card, the chart, the 3D line
        and the picture's line."""
        if not hasattr(self, "section_card"):
            return
        a, b = self.win.section_letters(getattr(self, "section_prof", None))
        self._draw_kept_sections()
        self.section_card.set_letters(a, b)
        if self.relief is not None:
            self.relief.section_letters = (a, b)
            self.relief.update()
        self.viewport.set_section_letters((a, b))

    def _section_hover(self, t):
        """The pointer on the chart: shown on the line in 3D and on the picture."""
        if self.relief is not None:
            self.relief.set_section_cursor(t)
        self.viewport.set_section_dot(t)

    def _relief_section(self, prof: dict):
        """A section drawn, moved or cleared (in the 3D view, or on the picture)."""
        if not hasattr(self, "section_card"):
            return
        self.section_card.set_profile(prof)
        self.section_prof = prof or None
        self.name_section()
        self._draw_section()
        if prof and not getattr(self, "_section_hover_wired", False):
            self.relief.sectionHover.connect(self.relief_profile.set_cursor)
            self.relief.sectionHover.connect(self.viewport.set_section_dot)
            self._section_hover_wired = True

    # ── chain lines along the section (helio.application.chain_lines) ──
    def _has_backlit(self) -> bool:
        b = getattr(self.win, "backlit", None)
        return bool(b) and b.get("H") is not None

    def _backlit_for(self, r):
        """The backlit frame lined up with the solve, float 0-1, cut to *r* (the selection's result when
        there is one); None without one."""
        whole = self.project.result
        if not self._has_backlit() or whole is None:
            return None
        import numpy as np
        from helio.application.capture_tools import warp_backlit
        b = self.win.backlit
        key = (id(b["raw"]), id(b["H"]), whole.h, whole.w)
        if b.get("float_at") != key:     # full precision, not the uint8 view's; one grey band (a third of RGB's memory)
            from helio.application.chain_lines import grey
            b["float"] = grey(warp_backlit(b["raw"], b["H"], b["ref_shape"], (whole.h, whole.w))).astype(np.float32)
            b["float_at"] = key
        a = b["float"]
        sel = self.win.selection
        if r is whole or not sel:
            return a
        x1, y1, x2, y2 = (int(round(v)) for v in sel)     # as helio.application.pipeline.crop_result cuts
        x1, x2 = max(0, min(x1, x2)), min(whole.w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(whole.h, max(y1, y2))
        return a[y1:y2, x1:x2]

    def _view_for_count(self, r):
        """The picture Inspect shows, unmirrored and unturned (in the solve's orientation) and cut to *r* (the
        selection's result when there is one), at whatever size it was drawn; None before one is drawn."""
        a = getattr(self, "_last_image", None)
        whole = self.project.result
        if a is None or whole is None:
            return None
        from helio.application.viewing import apply_transforms
        a = apply_transforms(a, self.win.display)       # flips and the half turn undo themselves
        sel = self.win.selection
        if r is whole or not sel:
            return a
        kx, ky = a.shape[1] / float(whole.w), a.shape[0] / float(whole.h)
        x1, y1, x2, y2 = (int(round(v)) for v in sel)
        x1, x2 = max(0, min(x1, x2)), min(whole.w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(whole.h, max(y1, y2))
        return a[int(y1 * ky):max(int(y2 * ky), int(y1 * ky) + 1), int(x1 * kx):max(int(x2 * kx), int(x1 * kx) + 1)]

    def count_section_lines(self):
        """Count the chain lines (laid lines, ruling) the section crosses, as the card asks; mark them."""
        card = getattr(self, "section_card", None)
        prof = getattr(self, "section_prof", None)
        opts = card.lines_options() if card is not None else None
        if not opts or not prof or self.relief is None:
            return None
        from helio.application.chain_lines import SOURCES, TRANSMISSIVE_NOTE, along, count_along, pictures
        card.set_backlit(self._has_backlit())
        opts = card.lines_options()
        r, _D = self._relief_source()
        if r is None:
            return None
        source = opts["source"]
        card.set_view_name(self.view_name(self.win.view) if self.win.view else "")
        img, ref = pictures(source, r.albedo_grey_full, r.depth_poisson,
                            self._backlit_for(r) if source == "backlit" else None,
                            self._view_for_count(r) if source == "view" else None)
        if img is None:
            card.set_line_count(None, note=("Show a view in Inspect first." if source == "view"
                                            else TRANSMISSIVE_NOTE))
            return None
        kx, ky = img.shape[1] / float(r.w), img.shape[0] / float(r.h)
        a = (prof["a_px"][0] * kx, prof["a_px"][1] * ky)
        b = (prof["b_px"][0] * kx, prof["b_px"][1] * ky)
        mm = self.relief.scale_mm
        try:
            found = count_along(img, a, b, kind=opts["kind"], swath_mm=opts["swath_mm"],
                                mm_per_px=(mm / kx) if mm else None, reference=ref)
        except Exception as exc:                          # said on the card, not a stale count left showing
            import logging
            logging.getLogger(__name__).exception("counting lines along the section failed")
            card.set_line_count(None, note=f"Could not count along this section: {exc}")
            return None
        note = (f"On this view, {self.view_name(self.win.view)}, as shown." if source == "view"
                else f"On the {SOURCES[source][0].lower() + SOURCES[source][1:]}.")
        if source != "backlit":
            note += " A backlit (transmitted-light) frame shows chain lines better: Capture › Add Backlit Image…"
        if not mm:
            note += " No scale: spacings in pixels, the range searched a guess (Measure › Set Scale… for mm)."
        card.set_line_count(found, along(found, a, b), note)
        return found

    def save_section(self):
        """The section: profile CSV, features CSV, chart SVG and PNG."""
        self.section_card.save(self.win, self.project.name)

    def _relief_message(self, what: str):
        if what == "turntable":
            self.win.record_relief_turntable()
        elif what == "wire" and self.relief is not None:
            mode = self.relief.wire_mode                   # X in the view: the boxes follow
            for box, on in ((self.chk_relief_wire, mode != "off"),
                            (self.chk_relief_wire_only, mode == "only")):
                box.blockSignals(True)
                box.setChecked(on)
                box.blockSignals(False)
            self.chk_relief_wire_only.setEnabled(mode != "off")

    def _relief_wire(self, _on=None):
        """The mesh boxes: its edges over the surface, or on their own."""
        on = self.chk_relief_wire.isChecked()
        self.chk_relief_wire_only.setEnabled(on)
        if self.relief is not None:
            self.relief.set_wireframe(on, only=self.chk_relief_wire_only.isChecked())

    # ── colour ────────────────────────────────────────────────────────
    def recolour(self, t):
        self.viewport.recolour(t)
        self.dome.recolour(t)
        self.hist.recolour(t)
        self.scalebar.recolour(t)
        self.strip.well = t.image_well

    # ── which view ────────────────────────────────────────────────────
    def set_suggested(self, views):
        """Put the object profile's *views* first, under "For this object"."""
        from helio.domain.registry import view_name
        top = self.suggested_top
        top.takeChildren()
        known = [v for v in views if v in self.items]
        for e in known:
            it = QTreeWidgetItem([view_name(e)])
            it.setData(0, Qt.ItemDataRole.UserRole, e)
            it.setToolTip(0, self.blurbs.get(e, ""))
            top.addChild(it)
        top.setText(0, f"For this object   {len(known)}")
        top.setHidden(not known)
        top.setExpanded(bool(known))

    def _filter(self, text: str):
        q = text.strip().lower()
        top = self.suggested_top
        shown = 0
        for i in range(top.childCount()):
            it = top.child(i)
            ok = not q or q in it.text(0).lower()
            it.setHidden(not ok)
            shown += ok
        top.setHidden(shown == 0)
        for fam, top in self.tops.items():
            shown = 0
            for i in range(top.childCount()):
                it = top.child(i)
                key = it.data(0, Qt.ItemDataRole.UserRole) or it.text(0)
                ok = (not q or q in it.text(0).lower() or q in key.lower()
                      or q in self.blurbs.get(key, "").lower())
                it.setHidden(not ok)
                shown += ok
            top.setHidden(shown == 0)
            if q:
                top.setExpanded(shown > 0)

    def _picked(self, item, _col=0):
        name = item.data(0, Qt.ItemDataRole.UserRole) or item.text(0)
        if name in self.items and self.project.solved:
            self.win.set_view(name)

    def _show_frame(self, i):
        self.mode = "frame"
        self.project.select(i)
        self.refresh()
        self.win.update_bottom()

    def back_to_view(self):
        """From a photograph back to the view that was up, its light and settings as they were."""
        if self.mode == "view" or not (self.project.solved and self.win.render_data is not None):
            return
        self.win.set_view(self.win.view or "Standard PBR")

    def _frame_light(self, i):
        """Frame *i*'s light in the screen frame (+Y down), or None."""
        import numpy as np
        L = self.project.lights
        if L is None or i < 0 or i >= len(L):
            return None
        v = np.asarray(L[i], np.float64)
        n = float(np.linalg.norm(v))
        if not np.isfinite(v).all() or n < 1e-9:
            return None
        v = v / n
        return float(v[0]), float(-v[1]), float(v[2])

    def _frame_lights(self):
        """(lights, tooltips) for the strip: the compass on each tile and its words."""
        from helio.domain.view_controls import angles
        lights, tips = [], []
        for i, path in enumerate(self.project.images):
            lv = self._frame_light(i)
            lights.append(lv)
            words = f"Frame {i + 1}: {os.path.basename(path)}"
            if lv is not None:
                az, el = angles(*lv)
                words += f"\nLit from {az:.0f}° (from north), {el:.0f}° up"
            words += "\nClick to see this photograph"
            if lv is not None:
                words += "; double-click to light the view the same way"
            tips.append(words)
        return lights, tips

    def light_from_frame(self, i):
        """Light the view as frame *i* was lit: the photograph and the
        relit surface can then be compared under the same light."""
        lv = self._frame_light(i)
        if lv is None or not (self.project.solved and self.win.render_data is not None):
            self._show_frame(i)
            return
        from helio.domain.view_controls import angles
        from helio.domain.view_controls import light_kind
        self.mode = "view"
        if light_kind(self.win.engine) is None:
            self.win.set_view("Standard PBR")
        lx, ly, lz = lv
        self.write_light(lx, ly, lz)
        self.dome.set_light(lx, ly, lz)
        self._show_angles(lx, ly, lz)
        self.refresh()
        self.render_view(final=True)
        az, el = angles(lx, ly, lz)
        self.win.flash(f"Lit as frame {i + 1}: from {az:.0f}°, {el:.0f}° up. Click the frame "
                       "to see its photograph.")

    def engine(self) -> str:
        return self.win.engine

    def settings(self) -> dict:
        return self.win.engine_settings.setdefault(self.win.engine, {})

    def set_view(self, name):
        """Choose a view: its light and settings come back with it (on_engine_change)."""
        self.mode = "view"
        self._light_for_view(name)
        it = self.items.get(name)
        if it is not None:
            self.tree.blockSignals(True)
            self.tree.setCurrentItem(it)
            it.parent().setExpanded(True)
            self.tree.blockSignals(False)
        self._sync_controls()
        self.render_view(final=True)
        self.refresh()

    # Views that show nothing with the light straight overhead: their extra
    # shine grows with the light's distance from overhead (u² + v²).
    LOW_LIGHT_VIEWS = ("Specular Enhancement",)

    def _light_for_view(self, name):
        """Specular Enhancement adds nothing with the light straight above, the
        default: chosen like that, the light is lowered to 45° (its direction
        kept), and it says so."""
        from helio.domain.view_controls import at_elevation, light_of
        if name not in self.LOW_LIGHT_VIEWS:
            return
        lx, ly, lz = light_of(self.win.engine_settings.get(name, {}), name)
        if lz < 0.97:                                   # already below about 76°
            return
        nx, ny, nz = at_elevation(lx, ly, 45.0)
        self.write_light(nx, ny, nz, engines=[name])
        self.win.flash("Specular Enhancement shows nothing with the light straight above: "
                       "lowered to 45°. Lower still brings out more shine.", 10)

    # ── the controls follow the view ──────────────────────────────────
    def _sync_controls(self):
        from helio.domain.view_controls import (controls_for, light_kind, light_of, takes_tone,
                                                value)
        win, view = self.win, self.win.view
        depth = view in self.SPECIAL
        eng = win.engine
        st = self.settings()
        self.view_title.setText(self.view_name(view) if view else "")
        self.view_family.setText(self.family_of.get(view, "").upper())
        self.picker.set_current(self.view_name(view) if view else "", self.family_of.get(view, ""))
        blurb = self.blurbs.get(view, "")
        self.view_blurb.setText(blurb)
        self.view_info.setToolTip(blurb)
        kind = None if depth else light_kind(eng)
        self.sec_light.setVisible(kind is not None)
        self.dome.setVisible(kind is not None and self.mode == "view")
        if kind:
            lx, ly, lz = light_of(st, eng)
            self.dome.set_light(lx, ly, lz)
            self._show_angles(lx, ly, lz)
        # the view's own settings
        while self.settings_lay.count():
            w = self.settings_lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self.rows = {}
        ctrls = [] if depth else controls_for(eng, st)
        for c in ctrls:
            r = row_for(c)
            r.set_value(value(st, c.key))
            r.changed.connect(self._setting_changed)
            self.settings_lay.addWidget(r)
            self.rows[c.key] = r
        self.no_settings.setVisible(not ctrls)
        self.btn_reset.setVisible(bool(ctrls) or (not depth and takes_tone(eng, st)))
        tone = (not depth) and takes_tone(eng, st)
        self.sec_tone.setVisible(tone)
        for k, r in self.tone_rows.items():
            r.set_value(value(st, k))
        # layer
        d = win.display
        a = self.reg.actions.get(f"view.layer:{d.layer}")
        if a is not None:
            a.setChecked(True)
        self.sec_layer.setVisible(not depth and tone)
        self.layer_seg.set_current(d.layer)
        self.colour_row.set_value(d.colour_opacity)
        # "Colour strength" fades the colour to grey: it read as a broken
        # brightness control, so the panel offers brightness alone.
        self.colour_row.setVisible(False)
        self.colour_gain_row.set_value(d.colour_gain)
        self.colour_gain_row.setVisible(d.layer == "Combined Colour")
        self.wb_box.setVisible(d.layer == "Combined Colour")
        self.wb_note.setVisible(d.layer == "Combined Colour")
        self._sync_white_balance()
        self._sync_solved_colour()
        self.rough_row.set_value(value(st, "rough_opacity"))
        self.rough_row.setVisible(d.layer == "Roughness Fusion")
        self.conf_box.setVisible(d.layer == "Confidence Overlay")
        for k, r in self.conf_rows.items():
            r.set_value(d.conf.get(k, r.c.default))
        for combo, k in ((self.conf_colour, "colour"), (self.conf_blend, "blend")):
            combo.blockSignals(True)
            combo.setCurrentText(str(d.conf.get(k)))
            combo.blockSignals(False)
        self.conf_mask.blockSignals(True)
        self.conf_mask.setChecked(bool(d.conf.get("mask")))
        self.conf_mask.blockSignals(False)
        for chk, attr in ((self.chk_invert_normals, "normal_invert"), (self.chk_flip_h, "flip_h"),
                          (self.chk_flip_v, "flip_v"), (self.chk_rot180, "rot180"),
                          (self.chk_depth_inv, "depth_invert")):
            chk.blockSignals(True)
            chk.setChecked(bool(getattr(d, attr)))
            chk.blockSignals(False)
        self.chk_compare.blockSignals(True)
        self.chk_compare.setChecked(win.compare_on)
        self.chk_compare.blockSignals(False)
        self.compare_combo.blockSignals(True)
        self.compare_combo.setCurrentText(win.compare_view)
        self.compare_combo.blockSignals(False)
        self._update_badge()
        self.dome.set_captured(getattr(win.render_data, "current_L_mat", None)
                               if win.render_data is not None else None)

    def _show_angles(self, lx, ly, lz):
        from helio.domain.view_controls import angles
        az, el = angles(lx, ly, lz)
        for sp, v in ((self.az, az), (self.el, max(5.0, min(85.0, el)))):
            sp.blockSignals(True)
            sp.setValue(round(v))
            sp.blockSignals(False)
        self.height_row.set_value(round(max(5.0, min(90.0, el))))
        self.az_row.set_value(round(az) % 360)

    def _az_moved(self, v, final):
        """The azimuth bar: turn the light round, keeping its height."""
        from helio.domain.view_controls import from_angles, light_of
        from helio.domain.view_controls import angles as _angles
        lx, ly, lz = light_of(self.settings(), self.win.engine)
        _az, el = _angles(lx, ly, lz)
        lx, ly, lz = from_angles(float(v) % 360, max(5.0, min(89.0, el)))
        self._resweep(lx, ly, lz)
        self.write_light(lx, ly, lz)
        self.dome.set_light(lx, ly, lz)
        for sp, val in ((self.az, float(v) % 360),):
            sp.blockSignals(True)
            sp.setValue(round(val))
            sp.blockSignals(False)
        if self._light_matters():
            self.schedule_render(final=final)

    def _rotation_shown(self, deg):
        if hasattr(self, "rot_row") and abs(self.rot_row.value() - float(deg)) > 1e-6:
            self.rot_row.set_value(float(deg))

    def _update_badge(self):
        text = self.win.display.transform_badge()
        rot = self.win.canvas_rotation
        if rot:
            text = (text + " · " if text else "") + f"turned {rot:g}°"
        self.badge.setText(f"  {text}  ")
        self.badge.setVisible(bool(text) and self.mode == "view")
        self.badge.adjustSize()
        self.viewport._place_overlays()

    # ── writing settings ──────────────────────────────────────────────
    def _setting_changed(self, key, value, final):
        st = self.settings()
        c = None
        from helio.domain.view_controls import CONTROLS
        c = CONTROLS.get(key)
        st[key] = int(round(value)) if (c is not None and (c.integer or c.choices)) else float(value)
        if key == "nr_mode":
            self._sync_controls()                    # its gain and tone come and go
        self.schedule_render(final=final)

    def reset_view_settings(self):
        """Every setting of this view back to its default (the light is kept)."""
        from helio.domain.registry import default_engine_settings
        eng = self.win.engine
        st = self.settings()
        keep = {k: st[k] for k in ("lx", "ly", "lz", "hs_azi", "hs_alt") if k in st}
        fresh = default_engine_settings((eng,))[eng]
        st.clear()
        st.update(fresh)
        st.update(keep)
        self._sync_controls()
        self.render_view(final=True)

    def _sync_solved_colour(self):
        """The box shows only with a solved reflectance, and says which colour is in use."""
        D = self.win.render_data
        solved = getattr(D, "solved_colour", None)
        self.chk_solved_colour.setVisible(solved is not None)
        self.chk_solved_colour.blockSignals(True)
        self.chk_solved_colour.setChecked(solved is not None and D.albedo_rgb_full is solved[0])
        self.chk_solved_colour.blockSignals(False)

    def _solved_colour_toggled(self, on: bool):
        from helio.application.pipeline import use_solved_colour
        use_solved_colour(self.win.render_data, on)
        self.win.refresh()
        self.schedule_render(final=True)

    # ── white balance (the colour layer only) ────────────────────────
    def _sync_white_balance(self):
        from helio.core.white_balance import describe
        wb = self.project.white_balance
        self.wb_note.setText(f"White balance: {describe(wb)}")
        self.btn_wb_reset.setEnabled(bool(wb))
        self.btn_wb_pick.setEnabled(self.project.solved)

    def _wb_pick_toggled(self, on: bool):
        if on:
            if not (self.project.solved and self.win.render_data is not None):
                self.btn_wb_pick.setChecked(False)
                self.win.flash("White balance needs the colour of a processed capture: process first.")
                return
            if self.mode != "view":
                self.btn_wb_pick.setChecked(False)
                self.win.flash("Pick the white on the view, not on a single photograph.")
                return
            self.btn_select.setChecked(False)
            self.btn_section.setChecked(False)
            self.win.flash("Click a white or grey that should be neutral, or drag a box over it. "
                           "Esc to cancel.", 12)
            self.viewport.set_tool("rect")
        elif self.viewport.tool == "rect":
            self.viewport.set_tool("")
            self.viewport.clear_rect()

    def pick_white_balance(self):
        """⌘K / menu: start the picker."""
        self.btn_wb_pick.setChecked(True)

    def _wb_rect(self, x1, y1, x2, y2):
        if not self.btn_wb_pick.isChecked():
            return
        self.btn_wb_pick.setChecked(False)
        try:
            self.set_white_from_box(x1, y1, x2, y2)
        except ValueError as e:
            self.win.flash(str(e), 10)
        except Exception as e:                   # never more than a message: it is only the colour
            import logging
            logging.getLogger(__name__).warning("white balance pick failed", exc_info=True)
            self.win.flash(f"White balance not set: {e}", 10)

    def set_white_from_box(self, x1, y1, x2, y2):
        """A box (or a click) on the screen image: the gains that make it neutral, applied."""
        from helio.application.pipeline import white_balance_base
        from helio.core.white_balance import describe, gains_from_patch
        r, D = self.project.result, self.win.render_data
        ax, ay = self._image_xy(x1, y1)
        bx, by = self._image_xy(x2, y2)
        xa, xb = sorted((int(round(ax)), int(round(bx))))
        ya, yb = sorted((int(round(ay)), int(round(by))))
        if xb - xa < 3 or yb - ya < 3:                    # a click: a small patch around it
            cx, cy = (xa + xb) // 2, (ya + yb) // 2
            half = max(2, int(round(0.004 * max(D.w, D.h))))
            xa, xb, ya, yb = cx - half, cx + half + 1, cy - half, cy + half + 1
        xa, ya = max(0, xa), max(0, ya)
        xb, yb = min(D.w, xb), min(D.h, yb)
        if xb <= xa or yb <= ya:
            raise ValueError("Pick inside the picture.")
        base = white_balance_base(r)
        solved = getattr(D, "solved_colour", None)
        src = base["refl"] if (solved and D.albedo_rgb_full is solved[0] and base["refl"] is not None) \
            else base["albedo"]
        gains = gains_from_patch(src[ya:yb, xa:xb])
        self.set_white_balance(gains, (xa, ya, xb, yb))
        self.win.flash(f"White balance set: {describe(gains)}. Reset puts the colour back.", 8)

    def set_white_balance(self, gains, patch=None):
        from helio.application.pipeline import apply_white_balance
        p, r, D = self.project, self.project.result, self.win.render_data
        p.white_balance = tuple(float(g) for g in gains) if gains else None
        p.white_patch = tuple(int(v) for v in patch) if (gains and patch) else None
        if r is not None:
            apply_white_balance(r, p.white_balance, D)
        self.viewport.clear_rect()
        self.win._sel_cache = None                         # a selection's cut colour is cut again
        self._sync_white_balance()
        self.win.refresh()
        self.schedule_render(final=True)

    def reset_white_balance(self):
        self.btn_wb_pick.setChecked(False)
        try:
            self.set_white_balance(None)
        except Exception:
            import logging
            logging.getLogger(__name__).warning("white balance reset failed", exc_info=True)

    def _colour_gain_changed(self, _k, v, final):
        self.win.display.colour_gain = float(v)
        self.schedule_render(final=final)

    def _colour_changed(self, _k, v, final):
        self.win.display.colour_opacity = float(v)
        self.schedule_render(final=final)

    def _conf_changed(self, key, v, final):
        if key in ("shadow", "highlight"):
            v = int(round(v))
        self.win.display.conf[key] = v
        self.schedule_render(final=final)

    def set_layer(self, layer):
        self.win.display.layer = layer
        self._sync_controls()
        self.render_view(final=True)
        self.win.update_bottom()

    def _adjust_changed(self, key, value, final):
        self.win.display.adjustments[key] = float(value)
        self.render_view(final=final)

    def _adjust_reset(self):
        from helio.application.viewing import GLOBAL_ADJUSTMENTS
        for k, v in GLOBAL_ADJUSTMENTS:
            self.win.display.adjustments[k] = v
            self.adjust_rows[k].set_value(v)
        self.render_view(final=True)

    def _display_flag(self, attr, on):
        setattr(self.win.display, attr, bool(on))
        self._update_badge()
        self.render_view(final=True)

    # ── split view ────────────────────────────────────────────────────
    def _compare_view(self, name: str):
        if name:
            self.win.compare_view = name
            if self.win.compare_on:
                self.render_view(final=True)

    def _compare_colour(self, on: bool):
        self.win.compare_colour = bool(on)
        self.compare_combo.setEnabled(not on)
        if self.win.compare_on:
            self.render_view(final=True)

    # ── the light ─────────────────────────────────────────────────────
    def write_light(self, lx, ly, lz, engines=None):
        """Store a light as the classic dome does: lx, ly, lz and the compass angles.
        The 3D relief takes the same light."""
        from helio.domain.view_controls import light_settings
        vals = light_settings(lx, ly, lz)
        for eng in engines or [self.win.engine]:
            self.win.engine_settings.setdefault(eng, {}).update(vals)
        if self.relief is not None:
            self.relief.set_light_screen(lx, ly, lz)
        if self.win.compare_on and self.win.compare_view in self.win.engine_settings:
            self.win.engine_settings[self.win.compare_view].update(vals)   # B relights with A

    def _light_matters(self) -> bool:
        """Whether moving the light can change the picture on screen.

        A view the light does not reach (helio.domain.view_controls.LIGHT_KIND,
        checked view by view in tests/test_render_speed.py) keeps its picture, so
        dragging the dome over it draws nothing at all."""
        from helio.domain.view_controls import light_kind
        w = self.win
        if w.view in self.SPECIAL:
            return False
        if light_kind(w.engine) is not None:
            return True
        return bool(w.compare_on and not w.compare_colour
                    and light_kind(w.compare_view) is not None)

    def _resweep(self, lx, ly, lz) -> bool:
        """A light placed while the sweep runs: the sweep carries on from there, at
        that height (it used to stop, and Stop then put back the light from before
        the sweep).  False when no sweep is running."""
        if not self.sweep_timer.isActive():
            return False
        from helio.application.viewing import Sweep
        self.sweep = Sweep.starting_at((lx, ly, lz), lower=False)
        self._sweep_restore = None                   # the user has chosen a light now
        return True

    def _dome_moved(self, lx, ly, lz, dragging):
        self._resweep(lx, ly, lz)
        self.write_light(lx, ly, lz)
        self._show_angles(lx, ly, lz)
        if self._light_matters():
            self.schedule_render(final=not dragging)

    def _relight_drag(self, dx, dy, dragging):
        from helio.domain.view_controls import from_dome, light_kind
        if self.win.view in self.SPECIAL or light_kind(self.win.engine) is None:
            return
        lx, ly, lz = from_dome(dx, dy)
        self._resweep(lx, ly, lz)
        self.write_light(lx, ly, lz)
        self.dome.set_light(lx, ly, lz)
        self._show_angles(lx, ly, lz)
        self.schedule_render(final=not dragging)

    def _typed_angles(self, *_):
        from helio.domain.view_controls import from_angles
        lx, ly, lz = from_angles(self.az.value() % 360, max(5.0, min(85.0, self.el.value())))
        self.write_light(lx, ly, lz)
        self.dome.set_light(lx, ly, lz)
        self.height_row.set_value(round(self.el.value()))
        self.render_view(final=True)

    def _height_moved(self, _k, v, final):
        from helio.domain.view_controls import at_elevation, light_of
        lx, ly, _lz = light_of(self.settings(), self.win.engine)
        lx, ly, lz = at_elevation(lx, ly, float(v))
        self._resweep(lx, ly, lz)
        self.write_light(lx, ly, lz)
        self.dome.set_light(lx, ly, lz)
        self._show_angles(lx, ly, lz)
        self.schedule_render(final=final)

    def same_light_everywhere(self):
        from helio.domain.view_controls import light_of
        lx, ly, lz = light_of(self.settings(), self.win.engine)
        self.write_light(lx, ly, lz, engines=list(self.win.engine_settings))
        self.win.flash("Every view now has this light.")

    def _show_captured(self, on):
        self.dome.show_captured = bool(on)
        self.dome.update()

    def best_raking(self):
        from helio.application.viewing import best_raking_light
        D = self.win.render_data
        L = getattr(D, "current_L_mat", None) if D is not None else None
        if L is None or D.normals_proxy is None:
            self.win.flash("Best raking needs the capture's lights: process with calibrated lights.")
            return
        i, (lx, ly, lz), az, el = best_raking_light(L, D.normals_proxy)
        self.write_light(lx, ly, lz)
        self.dome.set_light(lx, ly, lz)
        self._show_angles(lx, ly, lz)
        self.render_view(final=True)
        self.win.flash(f"Best contrast: light {i + 1} of {len(L)}, from {az:.0f}°, {el:.0f}° up.")

    def toggle_sweep(self):
        from helio.application.viewing import Sweep
        from helio.domain.view_controls import light_kind, light_of
        if self.sweep_timer.isActive():
            self.stop_sweep()
            return
        if self.win.view in self.SPECIAL or light_kind(self.win.engine) is None:
            self.btn_sweep.setChecked(False)
            self.win.flash("This view does not change with the light.")
            return
        before = tuple(float(v) for v in light_of(self.settings(), self.win.engine))
        self.sweep = Sweep.starting_at(before)
        # a light high overhead sweeps low (it would hardly move), and goes back on Stop
        self._sweep_restore = before if self.sweep.lowered else None
        if self.sweep.lowered:
            from helio.application.viewing import SWEEP_MAX_EL, SWEEP_RAKE_EL
            self.win.flash(f"The light is overhead, where a sweep shows little: sweeping at "
                           f"{SWEEP_RAKE_EL:.0f}° and putting it back on Stop. Place it on the "
                           f"dome to sweep at another height.")
        self.btn_sweep.setChecked(True)
        self.btn_sweep.setText("Stop")
        self.sweep_timer.start()

    def stop_sweep(self):
        if self.sweep_timer.isActive():
            self.sweep_timer.stop()
            self.btn_sweep.setChecked(False)
            self.btn_sweep.setText("Sweep")
            back = getattr(self, "_sweep_restore", None)
            self._sweep_restore = None
            if back is not None:
                self.write_light(*back)
                self.dome.set_light(*back)
                self._show_angles(*back)
            self.render_view(final=True)

    def _sweep_step(self):
        if self.sweep is None or self.mode != "view":
            return self.stop_sweep()
        if getattr(self, "_busy", False):
            return                                   # the last frame is still drawing
        lx, ly, lz = self.sweep.step(self.sweep_speed.value())
        self.write_light(lx, ly, lz)
        self.dome.set_light(lx, ly, lz)
        self._show_angles(lx, ly, lz)
        self.render_view(final=False)

    # ── rendering ─────────────────────────────────────────────────────
    def schedule_render(self, final=True):
        """Every view follows the light live.  (Until 2026-10-15 a view slower
        than 150 ms waited for the release, and a frame that finished after
        the next mouse move was dropped, so a drag showed nothing at all.)"""
        self.render_view(final=final)

    def _draw_fn(self, view, engine, settings, display, D, with_hist=True):
        """The image for the main channel: (A, B or None, histogram or None).

        The histogram is measured here, on the render thread, and only for
        the frame a drag ends on: the window thread never touches the
        picture's pixels."""
        from helio_qt.controls import Histogram
        import copy
        from helio.application.viewing import screen_image
        d = copy.deepcopy(display)
        if view == self.DEPTH_MAP:
            d.layer = "Depth Map (Greyscale)"
        st = dict(settings)
        win = self.win
        if view == self.BACKLIT:
            from helio.application.viewing import apply_transforms
            pn = D.normals_proxy
            img = win.backlit_image(pn.shape[:2])

            def fn_b():
                if img is None:
                    raise RuntimeError("no backlit frame lined up yet")
                a = apply_transforms(img, d)
                return a, None, (Histogram.measure(a) if with_hist else None), 0.0
            return fn_b
        cmp_on, cmp_view, cmp_colour = win.compare_on, win.compare_view, win.compare_colour
        cmp_st = dict(win.engine_settings.get(cmp_view, {}))

        def fn():
            import time as _t
            t0 = _t.perf_counter()
            a = screen_image(engine, D, st, d)
            b = None
            if cmp_on:
                if cmp_colour:
                    import cv2
                    alb = D.albedo_proxy
                    b = cv2.resize(alb, (a.shape[1], a.shape[0])) if alb.shape[:2] != a.shape[:2] else alb
                else:
                    db = copy.deepcopy(d)
                    if db.layer == "Depth Map (Greyscale)":
                        db.layer = "Combined Colour"
                    b = screen_image(cmp_view, D, cmp_st, db)
            ms = (_t.perf_counter() - t0) * 1000.0           # the picture, not the histogram
            return a, b, (Histogram.measure(a) if with_hist else None), ms
        return fn

    def render_view(self, final=True):
        """Draw the current view (preview resolution) on the worker."""
        D, view = self.win.render_data, self.win.view
        if D is None or view is None or self.mode != "view":
            return
        self._busy = True
        self._final_pending = final
        tag = self._main_tag()
        if tag != self._main_tag_seen:               # another picture: the old detail is not of it
            self._main_tag_seen = tag
            self.viewport.clear_detail()
        scale = 1.0 if final else self._drag_scale_for(view)
        src = D if scale >= 0.999 else self._scaled_data(D, scale)
        g = self.renderer.request("main", self._draw_fn(view, self.win.engine, self.settings(),
                                                        self.win.display, src,
                                                        with_hist=final and self.sec_hist.is_open()),
                                  tag=tag)
        self._req_scale[g] = scale
        self._req_scale.pop(g - 64, None)
        self._detail_after_main(final)

    def _main_tag(self):
        D, d, w = self.win.render_data, self.win.display, self.win
        return (w.view, w.engine, id(D), d.layer, d.normal_invert, d.normal_flip_x, d.depth_invert,
                d.flip_h, d.flip_v, d.rot180, w.compare_on, w.compare_view, w.compare_colour)

    # ── the part on screen at full resolution ──────────────────────────
    def _needs_refit(self) -> bool:
        """A tight machine did not keep the PTM; drawing this view full size would fit it
        again from the photographs (minutes), which the render thread must not do."""
        from helio.application.render import _PTM_ENGINES
        D = self.win.render_data
        return bool(getattr(D, "ptm_deferred", False) and getattr(D, "ptm_coeffs", None) is None
                    and self.win.engine in _PTM_ENGINES)

    def _detail_wanted(self):
        """(box in stored pixels, rectangle in scene pixels) to draw at full size, or None."""
        from helio.application.viewing import image_point
        D, w, vp = self.win.render_data, self.win, self.viewport
        from helio.application.viewing import DETAIL_SKIP
        if (not self.live_detail or D is None or self.mode != "view" or w.view == self.BACKLIT
                or w.engine in DETAIL_SKIP
                or w.compare_on or vp.item.pixmap().isNull() or self._needs_refit()):
            return None
        ratio = vp.item.pixmap().width() / float(max(D.w, 1))
        if vp.screen_px_per_image_px() <= ratio * DETAIL_ZOOM_OVER:
            return None                              # the preview is not being enlarged: nothing to gain
        r = vp.visible_rect()
        if r.isEmpty():
            return None
        px, py = r.width() * 0.08, r.height() * 0.08          # a margin, so a small pan is covered
        x1, y1 = max(0, int(r.left() - px)), max(0, int(r.top() - py))
        x2, y2 = min(D.w, int(r.right() + px) + 1), min(D.h, int(r.bottom() + py) + 1)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        from helio.utils.memory_plan import system_memory
        total = system_memory()[0]
        cap = DETAIL_PX[(total >= 24 * 2 ** 30) + (total >= 48 * 2 ** 30)]
        if (x2 - x1) * (y2 - y1) > cap:
            return None                              # zoomed out: the preview is close to full size
        ax, ay = image_point(x1, y1, D.w, D.h, w.display)
        bx, by = image_point(x2, y2, D.w, D.h, w.display)
        box = (int(round(min(ax, bx))), int(round(min(ay, by))),
               int(round(max(ax, bx))), int(round(max(ay, by))))
        return box, (x1, y1, x2, y2)

    def _detail_after_main(self, final: bool):
        """With each main frame: draw the detail too when it can keep up with the light."""
        if not self.live_detail:
            return
        if final:
            self._detail_timer.stop()
            self._detail_request()
        elif 0 < self._detail_ms.get(self.win.view or "", 0.0) <= DETAIL_LIVE_MS:
            self._detail_request()
        else:
            self.viewport.clear_detail()             # a different light beside the preview would mislead

    def _detail_request(self):
        import copy
        want = self._detail_wanted()
        if want is None:
            self.viewport.clear_detail()
            return
        box, rect = want
        D, view, eng = self.win.render_data, self.win.view, self.win.engine
        st = dict(self.settings())
        d = copy.deepcopy(self.win.display)
        if view == self.DEPTH_MAP:
            d.layer = "Depth Map (Greyscale)"
        tag = self._main_tag()

        def fn():
            from helio.application.viewing import detail_image
            return detail_image(eng, D, st, d, box), rect, tag
        self.renderer.request("detail", fn, priority=-1)

    def _detail_done(self, gen, out, err, ms):
        if not self.renderer.latest("detail", gen) or err or out is None or self.mode != "view":
            return
        img, rect, tag = out
        if tag != self._main_tag():
            return
        import time
        t0 = time.perf_counter()
        self.viewport.show_detail(img, rect)
        # what a live frame of it costs: drawn, and put on screen by the GUI thread
        self._detail_ms[self.win.view or ""] = ms + (time.perf_counter() - t0) * 1000.0

    def _view_moved(self):
        """Panned or zoomed: draw the new part at full size once the movement settles."""
        if self.live_detail and self.mode == "view":
            self._detail_timer.start()

    def _set_live_detail(self, on: bool):
        self.live_detail = bool(on)
        if on:
            self._detail_request()
        else:
            self.viewport.clear_detail()

    def _warm(self, view):
        """Draw this view once, in the background, at the scale its drag frames
        will use: the resized arrays and the engine's own caches are then
        ready, and the first frame of a drag is as quick as the rest."""
        D = self.win.render_data
        scale = self._drag_scale_for(view)
        key = (id(D), view, scale)
        if D is None or scale >= 0.999 or key in self._warmed:
            return
        self._warmed = {k for k in self._warmed if k[0] == id(D)} | {key}
        fn = self._draw_fn(view, self.win.engine, self.settings(), self.win.display,
                           self._scaled_data(D, scale), with_hist=False)
        self.renderer.request("warm", fn)

    def _scaled_data(self, D, scale):
        """The solve's preview arrays at *scale*, made once per solve and scale."""
        from helio.application.viewing import ScaledData
        key = (id(D), scale)
        sd = self._scaled.get(key)
        if sd is None or object.__getattribute__(sd, "_d") is not D:
            self._scaled = {k: v for k, v in self._scaled.items()
                            if k[0] == id(D) and object.__getattribute__(v, "_d") is D}
            sd = self._scaled[key] = ScaledData(D, scale)
        return sd

    def _drag_scale_for(self, view) -> float:
        """The step for this view's drag frames: from its last full-size time
        until a drag has been timed, then from the drag frames themselves."""
        s = self.drag_scale.get(view)
        if s is None:
            t = self.view_ms.get(view or "", 0.0)
            if t <= DRAG_TARGET_MS:
                return 1.0
            s = (DRAG_TARGET_MS / t) ** 0.5
        return next((q for q in DRAG_SCALES if q <= s + 1e-6), DRAG_SCALES[-1])

    def _timed(self, view, scale, ms):
        """A frame's time at *scale*: pixels go as the square of the scale, so
        the scale that would have met the target is scale·√(target / ms).
        Eased, so one slow frame does not halve the next."""
        if ms <= 0:
            return
        if scale >= 0.999:
            self.view_ms[view or ""] = ms
        ideal = max(DRAG_SCALES[-1], min(1.0, scale * (DRAG_TARGET_MS / ms) ** 0.5))
        prev = self.drag_scale.get(view)
        self.drag_scale[view] = ideal if prev is None else 0.65 * prev + 0.35 * ideal

    def wait_render(self):
        """Tests and exports: let the worker finish and deliver."""
        from PySide6.QtWidgets import QApplication
        self.renderer.wait()
        QApplication.processEvents()

    def _rendered(self, channel, gen, out, ms, err):
        if channel == "loupe":
            return self._loupe_done(gen, out, err)
        if channel == "detail":
            return self._detail_done(gen, out, err, ms)
        if channel in ("hist", "warm"):
            return
        latest = self.renderer.latest(channel, gen)
        if not latest and (gen <= self._shown_gen or not self.renderer.same_picture(channel, gen)):
            return                   # older than what is shown, or a picture no longer wanted
        self._shown_gen = gen
        if latest:
            self._busy = False
        if self.mode != "view":
            return                       # a photograph was chosen while this was drawing
        D = self.win.render_data
        if err:
            self.error = f"“{self.view_name(self.win.view)}” could not be drawn — {err}. The picture shown is the previous one."
            self.win.update_bottom()
            return
        self.error = ""
        a, b, hist, pic_ms = out
        self._timed(self.win.view, self._req_scale.pop(gen, 1.0), pic_ms)
        self._last_image = a
        if self.relief is not None and self.relief.isVisible():
            self._push_relief()
            self.relief.set_view_image(self._view_for_relief(a))
        self.last_ms = ms
        self.viewport.show_array(a, full_size=(D.w, D.h) if D is not None else None)
        self.viewport.set_compare(b)
        if hist is not None:
            self.hist.set_data(*hist)
        if not latest:
            return                           # the newer frame is on its way
        if self._final_pending:
            self._warm(self.win.view)
        if self._loupe_at is not None and self._final_pending:
            self._loupe_request()            # full resolution: once the drag ends
        if self._pending_final and not self.sweep_timer.isActive():
            self._pending_final = False
            self.render_view(final=True)
            return
        self.win.update_bottom()

    def _measure_summary(self):
        p = self.project
        mm = p.scale_mm_per_px
        self.m_rows["scale"].setText(f"Scale {mm:.4g} mm per pixel" if mm else "Scale not set")
        self.m_rows["height"].setText(f"Heights {p.z_mm_per_unit * 1000:.4g} µm per px of height" if p.z_mm_per_unit
                                      else "Heights relative")
        cc = getattr(p, "colour_check", None)
        self.m_rows["colour"].setText(f"Colour ΔE₀₀ {cc['mean']:.2f} average" if cc
                                      else "Colour not checked")
        if cc:
            self.m_rows["colour"].setToolTip(f"{cc['file']}: {cc['verdict']} (worst {cc['max']:.2f}, "
                                             f"{cc['worst']})")

    def _hist_opened(self, on):
        """Opening the histogram measures the picture already on screen."""
        a = getattr(self, "_last_image", None)
        if on and a is not None:
            self.hist.set_image(a)

    # ── pointer: probe and loupe ──────────────────────────────────────
    def _image_xy(self, x, y):
        """A point on the screen image (scene, full-size pixels) in stored image pixels."""
        from helio.application.viewing import image_point
        D = self.win.render_data
        return image_point(x, y, D.w, D.h, self.win.display)

    def _probed(self, x, y):
        D = self.win.render_data
        if self.mode != "view" or D is None or x < 0:
            self.probe.hide()
            self._probe_at = None
            return
        ix, iy = self._image_xy(x, y)
        ix, iy = int(min(max(ix, 0), D.w - 1)), int(min(max(iy, 0), D.h - 1))
        self._probe_at = (ix, iy)
        self.probe.setText(self.probe_text(ix, iy))
        self.probe.adjustSize()
        self.probe.show()
        if not self.loupe.held:
            self._loupe_at = (ix, iy)
            self._loupe_request()

    def probe_text(self, ix: int, iy: int) -> str:
        """Pixel, position in mm, height, normal and its slope, uncertainty."""
        D, p = self.win.render_data, self.project
        parts = [f"x {ix}  y {iy}"]
        mm = p.scale_mm_per_px
        if mm:
            parts[0] += f"   ({ix * mm:.2f}, {iy * mm:.2f} mm)"
        from helio.application.heights import height_px
        H = height_px(D)
        if H is not None and 0 <= iy < H.shape[0] and 0 <= ix < H.shape[1]:
            z = float(H[iy, ix])                            # true height above the lowest point
            zline = f"height {z:.2f} px"
            if p.z_mm_per_unit:
                zline += f"  ({z * p.z_mm_per_unit:.3f} mm)"
            parts.append(zline)
        nf = getattr(D, "normals_full", None)
        if nf is not None:
            n = nf[iy, ix]
            slope = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
            parts.append(f"normal ({n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f})  slope {slope:.1f}°")
        sg = getattr(D, "qc_sigma_full", None)
        if sg is not None:
            parts.append(f"normal ± {float(sg[iy, ix]):.2f}°")
        return "\n".join(parts)

    def _clicked(self, x, y):
        if self.mode != "view" or self.win.render_data is None:
            return
        ix, iy = self._image_xy(x, y)
        self._loupe_at = (int(ix), int(iy))
        self.loupe.set_held(True)
        self._loupe_request()

    def _double_clicked(self):
        self.loupe.set_held(False)
        self.loupe.level = 1.0                          # the classic resets the zoom too
        self.loupe.pct.setText("100%")

    def loupe_box(self):
        """(x1, y1, x2, y2) in stored full-resolution pixels (update_magnifier)."""
        D = self.win.render_data
        if D is None or self._loupe_at is None:
            return None
        cx, cy = self._loupe_at
        c = self.loupe.crop_px()
        x1, y1 = max(0, cx - c // 2), max(0, cy - c // 2)
        x2, y2 = min(D.w, x1 + c), min(D.h, y1 + c)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _loupe_request(self):
        import copy
        from helio.application.viewing import apply_transforms, screen_image
        D, box = self.win.render_data, self.loupe_box()
        if (D is None or box is None or not self.sec_loupe.is_open() or self.win.view == self.BACKLIT
                or self._needs_refit()):
            return
        view, eng = self.win.view, self.win.engine
        st = dict(self.settings())
        d = copy.deepcopy(self.win.display)
        if view == self.DEPTH_MAP:
            d.layer = "Depth Map (Greyscale)"

        def fn():
            img = screen_image(eng, D, st, d, use_proxy=False, crop=box, transforms=False)
            return apply_transforms(img, d), box
        self.renderer.request("loupe", fn)

    def _loupe_done(self, gen, out, err):
        if not self.renderer.latest("loupe", gen) or err or out is None:
            return
        img, (x1, y1, x2, y2) = out
        mm = self.project.scale_mm_per_px
        size = f"{x2 - x1} × {y2 - y1} px" + (f" · {(x2 - x1) * mm:.2f} mm wide" if mm else "")
        self.loupe.show_pixmap(pixmap_from_array(img), f"{x1}, {y1} · {size}")

    # ── saved states ──────────────────────────────────────────────────
    def save_state(self, name: str | None = None):
        from helio.application.viewing import bookmark
        from helio.domain.view_controls import light_of
        if name is None:
            name, ok = QInputDialog.getText(self.win, "Save this state",
                                            "A name for this view, light and settings:")
            if not ok or not name.strip():
                return
            name = name.strip()
        eng = self.win.engine
        self.win.bookmarks[name] = bookmark(
            eng, self.win.display.layer, light_of(self.settings(), eng), self.settings(),
            self.win.shared, mag=self.loupe.level)
        self.refresh_states()
        self.win.flash(f"Saved “{name}”.")

    def load_state(self, name: str):
        from helio.application.viewing import bookmark_light
        b = self.win.bookmarks.get(name)
        if not b:
            return
        eng = b.get("engine", self.win.engine)
        self.win.engine_settings[eng] = dict(b.get("settings", {}))
        lx, ly, lz = bookmark_light(b)
        self.win.engine_settings[eng].update(lx=lx, ly=ly, lz=lz)
        if b.get("layer"):
            self.win.display.layer = b["layer"] if b["layer"] != "Depth Map (Greyscale)" else \
                self.win.display.layer
        for k, v in (b.get("shared") or {}).items():
            if k in self.win.shared:
                self.win.shared[k] = v
        self.loupe.level = float(b.get("mag", self.loupe.level))
        self.loupe.pct.setText(f"{self.loupe.level * 100:.0f}%")
        self.win.set_view(eng)
        self.win.flash(f"Restored “{name}”.")

    def refresh_states(self):
        self.states.clear()
        for name in self.win.bookmarks:
            QListWidgetItem(name, self.states)

    # ── state → screen ───────────────────────────────────────────────
    def refresh(self):
        p = self.project
        self.win.model3d.refresh()
        r3 = self.relief
        if r3 is not None and (r3.scale_mm, r3.z_mm_per_unit) != (p.scale_mm_per_px, p.z_mm_per_unit):
            r3.set_scale(p.scale_mm_per_px, p.z_mm_per_unit)     # Measure set or changed it
        if hasattr(self, "annotator"):
            self.annotator.refresh()
        self._draw_selection()
        self._draw_section()
        if hasattr(self, "section_card"):
            self.section_card.set_backlit(self._has_backlit())
        if self.strip.paths != p.images:
            self.strip.set_frames(p.images)
            self._strip_lights = None
        if p.lights is not self._strip_lights:
            self._strip_lights = p.lights
            self.strip.set_lights(*self._frame_lights())
        solved = p.solved and self.win.render_data is not None
        self.tree.setEnabled(solved)
        self.search.setEnabled(solved)
        self.views_note.setVisible(not solved)
        self.btn_save_state.setEnabled(solved)
        self.btn_light_save.setEnabled(solved)
        if self.states.count() != len(self.win.bookmarks):
            self.refresh_states()
        bl = self.win.backlit
        self.items[self.BACKLIT].setHidden(not (bl and bl.get("H") is not None))
        if solved and self._shown_result is not p.result:
            self._shown_result = p.result
            self.mode = "view"
            self.set_view(self.win.view or "Standard PBR")
            return
        if not solved:
            self.mode = "frame"
            self.stop_sweep()
        if p.images and self.strip.currentRow() != p.current:
            self.strip.blockSignals(True)
            self.strip.setCurrentRow(p.current)
            self.strip.blockSignals(False)
        view_mode = self.mode == "view" and solved
        photo_over_view = solved and not view_mode and bool(p.images)
        back = f"◀ Back to {self.view_name(self.win.view or 'Standard PBR')}"
        for b in (self.btn_back_view, self.btn_back_view2):
            b.setText(back)
            b.setVisible(photo_over_view)
        for sc in getattr(self, "_esc_backs", ()):
            sc.setEnabled(photo_over_view)
        self.insp_stack.setCurrentIndex(1 if view_mode else 0)
        self._measure_summary()
        self.view_box.setVisible(view_mode)
        self.view_blurb.setVisible(view_mode)
        if view_mode:
            from helio.domain.view_controls import light_kind
            lit = self.win.view not in self.SPECIAL and light_kind(self.win.engine) is not None
            self.dome.setVisible(lit)
            self._update_badge()
        else:
            self.dome.hide()
            self.probe.hide()
            self.badge.hide()
            if photo_over_view:
                self.badge.setText(f"  Frame {p.current + 1} of {len(p.images)}, as photographed · Esc: back to "
                                   f"{self.view_name(self.win.view or 'Standard PBR')}  ")
                self.badge.show()
                self.badge.adjustSize()
                self.viewport._place_overlays()
            self.viewport.set_compare(None)
            if p.images:
                from helio_qt.workspaces import light_angles
                i = p.current
                self.viewport.set_image(p.images[i])
                self.frame_title.setText(f"Frame {i + 1} of {len(p.images)}")
                lines = [os.path.basename(p.images[i]), self.win.frame_size_text()]
                if p.lights is not None and i < len(p.lights) and \
                        bool(__import__("numpy").isfinite(p.lights[i]).all()):
                    az, el = light_angles(p.lights[i])
                    lines.append(f"Light from {az:.0f}° (from north), {el:.0f}° up")
                self.frame_info.setText("\n".join(lines))
                self.frame_hint.setText("The photograph as taken. The view is kept as you left it."
                                        if solved else
                                        "Relighting, depth and clay views appear here after "
                                        "processing.")
            else:
                self.viewport.set_image("")
                self.frame_title.setText("No frame")
                self.frame_info.setText("")

    def next_step(self):
        p = self.project
        if not p.images:
            return "Import images…", "file.import"
        if p.solved:
            return "Export… →", "file.export"
        if not p.lights_ready():
            return "Set the lights in Capture →", "view.ws:Capture"
        return "Go to Process →", "view.ws:Process"

    def status(self):
        p = self.project
        if not p.images:
            return "idle", "No images", "Import a folder of photographs to begin"
        if self.mode == "view" and p.solved and self.win.view:
            if self.error:
                return "problem", self.win.view, self.error
            from helio.domain.view_controls import light_kind
            bits = []
            if self.win.view not in self.SPECIAL and light_kind(self.win.engine):
                bits.append(self.dome.caption())
            layer = dict(LAYER_KEYS).get(self.win.display.layer, "")
            if self.win.view not in self.SPECIAL and layer:
                bits.append(layer)
            bits.append(f"drawn in {self.last_ms:.0f} ms")
            return "ok", self.win.view, " · ".join(bits)
        return ("idle", f"Frame {p.current + 1} of {len(p.images)}",
                f"{self.win.frame_size_text()} · {os.path.basename(p.images[p.current])}")
