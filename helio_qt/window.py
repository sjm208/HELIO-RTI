"""The main window: header, five workspaces, and the next-step bar (QT-001, UX-002)."""
from __future__ import annotations

import os
import time

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QGuiApplication, QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QHBoxLayout, QMainWindow,
                               QMessageBox, QRadioButton, QStackedWidget, QVBoxLayout, QWidget)

from helio_qt import __version__, icons, theme
from helio_qt.actions import Registry
from helio_qt.palette import CommandPalette, ShortcutsSheet
from helio_qt.state import Project
from helio_qt.widgets import (WORKSPACES, FrameList, Thumbs, WorkspaceBar, button, label,
                              restyle)
from helio_qt.workspaces import INSPECTOR_W, SIDE_W, WORKSPACE_CLASSES


# Solve choices with no classic session key: kept under ``helio3`` (written only when not the default).
SOLVE_KEPT = ("semi_calibrated", "metal_aware", "metal_solver", "metal_sensitivity", "rpca_lambda",
              "light_field", "lens_calibration")


def settings():
    """Preferences; HELIO_QT_SETTINGS points tests at a scratch file."""
    from PySide6.QtCore import QSettings
    path = os.environ.get("HELIO_QT_SETTINGS")
    if path:
        return QSettings(path, QSettings.Format.IniFormat)
    return QSettings("HELIO", "HELIO Pro 3")      # where preferences have always been: renaming would lose them


# ── sphere detection off the main thread ─────────────────────────────


def _height_basis() -> str:
    from helio.application.heights import HEIGHT_BASIS
    return HEIGHT_BASIS

class _Done(QObject):
    finished = Signal(object, object, str)


class _SphereJob(QRunnable):
    """Find the ball(s), then read every frame's highlight (helio.core.sphere)."""

    def __init__(self, paths, sig):
        super().__init__()
        self.paths, self.sig = list(paths), sig

    def run(self):
        try:
            from helio.core.processing import detect_sphere_candidates
            from helio.core.sphere import Sphere, calibrate, image_size
            size = image_size(self.paths[0])          # from the header: no 100 MP decode
            if size is None:
                raise RuntimeError(f"{os.path.basename(self.paths[0])} could not be read")
            w, h = size
            cands = detect_sphere_candidates(self.paths, h, w, want_two=True) or []
            notes = list(getattr(detect_sphere_candidates, "notes", [])) + [""] * len(cands)
            cal = calibrate(self.paths, Sphere(*cands[0], note=notes[0])) if cands else None
            if cal is not None and len(cands) > 1:
                from helio.core.sphere import with_second_ball
                cal = with_second_ball(self.paths, cal, Sphere(*cands[1])) or cal
            _emit_if_alive(self.sig.finished, cal, cands, "")
        except Exception as e:                      # reported, never swallowed
            _emit_if_alive(self.sig.finished, None, None, f"{type(e).__name__}: {e}")


class _BgSignals(QObject):
    done = Signal(object, object, str)          # callback, result, error


class _BgTask(QRunnable):
    def __init__(self, fn, cb, sig):
        super().__init__()
        self.fn, self.cb, self.sig = fn, cb, sig

    def run(self):
        try:
            result, err = self.fn(), ""
        except Exception as e:                      # reported, never swallowed
            import logging
            logging.getLogger(__name__).exception("background task failed")
            result, err = None, f"{type(e).__name__}: {e}"
        _emit_if_alive(self.sig.done, self.cb, result, err)


def _emit_if_alive(signal, *args):
    """Hand a finished task's result back, unless its window has closed
    meanwhile (quitting mid-task): then there is no one to tell."""
    try:
        signal.emit(*args)
    except RuntimeError as e:
        if "deleted" not in str(e):
            raise


class _JobSignals(QObject):
    """Job callbacks arrive on the job's thread; these queue them to the window's."""
    progress = Signal(int, str, float, str)
    done = Signal(object)
    failed = Signal(object)
    cancelled = Signal(object)
    stale = Signal(object)


def _plain(o):
    """*o* as JSON takes it: arrays and numpy numbers as lists and floats, tuples as lists."""
    import numpy as np
    if isinstance(o, np.ndarray):
        return [_plain(v) for v in o.tolist()]
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def section_for_sheet(prof: dict, card=None) -> dict:
    """A section's profile as the publication sheet keeps it: without its feature list, and with its parallel
    copies (their distances, heights, offsets and ends) when the section card has them on."""
    out = {k: v for k, v in prof.items() if k not in ("features", "series")}
    ser = getattr(card, "series", None) if card is not None else None
    if ser and len(ser.get("profiles") or []) > 1:
        out["series"] = [{k: p[k] for k in ("dist", "z", "offset", "a_px", "b_px") if k in p}
                         for p in ser["profiles"]]
        out["spacing"] = float(ser.get("spacing", 0.0))
    return out


class MainWindow(QMainWindow):
    def __init__(self, appearance: str | None = None):
        super().__init__()
        self.setWindowTitle("Helio")
        # Below this the workspace tabs, Inspect's toolbar and Process's text
        # cannot fit beside the two side panels (300 + 320 px): they were cut.
        self.setMinimumSize(1200, 760)
        self.prefs = settings()
        # Dark until the user picks otherwise (View › Appearance): images are
        # judged on the dark surround, whatever macOS is set to.
        self.appearance = appearance or str(self.prefs.value("appearance", theme.DEFAULT_APPEARANCE))
        if self.appearance not in theme.APPEARANCES:
            self.appearance = theme.DEFAULT_APPEARANCE
        self.scheme = str(self.prefs.value("scheme", theme.DEFAULT_FAMILY))
        if self.scheme not in theme.FAMILIES:
            self.scheme = theme.DEFAULT_FAMILY
        self.project = Project()
        self.thumbs = Thumbs()
        self.pool = QThreadPool(self)
        from helio.application.jobs import JobManager
        self.jobs = JobManager()
        self._job_sig = _JobSignals()
        self._job_sig.progress.connect(self._job_progress)
        self._job_sig.done.connect(self._solve_done)
        self._job_sig.failed.connect(self._solve_failed)
        self._job_sig.cancelled.connect(lambda job: self.project.finish(
            "Cancelled. The previous result, if any, is kept."))
        self._job_sig.stale.connect(lambda job: None)    # a newer solve is on its way
        self._export_sig = _JobSignals()
        self._export_sig.progress.connect(self._export_progress)
        self._export_sig.done.connect(self._export_done)
        self._export_sig.failed.connect(lambda job: self._export_stopped(
            job, f"Export stopped: {job.failure}"))
        self._export_sig.cancelled.connect(lambda job: self._export_stopped(
            job, "Export cancelled. Files already written are kept."))
        self._export_sig.stale.connect(lambda job: None)
        self._size_cache: tuple[str, str] = ("", "")
        # What Inspect shows (VIEW-002): the solved surface, held where the
        # render engines read it; the chosen view; each view's settings, its
        # light included (engine_settings[view]['lx', 'ly', 'lz'], screen
        # frame, as the classic dome writes them); the display state.
        from helio.application.exports import shared_defaults
        from helio.application.viewing import Display, History
        from helio.domain.registry import default_engine_settings
        self.render_data = None
        self._render_of = None
        self.view: str | None = None              # a view, or Inspect's "Depth map"
        self.engine = "Standard PBR"              # the last real view (render engine)
        self.engine_settings = default_engine_settings()
        self.display = Display(layer="Combined Colour",   # colour, as the approved design shows it
                               colour_gain=1.5)          # a dark photograph's colour, readable
        self.shared = shared_defaults()           # the 3D / export state (classic shared_state)
        from helio_qt.model3d import Model3D
        self.model3d = Model3D(self)              # the OpenGL viewer and its settings
        self.bookmarks: dict = {}                 # Safe States, carried by the session
        self.compare_on, self.compare_view, self.compare_colour = False, "Normals RGB", False
        self.flat_ref_normals = None              # Process › Flat-field correction (IMP-004)
        self.flat_ref_info: dict = {}
        self.solve_crop = None                    # Reprocess a region (REC-004): (x1, y1, x2, y2)
        self.grey_ref = {"state": "none"}         # CAL-003: what the grey-card search found
        self.selection = None                     # Inspect's selection, the solve's pixels
        self._sel_cache = None                    # ((result, box), (cut result, its data))
        self.solve_crop_ref = None
        self.canvas_rotation = 0
        self.history = History()
        self._flash = ("", 0.0)
        self.registry = Registry(self, self.project)
        self.registry.build_menus(self.menuBar())
        from helio_qt.dialog_tips import install as _tips
        _tips(QApplication.instance())                 # every dialog's controls say what they do

        root = QWidget()
        rl = QVBoxLayout(root)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        rl.addWidget(self._build_header())
        body = QHBoxLayout()
        body.setSpacing(0)
        self.side_stack, self.center_stack, self.insp_stack = (QStackedWidget(), QStackedWidget(),
                                                               QStackedWidget())
        self.side_stack.setFixedWidth(SIDE_W)
        self.insp_stack.setFixedWidth(INSPECTOR_W)
        # The left column: the steps already done, folded to a line each
        # (click one to go back), then the current step's own panel.
        from helio_qt.widgets import StepHead
        self.left = QWidget()
        self.left.setObjectName("Sidebar")
        self.left.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.left.setFixedWidth(SIDE_W)
        lcol = QVBoxLayout(self.left)
        lcol.setContentsMargins(0, 0, 0, 0)
        lcol.setSpacing(0)
        self.trail: dict[str, StepHead] = {}
        for i, name in enumerate(WORKSPACES):
            hd = StepHead(i, name)
            hd.clicked.connect(lambda _c=False, n=name: self.go_workspace(n))
            hd.hide()
            lcol.addWidget(hd)
            self.trail[name] = hd
        lcol.addWidget(self.side_stack, 1)
        body.addWidget(self.left)
        body.addWidget(self.center_stack, 1)
        body.addWidget(self.insp_stack)
        rl.addLayout(body, 1)
        rl.addWidget(self._build_bottom())
        self.setCentralWidget(root)

        self.workspaces = {}
        for cls in WORKSPACE_CLASSES:
            ws = cls(self)
            self.workspaces[ws.name] = ws
            self.side_stack.addWidget(ws.sidebar)
            self.center_stack.addWidget(ws.center)
            self.insp_stack.addWidget(ws.inspector)
        for name in ("Inspect", "Measure"):
            vp = self.workspaces[name].viewport
            for sid, spec in self.registry.specs.items():
                if spec.image_focus:
                    vp.addAction(self.registry.actions[sid])
        self.addActions([a for sid, a in self.registry.actions.items()
                         if not self.registry.specs[sid].image_focus])
        from helio_qt.actions import ALIASES
        from PySide6.QtGui import QAction, QKeySequence
        self.alias_actions = []
        for name in ("Inspect", "Measure"):
            vp = self.workspaces[name].viewport
            for key, sid in ALIASES:
                a = QAction(self)
                a.setShortcut(QKeySequence(key))
                a.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
                a.triggered.connect(lambda _c=False, sid=sid: self.registry.trigger(sid))
                vp.addAction(a)
                self.alias_actions.append((key, sid, a))

        self.palette_dlg = CommandPalette(self, self.registry)
        self.setAcceptDrops(True)
        self.project.changed.connect(self.refresh)
        self.thumbs.loaded.connect(lambda _p: None)
        self._sphere_sig = _Done()
        self._sphere_sig.finished.connect(self._sphere_done)
        self.bg_pool = QThreadPool(self)
        self.bg_pool.setMaxThreadCount(1)
        # The research tools run beside it, each on its own thread: one slow
        # tool held up every other one, and the capture checks, behind it.
        self.tool_pool = QThreadPool(self)
        self.tool_pool.setMaxThreadCount(2)       # two at most: each can hold GBs at 100 MP
        self._bg_sig = _BgSignals()
        self._bg_sig.done.connect(lambda cb, out, err: cb(out, err))
        self.backlit: dict = {}                    # the backlit frame: raw, path, H, aligned
        try:
            QGuiApplication.styleHints().colorSchemeChanged.connect(self._system_scheme_changed)
        except Exception:
            pass
        self.current = ""
        self.registry.actions["view.sidebar"].setChecked(True)
        self.registry.actions["view.inspector"].setChecked(True)
        self.set_appearance(self.appearance, save=False)
        from PySide6.QtCore import QTimer
        self._undo_timer = QTimer(self)
        self._undo_timer.setInterval(400)               # the classic _UNDO_POLL_MS
        self._undo_timer.timeout.connect(self._undo_poll)
        self._undo_timer.start()
        self.go_workspace("Capture")
        geo = self.prefs.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1440, 900)

    # ── chrome ────────────────────────────────────────────────────────
    def _build_header(self) -> QWidget:
        h = QWidget()
        h.setObjectName("Header")
        h.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(h)
        lay.setContentsMargins(14, 8, 14, 8)
        lay.setSpacing(12)
        self.btn_side = button("", "icon", "Show or hide the sidebar  ⌃⌘S")
        self.btn_side.setAccessibleName("Show or hide sidebar")
        self.btn_side.clicked.connect(lambda: self.registry.actions["view.sidebar"].trigger())
        lay.addWidget(self.btn_side)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        self.title = label("No project", "projectTitle")
        self.subtitle = label("Import a capture to begin", "projectSub")
        titles.addWidget(self.title)
        titles.addWidget(self.subtitle)
        tw = QWidget()
        tw.setLayout(titles)
        tw.setMinimumWidth(240)
        lay.addWidget(tw)
        lay.addStretch(1)
        self.ws_bar = WorkspaceBar()
        self.ws_bar.chosen.connect(self.go_workspace)
        lay.addWidget(self.ws_bar)
        lay.addStretch(1)
        self.btn_search = button("Search tools  ⌘K", "outline", "Find any tool")
        self.btn_search.clicked.connect(self.show_palette)
        lay.addWidget(self.btn_search)
        self.btn_insp = button("", "icon", "Show or hide the inspector  ⌥⌘I")
        self.btn_insp.setAccessibleName("Show or hide inspector")
        self.btn_insp.clicked.connect(lambda: self.registry.actions["view.inspector"].trigger())
        lay.addWidget(self.btn_insp)
        return h

    def _build_bottom(self) -> QWidget:
        b = QWidget()
        b.setObjectName("BottomBar")
        b.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        lay = QHBoxLayout(b)
        lay.setContentsMargins(18, 10, 18, 10)
        lay.setSpacing(12)
        self.status_icon = label("")
        self.status_text = label("", status="idle")
        self.status_detail = label("", "secondary")
        lay.addWidget(self.status_icon)
        lay.addWidget(self.status_text)
        lay.addWidget(self.status_detail, 1)
        self.reason = label("", "reason")
        lay.addWidget(self.reason)
        self.primary = button("", "primary")
        self.primary.setDefault(True)
        self.primary.clicked.connect(self._primary_clicked)
        lay.addWidget(self.primary)
        self._primary_action = ""
        return b

    def _primary_clicked(self):
        if self._primary_action:
            self.registry.trigger(self._primary_action)

    # ── state → screen ───────────────────────────────────────────────
    def frame_size_text(self) -> str:
        p = self.project
        if not p.images:
            return ""
        if self._size_cache[0] != p.images[0]:
            from helio.core.sphere import image_size
            s = image_size(p.images[0])
            self._size_cache = (p.images[0], f"{s[0]} × {s[1]} px" if s else "")
        return self._size_cache[1]

    def _sync_render_data(self):
        r = self.project.result
        if r is None:
            self.render_data, self._render_of = None, None
        elif self._render_of is not r:
            from helio.application.pipeline import store
            from helio.data_manager import DataManager
            D = DataManager()
            store(r, D)
            D.plate_paths = list(self.project.images)     # the colour views read the photos
            D.engine_settings = self.engine_settings
            if self.project.white_balance or getattr(r, "white_balance", None):
                try:                                  # the colour only: never stops the views
                    from helio.application.pipeline import apply_white_balance
                    apply_white_balance(r, self.project.white_balance, D)
                except Exception:
                    import logging
                    logging.getLogger(__name__).warning("white balance not applied", exc_info=True)
            self.render_data, self._render_of = D, r
            sel = self.selection
            if sel and (sel[2] > r.w or sel[3] > r.h):
                self.selection, self._sel_cache = None, None     # a new solve it does not fit
            self.view = self.view or self.engine or "Standard PBR"

    def refresh(self):
        p = self.project
        self._sync_render_data()
        self.title.setText(p.name)
        if p.images:
            bits = [f"{len(p.images)} frames"]
            bits.append("lights ready" if p.lights_ready() else "lights not set")
            bits.append("solved" if p.solved else "not processed")
            self.subtitle.setText(" · ".join(bits))
        else:
            self.subtitle.setText("Import a capture to begin")
        done = {"Capture"} if p.lights_ready() else set()
        if p.solved:
            done.add("Process")
        self.ws_bar.mark_done(done, self.tokens.ok)
        for ws in self.workspaces.values():
            ws.refresh()
        self._update_trail()
        self.update_bottom()

    def update_bottom(self):
        ws = self.workspaces.get(self.current)
        if not ws:
            return
        kind, text, detail = ws.status()
        icon_name = {"ok": "check", "warn": "alert", "problem": "problem", "info": "info"}.get(kind)
        colour = getattr(self.tokens, kind if kind in ("ok", "warn", "problem", "info") else "text2")
        self.status_icon.setPixmap(icons.icon(icon_name, colour, 16).pixmap(16, 16)
                                   if icon_name else icons.icon("info", self.tokens.text3, 16).pixmap(16, 16))
        self.status_text.setText(text)
        self.status_text.setProperty("status", kind)
        restyle(self.status_text)
        msg, until = getattr(self, "_flash", ("", 0.0))
        self.status_detail.setText(msg if msg and time.time() < until else detail)
        words, aid = ws.next_step()
        why = self.registry.reason(aid) if aid else ""
        self._primary_action = aid
        self.primary.setText(words)
        self.primary.setEnabled(bool(aid) and not why)
        spec = self.registry.specs.get(aid) if aid else None
        keys = f"  ({QKeySequence(spec.shortcut).toString(QKeySequence.SequenceFormat.NativeText)})" \
            if spec is not None and spec.shortcut else ""
        self.primary.setToolTip(why or (f"The next step: {spec.text}{keys}" if spec is not None
                                        else ""))
        self.reason.setText(why)
        self.reason.setVisible(bool(why))

    # ── handlers named by actions.Spec.handler ─────────────────────────
    def go_workspace(self, name: str):
        if name not in self.workspaces:
            return
        if name != "Inspect" and getattr(self, "_pick", None):
            self.cancel_pick(go_back=False)       # left Inspect: the box is not wanted now
        self.current = name
        i = WORKSPACES.index(name)
        for st in (self.side_stack, self.center_stack, self.insp_stack):
            st.setCurrentIndex(i)
        self.ws_bar.select(name)
        self.registry.actions[f"view.ws:{name}"].setChecked(True)
        self.workspaces[name].refresh()
        self._update_trail()
        self.update_bottom()

    def step_summary(self, name: str) -> tuple[bool, str]:
        """(finished, one line) for a step, as its folded row says it."""
        import numpy as np
        p = self.project
        if name == "Capture":
            if not p.images:
                return False, "No images yet"
            bits = [f"{len(p.images)} frames"]
            if p.lights_ready():
                src = p.light_source
                src = ("chrome sphere" if src == "sphere" else
                       src.split(":", 1)[1] if src.startswith("file:") else src or "lights set")
                bits.append(src)
                L = np.asarray(p.lights, np.float64)
                if L.ndim == 2 and len(L):
                    el = np.degrees(np.arcsin(np.clip(L[:, 2] / np.maximum(
                        np.linalg.norm(L, axis=1), 1e-9), -1, 1)))
                    bits.append(f"{el.min():.0f}–{el.max():.0f}° up")
                return True, " · ".join(bits)
            return False, " · ".join(bits + ["lights not set"])
        if name == "Process":
            if not p.solved:
                return False, "Not solved yet"
            st = p.result.settings
            return True, (f"{st.resolved_normal_engine()} · {st.resolved_depth_engine()}"
                          + (f" · {p.solve_seconds:.1f} s" if p.solve_seconds else ""))
        if name == "Inspect":
            return bool(p.solved), (self.view or "No view yet") if p.solved else "After solving"
        if name == "Measure":
            mm = p.photo_scale()
            return bool(mm), (f"{mm:.4g} mm per pixel" if mm else "No scale set")
        if name == "Publish":
            d = p.last_export_dir
            return bool(d), (os.path.basename(d.rstrip(os.sep)) if d else "Nothing exported yet")
        return False, ""

    def _update_trail(self):
        """The steps before the current one, folded, at the top of the left column."""
        if not hasattr(self, "trail"):
            return
        cur = WORKSPACES.index(self.current) if self.current in WORKSPACES else 0
        colours = dict(ok=self.tokens.ok, text3=self.tokens.text3)
        for i, name in enumerate(WORKSPACES):
            hd = self.trail[name]
            if i < cur:
                done, text = self.step_summary(name)
                hd.set_state(done, text, colours)
                hd.show()
            else:
                hd.hide()

    # ── measuring (MET-001) ───────────────────────────────────────────
    def measure_scale(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("scale")

    def measure_apply_scale(self):
        self.workspaces["Measure"]._set_scale()

    def measure_distance(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("distance")

    def measure_zscale(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("zscale")

    def measure_apply_zscale(self):
        self.workspaces["Measure"]._set_zscale()

    def measure_profile(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("profile")

    def measure_ruler(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("ruler")

    def measure_clear_zscale(self):
        self.workspaces["Measure"].clear_zscale()

    def measure_volume(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("volume")

    def measure_stats(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("stats")

    def measure_compare(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("compare")

    def measure_accuracy(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].use("accuracy")

    def measure_zscale_lateral(self):
        self.go_workspace("Measure")
        self.workspaces["Measure"].z_from_lateral()

    # ── sessions (SES-001) ──────────────────────────────────────────────
    def open_session(self):
        if not self.leave_capture("open another session"):
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a Helio session", str(self.prefs.value("last_folder", "")),
            "Helio session (*.helio)")
        if path:
            self.load_session_file(path, asked=True)

    def load_session_file(self, path: str, asked: bool = False) -> bool:
        """Open a .helio written by either app."""
        from helio.io import session as S
        try:
            sess = S.read(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Could not open the session", str(e))
            return False
        if not asked and not self.leave_capture("open another session"):
            return False
        self.forget_capture()
        found = [q for q in sess.plate_paths if os.path.exists(q)]
        if len(found) >= 3:
            self.thumbs.load(found)
            self.project.set_images(found)
        else:
            self.thumbs.load([])
            self.project.set_images([])
        p = self.project
        # SES-003: the lights come back too, when they match the images.
        L = sess.lights()
        lights_back = L is not None and len(L) == len(p.images) >= 3
        if lights_back:
            p.set_lights(L, sess.light_source or "session")
        p.settings = sess.processing_settings(p.settings)
        if p.settings.ambient_frame:                 # moved with the capture: beside it or the session
            amb = S.locate(p.settings.ambient_frame, os.path.dirname(found[0]) if found else "",
                           os.path.dirname(os.path.abspath(path)))
            if amb != p.settings.ambient_frame:
                p.update_settings(ambient_frame=amb)
        if p.settings.flash_ref:                     # the grey patch the session was solved with
            from helio.core.flash import parse_box
            self.grey_ref = {"state": "marked", "box": parse_box(p.settings.flash_ref)}
        # Bumps ↔ holes is not offered here: the 3D model inverts (View › 3D
        # Model › Open Inverted, or Flip Z), and so does any image editor.
        # A classic session that had it on is solved the right way out.
        was_inverted = bool(p.settings.invert_relief)
        if was_inverted:
            from dataclasses import replace
            p.settings = replace(p.settings, invert_relief=False)
        p.set_scale_photo(float(sess.mm_per_px) if sess.mm_per_px else None)   # photographs' pixels
        from helio.application.heights import HEIGHT_BASIS
        p.z_mm_per_unit = float(sess.z_mm_per_unit) if sess.z_mm_per_unit else None
        height_dropped = bool(p.z_mm_per_unit) and getattr(sess, "height_basis", "") != HEIGHT_BASIS
        if height_dropped:
            # measured on the display depth, whose heights depend on each feature's
            # width: no factor turns it into mm per pixel of true height
            p.z_mm_per_unit = None
        for eng, params in (sess.engine_settings or {}).items():
            if eng in self.engine_settings:           # engines that still exist
                self.engine_settings[eng].update(params)
        if sess.render_engine in self.engine_settings:
            self.view = self.engine = sess.render_engine
        self.bookmarks = dict(sess.bookmarks or {})
        self._restore_kept(sess)
        p.capture_meta = {str(k): str(v) for k, v in (sess.capture_metadata or {}).items()}
        self._show_object_profile()
        self.session, self.session_path = sess, path
        p.session_name = os.path.splitext(os.path.basename(path))[0]
        missing = len(sess.plate_paths) - len(found)
        if not sess.plate_paths:
            note = "Session opened. It names no images: import them."
        elif missing:
            note = (f"Session opened. {missing} of {len(sess.plate_paths)} images are not where "
                    f"the session says; import them again.")
        elif lights_back:
            note = (f"Session opened with its {len(L)} light directions. Process when ready.")
        elif L is not None:
            note = (f"Session opened. Its {len(L)} light directions do not match the "
                    f"{len(p.images)} images found: find the sphere or import the .lp.")
        else:
            note = ("Session opened. This session has no light directions (older sessions "
                    "did not store them): find the sphere or import the .lp, then process.")
        if was_inverted:
            note += (" It was saved with Bumps ↔ holes on; Helio solves the surface the "
                     "right way out (View › 3D Model › Open Inverted in 3D turns it inside out).")
        if height_dropped:
            note += (" Its height scale was measured on the display depth, which is not to scale "
                     "(Helio now measures true heights from the normals): set it again in "
                     "Measure › Height scale.")
        p.notice = note
        self.setWindowFilePath(path)
        self.go_workspace("Capture")
        p.changed.emit()
        self._saved_signature = self._work_signature()
        return True

    def _restore_kept(self, sess):
        """The annotations, kept sections and sheet choices a session carries (``helio3`` in its extra keys)."""
        keep = (sess.extra or {}).get("helio3") or {}
        solve = {k: v for k, v in (keep.get("solve") or {}).items() if k in SOLVE_KEPT}
        if solve:                                     # the Helio 3 solve choices the classic keys lack
            from dataclasses import replace
            self.project.settings = replace(self.project.settings, **solve)
        prec = keep.get("precision") or {}
        self.project.z_from_scale = bool(prec.get("z_from_scale")) and bool(self.project.z_mm_per_unit)
        self.project.accuracy = prec.get("accuracy") or None
        wb = keep.get("white_balance") or {}
        gains = wb.get("gains")
        if gains and len(gains) == 3 and all(0.1 < float(g) < 10 for g in gains):
            self.project.white_balance = tuple(float(g) for g in gains)
            self.project.white_patch = tuple(wb["patch"]) if wb.get("patch") else None
        else:
            self.project.white_balance = self.project.white_patch = None
        ins = self.workspaces.get("Inspect")
        ann = getattr(ins, "annotator", None)
        a = keep.get("annotations") or {}
        if ann is not None:
            ann.anns = [dict(x, pts=[tuple(q) for q in x.get("pts", [])]) for x in a.get("anns", [])]
            if a.get("layers"):
                ann.layers = [dict(ly, visible=ly.get("visible", True)) for ly in a["layers"]]
                ann.layer = a.get("layer") if any(ly["name"] == a.get("layer") for ly in ann.layers) \
                    else ann.layers[0]["name"]
            if hasattr(ann, "notes") and a.get("notes"):
                ann.notes.setPlainText(a["notes"])
            ann.redraw()
        import numpy as np

        def back(p_):
            out = dict(p_)
            for k in ("dist", "z"):
                if k in out:
                    out[k] = np.asarray(out[k], float)
            for k in ("a_px", "b_px"):
                if k in out:
                    out[k] = tuple(out[k])
            out["series"] = [back(c) for c in out.get("series") or []] or None
            if out["series"] is None:
                del out["series"]
            return out
        self.sheet_sections = [(back(s_), None) for s_ in keep.get("sections", [])]
        if keep.get("sheet"):
            self.prefs.setValue("sheet/options", __import__("json").dumps(keep["sheet"]))

    def session_state(self):
        """This window as a Session: classic-only choices carried through unchanged."""
        from dataclasses import replace
        from helio.io import session as S
        p = self.project
        base = getattr(self, "session", None) or S.Session()
        sess = replace(
            base,
            object_id=base.object_id or p.name,
            plate_paths=list(p.images),
            last_image_folder=os.path.dirname(p.images[0]) if p.images else "",
            render_engine=self.engine or base.render_engine,
            bookmarks=dict(self.bookmarks),
            normal_engine=p.settings.normal_engine, depth_engine=p.settings.depth_engine,
            invert_relief=bool(p.settings.invert_relief),
            flat_correction=bool(p.settings.flat_correction),
            near_field=bool(p.settings.near_field),
            near_field_from=p.settings.near_field_from,
            photo_tone=p.settings.photo_tone,
            even_flash=bool(p.settings.even_flash), flash_ref=p.settings.flash_ref or "",
            even_flat=bool(p.settings.even_flat), flat_areas=p.settings.flat_areas or "",
            mm_per_px=p.photo_scale() if p.scale_mm_per_px else base.mm_per_px,
            work_scale=float(p.settings.work_scale), ambient_frame=p.settings.ambient_frame or "",
            z_mm_per_unit=p.z_mm_per_unit or None,
            height_basis=_height_basis() if p.z_mm_per_unit else "",
            engine_settings={e: dict(v) for e, v in self.engine_settings.items()},
            capture_metadata=dict(p.capture_meta) or dict(base.capture_metadata or {}))
        # What was found and kept travels too (annotations, the sections kept for the sheet, the sheet's choices),
        # under one key the older format ignores and writes back.
        ins = self.workspaces.get("Inspect")
        ann = getattr(ins, "annotator", None)
        keep = {}
        if ann is not None and ann.anns:
            keep["annotations"] = _plain(dict(
                anns=ann.anns, layers=ann.layers, layer=ann.layer,
                notes=ann.notes.toPlainText() if hasattr(ann, "notes") else ""))
        secs = [p_ for p_, of in getattr(self, "sheet_sections", []) if of is self.render_data or of is None]
        if secs:
            keep["sections"] = _plain(secs)
        if self.sheet_prefs():
            keep["sheet"] = _plain(self.sheet_prefs())
        if p.z_from_scale or p.accuracy:
            keep["precision"] = _plain(dict(z_from_scale=bool(p.z_from_scale), accuracy=p.accuracy))
        if p.white_balance:
            keep["white_balance"] = dict(gains=[float(v) for v in p.white_balance],
                                         patch=[float(v) for v in p.white_patch] if p.white_patch else None)
        from helio.domain.settings import ProcessingSettings
        dflt = ProcessingSettings()
        solve = {k: getattr(p.settings, k) for k in SOLVE_KEPT if getattr(p.settings, k) != getattr(dflt, k)}
        if solve:
            keep["solve"] = solve
        extra = {k: v for k, v in (sess.extra or {}).items() if k != "helio3"}
        if keep:
            extra["helio3"] = keep
        sess = replace(sess, extra=extra)
        # SES-003: measured lights travel with the session (an Uncalibrated PCA
        # solve's are an estimate and are not stored).
        # A light per frame, every one usable: a session cannot hold a gap.
        complete = (p.lights_ready() and not p.excluded)
        return S.with_lights(sess, p.lights if complete else None,
                             p.light_source or "measured")

    def save_session(self):
        if getattr(self, "session_path", ""):
            self.write_session(self.session_path)
        else:
            self.save_session_as()

    def save_session_as(self):
        from helio_qt.publish import file_stem
        start = os.path.join(str(self.prefs.value("last_folder", os.path.expanduser("~"))),
                             f"{file_stem(self.session_state().object_id)}.helio")
        path, _ = QFileDialog.getSaveFileName(self, "Save the session", start,
                                              "Helio session (*.helio)")
        if path:
            if not path.endswith(".helio"):
                path += ".helio"
            self.write_session(path)

    def write_session(self, path: str) -> str:
        from helio.io import session as S
        sess = self.session_state()
        S.write(path, sess)
        self.session, self.session_path = sess, path
        self.project.session_name = os.path.splitext(os.path.basename(path))[0]
        self.setWindowFilePath(path)
        self.project.notice = f"Saved {os.path.basename(path)}"
        self.project.changed.emit()
        self._saved_signature = self._work_signature()
        return path

    # ── views (VIEW-002) ─────────────────────────────────────────────────
    def set_view(self, name: str):
        ins = self.workspaces["Inspect"]
        if name not in ins.order and name not in (ins.DEPTH_MAP, ins.BACKLIT):
            return                       # not a view here (Terrain views stay in the classic app)
        self.view = name
        if name not in (ins.DEPTH_MAP, ins.BACKLIT):
            self.engine = name
        if self.current != "Inspect":
            self.go_workspace("Inspect")
        ins.set_view(name)
        self.update_bottom()

    def _step_view(self, d: int):
        order = self.workspaces["Inspect"].order
        i = order.index(self.view) if self.view in order else 0
        self.set_view(order[(i + d) % len(order)])

    def view_next(self):
        self._step_view(1)

    def view_prev(self):
        self._step_view(-1)

    def view_depth(self):
        """D: the depth map, and back to the view before it (_kbd_toggle_depth_view)."""
        ins = self.workspaces["Inspect"]
        self.set_view(self.engine if self.view == ins.DEPTH_MAP else ins.DEPTH_MAP)

    def view_clay(self):
        """C: clay (grey shading) on and off; from the depth map, the last view in clay."""
        ins = self.workspaces["Inspect"]
        if self.view in ins.SPECIAL or self.view is None:
            self.display.layer = "Shading"
            self.set_view(self.engine)
            return
        self.display.layer = ("Combined Colour" if self.display.layer == "Shading" else "Shading")
        self.set_view(self.view)

    def set_layer(self, layer: str):
        if layer == "Depth Map (Greyscale)":
            return self.view_depth()
        ins = self.workspaces["Inspect"]
        if self.view in ins.SPECIAL:
            self.view = self.engine
        self.display.layer = layer
        self.set_view(self.view or self.engine)

    def save_state(self):
        self.workspaces["Inspect"].save_state()

    def best_raking(self):
        self.workspaces["Inspect"].best_raking()

    def toggle_sweep(self):
        self.workspaces["Inspect"].toggle_sweep()

    def toggle_compare(self):
        ins = self.workspaces["Inspect"]
        self.compare_on = not self.compare_on
        if self.compare_on:
            ins.sec_compare.set_open(True)
        ins.chk_compare.blockSignals(True)
        ins.chk_compare.setChecked(self.compare_on)
        ins.chk_compare.blockSignals(False)
        ins.render_view(final=True)

    def _rotate(self, deg, keep_view: bool = False):
        """Turn the view by *deg* (None: back to 0).  Quarter turns refit the picture;
        a fine turn (*keep_view*) keeps the zoom and the point at the centre."""
        self.set_rotation_angle(0.0 if deg is None else self.canvas_rotation + deg, keep_view)

    def set_rotation_angle(self, deg: float, keep_view: bool = False):
        """The view's turn, any angle (display only), held to -180 < a <= 180 and 0.1°."""
        a = round(float(deg) % 360.0, 1) % 360.0
        if a > 180.0:
            a -= 360.0
        self.canvas_rotation = 0 if a == 0 else (int(a) if a == int(a) else a)
        ins = self.workspaces["Inspect"]
        ins.viewport.set_rotation(self.canvas_rotation, keep_view=keep_view)
        ins._rotation_shown(self.canvas_rotation)
        ins._update_badge()
        self.update_bottom()

    def rotate_left(self):
        self._rotate(-90)

    def rotate_right(self):
        self._rotate(90)

    def rotate_left_fine(self):
        self._rotate(-1, keep_view=True)

    def rotate_right_fine(self):
        self._rotate(1, keep_view=True)

    def rotate_reset(self):
        self._rotate(None)

    def reset_transforms(self):
        """The classic "Reset Transforms": no mirror, no turn, no inversion."""
        d = self.display
        d.flip_h = d.flip_v = d.rot180 = d.depth_invert = d.normal_invert = False
        self._rotate(None)
        self.set_view(self.view or self.engine)

    def show_histogram(self):
        ins = self.workspaces["Inspect"]
        ins.sec_hist.set_open(True)
        ins.scroll.ensureWidgetVisible(ins.sec_hist)

    def export_loupe(self):
        """The loupe's region at full resolution (export_magnifier_region)."""
        import cv2
        from helio.application.viewing import apply_transforms, screen_image, turn_image
        ins = self.workspaces["Inspect"]
        box = ins.loupe_box()
        if box is None:
            QMessageBox.information(self, "Nothing in the loupe",
                                    "Move over the image (or click to hold a spot) first.")
            return
        x1, y1, x2, y2 = box
        from helio_qt.publish import file_stem
        start = os.path.join(str(self.prefs.value("export_folder", "")) or self.default_export_dir(),
                             f"{file_stem(self.project.name)}_magnifier_{x1}_{y1}_{x2}_{y2}.tif")
        path, _ = QFileDialog.getSaveFileName(self, "Save the loupe region", start,
                                              "TIFF (*.tif *.tiff);;PNG (*.png)")
        if not path:
            return
        d = self.display
        st = dict(self.engine_settings.get(self.engine, {}))
        layer = d.layer
        if self.view == ins.DEPTH_MAP:
            import copy
            d = copy.deepcopy(d)
            d.layer = "Depth Map (Greyscale)"
        img = apply_transforms(screen_image(self.engine, self.render_data, st, d, use_proxy=False,
                                            crop=box, transforms=False), d)
        img = turn_image(img, self.canvas_rotation)           # turned as Inspect shows it
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        del layer
        self.flash(f"Saved {os.path.basename(path)} ({x2 - x1} × {y2 - y1} px).")

    def save_view(self, what: str = "", relief=None):
        """Save View (S, ⌘⌥S): the image at full resolution, the 3D relief, or
        both as the centre shows them, chosen in one window (SaveViewDialog).
        *what* picks the choice it opens on; by default, what the centre shows.
        *relief* is the high-detail window's 3D view, when it asks."""
        from helio_qt.record_dialog import SaveViewDialog
        ins = self.workspaces["Inspect"]
        mode = getattr(ins, "centre_mode", "image")
        if not what:
            what = "3d" if relief is not None else {"3d": "3d", "both": "centre"}.get(mode, "image")
        r = ins.relief
        can_3d = relief is not None or r is None or bool(getattr(r, "gl_ok", False))
        self.save_view_dialog = SaveViewDialog(self, relief, what, centre_both=mode == "both"
                                               and relief is None, can_3d=can_3d)
        self.save_view_dialog.show()
        return self.save_view_dialog

    def snapshot(self):
        """The image at full resolution: Save View, opened on The image."""
        return self.save_view("image")

    def write_view_tiff(self, path: str) -> str:
        """The current view at full resolution into *path*, an 8-bit TIFF with
        the capture details embedded (publish.current_view)."""
        from helio_qt import publish
        # It is written as an 8-bit TIFF, so it is named one: a name typed or
        # remembered as .jpg would have held TIFF data under the wrong name.
        root, ext = os.path.splitext(path)
        if ext.lower() not in (".tif", ".tiff"):
            path = root + ".tif"
        out = publish.current_view(self.publish_source(), os.path.dirname(os.path.abspath(path)))
        if os.path.abspath(out) != os.path.abspath(path):
            os.replace(out, path)
        publish.embed_meta(path, self.project.capture_meta, self.view or "")
        self.flash(f"Saved {os.path.basename(path)}.")
        return path

    def flash(self, text: str, seconds: float = 6.0):
        """A short message in the bottom bar's detail line."""
        self._flash = (text, time.time() + seconds)
        self.update_bottom()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(int(seconds * 1000) + 50, self.update_bottom)

    # ── undo (the classic settle-then-commit history) ─────────────────
    def undo_state(self):
        from helio.application.viewing import undo_state
        if self.render_data is None:
            return None
        s = self.project.settings
        return undo_state(self.engine, s.normal_engine, s.depth_engine, self.display.layer,
                          self.engine_settings.get(self.engine, {}))

    def _undo_poll(self):
        if not self.project.busy:
            self.history.observe(self.undo_state())

    def _apply_undo(self, st):
        if not st:
            return
        eng = st.get("render_engine") or self.engine
        self.engine_settings.setdefault(eng, {}).update(st.get("settings", {}))
        if st.get("normal_engine") != self.project.settings.normal_engine or \
                st.get("depth_engine") != self.project.settings.depth_engine:
            self.project.update_settings(normal_engine=st.get("normal_engine"),
                                         depth_engine=st.get("depth_engine"))
        if st.get("view_mode") and st["view_mode"] != "Depth Map (Greyscale)":
            self.display.layer = st["view_mode"]
        self.set_view(eng)
        self.history.settled(self.undo_state())

    def undo(self):
        self._apply_undo(self.history.undo(self.undo_state()))

    def redo(self):
        self._apply_undo(self.history.redo(self.undo_state()))

    def go_process(self):
        self.go_workspace("Process")

    def go_publish(self):
        self.go_workspace("Publish")

    def new_project(self):
        if not self.leave_capture("start a new project"):
            return
        self.forget_capture()
        self.thumbs.load([])
        self.project.set_images([])
        self.go_workspace("Capture")

    def import_images(self):
        if not self.leave_capture("open other photographs"):
            return
        start = str(self.prefs.value("last_folder", os.path.expanduser("~")))
        folder = QFileDialog.getExistingDirectory(self, "Choose the folder of photographs", start)
        if folder:
            self.open_paths([folder], asked=True)

    def start_next_capture(self):
        """File › Start Next Capture…: the binding is finished; the next one
        starts clean in this window, after saving this one if it needs it."""
        self.import_images()

    # ── one capture after another (SES-005) ──────────────────────────
    def _work_signature(self):
        """What a save would keep, and what it would not: changed since the
        last save or open means there is work to lose."""
        import json
        from helio.io import session as S
        ins, meas = self.workspaces["Inspect"], self.workspaces["Measure"]
        try:
            st = S.encode(self.session_state())
            # How the views are set is how you are looking, not work on the
            # capture: changing a slider is not a reason to ask to save.
            for k in ("engine_settings", "render_engine"):
                st.pop(k, None)
            sess = json.dumps(st, sort_keys=True, default=str)
        except Exception:
            sess = ""
        return (sess, id(self.project.result), len(getattr(ins.annotator, "anns", [])),
                len(meas.measured))

    def unsaved_work(self) -> bool:
        """A processed capture, or annotations or measurements, not saved as they are."""
        p, ins, meas = self.project, self.workspaces["Inspect"], self.workspaces["Measure"]
        if not p.images:
            return False
        worked = p.solved or bool(getattr(ins.annotator, "anns", [])) or bool(meas.measured)
        return worked and self._work_signature() != getattr(self, "_saved_signature", None)

    def leave_capture(self, what: str = "open the next capture") -> bool:
        """May this capture be put away?  Stops a running job and offers to
        save first; False when the person would rather stay."""
        p = self.project
        if p.busy:
            job = {"solve": "Processing", "export": "An export",
                   "sphere": "The sphere search"}.get(p.busy, "A job")
            r = QMessageBox.question(
                self, "Stop and move on?",
                f"{job} is still running on {p.name}. Stop it and {what}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel)
            if r != QMessageBox.StandardButton.Yes:
                return False
            self.process_cancel()
            t = time.time()
            while p.busy and time.time() - t < 60:
                QApplication.processEvents()
                time.sleep(0.02)
            if p.busy:
                return False
        if not self.unsaved_work():
            return True
        r = QMessageBox.question(
            self, "Save this capture first?",
            f"Save the session for {p.name} before you {what}? Its lights, settings, scale "
            f"and views are kept in the session; the photographs are not changed either way.",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel, QMessageBox.StandardButton.Save)
        if r == QMessageBox.StandardButton.Cancel:
            return False
        if r == QMessageBox.StandardButton.Save:
            self.save_session()
            return not self.unsaved_work()          # the save dialog cancelled: stay
        return True

    def forget_capture(self):
        """Everything that belongs to one capture, put away before the next.
        The rig's choices stay: solver, depth method, views and their
        settings, the 3D settings, the appearance."""
        from helio.application.viewing import History
        p = self.project
        self.session, self.session_path = None, ""
        self.setWindowFilePath("")
        p.session_name = ""
        p.white_balance = p.white_patch = None
        p.z_from_scale, p.accuracy = False, None
        self.bookmarks = {}
        self.sheet_shots = []                      # the publication sheet's 3D pictures
        self.sheet_sections = []                   # …its added sections: (profile, surface)
        p.clear_scale()
        p.z_mm_per_unit = None
        p.poi, p.poi_source, p.capture_meta = {}, "", {}
        self._show_object_profile()
        p.colour_check, p.export_rows, p.light_frame = None, {}, None
        p.solve_seconds = 0.0
        p.result = None
        self.selection, self._sel_cache = None, None
        self.backlit = {}
        self.grey_ref = {"state": "none"}             # the card was in the last photographs
        if p.settings.flash_ref or p.settings.even_flash:
            p.update_settings(even_flash=False, flash_ref="")
        if p.settings.flat_areas or p.settings.even_flat:
            p.update_settings(even_flat=False, flat_areas="")     # boxes on the last photographs
        if p.settings.ambient_frame:
            p.update_settings(ambient_frame="")                    # the last capture's lights-off frame
        self.reference_depth, self.reference_name = None, ""
        self.solve_crop = self.solve_crop_ref = None
        self.render_data, self._render_of = None, None
        self.history = History()
        self.workspaces["Capture"].cancel_marking()
        self.model3d.close()
        if getattr(self, "hd3d", None) is not None:
            self.hd3d.close()                    # another capture: its surface is gone
        ins, meas = self.workspaces["Inspect"], self.workspaces["Measure"]
        if hasattr(ins, "annotator"):
            ins.annotator.clear()
        meas.z_diff, meas.profile, meas.drawn_px = 0.0, None, 0.0
        meas.measured, meas.region_rows, meas.compare_boxes = [], [], []
        meas.lines = []
        meas.compare_stats, meas._shown_key = None, None
        self._saved_signature = None

    def open_paths(self, paths, asked: bool = False) -> bool:
        from helio.io.capture import captures_in
        imgs = captures_in(paths)
        if len(imgs) < 3:
            QMessageBox.warning(self, "Need at least 3 images",
                                f"Photometric stereo needs 3 or more photographs, each lit from a "
                                f"different direction. That had {len(imgs)}.")
            return False
        if not asked and not self.leave_capture():
            return False
        self.forget_capture()
        self.prefs.setValue("last_folder", os.path.dirname(imgs[0]))
        self.thumbs.load(imgs)
        self.project.set_images(imgs)
        self.backlit = {}
        self.go_workspace("Capture")
        self.auto_light_file()
        # The drift check runs at import, the last moment re-shooting is cheap.
        self.check_drift(quiet=True)
        self.find_grey_reference()
        return True

    # ── hand-held flash: the grey reference (CAL-003) ─────────────────
    def find_grey_reference(self):
        """Look for a flat grey card in the frames, off the main thread; when
        one is found, Process evens out the flash strength with it."""
        from helio.core.flash import box_text, find_grey_reference
        imgs = list(self.project.images)
        self.grey_ref = {"state": "looking"}
        # A box from the last capture means nothing on these photographs.
        if self.project.settings.flash_ref:
            self.project.update_settings(even_flash=False, flash_ref="")

        def done(res, err):
            if imgs != list(self.project.images):
                return                                # another capture was opened since
            if err or not res:
                self.grey_ref = {"state": "none"}
            else:
                self.grey_ref = dict(res, state="found")
                self.project.update_settings(even_flash=True, flash_ref=box_text(res["box"]))
            self.refresh()
        self.run_background(lambda: find_grey_reference(imgs), done)

    # ── flat areas: the flash's fall-off across the frame (CAL-004) ──
    def mark_flat_area(self):
        """Drag a box over an area known to be flat; each one adds to the list."""
        if not self.project.images:
            return
        self.pick_rect(self._flat_drawn, "Drag a box over an area you know is flat and plain, "
                       "lit by every flash: the board round the object, a margin, the card.",
                       photo=True)

    def _flat_drawn(self, x1, y1, x2, y2):
        from helio.core.flash import boxes_text, check_reference, parse_boxes
        box = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
        if box[2] - box[0] < 12 or box[3] - box[1] < 12:
            self.flash("That box is too small: draw over more of the flat area.")
            return
        imgs = list(self.project.images)

        def done(res, err):
            if imgs != list(self.project.images):
                return
            if err or res is None or not res.get("ok"):
                why = (res or {}).get("problem") or "it could not be read"
                self.flash(f"Not used: {why}. Mark an area that is lit, never white or black, "
                           f"in every frame.", 12)
                return
            boxes = parse_boxes(self.project.settings.flat_areas) + [box]
            self.project.update_settings(even_flat=True, flat_areas=boxes_text(boxes))
            self.flash(f"Flat area marked ({len(boxes)} in all): the flash's fall-off is evened "
                       f"when you process." + (" Mark another across the object for the best."
                                               if len(boxes) == 1 else ""), 10)
            self.refresh()
        from helio.core.flash import FLAT_MIN_LEVEL
        self.run_background(lambda: check_reference(imgs, box, min_level=FLAT_MIN_LEVEL), done)

    def clear_flat_areas(self):
        # The boxes only: Lamp strengths stays on Flat areas, ready for new ones.
        self.project.update_settings(flat_areas="")
        self.flash("Flat areas cleared.")
        self.refresh()

    # ── one box on the picture, then back (the grey patch, a region) ──
    def pick_rect(self, done, message: str, back_to: str = "Process", photo: bool = False):
        """Inspect, the box tool on, until one box is drawn: *done(x1, y1,
        x2, y2)* in scene pixels, then back to *back_to*.  Esc, or leaving
        Inspect, gives it up; a second pick replaces the first."""
        self.cancel_pick(go_back=False)
        ins = self.workspaces["Inspect"]
        self.go_workspace("Inspect")
        if photo and ins.mode != "frame":
            ins._show_frame(self.project.current)
        vp = ins.viewport

        def drawn(x1, y1, x2, y2):
            self.cancel_pick(go_back=False)
            self.go_workspace(back_to)
            done(x1, y1, x2, y2)

        def escaped():
            self.cancel_pick()
            self.flash("Cancelled.")
        self._pick = (drawn, escaped, back_to)
        vp.rectDrawn.connect(drawn)
        vp.escaped.connect(escaped)
        vp.set_tool("rect")
        vp.setFocus()
        self.flash(message + " Esc to cancel.", 30)

    def cancel_pick(self, go_back: bool = True):
        pick = getattr(self, "_pick", None)
        if not pick:
            return
        self._pick = None
        vp = self.workspaces["Inspect"].viewport
        for sig, fn in ((vp.rectDrawn, pick[0]), (vp.escaped, pick[1])):
            try:
                sig.disconnect(fn)
            except (RuntimeError, TypeError):
                pass
        vp.set_tool("")
        vp.clear_rect()
        if go_back:
            self.go_workspace(pick[2])

    def mark_grey_reference(self):
        """Drag a box over a flat grey patch on a photograph; back in Process
        the patch is checked in every frame (helio.core.flash.check_reference)."""
        if not self.project.images:
            return
        self.pick_rect(self._grey_drawn, "Drag a box over a flat, matte grey patch lit by "
                       "every flash: a grey card beside the object.", photo=True)

    def _grey_drawn(self, x1, y1, x2, y2):
        from helio.core.flash import box_text, check_reference
        box = (int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2)))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            self.flash("That box is too small: draw over the grey patch itself.")
            return
        old = (self.grey_ref, self.project.settings.even_flash, self.project.settings.flash_ref)
        imgs, L = list(self.project.images), self.project.lights
        self.grey_ref = {"state": "checking", "box": box}
        self.flash("Checking the grey patch in every frame…", 10)
        self.refresh()

        def done(res, err):
            if imgs != list(self.project.images):
                return                                # another capture was opened since
            if err or res is None or not res.get("ok"):
                why = (res or {}).get("problem") or "it could not be read"
                self.grey_ref = old[0]                # the earlier reference stands
                self.project.update_settings(even_flash=old[1], flash_ref=old[2])
                self.flash(f"Not used: {why}. Mark a flat, matte grey patch that is never "
                           f"white or black in any frame.", 12)
            else:
                self.grey_ref = dict(res, state="marked", box=box)
                self.project.update_settings(even_flash=True, flash_ref=box_text(box))
                said = (f" The flash varied ×{res['spread']:.1f}." if res.get("spread") else "")
                warn = f" {res['warning']}" if res.get("warning") else ""
                self.flash("Grey patch marked: the flash strength is evened when you process."
                           + said + warn, 12)
            self.refresh()
        self.run_background(lambda: check_reference(imgs, box, L), done)

    def import_light_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a light-position file", str(self.prefs.value("last_folder", "")),
            "Light files (*.lp *.dome *.txt);;All files (*)")
        if path:
            self.load_light_file(path)

    def check_light_frame(self):
        """Is north at the top for these lights? (capture_tools.light_frame_check)

        Runs in the background whenever lights arrive; Capture shows a
        notice with Swap north and south when they read upside down."""
        from helio.application.capture_tools import light_frame_check
        p = self.project
        L, paths = p.lights, list(p.images)
        if L is None or len(L) != len(paths) or len(paths) < 3:
            return
        balls = [tuple(p.sphere)] if isinstance(p.sphere, tuple) else []
        token = object()
        self._light_frame_token = token

        def done(res, err):
            if err or getattr(self, "_light_frame_token", None) is not token or p.lights is not L:
                return
            p.light_frame = res
            p.changed.emit()
        self.run_background(lambda: light_frame_check(paths, L, balls=balls), done)

    def swap_lights_north_south(self):
        from helio.application.capture_tools import swap_north_south
        p = self.project
        if p.lights is None:
            return
        src = p.light_source if "north–south swapped" in p.light_source else \
            f"{p.light_source} (north–south swapped)"
        p.set_lights(swap_north_south(p.lights), src, sphere=p.sphere, calibration=p.calibration)
        self.check_light_frame()
        self.flash("North and south swapped in the lights. Process again to use them.")

    def auto_light_file(self) -> str:
        """A light file beside the photographs (helio.io.lp.find_light_file),
        loaded as Import Light Positions would: by name, else in shot order.
        Returns its path, or "" when there is none or the choice is the user's."""
        from helio.io.lp import find_light_file
        p = self.project
        if not p.images:
            return ""
        try:
            path, why = find_light_file(p.images)
        except Exception:                                  # noqa: BLE001 - a convenience, never a failure
            return ""
        if not path:
            if why:
                self.flash(why, 12)
            return ""
        if not self.load_light_file(path):
            return ""
        import numpy as np
        n = int(np.count_nonzero(np.isfinite(np.asarray(p.lights, np.float32)).all(axis=1)))
        self.flash(f"Light file found beside the photographs and loaded: {os.path.basename(path)} "
                   f"({n} lights; {why}). Import another from File › Import Light Positions… "
                   "if it is not this capture's.", 12)
        return path

    def load_light_file(self, path: str) -> bool:
        from helio.io.lp import read_lp_named
        try:
            names, lights = read_lp_named(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Could not read the light file", str(e))
            return False
        lights = self._paired_lights(names, lights)
        from helio.application.capture_tools import calibration_spheres
        b1, b2 = calibration_spheres(path)
        self.project.set_lights(lights, f"file:{os.path.basename(path)}", sphere=b1, sphere2=b2)
        self.check_light_frame()
        return True

    def _paired_lights(self, names, lights):
        """A light file's lights in the order of the photographs: by name when
        its names are theirs, else in shot (natural) order, never alphabetical
        (helio.io.lp.pair_lights).  Says so when the names matched none."""
        from helio.io.lp import pair_lights
        if not self.project.images:
            return lights
        lights, _how, warn = pair_lights(names, lights, self.project.images)
        if warn:
            self.flash(warn, 15)
        return lights

    # ── change over time and the multi-light composite ────────────────
    def load_change_reference(self):
        from helio.application.export_reports import load_reference_depth
        path, _ = QFileDialog.getOpenFileName(self, "An earlier depth map of the same object",
                                              str(self.prefs.value("last_folder", "")),
                                              "Depth (*.png *.tif *.tiff *.npy)")
        if not path:
            return
        try:
            self.reference_depth = load_reference_depth(path)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Change reference", str(e))
            return
        self.reference_name = os.path.basename(path)
        self.flash(f"Change reference loaded: {self.reference_name}. Save Change Map compares "
                   "this surface with it.")

    def _need_reference(self) -> bool:
        if getattr(self, "reference_depth", None) is None:
            QMessageBox.information(self, "Change map", "Load an earlier depth map of the same "
                                    "object first: Measure › Load Change Reference.")
            return False
        return True

    def save_change_map(self):
        from helio.application.export_reports import change_map
        from helio_qt.publish import file_stem
        if not self._need_reference():
            return
        p = self.project
        start = os.path.join(str(self.prefs.value("export_folder", "")) or self.default_export_dir(),
                             f"{file_stem(p.name)}_change.tif")
        path, _ = QFileDialog.getSaveFileName(self, "Save the change map", start, "GeoTIFF (*.tif)")
        if not path:
            return
        r = change_map(p.result.depth_vis_p, self.reference_depth, path)
        self.flash(f"Change map saved. Noise floor {r['noise']:.4g} {r['unit']} (robust); "
                   f"{r['moved']:,} pixels ({r['moved_pct']:.2f}%) moved beyond 3× that. "
                   "Positive is higher than the reference.", 12)

    def save_change_report(self):
        from helio.application.export_reports import change_detection
        from helio_qt.publish import file_stem
        if not self._need_reference():
            return
        folder = QFileDialog.getExistingDirectory(self, "Folder for the change report",
                                                  str(self.prefs.value("export_folder", "")) or
                                                  self.default_export_dir())
        if not folder:
            return
        r = self.project.result
        files = change_detection(r.depth_poisson, self.reference_depth, r.albedo_rgb_full, folder,
                                 file_stem(self.project.name),
                                 operator=self.project.capture_meta.get("operator", ""),
                                 baseline_name=getattr(self, "reference_name", ""))
        self.flash(f"Change report saved: {os.path.basename(files[0])} and its statistics.")

    def open_research(self, kind: str):
        """Research › a tool window (helio_qt.research): one per kind, kept."""
        from helio_qt.research import RESEARCH
        if not hasattr(self, "research_dialogs"):
            self.research_dialogs = {}
        dlg = self.research_dialogs.get(kind)
        if dlg is None:
            dlg = self.research_dialogs[kind] = RESEARCH[kind](self)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return dlg

    def open_ai(self, task: str):
        """Research › AI: identify, condition report or caption (helio.application.ai)."""
        from helio_qt.research import AIDialog
        if getattr(self, "ai_dialog", None) is None:
            self.ai_dialog = AIDialog(self, task)
        else:
            self.ai_dialog.task.set_current(task)
            self.ai_dialog._task_changed(task)
        self.ai_dialog.show()
        self.ai_dialog.raise_()
        return self.ai_dialog

    def multi_light_composite(self):
        from PySide6.QtWidgets import QInputDialog
        from helio.application.export_images import COMPOSITE_MODES, multi_light_composite
        from helio.application.render import composite, render
        from helio_qt.publish import file_stem
        p, D = self.project, self.render_data
        L = p.result.light_matrix if p.result is not None else None
        if L is None or len(L) < 2:
            QMessageBox.information(self, "Multi-light composite",
                                    "At least 2 calibrated light directions are needed.")
            return
        mode, ok = QInputDialog.getItem(self, "Multi-light composite", "Combine the relit views by",
                                        list(COMPOSITE_MODES), 0, False)
        if not ok:
            return
        start = os.path.join(str(self.prefs.value("export_folder", "")) or self.default_export_dir(),
                             f"{file_stem(p.name)}_composite_{mode.split()[0].lower()}.tif")
        path, _ = QFileDialog.getSaveFileName(self, "Save the composite", start,
                                              "TIFF (*.tif);;PNG (*.png)")
        if not path:
            return
        eng = self.engine
        base = dict(self.engine_settings.get(eng, {}))
        layer = self.display.layer if self.display.layer in ("Shading", "Combined Colour",
                                                             "Roughness Fusion") else "Combined Colour"

        import numpy as np

        def draw(lx, ly, lz):
            st = dict(base, lx=lx, ly=ly, lz=lz)
            img = render(eng, D, st, D.ph, D.pw, True)
            return composite(eng, img, D, st, *np.asarray(img).shape[:2], True, view_mode=layer)

        def done(res, err):
            if err or res is None:
                QMessageBox.warning(self, "Multi-light composite", str(err))
                return
            from PIL import Image
            Image.fromarray(res[0]).save(path)
            self.flash(f"Composite saved ({mode}, {res[1]} lights): {os.path.basename(path)}.")
        self.flash(f"Relighting {len(L)} times…", 20)
        self.run_background(lambda: multi_light_composite(draw, L, mode), done)

    # ── sharing (helio_qt.dialogs) ───────────────────────────────────
    def show_qr(self):
        from helio_qt.dialogs import QrDialog
        self.qr_dialog = QrDialog(self)
        self.qr_dialog.exec()

    def show_batch(self):
        from helio_qt.batch import BatchDialog
        self.batch_dialog = BatchDialog(self)
        self.batch_dialog.show()
        return self.batch_dialog

    def show_facsimile(self):
        from helio_qt.dialogs import FacsimileDialog
        self.facsimile_dialog = FacsimileDialog(self)
        self.facsimile_dialog.show()
        return self.facsimile_dialog

    def show_sketchfab(self):
        from helio_qt.dialogs import SketchfabDialog
        self.sketchfab_dialog = SketchfabDialog(self)
        self.sketchfab_dialog.show()
        return self.sketchfab_dialog

    def show_embed(self):
        from helio_qt.dialogs import EmbedDialog
        EmbedDialog(self).exec()

    def write_manifest(self):
        """Every file in a folder with its SHA-256 (helio.io.manifest), for deposit."""
        from helio.io.manifest import write_manifest
        folder = QFileDialog.getExistingDirectory(self, "Folder to write a manifest for",
                                                  str(self.prefs.value("export_folder", "")))
        if not folder:
            return
        path = write_manifest(folder, object_id=self.project.name if self.project.images else "",
                              note="written from Helio")
        n = sum(len(f) for _r, _d, f in os.walk(folder)) - 1
        self.flash(f"Saved {os.path.basename(path)}: {n} file(s) with a SHA-256 each. "
                   "Running it again is safe and overwrites it.")

    def verify_manifest(self):
        """Re-hash a deposited folder against its MANIFEST.txt."""
        from helio.io.manifest import verify_manifest
        folder = QFileDialog.getExistingDirectory(self, "Folder with a MANIFEST.txt to verify",
                                                  str(self.prefs.value("export_folder", "")))
        if not folder:
            return
        try:
            r = verify_manifest(folder)
        except FileNotFoundError as e:
            QMessageBox.warning(self, "Verify manifest", str(e))
            return

        def sample(lst, n=6):
            return "\n".join("    " + x for x in lst[:n]) + (f"\n    …and {len(lst) - n} more"
                                                             if len(lst) > n else "")
        bad = len(r["changed"]) + len(r["missing"])
        if not bad and not r["extra"]:
            QMessageBox.information(self, "Verify manifest",
                                    f"All {r['total']} file(s) match: {os.path.basename(folder)} is "
                                    "byte for byte what the manifest describes.")
            return
        lines = [f"{len(r['ok'])} of {r['total']} file(s) match.", ""]
        if r["changed"]:
            lines += [f"CHANGED ({len(r['changed'])}): content differs:", sample(r["changed"]), ""]
        if r["missing"]:
            lines += [f"MISSING ({len(r['missing'])}):", sample(r["missing"]), ""]
        if r["extra"]:
            lines += [f"NOT IN THE MANIFEST ({len(r['extra'])}):", sample(r["extra"]), ""]
        (QMessageBox.critical if bad else QMessageBox.warning)(self, "Verify manifest",
                                                               "\n".join(lines).strip())

    # ── the 3D model (helio_qt.model3d) ──────────────────────────────
    def open_3d(self, kind: str = "standard"):
        """The 3D relief in the window (⌘⇧3): as captured, flattened or
        inverted; the heatmap is its height colouring."""
        ins = self.workspaces["Inspect"]
        if self.current != "Inspect":
            self.go_workspace("Inspect")
        ins.set_centre_mode("3d")
        if ins.relief is None:
            return
        kind = kind or "standard"
        ins.relief_kind.set_current("standard" if kind == "heatmap" else kind)
        ins.relief_material.set_current("height" if kind == "heatmap" else "colour")
        ins.relief.set_material("height" if kind == "heatmap" else "colour")
        ins._relief_shape_changed()

    def open_3d_window(self):
        """The high-detail 3D window (helio_qt.relief_window): Helio's own 3D
        view built from up to 9 million points, following the 3D panel."""
        from helio_qt.relief_window import HighDetail3D
        if self.project.result is None:
            self.flash("Process the images first.")
            return
        if getattr(self, "hd3d", None) is None:
            self.hd3d = HighDetail3D(self)
        self.hd3d.show()
        self.hd3d.raise_()
        self.hd3d.activateWindow()
        self.hd3d.view.setFocus(Qt.FocusReason.ActiveWindowFocusReason)   # its keys work at once
        if self.hd3d.view.data is None or self.hd3d._of is not self.selected_result()[1]:
            self.hd3d.load()
        self.workspaces["Inspect"].update_relief_panel()

    def show_3d_settings(self):
        self.model3d.show_settings()

    def place_by_hand(self):
        """The Virtual Dome Calibrator: Capture's "By hand" page (helio_qt.hand_dome)."""
        self.go_workspace("Capture")
        self.workspaces["Capture"].hand.start()

    def import_rti_project(self):
        """An RTIBuilder / Relight project folder: its .lp matched to its
        photographs by name (helio.application.dome_tools.read_rti_project),
        then processed, as the classic import does."""
        from helio.application.dome_tools import read_rti_project
        start = str(self.prefs.value("last_folder", os.path.expanduser("~")))
        folder = QFileDialog.getExistingDirectory(self, "Choose the RTIBuilder or Relight project "
                                                  "folder", start)
        if not folder:
            return
        try:
            r = read_rti_project(folder)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "Could not import the RTI project", str(e))
            return
        if r["unmatched"] and r.get("paired") == "order":
            ans = QMessageBox.question(
                self, "The light file's names are not the photographs'",
                f"No name in {os.path.basename(r['lp'])} is one of the photographs', so its "
                f"{r['entries']} lights are paired with the {len(r['paths'])} photographs in "
                f"shot order: light 1 with {os.path.basename(r['paths'][0])}, and so on.\n\n"
                + "\n".join(str(n).strip("()") for n in r["unmatched"])
                + "\n\nContinue if that is the order they were taken in.")
            if ans != QMessageBox.StandardButton.Yes:
                return
        elif r["unmatched"]:
            un = r["unmatched"]
            ans = QMessageBox.question(
                self, "Some names did not match",
                f"Matched {len(r['paths'])} of {r['entries']} entries in "
                f"{os.path.basename(r['lp'])} to photographs.\n\nNot found: "
                + ", ".join(os.path.basename(str(n)) for n in un[:6]) + (" …" if len(un) > 6 else "")
                + "\n\nContinue with the matched photographs?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.prefs.setValue("last_folder", folder)
        self.thumbs.load(r["paths"])
        self.project.set_images(r["paths"])
        self.backlit = {}
        self.project.set_lights(r["lights"], f"file:{os.path.basename(r['lp'])}")
        self.check_light_frame()
        self.go_workspace("Capture")
        self.flash(f"RTI project: {len(r['paths'])} photographs with their lights from "
                   f"{os.path.basename(r['lp'])}.")
        self.registry.trigger("process.run")

    def find_sphere(self):
        self.project.set_busy("sphere")
        self.pool.start(_SphereJob(self.project.images, self._sphere_sig))

    def _sphere_done(self, cal, cands, error):
        p = self.project
        p.busy = ""
        p.sphere_candidates = list(cands or [])
        # Lights from a file, a rig, a session or placed by hand are kept when
        # no ball is found: the search used to set them to none, and the solve
        # made with them went too.
        kept = p.lights_ready() and not p.light_source.startswith("sphere")
        if error or cal is None:
            if kept:
                src = p.light_source.replace("file:", "").replace("rig:", "rig ")
                self.flash(f"No chrome ball found{': ' + error if error else ''}. The lights from "
                           f"{src} are kept.", 10)
                p.changed.emit()
            else:
                p.set_lights(None, "", sphere=False)
                if error:
                    QMessageBox.warning(self, "Sphere search failed", error)
        else:
            self.apply_calibration(cal)

    def apply_calibration(self, cal):
        """Use a sphere calibration's lights: one per frame, unusable ones left out."""
        self.project.set_lights(cal.lights, "sphere", sphere=cal.sphere.as_tuple(),
                                calibration=cal)

    # ── processing (PROC-001) ─────────────────────────────────────────
    def solve_request(self):
        from helio.application.pipeline import SolveRequest
        p = self.project
        pca = p.settings.resolved_normal_engine() == "Uncalibrated PCA"
        return SolveRequest(paths=p.solve_images(),
                            lights=None if (pca and not p.lights_ready()) else p.solve_lights(),
                            settings=p.settings, crop=self.solve_crop,
                            crop_ref_size=self.solve_crop_ref,
                            full_size_region=self.solve_crop is not None,
                            flat_ref_normals=self.flat_ref_normals,
                            depth_clip_pct=float(self.shared.get("depth_clip_pct", 0.5)),
                            mm_per_px=p.photo_scale(), light_origin=self.light_origin(),
                            light_positions=self.light_positions())

    def light_positions(self):
        """Each solved light's position from two balls (helio.core.light_positions), one row
        per frame solved, or None (one ball, a light file, lights placed by hand)."""
        import numpy as np
        p = self.project
        cal = p.calibration
        if cal is None or getattr(cal, "sphere2", None) is None or p.lights is None:
            return None
        if getattr(cal, "_positions", None) is None:
            from helio.core.light_positions import positions_from_two_balls
            r = positions_from_two_balls(cal)
            cal._positions = False if r is None or not r["placed"] else r["positions"]
        pos = cal._positions
        if pos is False or len(pos) != len(p.lights):
            return None
        return np.asarray(pos, np.float64)[p.used_frames()]

    def light_origin(self):
        """Where the lights were measured, in the photographs' pixels: the
        chrome ball's centre, or between the two balls; None when the lights
        came from elsewhere (a light file, by hand)."""
        p = self.project
        if not isinstance(p.sphere, tuple):
            return None
        x, y = float(p.sphere[0]), float(p.sphere[1])
        b2 = getattr(p, "sphere2", None)
        if b2 is not None:
            bx, by = (b2.cx, b2.cy) if hasattr(b2, "cx") else (b2[0], b2[1])
            x, y = (x + float(bx)) / 2.0, (y + float(by)) / 2.0
        return (x, y)

    def _ball_tuples(self):
        """The ball(s) the lights were measured on, as (cx, cy, r), for the files."""
        p = self.project
        b1 = p.sphere if isinstance(p.sphere, tuple) else None
        b2 = getattr(p, "sphere2", None)
        if b2 is not None and hasattr(b2, "as_tuple"):
            b2 = b2.as_tuple()
        return b1, b2

    def process_run(self):
        from helio.application import pipeline
        req = self.solve_request()
        self.project.set_busy("solve")
        self._solve_started = time.time()
        sig = self._job_sig
        self.jobs.submit(
            "solve", lambda ctx: pipeline.solve(req, ctx.progress, ctx.check),
            on_progress=lambda job, st, f, d: sig.progress.emit(job.id, st or "", f, d or ""),
            on_done=sig.done.emit, on_failed=sig.failed.emit,
            on_cancelled=sig.cancelled.emit, on_stale=sig.stale.emit)
        self.go_workspace("Process")

    def process_cancel(self):
        self.jobs.cancel(self.project.busy or None)
        self.project.set_progress("Stopping at the next step…", self.project.fraction)

    # ── exports (UX-005 on EXP-001) ───────────────────────────────────
    def default_export_dir(self) -> str:
        from helio_qt.publish import file_stem
        return os.path.join(os.path.expanduser("~/Pictures/Helio exports"),
                            file_stem(self.project.name))

    def publish_run(self):
        keys = self.workspaces["Publish"].chosen()
        if not keys:
            QMessageBox.information(self, "Nothing selected",
                                    "Tick one or more formats first.")
            return
        start = str(self.prefs.value("export_folder", "")) or self.default_export_dir()
        folder = QFileDialog.getExistingDirectory(self, "Export to which folder?",
                                                  os.path.dirname(start) if not os.path.isdir(start)
                                                  else start)
        if folder:
            self.publish_to(folder, keys)

    def export_one_format(self, key: str, label: str):
        """Inspect's one-click exports: ask for a folder, then send that one format to Publish's queue."""
        start = str(self.prefs.value("export_folder", "")) or self.default_export_dir()
        folder = QFileDialog.getExistingDirectory(self, f"{label}: export to which folder?",
                                                  os.path.dirname(start) if not os.path.isdir(start)
                                                  else start)
        if folder:
            self.publish_to(folder, [key], open_when_done=True)

    def export_rti_combined(self):
        self.export_one_format("rti_combined", "Combined RTI viewer")

    def export_rti_3d(self):
        self.export_one_format("rti_3d", "Relight and 3D viewer")

    def export_hf_viewer(self):
        self.export_one_format("hf_viewer", "3D viewer: fine detail")

    def publish_to(self, folder: str, keys=None, open_when_done: bool = False):
        from helio_qt import publish
        p = self.project
        keys = list(keys if keys is not None else self.workspaces["Publish"].chosen())
        src = self.publish_source()
        self.prefs.setValue("export_folder", folder)
        p.last_export_dir = folder
        p.set_export_rows({k: ("waiting", "", []) for k in keys})
        self._open_when_done = tuple(keys) if open_when_done else ()
        p.set_busy("export")
        sig = self._export_sig
        self.jobs.submit(
            "export", publish.job(keys, src, folder),
            on_progress=lambda job, st, f, d: sig.progress.emit(job.id, st or "", f, d or ""),
            on_done=sig.done.emit, on_failed=sig.failed.emit,
            on_cancelled=sig.cancelled.emit, on_stale=sig.stale.emit)
        self.go_workspace("Publish")

    # ── the selection (the classic's crop box) ────────────────────────
    def set_selection(self, box):
        """Inspect's selection, (x1, y1, x2, y2) in the solve's pixels, or None."""
        r = self.project.result
        if box is not None and r is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(r.w, x2), min(r.h, y2)
            box = (x1, y1, x2, y2) if x2 - x1 >= 16 and y2 - y1 >= 16 else None
        else:
            box = None
        if box != self.selection:
            self.selection, self._sel_cache = box, None
            self.flash(f"Selection {box[2] - box[0]} × {box[3] - box[1]} px: the 3D relief, "
                       f"Publish and the Research tools use this part alone. Clear it to go "
                       f"back to the whole." if box else "Selection cleared: the whole object again.", 8)
            ins = self.workspaces["Inspect"]
            if ins.relief is not None and ins.relief.isVisible():
                ins._push_relief()                    # the 3D shows the new part now
        self.refresh()

    def selected_result(self):
        """(result, render data) for the selection, or the whole solve's."""
        r, D = self.project.result, self.render_data
        sel = self.selection
        if r is None or not sel or sel[2] > r.w or sel[3] > r.h:
            return r, D
        # Held by the result itself, not its id(): a freed result's id can
        # come back on the next solve, with the old cut still cached.
        c = self._sel_cache
        if c is None or c[0][0] is not r or c[0][1] != sel:
            from helio.application.pipeline import crop_result, store
            from helio.data_manager import DataManager
            r2 = crop_result(r, sel)
            D2 = DataManager()
            store(r2, D2)
            # the colour views (Colour ICA, R-B difference ...) reread the photographs, at
            # the selection's part of them (solve_region): without the list they drew black
            D2.plate_paths = list(getattr(D, "plate_paths", None) or self.project.images)
            D2.engine_settings = self.engine_settings
            self._sel_cache = ((r, sel), (r2, D2))
        return self._sel_cache[1]

    def publish_source(self):
        """What the exports read: the solve, its lights, and the view as Inspect shows it;
        the selection alone when there is one."""
        import copy
        from helio_qt import publish
        p = self.project
        r, data = self.selected_result()
        if self.shared.get("export_colour") == "matte" and data is not None:
            r, data = self._matte_coloured(r, data)
        estimated = p.lights is None and r.light_matrix is not None
        ins = self.workspaces["Inspect"]
        return publish.Source(result=r, lights=r.light_matrix if estimated else p.solve_lights(),
                              light_source="estimated" if estimated else
                              ("measured" if p.lights is not None else ""),
                              images=p.solve_images(), name=p.name,
                              mm_per_px=float(p.scale_mm_per_px or 0.0),
                              z_mm_per_unit=float(p.z_mm_per_unit or 0.0),
                              shared=dict(self.shared),
                              data=data,
                              view=ins.DEPTH_MAP if self.view == ins.DEPTH_MAP else self.engine,
                              view_settings=dict(self.engine_settings.get(self.engine, {})),
                              display=copy.deepcopy(self.display),
                              meta=dict(p.capture_meta), poi=dict(p.poi),
                              poi_source=p.poi_source,
                              engine_settings=copy.deepcopy(self.engine_settings),
                              tone=r.settings.photo_tone,
                              compare_view=getattr(self, "compare_view", "") or "",
                              profile=self._profile_tuple(), facts=self._plate_facts(),
                              white_balance=p.white_balance,
                              turn=float(self.canvas_rotation or 0.0))

    def _matte_coloured(self, r, data):
        """Copies of the solve and its render data whose colour is the Matte Relight
        view's, at its light and the Colour layer's brightness (Publish's "Colour in
        the exports").  The solve itself keeps its colour."""
        import copy
        from helio.application.render import MATTE_VIEW, matte_colour
        from helio.domain.registry import default_engine_settings
        st = dict(self.engine_settings.get(MATTE_VIEW) or default_engine_settings((MATTE_VIEW,))[MATTE_VIEW])
        gain = float(getattr(self.display, "colour_gain", 1.0) or 1.0)
        full = matte_colour(data, st, gain, use_proxy=False)
        small = matte_colour(data, st, gain, use_proxy=True) if data.albedo_proxy is not None else None
        r2, d2 = copy.copy(r), copy.copy(data)
        r2.albedo_rgb_full, d2.albedo_rgb_full = full, full
        if r.proxy is not None and small is not None:
            r2.proxy = dict(r.proxy, albedo=small)
        if small is not None:
            d2.albedo_proxy = small
        return r2, d2

    def _plate_facts(self) -> dict:
        """What the publication plate says beyond the solve: colour, drift, the
        two-ball note, the 3D view's picture when it is open."""
        p = self.project
        fx = dict(solve_seconds=p.solve_seconds,
                  creator=str(self.prefs.value("rti_web/creator", "")),
                  institution=str(self.prefs.value("rti_web/institution", "")),
                  licence=str(self.prefs.value("publish/licence", "")))  # none until chosen
        if p.colour_check:
            fx["colour"] = dict(p.colour_check)
        if p.images:
            from helio.io.colour_space import profile_name
            prof = profile_name(p.images[0])
            if prof and "srgb" not in prof.lower():
                fx["notes"] = [f"photographs tagged {prof}, colour converted to sRGB"]
        D = self.render_data
        if getattr(D, "solved_colour", None) and D.albedo_rgb_full is D.solved_colour[0]:
            fx.setdefault("notes", []).append("colour: solved reflectance (relative)")
        acc = p.accuracy or {}
        if acc.get("error_mm") is not None:
            fx.setdefault("notes", []).append(
                f"accuracy on a {acc['known_mm']:.3f} mm reference step: {acc['error_mm'] * 1000:+.1f} µm "
                f"({acc['error_pct']:+.2f}%)")
        if p.white_balance:
            from helio.core.white_balance import describe
            fx.setdefault("notes", []).append(f"white balanced on a neutral patch ({describe(p.white_balance)})")
        if p.drift:
            fx["drift_px"] = float(p.drift.get("peak_px", 0.0))
        fl = getattr(p.result, "flash", None) if p.result is not None else None
        if fl and p.result.settings.even_flash:
            fx["flash_spread"] = float(fl["spread"])
        cal = p.calibration
        if p.light_source == "sphere":
            fx["light_note"] = ("two chrome balls" if cal is not None and getattr(cal, "sphere2", None)
                                is not None else "chrome ball")
        elif p.light_source.startswith("file:"):
            fx["light_note"] = "light file " + p.light_source.split(":", 1)[1]
        notes = []
        for n in (getattr(cal, "notes", None) or []):
            if n.startswith("Two balls found"):
                notes.append("the two balls disagree by " + n.split("disagree by ", 1)[1].split(" because")[0])
        fx["notes"] = fx.get("notes", []) + notes      # after the colour notes above, not over them
        ins = self.workspaces.get("Inspect")
        r3 = getattr(ins, "relief", None) if ins is not None else None
        if r3 is not None and r3.isVisible() and r3.gl_ok:
            fx["relief"] = r3.picture()
        prof = getattr(ins, "section_prof", None) if ins is not None else None
        # the line you drew, when the relief it was drawn on is still the
        # surface Publish exports (the selection may have changed since)
        if prof and r3 is not None and r3.section and ins._relief_of is self.selected_result()[1]:
            fx["section"] = (tuple(prof["a_px"]), tuple(prof["b_px"]))
            fx["section_prof"] = section_for_sheet(prof, getattr(ins, "section_card", None))
        fx["kept_sections"] = [dict(name="–".join(self.section_letters(p_)), a=tuple(p_["a_px"]), b=tuple(p_["b_px"]),
                                    copies=max(len(p_.get("series") or []) - 1, 0))
                               for p_, of in getattr(self, "sheet_sections", []) if of is self.render_data or of is None]
        fx["annotations"] = list(getattr(getattr(ins, "annotator", None), "anns", []) or [])
        fx["sheet"] = opt = self._sheet_options()
        if opt.get("clay", True):
            clay = self._sheet_clay()
            if clay is not None:
                opt["shots"].insert(0, ("Clay 3D, raking light", clay))
        return fx

    # ── the publication sheet (helio.application.pub_sheet) ──────────
    def sheet_prefs(self) -> dict:
        import json as _json
        try:
            d = _json.loads(str(self.prefs.value("sheet/options", "{}")) or "{}")
        except ValueError:
            d = {}
        return d if isinstance(d, dict) else {}

    def set_sheet_prefs(self, **kw):
        import json as _json
        d = self.sheet_prefs()
        d.update(kw)
        self.prefs.setValue("sheet/options", _json.dumps(d))

    def add_sheet_shot(self, where: str = "relief") -> bool:
        """The 3D relief's view (or the high-detail window's) as a picture for
        the publication sheet."""
        ins = self.workspaces["Inspect"]
        hd = getattr(self, "hd3d", None)
        v = hd.view if where == "hd" and hd is not None and hd.isVisible() else ins.relief
        if v is None or not v.gl_ok or v.data is None:
            self.flash("Open the 3D relief (or the high-detail window) first, and set the view "
                       "you want.")
            return False
        try:
            pic = v.picture_at(1800, 1200)
        except RuntimeError as e:
            self.flash(str(e))
            return False
        shots = getattr(self, "sheet_shots", [])
        name = "High-detail 3D" if v is not ins.relief else "3D relief"
        n3d = sum(1 for _c, _im, k in shots if k == "3d")
        return self._add_sheet_picture(f"{name}, {n3d + 1}", pic, "3d")

    def add_sheet_tool(self, caption: str, pic) -> bool:
        """A tool's result (a legend's strip, a line drawing, the zones, ...)
        as a picture for the sheet's "From the tools" part."""
        return self._add_sheet_picture(caption, pic, "tool")

    def _add_sheet_picture(self, caption: str, pic, kind: str) -> bool:
        """Keep *pic* for the sheet, asking first when it is all but the same
        as a picture already kept (the same 3D view taken twice)."""
        from helio.application.sheet_contents import close_picture
        shots = getattr(self, "sheet_shots", [])
        near = close_picture(pic, [im for _c, im, _k in shots])
        if near is not None:
            i, _alike = near
            ans = self.ask_sheet_duplicate(
                f"This picture is all but the same as “{shots[i][0]}”, already on the sheet.")
            if ans == "cancel":
                return False
            if ans == "replace":
                shots[i] = (shots[i][0], pic, kind)
                self.sheet_shots = shots
                self.flash(f"“{shots[i][0]}” replaced on the publication sheet.")
                self.project.changed.emit()
                return True
        shots.append((caption, pic, kind))
        self.sheet_shots = shots
        self.flash(f"Added to the publication sheet: {caption}. "
                   f"{len(shots)} picture{'s' if len(shots) != 1 else ''} in all.")
        self.project.changed.emit()
        return True

    def ask_sheet_duplicate(self, text: str) -> str:
        """'replace', 'add' or 'cancel': what to do with something close to
        what the sheet has already."""
        from PySide6.QtWidgets import QMessageBox
        box = QMessageBox(self)
        box.setWindowTitle("Already on the sheet?")
        box.setText(text)
        box.setInformativeText("Replace the one on the sheet with this, keep both, or leave it.")
        rep = box.addButton("Replace it", QMessageBox.ButtonRole.AcceptRole)
        add = box.addButton("Keep both", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return "replace" if box.clickedButton() is rep else "add" if box.clickedButton() is add else "cancel"

    # the contents list (Publish): take off, move, rename
    def remove_sheet_item(self, kind: str, i: int):
        """Take section or picture *i* off the sheet ('section' or 'picture')."""
        lst = list(getattr(self, "sheet_sections" if kind == "section" else "sheet_shots", []))
        if 0 <= i < len(lst):
            del lst[i]
            setattr(self, "sheet_sections" if kind == "section" else "sheet_shots", lst)
            if kind == "section":
                self.name_sections()
            self.project.changed.emit()

    def move_sheet_item(self, kind: str, i: int, by: int) -> int:
        """Move item *i* up (-1) or down (+1); sections take the letters of
        their new place.  Returns its new index."""
        name = "sheet_sections" if kind == "section" else "sheet_shots"
        lst = list(getattr(self, name, []))
        j = i + by
        if not (0 <= i < len(lst) and 0 <= j < len(lst)):
            return i
        lst[i], lst[j] = lst[j], lst[i]
        setattr(self, name, lst)
        if kind == "section":
            self.name_sections()
        self.project.changed.emit()
        return j

    def rename_sheet_picture(self, i: int, caption: str):
        shots = list(getattr(self, "sheet_shots", []))
        if 0 <= i < len(shots) and caption.strip():
            shots[i] = (caption.strip(), shots[i][1], shots[i][2])
            self.sheet_shots = shots
            self.project.changed.emit()

    def add_sheet_pictures(self, paths=None) -> int:
        """Pictures from files (a photograph, a screenshot, a raking-light
        shot) added to the sheet's pictures, captioned by their file names."""
        import cv2
        if paths is None:
            from PySide6.QtWidgets import QFileDialog
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Pictures for the publication sheet",
                str(self.prefs.value("last_folder", os.path.expanduser("~"))),
                "Pictures (*.jpg *.jpeg *.png *.tif *.tiff *.webp *.bmp)")
        shots = getattr(self, "sheet_shots", [])
        added = 0
        for path in paths or []:
            im = cv2.imread(path, cv2.IMREAD_COLOR)
            if im is None:
                self.flash(f"Could not read {os.path.basename(path)}.")
                continue
            from helio.io.colour_space import to_srgb
            im = cv2.cvtColor(to_srgb(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), path), cv2.COLOR_RGB2BGR)
            k = 2000 / max(im.shape[:2])
            if k < 1:
                im = cv2.resize(im, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            shots.append((os.path.splitext(os.path.basename(path))[0],
                          cv2.cvtColor(im, cv2.COLOR_BGR2RGB), "file"))
            added += 1
        self.sheet_shots = shots
        if added:
            self.flash(f"Added to the publication sheet: {added} picture"
                       + ("s" if added != 1 else "") + ".")
            self.project.changed.emit()
        return added

    def add_sheet_section(self, where: str = "relief") -> bool:
        """The section line drawn on the 3D relief (or the high-detail
        window) kept for the publication sheet, so several can be shown."""
        ins = self.workspaces["Inspect"]
        hd = getattr(self, "hd3d", None)
        if where == "hd" and hd is not None and hd.isVisible():
            prof, of, card = hd.card.prof, hd._of, hd.card
        else:
            prof, of, card = getattr(ins, "section_prof", None), ins._relief_of, getattr(ins, "section_card", None)
        if not prof or prof.get("dist") is None:
            self.flash("Draw a section line on the 3D view first.")
            return False
        from helio.application.sheet_contents import close_section, closeness_words
        secs = getattr(self, "sheet_sections", [])
        key = (tuple(prof["a_px"]), tuple(prof["b_px"]))
        if any((tuple(p["a_px"]), tuple(p["b_px"])) == key for p, _o in secs):
            self.flash("That section is on the sheet already.")
            return False
        kept = section_for_sheet(prof, card)
        same_surface = [(i, p_) for i, (p_, o_) in enumerate(secs) if o_ is of]
        near = close_section(kept, [p_ for _i, p_ in same_surface])
        if near is not None:
            i = same_surface[near[0]][0]
            a, b = self.section_letters(secs[i][0])
            ans = self.ask_sheet_duplicate(
                f"This line is close to section {a}–{b}, already on the sheet: "
                f"{closeness_words(near[1], self.project.scale_mm_per_px or 0.0)}.")
            if ans == "cancel":
                return False
            if ans == "replace":
                secs[i] = (kept, of)
                self.sheet_sections = secs
                self.name_sections()
                self.flash(f"Section {a}–{b} replaced on the publication sheet.")
                self.project.changed.emit()
                return True
        if len(secs) >= 6:
            self.flash("The sheet holds six sections; take one off to add another.")
            return False
        secs.append((kept, of))
        self.sheet_sections = secs
        self.name_sections()
        n_cp = len(kept.get("series") or [])
        self.flash(f"Section added to the publication sheet: {len(secs)} kept"
                   + (f", with its {n_cp - 1} parallel copies." if n_cp > 1 else ".")
                   + f" It is {'–'.join(self.section_letters(prof))} on the sheet.")
        self.project.changed.emit()
        return True

    def section_letters(self, prof) -> tuple[str, str]:
        """What the ends of *prof*'s section are called: the letters it has on the sheet if it is kept there
        (A–B, C–D, E–F …), else the next pair not yet taken."""
        kept = [p_ for p_, _of in getattr(self, "sheet_sections", [])]
        key = (tuple(prof["a_px"]), tuple(prof["b_px"])) if prof else None
        idx = next((i for i, p_ in enumerate(kept) if (tuple(p_["a_px"]), tuple(p_["b_px"])) == key), len(kept))
        idx %= 13
        return chr(65 + 2 * idx), chr(66 + 2 * idx)

    def name_sections(self):
        """Give the section line in Inspect (and in the high-detail window) its letters, everywhere it is drawn."""
        ins = self.workspaces.get("Inspect")
        if ins is not None and hasattr(ins, "name_section"):
            ins.name_section()
        hd = getattr(self, "hd3d", None)
        if hd is not None and hasattr(hd, "name_section"):
            hd.name_section()

    def _sheet_clay(self):
        """The 3D relief in clay at an angle, lit low from the side that shows
        the most relief, for the publication sheet; None when the 3D view has
        not been opened (it needs its graphics set up once)."""
        ins = self.workspaces["Inspect"]
        v = getattr(ins, "relief", None)
        if v is None or not v.gl_ok:
            return None
        try:
            ins._push_relief()                     # the selection's surface, if it changed
            if v.data is None:
                return None
            from helio.application.plate_doc import best_raking, from_above
            r, D = self.selected_result()
            L = getattr(r, "light_matrix", None)
            az = from_above(best_raking(L, D.normals_proxy)[0]) if D is not None else 315.0
            return v.clay_picture(light_az=az)
        except Exception:
            return None

    def _sheet_options(self) -> dict:
        """What the publication sheet shows besides the views: the measured
        lines and the boxed details (in the exported surface's pixels), the
        score and the uncertainty, the 3D pictures, the chosen views."""
        p = self.project
        opt = dict(self.sheet_prefs())
        r_all = p.result
        sel = self.selection
        ox, oy = (sel[0], sel[1]) if sel and self.selected_result()[1] is not self.render_data else (0, 0)
        r_sel = self.selected_result()[0]
        W, H = (r_sel.w, r_sel.h) if r_sel is not None else (0, 0)

        def inside(x1, y1, x2, y2):
            x1, x2, y1, y2 = x1 - ox, x2 - ox, y1 - oy, y2 - oy
            ok = all(-1 <= v <= W + 1 for v in (x1, x2)) and all(-1 <= v <= H + 1 for v in (y1, y2))
            return (x1, y1, x2, y2) if ok else None
        meas = getattr(self.workspaces.get("Measure"), "lines", []) or []
        opt["measures"] = [(*b, lab) for (x1, y1, x2, y2, lab) in meas
                           if (b := inside(x1, y1, x2, y2)) is not None]
        ann = getattr(self.workspaces["Inspect"], "annotator", None)
        details = []
        for a in (getattr(ann, "anns", []) or []):
            if a.get("type") == "rect" and len(a.get("pts", [])) >= 2:
                (x1, y1), (x2, y2) = a["pts"][0], a["pts"][-1]
                b = inside(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                if b is not None:
                    details.append((a.get("text") or f"Detail {len(details) + 1}",
                                    tuple(int(round(v)) for v in b)))
        opt["details"] = details[:6]
        # the annotation layer: every kind but the boxes (those are the details above), on the layers that are
        # showing, inside the exported surface
        vis = {ly["name"] for ly in (getattr(ann, "layers", None) or []) if ly.get("visible", True)}
        notes = []
        for a in (getattr(ann, "anns", []) or []):
            if a.get("type") in (None, "rect") or (vis and a.get("layer") not in vis):
                continue
            pts = [(x - ox, y - oy) for x, y in a.get("pts", [])]
            if pts and all(-1 <= x <= W + 1 and -1 <= y <= H + 1 for x, y in pts):
                notes.append(dict(a, pts=pts))
        opt["notes"] = notes[:60]
        opt["shots"] = [(c, im) for c, im, k in getattr(self, "sheet_shots", []) if k != "tool"]
        opt["tools"] = [(c, im) for c, im, k in getattr(self, "sheet_shots", []) if k == "tool"]
        if any(k == "file" for _c, _im, k in getattr(self, "sheet_shots", [])):
            opt["shots_heading"] = "Pictures"
        here = self.selected_result()[1]
        opt["sections"] = [p_ for p_, of in getattr(self, "sheet_sections", [])
                           if of is here or (of is None and here is self.render_data)]
        if r_all is not None:
            pw = self.workspaces.get("Process")
            try:
                opt["score"] = pw._score_of(r_all) if pw is not None else None
                from helio.application.process_tools import precision_rating, readable
                from helio.core.sphere import condition_number
                full = (r_all.quality or {}).get("sigma_deg")
                if full is not None:
                    _rt, med, p95 = precision_rating(full, readable(r_all.albedo_grey_full, full))
                    cond = (condition_number(r_all.light_matrix)
                            if r_all.light_matrix is not None else None)
                    opt["quality"] = dict(med=med, p95=p95, cond=cond)
            except Exception:
                opt["score"] = None
        return opt

    def _profile_tuple(self):
        """Measure's profile as the classic current_profile_data tuple."""
        pr = getattr(self.workspaces.get("Measure"), "profile", None)
        if not pr:
            return None
        return (pr["x_axis"], pr["z_plot"], pr["label"], pr["x_label"], pr["z_unit"],
                pr["Ra"], pr["Rq"], pr["Rz"], pr["Rp"], pr["Rv"])

    def _export_progress(self, job_id, stage, fraction, key):
        p = self.project
        if p.busy != "export":
            return
        rows = dict(p.export_rows)
        if stage and key in rows:
            rows[key] = ("writing", "", [])
        p.export_rows = rows
        p.set_progress(stage, fraction, "")

    def _export_done(self, job):
        rows = {k: (("done" if files and not why else "problem"), why, files)
                for k, files, why in job.result}
        self.project.export_rows = rows
        bad = sum(1 for v in rows.values() if v[0] == "problem")
        self.project.finish("" if not bad else f"{bad} of {len(rows)} could not be written")
        # the web viewers asked for from Inspect open in the browser as soon as they are written
        for k in getattr(self, "_open_when_done", ()):
            page = next((f for f in rows.get(k, ("", "", []))[2] if str(f).lower().endswith((".html", ".htm"))), "")
            if page:
                self.reveal(page)
        self._open_when_done = ()

    def _export_stopped(self, job, why):
        rows = {k: (v[0] if v[0] == "done" else "stopped", v[1], v[2])
                for k, v in self.project.export_rows.items()}
        self.project.export_rows = rows
        self.project.finish(why)

    def reveal_export(self):
        self.reveal(self.project.last_export_dir)

    def reveal(self, path: str):
        """Show *path* in the Finder."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    # ── the quick exports (EXP-010…012) ─────────────────────────────────
    def export_rti_web(self):
        from helio_qt.quick_exports import RtiWebDialog
        self._quick = RtiWebDialog(self)
        self._quick.exec()

    def pick_white_balance(self):
        """View › Pick White Balance: click or box a neutral white on Inspect's colour."""
        self.go_workspace("Inspect")
        ins = self.workspaces["Inspect"]
        if self.display.layer != "Combined Colour":
            ins.set_layer("Combined Colour")
        ins.pick_white_balance()

    def reset_white_balance(self):
        self.workspaces["Inspect"].reset_white_balance()

    def colour_check(self, path: str = ""):
        """Measure › Check Colour: find a ColorChecker in a photograph and say how far off it is."""
        from helio.application.colour_check import check_image
        from helio_qt.quick_exports import ColourCheckDialog
        p = self.project
        if not path:
            start = p.images[p.current] if p.images else os.path.expanduser("~")
            path, _ = QFileDialog.getOpenFileName(
                self, "A photograph with a ColorChecker in it", start,
                "Images (*.jpg *.jpeg *.tif *.tiff *.png *.iiq *.dng *.arw *.cr2 *.cr3 *.nef *.raf)")
            if not path:
                return None
        try:
            res = check_image(path)
        except (LookupError, ValueError) as e:
            QMessageBox.information(self, "Colour accuracy", str(e))
            return None
        p.colour_check = dict(mean=res["mean"], max=res["max"], worst=res["worst"],
                              file=os.path.basename(path), verdict=res["verdict"])
        self.refresh()
        self._colour_dlg = ColourCheckDialog(self, res)
        self._colour_dlg.exec()
        return res

    def send_to_qgis(self):
        from helio_qt.quick_exports import QgisDialog
        self._quick = QgisDialog(self)
        self._quick.exec()

    def export_usdz(self):
        from helio_qt.quick_exports import UsdzDialog
        self._quick = UsdzDialog(self)
        self._quick.exec()

    def _job_progress(self, job_id, stage, fraction, detail):
        if self.project.busy == "solve":
            self.project.set_progress(stage, fraction, detail)

    def _solve_done(self, job):
        self.project.set_result(job.result, time.time() - getattr(self, "_solve_started", time.time()))
        ws = float(getattr(job.result, "work_scale", 1.0) or 1.0)
        if ws < 1.0:
            r = job.result
            self.flash(f"Solved at {ws * 100:.0f}% size: {r.w} × {r.h} pixels, each {1 / ws:.2f}× a "
                       "photograph's. Measurements and the scale refer to these pixels.", 10)
        if self.backlit and self.backlit.get("H") is None:
            self.align_backlit()

    def _solve_failed(self, job):
        self.project.finish(f"Processing stopped: {job.failure}")
        QMessageBox.warning(self, "Processing stopped", str(job.failure))

    # ── Process tools (REC-002…004, IMP-004) and quality checks (QC-001…008) ──
    def recalc_depth(self):
        self.workspaces["Process"].recalc()

    def refine_normals(self):
        self.workspaces["Process"].refine()

    def process_region(self):
        self.workspaces["Process"].region()

    def load_flat(self):
        self.workspaces["Process"].load_flat()

    # ── Capture tools (IMP-002, CAL-001…003, TRN-001) ─────────────────
    def run_background(self, fn, cb, tool: bool = False):
        """Run *fn* off the main thread; *cb(result, error)* on it afterwards.
        *tool*: a research tool's job, which never waits behind another."""
        (self.tool_pool if tool else self.bg_pool).start(_BgTask(fn, cb, self._bg_sig))

    def wait_background(self):
        self.bg_pool.waitForDone(600000)
        self.tool_pool.waitForDone(600000)
        QApplication.processEvents()

    def check_drift(self, quiet: bool = False):
        from helio.application.capture_tools import drift_report
        paths = list(self.project.images)

        def done(r, err):
            if err or r is None or paths != self.project.images:
                return
            self.project.drift = r
            self.project.changed.emit()
            if r["verdict"] == "clean":
                if not quiet:
                    self.flash(f"No movement: {r['n']} frames compared, largest shift "
                               f"{r['peak_px']:.2f} px.")
                return
            head = ("The camera or object moved between frames." if r["verdict"] == "moved"
                    else "There may be slight movement between frames.")
            msg = (f"{head} Largest shift {r['peak_px']:.1f} px ({r['worst']}), median "
                   f"{r['median_px']:.1f} px. Movement smears the normals; Line Up Shifted Frames "
                   "can correct small shifts.")
            self.flash(msg, 12)
        self.run_background(lambda: drift_report(paths), done)

    def alignment_report(self):
        from helio.application.capture_tools import alignment_report
        paths = list(self.project.images)
        self.project.set_busy("tool")
        self.project.set_progress("Measuring each frame's shift…", 0.5)

        def done(shifts, err):
            self.project.finish("" if not err else f"Alignment report stopped: {err}")
            if err or shifts is None:
                return
            from helio_qt.report import ReportDialog
            mx = max((abs(dx) + abs(dy) for dx, dy in shifts), default=0.0)
            figs = [(f"Frame {i + 1}  {os.path.basename(paths[i]) if i < len(paths) else ''}",
                     f"({dx:+.2f}, {dy:+.2f}) px" + ("  low contrast: skipped"
                                                     if dx == 0.0 and dy == 0.0 else ""))
                    for i, (dx, dy) in enumerate(shifts)]
            need = mx > 0.3
            ReportDialog(self, "Frame alignment",
                         (f"Largest combined shift {mx:.2f} px: “Line up shifted frames” is now on "
                          "for the next Process." if need else
                          f"The frames are aligned (largest shift {mx:.2f} px)."),
                         "warn" if need else "ok", figs, save_name="alignment").exec()
            if need:
                self.project.update_settings(auto_align=True)
        self.run_background(lambda: alignment_report(paths), done)

    def save_calibration(self):
        from helio.application.capture_tools import write_calibration_hjson
        from helio.application.exports import light_file
        from helio_qt.publish import file_stem
        p = self.project
        n = len(p.solve_lights())
        start = os.path.join(str(self.prefs.value("last_folder", "")),
                             f"{file_stem(p.name)}_{n}lights.lp")
        path, flt = QFileDialog.getSaveFileName(
            self, "Save the calibration", start,
            "RTIBuilder light file (*.lp);;Helio calibration (*.hjson)")
        if not path:
            return
        b1, b2 = self._ball_tuples()
        ppi = 25.4 / p.photo_scale() if p.scale_mm_per_px else None
        if path.lower().endswith(".hjson") or "hjson" in flt:
            if not path.lower().endswith(".hjson"):
                path += ".hjson"
            write_calibration_hjson(path, p.solve_lights(), p.solve_images(), p.name, ppi,
                                    p.light_source or "measured", sphere=b1, sphere2=b2)
        else:
            if not path.lower().endswith(".lp"):
                path += ".lp"
            light_file(p.solve_lights(), "measured", path, p.solve_images())
            if b1 is not None:
                # The .lp stays RTIBuilder's own format; where the ball was goes in
                # the Helio calibration of the same name beside it.
                write_calibration_hjson(os.path.splitext(path)[0] + ".hjson", p.solve_lights(),
                                        p.solve_images(), p.name, ppi,
                                        p.light_source or "measured", sphere=b1, sphere2=b2)
        self.flash(f"Saved the calibration: {os.path.basename(path)} ({n} lights).")

    # ── rig profiles (helio.application.rigs) ─────────────────────────
    def rig_profiles(self) -> dict:
        from helio.application import rigs
        return rigs.load(str(self.prefs.value(rigs.KEY, "{}")))

    def save_rig_profile(self, name: str = "") -> str:
        """This capture's lights, ball, dome radius, near-field, scale and photo
        tone, saved as a rig profile for the next capture on the same rig."""
        from PySide6.QtWidgets import QInputDialog
        from helio.application import rigs
        p = self.project
        if not p.lights_ready():
            self.flash("Set the lights first: a rig profile keeps them for the next capture.")
            return ""
        if not name:
            name, ok = QInputDialog.getText(self, "Save as a rig profile",
                                            "A name for this rig (for example, Book rig A):")
            if not ok or not name.strip():
                return ""
        name = name.strip()
        b1, b2 = self._ball_tuples()
        s = p.settings
        try:
            prof = rigs.make(p.lights, sphere=b1, sphere2=b2, dome_radius_mm=s.dome_radius_mm,
                             near_field=s.near_field, near_field_from=s.near_field_from,
                             mm_per_px=p.photo_scale(), photo_tone=s.photo_tone,
                             light_field=s.light_field, lens_calibration=s.lens_calibration)
        except ValueError as e:
            QMessageBox.warning(self, "Rig profile", str(e))
            return ""
        allp = self.rig_profiles()
        allp[name] = prof
        self.prefs.setValue(rigs.KEY, rigs.dump(allp))
        self.flash(f"Rig profile “{name}” saved: {rigs.describe(prof)}.")
        self.project.changed.emit()
        return name

    def apply_rig_profile(self, name: str) -> bool:
        """A rig profile onto this capture: lights, ball, near-field, scale, tone."""
        import numpy as np
        from helio.application import rigs
        prof = self.rig_profiles().get(name)
        p = self.project
        if prof is None or not p.images:
            return False
        if int(prof.get("n", 0)) != len(p.images):
            QMessageBox.warning(self, "Rig profile", f"“{name}” has {prof.get('n')} lights; this "
                                f"capture has {len(p.images)} frames.")
            return False
        tup = lambda v: None if not v else tuple(float(x) for x in v)
        p.set_lights(np.asarray(prof["lights"], np.float32), f"rig:{name}",
                     sphere=tup(prof.get("sphere")), sphere2=tup(prof.get("sphere2")))
        kw = dict(near_field=bool(prof.get("near_field")),
                  near_field_from=prof.get("near_field_from") or "ball")
        if prof.get("dome_radius_mm"):
            kw["dome_radius_mm"] = float(prof["dome_radius_mm"])
        if prof.get("photo_tone"):
            kw["photo_tone"] = prof["photo_tone"]
        for k in ("light_field", "lens_calibration"):          # a fixed rig's calibrations, when still there
            v = prof.get(k) or ""
            kw[k] = v if v and os.path.isfile(v) else ""
        p.update_settings(**kw)
        if prof.get("mm_per_px"):
            p.set_scale_photo(float(prof["mm_per_px"]))
        self.check_light_frame()
        self.flash(f"Rig profile “{name}”: {rigs.describe(prof)}.")
        return True

    def delete_rig_profile(self, name: str):
        from helio.application import rigs
        allp = self.rig_profiles()
        allp.pop(name, None)
        self.prefs.setValue(rigs.KEY, rigs.dump(allp))
        self.project.changed.emit()

    def load_calibration(self):
        from helio.application.capture_tools import read_calibration
        path, _ = QFileDialog.getOpenFileName(
            self, "Load a calibration", str(self.prefs.value("last_folder", "")),
            "Calibrations (*.lp *.hjson *.json *.csv *.dome *.txt);;All files (*)")
        if not path:
            return
        try:
            L, src, names = read_calibration(path)
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.warning(self, "Could not read the calibration", str(e))
            return
        L = self._paired_lights(names if len(names) == len(L) else None, L)
        from helio.application.capture_tools import calibration_spheres
        b1, b2 = calibration_spheres(path)
        self.project.set_lights(L, f"file:{os.path.basename(path)}", sphere=b1, sphere2=b2)
        self.check_light_frame()
        self.flash(f"Calibration loaded from {src}: {len(L)} lights"
                   + (", with where the ball was." if b1 else "."))

    # ── precision measurement (helio.application.precision) ───────────
    def calibrate_light_field(self, paths=None, save_to: str = ""):
        """Process › Calibrate Lights on a White Board: one board frame per light, in the
        capture's order; each light's strength and fall-off, applied to every solve."""
        from helio.application.precision import board_field
        from helio.core import light_field as LF
        p = self.project
        if not p.lights_ready():
            self.flash("Set the lights first (the ball or a rig profile): the board is read with them.")
            return None
        n = len(p.used_frames())
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(
                self, f"The white-board frames: {n}, one per light, in the capture's order",
                os.path.dirname(p.images[0]),
                "Images (*.jpg *.jpeg *.png *.tif *.tiff *.dng *.cr2 *.cr3 *.nef *.arw *.raf *.orf *.rw2 *.iiq)")
            if not paths:
                return None
        from helio.io.capture import natural_key
        paths = sorted(paths, key=natural_key)
        try:
            from helio.core.radiometry import resolve_tone
            # read as the solve will read frames once this field evens them
            field = board_field(paths, p.solve_lights(), resolve_tone(p.settings.photo_tone, evened=True))
        except ValueError as e:
            QMessageBox.warning(self, "White-board calibration", str(e))
            return None
        if not save_to:
            save_to = os.path.join(os.path.dirname(paths[0]), "helio_light_field.npz")
        LF.save(save_to, field)
        p.update_settings(light_field=save_to)
        self.flash(f"Light field saved ({LF.describe(field)}). Process to apply; save a rig profile "
                   f"to use it on the next capture.", 10)
        return field

    def set_up_for_measurement(self):
        """Process › Set Up for Measurement (helio.application.precision.MEASUREMENT_SETTINGS)."""
        from helio.application.precision import measurement_changes
        r = self.project.result
        albedo = getattr(r, "albedo_rgb_full", None) if r is not None else None
        ch = measurement_changes(self.project.settings, albedo)
        if ch:
            self.project.update_settings(**ch)
        names = {"mask_surround": "depth on the object alone"}
        said = ", ".join(names.get(k, k.replace("_", " ")) + (" (on: a small object on its surround)"
                         if k == "mask_surround" and v else " (off: the object fills too much of "
                         "the frame for it)" if k == "mask_surround" else "") for k, v in ch.items())
        self.flash(("Set up for measurement: " + said + " changed. " if ch else
                    "Already set up for measurement. ")
                   + "Process › Precision Measurement says what is still to calibrate; "
                     "Measure › Accuracy Check measures the error.")

    def clear_light_field(self):
        self.project.update_settings(light_field="")
        self.flash("White-board light field off. Process to apply.")

    def calibrate_lens(self, paths=None, pattern=None, save_to: str = ""):
        """Process › Calibrate Lens from a Chessboard: 10-20 photographs of a printed
        chessboard across the frame; distortion taken out of every frame before the solve."""
        from PySide6.QtWidgets import QInputDialog
        from helio.application.precision import lens_from_photos, parse_pattern
        from helio.core import lens
        p = self.project
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Chessboard photographs (10-20, the board at different places and tilts)",
                os.path.dirname(p.images[0]) if p.images else str(self.prefs.value("last_folder", "")),
                "Images (*.jpg *.jpeg *.png *.tif *.tiff *.dng *.cr2 *.cr3 *.nef *.arw *.raf *.orf *.rw2 *.iiq)")
            if not paths:
                return None
        if pattern is None:
            text, ok = QInputDialog.getText(self, "Chessboard", "Inner corners, columns x rows "
                                            "(a board of 10 x 7 squares has 9 x 6):", text="9 x 6")
            if not ok:
                return None
            try:
                pattern = parse_pattern(text)
            except ValueError as e:
                QMessageBox.warning(self, "Chessboard", str(e))
                return None
        try:
            cal = lens_from_photos(paths, pattern)
        except ValueError as e:
            QMessageBox.warning(self, "Lens calibration", str(e))
            return None
        if not save_to:
            save_to = os.path.join(os.path.dirname(paths[0]), "helio_lens.npz")
        lens.save(save_to, cal)
        self.project.update_settings(lens_calibration=save_to)
        self.flash(f"Lens calibrated ({lens.describe(cal)}). Process to apply.", 10)
        return cal

    def print_targets(self, kind: str = "chessboard"):
        """File › Print Calibration Targets: the chessboard, scale bars or alignment grid as a PDF."""
        from helio_qt.print_targets import PrintTargetsDialog
        self._targets_dlg = PrintTargetsDialog(self, kind if isinstance(kind, str) else "chessboard")
        self._targets_dlg.show()
        return self._targets_dlg

    def print_chessboard(self):
        return self.print_targets("chessboard")

    def print_scale_bars(self):
        return self.print_targets("scales")

    def check_alignment(self, path: str = "", square_mm: float | None = None, ask: bool = True):
        """Capture › Check Camera Alignment: the printed chessboard flat on the baseboard says how
        far the camera is off square, and the scale there (offered as the scale)."""
        import cv2
        from PySide6.QtWidgets import QInputDialog
        from helio.core import lens
        from helio.core.frame_read import read_frame
        from helio.core.sphere import focal_px_from_exif
        p = self.project
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "A photograph of the chessboard lying flat on the baseboard",
                os.path.dirname(p.images[0]) if p.images else str(self.prefs.value("last_folder", "")),
                "Images (*.jpg *.jpeg *.png *.tif *.tiff *.dng *.cr2 *.cr3 *.nef *.arw *.raf *.orf *.rw2 *.iiq)")
            if not path:
                return None
        if square_mm is None and ask:
            v, ok = QInputDialog.getDouble(self, "Chessboard", "The size of one square, in mm (printed "
                                           "on the sheet; 0 if not known):", 22.0, 0.0, 500.0, 2)
            if not ok:
                return None
            square_mm = v or None
        g = read_frame(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            QMessageBox.warning(self, "Camera alignment", f"Could not read {os.path.basename(path)}.")
            return None
        if g.dtype != "uint8":
            g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        cal = None
        if p.settings.lens_calibration and os.path.isfile(p.settings.lens_calibration):
            cal = lens.load(p.settings.lens_calibration)
            if lens.check_size(cal, (g.shape[1], g.shape[0])):
                cal = None
        try:
            pose = lens.board_pose(g, (9, 6), square_mm, cal=cal, focal_px=focal_px_from_exif(path))
        except ValueError as e:
            QMessageBox.warning(self, "Camera alignment", str(e))
            return None
        lines = [lens.describe_pose(pose).capitalize() + ".",
                 f"Board turned {pose['roll_deg']:+.1f}° against the frame.",
                 f"Focal length from {pose['focal_from']}."]
        if "mm_per_px" in pose:
            lines.append(f"Scale on the board: {pose['mm_per_px'] * 1000:.2f} µm per pixel "
                         f"({25.4 / pose['mm_per_px']:.0f} PPI).")
        self.alignment = pose
        if ask:
            box = QMessageBox(self)
            box.setWindowTitle("Camera alignment")
            box.setText("\n".join(lines))
            use = None
            if "mm_per_px" in pose and p.images:
                use = box.addButton("Use as the scale", QMessageBox.ButtonRole.AcceptRole)
                use.setToolTip("Set the photographs' scale from the board: right for objects lying on "
                               "the baseboard at the board's height.")
            box.addButton(QMessageBox.StandardButton.Close)
            box.exec()
            if use is not None and box.clickedButton() is use:
                p.set_scale_photo(pose["mm_per_px"])
                p.changed.emit()
        return pose

    def clear_lens(self):
        self.project.update_settings(lens_calibration="")
        self.flash("Lens correction off. Process to apply.")

    def load_ambient(self):
        """The lights-off photograph: the light the lit frames share, subtracted before the solve."""
        import cv2
        path, _ = QFileDialog.getOpenFileName(
            self, "A photograph with the lamps off (everything else as for the lit frames)",
            str(self.prefs.value("last_folder", "")) or os.path.dirname(self.project.images[0]),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.dng *.cr2 *.cr3 *.nef *.arw *.raf *.orf *.rw2);;"
            "All files (*)")
        if not path:
            return
        if path in set(self.project.images):
            QMessageBox.warning(self, "Lights-off frame", "That photograph is one of the lit frames. Leave it "
                                "out of the capture first (Capture), then choose it here.")
            return
        from helio.io import load_and_process_single
        _rgb, grey = load_and_process_single(path)
        if grey is None:
            QMessageBox.warning(self, "Lights-off frame", f"Could not read {os.path.basename(path)}.")
            return
        self.project.update_settings(ambient_frame=path)
        self.flash(f"Lights-off frame {os.path.basename(path)}: its light is taken out of every frame "
                   "before the solve. Process to apply.")

    def clear_ambient(self):
        if not self.project.settings.ambient_frame:
            self.flash("No lights-off frame to remove.")
            return
        self.project.update_settings(ambient_frame="")
        self.flash("Lights-off frame removed. Process to apply.")

    def load_backlit(self):
        """An optional backlit (transmitted-light) photograph, kept apart from the solve."""
        import cv2
        path, _ = QFileDialog.getOpenFileName(
            self, "A backlit photograph (not one of the lit frames)",
            str(self.prefs.value("last_folder", "")), "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if not path:
            return
        if path in set(self.project.images):
            QMessageBox.warning(self, "Backlit image", "That photograph is one of the lit frames. "
                                "The backlit frame must be a separate photograph.")
            return
        from helio.core.frame_read import read_frame
        from helio.io.raw_loader import is_raw
        raw = read_frame(path, cv2.IMREAD_COLOR, sharp=True) if is_raw(path) else cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            QMessageBox.warning(self, "Backlit image", f"Could not read {os.path.basename(path)}.")
            return
        self.backlit = {"raw": raw, "path": path, "H": None, "aligned": None}
        if self.project.result is not None:
            self.align_backlit()
        else:
            self.flash(f"Backlit frame {os.path.basename(path)} kept; it is lined up with the "
                       "surface after Process.")

    def align_backlit(self):
        from helio.application.capture_tools import reference_grey, register_backlit, to_grey_u8
        b, r = self.backlit, self.project.result
        if not b or r is None:
            return
        raw, ref = b["raw"], reference_grey(r.albedo_rgb_full)

        def work():
            H, inl, flip = register_backlit(to_grey_u8(raw), ref)
            return H, inl, flip, ref.shape[:2]

        def done(out, err):
            if err or out is None:
                self.flash(f"Could not line up the backlit frame: {err}")
                return
            H, inl, flip, shape = out
            if H is None:
                self.flash(f"The backlit frame could not be lined up automatically ({inl} "
                           "matching features; 15 are needed).", 10)
                return
            b.update(H=H, ref_shape=shape, inliers=inl, flip=flip, aligned=None)
            self.flash(f"Backlit frame lined up ({inl} matching features"
                       + (", mirrored" if flip is not None else "") + "). Inspect › Backlit frame.")
            self.project.changed.emit()
        self.run_background(work, done)

    def backlit_image(self, shape):
        """The lined-up backlit frame at *shape* (h, w), uint8 RGB, or None."""
        import numpy as np
        from helio.application.capture_tools import warp_backlit
        b = self.backlit
        if not b or b.get("H") is None:
            return None
        key = tuple(shape)
        if b.get("aligned_at") != key:
            b["aligned"] = (np.clip(warp_backlit(b["raw"], b["H"], b["ref_shape"], key), 0, 1)
                            * 255).astype(np.uint8)
            b["aligned_at"] = key
        a = b["aligned"]
        return np.repeat(a[..., None], 3, axis=2) if a.ndim == 2 else a

    def show_capture_details(self):
        from helio_qt.dialogs import CaptureDetailsDialog
        CaptureDetailsDialog(self).exec()

    def _show_object_profile(self):
        """Inspect's "For this object" views and Publish's preset for the profile the capture
        details name, without touching the settings (an opened session keeps its own)."""
        from helio.domain.object_profiles import of
        ins = self.workspaces.get("Inspect")
        if ins is not None and hasattr(ins, "set_suggested"):
            ins.set_suggested(of(self.project.capture_meta).views)
        pub = self.workspaces.get("Publish")
        if pub is not None and hasattr(pub, "_reload_presets"):
            pub._reload_presets()

    def apply_object_profile(self, key: str):
        """A new object type (helio.domain.object_profiles): its solver and depth method, its
        views first in Inspect.  Its export preset is offered in Publish.  General changes
        nothing; nothing is hidden."""
        from helio.application.process_tools import preset
        from helio.domain.object_profiles import get
        p = get(key)
        changes = {}
        if p.material:
            settings, shared, _view = preset(p.material)
            changes.update(settings)
            self.shared.update(shared)
        changes.update(p.settings)
        if changes:
            self.project.update_settings(**changes)
        self._show_object_profile()
        if p.key != "general":
            self.flash(f"Object type “{p.name}”: solver and depth method set, its views first in "
                       "Inspect and its exports offered in Publish. Process to apply.")

    def mark_ball(self):
        self.workspaces["Capture"].mark_by_hand()

    def adjust_outline(self):
        cap = self.workspaces["Capture"]
        p = self.project
        if p.light_source != "sphere" and p.calibration is not None:
            self.apply_calibration(p.calibration)
        self.go_workspace("Capture")
        cap._zoom_ball()
        cap.view.setFocus()

    def _qc_sigma(self) -> float:
        return float(self.workspaces["Process"].rows["qc_sigma"].value())

    def qc_report(self):
        import matplotlib.cm as cm
        import numpy as np
        from helio.application.process_tools import colour, normal_qc
        from helio_qt.report import ReportDialog
        r = self.project.result
        flat = (self.flat_ref_normals if (self.flat_ref_normals is not None
                                          and self.project.settings.flat_correction) else None)
        q = normal_qc(r.normals_full, self._qc_sigma(), sigma_deg=(r.quality or {}).get("sigma_deg"),
                      flat_ref=flat)
        figs, imgs, raw = [], [], {}
        if "precision_stats" in q:
            ps = q["precision_stats"]
            figs += [("Normal uncertainty, median (1σ)", f"{ps['median']:.3f}°"),
                     ("Normal uncertainty, 95th percentile", f"{ps['p95']:.3f}°"),
                     ("Pixels uncertain by more than 1°", f"{ps['over_1']:.1f}%"),
                     ("Pixels uncertain by more than 3°", f"{ps['over_3']:.1f}%")]
            imgs.append(((cm.cividis(np.clip(q["precision"] / 5.0, 0, 1))[..., :3] * 255)
                         .astype(np.uint8), "Normal uncertainty, 1σ (0–5°)"))
            raw["normal_sigma_deg"] = q["precision"]
        vs = q["variation_stats"]
        figs += [("Surface variation from the mean normal", f"{vs['mean']:.3f}°"),
                 (f"  broad (σ {q['sigma']:.0f} px) / fine", f"{vs['broad']:.3f}° / {vs['fine']:.3f}°")]
        if "flat_residual" in q:
            figs.append(("Deviation from the corrected flat (mean / 95th)",
                         f"{q['flat_residual']['mean']:.3f}° / {q['flat_residual']['p95']:.3f}°"))
        imgs.append((colour(q["variation"], vmax=10), "Surface variation (0–10°): the relief"))
        status = ("ok" if q["rating"].startswith("Metrological") else
                  "info" if q["rating"].startswith("Research") else "warn")
        ReportDialog(self, "Normal QC report", q["rating"], status, figs, imgs,
                     note="The rating is the normals' measured precision, from how well each "
                          "pixel's lights agree. Surface variation is the object's own shape, "
                          "not error.", save_name="normal_qc", raw=raw).exec()

    def qc_bias(self):
        from helio.core.processing import load_reference_normal_map
        from helio.application.process_tools import bias_diagnostic
        from helio_qt.report import ReportDialog
        r = self.project.result
        ref = None
        if QMessageBox.question(self, "Low-frequency bias",
                                "Compare with a reference normal map from photogrammetry or a "
                                "scanner too?") == QMessageBox.StandardButton.Yes:
            path, _ = QFileDialog.getOpenFileName(self, "Reference normal map", "",
                                                  "Images (*.png *.tif *.tiff)")
            if path:
                ref = load_reference_normal_map(path, r.h, r.w)
        b = bias_diagnostic(r.normals_full, self._qc_sigma(), ref)
        figs = [(f"Fine deviation from the broad shape (σ {b['sigma']:.0f} px), mean",
                 f"{b['hp_mean']:.2f}°"), ("  95th percentile", f"{b['hp_p95']:.2f}°")]
        imgs = [(b["full_rgb"], "Normals"), (b["lp_rgb"], f"Broad shape (σ {b['sigma']:.0f} px)"),
                (b["hp_rgb"], "Fine deviation (viridis)")]
        if "mae_overall" in b:
            figs += [("Error against the reference, overall", f"{b['mae_overall']:.2f}°"),
                     ("  broad part (calibration bowl)", f"{b['mae_lp']:.2f}°"),
                     ("  fine part", f"{b['mae_hp']:.2f}°")]
            imgs.append((b["ref_lp_rgb"], "Reference, broad shape"))
        ReportDialog(self, "Low-frequency bias", b.get("verdict", ""),
                     "ok" if b.get("mae_lp", 0) < 3 else "warn", figs, imgs,
                     note="Laurent et al. 2025: calibration error sits in the broad shape; the "
                          "fine detail photometric stereo recovers is largely unaffected.",
                     save_name="bias_diagnostic", raw={"hp_deviation_deg": b["hp_angle"]}).exec()

    def qc_compare(self):
        from helio.core.processing import load_reference_normal_map
        from helio.application.process_tools import compare_to_reference
        from helio_qt.report import ReportDialog
        r = self.project.result
        path, _ = QFileDialog.getOpenFileName(self, "Reference normal map (photogrammetry or "
                                              "scanner)", "", "Images (*.png *.jpg *.tif *.tiff)")
        if not path:
            return
        c = compare_to_reference(r.normals_full, load_reference_normal_map(path, r.h, r.w),
                                 self._qc_sigma())
        figs = [("Mean angular error (all scales)", f"{c['mean']:.2f}°"),
                ("Median", f"{c['median']:.2f}°"), ("95th percentile", f"{c['p95']:.2f}°"),
                (f"Broad-scale error (σ {c['sigma']:.0f} px)", f"{c['lp']:.2f}°"),
                ("Fine-scale contribution (est.)", f"{c['hp']:.2f}°")]
        ReportDialog(self, "Compare with a reference normal map", c["verdict"],
                     "ok" if c["lp"] < 3.5 else "warn", figs,
                     [(c["ref_rgb"], "Reference normals"), (c["heat_rgb"], "Angular error"),
                      (c["lp_rgb"], "Broad-scale error")],
                     note="Laurent et al. 2025, Tables 1–2: broad-scale error under 2° is excellent "
                          "calibration; over 5° shows a bowl bias.",
                     save_name="mae", raw={"mae_deg": c["mae_map"]}).exec()

    def qc_disagreement(self):
        import cv2
        import numpy as np
        from helio.core.qc import integrator_disagreement
        from helio_qt.report import ReportDialog
        r = self.project.result
        diff, st = integrator_disagreement(np.asarray(r.depth_vis_p, np.float64),
                                           np.asarray(r.depth_vis_fc, np.float64))
        scale, unit = 1.0, "normalised units"          # two 0-1 display depths: no height scale applies
        mag = np.abs(np.nan_to_num(diff))
        hi = st["p95"] if st["p95"] > 1e-12 else (st["max"] or 1.0)
        vis = cv2.cvtColor(cv2.applyColorMap((np.clip(mag / hi, 0, 1) * 255).astype(np.uint8),
                                             cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
        ReportDialog(self, "Integrator disagreement",
                     f"{st['rms_frac'] * 100:.2f}% of the relief depends on the integrator chosen",
                     "ok" if st["rms_frac"] < 0.05 else "warn",
                     [("RMS disagreement", f"{st['rms'] * scale:.4g} {unit}"),
                      ("95th percentile", f"{st['p95'] * scale:.4g} {unit}"),
                      ("Worst pixel", f"{st['max'] * scale:.4g} {unit}")],
                     [(vis, "Where Poisson and Frankot-Chellappa disagree")],
                     note="After removing the gain and datum each is free to choose. Treat "
                          "measurements in bright areas with caution.",
                     save_name="integrator_disagreement",
                     raw={"signed_difference": diff * scale}).exec()

    def qc_uncertainty(self):
        import numpy as np
        from helio.io.exporters import write_geotiff
        r = self.project.result
        q = r.quality or {}
        if q.get("resid") is None and q.get("sigma_deg") is None:
            QMessageBox.information(self, "Uncertainty maps",
                                    "No uncertainty maps: they come from a calibrated solve.")
            return
        from helio_qt.publish import file_stem
        folder = QFileDialog.getExistingDirectory(self, "Save the uncertainty maps to",
                                                  self.default_export_dir())
        if not folder:
            return
        stem = file_stem(self.project.name)
        out = []
        if q.get("resid") is not None:
            p1 = os.path.join(folder, f"{stem}_uncertainty.tif")
            write_geotiff(p1, np.asarray(q["resid"], np.float32) / 255.0, tags={
                "HELIO_LAYER": "uncertainty", "HELIO_UNITS": "0-1 normalised RMS residual",
                "HELIO_UNCERTAINTY": "RMS residual of the per-pixel Lambertian fit"})
            out.append(p1)
        if q.get("sigma_deg") is not None:
            p2 = os.path.join(folder, f"{stem}_normal_sigma_deg.tif")
            write_geotiff(p2, np.asarray(q["sigma_deg"], np.float32), tags={
                "HELIO_LAYER": "normal_uncertainty", "HELIO_UNITS": "degrees (1 sigma)",
                "HELIO_UNCERTAINTY": "least-squares angular uncertainty of each normal"})
            out.append(p2)
        self.flash("Saved " + ", ".join(os.path.basename(x) for x in out))

    def qc_dome3d(self):
        from helio.application.process_tools import light_dome_figure
        from helio_qt.report import ReportDialog
        L = self.project.solve_lights()
        img = light_dome_figure(L, dark=self.tokens.name == "dark")
        from helio.core.sphere import condition_number, dome_coverage
        cov = dome_coverage(L)
        ReportDialog(self, "3D light dome", f"{len(L)} lights · dome coverage {cov['total']}/100 "
                     f"({cov['grade']})", "info",
                     [("Conditioning", f"{condition_number(L):.2f}"),
                      ("Elevations", f"{cov['elevations'][0]:.0f}°–{cov['elevations'][1]:.0f}°"),
                      ("Compass sectors with a light", f"{cov['sectors_filled']} of 12")],
                     [(img, "Each light on the hemisphere, coloured by height")],
                     save_name="light_dome").exec()

    def copy_image(self):
        from helio_qt.widgets import read_image
        img = read_image(self.project.images[self.project.current])
        if not img.isNull():
            QApplication.clipboard().setImage(img)

    def set_solver(self, name: str):
        if name != self.project.settings.normal_engine:
            self.project.update_settings(normal_engine=name)
        self.registry.actions[f"process.solver:{name}"].setChecked(True)

    def set_depth(self, name: str):
        if name != self.project.settings.depth_engine:
            self.project.update_settings(depth_engine=name)
        self.registry.actions[f"process.depth:{name}"].setChecked(True)

    def _viewport(self):
        ws = self.workspaces.get(self.current)
        return getattr(ws, "viewport", None) or self.workspaces["Inspect"].viewport

    def zoom_fit(self):
        self._viewport().fit()

    def zoom_in(self):
        self._viewport().zoom_by(1.25)

    def zoom_out(self):
        self._viewport().zoom_by(1 / 1.25)

    def toggle_sidebar(self):
        self.left.setVisible(not self.left.isVisible())
        self.registry.actions["view.sidebar"].setChecked(self.left.isVisible())

    def toggle_inspector(self):
        self.insp_stack.setVisible(not self.insp_stack.isVisible())
        self.registry.actions["view.inspector"].setChecked(self.insp_stack.isVisible())

    def set_scheme(self, name: str, save: bool = True):
        """Graphite and Paper (neutral) or Night Dome and Parchment."""
        if name not in theme.FAMILIES:
            return
        self.scheme = name
        if save:
            self.prefs.setValue("scheme", name)
        self.set_appearance(self.appearance, save=False)

    def set_appearance(self, mode: str, save: bool = True):
        self.appearance = mode
        self.tokens = theme.apply(QApplication.instance(), mode, self.scheme)
        self.registry.actions[f"view.appearance:{mode}"].setChecked(True)
        self.registry.actions[f"view.scheme:{self.scheme}"].setChecked(True)
        t = self.tokens
        self.btn_side.setIcon(icons.icon("sidebar", t.text2))
        self.btn_insp.setIcon(icons.icon("inspector", t.text2))
        self.btn_search.setIcon(icons.icon("search", t.text2))
        for ws in self.workspaces.values() if hasattr(self, "workspaces") else ():
            ws.recolour(t)
            for fl in (getattr(ws, "grid", None), getattr(ws, "strip", None)):
                if isinstance(fl, FrameList):
                    fl.set_frames(fl.paths, [fl.item(i).text() for i in range(fl.count())])
        if save:
            self.prefs.setValue("appearance", mode)
        if hasattr(self, "workspaces"):
            self.refresh()

    def _system_scheme_changed(self, *_):
        if self.appearance == "system":
            self.set_appearance("system", save=False)

    def toggle_zoomed(self):
        self.showNormal() if self.isMaximized() else self.showMaximized()

    def show_palette(self):
        self.palette_dlg.open_centered()

    def save_sweep_gif(self):
        """G: the light swept once round, saved.  With the 3D relief showing,
        Record video opens at its Light sweep; else Save the light sweep for
        the 2D view (height, length, size, GIF or MP4)."""
        ins = self.workspaces["Inspect"]
        if self.current == "Inspect" and ins.centre_mode != "image":
            return self.open_record_dialog("light")
        if self.project.result is None:
            self.flash("Process the images first.")
            return None
        from helio_qt.sweep_dialog import SweepDialog
        self.sweep_dialog = SweepDialog(self)
        self.sweep_dialog.show()
        return self.sweep_dialog

    def write_light_sweep(self, path: str, wait: bool = False, **kw) -> str:
        """The 2D view's light sweep into *path* (GIF or MP4), off the main
        thread (or here, with *wait*); keywords as publish.light_sweep."""
        from helio_qt import publish
        src = self.publish_source()
        if wait:
            return publish.light_sweep(src, path, **kw)

        def done(out, err):
            if err or not out:
                QMessageBox.warning(self, "Light sweep", str(err or "Nothing was written."))
                return
            mb = os.path.getsize(out) / 1e6 if os.path.isfile(out) else 0.0
            self.flash(f"Saved {os.path.basename(path)} ({mb:.1f} MB).")
        self.flash("Making the light sweep…", 30)
        self.run_background(lambda: publish.light_sweep(src, path, **kw), done)
        return ""

    # ── the centre: image, 3D relief, side by side (VIEW-004) ────────────
    def set_centre_mode(self, mode: str):
        ins = self.workspaces["Inspect"]
        if self.current != "Inspect":
            self.go_workspace("Inspect")
        ins.set_centre_mode(mode)
        a = self.registry.actions.get(f"view.mode:{ins.centre_mode}")
        if a is not None:
            a.setChecked(True)

    def _relief_ready(self):
        ins = self.workspaces["Inspect"]
        if ins.relief is None or ins.centre_mode == "image":
            self.set_centre_mode("3d")
        r = ins.relief
        if r is None or not r.gl_ok:
            QMessageBox.information(self, "3D relief", "The 3D view is not available here: "
                                    + (r.gl_error if r is not None and r.gl_error else
                                       "it needs OpenGL 4.1."))
            return None
        return r

    def save_relief_picture(self, relief=None):
        """The 3D relief (or *relief*, the high-detail window's) as a picture:
        Save View, opened on The 3D relief."""
        if relief is None and self._relief_ready() is None:
            return None
        return self.save_view("3d", relief)

    def write_relief_picture(self, r, path: str, long_side: int = 0, background: str = "") -> str:
        """*r* drawn into *path* (PNG, TIFF or JPEG by its name): as on screen,
        or afresh at *long_side* px in the view's shape, on *background*."""
        import cv2
        keep = r.surround
        try:
            if background:
                r.surround = background
            if long_side:
                w, h = max(1, r.width()), max(1, r.height())
                k = long_side / float(max(w, h))
                img = r.picture_at(int(round(w * k)), int(round(h * k)))
            elif background:
                dpr = r.devicePixelRatioF()
                img = r.picture_at(int(r.width() * dpr), int(r.height() * dpr))
            else:
                img = r.picture()
        except RuntimeError as e:
            QMessageBox.warning(self, "Save the 3D view", f"It could not be drawn: {e}")
            return ""
        finally:
            r.surround = keep
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        extra = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.lower().endswith((".jpg", ".jpeg")) else []
        if not cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), extra):
            QMessageBox.warning(self, "Save the 3D view", f"Could not write {path}.")
            return ""
        from helio_qt.publish import embed_meta
        embed_meta(path, self.project.capture_meta, "3D relief")
        self.flash(f"Saved {os.path.basename(path)}, {img.shape[1]} × {img.shape[0]}.")
        return path

    def save_centre_picture(self, path: str = "", long_side: int = 3840):
        """The centre (image, 3D relief, or both side by side) as one picture:
        into *path* when given, else Save View, opened on what the centre shows."""
        if not path:
            return self.save_view()
        return self.write_centre_picture(path, long_side)

    def write_centre_picture(self, path: str, long_side: int = 3840) -> str:
        """The centre as it is drawn now into *path* (PNG, TIFF or JPEG by its
        name), *long_side* px on its longer side (0: as on screen)."""
        ins = self.workspaces["Inspect"]
        if ins.centre_mode != "image" and self._relief_ready() is None:
            return ""
        if not long_side:
            c = ins.split if hasattr(ins, "split") else ins
            long_side = int(round(max(c.width(), c.height()) * c.devicePixelRatioF()))
        try:
            img = ins.centre_picture(long_side)
        except Exception as e:
            QMessageBox.warning(self, "Save View", f"It could not be drawn: {e}")
            return ""
        import cv2
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        extra = [cv2.IMWRITE_JPEG_QUALITY, 95] if path.lower().endswith((".jpg", ".jpeg")) else []
        if not cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), extra):
            QMessageBox.warning(self, "Save View", f"Could not write {path}.")
            return ""
        from helio_qt.publish import embed_meta
        embed_meta(path, self.project.capture_meta, "centre")
        self.flash(f"Saved {os.path.basename(path)}, {img.shape[1]} × {img.shape[0]}.")
        return path

    def record_relief_turntable(self, path: str = "", frames: int = 120, relief=None,
                                flyover: bool = False, size=None):
        """A turntable (or flyover) of the 3D relief (or *relief*). Without a
        path, the Record video window opens with that style chosen."""
        from helio.application.viewer3d import FLYOVER_FRAMES
        style = "flyover" if flyover else "turntable"
        if not path:
            return self.open_record_dialog(style, relief)
        if flyover and frames == 120:
            frames = FLYOVER_FRAMES
        return self.record_relief_video(path, style, frames=frames, relief=relief, size=size)

    def record_relief_flyover(self, path: str = "", relief=None):
        """A flyover of the 3D relief (or *relief*): one smooth loop."""
        return self.record_relief_turntable(path, relief=relief, flyover=True)

    def record_relief_sweep(self, relief=None):
        """Record video at its Light sweep: the camera still, the light circling."""
        return self.open_record_dialog("light", relief)

    def open_record_dialog(self, style: str = "turntable", relief=None):
        """Record video: the style, length, size and, for a light sweep, the light's height."""
        r = relief or self._relief_ready()
        if r is None:
            return ""
        from helio_qt.record_dialog import RecordDialog
        self.record_dialog = RecordDialog(self, r, style)
        self.record_dialog.show()
        return ""

    def record_relief_video(self, path: str, style: str = "turntable", frames: int = 0,
                            relief=None, size=None, light_el: float = 30.0, seconds: float = 0,
                            fps: int = 0, reverse: bool = False, light_az0: float = 0.0,
                            fit: bool = False, max_mb: float = 0.0, repeat: int = 1,
                            views=None, layout: str = "side", labels: bool = True):
        """The 3D relief (or *relief*) recorded in *style* (viewer3d.VIDEO_STYLES)
        as MP4 (or GIF by the file's name), at *size* (1920 x 1080 by default)
        whatever the window's size. With *fit*, the frame is cut to the object
        (as far as it goes in the whole video), its long side *size*'s, so a
        portrait object is not a strip down the middle of a wide frame. The
        camera and the light are put back.

        *views*, [(engine, layer), ...] with ("", "") for the relief as it is
        shown, records two or three surfaces: each other view is the 2D view
        draped on the relief (relit as the light goes round, in the light
        sweep); *layout* "side" by side or "after" one another
        (helio.application.multiview), *labels* naming each."""
        import copy
        import numpy as np
        from PySide6.QtGui import QColor
        from helio.application import multiview as MV
        from helio.application.viewer3d import (VIDEO_FPS, VIDEO_SIZE, fitted_frame, object_extent,
                                                video_pose, write_video)
        from helio.domain.view_controls import from_angles
        r = relief or self._relief_ready()
        if r is None:
            return ""
        fps = int(fps or VIDEO_FPS)
        frames = int(frames or round((seconds or 12) * fps))
        w, h = size or VIDEO_SIZE
        w0, h0 = w, h
        cam = r.cam
        keep = (cam.az, cam.el, cam.dist)
        keep_light = np.array(r.light, copy=True)
        keep_mat = r.material
        keep_view = None if r.data is None else r.data.get("view")
        vs = [tuple(v) for v in (views or [("", "")])][:MV.MAX_VIEWS]
        several = len(vs) > 1
        side = several and layout == "side"
        ins = self.workspaces["Inspect"]
        drapes = {}

        def drape(eng, layer, L=None):
            """The 2D view *eng* in *layer*, as the screen draws it, for the relief."""
            from helio.application.viewing import screen_image
            from helio.domain.registry import default_engine_settings
            if L is None and (eng, layer) in drapes:
                return drapes[(eng, layer)]
            d = copy.deepcopy(self.display)
            d.layer = layer
            st = dict(self.engine_settings.get(eng) or default_engine_settings((eng,)).get(eng) or {})
            if L is not None:
                from helio.domain.view_controls import light_settings
                st.update(light_settings(*L))
            img = ins._view_for_relief(screen_image(eng, self.render_data, st, d))
            if L is None:
                drapes[(eng, layer)] = img
            return img

        def label(eng, layer):
            if eng:
                return MV.label_of(eng, layer)
            from helio_qt.relief3d import MATERIALS
            if keep_mat == "view":
                return MV.label_of(self.engine, self.display.layer)
            return f"3D, {dict(MATERIALS).get(keep_mat, keep_mat).lower()}"
        names = [label(e, l) for e, l in vs] if several and labels else []

        def pose(i):
            az, el, dist, light = video_pose(style, i, frames, *keep, light_el=light_el,
                                             reverse=reverse, light_az0=light_az0)
            cam.az, cam.el, cam.dist = az, el, dist
            if light is not None:
                L = from_angles(*light)
                r.set_light_screen(*L)
                return L
            return None

        def surface(v, L):
            eng, layer = vs[v]
            if not eng:
                r.material = keep_mat
                if keep_view is not None and r.data is not None and r.data.get("view") is not keep_view:
                    r.set_view_image(keep_view)
            else:
                r.material = "view"
                r.set_view_image(drape(eng, layer, L if style == "light" else None))

        draw_w, draw_h, crop = w, h, None
        shots = []
        try:
            box = None
            if fit:                     # where the object goes, from a quick small pass
                k = 320.0 / max(w, h)
                sw, sh = max(16, int(w * k)), max(16, int(h * k))
                probe = []
                for i in range(0, frames, max(1, frames // 24)):
                    pose(i)
                    probe.append(r.picture_at(sw, sh))
                bg = QColor(getattr(r, "surround", None) or "#2B2B2B")
                box = object_extent(probe, (bg.red(), bg.green(), bg.blue()))
                if box is not None:
                    draw_w, draw_h, crop, w, h = fitted_frame(box, (w, h))
            if side:                    # each panel smaller, so the whole is the size chosen
                rows, cols = MV.grid(len(vs), w, h)
                k = min(1.0, max(w0, h0) / float(max(cols * w, rows * h)))
                if k < 0.999:
                    if box is not None:
                        draw_w, draw_h, crop, w, h = fitted_frame(box, (int(w0 * k), int(h0 * k)))
                    else:
                        w, h = int(w * k) // 2 * 2, int(h * k) // 2 * 2
                        draw_w, draw_h = w, h
            bg = QColor(getattr(r, "surround", None) or "#2B2B2B")
            bg = (bg.red(), bg.green(), bg.blue())

            def shot():
                pic = r.picture_at(draw_w, draw_h)
                if crop is not None:
                    x, y, cw, ch = crop
                    pic = np.ascontiguousarray(pic[y:y + ch, x:x + cw])
                return pic
            total = frames * (1 if side or not several else len(vs))
            for v in range(1 if side else len(vs)):
                for i in range(frames):
                    L = pose(i)
                    if side:
                        panels = []
                        for u in range(len(vs)):
                            surface(u, L)
                            panels.append(shot())
                        shots.append(MV.tile(panels, names, background=bg))
                    else:
                        if several:
                            surface(v, L)
                        pic = shot()
                        shots.append(MV.caption(pic, names[v]) if names else pic)
                    if len(shots) % 10 == 0:
                        if len(shots) % 60 == 0:
                            self.flash(f"Recording… {len(shots)} of {total} frames", 30)
                        QApplication.processEvents()
        finally:
            cam.az, cam.el, cam.dist = keep
            r.light = keep_light
            r.material = keep_mat
            if keep_view is not None and r.data is not None:
                r.set_view_image(keep_view)
            r.update()
        if not shots:
            return ""
        try:
            write_video(shots, path, fps, max_mb=max_mb, repeat=repeat)
        except RuntimeError as e:
            QMessageBox.warning(self, "Record video", str(e))
            return ""
        times = f", {repeat} times round" if repeat > 1 and not path.lower().endswith(".gif") else ""
        self.flash(f"Recorded {os.path.basename(path)} ({len(shots)} frames{times}, "
                   f"{os.path.getsize(path) / 1e6:.1f} MB).")
        return path

    def show_shortcuts(self):
        ShortcutsSheet(self).exec()

    def show_about(self):
        try:
            from helio_build import stamp
            build = f"Build {stamp()}\n"
        except Exception:
            build = ""
        QMessageBox.about(
            self, "About Helio",
            f"Helio {__version__}\n{build}Photometric stereo for cultural heritage, by Scott Maloney.\n\n"
            "Capture, process, inspect, measure and publish, in five steps.\n\n"
            "Support Helio: buymeacoffee.com/scottboro")

    def show_help(self, section: str = ""):
        from helio_qt.help import HelpDialog
        self.help_dialog = HelpDialog(self, section)
        self.help_dialog.show()
        return self.help_dialog

    def show_welcome(self):
        from helio_qt.help import WelcomeDialog
        self.welcome_dialog = WelcomeDialog(self)
        self.welcome_dialog.show()
        return self.welcome_dialog

    def report_problem(self, path: str | None = None):
        from helio_qt.help import ReportDialog
        self.report_dialog = ReportDialog(self, path)
        self.report_dialog.show()
        return self.report_dialog

    def show_settings(self):
        d = QDialog(self)
        d.setWindowTitle("Settings")
        lay = QVBoxLayout(d)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.addWidget(label("Appearance", "section"))
        for mode, text in (("system", "Match the system"), ("light", "Light: Parchment"),
                           ("dark", "Dark: Night Dome")):
            r = QRadioButton(text)
            r.setChecked(self.appearance == mode)
            r.toggled.connect(lambda on, m=mode: on and self.set_appearance(m))
            lay.addWidget(r)
        d.exec()

    # ── window events ───────────────────────────────────────────────────
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.open_paths(paths)
            e.acceptProposedAction()

    def quit_app(self):
        """Quit Helio, offering first to save the capture's work."""
        if self.leave_capture("quit"):
            self._quit_asked = True
            self.close()

    def closeEvent(self, e):
        # The window's own close button (and the Dock's Quit) asks too; a
        # close from code has already asked, or has no one to ask.
        if e.spontaneous() and not getattr(self, "_quit_asked", False) \
                and not self.leave_capture("quit"):
            e.ignore()
            return
        self.prefs.setValue("geometry", self.saveGeometry())
        if getattr(self, "phone_share", None) is not None:
            self.phone_share.stop()                    # phones can no longer open it
        self.model3d.close()
        self.jobs.shutdown(timeout=60)
        self.thumbs.shutdown()
        self.pool.waitForDone(10000)
        self.tool_pool.waitForDone(10000)
        self.palette_dlg.close()
        super().closeEvent(e)
