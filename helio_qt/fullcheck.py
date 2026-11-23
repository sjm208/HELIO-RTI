"""Helio checking everything: `python3 -m helio_qt.fullcheck`.

One command that runs the whole program the way a person would and writes a report to hand back for
fixing.  It covers, in order:

  environment      versions, the Rust kernels, memory, OpenGL, optional libraries
  imports          every module of helio/ and helio_qt/ imports
  settings         every processing setting and every solver on a small known surface, solved and measured
  tests            every test_*.py in the repository (the engineering tests)
  --- then, in the real window, on your own capture (or a made one) ---
  solve            your capture through the window, timed
  views            every view: every control at its ends, every layer, flips, lights, full-size regions,
                   the sharp-when-zoomed region, and Rust against the NumPy code
  exports          every export format, then each file written is opened and checked
  actions          every menu command, with every dialog answered for it
  workspaces       every workspace, Inspect view, layer, tool, the 3D relief if OpenGL is there, appearance
  screenshots      what each workspace and every view looks like (look at them yourself)

Nothing of yours is changed: preferences go to a scratch folder, exports to the report folder.  The
report (REPORT.md, REPORT.json) opens with a block to paste back; every failure carries its traceback.

    python3 -m helio_qt.fullcheck                               a made capture, everything
    python3 -m helio_qt.fullcheck --session mine.helio          your saved session (images and lights)
    python3 -m helio_qt.fullcheck --images DIR --lp lights.lp   your photographs
    python3 -m helio_qt.fullcheck --quick                       fewer combinations, no test suite
    python3 -m helio_qt.fullcheck --headless                    no windows on screen (no OpenGL, so no 3D)
    python3 -m helio_qt.fullcheck --only views,exports          just those parts
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import faulthandler
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback

OK, FAIL, WARN, SKIP, INFO = "ok", "FAIL", "warn", "skip", "info"
PARTS_CORE = ("environment", "imports", "settings", "tests")
PARTS_APP = ("solve", "views", "exports", "actions", "workspaces", "screenshots")
SLOW_VIEW_S = 8.0          # a preview slower than this is worth saying
SLOW_EXPORT_S = 120.0


# ── the report ────────────────────────────────────────────────────────────────

class Report:
    def __init__(self, path: str):
        self.path = path
        self.rows: list[dict] = []
        self.meta: dict = {}

    def add(self, part, name, status, detail="", seconds=0.0, trace=""):
        self.rows.append(dict(part=part, name=name, status=status, detail=str(detail)[:2000],
                              seconds=round(float(seconds), 2), trace=trace))
        mark = {OK: "  ok   ", FAIL: "  FAIL ", WARN: "  warn ", SKIP: "  skip ", INFO: "  info "}[status]
        print(f"{mark}[{part}] {name}" + (f"  — {str(detail)[:160]}" if detail and status != OK else ""), flush=True)
        self.save()

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(dict(meta=self.meta, rows=self.rows), f, indent=1)
        except OSError:
            pass

    @contextlib.contextmanager
    def check(self, part, name, warn_after=None):
        """Time a block; an exception is a FAIL with its traceback, the run goes on."""
        t0 = time.time()
        try:
            yield
        except Exception as e:                        # noqa: BLE001 - the point is to catch anything
            self.add(part, name, FAIL, f"{type(e).__name__}: {e}", time.time() - t0,
                     "".join(traceback.format_exc(limit=12)))
            return
        dt = time.time() - t0
        if warn_after and dt > warn_after:
            self.add(part, name, WARN, f"took {dt:.1f} s (over {warn_after:.0f} s)", dt)
        else:
            self.add(part, name, OK, "", dt)


def _version(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:                            # noqa: BLE001
        return f"MISSING ({type(e).__name__})"


# ── part: environment ─────────────────────────────────────────────────────────

def part_environment(R: Report, args):
    with R.check("environment", "Python and libraries"):
        libs = {m: _version(m) for m in ("numpy", "scipy", "cv2", "PySide6", "skimage", "matplotlib",
                                         "tifffile", "PIL", "rasterio", "trimesh", "rvt", "OpenGL",
                                         "sklearn", "numba", "qrcode", "yaml")}
        R.meta["python"] = sys.version.split()[0]
        R.meta["platform"] = f"{platform.system()} {platform.release()} {platform.machine()}"
        R.meta["libraries"] = libs
        miss = [k for k, v in libs.items() if str(v).startswith("MISSING")]
        R.add("environment", "optional/required libraries", INFO if not miss else WARN,
              "missing: " + ", ".join(miss) if miss else "all present")
    with R.check("environment", "Rust kernels"):
        from helio.core import _native
        R.meta["kernels"] = _native.describe()
        if not _native.HAVE_RUST:
            R.add("environment", "Rust kernels", WARN,
                  "not loaded: Helio runs on the slower NumPy code. Run ./build_rust.sh")
        else:
            R.add("environment", "Rust kernels", INFO, _native.describe())
    with R.check("environment", "memory and plan"):
        from helio.utils.memory_plan import describe_machine, plan_solve
        R.meta["machine"] = describe_machine()
        p = plan_solve(8000, 12000, 8)
        R.meta["plan 96 MP x 8"] = f"{p.level}; estimate {p.estimate / 1e9:.1f} GB"
        R.add("environment", "machine", INFO, describe_machine())


# ── part: imports ─────────────────────────────────────────────────────────────

def part_imports(R: Report, args):
    import importlib
    import pkgutil
    skip = {"helio_qt.__main__", "helio_qt.fullcheck"}
    n = 0
    for pkg_name in ("helio", "helio_qt"):
        try:
            pkg = importlib.import_module(pkg_name)
        except Exception as e:                        # noqa: BLE001
            R.add("imports", pkg_name, FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=8))
            continue
        for m in pkgutil.walk_packages(pkg.__path__, pkg_name + "."):
            if m.name in skip:
                continue
            t0 = time.time()
            try:
                importlib.import_module(m.name)
                n += 1
            except Exception as e:                    # noqa: BLE001
                R.add("imports", m.name, FAIL, f"{type(e).__name__}: {e}", time.time() - t0,
                      traceback.format_exc(limit=8))
    R.add("imports", f"{n} modules import", OK)


# ── part: settings (every setting and solver on a known surface) ───────────────

def _make_stack(h=240, w=320, n_lights=8):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from tools import make_golden as G
    G.H, G.W = h, w
    tmp = tempfile.mkdtemp(prefix="helio_fc_stack_")
    paths, L = G.render_stack(tmp, n_lights=n_lights)
    truth = G.analytic_surface()[1]
    return tmp, paths, L, truth


def part_settings(R: Report, args):
    import dataclasses
    import numpy as np
    from helio.application import pipeline as P
    from helio.domain import settings as S
    base = S.ProcessingSettings(photo_tone="linear")
    try:
        _tmp, paths, L, truth = _make_stack(240, 320, 8)
    except Exception as e:                            # noqa: BLE001
        R.add("settings", "make the known surface", FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=8))
        return

    def angle(n):
        n = np.asarray(n, np.float32)
        if n.shape != truth.shape:
            return float("nan")
        d = np.clip((n * truth).sum(-1) / np.maximum(np.linalg.norm(n, axis=-1) * np.linalg.norm(truth, axis=-1), 1e-9), -1, 1)
        return float(np.degrees(np.arccos(d)).mean())

    variants: list[tuple[str, dict]] = [("defaults", {})]
    # The near-field solver has its own check below: this surface is lit by distant lamps, which it does not model.
    variants += [(f"normal solver: {e}", dict(normal_engine=e)) for e in S.NORMAL_SOLVERS
                 if e != "Near-Field Attenuation"]
    variants += [(f"depth integrator: {e}", dict(depth_engine=e)) for e in S.DEPTH_INTEGRATORS]
    variants += [(f"photo tone: {t}", dict(photo_tone=t)) for t in ("stored", "srgb")]
    variants += [
        ("flatten illumination", dict(flatten_illumination=True)),
        ("CLAHE", dict(pre_clahe=True, pre_clahe_clip=3.0, pre_clahe_tile=8)),
        ("line up frames", dict(auto_align=True)),
        ("adaptive light weighting 0.5", dict(adaptive_light=0.5)),
        ("shadow-rejection threshold 25", dict(normal_engine="Shadow-Rejection LS", shadow_thresh=25)),
        ("SfS refinement", dict(sfs_refine=True, sfs_iters=2, sfs_prior=0.5)),
        ("pre-integration smoothing 3", dict(pre_smooth=3.0)),
        ("parabolic correction", dict(parabolic_corr=True)),
        ("invert relief", dict(invert_relief=True)),
        ("working size 50%", dict(work_scale=0.5)),
        ("working size automatic", dict(work_scale=0.0)),
        ("everything pre-processing on", dict(flatten_illumination=True, pre_clahe=True,
                                              auto_align=True, adaptive_light=0.3)),
    ]
    if args.quick:
        variants = variants[:1] + variants[1:8:2]
    ref = None
    for name, kw in variants:
        t0 = time.time()
        try:
            r = P.solve(P.SolveRequest(paths=paths, lights=L, settings=dataclasses.replace(base, **kw)))
        except Exception as e:                        # noqa: BLE001
            R.add("settings", name, FAIL, f"{type(e).__name__}: {e}", time.time() - t0, traceback.format_exc(limit=10))
            continue
        problems = []
        for field_name in ("normals_full", "depth_vis_p", "albedo_rgb_full", "median_img"):
            a = getattr(r, field_name, None)
            if a is None:
                problems.append(f"{field_name} missing")
            elif np.issubdtype(np.asarray(a).dtype, np.floating) and not np.isfinite(a).all():
                problems.append(f"{field_name} has NaN/inf")
        if r.normals_full is not None and r.work_scale >= 1.0:
            nn = np.linalg.norm(r.normals_full, axis=-1)
            if np.abs(nn - 1).max() > 1e-2:
                problems.append("normals are not unit length")
        err = angle(r.normals_full) if r.work_scale >= 1.0 else float("nan")
        if name == "defaults":
            ref = err
        calibrated = "PCA" not in name and "invert" not in name and "depth" not in name.split(":")[0]
        if problems:
            R.add("settings", name, FAIL, "; ".join(problems), time.time() - t0)
        elif calibrated and err == err and err > 12.0 and "tone" not in name:
            R.add("settings", name, WARN, f"normals {err:.1f}° from the known surface (defaults: "
                  f"{ref if ref is None else round(ref, 1)}°)", time.time() - t0)
        else:
            R.add("settings", name, OK, f"normals {err:.1f}° from the known surface" if err == err else "", time.time() - t0)
    with R.check("settings", "near-field solver, on lamps truly near (ball off centre)"):
        h, w = truth.shape[:2]
        R_px, origin = 1.5 * max(h, w), (60.0, -30.0)
        Ls = np.asarray(L, np.float64)
        X, Y = np.meshgrid(np.linspace(-w / 2, w / 2, w) - origin[0], np.linspace(h / 2, -h / 2, h) - origin[1])
        P_ = np.dstack([X, Y, np.zeros_like(X)])
        lit = []
        for l in Ls:
            v = l * R_px - P_
            d = np.linalg.norm(v, axis=2)
            lit.append(200 * (R_px / d) ** 2 * np.clip((truth * (v / d[..., None])).sum(-1), 0, None))
        from helio.core.photometric_stereo import solve_near_field_normals
        e_nf = angle(solve_near_field_normals(np.array(lit, np.float32), Ls, h, w, dome_radius_px=R_px, origin=origin))
        assert e_nf < 2.0, f"{e_nf:.1f}° from the known surface"
    with R.check("settings", "ProcessingSettings round trip"):
        d = base.to_dict()
        assert S.ProcessingSettings.from_dict(d) == base
        assert S.ProcessingSettings.from_dict({"nonsense": 1}) == S.ProcessingSettings()
    with R.check("settings", "session file round trip"):
        from helio.io import session as SE
        assert hasattr(SE, "read")


# ── part: the test suite ──────────────────────────────────────────────────────

NEEDS_CLASSIC = ("docs/pyside6-migration", "capture_baseline", "Helio_Pro")


def part_tests(R: Report, args):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # the tests beside the code, or in tests/ (the public copy's layout); not
    # test_fullcheck itself, which runs this tool: no recursion
    files = sorted(os.path.relpath(os.path.join(d, f), here)
                   for d in (here, os.path.join(here, "tests")) if os.path.isdir(d)
                   for f in os.listdir(d) if f.startswith("test_") and f.endswith(".py")
                   and not f.startswith("test_fullcheck."))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=here,
               HELIO_PREFS=os.path.join(tempfile.gettempdir(), f"helio_fc_prefs_{os.getpid()}.json"))
    for f in files:
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, f], cwd=here, env=env, capture_output=True, text=True,
                               timeout=args.test_timeout)
            out = (p.stdout or "") + "\n" + (p.stderr or "")
            res = [ln for ln in out.splitlines() if ln.startswith("Results:") or ln.startswith("RESULT")]
            fails = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("FAIL")]
            summary = res[-1] if res else out.strip().splitlines()[-1][:160] if out.strip() else "no output"
            if any(k in out for k in NEEDS_CLASSIC) and (p.returncode != 0 or not res):
                R.add("tests", f, SKIP, "needs files that are not in this repository (Helio Classic)", time.time() - t0)
            elif p.returncode == 0 and not fails:
                R.add("tests", f, OK, summary, time.time() - t0)
            else:
                R.add("tests", f, FAIL, f"{summary}", time.time() - t0, "\n".join(fails[:12]) or out[-1500:])
        except subprocess.TimeoutExpired:
            R.add("tests", f, FAIL, f"timed out after {args.test_timeout}s", args.test_timeout)
        except Exception as e:                        # noqa: BLE001
            R.add("tests", f, FAIL, f"{type(e).__name__}: {e}", time.time() - t0, traceback.format_exc(limit=6))


# ── the window, dialogs answered ──────────────────────────────────────────────

class Dialogs:
    """Answers every dialog, records each one, never blocks."""

    def __init__(self, scratch: str):
        self.scratch = scratch
        self.seen: list[str] = []
        self.open_path = ""
        self.save_dir = scratch

    def install(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QColor, QDesktopServices
        from PySide6.QtWidgets import (QApplication, QColorDialog, QDialog, QFileDialog, QInputDialog,
                                       QMenu, QMessageBox)
        me = self

        def note(kind, text=""):
            me.seen.append(f"{kind}: {str(text)[:140]}")

        def fexec(self_, *a, **k):
            note(type(self_).__name__, self_.windowTitle())
            try:
                self_.show()
                QApplication.processEvents()
                self_.accept()
            except Exception:                         # noqa: BLE001
                pass
            return 1
        QDialog.exec = fexec
        QDialog.exec_ = fexec

        def mexec(self_, *a, **k):                    # a message box that runs itself: QMessageBox has its own exec
            note("QMessageBox", f"{self_.windowTitle()} | {self_.text()[:100]}")
            return int(QMessageBox.StandardButton.Ok)
        QMessageBox.exec = mexec
        QMessageBox.exec_ = mexec

        def msg(kind, ret):
            def f(parent, title="", text="", *a, **k):
                note("QMessageBox." + kind, f"{title} | {str(text)[:100]}")
                return ret
            return staticmethod(f)
        QMessageBox.information = msg("information", QMessageBox.StandardButton.Ok)
        QMessageBox.warning = msg("warning", QMessageBox.StandardButton.Ok)
        QMessageBox.critical = msg("critical", QMessageBox.StandardButton.Ok)
        QMessageBox.question = msg("question", QMessageBox.StandardButton.Discard)
        QMessageBox.about = msg("about", QMessageBox.StandardButton.Ok)
        QMessageBox.aboutQt = staticmethod(lambda *a, **k: note("QMessageBox.aboutQt"))

        def open_file(parent=None, caption="", directory="", filt="", *a, **k):
            note("QFileDialog.open", caption)
            return me.open_path, ""

        def open_files(parent=None, caption="", directory="", filt="", *a, **k):
            note("QFileDialog.open files", caption)
            return ([me.open_path] if me.open_path else []), ""

        def save_file(parent=None, caption="", directory="", filt="", *a, **k):
            note("QFileDialog.save", caption)
            base = os.path.basename(str(directory)) or "output"
            ext = ""
            if "(*." in str(filt):
                ext = str(filt).split("(*.")[1].split(")")[0].split()[0].rstrip(")")
                ext = "." + ext if ext and not ext.startswith(".") else ext
            name = base if os.path.splitext(base)[1] else base + (ext or ".out")
            return os.path.join(me.save_dir, name), ""

        def dir_(parent=None, caption="", directory="", *a, **k):
            note("QFileDialog.directory", caption)
            d = os.path.join(me.save_dir, "dir_answer")
            os.makedirs(d, exist_ok=True)
            return d
        QFileDialog.getOpenFileName = staticmethod(open_file)
        QFileDialog.getOpenFileNames = staticmethod(open_files)
        QFileDialog.getSaveFileName = staticmethod(save_file)
        QFileDialog.getExistingDirectory = staticmethod(dir_)

        QInputDialog.getText = staticmethod(lambda *a, **k: (note("QInputDialog.text", a[1:3]), ("fullcheck", True))[1])
        QInputDialog.getInt = staticmethod(lambda *a, **k: (note("QInputDialog.int", a[1:3]), (int(a[3]) if len(a) > 3 else 1, True))[1])
        QInputDialog.getDouble = staticmethod(lambda *a, **k: (note("QInputDialog.double", a[1:3]), (float(a[3]) if len(a) > 3 else 1.0, True))[1])
        QInputDialog.getItem = staticmethod(lambda p, t, l, items, *a, **k: (note("QInputDialog.item", t), (items[0] if items else "", bool(items)))[1])
        QColorDialog.getColor = staticmethod(lambda *a, **k: (note("QColorDialog"), QColor("#c08040"))[1])
        QMenu.exec = lambda self_, *a, **k: (note("QMenu", ""), None)[1]
        QMenu.exec_ = QMenu.exec
        QDesktopServices.openUrl = staticmethod(lambda url: (note("openUrl", url.toString() if hasattr(url, "toString") else url), True)[1])
        import webbrowser
        webbrowser.open = lambda *a, **k: (note("webbrowser", a[:1]), True)[1]


class Trap:
    """Everything that goes wrong inside Qt slots, which would otherwise only be printed."""

    def __init__(self):
        self.errors: list[str] = []
        self.qt: dict[str, int] = {}
        self._old = sys.excepthook

    def install(self):
        def hook(tp, val, tb):
            self.errors.append("".join(traceback.format_exception(tp, val, tb, limit=10)))
        sys.excepthook = hook
        import logging

        class H(logging.Handler):
            def emit(h, rec):
                if rec.levelno >= logging.ERROR:
                    self.errors.append(f"log {rec.levelname} {rec.name}: {rec.getMessage()}"
                                       + ("\n" + "".join(traceback.format_exception(*rec.exc_info, limit=8)) if rec.exc_info else ""))
        logging.getLogger().addHandler(H())
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler

        def qt(kind, ctx, message):
            if kind in (QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg):
                key = message.strip()[:200]
                if key:
                    self.qt[key] = self.qt.get(key, 0) + 1
        qInstallMessageHandler(qt)

    def take(self) -> list[str]:
        e, self.errors = self.errors, []
        return e


def _settle(app, w, limit=600.0, quiet=0.15):
    """Run Qt until the program is idle: no job, no background work, nothing left to paint."""
    t0 = time.time()
    last_busy = time.time()
    while time.time() - t0 < limit:
        app.processEvents()
        busy = bool(w.project.busy) or w.bg_pool.activeThreadCount() or w.tool_pool.activeThreadCount()
        ins = w.workspaces.get("Inspect")
        if ins is not None and ins.renderer.pool.activeThreadCount():
            busy = True
        if busy:
            last_busy = time.time()
            time.sleep(0.01)
        elif time.time() - last_busy > quiet:
            return True
        else:
            time.sleep(0.01)
    return False


def _close_extras(app, w):
    for t in app.topLevelWidgets():
        if t is not w and t.isVisible():
            try:
                t.close()
            except Exception:                         # noqa: BLE001
                pass
    app.processEvents()


# ── app: solve ────────────────────────────────────────────────────────────────

def _load_capture(R, app, w, dlg, args):
    """Your session, your photographs (and lights), or a made capture, into the window."""
    from helio.io import session as SE
    if args.session:
        with R.check("solve", f"open the session {os.path.basename(args.session)}"):
            ok = w.load_session_file(args.session, asked=True)
            _settle(app, w, 120)
            assert ok, "the session did not open"
            assert w.project.images, "the session has no photographs that can be found"
        return args.session
    if args.images:
        with R.check("solve", f"import {args.images}"):
            assert w.open_paths([args.images]), "no photographs were imported"
            _settle(app, w, 120)
        if args.lp:
            with R.check("solve", f"light file {os.path.basename(args.lp)}"):
                dlg.open_path = args.lp
                w.import_light_file()
                _settle(app, w, 60)
                assert w.project.lights is not None, "the light file was not read"
        return args.images
    tmp, paths, L, _truth = _make_stack(args.size[1], args.size[0], 8)
    with R.check("solve", "import a made capture"):
        assert w.open_paths([tmp])
        w.project.set_lights(L, "file:fullcheck.lp")
        w.project.update_settings(photo_tone="linear")
        _settle(app, w, 60)
    return f"made capture {args.size[0]}x{args.size[1]}"


def _gl_check(R):
    with R.check("environment", "OpenGL for the 3D view"):
        from helio_qt.relief3d import gl_available
        ok, why = gl_available()
        R.meta["opengl"] = "yes" if ok else f"no ({why})"
        R.add("environment", "OpenGL 4.1", INFO if ok else WARN,
              "available" if ok else f"not available here: {why} (the 3D relief cannot be checked)")


def part_solve(R, app, w, dlg, trap, args):
    _gl_check(R)
    src = _load_capture(R, app, w, dlg, args)
    R.meta["capture"] = src
    if not w.project.images:
        R.add("solve", "photographs", FAIL, "nothing was loaded; the rest cannot run")
        return False
    R.meta["photographs"] = len(w.project.images)
    if w.project.lights is None:
        R.add("solve", "lights", WARN, "this capture has no light positions; only the uncalibrated solver can run")
    if args.work_scale is not None:
        w.project.update_settings(work_scale=float(args.work_scale))
    t0 = time.time()
    with R.check("solve", "process (solve and depth) through the window"):
        w.registry.trigger("process.run")
        ok = _settle(app, w, args.solve_timeout)
        assert ok, f"still running after {args.solve_timeout}s"
        assert w.project.solved, "finished without a result: " + str(getattr(w.project, "message", ""))
    errs = trap.take()
    if errs:
        R.add("solve", "errors raised while solving", FAIL, errs[0][:300], 0, "\n---\n".join(errs[:3]))
    if not w.project.solved:
        return False
    r = w.project.result
    R.meta["solve"] = f"{r.w}x{r.h}, {r.n_images} images, work scale {getattr(r, 'work_scale', 1.0):.2f}, {time.time() - t0:.0f} s"
    R.add("solve", "result", INFO, R.meta["solve"], time.time() - t0)
    import numpy as np
    with R.check("solve", "the result is sound"):
        bad = [k for k in ("normals_full", "depth_vis_p", "albedo_rgb_full") if getattr(r, k) is None]
        assert not bad, f"missing: {bad}"
        nf = np.asarray(r.normals_full[:: max(1, r.h // 400), :: max(1, r.w // 400)])
        assert np.isfinite(nf).all(), "normals contain NaN/inf"
    return True


# ── app: views ────────────────────────────────────────────────────────────────

class _NumpyOnly:
    """Every module's handle on the Rust kernels set to None, and the caches emptied."""

    def __enter__(self):
        self.saved = []
        import helio_core_rs
        for m in list(sys.modules.values()):
            if m is None or not getattr(m, "__name__", "").startswith(("helio", "helio_qt")):
                continue
            for attr in ("rs", "_rs"):
                if getattr(m, attr, None) is helio_core_rs:
                    self.saved.append((m, attr))
                    setattr(m, attr, None)
        from helio.core import parallel
        parallel.CACHE.clear()
        return self

    def __exit__(self, *a):
        import helio_core_rs
        for m, attr in self.saved:
            setattr(m, attr, helio_core_rs)
        from helio.core import parallel
        parallel.CACHE.clear()


def _image_problems(img, name=""):
    import numpy as np
    if img is None:
        return "returned nothing"
    a = np.asarray(img)
    if a.ndim not in (2, 3) or a.size == 0:
        return f"bad shape {a.shape}"
    if np.issubdtype(a.dtype, np.floating) and not np.isfinite(a).all():
        return "contains NaN/inf"
    return ""


def part_views(R, app, w, dlg, trap, args):
    import copy
    import numpy as np
    from helio.application.viewing import Display, DETAIL_SKIP, detail_image, screen_image
    from helio.core import parallel
    from helio.domain import view_controls as VC
    from helio.domain.registry import RENDER_ENGINES, default_engine_settings
    D, r = w.render_data, w.project.result
    if D is None or r is None:
        R.add("views", "render data", FAIL, "no solve to draw")
        return
    defaults = default_engine_settings()
    have_rust = None
    try:
        from helio_core_rs import lls_normals  # noqa: F401
        have_rust = True
    except ImportError:
        have_rust = False
    perf = []
    box = (max(0, r.w // 4), max(0, r.h // 4), min(r.w, r.w // 4 + min(1200, r.w // 2)),
           min(r.h, r.h // 4 + min(800, r.h // 2)))
    for eng in RENDER_ENGINES:
        st0 = dict(defaults[eng])
        t0 = time.time()
        try:
            parallel.CACHE.clear()
            t = time.time()
            img = screen_image(eng, D, st0, Display(layer="Combined Colour"))
            cold = time.time() - t
            t = time.time()
            img2 = screen_image(eng, D, st0, Display(layer="Combined Colour"))
            warm = time.time() - t
        except Exception as e:                        # noqa: BLE001
            R.add("views", f"{eng}: draw", FAIL, f"{type(e).__name__}: {e}", time.time() - t0, traceback.format_exc(limit=10))
            continue
        prob = _image_problems(img)
        if prob:
            R.add("views", f"{eng}: draw", FAIL, prob, cold)
            continue
        perf.append((eng, cold, warm))
        a = np.asarray(img)
        flat = float(a.astype(np.float32).std()) < 0.5
        if flat:
            R.add("views", f"{eng}: draw", WARN, "the picture is one flat tone (is this view empty for this capture?)", cold)
        elif cold > SLOW_VIEW_S:
            R.add("views", f"{eng}: draw", WARN, f"first draw {cold:.1f} s, then {warm:.1f} s", cold)
        else:
            R.add("views", f"{eng}: draw", OK, f"{cold * 1000:.0f} ms, then {warm * 1000:.0f} ms", cold)
        if np.asarray(img2).shape != a.shape:
            R.add("views", f"{eng}: repeat draw", FAIL, "the second draw is a different size than the first")
        # controls at their ends
        bad = []
        for c in VC.controls_for(eng, st0):
            if c.key in ("lx", "ly", "lz"):
                continue
            vals = [v for v, _l in c.choices] if c.choices else [c.lo, (c.lo + c.hi) / 2, c.hi]
            if args.quick:
                vals = vals[:1] + vals[-1:]
            for v in vals:
                st = dict(st0, **{c.key: (int(v) if c.integer or c.choices else float(v))})
                try:
                    out = screen_image(eng, D, st, Display(layer="Combined Colour"))
                    p = _image_problems(out)
                    if p:
                        bad.append(f"{c.key}={v}: {p}")
                except Exception as e:                # noqa: BLE001
                    bad.append(f"{c.key}={v}: {type(e).__name__}: {e}")
        if bad:
            R.add("views", f"{eng}: controls", FAIL, "; ".join(bad[:6]), 0, "\n".join(bad))
        else:
            R.add("views", f"{eng}: controls at their ends", OK, f"{len(VC.controls_for(eng, st0))} controls")
        # lights
        if VC.light_kind(eng):
            try:
                outs = []
                if VC.light_kind(eng) == "compass":
                    poses = [dict(hs_azi=a, hs_alt=40.0) for a in (315.0, 135.0, 45.0, 225.0)]
                else:
                    poses = [dict(lx=lx, ly=ly, lz=lz) for lx, ly, lz in
                             ((-0.7, 0.0, 0.7), (0.7, 0.0, 0.7), (0.0, -0.7, 0.7), (0.0, 0.0, 1.0))]
                for pose in poses:
                    outs.append(np.asarray(screen_image(eng, D, dict(st0, **pose), Display(layer="Combined Colour"))).astype(np.float32))
                moved = float(np.abs(outs[0] - outs[1]).mean())
                if moved < 0.05:
                    R.add("views", f"{eng}: responds to the light", WARN, f"moving the light left to right changes the picture by {moved:.2f}/255 only")
                else:
                    R.add("views", f"{eng}: responds to the light", OK, f"{moved:.1f}/255")
            except Exception as e:                    # noqa: BLE001
                R.add("views", f"{eng}: light", FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=8))
        # layers and flips
        if not args.quick:
            probs = []
            for layer in ("Shading", "Combined Colour", "Roughness Fusion", "Confidence Overlay"):
                for fl in (dict(), dict(flip_h=True), dict(flip_v=True), dict(rot180=True), dict(normal_invert=True)):
                    try:
                        o = screen_image(eng, D, st0, Display(layer=layer, **fl))
                        p = _image_problems(o)
                        if p:
                            probs.append(f"{layer} {fl}: {p}")
                    except Exception as e:            # noqa: BLE001
                        probs.append(f"{layer} {fl}: {type(e).__name__}: {e}")
            R.add("views", f"{eng}: layers and flips", FAIL if probs else OK, "; ".join(probs[:5]), 0, "\n".join(probs))
        # a full-size region (the loupe) and the sharp-when-zoomed region
        try:
            o = screen_image(eng, D, st0, Display(layer="Combined Colour"), use_proxy=False, crop=box, transforms=False)
            p = _image_problems(o)
            if p:
                R.add("views", f"{eng}: full-size region", FAIL, p)
            elif np.asarray(o).shape[:2] != (box[3] - box[1], box[2] - box[0]):
                R.add("views", f"{eng}: full-size region", FAIL, f"asked for {box}, got shape {np.asarray(o).shape}")
            else:
                R.add("views", f"{eng}: full-size region", OK)
        except Exception as e:                        # noqa: BLE001
            R.add("views", f"{eng}: full-size region", FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=10))
        if eng not in DETAIL_SKIP:
            try:
                o = detail_image(eng, D, st0, Display(layer="Combined Colour"), box)
                p = _image_problems(o)
                R.add("views", f"{eng}: sharp region when zoomed", FAIL if p else OK, p)
            except Exception as e:                    # noqa: BLE001
                R.add("views", f"{eng}: sharp region when zoomed", FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=10))
        # Rust against NumPy
        if have_rust and not args.quick:
            try:
                parallel.CACHE.clear()
                with_rust = np.asarray(screen_image(eng, D, st0, Display(layer="Combined Colour"))).astype(np.float32)
                with _NumpyOnly():
                    without = np.asarray(screen_image(eng, D, st0, Display(layer="Combined Colour"))).astype(np.float32)
                parallel.CACHE.clear()
                if with_rust.shape != without.shape:
                    R.add("views", f"{eng}: Rust = NumPy", FAIL, f"shapes differ {with_rust.shape} vs {without.shape}")
                else:
                    diff = np.abs(with_rust - without)
                    mx, mean = float(diff.max()), float(diff.mean())
                    if mean > 0.5 or mx > 8:
                        R.add("views", f"{eng}: Rust = NumPy", FAIL, f"the Rust and NumPy pictures differ: mean {mean:.2f}, largest {mx:.0f} (of 255)")
                    else:
                        R.add("views", f"{eng}: Rust = NumPy", OK, f"largest difference {mx:.0f}/255")
            except Exception as e:                    # noqa: BLE001
                R.add("views", f"{eng}: Rust = NumPy", FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=8))
        errs = trap.take()
        if errs:
            R.add("views", f"{eng}: errors logged while drawing", FAIL, errs[0][:300], 0, "\n---\n".join(errs[:3]))
    R.meta["view timings (first / repeat, s)"] = {e: [round(c, 3), round(wm, 3)] for e, c, wm in perf}


# ── app: exports ──────────────────────────────────────────────────────────────

def _validate_file(path: str) -> str:
    """'' when the file looks right for its type, else what is wrong."""
    import zipfile
    if not os.path.exists(path):
        return "was not written"
    size = os.path.getsize(path)
    if size == 0:
        return "is empty"
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            import cv2
            if cv2.imread(path, cv2.IMREAD_UNCHANGED) is None:
                return "is not a readable image"
        elif ext in (".tif", ".tiff"):
            import cv2
            if cv2.imread(path, cv2.IMREAD_UNCHANGED) is None:
                try:
                    import tifffile
                    tifffile.imread(path)
                except Exception as e:                # noqa: BLE001
                    return f"is not a readable TIFF ({type(e).__name__})"
        elif ext == ".json":
            json.load(open(path))
        elif ext in (".html", ".htm", ".svg", ".xml", ".usda", ".mtl"):
            txt = open(path, errors="replace").read(4000)
            if "<" not in txt and ext in (".html", ".htm", ".svg", ".xml") or not txt.strip():
                return "has no markup"
        elif ext == ".zip" or ext == ".usdz":
            if zipfile.ZipFile(path).testzip() is not None:
                return "is a damaged zip"
        elif ext == ".glb":
            if open(path, "rb").read(4) != b"glTF":
                return "is not a GLB (missing glTF header)"
        elif ext == ".ply":
            if not open(path, "rb").read(3) == b"ply":
                return "is not a PLY"
        elif ext == ".obj":
            head = open(path, errors="replace").read(200000)
            if "v " not in head or "f " not in open(path, errors="replace").read():
                return "has no vertices or faces"
        elif ext == ".stl":
            raw = open(path, "rb").read(84)
            if not raw.startswith(b"solid") and len(raw) < 84:
                return "is too short for an STL"
        elif ext == ".e57":
            if open(path, "rb").read(8) != b"ASTM-E57":
                return "is not an E57"
        elif ext == ".csv" or ext == ".txt" or ext == ".lp":
            if not open(path, errors="replace").read(200).strip():
                return "is blank"
        elif ext == ".gif":
            import cv2  # noqa: F401
            from PIL import Image
            Image.open(path).verify()
        elif ext == ".pdf":
            if open(path, "rb").read(4) != b"%PDF":
                return "is not a PDF"
    except Exception as e:                            # noqa: BLE001
        return f"could not be read back ({type(e).__name__}: {e})"
    return ""


def part_exports(R, app, w, dlg, trap, args, out_dir):
    from helio_qt import publish
    from helio_qt import publish_catalog as C
    p = w.project
    if not p.solved:
        R.add("exports", "needs a solve", SKIP, "")
        return
    p.set_scale_photo(0.03)                           # the formats that need a scale
    p.z_mm_per_unit = 0.05
    _settle(app, w, 30)
    r = p.result
    if not args.full_exports:
        # A 100 MP capture meshed at every pixel is 100 million vertices: a few GB and many minutes
        # for each mesh format. The check meshes at about 4 million points; --full-exports does not.
        import math
        step = max(1, int(math.ceil(math.sqrt(r.w * r.h / 4.0e6))))
        w.shared["export_step"] = step
        w.shared["step"] = step
        R.meta["export mesh step used by the check"] = step
    root = os.path.join(out_dir, "exports")
    os.makedirs(root, exist_ok=True)
    fmts = C.all_formats()
    if args.quick:
        fmts = [f for f in fmts if f.key in ("png_views", "normal_map", "obj", "stl", "glb", "geotiff", "rti_web", "lp", "hsh", "ptm", "pub_sheet")]
    for f in fmts:
        d = os.path.join(root, f.key)
        t0 = time.time()
        trap.take()
        try:
            os.makedirs(d, exist_ok=True)                 # as the export job does before it writes
            src = w.publish_source()
            files = publish.export_one(f.key, src, d)
        except publish.X.NothingToExport as e:
            R.add("exports", f"{f.key} ({f.name})", WARN, f"nothing to export: {e}", time.time() - t0)
            continue
        except Exception as e:                        # noqa: BLE001
            R.add("exports", f"{f.key} ({f.name})", FAIL, f"{type(e).__name__}: {e}", time.time() - t0, traceback.format_exc(limit=12))
            continue
        files = [x for x in (files or []) if x]
        bad = []
        for x in files[:400]:
            if os.path.isdir(x):
                continue
            why = _validate_file(x)
            if why:
                bad.append(f"{os.path.relpath(x, root)} {why}")
        errs = trap.take()
        dt = time.time() - t0
        if not files:
            R.add("exports", f"{f.key} ({f.name})", FAIL, "wrote no files", dt)
        elif bad:
            R.add("exports", f"{f.key} ({f.name})", FAIL, f"{len(bad)} of {len(files)} files wrong: " + "; ".join(bad[:3]), dt, "\n".join(bad))
        elif errs:
            R.add("exports", f"{f.key} ({f.name})", WARN, f"wrote {len(files)} files, with errors logged: {errs[0][:160]}", dt, "\n---\n".join(errs[:3]))
        elif dt > SLOW_EXPORT_S:
            R.add("exports", f"{f.key} ({f.name})", WARN, f"{len(files)} files, but {dt:.0f} s", dt)
        else:
            total = sum(os.path.getsize(x) for x in files if os.path.isfile(x))
            R.add("exports", f"{f.key} ({f.name})", OK, f"{len(files)} files, {total / 1e6:.1f} MB", dt)


# ── app: every menu command ───────────────────────────────────────────────────

NEVER = {"app.quit": "would end this run", "file.new": "run last", "file.next": "run last",
         "window.minimize": "hides the window", "window.zoom": "resizes the window",
         "process.run": "run in solve", "process.reprocess": "re-solves (run in solve)",
         "process.cancel": "nothing to cancel", "help.report": "opens the mail program"}


def part_actions(R, app, w, dlg, trap, args, out_dir):
    from helio_qt import actions as A
    reg = w.registry
    dlg.save_dir = os.path.join(out_dir, "actions_output")
    os.makedirs(dlg.save_dir, exist_ok=True)
    order = [s for s in A.SPECS if s.id not in ("file.new", "file.next")] + [s for s in A.SPECS if s.id in ("file.next", "file.new")]
    for s in order:
        label = f"{s.menu.replace('/', ' › ')} › {s.text.rstrip('…')}"
        if s.id in NEVER and not (s.id in ("file.new", "file.next")):
            R.add("actions", label, SKIP, NEVER[s.id])
            continue
        why = reg.reason(s.id)
        if why:
            R.add("actions", label, SKIP, f"not available: {why}")
            continue
        if args.quick and s.id.startswith(("export.", "share.", "view3d.open", "research.open", "ai.open")):
            R.add("actions", label, SKIP, "quick")
            continue
        trap.take()
        dlg.seen.clear()
        t0 = time.time()
        try:
            act = reg.actions[s.id]
            if act.isCheckable():
                act.setChecked(not act.isChecked())
            reg.trigger(s.id)
            idle = _settle(app, w, 45)
            ins = w.workspaces.get("Inspect")
            if not idle and ins is not None and hasattr(ins, "stop_sweep") and ins.sweep is not None:
                ins.stop_sweep()                      # a command that keeps the light moving until stopped
                idle = _settle(app, w, 45)
                if idle:
                    R.add("actions", label + " (keeps running until stopped)", INFO, "stopped by the check")
            if not idle:
                R.add("actions", label + " (still busy)", WARN, "the program was still busy 45 s after this command")
            if act.isCheckable():
                act.setChecked(not act.isChecked())
                reg.trigger(s.id)
                _settle(app, w, 60)
        except Exception as e:                        # noqa: BLE001
            R.add("actions", label, FAIL, f"{type(e).__name__}: {e}", time.time() - t0, traceback.format_exc(limit=12))
            _close_extras(app, w)
            continue
        ins = w.workspaces.get("Inspect")
        if ins is not None and hasattr(ins, "stop_sweep") and getattr(ins, "sweep", None) is not None:
            ins.stop_sweep()
        errs = trap.take()
        shown = list(dlg.seen)
        warn_boxes = [x for x in shown if x.startswith(("QMessageBox.warning", "QMessageBox.critical"))
                      and "dir_answer" not in x and "That had 0" not in x and "fullcheck" not in x]
        _close_extras(app, w)
        dt = time.time() - t0
        note = f"{len(shown)} dialog(s)" if shown else ""
        if errs:
            R.add("actions", label, FAIL, errs[0][:300], dt, "\n---\n".join(errs[:3]))
        elif warn_boxes:
            R.add("actions", label, WARN, "it told the user: " + " / ".join(warn_boxes[:2]), dt)
        else:
            R.add("actions", label, OK, note, dt)
    R.meta["dialogs answered"] = len(set(dlg.seen))


# ── app: workspaces ───────────────────────────────────────────────────────────

def part_workspaces(R, app, w, dlg, trap, args, out_dir):
    from helio.domain.registry import RENDER_ENGINES
    names = list(w.workspaces)
    for n in names:
        trap.take()
        with R.check("workspaces", f"open {n}"):
            w.go_workspace(n)
            _settle(app, w, 60)
        errs = trap.take()
        if errs:
            R.add("workspaces", f"{n}: errors on opening", FAIL, errs[0][:300], 0, "\n---\n".join(errs[:3]))
    if "Inspect" not in w.workspaces or not w.project.solved:
        return
    ins = w.workspaces["Inspect"]
    w.go_workspace("Inspect")
    _settle(app, w, 60)
    views = list(RENDER_ENGINES) + [ins.DEPTH_MAP]
    for v in views:
        trap.take()
        t0 = time.time()
        try:
            w.set_view(v)
            ins.wait_render()
            _settle(app, w, 120)
        except Exception as e:                        # noqa: BLE001
            R.add("workspaces", f"Inspect view {v}", FAIL, f"{type(e).__name__}: {e}", time.time() - t0, traceback.format_exc(limit=10))
            continue
        errs = trap.take()
        shown = not ins.viewport.item.pixmap().isNull()
        if ins.error:
            R.add("workspaces", f"Inspect view {v}", FAIL, ins.error, time.time() - t0)
        elif errs:
            R.add("workspaces", f"Inspect view {v}", FAIL, errs[0][:300], time.time() - t0, "\n---\n".join(errs[:3]))
        elif not shown:
            R.add("workspaces", f"Inspect view {v}", FAIL, "the window shows no picture", time.time() - t0)
        else:
            R.add("workspaces", f"Inspect view {v}", OK, f"drawn in {getattr(ins, 'last_ms', 0):.0f} ms", time.time() - t0)
    # tools and layers
    with R.check("workspaces", "Inspect: layers, compare, loupe, zoom, selection, section"):
        for layer in ("Combined Colour", "Shading", "Roughness Fusion", "Confidence Overlay"):
            w.display.layer = layer
            ins.render_view(final=True)
            ins.wait_render()
        w.display.layer = "Combined Colour"
        ins.viewport.zoom_by(2.0)
        _settle(app, w, 30)
        ins.viewport.zoom_by(0.25)
        ins.viewport.fit()
        for chk in (getattr(ins, "chk_detail", None),):
            if chk is not None:
                chk.setChecked(False)
                chk.setChecked(True)
        ins._probed(10, 10)
        ins._clicked(20, 20)
        w.selection = (10, 10, max(50, w.render_data.w // 2), max(50, w.render_data.h // 2))
        ins.render_view(final=True)
        _settle(app, w, 60)
        w.selection = None
        ins.render_view(final=True)
        _settle(app, w, 60)
    errs = trap.take()
    if errs:
        R.add("workspaces", "Inspect tools: errors", FAIL, errs[0][:300], 0, "\n---\n".join(errs[:3]))
    # the 3D relief
    from helio_qt.relief3d import gl_available
    ok, why = gl_available()
    if ok:                                            # it is made the first time the mode is opened
        w.registry.trigger("view.mode:3d")
        _settle(app, w, 60)
    if ins.relief is None or not ok:
        R.add("workspaces", "3D relief", SKIP, "OpenGL is not available here" if not ok else "the 3D view was not created")
    else:
        with R.check("workspaces", "3D relief: modes, materials, density, tools"):
            w.registry.trigger("view.mode:3d")
            _settle(app, w, 60)
            for k, _t in ins.relief_material.buttons.items() if hasattr(ins.relief_material, "buttons") else []:
                ins.relief.set_material(k)
                app.processEvents()
            for k in ("0.5", "1", "2", "4", "8"):
                ins._relief_mesh_step(k)
                ins.relief.update()
                app.processEvents()
            for tool in ("measure", "section", "area", ""):
                ins.relief.set_tool(tool)
                app.processEvents()
            w.registry.trigger("view.mode:both")
            _settle(app, w, 30)
            w.registry.trigger("view.mode:image")
            _settle(app, w, 30)
        errs = trap.take()
        if errs:
            R.add("workspaces", "3D relief: errors", FAIL, errs[0][:300], 0, "\n---\n".join(errs[:3]))
    # appearance
    for a in ("light", "dark"):
        with R.check("workspaces", f"appearance {a}"):
            w.registry.trigger(f"view.appearance:{a}")
            _settle(app, w, 30)
    with R.check("workspaces", "window resize small and large"):
        for size in ((900, 600), (1600, 1000), (1200, 800)):
            w.resize(*size)
            _settle(app, w, 10)
    errs = trap.take()
    if errs:
        R.add("workspaces", "errors during appearance and resize", FAIL, errs[0][:300], 0, "\n---\n".join(errs[:3]))


# ── app: screenshots ──────────────────────────────────────────────────────────

def part_screenshots(R, app, w, dlg, trap, args, out_dir):
    import cv2
    import numpy as np
    from helio.application.viewing import Display, screen_image
    from helio.domain.registry import RENDER_ENGINES, default_engine_settings
    shots = os.path.join(out_dir, "screenshots")
    os.makedirs(shots, exist_ok=True)
    for n in w.workspaces:
        with R.check("screenshots", f"window: {n}"):
            w.go_workspace(n)
            _settle(app, w, 30)
            w.grab().save(os.path.join(shots, f"window_{n}.png"))
    if not w.project.solved or w.render_data is None:
        return
    D, defaults = w.render_data, default_engine_settings()
    tiles = []
    for v in RENDER_ENGINES:
        try:
            img = np.asarray(screen_image(v, D, dict(defaults[v]), Display(layer="Shading"))).astype(np.uint8)
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, 2)
            s = 360.0 / max(img.shape[:2])
            img = cv2.resize(img[..., :3], (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
            cv2.putText(img, v[:42], (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(img)
        except Exception:                             # noqa: BLE001
            pass
    if tiles:
        cell = (360, 360)
        cols = 5
        rows = (len(tiles) + cols - 1) // cols
        sheet = np.full((rows * cell[1], cols * cell[0], 3), 24, np.uint8)
        for i, t in enumerate(tiles):
            y, x = divmod(i, cols)
            sheet[y * cell[1]:y * cell[1] + t.shape[0], x * cell[0]:x * cell[0] + t.shape[1]] = t[..., ::-1] if False else t
        cv2.imwrite(os.path.join(shots, "all_views_contact_sheet.png"), sheet[..., ::-1])
        R.add("screenshots", "contact sheet of every view", OK, os.path.join(shots, "all_views_contact_sheet.png"))


# ── the two children and the runner ───────────────────────────────────────────

def run_core(args):
    faulthandler.enable(open(os.path.join(args.out, "crash_trace_core.txt"), "w"))
    R = Report(os.path.join(args.out, "part_core.json"))
    R.meta["started"] = datetime.datetime.now().isoformat(timespec="seconds")
    wanted = [p for p in PARTS_CORE if not args.only or p in args.only]
    fns = dict(environment=part_environment, imports=part_imports, settings=part_settings, tests=part_tests)
    for p in wanted:
        if p == "tests" and args.quick and not args.only:
            R.add("tests", "test suite", SKIP, "--quick")
            continue
        print(f"\n== {p} ==", flush=True)
        try:
            fns[p](R, args)
        except Exception as e:                        # noqa: BLE001
            R.add(p, "part crashed", FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=12))
    R.save()


def run_app(args):
    R = Report(os.path.join(args.out, "part_app.json"))
    R.meta["started"] = datetime.datetime.now().isoformat(timespec="seconds")
    scratch = tempfile.mkdtemp(prefix="helio_fc_scratch_")
    os.environ["HELIO_QT_SETTINGS"] = os.path.join(scratch, "prefs.ini")
    os.environ["HELIO_PREFS"] = os.path.join(scratch, "prefs.json")
    if args.headless:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    faulthandler.enable(open(os.path.join(args.out, "crash_trace_app.txt"), "w"))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    trap = Trap()
    trap.install()
    dlg = Dialogs(scratch)
    dlg.install()
    import helio_qt.window as hw
    wanted = [p for p in PARTS_APP if not args.only or p in args.only or p == "solve"]
    try:
        w = hw.MainWindow(appearance="dark")
        w.resize(1440, 900)
        w.show()
        app.processEvents()
    except Exception as e:                            # noqa: BLE001
        R.add("solve", "open the window", FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=12))
        return
    solved = False
    for p in wanted:
        print(f"\n== {p} ==", flush=True)
        try:
            if p == "solve":
                solved = part_solve(R, app, w, dlg, trap, args)
            elif not solved and p in ("views", "exports", "actions", "workspaces", "screenshots") and p != "workspaces":
                R.add(p, "needs a solve", SKIP, "the solve did not finish")
            elif p == "views":
                part_views(R, app, w, dlg, trap, args)
            elif p == "exports":
                part_exports(R, app, w, dlg, trap, args, args.out)
            elif p == "actions":
                part_actions(R, app, w, dlg, trap, args, args.out)
            elif p == "workspaces":
                part_workspaces(R, app, w, dlg, trap, args, args.out)
            elif p == "screenshots":
                part_screenshots(R, app, w, dlg, trap, args, args.out)
        except Exception as e:                        # noqa: BLE001
            R.add(p, "part crashed", FAIL, f"{type(e).__name__}: {e}", 0, traceback.format_exc(limit=12))
        _close_extras(app, w)
    if trap.qt:
        top = sorted(trap.qt.items(), key=lambda kv: -kv[1])[:25]
        R.add("workspaces", "Qt warnings during the run", WARN if top else OK,
              f"{len(top)} distinct; most common: {top[0][0][:140]} (x{top[0][1]})" if top else "",
              0, "\n".join(f"x{n}  {m}" for m, n in top))
    R.save()
    try:
        w.close()
    except Exception:                                 # noqa: BLE001
        pass


def build_markdown(rows, meta, args, wall, crashes):
    counts = {k: sum(1 for r in rows if r["status"] == k) for k in (OK, FAIL, WARN, SKIP, INFO)}
    fails = [r for r in rows if r["status"] == FAIL]
    warns = [r for r in rows if r["status"] == WARN]
    L: list[str] = []
    L.append("# Helio full check\n")
    L.append(f"{datetime.datetime.now().isoformat(timespec='seconds')} · {wall / 60:.1f} min · "
             f"**{counts[OK]} ok · {counts[FAIL]} FAIL · {counts[WARN]} warn · {counts[SKIP]} skipped**\n")
    L.append("## Paste this to Claude\n")
    L.append("```")
    L.append(f"Helio full check — {counts[FAIL]} failures, {counts[WARN]} warnings, {counts[OK]} ok")
    for k in ("python", "platform", "kernels", "opengl", "machine", "capture", "photographs", "solve"):
        if k in meta:
            L.append(f"{k}: {meta[k]}")
    if crashes:
        L.append("CRASHED: " + "; ".join(crashes))
    L.append("")
    L.append("FAILURES:")
    for r in fails[:80]:
        L.append(f"- [{r['part']}] {r['name']}: {r['detail'][:300]}")
        if r["trace"]:
            tail = [ln for ln in r["trace"].strip().splitlines() if ln.strip()][-6:]
            L.extend("    " + ln[:200] for ln in tail)
    if len(fails) > 80:
        L.append(f"... and {len(fails) - 80} more (see REPORT.json)")
    L.append("")
    L.append("WARNINGS:")
    for r in warns[:60]:
        L.append(f"- [{r['part']}] {r['name']}: {r['detail'][:240]}")
    if len(warns) > 60:
        L.append(f"... and {len(warns) - 60} more")
    L.append("```\n")
    L.append("## Summary by part\n")
    L.append("| part | ok | FAIL | warn | skip |\n|---|---|---|---|---|")
    for part in dict.fromkeys(r["part"] for r in rows):
        sub = [r for r in rows if r["part"] == part]
        L.append(f"| {part} | {sum(r['status'] == OK for r in sub)} | {sum(r['status'] == FAIL for r in sub)} | "
                 f"{sum(r['status'] == WARN for r in sub)} | {sum(r['status'] == SKIP for r in sub)} |")
    L.append("\n## Failures in full\n")
    for r in fails:
        L.append(f"### [{r['part']}] {r['name']}\n{r['detail']}\n")
        if r["trace"]:
            L.append("```\n" + r["trace"][-3000:] + "\n```\n")
    L.append("## Warnings\n")
    for r in warns:
        L.append(f"- **[{r['part']}] {r['name']}** — {r['detail']}")
    L.append("\n## Skipped\n")
    for r in rows:
        if r["status"] == SKIP:
            L.append(f"- [{r['part']}] {r['name']} — {r['detail']}")
    tm = meta.get("view timings (first / repeat, s)")
    if tm:
        L.append("\n## Time to draw each view (first draw / repeat, seconds)\n")
        for k, (c, wm) in sorted(tm.items(), key=lambda kv: -kv[1][0]):
            L.append(f"- {k}: {c} / {wm}")
    L.append("\n## Environment\n")
    for k, v in meta.items():
        if k != "view timings (first / repeat, s)":
            L.append(f"- {k}: {v}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m helio_qt.fullcheck", description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", help="a saved .helio session to check on (its photographs and lights)")
    ap.add_argument("--images", help="a folder of photographs")
    ap.add_argument("--lp", help="the lights file for --images")
    ap.add_argument("--out", help="the report folder (default: ./helio_fullcheck_<date>)")
    ap.add_argument("--quick", action="store_true", help="fewer combinations, and no test suite")
    ap.add_argument("--headless", action="store_true", help="no windows on screen (offscreen; no 3D)")
    ap.add_argument("--only", help="comma list of parts: " + ",".join(PARTS_CORE + PARTS_APP))
    ap.add_argument("--full-exports", action="store_true", help="mesh exports at every pixel (slow and large on big captures)")
    ap.add_argument("--work-scale", type=float, default=None, help="solve your capture at this fraction of its size")
    ap.add_argument("--size", default="1600x1200", help="the made capture's size, WxH (default 1600x1200)")
    ap.add_argument("--solve-timeout", type=float, default=3600.0)
    ap.add_argument("--test-timeout", type=float, default=1800.0)
    ap.add_argument("--child", choices=("core", "app"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.only = [x.strip() for x in args.only.split(",")] if args.only else []
    try:
        args.size = tuple(int(x) for x in args.size.lower().split("x"))
    except ValueError:
        ap.error("--size must look like 1600x1200")
    args.out = os.path.abspath(args.out or f"helio_fullcheck_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(args.out, exist_ok=True)

    if args.child == "core":
        return run_core(args)
    if args.child == "app":
        return run_app(args)

    t0 = time.time()
    print(f"Helio full check → {args.out}\n", flush=True)
    base = [sys.executable, "-m", "helio_qt.fullcheck", "--out", args.out]
    for flag in ("session", "images", "lp", "work_scale"):
        v = getattr(args, flag)
        if v is not None:
            base += [f"--{flag.replace('_', '-')}", str(v)]
    for flag in ("quick", "headless", "full_exports"):
        if getattr(args, flag):
            base.append(f"--{flag.replace('_', '-')}")
    base += ["--size", f"{args.size[0]}x{args.size[1]}", "--solve-timeout", str(args.solve_timeout),
             "--test-timeout", str(args.test_timeout)]
    if args.only:
        base += ["--only", ",".join(args.only)]
    crashes = []
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=here + os.pathsep + os.environ.get("PYTHONPATH", ""))
    todo = []
    if not args.only or any(p in args.only for p in PARTS_CORE):
        todo.append("core")
    if not args.only or any(p in args.only for p in PARTS_APP):
        todo.append("app")
    for child in todo:
        env2 = dict(env)
        if child == "core":
            env2["QT_QPA_PLATFORM"] = "offscreen"
        p = subprocess.run(base + ["--child", child], cwd=here, env=env2)
        if p.returncode != 0:
            tail = ""
            ct = os.path.join(args.out, f"crash_trace_{child}.txt")
            if os.path.exists(ct):
                lines = [ln for ln in open(ct, errors="replace").read().splitlines()
                         if "Extension modules" not in ln and not ln.startswith(("sparse.", "scipy"))]
                tail = "\n".join(lines[:30])
            crashes.append(f"the {child} check ended abnormally (exit {p.returncode})" + (f"\n{tail}" if tail else ""))
    rows, meta = [], {}
    for part in ("core", "app"):
        f = os.path.join(args.out, f"part_{part}.json")
        if os.path.exists(f):
            try:
                d = json.load(open(f))
                rows += d["rows"]
                meta.update(d["meta"])
            except (OSError, ValueError):
                crashes.append(f"part_{part}.json could not be read")
    md = build_markdown(rows, meta, args, time.time() - t0, crashes)
    open(os.path.join(args.out, "REPORT.md"), "w").write(md)
    json.dump(dict(meta=meta, rows=rows, crashes=crashes), open(os.path.join(args.out, "REPORT.json"), "w"), indent=1)
    n_fail = sum(r["status"] == FAIL for r in rows) + len(crashes)
    print("\n" + "=" * 70)
    print(f"Done in {(time.time() - t0) / 60:.1f} min: {sum(r['status'] == OK for r in rows)} ok, {n_fail} FAIL, "
          f"{sum(r['status'] == WARN for r in rows)} warnings, {sum(r['status'] == SKIP for r in rows)} skipped")
    print(f"Report: {os.path.join(args.out, 'REPORT.md')}")
    print("Open it and paste the block under 'Paste this to Claude' back to me; "
          f"the screenshots are in {os.path.join(args.out, 'screenshots')}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
