"""Start the Qt app: python -m helio_qt [folder-of-images]."""
from __future__ import annotations

import os
import sys


def use_core_profile():
    """OpenGL 4.1 core for every context: call before the QApplication exists.

    macOS shares textures only between contexts of the same profile.  The 3D
    relief draws in a 4.1 core context; left at the default, the window's own
    compositor is a legacy 2.1 context that cannot see the relief's picture,
    and draws its own texture there instead: the whole window, upside down.
    Qt's QOpenGLWidget documentation asks for setDefaultFormat before the
    QApplication on macOS for this reason."""
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QSurfaceFormat
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    QSurfaceFormat.setDefaultFormat(fmt)
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)


def fit_preview_to_screens(app) -> int:
    """The preview as large as the largest screen's own pixels (see pipeline.set_preview_side)."""
    from helio.application.pipeline import set_preview_side
    from helio.utils.memory_plan import system_memory
    longest = 0
    for s in app.screens():
        g = s.geometry()
        longest = max(longest, int(round(max(g.width(), g.height()) * s.devicePixelRatio())))
    return set_preview_side(longest or None, system_memory()[0])


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if "--selftest" in argv:                       # the built app checking itself, no window
        from helio_qt import selftest
        i = argv.index("--selftest")
        return selftest.run(argv[i + 1] if i + 1 < len(argv) else "")
    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    use_core_profile()
    QCoreApplication.setOrganizationName("HELIO")
    QCoreApplication.setApplicationName("Helio")
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationDisplayName("Helio")
    from helio_qt.help import CrashNote, setup_logging
    from helio_qt.window import MainWindow
    setup_logging()
    fit_preview_to_screens(app)
    win = MainWindow()
    win.crash_note = CrashNote(win)
    win.show()
    from helio.config import HAS_RVT
    if not HAS_RVT:                                # nothing else says so: the views just draw differently
        import logging
        msg = ("rvt-py is not installed: Positive/Negative Openness, Sky-View and the other "
               "RVT views will differ from the approved pictures. "
               "Install it with: pip install --no-deps rvt-py==2.2.1")
        logging.getLogger("helio").warning(msg)
        win.flash(msg, 20.0)
    if not win.prefs.value("welcome_done", False, type=bool):
        from PySide6.QtCore import QTimer
        QTimer.singleShot(400, win.show_welcome)
    args = [a for a in argv[1:] if os.path.exists(a)]
    sessions = [a for a in args if a.lower().endswith(".helio")]
    folders = [a for a in args if a not in sessions]
    if sessions:                                   # python -m helio_qt object.helio
        win.load_session_file(sessions[0])
    elif folders:
        win.open_paths(folders)
    return app.exec()
