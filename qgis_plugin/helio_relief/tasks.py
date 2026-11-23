"""Background jobs: every long computation runs as a QgsTask so QGIS stays
responsive, progress shows in the dialog and the task manager, and jobs can
be cancelled."""

import traceback

from qgis.core import QgsTask
from qgis.PyQt.QtCore import pyqtSignal

from .relief_engine import Cancelled


class Job(QgsTask):
    """Runs ``fn(task)`` on a worker thread.

    ``fn`` reports with ``task.report(percent, message)`` (which also raises
    Cancelled when the user cancels) and must not touch widgets or the
    project. ``on_done(result)``, ``on_error(text)`` and ``on_cancel()`` run
    on the main thread."""

    message = pyqtSignal(str)

    def __init__(self, description, fn, on_done, on_error, on_cancel):
        super().__init__(description, QgsTask.Flag.CanCancel)
        self._fn, self._on_done, self._on_error, self._on_cancel = fn, on_done, on_error, on_cancel
        self.result = None
        self.error = None

    def report(self, percent, message=None):
        if self.isCanceled():
            raise Cancelled()
        self.setProgress(max(0.0, min(100.0, float(percent))))
        if message:
            self.message.emit(message)

    def run(self):
        try:
            self.result = self._fn(self)
            return True
        except Cancelled:
            return False
        except Exception as exc:  # reported on the main thread in finished()
            self.error = f"{exc}\n\n{traceback.format_exc()}"
            return False

    def finished(self, ok):
        if ok:
            self._on_done(self.result)
        elif self.error:
            self._on_error(self.error)
        else:
            self._on_cancel()


def sub_progress(task, lo, hi, prefix=""):
    """Map a 0-100 progress callback into the [lo, hi] slice of a task."""
    def report(pct, msg=None):
        task.report(lo + (hi - lo) * pct / 100.0, (prefix + msg) if msg else None)
    return report
