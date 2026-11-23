"""Long work off the interface thread: submit → progress → cancel → result (JOB-001).

Pure Python; no Tk, no Qt.  A job is a function given a ``JobContext`` it
reports progress through and checks for cancellation.  Callbacks run on the
job's own thread, so an interface marshals them to its main thread itself
(Qt: a queued signal; Tk: ``after``).

Rules the tests hold this to (tests/test_jobs.py):

* progress is reported in order and never goes backwards;
* cancelling stops the job at its next check point; the last good result of
  that kind is kept, never half-replaced;
* an exception becomes a structured failure (type, message, traceback);
* each submit of a kind gets a higher generation, and a job that finishes
  after a newer one of its kind was submitted is *stale*: its result is
  refused, so an old solve can never overwrite a newer one;
* ``shutdown`` cancels everything and leaves no thread behind.
"""
from __future__ import annotations

import itertools
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


class Cancelled(Exception):
    """Raised inside a job at a check point once cancel was asked for."""


@dataclass(frozen=True)
class Failure:
    kind: str           # exception class name
    message: str
    trace: str

    def __str__(self) -> str:
        return self.message or self.kind


@dataclass
class Job:
    id: int
    kind: str
    generation: int
    state: str = "queued"          # queued, running, done, failed, cancelled, stale
    stage: str = ""
    fraction: float = 0.0
    result: Any = None
    failure: Failure | None = None
    started: float = 0.0
    finished: float = 0.0
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)

    def cancel(self):
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")


class JobContext:
    """What a job function receives: report progress, check for cancel."""

    def __init__(self, job: Job, on_progress):
        self._job, self._on_progress = job, on_progress

    @property
    def cancelled(self) -> bool:
        return self._job.cancel_requested

    def check(self):
        """A check point: stop here if cancel was asked for."""
        if self._job.cancel_requested:
            raise Cancelled()

    def progress(self, stage: str | None, fraction: float | None = None, detail: str = ""):
        j = self._job
        if stage:
            j.stage = stage
        if fraction is not None:
            j.fraction = max(j.fraction, float(fraction))      # never backwards
        if self._on_progress:
            self._on_progress(j, stage, j.fraction, detail)


class JobManager:
    def __init__(self):
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._latest: dict[str, int] = {}          # kind → newest generation submitted
        self._last_good: dict[str, Any] = {}       # kind → last result accepted
        self.jobs: dict[int, Job] = {}
        self._closed = False

    def submit(self, kind: str, fn: Callable[[JobContext], Any], *,
               on_progress=None, on_done=None, on_failed=None, on_cancelled=None,
               on_stale=None) -> Job:
        """Run ``fn(ctx)`` on its own thread.  Callbacks get the Job."""
        with self._lock:
            if self._closed:
                raise RuntimeError("job manager is shut down")
            gen = self._latest.get(kind, 0) + 1
            self._latest[kind] = gen
            job = Job(next(self._ids), kind, gen)
            self.jobs[job.id] = job
        ctx = JobContext(job, on_progress)

        def _run():
            job.state, job.started = "running", time.time()
            try:
                result = fn(ctx)
                if job.cancel_requested:
                    raise Cancelled()
            except Cancelled:
                job.state, job.finished = "cancelled", time.time()
                if on_cancelled:
                    on_cancelled(job)
                return
            except BaseException as e:      # noqa: BLE001 - every failure is reported
                job.failure = Failure(type(e).__name__, str(e), traceback.format_exc())
                job.state, job.finished = "failed", time.time()
                if on_failed:
                    on_failed(job)
                return
            with self._lock:
                stale = self._latest.get(kind, 0) != job.generation
                if not stale:
                    self._last_good[kind] = result
            job.finished = time.time()
            if stale:
                job.state = "stale"
                if on_stale:
                    on_stale(job)
                return
            job.result, job.state = result, "done"
            if on_done:
                on_done(job)

        job._thread = threading.Thread(target=_run, name=f"helio-job-{kind}-{job.id}", daemon=True)
        job._thread.start()
        return job

    def last_result(self, kind: str):
        """The newest result of *kind* that finished and was not stale."""
        with self._lock:
            return self._last_good.get(kind)

    def active(self, kind: str | None = None) -> list[Job]:
        return [j for j in list(self.jobs.values())
                if j.active and (kind is None or j.kind == kind)]

    def cancel(self, kind: str | None = None):
        for j in self.active(kind):
            j.cancel()

    def wait(self, job: Job, timeout: float | None = None) -> bool:
        t = job._thread
        if t is not None:
            t.join(timeout)
        return not job.active

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Cancel everything and wait for every thread; True when none is left."""
        with self._lock:
            self._closed = True
        self.cancel()
        deadline = time.time() + timeout
        for j in list(self.jobs.values()):
            if j._thread is not None:
                j._thread.join(max(0.0, deadline - time.time()))
        return not any(j._thread is not None and j._thread.is_alive() for j in self.jobs.values())
