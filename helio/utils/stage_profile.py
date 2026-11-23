"""Time and peak memory of each stage of a solve, from the app or a script.

Set ``HELIO_PROFILE=1`` before starting Helio (or run ``tools/profile_solve.py``)
and every solve prints, and logs, a timeline like::

    Solve profile: 8 images, 100.0 MP each  (machine: 96.0 GB, 71.2 GB free)
        seconds   +sec   peak GB  stage
           12.4   12.4      9.8   Loading images 8/8
           ...

*peak GB* is the process's high-water mark so far (``ru_maxrss``): the stage
that raises it is the one that needs the memory.  Nothing here reads a
file's pixels or changes a result.  ``HELIO_PROFILE_FILE=path`` also writes
the report to a file.
"""
from __future__ import annotations

import logging
import os
import sys
import time

_log = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get("HELIO_PROFILE", "").strip().lower() in ("1", "true", "yes", "on")


def peak_rss_gb() -> float:
    """The process's peak resident memory so far, in GB."""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1e9 if sys.platform == "darwin" else 1e6)    # bytes on macOS, KB elsewhere
    except Exception:                                            # Windows has no resource module
        return 0.0


class StageProfile:
    def __init__(self, header: str = ""):
        self.header = header
        self.t0 = time.perf_counter()
        self.rows: list[tuple[float, float, str]] = []           # (t, peak GB, stage)

    def mark(self, stage: str) -> None:
        self.rows.append((time.perf_counter() - self.t0, peak_rss_gb(), stage))

    def wrap(self, progress):
        """A progress callback that also records every call that names a stage."""
        def call(msg=None, frac=None, *rest, **kw):
            label = msg or (rest[0] if rest and isinstance(rest[0], str) else None)
            if label:
                self.mark(label)
            return progress(msg, frac, *rest, **kw)
        return call

    def report(self) -> str:
        self.mark("done")
        out = ["Solve profile: " + self.header, "    seconds    +sec   peak GB  stage"]
        prev = 0.0
        for t, gb, stage in self.rows:
            out.append(f"{t:11.1f} {t - prev:7.1f} {gb:9.2f}  {stage}")
            prev = t
        return "\n".join(out)

    def emit(self) -> str:
        text = self.report()
        _log.info("\n%s", text)
        print(text, file=sys.stderr)
        path = os.environ.get("HELIO_PROFILE_FILE")
        if path:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(text + "\n\n")
            except OSError:
                _log.warning("could not write the profile to %s", path)
        return text
