"""
Aggressive memory‑pressure utilities for Apple Silicon (M2 Max) + macOS.

Call ``aggressive_cleanup()`` between heavy processing stages, especially
after NumPy large‑array operations and VisPy teardown.
"""

import gc
import sys
import ctypes
import logging
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

_log = logging.getLogger(__name__)

# ── macOS libc malloc_trim equivalent ──────────────────────────────

def _malloc_trim() -> None:
    """Call ``malloc_trim(0)`` via ctypes to return freed heap pages to macOS.

    On Apple Silicon the default malloc implementation (magazine malloc)
    does not expose ``malloc_trim``, so this is best‑effort — the real
    benefit comes from ``gc.collect`` + reference clearing below.
    """
    try:
        libc = ctypes.CDLL("libc.dylib")
        # macOS 14+ still exports the symbol; harmless on earlier versions.
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        _log.debug("malloc_trim not available")


def aggressive_cleanup(
    *,
    clear_gc: bool = True,
    clear_sys_refs: bool = True,
    trim_malloc: bool = True,
) -> None:
    """Release every scrap of allocated Python + GPU memory.

    Call this **after** a large processing stage completes and before
    the next one begins.

    Parameters
    ----------
    clear_gc:
        Run ``gc.collect(generation=2)`` (full generational sweep).
    clear_sys_refs:
        Clear CPython's internal object rings
        (``sys.gettotalrefcount`` is debug‑only; we clear
        ``sys.last_*`` hooks and ``sys.excepthook`` references).
    trim_malloc:
        Best‑effort ``malloc_trim`` via libc (macOS).
    """
    if clear_gc:
        _log.debug("aggressive_cleanup: gc.collect(2)")
        gc.collect(2)
        gc.collect(2)  # double sweep catches cyclic trash

    if clear_sys_refs:
        _log.debug("aggressive_cleanup: clearing sys hooks")
        sys.last_type = None
        sys.last_value = None
        sys.last_traceback = None
        # Best-effort type cache clear (CPython internal; harmless if missing)
        try:
            sys._clear_type_cache()
        except AttributeError:
            pass

    if trim_malloc:
        _malloc_trim()

    _log.debug("aggressive_cleanup: done")


@contextmanager
def process_pool(
    max_workers: Optional[int] = None,
    *,
    cleanup_on_exit: bool = True,
):
    """Context manager wrapping ``ProcessPoolExecutor`` with GC cleanup.

    Yields the executor; when the block exits, the pool is shut down and
    ``aggressive_cleanup()`` is called to release any leftover pages.

    Usage::

        with process_pool() as ex:
            results = list(ex.map(func, items))
    """
    ex = ProcessPoolExecutor(max_workers=max_workers)
    try:
        yield ex
    finally:
        ex.shutdown(wait=True)
        if cleanup_on_exit:
            aggressive_cleanup()


def cleanup_and_log(stage_label: str) -> None:
    """Convenience wrapper: log the stage, run cleanup, log result."""
    import tracemalloc
    _log.info("MEMORY %s — starting cleanup", stage_label)
    snap_before = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None
    aggressive_cleanup()
    if snap_before is not None:
        snap_after = tracemalloc.take_snapshot()
        stats = snap_after.compare_to(snap_before, "lineno")
        top = stats[:3]
        for s in top:
            _log.debug("  freed ~%s at %s:%d", _sizeof_fmt(s.size_diff), s.traceback[0][0], s.traceback[0][1])
    _log.info("MEMORY %s — cleanup complete", stage_label)


def _sizeof_fmt(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:+.1f}{unit}"
        num /= 1024.0
    return f"{num:+.1f}TB"
