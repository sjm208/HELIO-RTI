"""The compiled kernels (helio_core_rs), when they are there.

``helio_core_rs`` is the Rust build of the heaviest arithmetic in
``helio.core``.  It is optional: when the module is missing (a run from
source without ``maturin develop``, an unsupported platform), or Helio is
started with ``HELIO_NO_RUST=1``, ``rs`` is ``None`` and every caller uses
its NumPy code, which stays in the tree as the reference the Rust is checked
against.

Callers use it the same way everywhere::

    from helio.core._native import rs
    if rs is not None:
        return rs.some_kernel(...)
    ...  # the NumPy code
"""
from __future__ import annotations

import logging
import os

_log = logging.getLogger(__name__)

rs = None
if os.environ.get("HELIO_NO_RUST", "").strip().lower() not in ("1", "true", "yes", "on"):
    try:
        import helio_core_rs as rs
        # Run from the source tree, the helio_core_rs/ crate folder imports as an
        # empty namespace package; only a built extension has these.
        if not hasattr(rs, "lls_normals"):
            rs = None
    except ImportError:
        rs = None
    except Exception as e:             # a broken build must not stop Helio starting
        _log.warning("helio_core_rs failed to load (%s); using the NumPy kernels", e)
        rs = None

HAVE_RUST = rs is not None


def describe() -> str:
    """One line for About and problem reports: which kernels this run uses."""
    if rs is None:
        return "NumPy kernels"
    return f"helio_core_rs {rs.__version__} ({rs.threads()} threads)"
