"""Helio Pro's new interface, in PySide6 (BLUEPRINT.md; docs/pyside6-migration/).

A separate app beside the classic one: it reads the same GUI-free services in
helio/, and never imports Tk or Helio_Pro.py.  Run it with

    python -m helio_qt
"""
try:                                    # stamped by build_helio3.sh from the disk image's name
    from helio_build import RELEASE as _RELEASE
except ImportError:
    _RELEASE = ""
__version__ = _RELEASE or "3.0 (from source)"
