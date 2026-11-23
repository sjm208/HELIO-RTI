"""HELIO Pro 3: the entry point the built app runs (and `python3 helio3.py`).

The same as `python3 -m helio_qt`.  freeze_support() comes first: in the
built app a child process (the separate 3D window, the publication sheet's
renders) starts this file again, and must do its own work, not open a window.
"""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from helio_qt.app import main
    sys.exit(main(sys.argv))
