# Tests that need Helio Classic

These eleven tests compare Helio 3 with Helio Classic (they drive Classic's `Helio_Pro.py`, or read
`docs/pyside6-migration/` baselines and the `test_exports` helper that come with it). This repository is
Helio 3 only, so they cannot run here: each stops at a missing file. They are kept, unchanged, for a checkout
that has Classic beside it; copy them next to `Helio_Pro.py` and `docs/pyside6-migration/` to run them.

Moved out of the main folder so that `./run_tests.sh` and `python3 -m helio_qt.fullcheck` report only what
can really run. Everything they cover for Helio 3 alone is covered by the tests in the main folder
(`test_qt_views`, `test_live_detail`, `test_dome_directions`, `test_ridge`, `test_relief_detail`, ...).
