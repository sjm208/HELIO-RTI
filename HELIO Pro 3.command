#!/bin/bash
# HELIO Pro 3 — double-click to run the new app straight from this folder.
#
# Runs the source in this folder, so it always has the latest work, with no
# installing.  Leave this Terminal window open while HELIO runs: anything the
# app writes to the console (and any error) shows here.
cd "$(dirname "$0")" || exit 1
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
[ -x "$PY" ] || PY="$(command -v python3)"
printf '\n  HELIO Pro 3 — starting from %s\n\n' "$(pwd)"
exec "$PY" helio3.py "$@"
