#!/bin/bash
# Run Helio straight from this folder: always the latest changes,
# no DMG needed.  Double-click it, or run it from Terminal.
# It uses the folder's own Python setup (.venv, made as the README says) when
# there is one, else the python.org Python 3.12, else any python3.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
[ -x "$PY" ] || PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
[ -x "$PY" ] || PY="$(command -v python3)"
exec "$PY" -m helio_qt "$@"
