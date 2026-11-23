#!/bin/zsh
# Run a script with QGIS's bundled Python, headless - for the unit tests, smoke tests and the zip check.
# Uses the newest QGIS 4 in /Applications (the plugin needs QGIS 4 / PyQt6) unless QGIS_APP names one.
# Creates what it needs under dev/ (git-ignored), re-pointed on every run so QGIS upgrades just work:
#   dev/.pyhome/lib/python3.x -> QGIS's standard library (QGIS's python has no PYTHONHOME of its own)
#   dev/pkgs/helio_relief     -> the plugin folder, importable as the package "helio_relief"
#   dev/.pylib                -> extra packages (pytest):  python3 -m pip install --target dev/.pylib pytest
# Usage:  dev/qpy.sh dev/smoke_helio.py <folder>      dev/qpy.sh -m pytest tests -q -p no:cacheprovider
setopt null_glob
D="$(cd "$(dirname "$0")" && pwd)"
QGIS="${QGIS_APP:-$(printf '%s\n' /Applications/QGIS-final-4_*.app | sort -V | tail -1)}"
if [ ! -d "$QGIS" ]; then
    echo "qpy.sh: no QGIS 4 found in /Applications (QGIS-final-4_*.app) - set QGIS_APP to the .app" >&2
    exit 1
fi
PY="$(basename "$(printf '%s\n' "$QGIS"/Contents/Resources/python3.* | head -1)")"      # e.g. python3.12
mkdir -p "$D/.pyhome/lib" "$D/pkgs" "$D/.pylib"
ln -sfn "$QGIS/Contents/Resources/$PY" "$D/.pyhome/lib/$PY"
ln -sfn "$(dirname "$D")/helio_relief" "$D/pkgs/helio_relief"
export PYTHONHOME="$D/.pyhome"
export PYTHONPATH="$D/.pylib:$QGIS/Contents/Resources/$PY/site-packages:$QGIS/Contents/Resources/python:$QGIS/Contents/Resources/python/plugins"
export QT_QPA_PLATFORM=offscreen
export PROJ_DATA="$QGIS/Contents/Resources/qgis/proj"
export PROJ_LIB="$QGIS/Contents/Resources/qgis/proj"
export QGIS_PREFIX_PATH="$QGIS/Contents/MacOS"
exec "$QGIS/Contents/MacOS/$PY" "$@"
