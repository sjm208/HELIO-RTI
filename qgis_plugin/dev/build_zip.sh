#!/bin/zsh
# Package the plugin for QGIS "Plugins > Manage and Install Plugins > Install from ZIP", check the packaged copy
# imports and opens its dialog in QGIS's Python, and put it where Helio ships it (helio/vendor).
# Usage:  dev/build_zip.sh [output.zip]
set -e
D="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$D")"
PKG=helio_relief
SRC="$ROOT/$PKG"
VER=$(grep '^version=' "$SRC/metadata.txt" | cut -d= -f2)
OUT="${1:-$(dirname "$ROOT")/helio/vendor/${PKG}_v${VER}.zip}"
TMP=$(mktemp -d)
mkdir -p "$TMP/$PKG" "$TMP/verify"
cp "$SRC"/*.py "$SRC"/metadata.txt "$SRC"/icon.png "$TMP/$PKG/"
rm -f "$OUT"
(cd "$TMP" && zip -q -r -X "$OUT" "$PKG")
unzip -q "$OUT" -d "$TMP/verify"
"$D/qpy.sh" -c "
import sys; sys.path.insert(0, '$TMP/verify')
from qgis.core import QgsApplication
app = QgsApplication([], True); app.initQgis()
from $PKG import dialog as D
d = D.HelioReliefDialog(None, None); d.shutdown()
print('zip OK: $OUT')" 2>&1 | grep -v "qt.qpa\|^Warning\|FutureWarning\|warnings.warn"
rm -rf "$TMP"
find "$SRC" -name "*.pyc" -delete
