#!/bin/bash
# HELIO Pro — Intel launch diagnostic.
# Double-click this file on the Intel machine. It writes ~/Desktop/helio_diag.txt
OUT=~/Desktop/helio_diag.txt
{
echo "=== machine ==="
sw_vers; echo "cpu arch: $(uname -m)"
echo
echo "=== which HELIO app(s) are installed ==="
for A in "/Applications/HELIO Pro.app" "$HOME/Applications/HELIO Pro.app" /Volumes/*/HELIO\ Pro.app; do
  [ -d "$A" ] || continue
  echo "--- $A"
  echo "    executable archs : $(lipo -archs "$A/Contents/MacOS/HELIO Pro" 2>/dev/null)"
  N=$(lipo -archs "$A/Contents/Resources/lib/python3.12/numpy/core/_multiarray_umath.cpython-312-darwin.so" 2>/dev/null || \
      lipo -archs "$(find "$A" -name '_multiarray_umath*.so' 2>/dev/null | head -1)" 2>/dev/null)
  echo "    numpy archs      : ${N:-not found}"
  echo "    quarantined      : $(xattr -p com.apple.quarantine "$A" 2>/dev/null || echo no)"
done
echo
echo "=== THE ACTUAL ERROR (this is the bit that matters) ==="
APP=$(ls -d "/Applications/HELIO Pro.app" "$HOME/Applications/HELIO Pro.app" /Volumes/*/HELIO\ Pro.app 2>/dev/null | head -1)
if [ -n "$APP" ]; then
  echo "running: $APP/Contents/MacOS/HELIO Pro"
  "$APP/Contents/MacOS/HELIO Pro" 2>&1 | head -60
else
  echo "No HELIO Pro.app found — mount the DMG or copy the app to /Applications first."
fi
echo
echo "=== startup crash report (written by the app itself) ==="
# main.py now catches anything raised before the UI exists and writes the
# traceback here, because py2app's "Launch error" dialog shows nothing.
for C in ~/Desktop/HELIO_Pro_crash.txt ~/Library/Logs/HELIO_Pro/startup_crash.txt; do
  if [ -f "$C" ]; then echo "--- $C"; cat "$C"; fi
done
[ -f ~/Desktop/HELIO_Pro_crash.txt ] || [ -f ~/Library/Logs/HELIO_Pro/startup_crash.txt ] || \
  echo "(none — either it started fine, or it died before Python ran at all)"
echo
echo "=== app's own log, if it got that far ==="
tail -30 ~/Library/Logs/HELIO_Pro/helio_pro.log 2>/dev/null || echo "(no log — it failed before logging started)"
} > "$OUT" 2>&1
echo "Written to $OUT"
open -R "$OUT"
