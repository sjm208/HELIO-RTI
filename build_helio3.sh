#!/bin/bash
# Build Helio.app (PyInstaller, Apple silicon) in dist_qt/ from helio_qt.spec,
# check it, and make the Helio disk image.
#
# Usage: ./build_helio3.sh            (SKIP_APP=1 reuses dist_qt/Helio.app)
#
# Notarised, when there is an Apple Developer ID (the one fix for macOS's
# "damaged" and "Open Anyway" messages; without it the app is signed ad hoc and
# the disk image's installer clears the quarantine instead):
#
#   HELIO_SIGN_ID="Developer ID Application: Name (TEAMID)" \
#   HELIO_NOTARY_PROFILE=helio ./build_helio3.sh
#
# where the profile was stored once with
#   xcrun notarytool store-credentials helio --apple-id ... --team-id ...
#
# The built app runs its own self-test (helio_qt/selftest.py) before any disk
# image is made: a capture of a known surface solved and exported by the
# bundle's own interpreter and libraries.  If it fails, there is no DMG.

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
APP3="Helio"
DIST="${DIST:-dist_qt}"
WORK="${WORK:-build_qt}"
DMG_NAME="${DMG_NAME:-Helio_4.0}"
APP3_PATH="$DIST/$APP3.app"

if [ "$("$PYTHON" -c 'import platform; print(platform.machine())')" != "arm64" ]; then
    echo "ERROR: this builds the Apple silicon app; $PYTHON is not arm64."
    exit 1
fi
if pgrep -f "$APP3_PATH" >/dev/null 2>&1; then
    echo "ERROR: $APP3 is running from $DIST. Quit it first."
    exit 1
fi

if [ "${SKIP_APP:-0}" != "1" ]; then
    echo "Cleaning $WORK and $DIST..."
    rm -rf "$WORK" "$DIST" 2>/dev/null || { sleep 1; rm -rf "$WORK" "$DIST"; }

    # Stamp the build with its commit, as build_dmg.sh does: About and the
    # problem report name the build, and the committed file stays empty.
    STAMP_FILE="helio_build.py"
    # Backed up and put back below: a file already stamped (an interrupted
    # build) would be "restored" stamped, and then committed.
    if ! grep -q '^COMMIT = ""' "$STAMP_FILE"; then
        echo "ERROR: $STAMP_FILE is already stamped. Clear COMMIT, BUILT and ARCH first."
        exit 1
    fi
    STAMP_BACKUP="$(mktemp)"
    cp "$STAMP_FILE" "$STAMP_BACKUP"
    restore_stamp() { cp "$STAMP_BACKUP" "$STAMP_FILE"; rm -f "$STAMP_BACKUP"; }
    trap restore_stamp EXIT
    GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    if [ -n "$(git status --porcelain -uno 2>/dev/null | grep -v "$STAMP_FILE")" ]; then
        GIT_SHA="$GIT_SHA+dirty"
        echo "WARNING: building from a dirty tree — stamping $GIT_SHA"
    fi
    # Helio_3.0_preview16 -> 3.0 preview 16: About and the publication sheet
    # name the release without anyone editing a version by hand.
    RELEASE=$(echo "$DMG_NAME" | sed -E -e 's/^Helio_//' -e 's/_preview([0-9]*)$/ preview \1/' -e 's/_/ /g' -e 's/ +$//')
    /usr/bin/sed -i "" \
        -e "s/^COMMIT = \"\"/COMMIT = \"$GIT_SHA\"/" \
        -e "s/^RELEASE = \"\"/RELEASE = \"$RELEASE\"/" \
        -e "s/^BUILT = \"\"/BUILT = \"$(date -u +%Y-%m-%d)\"/" \
        -e "s/^ARCH = \"\"/ARCH = \"arm64\"/" \
        "$STAMP_FILE"
    "$PYTHON" -c "import helio_build; s=helio_build.stamp(); assert 'source' not in s, s; print('  stamp reads:', s)"

    # The Rust kernels: built into this Python so PyInstaller bundles them.  SKIP_RUST=1 builds
    # without them (the app then uses NumPy and is slower); REQUIRE_RUST=1 stops if they cannot be built.
    if [ "${SKIP_RUST:-0}" != "1" ]; then
        echo "Building the Rust kernels..."
        PYTHON="$PYTHON" REQUIRE_RUST="${REQUIRE_RUST:-0}" ./build_rust.sh
        if "$PYTHON" -c "from helio_core_rs import lls_normals" 2>/dev/null; then
            export HELIO_REQUIRE_RUST=1        # the self-test below then insists the app has them
        fi
    fi

    echo "Running PyInstaller..."
    "$PYTHON" -m PyInstaller --noconfirm --clean --distpath "$DIST" --workpath "$WORK" \
        helio_qt.spec 2>&1 | tee "$WORK.log" | grep -E "^[0-9]+ (INFO|WARNING|ERROR): (Building|Build complete|Copying|.*ERROR)" || true
    restore_stamp
    trap - EXIT
fi

if [ ! -d "$APP3_PATH" ]; then
    echo "ERROR: $APP3_PATH was not built (see $WORK.log)"
    exit 1
fi
echo "$APP3.app: $(du -sh "$APP3_PATH" | cut -f1)"

# The built app checks itself: its interpreter, its libraries, its data files.
echo "Self-test of the built app..."
SELF="$DIST/selftest"
rm -rf "$SELF"
if ! "$APP3_PATH/Contents/MacOS/$APP3" --selftest "$SELF" 2>&1 | grep -E "^  (ok|FAIL)|PASSED|FAILED"; then
    true
fi
if ! "$PYTHON" -c "import json,sys; r=json.load(open('$SELF/selftest.json')); sys.exit(0 if r['ok'] and r['frozen'] else 1)" 2>/dev/null; then
    echo "ERROR: the built app failed its self-test ($SELF/selftest.json). No disk image made."
    exit 1
fi
echo "  the built app passed its self-test"

SIGN_ID="${HELIO_SIGN_ID:-}"
if [ -n "$SIGN_ID" ]; then
    echo "Signing $APP3.app as $SIGN_ID (hardened runtime)..."
    codesign --force --deep --options runtime --timestamp \
        --entitlements packaging/helio_entitlements.plist -s "$SIGN_ID" "$APP3_PATH"
    codesign --verify --deep --strict "$APP3_PATH"
fi

# ── the disk image ─────────────────────────────────────────────────────
# Built read-write, laid out through Finder (a background
# and fixed icon positions), then compressed.  A refused Finder step still
# leaves a working, unstyled image.
echo "Creating DMG..."
DMG_PATH="$DIST/$DMG_NAME.dmg"
VOL_NAME="$DMG_NAME"
STAGE=$(mktemp -d)
RW_DMG="$DIST/${DMG_NAME}_rw.dmg"

cp -R "$APP3_PATH" "$STAGE/"
cp packaging/dmg_readme_v3.txt "$STAGE/Read Me.txt"
cp packaging/dmg_install.command "$STAGE/Install Helio.command"
chmod +x "$STAGE/Install Helio.command"
cp THIRD_PARTY_NOTICES.md "$STAGE/Third-Party Notices.md" 2>/dev/null || true
ln -s /Applications "$STAGE/Applications"
BG=packaging/helio3_dmg_background.tiff                 # 1x and 2x (tools/make_dmg_background.py)
[ -f "$BG" ] || BG=packaging/helio3_dmg_background.png
if [ -f "$BG" ]; then
    mkdir -p "$STAGE/.background"
    cp "$BG" "$STAGE/.background/background.${BG##*.}"
fi

CONTENT_MB=$(du -sm "$STAGE" | cut -f1)
DMG_SIZE_MB=$(( CONTENT_MB + CONTENT_MB / 4 + 100 ))
echo "DMG contents: ${CONTENT_MB}m, allocating ${DMG_SIZE_MB}m"
rm -f "$RW_DMG"
hdiutil create -volname "$VOL_NAME" -srcfolder "$STAGE" -ov \
    -format UDRW -size "${DMG_SIZE_MB}m" "$RW_DMG" >/dev/null

MOUNT_DIR="/Volumes/$VOL_NAME"
hdiutil detach "$MOUNT_DIR" >/dev/null 2>&1 || true
hdiutil attach "$RW_DMG" -readwrite -noverify -noautoopen >/dev/null

if [ -d "$MOUNT_DIR/.background" ]; then
    echo "Applying window layout..."
    for _i in 1 2 3 4 5 6 7 8 9 10; do
        [ -d "$MOUNT_DIR" ] && osascript -e "tell application \"Finder\" to exists disk \"$VOL_NAME\"" 2>/dev/null | grep -q true && break
        sleep 1
    done
    _OSA_SCRIPT=$(mktemp -t helio3_dmg_layout)
    cat > "$_OSA_SCRIPT" <<APPLESCRIPT
tell application "Finder"
    tell disk "$VOL_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {200, 80, 840, 760}
        set opts to the icon view options of container window
        set arrangement of opts to not arranged
        set icon size of opts to 96
        set text size of opts to 12
        set background picture of opts to file ".background:background.${BG##*.}"
        set position of item "$APP3.app" of container window to {180, 150}
        set position of item "Applications" of container window to {460, 150}
        try
            set position of item "Install Helio.command" of container window to {320, 360}
        end try
        try
            set position of item "Read Me.txt" of container window to {180, 490}
        end try
        try
            set position of item "Third-Party Notices.md" of container window to {460, 490}
        end try
        update without registering applications
        delay 1
        close
    end tell
end tell
APPLESCRIPT
    _OSA_ERR=$(osascript "$_OSA_SCRIPT" 2>&1 >/dev/null) || true
    rm -f "$_OSA_SCRIPT"
    if [ -n "$_OSA_ERR" ]; then
        echo "  WARNING: Finder layout failed — the DMG works but has no background,"
        echo "           and the background carries the \"damaged\" instructions."
        echo "           osascript said: $_OSA_ERR"
    fi
    sync
fi

hdiutil detach "$MOUNT_DIR" >/dev/null 2>&1 || hdiutil detach "$MOUNT_DIR" -force >/dev/null 2>&1 || true
echo "Compressing..."
rm -f "$DMG_PATH"
hdiutil convert "$RW_DMG" -format UDZO -imagekey zlib-level=9 -o "$DMG_PATH" >/dev/null
rm -f "$RW_DMG"
rm -rf "$STAGE"
if [ -n "$SIGN_ID" ]; then
    codesign --force --timestamp -s "$SIGN_ID" "$DMG_PATH"
    if [ -n "${HELIO_NOTARY_PROFILE:-}" ]; then
        echo "Notarising (this waits for Apple, usually a few minutes)..."
        xcrun notarytool submit "$DMG_PATH" --keychain-profile "$HELIO_NOTARY_PROFILE" --wait
        xcrun stapler staple "$DMG_PATH"
        spctl -a -vv -t open --context context:primary-signature "$DMG_PATH"
    fi
fi
echo "DMG created: $DMG_PATH ($(du -sh "$DMG_PATH" | cut -f1))"
