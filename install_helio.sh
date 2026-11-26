#!/bin/bash
# Install Helio on a Mac from Terminal, with no security warnings:
#
#   curl -fsSL https://raw.githubusercontent.com/sjm208/HELIO-RTI/main/install_helio.sh | bash
#
# A browser marks what it downloads with a "quarantine" flag, and macOS will
# not open a quarantined app that Apple has not notarised (Helio is not): it
# says the app "is damaged", or blocks it until you allow it in System
# Settings.  curl sets no such flag, so the disk image fetched here, and the
# app copied from it, open like any other.
#
# It downloads the latest Helio disk image from the project's Releases, copies
# Helio to Applications (asking before it replaces a copy already there), and
# opens it.  Running it again updates Helio.  Nothing else is installed.
set -euo pipefail

URL="${HELIO_DMG_URL:-https://github.com/sjm208/HELIO-RTI/releases/latest/download/Helio_4.0.dmg}"
DEST="${HELIO_DEST:-/Applications/Helio.app}"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
    echo "Helio's Mac app needs an Apple silicon Mac (M1 or later)."
    echo "On other computers, run it from the source: see the README."
    exit 1
fi

TMP="$(mktemp -d)"
MNT=""
cleanup() {
    [ -n "$MNT" ] && hdiutil detach -quiet "$MNT" 2>/dev/null || true
    rm -rf "$TMP"
}
trap cleanup EXIT

echo "Downloading Helio (about 210 MB)..."
curl -fL --progress-bar -o "$TMP/Helio.dmg" "$URL"

echo "Opening the disk image..."
MNT="$(hdiutil attach -nobrowse -readonly -noautoopen "$TMP/Helio.dmg" | awk -F'\t' '/\/Volumes\//{print $NF; exit}')"
if [ -z "$MNT" ] || [ ! -d "$MNT/Helio.app" ]; then
    echo "The disk image did not hold Helio.app. Nothing was installed."
    exit 1
fi

if [ -d "$DEST" ]; then
    REPLY=y
    if [ -r /dev/tty ]; then
        read -r -p "Helio is already in Applications. Replace it with this one? [Y/n] " REPLY </dev/tty || REPLY=y
    fi
    case "${REPLY:-y}" in
        [nN]*) echo "Left the existing copy alone."; exit 0 ;;
    esac
    if pgrep -f "$DEST/Contents/MacOS/" >/dev/null 2>&1; then
        echo "Quit Helio first, then run this again."
        exit 1
    fi
    rm -rf "$DEST"
fi

echo "Copying Helio to Applications..."
if ! ditto "$MNT/Helio.app" "$DEST" 2>/dev/null; then
    echo "Could not write to Applications. If your Mac is managed, ask whoever"
    echo "manages it; or copy Helio from the disk image to a folder you own."
    exit 1
fi
xattr -dr com.apple.quarantine "$DEST" 2>/dev/null || true

echo "Installed. Opening Helio..."
[ "${HELIO_NO_OPEN:-0}" = "1" ] || open "$DEST"
