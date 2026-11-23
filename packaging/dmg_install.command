#!/bin/bash
# Install Helio — double-click this file.
#
# Why this exists
# ---------------
# HELIO is not notarised by Apple. macOS quarantines anything downloaded
# from the internet, and refuses to open a quarantined app that Apple has not
# signed off — usually with "Helio is damaged and can't be opened. You
# should move it to the Trash."
#
# The app is not damaged. That is the message macOS shows for *unnotarised*,
# and it is the same message whether the app is fine or genuinely corrupt,
# which is why it frightens people.
#
# Right-click → Open does NOT clear this particular one. Only removing the
# quarantine flag does, and that is a Terminal command most people should not
# be asked to type. So it is in here instead: double-click, and this does it.
#
# It installs every app on the disk image beside it: Helio and Helio Classic
# (HELIO Pro 3 and HELIO Pro (classic) before October 2026, whose copies it
# offers to move to the Trash).

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
shopt -s nullglob
APPS=("$HERE"/*.app)

printf '\n  Helio — installer\n'
printf '  ================================================\n\n'

if [ ${#APPS[@]} -eq 0 ]; then
    printf '  Could not find an app next to this installer.\n'
    printf '  Run this from the mounted Helio disk image.\n\n'
    read -r -p "  Press Return to close. " _
    exit 1
fi

FAILED=0
INSTALLED=()
for SRC in "${APPS[@]}"; do
    APP="$(basename "$SRC")"
    NAME="${APP%.app}"
    DEST="/Applications/$APP"
    printf '  %s\n' "$NAME"

    if [ -d "$DEST" ]; then
        printf '    A copy is already in Applications.\n'
        read -r -p '    Replace it with this one? [y/N] ' REPLY
        case "$REPLY" in
            [yY]*) rm -rf "$DEST" || { printf '\n    Could not remove the old copy.\n'; FAILED=1; continue; } ;;
            *) printf '    Left the existing copy alone. Clearing its quarantine anyway.\n' ;;
        esac
    fi

    if [ ! -d "$DEST" ]; then
        printf '    Copying to Applications… (this takes a moment: it is a large app)\n'
        if ! cp -R "$SRC" "$DEST"; then
            printf '\n    Copy failed. If your Mac is managed by an IT department they\n'
            printf '    may not allow installing to Applications. Try dragging the app\n'
            printf '    to your Desktop instead, then run this installer again.\n\n'
            FAILED=1
            continue
        fi
    fi

    xattr -dr com.apple.quarantine "$DEST" 2>/dev/null
    # Verify rather than assume: if the flag is still there, saying "done" would
    # send someone to an error message we just promised they would not see.
    if xattr -p com.apple.quarantine "$DEST" >/dev/null 2>&1; then
        printf '\n    The quarantine flag could not be removed.\n'
        printf '    This usually means the app is owned by another user, or your\n'
        printf '    Mac is centrally managed. Ask your IT administrator to allow\n'
        printf '    %s, or run it from a folder you own.\n\n' "$NAME"
        FAILED=1
        continue
    fi
    printf '    Installed, and will open normally.\n\n'
    INSTALLED+=("$DEST")
done

if [ ${#INSTALLED[@]} -eq 0 ]; then
    read -r -p "  Nothing was installed. Press Return to close. " _
    exit 1
fi

# The same apps under their names before October 2026: offered to the Trash
# (moved, not deleted), and only once Helio itself is installed.
case " ${INSTALLED[*]} " in
    *"/Helio.app "*)
        for OLD in "HELIO Pro 3.app" "HELIO Pro.app"; do
            [ -d "/Applications/$OLD" ] || continue
            printf '  %s is the same app under its old name.\n' "${OLD%.app}"
            read -r -p '    Move the old copy to the Trash? [y/N] ' REPLY
            case "$REPLY" in
                [yY]*) mv "/Applications/$OLD" "$HOME/.Trash/$OLD $(date +%H%M%S).app" \
                           && printf '    Moved to the Trash.\n\n' \
                           || printf '    Could not move it; drag it to the Trash yourself.\n\n' ;;
                *) printf '    Left where it is.\n\n' ;;
            esac
        done ;;
esac

# Open the newest app on the image: Helio when it is there.
OPEN="${INSTALLED[0]}"
for D in "${INSTALLED[@]}"; do
    case "$D" in *"/Helio.app") OPEN="$D" ;; esac
done
printf '  Done. Opening %s…\n\n' "$(basename "${OPEN%.app}")"
open "$OPEN" 2>/dev/null

if [ "$FAILED" -ne 0 ]; then
    read -r -p "  One app could not be installed (see above). Press Return to close. " _
    exit 1
fi
printf '  You can close this window.\n\n'
