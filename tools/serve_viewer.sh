#!/bin/bash
# Serve a generated HTML viewer so a browser can actually open it.
#
# A file:// URL is refused by the sandboxed browser pane; an http:// one is
# not. Three sessions shipped an unverified RTI viewer on the assumption that
# "nothing here can open a local file in a browser" - it was the *scheme* that
# was blocked, not the file. See docs/HANDOFF.md section 5z.
#
#   tools/serve_viewer.sh ~/Desktop/HELIO_RTI_viewer_sample.html [port]
#
# Then point the browser at http://localhost:<port>/ and drive it. To measure
# what the canvas is actually drawing, remember that the viewer runs WebGL
# with preserveDrawingBuffer:false - gl.readPixels only returns real pixels in
# the SAME task as the draw. Read after an await and every mode measures black.

set -euo pipefail

SRC="${1:?usage: $0 <viewer.html> [port]}"
PORT="${2:-8731}"

if [ ! -f "$SRC" ]; then
    echo "no such file: $SRC" >&2
    exit 1
fi

DIR=$(mktemp -d -t helio_viewer)
cp "$SRC" "$DIR/index.html"

echo "serving $(basename "$SRC") at http://localhost:$PORT/"
echo "   (copied to $DIR — Ctrl-C to stop)"
exec python3 -m http.server "$PORT" --directory "$DIR"
