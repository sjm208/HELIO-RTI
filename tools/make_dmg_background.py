"""The disk image's window background, drawn to fit build_helio3.sh's layout.

    python tools/make_dmg_background.py      # writes packaging/helio3_dmg_background.png and .tiff

The window is 640 x 680 points (taller than the icons need: Finder on macOS 26 shows
a toolbar and status bar inside it); the icons (96 pt) sit where build_helio3.sh
places them: Helio at (180, 150), Applications at (460, 150), Install Helio
at (320, 360), Read Me and the notices at (180, 490) and (460, 490).  Finder
writes the icons' names in dark text, so the background is light.  The .tiff
carries the picture at 1x and 2x, so it is sharp on a Retina screen.
"""
from __future__ import annotations

import os
import subprocess

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 640, 680
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"       # 0 regular, 1 bold, 10 medium
INK, SOFT, ACCENT, WARN = (24, 34, 52), (84, 98, 120), (226, 132, 40), (176, 84, 20)


def font(size, index=0):
    return ImageFont.truetype(FONT, size, index=index)


def centred(d, s, y, size, fill, index=0, x=None):
    f = font(size, index)
    d.text((d.im.size[0] / 2 if x is None else x, y), s, font=f, fill=fill, anchor="mt")


def draw(k: int) -> Image.Image:
    """The picture at *k* times the window's size."""
    im = Image.new("RGB", (W * k, H * k), (240, 243, 247))
    relief = Image.open(os.path.join(ROOT, "packaging/installer_bg_source.png")).convert("L")
    rw = W * k
    relief = relief.resize((rw, int(relief.height * rw / relief.width)), Image.LANCZOS)
    relief = relief.crop((0, relief.height - int(330 * k), rw, relief.height))
    # the relief as a faint watermark along the bottom, fading in
    fade = Image.linear_gradient("L").resize(relief.size)
    tint = Image.new("RGB", relief.size, (206, 214, 226))
    shade = Image.composite(tint, Image.new("RGB", relief.size, (240, 243, 247)),
                            relief.point(lambda v: int(v * 0.55)))
    im.paste(shade, (0, H * k - relief.height), fade.point(lambda v: int(v * 0.85)))
    d = ImageDraw.Draw(im)

    def s(v):
        return v * k

    centred(d, "Helio", s(16), s(38), INK, 1)
    centred(d, "Surface imaging for objects that hide their detail", s(60), s(14), SOFT)
    # the arrow from Helio to Applications
    y = s(140)
    d.line([(s(252), y), (s(374), y)], fill=ACCENT, width=s(4))
    d.polygon([(s(388), y), (s(370), y - s(10)), (s(370), y + s(10))], fill=ACCENT)
    centred(d, "drag Helio to Applications", s(156), s(12), SOFT)
    d.line([(s(64), s(226)), (s(576), s(226))], fill=(200, 208, 220), width=max(1, k))
    centred(d, "Or double-click  “Install Helio”  below", s(240), s(16), INK, 1)
    centred(d, "macOS 15 or later, the first time: if it will not open, choose Done, then",
            s(266), s(12), WARN)
    centred(d, "System Settings › Privacy & Security › Open Anyway.  The Read Me explains.",
            s(282), s(12), WARN)
    return im


def main():
    one, two = draw(1), draw(2)
    png = os.path.join(ROOT, "packaging/helio3_dmg_background.png")
    one.save(png)
    p2 = os.path.join(ROOT, "build_qt_bg@2x.png")
    two.save(p2)
    tiff = os.path.join(ROOT, "packaging/helio3_dmg_background.tiff")
    subprocess.run(["tiffutil", "-cathidpicheck", png, p2, "-out", tiff], check=True,
                   capture_output=True)
    os.remove(p2)
    print(png, tiff)


if __name__ == "__main__":
    main()
