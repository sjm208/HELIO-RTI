"""Generate the installer artwork for both platforms.

Kept as a script rather than committed binaries alone, so the artwork can be
regenerated when the version or branding changes.

  helio_dmg_background.png   640x580  - the DMG window background
  installer_win_wizard.bmp   164x314  - Inno Setup left banner
  installer_win_small.bmp     55x55   - Inno Setup header icon

The palette is HELIO's own UI: #0d151d ground, #7dd4f8 cyan, #f59e42 amber.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = "2.6.0"

GROUND = (13, 21, 29)
GROUND_2 = (22, 36, 48)
CYAN = (125, 212, 248)
AMBER = (245, 158, 66)
MUTED = (142, 163, 184)
WHITE = (232, 239, 245)


def _font(size, bold=False):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
                 else "/System/Library/Fonts/Supplemental/Arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _vertical_gradient(w, h, top, bottom):
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    arr = ((1 - t) * np.array(top, np.float32)
           + t * np.array(bottom, np.float32))
    return np.repeat(arr[:, None, :], w, axis=1).astype(np.uint8)


def _raking_streaks(img, draw, w, h):
    """A few faint diagonal bands - raking light is what this app is for."""
    for i in range(-h, w, 46):
        draw.line([(i, h), (i + h, 0)], fill=(28, 44, 58), width=1)


def _clay_backdrop(W, H):
    """Backdrop from a real HELIO render.

    packaging/installer_bg_source.png is the manuscript reconstructed as 3D geometry and
    shown in the viewer's clay mode under a raking light: no colour, no text,
    just the shape of the parchment. It is the program's own output rather
    than a synthetic texture, which is the point.
    """
    import cv2
    here = os.path.join(HERE, "packaging/installer_bg_source.png")
    if not os.path.exists(here):
        raise SystemExit("packaging/installer_bg_source.png missing - see tools/make_installer_art.py")
    im = cv2.imread(here, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Cover-fit to the panel.
    ih, iw = im.shape
    sc = max(W / iw, H / ih)
    im = cv2.resize(im, (int(iw * sc + 1), int(ih * sc + 1)), interpolation=cv2.INTER_AREA)
    y0 = max(0, (im.shape[0] - H) // 2); x0 = max(0, (im.shape[1] - W) // 2)
    im = im[y0:y0 + H, x0:x0 + W]

    lo, hi = np.percentile(im, 2), np.percentile(im, 98)
    im = np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)

    # Cool monochrome: near-black ground lifting to a pale steel.
    lo_c = np.array(GROUND, np.float32)
    hi_c = np.array((196, 214, 232), np.float32)
    img = lo_c + (hi_c - lo_c) * (im[..., None] * 0.80)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    top  = np.clip(1.0 - yy / 158.0, 0, 1) ** 1.25
    icons = np.exp(-((yy - 190) ** 2) / (2 * 66.0 ** 2))
    foot = np.clip((yy - 246) / 214.0, 0, 1) ** 1.15
    scrim = np.clip(0.90 * top + 0.74 * icons + 0.80 * foot, 0, 0.95)
    img = img * (1 - scrim[..., None]) + lo_c * scrim[..., None]

    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= np.clip(1.08 - 0.36 * r, 0.5, 1.0)[..., None]
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make_dmg_background(path):
    # 640x580.
    #
    # It was 460, which fitted two icon rows and the text between them. It now
    # has to carry three — app/Applications, the installer, and the two
    # documents — and at 460 the installer's 96px icon landed on top of the
    # amber line about "damaged", covering the one sentence that stops someone
    # throwing the app away. Height is cheaper than cleverness here.
    W, H = 640, 580
    img = _clay_backdrop(W, H)
    d = ImageDraw.Draw(img)

    d.text((W // 2, 42), "HELIO Pro", font=_font(40, True), fill=WHITE, anchor="mm")
    d.line([(W // 2 - 132, 70), (W // 2 + 132, 70)], fill=(52, 76, 96), width=1)
    d.text((W // 2, 88), "Surface imaging for objects that hide their detail",
           font=_font(14), fill=CYAN, anchor="mm")
    d.text((W // 2, 110),
           "Reflectance transformation  ·  Photometric stereo  ·  Depth & 3D",
           font=_font(11), fill=MUTED, anchor="mm")
    d.text((W // 2, 126),
           "Terrain analysis  ·  Archival packages  ·  GIS & web viewers",
           font=_font(11), fill=MUTED, anchor="mm")

    # Row 1 (y=150 in Finder): the app and Applications, with the drag arrow.
    ax0, ax1, ay = 250, 394, 152
    d.line([(ax0, ay), (ax1 - 16, ay)], fill=AMBER, width=3)
    d.polygon([(ax1, ay), (ax1 - 18, ay - 10), (ax1 - 18, ay + 10)], fill=AMBER)
    d.text((W // 2, 180), "drag to install", font=_font(10), fill=(120, 140, 158),
           anchor="mm")

    # The part that actually matters.
    #
    # This app is not notarised, so a downloaded copy is quarantined and macOS
    # refuses it with "HELIO Pro is damaged and can't be opened. You should
    # move it to the Trash." People believe that message — it is the same one
    # shown for genuinely corrupt downloads — and they throw the app away.
    # Neither right-click > Open nor Open Anyway clears this particular one;
    # only removing the quarantine flag does. So the installer is put first,
    # and the wording people will actually see is named here, on the one
    # surface nobody can miss: the window they are already looking at.
    d.line([(64, 232), (W - 64, 232)], fill=(46, 66, 84), width=1)
    d.text((W // 2, 256), "Or just double-click  “Install HELIO Pro”  below",
           font=_font(15, True), fill=WHITE, anchor="mm")
    d.text((W // 2, 278),
           "If macOS says HELIO Pro is “damaged” — it is not. The installer fixes that.",
           font=_font(11), fill=AMBER, anchor="mm")
    # Nothing below y≈292. Finder centres the installer icon at 360, so its
    # 96px box starts at 312 — 34px of clear air under the amber line, which is
    # what was missing when this was 460 tall. The documents sit at 490.
    img.save(path)
    return path


def make_dmg_background_v3(path):
    """The HELIO Pro 3 disk image: two apps on the top row, then the installer.

    Same 640x580 window and the same rows as the classic image (installer at
    360, documents at 490), so the amber line about "damaged" keeps its clear
    air.  Finder puts HELIO Pro 3 at x=110, HELIO Pro (classic) at x=250 and
    Applications at x=510 on the y=150 row; the arrow runs between them.
    """
    W, H = 640, 580
    img = _clay_backdrop(W, H)
    d = ImageDraw.Draw(img)

    d.text((W // 2, 42), "HELIO Pro 3", font=_font(40, True), fill=WHITE, anchor="mm")
    d.line([(W // 2 - 150, 70), (W // 2 + 150, 70)], fill=(52, 76, 96), width=1)
    d.text((W // 2, 88), "Surface imaging for objects that hide their detail",
           font=_font(14), fill=CYAN, anchor="mm")
    d.text((W // 2, 110),
           "Photometric stereo  ·  Relighting  ·  3D relief  ·  Measurement",
           font=_font(11), fill=MUTED, anchor="mm")
    d.text((W // 2, 126),
           "Publication plates  ·  Web viewers  ·  GIS  ·  AR",
           font=_font(11), fill=MUTED, anchor="mm")

    # Row 1 (y=150 in Finder): both apps, then Applications.
    ax0, ax1, ay = 322, 446, 152
    d.line([(ax0, ay), (ax1 - 16, ay)], fill=AMBER, width=3)
    d.polygon([(ax1, ay), (ax1 - 18, ay - 10), (ax1 - 18, ay + 10)], fill=AMBER)
    d.text(((ax0 + ax1) // 2, 180), "drag both to install", font=_font(10),
           fill=(120, 140, 158), anchor="mm")

    d.line([(64, 232), (W - 64, 232)], fill=(46, 66, 84), width=1)
    d.text((W // 2, 256), "Or just double-click  “Install HELIO Pro”  below",
           font=_font(15, True), fill=WHITE, anchor="mm")
    d.text((W // 2, 278),
           "If macOS says HELIO Pro is “damaged” — it is not. The installer fixes that.",
           font=_font(11), fill=AMBER, anchor="mm")
    img.save(path)
    return path


def make_windows_art(wizard_path, small_path):
    # Left banner: tall, mostly quiet, branding at the top.
    W, H = 164, 314
    img = Image.fromarray(_vertical_gradient(W, H, GROUND_2, GROUND))
    d = ImageDraw.Draw(img)
    for i in range(-H, W, 34):
        d.line([(i, H), (i + H, 0)], fill=(28, 44, 58), width=1)
    d.text((W // 2, 54), "HELIO", font=_font(26, True), fill=WHITE, anchor="mm")
    d.text((W // 2, 80), "Pro", font=_font(19, True), fill=CYAN, anchor="mm")
    d.line([(34, 100), (130, 100)], fill=AMBER, width=2)
    d.text((W // 2, 122), "Surface imaging", font=_font(10), fill=MUTED, anchor="mm")
    d.text((W // 2, 280), "RTI  ·  Photometric\nStereo", font=_font(10),
           fill=(102, 120, 138), anchor="mm", align="center")
    img.convert("RGB").save(wizard_path, "BMP")

    # Header icon: small, high contrast, no text below 55px.
    S = 55
    img2 = Image.fromarray(_vertical_gradient(S, S, GROUND_2, GROUND))
    d2 = ImageDraw.Draw(img2)
    d2.ellipse([12, 12, 43, 43], outline=CYAN, width=3)
    d2.arc([12, 12, 43, 43], start=205, end=335, fill=AMBER, width=4)
    img2.convert("RGB").save(small_path, "BMP")
    return wizard_path, small_path


if __name__ == "__main__":
    import sys as _sys
    if "--v3" in _sys.argv:                      # only the HELIO Pro 3 image's background
        p3 = make_dmg_background_v3(os.path.join(HERE, "packaging/helio3_dmg_background.png"))
        print(f"  {os.path.basename(p3)}  {Image.open(p3).size}")
        raise SystemExit(0)
    bg = make_dmg_background(os.path.join(HERE, "helio_dmg_background.png"))
    wz, sm = make_windows_art(os.path.join(HERE, "installer_win_wizard.bmp"),
                              os.path.join(HERE, "installer_win_small.bmp"))
    for p in (bg, wz, sm):
        im = Image.open(p)
        print(f"  {os.path.basename(p):32s} {im.size[0]}x{im.size[1]} "
              f"{im.mode:5s} {os.path.getsize(p):>8,d} bytes")
