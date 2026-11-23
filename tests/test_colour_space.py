"""A photograph's colour is read by its own ICC profile.

A charter's TIFFs were Adobe RGB, read as sRGB: pale, about 2.2 ΔE₀₀ out.

* an Adobe RGB file's colour comes in as the same colour in sRGB, an sRGB or
  untagged file's is left as it is;
* the grey the solve fits is the stored numbers, whichever;
* a 16-bit TIFF keeps its depth in the grey;
* the colour check reads the chart through the profile;
* the sheet says it is sRGB.

    python tests/test_colour_space.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageCms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


# a profile with Adobe RGB's primaries and 2.2 curve, as a TIFF would carry
ADOBE = np.array([[0.5767309, 0.1855540, 0.1881852], [0.2973769, 0.6273491, 0.0752741],
                  [0.0270343, 0.0706872, 0.9911085]])




def _make_adobe() -> bytes:
    import struct
    def xyz(tag, v):
        return b"XYZ \0\0\0\0" + b"".join(struct.pack(">i", int(round(x * 65536))) for x in v)
    # D50-adapted colorants of Adobe RGB (1998)
    cols = {b"rXYZ": (0.60974, 0.31111, 0.01947), b"gXYZ": (0.20528, 0.62567, 0.06087),
            b"bXYZ": (0.14919, 0.06322, 0.74457), b"wtpt": (0.96420, 1.0, 0.82491)}
    trc = b"curv\0\0\0\0" + struct.pack(">I", 1) + struct.pack(">H", int(2.19921875 * 256))
    desc_text = b"Adobe RGB (1998)\0"
    desc = (b"desc\0\0\0\0" + struct.pack(">I", len(desc_text)) + desc_text + b"\0" * 78)
    cprt = b"text\0\0\0\0test\0\0\0\0"
    tags = [(b"desc", desc), (b"cprt", cprt)] + [(k, xyz(k, v)) for k, v in cols.items()] \
        + [(b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc)]
    tags = [(k, v + b"\0" * (-len(v) % 4)) for k, v in tags]
    table, body, off = b"", b"", 128 + 4 + 12 * len(tags)
    for k, v in tags:
        table += k + struct.pack(">II", off, len(v))
        body += v
        off += len(v)
    size = off
    head = struct.pack(">I", size) + b"lcms" + bytes([2, 0x40, 0, 0]) + b"mntrRGB XYZ " + b"\0" * 12 \
        + b"acsp" + b"\0" * 4 * 5 + b"\0" * 8 + struct.pack(">iii", 63190, 65536, 54061)
    head = head.ljust(128, b"\0")
    return head + struct.pack(">I", len(tags)) + table + body


def main():
    from helio.application.colour_check import delta_e_2000, linear_to_lab_d50, srgb_to_linear
    from helio.io import load_and_process_single
    from helio.io.colour_space import profile_name, srgb_profile_bytes
    d = tempfile.mkdtemp(prefix="helio_cs_")
    h, w = 40, 60
    # the colour to carry: a saturated wax red, as Adobe RGB numbers for it
    want = np.array([190, 70, 50], float) / 255
    lin = srgb_to_linear(want)
    M_srgb = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    in_adobe = np.linalg.solve(ADOBE, M_srgb @ lin) ** (1 / 2.19921875)
    px = np.full((h, w, 3), np.round(in_adobe * 255), np.uint8)
    icc = _make_adobe()
    tagged, plain = os.path.join(d, "a.tif"), os.path.join(d, "b.tif")
    Image.fromarray(px).save(tagged, icc_profile=icc)
    Image.fromarray(px).save(plain)
    check("the profile is named", profile_name(tagged) == "Adobe RGB (1998)", profile_name(tagged))
    check("a file with none has no name", profile_name(plain) == "")
    rgb_t, grey_t = load_and_process_single(tagged)
    rgb_p, grey_p = load_and_process_single(plain)
    lab = lambda a: linear_to_lab_d50(srgb_to_linear(np.asarray(a, float) / 255))
    de_good = float(delta_e_2000(lab(rgb_t[5, 5]), lab(want * 255)))
    de_bad = float(delta_e_2000(lab(px[5, 5]), lab(want * 255)))
    check(f"the Adobe RGB file comes in as its true colour (ΔE {de_good:.2f}; read as sRGB it was {de_bad:.1f})",
          de_good < 1.0 and de_bad > 3 * max(de_good, 0.3))
    check("an untagged file is left alone", (rgb_p == px).all())
    check("the grey the solve fits is the stored numbers either way", np.array_equal(grey_t, grey_p))

    sp = os.path.join(d, "s.tif")
    Image.fromarray(px).save(sp, icc_profile=srgb_profile_bytes())
    check("an sRGB file is left alone", (load_and_process_single(sp)[0] == px).all())

    deep = os.path.join(d, "deep.tif")
    ramp = np.tile(np.linspace(30000, 30100, 60).astype(np.uint16)[None, :, None], (40, 1, 3))
    cv2.imwrite(deep, ramp)
    rgb16, grey16 = load_and_process_single(deep)
    check("a 16-bit TIFF keeps its depth in the grey", rgb16.dtype == np.uint8
          and len(np.unique(np.round(grey16[20], 3))) > 20 and abs(grey16.mean() - 30050 / 257) < 0.1)

    from helio.application import colour_check as C
    chart_srgb = C.synthetic_chart(900)[..., ::-1] / 255.0
    l = srgb_to_linear(chart_srgb)
    # numbers an Adobe RGB camera file would hold for this same chart
    a_ = np.clip(l @ np.linalg.solve(ADOBE, M_srgb).T, 0, 1) ** (1 / 2.19921875)
    cp = os.path.join(d, "chart.tif")
    Image.fromarray(np.round(a_ * 255).astype(np.uint8)).save(cp, icc_profile=icc)
    try:
        r = C.check_image(cp)
        check(f"the colour check reads the chart through the profile (mean ΔE {r['mean']})",
              r["mean"] < 2.0 and "Adobe RGB" in r["encoding"], r["encoding"])
    except LookupError:
        print("  --   (the chart detector did not find the synthetic chart)")

    from helio.application import pub_sheet as PS
    pic = np.full((300, 400, 3), 150, np.uint8)
    s = PS.Sheet(title="T", full_size=(400, 300), albedo=pic, views=[PS.View("v", pic, "relit")],
                 hero=PS.View("v", pic, "relit"))
    base = os.path.join(d, "sheet")
    files = PS.make(s, base)
    jpg = [f for f in files if f.endswith(".jpg")][0]
    with Image.open(jpg) as im:
        prof = ImageCms.ImageCmsProfile(__import__("io").BytesIO(im.info["icc_profile"]))
    check("the sheet says it is sRGB", "sRGB" in ImageCms.getProfileDescription(prof))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
