"""RAW frames reach every reader, not only the solve's.

cv2.imread finds nothing in a Sony or Phase One file and only a thumbnail in a
DNG, so the ball search, the drift check and the tools failed on a RAW capture.

* read_frame gives a RAW frame at the size the loader does, reduced as asked,
  display-encoded or linear; anything else is cv2.imread;
* folder import takes the camera files;
* on a real file (when one is on this machine) it all agrees with the loader.

    python tests/test_raw_frames.py
"""
import glob
import os
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    from helio.core.frame_read import read_frame
    from helio.io import load_and_process_single, raw_loader
    from helio.io.capture import captures_in, capture_extensions

    d = tempfile.mkdtemp(prefix="helio_raw_")
    png = os.path.join(d, "a.png")
    cv2.imwrite(png, np.full((40, 60, 3), 90, np.uint8))
    check("an ordinary file is cv2.imread", read_frame(png, cv2.IMREAD_GRAYSCALE).shape == (40, 60))

    # a RAW decode stood in for: half size, as half_size gives
    linear = np.full((300, 400, 3), 13107, np.uint16)                  # 0.2 of full scale
    was = raw_loader.is_raw, raw_loader.read_raw_sized
    raw_loader.is_raw = lambda p: p.endswith(".iiq")
    raw_loader.read_raw_sized = lambda p, half=False: (linear, (800, 600))
    try:
        f = os.path.join(d, "x.iiq")
        check("whole size from a half-size decode", read_frame(f, cv2.IMREAD_GRAYSCALE).shape == (600, 800))
        check("reduced as asked", read_frame(f, cv2.IMREAD_REDUCED_COLOR_4).shape == (150, 200, 3))
        lin = read_frame(f, cv2.IMREAD_GRAYSCALE, linear=True)
        enc = read_frame(f, cv2.IMREAD_GRAYSCALE)
        check(f"linear for values ({lin.mean():.1f}), encoded for looking ({enc.mean():.0f})",
              abs(lin.mean() - 51) < 0.5 and abs(enc.mean() - 124) < 2)
    finally:
        raw_loader.is_raw, raw_loader.read_raw_sized = was

    if raw_loader.available():
        check("Phase One, Sony, Canon, Nikon, DNG are capture files",
              {".iiq", ".arw", ".cr3", ".nef", ".dng"} <= capture_extensions())
        for n in ("a.iiq", "b.IIQ", "c.arw", "notes.txt"):
            open(os.path.join(d, n), "w").close()
        check("folder import takes them", sum(p.lower().endswith((".iiq", ".arw")) for p in captures_in([d])) == 3)

    # the light cube: RAW at the solve's size, 16-bit and linear, at a working size too
    from helio.application.export_archive import light_cube
    ramp = np.tile(np.linspace(1000, 40000, 800).astype(np.uint16)[None, :, None], (600, 1, 3))
    was = raw_loader.is_raw, raw_loader.read_raw_sized
    raw_loader.is_raw = lambda p: p.endswith(".iiq")
    raw_loader.read_raw_sized = lambda p, half=False: (ramp, (800, 600))
    try:
        fr = [os.path.join(d, f"f{i}.iiq") for i in range(3)]
        cube = light_cube(fr, os.path.join(d, "cube.tif"), want_shape=(600, 800))
        small = light_cube(fr, os.path.join(d, "cube2.tif"), want_shape=(150, 200), base=(200, 150))
    finally:
        raw_loader.is_raw, raw_loader.read_raw_sized = was
    import tifffile
    a = tifffile.imread(cube["path"])
    check("the light cube is RAW frames at the solve's size, 16-bit linear",
          a.shape[-2:] == (600, 800) and a.dtype == np.uint16 and abs(int(a.max()) - 40000) < 300, (a.shape, a.dtype, a.max()))
    check("…and at a working size below full", tifffile.imread(small["path"]).shape[-2:] == (150, 200))

    # a dark RAW stack is brought up by one gain, the ratios between frames kept
    from helio.application.pipeline import _raw_exposure
    stack = np.stack([np.full((64, 64), v, np.float32) for v in (20.0, 10.0, 5.0)])
    rgbs = [np.full((64, 64, 3), 60, np.uint8) for _ in stack]
    up, cols, k = _raw_exposure(stack, rgbs, [True] * 3)
    check(f"a dark RAW stack comes up by one gain ({k:.1f})", abs(k - 9) < 0.1 and abs(float(up[0].mean()) - 180) < 1
          and abs(float(up[0].mean() / up[2].mean()) - 4) < 1e-4 and cols[0].mean() > 120)
    check("a TIFF stack, a mixed one or a bright one is left alone",
          _raw_exposure(stack, rgbs, [False] * 3)[2] == 1.0 and _raw_exposure(stack, rgbs, [True, False, True])[2] == 1.0
          and _raw_exposure(stack * 10, rgbs, [True] * 3)[2] == 1.0)
    from helio.application.quick_look import _size
    was_raw = raw_loader.is_raw, raw_loader.raw_size
    raw_loader.is_raw, raw_loader.raw_size = (lambda p: True), (lambda p: (8708, 11608))
    try:
        check("the quick size check asks the RAW decoder, not the thumbnail", _size("x.dng") == (8708, 11608))
    finally:
        raw_loader.is_raw, raw_loader.raw_size = was_raw

    real = sorted(glob.glob(os.path.expanduser("~/Pictures/ORDERS/Capture/BSS-00709*.iiq")))[:1]
    if real and raw_loader.available():
        from helio.core.sphere import image_size
        p = real[0]
        rgb, grey = load_and_process_single(p)
        g = read_frame(p, cv2.IMREAD_GRAYSCALE)
        check(f"a real IIQ: the size from its header is the loader's {rgb.shape[1::-1]}",
              image_size(p) == rgb.shape[1::-1] and g.shape == grey.shape)
        c = np.corrcoef(cv2.resize(g, (200, 150)).ravel(),
                        cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (200, 150)).ravel())[0, 1]
        check(f"…and the fast read matches it ({c:.4f})", c > 0.99)
        lin = read_frame(p, cv2.IMREAD_GRAYSCALE, linear=True)
        check("…the linear read is at the loader's level",
              abs(float(lin.mean()) / float(grey.mean()) - 1) < 0.1, (lin.mean(), grey.mean()))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
