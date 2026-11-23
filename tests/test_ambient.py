"""The lights-off (ambient) frame: subtracted from every frame before the solve.

* subtract_ambient takes light away in linear units, whatever the encoding,
  and keeps the frame's shape and type.
* A synthetic surface lit by twelve lamps and a steady room light, written as
  a camera writes it (sRGB): with the room light the normals tilt toward the
  camera (the relief comes out shallow); with the lights-off frame subtracted
  they come back to within a little of the room-light-free solve.
* The same through a quick look (frames read small), a working size and a
  crop; the level is reported; a missing frame does not stop the solve.
* Sessions keep the frame.

    python tests/test_ambient.py
"""
import os
import sys
import tempfile

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
    import dataclasses
    import cv2
    import numpy as np
    from helio.core.radiometry import linear_to_srgb, srgb_to_linear, subtract_ambient
    print("=" * 60)
    print("Lights-off frame")
    print("=" * 60)

    print("\nThe subtraction")
    f = np.full((4, 4), 200, np.uint8)
    a = np.full((4, 4), 100, np.uint8)
    out = subtract_ambient(f, a)
    want = linear_to_srgb(srgb_to_linear(200.0) - srgb_to_linear(100.0))
    check("in linear light, encoded back (not 200 - 100)", out.dtype == np.uint8
          and abs(int(out[0, 0]) - float(want)) <= 1 and out[0, 0] != 100, out[0, 0])
    check("never below black", (subtract_ambient(a, f) == 0).all())
    raw = subtract_ambient(np.full((2, 2), 90.0, np.float32), np.full((2, 2), 30.0, np.float32),
                           frame_raw=True, ambient_raw=True)
    check("RAW frames: a plain difference", np.allclose(raw, 60.0) and raw.dtype == np.float32)
    col = subtract_ambient(np.full((2, 2, 3), 180, np.uint8), np.full((2, 2), 60, np.uint8))
    check("a grey ambient taken from a colour frame", col.shape == (2, 2, 3))

    print("\nA surface under twelve lamps and room light")
    h = w = 96
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.45 * w) ** 2
    z = np.sqrt(np.clip(1.0 - r2, 0, None)) * 0.45 * w                    # a dome
    gy, gx = np.gradient(z)
    n = np.dstack([-gx, gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = []
    for i in range(12):
        az, el = 2 * np.pi * i / 12, np.radians(35 + 10 * (i % 2))
        L.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    L = np.array(L, np.float32)
    room = 0.12                                                          # the room light, linear
    tmp = tempfile.mkdtemp(prefix="helio_amb_")
    clean, lit = [], []
    for i, lv in enumerate(L):
        lin = np.clip(n.reshape(-1, 3) @ lv, 0, None).reshape(h, w) * 0.7
        for paths, extra, tag in ((clean, 0.0, "c"), (lit, room, "r")):
            enc = np.round(linear_to_srgb(np.clip(lin + extra, 0, 1) * 255.0)).astype(np.uint8)
            p = os.path.join(tmp, f"{tag}{i:02d}.png")
            cv2.imwrite(p, np.dstack([enc] * 3))
            paths.append(p)
    dark = os.path.join(tmp, "lights_off.png")
    cv2.imwrite(dark, np.dstack([np.full((h, w), round(float(linear_to_srgb(room * 255.0))), np.uint8)] * 3))

    from helio.application.pipeline import SolveRequest, solve
    from helio.domain.settings import ProcessingSettings
    inner = (r2 < 0.6)
    inner[:6], inner[-6:], inner[:, :6], inner[:, -6:] = False, False, False, False

    def err(r):
        d = np.clip((r.normals_full * n).sum(-1), -1, 1)
        return float(np.degrees(np.arccos(d))[inner].mean())

    def tilt(r):            # how far the normals lean from the camera, on average
        return float(np.degrees(np.arccos(np.clip(r.normals_full[..., 2], -1, 1)))[inner].mean())
    base = ProcessingSettings(photo_tone="srgb")
    r_clean = solve(SolveRequest(paths=clean, lights=L, settings=base))
    r_room = solve(SolveRequest(paths=lit, lights=L, settings=base))
    r_sub = solve(SolveRequest(paths=lit, lights=L, settings=dataclasses.replace(base, ambient_frame=dark)))
    e_clean, e_room, e_sub = err(r_clean), err(r_room), err(r_sub)
    print(f"      error: no room light {e_clean:.2f}°, room light {e_room:.2f}°, subtracted {e_sub:.2f}°;"
          f"  lean {tilt(r_clean):.1f}° / {tilt(r_room):.1f}° / {tilt(r_sub):.1f}°")
    check("room light makes the relief shallower", tilt(r_room) < tilt(r_clean) - 2 and e_room > e_clean + 2)
    check("the lights-off frame takes it back out (within 0.5° of no room light)", e_sub < e_clean + 0.5,
          (e_clean, e_sub))
    check("the level is recorded (the room light was about 13% of the lit frames')",
          r_sub.ambient is not None and r_sub.ambient["file"] == "lights_off.png"
          and 0.05 < r_sub.ambient["level"] < 0.4 and r_room.ambient is None, r_sub.ambient)
    check("…and the colour is cleaned too", r_sub.albedo_rgb_full.mean() < r_room.albedo_rgb_full.mean() - 5)

    print("\nOther ways the frames are read")
    r_half = solve(SolveRequest(paths=lit, lights=L, settings=dataclasses.replace(
        base, ambient_frame=dark, work_scale=0.5)))
    small = cv2.resize(n, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    small /= np.linalg.norm(small, axis=-1, keepdims=True)
    keep = cv2.resize(inner.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST) > 0
    e_half = float(np.degrees(np.arccos(np.clip((r_half.normals_full * small).sum(-1), -1, 1)))[keep].mean())
    check(f"at half the working size ({e_half:.2f}°)", r_half.ambient is not None
          and r_half.w == w // 2 and e_half < e_room - 3, e_half)
    r_crop = solve(SolveRequest(paths=lit, lights=L, crop=(10, 10, 80, 80),
                                settings=dataclasses.replace(base, ambient_frame=dark)))
    check("with a crop", r_crop.ambient is not None and r_crop.w == 70)
    from helio.application import quick_look
    quick_look.forget()
    q, _s = quick_look.quick_solve(SolveRequest(paths=lit, lights=L,
                                                settings=dataclasses.replace(base, ambient_frame=dark)))
    check("in a quick look", q.ambient is not None)
    r_gone = solve(SolveRequest(paths=lit, lights=L, settings=dataclasses.replace(
        base, ambient_frame=os.path.join(tmp, "missing.png"))))
    check("a missing frame: solved without it", r_gone.ambient is None and r_gone.normals_full is not None)

    print("\nSessions")
    from helio.io import session as S
    s = S.Session(ambient_frame=dark)
    back = S.decode(S.encode(s))
    check("the frame is kept", back.ambient_frame == dark
          and back.processing_settings(ProcessingSettings()).ambient_frame == dark)
    check("…and left out when there is none", "ambient_frame" not in S.encode(S.Session()))
    moved = tempfile.mkdtemp(prefix="helio_amb_moved_")
    import shutil
    shutil.copy(dark, moved)
    gone = os.path.join(tempfile.gettempdir(), "nowhere_now", "lights_off.png")
    check("a capture copied elsewhere: found beside the photographs",
          S.locate(gone, "", moved) == os.path.join(moved, "lights_off.png"))
    check("…its own place first, and left as it was when nowhere",
          S.locate(dark, moved) == dark and S.locate(gone, tmp + "_none") == gone)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
