"""The 3D relief as really drawn, on the graphics card.

The other tests run offscreen, where there is no OpenGL, so nothing checked the
pictures themselves.  This opens a window and reads back what is drawn: lit from
the right side, fine normals shading the surface, sharper close up with detail
textures, every material, and still drawing after the surface, size and
visibility change.  Skips without a display.

    python tests/test_relief_gl.py
"""
import os
import sys
import tempfile

os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

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


def normals_of(depth, fine=None):
    gy, gx = np.gradient(depth)
    if fine is not None:
        gx = gx + fine
    n = np.dstack([-gx, gy, np.ones_like(gx)])      # +Y up: rows run down
    return (n / np.linalg.norm(n, axis=2, keepdims=True)).astype(np.float32)


def hf(pic):
    g = cv2.cvtColor(pic, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape
    g = g[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return float((g - cv2.GaussianBlur(g, (0, 0), 3)).std())


def main():
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        print("Results: 0 passed, 0 failed out of 0 tests (no display)")
        return 0
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from helio_qt.relief3d import MATERIALS, ReliefView, gl_available
    ok, why = gl_available()
    if not ok:
        print(f"Results: 0 passed, 0 failed out of 0 tests ({why})")
        return 0
    v = ReliefView()
    v.resize(900, 700)
    v.show()
    for _ in range(20):
        app.processEvents()
    N = 512
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    bump = 40 * np.exp(-((xx - N / 2) ** 2 + (yy - N / 2) ** 2) / (2 * 90.0 ** 2))
    alb = np.full((N, N, 3), 170, np.uint8)
    ripple = 0.5 * np.sin(xx * 2 * np.pi / 6)                       # in the normals, not in the heights
    v.set_surface(bump, normals_of(bump, ripple), alb, roughness=np.full((N, N), 128, np.uint8),
                  trust=np.zeros((N, N), np.float32))
    for _ in range(20):
        app.processEvents()
    check("it draws", v.gl_ok and not v.gl_error, v.gl_error)

    def shot():
        for _ in range(5):
            app.processEvents()
        return v.picture_at(800, 600)
    pic = shot()
    check(f"a surface, not a blank (mean {pic.mean():.0f}, sd {pic.std():.0f})", 30 < pic.mean() < 240 and pic.std() > 5)

    v.set_material("clay")
    v.set_light_screen(0.0, -0.8, 0.6)            # the screen frame: +Y down
    top = cv2.cvtColor(shot(), cv2.COLOR_RGB2GRAY).astype(np.float32)
    v.set_light_screen(0.0, 0.8, 0.6)
    low = cv2.cvtColor(shot(), cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = top - low
    H = d.shape[0]
    check(f"lit from above the bump's top is the brighter ({d[H // 3:H // 2].mean():.0f} vs {d[H // 2:2 * H // 3].mean():.0f})",
          d[H // 3:H // 2].mean() > d[H // 2:2 * H // 3].mean() + 3)
    v.set_light_screen(0.7, 0.0, 0.7)
    right = cv2.cvtColor(shot(), cv2.COLOR_RGB2GRAY).astype(np.float32)
    v.set_light_screen(-0.7, 0.0, 0.7)
    left = cv2.cvtColor(shot(), cv2.COLOR_RGB2GRAY).astype(np.float32)
    e = right - left
    W = e.shape[1]
    check("lit from the right, its right side", e[:, W // 2:2 * W // 3].mean() > e[:, W // 3:W // 2].mean() + 3)

    v.normal_mapped = False
    v.smooth_shading = False
    v.update()
    v.cam.el = 20.0
    v.set_exaggeration(0.2)
    flat = cv2.cvtColor(shot(), cv2.COLOR_RGB2GRAY).astype(np.float32)
    v.set_exaggeration(6.0)
    tall = cv2.cvtColor(shot(), cv2.COLOR_RGB2GRAY).astype(np.float32)
    v.set_exaggeration(1.0)
    v.cam.el = 55.0
    v.smooth_shading = True
    v.normal_mapped = True
    check(f"the heights lift the mesh (exaggeration moves the picture by {np.abs(tall - flat).mean():.0f})",
          np.abs(tall - flat).mean() > 2)

    v.set_light_screen(0.5, 0.3, 0.8)
    v.normal_mapped = True
    v.update()
    with_n = hf(shot())
    v.normal_mapped = False
    v.update()
    without = hf(shot())
    v.normal_mapped = True
    v.update()
    check(f"fine detail in the normals shades the surface ({with_n:.1f} against {without:.1f} without)",
          with_n > 2 * without and with_n > 1)

    # sharper textures than the mesh: a ripple the mesh cannot hold
    M = 2048
    yy2, xx2 = np.mgrid[0:M, 0:M].astype(np.float32)
    sharp = normals_of(cv2.resize(bump, (M, M)) * 4, 0.5 * np.sin(xx2 * 2 * np.pi / 5))
    close = v.cam.dist
    v.cam.dist = close / 8
    v.update()
    before = hf(shot())
    v.set_detail(sharp, None)
    after = hf(shot())
    check(f"close up, detail textures draw detail the mesh has not ({before:.1f} to {after:.1f})", after > before + 0.5
          or after > 1.5 * before)
    v.set_detail(None, None)
    v.cam.dist = close
    v.update()

    pics = {}
    for k, _t in MATERIALS:
        v.set_material(k)
        pics[k] = shot()
        check(f"material {k}: draws", pics[k].std() > 1 and not v.gl_error, (pics[k].std(), v.gl_error))
    v.set_material("colour")

    v.adaptive = True
    v.update()
    a = shot()
    check(f"the adaptive mesh draws (tessellation {'on' if v.tess_ok else 'not available'})",
          a.std() > 5 and v.tess_ok is not False, v.tess_error if hasattr(v, "tess_error") else "")
    v.set_fine_heights(np.sin(xx / 3).astype(np.float32) * 0.01)
    v.cam.dist = close / 6
    v.update()
    check("…lifted by fine heights", shot().std() > 5 and not v.gl_error, v.gl_error)
    v.cam.dist = close
    v.adaptive = False
    for mode in ("over", "only", ""):
        v.set_wire_mode(mode)
        check(f"wire mode '{mode}': draws", shot().std() > 1 and not v.gl_error, v.gl_error)

    other = 20 * np.sin(xx[:300, :400] / 30)
    v.set_surface(other, normals_of(other), np.full((300, 400, 3), 120, np.uint8))
    p2 = shot()
    check("a new surface of another shape replaces the old", p2.std() > 3 and not v.gl_error, v.gl_error)
    v.hide()
    app.processEvents()
    v.show()
    v.resize(500, 400)
    check("hidden, shown and resized, it still draws", shot().std() > 3 and not v.gl_error, v.gl_error)

    big = np.dstack([np.zeros((11608, 8708), np.float32)] * 2 + [np.ones((11608, 8708), np.float32)])
    big[..., 0] = np.sin(np.arange(8708, dtype=np.float32) * 2 * np.pi / 5) * 0.4
    big /= np.linalg.norm(big, axis=2, keepdims=True)
    side = 11608 <= v.max_texture
    v.set_surface(bump, normals_of(bump), alb)
    v.set_detail(big, None)
    check(f"a 100 MP normal map goes up to the card (limit {v.max_texture:,})",
          side and shot().std() > 3 and not v.gl_error, v.gl_error)
    v.close()
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
