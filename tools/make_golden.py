"""Generate golden reference renders for the shading engines, plus a contact
sheet for human approval.

The surface is analytic — a dome, a raised ridge, an incised groove and a
stepped block over a textured plane — so the input is bit-identical on every
machine and no photometric solve is involved.  That isolates the *render*
engines: any later pixel change means an engine changed, not that the solver
drifted.

    python tools/make_golden.py            # write references + contact sheet
    python tools/make_golden.py --sheet    # rebuild only the contact sheet

The references are NOT self-approving.  Look at the contact sheet and confirm
the renders are correct before trusting `tests/test_golden.py`; after that, every
future change is checked automatically.
"""
import os
import sys
import multiprocessing

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PICTURES = (".jpg", ".jpeg", ".tif", ".tiff", ".png")


def test_images(passed=None, failed=None):
    """The folder of test photographs (a real eight-frame capture with a chrome
    ball): the one HELIO_TEST_IMAGES names, else "test images" beside the code.
    The public copy leaves them out (they are not ours to publish), so None when
    there are none; given the test's counts, the test ends here instead,
    reporting what ran and skipping the rest."""
    d = os.environ.get("HELIO_TEST_IMAGES") or os.path.join(ROOT, "test images")
    if os.path.isdir(d) and any(f.lower().endswith(PICTURES) for f in os.listdir(d)):
        return d
    if passed is None:
        return None
    print("\n  --   the test photographs are not here (set HELIO_TEST_IMAGES to their folder): "
          "the rest of this test is skipped")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    sys.exit(1 if failed else 0)


GOLDEN_DIR = os.path.join("tests", "golden")
SHEET = os.path.join("tests", "golden_contact_sheet.png")
H, W = 240, 320


def analytic_surface():
    """Deterministic height field with features each engine should reveal."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H
    z = np.zeros((H, W), np.float32)

    # dome
    d = np.sqrt((u - 0.25) ** 2 + (v - 0.5) ** 2)
    z += np.where(d < 0.16, np.sqrt(np.clip(0.16 ** 2 - d ** 2, 0, None)), 0)

    # raised ridge
    z += 0.06 * np.exp(-((u - 0.55) ** 2) / (2 * 0.012 ** 2))

    # incised groove
    z -= 0.05 * np.exp(-((v - 0.30) ** 2) / (2 * 0.010 ** 2))

    # stepped block
    z += np.where((u > 0.72) & (u < 0.92) & (v > 0.55) & (v < 0.85), 0.05, 0)

    # fine deterministic texture
    z += 0.004 * np.sin(60 * np.pi * u) * np.cos(48 * np.pi * v)

    z = cv2.GaussianBlur(z.astype(np.float32), (0, 0), 1.0)

    # Normals +Y up, the frame calibrated captures arrive in (helio.core.frames).
    # Until FIX-003 they were y-down, so every directional engine here was lit
    # from the mirrored side and the old integrators looked right (KD-009).
    gy, gx = np.gradient(z)
    n = np.dstack([-gx * 40.0, gy * 40.0, np.ones_like(z)]).astype(np.float32)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)

    albedo = np.zeros((H, W, 3), np.float32)
    albedo[..., 0] = 0.55 + 0.25 * np.sin(8 * np.pi * u)
    albedo[..., 1] = 0.50 + 0.20 * np.cos(6 * np.pi * v)
    albedo[..., 2] = 0.45
    albedo = (np.clip(albedo, 0, 1) * 255).astype(np.uint8)
    return z.astype(np.float32), n, albedo


def render_stack(tmp, n_lights=8):
    """Render the analytic surface under n lights — a real input stack.

    Injecting normals/depth by hand is not enough: two thirds of the engines
    also need PTM/HSH coefficients, terrain maps, PCA and specular variance,
    which only the real pipeline produces.  So we synthesise *images*, feed
    them through processing, and render from the genuine result.
    """
    z, n, albedo = analytic_surface()
    # Keep the albedo in COLOUR. It used to be collapsed with RGB2GRAY and
    # written back out as GRAY2BGR, which made R == G == B in every frame —
    # so "RGB-PS R-B Diff" could only ever output exactly zero, and its
    # reference was a pure-black image that no regression could ever disturb
    # (the "identical black images compare equal forever" trap). The three
    # channels of analytic_surface() carry different spatial patterns, which
    # is what the colour engines need in order to be exercised at all.
    alb = albedo.astype(np.float32) / 255.0                  # (H, W, 3) RGB
    L, paths = [], []
    for i in range(n_lights):
        az = 2 * np.pi * i / n_lights
        el = np.radians(40.0)
        lv = np.array([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az),   # +Y up
                       np.sin(el)], np.float32)
        lv /= np.linalg.norm(lv)
        L.append(lv)
        shade = np.clip(n @ lv, 0, None)[..., None]          # (H, W, 1)
        lit = shade * (0.35 + 0.65 * alb)                    # per-channel
        img = (np.clip(lit, 0, 1) * 245 + 5).astype(np.uint8)
        p = os.path.join(tmp, f"g_{i:02d}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(p)
    return paths, np.array(L, np.float32)


def build_app():
    """Build the app and run the real pipeline on a deterministic stack."""
    import tempfile
    multiprocessing.set_start_method("spawn", force=True)
    import Helio_Pro

    class _Box:
        def __getattr__(self, n):
            return lambda *a, **k: "ok"
    Helio_Pro.messagebox = _Box()

    app = Helio_Pro.RTIViewerPro()
    app.update()

    tmp = tempfile.mkdtemp(prefix="helio_golden_")
    paths, L = render_stack(tmp)
    app.data.plate_paths = list(paths)
    app.loaded_lp_matrix = L.copy()
    app.data.current_L_mat = L.copy()
    app.data.normal_engine.set("Linear Least Squares")
    app.data.depth_engine.set("Fast Poisson (FFT)")
    # render_stack writes radiance straight into the PNGs: linear values, not
    # a camera's sRGB, so the solve must not decode them (FIX-010).
    app.photo_tone_var.set("linear")

    state = {}

    def wait(t=0):
        th = getattr(app, "_proc_thread_obj", None)
        if th is not None and not th.is_alive() and \
                getattr(app.data, "normals_full", None) is not None:
            state["ok"] = True
            app.quit()
        elif t > 1200:
            state["timeout"] = True
            app.quit()
        else:
            app.after(200, lambda: wait(t + 1))

    app.after(200, lambda: (app.run_processing(), wait()))
    app.mainloop()
    if not state.get("ok"):
        raise SystemExit(f"golden fixture failed to process: {state}")
    return app


def render_all(app):
    out = {}
    for eng in app.engines_list:
        try:
            img = app.get_shading(H, W, False, engine_override=eng)
        except Exception as e:
            print(f"   !! {eng}: {type(e).__name__}: {e}")
            continue
        if img is None:
            print(f"   !! {eng}: returned None")
            continue
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = a.astype(np.float32)
            # get_shading returns float in 0..1 for most engines and 0..255 for
            # a few.  Clipping a 0..1 image straight to uint8 crushes it to
            # black — scale by 255 when the range says it is normalised.
            finite = a[np.isfinite(a)]
            hi = float(finite.max()) if finite.size else 0.0
            if hi <= 1.0 + 1e-6:
                a = a * 255.0
            a = np.clip(np.nan_to_num(a), 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        out[eng] = a[..., :3]
    return out


def safe(name):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def contact_sheet(renders, path):
    cols = 5
    rows = (len(renders) + cols - 1) // cols
    pad, label_h = 8, 22
    cw, ch = W, H + label_h
    sheet = np.full((rows * (ch + pad) + pad, cols * (cw + pad) + pad, 3),
                    18, np.uint8)
    for i, (eng, img) in enumerate(sorted(renders.items())):
        r, c = divmod(i, cols)
        y = pad + r * (ch + pad)
        x = pad + c * (cw + pad)
        sheet[y + label_h:y + label_h + H, x:x + W] = img[..., ::-1]
        cv2.putText(sheet, eng[:38], (x + 2, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1,
                    cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, sheet)
    return sheet.shape


def main():
    app = build_app()
    print(f"rendering {len(app.engines_list)} engines at {W}x{H}")
    renders = render_all(app)
    print(f"got {len(renders)} renders")

    # Record the optional-dependency state these references were built under.
    # RVT changes what several engines draw, so a reference set is only
    # meaningful alongside the answer to "was RVT installed?".
    try:
        import json
        from helio.core.shading_engines import HAS_RVT
        os.makedirs(os.path.join("tests", "golden"), exist_ok=True)
        with open(os.path.join("tests", "golden", "_env.json"), "w") as _f:
            json.dump({"has_rvt": bool(HAS_RVT)}, _f, indent=1)
        print(f"   environment: HAS_RVT={HAS_RVT}")
    except Exception as _e:
        print(f"   !! could not record environment: {_e}")

    if "--sheet" not in sys.argv:
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        for f in os.listdir(GOLDEN_DIR):
            os.remove(os.path.join(GOLDEN_DIR, f))
        for eng, img in renders.items():
            cv2.imwrite(os.path.join(GOLDEN_DIR, safe(eng) + ".png"),
                        img[..., ::-1])
        print(f"wrote {len(renders)} references to {GOLDEN_DIR}/")

    shape = contact_sheet(renders, SHEET)
    print(f"contact sheet: {SHEET}  {shape[1]}x{shape[0]}")
    print("\nREVIEW THE CONTACT SHEET before trusting tests/test_golden.py.")
    try:
        app.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
