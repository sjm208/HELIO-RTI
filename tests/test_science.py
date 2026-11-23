"""The 2026-10-07 science fixes, each pinned to a measured number.

* FIX-010  Linear light: camera photos are decoded from sRGB before the
  normal solve.  A perfect synthetic surface, written as an sRGB camera
  would write it, is solved end to end through the shared pipeline; the
  normals must come back within 0.3° mean (they were off by over 2°).
* FIX-011  Normal uncertainty: the per-pixel 1-sigma angle matches the
  noise put in, and is large where a pixel's lights disagree.
* Near-field solver: the batched solve equals the per-pixel loop it
  replaced.
* REC-020/021  Least-squares and bilateral integration (helio.core.bini):
  exact on a smooth surface, and flat beside a steep step where the FFT and
  DCT integrators tilt.
* Sessions record the photo tone; an older session means "stored".

    python tests/test_science.py
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


def angle_err(a, b):
    import numpy as np
    d = np.clip((a * b).sum(-1), -1, 1)
    return np.degrees(np.arccos(d))


def main():
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    import cv2
    import numpy as np
    from tools.make_golden import analytic_surface
    from helio.core.radiometry import linear_to_srgb, solve_intensities, srgb_to_linear
    print("=" * 60)
    print("Science fixes")
    print("=" * 60)

    print("\nLinear light (FIX-010)")
    x = np.linspace(0, 255, 256, dtype=np.float32)
    check("sRGB decode and encode are inverses", np.abs(linear_to_srgb(srgb_to_linear(x)) - x).max() < 0.01)
    check("mid-grey 128 is about 22% of white's light", abs(srgb_to_linear(128.0) / 255 - 0.2158) < 0.001)
    stack = np.full((3, 2, 2), 128.0, np.float32)
    out = solve_intensities(stack, "srgb", [False, True, False])
    check("RAW frames are left linear, photos decoded", out[1, 0, 0] == 128.0
          and abs(out[0, 0, 0] - srgb_to_linear(128.0)) < 1e-4)
    check("'stored' and 'linear' leave the stack alone",
          solve_intensities(stack, "stored") is stack and solve_intensities(stack, "linear") is stack)

    z, n, _alb = analytic_surface()
    h, w = n.shape[:2]
    L = []
    for i in range(12):
        az, el = 2 * np.pi * i / 12, np.radians(35 + 10 * (i % 2))
        L.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    L = np.array(L, np.float32)
    tmp = tempfile.mkdtemp(prefix="helio_lin_")
    paths, lin_paths = [], []
    for i, lv in enumerate(L):
        lin = np.clip(n.reshape(-1, 3) @ lv, 0, None).reshape(h, w) * 0.85     # radiance
        enc = np.round(linear_to_srgb(lin * 255.0)).astype(np.uint8)            # as a camera writes it
        p = os.path.join(tmp, f"s{i:02d}.png")
        cv2.imwrite(p, np.dstack([enc] * 3))
        paths.append(p)
        q = os.path.join(tmp, f"l{i:02d}.png")                                  # a linear TIFF's values
        cv2.imwrite(q, np.dstack([np.round(lin * 255.0).astype(np.uint8)] * 3))
        lin_paths.append(q)
    from helio.application.pipeline import SolveRequest, solve
    from helio.domain.settings import ProcessingSettings
    lit = (np.clip(n.reshape(-1, 3) @ L.T, 0, None) > 0.05).all(axis=1).reshape(h, w)
    inner = np.zeros((h, w), bool)
    inner[8:-8, 8:-8] = True
    sel = lit & inner
    errs = {}
    for key, ps_, tone in (("floor", lin_paths, "linear"), ("stored", paths, "stored"),
                           ("srgb", paths, "srgb")):
        r = solve(SolveRequest(paths=ps_, lights=L,
                               settings=ProcessingSettings(photo_tone=tone)))
        e = angle_err(r.normals_full, n)[sel]
        errs[key] = (float(e.mean()), float(np.percentile(e, 95)))
    print("      mean / 95th percentile error: "
          + ";  ".join(f"{k} {v[0]:.2f}° / {v[1]:.2f}°" for k, v in errs.items()))
    # The floor is the pipeline's own (the 1.5 px load blur against a sharp
    # truth), the same for linear input solved as linear.
    check("solved as stored, sRGB photos are well off the floor (the old behaviour)",
          errs["stored"][0] > errs["floor"][0] + 1.0 and errs["stored"][1] > 2 * errs["floor"][1])
    check("decoded to linear light they reach it (within 0.1° mean, 0.3° at the 95th)",
          abs(errs["srgb"][0] - errs["floor"][0]) < 0.1
          and errs["srgb"][1] < errs["floor"][1] + 0.3, errs)

    # Automatic (the default): decoded, evened lighting or not (DiLiGenT-Pi: as stored,
    # fine relief came out at 60% of its depth; until 2026-11-12 it was solved as stored).
    r = solve(SolveRequest(paths=paths, lights=L, settings=ProcessingSettings()))
    check("automatic, lighting not evened: decoded, and the result says so",
          r.settings.photo_tone == "srgb"
          and abs(float(angle_err(r.normals_full, n)[sel].mean()) - errs["srgb"][0]) < 1e-3)
    r = solve(SolveRequest(paths=paths, lights=L, settings=ProcessingSettings(
        even_flat=True, flat_areas=f"4,4,{w // 3},{h // 3}")))
    check("automatic, flat areas marked: decoded, and the result says so",
          r.settings.photo_tone == "srgb")

    from helio.core.photometric_stereo import (intensity_weighted_ls, irls_photometric_stereo,
                                               shadow_rejection_ls)
    flat = (np.clip(n.reshape(-1, 3) @ L.T, 0, None).T * 200).astype(np.float32)
    enc = linear_to_srgb(flat)
    for name, fn in (("Robust LS", lambda: intensity_weighted_ls(L, flat, h, w, I_ref=enc)),
                     ("Shadow-Rejection LS", lambda: shadow_rejection_ls(L, flat, h, w, 12, I_ref=enc)),
                     ("IRLS", lambda: irls_photometric_stereo(L, flat, h, w, I_ref=enc))):
        e = angle_err(fn(), n)[sel]
        check(f"{name} solves linear light with its weights judged on stored values "
              f"(mean {e.mean():.2f}°)", e.mean() < 1.0)

    print("\nNormal uncertainty (FIX-011)")
    from helio.core.processing import stack_quality_maps
    rng = np.random.default_rng(3)
    sigma_I = 2.0
    clean = np.clip(n.reshape(-1, 3) @ L.T, 0, None).T.reshape(len(L), h, w) * 180.0
    noisy = (clean + rng.normal(0, sigma_I, clean.shape)).astype(np.float32)
    qm = stack_quality_maps(noisy, L, h, w, solve_stack=noisy)
    sd = qm.get("sigma_deg")
    check("a map of degrees per pixel", sd is not None and sd.shape == (h, w) and sd.dtype == np.float32)
    # Theory where every light sees the pixel: sigma_theta ~ sigma_I / rho * sqrt(tr - nCn)
    Ci = np.linalg.inv(L.T.astype(np.float64) @ L.astype(np.float64))
    nn = n.reshape(-1, 3)
    pred = np.degrees(sigma_I / 180.0 * np.sqrt(np.trace(Ci) - np.einsum('pi,ij,pj->p', nn, Ci, nn)))
    pred = pred.reshape(h, w)
    ratio = float(np.median(sd[sel] / pred[sel]))
    check(f"it matches least-squares theory for the noise put in (ratio {ratio:.2f})",
          0.8 < ratio < 1.25)
    # A dome steep enough at its edge for some lights to be behind it
    # (attached shadow): those frames read 0 where the model says negative.
    yy0, xx0 = np.mgrid[0:96, 0:96].astype(float)
    rr0 = np.hypot(xx0 - 48, yy0 - 48) / 44.0
    nd = np.dstack([(xx0 - 48) / 44.0, (48 - yy0) / 44.0, np.sqrt(np.clip(1 - rr0 ** 2, 0, 1))])
    nd[rr0 >= 1] = (0, 0, 1)
    nd /= np.linalg.norm(nd, axis=2, keepdims=True)
    dots = nd.reshape(-1, 3) @ L.T
    dome = (np.clip(dots, 0, None).T * 180.0).reshape(len(L), 96, 96).astype(np.float32)
    sd0 = stack_quality_maps(dome, L, 96, 96)["sigma_deg"]
    behind = (dots < -0.05).any(axis=1).reshape(96, 96)
    facing = (dots > 0.05).all(axis=1).reshape(96, 96)
    check("without noise, doubtful only where some lights are behind the surface",
          float(np.median(sd0[behind])) > 0.5 and float(np.median(sd0[facing])) < 0.01,
          (float(np.median(sd0[behind])), float(np.median(sd0[facing]))))

    print("\nNear-field solver")
    from helio.core.photometric_stereo import solve_near_field_normals

    def loop(images, L_mat, hh, ww):
        R = max(hh, ww) * 1.5
        xx, yy = np.meshgrid(np.arange(ww) - (ww - 1) / 2.0, (hh - 1) / 2.0 - np.arange(hh))   # pixel centres
        S = L_mat * R
        I_flat = images.reshape(images.shape[0], -1)
        out = np.zeros((hh * ww, 3), np.float32)
        for i in range(hh * ww):
            Lp = S - np.array([xx.ravel()[i], yy.ravel()[i], 0.0])
            d = np.linalg.norm(Lp, axis=1, keepdims=True)
            g = np.linalg.lstsq((Lp / d) * (R * R / (d * d)), I_flat[:, i], rcond=None)[0]
            m = np.linalg.norm(g)
            if m > 0:
                out[i] = g / m
        return out.reshape(hh, ww, 3)
    imgs = rng.uniform(5, 250, size=(len(L), 21, 29)).astype(np.float32)
    check("the batched solve equals the per-pixel loop",
          np.abs(loop(imgs, L, 21, 29) - solve_near_field_normals(imgs, L, 21, 29)).max() < 1e-5)

    # A flat, white plane under lamps whose directions were measured on a ball
    # at the left of the frame: centred there, it solves flat everywhere;
    # centred on the middle, the far side leans.
    hh, ww = 60, 90
    ball = (-35.0, 10.0)                                   # px from the centre, +Y up
    R = max(hh, ww) * 1.5
    xx, yy = np.meshgrid(np.arange(ww) - (ww - 1) / 2.0, (hh - 1) / 2.0 - np.arange(hh))   # pixel centres
    lamps = L * R + np.array([ball[0], ball[1], 0.0])
    P = np.stack([xx, yy, np.zeros_like(xx)], -1)
    flat_imgs = []
    for s_ in lamps:
        v = s_ - P
        d = np.linalg.norm(v, axis=-1)
        flat_imgs.append(np.clip(v[..., 2] / d, 0, None) * (R * R / (d * d)) * 200.0)
    flat_imgs = np.array(flat_imgs, np.float32)

    def lean(n):
        return float(np.degrees(np.arccos(np.clip(n[..., 2], -1, 1))).max())
    from_ball = solve_near_field_normals(flat_imgs, L, hh, ww, origin=ball)
    from_mid = solve_near_field_normals(flat_imgs, L, hh, ww)
    check(f"lamps measured on a ball at the side: centred there the plane is flat "
          f"({lean(from_ball):.2f}°), on the middle it leans ({lean(from_mid):.1f}°)",
          lean(from_ball) < 0.1 and lean(from_mid) > 2.0)
    from helio.application.pipeline import near_field_origin
    from helio.domain.settings import ProcessingSettings
    ps_ = ProcessingSettings(near_field=True)
    check("the ball's place in the frame solved, a region cut out of it too",
          near_field_origin(ps_, (100.0, 50.0), None, 400, 600) == (-200.0, 150.0)
          and near_field_origin(ps_, (100.0, 50.0), (50, 20, 250, 220), 200, 200) == (-50.0, 70.0))
    check("'the centre', or no ball known, centres it on the middle as before",
          near_field_origin(ProcessingSettings(near_field_from="centre"), (100.0, 50.0), None, 400, 600)
          == (0.0, 0.0) and near_field_origin(ps_, None, None, 400, 600) == (0.0, 0.0))

    print("\nLeast-squares and bilateral integration (REC-020, REC-021)")
    from helio.core.bini import integrate
    from helio.core.depth_integration import poisson_solver, simchony_depth_integration
    from helio.core.frames import normal_slopes
    S = 256
    yy, xx = np.mgrid[0:S, 0:S].astype(float)

    def normals_of(zz, noise=0.0):
        gx, gy = np.gradient(zz, axis=1), np.gradient(zz, axis=0)
        nz = 1 / np.sqrt(1 + gx ** 2 + gy ** 2)
        nn_ = np.dstack([-gx * nz, gy * nz, nz])
        nn_ += rng.normal(0, noise, nn_.shape)
        return (nn_ / np.linalg.norm(nn_, axis=2, keepdims=True)).astype(np.float32)

    def rms(a, b):
        return float(np.sqrt(np.mean(((a - a.mean()) - (b - b.mean())) ** 2)))
    z1 = 8 * np.sin(xx / 25) * np.cos(yy / 33)
    n1 = normals_of(z1)
    e_ls, e_dct = rms(integrate(n1, bilateral=False), z1), rms(simchony_depth_integration(n1), z1)
    check(f"least squares is exact on a smooth surface ({e_ls:.3f} px; DCT {e_dct:.2f})",
          e_ls < 0.05 and e_ls < e_dct / 10)
    rr = np.hypot(xx - S / 2, yy - S / 2)
    z2 = np.clip((60 - rr) * 6.0, 0, 40.0)          # a raised disc with an 80° bevel
    n2 = normals_of(z2, noise=0.05)
    bg = rr > 75

    def flat_err(a):
        A = np.c_[xx[bg], yy[bg], np.ones(bg.sum())]
        c, *_ = np.linalg.lstsq(A, a[bg], rcond=None)
        return float(np.sqrt(np.mean((a[bg] - A @ c) ** 2)))
    fb = flat_err(integrate(n2, bilateral=True))
    fd = flat_err(simchony_depth_integration(n2))
    fp = flat_err(poisson_solver(*normal_slopes(n2)))
    check(f"beside a steep step the ground stays flat ({fb:.2f} px; DCT {fd:.2f}, FFT {fp:.2f})",
          fb < 0.3 and fb * 5 < min(fd, fp))
    zc = integrate(n1, bilateral=True, max_px=4000)
    check("a large map is solved reduced, detail added back", zc.shape == z1.shape
          and rms(zc, z1) < 1.0, rms(zc, z1))
    from helio.application.pipeline import integrate_depth
    from helio.domain.registry import DEPTH_METHODS
    check("both are offered as depth methods",
          {"Least-Squares (Free Boundary)", "Bilateral (Keeps Steps)"} <= set(DEPTH_METHODS)
          and integrate_depth(n1, "Least-Squares (Free Boundary)").shape == z1.shape)

    print("\nSessions record the photo tone")
    from helio.io import session as Sx
    p = os.path.join(tmp, "t.helio")
    Sx.write(p, Sx.Session(object_id="x", photo_tone="srgb"))
    check("written and read back", Sx.read(p).photo_tone == "srgb"
          and Sx.read(p).processing_settings(ProcessingSettings()).photo_tone == "srgb")
    Sx.write(p, Sx.Session(object_id="old"))
    import json
    check("a session without it writes no key, as before", "photo_tone" not in json.load(open(p)))
    check("and means 'stored' (how it was solved)",
          Sx.read(p).processing_settings(ProcessingSettings()).photo_tone == "stored")
    # FIX-010 made decoding the default on 2026-10-07; KD-052 put "stored" back on
    # 2026-10-14 (decoding doubled every slope on real captures, false tilt included);
    # in November 2026 "auto" decoded only when the lighting was evened (docs/BENCHMARK_DILIGENT.md);
    # from 2026-11-12 it always decodes (as stored, fine relief came out at 60% of its depth:
    # docs/BENCHMARK_DILIGENT_PI.md).
    check("new projects are automatic", ProcessingSettings().photo_tone == "auto")
    check("…which decodes with the lighting uneven",
          ProcessingSettings().resolved_photo_tone() == "srgb")
    check("…decoded with flat areas marked",
          ProcessingSettings(even_flat=True, flat_areas="0,0,9,9").resolved_photo_tone() == "srgb")
    check("…decoded when the light field named is missing",
          ProcessingSettings(light_field=os.path.join(tmp, "none.npz")).resolved_photo_tone() == "srgb")
    bf = os.path.join(tmp, "field.npz")
    open(bf, "wb").close()
    check("…decoded with a white-board light field",
          ProcessingSettings(light_field=bf).resolved_photo_tone() == "srgb")
    check("an explicit choice is kept",
          all(ProcessingSettings(photo_tone=t, light_field=bf).resolved_photo_tone() == t
              for t in ("stored", "srgb", "linear")))
    check("a session reads 'auto' back", (Sx.write(p, Sx.Session(object_id="a", photo_tone="auto")),
                                          Sx.read(p).processing_settings(ProcessingSettings())
                                          .photo_tone)[1] == "auto")

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
