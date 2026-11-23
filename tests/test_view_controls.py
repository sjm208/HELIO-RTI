"""Every view's controls move its picture; nothing listed is dead (VIEW-002).

helio.domain.view_controls says, for every view, which settings change it
and how it takes the light.  This renders each view, through the shared
renderer both apps use, on a solved stack with a specular sheen (so the
specular views have something to show), and checks:

* each listed control moves the picture (mean change over a quarter of a
  grey level at one end of its range or the other);
* each setting the classic panel shows but the table leaves out (INERT)
  does not;
* a view the table says is lit by the dome follows lx, ly, lz; a compass
  view follows hs_azi and hs_alt; any other view ignores both;
* the tone controls move exactly the views that take them.

    python tests/test_view_controls.py
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PASS = FAIL = 0
MOVES = 0.25


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def solved():
    """A DataManager solved from a 12-light stack of the analytic surface with a sheen."""
    import cv2
    import numpy as np
    from tools.make_golden import analytic_surface
    from helio.application.pipeline import SolveRequest, solve, store
    from helio.data_manager import DataManager
    from helio.domain.registry import default_engine_settings
    from helio.domain.settings import ProcessingSettings
    _z, n, albedo = analytic_surface()
    alb = albedo.astype(np.float32) / 255.0
    tmp = tempfile.mkdtemp(prefix="helio_ctl_")
    L, paths = [], []
    for i in range(12):
        az, el = 2 * np.pi * i / 12, np.radians(30 + 15 * (i % 3))
        lv = np.array([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)], np.float32)
        L.append(lv)
        hv = lv + np.array([0, 0, 1], np.float32)
        hv /= np.linalg.norm(hv)
        spec = np.clip(n @ hv, 0, None) ** 40 * 0.6
        lit = np.clip(n @ lv, 0, None)[..., None] * (0.35 + 0.65 * alb) + spec[..., None]
        img = (np.clip(lit, 0, 1) * 245 + 5).astype(np.uint8)
        p = os.path.join(tmp, f"c{i:02d}.png")
        cv2.imwrite(p, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        paths.append(p)
    r = solve(SolveRequest(paths=paths, lights=np.array(L, np.float32),
                           settings=ProcessingSettings(photo_tone="linear")))
    D = DataManager()
    store(r, D)
    D.plate_paths = paths
    D.engine_settings = default_engine_settings()
    return D


def main():
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    import numpy as np
    from helio.application.render import composite, render
    from helio.domain.registry import RENDER_ENGINES, default_engine_settings
    from helio.domain.view_controls import (CONTROLS, INERT, TONE, VIEW_SETTINGS, controls_for,
                                            from_angles, light_kind, light_settings, takes_tone)
    print("=" * 60)
    print("View controls")
    print("=" * 60)
    D = solved()
    ES = default_engine_settings()

    def shot(eng, st):
        img = render(eng, D, st, D.ph, D.pw, True)
        return composite(eng, img, D, st, D.ph, D.pw, True, view_mode="Shading").astype(np.float32)

    raking = light_settings(*from_angles(300, 25))
    overhead = light_settings(*from_angles(0, 89.9))

    def moved(eng, key, base, values):
        b = shot(eng, base)
        best = 0.0
        for v in values:
            o = shot(eng, dict(base, **{key: v}))
            best = max(best, float(np.abs(o - b).mean()))
        return best

    print("\nEvery view is in the table")
    check("one entry per view", set(VIEW_SETTINGS) == set(RENDER_ENGINES),
          set(RENDER_ENGINES) ^ set(VIEW_SETTINGS))

    print("\nThe light")
    for eng in RENDER_ENGINES:
        kind = light_kind(eng)
        base = dict(ES[eng], **raking)
        # Opposite azimuths at one height: a view lit "from every side" (8-Dir)
        # takes only the height, which is one of its own settings.
        a, b = light_settings(*from_angles(300, 35)), light_settings(*from_angles(120, 35))
        lxyz = float(np.abs(shot(eng, dict(base, **{k: a[k] for k in ("lx", "ly", "lz")}))
                            - shot(eng, dict(base, **{k: b[k] for k in ("lx", "ly", "lz")}))).mean())
        hs = float(np.abs(shot(eng, dict(base, hs_azi=a["hs_azi"], hs_alt=a["hs_alt"]))
                          - shot(eng, dict(base, hs_azi=b["hs_azi"], hs_alt=b["hs_alt"]))).mean())
        if kind == "dome":
            ok = lxyz > MOVES
        elif kind == "compass":
            ok = hs > MOVES and lxyz <= MOVES
        else:
            ok = lxyz <= MOVES and hs <= MOVES
        check(f"{eng}: {kind or 'no light'}", ok, f"lx/ly/lz {lxyz:.2f}, compass {hs:.2f}")

    print("\nEach listed control moves its view")
    for eng in RENDER_ENGINES:
        for c in controls_for(eng, dict(ES[eng], nr_mode=1)):
            ends = [v for v in (c.lo, c.hi) if v != c.default] or [c.lo]
            base = dict(ES[eng])
            if eng == "Normal Residual" and c.key != "nr_mode":
                base["nr_mode"] = 1          # its gain acts on the colour mode
            best = max(moved(eng, c.key, dict(base, **light), ends) for light in (raking, overhead))
            check(f"{eng} › {c.label}", best > MOVES, f"{best:.3f}")

    print("\nThe settings the classic panel showed in vain do nothing")
    for eng, keys in INERT.items():
        for key, why in keys.items():
            vals = {"hs_depth_src": ["frankot", "hf_depth"] if "8-Dir" not in eng else ["frankot"],
                    "hs_z": [0.1, 10.0], "nr_gain": [0.5, 5.0], "si_sigma": [0.5, 8.0],
                    "op_elev": [1, 45]}[key]
            best = moved(eng, key, dict(ES[eng], **raking), vals)
            check(f"{eng} › {key} ({why})", best <= MOVES, f"{best:.3f}")

    print("\nTone")
    for eng in RENDER_ENGINES:
        base = dict(ES[eng], **raking)
        best = max(moved(eng, k, base, [CONTROLS[k].lo, CONTROLS[k].hi]) for k in TONE)
        if takes_tone(eng, base):
            check(f"{eng} takes the tone controls", best > MOVES, f"{best:.3f}")
        else:
            check(f"{eng} is colour already: tone does not apply", best <= MOVES, f"{best:.3f}")

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
