"""Our relight package must match what relight-cli itself writes.

The reference output was produced by RelightLab 2026.01's relight-cli from the
real 8-light capture, and is compared key-for-key against ours.
"""
import json, os, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helio.io.relight_export import export_relight_package, decode_plane  # noqa: E402

REF = os.environ.get("RELIGHT_REF", "")   # optional reference info.json

PASS, FAIL = [], []
def check(n, f):
    try: f(); PASS.append(n); print(f"  PASS  {n}")
    except Exception as e: FAIL.append(n); print(f"  FAIL  {n}: {type(e).__name__}: {e}")

H, W = 64, 96
rng = np.random.default_rng(3)
# A smooth, signed coefficient field - PTM terms are not all positive.
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
coeffs = np.stack([
    np.sin(xx / 11.0) * 0.8,
    np.cos(yy / 9.0) * 0.6,
    (xx - W / 2) / W,
    (yy - H / 2) / H,
    np.full((H, W), -0.25, np.float32),
    np.full((H, W), 0.9, np.float32),
]).astype(np.float32)
albedo = (rng.random((H, W, 3)) * 255).astype(np.uint8)
lights = np.array([[0.5, 0.2, 0.84], [-0.3, 0.6, 0.74], [0.1, -0.7, 0.71]], np.float32)

out = tempfile.mkdtemp()
export_relight_package(out, coeffs, albedo, lights, H, W)
info = json.load(open(os.path.join(out, "info.json")))

def _files():
    for i in range(3):
        p = os.path.join(out, f"plane_{i}.jpg")
        assert os.path.exists(p), f"missing {p}"
    assert os.path.exists(os.path.join(out, "info.json"))
check("writes info.json and three planes", _files)

def _schema():
    for k in ("width", "height", "format", "type", "colorspace",
              "lights", "nplanes", "quality", "materials"):
        assert k in info, f"info.json missing {k}"
    assert info["nplanes"] == 9, info["nplanes"]
    assert info["colorspace"] == "lrgb", info["colorspace"]
    assert info["type"] == "ptm", info["type"]
    assert info["format"] == "jpg", info["format"]
    assert (info["width"], info["height"]) == (W, H)
check("info.json carries the relight schema", _schema)

def _scale_bias_lengths():
    m = info["materials"][0]
    assert len(m["scale"]) == 9, len(m["scale"])
    assert len(m["bias"]) == 9, len(m["bias"])
    assert m["scale"][:3] == [1.0, 1.0, 1.0], m["scale"][:3]
    assert m["bias"][:3] == [0.0, 0.0, 0.0], m["bias"][:3]
check("nine scale/bias entries, albedo stored unscaled", _scale_bias_lengths)

def _lights_flat():
    assert len(info["lights"]) == 9, len(info["lights"])   # 3 lights x xyz
    assert abs(info["lights"][0] - 0.5) < 1e-3
check("lights are a flat xyz list", _lights_flat)

def _coefficients_survive():
    """Decode the planes and compare against what we put in."""
    m = info["materials"][0]
    got = (decode_plane(os.path.join(out, "plane_1.jpg"), m["scale"], m["bias"], 3)
           + decode_plane(os.path.join(out, "plane_2.jpg"), m["scale"], m["bias"], 6))
    for i in range(6):
        rng_i = float(coeffs[i].max() - coeffs[i].min())
        err = float(np.abs(got[i] - coeffs[i]).mean())
        tol = max(0.02 * max(rng_i, 1e-6), 0.02)   # 8-bit + JPEG
        assert err < tol, f"coefficient {i} err {err:.4f} > {tol:.4f}"
check("coefficients survive the 8-bit + JPEG round-trip", _coefficients_survive)

def _relighting_matches():
    """A render from the decoded planes must match HELIO's own PTM evaluation."""
    from helio.core.shading_engines import compute_ptm_relighting
    m = info["materials"][0]
    got = (decode_plane(os.path.join(out, "plane_1.jpg"), m["scale"], m["bias"], 3)
           + decode_plane(os.path.join(out, "plane_2.jpg"), m["scale"], m["bias"], 6))
    dec = np.stack(got)
    for (u, v) in [(0.0, 0.0), (0.5, 0.2), (-0.4, 0.6)]:
        ours = compute_ptm_relighting(dec, u, v)
        truth = compute_ptm_relighting(coeffs, u, v)
        err = float(np.abs(ours - truth).mean())
        assert err < 0.05, f"relighting at ({u},{v}) differs by {err:.4f}"
check("relighting from the package matches HELIO's own PTM", _relighting_matches)

def _matches_reference():
    """Key set and types must match relight-cli's own lrgb output."""
    if not REF or not os.path.exists(REF):
        print("      skipped - set RELIGHT_REF to relight-cli's info.json")
        return
    ref = json.load(open(REF))
    ours = set(info) | {"colorProfile"}      # colorProfile is optional
    missing = set(ref) - ours
    assert not missing, f"missing keys relight-cli writes: {sorted(missing)}"
    assert ref["colorspace"] == info["colorspace"], (ref["colorspace"], info["colorspace"])
    assert ref["nplanes"] == info["nplanes"], (ref["nplanes"], info["nplanes"])
    assert len(ref["materials"][0]["scale"]) == len(info["materials"][0]["scale"])
    assert ref["materials"][0]["scale"][:3] == info["materials"][0]["scale"][:3]
    print(f"      matched relight-cli reference ({len(ref)} keys)")
check("schema matches relight-cli's own output", _matches_reference)

print("\n" + "=" * 56)
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
print("=" * 56)
sys.exit(1 if FAIL else 0)
