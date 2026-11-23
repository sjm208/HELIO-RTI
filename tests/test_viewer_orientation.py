"""Baked viewer layers must load the same way up as every other texture.

The combined RTI viewer uploads its four main maps through `mkTex`, which never
touches UNPACK_FLIP_Y_WEBGL and so uploads at the WebGL default of false. The
baked layer path set it to *true*, and nothing on the Python side compensated —
the layers are encoded top-down exactly like albedo and normals — so PCA 1-3
arrived vertically mirrored against the very maps they exist to be compared
with. Confirmed in a browser: a red band authored at the top of a baked layer
rendered at the bottom.

The check strips JS comments before looking at anything. The fix's own comment
says the path "used to set it true", and this file's history has twice recorded
a grep test matching its own explanatory text — so a naive search for
`UNPACK_FLIP_Y_WEBGL` near `true` would pass while the bug was present, and
fail once someone documented it.
"""
import os, re, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helio.io.exporters import export_combined_rti_viewer_html  # noqa: E402

PASS, FAIL = [], []
def check(n, f):
    try: f(); PASS.append(n); print(f"  PASS  {n}")
    except Exception as e: FAIL.append(n); print(f"  FAIL  {n}: {type(e).__name__}: {e}")


def _strip_js_comments(src):
    """Remove // line comments and /* */ blocks, keeping string literals safe
    enough for this purpose (the viewer has no // inside a string)."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(re.sub(r"//.*$", "", ln) for ln in src.splitlines())


H, W = 32, 40
normals = np.zeros((H, W, 3), np.float32)
normals[..., 2] = 1.0
albedo = np.full((H, W, 3), 128, np.uint8)
baked = np.zeros((H, W, 3), np.uint8)
baked[:8] = 255                       # a top band, and enough variance to ship
baked[16:24, 8:32] = 128

_out = os.path.join(tempfile.mkdtemp(), "v.html")
export_combined_rti_viewer_html(
    _out, normals, albedo, None, None, H, W, (0, 0, W, H), "T",
    extra_layers=[("PCA 1", baked)])
HTML = open(_out, encoding="utf-8").read()
CODE = _strip_js_comments(HTML)


def _no_flip_true():
    hits = re.findall(r"pixelStorei\s*\(\s*gl\.UNPACK_FLIP_Y_WEBGL\s*,\s*(\w+)\s*\)", CODE)
    bad = [h for h in hits if h != "false"]
    assert not bad, f"a texture is uploaded flipped: UNPACK_FLIP_Y_WEBGL={bad}"
check("no texture is uploaded with UNPACK_FLIP_Y_WEBGL true", _no_flip_true)


def _comment_stripper_works():
    # The guard above is only meaningful if the stripper really removes the
    # explanatory comment that contains the word it searches for.
    assert "UNPACK_FLIP_Y_WEBGL" in HTML
    assert "used to set it true" in HTML, "fix comment missing from output"
    assert "used to set it true" not in CODE, "comment stripper did not strip"
check("the comment stripper removes text the check would match", _comment_stripper_works)


def _baked_layer_shipped():
    assert "PCA 1" in HTML, "the baked layer never reached the viewer"
check("the baked layer is present to be tested at all", _baked_layer_shipped)


def _one_upload_convention():
    # mkTex is the reference path. If it ever starts setting the flag, the two
    # paths have diverged again and this test should be re-read rather than
    # trusted.
    mk = CODE.split("function mkTex")[1].split("function")[0]
    assert "UNPACK_FLIP_Y_WEBGL" not in mk, (
        "mkTex now sets the flip too — the two upload paths have diverged")
check("mkTex still defines the convention (sets no flip)", _one_upload_convention)


print("\n" + "=" * 56)
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS) + len(FAIL)} tests")
print("=" * 56)
sys.exit(1 if FAIL else 0)
