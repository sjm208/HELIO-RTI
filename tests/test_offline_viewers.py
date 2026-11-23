"""Exported web viewers must not need a network to open.

Three exports used to load their libraries from a CDN: the 3D Web Viewer and
the HF+Normals viewer pulled three.js r128 from jsdelivr, DeepZoom pulled
OpenSeadragon from cdnjs. Both GLB viewers even said "The viewer is
self-contained" in their own error text while doing it — true of the model,
false of the page.

That matters because these exports feed the Heritage Archive and Zenodo
deposits, whose promise is that they still open in twenty years. A deposit that
goes blank when a CDN retires a 2021 build has failed at the one job it had,
and it fails silently and late.

This pins the property that actually matters — no external fetch in a viewer —
rather than the mechanism, so it keeps holding if the libraries are later
updated, re-vendored, or replaced.
"""
import os, re, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []
def check(n, f):
    try: f(); PASS.append(n); print(f"  PASS  {n}")
    except Exception as e: FAIL.append(n); print(f"  FAIL  {n}: {type(e).__name__}: {e}")

from helio.io.exporters import _build_web_viewer_html, _build_hf_normals_html  # noqa: E402
from helio import vendor_assets  # noqa: E402

# Any absolute http(s) reference that the browser would actually fetch.
# Deliberately not a bare "http" search: the viewers carry citation links and
# licence URLs in visible text, and those are fine — it is src=/href= to a
# stylesheet or script that breaks an offline page.
_FETCHING = re.compile(
    r"""(?:src|href)\s*=\s*["']\s*https?://""", re.I)


def _no_remote_fetch(name, html):
    hits = _FETCHING.findall(html)
    assert not hits, f"{name} fetches {len(hits)} remote resource(s)"


_web = _build_web_viewer_html("")
_hf = _build_hf_normals_html("", "")

check("3D Web Viewer fetches nothing remote",
      lambda: _no_remote_fetch("3D Web Viewer", _web))
check("HF+Normals viewer fetches nothing remote",
      lambda: _no_remote_fetch("HF+Normals viewer", _hf))


def _three_actually_inlined():
    # Present *and* real: a page could pass the fetch check by simply having
    # lost its script tag, which would be a blank viewer rather than a fix.
    for nm, html in (("3D Web Viewer", _web), ("HF+Normals", _hf)):
        assert "THREE" in html, f"{nm} has no THREE global"
        assert len(html) > 700_000, (
            f"{nm} is only {len(html)} chars — three.js does not look inlined")
check("three.js is genuinely inlined, not merely unreferenced",
      _three_actually_inlined)


def _deepzoom_refs_are_relative():
    # The DeepZoom page moved to helio.application.export_images (EXP-001),
    # which both apps call.
    src = open(os.path.join("helio", "application", "export_images.py"), encoding="utf-8").read()
    i = src.index('tileSources:"albedo.dzi"')
    frag = src[i - 400:i + 200]
    assert "cdnjs" not in frag, "DeepZoom still points at cdnjs"
    assert 'prefixUrl:"osd_images/"' in frag, "DeepZoom prefixUrl is not local"
check("DeepZoom references OpenSeadragon locally", _deepzoom_refs_are_relative)


def _vendor_assets_present():
    assert len(vendor_assets.three_js_bundle()) > 700_000
    assert len(vendor_assets.read_bytes("openseadragon.min.js")) > 100_000
    pngs = [f for f in os.listdir(vendor_assets.osd_images_dir())
            if f.endswith(".png")]
    assert len(pngs) >= 20, f"only {len(pngs)} OpenSeadragon images vendored"
check("the vendored libraries are present and plausible", _vendor_assets_present)


def _missing_assets_raise():
    # The failure mode worth designing against: help_images once shipped absent
    # and produced a help page of blank gaps with nothing erroring. A viewer
    # without three.js is a white rectangle, so it must raise instead.
    real = vendor_assets._HERE
    try:
        vendor_assets._HERE = tempfile.mkdtemp()
        try:
            vendor_assets.vendor_dir()
        except vendor_assets.VendorAssetMissing:
            return
        raise AssertionError("a missing vendor dir was tolerated silently")
    finally:
        vendor_assets._HERE = real
check("a missing vendored library raises rather than shipping a blank page",
      _missing_assets_raise)


def _packaged_for_the_bundle():
    # Helio 3 is bundled by helio_qt.spec (PyInstaller); Helio Classic's setup.py (py2app) is not in this tree.
    path = "helio_qt.spec" if os.path.exists("helio_qt.spec") else "setup.py"
    setup = open(path, encoding="utf-8").read()
    assert "vendor" in setup and "osd_images" in setup, (
        f"{path} does not ship the vendored libraries — the .app would export "
        "blank viewers while the source tree works fine")
check("the bundle spec ships the vendored libraries in the .app", _packaged_for_the_bundle)


def _attributed():
    notices = open("THIRD_PARTY_NOTICES.md", encoding="utf-8").read()
    assert "three.js" in notices, "three.js is bundled but not attributed"
    assert "OpenSeadragon" in notices, "OpenSeadragon is bundled but not attributed"
    assert "Loaded from a CDN, not bundled" not in notices, (
        "the notices still describe OpenSeadragon as not bundled")
check("both bundled libraries are attributed", _attributed)


print("\n" + "=" * 56)
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS) + len(FAIL)} tests")
print("=" * 56)
sys.exit(1 if FAIL else 0)
