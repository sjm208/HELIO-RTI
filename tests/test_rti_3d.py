#!/usr/bin/env python3
"""Relight and 3D viewer (export_web.rti_3d_viewer): two whole viewers in one
HTML file behind tabs.  Each page is carried as a script string, so nothing in
it (a '</script>', a '<!--') can end the wrapper's script early; the mesh step
keeps the 3D tab light; the format is in Publish and the Quick Export menu."""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASSED, FAILED = 0, []


def check(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name}  {detail}")


def main():
    from helio.application import export_web as EW

    print("The mesh step")
    check("a 2177 x 2902 surface steps to stay under 100,000 vertices",
          (2177 // EW.rti_3d_step(2177, 2902)) * (2902 // EW.rti_3d_step(2177, 2902))
          <= EW.RTI_3D_MAX_VERTS)
    check("…and a small one is left at full density", EW.rti_3d_step(200, 300) == 1)

    print("\nThe page")
    nasty = ("<!doctype html><html><head><script>var a='</script>';</script></head>"
             "<body><!-- <script> --><p>RTI & \"quotes\"  </p></body></html>")
    model = "<!doctype html><html><body><script>var glb='data:model/gltf-binary;base64,AAAA'</script></body></html>"
    out = os.path.join(tempfile.mkdtemp(), "x.html")
    EW.rti_3d_viewer(out, nasty, model, "Binding <A&B>")
    doc = open(out, encoding="utf-8").read()
    scripts = re.findall(r"<script>(.*?)</script>", doc, re.S)
    check("one script in the wrapper: the pages' own '</script>'s did not end it",
          len(scripts) == 1, f"{len(scripts)} scripts")
    body = scripts[0] if scripts else ""
    m = re.search(r'var PAGES=\{"rti":(".*?"),\n"3d":(".*?")\};', body, re.S)
    check("both pages are there as strings", m is not None)
    if m:
        check("…and decode back exactly as written",
              json.loads(m.group(1)) == nasty and json.loads(m.group(2)) == model)
        check("…with no '<' left in them raw", "<" not in m.group(1) + m.group(2))
    check("the title is escaped", "Binding &lt;A&amp;B&gt;" in doc and "<A&B>" not in doc)
    check("two tabs, Relight first and chosen",
          'id="t-rti" aria-selected="true"' in doc and 'id="t-3d" aria-selected="false"' in doc)
    check("the 3D frame waits until its tab is opened", 'id="f-3d" title="3D model" allow="fullscreen" hidden' in doc)

    print("\nWhere it is offered")
    from helio_qt.publish import WEB_EXPORTS
    from helio_qt import publish_catalog as C
    keys = {f.key for fs in C.GROUPS.values() for f in fs} if hasattr(C, "GROUPS") else set()
    src = open(os.path.join(ROOT, "helio_qt", "publish_catalog.py"), encoding="utf-8").read()
    check("Publish lists Relight and 3D viewer", '"rti_3d", "Relight and 3D viewer"' in src or "rti_3d" in keys)
    check("…and can write it", "rti_3d" in WEB_EXPORTS)
    acts = open(os.path.join(ROOT, "helio_qt", "actions.py"), encoding="utf-8").read()
    check("File ▸ Quick Export has Relight and 3D Viewer…",
          '"export.rti_3d", "Relight and 3D Viewer…"' in acts)


if __name__ == "__main__":
    main()
    print(f"\nResults: {PASSED} passed, {len(FAILED)} failed out of {PASSED + len(FAILED)} tests")
    sys.exit(1 if FAILED else 0)
