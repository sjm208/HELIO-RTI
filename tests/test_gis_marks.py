"""Sections and annotations go into the GIS package, where they were drawn.

    python tests/test_gis_marks.py
"""
import json
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
    from helio.application import annotations as A, export_vectors as V
    d = tempfile.mkdtemp(prefix="helio_v_")
    anns = [A.make("rect", [(10, 20), (40, 60)], color="#00ff00", text="Stamp", layer="Damage"),
            A.make("ellipse", [(0, 0), (20, 10)]), A.make("polygon", [(0, 0), (5, 0), (5, 5)]),
            A.make("line", [(1, 1), (9, 9)]), A.make("dot", [(30, 40)]), A.make("measure", [(0, 0), (100, 0)]),
            A.make("line", [(3, 3)])]
    files = V.write(d, "Seal", sections=[dict(name="A–B", a=(100, 50), b=(200, 50), copies=3)], annotations=anns,
                    h=300, mm_per_px=0.05)
    check("a file of each", [os.path.basename(f) for f in files] == ["Seal_sections.geojson", "Seal_annotations.geojson"])
    sec = json.load(open(files[0], encoding="utf8"))["features"][0]
    check("a section in the raster's units, y up from its top row, with its length",
          sec["geometry"]["coordinates"] == [[100.0, 250.0], [200.0, 250.0]] and sec["properties"]["length"] == "5.0 mm")
    feats = json.load(open(files[1], encoding="utf8"))["features"]
    check("a one-point line is left out", len(feats) == 6)
    kinds = {f["properties"]["kind"]: f["geometry"]["type"] for f in feats}
    check("each kind its geometry", (kinds["rect"], kinds["dot"], kinds["line"]) == ("Polygon", "Point", "LineString")
          and kinds["ellipse"] == "Polygon")
    ring = feats[1]["geometry"]["coordinates"][0]
    check("an ellipse's ring closes", ring[0] == ring[-1] and len(ring) > 40)
    check("a measurement is labelled", feats[5]["properties"]["label"] == "5.0 mm")
    check("a reduced, cropped raster moves them with it", V.to_map(100, 0.5, (10, 20))(30, 40) == [10.0, 90.0])
    check("nothing marked, nothing written", V.write(tempfile.mkdtemp(), "X", h=10) == [])
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
