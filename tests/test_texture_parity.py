"""Every mesh writer must agree on which way up a texture goes.

HELIO builds UVs with the origin at the top-left, matching image indexing.
OBJ, glTF and trimesh all put it at the bottom-left, so V has to be flipped on
the way out. Most writers do:

  * _uv_to_trimesh()          for the GLB / trimesh paths
  * _write_obj_sketchfab_zip  for the Sketchfab OBJ
  * the inline OBJ writer     in export_3d_mesh

export_obj_mesh() did not, so a texture applied to its output came out
mirrored against the geometry - the same class of defect as the trimesh UV
flip found previously. A flipped texture is easy to miss on a symmetrical
surface and obvious on an inscription, which is the worst way for it to fail.
"""
import os
import re
import sys
import tempfile

import numpy as np

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
    print("=" * 62)
    print("Texture orientation parity")
    print("=" * 62)
    from helio.io.exporters import _uv_to_trimesh, export_obj_mesh
    from helio.utils.file_utils import _write_obj_sketchfab_zip

    # UVs spanning the full range, asymmetric in V so a flip is detectable
    uvs = np.array([[0.0, 0.0], [1.0, 0.25], [0.5, 1.0], [0.25, 0.75]], np.float32)
    verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int32)

    print("\nthe shared helper flips V")
    out = _uv_to_trimesh(uvs)
    check("U is left alone", np.allclose(out[:, 0], uvs[:, 0]))
    check("V is inverted", np.allclose(out[:, 1], 1.0 - uvs[:, 1]),
          f"{out[:, 1]} vs {1.0 - uvs[:, 1]}")
    check("the input array is not modified in place",
          np.allclose(uvs[:, 1], [0.0, 0.25, 1.0, 0.75]),
          "a caller reusing its UVs would get them flipped twice")

    print("\nthe plain OBJ writer agrees with it")
    tmp = tempfile.mkdtemp(prefix="helio_uv_")
    path = os.path.join(tmp, "m.obj")
    export_obj_mesh(verts, faces, uvs=uvs, path=path)
    vt = [tuple(float(x) for x in l.split()[1:3])
          for l in open(path) if l.startswith("vt ")]
    check("one vt per UV", len(vt) == len(uvs), f"{len(vt)} vs {len(uvs)}")
    want = [(float(u), float(1.0 - v)) for u, v in uvs]
    check("the OBJ vt lines match the flipped convention",
          all(abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5
              for a, b in zip(vt, want)),
          f"wrote {vt[:2]}, expected {want[:2]}")

    print("\nthe Sketchfab OBJ agrees too")
    zpath = os.path.join(tmp, "m.zip")
    rng = np.random.RandomState(0)
    _write_obj_sketchfab_zip(
        zpath, "m", verts, faces, uvs,
        rng.rand(len(verts), 3).astype(np.float32),
        rng.randint(0, 255, (8, 8, 3), dtype=np.uint8),
        rng.randint(0, 255, (8, 8, 3), dtype=np.uint8))
    import zipfile
    with zipfile.ZipFile(zpath) as z:
        obj = z.read("m.obj").decode()
    vt2 = [tuple(float(x) for x in l.split()[1:3])
           for l in obj.splitlines() if l.startswith("vt ")]
    check("the Sketchfab OBJ writes the same vt values",
          all(abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5
              for a, b in zip(vt2, want)),
          f"{vt2[:2]} vs {want[:2]}")
    check("both OBJ writers produce identical UVs",
          all(abs(a[0] - b[0]) < 1e-5 and abs(a[1] - b[1]) < 1e-5
              for a, b in zip(vt, vt2)),
          "the two OBJ paths disagree with each other")

    print("\nno writer is left on the raw convention")
    src = open("helio/io/exporters.py", encoding="utf-8").read()
    raw = re.findall(r'vt \{uv\[0\][^}]*\}\s+\{uv\[1\]', src)
    check("no writer emits vt straight from an unflipped V", not raw, str(raw))

    print("\nround trip through an independent reader")
    import trimesh
    m = trimesh.load(path, file_type="obj", process=False)
    got = np.asarray(m.visual.uv) if getattr(m.visual, "uv", None) is not None else None
    check("trimesh reads UVs back from the OBJ", got is not None)
    if got is not None:
        check("what trimesh reads is what we wrote",
              np.allclose(np.sort(got[:, 1]), np.sort(np.array(want)[:, 1]), atol=1e-5),
              f"{got[:2]} vs {want[:2]}")

    print("\n" + "=" * 62)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 62)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
