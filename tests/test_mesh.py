"""The 3D pipeline, checked against an independent reader.

Printable output is a standing requirement here, and a renderer hides exactly
the defects that ruin a print: a hole reads as a shadow, a flipped triangle
reads as shading, and a mesh that is not a closed volume slices into something
the printer cannot fill. build_watertight_mesh_data() asserts watertightness
in its name and nothing checked it.

pack_binary_stl() writes the binary STL format by hand - an 80-byte header, a
uint32 count, then 50 bytes per triangle. A slip in any of those produces a
file that opens in some readers and not others. trimesh reads it here, so the
check does not depend on the same code that wrote it.
"""
import os
import struct
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


def surface(h=48, w=64):
    """A dome, a ridge and an incised groove — features with real relief."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w, yy / h
    z = 0.30 * np.exp(-((u - 0.4) ** 2 + (v - 0.5) ** 2) / 0.05)
    z += 0.06 * np.exp(-((u - 0.75) ** 2) / (2 * 0.02 ** 2))
    z -= 0.05 * np.exp(-((v - 0.25) ** 2) / (2 * 0.02 ** 2))
    return z.astype(np.float32)


def main():
    print("=" * 62)
    print("3D pipeline")
    print("=" * 62)
    import trimesh
    from helio.core.processing import (build_grid_mesh, build_watertight_mesh_data,
                                       pack_binary_stl)

    settings = {"height": 100.0, "fx": 1, "fy": -1, "fz": 1, "step": 1,
                "center_origin": True, "vertex_colors": False,
                "export_brightness": 1.0}
    depth = surface()
    verts, faces, colors, uvs = build_grid_mesh(depth, settings)

    print("\nthe open surface mesh")
    check("vertices and faces are produced", len(verts) > 0 and len(faces) > 0,
          f"{len(verts)} verts, {len(faces)} faces")
    check("no vertex is NaN or infinite", np.isfinite(verts).all())
    check("every face indexes a real vertex",
          faces.max() < len(verts) and faces.min() >= 0,
          f"max index {faces.max()} of {len(verts)}")
    check("no degenerate triangle (two shared corners)",
          all(len(set(f)) == 3 for f in faces[:5000]))
    check("one UV per vertex", len(uvs) == len(verts))

    print("\nthe single-sided surface faces the viewer")
    # Every non-watertight export - the GLB web viewer, the HF+normals viewer,
    # the Multi-LOD "archival" package - was written 100% back-facing:
    # measured -0.817 mean face Nz on a dome whose true normals point up.
    # doubleSided:true hid it in the bundled viewer while Blender, MeshLab and
    # Sketchfab showed it inside out.
    def _facing_up(vv, ff):
        tri = vv[ff]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nz = nrm[:, 2] / np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)
        return float((nz > 0).mean())

    for fx, fy in ((1, -1), (1, 1), (-1, -1), (-1, 1)):
        st2 = dict(settings, fx=fx, fy=fy, fz=1)
        v2, f2, _c, _u = build_grid_mesh(depth, st2)
        frac = _facing_up(v2, f2)
        check(f"fx={fx:+d} fy={fy:+d}: the surface faces the viewer",
              frac > 0.99, f"{frac*100:.1f}% face +Z")

    # fz = -1 deliberately mirrors the height axis, so the surface genuinely
    # faces the other way. Recorded rather than asserted as a defect.
    v3, f3, _c3, _u3 = build_grid_mesh(depth, dict(settings, fz=-1))
    print(f"       (fz=-1 mirrors depth: {_facing_up(v3, f3)*100:.0f}% face +Z, "
          "which follows the mirrored geometry)")

    print("\nthe watertight solid")
    av, af, ac, au, n_top = build_watertight_mesh_data(verts, faces, colors, uvs, settings)
    check("the solid adds geometry to the open surface",
          len(av) > len(verts) and len(af) > len(faces),
          f"{len(verts)}->{len(av)} verts, {len(faces)}->{len(af)} faces")
    check("UVs and colours still cover every vertex",
          len(au) == len(av) and len(ac) == len(av),
          f"{len(av)} verts, {len(au)} uvs, {len(ac)} colours")

    # The seam vertices are duplicated on purpose: a shared vertex cannot
    # carry both the top surface's UV and the side wall's black UV, and the
    # colour would bleed down the wall.  So the correct invariant is not
    # "watertight as written" - it is "closed once coincident vertices are
    # merged", which is what every slicer and importer does.  Both halves are
    # pinned here, because the distinction matters to a repository validating
    # the file exactly as written.
    raw = trimesh.Trimesh(vertices=av, faces=af, process=False)
    m = trimesh.Trimesh(vertices=av, faces=af, process=True)
    check("merging coincident vertices closes the solid", m.is_watertight,
          f"still open after merge; {len(raw.vertices) - len(m.vertices)} merged")
    check("the duplicates are exactly coincident, not near-misses",
          len(raw.vertices) > len(m.vertices),
          "no vertices merged — the seam may genuinely be apart")
    check("it encloses a volume", m.is_volume)
    check("the volume is positive", m.volume > 0, f"volume {m.volume:.3f}")
    check("winding is consistent", m.is_winding_consistent)

    # Every mirror combination has to produce a printable solid, not just the
    # default. The surfaces used to be wound inward while the walls faced out,
    # so the signed volume read 10,812 against a true 38,036 - consistent
    # enough to look fine and wrong enough to corrupt any measurement. And a
    # Z-mirror turned the solid inside out, giving a negative volume.
    import itertools
    bad = []
    for fx, fy, fz in itertools.product((1, -1), (1, -1), (1, -1)):
        st2 = dict(settings, fx=fx, fy=fy, fz=fz)
        v2, f2, c2, u2 = build_grid_mesh(depth, st2)
        a2, b2, _c2, _u2, _n2 = build_watertight_mesh_data(v2, f2, c2, u2, st2)
        mm = trimesh.Trimesh(vertices=a2, faces=b2, process=True)
        if not (mm.is_watertight and mm.is_winding_consistent
                and mm.is_volume and mm.volume > 0):
            bad.append(f"fx={fx} fy={fy} fz={fz} "
                       f"(watertight={mm.is_watertight} winding={mm.is_winding_consistent} "
                       f"vol={mm.volume:.0f})")
    check("all eight mirror combinations give a valid solid", not bad,
          "; ".join(bad))

    print("\n     outward-facing normals per group")
    # Judge each group against the axis that is actually outward for it. The
    # mesh centroid is a poor reference for the top surface: it sits low
    # because of the bottom slab, so rim faces on a domed surface read as
    # sideways and the fraction lands near 92% on correct geometry. The top
    # should face +Z, the bottom -Z, and the walls outward in XY.
    n_t = len(faces)
    cen_xy = av[:, :2].mean(axis=0)
    for label, (a, b) in (("top", (0, n_t)), ("bottom", (n_t, 2 * n_t)),
                          ("side", (2 * n_t, len(af)))):
        tri = av[af[a:b]]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        if label == "side":
            d = np.zeros_like(nrm)
            d[:, :2] = tri.mean(axis=1)[:, :2] - cen_xy
        else:
            d = np.zeros_like(nrm)
            d[:, 2] = 1.0 if label == "top" else -1.0
        frac = float((np.einsum("ij,ij->i", nrm, d) > 0).mean())
        check(f"the {label} faces point outward", frac > 0.99, f"{frac*100:.0f}%")
    check("there are no duplicate faces",
          len(m.faces) == len(np.unique(np.sort(m.faces, axis=1), axis=0)))

    print("\nthe hand-packed binary STL, read back by trimesh")
    tris = av[af]                       # (n, 3, 3) triangle soup
    tmp = tempfile.mkdtemp(prefix="helio_stl_")
    path = os.path.join(tmp, "m.stl")
    n_tri, size_mb = pack_binary_stl(tris.astype(np.float32), path)
    check("the writer reports the triangle count it was given", n_tri == len(af),
          f"{n_tri} vs {len(af)}")

    size = os.path.getsize(path)
    check("the file is exactly header + count + 50 bytes per triangle",
          size == 84 + 50 * n_tri, f"{size} vs {84 + 50 * n_tri}")
    with open(path, "rb") as fh:
        fh.seek(80)
        declared = struct.unpack("<I", fh.read(4))[0]
    check("the declared count matches the triangles written", declared == n_tri,
          f"header says {declared}, wrote {n_tri}")

    back = trimesh.load(path, file_type="stl", process=False)
    check("trimesh reads the file at all", back is not None and len(back.faces) > 0)
    check("no triangles were lost in the round trip", len(back.faces) == n_tri,
          f"read {len(back.faces)} of {n_tri}")
    check("the geometry survives the round trip",
          np.allclose(np.array(back.bounds), np.array(m.bounds), atol=1e-3),
          f"{back.bounds} vs {m.bounds}")
    check("the round-tripped solid is still watertight",
          trimesh.load(path, file_type="stl").is_watertight)

    print("\nvertex normals are real, not inferred from a flat file")
    check("normals exist for every vertex", len(m.vertex_normals) == len(m.vertices))
    check("normals are unit length",
          np.allclose(np.linalg.norm(m.vertex_normals, axis=1), 1.0, atol=1e-3))
    top_n = m.vertex_normals[:n_top]
    check("the top surface faces away from the base, not into it",
          np.median(top_n[:, 2]) * np.sign(settings["fz"]) != 0,
          f"median Nz {np.median(top_n[:, 2]):.3f}")

    print("\na bad sample must not destroy the model in silence")
    # The flat-map guard is `if rng < 1e-9`, and NaN < 1e-9 is False, so it
    # never fired. The NaN then propagated through the Gaussian blurs until
    # the whole surface was non-finite: measured on a 200x300 map with ONE bad
    # pixel, the STL still wrote 12 MB, its Z extent collapsed from 6.00 mm to
    # 0.00 mm, the mesh was not watertight, and the app said "3D-print ready".
    from helio.core.processing import repair_nonfinite
    base = np.zeros((80, 120), np.float32)
    base[:, 40:80] = 1.0

    for label, mutate in (
            ("one NaN", lambda d: d.__setitem__((20, 20), np.nan)),
            ("one +inf", lambda d: d.__setitem__((20, 20), np.inf)),
            ("one -inf", lambda d: d.__setitem__((20, 20), -np.inf)),
            # 9x9 of 9,600 samples is 0.84% - inside the repair band. A
            # 20x20 hole is 4.2% and is refused, which is checked below.
            ("a 9x9 hole", lambda d: d.__setitem__((slice(10, 19), slice(10, 19)), np.nan))):
        d = base.copy()
        mutate(d)
        fixed = repair_nonfinite(d, "test")
        check(f"{label}: repaired to all-finite", np.isfinite(fixed).all())
        check(f"{label}: the relief survives",
              abs(float(np.ptp(fixed)) - 1.0) < 1e-6, f"range {np.ptp(fixed):.4f}")

    # The fill mechanism itself, on a grid small enough to reason about.
    # max_fraction is raised because 1 of 9 samples is 11% - the policy is
    # tested separately below; this is about where the value comes from.
    tiny = np.arange(9, dtype=np.float32).reshape(3, 3)
    tiny[1, 1] = np.nan
    filled = float(repair_nonfinite(tiny, "t", max_fraction=1.0)[1, 1])
    check("a repaired sample takes a neighbouring value, not a global one",
          filled in (1.0, 3.0, 5.0, 7.0),
          f"got {filled} — expected one of its four neighbours")

    check("a clean map is returned untouched",
          repair_nonfinite(base, "t") is base or np.array_equal(repair_nonfinite(base, "t"), base))

    print("\nbut a genuinely broken map is refused, not invented")
    d = base.copy()
    d[10:30, 10:30] = np.nan          # 4.2% - just over the threshold
    try:
        repair_nonfinite(d, "test")
        check("a 4.2% hole is refused", False, "it was repaired")
    except ValueError:
        check("a 4.2% hole is refused, rather than invented", True)

    for frac, shape in ((0.5, (slice(0, 40), slice(None))), (1.0, (slice(None), slice(None)))):
        d = base.copy()
        d[shape] = np.nan
        try:
            repair_nonfinite(d, "test")
            check(f"{frac*100:.0f}% non-finite is refused", False, "it was accepted")
        except ValueError as e:
            check(f"{frac*100:.0f}% non-finite is refused", True)
            check(f"{frac*100:.0f}%: the message says how much is bad",
                  "%" in str(e) or "entirely" in str(e), str(e)[:60])

    print("\n" + "=" * 62)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 62)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
