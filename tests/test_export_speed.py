"""The faster exporters give the same files.

* helio_core_rs.write_obj against the Python writers it replaces (byte for byte);
* mesh_export's OBJ: the same file with and without the Rust writer, and no longer written twice;
* the LOD OBJs: the same text trimesh writes;
* write_stl: valid binary STL, normals as the NumPy writer computes them;
* write_geotiff with threaded compression: identical pixels and georeferencing.

    python tests/test_export_speed.py
"""
import os
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def read(p):
    with open(p, "rb") as f:
        return f.read()


def main():
    import trimesh
    try:
        import helio_core_rs as rs
        rs.lls_normals
    except (ImportError, AttributeError):
        rs = None
    from helio.application import export_3d as E3
    tmp = tempfile.mkdtemp(prefix="helio_expspeed_")
    rng = np.random.default_rng(7)

    if rs is None:
        print("helio_core_rs not built: only the fallbacks are checked")
    else:
        print("write_obj against the Python writers")
        n, m = 30_000, 50_000
        verts = (rng.normal(size=(n, 3)) * 40).astype(np.float32)
        verts[:6] = [[0, -0.0, 1e-9], [1e10, -1e10, .5], [5e-5, 1.5e-4, 2.5e-4], [2.5, 3.5, -2.5], [1 / 3, 2 / 3, -1 / 3], [-4.99995, 0, 0]]
        vn = rng.normal(size=(n, 3)).astype(np.float32)
        uv = np.column_stack([rng.random(n), 1 - rng.random(n)]).astype(np.float32)
        faces = rng.integers(0, n, (m, 3))
        py = os.path.join(tmp, "py.obj")
        with open(py, "w") as f:
            f.write("# Helio — Photometric Stereo 3D Export\nmtllib x.mtl\nusemtl helio\n")
            f.writelines([f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in verts])
            f.writelines([f"vn {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n" for q in vn])
            f.writelines([f"vt {u[0]:.6f} {u[1]:.6f}\n" for u in uv])
            f.writelines([f"f {a[0]}/{a[0]}/{a[0]} {a[1]}/{a[1]}/{a[1]} {a[2]}/{a[2]}/{a[2]}\n" for a in faces.astype(np.int32) + 1])
        out = os.path.join(tmp, "rs.obj")
        rs.write_obj(out, E3._f64(verts), E3._u32_faces(faces), header="# Helio — Photometric Stereo 3D Export\nmtllib x.mtl\nusemtl helio\n",
                     decimals=6, normals=E3._f64(vn), uvs=E3._f64(uv), face_style="v/vt/vn")
        check("the PBR OBJ (v, vn, vt, v/vt/vn faces) is byte-identical, odd values included", read(py) == read(out))
        v4, f4 = rng.normal(size=(3000, 3)), rng.integers(0, 3000, (5000, 3))
        sv = os.path.join(tmp, "sv.obj")
        with open(sv, "w") as fh:
            fh.write("# hdr\n")
            np.savetxt(fh, v4, fmt="v %.4f %.4f %.4f")
            fh.write("\n")
            np.savetxt(fh, f4 + 1, fmt="f %d %d %d")
        rs.write_obj(os.path.join(tmp, "sv_rs.obj"), v4, E3._u32_faces(f4), header="# hdr\n", decimals=4, blank_after_vertices=True)
        check("the np.savetxt-style OBJ is byte-identical", read(sv) == read(os.path.join(tmp, "sv_rs.obj")))
        col = rng.integers(0, 255, (3000, 3)).astype(np.uint8)
        tm = trimesh.Trimesh(v4, f4, vertex_colors=np.column_stack([col, np.full(3000, 255, np.uint8)]), process=False)
        rs.write_obj(os.path.join(tmp, "tm_rs.obj"), v4, E3._u32_faces(f4), header="# https://github.com/mikedh/trimesh\n",
                     decimals=8, colors=col, color_decimals=8, footer="\n")
        check("the coloured trimesh OBJ is byte-identical", tm.export(file_type="obj").encode() == read(os.path.join(tmp, "tm_rs.obj")))

        big = 2_300_000                                   # more than one parallel wave of lines
        bv = np.arange(big * 3, dtype=np.float64).reshape(big, 3) * 0.001
        bf = np.stack([np.arange(big), (np.arange(big) + 1) % big, (np.arange(big) + 2) % big], axis=1)
        bp = os.path.join(tmp, "big.obj")
        rs.write_obj(bp, bv, E3._u32_faces(bf), decimals=3, face_style="v")
        with open(bp) as f:
            lines = f.read().split("\n")
        ok = len(lines) == 2 * big + 1 and lines[0] == "v 0.000 0.001 0.002" and lines[-2] == f"f {big} 1 2"
        for i in (32767, 32768, 2_097_151, 2_097_152, big - 1):                 # block and wave boundaries
            ok &= lines[i] == "v {:.3f} {:.3f} {:.3f}".format(*bv[i]) and lines[big + i] == f"f {bf[i, 0] + 1} {bf[i, 1] + 1} {bf[i, 2] + 1}"
        check(f"{big:,} vertices and faces across block and wave boundaries: every checked line right, in order", ok)

        def raises(fn):
            try:
                fn()
            except (ValueError, OSError):
                return True
            return False
        v3, f3 = np.zeros((3, 3)), np.array([[0, 1, 2]], np.uint32)
        check("a face past the last vertex is refused", raises(lambda: rs.write_obj(os.path.join(tmp, "x.obj"), v3, np.array([[0, 1, 3]], np.uint32))))
        check("face_style needing normals or UVs without them is refused", raises(lambda: rs.write_obj(os.path.join(tmp, "x.obj"), v3, f3, face_style="v/vt/vn")))
        check("an unknown face_style is refused", raises(lambda: rs.write_obj(os.path.join(tmp, "x.obj"), v3, f3, face_style="nope")))
        check("a wrong-sized normals array is refused", raises(lambda: rs.write_obj(os.path.join(tmp, "x.obj"), v3, f3, normals=np.zeros((2, 3)), face_style="v//vn")))
        check("an unwritable path raises", raises(lambda: rs.write_obj(os.path.join(tmp, "no", "such", "dir", "x.obj"), v3, f3)))

        print("write_stl")
        v = (rng.normal(size=(2000, 3)) * 10)
        f = rng.integers(0, 2000, (3000, 3))
        f[0] = [5, 5, 5]                                  # a degenerate triangle
        sp = os.path.join(tmp, "a.stl")
        rs.write_stl(sp, v, E3._u32_faces(f), header="HELIO test")
        E3_ref = os.path.join(tmp, "ref.stl")
        saved = E3._fast_io
        E3._fast_io = lambda: None
        E3._write_stl(E3_ref, v, f, "HELIO test")         # the NumPy writer
        E3._fast_io = saved
        a, b = read(sp), read(E3_ref)
        rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        ra, rb = np.frombuffer(a[84:], rec), np.frombuffer(b[84:], rec)
        check("header and triangle count match the NumPy writer", a[:84] == b[:84] and len(a) == len(b) == 84 + 50 * 3000)
        check("vertices are identical, normals within float32 rounding",
              np.array_equal(ra["v"], rb["v"]) and float(np.abs(ra["n"] - rb["n"]).max()) < 1e-5)
        check("the degenerate triangle gets (0, 0, 1)", tuple(ra["n"][0]) == (0.0, 0.0, 1.0))
        mesh = trimesh.load(sp, file_type="stl", process=False)
        check("trimesh reads it", len(mesh.faces) == 3000)

    print("mesh_export OBJ, end to end")
    import test_render_speed as T
    from helio.config import SHARED_STATE_DEFAULTS
    D = T.solved(300, 400)
    ss = {k: val[1] for k, val in SHARED_STATE_DEFAULTS.items()}
    ss["export_step"] = 1
    ss["height"] = 300.0
    outs = {}
    for label, patch in (("rust", None), ("python", lambda: None)):
        saved = E3._fast_io
        if patch:
            E3._fast_io = patch
        t = time.perf_counter()
        d = os.path.join(tmp, label)
        os.makedirs(d)
        E3.mesh_export(D, os.path.join(d, "m.obj"), "obj", ss=ss)
        outs[label] = (d, time.perf_counter() - t)
        E3._fast_io = saved
    names = sorted(os.listdir(outs["rust"][0]))
    check("the OBJ package has the same files", names == sorted(os.listdir(outs["python"][0])) and "m.obj" in names and "m.mtl" in names, names)
    check("m.obj is byte-identical with and without the Rust writer", read(os.path.join(outs["rust"][0], "m.obj")) == read(os.path.join(outs["python"][0], "m.obj")))
    text = read(os.path.join(outs["rust"][0], "m.obj")).decode()
    check("it is the UV OBJ (mtllib, vn, vt, v/vt/vn faces), not trimesh's plain one",
          text.startswith("# Helio") and "\nvt " in text and "\nvn " in text and "\nf 1/1/1 " in text)
    check("it reloads", len(trimesh.load(os.path.join(outs["rust"][0], "m.obj"), process=False).faces) > 1000)

    print("LOD package OBJs")
    lod = E3.lod_package(D, os.path.join(tmp, "lod"), "x", ss=ss)
    objs = sorted(f for f in os.listdir(os.path.join(tmp, "lod")) if f.endswith(".obj"))
    check("three OBJ levels are written", len(objs) == 3, objs)
    for o in objs[:1]:
        m = trimesh.load(os.path.join(tmp, "lod", o), process=False)
        text = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        check(f"{o} reloads with vertex colours", len(m.faces) > 0 and m.visual.kind == "vertex")
    if rs is not None:
        mesh = trimesh.Trimesh(rng.normal(size=(500, 3)), rng.integers(0, 500, (900, 3)),
                               vertex_colors=np.column_stack([rng.integers(0, 255, (500, 3)), np.full(500, 255)]).astype(np.uint8), process=False)
        p1 = os.path.join(tmp, "l.obj")
        E3._export_trimesh_obj(mesh, p1)
        check("_export_trimesh_obj writes exactly what trimesh writes", read(p1) == mesh.export(file_type="obj").encode())
        mesh.vertex_normals                                    # cached normals: handed back to trimesh
        p2 = os.path.join(tmp, "l2.obj")
        E3._export_trimesh_obj(mesh, p2)
        check("a mesh with normals still goes through trimesh (and keeps them)", b"\nvn " in read(p2))

    print("3D-print STL")
    pm = E3.print_mesh(D, os.path.join(tmp, "p.stl"))
    pmesh = trimesh.load(os.path.join(tmp, "p.stl"), file_type="stl", process=False)
    check("print_mesh writes a binary STL with every face", len(pmesh.faces) == pm["faces"] and os.path.getsize(os.path.join(tmp, "p.stl")) == 84 + 50 * pm["faces"])

    print("GeoTIFF with threaded compression")
    try:
        import rasterio
    except ImportError:
        rasterio = None
    if rasterio is None:
        print("  (rasterio is not installed: skipped)")
    else:
        from helio.io.exporters import write_geotiff
        z = np.random.default_rng(2).random((700, 900)).astype(np.float32)
        gp = os.path.join(tmp, "d.tif")
        write_geotiff(gp, z, tags={"HELIO_OBJECT_ID": "x"})
        with rasterio.open(gp) as src:
            check("pixels, CRS, nodata, tiling and tags survive",
                  np.array_equal(src.read(1), z) and src.crs.to_string() == "EPSG:32630" and src.profile["tiled"]
                  and src.profile["compress"].lower() == "lzw" and src.tags().get("HELIO_OBJECT_ID") == "x")
        write_geotiff(os.path.join(tmp, "c.tif"), rgb=(np.random.default_rng(3).random((300, 400, 3)) * 255).astype(np.uint8))
        with rasterio.open(os.path.join(tmp, "c.tif")) as src:
            check("an RGB GeoTIFF has three bands", src.count == 3)

    print("\n" + "=" * 56)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 56)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
