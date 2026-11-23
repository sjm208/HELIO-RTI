"""Publish's model and sheet options (exports.MESH_TARGETS, export_3d.texture_grid, pub_sheet.to_paper).

  * the texture size is its own choice: "mesh" is as coarse as the mesh,
    "2048" caps the longer side, "full" is every pixel the mesh covers;
  * the GLB carries a texture of that size; the OBJ writes a JPEG colour map
    when asked, its normal map stays PNG;
  * the Sketchfab textures keep the region's shape (they were padded to a
    square, which put the colour off the model);
  * a fresh project's settings match a "Made for" choice, and each choice
    is told apart; the size estimate grows with the texture.

    python tests/test_model_options.py
"""
import os
import sys
import tempfile
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from helio.application import exports as X                 # noqa: E402
from helio.application.export_3d import mesh_export, texture_grid   # noqa: E402

_results = []


def check(name, fn):
    try:
        fn()
        _results.append((name, None))
        print(f"  ok    {name}")
    except Exception as e:                                # noqa: BLE001
        _results.append((name, e))
        print(f"  FAIL  {name}: {e}")
        if not isinstance(e, AssertionError):
            traceback.print_exc()


_D = []


def data():
    """A small solve stored as the exports read it (240 x 160)."""
    if _D:
        return _D[0]
    from helio.application.pipeline import SolveRequest, solve, store
    from helio.data_manager import DataManager
    L = []
    for k in range(8):
        az, el = k * np.pi / 4, (0.6, 1.0)[k % 2]
        L.append([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    L = np.array(L, np.float32)
    yy, xx = np.mgrid[0:160, 0:240]
    z = 6 * np.sin(xx / 13.0) * np.cos(yy / 17.0)
    gy, gx = np.gradient(z)
    N = np.dstack([-gx, gy, np.ones_like(z)])
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    frames = []
    for l in L:
        g = np.clip(160 * np.maximum(N @ l, 0) + 20 * (xx % 40 < 20), 0, 255).astype(np.uint8)
        frames.append((np.dstack([g, g // 2, 255 - g]), g, False))
    r = solve(SolveRequest(paths=[], lights=L, frames=frames))
    D = DataManager()
    store(r, D)
    _D.append((D, r))
    return _D[0]


def t_texture_grid():
    img = np.arange(100 * 61 * 3).reshape(100, 61, 3).astype(np.uint8)
    assert texture_grid(img, 100, 61, 4, "mesh").shape == (25, 16, 3)
    full = texture_grid(img, 100, 61, 4, "full")
    assert full.shape == (97, 61, 3), full.shape         # the rows and columns the mesh spans
    assert texture_grid(img, 100, 61, 4, "48").shape[0] == 48
    assert texture_grid(img, 100, 61, 4, "4096").shape == (97, 61, 3)   # never enlarged


def export(fmt, ext, **options):
    D, r = data()
    sh = X.shared_defaults()
    sh["export_step"], sh["edge_trim"] = 4, 0.0
    out = tempfile.mkdtemp()
    res = mesh_export(D, os.path.join(out, "t" + ext), fmt, ss=sh, options=options,
                      depth_normalised=r.depth_normalised_full)
    return res, out


def glb_texture_size(path):
    import trimesh
    scene = trimesh.load(path)
    geom = list(scene.geometry.values())[0] if hasattr(scene, "geometry") else scene
    return geom.visual.material.baseColorTexture.size


def t_glb_texture_sizes():
    mesh, _ = export("glb", ".glb", tex_size="mesh")
    full, _ = export("glb", ".glb", tex_size="full")
    small, _ = export("glb", ".glb", tex_size="64")
    a, b, c = (glb_texture_size(x["path"]) for x in (mesh, full, small))
    assert a == (60, 40), a                                      # one texel a point (240/4, 160/4)
    assert b == (237, 157), b                                    # every pixel the mesh spans
    assert max(c) == 64 and abs(c[0] / c[1] - 237 / 157) < 0.05, c


def t_obj_jpeg():
    res, out = export("obj", ".obj", tex_size="full", sketchfab_jpeg=True)
    files = sorted(os.listdir(out))
    assert "t_albedo.jpg" in files and "t_normal.png" in files, files
    mtl = open(os.path.join(out, "t.mtl")).read()
    assert "map_Kd t_albedo.jpg" in mtl, mtl


def t_sketchfab_keeps_shape():
    import cv2
    res, _ = export("glb", ".glb", sketchfab=True, tex_size="128")
    d = res["export_dir"]
    base = [f for f in os.listdir(d) if "baseColor" in f][0]
    h, w = cv2.imread(os.path.join(d, base)).shape[:2]
    assert max(h, w) == 128 and abs(w / h - 1.5) < 0.05, (w, h)


def t_targets():
    sh = X.shared_defaults()
    assert X.mesh_target(sh) == "general", X.mesh_target(sh)
    seen = set()
    for key, (_n, vals) in X.MESH_TARGETS.items():
        assert X.mesh_target(dict(sh, **vals)) == key, key
        seen.add(tuple(sorted((k, str(v)) for k, v in vals.items())))
    assert len(seen) == len(X.MESH_TARGETS), "two choices set the same values"
    assert X.mesh_target(dict(sh, tex_size="2048", export_step=1)) == ""


def t_estimate():
    a = X.texture_bytes(4000, 3000, 2.0, "mesh", False)
    b = X.texture_bytes(4000, 3000, 2.0, "2048", False)
    c = X.texture_bytes(4000, 3000, 2.0, "full", False)
    d = X.texture_bytes(4000, 3000, 2.0, "full", True)
    assert a == 0 and 0 < b < c and d < c, (a, b, c, d)


def t_publish_runner_reads_the_choices():
    from helio_qt import publish as P
    D, r = data()
    sh = X.shared_defaults()
    sh.update(export_step=4, edge_trim=0.0, tex_size="full", tex_jpeg=1, mesh_watertight=0)
    src = P.Source(result=r, lights=None, light_source="", images=[], name="opt", shared=sh, data=D)
    out = tempfile.mkdtemp()
    files = P.MESH_EXPORTS["obj"](src, out)
    names = [os.path.basename(f) for f in files]
    assert any(n.endswith("_albedo.jpg") for n in names), names
    import trimesh
    m = trimesh.load([f for f in files if f.endswith(".obj")][0], process=False)
    assert not m.is_watertight, "a closed solid although the base was turned off"


def t_sheet_paper():
    from PIL import Image
    from helio.application import pub_sheet as PS
    page = Image.new("RGB", (PS.PAGE_W, PS.PAGE_H), (10, 20, 30))
    assert PS.to_paper(page, "A3").size == (3508, 4961)
    assert PS.to_paper(page, "A4").size == (2480, 3508)
    letter = PS.to_paper(page, "Letter")
    assert letter.size == (2550, 3300), letter.size
    assert letter.getpixel((5, 1650)) == PS.PAPER and letter.getpixel((1275, 1650)) == (10, 20, 30)
    assert PS.to_paper(page, "A3", 150).size == (1754, 2480)


def main():
    for name, fn in [
        ("texture_grid: mesh, full, capped, never enlarged", t_texture_grid),
        ("GLB texture follows the chosen size", t_glb_texture_sizes),
        ("OBJ: JPEG colour, PNG normal map", t_obj_jpeg),
        ("Sketchfab textures keep their shape", t_sketchfab_keeps_shape),
        ("Made for: defaults match, every choice distinct", t_targets),
        ("the size estimate grows with the texture", t_estimate),
        ("Publish's mesh runner reads the choices", t_publish_runner_reads_the_choices),
        ("publication sheet: A3, A4, US Letter, screen", t_sheet_paper),
    ]:
        check(name, fn)
    bad = [n for n, e in _results if e is not None]
    print(f"\n{len(_results) - len(bad)}/{len(_results)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
