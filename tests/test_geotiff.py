"""GeoTIFF writer tests.

Every GeoTIFF HELIO produces goes through helio.io.exporters.write_geotiff.
These tests run the whole surface twice — once with rasterio available and
once with it forced off — because the two branches used to disagree about
compression, tags and whether a .tfw was written at all.

    python tests/test_geotiff.py
"""
import os
import sys
import shutil
import tempfile

import numpy as np

PASS = FAIL = 0


def check(label, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {label}")
        PASS += 1
    except Exception as e:
        import traceback
        print(f"  FAIL  {label}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
        FAIL += 1


def read_tfw(path):
    with open(os.path.splitext(path)[0] + ".tfw") as f:
        return [float(x) for x in f.read().split()]


def run_suite(with_rasterio):
    import helio.io.exporters as ex
    import helio.config

    label = "rasterio" if with_rasterio else "cv2 fallback"
    ex.HAS_RASTERIO = with_rasterio
    print(f"\n── {label} branch ──")

    tmp = tempfile.mkdtemp(prefix="helio_gtif_")
    H, W = 40, 60
    depth = np.linspace(0, 1, H * W).reshape(H, W).astype(np.float32)
    rgb = (np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)

    def _single():
        p = os.path.join(tmp, "single.tif")
        ex.write_geotiff(p, depth, tags={"HELIO_RES_MM": "0.0123"})
        assert os.path.exists(p), "no tif written"
        assert os.path.getsize(p) > 0
    check(f"[{label}] single-band float32", _single)

    def _tfw():
        p = os.path.join(tmp, "single.tif")
        v = read_tfw(p)
        assert len(v) == 6, v
        # HELIO convention: 1.0 unit/pixel, north-up, origin at image height
        assert v[0] == 1.0 and v[3] == -1.0, f"pixel size not 1.0/-1.0: {v}"
        assert v[1] == 0.0 and v[2] == 0.0, f"rotation terms not zero: {v}"
        assert v[5] == float(H), f"origin Y should be image height: {v[5]} != {H}"
    check(f"[{label}] .tfw sidecar written with the 1.0 convention", _tfw)

    def _rgb():
        p = os.path.join(tmp, "rgb.tif")
        ex.write_geotiff(p, rgb=rgb, tags={"HELIO_LAYER": "albedo"})
        assert os.path.exists(p)
        assert os.path.exists(os.path.splitext(p)[0] + ".tfw")
    check(f"[{label}] 3-band uint8 RGB", _rgb)

    def _roundtrip():
        import cv2
        p = os.path.join(tmp, "single.tif")
        arr = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        assert arr is not None, "written file is not readable"
        assert arr.shape[:2] == (H, W), arr.shape
        assert np.isfinite(arr).all()
        assert abs(float(arr.max()) - 1.0) < 1e-4, f"value range lost: {arr.max()}"
    check(f"[{label}] round-trips through cv2 with values intact", _roundtrip)

    def _wrapper_compat():
        p = os.path.join(tmp, "wrapped.tif")
        # The legacy wrapper used to raise without rasterio; it must not now.
        ex.write_geotiff_with_tags(p, depth, H, W, bands=1,
                                   tags={"HELIO_OBJECT_ID": "obj1"})
        assert os.path.exists(p)
        assert os.path.exists(os.path.splitext(p)[0] + ".tfw"), \
            "legacy wrapper lost the world file"
    check(f"[{label}] write_geotiff_with_tags wrapper", _wrapper_compat)

    def _save_geotiff():
        p = os.path.join(tmp, "legacy.tif")
        ex.save_geotiff(p, depth, H, W, res_mm=0.05)
        assert os.path.exists(p)
        assert os.path.exists(os.path.splitext(p)[0] + ".tfw"), \
            "save_geotiff lost the world file"
    check(f"[{label}] save_geotiff wrapper", _save_geotiff)

    def _dem_package():
        out = os.path.join(tmp, "dem")
        paths = ex.save_dem_package(out, "OBJ", depth, depth * 0.5,
                                    albedo=rgb, mm_per_px=0.037)
        for key in ("poisson", "fc", "albedo"):
            assert key in paths, f"{key} missing from package"
            assert os.path.exists(paths[key]), f"{key} not written"
            v = read_tfw(paths[key])
            assert v[0] == 1.0, (
                f"{key} world file uses {v[0]} units/pixel — the batch and "
                f"interactive DEM exports must agree on the 1.0 convention")
        # albedo/normals used to be skipped entirely without rasterio
        assert os.path.exists(paths["albedo"]), "albedo dropped in fallback branch"
    check(f"[{label}] save_dem_package writes a consistent set", _dem_package)

    def _dict():
        out = os.path.join(tmp, "maps")
        written = ex.save_geotiff_dict(out, {"a.tif": depth, "b.tif": depth},
                                       H, W, res_mm=1.0)
        assert len(written) == 2
        for p in written:
            assert os.path.exists(os.path.splitext(p)[0] + ".tfw")
    check(f"[{label}] save_geotiff_dict", _dict)

    if with_rasterio:
        def _tags_present():
            import rasterio
            p = os.path.join(tmp, "single.tif")
            with rasterio.open(p) as src:
                tags = src.tags()
                assert tags.get("HELIO_RES_MM") == "0.0123", tags
                assert src.compression is not None, "not compressed"
                assert src.crs is not None, "no CRS"
        check(f"[{label}] HELIO tags + LZW + CRS embedded", _tags_present)

    shutil.rmtree(tmp, ignore_errors=True)


def main():
    import helio.io.exporters as ex
    original = ex.HAS_RASTERIO
    try:
        import rasterio  # noqa: F401
        have = True
    except ImportError:
        have = False

    if have:
        run_suite(True)
    else:
        print("\n(rasterio not installed — skipping the rasterio branch)")
    run_suite(False)
    ex.HAS_RASTERIO = original

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
