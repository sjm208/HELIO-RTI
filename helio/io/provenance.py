"""Provenance records for HELIO exports.

A render is evidence. Years later someone has to be able to say which images
produced it, which light vectors were used, which solver ran and what version
of the software did the arithmetic. This module builds that record once, so
every export can carry the same one rather than each inventing its own subset.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time

_log = logging.getLogger(__name__)

SCHEMA = "helio-provenance/1"

#: Files larger than this are recorded by size and mtime rather than hashed,
#: so a folder of 80 MB TIFFs does not stall an export for minutes.
HASH_SIZE_LIMIT = 256 * 1024 * 1024


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "cv2", "scipy", "trimesh", "rasterio", "customtkinter"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


def _file_record(path: str) -> dict:
    rec: dict = {"name": os.path.basename(path)}
    try:
        st = os.stat(path)
        rec["bytes"] = st.st_size
        rec["modified_utc"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime))
        if st.st_size <= HASH_SIZE_LIMIT:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rec["sha256"] = h.hexdigest()
        else:
            rec["sha256"] = None
            rec["sha256_skipped"] = "file exceeds HASH_SIZE_LIMIT"
    except OSError as e:
        rec["error"] = str(e)
    return rec


def _get(obj, name, default=None):
    try:
        v = getattr(obj, name, default)
        return v.get() if hasattr(v, "get") and not isinstance(v, (list, dict)) else v
    except Exception:
        return default


def build_provenance(app, *, hash_sources: bool = True,
                     outputs: list[str] | None = None) -> dict:
    """Collect everything needed to reproduce the current result."""
    data = getattr(app, "data", None)
    paths = list(_get(data, "plate_paths", []) or [])

    lights = _get(data, "current_L_mat", None)
    if lights is not None:
        try:
            lights = [[round(float(c), 6) for c in v] for v in lights]
        except Exception:
            lights = None

    # How the lights were obtained matters as much as their values.
    if _get(data, "current_L_mat", None) is None:
        light_source = "uncalibrated (solver estimated directions)"
    elif _get(app, "loaded_lp_matrix", None) is not None:
        light_source = "imported .lp / calibrated light matrix"
    else:
        light_source = "in-session calibration"

    engine = _get(data, "render_engine", None)
    settings = {}
    try:
        settings = dict((_get(data, "engine_settings", {}) or {}).get(engine, {}))
        settings = {k: v for k, v in settings.items()
                    if isinstance(v, (int, float, str, bool))}
    except Exception:
        pass

    rec = {
        "schema": SCHEMA,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "software": {
            "name": "Helio",
            "version": _get(app, "APP_VERSION", "2.6.0"),
            "build": "NoTorch fork",
            "environment": _versions(),
        },
        "object": {
            "id": _get(app, "object_id_var", "") or "",
        },
        "source": {
            "image_count": len(paths),
            "images": [_file_record(p) for p in paths] if hash_sources
                      else [{"name": os.path.basename(p)} for p in paths],
            "dimensions": {"height": _get(data, "h", None),
                           "width": _get(data, "w", None)},
        },
        "calibration": {
            "light_source": light_source,
            "light_count": 0 if lights is None else len(lights),
            "light_vectors": lights,
        },
        "processing": {
            "normal_engine": _get(data, "normal_engine", None),
            "depth_engine": _get(data, "depth_engine", None),
            "render_engine": engine,
            "engine_settings": settings,
            "crop": list(_get(app, "crop_coords", []) or []) or None,
        },
        "outputs": outputs or [],
        "note": (
            "Light vectors are unit vectors in Helio's calibrated frame: +Z toward "
            "the viewer, +Y toward the top of the image (north), so the azimuth "
            "(0 = north, clockwise) is atan2(x, y)."
        ),
    }
    return rec


def write_provenance(app, out_dir: str, filename: str = "helio_provenance.json",
                     **kw) -> str | None:
    """Write the record beside an export. Never raises into a caller."""
    try:
        rec = build_provenance(app, **kw)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        return path
    except Exception:
        _log.warning("could not write provenance record", exc_info=True)
        return None
