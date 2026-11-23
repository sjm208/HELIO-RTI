"""Batch processing, GUI-free (the classic Overnight Batch).

A parent folder holds one subfolder per object.  Each is calibrated by the
classic priority chain, solved, and given the chosen exports in its own
output folder; a log records what happened to each.  Both apps call
scan() and calibration_for(); the new interface also runs the loop here.

Changed from HELIO 2: a light file in an object's folder that could not be
read used to leave the previous object's lights in place, so the object was
solved with another object's calibration and the log said ".lp file"; now
an unreadable file is logged and the chain moves on.  Only files that look
like light files count (.lp, .dome, or a .txt named for lights), not every
.txt (notes, checksums).
"""
from __future__ import annotations

import math
import os
import time
from typing import Callable

import numpy as np

LIGHT_EXTS = (".lp", ".dome")


def _captures(folder: str) -> list[str]:
    from helio.io.capture import captures_in
    return captures_in([folder])


def _light_files(folder: str) -> list[str]:
    out = []
    for f in sorted(os.listdir(folder)):
        lo = f.lower()
        if lo.endswith(LIGHT_EXTS) or (lo.endswith(".txt") and "light" in lo):
            out.append(os.path.join(folder, f))
    return out


def scan(parent: str) -> tuple[list[tuple[str, int]], str | None]:
    """The subfolders of *parent* with at least 3 captures, with their
    counts, and a shared light file found in *parent* (one named for
    lights first), as the classic Scan Directory."""
    folders = []
    for d in sorted(os.listdir(parent)):
        p = os.path.join(parent, d)
        if os.path.isdir(p):
            n = len(_captures(p))
            if n >= 3:
                folders.append((p, n))
    files = [f for f in sorted(os.listdir(parent)) if os.path.isfile(os.path.join(parent, f))]
    lp = [f for f in files if f.lower().endswith(LIGHT_EXTS + (".txt",)) and "light" in f.lower()]
    lp = lp or [f for f in files if f.lower().endswith(LIGHT_EXTS)]
    return folders, (os.path.join(parent, lp[0]) if lp else None)


def read_lights(path: str, images=None, log: list | None = None):
    """The light directions in a file (+Y up), or None if it cannot be read.

    With *images*, put in their order: by name when the file names them,
    else in natural (shot) order (helio.io.lp.pair_lights); a warning about
    the pairing goes to *log*."""
    from helio.io.lp import pair_lights, read_lp_named
    try:
        names, L = read_lp_named(path)
        L = np.asarray(L, np.float32)
    except (OSError, ValueError):
        return None
    if L.ndim != 2 or len(L) < 3:
        return None
    if images:
        L, _how, warn = pair_lights(names, L, images)
        if warn and log is not None:
            log.append(f"         {os.path.basename(path)}: {warn}")
    return L


def calibration_for(folder: str, images: list[str], *, roi=None, auto_sphere: bool = True,
                    sphere_L=None, vdc_L=None, shared_lp: str | None = None,
                    detect: Callable | None = None, log: list | None = None):
    """(lights or None, how they were found, kind) for one object folder.

    The classic chain: a light file in the folder; the chrome ball in the
    drawn region (else the shared sphere lights, else uncalibrated); the
    ball found automatically; the shared sphere lights; the dome
    calibrator's; the shared light file; uncalibrated PCA (lights None).
    *kind* is file, sphere, vdc or pca: the classic switched the normal
    solver to least squares for sphere and dome lights, and to
    Uncalibrated PCA without lights."""
    if detect is None:
        from helio.core.processing import auto_detect_sphere

        def detect(paths):
            return auto_detect_sphere(paths, roi)
    for lp in _light_files(folder):
        L = read_lights(lp, images, log)
        if L is not None:
            return L, f"light file: {os.path.basename(lp)}", "file"
        if log is not None:
            log.append(f"         {os.path.basename(lp)} could not be read; not used")

    def fallback_chain(prefix=""):
        if sphere_L is not None:
            return np.asarray(sphere_L, np.float32), prefix + "shared sphere lights", "sphere"
        if vdc_L is not None:
            return (np.asarray(vdc_L, np.float32),
                    prefix + f"dome calibrator ({len(vdc_L)} lights)", "vdc")
        if shared_lp and os.path.exists(shared_lp):
            L = read_lights(shared_lp, images, log)
            if L is not None:
                return L, prefix + f"shared light file: {os.path.basename(shared_lp)}", "file"
        return None, prefix + "Uncalibrated PCA (no calibration found)", "pca"

    if roi is not None:
        L = detect(images)
        if L is not None and len(L) >= 3:
            return np.asarray(L, np.float32), f"the ball in the drawn region ({len(L)} lights)", "sphere"
        if sphere_L is not None:
            return (np.asarray(sphere_L, np.float32),
                    "the drawn region failed: shared sphere lights", "sphere")
        return None, "the drawn region failed: Uncalibrated PCA", "pca"
    if auto_sphere:
        L = detect(images)
        if L is not None and len(L) >= 3:
            return np.asarray(L, np.float32), f"chrome ball found ({len(L)} lights)", "sphere"
        return fallback_chain("no ball found: ")
    return fallback_chain()


# The classic batch exports, as the Publish formats that now write them
# (key, label, ticked by default, what it gives, Publish key).
BATCH_EXPORTS = (
    ("2D renders", (
        ("snapshot_all", "Every view", True, "Every render engine as an image", "all_views"),
        ("heatmap", "Elevation heatmap", True, "Turbo colourmap depth image", "heatmap"),
        ("freq_maps", "32-bit frequency maps", False, "High/low frequency TIFFs", "freq_maps"),
        ("pca", "PCA components", False, "3-channel PCA TIFF", "pca"),
        ("ptm", "PTM coefficients", False, "6-coefficient PTM per pixel", "ptm"))),
    ("3D and web", (
        ("glb", "GLB mesh", True, "Textured 3D model", "glb_mesh"),
        ("hf_viewer", "Fine-detail web viewer", True, "Self-contained Three.js page", "hf_viewer"),
        ("pbr_suite", "PBR suite", False, "Albedo, normal, roughness, displacement", "pbr"),
        ("spec_diff", "Specular / diffuse separation", False, "Separated material maps", "sds"),
        ("combined", "Combined renders", False, "Five complementary archival renders", "normal_map"),
        ("web3d", "3D web viewer", False, "Orbiting HTML viewer with its GLB", "glb"))),
    ("GIS and depth", (
        ("gis_dem", "DEM package", True, "Both integrators' depths, albedo, README", "geotiff"),
        ("gis_hillshade", "Multi-direction hillshade", False, "Eight azimuths averaged", "hillshade"),
        ("gis_analysis", "Terrain analysis maps", False, "Openness, relief, curvature, sky-view",
         "gis_maps"))),
    ("Archive", (
        ("sfs", "Raking set", False, "Eight raking renders at 30°", "raking"),
        ("archive_zip", "Heritage archive ZIP", False, "Self-contained scholarly package", "heritage"),
        ("pub_sheet", "Publication sheet", False, "Multi-panel archival figure", "sheet"))),
)


def batch_keys() -> dict[str, str]:
    return {k: pk for _g, items in BATCH_EXPORTS for k, _l, _d, _w, pk in items}


def raking_set(draw: Callable, out_dir: str, name: str, *, elevation: float = 30.0) -> list[str]:
    """The classic Shade-from-Shade set: the view relit from eight compass
    directions at *elevation*.  *draw(lx, ly, lz)* renders with the light
    in the screen frame (+Y down); azimuth 0° is the top of the image,
    clockwise, as everywhere else in HELIO (HELIO 2 started at the right
    and went anticlockwise on screen)."""
    import cv2
    files = []
    e = math.radians(elevation)
    for az in range(0, 360, 45):
        a = math.radians(az)
        lx, ly_up, lz = math.cos(e) * math.sin(a), math.cos(e) * math.cos(a), max(0.05, math.sin(e))
        img = draw(lx, -ly_up, lz)
        p = os.path.join(out_dir, f"{name}_SfS_{az:03d}deg.jpg")
        cv2.imwrite(p, cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        files.append(p)
    return files


def multi_hillshade(depth, out_dir: str, name: str) -> list[str]:
    """Hillshades from eight azimuths (export_images.hillshade) averaged:
    marks show whatever their direction."""
    import cv2
    from helio.application.export_images import hillshade
    hs = np.mean([hillshade(depth, az) for az in range(0, 360, 45)], axis=0)
    p = os.path.join(out_dir, f"{name}_Hillshade_MultiDir.tif")
    cv2.imwrite(p, (cv2.normalize(hs, None, 0, 1, cv2.NORM_MINMAX) * 255).astype(np.uint8))
    return [p]


def run(folders: list[str], keys: list[str], out_root: str | None, *,
        solve: Callable, export: Callable, calibrate: Callable,
        progress: Callable | None = None, check: Callable | None = None) -> dict:
    """Process each folder: *calibrate(folder, images, log)* → (lights,
    source, kind); *solve(folder, images, lights, kind)* → result;
    *export(key, result, out_dir, name)* → files.  Each object's exports go
    to out_root/name, or name_HELIO_exports inside its own folder.  A
    failure is logged and the run moves on.  Writes and returns the log."""
    progress = progress or (lambda *a: None)
    check = check or (lambda: None)
    lines = [f"Helio Batch — {time.strftime('%Y-%m-%d %H:%M')}", f"Folders: {len(folders)}",
             f"Exports: {', '.join(keys)}", ""]
    ok, errors, per = 0, [], {}
    for i, folder in enumerate(folders):
        check()
        name = os.path.basename(os.path.normpath(folder))
        progress(i, len(folders), name)
        try:
            imgs = _captures(folder)
            if len(imgs) < 3:
                errors.append(f"{name}: fewer than 3 images, skipped")
                lines.append(f"  SKIP  {name}  — fewer than 3 images")
                continue
            sub = []
            L, source, kind = calibrate(folder, imgs, sub)
            lines += sub
            if L is not None and len(L) == len(imgs):
                keep = [j for j in range(len(L)) if np.isfinite(L[j]).all()]
                if len(keep) < len(L):           # frames the light file does not name
                    imgs, L = [imgs[j] for j in keep], np.asarray(L)[keep]
            lines.append(f"  …     {name}: calibration — {source}")
            result = solve(folder, imgs, L, kind)
            out = os.path.join(out_root, name) if out_root else os.path.join(folder, f"{name}_HELIO_exports")
            os.makedirs(out, exist_ok=True)
            files, failed = [], []
            for k in keys:
                check()
                try:
                    files += export(k, result, out, name) or []
                except Exception as e:                     # one export failing is not the object failing
                    failed.append(f"{k}: {type(e).__name__}: {e}")
            per[name] = {"out": out, "files": files, "failed": failed, "calibration": source}
            ok += 1
            lines.append(f"  OK    {name}  — {len(files)} files" + (f", {len(failed)} exports failed"
                                                                      if failed else ""))
            lines += [f"          {f}" for f in failed]
        except Exception as e:
            if type(e).__name__ == "Cancelled":
                lines.append(f"  STOP  cancelled at {name}")
                break
            errors.append(f"{name}: {e}")
            lines.append(f"  ERR   {name}  — {type(e).__name__}: {e}")
    lines += ["", f"Complete: {ok}/{len(folders)} succeeded, {len(errors)} errors"]
    if errors:
        lines += ["", "Errors:"] + [f"  {e}" for e in errors]
    log_root = out_root or (os.path.dirname(os.path.normpath(folders[0])) if folders else ".")
    log_path = os.path.join(log_root, f"HELIO_batch_{time.strftime('%Y%m%d_%H%M%S')}.log")
    try:
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except OSError:
        log_path = ""
    progress(len(folders), len(folders), "")
    return {"ok": ok, "total": len(folders), "errors": errors, "objects": per, "log": log_path,
            "lines": lines}
