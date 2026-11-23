"""
File-system utilities: path validation, temp files, image listing, exception hook, GLB fix.
"""

import logging
import os
import sys
import time
import contextlib as _contextlib
import numpy as np

_log = logging.getLogger(__name__)


_HELIO_DEBUG = os.environ.get("HELIO_DEBUG", "0").strip() == "1"


def _helio_excepthook(exc_type, exc_val, exc_tb):
    import logging as _logging
    _logging.error("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc_val, exc_tb))
    sys.__excepthook__(exc_type, exc_val, exc_tb)


@_contextlib.contextmanager
def log_time(label):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        import logging as _logging
        if dt > 1.0:
            _logging.warning(f"SLOW  {label}  {dt:.2f}s")
        else:
            _logging.debug(f"OK    {label}  {dt:.3f}s")




def estimate_light_from_specular(cx, cy, r, hx, hy):
    nx = (hx - cx) / r
    ny = (cy - hy) / r
    nz = np.sqrt(max(0, 1 - nx**2 - ny**2))
    return np.array([2*nz*nx, 2*nz*ny, 2*nz**2 - 1])


def _extract_highlight_subpixel(roi_masked, cx_local, cy_local, r):
    import cv2
    _, max_val, _, max_loc = cv2.minMaxLoc(roi_masked)
    max_val = float(max_val)
    if max_val < 5:
        return float(max_loc[0]), float(max_loc[1]), 0.0
    _, thresh = cv2.threshold(roi_masked, max_val * 0.85, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    if M["m00"] != 0:
        hx = M["m10"] / M["m00"]
        hy = M["m01"] / M["m00"]
    else:
        hx, hy = float(max_loc[0]), float(max_loc[1])
    h_, w_ = roi_masked.shape
    yy, xx = np.ogrid[:h_, :w_]
    mask = (xx - cx_local)**2 + (yy - cy_local)**2 <= r**2
    mean_sp = float(roi_masked[mask].mean()) + 1e-6
    quality = max_val / mean_sp
    return hx, hy, quality


def _extract_highlight_robust(roi_masked, cx_local, cy_local, r,
                               all_positions=None, min_area_frac=0.002,
                               max_area_frac=0.15):
    import cv2
    _, max_val, _, max_loc = cv2.minMaxLoc(roi_masked)
    max_val = float(max_val)
    if max_val < 5:
        return float(max_loc[0]), float(max_loc[1]), 0.0, False
    _, thresh = cv2.threshold(roi_masked, max_val * 0.90, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        thresh, connectivity=8)
    sphere_area = np.pi * r ** 2
    min_area = sphere_area * min_area_frac
    max_area = sphere_area * max_area_frac
    peak_label = int(labels[int(max_loc[1]), int(max_loc[0])])
    best_hx, best_hy, best_quality, valid = float(max_loc[0]), float(max_loc[1]), 0.0, False
    for lbl in range(1, num_labels):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        blob_mask = (labels == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.4:
            continue
        hx_c, hy_c = float(centroids[lbl][0]), float(centroids[lbl][1])
        h_, w_ = roi_masked.shape
        yy, xx = np.ogrid[:h_, :w_]
        sphere_mask = (xx - cx_local)**2 + (yy - cy_local)**2 <= r**2
        mean_sp = float(roi_masked[sphere_mask].mean()) + 1e-6
        q = max_val / mean_sp
        if lbl == peak_label or q > best_quality:
            best_hx, best_hy, best_quality, valid = hx_c, hy_c, q, True
    if not valid:
        hx, hy, q = _extract_highlight_subpixel(roi_masked, cx_local, cy_local, r)
        return hx, hy, q, False
    return best_hx, best_hy, best_quality, True


def _write_sketchfab_metadata(folder, obj_id, description='', tags=None, license_='CC BY 4.0'):
    import json as _json
    meta = {
        'name': obj_id,
        'description': description or f'Photometric Stereo 3D -- Helio. Object: {obj_id}',
        'tags': tags or ['photometric-stereo', 'RTI', 'cultural-heritage', '3D-scan', 'Helio'],
        'license': license_,
        'private': False,
        'options': {'shading': 'shadeless', 'background': {'color': '#1a1a1e'}},
    }
    with open(os.path.join(folder, 'sketchfab.json'), 'w') as f:
        _json.dump(meta, f, indent=2)


def _write_obj_sketchfab_zip(zip_path, bn,
                              verts_final, faces_final, all_uvs, all_vnormals,
                              albedo_bgr, normal_bgr,
                              roughness_gray=None, ao_gray=None,
                              specular_gray=None, disp_u16=None, jpeg=False):
    import zipfile as _zfm, json as _json, cv2 as _cv2, numpy as _np_lod, io as _io_lod

    # JPEG mode trades a little fidelity for roughly an 85% smaller archive,
    # which is the difference between a Sketchfab upload that succeeds and one
    # that times out.  The normal map is never JPEG: its channels encode
    # direction, and block artefacts there show up as visible bumps in the
    # relit surface.  AO and displacement are dropped in JPEG mode - they are
    # the two largest maps and the least missed.
    _ext = '.jpg' if jpeg else '.png'

    def _png_bytes(img):
        ok, buf = _cv2.imencode('.png', img)
        return buf.tobytes() if ok else b''

    def _tex_bytes(img):
        if jpeg:
            ok, buf = _cv2.imencode('.jpg', img, [int(_cv2.IMWRITE_JPEG_QUALITY), 90])
        else:
            ok, buf = _cv2.imencode('.png', img)
        return buf.tobytes() if ok else b''

    mtl = [
        f'newmtl helio_mat',
        'Ka 0.1 0.1 0.1', 'Kd 0.9 0.9 0.9', 'Ks 0.05 0.05 0.05',
        'Ns 10', 'illum 2',
        f'map_Kd {bn}_baseColor{_ext}',
        f'norm {bn}_normal.png',
        f'map_bump {bn}_normal.png',
    ]
    if roughness_gray is not None:
        mtl.append(f'map_Pr {bn}_roughness{_ext}')
    if ao_gray is not None and not jpeg:
        mtl.append(f'map_Ka {bn}_ao.png')
    if specular_gray is not None:
        mtl.append(f'map_Ks {bn}_specular{_ext}')
    if disp_u16 is not None and not jpeg:
        mtl.append(f'disp   {bn}_displacement.png')

    _hdr_lines = [
        '# Helio Photometric Stereo 3D Export',
        '# UV-mapped OBJ -- Sketchfab / Blender / MorphoSource compatible',
        f'mtllib {bn}.mtl', f'usemtl helio_mat', f'o {bn}',
    ]
    _vb_lod = _io_lod.StringIO()
    _np_lod.savetxt(_vb_lod, verts_final, fmt='v  %.6f %.6f %.6f')
    _vnb_lod = _io_lod.StringIO()
    _np_lod.savetxt(_vnb_lod, all_vnormals, fmt='vn %.6f %.6f %.6f')
    _uvb_lod = _io_lod.StringIO()
    _np_lod.savetxt(_uvb_lod,
                    _np_lod.column_stack([all_uvs[:, 0], 1.0 - all_uvs[:, 1]]),
                    fmt='vt %.6f %.6f')
    _fi_lod = faces_final.astype(_np_lod.int32) + 1
    _fb_lod = '\n'.join(
        f'f {r[0]}/{r[0]}/{r[0]} {r[1]}/{r[1]}/{r[1]} {r[2]}/{r[2]}/{r[2]}'
        for r in _fi_lod)
    obj_b = ('\n'.join(_hdr_lines) + '\n' +
             _vb_lod.getvalue() + _vnb_lod.getvalue() +
             _uvb_lod.getvalue() + _fb_lod + '\n').encode('utf-8')
    mtl_b = '\n'.join(mtl).encode('utf-8')

    zi = lambda n: _zfm.ZipInfo(n)
    with _zfm.ZipFile(zip_path, 'w', _zfm.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr(f'{bn}.obj', obj_b)
        zf.writestr(f'{bn}.mtl', mtl_b)
        zf.writestr(zi(f'{bn}_baseColor{_ext}'), _tex_bytes(albedo_bgr))
        zf.writestr(zi(f'{bn}_normal.png'), _png_bytes(normal_bgr))
        if roughness_gray is not None:
            zf.writestr(zi(f'{bn}_roughness{_ext}'), _tex_bytes(roughness_gray))
        if ao_gray is not None and not jpeg:
            zf.writestr(zi(f'{bn}_ao.png'), _png_bytes(ao_gray))
        if specular_gray is not None:
            zf.writestr(zi(f'{bn}_specular{_ext}'), _tex_bytes(specular_gray))
        if disp_u16 is not None and not jpeg:
            _, db = _cv2.imencode('.png', disp_u16)
            zf.writestr(zi(f'{bn}_displacement.png'), db.tobytes())
        sfmeta = {
            'name': bn,
            'description': 'Photometric Stereo 3D -- Helio.',
            'tags': ['photometric-stereo', 'RTI', 'cultural-heritage', '3D-scan'],
        }
        zf.writestr('sketchfab.json', _json.dumps(sfmeta, indent=2).encode('utf-8'))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def list_images(folder, extensions=None):
    if extensions is None:
        extensions = frozenset({
            '.tif', '.tiff', '.cr2', '.nef', '.arw', '.dng', '.cr3',
            '.rw2', '.orf', '.raf', '.3fr', '.png', '.jpg', '.jpeg', '.bmp',
        })
    try:
        from helio.io.capture import natural_key
        files = sorted(os.listdir(folder), key=natural_key)
    except Exception:
        _log.warning("Cannot list folder %s", folder)
        return []
    return [
        os.path.join(folder, f) for f in files
        if os.path.splitext(f.lower())[1] in extensions
    ]


def sanitise_filename(name, max_len=200):
    safe = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in name)
    return safe[:max_len].strip()

def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / 1024 ** 2:.1f} MB"
    else:
        return f"{size_bytes / 1024 ** 3:.2f} GB"

def get_temp_path(suffix=".png"):
    import tempfile
    from helio.config import PUB_SHOT_DIR
    ensure_dir(PUB_SHOT_DIR)
    return os.path.join(PUB_SHOT_DIR, f"helio_tmp_{int(time.time() * 1e6)}_{suffix}")
