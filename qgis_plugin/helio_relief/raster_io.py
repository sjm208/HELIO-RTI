"""
raster_io - GDAL raster I/O for Helio Relief.

Uses osgeo.gdal, which ships with every QGIS install (rasterio often does
not). Every function opens its own dataset, so all of them are safe to call
from QgsTask worker threads.
"""

import math
import os

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

_GDAL_TYPES = {"uint8": gdal.GDT_Byte, "uint16": gdal.GDT_UInt16, "float32": gdal.GDT_Float32}


def gdal_path(source):
    """QGIS raster sources may carry '|option=..' suffixes that GDAL rejects."""
    return source.split("|")[0]


class RasterInfo:
    """Lightweight description of a raster, read once from its header."""

    def __init__(self, source):
        self.path = gdal_path(source)
        ds = gdal.Open(self.path, gdal.GA_ReadOnly)
        self.width, self.height, self.bands = ds.RasterXSize, ds.RasterYSize, ds.RasterCount
        self.geotransform = tuple(ds.GetGeoTransform())
        self.projection = ds.GetProjection()
        band = ds.GetRasterBand(1)
        self.nodata = band.GetNoDataValue()
        self.data_type = gdal.GetDataTypeName(band.DataType)
        gt = self.geotransform
        self.dx = abs(gt[1]) or 1.0
        self.dy = abs(gt[5]) or 1.0
        self.georeferenced = gt != (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        self.metadata = dict(ds.GetMetadata() or {})
        ds = None

    @property
    def megapixels(self):
        return self.width * self.height / 1e6

    def window_fit(self, max_dim, window=None):
        """Decimated read size for `window` so its long edge is <= max_dim.
        Returns (window, (out_w, out_h), px_scale)."""
        window = window or (0, 0, self.width, self.height)
        _, _, ws, hs = window
        scale = max(1.0, max(ws, hs) / float(max_dim))
        out = (max(1, int(round(ws / scale))), max(1, int(round(hs / scale))))
        return window, out, ws / float(out[0])

    def centre_window(self, size):
        w, h = min(size, self.width), min(size, self.height)
        return ((self.width - w) // 2, (self.height - h) // 2, w, h)


def read_array(source, window=None, out_size=None, band=1):
    """Read one band as float32 (nodata -> NaN).

    window = (xoff, yoff, xsize, ysize) in full-resolution pixels; out_size =
    (w, h) for an area-averaged decimated read. Returns (array, geotransform)."""
    ds = gdal.Open(gdal_path(source), gdal.GA_ReadOnly)
    b = ds.GetRasterBand(band)
    xoff, yoff, xs, ys = window or (0, 0, ds.RasterXSize, ds.RasterYSize)
    bw, bh = out_size or (xs, ys)
    alg = gdal.GRIORA_Average if (bw < xs or bh < ys) else gdal.GRIORA_NearestNeighbour
    arr = b.ReadAsArray(xoff, yoff, xs, ys, buf_xsize=bw, buf_ysize=bh, resample_alg=alg).astype(np.float32)
    nd = b.GetNoDataValue()
    if nd is not None and not math.isnan(nd):
        arr[arr == np.float32(nd)] = np.nan
    gt = ds.GetGeoTransform()
    sx, sy = xs / float(bw), ys / float(bh)
    new_gt = (gt[0] + xoff * gt[1] + yoff * gt[2], gt[1] * sx, gt[2] * sy,
              gt[3] + xoff * gt[4] + yoff * gt[5], gt[4] * sx, gt[5] * sy)
    return arr, new_gt


def _read_three(ds, ref_size, window=None, out_size=None):
    """Three bands of an image registered to a reference raster of size ref_size = (width, height): the window is
    in reference pixels, rescaled if the image has a different pixel count. float32 (H, W, 3), raw values."""
    sx, sy = ds.RasterXSize / float(ref_size[0]), ds.RasterYSize / float(ref_size[1])
    xoff, yoff, xs, ys = window or (0, 0, ref_size[0], ref_size[1])
    win = (int(xoff * sx), int(yoff * sy), max(1, int(xs * sx)), max(1, int(ys * sy)))
    win = (win[0], win[1], min(win[2], ds.RasterXSize - win[0]), min(win[3], ds.RasterYSize - win[1]))
    bw, bh = out_size or (xs, ys)
    chans = []
    for i in range(3):
        b = ds.GetRasterBand(min(i + 1, ds.RasterCount))
        a = b.ReadAsArray(*win, buf_xsize=bw, buf_ysize=bh, resample_alg=gdal.GRIORA_Average).astype(np.float32)
        chans.append(a)
    return np.dstack(chans)


def read_rgb(source, ref_size, window=None, out_size=None):
    """Read a colour/albedo image registered to a reference raster of size
    ref_size = (width, height). The window is given in reference pixels and
    rescaled if the colour image has a different pixel count. Returns float32
    (H, W, 3) in [0, 1]."""
    ds = gdal.Open(gdal_path(source), gdal.GA_ReadOnly)
    rgb = _read_three(ds, ref_size, window, out_size)
    rgb /= np.float32(white_level(ds, gdal_path(source)))
    return np.clip(np.nan_to_num(rgb), 0.0, 1.0)


def is_slow_to_window(source, min_megapixels=16.0):
    """True for a large raster that GDAL cannot read a window of without decoding everything before it - PNG
    and JPEG files, or TIFFs in whole-width strips of many rows. Such files are copied to a tiled GeoTIFF once."""
    ds = gdal.Open(gdal_path(source), gdal.GA_ReadOnly)
    if ds is None or ds.RasterXSize * ds.RasterYSize / 1e6 < min_megapixels:
        return False
    driver = ds.GetDriver().ShortName
    if driver in ("PNG", "JPEG"):
        return True
    bw, bh = ds.GetRasterBand(1).GetBlockSize()
    return driver == "GTiff" and bw >= ds.RasterXSize and bh > 64


def make_tiled_copy(source, dest, ref=None, progress=None):
    """Copy a raster to a tiled, compressed GeoTIFF (fast windows), bands and data type kept. With ref (a raster
    of the same pixel size) the copy takes ref's georeferencing, so QGIS draws it right over ref."""
    opts = dict(format="GTiff", creationOptions=["TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512", "COMPRESS=LZW",
                                                 "BIGTIFF=IF_SAFER"])
    if ref is not None:
        rds = gdal.Open(gdal_path(ref), gdal.GA_ReadOnly)
        gt = rds.GetGeoTransform(can_return_null=True)
        if gt:
            w, h = rds.RasterXSize, rds.RasterYSize
            opts["outputBounds"] = [gt[0], gt[3], gt[0] + w * gt[1], gt[3] + h * gt[5]]
            if rds.GetProjection():
                opts["outputSRS"] = rds.GetProjection()
    if progress is not None:
        opts["callback"] = lambda pc, _msg, _data: (progress(int(pc * 100), "Copying to a tiled GeoTIFF"), 1)[1]
    out = gdal.Translate(dest, gdal_path(source), **opts)
    if out is None:
        raise RuntimeError(f"Could not write {dest}")
    out = None
    return dest


def read_normals(source, ref_size, window=None, out_size=None):
    """Read a normal map registered to a reference raster (windowing as read_rgb), its encoded values scaled to
    0..1 by the data type's full range (255 / 65535) - decode with relief_engine.decode_normals."""
    ds = gdal.Open(gdal_path(source), gdal.GA_ReadOnly)
    rgb = _read_three(ds, ref_size, window, out_size)
    dtype = gdal.GetDataTypeName(ds.GetRasterBand(1).DataType)
    full = {"Byte": 255.0, "UInt16": 65535.0}.get(dtype)
    if full is None:
        full = float(max(np.nanmax(rgb), 1e-9))
    return np.clip(np.nan_to_num(rgb / np.float32(full)), 0.0, 1.0)


_WHITE = {}


def white_level(ds, path):
    """Level a colour image's white sits at: 255 for 8-bit; otherwise the aligned VRT's HPSS_WHITE, or the
    99.8th percentile of a decimated read of its bands (16-bit albedos rarely use the full range) - one value for
    the whole image, so previews and tiles get the same colour."""
    if gdal.GetDataTypeName(ds.GetRasterBand(1).DataType) == "Byte":
        return 255.0
    tagged = ds.GetMetadataItem("HPSS_WHITE")
    if tagged:
        return float(tagged)
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        key = (path, 0.0)
    if key not in _WHITE:
        k = max(1, int(max(ds.RasterXSize, ds.RasterYSize) / 1024))
        w, h = max(1, ds.RasterXSize // k), max(1, ds.RasterYSize // k)
        v = np.concatenate([ds.GetRasterBand(i + 1).ReadAsArray(buf_xsize=w, buf_ysize=h).astype(np.float64).ravel()
                            for i in range(min(3, ds.RasterCount))])
        v = v[np.isfinite(v) & (v > 0)]
        _WHITE[key] = float(max(np.percentile(v, 99.8), 1e-9)) if v.size else 1.0
    return _WHITE[key]


class GeoTiffWriter:
    """Tiled, compressed GeoTIFF written block by block (keeps memory flat for
    gigapixel renders). Builds overviews on close so QGIS pans it instantly."""

    def __init__(self, path, width, height, bands, dtype, geotransform, projection, nodata=None, options=None):
        # options=[] writes GDAL's plain default layout (uncompressed strips), exactly like gdaldem's own output
        opts = list(options) if options is not None else [
            "TILED=YES", "COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER", "PREDICTOR=%d" % (3 if dtype == "float32" else 2)]
        if bands == 3 and dtype != "float32":
            opts.append("PHOTOMETRIC=RGB")
        if os.path.exists(path):
            gdal.GetDriverByName("GTiff").Delete(path)
        self.path = path
        self.ds = gdal.GetDriverByName("GTiff").Create(path, width, height, bands, _GDAL_TYPES[dtype], options=opts)
        self.ds.SetGeoTransform(geotransform)
        if projection:
            self.ds.SetProjection(projection)
        if nodata is not None:
            for i in range(bands):
                self.ds.GetRasterBand(i + 1).SetNoDataValue(nodata)

    def set_descriptions(self, names):
        for i, name in enumerate(names[:self.ds.RasterCount]):
            self.ds.GetRasterBand(i + 1).SetDescription(str(name))

    def write(self, arr, xoff=0, yoff=0):
        """arr is (H, W) or (H, W, bands)."""
        if arr.ndim == 2:
            self.ds.GetRasterBand(1).WriteArray(arr, xoff, yoff)
        else:
            for i in range(arr.shape[2]):
                self.ds.GetRasterBand(i + 1).WriteArray(arr[:, :, i], xoff, yoff)

    def close(self, overviews=True):
        if self.ds is None:
            return
        if overviews:
            levels, f = [], 2
            while max(self.ds.RasterXSize, self.ds.RasterYSize) / f >= 256:
                levels.append(f)
                f *= 2
            if levels:
                self.ds.BuildOverviews("AVERAGE", levels)
        self.ds.FlushCache()
        self.ds = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(overviews=exc[0] is None)


def write_raster(path, arr, geotransform, projection, dtype="float32", nodata=None, overviews=True):
    """One-shot write of an (H, W), (H, W, B) or band-first (B, H, W) array.
    Band-first input is detected when the first axis is the shortest."""
    if arr.ndim == 3 and arr.shape[0] < min(arr.shape[1:]):
        arr = np.moveaxis(arr, 0, -1)
    bands = 1 if arr.ndim == 2 else arr.shape[2]
    with GeoTiffWriter(path, arr.shape[1], arr.shape[0], bands, dtype, geotransform, projection, nodata) as w:
        w.write(arr.astype(np.dtype(dtype), copy=False))
        if not overviews:
            w.close(overviews=False)
    return path


def export_png(tif_path, png_path):
    """Stream a finished 8/16-bit GeoTIFF into a PNG without loading it."""
    gdal.Translate(png_path, tif_path, format="PNG")
    aux = png_path + ".aux.xml"
    if os.path.exists(aux):
        os.remove(aux)
    return png_path


def build_vrt(path, sources, separate=True):
    gdal.BuildVRT(path, sources, separate=separate).FlushCache()
    return path
