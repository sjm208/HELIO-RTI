"""A flood plane: what lies below a level, how much of the surface that is, and how much it would hold.

A horizontal plane is set at a height; every cell of the surface lower than it is "below".  The figures are the
plain geometry of that:

* **area** - the footprint of those cells (cell count x the cell's area);
* **volume** - the space between the plane and the surface over them: the sum of (level - height) x the cell's
  area.  It is what the plane's level would fill if the hollow were closed round the edge;
* **deepest** and **mean depth** - below the plane;
* **basins** - the connected patches of "below" cells, each with its own area, volume and depth, and whether it
  is *closed* (the rim holds it all round) or *open* (it reaches the edge of the surface or of the region, so
  a plane that high would spill).

Heights, level and cell size are in one unit (mm when a scale is set, else pixels).  No Qt.
"""
from __future__ import annotations

import numpy as np


def flood_totals(z: np.ndarray, level: float, cell: float, mask: np.ndarray | None = None) -> dict:
    """Area, volume and depths below *level*, over the cells in *mask* (all, when None).  Cheap: no labelling."""
    z = np.asarray(z)
    below = z < level
    region = np.ones(z.shape, bool) if mask is None else np.asarray(mask, bool)
    sel = below & region
    n_region = int(region.sum())
    n = int(sel.sum())
    c2 = float(cell) ** 2
    out = dict(cells=n, region_cells=n_region, area=n * c2, region_area=n_region * c2,
               fraction=(n / n_region) if n_region else 0.0, volume=0.0, deepest=0.0, mean_depth=0.0)
    if n:
        depth = float(level) - z[sel].astype(np.float64)
        out["volume"] = float(depth.sum()) * c2
        out["deepest"] = float(depth.max())
        out["mean_depth"] = float(depth.mean())
    return out


def flood_basins(z: np.ndarray, level: float, cell: float, mask: np.ndarray | None = None, top: int = 12,
                 min_cells: int = 4) -> list[dict]:
    """The patches below *level*, largest volume first: ``label``, ``cells``, ``area``, ``volume``, ``deepest``,
    ``closed``, and the centre as ``uv`` (0..1 across the surface).  Patches smaller than *min_cells* are
    ignored (noise); ``top`` are returned."""
    from scipy import ndimage
    z = np.asarray(z)
    region = np.ones(z.shape, bool) if mask is None else np.asarray(mask, bool)
    sel = (z < level) & region
    if not sel.any():
        return []
    lab, n = ndimage.label(sel)                       # 4-connected: a patch must share an edge, not a corner
    # a patch is open when it touches the edge of the surface or of the region
    inner = ndimage.binary_erosion(region, border_value=0)
    edge_cells = region & ~inner
    open_labels = np.unique(lab[edge_cells & sel])
    open_set = set(int(v) for v in open_labels if v)
    idx = np.arange(1, n + 1)
    cells = ndimage.sum(np.ones_like(lab, dtype=np.int64), lab, idx).astype(np.int64)
    depth = float(level) - z.astype(np.float64)
    vol = ndimage.sum(np.where(sel, depth, 0.0), lab, idx) * float(cell) ** 2
    deepest = ndimage.maximum(np.where(sel, depth, 0.0), lab, idx)
    cy = ndimage.mean(np.broadcast_to(np.arange(z.shape[0])[:, None], z.shape).astype(np.float32), lab, idx)
    cx = ndimage.mean(np.broadcast_to(np.arange(z.shape[1])[None, :], z.shape).astype(np.float32), lab, idx)
    rows = []
    for k, label in enumerate(idx):
        if cells[k] < min_cells:
            continue
        rows.append(dict(label=int(label), cells=int(cells[k]), area=float(cells[k]) * float(cell) ** 2,
                         volume=float(vol[k]), deepest=float(deepest[k]), closed=int(label) not in open_set,
                         uv=((float(cx[k]) + 0.5) / z.shape[1], (float(cy[k]) + 0.5) / z.shape[0])))
    rows.sort(key=lambda r: -r["volume"])
    return rows[:top]
