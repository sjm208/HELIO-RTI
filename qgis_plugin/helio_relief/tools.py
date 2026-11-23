"""
Measuring, tracing and comparing height maps - the numerical side of the Tools tab (no Qt, no GDAL).

- profile / profile_stats / area_stats: heights along a line or in a box, in the height map's units, with the
  depth of the deepest groove below the line's trend and the volume of hollows below a box's best-fit plane;
- line_strength / trace_lines: incised (or raised) lines as polylines - Hessian curvature across the line at the
  line's width, non-maximum suppression across it, hysteresis, thinning, pixel chains, Douglas-Peucker - the
  line drawing of an archaeological illustration;
- compare_integrations / fuse_heights: two integrations of the same normals (Poisson, Frankot-Chellappa) side by
  side - fine detail, pixel noise, drift - and one height map with the broad form of one and the detail of the
  other.
"""

import math

import numpy as np
from scipy import ndimage as ndi

try:
    from . import relief_engine as eng
except ImportError:                     # tests import the modules directly
    import relief_engine as eng


# ---------------------------------------------------------------------------- measuring

def profile(z, x0, y0, x1, y1, step=0.5):
    """(distance px, heights) along the line from (x0, y0) to (x1, y1) (pixel centres of z), bilinear, a sample
    every `step` px."""
    length = math.hypot(x1 - x0, y1 - y0)
    n = max(2, int(length / step) + 1)
    t = np.linspace(0.0, 1.0, n)
    xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    z = np.asarray(z, dtype=np.float64)
    h = ndi.map_coordinates(np.nan_to_num(z, nan=np.nanmedian(z)), [ys, xs], order=1, mode="nearest")
    return t * length, h


def profile_stats(dist, h):
    """Relief along a profile. The straight trend (least squares) is the surface the marks are cut into:
    depth = the deepest point below it (an incision), height = the highest above it (a ridge, burr), width = the
    deepest feature's width at half its depth. Distances in px."""
    dist, h = np.asarray(dist, float), np.asarray(h, float)
    A = np.c_[dist, np.ones_like(dist)]
    c, *_ = np.linalg.lstsq(A, h, rcond=None)
    r = h - A @ c
    i = int(np.argmin(r))
    depth = float(-r[i])
    lo = hi = i
    while lo > 0 and r[lo - 1] < r[i] / 2.0:
        lo -= 1
    while hi < len(r) - 1 and r[hi + 1] < r[i] / 2.0:
        hi += 1
    return dict(length=float(dist[-1]), min=float(h.min()), max=float(h.max()), range=float(h.max() - h.min()),
                depth=depth, depth_at=float(dist[i]), height=float(r.max()), height_at=float(dist[int(np.argmax(r))]),
                width=float(dist[hi] - dist[lo]) if depth > 0 else 0.0, slope=float(c[0]), residual=r)


def area_stats(z):
    """Heights in a box: mean, min, max, sd; after removing the best-fit plane, the relief (99th - 1st
    percentile) and the volumes of hollows below the plane and bumps above it (height units x px^2)."""
    z = np.asarray(z, dtype=np.float64)
    ok = np.isfinite(z)
    zz = np.where(ok, z, np.nanmedian(z))
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    A = np.c_[xx[ok], yy[ok], np.ones(int(ok.sum()))]
    use = np.ones(int(ok.sum()), bool)
    tol0 = 1e-6 * (float(np.ptp(zz[ok])) + 1e-12)
    c, *_ = np.linalg.lstsq(A, zz[ok], rcond=None)
    for _ in range(4):                       # robust: the surface the marks are in, not pulled by the marks
        res = zz[ok] - A @ c
        centre = float(np.median(res[use]))
        mad = float(np.median(np.abs(res[use] - centre))) * 1.4826
        new = np.abs(res - centre) <= max(2.5 * mad, tol0)
        if new.sum() < 0.3 * use.size or (new == use).all():
            break
        use = new
        c, *_ = np.linalg.lstsq(A[use], zz[ok][use], rcond=None)
    r = zz - (c[0] * xx + c[1] * yy + c[2])
    rv = r[ok]
    return dict(pixels=int(ok.sum()), mean=float(zz[ok].mean()), min=float(zz[ok].min()), max=float(zz[ok].max()),
                sd=float(zz[ok].std()), relief=float(np.percentile(rv, 99) - np.percentile(rv, 1)),
                hollow_volume=float(-rv[rv < 0].sum()), bump_volume=float(rv[rv > 0].sum()))


# ---------------------------------------------------------------------------- line tracing

TRACE_KINDS = {"incised": "Incised lines (grooves, tooling, drypoint)", "raised": "Raised lines (ridges, burr)",
               "both": "Both"}


def line_strength(z, width_px, kind="incised"):
    """(strength, across-angle): how line-like each pixel is at the given line width - the Hessian of the
    Gaussian-smoothed surface, curvature across the line (positive for a groove, negative for a ridge) where it
    clearly exceeds the curvature along it; scale-normalised. The angle points across the line."""
    s = max(float(width_px) / 2.0, 0.7)
    z = np.asarray(z, dtype=np.float64)
    # SciPy's truncated derivative kernels do not sum to exactly zero, so a large height offset (a depth map at
    # ~8000 units) leaks into the second derivatives as a false curvature everywhere: take the local mean off first
    # (it also removes tilt and drift, which have no curvature at this scale anyway)
    z = z - ndi.gaussian_filter(z, 8.0 * s, mode="nearest")
    zrr = ndi.gaussian_filter(z, s, order=(2, 0))
    zcc = ndi.gaussian_filter(z, s, order=(0, 2))
    zrc = ndi.gaussian_filter(z, s, order=(1, 1))
    half = 0.5 * (zrr + zcc)
    rad = np.sqrt(0.25 * (zrr - zcc) ** 2 + zrc ** 2)
    l1, l2 = half + rad, half - rad                  # l1 >= l2
    eps = 1e-12
    # a groove bends up strongly across and hardly along: keep l1 where |l2| is small beside it - round pits
    # (l2 ~ l1) and saddles fade out, grainy but line-like grooves stay
    groove = np.clip(l1, 0.0, None) * np.clip(1.0 - np.abs(l2) / (np.abs(l1) + eps), 0.0, 1.0)
    ridge = np.clip(-l2, 0.0, None) * np.clip(1.0 - np.abs(l1) / (np.abs(l2) + eps), 0.0, 1.0)
    # eigenvector of the curvature that is used: for l1 (groove) and l2 (ridge)
    th1 = 0.5 * np.arctan2(2.0 * zrc, zrr - zcc)     # angle (from the row axis) of l1's eigenvector
    if kind == "incised":
        strength, theta = groove, th1
    elif kind == "raised":
        strength, theta = ridge, th1 + math.pi / 2.0
    else:
        use_ridge = ridge > groove
        strength, theta = np.where(use_ridge, ridge, groove), np.where(use_ridge, th1 + math.pi / 2.0, th1)
    return np.clip(strength, 0.0, None) * s * s, theta


# Line widths for the two kinds of line a tooled or engraved surface carries. No automatic choice held up on real
# scans (the scale-normalised peak rises with width, contrast ratios saturate, traced length favours grain): a
# surface has lines at several scales at once - on the 1563 binding fine hatching at 4-6 px, tool outlines at
# 16-32 px - so the user picks which.
TRACE_PRESETS = {"Fine": 4.0, "Main": 16.0}


def _suppress(strength, theta):
    """Keep pixels whose strength is a maximum across the line (non-maximum suppression)."""
    h, w = strength.shape
    rr, cc = np.mgrid[0:h, 0:w].astype(np.float64)
    dr, dc = np.cos(theta), np.sin(theta)             # across-line direction (row, column)
    a = ndi.map_coordinates(strength, [rr + dr, cc + dc], order=1, mode="nearest")
    b = ndi.map_coordinates(strength, [rr - dr, cc - dc], order=1, mode="nearest")
    return (strength >= a) & (strength >= b)


def _thin(mask):
    """Zhang-Suen thinning to one-pixel-wide lines."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            P = [img[1:-1, 1:-1], img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:], img[2:, 1:-1], img[2:, :-2],
                 img[1:-1, :-2], img[:-2, :-2]]
            p1, p2, p3, p4, p5, p6, p7, p8, p9 = P
            nb = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            seq = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            trans = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rm = (p1 == 1) & (nb >= 2) & (nb <= 6) & (trans == 1) & cond
            if rm.any():
                img[1:-1, 1:-1][rm] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]   # 4-neighbours first


def _chains(skel):
    """One-pixel-wide skeleton -> list of pixel chains [(row, col), ...], broken at junctions."""
    # crossing number - separate runs of set pixels round the 8-neighbourhood: 2 inside a line (a staircase corner
    # too, though it touches three pixels), 1 at an end, 3 or more only at a true junction
    img = np.pad(skel.astype(np.uint8), 1)
    ring = [img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:], img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2],
            img[:-2, :-2]]
    cross = sum(((ring[i] == 0) & (ring[(i + 1) % 8] == 1)).astype(np.uint8) for i in range(8))
    seg = skel & (cross <= 2)                         # junction pixels removed: every piece is a path or a loop
    lab, n = ndi.label(seg, structure=np.ones((3, 3)))
    chains = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        rs, cs = np.nonzero(sub)
        pts = set(zip(rs.tolist(), cs.tolist()))
        if len(pts) < 2:
            continue
        counts = {p: sum((p[0] + a, p[1] + b) in pts for a, b in _NB) for p in pts}
        start = min((p for p in pts if counts[p] <= 1), default=next(iter(pts)))
        chain, seen, cur = [start], {start}, start
        while True:
            nxt = next(((cur[0] + a, cur[1] + b) for a, b in _NB if (cur[0] + a, cur[1] + b) in pts
                        and (cur[0] + a, cur[1] + b) not in seen), None)
            if nxt is None:
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        r0, c0 = sl[0].start, sl[1].start
        chains.append([(r + r0, c + c0) for r, c in chain])
    return chains


def _simplify(pts, tol):
    """Douglas-Peucker simplification of [(x, y), ...]."""
    if len(pts) < 3 or tol <= 0:
        return pts
    a, b = np.asarray(pts[0], float), np.asarray(pts[-1], float)
    arr = np.asarray(pts, float)
    ab = b - a
    n = math.hypot(*ab)
    d = (np.abs(ab[0] * (arr[:, 1] - a[1]) - ab[1] * (arr[:, 0] - a[0])) / n if n > 0
         else np.hypot(arr[:, 0] - a[0], arr[:, 1] - a[1]))
    i = int(np.argmax(d))
    if d[i] <= tol:
        return [pts[0], pts[-1]]
    return _simplify(pts[:i + 1], tol)[:-1] + _simplify(pts[i:], tol)


def _polyline_length(pts):
    if len(pts) < 2:
        return 0.0
    a = np.asarray(pts, float)
    return float(np.hypot(*np.diff(a, axis=0).T).sum())


def _chaikin(pts, n):
    """Chaikin corner cutting, n rounds, ends kept: smooth curves from pixel staircases."""
    p = np.asarray(pts, float)
    for _ in range(int(n)):
        if len(p) < 3:
            break
        q, r = 0.75 * p[:-1] + 0.25 * p[1:], 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        p = np.vstack([p[:1], mid[1:-1], p[-1:]])
    return [(float(x), float(y)) for x, y in p]


def bridge_gaps(lines, max_gap, max_turn_deg=40.0):
    """Join line ends closer than max_gap px whose directions agree - a groove broken by a worn patch or by a line
    crossing it comes back as one line. Lines of different kinds are never joined."""
    from scipy.spatial import cKDTree
    lines = [dict(l, points=list(l["points"])) for l in lines if len(l["points"]) >= 2]
    cos_max = math.cos(math.radians(max_turn_deg))
    for _ in range(12):
        ends = []
        for i, l in enumerate(lines):
            p = np.asarray(l["points"], float)
            k = min(4, len(p) - 1)
            for which, a, b in ((0, p[0], p[k]), (1, p[-1], p[-1 - k])):
                t = a - b
                ends.append((i, which, a, t / (math.hypot(*t) + 1e-12)))
        if len(ends) < 2:
            break
        tree = cKDTree(np.array([e[2] for e in ends]))
        pairs = sorted(tree.query_pairs(max_gap), key=lambda ab: math.hypot(*(ends[ab[0]][2] - ends[ab[1]][2])))
        used, merged = set(), 0
        for a, b in pairs:
            ea, eb = ends[a], ends[b]
            if ea[0] == eb[0] or ea[0] in used or eb[0] in used:
                continue
            la, lb = lines[ea[0]], lines[eb[0]]
            if la["kind"] != lb["kind"]:
                continue
            gap = eb[2] - ea[2]
            g = math.hypot(*gap)
            if g > 1e-9 and (np.dot(ea[3], gap / g) < cos_max or np.dot(eb[3], -gap / g) < cos_max):
                continue
            pa = la["points"][::-1] if ea[1] == 0 else la["points"]
            pb = lb["points"] if eb[1] == 0 else lb["points"][::-1]
            wa, wb = _polyline_length(pa), _polyline_length(pb)
            lines[ea[0]] = dict(la, points=pa + pb,
                                strength=(la["strength"] * wa + lb["strength"] * wb) / max(wa + wb, 1e-12))
            lines[eb[0]] = None
            used.update((ea[0], eb[0]))
            merged += 1
        lines = [l for l in lines if l is not None]
        if not merged:
            break
    return lines


def trace_lines(z, width_px=3.0, sensitivity=0.5, min_length_px=12.0, kind="incised", simplify_px=0.7,
                strength=None, bridge_px=0.0, smooth=0, min_strength=0.0):
    """Incised / raised lines of a height map as polylines. Returns (lines, strength map) with lines =
    [dict(points [(x, y) pixel centres], length_px, strength, kind, depth, orientation)]. `sensitivity` 0..1:
    higher keeps fainter lines (the strong threshold is (1 - sensitivity) x the 99th-percentile strength; pixels
    half as strong are kept where they connect to strong ones). bridge_px joins ends that nearly meet and point
    the same way; smooth = rounds of corner cutting; min_strength drops lines whose mean strength is below that
    share of the 99th percentile. depth = the mark's depth (or height) against the surface round it, in height
    units; orientation = degrees anticlockwise from east, 0-180."""
    z = np.asarray(z, dtype=np.float64)
    z = np.where(np.isfinite(z), z, np.nanmedian(z))
    resid = z - ndi.gaussian_filter(z, 2.0 * max(float(width_px), 1.0), mode="nearest")
    kinds = ("incised", "raised") if kind == "both" else (kind,)
    lines, total = [], None
    for k in kinds:
        s, theta = line_strength(z, width_px, k) if strength is None else strength
        total = s if total is None else np.maximum(total, s)
        pos = s[s > 0]
        if pos.size == 0:
            continue
        hi = max(1e-12, (1.0 - float(sensitivity)) * float(np.percentile(pos, 99)))
        cand = _suppress(s, theta) & (s >= 0.5 * hi)
        lab, n = ndi.label(cand, structure=np.ones((3, 3)))
        if n == 0:
            continue
        strong = np.zeros(n + 1, bool)
        strong[np.unique(lab[cand & (s >= hi)])] = True
        strong[0] = False
        skel = _thin(strong[lab])
        raw = []
        for chain in _chains(skel):
            if len(chain) < 2:
                continue
            rr = np.array([r for r, _ in chain])
            cc = np.array([c for _, c in chain])
            raw.append(dict(points=[(float(c), float(r)) for r, c in chain], strength=float(s[rr, cc].mean()),
                            kind=k))
        if bridge_px > 0:
            raw = bridge_gaps(raw, float(bridge_px))
        p99 = float(np.percentile(pos, 99))
        for line in raw:
            pts = line["points"]
            length = _polyline_length(pts)
            if length < min_length_px or line["strength"] < float(min_strength) * p99:
                continue
            a = np.asarray(pts, float)
            ri = np.clip(np.rint(a[:, 1]).astype(int), 0, z.shape[0] - 1)
            ci = np.clip(np.rint(a[:, 0]).astype(int), 0, z.shape[1] - 1)
            depth = float(-np.mean(resid[ri, ci]) if k == "incised" else np.mean(resid[ri, ci]))
            c = a - a.mean(axis=0)
            cov = c.T @ c
            orient = float(np.degrees(0.5 * np.arctan2(2.0 * cov[0, 1], cov[0, 0] - cov[1, 1])))
            orient = (-orient) % 180.0                         # image rows run down: anticlockwise from east
            if smooth:
                pts = _chaikin(pts, smooth)
            lines.append(dict(points=_simplify(pts, simplify_px), length_px=length, strength=line["strength"],
                              kind=k, depth=depth, orientation=orient))
    return lines, (total if total is not None else np.zeros_like(z))


# ---------------------------------------------------------------------------- comparing integrations

def _levelled(z):
    """Heights with the drift (> 150 px) removed, scaled so the tooling-scale relief (the 8-32 px band, which
    every integration of the same normals holds alike) has unit spread - integrations in any units, with any
    amount of pixel noise, become comparable."""
    z = np.asarray(z, dtype=np.float32)
    z = z - eng.gaussian_blur(z, 150.0)
    ref = float(np.std(_band(z, 8.0, 32.0)))
    return z / max(ref, 1e-12)


def _band(z, s_small, s_large):
    return eng.gaussian_blur(z, s_small) - eng.gaussian_blur(z, s_large)


def compare_integrations(za, zb):
    """How two integrations of the same normals differ on one window (same grid): per integration, fine detail
    (sd of the 1.5-8 px band), pixel noise (sd below 1.5 px), drift (sd of the form broader than 150 px against the
    relief), signal to noise; plus how well their tooling-scale bands agree and b's detail gain over a's."""
    out = {}
    for name, z in (("a", za), ("b", zb)):
        z = np.asarray(z, dtype=np.float32)
        lv = _levelled(z)
        detail = float(np.std(_band(lv, 1.5, 8.0)))
        noise = float(np.std(lv - eng.gaussian_blur(lv, 1.5)))
        broad = z.astype(np.float64) - (z - eng.gaussian_blur(z, 150.0))
        rel = float(np.std(z - eng.gaussian_blur(z, 150.0)))
        out[name] = dict(detail=detail, noise=noise, snr=detail / max(noise, 1e-12),
                         drift=float(np.std(broad - broad.mean())) / max(rel, 1e-12))
    # agreement on the tooling-detail band, which both integrations resolve; broader bands differ by design
    ma, mb = _band(_levelled(za), 2.0, 8.0), _band(_levelled(zb), 2.0, 8.0)
    out["agreement"] = float(np.corrcoef(ma.ravel(), mb.ravel())[0, 1])
    out["detail_gain"] = out["b"]["detail"] / max(out["a"]["detail"], 1e-12)
    return out


def recommend(metrics_list, names=("A", "B")):
    """('a' | 'b' | 'fuse', reason) from compare_integrations results over several windows."""
    a = {k: float(np.median([m["a"][k] for m in metrics_list])) for k in ("detail", "noise", "snr", "drift")}
    b = {k: float(np.median([m["b"][k] for m in metrics_list])) for k in ("detail", "noise", "snr", "drift")}
    gain = b["detail"] / max(a["detail"], 1e-12)
    na, nb = names
    if b["snr"] >= a["snr"] * 1.1 and gain >= 0.95:
        return "b", f"{nb} holds {100 * (gain - 1):+.0f} % fine detail with a cleaner signal ({b['snr']:.2f} vs {a['snr']:.2f})"
    if a["snr"] >= b["snr"] * 1.1 and gain <= 1.15:
        return "a", (f"{na} is cleaner ({a['snr']:.2f} vs {b['snr']:.2f} detail / noise) and {nb} adds little "
                     f"detail ({100 * (gain - 1):+.0f} %)")
    if gain > 1.15:
        return "fuse", (f"{nb} holds {100 * (gain - 1):+.0f} % more fine detail but more noise; {na} is cleaner - fuse "
                        f"{na}'s form with {nb}'s detail (the edge-preserving smoothing then cleans it)")
    return ("a" if a["snr"] >= b["snr"] else "b"), "the two integrations are nearly the same here"


def fuse_gain(za, zb, sigma):
    """Scale that brings b's relief finer than `sigma` to a's (the two integrations use different height scales),
    from their 1.5 sigma - 4 sigma bands, which both hold."""
    ba, bb = _band(np.asarray(za, np.float32), 1.5 * sigma, 4.0 * sigma), _band(np.asarray(zb, np.float32),
                                                                                1.5 * sigma, 4.0 * sigma)
    return float(np.sum(ba * bb) / max(float(np.sum(bb * bb)), 1e-12))


def fuse_heights(za, zb, sigma=3.0, gain=1.0):
    """a's form and b's detail: a blurred at `sigma` px plus b's relief finer than that, scaled by `gain`."""
    za, zb = np.asarray(za, np.float32), np.asarray(zb, np.float32)
    return eng.gaussian_blur(za, sigma) + np.float32(gain) * (zb - eng.gaussian_blur(zb, sigma))
