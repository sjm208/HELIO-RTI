"""Whole-image arithmetic on every core, and the cache that keeps light-free work.

Two things make relighting fluid at the preview's full size (2500 px on the
long edge, 4.7 million pixels) without lowering its resolution:

``by_rows``
    Runs a per-pixel function over bands of rows on a small thread pool.
    NumPy and OpenCV release the GIL inside their loops, so the bands really
    do run at once.  A band computes exactly what the whole-array expression
    computes at those pixels, so only per-pixel maths belongs inside one:
    no blurs, no percentiles, nothing that reads a neighbour.

``cached``
    Keeps work that does not change when the light moves (normalised
    normals, smoothed normals, blurred colour, PTM planes at preview size)
    so a drag pays only for the part that does.  Entries are keyed by the
    *identity* of the arrays they were made from, held weakly: a new solve
    replaces the arrays (HELIO never writes into a solved array; it assigns
    a new one), the old entries stop matching, and the old arrays are freed
    as usual instead of being kept alive by the cache.
"""
from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MIN_PIXELS = 200_000            # below this a single thread is quicker
_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()
_IN_BAND = threading.local()


def workers() -> int:
    """Threads for whole-image work: every core but one, at most eight."""
    n = os.environ.get("HELIO_THREADS")
    if n and n.isdigit():
        return max(1, int(n))
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=workers(), thread_name_prefix="helio-rows")
        return _POOL


def _run_band(fn, y0, y1):
    _IN_BAND.on = True
    try:
        fn(y0, y1)
    finally:
        _IN_BAND.on = False


def by_rows(fn, h: int, w: int = 1, *, min_pixels: int = MIN_PIXELS,
            parts: int | None = None) -> None:
    """Call ``fn(y0, y1)`` for bands of rows covering ``0..h``, in parallel.

    *parts* is the number of bands (default: twice the threads, a little
    slack for uneven bands).  Returns when every band is done and re-raises
    the first error.  A call from inside a band runs inline, so nesting
    cannot deadlock the pool.
    """
    h = int(h)
    n = workers()
    if h <= 1 or n == 1 or h * max(1, int(w)) < min_pixels or getattr(_IN_BAND, "on", False):
        fn(0, h)
        return
    parts = min(h, parts or n * 2)
    edges = np.linspace(0, h, parts + 1).astype(int)
    futs = [_pool().submit(_run_band, fn, int(a), int(b))
            for a, b in zip(edges[:-1], edges[1:]) if b > a]
    err = None
    for f in futs:                         # wait for all before raising
        try:
            f.result()
        except BaseException as e:         # noqa: BLE001 - re-raised below
            err = err or e
    if err is not None:
        raise err


def map_rows(fn, a: np.ndarray, out: np.ndarray, *, min_pixels: int = MIN_PIXELS) -> np.ndarray:
    """``out[y0:y1] = fn(a[y0:y1])`` band by band; returns *out*."""
    def band(y0, y1):
        out[y0:y1] = fn(a[y0:y1])
    by_rows(band, a.shape[0], a.shape[1] if a.ndim > 1 else 1, min_pixels=min_pixels)
    return out


STRIP_BYTES = 128 * 2**20       # the working data one strip may hold


def parts_for(n: int, bytes_total: float) -> int:
    """How many strips to cut *n* rows (or pixels) into so that each holds
    about STRIP_BYTES of working data: strips run several at a time, so it
    is their size, not their number, that sets the memory peak."""
    return int(min(max(n, 1), max(2 * workers(), int(np.ceil(bytes_total / STRIP_BYTES)))))


def _lstsq_columns_rust(A: np.ndarray, B: np.ndarray):
    """``lstsq_columns`` from helio_core_rs: the pseudo-inverse of the (small)
    A is taken here with ``lstsq``'s own rank cut-off, so rank-deficient
    systems give the same minimum-norm answer, and the per-pixel product runs
    in Rust.  float32 result.  None when the module is missing or B is not an
    8/16-bit or float32 stack (NumPy then does it).  A float32, C-contiguous B
    is read in place; anything else is converted a strip at a time, so a
    46-megapixel stack is never copied whole."""
    from helio.core._native import rs
    if (rs is None or not hasattr(rs, "apply_pinv") or A.ndim != 2 or B.ndim != 2
            or A.shape[0] != B.shape[0] or A.shape[1] == 0
            or B.dtype not in (np.uint8, np.uint16, np.float32)):
        return None
    n, n_px = B.shape
    P = np.ascontiguousarray(np.linalg.pinv(np.asarray(A, np.float64),
                                            rcond=np.finfo(np.float64).eps * max(A.shape)))
    if B.dtype == np.float32 and B.flags.c_contiguous:
        return rs.apply_pinv(P, B)
    out = np.empty((A.shape[1], n_px), np.float32)
    step = -(-n_px // parts_for(n_px, n_px * n * 4))
    for a in range(0, n_px, step):
        b = min(a + step, n_px)
        out[:, a:b] = rs.apply_pinv(P, np.ascontiguousarray(B[:, a:b], dtype=np.float32))
    return out


def lstsq_columns(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``np.linalg.lstsq(A, B)[0]`` for many right-hand sides (the columns of
    *B*, one per pixel), in strips, in parallel: each pixel's solution is the
    same, without *B* turned to float64 all at once (12 GB for a 32-frame,
    46-megapixel stack)."""
    fast = _lstsq_columns_rust(A, B)
    if fast is not None:
        return fast
    n_px = B.shape[1]
    out = np.empty((A.shape[1], n_px), np.linalg.lstsq(A, B[:, :1], rcond=None)[0].dtype)

    def band(a, b):
        out[:, a:b] = np.linalg.lstsq(A, B[:, a:b], rcond=None)[0]
    by_rows(band, n_px, B.shape[0], parts=parts_for(n_px, n_px * B.shape[0] * 8 * 3))
    return out


def frames_median(frames, dtype=None) -> np.ndarray:
    """``np.median`` across frames (axis 0), band of rows by band of rows, in
    parallel: the same value for every pixel, without the whole stack copied
    (a 32-frame, 46-megapixel colour stack was copied twice, 9 GB, to take
    its median).  *frames* is a (N, H, ...) array or a list of equal arrays;
    the result has NumPy's median type, or *dtype*."""
    first = frames[0]
    h = first.shape[0]
    probe = np.median(np.stack([f[:1] for f in frames]), axis=0)
    out = np.empty(first.shape, dtype or probe.dtype)

    def band(y0, y1):
        part = frames[:, y0:y1] if isinstance(frames, np.ndarray) else np.stack([f[y0:y1] for f in frames])
        out[y0:y1] = np.median(part, axis=0)
    per_row = int(np.prod(first.shape[1:])) * len(frames)
    by_rows(band, h, per_row, parts=parts_for(h, h * per_row * first.dtype.itemsize * 3))
    return out


def frames_var(frames: np.ndarray) -> np.ndarray:
    """``np.var`` across frames (axis 0), band by band, in parallel: the same
    values, without a temporary the size of the whole stack."""
    out = np.empty(frames.shape[1:], np.var(frames[:, :1], axis=0).dtype)

    def band(y0, y1):
        out[y0:y1] = np.var(frames[:, y0:y1], axis=0)
    h = frames.shape[1]
    per_row = int(np.prod(frames.shape[2:])) * frames.shape[0]
    by_rows(band, h, per_row, parts=parts_for(h, h * per_row * 8 * 3))
    return out


def minmax(a: np.ndarray) -> tuple[float, float]:
    """(min, max) of a finite array, band by band."""
    flat = a.reshape(a.shape[0], -1)
    lo = np.empty(flat.shape[0])
    hi = np.empty(flat.shape[0])

    def band(y0, y1):
        lo[y0:y1] = flat[y0:y1].min(axis=1)
        hi[y0:y1] = flat[y0:y1].max(axis=1)
    by_rows(band, flat.shape[0], flat.shape[1])
    return float(lo.min()), float(hi.max())


class _OrderStats:
    """The k-th smallest values of a large array, found by binning, not sorting.

    Values are counted into 65,536 bins between the minimum and the maximum
    (band by band, in the array's own float type: binning only has to be
    monotonic, and float rounding is); the bin that holds rank k is then
    partitioned on its own.  The value returned is exactly the one a full
    sort would put at rank k.
    """
    NB = 65536

    def __init__(self, a: np.ndarray):
        self.rows = a.reshape(a.shape[0], -1)
        self.n = a.size
        self.ok = self.constant = False
        mn, mx = minmax(a)
        self.mn = mn
        if not (np.isfinite(mn) and np.isfinite(mx)):
            return                         # NaN or infinity: NumPy decides
        if mx <= mn:
            self.constant = True
            return
        ft = self.rows.dtype if self.rows.dtype in (np.float32, np.float64) else np.dtype(np.float64)
        lo = ft.type(mn)
        scale = ft.type((self.NB - 1) / (mx - mn))
        if not np.isfinite(scale):
            return
        nb = self.NB
        rows = self.rows
        self.idx = np.empty(rows.shape, np.int32)
        parts: list[np.ndarray] = []
        lock = threading.Lock()

        def count(y0, y1):
            t = np.subtract(rows[y0:y1], lo, dtype=ft)
            t *= scale
            ib = self.idx[y0:y1]
            ib[...] = t
            np.minimum(ib, nb - 1, out=ib)
            c = np.bincount(ib.ravel(), minlength=nb)
            with lock:
                parts.append(c)
        by_rows(count, rows.shape[0], rows.shape[1])
        self.cum = np.cumsum(np.sum(parts, axis=0))
        self.found: dict[int, object] = {}
        self.ok = True

    def kth(self, k: int):
        """The k-th smallest value (0-based), as the array's own scalar type."""
        if k in self.found:
            return self.found[k]
        b = int(np.searchsorted(self.cum, k, side="right"))
        before = int(self.cum[b - 1]) if b else 0
        rows, idx = self.rows, self.idx
        sels: list[np.ndarray] = []
        lock = threading.Lock()

        def gather(y0, y1):
            v = rows[y0:y1][idx[y0:y1] == b]
            with lock:
                sels.append(v)
        by_rows(gather, rows.shape[0], rows.shape[1])
        sel = np.concatenate(sels)
        v = np.partition(sel, k - before)[k - before]
        self.found[k] = v
        return v


def _usable(a: np.ndarray) -> bool:
    return a.size >= 1_000_000 and a.ndim >= 2 and a.flags.c_contiguous


def _rust_kth(a: np.ndarray, ks: list[int]):
    """{rank: value} for float32 *a* from helio_core_rs, or None when it is
    not built, *a* is not float32, or *a* holds a NaN/infinity (NumPy decides
    those).  One call for all the ranks: each call is a pass over the array."""
    from helio.core._native import rs
    if rs is None or not hasattr(rs, "kth_values") or a.dtype != np.float32:
        return None
    ks = sorted(set(ks))
    vals = rs.kth_values(a, ks)
    return None if vals is None else {k: np.float32(v) for k, v in zip(ks, vals)}


def quantiles(a: np.ndarray, qs) -> list[float]:
    """``np.percentile(a, qs)`` (linear), bit for bit, without sorting.

    A whole-image percentile costs about 40 ms at 4.7 million pixels; a view
    that stretches its contrast every frame cannot afford two.  The order
    statistics come from _OrderStats; the interpolation is NumPy's own
    (``_lerp``: the gap in the array's type, the weight in float64, and the
    upper value minus the remainder from halfway up).
    """
    a = np.asarray(a)
    qs = [float(q) for q in np.atleast_1d(qs)]
    if not _usable(a):
        return [float(v) for v in np.atleast_1d(np.percentile(a, qs))]
    n = a.size
    ranks = [min(int(np.floor((n - 1) * np.true_divide(q, 100))), n - 1) for q in qs]
    found = _rust_kth(a, ranks + [min(r + 1, n - 1) for r in ranks])
    if found is not None:
        kth = found.__getitem__
    else:
        st = _OrderStats(a)
        if st.constant:
            return [st.mn] * len(qs)
        if not st.ok:
            return [float(v) for v in np.atleast_1d(np.percentile(a, qs))]
        kth = st.kth
    out = []
    for q in qs:
        vi = (n - 1) * np.true_divide(q, 100)
        lo = int(np.floor(vi))
        if vi >= n - 1:
            out.append(float(kth(n - 1)))
            continue
        g = vi - lo
        a_lo, a_hi = kth(lo), kth(lo + 1)
        d = np.subtract(a_hi, a_lo)                       # in the array's type
        r = np.add(a_lo, d * g) if g < 0.5 else np.subtract(a_hi, d * (1 - g))
        out.append(float(r))
    return out


def median(a: np.ndarray) -> float:
    """``np.median(a)``, bit for bit, without sorting (see quantiles)."""
    a = np.asarray(a)
    if not _usable(a):
        return float(np.median(a))
    n = a.size
    found = _rust_kth(a, [(n - 1) // 2] if n % 2 else [n // 2 - 1, n // 2])
    if found is not None:
        kth = found.__getitem__
    else:
        st = _OrderStats(a)
        if not st.ok:
            return float(np.median(a))
        kth = st.kth
    if n % 2:
        return float(kth((n - 1) // 2))
    lo, hi = kth(n // 2 - 1), kth(n // 2)
    s = np.add(lo, hi)                                    # NumPy's mean of the pair
    return float(s.dtype.type(s / 2))


# ── the cache of light-free work ─────────────────────────────────────────

def _nbytes(v) -> int:
    if isinstance(v, np.ndarray):
        return int(v.nbytes)
    if isinstance(v, (tuple, list)):
        return sum(_nbytes(x) for x in v)
    if isinstance(v, dict):
        return sum(_nbytes(x) for x in v.values())
    return 64


def _budget() -> int:
    """Bytes the cache may hold: a sixteenth of the machine's memory, 256 MB to 2 GB."""
    env = os.environ.get("HELIO_CACHE_MB")
    if env and env.isdigit():
        return int(env) << 20
    total = 0
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if not total:
        total = 8 << 30
    return int(min(2 << 30, max(256 << 20, total // 16)))


class _Entry:
    __slots__ = ("refs", "value", "nbytes")

    def __init__(self, refs, value, nbytes):
        self.refs, self.value, self.nbytes = refs, value, nbytes

    def alive_for(self, sources) -> bool:
        return all(r() is s for r, s in zip(self.refs, sources))

    def dead(self) -> bool:
        return any(r() is None for r in self.refs)


class LightFreeCache:
    """Bounded, least-recently-used results keyed by their source arrays."""

    def __init__(self, max_bytes: int | None = None):
        self.max_bytes = _budget() if max_bytes is None else int(max_bytes)
        self._d: OrderedDict = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, tag: str, sources: tuple, params: tuple, compute):
        """The cached value for (*tag*, *sources*, *params*), computing it once.

        *sources* are the arrays the value is derived from, matched by
        identity; *params* any settings it depends on (hashable)."""
        key = (tag, tuple(id(s) for s in sources), params)
        with self._lock:
            e = self._d.get(key)
            if e is not None and e.alive_for(sources):
                self._d.move_to_end(key)
                self.hits += 1
                return e.value
        self.misses += 1
        value = compute()
        try:
            refs = tuple(weakref.ref(s) for s in sources)
        except TypeError:                  # a source that cannot be watched: do not keep
            return value
        nb = _nbytes(value)
        if nb > self.max_bytes // 2:       # a one-off giant (a full-size export): do not keep
            return value
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._d[key] = _Entry(refs, value, nb)
            self._bytes += nb
            self._evict(keep=key)
        return value

    def _evict(self, keep):
        for k in [k for k, e in self._d.items() if k != keep and e.dead()]:
            self._bytes -= self._d.pop(k).nbytes
        while self._bytes > self.max_bytes and len(self._d) > 1:
            k = next(iter(self._d))
            if k == keep:
                self._d.move_to_end(k)
                k = next(iter(self._d))
            self._bytes -= self._d.pop(k).nbytes

    def clear(self):
        with self._lock:
            self._d.clear()
            self._bytes = 0

    def stats(self) -> dict:
        with self._lock:
            return dict(entries=len(self._d), mb=round(self._bytes / 1048576, 1),
                        budget_mb=self.max_bytes >> 20, hits=self.hits, misses=self.misses)


CACHE = LightFreeCache()


def cached(tag: str, sources: tuple, params: tuple, compute):
    """``CACHE.get``: the shared cache of light-free work."""
    return CACHE.get(tag, sources, params, compute)
