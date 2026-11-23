"""How much memory the machine has, and how much a solve will need.

Helio is used on machines from 16 GB to 96 GB.  A 100-megapixel, 8-photo
capture needs tens of GB at its peak; this module lets the solve ask first,
so that on a small machine it can choose the way through that stays inside
memory (instead of swapping for many minutes) and say so, while keeping the
full-resolution normals and colour.  See ``plan_solve``.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

_log = logging.getLogger(__name__)
GB = 1e9


def _linux_meminfo() -> tuple[int, int] | None:
    try:
        info = {}
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0]) * 1024
        total, avail = info["MemTotal"], info.get("MemAvailable", info.get("MemFree", 0))
        for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            try:                                              # a container's own limit
                lim = int(open(path, encoding="ascii").read().strip())
                if 0 < lim < total:
                    used = 0
                    for up in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
                        try:
                            used = int(open(up, encoding="ascii").read().strip())
                            break
                        except OSError:
                            pass
                    total, avail = lim, min(avail, max(0, lim - used))
                    break
            except (OSError, ValueError):
                pass
        return total, avail
    except (OSError, KeyError, ValueError):
        return None


def _macos_memory() -> tuple[int, int] | None:
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
        page = 4096
        pages = {}
        for line in out.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            elif ":" in line:
                k, _, v = line.partition(":")
                try:
                    pages[k.strip()] = int(v.strip().rstrip("."))
                except ValueError:
                    pass
        # what a new allocation can have without swapping: free, cached files, purgeable
        free = sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative",
                                               "Pages purgeable")) * page
        return total, min(total, free)
    except Exception:
        return None


def _windows_memory() -> tuple[int, int] | None:
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return int(m.ullTotalPhys), int(m.ullAvailPhys)
    except Exception:
        return None


def system_memory() -> tuple[int, int]:
    """(total, available) physical memory in bytes.

    ``HELIO_MEMORY_GB`` sets the total (to test a small machine on a big one,
    or to cap what Helio may use) and takes 80% of it as available, as a
    machine with its usual programs open has.  Available is what can be allocated without
    pushing the machine into swap; when it cannot be read it is taken as half
    the total."""
    env = os.environ.get("HELIO_MEMORY_GB", "").strip()
    try:
        if env:
            v = int(float(env) * GB)
            return v, int(v * 0.8)
    except ValueError:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        return int(vm.total), int(vm.available)
    except Exception:
        pass
    got = None
    if sys.platform.startswith("linux"):
        got = _linux_meminfo()
    elif sys.platform == "darwin":
        got = _macos_memory()
    elif sys.platform == "win32":
        got = _windows_memory()
    if got:
        return got
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = 8 << 30
    return total, total // 2


def describe_machine() -> str:
    total, avail = system_memory()
    return f"machine: {total / GB:.0f} GB, {avail / GB:.1f} GB free"


# ── What a solve needs ────────────────────────────────────────────────────
#
# Measured (Linux, default settings, 8-bit input, after the lean depth
# solvers): the process's peak resident memory above its start-up level was
#
#     6 MP  x 8 photos  1.46 GB      12 MP x 4 photos  1.95 GB
#     12 MP x 8 photos  2.47 GB      12 MP x 16 photos 3.11 GB
#     24 MP x 8 photos  4.62 GB
#
# which is 0.4 GB plus about (116 + 8 x photos) bytes per pixel.  Most of that
# is the result itself (normals, PTM and HSH coefficients, depth, maps), which
# stays in memory for Inspect.  The model below is that line with a margin; the
# options that make more full-size copies are an *assumption* (4 bytes per
# pixel per photo each), so ``HELIO_PROFILE=1`` on a real capture is the way to
# check it.

APP_BASELINE = 1.5 * GB              # Python, Qt, OpenCV and friends, before any capture
MODEL_FIXED = 0.4 * GB
MODEL_PER_PX = 116.0
MODEL_PER_PX_PER_PHOTO = 8.0
MODEL_MARGIN = 1.25
OPTION_PER_PX_PER_PHOTO = 4.0


def _extra_copies(settings) -> int:
    """How many of the options that hold another full-size stack are on."""
    if settings is None:
        return 0
    n = 0
    for name in ("flatten_illumination", "pre_clahe", "auto_align", "multi_norm"):
        n += bool(getattr(settings, name, False))
    n += float(getattr(settings, "adaptive_light", 0.0) or 0.0) > 0.0
    return n


def estimate_solve_bytes(h: int, w: int, n_photos: int, settings=None) -> int:
    """The memory a solve of *n_photos* photographs of *h* x *w* pixels is
    expected to need at its peak, in bytes (including the app's own baseline)."""
    px = float(h) * float(w)
    per_px = MODEL_PER_PX + MODEL_PER_PX_PER_PHOTO * n_photos \
        + OPTION_PER_PX_PER_PHOTO * n_photos * _extra_copies(settings)
    return int(APP_BASELINE + (MODEL_FIXED + px * per_px) * MODEL_MARGIN)


class Plan:
    """How a solve should go on this machine.

    *level* is "comfortable" (fits in free memory), "tight" (fits in the
    machine's memory if other programs are closed) or "too_big" (more than the
    machine has: it will swap).  *message* is a sentence for the user, or ""
    when there is nothing to say.  *strip_bytes* is the working-set size for
    the strip-wise stages (``helio.core.parallel.STRIP_BYTES``): smaller when
    memory is short, so the transient copies they make are smaller too.
    *suggested_scale* is the largest working size (a fraction of the
    photographs' size) that would fit in free memory, or None when full size
    does or nothing does."""

    def __init__(self, estimate, total, available, level, message, strip_bytes, suggested_scale=None):
        self.estimate, self.total, self.available = estimate, total, available
        self.level, self.message = level, message
        self.strip_bytes, self.suggested_scale = strip_bytes, suggested_scale

    def __repr__(self):
        return (f"Plan({self.level}, needs {self.estimate / GB:.1f} GB of {self.total / GB:.0f} GB "
                f"({self.available / GB:.1f} free))")


WORK_SCALES = (0.75, 0.5, 0.35, 0.25)       # the working sizes offered, largest first


def suggest_scale(h: int, w: int, n_photos: int, settings=None, available: int | None = None):
    """The largest of WORK_SCALES whose solve fits in free memory, with its estimate."""
    avail = system_memory()[1] if available is None else available
    for s in WORK_SCALES:
        est = estimate_solve_bytes(int(h * s), int(w * s), n_photos, settings)
        if est <= 0.85 * avail:
            return s, est
    return None, None


def auto_scale(h: int, w: int, n_photos: int, settings=None) -> float:
    """The working size "Automatic" picks: full size when the solve fits in free memory,
    otherwise the largest of WORK_SCALES that does (the smallest if none does)."""
    avail = system_memory()[1]
    if estimate_solve_bytes(h, w, n_photos, settings) <= 0.85 * avail:
        return 1.0
    scale, _ = suggest_scale(h, w, n_photos, settings, avail)
    return scale or WORK_SCALES[-1]


def describe_choice(h: int, w: int, n_photos: int, settings=None, scale: float = 1.0) -> str:
    """One or two sentences for the Process panel: what this capture would need at the
    chosen working size (*scale*: 1 = full size, 0 = automatic) on this machine."""
    total, avail = system_memory()
    auto = scale <= 0
    s = auto_scale(h, w, n_photos, settings) if auto else min(1.0, scale)
    need = estimate_solve_bytes(int(h * s), int(w * s), n_photos, settings)
    shape = f"{n_photos} photographs of {h * w / 1e6:.3g} MP"
    head = (f"Automatic: {s * 100:.0f}% size. " if auto and s < 1.0 else
            "Automatic: full size fits. " if auto else "")
    at = f" at {s * 100:.0f}% size" if s < 1.0 and not auto else ""
    fits = need <= 0.85 * avail
    out = (f"{head}{shape}{at} needs about {need / GB:.3g} GB; {avail / GB:.3g} GB of "
           f"{total / GB:.3g} GB is free" + (": fits." if fits else "."))
    if not fits and s >= 1.0 or (not fits and not auto):
        sug, est = suggest_scale(h, w, n_photos, settings, avail)
        if sug and sug < s:
            out += f" {sug * 100:.0f}% size would need about {est / GB:.3g} GB and fit."
    return out


def plan_solve(h: int, w: int, n_photos: int, settings=None) -> Plan:
    total, avail = system_memory()
    need = estimate_solve_bytes(h, w, n_photos, settings)
    mp = h * w / 1e6
    shape = f"{n_photos} photographs of {mp:.3g} MP"
    if need <= 0.85 * avail:
        return Plan(need, total, avail, "comfortable", "", 128 << 20)
    scale, est = suggest_scale(h, w, n_photos, settings, avail)
    offer = (f" Working at {scale * 100:.0f}% size would need about {est / GB:.3g} GB and fit "
             f"(Working size in the processing settings, or HELIO_WORK_SCALE={scale:g}); "
             f"it keeps the detail a screen or a print shows, but pixels, scales and measurements "
             f"are then in the reduced pixels." if scale else "")
    if need <= 0.9 * total:
        msg = (f"Memory: {shape} needs about {need / GB:.3g} GB and {avail / GB:.3g} GB is free "
               f"(of {total / GB:.3g} GB). Close other programs to keep it from slowing down; "
               f"at full size the result is not reduced.{offer}")
        return Plan(need, total, avail, "tight", msg, 32 << 20, scale)
    msg = (f"Memory: {shape} needs about {need / GB:.3g} GB but this machine has {total / GB:.3g} GB, "
           f"so it will be slow while it swaps. Fewer photographs or a crop would avoid that.{offer}")
    return Plan(need, total, avail, "too_big", msg, 16 << 20, scale)
