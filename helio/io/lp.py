"""Read light-position (.lp / .dome / .txt) files (GUI-free; CAL-001).

The same parse as the legacy RTIViewerPro.load_external_calibration, line for
line, so both apps get the same light matrix from the same file.  Vectors are
kept verbatim apart from normalising: +Y up, the calibrated frame
(contracts.md §3, helio.core.frames).
"""
from __future__ import annotations

import numpy as np

LP_SUFFIXES = (".lp", ".dome", ".txt")


def parse_lp(text: str) -> np.ndarray:
    """(N, 3) float32 unit light vectors from the text of a light file.

    Accepts "x y z" and RTIBuilder's "name x y z" lines.  Count headers,
    blank, zero-length and malformed lines are skipped.  Raises ValueError
    when no vector is found.
    """
    return parse_lp_named(text)[1]


def parse_lp_named(text: str) -> tuple[list, np.ndarray]:
    """(names, lights): parse_lp's vectors with the file name each line gives
    (None for a bare "x y z" line), so they can be paired with photographs."""
    parsed, names = [], []
    for line in text.splitlines():
        tokens = line.strip().split()
        if len(tokens) < 3:
            continue
        try:
            try:
                float(tokens[0])
                x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])
                name = None
            except ValueError:
                if len(tokens) < 4:
                    continue
                # "name x y z": the vector is the last three numbers, so a file
                # name with spaces in it ("frame 01.jpg") still reads.
                x, y, z = float(tokens[-3]), float(tokens[-2]), float(tokens[-1])
                name = " ".join(tokens[:-3])
            mag = np.sqrt(x * x + y * y + z * z)
            if mag < 1e-9:
                continue
            parsed.append([x / mag, y / mag, z / mag])
            names.append(name)
        except (ValueError, IndexError):
            continue
    if not parsed:
        raise ValueError("No valid light vectors found. Expected lines like "
                         "'0.123 0.456 0.789' or 'img.jpg -0.257 0.576 0.775'.")
    return names, np.array(parsed, dtype=np.float32)


def read_lp(path: str) -> np.ndarray:
    with open(path, "r") as f:
        return parse_lp(f.read())


def read_lp_named(path: str) -> tuple[list, np.ndarray]:
    with open(path, "r") as f:
        return parse_lp_named(f.read())


def pair_lights(names, lights, images) -> tuple[np.ndarray, str, str]:
    """The lights of a light file put in the order of *images*.

    By name when the file's names are the photographs' (case ignored, then
    ignoring the extension): every photograph gets its own line, wherever
    it is in the file.  A photograph the file does not name gets NaN, which
    leaves the frame out.  When the file names none of the photographs (an
    RTIBuilder 01.jpg … 50.jpg against a camera's x-3.JPG … x-52.JPG), or
    gives no names, light k goes to the k-th photograph in natural (shot)
    order: x-3 before x-10, never the alphabetical x-10 before x-3, which
    pairs most frames with another frame's light.

    Returns (lights, how, warning): how is "name" or "order"; warning is ""
    or a sentence for the user.  When pairing by order and the counts
    differ, the lights come back as they are, for the caller's count check.
    """
    import os
    from helio.io.capture import natural_key
    L = np.asarray(lights, np.float32)
    images = [str(p) for p in images]
    names = list(names) if names is not None else [None] * len(L)
    n = len(images)
    if any(nm for nm in names):
        by_base, by_stem = {}, {}
        for i, nm in enumerate(names):
            if not nm:
                continue
            base = os.path.basename(str(nm).replace("\\", "/")).lower()
            by_base.setdefault(base, i)
            by_stem.setdefault(os.path.splitext(base)[0], i)
        idx = []
        for p in images:
            base = os.path.basename(p).lower()
            j = by_base.get(base)
            idx.append(j if j is not None else by_stem.get(os.path.splitext(base)[0]))
        hit = [j for j in idx if j is not None]
        if hit:
            out = np.full((n, 3), np.nan, np.float32)
            for i, j in enumerate(idx):
                if j is not None:
                    out[i] = L[j]
            miss = [os.path.basename(images[i]) for i, j in enumerate(idx) if j is None]
            warn = ""
            if miss:
                warn = (f"The light file names {len(hit)} of the {n} photographs; "
                        f"{len(miss)} without a light are left out ("
                        + ", ".join(miss[:4]) + (" …" if len(miss) > 4 else "") + ").")
            return out, "name", warn
        first = next(nm for nm in names if nm)
        warn = (f"The names in the light file ({first} …) match none of the photographs "
                f"({os.path.basename(images[0]) if images else 'none'} …), so the lights "
                "are paired in shot order: light 1 with the lowest-numbered frame. "
                "Check that this is the order they were taken in.")
    else:
        warn = ""
    if len(L) != n:
        return L, "order", warn
    order = sorted(range(n), key=lambda i: natural_key(images[i]))
    out = np.empty_like(L)
    for k, i in enumerate(order):
        out[i] = L[k]
    return out, "order", warn


def find_light_file(images) -> tuple[str, str]:
    """The light file that goes with *images*, found beside them: in their
    folder, else the folder above (a capture's JPG/ or TIFFS/ subfolder).

    A file is a candidate when it reads and fits: its names are the
    photographs' (at least three), or it has one light per photograph.  One
    candidate in the nearer folder is the answer; several, the one whose names
    match the most photographs, then the one named like the folder or the
    photographs.  Returns (path, why): path "" when there is none, or when
    several fit equally and the choice is the user's (why says which)."""
    import os
    images = [str(p) for p in images]
    if not images:
        return "", ""
    folder = os.path.dirname(os.path.abspath(images[0]))
    n = len(images)
    stems = {os.path.splitext(os.path.basename(p))[0].lower() for p in images}
    bases = {os.path.basename(p).lower() for p in images}
    for where in (folder, os.path.dirname(folder)):
        try:
            lps = sorted(os.path.join(where, f) for f in os.listdir(where)
                         if f.lower().endswith(".lp") and not f.startswith("."))
        except OSError:
            continue
        fits = []
        for path in lps:
            try:
                names, L = read_lp_named(path)
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            hits = sum(1 for nm in names if nm and (
                os.path.basename(str(nm).replace("\\", "/")).lower() in bases
                or os.path.splitext(os.path.basename(str(nm).replace("\\", "/")))[0].lower() in stems))
            if hits >= 3 or len(L) == n:
                stem = os.path.splitext(os.path.basename(path))[0].lower()
                named_like = stem in os.path.basename(where).lower() or any(stem in s for s in stems)
                fits.append((hits, named_like, path))
        if not fits:
            continue
        fits.sort(key=lambda t: (t[0], t[1]), reverse=True)
        if len(fits) > 1 and fits[0][:2] == fits[1][:2]:
            return "", (f"{len(fits)} light files fit these photographs in "
                        f"{os.path.basename(where) or where}: choose one with File › Import Light Positions…")
        hits, _nl, path = fits[0]
        why = (f"its names are {hits} of the photographs'" if hits >= 3
               else f"{n} lights for {n} photographs")
        return path, why
    return "", ""
