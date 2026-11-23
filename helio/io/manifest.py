"""A manifest of what an export actually wrote, with checksums.

Repositories ask two questions at deposit: what is in this folder, and has it
changed since you made it.  A folder of TIFFs answers neither.  Writing the
list and a SHA-256 for each file at export time answers both, and costs a
single pass over files that were just written and are still in the page cache.
"""
import hashlib
import os
import time

_CHUNK = 1 << 20


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def write_manifest(out_dir: str, *, object_id: str = "",
                   note: str = "", filename: str = "MANIFEST.txt") -> str:
    """Write *filename* into *out_dir* listing every other file with its hash.

    The manifest excludes itself, so re-running it is idempotent and the file
    never has to describe its own hash.  Returns the path written.
    """
    path = os.path.join(out_dir, filename)
    rows = []
    for root, _dirs, files in os.walk(out_dir):
        for f in sorted(files):
            full = os.path.join(root, f)
            if os.path.abspath(full) == os.path.abspath(path):
                continue
            rel = os.path.relpath(full, out_dir)
            try:
                rows.append((rel, os.path.getsize(full), sha256_of(full)))
            except OSError:
                rows.append((rel, -1, "unreadable"))

    total = sum(r[1] for r in rows if r[1] > 0)
    width = max([len(r[0]) for r in rows], default=4)
    lines = [
        "Helio — export manifest",
        f"Object:  {object_id or '(unnamed)'}",
        f"Written: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"Files:   {len(rows)}  ({total / 1e6:.1f} MB)",
    ]
    if note:
        lines.append(f"Note:    {note}")
    lines += ["", "SHA-256                                                          "
                  "         Size  File", "-" * (74 + width)]
    for rel, size, digest in rows:
        lines.append(f"{digest}  {size:12d}  {rel}")
    lines += ["", "Verify on macOS or Linux with:",
              "    shasum -a 256 -c <(awk 'NR>7 {print $1\"  \"$3}' MANIFEST.txt)"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def read_manifest(path: str):
    """Parse a manifest written by :func:`write_manifest`.

    Returns ``[(relative_path, size, sha256), ...]``. Tolerant of the header
    and trailer by construction: a data row is exactly a 64-hex digest, an
    integer size, and a path, so nothing else can be mistaken for one.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split(None, 2)
            if len(parts) != 3:
                continue
            digest, size, rel = parts
            if len(digest) != 64 or not all(c in "0123456789abcdef" for c in digest):
                continue
            try:
                size = int(size)
            except ValueError:
                continue
            rows.append((rel, size, digest))
    return rows


def verify_manifest(out_dir: str, filename: str = "MANIFEST.txt"):
    """Re-hash a folder and compare it against its manifest.

    ``write_manifest`` has been able to write checksums since the DEM package
    work, and nothing in the tree ever read one back. A checksum nobody checks
    is a receipt, not an integrity guarantee — the whole point of depositing
    one is that somebody can later answer "has this changed", and that
    somebody could not be us.

    Returns a dict with ``ok``, ``changed``, ``missing``, ``extra`` (lists of
    relative paths) and ``total``. ``extra`` lists files present in the folder
    but absent from the manifest, which matters as much as a changed hash: a
    deposit that has grown a file is no longer the thing that was described.
    """
    man_path = os.path.join(out_dir, filename)
    if not os.path.isfile(man_path):
        raise FileNotFoundError(
            f"no {filename} in {out_dir} — write one first with Manifest + Checksums")

    listed = read_manifest(man_path)
    if not listed:
        raise ValueError(f"{filename} contains no checksum rows")

    ok, changed, missing = [], [], []
    for rel, size, digest in listed:
        full = os.path.join(out_dir, rel)
        if not os.path.isfile(full):
            missing.append(rel)
            continue
        try:
            actual = sha256_of(full)
        except OSError:
            changed.append(rel)
            continue
        (ok if actual == digest else changed).append(rel)

    known = {r[0] for r in listed}
    extra = []
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            full = os.path.join(root, f)
            if os.path.abspath(full) == os.path.abspath(man_path):
                continue
            rel = os.path.relpath(full, out_dir)
            if rel not in known:
                extra.append(rel)

    return {"ok": ok, "changed": changed, "missing": missing,
            "extra": sorted(extra), "total": len(listed)}
