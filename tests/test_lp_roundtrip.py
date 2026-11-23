"""CHI .lp files must be readable by a strict third-party parser.

The format is a bare light count on line 1, then one
"<filename> <x> <y> <z>" line per light. HELIO's own importer takes the last
three floats on each line, so it tolerated a leading comment that RTIBuilder
and RTIViewer do not.
"""
import os, sys, tempfile
import numpy as np

sys.path.insert(0, ".")
PASS, FAIL = [], []
def check(name, fn):
    try:
        fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name); print(f"  FAIL  {name}: {type(e).__name__}: {e}")


def strict_read_lp(path):
    """A deliberately unforgiving reader, standing in for RTIViewer."""
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    n = int(lines[0])                      # no comments tolerated
    names, vecs = [], []
    for line in lines[1:n + 1]:
        parts = line.split()
        assert len(parts) >= 4, f"expected '<name> x y z', got {line!r}"
        names.append(" ".join(parts[:-3]))
        vecs.append([float(v) for v in parts[-3:]])
    assert len(vecs) == n, f"header says {n} lights, found {len(vecs)}"
    return names, np.array(vecs, np.float32)


def write_lp(path, names, L):
    """Mirrors Helio_Pro._export_lp's on-disk format."""
    import math
    with open(path, "w") as f:
        f.write(f"{len(L)}\n")
        for nm, (lx, ly, lz) in zip(names, L):
            n = math.sqrt(lx * lx + ly * ly + lz * lz)
            f.write(f"{nm} {lx/n:.6f} {ly/n:.6f} {max(0.001, lz/n):.6f}\n")


tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)
L = rng.normal(size=(8, 3)).astype(np.float32)
L[:, 2] = np.abs(L[:, 2]) + 0.3
L /= np.linalg.norm(L, axis=1, keepdims=True)
names = [f"IMG_{i:04d}.jpg" for i in range(8)]
path = os.path.join(tmp, "lights.lp")
write_lp(path, names, L)

def _no_comment_header():
    first = open(path).readline().strip()
    int(first)                                   # raises if a comment sneaks back
    assert first == "8", f"first line should be the count, got {first!r}"
check("line 1 is a bare light count", _no_comment_header)

def _strict_parse():
    got_names, got = strict_read_lp(path)
    assert got_names == names, got_names[:2]
    assert got.shape == (8, 3), got.shape
check("a strict third-party parser reads the file", _strict_parse)

def _vectors_survive():
    _, got = strict_read_lp(path)
    assert np.allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-4)
    ang = np.degrees(np.arccos(np.clip((got * L).sum(1), -1, 1)))
    assert ang.max() < 0.05, f"worst vector moved {ang.max():.4f} deg in round-trip"
check("light vectors survive the round-trip", _vectors_survive)

def _helio_importer_agrees():
    """HELIO's own parse rule: last three floats on each line."""
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    got = []
    for line in lines[1:]:
        nums = []
        for part in line.split():
            try: nums.append(float(part))
            except ValueError: pass
        if len(nums) >= 3: got.append(nums[-3:])
    got = np.array(got, np.float32)
    assert got.shape == L.shape, got.shape
    ang = np.degrees(np.arccos(np.clip((got * L).sum(1), -1, 1)))
    assert ang.max() < 0.05
check("HELIO's own importer reads it identically", _helio_importer_agrees)

def _filenames_with_spaces():
    p2 = os.path.join(tmp, "spaced.lp")
    nm = ["MS-KK 00005 Large.jpeg"] * 3
    write_lp(p2, nm, L[:3])
    got_names, got = strict_read_lp(p2)
    assert got_names == nm, got_names
check("filenames containing spaces round-trip", _filenames_with_spaces)

print("\n" + "=" * 56)
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
print("=" * 56)
sys.exit(1 if FAIL else 0)
