"""Which build this is.

`APP_VERSION` has said 2.6.0 since June and will go on saying it, so it cannot
answer the only question that matters when a bug is reported: *is the app you
are running the one with the fix in it?*  Every DMG shipped so far has been
indistinguishable in About from every other DMG, and the pair sitting in
`dist/` on 2026-08-26 differed from `HEAD` by twenty-four commits of mesh
correctness work - winding, scale, NaN handling, back-facing normals.  Nothing
in the bundle said so.

The three values below are rewritten by `build_dmg.sh` immediately before
py2app runs and restored from git immediately after, so the committed file
always reads as it does here: empty.  Empty means *not packaged* - a run from
source - and `stamp()` says so rather than inventing a commit.
"""

from __future__ import annotations

COMMIT = ""
BUILT = ""
ARCH = ""
RELEASE = ""        # Helio's name for the build, from the disk image's: "3.0 preview 16"


def stamp() -> str:
    """One line naming this build, for About and for problem reports.

    From a packaged app this is the commit recorded at build time.  From
    source it is asked of git live, because a developer wants the commit in
    front of them and not the one some earlier build froze; it falls back to a
    bare "source" when git is absent or the tree is not a repository.
    """
    if COMMIT:
        return " · ".join(p for p in (COMMIT, BUILT, ARCH) if p)

    import os
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if sha.returncode != 0:
            return "source"
        head = sha.stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        if dirty.returncode == 0 and dirty.stdout.strip():
            head += "+dirty"
        return f"source · {head}"
    except Exception:
        return "source"
