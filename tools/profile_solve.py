#!/usr/bin/env python3
"""Time and peak memory of each stage of a solve, on a synthetic capture.

    python tools/profile_solve.py                    # 8 photographs of 12 MP
    python tools/profile_solve.py 6000 4000 8        # width, height, photographs
    HELIO_MEMORY_GB=16 python tools/profile_solve.py 6000 4000 8
                                                     # plan for a 16 GB machine

To profile a REAL capture, run Helio itself with the profiler on and open the
capture as usual; the timeline prints in the terminal when the solve ends:

    HELIO_PROFILE=1 python3 -m helio_qt              (or: HELIO_PROFILE=1 ./"Run Helio.command")
    HELIO_PROFILE_FILE=~/helio_profile.txt           (also append it to a file)

Send that timeline (and the machine line at its top) along with a problem report.
"""
import os
import sys
import tempfile
import time

os.environ["HELIO_PROFILE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    from tools import make_golden as g
    from helio.application.pipeline import SolveRequest, solve
    from helio.domain.settings import ProcessingSettings
    g.H, g.W = h, w
    t = time.perf_counter()
    paths, lights = g.render_stack(tempfile.mkdtemp(prefix="helio_profile_"), n_lights=n)
    print(f"made {n} synthetic photographs of {w}x{h} ({w * h / 1e6:.0f} MP) in {time.perf_counter() - t:.0f} s")
    solve(SolveRequest(paths=paths, lights=lights, settings=ProcessingSettings(photo_tone="linear")))


if __name__ == "__main__":
    main()
