"""Recalculate Depth must reach a running 3D viewer without restarting it.

The viewer receives its arrays by value when the process is spawned, so the
only way to show a new depth map was to terminate the viewer and launch
another 400ms later. That discards the camera, flashes a window, and races the
dying process for the GL context.

The viewer already polls `shared` every 50ms and rebuilds when a value
changes, so the trigger was never missing - only a way to get the pixels
across. These checks drive that path across a real process boundary, which is
the only place it can actually fail: a buffer that works in-process and not
across a spawn would pass any single-process test and help nobody.
"""
import os
import sys
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _reader(spec, shared, out, ready):
    """Stand in for the viewer: map the buffers and report what is seen."""
    from helio.render.live_depth import attach
    arrays, handles = attach(spec)
    out["first_hp"] = float(arrays["d_hp"].mean())
    out["first_gen"] = shared["depth_gen"].value
    # Tell the parent the first read is done. Sleeping instead raced: a spawn
    # plus imports takes longer than any wait worth writing, so the parent
    # overwrote the buffer before the child had looked at it.
    ready.set()
    # wait for the parent to publish, exactly as the viewer's poll would
    for _ in range(200):
        if shared["depth_gen"].value != out["first_gen"]:
            break
        import time
        time.sleep(0.01)
    out["second_hp"] = float(arrays["d_hp"].mean())
    out["second_full"] = float(arrays["d_full"].mean())
    out["second_gen"] = shared["depth_gen"].value
    for h in handles:
        h.close()


def main():
    mp.set_start_method("spawn", force=True)
    print("=" * 62)
    print("Live depth channel")
    print("=" * 62)
    from helio.render.live_depth import DepthChannel, HAVE_SHM

    check("shared memory is available on this platform", HAVE_SHM)
    if not HAVE_SHM:
        sys.exit(1)

    shape = (64, 80)
    ch = DepthChannel({"d_full": np.zeros(shape, np.float32),
                       "d_hp": np.ones(shape, np.float32)})

    print("\nthe channel describes itself well enough to be mapped elsewhere")
    check("the spec names a buffer per array",
          set(ch.spec) == {"d_full", "d_hp"}, str(list(ch.spec)))
    check("the spec carries shape and dtype",
          all(len(v) == 3 and tuple(v[1]) == shape for v in ch.spec.values()),
          str(ch.spec))

    print("\na separate process sees what the parent writes")
    mgr = mp.Manager()
    out = mgr.dict()
    shared = {"depth_gen": mp.Value("i", 0)}
    ready = mp.Event()
    proc = mp.Process(target=_reader, args=(ch.spec, shared, out, ready))
    proc.start()
    check("the reader mapped the buffers and read them",
          ready.wait(timeout=30), "child never signalled")
    ok = ch.write({"d_full": np.full(shape, 4.0, np.float32),
                   "d_hp": np.full(shape, 9.0, np.float32)})
    check("the parent can overwrite the buffers in place", ok)
    shared["depth_gen"].value += 1
    proc.join(timeout=15)
    check("the reader process finished", not proc.is_alive())

    check("it saw the original contents first",
          abs(out.get("first_hp", -1) - 1.0) < 1e-6, str(out.get("first_hp")))
    check("it saw the new contents after the counter moved",
          abs(out.get("second_hp", -1) - 9.0) < 1e-6, str(out.get("second_hp")))
    check("both arrays updated, not just one",
          abs(out.get("second_full", -1) - 4.0) < 1e-6, str(out.get("second_full")))
    check("the generation counter reached the reader",
          out.get("second_gen", 0) == 1, str(out.get("second_gen")))

    print("\na changed shape is refused, so a crop still relaunches")
    check("a differently shaped array is rejected",
          ch.write({"d_full": np.zeros((8, 8), np.float32),
                    "d_hp": np.zeros((8, 8), np.float32)}) is False)
    check("the refusal leaves the buffers intact",
          ch.write({"d_full": np.zeros(shape, np.float32),
                    "d_hp": np.zeros(shape, np.float32)}) is True)

    print("\nthe buffers are released")
    ch.close()
    check("closing twice is harmless", (ch.close() or True))

    print("\nthe viewer accepts the argument without it")
    import inspect
    from helio.render import run_opengl_viewer
    sig = inspect.signature(run_opengl_viewer)
    check("depth_spec is optional, so older call sites still work",
          sig.parameters["depth_spec"].default is None)
    check("the viewer still takes its arrays by value as a fallback",
          "raw_depth_full" in sig.parameters and "raw_depth_hp" in sig.parameters)

    print("\n" + "=" * 62)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 62)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
