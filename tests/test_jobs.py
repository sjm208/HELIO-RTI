"""Long work runs off the interface thread and behaves (JOB-001).

helio.application.jobs is pure Python, so these run in a second with no
window.  They hold the job manager to its contract: progress in order, cancel
keeps the last good result, a raised error becomes a structured failure, a
stale result is refused, and shutdown leaves no thread behind.

    python tests/test_jobs.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helio.application.jobs import JobManager

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def main():
    print("=" * 60)
    print("Jobs: submit, progress, cancel, failure, stale, shutdown")
    print("=" * 60)
    jm = JobManager()

    print("\nProgress and result")
    seen, done = [], []
    def work(ctx):
        for stage, f in (("Loading", 0.1), ("Solving", 0.5), (None, 0.3), ("Depth", 0.9)):
            ctx.progress(stage, f)
        return "first"
    j = jm.submit("solve", work, on_progress=lambda job, s, f, d: seen.append((s, f)),
                  on_done=lambda job: done.append(job.result))
    jm.wait(j, 5)
    fr = [f for _, f in seen]
    check("progress arrives in order and never goes backwards",
          fr == sorted(fr) and fr[-1] == 0.9, fr)
    check("stages are reported", [s for s, _ in seen if s] == ["Loading", "Solving", "Depth"])
    check("the result is delivered once", done == ["first"] and j.state == "done")
    check("it is the last good result of its kind", jm.last_result("solve") == "first")

    print("\nCancel between stages")
    gate, cancelled = threading.Event(), []
    def slow(ctx):
        ctx.progress("Solving", 0.4)
        gate.wait(5)
        ctx.check()                         # the next check point
        return "second"
    j2 = jm.submit("solve", slow, on_cancelled=lambda job: cancelled.append(job.id),
                   on_done=lambda job: done.append(job.result))
    time.sleep(0.05)
    j2.cancel()
    gate.set()
    jm.wait(j2, 5)
    check("the job stops at its next check point", j2.state == "cancelled" and cancelled == [j2.id])
    check("the previous result is kept", jm.last_result("solve") == "first" and done == ["first"])

    print("\nAn error becomes a structured failure")
    failed = []
    def broken(ctx):
        raise ValueError("no light directions")
    j3 = jm.submit("solve", broken, on_failed=lambda job: failed.append(job.failure))
    jm.wait(j3, 5)
    f = failed[0] if failed else None
    check("state, type and message", j3.state == "failed" and f is not None
          and f.kind == "ValueError" and f.message == "no light directions")
    check("the traceback is kept for the log", f is not None and "ValueError" in f.trace)
    check("the last good result survives a failure", jm.last_result("solve") == "first")

    print("\nA stale result is refused")
    gate2, stale = threading.Event(), []
    def old(ctx):
        gate2.wait(5)
        return "old"
    j4 = jm.submit("solve", old, on_stale=lambda job: stale.append(job.id),
                   on_done=lambda job: done.append(job.result))
    j5 = jm.submit("solve", lambda ctx: "new", on_done=lambda job: done.append(job.result))
    jm.wait(j5, 5)
    gate2.set()
    jm.wait(j4, 5)
    check("each submit gets a higher generation", j5.generation == j4.generation + 1)
    check("the older job finishing last is marked stale, not delivered",
          j4.state == "stale" and stale == [j4.id] and "old" not in done)
    check("the newer result stands", jm.last_result("solve") == "new")
    check("kinds are independent", jm.submit("export", lambda c: 1).generation == 1)

    print("\nShut down")
    def forever(ctx):
        while True:
            ctx.check()
            time.sleep(0.01)
    jm.submit("solve", forever)
    jm.submit("export", forever)
    time.sleep(0.05)
    clean = jm.shutdown(timeout=5)
    left = [t.name for t in threading.enumerate() if t.name.startswith("helio-job-")]
    check("shutdown cancels running jobs and no job thread is left", clean and not left, left)
    try:
        jm.submit("solve", lambda c: 1)
        refused = False
    except RuntimeError:
        refused = True
    check("nothing can be submitted after shutdown", refused)

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
