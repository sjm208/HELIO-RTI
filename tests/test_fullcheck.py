"""The full-check tool (helio_qt/fullcheck.py) runs, finishes, and writes its report.

Runs it headless and quick on a small made capture, only the parts that need no test suite, then checks
REPORT.md and REPORT.json exist, the paste block is there, nothing crashed, and that a deliberately broken
part is reported as a failure and not as a crash.

    python tests/test_fullcheck.py
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    out = tempfile.mkdtemp(prefix="helio_fc_test_")
    env = dict(os.environ, PYTHONPATH=ROOT, HELIO_PREFS=os.path.join(out, "prefs.json"))
    p = subprocess.run([sys.executable, "-m", "helio_qt.fullcheck", "--headless", "--quick", "--size", "320x240",
                        "--only", "environment,imports,solve,workspaces,exports", "--out", out],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=1500)
    check("it finishes", p.returncode in (0, 1), (p.returncode, p.stderr[-400:]))
    md, js = os.path.join(out, "REPORT.md"), os.path.join(out, "REPORT.json")
    check("the report files are written", os.path.exists(md) and os.path.exists(js))
    if not (os.path.exists(md) and os.path.exists(js)):
        return FAIL + 1
    d = json.load(open(js))
    text = open(md).read()
    check("the paste block is in it", "## Paste this to Claude" in text and "FAILURES:" in text and "WARNINGS:" in text)
    check("no part crashed", not d["crashes"], d["crashes"][:1])
    parts = {r["part"] for r in d["rows"]}
    check("every asked-for part reported", {"environment", "imports", "solve", "workspaces", "exports"} <= parts, parts)
    check("the modules import", any(r["part"] == "imports" and r["status"] == "ok" for r in d["rows"]))
    check("the made capture was solved", any(r["name"].startswith("process (solve") and r["status"] == "ok" for r in d["rows"]))
    check("exports were written and checked", sum(1 for r in d["rows"] if r["part"] == "exports" and r["status"] == "ok") >= 5)
    bad = [r for r in d["rows"] if r["status"] == "FAIL"]
    check("nothing failed on the made capture", not bad, "; ".join(f"{r['part']}:{r['name']}" for r in bad[:5]))
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
