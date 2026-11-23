"""Find controls whose value nothing ever reads.

The Recalculate Depth bug had a specific signature: a dropdown offered six
choices, the user picked one, and the code path behind the button never
consulted the variable at all.  Nothing crashed and nothing looked wrong —
the setting simply did nothing.  That class of defect is invisible to the
test suite, because every test drives the function directly rather than
through the control.

This is a static audit for the same shape of problem:

  * dropdown options that no code ever compares against
  * slider keys that are declared but never read back
  * tk variables that are created and set but whose .get() is never called

Everything it prints is a *candidate*, not a verdict — a value read through
getattr(), a dict comprehension or a shared-state sweep will look unread here.
Check each one before believing it.

    python tools/wiring_audit.py
"""
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = []
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs
               if d not in {"dist", "dist_intel", "build", "build_intel",
                            "__pycache__", ".git", "tests", "tools", "help_images"}]
    for f in files:
        if f.endswith(".py"):
            SOURCES.append(os.path.join(base, f))

TEXT = {p: open(p, encoding="utf-8", errors="replace").read() for p in SOURCES}
ALL = "\n".join(TEXT.values())
# The app file alone — a value "used" only by a test is still unwired in the app.
APP = "\n".join(t for p, t in TEXT.items()
                if not os.path.basename(p).startswith(("test_", "tools_", "smoke_"))
                and os.path.relpath(p, ROOT).split(os.sep)[0] not in ("tests", "tools"))


def report(title, rows, note=""):
    print(f"\n{title}")
    print("-" * len(title))
    if note:
        print(f"  ({note})")
    if not rows:
        print("  nothing found")
    for r in rows:
        print(f"  {r}")
    return len(rows)


def dropdown_options():
    """Option strings that appear in a values=[...] list and nowhere else."""
    dead = []
    for m in re.finditer(r"CTkOptionMenu\((.*?)\)\s*$", APP, re.S | re.M):
        pass
    for m in re.finditer(r"values\s*=\s*\[(.*?)\]", APP, re.S):
        block = m.group(1)
        opts = re.findall(r"[\"']([^\"'\n]{3,60})[\"']", block)
        for o in opts:
            # how many times does this exact string appear in the app at all?
            n = APP.count(f'"{o}"') + APP.count(f"'{o}'")
            if n <= 1:
                dead.append(f'{o!r}  — appears once, only in its own menu')
    return sorted(set(dead))


def slider_keys():
    """Slider state keys that are declared but never read."""
    keys = set()
    for pat in (r"make_ui_slider\([^,]+,\s*[^,]+,\s*[\"']([\w]+)[\"']",
                r"make_3d_slider\(\s*[\"']([\w]+)[\"']"):
        keys |= set(re.findall(pat, APP))
    unread = []
    for k in sorted(keys):
        reads = (APP.count(f"shared_state['{k}']") + APP.count(f'shared_state["{k}"]')
                 + APP.count(f"settings.get('{k}'") + APP.count(f'settings.get("{k}"')
                 + APP.count(f"_ss['{k}']") + APP.count(f'_ss["{k}"]')
                 + APP.count(f"shared['{k}']") + APP.count(f'shared["{k}"]')
                 + APP.count(f"get('{k}'") + APP.count(f'get("{k}"'))
        if reads == 0:
            unread.append(f"{k!r}  — slider exists, value never read")
    return unread


def tk_variables():
    """self.<name> tk variables that are never .get()."""
    decls = re.findall(r"self\.(\w+)\s*=\s*(?:tk|ctk)\.(?:Boolean|String|Int|Double)Var\(", APP)
    unread = []
    for name in sorted(set(decls)):
        if APP.count(f"self.{name}.get()") == 0 and APP.count(f"{name}.get()") == 0:
            # may still be read via getattr(self, 'name')
            if f"'{name}'" in APP or f'"{name}"' in APP:
                unread.append(f"self.{name}  — no .get(), but the name appears as a string (probably getattr)")
            else:
                unread.append(f"self.{name}  — created and set, never read")
    return unread


def bound_but_unread():
    """The sharpest signal: a widget bound with variable=self.X, where X is
    never read.

    A control the user can actually operate, whose value goes nowhere.  This
    excludes the common false positives - an OptionMenu with command=... gets
    the value as a callback argument, and a textvariable= is display-only.
    """
    bound = set(re.findall(r"variable\s*=\s*self\.(\w+)", APP))
    rows = []
    for name in sorted(bound):
        reads = (APP.count(f"self.{name}.get()")
                 + APP.count(f"'{name}'") + APP.count(f'"{name}"'))
        if reads:
            continue
        # A widget with command=... hands the new value straight to a callback,
        # so the variable is genuinely never read and that is correct.  Seven of
        # the ten this check first reported were of exactly that shape; without
        # this filter the report cries wolf and stops being read.
        # Balancing parentheses by regex fails on real call sites such as
        # values=list(_FR_PRESETS.keys()) + ["Custom"].  Scan a window around
        # the binding instead - crude, but it does not silently miss.
        m = re.search(r"variable\s*=\s*self\." + re.escape(name) + r"\b", APP)
        if m and "command=" in APP[max(0, m.start() - 400):m.end() + 400].replace(" ", ""):
            continue
        # find the widget so the report can name the control the user sees
        m = re.search(r"CTk(CheckBox|Switch|RadioButton|OptionMenu|Slider)\("
                      r"[^)]*?text\s*=\s*[\"']([^\"']{0,40})[\"'][^)]*?"
                      r"variable\s*=\s*self\." + re.escape(name), APP, re.S)
        if not m:
            m = re.search(r"CTk(CheckBox|Switch|RadioButton|OptionMenu|Slider)\("
                          r"[^)]*?variable\s*=\s*self\." + re.escape(name)
                          + r"[^)]*?text\s*=\s*[\"']([^\"']{0,40})[\"']", APP, re.S)
        if m:
            rows.append(f"self.{name}  — {m.group(1)} labelled {m.group(2)!r}: the user can set it, nothing reads it")
        else:
            rows.append(f"self.{name}  — bound to a widget, never read")
    return rows


def duplicated_dispatch():
    """The same engine string compared in more than one place.

    Not a defect by itself, but it is how Recalculate Depth drifted from
    run_processing: two copies of one decision, only one kept up to date.
    """
    rows = []
    for m in re.finditer(r"values\s*=\s*\[(.*?)\]", APP, re.S):
        for o in re.findall(r"[\"']([^\"'\n]{3,60})[\"']", m.group(1)):
            sites = APP.count(f'== "{o}"') + APP.count(f"== '{o}'")
            if sites > 1:
                rows.append(f"{o!r}  — compared with == in {sites} places")
    return sorted(set(rows))


def main():
    print("=" * 68)
    print("Wiring audit — controls whose value nothing reads")
    print("=" * 68)
    n = 0
    n += report("Dropdown options nothing compares against", dropdown_options(),
                "a menu entry the user can pick that no branch tests for")
    n += report("Slider keys never read", slider_keys(),
                "the slider moves, the value goes nowhere")
    n += report("Tk variables never read", tk_variables(),
                "checkbox/toggle state that no code consults")
    n += report("Controls bound to a widget but never read", bound_but_unread(),
                "the sharpest signal: a control the user can operate that does nothing")
    n += report("Engine strings compared in several places", duplicated_dispatch(),
                "duplicated decisions drift — this is how Recalculate Depth broke")
    print(f"\n{n} candidates. Each needs checking by hand before it is a bug.")


if __name__ == "__main__":
    main()
