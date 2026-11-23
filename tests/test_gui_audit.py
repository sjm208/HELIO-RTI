"""The interface hangs together: no dead menu entries, clashing shortcuts, unconnected buttons, unwritable
Publish formats, or words spelt two ways (tools/gui_audit.py); and an "i" button shows its text when clicked.

    python tests/test_gui_audit.py
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
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
    from tools import gui_audit as T
    T.PROBLEMS.clear()
    T.main()
    found = [p for p in T.PROBLEMS if "Classic" not in p]          # Helio Classic's mentions are a separate decision
    check("the audit finds nothing to fix", not found, "; ".join(found[:5]))

    from PySide6.QtWidgets import QApplication
    from helio_qt.widgets import button
    QApplication.instance() or QApplication([])
    shown = []
    from PySide6.QtWidgets import QToolTip
    real = QToolTip.showText
    QToolTip.showText = staticmethod(lambda pos, text, *a, **k: shown.append(text))
    try:
        b = button("i", "info", "What this does.")
        b.click()
        c = button("Go", "", "A tip")
        c.click()
    finally:
        QToolTip.showText = real
    check("clicking an 'i' button shows its text", shown == ["What this does."], shown)
    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
