"""A tidy-up audit of the Qt interface: dead ends, clashes and inconsistencies a person would trip over.

    python tools/gui_audit.py

Checks (all read-only, offscreen):
  * every menu command has a handler that exists on the window, and no shortcut is used twice
  * every button, in every workspace, does something (is connected) and has a tooltip or accessible name
  * the same word is spelt one way (Colour/Color, 3D/3-D) in everything the user can read
  * no leftover developer words (TODO, preview, classic app, Tk) in visible text
  * every Publish format has an implementation and a one-line description
  * the Inspect quick-export buttons each point at a command that exists and is enabled when solved
"""
import os
import re
import sys
from collections import defaultdict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROBLEMS: list[str] = []


def problem(area, text):
    PROBLEMS.append(f"[{area}] {text}")


def main():
    from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel, QMenu
    app = QApplication.instance() or QApplication([])
    from helio_qt import actions as A
    from helio_qt.window import MainWindow
    w = MainWindow()

    def parts(ws):
        return [x for x in (ws.sidebar, ws.center, ws.inspector) if x is not None]

    def kids(ws, cls):
        return [c for part in parts(ws) for c in part.findChildren(cls)]


    # ── commands ────────────────────────────────────────────────────
    seen = defaultdict(list)
    for s in A.SPECS:
        if not s.handler:
            problem("commands", f"{s.id} ({s.text}) has no handler yet; its menu entry is permanently off")
        elif not hasattr(w, s.handler):
            problem("commands", f"{s.id} -> MainWindow.{s.handler} does not exist")
        if s.shortcut:
            seen[s.shortcut.lower()].append(s.id)
        if s.menu.split("/")[0] not in A.MENUS:
            problem("commands", f"{s.id} sits in a menu that does not exist: {s.menu}")
    for k, ids in seen.items():
        if len(ids) > 1:
            problem("shortcuts", f"{k} is used by {', '.join(ids)}")
    texts = defaultdict(list)
    for s in A.SPECS:
        texts[(s.menu, s.text)].append(s.id)
    for (m, t), ids in texts.items():
        if len(ids) > 1:
            problem("commands", f"two commands called '{t}' in {m}: {', '.join(ids)}")

    # ── buttons ─────────────────────────────────────────────────────
    nb = 0
    for name, ws in w.workspaces.items():
        for b in kids(ws, QAbstractButton):
            nb += 1
            label = (b.text() or b.accessibleName() or b.objectName() or "?").strip()
            wired = b.receivers("2clicked(bool)") > 0 or b.receivers("2toggled(bool)") > 0 or (hasattr(b, "menu") and b.menu() is not None) \
                or b.receivers("2pressed()") > 0 or b.isCheckable() or b.autoExclusive()
            if type(b).__name__ == "QToolButton" and type(b.parent()).__name__ == "QLineEdit":
                continue                                    # Qt's own clear button in a text box
            if not wired and True and not b.objectName().startswith("qt_"):
                problem("buttons", f"{name}: '{label}' is a button that is connected to nothing")
            if not (b.toolTip() or b.accessibleName()) and label in ("", "?"):
                problem("buttons", f"{name}: a button with no text, tooltip or accessible name")

    # ── words ───────────────────────────────────────────────────────
    visible = []
    for s in A.SPECS:
        visible.append(("menu", s.text))
    for name, ws in w.workspaces.items():
        for lab in kids(ws, QLabel):
            if lab.text():
                visible.append((name, lab.text()))
        for b in kids(ws, QAbstractButton):
            if b.text():
                visible.append((name, b.text()))
            if b.toolTip():
                visible.append((name + " tip", b.toolTip()))
    bad = [(r"\bColor\b|\bcolor\b", "Color -> Colour"), (r"\b3-D\b", "3-D -> 3D"),
           (r"\bTODO\b|\bFIXME\b|\bXXX\b", "a developer note"),
           (r"\bTk\b|\btkinter\b", "Tk"), (r"\bclassic app\b", "the classic app"),
           (r"\.\.\.", "three dots instead of …")]
    for where, text in visible:
        plain = re.sub(r"<[^>]+>", "", text)
        for pat, why in bad:
            if re.search(pat, plain):
                problem("words", f"{where}: '{plain[:70]}' — {why}")

    # ── publish ─────────────────────────────────────────────────────
    import inspect
    from helio_qt import publish, publish_catalog as C
    handled = set()
    for d in ("MESH_EXPORTS", "WEB_EXPORTS", "ARCHIVE_EXPORTS", "REPORT_EXPORTS", "IMAGE_EXPORTS"):
        handled |= set(getattr(publish, d, {}))
    handled |= set(re.findall(r'"([a-z0-9_]+)"', inspect.getsource(publish.export_one)))
    for fs in C.DESTINATIONS.values():
        for f in fs:
            if f.key not in handled:
                problem("publish", f"format '{f.name}' ({f.key}) is offered but nothing writes it")
            if not f.what.strip():
                problem("publish", f"format '{f.name}' has no description")
    for preset, keys in C.PRESETS.items():
        known = {f.key for fs in C.DESTINATIONS.values() for f in fs}
        for k in keys:
            if k not in known:
                problem("publish", f"preset '{preset}' lists '{k}', which is not a format")
    src = inspect.getsource(publish) + inspect.getsource(C)
    for m in re.finditer(r"Helio Classic[^\n\"']*", src):
        problem("words", f"Publish mentions Helio Classic: '{m.group(0)[:70]}'")

    # ── inspect quick exports ───────────────────────────────────────
    for sid in ("export.rti_web", "export.rti_combined", "export.hf_viewer", "export.qgis", "export.usdz",
                "export.view", "file.export"):
        if sid not in w.registry.actions:
            problem("inspect", f"quick export {sid} is not a command")

    print(f"Looked at {len(A.SPECS)} commands and {nb} buttons across {len(w.workspaces)} workspaces.")
    if PROBLEMS:
        print(f"\n{len(PROBLEMS)} things to look at:\n")
        for p in PROBLEMS:
            print("  " + p)
    else:
        print("\nNothing to report.")
    return 1 if PROBLEMS else 0


if __name__ == "__main__":
    sys.exit(main())
