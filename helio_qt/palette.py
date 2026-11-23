"""⌘K: find any command, see where it lives and why it is unavailable (UX-004)."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeySequence
from PySide6.QtWidgets import (QDialog, QFrame, QLineEdit, QListWidget, QListWidgetItem, QStyle,
                               QStyledItemDelegate, QVBoxLayout)

from helio_qt.actions import SPECS, location
from helio_qt.theme import CAPTION_PT


def search(registry, text: str) -> list[tuple[str, str]]:
    """(action id, reason) for every command matching *text*, best first.

    Every word typed must appear in the command, its location or its
    keywords; commands whose name starts with the text rank first, then
    available ones, then those whose name (not only keywords) has every
    word, shorter (more general) names before longer."""
    words = text.lower().split()
    out = []
    for s in SPECS:
        from helio_qt.action_tips import tip
        hay = f"{s.text} {location(s)} {s.keywords} {tip(s.id)}".lower()
        if all(w in hay for w in words):
            why = registry.reason(s.id)
            name = s.text.lower()
            starts = name.startswith(text.lower().strip()) if words else False
            in_name = all(w in name for w in words)
            out.append((0 if starts else 1, 1 if why else 0, 0 if in_name else 1,
                        len(name) if words else 0, name, s.id, why))
    out.sort()
    return [(sid, why) for *_, sid, why in out]


def shortcut_text(seq: str) -> str:
    return QKeySequence(seq).toString(QKeySequence.SequenceFormat.NativeText) if seq else ""


class _RowDelegate(QStyledItemDelegate):
    """Two lines per command: its name, then where it lives or why it is unavailable;
    the shortcut on the right."""

    def __init__(self, window):
        super().__init__()
        self.window = window

    def sizeHint(self, option, index):
        fm = QFontMetrics(option.font)
        return QSize(option.rect.width(), fm.height() * 2 + 18)

    def paint(self, p, option, index):
        t = self.window.tokens
        spec, why = index.data(Qt.ItemDataRole.UserRole + 1)
        r = option.rect
        p.save()
        if option.state & QStyle.StateFlag.State_Selected:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(t.raised))
            p.drawRoundedRect(r.adjusted(6, 1, -6, -1), 8, 8)
        f = QFont(option.font)
        fm = QFontMetrics(f)
        y1 = r.top() + 8 + fm.ascent()
        p.setFont(f)
        p.setPen(QColor(t.text2 if why else t.text))
        p.drawText(r.left() + 16, y1, spec.text)
        keys = shortcut_text(spec.shortcut)
        if keys:
            p.setPen(QColor(t.text3))
            p.drawText(r.right() - 16 - fm.horizontalAdvance(keys), y1, keys)
        small = QFont(f)
        small.setPointSize(CAPTION_PT)
        p.setFont(small)
        p.setPen(QColor(t.warn if why else t.text3))
        sub = QFontMetrics(small).elidedText(why or location(spec), Qt.TextElideMode.ElideRight,
                                             r.width() - 32)
        p.drawText(r.left() + 16, y1 + fm.height() + 2, sub)
        p.restore()


class CommandPalette(QDialog):
    def __init__(self, window, registry):
        super().__init__(window, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.registry = registry
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAccessibleName("Search tools")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        frame = QFrame()
        frame.setObjectName("Palette")
        outer.addWidget(frame)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 8)
        lay.setSpacing(0)
        self.query = QLineEdit()
        self.query.setPlaceholderText("Search every tool…")
        self.query.setAccessibleName("Search")
        lay.addWidget(self.query)
        self.list = QListWidget()
        self.list.setItemDelegate(_RowDelegate(window))
        self.list.setUniformItemSizes(True)
        lay.addWidget(self.list)
        self.query.textChanged.connect(self.populate)
        self.query.returnPressed.connect(self.run_current)
        self.list.itemActivated.connect(lambda _: self.run_current())
        self.resize(560, 460)
        self.populate("")

    def populate(self, text: str):
        self.list.clear()
        specs = {s.id: s for s in SPECS}
        for sid, why in search(self.registry, text)[:60]:
            it = QListWidgetItem(specs[sid].text)
            it.setData(Qt.ItemDataRole.UserRole, sid)
            it.setData(Qt.ItemDataRole.UserRole + 1, (specs[sid], why))
            from helio_qt.action_tips import tip
            what = tip(sid)
            it.setToolTip("\n\n".join(t for t in (what, f"Not available: {why}" if why
                                                     else location(specs[sid])) if t))
            self.list.addItem(it)
        # Start on the first command that can run now.
        for i in range(self.list.count()):
            if not self.list.item(i).data(Qt.ItemDataRole.UserRole + 1)[1]:
                self.list.setCurrentRow(i)
                break
        else:
            if self.list.count():
                self.list.setCurrentRow(0)

    def run_current(self):
        it = self.list.currentItem()
        if not it:
            return
        sid = it.data(Qt.ItemDataRole.UserRole)
        if self.registry.reason(sid):
            return                      # its reason is already on screen
        self.accept()
        self.registry.trigger(sid)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up) and self.query.hasFocus():
            r = self.list.currentRow() + (1 if e.key() == Qt.Key.Key_Down else -1)
            self.list.setCurrentRow(max(0, min(self.list.count() - 1, r)))
            return
        super().keyPressEvent(e)

    def open_centered(self):
        w = self.parentWidget()
        g = w.geometry()
        self.move(g.x() + (g.width() - self.width()) // 2, g.y() + 90)
        self.query.clear()
        self.populate("")
        self.show()
        self.query.setFocus()


class ShortcutsSheet(QDialog):
    """⌘/: every keyboard shortcut, grouped by menu, then the image's own keys
    and the 3D view's; type to narrow the list."""

    def __init__(self, window):
        super().__init__(window)
        self.setWindowTitle("Keyboard Shortcuts")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(8)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Type to find a shortcut: a key or what it does")
        self.filter.setClearButtonEnabled(True)
        self.filter.setAccessibleName("Find a shortcut")
        lay.addWidget(self.filter)
        lst = self.list = QListWidget()
        lst.setAccessibleName("Keyboard shortcuts")
        self.groups: list[tuple[QListWidgetItem, list[QListWidgetItem]]] = []

        def heading(text):
            head = QListWidgetItem(text)
            head.setFlags(Qt.ItemFlag.NoItemFlags)
            f = head.font()
            f.setBold(True)
            head.setFont(f)
            lst.addItem(head)
            self.groups.append((head, []))

        def row(keys, what):
            it = QListWidgetItem(f"{keys:>12}    {what}")
            lst.addItem(it)
            self.groups[-1][1].append(it)

        last = None
        for s in SPECS:
            if not s.shortcut:
                continue
            menu = location(s)
            if menu != last:
                heading(menu.upper())
                last = menu
            note = "  (when the image has focus)" if s.image_focus else ""
            row(shortcut_text(s.shortcut), s.text + note)
        from helio_qt.actions import ALIASES
        heading("ALSO, AS IN Helio Classic (WHEN THE IMAGE HAS FOCUS)")
        specs = {s.id: s for s in SPECS}
        for key, sid in ALIASES:
            row(shortcut_text(key), specs[sid].text)
        from helio_qt.relief3d import KEYS
        for group, keys in KEYS:
            heading(f"IN THE 3D VIEW: {group} (WHEN IT HAS FOCUS)")
            for k, what in keys:
                row(k, what)
        lay.addWidget(lst)
        self.filter.textChanged.connect(self.narrow)
        self.resize(520, 620)

    def narrow(self, text: str):
        """Only the rows with every word typed, under their headings."""
        words = text.lower().split()
        for head, rows in self.groups:
            hit = 0
            for it in rows:
                show = all(w in it.text().lower() or w in head.text().lower() for w in words)
                it.setHidden(not show)
                hit += show
            head.setHidden(hit == 0)
