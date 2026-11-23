"""Outline icons drawn for Helio, recoloured from the theme (UX-006).

One consistent set in the style the design asks for: 24-unit grid, 1.75
stroke, round ends, no fill.  Drawn here rather than downloaded, so the app
carries no third-party artwork.  Icons always sit beside a text label; the
few icon-only buttons (zoom, fit, sidebar) carry an accessible name.
"""
from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_PATHS = {
    "sidebar": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
    "inspector": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
    "search": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
    "images": '<rect x="3" y="5" width="14" height="12" rx="2"/><path d="M7 21h12a2 2 0 0 0 2-2V9"/>'
              '<path d="M3 14l4-4 4 4 2-2 4 4"/>',
    "camera": '<path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z"/>'
              '<circle cx="12" cy="13" r="3.5"/>',
    "sphere": '<circle cx="12" cy="12" r="8"/><circle cx="9" cy="9" r="1.6"/>',
    "file": '<path d="M7 3h7l5 5v12a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/><path d="M14 3v5h5"/>',
    "dome": '<path d="M4 18a8 8 0 0 1 16 0"/><path d="M3 18h18"/><circle cx="9" cy="11" r="1.2"/>'
            '<circle cx="15" cy="9" r="1.2"/>',
    "process": '<path d="M5 12h4l2-6 2 12 2-6h4"/>',
    "layers": '<path d="M12 4l9 5-9 5-9-5z"/><path d="M3 14l9 5 9-5"/>',
    "ruler": '<path d="M4 17L17 4l3 3L7 20z"/><path d="M8 13l2 2M11 10l2 2M14 7l2 2"/>',
    "publish": '<path d="M12 15V4"/><path d="M8 8l4-4 4 4"/><path d="M5 13v6a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-6"/>',
    "zoom_in": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4M8 11h6M11 8v6"/>',
    "zoom_out": '<circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4M8 11h6"/>',
    "fit": '<path d="M4 9V5a1 1 0 0 1 1-1h4M15 4h4a1 1 0 0 1 1 1v4M20 15v4a1 1 0 0 1-1 1h-4M9 20H5a1 1 0 0 1-1-1v-4"/>',
    "check": '<path d="M5 12l5 5 9-10"/>',
    "alert": '<path d="M12 4l9 16H3z"/><path d="M12 10v4M12 17v.5"/>',
    "problem": '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/>',
    "move": '<path d="M12 3v18M3 12h18M9 6l3-3 3 3M9 18l3 3 3-3M6 9l-3 3 3 3M18 9l3 3-3 3"/>',
    "light": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5'
             'M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5"/>',
    "crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M2 6h14a2 2 0 0 1 2 2v14"/>',
    "pen": '<path d="M4 20l4-1L19 8l-3-3L5 16z"/><path d="M14 7l3 3"/>',
    "chevron": '<path d="M9 6l6 6-6 6"/>',
    "queue": '<path d="M4 6h16M4 12h16M4 18h10"/>',
}


def svg(name: str, colour: str) -> bytes:
    body = _PATHS[name]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="{colour}" stroke-width="1.75" stroke-linecap="round" '
            f'stroke-linejoin="round">{body}</svg>').encode()


def icon(name: str, colour: str, size: int = 18, dpr: float = 2.0) -> QIcon:
    """A crisp icon in *colour* at *size* logical pixels."""
    px = QPixmap(int(size * dpr), int(size * dpr))
    px.fill(Qt.GlobalColor.transparent)
    r = QSvgRenderer(QByteArray(svg(name, colour)))
    p = QPainter(px)
    r.render(p, QRectF(0, 0, px.width(), px.height()))
    p.end()
    px.setDevicePixelRatio(dpr)
    return QIcon(px)


NAMES = tuple(_PATHS)
