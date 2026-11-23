"""Tooltips on the Mac's own menu bar.

Qt draws Helio's menus with the Mac's native menu bar, and does not hand a
QAction's tooltip to the native menu item: QMenu.setToolTipsVisible only
works for menus Qt draws itself.  So the description of every command
(helio_qt.action_tips) is set on the native NSMenuItem here, through the
Objective-C runtime with ctypes (no extra dependency), matched by the item's
path in the menu bar.  Anywhere else, or if anything about the native menu is
not as expected, this does nothing.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import sys

_log = logging.getLogger(__name__)
_objc = None


def _rt():
    global _objc
    if _objc is None:
        lib = ctypes.util.find_library("objc")
        if not lib:
            raise OSError("no Objective-C runtime")
        _objc = ctypes.cdll.LoadLibrary(lib)
        _objc.objc_getClass.restype = ctypes.c_void_p
        _objc.objc_getClass.argtypes = [ctypes.c_char_p]
        _objc.sel_registerName.restype = ctypes.c_void_p
        _objc.sel_registerName.argtypes = [ctypes.c_char_p]
    return _objc


def _send(obj, sel: str, *args, restype=ctypes.c_void_p, argtypes=()):
    rt = _rt()
    f = ctypes.cast(rt.objc_msgSend, ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes))
    return f(obj, rt.sel_registerName(sel.encode()), *args)


def _cls(name: str):
    return _rt().objc_getClass(name.encode())


def _nsstring(text: str):
    return _send(_cls("NSString"), "stringWithUTF8String:", text.encode("utf-8"),
                 argtypes=(ctypes.c_char_p,))


def _pystr(ns) -> str:
    if not ns:
        return ""
    s = _send(ns, "UTF8String", restype=ctypes.c_char_p)
    return s.decode("utf-8") if s else ""


def _walk(menu, path, out):
    n = _send(menu, "numberOfItems", restype=ctypes.c_long)
    for i in range(n):
        item = _send(menu, "itemAtIndex:", i, argtypes=(ctypes.c_long,))
        if not item:
            continue
        title = _pystr(_send(item, "title"))
        sub = _send(item, "submenu")
        if sub:
            _walk(sub, path + (title,), out)
        elif title:
            out.append((path, title, item))


def native_items():
    """[(menu path, title, NSMenuItem)] of the app's main menu, or [] off a Mac."""
    if sys.platform != "darwin":
        return []
    app = _send(_cls("NSApplication"), "sharedApplication")
    main = _send(app, "mainMenu") if app else None
    if not main:
        return []
    out = []
    _walk(main, (), out)
    return out


def set_tooltips(tip_of) -> int:
    """Give every native menu item the tooltip *tip_of(path, title)* returns
    (path: the menu titles from the menu bar down).  The number set."""
    if sys.platform != "darwin":
        return 0
    n = 0
    try:
        for path, title, item in native_items():
            t = tip_of(path, title)
            if t:
                _send(item, "setToolTip:", _nsstring(t), argtypes=(ctypes.c_void_p,))
                n += 1
    except Exception:                                     # noqa: BLE001 - a nicety, never a failure
        _log.debug("native menu tooltips not set", exc_info=True)
    return n


def tooltip_of(item_title_path) -> str:
    """The tooltip a native item has now (for the test), '' if none."""
    for path, title, item in native_items():
        if path + (title,) == tuple(item_title_path):
            return _pystr(_send(item, "toolTip"))
    return ""
