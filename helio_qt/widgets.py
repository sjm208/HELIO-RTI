"""Building blocks shared by the workspaces: the workspace switch, the image
well with its zoom controls, the filmstrip, choice cards and small labels."""
from __future__ import annotations

import os

from PySide6.QtCore import (QEvent, QObject, QPointF, QRunnable, QSize, Qt, QThreadPool,
                            Signal)
from PySide6.QtGui import (QColor, QColorSpace, QFont, QIcon, QImage, QImageReader, QPainter, QPen, QPixmap,
                           QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QFrame, QGraphicsPixmapItem,
                               QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel,
                               QListView, QListWidget, QListWidgetItem, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from helio_qt import icons

WORKSPACES = ("Capture", "Process", "Inspect", "Measure", "Publish")


ITEM_PAD = 16          # list item padding top + bottom (theme: 7px each) and border


class WrapLabel(QLabel):
    """A label that wraps and asks for the height its text needs at the width
    it has.  Qt lays a wrapped label out at its natural height, which ignores
    the extra lines when the panel is narrow, so text was cut at one and a
    half lines (Process's solver notes, the grey-card line)."""

    def __init__(self, text=""):
        super().__init__(text)
        self.setWordWrap(True)

    def _fit(self):
        if self.width() > 0:
            h = self.heightForWidth(self.width())
            if h > 0 and h != self.minimumHeight():
                self.setMinimumHeight(h)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit()

    def setText(self, text):
        super().setText(text)
        self._fit()


def label(text: str = "", role: str = "", wrap: bool = False, status: str = "") -> QLabel:
    lb = WrapLabel(text) if wrap else QLabel(text)
    if role:
        lb.setProperty("role", role)
    if status:
        lb.setProperty("status", status)
    lb.setWordWrap(wrap)
    lb.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
    return lb


def restyle(w: QWidget):
    """Re-apply the stylesheet after a dynamic property changed."""
    w.style().unpolish(w)
    w.style().polish(w)
    w.update()


def button(text: str, kind: str = "", tip: str = "") -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    if tip:
        b.setToolTip(tip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    if kind == "info":
        # An "i" that does nothing when clicked looks broken: show its text where it was clicked.
        def _show(_c=False, b=b):
            from PySide6.QtGui import QCursor
            from PySide6.QtWidgets import QToolTip
            if b.toolTip():
                QToolTip.showText(QCursor.pos(), b.toolTip(), b, b.rect(), 20000)
        b.clicked.connect(_show)
    return b


def fit_list(lw: QListWidget):
    """Size a short list to its rows, so a sidebar never scrolls or clips."""
    # Measured from the font, not sizeHintForRow: before the style sheet is
    # applied that leaves out the item padding and the list comes out short.
    from PySide6.QtGui import QFontMetrics
    lw.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    lw.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    from helio_qt.theme import BODY_PT
    f = lw.font()
    f.setPointSize(BODY_PT)
    line = QFontMetrics(f).lineSpacing()
    rows = sum(line * (lw.item(i).text().count("\n") + 1) + ITEM_PAD for i in range(lw.count()))
    lw.setFixedHeight(rows + 2 * lw.frameWidth() + 4)


def rule() -> QFrame:
    f = QFrame()
    f.setProperty("rule", "true")
    return f


def panel(object_name: str, width: int) -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    w.setObjectName(object_name)
    w.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
    w.setFixedWidth(width)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(12, 14, 12, 14)
    lay.setSpacing(8)
    return w, lay


STEP_HINTS = {
    "Capture": "Import the photographs and set where the lights were",
    "Process": "Solve the surface: normals, colour and height",
    "Inspect": "Relight and read the surface; every view and its light",
    "Measure": "Scale, distances, profiles, areas and roughness",
    "Publish": "Exports: images, 3D, web, archives and reports",
}


class WorkspaceBar(QFrame):
    """The five steps, numbered, as one segmented control (⌘1–⌘5).  A step
    that is finished shows a tick in place of its number."""
    chosen = Signal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("WorkspaceBar")
        self.setAccessibleName("Steps")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        self.group = QButtonGroup(self)
        self.buttons: dict[str, QPushButton] = {}
        for i, name in enumerate(WORKSPACES):
            b = QPushButton(f"{i + 1}   {name}")
            b.setCheckable(True)
            b.setToolTip(f"{STEP_HINTS.get(name, name)}  ⌘{i + 1}")
            b.setAccessibleName(name)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            self.group.addButton(b)
            lay.addWidget(b)
            self.buttons[name] = b
            b.clicked.connect(lambda _=False, n=name: self.chosen.emit(n))
            self._fit(b)

    @staticmethod
    def _fit(b):
        """Wide enough for the label in the chosen step's bold: it never clips when chosen."""
        from PySide6.QtGui import QFontMetrics
        f = QFont(b.font())
        f.setWeight(QFont.Weight.DemiBold)
        icon = 0 if b.icon().isNull() else 14 + 6
        b.setMinimumWidth(QFontMetrics(f).horizontalAdvance(b.text()) + 24 + icon + 4)

    def select(self, name: str):
        self.buttons[name].setChecked(True)

    def mark_done(self, done: set[str], ok_colour: str):
        for i, (name, b) in enumerate(self.buttons.items()):
            if name in done:
                b.setIcon(icons.icon("check", ok_colour, 14))
                b.setText(name)
            else:
                b.setIcon(QIcon())
                b.setText(f"{i + 1}   {name}")
            self._fit(b)


class StepHead(QPushButton):
    """A finished step, folded to one line at the top of the left column:
    its tick, its name and what it settled.  Click to go back to it."""

    def __init__(self, number: int, name: str):
        super().__init__("")
        self.setObjectName("StepHead")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.name, self.number = name, number
        self.setAccessibleName(f"{name}: go back to this step")
        self.setToolTip(f"{STEP_HINTS.get(name, name)}  ⌘{number + 1}")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 10, 6)
        lay.setSpacing(8)
        self.mark = QLabel()
        self.mark.setFixedSize(18, 18)
        lay.addWidget(self.mark, 0, Qt.AlignmentFlag.AlignTop)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = label(name, "")
        f = self.title.font()
        f.setWeight(QFont.Weight.DemiBold)
        self.title.setFont(f)
        self.summary = label("", "caption")
        self.summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        col.addWidget(self.title)
        col.addWidget(self.summary)
        lay.addLayout(col, 1)
        for w in (self.mark, self.title, self.summary):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    # Two lines and their padding.  Set here, not by setMinimumHeight: the
    # stylesheet's QPushButton min-height overrides that, and the rows were
    # squeezed to one line's height with the title cut at the top.
    def _row_h(self) -> int:
        m = self.layout().contentsMargins()
        return (m.top() + m.bottom() + self.title.sizeHint().height()
                + self.summary.sizeHint().height() + 1)

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self._row_h())

    def minimumSizeHint(self):
        return QSize(super().minimumSizeHint().width(), self._row_h())

    def set_state(self, done: bool, summary: str, colours: dict):
        icon = icons.icon("check", colours["ok"], 14) if done else icons.icon("info", colours["text3"], 14)
        self.mark.setPixmap(icon.pixmap(14, 14))
        self.summary.setText(summary)
        self.summary.setToolTip(summary)


# ── Thumbnails ────────────────────────────────────────────────────────

class _ThumbSignals(QObject):
    ready = Signal(int, str, QImage)


class _ThumbJob(QRunnable):
    def __init__(self, gen: int, path: str, box: QSize, sig: _ThumbSignals):
        super().__init__()
        self.gen, self.path, self.box, self.sig = gen, path, box, sig

    def run(self):
        from helio.io.raw_loader import embedded_preview, is_raw
        if is_raw(self.path):                # Qt cannot read RAW: the camera's own preview
            jpeg = embedded_preview(self.path)
            img = QImage.fromData(jpeg) if jpeg else QImage()
            if img.isNull():                 # Phase One files hold none: a reduced decode
                import cv2
                from helio.core.frame_read import read_frame
                bgr = read_frame(self.path, cv2.IMREAD_REDUCED_COLOR_8)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1],
                                 QImage.Format.Format_RGB888).copy()
            if not img.isNull():
                img = img.scaled(self.box, Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            self.sig.ready.emit(self.gen, self.path, img)
            return
        r = QImageReader(self.path)
        r.setAutoTransform(True)
        r.setAllocationLimit(1024)       # MB, as read_image: a 46 MP 16-bit TIFF passes Qt's 256
        size = r.size()
        if size.isValid():
            r.setScaledSize(size.scaled(self.box, Qt.AspectRatioMode.KeepAspectRatio))
        img = as_srgb(r.read())
        self.sig.ready.emit(self.gen, self.path, img)


class Thumbs(QObject):
    """Loads thumbnails off the main thread and keeps them for every view."""
    loaded = Signal(str)

    BOX = QSize(192, 256)

    def __init__(self):
        super().__init__()
        self.cache: dict[str, QPixmap] = {}
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max(2, min(4, os.cpu_count() or 2)))
        self.gen = 0
        self.sig = _ThumbSignals()
        self.sig.ready.connect(self._ready)

    def load(self, paths: list[str]):
        self.gen += 1
        self.cache.clear()
        for p in paths:
            self.pool.start(_ThumbJob(self.gen, p, self.BOX, self.sig))

    def _ready(self, gen: int, path: str, img: QImage):
        if gen != self.gen:
            return
        self.cache[path] = QPixmap.fromImage(img) if not img.isNull() else QPixmap()
        self.loaded.emit(path)

    def get(self, path: str, size: QSize, well: str) -> QIcon:
        """The thumbnail centred on a neutral tile of *size*, or the bare tile."""
        return QIcon(self.tile(path, size, well))

    def tile(self, path: str, size: QSize, well: str) -> QPixmap:
        """The whole frame, fitted inside a neutral tile of *size* (2x for Retina)."""
        tile = QPixmap(size * 2)
        tile.setDevicePixelRatio(2.0)
        tile.fill(QColor(well))
        px = self.cache.get(path)
        if px is not None and not px.isNull():
            s = px.scaled(size * 2, Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
            p = QPainter(tile)
            p.drawPixmap((tile.width() - s.width()) // 2, (tile.height() - s.height()) // 2, s)
            p.end()
        return tile

    def shutdown(self):
        self.gen += 1
        self.pool.clear()
        self.pool.waitForDone(3000)


class FrameList(QListWidget):
    """Thumbnails of the frames: a one-row filmstrip, or a wrapping grid."""

    def __init__(self, thumbs: Thumbs, tile: QSize, grid: bool = False, name: str = "Filmstrip"):
        super().__init__()
        self.thumbs, self.tile, self.grid_mode = thumbs, tile, grid
        self.setObjectName(name)
        self.setAccessibleName("Frames")
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setMovement(QListView.Movement.Static)
        self.setResizeMode(QListView.ResizeMode.Adjust)
        self.setIconSize(tile)
        self.setSpacing(6)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setWrapping(grid)
        self.setFlow(QListView.Flow.LeftToRight)
        if grid:
            self.setWordWrap(True)
            self.setGridSize(QSize(tile.width() + 56, tile.height() + 58))
        else:
            self.setFixedHeight(tile.height() + 54)     # icon, caption and the scroll bar
            self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.paths: list[str] = []
        self.lights: list = []
        self.well = "#000000"
        thumbs.loaded.connect(self._refresh_one)

    def set_frames(self, paths: list[str], captions: list[str] | None = None,
                   lights=None, tips: list[str] | None = None):
        """*lights*: one (lx, ly, lz) per frame in the screen frame (+Y down),
        or None: each tile then carries a small compass with its light."""
        self.paths = list(paths)
        self.lights = list(lights) if lights is not None else []
        self.clear()
        for i, p in enumerate(paths):
            it = QListWidgetItem(self._icon(i), captions[i] if captions else str(i + 1))
            it.setToolTip(tips[i] if tips and i < len(tips) else os.path.basename(p))
            it.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.addItem(it)

    def set_lights(self, lights, tips: list[str] | None = None):
        self.lights = list(lights) if lights is not None else []
        for i in range(min(self.count(), len(self.paths))):
            self.item(i).setIcon(self._icon(i))
            if tips and i < len(tips):
                self.item(i).setToolTip(tips[i])

    def _icon(self, i: int) -> QIcon:
        px = self.thumbs.tile(self.paths[i], self.tile, self.well)
        lv = self.lights[i] if i < len(self.lights) else None
        if lv is not None:
            import math
            lx, ly, lz = (float(c) for c in lv)
            if all(math.isfinite(c) for c in (lx, ly, lz)):
                # The light seen from above, as on the dome: north up, the dot
                # nearer the rim the lower the light.
                p = QPainter(px)
                p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                r, c = 11.0, QPointF(15.0, 15.0)
                p.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
                p.setBrush(QColor(0, 0, 0, 150))
                p.drawEllipse(c, r, r)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#E9B44C"))
                p.drawEllipse(c + QPointF(lx, ly) * r, 3.4, 3.4)
                p.end()
        return QIcon(px)

    def set_captions(self, captions: list[str]):
        for i, c in enumerate(captions[: self.count()]):
            self.item(i).setText(c)

    def _refresh_one(self, path: str):
        for i, p in enumerate(self.paths):
            if p == path and i < self.count():
                self.item(i).setIcon(self._icon(i))


# ── Image well ───────────────────────────────────────────────────────

def qimage_from_array(a) -> QImage:
    """A QImage that owns a copy of a uint8 (h, w), (h, w, 3) or (h, w, 4) array."""
    import numpy as np
    a = np.ascontiguousarray(a)
    h, w = a.shape[:2]
    if a.ndim == 2:
        fmt, bpl = QImage.Format.Format_Grayscale8, w
    elif a.shape[2] == 3:
        fmt, bpl = QImage.Format.Format_RGB888, 3 * w
    else:
        fmt, bpl = QImage.Format.Format_RGBA8888, 4 * w
    return QImage(a.data, w, h, bpl, fmt).copy()


def pixmap_from_array(a) -> QPixmap:
    return QPixmap.fromImage(qimage_from_array(a))


_raw_shown: dict[str, QImage] = {}         # the last RAW frames shown: each takes seconds to decode


def as_srgb(img: QImage) -> QImage:
    """*img* in sRGB by its own embedded profile, as helio.io.colour_space.to_srgb
    brings every frame the solve and the views use.  QPainter draws a QImage's
    numbers as they are, so an Adobe RGB photograph drawn unconverted looks
    paler and yellower than the same frame in Inspect (on a leather binding,
    C* 12.7 against 16.8, hue 65 against 60 degrees)."""
    if img.isNull():
        return img
    cs = img.colorSpace()
    if not cs.isValid() or cs == QColorSpace(QColorSpace.NamedColorSpace.SRgb):
        return img
    try:
        return img.convertedToColorSpace(QColorSpace(QColorSpace.NamedColorSpace.SRgb))
    except Exception:                    # a profile Qt cannot use: as the file has it
        return img


def read_image(path: str) -> QImage:
    from helio.io.raw_loader import is_raw
    if is_raw(path):                        # at the frame's full size: what is drawn on it is in its pixels
        if path not in _raw_shown:
            import cv2
            from helio.core.frame_read import read_frame
            bgr = read_frame(path, cv2.IMREAD_COLOR)
            if bgr is None:
                return QImage()
            h, w = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if len(_raw_shown) >= 2:
                _raw_shown.pop(next(iter(_raw_shown)))
            _raw_shown[path] = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        return _raw_shown[path]
    r = QImageReader(path)
    r.setAutoTransform(True)
    r.setAllocationLimit(1024)           # MB: large captures are normal here
    return as_srgb(r.read())


class Viewport(QGraphicsView):
    """The image on a neutral well: wheel or pinch to zoom under the pointer,
    drag to pan, Fit to see it whole.  Zoom controls float bottom right.

    With the line tool on (``set_tool("line")``) a drag draws a line instead
    and ``lineDrawn`` gives its ends in full-resolution image pixels."""
    zoomed = Signal(float)
    lineDrawn = Signal(float, float, float, float)
    rectDrawn = Signal(float, float, float, float)     # the region tool: x1, y1, x2, y2
    escaped = Signal()                     # Esc with a drawing tool on: the drawing is given up
    selected = Signal(float, float, float, float)      # a selection drawn (Shift-drag or Select)
    probed = Signal(float, float)          # image pixels under the pointer; (-1, -1) on leaving
    clicked = Signal(float, float)         # a click without a drag, image pixels
    doubleClicked = Signal()
    relight = Signal(float, float, bool)   # Option-drag: (dx, dy) from the centre, unit radius; dragging
    wipeMoved = Signal(float)              # the split-view divider, 0..1 across the image
    viewChanged = Signal()                 # panned, zoomed, resized or fitted: what is visible moved

    def __init__(self, empty_text: str = "No image"):
        super().__init__()
        self.setObjectName("ImageWell")
        self.setAccessibleName("Image")
        self.setScene(QGraphicsScene(self))
        self.item = QGraphicsPixmapItem()
        self.item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self.item)
        # The part being looked at, drawn again at full resolution over the preview
        # (show_detail); above the picture, below the split-view overlay.
        self.detail = QGraphicsPixmapItem()
        self.detail.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.detail.setZValue(1)
        self.detail.hide()
        self.scene().addItem(self.detail)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._fitted = True
        self.path = ""
        self.tool = ""
        self._line = None
        self._line_from = None
        self.line_colour = QColor("#E9B44C")
        self.rotation = 0                   # display only (the classic canvas rotation)
        self._overlays: list[tuple[QWidget, str]] = []
        self._press = None
        self._relighting = False
        self.annotator = None               # helio_qt.annotate.Annotator, for the "annotate" tool
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        # Split view: the second view, clipped to the right of the divider.
        from PySide6.QtWidgets import QGraphicsLineItem, QGraphicsRectItem
        self._clip = QGraphicsRectItem()
        self._clip.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self._clip.setPen(Qt.PenStyle.NoPen)
        self._clip.setZValue(5)
        self.scene().addItem(self._clip)
        self.item_b = QGraphicsPixmapItem(self._clip)
        self.item_b.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._divider = QGraphicsLineItem()
        from PySide6.QtGui import QPen as _QPen
        pen = _QPen(QColor("#FFFFFF"), 2)
        pen.setCosmetic(True)
        self._divider.setPen(pen)
        self._divider.setZValue(6)
        self.scene().addItem(self._divider)
        self.horizontalScrollBar().valueChanged.connect(lambda _v: self.viewChanged.emit())
        self.verticalScrollBar().valueChanged.connect(lambda _v: self.viewChanged.emit())
        self.wipe = 0.5
        self._wiping = False
        self._clip.hide()
        self._divider.hide()

        self.empty = label(empty_text, "secondary")
        self.empty.setParent(self)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.hud = QFrame(self)
        self.hud.setProperty("hud", "true")
        h = QHBoxLayout(self.hud)
        h.setContentsMargins(4, 3, 4, 3)
        h.setSpacing(2)
        self.btn_out = button("", "icon")
        self.btn_out.setAccessibleName("Zoom out")
        self.btn_out.setToolTip("Zoom out  ⌘−")
        self.pct = label("100%", "secondary")
        self.pct.setMinimumWidth(48)
        self.pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_in = button("", "icon")
        self.btn_in.setAccessibleName("Zoom in")
        self.btn_in.setToolTip("Zoom in  ⌘=")
        self.btn_fit = button("Fit", "outline", "Zoom to fit  ⌘0")
        for w in (self.btn_out, self.pct, self.btn_in, self.btn_fit):
            h.addWidget(w)
        self.btn_out.clicked.connect(lambda: self.zoom_by(1 / 1.25))
        self.btn_in.clicked.connect(lambda: self.zoom_by(1.25))
        self.btn_fit.clicked.connect(self.fit)
        self.hud.hide()
        # What surrounds the image: the theme's neutral well unless chosen
        # (black, dark grey, mid-grey, white), and colour assessment.
        self.surround: str | None = None
        self.assessing = False
        self._well = "#2B2B2B"
        from PySide6.QtWidgets import QGraphicsRectItem as _GR
        self._frame = _GR()
        self._frame.setBrush(QColor("#FFFFFF"))
        self._frame.setPen(Qt.PenStyle.NoPen)
        self._frame.setZValue(-1)
        self._frame.hide()
        self.scene().addItem(self._frame)

    # ISO 12646 viewing: a mid-grey surround (L* 50, sRGB 119) as the tone
    # reference and a thin white border as the white reference, the image
    # held well inside it (darktable's colour assessment mode follows the
    # same standard).  A dark surround makes an image look brighter, less
    # colourful and flatter than it is (the Hunt and Bartleson-Breneman
    # effects), so pictures judged on black are often finished too dark.
    ASSESS_GREY = "#777777"

    def recolour(self, t):
        self._well = t.image_well
        self._apply_surround()
        self.btn_out.setIcon(icons.icon("zoom_out", t.text))
        self.btn_in.setIcon(icons.icon("zoom_in", t.text))

    def set_surround(self, colour: str | None):
        """The colour around the image: None follows the theme's neutral well."""
        self.surround = colour
        self._apply_surround()

    def set_assessment(self, on: bool):
        """Colour assessment (ISO 12646): mid-grey surround, white border, image held inside."""
        self.assessing = bool(on)
        self._apply_surround()
        self.fit()

    def _apply_surround(self):
        bg = self.ASSESS_GREY if self.assessing else (self.surround or self._well)
        self.setBackgroundBrush(QColor(bg))
        self._place_frame()

    def _place_frame(self):
        if not self.assessing or self.item.pixmap().isNull():
            self._frame.hide()
            return
        r = self.item.sceneBoundingRect()
        b = max(r.width(), r.height()) * 0.012
        self._frame.setRect(r.adjusted(-b, -b, b, b))
        self._frame.show()

    def set_image(self, path: str) -> bool:
        # Only a real path is kept as it is: a rendered view leaves the path empty,
        # so "" must always clear (it kept the last capture's view after New Project).
        if path and path == self.path and not self.item.pixmap().isNull():
            return True
        self.path = path
        self.clear_detail()
        img = read_image(path) if path else QImage()
        ok = not img.isNull()
        self.item.setPixmap(QPixmap.fromImage(img) if ok else QPixmap())
        self.item.setTransform(QTransform())
        self.item.setScale(1.0)
        self.empty.setVisible(not ok)
        if path and not ok:
            self.empty.setText("This file cannot be previewed here")
        self.hud.setVisible(ok)
        self._scene_to(self.item.sceneBoundingRect())
        return ok

    def _scene_to(self, rect):
        """The scene becomes *rect*.  The zoom and position stay as they are
        unless the picture is a different size (another capture) or the
        view is fitted: a photo, another view or another layer of the same
        capture keeps the place being looked at."""
        old = self.scene().sceneRect()
        if rect == old and not self._fitted:
            return
        same = (old.width() > 0 and abs(old.width() - rect.width()) <= 1.0
                and abs(old.height() - rect.height()) <= 1.0)
        self.scene().setSceneRect(rect)
        if self._fitted or not same:
            self.fit()
        else:
            self._place_frame()

    def show_array(self, arr, full_size: tuple[int, int] | None = None):
        """Show a rendered view (uint8 RGB).  *full_size* (w, h) is the photo's
        size: a preview-resolution render is scaled up to it, so switching
        between a photo and a view keeps the zoom and position."""
        px = pixmap_from_array(arr)
        self.path = ""
        self.item.setPixmap(px)
        self.item.setScale(1.0)
        # Scaled to exactly the photo's size on both axes: a reduced frame
        # while the light moves, or a rounding in the preview's size, then
        # covers the same scene and leaves the zoom alone.
        if full_size and px.width() and px.height():
            self.item.setTransform(QTransform.fromScale(full_size[0] / px.width(),
                                                        full_size[1] / px.height()))
        else:
            self.item.setTransform(QTransform())
        rect = self.item.sceneBoundingRect()
        if rect != self.scene().sceneRect():
            self._scene_to(rect)
        self._place_wipe()
        self.empty.hide()
        self.hud.show()

    # ── split view ──────────────────────────────────────────────────────
    def set_compare(self, arr=None):
        """Show *arr* (uint8 RGB, the preview's size) right of the divider; None hides it."""
        if arr is None:
            self._clip.hide()
            self._divider.hide()
            return
        self.clear_detail()                  # the comparison is drawn at preview size
        px = pixmap_from_array(arr)
        self.item_b.setPixmap(px)
        r = self.item.sceneBoundingRect()
        self.item_b.setTransform(QTransform.fromScale(r.width() / max(px.width(), 1),
                                                      r.height() / max(px.height(), 1)))
        self.item_b.setPos(r.topLeft())
        self._clip.show()
        self._divider.show()
        self._place_wipe()

    def comparing(self) -> bool:
        return self._clip.isVisible()

    def set_wipe(self, f: float):
        self.wipe = max(0.02, min(0.98, float(f)))
        self._place_wipe()

    def _place_wipe(self):
        from PySide6.QtCore import QRectF as _R
        r = self.item.sceneBoundingRect()
        x = r.left() + r.width() * self.wipe
        self._clip.setRect(_R(x, r.top(), r.right() - x, r.height()))
        self._divider.setLine(x, r.top(), x, r.bottom())

    def _near_divider(self, pos) -> bool:
        if not self.comparing():
            return False
        line = self._divider.line()
        a = self.mapFromScene(line.p1())
        return abs(pos.x() - a.x()) <= 14

    # ── rotation (display only) ────────────────────────────────────────
    rotated = Signal(float)              # a two-finger turn on the trackpad: the new angle

    def set_rotation(self, deg: float, keep_view: bool = False):
        """Turn the view to *deg* (any angle).  Refits, unless *keep_view* and the picture is
        zoomed in, when the zoom and the point at the centre stay where they are."""
        new = float(deg) % 360.0
        if not keep_view or self._fitted or self.item.pixmap().isNull():
            self.rotation = new
            self.fit()
            return
        centre = self.mapToScene(self.viewport().rect().center())
        self.rotate(new - self.rotation)
        self.rotation = new
        self.centerOn(centre)
        self._zoom_changed()

    def set_tool(self, tool: str):
        """'' to pan, 'line' to draw a measuring line."""
        self.tool = tool
        self.setDragMode(QGraphicsView.DragMode.NoDrag if tool
                         else QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if tool else Qt.CursorShape.OpenHandCursor)

    def show_line(self, x1, y1, x2, y2):
        """Draw (or move) the measuring line, in image pixels."""
        from PySide6.QtGui import QPen
        from PySide6.QtWidgets import QGraphicsLineItem
        if self._line is None:
            self._line = QGraphicsLineItem()
            pen = QPen(self.line_colour, 2)
            pen.setCosmetic(True)                  # 2 screen pixels at any zoom
            self._line.setPen(pen)
            self._line.setZValue(10)
            self.scene().addItem(self._line)
        self._line.setLine(x1, y1, x2, y2)
        self._line.show()

    def clear_line(self):
        if self._line is not None:
            self._line.hide()

    def _relight_from(self, pos, dragging):
        """The classic click_to_relight: the pointer's offset from the view's
        centre, over half its shorter side, is the light's dome position."""
        w, h = self.viewport().width(), self.viewport().height()
        r = min(w, h) / 2.0
        dx, dy = (pos.x() - w / 2.0) / r, (pos.y() - h / 2.0) / r
        self.relight.emit(dx, dy, dragging)

    def show_rect(self, x1, y1, x2, y2):
        from PySide6.QtCore import QRectF as _R
        from PySide6.QtGui import QPen
        from PySide6.QtWidgets import QGraphicsRectItem
        if getattr(self, "_rect", None) is None:
            self._rect = QGraphicsRectItem()
            pen = QPen(self.line_colour, 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._rect.setPen(pen)
            self._rect.setZValue(10)
            self.scene().addItem(self._rect)
        self._rect.setRect(_R(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
        self._rect.show()

    def clear_rect(self):
        if getattr(self, "_rect", None) is not None:
            self._rect.hide()

    # ── the selection: Shift-drag, or drag with Select on (the classic crop) ──
    select_mode = False
    can_select = False                     # only Inspect's picture has a selection

    def show_selection(self, x1, y1, x2, y2):
        from PySide6.QtCore import QRectF as _R
        from PySide6.QtGui import QColor, QPen
        from PySide6.QtWidgets import QGraphicsRectItem
        if getattr(self, "_sel", None) is None:
            self._sel = QGraphicsRectItem()
            pen = QPen(QColor("#3FC7E8"), 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._sel.setPen(pen)
            self._sel.setZValue(11)
            self.scene().addItem(self._sel)
        self._sel.setRect(_R(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))
        self._sel.show()

    def clear_selection(self):
        if getattr(self, "_sel", None) is not None:
            self._sel.hide()

    def show_section(self, ax, ay, bx, by):
        """The 3D relief's section line A–B, in image pixels, labelled at its ends."""
        from PySide6.QtGui import QBrush, QFont, QPen
        from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsLineItem,
                                       QGraphicsSimpleTextItem)
        if getattr(self, "_secline", None) is None:
            col = QColor("#F2C14E")
            self._secline = QGraphicsLineItem()
            pen = QPen(col, 2)
            pen.setCosmetic(True)
            self._secline.setPen(pen)
            self._secline.setZValue(12)
            self.scene().addItem(self._secline)
            f = QFont()
            f.setBold(True)
            f.setPointSize(13)
            self._secends = []
            for t in self.section_letters:
                it = QGraphicsSimpleTextItem(t)
                it.setFont(f)
                it.setBrush(QBrush(col))
                it.setPen(QPen(QColor(0, 0, 0, 200), 0.6))
                it.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
                it.setZValue(13)
                self.scene().addItem(it)
                self._secends.append(it)
            self._secdot = QGraphicsEllipseItem(-5, -5, 10, 10)
            self._secdot.setBrush(QBrush(col))
            self._secdot.setPen(QPen(QColor(0, 0, 0, 200), 1))
            self._secdot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            self._secdot.setZValue(13)
            self.scene().addItem(self._secdot)
        self._secline.setLine(ax, ay, bx, by)
        self._secline.show()
        for it, (x, y) in zip(self._secends, ((ax, ay), (bx, by))):
            it.setPos(x, y)
            it.show()
        self._secdot.hide()
        self.section_ends = ((ax, ay), (bx, by))

    def show_kept_sections(self, items):
        """The sections already on the sheet, faint, each with its letters
        (ImageJ's Show All), so a new line is not drawn over one already kept.
        *items*: (ax, ay, bx, by, 'A–B', strong) in image pixels; strong is the
        one chosen in Publish's list, drawn bold."""
        from PySide6.QtGui import QBrush, QFont, QPen
        from PySide6.QtWidgets import QGraphicsItem, QGraphicsLineItem, QGraphicsSimpleTextItem
        for it in getattr(self, "_kept_items", []):
            self.scene().removeItem(it)
        self._kept_items = []
        f = QFont()
        f.setBold(True)
        f.setPointSize(11)
        for ax, ay, bx, by, name, strong in items:
            col = QColor("#7FB2FF") if not strong else QColor("#FFFFFF")
            col.setAlpha(255 if strong else 150)
            ln = QGraphicsLineItem(ax, ay, bx, by)
            pen = QPen(col, 3 if strong else 1.5, Qt.PenStyle.SolidLine if strong else Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            ln.setPen(pen)
            ln.setZValue(11.5)
            self.scene().addItem(ln)
            tx = QGraphicsSimpleTextItem(name)
            tx.setFont(f)
            tx.setBrush(QBrush(col))
            tx.setPen(QPen(QColor(0, 0, 0, 180), 0.5))
            tx.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
            tx.setPos((ax + bx) / 2, (ay + by) / 2)
            tx.setZValue(11.6)
            self.scene().addItem(tx)
            self._kept_items += [ln, tx]

    section_letters = ("A", "B")

    def set_section_letters(self, letters):
        """What the ends of the section line on the picture are called."""
        self.section_letters = tuple(letters)
        for it, t in zip(getattr(self, "_secends", []), self.section_letters):
            it.setText(t)

    def set_section_dot(self, t):
        """The chart's pointer on the line: *t* 0..1 from A, None to hide."""
        ends = getattr(self, "section_ends", None)
        if getattr(self, "_secdot", None) is None or not ends or not self._secline.isVisible():
            return
        if t is None or t < 0:
            self._secdot.hide()
            return
        (ax, ay), (bx, by) = ends
        self._secdot.setPos(ax + (bx - ax) * t, ay + (by - ay) * t)
        self._secdot.show()

    def clear_section(self):
        self.section_ends = None
        if getattr(self, "_secline", None) is not None:
            for it in (self._secline, self._secdot, *self._secends):
                it.hide()

    def _selecting(self, e) -> bool:
        return (self.can_select and not self.tool and e.button() == Qt.MouseButton.LeftButton
                and not self.item.pixmap().isNull()
                and (self.select_mode or bool(e.modifiers() & Qt.KeyboardModifier.ShiftModifier)))

    def _annotating(self, e) -> bool:
        return self.tool == "annotate" and self.annotator is not None and \
            not self.item.pixmap().isNull()

    def mousePressEvent(self, e):
        if self._annotating(e) and e.button() == Qt.MouseButton.LeftButton:
            pt = self.mapToScene(e.position().toPoint())
            self.annotator.press(pt.x(), pt.y(), e)
            return
        if self.tool in ("line", "rect") and e.button() == Qt.MouseButton.LeftButton:
            pt = self.mapToScene(e.position().toPoint())
            self._line_from = (pt.x(), pt.y())
            (self.show_line if self.tool == "line" else self.show_rect)(pt.x(), pt.y(), pt.x(), pt.y())
            return
        if self._selecting(e):
            pt = self.mapToScene(e.position().toPoint())
            self._sel_from = (pt.x(), pt.y())
            self.show_selection(pt.x(), pt.y(), pt.x(), pt.y())
            return
        if e.button() == Qt.MouseButton.LeftButton and \
                e.modifiers() & Qt.KeyboardModifier.AltModifier and not self.item.pixmap().isNull():
            self._relighting = True
            self._relight_from(e.position(), True)
            return
        if e.button() == Qt.MouseButton.LeftButton and self._near_divider(e.position()):
            self._wiping = True
            return
        if e.button() == Qt.MouseButton.MiddleButton:          # the classic middle-button pan
            self._mid_pan = e.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        self._press = e.position()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if getattr(self, "_mid_pan", None) is not None:
            d = e.position() - self._mid_pan
            self._mid_pan = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        if self._annotating(e):
            pt = self.mapToScene(e.position().toPoint())
            self.annotator.move(pt.x(), pt.y(), bool(e.buttons() & Qt.MouseButton.LeftButton))
            return
        if self.tool in ("line", "rect") and self._line_from is not None:
            pt = self.mapToScene(e.position().toPoint())
            (self.show_line if self.tool == "line" else self.show_rect)(*self._line_from, pt.x(), pt.y())
            return
        if getattr(self, "_sel_from", None) is not None:
            pt = self.mapToScene(e.position().toPoint())
            r = self.sceneRect()
            self.show_selection(*self._sel_from, max(r.left(), min(r.right(), pt.x())),
                                max(r.top(), min(r.bottom(), pt.y())))
            return
        if self._relighting:
            self._relight_from(e.position(), True)
            return
        if self._wiping:
            r = self.item.sceneBoundingRect()
            pt = self.mapToScene(e.position().toPoint())
            self.set_wipe((pt.x() - r.left()) / max(r.width(), 1.0))
            self.wipeMoved.emit(self.wipe)
            return
        if self.comparing():
            self.viewport().setCursor(Qt.CursorShape.SplitHCursor if self._near_divider(e.position())
                                      else (Qt.CursorShape.CrossCursor if self.tool
                                            else Qt.CursorShape.OpenHandCursor))
        if not self.item.pixmap().isNull():
            pt = self.mapToScene(e.position().toPoint())
            if self.item.sceneBoundingRect().contains(pt):
                self.probed.emit(pt.x(), pt.y())
            else:
                self.probed.emit(-1.0, -1.0)
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if getattr(self, "_mid_pan", None) is not None and e.button() == Qt.MouseButton.MiddleButton:
            self._mid_pan = None
            self.viewport().unsetCursor()
            return
        if self._annotating(e) and e.button() == Qt.MouseButton.LeftButton:
            pt = self.mapToScene(e.position().toPoint())
            self.annotator.release(pt.x(), pt.y())
            return
        if getattr(self, "_sel_from", None) is not None and e.button() == Qt.MouseButton.LeftButton:
            pt = self.mapToScene(e.position().toPoint())
            r = self.sceneRect()
            x1, y1 = self._sel_from
            x2, y2 = max(r.left(), min(r.right(), pt.x())), max(r.top(), min(r.bottom(), pt.y()))
            self._sel_from = None
            self.selected.emit(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            return
        if self.tool in ("line", "rect") and self._line_from is not None:
            pt = self.mapToScene(e.position().toPoint())
            r = self.sceneRect()
            clamp = lambda v, lo, hi: max(lo, min(hi, v))     # noqa: E731
            x1, y1 = self._line_from
            x2, y2 = clamp(pt.x(), r.left(), r.right()), clamp(pt.y(), r.top(), r.bottom())
            self._line_from = None
            if self.tool == "line":
                self.show_line(x1, y1, x2, y2)
                self.lineDrawn.emit(x1, y1, x2, y2)
            else:
                self.show_rect(x1, y1, x2, y2)
                self.rectDrawn.emit(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
            return
        if self._relighting:
            self._relighting = False
            self._relight_from(e.position(), False)
            return
        if self._wiping:
            self._wiping = False
            self.wipeMoved.emit(self.wipe)
            return
        press, self._press = self._press, None
        super().mouseReleaseEvent(e)
        if press is not None and e.button() == Qt.MouseButton.LeftButton and \
                (e.position() - press).manhattanLength() < 4 and not self.item.pixmap().isNull():
            pt = self.mapToScene(e.position().toPoint())
            if self.item.sceneBoundingRect().contains(pt):
                self.clicked.emit(pt.x(), pt.y())

    def mouseDoubleClickEvent(self, e):
        if self._annotating(e):
            self.annotator.double_click()
            return
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(e)

    def leaveEvent(self, e):
        self.probed.emit(-1.0, -1.0)
        super().leaveEvent(e)

    def cancel_tool(self) -> bool:
        """Esc: give up the drawing under way.  False when no drawing tool is on."""
        if self.tool == "annotate" and self.annotator is not None:
            self.annotator.cancel()
            return True
        if self.tool:
            self._line_from = None
            self.clear_line()
            self.clear_rect()
            self.escaped.emit()
            return True
        return False

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self.cancel_tool():
            return
        super().keyPressEvent(e)

    def add_overlay(self, w: QWidget, corner: str = "bl"):
        """A floating control in a corner: 'bl' (the light dome), 'tl', 'tr'."""
        w.setParent(self)
        self._overlays.append((w, corner))
        self._place_overlays()

    def _place_overlays(self):
        for ov, corner in self._overlays:
            ov.adjustSize() if corner != "bl" else None
            if corner == "bl":
                ov.move(16, self.height() - ov.height() - 16)
            elif corner == "bc":
                # Centred, but never under the zoom controls at the bottom right: moved
                # left of them, or above them when the view is too narrow (side by side).
                x, y = (self.width() - ov.width()) // 2, self.height() - ov.height() - 18
                if self.hud.isVisible() and x + ov.width() > self.hud.x() - 8:
                    x = self.hud.x() - ov.width() - 8
                    if x < 16:
                        x, y = self.width() - ov.width() - 16, self.hud.y() - ov.height() - 8
                ov.move(x, y)
            elif corner == "tl":
                ov.move(14, 14)
            elif corner == "tr":
                ov.move(self.width() - ov.width() - 14, 14)

    def zoom(self) -> float:
        t = self.transform()
        return abs(t.m11() * t.m22() - t.m12() * t.m21()) ** 0.5

    def fit(self):
        if self.item.pixmap().isNull():
            return
        self.resetTransform()
        if self.rotation:
            self.rotate(self.rotation)
        r = self.transform().mapRect(self.item.sceneBoundingRect())
        vp = self.viewport().rect()
        if r.width() > 0 and r.height() > 0:
            k = min((vp.width() - 2) / r.width(), (vp.height() - 2) / r.height())
            if self.assessing:
                k *= 0.78                    # a wide grey margin round the image
            self.scale(k, k)
        self._place_frame()
        self.centerOn(self.item)
        self._fitted = True
        self._zoom_changed()

    def zoom_by(self, f: float):
        if self.item.pixmap().isNull():
            return
        z = self.zoom()
        f = max(0.02 / z, min(32.0 / z, f))
        self.scale(f, f)
        self._fitted = False
        self._zoom_changed()

    def _zoom_changed(self):
        self.pct.setText(f"{self.zoom() * 100:.0f}%")
        self.zoomed.emit(self.zoom())
        self.viewChanged.emit()

    # ── full-resolution detail over the preview ─────────────────────────
    def visible_rect(self):
        """The part of the picture on screen, in scene (full-size image) pixels."""
        r = self.mapToScene(self.viewport().rect()).boundingRect()
        return r.intersected(self.scene().sceneRect())

    def screen_px_per_image_px(self) -> float:
        """Device pixels on screen for one image pixel at the present zoom."""
        return self.zoom() * float(self.devicePixelRatioF())

    def show_detail(self, arr, rect):
        """Draw *arr* (uint8 RGB) at 1:1 over the scene rectangle *rect* (x1, y1, x2, y2)."""
        px = pixmap_from_array(arr)
        x1, y1, x2, y2 = rect
        self.detail.setPixmap(px)
        self.detail.setTransform(QTransform.fromScale((x2 - x1) / max(px.width(), 1),
                                                      (y2 - y1) / max(px.height(), 1)))
        self.detail.setPos(x1, y1)
        self.detail.show()

    def clear_detail(self):
        if self.detail.isVisible():
            self.detail.hide()
            self.detail.setPixmap(QPixmap())

    # Scrolling zooms, as in HELIO Pro classic: the mouse wheel, and two
    # fingers sliding up or down on a trackpad or Magic Mouse (Qt reports a
    # trackpad's travel as angle too, two eighths of a degree per pixel, so
    # one rate suits both: a wheel notch is ×1.15, the classic step).  A
    # pinch zooms too.  Shift + scroll, or two fingers sliding sideways,
    # pans; dragging the image pans as before.
    WHEEL_RATE = 1.00117

    def wheelEvent(self, e):
        d = e.angleDelta()
        sideways = abs(d.x()) > abs(d.y())
        if (e.modifiers() & Qt.KeyboardModifier.ShiftModifier) or sideways:
            return super().wheelEvent(e)
        if d.y():
            self.zoom_by(self.WHEEL_RATE ** d.y())
        e.accept()

    def viewportEvent(self, e):
        if e.type() == QEvent.Type.NativeGesture:
            g = e.gestureType()
            if g == Qt.NativeGestureType.ZoomNativeGesture:
                self.zoom_by(1.0 + e.value())
            elif g == Qt.NativeGestureType.RotateNativeGesture:      # two-finger turn
                self.rotated.emit(self.rotation - float(e.value()))
            elif g == Qt.NativeGestureType.SmartZoomNativeGesture:   # two-finger double tap
                if self._fitted:
                    self.zoom_by(2.0)
                else:
                    self.fit()
            e.accept()                   # the whole gesture, Begin to End, is ours
            return True
        return super().viewportEvent(e)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.empty.setGeometry(0, 0, self.width(), self.height())
        self.hud.adjustSize()
        self.hud.move(self.width() - self.hud.width() - 16, self.height() - self.hud.height() - 16)
        self._place_overlays()
        if self._fitted:
            self.fit()
        self.viewChanged.emit()


# ── Scale bar ─────────────────────────────────────────────────────────

class ScaleBar(QWidget):
    """A scale bar for the image as it is shown: a round length (1, 2 or 5
    times a power of ten) about 120 screen pixels long, in millimetres once a
    scale has been measured, else in image pixels.  It follows the zoom."""

    def __init__(self, view: "Viewport", mm_per_px):
        super().__init__(view)
        self.view, self.mm_per_px = view, mm_per_px
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFixedSize(280, 34)
        self.setAccessibleName("Scale bar")
        view.zoomed.connect(lambda _z: self.update())
        self.colours = {}

    def recolour(self, t):
        self.colours = dict(bg=t.popover, line=t.popover_line, text=t.text, bar=t.text)
        self.update()

    def _length(self):
        import math as _m
        z = self.view.zoom() or 1.0                 # screen px per image px
        mm = self.mm_per_px() if callable(self.mm_per_px) else self.mm_per_px
        unit_per_px = (mm or 1.0) / z               # mm (or image px) per screen px
        target = 120 * unit_per_px
        p = 10 ** _m.floor(_m.log10(target)) if target > 0 else 1
        nice = min((p * k for k in (1, 2, 5, 10)), key=lambda v: abs(v - target))
        return nice, nice / unit_per_px, bool(mm)

    def paintEvent(self, e):
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPen
        if self.view.item.pixmap().isNull():
            return
        c = self.colours or dict(bg="#1E2745", line="#34405F", text="#ECE7DA", bar="#ECE7DA")
        value, px, real = self._length()
        if not (4 < px < self.width() - 12):
            return
        if real:
            text = (f"{value * 1000:g} µm" if value < 1 else f"{value:g} mm" if value < 10
                    else f"{value / 10:g} cm")
        else:
            text = f"{value:g} px"
        if not real:
            text += " · no scale set"
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        f = p.font()
        f.setPointSize(12)
        p.setFont(f)
        from PySide6.QtGui import QFontMetrics
        tw = QFontMetrics(f).horizontalAdvance(text)
        p.setPen(QPen(QColor(c["line"]), 1))
        bg = QColor(c["bg"])
        bg.setAlpha(225)
        p.setBrush(bg)
        w = min(self.width() - 1, max(px + 24, tw + 20))
        x0 = (self.width() - w) / 2
        p.drawRoundedRect(QRectF(x0, 0.5, w, self.height() - 1), 7, 7)
        y = self.height() - 10
        bx = (self.width() - px) / 2
        p.setPen(QPen(QColor(c["bar"]), 2))
        p.drawLine(QPointF(bx, y), QPointF(bx + px, y))
        p.drawLine(QPointF(bx, y - 4), QPointF(bx, y + 3))
        p.drawLine(QPointF(bx + px, y - 4), QPointF(bx + px, y + 3))
        p.setPen(QColor(c["text"]))
        p.drawText(QRectF(x0, 2, w, 16), Qt.AlignmentFlag.AlignCenter, text)
        p.end()


# ── Choice cards ──────────────────────────────────────────────────────

class Card(QFrame):
    """A large choice: title, badge, what it does, and its current status."""
    clicked = Signal()

    def __init__(self, title: str, body: str, badge: str = "", icon_name: str = ""):
        super().__init__()
        self.setProperty("card", "true")
        self.setProperty("selected", "false")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(title)
        # Ignored width: cards in a row share it equally, whatever their words.
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(150)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)
        top = QHBoxLayout()
        self.icon = QLabel()
        self.icon_name = icon_name
        self.icon.setVisible(bool(icon_name))
        self.badge = label(badge, "caption")
        top.addWidget(self.icon)
        top.addStretch(1)
        top.addWidget(self.badge)
        if icon_name or badge:
            lay.addLayout(top)
        self.title = label(title, "section", wrap=True)
        lay.addWidget(self.title)
        self.body = label(body, "secondary", wrap=True)
        self.status = label("", "caption", wrap=True)
        lay.addWidget(self.body)
        lay.addWidget(self.status)
        lay.addStretch(1)

    def recolour(self, t):
        if self.icon_name:
            self.icon.setPixmap(icons.icon(self.icon_name, t.accent_text, 20).pixmap(20, 20))

    def set_selected(self, on: bool):
        self.setProperty("selected", "true" if on else "false")
        restyle(self)

    def set_status(self, text: str, status: str = ""):
        self.status.setText(text)
        self.status.setProperty("status", status)
        self.status.setProperty("role", "caption" if not status else "")
        restyle(self.status)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.rect().contains(e.position().toPoint()):
            self.clicked.emit()
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(e)


# ── Light dome ────────────────────────────────────────────────────────

class LightDome(QWidget):
    """Where the light is: drag the dot.  Centre is overhead, the rim grazing.

    The light is kept in screen frame (+Y down), as the classic dome and saved
    bookmarks store it; helio.application.render converts it once.  Arrow
    keys move it 5 degrees: left/right turn it, up/down raise and lower it.
    """
    changed = Signal(float, float, float, bool)      # lx, ly, lz, still dragging

    SIZE, R = 132, 50

    def __init__(self, size: int = 132):
        super().__init__()
        self.SIZE = size
        self.R = int(size * 0.38)
        self.setFixedSize(self.SIZE, self.SIZE + 22)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Light direction")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lx, self.ly, self.lz = 0.0, 0.0, 1.0
        self.captured: list[tuple[float, float]] = []
        self.show_captured = True
        self.colours = {}

    def recolour(self, t):
        self.colours = dict(bg=t.popover, line=t.line, text=t.text2, dot=t.accent, rim=t.text3,
                            cap=t.info)
        self.update()

    def set_captured(self, lights):
        """The capture's lights as dots: (N, 3) +Y up (the light matrix), drawn
        in the dome's screen frame, so a light from the top sits at the top
        (the classic dome drew them upside down, KD-018)."""
        import numpy as np
        self.captured = []
        if lights is not None:
            for v in np.asarray(lights, np.float64):
                n = float(np.linalg.norm(v))
                if n > 1e-6 and np.isfinite(v).all():
                    self.captured.append((float(v[0] / n), float(-v[1] / n)))
        self.update()

    def angles(self) -> tuple[float, float]:
        import math
        az = math.degrees(math.atan2(self.lx, -self.ly)) % 360.0
        el = math.degrees(math.asin(max(-1.0, min(1.0, self.lz))))
        return az, el

    def set_light(self, lx, ly, lz):
        self.lx, self.ly, self.lz = float(lx), float(ly), float(lz)
        self.setToolTip(self.caption())
        self.update()

    def caption(self) -> str:
        az, el = self.angles()
        if el > 89.5:
            return "Light overhead"
        return f"Light {az:.0f}° · {el:.0f}° up"

    def _centre(self):
        return self.SIZE / 2, self.SIZE / 2

    def _from_pos(self, x, y, dragging):
        import math
        cx, cy = self._centre()
        dx, dy = (x - cx) / self.R, (y - cy) / self.R
        r = math.hypot(dx, dy)
        if r > 0.996:                            # keep the light above the horizon
            dx, dy = dx / r * 0.996, dy / r * 0.996
            r = 0.996
        self.set_light(dx, dy, math.sqrt(max(0.0, 1.0 - r * r)))
        self.changed.emit(self.lx, self.ly, self.lz, dragging)

    def mousePressEvent(self, e):
        self._from_pos(e.position().x(), e.position().y(), True)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._from_pos(e.position().x(), e.position().y(), True)

    def mouseReleaseEvent(self, e):
        self._from_pos(e.position().x(), e.position().y(), False)

    def keyPressEvent(self, e):
        import math
        az, el = self.angles()
        step = {Qt.Key.Key_Left: (-5, 0), Qt.Key.Key_Right: (5, 0),
                Qt.Key.Key_Up: (0, 5), Qt.Key.Key_Down: (0, -5)}.get(e.key())
        if not step:
            return super().keyPressEvent(e)
        az, el = (az + step[0]) % 360, max(5.0, min(90.0, el + step[1]))
        c = math.cos(math.radians(el))
        self.set_light(c * math.sin(math.radians(az)), -c * math.cos(math.radians(az)),
                       math.sin(math.radians(el)))
        self.changed.emit(self.lx, self.ly, self.lz, False)

    def paintEvent(self, e):
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QPen
        c = self.colours or dict(bg="#1E2745", line="#2C3756", text="#AEB5C8", dot="#E9B44C",
                                 rim="#959DB6")
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(QColor(c["line"]), 1))
        p.setBrush(QColor(c["bg"]))
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 12, 12)
        cx, cy = self._centre()
        p.setBrush(Qt.BrushStyle.NoBrush)
        for frac in (1.0, 0.866, 0.5):           # horizon, 30° and 60° elevation
            p.setPen(QPen(QColor(c["rim"] if frac == 1.0 else c["line"]), 1))
            p.drawEllipse(QPointF(cx, cy), self.R * frac, self.R * frac)
        p.setPen(QPen(QColor(c["line"]), 1))
        p.drawLine(QPointF(cx - self.R, cy), QPointF(cx + self.R, cy))
        p.drawLine(QPointF(cx, cy - self.R), QPointF(cx, cy + self.R))
        p.setPen(QColor(c["text"]))
        f = p.font()
        f.setPointSize(12)
        p.setFont(f)
        p.drawText(QRectF(cx - 10, cy - self.R - 16, 20, 14), Qt.AlignmentFlag.AlignCenter, "N")
        if self.SIZE >= 160:
            p.drawText(QRectF(cx + self.R + 2, cy - 7, 14, 14), Qt.AlignmentFlag.AlignCenter, "E")
            p.drawText(QRectF(cx - self.R - 16, cy - 7, 14, 14), Qt.AlignmentFlag.AlignCenter, "W")
            p.drawText(QRectF(cx - 10, cy + self.R + 1, 20, 14), Qt.AlignmentFlag.AlignCenter, "S")
            p.setPen(QColor(c["rim"]))
            small = p.font()
            small.setPointSize(12)
            p.setFont(small)
            p.drawText(QPointF(cx + 3, cy - self.R * 0.866 + 13), "30°")
            p.drawText(QPointF(cx + 3, cy - self.R * 0.5 + 13), "60°")
        if self.show_captured and self.captured:
            cap = QColor(c.get("cap", "#6CC4F0"))
            cap.setAlpha(200)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(cap)
            for x, y in self.captured:
                p.drawEllipse(QPointF(cx + x * self.R, cy + y * self.R), 2.6, 2.6)
        if self.hasFocus():
            p.setPen(QPen(QColor(c["dot"]), 2))
            p.drawEllipse(QPointF(cx, cy), self.R + 3, self.R + 3)
        dot = QColor(c["dot"])
        p.setPen(Qt.PenStyle.NoPen)
        glow = QColor(dot)
        glow.setAlpha(70)
        p.setBrush(glow)
        x, y = cx + self.lx * self.R, cy + self.ly * self.R
        p.drawEllipse(QPointF(x, y), 11, 11)
        p.setBrush(dot)
        p.drawEllipse(QPointF(x, y), 6.5, 6.5)
        p.setPen(QColor(c["text"]))
        az, el = self.angles()
        short = "Overhead" if el > 89.5 else f"{az:.0f}° · {el:.0f}° up"
        p.drawText(QRectF(0, self.SIZE - 4, self.width(), 20), Qt.AlignmentFlag.AlignCenter, short)
        p.end()


class ProfileChart(QWidget):
    """A height profile as a line, with its axis labels (MET-002)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(170)
        self.setAccessibleName("Height profile")
        self.x = self.z = None
        self.x_label = self.z_unit = ""
        self.colours = {}

    def recolour(self, t):
        self.colours = dict(bg=t.well, line=t.line, text=t.text3, curve=t.accent)
        self.update()

    def set_profile(self, x, z, x_label, z_unit):
        self.x, self.z, self.x_label, self.z_unit = x, z, x_label, z_unit
        self.update()

    def paintEvent(self, e):
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QPainterPath, QPen
        c = self.colours or dict(bg="#0D1224", line="#2C3756", text="#959DB6", curve="#E9B44C")
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(QColor(c["line"]), 1))
        p.setBrush(QColor(c["bg"]))
        p.drawRoundedRect(r, 8, 8)
        f = p.font()
        f.setPointSize(12)
        p.setFont(f)
        if self.x is None or len(self.x) < 2:
            p.setPen(QColor(c["text"]))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "Drag across the surface")
            p.end()
            return
        plot = QRectF(r.left() + 10, r.top() + 10, r.width() - 20, r.height() - 38)
        x0, x1 = float(self.x[0]), float(self.x[-1])
        z0, z1 = float(min(self.z)), float(max(self.z))
        span_x, span_z = (x1 - x0) or 1.0, (z1 - z0) or 1.0
        path = QPainterPath()
        for i, (xv, zv) in enumerate(zip(self.x, self.z)):
            pt = QPointF(plot.left() + (float(xv) - x0) / span_x * plot.width(),
                         plot.bottom() - (float(zv) - z0) / span_z * plot.height())
            path.moveTo(pt) if i == 0 else path.lineTo(pt)
        p.setPen(QPen(QColor(c["curve"]), 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.setPen(QColor(c["text"]))
        p.drawText(QRectF(r.left() + 10, r.bottom() - 24, r.width() - 20, 18),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"{self.x_label}: 0 – {x1:.2f}")
        p.drawText(QRectF(r.left() + 10, r.bottom() - 24, r.width() - 20, 18),
                   Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                   f"height {z1 - z0:.3g} {self.z_unit}")
        p.end()
