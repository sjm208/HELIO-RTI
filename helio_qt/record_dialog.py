"""Record video: a turntable, flyover loop, overhead orbit or light sweep of
the 3D relief (or the high-detail window's), at a chosen length, size, frame
rate, direction and file type.
The recording itself is MainWindow.record_relief_video (helio.application.
viewer3d.video_pose for the camera and the light)."""
from __future__ import annotations

import os

import json

from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QFileDialog, QHBoxLayout, QVBoxLayout

from helio.application.viewer3d import (REPEATS, SIZE_LIMITS, VIDEO_LENGTHS, VIDEO_RATES, VIDEO_SIZES,
                                        VIDEO_STYLES)
from helio.domain.view_controls import Control
from helio_qt.controls import ParamRow, Segmented
from helio_qt.widgets import button, label


def limit_row(lay, win) -> Segmented:
    """The File size row (viewer3d.SIZE_LIMITS), at the limit chosen last."""
    row = QHBoxLayout()
    row.addWidget(label("File size", "secondary"))
    seg = Segmented([(str(k), t) for k, (t, _b) in SIZE_LIMITS.items()],
                    {str(k): b for k, (_t, b) in SIZE_LIMITS.items()})
    try:
        last = str(int(float(win.prefs.value("video/max_mb", 0) or 0)))
    except Exception:                          # noqa: BLE001
        last = "0"
    seg.set_current(last if int(last) in SIZE_LIMITS else "0")
    row.addWidget(seg)
    row.addStretch(1)
    lay.addLayout(row)
    return seg


class ViewsRows:
    """Two or three views in one video (helio.application.multiview): the view
    as shown, and up to two more, each in colour or grey; side by side or one
    after another; named or not.  Remembered for next time."""

    NONE = "—"

    def __init__(self, lay, win, first: str = "The view as shown"):
        from helio.application.multiview import COLOUR, GREY, LAYOUTS
        from helio.domain.registry import view_name
        self.win = win
        ins = win.workspaces["Inspect"]
        skip = {getattr(ins, "DEPTH_MAP", ""), getattr(ins, "BACKLIT", "")}
        engines = [e for e in ins.order if e not in skip]
        try:
            last = json.loads(str(win.prefs.value("video/views", "{}")) or "{}")
        except Exception:                          # noqa: BLE001
            last = {}
        lay.addWidget(label("Views", "secondary"))
        lay.addWidget(label(f"{first}, and up to two more: a sweep or turn that shows the object "
                            "in several views at once, or one after another.", "caption", wrap=True))
        self.slots = []
        for n in range(2):
            row = QHBoxLayout()
            row.addWidget(label("Also" if n == 0 else "And", "secondary"))
            eng = QComboBox()
            eng.addItem(self.NONE, "")
            for e in engines:
                eng.addItem(view_name(e), e)
            eng.setAccessibleName(f"Extra view {n + 1}")
            eng.setToolTip("Another view to show in the same video, with its own settings as you "
                           "last left them.")
            layer = QComboBox()
            layer.addItem("Colour", COLOUR)
            layer.addItem("Grey", GREY)
            layer.setAccessibleName(f"Extra view {n + 1}: colour or grey")
            layer.setToolTip("Colour: the view over the object's colour, as the Colour layer "
                             "shows it. Grey: the shading alone.")
            saved = (last.get("views") or [])
            if n < len(saved):
                i = eng.findData(saved[n][0])
                eng.setCurrentIndex(max(0, i))
                layer.setCurrentIndex(max(0, layer.findData(saved[n][1])))
            row.addWidget(eng, 1)
            row.addWidget(layer)
            lay.addLayout(row)
            eng.currentIndexChanged.connect(self._follow)
            self.slots.append((eng, layer))
        row = QHBoxLayout()
        row.addWidget(label("Show them", "secondary"))
        self.layout = Segmented([(k, t) for k, (t, _b) in LAYOUTS.items()],
                                {k: b for k, (_t, b) in LAYOUTS.items()})
        self.layout.set_current(last.get("layout") if last.get("layout") in LAYOUTS else "side")
        row.addWidget(self.layout)
        self.names = QCheckBox("Name each view")
        self.names.setToolTip("The view's name in the lower left corner of each picture.")
        self.names.setChecked(bool(last.get("labels", True)))
        row.addWidget(self.names)
        row.addStretch(1)
        lay.addLayout(row)
        self._row = [self.layout, self.names]
        self._follow()

    def _follow(self, *_a):
        several = len(self.views()) > 1
        for wdg in self._row:
            wdg.setEnabled(several)
        for eng, layer in self.slots:
            layer.setEnabled(bool(eng.currentData()))

    def views(self) -> list:
        out = [("", "")]
        for eng, layer in self.slots:
            if eng.currentData():
                out.append((str(eng.currentData()), str(layer.currentData())))
        return out

    def settings(self) -> dict:
        return dict(views=self.views(), layout=self.layout.current() or "side",
                    labels=self.names.isChecked())

    def keep(self):
        st = self.settings()
        try:
            self.win.prefs.setValue("video/views", json.dumps(
                dict(views=[list(v) for v in st["views"][1:]], layout=st["layout"],
                     labels=st["labels"])))
        except Exception:                          # noqa: BLE001
            pass


def repeat_box(row, fmt: Segmented) -> QComboBox:
    """How many times an MP4 plays round (viewer3d.REPEATS), beside the File
    choice; not for a GIF, which loops by itself."""
    row.addSpacing(16)
    row.addWidget(label("Plays round", "secondary"))
    box = QComboBox()
    for n in REPEATS:
        box.addItem("Once" if n == 1 else f"{n} times", n)
    box.setAccessibleName("Plays round")
    tip = ("The MP4 goes round this many times, one after another, so it keeps playing where "
           "the slide or player is not set to loop. The file is that many times longer (or, "
           "under a size limit, its picture smaller).")

    def follow(key=None):
        gif = (key or fmt.current()) == "gif"
        box.setEnabled(not gif)
        box.setToolTip("A GIF loops by itself, for ever." if gif else tip)
    fmt.chosen.connect(follow)
    follow()
    row.addWidget(box)
    return box


def times_round(box: QComboBox, fmt: str) -> int:
    return 1 if fmt == "gif" else int(box.currentData() or 1)


def keep_limit(win, mb: float):
    try:
        win.prefs.setValue("video/max_mb", int(mb))
    except Exception:                          # noqa: BLE001
        pass


class RecordDialog(QDialog):
    def __init__(self, win, relief, style: str = "turntable"):
        super().__init__(win)
        self.win, self.relief = win, relief
        self.setWindowTitle("Record video")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(10)
        lay.addWidget(label("Record video", "title"))
        self.style = Segmented([(k, t) for k, (t, _b) in VIDEO_STYLES.items()],
                               {k: b for k, (_t, b) in VIDEO_STYLES.items()})
        self.style.chosen.connect(self._style_changed)
        lay.addWidget(self.style)
        self.blurb = label("", "caption", wrap=True)
        lay.addWidget(self.blurb)
        row = QHBoxLayout()
        row.addWidget(label("One turn takes", "secondary"))
        self.length = QComboBox()
        for s in VIDEO_LENGTHS:
            self.length.addItem(f"{s} s", s)
        self.length.setCurrentIndex(VIDEO_LENGTHS.index(12))
        self.length.setAccessibleName("One turn takes")
        self.length.setToolTip("The speed: the camera (or, in the light sweep, the light) goes "
                               "once round in this time, and the video is that long. Longer is "
                               "slower.")
        row.addWidget(self.length)
        self.speed = label("", "caption")
        row.addWidget(self.speed)
        self.length.currentIndexChanged.connect(self._say_speed)
        row.addSpacing(16)
        row.addWidget(label("Frames a second", "secondary"))
        self.fps = QComboBox()
        for f in VIDEO_RATES:
            self.fps.addItem(f"{f}", f)
        self.fps.setCurrentIndex(VIDEO_RATES.index(30))
        self.fps.setAccessibleName("Frames a second")
        self.fps.setToolTip("24 looks like film, 30 is usual, 60 is the smoothest (and twice "
                            "as long to record).")
        row.addWidget(self.fps)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(label("Size", "secondary"))
        self.size = Segmented([(k, k) for k in VIDEO_SIZES],
                              {k: f"{w} × {h} px" for k, (w, h) in VIDEO_SIZES.items()}, columns=3)
        self.size.set_current("HD")
        lay.addWidget(self.size)
        row = QHBoxLayout()
        row.addWidget(label("Frame", "secondary"))
        self.fit = Segmented([("fit", "Fit the object"), ("full", "The whole size")],
                             {"fit": "Cut to the object, as far as it goes in the whole video, its long "
                                     "side the size chosen: a portrait object fills the frame instead "
                                     "of a strip down the middle.",
                              "full": "Exactly the size chosen, with the background round the object."})
        self.fit.set_current("fit")
        row.addWidget(self.fit)
        row.addStretch(1)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(label("File", "secondary"))
        self.fmt = Segmented([("mp4", "MP4 video"), ("gif", "GIF")],
                             {"mp4": "Plays everywhere: browsers, QuickTime, PowerPoint.",
                              "gif": "Loops on any web page or slide; larger, with fewer colours."})
        self.fmt.set_current("mp4")
        row.addWidget(self.fmt)
        self.repeat = repeat_box(row, self.fmt)
        row.addSpacing(16)
        row.addWidget(label("Turn", "secondary"))
        self.turn = Segmented([("cw", "Clockwise"), ("ccw", "Anticlockwise")],
                              {"cw": "Round the way the compass reads: north, east, south, west.",
                               "ccw": "The other way round."})
        self.turn.set_current("cw")
        row.addWidget(self.turn)
        row.addStretch(1)
        lay.addLayout(row)
        self.limit = limit_row(lay, win)
        self.views = ViewsRows(lay, win, "The relief as it is shown")
        self.light_el = ParamRow(Control("light_el", "Light height", 10.0, 60.0, 30.0, 1.0,
                                         "How far above the object the light circles: lower "
                                         "rakes harder and shows more.", unit="°"))
        lay.addWidget(self.light_el)
        self.light_from = ParamRow(Control("light_from", "Light starts from", 0.0, 359.0, 0.0, 1.0,
                                           "The compass bearing the light starts (and ends) at: "
                                           "0° north, 90° east.", unit="°"))
        lay.addWidget(self.light_from)
        try:                                       # from where the light is now
            self.light_from.set_value(float(win.workspaces["Inspect"].az_row.value()))
        except Exception:
            pass
        brow = QHBoxLayout()
        brow.addStretch(1)
        close = button("Cancel", "quiet")
        close.clicked.connect(self.reject)
        self.btn_record = button("Record…", "primary")
        self.btn_record.clicked.connect(self.record)
        brow.addWidget(close)
        brow.addWidget(self.btn_record)
        lay.addLayout(brow)
        self.style.set_current(style if style in VIDEO_STYLES else "turntable")
        self._style_changed(self.style.current())
        self._say_speed()

    def _say_speed(self, *_a):
        s = float(self.length.currentData() or 12)
        self.speed.setText(f"{360.0 / s:.0f}° a second" if 360.0 / s >= 1 else f"{360.0 / s:.1f}° a second")

    def _style_changed(self, key):
        self.blurb.setText(VIDEO_STYLES.get(key, ("", ""))[1])
        self.light_el.setVisible(key == "light")
        self.light_from.setVisible(key == "light")

    def settings(self) -> dict:
        return dict(style=self.style.current() or "turntable",
                    seconds=int(self.length.currentData()),
                    size=VIDEO_SIZES[self.size.current() or "HD"],
                    light_el=float(self.light_el.value()),
                    light_az0=float(self.light_from.value()),
                    fps=int(self.fps.currentData()),
                    fmt=self.fmt.current() or "mp4",
                    reverse=self.turn.current() == "ccw",
                    fit=(self.fit.current() or "fit") == "fit",
                    max_mb=float(self.limit.current() or 0),
                    repeat=times_round(self.repeat, self.fmt.current() or "mp4"),
                    **self.views.settings())

    def record(self, path: str = ""):
        from helio_qt.publish import file_stem
        st = self.settings()
        if not path:
            ext = st["fmt"]
            start = os.path.join(str(self.win.prefs.value("export_folder", ""))
                                 or self.win.default_export_dir(),
                                 f"{file_stem(self.win.project.name)}_3d_{st['style']}.{ext}")
            path, _ = QFileDialog.getSaveFileName(
                self, "Record video", start,
                "MP4 video (*.mp4)" if ext == "mp4" else "GIF (*.gif)")
            if not path:
                return ""
            root, e = os.path.splitext(path)
            if e.lower() not in (".mp4", ".gif"):
                path = f"{root}.{ext}"
        keep_limit(self.win, st["max_mb"])
        self.views.keep()
        self.accept()
        return self.win.record_relief_video(path, st["style"], relief=self.relief,
                                            size=st["size"], light_el=st["light_el"],
                                            seconds=st["seconds"], fps=st["fps"],
                                            reverse=st["reverse"], light_az0=st["light_az0"],
                                            fit=st["fit"], max_mb=st["max_mb"],
                                            repeat=st["repeat"], views=st["views"],
                                            layout=st["layout"], labels=st["labels"])


PICTURE_SIZES = {"screen": ("As on screen", 0), "hd": ("HD", 1920), "4k": ("4K", 3840),
                 "6k": ("6K", 5760), "8k": ("8K", 7680)}
BACKGROUNDS = {"screen": ("As on screen", ""), "white": ("White", "#FFFFFF"),
               "black": ("Black", "#000000"), "grey": ("Grey", "#808080")}


# What Save View can save, and what each is (key: button, tip).
SAVE_WHATS = {
    "image": ("The image", "The view at the photographs' full resolution, as a TIFF with the "
              "capture details in it."),
    "3d": ("The 3D relief", "The 3D view as on screen, or drawn afresh at up to 8K, on a chosen "
           "background."),
    "centre": ("Both, as in the centre", "The image and the 3D relief side by side, as the "
               "centre shows them, as one picture."),
}


class SaveViewDialog(QDialog):
    """Save View: one command for every still picture of what is on screen
    (it was four: Save Current View, This View at Full Resolution, Save the 3D
    View as a Picture, Save the Centre View as a Picture).  *what* is the
    image at full resolution (TIFF), the 3D relief (as on screen or up to 8K,
    on a chosen background), or the centre's image and relief side by side.
    *relief* is the 3D view to draw (the high-detail window's, or the
    centre's, found when it is saved if not given); with *can_3d* False
    (no OpenGL here) only the image can be saved."""

    def __init__(self, win, relief=None, what: str = "image", centre_both: bool = False,
                 can_3d: bool = True):
        super().__init__(win)
        self.win, self.relief = win, relief
        self.setWindowTitle("Save View")
        self.setMinimumWidth(500)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(10)
        lay.addWidget(label("Save View", "title"))
        self.what = Segmented([(k, t) for k, (t, _d) in SAVE_WHATS.items()],
                              {k: d for k, (_t, d) in SAVE_WHATS.items()})
        why = {"3d": "" if can_3d else "The 3D view is not available here: it needs OpenGL 4.1.",
               "centre": "" if can_3d and centre_both else
               "Show the image and the 3D relief side by side first (View › Centre, ⌥3)."}
        for k, reason in why.items():
            if reason:
                self.what.buttons[k].setEnabled(False)
                self.what.buttons[k].setToolTip(f"{SAVE_WHATS[k][1]}\n\n{reason}")
        if not self.what.buttons.get(what) or not self.what.buttons[what].isEnabled():
            what = "image"
        self.what.set_current(what)
        self.what.chosen.connect(lambda _k: self._show())
        lay.addWidget(self.what)
        self.note = label("", "caption", wrap=True)
        lay.addWidget(self.note)
        self.size_label = label("Size (the long side)", "secondary")
        lay.addWidget(self.size_label)
        self.size = Segmented([(k, t) for k, (t, _n) in PICTURE_SIZES.items()],
                              {k: (f"Drawn afresh, {n:,} px on the long side" if n else
                                   "The view as it is on screen") for k, (_t, n) in
                               PICTURE_SIZES.items()})
        self.size.set_current("4k")
        lay.addWidget(self.size)
        self.bg_label = label("Background", "secondary")
        lay.addWidget(self.bg_label)
        self.bg = Segmented([(k, t) for k, (t, _c) in BACKGROUNDS.items()])
        self.bg.set_current("screen")
        lay.addWidget(self.bg)
        self.fmt_label = label("File", "secondary")
        lay.addWidget(self.fmt_label)
        self.fmt = Segmented([("png", "PNG"), ("tif", "TIFF"), ("jpg", "JPEG")],
                             {"png": "Lossless, for the web and slides.",
                              "tif": "Lossless, for print and archives.",
                              "jpg": "Smallest, for email."})
        self.fmt.set_current("png")
        lay.addWidget(self.fmt)
        brow = QHBoxLayout()
        brow.addStretch(1)
        close = button("Cancel", "quiet")
        close.clicked.connect(self.reject)
        self.btn_save = button("Save…", "primary")
        self.btn_save.clicked.connect(lambda: self.save())
        brow.addWidget(close)
        brow.addWidget(self.btn_save)
        lay.addLayout(brow)
        self._show()

    def _show(self):
        """Only the rows that apply to what is being saved."""
        what = self.what.current() or "image"
        image = what == "image"
        for wd in (self.size_label, self.size, self.fmt_label, self.fmt):
            wd.setVisible(not image)
        for wd in (self.bg_label, self.bg):
            wd.setVisible(what == "3d")
        note = ""
        if image:
            res = getattr(self.win.project, "result", None)
            size = f", {res.w:,} × {res.h:,} px" if res is not None and getattr(res, "w", 0) else ""
            note = f"At the photographs' full resolution{size}, as a TIFF with the capture details in it."
        elif what == "centre":
            note = "The image and the 3D relief side by side, as the centre shows them."
        self.note.setText(note)
        self.note.setVisible(bool(note))
        self.adjustSize()

    def settings(self) -> dict:
        what = self.what.current() or "image"
        if what == "image":
            return dict(what=what, fmt="tif")
        st = dict(what=what, long_side=PICTURE_SIZES[self.size.current() or "4k"][1],
                  fmt=self.fmt.current() or "png")
        if what == "3d":
            st["background"] = BACKGROUNDS[self.bg.current() or "screen"][1]
        return st

    def save(self, path: str = "") -> str:
        from helio_qt.publish import file_stem
        st = self.settings()
        what, ext = st.pop("what"), st.pop("fmt")
        if not path:
            tag = "" if what == "image" else (self.size.current() or "4k")
            view = file_stem(self.win.view or "view")[:24] if what == "image" else \
                ("3d" if what == "3d" else "centre")
            start = os.path.join(str(self.win.prefs.value("export_folder", ""))
                                 or self.win.default_export_dir(),
                                 f"{file_stem(self.win.project.name)}_{view}"
                                 f"{'' if tag in ('', 'screen') else '_' + tag}.{ext}")
            filters = {"png": "PNG (*.png)", "tif": "TIFF (*.tif *.tiff)", "jpg": "JPEG (*.jpg *.jpeg)"}
            path, _ = QFileDialog.getSaveFileName(self, "Save View", start, filters[ext])
            if not path:
                return ""
        root, e = os.path.splitext(path)
        allowed = (".tif", ".tiff") if what == "image" else (".png", ".tif", ".tiff", ".jpg", ".jpeg")
        if e.lower() not in allowed:
            path = f"{root}.{ext}"
        self.accept()
        if what == "image":
            return self.win.write_view_tiff(path)
        if what == "3d":
            r = self.relief or self.win._relief_ready()
            return self.win.write_relief_picture(r, path, **st) if r is not None else ""
        return self.win.write_centre_picture(path, st["long_side"])
