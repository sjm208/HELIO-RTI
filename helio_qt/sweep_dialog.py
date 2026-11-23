"""Save the light sweep of the 2D view: the light once round the object, as
in RTI, at a chosen height, length, size and file type (GIF or MP4).  The
frames are drawn by helio_qt.publish.light_sweep; the 3D view's own sweep is
Record video's Light sweep (helio_qt.record_dialog)."""
from __future__ import annotations

import os

from PySide6.QtWidgets import QComboBox, QDialog, QFileDialog, QHBoxLayout, QVBoxLayout

from helio.domain.view_controls import Control
from helio_qt.controls import ParamRow, Segmented
from helio_qt.record_dialog import ViewsRows, keep_limit, limit_row, repeat_box, times_round
from helio_qt.widgets import button, label

LENGTHS = (3, 4, 6, 8, 12, 20, 30)          # seconds for once round: the speed
SIZES = {"screen": ("As on screen", 0, False), "720": ("720 px", 720, True),
         "1080": ("1080 px", 1080, True), "2k": ("2,000 px", 2000, True),
         "4k": ("4K, 3,840 px", 3840, True), "full": ("Full resolution", 0, True)}
SIZE_TIPS = {"screen": "The preview, as Inspect draws it: quick.",
             "720": "720 px on the long side: small, for e-mail and slides.",
             "1080": "1080 px on the long side, for the web and presentations.",
             "2k": "2,000 px on the long side.",
             "4k": "3,840 px on the long side, drawn from the full photographs. A 4K GIF "
                   "runs to a hundred megabytes or more: MP4 is a tenth of that.",
             "full": "Every frame at the photographs' full size: slow, and very large as a GIF."}


class SweepDialog(QDialog):
    def __init__(self, win):
        super().__init__(win)
        self.win = win
        ins = win.workspaces["Inspect"]
        self.setWindowTitle("Save the light sweep")
        self.setMinimumWidth(500)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 16)
        lay.setSpacing(10)
        lay.addWidget(label("Save the light sweep", "title"))
        lay.addWidget(label("The light once round the object in this view.", "caption", wrap=True))
        self.light_el = ParamRow(Control("sweep_el", "Light height", 5.0, 80.0,
                                         float(ins.height_row.value()), 1.0,
                                         "How far above the object the light circles: lower "
                                         "rakes harder and shows more. Starts at the light's "
                                         "height now.", unit="°"))
        lay.addWidget(self.light_el)
        self.light_from = ParamRow(Control("sweep_from", "Starts from", 0.0, 359.0,
                                           float(ins.az_row.value()), 1.0,
                                           "The compass bearing the light starts (and ends) at: "
                                           "0° north, 90° east.", unit="°"))
        lay.addWidget(self.light_from)
        row = QHBoxLayout()
        row.addWidget(label("One turn takes", "secondary"))
        self.length = QComboBox()
        for s in LENGTHS:
            self.length.addItem(f"{s} s", s)
        self.length.setCurrentIndex(LENGTHS.index(4))
        self.length.setAccessibleName("One turn takes")
        self.length.setToolTip("The speed: the light goes once round in this time, and the "
                               "sweep is that long. Longer is slower.")
        row.addWidget(self.length)
        self.speed = label("", "caption")
        row.addWidget(self.speed)
        self.length.currentIndexChanged.connect(self._say_speed)
        self._say_speed()
        row.addSpacing(16)
        row.addWidget(label("Turn", "secondary"))
        self.turn = Segmented([("cw", "Clockwise"), ("ccw", "Anticlockwise")],
                              {"cw": "North, east, south, west.", "ccw": "The other way round."})
        self.turn.set_current("cw")
        row.addWidget(self.turn)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addWidget(label("Size (the long side)", "secondary"))
        self.size = Segmented([(k, t) for k, (t, _n, _f) in SIZES.items()], SIZE_TIPS, columns=3)
        self.size.set_current("screen")
        lay.addWidget(self.size)
        row = QHBoxLayout()
        row.addWidget(label("File", "secondary"))
        self.fmt = Segmented([("gif", "GIF"), ("mp4", "MP4 video")],
                             {"gif": "Loops on any web page or slide.",
                              "mp4": "Smaller and sharper; plays in browsers, QuickTime, PowerPoint."})
        self.fmt.set_current("gif")
        row.addWidget(self.fmt)
        self.repeat = repeat_box(row, self.fmt)
        row.addStretch(1)
        lay.addLayout(row)
        self.limit = limit_row(lay, win)
        self.views = ViewsRows(lay, win, "The view on screen")
        brow = QHBoxLayout()
        brow.addStretch(1)
        close = button("Cancel", "quiet")
        close.clicked.connect(self.reject)
        self.btn_save = button("Save…", "primary")
        self.btn_save.clicked.connect(lambda: self.save())
        brow.addWidget(close)
        brow.addWidget(self.btn_save)
        lay.addLayout(brow)

    def _say_speed(self, *_a):
        self.speed.setText(f"{360.0 / float(self.length.currentData() or 4):.0f}° a second")

    def settings(self) -> dict:
        _t, long_side, full = SIZES[self.size.current() or "screen"]
        fmt = self.fmt.current() or "gif"
        return dict(elevation=float(self.light_el.value()), start_az=float(self.light_from.value()),
                    seconds=float(self.length.currentData()), reverse=self.turn.current() == "ccw",
                    full=full, long_side=long_side, fmt=fmt, fps=20 if fmt == "gif" else 30,
                    max_mb=float(self.limit.current() or 0), repeat=times_round(self.repeat, fmt),
                    **self.views.settings())

    def save(self, path: str = "") -> str:
        from helio_qt.publish import file_stem
        st = self.settings()
        w = self.win
        ext = st.pop("fmt")
        if not path:
            start = os.path.join(str(w.prefs.value("export_folder", "")) or w.default_export_dir(),
                                 f"{file_stem(w.project.name)}_{file_stem(w.view or 'view')[:24]}"
                                 f"_light_sweep.{ext}")
            path, _ = QFileDialog.getSaveFileName(self, "Save the light sweep", start,
                                                  "GIF (*.gif)" if ext == "gif" else "MP4 video (*.mp4)")
            if not path:
                return ""
        root, e = os.path.splitext(path)
        if e.lower() not in (".gif", ".mp4"):
            path = f"{root}.{ext}"
        keep_limit(w, st["max_mb"])
        self.views.keep()
        self.accept()
        return w.write_light_sweep(path, **st)
