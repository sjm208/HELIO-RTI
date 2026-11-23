"""Videos and GIFs kept under a size, for slides: viewer3d.write_video's *max_mb*,
and the File size row of Save the light sweep and Record video.

    python tests/test_video_size.py
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def frames(n, w, h):
    """Moving detail that does not compress well: the hard case."""
    import numpy as np
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, (h, w, 3), np.uint8)
    return [np.roll(base, 7 * i, axis=1) for i in range(n)]


def main():
    from PIL import Image
    from helio.application import viewer3d as V
    out = tempfile.mkdtemp(prefix="helio_vsize_")
    print("Kept under a size")
    fr = frames(120, 960, 720)                         # 6 s at 20 a second
    g = os.path.join(out, "a.gif")
    V.write_video(fr, g, 20)
    big = os.path.getsize(g) / 1e6
    V.write_video(fr, g, 20, max_mb=20)
    mb = os.path.getsize(g) / 1e6
    im = Image.open(g)
    n = getattr(im, "n_frames", 1)
    dur = im.info.get("duration", 0) * n / 1000.0
    check(f"a GIF of {big:.0f} MB comes in under 20 MB ({mb:.1f} MB, {im.size}, {n} frames)",
          mb <= 20 and big > 20)
    check(f"…and is as long as before ({dur:.1f} s)", abs(dur - 6.0) < 0.3)
    fr = frames(240, 1920, 1080)                       # 8 s at 30
    keep = V._ffmpeg
    for how, ff in (("with ffmpeg", keep), ("without ffmpeg", lambda: "")):
        if how == "with ffmpeg" and not keep():
            print("  --   no ffmpeg here: skipped")
            continue
        V._ffmpeg = ff
        p = os.path.join(out, f"b_{how[:4]}.mp4")
        V.write_video(fr, p, 30, max_mb=5)
        mb = os.path.getsize(p) / 1e6
        import cv2
        c = cv2.VideoCapture(p)
        nf, w = int(c.get(cv2.CAP_PROP_FRAME_COUNT)), int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        c.release()
        check(f"an MP4 {how} comes in under 5 MB ({mb:.1f} MB, {w} px wide, {nf} frames)",
              mb <= 5 and nf == 240)
    V._ffmpeg = keep
    p = os.path.join(out, "c.mp4")
    V.write_video(frames(30, 320, 240), p, 30)
    check("no limit: as before", os.path.getsize(p) > 0)

    print("\nPlays round more than once")
    import cv2

    def count(p):
        c = cv2.VideoCapture(p)
        n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
        c.release()
        return n
    p = os.path.join(out, "r.mp4")
    V.write_video(frames(30, 320, 240), p, 30, repeat=3)
    check(f"an MP4 three times round has three times the frames ({count(p)})", count(p) == 90)
    p = os.path.join(out, "r20.mp4")
    V.write_video(fr, p, 30, max_mb=5, repeat=3)
    mb = os.path.getsize(p) / 1e6
    check(f"…and, three times round, still under the limit ({mb:.1f} MB, {count(p)} frames)",
          mb <= 5 and count(p) == 720)
    g = os.path.join(out, "r.gif")
    V.write_video(frames(20, 160, 120), g, 20, repeat=3)
    check("a GIF is not repeated: it loops by itself", Image.open(g).n_frames == 20
          and Image.open(g).info.get("loop") == 0)

    print("\nThe dialogs")
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    w = hw.MainWindow(appearance="dark")
    from helio_qt.sweep_dialog import LENGTHS, SweepDialog
    d = SweepDialog(w)
    check("the sweep: slower turns, and its speed shown", max(LENGTHS) >= 30
          and "° a second" in d.speed.text())
    check("…a File size row, with no limit at first", d.settings()["max_mb"] == 0)
    check("…Plays round, greyed for a GIF", not d.repeat.isEnabled() and d.settings()["repeat"] == 1)
    d.fmt.buttons["mp4"].click()
    d.repeat.setCurrentIndex(d.repeat.findData(4))
    check("…and offered for an MP4 (4 times)", d.repeat.isEnabled() and d.settings()["repeat"] == 4)
    d.limit.set_current("20")
    from helio_qt.record_dialog import keep_limit
    keep_limit(w, d.settings()["max_mb"])
    d.close()
    from helio_qt.record_dialog import RecordDialog
    r = RecordDialog(w, None)
    check("Record video remembers the limit chosen last (20 MB)", r.settings()["max_mb"] == 20)
    r.repeat.setCurrentIndex(r.repeat.findData(2))
    check("Record video: an MP4 twice round", r.repeat.isEnabled() and r.settings()["repeat"] == 2)
    r.fmt.buttons["gif"].click()
    check("…a GIF once (it loops)", not r.repeat.isEnabled() and r.settings()["repeat"] == 1)
    r.close()
    w.close()
    app.processEvents()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
