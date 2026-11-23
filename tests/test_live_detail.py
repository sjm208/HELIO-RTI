"""Inspect draws the part on screen at full resolution once the preview is enlarged.

Runs the Qt app offscreen on a 5000 x 3400 stack (its preview is 2500 px, so zoomed in the
preview really is enlarged):

* zoomed out, nothing extra is drawn; zoomed in, the visible part is drawn from the full-size
  data, and it equals an independent full-size render of the same box;
* it is sharper than the enlarged preview;
* it follows the light, and is dropped while a drag is too slow for it;
* mirrored views place it correctly; comparing, photographs and switching it off remove it;
* the loupe and the detail are skipped (not fitted) when a tight machine dropped the PTM.

    python tests/test_live_detail.py
"""
import os
import sys
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
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
    import numpy as np
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    import helio_qt.inspect_ws as iw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    from tools import make_golden as G
    G.H, G.W = 3400, 5000
    tmp = tempfile.mkdtemp(prefix="helio_detail_")
    paths, L = G.render_stack(tmp, n_lights=5)
    w = hw.MainWindow(appearance="dark")
    w.resize(1440, 900)
    w.show()

    def idle(limit=600.0):
        t0 = time.time()
        while w.project.busy and time.time() - t0 < limit:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    def settle(ms=700):
        t0 = time.time()
        while time.time() - t0 < ms / 1000.0:
            app.processEvents()
            time.sleep(0.01)
        ins.wait_render()
        for _ in range(3):
            app.processEvents()
            ins.wait_render()

    w.open_paths([tmp])
    w.project.set_lights(L, "file:golden.lp")
    w.project.update_settings(photo_tone="linear")
    w.registry.trigger("process.run")
    idle()
    ins = w.workspaces["Inspect"]
    w.go_workspace("Inspect")
    app.processEvents()
    ins.wait_render()
    vp = ins.viewport
    D = w.render_data
    check("solved at 5000 x 3400 with a smaller preview", w.project.solved and D.w == 5000 and D.pw < D.w,
          (D.w, D.pw))

    print("Zoomed out")
    vp.fit()
    settle()
    check("fitted: no detail drawn", not vp.detail.isVisible())

    print("Zoomed in")
    w.set_view("Standard PBR")
    settle()
    vp.resetTransform()
    vp.scale(2.0, 2.0)                              # two screen pixels per image pixel
    vp.centerOn(2500, 1700)
    vp._fitted = False
    vp.viewChanged.emit()
    settle()
    check("the part on screen is drawn at full size", vp.detail.isVisible())
    px = vp.detail.pixmap()
    r = vp.visible_rect()
    check("it covers what is visible", vp.detail.pos().x() <= r.left() + 1 and
          vp.detail.pos().x() + px.width() >= r.right() - 1 and vp.detail.pos().y() <= r.top() + 1
          and vp.detail.pos().y() + px.height() >= r.bottom() - 1, (vp.detail.pos(), px.size(), r))
    x1, y1 = int(vp.detail.pos().x()), int(vp.detail.pos().y())
    box = (x1, y1, x1 + px.width(), y1 + px.height())
    from helio.application.viewing import apply_transforms, screen_image
    import copy
    ref = apply_transforms(screen_image(w.engine, D, dict(ins.settings()), copy.deepcopy(w.display),
                                        use_proxy=False, crop=box, transforms=False), w.display)
    shown = px.toImage().convertToFormat(px.toImage().Format.Format_RGB888)
    a = np.frombuffer(shown.constBits(), np.uint8).reshape(shown.height(), shown.bytesPerLine())[:, :shown.width() * 3]
    a = a.reshape(shown.height(), shown.width(), 3)
    check("it equals an independent full-size render of the same box", a.shape == ref.shape and
          np.abs(a.astype(int) - ref.astype(int)).max() <= 1, (a.shape, ref.shape))
    prev = ins._last_image
    sy, sx = prev.shape[0] / D.h, prev.shape[1] / D.w
    up = prev[int(box[1] * sy):int(box[3] * sy), int(box[0] * sx):int(box[2] * sx)]
    import cv2
    up = cv2.resize(up, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def hf(x):
        g = x.astype(np.float32).mean(-1)
        return float(np.abs(g - cv2.GaussianBlur(g, (0, 0), 2)).mean())
    check("and carries more fine detail than the enlarged preview", hf(a) > hf(up), (hf(a), hf(up)))

    print("It follows the light")
    before = a.copy()
    ins._dome_moved(0.6, 0.0, 0.8, False)
    settle()
    check("after the light moves the detail is drawn again", vp.detail.isVisible())
    px2 = vp.detail.pixmap().toImage()
    b = np.frombuffer(px2.convertToFormat(px2.Format.Format_RGB888).constBits(), np.uint8)
    check("with the new light", b.size == before.size and not np.array_equal(b.reshape(before.shape), before))
    ins._detail_ms[w.view] = 5000.0
    ins.render_view(final=False)
    check("a drag too slow for it hides the detail", not vp.detail.isVisible())
    ins.render_view(final=True)
    settle()
    check("and the frame the drag ends on brings it back", vp.detail.isVisible())

    print("Mirrored")
    w.display.flip_h = True
    ins.render_view(final=True)
    settle()
    check("a mirrored view still gets its detail", vp.detail.isVisible())
    from helio.application.viewing import image_point
    pm = vp.detail.pixmap().toImage()
    pm = pm.convertToFormat(pm.Format.Format_RGB888)
    m = np.frombuffer(pm.constBits(), np.uint8).reshape(pm.height(), pm.bytesPerLine())[:, :pm.width() * 3]
    m = m.reshape(pm.height(), pm.width(), 3)
    sx1, sy1 = int(vp.detail.pos().x()), int(vp.detail.pos().y())
    ax, ay = image_point(sx1, sy1, D.w, D.h, w.display)
    bx, by = image_point(sx1 + pm.width(), sy1 + pm.height(), D.w, D.h, w.display)
    mbox = (int(min(ax, bx)), int(min(ay, by)), int(max(ax, bx)), int(max(ay, by)))
    mref = apply_transforms(screen_image(w.engine, D, dict(ins.settings()), copy.deepcopy(w.display),
                                         use_proxy=False, crop=mbox, transforms=False), w.display)
    check("and it is the mirror of the right part of the picture", m.shape == mref.shape and
          np.abs(m.astype(int) - mref.astype(int)).max() <= 1, (m.shape, mref.shape, mbox))
    w.display.flip_h = False
    ins.render_view(final=True)
    settle()

    print("Removed when it should be")
    ins.chk_detail.setChecked(False)
    check("switched off", not vp.detail.isVisible())
    ins.chk_detail.setChecked(True)
    settle()
    check("switched on again", vp.detail.isVisible())
    vp.fit()
    settle()
    check("zooming back out removes it", not vp.detail.isVisible())
    vp.resetTransform(); vp.scale(2.0, 2.0); vp.centerOn(2500, 1700); vp.viewChanged.emit()
    settle()
    w.set_view("Specular Enhancement")
    settle()
    check("another view: drawn for that view", vp.detail.isVisible())
    D.ptm_deferred, saved = True, D.ptm_coeffs
    D.ptm_coeffs = None
    check("a PTM view on a tight machine is not fitted for it", ins._needs_refit() and ins._detail_wanted() is None)
    D.ptm_deferred, D.ptm_coeffs = False, saved

    print("Every view: the region equals the preview of it, or is tone-matched, or is not drawn")
    from helio.application.viewing import DETAIL_SKIP, DETAIL_TONE_MATCH, Display, detail_image, screen_image
    from helio.domain.registry import RENDER_ENGINES, default_engine_settings
    import cv2
    eng = default_engine_settings()
    bx = (1800, 1200, 3000, 2000)
    sc = D.pw / D.w
    worst = []
    for e in RENDER_ENGINES:
        if e in DETAIL_SKIP:
            continue
        st = dict(eng[e])
        st.update(lx=-0.5, ly=-0.3, lz=0.8)
        pv = screen_image(e, D, st, Display())
        dt = detail_image(e, D, st, Display(), bx)
        pc = pv[int(bx[1] * sc):int(bx[3] * sc), int(bx[0] * sc):int(bx[2] * sc)]
        dd = cv2.resize(dt, (pc.shape[1], pc.shape[0]), interpolation=cv2.INTER_AREA)
        m = float(np.abs(dd.astype(float) - pc.astype(float)).mean())
        limit = 6.0 if e in DETAIL_TONE_MATCH else 1.5
        if m > limit:
            worst.append(f"{e}: {m:.1f}")
    check("no view that is drawn this way jumps in look when its sharp part appears", not worst, "; ".join(worst))
    check("the views left out are the ones measured to differ", all(e in RENDER_ENGINES for e in DETAIL_SKIP)
          and "Standard PBR" not in DETAIL_SKIP and "Image Quotient (IQ)" in DETAIL_TONE_MATCH)

    print(f"\nResults: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    return FAIL


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
