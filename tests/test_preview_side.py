"""The preview fitted to the screen, and the section card folded out of the way.

* set_preview_side: the screen's own pixels on a machine with 24 GB or more,
  never under 2,500 and never over 4,096; 2,500 on a smaller machine;
  HELIO_PREVIEW_SIDE overrides; proxy_dims follows it.
* The 3D panel's mesh stays at one point a pixel under its limit when the
  preview is larger than that limit allows (built from the full size, not the
  preview stepped down to every second point).
* SectionCard: Fold leaves only the head line; Unfold brings the chart back.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + ("" if ok else f"  {detail}"))


def main():
    import helio.application.pipeline as P
    GB = 2 ** 30
    os.environ.pop("HELIO_PREVIEW_SIDE", None)
    keep = P._preview_side
    try:
        print("The preview's size")
        P._preview_side = None
        check("nothing set: 2,500", P.preview_side() == 2500, P.preview_side())
        check("a 16 GB machine keeps 2,500", P.set_preview_side(3456, 16 * GB) == 2500)
        check("a 96 GB machine on a 3456 px screen: 3,456", P.set_preview_side(3456, 96 * GB) == 3456)
        check("proxy_dims follows it", P.proxy_dims(5504, 8256) == (2304, 3456), P.proxy_dims(5504, 8256))
        check("never upsampled", P.proxy_dims(1000, 1500) == (1000, 1500))
        check("a 6K screen: held to 4,096", P.set_preview_side(6016, 96 * GB) == 4096)
        check("a small screen: never under 2,500", P.set_preview_side(1440, 96 * GB) == 2500)
        check("no screen known: 2,500", P.set_preview_side(None, 96 * GB) == 2500)
        os.environ["HELIO_PREVIEW_SIDE"] = "3000"
        check("HELIO_PREVIEW_SIDE overrides", P.preview_side() == 3000)
        os.environ["HELIO_PREVIEW_SIDE"] = "99999"
        check("... held to 4,096", P.preview_side() == 4096)
    finally:
        os.environ.pop("HELIO_PREVIEW_SIDE", None)
        P._preview_side = keep

    print("The section card folds")
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    import tempfile
    os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(), "s.ini")
    from helio_qt.section_card import SectionCard
    c = SectionCard()
    c.set_folded(False)
    c.show()
    app.processEvents()
    h_open = c.sizeHint().height()
    c.set_folded(True)
    app.processEvents()
    check("folded: the chart is hidden, the head stays", not c.body.isVisible() and c.title.isVisible()
          and c.btn_fold.text() == "Unfold")
    check("folded: far shorter", c.sizeHint().height() < h_open / 2, (c.sizeHint().height(), h_open))
    check("a new card opens folded, as last left", SectionCard().btn_fold.isChecked())
    c.btn_fold.click()
    app.processEvents()
    check("Unfold brings the chart back", c.body.isVisible() and c.btn_fold.text() == "Fold")

    print("The 3D panel's mesh under a larger preview")
    import time
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    from tools import make_golden as G
    G.H, G.W = 3400, 5000
    tmp = tempfile.mkdtemp(prefix="helio_pside_")
    paths, L = G.render_stack(tmp, n_lights=5)
    os.environ["HELIO_PREVIEW_SIDE"] = "4096"            # 4096 x 2785: 11.4 million, over the 7 million limit
    try:
        w = hw.MainWindow(appearance="dark")
        w.resize(1440, 900)
        w.show()
        w.open_paths([tmp])
        w.project.set_lights(L, "file:golden.lp")
        w.registry.trigger("process.run")
        t0 = time.time()
        while w.project.busy and time.time() - t0 < 600:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()
        D = w.render_data
        check("the preview is 4,096 px", D is not None and D.pw == 4096, None if D is None else D.pw)
        ins = w.workspaces["Inspect"]
        w.go_workspace("Inspect")
        from helio_qt.relief3d import ReliefView
        A = ins._relief_arrays(D)
        n = A["depth"].shape[0] * A["depth"].shape[1]
        check("the mesh's surface is made under the limit, from the full size",
              ReliefView.MAX_MESH_POINTS * 0.95 < n <= ReliefView.MAX_MESH_POINTS, (A["depth"].shape, n))
        check("its normals match it", A["normals"].shape[:2] == A["depth"].shape[:2])
        w.project.busy = False
        w.close()
    finally:
        os.environ.pop("HELIO_PREVIEW_SIDE", None)

    print(f"\n{'=' * 60}\nResults: {len(PASS)} passed, {len(FAIL)} failed out of {len(PASS) + len(FAIL)} tests\n{'=' * 60}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
