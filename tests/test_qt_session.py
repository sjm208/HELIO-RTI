"""Sessions in the new interface open and save classic .helio files (SES-001).

Offscreen.  Opens tests/fixtures/session_golden.helio (written by the classic
app) and checks the solve settings, scale and view settings arrive; saves it
back and checks every choice the new app does not have yet (bookmarks,
publication format, sharpening, capture metadata) is carried through
unchanged, in the classic key order; opens a session that names real images
and checks they load.

    python tests/test_qt_session.py
"""
import json
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["HELIO_QT_SETTINGS"] = os.path.join(tempfile.mkdtemp(prefix="helio_qt_"), "prefs.ini")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.make_golden import test_images  # noqa: E402
GOLDEN = os.path.join(ROOT, "tests", "fixtures", "session_golden.helio")

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
    print("=" * 60)
    print("Sessions in the new interface")
    print("=" * 60)
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    import helio_qt.window as hw
    hw.QMessageBox.warning = staticmethod(lambda *a, **k: None)
    hw.QMessageBox.question = staticmethod(   # the next capture: don't save
        lambda *a, **k: hw.QMessageBox.StandardButton.Discard)
    w = hw.MainWindow(appearance="dark")
    w.show()

    print("\nOpen a classic session")
    check("Save needs something to save", w.registry.reason("file.save") != "")
    w.load_session_file(GOLDEN)
    app.processEvents()
    s = w.project.settings
    check("the solve settings arrive", s.normal_engine == "Shadow-Rejection LS"
          and s.depth_engine == "Frankot-Chellappa" and s.flat_correction)
    check("…except Bumps ↔ holes, which Pro 3 does not offer: off, and said so",
          not s.invert_relief and "Bumps ↔ holes" in w.project.notice, w.project.notice)
    check("the scale arrives", w.project.scale_mm_per_px == 0.042)
    check("the view and its settings arrive", w.view == "Exaggerated Shading"
          and w.engine_settings["Standard PBR"]["gain"] == 0.4
          and w.engine_settings["Exaggerated Shading"]["lx"] == -0.3)
    check("images that are not where the session says are reported, not guessed",
          w.project.images == [] and "not where the session says" in w.project.notice
          and w.status_text.text() == "Session")

    print("\nSave it back")
    out = os.path.join(tempfile.mkdtemp(), "back.helio")
    w.write_session(out)
    a, b = json.load(open(out)), json.load(open(GOLDEN))
    check("the classic keys, in the classic order", list(a)[:len(b)] == list(b), list(a))
    same = [k for k in ("bookmarks", "sharpen_params", "pub_format", "pub_bg", "capture_metadata",
                        "crop_box", "drawn_scale_px", "calculated_ppi",
                        "object_id", "bake_depth_pct", "watertight", "sketchfab")
            if a[k] != b[k]]
    check("choices the new app does not have yet are carried through unchanged", not same, same)
    check("a height scale measured on the display depth is not carried through (it is not to scale)",
          not a.get("z_mm_per_unit") and "height_basis" not in a, (a.get("z_mm_per_unit"), b.get("z_mm_per_unit")))
    check("its own choices are written", a["normal_engine"] == "Shadow-Rejection LS"
          and a["render_engine"] == "Exaggerated Shading" and a["mm_per_px"] == 0.042
          and a["engine_settings"]["Standard PBR"]["gain"] == 0.4)
    check("bake depth still in percent", a["bake_depth_pct"] == 7.5)
    from helio.io import session as S
    check("the classic codec reads what the new app wrote", S.read(out).normal_engine == "Shadow-Rejection LS")

    print("\nA session that names real images")
    tdir = test_images(PASS, FAIL)
    imgs = sorted(os.path.join(tdir, f) for f in os.listdir(tdir)
                  if f.lower().endswith(".jpeg"))
    real = os.path.join(tempfile.mkdtemp(), "real.helio")
    S.write(real, S.Session(object_id="manuscript", plate_paths=imgs,
                            last_image_folder=os.path.dirname(imgs[0]),
                            normal_engine="Uncalibrated PCA"))
    w.load_session_file(real)
    app.processEvents()
    check("its images load, in order", w.project.images == imgs)
    check("and the next step says what a session cannot carry: the lights",
          "light directions" in w.project.notice and w.current == "Capture")
    w.registry.trigger("file.save")
    check("Save writes to the file it came from", S.read(real).object_id == "manuscript"
          and S.read(real).plate_paths == imgs)

    print("\nThe flash corrections travel with the session (CAL-003, CAL-004)")
    w.project.update_settings(even_flash=True, flash_ref="10,20,110,120",
                              even_flat=True, flat_areas="5,5,60,400;900,5,955,400")
    corr = os.path.join(tempfile.mkdtemp(), "corr.helio")
    w.write_session(corr)
    raw_c = json.load(open(corr))
    check("the grey patch and the flat areas are written to the session",
          raw_c.get("flash_ref") == "10,20,110,120" and raw_c.get("even_flash") is True
          and raw_c.get("flat_areas") == "5,5,60,400;900,5,955,400" and raw_c.get("even_flat") is True)
    w.project.update_settings(even_flash=False, flash_ref="", even_flat=False, flat_areas="")
    w.load_session_file(corr)
    app.processEvents()
    s2 = w.project.settings
    check("…and come back when it is opened, the grey patch marked",
          s2.even_flash and s2.flash_ref == "10,20,110,120" and s2.even_flat
          and s2.flat_areas == "5,5,60,400;900,5,955,400" and w.grey_ref.get("state") == "marked",
          (s2.flash_ref, s2.flat_areas, w.grey_ref))
    w.project.update_settings(near_field=True, near_field_from="centre")
    nf = os.path.join(tempfile.mkdtemp(), "nf.helio")
    w.write_session(nf)
    w.project.update_settings(near_field=False, near_field_from="ball")
    w.load_session_file(nf)
    app.processEvents()
    check("near-field measured from the centre is kept in the session, and comes back",
          json.load(open(nf)).get("near_field_from") == "centre"
          and w.project.settings.near_field and w.project.settings.near_field_from == "centre")
    pw = w.workspaces["Process"]
    pw.refresh()
    check("Process shows where the lamps are measured from",
          not pw.nf_from.isHidden() and "middle" in pw.nf_note.text(), pw.nf_note.text())
    w.project.update_settings(near_field=False, near_field_from="ball")
    plain = os.path.join(tempfile.mkdtemp(), "plain.helio")
    S.write(plain, S.Session(object_id="x"))
    check("a session without them writes none of their keys (the classic file is unchanged)",
          not any(k in json.load(open(plain)) for k in ("flash_ref", "even_flash", "flat_areas", "even_flat",
                                                         "near_field_from")))

    print("\nThe lights travel with the session (SES-003)")
    import time
    w.registry.trigger("capture.sphere")
    t0 = time.time()
    while w.project.busy and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    lights = w.project.lights.copy()
    lit = os.path.join(tempfile.mkdtemp(), "lit.helio")
    w.write_session(lit)
    check("Save writes the sphere's lights", S.read(lit).light_source == "sphere"
          and (S.read(lit).lights() == lights).all())
    w2 = hw.MainWindow(appearance="dark")
    w2.show()
    w2.load_session_file(lit)
    app.processEvents()
    check("a fresh window opening it gets the same lights, and Process is ready at once",
          w2.project.lights_ready() and (w2.project.lights == lights).all()
          and not w2.registry.reason("process.run") and "light directions" in w2.project.notice
          and w2.primary.text() == "Continue to Process →")
    old = os.path.join(tempfile.mkdtemp(), "old.helio")
    S.write(old, S.Session(object_id="old", plate_paths=imgs))
    w2.load_session_file(old)
    app.processEvents()
    check("an older session without lights says so, and asks for the sphere",
          w2.project.lights is None and "no light directions" in w2.project.notice)
    w2.close()

    w.close()
    app.processEvents()
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    print("=" * 60)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
