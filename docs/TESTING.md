# How to test Helio

A self-contained guide: how to check the whole program without clicking
through it by hand, what the tests cover, and the traps that have made a green
run hide a broken program before.

## 1. What is being tested

Helio is a macOS (and Windows, from source) application for photometric
stereo and RTI. It takes photographs of one object, each lit from a different
direction, solves the surface's direction at every pixel (the normals),
integrates them into a height map, shows the surface relit and in 3D, measures
it, and publishes it as images, meshes, web viewers, GIS rasters and archives.

- `helio/` is the engine: `core/` (solvers, lights, images), `application/`
  (the solve pipeline, exports, publishing), `io/`, `domain/` (settings,
  registries). It has no interface code.
- `helio_qt/` is the interface (PySide6): the window, five workspaces
  (Capture, Process, Inspect, Measure, Publish), the 3D relief.
- `helio_core_rs/` holds optional Rust kernels for the heaviest arithmetic.
  The NumPy code beside them stays as the reference they are tested against.

## 2. Run everything

```bash
./run_tests.sh              # every test, with the Rust kernels and again without
./run_tests.sh numpy        # the NumPy kernels only
./run_tests.sh rust         # the Rust kernels only (fails if they are not built)
(cd helio_core_rs && cargo test)
```

Each test script prints `ok` or `FAIL` per check, then
`Results: N passed, M failed`. Qt tests run offscreen, each with scratch
preferences, so a run never touches yours. One script on its own:

```bash
HELIO_QT_SETTINGS=$(mktemp -d)/q.ini HELIO_PREFS=$(mktemp -d)/p.json \
  QT_QPA_PLATFORM=offscreen python3 tests/test_qt_views.py
```

The tests that need real photographs open the capture named by
`HELIO_TEST_IMAGES` (a folder of photographs with a chrome ball), and skip
themselves when there is none.

### The whole program, the way a person uses it

```bash
python3 -m helio_qt.fullcheck                               # a made capture, everything
python3 -m helio_qt.fullcheck --images DIR --lp lights.lp   # your own photographs
python3 -m helio_qt.fullcheck --quick
```

It runs every setting and solver, every test, then the real window: every
view, every export (each file opened and checked), every menu command, every
workspace, and screenshots to look at. It writes `REPORT.md` with a block to
paste back.

### The suite, by area

| Files | Cover |
|---|---|
| `tests/test_engines.py`, `tests/test_robust_ps.py`, `tests/test_precision_ps.py`, `tests/test_science.py` | The solvers and shading engines, against known surfaces |
| `tests/test_no_ball.py`, `tests/test_flat_objects.py`, `tests/test_near_field.py`, `tests/test_flash.py` | Lights found from the photographs; flat objects under near, unequal or hand-held lamps, with the truth known |
| `tests/test_sphere.py`, `test_lp_*.py`, `tests/test_light_positions.py`, `test_dome_*.py` | Chrome balls, light files, domes and lamp positions |
| `tests/test_qt_capture.py`, `tests/test_qt_dome.py`, `tests/test_qt_process_tools.py`, `tests/test_qt_precision.py` | Capture and Process in the window |
| `tests/test_qt_views.py`, `tests/test_region_views.py`, `tests/test_view_controls.py`, `test_live_*.py` | Every view, its controls, full-size regions |
| `tests/test_qt_3d.py`, `test_relief_*.py`, `tests/test_rti_3d.py`, `tests/test_mesh.py` | The 3D relief, meshes and their exports |
| `tests/test_qt_measure.py`, `test_measure_*.py`, `tests/test_scale.py` | Scale, distances, heights, profiles |
| `tests/test_qt_saves.py`, `tests/test_quick_exports.py`, `tests/test_plate.py`, `test_sheet_*.py`, `tests/test_masters.py` | Pictures, quick exports, the publication sheet and plate, archival masters |
| `tests/test_geotiff.py`, `tests/test_gis_marks.py` | GIS rasters and the DEM package |
| `tests/test_offline_viewers.py`, `tests/test_open_viewers.py`, `test_viewer_*.py`, `tests/test_qr_share.py` | Web viewers and the phone view |
| `tests/test_qt_menus.py` | Every menu item no other test reaches, chosen from its menu; fails if any menu item goes untested |
| `tests/test_qt_session.py`, `tests/test_qt_next_capture.py`, `tests/test_qt_batch.py`, `tests/test_jobs.py` | Sessions, one capture after another, batch |
| `tests/test_memory_plan.py`, `tests/test_depth_memory.py`, `test_work_scale*.py` | The memory plan on machines of every size |
| `test_rust_*.py`, `tests/test_render_speed.py`, `tests/test_export_speed.py` | The Rust kernels against NumPy, and speed limits |
| `tests/test_gui_audit.py` | The interface hangs together: every command has a handler and a unique shortcut, every button is connected |

## 3. The built app checks itself

`./build_helio3.sh` makes no disk image unless the built app passes its own
test:

```bash
/Applications/Helio.app/Contents/MacOS/Helio --selftest "$(mktemp -d)"
```

It solves a known surface and writes exports with the bundle's own
interpreter and libraries, then prints `PASSED: 9 of 9 steps · build <commit>`.
Run it on the installed copy after an install: tests of the source prove
nothing about the copy that is launched.

## 4. Traps that make tests lie

- **Real captures are read-only.** Test against copies, or only read them.
  Write results to a temporary folder.
- **Don't edit source while a build runs.** PyInstaller packages the files as
  they are on disk at that moment.
- **The 3D view needs a screen to draw on.** Offscreen, `gl_ok` is false, and
  the 3D tests check only what works without it. Check 3D pictures on the real
  window.
- **A process pool re-runs the script.** A test script that starts one needs
  `if __name__ == "__main__":`.
- **Call `sys.stdout.flush()` before `os._exit`,** or the last lines of output
  are lost and the run looks like a crash.
- **Never regex-rename an identifier in a file that builds another language.**
  The exporters build HTML in strings: renaming a variable `html` with a regex
  also rewrote `</html>` and `index.html`, and six exports silently wrote
  broken pages while every test stayed green. Rename from the code's syntax
  tree, or open the files the exports write.
- **Test an approximation across its whole range.** Standard PBR's lookup
  table is exact only while gain is at most 1; at the default gain the error
  was invisible, at gain 5 it was 48/255. `tests/test_engines.py` sweeps the range
  and checks that the table declines where it would be wrong.
- **Test with the truth known, and with the objects people photograph.** A
  curved benchmark (DiLiGenT) says little about flat bindings and pages, where
  what goes wrong is the broad shape. `tests/test_flat_objects.py` renders a flat
  binding as a camera records it (8-bit sRGB) under realistic lamps and checks
  the solve against the truth.
- **Fixtures must be realistic in size.** Some analyses fail on tiny images
  (96 × 128) in ways that hide everything downstream; use at least 384 × 512
  unless the test is about small inputs (`tests/test_small_inputs.py`).

## 5. What automation can and cannot prove

**Can:** that nothing crashes; that every view returns finite output; that
exports write files other programs can open; that a solve matches a known
surface; that behaviour has not changed since a known-good state.

**Cannot:** that a picture is right for a reader the first time. Whether a
view shows the tooling of a binding clearly is a human judgement, needed once
per feature; after that, the tests keep it from changing unnoticed.

## 6. Environment

Install from `requirements.txt` into the folder's own `.venv` (the README
explains). Checked with Python 3.12, NumPy 2.5, OpenCV 5.0, SciPy 1.18 and
PySide6 6.12. `rasterio` is optional: the GeoTIFF writer falls back to OpenCV,
and `tests/test_geotiff.py` covers both.

If `import cv2` fails with `numpy.core.multiarray failed to import`, OpenCV was
built against NumPy 1 and NumPy 2 is installed. Upgrade OpenCV
(`pip install --upgrade "opencv-contrib-python>=4.10"`); do not downgrade
NumPy, which the code needs at version 2.

## 7. Checks by eye, now and then

1. **Launch it** (`Run Helio.command`, or the installed app): no error on
   start, the five workspaces there.
2. **Light direction.** Drag the light north in Inspect; north-facing slopes
   should brighten. A sign error here passes every automated test.
3. **One real capture end to end:** open, find the ball (or No ball), solve,
   look at the relief under a raking light, save a mesh and open it.
4. **The 3D relief** on a real screen: turn it, light it, save a picture.
