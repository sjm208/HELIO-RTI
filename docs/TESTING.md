# How to test Helio and Helio Classic

A self-contained guide. Assumes no prior knowledge of the codebase or of any
previous work on it. Written so that an engineer — or another AI — can verify
the whole program without opening the GUI and clicking through it by hand.

---

> **Two apps, one engine.** Sections 1–6 below were written for Helio Classic
> (HELIO Pro 2.6, CustomTkinter) and still hold for it and for the shared
> engine in `helio/`. **Helio 3.0** (PySide6, `helio_qt/`) has its own suite:
> see [section 7](#7-helio-30-helio_qt).

## 1. What HELIO Pro is

A macOS desktop application for Reflectance Transformation Imaging (RTI) and
photometric stereo. You give it a stack of photographs of one object, each lit
from a different direction. It solves the surface normals, integrates them into
a depth map, renders the surface under a re-positionable virtual light, and
exports meshes, GeoTIFFs, web viewers and archival packages.

* `Helio_Pro.py` — ~32,000 lines, one `RTIViewerPro(ctk.CTk)` class with ~350
  methods. This is the application.
* `helio/` — extracted library: `core/` (solvers, shading engines, image
  processing), `io/` (exporters, loaders), `render/` (VisPy 3D viewer),
  `utils/`.
* `main.py` — entry point. GUI by default, `--headless` for batch rendering.
* GUI toolkit is **CustomTkinter** (on top of Tk).

---

## 2. Run everything

```bash
cd ~/Desktop/Helio_NoTorch
./run_all_tests.sh
```

~380 checks, about 3 minutes. Use a specific interpreter with
`PYTHON=/path/to/python ./run_all_tests.sh`.

Expected output — anything else is a failure:

```
── compile ──                    compile OK
── pyflakes ──                   clean
── tests/test_engines.py ──            195 passed, 0 failed
── tests/test_geotiff.py ──            17 passed, 0 failed
── tests/test_memory.py ──             7 passed, 0 failed
── tests/test_golden.py ──             40 passed, 0 changed, 0 gone, 0 new
── tests/test_real_capture.py ──       12 passed, 0 failed
── tests/test_interactions.py ──       29 passed, 0 failed
── smoke_test.py ──              ALL SMOKE TESTS PASSED  (40/40 engines)
```

### The suite, file by file

| File | Covers |
|---|---|
| `smoke_test.py` | Solvers, all 6 depth integrators, full UI build, end-to-end processing on two engine combinations, all 40 render engines, array release |
| `tests/test_engines.py` | 195 shading-engine checks at full and proxy resolution, plus graceful handling of PTM shape mismatches and the Standard PBR lookup-table accuracy contract |
| `tests/test_geotiff.py` | The GeoTIFF writer, on **both** the rasterio and the cv2-fallback paths (world files, tags, compression, DEM packages) |
| `tests/test_interactions.py` | UI **callbacks** — panel switchers, tab changes, engine dropdowns, collapsible sections — and that no two captioned controls share a grid cell |
| `tests/test_memory.py` | Proxy sizing, array aliasing, idle release, and that rendering + GIS export still work *after* a release |
| `tests/test_golden.py` | Re-renders all 40 engines from a deterministic surface and diffs against approved references in `tests/golden/` |
| `tests/test_real_capture.py` | Solves the real 8-frame manuscript capture in `test images/` — chrome-sphere auto-detection, unit normals, non-flat depth, detail retention, all engines |
| `tests/test_exports.py` | Runs export methods headlessly and validates their output (~5 min, not in the suite). Now runs in scripted mode: 41 OK · 4 needs input · 5 no output · 0 failures |
| `tests/test_dnd.py` | Drag-and-drop path handling, and that a dropped capture routes to 1B or 1C by whether a sphere is really found |
| `tests/test_provenance.py` | The provenance record: source hashes, light vectors, versions; a one-byte source change must be detectable |
| `tests/test_relight_export.py` | The relight / OpenLIME package, compared key-for-key against relight-cli's own output |
| `tests/test_session_controls.py` | Undo/redo, staged progress and cancel, scripted exports |
| `tests/test_small_inputs.py` | Terrain analysis from 8x8 up; must never upsample its source |
| `tests/test_lp_roundtrip.py` | CHI `.lp` files parsed by a deliberately strict reader standing in for RTIViewer |
| `tests/test_headless.py` | The `--headless` batch interface, driven as a real subprocess — the only way to catch argument-parsing bugs |
| `tests/test_sphere_ui.py` | The sphere calibrator's fine-tune controls, driven by calling each button's command |
| `tools/ui_audit.py` | Not a test — measures widget geometry to find clipped captions and inconsistent sizing. Calibrate before trusting it (see its comments). |
| `memprofile.py` | Not a test — prints every ndarray held after a run, with sizes. Use it to see where memory goes. |

---

## 3. Nine traps that make naive tests lie

Read this section before writing any new test. Each of these has already caused
a green test suite to hide a broken program.

### 3.1 `__init__` swallows UI build failures

```python
try:
    self.build_ui()
except Exception as _build_err:
    ...  # shows a message box, then carries on
```

So `RTIViewerPro()` **returns successfully on a half-built UI**. A test that
constructs the window and asserts nothing else proves nothing. Wrap the method
so it re-raises:

```python
_real = Helio_Pro.RTIViewerPro.build_ui
errors = []
def watched(self):
    try:
        return _real(self)
    except Exception as e:
        errors.append(e); raise
Helio_Pro.RTIViewerPro.build_ui = watched
app = Helio_Pro.RTIViewerPro()
assert not errors, errors[0]
```

### 3.2 Building the UI says nothing about callbacks

Callbacks are closures defined inside the build methods. A closure referencing a
name from another scope raises **only when the user clicks** — construction is
clean. Two such bugs (a panel switcher and the tab handler) survived a passing
build test. Exercise callbacks explicitly; `tests/test_interactions.py` shows how:
walk the widget tree, pull `widget.cget("command")`, call it.

### 3.3 Modal dialogs deadlock an automated run

`messagebox.showerror` inside the Tk mainloop blocks forever with no user.
Always patch it, and record what was raised — an unexpected error dialog is
itself a test failure:

```python
class _Box:
    def __getattr__(self, name):
        def _rec(title="", message="", *a, **k):
            DIALOGS.append((name, title, str(message)[:200])); return "ok"
        return _rec
Helio_Pro.messagebox = _Box()
```

Do the same for `filedialog` (return a temp path instead of opening a picker).

### 3.4 Processing is threaded

`run_processing()` starts a background thread and dispatches UI updates through
`self.after()`, so it **needs a live `mainloop()`** — pumping `app.update()` in a
loop is not enough and raises "main thread is not in main loop".

Wait for the *thread*, not for results to appear:

```python
th = getattr(app, "_proc_thread_obj", None)
done = th is not None and not th.is_alive()
```

`normals_full` and `depth_poisson` are set part-way through. Polling on them and
then starting the next run overlaps two runs, and the second run's array-release
step frees arrays the first is still reading. (The app now refuses a concurrent
run, so the second is simply dropped — which will also make a test hang.)

### 3.5 CustomTkinter hides its bindings

`CTkButton.bind()` delegates to internal canvas/label children, so
`widget.bind("<Enter>")` on the button itself always returns `None`. Any test
that uses it to detect "does this control have a tooltip?" asserts something
that can never be true. `ToolTip` registers `widget._helio_tooltip` — check
that instead.

### 3.6 Never regex-rename an identifier in a file that embeds another language

`helio/io/exporters.py` builds HTML in f-strings. Renaming a local variable
`html` with `re.sub(r"\bhtml\b", "html_doc", ...)` also rewrote `</html>`,
`index.html`, a dict key and CSS selectors — six exports silently produced
malformed HTML. pyflakes cannot see inside strings and no other test opened an
exported file, so everything stayed green.

Rename from **AST `Name` node positions** instead; those are code by
construction and can never be string content.

### 3.7 Tk stacks widgets that share a grid cell — it never complains

Two widgets gridded into the same cell both get managed; the later `grid()`
call draws on top. Five export buttons were wired onto cells the archival block
already held and were invisible behind it, while passing every construction and
callback test. Only geometry reveals it — with the tab selected, both buttons
report identical x/y/w/h.

`tests/test_interactions.py` now checks this, but note what it took to make the check
real: restrict to widgets whose parent is a *container* (every CTk control
holds internal children at row 0 col 0 of itself — 819 false hits), restrict to
*captioned* controls (bare frames legitimately share a cell in the `_sw_show`
card stack), and ask for the caption with `cget("text")` — **not**
`"text" in w.keys()`, which is always false on a `CTkButton` and silently
skipped every widget the check existed to test.

### 3.8 An approximation must be tested across the whole parameter range

`Standard PBR` replaces its per-pixel exponent with a 4096-entry lookup table,
which is exact enough to be invisible **only while gain <= 1**. Above that the
exponent falls below 1, the curve has unbounded slope at zero, and the first
bin alone is wrong by up to 48/255 at gain 5 - a value the slider reaches.
Testing at the default gain of 0.15 (error 0.41/255) would have called that
fine. `tests/test_engines.py` therefore sweeps the gain range and asserts both
halves of the contract: accurate where the table is used, and that the table
**declines** where it would not be.

### 3.9 Fixtures must be realistic in size

Use **at least 384×512**. At 96×128 `compute_terrain_analysis` dies inside
`cv2.resize` with `!ssize.empty()`. That is a genuine (unfixed) limitation for
very small inputs, not a test artefact — but it will mask everything downstream
if your fixture triggers it.

---

## 4. Writing a new test

The standard skeleton — synthetic lit hemisphere, process it, assert:

```python
import multiprocessing, tempfile, os
import numpy as np, cv2

def make_stack(tmp, n=8, h=384, w=512):
    """n images of a hemisphere lit from n directions + ground-truth lights."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy, r = w/2, h/2, min(h, w)*0.42
    d  = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    nz = np.sqrt(np.clip(1-(d/r)**2, 0, 1))
    N  = np.dstack([np.where(d<r, (xx-cx)/r, 0.0),
                    np.where(d<r, (yy-cy)/r, 0.0),
                    np.maximum(nz, 0.05)])
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    L, paths = [], []
    for i in range(n):
        a  = 2*np.pi*i/n
        lv = np.array([0.6*np.cos(a), 0.6*np.sin(a), 0.7], np.float32)
        lv /= np.linalg.norm(lv); L.append(lv)
        img = np.clip(N @ lv, 0, 1)*235 + 10
        p = os.path.join(tmp, f"img_{i:02d}.png")
        cv2.imwrite(p, cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR))
        paths.append(p)
    return paths, np.array(L, np.float32)

multiprocessing.set_start_method("spawn", force=True)
import Helio_Pro
Helio_Pro.messagebox = _Box()                    # §3.3

tmp = tempfile.mkdtemp()
paths, L = make_stack(tmp)
app = Helio_Pro.RTIViewerPro(); app.update()
app.data.plate_paths    = list(paths)
app.loaded_lp_matrix    = L.copy()
app.data.current_L_mat  = L.copy()
app.data.normal_engine.set("Linear Least Squares")

def wait(t=0):                                   # §3.4
    th = getattr(app, "_proc_thread_obj", None)
    if th is not None and not th.is_alive() and app.data.normals_full is not None:
        app.quit()
    elif t > 900: app.quit()
    else: app.after(200, lambda: wait(t+1))

app.after(200, lambda: (app.run_processing(), wait()))
app.mainloop()                                   # required
# ... assertions here ...
```

Useful state after a run:

| Attribute | Meaning |
|---|---|
| `app.data.normals_full` | (h, w, 3) float32 surface normals — source of truth |
| `app.depth_poisson` | (h, w) float32 integrated depth |
| `app.data.depth_vis_p` | normalised depth for display |
| `app.data.albedo_rgb_full` | (h, w, 3) uint8 |
| `app.data._svf_full` etc. | full-res terrain maps (**export-only**; may be released) |
| `app.data.*_proxy` | downsampled previews — what the viewport renders from |
| `app.data.h/w`, `app.data.ph/pw` | full and proxy dimensions |
| `app.engines_list` | the 40 render engine names |
| `app.get_shading(h, w, use_proxy, engine_override=…)` | render one engine |

---

## 5. What automation can and cannot prove

**Can:** that nothing crashes; that every render engine returns finite output;
that exports produce genuinely parseable files; that behaviour has not *changed*
since a known-good state; that memory is released and rebuilt correctly.

**Cannot:** that a result is scientifically correct the first time. No test can
tell you a hillshade is revealing the drypoint burr properly. That judgement is
human — but it is only needed **once per feature**, if you capture the approved
result as a reference (§6, Tier 2). After that, every future change is checked
automatically.

---

## 6. Coverage gaps and how to close them

### Tier 1 — export coverage (DONE)

`tests/test_exports.py` exists. It drives all 49 zero-argument export methods with a
processed dataset, redirects their file dialogs into a temp directory, and
validates output by *parsing* it — meshes must load in trimesh, binary STL
triangle counts must match the file length, TIFFs must open, HTML must have a
closing tag and no leaked error text, JSON/XML must be well formed, ZIPs must
pass a CRC check. Exports that stop on a configuration window are reported as
"needs input" rather than failed.

```bash
python3 tests/test_exports.py          # ~5 min; not part of run_all_tests.sh
```

Latest run: **33 OK · 12 needs input · 4 no output · 0 failures.**

* *needs input* — the export opens a configuration window; it cannot be driven
  further without simulating that dialog. Not a failure.
* *no output* — a legitimate unmet precondition (no crop drawn, 3D viewer not
  open, no PoI data, no profile measured).

It is kept out of `run_all_tests.sh` because it takes ~5 minutes. Run it after
touching anything in `helio/io/exporters.py` or any `export_*` method.

Three things the harness must do, all learned the hard way:

1. **Run inside a live `mainloop()`.** Several exports dispatch UI work from a
   worker thread via `after()`, which raises "main thread is not in main loop"
   otherwise.
2. **Neutralise `wait_window` / `grab_set`.** A modal config window with no
   user to dismiss it hangs the sweep forever.
3. **Reset `app._busy` before each export.** One threaded export that does not
   finish leaves the guard set and every later export reports "HELIO Busy".

This test is what caught a regression where a regex rename had rewritten
`</html>` to `</html_doc>` inside six viewer exports — see §3.6.

### Tier 2 — golden-image regression (DONE)

`tools/make_golden.py` renders all 40 engines from a deterministic analytic
surface (dome, raised ridge, incised groove, stepped block, fine texture),
writes references to `tests/golden/` and a review contact sheet to
`tests/golden_contact_sheet.png`. `tests/test_golden.py` re-renders and diffs.

Tolerances: mean absolute difference ≤ 1.0 (0–255 scale) and ≤ 2% of pixels
differing by more than 32, which absorbs platform float jitter without hiding
anything a human would see.

When a change to an engine is intentional: re-run `tools/make_golden.py`, look
at the new contact sheet, and keep the updated references.
`tests/test_golden.py --report` writes `tests/golden_diff.png` — reference | current
| difference ×8, one row per changed engine.

**Two traps this test walked into first — check for both in any golden test:**

1. `get_shading` returns float in **0–1** for most engines and 0–255 for a few.
   Clipping a 0–1 image straight to `uint8` crushes every render to black, and
   33 identical black images compare equal forever.
2. Injecting `normals_full`/`depth_vis_p` by hand is not enough — two thirds of
   the engines also need PTM/HSH coefficients, terrain maps, PCA and specular
   variance, which only the real pipeline produces. The fixture therefore
   synthesises *images* and runs them through `run_processing()`.

Both were caught by asking "does this test fail when I break something?" — a
3% change to one engine must produce exactly one failure. Verify that whenever
you add a golden test.

### Tier 3 — real-capture coverage (DONE)

`test images/` holds a genuine 8-frame photometric-stereo capture of a
manuscript page (960×1280, chrome sphere in frame). `tests/test_real_capture.py`
drives it and asserts on properties rather than known values, since real data
has no ground-truth light vectors:

* `auto_detect_sphere` recovers unit light vectors from the real specular
  highlights — **8 of 8** frames, min pairwise separation 16.6°. It recovered
  only 4 until the ROI-localisation bug was fixed (HANDOFF §5d); the test now
  asserts one light per frame, and that the located ROI is a small disc inside
  the frame, because unit length and `lz > 0` pass on garbage too
* normals are unit length and match the non-square source dimensions
* depth is finite, non-flat, and retains high-frequency page texture
* albedo lands at a plausible level
* all 40 render engines work on real data

This is the only test that exercises JPEG artefacts, sensor noise, uneven
illumination, genuine specularity and the sphere-calibration path.

---

## 7. Environment

Verified working:

```
python 3.12.4 · numpy 2.4.6 · opencv-contrib-python 5.0.0 · customtkinter 5.2.2
scipy 1.17.0 · trimesh 4.11.2 · rasterio · vispy · glfw · pyflakes
```

Install from `requirements.txt`. `rasterio` is optional — the GeoTIFF writer
falls back to cv2 and `tests/test_geotiff.py` covers both branches — but installing it
makes the suite test more.

**The one environment trap.** If `import cv2` fails with
`numpy.core.multiarray failed to import`, OpenCV was compiled against the
NumPy 1.x C API and NumPy 2 is installed. Fix:

```bash
python3 -m pip install --upgrade "opencv-contrib-python>=4.10"
```

Do **not** downgrade NumPy — the codebase uses NumPy-2 APIs. To confirm it is
environmental rather than a code regression, run `python3 -c "import Helio_Pro"`
from `~/Desktop/Helio_backup` (the untouched original): it fails identically.

---

## 8. Manual checks still worth doing

Automation covers the mechanics; these need eyes, and only occasionally:

1. **Launch it.** `python3 Helio_Pro.py` — no error dialog on startup, panels
   populated to the bottom of the left column.
2. **Light direction.** Drag the dome light north; north-facing surfaces should
   brighten. A sign error here is invisible to automated tests, and there is
   history: pointing the dome north used to light southern faces.
3. **One real dataset end to end** — load, calibrate against a sphere, process,
   check the render looks physically plausible, export an STL and open it.
4. **The 3D viewer** — it runs in a separate process with VisPy/GLFW and is not
   covered by any automated test.

---

## 7. Helio 3.0 (`helio_qt/`)

### Running it

Each `test_qt_*.py` builds the real window offscreen and prints `ok` or `FAIL`
per check, then `Results: N passed, M failed`. Run them one by one, each with
its own scratch preferences, so a test never touches yours:

```bash
cd ~/Desktop/Helio_NoTorch
for t in test_qt_*.py; do
  HELIO_QT_SETTINGS=$(mktemp -d)/q.ini HELIO_PREFS=$(mktemp -d)/p.json \
    python3 "$t" 2>&1 | grep -E "FAIL|^Results"
done
```

- `HELIO_QT_SETTINGS` points Helio's preferences at a scratch file.
- `HELIO_PREFS` does the same for the classic preferences the engine reads.
- With the engine and Classic suites, the whole run is about 2,100 checks.

| File | Covers |
|---|---|
| `tests/test_qt_shell.py` | The window, workspaces, menus, themes; every control has a tooltip |
| `tests/test_qt_capture.py`, `tests/test_qt_dome.py` | Chrome sphere, light files, placing lights by hand |
| `tests/test_qt_process.py` | The solve gives exactly the classic app's normals and depth |
| `tests/test_qt_process_tools.py` | Every Process setting, quick looks, scores, Try a few, rig profiles |
| `tests/test_qt_views.py` | Every view as the classic app draws it |
| `tests/test_qt_relief.py`, `tests/test_qt_3d.py` | The 3D relief, the high-detail window, video, area and volume |
| `tests/test_qt_selection.py`, `tests/test_qt_tools.py`, `tests/test_qt_annotate.py` | Selection, Inspect and Measure tools, annotations and their exports |
| `tests/test_qt_measure.py` | Scale, distances, profiles, areas |
| `test_qt_publish*.py` | Every Publish format writes real, valid files: images, reports, the publication sheet, 3D, web, archive, GIS, the QGIS suite |
| `tests/test_qt_saves.py` | File names from the object, the 2D and 3D light sweeps, Record video's options, 3D picture sizes, mesh densities, capture details in saved pictures, the sheet on A4 |
| `tests/test_qr_share.py` | The QR code a phone can open: doubled https mended, localhost and file paths called out, the code on screen; Share serves a web export |
| `tests/test_qt_menus.py` | Every menu item no other test reaches, chosen from its menu: open and save, the quick exports, the colour check, every solver, the quality checks, the measure, view and window items; fails if any menu item goes untested |
| `tests/test_qt_research.py` | Micro-features, toolpaths, unmixing, registration, AI |
| `tests/test_qt_session.py`, `tests/test_qt_next_capture.py`, `tests/test_qt_batch.py` | Sessions, one capture after another, batch |
| `tests/test_qt_help.py` | Help, Getting Started, Report a Problem |

### The built app checks itself

`./build_helio3.sh` makes no disk image unless the built app passes its own
test:

```bash
/Applications/Helio.app/Contents/MacOS/Helio --selftest "$(mktemp -d)"
```

It solves a known surface and writes exports with the bundle's own
interpreter and libraries, then prints `PASSED: 8 of 8 steps · build <commit>`.
Run it on the installed copy after every install. Tests of the source prove
nothing about the copy that is launched.

### Rules that have bitten before

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
