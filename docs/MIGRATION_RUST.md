# Helio-Rust: what was copied from Helio_3

Source: `sjm208/Helio_3` at commit `7047d85`.

Copied: everything Helio 3 uses (`helio/` minus `helio/ui/`, `helio_qt/`, tests,
docs, vendor assets, build files for the macOS Helio 3 app).

Left out (Helio Classic, out of scope): `Helio_Pro.py`, `main.py`, `setup.py`,
`build_dmg.sh`, `helio_splash.py`, `helio/ui/`, the Windows build files, the
Classic manual, Classic-only tests (those importing `Helio_Pro`, `helio.ui`,
`helio_splash`), `tests/test_golden.py` and `tools_*` that build the Classic app,
`docs/pyside6-migration/`.

Known follow-ups:
- `helio/data_manager.py`, `helio_help.py`, `helio/io/provenance.py`,
  `helio_qt/window.py` and a few `test_qt_*` tests still mention Classic
  (lazy imports, a "Open Helio Classic" menu action). Untouched on purpose.
- `build_helio3.sh` expects Classic's `dist/` app; not adapted yet.
- The golden-image test (`tests/test_golden.py`) is removed: it builds the Classic Tk app. `tools/make_golden.py` is kept for its pure helpers; its `build_app`/`main` still need Classic.
- `tests/test_headless.py` removed (drives Classic `main.py`).
- Environment-only failures here: Qt tests need `libEGL`; `test_render_speed`, `test_view_controls` fail the same with and without Rust.

Plan: Step 1 crate scaffold (done), Step 2 Linear Least-Squares kernel (done), Step 3 verification (done): `tests/test_rust_lls.py`, Rust vs NumPy within 1e-5, frozen golden in `tests/fixtures/golden_lls_normals.npz` (`python tests/test_rust_lls.py --update-golden` rewrites it).
Step 3 verification (done): `tests/test_rust_lls.py`, Rust vs NumPy within 1e-5, frozen golden in `tests/fixtures/golden_lls_normals.npz` (`python tests/test_rust_lls.py --update-golden` rewrites it).

## Relighting kernels (shading_engines.py)

Done: Standard PBR (table path), hillshade, multi-hillshade / 8-Dir, Diffuse Gain,
Exaggerated Shading, and exact percentiles (`parallel.quantiles`, `median`).
Verified by `tests/test_rust_relight.py` (engines Rust vs NumPy, quantiles bit for bit).
Cast-Shadow Raking done (`shadow.rs`; `tests/test_render_speed.py` passes with Rust, its
drag frame was over the 150 ms budget without). RTI Specular Enhancement done (`rti.rs`). Whole drag frames at 2500 px in
`tests/test_render_speed.py` now: Cast-Shadow Raking 115 ms, SSS 78, RTI Image USM 71,
Robust PBR 66, Exaggerated Shading 65, Normals Unsharp Masking 59. Much of each
frame is the colour/tone composite (`render._tone_rows`), shared by every view. Needs a vectorised pow to win on Robust PBR.

## Tone composite

`render._tone_rows` now calls `helio_core_rs.tone_rgb` (`tone.rs`); the NumPy code stays
as the fallback and the reference. Bit-identical (1152 combinations in
`tests/test_rust_relight.py`, plus the classic-composite check in `tests/test_render_speed.py`).
Whole drag frames at 2500 px, Rust, before -> after the composite: Standard PBR 24 -> 16 ms,
PS Normal-Map Lit (Single) 25 -> 12, RTI Specular Enhancement 38 -> 26, SSS 78 -> 55,
RTI Image USM 71 -> 56, Normals Unsharp Masking 59 -> 44, Cast-Shadow Raking 115 -> 107.
Still the slowest: Cast-Shadow Raking 107 ms, Robust PBR 72, Exaggerated Shading 57,
RTI Image USM 56, SSS 55, Bilateral Normal Sharpening 52.

## Where a solve spends its time (8 images, 12 MP, 4 cores)

Before the PTM/HSH change: 32.5 s = depth integration 11.8 s, PTM/HSH fits 8.0 s,
quality maps 3.2 s, light-count map 1.8 s, depth views 1.5 s, terrain 0.8 s, the
Rust normal solve < 1 s, loading 0.7 s (synthetic PNGs; real RAW/TIFF decode not measured).
After the PTM/HSH fits moved to Rust (`lstsq_columns` -> `apply_pinv`): 23.1 s, of which
the Poisson depth solve is 11.3 s. It pads to 4x the pixels in float64 complex
(6.4 GB per array at 100 MP): the next target, and the likely memory pressure on big captures.

## Memory, depth integration and the stage profiler

* `helio/core/depth_integration.py`: `poisson_solver` (the default "Fast Poisson (FFT)")
  now solves the mirrored-boundary Poisson problem with a type-2 DCT on the image itself
  instead of a padded complex FFT (the same answer; against a float64 reference it is
  slightly *more* accurate than before). `frankot_chellappa` uses the half spectrum and
  in-place strips (even-size Nyquist row/column handled explicitly), `simchony` and the
  polynomial baseline (`remove_poly22`, used by `pipeline.parabolic_baseline_correction`
  and `processing.subtract_parabolic_baseline`) avoid their full-size temporaries. The old
  versions stay in the file (`_poisson_solver_fft`, `_frankot_chellappa_grid`) as the
  references `tests/test_depth_memory.py` compares with. The baseline fit now returns the depth
  in its own dtype (float32 stays float32; it used to come back float64).
  Peak bytes per pixel: Poisson 244 -> 24, Frankot-Chellappa 80 -> 44, baseline fit
  432 MB for 3 MP and growing -> a fixed ~44 MB.
* 12 MP x 8 photos, whole solve: 32.5 s -> 13.9 s; peak memory 4.9 GB -> 3.5 GB. The depth stage
  is no longer the peak; the result itself (172 B/px, of which PTM+HSH are 60) is.
* `helio/utils/memory_plan.py`: reads the machine's total and free memory (Linux, macOS,
  Windows; `HELIO_MEMORY_GB` overrides) and estimates a solve's peak from the measured model
  (0.4 GB + (116 + 8 x photos) B/px, +25% margin, +4 B/px/photo per option that holds another
  stack - that last part is an assumption). `solve()` reports "comfortable", "tight" (fits
  if other programs are closed) or "too_big" (more than the machine has) in the progress
  detail line and the log, and shrinks the strip size when memory is short. It does NOT
  reduce the result's resolution: that would change pixel units (scale, measurements,
  near-field light position) and needs a decision.
* `HELIO_PROFILE=1` (or `tools/profile_solve.py`): per-stage time and peak-memory timeline.
* 100 MP x 8 photos is estimated at about 24 GB: comfortable on 64-96 GB, tight on 32 GB,
  too big for 16 GB.
* `tests/test_offline_viewers.py` has one failure that predates this work: it checks Classic's
  `setup.py`, which this repository does not carry.

## Inspect: first visit and revisits (2500 px preview, 40 views, 4 cores)

Sum of first-visit times: 45 s -> 21 s.  Positive / Negative Openness (and Sky-View Factor,
which shares their scan): about 7.5 s each -> 0.5 s (`horizon_scan`, needs `rvt-py` as before).
Normal Residual and Ambient Occlusion no longer recompute on every visit (their wide blurs
depend on the solve and a setting only): revisits 4.7 s / 3.3 s -> 0.03 s. First visits are
unchanged: Normal Residual 4.8 s, Colour ICA 3.4 s (scikit-learn FastICA), Ambient Occlusion
3.2 s, Exaggerated Shading 2.3 s, Bilateral Normal Sharpening 1.4 s, MSRM 1.1 s.
Note: with `rvt-py` NOT installed the Openness / SVF views use `calc_svf` (a different,
much slower algorithm); install it as `requirements.txt` says.

## New views, 3D and GIS exports (12 MP capture, 4 cores)

* Two views: "Multi-Directional Hillshading (Blend / PCA)" and "Albedo-Suppressed Raking (Clay Mode)"
  (`tests/test_new_views.py`).
* OBJ export (4 M vertices) 110 s -> 9.4 s; 3D-print STL 10.8 -> 2.5 s; LOD package 71 -> 18 s;
  DEM / GIS package 4.5 -> 3.3 s (threaded GeoTIFF compression). GLB (9.6 s), PLY (4 s) and the
  point cloud (1.4 s) are trimesh internals and unchanged.
* Toolpaths: ~1 s at 12 MP (about 8 s at 100 MP); the cost is in OpenCV / scikit-image, so no Rust.

## Optional reduced working size (for machines short of memory)

`ProcessingSettings.work_scale` (default 1.0 = full size) or `HELIO_WORK_SCALE=0.5` makes
`solve()` reduce each photograph (area-averaged) as it is read, so the full-size stack never
exists; the memory plan runs first, from the photographs' header sizes, and offers the largest
of 75 / 50 / 35 / 25% that fits. Every pixel-unit input is converted exactly as Quick look
does it (`reduced_request`): the grey-card and flat-area boxes, the light origin, the flat
reference, the near-field scale (`mm_per_px / s`), and a crop drawn on the photograph is
scaled by the result's own size. The result records `work_scale` and `source_size`
(`DataManager.work_scale`), the user's own settings are kept in it, and
`pipeline.result_mm_per_px(result, mm_per_px_of_the_photographs)` gives the reduced pixel's size.
Tested (`tests/test_work_scale.py`): at 50% memory is a quarter and the normals are as accurate.

**The interface (done, `tests/test_work_scale_ui.py`).** The Process panel has *The photographs > Working
size*: Full size / 75% / 50% / 35% / 25% / Automatic (fit this machine's memory; `work_scale = 0`
is resolved by `memory_plan.auto_scale`). The choice is saved in sessions (optional `work_scale`
key, written only when not 1.0), and the status line says "Solved at N% size". Help > diagnostics
lists it.

**Scale re-basing (done).** `Project.scale_mm_per_px` is the size of one pixel *of the current
result*, and `Project.scale_basis` is the working size those pixels are at. `photo_scale()`
(= scale x basis) is the scale of the photographs themselves, and is what solve requests,
sessions and rig profiles use. `set_result()` re-bases automatically after a solve at a different
size, so a scale entered or measured before or after a reduced solve stays right, and
measurements and mm exports from a reduced result are correct.

## PTM / HSH on demand (tight machines only)

The PTM and HSH coefficients are 60 bytes a pixel (about 6 GB at 100 MP), the largest thing a
result keeps after the solve. When the memory plan says the machine is not comfortable, the
solve makes the two things Inspect needs (the preview PTM and the six preview-size planes the
live Specular / IQ views relight), drops the full-size coefficients, and records the request.
Whatever needs them (the PTM/HSH, RTI-viewer, relight, WebRTI, DeepZoom and web-package exports,
the "PTM Base" report page, a full-size render of Specular Enhancement / IQ / RTI Specular
Enhancement) calls `pipeline.ensure_ptm_hsh`, which runs the same preprocessing from the
photographs again (`solve(..., ptm_only=True)`: it stops before the normals) and keeps the
result. Same coefficients to 1e-6; Inspect views are identical (`tests/test_ptm_deferred.py`).
A roomy machine, a quick look, or `HELIO_KEEP_PTM=1` keeps them as before. The cost is the
first such export re-reading the photographs, and that they must still be where they were
(otherwise a clear error). Selections (`crop_result`) cut the planes and refit inside the box.

## Live full-resolution main view, and the Ridge and Groove Detector

**Live detail (`helio_qt/inspect_ws.py`, `helio/application/viewing.py`, `tests/test_live_detail.py`).** Inspect
draws the preview (long side <= 2500 px). When the zoom enlarges it by more than 1.2x, the part on
screen (plus 8%) is drawn again from the full-size arrays and laid over the preview
(`Viewport.show_detail`), at lower priority than the main frame, on the same render thread. It is
redrawn when the view settles after a pan or zoom (140 ms), follows the light while a region takes
<= 90 ms (measured per view), and is hidden during a drag that is slower, then drawn when the drag
ends. The region is capped (16 MP; 6 MP under 24 GB of memory). Off: *Loupe > Sharp at full size
when zoomed in*. Not drawn: a split view, photographs, the backlit frame, and PTM views on a tight
machine whose coefficients were dropped (the loupe likewise: neither may fit them on the render thread).

Measured per view (`match.py`-style, now `tests/test_live_detail.py`): lighting and normal views are the same
picture at both sizes (difference ~0); IQ, Openness, Slope Gradient, Diffuse Gain, Hillshading, Raking
Composite and Specular View stretch to the range they are given, so a region is matched to the preview's
tone (gain and offset per channel, `viewing.tone_match`; IQ 38 -> 4.4 /255); AO, Retinex, Cast-Shadow
Raking, Surface Curvature, Normal Residual, Curvedness, Shape Index, Exaggerated Shading and the two
plate decompositions size their filters in pixels or work out their decomposition again on the region, so
they are a different picture and keep the enlarged preview (`DETAIL_SKIP`). The test fails if a view that
is drawn this way jumps in look.

**Ridge and Groove Detector.** The Hessian of the depth at three widths (x1, 1.6, 2.5 of *Line width*, in
full-size pixels, so preview and full size find the same lines), Frangi blobness (beta 0.5) to suppress round
bumps, strength normalised sigma^2, best width kept. Modes: ridges and grooves on grey, grooves, ridges,
direction (hue = direction the line runs, value = strength). One white level, taken from the preview, so a
region is not stretched to its own contents. The per-pixel step is `helio_core_rs::ridge_update` (in place,
Rayon, same float32 arithmetic and order; 12 MP: 5.2 s -> 1.4 s, the remainder is OpenCV's separable filters;
agrees with NumPy to ~1e-7). `tests/test_ridge.py`: ridge vs groove vs bump, directions 0 / pi/2 / pi/4, Rust vs
NumPy, preview vs full-size width, modes, gain.

## 3D relief mesh: Rust grid builder and 2 : 1 density

`helio_core_rs::grid_mesh` builds the relief panel's vertex grid and triangle indices in one parallel pass,
bit-identical to the NumPy builder it replaces (kept as `_build_grid_numpy`, the reference): 4.2 M points
(1 : 1 of a 2500 px preview) 376 ms -> 24 ms, and no meshgrid / stack temporaries, so changing the density no
longer stalls the window. The panel gains **2 : 1** (two vertices to every pixel, step 0.5). The height is a
texture read with linear filtering in the vertex shader, so vertices between pixels take interpolated heights
and no shader changed. It adds smoother silhouettes and edges, not information (the heights are the pixels'),
and the shading already comes from the normals at pixel size. The point limit (7 M; 9 M in the high-detail
window) still holds: a surface too big for 2 : 1 is built at 1 : 1 and the readout says so
("mesh 1 : 1 (2 : 1 would pass 7 million points)"), so in practice 2 : 1 is for a selection or a small capture.
The separate 3D window and the exports keep whole pixels (`step` there stays an integer >= 1).
`tests/test_grid_mesh.py` (Rust = NumPy, fractional steps, the fall-back and its wording). Not exercised on a real
GPU here (no OpenGL in this environment): whole steps hand the GPU the same bytes as before.

## 3D relief panel: shaded at the solve's own resolution

The panel's mesh is the solve cut to at most 7 M points, and it was shaded with normals of that same size,
so a denser mesh could not look sharper (at 7 M points 2 : 1 cannot be built at all). It is now shaded like the
high-detail window: `viewer3d.detail_textures` makes the solve's normals and colour at up to 4,500 px on the long
side (3,200 under ~12 GB free, 2,400 under ~6 GB), area-averaged from every pixel, off the main thread, after the
mesh is shown (`InspectWorkspace._sharpen_relief`; `ReliefView.set_detail`). Nothing is made when the mesh is already
as sharp as the solve. `tests/test_relief_detail.py`. Not exercised on a real GPU here.

## Standard and Robust PBR: steady brightness as the light lowers

Both draw (N.L) ** (1 / gain), 6.7 at the default gain 0.15, so a flat area is sin(elevation) ** 6.7: 0.23 with the
light 53 degrees up, 0.003 at 25, black under 20. Dragging the dome toward its rim made the picture "die"
(`tests/test_dome_directions.py` measured flat ground 0/255 at 15 degrees, 202 at 75). New control *Light height*
(`pbr_expose`, on by default): a factor F = 0.5 + 0.5 sin(el) ** 8 keeps a flat area at 130-150/255 from 15 to 53 degrees
and exactly the old 1.0 overhead, so the default (overhead) picture and the Standard PBR reference image are
unchanged; contrast under a 15 degree light goes from 0.1 to 71 /255. *As the classic app* restores the old curve.
Done by scaling the lookup table (Rust and NumPy alike, parity checked at 20, 35, 55 degrees). Saved sessions with
no `pbr_expose` open with it on, so a low-light PBR view looks different from before.

## Measure tab at a reduced working size (bug fixed)

Setting the scale on the Measure tab's **Photograph** view (photograph pixels) stored it as if drawn on the *result*
(the call was `set_scale_measured`), and every distance there used the result's scale on photograph pixels. At 35% a
result pixel is 2.86 photograph pixels, so Inspect's sections, the 2D scale bar and the 3D viewer's measure, section,
area and volume came out 2.86 x too short (20 mm read 7 mm); measurements agreed only on the picture they were
drawn on. Now the Measure tab works in the photograph's pixels on every one of its pictures (the relief and height
views are laid over the photograph at its size and place), sets and uses `photo_scale()`, and converts to the solve's
pixels (x ws, minus a region solve's offset) only where it reads solved data (profiles, regions, volume, the ruler).
Inspect and the 3D viewers keep working in result pixels with `scale_mm_per_px` (per result pixel), which is now right.
`tests/test_measure_scale.py` (41 checks at 50% and 35%): the same two points measure 20.00 / 25.00 mm on the photograph, the
relief and height views, in the 3D viewer (distance, diagonal, section), and Inspect's pixels; scale before or after
the solve, a re-solve at another size, a region solve's offset. Status line, calibration file PPI and capture-details
scale now say the photographs' pixel.

## Parallel sections and section drawings (3D relief)

On the section card (Inspect's, and the high-detail window's): **Copies** (1-25) and **every** (mm once a scale
is set, else pixels; starts at a tenth of the section's length) repeat the section at even steps to both sides of it
(odd numbers keep yours in the middle; positive offsets are to the right looking A to B). They are drawn dashed on the
relief and as a **stacked** (ridgeline, one height bar) or overlaid chart; copies that would leave the surface are
left out and counted, not clipped; moving the section moves its copies. **Drawing...** writes a section drawing
(`.svg`: to a round stated scale, e.g. 25:1, axes in mm, scale bar, grooves and ridges numbered, a title block with
object, scale, vertical exaggeration, A and B, Ra/Rq/Rz, solve), a CAD file (`.dxf`, R12, mm, layers PROFILE / FEATURES /
FRAME / TEXT) and, for copies, a CSV; Save... also writes the series CSV. *Heights x* exaggerates for shallow relief and the
sheet says so. Code: `helio/core/sections.py` (geometry), `helio/application/section_export.py` (files),
`ReliefView.profile_between / section_series`, `helio_qt/section_card.py`. `tests/test_section_series.py` (53 checks): geometry,
known ramps (copies equal along a contour, step = slope x spacing across it), the files read back (the SVG's own
coordinates give the stated scale and heights; the DXF passes `ezdxf`'s reader and audit when installed). The dashed
lines on the 3D relief itself are drawn by the same code as the A and B markers and could not be looked at here (no OpenGL).

## Flood plane (3D relief) and the copies chart beside the section

**Flood** (toolbar button, or **W**): a flat plane at a height above the surface's lowest point. The surface below it is
tinted blue with a pale shoreline (fragment shader: `u_flood_on`, `u_flood`, same constructs as the contour lines); under
the 3D view a card has the level (slider, number, or **Set from the surface**: click a hollow's rim and the plane goes to
that height), the part to measure (the whole surface, or inside the **Area** outline once closed) and the figures: area
below the plane, what it would hold (mm3 and ml), deepest and mean depth, and each separate patch (4-connected, noise
ignored) with whether its rim holds it (**closed**) or it runs off the edge of the surface or region (**open**). Dragging
the slider is live (area, volume, depths); the patches follow when it is let go. Save writes a CSV. Units are mm when a scale
is set, else pixels. `helio/core/flood.py`, `helio_qt/flood_card.py`, `ReliefView.flood_*`. `tests/test_flood_plane.py` (44 checks):
a paraboloid bowl against its exact volume (0.02%) and area (1%), patches, the viewer's shader level tinting exactly the
cells the figures count, click-to-set-level, region, the card. The shaders are compiled with `glslangValidator` (desktop
GLSL 4.10) when it is installed, which also catches a deliberate error; the tint itself could not be looked at here (no OpenGL).
The same card is in the high-detail window.

The section card's **copies chart** now sits to the right of the section's own chart (3 : 2 of the row) while copies are on,
so the card does not grow taller; alone, the section has the whole width.

## macOS (Apple silicon)

Proved on a GitHub `macos-14` runner (arm64, Python 3.12, Rust 1.99; `.github/workflows/macos.yml`, run 5, green):
the kernels compile in ~25 s, `./build_rust.sh` installs them and its check passes, the universal2
(arm64 + x86_64) wheel builds (artifact `helio_core_rs-macos-universal2-wheel`), and the test files
pass both with the Rust kernels and with `HELIO_NO_RUST=1`. Normals for 4 MP x 8 lights: Rust 0.036 s,
NumPy 0.95 s (27x) on that 2-thread runner. The one failure on the way was a test tolerance (float64
FFT parity, 2-3e-9 of the range on arm64; the output is float32).

Not proved: the PyInstaller `.app` itself (spec change, `build_helio3.sh` hook, self-test step) - the
workflow does not build the app, and it needs the Classic app present (`dist/`), which this repository
does not carry. Build on a Mac with `./build_helio3.sh`; its self-test now fails if the kernels were
built but are not in the bundle.

## Inspect export buttons, colour slider, new defaults

- Inspect › Export: "RTI web package" is replaced by **Combined RTI viewer** and **3D viewer: fine detail** (each asks for a folder, then runs through Publish). The RTI web package is still in File › Quick Export and in Publish.
- The fine-detail 3D viewer page has a **Photograph colour** slider (0 = relief only, 1 = photograph colour laid over it).
- Publish opens with nothing ticked (it opened with the two 3D web viewers until 2026-10-29); mesh density 1:4, height exaggeration 90, edge trim 2.0%.
- Standard/Robust PBR default to the classic lighting (`pbr_expose` = 0); "Keep brightness steady" is the option, now with a floor-only lift and a soft shoulder so it clips far less.

## Phone view (QR, no upload)

- Inspect › Export › **Phone view (QR)** (also File › Share › Phone View…) starts a small web server on this computer and shows a QR code. A phone on the same Wi-Fi (or hotspot) scans it and the relief opens live in its browser: a dome to move the light, Clay or Colour, Sweep, pinch to zoom. Nothing is uploaded; the server stops when the window closes.
- `helio/application/phone_view.py` (server, page, picture prep); `helio_qt/dialogs.py::PhoneViewDialog`; `tests/test_phone_view.py` checks the server, the page in a phone-sized browser (lit like a NumPy Lambert render, not the opposite light) and the dialog.
- The address is random per session and only the page and its two pictures are served. Live camera AR is not part of this (it needs an https address).

## Tidy-up before sharing

- `tools/gui_audit.py` (and `tests/test_gui_audit.py`) check the interface hangs together: every menu command has a handler and a unique shortcut, every button is connected, Publish formats are all written, words are spelt one way (Colour, 3D). Run `python tools/gui_audit.py`.
- The little **i** buttons now show their text when clicked.
- Help text now names the real menu entries (View › 3D Model › Open Inverted in 3D; View › Appearance).
- Phone view: lists every network address this computer has, with a "Test from this computer" button that says whether Helio, the Wi-Fi or the firewall is the problem.
- Speed tests are held to a looser limit on the NumPy fallback than on Rust.

- Phone view, when the phone cannot reach the computer: the Wi-Fi/Ethernet address is offered first on a Mac (a VPN can no longer win), **Open on this computer** shows whether the page itself works, and **Save as one file for the phone…** writes the page with the relief inside it, for AirDrop/e-mail, with no network at all.
- Removed: Help › Open Helio Classic, and every "Helio Classic has it" message.

## Zoomed-in lag (full-size detail overlay)

Measured with a 4300 x 2900 solve in a 1780 x 1164 view: a light-drag frame took ~19 ms with the full-size overlay off and 32-49 ms with it on, even for a 0.8 MP region (the GUI thread turns each region into a picture on every frame; a 6 MP region costs ~12 ms, 16 MP ~60-90 ms). Now the overlay follows the light only when drawing it AND showing it takes under 8 ms (it was 90 ms of drawing alone); otherwise it appears when the light is let go. It also starts at 1.5x enlargement (was 1.2x) and its region is capped at 4 / 9 MP (was 6 / 16). Drag frames with it on are back to ~20 ms.

Phone view: `<address>ping` answers in plain text, a page error is shown on the page, and the check message names macOS's Local Network privacy switch.

## Notes and parallel sections on the publication sheet; kept work in the session

- **Notes:** everything drawn with Annotate (arrows, ellipses, lines, measures, polygons, freehand, dots, numbers, text) on the layers that are showing goes on the sheet's large view in its own colour. Each one with a caption gets a tag n1, n2 … and a legend line under the picture; rectangles stay the boxed, close-up details. Publish › the sheet has a tick box ("Your annotations on the large view", on by default).
- **Parallel sections:** a section with copies on keeps its copies when added to the sheet (the section card now has **Add to sheet**, in Inspect and in the high-detail window). The copies are drawn faintly on the large view and the chart is the stacked series with its offsets and one height bar.
- **Saved with the session:** annotations (layers, which are hidden, captions), the sections kept for the sheet (with copies) and the sheet's choices travel in the `.helio` file under one extra key, `helio3`, which the older reader ignores and writes back. A session with none of these is byte for byte what it was. Before this, annotations and kept sections were lost when a session was closed.
- The Combined RTI viewer and 3D viewer: fine detail buttons in Inspect open the page in the browser once it is written.
- Tests: `tests/test_sheet_notes.py`, `tests/test_kept_work.py`, `tests/test_open_viewers.py`.

- Sections are named **A–B, C–D, E–F …** (thirteen pairs): the card's title, its chart, the line in 3D, the line on the picture, the CSV and drawing files, and the publication sheet all use the same letters. A section kept for the sheet keeps its letters; the next line you draw is the next pair. `tests/test_section_names.py`.

- **Publication sheet, sections:** each section (A–B, C–D …) now has a picture of the area it was taken from beside its graph: the large view cut round the line (and its copies), with the line and its ends drawn on it, a scale bar and the caption "Where A–B was taken". The picture alternates sides down the page (left of the first graph, right of the second …), so the graphs no longer run the whole width of the page. Tick labels are rounded to what the range deserves.
