# Helio Architecture Blueprint

**Helio 3.0 (preview 16) and Helio Classic** · updated 24 October 2026

## Overview

Helio is a photometric stereo and RTI application for cultural heritage. It
takes photographs of one object, each lit from a different direction, and:

- solves the surface's direction at every pixel (the normals);
- integrates the normals into a height map;
- shows the surface in 36 views, relit and in 3D;
- measures it;
- publishes it as images, meshes, web viewers, GIS rasters, archives and
  publication sheets.

Two apps share one engine:

| App | Interface | Entry point | Built with |
|---|---|---|---|
| **Helio** (3.0) | PySide6, five workspaces | `helio3.py` (`python3 -m helio_qt`) | PyInstaller, `helio_qt.spec`, `build_helio3.sh` |
| **Helio Classic** (HELIO Pro 2.6) | CustomTkinter | `main.py` → `Helio_Pro.py` | py2app, `setup.py`, `build_dmg.sh` |

One disk image carries both. Helio Classic stays the home of the Terrain
(Freq / Engrave) and Forensic tabs, which Helio leaves out. Everything else
is in both, running the same code.

```
┌──────────────────────────────────────┐   ┌─────────────────────────────────┐
│  helio_qt/   Helio (PySide6)         │   │  Helio_Pro.py   Helio Classic   │
│  window · workspaces · actions       │   │  RTIViewerPro (CustomTkinter)   │
│  capture_ws · process_ws ·           │   │  + helio/ui, helio/render       │
│  inspect_ws · measure_ws · publish   │   │    (VisPy viewer subprocess)    │
│  relief3d · relief_window (OpenGL)   │   │                                 │
└──────────────────┬───────────────────┘   └────────────────┬────────────────┘
                   │      neither imports the other         │
                   ▼                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  helio/application/   GUI-free services both apps call                   │
│  pipeline (the solve) · jobs · quick_look · process_tools · render ·     │
│  viewing · exports + export_* · pub_sheet · plate_doc · research ·       │
│  annotations · capture_tools · dome_tools · rigs · viewer3d · batch · ai │
├──────────────────────────────────────────────────────────────────────────┤
│  helio/domain/   settings snapshot · registry (solvers, depth methods,   │
│                  views) · view_controls · capture_meta                   │
├──────────────────────────────────────────────────────────────────────────┤
│  helio/core/     photometric_stereo · depth_integration · bini ·         │
│                  shading_engines · processing · parallel · sphere ·      │
│                  metrology · qc · radiometry · flash · relief_analysis   │
├──────────────────────────────────────────────────────────────────────────┤
│  helio/io/       capture · image_loader · raw_loader · lp · session ·    │
│                  exporters · provenance · manifest · deposit · figures   │
├──────────────────────────────────────────────────────────────────────────┤
│  helio/vendor/   three.js and OpenSeadragon for offline viewers;         │
│                  Helio Relief (the QGIS plugin; source in qgis_plugin/)  │
└──────────────────────────────────────────────────────────────────────────┘
```

**The rule:** a feature lives in `helio/` with no interface code, and each app
is a thin layer over it. Moving a feature out of the classic app into
`helio/application/` is checked twice: the classic tests show nothing
changed, and Helio's tests show the wiring.

## Data flow

```
Photographs (JPEG, TIFF, camera raw)
     │  helio/io/capture, image_loader, raw_loader (linear when raw)
     ▼
Lights ── chrome ball(s)       core/sphere           (ball position kept in the light file)
       ── light file .lp       io/lp
       ── placed by hand       application/dome_tools (Virtual Dome Calibrator)
       ── rig profile          application/rigs
     │
     ▼
Solve   application/pipeline.solve(SolveRequest) → SolveResult
     │   normals: 7 solvers (Linear LS, Robust LS, IRLS, Iterative Specular
     │            Separation, Shadow-Rejection LS, Near-Field, Uncalibrated PCA)
     │   near-field measured from the ball; hand-held flash evened (strength, fall-off)
     │   quality: per-pixel uncertainty (σ°), fit residual, light spread
     │   depth: 9 methods (Fast Poisson FFT, Frankot-Chellappa, Simchony DCT,
     │          Confidence-Weighted, Multi-Scale, Gradient-Smoothed,
     │          Directional, Least-Squares free boundary, Bilateral)
     ▼
Render data (full size + a 2,500 px preview)
     │
     ├─► Inspect: 36 views (core/shading_engines via application/render),
     │            relit live; the 3D relief and the high-detail 3D window
     ├─► Measure: scale, distances, profiles (ISO 4287), areas, volumes,
     │            ISO 25178 statistics            core/metrology
     ├─► Research: micro-features, toolpaths, unmixing, registration,
     │            focus, recto/verso, AI descriptions
     └─► Publish: one queue of exports, run as a job (application/jobs)
          images · reports and figures · 3D models · web viewers ·
          archive and sharing · GIS and data
```

Quick looks (`application/quick_look`) run the same solve at about 1,000 px in
seconds, with an overall score (`process_tools.overall_score`). **Try a few**
compares several settings at once.

## Speed and memory

- **Strips on every core** (`core/parallel`): `by_rows`, `parts_for`,
  `lstsq_columns`, `frames_median` and `frames_var` split whole-image work
  into bands of rows on a thread pool. Each band computes exactly what the
  whole-array expression computes, so results are bit-identical (checked
  against 48 arrays).
- **The depth FFT** uses SciPy's pocketfft on every core (`workers=-1`) and
  falls back to NumPy if the pool is shutting down.
- **A cache of light-free work** (`parallel.cached`): normalised normals,
  blurred colour and PTM planes are kept per solve, keyed weakly by array
  identity, so moving the light only pays for the part that changes.
- **Photographs are read on threads**, not processes, everywhere.
- **A 32-photo, 46-megapixel binding** solves in about 74 s with a peak of
  28 GB (it was 433 s and 50 GB).
- **Later, if speed matters:** Metal (Apple GPU) could save about 40 s more
  per capture of that size.

## Core data

| Name | Shape | What |
|---|---|---|
| frames | `N × (H, W[, 3])` | the photographs, loaded on threads |
| `light_matrix` | `(N, 3)` float | light directions, +Y up |
| `normals_full` | `(H, W, 3)` float32 | unit surface normals |
| `albedo_*` | `(H, W[, 3])` | surface colour without shading |
| `depth_poisson`, `depth_vis_p` | `(H, W)` float32 | integrated height; levelled display height |
| `quality["sigma_deg"]` | `(H, W)` float32 | how sure each normal is, in degrees |
| `ProcessingSettings` | frozen dataclass | every solve setting, one snapshot (`domain/settings`) |
| `.helio` session | JSON | images, lights (with the ball's place), settings, views, scale (`io/session`) |

## Helio (helio_qt/)

- **window.py:** the main window, its header, the five workspaces and the
  next-step bar.
- **The five workspaces:** **Capture** (`capture_ws`, `hand_dome`), **Process**
  (`process_ws`, `compare_dialog`), **Inspect** (`inspect_ws`, `annotate`,
  `relief3d`, `section_card`), **Measure** (`measure_ws`) and **Publish**
  (`workspaces`, `publish`, `publish_catalog`).
- **actions.py:** every command once, with its menu, shortcut and the reason it
  is unavailable. **palette.py** is ⌘K, which finds any command.
- **3D:** `relief3d.py` is the 3D relief inside the window (OpenGL 4.1 core,
  section line, area tool, double-click pivot, off-screen pictures).
  `relief_window.py` is the high-detail window: 4,500 px detail, its own
  section card and Keep on top. `record_dialog.py` records turntable,
  flyover, orbit and light-sweep videos.
- **research.py:** the Research menu. **quick_exports.py:** the RTI web
  package, Send to QGIS and USDZ.
- **selftest.py:** the built app checks itself (`Helio --selftest <folder>`)
  before a disk image is made.
- **Look and help:** `theme.py` has four colour schemes (Graphite, Paper,
  Night Dome, Parchment). `help.py` has Help, Getting Started and Report a
  Problem.

## Publishing

- **The queue** (`helio_qt/publish.py`): the chosen formats run one by one on
  the job thread. Each writes its own files and reports them.
- **Publication sheet** (`application/pub_sheet.py`, A3 at 300 dpi, JPG pages and
  one PDF):
  - the title, a score badge, and the object and capture boxes;
  - a large view with its numbered details, sections A–B, C–D, …, measured
    lines and a scale bar;
  - up to 20 views, each cut to the object and balanced;
  - each detail close up in several views;
  - a profile for each section;
  - a clay 3D picture, other 3D pictures and pictures from files;
  - a light dome;
  - optionally, a second page lit from eight directions.
  - Details come from Annotate rectangles, measured lines from Measure, and
    sections from the 3D views.
- **Publication plate** (`application/plate_doc.py`): the A3 plate.
- **Classic contact sheet:** every panel on one long sheet.
- **GIS:** the DEM package (`export_dem`), terrain GeoTIFFs and
  **Send to QGIS** (`export_qgis`). The QGIS plugin export is the
  **Helio Relief 4.0.0** (QGIS 4), bundled in `helio/vendor/` and built
  from `qgis_plugin/` by `qgis_plugin/dev/build_zip.sh`. Its Import HELIO
  reads the DEM package.

## Building and checking

- `./build_helio3.sh` builds Helio with PyInstaller. Then:
  - the built app runs its self-test;
  - the script makes the disk image with Helio and the already-built
    Helio Classic, plus `Install Helio.command`.
- **Apple silicon only**, macOS 13 or later. Not code-signed.
- **Tests:**
  - `test_qt_*.py` cover Helio;
  - the other `test_*.py` files cover the engine and Helio Classic;
  - `tests/test_golden.py` compares views with approved pictures.
  - See [TESTING.md](../TESTING.md).
- **Planning and status:** the migration's plan, contracts and decisions are in
  [docs/pyside6-migration/](pyside6-migration/). Start with
  [migration-status.md](pyside6-migration/migration-status.md).
