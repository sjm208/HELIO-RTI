# HELIO Modular Refactoring — Migration Guide

## Overview

This document is the step-by-step plan for extracting the monolithic
`Helio_Pro.py` (~40K lines) into the `helio/` package.  Work is split
into **three phases** — each is self-contained and testable.

### Working assumption

Source file to refactor: `Helio_Pro.py`
Target package: `helio/` (this directory)

You have both files open side‑by‑side.  After each phase you can run:

    python -c "from helio.ui import RTIViewerPro; print('OK')"

to verify the package still imports cleanly.

---

## Phase 1 — Extract standalone module-level functions

**Goal**: Move every function that does **not** reference `self` into the
appropriate submodule.  At the end of Phase 1, `Helio_Pro.py` is ~10K lines
shorter and the package is already usable for headless/batch workflows.

### Step 1.1 — Core photometric stereo solvers

| Function (from Helio_Pro.py) | → Target file |
|---|---|
| `least_squares_normals` | `helio/core/photometric_stereo.py` |
| `robust_mvs_normals` | same |
| `uncalibrated_normals` | same |
| `rbd_normals` | same |
| `pca_normals` | same |
| `rbd_plate_normals` | same |
| `generate_lighting_L` | same |
| Any other `def` that takes arrays and returns arrays | same |

**Action**: Copy-paste each function verbatim.  No changes to signatures
or bodies.  The `...` placeholder in the skeleton is replaced with the
real code.

### Step 1.2 — Depth integration solvers

| Function | → Target file |
|---|---|
| `poisson_depth_fft` | `helio/core/depth_integration.py` |
| `poisson_depth_dct` | same |
| `frankot_chellappa_depth` | same |
| `anisotropic_depth` | same |

### Step 1.3 — Image loading and I/O

| Function | → Target file |
|---|---|
| `load_and_process_single` | `helio/io/image_loader.py` |
| `load_plates` (if it exists as module-level) | same |
| `headless_render_worker` | same |
| `export_image` | same |
| `export_all_images` | same |

### Step 1.4 — Mesh / publication-sheet exporters

| Function | → Target file |
|---|---|
| `export_glb_mesh` | `helio/io/exporters.py` |
| `export_obj_mesh` | same |
| `export_stl_mesh` | same |
| `export_point_cloud_ply` | same |
| `export_publication_sheet` | same |
| `export_video_walkthrough` | same |

### Step 1.5 — 3D viewer subprocess entry

| Function | → Target file |
|---|---|
| `run_opengl_viewer` | `helio/render/viewer.py` |
| `build_viewer_mesh` | same |

### Step 1.6 — Utility functions

| Function | → Target file |
|---|---|
| `list_images` | `helio/utils/file_utils.py` |
| `ensure_dir` | same |
| `sanitise_filename` | same |
| `format_file_size` | same |

### Step 1.7 — Shading engines (the big one)

This is the **largest extraction**.  The original `get_shading()` method
in `RTIViewerPro` is roughly lines 15108–16250 (≈1100 lines) containing
30+ `if engine == "…":` blocks.

**Strategy**:
1. Move `get_shading()` → `helio/core/shading_engines.py` as the
   dispatcher function `render_shading()`.
2. Each `if engine == …` block inside becomes a private function
   `_engine_<name>()` in the same file.
3. The key transformation: replace every `self.xxx` reference with
   `data.xxx` (the `DataManager` object passed as first argument).

**Example transformation**:

```python
# Before (in RTIViewerPro.get_shading)
normals = self.normals_proxy if h != self.h else self.normals_full
albedo = self.albedo_proxy if h != self.h else self.albedo_rgb_full

# After (in shading_engines.py)
normals = data.normals_proxy if h != data.h else data.normals_full
albedo = data.albedo_proxy if h != data.h else data.albedo_rgb_full
```

---

## Phase 2 — DataManager integration

**Goal**: Migrate all `self.xxx` attribute storage **off** the GUI class
and **onto** `DataManager`.  At the end of Phase 2, `RTIViewerPro`
references data through `self.data.xxx` everywhere.

### Step 2.1 — Replace attribute writes in `__init__`

In `Helio_Pro.py`, the `__init__` of `RTIViewerPro` initialises hundreds
of `self.xxx` attributes (lines ~4725–5610, ≈900 lines).  Parallel edits:

- Replace `self.plate_paths = [...]` → `self.data.plate_paths = [...]`
- Replace `self.current_L_mat = None` → `self.data.current_L_mat = None`
- etc.

**Tip**: Use a text editor with multi-cursor or search-and-replace across
a selection.  Match `self.<FIELD>` where `<FIELD>` appears in
`DataManager.__init__`.

### Step 2.2 — Replace attribute reads in processing pipeline

Search for `self.normals_full`, `self.normals_proxy`, `self.depth_vis_p`,
`self.albedo_rgb_full`, etc. in `RTIViewerPro` methods and replace with
`self.data.normals_full`, etc.

### Step 2.3 — Replace shared_state initialisation

Remove the monolithic shared-state Value creation (original lines ~5400–5550).
`DataManager._build_shared_state()` already does this.

### Step 2.4 — Redirect cleanup

Replace:

```python
def on_closing(self):
    ...
    # old cleanup code
    ...
```

with:

```python
def on_closing(self):
    self.close_viewer()
    self.data.cleanup()
    self.destroy()
```

### Step 2.5 — Update all call sites

Every method that currently reads `self.normals_full` directly must be
updated.  The most common patterns are:

| Pattern | Replace with |
|---|---|
| `self.normals_full` | `self.data.normals_full` |
| `self.normals_proxy` | `self.data.normals_proxy` |
| `self.depth_vis_p` | `self.data.depth_vis_p` |
| `self.current_L_mat` | `self.data.current_L_mat` |
| `self.light_vecs` | `self.data.light_vecs` |
| `self.plate_paths` | `self.data.plate_paths` |
| `self.h, self.w` | `self.data.h, self.data.w` |
| `self.shared_state` | `self.data.shared_state` |
| `self.engine_settings` | `self.data.engine_settings` |
| `self.render_engine` | `self.data.render_engine` |
| `self.object_id` | `self.data.object_id` |
| `self.last_folder` | `self.data.last_folder` |
| `self._mm_per_px` | `self.data._mm_per_px` |

---

## Phase 3 — Incremental method extraction

**Goal**: Move the **method implementations** out of `RTIViewerPro` and
into the module files, leaving only stubs in the GUI class that call
the module functions.

### Step 3.1 — Processing pipeline

The method `run_processing` (original lines ~8040–8580) should become:

```python
# helio/core/processing_pipeline.py  (NEW FILE)
def run_processing_pipeline(data: DataManager, ...):
    ...
```

And the GUI stub:

```python
def run_processing(self):
    from helio.core.processing_pipeline import run_processing_pipeline
    run_processing_pipeline(self.data, ...)
```

### Step 3.2 — Export methods

Each `def export_xxx(self)` in the original (roughly lines 19000–29000)
can become a thin wrapper that delegates to `helio.io.exporters.xxx()`.

**Example**:

```python
def export_mesh_glb(self):
    from helio.io.exporters import export_glb_mesh
    path = filedialog.asksaveasfilename(...)
    export_glb_mesh(
        vertices=self.data.vertices,
        faces=self.data.faces,
        normals=self.data.normals_full,
        path=path,
    )
```

### Step 3.3 — Settings panels

The UI widget‑building methods (e.g. `_build_engine_settings_frame`,
`_on_engine_changed`) are tightly coupled to CustomTkinter and should
**stay** in `RTIViewerPro`.  Extract only the data‑computation parts.

---

## Verification checklist

After each phase:

```bash
# 1. No syntax errors
python -c "import helio; print('package OK')"

# 2. Subprocess workers import cleanly (no CTk dependency)
python -c "from helio.render.viewer import run_opengl_viewer; print('viewer OK')"

# 3. GUI starts
python main.py
python main.py

# 4. All original functionality works
#    - Load plates, run processing, switch engines, export

# 5. Session load/save still roundtrips

# 6. 3D viewer launches from subprocess
```

---

## Estimated effort

| Phase | Files touched | Lines changed | Risk |
|---|---|---|---|
| 1 (pure functions) | 9 files created | ~5,000 lines moved | **Low** — no logic changes |
| 2 (DataManager) | 2 files (Helio_Pro.py, data_manager.py) | ~800 `self.` → `self.data.` edits | **Medium** — mechanical but tedious |
| 3 (method stubs) | 15+ files | ~5,000 lines moved | **Low-Medium** — one method at a time |

**Total estimated time**: 4–6 hours of careful work, or 2–3 hours with
aggressive multi‑cursor / sed.

---

## Key risks & mitigations

| Risk | Mitigation |
|---|---|
| **Circular import** between `ui.app` and `core.shading_engines` | Use `TYPE_CHECKING` guards + lazy `import` inside method bodies (already standard in codebase).  Core modules never import `ui`. |
| **Subprocess pickle error** — `DataManager` objects passed to `Process()` | The viewer subprocess receives only raw arrays + `shared_state` dict (no `DataManager`).  This pattern is unchanged. |
| **Missed a `self.xxx` reference** and it becomes `None` at runtime | Run the app and click through all engines after Phase 2.  Add `assert` guards during migration. |
| **Performance regression** from lazy imports | Every `import` inside a method body adds ~1–10 µs on the second call (already cached by Python).  Negligible. |
