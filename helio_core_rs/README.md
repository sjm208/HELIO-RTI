# helio_core_rs

Rust compute kernels for Helio, exposed to Python with PyO3 and built with maturin.

## Scope

Only the heavy numerical code in `helio/core/` moves here:

| Python module | Kernels | Status |
|---|---|---|
| `photometric_stereo.py` | `lls_normals`: Linear Least Squares normals (Python wrapper `linear_least_squares_normals`) | done (Step 2) |
| `shading_engines.py` | live relighting kernels (`relight.rs`) | Standard PBR, hillshade, multi-hillshade, Diffuse Gain, Exaggerated Shading done; Cast-Shadow Raking, RTI Specular Enhancement done; SSS done; RTI Image USM, Normals Unsharp Masking next |
| `parallel.py` | exact percentiles / median (`stats.rs`) | done |

The PySide6 interface (`helio_qt/`), the 3D views, the publication sheet and the
QGIS integration stay in Python.

## FFI rules

- Inputs are NumPy `float32` arrays borrowed as `PyReadonlyArray` (no copy).
- Outputs are allocated in Rust and returned as owned NumPy arrays (`PyArray`).
- The GIL is released while a kernel runs; work is split across pixels/rows with Rayon.
- Thread count follows `helio.core.parallel.workers()`: `HELIO_THREADS`, else cores - 1, max 8.

## Python fallback

`helio/core/_native.py` does `try: import helio_core_rs ... except ImportError`.
Callers use `rs` when it is not `None` and their NumPy code otherwise.
`HELIO_NO_RUST=1` forces the NumPy path. The NumPy code stays as the reference.

## Build

```bash
pip install maturin
cd helio_core_rs
maturin develop --release     # into the active virtualenv
cargo test                    # Rust unit tests (no Python needed)
```

From the repository root, `./run_tests.sh` runs the Python tests with the Rust
kernels and again with `HELIO_NO_RUST=1`.

## Linear Least Squares kernel

`lls_normals(pinv, images)`: `pinv` is the `(3, n_lights)` float64 pseudo-inverse
(taken in Python with `np.linalg.pinv`, same rank cut-off as `lstsq`, so
rank-deficient light sets agree); `images` is a C-contiguous `(n_lights, n_px)`
float32 stack. Returns `(n_px, 3)` float32 unit normals. Pixels are processed in
1024-pixel blocks with contiguous per-light inner loops (auto-vectorised), blocks
spread over Rayon. Non-float32 stacks (e.g. uint8) are converted a strip at a time
by `linear_least_squares_normals`. Measured on 1M pixels: ~1e-7 from NumPy,
7-30x faster.

## Verification

`python tests/test_rust_lls.py` (repository root) checks the Rust normals against a
frozen NumPy result (`tests/fixtures/golden_lls_normals.npz`) and the live NumPy solve to
1e-5, on the analytic surface (float32, uint8, float64), the photographs in
`tests/fixtures`, and edge cases (zero pixels, rank-deficient lights, block
boundaries, non-contiguous input, bad shapes). It also runs under
`./run_tests.sh`. Without the compiled module only the NumPy-vs-frozen check runs.

## Relighting kernels

All take C-contiguous float32 normals `(n_px, 3)` and return `(n_px,)` float32; the
engine in `shading_engines.py` reshapes. Each engine keeps its NumPy code and uses it
when the module is absent or an array is not float32 and contiguous.

| Kernel | Engine | 4.7 MP (4 cores) NumPy -> Rust |
|---|---|---|
| `standard_pbr` (lookup table, gain <= 1) | Standard PBR | 20 -> 10 ms |
| `hillshade` | PS Normal-Map Lit (Single) | 9 -> 8 ms (memory-bound) |
| `multi_lambert` | PS Normal-Map Lit (8-Dir), multi-hillshade | 50 -> 14 ms |
| `diffuse_gain` + `kth_values` | Diffuse Gain | 340 -> ~50 ms |
| `scaled_lambert` + `kth_values` | Exaggerated Shading | 100 -> 33 ms |
| `shadow_margin` + `shadow_lit` + `shadow_shade` (`shadow.rs`) | Cast-Shadow Raking | 98 -> 54 ms oblique, 61 -> 29 ms on-axis; identical output |
| `rti_specular` (`rti.rs`) | RTI Specular Enhancement | whole drag frame ~140 -> 38 ms (PTM, median or diffuse source) |
| `tone_rgb` (`tone.rs`) | every view's tone composite (`render._tone_rows`) | 1.3-3.7x per layer; bit for bit the NumPy / classic picture |
| `sss` | Subsurface Scattering | engine 25 -> 9.5 ms (blur cached); bit for bit |
| `apply_pinv` | `parallel.lstsq_columns`: PTM (6) and HSH (9) fits, Lambertian residual | 10-19x; 12 MP x 8 lights: 8.0 s -> 0.8 s |
| `horizon_scan` (`horizon.rs`) | Sky-View Factor, Positive/Negative Openness (RVT's scan) | 10 s -> 0.45 s per scan at 2500 px (22x); equals RVT to 1e-14 |
| `clay_lambert` + `clay_finish` | Albedo-Suppressed Raking (Clay Mode) | light-drag frame 61 -> ~25 ms |
| `write_obj`, `write_stl` (`meshio.rs`) | OBJ export (`mesh_export`, LOD package, facsimile, displacement base) and the 3D-print STL | OBJ 110 s -> 9 s at 12 MP (4 M vertices); byte-identical text |
| `kth_values` | `parallel.quantiles` / `median` | 56 -> 4.5 ms, bit for bit NumPy's |

Not ported, because scalar `powf` loses to NumPy's SIMD `power`: Robust PBR and Standard
PBR at gain > 1. A vectorised pow would change that.
`python tests/test_rust_relight.py` checks every engine Rust vs NumPy.

The shadow march keeps OpenCV's `warpAffine` for the light-aligned rotation (its
sampling positions are fixed-point, so reproducing them exactly is not worth it) and
replaces tilt + dilate + margin by one per-row monotonic-deque sweep, O(width)
whatever `shadow_steps` is.

`rti_specular` reads the PTM coefficients in place in either memory order. The preview's
are a `(6, n)` view of `(h, w, 6)` data (Fortran-ordered), and the interface passes the
light as float64; both are handled without a copy (the light is cast to float32). It
returns float32 where the NumPy engine returned float64 holding float32 values.

## Tone composite

`tone_rgb` turns the `(h, w)` float32 or float64 shading into `(h, w, 3)` uint8 in one pass:
layer (Combined Colour, Shading, Roughness Fusion), colour opacity and gain, micro-detail,
brightness, contrast and the gamma table. It reproduces the classic composite's float64
arithmetic operation by operation (Rust does not fuse multiply-adds), including the one
float32 step in Roughness Fusion, NaN and infinity. `tests/test_rust_relight.py` checks 1152
combinations against the NumPy code with `np.array_equal`. At 2500x1875 (NumPy -> Rust):
Combined Colour 12 -> 6 ms, opacity 0.6 49 -> 13 ms, micro-detail 48 -> 14 ms, Shading
7.5 -> 3.5 ms, Roughness Fusion 14 -> 6 ms. A cropped (non-contiguous) albedo is copied
once per frame.

## Least-squares fits with any number of unknowns

`apply_pinv(pinv, images)` is `lls_normals` without the normalisation and for any `k`:
`parallel.lstsq_columns` takes the pseudo-inverse in Python (`lstsq`'s rank cut-off) and
the per-pixel product here. float32 result (NumPy returned float64 when the lights were
float64). Agreement with NumPy about 2e-7 of the coefficient scale; both are float32-
accurate against a float64 reference (the largest error seen on random data was 2.6e-4 on
coefficients of order 100, which end up divided by 255 in an 8-bit picture).

## Horizon scan (Sky-View Factor and Openness)

`horizon_scan` is RVT's `sky_view_factor_compute` for the case Helio uses (SVF and openness
together): RVT's own table of search points (`horizon_shift_vector`) is passed in, so
directions, radii and their order are RVT's; the scan runs row by row (contiguous, so the
compiler vectorises it) instead of a full-image `np.roll` / `np.fmax` per direction and radius.
The slope is a float32 difference divided by a float64 radius, as in NumPy; `sin(atan m)` is
`m / sqrt(1 + m^2)`. `shading_engines._horizon_scan_rust` repeats RVT's preparation (float32,
NaN mask, pixel size, reflect padding) and falls back to RVT itself when the module is
missing or old. `tests/test_rust_relight.py` compares it with RVT on several sizes, radii, pixel
sizes, NaN regions and inverted surfaces.

## Mesh files

`write_obj` formats the lines of an OBJ on every core and writes them in order, in any of
the formats Helio's Python writers produced (`%.6f` with `v/vt/vn` faces for the PBR OBJ,
`%.4f` for the facsimile and displacement base, trimesh's `%.8f` with colour columns for the
LOD levels): `tests/test_export_speed.py` checks them byte for byte against the Python code.
`mesh_export` also used to write the OBJ twice - once through trimesh, then again to add the
UVs - and now writes it once. `write_stl` writes the binary STL records directly. Everything
falls back to the Python writers when the module is missing. GeoTIFFs are compressed on every
core (`num_threads="all_cpus"`, a GDAL option): 2.5-3.5x, same bytes of pixels.
Not changed because Rust would not help: toolpath extraction (OpenCV / scikit-image C code,
~1 s at 12 MP), the toolpath SVG (PNG encoding of the albedo), GLB / PLY (trimesh internals).

## Building on macOS (Apple silicon)

```bash
curl https://sh.rustup.rs -sSf | sh        # once: the Rust toolchain
./build_rust.sh                            # builds helio_core_rs into the Python that runs Helio
./build_helio3.sh                          # the app: builds the Rust kernels first, bundles them,
                                           # and its self-test fails if the app lacks them
```

`build_helio3.sh` takes `SKIP_RUST=1` (build without the kernels, slower app) and `REQUIRE_RUST=1`
(stop if they cannot be built). `helio_qt.spec` lists `helio_core_rs` as a hidden import when it is
installed. `.github/workflows/macos.yml` builds and tests the kernels on a GitHub macOS (Apple silicon)
runner and uploads a universal2 wheel. The PyInstaller bundling itself (the `.app`) can only be
checked by building on a Mac: the workflow does not build the `.app`.
