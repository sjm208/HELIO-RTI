//! helio_core_rs: Rust compute kernels for Helio 3.
//!
//! The Python package `helio` stays the source of truth.  Every function here
//! has a NumPy twin in `helio/core/`, and `helio/core/_native.py` picks
//! this module when it imports and the NumPy code when it does not, so a
//! build without the compiled module behaves exactly as Helio did before.
//!
//! FFI conventions (all kernels follow them):
//!
//! * Inputs arrive as `PyReadonlyArray*<f32>`: a borrow of NumPy's buffer,
//!   no copy.  Non-contiguous or non-f32 inputs are converted on the Python
//!   side, once, by the wrapper in `_native.py`.
//! * Outputs are allocated here and handed to NumPy as owned arrays
//!   (`PyArray`), so Python frees them like any other array.
//! * The GIL is released (`Python::detach`) for the whole computation, and the
//!   work is split over pixels or rows with Rayon.

use numpy::{
    PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyReadwriteArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod gridmesh;
pub mod horizon;
pub mod lls;
pub mod meshio;
pub mod relight;
pub mod ridge;
pub mod robust;
pub mod rti;
pub mod shadow;
pub mod stats;
pub mod tone;

/// Rayon threads, matching `helio.core.parallel.workers()`: `HELIO_THREADS`
/// if set, otherwise every core but one, at most eight.
fn default_threads() -> usize {
    if let Some(n) = std::env::var("HELIO_THREADS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
    {
        return n.max(1);
    }
    let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
    cores.saturating_sub(1).clamp(1, 8)
}

/// Size Rayon's global pool once, the first time the module is imported.
/// A second call (the pool already built, e.g. by an embedding host) is not
/// an error: the existing pool is used.
fn init_thread_pool() {
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(default_threads())
        .thread_name(|i| format!("helio-rs-{i}"))
        .build_global();
}

/// The number of threads the kernels run on.
#[pyfunction]
fn threads() -> usize {
    rayon::current_num_threads()
}

/// Linear Least-Squares normals for every pixel.
///
/// `pinv` is the `(3, n_lights)` pseudo-inverse of the light matrix (float64,
/// from `np.linalg.pinv`); `images` is the C-contiguous `(n_lights, n_px)`
/// float32 stack.  Returns `(n_px, 3)` float32 unit normals.
#[pyfunction]
fn lls_normals<'py>(
    py: Python<'py>,
    pinv: PyReadonlyArray2<'py, f64>,
    images: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let (k, n) = (pinv.shape()[0], pinv.shape()[1]);
    let (n_img, m) = (images.shape()[0], images.shape()[1]);
    if k != 3 || n != n_img {
        return Err(PyValueError::new_err(format!(
            "pinv must be (3, {n_img}) to match images (n_lights, n_px); got ({k}, {n})"
        )));
    }
    let p: Vec<f32> = pinv.as_array().iter().map(|&v| v as f32).collect();
    let img = images
        .as_slice()
        .map_err(|_| PyValueError::new_err("images must be C-contiguous float32"))?;
    let mut out = vec![0f32; 3 * m];
    py.detach(|| lls::lls_normals(&p, n, img, m, &mut out));
    PyArray1::from_vec(py, out).reshape([m, 3])
}

/// `pinv @ images` for `pinv` `(k, n_lights)` float64 and the C-contiguous
/// `(n_lights, n_px)` float32 stack: the `(k, n_px)` float32 least-squares
/// coefficients of every pixel (PTM k = 6, HSH k = 9).
#[pyfunction]
fn apply_pinv<'py>(
    py: Python<'py>,
    pinv: PyReadonlyArray2<'py, f64>,
    images: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let (k, n) = (pinv.shape()[0], pinv.shape()[1]);
    let (n_img, m) = (images.shape()[0], images.shape()[1]);
    if k == 0 || n != n_img {
        return Err(PyValueError::new_err(format!(
            "pinv must be (k, {n_img}) to match images (n_lights, n_px); got ({k}, {n})"
        )));
    }
    let p: Vec<f32> = pinv.as_array().iter().map(|&v| v as f32).collect();
    let img = images
        .as_slice()
        .map_err(|_| PyValueError::new_err("images must be C-contiguous float32"))?;
    let mut out = vec![0f32; k * m];
    py.detach(|| lls::apply_pinv(&p, k, n, img, m, &mut out));
    PyArray1::from_vec(py, out).reshape([k, m])
}

/// `(n_px, 3)` f32 normals as a slice, or a ValueError naming the argument.
fn normals_slice<'a>(a: &'a PyReadonlyArray2<'_, f32>, what: &str) -> PyResult<&'a [f32]> {
    if a.shape()[1] != 3 {
        return Err(PyValueError::new_err(format!("{what} must be (n_px, 3)")));
    }
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous float32")))
}

fn vec3(a: &PyReadonlyArray1<'_, f32>, what: &str) -> PyResult<[f32; 3]> {
    match a.as_slice() {
        Ok([x, y, z]) => Ok([*x, *y, *z]),
        _ => Err(PyValueError::new_err(format!(
            "{what} must be three float32 values"
        ))),
    }
}

/// `(k, 3)` f32 lights as an array of vectors.
fn light_rows(a: &PyReadonlyArray2<'_, f32>) -> PyResult<Vec<[f32; 3]>> {
    let s = a
        .as_slice()
        .ok()
        .filter(|_| a.shape()[1] == 3)
        .ok_or_else(|| PyValueError::new_err("lights must be C-contiguous float32 (k, 3)"))?;
    Ok(s.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}

/// Standard PBR through a lookup table of `x ** inv_gain` over 0..1:
/// `lut[min(max(n.l, 0) * len, len - 1)]`. Returns `(n_px,)` float32.
#[pyfunction]
fn standard_pbr<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f32>,
    light: PyReadonlyArray1<'py, f32>,
    lut: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n = normals_slice(&normals, "normals")?;
    let l = vec3(&light, "light")?;
    let table = lut
        .as_slice()
        .ok()
        .filter(|t| !t.is_empty())
        .ok_or_else(|| PyValueError::new_err("lut must be a non-empty contiguous float32 array"))?;
    let mut out = vec![0f32; n.len() / 3];
    py.detach(|| relight::standard_pbr(n, l, table, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

/// Hillshade, `clip(clip(|n.l|, 0, 1) + ambient, 0, 1)`. Returns `(n_px,)` float32.
#[pyfunction]
fn hillshade<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f32>,
    light: PyReadonlyArray1<'py, f32>,
    ambient: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n = normals_slice(&normals, "normals")?;
    let l = vec3(&light, "light")?;
    let mut out = vec![0f32; n.len() / 3];
    py.detach(|| relight::hillshade(n, l, ambient, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

/// `sum_k w_k * max(0, n.l_k)` per pixel, and with `with_max` also the largest
/// `max(0, n.l_k)`. Returns `(sum, max_or_None)`, each `(n_px,)` float32.
#[pyfunction]
#[pyo3(signature = (normals, lights, weights, with_max=false))]
#[allow(clippy::type_complexity)]
fn multi_lambert<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f32>,
    lights: PyReadonlyArray2<'py, f32>,
    weights: PyReadonlyArray1<'py, f32>,
    with_max: bool,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Option<Bound<'py, PyArray1<f32>>>)> {
    let n = normals_slice(&normals, "normals")?;
    let ls = light_rows(&lights)?;
    let w = weights
        .as_slice()
        .ok()
        .filter(|w| w.len() == ls.len())
        .ok_or_else(|| {
            PyValueError::new_err("weights must be contiguous float32, one per light")
        })?;
    let m = n.len() / 3;
    let mut sum = vec![0f32; m];
    let mut top = if with_max { Some(vec![0f32; m]) } else { None };
    py.detach(|| relight::multi_lambert(n, &ls, w, &mut sum, top.as_deref_mut()));
    Ok((
        PyArray1::from_vec(py, sum),
        top.map(|t| PyArray1::from_vec(py, t)),
    ))
}

/// Diffuse Gain before its percentile stretch. Returns `(n_px,)` float32.
#[pyfunction]
fn diffuse_gain<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f32>,
    lights: PyReadonlyArray2<'py, f32>,
    strength: f32,
    gamma: f32,
    spec: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n = normals_slice(&normals, "normals")?;
    let ls = light_rows(&lights)?;
    if ls.is_empty() {
        return Err(PyValueError::new_err("lights is empty"));
    }
    let mut out = vec![0f32; n.len() / 3];
    py.detach(|| relight::diffuse_gain(n, &ls, strength, gamma, spec, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

/// Exaggerated Shading's per-scale sum over a list of `(n_px, 3)` float32
/// arrays, one per scale: `sum_i w_i * max(0, p_i.l) / total`.
#[pyfunction]
fn scaled_lambert<'py>(
    py: Python<'py>,
    scales: Vec<PyReadonlyArray2<'py, f32>>,
    light: PyReadonlyArray1<'py, f32>,
    weights: PyReadonlyArray1<'py, f32>,
    total: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let l = vec3(&light, "light")?;
    let slices = scales
        .iter()
        .map(|s| normals_slice(s, "scale"))
        .collect::<PyResult<Vec<_>>>()?;
    let w = weights
        .as_slice()
        .ok()
        .filter(|w| w.len() == slices.len())
        .ok_or_else(|| {
            PyValueError::new_err("weights must be contiguous float32, one per scale")
        })?;
    let m = slices.first().map_or(0, |s| s.len() / 3);
    if slices.iter().any(|s| s.len() != 3 * m) {
        return Err(PyValueError::new_err(
            "all scales must have the same number of pixels",
        ));
    }
    let mut out = vec![0f32; m];
    py.detach(|| relight::scaled_lambert(&slices, l, w, total.max(1e-6), &mut out));
    Ok(PyArray1::from_vec(py, out))
}

/// The values at 0-based ranks `ks` of a C-contiguous float32 array of any
/// shape, exactly as a full sort would give them; `None` if the array holds a
/// NaN or infinity or a rank is out of range.
#[pyfunction]
fn kth_values(
    py: Python<'_>,
    a: numpy::PyReadonlyArrayDyn<'_, f32>,
    ks: Vec<usize>,
) -> PyResult<Option<Vec<f32>>> {
    let s = a
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous float32"))?;
    Ok(py.detach(|| stats::kth_values(s, &ks)))
}

/// Cast-shadow margin of a light-aligned height field `(rows, width)` float32:
/// `max(g[x+1 ..= x+steps]) - g[x]` per row with `g = z - x * tan_elev`
/// (positive = in shadow; the last column is -1). Returns the same shape.
#[pyfunction]
fn shadow_margin<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f32>,
    tan_elev: f32,
    steps: usize,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let (rows, width) = (z.shape()[0], z.shape()[1]);
    let zs = z
        .as_slice()
        .map_err(|_| PyValueError::new_err("z must be C-contiguous float32"))?;
    if steps == 0 {
        return Err(PyValueError::new_err("steps must be at least 1"));
    }
    let mut out = vec![0f32; zs.len()];
    py.detach(|| shadow::shadow_margin(zs, width, tan_elev, steps, &mut out));
    PyArray1::from_vec(py, out).reshape([rows, width])
}

/// `ambient` where the margin is positive (shadow), otherwise 1; same shape.
#[pyfunction]
fn shadow_lit<'py>(
    py: Python<'py>,
    margin: PyReadonlyArray2<'py, f32>,
    ambient: f32,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let (rows, width) = (margin.shape()[0], margin.shape()[1]);
    let m = margin
        .as_slice()
        .map_err(|_| PyValueError::new_err("margin must be C-contiguous float32"))?;
    let mut out = vec![0f32; m.len()];
    py.detach(|| shadow::shadow_lit(m, ambient, &mut out));
    PyArray1::from_vec(py, out).reshape([rows, width])
}

/// `clip(max(0, n.l) * lit, 0, 1) ** inv_gain` per pixel; `lit` is `(n_px,)` or
/// any C-contiguous float32 array with one value per pixel. Returns `(n_px,)`.
#[pyfunction]
fn shadow_shade<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f32>,
    light: PyReadonlyArray1<'py, f32>,
    lit: numpy::PyReadonlyArrayDyn<'py, f32>,
    inv_gain: f32,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n = normals_slice(&normals, "normals")?;
    let l = vec3(&light, "light")?;
    let lt = lit
        .as_slice()
        .ok()
        .filter(|s| s.len() * 3 == n.len())
        .ok_or_else(|| {
            PyValueError::new_err("lit must be C-contiguous float32, one value per pixel")
        })?;
    let mut out = vec![0f32; lt.len()];
    py.detach(|| shadow::shadow_shade(n, l, lt, inv_gain, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

/// RTI Specular Enhancement per pixel, `(n_px,)` float32.
///
/// `ptm` is the `(6, n_px)` C-contiguous PTM coefficients (float32 or float64)
/// and `med` a `(n_px,)`/`(h, w)` median image (uint8 or float32, 0..255); with
/// neither, the prediction is the diffuse term and nothing is enhanced.
#[pyfunction]
#[pyo3(signature = (normals, light, floor, coeff, blend, inv_gain, ptm=None, med=None))]
#[allow(clippy::too_many_arguments)]
fn rti_specular<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f32>,
    light: PyReadonlyArray1<'py, f32>,
    floor: f32,
    coeff: f32,
    blend: f32,
    inv_gain: f32,
    ptm: Option<Bound<'py, PyAny>>,
    med: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    use numpy::PyReadonlyArrayDyn;
    let n = normals_slice(&normals, "normals")?;
    let m = n.len() / 3;
    let p = rti::Params {
        light: vec3(&light, "light")?,
        floor,
        coeff,
        blend,
        inv_gain,
    };
    let bad =
        |what: &str| PyValueError::new_err(format!("{what} has the wrong type, shape or layout"));
    // Hold the borrows here so the slices outlive the call.
    let ptm32 = ptm
        .as_ref()
        .and_then(|a| a.extract::<PyReadonlyArray2<'py, f32>>().ok());
    let ptm64 = ptm
        .as_ref()
        .and_then(|a| a.extract::<PyReadonlyArray2<'py, f64>>().ok());
    let med8 = med
        .as_ref()
        .and_then(|a| a.extract::<PyReadonlyArrayDyn<'py, u8>>().ok());
    let med32 = med
        .as_ref()
        .and_then(|a| a.extract::<PyReadonlyArrayDyn<'py, f32>>().ok());
    // (data, row stride, pixel stride) for a (6, n_px) array in either contiguous order
    fn layout<'a, T: numpy::Element>(
        a: &'a PyReadonlyArray2<'_, T>,
        m: usize,
    ) -> Option<(&'a [T], usize, usize)> {
        let st = a.strides();
        if a.shape() != [6, m] || st.iter().any(|&x| x <= 0) || !a.is_contiguous() {
            return None;
        }
        let elem = std::mem::size_of::<T>() as isize;
        let (s0, s1) = ((st[0] / elem) as usize, (st[1] / elem) as usize);
        // SAFETY: the array is contiguous (C or Fortran order) with 6 * m elements,
        // and the read-only borrow `a` keeps NumPy from writing to it meanwhile.
        Some((
            unsafe { std::slice::from_raw_parts(a.data(), 6 * m) },
            s0,
            s1,
        ))
    }
    let src = if ptm.is_some() {
        let l32 = ptm32.as_ref().and_then(|a| layout(a, m));
        let l64 = ptm64.as_ref().and_then(|a| layout(a, m));
        match (l32, l64) {
            (Some((data, s0, s1)), _) => rti::Source::PtmF32 { data, s0, s1 },
            (_, Some((data, s0, s1))) => rti::Source::PtmF64 { data, s0, s1 },
            _ => {
                return Err(bad(
                    "ptm (expected (6, n_px) float32 or float64, contiguous in C or Fortran order)",
                ));
            }
        }
    } else if med.is_some() {
        let sl8 = med8
            .as_ref()
            .and_then(|a| a.as_slice().ok())
            .filter(|s| s.len() == m);
        let sl32 = med32
            .as_ref()
            .and_then(|a| a.as_slice().ok())
            .filter(|s| s.len() == m);
        match (sl8, sl32) {
            (Some(s), _) => rti::Source::MedianU8(s),
            (_, Some(s)) => rti::Source::MedianF32(s),
            _ => return Err(bad("med (expected one uint8 or float32 value per pixel)")),
        }
    } else {
        rti::Source::Diffuse
    };
    let mut out = vec![0f32; m];
    py.detach(|| rti::specular_enhance(n, &src, &p, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

/// The tone composite: `(h, w)` float32 or float64 `shading` to `(h, w, 3)` uint8.
///
/// `layer` is "Combined Colour" (needs `albedo`, `(h, w, 3)` uint8), "Shading",
/// or "Roughness Fusion" (needs `rough`, `(h, w)` uint8). `detail` is the
/// `(h, w)` float64 micro-detail map, used when `micro` is not 0; `lut` a
/// 256-entry uint8 gamma table. Everything is C-contiguous.
#[pyfunction]
#[pyo3(signature = (shading, layer, opacity, colour_gain, rough_opacity, bright, k, micro, skip_tone, albedo=None, rough=None, detail=None, lut=None))]
#[allow(clippy::too_many_arguments)]
fn tone_rgb<'py>(
    py: Python<'py>,
    shading: Bound<'py, PyAny>,
    layer: &str,
    opacity: f64,
    colour_gain: f64,
    rough_opacity: f64,
    bright: f64,
    k: f64,
    micro: f64,
    skip_tone: bool,
    albedo: Option<PyReadonlyArray3<'py, u8>>,
    rough: Option<PyReadonlyArray2<'py, u8>>,
    detail: Option<PyReadonlyArray2<'py, f64>>,
    lut: Option<PyReadonlyArray1<'py, u8>>,
) -> PyResult<Bound<'py, numpy::PyArray3<u8>>> {
    let layer = match layer {
        "Combined Colour" => tone::Layer::Combined,
        "Shading" => tone::Layer::Shading,
        "Roughness Fusion" => tone::Layer::Roughness,
        other => return Err(PyValueError::new_err(format!("unknown layer {other:?}"))),
    };
    let bad = |what: &str| PyValueError::new_err(format!("{what}: wrong type, shape or layout"));
    let s32 = shading.extract::<PyReadonlyArray2<'py, f32>>().ok();
    let s64 = shading.extract::<PyReadonlyArray2<'py, f64>>().ok();
    let (h, w) = match (&s32, &s64) {
        (Some(a), _) => (a.shape()[0], a.shape()[1]),
        (_, Some(a)) => (a.shape()[0], a.shape()[1]),
        _ => return Err(bad("shading (float32 or float64, 2-D)")),
    };
    let n = h * w;
    let alb: &[u8] = match (&albedo, layer) {
        (Some(a), _) => {
            if a.shape() != [h, w, 3] {
                return Err(bad("albedo (expected (h, w, 3))"));
            }
            a.as_slice().map_err(|_| bad("albedo (not C-contiguous)"))?
        }
        (None, tone::Layer::Combined) => return Err(bad("albedo (required for Combined Colour)")),
        (None, _) => &[],
    };
    let rgh: &[u8] = match (&rough, layer) {
        (Some(a), _) => {
            if a.shape() != [h, w] {
                return Err(bad("rough (expected (h, w))"));
            }
            a.as_slice().map_err(|_| bad("rough (not C-contiguous)"))?
        }
        (None, tone::Layer::Roughness) => return Err(bad("rough (required for Roughness Fusion)")),
        (None, _) => &[],
    };
    let det = match &detail {
        Some(d) if micro != 0.0 => {
            if d.shape() != [h, w] {
                return Err(bad("detail (expected (h, w))"));
            }
            Some(d.as_slice().map_err(|_| bad("detail (not C-contiguous)"))?)
        }
        _ => None,
    };
    let table = match &lut {
        Some(l) => Some(
            l.as_slice()
                .ok()
                .filter(|t| t.len() == 256)
                .ok_or_else(|| bad("lut (expected 256 uint8)"))?,
        ),
        None => None,
    };
    let t = tone::Tone {
        layer,
        opacity,
        colour_gain,
        rough_opacity,
        bright,
        k,
        micro,
        detail: det,
        lut: table,
        skip_tone,
    };
    let mut out = vec![0u8; 3 * n];
    if let Some(a) = &s32 {
        let s = a
            .as_slice()
            .map_err(|_| bad("shading (not C-contiguous)"))?;
        py.detach(|| tone::tone(s, alb, rgh, &t, &mut out));
    } else if let Some(a) = &s64 {
        let s = a
            .as_slice()
            .map_err(|_| bad("shading (not C-contiguous)"))?;
        py.detach(|| tone::tone(s, alb, rgh, &t, &mut out));
    }
    PyArray1::from_vec(py, out).reshape([h, w, 3])
}

/// Subsurface Scattering per pixel from `(n_px, 3)` float32 normals, a float64
/// light, the `(n_px,)`/`(h, w)` float32 blurred-albedo luminance and the blend.
/// Returns `(n_px,)` float32.
#[pyfunction]
fn sss<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f32>,
    light: PyReadonlyArray1<'py, f64>,
    scat: numpy::PyReadonlyArrayDyn<'py, f32>,
    blend: f64,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n = normals_slice(&normals, "normals")?;
    let l = match light.as_slice() {
        Ok([x, y, z]) => [*x, *y, *z],
        _ => return Err(PyValueError::new_err("light must be three float64 values")),
    };
    let sc = scat
        .as_slice()
        .ok()
        .filter(|s| s.len() * 3 == n.len())
        .ok_or_else(|| {
            PyValueError::new_err("scat must be C-contiguous float32, one value per pixel")
        })?;
    let mut out = vec![0f32; sc.len()];
    py.detach(|| relight::sss(n, l, sc, blend, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

/// RVT-compatible horizon scan on a surface reflect-padded by `r` pixels.
///
/// `padded` is `(h + 2r, w + 2r)` float32 C-contiguous; `shifts` `(n_points, 2)`
/// int32 (np.roll shifts), `dist` `(n_points,)` float64 and `starts`
/// `(n_dir + 1,)` int64 describe RVT's search points per direction. Returns
/// `(svf, opns)` as `(h, w)` float64 arrays, `None` for the one not asked for.
#[pyfunction]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn horizon_scan<'py>(
    py: Python<'py>,
    padded: PyReadonlyArray2<'py, f32>,
    r: usize,
    shifts: PyReadonlyArray2<'py, i32>,
    dist: PyReadonlyArray1<'py, f64>,
    starts: PyReadonlyArray1<'py, i64>,
    want_svf: bool,
    want_opns: bool,
) -> PyResult<(
    Option<Bound<'py, numpy::PyArray2<f64>>>,
    Option<Bound<'py, numpy::PyArray2<f64>>>,
)> {
    let (ph, pw) = (padded.shape()[0], padded.shape()[1]);
    if ph <= 2 * r || pw <= 2 * r {
        return Err(PyValueError::new_err(
            "padded surface is smaller than twice the radius",
        ));
    }
    let (h, w) = (ph - 2 * r, pw - 2 * r);
    let pad = padded
        .as_slice()
        .map_err(|_| PyValueError::new_err("padded must be C-contiguous float32"))?;
    let sh = shifts
        .as_slice()
        .ok()
        .filter(|_| shifts.shape()[1] == 2)
        .ok_or_else(|| PyValueError::new_err("shifts must be C-contiguous int32 (n, 2)"))?;
    let d = dist
        .as_slice()
        .ok()
        .filter(|d| d.len() * 2 == sh.len() && d.iter().all(|&x| x > 0.0))
        .ok_or_else(|| PyValueError::new_err("dist must be positive float64, one per shift"))?;
    let st: Vec<usize> = starts
        .as_slice()
        .map_err(|_| PyValueError::new_err("starts must be contiguous int64"))?
        .iter()
        .map(|&x| x.max(0) as usize)
        .collect();
    if st.len() < 2 || st.windows(2).any(|p| p[0] > p[1]) || *st.last().unwrap() != d.len() {
        return Err(PyValueError::new_err(
            "starts must be increasing and end at the number of shifts",
        ));
    }
    // every search point must stay inside the padded surface
    if sh
        .chunks_exact(2)
        .any(|s| s[0].unsigned_abs() as usize > r || s[1].unsigned_abs() as usize > r)
    {
        return Err(PyValueError::new_err("a shift is larger than the padding"));
    }
    let mut svf = if want_svf {
        Some(vec![0f64; h * w])
    } else {
        None
    };
    let mut opn = if want_opns {
        Some(vec![0f64; h * w])
    } else {
        None
    };
    let table = horizon::Shifts {
        shifts: sh,
        dist: d,
        starts: &st,
    };
    py.detach(|| horizon::horizon_scan(pad, pw, r, &table, svf.as_deref_mut(), opn.as_deref_mut()));
    let to_arr = |v: Option<Vec<f64>>| {
        v.map(|v| PyArray1::from_vec(py, v).reshape([h, w]))
            .transpose()
    };
    Ok((to_arr(svf)?, to_arr(opn)?))
}

/// Fold one width's Hessian into the running ridge / groove strength and direction, in place.
/// All seven arrays are the same shape, C-contiguous float32; the last four are updated.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn ridge_update<'py>(
    py: Python<'py>,
    hxx: PyReadonlyArray2<'py, f32>,
    hxy: PyReadonlyArray2<'py, f32>,
    hyy: PyReadonlyArray2<'py, f32>,
    sigma2: f32,
    denom: f32,
    mut ridge: PyReadwriteArray2<'py, f32>,
    mut groove: PyReadwriteArray2<'py, f32>,
    mut th_ridge: PyReadwriteArray2<'py, f32>,
    mut th_groove: PyReadwriteArray2<'py, f32>,
) -> PyResult<()> {
    let bad = || PyValueError::new_err("ridge_update: arrays must be C-contiguous float32 of one shape");
    let shape = hxx.shape().to_vec();
    if [hxy.shape(), hyy.shape(), ridge.shape(), groove.shape(), th_ridge.shape(), th_groove.shape()]
        .iter()
        .any(|s| *s != shape.as_slice())
    {
        return Err(bad());
    }
    let (xx, xy, yy) = (
        hxx.as_slice().map_err(|_| bad())?,
        hxy.as_slice().map_err(|_| bad())?,
        hyy.as_slice().map_err(|_| bad())?,
    );
    let (r, g, tr, tg) = (
        ridge.as_slice_mut().map_err(|_| bad())?,
        groove.as_slice_mut().map_err(|_| bad())?,
        th_ridge.as_slice_mut().map_err(|_| bad())?,
        th_groove.as_slice_mut().map_err(|_| bad())?,
    );
    py.detach(|| ridge::ridge_update(xx, xy, yy, sigma2, denom, r, g, tr, tg));
    Ok(())
}

/// The relief panel's grid: `(vertices (gw*gh, 2) float32, triangles (6*(gw-1)*(gh-1),) uint32)`
/// for a `w` x `h` surface sampled every `step` pixels (`step` may be below 1).
#[pyfunction]
fn grid_mesh<'py>(
    py: Python<'py>,
    gw: usize,
    gh: usize,
    w: usize,
    h: usize,
    step: f32,
) -> PyResult<(Bound<'py, numpy::PyArray2<f32>>, Bound<'py, PyArray1<u32>>)> {
    if gw < 2 || gh < 2 || w == 0 || h == 0 || !(step > 0.0) {
        return Err(PyValueError::new_err("grid_mesh: a grid of at least 2 x 2 over a non-empty surface, step > 0"));
    }
    let points = gw.checked_mul(gh).filter(|&n| n <= u32::MAX as usize);
    let Some(points) = points else {
        return Err(PyValueError::new_err("grid_mesh: more vertices than 32-bit indices can number"));
    };
    let mut verts = vec![0f32; points * 2];
    let mut tris = vec![0u32; (gw - 1) * (gh - 1) * 6];
    py.detach(|| gridmesh::grid_mesh(gw, gh, w, h, step, &mut verts, &mut tris));
    Ok((PyArray1::from_vec(py, verts).reshape([points, 2])?, PyArray1::from_vec(py, tris)))
}

/// `max(0, n.l)` per pixel. Returns `(n_px,)` float32.
#[pyfunction]
fn clay_lambert<'py>(
    py: Python<'py>,
    normals: PyReadonlyArray2<'py, f32>,
    light: PyReadonlyArray1<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let n = normals_slice(&normals, "normals")?;
    let l = vec3(&light, "light")?;
    let mut out = vec![0f32; n.len() / 3];
    py.detach(|| relight::lambert(n, l, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

/// Clay Mode's finish: `d`, the optional blurred `low` (same shape), micro-
/// roughness gain, shadow floor and `1 / gain` to a `(h, w, 3)` float32 grey RGB.
#[pyfunction]
#[pyo3(signature = (d, micro, floor, inv_gain, low=None))]
fn clay_finish<'py>(
    py: Python<'py>,
    d: PyReadonlyArray2<'py, f32>,
    micro: f32,
    floor: f64,
    inv_gain: f32,
    low: Option<PyReadonlyArray2<'py, f32>>,
) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
    let (h, w) = (d.shape()[0], d.shape()[1]);
    let dd = d
        .as_slice()
        .map_err(|_| PyValueError::new_err("d must be C-contiguous float32"))?;
    let lo = match &low {
        Some(l) => Some(
            l.as_slice()
                .ok()
                .filter(|s| s.len() == dd.len())
                .ok_or_else(|| {
                    PyValueError::new_err("low must be C-contiguous float32, the shape of d")
                })?,
        ),
        None => None,
    };
    let mut out = vec![0f32; 3 * dd.len()];
    py.detach(|| relight::clay_finish(dd, lo, micro, floor, inv_gain, &mut out));
    PyArray1::from_vec(py, out).reshape([h, w, 3])
}

fn io_err(e: std::io::Error) -> PyErr {
    pyo3::exceptions::PyOSError::new_err(e.to_string())
}

/// Write a Wavefront OBJ. `verts` `(n, 3)` float64 and `faces` `(m, 3)` uint32
/// (zero-based) are C-contiguous; `normals` `(n, 3)` float64, `uvs` `(n, 2)`
/// float64 and `colors` `(n, 3)` uint8 are optional. `face_style` is "v", "v//vn",
/// "v/vt" or "v/vt/vn" (the same index for each). Number formats are `%.Nf`, as
/// Python's. The text before the vertices is `header`, between the vertex data
/// and the faces `mid`, after the faces `footer`.
#[pyfunction]
#[pyo3(signature = (path, verts, faces, header="", decimals=6, normals=None, normal_decimals=6,
                    uvs=None, uv_decimals=6, colors=None, color_decimals=8, face_style="v",
                    blank_after_vertices=false, mid="", footer=""))]
#[allow(clippy::too_many_arguments)]
fn write_obj(
    py: Python<'_>,
    path: &str,
    verts: numpy::PyReadonlyArray2<'_, f64>,
    faces: numpy::PyReadonlyArray2<'_, u32>,
    header: &str,
    decimals: usize,
    normals: Option<numpy::PyReadonlyArray2<'_, f64>>,
    normal_decimals: usize,
    uvs: Option<numpy::PyReadonlyArray2<'_, f64>>,
    uv_decimals: usize,
    colors: Option<numpy::PyReadonlyArray2<'_, u8>>,
    color_decimals: usize,
    face_style: &str,
    blank_after_vertices: bool,
    mid: &str,
    footer: &str,
) -> PyResult<()> {
    let n = verts.shape()[0];
    let bad = |what: &str| PyValueError::new_err(format!("{what}: wrong type, shape or layout"));
    if verts.shape()[1] != 3 || faces.shape()[1] != 3 {
        return Err(bad("verts and faces must be (n, 3)"));
    }
    let vs = verts
        .as_slice()
        .map_err(|_| bad("verts (C-contiguous float64)"))?;
    let fs = faces
        .as_slice()
        .map_err(|_| bad("faces (C-contiguous uint32)"))?;
    if fs.iter().any(|&i| i as usize >= n) {
        return Err(PyValueError::new_err(
            "a face refers to a vertex that does not exist",
        ));
    }
    let side = |a: &Option<numpy::PyReadonlyArray2<'_, f64>>,
                cols: usize,
                what: &str|
     -> PyResult<Option<Vec<f64>>> {
        match a {
            None => Ok(None),
            Some(x) if x.shape() == [n, cols] => {
                Ok(Some(x.as_slice().map_err(|_| bad(what))?.to_vec()))
            }
            Some(_) => Err(bad(what)),
        }
    };
    let nv = side(&normals, 3, "normals (n, 3) float64")?;
    let uv = side(&uvs, 2, "uvs (n, 2) float64")?;
    let col = match &colors {
        None => None,
        Some(c) if c.shape() == [n, 3] => Some(c.as_slice().map_err(|_| bad("colors"))?.to_vec()),
        Some(_) => return Err(bad("colors (n, 3) uint8")),
    };
    let style = match face_style {
        "v" => meshio::FaceStyle::V,
        "v//vn" => meshio::FaceStyle::VN,
        "v/vt" => meshio::FaceStyle::VT,
        "v/vt/vn" => meshio::FaceStyle::VTN,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown face_style {other:?}"
            )));
        }
    };
    if (matches!(style, meshio::FaceStyle::VT | meshio::FaceStyle::VTN) && uv.is_none())
        || (matches!(style, meshio::FaceStyle::VN | meshio::FaceStyle::VTN) && nv.is_none())
    {
        return Err(PyValueError::new_err(
            "face_style needs the normals / uvs it refers to",
        ));
    }
    let o = meshio::ObjOptions {
        header,
        decimals,
        normal_decimals,
        uv_decimals,
        color_decimals,
        face_style: style,
        blank_after_vertices,
        mid,
        footer,
    };
    py.detach(|| {
        meshio::write_obj(
            path,
            vs,
            fs,
            nv.as_deref(),
            uv.as_deref(),
            col.as_deref(),
            &o,
        )
    })
    .map_err(io_err)
}

/// Write a binary STL: `verts` `(n, 3)` float64, `faces` `(m, 3)` uint32. Normals
/// are the normalised cross products ((0, 0, 1) for a degenerate triangle).
#[pyfunction]
#[pyo3(signature = (path, verts, faces, header=""))]
fn write_stl(
    py: Python<'_>,
    path: &str,
    verts: numpy::PyReadonlyArray2<'_, f64>,
    faces: numpy::PyReadonlyArray2<'_, u32>,
    header: &str,
) -> PyResult<()> {
    let n = verts.shape()[0];
    if verts.shape()[1] != 3 || faces.shape()[1] != 3 {
        return Err(PyValueError::new_err("verts and faces must be (n, 3)"));
    }
    let vs = verts
        .as_slice()
        .map_err(|_| PyValueError::new_err("verts must be C-contiguous float64"))?;
    let fs = faces
        .as_slice()
        .map_err(|_| PyValueError::new_err("faces must be C-contiguous uint32"))?;
    if fs.iter().any(|&i| i as usize >= n) {
        return Err(PyValueError::new_err(
            "a face refers to a vertex that does not exist",
        ));
    }
    py.detach(|| meshio::write_stl(path, header.as_bytes(), vs, fs))
        .map_err(io_err)
}

/// `(n, 3)` light rows as a contiguous slice, or a ValueError.
fn lights_slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray2<'_, T>, n: usize) -> PyResult<&'a [T]> {
    if a.shape() != [n, 3] {
        return Err(PyValueError::new_err(format!("lights must be ({n}, 3) to match the images")));
    }
    a.as_slice().map_err(|_| PyValueError::new_err("lights must be C-contiguous"))
}

/// One alternation of the semi-calibrated solve (`robust_ps.light_strengths`):
/// lights `(n, 3)` float64, strengths `(n,)` float64, the `(n, n_px)` float32
/// sample of the stack and the trusted window `lo < I < hi`.  Returns each
/// lamp's `(numerator, denominator)` as two `(n,)` float64 arrays.
#[pyfunction]
fn semi_cal_step<'py>(
    py: Python<'py>,
    lights: PyReadonlyArray2<'py, f64>,
    strengths: PyReadonlyArray1<'py, f64>,
    images: PyReadonlyArray2<'py, f32>,
    lo: f64,
    hi: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (n, m) = (images.shape()[0], images.shape()[1]);
    let l = lights_slice(&lights, n)?;
    let e = strengths
        .as_slice()
        .map_err(|_| PyValueError::new_err("strengths must be C-contiguous float64"))?;
    if e.len() != n {
        return Err(PyValueError::new_err(format!("strengths must be ({n},)")));
    }
    let img = images
        .as_slice()
        .map_err(|_| PyValueError::new_err("images must be C-contiguous float32"))?;
    let (num, den) = py.detach(|| robust::semi_cal_step(l, e, img, n, m, lo, hi));
    Ok((PyArray1::from_vec(py, num), PyArray1::from_vec(py, den)))
}

/// Sparse-outlier least-squares normals (`robust_ps.sparse_outlier_normals`):
/// lights `(n, 3)` and their pseudo-inverse `(3, n)`, float32; the
/// `(n, n_px)` float32 stack; the outlier threshold `lam` (a fraction of each
/// pixel's mean), the most iterations and the tolerance.  Returns `(n_px, 3)`.
#[pyfunction]
fn sparse_ps<'py>(
    py: Python<'py>,
    lights: PyReadonlyArray2<'py, f32>,
    pinv: PyReadonlyArray2<'py, f32>,
    images: PyReadonlyArray2<'py, f32>,
    lam: f32,
    iters: usize,
    tol: f32,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let (n, m) = (images.shape()[0], images.shape()[1]);
    let l = lights_slice(&lights, n)?;
    if pinv.shape() != [3, n] {
        return Err(PyValueError::new_err(format!("pinv must be (3, {n})")));
    }
    let p = pinv
        .as_slice()
        .map_err(|_| PyValueError::new_err("pinv must be C-contiguous float32"))?;
    let img = images
        .as_slice()
        .map_err(|_| PyValueError::new_err("images must be C-contiguous float32"))?;
    let mut out = vec![0f32; 3 * m];
    py.detach(|| robust::sparse_ps(l, p, img, n, m, lam, iters, tol, &mut out));
    PyArray1::from_vec(py, out).reshape([m, 3])
}

/// The specular excess of every pixel (`robust_ps.specular_excess`): lights
/// `(n, 3)`, normals `(n_px, 3)` and the `(n, n_px)` stack, all float32.
/// Returns `(n_px,)` float32.
#[pyfunction]
fn specular_excess<'py>(
    py: Python<'py>,
    lights: PyReadonlyArray2<'py, f32>,
    normals: PyReadonlyArray2<'py, f32>,
    images: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let (n, m) = (images.shape()[0], images.shape()[1]);
    let l = lights_slice(&lights, n)?;
    if normals.shape() != [m, 3] {
        return Err(PyValueError::new_err(format!("normals must be ({m}, 3)")));
    }
    let nv = normals_slice(&normals, "normals")?;
    let img = images
        .as_slice()
        .map_err(|_| PyValueError::new_err("images must be C-contiguous float32"))?;
    let mut out = vec![0f32; m];
    py.detach(|| robust::specular_excess(l, nv, img, n, m, &mut out));
    Ok(PyArray1::from_vec(py, out))
}

#[pymodule]
fn helio_core_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_thread_pool();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(threads, m)?)?;
    m.add_function(wrap_pyfunction!(lls_normals, m)?)?;
    m.add_function(wrap_pyfunction!(apply_pinv, m)?)?;
    m.add_function(wrap_pyfunction!(standard_pbr, m)?)?;
    m.add_function(wrap_pyfunction!(kth_values, m)?)?;
    m.add_function(wrap_pyfunction!(write_obj, m)?)?;
    m.add_function(wrap_pyfunction!(write_stl, m)?)?;
    m.add_function(wrap_pyfunction!(clay_lambert, m)?)?;
    m.add_function(wrap_pyfunction!(clay_finish, m)?)?;
    m.add_function(wrap_pyfunction!(horizon_scan, m)?)?;
    m.add_function(wrap_pyfunction!(ridge_update, m)?)?;
    m.add_function(wrap_pyfunction!(grid_mesh, m)?)?;
    m.add_function(wrap_pyfunction!(sss, m)?)?;
    m.add_function(wrap_pyfunction!(tone_rgb, m)?)?;
    m.add_function(wrap_pyfunction!(rti_specular, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_margin, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_lit, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_shade, m)?)?;
    m.add_function(wrap_pyfunction!(hillshade, m)?)?;
    m.add_function(wrap_pyfunction!(multi_lambert, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_gain, m)?)?;
    m.add_function(wrap_pyfunction!(scaled_lambert, m)?)?;
    m.add_function(wrap_pyfunction!(semi_cal_step, m)?)?;
    m.add_function(wrap_pyfunction!(sparse_ps, m)?)?;
    m.add_function(wrap_pyfunction!(specular_excess, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_threads_is_at_least_one() {
        assert!(default_threads() >= 1);
    }
}
