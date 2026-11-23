//! Live relighting kernels: per-pixel shading of unit normals under one or
//! several lights (helio/core/shading_engines.py).
//!
//! Normals are an `(n_px, 3)` row-major f32 slice.  Every kernel is one pass
//! that writes straight into an `(n_px,)` f32 output, split in blocks over
//! Rayon, with no temporaries; the NumPy engines they stand in for make three
//! to ten full-size temporaries per frame.

use rayon::prelude::*;

/// Pixels per parallel block.
const BLOCK: usize = 4096;

#[inline(always)]
fn dot(n: &[f32], i: usize, l: &[f32; 3]) -> f32 {
    n[3 * i] * l[0] + n[3 * i + 1] * l[1] + n[3 * i + 2] * l[2]
}

/// `out[i] = f(n_i . light)` for every pixel.
fn map_dot<F>(normals: &[f32], light: [f32; 3], out: &mut [f32], f: F)
where
    F: Fn(f32) -> f32 + Sync,
{
    debug_assert_eq!(normals.len(), 3 * out.len());
    out.par_chunks_mut(BLOCK).enumerate().for_each(|(b, o)| {
        let s = b * BLOCK;
        for (j, v) in o.iter_mut().enumerate() {
            *v = f(dot(normals, s + j, &light));
        }
    });
}

/// Standard PBR through a lookup table: `lut[min(max(n.l, 0) * len, len - 1)]`,
/// where `lut` is `x ** inv_gain` over 0..1 as `_pbr_lut` builds it.  (The
/// exact power is left to NumPy, whose SIMD `power` beats a scalar `powf`.)
pub fn standard_pbr(normals: &[f32], light: [f32; 3], lut: &[f32], out: &mut [f32]) {
    let n = lut.len();
    let top = n as i32 - 1;
    map_dot(normals, light, out, |d| {
        let d = d.max(0.0);
        lut[((d * n as f32) as i32).min(top) as usize]
    });
}

/// Single-direction hillshade: `clip(clip(|n.l|, 0, 1) + ambient, 0, 1)`.
pub fn hillshade(normals: &[f32], light: [f32; 3], ambient: f32, out: &mut [f32]) {
    map_dot(normals, light, out, |d| {
        (d.abs().clamp(0.0, 1.0) + ambient).clamp(0.0, 1.0)
    });
}

/// Weighted sum of `max(0, n.l_k)` over lights, accumulated light by light in
/// order (as the NumPy loop does), and the largest term when `max_out` is given.
pub fn multi_lambert(
    normals: &[f32],
    lights: &[[f32; 3]],
    weights: &[f32],
    sum_out: &mut [f32],
    max_out: Option<&mut [f32]>,
) {
    debug_assert_eq!(lights.len(), weights.len());
    match max_out {
        None => sum_out
            .par_chunks_mut(BLOCK)
            .enumerate()
            .for_each(|(b, sum)| {
                let s = b * BLOCK;
                for (j, v) in sum.iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for (l, &w) in lights.iter().zip(weights) {
                        acc += dot(normals, s + j, l).max(0.0) * w;
                    }
                    *v = acc;
                }
            }),
        Some(mx_out) => sum_out
            .par_chunks_mut(BLOCK)
            .zip(mx_out.par_chunks_mut(BLOCK))
            .enumerate()
            .for_each(|(b, (sum, mx))| {
                let s = b * BLOCK;
                for (j, (v, m)) in sum.iter_mut().zip(mx.iter_mut()).enumerate() {
                    let (mut acc, mut top) = (0f32, 0f32);
                    for (l, &w) in lights.iter().zip(weights) {
                        let t = dot(normals, s + j, l).max(0.0);
                        acc += t * w;
                        top = top.max(t);
                    }
                    *v = acc;
                    *m = top;
                }
            }),
    }
}

/// Diffuse Gain before its percentile stretch:
/// `clip(max(0, mean_k - spec * max_k) * strength, 0, 1) ^ gamma`
/// over the given lights (`spec` of 0.01 or less skips the subtraction).
pub fn diffuse_gain(
    normals: &[f32],
    lights: &[[f32; 3]],
    strength: f32,
    gamma: f32,
    spec: f32,
    out: &mut [f32],
) {
    let k = lights.len() as f32;
    out.par_chunks_mut(BLOCK).enumerate().for_each(|(b, o)| {
        let s = b * BLOCK;
        for (j, v) in o.iter_mut().enumerate() {
            let (mut sum, mut top) = (0f32, 0f32);
            for l in lights {
                let t = dot(normals, s + j, l).max(0.0);
                sum += t;
                top = top.max(t);
            }
            let mut avg = sum / k;
            if spec > 0.01 {
                avg = (avg - spec * top).max(0.0);
            }
            *v = (avg * strength).clamp(0.0, 1.0).powf(gamma);
        }
    });
}

/// Exaggerated Shading's per-scale sum: `sum_i w_i * max(0, p_i . l) / total`,
/// where `p_i` is the i-th scale's pre-rotated normals.
pub fn scaled_lambert(
    scales: &[&[f32]],
    light: [f32; 3],
    weights: &[f32],
    total: f32,
    out: &mut [f32],
) {
    let inv = 1.0 / total;
    out.par_chunks_mut(BLOCK).enumerate().for_each(|(b, o)| {
        let s = b * BLOCK;
        for (j, v) in o.iter_mut().enumerate() {
            let mut acc = 0f32;
            for (p, &w) in scales.iter().zip(weights) {
                acc += dot(p, s + j, &light).max(0.0) * w;
            }
            *v = acc * inv;
        }
    });
}

/// Subsurface Scattering per pixel:
/// `d = clip(|n.l|, 0, 1)`, `out = clip(d * (1 - b) + (scat * b) * d, 0, 1)`.
///
/// As the NumPy engine does it, `n.l` and `d` are float64 (the light is), and
/// `scat * b` is float32 times the float32-rounded blend (a float32 array
/// times a Python float).
pub fn sss(normals: &[f32], light: [f64; 3], scat: &[f32], blend: f64, out: &mut [f32]) {
    let b32 = blend as f32;
    out.par_chunks_mut(BLOCK)
        .zip(scat.par_chunks(BLOCK))
        .enumerate()
        .for_each(|(b, (o, sc))| {
            let s = b * BLOCK;
            for (j, (v, &c)) in o.iter_mut().zip(sc).enumerate() {
                let i = 3 * (s + j);
                let dot = normals[i] as f64 * light[0]
                    + normals[i + 1] as f64 * light[1]
                    + normals[i + 2] as f64 * light[2];
                let d = dot.abs().clamp(0.0, 1.0);
                let shaded = d * (1.0 - blend) + (c * b32) as f64 * d;
                *v = (shaded as f32).clamp(0.0, 1.0);
            }
        });
}

/// `max(0, n.l)` per pixel (Lambertian shading with a white albedo).
pub fn lambert(normals: &[f32], light: [f32; 3], out: &mut [f32]) {
    map_dot(normals, light, out, |d| d.max(0.0));
}

/// Clay Mode's last step, writing grey as RGB:
/// `grey = clip(d + (d - low) * micro, 0, 1)`, `floor + (1 - floor) * grey`,
/// then `^ inv_gain` unless it is 1.  `low` is the blurred `d` (absent when
/// `micro` is 0).  float32 throughout, the scalars rounded to float32 as NumPy
/// does for an array times a Python float.
pub fn clay_finish(
    d: &[f32],
    low: Option<&[f32]>,
    micro: f32,
    floor: f64,
    inv_gain: f32,
    out: &mut [f32],
) {
    let floor32 = floor as f32;
    let span = (1.0 - floor) as f32;
    out.par_chunks_mut(3 * BLOCK)
        .zip(d.par_chunks(BLOCK))
        .enumerate()
        .for_each(|(b, (o, dd))| {
            for (j, (px, &v)) in o.chunks_exact_mut(3).zip(dd).enumerate() {
                let v = match low {
                    Some(l) => v + (v - l[b * BLOCK + j]) * micro,
                    None => v,
                };
                let g = floor32 + span * v.clamp(0.0, 1.0);
                px.fill(if inv_gain == 1.0 { g } else { g.powf(inv_gain) });
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(m: usize) -> Vec<f32> {
        // unit normals tilting from +z towards +x
        (0..m)
            .flat_map(|i| {
                let t = i as f32 / m as f32 * 1.4;
                [t.sin(), 0.0, t.cos()]
            })
            .collect()
    }

    #[test]
    fn standard_pbr_follows_the_table_across_blocks() {
        let m = 3 * BLOCK + 11;
        let n = ramp(m);
        let l = [0.6, 0.0, 0.8];
        let lut: Vec<f32> = (0..4096).map(|i| (i as f32 / 4095.0).powi(2)).collect();
        let mut out = vec![0f32; m];
        standard_pbr(&n, l, &lut, &mut out);
        for i in [0, 1, BLOCK - 1, BLOCK, m - 1] {
            let d = (n[3 * i] * l[0] + n[3 * i + 2] * l[2]).max(0.0);
            assert!((out[i] - d * d).abs() < 1e-3, "pixel {i}");
        }
    }

    #[test]
    fn table_clamps_both_ends() {
        let t: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let n = [0.0, 0.0, 1.0, 0.0, 0.0, -1.0];
        let mut out = [0f32; 2];
        standard_pbr(&n, [0.0, 0.0, 1.0], &t, &mut out);
        assert_eq!(out, [15.0, 0.0]); // d=1 clamps to the last entry; d<0 -> entry 0
    }

    #[test]
    fn multi_lambert_sum_and_max() {
        let n = [0.0, 0.0, 1.0];
        let lights = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]];
        let (mut s, mut m) = ([0f32], [0f32]);
        multi_lambert(&n, &lights, &[2.0, 1.0, 1.0], &mut s, Some(&mut m));
        assert_eq!((s[0], m[0]), (2.0, 1.0));
    }

    #[test]
    fn clay_floor_and_micro() {
        let d = [0.0f32, 0.5, 2.0];
        let mut out = [0f32; 9];
        clay_finish(&d, None, 0.0, 0.2, 1.0, &mut out);
        assert_eq!(out, [0.2, 0.2, 0.2, 0.6, 0.6, 0.6, 1.0, 1.0, 1.0]);
        // micro 1 with low = 0.25: d + (d - 0.25) = 0.75 for d = 0.5
        clay_finish(&[0.5], Some(&[0.25]), 1.0, 0.0, 1.0, &mut out[..3]);
        assert_eq!(&out[..3], &[0.75, 0.75, 0.75]);
    }

    #[test]
    fn sss_blends_the_scattered_light() {
        let n = [0.0, 0.0, 1.0, 0.0, 0.0, 0.5];
        let mut out = [0f32; 2];
        sss(&n, [0.0, 0.0, 1.0], &[1.0, 0.0], 0.5, &mut out);
        assert_eq!(out, [1.0, 0.25]); // d=1: 0.5 + 1*0.5*1 ; d=0.5, scat=0: 0.5*0.5
    }

    #[test]
    fn scaled_lambert_weights_scales() {
        let a = [0.0f32, 0.0, 1.0];
        let b = [1.0f32, 0.0, 0.0];
        let mut out = [0f32];
        scaled_lambert(&[&a, &b], [0.0, 0.0, 1.0], &[1.0, 0.5], 1.5, &mut out);
        assert!((out[0] - 1.0 / 1.5).abs() < 1e-7);
    }
}
