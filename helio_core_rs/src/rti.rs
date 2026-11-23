//! RTI Specular Enhancement (helio/core/shading_engines.py).
//!
//! Per pixel: the diffuse term `d = max(0, n.l)`, the "predicted" brightness
//! (the PTM polynomial at the light's (u, v), a median image, or `d` itself),
//! and the part of the prediction that the diffuse term cannot explain,
//! `max(0, predicted - d)`, which above a floor is taken for specular and
//! amplified.  Arithmetic is done in f32, in the order the NumPy engine does
//! it; only the PTM polynomial runs in the coefficients' own type (f32 or f64).

use rayon::prelude::*;

const BLOCK: usize = 4096;

/// Where the prediction comes from.
pub enum Source<'a> {
    Diffuse,
    /// `(6, n_px)` coefficients in any contiguous memory order: element
    /// `(k, i)` is `data[k * s0 + i * s1]`.
    PtmF32 {
        data: &'a [f32],
        s0: usize,
        s1: usize,
    },
    PtmF64 {
        data: &'a [f64],
        s0: usize,
        s1: usize,
    },
    MedianU8(&'a [u8]),
    MedianF32(&'a [f32]),
}

pub struct Params {
    pub light: [f32; 3],
    pub floor: f32,
    pub coeff: f32,
    pub blend: f32,
    pub inv_gain: f32,
}

/// The six-term PTM polynomial `c0 u^2 + c1 v^2 + c2 u v + c3 u + c4 v + c5`,
/// summed left to right with `u v` as `(c2 u) v`, as `compute_ptm_relighting`.
macro_rules! ptm_poly {
    ($c:expr, $s0:expr, $s1:expr, $i:expr, $t:ty, $u:expr, $v:expr) => {{
        let (u, v) = ($u as $t, $v as $t);
        let (uu, vv) = (($u * $u) as $t, ($v * $v) as $t);
        let c = |k: usize| $c[k * $s0 + $i * $s1];
        (c(0) * uu + c(1) * vv + c(2) * u * v + c(3) * u + c(4) * v + c(5)) as f32
    }};
}

pub fn specular_enhance(normals: &[f32], src: &Source, p: &Params, out: &mut [f32]) {
    let [lx, ly, lz] = p.light;
    let (u, v) = (lx, ly);
    let unit = p.inv_gain == 1.0;
    out.par_chunks_mut(BLOCK).enumerate().for_each(|(b, o)| {
        let s = b * BLOCK;
        for (j, r) in o.iter_mut().enumerate() {
            let i = s + j;
            let d =
                (normals[3 * i] * lx + normals[3 * i + 1] * ly + normals[3 * i + 2] * lz).max(0.0);
            let predicted = match src {
                Source::Diffuse => d,
                Source::PtmF32 { data, s0, s1 } => {
                    (ptm_poly!(data, *s0, *s1, i, f32, u, v) / 255.0).clamp(0.0, 1.0)
                }
                Source::PtmF64 { data, s0, s1 } => {
                    (ptm_poly!(data, *s0, *s1, i, f64, u, v) / 255.0).clamp(0.0, 1.0)
                }
                Source::MedianU8(m) => (m[i] as f32 / 255.0).clamp(0.0, 1.0),
                Source::MedianF32(m) => (m[i] / 255.0).clamp(0.0, 1.0),
            };
            let raw = (predicted - d).max(0.0);
            let mask = if raw > p.floor { 1.0 } else { 0.0 };
            let amp = raw * mask * p.coeff;
            let mixed = d * (1.0 - p.blend) + (d + amp).clamp(0.0, 1.0) * p.blend;
            let t = mixed.clamp(0.0, 1.0);
            *r = if unit { t } else { t.powf(p.inv_gain) };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(blend: f32) -> Params {
        Params {
            light: [0.0, 0.0, 1.0],
            floor: 0.05,
            coeff: 2.0,
            blend,
            inv_gain: 1.0,
        }
    }

    #[test]
    fn diffuse_source_has_no_specular() {
        let n = [0.0, 0.0, 0.5, 0.0, 0.0, 1.0];
        let mut out = [0f32; 2];
        specular_enhance(&n, &Source::Diffuse, &params(0.6), &mut out);
        assert_eq!(out, [0.5, 1.0]);
    }

    #[test]
    fn median_brighter_than_diffuse_adds_amplified_specular() {
        // d = 0.5, predicted = 0.7 -> raw 0.2 > floor, amp 0.4; blend 1 -> clip(0.9)
        let n = [0.0, 0.0, 0.5];
        let m = [(0.7f32 * 255.0) as u8];
        let mut out = [0f32; 1];
        specular_enhance(&n, &Source::MedianU8(&m), &params(1.0), &mut out);
        assert!((out[0] - 0.9).abs() < 0.01, "{}", out[0]);
    }

    #[test]
    fn pixel_major_layout_reads_the_same_values() {
        // (6, 2) coefficients stored pixel-major: element (k, i) at k + 6 i
        let n = [0.0, 0.0, 0.2, 0.0, 0.0, 0.2];
        let mut c = vec![0f32; 12];
        c[5] = 255.0 * 0.6;
        c[6 + 5] = 255.0 * 0.3;
        let mut out = [0f32; 2];
        specular_enhance(
            &n,
            &Source::PtmF32 {
                data: &c,
                s0: 1,
                s1: 6,
            },
            &params(1.0),
            &mut out,
        );
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - (0.2 + 0.1 * 2.0)).abs() < 1e-5);
    }

    #[test]
    fn ptm_constant_term_is_the_prediction() {
        let n = [0.0, 0.0, 0.2, 0.0, 0.0, 0.2];
        let mut c = vec![0f32; 12];
        c[5 * 2..5 * 2 + 2].fill(255.0 * 0.6); // c5 = 0.6 * 255 for both pixels
        let mut out = [0f32; 2];
        specular_enhance(
            &n,
            &Source::PtmF32 {
                data: &c,
                s0: 2,
                s1: 1,
            },
            &params(1.0),
            &mut out,
        );
        assert!((out[0] - 1.0).abs() < 1e-5); // 0.2 + (0.6-0.2)*2 = 1.0
        let c64: Vec<f64> = c.iter().map(|&x| x as f64).collect();
        let mut out64 = [0f32; 2];
        specular_enhance(
            &n,
            &Source::PtmF64 {
                data: &c64,
                s0: 2,
                s1: 1,
            },
            &params(1.0),
            &mut out64,
        );
        assert!((out64[0] - out[0]).abs() < 1e-5);
    }
}
