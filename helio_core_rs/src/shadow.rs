//! Cast-shadow raking (helio/core/shading_engines.py, `_cast_shadow_mask`).
//!
//! The height field has already been turned so the light runs along +x (that
//! rotation stays with OpenCV: its bilinear sampling is fixed-point and not
//! worth imitating).  In that frame a pixel is shadowed when something within
//! `steps` pixels ahead of it, rising `tan_elev` per pixel less than the
//! light, is higher: with `g(x) = z(x) - x * tan_elev`, when
//! `max(g[x+1 ..= x+steps]) > g[x]`.  `margin` is that difference; a positive
//! margin is shadow.

use rayon::prelude::*;
use std::collections::VecDeque;

const BLOCK: usize = 4096;

/// For every row of the `(rows, width)` height field `z`: `margin[x] =
/// max(g[x+1 ..= x+steps]) - g[x]` with `g = z - x * tan_elev` in f32 (as the
/// NumPy/OpenCV version computes it), and `-1` in the last column.
/// Rows are independent, so they are spread over Rayon; each is one right-to-
/// left sweep with a monotonic deque, O(width) however large `steps` is.
pub fn shadow_margin(z: &[f32], width: usize, tan_elev: f32, steps: usize, margin: &mut [f32]) {
    debug_assert_eq!(z.len(), margin.len());
    if width == 0 {
        return;
    }
    margin
        .par_chunks_mut(width)
        .zip(z.par_chunks(width))
        .for_each_init(
            || (VecDeque::<usize>::new(), Vec::<f32>::new()),
            |(dq, g), (m, zr)| {
                g.clear();
                g.extend(zr.iter().enumerate().map(|(x, &v)| v - x as f32 * tan_elev));
                dq.clear();
                m[width - 1] = -1.0;
                // window for x is g[x+1 ..= x+steps]; sweep x from the right
                for x in (0..width - 1).rev() {
                    let incoming = x + 1;
                    while dq.back().is_some_and(|&b| g[b] <= g[incoming]) {
                        dq.pop_back();
                    }
                    dq.push_back(incoming);
                    while dq.front().is_some_and(|&f| f > x + steps) {
                        dq.pop_front();
                    }
                    m[x] = g[*dq.front().unwrap()] - g[x];
                }
            },
        );
}

/// `lit = if margin > 0 { ambient } else { 1 }`.
pub fn shadow_lit(margin: &[f32], ambient: f32, out: &mut [f32]) {
    out.par_chunks_mut(BLOCK)
        .zip(margin.par_chunks(BLOCK))
        .for_each(|(o, m)| {
            for (v, &g) in o.iter_mut().zip(m) {
                *v = if g > 0.0 { ambient } else { 1.0 };
            }
        });
}

/// `clip(max(0, n.l) * lit, 0, 1) ^ inv_gain` per pixel.
pub fn shadow_shade(normals: &[f32], light: [f32; 3], lit: &[f32], inv_gain: f32, out: &mut [f32]) {
    let unit = inv_gain == 1.0;
    out.par_chunks_mut(BLOCK)
        .zip(lit.par_chunks(BLOCK))
        .enumerate()
        .for_each(|(b, (o, lt))| {
            let s = b * BLOCK;
            for (j, (v, &l)) in o.iter_mut().zip(lt).enumerate() {
                let i = 3 * (s + j);
                let d =
                    (normals[i] * light[0] + normals[i + 1] * light[1] + normals[i + 2] * light[2])
                        .max(0.0);
                let t = (d * l).clamp(0.0, 1.0);
                *v = if unit { t } else { t.powf(inv_gain) };
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straight-from-the-definition margin.
    fn brute(z: &[f32], width: usize, tan: f32, steps: usize) -> Vec<f32> {
        let mut out = vec![0f32; z.len()];
        for (r, zr) in z.chunks(width).enumerate() {
            let g: Vec<f32> = zr
                .iter()
                .enumerate()
                .map(|(x, &v)| v - x as f32 * tan)
                .collect();
            for x in 0..width {
                out[r * width + x] = if x == width - 1 {
                    -1.0
                } else {
                    let hi = (x + steps).min(width - 1);
                    g[x + 1..=hi]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max)
                        - g[x]
                };
            }
        }
        out
    }

    #[test]
    fn margin_matches_the_definition() {
        let mut s = 7u32;
        let (rows, width) = (5, 211);
        let z: Vec<f32> = (0..rows * width)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                (s >> 8) as f32 / (1u32 << 24) as f32 * 30.0
            })
            .collect();
        for steps in [1, 2, 7, 48, 210, 500] {
            let mut got = vec![0f32; z.len()];
            shadow_margin(&z, width, 0.4, steps, &mut got);
            assert_eq!(got, brute(&z, width, 0.4, steps), "steps {steps}");
        }
    }

    #[test]
    fn a_ridge_shades_the_pixels_behind_it() {
        // light along +x, 45 degrees; a 10-high step at x=20 hides x in 10..20
        let mut z = vec![0f32; 40];
        z[20..].fill(10.0);
        let mut m = vec![0f32; 40];
        shadow_margin(&z, 40, 1.0, 48, &mut m);
        assert!(m[12] > 0.0 && m[19] > 0.0 && m[9] <= 0.0, "{:?}", &m[8..21]);
    }

    #[test]
    fn lit_and_shade() {
        let mut lit = [0f32; 3];
        shadow_lit(&[1.0, 0.0, -2.0], 0.2, &mut lit);
        assert_eq!(lit, [0.2, 1.0, 1.0]);
        let n = [0.0, 0.0, 1.0, 0.0, 0.0, -1.0];
        let mut out = [0f32; 2];
        shadow_shade(&n, [0.0, 0.0, 1.0], &[0.5, 1.0], 1.0, &mut out);
        assert_eq!(out, [0.5, 0.0]);
        shadow_shade(&n, [0.0, 0.0, 1.0], &[0.25, 1.0], 0.5, &mut out);
        assert_eq!(out, [0.5, 0.0]);
    }
}
