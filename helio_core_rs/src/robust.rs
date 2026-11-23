//! Semi-calibrated strengths, sparse-outlier normals and the specular excess.
//!
//! The NumPy twins, and what each computes, are in `helio/core/robust_ps.py`.
//! All three read the `(n_lights, n_px)` row-major stack, where one pixel's
//! readings are `n_px` apart.  Each block of `BLOCK` pixels is first copied
//! pixel-major into a small buffer, so the per-pixel loops (an iteration per
//! pixel for the sparse solve) run over contiguous memory; blocks are spread
//! over Rayon threads.

use rayon::prelude::*;

/// Pixels per block: `BLOCK * n_lights` f32 stay in L2 for any real dome.
const BLOCK: usize = 512;

/// Copy pixels `start..start + len` of the `(n, m)` stack pixel-major into `buf`.
fn gather(images: &[f32], n: usize, m: usize, start: usize, len: usize, buf: &mut Vec<f32>) {
    buf.clear();
    buf.resize(len * n, 0.0);
    for j in 0..n {
        let row = &images[j * m + start..j * m + start + len];
        for (p, &v) in row.iter().enumerate() {
            buf[p * n + j] = v;
        }
    }
}

/// Solve the symmetric 3x3 system `a x = r`; None when it is (nearly) singular.
fn solve3(a: &[f64; 9], r: &[f64; 3]) -> Option<[f64; 3]> {
    let c00 = a[4] * a[8] - a[5] * a[7];
    let c01 = a[5] * a[6] - a[3] * a[8];
    let c02 = a[3] * a[7] - a[4] * a[6];
    let det = a[0] * c00 + a[1] * c01 + a[2] * c02;
    if !(det > 1e-12) {
        return None;
    }
    let inv = [
        c00,
        a[2] * a[7] - a[1] * a[8],
        a[1] * a[5] - a[2] * a[4],
        c01,
        a[0] * a[8] - a[2] * a[6],
        a[2] * a[3] - a[0] * a[5],
        c02,
        a[1] * a[6] - a[0] * a[7],
        a[0] * a[4] - a[1] * a[3],
    ];
    let mut x = [0f64; 3];
    for (i, xi) in x.iter_mut().enumerate() {
        *xi = (inv[3 * i] * r[0] + inv[3 * i + 1] * r[1] + inv[3 * i + 2] * r[2]) / det;
    }
    Some(x)
}

/// One alternation of the semi-calibrated solve: every pixel's scaled normal
/// from the strengths `e`, then each lamp's `(numerator, denominator)` of its
/// least-squares strength, summed over the pixels.  A reading counts when it
/// lies strictly between `lo` and `hi` and the pixel faces the lamp.
pub fn semi_cal_step(
    l: &[f64],
    e: &[f64],
    images: &[f32],
    n: usize,
    m: usize,
    lo: f64,
    hi: f64,
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(l.len(), 3 * n);
    debug_assert_eq!(e.len(), n);
    debug_assert_eq!(images.len(), n * m);
    let le: Vec<f64> = (0..3 * n).map(|k| l[k] * e[k / 3]).collect();
    (0..m.div_ceil(BLOCK))
        .into_par_iter()
        .fold(
            || (vec![0f64; n], vec![0f64; n], Vec::new()),
            |(mut num, mut den, mut buf), b| {
                let start = b * BLOCK;
                let len = BLOCK.min(m - start);
                gather(images, n, m, start, len, &mut buf);
                for p in 0..len {
                    let px = &buf[p * n..(p + 1) * n];
                    let mut a = [0f64; 9];
                    let mut r = [0f64; 3];
                    for (j, &v) in px.iter().enumerate() {
                        let v = v as f64;
                        if !(v > lo && v < hi) {
                            continue;
                        }
                        let (x, y, z) = (le[3 * j], le[3 * j + 1], le[3 * j + 2]);
                        a[0] += x * x;
                        a[1] += x * y;
                        a[2] += x * z;
                        a[4] += y * y;
                        a[5] += y * z;
                        a[8] += z * z;
                        r[0] += x * v;
                        r[1] += y * v;
                        r[2] += z * v;
                    }
                    a[3] = a[1];
                    a[6] = a[2];
                    a[7] = a[5];
                    let Some(g) = solve3(&a, &r) else { continue };
                    for (j, &v) in px.iter().enumerate() {
                        let v = v as f64;
                        if !(v > lo && v < hi) {
                            continue;
                        }
                        let s = l[3 * j] * g[0] + l[3 * j + 1] * g[1] + l[3 * j + 2] * g[2];
                        if s > 0.0 {
                            num[j] += v * s;
                            den[j] += s * s;
                        }
                    }
                }
                (num, den, buf)
            },
        )
        .map(|(num, den, _)| (num, den))
        .reduce(
            || (vec![0f64; n], vec![0f64; n]),
            |(mut a, mut b), (c, d)| {
                for j in 0..n {
                    a[j] += c[j];
                    b[j] += d[j];
                }
                (a, b)
            },
        )
}

/// Sparse-outlier least-squares normals: for every pixel, alternate
/// `b = P (I - e)` and `e = soft(I - L b, lam * mean(I))` until no error moves
/// by more than `tol * mean(I)`, or `iters` times; then `b / |b|`.
///
/// * `l`: `(n, 3)` lights; `pinv`: `(3, n)` their pseudo-inverse.
/// * `out`: `(m, 3)` unit normals (zero where `b` is zero).
#[allow(clippy::too_many_arguments)]
pub fn sparse_ps(
    l: &[f32],
    pinv: &[f32],
    images: &[f32],
    n: usize,
    m: usize,
    lam: f32,
    iters: usize,
    tol: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(l.len(), 3 * n);
    debug_assert_eq!(pinv.len(), 3 * n);
    debug_assert_eq!(out.len(), 3 * m);
    out.par_chunks_mut(3 * BLOCK).enumerate().for_each(|(blk, dst)| {
        let start = blk * BLOCK;
        let len = dst.len() / 3;
        let mut buf = Vec::new();
        gather(images, n, m, start, len, &mut buf);
        let mut e = vec![0f32; n];
        let mut d = vec![0f32; n];
        for p in 0..len {
            let px = &buf[p * n..(p + 1) * n];
            let scale = (px.iter().sum::<f32>() / n as f32).max(1e-6);
            let floor = SHADOW_FRACTION * px.iter().copied().fold(0f32, f32::max);
            e.fill(0.0);
            for k in 0..iters {
                let frac = continuation(lam, k);
                let thr = frac * scale;
                for j in 0..n {
                    d[j] = px[j] - e[j];
                }
                let b = apply3(pinv, &d, n);
                let mut moved = 0f32;
                for j in 0..n {
                    let res = px[j] - (l[3 * j] * b[0] + l[3 * j + 1] * b[1] + l[3 * j + 2] * b[2]);
                    // a shadow is missing, not a reading: its error takes all of it
                    let new = if px[j] > floor { res.signum() * (res.abs() - thr).max(0.0) } else { res };
                    moved = moved.max((new - e[j]).abs());
                    e[j] = new;
                }
                if frac <= lam && moved / scale < tol {
                    break;
                }
            }
            for j in 0..n {
                d[j] = px[j] - e[j];
            }
            let b = refit_inliers(l, px, &e, apply3(pinv, &d, n));
            let mag = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
            let o = &mut dst[3 * p..3 * p + 3];
            if mag > 0.0 {
                o[0] = b[0] / mag;
                o[1] = b[1] / mag;
                o[2] = b[2] / mag;
            } else {
                o.fill(0.0);
            }
        }
    });
}

/// The threshold at iteration `k` (`continuation`): from twice the mean down
/// to `lam`, by a quarter a step, so the grossest outliers go first.
fn continuation(lam: f32, k: usize) -> f32 {
    lam.max(2.0 * 0.75f32.powi(k as i32))
}

/// A reading below this share of the pixel's brightest is shadow (`SHADOW_FRACTION`).
const SHADOW_FRACTION: f32 = 0.02;

/// Least squares again on the readings the sparse error left alone and above
/// the shadows (`_refit_inliers`): `b` when fewer than three remain.
fn refit_inliers(l: &[f32], px: &[f32], e: &[f32], b: [f32; 3]) -> [f32; 3] {
    let floor = SHADOW_FRACTION * px.iter().copied().fold(0f32, f32::max);
    let mut a = [0f64; 9];
    let mut r = [0f64; 3];
    let mut used = 0;
    for (j, &v) in px.iter().enumerate() {
        if e[j] != 0.0 || !(v > floor) {
            continue;
        }
        used += 1;
        let (x, y, z) = (l[3 * j] as f64, l[3 * j + 1] as f64, l[3 * j + 2] as f64);
        let v = v as f64;
        a[0] += x * x;
        a[1] += x * y;
        a[2] += x * z;
        a[4] += y * y;
        a[5] += y * z;
        a[8] += z * z;
        r[0] += x * v;
        r[1] += y * v;
        r[2] += z * v;
    }
    if used < 3 {
        return b;
    }
    a[3] = a[1];
    a[6] = a[2];
    a[7] = a[5];
    match solve3(&a, &r) {
        Some(g) => [g[0] as f32, g[1] as f32, g[2] as f32],
        None => b,
    }
}

/// `pinv @ v` for the `(3, n)` pseudo-inverse.
fn apply3(pinv: &[f32], v: &[f32], n: usize) -> [f32; 3] {
    let mut b = [0f32; 3];
    for (r, br) in b.iter_mut().enumerate() {
        let row = &pinv[r * n..(r + 1) * n];
        *br = row.iter().zip(v).map(|(p, x)| p * x).sum();
    }
    b
}

/// The specular excess of every pixel: with `s_j = max(0, L_j . N)` and the
/// best albedo `rho = sum(I s) / sum(s^2)`, `max_j(I_j - rho s_j) / mean(I)`.
pub fn specular_excess(l: &[f32], normals: &[f32], images: &[f32], n: usize, m: usize, out: &mut [f32]) {
    debug_assert_eq!(normals.len(), 3 * m);
    debug_assert_eq!(out.len(), m);
    out.par_chunks_mut(BLOCK).enumerate().for_each(|(blk, dst)| {
        let start = blk * BLOCK;
        let len = dst.len();
        let mut buf = Vec::new();
        gather(images, n, m, start, len, &mut buf);
        let mut s = vec![0f32; n];
        for p in 0..len {
            let px = &buf[p * n..(p + 1) * n];
            let nv = &normals[3 * (start + p)..3 * (start + p) + 3];
            let (mut is, mut ss, mut sum) = (0f32, 0f32, 0f32);
            for j in 0..n {
                let v = (l[3 * j] * nv[0] + l[3 * j + 1] * nv[1] + l[3 * j + 2] * nv[2]).max(0.0);
                s[j] = v;
                is += px[j] * v;
                ss += v * v;
                sum += px[j];
            }
            let rho = if ss > 0.0 { is / ss.max(1e-12) } else { 0.0 };
            let mean = (sum / n as f32).max(1e-6);
            let mut top = f32::NEG_INFINITY;
            for j in 0..n {
                top = top.max(px[j] - rho * s[j]);
            }
            dst[p] = top / mean;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matte sphere under known lights with unequal strengths: the
    /// alternation finds the strengths back.
    fn sphere(n_side: usize, lights: &[[f32; 3]], gain: &[f32]) -> (Vec<f32>, Vec<f32>, usize) {
        let mut normals = Vec::new();
        for y in 0..n_side {
            for x in 0..n_side {
                let u = (x as f32 + 0.5) / n_side as f32 * 1.6 - 0.8;
                let v = (y as f32 + 0.5) / n_side as f32 * 1.6 - 0.8;
                let z = (1.0 - u * u - v * v).max(0.05).sqrt();
                let mag = (u * u + v * v + z * z).sqrt();
                normals.extend_from_slice(&[u / mag, v / mag, z / mag]);
            }
        }
        let m = n_side * n_side;
        let n = lights.len();
        let mut img = vec![0f32; n * m];
        for j in 0..n {
            for p in 0..m {
                let d = lights[j][0] * normals[3 * p] + lights[j][1] * normals[3 * p + 1]
                    + lights[j][2] * normals[3 * p + 2];
                img[j * m + p] = 200.0 * gain[j] * d.max(0.0);
            }
        }
        (img, normals, m)
    }

    fn lights() -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        for k in 0..8 {
            let az = k as f32 * std::f32::consts::TAU / 8.0;
            let el = if k % 2 == 0 { 0.6f32 } else { 1.0f32 };
            out.push([el.cos() * az.cos(), el.cos() * az.sin(), el.sin()]);
        }
        out
    }

    #[test]
    fn semi_calibrated_recovers_lamp_strengths() {
        let ls = lights();
        let gain = [1.0, 1.3, 0.8, 1.1, 0.9, 1.25, 1.0, 0.85];
        let (img, _, m) = sphere(48, &ls, &gain);
        let n = ls.len();
        let l: Vec<f64> = ls.iter().flat_map(|v| v.iter().map(|&x| x as f64)).collect();
        let mut e = vec![1f64; n];
        for _ in 0..60 {
            let (num, den) = semi_cal_step(&l, &e, &img, n, m, 1.0, 1e9);
            let mut new: Vec<f64> = num.iter().zip(&den).map(|(a, b)| a / b).collect();
            let g = (new.iter().map(|v| v.ln()).sum::<f64>() / n as f64).exp();
            new.iter_mut().for_each(|v| *v /= g);
            e = new;
        }
        let g = (gain.iter().map(|&v| (v as f64).ln()).sum::<f64>() / n as f64).exp();
        for j in 0..n {
            assert!((e[j] - gain[j] as f64 / g).abs() < 5e-3, "lamp {j}: {} vs {}", e[j], gain[j] as f64 / g);
        }
    }

    #[test]
    fn sparse_ps_ignores_a_highlight() {
        let ls = lights();
        let (mut img, normals, m) = sphere(16, &ls, &[1.0; 8]);
        let n = ls.len();
        for p in 0..m {
            img[2 * m + p] += 400.0; // a burning highlight in one frame, everywhere
        }
        let l: Vec<f32> = ls.iter().flatten().copied().collect();
        // pseudo-inverse by the normal equations (well conditioned here)
        let mut a = [0f64; 9];
        for v in &ls {
            for r in 0..3 {
                for c in 0..3 {
                    a[3 * r + c] += v[r] as f64 * v[c] as f64;
                }
            }
        }
        let mut pinv = vec![0f32; 3 * n];
        for j in 0..n {
            for r in 0..3 {
                let mut e = [0f64; 3];
                e[r] = 1.0;
                let col = solve3(&a, &e).unwrap();
                pinv[r * n + j] =
                    (col[0] * ls[j][0] as f64 + col[1] * ls[j][1] as f64 + col[2] * ls[j][2] as f64) as f32;
            }
        }
        let mut out = vec![0f32; 3 * m];
        sparse_ps(&l, &pinv, &img, n, m, 0.3, 100, 1e-5, &mut out);
        let mut worst = 0f32;
        for p in 0..m {
            let dot: f32 = (0..3).map(|k| out[3 * p + k] * normals[3 * p + k]).sum();
            worst = worst.max(dot.clamp(-1.0, 1.0).acos().to_degrees());
        }
        assert!(worst < 1.0, "worst error {worst} degrees");
    }

    #[test]
    fn excess_is_zero_on_a_matte_surface() {
        let ls = lights();
        let (img, normals, m) = sphere(16, &ls, &[1.0; 8]);
        let l: Vec<f32> = ls.iter().flatten().copied().collect();
        let mut out = vec![0f32; m];
        specular_excess(&l, &normals, &img, ls.len(), m, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-3));
    }
}
