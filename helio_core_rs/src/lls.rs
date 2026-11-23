//! Baseline Linear Least-Squares normals.
//!
//! The Python side computes the pseudo-inverse of the (small) light matrix
//! once, with `np.linalg.pinv`, so rank-deficient light sets give the same
//! minimum-norm answer as `np.linalg.lstsq`.  What is left, and what costs
//! the time, is `G = P @ I` for every pixel followed by the unit-length
//! normalisation.  That is what this does, in one pass over the image stack.
//!
//! Layout: `I` is `(n_lights, n_px)` row-major, so for one light the pixels
//! are contiguous.  Pixels are handled in blocks of `BLOCK`; for each light
//! the block's three accumulators are updated by a straight loop over
//! contiguous memory, which the compiler turns into SIMD multiply-adds.
//! Blocks are spread over Rayon threads, each writing its own slice of the
//! `(n_px, 3)` output.

use rayon::prelude::*;

/// Pixels per block: three f32 accumulators of this size stay in L1.
const BLOCK: usize = 1024;

/// Solve normals for every pixel.
///
/// * `pinv`: `(3, n)` row-major pseudo-inverse of the light matrix.
/// * `images`: `(n, m)` row-major intensities.
/// * `out`: `(m, 3)` row-major unit normals (zero where the solution is zero).
pub fn lls_normals(pinv: &[f32], n: usize, images: &[f32], m: usize, out: &mut [f32]) {
    debug_assert_eq!(pinv.len(), 3 * n);
    debug_assert_eq!(images.len(), n * m);
    debug_assert_eq!(out.len(), 3 * m);
    out.par_chunks_mut(3 * BLOCK)
        .enumerate()
        .for_each(|(b, dst)| {
            let start = b * BLOCK;
            let len = dst.len() / 3;
            let mut a0 = [0f32; BLOCK];
            let mut a1 = [0f32; BLOCK];
            let mut a2 = [0f32; BLOCK];
            let (a0, a1, a2) = (&mut a0[..len], &mut a1[..len], &mut a2[..len]);
            for i in 0..n {
                let row = &images[i * m + start..i * m + start + len];
                let (p0, p1, p2) = (pinv[i], pinv[n + i], pinv[2 * n + i]);
                for j in 0..len {
                    let v = row[j];
                    a0[j] += p0 * v;
                    a1[j] += p1 * v;
                    a2[j] += p2 * v;
                }
            }
            for j in 0..len {
                let (x, y, z) = (a0[j], a1[j], a2[j]);
                let mag = (x * x + y * y + z * z).sqrt();
                let o = &mut dst[3 * j..3 * j + 3];
                if mag != 0.0 {
                    o[0] = x / mag;
                    o[1] = y / mag;
                    o[2] = z / mag;
                } else {
                    o.fill(0.0);
                }
            }
        });
}

/// A raw output pointer that parallel blocks write disjoint parts of.
struct SharedOut(*mut f32);
// SAFETY: every block writes only its own column range of every row (see
// `apply_pinv`), so no two threads touch the same element.
unsafe impl Sync for SharedOut {}

/// Pixels per block for `apply_pinv`: `k` accumulators of this size stay in L1.
const BLOCK_K: usize = 1024;

/// `out = pinv @ images` for any number of unknowns: `pinv` is `(k, n)`,
/// `images` `(n, m)` and `out` `(k, m)`, all row-major.  The least-squares
/// coefficients of every pixel (PTM: k = 6, HSH: k = 9) in one pass over the
/// stack, blocks of pixels spread over Rayon.
pub fn apply_pinv(pinv: &[f32], k: usize, n: usize, images: &[f32], m: usize, out: &mut [f32]) {
    debug_assert_eq!(pinv.len(), k * n);
    debug_assert_eq!(images.len(), n * m);
    debug_assert_eq!(out.len(), k * m);
    let dst = SharedOut(out.as_mut_ptr());
    let dst = &dst;
    (0..m.div_ceil(BLOCK_K)).into_par_iter().for_each(|b| {
        let start = b * BLOCK_K;
        let len = BLOCK_K.min(m - start);
        let mut acc = vec![0f32; k * len];
        for i in 0..n {
            let row = &images[i * m + start..i * m + start + len];
            for (r, a) in acc.chunks_exact_mut(len).enumerate() {
                let p = pinv[r * n + i];
                for (x, &v) in a.iter_mut().zip(row) {
                    *x += p * v;
                }
            }
        }
        for (r, a) in acc.chunks_exact(len).enumerate() {
            // SAFETY: row r, columns start..start + len belong to this block only.
            unsafe { std::ptr::copy_nonoverlapping(a.as_ptr(), dst.0.add(r * m + start), len) };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lights along x, y, z: the pseudo-inverse is the identity, so the
    /// normal is the (normalised) intensity triple.
    #[test]
    fn apply_pinv_is_a_matrix_product_over_ragged_blocks() {
        let (k, n, m) = (3, 2, 2 * BLOCK_K + 7);
        let pinv = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let images: Vec<f32> = (0..n * m).map(|i| (i % 97) as f32).collect();
        let mut out = vec![0f32; k * m];
        apply_pinv(&pinv, k, n, &images, m, &mut out);
        for j in [0, 1, BLOCK_K - 1, BLOCK_K, m - 1] {
            let (a, b) = (images[j], images[m + j]);
            assert_eq!(
                [out[j], out[m + j], out[2 * m + j]],
                [a, b, 0.5 * a + 0.5 * b],
                "pixel {j}"
            );
        }
    }

    #[test]
    fn identity_lights_return_normalised_intensities() {
        let pinv = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let m = 2 * BLOCK + 5; // several blocks and a ragged tail
        let mut images = vec![0f32; 3 * m];
        for j in 0..m {
            images[j] = 3.0;
            images[m + j] = 0.0;
            images[2 * m + j] = 4.0;
        }
        images[m + 7] = 0.0;
        for k in 0..3 {
            images[k * m + 3] = 0.0; // an all-zero pixel
        }
        let mut out = vec![9f32; 3 * m];
        lls_normals(&pinv, 3, &images, m, &mut out);
        assert_eq!(&out[0..3], &[0.6, 0.0, 0.8]);
        assert_eq!(&out[9..12], &[0.0, 0.0, 0.0]);
        let last = 3 * (m - 1);
        assert_eq!(&out[last..last + 3], &[0.6, 0.0, 0.8]);
    }
}
