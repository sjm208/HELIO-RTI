//! Ridge and Groove Detector: the per-pixel step
//! (helio/core/shading_engines.py, `ridge_response`).
//!
//! The Hessian of the height at one width is made with OpenCV's separable
//! filters; this turns it into line strength and direction and keeps the best
//! over the widths, in place.  The arithmetic is the NumPy version's, in
//! float32, in the same order:
//!
//! ```text
//! m = (hxx + hyy) / 2        q = hypot((hxx - hyy) / 2, hxy)
//! l1 = m + q   l2 = m - q    theta = atan2(2 hxy, hxx - hyy) / 2     (the l1 axis)
//! ridge:  a = -l2 sigma^2    b = |l1| sigma^2
//! groove: a =  l1 sigma^2    b = |l2| sigma^2
//! resp = a exp(-(b/a)^2 / (2 beta^2))   if a > 0, else 0
//! ```
//!
//! A response replaces the stored one only when it is larger (`resp > acc`,
//! so NaN never replaces anything), and takes its `theta` with it.

use rayon::prelude::*;

/// Pixels per Rayon task: big enough to be worth a task, small enough to share the work out.
const BLOCK: usize = 1 << 15;

#[inline]
fn one(
    hxx: f32,
    hxy: f32,
    hyy: f32,
    s2: f32,
    denom: f32,
    acc: &mut [&mut f32; 4], // ridge, groove, th_ridge, th_groove
) {
    let m = (hxx + hyy) * 0.5;
    let q = ((hxx - hyy) * 0.5).hypot(hxy);
    let l1 = m + q;
    let l2 = m - q;
    let theta = 0.5 * (2.0 * hxy).atan2(hxx - hyy);
    for (a, b, ai, ti) in [(-l2, l1, 0usize, 2usize), (l1, l2, 1, 3)] {
        let a = a * s2;
        let b = b.abs() * s2;
        let resp = if a > 0.0 {
            let rb = b / a;
            a * (-(rb * rb) / denom).exp()
        } else {
            0.0
        };
        if resp > *acc[ai] {
            *acc[ai] = resp;
            *acc[ti] = theta;
        }
    }
}

/// Fold one width's Hessian into the running best.  All slices have the same length.
#[allow(clippy::too_many_arguments)]
pub fn ridge_update(
    hxx: &[f32],
    hxy: &[f32],
    hyy: &[f32],
    sigma2: f32,
    denom: f32,
    ridge: &mut [f32],
    groove: &mut [f32],
    th_ridge: &mut [f32],
    th_groove: &mut [f32],
) {
    hxx.par_chunks(BLOCK)
        .zip(hxy.par_chunks(BLOCK))
        .zip(hyy.par_chunks(BLOCK))
        .zip(ridge.par_chunks_mut(BLOCK))
        .zip(groove.par_chunks_mut(BLOCK))
        .zip(th_ridge.par_chunks_mut(BLOCK))
        .zip(th_groove.par_chunks_mut(BLOCK))
        .for_each(|((((((xx, xy), yy), r), g), tr), tg)| {
            for i in 0..xx.len() {
                // each pixel's four accumulators, borrowed one by one
                let (mut r_i, mut g_i, mut tr_i, mut tg_i) = (r[i], g[i], tr[i], tg[i]);
                one(xx[i], xy[i], yy[i], sigma2, denom,
                    &mut [&mut r_i, &mut g_i, &mut tr_i, &mut tg_i]);
                r[i] = r_i;
                g[i] = g_i;
                tr[i] = tr_i;
                tg[i] = tg_i;
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(hxx: f32, hxy: f32, hyy: f32) -> (f32, f32, f32, f32) {
        let (mut r, mut g, mut tr, mut tg) = ([0f32], [0f32], [0f32], [0f32]);
        ridge_update(&[hxx], &[hxy], &[hyy], 1.0, 0.5, &mut r, &mut g, &mut tr, &mut tg);
        (r[0], g[0], tr[0], tg[0])
    }

    #[test]
    fn raised_line_is_a_ridge_not_a_groove() {
        // strong downward bend across x, none along y
        let (r, g, _, _) = run(-1.0, 0.0, 0.0);
        assert!(r > 0.9 && g == 0.0, "{r} {g}");
    }

    #[test]
    fn incised_line_is_a_groove() {
        let (r, g, _, _) = run(0.0, 0.0, 1.0);
        assert!(g > 0.9 && r == 0.0, "{r} {g}");
    }

    #[test]
    fn round_bump_is_suppressed() {
        // equal curvature both ways: exp(-1 / (2 beta^2)) = e^-2 of what a line of that strength gets
        let (r, _, _, _) = run(-1.0, 0.0, -1.0);
        let (line, _, _, _) = run(-1.0, 0.0, 0.0);
        assert!((r / line - (-2.0f32).exp()).abs() < 1e-5, "{r} {line}");
    }

    #[test]
    fn a_weaker_width_does_not_replace_a_stronger_one() {
        let (mut r, mut g, mut tr, mut tg) = ([0f32], [0f32], [0f32], [0f32]);
        ridge_update(&[-2.0], &[0.0], &[0.0], 1.0, 0.5, &mut r, &mut g, &mut tr, &mut tg);
        let first = r[0];
        ridge_update(&[-1.0], &[0.0], &[0.0], 1.0, 0.5, &mut r, &mut g, &mut tr, &mut tg);
        assert_eq!(r[0], first);
    }

    #[test]
    fn nan_never_replaces_anything() {
        let (mut r, mut g, mut tr, mut tg) = ([0.5f32], [0f32], [0.3f32], [0f32]);
        ridge_update(&[f32::NAN], &[0.0], &[0.0], 1.0, 0.5, &mut r, &mut g, &mut tr, &mut tg);
        assert_eq!((r[0], tr[0]), (0.5, 0.3));
    }
}
