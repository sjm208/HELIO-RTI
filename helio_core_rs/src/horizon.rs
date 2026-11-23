//! Horizon scan: sky-view factor and openness
//! (helio/core/shading_engines.py, `_svf_bundle`).
//!
//! The same computation as RVT's `sky_view_factor_compute` (Relief
//! Visualization Toolbox), one pass per pixel instead of a full-image
//! `np.roll` / `np.fmax` for every direction and radius.  RVT's own shift
//! table (`horizon_shift_vector`) is passed in, so directions, radii and their
//! order are exactly RVT's; what is done here is the arithmetic:
//!
//! * `slope = (height[shifted] - height[centre]) / radius`: the difference in
//!   float32 (the surface is float32), the division in float64 (the radius is a
//!   float64), as NumPy does it;
//! * the maximum over radii starts at -1000 and ignores NaN (`np.fmax`);
//! * per direction `angle = atan(max_slope)`; SVF sums `1 - sin(max(angle, 0))`,
//!   openness sums `angle`.
//!
//! `padded` is the surface reflect-padded by `r` on every side, so the scan
//! never leaves it.

use rayon::prelude::*;

pub struct Shifts<'a> {
    /// `(row, column)` shift per search point, flat, `np.roll` convention
    /// (`out[x] = in[x - shift]`).
    pub shifts: &'a [i32],
    /// Distance per search point.
    pub dist: &'a [f64],
    /// Index of each direction's first search point; `n_dir + 1` entries.
    pub starts: &'a [usize],
}

/// Returns nothing; `svf` and `opns` (each `h * w`, float64) are filled when
/// given.  `h = padded_h - 2 r`, `w = padded_w - 2 r`.
pub fn horizon_scan(
    padded: &[f32],
    padded_w: usize,
    r: usize,
    sh: &Shifts,
    svf: Option<&mut [f64]>,
    opns: Option<&mut [f64]>,
) {
    let h = padded.len() / padded_w - 2 * r;
    let w = padded_w - 2 * r;
    let n_dir = sh.starts.len() - 1;
    let ndir = n_dir as f64;
    let rows: Vec<usize> = (0..h).collect();
    // Split the outputs into one slice per row so rows can run in parallel.
    let svf_rows: Vec<Option<&mut [f64]>> = match svf {
        Some(s) => s.chunks_mut(w).map(Some).collect(),
        None => rows.iter().map(|_| None).collect(),
    };
    let opn_rows: Vec<Option<&mut [f64]>> = match opns {
        Some(s) => s.chunks_mut(w).map(Some).collect(),
        None => rows.iter().map(|_| None).collect(),
    };
    rows.into_par_iter()
        .zip(svf_rows)
        .zip(opn_rows)
        .for_each_init(
            || (vec![0f64; w], vec![0f64; w], vec![0f64; w]),
            |(max_slope, sum_svf, sum_opn), ((i, mut sv), mut op)| {
                let ci = i + r;
                let centre = &padded[ci * padded_w + r..ci * padded_w + r + w];
                sum_svf.fill(0.0);
                sum_opn.fill(0.0);
                for d in 0..n_dir {
                    max_slope.fill(-1000.0);
                    for k in sh.starts[d]..sh.starts[d + 1] {
                        let (s0, s1) = (sh.shifts[2 * k] as isize, sh.shifts[2 * k + 1] as isize);
                        let row = (ci as isize - s0) as usize * padded_w;
                        let from = (row as isize + r as isize - s1) as usize;
                        let src = &padded[from..from + w];
                        let dist = sh.dist[k];
                        // contiguous in j: the compiler turns this into SIMD
                        for ((m, &v), &c) in max_slope.iter_mut().zip(src).zip(centre) {
                            *m = m.max((v - c) as f64 / dist); // f64::max ignores NaN, like np.fmax
                        }
                    }
                    for ((&m, svf), opn) in max_slope
                        .iter()
                        .zip(sum_svf.iter_mut())
                        .zip(sum_opn.iter_mut())
                    {
                        *opn += m.atan();
                        // 1 - sin(max(atan m, 0)); sin(atan m) = m / sqrt(1 + m^2)
                        *svf += 1.0
                            - if m > 0.0 {
                                m / (1.0 + m * m).sqrt()
                            } else {
                                0.0
                            };
                    }
                }
                if let Some(s) = sv.as_deref_mut() {
                    for (o, &t) in s.iter_mut().zip(sum_svf.iter()) {
                        *o = t / ndir;
                    }
                }
                if let Some(o) = op.as_deref_mut() {
                    for (x, &t) in o.iter_mut().zip(sum_opn.iter()) {
                        *x = (0.5 * std::f64::consts::PI - t / ndir).to_degrees();
                    }
                }
            },
        );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat surface sees the whole sky: SVF 1, openness 90 degrees.
    #[test]
    fn flat_surface_is_fully_open() {
        let (h, w, r) = (6usize, 7usize, 2usize);
        let padded = vec![5f32; (h + 2 * r) * (w + 2 * r)];
        // two directions: +row and +column, radii 1 and 2
        let shifts = [1, 0, 2, 0, 0, 1, 0, 2];
        let dist = [1.0, 2.0, 1.0, 2.0];
        let starts = [0usize, 2, 4];
        let (mut svf, mut opn) = (vec![0f64; h * w], vec![0f64; h * w]);
        horizon_scan(
            &padded,
            w + 2 * r,
            r,
            &Shifts {
                shifts: &shifts,
                dist: &dist,
                starts: &starts,
            },
            Some(&mut svf),
            Some(&mut opn),
        );
        assert!(svf.iter().all(|&v| (v - 1.0).abs() < 1e-12));
        assert!(opn.iter().all(|&v| (v - 90.0).abs() < 1e-9));
    }

    /// A pixel in a pit, with a wall one step away in the single direction
    /// scanned, has horizon angle atan(slope).
    #[test]
    fn a_wall_raises_the_horizon() {
        let (h, w, r) = (1usize, 1usize, 1usize);
        let mut padded = vec![0f32; 9];
        padded[3 + 1 - 1] = 0.0;
        // np.roll shift (0, 1): out[x] = in[x - (0, 1)], i.e. the pixel to the left
        padded[3] = 2.0; // row 1, column 0
        let sh = Shifts {
            shifts: &[0, 1],
            dist: &[1.0],
            starts: &[0, 1],
        };
        let (mut svf, mut opn) = (vec![0f64; 1], vec![0f64; 1]);
        horizon_scan(&padded, 3, r, &sh, Some(&mut svf), Some(&mut opn));
        let angle = 2f64.atan();
        assert!((svf[0] - (1.0 - angle.sin())).abs() < 1e-12, "{}", svf[0]);
        assert!((opn[0] - (0.5 * std::f64::consts::PI - angle).to_degrees()).abs() < 1e-9);
        let _ = (h, w);
    }
}
