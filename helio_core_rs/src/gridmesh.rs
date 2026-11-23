//! The 3D relief panel's mesh: a regular grid of vertices over the surface and the
//! triangles that join them (helio_qt/relief3d.py, `_upload`).
//!
//! The same arrays the NumPy code makes, bit for bit, in one parallel pass and with no
//! intermediate arrays (the NumPy version builds a meshgrid, a stack, four index views and
//! a second stack: about 0.9 s and 3x the memory for 4 million points, every time the
//! density changes).
//!
//! * vertex `(j, i)` sits at `u = min((i * step + 0.5) / w, 1)`, `v = min((j * step + 0.5) / h, 1)`,
//!   float32 arithmetic in that order (`step` may be below 1: more vertices than pixels, which
//!   the height texture's linear filtering fills in);
//! * each cell `(j, i)` makes two triangles `a c b` and `b c d`, with `a = j*gw + i`, `b = a + 1`,
//!   `c = a + gw`, `d = c + 1`: 6 indices a cell.

use rayon::prelude::*;

pub fn grid_mesh(gw: usize, gh: usize, w: usize, h: usize, step: f32, verts: &mut [f32], tris: &mut [u32]) {
    let (wf, hf) = (w as f32, h as f32);
    let us: Vec<f32> = (0..gw).map(|i| (((i as f32) * step + 0.5) / wf).min(1.0)).collect();
    verts
        .par_chunks_mut(2 * gw)
        .enumerate()
        .for_each(|(j, row)| {
            let v = (((j as f32) * step + 0.5) / hf).min(1.0);
            for (i, p) in row.chunks_exact_mut(2).enumerate() {
                p[0] = us[i];
                p[1] = v;
            }
        });
    if gw < 2 || gh < 2 {
        return;
    }
    tris.par_chunks_mut(6 * (gw - 1))
        .enumerate()
        .for_each(|(j, row)| {
            for (i, t) in row.chunks_exact_mut(6).enumerate() {
                let a = (j * gw + i) as u32;
                let (b, c) = (a + 1, a + gw as u32);
                let d = c + 1;
                t.copy_from_slice(&[a, c, b, b, c, d]);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_by_two_is_two_triangles() {
        let (mut v, mut t) = (vec![0f32; 8], vec![0u32; 6]);
        grid_mesh(2, 2, 2, 2, 1.0, &mut v, &mut t);
        assert_eq!(t, vec![0, 2, 1, 1, 2, 3]);
        assert_eq!(v, vec![0.25, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75]);
    }

    #[test]
    fn positions_stop_at_the_edge() {
        let (gw, gh) = (5usize, 3usize);
        let (mut v, mut t) = (vec![0f32; gw * gh * 2], vec![0u32; (gw - 1) * (gh - 1) * 6]);
        grid_mesh(gw, gh, 4, 2, 1.0, &mut v, &mut t);
        assert!(v.iter().all(|&x| (0.0..=1.0).contains(&x)));
        assert_eq!(*v.last().unwrap(), 1.0);
    }

    #[test]
    fn half_steps_have_twice_the_vertices_between() {
        let (gw, gh) = (7usize, 7usize);
        let (mut v, mut t) = (vec![0f32; gw * gh * 2], vec![0u32; (gw - 1) * (gh - 1) * 6]);
        grid_mesh(gw, gh, 4, 4, 0.5, &mut v, &mut t);
        assert_eq!(v[2] - v[0], 0.5 / 4.0);
        assert!(t.iter().all(|&i| (i as usize) < gw * gh));
    }
}
