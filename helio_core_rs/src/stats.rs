//! Exact order statistics of a large f32 array, without sorting it.
//!
//! Values are mapped to u32 keys that order like the floats do; a parallel
//! histogram over the top 16 bits locates the bin holding each requested
//! rank, and only that bin's values are gathered and selected.  The answer is
//! exactly what a full sort would put at that rank.

use rayon::prelude::*;

const BINS: usize = 1 << 16;
const CHUNK: usize = 1 << 16;

#[inline(always)]
fn key(x: f32) -> u32 {
    let b = x.to_bits();
    if b >> 31 == 1 { !b } else { b | 0x8000_0000 }
}

/// The values at ranks `ks` (0-based, ascending order) of `a`.
/// `None` if `a` has a NaN or an infinity (the caller falls back to NumPy,
/// which defines what those mean) or a rank is out of range.
pub fn kth_values(a: &[f32], ks: &[usize]) -> Option<Vec<f32>> {
    if ks.iter().any(|&k| k >= a.len()) {
        return None;
    }
    let (hist, finite) = a
        .par_chunks(CHUNK)
        .fold(
            || (vec![0u32; BINS], true),
            |(mut h, mut ok), c| {
                for &x in c {
                    ok &= x.is_finite();
                    h[(key(x) >> 16) as usize] += 1;
                }
                (h, ok)
            },
        )
        .reduce(
            || (vec![0u32; BINS], true),
            |(mut h, ok), (g, ok2)| {
                h.iter_mut().zip(&g).for_each(|(x, y)| *x += y);
                (h, ok && ok2)
            },
        );
    if !finite {
        return None;
    }
    // bin and rank-within-bin of every requested rank
    let mut cum = vec![0usize; BINS];
    let mut run = 0usize;
    for (c, &h) in cum.iter_mut().zip(&hist) {
        run += h as usize;
        *c = run;
    }
    let place: Vec<(usize, usize)> = ks
        .iter()
        .map(|&k| {
            let b = cum.partition_point(|&c| c <= k);
            (b, k - (cum[b] - hist[b] as usize))
        })
        .collect();
    let mut bins: Vec<usize> = place.iter().map(|p| p.0).collect();
    bins.sort_unstable();
    bins.dedup();
    let gathered: Vec<(usize, Vec<f32>)> = bins
        .iter()
        .map(|&b| {
            let v: Vec<f32> = a
                .par_iter()
                .copied()
                .filter(|&x| (key(x) >> 16) as usize == b)
                .collect();
            (b, v)
        })
        .collect();
    Some(
        place
            .iter()
            .map(|&(b, r)| {
                let mut v = gathered.iter().find(|g| g.0 == b).unwrap().1.clone();
                *v.select_nth_unstable_by(r, |x, y| x.total_cmp(y)).1
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(n: usize) -> Vec<f32> {
        let mut s = 12345u64;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * 4.0 - 1.5
            })
            .collect()
    }

    #[test]
    fn matches_a_full_sort() {
        let a = lcg(300_003);
        let mut sorted = a.clone();
        sorted.sort_by(|x, y| x.total_cmp(y));
        let ks = [0, 1, 7, 150_001, 299_999, 300_002];
        let got = kth_values(&a, &ks).unwrap();
        for (k, g) in ks.iter().zip(got) {
            assert_eq!(g, sorted[*k], "rank {k}");
        }
    }

    #[test]
    fn handles_ties_and_negatives() {
        let mut a = vec![0.0f32; 1000];
        a.extend(std::iter::repeat_n(-2.5, 500));
        a.extend(lcg(500));
        let mut sorted = a.clone();
        sorted.sort_by(|x, y| x.total_cmp(y));
        let ks = [0, 499, 500, 1200, 1999];
        assert_eq!(
            kth_values(&a, &ks).unwrap(),
            ks.iter().map(|&k| sorted[k]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn refuses_non_finite_and_bad_ranks() {
        assert!(kth_values(&[1.0, f32::NAN], &[0]).is_none());
        assert!(kth_values(&[1.0, f32::INFINITY], &[0]).is_none());
        assert!(kth_values(&[1.0, 2.0], &[2]).is_none());
    }
}
