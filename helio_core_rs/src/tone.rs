//! The tone composite (helio/application/render.py, `_tone_rows`): layer,
//! micro-detail, brightness, contrast and gamma, shading to 8-bit RGB.
//!
//! The picture must equal the classic composite bit for bit, so the
//! arithmetic is the classic's, in float64 and in its order: every operation
//! is a separate IEEE operation (Rust does not fuse multiply-add), and the one
//! place the classic code works in float32 (a float32 shading times a Python
//! float stays float32) is kept.  One pass over the pixels replaces the
//! several full-size float64 temporaries NumPy makes.

use rayon::prelude::*;

/// Pixels per parallel block.
const BLOCK: usize = 8192;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Layer {
    /// The albedo relit by the shading.
    Combined,
    /// The shading alone, grey.
    Shading,
    /// The shading mixed with the roughness map.
    Roughness,
}

pub struct Tone<'a> {
    pub layer: Layer,
    /// Colour opacity (Combined) and the colour layer's own brightness.
    pub opacity: f64,
    pub colour_gain: f64,
    /// Roughness opacity (Roughness).
    pub rough_opacity: f64,
    pub bright: f64,
    pub k: f64,
    /// Micro-detail strength, applied when `detail` is given.
    pub micro: f64,
    pub detail: Option<&'a [f64]>,
    pub lut: Option<&'a [u8]>,
    /// Skip brightness and contrast (they would change no bit).
    pub skip_tone: bool,
}

/// A shading pixel, in float32 or float64.
pub trait Shade: Copy + Sync {
    fn to_f64(self) -> f64;
    /// `self * c` in the shading's own precision, as NumPy does for an array
    /// times a Python float.
    fn mul_weak(self, c: f64) -> f64;
}

impl Shade for f32 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn mul_weak(self, c: f64) -> f64 {
        (self * c as f32) as f64
    }
}

impl Shade for f64 {
    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn mul_weak(self, c: f64) -> f64 {
        self * c
    }
}

#[inline(always)]
fn finish(mut o: f64, t: &Tone) -> u8 {
    if !t.skip_tone {
        o *= t.bright;
        o -= 128.0;
        o *= t.k;
        o += 128.0;
    }
    let v = o.clamp(0.0, 255.0) as u8; // NaN -> 0, as NumPy's cast gives
    match t.lut {
        Some(l) => l[v as usize],
        None => v,
    }
}

#[inline(always)]
fn detail_add(o: f64, i: usize, t: &Tone) -> f64 {
    match t.detail {
        Some(d) => (o + d[i] * t.micro * 2.0).clamp(0.0, 255.0),
        None => o,
    }
}

/// `albedo` is `(n, 3)` bytes (Combined only), `rough` `n` bytes (Roughness
/// only); `out` is `(n, 3)` bytes.
pub fn tone<S: Shade>(shading: &[S], albedo: &[u8], rough: &[u8], t: &Tone, out: &mut [u8]) {
    out.par_chunks_mut(3 * BLOCK)
        .zip(shading.par_chunks(BLOCK))
        .enumerate()
        .for_each(|(b, (o, sh))| {
            let s0 = b * BLOCK;
            match t.layer {
                Layer::Combined => combined(sh, &albedo[3 * s0..3 * (s0 + sh.len())], s0, t, o),
                Layer::Shading => {
                    for (j, (px, &s)) in o.chunks_exact_mut(3).zip(sh).enumerate() {
                        px.fill(finish(detail_add(s.to_f64() * 255.0, s0 + j, t), t));
                    }
                }
                Layer::Roughness => {
                    let rg = &rough[s0..s0 + sh.len()];
                    for (j, ((px, &s), &r)) in o.chunks_exact_mut(3).zip(sh).zip(rg).enumerate() {
                        let r = r as f64 / 255.0;
                        let v = 255.0 * (s.mul_weak(1.0 - t.rough_opacity) + r * t.rough_opacity);
                        px.fill(finish(detail_add(v, s0 + j, t), t));
                    }
                }
            }
        });
}

/// Combined Colour for one block.  The common case (full opacity, no colour
/// gain, micro-detail, tone or gamma) is `trunc(clip(albedo * shading))`.
fn combined<S: Shade>(sh: &[S], alb: &[u8], s0: usize, t: &Tone, o: &mut [u8]) {
    let general = |a: u8, sf: f64| -> f64 {
        let a = a as f64;
        let mut v = if t.opacity == 1.0 && sf.is_finite() {
            a * sf
        } else {
            255.0 * sf * (1.0 - t.opacity) + a * sf * t.opacity
        };
        if t.colour_gain != 1.0 {
            v *= t.colour_gain;
        }
        v
    };
    if t.opacity == 1.0
        && t.colour_gain == 1.0
        && t.detail.is_none()
        && t.skip_tone
        && t.lut.is_none()
    {
        for ((px, a), &s) in o.chunks_exact_mut(3).zip(alb.chunks_exact(3)).zip(sh) {
            let sf = s.to_f64();
            if sf.is_finite() {
                for (p, &a) in px.iter_mut().zip(a) {
                    *p = (a as f64 * sf).clamp(0.0, 255.0) as u8;
                }
            } else {
                for (p, &a) in px.iter_mut().zip(a) {
                    *p = general(a, sf).clamp(0.0, 255.0) as u8;
                }
            }
        }
        return;
    }
    for (j, ((px, a), &s)) in o
        .chunks_exact_mut(3)
        .zip(alb.chunks_exact(3))
        .zip(sh)
        .enumerate()
    {
        let sf = s.to_f64();
        for (p, &a) in px.iter_mut().zip(a) {
            *p = finish(detail_add(general(a, sf), s0 + j, t), t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base(layer: Layer) -> Tone<'static> {
        Tone {
            layer,
            opacity: 1.0,
            colour_gain: 1.0,
            rough_opacity: 0.35,
            bright: 1.0,
            k: 1.0,
            micro: 0.0,
            detail: None,
            lut: None,
            skip_tone: true,
        }
    }

    #[test]
    fn shading_layer_is_grey_and_truncates() {
        let s = [0.5f32, 1.2, -0.3, f32::NAN];
        let mut out = [9u8; 12];
        tone(&s, &[], &[], &base(Layer::Shading), &mut out);
        assert_eq!(out, [127, 127, 127, 255, 255, 255, 0, 0, 0, 0, 0, 0]); // 127.5 -> 127
    }

    #[test]
    fn combined_multiplies_albedo_by_shading() {
        let s = [0.5f32];
        let a = [200u8, 100, 0];
        let mut out = [0u8; 3];
        tone(&s, &a, &[], &base(Layer::Combined), &mut out);
        assert_eq!(out, [100, 50, 0]);
        let mut half = base(Layer::Combined);
        half.opacity = 0.5; // 255*0.5*0.5 + a*0.5*0.5
        tone(&s, &a, &[], &half, &mut out);
        assert_eq!(out, [(63.75 + 50.0) as u8, (63.75 + 25.0) as u8, 63]);
    }

    #[test]
    fn tone_lut_and_f64_shading() {
        let lut: Vec<u8> = (0..=255u8).rev().collect();
        let mut t = base(Layer::Shading);
        t.lut = Some(&lut);
        let mut out = [0u8; 3];
        tone(&[1.0f64], &[], &[], &t, &mut out);
        assert_eq!(out, [0, 0, 0]);
        let mut u = base(Layer::Shading);
        u.skip_tone = false;
        u.bright = 2.0; // (127.5*2 - 128) * 1 + 128 = 255
        tone(&[0.5f64], &[], &[], &u, &mut out);
        assert_eq!(out, [255, 255, 255]);
    }

    #[test]
    fn roughness_mixes_in_the_map() {
        let mut out = [0u8; 3];
        let mut t = base(Layer::Roughness);
        t.rough_opacity = 0.5;
        tone(&[1.0f32], &[], &[0u8], &t, &mut out);
        assert_eq!(out, [127, 127, 127]); // 255 * (0.5 + 0)
    }
}
