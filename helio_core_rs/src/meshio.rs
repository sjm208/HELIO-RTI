//! Mesh file writers: Wavefront OBJ (text) and binary STL.
//!
//! The Python writers format every number with an f-string, one at a time:
//! about 110 s for the 4-million-vertex mesh of a 12-megapixel capture, twice
//! (once through trimesh, then again to add the UVs).  Here the lines are
//! formatted in blocks on every core and written in order, the same bytes.
//!
//! Numbers are formatted by Rust's `{:.N}`, which, like C's `%.Nf` and Python's
//! `:.Nf`, rounds the exact binary value correctly, so the text is identical.

use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufWriter, Write};

/// Lines per work block, and blocks per wave (a wave is formatted in parallel,
/// then written, so memory stays a few hundred MB for any mesh).
const BLOCK: usize = 1 << 15;
const WAVE: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FaceStyle {
    /// `f a b c`
    V,
    /// `f a//a b//b c//c`
    VN,
    /// `f a/a b/b c/c`
    VT,
    /// `f a/a/a b/b/b c/c/c`
    VTN,
}

pub struct ObjOptions<'a> {
    pub header: &'a str,
    pub decimals: usize,
    pub normal_decimals: usize,
    pub uv_decimals: usize,
    pub color_decimals: usize,
    pub face_style: FaceStyle,
    /// An empty line between the vertices and what follows (`np.savetxt` style).
    pub blank_after_vertices: bool,
    /// Text written between the vertex data and the faces.
    pub mid: &'a str,
    /// Text written after the faces.
    pub footer: &'a str,
}

/// Write blocks of lines: `lines` is the count, `fmt(i, buf)` appends line `i`.
fn write_lines<W: Write, F>(out: &mut W, lines: usize, fmt: F) -> io::Result<()>
where
    F: Fn(usize, &mut Vec<u8>) + Sync,
{
    let blocks = lines.div_ceil(BLOCK);
    let mut b = 0;
    while b < blocks {
        let end = (b + WAVE).min(blocks);
        let bufs: Vec<Vec<u8>> = (b..end)
            .into_par_iter()
            .map(|k| {
                let (lo, hi) = (k * BLOCK, ((k + 1) * BLOCK).min(lines));
                let mut buf = Vec::with_capacity((hi - lo) * 48);
                for i in lo..hi {
                    fmt(i, &mut buf);
                }
                buf
            })
            .collect();
        for buf in bufs {
            out.write_all(&buf)?;
        }
        b = end;
    }
    Ok(())
}

fn push_f(buf: &mut Vec<u8>, v: f64, decimals: usize) {
    // Vec<u8> implements io::Write; formatting into it cannot fail.
    let _ = write!(buf, "{v:.decimals$}");
}

/// `verts` is `(n, 3)`, `normals` and `uvs` (`(n, 3)` / `(n, 2)`) and `colors`
/// (`(n, 3)` bytes, written as `v x y z r g b` with r = byte / 255) are optional;
/// `faces` is `(m, 3)` zero-based vertex indices.
#[allow(clippy::too_many_arguments)]
pub fn write_obj(
    path: &str,
    verts: &[f64],
    faces: &[u32],
    normals: Option<&[f64]>,
    uvs: Option<&[f64]>,
    colors: Option<&[u8]>,
    o: &ObjOptions,
) -> io::Result<()> {
    let n = verts.len() / 3;
    let m = faces.len() / 3;
    let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
    out.write_all(o.header.as_bytes())?;
    write_lines(&mut out, n, |i, buf| {
        buf.extend_from_slice(b"v ");
        push_f(buf, verts[3 * i], o.decimals);
        buf.push(b' ');
        push_f(buf, verts[3 * i + 1], o.decimals);
        buf.push(b' ');
        push_f(buf, verts[3 * i + 2], o.decimals);
        if let Some(c) = colors {
            for k in 0..3 {
                buf.push(b' ');
                push_f(buf, c[3 * i + k] as f64 / 255.0, o.color_decimals);
            }
        }
        buf.push(b'\n');
    })?;
    if o.blank_after_vertices {
        out.write_all(b"\n")?;
    }
    if let Some(nv) = normals {
        write_lines(&mut out, n, |i, buf| {
            buf.extend_from_slice(b"vn ");
            push_f(buf, nv[3 * i], o.normal_decimals);
            buf.push(b' ');
            push_f(buf, nv[3 * i + 1], o.normal_decimals);
            buf.push(b' ');
            push_f(buf, nv[3 * i + 2], o.normal_decimals);
            buf.push(b'\n');
        })?;
    }
    if let Some(uv) = uvs {
        write_lines(&mut out, n, |i, buf| {
            buf.extend_from_slice(b"vt ");
            push_f(buf, uv[2 * i], o.uv_decimals);
            buf.push(b' ');
            push_f(buf, uv[2 * i + 1], o.uv_decimals);
            buf.push(b'\n');
        })?;
    }
    out.write_all(o.mid.as_bytes())?;
    let style = o.face_style;
    write_lines(&mut out, m, |i, buf| {
        buf.extend_from_slice(b"f");
        let mut num = itoa::Buffer::new();
        for k in 0..3 {
            let idx = faces[3 * i + k] as u64 + 1;
            let s = num.format(idx).as_bytes();
            buf.push(b' ');
            buf.extend_from_slice(s);
            match style {
                FaceStyle::V => {}
                FaceStyle::VN => {
                    buf.extend_from_slice(b"//");
                    buf.extend_from_slice(s);
                }
                FaceStyle::VT => {
                    buf.push(b'/');
                    buf.extend_from_slice(s);
                }
                FaceStyle::VTN => {
                    buf.push(b'/');
                    buf.extend_from_slice(s);
                    buf.push(b'/');
                    buf.extend_from_slice(s);
                }
            }
        }
        buf.push(b'\n');
    })?;
    out.write_all(o.footer.as_bytes())?;
    out.flush()
}

/// Binary STL: an 80-byte header, the triangle count, then per triangle its
/// unit normal (the normalised cross product; `(0, 0, 1)` when degenerate),
/// three float32 vertices and a zero attribute.  Same records as the numpy
/// writer `_write_stl` in export_3d.py.
pub fn write_stl(path: &str, header: &[u8], verts: &[f64], faces: &[u32]) -> io::Result<()> {
    let m = faces.len() / 3;
    let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
    let mut head = [b' '; 80];
    let h = &header[..header.len().min(80)];
    head[..h.len()].copy_from_slice(h);
    out.write_all(&head)?;
    out.write_all(&(m as u32).to_le_bytes())?;
    let blocks = m.div_ceil(BLOCK);
    let mut b = 0;
    while b < blocks {
        let end = (b + WAVE).min(blocks);
        let bufs: Vec<Vec<u8>> = (b..end)
            .into_par_iter()
            .map(|k| {
                let (lo, hi) = (k * BLOCK, ((k + 1) * BLOCK).min(m));
                let mut buf = Vec::with_capacity((hi - lo) * 50);
                for i in lo..hi {
                    let p = |c: usize| -> [f32; 3] {
                        let v = faces[3 * i + c] as usize;
                        [
                            verts[3 * v] as f32,
                            verts[3 * v + 1] as f32,
                            verts[3 * v + 2] as f32,
                        ]
                    };
                    let (a, bb, c) = (p(0), p(1), p(2));
                    let u = [bb[0] - a[0], bb[1] - a[1], bb[2] - a[2]];
                    let w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                    let cr = [
                        u[1] * w[2] - u[2] * w[1],
                        u[2] * w[0] - u[0] * w[2],
                        u[0] * w[1] - u[1] * w[0],
                    ];
                    let ln = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
                    let nrm = if ln > 1e-12 {
                        [cr[0] / ln, cr[1] / ln, cr[2] / ln]
                    } else {
                        [0.0, 0.0, 1.0]
                    };
                    for x in nrm.iter().chain(&a).chain(&bb).chain(&c) {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                    buf.extend_from_slice(&[0, 0]);
                }
                buf
            })
            .collect();
        for buf in bufs {
            out.write_all(&buf)?;
        }
        b = end;
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("helio_meshio_{}_{name}", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    fn opts<'a>(style: FaceStyle) -> ObjOptions<'a> {
        ObjOptions {
            header: "# test\n",
            decimals: 4,
            normal_decimals: 6,
            uv_decimals: 6,
            color_decimals: 8,
            face_style: style,
            blank_after_vertices: false,
            mid: "",
            footer: "",
        }
    }

    #[test]
    fn obj_text_matches_the_f_strings() {
        let p = tmp("a.obj");
        let v = [0.0, -0.00001, 1.23456789, 2.5, 3.0, -4.99995];
        let f = [0u32, 1, 1];
        write_obj(
            &p,
            &v,
            &f,
            Some(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
            Some(&[0.25, 0.75, 1.0, 0.0]),
            None,
            &opts(FaceStyle::VTN),
        )
        .unwrap();
        let s = std::fs::read_to_string(&p).unwrap();
        assert_eq!(
            s,
            "# test\nv 0.0000 -0.0000 1.2346\nv 2.5000 3.0000 -5.0000\nvn 0.000000 0.000000 1.000000\nvn 1.000000 0.000000 0.000000\nvt 0.250000 0.750000\nvt 1.000000 0.000000\nf 1/1/1 2/2/2 2/2/2\n"
        );
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn colours_blank_line_and_mid() {
        let p = tmp("b.obj");
        let mut o = opts(FaceStyle::V);
        o.blank_after_vertices = true;
        o.mid = "usemtl x\n";
        write_obj(
            &p,
            &[0.0, 0.0, 0.0],
            &[0, 0, 0],
            None,
            None,
            Some(&[255, 0, 51]),
            &o,
        )
        .unwrap();
        let s = std::fs::read_to_string(&p).unwrap();
        assert_eq!(
            s,
            "# test\nv 0.0000 0.0000 0.0000 1.00000000 0.00000000 0.20000000\n\nusemtl x\nf 1 1 1\n"
        );
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn stl_records() {
        let p = tmp("c.stl");
        write_stl(
            &p,
            b"hello",
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[0, 1, 2, 0, 0, 0],
        )
        .unwrap();
        let b = std::fs::read(&p).unwrap();
        assert_eq!(b.len(), 80 + 4 + 2 * 50);
        assert_eq!(&b[..5], b"hello");
        assert_eq!(u32::from_le_bytes(b[80..84].try_into().unwrap()), 2);
        let nz = f32::from_le_bytes(b[84 + 8..84 + 12].try_into().unwrap());
        assert_eq!(nz, 1.0); // (1,0,0) x (0,1,0) = +z
        let degenerate_nz = f32::from_le_bytes(b[84 + 50 + 8..84 + 50 + 12].try_into().unwrap());
        assert_eq!(degenerate_nz, 1.0);
        let _ = std::fs::remove_file(p);
    }
}
