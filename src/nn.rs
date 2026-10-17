//! Float networks: upsampling, synthesis and PyTorch-compatible resampling
//! (upstream `coolchic/component/core/{upsampling,synthesis}.py`).
//!
//! The decoder runs these modules in upstream's *training* code path (freshly built
//! `nn.Module`s are in training mode), i.e. the 2-D `kron(w, wᵀ)` kernels rather than the
//! separable ones. Summation order differs from PyTorch's, so outputs match to rounding,
//! not bit for bit (`PORTING.md` records the measured difference).

#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

use crate::header::{FinalUpsampling, SynLayer};

/// Planar f32 tensor `[c][h][w]`.
#[derive(Debug, Clone)]
pub struct Planes {
    pub c: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Planes {
    pub fn zeros(c: usize, h: usize, w: usize) -> Self {
        Self {
            c,
            h,
            w,
            data: vec![0.0; c * h * w],
        }
    }

    #[inline]
    pub fn plane(&self, c: usize) -> &[f32] {
        &self.data[c * self.h * self.w..(c + 1) * self.h * self.w]
    }

    #[inline]
    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let n = self.h * self.w;
        &mut self.data[c * n..(c + 1) * n]
    }
}

#[inline]
fn fmadd(a: f32, b: f32, c: f32) -> f32 {
    libm::fmaf(a, b, c)
}

/// `_Parameterization_Symmetric_1d`: full symmetric kernel from its first half.
pub(crate) fn symmetric_kernel(half: &[f32], k: usize) -> Vec<f32> {
    let mut out = half.to_vec();
    let rev: Vec<f32> = half.iter().rev().copied().collect();
    out.extend_from_slice(&rev[k % 2..]);
    debug_assert_eq!(out.len(), k);
    out
}

/// `kron(w, wᵀ)`: `k2[r][c] = w[c] · w[r]`.
fn kron(w: &[f32]) -> Vec<f32> {
    let k = w.len();
    let mut out = vec![0.0; k * k];
    for r in 0..k {
        for c in 0..k {
            out[r * k + c] = w[c] * w[r];
        }
    }
    out
}

/// `UpsamplingSeparableSymmetricConvTranspose2d.forward` (training path) on one plane:
/// replicate-pad by `k/2`, stride-2 transposed convolution with the 2-D kernel, crop.
/// Output is `2h × 2w`.
pub(crate) fn conv_transpose_up2(src: &[f32], h: usize, w: usize, kernel: &[f32]) -> Vec<f32> {
    let k = kernel.len();
    let k2 = kron(kernel);
    let p0 = k / 2;
    let crop = 2 * p0 - 1 + k / 2;
    let (ph, pw) = (h + 2 * p0, w + 2 * p0);
    let at = |y: usize, x: usize| -> f32 {
        let sy = y.saturating_sub(p0).min(h - 1);
        let sx = x.saturating_sub(p0).min(w - 1);
        src[sy * w + sx]
    };
    let (oh, ow) = (2 * h, 2 * w);
    let mut out = vec![0.0f32; oh * ow];
    // Gather form: full-output pixel (fy, fx) = Σ xp[iy][ix] · K[fy - 2iy][fx - 2ix].
    for oy in 0..oh {
        let fy = oy + crop;
        for ox in 0..ow {
            let fx = ox + crop;
            let mut acc = 0.0f32;
            // iy ranges over input rows with 0 <= fy - 2iy < k.
            let iy_hi = (fy / 2).min(ph - 1);
            let iy_lo = (fy + 1).saturating_sub(k).div_ceil(2);
            let ix_hi = (fx / 2).min(pw - 1);
            let ix_lo = (fx + 1).saturating_sub(k).div_ceil(2);
            for iy in iy_lo..=iy_hi {
                let ky = fy - 2 * iy;
                if ky >= k {
                    continue;
                }
                for ix in ix_lo..=ix_hi {
                    let kx = fx - 2 * ix;
                    if kx >= k {
                        continue;
                    }
                    acc = fmadd(at(iy, ix), k2[ky * k + kx], acc);
                }
            }
            out[oy * ow + ox] = acc;
        }
    }
    out
}

/// `UpsamplingSeparableSymmetricConv2d.forward` (training path) on one plane: zero-padded
/// `k × k` convolution with `kron(w, wᵀ)`, plus the input.
pub(crate) fn sym_conv_residual(src: &[f32], h: usize, w: usize, kernel: &[f32]) -> Vec<f32> {
    let k = kernel.len();
    let k2 = kron(kernel);
    let p = (k / 2) as isize;
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for ky in 0..k {
                let sy = y as isize + ky as isize - p;
                if sy < 0 || sy >= h as isize {
                    continue;
                }
                for kx in 0..k {
                    let sx = x as isize + kx as isize - p;
                    if sx < 0 || sx >= w as isize {
                        continue;
                    }
                    acc = fmadd(src[sy as usize * w + sx as usize], k2[ky * k + kx], acc);
                }
            }
            out[y * w + x] = acc + src[y * w + x];
        }
    }
    out
}

/// Upsampling module: the decoder-side latents (finest first, one channel each) are
/// merged coarsest to finest; each step upsamples the stack ×2 (per channel), crops it to
/// the next grid and prepends that grid, pre-filtered.
pub(crate) struct Upsampling {
    pub(crate) conv_t: Vec<Vec<f32>>, // full kernels
    pub(crate) conv: Vec<Vec<f32>>,
}

impl Upsampling {
    pub(crate) fn forward(&self, latents: &[Planes]) -> Planes {
        let n = latents.len();
        let mut up = latents[n - 1].clone();
        for (idx, target) in latents[..n - 1].iter().rev().enumerate() {
            if target.c == 0 {
                break;
            }
            let kt = &self.conv_t[idx % self.conv_t.len()];
            let kc = &self.conv[idx % self.conv.len()];
            let (th, tw) = (target.h, target.w);
            let mut next = Planes::zeros(target.c + up.c, th, tw);
            for c in 0..target.c {
                let hb = sym_conv_residual(target.plane(c), th, tw, kc);
                next.plane_mut(c).copy_from_slice(&hb);
            }
            for c in 0..up.c {
                let x = conv_transpose_up2(up.plane(c), up.h, up.w, kt);
                let dst = next.plane_mut(target.c + c);
                for y in 0..th {
                    dst[y * tw..(y + 1) * tw].copy_from_slice(&x[y * 2 * up.w..y * 2 * up.w + tw]);
                }
            }
            up = next;
        }
        up
    }
}

/// `SynthesisConv2d`: replicate-padded `k × k` convolution with bias, optional residual.
pub(crate) struct Conv {
    pub(crate) c_in: usize,
    pub(crate) c_out: usize,
    pub(crate) k: usize,
    /// `[out][in][ky][kx]`
    pub(crate) weight: Vec<f32>,
    pub(crate) bias: Vec<f32>,
    pub(crate) residual: bool,
}

impl Conv {
    pub(crate) fn forward(&self, x: &Planes) -> Planes {
        debug_assert_eq!(x.c, self.c_in);
        let (h, w, k) = (x.h, x.w, self.k);
        let p = (k / 2) as isize;
        let mut out = Planes::zeros(self.c_out, h, w);
        for o in 0..self.c_out {
            let dst = &mut out.data[o * h * w..(o + 1) * h * w];
            dst.fill(self.bias[o]);
            for i in 0..self.c_in {
                let src = x.plane(i);
                for ky in 0..k {
                    for kx in 0..k {
                        let wv = self.weight[((o * self.c_in + i) * k + ky) * k + kx];
                        for y in 0..h {
                            let sy =
                                (y as isize + ky as isize - p).clamp(0, h as isize - 1) as usize;
                            let srow = &src[sy * w..(sy + 1) * w];
                            let drow = &mut dst[y * w..(y + 1) * w];
                            if k == 1 {
                                for xx in 0..w {
                                    drow[xx] = fmadd(srow[xx], wv, drow[xx]);
                                }
                            } else {
                                for xx in 0..w {
                                    let sx = (xx as isize + kx as isize - p)
                                        .clamp(0, w as isize - 1)
                                        as usize;
                                    drow[xx] = fmadd(srow[sx], wv, drow[xx]);
                                }
                            }
                        }
                    }
                }
            }
            if self.residual {
                let src = x.plane(o);
                for (d, s) in dst.iter_mut().zip(src) {
                    *d += *s;
                }
            }
        }
        out
    }
}

pub(crate) struct Synthesis {
    pub(crate) layers: Vec<(Conv, bool)>,         // (conv, relu)
    pub(crate) stabiliser: Option<(Conv, usize)>, // (conv, n_input_ft)
    pub(crate) output_transform: Conv,
}

impl Synthesis {
    pub(crate) fn forward(&self, x: &Planes) -> Planes {
        let mut cur = x.clone();
        for (conv, relu) in &self.layers {
            cur = conv.forward(&cur);
            if *relu {
                for v in &mut cur.data {
                    *v = v.max(0.0);
                }
            }
        }
        if let Some((st, n_in)) = &self.stabiliser {
            let sub = Planes {
                c: *n_in,
                h: x.h,
                w: x.w,
                data: x.data[..n_in * x.h * x.w].to_vec(),
            };
            let s = st.forward(&sub);
            for (d, v) in cur.data.iter_mut().zip(&s.data) {
                *d += *v;
            }
        }
        self.output_transform.forward(&cur)
    }
}

pub(crate) fn layer_dims(input_ft: usize, layers: &[SynLayer]) -> Vec<(usize, usize, usize)> {
    let mut c_in = input_ft;
    layers
        .iter()
        .map(|l| {
            let d = (c_in, l.out_ft as usize, l.k_size as usize);
            c_in = l.out_ft as usize;
            d
        })
        .collect()
}

/// Source index scale used by `F.interpolate` without `align_corners`.
#[derive(Clone, Copy)]
pub(crate) enum Scale {
    /// `scale_factor=` given: source step is `1 / factor`.
    Factor(f64),
    /// `size=` given: source step is `in / out`.
    Size,
}

fn src_scale(s: Scale, n_in: usize, n_out: usize) -> f32 {
    match s {
        Scale::Factor(f) => (1.0 / f) as f32,
        Scale::Size => n_in as f32 / n_out as f32,
    }
}

fn cubic1(x: f32, a: f32) -> f32 {
    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}

fn cubic2(x: f32, a: f32) -> f32 {
    ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
}

fn cubic_coeffs(t: f32) -> [f32; 4] {
    let a = -0.75f32;
    let t2 = 1.0 - t;
    [
        cubic2(t + 1.0, a),
        cubic1(t, a),
        cubic1(t2, a),
        cubic2(t2 + 1.0, a),
    ]
}

/// `F.interpolate(mode="bicubic", align_corners=False)` on one plane.
pub(crate) fn bicubic(src: &[f32], h: usize, w: usize, oh: usize, ow: usize, s: Scale) -> Vec<f32> {
    if oh == h && ow == w {
        return src.to_vec();
    }
    let sy = src_scale(s, h, oh);
    let sx = src_scale(s, w, ow);
    let get = |y: isize, x: isize| {
        src[(y.clamp(0, h as isize - 1) as usize) * w + x.clamp(0, w as isize - 1) as usize]
    };
    let mut out = vec![0.0f32; oh * ow];
    for oy in 0..oh {
        let ry = sy * (oy as f32 + 0.5) - 0.5;
        let iy = libm::floorf(ry);
        let cy = cubic_coeffs(ry - iy);
        let iy = iy as isize;
        for ox in 0..ow {
            let rx = sx * (ox as f32 + 0.5) - 0.5;
            let ix = libm::floorf(rx);
            let cx = cubic_coeffs(rx - ix);
            let ix = ix as isize;
            let mut acc = 0.0f32;
            for (j, &wy) in cy.iter().enumerate() {
                let yy = iy - 1 + j as isize;
                let row = get(yy, ix - 1) * cx[0]
                    + get(yy, ix) * cx[1]
                    + get(yy, ix + 1) * cx[2]
                    + get(yy, ix + 2) * cx[3];
                acc += row * wy;
            }
            out[oy * ow + ox] = acc;
        }
    }
    out
}

/// `F.interpolate(mode="bilinear", align_corners=False)` on one plane.
pub(crate) fn bilinear(
    src: &[f32],
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    s: Scale,
) -> Vec<f32> {
    if oh == h && ow == w {
        return src.to_vec();
    }
    let sy = src_scale(s, h, oh);
    let sx = src_scale(s, w, ow);
    let idx = |r: f32, n: usize| -> (usize, usize, f32) {
        let r = r.max(0.0);
        let i0 = (r as usize).min(n - 1);
        let i1 = if i0 < n - 1 { i0 + 1 } else { i0 };
        (i0, i1, r - i0 as f32)
    };
    let mut out = vec![0.0f32; oh * ow];
    for oy in 0..oh {
        let (y0, y1, ly) = idx(sy * (oy as f32 + 0.5) - 0.5, h);
        for ox in 0..ow {
            let (x0, x1, lx) = idx(sx * (ox as f32 + 0.5) - 0.5, w);
            let top = src[y0 * w + x0] * (1.0 - lx) + src[y0 * w + x1] * lx;
            let bot = src[y1 * w + x0] * (1.0 - lx) + src[y1 * w + x1] * lx;
            out[oy * ow + ox] = top * (1.0 - ly) + bot * ly;
        }
    }
    out
}

/// `F.interpolate(mode="nearest")` on one plane.
pub(crate) fn nearest(src: &[f32], h: usize, w: usize, oh: usize, ow: usize, s: Scale) -> Vec<f32> {
    let sy = src_scale(s, h, oh);
    let sx = src_scale(s, w, ow);
    let mut out = vec![0.0f32; oh * ow];
    for oy in 0..oh {
        let y = (libm::floorf(oy as f32 * sy) as usize).min(h - 1);
        for ox in 0..ow {
            let x = (libm::floorf(ox as f32 * sx) as usize).min(w - 1);
            out[oy * ow + ox] = src[y * w + x];
        }
    }
    out
}

pub(crate) fn resize(p: &Planes, oh: usize, ow: usize, mode: FinalUpsampling, s: Scale) -> Planes {
    let mut out = Planes::zeros(p.c, oh, ow);
    for c in 0..p.c {
        let r = match mode {
            FinalUpsampling::Nearest => nearest(p.plane(c), p.h, p.w, oh, ow, s),
            FinalUpsampling::Bilinear => bilinear(p.plane(c), p.h, p.w, oh, ow, s),
            FinalUpsampling::Bicubic => bicubic(p.plane(c), p.h, p.w, oh, ow, s),
        };
        out.plane_mut(c).copy_from_slice(&r);
    }
    out
}

/// `fixed_upsampling(latents, mode)` for float planes (common randomness): stacks the
/// grids (finest first) at the finest resolution, resizing ×2 + crop at each step.
pub(crate) fn fixed_upsampling(latents: &[Planes], mode: FinalUpsampling) -> Planes {
    let n = latents.len();
    let mut up = latents[n - 1].clone();
    for target in latents[..n - 1].iter().rev() {
        let x = if up.h != target.h || up.w != target.w {
            let big = resize(&up, 2 * up.h, 2 * up.w, mode, Scale::Factor(2.0));
            let mut cropped = Planes::zeros(up.c, target.h, target.w);
            for c in 0..up.c {
                let src = big.plane(c);
                let dst = cropped.plane_mut(c);
                for y in 0..target.h {
                    dst[y * target.w..(y + 1) * target.w]
                        .copy_from_slice(&src[y * big.w..y * big.w + target.w]);
                }
            }
            cropped
        } else {
            up
        };
        let mut next = Planes::zeros(target.c + x.c, target.h, target.w);
        let n0 = target.data.len();
        next.data[..n0].copy_from_slice(&target.data);
        next.data[n0..].copy_from_slice(&x.data);
        up = next;
    }
    up
}

/// `CommonGaussianNoiseGenerator`: Park–Miller LCG + Box–Muller, f64, fixed seed.
pub(crate) struct CommonNoise {
    seed: u64,
}

impl CommonNoise {
    pub(crate) fn new() -> Self {
        Self { seed: 18101995 }
    }

    fn grand(&mut self) -> f64 {
        const A: u64 = 16807;
        const M: u64 = (1 << 31) - 1;
        self.seed = (A * self.seed) % M;
        let u1 = self.seed as f64 / M as f64;
        self.seed = (A * self.seed) % M;
        let u2 = self.seed as f64 / M as f64;
        // Upstream's truncated pi, not `core::f64::consts::PI`: the noise must match it.
        #[allow(clippy::approx_constant)]
        const PI: f64 = 3.14159265359;
        libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * PI * u2)
    }

    pub(crate) fn sample(&mut self, h: usize, w: usize) -> Planes {
        let mut p = Planes::zeros(1, h, w);
        for v in &mut p.data {
            *v = self.grand() as f32;
        }
        p
    }
}
