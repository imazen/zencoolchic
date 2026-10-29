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

/// Copy of a plane with `p` replicated (or zero) border pixels on every side.
fn pad_plane(src: &[f32], h: usize, w: usize, p: usize, replicate: bool) -> Vec<f32> {
    let pw = w + 2 * p;
    let mut out = vec![0.0f32; (h + 2 * p) * pw];
    for py in 0..h + 2 * p {
        let sy = if replicate {
            py.saturating_sub(p).min(h - 1)
        } else if py < p || py >= h + p {
            continue;
        } else {
            py - p
        };
        let srow = &src[sy * w..(sy + 1) * w];
        let drow = &mut out[py * pw..(py + 1) * pw];
        drow[p..p + w].copy_from_slice(srow);
        if replicate {
            drow[..p].fill(srow[0]);
            drow[p + w..].fill(srow[w - 1]);
        }
    }
    out
}

/// `UpsamplingSeparableSymmetricConvTranspose2d.forward` (training path) on one plane:
/// replicate-pad by `k/2`, stride-2 transposed convolution with the 2-D kernel, crop.
/// Output is `2h × 2w`.
///
/// Computed per output phase: output `f = 2m + parity` (in uncropped coordinates) gathers
/// `xp[m - a][n - b] · K[py + 2a][px + 2b]`, a contiguous correlation per row. Taps are
/// accumulated with input rows and columns ascending.
pub(crate) fn conv_transpose_up2(src: &[f32], h: usize, w: usize, kernel: &[f32]) -> Vec<f32> {
    let k = kernel.len();
    let k2 = kron(kernel);
    let half = k / 2;
    let p0 = k / 2;
    let crop = 2 * p0 - 1 + k / 2;
    let pw = w + 2 * p0;
    let xp = pad_plane(src, h, w, p0, true);
    let (oh, ow) = (2 * h, 2 * w);
    let m0 = crop >> 1;
    let nm = ((ow - 1 + crop) >> 1) - m0 + 1;
    let mut out = vec![0.0f32; oh * ow];
    let mut tmp = [vec![0.0f32; nm], vec![0.0f32; nm]];
    for oy in 0..oh {
        let f = oy + crop;
        let (my, py) = (f >> 1, f & 1);
        for (px, t) in tmp.iter_mut().enumerate() {
            t.fill(0.0);
            for a in (0..half).rev() {
                let row = &xp[(my - a) * pw..(my - a + 1) * pw];
                let ky = py + 2 * a;
                for b in (0..half).rev() {
                    let kx = px + 2 * b;
                    crate::simd::axpy(t, k2[ky * k + kx], &row[m0 - b..m0 - b + nm]);
                }
            }
        }
        let orow = &mut out[oy * ow..(oy + 1) * ow];
        for (ox, o) in orow.iter_mut().enumerate() {
            let f = ox + crop;
            *o = tmp[f & 1][(f >> 1) - m0];
        }
    }
    out
}

/// `UpsamplingSeparableSymmetricConv2d.forward` (training path) on one plane: zero-padded
/// `k × k` convolution with `kron(w, wᵀ)`, plus the input.
pub(crate) fn sym_conv_residual(src: &[f32], h: usize, w: usize, kernel: &[f32]) -> Vec<f32> {
    let k = kernel.len();
    let k2 = kron(kernel);
    let p = k / 2;
    let pw = w + 2 * p;
    let xp = pad_plane(src, h, w, p, false);
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        let orow = &mut out[y * w..(y + 1) * w];
        for ky in 0..k {
            let row = &xp[(y + ky) * pw..(y + ky + 1) * pw];
            for kx in 0..k {
                crate::simd::axpy(orow, k2[ky * k + kx], &row[kx..kx + w]);
            }
        }
        for (o, s) in orow.iter_mut().zip(&src[y * w..(y + 1) * w]) {
            *o += *s;
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
        // Upstream `SynthesisConv2d` pads by `(k-1)//2`, so an even kernel shrinks
        // the output by one pixel per spatial dimension.
        let p = (k - 1) / 2;
        let (oh, ow) = (h + 2 * p + 1 - k, w + 2 * p + 1 - k);
        let pw = w + 2 * p;
        // Replicate-padded inputs, once per layer (1×1 convs read the planes directly).
        let padded: Vec<Vec<f32>> = if k == 1 {
            Vec::new()
        } else {
            (0..self.c_in)
                .map(|i| pad_plane(x.plane(i), h, w, p, true))
                .collect()
        };
        let mut out = Planes::zeros(self.c_out, oh, ow);
        for o in 0..self.c_out {
            let dst = &mut out.data[o * oh * ow..(o + 1) * oh * ow];
            dst.fill(self.bias[o]);
            for i in 0..self.c_in {
                let wk = &self.weight[(o * self.c_in + i) * k * k..(o * self.c_in + i + 1) * k * k];
                if k == 1 {
                    crate::simd::axpy(dst, wk[0], x.plane(i));
                    continue;
                }
                let src = &padded[i];
                for y in 0..oh {
                    let drow = &mut dst[y * ow..(y + 1) * ow];
                    for ky in 0..k {
                        let row = &src[(y + ky) * pw..(y + ky + 1) * pw];
                        for kx in 0..k {
                            crate::simd::axpy(drow, wk[ky * k + kx], &row[kx..kx + ow]);
                        }
                    }
                }
            }
            if self.residual {
                debug_assert_eq!((oh, ow), (h, w));
                for (d, s) in dst.iter_mut().zip(x.plane(o)) {
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
            // The 1×1 stabiliser keeps input size: it can only be added when no
            // layer shrank the stack (even kernels are refused with a stabiliser).
            debug_assert_eq!(s.data.len(), cur.data.len());
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

/// Transpose of [`bicubic`]: `F.interpolate` backward scatters each output
/// gradient to the (edge-clamped) taps of its gather, weighted the same.
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
pub(crate) fn bicubic_backward(
    dout: &[f32],
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    s: Scale,
) -> Vec<f32> {
    if oh == h && ow == w {
        return dout.to_vec();
    }
    let sy = src_scale(s, h, oh);
    let sx = src_scale(s, w, ow);
    let mut g = vec![0.0f32; h * w];
    for oy in 0..oh {
        let ry = sy * (oy as f32 + 0.5) - 0.5;
        let iy = libm::floorf(ry) as isize;
        let cy = cubic_coeffs(ry - iy as f32);
        for ox in 0..ow {
            let rx = sx * (ox as f32 + 0.5) - 0.5;
            let ix = libm::floorf(rx) as isize;
            let cx = cubic_coeffs(rx - ix as f32);
            let d = dout[oy * ow + ox];
            for (j, &wy) in cy.iter().enumerate() {
                let yy = (iy - 1 + j as isize).clamp(0, h as isize - 1) as usize;
                let row = &mut g[yy * w..yy * w + w];
                for (k, &wx) in cx.iter().enumerate() {
                    let xx = (ix - 1 + k as isize).clamp(0, w as isize - 1) as usize;
                    row[xx] += d * wy * wx;
                }
            }
        }
    }
    g
}

/// Transpose of [`bilinear`].
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
pub(crate) fn bilinear_backward(
    dout: &[f32],
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    s: Scale,
) -> Vec<f32> {
    let sy = src_scale(s, h, oh);
    let sx = src_scale(s, w, ow);
    let idx = |r: f32, n: usize| -> (usize, usize, f32) {
        let r = r.max(0.0);
        let i0 = (r as usize).min(n - 1);
        let i1 = if i0 < n - 1 { i0 + 1 } else { i0 };
        (i0, i1, r - i0 as f32)
    };
    let mut g = vec![0.0f32; h * w];
    for oy in 0..oh {
        let (y0, y1, ly) = idx(sy * (oy as f32 + 0.5) - 0.5, h);
        for ox in 0..ow {
            let (x0, x1, lx) = idx(sx * (ox as f32 + 0.5) - 0.5, w);
            let d = dout[oy * ow + ox];
            g[y0 * w + x0] += d * (1.0 - lx) * (1.0 - ly);
            g[y0 * w + x1] += d * lx * (1.0 - ly);
            g[y1 * w + x0] += d * (1.0 - lx) * ly;
            g[y1 * w + x1] += d * lx * ly;
        }
    }
    g
}

/// Transpose of [`nearest`]: each output cell sends its gradient to one input.
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
pub(crate) fn nearest_backward(
    dout: &[f32],
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    s: Scale,
) -> Vec<f32> {
    let sy = src_scale(s, h, oh);
    let sx = src_scale(s, w, ow);
    let mut g = vec![0.0f32; h * w];
    for oy in 0..oh {
        let y = (libm::floorf(oy as f32 * sy) as usize).min(h - 1);
        for ox in 0..ow {
            let x = (libm::floorf(ox as f32 * sx) as usize).min(w - 1);
            g[y * w + x] += dout[oy * ow + ox];
        }
    }
    g
}

/// Transpose of [`resize`] on a flat `[c][h][w]` buffer: `dout` holds `c`
/// planes of `out` size; the result is `c` planes of `inp` size.
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
pub(crate) fn resize_backward(
    dout: &[f32],
    c: usize,
    inp: (usize, usize),
    out: (usize, usize),
    mode: FinalUpsampling,
    s: Scale,
) -> Vec<f32> {
    let (h, w) = inp;
    let (oh, ow) = out;
    let mut g = vec![0.0f32; c * h * w];
    for ch in 0..c {
        let d = &dout[ch * oh * ow..(ch + 1) * oh * ow];
        let r = match mode {
            FinalUpsampling::Nearest => nearest_backward(d, h, w, oh, ow, s),
            FinalUpsampling::Bilinear => bilinear_backward(d, h, w, oh, ow, s),
            FinalUpsampling::Bicubic => bicubic_backward(d, h, w, oh, ow, s),
        };
        g[ch * h * w..(ch + 1) * h * w].copy_from_slice(&r);
    }
    g
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn conv_even_kernel_shrinks_by_one() {
        // Upstream `SynthesisConv2d` uses pad=(k-1)//2: for k=2 that is no
        // padding at all, i.e. a valid 2×2 convolution over the 3×3 input.
        let c = Conv {
            c_in: 1,
            c_out: 1,
            k: 2,
            weight: vec![1.0; 4],
            bias: vec![0.0],
            residual: false,
        };
        let x = Planes {
            c: 1,
            h: 3,
            w: 3,
            data: (1..=9).map(|v| v as f32).collect(),
        };
        let out = c.forward(&x);
        assert_eq!((out.h, out.w), (2, 2));
        assert_eq!(out.data, vec![12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn conv_even_kernel_replicate_pad() {
        // k=4 pads one pixel and still shrinks by one: out = h+2·1-4+1 = h-1.
        let c = Conv {
            c_in: 1,
            c_out: 1,
            k: 4,
            weight: vec![1.0; 16],
            bias: vec![0.0],
            residual: false,
        };
        let x = Planes {
            c: 1,
            h: 3,
            w: 3,
            data: (1..=9).map(|v| v as f32).collect(),
        };
        let out = c.forward(&x);
        assert_eq!((out.h, out.w), (2, 2));
        // Hand-computed sums of the 4×4 windows over the replicate-padded input.
        assert_eq!(out.data, vec![64.0, 72.0, 88.0, 96.0]);
    }
}
