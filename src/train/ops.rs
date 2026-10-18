//! Forward and backward of the training graph (see the module docs in `mod.rs`).

// Plane kernels take (data, dims, kernel, outputs) explicitly.
#![allow(clippy::too_many_arguments)]

use alloc::vec;
use alloc::vec::Vec;

use super::arm_batch::{
    ArmBackward, MlpGrad as ArmGrad, arm_pass, to_channel_major, to_pixel_major,
};
use super::{
    ConvLayer, Linear, LossTerms, Mlp, Params, Quantizer, Shape, StepCfg, WINDOW, full_kernel,
};
use crate::simd::{axpy, dot};

const PAD: usize = (WINDOW - 1) / 2;

// ---------------------------------------------------------------------------------------
// Quantizer

#[inline]
fn softround(x: f32, t: f32) -> (f32, f32) {
    let fl = libm::floorf(x);
    let delta = x - fl - 0.5;
    let th = libm::tanhf(delta / t);
    let denom = libm::tanhf(1.0 / (2.0 * t));
    let y = fl + 0.5 * th / denom + 0.5;
    let dy = 0.5 * (1.0 - th * th) / t / denom;
    (y, dy)
}

/// `generate_kumaraswamy_noise`.
#[inline]
pub(super) fn kumaraswamy(u: f32, a: f32) -> f32 {
    let b = (libm::powf(2.0, a) * (a - 1.0) + 1.0) / a;
    libm::powf(1.0 - libm::powf(1.0 - u, 1.0 / b), 1.0 / a) - 0.5
}

// ---------------------------------------------------------------------------------------
// Laplace rate

#[inline]
fn cdf_and_grads(v: f32, mu: f32, b: f32) -> (f32, f32, f32) {
    // cdf = 0.5 - 0.5 * sign(d) * expm1(-|d| / b), d = v - mu
    let d = v - mu;
    let s = if d > 0.0 {
        1.0
    } else if d < 0.0 {
        -1.0
    } else {
        0.0
    };
    let ad = d.abs();
    let e = libm::expf(-ad / b);
    let cdf = 0.5 - 0.5 * s * libm::expm1f(-ad / b);
    let dd = 0.5 * s * s / b * e; // ∂cdf/∂d
    let db = -0.5 * s * e * (ad / (b * b)); // ∂cdf/∂b
    (cdf, dd, db)
}

/// Rate in bits of value `x` under the ARM output `(mu, raw log-scale)`, and its gradients
/// with respect to `x`, `mu`, raw log-scale.
#[inline]
pub(super) fn rate_and_grads(x: f32, mu: f32, raw_ls: f32) -> (f32, f32, f32, f32) {
    let ls = raw_ls - 4.0;
    let lsc = ls.clamp(-5.0, 5.0);
    let b = libm::expf(lsc);
    let (hi, dhi, bhi) = cdf_and_grads(x + 0.5, mu, b);
    let (lo, dlo, blo) = cdf_and_grads(x - 0.5, mu, b);
    let p = hi - lo;
    const P_MIN: f32 = 1.0 / 65536.0;
    let pc = p.max(P_MIN);
    let rate = -libm::log2f(pc);
    let drate_dp = if p >= P_MIN {
        -1.0 / (p * core::f32::consts::LN_2)
    } else {
        0.0
    };
    let dx = drate_dp * (dhi - dlo);
    let dmu = -dx;
    let db = drate_dp * (bhi - blo);
    let dls = if (-5.0..=5.0).contains(&ls) {
        db * b
    } else {
        0.0
    };
    (rate, dx, dmu, dls)
}

// ---------------------------------------------------------------------------------------
// MLP (one context vector at a time)

/// Forward through an MLP; `acts[l]` receives layer `l`'s input, `acts[n]` the output.
pub(super) fn mlp_forward(m: &Mlp, x: &[f32], acts: &mut Vec<Vec<f32>>) {
    acts.resize(m.layers.len() + 1, Vec::new());
    acts[0].clear();
    acts[0].extend_from_slice(x);
    let last = m.layers.len() - 1;
    for (l, lin) in m.layers.iter().enumerate() {
        let (inp, out) = acts.split_at_mut(l + 1);
        let h = &inp[l];
        let y = &mut out[0];
        y.clear();
        for o in 0..lin.n_out {
            let row = &lin.w[o * lin.n_in..(o + 1) * lin.n_in];
            let mut acc = lin.b[o];
            for i in 0..lin.n_in {
                acc += row[i] * h[i];
            }
            if lin.residual {
                acc += h[o];
            }
            y.push(if l < last { acc.max(0.0) } else { acc });
        }
    }
    if let Some(st) = &m.stabiliser {
        let n = acts.len() - 1;
        for o in 0..st.n_out {
            let row = &st.w[o * st.n_in..(o + 1) * st.n_in];
            let mut acc = st.b[o];
            for i in 0..st.n_in {
                acc += row[i] * x[i];
            }
            acts[n][o] += acc;
        }
    }
}

/// Backward for [`mlp_forward`]: accumulates weight gradients into `g` (f64) and writes the
/// input gradient to `dx`.
#[inline(never)]
fn mlp_backward(m: &Mlp, acts: &[Vec<f32>], dout: &[f32], g: &mut MlpGrad, dx: &mut [f32]) {
    let x = &acts[0];
    dx.fill(0.0);
    if let Some(st) = &m.stabiliser {
        let gs = g.stab.as_mut().unwrap();
        for o in 0..st.n_out {
            let d = dout[o];
            gs.b[o] += d as f64;
            for i in 0..st.n_in {
                gs.w[o * st.n_in + i] += (d * x[i]) as f64;
                dx[i] += st.w[o * st.n_in + i] * d;
            }
        }
    }
    let n = m.layers.len();
    let mut d: Vec<f32> = dout.to_vec();
    for l in (0..n).rev() {
        let lin = &m.layers[l];
        let h = &acts[l];
        if l < n - 1 {
            // ReLU on this layer's output.
            for (dv, &a) in d.iter_mut().zip(&acts[l + 1]) {
                if a <= 0.0 {
                    *dv = 0.0;
                }
            }
        }
        let gl = &mut g.layers[l];
        let mut dh = vec![0.0f32; lin.n_in];
        for o in 0..lin.n_out {
            let dv = d[o];
            gl.b[o] += dv as f64;
            for i in 0..lin.n_in {
                gl.w[o * lin.n_in + i] += (dv * h[i]) as f64;
                dh[i] += lin.w[o * lin.n_in + i] * dv;
            }
            if lin.residual {
                dh[o] += dv;
            }
        }
        d = dh;
    }
    for (a, b) in dx.iter_mut().zip(&d) {
        *a += *b;
    }
}

struct LinGrad {
    w: Vec<f64>,
    b: Vec<f64>,
}

struct MlpGrad {
    layers: Vec<LinGrad>,
    stab: Option<LinGrad>,
}

impl MlpGrad {
    fn new(m: &Mlp) -> Self {
        let lg = |l: &Linear| LinGrad {
            w: vec![0.0; l.w.len()],
            b: vec![0.0; l.b.len()],
        };
        Self {
            layers: m.layers.iter().map(lg).collect(),
            stab: m.stabiliser.as_ref().map(lg),
        }
    }

    fn write(&self, dst: &mut Mlp) {
        for (d, s) in dst
            .layers
            .iter_mut()
            .chain(dst.stabiliser.as_mut())
            .zip(self.layers.iter().chain(self.stab.as_ref()))
        {
            for (a, b) in d.w.iter_mut().zip(&s.w) {
                *a = *b as f32;
            }
            for (a, b) in d.b.iter_mut().zip(&s.b) {
                *a = *b as f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// Convolutions

pub(super) fn kron(w: &[f32]) -> Vec<f32> {
    let k = w.len();
    let mut out = vec![0.0; k * k];
    for r in 0..k {
        for c in 0..k {
            out[r * k + c] = w[c] * w[r];
        }
    }
    out
}

/// Gradient of the half kernel from the gradient of `kron(w, w)` (`w` = full kernel).
fn kron_backward(dk2: &[f32], w: &[f32], dhalf: &mut [f32]) {
    let k = w.len();
    for m in 0..k {
        let mut d = 0.0f32;
        for c in 0..k {
            d += dk2[m * k + c] * w[c];
        }
        for r in 0..k {
            d += dk2[r * k + m] * w[r];
        }
        dhalf[m.min(k - 1 - m)] += d;
    }
}

fn pad(src: &[f32], h: usize, w: usize, p: usize, replicate: bool) -> Vec<f32> {
    let pw = w + 2 * p;
    let mut out = vec![0.0f32; (h + 2 * p) * pw];
    for py in 0..h + 2 * p {
        for px in 0..pw {
            let inside = py >= p && py < h + p && px >= p && px < w + p;
            if inside || replicate {
                let sy = py.saturating_sub(p).min(h - 1);
                let sx = px.saturating_sub(p).min(w - 1);
                out[py * pw + px] = src[sy * w + sx];
            }
        }
    }
    out
}

/// Backward of [`pad`]: folds a padded gradient back onto the plane.
#[inline(never)]
fn unpad(dp: &[f32], h: usize, w: usize, p: usize, replicate: bool, dst: &mut [f32]) {
    let pw = w + 2 * p;
    for py in 0..h + 2 * p {
        for px in 0..pw {
            let inside = py >= p && py < h + p && px >= p && px < w + p;
            if inside || replicate {
                let sy = py.saturating_sub(p).min(h - 1);
                let sx = px.saturating_sub(p).min(w - 1);
                dst[sy * w + sx] += dp[py * pw + px];
            }
        }
    }
}

/// Train-mode transposed conv (`UpsamplingSeparableSymmetricConvTranspose2d`), cropped to
/// `(th, tw)` (≤ 2h × 2w).
///
/// Per output phase: uncropped output `f = 2m + parity` gathers `xp[m - a][n - b] ·
/// K[py + 2a][px + 2b]`, a contiguous correlation per row.
#[inline(never)]
pub(super) fn conv_t_forward(
    x: &[f32],
    h: usize,
    w: usize,
    k2: &[f32],
    k: usize,
    th: usize,
    tw: usize,
) -> Vec<f32> {
    let half = k / 2;
    let p0 = k / 2;
    let c = 2 * p0 - 1 + k / 2;
    let pw = w + 2 * p0;
    let xp = pad(x, h, w, p0, true);
    let m0 = c >> 1;
    let nm = ((tw - 1 + c) >> 1) - m0 + 1;
    let mut out = vec![0.0f32; th * tw];
    let mut tmp = [vec![0.0f32; nm], vec![0.0f32; nm]];
    for oy in 0..th {
        let f = oy + c;
        let (my, py) = (f >> 1, f & 1);
        for (px, t) in tmp.iter_mut().enumerate() {
            t.fill(0.0);
            for a in 0..half {
                let row = &xp[(my - a) * pw..(my - a + 1) * pw];
                for b in 0..half {
                    axpy(
                        t,
                        k2[(py + 2 * a) * k + px + 2 * b],
                        &row[m0 - b..m0 - b + nm],
                    );
                }
            }
        }
        for (ox, o) in out[oy * tw..(oy + 1) * tw].iter_mut().enumerate() {
            let f = ox + c;
            *o = tmp[f & 1][(f >> 1) - m0];
        }
    }
    out
}

#[inline(never)]
fn conv_t_backward(
    x: &[f32],
    h: usize,
    w: usize,
    k2: &[f32],
    k: usize,
    th: usize,
    tw: usize,
    dout: &[f32],
    dx: &mut [f32],
    dk2: &mut [f32],
) {
    let half = k / 2;
    let p0 = k / 2;
    let c = 2 * p0 - 1 + k / 2;
    let (ph, pw) = (h + 2 * p0, w + 2 * p0);
    let xp = pad(x, h, w, p0, true);
    let mut dxp = vec![0.0f32; ph * pw];
    let m0 = c >> 1;
    let nm = ((tw - 1 + c) >> 1) - m0 + 1;
    let mut dt = [vec![0.0f32; nm], vec![0.0f32; nm]];
    let mut dk = vec![0.0f64; k * k];
    for oy in 0..th {
        let f = oy + c;
        let (my, py) = (f >> 1, f & 1);
        dt[0].fill(0.0);
        dt[1].fill(0.0);
        for (ox, &d) in dout[oy * tw..(oy + 1) * tw].iter().enumerate() {
            let f = ox + c;
            dt[f & 1][(f >> 1) - m0] = d;
        }
        for (px, t) in dt.iter().enumerate() {
            for a in 0..half {
                let r0 = (my - a) * pw;
                for b in 0..half {
                    let ki = (py + 2 * a) * k + px + 2 * b;
                    dk[ki] += dot(t, &xp[r0 + m0 - b..r0 + m0 - b + nm]) as f64;
                    axpy(&mut dxp[r0 + m0 - b..r0 + m0 - b + nm], k2[ki], t);
                }
            }
        }
    }
    for (a, b) in dk2.iter_mut().zip(&dk) {
        *a += *b as f32;
    }
    unpad(&dxp, h, w, p0, true, dx);
}

/// Train-mode `UpsamplingSeparableSymmetricConv2d`: zero-padded conv with `kron(w, w)`, plus
/// the input.
#[inline(never)]
pub(super) fn sym_conv_forward(x: &[f32], h: usize, w: usize, k2: &[f32], k: usize) -> Vec<f32> {
    let p = k / 2;
    let pw = w + 2 * p;
    let xp = pad(x, h, w, p, false);
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        let orow = &mut out[y * w..(y + 1) * w];
        for ky in 0..k {
            let row = &xp[(y + ky) * pw..(y + ky + 1) * pw];
            for kx in 0..k {
                axpy(orow, k2[ky * k + kx], &row[kx..kx + w]);
            }
        }
        for (o, s) in orow.iter_mut().zip(&x[y * w..(y + 1) * w]) {
            *o += *s;
        }
    }
    out
}

#[inline(never)]
fn sym_conv_backward(
    x: &[f32],
    h: usize,
    w: usize,
    k2: &[f32],
    k: usize,
    dout: &[f32],
    dx: &mut [f32],
    dk2: &mut [f32],
) {
    let p = k / 2;
    let pw = w + 2 * p;
    let xp = pad(x, h, w, p, false);
    let mut dxp = vec![0.0f32; xp.len()];
    let mut dk = vec![0.0f64; k * k];
    for y in 0..h {
        let drow = &dout[y * w..(y + 1) * w];
        for ky in 0..k {
            let r0 = (y + ky) * pw;
            for kx in 0..k {
                dk[ky * k + kx] += dot(drow, &xp[r0 + kx..r0 + kx + w]) as f64;
                axpy(&mut dxp[r0 + kx..r0 + kx + w], k2[ky * k + kx], drow);
            }
        }
    }
    for (a, b) in dk2.iter_mut().zip(&dk) {
        *a += *b as f32;
    }
    unpad(&dxp, h, w, p, false, dx);
    for (a, b) in dx.iter_mut().zip(dout) {
        *a += *b;
    }
}

/// Synthesis conv forward (replicate padding, bias, residual; no activation). Parallel
/// over output channels.
pub(super) fn conv_forward(l: &ConvLayer, x: &[f32], h: usize, w: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let (k, p) = (l.k, l.k / 2);
    let pw = w + 2 * p;
    let n = h * w;
    let xp: Vec<Vec<f32>> = if k == 1 {
        Vec::new()
    } else {
        (0..l.c_in)
            .map(|i| pad(&x[i * n..(i + 1) * n], h, w, p, true))
            .collect()
    };
    let mut out = vec![0.0f32; l.c_out * n];
    out.par_chunks_mut(n).enumerate().for_each(|(o, dst)| {
        dst.fill(l.b[o]);
        for i in 0..l.c_in {
            let wk = &l.w[(o * l.c_in + i) * k * k..(o * l.c_in + i + 1) * k * k];
            if k == 1 {
                axpy(dst, wk[0], &x[i * n..(i + 1) * n]);
                continue;
            }
            for y in 0..h {
                let drow = &mut dst[y * w..(y + 1) * w];
                for ky in 0..k {
                    let row = &xp[i][(y + ky) * pw..(y + ky + 1) * pw];
                    for kx in 0..k {
                        axpy(drow, wk[ky * k + kx], &row[kx..kx + w]);
                    }
                }
            }
        }
        if l.residual {
            for (d, s) in dst.iter_mut().zip(&x[o * n..(o + 1) * n]) {
                *d += *s;
            }
        }
    });
    out
}

/// Backward of [`conv_forward`]; parallel over input channels (each owns its input
/// gradient and its column of weight gradients).
fn conv_backward(
    l: &ConvLayer,
    x: &[f32],
    h: usize,
    w: usize,
    dout: &[f32],
    dx: &mut [f32],
    g: &mut ConvLayer,
) {
    use rayon::prelude::*;
    let (k, p) = (l.k, l.k / 2);
    let pw = w + 2 * p;
    let n = h * w;
    let kk = k * k;
    // Per input channel: its weight gradients `[o][ky][kx]`.
    let gws: Vec<Vec<f32>> = dx[..l.c_in * n]
        .par_chunks_mut(n)
        .enumerate()
        .map(|(i, dxi)| {
            let xi = &x[i * n..(i + 1) * n];
            let mut gw = vec![0.0f32; l.c_out * kk];
            if k == 1 {
                for o in 0..l.c_out {
                    let d = &dout[o * n..(o + 1) * n];
                    let widx = o * l.c_in + i;
                    gw[o] = dot(d, xi);
                    axpy(dxi, l.w[widx], d);
                }
                return gw;
            }
            let xp = pad(xi, h, w, p, true);
            let mut dxp = vec![0.0f32; xp.len()];
            for o in 0..l.c_out {
                let d = &dout[o * n..(o + 1) * n];
                for ky in 0..k {
                    for kx in 0..k {
                        let wv = l.w[((o * l.c_in + i) * k + ky) * k + kx];
                        let mut acc = 0.0f64;
                        for y in 0..h {
                            let base = (y + ky) * pw + kx;
                            let drow = &d[y * w..(y + 1) * w];
                            acc += dot(drow, &xp[base..base + w]) as f64;
                            axpy(&mut dxp[base..base + w], wv, drow);
                        }
                        gw[o * kk + ky * k + kx] = acc as f32;
                    }
                }
            }
            unpad(&dxp, h, w, p, true, dxi);
            gw
        })
        .collect();
    for (i, gw) in gws.iter().enumerate() {
        for o in 0..l.c_out {
            for t in 0..kk {
                g.w[(o * l.c_in + i) * kk + t] += gw[o * kk + t];
            }
        }
    }
    for o in 0..l.c_out {
        let d = &dout[o * n..(o + 1) * n];
        g.b[o] += d.iter().map(|&v| v as f64).sum::<f64>() as f32;
        if l.residual {
            for (a, b) in dx[o * n..(o + 1) * n].iter_mut().zip(d) {
                *a += *b;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// IFCE stacking

/// For grid `idx`, where each channel of its IFCE input comes from: `(grid, flat index map)`
/// per channel, at the resolution `(uh, uw)` of grid `idx + 1`.
pub(super) struct Stack {
    pub(super) uh: usize,
    pub(super) uw: usize,
    /// (source grid, source flat index per stacked pixel); empty = the all-zero channel of
    /// the coarsest grid.
    pub(super) channels: Vec<(usize, Vec<u32>)>,
}

pub(super) fn ifce_stack(shape: &Shape, idx: usize) -> Stack {
    let n = shape.grids.len();
    if idx == n - 1 {
        let (h, w) = shape.grids[idx];
        return Stack {
            uh: h,
            uw: w,
            channels: vec![(usize::MAX, Vec::new())],
        };
    }
    let (uh, uw) = shape.grids[idx + 1];
    let mut channels = Vec::new();
    for j in idx + 1..n {
        let (_, gw) = shape.grids[j];
        let mut map = Vec::with_capacity(uh * uw);
        for y in 0..uh {
            for x in 0..uw {
                let (mut yy, mut xx) = (y, x);
                for m in idx + 1..j {
                    if shape.grids[m] != shape.grids[m + 1] {
                        yy /= 2;
                        xx /= 2;
                    }
                }
                map.push((yy * gw + xx) as u32);
            }
        }
        channels.push((j, map));
    }
    Stack { uh, uw, channels }
}

// ---------------------------------------------------------------------------------------
// The step

pub(super) fn step(
    shape: &Shape,
    p: &Params,
    target: &[f32],
    noise: &[f32],
    cfg: StepCfg,
    grads: &mut Params,
) -> LossTerms {
    let lambda = cfg.lambda;
    let backward = cfg.eval_bitdepth.is_none();
    grads.for_each_mut(|v| v.fill(0.0));
    let n_grids = shape.grids.len();
    let (ih, iw) = shape.img;
    let n_pix = ih * iw;

    // 1. Quantize.
    let mut q: Vec<Vec<f32>> = Vec::with_capacity(n_grids);
    let mut dq_dy: Vec<Vec<f32>> = Vec::with_capacity(n_grids);
    let mut off = 0;
    for lat in &p.latents {
        let mut qq = Vec::with_capacity(lat.len());
        let mut dd = Vec::with_capacity(lat.len());
        for (i, &y) in lat.iter().enumerate() {
            let x = y * cfg.gain;
            match cfg.quantizer {
                Quantizer::Softround { temperature } => {
                    let (s1, d1) = softround(x, temperature);
                    let n = noise.get(off + i).copied().unwrap_or(0.0);
                    let (s2, d2) = softround(s1 + n, temperature);
                    qq.push(s2);
                    dd.push(cfg.gain * d1 * d2);
                }
                Quantizer::Hardround => {
                    qq.push(libm::rintf(x));
                    dd.push(0.0);
                }
            }
        }
        off += lat.len();
        q.push(qq);
        dq_dy.push(dd);
    }
    let mut dq: Vec<Vec<f32>> = q.iter().map(|g| vec![0.0; g.len()]).collect();

    // 2. IFCE contexts (forward).
    let n_ifce = shape.n_ifce_out;
    struct IfceTape {
        stack: Stack,
        input: Vec<f32>, // pixel-major, channels
        out: Vec<f32>,   // pixel-major, n_ifce
    }
    let mut ifce_tape: Vec<Option<IfceTape>> = (0..n_grids).map(|_| None).collect();
    let mut inter: Vec<Option<Vec<f32>>> = (0..n_grids).map(|_| None).collect();
    if n_ifce > 0 {
        for idx in 0..n_grids {
            let (gh, gw) = shape.grids[idx];
            let mut ctx = vec![0.0f32; gh * gw * n_ifce];
            if let Some(m) = &p.ifce[idx] {
                let st = ifce_stack(shape, idx);
                let uc = st.channels.len();
                let mut input = vec![0.0f32; st.uh * st.uw * uc];
                for (c, (j, map)) in st.channels.iter().enumerate() {
                    if *j == usize::MAX {
                        continue;
                    }
                    for (px, &s) in map.iter().enumerate() {
                        input[px * uc + c] = q[*j][s as usize];
                    }
                }
                let mut out = vec![0.0f32; st.uh * st.uw * n_ifce];
                let mut acts = Vec::new();
                for px in 0..st.uh * st.uw {
                    mlp_forward(m, &input[px * uc..(px + 1) * uc], &mut acts);
                    out[px * n_ifce..(px + 1) * n_ifce].copy_from_slice(&acts[acts.len() - 1]);
                }
                for y in 0..gh {
                    for x in 0..gw {
                        let s = ((y / 2) * st.uw + x / 2) * n_ifce;
                        ctx[(y * gw + x) * n_ifce..(y * gw + x + 1) * n_ifce]
                            .copy_from_slice(&out[s..s + n_ifce]);
                    }
                }
                ifce_tape[idx] = Some(IfceTape {
                    stack: st,
                    input,
                    out,
                });
            }
            inter[idx] = Some(ctx);
        }
    }
    let mut d_inter: Vec<Vec<f32>> = inter
        .iter()
        .map(|c| c.as_ref().map_or(Vec::new(), |c| vec![0.0; c.len()]))
        .collect();

    // 3. ARM + rate (row-batched, `arm_batch`).
    let rate_w = lambda / n_pix as f32;
    let inter_cm: Vec<Option<Vec<f32>>> = inter
        .iter()
        .enumerate()
        .map(|(g, c)| {
            c.as_ref()
                .map(|c| to_channel_major(c, shape.grids[g].0 * shape.grids[g].1, n_ifce))
        })
        .collect();
    let mut d_inter_cm: Vec<Vec<f32>> = inter_cm
        .iter()
        .map(|c| c.as_ref().map_or(Vec::new(), |c| vec![0.0; c.len()]))
        .collect();
    let mut arm_g = ArmGrad::new(&p.arm);
    let rate_total = arm_pass(
        shape,
        &p.arm,
        &q,
        &inter_cm,
        backward.then_some(ArmBackward {
            grad: &mut arm_g,
            dq: &mut dq,
            d_inter: &mut d_inter_cm,
            rate_w,
        }),
    );
    if backward {
        arm_g.write(&mut grads.arm);
        for (g, d) in d_inter_cm.iter().enumerate() {
            if !d.is_empty() {
                d_inter[g] = to_pixel_major(d, shape.grids[g].0 * shape.grids[g].1, n_ifce);
            }
        }
    }

    // 4. Upsampling + synthesis forward, with the tape needed for backward.
    let lat_idx: Vec<usize> = (0..n_grids).filter(|&g| !shape.is_hyper[g]).collect();
    let nl = lat_idx.len();
    let n_ups = p.ups_t.len().max(1);
    let full_t: Vec<Vec<f32>> = p
        .ups_t
        .iter()
        .map(|h| full_kernel(h, shape.ups_k))
        .collect();
    let full_c: Vec<Vec<f32>> = p
        .ups_c
        .iter()
        .map(|h| full_kernel(h, shape.ups_pre_k))
        .collect();
    let k2_t: Vec<Vec<f32>> = full_t.iter().map(|w| kron(w)).collect();
    let k2_c: Vec<Vec<f32>> = full_c.iter().map(|w| kron(w)).collect();
    struct UpsTape {
        prev: Vec<f32>, // stack before this step: c planes at (ph, pw)
        pc: usize,
        ph: usize,
        pw: usize,
        target: usize, // grid index
    }
    let mut ups_tape = Vec::new();
    let last = lat_idx[nl - 1];
    let mut up = q[last].clone();
    let (mut uc, mut uh, mut uw) = (1usize, shape.grids[last].0, shape.grids[last].1);
    for (step_i, &t) in lat_idx[..nl - 1].iter().rev().enumerate() {
        let (th, tw) = shape.grids[t];
        let ki = step_i % n_ups;
        let mut next = Vec::with_capacity((uc + 1) * th * tw);
        next.extend(sym_conv_forward(&q[t], th, tw, &k2_c[ki], shape.ups_pre_k));
        {
            use rayon::prelude::*;
            let chans: Vec<Vec<f32>> = (0..uc)
                .into_par_iter()
                .map(|c| {
                    conv_t_forward(
                        &up[c * uh * uw..(c + 1) * uh * uw],
                        uh,
                        uw,
                        &k2_t[ki],
                        shape.ups_k,
                        th,
                        tw,
                    )
                })
                .collect();
            for ch in chans {
                next.extend(ch);
            }
        }
        ups_tape.push(UpsTape {
            prev: core::mem::take(&mut up),
            pc: uc,
            ph: uh,
            pw: uw,
            target: t,
        });
        up = next;
        uc += 1;
        uh = th;
        uw = tw;
    }
    debug_assert_eq!((uh, uw), (ih, iw));

    let mut syn_inputs: Vec<Vec<f32>> = Vec::with_capacity(p.syn.len());
    let mut cur = up.clone();
    for l in &p.syn {
        syn_inputs.push(cur.clone());
        let mut o = conv_forward(l, &cur, ih, iw);
        if l.relu {
            for v in &mut o {
                *v = v.max(0.0);
            }
        }
        cur = o;
    }
    if let Some(st) = &p.syn_stab {
        let s = conv_forward(st, &up[..st.c_in * n_pix], ih, iw);
        for (a, b) in cur.iter_mut().zip(&s) {
            *a += *b;
        }
    }
    let pre_out = cur;
    let raw = conv_forward(&p.out_transform, &pre_out, ih, iw);

    // 5. Loss.
    let n_el = raw.len();
    let mut mse = 0.0f64;
    let mut draw = vec![0.0f32; n_el];
    let out_max = cfg.eval_bitdepth.map(|b| ((1u32 << b) - 1) as f32);
    for i in 0..n_el {
        let mut c = raw[i].clamp(0.0, 1.0);
        if let Some(m) = out_max {
            c = libm::rintf(c * m) / m;
        }
        let e = c - target[i];
        mse += (e * e) as f64;
        if (0.0..=1.0).contains(&raw[i]) {
            draw[i] = 2.0 * e / n_el as f32;
        }
    }
    let mse = mse / n_el as f64;
    if !backward {
        return LossTerms {
            loss: mse + lambda as f64 * rate_total / n_pix as f64,
            mse,
            rate_bits: rate_total,
        };
    }

    // 6. Synthesis backward.
    let mut d_pre = vec![0.0f32; pre_out.len()];
    conv_backward(
        &p.out_transform,
        &pre_out,
        ih,
        iw,
        &draw,
        &mut d_pre,
        &mut grads.out_transform,
    );
    let mut d_up = vec![0.0f32; up.len()];
    if let (Some(st), Some(gst)) = (&p.syn_stab, grads.syn_stab.as_mut()) {
        conv_backward(
            st,
            &up[..st.c_in * n_pix],
            ih,
            iw,
            &d_pre,
            &mut d_up[..st.c_in * n_pix],
            gst,
        );
    }
    let mut d = d_pre;
    for (li, l) in p.syn.iter().enumerate().rev() {
        let x = &syn_inputs[li];
        if l.relu {
            // Output of this layer = input of the next (or pre_out for the last one).
            let o = if li + 1 < p.syn.len() {
                &syn_inputs[li + 1]
            } else {
                &pre_out_before_stab(&p.syn, &syn_inputs, ih, iw)
            };
            for (dv, &ov) in d.iter_mut().zip(o) {
                if ov <= 0.0 {
                    *dv = 0.0;
                }
            }
        }
        let g = &mut grads.syn[li];
        let mut dx = vec![0.0f32; x.len()];
        conv_backward(l, x, ih, iw, &d, &mut dx, g);
        d = dx;
    }
    for (a, b) in d_up.iter_mut().zip(&d) {
        *a += *b;
    }

    // 7. Upsampling backward.
    let mut dk2_t: Vec<Vec<f32>> = k2_t.iter().map(|k| vec![0.0; k.len()]).collect();
    let mut dk2_c: Vec<Vec<f32>> = k2_c.iter().map(|k| vec![0.0; k.len()]).collect();
    let (mut dc, mut dh, mut dw) = (uc, uh, uw);
    let mut dcur = d_up;
    for (step_i, tape) in ups_tape.iter().enumerate().rev() {
        let ki = step_i % n_ups;
        let (th, tw) = (dh, dw);
        debug_assert_eq!(shape.grids[tape.target], (th, tw));
        let n = th * tw;
        // Channel 0: the pre-filtered target.
        sym_conv_backward(
            &q[tape.target],
            th,
            tw,
            &k2_c[ki],
            shape.ups_pre_k,
            &dcur[..n],
            &mut dq[tape.target],
            &mut dk2_c[ki],
        );
        let mut dprev = vec![0.0f32; tape.pc * tape.ph * tape.pw];
        let pn = tape.ph * tape.pw;
        {
            // Parallel over channels; kernel gradients merged in channel order.
            use rayon::prelude::*;
            let k2 = &k2_t[ki];
            let dks: Vec<Vec<f32>> = dprev
                .par_chunks_mut(pn)
                .enumerate()
                .map(|(c, dp)| {
                    let mut dk = vec![0.0f32; k2.len()];
                    conv_t_backward(
                        &tape.prev[c * pn..(c + 1) * pn],
                        tape.ph,
                        tape.pw,
                        k2,
                        shape.ups_k,
                        th,
                        tw,
                        &dcur[(1 + c) * n..(2 + c) * n],
                        dp,
                        &mut dk,
                    );
                    dk
                })
                .collect();
            for dk in dks {
                for (a, b) in dk2_t[ki].iter_mut().zip(&dk) {
                    *a += *b;
                }
            }
        }
        dcur = dprev;
        dc -= 1;
        dh = tape.ph;
        dw = tape.pw;
    }
    debug_assert_eq!(dc, 1);
    for (a, b) in dq[last].iter_mut().zip(&dcur) {
        *a += *b;
    }
    for (i, half) in grads.ups_t.iter_mut().enumerate() {
        kron_backward(&dk2_t[i], &full_t[i], half);
    }
    for (i, half) in grads.ups_c.iter_mut().enumerate() {
        kron_backward(&dk2_c[i], &full_c[i], half);
    }

    // 8. IFCE backward.
    for idx in 0..n_grids {
        let (Some(m), Some(tape)) = (&p.ifce[idx], &ifce_tape[idx]) else {
            continue;
        };
        let (gh, gw) = shape.grids[idx];
        let st = &tape.stack;
        let uc = st.channels.len();
        let mut dout = vec![0.0f32; st.uh * st.uw * n_ifce];
        for y in 0..gh {
            for x in 0..gw {
                let s = ((y / 2) * st.uw + x / 2) * n_ifce;
                for k in 0..n_ifce {
                    dout[s + k] += d_inter[idx][(y * gw + x) * n_ifce + k];
                }
            }
        }
        let mut mg = MlpGrad::new(m);
        let mut acts = Vec::new();
        let mut dx = vec![0.0f32; uc];
        for px in 0..st.uh * st.uw {
            mlp_forward(m, &tape.input[px * uc..(px + 1) * uc], &mut acts);
            debug_assert_eq!(
                acts[acts.len() - 1][..],
                tape.out[px * n_ifce..(px + 1) * n_ifce]
            );
            mlp_backward(
                m,
                &acts,
                &dout[px * n_ifce..(px + 1) * n_ifce],
                &mut mg,
                &mut dx,
            );
            for (c, (j, map)) in st.channels.iter().enumerate() {
                if *j != usize::MAX {
                    dq[*j][map[px] as usize] += dx[c];
                }
            }
        }
        mg.write(grads.ifce[idx].as_mut().unwrap());
    }

    // 9. Quantizer backward.
    for g in 0..n_grids {
        for ((d, &a), &b) in grads.latents[g].iter_mut().zip(&dq[g]).zip(&dq_dy[g]) {
            *d = a * b;
        }
    }

    let rate_bits = rate_total;
    LossTerms {
        loss: mse + lambda as f64 * rate_bits / n_pix as f64,
        mse,
        rate_bits,
    }
}

/// The last main-branch layer's output (before the stabiliser is added), recomputed for
/// its ReLU mask.
fn pre_out_before_stab(syn: &[ConvLayer], inputs: &[Vec<f32>], h: usize, w: usize) -> Vec<f32> {
    let l = syn.last().unwrap();
    let mut o = conv_forward(l, inputs.last().unwrap(), h, w);
    if l.relu {
        for v in &mut o {
            *v = v.max(0.0);
        }
    }
    o
}
