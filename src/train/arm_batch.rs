//! The ARM over every latent, one grid row at a time, channel-major: each context channel
//! is a contiguous slice of a zero-padded latent row, so every layer is a run of FMA axpy
//! passes and weight gradients are dot products (`crate::simd`).

use alloc::vec;
use alloc::vec::Vec;

use super::ops::rate_and_grads;
use super::{Mlp, Shape, WINDOW};
use crate::simd::{axpy, dot};

const PAD: usize = (WINDOW - 1) / 2;

/// f64 gradient accumulators of an MLP, `[layer](w, b)` then the stabiliser.
pub(super) struct MlpGrad {
    pub(super) layers: Vec<(Vec<f64>, Vec<f64>)>,
    pub(super) stab: Option<(Vec<f64>, Vec<f64>)>,
}

impl MlpGrad {
    pub(super) fn new(m: &Mlp) -> Self {
        Self {
            layers: m
                .layers
                .iter()
                .map(|l| (vec![0.0; l.w.len()], vec![0.0; l.b.len()]))
                .collect(),
            stab: m
                .stabiliser
                .as_ref()
                .map(|l| (vec![0.0; l.w.len()], vec![0.0; l.b.len()])),
        }
    }

    pub(super) fn write(&self, dst: &mut Mlp) {
        for (d, s) in dst
            .layers
            .iter_mut()
            .chain(dst.stabiliser.as_mut())
            .zip(self.layers.iter().chain(self.stab.as_ref()))
        {
            for (a, b) in d.w.iter_mut().zip(&s.0) {
                *a = *b as f32;
            }
            for (a, b) in d.b.iter_mut().zip(&s.1) {
                *a = *b as f32;
            }
        }
    }
}

/// Outputs of [`arm_pass`] when differentiating.
pub(super) struct ArmBackward<'a> {
    pub(super) grad: &'a mut MlpGrad,
    /// Gradient w.r.t. each grid's quantized latents (accumulated).
    pub(super) dq: &'a mut [Vec<f32>],
    /// Gradient w.r.t. each grid's IFCE context, channel-major `[k][h·w]` (accumulated).
    pub(super) d_inter: &'a mut [Vec<f32>],
    /// `λ / pixels`: weight of the rate in the loss.
    pub(super) rate_w: f32,
}

/// Total latent rate in bits; with `bw`, also the backward pass of `rate_w · rate`.
/// `inter` holds each grid's IFCE context channel-major (`[k][h·w]`), or None.
pub(super) fn arm_pass(
    shape: &Shape,
    arm: &Mlp,
    q: &[Vec<f32>],
    inter: &[Option<Vec<f32>>],
    mut bw: Option<ArmBackward<'_>>,
) -> f64 {
    let n_sp = shape.window.len();
    let n_ifce = shape.n_ifce_out;
    let offsets: Vec<(usize, usize)> = shape
        .window
        .iter()
        .map(|&cell| (cell / WINDOW, cell % WINDOW))
        .collect();
    let n_layers = arm.layers.len();
    let mut total = 0.0f64;
    for (g, qg) in q.iter().enumerate() {
        let (gh, gw) = shape.grids[g];
        let pw = gw + 2 * PAD;
        let mut qpad = vec![0.0f32; (gh + 2 * PAD) * pw];
        for y in 0..gh {
            qpad[(y + PAD) * pw + PAD..(y + PAD) * pw + PAD + gw]
                .copy_from_slice(&qg[y * gw..(y + 1) * gw]);
        }
        let mut dqpad = if bw.is_some() {
            vec![0.0f32; qpad.len()]
        } else {
            Vec::new()
        };
        let dim = n_sp + if inter[g].is_some() { n_ifce } else { 0 };
        debug_assert_eq!(dim, arm.layers[0].n_in);
        let n = gw;
        // acts[l]: input of layer l, `[channel][n]`; acts[n_layers]: output (2 × n).
        let mut acts: Vec<Vec<f32>> = (0..=n_layers)
            .map(|l| vec![0.0f32; if l == 0 { dim } else { arm.layers[l - 1].n_out } * n])
            .collect();
        let mut dout = vec![0.0f32; 2 * n];
        for y in 0..gh {
            // Contexts.
            {
                let x = &mut acts[0];
                for (j, &(cy, cx)) in offsets.iter().enumerate() {
                    let src = (y + cy) * pw + cx; // padded row y + cy - PAD, column cx - PAD
                    x[j * n..(j + 1) * n].copy_from_slice(&qpad[src..src + n]);
                }
                if let Some(ic) = &inter[g] {
                    for k in 0..n_ifce {
                        x[(n_sp + k) * n..(n_sp + k + 1) * n]
                            .copy_from_slice(&ic[k * gh * gw + y * gw..k * gh * gw + (y + 1) * gw]);
                    }
                }
            }
            // Forward.
            for (l, lin) in arm.layers.iter().enumerate() {
                let (inp, out) = acts.split_at_mut(l + 1);
                let h = &inp[l];
                let z = &mut out[0];
                for o in 0..lin.n_out {
                    let zo = &mut z[o * n..(o + 1) * n];
                    zo.fill(lin.b[o]);
                    for i in 0..lin.n_in {
                        axpy(zo, lin.w[o * lin.n_in + i], &h[i * n..(i + 1) * n]);
                    }
                    if lin.residual {
                        for (a, b) in zo.iter_mut().zip(&h[o * n..(o + 1) * n]) {
                            *a += *b;
                        }
                    }
                    if l + 1 < n_layers {
                        for v in zo.iter_mut() {
                            *v = v.max(0.0);
                        }
                    }
                }
            }
            if let Some(st) = &arm.stabiliser {
                let (inp, out) = acts.split_at_mut(n_layers);
                let x = &inp[0];
                let z = &mut out[0];
                for o in 0..2 {
                    let mut s = vec![st.b[o]; n];
                    for i in 0..st.n_in {
                        axpy(&mut s, st.w[o * st.n_in + i], &x[i * n..(i + 1) * n]);
                    }
                    for (a, b) in z[o * n..(o + 1) * n].iter_mut().zip(&s) {
                        *a += *b;
                    }
                }
            }
            // Rate.
            let out = &acts[n_layers];
            let row = &qg[y * gw..(y + 1) * gw];
            for k in 0..n {
                let (rate, dx, dmu, dls) = rate_and_grads(row[k], out[k], out[n + k]);
                total += rate as f64;
                if let Some(b) = &mut bw {
                    dout[k] = dmu * b.rate_w;
                    dout[n + k] = dls * b.rate_w;
                    b.dq[g][y * gw + k] += dx * b.rate_w;
                }
            }
            let Some(b) = &mut bw else { continue };
            // Backward.
            let x0 = &acts[0];
            let mut dx = vec![0.0f32; dim * n];
            if let (Some(st), Some(gs)) = (&arm.stabiliser, &mut b.grad.stab) {
                for o in 0..2 {
                    let d = &dout[o * n..(o + 1) * n];
                    gs.1[o] += d.iter().map(|&v| v as f64).sum::<f64>();
                    for i in 0..st.n_in {
                        gs.0[o * st.n_in + i] += dot(d, &x0[i * n..(i + 1) * n]) as f64;
                        axpy(&mut dx[i * n..(i + 1) * n], st.w[o * st.n_in + i], d);
                    }
                }
            }
            let mut d = dout.clone();
            for l in (0..n_layers).rev() {
                let lin = &arm.layers[l];
                if l + 1 < n_layers {
                    for (dv, &a) in d.iter_mut().zip(&acts[l + 1]) {
                        if a <= 0.0 {
                            *dv = 0.0;
                        }
                    }
                }
                let h = &acts[l];
                let (gw_l, gb_l) = &mut b.grad.layers[l];
                let mut dh = vec![0.0f32; lin.n_in * n];
                for o in 0..lin.n_out {
                    let dz = &d[o * n..(o + 1) * n];
                    gb_l[o] += dz.iter().map(|&v| v as f64).sum::<f64>();
                    for i in 0..lin.n_in {
                        gw_l[o * lin.n_in + i] += dot(dz, &h[i * n..(i + 1) * n]) as f64;
                        axpy(&mut dh[i * n..(i + 1) * n], lin.w[o * lin.n_in + i], dz);
                    }
                    if lin.residual {
                        for (a, c) in dh[o * n..(o + 1) * n].iter_mut().zip(dz) {
                            *a += *c;
                        }
                    }
                }
                d = dh;
            }
            for (a, c) in dx.iter_mut().zip(&d) {
                *a += *c;
            }
            // Scatter the context gradients.
            for (j, &(cy, cx)) in offsets.iter().enumerate() {
                let dst = (y + cy) * pw + cx;
                for (a, c) in dqpad[dst..dst + n].iter_mut().zip(&dx[j * n..(j + 1) * n]) {
                    *a += *c;
                }
            }
            if inter[g].is_some() {
                for k in 0..n_ifce {
                    let di = &mut b.d_inter[g][k * gh * gw + y * gw..k * gh * gw + (y + 1) * gw];
                    for (a, c) in di.iter_mut().zip(&dx[(n_sp + k) * n..(n_sp + k + 1) * n]) {
                        *a += *c;
                    }
                }
            }
        }
        if let Some(b) = &mut bw {
            for y in 0..gh {
                for (a, c) in b.dq[g][y * gw..(y + 1) * gw]
                    .iter_mut()
                    .zip(&dqpad[(y + PAD) * pw + PAD..(y + PAD) * pw + PAD + gw])
                {
                    *a += *c;
                }
            }
        }
    }
    total
}

/// Pixel-major `[pos][k]` → channel-major `[k][pos]`.
pub(super) fn to_channel_major(v: &[f32], n_pos: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    for p in 0..n_pos {
        for c in 0..k {
            out[c * n_pos + p] = v[p * k + c];
        }
    }
    out
}

/// Channel-major `[k][pos]` → pixel-major `[pos][k]`.
pub(super) fn to_pixel_major(v: &[f32], n_pos: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; v.len()];
    for p in 0..n_pos {
        for c in 0..k {
            out[p * k + c] = v[c * n_pos + p];
        }
    }
    out
}
