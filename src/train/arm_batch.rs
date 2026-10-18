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

/// Rows per parallel work item. Fixed (not derived from the thread count) so results do
/// not depend on how many threads run.
const BLOCK_ROWS: usize = 16;

/// Per-block results, merged in block order.
struct BlockOut {
    rate: f64,
    grad: Option<MlpGrad>,
    /// Gradient w.r.t. the padded latent rows `y0 .. y1 + 2·PAD` (context scatter).
    dqpad: Vec<f32>,
    /// Gradient w.r.t. the block's own latents (rate term), rows `y0 .. y1`.
    dq_own: Vec<f32>,
    /// Channel-major IFCE-context gradient of the block's rows: `[k][(y1 - y0)·w]`.
    d_inter: Vec<f32>,
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
    use rayon::prelude::*;
    let mut total = 0.0f64;
    let rate_w = bw.as_ref().map(|b| b.rate_w);
    for (g, qg) in q.iter().enumerate() {
        let (gh, gw) = shape.grids[g];
        let pw = gw + 2 * PAD;
        let mut qpad = vec![0.0f32; (gh + 2 * PAD) * pw];
        for y in 0..gh {
            qpad[(y + PAD) * pw + PAD..(y + PAD) * pw + PAD + gw]
                .copy_from_slice(&qg[y * gw..(y + 1) * gw]);
        }
        let n_blocks = gh.div_ceil(BLOCK_ROWS);
        let outs: Vec<BlockOut> = (0..n_blocks)
            .into_par_iter()
            .map(|bi| {
                let y0 = bi * BLOCK_ROWS;
                let y1 = (y0 + BLOCK_ROWS).min(gh);
                block_pass(
                    shape,
                    arm,
                    g,
                    qg,
                    &qpad,
                    inter[g].as_deref(),
                    y0,
                    y1,
                    rate_w,
                )
            })
            .collect();
        let n_ifce = shape.n_ifce_out;
        for (bi, o) in outs.into_iter().enumerate() {
            total += o.rate;
            let Some(b) = &mut bw else { continue };
            let y0 = bi * BLOCK_ROWS;
            let y1 = (y0 + BLOCK_ROWS).min(gh);
            if let Some(gr) = &o.grad {
                for (d, s) in b.grad.layers.iter_mut().zip(&gr.layers) {
                    for (a, c) in d.0.iter_mut().zip(&s.0) {
                        *a += *c;
                    }
                    for (a, c) in d.1.iter_mut().zip(&s.1) {
                        *a += *c;
                    }
                }
                if let (Some(d), Some(s)) = (&mut b.grad.stab, &gr.stab) {
                    for (a, c) in d.0.iter_mut().zip(&s.0) {
                        *a += *c;
                    }
                    for (a, c) in d.1.iter_mut().zip(&s.1) {
                        *a += *c;
                    }
                }
            }
            for (a, c) in b.dq[g][y0 * gw..y1 * gw].iter_mut().zip(&o.dq_own) {
                *a += *c;
            }
            // Fold the padded patch (padded rows y0 .. y1 + 2·PAD) onto the latents.
            for pr in 0..(y1 - y0 + 2 * PAD) {
                let py = y0 + pr;
                if py < PAD || py >= gh + PAD {
                    continue;
                }
                let y = py - PAD;
                for (a, c) in b.dq[g][y * gw..(y + 1) * gw]
                    .iter_mut()
                    .zip(&o.dqpad[pr * pw + PAD..pr * pw + PAD + gw])
                {
                    *a += *c;
                }
            }
            if inter[g].is_some() {
                let rows = (y1 - y0) * gw;
                for k in 0..n_ifce {
                    for (a, c) in b.d_inter[g][k * gh * gw + y0 * gw..k * gh * gw + y1 * gw]
                        .iter_mut()
                        .zip(&o.d_inter[k * rows..(k + 1) * rows])
                    {
                        *a += *c;
                    }
                }
            }
        }
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn block_pass(
    shape: &Shape,
    arm: &Mlp,
    g: usize,
    qg: &[f32],
    qpad: &[f32],
    inter: Option<&[f32]>,
    y0: usize,
    y1: usize,
    rate_w: Option<f32>,
) -> BlockOut {
    let n_sp = shape.window.len();
    let n_ifce = shape.n_ifce_out;
    let offsets: Vec<(usize, usize)> = shape
        .window
        .iter()
        .map(|&cell| (cell / WINDOW, cell % WINDOW))
        .collect();
    let n_layers = arm.layers.len();
    let (gh, gw) = shape.grids[g];
    let pw = gw + 2 * PAD;
    let backward = rate_w.is_some();
    let rows = y1 - y0;
    let mut out = BlockOut {
        rate: 0.0,
        grad: backward.then(|| MlpGrad::new(arm)),
        dqpad: if backward {
            vec![0.0; (rows + 2 * PAD) * pw]
        } else {
            Vec::new()
        },
        dq_own: if backward {
            vec![0.0; rows * gw]
        } else {
            Vec::new()
        },
        d_inter: if backward && inter.is_some() {
            vec![0.0; n_ifce * rows * gw]
        } else {
            Vec::new()
        },
    };
    let dim = n_sp + if inter.is_some() { n_ifce } else { 0 };
    let n = gw;
    let mut acts: Vec<Vec<f32>> = (0..=n_layers)
        .map(|l| vec![0.0f32; if l == 0 { dim } else { arm.layers[l - 1].n_out } * n])
        .collect();
    let mut dout = vec![0.0f32; 2 * n];
    let mut dx = vec![0.0f32; dim * n];
    for y in y0..y1 {
        {
            let x = &mut acts[0];
            for (j, &(cy, cx)) in offsets.iter().enumerate() {
                let src = (y + cy) * pw + cx;
                x[j * n..(j + 1) * n].copy_from_slice(&qpad[src..src + n]);
            }
            if let Some(ic) = inter {
                for k in 0..n_ifce {
                    x[(n_sp + k) * n..(n_sp + k + 1) * n]
                        .copy_from_slice(&ic[k * gh * gw + y * gw..k * gh * gw + (y + 1) * gw]);
                }
            }
        }
        for (l, lin) in arm.layers.iter().enumerate() {
            let (inp, o) = acts.split_at_mut(l + 1);
            let h = &inp[l];
            let z = &mut o[0];
            for oc in 0..lin.n_out {
                let zo = &mut z[oc * n..(oc + 1) * n];
                zo.fill(lin.b[oc]);
                for i in 0..lin.n_in {
                    axpy(zo, lin.w[oc * lin.n_in + i], &h[i * n..(i + 1) * n]);
                }
                if lin.residual {
                    for (a, b) in zo.iter_mut().zip(&h[oc * n..(oc + 1) * n]) {
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
            let (inp, o) = acts.split_at_mut(n_layers);
            let x = &inp[0];
            let z = &mut o[0];
            let mut s = vec![0.0f32; n];
            for oc in 0..2 {
                s.fill(st.b[oc]);
                for i in 0..st.n_in {
                    axpy(&mut s, st.w[oc * st.n_in + i], &x[i * n..(i + 1) * n]);
                }
                for (a, b) in z[oc * n..(oc + 1) * n].iter_mut().zip(&s) {
                    *a += *b;
                }
            }
        }
        let res = &acts[n_layers];
        let row = &qg[y * gw..(y + 1) * gw];
        for k in 0..n {
            let (rate, drx, dmu, dls) = rate_and_grads(row[k], res[k], res[n + k]);
            out.rate += rate as f64;
            if let Some(rw) = rate_w {
                dout[k] = dmu * rw;
                dout[n + k] = dls * rw;
                out.dq_own[(y - y0) * gw + k] += drx * rw;
            }
        }
        let Some(gr) = &mut out.grad else { continue };
        let x0 = &acts[0];
        dx.fill(0.0);
        if let (Some(st), Some(gs)) = (&arm.stabiliser, &mut gr.stab) {
            for oc in 0..2 {
                let d = &dout[oc * n..(oc + 1) * n];
                gs.1[oc] += d.iter().map(|&v| v as f64).sum::<f64>();
                for i in 0..st.n_in {
                    gs.0[oc * st.n_in + i] += dot(d, &x0[i * n..(i + 1) * n]) as f64;
                    axpy(&mut dx[i * n..(i + 1) * n], st.w[oc * st.n_in + i], d);
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
            let (gw_l, gb_l) = &mut gr.layers[l];
            let mut dh = vec![0.0f32; lin.n_in * n];
            for oc in 0..lin.n_out {
                let dz = &d[oc * n..(oc + 1) * n];
                gb_l[oc] += dz.iter().map(|&v| v as f64).sum::<f64>();
                for i in 0..lin.n_in {
                    gw_l[oc * lin.n_in + i] += dot(dz, &h[i * n..(i + 1) * n]) as f64;
                    axpy(&mut dh[i * n..(i + 1) * n], lin.w[oc * lin.n_in + i], dz);
                }
                if lin.residual {
                    for (a, c) in dh[oc * n..(oc + 1) * n].iter_mut().zip(dz) {
                        *a += *c;
                    }
                }
            }
            d = dh;
        }
        for (a, c) in dx.iter_mut().zip(&d) {
            *a += *c;
        }
        // Context-gradient scatter into the block's padded patch (row y + cy is patch row
        // y - y0 + cy).
        for (j, &(cy, cx)) in offsets.iter().enumerate() {
            let dst = (y - y0 + cy) * pw + cx;
            for (a, c) in out.dqpad[dst..dst + n]
                .iter_mut()
                .zip(&dx[j * n..(j + 1) * n])
            {
                *a += *c;
            }
        }
        if inter.is_some() {
            let r = rows * gw;
            for k in 0..n_ifce {
                let di = &mut out.d_inter[k * r + (y - y0) * gw..k * r + (y - y0 + 1) * gw];
                for (a, c) in di.iter_mut().zip(&dx[(n_sp + k) * n..(n_sp + k + 1) * n]) {
                    *a += *c;
                }
            }
        }
    }
    out
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
