//! Inter-feature context extraction (IFCE), forward and backward, channel-major.
//!
//! For grid `idx`, the IFCE input stacks the already-coded coarser grids at the resolution
//! of grid `idx + 1` (nearest ×2 at each resolution step; a single zero channel for the
//! coarsest grid). Upstream's IFCE is one linear layer: `out = W · input + b`, then nearest
//! ×2 and a crop to grid `idx`'s size.

use alloc::vec;
use alloc::vec::Vec;

use super::{Mlp, Shape};
use crate::simd::{axpy, dot};

/// Where each IFCE input channel of a grid comes from.
#[derive(Clone, Debug)]
pub(crate) struct Stack {
    pub(crate) uh: usize,
    pub(crate) uw: usize,
    /// (source grid or `usize::MAX` for the zero channel, flat source index per pixel).
    pub(crate) channels: Vec<(usize, Vec<u32>)>,
}

/// The stack layout of grid `idx` (a function of the shapes only).
pub(crate) fn stack_for(grids: &[(usize, usize)], idx: usize) -> Stack {
    let n = grids.len();
    if idx == n - 1 {
        let (h, w) = grids[idx];
        return Stack {
            uh: h,
            uw: w,
            channels: vec![(usize::MAX, Vec::new())],
        };
    }
    let (uh, uw) = grids[idx + 1];
    let mut channels = Vec::new();
    for j in idx + 1..n {
        let (_, gw) = grids[j];
        let mut map = Vec::with_capacity(uh * uw);
        for y in 0..uh {
            for x in 0..uw {
                let (mut yy, mut xx) = (y, x);
                for m in idx + 1..j {
                    if grids[m] != grids[m + 1] {
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

/// Saved inputs of the forward pass, channel-major `[c][uh·uw]`, per grid with an IFCE.
pub(crate) type Tape = Vec<Option<Vec<f32>>>;

/// Per grid: (context, saved input).
type GridForward = (Option<Vec<f32>>, Option<Vec<f32>>);

/// Per grid: (weight gradient, bias gradient, input gradient).
type GridBackward = Option<(Vec<f32>, Vec<f32>, Vec<f32>)>;

/// IFCE contexts of every grid, channel-major `[k][h·w]` (None without IFCE), and the tape.
pub(crate) fn forward(
    shape: &Shape,
    ifce: &[Option<Mlp>],
    q: &[Vec<f32>],
) -> (Vec<Option<Vec<f32>>>, Tape) {
    use rayon::prelude::*;
    let n_ifce = shape.n_ifce_out;
    let n = shape.grids.len();
    if n_ifce == 0 {
        return (
            (0..n).map(|_| None).collect(),
            (0..n).map(|_| None).collect(),
        );
    }
    let res: Vec<GridForward> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let (gh, gw) = shape.grids[idx];
            let mut ctx = vec![0.0f32; n_ifce * gh * gw];
            let Some(m) = &ifce[idx] else {
                return (Some(ctx), None);
            };
            let lin = &m.layers[0];
            debug_assert!(m.layers.len() == 1 && m.stabiliser.is_none());
            let st = &shape.ifce_stacks[idx];
            let np = st.uh * st.uw;
            let uc = st.channels.len();
            let mut input = vec![0.0f32; uc * np];
            for (c, (j, map)) in st.channels.iter().enumerate() {
                if *j != usize::MAX {
                    for (d, &s) in input[c * np..(c + 1) * np].iter_mut().zip(map) {
                        *d = q[*j][s as usize];
                    }
                }
            }
            let mut out = vec![0.0f32; np];
            for k in 0..n_ifce {
                out.fill(lin.b[k]);
                for c in 0..uc {
                    axpy(&mut out, lin.w[k * uc + c], &input[c * np..(c + 1) * np]);
                }
                let dst = &mut ctx[k * gh * gw..(k + 1) * gh * gw];
                for y in 0..gh {
                    for x in 0..gw {
                        dst[y * gw + x] = out[(y / 2) * st.uw + x / 2];
                    }
                }
            }
            (Some(ctx), Some(input))
        })
        .collect();
    res.into_iter().unzip()
}

/// Backward: `d_inter` is the context gradient (channel-major, as [`forward`] returns);
/// accumulates into `dq` and writes the IFCE weight gradients into `grads`.
pub(crate) fn backward(
    shape: &Shape,
    ifce: &[Option<Mlp>],
    tape: &Tape,
    d_inter: &[Vec<f32>],
    dq: &mut [Vec<f32>],
    grads: &mut [Option<Mlp>],
) {
    use rayon::prelude::*;
    let n_ifce = shape.n_ifce_out;
    // Per grid (in parallel): weight gradients and the input gradient; scatter afterwards
    // in grid order.
    let parts: Vec<GridBackward> = (0..shape.grids.len())
        .into_par_iter()
        .map(|idx| {
            let (Some(m), Some(input)) = (&ifce[idx], &tape[idx]) else {
                return None;
            };
            let lin = &m.layers[0];
            let (gh, gw) = shape.grids[idx];
            let st = &shape.ifce_stacks[idx];
            let np = st.uh * st.uw;
            let uc = st.channels.len();
            let mut gw_ = vec![0.0f32; lin.w.len()];
            let mut gb = vec![0.0f32; lin.b.len()];
            let mut din = vec![0.0f32; uc * np];
            let mut dout = vec![0.0f32; np];
            for k in 0..n_ifce {
                dout.fill(0.0);
                let src = &d_inter[idx][k * gh * gw..(k + 1) * gh * gw];
                for y in 0..gh {
                    for x in 0..gw {
                        dout[(y / 2) * st.uw + x / 2] += src[y * gw + x];
                    }
                }
                gb[k] = dout.iter().map(|&v| v as f64).sum::<f64>() as f32;
                for c in 0..uc {
                    gw_[k * uc + c] = dot(&dout, &input[c * np..(c + 1) * np]);
                    axpy(&mut din[c * np..(c + 1) * np], lin.w[k * uc + c], &dout);
                }
            }
            Some((gw_, gb, din))
        })
        .collect();
    for (idx, part) in parts.into_iter().enumerate() {
        let Some((gw_, gb, din)) = part else { continue };
        let g = &mut grads[idx].as_mut().unwrap().layers[0];
        g.w.copy_from_slice(&gw_);
        g.b.copy_from_slice(&gb);
        let st = &shape.ifce_stacks[idx];
        let np = st.uh * st.uw;
        for (c, (j, map)) in st.channels.iter().enumerate() {
            if *j != usize::MAX {
                for (&d, &s) in din[c * np..(c + 1) * np].iter().zip(map) {
                    dq[*j][s as usize] += d;
                }
            }
        }
    }
}
