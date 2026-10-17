//! Fixed-point ARM and IFCE (upstream `coolchic/bitstream/component/armint.py`).
//!
//! The entropy model must match the encoder bit for bit, so it runs in i64 fixed point:
//! weights carry 16 fractional bits, biases 32, activations are shifted back to 16 after
//! every hidden layer. Residual hidden layers fold the identity into the weight matrix.
//! Arithmetic wraps on overflow like PyTorch's int64 `addmm`.

#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

use crate::error::{Error, Result};

pub(crate) const WEIGHT_SHIFT: u32 = 16;
/// Fractional bits of the IFCE output fed to the ARM.
pub(crate) const N_FRAC_BIT_INTER_FT_CTX: u32 = 8;
/// Fractional bits of the ARM's (mu, log-scale) output.
pub(crate) const N_FRAC_BIT_MU_SCALE: u32 = 8;

/// One linear layer, `y = W x + b` with `W` stored `[out][in]` (row per output).
#[derive(Debug, Clone)]
pub(crate) struct FixedLinear {
    pub(crate) n_in: usize,
    pub(crate) n_out: usize,
    pub(crate) w: Vec<i64>,
    pub(crate) b: Vec<i64>,
}

/// An MLP in fixed point: hidden layers (ReLU) + a last layer + a linear stabiliser branch
/// running from the input (all-zero when the model has none).
#[derive(Debug, Clone)]
pub(crate) struct FixedArm {
    pub(crate) layers: Vec<FixedLinear>,
    pub(crate) stabiliser: FixedLinear,
}

/// Quantized (integer) parameters of one float-side ARM, in upstream layout:
/// `weights[l]` is `[out][in]` row-major, then the optional stabiliser.
pub(crate) struct QuantArm {
    pub(crate) dims: Vec<(usize, usize)>, // (in, out) per trunk layer
    pub(crate) weights: Vec<Vec<i64>>,
    pub(crate) biases: Vec<Vec<i64>>,
    pub(crate) stabiliser: Option<(Vec<i64>, Vec<i64>)>,
}

pub(crate) struct FixedPointOptions {
    /// Apply the hard-coded `-4` log-scale shift to the last layer's second bias.
    pub(crate) subtract_last_layer: bool,
    /// Number of trailing inputs that are IFCE outputs (8 fractional bits instead of 0).
    pub(crate) n_inter_ft_ctx: usize,
    /// Square layers are plain, not residual (true for the IFCE).
    pub(crate) no_residual_layer: bool,
}

fn shl(v: i64, s: i32) -> Result<i64> {
    // q_step >= 2^-16 and target shifts are 16 / 32, so shifts here are >= 0.
    if !(0..63).contains(&s) {
        return Err(Error::Invalid("fixed-point shift out of range"));
    }
    Ok(v.wrapping_mul(1i64 << s))
}

/// `arm_to_fixed_point_param`: converts integer-quantized parameters (`q_param =
/// param / q_step`) to fixed point. `q_shift_w` / `q_shift_b` are log2 of the steps.
pub(crate) fn to_fixed_point(
    q: &mut QuantArm,
    q_shift_w: i32,
    q_shift_b: i32,
    opt: &FixedPointOptions,
) -> Result<FixedArm> {
    let n_layers = q.dims.len();
    if n_layers == 0
        || q.dims
            .iter()
            .any(|&(i, o)| i == 0 || i > MAX_WIDTH || o > MAX_WIDTH)
    {
        return Err(Error::Invalid("ARM layer width"));
    }
    let mut layers = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let (n_in, n_out) = q.dims[l];
        let is_last = l == n_layers - 1;
        let inter_cols = if l == 0 { opt.n_inter_ft_ctx } else { 0 };

        // Weights: target 16 fractional bits, 8 fewer on IFCE input columns.
        let mut w = vec![0i64; n_in * n_out];
        for o in 0..n_out {
            for i in 0..n_in {
                let mut s = WEIGHT_SHIFT as i32 + q_shift_w;
                let mut ident = WEIGHT_SHIFT as i32;
                if i >= n_in - inter_cols {
                    s -= N_FRAC_BIT_INTER_FT_CTX as i32;
                    ident -= N_FRAC_BIT_INTER_FT_CTX as i32;
                }
                let mut v = shl(q.weights[l][o * n_in + i], s)?;
                if n_in == n_out && !opt.no_residual_layer && i == o {
                    v = v.wrapping_add(1i64 << ident);
                }
                w[o * n_in + i] = v;
            }
        }

        // Biases: target 32 fractional bits.
        let bias = &mut q.biases[l];
        if is_last && opt.subtract_last_layer {
            if bias.len() < 2 || q_shift_b > 0 {
                return Err(Error::Invalid("ARM last layer"));
            }
            bias[1] = bias[1].wrapping_sub(4i64 << (-q_shift_b));
        }
        let s = 2 * WEIGHT_SHIFT as i32 + q_shift_b;
        let b = bias
            .iter()
            .map(|&v| shl(v, s))
            .collect::<Result<Vec<_>>>()?;
        layers.push(FixedLinear { n_in, n_out, w, b });
    }

    let (n_in, _) = q.dims[0];
    let n_out = q.dims[n_layers - 1].1;
    let stabiliser = match &q.stabiliser {
        Some((sw, sb)) => {
            let mut w = vec![0i64; n_in * n_out];
            for o in 0..n_out {
                for i in 0..n_in {
                    let mut s = WEIGHT_SHIFT as i32 + q_shift_w;
                    if i >= n_in - opt.n_inter_ft_ctx {
                        s -= N_FRAC_BIT_INTER_FT_CTX as i32;
                    }
                    w[o * n_in + i] = shl(sw[o * n_in + i], s)?;
                }
            }
            let s = 2 * WEIGHT_SHIFT as i32 + q_shift_b;
            let b = sb.iter().map(|&v| shl(v, s)).collect::<Result<Vec<_>>>()?;
            FixedLinear { n_in, n_out, w, b }
        }
        None => FixedLinear {
            n_in,
            n_out,
            w: vec![0; n_in * n_out],
            b: vec![0; n_out],
        },
    };
    Ok(FixedArm { layers, stabiliser })
}

impl FixedArm {
    pub(crate) fn n_in(&self) -> usize {
        self.stabiliser.n_in
    }

    pub(crate) fn n_out(&self) -> usize {
        self.stabiliser.n_out
    }

    /// `fixed_point_arm` for one context vector: writes `n_out` values to `out`.
    pub(crate) fn forward(&self, ctx: &[i64], output_shift: u32, out: &mut [i64]) {
        let n_in = self.n_in();
        debug_assert_eq!(ctx.len(), n_in);
        let mut buf_a = [0i64; MAX_WIDTH];
        let mut buf_b = [0i64; MAX_WIDTH];
        for (c, &x) in buf_a.iter_mut().zip(ctx) {
            *c = x.wrapping_shl(WEIGHT_SHIFT);
        }
        let x0 = &buf_a[..n_in];

        // Stabiliser from the shifted input, then the trunk.
        let st = &self.stabiliser;
        for ((o, row), &b) in out.iter_mut().zip(st.w.chunks_exact(n_in)).zip(&st.b) {
            *o = dot(row, x0).wrapping_add(b);
        }

        let (last, hidden) = self.layers.split_last().expect("ARM has a last layer");
        let (mut cur, mut next) = (&mut buf_a, &mut buf_b);
        for l in hidden {
            let x = &cur[..l.n_in];
            for ((y, row), &b) in next.iter_mut().zip(l.w.chunks_exact(l.n_in)).zip(&l.b) {
                *y = dot(row, x).wrapping_add(b).max(0) >> WEIGHT_SHIFT;
            }
            core::mem::swap(&mut cur, &mut next);
        }
        let x = &cur[..last.n_in];
        for ((o, row), &b) in out
            .iter_mut()
            .zip(last.w.chunks_exact(last.n_in))
            .zip(&last.b)
        {
            *o = dot(row, x).wrapping_add(b).wrapping_add(*o) >> output_shift;
        }
    }
}

/// Widest layer the fixed-point MLPs can have: 40 spatial contexts + 31 IFCE features.
pub(crate) const MAX_WIDTH: usize = 72;

/// Wrapping i64 dot product (order-independent: integer addition is associative).
#[inline(always)]
fn dot(a: &[i64], b: &[i64]) -> i64 {
    a.iter()
        .zip(b)
        .fold(0i64, |acc, (&x, &y)| acc.wrapping_add(x.wrapping_mul(y)))
}
