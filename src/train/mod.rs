//! Training-time Cool-chic: the differentiable forward pass of upstream's
//! `CoolChicEncoder.forward` (training mode) and its hand-written backward pass.
//!
//! Graph, per upstream `a6fe38a`:
//! 1. latents `y` (one grid per resolution, finest first) are scaled by the encoder gain
//!    (16) and quantized: `q = softround(softround(16 y, T) + noise, T)`;
//! 2. a float ARM sees, for every latent value, its causal 9×9 neighbours (priority order)
//!    and the IFCE context computed from the coarser grids, and outputs (mu, log-scale);
//!    the rate is `-log2(clamp(F(q + 1/2) - F(q - 1/2), 2^-16))` under that Laplace;
//! 3. upsampling (train-mode `kron` kernels) and synthesis turn the non-hyper grids into
//!    the picture, clamped to [0, 1];
//! 4. `loss = mse + lambda · Σrate / n_pixels`.
//!
//! The backward pass is checked against `torch.autograd` (`tests` below, `--features
//! oracle`, dumps from `scripts/oracle/dump_grad.py`).

#![allow(clippy::needless_range_loop)]

use alloc::vec;
use alloc::vec::Vec;

use crate::decode::Arch;
use crate::latent::{MASK_SIZE, context_window_indices};
use crate::nn::{layer_dims, symmetric_kernel};

mod arm_batch;
pub(crate) mod config;
pub(crate) mod eval;
pub(crate) mod ifce;
mod ops;
pub(crate) mod quantize;
pub(crate) mod rdoq;
pub(crate) mod soap;
#[cfg(all(test, feature = "oracle"))]
mod tests;
pub(crate) mod trainer;
pub(crate) mod wasserstein;

/// Float linear layer, `y = W x + b (+ x if residual)`, `W` row-major `[out][in]`.
#[derive(Clone, Debug)]
pub(crate) struct Linear {
    pub(crate) n_in: usize,
    pub(crate) n_out: usize,
    pub(crate) residual: bool,
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

impl Linear {
    pub(crate) fn zeros(n_in: usize, n_out: usize, residual: bool) -> Self {
        Self {
            n_in,
            n_out,
            residual,
            w: vec![0.0; n_in * n_out],
            b: vec![0.0; n_out],
        }
    }
}

/// Float ARM: residual hidden layers with ReLU, a last layer and an optional linear
/// stabiliser from the input.
#[derive(Clone, Debug)]
pub(crate) struct Mlp {
    pub(crate) layers: Vec<Linear>,
    pub(crate) stabiliser: Option<Linear>,
}

/// `SynthesisConv2d`: `[out][in][k][k]` weights, replicate padding.
#[derive(Clone, Debug)]
pub(crate) struct ConvLayer {
    pub(crate) c_in: usize,
    pub(crate) c_out: usize,
    pub(crate) k: usize,
    pub(crate) residual: bool,
    pub(crate) relu: bool,
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
}

impl ConvLayer {
    fn zeros(c_in: usize, c_out: usize, k: usize, residual: bool, relu: bool) -> Self {
        Self {
            c_in,
            c_out,
            k,
            residual,
            relu,
            w: vec![0.0; c_out * c_in * k * k],
            b: vec![0.0; c_out],
        }
    }
}

/// Every trainable tensor of one Cool-chic (and, as [`Params::zeros_like`], its gradient).
#[derive(Clone, Debug)]
pub(crate) struct Params {
    /// Latent grids, finest first, before the encoder gain.
    pub(crate) latents: Vec<Vec<f32>>,
    pub(crate) arm: Mlp,
    /// One IFCE per grid (None where the grid has none).
    pub(crate) ifce: Vec<Option<Mlp>>,
    /// Half kernels of the transposed convs and pre-concat convs.
    pub(crate) ups_t: Vec<Vec<f32>>,
    pub(crate) ups_c: Vec<Vec<f32>>,
    pub(crate) syn: Vec<ConvLayer>,
    pub(crate) syn_stab: Option<ConvLayer>,
    /// Frozen (not optimized) but differentiated through.
    pub(crate) out_transform: ConvLayer,
}

impl Params {
    pub(crate) fn zeros_like(&self) -> Self {
        let mut p = self.clone();
        p.for_each_mut(|v| v.fill(0.0));
        p
    }

    /// Every tensor's shape and role, in [`Params::for_each_mut`] order.
    pub(crate) fn layout(&self) -> Vec<(Vec<usize>, Role)> {
        let mut out = Vec::new();
        for l in &self.latents {
            out.push((vec![1, 1, l.len()], Role::Latent));
        }
        let mlp = |m: &Mlp, out: &mut Vec<(Vec<usize>, Role)>| {
            for l in m.layers.iter().chain(m.stabiliser.as_ref()) {
                out.push((vec![l.n_out, l.n_in], Role::Weight));
                out.push((vec![l.n_out], Role::Weight));
            }
        };
        mlp(&self.arm, &mut out);
        for m in self.ifce.iter().flatten() {
            mlp(m, &mut out);
        }
        for k in self.ups_t.iter().chain(&self.ups_c) {
            out.push((vec![k.len()], Role::Weight));
        }
        for c in self.syn.iter().chain(self.syn_stab.as_ref()) {
            out.push((vec![c.c_out, c.c_in, c.k, c.k], Role::Weight));
            out.push((vec![c.c_out], Role::Weight));
        }
        let c = &self.out_transform;
        out.push((vec![c.c_out, c.c_in, c.k, c.k], Role::Frozen));
        out.push((vec![c.c_out], Role::Frozen));
        out
    }

    /// Read-only views of every tensor, in [`Params::for_each_mut`] order.
    pub(crate) fn tensors(&self) -> Vec<&[f32]> {
        let mut out: Vec<&[f32]> = Vec::new();
        for l in &self.latents {
            out.push(l);
        }
        for m in core::iter::once(&self.arm).chain(self.ifce.iter().flatten()) {
            for l in m.layers.iter().chain(m.stabiliser.as_ref()) {
                out.push(&l.w);
                out.push(&l.b);
            }
        }
        for k in self.ups_t.iter().chain(&self.ups_c) {
            out.push(k);
        }
        for c in self
            .syn
            .iter()
            .chain(self.syn_stab.as_ref())
            .chain(Some(&self.out_transform))
        {
            out.push(&c.w);
            out.push(&c.b);
        }
        out
    }

    /// Visits every tensor (latents, weights, biases, including the frozen ones).
    pub(crate) fn for_each_mut(&mut self, mut f: impl FnMut(&mut [f32])) {
        for l in &mut self.latents {
            f(l);
        }
        let mlp = |m: &mut Mlp, f: &mut dyn FnMut(&mut [f32])| {
            for l in m.layers.iter_mut().chain(m.stabiliser.as_mut()) {
                f(&mut l.w);
                f(&mut l.b);
            }
        };
        mlp(&mut self.arm, &mut f);
        for m in self.ifce.iter_mut().flatten() {
            mlp(m, &mut f);
        }
        for k in self.ups_t.iter_mut().chain(&mut self.ups_c) {
            f(k);
        }
        for c in self
            .syn
            .iter_mut()
            .chain(self.syn_stab.as_mut())
            .chain(Some(&mut self.out_transform))
        {
            f(&mut c.w);
            f(&mut c.b);
        }
    }
}

/// Which optimizer group a tensor belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Latent,
    Weight,
    /// The synthesis output transform: never optimized upstream.
    Frozen,
}

/// Static shape information derived from the header-level architecture.
#[derive(Clone, Debug)]
pub(crate) struct Shape {
    pub(crate) img: (usize, usize),
    pub(crate) grids: Vec<(usize, usize)>,
    pub(crate) is_hyper: Vec<bool>,
    /// Window cells (0..40) of the spatial contexts, priority order.
    pub(crate) window: Vec<usize>,
    pub(crate) n_ifce_out: usize,
    pub(crate) ups_k: usize,
    pub(crate) ups_pre_k: usize,
    /// `Some(n)`: the synthesis stabiliser reads the first `n` input channels.
    pub(crate) syn_stab_in: Option<usize>,
    /// Common-randomness synthesis channels at image resolution (`[c][h][w]` planes),
    /// generated once from the fixed `CommonGaussianNoiseGenerator` stream.
    pub(crate) cr: Option<Vec<f32>>,
    /// IFCE input layout of every grid (a function of the grid sizes only).
    pub(crate) ifce_stacks: Vec<ifce::Stack>,
}

impl Shape {
    pub(crate) fn new(h: &crate::header::CoolChicHeader, a: &Arch) -> crate::Result<Self> {
        if h.latent_resolution[0] != 0 {
            return Err(crate::Error::Unsupported(
                "training with latents below full resolution",
            ));
        }
        let (ih, iw) = (h.img_size[0] as usize, h.img_size[1] as usize);
        // Same stream as the decoder: one noise plane per common-randomness latent,
        // `fixed_upsampling` + bicubic resize to the image size (upstream
        // `CoolChicEncoder.get_synthesis_input`).
        let cr = h.flag_common_randomness.then(|| {
            let mut noise = crate::nn::CommonNoise::new();
            let planes: Vec<crate::nn::Planes> = a
                .size_per_latent_cr
                .iter()
                .map(|&(ch, cw)| noise.sample(ch, cw))
                .collect();
            let up = crate::nn::fixed_upsampling(&planes, crate::header::FinalUpsampling::Bicubic);
            crate::nn::resize(
                &up,
                ih,
                iw,
                crate::header::FinalUpsampling::Bicubic,
                crate::nn::Scale::Size,
            )
            .data
        });
        Ok(Self {
            img: (ih, iw),
            grids: a.size_per_latent.clone(),
            is_hyper: a.is_hyperlatent.clone(),
            window: context_window_indices(h.spatial_context_arm as usize)?,
            n_ifce_out: h.output_feature_ifce as usize,
            ups_k: h.ups_k_size as usize,
            ups_pre_k: h.ups_preconcat_k_size as usize,
            // With common randomness the stabiliser still reads only the latent
            // channels (upstream divides `input_feature_synthesis` by two).
            syn_stab_in: h
                .linear_stabiliser_synth
                .then_some(if h.flag_common_randomness {
                    a.input_feature_synthesis / 2
                } else {
                    a.input_feature_synthesis
                }),
            cr,
            ifce_stacks: (0..a.size_per_latent.len())
                .map(|i| ifce::stack_for(&a.size_per_latent, i))
                .collect(),
        })
    }

    /// Zero-initialized parameters with the right shapes.
    pub(crate) fn zero_params(&self, h: &crate::header::CoolChicHeader, a: &Arch) -> Params {
        let dim = a.total_context_arm;
        let mut layers: Vec<Linear> = (0..h.n_hidden_layers_arm)
            .map(|_| Linear::zeros(dim, dim, true))
            .collect();
        layers.push(Linear::zeros(dim, 2, false));
        let arm = Mlp {
            layers,
            stabiliser: h
                .linear_stabiliser_arm
                .then(|| Linear::zeros(dim, 2, false)),
        };
        let ifce = a
            .input_features_ifce
            .iter()
            .map(|&n| {
                (a.flag_ifce && n > 0).then(|| Mlp {
                    layers: vec![Linear::zeros(n, self.n_ifce_out, false)],
                    stabiliser: None,
                })
            })
            .collect();
        let n_ups = h.latent_resolution[1] as usize;
        let syn_dims = layer_dims(a.input_feature_synthesis, &h.layers_synthesis);
        let syn = syn_dims
            .iter()
            .zip(&h.layers_synthesis)
            .map(|(&(ci, co, k), l)| ConvLayer::zeros(ci, co, k, l.residual, l.relu))
            .collect();
        let c_out = h.layers_synthesis.last().unwrap().out_ft as usize;
        Params {
            latents: self
                .grids
                .iter()
                .map(|&(gh, gw)| vec![0.0; gh * gw])
                .collect(),
            arm,
            ifce,
            ups_t: vec![vec![0.0; self.ups_k.div_ceil(2)]; n_ups],
            ups_c: vec![vec![0.0; self.ups_pre_k.div_ceil(2)]; n_ups],
            syn,
            syn_stab: self
                .syn_stab_in
                .map(|n| ConvLayer::zeros(n, c_out, 1, false, false)),
            out_transform: ConvLayer::zeros(c_out, c_out, 1, false, false),
        }
    }
}

/// How latents are quantized in one forward pass (upstream `quantize`).
#[derive(Clone, Copy, Debug)]
pub(crate) enum Quantizer {
    /// `softround(softround(gain · y, T) + noise, T)`; noise is supplied by the caller.
    Softround { temperature: f32 },
    /// `round(gain · y)`: no gradient reaches the latents.
    Hardround,
}

/// Per-metric distortion weights of one forward pass (upstream `dist_weight`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dist {
    pub(crate) mse: f32,
    /// Weight of the VGG16 Wasserstein term (0 for the default MSE tuning).
    pub(crate) wass: f32,
}

impl Dist {
    /// `{"mse": 1.0}` (upstream `--tune mse`, also the warm-up phases of every preset).
    pub(crate) const MSE: Self = Self {
        mse: 1.0,
        wass: 0.0,
    };
    /// `{"mse": 0.2, "wasserstein": 0.8 / 200}` (upstream `--tune wasserstein`).
    pub(crate) const WASSERSTEIN: Self = Self {
        mse: 0.2,
        wass: 0.004,
    };
}

/// Settings of one forward pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct StepCfg {
    pub(crate) quantizer: Quantizer,
    /// Upstream `encoder_gain` (16, or 20 / 24 at low lambda).
    pub(crate) gain: f32,
    pub(crate) lambda: f32,
    /// `Some(bitdepth)`: evaluation pass (upstream `test()`): output rounded to the bit
    /// depth before the distortion, no backward pass.
    pub(crate) eval_bitdepth: Option<u32>,
    /// Distortion weights of the phase (upstream `training_phase.dist_weight`).
    pub(crate) dist: Dist,
    /// Layout of the frame data the distortion sees (upstream `frame_data_type`):
    /// the synthesis output is downsampled to 4:2:0 before scoring for `Yuv420`.
    pub(crate) frame_data_type: crate::header::FrameDataType,
}

/// Loss terms of one forward pass.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LossTerms {
    /// `dist + lambda · rate / n_pixels`.
    pub(crate) loss: f64,
    /// `dist_weight · distortion`: `w_mse · mse + w_wass · wasserstein`.
    pub(crate) dist: f64,
    /// The MSE alone (PSNR reporting).
    pub(crate) mse: f64,
    /// The Wasserstein distance alone (0 without the feature weights).
    pub(crate) wass: f64,
    pub(crate) rate_bits: f64,
}

/// One forward pass, plus the backward pass unless `cfg.eval_bitdepth` is set. `noise` is
/// the additive quantization noise, one value per latent (all grids concatenated, finest
/// first); empty for no noise. `grads` is overwritten (left zero for an evaluation pass).
/// `wass` must be given when `cfg.dist.wass` is non-zero.
pub(crate) fn forward_backward(
    shape: &Shape,
    p: &Params,
    target: &[f32],
    noise: &[f32],
    cfg: StepCfg,
    wass: Option<&wasserstein::Wasserstein>,
    grads: &mut Params,
) -> LossTerms {
    ops::step(shape, p, target, noise, cfg, wass, grads)
}

/// Kumaraswamy noise from uniform samples (upstream `generate_kumaraswamy_noise`).
pub(crate) fn kumaraswamy_noise(uniform: &[f32], a: f32) -> Vec<f32> {
    uniform.iter().map(|&u| ops::kumaraswamy(u, a)).collect()
}

pub(crate) fn full_kernel(half: &[f32], k: usize) -> Vec<f32> {
    symmetric_kernel(half, k)
}

pub(crate) const WINDOW: usize = MASK_SIZE;
