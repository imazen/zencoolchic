//! Training loop, initialization, presets and warm-up (upstream `training/train.py`,
//! `training/presets.py`, `training/warmup.py`, `component/core/*.initialize_parameters`).

use alloc::vec;
use alloc::vec::Vec;

use super::soap::{SoapConfig, SoapState};
use super::wasserstein::Wasserstein;
use super::{Dist, LossTerms, Params, Quantizer, Role, Shape, StepCfg, forward_backward};

/// xoshiro256** with Box–Muller normals. Not torch's generator: only the distributions match.
#[derive(Clone, Debug)]
pub(crate) struct Rng {
    s: [u64; 4],
    spare: Option<f32>,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Self {
            s: [next(), next(), next(), next()],
            spare: None,
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub(crate) fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub(crate) fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u1 = ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let r = libm::sqrt(-2.0 * libm::log(u1));
        let (s, c) = (
            libm::sin(2.0 * core::f64::consts::PI * u2),
            libm::cos(2.0 * core::f64::consts::PI * u2),
        );
        self.spare = Some((r * s) as f32);
        (r * c) as f32
    }
}

/// Upstream's initialization of every module; latents start at zero.
pub(crate) fn initialize(p: &mut Params, shape: &Shape, rng: &mut Rng) {
    for l in &mut p.latents {
        l.fill(0.0);
    }
    // ArmLinear: zero bias; residual layers zero, others N(0, 1) / out².
    let lin = |l: &mut super::Linear, rng: &mut Rng| {
        l.b.fill(0.0);
        if l.residual {
            l.w.fill(0.0);
        } else {
            let s = 1.0 / (l.n_out * l.n_out) as f32;
            for w in &mut l.w {
                *w = rng.normal() * s;
            }
        }
    };
    for l in p.arm.layers.iter_mut().chain(p.arm.stabiliser.as_mut()) {
        lin(l, rng);
    }
    for m in p.ifce.iter_mut().flatten() {
        for l in m.layers.iter_mut() {
            lin(l, rng);
        }
    }
    // SynthesisConv2d: zero bias; residual layers zero, others U(-1, 1) · sqrt(1 / fan_in) / out².
    let conv = |c: &mut super::ConvLayer, rng: &mut Rng| {
        c.b.fill(0.0);
        if c.residual {
            c.w.fill(0.0);
        } else {
            let k = libm::sqrtf(1.0 / (c.c_in * c.k * c.k) as f32) / (c.c_out * c.c_out) as f32;
            for w in &mut c.w {
                *w = (rng.uniform() - 0.5) * 2.0 * k;
            }
        }
    };
    for c in p.syn.iter_mut().chain(p.syn_stab.as_mut()) {
        conv(c, rng);
    }
    // Output transform: identity (upstream rescales it only for YUV inputs).
    let ot = &mut p.out_transform;
    ot.w.fill(0.0);
    ot.b.fill(0.0);
    for c in 0..ot.c_out {
        ot.w[c * ot.c_in + c] = 1.0;
    }
    // Upsampling: bilinear (k < 8) or bicubic-like transposed kernels; identity pre-concat.
    let core_t: &[f32] = if shape.ups_k < 8 {
        &[0.25, 0.75]
    } else {
        &[0.0351562, 0.1054687, -0.2617187, -0.8789063]
    };
    for k in &mut p.ups_t {
        k.fill(0.0);
        let n = k.len();
        k[n - core_t.len()..].copy_from_slice(core_t);
    }
    for k in &mut p.ups_c {
        k.fill(0.0);
        *k.last_mut().unwrap() = 1.0;
    }
}

/// Quantization noise of a training phase (upstream `quantizer_noise_type`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Noise {
    None,
    /// `N(0, 1) · noise_parameter`.
    Gaussian,
    /// Kumaraswamy with `a = noise_parameter`.
    Kumaraswamy,
}

/// One training phase (upstream `TrainerPhase`).
#[derive(Clone, Debug)]
pub(crate) struct Phase {
    pub(crate) lr: f32,
    pub(crate) max_itr: usize,
    pub(crate) freq_valid: usize,
    pub(crate) patience: usize,
    pub(crate) schedule_lr: bool,
    pub(crate) softround_temperature: (f32, f32),
    pub(crate) noise_parameter: (f32, f32),
    pub(crate) noise: Noise,
    pub(crate) hardround: bool,
    pub(crate) betas_model: (f32, f32),
    pub(crate) betas_latent: (f32, f32),
    pub(crate) precondition_frequency_model: usize,
    /// Distortion weights (`dist_weight`): warm-up phases are pure MSE; the main and
    /// hard-round phases use the preset's weights.
    pub(crate) dist: Dist,
}

impl Phase {
    fn defaults(lr: f32, max_itr: usize) -> Self {
        Self {
            lr,
            max_itr,
            freq_valid: 100,
            patience: 10000,
            schedule_lr: false,
            softround_temperature: (0.3, 0.3),
            noise_parameter: (1.0, 1.0),
            noise: Noise::Kumaraswamy,
            hardround: false,
            betas_model: (0.95, 0.95),
            betas_latent: (0.9, 0.999),
            precondition_frequency_model: 10,
            dist: Dist::MSE,
        }
    }
}

/// Upstream `PresetIntra`: two warm-up stages (5 then 2 candidates), the main softround
/// phase with a cosine learning rate, then 500 hardround iterations.
#[derive(Clone, Debug)]
pub(crate) struct Preset {
    pub(crate) lambda: f32,
    /// The preset's distortion weights (main and hard-round phases; warm-up is MSE).
    pub(crate) dist: Dist,
    /// (candidates, phase) per warm-up stage.
    pub(crate) warmup: Vec<(usize, Phase)>,
    pub(crate) phases: Vec<Phase>,
}

impl Preset {
    pub(crate) fn intra(lambda: f32, start_lr: f32, n_itr: usize, dist: Dist) -> Self {
        assert!(
            n_itr >= 2000,
            "the intra preset needs at least 2000 iterations"
        );
        let init_noise = (-0.432 * libm::log10f(lambda) + 0.747) / 10.0;
        let iter_ste = 500;
        let wu_n_iter = if n_itr < 4000 {
            50
        } else if n_itr < 9000 {
            100
        } else if n_itr < 12000 {
            200
        } else {
            400
        };
        let mut core = n_itr - iter_ste;
        let mut warmup = Vec::new();
        for (stage, cand) in [5usize, 2].into_iter().enumerate() {
            let _ = stage;
            let mut ph = Phase::defaults(start_lr, wu_n_iter);
            ph.patience = wu_n_iter;
            ph.softround_temperature = (0.35, 0.35);
            ph.noise_parameter = (init_noise, init_noise);
            ph.noise = Noise::Gaussian;
            ph.betas_latent = (0.725, 0.97);
            ph.precondition_frequency_model = 1;
            warmup.push((cand, ph));
            core -= cand * wu_n_iter;
        }
        let mut main = Phase::defaults(start_lr, core);
        main.patience = 5000;
        main.schedule_lr = true;
        main.noise = Noise::Gaussian;
        main.softround_temperature = (0.35, 0.08);
        main.noise_parameter = (0.22, 0.15);
        main.dist = dist;
        let mut ste = Phase::defaults(1.0e-4, iter_ste);
        ste.schedule_lr = true;
        ste.hardround = true;
        ste.noise = Noise::None;
        ste.dist = dist;
        Self {
            lambda,
            dist,
            warmup,
            phases: vec![main, ste],
        }
    }
}

/// Upstream's encoder gain for intra frames.
pub(crate) fn encoder_gain(lambda: f32) -> f32 {
    if lambda < 0.0002 {
        24.0
    } else if lambda < 0.0005 {
        20.0
    } else {
        16.0
    }
}

fn linear_schedule(r: (f32, f32), cur: usize, max: usize) -> f32 {
    cur as f32 * (r.1 - r.0) / max as f32 + r.0
}

/// Everything the training loop needs about the picture.
pub(crate) struct Problem<'a> {
    pub(crate) shape: &'a Shape,
    /// Normalized frame data in `frame_data_type` layout: `[c][h][w]` planes for
    /// RGB/YUV444, `[y | u | v]` with half-resolution chroma for YUV420.
    pub(crate) target: &'a [f32],
    /// Plane layout of `target` and of the scored frame data.
    pub(crate) frame_data_type: crate::header::FrameDataType,
    pub(crate) bitdepth: u32,
    pub(crate) lambda: f32,
    pub(crate) gain: f32,
    /// The preset's distortion weights: everything outside a training phase (network
    /// quantization, RDOQ, the final evaluation) scores with these, like upstream's
    /// `training_preset.dist_weight`.
    pub(crate) dist: Dist,
    /// Feature extractor of the Wasserstein term; required when `dist.wass != 0`.
    pub(crate) wass: Option<&'a Wasserstein>,
}

/// Upstream `test()`: hardround latents, no noise, output rounded to the bit depth.
/// `dist` is the phase's `dist_weight` — the preset's everywhere but in warm-up.
pub(crate) fn evaluate_dist(
    pb: &Problem<'_>,
    p: &Params,
    dist: Dist,
    scratch: &mut Params,
) -> LossTerms {
    let cfg = StepCfg {
        quantizer: Quantizer::Hardround,
        gain: pb.gain,
        lambda: pb.lambda,
        eval_bitdepth: Some(pb.bitdepth),
        dist,
        frame_data_type: pb.frame_data_type,
    };
    forward_backward(pb.shape, p, pb.target, &[], cfg, pb.wass, scratch)
}

/// The evaluation pass at the preset's distortion weights.
pub(crate) fn evaluate(pb: &Problem<'_>, p: &Params, scratch: &mut Params) -> LossTerms {
    evaluate_dist(pb, p, pb.dist, scratch)
}

/// Progress callback: (iteration in phase, training loss, last validation).
pub(crate) type Progress<'a> = &'a mut dyn FnMut(usize, &LossTerms, Option<&LossTerms>);

/// Upstream `train()` for one phase; returns the best parameters seen at validation.
pub(crate) fn train(
    pb: &Problem<'_>,
    mut p: Params,
    ph: &Phase,
    rng: &mut Rng,
    progress: Progress<'_>,
    stop: &dyn enough::Stop,
) -> crate::Result<(Params, LossTerms)> {
    stop.check()?;
    let mut grads = p.zeros_like();
    let mut best = evaluate_dist(pb, &p, ph.dist, &mut grads);
    let mut best_p = p.clone();
    let layout = p.layout();
    let n_lat: usize = p.latents.iter().map(|l| l.len()).sum();
    let mut states: Vec<Option<SoapState>> = layout
        .iter()
        .map(|(shape, role)| match role {
            Role::Latent => Some(SoapState::new(shape, 0)),
            Role::Weight => Some(SoapState::new(shape, 256)),
            Role::Frozen => None,
        })
        .collect();
    let mut lr = ph.lr;
    let mut lr_steps = 0usize;
    let t_max = ph.max_itr as f32 / ph.freq_valid as f32;
    let eta_min = 1e-5f32;
    let mut temperature = linear_schedule(ph.softround_temperature, 0, ph.max_itr);
    let mut noise_param = linear_schedule(ph.noise_parameter, 0, ph.max_itr);
    let mut cnt_record = 0usize;
    let mut noise = vec![0.0f32; n_lat];
    for cnt in 0..ph.max_itr {
        stop.check()?;
        if cnt - cnt_record > ph.patience {
            if ph.schedule_lr {
                p = best_p.clone();
                cnt_record = cnt;
            } else {
                break;
            }
        }
        let noise_slice: &[f32] = match (ph.hardround, ph.noise) {
            (true, _) | (_, Noise::None) => &[],
            (false, kind) => {
                // One generator per fixed 4096-value block, seeded from the main generator:
                // parallel and independent of the thread count.
                use rayon::prelude::*;
                let seed = rng.next_u64();
                noise.par_chunks_mut(4096).enumerate().for_each(|(bi, ch)| {
                    let mut r = Rng::new(seed ^ (bi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    for v in ch.iter_mut() {
                        *v = match kind {
                            Noise::Gaussian => r.normal() * noise_param,
                            _ => super::ops::kumaraswamy(r.uniform(), noise_param),
                        };
                    }
                });
                &noise
            }
        };
        let quantizer = if ph.hardround {
            Quantizer::Hardround
        } else {
            Quantizer::Softround { temperature }
        };
        let cfg = StepCfg {
            quantizer,
            gain: pb.gain,
            lambda: pb.lambda,
            eval_bitdepth: None,
            dist: ph.dist,
            frame_data_type: pb.frame_data_type,
        };
        let terms = forward_backward(
            pb.shape,
            &p,
            pb.target,
            noise_slice,
            cfg,
            pb.wass,
            &mut grads,
        );

        // clip_grad_norm_(weights, 0.1)
        {
            let gts = grads.tensors();
            let norm2: f64 = gts
                .iter()
                .zip(&layout)
                .filter(|(_, (_, r))| *r == Role::Weight)
                .flat_map(|(g, _)| g.iter())
                .map(|&v| (v as f64) * (v as f64))
                .sum();
            let norm = libm::sqrt(norm2) as f32;
            let clip = (0.1 / (norm + 1e-6)).min(1.0);

            let mut idx = 0;
            p.for_each_mut(|t| {
                let (shape, role) = &layout[idx];
                let g = gts[idx];
                if let Some(st) = &mut states[idx] {
                    let cfg = match role {
                        Role::Latent => SoapConfig {
                            lr,
                            betas: ph.betas_latent,
                            eps: 1e-8,
                            weight_decay: 0.0,
                            precondition_frequency: 1,
                            max_precond_dim: 0,
                        },
                        _ => SoapConfig {
                            lr,
                            betas: ph.betas_model,
                            eps: 1e-8,
                            weight_decay: 0.01,
                            precondition_frequency: ph.precondition_frequency_model,
                            max_precond_dim: 256,
                        },
                    };
                    if *role == Role::Weight && clip < 1.0 {
                        let gc: Vec<f32> = g.iter().map(|&v| v * clip).collect();
                        st.step(&cfg, t, &gc);
                    } else {
                        st.step(&cfg, t, g);
                    }
                }
                let _ = shape;
                idx += 1;
            });
        }

        let mut valid = None;
        if (cnt + 1) % ph.freq_valid == 0 || cnt + 1 == ph.max_itr {
            let v = evaluate_dist(pb, &p, ph.dist, &mut grads);
            if v.loss < best.loss {
                best = v;
                best_p = p.clone();
                cnt_record = cnt;
            }
            temperature = linear_schedule(ph.softround_temperature, cnt, ph.max_itr);
            noise_param = linear_schedule(ph.noise_parameter, cnt, ph.max_itr);
            if ph.schedule_lr {
                lr_steps += 1;
                lr = eta_min
                    + (ph.lr - eta_min)
                        * (1.0 + libm::cosf(core::f32::consts::PI * lr_steps as f32 / t_max))
                        / 2.0;
            }
            valid = Some(v);
        }
        progress(cnt, &terms, valid.as_ref());
    }
    Ok((best_p, best))
}

/// Upstream `warmup()` followed by the preset's training phases.
pub(crate) fn encode_training(
    pb: &Problem<'_>,
    template: &Params,
    preset: &Preset,
    seed: u64,
    progress: Progress<'_>,
    stop: &dyn enough::Stop,
) -> crate::Result<(Params, LossTerms)> {
    stop.check()?;
    let mut rng = Rng::new(seed);
    let n0 = preset.warmup.first().map_or(1, |w| w.0);
    let mut candidates: Vec<(Params, LossTerms)> = (0..n0)
        .map(|_| {
            let mut p = template.clone();
            initialize(&mut p, pb.shape, &mut rng);
            (
                p,
                LossTerms {
                    loss: f64::INFINITY,
                    ..Default::default()
                },
            )
        })
        .collect();
    for (n, ph) in &preset.warmup {
        candidates.truncate(*n);
        for c in candidates.iter_mut() {
            let (p, l) = train(
                pb,
                core::mem::replace(&mut c.0, template.clone()),
                ph,
                &mut rng,
                progress,
                stop,
            )?;
            *c = (p, l);
        }
        candidates.sort_by(|a, b| {
            a.1.loss
                .partial_cmp(&b.1.loss)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
    }
    let (mut p, mut l) = candidates.swap_remove(0);
    for ph in &preset.phases {
        let r = train(pb, p, ph, &mut rng, progress, stop)?;
        p = r.0;
        l = r.1;
    }
    Ok((p, l))
}
