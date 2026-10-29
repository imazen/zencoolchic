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

/// Upstream `PresetIntra`/`PresetInter`: warm-up stages, the training phases
/// and the optional motion pre-training phase.
#[derive(Clone, Debug)]
pub(crate) struct Preset {
    pub(crate) lambda: f32,
    /// The preset's distortion weights (main and hard-round phases; warm-up is MSE).
    pub(crate) dist: Dist,
    /// (candidates, phase) per warm-up stage.
    pub(crate) warmup: Vec<(usize, Phase)>,
    pub(crate) phases: Vec<Phase>,
    /// Upstream `motion_pretrain_phase`: one fixed-lr Kumaraswamy phase
    /// (`Some` only on the inter preset). The problem carrying it sets
    /// `lambda = 20 · preset.lambda` like upstream's `lmbda=20 * self.lmbda`.
    pub(crate) motion_pretrain: Option<Phase>,
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
            motion_pretrain: None,
        }
    }

    /// Upstream `PresetInter`: one warm-up stage (2 candidates, 600 iterations),
    /// a single softround main phase, plus the motion pre-training phase.
    /// Upstream requires `n_itr >= 2000` only for the intra preset; inter frames
    /// have no warmup candidates below `n_candidates` though, so `n_itr` stays
    /// free like upstream.
    pub(crate) fn inter(
        lambda: f32,
        start_lr: f32,
        n_itr: usize,
        itr_motion: usize,
        dist: Dist,
    ) -> Self {
        // Warm-up: 2 candidates, constant lr, strong noise.
        let mut wu = Phase::defaults(start_lr, 600);
        wu.freq_valid = 600;
        wu.patience = 100000;
        wu.noise_parameter = (2.0, 2.0);
        wu.betas_latent = (0.725, 0.97);
        wu.precondition_frequency_model = 1;
        let mut main = Phase::defaults(start_lr, n_itr);
        main.patience = 5000;
        main.schedule_lr = true;
        main.noise = Noise::Gaussian;
        main.softround_temperature = (0.3, 0.1);
        main.noise_parameter = (0.25, 0.1);
        main.dist = dist;
        let mut motion = Phase::defaults(1.0e-2, itr_motion);
        motion.patience = itr_motion;
        motion.noise_parameter = (2.0, 2.0);
        Self {
            lambda,
            dist,
            warmup: vec![(2, wu)],
            phases: vec![main],
            motion_pretrain: Some(motion),
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
#[derive(Clone)]
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
    /// Reference/motion context for a P/B frame objective (`None` for intra).
    pub(crate) inter: Option<super::inter::InterCtx<'a>>,
    /// Upstream `img_min_max`: per-channel `(min, max)` of the frame data,
    /// applied by `init_output_transform` to every warm-up candidate's
    /// residue synthesis. `Some` only for I-frames with YUV data — upstream
    /// leaves it `None` for RGB and for all inter frames.
    pub(crate) out_init: Option<[(f32, f32); 3]>,
}

/// Upstream `Synthesis.init_output_transform(img_min_max)`: a frozen diagonal
/// affine mapping unit-range synthesis output into the frame's per-channel
/// range (identity init plus `w[c][c] = max - min`, `b[c] = min`).
pub(crate) fn init_output_transform(p: &mut Params, mm: &[(f32, f32); 3]) {
    let ot = &mut p.out_transform;
    for (c, &(mn, mx)) in mm.iter().enumerate().take(ot.c_out) {
        ot.w[c * ot.c_in + c] = mx - mn;
        ot.b[c] = mn;
    }
}

impl Problem<'_> {
    /// The `StepAux` matching this problem's optional contexts.
    pub(crate) fn aux(&self) -> super::ops::StepAux<'_> {
        super::ops::StepAux {
            wass: self.wass,
            inter: self.inter,
            ..Default::default()
        }
    }
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
    forward_backward(pb.shape, p, pb.target, &[], cfg, pb.aux(), scratch).terms
}

/// The evaluation pass at the preset's distortion weights.
pub(crate) fn evaluate(pb: &Problem<'_>, p: &Params, scratch: &mut Params) -> LossTerms {
    evaluate_dist(pb, p, pb.dist, scratch)
}

/// Progress callback: (iteration in phase, training loss, last validation).
pub(crate) type Progress<'a> = &'a mut dyn FnMut(usize, &LossTerms, Option<&LossTerms>);

/// One model's optimizer state inside `train`: parameters, gradient buffer,
/// tensor layout and per-tensor SOAP state (upstream's single `optimizer`
/// holding the joint parameter list, decomposed per model here).
struct Opt {
    p: Params,
    grads: Params,
    states: Vec<Option<SoapState>>,
    layout: Vec<(Vec<usize>, Role)>,
}

impl Opt {
    fn new(p: Params) -> Self {
        let layout = p.layout();
        let states = layout
            .iter()
            .map(|(shape, role)| match role {
                Role::Latent => Some(SoapState::new(shape, 0)),
                Role::Weight => Some(SoapState::new(shape, 256)),
                Role::Frozen => None,
            })
            .collect();
        let grads = p.zeros_like();
        Opt {
            p,
            grads,
            states,
            layout,
        }
    }

    /// Σ grads² over `Role::Weight` tensors — one term of the joint norm
    /// `clip_grad_norm_` computes on upstream's combined parameter list.
    fn weight_norm2(&self) -> f64 {
        self.grads
            .tensors()
            .iter()
            .zip(&self.layout)
            .filter(|(_, (_, r))| *r == Role::Weight)
            .flat_map(|(g, _)| g.iter())
            .map(|&v| (v as f64) * (v as f64))
            .sum()
    }

    /// SOAP update on every trainable tensor; weights receive the jointly
    /// clipped gradient (`clip = min(1, 0.1 / total_norm)` upstream).
    fn apply(&mut self, clip: f32, lr: f32, ph: &Phase) {
        let mut idx = 0;
        let gts = self.grads.tensors();
        self.p.for_each_mut(|t| {
            let (shape, role) = &self.layout[idx];
            let g = gts[idx];
            if let Some(st) = &mut self.states[idx] {
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
}

/// Upstream `train()` for one phase; returns the best parameters seen at validation.
pub(crate) fn train(
    pb: &Problem<'_>,
    p: Params,
    ph: &Phase,
    rng: &mut Rng,
    progress: Progress<'_>,
    stop: &dyn enough::Stop,
) -> crate::Result<(Params, LossTerms)> {
    stop.check()?;
    let mut opt = Opt::new(p);
    let mut best = evaluate_dist(pb, &opt.p, ph.dist, &mut opt.grads);
    let mut best_p = opt.p.clone();
    let n_lat: usize = opt.p.latents.iter().map(|l| l.len()).sum();
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
                opt.p = best_p.clone();
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
            &opt.p,
            pb.target,
            noise_slice,
            cfg,
            pb.aux(),
            &mut opt.grads,
        )
        .terms;

        // clip_grad_norm_(weights, 0.1) on the (here single-model) weight list.
        {
            let norm = libm::sqrt(opt.weight_norm2()) as f32;
            let clip = (0.1 / (norm + 1e-6)).min(1.0);
            opt.apply(clip, lr, ph);
        }

        let mut valid = None;
        if (cnt + 1) % ph.freq_valid == 0 || cnt + 1 == ph.max_itr {
            let v = evaluate_dist(pb, &opt.p, ph.dist, &mut opt.grads);
            if v.loss < best.loss {
                best = v;
                best_p = opt.p.clone();
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

/// A P/B frame's joint training problem: both models, the shared inter
/// context and their current parameters.
pub(crate) struct FramePair<'a> {
    /// Residue model's problem (4-channel output for P, 5 for B).
    pub(crate) res: Problem<'a>,
    /// Motion model's problem (2-channel output for P, 4 for B).
    pub(crate) mot: Problem<'a>,
    /// References + global translations + warper parameters.
    pub(crate) shared: &'a super::inter::SharedInter,
    /// Current residue parameters.
    pub(crate) p_r: Params,
    /// Current motion parameters.
    pub(crate) p_m: Params,
}

/// Upstream `train()` for a P/B frame: the residue and motion Cool-chics are
/// one joint problem (upstream's single optimizer over both parameter
/// lists). Validation and patience use the joint loss `dist + λ·(rate_r +
/// rate_m)/pixels` — each model's eval reports `dist + λ·rate_own`, so the
/// joint is their sum minus the shared distortion term.
pub(crate) fn train_pair(
    pair: FramePair<'_>,
    ph: &Phase,
    rng: &mut Rng,
    progress: Progress<'_>,
    stop: &dyn enough::Stop,
) -> crate::Result<(Params, Params, LossTerms)> {
    stop.check()?;
    let FramePair {
        res: pb_r,
        mot: pb_m,
        shared,
        p_r,
        p_m,
    } = pair;
    let pb_r = &pb_r;
    let pb_m = &pb_m;
    let mut opt_r = Opt::new(p_r);
    let mut opt_m = Opt::new(p_m);
    // Joint eval loss: shared distortion + both models' rate terms.
    let joint = |a: &LossTerms, b: &LossTerms| -> LossTerms {
        LossTerms {
            loss: a.loss + b.loss - a.dist,
            ..*a
        }
    };
    // Joint eval: `pair_eval` composites the shared picture from both
    // models' eval-mode raws; the joint loss adds both rate terms to the
    // shared distortion (upstream `test()` + `loss_function` on the pair).
    let eval_pair = |o_r: &mut Opt, o_m: &mut Opt| -> LossTerms {
        let cfg = |pb: &Problem<'_>| StepCfg {
            quantizer: Quantizer::Hardround,
            gain: pb.gain,
            lambda: pb.lambda,
            eval_bitdepth: Some(pb.bitdepth),
            dist: ph.dist,
            frame_data_type: pb.frame_data_type,
        };
        let mut scratch_r = o_r.p.zeros_like();
        let mut scratch_m = o_m.p.zeros_like();
        let (a, b) = super::inter::pair_eval(
            shared,
            super::inter::PairIn {
                shape: pb_r.shape,
                p: &o_r.p,
                target: pb_r.target,
                noise: &[],
                cfg: cfg(pb_r),
                wass: pb_r.wass,
                scratch: &mut scratch_r,
                grads: &mut o_r.grads,
            },
            super::inter::PairIn {
                shape: pb_m.shape,
                p: &o_m.p,
                target: pb_m.target,
                noise: &[],
                cfg: cfg(pb_m),
                wass: pb_m.wass,
                scratch: &mut scratch_m,
                grads: &mut o_m.grads,
            },
        );
        joint(&a.terms, &b.terms)
    };
    let mut best = eval_pair(&mut opt_r, &mut opt_m);
    let mut best_p = (opt_r.p.clone(), opt_m.p.clone());
    let n_lat_r: usize = opt_r.p.latents.iter().map(|l| l.len()).sum();
    let n_lat_m: usize = opt_m.p.latents.iter().map(|l| l.len()).sum();
    let mut lr = ph.lr;
    let mut lr_steps = 0usize;
    let t_max = ph.max_itr as f32 / ph.freq_valid as f32;
    let eta_min = 1e-5f32;
    let mut temperature = linear_schedule(ph.softround_temperature, 0, ph.max_itr);
    let mut noise_param = linear_schedule(ph.noise_parameter, 0, ph.max_itr);
    let mut cnt_record = 0usize;
    let mut noise_r = vec![0.0f32; n_lat_r];
    let mut noise_m = vec![0.0f32; n_lat_m];
    for cnt in 0..ph.max_itr {
        stop.check()?;
        if cnt - cnt_record > ph.patience {
            if ph.schedule_lr {
                (opt_r.p, opt_m.p) = best_p.clone();
                cnt_record = cnt;
            } else {
                break;
            }
        }
        // Per-model noise draws in upstream's forward order: residue then
        // motion (each `generate_kumaraswamy_noise` inside its cc forward).
        let mut fill = |noise: &mut [f32]| match (ph.hardround, ph.noise) {
            (true, _) | (_, Noise::None) => {}
            (false, kind) => {
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
            }
        };
        fill(&mut noise_r);
        fill(&mut noise_m);
        let quantizer = if ph.hardround {
            Quantizer::Hardround
        } else {
            Quantizer::Softround { temperature }
        };
        let cfg = |pb: &Problem<'_>| StepCfg {
            quantizer,
            gain: pb.gain,
            lambda: pb.lambda,
            eval_bitdepth: None,
            dist: ph.dist,
            frame_data_type: pb.frame_data_type,
        };
        let mut scratch_r = opt_r.p.zeros_like();
        let mut scratch_m = opt_m.p.zeros_like();
        let (out_r, out_m) = super::inter::pair_step(
            shared,
            super::inter::PairIn {
                shape: pb_r.shape,
                p: &opt_r.p,
                target: pb_r.target,
                noise: &noise_r,
                cfg: cfg(pb_r),
                wass: pb_r.wass,
                scratch: &mut scratch_r,
                grads: &mut opt_r.grads,
            },
            super::inter::PairIn {
                shape: pb_m.shape,
                p: &opt_m.p,
                target: pb_m.target,
                noise: &noise_m,
                cfg: cfg(pb_m),
                wass: pb_m.wass,
                scratch: &mut scratch_m,
                grads: &mut opt_m.grads,
            },
        );
        let terms = joint(&out_r.terms, &out_m.terms);

        // clip_grad_norm_ over the joint weight-parameter list (upstream's
        // single optimizer sees both models).
        let norm = libm::sqrt(opt_r.weight_norm2() + opt_m.weight_norm2()) as f32;
        let clip = (0.1 / (norm + 1e-6)).min(1.0);
        opt_r.apply(clip, lr, ph);
        opt_m.apply(clip, lr, ph);

        let mut valid = None;
        if (cnt + 1) % ph.freq_valid == 0 || cnt + 1 == ph.max_itr {
            let v = eval_pair(&mut opt_r, &mut opt_m);
            if v.loss < best.loss {
                best = v;
                best_p = (opt_r.p.clone(), opt_m.p.clone());
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
    Ok((best_p.0, best_p.1, best))
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
            if let Some(mm) = pb.out_init {
                init_output_transform(&mut p, &mm);
            }
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

/// Upstream's candidate construction + `warmup()` + the preset's training
/// phases for a P/B frame. `pretrained_motion` carries the optional
/// flow-guided initialization (`guided_motion_pretraining`): it replaces the
/// motion parameters of every odd-numbered candidate, its latent grids
/// pre-scaled by `gain_ratio` (upstream's `motion.encoder_gain /
/// pre_trained.encoder_gain` — 1 when both use the default 16).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_training_pair(
    res: &Problem<'_>,
    mot: &Problem<'_>,
    shared: &super::inter::SharedInter,
    t_r: &Params,
    t_m: &Params,
    preset: &Preset,
    pretrained_motion: Option<(&Params, f32)>,
    seed: u64,
    progress: Progress<'_>,
    stop: &dyn enough::Stop,
) -> crate::Result<(Params, Params, LossTerms)> {
    stop.check()?;
    let mut rng = Rng::new(seed);
    let n0 = preset.warmup.first().map_or(1, |w| w.0);
    let mut candidates: Vec<(Params, Params, LossTerms)> = (0..n0)
        .map(|i| {
            let mut r = t_r.clone();
            initialize(&mut r, res.shape, &mut rng);
            let mut m = t_m.clone();
            initialize(&mut m, mot.shape, &mut rng);
            // Upstream loads the pre-trained motion only on odd candidates.
            if i % 2 == 1
                && let Some((pm, ratio)) = pretrained_motion
            {
                m = pm.clone();
                if ratio != 1.0 {
                    for l in &mut m.latents {
                        for v in l.iter_mut() {
                            *v *= ratio;
                        }
                    }
                }
            }
            (
                r,
                m,
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
            let (r, m, l) = train_pair(
                FramePair {
                    res: res.clone(),
                    mot: mot.clone(),
                    shared,
                    p_r: core::mem::replace(&mut c.0, t_r.clone()),
                    p_m: core::mem::replace(&mut c.1, t_m.clone()),
                },
                ph,
                &mut rng,
                progress,
                stop,
            )?;
            *c = (r, m, l);
        }
        candidates.sort_by(|a, b| {
            a.2.loss
                .partial_cmp(&b.2.loss)
                .unwrap_or(core::cmp::Ordering::Equal)
        });
    }
    let (mut r, mut m, mut l) = candidates.swap_remove(0);
    for ph in &preset.phases {
        let (nr, nm, nl) = train_pair(
            FramePair {
                res: res.clone(),
                mot: mot.clone(),
                shared,
                p_r: r,
                p_m: m,
            },
            ph,
            &mut rng,
            progress,
            stop,
        )?;
        r = nr;
        m = nm;
        l = nl;
    }
    Ok((r, m, l))
}
