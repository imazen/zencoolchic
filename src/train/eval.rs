//! Forward-only pieces of the evaluation pass (hardround latents), for RDOQ's partial
//! scores (upstream `nnquant/rdoq.py`: `score_arm`, `score_ifce`, `score_syn_image`).
//! `tests::stages_compose_to_the_evaluation_pass` holds them equal to `ops::step`.

use alloc::vec;
use alloc::vec::Vec;

use super::arm_batch::arm_pass;
use super::ops::{conv_forward, conv_t_forward, kron, sym_conv_forward};
use super::{Mlp, Params, Shape, WINDOW, full_kernel};

const PAD: usize = (WINDOW - 1) / 2;

/// `round(gain · y)` per grid.
pub(crate) fn quantize_hard(p: &Params, gain: f32) -> Vec<Vec<f32>> {
    p.latents
        .iter()
        .map(|l| l.iter().map(|&y| libm::rintf(y * gain)).collect())
        .collect()
}

/// IFCE context of every grid, channel-major `[k][h·w]`; None without IFCE.
pub(crate) fn ifce_contexts(
    shape: &Shape,
    ifce: &[Option<Mlp>],
    q: &[Vec<f32>],
) -> Vec<Option<Vec<f32>>> {
    super::ifce::forward(shape, ifce, q).0
}

/// Total latent rate in bits under the float ARM (`inter` as [`ifce_contexts`] returns it).
pub(crate) fn latent_rate(
    shape: &Shape,
    arm: &Mlp,
    q: &[Vec<f32>],
    inter: &[Option<Vec<f32>>],
) -> f64 {
    arm_pass(shape, arm, q, inter, None)
}

/// Upsampling of the non-hyper grids: the synthesis input, `[c][h][w]`.
pub(crate) fn upsample(shape: &Shape, p: &Params, q: &[Vec<f32>]) -> Vec<f32> {
    let lat: Vec<usize> = (0..shape.grids.len())
        .filter(|&g| !shape.is_hyper[g])
        .collect();
    let n_ups = p.ups_t.len().max(1);
    let k2_t: Vec<Vec<f32>> = p
        .ups_t
        .iter()
        .map(|h| kron(&full_kernel(h, shape.ups_k)))
        .collect();
    let k2_c: Vec<Vec<f32>> = p
        .ups_c
        .iter()
        .map(|h| kron(&full_kernel(h, shape.ups_pre_k)))
        .collect();
    let last = lat[lat.len() - 1];
    let mut up = q[last].clone();
    let (mut uc, mut uh, mut uw) = (1usize, shape.grids[last].0, shape.grids[last].1);
    for (i, &t) in lat[..lat.len() - 1].iter().rev().enumerate() {
        let (th, tw) = shape.grids[t];
        let ki = i % n_ups;
        let mut next = Vec::with_capacity((uc + 1) * th * tw);
        next.extend(sym_conv_forward(&q[t], th, tw, &k2_c[ki], shape.ups_pre_k));
        for c in 0..uc {
            next.extend(conv_t_forward(
                &up[c * uh * uw..(c + 1) * uh * uw],
                uh,
                uw,
                &k2_t[ki],
                shape.ups_k,
                th,
                tw,
            ));
        }
        up = next;
        uc += 1;
        uh = th;
        uw = tw;
    }
    up
}

/// Synthesis + output transform of the latent-channel stack `up`, then
/// `rescale_output`: the decoded picture before clamping, `[c][ih][iw]` at
/// image resolution. The common-randomness planes are appended when the
/// shape has them (upstream `get_synthesis_input`).
pub(crate) fn synthesize(shape: &Shape, p: &Params, up: &[f32]) -> Vec<f32> {
    let (sh, sw) = shape.syn_size;
    let n = sh * sw;
    let mut syn_in = Vec::with_capacity(up.len() + shape.cr.as_ref().map_or(0, Vec::len));
    syn_in.extend_from_slice(up);
    if let Some(cr) = &shape.cr {
        syn_in.extend_from_slice(cr);
    }
    let mut cur = syn_in.clone();
    for l in &p.syn {
        let mut o = conv_forward(l, &cur, sh, sw);
        if l.relu {
            for v in &mut o {
                *v = v.max(0.0);
            }
        }
        cur = o;
    }
    if let Some(st) = &p.syn_stab {
        let s = conv_forward(st, &syn_in[..st.c_in * n], sh, sw);
        for (a, b) in cur.iter_mut().zip(&s) {
            *a += *b;
        }
    }
    let raw = conv_forward(&p.out_transform, &cur, sh, sw);
    rescale_output(shape, &raw)
}

/// Upstream `rescale_output`: `F.interpolate(raw, size=img_size)` when the
/// synthesis ran below image resolution; a copy otherwise.
pub(crate) fn rescale_output(shape: &Shape, raw: &[f32]) -> Vec<f32> {
    let (sh, sw) = shape.syn_size;
    let (ih, iw) = shape.img;
    if (sh, sw) == (ih, iw) {
        return raw.to_vec();
    }
    let c = raw.len() / (sh * sw);
    let mut out = vec![0.0f32; c * ih * iw];
    for ch in 0..c {
        let plane = crate::nn::Planes {
            c: 1,
            h: sh,
            w: sw,
            data: raw[ch * sh * sw..(ch + 1) * sh * sw].to_vec(),
        };
        let r = crate::nn::resize(&plane, ih, iw, shape.final_ups, crate::nn::Scale::Size);
        out[ch * ih * iw..(ch + 1) * ih * iw].copy_from_slice(&r.data);
    }
    out
}

/// `round(clamp(raw, 0, 1) · max) / max`: the bit-quantized evaluation picture.
pub(crate) fn quantized(raw: &[f32], bitdepth: u32) -> Vec<f32> {
    let m = ((1u32 << bitdepth) - 1) as f32;
    raw.iter()
        .map(|&r| libm::rintf(r.clamp(0.0, 1.0) * m) / m)
        .collect()
}

/// The frame data the distortion sees (upstream `frame.forward`'s tail):
/// `clamp(raw)` for RGB/YUV444; `clamp(convert_444_to_420(raw))` for YUV420 —
/// chroma planes are 2×2 averages at half resolution; the raw output
/// unmodified for Flow (dense outputs — motion pretraining). `out_max`
/// additionally quantizes each sample to `round(v · max) / max` (upstream
/// `test()` mode). Returns `[y | u | v]`/`[r | g | b]` concatenated planes.
pub(crate) fn decoded_dict(
    raw: &[f32],
    w: usize,
    h: usize,
    frame_data_type: crate::header::FrameDataType,
    out_max: Option<f32>,
) -> Vec<f32> {
    let n = w * h;
    let mut dec: Vec<f32> = Vec::with_capacity(match frame_data_type {
        crate::header::FrameDataType::Yuv420 => 3 * n / 2,
        crate::header::FrameDataType::Flow => raw.len(),
        _ => 3 * n,
    });
    match frame_data_type {
        crate::header::FrameDataType::Yuv420 => {
            debug_assert!(
                h.is_multiple_of(2) && w.is_multiple_of(2),
                "yuv420 needs even dims"
            );
            dec.extend(raw[..n].iter().map(|&r| r.clamp(0.0, 1.0)));
            for c in 1..3 {
                let src = &raw[c * n..(c + 1) * n];
                for by in 0..h / 2 {
                    for bx in 0..w / 2 {
                        let s = src[2 * by * w + 2 * bx]
                            + src[2 * by * w + 2 * bx + 1]
                            + src[(2 * by + 1) * w + 2 * bx]
                            + src[(2 * by + 1) * w + 2 * bx + 1];
                        dec.push((s / 4.0).clamp(0.0, 1.0));
                    }
                }
            }
        }
        crate::header::FrameDataType::Flow => dec.extend_from_slice(raw),
        _ => dec.extend(raw.iter().map(|&r| r.clamp(0.0, 1.0))),
    }
    if let Some(m) = out_max {
        for v in &mut dec {
            *v = libm::rintf(*v * m) / m;
        }
    }
    dec
}

/// MSE of two same-length planes (e.g. the [`quantized`] picture against the target).
pub(crate) fn distortion_of(a: &[f32], target: &[f32]) -> f64 {
    a.iter()
        .zip(target)
        .map(|(&x, &t)| ((x - t) * (x - t)) as f64)
        .sum::<f64>()
        / a.len() as f64
}

/// MSE of the clamped picture rounded to `bitdepth`.
pub(crate) fn distortion(raw: &[f32], target: &[f32], bitdepth: u32) -> f64 {
    distortion_of(&quantized(raw, bitdepth), target)
}

#[cfg(test)]
mod tests {
    use super::super::trainer::{Problem, Rng, evaluate, initialize};
    use super::super::{Shape, config};
    use super::*;

    #[test]
    fn stages_compose_to_the_evaluation_pass() {
        let hdr = config::header(config::Arch::Medium, 37, 53);
        let a = crate::decode::arch(&hdr).unwrap();
        let shape = Shape::new(&hdr, &a).unwrap();
        let mut p = shape.zero_params(&hdr, &a);
        let mut rng = Rng::new(5);
        initialize(&mut p, &shape, &mut rng);
        // Non-trivial latents and weights.
        p.for_each_mut(|t| {
            for v in t.iter_mut() {
                *v += rng.normal() * 0.1;
            }
        });
        let target: Vec<f32> = (0..3 * 37 * 53)
            .map(|i| ((i * 7919) % 256) as f32 / 255.0)
            .collect();
        let pb = Problem {
            shape: &shape,
            target: &target,
            frame_data_type: crate::header::FrameDataType::Rgb,
            bitdepth: 8,
            lambda: 1e-3,
            gain: 16.0,
            dist: crate::train::Dist::MSE,
            wass: None,
            inter: None,
            out_init: None,
        };
        let mut scratch = p.zeros_like();
        let want = evaluate(&pb, &p, &mut scratch);
        let q = quantize_hard(&p, 16.0);
        let inter = ifce_contexts(&shape, &p.ifce, &q);
        let rate = latent_rate(&shape, &p.arm, &q, &inter);
        let mse = distortion(
            &synthesize(&shape, &p, &upsample(&shape, &p, &q)),
            &target,
            8,
        );
        assert_eq!(rate, want.rate_bits);
        assert_eq!(mse, want.mse);
    }

    /// A tiny B frame through `train_pair`: the joint objective, clip and
    /// best-model bookkeeping run end to end on real models.
    #[test]
    fn pair_trains_b_frame_jointly() {
        use super::super::trainer::{FramePair, Phase, train_pair};
        let mk = |outs: u32| {
            let mut hdr = config::header(config::Arch::Low, 24, 32);
            let n = hdr.layers_synthesis.len();
            for (i, l) in hdr.layers_synthesis.iter_mut().enumerate() {
                if i + 2 >= n {
                    l.out_ft = outs;
                }
            }
            let a = crate::decode::arch(&hdr).unwrap();
            (hdr, a)
        };
        let (h_r, a_r) = mk(5);
        let (h_m, a_m) = mk(4);
        let s_r = Shape::new(&h_r, &a_r).unwrap();
        let s_m = Shape::new(&h_m, &a_m).unwrap();
        let mut rng = Rng::new(7);
        let (mut p_r, mut p_m) = (s_r.zero_params(&h_r, &a_r), s_m.zero_params(&h_m, &a_m));
        initialize(&mut p_r, &s_r, &mut rng);
        initialize(&mut p_m, &s_m, &mut rng);
        let target: Vec<f32> = (0..3 * 24 * 32)
            .map(|i| ((i * 31) % 256) as f32 / 255.0)
            .collect();
        let mkref = |seed: u32| crate::nn::Planes {
            c: 3,
            h: 24,
            w: 32,
            data: (0..3 * 24 * 32)
                .map(|i| (((i * 17 + seed as usize) % 251) as f32) / 255.0)
                .collect(),
        };
        let shared = crate::train::inter::SharedInter {
            refs: alloc::vec::Vec::from([mkref(3), mkref(91)]),
            global_flow: alloc::vec::Vec::from([(1, -1), (-2, 0)]),
            warp_filter: 4,
            frame_type: crate::header::FrameType::B,
        };
        fn pb<'a>(s: &'a Shape, target: &'a [f32]) -> Problem<'a> {
            Problem {
                shape: s,
                target,
                frame_data_type: crate::header::FrameDataType::Rgb,
                bitdepth: 8,
                lambda: 1e-3,
                gain: 16.0,
                dist: crate::train::Dist::MSE,
                wass: None,
                inter: None,
                out_init: None,
            }
        }
        let (p_r2, p_m2, best) = train_pair(
            FramePair {
                res: pb(&s_r, &target),
                mot: pb(&s_m, &target),
                shared: &shared,
                p_r,
                p_m,
            },
            &Phase {
                lr: 3e-3,
                max_itr: 4,
                freq_valid: 2,
                patience: 100,
                schedule_lr: false,
                softround_temperature: (0.3, 0.3),
                noise_parameter: (1.0, 1.0),
                noise: super::super::trainer::Noise::Kumaraswamy,
                hardround: false,
                betas_model: (0.95, 0.95),
                betas_latent: (0.9, 0.999),
                precondition_frequency_model: 10,
                dist: crate::train::Dist::MSE,
            },
            &mut rng,
            &mut |_, _, _| {},
            &enough::Unstoppable,
        )
        .unwrap();
        assert!(best.loss.is_finite() && best.mse.is_finite());
        assert!(
            p_r2.tensors()
                .iter()
                .all(|t| t.iter().all(|v| v.is_finite()))
                && p_m2
                    .tensors()
                    .iter()
                    .all(|t| t.iter().all(|v| v.is_finite()))
        );
    }

    /// Upstream's candidate loop + `warmup()` for a pair: fresh models, the
    /// pretrained-motion injection on odd candidates, selection by joint loss.
    #[test]
    fn encode_training_pair_warmup_smoke() {
        use super::super::trainer::{Preset, encode_training_pair};
        let mk = |outs: u32| {
            let mut hdr = config::header(config::Arch::Low, 24, 32);
            let n = hdr.layers_synthesis.len();
            for (i, l) in hdr.layers_synthesis.iter_mut().enumerate() {
                if i + 2 >= n {
                    l.out_ft = outs;
                }
            }
            let a = crate::decode::arch(&hdr).unwrap();
            (hdr, a)
        };
        let (h_r, a_r) = mk(5);
        let (h_m, a_m) = mk(4);
        let s_r = Shape::new(&h_r, &a_r).unwrap();
        let s_m = Shape::new(&h_m, &a_m).unwrap();
        let t_r = s_r.zero_params(&h_r, &a_r);
        let t_m = s_m.zero_params(&h_m, &a_m);
        // The "pre-trained" motion model: any valid params exercise the
        // injection path (upstream ships them on odd candidates).
        let mut pre_m = s_m.zero_params(&h_m, &a_m);
        let mut rng = Rng::new(5);
        initialize(&mut pre_m, &s_m, &mut rng);
        for l in &mut pre_m.latents {
            l.fill(0.25);
        }
        let target: Vec<f32> = (0..3 * 24 * 32)
            .map(|i| ((i * 31) % 256) as f32 / 255.0)
            .collect();
        let mkref = |seed: u32| crate::nn::Planes {
            c: 3,
            h: 24,
            w: 32,
            data: (0..3 * 24 * 32)
                .map(|i| (((i * 17 + seed as usize) % 251) as f32) / 255.0)
                .collect(),
        };
        let shared = crate::train::inter::SharedInter {
            refs: alloc::vec::Vec::from([mkref(3), mkref(91)]),
            global_flow: alloc::vec::Vec::from([(1, -1), (-2, 0)]),
            warp_filter: 4,
            frame_type: crate::header::FrameType::B,
        };
        fn pb<'a>(s: &'a Shape, target: &'a [f32]) -> Problem<'a> {
            Problem {
                shape: s,
                target,
                frame_data_type: crate::header::FrameDataType::Rgb,
                bitdepth: 8,
                lambda: 1e-3,
                gain: 16.0,
                dist: crate::train::Dist::MSE,
                wass: None,
                inter: None,
                out_init: None,
            }
        }
        let pb_r = pb(&s_r, &target);
        let pb_m = pb(&s_m, &target);
        let mut preset = Preset::inter(1e-3, 3e-3, 8, 4, crate::train::Dist::MSE);
        for (_, ph) in preset.warmup.iter_mut() {
            ph.max_itr = 3;
            ph.freq_valid = 3;
        }
        for ph in preset.phases.iter_mut() {
            ph.max_itr = 4;
            ph.freq_valid = 2;
        }
        let (r, m, l) = encode_training_pair(
            &pb_r,
            &pb_m,
            &shared,
            &t_r,
            &t_m,
            &preset,
            Some((&pre_m, 1.0)),
            11,
            &mut |_, _, _| {},
            &enough::Unstoppable,
        )
        .unwrap();
        assert!(l.loss.is_finite() && l.mse.is_finite());
        for p in [&r, &m] {
            assert!(p.tensors().iter().all(|t| t.iter().all(|v| v.is_finite())));
        }
    }

    /// Upstream `init_output_transform(img_min_max)`: diagonal weight
    /// `max - min`, bias `min`, frozen afterwards (`Role::Frozen`).
    #[test]
    fn output_transform_affine_init() {
        let hdr = config::header(config::Arch::Low, 24, 32);
        let a = crate::decode::arch(&hdr).unwrap();
        let s = Shape::new(&hdr, &a).unwrap();
        let mut p = s.zero_params(&hdr, &a);
        let mut rng = Rng::new(5);
        initialize(&mut p, &s, &mut rng);
        super::super::trainer::init_output_transform(&mut p, &[(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]);
        let ot = &p.out_transform;
        for c in 0..3 {
            let diff = [0.9f32, 0.8, 0.7][c] - [0.1, 0.2, 0.3][c];
            for i in 0..ot.c_in {
                let want = if i == c { diff } else { 0.0 };
                assert_eq!(ot.w[c * ot.c_in + i], want);
            }
            assert_eq!(ot.b[c], [0.1, 0.2, 0.3][c]);
        }
        // Frozen: the tensor layout marks it Role::Frozen so SOAP skips it.
        let frozen = p
            .layout()
            .iter()
            .filter(|(_, r)| *r == super::super::Role::Frozen)
            .count();
        assert_eq!(frozen, 2); // out_transform weight + bias
    }

    #[test]
    fn training_step_is_independent_of_thread_count() {
        use super::super::{Quantizer, StepCfg, forward_backward, ops::StepAux};
        let hdr = config::header(config::Arch::High, 70, 90);
        let a = crate::decode::arch(&hdr).unwrap();
        let shape = Shape::new(&hdr, &a).unwrap();
        let mut p = shape.zero_params(&hdr, &a);
        let mut rng = Rng::new(9);
        initialize(&mut p, &shape, &mut rng);
        p.for_each_mut(|t| {
            for v in t.iter_mut() {
                *v += rng.normal() * 0.1;
            }
        });
        let target: Vec<f32> = (0..3 * 70 * 90)
            .map(|i| ((i * 7919) % 256) as f32 / 255.0)
            .collect();
        let n_lat: usize = p.latents.iter().map(|l| l.len()).sum();
        let noise: Vec<f32> = (0..n_lat).map(|_| rng.normal() * 0.2).collect();
        let cfg = StepCfg {
            quantizer: Quantizer::Softround { temperature: 0.3 },
            gain: 16.0,
            lambda: 1e-3,
            eval_bitdepth: None,
            dist: crate::train::Dist::MSE,
            frame_data_type: crate::header::FrameDataType::Rgb,
        };
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut g = p.zeros_like();
                let t =
                    forward_backward(&shape, &p, &target, &noise, cfg, StepAux::default(), &mut g)
                        .terms;
                let bits: Vec<u32> = g
                    .tensors()
                    .iter()
                    .flat_map(|t| t.iter().map(|v| v.to_bits()))
                    .collect();
                (t.loss.to_bits(), bits)
            })
        };
        assert_eq!(run(1), run(4));
    }
}
