//! Forward-only pieces of the evaluation pass (hardround latents), for RDOQ's partial
//! scores (upstream `nnquant/rdoq.py`: `score_arm`, `score_ifce`, `score_syn_image`).
//! `tests::stages_compose_to_the_evaluation_pass` holds them equal to `ops::step`.

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

/// Synthesis + output transform of the latent-channel stack `up`: the decoded picture
/// before clamping, `[c][h][w]`. The common-randomness planes are appended when the
/// shape has them (upstream `get_synthesis_input`).
pub(crate) fn synthesize(shape: &Shape, p: &Params, up: &[f32]) -> Vec<f32> {
    let (ih, iw) = shape.img;
    let n = ih * iw;
    let mut syn_in = Vec::with_capacity(up.len() + shape.cr.as_ref().map_or(0, Vec::len));
    syn_in.extend_from_slice(up);
    if let Some(cr) = &shape.cr {
        syn_in.extend_from_slice(cr);
    }
    let mut cur = syn_in.clone();
    for l in &p.syn {
        let mut o = conv_forward(l, &cur, ih, iw);
        if l.relu {
            for v in &mut o {
                *v = v.max(0.0);
            }
        }
        cur = o;
    }
    if let Some(st) = &p.syn_stab {
        let s = conv_forward(st, &syn_in[..st.c_in * n], ih, iw);
        for (a, b) in cur.iter_mut().zip(&s) {
            *a += *b;
        }
    }
    conv_forward(&p.out_transform, &cur, ih, iw)
}

/// `round(clamp(raw, 0, 1) · max) / max`: the bit-quantized evaluation picture.
pub(crate) fn quantized(raw: &[f32], bitdepth: u32) -> Vec<f32> {
    let m = ((1u32 << bitdepth) - 1) as f32;
    raw.iter()
        .map(|&r| libm::rintf(r.clamp(0.0, 1.0) * m) / m)
        .collect()
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
            bitdepth: 8,
            lambda: 1e-3,
            gain: 16.0,
            dist: crate::train::Dist::MSE,
            wass: None,
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

    #[test]
    fn training_step_is_independent_of_thread_count() {
        use super::super::{Quantizer, StepCfg, forward_backward};
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
        };
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut g = p.zeros_like();
                let t = forward_backward(&shape, &p, &target, &noise, cfg, None, &mut g);
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
