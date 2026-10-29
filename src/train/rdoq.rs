//! Rate-distortion optimized quantization of the network parameters (upstream
//! `nnquant/rdoq.py`).
//!
//! For every module (sorted by name) and for its weights, then its biases, parameters are
//! visited in random order; each is moved by `±1..15` quantization steps (one sign at a
//! time, stopping at the first shift that does not lower the score) and kept at the best
//! value. Scores are local, as upstream's:
//! - ARM: latent rate + the ARM's parameter rate (bits);
//! - IFCE: latent rate with the new IFCE contexts + the IFCE's parameter rate (bits);
//! - synthesis: MSE (8-bit output) + λ · synthesis parameter rate / pixels;
//! - upsampling: the full evaluation loss with every module's parameter rate.

use alloc::vec::Vec;

use super::eval;
use super::eval::SynSel;
use super::quantize::{Module, NnQuant, module_values_mut, network_rate, nn_rate_of, step_of};
use super::trainer::{Problem, Rng};
use super::{ConvLayer, Params, Shape};

/// Weighted distortion of the bitdepth-quantized frame data, like upstream's
/// `loss_function` on the eval image (`score_syn_image`).
fn dist_quant(pb: &Problem<'_>, raw: &[f32]) -> f64 {
    let (ih, iw) = pb.shape.img;
    // P/B frames score the decoder's composited picture (quantized flow).
    let img;
    let raw = match pb.inter {
        Some(i) => {
            img = super::inter::composite(&i, raw, iw, ih, true)
                .expect("inter frame compositing")
                .0;
            &img
        }
        None => raw,
    };
    let dec = eval::decoded_dict(
        raw,
        iw,
        ih,
        pb.frame_data_type,
        Some(((1u32 << pb.bitdepth) - 1) as f32),
    );
    let mut d = pb.dist.mse as f64 * eval::distortion_of(&dec, pb.target);
    if pb.dist.wass != 0.0 {
        let w = pb.wass.expect("wasserstein weight without feature weights");
        d += pb.dist.wass as f64 * w.eval(&dec, ih, iw);
    }
    d
}

fn score(pb: &Problem<'_>, p: &Params, q: &NnQuant, m: Module, fixed: &Fixed) -> f64 {
    let shape: &Shape = pb.shape;
    let n_pix = (shape.img.0 * shape.img.1) as f64;
    match m {
        Module::Arm => {
            eval::latent_rate(shape, &p.arm, &fixed.q, &fixed.inter) + nn_rate_of(p, q, m) as f64
        }
        Module::Ifce => {
            let inter = eval::ifce_contexts(shape, &p.ifce, &fixed.q);
            eval::latent_rate(shape, &p.arm, &fixed.q, &inter) + nn_rate_of(p, q, m) as f64
        }
        Module::Synthesis => {
            let raw = eval::synthesize(shape, p, &fixed.syn_in);
            dist_quant(pb, &raw) + pb.lambda as f64 * nn_rate_of(p, q, m) as f64 / n_pix
        }
        Module::Upsampling => {
            let syn_in = eval::upsample(shape, p, &fixed.q);
            let raw = eval::synthesize(shape, p, &syn_in);
            let rate = eval::latent_rate(shape, &p.arm, &fixed.q, &fixed.inter);
            dist_quant(pb, &raw) + pb.lambda as f64 * (rate + network_rate(p, q) as f64) / n_pix
        }
    }
}

/// Quantities that do not depend on the module being optimized.
struct Fixed {
    q: Vec<Vec<f32>>,
    inter: Vec<Option<Vec<f32>>>,
    syn_in: Vec<f32>,
}

/// Flat `module_values_mut` index → (tensor, output channel) for the synthesis
/// module, in `module_tensors_mut` order (output transform, optional
/// stabiliser, then the layers).
fn syn_flat_map(p: &Params, bias: bool) -> Vec<(SynSel, usize)> {
    let mut convs: Vec<(&ConvLayer, SynSel)> = Vec::with_capacity(p.syn.len() + 2);
    convs.push((&p.out_transform, SynSel::OutTransform));
    if let Some(st) = &p.syn_stab {
        convs.push((st, SynSel::Stabiliser));
    }
    convs.extend(p.syn.iter().enumerate().map(|(l, c)| (c, SynSel::Layer(l))));
    let mut map = Vec::new();
    for (c, sel) in convs {
        if bias {
            map.extend((0..c.b.len()).map(|j| (sel, j)));
        } else {
            let ch = c.c_in * c.k * c.k;
            map.extend((0..c.w.len()).map(|j| (sel, j / ch)));
        }
    }
    map
}

/// Runs RDOQ in place. `log` gets one line per (module, weight|bias).
pub(crate) fn rdoq(
    pb: &Problem<'_>,
    p: &mut Params,
    q: &NnQuant,
    rng: &mut Rng,
    log: &mut dyn FnMut(alloc::string::String),
    stop: &dyn enough::Stop,
) -> crate::Result<()> {
    let shape = pb.shape;
    let n_pix = shape.img.0 * shape.img.1;
    let qlat = eval::quantize_hard(p, pb.gain);
    for m in Module::SORTED {
        if q.shift[m.slot()].is_none() {
            continue;
        }
        // Recomputed per module: earlier modules may have changed.
        let fixed = Fixed {
            inter: eval::ifce_contexts(shape, &p.ifce, &qlat),
            syn_in: eval::upsample(shape, p, &qlat),
            q: qlat.clone(),
        };
        // Synthesis candidates score through the incremental forward: a
        // parameter move touches one output channel of one tensor, so the
        // accepted prefix outputs and stabiliser are reused. Bit-identical to
        // the full `synthesize` (channel accumulation is independent).
        let mut inc =
            (m == Module::Synthesis).then(|| eval::SynInc::new(shape, p, &fixed.syn_in));
        for is_bias in [false, true] {
            let step = step_of(q, m, is_bias);
            let n = module_values_mut(p, m, is_bias).len();
            if n == 0 {
                continue;
            }
            let syn_map = inc.is_some().then(|| syn_flat_map(p, is_bias));
            let mut best = super::pool(n_pix).install(|| score(pb, p, q, m, &fixed));
            let start = best;
            // Random visiting order (upstream torch.randperm).
            let mut order: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (rng.uniform() * (i + 1) as f32) as usize % (i + 1);
                order.swap(i, j);
            }
            let mut evals = 0usize;
            for &idx in &order {
                let initial = *module_values_mut(p, m, is_bias)[idx];
                let mut best_value = initial;
                for sign in [-1.0f32, 1.0] {
                    for shift in 1..16 {
                        stop.check()?;
                        *module_values_mut(p, m, is_bias)[idx] =
                            initial + sign * shift as f32 * step;
                        evals += 1;
                        let s = super::pool(n_pix).install(|| {
                            match (&syn_map, inc.as_mut()) {
                                (Some(map), Some(inc)) => {
                                    let (sel, chan) = map[idx];
                                    let raw = inc.candidate(pb.shape, p, sel, chan);
                                    dist_quant(pb, &raw)
                                        + pb.lambda as f64 * nn_rate_of(p, q, m) as f64
                                            / n_pix as f64
                                }
                                _ => score(pb, p, q, m, &fixed),
                            }
                        });
                        if s < best {
                            best = s;
                            best_value = *module_values_mut(p, m, is_bias)[idx];
                        } else {
                            break;
                        }
                    }
                }
                *module_values_mut(p, m, is_bias)[idx] = best_value;
                if best_value != initial {
                    if let (Some(map), Some(inc)) = (&syn_map, inc.as_mut()) {
                        inc.invalidate(map[idx].0);
                    }
                }
            }
            log(alloc::format!(
                "rdoq {:?} {}: score {start:.6} -> {best:.6} ({n} values, {evals} evaluations)",
                m,
                if is_bias { "bias" } else { "weight" }
            ));
        }
    }
    Ok(())
}
