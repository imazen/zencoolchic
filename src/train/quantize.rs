//! Network quantization (upstream `nnquant/quantizemodel.py`) and conversion of a trained
//! model to bitstream integers (`bitstream/encode.py`, `neuralnet/neuralnet.py`).

use alloc::vec::Vec;

use super::trainer::{Problem, evaluate};
use super::{LossTerms, Params};
use crate::header::{CoolChicHeader, Q_SHIFT_MIN};

/// Module order of the search (upstream iterates `sorted(module_names)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Module {
    Arm,
    Ifce,
    Synthesis,
    Upsampling,
}

impl Module {
    pub(crate) const SORTED: [Module; 4] = [
        Module::Arm,
        Module::Ifce,
        Module::Synthesis,
        Module::Upsampling,
    ];

    /// Index into [`PerModule`](crate::header::PerModule) arrays.
    pub(crate) fn slot(self) -> usize {
        match self {
            Module::Arm => 0,
            Module::Ifce => 1,
            Module::Upsampling => 2,
            Module::Synthesis => 3,
        }
    }

    /// Candidate log2 steps (weight, bias): upstream `POSSIBLE_Q_STEP`.
    fn shifts(
        self,
    ) -> (
        core::ops::RangeInclusive<i32>,
        core::ops::RangeInclusive<i32>,
    ) {
        let m = [
            Q_SHIFT_MIN.arm,
            Q_SHIFT_MIN.ifce,
            Q_SHIFT_MIN.upsampling,
            Q_SHIFT_MIN.synthesis,
        ][self.slot()];
        (m[0]..=0, m[1]..=0)
    }
}

/// Chosen quantization of every module: log2 steps and exp-Golomb orders, `[weight, bias]`.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct NnQuant {
    pub(crate) shift: [Option<[i32; 2]>; 4],
    pub(crate) expgol: [[u32; 2]; 4],
}

/// Weights and biases of one module, in upstream `get_param` order (all weights, then all
/// biases). The upsampling biases, unused by the forward pass and always zero, are included
/// because they are sent.
fn module_tensors(p: &Params, m: Module) -> (Vec<&[f32]>, Vec<&[f32]>, usize) {
    let mut w: Vec<&[f32]> = Vec::new();
    let mut b: Vec<&[f32]> = Vec::new();
    let mut zero_biases = 0;
    match m {
        Module::Arm => {
            for l in p.arm.layers.iter().chain(p.arm.stabiliser.as_ref()) {
                w.push(&l.w);
                b.push(&l.b);
            }
        }
        Module::Ifce => {
            for mlp in p.ifce.iter().flatten() {
                for l in &mlp.layers {
                    w.push(&l.w);
                    b.push(&l.b);
                }
            }
        }
        Module::Upsampling => {
            for k in p.ups_t.iter().chain(&p.ups_c) {
                w.push(k);
            }
            zero_biases = p.ups_t.len() + p.ups_c.len();
        }
        Module::Synthesis => {
            for c in core::iter::once(&p.out_transform)
                .chain(p.syn_stab.as_ref())
                .chain(&p.syn)
            {
                w.push(&c.w);
                b.push(&c.b);
            }
        }
    }
    (w, b, zero_biases)
}

fn module_tensors_mut(p: &mut Params, m: Module) -> Vec<(&mut Vec<f32>, bool)> {
    let mut out: Vec<(&mut Vec<f32>, bool)> = Vec::new();
    match m {
        Module::Arm => {
            for l in p.arm.layers.iter_mut().chain(p.arm.stabiliser.as_mut()) {
                out.push((&mut l.w, false));
                out.push((&mut l.b, true));
            }
        }
        Module::Ifce => {
            for mlp in p.ifce.iter_mut().flatten() {
                for l in &mut mlp.layers {
                    out.push((&mut l.w, false));
                    out.push((&mut l.b, true));
                }
            }
        }
        Module::Upsampling => {
            for k in p.ups_t.iter_mut().chain(p.ups_c.iter_mut()) {
                out.push((k, false));
            }
        }
        Module::Synthesis => {
            for c in core::iter::once(&mut p.out_transform)
                .chain(p.syn_stab.as_mut())
                .chain(p.syn.iter_mut())
            {
                out.push((&mut c.w, false));
                out.push((&mut c.b, true));
            }
        }
    }
    out
}

/// Upstream `exp_golomb_nbins` on already-divided values (its rate estimate, which the
/// search and RDOQ optimize; it is 1 bit per value off the exact code length).
fn nbins(values: impl Iterator<Item = f32>, count: u32) -> f32 {
    let c = libm::exp2f(count as f32);
    values
        .map(|v| {
            2.0 * libm::floorf(libm::log2f(2.0 * v.abs() / c + 1.0))
                + count as f32
                + if v != 0.0 { 1.0 } else { 0.0 }
        })
        .sum()
}

/// Exp-Golomb rate estimate of module `m` with steps `shift` and orders `eg` (upstream
/// `measure_expgolomb_rate`).
fn module_rate(p: &Params, m: Module, shift: [i32; 2], eg: [u32; 2]) -> f32 {
    let (w, b, zeros) = module_tensors(p, m);
    let sw = libm::exp2f(shift[0] as f32);
    let sb = libm::exp2f(shift[1] as f32);
    let rw = nbins(w.iter().flat_map(|t| t.iter()).map(|&v| v / sw), eg[0]);
    let rb = nbins(
        b.iter()
            .flat_map(|t| t.iter())
            .map(|&v| v / sb)
            .chain(core::iter::repeat_n(0.0, zeros)),
        eg[1],
    );
    rw + rb
}

fn best_order(values: &[f32]) -> u32 {
    let mut best = (f32::INFINITY, 0u32);
    for c in 0..=12u32 {
        let r = nbins(values.iter().copied(), c);
        if r < best.0 {
            best = (r, c);
        }
    }
    best.1
}

/// Total network rate of the modules already decided (upstream `get_network_rate`).
pub(crate) fn network_rate(p: &Params, q: &NnQuant) -> f32 {
    Module::SORTED
        .iter()
        .filter_map(|&m| q.shift[m.slot()].map(|s| module_rate(p, m, s, q.expgol[m.slot()])))
        .sum()
}

/// Rounds module `m` to steps `shift`; `None` if a value needs more than 16 bits.
fn quantize_module(fp: &Params, m: Module, shift: [i32; 2]) -> Option<Params> {
    let mut p = fp.clone();
    let steps = [libm::exp2f(shift[0] as f32), libm::exp2f(shift[1] as f32)];
    for (t, is_bias) in module_tensors_mut(&mut p, m) {
        let s = steps[is_bias as usize];
        for v in t.iter_mut() {
            let q = libm::rintf(*v / s);
            if q.abs() > 65535.0 {
                return None;
            }
            *v = q * s;
        }
    }
    Some(p)
}

fn orders_for(p: &Params, m: Module, shift: [i32; 2]) -> [u32; 2] {
    let (w, b, zeros) = module_tensors(p, m);
    let sw = libm::exp2f(shift[0] as f32);
    let sb = libm::exp2f(shift[1] as f32);
    let wv: Vec<f32> = w.iter().flat_map(|t| t.iter()).map(|&v| v / sw).collect();
    let bv: Vec<f32> = b
        .iter()
        .flat_map(|t| t.iter())
        .map(|&v| v / sb)
        .chain(core::iter::repeat_n(0.0, zeros))
        .collect();
    [best_order(&wv), best_order(&bv)]
}

/// Loss of the evaluation pass with the network rate of the decided modules added.
fn loss_with_nn(pb: &Problem<'_>, t: &LossTerms, nn_bits: f32) -> f64 {
    let n_pix = (pb.shape.img.0 * pb.shape.img.1) as f64;
    t.dist + pb.lambda as f64 * (t.rate_bits + nn_bits as f64) / n_pix
}

/// Upstream `quantize_model`: per module (sorted by name), try every (weight, bias) step and
/// keep the one with the lowest loss; the exp-Golomb orders minimize the estimated rate.
pub(crate) fn quantize_model(
    pb: &Problem<'_>,
    fp: &Params,
    has_ifce: bool,
    stop: &dyn enough::Stop,
) -> crate::Result<(Params, NnQuant)> {
    let mut q = NnQuant::default();
    let mut cur = fp.clone();
    let scratch = fp.zeros_like();
    for m in Module::SORTED {
        if m == Module::Ifce && !has_ifce {
            continue;
        }
        let (rw, rb) = m.shifts();
        let steps: Vec<[i32; 2]> = rw
            .clone()
            .flat_map(|sw| rb.clone().map(move |sb| [sw, sb]))
            .collect();
        // The candidate evaluations are independent (`cur` is fixed during the
        // sweep); run them on the problem-sized pool and select the first
        // minimum in stream order afterwards, exactly like the sequential scan.
        let n_pix = pb.shape.img.0 * pb.shape.img.1;
        let losses: Vec<Option<(f64, [u32; 2])>> = super::pool(n_pix).install(|| {
            use rayon::prelude::*;
            steps
                .par_iter()
                .map_init(
                    || scratch.clone(),
                    |scratch, &[sw, sb]| -> crate::Result<Option<(f64, [u32; 2])>> {
                        stop.check()?;
                        let Some(cand) = quantize_module(&cur, m, [sw, sb]) else {
                            return Ok(None);
                        };
                        let t = evaluate(pb, &cand, scratch);
                        let eg = orders_for(&cand, m, [sw, sb]);
                        let mut qq = q;
                        qq.shift[m.slot()] = Some([sw, sb]);
                        qq.expgol[m.slot()] = eg;
                        Ok(Some((loss_with_nn(pb, &t, network_rate(&cand, &qq)), eg)))
                    },
                )
                .collect::<crate::Result<Vec<_>>>()
        })?;
        let mut best: Option<(f64, [i32; 2], [u32; 2])> = None;
        for (&[sw, sb], cand_loss) in steps.iter().zip(losses) {
            let Some((loss, eg)) = cand_loss else {
                continue;
            };
            if best.is_none_or(|b| loss < b.0) {
                best = Some((loss, [sw, sb], eg));
            }
        }
        let (_, s, eg) = best.expect("at least one step fits 16 bits");
        q.shift[m.slot()] = Some(s);
        q.expgol[m.slot()] = eg;
        cur = quantize_module(&cur, m, s).unwrap();
    }
    Ok((cur, q))
}

/// Integers of every network parameter in stream order (`decode::plan`), and the latents
/// `clamp(round(gain · y), -64, 63)`, finest first.
pub(crate) fn to_stream(p: &Params, q: &NnQuant, gain: f32) -> (Vec<i64>, Vec<Vec<i64>>) {
    let mut ints = Vec::new();
    for m in [
        Module::Arm,
        Module::Ifce,
        Module::Upsampling,
        Module::Synthesis,
    ] {
        let Some(s) = q.shift[m.slot()] else { continue };
        let (w, b, zeros) = module_tensors(p, m);
        let (sw, sb) = (libm::exp2f(s[0] as f32), libm::exp2f(s[1] as f32));
        // Stream order inside a module matches `get_param(which)`: all weights, then all biases.
        for t in &w {
            ints.extend(t.iter().map(|&v| libm::rintf(v / sw) as i64));
        }
        for t in &b {
            ints.extend(t.iter().map(|&v| libm::rintf(v / sb) as i64));
        }
        ints.extend(core::iter::repeat_n(0i64, zeros));
    }
    let latents = p
        .latents
        .iter()
        .map(|l| {
            l.iter()
                .map(|&y| (libm::rintf(y * gain) as i64).clamp(-64, 63))
                .collect()
        })
        .collect();
    (ints, latents)
}

/// Fills the quantization fields of `h`.
pub(crate) fn apply_to_header(h: &mut CoolChicHeader, q: &NnQuant) {
    let get = |m: Module| q.shift[m.slot()].unwrap_or([0, 0]);
    let s = [
        get(Module::Arm),
        get(Module::Ifce),
        get(Module::Upsampling),
        get(Module::Synthesis),
    ];
    h.nn_q_shift = crate::header::PerModule {
        arm: s[0],
        ifce: s[1],
        upsampling: s[2],
        synthesis: s[3],
    };
    let st = |v: [i32; 2]| [libm::exp2f(v[0] as f32), libm::exp2f(v[1] as f32)];
    h.nn_q_step = crate::header::PerModule {
        arm: st(s[0]),
        ifce: st(s[1]),
        upsampling: st(s[2]),
        synthesis: st(s[3]),
    };
    let e = q.expgol;
    h.nn_expgol_cnt = crate::header::PerModule {
        arm: e[0],
        ifce: e[1],
        upsampling: e[2],
        synthesis: e[3],
    };
}

/// Every weight (or bias) value of module `m`, flat in upstream `get_param(which)` order.
/// The upsampling biases are not modelled (always zero, unused by the forward pass).
pub(crate) fn module_values_mut(p: &mut Params, m: Module, bias: bool) -> Vec<&mut f32> {
    module_tensors_mut(p, m)
        .into_iter()
        .filter(|(_, b)| *b == bias)
        .flat_map(|(t, _)| t.iter_mut())
        .collect()
}

/// Parameter rate estimate of one module under its chosen quantization.
pub(crate) fn nn_rate_of(p: &Params, q: &NnQuant, m: Module) -> f32 {
    q.shift[m.slot()].map_or(0.0, |s| module_rate(p, m, s, q.expgol[m.slot()]))
}

pub(crate) fn step_of(q: &NnQuant, m: Module, bias: bool) -> f32 {
    libm::exp2f(q.shift[m.slot()].unwrap()[bias as usize] as f32)
}
