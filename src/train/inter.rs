//! Inter-frame (P/B) training objective: reference compositing and the
//! gradient distribution into the residue and motion Cool-chic models.
//!
//! Upstream `FrameEncoder.forward` (`component/frame.py`) builds the decoded
//! picture from both models' raw outputs:
//!
//! ```text
//! alpha = clamp(residue_raw[3] + 0.5, 0, 1)      # P and B
//! beta  = clamp(residue_raw[4] + 0.5, 0, 1)      # B only
//! pred  = warp(shifted_ref0, motion_raw[0:2])                       # P
//!       = beta·warp(ref0, f1) + (1−beta)·warp(ref1, f2)             # B
//! img   = alpha · pred + residue_raw[0:3]
//! ```
//!
//! Training-mode warps use the unquantized flow; `global_translation` shifts
//! are applied to the references once per frame by the caller. The layout
//! constants (first 3 residue channels = RGB residue, channel 3 = alpha, 4 =
//! beta; motion pairs = (x, y) per reference) match `decode.rs`.

use alloc::vec;
use alloc::vec::Vec;

use super::ops::StepAux;
use super::wasserstein::Wasserstein;
use super::{Params, Shape, StepCfg, StepOut, forward_backward};
use crate::header::FrameType;
use crate::nn::Planes;
use crate::warp;

/// Which Cool-chic a `step` call is producing gradients for.
#[derive(Clone, Copy, Debug)]
pub(crate) enum InterRole {
    /// The residue model (4-channel output for P, 5 for B).
    Residue,
    /// The motion model (2-channel output for P, 4 for B).
    Motion,
}

/// Shared per-frame inter context: the unshifted references, per-reference
/// integer global translations, and the warper parameters.
pub(crate) struct SharedInter {
    /// References expanded to 4:4:4 at image resolution: one plane set for
    /// P, two for B — before the global translation.
    pub(crate) refs: Vec<Planes>,
    /// Integer global translation `(dx, dy)` per reference, applied inside
    /// `composite` before warping (upstream `apply_global_translation`).
    pub(crate) global_flow: Vec<(i32, i32)>,
    /// `warp_filter_size` of the frame (2, 4, 8 or 12).
    pub(crate) warp_filter: u32,
    /// Frame type — only P and B take an inter context.
    pub(crate) frame_type: FrameType,
}

/// A single model's view of the frame context for one step: its role and
/// the partner model's raw output (constant during that step).
#[derive(Clone, Copy)]
pub(crate) struct InterCtx<'a> {
    /// Which model's raw output the current `step` call owns.
    pub(crate) role: InterRole,
    /// The frame's shared inter state.
    pub(crate) shared: &'a SharedInter,
    /// Partner model's raw output at image resolution — `[channels][h][w]`
    /// concatenated planes (5n for a B residue partner, 4n for motion).
    pub(crate) other_raw: &'a [f32],
}

/// One model's inputs of a joint P/B iteration.
pub(crate) struct PairIn<'a> {
    /// Model layout.
    pub(crate) shape: &'a Shape,
    /// Current parameters (not mutated by the step itself).
    pub(crate) p: &'a Params,
    /// The frame target (both models score the same composited picture).
    pub(crate) target: &'a [f32],
    /// Quantization noise for this iteration.
    pub(crate) noise: &'a [f32],
    /// Step configuration (quantizer, λ, distortion weights).
    pub(crate) cfg: StepCfg,
    /// Wasserstein weights — only meaningful on the residue model.
    pub(crate) wass: Option<&'a Wasserstein>,
    /// Scratch params for the partner-capture forward (zeroed, unused).
    pub(crate) scratch: &'a mut Params,
    /// Output gradient buffer, overwritten like `forward_backward`.
    pub(crate) grads: &'a mut Params,
}

/// One joint iteration on a P/B frame — upstream's single
/// `frame_encoder.forward` + `loss.backward()` over both Cool-chics.
/// Both raw outputs come from the same parameter snapshot: a forward-only
/// pass captures the motion raw first (upstream's `cc_enc_out` dict order
/// computes both forwards before any backward), so each model's composite
/// sees the same-iteration partner output as in the joint autograd graph;
/// each backward then scatters the shared distortion gradient into its own
/// parameters. The caller applies one optimizer update over both `grads`.
pub(crate) fn pair_step(
    inter: &SharedInter,
    res: PairIn<'_>,
    mot: PairIn<'_>,
) -> (StepOut, StepOut) {
    let PairIn {
        shape: ms,
        p: mp,
        target: mt,
        noise: mn,
        cfg: mc,
        wass: mw,
        scratch: mscratch,
        grads: mg,
    } = mot;
    // Forward-only capture with the noise the real step will reuse.
    let fwd = StepCfg {
        eval_bitdepth: Some(8),
        ..mc
    };
    let raw_m = forward_backward(ms, mp, mt, mn, fwd, StepAux::default(), mscratch).raw;
    let ctx_r = InterCtx {
        role: InterRole::Residue,
        shared: inter,
        other_raw: &raw_m,
    };
    let out_r = forward_backward(
        res.shape,
        res.p,
        res.target,
        res.noise,
        res.cfg,
        StepAux {
            wass: res.wass,
            inter: Some(ctx_r),
        },
        res.grads,
    );
    let ctx_m = InterCtx {
        role: InterRole::Motion,
        shared: inter,
        other_raw: &out_r.raw,
    };
    let out_m = forward_backward(
        ms,
        mp,
        mt,
        mn,
        mc,
        StepAux {
            wass: mw,
            inter: Some(ctx_m),
        },
        mg,
    );
    (out_r, out_m)
}

/// Joint evaluation of a P/B frame — upstream `test()` on
/// `frame_encoder.forward`: both models' eval-mode raw outputs composite
/// the decoded picture once; the joint loss adds both models' rate terms
/// to the shared distortion.
pub(crate) fn pair_eval(
    shared: &SharedInter,
    res: PairIn<'_>,
    mot: PairIn<'_>,
) -> (StepOut, StepOut) {
    // Same capture order as `pair_step`; eval cfg already forward-only.
    let raw_m = forward_backward(
        mot.shape,
        mot.p,
        mot.target,
        &[],
        mot.cfg,
        StepAux::default(),
        mot.scratch,
    )
    .raw;
    let ctx_r = InterCtx {
        role: InterRole::Residue,
        shared,
        other_raw: &raw_m,
    };
    let out_r = forward_backward(
        res.shape,
        res.p,
        res.target,
        &[],
        res.cfg,
        StepAux {
            wass: res.wass,
            inter: Some(ctx_r),
        },
        res.grads,
    );
    let ctx_m = InterCtx {
        role: InterRole::Motion,
        shared,
        other_raw: &out_r.raw,
    };
    let out_m = forward_backward(
        mot.shape,
        mot.p,
        mot.target,
        &[],
        mot.cfg,
        StepAux {
            wass: mot.wass,
            inter: Some(ctx_m),
        },
        mot.grads,
    );
    (out_r, out_m)
}

/// Forward intermediates the gradient scatter reuses.
pub(crate) struct InterTape {
    /// `alpha·(beta·w0 + (1-beta)·w1)` per channel — the masked prediction
    /// upstream calls `masked_pred` before the residue is added.
    pred: Planes,
    /// Warped reference 0 (all P/B frames).
    w0: Planes,
    /// Warped reference 1 (B frames only).
    w1: Option<Planes>,
    /// Reparameterized alpha, `n` samples.
    alpha: Vec<f32>,
    /// Reparameterized beta, `n` samples (B frames only).
    beta: Option<Vec<f32>>,
}

/// Clamped reparam of one channel: `clamp(raw + 0.5, 0, 1)` (upstream
/// `alpha_reparam`/`beta_reparam`).
fn reparam(raw: &[f32]) -> Vec<f32> {
    raw.iter().map(|&r| (r + 0.5).clamp(0.0, 1.0)).collect()
}

/// `d/draw clamp(raw + 0.5)` — inclusive gate matching the clamped-output
/// convention used by the main distortion path.
#[inline]
fn reparam_gate(r: f32) -> f32 {
    f32::from((-0.5..=0.5).contains(&r))
}

/// Builds the decoded picture from `own_raw` + the context — upstream's
/// P/B reconstruction. `quant` selects upstream's evaluation path: flow is
/// rounded to 1/64-pixel precision (`vanilla_warp_fn`) instead of the
/// training-mode unquantized flow. Returns the 3-channel image (unclamped;
/// the caller's `decoded_dict` applies the data-type tail) and the forward
/// tape.
pub(crate) fn composite(
    ctx: &InterCtx<'_>,
    own_raw: &[f32],
    w: usize,
    h: usize,
    quant: bool,
) -> crate::error::Result<(Vec<f32>, InterTape)> {
    let n = w * h;
    let b = ctx.shared.frame_type == FrameType::B;
    let (res_raw, mot_raw) = match ctx.role {
        InterRole::Residue => (own_raw, ctx.other_raw),
        InterRole::Motion => (ctx.other_raw, own_raw),
    };
    let res_need = (3 + ctx.shared.refs.len()) * n;
    let mot_need = 2 * ctx.shared.refs.len() * n;
    if res_raw.len() != res_need || mot_raw.len() != mot_need {
        return Err(crate::error::Error::Invalid("inter raw channel count"));
    }
    let alpha = reparam(&res_raw[3 * n..4 * n]);
    let beta = b.then(|| reparam(&res_raw[4 * n..5 * n]));
    let wf = |r: usize, fx: &[f32], fy: &[f32]| {
        let (dx, dy) = ctx.shared.global_flow.get(r).copied().unwrap_or((0, 0));
        let shifted = warp::translate(&ctx.shared.refs[r], dx, dy);
        if quant {
            warp::warp(&shifted, fx, fy, ctx.shared.warp_filter)
        } else {
            warp::warp_train(&shifted, fx, fy, ctx.shared.warp_filter)
        }
    };
    let (fx1, fy1) = (&mot_raw[..n], &mot_raw[n..2 * n]);
    let w0 = wf(0, fx1, fy1)?;
    let w1 = if b {
        let (fx2, fy2) = (&mot_raw[2 * n..3 * n], &mot_raw[3 * n..4 * n]);
        Some(wf(1, fx2, fy2)?)
    } else {
        None
    };
    // pred[c] = w0[c] (P) or beta·w0[c] + (1-beta)·w1[c] (B).
    let mut pred = Planes {
        c: 3,
        h,
        w,
        data: w0.data.clone(),
    };
    if let (Some(w1), Some(beta)) = (&w1, &beta) {
        for c in 0..3 {
            for i in 0..n {
                let k = c * n + i;
                pred.data[k] = beta[i] * w0.data[k] + (1.0 - beta[i]) * w1.data[k];
            }
        }
    }
    let mut img = vec![0.0f32; 3 * n];
    for c in 0..3 {
        for i in 0..n {
            img[c * n + i] = alpha[i] * pred.data[c * n + i] + res_raw[c * n + i];
        }
    }
    Ok((
        img,
        InterTape {
            pred,
            w0,
            w1,
            alpha,
            beta,
        },
    ))
}

/// Distributes `d_img` (gradient on the composited 3-channel picture) into
/// this model's raw output — the transpose of [`composite`].
pub(crate) fn scatter(
    ctx: &InterCtx<'_>,
    tape: &InterTape,
    own_raw: &[f32],
    d_img: &[f32],
    w: usize,
    h: usize,
) -> crate::error::Result<Vec<f32>> {
    let n = w * h;
    match ctx.role {
        InterRole::Residue => {
            // draw[0:3n] = d_img; alpha channel gets Σ_c d_img·pred, beta
            // channel Σ_c d_img·alpha·(w0−w1), both through the reparam gate.
            let mut draw = vec![0.0f32; own_raw.len()];
            draw[..3 * n].copy_from_slice(&d_img[..3 * n]);
            let a_raw = &own_raw[3 * n..4 * n];
            for i in 0..n {
                let mut g = 0.0f32;
                for c in 0..3 {
                    g += d_img[c * n + i] * tape.pred.data[c * n + i];
                }
                draw[3 * n + i] = g * reparam_gate(a_raw[i]);
            }
            // `w1` exists exactly on B frames, where the beta channel does.
            if let Some(w1) = &tape.w1 {
                let b_raw = &own_raw[4 * n..5 * n];
                for i in 0..n {
                    let mut g = 0.0f32;
                    for c in 0..3 {
                        g += d_img[c * n + i]
                            * tape.alpha[i]
                            * (tape.w0.data[c * n + i] - w1.data[c * n + i]);
                    }
                    draw[4 * n + i] = g * reparam_gate(b_raw[i]);
                }
            }
            Ok(draw)
        }
        InterRole::Motion => {
            // dout for each warped reference is alpha·d_img scaled by the
            // blend weight; warp_flow_backward maps it to the flow channels.
            let mut draw = Vec::with_capacity(own_raw.len());
            for r in 0..ctx.shared.refs.len() {
                let dout = Planes {
                    c: 3,
                    h,
                    w,
                    data: (0..3 * n)
                        .map(|k| {
                            let i = k % n;
                            let blend = match &tape.beta {
                                Some(b) if r == 0 => b[i],
                                Some(b) => 1.0 - b[i],
                                None => 1.0,
                            };
                            tape.alpha[i] * blend * d_img[k]
                        })
                        .collect(),
                };
                let fx = &own_raw[r * 2 * n..(r * 2 + 1) * n];
                let fy = &own_raw[(r * 2 + 1) * n..(r + 1) * 2 * n];
                let (dx, dy) = ctx.shared.global_flow.get(r).copied().unwrap_or((0, 0));
                let shifted = warp::translate(&ctx.shared.refs[r], dx, dy);
                let (dfx, dfy) =
                    warp::warp_flow_backward(&dout, &shifted, fx, fy, ctx.shared.warp_filter)?;
                draw.extend(dfx);
                draw.extend(dfy);
            }
            Ok(draw)
        }
    }
}

#[cfg(all(test, feature = "oracle"))]
mod tests {
    use super::*;
    use crate::oracle_fixtures;
    use std::{eprintln, format, vec::Vec};

    fn f32_file(case: &str, name: &str) -> Vec<f32> {
        oracle_fixtures::read("warp", &format!("{case}.{name}.f32"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Frozen `dump_inter.py` cases: `comp` (B) and `compp` (P) — composited
    /// picture and `d mse / d raw` for both the residue and motion models.
    #[test]
    fn matches_upstream_inter_compositing() {
        for case in oracle_fixtures::cases("warp") {
            if !case.starts_with("comp") {
                continue;
            }
            let meta: serde_json::Value =
                serde_json::from_slice(&oracle_fixtures::read("warp", &format!("{case}.json")))
                    .unwrap();
            let (h, w, fs) = (
                meta["H"].as_u64().unwrap() as usize,
                meta["W"].as_u64().unwrap() as usize,
                meta["warp_filter_size"].as_u64().unwrap() as u32,
            );
            let ft = match meta["frame_type"].as_str().unwrap() {
                "P" => FrameType::P,
                "B" => FrameType::B,
                other => panic!("frame_type {other}"),
            };
            let n = h * w;
            let res = f32_file(&case, "res");
            let mot = f32_file(&case, "mot");
            let mut refs = alloc::vec::Vec::new();
            for r in 0..usize::from(ft == FrameType::B) + 1 {
                refs.push(Planes {
                    c: 3,
                    h,
                    w,
                    data: f32_file(&case, &format!("ref{r}")),
                });
            }
            let tgt = f32_file(&case, "tgt");
            let want_img = f32_file(&case, "img");
            let shared = SharedInter {
                refs,
                global_flow: alloc::vec![(0, 0); usize::from(ft == FrameType::B) + 1],
                warp_filter: fs,
                frame_type: ft,
            };
            let ctx_r = InterCtx {
                role: InterRole::Residue,
                shared: &shared,
                other_raw: &mot,
            };
            let (img, tape) = composite(&ctx_r, &res, w, h, false).unwrap();
            let max = img
                .iter()
                .zip(&want_img)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("{case}: max |composite diff| {max:e}");
            assert!(max <= 1e-6, "{case}: composited picture error {max:e}");
            // `d_img`: upstream's backward through `clamp(img)` + mean MSE.
            let d_img: Vec<f32> = img
                .iter()
                .zip(&tgt)
                .map(|(&r, &t)| {
                    if (0.0..=1.0).contains(&r) {
                        2.0 * (r.clamp(0.0, 1.0) - t) / (3 * n) as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            let draw = scatter(&ctx_r, &tape, &res, &d_img, w, h).unwrap();
            let want = f32_file(&case, "dres");
            let max = draw
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("{case}: max |d residue diff| {max:e}");
            assert!(max <= 1e-6, "{case}: residue draw error {max:e}");

            let ctx_m = InterCtx {
                role: InterRole::Motion,
                shared: &shared,
                other_raw: &res,
            };
            let (img2, tape2) = composite(&ctx_m, &mot, w, h, false).unwrap();
            assert_eq!(img, img2, "{case}: role-switched composite differs");
            let draw = scatter(&ctx_m, &tape2, &mot, &d_img, w, h).unwrap();
            let want = f32_file(&case, "dmot");
            let max = draw
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            eprintln!("{case}: max |d motion diff| {max:e}");
            assert!(max <= 1e-4, "{case}: motion draw error {max:e}");
        }
    }
}
