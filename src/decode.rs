//! Bitstream → picture (upstream `coolchic/bitstream/decode.py` and
//! `coolchic/bitstream/component/coolchic.py`, decode direction).

use alloc::vec;
use alloc::vec::Vec;

use crate::arm::{self, FixedArm, FixedPointOptions, QuantArm};
use crate::error::{Error, Result};
use crate::expgolomb;
use crate::header::{
    CoolChicHeader, FinalUpsampling, FrameDataType, FrameHeader, FrameType, VideoHeader,
};
use crate::latent::{self, Grid};
use crate::nn::{self, CommonNoise, Conv, Planes, Scale, Synthesis, Upsampling};
use crate::rangecoder::RangeDecoder;
use crate::warp;

/// Pre-flight limits, checked against the headers before any picture-sized allocation.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_pixels: u64,
    pub max_frames: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_pixels: 1 << 28,
            max_frames: 1,
        }
    }
}

/// A decoded picture.
#[derive(Debug, Clone)]
pub struct Picture {
    pub width: u32,
    pub height: u32,
    pub bitdepth: u32,
    pub data_type: FrameDataType,
    /// One plane per channel, row-major. RGB / YUV 4:4:4: three full-size planes;
    /// YUV 4:2:0: Y full size, U and V `ceil`-free half size (`floor(h/2) × floor(w/2)`,
    /// upstream's `avg_pool2d`).
    pub planes: Vec<Vec<u16>>,
    pub plane_dims: Vec<(u32, u32)>,
}

impl Picture {
    /// Interleaved 8-bit RGB, for 8-bit RGB pictures.
    pub fn to_rgb8(&self) -> Option<Vec<u8>> {
        if self.data_type != FrameDataType::Rgb || self.bitdepth != 8 {
            return None;
        }
        let n = (self.width * self.height) as usize;
        let mut out = Vec::with_capacity(3 * n);
        for i in 0..n {
            for p in &self.planes {
                out.push(p[i] as u8);
            }
        }
        Some(out)
    }
}

/// Intermediate results of a decode, for parity tests and inspection.
pub struct DecodeDetails {
    /// Integer latents `(h, w, values)`, finest first (upstream `size_per_latent` order).
    pub grids: Vec<(usize, usize, Vec<i64>)>,
    pub is_hyperlatent: Vec<bool>,
    /// Synthesis output before quantization, `[c][h][w]`.
    pub raw: Vec<f32>,
    pub raw_channels: usize,
}

/// Derived architecture (upstream `CoolChicEncoderParameter.__post_init__`).
pub(crate) struct Arch {
    pub(crate) size_per_latent: Vec<(usize, usize)>,
    pub(crate) is_hyperlatent: Vec<bool>,
    pub(crate) input_features_ifce: Vec<usize>,
    pub(crate) flag_ifce: bool,
    pub(crate) size_per_latent_cr: Vec<(usize, usize)>,
    pub(crate) input_feature_synthesis: usize,
    pub(crate) total_context_arm: usize,
}

fn ceil_div_pow2(x: u32, i: u32) -> usize {
    // math.ceil(x / 2**i) on exact integers.
    (x as u64).div_ceil(1u64 << i) as usize
}

pub(crate) fn arch(h: &CoolChicHeader) -> Result<Arch> {
    let [lr0, lr1] = h.latent_resolution;
    if lr0 > lr1 {
        return Err(Error::Invalid("latent_resolution"));
    }
    let (min_d, max_d) = match h.hyperlatent_resolution {
        Some([a, b]) => (lr0.min(a).min(b), lr1.max(a).max(b)),
        None => (lr0, lr1),
    };
    if max_d > 14 {
        return Err(Error::Invalid("resolution"));
    }
    let [ih, iw] = h.img_size;
    let mut size_per_latent = Vec::new();
    let mut is_hyperlatent = Vec::new();
    for i in min_d..=max_d {
        let s = (ceil_div_pow2(ih, i), ceil_div_pow2(iw, i));
        if lr0 <= i && i <= lr1 {
            size_per_latent.push(s);
            is_hyperlatent.push(false);
        }
        if let Some([a, b]) = h.hyperlatent_resolution
            && a <= i
            && i <= b
        {
            size_per_latent.push(s);
            is_hyperlatent.push(true);
        }
    }
    let n = size_per_latent.len();
    if n != h.n_latent_grids as usize {
        return Err(Error::Invalid(
            "n_latent_grids does not match the resolutions",
        ));
    }
    let size_per_latent_cr = if h.flag_common_randomness {
        (lr0..=lr1)
            .map(|i| (ceil_div_pow2(ih, i), ceil_div_pow2(iw, i)))
            .collect()
    } else {
        Vec::new()
    };
    let flag_ifce = h.ifce_resolution.is_some();
    let mut input_features_ifce = Vec::with_capacity(n);
    for (i, &(gh, _)) in size_per_latent.iter().enumerate() {
        let ratio = libm::ceil(libm::log2(ih as f64 / gh as f64)) as i64;
        let v = match h.ifce_resolution {
            Some([a, b]) if (a as i64) <= ratio && ratio <= b as i64 => (n - 1 - i).max(1),
            _ => 0,
        };
        input_features_ifce.push(v);
    }
    let mut input_feature_synthesis = (lr1 - lr0 + 1) as usize;
    if h.flag_common_randomness {
        input_feature_synthesis *= 2;
    }
    Ok(Arch {
        size_per_latent,
        is_hyperlatent,
        input_features_ifce,
        flag_ifce,
        size_per_latent_cr,
        input_feature_synthesis,
        total_context_arm: (h.spatial_context_arm + h.output_feature_ifce) as usize,
    })
}

/// Shapes of every parameter tensor, in upstream serialization order.
struct ParamPlan {
    /// (module index 0..4, is_bias, numel), in stream order.
    tensors: Vec<(usize, bool, usize)>,
}

const ARM: usize = 0;
const IFCE: usize = 1;
const UPS: usize = 2;
const SYN: usize = 3;

struct Shapes {
    arm_dims: Vec<(usize, usize)>,
    arm_stab: bool,
    ifce: Vec<(usize, usize)>, // (in, out) per IFCE arm, increasing grid index
    ups_t: (usize, usize),     // (count, half size)
    ups_c: (usize, usize),
    syn_out: usize,
    syn_stab_in: Option<usize>,
    syn_layers: Vec<(usize, usize, usize)>,
}

fn shapes(h: &CoolChicHeader, a: &Arch) -> Result<Shapes> {
    let dim = a.total_context_arm;
    if dim == 0 {
        return Err(Error::Invalid("ARM with no context"));
    }
    let mut arm_dims = vec![(dim, dim); h.n_hidden_layers_arm as usize];
    arm_dims.push((dim, 2));
    let ifce = if a.flag_ifce {
        a.input_features_ifce
            .iter()
            .filter(|&&n| n > 0)
            .map(|&n| (n, h.output_feature_ifce as usize))
            .collect()
    } else {
        Vec::new()
    };
    let n_ups = h.latent_resolution[1] as usize;
    let (kt, kc) = (h.ups_k_size as usize, h.ups_preconcat_k_size as usize);
    if n_ups > 0 && (kt < 4 || kt % 2 != 0 || kc % 2 != 1) {
        return Err(Error::Invalid("upsampling kernel sizes"));
    }
    if h.layers_synthesis.is_empty() {
        return Err(Error::Invalid("synthesis has no layers"));
    }
    let syn_layers = nn::layer_dims(a.input_feature_synthesis, &h.layers_synthesis);
    for (l, &(ci, co, k)) in h.layers_synthesis.iter().zip(&syn_layers) {
        if k == 0 || (k % 2 == 0 && l.residual) {
            // Upstream: pad=(k-1)//2 — k=0 underflows, and an even kernel shrinks
            // the output by one so `y + x` broadcast-fails on a residual layer.
            return Err(Error::Invalid("synthesis kernel size"));
        }
        if l.residual && ci != co {
            return Err(Error::Invalid("residual synthesis layer changes width"));
        }
    }
    if h.linear_stabiliser_synth && syn_layers.iter().any(|&(.., k)| k % 2 == 0) {
        // The 1×1 stabiliser output keeps input size; adding it to a shrunk
        // stack fails upstream's broadcast too.
        return Err(Error::Invalid("even synthesis kernel with stabiliser"));
    }
    let syn_out = h.layers_synthesis.last().unwrap().out_ft as usize;
    let syn_stab_in = h
        .linear_stabiliser_synth
        .then_some(if h.flag_common_randomness {
            a.input_feature_synthesis / 2
        } else {
            a.input_feature_synthesis
        });
    Ok(Shapes {
        arm_dims,
        arm_stab: h.linear_stabiliser_arm,
        ifce,
        ups_t: (n_ups, kt.div_ceil(2)),
        ups_c: (n_ups, kc.div_ceil(2)),
        syn_out,
        syn_stab_in,
        syn_layers,
    })
}

fn plan(s: &Shapes) -> ParamPlan {
    let mut t = Vec::new();
    // ARM
    for &(i, o) in &s.arm_dims {
        t.push((ARM, false, i * o));
    }
    if s.arm_stab {
        t.push((ARM, false, s.arm_dims[0].0 * 2));
    }
    for &(_, o) in &s.arm_dims {
        t.push((ARM, true, o));
    }
    if s.arm_stab {
        t.push((ARM, true, 2));
    }
    // IFCE
    for &(i, o) in &s.ifce {
        t.push((IFCE, false, i * o));
    }
    for &(_, o) in &s.ifce {
        t.push((IFCE, true, o));
    }
    // Upsampling: weights of all transposed convs, then all pre-concat convs; then biases.
    for _ in 0..s.ups_t.0 {
        t.push((UPS, false, s.ups_t.1));
    }
    for _ in 0..s.ups_c.0 {
        t.push((UPS, false, s.ups_c.1));
    }
    for _ in 0..s.ups_t.0 + s.ups_c.0 {
        t.push((UPS, true, 1));
    }
    // Synthesis: output transform, stabiliser, main branch.
    t.push((SYN, false, s.syn_out * s.syn_out));
    if let Some(n) = s.syn_stab_in {
        t.push((SYN, false, s.syn_out * n));
    }
    for &(ci, co, k) in &s.syn_layers {
        t.push((SYN, false, co * ci * k * k));
    }
    t.push((SYN, true, s.syn_out));
    if s.syn_stab_in.is_some() {
        t.push((SYN, true, s.syn_out));
    }
    for &(_, co, _) in &s.syn_layers {
        t.push((SYN, true, co));
    }
    // The stream is grouped module by module (weights then biases), which the push order
    // above already follows.
    ParamPlan { tensors: t }
}

fn exp_golomb_order(h: &CoolChicHeader, module: usize, bias: bool) -> u32 {
    let e = &h.nn_expgol_cnt;
    let m = match module {
        ARM => e.arm,
        IFCE => e.ifce,
        UPS => e.upsampling,
        _ => e.synthesis,
    };
    m[bias as usize]
}

pub(crate) struct Networks {
    arm: FixedArm,
    /// One fixed-point IFCE per grid (None where the grid has no IFCE).
    ifce: Vec<Option<FixedArm>>,
    upsampling: Upsampling,
    synthesis: Synthesis,
}

/// Exp-Golomb order of every network parameter, in stream order.
pub(crate) fn param_orders(h: &CoolChicHeader, a: &Arch) -> Result<Vec<u32>> {
    let p = plan(&shapes(h, a)?);
    let total: usize = p.tensors.iter().map(|t| t.2).sum();
    if total > 1 << 24 {
        return Err(Error::LimitExceeded("network parameter count"));
    }
    let mut orders = Vec::with_capacity(total);
    for &(m, b, n) in &p.tensors {
        orders.extend(core::iter::repeat_n(exp_golomb_order(h, m, b), n));
    }
    Ok(orders)
}

/// Parses the network parameters; returns the networks and the raw quantized integers.
pub(crate) fn build_networks(
    h: &CoolChicHeader,
    a: &Arch,
    nn_bytes: &[u8],
) -> Result<(Networks, Vec<i64>)> {
    let s = shapes(h, a)?;
    let orders = param_orders(h, a)?;
    let values = expgolomb::decode(nn_bytes, h.nn_n_bit_pad, &orders)?;
    let raw = values.clone();
    let mut it = values.into_iter();
    let mut take = |n: usize| -> Vec<i64> { it.by_ref().take(n).collect() };

    // ARM
    let mut arm_w: Vec<Vec<i64>> = s.arm_dims.iter().map(|&(i, o)| take(i * o)).collect();
    let arm_sw = s.arm_stab.then(|| take(s.arm_dims[0].0 * 2));
    let arm_b: Vec<Vec<i64>> = s.arm_dims.iter().map(|&(_, o)| take(o)).collect();
    let arm_sb = s.arm_stab.then(|| take(2));
    let mut q = QuantArm {
        dims: s.arm_dims.clone(),
        weights: core::mem::take(&mut arm_w),
        biases: arm_b,
        stabiliser: arm_sw.zip(arm_sb),
    };
    let arm = arm::to_fixed_point(
        &mut q,
        h.nn_q_shift.arm[0],
        h.nn_q_shift.arm[1],
        &FixedPointOptions {
            subtract_last_layer: true,
            n_inter_ft_ctx: h.output_feature_ifce as usize,
            no_residual_layer: false,
        },
    )?;

    // IFCE
    let ifce_w: Vec<Vec<i64>> = s.ifce.iter().map(|&(i, o)| take(i * o)).collect();
    let ifce_b: Vec<Vec<i64>> = s.ifce.iter().map(|&(_, o)| take(o)).collect();
    let mut ifce_arms = ifce_w.into_iter().zip(ifce_b).zip(&s.ifce);
    let mut ifce = Vec::with_capacity(a.size_per_latent.len());
    for &n in &a.input_features_ifce {
        if a.flag_ifce && n > 0 {
            let ((w, b), &dims) = ifce_arms.next().ok_or(Error::Invalid("IFCE count"))?;
            let mut q = QuantArm {
                dims: vec![dims],
                weights: vec![w],
                biases: vec![b],
                stabiliser: None,
            };
            ifce.push(Some(arm::to_fixed_point(
                &mut q,
                h.nn_q_shift.ifce[0],
                h.nn_q_shift.ifce[1],
                &FixedPointOptions {
                    subtract_last_layer: false,
                    n_inter_ft_ctx: 0,
                    no_residual_layer: true,
                },
            )?));
        } else {
            ifce.push(None);
        }
    }

    // Upsampling (float, dequantized).
    let deq =
        |v: Vec<i64>, step: f32| -> Vec<f32> { v.into_iter().map(|x| x as f32 * step).collect() };
    let uw = h.nn_q_step.upsampling[0];
    let conv_t: Vec<Vec<f32>> = (0..s.ups_t.0)
        .map(|_| nn::symmetric_kernel(&deq(take(s.ups_t.1), uw), h.ups_k_size as usize))
        .collect();
    let conv: Vec<Vec<f32>> = (0..s.ups_c.0)
        .map(|_| nn::symmetric_kernel(&deq(take(s.ups_c.1), uw), h.ups_preconcat_k_size as usize))
        .collect();
    let _ups_bias = take(s.ups_t.0 + s.ups_c.0); // unused by the forward pass
    let upsampling = Upsampling { conv_t, conv };

    // Synthesis
    let (sw, sb) = (h.nn_q_step.synthesis[0], h.nn_q_step.synthesis[1]);
    let ot_w = deq(take(s.syn_out * s.syn_out), sw);
    let st_w = s.syn_stab_in.map(|n| deq(take(s.syn_out * n), sw));
    let layer_w: Vec<Vec<f32>> = s
        .syn_layers
        .iter()
        .map(|&(ci, co, k)| deq(take(co * ci * k * k), sw))
        .collect();
    let ot_b = deq(take(s.syn_out), sb);
    let st_b = s.syn_stab_in.map(|_| deq(take(s.syn_out), sb));
    let layer_b: Vec<Vec<f32>> = s
        .syn_layers
        .iter()
        .map(|&(_, co, _)| deq(take(co), sb))
        .collect();
    if it.next().is_some() {
        return Err(Error::Invalid("extra network parameters"));
    }

    let output_transform = Conv {
        c_in: s.syn_out,
        c_out: s.syn_out,
        k: 1,
        weight: ot_w,
        bias: ot_b,
        residual: false,
    };
    let stabiliser = s.syn_stab_in.zip(st_w.zip(st_b)).map(|(n, (w, b))| {
        (
            Conv {
                c_in: n,
                c_out: s.syn_out,
                k: 1,
                weight: w,
                bias: b,
                residual: false,
            },
            n,
        )
    });
    let layers = s
        .syn_layers
        .iter()
        .zip(layer_w.into_iter().zip(layer_b))
        .zip(&h.layers_synthesis)
        .map(|((&(ci, co, k), (w, b)), l)| {
            (
                Conv {
                    c_in: ci,
                    c_out: co,
                    k,
                    weight: w,
                    bias: b,
                    residual: l.residual,
                },
                l.relu,
            )
        })
        .collect();
    Ok((
        Networks {
            arm,
            ifce,
            upsampling,
            synthesis: Synthesis {
                layers,
                stabiliser,
                output_transform,
            },
        },
        raw,
    ))
}

/// Produces each grid in coding order: decodes it, or encodes a known one.
pub(crate) trait GridCoder {
    fn grid(
        &mut self,
        idx: usize,
        arm: &FixedArm,
        window: &[usize],
        inter: Option<&latent::InterContext>,
        hw: (usize, usize),
    ) -> Result<Grid>;
}

impl GridCoder for RangeDecoder<'_> {
    fn grid(
        &mut self,
        _: usize,
        arm: &FixedArm,
        window: &[usize],
        inter: Option<&latent::InterContext>,
        (h, w): (usize, usize),
    ) -> Result<Grid> {
        latent::code_grid(self, arm, window, inter, h, w)
    }
}

fn decode_latents(
    h: &CoolChicHeader,
    a: &Arch,
    nets: &Networks,
    latent_bytes: &[u8],
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<Grid>> {
    let mut rc = RangeDecoder::new(latent_bytes)?;
    code_latents_checked(h, a, nets, &mut rc, check)
}

/// Walks the grids coarsest first, computing each grid's IFCE context from the grids
/// already coded.
#[cfg(feature = "encoder")]
pub(crate) fn code_latents(
    h: &CoolChicHeader,
    a: &Arch,
    nets: &Networks,
    rc: &mut impl GridCoder,
) -> Result<Vec<Grid>> {
    code_latents_checked(h, a, nets, rc, &mut || Ok(()))
}

fn code_latents_checked(
    h: &CoolChicHeader,
    a: &Arch,
    nets: &Networks,
    rc: &mut impl GridCoder,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<Grid>> {
    let window = latent::context_window_indices(h.spatial_context_arm as usize)?;
    let n = a.size_per_latent.len();
    let mut grids: Vec<Option<Grid>> = (0..n).map(|_| None).collect();
    for idx in (0..n).rev() {
        check()?;
        let (gh, gw) = a.size_per_latent[idx];
        let inter = if a.flag_ifce {
            let coded: Vec<&Grid> = grids[idx + 1..]
                .iter()
                .map(|g| g.as_ref().unwrap())
                .collect();
            Some(latent::inter_context(
                &coded,
                (gh, gw),
                nets.ifce[idx].as_ref(),
                h.output_feature_ifce as usize,
            ))
        } else {
            None
        };
        grids[idx] = Some(rc.grid(idx, &nets.arm, &window, inter.as_ref(), (gh, gw))?);
    }
    Ok(grids.into_iter().map(|g| g.unwrap()).collect())
}

fn synthesize(h: &CoolChicHeader, a: &Arch, nets: &Networks, grids: &[Grid]) -> Result<Planes> {
    let latents: Vec<Planes> = grids
        .iter()
        .zip(&a.is_hyperlatent)
        .filter(|(_, hyper)| !**hyper)
        .map(|(g, _)| Planes {
            c: 1,
            h: g.h,
            w: g.w,
            data: g.data.iter().map(|&v| v as f32).collect(),
        })
        .collect();
    let mut syn_in = nets.upsampling.forward(&latents);
    let [ih, iw] = h.img_size.map(|v| v as usize);
    if h.flag_common_randomness {
        let mut noise = CommonNoise::new();
        let cr: Vec<Planes> = a
            .size_per_latent_cr
            .iter()
            .map(|&(ch, cw)| noise.sample(ch, cw))
            .collect();
        let ups = nn::fixed_upsampling(&cr, FinalUpsampling::Bicubic);
        let ups = nn::resize(&ups, ih, iw, FinalUpsampling::Bicubic, Scale::Size);
        let mut cat = Planes::zeros(syn_in.c + ups.c, ih, iw);
        let n0 = syn_in.data.len();
        cat.data[..n0].copy_from_slice(&syn_in.data);
        cat.data[n0..].copy_from_slice(&ups.data);
        syn_in = cat;
    }
    // Each even synthesis kernel removes one pixel per spatial dimension
    // (upstream pad=(k-1)//2); below one pixel there is nothing left to decode.
    let shrink = nets
        .synthesis
        .layers
        .iter()
        .filter(|(c, _)| c.k % 2 == 0)
        .count();
    if syn_in.h <= shrink || syn_in.w <= shrink {
        return Err(Error::Invalid(
            "even synthesis kernels shrink image below one pixel",
        ));
    }
    let out = nets.synthesis.forward(&syn_in);
    Ok(nn::resize(&out, ih, iw, h.final_upsampling, Scale::Size))
}

/// Decodes one Cool-chic payload: returns the synthesis output (`[c][H][W]`, float, not
/// yet quantized) and the remaining bytes.
/// Synthesis output, decoded grids, hyperlatent flags, remaining bytes.
type CoolChicOut<'a> = (Planes, Vec<Grid>, Vec<bool>, &'a [u8]);

pub(crate) fn decode_coolchic<'a>(
    data: &'a [u8],
    limits: &Limits,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<CoolChicOut<'a>> {
    check()?;
    let (h, rest) = CoolChicHeader::parse(data)?;
    let px = h.img_size[0] as u64 * h.img_size[1] as u64;
    if px == 0 {
        return Err(Error::Invalid("empty picture"));
    }
    if px > limits.max_pixels {
        return Err(Error::LimitExceeded("pixels"));
    }
    let a = arch(&h)?;
    let nn_len = h.nn_n_bytes as usize;
    let lat_len = h.n_bytes_latent as usize;
    if rest.len() < nn_len + lat_len {
        return Err(Error::Truncated("cool-chic payload"));
    }
    let (nn_bytes, rest) = rest.split_at(nn_len);
    let (lat_bytes, rest) = rest.split_at(lat_len);
    let (nets, _) = build_networks(&h, &a, nn_bytes)?;
    check()?;
    let grids = decode_latents(&h, &a, &nets, lat_bytes, check)?;
    check()?;
    let out = synthesize(&h, &a, &nets, &grids)?;
    check()?;
    Ok((out, grids, a.is_hyperlatent, rest))
}

#[inline]
fn quantize(v: f32, max: f32) -> f32 {
    libm::rintf(v * max) / max
}

/// A decoded frame kept for output and later as a warping reference: float planes
/// at their stored resolution, quantized to the bitdepth grid (upstream `FrameData`).
pub(crate) struct DecodedFrame {
    planes: Vec<Vec<f32>>,
    plane_dims: Vec<(u32, u32)>,
    bitdepth: u32,
    data_type: FrameDataType,
}

impl DecodedFrame {
    /// The frame as a 3-channel full-resolution tensor — upstream converts
    /// 4:2:0 references to 4:4:4 with a `scale_factor=2` nearest upsample
    /// before warping (`convert_420_to_444`).
    pub(crate) fn as_444(&self) -> Result<Planes> {
        planes_to_444(&self.planes, &self.plane_dims, self.data_type)
    }

    /// The reconstruction at its stored resolution — the same samples upstream
    /// evaluates PSNR on (for 4:2:0, luma plus half-resolution chroma).
    #[cfg_attr(not(feature = "encoder"), allow(dead_code))]
    pub(crate) fn stored_planes(&self) -> &[Vec<f32>] {
        &self.planes
    }
}

/// `DecodedFrame::as_444` lifted onto loose planes: the frame as a 3-channel
/// full-resolution tensor (upstream `convert_420_to_444`).
pub(crate) fn planes_to_444(
    planes: &[Vec<f32>],
    plane_dims: &[(u32, u32)],
    data_type: FrameDataType,
) -> Result<Planes> {
    {
        let (w, h) = plane_dims[0];
        let (w, h) = (w as usize, h as usize);
        match data_type {
            FrameDataType::Rgb | FrameDataType::Yuv444 => Ok(Planes {
                c: 3,
                h,
                w,
                data: planes.concat(),
            }),
            FrameDataType::Yuv420 => {
                let (cw, ch) = (plane_dims[1].0 as usize, plane_dims[1].1 as usize);
                if 2 * cw != w || 2 * ch != h {
                    return Err(Error::Invalid("4:2:0 chroma/luma size mismatch"));
                }
                let mut data = Vec::with_capacity(3 * h * w);
                data.extend_from_slice(&planes[0]);
                for p in &planes[1..3] {
                    for y in 0..h {
                        let row = &p[(y / 2) * cw..(y / 2) * cw + cw];
                        for x in 0..w {
                            data.push(row[x / 2]);
                        }
                    }
                }
                debug_assert_eq!((2 * cw, 2 * ch), (w, h));
                Ok(Planes { c: 3, h, w, data })
            }
            FrameDataType::Flow => Err(Error::Invalid("flow reference frame")),
        }
    }
}

impl DecodedFrame {
    /// Integer samples (`round(x · (2^b - 1))`), one plane per channel.
    pub(crate) fn to_picture(&self) -> Picture {
        let max = ((1u32 << self.bitdepth) - 1) as f32;
        Picture {
            width: self.plane_dims[0].0,
            height: self.plane_dims[0].1,
            bitdepth: self.bitdepth,
            data_type: self.data_type,
            planes: self
                .planes
                .iter()
                .map(|p| p.iter().map(|&v| libm::rintf(v * max) as u16).collect())
                .collect(),
            plane_dims: self.plane_dims.clone(),
        }
    }
}

/// Cool-chic internals of one frame's network, kept for oracle inspection
/// (upstream drops them).
pub(crate) struct CcAux {
    grids: Vec<Grid>,
    is_hyperlatent: Vec<bool>,
    /// Synthesis output before the reconstruction tail, `[c][h][w]`.
    raw: Planes,
}

/// Per-frame internals: the residue network always, plus the motion network
/// for inter frames.
pub(crate) struct FrameAux {
    residue: CcAux,
    /// Only oracle tests read the motion network's internals.
    #[cfg_attr(not(all(test, feature = "oracle")), allow(dead_code))]
    motion: Option<CcAux>,
}

/// The shared reconstruction tail of upstream `decode_frame`: quantize the
/// synthesized image to the bitdepth grid, downsample to 4:2:0 when asked,
/// clamp, quantize once more — the stored `FrameData`.
pub(crate) fn finish_frame(decoded: &Planes, fh: &FrameHeader) -> Result<DecodedFrame> {
    if decoded.c < 3 {
        return Err(Error::Invalid("synthesis has fewer than 3 output channels"));
    }
    let max = ((1u32 << fh.bitdepth) - 1) as f32;
    let (h, w) = (decoded.h, decoded.w);
    let n = h * w;
    // round((2^b - 1) · x) / (2^b - 1) on the frame
    let mut img: Vec<f32> = decoded.data[..3 * n]
        .iter()
        .map(|&v| quantize(v, max))
        .collect();

    let (planes_f, dims): (Vec<Vec<f32>>, Vec<(u32, u32)>) = match fh.frame_data_type {
        FrameDataType::Rgb | FrameDataType::Yuv444 => {
            for v in &mut img {
                *v = v.clamp(0.0, 1.0);
            }
            (
                (0..3).map(|c| img[c * n..(c + 1) * n].to_vec()).collect(),
                vec![(w as u32, h as u32); 3],
            )
        }
        FrameDataType::Yuv420 => {
            let y: Vec<f32> = img[..n].iter().map(|v| v.clamp(0.0, 1.0)).collect();
            let (ch, cw) = (h / 2, w / 2);
            let mut planes = vec![y];
            for c in 1..3 {
                let src = &img[c * n..(c + 1) * n];
                let mut d = vec![0.0f32; ch * cw];
                for yy in 0..ch {
                    for xx in 0..cw {
                        let s = src[2 * yy * w + 2 * xx]
                            + src[2 * yy * w + 2 * xx + 1]
                            + src[(2 * yy + 1) * w + 2 * xx]
                            + src[(2 * yy + 1) * w + 2 * xx + 1];
                        d[yy * cw + xx] = (s / 4.0).clamp(0.0, 1.0);
                    }
                }
                planes.push(d);
            }
            (
                planes,
                vec![
                    (w as u32, h as u32),
                    (cw as u32, ch as u32),
                    (cw as u32, ch as u32),
                ],
            )
        }
        FrameDataType::Flow => return Err(Error::Unsupported("flow frames")),
    };
    // Second quantization pass onto the bitdepth grid.
    let planes: Vec<Vec<f32>> = planes_f
        .into_iter()
        .map(|p| p.into_iter().map(|v| quantize(v, max)).collect())
        .collect();
    Ok(DecodedFrame {
        planes,
        plane_dims: dims,
        bitdepth: fh.bitdepth,
        data_type: fh.frame_data_type,
    })
}

/// Decodes one P or B frame: residue + motion Cool-chic payloads, globally
/// shifted references, warping and alpha/beta compositing — the inter branch of
/// upstream `decode_frame`. Returns the frame and the remaining bytes.
pub(crate) fn decode_inter<'a>(
    fh: &FrameHeader,
    index_references: &[u32],
    data: &'a [u8],
    frames: &[Option<DecodedFrame>],
    limits: &Limits,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<(DecodedFrame, FrameAux, &'a [u8])> {
    // Residue first, then motion, in stream order.
    let (residue, r_grids, r_hyper, rest) = decode_coolchic(data, limits, check)?;
    let (motion, m_grids, m_hyper, rest) = decode_coolchic(rest, limits, check)?;

    let mut refs = Vec::with_capacity(index_references.len());
    for &d in index_references {
        let rf = frames
            .get(d as usize)
            .and_then(|f| f.as_ref())
            .ok_or(Error::Invalid("missing reference frame"))?;
        // 4:2:0 references are upsampled to 4:4:4 before warping.
        refs.push(rf.as_444()?);
    }
    let (h, w) = (refs[0].h, refs[0].w);
    if refs[1..].iter().any(|r| r.h != h || r.w != w)
        || residue.h != h
        || residue.w != w
        || motion.h != h
        || motion.w != w
    {
        return Err(Error::Invalid("reference/residue/motion size mismatch"));
    }
    if fh.global_flow.len() != 2 * refs.len() {
        return Err(Error::Invalid("global_flow"));
    }
    let filter_size = fh
        .warp_filter_size
        .ok_or(Error::Invalid("missing warp_filter_size"))?;
    let (need_res, need_mot) = if fh.frame_type == FrameType::B {
        (5, 4)
    } else {
        (4, 2)
    };
    if residue.c < need_res || motion.c < need_mot {
        return Err(Error::Invalid("residue/motion channel count"));
    }
    let n = h * w;

    // Integer global translation per reference ((x, y) pairs in stream order).
    let shifted: Vec<Planes> = refs
        .iter()
        .enumerate()
        .map(|(i, r)| warp::translate(r, fh.global_flow[2 * i], fh.global_flow[2 * i + 1]))
        .collect();

    // Flow channel 0 is the horizontal (x) displacement, channel 1 vertical.
    let mut pred = warp::warp(
        &shifted[0],
        &motion.data[..n],
        &motion.data[n..2 * n],
        filter_size,
    )?;
    if fh.frame_type == FrameType::B {
        let pred1 = warp::warp(
            &shifted[1],
            &motion.data[2 * n..3 * n],
            &motion.data[3 * n..4 * n],
            filter_size,
        )?;
        // pred = beta * warp(ref0, flow1) + (1 - beta) * warp(ref1, flow2)
        // with beta = clamp(residue[4] + 0.5, 0, 1), shared across channels.
        let beta = &residue.data[4 * n..5 * n];
        for c in 0..3 {
            for (i, &b) in beta.iter().enumerate() {
                let b = (b + 0.5).clamp(0.0, 1.0);
                pred.data[c * n + i] = b * pred.data[c * n + i] + (1.0 - b) * pred1.data[c * n + i];
            }
        }
    }

    // decoded = clamp(residue[3] + 0.5, 0, 1) * pred + residue[:, :3]
    let alpha = &residue.data[3 * n..4 * n];
    let mut img = Vec::with_capacity(3 * n);
    for c in 0..3 {
        for (i, &a) in alpha.iter().enumerate() {
            img.push((a + 0.5).clamp(0.0, 1.0) * pred.data[c * n + i] + residue.data[c * n + i]);
        }
    }
    let decoded = Planes {
        c: 3,
        h,
        w,
        data: img,
    };
    let frame = finish_frame(&decoded, fh)?;
    let aux = FrameAux {
        residue: CcAux {
            grids: r_grids,
            is_hyperlatent: r_hyper,
            raw: residue,
        },
        motion: Some(CcAux {
            grids: m_grids,
            is_hyperlatent: m_hyper,
            raw: motion,
        }),
    };
    Ok((frame, aux, rest))
}

/// Decodes the first frame of a Cool-chic bitstream (an image).
pub fn decode(data: &[u8], limits: &Limits) -> Result<Picture> {
    decode_inner(data, limits, &mut || Ok(())).map(|(p, _)| p)
}

/// Decodes every frame of a Cool-chic bitstream, in display order (upstream
/// `decode_video`). `limits.max_frames` caps the number of frames read.
pub fn decode_video(data: &[u8], limits: &Limits) -> Result<Vec<Picture>> {
    decode_video_inner(data, limits, false, false, &mut || Ok(()))
        .map(|(frames, _)| frames.iter().map(DecodedFrame::to_picture).collect())
}

/// Like [`decode`], also returning the decoded latents and the unquantized synthesis output.
pub fn decode_with_details(data: &[u8], limits: &Limits) -> Result<(Picture, DecodeDetails)> {
    decode_inner(data, limits, &mut || Ok(()))
}

#[cfg(feature = "zencodec")]
pub(crate) fn decode_checked(
    data: &[u8],
    limits: &Limits,
    stop: &dyn enough::Stop,
) -> Result<Picture> {
    decode_inner(data, limits, &mut || stop.check().map_err(Error::from)).map(|(p, _)| p)
}

fn decode_inner(
    data: &[u8],
    limits: &Limits,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<(Picture, DecodeDetails)> {
    let (frames, aux) = decode_video_inner(data, limits, true, true, check)?;
    let aux = aux.into_iter().next().ok_or(Error::Invalid("no frames"))?;
    let latents = DecodeDetails {
        grids: aux
            .residue
            .grids
            .into_iter()
            .map(|g| (g.h, g.w, g.data))
            .collect(),
        is_hyperlatent: aux.residue.is_hyperlatent,
        raw_channels: aux.residue.raw.c,
        raw: aux.residue.raw.data,
    };
    Ok((frames[0].to_picture(), latents))
}

/// Incremental video decoder (upstream `decode_video` + `decode_frame`): the
/// header and coding structure are parsed at construction and each [`step`]
/// decodes one frame in coding order. Frames are stored display-indexed — they
/// double as P/B references — so a caller may pull display-order frames as they
/// become available via [`pull_display`]. `D` is the stream carrier: a slice for
/// borrowed input, an owned `Vec<u8>` for a self-contained (e.g. `'static`)
/// decoder.
///
/// [`step`]: Self::step
/// [`pull_display`]: Self::pull_display
pub(crate) struct VideoDecoder<D> {
    data: D,
    /// Offset into `data` of the next frame header.
    rest: usize,
    structure: Vec<crate::header::StructFrame>,
    /// Next coding-order position to decode.
    pos: usize,
    frames: Vec<Option<DecodedFrame>>,
    /// Per-frame internals for `decode_with_details`; kept empty when not collected.
    aux: Vec<Option<FrameAux>>,
    limits: Limits,
}

impl<D: AsRef<[u8]>> VideoDecoder<D> {
    /// Parses the video header and builds the coding structure. `single_frame`
    /// selects the still-image contract (a multi-frame stream is rejected).
    /// `collect_aux` keeps per-frame internals for `decode_with_details`/oracle
    /// tests; video decodes pass `false` so latents aren't kept after use.
    pub(crate) fn new(
        data: D,
        limits: &Limits,
        single_frame: bool,
        collect_aux: bool,
    ) -> Result<Self> {
        let (vh, rest) = VideoHeader::parse(data.as_ref())?;
        if vh.n_frames > limits.max_frames {
            return Err(Error::LimitExceeded("frames"));
        }
        if single_frame && vh.n_frames != 1 {
            return Err(Error::Unsupported("video (more than one frame)"));
        }
        let structure = vh.coding_structure()?;
        let n = vh.n_frames as usize;
        let rest = data.as_ref().len() - rest.len();
        let mut frames = Vec::new();
        frames.resize_with(n, || None);
        let mut aux = Vec::new();
        if collect_aux {
            aux.resize_with(n, || None);
        }
        Ok(Self {
            data,
            rest,
            structure,
            pos: 0,
            frames,
            aux,
            limits: *limits,
        })
    }

    /// Total declared frames.
    #[cfg(feature = "zencodec")]
    pub(crate) fn n_frames(&self) -> usize {
        self.frames.len()
    }

    /// The decoded frame at a display index, once its coding pass has run.
    #[cfg(feature = "zencodec")]
    pub(crate) fn frame(&self, display: usize) -> Option<&DecodedFrame> {
        self.frames.get(display).and_then(|f| f.as_ref())
    }

    /// Decodes the next frame in coding order; `false` once all are decoded.
    pub(crate) fn step(&mut self, check: &mut dyn FnMut() -> Result<()>) -> Result<bool> {
        let Some(sf) = self.structure.get(self.pos) else {
            return Ok(false);
        };
        check()?;
        let data = self.data.as_ref();
        let (fh, r) = FrameHeader::parse(&data[self.rest..])?;
        self.rest = data.len() - r.len();
        // The frame header repeats what the coding structure already says;
        // upstream uses the structure, so a mismatch means a corrupt stream.
        let frame_type = match sf.index_references.len() {
            0 => FrameType::I,
            1 => FrameType::P,
            _ => FrameType::B,
        };
        if fh.display_index != sf.display_order
            || fh.frame_type != frame_type
            || fh.index_references != sf.index_references
        {
            return Err(Error::Invalid(
                "frame header does not match the coding structure",
            ));
        }
        let display = sf.display_order as usize;
        match fh.frame_type {
            FrameType::I => {
                let (raw, grids, hyper, r) =
                    decode_coolchic(&data[self.rest..], &self.limits, check)?;
                self.rest = data.len() - r.len();
                self.frames[display] = Some(finish_frame(&raw, &fh)?);
                if let Some(slot) = self.aux.get_mut(display) {
                    *slot = Some(FrameAux {
                        residue: CcAux {
                            grids,
                            is_hyperlatent: hyper,
                            raw,
                        },
                        motion: None,
                    });
                }
            }
            FrameType::P | FrameType::B => {
                let (frame, f_aux, r) = decode_inter(
                    &fh,
                    &sf.index_references,
                    &data[self.rest..],
                    &self.frames,
                    &self.limits,
                    check,
                )?;
                self.rest = data.len() - r.len();
                self.frames[display] = Some(frame);
                if let Some(slot) = self.aux.get_mut(display) {
                    *slot = Some(f_aux);
                }
            }
        }
        self.pos += 1;
        Ok(true)
    }

    /// Decodes coding-order frames until the frame at display index `display`
    /// is available; `false` when the stream ends with it uncoded.
    #[cfg(feature = "zencodec")]
    pub(crate) fn pull_display(
        &mut self,
        display: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<bool> {
        while self.pos < self.structure.len()
            && self.frames.get(display).is_some_and(|f| f.is_none())
        {
            self.step(check)?;
        }
        Ok(self.frame(display).is_some())
    }

    /// Consumes the decoder once fully stepped, returning display-order frames
    /// and (when collected) per-frame internals.
    fn finish(self) -> Result<(Vec<DecodedFrame>, Vec<FrameAux>)> {
        // A valid structure covers every display position exactly once.
        let frames = self
            .frames
            .into_iter()
            .map(|f| f.ok_or(Error::Invalid("uncoded display position")))
            .collect::<Result<Vec<_>>>()?;
        let aux = self
            .aux
            .into_iter()
            .map(|a| a.ok_or(Error::Invalid("uncoded display position")))
            .collect::<Result<Vec<_>>>()?;
        Ok((frames, aux))
    }
}

/// Decodes all frames in coding order (upstream `decode_video` + `decode_frame`),
/// returning them in display order. `single_frame` selects the still-image API:
/// streams with more than one frame are rejected, and the frame's internals are
/// returned for [`decode_with_details`].
fn decode_video_inner(
    data: &[u8],
    limits: &Limits,
    single_frame: bool,
    collect_aux: bool,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<(Vec<DecodedFrame>, Vec<FrameAux>)> {
    check()?;
    let mut dec = VideoDecoder::new(data, limits, single_frame, collect_aux)?;
    while dec.step(check)? {}
    dec.finish()
}

#[cfg(test)]
mod stream_variants {
    use super::*;
    use crate::header::VideoHeader;

    /// Upstream-encoded stream with two even (k=2) synthesis kernels
    /// (`--layers_synthesis_residue 8-2-linear-relu,3-2-linear-none` style arch);
    /// upstream v5.0.1 decodes it bit-identically to us.
    const EVENK: &[u8] = include_bytes!("../fuzz/seeds/evenk64_debug.cool");
    /// Same stream with the 2-bit `final_upsampling` field flipped to bilinear —
    /// a format variant upstream's encoder never emits but its decoder accepts.
    const EVENK_BILINEAR: &[u8] = include_bytes!("../fuzz/seeds/evenk64_bilinear_debug.cool");

    fn fnv_planes(p: &Picture) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for pl in &p.planes {
            for &v in pl {
                for b in v.to_le_bytes() {
                    h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
        }
        h
    }

    fn evenk_cc_header() -> CoolChicHeader {
        let (_vh, r) = VideoHeader::parse(EVENK).unwrap();
        let (_fh, r) = FrameHeader::parse(r).unwrap();
        CoolChicHeader::parse(r).unwrap().0
    }

    #[test]
    fn decodes_even_synthesis_kernels() {
        let h = evenk_cc_header();
        assert!(h.layers_synthesis.iter().all(|l| l.k_size % 2 == 0));
        let p = decode(EVENK, &Limits::default()).unwrap();
        assert_eq!((p.width, p.height, p.bitdepth), (64, 64, 8));
        assert_eq!(p.planes.len(), 3);
        // Upstream's decode of this stream is bit-identical (max|Δ|=0 over all
        // samples, verified against v5.0.1); this checksum pins that output.
        assert_eq!(fnv_planes(&p), 0x3e19_4aee_c1f8_32cf);
    }

    #[test]
    fn rejects_bad_synthesis_kernels() {
        let shp = |h: &CoolChicHeader| shapes(h, &arch(h).unwrap());
        let mut h = evenk_cc_header();
        h.layers_synthesis[0].k_size = 0;
        assert!(matches!(
            shp(&h),
            Err(Error::Invalid("synthesis kernel size"))
        ));
        let mut h = evenk_cc_header();
        h.layers_synthesis[0].residual = true;
        // An even kernel shrinks the layer output by one pixel, so `y + x`
        // broadcast-fails upstream too.
        assert!(matches!(
            shp(&h),
            Err(Error::Invalid("synthesis kernel size"))
        ));
        let mut h = evenk_cc_header();
        h.linear_stabiliser_synth = true;
        // The 1×1 stabiliser keeps input size and can't be added to a shrunk stack.
        assert!(matches!(
            shp(&h),
            Err(Error::Invalid("even synthesis kernel with stabiliser"))
        ));
    }

    #[test]
    fn decodes_bilinear_final_upsampling() {
        let p = decode(EVENK_BILINEAR, &Limits::default()).unwrap();
        assert_eq!((p.width, p.height, p.bitdepth), (64, 64, 8));
        // Bit-identical to upstream's decode of the same mutated stream.
        assert_eq!(fnv_planes(&p), 0x951c_b758_96d0_4955);
        // The mode took effect: bilinear differs from the bicubic decode.
        let bicubic = decode(EVENK, &Limits::default()).unwrap();
        assert!(p.planes.iter().zip(&bicubic.planes).any(|(a, b)| a != b));
    }
}

#[cfg(all(test, feature = "encoder"))]
mod tests {
    use super::*;
    use crate::header::{FrameType, VideoHeader};
    use alloc::vec;

    /// Two independently encoded frames assembled into one all-intra stream.
    fn two_frame_stream(display_a: u32, display_b: u32) -> (Vec<u8>, Picture, Picture) {
        let opt = crate::unstable::TrainOptions {
            arch: 3,
            n_itr: 2000,
            rdoq: false,
            ..Default::default()
        };
        let rgb_a: Vec<u8> = (0..16 * 16)
            .map(|i| (i % 256) as u8)
            .flat_map(|v| [v, 255 - v, 128])
            .collect();
        let rgb_b: Vec<u8> = (0..16 * 16)
            .map(|i| (i % 64) as u8)
            .flat_map(|v| [200, v, v])
            .collect();
        let a = crate::unstable::encode_rgb8(&rgb_a, 16, 16, &opt, &mut |_| {}).unwrap();
        let b = crate::unstable::encode_rgb8(&rgb_b, 16, 16, &opt, &mut |_| {}).unwrap();
        let frame = |data: &[u8], display: u32| {
            let (_vh, rest) = VideoHeader::parse(data).unwrap();
            let (mut fh, payload) = FrameHeader::parse(rest).unwrap();
            fh.display_index = display;
            let mut out = fh.to_bytes();
            out.extend_from_slice(payload);
            out
        };
        let vh = VideoHeader {
            n_frames: 2,
            intra_pos: vec![0, 1],
            p_pos: Vec::new(),
        };
        let mut stream = vh.to_bytes();
        stream.extend_from_slice(&frame(&a.bitstream, display_a));
        stream.extend_from_slice(&frame(&b.bitstream, display_b));
        let pa = decode(&a.bitstream, &Limits::default()).unwrap();
        let pb = decode(&b.bitstream, &Limits::default()).unwrap();
        (stream, pa, pb)
    }

    #[test]
    fn two_intra_frames_decode_in_display_order() {
        let (stream, pa, pb) = two_frame_stream(0, 1);
        let pics = decode_video(
            &stream,
            &Limits {
                max_frames: 8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(pics.len(), 2);
        assert_eq!(pics[0].planes, pa.planes);
        assert_eq!(pics[1].planes, pb.planes);
    }

    #[test]
    fn multi_frame_stream_is_rejected_by_decode() {
        let (stream, _, _) = two_frame_stream(0, 1);
        // `decode` is the still-image API: a video stream is not an image.
        assert!(matches!(
            decode(
                &stream,
                &Limits {
                    max_frames: 8,
                    ..Default::default()
                }
            ),
            Err(Error::Unsupported(_))
        ));
        // And a frame cap below the stream's count is a limits error.
        assert!(matches!(
            decode_video(&stream, &Limits::default()),
            Err(Error::LimitExceeded(_))
        ));
    }

    #[test]
    fn frame_header_must_match_the_coding_structure() {
        let (stream, _, _) = two_frame_stream(0, 7);
        // Structure says the second frame has display index 1; the header says 7.
        assert!(matches!(
            decode_video(
                &stream,
                &Limits {
                    max_frames: 8,
                    ..Default::default()
                }
            ),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn inter_frame_without_payloads_errors() {
        let (stream, _, _) = two_frame_stream(0, 1);
        // Take frame 0 from the real stream, then append a bare P-frame header:
        // its residue and motion Cool-chic payloads are absent.
        let (_vh, rest) = VideoHeader::parse(&stream).unwrap();
        let (_fh, r1) = FrameHeader::parse(rest).unwrap();
        let (ch, r2) = CoolChicHeader::parse(r1).unwrap();
        let payload = ch.nn_n_bytes as usize + ch.n_bytes_latent as usize;
        // `r2` spans frame 0's payload plus the rest of the stream.
        let frame0_end = stream.len() - (r2.len() - payload);
        let frame0_start = stream.len() - rest.len();
        let mut video = VideoHeader {
            n_frames: 2,
            intra_pos: vec![0],
            p_pos: vec![1],
        }
        .to_bytes();
        video.extend_from_slice(&stream[frame0_start..frame0_end]);
        video.extend_from_slice(
            &FrameHeader {
                display_index: 1,
                frame_type: FrameType::P,
                frame_data_type: FrameDataType::Rgb,
                bitdepth: 8,
                index_references: vec![0],
                global_flow: vec![0, 0],
                warp_filter_size: Some(4),
            }
            .to_bytes(),
        );
        assert!(matches!(
            decode_video(
                &video,
                &Limits {
                    max_frames: 8,
                    ..Default::default()
                }
            ),
            Err(Error::Truncated(_)) | Err(Error::Invalid(_))
        ));
    }
}

#[cfg(all(test, feature = "oracle"))]
mod oracle {
    use super::*;
    use crate::oracle_fixtures;
    use alloc::vec;
    use std::{eprintln, format, vec::Vec};

    fn read_f32(data: &[u8]) -> Vec<f32> {
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Per-frame raw Cool-chic outputs (residue always, motion for inter)
    /// match upstream's synthesis before the reconstruction tail.
    #[test]
    fn video_raw_outputs_match_upstream() {
        for name in oracle_fixtures::cases("video") {
            let data = oracle_fixtures::read("video", &format!("{name}.cool"));
            let meta: serde_json::Value =
                serde_json::from_slice(&oracle_fixtures::read("video", &format!("{name}.json")))
                    .unwrap();
            let n_frames = meta["n_frames"].as_u64().unwrap() as usize;
            let limits = Limits {
                max_frames: n_frames as u32,
                ..Default::default()
            };
            let (_, aux) = decode_video_inner(&data, &limits, false, true, &mut || Ok(())).unwrap();
            for (i, fmeta) in meta["frames"].as_array().unwrap().iter().enumerate() {
                let frame_type = fmeta["frame_type"].as_str().unwrap_or("I");
                let mut ccs = vec![("residue", &aux[i].residue.raw)];
                if let Some(m) = aux[i].motion.as_ref() {
                    ccs.push(("motion", &m.raw));
                }
                for (cc, raw) in ccs {
                    let shape = fmeta[&format!("{cc}_shape")].as_array().unwrap();
                    assert_eq!(raw.c as u64, shape[1].as_u64().unwrap());
                    assert_eq!(raw.h as u64, shape[2].as_u64().unwrap());
                    assert_eq!(raw.w as u64, shape[3].as_u64().unwrap());
                    let want = read_f32(&oracle_fixtures::read(
                        "video",
                        &format!("{name}.{i}.{cc}.f32"),
                    ));
                    assert_eq!(raw.data.len(), want.len());
                    let max = raw
                        .data
                        .iter()
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    eprintln!("{name} frame {i} {cc} ({frame_type}): raw max |Δ| {max:e}");
                    assert!(
                        max <= 1e-6,
                        "{name} frame {i} {cc}: raw error {max:e} exceeds recorded bound"
                    );
                }
            }
        }
    }
}
