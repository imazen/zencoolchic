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
    for &(ci, co, k) in &syn_layers {
        if k % 2 != 1 {
            return Err(Error::Unsupported("even synthesis kernel"));
        }
        let _ = (ci, co);
    }
    for (l, &(ci, co, _)) in h.layers_synthesis.iter().zip(&syn_layers) {
        if l.residual && ci != co {
            return Err(Error::Invalid("residual synthesis layer changes width"));
        }
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
) -> Result<Vec<Grid>> {
    let mut rc = RangeDecoder::new(latent_bytes)?;
    code_latents(h, a, nets, &mut rc)
}

/// Walks the grids coarsest first, computing each grid's IFCE context from the grids
/// already coded.
pub(crate) fn code_latents(
    h: &CoolChicHeader,
    a: &Arch,
    nets: &Networks,
    rc: &mut impl GridCoder,
) -> Result<Vec<Grid>> {
    let window = latent::context_window_indices(h.spatial_context_arm as usize)?;
    let n = a.size_per_latent.len();
    let mut grids: Vec<Option<Grid>> = (0..n).map(|_| None).collect();
    for idx in (0..n).rev() {
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

fn synthesize(h: &CoolChicHeader, a: &Arch, nets: &Networks, grids: &[Grid]) -> Planes {
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
    let out = nets.synthesis.forward(&syn_in);
    nn::resize(&out, ih, iw, h.final_upsampling, Scale::Size)
}

/// Decodes one Cool-chic payload: returns the synthesis output (`[c][H][W]`, float, not
/// yet quantized) and the remaining bytes.
/// Synthesis output, decoded grids, hyperlatent flags, remaining bytes.
type CoolChicOut<'a> = (Planes, Vec<Grid>, Vec<bool>, &'a [u8]);

fn decode_coolchic<'a>(data: &'a [u8], limits: &Limits) -> Result<CoolChicOut<'a>> {
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
    let grids = decode_latents(&h, &a, &nets, lat_bytes)?;
    let out = synthesize(&h, &a, &nets, &grids);
    Ok((out, grids, a.is_hyperlatent, rest))
}

#[inline]
fn quantize(v: f32, max: f32) -> f32 {
    libm::rintf(v * max) / max
}

/// Decodes the first frame of a Cool-chic bitstream (an image).
pub fn decode(data: &[u8], limits: &Limits) -> Result<Picture> {
    decode_inner(data, limits).map(|(p, _)| p)
}

/// Like [`decode`], also returning the decoded latents and the unquantized synthesis output.
pub fn decode_with_details(data: &[u8], limits: &Limits) -> Result<(Picture, DecodeDetails)> {
    decode_inner(data, limits)
}

fn decode_inner(data: &[u8], limits: &Limits) -> Result<(Picture, DecodeDetails)> {
    let (vh, rest) = VideoHeader::parse(data)?;
    if vh.n_frames > limits.max_frames {
        return Err(Error::LimitExceeded("frames"));
    }
    if vh.n_frames != 1 {
        return Err(Error::Unsupported("video (more than one frame)"));
    }
    let (fh, rest) = FrameHeader::parse(rest)?;
    if fh.frame_type != FrameType::I {
        return Err(Error::Unsupported("inter frames"));
    }
    let (raw, grids, hyper, _rest) = decode_coolchic(rest, limits)?;
    if raw.c < 3 {
        return Err(Error::Invalid("synthesis has fewer than 3 output channels"));
    }
    let max = ((1u32 << fh.bitdepth) - 1) as f32;
    let (h, w) = (raw.h, raw.w);
    let n = h * w;
    // round((2^b - 1) · x) / (2^b - 1) on the frame
    let mut img: Vec<f32> = raw.data[..3 * n]
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
    // Second quantization pass, then integer samples (`round(x · (2^b - 1))`).
    let planes: Vec<Vec<u16>> = planes_f
        .into_iter()
        .map(|p| {
            p.into_iter()
                .map(|v| libm::rintf(quantize(v, max) * max) as u16)
                .collect()
        })
        .collect();
    let picture = Picture {
        width: w as u32,
        height: h as u32,
        bitdepth: fh.bitdepth,
        data_type: fh.frame_data_type,
        planes,
        plane_dims: dims,
    };
    let latents = DecodeDetails {
        grids: grids.into_iter().map(|g| (g.h, g.w, g.data)).collect(),
        is_hyperlatent: hyper,
        raw_channels: raw.c,
        raw: raw.data,
    };
    Ok((picture, latents))
}
