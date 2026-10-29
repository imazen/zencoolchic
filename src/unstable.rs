//! Experimental encoder entry points. **Not covered by semver**: the encoder API is still
//! being designed and will go through review before release.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::train::config::{self, Arch};
use crate::train::trainer::{self, Preset, Problem};
use crate::train::wasserstein::Wasserstein;
use crate::train::{Dist, Shape};

pub use crate::train::wasserstein::Vgg16;

/// Distortion tuning (upstream `--tune`): which metrics the rate–distortion loss uses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Tune {
    /// `{"mse": 1.0}` (upstream default).
    #[default]
    Mse,
    /// `{"mse": 0.2, "wasserstein": 0.004}`: the perceptual tuning of "Perceptually
    /// optimised Cool-chic for CLIC 2025". Turns on the common-randomness synthesis
    /// channels and needs [`TrainOptions::vgg`] (RGB only, like upstream).
    Wasserstein,
}

/// Encoder settings (upstream `cc_encode.py` defaults: lambda 1e-3, lr 1e-2, 10,000
/// iterations, `hop`).
#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub lambda: f32,
    pub start_lr: f32,
    pub n_itr: usize,
    /// 0 = vhop, 1 = hop, 2 = mop, 3 = lop.
    pub arch: u8,
    pub seed: u64,
    /// Run RDOQ on the quantized networks (upstream always does).
    pub rdoq: bool,
    /// Distortion tuning of the main and hard-round phases (warm-up is always MSE).
    pub tune: Tune,
    /// Frozen VGG16 feature weights for [`Tune::Wasserstein`]
    /// (`scripts/oracle/dump_vgg.py`, `Vgg16::from_le_bytes`).
    pub vgg: Option<Arc<Vgg16>>,
    /// `latent_resolution` range `(lo, hi)` (upstream `--latent_residue lo-hi`):
    /// `lo` is the finest grid's downsample exponent and `hi` the coarsest.
    /// `None` uses the preset default `(0, 6..8 by image size)`. `lo > 0` runs
    /// synthesis below image resolution and bicubic-rescales the output
    /// (upstream `rescale_output`); it is incompatible with
    /// [`Tune::Wasserstein`]'s common randomness.
    pub latent_res: Option<(u8, u8)>,
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            lambda: 1e-3,
            start_lr: 1e-2,
            n_itr: 10_000,
            arch: 1,
            seed: 0,
            rdoq: true,
            tune: Tune::Mse,
            vgg: None,
            latent_res: None,
        }
    }
}

/// Result of a training run, measured like upstream's final `test()` before network
/// quantization: hardround latents, 8-bit output, latent rate from the ARM.
#[derive(Clone, Debug)]
pub struct TrainReport {
    pub psnr_db: f64,
    pub rate_latent_bpp: f64,
    pub loss: f64,
}

/// Result of a full encode.
#[derive(Clone, Debug)]
pub struct EncodeReport {
    /// Before network quantization (upstream's last training `test()`).
    pub trained: TrainReport,
    /// After network quantization, from the encoder's own evaluation (float ARM).
    pub quantized: TrainReport,
    pub bitstream: Vec<u8>,
}

/// One sample plane of a frame to encode: tightly packed `width × height`
/// unsigned samples in `0 ..= (1 << bitdepth) - 1`, top-left first.
#[derive(Clone, Copy, Debug)]
pub struct Plane<'a> {
    pub width: u32,
    pub height: u32,
    pub samples: &'a [u16],
}

/// Encodes an interleaved 8-bit RGB picture: training, network quantization, bitstream.
pub fn encode_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
) -> crate::Result<EncodeReport> {
    encode_rgb8_with_stop(rgb, width, height, opt, log, &enough::Unstoppable)
}

/// Encodes a planar frame: `planes` follow `frame_data_type` layout — three
/// full-resolution planes for `Rgb`/`Yuv444`, or a full-resolution luma plane
/// followed by two half-resolution chroma planes for `Yuv420` (which requires
/// even luma dimensions). Samples are integers in `0 ..= (1 << bitdepth) - 1`.
/// [`Tune::Wasserstein`] needs a full-resolution three-channel frame: it is
/// rejected for `Yuv420`, matching upstream.
pub fn encode_frame(
    frame_data_type: crate::FrameDataType,
    bitdepth: u8,
    planes: &[Plane<'_>],
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
) -> crate::Result<EncodeReport> {
    encode_frame_with_stop(
        frame_data_type,
        bitdepth,
        planes,
        opt,
        log,
        &enough::Unstoppable,
    )
}

/// Interleaved RGB8 → three planar u16 buffers (the caller builds `Plane` views
/// so the buffers live as long as needed).
fn rgb_planes(rgb: &[u8], width: u32, height: u32) -> crate::Result<[Vec<u16>; 3]> {
    let (w, h) = (width as usize, height as usize);
    if w.checked_mul(h).and_then(|n| n.checked_mul(3)) != Some(rgb.len()) {
        return Err(crate::Error::InvalidArgument("rgb buffer size"));
    }
    let mut bufs = [
        Vec::with_capacity(w * h),
        Vec::with_capacity(w * h),
        Vec::with_capacity(w * h),
    ];
    for px in rgb.as_chunks::<3>().0 {
        for (b, &v) in bufs.iter_mut().zip(px) {
            b.push(v as u16);
        }
    }
    Ok(bufs)
}

pub(crate) fn encode_rgb8_with_stop(
    rgb: &[u8],
    width: u32,
    height: u32,
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
    stop: &dyn enough::Stop,
) -> crate::Result<EncodeReport> {
    let bufs = rgb_planes(rgb, width, height)?;
    let planes: Vec<Plane> = bufs
        .iter()
        .map(|b| Plane {
            width,
            height,
            samples: b,
        })
        .collect();
    encode_frame_with_stop(crate::FrameDataType::Rgb, 8, &planes, opt, log, stop)
}

pub(crate) fn encode_frame_with_stop(
    frame_data_type: crate::FrameDataType,
    bitdepth: u8,
    planes: &[Plane<'_>],
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
    stop: &dyn enough::Stop,
) -> crate::Result<EncodeReport> {
    stop.check()?;
    let (hdr, shape, template, target, wass, dist, gain, n_pix, out_init) =
        setup(frame_data_type, bitdepth, planes, opt)?;
    let pb = Problem {
        shape: &shape,
        target: &target,
        frame_data_type,
        bitdepth: bitdepth as u32,
        lambda: opt.lambda,
        gain,
        dist,
        wass: wass.as_ref(),
        inter: None,
        out_init,
    };
    let n_pix = n_pix as f64;
    let (p, l) = run_training(&pb, &template, opt, log, stop)?;
    let trained = TrainReport {
        psnr_db: -10.0 * libm::log10(l.mse),
        rate_latent_bpp: l.rate_bits / n_pix,
        loss: l.loss,
    };
    let has_ifce = p.ifce.iter().any(|m| m.is_some());
    let (mut pq, q) = crate::train::quantize::quantize_model(&pb, &p, has_ifce, stop)?;
    if opt.rdoq {
        let mut rng = trainer::Rng::new(opt.seed ^ 0x5244_4f51);
        crate::train::rdoq::rdoq(&pb, &mut pq, &q, &mut rng, log, stop)?;
    }
    let mut scratch = pq.zeros_like();
    let lq = trainer::evaluate(&pb, &pq, &mut scratch);
    let nn_bits = crate::train::quantize::network_rate(&pq, &q) as f64;
    let quantized = TrainReport {
        psnr_db: -10.0 * libm::log10(lq.mse),
        rate_latent_bpp: lq.rate_bits / n_pix,
        loss: lq.mse + pb.lambda as f64 * (lq.rate_bits + nn_bits) / n_pix,
    };
    log(alloc::format!(
        "quantized: psnr {:.3} dB, latent {:.4} bpp, network {:.4} bpp (estimate)",
        quantized.psnr_db,
        quantized.rate_latent_bpp,
        nn_bits / n_pix
    ));
    let (ints, lat) = crate::train::quantize::to_stream(&pq, &q, gain);
    let mut h = hdr;
    crate::train::quantize::apply_to_header(&mut h, &q);
    let grids: Vec<crate::latent::Grid> = shape
        .grids
        .iter()
        .zip(lat)
        .map(|(&(gh, gw), data)| crate::latent::Grid { h: gh, w: gw, data })
        .collect();
    stop.check()?;
    let bitstream = crate::write::write_image(&h, frame_data_type, bitdepth as u32, &ints, &grids)?;
    Ok(EncodeReport {
        trained,
        quantized,
        bitstream,
    })
}

/// Inter-frame settings for [`encode_video`] (upstream `--warp_filter_size` and
/// `--n_itr_pretrain_motion` plus the external optical-flow hook replacing
/// upstream's internal RAFT call).
#[derive(Default)]
pub struct VideoOptions<'a> {
    /// Warping interpolation taps written into inter frame headers
    /// (upstream `--warp_filter_size`, default 8; valid values 2, 4, 8, 12).
    /// `None` uses upstream's default.
    pub warp_filter_size: Option<u32>,
    /// Motion pre-training iterations (upstream `--n_itr_pretrain_motion`,
    /// default 1000). `0` keeps upstream's default.
    pub itr_motion_pretrain: usize,
    /// External optical-flow estimator — upstream feeds RAFT the frame and each
    /// globally-shifted reference as 4:4:4 tensors; this callback receives the
    /// same tensors as `[3][h*w]` `0..=1` planes, `(display_index, frame,
    /// shifted_refs)`, and returns one `[2][h*w]` pixel-unit flow field per
    /// reference (each flow warps its shifted reference onto the frame), or
    /// `None` to skip guided pre-training for this frame. The trained model is
    /// injected into every odd warm-up candidate, like upstream.
    pub flow: Option<FlowEstimator<'a>>,
}

/// The external optical-flow callback of [`VideoOptions::flow`].
pub type FlowEstimator<'a> = &'a mut dyn FnMut(u32, &[f32], &[&[f32]]) -> Option<Vec<Vec<f32>>>;

/// One frame's entry in [`VideoReport::frames`].
#[derive(Clone, Debug)]
pub struct VideoFrameReport {
    /// Position in display order.
    pub display_index: u32,
    /// I, P or B — one per reference count, like upstream's header.
    pub frame_type: crate::header::FrameType,
    /// The header written ahead of the frame's payload(s).
    pub header: crate::header::FrameHeader,
    /// PSNR of the decoded frame (bitdepth-quantized, like upstream's final
    /// `test()` numbers).
    pub psnr_db: f64,
}

/// Result of [`encode_video`].
#[derive(Clone, Debug)]
pub struct VideoReport {
    /// Per-frame results in coding order.
    pub frames: Vec<VideoFrameReport>,
    /// The complete bitstream.
    pub bitstream: Vec<u8>,
}

/// Network quantization, optional RDOQ and the stream conversion of one
/// cool-chic (upstream `quantize_model` + `rdoq_model` + `cc_enc.get_param`).
fn finish_cc(
    pb: &Problem<'_>,
    p: &crate::train::Params,
    hdr: &mut crate::header::CoolChicHeader,
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
    stop: &dyn enough::Stop,
) -> crate::Result<Vec<u8>> {
    let has_ifce = p.ifce.iter().any(|m| m.is_some());
    let (mut pq, q) = crate::train::quantize::quantize_model(pb, p, has_ifce, stop)?;
    if opt.rdoq {
        let mut rng = trainer::Rng::new(opt.seed ^ 0x5244_4f51);
        crate::train::rdoq::rdoq(pb, &mut pq, &q, &mut rng, log, stop)?;
    }
    let (ints, lat) = crate::train::quantize::to_stream(&pq, &q, pb.gain);
    crate::train::quantize::apply_to_header(hdr, &q);
    let grids: Vec<crate::latent::Grid> = pb
        .shape
        .grids
        .iter()
        .zip(lat)
        .map(|(&(gh, gw), data)| crate::latent::Grid { h: gh, w: gw, data })
        .collect();
    crate::write::write_coolchic_full(hdr, &ints, &grids)
}

/// Encodes a video: upstream `encode_one_frame` iterated in coding order.
/// `frames` are display-order planar frames (same layout rules as
/// [`encode_frame`]); `intra_pos`/`p_pos` mark I/P display positions exactly
/// like upstream's `--intra_pos`/`--p_pos` (every other position is a B frame —
/// both the first and last display index must be I or P). Every decoded frame
/// is kept and handed to later frames as a warping reference, so a stream
/// encodes deterministically and round-trips through [`crate::decode_video`].
///
/// Inter frames use [`Tune`]'s distortion on the composited picture (upstream's
/// joint objective) — [`Tune::Wasserstein`] is allowed on YUV444/RGB video and
/// attaches common randomness to the residue model only, like upstream.
#[allow(clippy::too_many_arguments)]
pub fn encode_video(
    frame_data_type: crate::FrameDataType,
    bitdepth: u8,
    frames: &[&[Plane<'_>]],
    intra_pos: &[u32],
    p_pos: &[u32],
    opt: &TrainOptions,
    vopt: &mut VideoOptions<'_>,
    log: &mut dyn FnMut(String),
) -> crate::Result<VideoReport> {
    encode_video_with_stop(
        frame_data_type,
        bitdepth,
        frames,
        intra_pos,
        p_pos,
        opt,
        vopt,
        log,
        &enough::Unstoppable,
    )
}

/// [`encode_video`] with cooperative cancellation checked between training
/// phases and frames.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_video_with_stop(
    frame_data_type: crate::FrameDataType,
    bitdepth: u8,
    frames: &[&[Plane<'_>]],
    intra_pos: &[u32],
    p_pos: &[u32],
    opt: &TrainOptions,
    vopt: &mut VideoOptions<'_>,
    log: &mut dyn FnMut(String),
    stop: &dyn enough::Stop,
) -> crate::Result<VideoReport> {
    let vh = crate::header::VideoHeader {
        n_frames: frames.len() as u32,
        intra_pos: intra_pos.into(),
        p_pos: p_pos.into(),
    };
    let structure = vh.coding_structure()?;
    if opt.tune == Tune::Wasserstein && frame_data_type == crate::FrameDataType::Yuv420 {
        return Err(crate::Error::InvalidArgument(
            "tune=wasserstein does not support yuv420 frame data",
        ));
    }
    if opt.tune == Tune::Wasserstein && opt.vgg.is_none() {
        return Err(crate::Error::InvalidArgument(
            "tune=wasserstein needs Vgg16 weights in TrainOptions::vgg",
        ));
    }
    if opt.tune == Tune::Wasserstein && opt.latent_res.is_some_and(|(lo, _)| lo > 0) {
        return Err(crate::Error::InvalidArgument(
            "tune=wasserstein needs full-resolution latents",
        ));
    }
    let warp_filter = vopt.warp_filter_size.unwrap_or(8);
    if ![2, 4, 8, 12].contains(&warp_filter) {
        return Err(crate::Error::InvalidArgument(
            "warp_filter_size must be 2, 4, 8 or 12",
        ));
    }
    let itr_motion = if vopt.itr_motion_pretrain == 0 {
        1000
    } else {
        vopt.itr_motion_pretrain
    };
    // Validate every frame up front and normalize to f32 planes.
    let max = ((1u32 << bitdepth) - 1) as f32;
    let mut inputs = Vec::with_capacity(frames.len());
    for planes in frames.iter() {
        if planes.len() != 3 {
            return Err(crate::Error::InvalidArgument(
                "expected three planes per frame",
            ));
        }
        let (w, h) = (planes[0].width as usize, planes[0].height as usize);
        let (cw, ch) = match frame_data_type {
            crate::FrameDataType::Yuv420 => {
                if w % 2 != 0 || h % 2 != 0 {
                    return Err(crate::Error::InvalidArgument(
                        "yuv420 needs even luma dimensions",
                    ));
                }
                (w / 2, h / 2)
            }
            crate::FrameDataType::Rgb | crate::FrameDataType::Yuv444 => (w, h),
            _ => {
                return Err(crate::Error::InvalidArgument(
                    "frame_data_type must be Rgb/Yuv420/Yuv444",
                ));
            }
        };
        let mut norm = Vec::with_capacity(3);
        let mut dims = Vec::with_capacity(3);
        for (i, p) in planes.iter().enumerate() {
            let (pw, ph) = if i == 0 { (w, h) } else { (cw, ch) };
            if p.width as usize != pw
                || p.height as usize != ph
                || pw.checked_mul(ph) != Some(p.samples.len())
            {
                return Err(crate::Error::InvalidArgument(
                    "plane dimensions do not match frame_data_type",
                ));
            }
            if p.samples.iter().any(|&s| s as f32 > max) {
                return Err(crate::Error::InvalidArgument(
                    "sample exceeds the bitdepth range",
                ));
            }
            norm.push(
                p.samples
                    .iter()
                    .map(|&s| s as f32 / max)
                    .collect::<Vec<f32>>(),
            );
            dims.push((p.width, p.height));
        }
        inputs.push((norm, dims));
    }
    let n_frames = frames.len();
    let mut decoded: Vec<Option<crate::decode::DecodedFrame>> =
        (0..n_frames).map(|_| None).collect();
    let mut reports = Vec::with_capacity(n_frames);
    let mut out = vh.to_bytes();
    let dist = match opt.tune {
        Tune::Mse => Dist::MSE,
        Tune::Wasserstein => Dist::WASSERSTEIN,
    };
    for sf in &structure {
        let (norm, dims) = &inputs[sf.display_order as usize];
        let (w, h) = (dims[0].0 as usize, dims[0].1 as usize);
        let n_pix = (w * h) as f64;
        let target: Vec<f32> = norm.concat();
        let mut fh = crate::header::FrameHeader {
            display_index: sf.display_order,
            frame_type: match sf.index_references.len() {
                0 => crate::header::FrameType::I,
                1 => crate::header::FrameType::P,
                _ => crate::header::FrameType::B,
            },
            frame_data_type,
            bitdepth: bitdepth as u32,
            index_references: sf.index_references.clone(),
            global_flow: Vec::new(),
            warp_filter_size: None,
        };
        let cc_bytes: Vec<Vec<u8>> = if fh.frame_type == crate::header::FrameType::I {
            let planes: Vec<Plane> = frames[sf.display_order as usize].to_vec();
            let (hdr, shape, template, _t, wass, d, gain, _n, out_init) =
                setup(frame_data_type, bitdepth, &planes, opt)?;
            let pb = Problem {
                shape: &shape,
                target: &target,
                frame_data_type,
                bitdepth: bitdepth as u32,
                lambda: opt.lambda,
                gain,
                dist: d,
                wass: wass.as_ref(),
                inter: None,
                out_init,
            };
            let (p, l) = run_training(&pb, &template, opt, log, stop)?;
            log(alloc::format!(
                "I{}: trained psnr {:.3} dB, rate {:.4} bpp",
                sf.display_order,
                -10.0 * libm::log10(l.mse),
                l.rate_bits / n_pix
            ));
            let mut hdr = hdr;
            let bytes = finish_cc(&pb, &p, &mut hdr, opt, log, stop)?;
            let (raw, _g, _hy, _rest) =
                crate::decode::decode_coolchic(&bytes, &crate::Limits::default(), &mut || Ok(()))?;
            decoded[sf.display_order as usize] = Some(crate::decode::finish_frame(&raw, &fh)?);
            alloc::vec![bytes]
        } else {
            let n_refs = sf.index_references.len();
            let mut refs = Vec::with_capacity(n_refs);
            for &d in &sf.index_references {
                let rf = decoded[d as usize]
                    .as_ref()
                    .ok_or(crate::Error::Invalid("missing reference frame"))?;
                refs.push(rf.as_444()?);
            }
            // Global integer translation per reference, on the 4:4:4 tensors
            // (upstream get_global_translation converts yuv420 the same way).
            let target444 = crate::decode::planes_to_444(norm, dims, frame_data_type)?;
            let global_flow: Vec<(i32, i32)> = refs
                .iter()
                .map(|r| crate::warp::global_translation(&target444, r))
                .collect();
            let shifted: Vec<crate::nn::Planes> = refs
                .iter()
                .zip(&global_flow)
                .map(|(r, &(dx, dy))| crate::warp::translate(r, dx, dy))
                .collect();
            let arch = match opt.arch {
                0 => Arch::VeryHigh,
                1 => Arch::High,
                2 => Arch::Medium,
                _ => Arch::Low,
            };
            let mut h_r = config::header_inter(arch, h as u32, w as u32, 3 + n_refs as u32);
            if let Some((lo, hi)) = opt.latent_res {
                if lo > hi || hi > 15 {
                    return Err(crate::Error::InvalidArgument(
                        "latent_res must satisfy lo <= hi <= 15",
                    ));
                }
                h_r.latent_resolution = [lo as u32, hi as u32];
                h_r.n_latent_grids =
                    (hi - lo + 1) as u32 + h_r.hyperlatent_resolution.map_or(0, |[a, b]| b - a + 1);
            }
            // Upstream: common randomness only on the residue encoder, and only
            // for `--tune wasserstein` (`utils/parsecli.py`).
            h_r.flag_common_randomness = opt.tune == Tune::Wasserstein;
            let h_m = config::header_motion(arch, h as u32, w as u32, 2 * n_refs as u32);
            let a_r = crate::decode::arch(&h_r)?;
            let a_m = crate::decode::arch(&h_m)?;
            let s_r = Shape::new(&h_r, &a_r)?;
            let s_m = Shape::new(&h_m, &a_m)?;
            let t_r = s_r.zero_params(&h_r, &a_r);
            let t_m = s_m.zero_params(&h_m, &a_m);
            // Guided motion pre-training: the external flow fields are learned
            // like an image (upstream `guided_motion_pretraining`), then handed
            // to every odd warm-up candidate.
            let mut rng = trainer::Rng::new(opt.seed ^ 0x9d2c_5681_u64);
            let mut pretrained = None;
            if let Some(flow) = vopt.flow.as_deref_mut() {
                let ref_slices: Vec<&[f32]> = shifted.iter().map(|r| r.data.as_slice()).collect();
                if let Some(flows) = flow(sf.display_order, &target444.data, &ref_slices) {
                    if flows.len() != n_refs || flows.iter().any(|f| f.len() != 2 * w * h) {
                        return Err(crate::Error::InvalidArgument(
                            "flow estimator must return one [2][h*w] field per reference",
                        ));
                    }
                    let flow_target: Vec<f32> = flows.concat();
                    let pb_flow = Problem {
                        shape: &s_m,
                        target: &flow_target,
                        frame_data_type: crate::FrameDataType::Flow,
                        bitdepth: 8,
                        lambda: 20.0 * opt.lambda,
                        gain: 16.0,
                        dist: Dist::MSE,
                        wass: None,
                        inter: None,
                        out_init: None,
                    };
                    let preset =
                        Preset::inter(opt.lambda, opt.start_lr, opt.n_itr, itr_motion, dist);
                    let phase = preset.motion_pretrain.as_ref().unwrap();
                    let mut pm = t_m.clone();
                    trainer::initialize(&mut pm, &s_m, &mut rng);
                    let (pm, _l) =
                        trainer::train(&pb_flow, pm, phase, &mut rng, &mut |_, _, _| {}, stop)?;
                    pretrained = Some(pm);
                }
            }
            let shared = crate::train::inter::SharedInter {
                refs,
                global_flow: global_flow.clone(),
                warp_filter,
                frame_type: fh.frame_type,
            };
            let wass = opt
                .vgg
                .as_ref()
                .filter(|_| opt.tune == Tune::Wasserstein)
                .map(|v| Wasserstein::new(v.clone(), &target, h, w));
            fn mk_pb<'a>(
                shape: &'a Shape,
                target: &'a [f32],
                fdt: crate::FrameDataType,
                bitdepth: u32,
                lambda: f32,
                dist: Dist,
                wass: Option<&'a Wasserstein>,
            ) -> Problem<'a> {
                Problem {
                    shape,
                    target,
                    frame_data_type: fdt,
                    bitdepth,
                    lambda,
                    gain: 16.0,
                    dist,
                    wass,
                    inter: None,
                    out_init: None,
                }
            }
            let pb_r = mk_pb(
                &s_r,
                &target,
                frame_data_type,
                bitdepth as u32,
                opt.lambda,
                dist,
                wass.as_ref(),
            );
            let pb_m = mk_pb(
                &s_m,
                &target,
                frame_data_type,
                bitdepth as u32,
                opt.lambda,
                dist,
                wass.as_ref(),
            );
            let preset = Preset::inter(opt.lambda, opt.start_lr, opt.n_itr, itr_motion, dist);
            let mut progress =
                |it: usize, _t: &crate::train::LossTerms, v: Option<&crate::train::LossTerms>| {
                    if let Some(v) = v {
                        log(alloc::format!(
                            "frame {} itr {it:>5}: psnr {:.3} dB, loss {:.6}",
                            sf.display_order,
                            -10.0 * libm::log10(v.mse),
                            v.loss
                        ));
                    }
                };
            let (p_r, p_m, _l) = trainer::encode_training_pair(
                &pb_r,
                &pb_m,
                &shared,
                &t_r,
                &t_m,
                &preset,
                pretrained.as_ref().map(|p| (p, 1.0f32)),
                opt.seed ^ (sf.display_order as u64).wrapping_mul(0x9E37_79B9),
                &mut progress,
                stop,
            )?;
            // Upstream `quantize_model`/`rdoq_model` iterate residue then
            // motion; each pass sees the partner's current parameters. The
            // residue pass uses the trained motion's eval raw; the motion pass
            // decodes the residue's just-written payload — the partner the
            // decoder will actually see.
            let mut pb_r = pb_r;
            let mut pb_m = pb_m;
            let raw_m = crate::train::eval::eval_raw(&s_m, &p_m, 16.0);
            pb_r.inter = Some(crate::train::inter::InterCtx {
                role: crate::train::inter::InterRole::Residue,
                shared: &shared,
                other_raw: &raw_m,
            });
            let mut h_r_q = h_r.clone();
            let r_bytes = finish_cc(&pb_r, &p_r, &mut h_r_q, opt, log, stop)?;
            let (raw_r, _g, _hy, _rest) = crate::decode::decode_coolchic(
                &r_bytes,
                &crate::Limits::default(),
                &mut || Ok(()),
            )?;
            pb_m.inter = Some(crate::train::inter::InterCtx {
                role: crate::train::inter::InterRole::Motion,
                shared: &shared,
                other_raw: &raw_r.data,
            });
            let mut h_m_q = h_m.clone();
            let m_bytes = finish_cc(&pb_m, &p_m, &mut h_m_q, opt, log, stop)?;
            fh.global_flow = global_flow.iter().flat_map(|&(x, y)| [x, y]).collect();
            fh.warp_filter_size = Some(warp_filter);
            let mut payload = Vec::new();
            payload.extend_from_slice(&r_bytes);
            payload.extend_from_slice(&m_bytes);
            let (frame, _aux, _rest) = crate::decode::decode_inter(
                &fh,
                &sf.index_references,
                &payload,
                &decoded,
                &crate::Limits::default(),
                &mut || Ok(()),
            )?;
            decoded[sf.display_order as usize] = Some(frame);
            alloc::vec![r_bytes, m_bytes]
        };
        // PSNR of the stored reconstruction on the stored planes — the same
        // samples upstream evaluates on (420 chroma at half resolution).
        let f = decoded[sf.display_order as usize].as_ref().unwrap();
        let stored = f.stored_planes();
        let mut num = 0.0f64;
        let mut cnt = 0usize;
        for (p, q) in stored.iter().zip(norm.iter()) {
            for (a, b) in p.iter().zip(q.iter()) {
                let d = f64::from(*a - *b);
                num += d * d;
            }
            cnt += p.len();
        }
        let mse = num / cnt.max(1) as f64;
        reports.push(VideoFrameReport {
            display_index: sf.display_order,
            frame_type: fh.frame_type,
            header: fh.clone(),
            psnr_db: -10.0 * libm::log10(mse.max(1e-30)),
        });
        out.extend_from_slice(&crate::write::write_frame(&fh, &cc_bytes));
    }
    Ok(VideoReport {
        frames: reports,
        bitstream: out,
    })
}

/// The last tuple element is the luma sample count — the rate/PSNR
/// normalization (upstream `n_pixels`).
type Setup = (
    crate::header::CoolChicHeader,
    Shape,
    crate::train::Params,
    Vec<f32>,
    Option<Wasserstein>,
    Dist,
    f32,
    usize,
    Option<[(f32, f32); 3]>,
);

fn setup(
    frame_data_type: crate::FrameDataType,
    bitdepth: u8,
    planes: &[Plane<'_>],
    opt: &TrainOptions,
) -> crate::Result<Setup> {
    if frame_data_type == crate::FrameDataType::Flow || !(8..=16).contains(&bitdepth) {
        return Err(crate::Error::InvalidArgument(
            "frame_data_type must be Rgb/Yuv420/Yuv444, bitdepth 8..=16",
        ));
    }
    if planes.len() != 3 {
        return Err(crate::Error::InvalidArgument(
            "expected three planes (luma + two chroma, or R + G + B)",
        ));
    }
    let (w, h) = (planes[0].width as usize, planes[0].height as usize);
    if w == 0 || h == 0 || w > 16383 || h > 16383 {
        return Err(crate::Error::InvalidArgument(
            "luma dimensions must be 1..=16383",
        ));
    }
    let (cw, ch) = match frame_data_type {
        crate::FrameDataType::Yuv420 => {
            if w % 2 != 0 || h % 2 != 0 {
                return Err(crate::Error::InvalidArgument(
                    "yuv420 needs even luma dimensions",
                ));
            }
            (w / 2, h / 2)
        }
        _ => (w, h),
    };
    let max = ((1u32 << bitdepth) - 1) as f32;
    for (i, p) in planes.iter().enumerate() {
        let (pw, ph) = if i == 0 { (w, h) } else { (cw, ch) };
        if p.width as usize != pw || p.height as usize != ph {
            return Err(crate::Error::InvalidArgument(
                "plane dimensions do not match frame_data_type",
            ));
        }
        if pw.checked_mul(ph) != Some(p.samples.len()) {
            return Err(crate::Error::InvalidArgument("plane buffer size"));
        }
        if p.samples.iter().any(|&s| s as f32 > max) {
            return Err(crate::Error::InvalidArgument(
                "sample exceeds the bitdepth range",
            ));
        }
    }
    if !opt.lambda.is_finite() || opt.lambda <= 0.0 || opt.lambda > 1.0 {
        return Err(crate::Error::InvalidArgument(
            "lambda must be finite in (0, 1]",
        ));
    }
    if !opt.start_lr.is_finite() || opt.start_lr <= 0.0 || opt.n_itr < 2000 || opt.arch > 3 {
        return Err(crate::Error::InvalidArgument(
            "lr must be positive, iterations >= 2000, arch <= 3",
        ));
    }
    let (dist, flag_cr) = match opt.tune {
        Tune::Mse => (Dist::MSE, false),
        Tune::Wasserstein => {
            if opt.vgg.is_none() {
                return Err(crate::Error::InvalidArgument(
                    "tune=wasserstein needs Vgg16 weights in TrainOptions::vgg \
                     (dumped by scripts/oracle/dump_vgg.py)",
                ));
            }
            if frame_data_type == crate::FrameDataType::Yuv420 {
                return Err(crate::Error::InvalidArgument(
                    "tune=wasserstein does not support yuv420 frame data",
                ));
            }
            if opt.latent_res.is_some_and(|(lo, _)| lo > 0) {
                return Err(crate::Error::InvalidArgument(
                    "tune=wasserstein needs full-resolution latents",
                ));
            }
            (Dist::WASSERSTEIN, true)
        }
    };
    let arch = match opt.arch {
        0 => Arch::VeryHigh,
        1 => Arch::High,
        2 => Arch::Medium,
        _ => Arch::Low,
    };
    let mut hdr = config::header(arch, planes[0].height, planes[0].width);
    if let Some((lo, hi)) = opt.latent_res {
        if lo > hi || hi > 15 {
            return Err(crate::Error::InvalidArgument(
                "latent_res must satisfy lo <= hi <= 15",
            ));
        }
        hdr.latent_resolution = [lo as u32, hi as u32];
        // Upstream's auto hyperlatent range stays (4, v) — image-size-derived,
        // independent of the latent range — so hyper grids may sit below the
        // coarsest latent level.
        hdr.n_latent_grids =
            (hi - lo + 1) as u32 + hdr.hyperlatent_resolution.map_or(0, |[a, b]| b - a + 1);
    }
    // Upstream: common randomness only on the residue encoder, and only for
    // `--tune wasserstein` (`utils/parsecli.py`).
    hdr.flag_common_randomness = flag_cr;
    let a = crate::decode::arch(&hdr)?;
    let shape = Shape::new(&hdr, &a)?;
    let template = shape.zero_params(&hdr, &a);
    let mut target = Vec::with_capacity(planes.iter().map(|p| p.samples.len()).sum());
    for p in planes {
        target.extend(p.samples.iter().map(|&s| s as f32 / max));
    }
    let wass = opt
        .vgg
        .as_ref()
        .filter(|_| flag_cr)
        .map(|v| Wasserstein::new(v.clone(), &target, h, w));
    // Upstream `img_min_max`: only for I-frames with YUV data — per-plane
    // (min, max) of the normalized samples (420 chroma is scored at its own
    // resolution, like upstream's per-plane `c.min()`).
    let out_init = match frame_data_type {
        crate::FrameDataType::Yuv420 | crate::FrameDataType::Yuv444 => {
            let mut mm = [(0.0f32, 0.0f32); 3];
            for (i, p) in planes.iter().enumerate() {
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for &s in p.samples {
                    let v = s as f32 / max;
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                mm[i] = (mn, mx);
            }
            Some(mm)
        }
        _ => None,
    };
    Ok((
        hdr,
        shape,
        template,
        target,
        wass,
        dist,
        trainer::encoder_gain(opt.lambda),
        w * h,
        out_init,
    ))
}

fn run_training(
    pb: &Problem<'_>,
    template: &crate::train::Params,
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
    stop: &dyn enough::Stop,
) -> crate::Result<(crate::train::Params, crate::train::LossTerms)> {
    let preset = Preset::intra(opt.lambda, opt.start_lr, opt.n_itr, pb.dist);
    let n_pix = (pb.shape.img.0 * pb.shape.img.1) as f64;
    let mut total = 0usize;
    let mut progress =
        |it: usize, _t: &crate::train::LossTerms, v: Option<&crate::train::LossTerms>| {
            total += 1;
            if let Some(v) = v {
                log(alloc::format!(
                    "itr {total:>6} (phase itr {it:>5}): psnr {:.3} dB, rate {:.4} bpp, loss {:.6}",
                    -10.0 * libm::log10(v.mse),
                    v.rate_bits / n_pix,
                    v.loss
                ));
            }
        };
    trainer::encode_training(pb, template, &preset, opt.seed, &mut progress, stop)
}

/// Trains a Cool-chic on an interleaved 8-bit RGB picture. `log` receives one line per
/// validation.
pub fn train_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
) -> crate::Result<TrainReport> {
    let bufs = rgb_planes(rgb, width, height)?;
    let planes: Vec<Plane> = bufs
        .iter()
        .map(|b| Plane {
            width,
            height,
            samples: b,
        })
        .collect();
    train_frame(crate::FrameDataType::Rgb, 8, &planes, opt, log)
}

/// Trains a Cool-chic on a planar frame (same layout rules as [`encode_frame`]).
pub fn train_frame(
    frame_data_type: crate::FrameDataType,
    bitdepth: u8,
    planes: &[Plane<'_>],
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
) -> crate::Result<TrainReport> {
    let (_hdr, shape, template, target, wass, dist, gain, n_pix, out_init) =
        setup(frame_data_type, bitdepth, planes, opt)?;
    let pb = Problem {
        shape: &shape,
        target: &target,
        frame_data_type,
        bitdepth: bitdepth as u32,
        lambda: opt.lambda,
        gain,
        dist,
        wass: wass.as_ref(),
        inter: None,
        out_init,
    };
    let n_pix = n_pix as f64;
    let (_p, l) = run_training(&pb, &template, opt, log, &enough::Unstoppable)?;
    Ok(TrainReport {
        psnr_db: -10.0 * libm::log10(l.mse),
        rate_latent_bpp: l.rate_bits / n_pix,
        loss: l.loss,
    })
}

/// Runs `n` training steps (main-phase settings) and returns milliseconds per step; for
/// benchmarks and profiles.
pub fn bench_train_steps(
    rgb: &[u8],
    width: u32,
    height: u32,
    arch: u8,
    n: usize,
) -> crate::Result<f64> {
    let opt = TrainOptions {
        arch,
        ..Default::default()
    };
    let bufs = rgb_planes(rgb, width, height)?;
    let planes: Vec<Plane> = bufs
        .iter()
        .map(|b| Plane {
            width,
            height,
            samples: b,
        })
        .collect();
    let (_hdr, shape, template, target, wass, dist, gain, _n_pix, out_init) =
        setup(crate::FrameDataType::Rgb, 8, &planes, &opt)?;
    let pb = Problem {
        shape: &shape,
        target: &target,
        frame_data_type: crate::FrameDataType::Rgb,
        bitdepth: 8,
        lambda: opt.lambda,
        gain,
        dist,
        wass: wass.as_ref(),
        inter: None,
        out_init,
    };
    let mut rng = trainer::Rng::new(1);
    let mut p = template.clone();
    trainer::initialize(&mut p, &shape, &mut rng);
    let mut ph = Preset::intra(opt.lambda, 1e-2, 10_000, pb.dist).phases[0].clone();
    ph.max_itr = n;
    ph.freq_valid = n + 1; // one validation at the end only
    let t = std::time::Instant::now();
    let _ = trainer::train(
        &pb,
        p,
        &ph,
        &mut rng,
        &mut |_, _, _| {},
        &enough::Unstoppable,
    )?;
    Ok(t.elapsed().as_secs_f64() * 1e3 / n as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// `Tune::Wasserstein` with embedded miniature feature weights: the whole path
    /// (common-randomness synthesis channels, weighted distortion, stream write)
    /// produces a stream the decoder accepts, carrying `flag_common_randomness`.
    #[test]
    fn wasserstein_encode_writes_common_randomness_stream() {
        // Same (c_in, c_out) structure as `dump_grad.py`'s `miniwass`.
        let dims = [
            (3usize, 4usize),
            (4, 4),
            (4, 8),
            (8, 8),
            (8, 4),
            (4, 4),
            (4, 8),
            (8, 4),
            (4, 4),
            (4, 4),
        ];
        let n: usize = dims.iter().map(|&(ci, co)| co * ci * 9 + co).sum();
        let mut weights = Vec::with_capacity(n);
        let mut s = 0x9e3779b9u32;
        for _ in 0..n {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            weights.push((((s >> 8) & 0xffffff) as f32 / 16777216.0 - 0.5) * 0.1);
        }
        let vgg = Arc::new(Vgg16::from_dims(&dims, &weights).unwrap());
        let (w, h) = (16usize, 16usize);
        let rgb: Vec<u8> = (0..3 * w * h).map(|i| (i * 5) as u8).collect();
        let opt = TrainOptions {
            arch: 3,
            tune: Tune::Wasserstein,
            vgg: Some(vgg),
            rdoq: false,
            ..Default::default()
        };
        let report = encode_rgb8(&rgb, w as u32, h as u32, &opt, &mut |_| {}).unwrap();
        assert!(report.trained.loss.is_finite());
        let (_, rest) = crate::header::VideoHeader::parse(&report.bitstream).unwrap();
        let (_, rest) = crate::header::FrameHeader::parse(rest).unwrap();
        let (ch, _) = crate::header::CoolChicHeader::parse(rest).unwrap();
        assert!(ch.flag_common_randomness);
        let pic = crate::decode(&report.bitstream, &crate::Limits::default()).unwrap();
        assert_eq!((pic.width, pic.height), (w as u32, h as u32));
    }

    /// `Tune::Wasserstein` without [`TrainOptions::vgg`] is an argument error.
    #[test]
    fn wasserstein_tune_requires_vgg_weights() {
        let rgb = vec![0u8; 3 * 16 * 16];
        let e = encode_rgb8(
            &rgb,
            16,
            16,
            &TrainOptions {
                tune: Tune::Wasserstein,
                ..Default::default()
            },
            &mut |_| {},
        );
        assert!(matches!(e, Err(crate::Error::InvalidArgument(_))));
    }

    /// An encoded 4:2:0 frame round-trips: the stream carries `yuv420` frame data
    /// and the decoder returns per-plane buffers at chroma half resolution.
    #[test]
    fn yuv420_encode_round_trips() {
        let (w, h) = (16usize, 16usize);
        let planes_data: Vec<Vec<u16>> = (0..3)
            .map(|c| {
                let (pw, ph) = if c == 0 { (w, h) } else { (w / 2, h / 2) };
                (0..pw * ph)
                    .map(|i| ((i * 17 + c * 91) % 256) as u16)
                    .collect()
            })
            .collect();
        let planes: Vec<Plane> = planes_data
            .iter()
            .enumerate()
            .map(|(c, s)| {
                let (pw, ph) = if c == 0 { (w, h) } else { (w / 2, h / 2) };
                Plane {
                    width: pw as u32,
                    height: ph as u32,
                    samples: s,
                }
            })
            .collect();
        let report = encode_frame(
            crate::FrameDataType::Yuv420,
            8,
            &planes,
            &TrainOptions {
                arch: 3,
                rdoq: false,
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
        let (_, rest) = crate::header::VideoHeader::parse(&report.bitstream).unwrap();
        let (fh, _) = crate::header::FrameHeader::parse(rest).unwrap();
        assert_eq!(fh.frame_data_type, crate::FrameDataType::Yuv420);
        assert_eq!(fh.bitdepth, 8);
        let frames = crate::decode_video(&report.bitstream, &crate::Limits::default()).unwrap();
        assert_eq!(frames.len(), 1);
        // A trained model approximates the target within its latent capacity;
        // the point of this test is the plumbing, not the quality.
        assert!(report.quantized.psnr_db > 10.0);
        let _ = frames;
    }

    /// A stream trained with `latent_res` below full resolution carries the
    /// matching `latent_resolution`/`final_upsampling` header fields and
    /// decodes through `rescale_output` (upstream `--latent_residue lo-hi`).
    #[test]
    fn below_full_res_latents_encode_round_trips() {
        let rgb: Vec<u8> = (0..3 * 16 * 16)
            .map(|i| ((i * 7 + 13) % 256) as u8)
            .collect();
        let report = encode_rgb8(
            &rgb,
            16,
            16,
            &TrainOptions {
                arch: 3,
                rdoq: false,
                latent_res: Some((2, 4)),
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
        let (_, rest) = crate::header::VideoHeader::parse(&report.bitstream).unwrap();
        let (_, rest) = crate::header::FrameHeader::parse(rest).unwrap();
        let (ch, _) = crate::header::CoolChicHeader::parse(rest).unwrap();
        assert_eq!(ch.latent_resolution, [2, 4]);
        assert_eq!(ch.final_upsampling, crate::header::FinalUpsampling::Bicubic);
        let pic = crate::decode(&report.bitstream, &crate::Limits::default()).unwrap();
        assert_eq!((pic.width, pic.height), (16, 16));
        assert!(report.quantized.psnr_db > 10.0);
        // The incompatible combination upstream cannot run is refused early.
        assert!(matches!(
            encode_rgb8(
                &rgb,
                16,
                16,
                &TrainOptions {
                    tune: Tune::Wasserstein,
                    latent_res: Some((2, 4)),
                    vgg: Some(std::sync::Arc::new(
                        Vgg16::from_dims(&[(3, 4); 10], &[0.0; 10 * 4 * 3 * 9 + 40]).unwrap()
                    )),
                    ..Default::default()
                },
                &mut |_| {},
            ),
            Err(crate::Error::InvalidArgument(_))
        ));
    }

    /// A 10-bit 4:4:4 frame encodes with the right header fields and decodes.
    #[test]
    fn yuv444_10bit_encode_round_trips() {
        let (w, h) = (16usize, 16usize);
        let planes_data: Vec<Vec<u16>> = (0..3)
            .map(|c| {
                (0..w * h)
                    .map(|i| ((i * 7 + c * 311) % 1024) as u16)
                    .collect()
            })
            .collect();
        let planes: Vec<Plane> = planes_data
            .iter()
            .map(|s| Plane {
                width: w as u32,
                height: h as u32,
                samples: s,
            })
            .collect();
        let report = encode_frame(
            crate::FrameDataType::Yuv444,
            10,
            &planes,
            &TrainOptions {
                arch: 3,
                rdoq: false,
                ..Default::default()
            },
            &mut |_| {},
        )
        .unwrap();
        let (_, rest) = crate::header::VideoHeader::parse(&report.bitstream).unwrap();
        let (fh, _) = crate::header::FrameHeader::parse(rest).unwrap();
        assert_eq!(fh.frame_data_type, crate::FrameDataType::Yuv444);
        assert_eq!(fh.bitdepth, 10);
        let pic = crate::decode(&report.bitstream, &crate::Limits::default()).unwrap();
        assert_eq!((pic.width, pic.height), (w as u32, h as u32));
    }

    /// A three-frame video (I0, B1, P2 — the minimal GOP exercising every frame
    /// type) encodes to a stream the decoder accepts in coding order: the B
    /// frame trains against both decoded references and writes two payloads.
    #[test]
    fn video_ibp_encodes_and_decodes() {
        let (w, h) = (16usize, 16usize);
        let data: Vec<Vec<u16>> = (0..3)
            .map(|shift| {
                (0..3 * w * h)
                    .map(|i| ((i * 5 + shift * 41) % 256) as u16)
                    .collect()
            })
            .collect();
        let frames: Vec<Vec<Plane>> = data
            .iter()
            .map(|d| {
                (0..3)
                    .map(|c| Plane {
                        width: w as u32,
                        height: h as u32,
                        samples: &d[c * w * h..(c + 1) * w * h],
                    })
                    .collect()
            })
            .collect();
        let views: Vec<&[Plane]> = frames.iter().map(|f| f.as_slice()).collect();
        let opt = TrainOptions {
            arch: 3,
            n_itr: 2000,
            rdoq: false,
            ..Default::default()
        };
        let report = encode_video(
            crate::FrameDataType::Rgb,
            8,
            &views,
            &[0],
            &[2],
            &opt,
            &mut VideoOptions::default(),
            &mut |_| {},
        )
        .unwrap();
        // Coding order is I0, P2, B1.
        assert_eq!(
            report
                .frames
                .iter()
                .map(|f| (f.display_index, f.frame_type))
                .collect::<Vec<_>>(),
            vec![
                (0, crate::header::FrameType::I),
                (2, crate::header::FrameType::P),
                (1, crate::header::FrameType::B)
            ]
        );
        let decoded = crate::decode_video(
            &report.bitstream,
            &crate::Limits {
                max_frames: 8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(decoded.len(), 3);
        // Headers: the B frame carries two references, two global flows and the
        // warp filter size; the P frame one of each.
        let b = &report.frames[2].header;
        let p = &report.frames[1].header;
        assert_eq!(b.frame_type, crate::header::FrameType::B);
        assert_eq!(b.index_references, [0, 2]);
        assert_eq!(b.global_flow.len(), 4);
        assert_eq!(b.warp_filter_size, Some(8));
        assert_eq!(p.frame_type, crate::header::FrameType::P);
        assert_eq!(p.index_references, [0]);
        assert_eq!(p.global_flow.len(), 2);
        assert!(
            decoded
                .iter()
                .all(|f| f.width == w as u32 && f.height == h as u32)
        );
    }

    /// A 4:2:0 I/P/B stream: references and the target are 4:4:4-converted
    /// before global translation and warping; PSNR is evaluated on the stored
    /// (half-resolution chroma) planes like upstream.
    #[test]
    fn video_yuv420_ibp_encodes_and_decodes() {
        let (w, h) = (16usize, 16usize);
        let (cw, ch) = (w / 2, h / 2);
        let data: Vec<Vec<u16>> = (0..3)
            .map(|shift| {
                let mut f = Vec::with_capacity(w * h + 2 * cw * ch);
                f.extend((0..w * h).map(|i| ((i * 7 + shift * 29) % 256) as u16));
                f.extend((0..cw * ch).map(|i| ((i * 3 + shift * 11) % 256) as u16));
                f.extend((0..cw * ch).map(|i| ((i * 5 + shift * 17) % 256) as u16));
                f
            })
            .collect();
        let frames: Vec<Vec<Plane>> = data
            .iter()
            .map(|d| {
                let luma = Plane {
                    width: w as u32,
                    height: h as u32,
                    samples: &d[..w * h],
                };
                let u = Plane {
                    width: cw as u32,
                    height: ch as u32,
                    samples: &d[w * h..w * h + cw * ch],
                };
                let v = Plane {
                    width: cw as u32,
                    height: ch as u32,
                    samples: &d[w * h + cw * ch..],
                };
                alloc::vec![luma, u, v]
            })
            .collect();
        let views: Vec<&[Plane]> = frames.iter().map(|f| f.as_slice()).collect();
        let opt = TrainOptions {
            arch: 3,
            n_itr: 2000,
            rdoq: false,
            ..Default::default()
        };
        let report = encode_video(
            crate::FrameDataType::Yuv420,
            8,
            &views,
            &[0],
            &[2],
            &opt,
            &mut VideoOptions::default(),
            &mut |_| {},
        )
        .unwrap();
        assert!(report.frames.iter().all(|f| f.psnr_db.is_finite()));
        let decoded = crate::decode_video(
            &report.bitstream,
            &crate::Limits {
                max_frames: 8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(decoded.len(), 3);
        assert!(
            decoded
                .iter()
                .all(|f| f.data_type == crate::FrameDataType::Yuv420)
        );
        // Stored chroma at half resolution, like upstream's avg_pool2d.
        assert_eq!(decoded[0].plane_dims[1], (cw as u32, ch as u32));
    }

    /// RDOQ scores each candidate through the inter composite (the partner
    /// model's raw output, quantized flow) rather than the model's own raw
    /// planes — this exercises that path end to end on an I/P stream.
    #[test]
    fn video_ip_rdoq() {
        let (w, h) = (16usize, 16usize);
        let data: Vec<Vec<u16>> = (0..2)
            .map(|shift| {
                (0..3 * w * h)
                    .map(|i| ((i * 7 + shift * 53) % 256) as u16)
                    .collect()
            })
            .collect();
        let frames: Vec<Vec<Plane>> = data
            .iter()
            .map(|d| {
                (0..3)
                    .map(|c| Plane {
                        width: w as u32,
                        height: h as u32,
                        samples: &d[c * w * h..(c + 1) * w * h],
                    })
                    .collect()
            })
            .collect();
        let views: Vec<&[Plane]> = frames.iter().map(|f| f.as_slice()).collect();
        let opt = TrainOptions {
            arch: 3,
            n_itr: 2000,
            rdoq: true,
            ..Default::default()
        };
        let report = encode_video(
            crate::FrameDataType::Rgb,
            8,
            &views,
            &[0],
            &[1],
            &opt,
            &mut VideoOptions::default(),
            &mut |_| {},
        )
        .unwrap();
        assert_eq!(report.frames.len(), 2);
        assert!(report.frames.iter().all(|f| f.psnr_db.is_finite()));
        let decoded = crate::decode_video(
            &report.bitstream,
            &crate::Limits {
                max_frames: 8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(decoded.len(), 2);
    }

    /// An external flow field is learned through the `flow` objective and
    /// injected into odd warm-up candidates (upstream
    /// `guided_motion_pretraining`); the callback sees each inter frame's
    /// 4:4:4 target and globally shifted references in coding order. A
    /// non-default warp filter and a higher bitdepth ride along.
    #[test]
    fn video_external_flow_and_warp_filter() {
        let (w, h) = (16usize, 16usize);
        let data: Vec<Vec<u16>> = (0..3)
            .map(|shift| {
                (0..3 * w * h)
                    .map(|i| ((i * 5 + shift * 37) % 1024) as u16)
                    .collect()
            })
            .collect();
        let frames: Vec<Vec<Plane>> = data
            .iter()
            .map(|d| {
                (0..3)
                    .map(|c| Plane {
                        width: w as u32,
                        height: h as u32,
                        samples: &d[c * w * h..(c + 1) * w * h],
                    })
                    .collect()
            })
            .collect();
        let views: Vec<&[Plane]> = frames.iter().map(|f| f.as_slice()).collect();
        let opt = TrainOptions {
            arch: 3,
            n_itr: 2000,
            rdoq: false,
            ..Default::default()
        };
        // The estimator records its invocations and returns a constant +1/+1
        // pixel flow per reference.
        let calls = core::cell::RefCell::new(Vec::new());
        let mut estimator = |display: u32, target: &[f32], refs: &[&[f32]]| {
            assert_eq!(target.len(), 3 * w * h);
            assert!(refs.iter().all(|r| r.len() == 3 * w * h));
            calls.borrow_mut().push((display, refs.len()));
            Some(
                refs.iter()
                    .map(|_| alloc::vec![1.0f32; 2 * w * h])
                    .collect::<Vec<Vec<f32>>>(),
            )
        };
        let mut vopt = VideoOptions {
            warp_filter_size: Some(4),
            itr_motion_pretrain: 100,
            flow: Some(&mut estimator),
        };
        let report = encode_video(
            crate::FrameDataType::Rgb,
            10,
            &views,
            &[0],
            &[2],
            &opt,
            &mut vopt,
            &mut |_| {},
        )
        .unwrap();
        // The estimator ran once per inter frame, in coding order (P2 then B1),
        // with one field per reference.
        assert_eq!(*calls.borrow(), [(2, 1), (1, 2)]);
        assert!(report.frames.iter().all(|f| f.psnr_db.is_finite()));
        for f in &report.frames {
            if f.frame_type == crate::header::FrameType::I {
                continue;
            }
            assert_eq!(f.header.warp_filter_size, Some(4));
        }
        let decoded = crate::decode_video(
            &report.bitstream,
            &crate::Limits {
                max_frames: 8,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(decoded.len(), 3);
        assert!(decoded.iter().all(|f| f.bitdepth == 10));
    }

    /// Every warp filter size round-trips end to end: the header records it and
    /// the produced stream decodes. Filter 4 is covered by
    /// `video_external_flow_and_warp_filter`; this sweeps 2 (bilinear), 8 and 12
    /// (sinc kernels) through a two-frame I/P encode.
    #[test]
    fn video_warp_filter_variants() {
        let (w, h) = (16usize, 16usize);
        let data: Vec<Vec<u16>> = (0..2)
            .map(|shift| {
                (0..3 * w * h)
                    .map(|i| ((i * 11 + shift * 53) % 256) as u16)
                    .collect()
            })
            .collect();
        let frames: Vec<Vec<Plane>> = data
            .iter()
            .map(|d| {
                (0..3)
                    .map(|c| Plane {
                        width: w as u32,
                        height: h as u32,
                        samples: &d[c * w * h..(c + 1) * w * h],
                    })
                    .collect()
            })
            .collect();
        let views: Vec<&[Plane]> = frames.iter().map(|f| f.as_slice()).collect();
        let opt = TrainOptions {
            arch: 3,
            n_itr: 2000,
            rdoq: false,
            ..Default::default()
        };
        for filter in [2u32, 8, 12] {
            let mut vopt = VideoOptions {
                warp_filter_size: Some(filter),
                itr_motion_pretrain: 100,
                flow: None,
            };
            let report = encode_video(
                crate::FrameDataType::Rgb,
                8,
                &views,
                &[0],
                &[1],
                &opt,
                &mut vopt,
                &mut |_| {},
            )
            .unwrap();
            assert_eq!(report.frames[1].header.warp_filter_size, Some(filter));
            assert!(report.frames.iter().all(|f| f.psnr_db.is_finite()));
            let decoded = crate::decode_video(
                &report.bitstream,
                &crate::Limits {
                    max_frames: 4,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(decoded.len(), 2);
        }
    }

    /// Plane layout, dimension and sample-range violations are argument errors.
    #[test]
    fn frame_input_validation() {
        let ok: Vec<Plane> = (0..3)
            .map(|c| {
                let (pw, ph) = if c == 0 { (16u32, 16u32) } else { (8u32, 8u32) };
                Plane {
                    width: pw,
                    height: ph,
                    samples: &[0u16; 256][..pw as usize * ph as usize],
                }
            })
            .collect();
        let opt = TrainOptions::default();
        let mut log = |_| {};
        // Odd luma dims for yuv420.
        let mut odd = ok.clone();
        odd[0] = Plane {
            width: 15,
            height: 16,
            samples: &[0u16; 240],
        };
        for (fdt, bd, planes, label) in [
            (crate::FrameDataType::Yuv420, 8u8, &odd[..], "odd luma"),
            (crate::FrameDataType::Yuv420, 7, &ok[..], "bitdepth"),
            (crate::FrameDataType::Flow, 8, &ok[..], "flow"),
            (crate::FrameDataType::Rgb, 8, &ok[..], "plane dims"),
        ] {
            assert!(
                matches!(
                    encode_frame(fdt, bd, planes, &opt, &mut log),
                    Err(crate::Error::InvalidArgument(_))
                ),
                "{label}"
            );
        }
        // Out-of-range sample for the declared bitdepth.
        let mut s255 = [0u16; 256];
        s255[0] = 256;
        let oor = [
            Plane {
                width: 16,
                height: 16,
                samples: &s255,
            },
            ok[1],
            ok[2],
        ];
        assert!(matches!(
            encode_frame(crate::FrameDataType::Rgb, 8, &oor, &opt, &mut log),
            Err(crate::Error::InvalidArgument(_))
        ));
        // Wasserstein on 4:2:0 is rejected.
        let dims = [(3usize, 4usize); 10];
        let weights = vec![0.01f32; dims.iter().map(|&(ci, co)| co * ci * 9 + co).sum()];
        let vgg = std::sync::Arc::new(Vgg16::from_dims(&dims, &weights).unwrap());
        assert!(matches!(
            encode_frame(
                crate::FrameDataType::Yuv420,
                8,
                &ok,
                &TrainOptions {
                    tune: Tune::Wasserstein,
                    vgg: Some(vgg),
                    ..Default::default()
                },
                &mut log
            ),
            Err(crate::Error::InvalidArgument(_))
        ));
    }
}
