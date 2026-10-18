//! Experimental encoder entry points. **Not covered by semver**: the encoder API is still
//! being designed and will go through review before release.

use alloc::string::String;
use alloc::vec::Vec;

use crate::train::Shape;
use crate::train::config::{self, Arch};
use crate::train::trainer::{self, Preset, Problem};

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
}

impl Default for TrainOptions {
    fn default() -> Self {
        Self {
            lambda: 1e-3,
            start_lr: 1e-2,
            n_itr: 10_000,
            arch: 1,
            seed: 0,
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

/// Encodes an interleaved 8-bit RGB picture: training, network quantization, bitstream.
pub fn encode_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
) -> crate::Result<EncodeReport> {
    let (hdr, shape, template, target, pb_lambda, gain) = setup(rgb, width, height, opt)?;
    let pb = Problem {
        shape: &shape,
        target: &target,
        bitdepth: 8,
        lambda: pb_lambda,
        gain,
    };
    let n_pix = (width as usize * height as usize) as f64;
    let (p, l) = run_training(&pb, &template, opt, log);
    let trained = TrainReport {
        psnr_db: -10.0 * libm::log10(l.mse),
        rate_latent_bpp: l.rate_bits / n_pix,
        loss: l.loss,
    };
    let has_ifce = p.ifce.iter().any(|m| m.is_some());
    let (pq, q) = crate::train::quantize::quantize_model(&pb, &p, has_ifce);
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
    let bitstream = crate::write::write_image(&h, crate::FrameDataType::Rgb, 8, &ints, &grids)?;
    Ok(EncodeReport {
        trained,
        quantized,
        bitstream,
    })
}

type Setup = (
    crate::header::CoolChicHeader,
    Shape,
    crate::train::Params,
    Vec<f32>,
    f32,
    f32,
);

fn setup(rgb: &[u8], width: u32, height: u32, opt: &TrainOptions) -> crate::Result<Setup> {
    let (w, h) = (width as usize, height as usize);
    if rgb.len() != 3 * w * h {
        return Err(crate::Error::Invalid("rgb buffer size"));
    }
    let arch = match opt.arch {
        0 => Arch::VeryHigh,
        1 => Arch::High,
        2 => Arch::Medium,
        _ => Arch::Low,
    };
    let hdr = config::header(arch, height, width);
    let a = crate::decode::arch(&hdr)?;
    let shape = Shape::new(&hdr, &a)?;
    let template = shape.zero_params(&hdr, &a);
    let mut target = Vec::with_capacity(3 * w * h);
    for c in 0..3 {
        for i in 0..w * h {
            target.push(rgb[3 * i + c] as f32 / 255.0);
        }
    }
    Ok((
        hdr,
        shape,
        template,
        target,
        opt.lambda,
        trainer::encoder_gain(opt.lambda),
    ))
}

fn run_training(
    pb: &Problem<'_>,
    template: &crate::train::Params,
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
) -> (crate::train::Params, crate::train::LossTerms) {
    let preset = Preset::intra(opt.lambda, opt.start_lr, opt.n_itr);
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
    trainer::encode_training(pb, template, &preset, opt.seed, &mut progress)
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
    let (_hdr, shape, template, target, lambda, gain) = setup(rgb, width, height, opt)?;
    let pb = Problem {
        shape: &shape,
        target: &target,
        bitdepth: 8,
        lambda,
        gain,
    };
    let n_pix = (width as usize * height as usize) as f64;
    let (_p, l) = run_training(&pb, &template, opt, log);
    Ok(TrainReport {
        psnr_db: -10.0 * libm::log10(l.mse),
        rate_latent_bpp: l.rate_bits / n_pix,
        loss: l.loss,
    })
}
