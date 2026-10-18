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

/// Trains a Cool-chic on an interleaved 8-bit RGB picture. `log` receives one line per
/// validation.
pub fn train_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    opt: &TrainOptions,
    log: &mut dyn FnMut(String),
) -> crate::Result<TrainReport> {
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
    let pb = Problem {
        shape: &shape,
        target: &target,
        bitdepth: 8,
        lambda: opt.lambda,
        gain: trainer::encoder_gain(opt.lambda),
    };
    let preset = Preset::intra(opt.lambda, opt.start_lr, opt.n_itr);
    let n_pix = (w * h) as f64;
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
    let (_p, l) = trainer::encode_training(&pb, &template, &preset, opt.seed, &mut progress);
    Ok(TrainReport {
        psnr_db: -10.0 * libm::log10(l.mse),
        rate_latent_bpp: l.rate_bits / n_pix,
        loss: l.loss,
    })
}
