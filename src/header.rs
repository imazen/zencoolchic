//! Video, frame and Cool-chic headers (upstream `coolchic/bitstream/header/`).
//!
//! Every header is a run of MSB-first fields followed by `n_bytes_header` (16 bits), then
//! a variable-length part whose layout depends on the fixed part, padded to a whole number
//! of bytes. `n_bytes_header` gives the total header size, so the payload starts there.

use alloc::vec::Vec;

use crate::bitio::BitReader;
#[cfg(feature = "encoder")]
use crate::bitio::BitWriter;
use crate::error::{Error, Result};

/// Video header: the coding structure (which frames are intra / P / B).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoHeader {
    pub n_frames: u32,
    pub intra_pos: Vec<u32>,
    pub p_pos: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    I,
    P,
    B,
}

/// Sample layout of a decoded frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDataType {
    Rgb,
    Yuv420,
    Yuv444,
    /// Optical flow (only produced inside the video coder).
    Flow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    pub display_index: u32,
    pub frame_type: FrameType,
    pub frame_data_type: FrameDataType,
    pub bitdepth: u32,
    pub index_references: Vec<u32>,
    pub global_flow: Vec<i32>,
    pub warp_filter_size: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalUpsampling {
    Nearest,
    Bilinear,
    Bicubic,
}

/// One synthesis layer, `<out_ft>-<k_size>-<linear|residual>-<none|relu>` upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynLayer {
    pub out_ft: u32,
    pub k_size: u32,
    pub residual: bool,
    pub relu: bool,
}

/// A value per (module, weight|bias), in upstream `DescriptorCoolChic` field order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerModule<T> {
    pub arm: [T; 2],
    pub ifce: [T; 2],
    pub upsampling: [T; 2],
    pub synthesis: [T; 2],
}

/// Header of one Cool-chic (one overfitted decoder: latents + ARM + upsampling + synthesis).
#[derive(Debug, Clone, PartialEq)]
pub struct CoolChicHeader {
    pub linear_stabiliser_synth: bool,
    pub ups_k_size: u32,
    pub ups_preconcat_k_size: u32,
    pub output_feature_ifce: u32,
    pub spatial_context_arm: u32,
    pub linear_stabiliser_arm: bool,
    pub n_hidden_layers_arm: u32,
    /// (height, width)
    pub img_size: [u32; 2],
    pub latent_resolution: [u32; 2],
    pub n_latent_grids: u32,
    pub flag_common_randomness: bool,
    pub final_upsampling: FinalUpsampling,
    /// Quantization step of each module's weights / biases (always a power of two).
    pub nn_q_step: PerModule<f32>,
    /// log2 of `nn_q_step` (exact integer).
    pub nn_q_shift: PerModule<i32>,
    pub nn_expgol_cnt: PerModule<u32>,
    pub nn_n_bytes: u32,
    pub nn_n_bit_pad: u32,
    pub n_bytes_latent: u32,
    pub ifce_resolution: Option<[u32; 2]>,
    pub hyperlatent_resolution: Option<[u32; 2]>,
    pub layers_synthesis: Vec<SynLayer>,
}

fn take<'a>(data: &'a [u8], n_bytes_header: u32, what: &'static str) -> Result<&'a [u8]> {
    let n = n_bytes_header as usize;
    if n > data.len() {
        return Err(Error::Truncated(what));
    }
    Ok(&data[n..])
}

impl VideoHeader {
    /// Parses the header and returns it with the remaining bytes.
    pub fn parse(data: &[u8]) -> Result<(Self, &[u8])> {
        const W: &str = "video header";
        let mut r = BitReader::new(data);
        let n_frames = r.bits(12, W)?;
        let n_intras = r.bits(12, W)?;
        let n_p = r.bits(12, W)?;
        let n_bytes_header = r.bits(16, W)?;
        let intra_pos = (0..n_intras)
            .map(|_| r.bits(12, W))
            .collect::<Result<Vec<_>>>()?;
        let p_pos = (0..n_p)
            .map(|_| r.bits(12, W))
            .collect::<Result<Vec<_>>>()?;
        let rest = take(data, n_bytes_header, W)?;
        Ok((
            Self {
                n_frames,
                intra_pos,
                p_pos,
            },
            rest,
        ))
    }
}

impl FrameHeader {
    pub fn parse(data: &[u8]) -> Result<(Self, &[u8])> {
        const W: &str = "frame header";
        let mut r = BitReader::new(data);
        let display_index = r.bits(12, W)?;
        let frame_type = match r.bits(2, W)? {
            0 => FrameType::I,
            1 => FrameType::P,
            2 => FrameType::B,
            _ => return Err(Error::Invalid("frame_type")),
        };
        let frame_data_type = match r.bits(2, W)? {
            0 => FrameDataType::Rgb,
            1 => FrameDataType::Yuv420,
            2 => FrameDataType::Yuv444,
            _ => FrameDataType::Flow,
        };
        let bitdepth_idx = r.bits(4, W)?;
        if bitdepth_idx > 8 {
            return Err(Error::Invalid("bitdepth"));
        }
        let bitdepth = 8 + bitdepth_idx;
        let n_bytes_header = r.bits(16, W)?;
        let n_refs = match frame_type {
            FrameType::I => 0,
            FrameType::P => 1,
            FrameType::B => 2,
        };
        let index_references = (0..n_refs)
            .map(|_| r.bits(12, W))
            .collect::<Result<Vec<_>>>()?;
        let global_flow = (0..2 * n_refs)
            .map(|_| r.signed(14, W))
            .collect::<Result<Vec<_>>>()?;
        let warp_filter_size = if n_refs > 0 {
            Some(r.bits(4, W)?)
        } else {
            None
        };
        let rest = take(data, n_bytes_header, W)?;
        Ok((
            Self {
                display_index,
                frame_type,
                frame_data_type,
                bitdepth,
                index_references,
                global_flow,
                warp_filter_size,
            },
            rest,
        ))
    }
}

/// Smallest log2 quantization step per (module, weight|bias); the header stores an index
/// on top of it (upstream `POSSIBLE_Q_STEP`).
pub(crate) const Q_SHIFT_MIN: PerModule<i32> = PerModule {
    arm: [-8, -16],
    ifce: [-8, -16],
    upsampling: [-12, 0],
    synthesis: [-12, -24],
};

impl CoolChicHeader {
    pub fn parse(data: &[u8]) -> Result<(Self, &[u8])> {
        const W: &str = "cool-chic header";
        let mut r = BitReader::new(data);
        let linear_stabiliser_synth = r.bits(1, W)? == 1;
        let n_layer_synthesis = r.bits(3, W)?;
        let ups_k_size = r.bits(4, W)?;
        let ups_preconcat_k_size = r.bits(4, W)?;
        let output_feature_ifce = r.bits(5, W)?;
        let spatial_context_arm = r.bits(6, W)?;
        let linear_stabiliser_arm = r.bits(1, W)? == 1;
        let n_hidden_layers_arm = r.bits(3, W)?;
        let img_size = [r.bits(14, W)?, r.bits(14, W)?];
        let latent_resolution = [r.bits(4, W)?, r.bits(4, W)?];
        let n_latent_grids = r.bits(5, W)?;
        let flag_hyperlatent = r.bits(1, W)? == 1;
        let flag_common_randomness = r.bits(1, W)? == 1;
        let final_upsampling = match r.bits(2, W)? {
            0 => FinalUpsampling::Nearest,
            1 => FinalUpsampling::Bilinear,
            2 => FinalUpsampling::Bicubic,
            _ => return Err(Error::Invalid("final_upsampling_type")),
        };

        let mut q_idx = [0u32; 8];
        for v in &mut q_idx {
            *v = r.bits(5, W)?;
        }
        let mut eg = [0u32; 8];
        for v in &mut eg {
            *v = r.bits(4, W)?;
            if *v > 12 {
                return Err(Error::Invalid("nn_expgol_cnt"));
            }
        }
        let mins = [
            Q_SHIFT_MIN.arm[0],
            Q_SHIFT_MIN.arm[1],
            Q_SHIFT_MIN.ifce[0],
            Q_SHIFT_MIN.ifce[1],
            Q_SHIFT_MIN.upsampling[0],
            Q_SHIFT_MIN.upsampling[1],
            Q_SHIFT_MIN.synthesis[0],
            Q_SHIFT_MIN.synthesis[1],
        ];
        let mut shift = [0i32; 8];
        for i in 0..8 {
            // Every list of possible steps ends at 2^0.
            if q_idx[i] as i32 > -mins[i] {
                return Err(Error::Invalid("nn_q_step index"));
            }
            shift[i] = mins[i] + q_idx[i] as i32;
        }
        let per = |a: [i32; 8]| PerModule {
            arm: [a[0], a[1]],
            ifce: [a[2], a[3]],
            upsampling: [a[4], a[5]],
            synthesis: [a[6], a[7]],
        };
        let nn_q_shift = per(shift);
        let step = |s: i32| libm::exp2f(s as f32);
        let nn_q_step = PerModule {
            arm: nn_q_shift.arm.map(step),
            ifce: nn_q_shift.ifce.map(step),
            upsampling: nn_q_shift.upsampling.map(step),
            synthesis: nn_q_shift.synthesis.map(step),
        };
        let nn_expgol_cnt = PerModule {
            arm: [eg[0], eg[1]],
            ifce: [eg[2], eg[3]],
            upsampling: [eg[4], eg[5]],
            synthesis: [eg[6], eg[7]],
        };
        let nn_n_bytes = r.bits(14, W)?;
        let nn_n_bit_pad = r.bits(3, W)?;
        let n_bytes_latent = r.bits(28, W)?;
        let n_bytes_header = r.bits(16, W)?;

        let ifce_resolution = if output_feature_ifce > 0 {
            Some([r.bits(4, W)?, r.bits(4, W)?])
        } else {
            None
        };
        let hyperlatent_resolution = if flag_hyperlatent {
            Some([r.bits(4, W)?, r.bits(4, W)?])
        } else {
            None
        };
        let mut layers_synthesis = Vec::with_capacity(n_layer_synthesis as usize);
        for _ in 0..n_layer_synthesis {
            let out_ft = r.bits(7, W)?;
            let k_size = r.bits(4, W)?;
            let residual = r.bits(1, W)? == 1;
            let relu = r.bits(1, W)? == 1;
            layers_synthesis.push(SynLayer {
                out_ft,
                k_size,
                residual,
                relu,
            });
        }
        let rest = take(data, n_bytes_header, W)?;
        Ok((
            Self {
                linear_stabiliser_synth,
                ups_k_size,
                ups_preconcat_k_size,
                output_feature_ifce,
                spatial_context_arm,
                linear_stabiliser_arm,
                n_hidden_layers_arm,
                img_size,
                latent_resolution,
                n_latent_grids,
                flag_common_randomness,
                final_upsampling,
                nn_q_step,
                nn_q_shift,
                nn_expgol_cnt,
                nn_n_bytes,
                nn_n_bit_pad,
                n_bytes_latent,
                ifce_resolution,
                hyperlatent_resolution,
                layers_synthesis,
            },
            rest,
        ))
    }
}

#[cfg(feature = "encoder")]
/// Bits of `n_bytes_header` itself, written at the end of every fixed part.
const N_BYTES_HEADER_BITS: usize = 16;

#[cfg(feature = "encoder")]
/// Finishes a header: `fixed` is written, then `n_bytes_header`, then `var`; the total
/// size (rounded up to bytes) is what `n_bytes_header` holds.
fn assemble(fixed: BitWriter, var: BitWriter) -> Vec<u8> {
    let total = fixed.len_bits() + N_BYTES_HEADER_BITS + var.len_bits();
    let n_bytes = total.div_ceil(8);
    let mut w = BitWriter::default();
    copy_bits(&mut w, &fixed);
    w.bits(n_bytes as u64, 16);
    copy_bits(&mut w, &var);
    w.into_bytes()
}

#[cfg(feature = "encoder")]
fn copy_bits(dst: &mut BitWriter, src: &BitWriter) {
    let n = src.len_bits();
    let bytes = src.clone_bytes();
    for i in 0..n {
        dst.bit((bytes[i / 8] >> (7 - i % 8)) & 1 == 1);
    }
}

#[cfg(feature = "encoder")]
impl VideoHeader {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut f = BitWriter::default();
        f.bits(self.n_frames as u64, 12);
        f.bits(self.intra_pos.len() as u64, 12);
        f.bits(self.p_pos.len() as u64, 12);
        let mut v = BitWriter::default();
        for &p in self.intra_pos.iter().chain(&self.p_pos) {
            v.bits(p as u64, 12);
        }
        assemble(f, v)
    }
}

#[cfg(feature = "encoder")]
impl FrameHeader {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut f = BitWriter::default();
        f.bits(self.display_index as u64, 12);
        f.bits(self.frame_type as u64, 2);
        f.bits(self.frame_data_type as u64, 2);
        f.bits((self.bitdepth - 8) as u64, 4);
        let mut v = BitWriter::default();
        for &r in &self.index_references {
            v.bits(r as u64, 12);
        }
        for &g in &self.global_flow {
            v.signed(g as i64, 14);
        }
        if let Some(s) = self.warp_filter_size {
            v.bits(s as u64, 4);
        }
        assemble(f, v)
    }
}

#[cfg(feature = "encoder")]
impl CoolChicHeader {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut f = BitWriter::default();
        f.bits(self.linear_stabiliser_synth as u64, 1);
        f.bits(self.layers_synthesis.len() as u64, 3);
        f.bits(self.ups_k_size as u64, 4);
        f.bits(self.ups_preconcat_k_size as u64, 4);
        f.bits(self.output_feature_ifce as u64, 5);
        f.bits(self.spatial_context_arm as u64, 6);
        f.bits(self.linear_stabiliser_arm as u64, 1);
        f.bits(self.n_hidden_layers_arm as u64, 3);
        for &s in &self.img_size {
            f.bits(s as u64, 14);
        }
        for &r in &self.latent_resolution {
            f.bits(r as u64, 4);
        }
        f.bits(self.n_latent_grids as u64, 5);
        f.bits(self.hyperlatent_resolution.is_some() as u64, 1);
        f.bits(self.flag_common_randomness as u64, 1);
        f.bits(self.final_upsampling as u64, 2);
        let flat = |p: &PerModule<i32>| [p.arm, p.ifce, p.upsampling, p.synthesis].concat();
        for (s, m) in flat(&self.nn_q_shift).into_iter().zip(flat(&Q_SHIFT_MIN)) {
            f.bits((s - m) as u64, 5);
        }
        let e = &self.nn_expgol_cnt;
        for c in [e.arm, e.ifce, e.upsampling, e.synthesis].concat() {
            f.bits(c as u64, 4);
        }
        f.bits(self.nn_n_bytes as u64, 14);
        f.bits(self.nn_n_bit_pad as u64, 3);
        f.bits(self.n_bytes_latent as u64, 28);
        let mut v = BitWriter::default();
        for r in self
            .ifce_resolution
            .iter()
            .chain(&self.hyperlatent_resolution)
        {
            v.bits(r[0] as u64, 4);
            v.bits(r[1] as u64, 4);
        }
        for l in &self.layers_synthesis {
            v.bits(l.out_ft as u64, 7);
            v.bits(l.k_size as u64, 4);
            v.bits(l.residual as u64, 1);
            v.bits(l.relu as u64, 1);
        }
        assemble(f, v)
    }
}
