//! Bitstream writing (upstream `coolchic/bitstream/encode.py` and the encode direction of
//! `coolchic/bitstream/component/coolchic.py`).

use alloc::vec::Vec;

use crate::arm::FixedArm;
use crate::decode::{self, GridCoder, Networks};
use crate::error::{Error, Result};
use crate::expgolomb;
use crate::header::{CoolChicHeader, FrameHeader, VideoHeader};
use crate::latent::{self, GridEncoder, InterContext};
use crate::rangecoder::{RangeDecoder, RangeEncoder};

/// Encodes known grids (finest first) into one range coder, in coding order.
struct LatentsEncoder<'a> {
    rc: RangeEncoder,
    source: &'a [latent::Grid],
}

impl GridCoder for LatentsEncoder<'_> {
    fn grid(
        &mut self,
        idx: usize,
        arm: &FixedArm,
        window: &[usize],
        inter: Option<&InterContext>,
        (h, w): (usize, usize),
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<latent::Grid> {
        let g = &self.source[idx];
        if (g.h, g.w) != (h, w) {
            return Err(Error::Invalid("latent grid size"));
        }
        let mut enc = GridEncoder {
            rc: &mut self.rc,
            values: &g.data,
            w,
        };
        latent::code_grid(&mut enc, arm, window, inter, h, w, check)
    }
}

/// Serializes one Cool-chic: header (with the payload sizes filled in), network
/// parameters, latents.
pub(crate) fn write_coolchic(
    h: &CoolChicHeader,
    nets: &Networks,
    params: &[i64],
    grids: &[latent::Grid],
) -> Result<Vec<u8>> {
    let a = decode::arch(h)?;
    let orders = decode::param_orders(h, &a)?;
    if orders.len() != params.len() {
        return Err(Error::Invalid(
            "parameter count does not match the architecture",
        ));
    }
    let (nn_bytes, pad) = expgolomb::encode(params, &orders);
    let mut le = LatentsEncoder {
        rc: RangeEncoder::new(),
        source: grids,
    };
    decode::code_latents(h, &a, nets, &mut le)?;
    let latent_bytes = le.rc.finish();

    let mut h = h.clone();
    h.nn_n_bytes = u32::try_from(nn_bytes.len())
        .ok()
        .filter(|&n| n < 1 << 14)
        .ok_or(Error::LimitExceeded("network bytes"))?;
    h.nn_n_bit_pad = pad;
    h.n_bytes_latent = u32::try_from(latent_bytes.len())
        .ok()
        .filter(|&n| n < 1 << 28)
        .ok_or(Error::LimitExceeded("latent bytes"))?;
    let mut out = h.to_bytes();
    out.extend_from_slice(&nn_bytes);
    out.extend_from_slice(&latent_bytes);
    Ok(out)
}

/// Serializes one cool-chic payload (header + expgolomb params + ARM-coded
/// latents): the `write_image` inner step, shared with video frames.
pub(crate) fn write_coolchic_full(
    h: &CoolChicHeader,
    params: &[i64],
    grids: &[latent::Grid],
) -> Result<Vec<u8>> {
    let a = decode::arch(h)?;
    let orders = decode::param_orders(h, &a)?;
    // The fixed-point ARM is built from the integers exactly as the decoder will build it.
    let (nn_bytes, pad) = expgolomb::encode(params, &orders);
    let mut hh = h.clone();
    hh.nn_n_bit_pad = pad;
    let (nets, back) = decode::build_networks(&hh, &a, &nn_bytes)?;
    debug_assert_eq!(back, params);
    write_coolchic(h, &nets, params, grids)
}

/// Writes one video frame: frame header, residue payload, then the motion
/// payload for inter frames (upstream `encode_frame`'s stream order).
pub(crate) fn write_frame(fh: &FrameHeader, ccs: &[Vec<u8>]) -> Vec<u8> {
    let mut out = fh.to_bytes();
    for cc in ccs {
        out.extend_from_slice(cc);
    }
    out
}

/// Writes a single-image stream from quantized integers: `params` in stream order,
/// `grids` finest first. The header's quantization fields must already be set.
pub(crate) fn write_image(
    h: &CoolChicHeader,
    data_type: crate::header::FrameDataType,
    bitdepth: u32,
    params: &[i64],
    grids: &[latent::Grid],
) -> Result<Vec<u8>> {
    let vh = VideoHeader {
        n_frames: 1,
        intra_pos: alloc::vec![0],
        p_pos: Vec::new(),
    };
    let fh = FrameHeader {
        display_index: 0,
        frame_type: crate::header::FrameType::I,
        frame_data_type: data_type,
        bitdepth,
        index_references: Vec::new(),
        global_flow: Vec::new(),
        warp_filter_size: None,
    };
    let mut out = vh.to_bytes();
    out.extend_from_slice(&fh.to_bytes());
    out.extend_from_slice(&write_coolchic_full(h, params, grids)?);
    Ok(out)
}

/// Decodes a single-image stream to its integers and writes it back. Every byte must
/// come back identical: proves the writer and the decoder agree on the format.
pub(crate) fn reencode(data: &[u8]) -> Result<Vec<u8>> {
    let (vh, rest) = VideoHeader::parse(data)?;
    let (fh, rest) = FrameHeader::parse(rest)?;
    let (ch, rest) = CoolChicHeader::parse(rest)?;
    let a = decode::arch(&ch)?;
    let (nn_bytes, rest) = rest.split_at(ch.nn_n_bytes as usize);
    let lat_bytes = &rest[..ch.n_bytes_latent as usize];
    let (nets, params) = decode::build_networks(&ch, &a, nn_bytes)?;
    let mut rd = RangeDecoder::new(lat_bytes)?;
    let grids = decode::code_latents(&ch, &a, &nets, &mut rd)?;

    let mut out = vh.to_bytes();
    out.extend_from_slice(&fh.to_bytes());
    out.extend_from_slice(&write_coolchic(&ch, &nets, &params, &grids)?);
    Ok(out)
}

#[cfg(all(test, feature = "oracle"))]
mod tests {
    #[test]
    fn reencode_is_byte_identical() {
        let mut n = 0;
        for name in crate::oracle_fixtures::cases("decoder") {
            let data = crate::oracle_fixtures::read("decoder", &std::format!("{name}.cool"));
            let again = super::reencode(&data).unwrap();
            assert!(
                again == data,
                "{name}: re-encoded {} bytes vs {} original",
                again.len(),
                data.len()
            );
            n += 1;
        }
        assert!(n > 0);
        std::eprintln!("{n} streams re-encoded byte for byte");
    }
}
