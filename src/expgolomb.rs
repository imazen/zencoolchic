//! Signed exp-Golomb coding of the quantized network parameters
//! (upstream `coolchic/bitstream/neuralnet/expgolomb.py`).
//!
//! The payload is prefixed with `n_padding_bits` zero bits so it fills whole bytes. Each
//! value has its own order `k`: a unary prefix of zeros, a `1`, `zeros` more bits (the
//! quotient + 1), then `k` remainder bits. The unsigned code maps back to signed as
//! odd → positive, even → non-positive.

use alloc::vec::Vec;

use crate::bitio::BitReader;
use crate::error::{Error, Result};

/// Decodes `orders.len()` values; `orders[i]` is the exp-Golomb order of value `i`.
pub(crate) fn decode(data: &[u8], n_padding_bits: u32, orders: &[u32]) -> Result<Vec<i64>> {
    const W: &str = "network parameters";
    let mut r = BitReader::new(data);
    r.skip(n_padding_bits as usize, W)?;
    let mut out = Vec::with_capacity(orders.len());
    for &k in orders {
        let mut zeros = 0u32;
        while r.bit(W)? == 0 {
            zeros += 1;
            if zeros > 40 {
                return Err(Error::Invalid("exp-golomb prefix too long"));
            }
        }
        // The `1` just read is the top bit of the (zeros + 1)-bit quotient field.
        let low = r.bits(zeros, W)? as u64;
        let quotient = ((1u64 << zeros) | low) - 1;
        let remainder = if k == 0 { 0 } else { r.bits(k, W)? as u64 };
        let value = (quotient << k) + remainder;
        let v = value as i64;
        out.push(if value % 2 == 1 {
            (v + 1) / 2
        } else {
            -(v / 2)
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Port of upstream `encode_exp_golomb`, used only to round-trip the decoder.
    fn encode(data: &[i64], orders: &[u32]) -> (Vec<u8>, u32) {
        let mut bits: Vec<u8> = Vec::new();
        for (&x, &k) in data.iter().zip(orders) {
            let x = if x <= 0 { -2 * x } else { 2 * x - 1 } as u64;
            let x = x + (1u64 << k) - 1;
            let n_leading_zeros = 63 - (x + 1).leading_zeros();
            let mut code: Vec<u8> = Vec::new();
            code.extend(core::iter::repeat_n(0, n_leading_zeros as usize));
            let len = 64 - (x + 1).leading_zeros();
            for i in (0..len).rev() {
                code.push(((x + 1) >> i & 1) as u8);
            }
            bits.extend_from_slice(&code[k as usize..]);
        }
        let pad = ((8 - bits.len() % 8) % 8) as u32;
        let mut all = alloc::vec![0u8; pad as usize];
        all.extend(bits);
        let bytes = all
            .chunks(8)
            .map(|c| c.iter().fold(0u8, |a, &b| (a << 1) | b))
            .collect();
        (bytes, pad)
    }

    #[test]
    fn round_trip_all_orders() {
        let mut data = Vec::new();
        let mut orders = Vec::new();
        for k in 0..=12u32 {
            for x in -300i64..=300 {
                data.push(x);
                orders.push(k);
            }
        }
        let (bytes, pad) = encode(&data, &orders);
        assert_eq!(decode(&bytes, pad, &orders).unwrap(), data);
    }
}
