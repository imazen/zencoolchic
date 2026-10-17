//! MSB-first bit reader, the bit order of Cool-chic's headers and exp-Golomb payloads.

use crate::error::{Error, Result};

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    /// Position in bits from the start of `data`.
    pos: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    pub(crate) fn bit(&mut self, what: &'static str) -> Result<u32> {
        let byte = *self.data.get(self.pos / 8).ok_or(Error::Truncated(what))?;
        let b = (byte >> (7 - (self.pos % 8))) & 1;
        self.pos += 1;
        Ok(b as u32)
    }

    /// Reads `n` bits (n <= 32) as an unsigned big-endian integer.
    pub(crate) fn bits(&mut self, n: u32, what: &'static str) -> Result<u32> {
        debug_assert!(n <= 32);
        let mut v: u64 = 0;
        for _ in 0..n {
            v = (v << 1) | self.bit(what)? as u64;
        }
        Ok(v as u32)
    }

    /// Sign-magnitude value: one sign bit (1 = negative) then `n - 1` magnitude bits.
    pub(crate) fn signed(&mut self, n: u32, what: &'static str) -> Result<i32> {
        let neg = self.bit(what)? == 1;
        let mag = self.bits(n - 1, what)? as i32;
        Ok(if neg { -mag } else { mag })
    }

    pub(crate) fn skip(&mut self, n: usize, what: &'static str) -> Result<()> {
        if self.pos + n > self.data.len() * 8 {
            return Err(Error::Truncated(what));
        }
        self.pos += n;
        Ok(())
    }
}

#[cfg(feature = "encoder")]
/// MSB-first bit writer; [`BitWriter::into_bytes`] pads the last byte with zeros.
#[derive(Default)]
pub(crate) struct BitWriter {
    bytes: alloc::vec::Vec<u8>,
    n_bits: usize,
}

#[cfg(feature = "encoder")]
impl BitWriter {
    pub(crate) fn bit(&mut self, b: bool) {
        if self.n_bits % 8 == 0 {
            self.bytes.push(0);
        }
        if b {
            *self.bytes.last_mut().unwrap() |= 0x80 >> (self.n_bits % 8);
        }
        self.n_bits += 1;
    }

    /// Writes the low `n` bits of `v`, most significant first.
    pub(crate) fn bits(&mut self, v: u64, n: u32) {
        for i in (0..n).rev() {
            self.bit((v >> i) & 1 == 1);
        }
    }

    /// Sign bit (1 = negative) then `n - 1` magnitude bits.
    pub(crate) fn signed(&mut self, v: i64, n: u32) {
        self.bit(v < 0);
        self.bits(v.unsigned_abs(), n - 1);
    }

    pub(crate) fn len_bits(&self) -> usize {
        self.n_bits
    }

    pub(crate) fn clone_bytes(&self) -> alloc::vec::Vec<u8> {
        self.bytes.clone()
    }

    pub(crate) fn into_bytes(self) -> alloc::vec::Vec<u8> {
        self.bytes
    }
}
