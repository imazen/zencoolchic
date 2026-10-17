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
