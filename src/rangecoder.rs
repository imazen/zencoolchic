//! Latent entropy decoder: the range coder and leaky-quantized Laplace model of
//! `constriction` 0.4.2 (`stream::queue::DefaultRangeDecoder` with
//! `stream::model::QuantizedLaplace`), which upstream Cool-chic calls from Python.
//!
//! - Words are `u32`, read little-endian from the byte payload (upstream does
//!   `np.frombuffer(raw_bytes, dtype=np.uint32)`); the state is `u64`; probabilities have
//!   24 bits of precision.
//! - Symbols live in `[-AC_MAX_VAL, AC_MAX_VAL - 1]`. Each symbol keeps at least one unit of
//!   probability ("leaky"): `left(s) = floor(free_weight * F(s - 0.5)) + (s - min)` with `F`
//!   the Laplace CDF evaluated in f64.
//!
//! Decoding only needs `left(s)` to match the encoder's bit for bit: the symbol is the unique
//! `s` with `left(s) <= quantile < left(s + 1)`, however it is searched for.

use crate::error::{Error, Result};
use crate::scale_table::{N_SCALE, SCALE_BITS};

pub(crate) const AC_MAX_VAL: i32 = 64;
const MIN_SYMBOL: i32 = -AC_MAX_VAL;
const MAX_SYMBOL: i32 = AC_MAX_VAL - 1;
const PRECISION: u32 = 24;
/// `(2^24 - 1) - (MAX_SYMBOL - MIN_SYMBOL)`: probability mass left after reserving one unit
/// per symbol.
const FREE_WEIGHT: f64 = ((1u32 << PRECISION) - 1 - (MAX_SYMBOL - MIN_SYMBOL) as u32) as f64;

/// Number of distinct expectations: `[-64, 64 - 1/256]` in steps of 1/256.
pub(crate) const N_MU: usize = 32768;
pub(crate) const N_SCALE_IDX: usize = N_SCALE;

/// Expectation for table index `i` (exact in f32).
#[inline]
pub(crate) fn mu_of(i: usize) -> f32 {
    -64.0 + i as f32 / 256.0
}

#[inline]
pub(crate) fn scale_of(i: usize) -> f32 {
    f32::from_bits(SCALE_BITS[i])
}

/// `probability::distribution::Laplace::distribution` (probability 0.20.3).
#[inline]
fn laplace_cdf(x: f64, mu: f64, b: f64) -> f64 {
    if x <= mu {
        0.5 * libm::exp((x - mu) / b)
    } else {
        1.0 - 0.5 * libm::exp(-(x - mu) / b)
    }
}

/// Leaky cumulative `left(s)` for one (mu, scale) model.
#[inline]
fn left_cumulative(s: i32, mu: f64, b: f64) -> u32 {
    if s == MIN_SYMBOL {
        return 0;
    }
    // `f64 as u32` saturates like num-traits' `AsPrimitive` used upstream.
    let non_leaky = (FREE_WEIGHT * laplace_cdf(s as f64 - 0.5, mu, b)) as u32;
    non_leaky + (s - MIN_SYMBOL) as u32
}

#[inline]
fn right_cumulative(s: i32, mu: f64, b: f64) -> u32 {
    if s == MAX_SYMBOL {
        1 << PRECISION
    } else {
        left_cumulative(s + 1, mu, b)
    }
}

pub(crate) struct RangeDecoder<'a> {
    words: &'a [u8],
    /// Next word index.
    pos: usize,
    lower: u64,
    range: u64,
    point: u64,
}

impl<'a> RangeDecoder<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Result<Self> {
        if !bytes.len().is_multiple_of(4) {
            return Err(Error::Invalid(
                "latent payload is not a whole number of u32 words",
            ));
        }
        let mut d = Self {
            words: bytes,
            pos: 0,
            lower: 0,
            range: u64::MAX,
            point: 0,
        };
        // `RangeDecoder::read_point`: up to two words, left-aligned if fewer exist.
        let mut n = 0;
        while n < 2 {
            match d.read_word() {
                Some(w) => {
                    d.point = (d.point << 32) | w as u64;
                    n += 1;
                }
                None => break,
            }
        }
        if n == 1 {
            d.point <<= 32;
        }
        Ok(d)
    }

    #[inline]
    fn read_word(&mut self) -> Option<u32> {
        let i = self.pos * 4;
        let b = self.words.get(i..i + 4)?;
        self.pos += 1;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Decodes one symbol under the quantized Laplace with table indices
    /// `(idx_mu, idx_scale)` (already clipped into range).
    pub(crate) fn decode(&mut self, idx_mu: usize, idx_scale: usize) -> Result<i32> {
        let mu = mu_of(idx_mu) as f64;
        let b = scale_of(idx_scale) as f64;

        let scale = self.range >> PRECISION;
        let quantile = self.point.wrapping_sub(self.lower) / scale;
        if quantile >= 1u64 << PRECISION {
            return Err(Error::Invalid("range decoder quantile out of range"));
        }
        let q = quantile as u32;

        // Binary search for the largest s with left(s) <= q. left(MIN_SYMBOL) = 0 <= q.
        let (mut lo, mut hi) = (MIN_SYMBOL, MAX_SYMBOL);
        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2;
            if left_cumulative(mid, mu, b) <= q {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let symbol = lo;
        let left = left_cumulative(symbol, mu, b);
        let right = right_cumulative(symbol, mu, b);
        let probability = right.wrapping_sub(left);
        if probability == 0 {
            return Err(Error::Invalid("zero-probability symbol"));
        }

        self.lower = self.lower.wrapping_add(scale * left as u64);
        self.range = scale * probability as u64;
        if self.range < 1u64 << 32 {
            self.lower <<= 32;
            self.range <<= 32;
            self.point <<= 32;
            if let Some(w) = self.read_word() {
                self.point |= w as u64;
            }
        }
        Ok(symbol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_cover_documented_ranges() {
        assert_eq!(N_SCALE_IDX, 2561);
        assert_eq!(mu_of(0), -64.0);
        assert_eq!(mu_of(N_MU - 1), 64.0 - 1.0 / 256.0);
        // Scale index 1280 is log-scale 0 -> scale 1.
        assert_eq!(scale_of(1280), 1.0);
    }

    #[test]
    fn cumulatives_are_monotone_and_leaky() {
        for &(im, is) in &[(0usize, 0usize), (16384, 1280), (32767, 2560), (12345, 17)] {
            let (mu, b) = (mu_of(im) as f64, scale_of(is) as f64);
            let mut prev = 0;
            for s in MIN_SYMBOL..=MAX_SYMBOL {
                let r = right_cumulative(s, mu, b);
                assert!(r > left_cumulative(s, mu, b));
                assert!(r > prev);
                prev = r;
            }
            assert_eq!(prev, 1 << PRECISION);
        }
    }
}
