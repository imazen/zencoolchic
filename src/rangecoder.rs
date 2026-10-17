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

        // Largest s with left(s) <= q. Start from the inverse CDF of the (non-leaky) Laplace
        // and walk: the leak shifts cumulatives by at most 127 / 2^24, so the guess is
        // almost always within one symbol. left(MIN_SYMBOL) = 0 <= q always holds.
        let p = (q as f64 + 0.5) / (1u64 << PRECISION) as f64;
        let guess = if p > 0.5 {
            mu - b * libm::log(2.0 - 2.0 * p)
        } else {
            mu + b * libm::log(2.0 * p)
        };
        let mut symbol = if guess.is_nan() {
            0
        } else {
            (libm::floor(guess) as i32).clamp(MIN_SYMBOL, MAX_SYMBOL)
        };
        let mut left = left_cumulative(symbol, mu, b);
        let mut right;
        if left > q {
            loop {
                right = left;
                symbol -= 1;
                left = left_cumulative(symbol, mu, b);
                if left <= q {
                    break;
                }
            }
        } else {
            loop {
                right = right_cumulative(symbol, mu, b);
                if right > q {
                    break;
                }
                symbol += 1;
                left = right;
            }
        }
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

#[cfg(feature = "encoder")]
/// `constriction::stream::queue::DefaultRangeEncoder` (u32 words, u64 state, 24-bit
/// probabilities). Output words are little-endian bytes, as upstream writes them.
pub(crate) struct RangeEncoder {
    words: alloc::vec::Vec<u32>,
    lower: u64,
    range: u64,
    /// Pending carry: `(count, first word)` of words whose final value depends on a carry
    /// that may still propagate (constriction's `EncoderSituation::Inverted`).
    inverted: Option<(usize, u32)>,
}

#[cfg(feature = "encoder")]
impl RangeEncoder {
    pub(crate) fn new() -> Self {
        Self {
            words: alloc::vec::Vec::new(),
            lower: 0,
            range: u64::MAX,
            inverted: None,
        }
    }

    fn flush_inverted(&mut self, carry: bool) {
        if let Some((n, first)) = self.inverted.take() {
            let (first, rest) = if carry {
                (first.wrapping_add(1), 0)
            } else {
                (first, u32::MAX)
            };
            self.words.push(first);
            for _ in 1..n {
                self.words.push(rest);
            }
        }
    }

    /// Encodes `symbol` under the quantized Laplace with table indices `(idx_mu, idx_scale)`.
    pub(crate) fn encode(&mut self, symbol: i32, idx_mu: usize, idx_scale: usize) -> Result<()> {
        if !(MIN_SYMBOL..=MAX_SYMBOL).contains(&symbol) {
            return Err(Error::Invalid("latent value outside [-64, 63]"));
        }
        let mu = mu_of(idx_mu) as f64;
        let b = scale_of(idx_scale) as f64;
        let left = left_cumulative(symbol, mu, b);
        let probability = right_cumulative(symbol, mu, b).wrapping_sub(left);

        let scale = self.range >> PRECISION;
        self.range = scale * probability as u64;
        if self.range == 0 {
            return Err(Error::Invalid("zero-probability symbol"));
        }
        let new_lower = self.lower.wrapping_add(scale * left as u64);
        if self.inverted.is_some() && new_lower.wrapping_add(self.range) > new_lower {
            self.flush_inverted(new_lower < self.lower);
        }
        self.lower = new_lower;

        if self.range < 1u64 << 32 {
            self.range <<= 32;
            let lower_word = (self.lower >> 32) as u32;
            self.lower <<= 32;
            if let Some((n, _)) = &mut self.inverted {
                *n += 1;
            } else if self.lower.wrapping_add(self.range) > self.lower {
                self.words.push(lower_word);
            } else {
                self.inverted = Some((1, lower_word));
            }
        }
        Ok(())
    }

    /// `seal` + byte serialization.
    pub(crate) fn finish(mut self) -> alloc::vec::Vec<u8> {
        if self.range != u64::MAX {
            let point = self.lower.wrapping_add((1u64 << 32) - 1);
            if self.inverted.is_some() {
                self.flush_inverted(point < self.lower);
            }
            let point_word = (point >> 32) as u32;
            self.words.push(point_word);
            let upper_word = (self.lower.wrapping_add(self.range) >> 32) as u32;
            if upper_word == point_word {
                self.words.push(0);
            }
        }
        self.words.iter().flat_map(|w| w.to_le_bytes()).collect()
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

    #[cfg(feature = "encoder")]
    #[test]
    fn encoder_round_trips_through_decoder() {
        // Deterministic mix of models and symbols, including extreme tails.
        let mut seed = 0x1234_5678u64;
        let mut rnd = |n: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        let mut items = alloc::vec::Vec::new();
        for _ in 0..20000 {
            let im = rnd(N_MU as u64) as usize;
            let is = rnd(N_SCALE_IDX as u64) as usize;
            let s = if rnd(10) == 0 {
                rnd(128) as i32 - 64
            } else {
                (mu_of(im) as i32).clamp(-64, 63) + rnd(3) as i32 - 1
            };
            items.push((s.clamp(-64, 63), im, is));
        }
        let mut enc = RangeEncoder::new();
        for &(s, im, is) in &items {
            enc.encode(s, im, is).unwrap();
        }
        let bytes = enc.finish();
        let mut dec = RangeDecoder::new(&bytes).unwrap();
        for &(s, im, is) in &items {
            assert_eq!(dec.decode(im, is).unwrap(), s);
        }
    }
}
