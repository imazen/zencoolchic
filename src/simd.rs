//! SIMD kernels for the float networks, dispatched with archmage.
//!
//! Every tier computes each output element with the same sequence of fused multiply-adds,
//! so all tiers give bit-identical results: v3 (AVX2 + FMA) and NEON use magetypes'
//! fused `mul_add`; the scalar tier (and the tails) use `libm::fmaf`, which is exact.

use archmage::prelude::*;

/// `acc[i] = fma(x[i], w, acc[i])`.
#[inline]
pub(crate) fn axpy(acc: &mut [f32], w: f32, x: &[f32]) {
    debug_assert_eq!(acc.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        return axpy_impl_v3(t, acc, w, x);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        return axpy_impl_neon(t, acc, w, x);
    }
    axpy_fused_scalar(acc, w, x)
}

// The macro also emits an (unfused, unused) `axpy_impl_scalar`.
#[allow(dead_code)]
#[magetypes(define(f32x8), v3, neon)]
fn axpy_impl(token: Token, acc: &mut [f32], w: f32, x: &[f32]) {
    let n = acc.len().min(x.len());
    let (acc, x) = (&mut acc[..n], &x[..n]);
    let wv = f32x8::splat_t(token, w);
    let (ac, at) = acc.as_chunks_mut::<8>();
    let (xc, xt) = x.as_chunks::<8>();
    for (a, xv) in ac.iter_mut().zip(xc) {
        f32x8::load_t(token, xv)
            .mul_add(wv, f32x8::load_t(token, a))
            .store(a);
    }
    for (a, &xv) in at.iter_mut().zip(xt) {
        *a = libm::fmaf(xv, w, *a);
    }
}

/// `Σ a[i] · b[i]` with a fixed reduction order on every tier: eight lane accumulators
/// (element `i` goes to lane `i % 8`, fused multiply-add), the tail into lanes `0..`, then
/// `((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))`.
#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = X64V3Token::summon() {
        return dot_impl_v3(t, a, b);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = NeonToken::summon() {
        return dot_impl_neon(t, a, b);
    }
    dot_fused_scalar(a, b)
}

#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
#[inline(always)]
fn reduce8(l: [f32; 8]) -> f32 {
    ((l[0] + l[1]) + (l[2] + l[3])) + ((l[4] + l[5]) + (l[6] + l[7]))
}

// The macro also emits an (unfused, unused) `dot_impl_scalar`.
#[allow(dead_code)]
#[magetypes(define(f32x8), v3, neon)]
fn dot_impl(token: Token, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = f32x8::zero_t(token);
    let (ac, at) = a.as_chunks::<8>();
    let (bc, bt) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        acc = f32x8::load_t(token, x).mul_add(f32x8::load_t(token, y), acc);
    }
    let mut l = [0.0f32; 8];
    acc.store(&mut l);
    for (i, (&x, &y)) in at.iter().zip(bt).enumerate() {
        l[i] = libm::fmaf(x, y, l[i]);
    }
    reduce8(l)
}

#[cfg_attr(not(feature = "encoder"), allow(dead_code))]
fn dot_fused_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut l = [0.0f32; 8];
    let (ac, at) = a.as_chunks::<8>();
    let (bc, bt) = b.as_chunks::<8>();
    for (x, y) in ac.iter().zip(bc) {
        for k in 0..8 {
            l[k] = libm::fmaf(x[k], y[k], l[k]);
        }
    }
    for (i, (&x, &y)) in at.iter().zip(bt).enumerate() {
        l[i] = libm::fmaf(x, y, l[i]);
    }
    reduce8(l)
}

/// Scalar tier, fused like the SIMD tiers (magetypes' own scalar backend is unfused).
fn axpy_fused_scalar(acc: &mut [f32], w: f32, x: &[f32]) {
    for (a, &xv) in acc.iter_mut().zip(x) {
        *a = libm::fmaf(xv, w, *a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn dispatched_tier_matches_fused_scalar_bit_for_bit() {
        let mut seed = 0x2545_f491u32;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for n in [0usize, 1, 7, 8, 9, 31, 64, 1000] {
            let x: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let a0: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let w = rnd();
            let (mut a, mut b) = (a0.clone(), a0.clone());
            axpy(&mut a, w, &x);
            axpy_fused_scalar(&mut b, w, &x);
            let (a, b): (Vec<u32>, Vec<u32>) = (
                a.iter().map(|v| v.to_bits()).collect(),
                b.iter().map(|v| v.to_bits()).collect(),
            );
            assert_eq!(a, b, "n = {n}");
            let y: Vec<f32> = (0..n).map(|_| rnd()).collect();
            assert_eq!(
                dot(&x, &y).to_bits(),
                dot_fused_scalar(&x, &y).to_bits(),
                "dot n = {n}"
            );
        }
    }
}
