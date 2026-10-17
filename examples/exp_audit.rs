//! Exhaustive audit of the range coder's Laplace CDF: does `libm::exp` (what this crate
//! uses) give the same leaky cumulative as the platform libm behind Rust std (what
//! upstream's constriction wheels use: glibc on Linux, Apple libm on macOS)?
//!
//! Every (mu, scale, symbol boundary) the decoder can evaluate: 32768 × 2561 × 128.
//! `cargo run --release --example exp_audit --features std [-- <threads>]`
//!
//! The formula mirrors `src/rangecoder.rs` (`laplace_cdf`, `left_cumulative`).

use std::sync::atomic::{AtomicU64, Ordering};

#[path = "../src/scale_table.rs"]
#[allow(dead_code)]
mod scale_table;

const FREE_WEIGHT: f64 = ((1u32 << 24) - 1 - 127) as f64;

#[inline(always)]
fn cdf(x: f64, mu: f64, b: f64, exp: impl Fn(f64) -> f64) -> f64 {
    if x <= mu {
        0.5 * exp((x - mu) / b)
    } else {
        1.0 - 0.5 * exp(-(x - mu) / b)
    }
}

fn main() {
    let threads: usize = std::env::args().nth(1).map_or(8, |s| s.parse().unwrap());
    let exp_diff = AtomicU64::new(0);
    let cum_diff = AtomicU64::new(0);
    let total = AtomicU64::new(0);
    let next = AtomicU64::new(0);
    let t0 = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let im = next.fetch_add(1, Ordering::Relaxed);
                    if im >= 32768 {
                        break;
                    }
                    let mu = (-64.0f32 + im as f32 / 256.0) as f64;
                    let (mut e, mut c, mut n) = (0u64, 0u64, 0u64);
                    for &bits in scale_table::SCALE_BITS.iter() {
                        let b = f32::from_bits(bits) as f64;
                        // Boundaries s - 0.5 for s in -63..=64 (the left cumulative of every
                        // symbol but the first, which is fixed at 0).
                        for s in -63i32..=64 {
                            let x = s as f64 - 0.5;
                            let arg = if x <= mu { (x - mu) / b } else { -(x - mu) / b };
                            let (a, z) = (libm::exp(arg), arg.exp());
                            if a.to_bits() != z.to_bits() {
                                e += 1;
                                let ca = (FREE_WEIGHT * cdf(x, mu, b, libm::exp)) as u32;
                                let cz = (FREE_WEIGHT * cdf(x, mu, b, f64::exp)) as u32;
                                if ca != cz {
                                    c += 1;
                                    if c <= 3 {
                                        eprintln!("cumulative differs: mu_idx {im} scale_bits {bits:#x} s {s}: libm {ca} std {cz}");
                                    }
                                }
                            }
                            n += 1;
                        }
                    }
                    exp_diff.fetch_add(e, Ordering::Relaxed);
                    cum_diff.fetch_add(c, Ordering::Relaxed);
                    total.fetch_add(n, Ordering::Relaxed);
                }
            });
        }
    });
    println!(
        "{} {}: evaluations {}; exp bit differences {}; leaky-cumulative differences {}; {:.1} s",
        std::env::consts::OS,
        std::env::consts::ARCH,
        total.load(Ordering::Relaxed),
        exp_diff.load(Ordering::Relaxed),
        cum_diff.load(Ordering::Relaxed),
        t0.elapsed().as_secs_f64()
    );
}
