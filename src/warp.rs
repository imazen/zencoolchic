//! Motion-compensation warping (upstream `component/intercoding/warp.py` and
//! `globalmotion.py`).
//!
//! Optical flow is a two-channel tensor whose channel 0 is the horizontal (x)
//! displacement and channel 1 the vertical (y) one: output pixel `(y, x)`
//! samples `x[y + flow_y, x + flow_x]`. At inference time the flow is quantized
//! to `1 / 64`-pixel precision, then evaluated with a `warp_filter_size`-tap
//! separable kernel: 2 taps is bilinear, 4 is bicubic (both bit-equivalent in
//! math to `torch.nn.functional.grid_sample(..., align_corners=True,
//! padding_mode="border")`), and any larger even size is the sinc window of
//! "Efficient Sub-pixel Motion Compensation in Learned Video Codecs".

use crate::error::{Error, Result};
use crate::nn::Planes;
use alloc::vec;
use alloc::vec::Vec;

/// Upstream `WarpParameter.fractional_accuracy`.
const FRAC_ACCURACY: f32 = 64.0;
/// `warp_filter_size` is a 4-bit field; the largest even encodable value is 14.
const MAX_TAPS: usize = 14;

/// The interpolation kernel selected by `warp_filter_size` (upstream
/// `WarpParameter.__post_init__`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kernel {
    /// `filter_size = 2` — `grid_sample` bilinear.
    Bilinear,
    /// `filter_size = 4` — `grid_sample` bicubic (`a = -0.75`).
    Bicubic,
    /// Any other even `filter_size` — `cos(pi t / F) * sinc(t)` taps.
    Sinc,
}

impl Kernel {
    fn of(filter_size: u32) -> Result<Self> {
        if filter_size < 2 || !filter_size.is_multiple_of(2) {
            return Err(Error::Invalid("warp_filter_size"));
        }
        Ok(match filter_size {
            2 => Kernel::Bilinear,
            4 => Kernel::Bicubic,
            _ => Kernel::Sinc,
        })
    }
}

/// Interpolation weights for fractional displacement `t in [0, 1)`, laid out
/// against the `filter_size` neighbor offsets `-(F/2) + 1 ..= F/2` (upstream
/// `Warper.get_coeffs`).
fn coeffs(t: f32, filter_size: usize, kernel: Kernel) -> [f32; MAX_TAPS] {
    let mut c = [0.0f32; MAX_TAPS];
    match kernel {
        Kernel::Bilinear => {
            c[0] = 1.0 - t;
            c[1] = t;
        }
        Kernel::Bicubic => {
            // Upstream's `B @ [1, t, t^2, t^3]` with `a = -0.75`, which is
            // PyTorch's bicubic `cubic_convolution` exactly.
            let a = -0.75f32;
            let (t2, t3) = (t * t, t * t * t);
            c[0] = a * (t - 2.0 * t2 + t3);
            c[1] = 1.0 - (a + 3.0) * t2 + (a + 2.0) * t3;
            c[2] = -a * t + (2.0 * a + 3.0) * t2 - (a + 2.0) * t3;
            c[3] = a * (t2 - t3);
        }
        Kernel::Sinc => {
            let f = filter_size as f32;
            let first = -(filter_size as i32 / 2) + 1;
            for (k, v) in c[..filter_size].iter_mut().enumerate() {
                let d = t - (first + k as i32) as f32;
                // cos(pi d / F) * sinc(d) = cos(pi d / F) * sin(pi d) / (pi d)
                let pd = core::f32::consts::PI * d;
                let sinc = if d == 0.0 { 1.0 } else { libm::sinf(pd) / pd };
                *v = libm::cosf(pd / f) * sinc;
            }
        }
    }
    c
}

#[inline]
fn clamp_idx(v: i64, n: usize) -> usize {
    v.clamp(0, n as i64 - 1) as usize
}

/// Warps one `h*w` plane: the flow is quantized to 1/64-pixel precision here
/// (upstream `Warper.forward` eval mode), split into an integer re-index and a
/// fractional part evaluated with the interpolation kernel. `flow_x`/`flow_y`
/// are `h*w` planar displacements in pixels.
fn warp_plane(
    x: &[f32],
    w: usize,
    h: usize,
    flow_x: &[f32],
    flow_y: &[f32],
    filter_size: usize,
    kernel: Kernel,
) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let mut line = [0.0f32; MAX_TAPS];
    let first = -(filter_size as i64 / 2) + 1;
    for y in 0..h {
        for x_ in 0..w {
            let i = y * w + x_;
            // round(flow * 64) / 64 — inference-time flow quantization.
            let qx = libm::rintf(flow_x[i] * FRAC_ACCURACY) / FRAC_ACCURACY;
            let qy = libm::rintf(flow_y[i] * FRAC_ACCURACY) / FRAC_ACCURACY;
            let bx = libm::floorf(qx);
            let by = libm::floorf(qy);
            let cx = coeffs(qx - bx, filter_size, kernel);
            let cy = coeffs(qy - by, filter_size, kernel);
            let (bx, by) = (bx as i64, by as i64);
            // Integer part of the flow re-indexes the source with border
            // clamping; coefficients interpolate the fractional part along
            // columns first, then rows (upstream `interpolate_1d` twice).
            for (r, slot) in line[..filter_size].iter_mut().enumerate() {
                let yy = clamp_idx(y as i64 + by + first + r as i64, h);
                let row = &x[yy * w..yy * w + w];
                let mut acc = 0.0f32;
                for (s, &cx_s) in cx[..filter_size].iter().enumerate() {
                    acc += cx_s * row[clamp_idx(x_ as i64 + bx + first + s as i64, w)];
                }
                *slot = acc;
            }
            out[i] = cy[..filter_size]
                .iter()
                .zip(&line)
                .map(|(c, l)| c * l)
                .sum();
        }
    }
    out
}

/// Warps every plane of `x` with the same flow — upstream `Warper.forward` on a
/// `[1, C, H, W]` tensor. `filter_size` comes from the frame header.
pub(crate) fn warp(x: &Planes, flow_x: &[f32], flow_y: &[f32], filter_size: u32) -> Result<Planes> {
    let kernel = Kernel::of(filter_size)?;
    let f = filter_size as usize;
    debug_assert!(f <= MAX_TAPS);
    Ok(Planes {
        c: x.c,
        h: x.h,
        w: x.w,
        data: (0..x.c)
            .flat_map(|c| {
                warp_plane(
                    &x.data[c * x.h * x.w..(c + 1) * x.h * x.w],
                    x.w,
                    x.h,
                    flow_x,
                    flow_y,
                    f,
                    kernel,
                )
            })
            .collect(),
    })
}

/// Applies an integer global translation to every plane — upstream
/// `apply_global_translation`, a `nearest`-mode warp with constant integer
/// flow, which is exactly `out[y, x] = x[clamp(y + dy), clamp(x + dx)]`.
pub(crate) fn translate(x: &Planes, dx: i32, dy: i32) -> Planes {
    let (w, h) = (x.w, x.h);
    let mut data = Vec::with_capacity(x.c * w * h);
    for c in 0..x.c {
        let src = &x.data[c * w * h..(c + 1) * w * h];
        for y in 0..h {
            let yy = clamp_idx(y as i64 + dy as i64, h);
            let row = &src[yy * w..yy * w + w];
            for xx in 0..w {
                data.push(row[clamp_idx(xx as i64 + dx as i64, w)]);
            }
        }
    }
    Planes { c: x.c, h, w, data }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use alloc::vec;

    fn planes(c: usize, h: usize, w: usize, data: Vec<f32>) -> Planes {
        Planes { c, h, w, data }
    }

    #[test]
    fn translate_shifts_with_border_clamp() {
        // 2x3 ramp; out[y, x] = x[clamp(y + dy), clamp(x + dx)].
        let x = planes(1, 2, 3, vec![0., 1., 2., 3., 4., 5.]);
        let got = translate(&x, 1, 0);
        assert_eq!(got.data, vec![1., 2., 2., 4., 5., 5.]);
        let got = translate(&x, 0, 1);
        assert_eq!(got.data, vec![3., 4., 5., 3., 4., 5.]);
        let got = translate(&x, -1, -1);
        assert_eq!(got.data, vec![0., 0., 1., 0., 0., 1.]);
    }

    #[test]
    fn integer_flow_is_a_shift() {
        let x = planes(1, 2, 4, vec![0., 10., 20., 30., 40., 50., 60., 70.]);
        // flow_x = -1 shifts every sample right by one column.
        let got = warp(&x, &[-1.0; 8], &[0.0; 8], 2).unwrap();
        assert_eq!(
            got.data,
            vec![0., 0., 10., 20., 40., 40., 50., 60.],
            "out[y,x] = x[y, x - 1]"
        );
        // Vertical flow +1: out[y, x] = x[y + 1, x].
        let got = warp(&x, &[0.0; 8], &[1.0; 8], 4).unwrap();
        assert_eq!(got.data, vec![40., 50., 60., 70., 40., 50., 60., 70.]);
    }

    #[test]
    fn half_pixel_bilinear() {
        // 1x3 row: [0, 2, 4]; flow_x = -0.5 → out[x] = x[x - 0.5]:
        // x[-0.5] → border-clamped taps {0,0} → 0; x[0.5] → (0+2)/2 = 1; x[1.5] → 3.
        // -0.5 quantizes exactly at 1/64 precision.
        let x = planes(1, 1, 3, vec![0., 2., 4.]);
        let got = warp(&x, &[-0.5; 3], &[0.0; 3], 2).unwrap();
        assert_eq!(got.data, vec![0., 1., 3.]);
    }

    #[test]
    fn flow_quantizes_to_sixty_fourths() {
        // -0.4 px rounds to -26/64 = -0.40625, so fractional part 0.59375.
        let x = planes(1, 1, 4, vec![0., 64., 128., 192.]);
        let got = warp(&x, &[-0.4; 4], &[0.0; 4], 2).unwrap();
        let t = 38.0f32 / 64.0;
        // out[x] = (1 - t) * x[x - 1] + t * x[x], with border clamping.
        let want = [
            0.0,
            64.0 * t,
            64.0 * (1.0 - t) + 128.0 * t,
            128.0 * (1.0 - t) + 192.0 * t,
        ];
        for (a, b) in got.data.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn rejects_bad_filter_size() {
        let x = planes(1, 1, 2, vec![0., 1.]);
        for fs in [0, 1, 3, 5, 15] {
            assert!(warp(&x, &[0.0; 2], &[0.0; 2], fs).is_err(), "fs={fs}");
        }
    }
}

#[cfg(all(test, feature = "oracle"))]
mod tests {
    use super::*;
    use crate::oracle_fixtures;
    use std::{eprintln, format, vec::Vec};

    /// Reads a little-endian f32 dump written by `scripts/oracle/dump_warp.py`.
    fn f32_file(case: &str, name: &str) -> Vec<f32> {
        oracle_fixtures::read("warp", &format!("{case}.{name}.f32"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    /// Frozen upstream `Warper` outputs for every filter size plus an integer
    /// global translation (`vanilla_warp_fn` nearest), dumped by
    /// `dump_warp.py`.
    #[test]
    fn matches_upstream_warper() {
        for case in oracle_fixtures::cases("warp") {
            let meta: serde_json::Value =
                serde_json::from_slice(&oracle_fixtures::read("warp", &format!("{case}.json")))
                    .unwrap();
            let (h, w, c) = (
                meta["H"].as_u64().unwrap() as usize,
                meta["W"].as_u64().unwrap() as usize,
                meta["C"].as_u64().unwrap() as usize,
            );
            let n = h * w;
            let x = Planes {
                c,
                h,
                w,
                data: f32_file(&case, "x"),
            };
            let flow = f32_file(&case, "flow");
            let (fx, fy) = (&flow[..n], &flow[n..2 * n]);
            for fs in [2, 4, 8, 12] {
                let want = f32_file(&case, &format!("out{fs}"));
                let got = warp(&x, fx, fy, fs).unwrap();
                let max = got
                    .data
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                eprintln!("{case} filter_size={fs}: max |warp diff| {max:e}");
                assert!(max <= 1e-5, "{case} filter_size={fs}: warp error {max:e}");
            }
            // Integer global translation: dx=3, dy=-2.
            let want = f32_file(&case, "shift");
            let got = translate(&x, 3, -2);
            assert_eq!(got.data, want, "{case}: global translation mismatch");
        }
    }
}
