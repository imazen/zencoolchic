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

/// Derivative `d/dt` of [`coeffs`], for the training-time gradient back to the
/// flow (upstream differentiates `grid_sample`/`interpolate_1d` in the same
/// separable way).
#[cfg(feature = "encoder")]
fn dcoeffs(t: f32, filter_size: usize, kernel: Kernel) -> [f32; MAX_TAPS] {
    let mut c = [0.0f32; MAX_TAPS];
    match kernel {
        Kernel::Bilinear => {
            c[0] = -1.0;
            c[1] = 1.0;
        }
        Kernel::Bicubic => {
            let a = -0.75f32;
            let t2 = t * t;
            // f'(t) of the four cubic taps.
            c[0] = a * (1.0 - 4.0 * t + 3.0 * t2);
            c[1] = -2.0 * (a + 3.0) * t + 3.0 * (a + 2.0) * t2;
            c[2] = -a + 2.0 * (2.0 * a + 3.0) * t - 3.0 * (a + 2.0) * t2;
            c[3] = a * (2.0 * t - 3.0 * t2);
        }
        Kernel::Sinc => {
            let f = filter_size as f32;
            let first = -(filter_size as i32 / 2) + 1;
            for (k, v) in c[..filter_size].iter_mut().enumerate() {
                let d = t - (first + k as i32) as f32;
                // d/dd [cos(pi d / F) * sin(pi d) / (pi d)]
                let pd = core::f32::consts::PI * d;
                if pd == 0.0 {
                    *v = 0.0;
                } else {
                    let sinc = libm::sinf(pd) / pd;
                    let dsinc = (pd * libm::cosf(pd) - libm::sinf(pd)) / (pd * d);
                    *v = -libm::sinf(pd / f) * sinc * (core::f32::consts::PI / f)
                        + libm::cosf(pd / f) * dsinc;
                }
            }
        }
    }
    c
}

/// Warps one `h*w` plane: the flow is quantized to 1/64-pixel precision here
/// (upstream `Warper.forward` eval mode), split into an integer re-index and a
/// fractional part evaluated with the interpolation kernel. `flow_x`/`flow_y`
/// are `h*w` planar displacements in pixels.
fn warp_plane(
    x: &Planes,
    c: usize,
    flow: (&[f32], &[f32]),
    filter_size: usize,
    kernel: Kernel,
    quantize: bool,
) -> Vec<f32> {
    let (w, h) = (x.w, x.h);
    let x = &x.data[c * w * h..(c + 1) * w * h];
    let (flow_x, flow_y) = flow;
    let mut out = vec![0.0f32; w * h];
    let first = -(filter_size as i64 / 2) + 1;
    // Every output pixel is an independent filter_size² gather — row bands
    // fan out to scoped helpers under the same policy as the latent
    // wavefront. Output is identical either way.
    let eval = |y0: usize, y1: usize, out: &mut [f32]| {
        let mut line = [0.0f32; MAX_TAPS];
        for y in y0..y1 {
            for x_ in 0..w {
                let i = (y - y0) * w + x_;
                let fi = y * w + x_;
                // round(flow * 64) / 64 — inference-time flow quantization;
                // training mode (upstream `Warper.training`) uses the flow as-is.
                let (qx, qy) = if quantize {
                    (
                        libm::rintf(flow_x[fi] * FRAC_ACCURACY) / FRAC_ACCURACY,
                        libm::rintf(flow_y[fi] * FRAC_ACCURACY) / FRAC_ACCURACY,
                    )
                } else {
                    (flow_x[fi], flow_y[fi])
                };
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
    };
    #[cfg(feature = "std")]
    if w * h >= 32768 && crate::latent::wave_threads() > 1 {
        let workers = crate::latent::wave_threads().min(h.div_ceil(8));
        let band = h.div_ceil(workers);
        let eval = &eval;
        std::thread::scope(|s| {
            let mut chunks = out.chunks_mut(band * w);
            let first_chunk = chunks.next().unwrap();
            for (b, chunk) in chunks.enumerate() {
                let y0 = (b + 1) * band;
                s.spawn(move || eval(y0, (y0 + band).min(h), chunk));
            }
            eval(0, band.min(h), first_chunk);
        });
    } else {
        eval(0, h, &mut out);
    }
    #[cfg(not(feature = "std"))]
    eval(0, h, &mut out);
    out
}

/// Warps every plane of `x` with the same flow — upstream `Warper.forward` on a
/// `[1, C, H, W]` tensor. `filter_size` comes from the frame header.
pub(crate) fn warp(x: &Planes, flow_x: &[f32], flow_y: &[f32], filter_size: u32) -> Result<Planes> {
    warp_impl(x, flow_x, flow_y, filter_size, true)
}

fn warp_impl(
    x: &Planes,
    flow_x: &[f32],
    flow_y: &[f32],
    filter_size: u32,
    quantize: bool,
) -> Result<Planes> {
    let kernel = Kernel::of(filter_size)?;
    let f = filter_size as usize;
    debug_assert!(f <= MAX_TAPS);
    Ok(Planes {
        c: x.c,
        h: x.h,
        w: x.w,
        data: (0..x.c)
            .flat_map(|c| warp_plane(x, c, (flow_x, flow_y), f, kernel, quantize))
            .collect(),
    })
}

/// Training-mode warp: same math as [`warp`] on the **unquantized** flow
/// (upstream `self.training` skips the 1/64-pixel rounding).
#[cfg(feature = "encoder")]
pub(crate) fn warp_train(
    x: &Planes,
    flow_x: &[f32],
    flow_y: &[f32],
    filter_size: u32,
) -> Result<Planes> {
    warp_impl(x, flow_x, flow_y, filter_size, false)
}

/// Gradient of the *training-mode* warp with respect to the flow — upstream
/// differentiates `F.grid_sample` (filter 2/4) or the custom `interpolate_1d`
/// path (sinc) on the **unquantized** flow; both are the same separable math.
/// `dout` is the upstream gradient per output plane (same `c*h*w` layout as
/// [`warp`]'s result); returns `(d_flow_x, d_flow_y)` accumulated over all
/// channels since one flow field warps every plane.
#[cfg(feature = "encoder")]
pub(crate) fn warp_flow_backward(
    dout: &Planes,
    x: &Planes,
    flow_x: &[f32],
    flow_y: &[f32],
    filter_size: u32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let kernel = Kernel::of(filter_size)?;
    let f = filter_size as usize;
    let (w, h) = (x.w, x.h);
    let mut dfx = alloc::vec![0.0f32; w * h];
    let mut dfy = alloc::vec![0.0f32; w * h];
    let mut line = [0.0f32; MAX_TAPS];
    let first = -(f as i64 / 2) + 1;
    for c in 0..x.c {
        let src = &x.data[c * w * h..(c + 1) * w * h];
        let dsrc = &dout.data[c * w * h..(c + 1) * w * h];
        for y in 0..h {
            for xi in 0..w {
                let i = y * w + xi;
                let g = dsrc[i];
                if g == 0.0 {
                    continue;
                }
                // Training mode: the flow is used unquantized; `floor` is
                // piecewise constant so `d frac/d flow = 1` (upstream autograd
                // on `flow - torch.floor(flow)`).
                let (bx, by) = (libm::floorf(flow_x[i]), libm::floorf(flow_y[i]));
                let cx = coeffs(flow_x[i] - bx, f, kernel);
                let cy = coeffs(flow_y[i] - by, f, kernel);
                let dcx = dcoeffs(flow_x[i] - bx, f, kernel);
                let dcy = dcoeffs(flow_y[i] - by, f, kernel);
                let (bx, by) = (bx as i64, by as i64);
                for (r, slot) in line[..f].iter_mut().enumerate() {
                    let yy = clamp_idx(y as i64 + by + first + r as i64, h);
                    let row = &src[yy * w..yy * w + w];
                    let mut acc = 0.0f32;
                    let mut dacc = 0.0f32;
                    for (s, &cxs) in cx[..f].iter().enumerate() {
                        let v = row[clamp_idx(xi as i64 + bx + first + s as i64, w)];
                        acc += cxs * v;
                        dacc += dcx[s] * v;
                    }
                    *slot = acc;
                    dfy[i] += g * dcy[r] * acc;
                    dfx[i] += g * cy[r] * dacc;
                }
            }
        }
    }
    Ok((dfx, dfy))
}

/// Mean squared error over every plane sample.
#[cfg(feature = "encoder")]
fn mse(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / a.len() as f32
}

/// Best integer global translation of `refr` toward `target` — upstream
/// `get_global_translation`: a coarse-to-fine search over step sizes
/// `2^n, n = 8..=1`, each level trying the 9 candidate shifts around the
/// current centre and keeping the MSE minimum. Upstream re-baselines every
/// level on the *unshifted* reference MSE (not the current centre's), and
/// updates only on strict improvement in a fixed candidate order; both are
/// reproduced here. Inputs are same-shaped 4:4:4 planes; returns `(dx, dy)`
/// in `translate` convention (ch0 → x, ch1 → y).
#[cfg(feature = "encoder")]
pub(crate) fn global_translation(target: &Planes, refr: &Planes) -> (i32, i32) {
    debug_assert_eq!((target.c, target.h, target.w), (refr.c, refr.h, refr.w));
    let (mut cx, mut cy) = (0i32, 0i32);
    for n in (1..=8).rev() {
        let s = 1i32 << n;
        // Upstream: `initial_mse = _compute_mse(raw_ref_data, raw_target_data)`
        // — the unshifted reference is the bar for every level.
        let mut best = mse(&target.data, &refr.data);
        let (mut bx, mut by) = (0, 0);
        // Upstream `all_shifts` order: centre, then axis then diagonal pairs.
        for (dx, dy) in [
            (0, 0),
            (0, s),
            (0, -s),
            (s, 0),
            (-s, 0),
            (s, s),
            (-s, s),
            (s, -s),
            (-s, -s),
        ] {
            let m = mse(&target.data, &translate(refr, cx + dx, cy + dy).data);
            if m < best {
                best = m;
                (bx, by) = (dx, dy);
            }
        }
        cx += bx;
        cy += by;
    }
    (cx, cy)
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

    /// The row-band fan-out (>= 32768 px under `std`) must produce the same
    /// output as the serial pass. 256×128 hits the gate.
    #[cfg(feature = "std")]
    #[test]
    fn parallel_bands_match_serial() {
        let (h, w) = (256usize, 128usize);
        let n = h * w;
        let x = planes(
            1,
            h,
            w,
            (0..n)
                .map(|i| ((i * 37 + 11) % 251) as f32 * 0.01 - 1.0)
                .collect(),
        );
        let flow_x: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 61) as f32 * 0.05).collect();
        let flow_y: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 47) as f32 * -0.05).collect();
        let serial = {
            let _seq = crate::latent::SequentialThreads::install();
            warp(&x, &flow_x, &flow_y, 8).unwrap()
        };
        let par = warp(&x, &flow_x, &flow_y, 8).unwrap();
        assert_eq!(serial.data, par.data);
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
            // `comp` covers the inter compositing in `train::inter`'s tests.
            if case != "vec" {
                continue;
            }
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
            // Training-mode flow gradient (autograd through the unquantized
            // warp) and the coarse-to-fine global-translation search, both
            // checked for every filter size / dumped pair.
            #[cfg(feature = "encoder")]
            {
                let dout = Planes {
                    c,
                    h,
                    w,
                    data: f32_file(&case, "dout"),
                };
                for fs in [2, 4, 8, 12] {
                    let want = f32_file(&case, &format!("dflow{fs}"));
                    let (dfx, dfy) = warp_flow_backward(&dout, &x, fx, fy, fs).unwrap();
                    let max = dfx
                        .iter()
                        .chain(&dfy)
                        .zip(&want)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    eprintln!("{case} filter_size={fs}: max |dflow diff| {max:e}");
                    assert!(max <= 1e-3, "{case} filter_size={fs}: dflow error {max:e}");
                }
                // Upstream `get_global_translation` answers on the two dumped
                // pairs — the search's coarse-step quirks are reproduced
                // exactly, including results that overshoot the true shift.
                for (i, (img, tgt)) in [("", ("x", "target")), ("2", ("x2", "target2"))] {
                    let refr = Planes {
                        c,
                        h,
                        w,
                        data: f32_file(&case, img),
                    };
                    let target = Planes {
                        c,
                        h,
                        w,
                        data: f32_file(&case, tgt),
                    };
                    let want: Vec<i32> = meta[format!("global_translation{i}")]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap() as i32)
                        .collect();
                    assert_eq!(
                        global_translation(&target, &refr),
                        (want[0], want[1]),
                        "{case}: global_translation{i} mismatch"
                    );
                }
            }
        }
    }
}
