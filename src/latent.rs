//! Latent grid decoding (upstream `coolchic/bitstream/component/latent.py` and the latent
//! loop of `coolchic/bitstream/component/coolchic.py`).
//!
//! Grids are decoded coarsest first. For each value, the ARM sees up to 40 causal spatial
//! neighbours from a 9×9 window (picked by upstream's priority order) plus, when the grid
//! has an IFCE, a context vector computed from the already decoded coarser grids. The
//! symbols are read from the range coder in upstream's wavefront order: diagonal `t`
//! holds the positions with `x + 10·y = t`, visited by increasing `y`.

use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicI8, Ordering};

use crate::arm::{FixedArm, MAX_WIDTH, N_FRAC_BIT_INTER_FT_CTX, N_FRAC_BIT_MU_SCALE, WEIGHT_SHIFT};
use crate::error::{Error, Result};
#[cfg(feature = "encoder")]
use crate::rangecoder::RangeEncoder;
use crate::rangecoder::{N_MU, N_SCALE_IDX, RangeDecoder};

pub(crate) const MASK_SIZE: usize = 9;
pub(crate) const PAD: usize = (MASK_SIZE - 1) / 2;

/// Priority of each causal position of the 9×9 window (row-major, first 40 cells). The
/// `n` spatial contexts are the positions with priority `0..n`.
const PRIORITY_ORDER: [u8; 40] = [
    38, 35, 30, 25, 23, 31, 36, 37, 39, //
    33, 28, 21, 20, 6, 15, 22, 29, 34, //
    32, 18, 12, 10, 5, 9, 14, 19, 27, //
    24, 13, 8, 2, 1, 3, 11, 17, 26, //
    16, 7, 4, 0,
];

/// `_get_non_zero_pixel_ctx_index`: window indices (0..40) of the first `n` contexts,
/// ordered by priority.
pub(crate) fn context_window_indices(n: usize) -> Result<Vec<usize>> {
    if n > PRIORITY_ORDER.len() {
        return Err(Error::Invalid("spatial_context_arm > 40"));
    }
    // argsort(stable) of the priority order: the cell with priority p goes to slot p.
    let mut by_priority = [0usize; 40];
    for (cell, &p) in PRIORITY_ORDER.iter().enumerate() {
        by_priority[p as usize] = cell;
    }
    Ok(by_priority[..n].to_vec())
}

/// One integer grid, `h × w`.
#[derive(Debug, Clone)]
pub(crate) struct Grid {
    pub(crate) h: usize,
    pub(crate) w: usize,
    pub(crate) data: Vec<i64>,
}

/// Inter-feature context for one grid: `c` values per pixel, pixel-major.
pub(crate) struct InterContext {
    pub(crate) c: usize,
    pub(crate) data: Vec<i64>,
}

/// `fixed_upsampling(coded, mode="nearest")` for the IFCE: stacks the already decoded
/// grids (finest first) at the resolution of `coded[0]`, upsampling coarser ones by
/// nearest ×2 + crop at each step. Returns `(h, w, channels)` pixel-major.
fn stack_nearest(coded: &[&Grid]) -> (usize, usize, usize, Vec<i64>) {
    // Start from the smallest (last) grid; each step puts the finer grid's channel first.
    let last = coded[coded.len() - 1];
    let (mut h, mut w) = (last.h, last.w);
    let mut c = 1usize;
    let mut cur: Vec<i64> = last.data.clone(); // pixel-major, c channels
    for target in coded[..coded.len() - 1].iter().rev() {
        let (th, tw) = (target.h, target.w);
        let mut next = vec![0i64; th * tw * (c + 1)];
        let same = th == h && tw == w;
        for y in 0..th {
            for x in 0..tw {
                let (sy, sx) = if same { (y, x) } else { (y / 2, x / 2) };
                let o = (y * tw + x) * (c + 1);
                next[o] = target.data[y * tw + x];
                let s = (sy * w + sx) * c;
                next[o + 1..o + 1 + c].copy_from_slice(&cur[s..s + c]);
            }
        }
        cur = next;
        h = th;
        w = tw;
        c += 1;
    }
    (h, w, c, cur)
}

/// Computes the IFCE context for grid `idx` from the decoded coarser grids.
///
/// `coded` holds grids `idx + 1 ..` (finest first), empty for the coarsest grid.
pub(crate) fn inter_context(
    coded: &[&Grid],
    target_hw: (usize, usize),
    ifce: Option<&FixedArm>,
    n_out: usize,
) -> InterContext {
    let (h_i, w_i) = target_hw;
    let (uh, uw, uc, stacked) = if coded.is_empty() {
        (h_i, w_i, 1, vec![0i64; h_i * w_i])
    } else {
        stack_nearest(coded)
    };

    // IFCE output at the stacked resolution. The per-pixel forwards are
    // independent (immutable `stacked`, disjoint `small` slices), so the
    // largest passes fan out to scoped helpers under the same policy as the
    // wavefront — output is identical either way.
    let mut small = vec![0i64; uh * uw * n_out];
    if let Some(arm) = ifce {
        let shift = 2 * WEIGHT_SHIFT - N_FRAC_BIT_INTER_FT_CTX;
        let eval = |base: usize, chunk: &mut [i64]| {
            let mut scratch = [[0i64; MAX_WIDTH]; 2];
            for (i, out) in chunk.chunks_mut(n_out).enumerate() {
                let p = base + i;
                arm.forward(&stacked[p * uc..(p + 1) * uc], shift, out, &mut scratch);
            }
        };
        #[cfg(feature = "std")]
        if uh * uw >= 16384 && wave_threads() > 1 {
            let np = uh * uw;
            let per = np.div_ceil(wave_threads().min(np.div_ceil(4096)));
            let eval = &eval;
            std::thread::scope(|s| {
                let mut chunks = small.chunks_mut(per * n_out);
                let first = chunks.next().unwrap();
                for (c, chunk) in chunks.enumerate() {
                    s.spawn(move || eval((c + 1) * per, chunk));
                }
                eval(0, first);
            });
        } else {
            eval(0, &mut small);
        }
        #[cfg(not(feature = "std"))]
        eval(0, &mut small);
    }

    // Nearest ×2 then crop. Upstream round-trips through f32 here.
    let mut data = vec![0i64; h_i * w_i * n_out];
    for y in 0..h_i {
        for x in 0..w_i {
            let s = ((y / 2) * uw + x / 2) * n_out;
            let d = (y * w_i + x) * n_out;
            for k in 0..n_out {
                data[d + k] = small[s + k] as f32 as i64;
            }
        }
    }
    InterContext { c: n_out, data }
}

/// Where the symbols of a grid come from (decoding) or go to (encoding).
pub(crate) trait SymbolCoder {
    /// Codes the value at `(y, x)` under model `(idx_mu, idx_scale)` and returns it.
    fn code(&mut self, y: usize, x: usize, idx_mu: usize, idx_scale: usize) -> Result<i32>;
}

impl SymbolCoder for RangeDecoder<'_> {
    #[inline]
    fn code(&mut self, _: usize, _: usize, idx_mu: usize, idx_scale: usize) -> Result<i32> {
        self.decode(idx_mu, idx_scale)
    }
}

#[cfg(feature = "encoder")]
/// Encodes a known grid (`w`-wide, row-major) into a range encoder.
pub(crate) struct GridEncoder<'a> {
    pub(crate) rc: &'a mut RangeEncoder,
    pub(crate) values: &'a [i64],
    pub(crate) w: usize,
}

#[cfg(feature = "encoder")]
impl SymbolCoder for GridEncoder<'_> {
    #[inline]
    fn code(&mut self, y: usize, x: usize, idx_mu: usize, idx_scale: usize) -> Result<i32> {
        let v = self.values[y * self.w + x];
        let v = i32::try_from(v).map_err(|_| Error::Invalid("latent value outside [-64, 63]"))?;
        self.rc.encode(v, idx_mu, idx_scale)?;
        Ok(v)
    }
}

const MU_SHIFT: u32 = 2 * WEIGHT_SHIFT - N_FRAC_BIT_MU_SCALE;
const MU_MIN_FP: i64 = -64 << N_FRAC_BIT_MU_SCALE;
const LOG_SCALE_MIN_FP: i64 = -5 << N_FRAC_BIT_MU_SCALE;

/// Shared read-only state for one grid's ARM evaluation.
struct GridCtx<'a> {
    arm: &'a FixedArm,
    /// Padded raster of already-coded symbols: `i8` atomics because the
    /// wavefront workers read it while only the serial phase writes (the
    /// diagonal barrier provides the happens-before, so Relaxed suffices).
    padded: &'a [AtomicI8],
    offsets: &'a [isize],
    inter: Option<&'a InterContext>,
    w: usize,
    h: usize,
    pw: usize,
    n_spatial: usize,
}

/// ARM evaluation for one pixel: model index `(idx_mu, idx_scale)`.
/// Pure function of the already-coded neighbourhood — safe to run on any thread.
#[inline]
fn eval_px(
    g: &GridCtx,
    ctx: &mut [i64],
    scratch: &mut [[i64; MAX_WIDTH]; 2],
    y: usize,
    x: usize,
) -> (usize, usize) {
    let centre = (y + PAD) * g.pw + x + PAD;
    for (c, &off) in ctx.iter_mut().zip(g.offsets) {
        *c = g.padded[(centre as isize + off) as usize].load(Ordering::Relaxed) as i64;
    }
    if let Some(ic) = g.inter {
        let p = (y * g.w + x) * ic.c;
        ctx[g.n_spatial..].copy_from_slice(&ic.data[p..p + ic.c]);
    }
    let mut out = [0i64; 2];
    g.arm.forward(ctx, MU_SHIFT, &mut out, scratch);
    let im = (out[0] - MU_MIN_FP).clamp(0, N_MU as i64 - 1) as usize;
    let is = (out[1] - LOG_SCALE_MIN_FP).clamp(0, N_SCALE_IDX as i64 - 1) as usize;
    (im, is)
}

/// Serial diagonal traversal: upstream's symbol order for wide grids.
fn serial_diagonals(
    g: &GridCtx,
    rc: &mut impl SymbolCoder,
    step: usize,
    max_t: usize,
    ctx: &mut [i64],
    scratch: &mut [[i64; MAX_WIDTH]; 2],
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    for t in 0..=max_t {
        check()?;
        // Smallest y with x = t - step*y <= w - 1.
        let y0 = if t >= g.w { (t - g.w) / step + 1 } else { 0 };
        let mut y = y0;
        while y < g.h && step * y <= t {
            let x = t - step * y;
            let (im, is) = eval_px(g, ctx, scratch, y, x);
            let sym = rc.code(y, x, im, is)?;
            g.padded[(y + PAD) * g.pw + x + PAD].store(sym as i8, Ordering::Relaxed);
            y += 1;
        }
    }
    Ok(())
}

/// Parallel variant of [`serial_diagonals`]: within one diagonal every ARM
/// evaluation reads only earlier diagonals, so workers compute the model
/// indices while the range coder runs serially in upstream order on the
/// calling thread. Output is identical to the serial traversal — only the
/// thread that evaluates the ARM differs. Returns `false` (having run
/// nothing) when the grid is small, one thread is available, or the zencodec
/// sequential policy is installed.
#[cfg(feature = "std")]
fn wave_diagonals(
    g: &GridCtx,
    rc: &mut impl SymbolCoder,
    step: usize,
    max_t: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<bool> {
    use core::sync::atomic::{AtomicU32, AtomicUsize};
    use std::thread;

    // Barriers cost ~1µs per diagonal, so parallelize only when diagonals
    // carry enough work (~32+ pixels) and the grid is worth spawning for.
    // Measured: −1 ms at 256² (26 px/diagonal), −35% at 384²/512² (38–51 px).
    const MIN_DIAG: usize = 32;
    const MIN_PX: u64 = 65_536;
    if (g.w as u64) * (g.h as u64) < MIN_PX || g.h.min(g.w / step + 1) < MIN_DIAG {
        return Ok(false);
    }
    let threads = wave_threads();
    if threads < 2 {
        return Ok(false);
    }
    let n_ctx = g.n_spatial + g.inter.map_or(0, |c| c.c);

    // Per-diagonal job slots: `h` is the largest possible diagonal.
    let diag_px: Vec<AtomicU32> = (0..g.h).map(|_| AtomicU32::new(0)).collect();
    let out_idx: Vec<AtomicU32> = (0..g.h).map(|_| AtomicU32::new(0)).collect();
    let diag_len = AtomicUsize::new(0);
    // Diagonal currently offered for evaluation (t + 1); `usize::MAX` quits.
    let epoch = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);

    let eval = |id: usize, n: usize, ctx: &mut [i64], scratch: &mut [[i64; MAX_WIDTH]; 2]| {
        let mut i = id;
        while i < n {
            let p = diag_px[i].load(Ordering::Relaxed);
            let (im, is) = eval_px(g, ctx, scratch, (p >> 16) as usize, (p & 0xffff) as usize);
            out_idx[i].store((im as u32) << 16 | is as u32, Ordering::Relaxed);
            i += threads;
        }
    };

    thread::scope(|s| {
        for id in 1..threads {
            let (eval, epoch, done, diag_len) = (&eval, &epoch, &done, &diag_len);
            s.spawn(move || {
                let mut ctx = vec![0i64; n_ctx];
                let mut scratch = [[0i64; MAX_WIDTH]; 2];
                let mut last = 0usize;
                loop {
                    let e = epoch.load(Ordering::Acquire);
                    if e == last {
                        core::hint::spin_loop();
                        continue;
                    }
                    if e == usize::MAX {
                        return;
                    }
                    last = e;
                    eval(id, diag_len.load(Ordering::Relaxed), &mut ctx, &mut scratch);
                    done.fetch_add(1, Ordering::Release);
                }
            });
        }

        let mut ctx = vec![0i64; n_ctx];
        let mut scratch = [[0i64; MAX_WIDTH]; 2];
        let result = (|| -> Result<()> {
            for t in 0..=max_t {
                check()?;
                let mut n = 0;
                let y0 = if t >= g.w { (t - g.w) / step + 1 } else { 0 };
                let mut y = y0;
                while y < g.h && step * y <= t {
                    diag_px[n].store((y as u32) << 16 | (t - step * y) as u32, Ordering::Relaxed);
                    n += 1;
                    y += 1;
                }
                diag_len.store(n, Ordering::Relaxed);
                done.store(0, Ordering::Relaxed);
                epoch.store(t + 1, Ordering::Release);
                eval(0, n, &mut ctx, &mut scratch);
                while done.load(Ordering::Acquire) + 1 < threads {
                    core::hint::spin_loop();
                }
                for i in 0..n {
                    let p = diag_px[i].load(Ordering::Relaxed);
                    let o = out_idx[i].load(Ordering::Relaxed);
                    let (y, x) = ((p >> 16) as usize, (p & 0xffff) as usize);
                    let sym = rc.code(y, x, (o >> 16) as usize, (o & 0xffff) as usize)?;
                    g.padded[(y + PAD) * g.pw + x + PAD].store(sym as i8, Ordering::Relaxed);
                }
            }
            Ok(())
        })();
        epoch.store(usize::MAX, Ordering::Release);
        result
    })?;
    Ok(true)
}

/// Scoped install of a sequential-threading policy (the zencodec
/// `Sequential`/`SingleThread` limit, on either side of the codec):
/// [`wave_threads`] yields 1 while the guard lives, so the wavefront diagonal
/// evaluation and the IFCE pass run on the calling thread only (identical
/// output).
#[cfg(feature = "std")]
mod decode_seq {
    std::thread_local! {
        static SEQ: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    }
    pub(crate) fn is_sequential() -> bool {
        SEQ.with(|c| c.get())
    }
    pub(crate) struct SequentialThreads(());
    impl SequentialThreads {
        /// Dead in builds that decode but never install a threading policy.
        #[allow(dead_code)]
        pub(crate) fn install() -> Self {
            SEQ.with(|c| c.set(true));
            Self(())
        }
    }
    impl Drop for SequentialThreads {
        fn drop(&mut self) {
            SEQ.with(|c| c.set(false));
        }
    }
}
#[cfg(feature = "std")]
#[allow(unused_imports)]
pub(crate) use decode_seq::SequentialThreads;
#[cfg(feature = "std")]
use decode_seq::is_sequential;

/// Threads for the wavefront diagonal evaluation: available parallelism
/// capped at 8 (diagonals are short), 1 under the sequential policy.
#[cfg(feature = "std")]
fn wave_threads() -> usize {
    let n = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(8);
    if is_sequential() { 1 } else { n }
}

/// Decodes (or, with a [`GridEncoder`], encodes) one `h × w` grid in upstream's symbol order.
pub(crate) fn code_grid(
    rc: &mut impl SymbolCoder,
    arm: &FixedArm,
    window: &[usize],
    inter: Option<&InterContext>,
    h: usize,
    w: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Grid> {
    let n_spatial = window.len();
    let n_inter = inter.map_or(0, |c| c.c);
    if arm.n_in() != n_spatial + n_inter || arm.n_out() != 2 {
        return Err(Error::Invalid("ARM dimensions do not match its contexts"));
    }
    let pw = w + 2 * PAD;
    let padded: Vec<AtomicI8> = (0..(h + 2 * PAD) * pw).map(|_| AtomicI8::new(0)).collect();
    // Offsets (in padded raster order) from the decoded value to each context cell.
    let offsets: Vec<isize> = window
        .iter()
        .map(|&cell| {
            let dy = (cell / MASK_SIZE) as isize - PAD as isize;
            let dx = (cell % MASK_SIZE) as isize - PAD as isize;
            dy * pw as isize + dx
        })
        .collect();
    let g = GridCtx {
        arm,
        padded: &padded,
        offsets: &offsets,
        inter,
        w,
        h,
        pw,
        n_spatial,
    };

    let mut ctx = vec![0i64; n_spatial + n_inter];
    let mut scratch = [[0i64; MAX_WIDTH]; 2];

    if w <= MASK_SIZE {
        for y in 0..h {
            check()?;
            for x in 0..w {
                let (im, is) = eval_px(&g, &mut ctx, &mut scratch, y, x);
                let sym = rc.code(y, x, im, is)?;
                g.padded[(y + PAD) * pw + x + PAD].store(sym as i8, Ordering::Relaxed);
            }
        }
    } else if h > 0 {
        let step = MASK_SIZE + 1;
        let max_t = (w - 1) + step * (h - 1);
        #[cfg(feature = "std")]
        let serial = !wave_diagonals(&g, rc, step, max_t, check)?;
        #[cfg(not(feature = "std"))]
        let serial = true;
        if serial {
            serial_diagonals(&g, rc, step, max_t, &mut ctx, &mut scratch, check)?;
        }
    }

    let mut data = vec![0i64; h * w];
    for y in 0..h {
        let row = &padded[(y + PAD) * pw + PAD..(y + PAD) * pw + PAD + w];
        for (d, a) in data[y * w..(y + 1) * w].iter_mut().zip(row) {
            *d = a.load(Ordering::Relaxed) as i64;
        }
    }
    Ok(Grid { h, w, data })
}

#[cfg(all(test, feature = "std", feature = "encoder"))]
mod tests {
    use super::*;
    use crate::arm::FixedLinear;
    use crate::rangecoder::RangeEncoder;

    /// 6-in ARM with one hidden layer — dims don't matter for the round trip,
    /// only that both coding directions run the same traversal.
    fn test_arm() -> FixedArm {
        let layer = |n_in, n_out, seed| FixedLinear {
            n_in,
            n_out,
            w: (0..n_in * n_out)
                .map(|i| (((i + seed) % 11) as i64 - 5) << 10)
                .collect(),
            b: (0..n_out).map(|i| ((i + seed) % 7) as i64 - 3).collect(),
        };
        FixedArm {
            layers: vec![layer(6, 6, 0), layer(6, 2, 1)],
            stabiliser: layer(6, 2, 2),
        }
    }

    /// The wavefront traversal (>= 32 px per diagonal under `std`) must produce
    /// the same bytes and grid as the serial traversal. 512×128 hits the gate.
    #[test]
    fn wavefront_matches_serial() {
        let (h, w) = (128usize, 512usize);
        let arm = test_arm();
        let window = context_window_indices(6).unwrap();
        let values: Vec<i64> = (0..h * w)
            .map(|i| ((i * 31 + 7) % 127) as i64 - 63)
            .collect();
        let mut check = || -> Result<()> { Ok(()) };

        // Sequential policy → serial traversal for both directions.
        let (bytes_seq, grid_seq);
        {
            let _seq = SequentialThreads::install();
            let mut enc = RangeEncoder::new();
            {
                let mut ge = GridEncoder {
                    rc: &mut enc,
                    values: &values,
                    w,
                };
                grid_seq = code_grid(&mut ge, &arm, &window, None, h, w, &mut check).unwrap();
            }
            bytes_seq = enc.finish();
        }
        // Default policy → wavefront: identical bitstream and grid.
        let mut enc = RangeEncoder::new();
        {
            let mut ge = GridEncoder {
                rc: &mut enc,
                values: &values,
                w,
            };
            let grid_par = code_grid(&mut ge, &arm, &window, None, h, w, &mut check).unwrap();
            assert_eq!(grid_seq.data, grid_par.data);
        }
        let bytes_par = enc.finish();
        assert_eq!(
            bytes_seq, bytes_par,
            "parallel encode changed the bitstream"
        );

        // Decode either stream under either policy: identical grid.
        for (bytes, seq) in [(&bytes_seq, true), (&bytes_par, false), (&bytes_par, true)] {
            let _seq = seq.then(SequentialThreads::install);
            let mut dec = RangeDecoder::new(bytes).unwrap();
            let g = code_grid(&mut dec, &arm, &window, None, h, w, &mut check).unwrap();
            assert_eq!(g.data, values);
        }
    }

    /// The parallel IFCE pass (>= 16384 stacked pixels under `std`) must
    /// produce the same context as the serial pass. One 128² coded grid hits
    /// the gate exactly.
    #[test]
    fn ifce_parallel_matches_serial() {
        let layer = |n_in, n_out, seed| FixedLinear {
            n_in,
            n_out,
            w: (0..n_in * n_out)
                .map(|i| (((i + seed) % 11) as i64 - 5) << 10)
                .collect(),
            b: (0..n_out).map(|i| ((i + seed) % 7) as i64 - 3).collect(),
        };
        // 1-input IFCE arm: `stack_nearest` over a single coded grid has c=1.
        let arm = FixedArm {
            layers: vec![layer(1, 4, 0), layer(4, 3, 1)],
            stabiliser: layer(1, 3, 2),
        };
        let coded = Grid {
            h: 128,
            w: 128,
            data: (0..128 * 128)
                .map(|i| ((i * 17 + 3) % 127) as i64 - 63)
                .collect(),
        };
        let serial = {
            let _seq = SequentialThreads::install();
            inter_context(&[&coded], (256, 256), Some(&arm), 3)
        };
        let par = inter_context(&[&coded], (256, 256), Some(&arm), 3);
        assert_eq!(serial.data, par.data);
    }
}
