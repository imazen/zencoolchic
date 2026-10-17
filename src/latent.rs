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

use crate::arm::{FixedArm, N_FRAC_BIT_INTER_FT_CTX, N_FRAC_BIT_MU_SCALE, WEIGHT_SHIFT};
use crate::error::{Error, Result};
use crate::rangecoder::{N_MU, N_SCALE_IDX, RangeDecoder};

pub(crate) const MASK_SIZE: usize = 9;
const PAD: usize = (MASK_SIZE - 1) / 2;

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

    // IFCE output at the stacked resolution.
    let mut small = vec![0i64; uh * uw * n_out];
    if let Some(arm) = ifce {
        let shift = 2 * WEIGHT_SHIFT - N_FRAC_BIT_INTER_FT_CTX;
        for p in 0..uh * uw {
            arm.forward(
                &stacked[p * uc..(p + 1) * uc],
                shift,
                &mut small[p * n_out..(p + 1) * n_out],
            );
        }
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

/// Decodes one `h × w` grid.
pub(crate) fn decode_grid(
    rc: &mut RangeDecoder<'_>,
    arm: &FixedArm,
    window: &[usize],
    inter: Option<&InterContext>,
    h: usize,
    w: usize,
) -> Result<Grid> {
    let n_spatial = window.len();
    let n_inter = inter.map_or(0, |c| c.c);
    if arm.n_in() != n_spatial + n_inter || arm.n_out() != 2 {
        return Err(Error::Invalid("ARM dimensions do not match its contexts"));
    }
    let pw = w + 2 * PAD;
    let mut padded = vec![0i64; (h + 2 * PAD) * pw];
    // Offsets (in padded raster order) from the decoded value to each context cell.
    let offsets: Vec<isize> = window
        .iter()
        .map(|&cell| {
            let dy = (cell / MASK_SIZE) as isize - PAD as isize;
            let dx = (cell % MASK_SIZE) as isize - PAD as isize;
            dy * pw as isize + dx
        })
        .collect();

    let mut ctx = vec![0i64; n_spatial + n_inter];
    let mut out = [0i64; 2];
    let shift = 2 * WEIGHT_SHIFT - N_FRAC_BIT_MU_SCALE;
    let mu_min_fp: i64 = -64 << N_FRAC_BIT_MU_SCALE;
    let log_scale_min_fp: i64 = -5 << N_FRAC_BIT_MU_SCALE;

    let mut decode_at = |y: usize, x: usize, padded: &mut Vec<i64>| -> Result<()> {
        let centre = (y + PAD) * pw + x + PAD;
        for (c, &off) in ctx.iter_mut().zip(&offsets) {
            *c = padded[(centre as isize + off) as usize];
        }
        if let Some(ic) = inter {
            let p = (y * w + x) * ic.c;
            ctx[n_spatial..].copy_from_slice(&ic.data[p..p + ic.c]);
        }
        arm.forward(&ctx, shift, &mut out);
        let im = (out[0] - mu_min_fp).clamp(0, N_MU as i64 - 1) as usize;
        let is = (out[1] - log_scale_min_fp).clamp(0, N_SCALE_IDX as i64 - 1) as usize;
        padded[centre] = rc.decode(im, is)? as i64;
        Ok(())
    };

    if w <= MASK_SIZE {
        for y in 0..h {
            for x in 0..w {
                decode_at(y, x, &mut padded)?;
            }
        }
    } else if h > 0 {
        let step = MASK_SIZE + 1;
        let max_t = (w - 1) + step * (h - 1);
        for t in 0..=max_t {
            // Smallest y with x = t - step*y <= w - 1.
            let y0 = if t >= w { (t - w) / step + 1 } else { 0 };
            let mut y = y0;
            while y < h && step * y <= t {
                let x = t - step * y;
                decode_at(y, x, &mut padded)?;
                y += 1;
            }
        }
    }

    let mut data = vec![0i64; h * w];
    for y in 0..h {
        data[y * w..(y + 1) * w]
            .copy_from_slice(&padded[(y + PAD) * pw + PAD..(y + PAD) * pw + PAD + w]);
    }
    Ok(Grid { h, w, data })
}
