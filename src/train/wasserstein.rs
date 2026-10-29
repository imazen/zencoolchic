//! Upstream `training/metrics/wasserstein.py` (`a6fe38a`): the Wasserstein distance on
//! VGG16 features used by `--tune wasserstein` (`dist_weight {"mse": 0.2, "wasserstein":
//! 0.8/200}`). Inspired by Google codex's `wasserstein.py`.
//!
//! The feature extractor is `torchvision vgg16.features[:23]` (weights IMAGENET1K_V1):
//! ten 3×3 zero-padded convolutions, each followed by ReLU, MaxPool2d(2) after convs
//! 1, 3 and 6 (feature indices 4, 9, 16); taps are the post-ReLU outputs right before
//! each pool and at the end (feature indices 3, 8, 15, 22). The weights are not part of
//! the crate: [`Vgg16::from_f32s`] parses the flat dump of `scripts/oracle/dump_vgg.py`.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use rayon::prelude::*;

use crate::simd::axpy;

/// (input, output) channels of `vgg16.features` convolutions {0,2,5,7,10,12,14,17,19,21}.
const DIMS: [(usize, usize); 10] = [
    (3, 64),
    (64, 64),
    (64, 128),
    (128, 128),
    (128, 256),
    (256, 256),
    (256, 256),
    (256, 512),
    (512, 512),
    (512, 512),
];

/// Convs (of the 10) after which a MaxPool2d(2, 2) sits: feature index 4, 9, 16.
const POOL_AFTER: [usize; 3] = [1, 3, 6];
/// Convs whose post-ReLU output is a wasserstein tap: feature index 3, 8, 15, 22.
const TAPS: [usize; 4] = [1, 3, 6, 9];

/// `torchvision vgg16.features[:23]` weights (ImageNet), frozen.
pub struct Vgg16 {
    /// `w` is `[c_out][c_in][3][3]`, `b` is `[c_out]`, in [`DIMS`] order.
    convs: Vec<(Vec<f32>, Vec<f32>)>,
    /// `(c_in, c_out)` of `convs[i]`; [`DIMS`] for the real net.
    dims: Vec<(usize, usize)>,
}

impl core::fmt::Debug for Vgg16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Vgg16")
            .field("convs", &self.convs.len())
            .finish()
    }
}

impl Vgg16 {
    /// Number of f32 coefficients of one dump: `sum(co·ci·9 + co)` over [`DIMS`].
    pub const N_WEIGHTS: usize = 7_635_264;

    /// Parses the flat f32 dump of `scripts/oracle/dump_vgg.py`: for every conv in
    /// [`DIMS`] order, `co·ci·9` weights then `co` biases.
    pub fn from_f32s(v: &[f32]) -> crate::Result<Self> {
        if v.len() != Self::N_WEIGHTS {
            return Err(crate::Error::InvalidArgument("vgg16 weight count"));
        }
        Self::from_dims(&DIMS, v)
    }

    /// Same flat layout as [`Vgg16::from_f32s`] with explicit `(c_in, c_out)` conv dims.
    /// The pool/tap positions stay [`POOL_AFTER`]/[`TAPS`]: oracle miniatures share the
    /// 23-slot structure with smaller channels.
    pub(crate) fn from_dims(dims: &[(usize, usize)], v: &[f32]) -> crate::Result<Self> {
        if dims.len() != DIMS.len() {
            return Err(crate::Error::InvalidArgument("vgg conv count"));
        }
        let mut convs = Vec::with_capacity(dims.len());
        let mut off = 0;
        for &(ci, co) in dims {
            let end = off + co * ci * 9 + co;
            if end > v.len() {
                return Err(crate::Error::InvalidArgument("vgg weight count"));
            }
            convs.push((
                v[off..off + co * ci * 9].to_vec(),
                v[off + co * ci * 9..end].to_vec(),
            ));
            off = end;
        }
        if off != v.len() {
            return Err(crate::Error::InvalidArgument("vgg weight count"));
        }
        Ok(Self {
            convs,
            dims: dims.to_vec(),
        })
    }

    /// Little-endian f32 byte dump (same layout as [`Vgg16::from_f32s`]).
    pub fn from_le_bytes(v: &[u8]) -> crate::Result<Self> {
        if v.len() != Self::N_WEIGHTS * 4 {
            return Err(crate::Error::InvalidArgument("vgg16 weight bytes"));
        }
        Self::from_f32s(
            &v.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect::<Vec<_>>(),
        )
    }

    /// The four post-ReLU taps and, when `tape` is given, everything backward needs.
    fn features(&self, img: &[f32], h: usize, w: usize, tape: Option<&mut VggTape>) -> Vec<Tap> {
        let mut tape = tape;
        let (mut c, mut hh, mut ww) = (3usize, h, w);
        let mut x = img.to_vec();
        let mut taps = Vec::with_capacity(4);
        for (i, (wc, b)) in self.convs.iter().enumerate() {
            x = conv_relu(&x, c, hh, ww, self.dims[i].1, wc, b);
            c = self.dims[i].1;
            if let Some(t) = tape.as_deref_mut() {
                t.outs.push(x.clone());
                t.dims.push((c, hh, ww));
            }
            if TAPS.contains(&i) {
                taps.push(Tap {
                    c,
                    h: hh,
                    w: ww,
                    data: x.clone(),
                });
            }
            if POOL_AFTER.contains(&i) {
                let (p, idx, nh, nw) = maxpool(&x, c, hh, ww);
                if let Some(t) = tape.as_deref_mut() {
                    t.pool_idx.push(idx);
                    t.pool_in.push((hh, ww));
                }
                x = p;
                hh = nh;
                ww = nw;
            }
        }
        taps
    }

    /// Gradient with respect to `img`, given the tap gradients.
    fn backward(&self, tape: &VggTape, taps: [Vec<f32>; 4]) -> Vec<f32> {
        let mut d = taps[3].clone(); // grad at outs[9]
        for i in (0..10).rev() {
            let (c, hh, ww) = tape.dims[i];
            debug_assert_eq!(d.len(), c * hh * ww);
            // ReLU: the saved output is its own mask.
            for (dv, &o) in d.iter_mut().zip(&tape.outs[i]) {
                if o <= 0.0 {
                    *dv = 0.0;
                }
            }
            let ci = self.dims[i].0;
            d = conv_back(&d, c, hh, ww, ci, &self.convs[i].0);
            if i == 0 {
                break;
            }
            if let Some(p) = POOL_AFTER.iter().position(|&a| a == i - 1) {
                // conv i read pool p's output: scatter back to outs[i-1]'s shape.
                let (ph, pw) = tape.pool_in[p];
                d = unpool(&d, ci, ph, pw, &tape.pool_idx[p]);
            }
            if let Some(t) = TAPS[..3].iter().position(|&a| a == i - 1) {
                debug_assert_eq!(d.len(), taps[t].len(), "i={i} tap={t}");
                for (a, b) in d.iter_mut().zip(&taps[t]) {
                    *a += *b;
                }
            }
        }
        d
    }
}

/// Feature planes of one tap, channel-major `[c][h][w]`.
struct Tap {
    c: usize,
    h: usize,
    w: usize,
    data: Vec<f32>,
}

/// Forward tape of one [`Vgg16::features`] call.
struct VggTape {
    /// Post-ReLU outputs of every conv.
    outs: Vec<Vec<f32>>,
    /// (channels, h, w) of `outs[i]`.
    dims: Vec<(usize, usize, usize)>,
    /// Flat input index of the max, per output cell, per pool.
    pool_idx: Vec<Vec<u32>>,
    /// Input (h, w) of every pool.
    pool_in: Vec<(usize, usize)>,
}

/// Zero-pads a plane by 1 on every side.
fn zpad(x: &[f32], h: usize, w: usize) -> Vec<f32> {
    let pw = w + 2;
    let mut out = vec![0.0f32; (h + 2) * pw];
    for y in 0..h {
        out[(y + 1) * pw + 1..(y + 1) * pw + 1 + w].copy_from_slice(&x[y * w..(y + 1) * w]);
    }
    out
}

/// Zero-padded 3×3 conv (stride 1) + ReLU; `x` is `[ci][h][w]`, `wc` is `[co][ci][9]`.
fn conv_relu(
    x: &[f32],
    ci: usize,
    h: usize,
    w: usize,
    co: usize,
    wc: &[f32],
    b: &[f32],
) -> Vec<f32> {
    let pw = w + 2;
    let n = h * w;
    let xp: Vec<Vec<f32>> = (0..ci)
        .map(|i| zpad(&x[i * n..(i + 1) * n], h, w))
        .collect();
    let mut out = vec![0.0f32; co * n];
    out.par_chunks_mut(n).enumerate().for_each(|(o, dst)| {
        dst.fill(b[o]);
        for i in 0..ci {
            let wk = &wc[(o * ci + i) * 9..(o * ci + i + 1) * 9];
            let xi = &xp[i];
            for y in 0..h {
                let row = &mut dst[y * w..(y + 1) * w];
                for ky in 0..3 {
                    let r = &xi[(y + ky) * pw..];
                    for kx in 0..3 {
                        axpy(row, wk[ky * 3 + kx], &r[kx..kx + w]);
                    }
                }
            }
        }
        for v in dst.iter_mut() {
            *v = v.max(0.0);
        }
    });
    out
}

/// Input gradient of [`conv_relu`]'s conv: `dout` is `[co][h][w]`, returns `[ci][h][w]`
/// (correlation of the zero-padded `dout` with the 180°-rotated kernels).
fn conv_back(dout: &[f32], co: usize, h: usize, w: usize, ci: usize, wc: &[f32]) -> Vec<f32> {
    let pw = w + 2;
    let n = h * w;
    let dp: Vec<Vec<f32>> = (0..co)
        .map(|o| zpad(&dout[o * n..(o + 1) * n], h, w))
        .collect();
    let mut dx = vec![0.0f32; ci * n];
    dx.par_chunks_mut(n).enumerate().for_each(|(i, dst)| {
        for o in 0..co {
            let wk = &wc[(o * ci + i) * 9..(o * ci + i + 1) * 9];
            let dpo = &dp[o];
            for y in 0..h {
                let row = &mut dst[y * w..(y + 1) * w];
                for ky in 0..3 {
                    let r = &dpo[(y + ky) * pw..];
                    for kx in 0..3 {
                        axpy(row, wk[(2 - ky) * 3 + (2 - kx)], &r[kx..kx + w]);
                    }
                }
            }
        }
    });
    dx
}

/// `MaxPool2d(2, 2)`: `floor(h/2)` outputs, first max wins (torch index order).
fn maxpool(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, Vec<u32>, usize, usize) {
    let (oh, ow) = (h / 2, w / 2);
    let mut out = vec![0.0f32; c * oh * ow];
    let mut idx = vec![0u32; c * oh * ow];
    for ch in 0..c {
        let src = &x[ch * h * w..(ch + 1) * h * w];
        for oy in 0..oh {
            for ox in 0..ow {
                let base = 2 * oy * w + 2 * ox;
                let (mut m, mut mi) = (src[base], base);
                for k in 1..4usize {
                    let p = (2 * oy + k / 2) * w + 2 * ox + (k % 2);
                    if src[p] > m {
                        m = src[p];
                        mi = p;
                    }
                }
                let o = ch * oh * ow + oy * ow + ox;
                out[o] = m;
                idx[o] = mi as u32;
            }
        }
    }
    (out, idx, oh, ow)
}

/// Scatters pooled-output gradients back to the recorded argmax positions: `dout` is
/// `[c][ph][pw]`, returns `[c][ih][iw]` where `(ih, iw)` are the pool's input dims.
fn unpool(dout: &[f32], c: usize, ih: usize, iw: usize, idx: &[u32]) -> Vec<f32> {
    let (oh, ow) = (ih / 2, iw / 2);
    let mut dx = vec![0.0f32; c * ih * iw];
    for ch in 0..c {
        let d = &dout[ch * oh * ow..(ch + 1) * oh * ow];
        let i = &idx[ch * oh * ow..(ch + 1) * oh * ow];
        let dst = &mut dx[ch * ih * iw..(ch + 1) * ih * iw];
        for (o, &p) in d.iter().zip(i) {
            dst[p as usize] += *o;
        }
    }
    dx
}

// -------------------------------------------------------------------------------------
// Multiscale statistics (`compute_multiscale_stats`, `wasserstein_distortion`).

/// `outer([0.25, 0.5, 0.25])`: the 3×3 low-pass kernel.
const LP: [f32; 9] = [
    0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625,
];

/// Upstream `num_levels = 5`: stats levels 0–4 under `wd_maps[i] = stats level i−1`.
/// The level weight is `relu(1 − |σ − i|)` with `σ` the constant `log2_sigma = 3` map
/// `lowpass(·, stride=2)`ed once per level — border attenuation makes it nonzero at
/// `i = 2` (stats level 1) and `i = 3` (stats level 2) only, so levels 0–2 suffice.
const LEVELS: usize = 3;

/// `relu(1 − |σ_l − (l + 1)|)`: the weight map of stats level `l` (upstream `i = l+1`).
/// `σ_l = 3 · r_h ⊗ r_w` where `r` is a constant-ones signal `lowpass(·, stride=2)`ed
/// `l` times; since `σ_l ≤ 3` the weight is zero for `l ∉ {1, 2}`.
fn sigma_weight(h: usize, w: usize, l: usize) -> Vec<f32> {
    debug_assert!(l == 1 || l == 2);
    // One stride-2, pad-1 pass of the separable [0.25, 0.5, 0.25] kernel.
    let lp = |v: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0; v.len().div_ceil(2)];
        for (y, o) in out.iter_mut().enumerate() {
            for (k, kv) in [0.25f32, 0.5, 0.25].into_iter().enumerate() {
                let i = 2 * y + k;
                if i > 0 && i <= v.len() {
                    *o += kv * v[i - 1];
                }
            }
        }
        out
    };
    let (mut ry, mut rx) = (vec![1.0; h], vec![1.0; w]);
    for _ in 0..l {
        ry = lp(&ry);
        rx = lp(&rx);
    }
    let mut wt = vec![0.0; ry.len() * rx.len()];
    for (y, row) in wt.chunks_mut(rx.len()).enumerate() {
        for (x, wj) in row.iter_mut().enumerate() {
            *wj = (1.0 - (3.0 * ry[y] * rx[x] - (l + 1) as f32).abs()).max(0.0);
        }
    }
    wt
}
/// `safe_clamp_min(·, 5e-7)` on variances, clamping `sqrt`'s input. Despite its
/// docstring, upstream's version is `x - x + clamp(x)` under `no_grad`: it detaches
/// completely, so the std term carries no gradient (see [`tap_grad`]).
const VAR_MIN: f32 = 5e-7;

/// Stride-1 low-pass of one plane (zero padding). Self-adjoint: the kernel is symmetric.
fn lowpass(x: &[f32], h: usize, w: usize) -> Vec<f32> {
    let pw = w + 2;
    let xp = zpad(x, h, w);
    let mut out = vec![0.0f32; h * w];
    for y in 0..h {
        let row = &mut out[y * w..(y + 1) * w];
        for ky in 0..3 {
            let r = &xp[(y + ky) * pw..];
            for kx in 0..3 {
                axpy(row, LP[ky * 3 + kx], &r[kx..kx + w]);
            }
        }
    }
    out
}

/// `[.., ::2, ::2]` of one plane, `ceil(h/2) × ceil(w/2)`.
fn subsample(x: &[f32], h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (oh, ow) = (h.div_ceil(2), w.div_ceil(2));
    let mut out = vec![0.0f32; oh * ow];
    for oy in 0..oh {
        for ox in 0..ow {
            out[oy * ow + ox] = x[2 * oy * w + 2 * ox];
        }
    }
    (out, oh, ow)
}

/// Backward of [`subsample`]: adds the gradient at even indices of an `h × w` plane.
fn subsample_back(d: &[f32], ih: usize, iw: usize, dst: &mut [f32]) {
    let (oh, ow) = (ih.div_ceil(2), iw.div_ceil(2));
    for oy in 0..oh {
        for ox in 0..ow {
            dst[2 * oy * iw + 2 * ox] += d[oy * ow + ox];
        }
    }
}

/// `lowpass(·, stride=1)` on every channel of a `[c][h][w]` plane stack.
fn lowpass_planes(x: &[f32], c: usize, h: usize, w: usize) -> Vec<f32> {
    let n = h * w;
    let mut out = vec![0.0f32; c * n];
    out.par_chunks_mut(n).enumerate().for_each(|(i, dst)| {
        dst.copy_from_slice(&lowpass(&x[i * n..(i + 1) * n], h, w));
    });
    out
}

/// `subsample` on every channel of a `[c][h][w]` plane stack.
fn subsample_planes(x: &[f32], c: usize, h: usize, w: usize) -> (Vec<f32>, usize, usize) {
    let (oh, ow) = (h.div_ceil(2), w.div_ceil(2));
    let n = h * w;
    let mut out = vec![0.0f32; c * oh * ow];
    out.par_chunks_mut(oh * ow)
        .enumerate()
        .for_each(|(i, dst)| {
            let (s, _, _) = subsample(&x[i * n..(i + 1) * n], h, w);
            dst.copy_from_slice(&s);
        });
    (out, oh, ow)
}

/// Means and (clamped) standard deviations of the [`LEVELS`] scales of one tap.
struct TapStats {
    c: usize,
    dims: [(usize, usize); LEVELS],
    means: [Vec<f32>; LEVELS],
    stds: [Vec<f32>; LEVELS],
}

/// `compute_multiscale_stats(f, LEVELS)`: `means[l] = lowpass(feat_l)`,
/// `stds[l] = sqrt(max(lowpass(feat_l²) − means[l]², 5e-7))`,
/// `feat_{l+1} = means[l][::2,::2]` (and `feat²_{l+1}` from the low-passed squares).
fn multiscale(f: &[f32], c: usize, h: usize, w: usize) -> TapStats {
    let mut feats = f.to_vec();
    let mut sqs: Vec<f32> = feats.iter().map(|v| v * v).collect();
    let (mut hh, mut ww) = (h, w);
    let mut stats = TapStats {
        c,
        dims: [(hh, ww); LEVELS],
        means: Default::default(),
        stds: Default::default(),
    };
    for l in 0..LEVELS {
        let m = lowpass_planes(&feats, c, hh, ww);
        let p = lowpass_planes(&sqs, c, hh, ww);
        stats.dims[l] = (hh, ww);
        stats.stds[l] = m
            .iter()
            .zip(&p)
            .map(|(&mm, &pp)| libm::sqrtf((pp - mm * mm).max(VAR_MIN)))
            .collect();
        let (nf, nh, nw) = subsample_planes(&m, c, hh, ww);
        let (ns, _, _) = subsample_planes(&p, c, hh, ww);
        stats.means[l] = m;
        feats = nf;
        sqs = ns;
        hh = nh;
        ww = nw;
    }
    stats
}

/// Wasserstein distance between the features of a decoded picture and cached target
/// features — `WassersteinLoss` (`wasserstein_fn`).
pub struct Wasserstein {
    vgg: Arc<Vgg16>,
    /// Target stats of every tap, in tap order.
    target: Vec<TapStats>,
}

/// Tape of one [`Wasserstein::forward`], for [`Wasserstein::backward`].
pub(crate) struct WassTape {
    vgg: VggTape,
    dec: Vec<TapStats>,
}

impl Wasserstein {
    /// Runs the feature extractor on `target` once and caches its multiscale statistics
    /// (upstream caches `target_features`; the derived stats are equivalent). `target`
    /// is `[3][h][w]` in `[0, 1]`.
    pub fn new(vgg: Arc<Vgg16>, target: &[f32], h: usize, w: usize) -> Self {
        let taps = vgg.features(target, h, w, None);
        let target = taps
            .iter()
            .map(|t| multiscale(&t.data, t.c, t.h, t.w))
            .collect();
        Self { vgg, target }
    }

    /// Per-tap distances of `dec`, for oracle debugging.
    #[cfg(test)]
    pub(crate) fn tap_dists(&self, dec: &[f32], h: usize, w: usize) -> Vec<f64> {
        let taps = self.vgg.features(dec, h, w, None);
        taps.iter()
            .map(|t| multiscale(&t.data, t.c, t.h, t.w))
            .zip(&self.target)
            .map(|(a, b)| tap_dist(&a, b))
            .collect()
    }

    /// Tap-feature gradients of `dec` (stats-path backward only), for oracle debugging.
    #[cfg(test)]
    pub(crate) fn tap_grads(&self, dec: &[f32], h: usize, w: usize) -> Vec<Vec<f32>> {
        let (_d, tape) = self.forward(dec, h, w);
        (0..4)
            .map(|i| tap_grad(&tape.dec[i], &self.target[i]))
            .collect()
    }

    /// Raw tap feature planes of `dec`, for oracle debugging.
    #[cfg(test)]
    pub(crate) fn tap_features(&self, dec: &[f32], h: usize, w: usize) -> Vec<Vec<f32>> {
        self.vgg
            .features(dec, h, w, None)
            .into_iter()
            .map(|t| t.data)
            .collect()
    }

    /// Distortion of the four taps of `dec` (`[3][h][w]`): forward only.
    pub fn eval(&self, dec: &[f32], h: usize, w: usize) -> f64 {
        let taps = self.vgg.features(dec, h, w, None);
        taps.iter()
            .map(|t| multiscale(&t.data, t.c, t.h, t.w))
            .zip(&self.target)
            .map(|(a, b)| tap_dist(&a, b))
            .sum()
    }

    /// Distortion + everything [`Self::backward`] needs.
    pub fn forward(&self, dec: &[f32], h: usize, w: usize) -> (f64, WassTape) {
        let mut vgg_tape = VggTape {
            outs: Vec::new(),
            dims: Vec::new(),
            pool_idx: Vec::new(),
            pool_in: Vec::new(),
        };
        let taps = self.vgg.features(dec, h, w, Some(&mut vgg_tape));
        let dec: Vec<TapStats> = taps
            .iter()
            .map(|t| multiscale(&t.data, t.c, t.h, t.w))
            .collect();
        let dist = dec
            .iter()
            .zip(&self.target)
            .map(|(a, b)| tap_dist(a, b))
            .sum();
        (dist, WassTape { vgg: vgg_tape, dec })
    }

    /// Gradient of the distortion with respect to `dec` (`[3][h][w]`).
    pub fn backward(&self, tape: &WassTape) -> Vec<f32> {
        let tap_grads: [Vec<f32>; 4] =
            core::array::from_fn(|i| tap_grad(&tape.dec[i], &self.target[i]));
        self.vgg.backward(&tape.vgg, tap_grads)
    }
}

/// `wasserstein_distortion` of one tap: `Σ_i mean(wd_map_i · relu(1 − |σ − i|))` over
/// the six `wd_maps` (the pixelwise one plus five stats levels). With the constant
/// `log2_sigma = 3` only stats levels 1 and 2 contribute; the sigma map's border
/// attenuation (stride-2 lowpass on a constant) weights their pixels per
/// [`sigma_weight`]. Chunked f64 partials keep the sum thread-count independent.
fn tap_dist(a: &TapStats, b: &TapStats) -> f64 {
    let mut dist = 0.0f64;
    for l in 1..LEVELS {
        let (hl, wl) = a.dims[l];
        let npix = hl * wl;
        let wt = sigma_weight(a.dims[0].0, a.dims[0].1, l);
        debug_assert_eq!(wt.len(), npix);
        const CH: usize = 1 << 14;
        let partial: Vec<f64> = a.means[l]
            .par_chunks(CH)
            .zip(a.stds[l].par_chunks(CH))
            .zip(b.means[l].par_chunks(CH))
            .zip(b.stds[l].par_chunks(CH))
            .enumerate()
            .map(|(ci, (((ma, sa), mb), sb))| {
                let mut m = 0.0f64;
                let base = ci * CH;
                for (o, (((&ma, &sa), &mb), &sb)) in ma.iter().zip(sa).zip(mb).zip(sb).enumerate() {
                    let wj = wt[(base + o) % npix] as f64;
                    let (dm, ds) = (ma - mb, sa - sb);
                    m += wj * (dm * dm + ds * ds) as f64;
                }
                m
            })
            .collect();
        dist += partial.iter().sum::<f64>() / (a.c * npix) as f64;
    }
    dist
}

/// Gradient of [`tap_dist`] with respect to the tap features (`[c][h][w]`).
/// Upstream's `safe_clamp_min` evaluates `x - x + clamp(x)` under `no_grad`, so
/// the returned std is fully detached: the `(sa − sb)²` term contributes to the
/// distance but carries no gradient. Only the mean terms backpropagate.
fn tap_grad(a: &TapStats, b: &TapStats) -> Vec<f32> {
    let c = a.c;
    // Per-level gradients w.r.t. `means[l]`: the chain back to the features runs
    // over levels 0–2; only levels 1 and 2 get a direct loss gradient.
    let mut dm: [Vec<f32>; LEVELS] = Default::default();
    for l in 0..LEVELS {
        dm[l] = vec![0.0; c * a.dims[l].0 * a.dims[l].1];
    }
    for l in 1..LEVELS {
        let (hl, wl) = a.dims[l];
        let npix = hl * wl;
        let wt = sigma_weight(a.dims[0].0, a.dims[0].1, l);
        debug_assert_eq!(wt.len(), npix);
        let inv_n = 1.0 / (c * npix) as f32;
        for ch in 0..c {
            let (dml, mal, mbl) = (
                &mut dm[l][ch * npix..(ch + 1) * npix],
                &a.means[l][ch * npix..(ch + 1) * npix],
                &b.means[l][ch * npix..(ch + 1) * npix],
            );
            for j in 0..npix {
                dml[j] = wt[j] * inv_n * 2.0 * (mal[j] - mbl[j]);
            }
        }
    }
    // Chain rule through `feat_{l+1} = means[l][::2,::2]`, from the deepest
    // contributing level back to the features.
    let mut dfeat = vec![0.0f32; 0];
    for l in (0..3usize).rev() {
        let (hl, wl) = a.dims[l];
        let npix = hl * wl;
        if l + 1 < 3 {
            let (oh, ow) = (hl.div_ceil(2), wl.div_ceil(2));
            for ch in 0..c {
                subsample_back(
                    &dfeat[ch * oh * ow..(ch + 1) * oh * ow],
                    hl,
                    wl,
                    &mut dm[l][ch * npix..(ch + 1) * npix],
                );
            }
        }
        // `lowpass` is self-adjoint: the same kernel transposes it.
        let mut nf = vec![0.0f32; c * npix];
        for ch in 0..c {
            nf[ch * npix..(ch + 1) * npix].copy_from_slice(&lowpass(
                &dm[l][ch * npix..(ch + 1) * npix],
                hl,
                wl,
            ));
        }
        dfeat = nf;
    }
    dfeat
}
