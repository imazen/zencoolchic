# Porting status

Upstream: [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) `a6fe38a` (2026-07-17,
v5.0.1), BSD-3-Clause (notice in `upstream-notices/`). Entropy coder: `constriction` 0.4.2
(the Python package's Rust source) with `probability` 0.20.3's Laplace CDF.

## Decoder

| Module | Upstream | Status | Parity evidence |
|---|---|---|---|
| Video / frame / Cool-chic headers | `bitstream/header/*.py` | done (I frames) | 12 oracle streams; `tests/oracle.rs` |
| Network parameters (exp-Golomb, per-module q-step) | `bitstream/neuralnet/*.py` | done | `expgolomb::tests::round_trip_all_orders`; kodim14 |
| Fixed-point ARM / IFCE | `bitstream/component/armint.py` | done | latents bit-exact on 12 streams, incl. no-IFCE and no-stabiliser |
| Wavefront latent decode | `bitstream/component/latent.py` | done (wavefront and width ≤ 9 raster order) | latents bit-exact on 12 streams |
| Range decoder + leaky quantized Laplace | constriction 0.4.2 `queue.rs`, `quantize.rs` | done | latents bit-exact on 12 streams; exhaustive `exp` audit below |
| Upsampling (train-mode kron kernels) | `component/core/upsampling.py` | done, scalar | raw output max abs error ≤ 9.3e-7 (9.24e-7) on 12 streams |
| Synthesis | `component/core/synthesis.py` | done, scalar | as above |
| Final resize (bicubic; nearest / bilinear ported) | `F.interpolate` | bicubic tested (`mid_lat1_debug`, latents from 1/2 resolution); nearest and bilinear are only used by the motion decoder and are untested | `mid_lat1_debug` |
| Common randomness | `component/core/noise.py` | done | `mid_wasserstein_debug` (`--tune wasserstein`): all samples identical |
| YUV 4:2:0 / 4:4:4 output | `io/format/yuv.py` | done | `yuv420_8b_debug`, `yuv444_10b_debug` (10-bit) |
| Inter frames (P/B), warper, global motion | `bitstream/decode.py`, `intercoding/` | not ported: `Error::Unsupported` | — |

### Measured parity (2026-10-17, mac, torch 2.14.1 CPU)

`just oracle` over upstream's `samples/bitstreams/kodim14.cool` plus 11 streams from
`scripts/oracle/make_streams.sh` (crops of kodim14 encoded by upstream; `--debug` streams train
for a few iterations, `_2k` ones for 2000):

| stream | size | latents identical | raw max \|Δ\| | samples off by 1 |
|---|---|---|---|---|
| kodim14 (upstream sample, hop) | 768×512 RGB | 526,272 / 526,272 | 3.6e-7 | 5 / 1,179,648 |
| small64_hop_2k / lop_2k / vhop_2k | 64×64 RGB | 5,482 each | ≤ 3.0e-7 | 0 |
| mid_lat1_debug (latents from 1/2 res, bicubic resize) | 130×97 | 4,409 | 9.2e-7 | 1 / 37,830 |
| mid_noifce_nohyper | 130×97 | 16,930 | 2.4e-7 | 0 |
| mid_nostab_debug (ARM 8,1, no stabilisers) | 130×97 | 17,019 | 1.2e-7 | 1 / 37,830 |
| mid_wasserstein_debug (common randomness) | 130×97 | 17,019 | 2.4e-7 | 0 |
| odd37x23_debug | 37×23 | 1,172 | 3.0e-8 | 0 |
| narrow7x40_debug (raster order) | 7×40 | 397 | 1.2e-7 | 0 |
| yuv420_8b_debug | 48×32 4:2:0 | 2,058 | 6.0e-8 | 0 |
| yuv444_10b_debug | 40×24 4:4:4 10-bit | 1,293 | 0 | 0 |

No sample differs by more than one step.

### Bit-exactness risks

- The leaky Laplace CDF calls `exp` in f64. Upstream gets it from the platform libm through
  Rust std (glibc in Linux wheels, Apple libm on macOS); this port uses the `libm` crate.
  `examples/exp_audit.rs` evaluates every (mu, scale, symbol boundary) the model can produce
  (10,741,612,544 cases): `libm::exp` differs from the platform `exp` in about 8% of them
  (Apple libm 859,272,296; glibc 858,811,983 (2.43 x86_64 on dev; Ubuntu 24.04 aarch64 runner); Windows x64 836,094,391;
  Windows arm64 859,089,903), and in **none** does the leaky cumulative
  `floor(free_weight · F(x))` change (`benchmarks/exp_audit_2026-10-17.log`; the
  `exp audit` workflow reruns it). Streams encoded on any of these platforms decode
  identically here.
- The Laplace scale table cannot be recomputed bit-exactly (23 of 2561 entries differ from
  `f32(exp(f64))`), so it is embedded from upstream's `mu_scale.npy`
  (`scripts/gen_scale_table.py`).

## Encoder

Not started. Plan: hand-written forward + backward passes on archmage (CPU first), upstream
PyTorch as the per-op gradient oracle; SOAP optimizer, softround + Kumaraswamy noise, STE,
RDOQ and the network quantization search are new code. Reuse candidates surveyed 2026-10-17:
zensim `adam_simd.rs`, `zensim-train-core` dense SIMD kernels, `qat_quantize_copy`,
`zensim-train-gpu` CubeCL kernel templates (all MIT/Apache).

## Reference environment

```bash
git clone https://github.com/Orange-OpenSource/Cool-Chic.git ~/tmp/coolchic-src/Cool-Chic
cd ~/tmp/coolchic-src/Cool-Chic
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p ~/tmp/coolchic-ref && cp samples/bitstreams/kodim14.cool ~/tmp/coolchic-ref/
cd ~/work/zen/zencoolchic && just oracle-dump && just oracle
```

Upstream's decoder builds fresh `nn.Module`s, which are in training mode, so it runs the
non-separable `kron(w, wᵀ)` upsampling kernels. The port follows that path.
