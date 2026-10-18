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

### Speed (2026-10-17, mac M4 Pro, single thread)

kodim14 (768×512): **178 ms** (`cargo run --release --example decode --features std --
kodim14.cool --repeat 5`, best of 5; commit 9ee5fa7). Upstream's decoder reports
`Decoding frame 0 time = 0.46 seconds` for the same stream on the same machine (torch 2.14.1
CPU, its own timer, model construction included).

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

Reuse plan (user, 2026-10-17: reuse owned code, relicensed): zensim-train-core / zensim-validate
(dense SIMD forward/backward, SIMD Adam, QAT), zenpicker-train (Adam MLP trainer), zenjpegai
`nn::fast` (conv forward engine), zentrain (per-image preset picker).

| Piece | Upstream | Status | Evidence |
|---|---|---|---|
| Bitstream writer: range encoder, exp-Golomb encoder, headers | `bitstream/encode.py`, constriction encoder | done (`encoder` feature) | all 12 oracle streams re-encode byte for byte (`write::tests`) |
| Training forward: quantizer (softround ∘ noise ∘ softround, Kumaraswamy), float ARM + IFCE, Laplace rate, train-mode upsampling, synthesis, MSE + λ·rate | `component/core/*.py`, `training/loss.py` | done, scalar, full-resolution latents only | loss within 3e-8 of torch (`just grad`) |
| Backward of all of the above | torch autograd | done, scalar | vs float64 torch: worst relative gradient error 3.7e-4 (mid130×97 hop), 7.3e-4 (64×64 hop), 5.8e-4 (64×64 small); torch's own float32 pass is off by up to 1.4e-2 on the same tensors |
| SOAP optimizer (Adam mode for latents and 1-D tensors) | `training/soap.py` | done, scalar | vs upstream after 25 steps on fixed gradients (`just soap`): 20×20 weight 2.1e-4 of the parameter movement, 1-D 6.6e-7, latents 1.3e-6. Rank-deficient factors (non-square weights) are not comparable: upstream against itself with a rotated null-space basis differs by 1.6% (`scripts/oracle/soap_null_space_probe.py`); this port differs by 2.2% |
| Training loop: grad clip 0.1, cosine LR, softround / noise schedules, patience, best-model restore; initializers; intra preset; 5 → 2 → 1 warm-up; lambda-dependent encoder gain | `training/train.py`, `presets.py`, `warmup.py`, `initialize_parameters` | done, scalar; RNG is not torch's | small64 (64×64), 2000 itr, λ 1e-3, hop, one seed each: upstream 33.808 dB / 0.5983 bpp latent / loss 1.0143e-3 in 20.7 s (torch CPU); this port 34.094 dB / 0.5949 bpp / 0.984e-3 in 25.8 s (1 thread, scalar). Not yet compared over several seeds or images |
| Network quantization search (per module, every weight/bias step; exp-Golomb orders from upstream's bit estimate) | `nnquant/quantizemodel.py` | done, scalar | small64 end to end (below) |
| RDOQ (greedy ±1..15-step moves with upstream's local scores) | `nnquant/rdoq.py` | done, scalar; every score is a full pass (no incremental update yet) | small64 below |
| Trained model → stream integers → bitstream | `bitstream/encode.py` | done | upstream's decoder reads this encoder's stream: small64 decodes to 33.828 dB in both decoders |
| SIMD / threaded training kernels | — | not started | — |

End to end on small64 (64×64, 2000 itr, λ 1e-3, hop; `examples/encode.rs`, 2026-10-17, one
seed per side):

| encoder | bytes | bpp | PSNR | D + λR (upstream's loss) | time |
|---|---|---|---|---|---|
| upstream (torch CPU) | 1,767 | 3.451 | 33.322 dB | 0.003916 | — |
| this port, no RDOQ | 1,909 | 3.729 | 33.828 dB | 0.004143 | 27.5 s |
| this port, RDOQ | 1,858 | 3.629 | 33.979 dB | 0.004029 | 33.6 s |

Both decoders (this crate's and upstream's `cc_decode.py`) decode this port's streams to the
same PSNR. One seed on a 64×64 picture, where network parameters are ~80% of the stream, does
not separate the encoders; a multi-seed, multi-image comparison needs the faster trainer.

The gradient check compares against a float64 run of upstream (`dump_grad.py ... f64`): the
rate term `F(q + ½) − F(q − ½)` cancels badly in float32, so torch's float32 gradients are not
a usable reference at 1e-3.

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
