# Porting status

Upstream: [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) `a6fe38a` (2026-07-17,
v5.0.1), BSD-3-Clause (notice in `upstream-notices/`). Entropy coder: `constriction` 0.4.2
(the Python package's Rust source) with `probability` 0.20.3's Laplace CDF.

[Remaining work and implementation entry points](docs/REMAINING-WORK.md) distinguishes
missing codec features, validation gaps, performance work and integration limits.

## Decoder

| Module | Upstream | Status | Parity evidence |
|---|---|---|---|
| Video / frame / Cool-chic headers | `bitstream/header/*.py` | done | 12 oracle streams + 3-frame `multi_i`; `tests/oracle.rs` |
| Network parameters (exp-Golomb, per-module q-step) | `bitstream/neuralnet/*.py` | done | `expgolomb::tests::round_trip_all_orders`; kodim14 |
| Fixed-point ARM / IFCE | `bitstream/component/armint.py` | done | latents bit-exact on 12 streams, incl. no-IFCE and no-stabiliser |
| Wavefront latent decode | `bitstream/component/latent.py` | done (wavefront and width ≤ 9 raster order) | latents bit-exact on 12 streams |
| Range decoder + leaky quantized Laplace | constriction 0.4.2 `queue.rs`, `quantize.rs` | done | latents bit-exact on 12 streams; exhaustive `exp` audit below |
| Upsampling (train-mode kron kernels) | `component/core/upsampling.py` | done; SIMD AXPY | raw output max abs error ≤ 9.3e-7 (9.24e-7) on 12 streams |
| Synthesis | `component/core/synthesis.py` | done; SIMD AXPY | as above |
| Final resize (nearest / bilinear / bicubic) | `F.interpolate` | bicubic: `mid_lat1_debug`; nearest: `pb_128` motion payloads; bilinear: `evenk64_bilinear_debug` (header-retagged upstream stream) — all bit-identical to upstream decode | `mid_lat1_debug`, `pb_128`, `evenk64_bilinear_debug` |
| Common randomness | `component/core/noise.py` | done | `mid_wasserstein_debug` (`--tune wasserstein`): all samples identical |
| YUV 4:2:0 / 4:4:4 output | `io/format/yuv.py` | done | `yuv420_8b_debug`, `yuv444_10b_debug` (10-bit) |
| Multi-frame all-intra decode | `bitstream/decode.py::decode_video`, `utils/codingstructure.py` | done: `decode_video` returns frames in display order; coding structure rebuilt and cross-checked against each frame header | `multi_i` oracle case: three upstream-decoded frames (mixed RGB / 4:2:0, mixed sizes) bit-exact |
| Inter frames (P/B), warper, global motion | `bitstream/decode.py::decode_frame`, `component/intercoding/` | done: integer global translation, flow quantized to 1/64 px, bilinear/bicubic/sinc warper (filter sizes 2/4/8/12), alpha/beta compositing, 4:2:0→4:4:4 reference upsampling | `pb_128` oracle case (upstream-encoded I→P→B→B): I/P frames bit-exact, B frames ≤1 quantization step on ~1% of samples; raw residue/motion ≤ 1e-6 |

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

Decode threading: `code_grid` evaluates the per-diagonal ARM contexts on up to 8
scoped worker threads while the range coder stays serial on the calling thread
(output is identical — evaluations are pure functions of earlier diagonals).
The gate needs ~32 px per diagonal, so it activates around 320²+ image sizes:
measured best-of-5 on this host, `examples/decode`, lop encodes — 512² **124 ms**
vs 192 ms serial; 384² **63 ms** vs 99 ms; 256² stays serial (26 px/diagonal loses
~1 ms to barrier overhead). The zencodec `Sequential`/`SingleThread` policy and
no-`std` builds keep the serial traversal. The IFCE context pass is likewise
fanned out above 16k stacked pixels (~3 ms of a 512² decode); the sequential
policy covers both directions — encode installs it alongside the single-thread
rayon pool so a one-thread encode stays one thread through latent coding.
The inference warper fans out over row bands above 32k pixels (512² ×3ch,
8-tap filter: **27 ms** vs 154 ms serial) — the dominant inter-frame cost
after the latents — same policy, same bit-identity argument. End to end on a
256² two-frame I/P stream (wavefront idle at that size): **75.8 ms** vs
100.4 ms serial.

Declined after measurement: a four-accumulator `dot` unroll lost to the
simple fold on the ARM's short vectors (6–72 elements); i64 SIMD is out
because no `i64×i64` vector multiply exists below AVX-512DQ and magetypes
has no i64 multiply backend; a Laplace-CDF tail early-out that skips `exp`
only fires for |arg| ≳ 16 (too rare); CDF precomputation is pointless
because the decoder search is already logarithmic; upsampling-module RDOQ
incrementalization covers 48 of 1869 candidate parameters (2.6%, ~1%
ceiling); the remaining memory-bound decode loops (`translate`, B-frame
beta blend, latent→f32, `stack_nearest`) are ~1 ms each at 512² against
~0.3 ms of spawn cost; and the 17% of encode time in `expm1f`/`expf`
cannot take an approximate SIMD exp because the Laplace rate feeds the
loss → training trajectory → serialized bytes. The range decoder itself
stays serial by construction: its arithmetic state is the serialized
stream.

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
| Training forward: quantizer (softround ∘ noise ∘ softround, Kumaraswamy), float ARM + IFCE, Laplace rate, train-mode upsampling, synthesis, common-randomness synthesis inputs, weighted MSE + λ·rate, 4:2:0 dict downsample-then-clamp scoring, `rescale_output` for latent ranges starting below full resolution | `component/core/*.py`, `training/loss.py`, `io/format/yuv.py`, `F.interpolate` | done; SIMD/rayon | loss within 3e-8 of torch (`just grad`) |
| Wasserstein distortion (`--tune wasserstein`: VGG16 feature taps, multiscale mean/std statistics, `dist = 0.2·mse + 0.004·wass`) | `training/metrics/wasserstein.py` | done; `src/train/wasserstein.rs`, real VGG16 flat weights (`dump_vgg.py`) or arbitrary (c_in, c_out) stacks | `small64_miniwass` below; note upstream's `safe_clamp_min` returns a detached tensor, so the `(σ_a − σ_b)²` term adds to the distance but carries no gradient — only the mean terms backpropagate |
| Backward of all of the above | torch autograd | done; SIMD/rayon | vs float64 torch: worst relative gradient error 3.7e-4 (mid130×97 hop), 7.3e-4 (64×64 hop), 5.8e-4 (64×64 small), 2.8e-4 (64×64 miniwass, common randomness + Wasserstein; d wasserstein/d decoded rel 1.8e-7), 6.9e-4 (64×64 yuv420, 4:2:0 chroma-average loss), 1.0e-4 (64×64 lop, latents (2,4) + bicubic `rescale_output`); torch's own float32 pass is off by up to 1.4e-2 on the same tensors |
| SOAP optimizer (Adam mode for latents and 1-D tensors) | `training/soap.py` | done, scalar | vs upstream after 25 steps on fixed gradients (`just soap`): 20×20 weight 2.1e-4 of the parameter movement, 1-D 6.6e-7, latents 1.3e-6. Rank-deficient factors (non-square weights) are not comparable: upstream against itself with a rotated null-space basis differs by 1.6% (`scripts/oracle/soap_null_space_probe.py`); this port differs by 2.2% |
| Training loop: grad clip 0.1, cosine LR, softround / noise schedules, patience, best-model restore; initializers; intra preset; 5 → 2 → 1 warm-up; lambda-dependent encoder gain | `training/train.py`, `presets.py`, `warmup.py`, `initialize_parameters` | done; parallel deterministic noise; RNG is not torch's | small64 (64×64), 2000 itr, λ 1e-3, hop, one seed each: upstream 33.808 dB / 0.5983 bpp latent / loss 1.0143e-3 in 20.7 s (torch CPU); this port 34.094 dB / 0.5949 bpp / 0.984e-3 in 25.8 s (1 thread, scalar). Not yet compared over several seeds or images |
| Network quantization search (per module, every weight/bias step; exp-Golomb orders from upstream's bit estimate) | `nnquant/quantizemodel.py` | done; independent candidate evaluations on the problem-sized rayon pool, first-minimum scan in stream order — byte-identical streams | small64 end to end (below) |
| RDOQ (greedy ±1..15-step moves with upstream's local scores) | `nnquant/rdoq.py` | done; per-module invariant caches plus channel-level incremental synthesis scoring (`SynInc`: one conv channel + downstream suffix per candidate, cached prefix/stabiliser outputs; bit-identical to the full pass, verified per-parameter and end-to-end) | small64 below |
| Trained model → stream integers → bitstream | `bitstream/encode.py` | done | upstream's decoder reads this encoder's stream: small64 decodes to 33.828 dB in both decoders |
| SIMD / threaded training kernels | — | implemented: AXPY/dot, batched ARM/IFCE, convolution forward/backward and work splitting; further tuning remains | `src/simd.rs`, `src/train/{ops,arm_batch,ifce}.rs`; step and tiny-stream identity with 1/4 threads |
| Inter training primitives: training-mode warp flow gradient, global-translation search, `flow` frame data type | `component/intercoding/{warp,globalmotion}.py`, `component/frame.py` (`flow` skips the output clamp), `video.py::guided_motion_pretraining` | done; `src/warp.rs`, `src/train/{eval,ops}.rs` | `vec` warp case: `d(warp)/d(flow)` ≤ 1.3e-4 vs autograd at filter sizes 2/4/8/12; `get_global_translation` bit-equal on both dumped pairs (coarse-step quirks reproduced); `small64_flow` gradient case worst 8.4e-4 vs float64 |
| Joint residue+motion P/B training: alpha/beta reparam, composite, synchronous two-model step | `component/frame.py` (inter forward), `component/video.py::encode_one_frame` (one optimizer over both ccs) | done; `src/train/inter.rs`, `train_pair` in `src/train/trainer.rs` | frozen `comp`/`compp` cases: composite ≤2.4e-7, raw-output gradients ≤2.3e-7 vs autograd; `pair_step`/`pair_eval` reproduce upstream's joint forward |
| Inter preset, warm-up candidates, motion pretrain, inter/motion headers | `training/presets.py::PresetInter`, `warmup.py`, `cfg/dec/{residue,motion}/*.cfg` | done; `Preset::inter`, `encode_training_pair`, `config::{header_inter,header_motion}` | unit-tested: odd candidates inject pretrained motion with gain-scaled latents; YUV `img_min_max` output transform init |
| Video encode driver | `component/video.py::encode_one_frame`, `bitstream/encode.py` | done; `encode_video` in `src/unstable.rs` | generated 64×64 I/P/B stream decoded by upstream's own `decode_video`, reproducing the reported PSNRs exactly (I 41.72 / B 43.22 / P 40.80 dB); in-crate `video_ibp_encodes_and_decodes` covers coding order, header fields and multi-frame decode |

End to end on small64 (64×64, 2000 itr, λ 1e-3, hop; `examples/encode.rs`, 2026-10-17, one
seed per side):

| encoder | bytes | bpp | PSNR | D + λR (upstream's loss) | time |
|---|---|---|---|---|---|
| upstream (torch CPU) | 1,767 | 3.451 | 33.322 dB | 0.003916 | — |
| this port, no RDOQ | 1,909 | 3.729 | 33.828 dB | 0.004143 | 27.5 s |
| this port, RDOQ | 1,858 | 3.629 | 33.979 dB | 0.004029 | 33.6 s |

Full size, CID22-512 `validation/1279330.png` (512×512), upstream defaults (λ 1e-3, 10,000
itr, hop), mac M4 Pro, one seed each (`benchmarks/encode_cid22_1279330_2026-10-17.tsv`):

| encoder | bytes | bpp | PSNR | D + λR | wall time |
|---|---|---|---|---|---|
| upstream (torch 2.14.1 CPU) | 10,769 | 0.3286 | 38.157 dB | 0.0004815 | 1,065 s |
| this port (`39c0f16`) | 10,692 | 0.3263 | 37.988 dB | 0.0004851 | 507 s |

Both decoders (this crate's and upstream's `cc_decode.py`) decode this port's streams to the
same PSNR. One seed on a 64×64 picture, where network parameters are ~80% of the stream, does
not separate the encoders; a multi-seed, multi-image comparison needs the faster trainer.

The gradient check compares against a float64 run of upstream (`dump_grad.py ... f64`): the
rate term `F(q + ½) − F(q − ½)` cancels badly in float32, so torch's float32 gradients are not
a usable reference at 1e-3. The `miniwass` dump additionally stores the clamped decoded
image, its VGG tap features, per-tap distances and tap/decoded-image gradients, so the
test can assert each Wasserstein stage separately; its miniature extractor weights are
embedded in the dump (`vgg`, `vgg_dims`).

## Reference environment

Ordinary regression tests use the [frozen references](tests/fixtures/upstream/README.md)
in this repository: `just oracle-all` needs Rust, not Python or external data.
For regenerating or extending the oracle, [oracle maintenance](scripts/oracle/README.md)
records the exact upstream revision, environment, saved inputs and commands. The
Python checkout/venv can be rebuilt; the original golden arrays are preserved in git.

Upstream's decoder builds fresh `nn.Module`s, which are in training mode, so it runs the
non-separable `kron(w, wᵀ)` upsampling kernels. The port follows that path.

## zencodec bridge

The optional `zencodec` feature adds header-only probing, RGB8/RGB16 decode and custom
`coolchic` format registration. Add `encoder` for public trainer-backed encode configurations.
The registered eight-byte still header is `00 10 01 00 00 00 80 00`: the 16-bit header-length
field starts after three 12-bit fields, not at a byte boundary. This was checked against
upstream's kodim14 stream and the writer's output. Detection is heuristic, not magic.

RGB8 sRGB is the only encode descriptor. Nonidentity metadata orientation is baked through
`zenpixels-convert`; YUV, HDR, other descriptors and video are refused. Finite working-memory
budgets are refused because this codec has no complete heap accounting; it does not advertise
memory-limit enforcement. Input/output bytes, dimensions, pixels and frames are checked.
Cancellation reaches each training iteration, network quantization/RDOQ candidate and decode
grid/stage. A whole convolution or latent grid is not interruptible. Quality mapping and
resource estimates are uncalibrated/unknown, respectively.

Tests cover native-versus-trait decode, nonzero 8/10/16-bit output, strided encode/sink buffers,
minimum-effort training and cancellation, output caps, orientation versus pre-oriented input,
and complete tiny-stream identity with 1 and 4 threads. Published testkit 0.1.0's capability,
metadata-leakage, error-envelope and truncation checks apply. Its pixel/orientation round-trip
checks require original-pixel identity and do not apply to lossy encoding; the orientation
replacement retains exact comparison of the two encoded streams. Existing upstream parity
bounds are unchanged.

Independent check (2026-10-17, mac): `examples/codec_encode.rs` at minimum generic effort
(2000 iterations, default hop/lambda/seed/RDOQ) encoded a 4×3 RGB8 pattern (`sample[i] = 7*i`,
36 samples). Upstream `a6fe38a` `cc_decode.py` and this crate's `examples/decode.rs` produced
identical 36/36 RGB8 samples. That particular generated stream is not a checked-in
fixture; the upstream decoder/writer oracle streams are preserved in compressed form.
