# Remaining Cool-chic 5 port work

Target: upstream [Cool-chic v5.0.1 at a6fe38a414dd098b39c41636bd6e423626402f7e](https://github.com/Orange-OpenSource/Cool-Chic/tree/a6fe38a414dd098b39c41636bd6e423626402f7e).
Audited against Rust and pinned upstream source on 2026-10-18.

**Status:** the video encode driver is landed — [encode_video](../src/unstable.rs)
walks `coding_structure()` in coding order, trains I frames on the intra
preset and P/B frames as synchronous residue+motion pairs on the inter preset,
computes per-reference global translations, pretrains motion on an optional
external flow field, quantizes residue then motion (motion's partner raw comes
from the decoded residue payload, matching the decoder-visible state), writes
each frame header + payload(s), and decodes immediately to feed the reference
store. Upstream's own decoder reproduces the reported PSNRs on a generated
I/P/B stream. Remaining gaps: untested decoder variants, incomplete
resource/streaming integrations, and validation of the less-common video
configuration combinations (4:2:0 inter encode, custom warp filters, RDOQ on
inter streams, external-flow end-to-end runs). The current implementation is a
still-image and video decoder (intra + P/B inter) plus a planar RGB/YUV
8–16-bit still-image and video encoder, not a complete port of every upstream
workflow.

## Start here

1. [PORTING.md](../PORTING.md) records implemented pieces, exact reference
   identity, measured tolerances and historical performance observations.
2. [Encoder guide](ENCODER.md) describes the training/writer pipeline, public
   zencodec configuration, defaults and deliberate API refusals.
3. Run `just check`. It includes frozen upstream comparisons and zencodec
   conformance. `just oracle-all` isolates upstream parity. The checked-in
   [references](../tests/fixtures/upstream/README.md) need Rust alone at runtime.
4. [Oracle maintenance](../scripts/oracle/README.md) has exact upstream setup,
   environment records, source inputs and dump/regeneration commands.

The following paths under `coolchic/` refer to the pinned upstream tree above.
Their Rust counterparts are linked directly.

## Missing codec and training behavior

| Work | Current boundary and implementation entry points | Upstream reference and acceptance gate |
|---|---|---|
| Multiple I frames | done: [decode_video](../src/decode.rs) returns every frame in display order (`decode` stays the still-image API). [coding_structure](../src/header.rs) rebuilds coding order and references from the video header and cross-checks each frame header against it; malformed structures that upstream asserts on or silently misbuilds are rejected. Frozen `multi_i` oracle case decodes three upstream-generated frames (mixed RGB/YUV420, mixed sizes) bit-exactly. | `bitstream/decode.py::decode_video`, `utils/codingstructure.py`. |
| P/B frame decoding | done: [decode_inter](../src/decode.rs) decodes the residue and motion Cool-chic payloads separately, applies integer global translation, warps 4:2:0 references (converted to 4:4:4 by nearest 2× chroma replication) through [warp.rs](../src/warp.rs), blends P/B predictions with the residue alpha/beta channels, adds the RGB residue and finishes through the shared quantization tail. The warper quantizes flow to 1/64-pixel steps and implements upstream's bilinear, bicubic and sinc separable kernels (filter sizes 2/4/8/12). Frozen `pb_128` oracle case: upstream-encoded 4-frame 4:2:0 stream (I→P→B→B) with large global translations. I and P frames decode bit-exactly; B frames differ on ~1% of samples by at most one quantization step (float accumulation order). Per-frame residue/motion raw outputs match to 1e-6. | `bitstream/decode.py::decode_frame`, `component/intercoding/globalmotion.py`, `component/intercoding/warp.py`. |
| ~~Motion/video encoding~~ | done: [encode_video](../src/unstable.rs) drives the full pipeline through [trainer](../src/train/trainer.rs), [config](../src/train/config.rs) and [write](../src/write.rs). The training-side inter primitives are in [warp.rs](../src/warp.rs) and oracle-verified: [warp_flow_backward](../src/warp.rs) is the training-mode `d(warp)/d(flow)` (unquantized flow, ≤1.3e-4 vs autograd at every filter size) and [global_translation](../src/warp.rs) is the coarse-to-fine integer search (bit-equal to `get_global_translation` on the frozen pairs, coarse-step quirks included). The `flow` frame data type in [eval::decoded_dict](../src/train/eval.rs)/[ops](../src/train/ops.rs) reproduces the unclamped flat-MSE objective behind `guided_motion_pretraining` (`small64_flow`, worst grad 8.4e-4 vs float64). The joint residue+motion objective is in [train::inter](../src/train/inter.rs): alpha/beta reparam, per-reference global translation inside the composite, beta blending, and the gradient scatter through `warp_flow_backward` — oracle-verified on frozen P (`compp`) and B (`comp`) cases (composite ≤2.4e-7, raw-output gradients ≤2.3e-7). [pair_step/pair_eval](../src/train/inter.rs) reproduce upstream's synchronous joint forward, and [train_pair](../src/train/trainer.rs) runs the phase loop over both models with a joint clip/SOAP state. [Preset::inter](../src/train/trainer.rs) is the exact `PresetInter` port, [encode_training_pair](../src/train/trainer.rs) runs the upstream candidate loop (fresh models per candidate, pretrained motion injected on odd candidates, warm-up selection by joint loss), [config::header_inter](../src/train/config.rs)/[header_motion](../src/train/config.rs) reproduce the upstream `cfg/dec/residue` and `cfg/dec/motion` tables (motion: latent 2-6, no hyperlatent, nearest upsampling), and [encode_video](../src/unstable.rs) is the frame orchestrator: coding-order traversal, per-reference global translations on 4:4:4-converted planes, optional external-flow pretraining (`FlowEstimator`), joint pair training, residue-then-motion quantize/RDOQ with the motion partner raw taken from the decoded residue payload, frame-header + payload serialization via [write_frame](../src/write.rs), immediate decode into the reference store, and stored-domain PSNR reporting. A generated 64x64 I/P/B stream decodes under upstream's own Python decoder reproducing the reported PSNRs exactly. Residual: 4:2:0 inter encode, custom warp filters and external-flow runs are exercised only through unit paths, not end to end. | `component/video.py::encode_one_frame` / `guided_motion_pretraining`, `component/frame.py`, `training/presets.py`, `bitstream/encode.py`, `component/intercoding/globalmotion.py`, `component/intercoding/warp.py`. Upstream's `component/intercoding/raft.py` uses pretrained torchvision RAFT for encoder initialization — the only non-portable input; the API will take an external flow field so a real RAFT export or any estimator can drive pretraining. `get_global_translation` needs no RAFT (integer MSE search). |
| ~~YUV and higher-depth encode input~~ | done: [encode_frame](../src/unstable.rs) trains planar u16 frames — RGB and 4:4:4 (three full-resolution planes) or 4:2:0 (full luma, half chroma, even dims) at 8–16 bit — reproducing upstream's downsample-then-clamp target layout, per-plane-sample MSE weighting, luma-normalized rate and quantized-evaluation tail through [StepCfg/Problem](../src/train/trainer.rs), [eval::decoded_dict](../src/train/eval.rs) and [RDOQ](../src/train/rdoq.rs). `small64_yuv420` differential gradient case (worst 6.9e-4 vs float64 torch); end-to-end 4:2:0/4:4:4-10b encode–decode round trips pass. `Tune::Wasserstein` on 4:2:0 is rejected, as upstream's VGG path requires a full-resolution three-channel tensor. Residual: the [zencodec trait](../src/codec/encode.rs) still accepts RGB8 only; higher-depth RGB and further YUV geometry/depth combinations would benefit from more differential fixtures. | `training/loss.py`, `io/format/yuv.py`, `io/framedata.py`, `component/frame.py`. |
| ~~Starting latents below full resolution~~ | done: [TrainOptions::latent_res](../src/unstable.rs) accepts `(lo, hi)` (upstream `--latent_residue lo-hi`). Synthesis runs at the finest latent grid (`ceil(img/2^lo)`); [eval::rescale_output](../src/train/eval.rs) interpolates the output to image size with the header's `final_upsampling` mode and [nn::resize_backward](../src/nn.rs) carries the adjoint (nearest/bilinear/bicubic transposes). `small64_lop` differential gradient case at `(2,4)` (worst 1.0e-4 vs float64 torch); end-to-end encode→decode round trip passes. Common randomness is rejected for `lo>0`: upstream's `get_synthesis_input` concatenates full-resolution noise planes with the smaller synthesis input, so that combination cannot run there either. | `component/core/coolchic.py::rescale_output`, `F.interpolate`. |
| Preset/configuration breadth | Public encode exposes four fixed intra architectures. Internal header/shape code handles more configurations than those presets expose; four gradient cases do not establish all of them. | `cfg/dec/`, `utils/parsecli.py`, `training/presets.py`. Validate alternative IFCE/hyperlatent/architecture choices individually before exposing new knobs. Motion/debug/custom training schedules are not a general Rust preset interface. |

For video, [dump_video.py](../scripts/oracle/dump_video.py) assembles
multi-frame streams from single-frame payloads with upstream's own header
classes (`--assemble`) or decodes an upstream-encoded inter stream
(`--decode`), capturing per-frame planes plus the raw residue/motion
synthesis outputs. Add new frozen cases while
retaining the original still-image set. Do not infer video support from parsed
headers.

## Coverage gaps and known limits

- The frozen suite covers 12 still-image decoder/writer cases, a three-frame
  all-intra video case and a four-frame upstream-encoded inter stream, four
  float64 gradient cases and fixed-gradient SOAP
  behavior. It is not exhaustive format conformance. It includes odd RGB
  dimensions, narrow raster-order decoding, no-IFCE/no-hyperlatent, no
  stabilisers, common randomness, half-resolution bicubic resize, YUV420 8-bit,
  YUV444 10-bit, and sinc-filter warping with global translation and B-frame
  bidirectional references.
- ~~nearest/bilinear resize + even synthesis kernels~~ done: [nn::resize](../src/nn.rs)
  bilinear matches `F.interpolate(mode="bilinear", align_corners=False)`;
  nearest is exercised by every motion payload in the `pb_128` oracle case.
  [Conv::forward](../src/nn.rs) implements upstream's `pad=(k-1)//2` shrink —
  each even kernel removes one pixel per spatial dimension, and even kernels are
  refused on residual layers or with the 1×1 stabiliser (upstream broadcast-fails
  there too). `evenk64_debug` (upstream-encoded, two k=2 layers) and
  `evenk64_bilinear_debug` (same stream with `final_upsampling` re-tagged to
  bilinear — a variant upstream's encoder never emits but its decoder accepts)
  both decode bit-identically to upstream v5.0.1; pinned as fuzz seeds plus
  [stream_variants](../src/decode.rs) checksum tests. The trait tests' synthetic
  16-bit path is still not an upstream 16-bit golden reference.
- SOAP's [rank-deficient eigenspaces](../src/train/soap.rs) have arbitrary bases.
  Full-rank/Adam cases retain their asserted trajectory bound; non-square
  trajectories are diagnostic. Read the tracked [null-space probe](../scripts/oracle/soap_null_space_probe.py)
  before changing the optimizer or proposing a new comparison contract.
- RNG initialization differs from torch. Complete training trajectories are
  not torch-identical. There are step-level and complete tiny-stream thread-count
  identity tests, but no measured cross-platform full-training identity guarantee.
- No multi-image/multi-seed rate-distortion calibration exists. The generic quality
  mapping is explicitly uncalibrated. Historical one-image/one-seed benchmarks do
  not establish a general quality or speed ranking.
- Two libFuzzer targets cover the public decode entry points
  ([fuzz/fuzz_targets](../fuzz/fuzz_targets/)), with real-stream seeds under
  [fuzz/seeds](../fuzz/seeds/), a minimized crash corpus under
  [fuzz/regression](../fuzz/regression/), and stable replay in
  [tests/fuzz_regression.rs](../tests/fuzz_regression.rs). The live corpus is on
  block storage at `/mnt/v/fuzzes/zencoolchic/` (see the workspace fuzz rules).
  The first session found and fixed a `bits(n>32)` panic in exp-Golomb decode;
  deeper runs remain worthwhile.

## Performance and integration work

SIMD and threaded training are **already implemented**. [simd.rs](../src/simd.rs)
provides AXPY/dot; [ops.rs](../src/train/ops.rs), [arm_batch.rs](../src/train/arm_batch.rs)
and [ifce.rs](../src/train/ifce.rs) use those kernels and rayon. Further optimization
must start from the current implementation and preserve arithmetic order.

RDOQ [score](../src/train/rdoq.rs) caches per-module invariant latents,
IFCE contexts and synthesis inputs. Synthesis-module candidates additionally
score through [SynInc](../src/train/eval.rs): a move touches one output channel
of one tensor, so only that channel and the downstream suffix are recomputed
while accepted prefix-layer outputs and the stabiliser are cached. The result is
bit-identical to the full forward — proven by
`eval::tests::syninc_matches_full_synthesize_per_parameter` (every parameter,
both biases and weights, across accepted-move invalidations) and by identical
scores, selected moves and stream bytes end to end. The other modules still
score full-stage per candidate, matching upstream, which does the same; an
upsampling candidate could reuse a similar cache if profiles justify it.
Quantization evaluates its independent (weight, bias) shift candidates on the
problem-sized rayon pool with per-worker scratch and selects the first minimum
in stream order — identical outputs, roughly 9% faster on the 128px RDOQ
encode (this host, Zen 5).

The [shared-NN assessment](zennn-assessment.md) identifies owned kernels to reuse,
their source revisions, consumers and bit-order/performance risks. No `zennn`
crate was created. Read it before replacing kernels or creating a new framework.

The trait bridge now decodes video through `AnimationFrameDecoder` (lazy coding-order
decode via `decode::VideoDecoder`, display-order RGB frames, references retained),
serves pull streaming as one-shot decode plus ≤64-row batches, and honors a sequential
encode threading policy with a scoped single-thread pool (identical stream bytes).
Decode-side `max_memory_bytes` is honored via a pre-flight working-set estimate
(`decode::estimate_decode_bytes`: network parameters, latent grids and ARM/IFCE
contexts, upsample/synthesis conv buffers, stored reference frames, output canvas —
padded above the true peak, so passing decodes fit while rejected ones may only
have been near their budget).
Animation encode is served: `push_frame` buffers RGB8 planes and `finish` trains
the whole video with upstream's `intra_pos 0`, `p_pos <last>` GOP — Cool-chic is
a whole-video codec, so there is no incremental encoding, and durations are
dropped (the stream has no timing field).
Encode-side `max_memory_bytes` is likewise honored via `decode::estimate_encode_bytes`
(same payload terms plus gradients, SOAP preconditioner pairs and eigensolver
transients, f32 latents and backward activations, retained references), checked
once frame dimensions are known — at `encode()` for stills and `finish()` for
animations. Still refused: YUV output (`PixelDescriptor` has no YCbCr layout or
subsampling carrier). RGB streams have no
color metadata carrier; the bridge assumes sRGB. Fallible allocation coverage and
finer cancellation granularity remain engineering work. Current cancellation checks
are per grid/stage or training/quantization/RDOQ iteration; an entire convolution is
not interruptible.

Training uses a CPU backend. Upstream GPU/torch compilation machinery, general
training checkpoint/resume and PyTorch `.pt` checkpoint import/export are not
ported interfaces. These are workflow/backend work in addition to the missing
bitstream features above; they are not dependencies of the current still decoder.

## Validation when extending the port

Preserve exact integer latents and writer bytes. The existing float reconstruction
bound is 9.3e-7 and final samples differ by at most one step; both are enforced.
Gradient loss/relative-L2 and full-rank SOAP thresholds are recorded in their tests.
Do not change a bound to make a new path pass. Extend upstream capture first and
compare the smallest new behavior independently before wiring it into the codec.

Run `just check`, default/no_std checks and the relevant upstream comparisons.
CI runs frozen comparisons on Linux x64/ARM64, Windows x64/ARM64, macOS ARM64 and
i686 through cross; it also checks wasm no_std and the declared Rust MSRV. The
separate [exp audit workflow](../.github/workflows/exp-audit.yml) is relevant when
changing Laplace CDF/libm/scale-table behavior; its historical measurements are
in [PORTING.md](../PORTING.md#bit-exactness-risks).

The checked-in code, fixtures, source inputs and maintenance scripts are sufficient
for current regression testing and for starting the next port slice. The pinned
upstream checkout/Python environment can be rebuilt for new comparisons. Benchmark
datasets and any future pretrained motion model are separate, explicitly sourced
inputs; neither is required for ordinary regression or conformance tests.
