# Should zennn be built now?

**Recommendation: build a small CPU primitives crate now, with two consumers migrated in the
same effort. Defer a general training framework and GPU backend.** There is enough concrete
convolution and dense-kernel duplication to justify an extraction. A crate containing copied
code with no migrated consumers would add another implementation to maintain. This assessment
does not create the crate or change any surveyed consumer.

The first gate should be two real consumer adapters, preserving each consumer's arithmetic
contract, plus end-to-end parity and measured performance at tiny and representative sizes.
If that gate cannot be funded now, continue direct reuse from the owned sources and postpone
creating zennn. Licensing is not a reason to rewrite owned code; preserve third-party notices
and cite each extraction's source commit.

## Survey and scope

Source inspected on 2026-10-17 (local date). Counts are physical lines from `*.rs` under the
named source directories, including inline tests, comments and source-directory binaries.
They are inventory counts, not interchangeable-code counts or performance measurements.
Sibling source files matched their commits; unrelated local lockfile changes were left alone.
No GPU execution or migration benchmark was performed for this assessment.

| Repository snapshot | Source scope | Files / lines | What is actually there |
|---|---|---|---|
| [zenjpegai a04b37a](https://github.com/imazen/zenjpegai/tree/a04b37a9cac04de1ee109ba02288ada5c06fcb5a/src/nn) | `src/nn/` | 11 / 3291 | Reference layers, packed float and integer convolution, transposed/depthwise convolution, blocked tensors, recycling pool, activation/gating/norm/pixel shuffle, runtime SIMD engine. `src/tensor.rs` and `src/mem.rs` are additional infrastructure outside this count. No training backward engine in this directory. |
| [zensr 12e1054](https://github.com/imazen/zensr/tree/12e1054d09a5ac3a8d7d941c7efe8d88d7ce772a/crates/zensr-micro/src) | `crates/zensr-micro/src/` | 11 / 5116 | SR-specific packed 3×3/1×1 convolutions, SIMD activation/gates, pixel shuffle, tiled execution, reusable `Scratch`, direct and Winograd paths; coefficient consistency and image guards. `simd.rs` alone is 1488 lines. |
| [zencoolchic 5b48b50](https://github.com/imazen/zencoolchic/tree/5b48b509d31b/src) | `src/nn.rs`, `src/simd.rs` | 2 / 632 | Planar float inference, symmetric upsampling, synthesis, resize, common noise; fixed-order SIMD AXPY and dot. |
| [zencoolchic 5b48b50](https://github.com/imazen/zencoolchic/tree/5b48b509d31b/src/train) | `src/train/` | 11 / 3704 | Hand-written forward/backward, ARM batching, IFCE, SOAP, trainer, quantization and RDOQ. Counts use the pre-bridge commit, so cancellation additions do not change this inventory. |
| [zensim 0dbb08a](https://github.com/imazen/zensim/tree/0dbb08a69625c4179c1df2a61ce5a6ffce3a2c4c/zensim-train-core/src) | `zensim-train-core/src/` | 9 / 6010 | Existing shared dense training infrastructure: f32/f64 SIMD forward/backward, two-layer MLP, Adam, RNG/statistics, specialized pooling/hybrid/per-sample-alpha heads. `simd_encoder.rs` is 1176 lines; `adam.rs` is 73. Already a reuse destination, not a missing library. |
| [zensim 0dbb08a](https://github.com/imazen/zensim/blob/0dbb08a69625c4179c1df2a61ce5a6ffce3a2c4c/zensim-validate/src/adam_simd.rs) | `zensim-validate/src/adam_simd.rs` | 1 / 1053 | Borrowed-slice f64 Adam, scalar reference, SIMD variants and explicit reciprocal-square-root precision modes. |
| [zenanalyze 6bbcacb](https://github.com/imazen/zenanalyze/blob/6bbcacb11313923ded6e5eeebb725878c5736453/zenpicker-train/src/mlp.rs) | `zenpicker-train/src/mlp.rs` | 1 / 588 | Arbitrary-depth f64 LeakyReLU MLP, Adam minibatches, masked-MSE targets, scaler and bake adapter. Not the same trainer contract as the two-layer metric head. |
| [zenanalyze 6bbcacb](https://github.com/imazen/zenanalyze/tree/6bbcacb11313923ded6e5eeebb725878c5736453/zenpredict/src) | `zenpredict/src/` | 20 / 9849 | Prediction format/parser, safety/search and inference; only `inference.rs` (540 lines) is the core NN execution path. Dense inference supports f32/f16/i8 weights, caller-owned scratch and SIMD/scalar parity tests. No training or spatial convolution. |
| [zensim 0dbb08a](https://github.com/imazen/zensim/tree/0dbb08a69625c4179c1df2a61ce5a6ffce3a2c4c/zensim-train-gpu/src) | `zensim-train-gpu/src/` | 3 / 2754 | CubeCL per-sample-alpha trainer: forward, losses, backward and Adam GPU kernels, persistent device buffers and launch orchestration. `backend.rs` is 1557 lines and `kernels.rs` 795. Source contains anchor/equivalence/sigma-floor kernels despite historical module prose describing them as future work. |
| [zenmetrics 080263b](https://github.com/imazen/zenmetrics/tree/080263b10ee0facc4c71522d83342cc500e4ab0a/crates/zensim-gpu/src) | `crates/zensim-gpu/src/` | 17 / 10276 | CubeCL metric feature pipeline, image/color conversion, blur/statistics/reductions, reference caching, formula revisions and memory modes. This is not a general NN trainer. `pipeline.rs` accounts for 3900 lines. |

## Concrete overlap and the differences that matter

**Convolution.** zenjpegai's
[`PackedConv::forward`](https://github.com/imazen/zenjpegai/blob/a04b37a9cac04de1ee109ba02288ada5c06fcb5a/src/nn/fast/conv.rs#L457),
zensr-micro's
[`conv3x3_packed_dispatch`](https://github.com/imazen/zensr/blob/12e1054d09a5ac3a8d7d941c7efe8d88d7ce772a/crates/zensr-micro/src/simd.rs#L1123),
and zencoolchic's `nn::Conv::forward` / `train::ops::conv_forward` all accumulate weighted
neighborhoods with optional bias, nonlinearities and residual addition. Packing, padding,
SIMD axis and accumulation order differ. zenjpegai packs channel blocks into `BTensor`;
zencoolchic uses channel-major planes and vectorizes positions; zensr also has fixed SR and
Winograd paths. Converting every call to one canonical tensor layout could cost more than
sharing the arithmetic saves. That cost has not been measured here.

**Transposed convolution.** zenjpegai's `PackedConvTranspose::forward` and zencoolchic's
`nn::conv_transpose_up2` / `train::ops::conv_t_forward` both split stride parity into output
phases. Cool-chic's kernels are formed from symmetric one-dimensional weights with a
specific replicated-border crop. Share phase execution and padding machinery only if the
adapter reproduces those boundary semantics. Its `conv_t_backward` and `kron_backward` are
training-specific; copying an inference forward kernel does not supply their derivatives.

**Dense layers and AXPY.** `zensim-train-core::simd_encoder::encoder_forward_f32`,
`zenpredict::inference::saxpy_matmul_f32`, and zencoolchic's `train::arm_batch::block_pass`
with `simd::axpy` repeat input-major dense accumulation. The first two skip zero-valued
inputs. zencoolchic's `simd::dot` uses eight lane accumulators and a fixed reduction tree;
a generic SIMD reduction with another lane width is not an equivalent replacement.
Forward/backward adapters also need to preserve the layout of gradient accumulation.

**Adam.** `zensim-train-core::adam::AdamState::step`,
`zensim-validate::adam_simd::adam_update`, `zenpicker-train::mlp::train_minibatch`'s update
loop and `zensim-train-gpu::kernels::{adam_step_kernel,adam_step_atomic_grad_kernel}` repeat
moment updates and bias correction. Their precision, gradient reset/accumulation behavior,
parameter grouping and execution order differ. The CPU core uses f64 and clears gradients;
its state is hardwired to four parameter groups. The GPU implementation uses f32.
Cool-chic's `SoapState::step` additionally has preconditioner initialization, projection,
QR/eigendecomposition and decoupled weight decay. Its Adam-mode epsilon/bias-correction
formula must remain upstream-compatible; substituting a mathematically similar generic
Adam update could alter training. SOAP is not an extraction of the existing Adam alone.

**Tensors and scratch.** zenjpegai's `Tensor<T>`, `BTensor`, `Scratch`-style pool and RAII
`mem::Charge` provide owned checked storage and accounting; zensr-micro has reusable
image-shaped `Scratch`; zencoolchic uses `Planes` and parameter/tape `Vec`s. Shared borrowed
views and optional scratch allocation are plausible. A process-wide hidden recycling pool
should not become a requirement of a `no_std + alloc` core. Also consider the duplicated
half conversion `zensr_micro::f16_to_f32` / `zenpredict::inference::f16_bits_to_f32` as a small
independent extraction, with exhaustive bit-pattern tests before replacement.

## Proposed boundary and API

Start with owned-code extraction, not a new graph runtime or autodiff system. These names
are a design sketch, not approved public API:

```rust,ignore
TensorView<'a, T>       // channels, height, width, row/channel strides, checked extents
TensorViewMut<'a, T>
ArithmeticPolicy       // explicit FMA, accumulation and reduction-order contracts
Execution              // detected SIMD tier, serial/ambient-pool choice, Stop
PackedConv             // packing belongs to this operator/layout, not every tensor
ConvPlan               // padding, dilation, groups, stride, boundary extension, crop
Scratch                // caller-owned, fallible growth; optional allocator/budget adapter
conv_forward_into(...) // views + plan + execution + scratch
conv_backward_into(...)// separate input/weight/bias gradient outputs, explicit accumulation
DensePlan              // input-major layout, precision, zero-skip and activation contract
AdamState<T>           // independent parameter group, explicit moments and step count
adam_step_into(...)    // exact versus approximate reciprocal mode is explicit
```

Default to `no_std + alloc`, `forbid(unsafe_code)`, explicit strided views, no codec or model
serialization dependencies. Keep scalar arithmetic as the reference and preserve dispatch
through archmage/magetypes. A `std` feature can use an ambient rayon pool; it must honor
sequential execution without creating its own pool. Fallible scratch allocation and a
caller-supplied memory budget should be designed before adding a global cache.

Do not force f64 trainers through f32, fuse an originally unfused expression, change a dot
reduction tree, or normalize asymmetric borders into a different convolution convention.
An operator's arithmetic policy is part of its behavior, not merely an optimization hint.
Keep integer entropy models and format-specific fixed-point rules in their codecs; later
integer convolution reuse needs explicit accumulator/rounding/saturation contracts.

## Consumers and migration cost

| Consumer | First useful migration | Cost and acceptance gate |
|---|---|---|
| zencoolchic | SIMD AXPY/dot plus planar synthesis/conv training adapters | Preserve gradient and float-output bounds, integer latents and writer bytes; test every boundary/stride and keep complete-stream thread-count identity. Decoder kernels and training kernels have distinct accumulation histories. |
| zenjpegai | Extract its existing float convolution/blocked tensor machinery as the source implementation, then call it through an adapter | Its `src/nn/mod.rs` fixes `bias → ic → ky → kx` with fused native arithmetic and no reassociation. On wasm all tiers intentionally use unfused arithmetic. Preserve each target's tier/thread bit identity, model packing, pool accounting, integer paths and reference tests. |
| zensr-micro | Adopt compatible packed 1×1/3×3 kernels after adapter proof | Keep its fixed-network optimizations, tile halos and pixel shuffle. Direct and Winograd arithmetic are distinct modes; changing between them is not a parity-preserving refactor. Run existing adopted-model and tiled goldens. |
| zensim-train-core / zensim-validate | Extract borrowed f32/f64 dense kernels and independently configured Adam groups | Keep current f64 training and f32 SIMD tail/FMA behavior. `simd_encoder.rs` explicitly preserves the eight-lane fused domain and unfused tails across wider x86 tiers. Approximate rsqrt variants stay opt-in. Existing lockstep and Adam equivalence tests are migration gates. |
| zenpicker-train | Use dense and f64 Adam primitives under its current training loop | Preserve masked NaN targets, batch normalization of gradients, scaler/bake layout, arbitrary depth and f64 results. It cannot simply switch to the metric trainer's two-layer high-level API. |
| zenpredict | Reuse allocation-free dense primitives, retaining its ZNPR parser and scratch API | Preserve f32/f16/i8 storage, zero skipping, activation order, no_std operation, model-size caps and dispatcher/scalar bit tests. Avoid pulling training, rayon or a tensor pool into inference-only builds. |
| zensim-train-gpu | Later optional backend for compatible dense/Adam operations | Device-resident buffers, atomic gradient accumulation and f32 arithmetic need their own contract and benchmarks. Existing launch sites contain unsafe API usage; these cannot be moved into a safe core by relabeling them. |
| zenmetrics GPU metrics | No immediate NN migration | Reuse existing GPU session/allocation/reduction infrastructure where useful; metric-specific color/statistics/formula-revision kernels remain metric code. A shared CPU NN crate does not remove their orchestration or device-memory work. |

## Concrete start and trigger for expansion

1. Extract a narrow CPU convolution or dense primitive from an existing implementation.
   Select the second consumer before choosing the layout and policy API. zenjpegai plus
   zencoolchic is the convolution pair; zensim plus zenpredict is the dense pair.
2. Land both adapters with existing oracle/bit-parity tests and strided/boundary cases.
   Measure complete operations before/after at tiny, medium and large sizes; include packing,
   scratch/allocation and caller threading. No performance benefit is claimed in this report.
3. Expand to backward or optimizer primitives only when a second real consumer uses the
   same numerical contract. Keep Cool-chic's quantizer, ARM likelihood, warm-up, RDOQ and
   SOAP policy local until such a consumer exists.
4. Consider GPU only after a CPU boundary is stable and two device consumers can reuse an
   operation without per-step uploads or buffer allocation. GPU runtime/session ownership
   should remain separate from the `no_std` CPU package.

This is worth building as shared arithmetic with explicit adapters now. It is not evidence
that a combined inference/training/GPU framework should be built in one pass.
