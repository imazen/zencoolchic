# Encoder

Enable the `encoder` feature. `examples/encode.rs` reads a binary P6 RGB8 PPM, trains and
writes a `.cool` stream, decodes it back, and reports size and PSNR. Its positional arguments
are input, output, iteration budget, lambda, and architecture index:

```sh
cargo run --release --example encode --features encoder -- input.ppm output.cool 10000 0.001 1
```

The current direct entry points live in the doc-hidden `unstable` module and are experimental.
The encoder accepts interleaved RGB8; it does not convert color. Use full-range sRGB input.
It does not encode YUV, higher bit depths, common randomness / Wasserstein tuning,
sub-full-resolution starting latents, or video. The decoder supports more stream tools than
the encoder creates; see [the porting table](../PORTING.md#decoder).

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `lambda` | 0.001 | Objective `MSE + lambda × bits/pixel`; larger values favor smaller streams |
| `n_itr` | 10,000 | Total budget including warm-up and 500 hard-round iterations; minimum 2000 |
| `start_lr` | 0.01 | Initial learning rate, followed by the phase's cosine schedule |
| `arch` | 1 (`hop`) | 0 `vhop`, 1 `hop`, 2 `mop`, 3 `lop` |
| `seed` | 0 | Initialization and quantization-noise seed; the RNG differs from torch |
| `rdoq` | true | Greedy rate-distortion refinement after network quantization |

Architectures follow upstream's intra configurations. ARM spatial contexts are 20, 14, 10,
6 respectively; first synthesis widths are 64, 48, 16, 8. Latents start at full resolution;
the last grid is at resolution 6 below 1,000,000 pixels, 7 below 3,000,000, otherwise 8.
Architecture controls the coded decoder as well as encoding cost. No generic quality
calibration has been measured; lambda is the native rate-distortion control.

Upstream's encoder-gain rule is preserved exactly in `train::trainer::encoder_gain`:
lambda < 0.0002 uses gain 24; 0.0002 ≤ lambda < 0.0005 uses 20; lambda ≥ 0.0005 uses 16.
This rescales latents internally; it is not a calibrated output quality scale.

## Pipeline

1. Initialize five candidate models. Warm-up trains five, then the best two, retaining the
   best one. Each candidate phase uses 50/100/200/400 iterations for total budgets below
   4000/9000/12000/otherwise. These steps count against the total budget.
2. Train the main phase with Gaussian noise and softround, gradient clipping at 0.1, SOAP
   for model tensors and Adam-mode SOAP for latents, cosine learning rate, validation,
   patience, and restoration of the best validated parameters.
3. Run 500 hard-round straight-through iterations with no quantization noise. The training
   rate estimate covers latents, not the final serialized network overhead.
4. Search weight/bias quantization steps per module, choosing exp-Golomb orders and including
   estimated network rate in the objective.
5. Optionally run RDOQ with greedy ±1 through ±15 moves. Each candidate score currently runs
   a full evaluation; incremental updates are not implemented.
6. Convert to stream integers and write network parameters, integer ARM/IFCE latents, range
   coding, and headers. The reported final PSNR comes from decoding the written stream.

For a fixed seed, input, settings and CPU SIMD tier, the training kernels preserve arithmetic
order across thread counts. `train::eval::tests::training_step_is_independent_of_thread_count`
checks bit-identical gradients and loss for a forward/backward pass with 1 and 4 threads.
This is a step-level contract, not a measured cross-platform complete-stream guarantee.
RNG noise is partitioned into fixed blocks of 4096 samples, independent of worker count.
Cross-platform bit-identical float training and torch-identical random initialization are
not promised.

## Evidence and limits

[PORTING.md](../PORTING.md#encoder) records float64 gradient checks, SOAP comparisons,
byte-identical re-encoding of 12 upstream oracle streams, and upstream decoding of this
encoder's small64 and CID22 output. It also records the rank-deficient SOAP null-space issue.
[The CID22 benchmark](../benchmarks/encode_cid22_1279330_2026-10-17.tsv) records commit,
settings, bytes, PSNR, loss and wall time for both encoders. It is one image and one seed per
encoder; no multi-image, multi-seed rate-distortion or quality calibration exists yet.

Run `just check` for the local gate and `just oracle-all` for decoder, writer, gradient and
optimizer parity. The oracle needs the external pinned upstream checkout and dumps described
in [PORTING.md](../PORTING.md#reference-environment).
