# zencoolchic

A safe-Rust port of [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) 5, Orange's
overfitted neural image codec: each image carries its own tiny decoder (latent grids, an
auto-regressive entropy model, upsampling and a small synthesis CNN, about 2000 MAC per pixel).

**Status: still-image decoder and experimental encoder.** The decoder's integer latents
match upstream on 12 oracle streams; floating-point reconstruction differs by at most one
output step. Inter frames are not ported. See [`PORTING.md`](PORTING.md) for evidence.

```rust
let data = std::fs::read("kodim14.cool")?;
let picture = zencoolchic::decode(&data, &zencoolchic::Limits::default())?;
let rgb = picture.to_rgb8(); // Some(interleaved RGB) for 8-bit RGB streams
```

`#![no_std]` + `alloc`, `#![forbid(unsafe_code)]`.

## Encoder

Enable `encoder` to train a decoder for each RGB8 image, quantize its networks, run optional
RDOQ, and write a `.cool` stream. For a binary P6 RGB8 PPM:

```sh
cargo run --release --example encode --features encoder -- input.ppm output.cool 10000 0.001 1
```

The last arguments are iteration budget (minimum 2000), lambda, and architecture
(`0` vhop, `1` hop, `2` mop, `3` lop). On CID22 image 1279330 (512×512, lambda 0.001,
10,000 iterations, hop, one seed each), this port produced **10,692 bytes / 37.988 dB in
507 s**, versus upstream's **10,769 bytes / 38.157 dB in 1,065 s** on the same M4 Pro.
These are one-image observations, not a quality or speed calibration
([recorded benchmark](benchmarks/encode_cid22_1279330_2026-10-17.tsv)).

Encoding YUV/10-bit input, common randomness / Wasserstein tuning, latents starting below
full resolution, and video are not supported. [Encoder details](docs/ENCODER.md) covers
settings, determinism, the training pipeline, and parity evidence.

Optional `zencodec` support provides RGB probing/decoding and, with `encoder`, a public
trainer-backed encoder configuration. See [the trait API](docs/ENCODER.md#zencodec-api)
for cancellation, limits and the uncalibrated quality dial.

## License

BSD-3-Clause. Derived from Cool-chic (BSD-3-Clause, © Orange), whose notice is in
[`upstream-notices/`](upstream-notices/).
