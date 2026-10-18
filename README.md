# zencoolchic

A safe-Rust port of [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) 5, Orange's
overfitted neural image codec: each image carries its own tiny decoder (latent grids, an
auto-regressive entropy model, upsampling and a small synthesis CNN, about 2000 MAC per pixel).

**Status: decoder for still images; no encoder yet.** On upstream's sample stream the integer
latents match the reference decoder exactly and 5 of 1,179,648 output samples differ by one
step. Inter frames, and an encoder, are not ported. [`PORTING.md`](PORTING.md) has the
module-by-module status and what each claim rests on.

```rust
let data = std::fs::read("kodim14.cool")?;
let picture = zencoolchic::decode(&data, &zencoolchic::Limits::default())?;
let rgb = picture.to_rgb8(); // Some(interleaved RGB) for 8-bit RGB streams
```

`#![no_std]` + `alloc`, `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. Derived from Cool-chic (BSD-3-Clause, © Orange), whose notice is in
[`upstream-notices/`](upstream-notices/).
