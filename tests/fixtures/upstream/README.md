# Frozen Cool-chic 5 regression references

These 113 zstd payloads preserve every original reference byte: 12 decoder/writer
cases, seven gradient cases, two video cases, a warper vector case, and the SOAP
optimizer reference. Original payloads total 16,839,098 bytes; compressed payloads
total 7,000,148 bytes. No Python
environment, external executable, download or shared storage is needed to run:

```sh
cargo test --release --features oracle,zencodec
# or: just oracle-all
```

The shared [test helper](../../support/oracle_fixtures.rs) decompresses each file
in memory with an exact output-size cap. It verifies compressed and original
SHA-256s, lengths, byte-plane layout, and the complete required case inventory.
The decoder is a pinned pure-Rust `zenzstd` test dependency. Corruption and missing
files fail tests. `oracle` implies `encoder`; plain `cargo test` does not enable
the upstream-oracle feature. CI and `just check` enable it explicitly.

## Format

`manifest.json` schema 1 lists original relative paths, original sizes/SHA-256s,
compressed sizes/SHA-256s and `shuffle_word_bytes`. Each `<path>.zst` is one
dictionary-free zstd frame. Before compression, bytes are transposed over the
whole original file: for N samples of B bytes, stored byte `b*N+i` is original
byte `i*B+b`. B is 8 for I64 latents, 4 for F32 reconstruction/gradient/SOAP
tensors, 2 for U16 pixels, and 1 for JSON and input streams. All numeric samples
retain their original little-endian bit patterns; no numerical conversion occurs.

Expected outputs contain full arrays, not hashes alone. Exact latents and writer
bytes are asserted. Decoder reconstruction is held to the existing raw bound
1e-6 (residue and motion synthesis) and output-sample difference at most one
step. Gradient and SOAP thresholds
retain their existing values; rank-deficient SOAP trajectories remain diagnostic.

## Provenance and maintenance

References come from [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic)
`a6fe38a414dd098b39c41636bd6e423626402f7e` (v5.0.1), using the tracked scripts
in [scripts/oracle](../../../scripts/oracle). The kodim14 stream is upstream's
`samples/bitstreams/kodim14.cool`; the other streams use `make_streams.sh`.
The original environment was macOS ARM64, Python 3.12.13, torch 2.14.1 CPU.
Gradient references ran torch in float64 and serialized tensors as F32; the
gradient seed is 1234 and SOAP seed is 7. Decoder metadata retains shapes and the
torch version. `small64_yuv420` exercises the 4:2:0 training loss: the target is
upstream's `convert_444_to_420(img)` dict and the distortion is the
per-plane-sample-weighted MSE over the downsampled-clamped decode.
`small64_lop` exercises below-full-resolution latents (`latent_resolution`
(2,4)): the synthesis runs at quarter resolution and `rescale_output`
bicubically interpolates back to the image size.
`small64_miniwass` exercises the Wasserstein distortion path with
common randomness; its feature extractor is a deterministic ten-convolution
miniature (seed 4242, embedded in the dump as `vgg`/`vgg_dims`) rather than
ImageNet VGG16, so no 528 MB download is needed. The `multi_i` video case is a
three-frame all-intra stream assembled from `small64_lop_2k`, `yuv420_8b_debug`
and `small64_vhop_2k` by `dump_video.py` using upstream's own header classes;
the expected planes are upstream `decode_video` output. The `pb_128` video case
was encoded by upstream's `cc_encode.py` (`--debug` preset, warp filter size 8)
on 2026-10-29 on Linux x86-64, Python 3.12, torch 2.15.0+cpu: a synthetic
four-frame 128×128 4:2:0 source with real motion, coded I→P→B→B.
`dump_video.py --decode` captured upstream's decoded planes plus the raw
residue and motion synthesis outputs (`.f32`) for each frame.
`small64_flow` exercises the `flow` frame data type used by
`guided_motion_pretraining`: a two-channel dense output trained against a
seeded flow target with a flat unclamped MSE. The `vec` warp
case holds `dump_warp.py` output: one seeded 20×24×3 input, its flow field,
upstream `Warper` outputs at filter sizes 2/4/8/12, the training-mode autograd
flow gradients `d(warp)/d(flow)` per filter size, a `vanilla_warp_fn`
nearest translation (dx=3, dy=-2), and two `get_global_translation`
search pairs with upstream's answers. Upstream's license
notice is retained in
[upstream-notices](../../../upstream-notices/COOL-CHIC-LICENSE.txt).

Packing uses [pack_fixtures.py](../../../scripts/oracle/pack_fixtures.py) with
zstd 1.5.7, `-19 -T1`, from the original layout and its manifest. It refuses to
replace an existing directory and verifies all original hashes. The small size
difference from the earlier zstd study is due to packing from stdin instead of
named files; original hashes are identical. Regenerating upstream training on a
new platform is not promised to reproduce the same bytes. Preserve these goldens
and add new cases with their own provenance instead of silently replacing them.

For explicit regeneration comparisons, `ZENCOOLCHIC_ORACLE_DIR`,
`ZENCOOLCHIC_GRAD_DIR`, `ZENCOOLCHIC_SOAP_DIR`, `ZENCOOLCHIC_VIDEO_DIR`, and
`ZENCOOLCHIC_WARP_DIR` select raw, uncompressed directories. They must include every required case; additional cases are tested
too. These explicitly selected new dumps are compared numerically and are not
required to have the frozen files' hashes. No missing-file fallback occurs.
