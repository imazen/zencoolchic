# Frozen Cool-chic 5 regression references

These 70 zstd payloads preserve every original reference byte: 12 decoder/writer
cases, four gradient cases, and the SOAP optimizer reference. Original payloads
total 14,331,585 bytes; compressed payloads total 5,858,006 bytes. No Python
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
9.3e-7 and output-sample difference at most one step. Gradient and SOAP thresholds
retain their existing values; rank-deficient SOAP trajectories remain diagnostic.

## Provenance and maintenance

References come from [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic)
`a6fe38a414dd098b39c41636bd6e423626402f7e` (v5.0.1), using the tracked scripts
in [scripts/oracle](../../../scripts/oracle). The kodim14 stream is upstream's
`samples/bitstreams/kodim14.cool`; the other streams use `make_streams.sh`.
The original environment was macOS ARM64, Python 3.12.13, torch 2.14.1 CPU.
Gradient references ran torch in float64 and serialized tensors as F32; the
gradient seed is 1234 and SOAP seed is 7. Decoder metadata retains shapes and the
torch version. `small64_miniwass` exercises the Wasserstein distortion path with
common randomness; its feature extractor is a deterministic ten-convolution
miniature (seed 4242, embedded in the dump as `vgg`/`vgg_dims`) rather than
ImageNet VGG16, so no 528 MB download is needed. Upstream's license notice is
retained in [upstream-notices](../../../upstream-notices/COOL-CHIC-LICENSE.txt).

Packing uses [pack_fixtures.py](../../../scripts/oracle/pack_fixtures.py) with
zstd 1.5.7, `-19 -T1`, from the original layout and its manifest. It refuses to
replace an existing directory and verifies all original hashes. The small size
difference from the earlier zstd study is due to packing from stdin instead of
named files; original hashes are identical. Regenerating upstream training on a
new platform is not promised to reproduce the same bytes. Preserve these goldens
and add new cases with their own provenance instead of silently replacing them.

For explicit regeneration comparisons, `ZENCOOLCHIC_ORACLE_DIR`,
`ZENCOOLCHIC_GRAD_DIR`, and `ZENCOOLCHIC_SOAP_DIR` select raw, uncompressed
directories. They must include every required case; additional cases are tested
too. These explicitly selected new dumps are compared numerically and are not
required to have the frozen files' hashes. No missing-file fallback occurs.
