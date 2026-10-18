# Zstd storage of frozen regression fixtures

Measured 2026-10-18, macOS ARM64, M4 Pro. Zencoolchic base:
`fa87666e97b93ba496b9e5e13b6d0771b0581837`. Inputs are all 68 fixture files
from the portable bundle, totaling 13,877,910 bytes: 12 decoder cases, three
gradient cases, SOAP data and their JSON metadata. The upstream source is
[Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic)
`a6fe38a414dd098b39c41636bd6e423626402f7e`.

Every file is an independent zstd frame, with no external dictionary. Sizes below
include frame overhead. They exclude filesystem/enclosing-archive overhead and
the bundle-level manifest/environment freeze. The earlier 6,709,984-byte gzip
archive includes those extras and tar headers.

| Encoder | Level | Original layout | Reversible byte planes |
|---|---:|---:|---:|
| zenzstd | 3 | 7,485,231 | 6,259,097 |
| zenzstd | 9 | 7,359,290 | 6,087,910 |
| zenzstd | 15 | 7,276,107 | 5,845,042 |
| C zstd | 19 | 6,485,213 | 5,581,265 |

These are selected settings, not an effort calibration or encoder ranking.
The [category CSV](fixture_zstd_2026-10-18.csv) includes all measured cells and
separates the decoder arrays from other data for comparison with the
[PNG/IMI study](fixture_compression_2026-10-18.md).

## Method and verification

- [zenzstd](https://github.com/imazen/zenzstd) source:
  `21ba0c294900ef8d346eaa7255a489d84fdce8f3`. Features: `std`, `hash`; SIMD
  disabled. C reference: `zstd --version` reported 1.5.7, `-T1` encode.
- Plain mode compresses each original file unchanged. Byte-plane mode first
  transposes each I64, U16 or F32 array into eight, two or four byte planes.
  Gradient/SOAP `.bin` payloads are F32 arrays, also transposed four ways. No
  arithmetic, float rounding or integer narrowing occurs. JSON and streams use
  a byte width of one, so remain unchanged before compression.
- Every frame, including C-encoded frames, is decoded with
  `zenzstd::stream::decode_all_with_max(frame, Some(expected_length))`, then
  inverse-transposed if necessary and compared to every original byte. All
  68 files pass in all eight encoder/level/layout combinations. Our encoder's
  frames additionally pass independent C-zstd decoding with exact byte identity.
- Byte shuffling is not universally beneficial: C level 19 gradient/SOAP data
  grows from 478,634 to 505,351 bytes, while raw decoder floats shrink from
  4,838,142 to 3,989,345. The CSV retains both results.
- `per-file.csv` in the measurement output retains individual sizes. It is not
  committed because it exceeds the generated-data size limit. The tracked tool
  regenerates it from the fixture bundle with:

```sh
just fixture-zstd "$FIXTURE_ROOT" "$NEW_OUTPUT_DIR"
```

Run under the development resource wrapper and retain stdout/stderr. The
successful compression plus verification run reported:

```text
run-heavy: done rc=0 11s | peak-RSS 0.44GiB | min-avail 14020MiB | peak-load 2.70
```

This is whole-harness accounting, not a decompression speed or codec memory
benchmark.

## Proposed regression loading

Keep independent `.zst` payloads and a manifest containing the required case
inventory, original lengths, hashes, sample layout and upstream provenance.
A shared test helper reads a payload from the repository, decodes in memory with
the expected length as its output cap, reverses any byte packing and verifies
length and SHA-256 before interpreting the reference values. Missing cases,
corrupt files or hash mismatches fail the tests.

This runs during `cargo test --release --features oracle,zencodec`; `oracle`
already enables `encoder`. The decoder belongs in the test dependency graph.
There is no build script, external decompression command, extraction directory,
Python environment, reference regeneration, network fetch or NAS dependency at
test runtime. Zstd's byte format is documented in
[RFC 8878](https://www.rfc-editor.org/rfc/rfc8878.html).

The Rust decompressor has been exercised by this measurement tool. The
regression loader and checked-in binary fixtures are not installed by this
study. Current oracle tests still read uncompressed files from explicit
environment-variable directories. Binary-commit approval and the requested
strengthening of existing regression assertions remain pending.
