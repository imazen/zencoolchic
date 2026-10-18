# Lossless storage of decoder regression references

Measured 2026-10-18 on macOS ARM64, M4 Pro. Source inputs are the 12 decoder
oracle cases documented in [PORTING.md](../PORTING.md), from upstream
[Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) commit
`a6fe38a414dd098b39c41636bd6e423626402f7e`. Zencoolchic base:
`31fef7d426aec8ee562ee5b51b18f15b41c2d536`.

All sizes below are measured bytes, including codec headers. Every one of the
188 grids/planes round-tripped byte for byte through both tested codecs. All
PNGs additionally decoded with Pillow 13.0.0 and recovered the original arrays
exactly. No float rounding, integer narrowing or tolerance was used for storage.

| Reference data | Original dumps | gzip-9 | PNG | IMI |
|---|---:|---:|---:|---:|
| Integer latents, I64 | 4,824,120 | 114,557 | 270,795 | 105,439 |
| Final pixel samples, U16 | 2,752,818 | 1,183,442 | 934,157 | 695,775 |
| Raw reconstruction, F32 | 5,514,852 | 4,842,343 | 3,831,527 | 3,633,760 |
| Total decoder arrays | 13,091,790 | 6,140,342 | 5,036,479 | 4,434,974 |

PNG is 17.98% smaller and IMI 27.77% smaller than gzip-9 for these decoder arrays
in aggregate. PNG is larger than gzip-9 for the latents. These are the tested
settings, not an encoder-effort search or a general codec ranking.

## Method and revisions

- `imij` from [imazen/zqi](https://github.com/imazen/zqi),
  `a34cc0b7c162c8a78b135343777f111dc37d9bb9`: lossless default effort 7.5,
  one thread, native grayscale I64/U16/F32.
- [zenpng](https://github.com/imazen/zenpng)
  `e4f4ed3e1dbb55b001b8b9f812599fa4f76ccba8`: Balanced (effort 13), one thread,
  all downcast flags disabled; zenflate
  `a0ea03637a2cc9a193e343018ae2cfb1a6c31edd`.
- PNG stores U16 directly. F32 and I64 store their exact bits as four/eight
  vertically stacked Gray8 byte planes. They are numeric containers, not display
  renditions. The source JSON retains shapes and the original sample types.
- PNG/IMI have one file per latent grid or image plane. Gzip-9 compresses each
  original array file separately (12 per category), retaining its original
  layout. Different framing is included in the measured totals.
- Byte order is little endian; the measurement helper refuses big-endian hosts.
- This comparison excludes `.cool` streams, gradient/SOAP tensors, metadata and
  enclosing archives. The previously prepared complete gzip bundle is 6,709,984
  bytes and is not directly comparable to the decoder-only total above.

[Per-part measurements](fixture_compression_2026-10-18.csv) and the
[measurement tool](../tools/fixture-compression/README.md) retain the commands
and exact packing/recovery procedure. Example invocation, with absolute paths:

```sh
just fixture-compression "$DECODER_DIR" "$NEW_OUTPUT_DIR" > "$RUN_LOG" 2>&1
python tools/fixture-compression/summarize.py \
  "$DECODER_DIR" "$NEW_OUTPUT_DIR" "$RUN_LOG" "$NEW_CSV"
```

Run the command under the development resource wrapper; its successful run was:

```text
run-heavy: done rc=0 7s | peak-RSS 0.07GiB | min-avail 14901MiB | peak-load 3.49
```

This is harness process accounting, not a codec memory or throughput benchmark.
The wrapper on macOS reports a monitored process-tree RSS guard, not a cgroup cap.

## Fixture storage implications

Both image formats reduce the decoder reference size at the tested settings.
PNG is a long-established independent interchange format for the final pixels;
its float/integer byte-plane layout requires explicit recovery code. IMI supports
the numeric types directly and produces the smaller tested files, but its format
is currently experimental and unfrozen. Long-lived fixtures would need a pinned
reader or migration support before adopting it.

The original dumps and complete gzip bundle remain intact. This study changes no
test expectation or fixture loader, and commits no encoded binary. Retaining a
gzipped original copy preserves the current Rust-only `just oracle-all` workflow.
