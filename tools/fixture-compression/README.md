# Exact fixture storage comparison

Developer measurement tool; it does not change the regression fixture format or the
codec's dependencies. Requires sibling `zqi` (the `imij` crate) and `zenpng` checkouts.
The measured revisions and results are in
[the fixture storage study](../../benchmarks/fixture_compression_2026-10-18.md).

```sh
just fixture-compression /absolute/path/to/decoder /absolute/path/to/new-output-directory
```

Run under the usual resource wrapper and save the full output to a log. The output
directory must not already exist. Every one of the 12 decoder metadata files is
required. All latent grids, final sample planes and raw float planes are encoded,
decoded and checked for byte identity before their sizes are reported.

IMI uses native I64, U16 and F32 grayscale planes with lossless default effort 7.5,
one encoder thread. PNG uses zenpng Balanced (effort 13), one thread, downcasting
disabled. U16 samples are encoded directly. I64 and F32 bits are transposed into
eight/four vertically stacked Gray8 byte planes; no numerical conversion is done.
The tool currently requires a little-endian host, matching the fixture dumps.

For the independent PNG check and original-array gzip-9 baseline, use a Python
environment with Pillow installed:

```sh
python tools/fixture-compression/summarize.py DECODER_DIR OUTPUT_DIR RUN_LOG NEW_CSV
```

This checks the complete case/part inventory, decodes every PNG with Pillow,
reconstructs the original bytes and asserts identity. The gzip baseline compresses
each original array file separately, while PNG/IMI have a separate file per plane
or grid; reported sizes include file headers. There is no encoder-effort sweep or
speed claim. Gradient/SOAP tensors and `.cool` inputs are outside this comparison.

These transformed files are storage experiments. Current `just oracle-all` reads
the original dump layout; using PNG/IMI fixtures would require a restoration step.

## Zstd and test-time loading

```sh
just fixture-zstd /absolute/path/to/fixture-root /absolute/path/to/new-output-directory
```

This mode requires the `zenzstd` sibling checkout and a `zstd` executable as an
independent measurement/decode oracle. It reads all 68 entries in the portable
bundle's `manifest.json`, compares ordinary zstd with reversible byte-plane
packing, and records per-file and category totals. Every frame is decompressed
with the pure-Rust decoder under its exact expected output-size cap; all original
bytes must agree. Our encoder's frames are additionally decoded with C zstd.
The source fixtures remain intact. The large per-file table stays in the output
directory; category totals and method are in
[the zstd study](../../benchmarks/fixture_zstd_2026-10-18.md).

The measurement helper's `zenzstd::stream::decode_all_with_max` call demonstrates
bounded in-memory recovery. A future regression loader can use the same API as a
test dependency, with checked-in `.zst` files, expected lengths and SHA-256s. It
would require no executable, Python, network or extraction at test runtime.
That regression-loading path is proposed; it is not installed by this study.
