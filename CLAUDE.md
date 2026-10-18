# zencoolchic — project instructions

Port of Cool-chic 5 (Orange-OpenSource/Cool-Chic, `a6fe38a`) to safe Rust. User request
2026-10-17: "port cool-chic 5, reusing stuff from zenjpegai"; license BSD-3-Clause (user
choice). Imazen owns all imazen/lilith code outright, AGPL repos included: copy from zenjpegai,
zensim, zenanalyze/zentrain, zensr, zenmetrics freely, relicensed to BSD-3-Clause, and cite
the origin repo + commit in the commit message. Don't rewrite what those repos already have.

Read `PORTING.md` first: the status table and the parity evidence.

## Hard rules

- `#![forbid(unsafe_code)]`, `no_std + alloc`.
- The entropy stage (headers, exp-Golomb, fixed-point ARM/IFCE, range decoder) is integer or
  pinned-f64 code and MUST reproduce upstream's decoded latents exactly. Any mismatch is a bug.
- Float networks (upsampling, synthesis, resize) match PyTorch to rounding only; the measured
  difference is recorded in `PORTING.md`. Don't loosen a recorded bound without asking.
- Upstream Python is the differential oracle. Frozen dumps live in
  `tests/fixtures/upstream/` and decompress in Rust during `--features oracle` tests.
  Python and the pinned upstream checkout are needed only for regeneration.
  `ZENCOOLCHIC_ORACLE_DIR` / `ZENCOOLCHIC_GRAD_DIR` / `ZENCOOLCHIC_SOAP_DIR` explicitly
  select uncompressed regenerated dumps; no missing-file fallback occurs.
- The user approved the initial frozen binary fixture set on 2026-10-18. Future binary
  additions and changes to golden references still need approval.

## Commands

- `just check` (includes codec conformance and the frozen upstream oracle)
- `just oracle-all`, or `cargo test --release --features oracle,zencodec`
- `just oracle-dump` regenerates decoder dumps with the upstream venv; ordinary tests do
  not regenerate references. See `tests/fixtures/upstream/README.md`.

## Known Bugs

- Fixed validation gap (2026-10-18): the decoder oracle printed reconstruction errors
  without asserting the recorded bounds and could overlook a missing whole case. The
  frozen inventory, lengths, shapes, hashes, raw bound and sample bound are now enforced.
