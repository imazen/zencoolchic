# zencoolchic — project instructions

Port of Cool-chic 5 (Orange-OpenSource/Cool-Chic, `a6fe38a`) to safe Rust. User request
2026-10-17: "port cool-chic 5, reusing stuff from zenjpegai"; license MIT OR Apache-2.0 (user
choice), so AGPL code from zenjpegai / zensr / zenmetrics is not copied in without the user
relicensing it. MIT/Apache sources (zensim, zenpredict) may be reused.

Read `PORTING.md` first: the status table and the parity evidence.

## Hard rules

- `#![forbid(unsafe_code)]`, `no_std + alloc`.
- The entropy stage (headers, exp-Golomb, fixed-point ARM/IFCE, range decoder) is integer or
  pinned-f64 code and MUST reproduce upstream's decoded latents exactly. Any mismatch is a bug.
- Float networks (upsampling, synthesis, resize) match PyTorch to rounding only; the measured
  difference is recorded in `PORTING.md`. Don't loosen a recorded bound without asking.
- Upstream Python is the differential oracle and lives outside the repo
  (`~/tmp/coolchic-src/Cool-Chic` with a `.venv`); oracle dumps live outside too
  (`ZENCOOLCHIC_ORACLE_DIR`, default `~/tmp/coolchic-ref`). No bitstreams in git without the
  user's OK.

## Commands

- `just test`, `just clippy`
- `just oracle-dump` (needs the upstream venv), then `just oracle`
