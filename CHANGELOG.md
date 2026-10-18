# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### QUEUED BREAKING CHANGES

- None yet (the crate is unpublished).

### Changed

- Relicensed to BSD-3-Clause to match upstream; retained the Orange notice.

### Added

- Source-based assessment of shared NN primitives in `docs/zennn-assessment.md`.

- Optional zencodec RGB still-image bridge and public encoder configuration; cooperative
  cancellation, header-only probe, strided buffers, and documented uncalibrated quality.

- Encoder usage and pipeline documentation in `docs/ENCODER.md`.

- Cool-chic 5 image decoder: headers, exp-Golomb network parameters, fixed-point ARM and
  IFCE, constriction-compatible range decoder, upsampling, synthesis, RGB / YUV 4:2:0 /
  YUV 4:4:4 output at 8–16 bits.
- Differential test against the upstream Python decoder (`--features oracle`).
