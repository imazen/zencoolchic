# Oracle maintenance

Normal tests use the frozen [upstream references](../../tests/fixtures/upstream/README.md):
`just oracle-all` or `cargo test --release --features oracle,zencodec`. They need no
Python, zstd executable, network fetch or shared data store at test runtime.

This directory is for extending or regenerating the differential oracle. Keep the
existing goldens intact: regenerated training is not guaranteed to produce the
same bytes across Python/torch versions, devices or random generators.

## Saved inputs and environment

- [Source inputs](../../tests/fixtures/sources/README.md): four RGB crops and two
  planar YUV inputs, exactly as used for the original runs, with hashes. Their
  restore helper uses only the Python standard library.
- [Environment freeze](requirements-reference-macos-py312.txt): the original
  macOS ARM64 Python 3.12.13 environment, including torch 2.14.1, torchvision
  0.29.1 and constriction 0.4.2. It records the environment, not a promise that
  these wheels are available on every platform. Normal Rust tests do not need it.
- Upstream [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) commit
  `a6fe38a414dd098b39c41636bd6e423626402f7e` (v5.0.1). Use that exact source
  when judging this port; a different upstream revision is a different oracle.
- `dump_grad.py`: seed 1234, deterministic algorithms, float64 reference mode;
  initial tensors and serialized outputs are F32. The dump includes targets,
  uniform noise, parameters and gradients. Its `wass` config uses the real
  ImageNet VGG16 (a ~528 MB torchvision download on first use); `miniwass`
  substitutes a deterministic ten-convolution miniature (seed 4242) whose flat
  weights are embedded in the dump as `vgg` with `vgg_dims`, so the frozen
  gradient case needs no external weights. Wasserstein dumps also store the
  clamped `decoded` image, `tap{i}` features, `gtap{i}` tap gradients,
  `grad_dec` (`d wasserstein/d decoded`), `wasserstein_per_tap` and
  `dist_weight`.
- `dump_vgg.py`: writes the real VGG16 `features[:23]` weights and biases as one
  flat little-endian F32 file for `Vgg16::from_le_bytes`. Point
  `ZENCOOLCHIC_VGG16` at that file for real-VGG gradient dumps; `TrainOptions::vgg`
  takes the same bytes for `Tune::Wasserstein` encoding.
- `dump_soap.py`: seed 7, initial parameters, fixed gradient steps and final
  parameters. Rank-deficient trajectory differences are explained by
  [soap_null_space_probe.py](soap_null_space_probe.py) and [PORTING.md](../../PORTING.md).

## Set up a fresh reference checkout

The following creates new directories; choose unused paths. Keep scratch on disk.
The original environment freeze is intended for macOS ARM64. On another OS, use
upstream's requirements and record the actual resolved environment separately.

```sh
export TMPDIR="$HOME/tmp"
mkdir -p "$TMPDIR"
export COOLCHIC_UPSTREAM="$HOME/tmp/coolchic-src/Cool-Chic"
export ZENCOOLCHIC_SOURCES="$HOME/tmp/coolchic-sources"

# From the zencoolchic repository:
python3 scripts/oracle/make_sources.py "$ZENCOOLCHIC_SOURCES"

git clone https://github.com/Orange-OpenSource/Cool-Chic.git "$COOLCHIC_UPSTREAM"
cd "$COOLCHIC_UPSTREAM"
jj git init --colocate
jj new a6fe38a414dd098b39c41636bd6e423626402f7e -m "Pinned Cool-chic oracle"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Use the recorded freeze instead of `requirements.txt` when rebuilding the original
macOS environment. Keep the absolute zencoolchic path available for its scripts.
The scripts take the upstream checkout explicitly; the dump commands below are
run from the zencoolchic repository.

## Regenerate into new raw directories

```sh
export ZENCOOLCHIC_ORACLE_DIR="$HOME/tmp/coolchic-regenerated/decoder"
export ZENCOOLCHIC_GRAD_DIR="$HOME/tmp/coolchic-regenerated/gradients"
export ZENCOOLCHIC_SOAP_DIR="$HOME/tmp/coolchic-regenerated/soap"

# Run the encoder under the usual resource controls; this performs real training.
bash scripts/oracle/make_streams.sh \
  "$COOLCHIC_UPSTREAM" "$ZENCOOLCHIC_SOURCES" "$ZENCOOLCHIC_ORACLE_DIR"
cp "$COOLCHIC_UPSTREAM/samples/bitstreams/kodim14.cool" "$ZENCOOLCHIC_ORACLE_DIR/"
just oracle-dump
just grad-dump
just soap-dump
just oracle-all
```

The explicit environment variables make tests use those raw dumps. All 12 decoder
and four gradient cases are still required, and any additional cases are checked.
The checked-in compressed payloads are always tested for integrity independently
of overrides. Unset the three override variables to test frozen references again.

The decoder dumper currently captures a **single still frame**. Before porting
video, extend it to label intermediates by frame and residue/motion network;
one shared `captured["raw"]` is insufficient for inter frames.

## Restore and repack the original frozen bytes

For inspecting reference arrays or rebuilding the original fixture layout:

```sh
just fixture-unpack "$HOME/tmp/coolchic-frozen-raw"
just fixture-pack "$HOME/tmp/coolchic-frozen-raw" "$HOME/tmp/coolchic-repacked"
```

These maintenance helpers need a zstd executable; packing pins version 1.5.7.
Both refuse to replace an existing directory and verify hashes. The original
compressed files and all six source inputs were restored, checked and repacked
successfully on 2026-10-18. Repacking reproduced all 68 compressed payloads exactly
on that environment. Original SHA-256s remain the primary reference identity.
`small64_miniwass` was added on 2026-10-29 (70 payloads) with its own provenance
above; the 68 older payloads are byte-identical.

Do not point the packer at newly generated data while retaining an old manifest:
it will reject changed hashes. Accepting a new golden set requires an intentional
provenance/manifest update and review; it must not disguise a failed comparison.
Keep existing still-image regressions when adding video or new encoder tools.
