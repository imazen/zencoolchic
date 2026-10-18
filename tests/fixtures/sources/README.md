# Frozen oracle generator inputs

The six files here (37,750 bytes total) are the original inputs used for the
upstream streams and gradient dumps. Restore them to a new directory with:

```sh
python3 scripts/oracle/make_sources.py "$HOME/tmp/coolchic-sources"
```

The helper validates sizes and SHA-256s before copying. It does not transform the
images or regenerate these bytes from a decoder under test. The four RGB images
were verified pixel for pixel as crops of the recorded kodim14 source image;
`manifest.json` records their crop rectangles and the source image's hash. The
YUV input bytes are preserved directly; their original conversion recipe was not
reconstructed. This does not prevent reproducing the oracle input.

The YUV filenames follow upstream's dimension/subsampling/bitdepth convention.
Samples are planar; 10-bit samples are stored in little-endian 16-bit words.
The input files are not needed for ordinary Rust regression tests. See
[oracle maintenance](../../../scripts/oracle/README.md) for the pinned upstream
environment and regeneration procedure.
