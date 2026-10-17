# Porting status

Upstream: [Cool-chic](https://github.com/Orange-OpenSource/Cool-Chic) `a6fe38a` (2026-07-17,
v5.0.1), BSD-3-Clause (notice in `upstream-notices/`). Entropy coder: `constriction` 0.4.2
(the Python package's Rust source) with `probability` 0.20.3's Laplace CDF.

## Decoder

| Module | Upstream | Status | Parity evidence |
|---|---|---|---|
| Video / frame / Cool-chic headers | `bitstream/header/*.py` | done (I frames) | kodim14 decodes; `tests/oracle.rs` |
| Network parameters (exp-Golomb, per-module q-step) | `bitstream/neuralnet/*.py` | done | `expgolomb::tests::round_trip_all_orders`; kodim14 |
| Fixed-point ARM / IFCE | `bitstream/component/armint.py` | done | latents bit-exact on kodim14 |
| Wavefront latent decode | `bitstream/component/latent.py` | done | latents bit-exact on kodim14 |
| Range decoder + leaky quantized Laplace | constriction 0.4.2 `queue.rs`, `quantize.rs` | done | latents bit-exact on kodim14 |
| Upsampling (train-mode kron kernels) | `component/core/upsampling.py` | done, scalar | raw output max abs error 3.6e-7 on kodim14 |
| Synthesis | `component/core/synthesis.py` | done, scalar | as above |
| Final resize (nearest / bilinear / bicubic) | `F.interpolate` | done; only the identity case is exercised (kodim14 is full-resolution) | not yet tested on a stream that resizes |
| Common randomness | `component/core/noise.py` | ported, **untested**: no stream with `flag_common_randomness` yet | — |
| YUV 4:2:0 / 4:4:4 output | `io/format/yuv.py` | ported, **untested**: no YUV stream yet | — |
| Inter frames (P/B), warper, global motion | `bitstream/decode.py`, `intercoding/` | not ported: `Error::Unsupported` | — |

### Measured parity (2026-10-17, mac, torch 2.14.1 CPU)

`just oracle` on upstream's `samples/bitstreams/kodim14.cool` (768×512, 10 latent grids incl.
3 hyperlatent, ARM 14 contexts + 6 IFCE features, synthesis `48-1,3-1,3-3r,3-3r`):

- integer latents: 526,272 / 526,272 identical;
- synthesis output before quantization: max |Δ| 3.6e-7;
- 8-bit samples: 1,179,643 identical, 5 differ by one step, none by more.

### Bit-exactness risks

- The leaky Laplace CDF calls `exp` in f64. Upstream gets it from the platform libm through
  Rust std (glibc on Linux wheels, Apple libm on macOS); this port uses the `libm` crate. A
  1-ulp disagreement that moves `floor(free_weight · F(x))` across an integer would change one
  cumulative and desynchronise the range coder. Not yet measured: the next step is an
  exhaustive comparison over every (mu, scale, symbol) the model can produce
  (32768 × 2561 × 128 evaluations) of `libm::exp` vs glibc and Apple libm.
- The Laplace scale table cannot be recomputed bit-exactly (23 of 2561 entries differ from
  `f32(exp(f64))`), so it is embedded from upstream's `mu_scale.npy`
  (`scripts/gen_scale_table.py`).

## Encoder

Not started. Plan: hand-written forward + backward passes on archmage (CPU first), upstream
PyTorch as the per-op gradient oracle; SOAP optimizer, softround + Kumaraswamy noise, STE,
RDOQ and the network quantization search are new code. Reuse candidates surveyed 2026-10-17:
zensim `adam_simd.rs`, `zensim-train-core` dense SIMD kernels, `qat_quantize_copy`,
`zensim-train-gpu` CubeCL kernel templates (all MIT/Apache).

## Reference environment

```bash
git clone https://github.com/Orange-OpenSource/Cool-Chic.git ~/tmp/coolchic-src/Cool-Chic
cd ~/tmp/coolchic-src/Cool-Chic
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p ~/tmp/coolchic-ref && cp samples/bitstreams/kodim14.cool ~/tmp/coolchic-ref/
cd ~/work/zen/zencoolchic && just oracle-dump && just oracle
```

Upstream's decoder builds fresh `nn.Module`s, which are in training mode, so it runs the
non-separable `kron(w, wᵀ)` upsampling kernels. The port follows that path.
