"""Gradient oracle: one deterministic training forward + backward of upstream's CoolChicEncoder.

Usage: python -I scripts/oracle/dump_grad.py <Cool-Chic checkout> <image.png> <out_prefix> [config]
config: "hop" (default), "small" (tiny networks, fast), "wass" (common randomness +
  Wasserstein tune on the small networks), "miniwass" (same, but the loss's VGG16 is
  replaced by a tiny seeded conv stack with the same tap/pool positions; the weights
  ride along in the dump so no external file is needed), "yuv420" (small networks;
  the target is the input downsampled to a 4:2:0 dict, and the loss is upstream's
  per-plane-sample-weighted MSE).

Latents and all network weights get seeded random values (upstream initializes latents to
zero, which would make most gradients trivially zero). The pass is deterministic: the
quantizer's uniform noise (`torch.rand_like`) is replaced by a seeded tensor that is dumped
too (upstream's "none" noise type crashes with the "softround" quantizer).

Writes <out_prefix>.json (config, loss terms, tensor index) and <out_prefix>.bin: every
tensor as little-endian f32, in the order listed in the index: target image, each parameter,
then each parameter's gradient.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])

import numpy as np  # noqa: E402
import torch  # noqa: E402

from coolchic.component.core.coolchic import CoolChicEncoder, CoolChicEncoderParameter  # noqa: E402
from coolchic.io.format.png import read_png  # noqa: E402

torch.manual_seed(1234)
torch.use_deterministic_algorithms(True)
config = sys.argv[4] if len(sys.argv) > 4 else "hop"
# Optional 5th argument "f64": run the same pass in float64 (ground truth for judging
# float32 rounding); tensors are still written as f32.
dtype = torch.float64 if len(sys.argv) > 5 and sys.argv[5] == "f64" else torch.float32
img, _bitdepth = read_png(sys.argv[2])  # [1, 3, H, W] in [0, 1]
h, w = img.shape[-2:]
flag_cr = config in ("wass", "miniwass")
dist_weight = {"mse": 0.2, "wasserstein": 0.8 / 200} if flag_cr else {"mse": 1.0}
# Miniature VGG replacement (miniwass): (c_in, c_out) of the 10 convs, same
# conv/ReLU/pool/tap slot pattern as torchvision vgg16.features[:23].
MINI_VGG_DIMS = [
    (3, 4), (4, 4), (4, 8), (8, 8), (8, 4), (4, 4), (4, 8), (8, 4), (4, 4), (4, 4)
]
if config == "hop":
    layers = ["48-1-linear-relu", "3-1-linear-none", "3-3-residual-relu", "3-3-residual-none"]
    arm = dict(spatial_context_arm=14, n_hidden_layers_arm=2, output_feature_ifce=6)
    hyper = (4, 6)
    lat = (0, 6)
else:
    # "small", "wass" and "miniwass" share the tiny networks.
    layers = ["8-1-linear-relu", "3-1-linear-none", "3-3-residual-none"]
    arm = dict(spatial_context_arm=6, n_hidden_layers_arm=1, output_feature_ifce=2)
    hyper = (3, 4)
    lat = (0, 4)
param = CoolChicEncoderParameter(
    layers_synthesis=layers,
    linear_stabiliser_synth=True,
    ups_k_size=8,
    ups_preconcat_k_size=7,
    ifce_resolution=(0, 2),
    linear_stabiliser_arm=True,
    latent_resolution=lat,
    hyperlatent_resolution=hyper,
    flag_common_randomness=flag_cr,
    img_size=(h, w),
    final_upsampling_type="bicubic",
    **arm,
)
cc = CoolChicEncoder(param)
with torch.no_grad():
    for name, p in cc.named_parameters():
        if "latent_grids" in name:
            p.copy_(torch.randn_like(p) * 0.4)
        elif "output_transform" in name:
            continue
        else:
            p.copy_(torch.randn_like(p) * 0.2)
cc = cc.to(dtype)
img = img.to(dtype)
# `self.cr` is a plain tensor list, not a buffer: `.to(dtype)` does not reach it.
cc.cr = [t.to(dtype) for t in cc.cr]
cc.train()
if flag_cr:
    # Build the global wasserstein module in the run's dtype before first use, without
    # WassersteinLoss.__init__ (it would always construct the 512-channel Vgg16).
    # `lowpass_2d_kernel` is a plain attribute: convert it by hand.
    import coolchic.training.metrics.wasserstein as wmod  # noqa: E402
    from einops import rearrange  # noqa: E402

    wl = wmod.WassersteinLoss.__new__(wmod.WassersteinLoss)
    torch.nn.Module.__init__(wl)
    wl.device = "cpu"
    wl.lowpass_2d_kernel = torch.outer(
        torch.tensor([0.25, 0.5, 0.25]), torch.tensor([0.25, 0.5, 0.25])
    ).view(1, 1, 3, 3)
    wl.target_features = None
    if config == "miniwass":
        # Same 23-slot structure as torchvision's (convs at {0,2,5,7,10,12,14,17,19,21},
        # pools at {4,9,16}, taps after {3,8,15,22}), tiny channels, seeded weights.
        mini_g = torch.Generator().manual_seed(4242)
        conv_pos = {0, 2, 5, 7, 10, 12, 14, 17, 19, 21}
        pool_pos = {4, 9, 16}
        mods = []
        conv_i = 0
        for slot in range(23):
            if slot in conv_pos:
                ci, co = MINI_VGG_DIMS[conv_i]
                conv_i += 1
                m = torch.nn.Conv2d(ci, co, 3, padding=1)
                with torch.no_grad():
                    m.weight.copy_(0.1 * torch.randn(m.weight.shape, generator=mini_g))
                    m.bias.copy_(0.1 * torch.randn(m.bias.shape, generator=mini_g))
                mods.append(m)
            elif slot in pool_pos:
                mods.append(torch.nn.MaxPool2d(2))
            else:
                mods.append(torch.nn.ReLU())

        class MiniWD(torch.nn.Module):
            def forward(self, x):
                out = []
                for i, lay in enumerate(self.features):
                    x = lay(x)
                    if i in (3, 8, 15, 22):
                        out.append(rearrange(x, "b c h w -> (b c) 1 h w", b=1))
                return out

        net = MiniWD()
        net.features = torch.nn.ModuleList(mods)
        wl.wd_net = net
    else:
        # Real VGG16 (torchvision IMAGENET1K_V1, downloaded once by torch.hub).
        wl.wd_net = wmod.Vgg16()
    wl.wd_net.to(dtype)
    wl.lowpass_2d_kernel = wl.lowpass_2d_kernel.to(dtype)
    # `torch.ones` builds `log2_sigma` in the default dtype even in an f64 run:
    # conv2d would complain about mixed dtypes. Cast inside the lowpass instead.
    import torch.nn.functional as F  # noqa: E402

    lp_kernel = wl.lowpass_2d_kernel
    wl.lowpass = lambda x, stride: F.conv2d(
        x.to(lp_kernel.dtype), lp_kernel, stride=stride, padding=1
    )
    wmod.wasserstein_loss_module = wl
import coolchic.component.core.quantizer as qmod  # noqa: E402

n_latent = sum(p.numel() for n, p in cc.named_parameters() if "latent_grids" in n)
uniform = torch.rand(n_latent).to(dtype)
qmod.torch.rand_like = lambda x, **kw: uniform.view_as(x).clone()
lmbda = 1e-3
noise_parameter = 2.0
temperature = torch.tensor(0.3, dtype=dtype)
out = cc.forward(
    quantizer_noise_type="kumaraswamy",
    quantizer_type="softround",
    soft_round_temperature=temperature,
    noise_parameter=torch.tensor(noise_parameter, dtype=dtype),
)
decoded = torch.clamp(out["raw_out"], 0.0, 1.0)
mse = ((decoded - img) ** 2).mean()
wd = torch.zeros((), dtype=dtype)
wass_taps = None
if "wasserstein" in dist_weight:
    from coolchic.training.metrics.wasserstein import wasserstein_fn  # noqa: E402

    wd = wasserstein_fn(decoded, img).reshape(())
    # Per-tap distances: localizes which VGG stage a port gets wrong.
    wl = wmod.wasserstein_loss_module
    dft = wl.wd_net(decoded)
    tft = wl.wd_net(img) if wl.target_features is None else wl.target_features
    wass_taps = []
    for fa, fb in zip(dft, tft):
        ls = torch.ones((1, 1, *fa.size()[-2:]), device=fa.device) * 3
        wass_taps.append(
            wl.wasserstein_distortion(fa, fb, ls, num_levels=5).item()
        )
    # The decoded picture itself, for comparing feature extractors directly.
    tensors_decoded = decoded.detach().clone()
    tensors_taps = [t.detach().clone() for t in dft]
    # d wasserstein / d decoded: full oracle for the metric's backward pass.
    tensors_grad_dec = torch.autograd.grad(wd, decoded, retain_graph=True)[0].detach().clone()
    # d wasserstein / d tap_i: isolates the stats gradient from the VGG chain. The
    # taps are leaves of a fresh graph since `dft` is not part of `wd`'s graph.
    tensors_grad_taps = []
    for fa, fb in zip(dft, tft):
        fa = fa.detach().clone().requires_grad_(True)
        ls = torch.ones((1, 1, *fa.size()[-2:]), device=fa.device) * 3
        di = wl.wasserstein_distortion(fa, fb, ls, num_levels=5)
        tensors_grad_taps.append(torch.autograd.grad(di, fa)[0].detach().clone())
rate_bits = out["rate"].sum()
n_pixels = h * w
if config == "yuv420":
    from coolchic.io.format.yuv import convert_444_to_420, yuv_dict_clamp  # noqa: E402
    from coolchic.training.loss import _compute_mse  # noqa: E402

    # Upstream's encode-side reconstruction tail: downsample the clamped output
    # to the {y,u,v} dict, then the per-plane-sample-weighted MSE.
    dec_d = yuv_dict_clamp(convert_444_to_420(out["raw_out"]), 0.0, 1.0)
    tgt_d = convert_444_to_420(img)
    mse = _compute_mse(dec_d, tgt_d).reshape(())
    flat_target = torch.cat(
        [tgt_d["y"].reshape(-1), tgt_d["u"].reshape(-1), tgt_d["v"].reshape(-1)]
    )
else:
    flat_target = img.reshape(-1)
dist = dist_weight["mse"] * mse + dist_weight.get("wasserstein", 0.0) * wd
loss = dist + lmbda * rate_bits / n_pixels
loss.backward()

tensors = [("target", flat_target), ("uniform_noise", uniform)]
names = []
for name, p in cc.named_parameters():
    names.append(name)
    tensors.append((name, p.detach()))
for name, p in cc.named_parameters():
    g = p.grad if p.grad is not None else torch.zeros_like(p)
    tensors.append(("grad:" + name, g))
if config == "miniwass":
    # Flat conv weights+biases of the miniature stack, in MINI_VGG_DIMS order
    # (same layout as `scripts/oracle/dump_vgg.py` produces for the real VGG16).
    flat = torch.cat(
        [
            torch.cat([m.weight.reshape(-1), m.bias])
            for m in wl.wd_net.features
            if isinstance(m, torch.nn.Conv2d)
        ]
    )
    tensors.append(("vgg", flat))
if wass_taps is not None:
    tensors.append(("decoded", tensors_decoded))
    tensors.append(("grad_dec", tensors_grad_dec))
    for i, t in enumerate(tensors_taps):
        tensors.append((f"tap{i}", t))
        tensors.append((f"gtap{i}", tensors_grad_taps[i]))
index = []
with open(sys.argv[3] + ".bin", "wb") as f:
    for name, t in tensors:
        a = t.detach().cpu().numpy().astype("<f4")
        index.append({"name": name, "shape": list(a.shape)})
        f.write(a.tobytes())
json.dump(
    {
        "config": config,
        "img_size": [h, w],
        "layers_synthesis": layers,
        "latent_resolution": list(lat),
        "hyperlatent_resolution": list(hyper),
        "ifce_resolution": [0, 2],
        **arm,
        "lmbda": lmbda,
        "softround_temperature": 0.3,
        "noise_parameter": noise_parameter,
        "flag_common_randomness": flag_cr,
        "frame_data_type": "yuv420" if config == "yuv420" else "rgb",
        "dist_weight": dist_weight,
        "vgg_dims": MINI_VGG_DIMS if config == "miniwass" else None,
        "loss": loss.item(),
        "mse": mse.item(),
        "wasserstein": wd.item(),
        "wasserstein_per_tap": wass_taps,
        "rate_bits": rate_bits.item(),
        "params": names,
        "tensors": index,
        "torch": torch.__version__,
    },
    open(sys.argv[3] + ".json", "w"),
    indent=1,
)
print("ok", sys.argv[3], "loss", loss.item(), "mse", mse.item(), "rate", rate_bits.item())
