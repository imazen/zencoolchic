"""Gradient oracle: one deterministic training forward + backward of upstream's CoolChicEncoder.

Usage: python -I scripts/oracle/dump_grad.py <Cool-Chic checkout> <image.png> <out_prefix> [config]
config: "hop" (default) or "small" (tiny networks, fast)

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
if config == "hop":
    layers = ["48-1-linear-relu", "3-1-linear-none", "3-3-residual-relu", "3-3-residual-none"]
    arm = dict(spatial_context_arm=14, n_hidden_layers_arm=2, output_feature_ifce=6)
    hyper = (4, 6)
    lat = (0, 6)
else:
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
    flag_common_randomness=False,
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
cc.train()
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
rate_bits = out["rate"].sum()
n_pixels = h * w
loss = mse + lmbda * rate_bits / n_pixels
loss.backward()

tensors = [("target", img), ("uniform_noise", uniform)]
names = []
for name, p in cc.named_parameters():
    names.append(name)
    tensors.append((name, p.detach()))
for name, p in cc.named_parameters():
    g = p.grad if p.grad is not None else torch.zeros_like(p)
    tensors.append(("grad:" + name, g))
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
        "loss": loss.item(),
        "mse": mse.item(),
        "rate_bits": rate_bits.item(),
        "params": names,
        "tensors": index,
        "torch": torch.__version__,
    },
    open(sys.argv[3] + ".json", "w"),
    indent=1,
)
print("ok", sys.argv[3], "loss", loss.item(), "mse", mse.item(), "rate", rate_bits.item())
