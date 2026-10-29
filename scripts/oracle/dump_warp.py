"""Dump upstream warper reference vectors for the Rust warper parity test.

Usage (from anywhere; run with the upstream venv's python, -I):
    python -I scripts/oracle/dump_warp.py <Cool-Chic checkout> <out_prefix>

Writes:
    <out_prefix>.json            {"H":, "W":, "C":, "global_translation": [dx, dy],
                                "global_translation2": [dx, dy]}
    <out_prefix>.x.f32           input image [C][H][W] little-endian f32
    <out_prefix>.flow.f32        flow field [2][H][W] (channel 0 = horizontal/x)
    <out_prefix>.out{2,4,8,12}.f32  upstream Warper output per filter size
    <out_prefix>.dout.f32        seeded upstream gradient for the backward checks
    <out_prefix>.dflow{2,4,8,12}.f32  autograd d(warp)/d(flow) per filter size
    <out_prefix>.shift.f32       nearest-neighbour integer translation, dx=3 dy=-2
    <out_prefix>.x2.f32          structured (smooth) second image
    <out_prefix>.target{,2}.f32  vanilla_warp(x, (9,-6)) translation targets
"""
import json
import sys

sys.path.insert(0, sys.argv[1])

import torch  # noqa: E402

from coolchic.component.intercoding.warp import (  # noqa: E402
    Warper,
    WarpParameter,
    vanilla_warp_fn,
)
from coolchic.component.intercoding.globalmotion import (  # noqa: E402
    get_global_translation,
)
from coolchic.io.framedata import FrameData  # noqa: E402

prefix = sys.argv[2]
torch.manual_seed(777)
H, W, C = 20, 24, 3
x = torch.rand(1, C, H, W)
flow = torch.randn(1, 2, H, W) * 1.7
out = {
    "x": x[0],
    "flow": flow[0],
}
for fs in (2, 4, 8, 12):
    warper = Warper(WarpParameter(filter_size=fs), (H, W)).eval()
    with torch.no_grad():
        out[f"out{fs}"] = warper(x, flow)[0]
# Training-mode flow gradient: autograd through the unquantized warp.
dout = torch.randn(1, C, H, W)
out["dout"] = dout[0]
for fs in (2, 4, 8, 12):
    warper = Warper(WarpParameter(filter_size=fs), (H, W)).train()
    f2 = flow.clone().requires_grad_(True)
    warper(x, f2).backward(dout)
    out[f"dflow{fs}"] = f2.grad[0]
# Integer global translation: constant flow field, dx=3, dy=-2.
shift_flow = torch.zeros(1, 2, H, W)
shift_flow[:, 0] = 3.0
shift_flow[:, 1] = -2.0
with torch.no_grad():
    out["shift"] = vanilla_warp_fn(x, shift_flow, mode="nearest")[0]

# Global-translation search: a second image related to `x` by a real
# translation (plus noise so the MSE surface is not degenerate).
tflow = torch.zeros(1, 2, H, W)
tflow[:, 0] = 9.0
tflow[:, 1] = -6.0
with torch.no_grad():
    target = vanilla_warp_fn(x, tflow, mode="nearest") + 0.02 * torch.randn(
        1, C, H, W
    )
out["target"] = target[0]
frame = FrameData(bitdepth=8, frame_data_type="rgb", data=target)
ref = FrameData(bitdepth=8, frame_data_type="rgb", data=x)
_, gflow = get_global_translation(frame, [ref])
gt = [int(gflow[0][0].item()), int(gflow[0][1].item())]

# Second pair on a structured (smooth) image — the search landscape is
# convex-ish so the coarse-to-fine ladder matters, unlike pure noise.
gy, gx = torch.meshgrid(
    torch.arange(H, dtype=torch.float32),
    torch.arange(W, dtype=torch.float32),
    indexing="ij",
)
x2 = (0.4 + 0.6 * torch.exp(-(((gx - 12) ** 2 + (gy - 8) ** 2) / 30.0))).repeat(
    1, C, 1, 1
)
out["x2"] = x2[0]
with torch.no_grad():
    target2 = vanilla_warp_fn(x2, tflow, mode="nearest")
out["target2"] = target2[0]
frame2 = FrameData(bitdepth=8, frame_data_type="rgb", data=target2)
ref2 = FrameData(bitdepth=8, frame_data_type="rgb", data=x2)
_, gflow2 = get_global_translation(frame2, [ref2])
gt2 = [int(gflow2[0][0].item()), int(gflow2[0][1].item())]

for name, tensor in out.items():
    tensor.contiguous().numpy().astype("<f4").tofile(f"{prefix}.{name}.f32")
with open(f"{prefix}.json", "w") as f:
    json.dump(
        {
            "H": H,
            "W": W,
            "C": C,
            "global_translation": gt,
            "global_translation2": gt2,
        },
        f,
    )
print(f"ok {prefix}")
