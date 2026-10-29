"""Dump upstream warper reference vectors for the Rust warper parity test.

Usage (from anywhere; run with the upstream venv's python, -I):
    python -I scripts/oracle/dump_warp.py <Cool-Chic checkout> <out_prefix>

Writes:
    <out_prefix>.json            {"H":, "W":, "C":}
    <out_prefix>.x.f32           input image [C][H][W] little-endian f32
    <out_prefix>.flow.f32        flow field [2][H][W] (channel 0 = horizontal/x)
    <out_prefix>.out{2,4,8,12}.f32  upstream Warper output per filter size
    <out_prefix>.shift.f32       nearest-neighbour integer translation, dx=3 dy=-2
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
# Integer global translation: constant flow field, dx=3, dy=-2.
shift_flow = torch.zeros(1, 2, H, W)
shift_flow[:, 0] = 3.0
shift_flow[:, 1] = -2.0
with torch.no_grad():
    out["shift"] = vanilla_warp_fn(x, shift_flow, mode="nearest")[0]

for name, tensor in out.items():
    tensor.contiguous().numpy().astype("<f4").tofile(f"{prefix}.{name}.f32")
with open(f"{prefix}.json", "w") as f:
    json.dump({"H": H, "W": W, "C": C}, f)
print(f"ok {prefix}")
