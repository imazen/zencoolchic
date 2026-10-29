"""Dump upstream inter-frame compositing references for the Rust objective test.

Usage (from anywhere; run with the upstream venv's python, -I):
    python -I scripts/oracle/dump_inter.py <Cool-Chic checkout> <out_prefix>

Replicates `FrameEncoder.forward`'s P/B tail on seeded tensors with upstream's
own `Warper` in training mode (unquantized flow): alpha/beta reparam,
beta-blended bidirectional prediction, masked prediction plus residue, then a
flat MSE on the clamped picture. Dumps the composited picture and
`d loss / d raw` for both the residue and motion models.

Writes:
    <out_prefix>.json         {"H":, "W":, "warp_filter_size":, "mse":}
    <out_prefix>.res.f32      residue raw output [5][H][W] (res3|alpha|beta)
    <out_prefix>.mot.f32      motion raw output [4][H][W] (f1x|f1y|f2x|f2y)
    <out_prefix>.ref{0,1}.f32 shifted references [3][H][W]
    <out_prefix>.tgt.f32      target picture [3][H][W]
    <out_prefix>.img.f32      composited decoded picture [3][H][W]
    <out_prefix>.dres.f32     d mse / d residue raw [5][H][W]
    <out_prefix>.dmot.f32     d mse / d motion raw [4][H][W]
"""
import json
import sys

sys.path.insert(0, sys.argv[1])

import torch  # noqa: E402

from coolchic.component.intercoding.warp import Warper, WarpParameter  # noqa: E402

prefix = sys.argv[2]
torch.manual_seed(2024)
torch.use_deterministic_algorithms(True)
H, W, FS = 20, 24, 8

ref0 = torch.rand(1, 3, H, W)
ref1 = torch.rand(1, 3, H, W)
tgt = torch.rand(1, 3, H, W)
# Residue raw: three residue channels then raw alpha/beta (pre-reparam).
res_raw = (torch.randn(1, 5, H, W) * 0.5).requires_grad_(True)
# Motion raw: two (x, y) flow fields in pixel units.
mot_raw = (torch.randn(1, 4, H, W) * 1.5).requires_grad_(True)

warper = Warper(WarpParameter(filter_size=FS), (H, W)).train()
alpha = torch.clamp(res_raw[:, 3:4] + 0.5, 0.0, 1.0)
beta = torch.clamp(res_raw[:, 4:5] + 0.5, 0.0, 1.0)
w0 = warper(ref0, mot_raw[:, 0:2])
w1 = warper(ref1, mot_raw[:, 2:4])
pred = beta * w0 + (1.0 - beta) * w1
img = alpha * pred + res_raw[:, 0:3]
dec = torch.clamp(img, 0.0, 1.0)
mse = ((dec - tgt) ** 2).mean()
mse.backward()

out = {
    "res": res_raw[0],
    "mot": mot_raw[0],
    "ref0": ref0[0],
    "ref1": ref1[0],
    "tgt": tgt[0],
    "img": img[0],
    "dres": res_raw.grad[0],
    "dmot": mot_raw.grad[0],
}
for name, tensor in out.items():
    tensor.detach().contiguous().numpy().astype("<f4").tofile(f"{prefix}.{name}.f32")
with open(f"{prefix}.json", "w") as f:
    json.dump({"H": H, "W": W, "warp_filter_size": FS, "frame_type": "B", "mse": mse.item()}, f)
print(f"ok {prefix} mse {mse.item()}")

# Optional P-frame variant (single reference, residue 4ch, motion 2ch):
#   python -I dump_inter.py <checkout> comp comp_p
if len(sys.argv) > 3:
    prefix = sys.argv[3]
    res_raw = (torch.randn(1, 4, H, W) * 0.5).requires_grad_(True)
    mot_raw = (torch.randn(1, 2, H, W) * 1.5).requires_grad_(True)
    alpha = torch.clamp(res_raw[:, 3:4] + 0.5, 0.0, 1.0)
    w0 = warper(ref0, mot_raw[:, 0:2])
    img = alpha * w0 + res_raw[:, 0:3]
    dec = torch.clamp(img, 0.0, 1.0)
    mse = ((dec - tgt) ** 2).mean()
    mse.backward()
    out = {
        "res": res_raw[0],
        "mot": mot_raw[0],
        "ref0": ref0[0],
        "tgt": tgt[0],
        "img": img[0],
        "dres": res_raw.grad[0],
        "dmot": mot_raw.grad[0],
    }
    for name, tensor in out.items():
        tensor.detach().contiguous().numpy().astype("<f4").tofile(f"{prefix}.{name}.f32")
    with open(f"{prefix}.json", "w") as f:
        json.dump({"H": H, "W": W, "warp_filter_size": FS, "frame_type": "P", "mse": mse.item()}, f)
    print(f"ok {prefix} mse {mse.item()}")
