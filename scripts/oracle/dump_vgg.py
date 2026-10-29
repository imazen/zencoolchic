"""Dump the VGG16 feature weights used by upstream's WassersteinLoss.

Usage: python -I scripts/oracle/dump_vgg.py <out.bin>

Writes a flat little-endian f32 file: for each of the ten 3x3 convolutions of
`torchvision vgg16.features[:23]` (indices 0,2,5,7,10,12,14,17,19,21, in order),
the `[c_out][c_in][3][3]` weights followed by the `[c_out]` biases. That is the
layout `zencoolchic::unstable::Vgg16::from_le_bytes` parses (7,635,264 floats).

The weights (torchvision VGG16_Weights.IMAGENET1K_V1) are torchvision's, not
Cool-chic's; keep the dump out of the repo unless the licence question is settled.
"""

import sys

import torch  # noqa: E402
import torchvision  # noqa: E402

out = sys.argv[1]
features = torchvision.models.vgg16(
    weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
).features[:23]
convs = [m for m in features if isinstance(m, torch.nn.Conv2d)]
assert len(convs) == 10, f"expected 10 convs in vgg16.features[:23], got {len(convs)}"
n = 0
with open(out, "wb") as f:
    for m in convs:
        w = m.weight.detach().cpu().contiguous().numpy().astype("<f4")
        b = m.bias.detach().cpu().numpy().astype("<f4")
        n += w.size + b.size
        f.write(w.tobytes())
        f.write(b.tobytes())
assert n == 7_635_264, n
print("ok", out, n, "floats")
