"""Decode a Cool-chic bitstream with upstream Python and dump intermediates for parity tests.

Usage (from anywhere; run with the upstream venv's python, -I):
    python -I scripts/oracle/dump_decode.py <Cool-Chic checkout> <in.cool> <out_prefix>

Writes:
    <out_prefix>.json         dims and per-grid shapes
    <out_prefix>.latents.i64  decoded integer latents, finest grid first, little-endian i64
    <out_prefix>.raw.f32      synthesis output before quantization, [c][h][w] little-endian f32
    <out_prefix>.planes.u16   final samples, plane after plane, little-endian u16
"""
import json
import sys

sys.path.insert(0, sys.argv[1])

import numpy as np  # noqa: E402
import torch  # noqa: E402

import coolchic.bitstream.component.coolchic as ccmod  # noqa: E402
from coolchic.bitstream.decode import decode_video  # noqa: E402

captured = {}
orig_latent = ccmod.entropy_coding_latent_arm


def spy_latent(*args, **kwargs):
    out = orig_latent(*args, **kwargs)
    captured.setdefault("latents", []).insert(0, out.clone())
    return out


ccmod.entropy_coding_latent_arm = spy_latent
orig_cc = ccmod.encode_decode_coolchic


def spy_cc(*args, **kwargs):
    out, b = orig_cc(*args, **kwargs)
    captured["raw"] = out.clone()
    return out, b


import coolchic.bitstream.decode as decmod  # noqa: E402

decmod.encode_decode_coolchic = spy_cc

frames = decode_video(sys.argv[2])
prefix = sys.argv[3]
frame = frames["0"]
lat = captured["latents"]
np.concatenate([l.reshape(-1).numpy().astype("<i8") for l in lat]).tofile(prefix + ".latents.i64")
raw = captured["raw"][0].numpy().astype("<f4")
raw.tofile(prefix + ".raw.f32")
maxv = 2 ** frame.bitdepth - 1
if isinstance(frame.data, dict):
    planes = [frame.data[k][0, 0] for k in ("y", "u", "v")]
else:
    planes = [frame.data[0, c] for c in range(frame.data.shape[1])]
samples = [torch.round(p * maxv).numpy().astype("<u2") for p in planes]
np.concatenate([s.reshape(-1) for s in samples]).tofile(prefix + ".planes.u16")
json.dump(
    {
        "latent_shapes": [list(l.shape[-2:]) for l in lat],
        "raw_shape": list(raw.shape),
        "plane_shapes": [list(s.shape) for s in samples],
        "bitdepth": frame.bitdepth,
        "frame_data_type": frame.frame_data_type,
        "torch": torch.__version__,
    },
    open(prefix + ".json", "w"),
    indent=1,
)
print("ok", prefix)
