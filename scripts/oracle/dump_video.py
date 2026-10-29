"""Video oracle: assemble a multi-intra-frame stream from single-frame .cool files
with upstream's own header classes, decode it with upstream's decode_video, and
dump the container plus the decoded planes.

Usage (run with the upstream venv's python, -I):
    python -I scripts/oracle/dump_video.py <Cool-Chic checkout> <out_prefix> <in.cool>...

Each input must be a single-frame stream. Frame i gets display_index i and the
container declares an all-intra structure (intra_pos = 0..n-1). The assembled
stream is written to <out_prefix>.cool, and every decoded frame's samples to
<out_prefix>.<i>.planes.u16 (plane after plane, little-endian u16), with
<out_prefix>.json describing the frames.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, sys.argv[1])

import numpy as np  # noqa: E402
import torch  # noqa: E402

from coolchic.bitstream.decode import decode_video  # noqa: E402
from coolchic.bitstream.header.header import FrameHeader, VideoHeader  # noqa: E402
from coolchic.utils.codingstructure import CodingStructure  # noqa: E402

prefix = sys.argv[2]
streams = [open(p, "rb").read() for p in sys.argv[3:]]

frame_bytes = []
for i, data in enumerate(streams):
    vh = VideoHeader()
    rest = vh.read_header(data)
    assert vh.get_value("n_frames") == 1, "input must be a single-frame stream"
    fh = FrameHeader()
    payload = fh.read_header(rest)
    assert fh.get_value("frame_type") == "I"
    fh.set_value("display_index", i)
    frame_bytes.append(fh.to_bytes() + payload)

cs = CodingStructure(
    n_frames=len(streams), intra_pos=list(range(len(streams))), p_pos=[]
)
vh = VideoHeader()
vh.set_header(cs)
bitstream = vh.to_bytes() + b"".join(frame_bytes)

with tempfile.NamedTemporaryFile(suffix=".cool", delete=False) as f:
    f.write(bitstream)
    path = f.name
try:
    decoded = decode_video(path)
finally:
    os.unlink(path)

with open(prefix + ".cool", "wb") as f:
    f.write(bitstream)

meta = []
for i in range(len(streams)):
    frame = decoded[str(i)]
    maxv = 2**frame.bitdepth - 1
    if isinstance(frame.data, dict):
        planes = [frame.data[k][0, 0] for k in ("y", "u", "v")]
    else:
        planes = [frame.data[0, c] for c in range(frame.data.shape[1])]
    samples = [torch.round(p * maxv).numpy().astype("<u2") for p in planes]
    np.concatenate([s.reshape(-1) for s in samples]).tofile(
        f"{prefix}.{i}.planes.u16"
    )
    meta.append(
        {
            "plane_shapes": [list(s.shape) for s in samples],
            "bitdepth": frame.bitdepth,
            "frame_data_type": frame.frame_data_type,
        }
    )
json.dump(
    {"n_frames": len(streams), "frames": meta, "torch": torch.__version__},
    open(prefix + ".json", "w"),
    indent=1,
)
print("ok", prefix, len(streams), "frames")
