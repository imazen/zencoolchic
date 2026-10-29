"""Video oracle: two modes.

``--decode`` decodes an existing upstream-encoded multi-frame stream with
upstream's ``decode_video`` and dumps the container plus the decoded planes.
``--assemble`` concatenates single-frame .cool files into an all-intra
multi-frame stream with upstream's own header classes, then dumps the same way.

Usage (run with the upstream venv's python, -I):
    python -I scripts/oracle/dump_video.py <Cool-Chic checkout> --assemble <out_prefix> <in.cool>...
    python -I scripts/oracle/dump_video.py <Cool-Chic checkout> --decode <out_prefix> <in.cool>

Outputs per case: ``<out_prefix>.cool`` (the stream), ``<out_prefix>.json``
(frame metadata), ``<out_prefix>.<i>.planes.u16`` (decoded samples of frame i
in display order, plane after plane, little-endian u16), and the raw Cool-chic
synthesis outputs ``<out_prefix>.<i>.residue.f32`` (always) and
``<out_prefix>.<i>.motion.f32`` (inter frames only), little-endian f32 in
``[c][h][w]`` order. The residue dump keeps every channel: RGB/YUV at 0:3,
alpha at 3, beta at 4 for B frames.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, sys.argv[1])

import numpy as np  # noqa: E402
import torch  # noqa: E402

import coolchic.bitstream.decode as cc_decode  # noqa: E402
from coolchic.bitstream.decode import decode_video  # noqa: E402
from coolchic.bitstream.header.header import FrameHeader, VideoHeader  # noqa: E402
from coolchic.utils.codingstructure import CodingStructure  # noqa: E402

mode = sys.argv[2]
prefix = sys.argv[3]


def dump_decoded(bitstream: bytes, prefix: str) -> None:
    """Decode ``bitstream`` with upstream, writing planes, per-cc raw outputs
    and frame metadata to ``prefix.*``. Raw Cool-chic outputs are captured by
    wrapping ``encode_decode_coolchic``; per frame (display order) the first
    call is the residue, the second the motion network."""
    cc_raws: dict[int, list[np.ndarray]] = {}
    meta_by_display: dict[int, dict] = {}

    orig_frame = cc_decode.decode_frame
    orig_cc = cc_decode.encode_decode_coolchic

    def wrapped_frame(bitstream_bytes, reference_frames=None, verbosity=0):
        fh = FrameHeader()
        fh.read_header(bitstream_bytes)
        disp = fh.get_value("display_index")
        calls = []

        def capture(*args, **kwargs):
            raw, rest = orig_cc(*args, **kwargs)
            calls.append(raw.detach().cpu().numpy().astype("<f4"))
            return raw, rest

        cc_decode.encode_decode_coolchic = capture
        try:
            out = orig_frame(bitstream_bytes, reference_frames, verbosity)
        finally:
            cc_decode.encode_decode_coolchic = orig_cc
        cc_raws[disp] = calls
        meta_by_display[disp] = {
            "display_index": disp,
            "frame_type": fh.get_value("frame_type"),
            "frame_data_type": fh.get_value("frame_data_type"),
            "bitdepth": fh.get_value("bitdepth"),
            "index_references": fh.get_value("index_references"),
            "global_flow": fh.get_value("global_flow"),
            "warp_filter_size": fh.get_value("warp_filter_size"),
        }
        return out

    cc_decode.decode_frame = wrapped_frame
    try:
        with tempfile.NamedTemporaryFile(suffix=".cool", delete=False) as f:
            f.write(bitstream)
            path = f.name
        try:
            decoded = decode_video(path)
        finally:
            os.unlink(path)
    finally:
        cc_decode.decode_frame = orig_frame

    n_frames = len(decoded)
    meta = []
    for i in range(n_frames):
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
        raws = cc_raws[i]
        raws[0].tofile(f"{prefix}.{i}.residue.f32")
        raw_meta = {"residue_shape": list(raws[0].shape)}
        if len(raws) > 1:
            raws[1].tofile(f"{prefix}.{i}.motion.f32")
            raw_meta["motion_shape"] = list(raws[1].shape)
        entry = dict(meta_by_display[i])
        entry.update(
            {
                "plane_shapes": [list(s.shape) for s in samples],
                **raw_meta,
            }
        )
        meta.append(entry)
    json.dump(
        {"n_frames": n_frames, "frames": meta, "torch": torch.__version__},
        open(prefix + ".json", "w"),
        indent=1,
    )
    print("ok", prefix, n_frames, "frames")


def assemble(prefix: str, inputs) -> None:
    """Build an all-intra multi-frame stream from single-frame .cool files."""
    streams = [open(p, "rb").read() for p in inputs]
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
    with open(prefix + ".cool", "wb") as f:
        f.write(bitstream)
    return bitstream


if mode == "--decode":
    bitstream = open(sys.argv[4], "rb").read()
elif mode == "--assemble":
    bitstream = assemble(prefix, sys.argv[4:])
else:
    raise SystemExit(f"unknown mode {mode}")
dump_decoded(bitstream, prefix)
