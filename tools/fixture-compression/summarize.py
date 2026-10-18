"""Summarize an exact-storage run and independently decode its PNGs with Pillow.

Usage: python summarize.py DECODER_DIR ENCODED_DIR RUN_LOG OUTPUT_CSV
Pillow is needed only for this independent measurement check, not regression tests.
"""

import csv
import gzip
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


def main():
    fixture, encoded, log, output_csv = map(Path, sys.argv[1:])
    lines = [line for line in log.read_text().splitlines() if line.count(",") == 7]
    rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
    assert rows, "missing measured rows"
    assert "fixture-compression: all inputs verified" in log.read_text(), "measurement did not finish"
    expected = {}
    for path in sorted(fixture.glob("*.json")):
        meta = json.loads(path.read_text())
        for kind, key, sample_bytes in [
            ("latents.i64", "latent_shapes", 8),
            ("planes.u16", "plane_shapes", 2),
            ("raw.f32", "raw_shape", 4),
        ]:
            data = (fixture / f"{path.stem}.{kind}").read_bytes()
            shapes = meta[key]
            if kind == "raw.f32":
                shapes = [shapes[1:]] * shapes[0]
            offset = 0
            for part, (height, width) in enumerate(shapes):
                size = width * height * sample_bytes
                key_part = (path.stem, kind, str(part))
                expected[key_part] = (data[offset : offset + size], sample_bytes)
                offset += size
            assert offset == len(data)
    assert len(rows) == len(expected)
    seen = set()
    totals = defaultdict(lambda: defaultdict(int))
    for row in rows:
        key = (row["case"], row["kind"], row["part"])
        assert key not in seen
        seen.add(key)
        raw, sample_bytes = expected[key]
        assert len(raw) == int(row["raw_bytes"])
        image = Image.open(encoded / (".".join(key) + ".png"))
        width, height = int(row["width"]), int(row["height"])
        assert image.size == (width, height * (sample_bytes if sample_bytes != 2 else 1))
        if sample_bytes == 2:
            back = b"".join(int(v).to_bytes(2, "little") for v in image.get_flattened_data())
        else:
            packed = image.tobytes()
            back = bytearray(len(raw))
            for b in range(sample_bytes):
                back[b::sample_bytes] = packed[b * width * height : (b + 1) * width * height]
        assert back == raw, f"Pillow mismatch: {key}"
        for field in ["raw_bytes", "png_bytes", "imij_bytes"]:
            totals[row["kind"]][field] += int(row[field])
    assert seen == expected.keys()
    for kind in totals:
        files = sorted(fixture.glob(f"*.{kind}"))
        assert len(files) == 12
        totals[kind]["gzip9_bytes"] = sum(
            len(gzip.compress(p.read_bytes(), compresslevel=9, mtime=0)) for p in files
        )
    with output_csv.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"parts_verified": len(rows), "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
