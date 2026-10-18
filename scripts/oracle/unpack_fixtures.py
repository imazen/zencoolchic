"""Restore frozen references for maintenance, preserving their original bytes.

Usage: python unpack_fixtures.py PACKAGED_ROOT NEW_RAW_DIRECTORY
Requires a zstd executable. Rust regression tests do not use this script.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    source, target = map(Path, sys.argv[1:])
    assert not target.exists(), "rename the previous raw directory before replacing it"
    manifest = json.loads((source / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    target.mkdir(parents=True)
    for entry in manifest["files"]:
        path = source / (entry["path"] + ".zst")
        frame = path.read_bytes()
        assert len(frame) == entry["compressed_bytes"], str(path)
        assert hashlib.sha256(frame).hexdigest() == entry["compressed_sha256"], str(path)
        packed = subprocess.run(["zstd", "-q", "-d", "-c", str(path)],
                                capture_output=True, check=True).stdout
        assert len(packed) == entry["bytes"], str(path)
        word = entry["shuffle_word_bytes"]
        assert word in [1, 2, 4, 8] and len(packed) % word == 0
        count = len(packed) // word
        data = bytearray(len(packed))
        for b in range(word):
            data[b::word] = packed[b * count : (b + 1) * count]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], str(path)
        dest = target / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    shutil.copyfile(source / "manifest.json", target / "manifest.json")
    print(f"Restored and SHA-256 verified {len(manifest['files'])} original references")


if __name__ == "__main__":
    main()
