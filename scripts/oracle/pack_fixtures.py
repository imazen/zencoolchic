"""Pack a frozen-fixture layout with manifest.json into portable zstd files.

Usage: python pack_fixtures.py INPUT_ROOT NEW_OUTPUT_ROOT
Requires zstd 1.5.7 for regeneration; tests use a pure-Rust decoder.
Input files and their hashes are retained unchanged in the output manifest.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def word_bytes(name):
    if name.endswith(".latents.i64"):
        return 8
    if name.endswith((".raw.f32", ".bin")):
        return 4
    if name.endswith(".planes.u16"):
        return 2
    return 1


def main():
    source, dest = map(Path, sys.argv[1:])
    assert not dest.exists(), "rename the previous fixture set before replacing it"
    manifest = json.loads((source / "manifest.json").read_text())
    version = subprocess.check_output(["zstd", "--version"], text=True).strip()
    assert "v1.5.7" in version, f"expected zstd 1.5.7, got {version}"
    dest.mkdir(parents=True)
    for entry in manifest["files"]:
        name = entry["path"]
        data = (source / name).read_bytes()
        assert len(data) == entry["bytes"], name
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], name
        word = word_bytes(name)
        assert len(data) % word == 0, name
        packed = b"".join(data[b::word] for b in range(word))
        frame = subprocess.run(
            ["zstd", "-q", "-19", "-T1", "-c"],
            input=packed, capture_output=True, check=True,
        ).stdout
        back = subprocess.run(
            ["zstd", "-q", "-d", "-c"],
            input=frame, capture_output=True, check=True,
        ).stdout
        assert back == packed, name
        target = dest / (name + ".zst")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(frame)
        entry["shuffle_word_bytes"] = word
        entry["compressed_bytes"] = len(frame)
        entry["compressed_sha256"] = hashlib.sha256(frame).hexdigest()
    manifest["schema_version"] = 1
    manifest["soap_cases"] = ["soap"]
    manifest["compression"] = {
        "format": "zstd", "level": 19, "encoder": version,
        "layout": "byte planes, least-significant byte first, whole file",
        "dictionary": None,
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Packed {len(manifest['files'])} fixtures: "
          f"{sum(e['compressed_bytes'] for e in manifest['files'])} compressed bytes")


if __name__ == "__main__":
    main()
