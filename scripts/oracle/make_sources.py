"""Restore the exact inputs used for the frozen upstream oracle.

Usage: python make_sources.py NEW_DIRECTORY
No Pillow, numpy, image download or color conversion is needed for restoration.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[2] / "tests/fixtures/sources"
    target = Path(sys.argv[1])
    assert not target.exists(), "rename the previous source directory before replacing it"
    manifest = json.loads((source / "manifest.json").read_text())
    assert len(manifest["files"]) == 6
    for entry in manifest["files"]:
        data = (source / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"], entry["path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    target.mkdir(parents=True)
    for entry in manifest["files"]:
        shutil.copyfile(source / entry["path"], target / entry["path"])
    print(f"Restored {len(manifest['files'])} source inputs to {target}")


if __name__ == "__main__":
    main()
