#!/usr/bin/env python3
"""Validate local Parquet data and publish a deterministic index (CPU only)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import pyarrow.parquet as pq

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("data_dir", type=Path)
p.add_argument("--verify-sha256", action="store_true")
a = p.parse_args()
root = a.data_dir.resolve(strict=True)
manifest = json.loads((root / "subset_manifest.json").read_text())
entries = sorted(manifest["files"], key=lambda item: item["path"])
actual = sorted(str(f.relative_to(root)) for f in root.rglob("*.parquet"))
assert actual == [e["path"] for e in entries], "Manifest and local Parquet files differ"
paths, lengths = [], []
for e in entries:
    f = (root / e["path"]).resolve(strict=True)
    assert f.is_relative_to(root), f"File outside dataset: {f}"
    assert f.stat().st_size == e["size_bytes"], f"Size mismatch: {f}"
    rows = pq.ParquetFile(f).metadata.num_rows
    assert rows == e["num_rows"], f"Row count mismatch: {f}"
    if a.verify_sha256:
        h = hashlib.sha256()
        with f.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        assert h.hexdigest() == e["sha256"], f"SHA256 mismatch: {f}"
    paths.append(str(f))
    lengths.append(rows)
assert sum(lengths) == manifest["num_samples"]
cache = root / "map_style_cache" / "file_info.pkl"
cache.parent.mkdir(parents=True, exist_ok=True)
payload = pickle.dumps((tuple(paths), tuple(lengths)), protocol=4)
fd, name = tempfile.mkstemp(prefix="file_info.", dir=cache.parent)
try:
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(name, cache)
finally:
    if os.path.exists(name):
        os.unlink(name)
print(json.dumps({"status": "PASS", "files": len(paths), "rows": sum(lengths),
                  "verified_sha256": a.verify_sha256, "cache": str(cache),
                  "cache_sha256": hashlib.sha256(payload).hexdigest()}, sort_keys=True), flush=True)
