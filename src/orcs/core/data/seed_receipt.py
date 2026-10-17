"""Source-content receipts for stair seeds; paths may move between machines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def source_digest(sample: Path) -> str:
    h = hashlib.sha256()
    for path in (sample / "smpl_motion.npz", sample / "metadata.json", sample.parent / "tile.json"):
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def write_seed_receipt(sample: Path) -> None:
    if not (sample / "smpl_motion.npz").exists():
        return
    if not (sample.parent / "tile.json").exists():
        return
    receipt = {"source_sha256": source_digest(sample)}
    (sample / "seed_source.json").write_text(json.dumps(receipt) + "\n")


def seed_matches_source(sample: Path) -> bool:
    try:
        receipt = json.loads((sample / "seed_source.json").read_text())
        return receipt["source_sha256"] == source_digest(sample)
    except (OSError, ValueError, KeyError):
        return False
