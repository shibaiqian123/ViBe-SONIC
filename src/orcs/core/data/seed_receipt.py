"""Source-content receipts for stair seeds; paths may move between machines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def source_digest(sample: Path) -> str:
    h = hashlib.sha256()
    paths = [sample / "smpl_motion.npz", sample / "metadata.json", sample.parent / "tile.json"]
    contact = sample / "stair_contact_targets.npz"
    if contact.exists():
        paths.append(contact)
    for path in paths:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def write_seed_receipt(sample: Path, version: str | None = None, identity: str | None = None) -> None:
    if not (sample / "smpl_motion.npz").exists():
        return
    if not (sample.parent / "tile.json").exists():
        return
    receipt = {"source_sha256": source_digest(sample)}
    destination = sample / "seed_source.json"
    if version is not None:
        from orcs.core.sonic.runtime import model_info
        from orcs.core.sonic.specs import CONTRACT_VERSION, seed_path
        _, model_meta = model_info(version, "smpl")
        if identity != model_meta['contract_id']:
            raise ValueError("seed generator used a different SONIC model contract")
        from orcs.core.data.seed_identity import stair_seed_identity
        receipt.update(stair_seed_identity(sample))
        receipt.update(sonic_version=version, contract_id=identity,
                       base_sha256=model_meta['source_sha256'], observation_version=CONTRACT_VERSION)
        destination = seed_path(sample, version).with_name('seed_source.json')
        destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(receipt) + "\n")


def seed_matches_source(sample: Path, version: str | None = None, identity: str | None = None) -> bool:
    try:
        path = sample / "seed_source.json"
        if version is not None:
            from orcs.core.sonic.specs import seed_path
            path = seed_path(sample, version).with_name('seed_source.json')
        receipt = json.loads(path.read_text())
        if version is not None and (receipt.get('sonic_version') != version or
                                    not identity or receipt.get('contract_id') != identity):
            return False
        if version is not None:
            from orcs.core.data.seed_identity import stair_seed_identity
            expected = stair_seed_identity(sample)
            if any(receipt.get(key) != value for key, value in expected.items()):
                return False
        return receipt["source_sha256"] == source_digest(sample)
    except (OSError, ValueError, KeyError):
        return False


def preserve_existing_seed(path: Path) -> None:
    """Keep reproducible old seed/receipt pairs before successful replacement."""
    if not path.exists():
        return
    import shutil
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    backup = path.parent / "seed_history" / digest
    backup.mkdir(parents=True, exist_ok=True)
    for name in (path.name, "seed_source.json", "contact_quality.json", "contact_quality.npz", "stair_contact_targets.npz", "stair_contact_targets.json"):
        source = path.parent / name
        target = backup / name
        if source.exists() and not target.exists():
            shutil.copy2(source, target)
