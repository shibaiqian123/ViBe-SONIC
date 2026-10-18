"""Fail closed for stair recon/rollout pairing without verified correspondence."""
import json
from pathlib import Path


def require_stair_alignment(sample: Path) -> None:
    metadata = json.loads((sample / "metadata.json").read_text())
    if metadata.get("source") not in ("grail/stair_p1", "grail/stair_p2"):
        return
    # No released source-to-rollout mapping has been verified yet. A boolean
    # in metadata or equal frame counts cannot certify a correspondence.
    raise ValueError(
        f"{sample}: GRAIL stair SMPL/robot time alignment is unverified; "
        "run orcs-audit-grail on the original sample. Robot-reference tasks remain available; "
        "rebuilding seed does not establish time correspondence."
    )
