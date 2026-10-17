"""Lightweight stair seed mapping/hand identity, independent of task imports."""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

STAIR_POINT_MAPPING = "stairs-ankle-v1"


def stair_seed_identity() -> dict:
    path = os.environ.get("ORCS_G1_HAND_URDF")
    if not path:
        return {"point_mapping": STAIR_POINT_MAPPING, "hand_config": "flat-hand-v1"}
    urdf = Path(path).expanduser().resolve(strict=True)
    root = ET.parse(urdf).getroot()
    h = hashlib.sha256(b"dex1-fixed-zero-v1")
    # Include source content rather than host-specific absolute paths.
    h.update(urdf.read_bytes())
    for link in root.findall("link"):
        if "dex1" not in link.get("name", ""):
            continue
        for mesh in link.findall(".//mesh"):
            target = (urdf.parent / mesh.get("filename")).resolve(strict=True)
            h.update(mesh.get("filename").encode())
            with target.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
    return {
        "point_mapping": STAIR_POINT_MAPPING,
        "hand_config": "dex1-fixed-zero:" + h.hexdigest(),
    }
