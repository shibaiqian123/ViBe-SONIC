"""Lightweight stair seed mapping/hand identity, independent of task imports."""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path

STAIR_POINT_MAPPING = "stairs-contact-v2"
STAIR_CONTACT_CONFIG = "distance<=3mm;normal<=30deg;transition=0.1s"


def stair_seed_identity(sample=None) -> dict:
    mapping, contact = STAIR_POINT_MAPPING, STAIR_CONTACT_CONFIG
    if sample is not None:
        from orcs.core.data.human_timeline import is_human_clock, validate
        if is_human_clock(sample):
            validate(sample)
            mapping, contact = "legacy", "disabled"
    path = os.environ.get("ORCS_G1_HAND_URDF")
    if not path:
        return {
            "point_mapping": mapping,
            "contact_config": contact,
            "hand_config": "flat-hand-v1",
        }
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
        "point_mapping": mapping,
        "contact_config": contact,
        "hand_config": "dex1-fixed-zero:" + h.hexdigest(),
    }
