"""Human-clock dataset contract; independent of simulation imports."""
import json
from pathlib import Path

import numpy as np

CONTRACT = "grail-human-clock-v1"


def source_fps(data):
    rec = data.get("human_data", data)
    if "poses" not in rec:
        if len(rec) != 1:
            raise ValueError("ambiguous human record")
        rec = next(iter(rec.values()))
    value = float(rec["mocap_frame_rate"])
    if not np.isfinite(value) or value <= 0:
        raise ValueError("invalid human mocap_frame_rate")
    return value


def is_human_clock(sample):
    path = Path(sample)/"metadata.json"
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    return meta.get("timeline_contract") == CONTRACT


def validate(sample):
    sample = Path(sample)
    m = json.loads((sample/"metadata.json").read_text())
    if m.get("timeline_contract") != CONTRACT:
        raise ValueError("unverified human timeline")
    fps = float(m["source_human_fps"])
    count = int(m["source_human_frames"])
    if not np.isfinite(fps) or fps <= 0 or count < 3 or m["output_fps"] != 50:
        raise ValueError("invalid human timebase")
    expected = len(np.arange(0, (count-1)/fps, 1/50))
    if m["output_frames"] != expected or m["frames"] != expected:
        raise ValueError("human clock frame count mismatch")
    if m.get("point_mapping") != "legacy" or m.get("wrist_reference") != "zero":
        raise ValueError("unsupported human mapping/wrist contract")
    if len(m.get("recon_sha256", "")) != 64:
        raise ValueError("missing recon provenance")
    if (sample/"stair_contact_targets.npz").exists():
        raise ValueError("human-clock baseline forbids experimental contact targets")
    with np.load(sample/"smpl_motion.npz", allow_pickle=False) as h:
        for name, shape in (("smpl_joints", (expected,24,3)),
                            ("smpl_joints_viz_w", (expected,24,3)),
                            ("smpl_root_quat_w", (expected,4))):
            if h[name].shape != shape or not np.isfinite(h[name]).all():
                raise ValueError(f"invalid human channel {name}")
    with np.load(sample/"motion.npz", allow_pickle=False) as robot:
        if robot["joint_pos"].shape != (expected,29) or np.any(robot["joint_pos"]):
            raise ValueError("human clock requires zero placeholder joint references")
    return m
