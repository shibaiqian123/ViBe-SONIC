"""Explicit release contracts; dimensions never identify a model version alone."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

REVISION = "6733128a3d8a523b1418b06bca3cdf61c8b0987f"
OFFICIAL_SHA256 = {
    "release": "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909",
    "v1.1": "af24831ae59424a0cf92cb56e9bb6dc1a59ab859fd055ba13187e9e6f0a59f43",
    "low-latency": "0031ae7db24747445d6eb7c27697640973a837546f0b8763e775143c47d4507c",
}
CONTRACT_VERSION = 1
ENCODER_HIDDEN = (2048, 1024, 512, 512)


@dataclass(frozen=True)
class SonicSpec:
    version: str
    directory: str
    decoder_hidden: tuple[int, ...]
    g1_skip: int
    smpl_frames: int
    orientation: str
    g1_frames: int = 10
    smpl_skip: int = 1
    history: int = 10
    dt: float = 0.02
    num_tokens: int = 2
    token_dim: int = 32
    fsq_levels: int = 32

    def frames(self, mode: str) -> int:
        if mode not in ("g1", "smpl"):
            raise ValueError(f"unsupported SONIC encoder mode: {mode}")
        return self.g1_frames if mode == "g1" else self.smpl_frames

    def width(self, mode: str) -> int:
        return self.frames(mode) * (64 if mode == "g1" else 84)

    def contract(self, mode: str) -> dict:
        self.frames(mode)
        return {**asdict(self), "contract_version": CONTRACT_VERSION, "encoder_mode": mode,
                "tokenizer_dim": self.width(mode), "proprio_dim": 930, "action_dim": 29,
                "encoder_hidden": list(ENCODER_HIDDEN),
                "quaternion_order": "wxyz", "history_order": "oldest_first",
                "joint_order": list(MJ_ORDER), "proprio_order": ["base_ang_vel", "joint_pos", "joint_vel", "actions", "gravity_dir"],
                "action_convention": "default_position + action * (0.25 * effort_limit / stiffness)"}


SPECS = {
    "release": SonicSpec("release", "sonic_release", (2048, 2048, 1024, 1024, 512, 512), 5, 10, "body"),
    "v1.1": SonicSpec("v1.1", "sonic_v1_1", (4096, 4096, 2048, 2048, 1024, 1024, 512, 512), 5, 10, "heading"),
    "low-latency": SonicSpec("low-latency", "low_latency", (4096, 4096, 2048, 2048, 1024, 1024, 512, 512), 1, 4, "body"),
}
IL_ORDER = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)
MJ_ORDER = tuple(
    [f"{side}_{joint}_joint" for side in ("left", "right")
     for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{side}_{joint}_joint" for side in ("left", "right")
       for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)
WRIST_MJ = tuple(MJ_ORDER.index(name) for name in IL_ORDER[23:29])


def get_spec(version: str) -> SonicSpec:
    try:
        return SPECS[version]
    except KeyError as exc:
        raise ValueError(f"unknown SONIC version {version!r}") from exc


def checkpoint_path(version: str, mode: str) -> Path:
    from orcs.core.paths import DEPS_ROOT
    get_spec(version).frames(mode)
    return DEPS_ROOT / "checkpoints" / "sonic" / version / "ported" / (
        "smpl_ported.pt" if mode == "smpl" else "last_ported.pt")


def contract_id(contract: dict, source_sha256: str) -> str:
    return hashlib.sha256(json.dumps([contract, source_sha256], sort_keys=True).encode()).hexdigest()


def seed_path(sample: Path, version: str = "release") -> Path:
    get_spec(version)
    return sample / "seed_state.npz" if version == "release" else sample / "seeds" / version / "seed_state.npz"
