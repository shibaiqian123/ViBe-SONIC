"""Pinned, take-specific stair source. No curb yaw/vertex-layout assumptions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from orcs.cli.grail_stairs import check_payload, payload_paths
from orcs.tasks.perloco.sources.grail import _JOINT_NAMES
from orcs.tasks.perloco.sources.smplx_fk import smpl_channels, smplx_dir
from orcs.tasks.perloco.sources.stair_geometry import (
    collision_parts,
    static_object_transform,
    usd_meshes,
)
from orcs.tasks.perloco.terrain_spec import (
    ClipSpec,
    MeshSpec,
    SmplSpec,
    TileSpec,
    box_from_vertices,
)

STAGE_VERSION = 2


def fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def record(data: dict, keys: tuple[str, ...]) -> dict:
    if any(k in data for k in keys):
        return data
    if len(data) == 1 and isinstance(next(iter(data.values())), dict):
        return record(next(iter(data.values())), keys)
    raise ValueError(f"ambiguous/unsupported GRAIL record; expected {keys}, found {list(data)}")


class GrailStairsSource:
    name = "grail_stairs"
    default_root = "GRAIL-stairs"

    def __init__(self, root, *, selection, families=None, levels=None, smpl=False, **kwargs):
        del levels, kwargs
        self.root = Path(root)
        self.selection = json.loads(Path(selection).read_text())
        self.smpl = smpl
        self.samples = [x for x in self.selection["samples"] if families is None or x["tile_id"] in families]
        if not self.samples:
            raise ValueError("empty stair selection")
        check_payload(self.root, payload_paths(self.selection, smpl, False))
        self._tiles = {}
        self._clips = {}

    def _load(self, row):
        import joblib
        if row["tile_id"] in self._clips:
            return
        files = {k: self.root / p for k, p in row["files"].items()}
        robot = record(joblib.load(files["robot"]), ("dof", "joint_pos"))
        # Released legacy GRAIL trajectories use xyzw root_rot; explicit *_w is wxyz.
        if "dof" in robot:
            joints = np.asarray(robot["dof"], float)
            pos = np.asarray(robot["root_trans_offset"], float)
            quat = np.asarray(robot["root_rot"], float)[:, [3, 0, 1, 2]]
        else:
            joints = np.asarray(robot["joint_pos"], float)
            pos = np.asarray(robot["root_pos_w"], float)
            quat = np.asarray(robot["root_quat_w"], float)
        names = tuple(robot.get("joint_names", _JOINT_NAMES))
        fps = float(robot["fps"])
        if (len(pos) < 3 or joints.shape != (len(pos), len(names)) or quat.shape != (len(pos), 4)
                or pos.shape != (len(pos), 3) or fps <= 0
                or not all(np.isfinite(x).all() for x in (pos, quat, joints))):
            raise ValueError(f"invalid robot channels: {row['stem']}")
        norms = np.linalg.norm(quat, axis=1, keepdims=True)
        if np.any(norms < 1e-8):
            raise ValueError("zero robot quaternion")
        quat = quat / norms
        metadata = joblib.load(files["meta"])
        if not isinstance(metadata, dict):
            raise ValueError("GRAIL metadata must be a dictionary")
        obj = joblib.load(files["objects"])
        t, r, scale = static_object_transform(obj)
        meshes = usd_meshes(files["object_usd"])
        boxes, hulls, points = [], [], [pos]
        for vertices, faces in meshes:
            vertices = (vertices * scale) @ r.T + t
            points.append(vertices)
            # Only promote a whole convex eight-corner cuboid, never vertex-count chunks.
            unique = np.unique(np.round(vertices, 8), axis=0)
            try:
                if len(unique) != 8:
                    raise ValueError("not a box")
                import trimesh
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
                if not mesh.is_convex or not mesh.is_watertight:
                    raise ValueError("not a solid convex box")
                boxes.append(box_from_vertices(unique))
            except ValueError:
                hulls.extend(MeshSpec(v, f) for v, f in collision_parts(vertices, faces))
        extent = np.max(np.abs(np.concatenate(points)[:, :2]), axis=0)
        tile_size = tuple(np.maximum(6.0, 2 * (extent + 1.0)))
        smpl = None
        if self.smpl:
            smpl = SmplSpec(*smpl_channels(files["recon"], smplx_dir()))
            if len(smpl.joints) < 3 or not all(np.isfinite(x).all() for x in (smpl.joints, smpl.root_quat, smpl.joints_viz)):
                raise ValueError("invalid SMPL channels")
            # Endpoint phase is a diagnostic approximation, not verified correspondence.
            endpoint_error = np.linalg.norm(smpl.joints_viz[[0, -1], 0] - pos[[0, -1]], axis=1)
            if np.any(endpoint_error > 1.0):
                raise ValueError(f"SMPL/robot endpoint mismatch >1m: {endpoint_error}; inspect source alignment")
        source_paths = [p for k, p in files.items() if k != "recon" or self.smpl]
        source_paths += [self.root / p for p in self.selection.get("dependencies", [])]
        digest = fingerprint(source_paths)
        meta = {"source": f"grail/{row['category']}", "take": row["stem"],
                "revision": self.selection["revision"], "stage_version": STAGE_VERSION,
                "input_sha256": digest, "coordinates": "z-up metres, tile-local, wxyz",
                "alignment": "unverified endpoint-normalized phase; diagnostic only",
                "smpl_alignment_verified": False,
                "smpl_alignment_reason": "released robot rollout has no verified recon-frame mapping",
                "robot_frames": len(pos), "src_fps": fps,
                "smpl_frames": len(smpl.joints) if smpl else None,
                "tile_size": tile_size}
        tile = TileSpec(family=row["tile_id"], level=0.0, boxes=tuple(boxes), meshes=tuple(hulls), size=tile_size)
        clip = ClipSpec(name=row["stem"], root_pos=pos, root_quat=quat, joint_pos=joints,
                        joint_names=names, fps=fps, family=row["tile_id"], level=0.0, smpl=smpl, meta=meta)
        self._tiles[row["tile_id"]] = tile
        self._clips[row["tile_id"]] = clip

    def tiles(self):
        for row in self.samples:
            self._load(row)
            yield self._tiles[row["tile_id"]]

    def clips(self):
        for row in self.samples:
            self._load(row)
            yield self._clips[row["tile_id"]]
