"""Audit stair seeds against actual robot/terrain collision geometry, without rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def audit_source_contacts(sample: Path, raw_root: Path) -> dict:
    """Record unverified raw channels; never manufacture semantic labels."""
    import joblib

    meta = json.loads((sample / "metadata.json").read_text())

    # Staging may wrap source metadata; find the explicit source/take record.
    def find(value):
        if isinstance(value, dict):
            if "take" in value and "source" in value:
                return value
            for child in value.values():
                result = find(child)
                if result is not None:
                    return result
        return None

    record = find(meta)
    if record is None:
        return {"status": "source metadata unavailable", "used_as_constraint": False}
    category = str(record["source"]).split("/")[-1]
    path = raw_root / "data" / category / "recon" / (record["take"] + ".pkl")
    if not path.is_file():
        return {"status": "raw recon unavailable", "used_as_constraint": False}
    source = joblib.load(path).get("human_data", {})
    while "poses" not in source and len(source) == 1:
        source = next(iter(source.values()))
    probs = np.asarray(source.get("foot_contact_probs", []))
    with np.load(sample / "motion.npz") as motion:
        staged_frames = len(motion['joint_pos'])
        staged_fps = float(motion['fps'].reshape(-1)[0])
    return {
        "status": "unverified channel meanings",
        "used_as_constraint": False,
        "shape": list(probs.shape),
        "finite": bool(np.isfinite(probs).all()),
        "range": [float(probs.min()), float(probs.max())] if probs.size else None,
        "source_fps": float(source.get("mocap_frame_rate", 0)),
        "source_frames": len(source.get('poses', [])),
        "staged_frames": staged_frames,
        "staged_fps": staged_fps,
        "time_mapping": "endpoint-normalized phase, not verified timestamp synchronization",
    }


def main():
    import mujoco

    from orcs.cli.view_terrain_motions import _build_model
    from orcs.core.data.foot_quality import analyze_seed, save_report
    from orcs.core.data.seed_identity import stair_seed_identity
    from orcs.core.data.seeds import SeedMotion
    from orcs.core.paths import DATA_ROOT
    from orcs.core.sonic.specs import seed_path
    from orcs.tasks.perloco.terrain import staged_tile_size

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--reference",
        action="store_true",
        help="Audit paired robot motion instead of seed",
    )
    parser.add_argument(
        "--seed",
        type=Path,
        help="Explicit old seed for comparison; never used for training",
    )
    parser.add_argument(
        "--sonic-version", choices=("release", "v1.1", "low-latency"), default="release"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, default=DATA_ROOT / "GRAIL-stairs")
    args = parser.parse_args()
    if (
        bool(args.source) == args.all
        or (args.seed and args.all)
        or (args.seed and args.reference)
    ):
        parser.error("choose --source or --all; --seed requires --source")
    samples = (
        [args.source]
        if args.source
        else sorted(
            (DATA_ROOT / "terrain_motions/grail_stairs").glob("*/level_*/sample*")
        )
    )
    if not samples:
        parser.error("no stair samples")
    failed = {}
    for sample in samples:
        key = sample.parent.parent.name + "_" + sample.name
        try:
            path = args.seed or seed_path(sample, args.sonic_version)
            if args.reference:
                path = sample / "motion.npz"
                with np.load(path) as motion:
                    n = len(motion["joint_pos"])
                    seed = SimpleNamespace(
                        num_frames=n,
                        fps=float(motion["fps"].reshape(-1)[0]),
                        joint_names=tuple(motion["joint_names"].tolist()),
                        joint_pos=motion["joint_pos"].copy(),
                        robot_root_pos_w=motion["body_pos_w"][:, 0].copy(),
                        robot_root_quat_w=motion["body_quat_w"][:, 0].copy(),
                        body_tracking_error=np.zeros((n, 1)),
                        assist_force_w=np.zeros((n, 1, 3)),
                        valid=np.isfinite(motion["joint_pos"]).all(-1),
                    )
            else:
                seed = SeedMotion.load(path)
            root = sample.parents[2]
            model = _build_model(
                root,
                sample.parent.parent.name,
                float(sample.parent.name.removeprefix("level_")),
            )
            free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
            if len(free_ids) != 1:
                raise ValueError("stair audit expects one free robot root")
            root_address = int(model.jnt_qposadr[free_ids[0]])
            joint_addresses = np.array(
                [int(model.joint(name).qposadr[0]) for name in seed.joint_names]
            )
            size = staged_tile_size(root)
            origin = np.array([size[0] / 2, size[1] / 2, 0.0])
            arrays, summary = analyze_seed(
                model,
                seed,
                origin,
                np.arange(root_address, root_address + 7),
                joint_addresses,
            )
            if args.reference:
                for field in (
                    "tracking_error_mean_m",
                    "tracking_error_max_m",
                    "assist_force_max_n",
                ):
                    summary[field] = None
                arrays.pop("body_tracking_error_m")
                arrays.pop("assist_force_norm_n")
            summary["trajectory_kind"] = "robot_reference" if args.reference else "seed"
            receipt = path.with_name("seed_source.json")
            summary.update(
                seed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                seed_receipt=json.loads(receipt.read_text())
                if receipt.exists()
                else None,
                evaluation_identity=stair_seed_identity(),
                source_contacts=audit_source_contacts(sample, args.raw_root),
            )
            save_report(args.out_dir / key, arrays, summary)
            print(key, json.dumps(summary), flush=True)
        except Exception as exc:
            failed[key] = f"{type(exc).__name__}: {exc}"
            print(key, failed[key], flush=True)
    if failed:
        raise SystemExit(f"seed quality audit failed: {failed}")


if __name__ == "__main__":
    main()
