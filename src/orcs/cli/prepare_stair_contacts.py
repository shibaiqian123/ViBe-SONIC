"""Prepare stairs-contact-v2 targets from a paired robot-reference audit file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from orcs.core.data.contact_support import SurfaceSample, generate_support_targets, label_support


def _prepare(sample: Path, reference: Path, *, fps: float, overwrite: bool) -> dict:
    output = sample / "stair_contact_targets.npz"
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    with np.load(reference) as data:
        required = ("foot_positions_w", "foot_velocities_w", "surface_points_w", "surface_normals_w", "ankle_positions_w", "ankle_to_sole_w")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{reference}: missing {missing}")
        foot_pos = data["foot_positions_w"]
        foot_vel = data["foot_velocities_w"]
        surface_points = data["surface_points_w"]
        surface_normals = data["surface_normals_w"]
        ankles = data["ankle_positions_w"]
        offset = data["ankle_to_sole_w"]
    frames = len(foot_pos)
    if any(array.shape[0] != frames for array in (foot_vel, surface_points, surface_normals, ankles)):
        raise ValueError(f"{reference}: frame counts do not match")
    if surface_points.ndim != 3 or surface_points.shape[-1] != 3 or surface_normals.shape != surface_points.shape:
        raise ValueError(f"{reference}: surface arrays must be (T,S,3)")
    if offset.shape != (2, 3):
        raise ValueError(f"{reference}: ankle_to_sole_w must be (2,3)")
    surfaces = [
        [SurfaceSample(point, normal) for point, normal in zip(surface_points[t], surface_normals[t], strict=True)
         if np.isfinite(point).all() and np.isfinite(normal).all()]
        for t in range(frames)
    ]
    support, points, normals = label_support(foot_pos, foot_vel, surfaces)
    targets, weights = generate_support_targets(
        ankles, support, points, normals, ankle_to_sole_w=offset, fps=fps
    )
    np.savez_compressed(
        output,
        schema_version=np.array("stairs-contact-v2"),
        support_mask=support,
        support_target_ankle_w=targets,
        support_surface_normal_w=np.nan_to_num(normals),
        support_transition_weight=weights,
    )
    report = {
        "status": "ready",
        "sample": str(sample),
        "reference": str(reference),
        "frames": frames,
        "support_frames": int(support.sum()),
        "support_by_foot": support.sum(axis=0).tolist(),
        "output": str(output),
    }
    (sample / "stair_contact_targets.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True, help="staged sample directory")
    parser.add_argument("--reference", type=Path, default=None, help="paired reference .npz; defaults to sample/robot_reference.npz")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    sample = args.sample.expanduser().resolve()
    reference = (args.reference or sample / "robot_reference.npz").expanduser().resolve()
    print(json.dumps(_prepare(sample, reference, fps=args.fps, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
