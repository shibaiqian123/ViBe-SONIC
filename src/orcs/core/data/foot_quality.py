"""Read-only, CPU MuJoCo foot/terrain diagnostics; no acceptance gate."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

REPORT_VERSION = "stair-contact-v2"
CONTACT_DISTANCE_M = (
    0.003  # Diagnostic near-contact band, not a quality pass threshold.
)


class FootDiagnostics:
    def __init__(self, model):
        self.model = model
        moving = np.zeros(model.nbody, dtype=bool)
        for i in range(1, model.nbody):
            moving[i] = moving[model.body_parentid[i]] or model.body_jntnum[i] > 0
        self.terrain = ~moving[model.geom_bodyid]
        self.feet = []
        for side in ("left", "right"):
            name = side + "_ankle_roll_link"
            matches = [
                i
                for i in range(model.nbody)
                if model.body(i).name == name or model.body(i).name.endswith("/" + name)
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one robot foot body {name}, got {matches}")
            body = matches[0]
            ids = [
                i
                for i in range(model.ngeom)
                if model.geom_bodyid[i] == body
                and (model.geom_contype[i] or model.geom_conaffinity[i])
            ]
            candidates = []
            for i in ids:
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, model.geom_quat[i])
                rotation = rotation.reshape(3, 3)
                kind = model.geom_type[i]
                if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    for sign in (-1, 1):
                        candidates.append(
                            (
                                model.geom_pos[i]
                                + sign * model.geom_size[i, 1] * rotation[:, 2],
                                float(model.geom_size[i, 0]),
                            )
                        )
                elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    candidates.append(
                        (model.geom_pos[i].copy(), float(model.geom_size[i, 0]))
                    )
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    from itertools import product

                    for signs in product((-1, 1), repeat=3):
                        candidates.append(
                            (
                                model.geom_pos[i]
                                + rotation @ (model.geom_size[i] * signs),
                                0.0,
                            )
                        )
                else:
                    raise ValueError(
                        f"unsupported foot collision geometry: {model.geom(i).name}"
                    )
            if not candidates:
                raise ValueError(f"no foot collision geometry for {side}")
            local = np.stack([p for p, _ in candidates])
            radius = np.array([r for _, r in candidates])
            mid = (local[:, 0].min() + local[:, 0].max()) / 2
            regions = (
                np.flatnonzero(local[:, 0] <= mid),
                np.flatnonzero(local[:, 0] > mid),
            )
            if any(not len(region) for region in regions):
                raise ValueError(f"cannot distinguish heel/toe geometry for {side}")
            self.feet.append((body, set(ids), local, radius, regions))

    def measure(self, data):
        """Distances are vertical rays to actual static collision surfaces.

        contact is geometry proximity, not measured load. Rays include the floor.
        Missing surfaces remain NaN, never silently treated as flat ground.
        """
        model = self.model
        groups = model.geom_group.copy()
        gaps, points, choices, pitch, contacts, penetrations = [], [], [], [], [], []
        try:
            model.geom_group[:] = 5
            model.geom_group[self.terrain] = 4
            ray_groups = np.array([0, 0, 0, 0, 1, 0], dtype=np.uint8)
            ray_height = max(10.0, 2 * float(model.stat.extent))
            for body, ids, local, radii, regions in self.feet:
                rotation = data.xmat[body].reshape(3, 3)
                world = local @ rotation.T + data.xpos[body]
                world[:, 2] -= radii
                distances = []
                for point in world:
                    start = point + [0.0, 0.0, ray_height]
                    gid = np.array([-1], dtype=np.int32)
                    distance = mujoco.mj_ray(
                        model,
                        data,
                        start,
                        np.array([0.0, 0.0, -1.0]),
                        ray_groups,
                        True,
                        -1,
                        gid,
                    )
                    distances.append(distance - ray_height if distance >= 0 else np.nan)
                distances = np.asarray(distances)
                selected = [
                    int(region[np.nanargmin(distances[region])])
                    if np.isfinite(distances[region]).any()
                    else int(region[0])
                    for region in regions
                ]
                gaps.append(distances[selected])
                points.append(world[selected])
                choices.append(selected)
                pitch.append(np.degrees(np.arcsin(np.clip(rotation[2, 0], -1, 1))))
                ds = [
                    float(c.dist)
                    for c in data.contact
                    if (int(c.geom[0]) in ids and self.terrain[int(c.geom[1])])
                    or (int(c.geom[1]) in ids and self.terrain[int(c.geom[0])])
                ]
                contacts.append(bool(ds and min(ds) <= CONTACT_DISTANCE_M))
                penetrations.append(max(0.0, -min(ds)) if ds else 0.0)
        finally:
            model.geom_group[:] = groups
        return dict(
            sole_gap_m=np.array(gaps),
            sole_points_w=np.array(points),
            sole_point_index=np.array(choices),
            foot_pitch_deg=np.array(pitch),
            geometric_contact=np.array(contacts),
            penetration_m=np.array(penetrations),
        )


def longest_run(mask):
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def analyze_seed(model, seed, origin, root_addresses, joint_addresses):
    diagnostics = FootDiagnostics(model)
    data = mujoco.MjData(model)
    rows = []
    for frame in range(seed.num_frames):
        data.qpos[root_addresses[:3]] = seed.robot_root_pos_w[frame] + origin
        data.qpos[root_addresses[3:7]] = seed.robot_root_quat_w[frame]
        data.qpos[joint_addresses] = seed.joint_pos[frame]
        mujoco.mj_forward(model, data)
        rows.append(diagnostics.measure(data))
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    velocity = (
        np.linalg.norm(np.diff(arrays["sole_points_w"][..., :2], axis=0), axis=-1)
        * seed.fps
    )
    stable = arrays["sole_point_index"][1:] == arrays["sole_point_index"][:-1]
    support = arrays["geometric_contact"][1:] & arrays["geometric_contact"][:-1]
    slip = np.full((seed.num_frames, 2, 2), np.nan)
    slip[1:] = np.where(stable & support[..., None], velocity, np.nan)
    arrays["support_slip_m_s"] = slip
    arrays["body_tracking_error_m"] = seed.body_tracking_error
    arrays["assist_force_norm_n"] = np.linalg.norm(seed.assist_force_w, axis=-1)
    arrays["finite_state"] = seed.valid
    no_contact = ~arrays["geometric_contact"].any(-1)
    expected = seed.support_mask
    expected_misses = None
    support_gap_p95 = None
    if expected is not None:
        expected = np.asarray(expected, dtype=bool)
        measured_gap = np.nanmax(np.abs(arrays["sole_gap_m"]), axis=-1)
        expected_misses = expected & (
            ~np.isfinite(measured_gap)
            | (measured_gap > 0.01)
            | (arrays["penetration_m"][:, None] > 0.01)
        )
        arrays["expected_support"] = expected
        arrays["expected_support_gap_m"] = measured_gap
        arrays["expected_support_miss"] = expected_misses
        finite = measured_gap[expected & np.isfinite(measured_gap)]
        support_gap_p95 = float(np.percentile(finite, 95)) if len(finite) else None
    summary = dict(
        report_version=REPORT_VERSION,
        frames=seed.num_frames,
        fps=seed.fps,
        coordinate_frame="MuJoCo world; origin added once",
        origin=np.asarray(origin).tolist(),
        feet=["left", "right"],
        sole_regions=["heel", "toe"],
        contact_definition=f"collision distance <= {CONTACT_DISTANCE_M} m; not load-bearing proof",
        quality_pass=(
            bool(expected.any() and not expected_misses.any())
            if expected is not None else None
        ),
        contact_labels=(
            "support_mask uses paired robot geometry; source foot_contact_probs remain diagnostic only"
            if expected is not None else "diagnostic only; source channel semantics unverified"
        ),
        no_foot_contact_frames=int(no_contact.sum()),
        longest_no_foot_contact_s=longest_run(no_contact) / seed.fps,
        penetration_max_m=float(arrays["penetration_m"].max()),
        tracking_error_mean_m=float(seed.body_tracking_error.mean()),
        tracking_error_max_m=float(seed.body_tracking_error.max()),
        assist_force_max_n=float(arrays["assist_force_norm_n"].max()),
        expected_support_frames=int(expected.sum()) if expected is not None else 0,
        expected_support_misses=int(expected_misses.sum()) if expected_misses is not None else 0,
        expected_support_gap_p95_m=support_gap_p95,
    )
    return arrays, summary


def save_report(prefix: Path, arrays: dict, summary: dict):
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prefix.with_suffix(".npz"), **arrays)
    prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
