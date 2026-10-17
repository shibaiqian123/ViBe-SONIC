from __future__ import annotations

import numpy as np
import pytest

from orcs.core.data.contact_support import SurfaceSample, generate_support_targets, label_support
from orcs.core.data.seeds import SeedMotion


def _surface(z: float = 0.0, normal=(0.0, 0.0, 1.0)) -> SurfaceSample:
    return SurfaceSample(np.array([0.0, 0.0, z]), np.array(normal))


def test_support_rejects_vertical_riser_and_accepts_flat_tread() -> None:
    positions = np.zeros((2, 2, 3))
    velocities = np.zeros_like(positions)
    surfaces = [[_surface(normal=(1.0, 0.0, 0.0))], [_surface(z=0.2)]]
    mask, _, _ = label_support(positions, velocities, surfaces)
    assert not mask[0].any()
    assert mask[1].all()


def test_support_target_is_fixed_during_run_and_has_transition() -> None:
    positions = np.zeros((10, 2, 3))
    mask = np.zeros((10, 2), dtype=bool)
    mask[2:8] = True
    points = np.zeros((10, 2, 3))
    points[2:8, :, 2] = 0.3
    normals = np.zeros_like(points)
    normals[..., 2] = 1.0
    targets, weights = generate_support_targets(
        positions, mask, points, normals,
        ankle_to_sole_w=np.zeros((2, 3)), fps=50.0,
    )
    np.testing.assert_allclose(targets[2:8, :, 2], 0.3)
    assert weights[2, 0] < 1.0
    assert weights[4, 0] == pytest.approx(1.0)
    assert weights[0, 0] == 0.0


def test_support_requires_one_surface_list_per_frame() -> None:
    positions = np.zeros((3, 2, 3))
    with pytest.raises(ValueError, match="one surface list"):
        label_support(positions, np.zeros_like(positions), [[]])


def test_seed_support_channels_round_trip(tmp_path) -> None:
    t, joints, bodies = 2, 1, 1
    seed = SeedMotion(
        fps=50.0,
        source_frame_idx=np.arange(t),
        robot_root_pos_w=np.zeros((t, 3)), robot_root_quat_w=np.tile([1, 0, 0, 0], (t, 1)),
        robot_root_lin_vel_w=np.zeros((t, 3)), robot_root_ang_vel_w=np.zeros((t, 3)),
        joint_pos=np.zeros((t, joints)), joint_vel=np.zeros((t, joints)), last_action=np.zeros((t, joints)),
        body_pos_w=np.zeros((t, bodies, 3)), body_quat_w=np.tile([1, 0, 0, 0], (t, bodies, 1)),
        assist_force_w=np.zeros((t, bodies, 3)), assist_torque_w=np.zeros((t, bodies, 3)),
        body_tracking_error=np.zeros((t, bodies)), assist_saturation=np.ones(t), valid=np.ones(t, dtype=bool),
        joint_names=("j",), body_names=("b",),
        support_mask=np.ones((t, 2), dtype=bool), support_target_ankle_w=np.zeros((t, 2, 3)),
        support_surface_normal_w=np.tile([0, 0, 1.0], (t, 2, 1)), support_transition_weight=np.ones((t, 2)),
    )
    path = seed.save(tmp_path / "seed_state.npz")
    loaded = SeedMotion.load(path)
    assert loaded.support_mask is not None
    np.testing.assert_array_equal(loaded.support_mask, seed.support_mask)
