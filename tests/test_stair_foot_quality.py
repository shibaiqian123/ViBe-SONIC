"""Actual collision-surface diagnostics, distinct from numerical seed validity."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
spec = importlib.util.spec_from_file_location(
    "foot_quality",
    Path(__file__).resolve().parents[1] / "src/orcs/core/data/foot_quality.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FootDiagnostics, analyze_seed = module.FootDiagnostics, module.analyze_seed


def model():
    return mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
    <geom type="plane" size="5 5 .1"/>
    <body name="pelvis"><freejoint/><geom type="sphere" size=".05" mass="1" contype="0" conaffinity="0"/>
    <body name="left_ankle_roll_link" pos="0 .2 0"><geom name="left_foot_collision" type="capsule" fromto="-.1 0 .01 .1 0 .01" size=".01"/></body>
    <body name="right_ankle_roll_link" pos="0 -.2 0"><geom name="right_foot_collision" type="capsule" fromto="-.1 0 .01 .1 0 .01" size=".01"/></body>
    </body></worldbody></mujoco>""")


def test_flat_feet_and_nonzero_origin():
    m = model()
    d = mujoco.MjData(m)
    d.qpos[:3] = [2, 3, 0]
    mujoco.mj_forward(m, d)
    groups = m.geom_group.copy()
    q = FootDiagnostics(m).measure(d)
    np.testing.assert_allclose(q["sole_gap_m"], 0, atol=1e-7)
    np.testing.assert_allclose(q["sole_points_w"][:, 0, :2], [[1.9, 3.2], [1.9, 2.8]])
    np.testing.assert_array_equal(m.geom_group, groups)
    assert q["geometric_contact"].all()


def test_toe_support_and_swing_are_different():
    m = model()
    d = mujoco.MjData(m)
    angle = np.deg2rad(25)
    d.qpos[3:7] = [np.cos(angle / 2), 0, np.sin(angle / 2), 0]
    d.qpos[2] = 0.1 * np.sin(angle) + 0.01 * (1 - np.cos(angle))
    mujoco.mj_forward(m, d)
    q = FootDiagnostics(m).measure(d)
    assert np.all(q["sole_gap_m"][:, 0] > 0.08)
    np.testing.assert_allclose(q["sole_gap_m"][:, 1], 0, atol=1e-7)
    np.testing.assert_allclose(q["foot_pitch_deg"], -25)
    d.qpos[2] += 0.2
    mujoco.mj_forward(m, d)
    q = FootDiagnostics(m).measure(d)
    assert not q["geometric_contact"].any()
    assert np.all(q["sole_gap_m"] > 0.19)


def test_step_edge_uses_different_surfaces():
    spec = mujoco.MjSpec.from_string("""<mujoco><worldbody><geom type="plane" size="5 5 .1"/>
    <geom type="box" size=".5 2 .1" pos=".5 0 .1"/>
    <body name="pelvis"><freejoint/>
    <body name="left_ankle_roll_link"><geom type="capsule" fromto="-.1 .2 .01 .1 .2 .01" size=".01"/></body>
    <body name="right_ankle_roll_link"><geom type="capsule" fromto="-.1 -.2 .01 .1 -.2 .01" size=".01"/></body>
    </body></worldbody></mujoco>""")
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2] = 0.2
    mujoco.mj_forward(m, d)
    q = FootDiagnostics(m).measure(d)
    np.testing.assert_allclose(q["sole_gap_m"], [[0.2, 0], [0.2, 0]], atol=1e-7)


def test_report_does_not_turn_finite_into_contact_pass():
    m = model()
    frames = 3
    s = SimpleNamespace(
        num_frames=frames,
        fps=50,
        robot_root_pos_w=np.tile([0, 0, 0.2], (frames, 1)),
        robot_root_quat_w=np.tile([1, 0, 0, 0], (frames, 1)),
        joint_pos=np.zeros((frames, 0)),
        body_tracking_error=np.zeros((frames, 1)),
        assist_force_w=np.zeros((frames, 1, 3)),
        valid=np.ones(frames, dtype=bool),
    )
    arrays, report = analyze_seed(
        m, s, np.zeros(3), np.arange(7), np.array([], dtype=int)
    )
    assert report["quality_pass"] is None
    assert report["no_foot_contact_frames"] == 3
    assert report["longest_no_foot_contact_s"] == pytest.approx(0.06)
    assert np.isnan(arrays["support_slip_m_s"]).all()
