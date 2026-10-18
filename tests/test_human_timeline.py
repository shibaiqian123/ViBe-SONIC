"""Human source clock acceptance/rejection without simulator imports."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("human_clock", Path(__file__).resolve().parents[1]/"src/orcs/core/data/human_timeline.py")
clock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clock)


def sample(tmp_path):
    n = 400  # 241 human frames at 30 Hz -> 8 seconds, exclusive endpoint at 50 Hz
    meta = dict(timeline_contract=clock.CONTRACT, source_human_fps=30,
                source_human_frames=241, output_fps=50, output_frames=n, frames=n,
                point_mapping="legacy", wrist_reference="zero", recon_sha256="a"*64)
    (tmp_path/"metadata.json").write_text(json.dumps(meta))
    q = np.zeros((n,4))
    q[:,0] = 1
    np.savez(tmp_path/"smpl_motion.npz", smpl_joints=np.zeros((n,24,3)),
             smpl_joints_viz_w=np.zeros((n,24,3)), smpl_root_quat_w=q)
    np.savez(tmp_path/"motion.npz", joint_pos=np.zeros((n,29)))
    return meta


def test_source_rate_requires_evidence():
    assert clock.source_fps({"human_data":{"p":{"mocap_frame_rate":30}}}) == 30
    with pytest.raises(KeyError):
        clock.source_fps({"poses":[],"fps":25})
    with pytest.raises(ValueError):
        clock.source_fps({"poses":[],"mocap_frame_rate":float("nan")})


def test_human_clock_not_robot_endpoint(tmp_path):
    meta = sample(tmp_path)
    assert clock.validate(tmp_path)["frames"] == 400
    meta["frames"] = 499
    (tmp_path/"metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="count"):
        clock.validate(tmp_path)


def test_compensation_and_robot_wrist_rejected(tmp_path):
    sample(tmp_path)
    np.savez(tmp_path/"motion.npz",joint_pos=np.ones((400,29)))
    with pytest.raises(ValueError,match="zero"):
        clock.validate(tmp_path)
    sample(tmp_path)
    (tmp_path/"stair_contact_targets.npz").touch()
    with pytest.raises(ValueError,match="contact"):
        clock.validate(tmp_path)


def test_missing_metadata_is_not_opt_in(tmp_path):
    assert not clock.is_human_clock(tmp_path)
