import importlib.util
from pathlib import Path

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / f"src/orcs/core/data/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mapping_and_same_target_path():
    torch = pytest.importorskip("torch")
    from orcs.core.data.point_reference import (
        G1_SMPL_BODY_MAP,
        infer_morphology_scale,
        point_body_map,
        scaled_smpl_targets,
    )
    from orcs.core.data.seed_identity import STAIR_POINT_MAPPING

    assert point_body_map() == G1_SMPL_BODY_MAP
    assert dict(point_body_map(STAIR_POINT_MAPPING))["left_ankle_roll_link"] == 7
    assert dict(point_body_map())["left_ankle_roll_link"] == 10
    j = torch.zeros(2, 24, 3)
    j[:, 0, 2] = 1
    for hip, knee, ankle, foot in [(1, 4, 7, 10), (2, 5, 8, 11)]:
        j[:, hip, 2] = 0.9
        j[:, knee, 2] = 0.5
        j[:, ankle, 2] = 0.1
        j[:, foot, 2] = 0
    names = [n for n, _ in G1_SMPL_BODY_MAP]
    robot = j[0, [idx for _, idx in point_body_map(STAIR_POINT_MAPPING)]]
    assert infer_morphology_scale(
        j, robot, names, mapping=STAIR_POINT_MAPPING
    ) == pytest.approx(1)
    assert infer_morphology_scale(j, robot, names) == pytest.approx(0.9)
    origin = torch.tensor([2.0, 3.0, 4.0])
    expected = j[:, [idx for _, idx in point_body_map(STAIR_POINT_MAPPING)]] + origin
    actual = scaled_smpl_targets(j, 1, origin, mapping=STAIR_POINT_MAPPING)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        scaled_smpl_targets(j, 1, origin),
        j[:, [idx for _, idx in G1_SMPL_BODY_MAP]] + origin,
    )


def test_identity_changes_with_hand_assets(tmp_path, monkeypatch):
    stair_seed_identity = load("seed_identity").stair_seed_identity
    monkeypatch.delenv("ORCS_G1_HAND_URDF", raising=False)
    assert stair_seed_identity()["hand_config"] == "flat-hand-v1"
    p = tmp_path / "robot.urdf"
    mesh = tmp_path / "hand.stl"
    mesh.write_bytes(b"first")
    p.write_text(
        '<robot><link name="left_dex1_base_link"><visual><geometry><mesh filename="hand.stl"/></geometry></visual></link></robot>'
    )
    monkeypatch.setenv("ORCS_G1_HAND_URDF", str(p))
    old = stair_seed_identity()
    mesh.write_bytes(b"second")
    assert stair_seed_identity() != old


def test_preserve_old_seed_pair(tmp_path):
    preserve_existing_seed = load("seed_receipt").preserve_existing_seed
    p = tmp_path / "seed_state.npz"
    p.write_bytes(b"old")
    (tmp_path / "seed_source.json").write_text("{}")
    preserve_existing_seed(p)
    p.write_bytes(b"new")
    copies = list(tmp_path.glob("seed_history/*/seed_state.npz"))
    assert len(copies) == 1 and copies[0].read_bytes() == b"old"
    assert copies[0].with_name("seed_source.json").is_file()
