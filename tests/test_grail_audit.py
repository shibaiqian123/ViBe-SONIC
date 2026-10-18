import importlib.util
import json
from pathlib import Path

import pytest


def load(name, relative):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(name, root / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("audit", "src/orcs/cli/audit_grail.py")
alignment = load("alignment", "src/orcs/core/data/grail_alignment.py")


def test_equal_frames_do_not_prove_alignment():
    result = audit.timeline_report({"dof": [0]*10, "fps": 30}, {"poses": [0]*10, "mocap_frame_rate": 30})
    assert result["status"] == "unverified"
    assert result["corresponding_human_frames"] is None


def test_ambiguous_record_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        audit.unwrap({"a": {"dof": []}, "b": {"dof": []}}, "dof")
    assert audit.unwrap({"a": {"dof": [1]}}, "dof")["dof"] == [1]


@pytest.mark.parametrize("source", ["grail/stair_p1", "grail/stair_p2"])
def test_old_seed_and_boolean_cannot_certify_time_mapping(tmp_path, source):
    (tmp_path / "metadata.json").write_text(json.dumps({"source": source, "smpl_alignment_verified": True}))
    with pytest.raises(ValueError, match="time alignment is unverified"):
        alignment.require_stair_alignment(tmp_path)


@pytest.mark.parametrize("source", ["grail/curb", "uolm"])
def test_other_tasks_unchanged(tmp_path, source):
    (tmp_path / "metadata.json").write_text(json.dumps({"source": source}))
    alignment.require_stair_alignment(tmp_path)


def test_joblib_numpy_payload(tmp_path):
    joblib = pytest.importorskip("joblib")
    np = pytest.importorskip("numpy")
    path = tmp_path / "robot.pkl"
    joblib.dump({"clip": {"dof": np.arange(87).reshape(3, 29)}}, path)
    assert audit.unwrap(joblib.load(path), "dof")["dof"].shape == (3, 29)


@pytest.mark.parametrize("root_offset,expected", [(0., 0.), (1., 1.)])
def test_official_replaces_root_transform_not_double_apply(tmp_path, root_offset, expected):
    np = pytest.importorskip("numpy")
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom
    path = tmp_path / "terrain.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    root = UsdGeom.Xform.Define(stage, "/terrain")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(root_offset, 0, 0))
    mesh = UsdGeom.Mesh.Define(stage, "/terrain/mesh")
    mesh.CreatePointsAttr([(0,0,0), (1,0,0), (0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0,1,2])
    stage.GetRootLayer().Save()
    before = path.read_bytes()
    obj = {"root_pos": np.array([[2., 0., 0.]]), "root_quat": np.array([[1.,0.,0.,0.]]), "scale": np.ones(3)}
    result, _ = audit.compare_terrain(path, obj)
    assert result["all_frames_max_m"] == pytest.approx(expected, abs=1e-6)
    assert path.read_bytes() == before


def test_foot_metrics_use_anatomical_indices_and_no_world_offset():
    np = pytest.importorskip("numpy")
    joints = np.zeros((4, 24, 3))
    joints[:, [7, 8], 2] = .1
    joints[:, [10, 11], 0] = .2
    ankle, foot, angle, speed = audit.foot_diagnostics(joints, 30)
    shifted = audit.foot_diagnostics(joints + [10, 20, 30], 30)
    np.testing.assert_allclose(angle, shifted[2])
    np.testing.assert_allclose(speed, 0)
    assert np.all(angle < 0)
    np.testing.assert_allclose(ankle[..., 2], .1)
    np.testing.assert_allclose(foot[..., 2], 0)


def test_foot_report_does_not_claim_contact(tmp_path):
    np = pytest.importorskip("numpy")
    audit.write_foot_diagnostics(tmp_path, {"official_standard": np.zeros((3, 24, 3))},
                                 {"timeline": {"human_fps": 30}})
    report = json.loads((tmp_path / "foot_diagnostics.json").read_text())
    assert report["branches"]["official_standard"]["support_phase"] == "unverified"
    assert len((tmp_path / "official_standard_feet.csv").read_text().splitlines()) == 7
