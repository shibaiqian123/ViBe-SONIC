"""Fixed DEX1 replacement keeps policy DOFs and non-hand dynamics intact."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
SOURCE = Path(__file__).resolve().parents[1] / "src/orcs/assets/dex1.py"
spec_import = importlib.util.spec_from_file_location("dex1", SOURCE)
dex = importlib.util.module_from_spec(spec_import)
spec_import.loader.exec_module(dex)


def fixture(tmp_path):
    mesh = tmp_path / "shape.obj"
    mesh.write_text(
        "v 0 0 0\nv 0.02 0 0\nv 0 0.02 0\nv 0 0 0.02\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    root = ET.Element("robot", name="test")
    spec = mujoco.MjSpec()
    for side in ("left", "right"):
        wrist_name = f"{side}_wrist_yaw_link"
        wrist = spec.worldbody.add_body(name=wrist_name, mass=0.25, inertia=[0.01] * 3)
        wrist.add_joint(
            name=f"{side}_wrist_yaw_joint", type=mujoco.mjtJoint.mjJNT_HINGE
        )
        spec.add_mesh(name=f"{side}_rubber_hand", file=str(mesh))
        wrist.add_geom(
            name=f"{side}_hand_collision",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
        )
        wrist.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{side}_rubber_hand")
        names = [
            wrist_name,
            f"{side}_dex1_base_link",
            f"{side}_dex1_finger_link_1",
            f"{side}_dex1_finger_link_2",
        ]
        for name in names:
            link = ET.SubElement(root, "link", name=name)
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz="0.01 0 0", rpy="0.1 0.2 0.3")
            ET.SubElement(inertial, "mass", value="0.1")
            ET.SubElement(
                inertial,
                "inertia",
                ixx="0.002",
                iyy="0.003",
                izz="0.004",
                ixy="0",
                ixz="0",
                iyz="0",
            )
            for kind in ("visual", "collision"):
                geometry = ET.SubElement(ET.SubElement(link, kind), "geometry")
                ET.SubElement(geometry, "mesh", filename="shape.obj")
        for idx in range(1, 4):
            name = (
                f"{side}_base_joint"
                if idx == 1
                else f"{side}_dex1_finger_joint_{idx - 1}"
            )
            j = ET.SubElement(
                root, "joint", name=name, type="fixed" if idx == 1 else "prismatic"
            )
            ET.SubElement(j, "parent", link=names[0] if idx == 1 else names[1])
            ET.SubElement(j, "child", link=names[idx])
            ET.SubElement(j, "origin", xyz="0.0415 0 0" if idx == 1 else "0 0 0")
            if idx > 1:
                ET.SubElement(j, "limit", lower="-0.02", upper="0.0245")
    path = tmp_path / "robot.urdf"
    ET.ElementTree(root).write(path)
    return spec, path


def test_replacement_compiles_with_same_dofs_and_correct_mass(tmp_path):
    spec, path = fixture(tmp_path)
    original = spec.compile()
    model = dex.replace_with_dex1(spec, path).compile()
    assert model.nq == original.nq == 2
    assert [model.joint(i).name for i in range(model.njnt)] == [
        original.joint(i).name for i in range(original.njnt)
    ]
    assert model.body_mass.sum() == pytest.approx(0.8)
    for side in ("left", "right"):
        assert model.body(f"{side}_dex1_base_link").pos[0] == pytest.approx(0.0415)
        assert model.body(f"{side}_wrist_yaw_link").mass[0] == pytest.approx(0.1)
        assert not any(
            model.geom(i).name == f"{side}_hand_collision" for i in range(model.ngeom)
        )
    np.testing.assert_allclose(
        model.body("left_dex1_base_link").inertia, [0.002, 0.003, 0.004]
    )


def test_missing_mesh_fails(tmp_path):
    spec, path = fixture(tmp_path)
    (tmp_path / "shape.obj").unlink()
    with pytest.raises(FileNotFoundError):
        dex.replace_with_dex1(spec, path)


def test_finger_zero_outside_limit_fails(tmp_path):
    spec, path = fixture(tmp_path)
    root = ET.parse(path)
    for limit in root.findall(".//limit"):
        limit.set("lower", "0.01")
    root.write(path)
    with pytest.raises(ValueError, match="fixed-zero"):
        dex.replace_with_dex1(spec, path)
