"""Opt-in fixed-pose DEX1 hands from an external URDF; no extra policy DOFs."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def _vector(element, key, default):
    return np.asarray([float(v) for v in element.get(key, default).split()])


def _pose(element):
    if element is None:
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    r, p, y = _vector(element, "rpy", "0 0 0") / 2
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return _vector(element, "xyz", "0 0 0"), np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def _inertia(body, link):
    inertial = link.find("inertial")
    if inertial is None:
        raise ValueError(f"missing inertia for {link.get('name')}")
    body.mass = float(inertial.find("mass").get("value"))
    body.ipos, quat = _pose(inertial.find("origin"))
    values = inertial.find("inertia").attrib
    xx, yy, zz, xy, xz, yz = [
        float(values[k]) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    ]
    matrix = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    matrix = rotation @ matrix @ rotation.T
    moments, axes = np.linalg.eigh(matrix)
    if body.mass <= 0 or not np.isfinite(matrix).all() or np.any(moments <= 0):
        raise ValueError(f"invalid inertia for {link.get('name')}")
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    principal_quat = np.empty(4)
    mujoco.mju_mat2Quat(principal_quat, axes.reshape(-1))
    body.inertia = moments
    body.iquat = principal_quat
    body.explicitinertial = True


def replace_with_dex1(spec, urdf_path):
    """Replace only the distal hands; wrist-yaw bare inertia avoids double mass.

    The 4 prismatic joints are welded at q=0. Shoulder/elbow/wrist joints,
    actuators, sites and the rest of the robot keep their original definitions.
    """
    path = Path(urdf_path).expanduser().resolve(strict=True)
    root = ET.parse(path).getroot()
    links = {x.get("name"): x for x in root.findall("link")}
    joints = {x.get("name"): x for x in root.findall("joint")}

    def geometry(body, link, kind):
        for index, item in enumerate(link.findall(kind)):
            pos, quat = _pose(item.find("origin"))
            geom = item.find("geometry")
            mesh = geom.find("mesh")
            if mesh is None:
                raise ValueError(f"DEX1 expects mesh {kind}: {link.get('name')}")
            filename = mesh.get("filename")
            mesh_path = (path.parent / filename).resolve(strict=True)
            name = f"orcs_dex1_{link.get('name')}_{kind}_{index}"
            spec.add_mesh(
                name=name, file=str(mesh_path), scale=_vector(mesh, "scale", "1 1 1")
            )
            visual = kind == "visual"
            body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=name,
                pos=pos,
                quat=quat,
                group=2 if visual else 3,
                contype=0 if visual else 1,
                conaffinity=0 if visual else 1,
                density=0,
                rgba=[0.41, 0.41, 0.41, 1],
            )

    def find_body(body, name):
        if body.name == name:
            return body
        for child in body.bodies:
            found = find_body(child, name)
            if found is not None:
                return found
        return None

    for side in ("left", "right"):
        # Name-index caches can be stale after adding bodies to a compiled spec.
        wrist = find_body(spec.worldbody, f"{side}_wrist_yaw_link")
        if wrist is None:
            raise ValueError(f"missing {side} wrist body")
        base_joint = joints[f"{side}_base_joint"]
        if (
            base_joint.get("type") != "fixed"
            or base_joint.find("parent").get("link") != wrist.name
        ):
            raise ValueError(
                f"unsupported {side} DEX1 mount: type={base_joint.get('type')} "
                f"parent={base_joint.find('parent').get('link')} body={wrist.name}"
            )
        # The original G1 combines a rubber hand into wrist-yaw inertia.
        # Use the URDF's bare wrist-yaw inertia before adding explicit DEX1 mass.
        _inertia(wrist, links[wrist.name])
        removed = 0
        for geom in list(wrist.geoms):
            if (
                geom.name == f"{side}_hand_collision"
                or geom.meshname == f"{side}_rubber_hand"
            ):
                spec.delete(geom)
                removed += 1
        if removed != 2:
            raise ValueError(
                f"expected original visual/collision hand pair on {side}, got {removed}"
            )
        base_name = base_joint.find("child").get("link")
        pos, quat = _pose(base_joint.find("origin"))
        base = wrist.add_body(name=base_name, pos=pos, quat=quat)
        _inertia(base, links[base_name])
        for kind in ("visual", "collision"):
            geometry(base, links[base_name], kind)
        for finger in (1, 2):
            joint = joints[f"{side}_dex1_finger_joint_{finger}"]
            limit = joint.find("limit")
            if (
                joint.get("type") != "prismatic"
                or joint.find("parent").get("link") != base_name
                or limit is None
                or not float(limit.get("lower")) <= 0 <= float(limit.get("upper"))
            ):
                raise ValueError(
                    f"unsupported fixed-zero DEX1 finger: {joint.get('name')}"
                )
            name = joint.find("child").get("link")
            pos, quat = _pose(joint.find("origin"))
            body = base.add_body(name=name, pos=pos, quat=quat)
            _inertia(body, links[name])
            for kind in ("visual", "collision"):
                geometry(body, links[name], kind)
    return spec
