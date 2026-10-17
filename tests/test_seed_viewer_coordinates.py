"""Seed local coordinates gain the scene origin once, only during display."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'src/orcs/cli/view_seed_motion.py'
tree = ast.parse(SOURCE.read_text())
nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'_set_qpos', '_force_arrows'}]
namespace = {'np': np}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(SOURCE), 'exec'), namespace)


class Indices:
    def __init__(self, values):
        self.values = np.array(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def seed():
    return SimpleNamespace(
        robot_root_pos_w=np.array([[1., 2., 3.]]),
        robot_root_quat_w=np.array([[1., 0., 0., 0.]]),
        joint_pos=np.array([[.4, .5]]),
        body_pos_w=np.array([[[1., 2., 3.], [2., 3., 4.]]]),
        assist_force_w=np.array([[[0., 0., 2.], [1., 0., 0.]]]),
        object_pos_w=np.array([[4., 5., 6.]]),
        object_quat_w=np.array([[0., 1., 0., 0.]]),
        object_assist_force_w=np.array([[0., 2., 0.]]))


@pytest.mark.parametrize('origin', [np.zeros(3), np.array([8., -4., 1.5])])
def test_robot_object_and_force_arrows_share_origin_without_accumulation(origin):
    s = seed()
    original = {k: v.copy() for k, v in vars(s).items()}
    robot = SimpleNamespace(indexing=SimpleNamespace(
        free_joint_q_adr=Indices(range(7)), joint_q_adr=Indices([7, 8])))
    obj = SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=Indices(range(9, 16))))
    data = SimpleNamespace(qpos=np.zeros(16))
    for _ in range(2):
        namespace['_set_qpos'](data, robot, s, 0, obj, origin=origin)
        np.testing.assert_allclose(data.qpos[:3], s.robot_root_pos_w[0] + origin)
        np.testing.assert_allclose(data.qpos[9:12], s.object_pos_w[0] + origin)
        np.testing.assert_array_equal(data.qpos[3:7], s.robot_root_quat_w[0])
        np.testing.assert_array_equal(data.qpos[7:9], s.joint_pos[0])
        np.testing.assert_array_equal(data.qpos[12:16], s.object_quat_w[0])
        points, _ = namespace['_force_arrows'](s, 0, .2, origin=origin)
        starts = np.concatenate([s.body_pos_w[0], s.object_pos_w])
        forces = np.concatenate([s.assist_force_w[0], s.object_assist_force_w])
        np.testing.assert_allclose(points[:, 0], starts + origin)
        np.testing.assert_allclose(points[:, 1] - points[:, 0], .2 * forces, atol=1e-6)
    for k, v in original.items():
        np.testing.assert_array_equal(getattr(s, k), v)


def test_robot_only_force_overlay():
    s = seed()
    s.object_pos_w = None
    s.object_assist_force_w = None
    origin = np.array([2., 3., 4.])
    points, colors = namespace['_force_arrows'](s, 0, .1, origin=origin)
    assert points.shape == (2, 2, 3)
    assert colors.shape == (2, 3)
    np.testing.assert_allclose(points[:, 0], s.body_pos_w[0] + origin)
